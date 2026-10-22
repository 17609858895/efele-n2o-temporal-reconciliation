"""Exact held-out LightGBM TreeSHAP in each fitted booster's own output units."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import generate_paper_figures as b
RES=b.OUT/'results'

def main():
    if (RES/'shap_completion.json').exists(): return
    selected=json.loads((b.R08/'results/lgb_selection_log.json').read_text(encoding='utf-8'))
    features=json.loads((b.R08/'results/frozen_run.json').read_text(encoding='utf-8'))['features']
    groups=json.loads((RES/'interpretation_frozen.json').read_text(encoding='utf-8'))['feature_groups']
    group={f:g for g,ff in groups.items() for f in ff}
    modelpaths=list((b.OUT/'models').glob('*_LightGBM.txt'));assert len(modelpaths)==8
    sources=[Path(__file__),b.OUT/'shap_protocol.md',b.R08/'results/lgb_selection_log.json']+modelpaths
    b.dump(dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),lightgbm=lgb.__version__,
        source_hashes={str(p):b.sha(p) for p in sources}),RES/'shap_frozen.json')
    rows=[];summary=[];checks=[]
    for held in range(2014,2022):
        key=[y for y in range(2014,2022) if y!=held]
        cfg=next(x['config'] for x in selected if x['train_cycles']==key)
        assert cfg['transform'] in ['asinh','raw']
        path=b.R08/'fold_data'/('_'.join(map(str,key))+'.csv');d=b.read(path)
        tr=d.season.ne(held)&d.y.notna();te=d.season.eq(held)&d.y.notna()
        train=d.loc[tr,features].to_numpy();q=d.loc[te,features].to_numpy()
        model=lgb.Booster(model_str=(b.OUT/'models'/f'{held}_LightGBM.txt').read_text(encoding='utf-8'))
        raw=model.predict(q,raw_score=True,num_threads=2);con=model.predict(q,pred_contrib=True,num_threads=2)
        err=float(np.max(np.abs(con.sum(axis=1)-raw)));assert err<1e-10
        denom=float(np.std(model.predict(train,raw_score=True,num_threads=2)))
        assert denom>0
        norm=con[:,:-1]/denom;imp=np.mean(abs(norm),axis=0);imp=100*imp/imp.sum()
        checks.append(dict(season=held,target_transform=cfg['transform'],additivity_max_error=err,training_margin_sd=denom,n=len(q),input_sha=b.sha(path)))
        for j,feature in enumerate(features):
            vals=np.sort(train[:,j]);pct=np.searchsorted(vals,q[:,j],side='right')/len(vals)
            rr=d.loc[te,['row_id','season','treatment','Date']].copy();rr['feature']=feature;rr['group']=group[feature]
            rr['value']=q[:,j];rr['training_percentile']=pct;rr['shap_margin']=con[:,j]
            rr['target_transform']=cfg['transform']
            rr['shap_normalized']=norm[:,j];rr['expected_margin']=con[:,-1];rows.append(rr)
            summary.append(dict(season=held,feature=feature,group=group[feature],target_transform=cfg['transform'],share_pct=imp[j]))
    b.save(pd.concat(rows,ignore_index=True),RES/'tree_shap_heldout.csv')
    b.save(pd.DataFrame(summary),RES/'tree_shap_feature_shares.csv')
    b.dump(dict(passed=True,folds=8,features=42,observations=3319,checks=checks),RES/'shap_completion.json')

if __name__=='__main__':main()
