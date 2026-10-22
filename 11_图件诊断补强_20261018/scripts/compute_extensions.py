"""Frozen-prediction diagnostics and exact held-out TreeSHAP interactions."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['NUMBA_NUM_THREADS']='2'
import json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
R08=ROOT/'08_形状水平解耦与施肥约束_20261018'
RES=OUT/'results';TAB=OUT/'tables'
MODELS=['OriginalLightGBM','TabPFN_DailyRaw','AffineReconciled']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(x,p):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def save(x,p):x.to_csv(p,index=False,encoding='utf-8-sig')

def diagnostics():
    d=pd.read_csv(R08/'results/all_daily_predictions.csv',float_precision='round_trip')
    d=d.loc[d.model.isin(MODELS)&d.y.notna()].copy()
    obs=d.loc[d.model.eq(MODELS[0])].sort_values('row_id')
    qs=[0,.5,.75,.9,.95,1];edges=np.quantile(obs.y,qs)
    rng=np.random.default_rng(20261018);draws=rng.choice(np.arange(2014,2022),size=(2000,8),replace=True)
    weights=np.array([np.bincount(v-2014,minlength=8) for v in draws])
    records=[];conditional=[];biasparts=[]
    for model,g in d.groupby('model',sort=False):
        g=g.sort_values('row_id');assert np.array_equal(g.row_id,obs.row_id)
        assert np.array_equal(g.y,obs.y)
        y=g.y.to_numpy();p=g.prediction.to_numpy();e=p-y
        sy=y.std(ddof=0);sp=p.std(ddof=0);r=np.corrcoef(y,p)[0,1]
        crmse=np.sqrt(np.mean(((p-p.mean())-(y-y.mean()))**2))
        assert np.isclose((crmse/sy)**2,1+(sp/sy)**2-2*r*(sp/sy))
        records.append(dict(model=model,n=len(g),n_cycles=g.season.nunique(),observed_sd=sy,predicted_sd=sp,
            sd_ratio=sp/sy,correlation=r,centered_rmse=crmse,normalized_centered_rmse=crmse/sy,
            R2=1-np.sum(e*e)/np.sum((y-y.mean())**2),RMSE=np.sqrt(np.mean(e*e)),bias=e.mean()))
        g['bin']=np.clip(np.searchsorted(edges,g.y,side='right')-1,0,len(edges)-2);g['error']=e
        for bi,z in g.groupby('bin'):
            sums=z.groupby('season').error.sum().reindex(range(2014,2022),fill_value=0).to_numpy()
            ns=z.groupby('season').size().reindex(range(2014,2022),fill_value=0).to_numpy()
            den=weights@ns;vals=np.divide(weights@sums,den,out=np.full(2000,np.nan),where=den>0)
            lo,hi=np.nanquantile(vals,[.025,.975])
            conditional.append(dict(model=model,bin=int(bi),quantile_low=qs[bi],quantile_high=qs[bi+1],
                flux_low=edges[bi],flux_high=edges[bi+1],n=len(z),n_cycles=z.season.nunique(),
                mean_observed=z.y.mean(),bias=z.error.mean(),q025=lo,q975=hi,valid_bootstraps=int(np.isfinite(vals).sum())))
        for band,mask in [('top_5pct',g.y.ge(edges[-2])),('lower_95pct',g.y.lt(edges[-2]))]:
            z=g.loc[mask];biasparts.append(dict(model=model,band=band,n=len(z),mean_observed=z.y.mean(),bias=z.error.mean()))
    save(pd.DataFrame(records),TAB/'daily_dispersion_metrics.csv')
    save(pd.DataFrame(conditional),TAB/'conditional_bias.csv')
    save(pd.DataFrame(biasparts),TAB/'tail_bias.csv')
    dump(dict(prediction_source=str(R08/'results/all_daily_predictions.csv'),sha256=sha(R08/'results/all_daily_predictions.csv'),
        quantile_edges=edges.tolist(),quantiles=qs,bootstrap_unit='management cycle, both treatments retained together',
        bootstrap_replicates=2000,seed=20261018,thresholds='fixed from pooled held-out observations; descriptive conditioning',
        selection='all 3319 observed treatment-days; identical support for all three models'),RES/'diagnostics_protocol.json')
    print(pd.DataFrame(records)[['model','sd_ratio','correlation','R2']].to_string(index=False),flush=True)
    print(pd.DataFrame(biasparts).to_string(index=False),flush=True)

def interactions():
    if (RES/'interaction_completion.json').exists():return
    features=json.loads((R08/'results/frozen_run.json').read_text(encoding='utf-8'))['features']
    old=json.loads((RES/'shap_completion.json').read_text(encoding='utf-8'))
    checks={v['season']:v for v in old['checks']}
    sources=[R08/'results/frozen_run.json',OUT/'shap_protocol.md',Path(__file__)]+list((OUT/'models').glob('*_LightGBM.txt'))
    dump(dict(started=time.strftime('%Y-%m-%dT%H:%M:%S'),shap=shap.__version__,lightgbm=lgb.__version__,
        features=features,source_hashes={str(p):sha(p) for p in sources},method='TreeSHAP tree_path_dependent, raw model output; no training',
        normalization='divide each fold by SD of its training-set raw model output',
        pair='2 * interaction[Imin_sum60, W]',pooling='equal fold weights for mean absolute component matrix'),RES/'interaction_frozen.json')
    rows=[];matrix=[];verify=[];saved=RES/'interaction_arrays';saved.mkdir(exist_ok=True)
    for held in range(2014,2022):
        started=time.time();key=[s for s in range(2014,2022) if s!=held]
        f=R08/'fold_data'/('_'.join(map(str,key))+'.csv');d=pd.read_csv(f,float_precision='round_trip')
        te=d.season.eq(held)&d.y.notna();tr=d.season.ne(held)&d.y.notna()
        x=d.loc[te,features].to_numpy();xt=d.loc[tr,features].to_numpy()
        model=lgb.Booster(model_str=(OUT/'models'/f'{held}_LightGBM.txt').read_text(encoding='utf-8'),params={'num_threads':2})
        raw=model.predict(x,raw_score=True,num_threads=2)
        contrib=model.predict(x,pred_contrib=True,num_threads=2)
        sd=float(np.std(model.predict(xt,raw_score=True,num_threads=2)))
        assert np.isclose(sd,checks[held]['training_margin_sd'])
        explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw')
        inter=np.asarray(explainer.shap_interaction_values(x))
        assert inter.shape==(len(x),len(features),len(features))
        err=float(np.max(abs(inter.sum(axis=2)-contrib[:,:-1])))
        sym=float(np.max(abs(inter-inter.transpose(0,2,1))))
        total=float(np.max(abs(inter.sum(axis=(1,2))+contrib[:,-1]-raw)))
        assert max(err,sym,total)<1e-7,(held,err,sym,total)
        norm=inter/sd;meanabs=np.mean(abs(norm),axis=0)
        for i,feature in enumerate(features):
            for j,other in enumerate(features):
                matrix.append(dict(season=held,feature=feature,other_feature=other,
                    mean_abs_component=meanabs[i,j],display_magnitude=meanabs[i,j]*(1 if i==j else 2),
                    target_transform=checks[held]['target_transform']))
        i=features.index('Imin_sum60');j=features.index('W')
        part=d.loc[te,['row_id','season','treatment','Date','y','Imin_sum60','W','T']].copy()
        part['pair_interaction']=2*norm[:,i,j];part['target_transform']=checks[held]['target_transform']
        tw=np.sort(xt[:,j]);part['WFPS_training_percentile']=np.searchsorted(tw,x[:,j],side='right')/len(tw)
        part['WFPS_group']=pd.cut(part.WFPS_training_percentile,[-.001,1/3,2/3,1.001],labels=['Low','Middle','High']).astype(str)
        rows.append(part)
        np.savez_compressed(saved/f'{held}.npz',row_id=part.row_id.to_numpy(),interactions=inter.astype('float32'),features=np.array(features))
        verify.append(dict(season=held,n=len(x),input_sha256=sha(f),training_margin_sd=sd,
            row_sum_max_error=err,symmetry_max_error=sym,additivity_max_error=total,seconds=time.time()-started))
        print('SHAP interactions',held,len(x),'rows;',round(time.time()-started,1),'s; check',max(err,sym,total),flush=True)
    save(pd.concat(rows,ignore_index=True),RES/'nitrogen_water_interactions.csv')
    save(pd.DataFrame(matrix),RES/'interaction_matrix_by_fold.csv')
    dump(dict(passed=True,folds=8,n=sum(x['n'] for x in verify),checks=verify),RES/'interaction_completion.json')

if __name__=='__main__':diagnostics();interactions()
