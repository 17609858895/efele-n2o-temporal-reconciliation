"""Actual held-out two-head interpretation; never refit/select the head coefficients."""
from pathlib import Path
import os, sys, json, hashlib, time, gc, warnings
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from tabpfn import TabPFNRegressor
from importlib.metadata import version

OUT=Path(__file__).resolve().parents[1]; ROOT=OUT.parent
R08=ROOT/'08_形状水平解耦与施肥约束_20261018'; R07=ROOT/'07_周期损失与TabPFN_20261018'
RES=OUT/'results'; YEARS=range(2014,2022); SEED=20261018
CKPT=Path(os.environ.get('TABPFN_CHECKPOINT', str(Path.home()/'.cache/tabpfn/tabpfn-v2-regressor.ckpt'))).expanduser()
FEATURES=json.loads((R08/'results/frozen_run.json').read_text(encoding='utf-8'))['features']
GROUPS={
 'Temperature':['Ta_min','Ta_max','temp_3','temp_7','temp_30','T'],
 'Water':['Rainfall_sum']+[f'rain_{n}' for n in [3,7,14,30,60,90]]+['W'],
 'Atmosphere':['Rh_min','Rh_max','WS_mean','Rg_sum'],
 'Calendar':['doy_sin','doy_cos'],
 'Crop':['crop_WHEAT','crop_MAIZE','crop_MUSTARD','crop_age','tillage_14'],
 'Fertilizer':['is_PS','DSFA','last_Ndose','Imin','Iorg']+[f'I{f}_{k}{n}' for f in ['min','org'] for k in ['sum','exp'] for n in [7,30,60]]}
assert sorted(sum(GROUPS.values(),[]))==sorted(FEATURES)
torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore',message='X does not have valid feature names')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(obj,p): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def save(d,p): d.to_csv(p,index=False,encoding='utf-8-sig')
def read(p): return pd.read_csv(p,float_precision='round_trip')
def means(d,p):
    ans=np.zeros(len(d))
    for _,ids in d.groupby('treatment',sort=False).groups.items():
        ids=np.asarray(ids); obs=ids[d.loc[ids,'y'].notna().to_numpy()];ans[ids]=np.mean(p[obs])
    return ans
def components(d,pl,pt,head):
    a=head['sl']*head['beta_l'][0];b=1+head['beta_l'][1]
    return a+b*means(d,pl),pt-means(d,pt)
def metrics(d,lev,shape):
    obs=d.y.notna().to_numpy();y=d.y.to_numpy();yc=means(d,np.nan_to_num(y))
    units=[ids[0] for ids in d.groupby('treatment',sort=False).groups.values()]
    return dict(level_mse=float(np.mean((yc[units]-lev[units])**2)),
        shape_mse=float(np.mean(((y-yc)[obs]-shape[obs])**2)),
        daily_mse=float(np.mean((y[obs]-lev[obs]-shape[obs])**2)),n_obs=int(obs.sum()))
def query(model,X):
    return np.concatenate([model.predict(X[j:j+256].astype('float32')) for j in range(0,len(X),256)])

def main():
    if (RES/'interpretation_completion.json').exists(): raise RuntimeError('Completed analysis is immutable.')
    RES.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    sources=[Path(__file__),OUT/'interpretation_protocol.md',CKPT,R08/'results/all_daily_predictions.csv',R08/'results/lgb_selection_log.json']
    sources+=list((R08/'models').glob('*_AffineReconciled.json'))
    sources+=list((R07/'fold_data').glob('outer_*_daily.csv'))
    sources+=list(p for p in (R08/'fold_data').glob('*.csv') if len(p.stem.split('_'))==7)
    dump(dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),feature_groups=GROUPS,
        shifts=[1/3,2/3],temperature_increment=1,wfps_increment=5,seed=SEED,
        source_hashes={str(p):sha(p) for p in sources},
        versions={k:version(k) for k in ['tabpfn','torch','lightgbm','numpy','pandas']}),RES/'interpretation_frozen.json')
    ref=read(R08/'results/all_daily_predictions.csv')
    selection=json.loads((R08/'results/lgb_selection_log.json').read_text(encoding='utf-8'))
    records=[];repro=[];responses=[];preds=[];start=time.perf_counter()
    for held in YEARS:
        foldstart=time.perf_counter();key=[v for v in YEARS if v!=held]
        # Separate original input reads preserve each algorithm's floating-point pipeline.
        dl=read(R08/'fold_data'/('_'.join(map(str,key))+'.csv'))
        dt=pd.read_csv(R07/'fold_data'/f'outer_{held}_daily.csv')
        assert dl.row_id.tolist()==dt.row_id.tolist()
        cfg=next(s['config'] for s in selection if s['train_cycles']==key)
        tr=dl.season.ne(held)&dl.y.notna();te=dl.season.eq(held)
        x=dl[FEATURES].to_numpy();y=dl.loc[tr,'y'].to_numpy()
        scale=max(1.,float(np.median(np.abs(y))))
        target=np.arcsinh(y/scale) if cfg['transform']=='asinh' else y
        w=1/dl.loc[tr].groupby(['season','treatment']).y.transform('count').to_numpy();w/=w.mean()
        lm=lgb.LGBMRegressor(n_estimators=cfg['trees'],num_leaves=cfg['leaves'],learning_rate=.04,
            min_child_samples=25,reg_lambda=10,verbosity=-1,n_jobs=2,random_state=SEED,deterministic=True,force_col_wise=True)
        lm.fit(x[tr],target,sample_weight=w)
        def lpred(q):
            p=lm.predict(q);return scale*np.sinh(np.clip(p,-10,10)) if cfg['transform']=='asinh' else p
        xl=x[te].copy();xt=dt.loc[te,FEATURES].to_numpy(dtype='float32')
        d=dl.loc[te].reset_index(drop=True)
        pl=lpred(xl);lerr=float(np.max(np.abs(pl-d.p_lgb)))
        assert lerr<1e-9,(held,lerr)
        (OUT/'models'/f'{held}_LightGBM.txt').write_text(lm.booster_.model_to_string(),encoding='utf-8')
        head=json.loads((R08/'models'/f'{held}_AffineReconciled.json').read_text(encoding='utf-8'))
        tpref=ref.loc[ref.model.eq('TabPFN_DailyRaw')&ref.season.eq(held)].set_index('row_id').loc[d.row_id,'prediction'].to_numpy()
        for mode in ['fit_preprocessors','low_memory']:
            tm=TabPFNRegressor(model_path=str(CKPT),device='cuda',n_estimators=4,random_state=SEED,
                n_jobs=1,fit_mode=mode,memory_saving_mode=True)
            tm.fit(dt.loc[tr,FEATURES].to_numpy(dtype='float32'),dt.loc[tr,'y'].to_numpy(dtype='float32'))
            pt=query(tm,xt)
            ob=d.y.notna().to_numpy();sst=np.sum((d.y[ob]-d.y[ob].mean())**2)
            r2delta=float(abs(np.sum((pt[ob]-d.y[ob])**2)-np.sum((tpref[ob]-d.y[ob])**2))/sst)
            if np.allclose(pt,tpref,atol=3e-4,rtol=1e-5) and r2delta<1e-5: break
            del tm;gc.collect();torch.cuda.empty_cache()
        else: raise RuntimeError(f'TabPFN reproduction failed: {held}')
        baselev,baseshape=components(d,pl,pt,head);basem=metrics(d,baselev,baseshape)
        afref=ref.loc[ref.model.eq('AffineReconciled')&ref.season.eq(held)].set_index('row_id').loc[d.row_id,'prediction'].to_numpy()
        assert np.allclose(baselev+baseshape,afref,atol=6e-4,rtol=2e-5)
        repro.append(dict(held=held,lightgbm_max_error=lerr,tabpfn_max_error=float(np.max(abs(pt-tpref))),tabpfn_r2_difference=r2delta,mode=mode))
        dump(repro,RES/'reproduction.json')
        records.append(dict(season=held,group='Baseline',shift=0,**basem))
        baseline=d[['row_id','Date','season','treatment','y']].copy()
        baseline['variant']='Baseline';baseline['p_lgb']=pl;baseline['p_tab']=pt;baseline['level']=baselev;baseline['shape']=baseshape;preds.append(baseline)
        for group,names in GROUPS.items():
            cols=[FEATURES.index(n) for n in names]
            for shift in [1/3,2/3]:
                ql=xl.copy();qt=xt.copy()
                for _,ids in d.groupby('treatment',sort=False).groups.items():
                    ids=np.asarray(ids);donor=np.roll(ids,int(np.floor(len(ids)*shift)))
                    ql[np.ix_(ids,cols)]=xl[np.ix_(donor,cols)];qt[np.ix_(ids,cols)]=xt[np.ix_(donor,cols)]
                lp=lpred(ql);tp=query(tm,qt);lev,shape=components(d,lp,tp,head)
                records.append(dict(season=held,group=group,shift=shift,**metrics(d,lev,shape)))
                row=baseline.iloc[:,:5].copy();row['variant']=group+f'_{shift:.6f}';row['p_lgb']=lp;row['p_tab']=tp;row['level']=lev;row['shape']=shape;preds.append(row)
            print(f'{held}: temporal reliance {group} done; {time.perf_counter()-foldstart:.1f}s',flush=True)
        local={}
        for kind,changes in [('T',{'T':1.}),('W',{'W':5.}),('TW',{'T':1.,'W':5.})]:
            ql=xl.copy();qt=xt.copy();actual={}
            for name,inc in changes.items():
                col=FEATURES.index(name);lo,hi=dl.loc[tr,name].min(),dl.loc[tr,name].max()
                # Bound only the perturbation; preserve held-out values already outside training support.
                change=np.where((xl[:,col]>=lo)&(xl[:,col]<=hi),np.minimum(inc,hi-xl[:,col]),0.)
                ql[:,col]+=change;qt[:,col]+=change.astype('float32');actual[name]=change
            lp=lpred(ql);tp=query(tm,qt);lev,shape=components(d,lp,tp,head)
            local[kind]=(lp,tp,lev,shape)
            row=d[['row_id','Date','season','treatment','y','T','W']].copy();row['perturbation']=kind
            row['actual_T_change']=actual.get('T',np.zeros(len(d)));row['actual_W_change']=actual.get('W',np.zeros(len(d)))
            row['level_row_response']=(1+head['beta_l'][1])*(lp-pl)
            row['level_mean_response']=lev-baselev;row['shape_response']=shape-baseshape
            row['total_response']=lev+shape-baselev-baseshape;responses.append(row)
            for _,ids in d.groupby('treatment').groups.items():
                obs=np.asarray(ids)[d.loc[ids,'y'].notna().to_numpy()]
                assert abs(row.loc[obs,'level_row_response'].mean()-row.loc[obs,'level_mean_response'].iloc[0])<1e-10
                assert abs(row.loc[obs,'shape_response'].mean())<1e-5
        save(pd.DataFrame(records),RES/'temporal_reliance_raw.csv')
        save(pd.concat(responses,ignore_index=True),RES/'local_responses.csv')
        save(pd.concat(preds,ignore_index=True),RES/'perturbation_predictions.csv')
        print(f'Finished fold {held}: {time.perf_counter()-foldstart:.1f}s; total {time.perf_counter()-start:.1f}s',flush=True)
        del tm,lm;gc.collect();torch.cuda.empty_cache()
    dump(dict(status='complete',folds=8,groups=GROUPS,n_baseline_days=5970,n_observed_days=3319,
        seconds=time.perf_counter()-start,variants_per_fold=16,reproduction=repro),RES/'interpretation_completion.json')

if __name__=='__main__': main()
