"""Offline TabPFN v2, fixed cycle/direct and daily/raw/asinh comparisons."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from importlib.metadata import version

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)
OUT = Path(__file__).resolve().parents[1]
RES = OUT/'results'
FOLD = OUT/'fold_data'
CHECKPOINT = Path(os.environ.get('TABPFN_CHECKPOINT', str(Path.home()/'.cache/tabpfn/tabpfn-v2-regressor.ckpt'))).expanduser()
SEED = 20261018
YEARS = tuple(range(2014,2022))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(obj,path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def save(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')

def estimator(n, device):
    return TabPFNRegressor(model_path=str(CHECKPOINT),device=device,n_estimators=n,
                          random_state=SEED,n_jobs=1,fit_mode='low_memory',
                          memory_saving_mode=True)

def smoke():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = dict(purpose='synthetic technical check, no EFELE data or score',
                  torch=torch.__version__,tabpfn=version('tabpfn'),python=platform.python_version(),
                  python_executable=sys.executable, device=device, cuda_available=torch.cuda.is_available(),
                  gpu_name=torch.cuda.get_device_name(0) if device=='cuda' else None,
                  checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT))
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(128,8)).astype('float32')
    y = (X[:,0] + X[:,1]**2).astype('float32')
    model = estimator(1,device)
    start = time.perf_counter()
    try:
        model.fit(X[:120],y[:120])
        p = model.predict(X[120:])
        result.update(seconds=time.perf_counter()-start,finite=bool(np.isfinite(p).all()),
                      prediction_shape=list(p.shape),status='passed')
        assert result['finite'] and len(p)==8
    except Exception as exc:
        result.update(status='failed',exception=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        dump(result,RES/'tabpfn_smoke.json')
        print(json.dumps(result,ensure_ascii=False),flush=True)

def run():
    smoke_info=json.loads((RES/'tabpfn_smoke.json').read_text(encoding='utf-8'))
    assert smoke_info['status']=='passed'
    device=smoke_info['device']
    frozen=json.loads((RES/'joint_frozen_run.json').read_text(encoding='utf-8'))
    feature_names=frozen['features']
    cycle_features=frozen['cycle_features']
    if (RES/'tabpfn_cycle_predictions.csv').exists():
        raise RuntimeError('Finished result exists. Do not overwrite.')
    assert all((FOLD/f'outer_{held}_daily.csv').exists() for held in YEARS)
    sources=[Path(__file__),OUT/'protocol.md',CHECKPOINT]
    sources+=list(FOLD.glob('*.csv'))
    dump(dict(source_hashes={str(p):sha(p) for p in sources},smoke=smoke_info,
              cycle_estimators=8,daily_estimators=4,seed=SEED,batch_size=256,
              inference_mode='low_memory',memory_saving=True,
              versions={k:version(k) for k in ['tabpfn','torch','scikit-learn','numpy','pandas']}),
         RES/'tabpfn_frozen_run.json')
    logs,cycle_rows,daily_rows=[],[],[]
    start_all=time.perf_counter()
    for held in YEARS:
        d=pd.read_csv(FOLD/f'outer_{held}_daily.csv')
        c=pd.read_csv(FOLD/f'outer_{held}_cycle.csv')
        for name in ['TabPFN_Cycle','TabPFN_DailyRaw','TabPFN_DailyAsinh']:
            tic=time.perf_counter()
            row=dict(outer_test=held,model=name,device=device,
                     train_cycles=[y for y in YEARS if y!=held])
            try:
                if name=='TabPFN_Cycle':
                    train=c.season.ne(held)
                    test=c.season.eq(held)
                    X=c.loc[train,cycle_features].to_numpy(dtype='float32')
                    y=c.loc[train,'y'].to_numpy(dtype='float32')
                    Q=c.loc[test,cycle_features].to_numpy(dtype='float32')
                    model=estimator(8,device)
                    row.update(train_n=len(X),features=X.shape[1],n_estimators=8)
                    model.fit(X,y)
                    p=model.predict(Q)
                    cr=c.loc[test,['season','treatment','y','n_obs']].copy()
                    cr['prediction'],cr['model']=p,name
                    cycle_rows.append(cr)
                else:
                    train=d.season.ne(held)&d.y.notna()
                    test=d.season.eq(held)
                    X=d.loc[train,feature_names].to_numpy(dtype='float32')
                    y=d.loc[train,'y'].to_numpy(dtype='float32')
                    Q=d.loc[test,feature_names].to_numpy(dtype='float32')
                    scale=max(1.,float(np.median(np.abs(y))))
                    target=np.arcsinh(y/scale) if name.endswith('Asinh') else y
                    model=estimator(4,device)
                    row.update(train_n=len(X),features=X.shape[1],n_estimators=4,scale=scale)
                    model.fit(X,target)
                    # Only memory batch size changes on an OOM; no labels or rows removed.
                    parts=[]
                    batch_size=256
                    for begin in range(0,len(Q),batch_size):
                        query=Q[begin:begin+batch_size]
                        try:
                            part=model.predict(query)
                        except torch.cuda.OutOfMemoryError:
                            gc.collect(); torch.cuda.empty_cache()
                            row.setdefault('oom_retries',[]).append(dict(begin=begin,batch=64))
                            part=np.concatenate([model.predict(query[j:j+64]) for j in range(0,len(query),64)])
                        parts.append(part)
                        print(f'{held} {name}: predicted {min(begin+batch_size,len(Q))}/{len(Q)} days',flush=True)
                    p=np.concatenate(parts)
                    if name.endswith('Asinh'):
                        p=scale*np.sinh(p)
                    dr=d.loc[test,['row_id','Date','season','treatment','y']].copy()
                    dr['prediction'],dr['model']=p,name
                    daily_rows.append(dr)
                    cr=dr.loc[dr.y.notna()].groupby(['season','treatment'],as_index=False).agg(
                        y=('y','mean'),prediction=('prediction','mean'),n_obs=('y','size'))
                    cr['model']=name
                    cycle_rows.append(cr)
                assert np.isfinite(p).all()
                row.update(status='success',seconds=time.perf_counter()-tic,
                           gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
                del model
                gc.collect()
                if device=='cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            except Exception as exc:
                row.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                logs.append(row); dump(logs,RES/'tabpfn_fit_log.json')
                raise
            logs.append(row)
            dump(logs,RES/'tabpfn_fit_log.json')
            save(pd.concat(cycle_rows,ignore_index=True),RES/'tabpfn_cycle_predictions_partial.csv')
            if daily_rows: save(pd.concat(daily_rows,ignore_index=True),RES/'tabpfn_daily_predictions_partial.csv')
            print(f'{held} {name} complete in {row["seconds"]:.1f}s; total {time.perf_counter()-start_all:.1f}s',flush=True)
    save(pd.concat(cycle_rows,ignore_index=True),RES/'tabpfn_cycle_predictions.csv')
    save(pd.concat(daily_rows,ignore_index=True),RES/'tabpfn_daily_predictions.csv')
    dump(dict(status='completed',fits=len(logs),wall_seconds=time.perf_counter()-start_all,
              all_folds=sorted(set(r['outer_test'] for r in logs))),RES/'tabpfn_completion.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    RES.mkdir(parents=True,exist_ok=True)
    smoke() if args.smoke else run()
