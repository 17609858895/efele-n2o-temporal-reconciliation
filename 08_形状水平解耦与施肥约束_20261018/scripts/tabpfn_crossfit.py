"""GPU inference for all 28 inner crossfit contexts. No outer score selection."""
from pathlib import Path
import gc
import os
import json
import time
import sys
import hashlib
from importlib.metadata import version
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from common import OUT, ROOT, RES, FOLDS, CACHE, YEARS, SEED, dump, save, sha, read

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)
CHECKPOINT=Path(os.environ.get('TABPFN_CHECKPOINT', str(Path.home()/'.cache/tabpfn/tabpfn-v2-regressor.ckpt'))).expanduser()

def main():
    if (RES/'tabpfn_crossfit_completion.json').exists():
        raise RuntimeError('Completed crossfit exists; preserve it.')
    frozen=json.loads((RES/'frozen_run.json').read_text(encoding='utf-8'))
    files=sorted(p for p in FOLDS.glob('*.csv') if len(p.stem.split('_'))==6)
    assert len(files)==28 and torch.cuda.is_available()
    dump(dict(source_hashes={str(p):sha(p) for p in [Path(__file__),OUT/'protocol.md',CHECKPOINT]+files},
              gpu=torch.cuda.get_device_name(0), estimators=4, seed=SEED,
              versions={k:version(k) for k in ['tabpfn','torch','scikit-learn','numpy','pandas']}),
         RES/'tabpfn_crossfit_frozen.json')
    start=time.perf_counter(); logs=[]
    for i,path in enumerate(files):
        tic=time.perf_counter()
        key=tuple(map(int,path.stem.split('_')))
        d=read(path)
        tr=d.season.isin(key)&d.y.notna()
        X=d.loc[tr,frozen['features']].to_numpy(dtype='float32')
        y=d.loc[tr,'y'].to_numpy(dtype='float32')
        model=TabPFNRegressor(model_path=str(CHECKPOINT),device='cuda',n_estimators=4,
            random_state=SEED,n_jobs=1,fit_mode='low_memory',memory_saving_mode=True)
        model.fit(X,y)
        rows=[]
        for held in YEARS:
            if held in key: continue
            test=d.season.eq(held)
            q=d.loc[test,frozen['features']].to_numpy(dtype='float32')
            pp=[]
            for begin in range(0,len(q),256):
                query=q[begin:begin+256]
                try: part=model.predict(query)
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache()
                    part=np.concatenate([model.predict(query[j:j+64]) for j in range(0,len(query),64)])
                pp.append(part)
            p=np.concatenate(pp)
            assert np.isfinite(p).all()
            rr=d.loc[test,['row_id','season','treatment','Date']].copy()
            rr['p_tab']=p
            rows.append(rr)
        save(pd.concat(rows,ignore_index=True),CACHE/('tab_'+path.stem+'.csv'))
        logs.append(dict(train_cycles=list(key), predicted_cycles=[v for v in YEARS if v not in key],
                         train_observations=len(y), predicted_days=sum(len(r) for r in rows),
                         seconds=time.perf_counter()-tic, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        dump(logs,RES/'tabpfn_crossfit_log.json')
        del model
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        print(f'TabPFN internal contexts {i+1}/28; {time.perf_counter()-start:.1f}s',flush=True)
    dump(dict(status='completed', fits=len(logs), seconds=time.perf_counter()-start),RES/'tabpfn_crossfit_completion.json')

if __name__=='__main__': main()
