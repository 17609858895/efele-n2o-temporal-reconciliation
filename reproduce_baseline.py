"""Refit only the paper's LightGBM baseline using the original split/fit functions."""
from pathlib import Path
import importlib.util
import pandas as pd

ROOT=Path(__file__).resolve().parent

def main():
    spec=importlib.util.spec_from_file_location('baseline_source',ROOT/'scripts/pilot_memory_n2o.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    target=old.OUT/'all_heldout_predictions.csv'
    if target.exists():raise RuntimeError('Baseline output exists; run inside a fresh training workspace.')
    data,manifest,events,arrays=old.load_data()
    cache=old.ForceCache(data);rows=[]
    for held in old.YEARS:
        years=tuple(y for y in old.YEARS if y!=held)
        folder=old.OUT/'folds'/f'loco_{held}';folder.mkdir(parents=True)
        prediction,_,_=old.select_fit('LightGBM',data,arrays,cache,years,'loco',folder)
        mask=data.season.eq(held)
        r=data.loc[mask,['row_id','Date','season','treatment','y']].copy()
        r['prediction']=prediction[mask];r['protocol']='loco';r['model']='LightGBM'
        rows.append(r)
        print('Completed baseline held-out cycle',held,flush=True)
    pd.concat(rows,ignore_index=True).to_csv(target,index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
