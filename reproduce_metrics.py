"""Recompute all 18 models' reported metrics from saved predictions, without fitting."""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys
import numpy as np
import pandas as pd
from verify_package import main as verify_package

ROOT=Path(__file__).resolve().parent
R07='07_周期损失与TabPFN_20261018'
R08='08_形状水平解耦与施肥约束_20261018'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'_runs/metrics')
    args=ap.parse_args()
    verify_package()
    out=args.output.resolve()
    if out.exists(): raise RuntimeError(f'Output already exists; choose a fresh --output: {out}')
    out.mkdir(parents=True)
    # Preserve model predictions and archived run records; evaluation writes only here.
    for name in ['new_daily_predictions.csv','frozen_run.json','head_frozen_run.json','tabpfn_crossfit_frozen.json']:
        shutil.copy2(ROOT/R08/'results'/name,out/name)
    sys.path.insert(0,str(ROOT/R08/'scripts'))
    spec=importlib.util.spec_from_file_location('public_evaluation',ROOT/R08/'scripts/evaluate_results.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.RES=out
    mod.main()
    recomputed=pd.read_csv(out/'all_model_metrics.csv',float_precision='round_trip').set_index('model')
    reference=pd.read_csv(ROOT/R08/'results/all_model_metrics.csv',float_precision='round_trip').set_index('model')
    numeric=[c for c in reference if pd.api.types.is_numeric_dtype(reference[c])
             and not pd.api.types.is_bool_dtype(reference[c]) and c in recomputed]
    delta=(recomputed.loc[reference.index,numeric]-reference[numeric]).abs()
    maximum=float(delta.max().max())
    assert maximum<1e-9,maximum
    pair=pd.read_csv(out/'paired_R2_differences.csv',float_precision='round_trip')
    oldpair=pd.read_csv(ROOT/R08/'results/paired_R2_differences.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(pair,oldpair,check_exact=False,atol=1e-10,rtol=1e-10)
    check={m:{k:float(recomputed.loc[m,k]) for k in ['R2_daily','R2_cycle','continuous_window_WAPE']}
           for m in ['OriginalLightGBM','TabPFN_DailyRaw','MeanReconciled','AffineReconciled','DecoupledPulse']}
    report=dict(passed=True,models=len(recomputed),maximum_absolute_metric_difference=maximum,
                paired_intervals_match=True,core_metrics=check,
                scope='Recomputation conditional on frozen predictions; no model training')
    (out/'reproduction_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
