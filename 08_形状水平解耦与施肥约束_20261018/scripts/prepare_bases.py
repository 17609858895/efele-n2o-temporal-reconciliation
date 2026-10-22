"""Reconstruct the original algorithm inside each outer training set."""
import itertools
import time
import platform
from common import *

def main():
    for p in [RES, CACHE, FOLDS, MODELS]: p.mkdir(parents=True, exist_ok=True)
    if (RES/'base_completion.json').exists():
        raise RuntimeError('Completed preparation exists; preserve it.')
    old = module('original08', ROOT/'scripts/pilot_memory_n2o.py')
    start = time.perf_counter()
    data, manifest, events, arrays = old.load_data()
    np.savez_compressed(CACHE/'event_arrays.npz', A=arrays[0], IM=arrays[1], IO=arrays[2])
    save(events, RES/'fertilizer_events.csv')
    save(manifest, RES/'season_manifest.csv')
    sources = [Path(__file__), OUT/'protocol.md', OUT/'scripts/common.py',
               ROOT/'scripts/pilot_memory_n2o.py',
               ROOT/'03_首轮实测_20261018/all_heldout_predictions.csv',
               PREV/'results/tabpfn_daily_predictions.csv'] + list(old.RAW.glob('EFELE_*'))
    dump(dict(source_hashes={str(p):sha(p) for p in sources}, features=old.FEATURES,
              seed=SEED, lightgbm=old.lightgbm.__version__, python=platform.python_version(),
              start_time=time.strftime('%Y-%m-%dT%H:%M:%S'),
              main_model='DecoupledPulse', primary_lambda=1., sensitivity_lambdas=[.1,10.]),
         RES/'frozen_run.json')
    driver = old.ForceCache(data)
    configs = old.candidates('LightGBM')
    predictions, logs, selections = {}, [], []
    def candidates(key):
        if key in predictions: return predictions[key]
        dd = driver.get(key)
        pp = []
        for ci, cfg in enumerate(configs):
            pred, _, info = old.fit_model('LightGBM', cfg, dd, arrays, key)
            pp.append(pred)
            logs.append(dict(train_cycles=list(key), candidate=ci, config=cfg, **info))
        predictions[key] = np.column_stack(pp)
        np.savez_compressed(CACHE/('lgb_candidates_'+key_name(key)+'.npz'), predictions=predictions[key])
        return predictions[key]
    keys = [tuple(y for y in YEARS if y not in excluded)
            for n in [2, 1] for excluded in itertools.combinations(YEARS,n)]
    original = read(ROOT/'03_首轮实测_20261018/all_heldout_predictions.csv')
    original = original.query("protocol=='loco' and model=='LightGBM'").set_index('row_id')
    reproduction = []
    for ki, key in enumerate(keys):
        scores = np.zeros(4)
        for tr, val in old.inner_splits(key,'loco'):
            pp = candidates(tr)
            dd = driver.get(tr)
            scores += np.array([old.cycle_rmse(dd, pp[:,j], [val]) for j in range(4)])/2
        choice = int(np.argmin(scores))
        pp = candidates(key)[:, choice]
        dd = driver.get(key).copy()
        dd['p_lgb'] = pp
        dd = dd.loc[dd.season.isin(YEARS)]
        save(dd, FOLDS/(key_name(key)+'.csv'))
        selections.append(dict(train_cycles=list(key), config=configs[choice], candidate=choice,
                               inner_scores=scores.tolist(), internal_validation=[v for _,v in old.inner_splits(key,'loco')]))
        if len(key)==7:
            held = next(y for y in YEARS if y not in key)
            test = dd.loc[dd.season.eq(held)]
            ref = original.loc[test.row_id, 'prediction'].to_numpy()
            err = float(np.max(np.abs(test.p_lgb.to_numpy()-ref)))
            reproduction.append(dict(held=held, max_prediction_difference=err, n=len(test)))
            assert err < 1e-9, (held, err)
        dump(logs, RES/'lgb_fit_log.json')
        dump(selections, RES/'lgb_selection_log.json')
        print(f'Prepared {ki+1}/36 subsets; {len(logs)} LightGBM fits; {time.perf_counter()-start:.1f}s',flush=True)
    save(pd.DataFrame(driver.audit), RES/'driver_imputation_audit.csv')
    dump(dict(status='completed', training_subsets=len(keys), lgb_fits=len(logs),
              source_reproduction=reproduction, seconds=time.perf_counter()-start),RES/'base_completion.json')
    print('Original LightGBM reproduced on every outer row; internal bases ready.',flush=True)

if __name__=='__main__': main()

