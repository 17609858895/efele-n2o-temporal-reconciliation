"""Fixed exploratory cycle-target pilot. All meta fitting is nested by cycle."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys
import time
import platform

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent
RES = OUT / 'results'
spec = importlib.util.spec_from_file_location('original_pilot', ROOT / 'scripts/pilot_memory_n2o.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
YEARS = tuple(old.YEARS)
ALPHAS = (1., 10., 100.)
PENALTY = .1
FEATURES = ['is_PS', 'main_maize', 'calendar_days', 'Imin_total', 'Iorg_total',
            'rain_total', 'T_mean', 'W_mean', 'wet_fraction', 'Nmin30_mean',
            'Norg60_mean', 'crop_age_mean']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(obj, name):
    (RES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def save(frame, name):
    frame.to_csv(RES / name, index=False, encoding='utf-8-sig')


def cycle_table(d):
    rows = []
    for (year, treatment), c in d.loc[d.season.isin(YEARS)].groupby(['season', 'treatment']):
        o = c.loc[c.y.notna()]
        rows.append(dict(season=year, treatment=treatment, y=o.y.mean(), n_obs=len(o),
                         is_PS=float(treatment == 'PS'), main_maize=float(year not in [2015, 2017, 2019]),
                         calendar_days=len(c), Imin_total=c.Imin.sum(), Iorg_total=c.Iorg.sum(),
                         rain_total=c.Rainfall_sum.sum(), T_mean=o['T'].mean(), W_mean=o.W.mean(),
                         wet_fraction=(o.W >= 60).mean(), Nmin30_mean=o.Imin_exp30.mean(),
                         Norg60_mean=o.Iorg_exp60.mean(), crop_age_mean=o.crop_age.mean()))
    result = pd.DataFrame(rows).sort_values(['season', 'treatment']).reset_index(drop=True)
    assert len(result) == 16 and np.isfinite(result[FEATURES + ['y']]).all().all()
    return result


def aggregate(d, predictions):
    return (d.assign(prediction=predictions).loc[d.y.notna() & d.season.isin(YEARS)]
            .groupby(['season', 'treatment']).prediction.mean().reindex(
                pd.MultiIndex.from_product([YEARS, ['AN', 'PS']])).to_numpy())


def hybrid_weights(p, y, prior):
    prior = np.array(prior, dtype=float)
    scale2 = max(1., float(np.var(y)))
    def loss(w):
        return np.mean((p @ w-y)**2)/scale2 + PENALTY*np.sum((w-prior)**2)
    result = minimize(loss, prior, method='SLSQP', bounds=[(0., 1.)]*len(prior),
                      constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
                      options={'ftol': 1e-12, 'maxiter': 200})
    assert result.success, result.message
    w = np.clip(result.x, 0, 1)
    return w/w.sum()


def compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for value in range(total+1):
            for rest in compositions(total-value, parts-1):
                yield (value,)+rest


def weighted_q(x, probability):
    ok = np.isfinite(x)
    order = np.argsort(x[ok])
    values, weights = x[ok][order], probability[ok][order]
    return values[np.searchsorted(np.cumsum(weights)/weights.sum(), [.025, .5, .975])]


def evaluate(predictions):
    counts = np.array(list(compositions(8, 8)))
    multiplicities = np.array([math.factorial(8)//math.prod(math.factorial(int(k)) for k in row) for row in counts])
    assert len(counts) == 6435 and multiplicities.sum() == 8**8
    probability = multiplicities / 8**8
    base = predictions.query("model == 'OriginalLightGBM'").sort_values(['season', 'treatment'])
    y = base.y.to_numpy()
    base_sse = ((base.prediction.to_numpy()-y)**2).reshape(8, 2).sum(1)
    sy, syy = y.reshape(8, 2).sum(1), (y*y).reshape(8, 2).sum(1)
    den = counts@syy - (counts@sy)**2/16
    base_boot = 1-(counts@base_sse)/den
    metrics, paired = [], []
    for name, d in predictions.groupby('model'):
        d = d.sort_values(['season', 'treatment'])
        assert np.allclose(d.y, y) and len(d) == 16
        p = d.prediction.to_numpy()
        error = p-y
        sse = (error**2).reshape(8, 2).sum(1)
        boot = 1-(counts@sse)/den
        lo, med, hi = weighted_q(boot, probability)
        rmse = float(np.sqrt(np.mean(error**2)))
        point = float(r2_score(y, p))
        row = dict(model=name, n_cycle_treatment=16, n_clusters=8,
                   prediction_R2=point, R2_lower=lo, R2_upper=hi, RMSE=rmse,
                   MAE=float(np.mean(np.abs(error))), rRMSE=rmse/np.mean(y),
                   RPIQ=float((np.quantile(y, .75)-np.quantile(y, .25))/rmse),
                   cycle_mean_bias_pct=100*error.sum()/y.sum(),
                   observed_sum_bias_pct=100*np.sum(error*d.n_obs)/np.sum(y*d.n_obs),
                   correlation_squared=float(np.corrcoef(y, p)[0, 1]**2))
        metrics.append(row)
        dlo, dmed, dhi = weighted_q(boot-base_boot, probability)
        squared_wins = int((sse < base_sse-1e-10).sum())
        absolute_wins = int((np.abs(error).reshape(8, 2).mean(1) <
                             np.abs(base.prediction.to_numpy()-y).reshape(8, 2).mean(1)-1e-10).sum())
        paired.append(dict(model=name, delta_R2=point-r2_score(y, base.prediction),
                           delta_R2_lower=dlo, delta_R2_upper=dhi,
                           cycle_MSE_wins=squared_wins, cycle_MAE_wins=absolute_wins))
    save(pd.DataFrame(metrics).sort_values('prediction_R2', ascending=False), 'cycle_metrics.csv')
    save(pd.DataFrame(paired), 'paired_vs_original.csv')
    save(predictions.assign(absolute_error=lambda z: abs(z.prediction-z.y),
                            squared_error=lambda z: (z.prediction-z.y)**2)
         .groupby(['model', 'season'], as_index=False).agg(MAE=('absolute_error', 'mean'),
                                                          MSE=('squared_error', 'mean')), 'per_cycle_errors.csv')


def main():
    RES.mkdir(parents=True, exist_ok=True)
    if (RES / 'cycle_metrics.csv').exists():
        raise RuntimeError('This frozen pilot already has results; preserve them rather than overwrite.')
    start = time.perf_counter()
    sources = [Path(__file__), OUT/'protocol.md', ROOT/'scripts/pilot_memory_n2o.py',
               ROOT/'03_首轮实测_20261018/all_heldout_predictions.csv']
    sources += list(old.RAW.glob('EFELE_*'))
    hashes = {str(p): sha(p) for p in sources}
    dump(dict(frozen_hashes=hashes, seed=old.SEED, alphas=ALPHAS, penalty=PENALTY,
              features=FEATURES, python=platform.python_version(), platform=platform.platform(),
              lightgbm=old.lightgbm.__version__, sklearn=old.sklearn.__version__,
              numpy=np.__version__, pandas=pd.__version__), 'frozen_run.json')
    data, manifest, events, arrays = old.load_data()
    driver_cache = old.ForceCache(data)
    grid = old.candidates('LightGBM')
    fit_cache, feature_cache, fit_audit = {}, {}, []

    def get_fit(train):
        key = tuple(sorted(train))
        if key not in fit_cache:
            dd = driver_cache.get(key)
            c = cycle_table(dd)
            daily_predictions = []
            for ci, cfg in enumerate(grid):
                p, _, info = old.fit_model('LightGBM', cfg, dd, arrays, key)
                daily_predictions.append(p)
                fit_audit.append(dict(train_cycles=key, candidate=ci,
                                      train_observations=int((dd.season.isin(key)&dd.y.notna()).sum()),
                                      target_scale=info['target_scale']))
            raw = np.column_stack(daily_predictions)
            agg = np.column_stack([aggregate(dd, raw[:, i]) for i in range(len(grid))])
            trainmask = c.season.isin(key)
            ridge = []
            for alpha in ALPHAS:
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                estimator.fit(c.loc[trainmask, FEATURES], c.loc[trainmask, 'y'])
                ridge.append(estimator.predict(c[FEATURES]))
            fit_cache[key] = (agg, np.column_stack(ridge), raw)
            feature_cache[key] = c
            print(f'fit training cycles {key}: {len(fit_cache)} unique subsets', flush=True)
        return fit_cache[key]

    # This template supplies only the common target and group index. No fitted feature
    # from this call is reused for a different training subset.
    get_fit(tuple(y for y in YEARS if y != YEARS[0]))
    template = feature_cache[tuple(y for y in YEARS if y != YEARS[0])]
    folds, outputs, daily_out, inner_rows = [], [], [], []
    for held in YEARS:
        train = tuple(y for y in YEARS if y != held)
        train_idx = template.season.isin(train).to_numpy()
        test_idx = template.season.eq(held).to_numpy()
        inner_lgb = np.full((16, 4), np.nan)
        inner_ridge = np.full((16, 3), np.nan)
        for validation in train:
            inner_train = tuple(y for y in train if y != validation)
            assert held not in inner_train and validation not in inner_train
            lp, rp, _ = get_fit(inner_train)
            val_idx = template.season.eq(validation).to_numpy()
            inner_lgb[val_idx] = lp[val_idx]
            inner_ridge[val_idx] = rp[val_idx]
            for row_idx in np.flatnonzero(val_idx):
                row = dict(outer_test=held, inner_validation=validation, inner_train=';'.join(map(str, inner_train)),
                           treatment=template.treatment.iloc[row_idx], y=template.y.iloc[row_idx])
                row.update({f'lgb_{i}': lp[row_idx, i] for i in range(4)})
                row.update({f'ridge_{a:g}': rp[row_idx, i] for i, a in enumerate(ALPHAS)})
                inner_rows.append(row)
        y_inner = template.y.to_numpy()[train_idx]
        assert np.isfinite(inner_lgb[train_idx]).all() and np.isnan(inner_lgb[test_idx]).all()
        l_scores = np.mean((inner_lgb[train_idx]-y_inner[:, None])**2, axis=0)
        r_scores = np.mean((inner_ridge[train_idx]-y_inner[:, None])**2, axis=0)
        lc, rc = int(np.argmin(l_scores)), int(np.argmin(r_scores))
        raw_ids = [i for i, cfg in enumerate(grid) if cfg['transform'] == 'raw']
        asinh_ids = [i for i, cfg in enumerate(grid) if cfg['transform'] == 'asinh']
        raw_c = min(raw_ids, key=lambda i: l_scores[i])
        asinh_c = min(asinh_ids, key=lambda i: l_scores[i])
        w2 = hybrid_weights(np.column_stack([inner_lgb[train_idx, lc], inner_ridge[train_idx, rc]]),
                            y_inner, [.5, .5])
        w3 = hybrid_weights(np.column_stack([inner_lgb[train_idx, raw_c], inner_lgb[train_idx, asinh_c],
                                             inner_ridge[train_idx, rc]]), y_inner, [.5, .5, 0.])
        lp, rp, daily = get_fit(train)
        pred2 = np.column_stack([lp[:, lc], rp[:, rc]])@w2
        pred3 = np.column_stack([lp[:, raw_c], lp[:, asinh_c], rp[:, rc]])@w3
        for name, pred in [('CycleTunedLGB', lp[:, lc]), ('DirectCycleRidge', rp[:, rc]),
                           ('TwoViewHybrid', pred2), ('ThreeExpertHybrid', pred3)]:
            t = template.loc[test_idx, ['season', 'treatment', 'y', 'n_obs']].copy()
            t['prediction'], t['model'] = pred[test_idx], name
            outputs.append(t)
        t = template.loc[test_idx, ['season', 'treatment', 'y', 'n_obs']].copy()
        t['model'], t['prediction'] = 'GlobalCycleMean', y_inner.mean()
        outputs.append(t)
        for idx in np.flatnonzero(test_idx):
            r = template.iloc[idx]
            same = train_idx & (template.main_maize == r.main_maize) & (template.treatment == r.treatment)
            assert same.any()
            outputs.append(pd.DataFrame([dict(season=held, treatment=r.treatment, y=r.y, n_obs=r.n_obs,
                                               prediction=template.loc[same, 'y'].mean(), model='CropTreatmentMean')]))
        dd = data.loc[data.season.eq(held), ['Date', 'season', 'treatment', 'y']].copy()
        dd['prediction'] = daily[data.season.eq(held), lc]
        dd['model'] = 'CycleTunedLGB'
        daily_out.append(dd)
        feature = feature_cache[train].copy()
        feature['outer_test'] = held
        save(feature, f'features_outer_{held}.csv')
        folds.append(dict(test_cycle=held, train_cycles=train, lgb_choice=grid[lc], ridge_alpha=ALPHAS[rc],
                          lgb_inner_MSE=l_scores.tolist(), ridge_inner_MSE=r_scores.tolist(),
                          two_view_weights=w2.tolist(), three_expert_weights=w3.tolist(),
                          raw_config=grid[raw_c], asinh_config=grid[asinh_c]))
        print(f'outer {held} finished; predictions retained without interim performance selection', flush=True)
    source = pd.read_csv(ROOT/'03_首轮实测_20261018/all_heldout_predictions.csv')
    ref = source.query("protocol == 'loco' and model == 'LightGBM'").dropna(subset=['y'])
    ref = ref.groupby(['season', 'treatment'], as_index=False).agg(y=('y', 'mean'), prediction=('prediction', 'mean'), n_obs=('y', 'size'))
    ref['model'] = 'OriginalLightGBM'
    outputs.append(ref)
    preds = pd.concat(outputs, ignore_index=True).sort_values(['model', 'season', 'treatment'])
    assert not preds.duplicated(['model', 'season', 'treatment']).any()
    for _, p in preds.groupby('model'):
        assert np.allclose(p.y.to_numpy(), template.y.to_numpy())
        assert np.array_equal(p.n_obs.to_numpy(), template.n_obs.to_numpy())
    # Test the strongest information boundary on one complete outer fit: changing
    # held-out flux values while retaining the observation mask cannot affect it.
    train = tuple(y for y in YEARS if y != 2014)
    changed = data.copy()
    m = changed.season.eq(2014) & changed.y.notna()
    changed.loc[m, 'y'] += 1e6
    dd = old.ForceCache(changed).get(train)
    trial, _, _ = old.fit_model('LightGBM', grid[0], dd, arrays, train)
    original = get_fit(train)[2][:, 0]
    assert np.allclose(trial, original, rtol=0, atol=1e-10)
    changed_c = cycle_table(dd)
    assert np.allclose(changed_c[FEATURES], feature_cache[train][FEATURES])
    save(preds, 'cycle_predictions.csv')
    save(pd.concat(daily_out, ignore_index=True), 'cycle_tuned_daily_predictions.csv')
    save(pd.DataFrame(inner_rows), 'nested_inner_predictions.csv')
    save(pd.DataFrame(driver_cache.audit), 'soil_driver_audit.csv')
    dump(folds, 'fold_choices.json')
    dump(fit_audit, 'fit_audit.json')
    evaluate(preds)
    unchanged = {path: sha(path) == value for path, value in hashes.items()}
    assert all(unchanged.values())
    dump(dict(status='passed', n_unique_train_subsets=len(fit_cache), n_base_lgb_fits=len(fit_audit),
              n_models=preds.model.nunique(), n_prediction_rows=len(preds),
              n_inner_prediction_rows=len(inner_rows), outer_flux_perturbation_invariant=True,
              all_source_hashes_unchanged=unchanged, seconds=time.perf_counter()-start,
              interpretation='exploratory development; no implemented LLM Agent; no independent DayCent comparison'),
         'verification.json')
    print(pd.read_csv(RES/'cycle_metrics.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
