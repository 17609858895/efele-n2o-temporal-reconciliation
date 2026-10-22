"""Reanalyse saved predictions only. No fitting, target filling, or model selection.

All scales use exactly the same observed target dates for y and prediction.
Bootstrap retains repeated cycle draws and uses common draws for every model.
Intervals condition on the saved cross-validated predictions (not refit CIs).
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'results'
ROOT = HERE.parents[1]
B = 20000
SEED = 0
YEARS = np.arange(2014, 2022)
KEY = ['season', 'treatment', 'Date']


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    denom = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((p - y) ** 2) / denom if denom > 0 else np.nan


def tables(g):
    d = g.loc[g.y.notna()].copy()
    assert np.isfinite(d.prediction).all()
    result = {'daily': d}
    for name, freq, min_n in [('weekly', 'W', 4), ('monthly', 'M', 10)]:
        x = d.assign(period=d.Date.dt.to_period(freq))
        x = x.groupby(['season', 'treatment', 'period'], as_index=False).agg(
            y=('y', 'mean'), prediction=('prediction', 'mean'), n_obs=('y', 'size'))
        result[name] = x.loc[x.n_obs >= min_n].copy()
    result['cycle'] = d.groupby(['season', 'treatment'], as_index=False).agg(
        y=('y', 'mean'), prediction=('prediction', 'mean'), n_obs=('y', 'size'))
    return result


def cluster_stats(d):
    x = d.assign(sse=(d.prediction - d.y) ** 2, yy=d.y ** 2)
    return x.groupby('season').agg(n=('y', 'size'), sy=('y', 'sum'),
                                  syy=('yy', 'sum'), sse=('sse', 'sum')).reindex(YEARS)


def bootstrap_r2(d, counts):
    s = cluster_stats(d)
    a = counts @ s[['n', 'sy', 'syy', 'sse']].to_numpy()
    denom = a[:, 2] - a[:, 1] ** 2 / a[:, 0]
    return np.divide(a[:, 3], denom, out=np.full(len(a), np.nan), where=denom > 1e-12) * -1 + 1


def compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for value in range(total + 1):
            for rest in compositions(total - value, parts - 1):
                yield (value,) + rest


def weighted_quantiles(values, probability):
    ok = np.isfinite(values)
    order = np.argsort(values[ok])
    x, w = values[ok][order], probability[ok][order]
    cdf = np.cumsum(w)/w.sum()
    return x[np.searchsorted(cdf, [.025, .5, .975])]


def cycle_rmse(d):
    x = d.assign(sse=(d.prediction - d.y) ** 2)
    z = x.groupby(['season', 'treatment']).sse.mean().pow(.5)
    return z.groupby('season').mean().reindex(sorted(d.season.unique()))


def windows(g):
    rows = []
    for (s, tr), x in g.groupby(['season', 'treatment']):
        x = x.sort_values('Date').copy()
        ok = x.y.notna()
        blocks = (ok.ne(ok.shift()) | x.Date.diff().ne(pd.Timedelta(days=1))).cumsum()
        for _, w in x.groupby(blocks):
            if not w.y.notna().all() or len(w) < 7:
                continue
            rows.append(dict(season=s, treatment=tr, start=w.Date.min(), end=w.Date.max(),
                             n_days=len(w), observed_integral=w.y.sum(),
                             predicted_integral=w.prediction.sum(),
                             absolute_observed_sum=max(1., w.y.abs().sum()),
                             abs_error=abs(w.prediction.sum() - w.y.sum())))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / '03_首轮实测_20261018/all_heldout_predictions.csv',
               ROOT / '04_R2诊断_20261018/results/variant_heldout_predictions.csv',
               ROOT / '04_R2诊断_20261018/results/lightgbm_variant_summary.csv',
               ROOT / '04_R2诊断_20261018/results/pilot_season_bootstrap_R2.csv',
               ROOT / '04_R2诊断_20261018/scripts/05_summarize.py',
               ROOT / '04_R2诊断_20261018/R2诊断与修改方案.md']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    pil = pd.read_csv(sources[0], parse_dates=['Date'])
    var = pd.read_csv(sources[1], parse_dates=['Date']).rename(columns={'variant': 'model'})
    pil['family'] = 'pilot'
    var['family'], var['protocol'] = 'variant', 'loco'
    allp = pd.concat([pil, var], ignore_index=True)
    assert not allp.duplicated(['protocol', 'model'] + KEY).any()
    ref = allp.query("protocol == 'loco' and model == 'LightGBM'").sort_values(KEY).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(len(YEARS), (B, len(YEARS)), replace=True)
    mc_counts = np.stack([(draws == i).sum(axis=1) for i in range(len(YEARS))], axis=1)
    np.save(OUT / 'cycle_bootstrap_draw_indices.npy', draws)
    counts = np.array(list(compositions(8, 8)))
    multiplicities = np.array([math.factorial(8)//math.prod(math.factorial(int(k)) for k in row) for row in counts])
    probability = multiplicities/8**8
    assert len(counts) == 6435 and multiplicities.sum() == 8**8
    save(pd.DataFrame(counts, columns=[str(y) for y in YEARS]).assign(sequence_multiplicity=multiplicities, probability=probability),
         'exact_cluster_bootstrap_compositions.csv')
    metrics, interval_rows, cycles, support, wins, all_windows = [], [], [], [], [], []
    boot, points = {}, {}
    for (protocol, model), g in allp.groupby(['protocol', 'model']):
        if protocol == 'loco':
            z = g.sort_values(KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(z[KEY], ref[KEY])
            assert np.allclose(z.y, ref.y, equal_nan=True)
        family = g.family.iloc[0]
        by_scale = tables(g)
        d, c = by_scale['daily'], by_scale['cycle']
        cyc = cycle_rmse(d)
        win = windows(g)
        all_windows.append(win.assign(protocol=protocol, model=model))
        row = dict(protocol=protocol, family=family, model=model, n_calendar_rows=len(g),
                   n_observed_rows=len(d), n_cycles=d.season.nunique(),
                   macro_rmse=cyc.mean(), daily_rmse=np.sqrt(np.mean((d.prediction-d.y)**2)),
                   daily_mae=np.mean(np.abs(d.prediction-d.y)),
                   observed_sum_bias_pct=100*(d.prediction.sum()-d.y.sum())/d.y.sum(),
                   continuous_window_wape=win.abs_error.sum()/win.absolute_observed_sum.sum(),
                   n_continuous_windows=len(win))
        for scale, t in by_scale.items():
            value = r2(t.y, t.prediction)
            assert np.isclose(value, r2_score(t.y, t.prediction))
            row[f'R2_{scale}'] = value
            row[f'n_{scale}'] = len(t)
            row[f'RMSE_{scale}'] = np.sqrt(np.mean((t.prediction-t.y)**2))
            row[f'corr_squared_{scale}'] = np.corrcoef(t.y, t.prediction)[0, 1]**2
            if protocol == 'loco':
                samples = bootstrap_r2(t, counts)
                boot[(model, scale)], points[(model, scale)] = samples, value
                low, med, high = weighted_quantiles(samples, probability)
                interval_rows.append(dict(model=model, scale=scale, estimate=value,
                                          q025=low, median=med, q975=high,
                                          finite_compositions=int(np.isfinite(samples).sum()),
                                          finite_probability=float(probability[np.isfinite(samples)].sum())))
        row['rRMSE_cycle'] = row['RMSE_cycle']/c.y.mean()
        row['RPIQ_cycle'] = (c.y.quantile(.75)-c.y.quantile(.25))/row['RMSE_cycle']
        other = d.loc[~((d.season == 2014) & (d.treatment == 'AN'))]
        row['R2_daily_ex2014AN'] = r2(other.y, other.prediction)
        row['R2_asinh10'] = r2(np.arcsinh(d.y/10), np.arcsinh(d.prediction/10))
        metrics.append(row)
        if protocol == 'loco':
            boot[(model, 'macro_rmse')] = counts @ cyc.reindex(YEARS).to_numpy()/len(YEARS)
            points[(model, 'macro_rmse')] = cyc.mean()
        for season, err in cyc.items():
            cycles.append(dict(protocol=protocol, model=model, season=season, macro_rmse=err))
        if model == 'LightGBM' and protocol == 'loco':
            nc = g.groupby(['season', 'treatment']).size().rename('n_calendar').reset_index()
            cov = c.merge(nc, on=['season', 'treatment'])
            cov['coverage'] = cov.n_obs/cov.n_calendar
            save(cov, 'observed_day_cycle_means_and_coverage.csv')
            for scale, t in by_scale.items():
                for s, x in t.groupby('season'):
                    support.append(dict(season=s, scale=scale, units=len(x)))
    metric_frame = pd.DataFrame(metrics).sort_values(['protocol', 'R2_daily'], ascending=[True, False])
    save(metric_frame, 'corrected_multiscale_metrics.csv')
    save(pd.DataFrame(interval_rows), 'corrected_cluster_bootstrap_R2.csv')
    save(pd.DataFrame(cycles), 'cycle_RMSE.csv')
    save(pd.DataFrame(support), 'scale_support_by_cycle.csv')
    save(pd.concat(all_windows, ignore_index=True), 'continuous_observed_windows.csv')
    pairs = []
    for (model, scale), dist in boot.items():
        if model == 'LightGBM':
            continue
        delta = dist-boot[('LightGBM', scale)]
        lo, med, hi = weighted_quantiles(delta, probability)
        pairs.append(dict(model=model, reference='LightGBM', metric=scale,
                          delta=points[(model, scale)]-points[('LightGBM', scale)],
                          q025=lo, median=med, q975=hi))
    save(pd.DataFrame(pairs), 'paired_differences_vs_LightGBM.csv')
    cy = pd.DataFrame(cycles).query("protocol == 'loco'").pivot(index='season', columns='model', values='macro_rmse')
    for m in cy:
        wins.append(dict(model=m, better_cycles=int((cy[m]<cy.LightGBM).sum()),
                         tied_cycles=int(np.isclose(cy[m], cy.LightGBM).sum()),
                         worse_cycles=int((cy[m]>cy.LightGBM).sum())))
    save(pd.DataFrame(wins), 'paired_cycle_wins.csv')
    # Decisive checks: masks, original metrics, and literal cluster multiplicities.
    d = tables(ref)['daily']
    first = metric_frame.query("protocol == 'loco' and model == 'LightGBM'").iloc[0]
    assert len(d) == 3319 and len(ref) == 5970
    assert np.isclose(first.R2_daily, .2777018390485084)
    assert np.isclose(first.R2_cycle, .7634799697989234)
    assert np.isclose(first.continuous_window_wape, .567877, atol=1e-6)
    for scale, t in tables(ref).items():
        for i in [0, 1, 17]:
            explicit = pd.concat([t.loc[t.season == YEARS[j]] for j in draws[i]])
            assert np.isclose(r2(explicit.y, explicit.prediction), bootstrap_r2(t, mc_counts[i:i+1])[0])
    known = np.array([0, 0, 0, 1, 2, 3, 4, 5])
    c = tables(ref)['cycle']
    explicit = pd.concat([c.loc[c.season == YEARS[j]] for j in known])
    collapsed = explicit.groupby(['season', 'treatment']).mean(numeric_only=True).reset_index()
    assert len(explicit) == 16 and len(collapsed) == 12
    old_var = pd.read_csv(sources[2])
    old_ci = pd.read_csv(sources[3])
    checks = dict(status='PASS', no_model_training=True, bootstrap_seed=SEED,
                  primary_bootstrap='exact multinomial cluster bootstrap; inverse weighted empirical CDF',
                  primary_bootstrap_compositions=len(counts), ordered_resamples=8**8,
                  monte_carlo_replicates_for_sensitivity=B, independent_clusters=8,
                  scored_rows_per_loco_model=3319, predicted_rows_per_loco_model=len(ref),
                  prediction_and_truth_masks_identical=True, all_loco_models_identical_support=True,
                  multiplicity_check_explicit_rows=len(explicit), multiplicity_check_collapsed_rows=len(collapsed),
                  multiplicity_check_correct_R2=r2(explicit.y, explicit.prediction),
                  multiplicity_check_incorrect_R2=r2(collapsed.y, collapsed.prediction),
                  old_variant_summary_missing_daily_R2=int(old_var.R2_daily.isna().sum()),
                  old_LightGBM_cycle_CI=old_ci.loc[old_ci.model=='LightGBM', ['cycle_R2_q025','cycle_R2_q975']].iloc[0].to_dict(),
                  first_2000_original_seed_daily_CI=np.percentile(bootstrap_r2(d, mc_counts[:2000]), [2.5,97.5]).tolist(),
                  first_2000_corrected_cycle_CI=np.percentile(bootstrap_r2(c, mc_counts[:2000]), [2.5,97.5]).tolist(),
                  mc_20000_daily_CI=np.percentile(bootstrap_r2(d, mc_counts), [2.5,97.5]).tolist(),
                  mc_20000_cycle_CI=np.percentile(bootstrap_r2(c, mc_counts), [2.5,97.5]).tolist(),
                  input_hashes=hashes)
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(metric_frame.query("protocol == 'loco'")[['model','R2_daily','R2_monthly','R2_cycle','macro_rmse','observed_sum_bias_pct','continuous_window_wape']].round(6).to_string(index=False))
    print(pd.DataFrame(interval_rows).query("model == 'LightGBM'").to_string(index=False))
    print(pd.DataFrame(pairs).query("model in ['V6_plus_offW_asinh','s20_reg','s20_reg_cal','FixedMemory','GatedMemory']").to_string(index=False))


if __name__ == '__main__':
    main()
