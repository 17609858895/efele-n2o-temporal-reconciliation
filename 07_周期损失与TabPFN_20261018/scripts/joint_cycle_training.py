"""Frozen nested cycle-loss experiment. No outer target enters model selection."""
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
RES = OUT / 'results'
MODELS = OUT / 'models'
FEATURES_DIR = OUT / 'fold_data'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

old = module('original_pilot', ROOT / 'scripts/pilot_memory_n2o.py')
prior = module('cycle_pilot', ROOT / '06_周期目标与Agent方法_20261018/scripts/cycle_target_pilot.py')
YEARS = tuple(old.YEARS)
CAPS = [(7, 200), (15, 300)]
CANDIDATES = []
for cap, (leaves, trees) in enumerate(CAPS):
    for name, a, b in [('HuberDay', 1, 0), ('Joint01', 1, .1), ('Joint1', 1, 1),
                       ('Joint10', 1, 10), ('CycleOnly', 0, 1),
                       ('RawMSE', None, None), ('AsinhMSE', None, None)]:
        CANDIDATES.append(dict(id=f'{name}_c{cap}', family=name, cap=cap,
                               leaves=leaves, trees=trees, a=a, b=b))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def save(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')

class Objective:
    """Exact gradient and a local diagonal curvature upper bound for coupled means."""
    def __init__(self, y, groups, a, b):
        self.y = np.asarray(y, dtype=float)
        self.g = np.asarray(groups, dtype=int)
        self.a, self.b = a, b
        self.n = np.bincount(self.g)
        self.G, self.N = len(self.n), len(self.y)
        assert (self.n > 0).all()
        self.ym = np.bincount(self.g, weights=self.y) / self.n
        self.sd = max(1., float(np.median(np.abs(self.y))))
        self.sc = max(1., float(np.std(self.ym)))
        self.zt = np.arcsinh(self.y / self.sd)
        self.w = self.N / (self.G * self.n[self.g])
        self.z0 = float(np.sum(self.w * self.zt) / self.N)
        self.history = []

    def components(self, z):
        f = self.sd * np.sinh(z)
        residual = np.bincount(self.g, weights=f) / self.n - self.ym
        day = np.sum(self.w * (np.sqrt(1 + (z-self.zt)**2)-1)) / self.N
        cyc = .5 * np.mean(residual**2) / self.sc**2
        return float(day), float(cyc), residual

    def derivatives(self, z):
        z = np.asarray(z, dtype=float)
        assert np.isfinite(z).all() and np.max(np.abs(z)) < 20, 'Invalid latent margin'
        e = z - self.zt
        f, u = self.sd * np.sinh(z), self.sd * np.cosh(z)
        residual = np.bincount(self.g, weights=f) / self.n - self.ym
        mean_u = np.bincount(self.g, weights=u) / self.n
        gd = self.w * e / np.sqrt(1+e*e)
        hd = self.w / np.power(1+e*e, 1.5)
        gc = self.w * residual[self.g] * u / self.sc**2
        hc = self.w * (u * mean_u[self.g] + np.abs(residual[self.g]*f)) / self.sc**2
        return self.a*gd + self.b*gc, np.maximum(self.a*hd + self.b*hc, 1e-6)

    def __call__(self, z, dataset):
        grad, hess = self.derivatives(z)
        if len(self.history) == 0 or self.calls % 50 == 0:
            day, cyc, _ = self.components(z)
            self.history.append(dict(iteration=self.calls, day=day, cycle=cyc,
                                     total=self.a*day+self.b*cyc,
                                     gradient_abs_mean=float(np.mean(np.abs(grad))),
                                     hessian_min=float(hess.min())))
        self.calls += 1
        return grad, hess

def training_groups(d, train):
    mask = (d.season.isin(train) & d.y.notna()).to_numpy()
    sub = d.loc[mask]
    group, _ = pd.factorize(pd.MultiIndex.from_arrays([sub.season, sub.treatment]), sort=True)
    return mask, group

def fit_candidate(d, train, cfg, save_model=None, rounds=None):
    tic = time.perf_counter()
    mask, groups = training_groups(d, train)
    X = d[old.FEATURES].to_numpy()
    y = d.loc[mask, 'y'].to_numpy()
    ntrees = rounds or cfg['trees']
    params = dict(num_leaves=cfg['leaves'], learning_rate=.04, min_data_in_leaf=25,
                  lambda_l2=10., verbosity=-1, num_threads=2, seed=old.SEED,
                  deterministic=True, force_col_wise=True, metric='None')
    if cfg['a'] is not None:
        obj = Objective(y, groups, cfg['a'], cfg['b'])
        obj.calls = 0
        trainset = lgb.Dataset(X[mask], label=y, init_score=np.full(len(y), obj.z0),
                               feature_name=old.FEATURES, free_raw_data=False)
        # LightGBM deep-copies params; a closure keeps the audit history on this obj.
        params.update(objective=lambda z, ds: obj(z, ds), max_delta_step=2.)
        booster = lgb.train(params, trainset, num_boost_round=ntrees)
        z = obj.z0 + booster.predict(X, num_threads=2)
        pred = obj.sd * np.sinh(z)
        day, cyc, _ = obj.components(z[mask])
        obj.history.append(dict(iteration=ntrees, day=day, cycle=cyc,
                                 total=obj.a*day+obj.b*cyc))
        info = dict(s_day=obj.sd, s_cycle=obj.sc, intercept=obj.z0,
                    target_group_means=obj.ym.tolist(), history=obj.history)
        assert len(obj.history) >= (3 if ntrees > 50 else 2), 'Missing objective history'
        assert obj.history[-1]['total'] < obj.history[0]['total'], 'Loss did not decrease'
    else:
        sd = max(1., float(np.median(np.abs(y))))
        target = np.arcsinh(y/sd) if cfg['family']=='AsinhMSE' else y
        trainset = lgb.Dataset(X[mask], label=target, weight=old.weights(d, mask),
                               feature_name=old.FEATURES, free_raw_data=False)
        params.update(objective='regression')
        booster = lgb.train(params, trainset, num_boost_round=ntrees)
        z = booster.predict(X, num_threads=2)
        pred = sd*np.sinh(np.clip(z, -10, 10)) if cfg['family']=='AsinhMSE' else z
        info = dict(s_day=sd)
    assert np.isfinite(pred).all()
    info.update(config=cfg, train_cycles=list(train), train_observations=int(mask.sum()),
                train_groups=int(len(np.unique(groups))), seconds=time.perf_counter()-tic)
    if save_model:
        # The Windows C++ file API rejects Chinese paths; Python UTF-8 I/O works.
        model_text = booster.model_to_string()
        Path(save_model).write_text(model_text, encoding='utf-8')
        restored = lgb.Booster(model_str=Path(save_model).read_text(encoding='utf-8'))
        assert np.allclose(restored.predict(X[:3], num_threads=2), booster.predict(X[:3], num_threads=2))
        dump(info, Path(str(save_model)+'.json'))
    return pred, info

def mathematical_checks():
    rng = np.random.default_rng(old.SEED)
    y = np.array([-.5, 2, 12, 5, 35, 9, 8.])
    g = np.array([0,0,0,1,1,2,2])
    obj = Objective(y, g, 1, 1)
    z = np.arcsinh(y/obj.sd) + rng.normal(0,.5,len(y))
    grad, major = obj.derivatives(z)
    def total(zz):
        d, c, _ = obj.components(zz)
        return obj.N * (d+c)
    eps = 1e-6
    numerical = np.array([(total(z+eps*np.eye(len(z))[i])-total(z-eps*np.eye(len(z))[i]))/(2*eps)
                          for i in range(len(z))])
    tz = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    def torch_loss(v):
        p = obj.sd * torch.sinh(v)
        value = 0.
        for k in range(obj.G):
            idx = np.flatnonzero(g==k)
            e = v[idx]-torch.tensor(obj.zt[idx])
            value = value + (torch.sqrt(1+e*e)-1).mean()
            value = value + .5*((p[idx].mean()-obj.ym[k])/obj.sc)**2
        return value * obj.N / obj.G
    H = torch.autograd.functional.hessian(torch_loss, tz).detach().numpy()
    eigen = np.linalg.eigvalsh(np.diag(major)-H)
    result = dict(analytic_gradient_max_error=float(np.max(np.abs(grad-numerical))),
                  diagonal_upper_bound_min_eigenvalue=float(eigen.min()),
                  coupled_hessian_max_off_diagonal=float(np.max(np.abs(H-np.diag(np.diag(H))))))
    assert np.allclose(grad, numerical, atol=1e-6, rtol=1e-6)
    assert eigen.min() > -1e-9
    assert result['coupled_hessian_max_off_diagonal'] > 0
    return result

def prepare_outer_data():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    data, _, _, _ = old.load_data()
    cache = old.ForceCache(data)
    for held in YEARS:
        d = cache.get(tuple(y for y in YEARS if y != held))
        save(d.loc[d.season.isin(YEARS)], FEATURES_DIR/f'outer_{held}_daily.csv')
        save(prior.cycle_table(d), FEATURES_DIR/f'outer_{held}_cycle.csv')
    print('All eight outer-fold driver tables prepared.', flush=True)

def run():
    for p in [RES, MODELS, FEATURES_DIR]: p.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT/'fit_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (RES/'joint_daily_predictions.csv').exists():
        raise RuntimeError('Frozen run already finished; preserve its results.')
    start = time.perf_counter()
    sources = [Path(__file__), OUT/'protocol.md', ROOT/'scripts/pilot_memory_n2o.py',
               ROOT/'06_周期目标与Agent方法_20261018/scripts/cycle_target_pilot.py',
               ROOT/'03_首轮实测_20261018/all_heldout_predictions.csv']
    sources += list(old.RAW.glob('EFELE_*'))
    dump(dict(source_hashes={str(p):sha(p) for p in sources}, candidates=CANDIDATES,
              seed=old.SEED, lightgbm=lgb.__version__, features=old.FEATURES,
              cycle_features=prior.FEATURES, start_time=time.strftime('%Y-%m-%dT%H:%M:%S')),
         RES/'joint_frozen_run.json')
    checks = mathematical_checks()
    data, manifest, events, arrays = old.load_data()
    driver_cache = old.ForceCache(data)
    cache, fit_audit = {}, []
    keys = [tuple(y for y in YEARS if y not in excluded)
            for k in [2, 1] for excluded in itertools.combinations(YEARS, k)]
    assert len(keys) == 36
    template = prior.cycle_table(driver_cache.get(keys[0]))
    save(template[['season','treatment','y','n_obs']], RES/'target_cycle_means.csv')
    for ki, key in enumerate(keys):
        d = driver_cache.get(key)
        # These files allow TabPFN to reuse precisely the outer fold driver inputs.
        if len(key) == 7:
            held = next(y for y in YEARS if y not in key)
            daily_path, cycle_path = FEATURES_DIR/f'outer_{held}_daily.csv', FEATURES_DIR/f'outer_{held}_cycle.csv'
            if daily_path.exists() and cycle_path.exists():
                stored = pd.read_csv(daily_path)
                current = d.loc[d.season.isin(YEARS)]
                assert np.allclose(stored[old.FEATURES], current[old.FEATURES], atol=1e-10)
                assert np.allclose(stored.y, current.y, equal_nan=True)
            else:
                save(d.loc[d.season.isin(YEARS)], daily_path)
                save(prior.cycle_table(d), cycle_path)
        raw, agg = [], []
        for cfg in CANDIDATES:
            modelpath = MODELS/('_'.join(map(str,key))+'__'+cfg['id']+'.txt') if len(key)==7 else None
            pred, info = fit_candidate(d, key, cfg, modelpath)
            raw.append(pred)
            agg.append(prior.aggregate(d, pred))
            fit_audit.append(info)
        cache[key] = (np.column_stack(raw), np.column_stack(agg))
        np.savez_compressed(cache_dir/('_'.join(map(str,key))+'.npz'), daily=cache[key][0], cycle=cache[key][1])
        dump(fit_audit, RES/'joint_fit_audit.json')
        print(f'Joint: {ki+1}/36 training subsets, {len(fit_audit)}/504 fits; {time.perf_counter()-start:.1f}s', flush=True)
    daily_rows, cycle_rows, inner_rows, selections = [], [], [], []
    for held in YEARS:
        train = tuple(y for y in YEARS if y != held)
        d = driver_cache.get(train)
        train_idx = template.season.isin(train).to_numpy()
        inner = np.full((16,len(CANDIDATES)), np.nan)
        for val in train:
            key = tuple(y for y in train if y!=val)
            val_idx = template.season.eq(val).to_numpy()
            inner[val_idx] = cache[key][1][val_idx]
            for row_idx in np.flatnonzero(val_idx):
                rr = dict(outer_test=held, validation=val, inner_train=';'.join(map(str,key)),
                          treatment=template.treatment.iloc[row_idx], y=template.y.iloc[row_idx])
                rr.update({cfg['id']:float(inner[row_idx,j]) for j,cfg in enumerate(CANDIDATES)})
                inner_rows.append(rr)
        assert np.isfinite(inner[train_idx]).all() and np.isnan(inner[~train_idx]).all()
        scores = np.mean((inner[train_idx]-template.y.to_numpy()[train_idx,None])**2, axis=0)
        families = {
            'JointSelected': [i for i,c in enumerate(CANDIDATES) if c['family'].startswith('Joint')],
            'HuberDay': [i for i,c in enumerate(CANDIDATES) if c['family']=='HuberDay'],
            'CycleOnly': [i for i,c in enumerate(CANDIDATES) if c['family']=='CycleOnly'],
            'RawMSE': [i for i,c in enumerate(CANDIDATES) if c['family']=='RawMSE'],
            'AsinhMSE': [i for i,c in enumerate(CANDIDATES) if c['family']=='AsinhMSE'],
            **{family:[i for i,c in enumerate(CANDIDATES) if c['family']==family]
               for family in ['Joint01','Joint1','Joint10']}}
        chosen = {name:ids[int(np.argmin(scores[ids]))] for name,ids in families.items()}
        joint_cap = CANDIDATES[chosen['JointSelected']]['cap']
        chosen['HuberMatchedCapacity'] = next(i for i,c in enumerate(CANDIDATES)
                                             if c['family']=='HuberDay' and c['cap']==joint_cap)
        chosen['HuberBiasCorrected'] = chosen['HuberDay']
        offset = float(np.mean(template.y.to_numpy()[train_idx]-inner[train_idx,chosen['HuberDay']]))
        daily_all, _ = cache[train]
        testmask = (d.season.eq(held)).to_numpy()
        for name, ci in chosen.items():
            pred = daily_all[:,ci].copy()
            shift = offset if name=='HuberBiasCorrected' else 0.
            pred += shift
            rr = d.loc[testmask,['row_id','Date','season','treatment','y']].copy()
            rr['model'], rr['prediction'] = name, pred[testmask]
            daily_rows.append(rr)
            cr = prior.cycle_table(d)[['season','treatment','y','n_obs']]
            cr['prediction'] = prior.aggregate(d,pred)
            cr['model'] = name
            cycle_rows.append(cr.loc[cr.season.eq(held)])
            selections.append(dict(outer_test=held, model=name, candidate=CANDIDATES[ci],
                                   inner_MSE=float(scores[ci]), additive_correction=shift))
        print(f'Outer {held}: frozen selections completed', flush=True)
    # A finite-difference check covers math; this check covers the data path.
    train = tuple(y for y in YEARS if y != 2014)
    dd = driver_cache.get(train)
    changed = dd.copy()
    changed.loc[~changed.season.isin(train) & changed.y.notna(),'y'] += 10000
    cfg = next(c for c in CANDIDATES if c['id']=='Joint1_c0')
    p1, i1 = fit_candidate(dd,train,cfg,rounds=12)
    p2, i2 = fit_candidate(changed,train,cfg,rounds=12)
    checks['heldout_target_perturbation_max_prediction_change'] = float(np.max(np.abs(p1-p2)))
    assert np.array_equal(p1,p2) and i1['s_day']==i2['s_day'] and i1['s_cycle']==i2['s_cycle']
    checks.update(status='passed', candidates=len(CANDIDATES), unique_training_sets=len(cache),
                  experiment_fits=len(fit_audit), extra_technical_check_fits=2,
                  wall_seconds=time.perf_counter()-start)
    save(pd.concat(daily_rows,ignore_index=True), RES/'joint_daily_predictions.csv')
    save(pd.concat(cycle_rows,ignore_index=True), RES/'joint_cycle_predictions.csv')
    save(pd.DataFrame(inner_rows), RES/'joint_inner_predictions.csv')
    save(pd.DataFrame(driver_cache.audit), RES/'driver_imputation_audit.csv')
    dump(selections, RES/'joint_fold_choices.json')
    dump(checks, RES/'joint_verification.json')
    print(json.dumps(checks,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    prepare_outer_data() if '--prepare' in sys.argv else run()
