"""Frozen, exploratory EFELE pilot. Run with --prepare, --smoke, or --run.

No target interpolation or target-history features. Every fitted preprocessor is
rebuilt inside the corresponding training split. Scientific scope is documented
in 03_首轮实测_20261018/protocol.md; this is not a final manuscript benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import scipy
from scipy.signal import lfilter
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import lightgbm
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '03_首轮实测_20261018'
RAW = ROOT / '00_原始数据/EJPSoil-WP4 - N2O Flux Database/Field data - SOERE-PRO EFELE'
AUDIT = ROOT / '02_新论文设计/审查依据_20261018/data_audit'
SEED = 20261018
YEARS = list(range(2014, 2022))
MODELS = ['Calendar', 'GAM', 'LightGBM', 'FixedMemory', 'LearnedMemory', 'GatedMemory', 'MemoryLightGBM']
BOUNDARIES = [
    (2013, '2013-01-01', 'wheat', 'left_truncated'),
    (2014, '2013-09-06', 'maize', ''),
    (2015, '2014-10-23', 'wheat', ''),
    (2016, '2015-09-09', 'maize', ''),
    (2017, '2016-10-26', 'wheat', ''),
    (2018, '2017-09-12', 'maize', ''),
    (2019, '2018-10-30', 'wheat', ''),
    (2020, '2019-09-06', 'maize', ''),
    (2021, '2020-09-22', 'maize', ''),
    (2022, '2021-11-08', 'wheat', 'right_truncated'),
]
CROPS = ['BARE', 'WHEAT', 'MAIZE', 'MUSTARD']


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(name):
    d = pd.read_csv(RAW / name, skiprows=[1], encoding='latin1', na_values=-9999)
    d.rename(columns={d.columns[0]:'Date'}, inplace=True)
    d['Date'] = pd.to_datetime(d['Date'], format='%d.%m.%Y')
    return d.set_index('Date')


def load_data():
    flux = read_csv('EFELE_DailyN2Ofluxes_2013-2021.csv')
    met = read_csv('EFELE_DailyMetData_2013-2021.csv')
    assert flux.index.equals(met.index) and len(flux) == 3319
    assert (flux.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    m = pd.read_excel(RAW / 'EFELE_SoilCropManagement_2013-2021.xls', header=None)
    start = m.index[m[0].eq('Date')][0] + 1
    ev = []
    for i, row in m.loc[start:].iterrows():
        if not hasattr(row[0], 'year'):
            if ev:
                break
            continue
        for tr, ic, nc in [('PS', 2, 3), ('AN', 6, 7)]:
            if pd.isna(row[ic]):
                continue
            ident = str(row[ic])
            edate = pd.to_datetime(ident.split('_', 1)[1], format='%d/%m/%y')
            reason = 'Identifier date; confirmed against operations where available'
            if tr == 'AN' and str(edate.date()) == '2013-05-16':
                reason = 'Dose table and identifier May 16; operations May 15; main source retained'
            ev.append(dict(treatment=tr, date=edate, table_date=pd.Timestamp(row[0]).normalize(),
                           identifier=ident, excel_row=i+1, Ntotal=float(row[nc]),
                           Nmin=float(row[4] if tr == 'PS' else row[7]),
                           Norg=float(row[5] if tr == 'PS' else 0), reason=reason))
    events = pd.DataFrame(ev)
    assert len(events) == 22 and np.allclose(events.Ntotal, events.Nmin + events.Norg)
    opstart = m.index[m[0].eq('Land use')][0] + 1
    ops = []
    for i, r in m.loc[opstart:].iterrows():
        if hasattr(r[1], 'year'):
            ops.append(dict(date=pd.Timestamp(r[1]).normalize(), operation=str(r[2]), excel_row=i+1))
    ops = pd.DataFrame(ops).sort_values('date', kind='stable')
    daily = met.join(flux[['Tsoil{-5cm}', 'WFPS_measured']]).copy()
    daily['Tair'] = (daily.Ta_min + daily.Ta_max) / 2
    daily['RH'] = (daily.Rh_min + daily.Rh_max) / 2
    daily['doy_sin'] = np.sin(2*np.pi*daily.index.dayofyear/365.25)
    daily['doy_cos'] = np.cos(2*np.pi*daily.index.dayofyear/365.25)
    for window in [3, 7, 14, 30, 60, 90]:
        daily[f'rain_{window}'] = daily.Rainfall_sum.rolling(window, min_periods=1).sum()
    for window in [3, 7, 30]:
        daily[f'temp_{window}'] = daily.Tair.rolling(window, min_periods=1).mean()
    # Crop state follows only operations at or before a date; no harvest yields.
    daily['crop'] = 'WHEAT'
    daily['days_since_sowing'] = (daily.index - pd.Timestamp('2012-10-26')).days
    till = np.zeros(len(daily))
    for r in ops.itertuples():
        lo = r.operation.lower()
        c = None
        if 'seeding' in lo:
            c = 'WHEAT' if 'wheat' in lo else 'MAIZE' if 'maize' in lo else 'MUSTARD'
            mask = daily.index >= r.date
            daily.loc[mask, 'days_since_sowing'] = (daily.index[mask]-r.date).days
        elif 'harvest' in lo or 'mulching' in lo or 'stubble' in lo:
            c = 'BARE'
        if c is not None:
            daily.loc[daily.index >= r.date, 'crop'] = c
        if any(s in lo for s in ['till', 'plough', 'stubble', 'mulch']):
            if r.date in daily.index:
                till[daily.index.get_loc(r.date)] = 1
    daily['tillage_14'] = lfilter([1.], [1., -np.exp(-1/14)], till)
    daily['season'] = 0
    manifest = []
    for j, (year, begin, crop, trunc) in enumerate(BOUNDARIES):
        begin = pd.Timestamp(begin)
        end = pd.Timestamp(BOUNDARIES[j+1][1])-pd.Timedelta(days=1) if j+1 < len(BOUNDARIES) else daily.index[-1]
        mask = (daily.index >= begin) & (daily.index <= end)
        daily.loc[mask, 'season'] = year
        manifest.append(dict(season=year, main_crop=crop, start=str(begin.date()), end=str(end.date()),
                             days=int(mask.sum()), warmup_days=min(180, (begin-daily.index[0]).days),
                             PS_valid=int(flux.loc[mask, 'N2O_PigSlurry_mean'].notna().sum()),
                             AN_valid=int(flux.loc[mask, 'N2O_AmmoniumNitrate_mean'].notna().sum()),
                             included=year in YEARS, status=trunc or 'eligible',
                             boundary='Next main-crop sowing or cover sowing associated with following maize; dates from management Table 4'))
    frames = []
    for tr, target in [('PS', 'N2O_PigSlurry_mean'), ('AN', 'N2O_AmmoniumNitrate_mean')]:
        d = daily.copy()
        d['treatment'] = tr
        d['is_PS'] = float(tr == 'PS')
        d['y'] = flux[target]
        d['Imin'] = 0.
        d['Iorg'] = 0.
        last = np.full(len(d), 365.)
        lastdose = np.zeros(len(d))
        for e in events.loc[events.treatment.eq(tr)].itertuples():
            d.loc[e.date, 'Imin'] += e.Nmin
            d.loc[e.date, 'Iorg'] += e.Norg
            after = d.index >= e.date
            last[after] = (d.index[after]-e.date).days
            lastdose[after] = e.Ntotal
        d['DSFA'] = np.minimum(last, 365.)
        d['last_Ndose'] = lastdose
        for window in [7, 30, 60]:
            for col in ['Imin', 'Iorg']:
                d[f'{col}_sum{window}'] = d[col].rolling(window, min_periods=1).sum()
                # Recomputed below per episode, after zero initialization + warmup.
                d[f'{col}_exp{window}'] = 0.
        frames.append(d.reset_index())
    data = pd.concat(frames, ignore_index=True)
    # Precompute event ages and doses for the exact constant-rate recurrence.
    # Each episode has its OWN zero state 180 days before its start. No target-dependent initial state.
    ages, mins, orgs = [], [], []
    for _, row in data.iterrows():
        yr = int(row.season)
        begin = pd.Timestamp(next(x[1] for x in BOUNDARIES if x[0] == yr))
        warm = max(daily.index[0], begin-pd.Timedelta(days=180))
        ee = events.loc[events.treatment.eq(row.treatment) & events.date.ge(warm) & events.date.le(row.Date)]
        age = np.zeros(5)
        ni = np.zeros(5)
        no = np.zeros(5)
        assert len(ee) <= 5
        for n, e in enumerate(ee.itertuples()):
            age[n] = (row.Date-e.date).days
            ni[n], no[n] = e.Nmin, e.Norg
        ages.append(age)
        mins.append(ni)
        orgs.append(no)
    A, IM, IO = np.array(ages), np.array(mins), np.array(orgs)
    for window in [7, 30, 60]:
        data[f'Imin_exp{window}'] = (IM*np.exp(-A/window)).sum(1)
        data[f'Iorg_exp{window}'] = (IO*np.exp(-A/window)).sum(1)
    for c in CROPS[1:]:
        data[f'crop_{c}'] = (data.crop == c).astype(float)
    data['crop_age'] = np.minimum(data.days_since_sowing, 365.)/365.
    data['row_id'] = np.arange(len(data))
    return data, pd.DataFrame(manifest), events, (A, IM, IO)


WEATHER = ['Ta_min', 'Ta_max', 'Rh_min', 'Rh_max', 'WS_mean', 'Rg_sum', 'Rainfall_sum',
           'doy_sin', 'doy_cos'] + [f'rain_{x}' for x in [3, 7, 14, 30, 60, 90]] + [f'temp_{x}' for x in [3, 7, 30]]
BG = ['is_PS', 'crop_WHEAT', 'crop_MAIZE', 'crop_MUSTARD', 'doy_sin', 'doy_cos']
FEATURES = WEATHER + ['T', 'W', 'is_PS', 'crop_WHEAT', 'crop_MAIZE', 'crop_MUSTARD', 'crop_age',
                      'tillage_14', 'DSFA', 'last_Ndose', 'Imin', 'Iorg'] + [f'I{form}_{kind}{w}' for form in ['min', 'org'] for kind in ['sum', 'exp'] for w in [7, 30, 60]]


def weights(data, mask):
    sub = data.loc[mask]
    count = sub.groupby(['season', 'treatment']).y.transform('count').to_numpy()
    w = 1/np.maximum(count, 1)
    return w/w.mean()


class ForceCache:
    def __init__(self, data):
        self.data = data
        self.cache = {}
        self.audit = []

    def get(self, years):
        key = tuple(sorted(years))
        if key in self.cache:
            return self.cache[key]
        d = self.data.copy()
        unique = d.treatment.eq('PS')
        train = unique & d.season.isin(years)
        wx = d[WEATHER].to_numpy()
        for source, target in [('Tsoil{-5cm}', 'T'), ('WFPS_measured', 'W')]:
            valid = d[source].notna()
            if target == 'W':
                valid &= d[source].between(0, 100)
            fit = train & valid
            estimator = LGBMRegressor(n_estimators=160, num_leaves=7, min_child_samples=30,
                                     learning_rate=.04, reg_lambda=10, verbosity=-1, n_jobs=2,
                                     random_state=SEED, deterministic=True, force_col_wise=True)
            estimator.fit(wx[fit], d.loc[fit, source])
            estimated = estimator.predict(wx)
            if target == 'W':
                estimated = np.clip(estimated, 0, 100)
            d[target] = np.where(valid, d[source], estimated)
            d[f'{target}_imputed'] = ~valid
            for year in YEARS:
                test = unique & d.season.eq(year) & valid
                if year not in years and test.any():
                    self.audit.append(dict(train_seasons=';'.join(map(str,key)), test_season=year,
                                           variable=target, n=int(test.sum()),
                                           rmse=float(np.sqrt(mean_squared_error(d.loc[test, source], estimated[test]))),
                                           r2=float(r2_score(d.loc[test, source], estimated[test]))))
        assert np.isfinite(d[FEATURES].to_numpy()).all()
        self.cache[key] = d
        return d


def recurrence_numpy(A, IM, IO, tau_o=90., tau_m=30.):
    ao, am = np.exp(-1/tau_o), np.exp(-1/tau_m)
    opre = (IO*ao**A).sum(1)
    if abs(am-ao) < 1e-9:
        conv = (A+1)*ao**A
    else:
        conv = (am**(A+1)-ao**(A+1))/(am-ao)
    mpre = (IM*am**A + IO*(1-ao)*conv).sum(1)
    return opre, mpre, (1-am)*mpre


class MemoryNet(nn.Module):
    def __init__(self, nt, nw, nb, mode, start=0):
        super().__init__()
        self.mode = mode
        self.log_tau = nn.Parameter(torch.log(torch.tensor([90., 30.] if start == 0 else [30., 10.])), requires_grad=mode != 'FixedMemory')
        self.intercept = nn.Parameter(torch.tensor(-5.0))
        self.bt = nn.Parameter(torch.zeros(nt))
        self.bw = nn.Parameter(torch.zeros(nw))
        self.bd = nn.Parameter(torch.zeros(nw), requires_grad=mode == 'GatedMemory')
        self.bg = nn.Parameter(torch.zeros(nb))
        self.bg.data[0] = 3.
        self.gate_a = nn.Parameter(torch.tensor(-1.), requires_grad=mode == 'GatedMemory')
        self.gate_b = nn.Parameter(torch.tensor(0.), requires_grad=mode == 'GatedMemory')

    def states(self, A, IM, IO):
        # Bounded timescales are a prespecified numerical/finite-history pilot restriction.
        tau = torch.clamp(torch.exp(self.log_tau), 2., 365.)
        lo, lm = 1/tau[0], 1/tau[1]
        ao, am = torch.exp(-lo), torch.exp(-lm)
        delta = am-ao
        if abs(float(delta.detach())) < 1e-5:
            r = (lo+lm)/2
            h = (lm-lo)/2
            n = A+1
            conv = torch.exp(-r*A)*n*(1+(n*n-1)*h*h/6)
        else:
            conv = (torch.exp(-lm*(A+1))-torch.exp(-lo*(A+1)))/delta
        opre = (IO*torch.exp(-lo*A)).sum(1)
        mpre = (IM*torch.exp(-lm*A)+IO*(1-ao)*conv).sum(1)
        return opre, mpre, (1-am)*mpre

    def forward(self, A, IM, IO, T, W, B):
        opre, mpre, c = self.states(A, IM, IO)
        gate = torch.sigmoid(self.gate_a+F.softplus(self.gate_b)*torch.log1p(mpre/100.))
        eta = self.intercept+T@self.bt+W@self.bw
        if self.mode == 'GatedMemory':
            eta = eta+gate*(W@self.bd)
        pred = 1000*c*torch.sigmoid(eta)+B@self.bg
        return pred, (opre, mpre, c, gate)


class MemoryEstimator:
    def __init__(self, mode, config):
        self.mode, self.config = mode, config

    def transform(self, data, event_arrays):
        T = self.st.transform(data[['T']])-self.center_t
        W = self.sw.transform(data[['W']])-self.center_w
        B = np.column_stack([np.ones(len(data)), data[BG].to_numpy()])
        return tuple(torch.tensor(a) for a in (*event_arrays, T, W, B))

    def fit(self, data, arrays, mask):
        then = time.perf_counter()
        self.st = SplineTransformer(n_knots=4, degree=3, knots='uniform', extrapolation='constant', include_bias=False).fit(data.loc[mask, ['T']])
        self.sw = SplineTransformer(n_knots=4, degree=3, knots='uniform', extrapolation='constant', include_bias=False).fit(data.loc[mask, ['W']])
        self.center_t = self.st.transform(pd.DataFrame({'T':[data.loc[mask, 'T'].median()]}))
        self.center_w = self.sw.transform(pd.DataFrame({'W':[data.loc[mask, 'W'].median()]}))
        xall = self.transform(data, arrays)
        x = tuple(z[mask] for z in xall)
        y = torch.tensor(data.loc[mask, 'y'].to_numpy())
        w = torch.tensor(weights(data, mask))
        self.scale = max(1., float(np.median(np.abs(y.numpy()))))
        rawscale = max(5., float(np.sqrt(np.average(y.numpy()**2, weights=w.numpy()))))
        best, self.optim = np.inf, []
        for start in [0, 1]:
            net = MemoryNet(x[3].shape[1], x[4].shape[1], x[5].shape[1], self.mode, start)
            # Fixed-memory model has exactly the same two initial rate constants in both starts.
            if self.mode == 'FixedMemory':
                net.log_tau.data = torch.log(torch.tensor([90., 30.]))
                net.intercept.data.fill_(-5.0 if start == 0 else -4.0)
            optimizer = torch.optim.LBFGS([p for p in net.parameters() if p.requires_grad], lr=.8,
                                         max_iter=180, max_eval=240, tolerance_grad=1e-7,
                                         tolerance_change=1e-9, line_search_fn='strong_wolfe')
            calls = [0]
            def objective(backward=True):
                optimizer.zero_grad()
                pred, _ = net(*x)
                transformed = F.huber_loss(torch.asinh(pred/self.scale), torch.asinh(y/self.scale), reduction='none', delta=1.)
                raw = ((pred-y)/rawscale)**2
                loss = (w*(transformed+self.config['raw_weight']*raw)).mean()
                smooth = sum(torch.diff(v, n=2).pow(2).mean() for v in [net.bt, net.bw])
                penalty = .001*(net.bt.pow(2).mean()+net.bw.pow(2).mean())
                if self.mode == 'GatedMemory':
                    smooth = smooth + torch.diff(net.bd, n=2).pow(2).mean()
                    penalty = penalty + self.config['reg']*net.bd.pow(2).mean()
                penalty = penalty + .0005*net.bg[1:].pow(2).mean()
                loss = loss+self.config['reg']*smooth+penalty
                if backward:
                    loss.backward()
                    calls[0] += 1
                return loss
            optimizer.step(objective)
            value = float(objective(False).detach())
            finite = np.isfinite(value) and all(torch.isfinite(p).all() for p in net.parameters())
            self.optim.append(dict(start=start, objective=value, closure_calls=calls[0], finite=bool(finite)))
            if finite and value < best:
                best, self.net = value, net
        if not np.isfinite(best):
            raise RuntimeError('Both memory starts failed')
        self.fit_seconds = time.perf_counter()-then
        self.nparams = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.train_prediction, _ = self.predict(data, arrays)
        return self

    def predict(self, data, arrays):
        with torch.no_grad():
            p, states = self.net(*self.transform(data, arrays))
        return p.numpy(), np.column_stack([s.numpy() for s in states])

    def info(self):
        return dict(timescales=np.clip(np.exp(self.net.log_tau.detach().numpy()), 2, 365).tolist(),
                    gate_intercept=float(self.net.gate_a.detach()),
                    gate_slope=float(F.softplus(self.net.gate_b).detach()),
                    parameter_count=self.nparams, fit_seconds=self.fit_seconds, optimizer=self.optim,
                    parameters={k:v.detach().numpy().tolist() for k,v in self.net.state_dict().items()})


def candidates(model):
    if model == 'Calendar':
        return [dict(alpha=10., transform='raw')]
    if model == 'GAM':
        return [dict(alpha=a, transform=t) for a in [10., 100.] for t in ['raw', 'asinh']]
    if model in ['LightGBM', 'MemoryLightGBM']:
        return [dict(leaves=l, trees=n, transform=t) for l,n in [(7,200),(15,300)] for t in ['raw','asinh']]
    return [dict(reg=r, raw_weight=w) for r in [.01,.1] for w in [0., .25]]


def fit_model(model, config, data, arrays, train_years):
    mask = (data.season.isin(train_years)&data.y.notna()).to_numpy()
    tic = time.perf_counter()
    if model in ['FixedMemory', 'LearnedMemory', 'GatedMemory']:
        e = MemoryEstimator(model, config).fit(data, arrays, mask)
        pred, states = e.predict(data, arrays)
        info = e.info()
        info['_estimator'] = e
        return pred, states, info
    x = data[FEATURES].to_numpy()
    w = weights(data, mask)
    yy = data.loc[mask, 'y'].to_numpy()
    scale = max(1., float(np.median(np.abs(yy))))
    y = np.arcsinh(yy/scale) if config['transform']=='asinh' else yy
    states = None
    info = {}
    if model == 'MemoryLightGBM':
        # Prespecified core configuration; learned only on the current fit subset.
        core = MemoryEstimator('LearnedMemory', dict(reg=.1,raw_weight=.25)).fit(data, arrays, mask)
        cp, states = core.predict(data, arrays)
        x = np.column_stack([x,states[:,:3],cp])
        info['core'] = core.info()
    if model == 'Calendar':
        x = data[BG+['crop_age']].to_numpy()
        e = Ridge(alpha=config['alpha'])
    elif model == 'GAM':
        selected = ['T','W','Tair','Rainfall_sum','rain_7','rain_30','DSFA','Imin_exp30','Iorg_exp60','crop_age','tillage_14']
        spline = SplineTransformer(n_knots=4,degree=3,extrapolation='constant',include_bias=False)
        spline.fit(data.loc[mask,selected])
        x = np.column_stack([spline.transform(data[selected]), data[BG].to_numpy()])
        e = Ridge(alpha=config['alpha'])
    else:
        e = LGBMRegressor(n_estimators=config['trees'],num_leaves=config['leaves'],learning_rate=.04,
                          min_child_samples=25,reg_lambda=10,verbosity=-1,n_jobs=2,
                          random_state=SEED,deterministic=True,force_col_wise=True)
    e.fit(x[mask], y, sample_weight=w)
    pred = e.predict(x)
    if config['transform'] == 'asinh':
        pred = scale*np.sinh(np.clip(pred, -10,10))
    info.update(fit_seconds=time.perf_counter()-tic, target_scale=scale)
    assert np.isfinite(pred).all()
    return pred, states, info


def cycle_rmse(data, pred, years):
    scores = []
    for yr in years:
        vals=[]
        for tr in ['PS','AN']:
            m = data.season.eq(yr)&data.treatment.eq(tr)&data.y.notna()
            if m.any():
                vals.append(np.sqrt(mean_squared_error(data.loc[m,'y'],pred[m])))
        scores.append(np.mean(vals))
    return float(np.mean(scores))


def inner_splits(train_years, protocol):
    if protocol=='forward':
        vals = train_years[-2:]
        return [(tuple(y for y in train_years if y < v),v) for v in vals]
    wheat = [y for y in train_years if y in [2015,2017,2019]]
    maize = [y for y in train_years if y not in wheat]
    vals = [wheat[-1],maize[-1]]
    return [(tuple(y for y in train_years if y!=v),v) for v in vals]


def select_fit(model, data, arrays, cache, years, protocol, fold_dir):
    logs=[]
    grid=candidates(model)
    scores=[]
    for ci,cfg in enumerate(grid):
        vv=[]
        for tr, val in inner_splits(years,protocol):
            assert val not in tr and len(tr)>0
            dd=cache.get(tr)
            p,_,info=fit_model(model,cfg,dd,arrays,tr)
            score=cycle_rmse(dd,p,[val])
            vv.append(score)
            logs.append(dict(model=model,candidate=ci,config=json.dumps(cfg),inner_train=';'.join(map(str,tr)),
                             inner_validation=val,rmse=score,fit_seconds=info['fit_seconds']))
        scores.append(float(np.mean(vv)))
    chosen=int(np.argmin(scores))
    dd=cache.get(years)
    p,states,info=fit_model(model,grid[chosen],dd,arrays,years)
    if '_estimator' in info:
        core=info.pop('_estimator')
        import joblib
        joblib.dump(dict(temperature_spline=core.st,moisture_spline=core.sw,
                         temperature_center=core.center_t,moisture_center=core.center_w,
                         scale=core.scale,mode=core.mode,config=core.config,
                         state_dict={k:v.detach().numpy() for k,v in core.net.state_dict().items()}),
                    fold_dir/f'{model}_estimator.joblib')
    info.update(selected_config=grid[chosen],inner_candidate_scores=scores,selected_inner_score=scores[chosen],
                train_macro_rmse=cycle_rmse(dd,p,years))
    pd.DataFrame(logs).to_csv(fold_dir/f'{model}_inner_search.csv',index=False,encoding='utf-8-sig')
    dump(info,fold_dir/f'{model}_fit.json')
    return p,states,info


def prepare(data,manifest,events):
    OUT.mkdir(exist_ok=True)
    manifest.to_csv(OUT/'season_manifest.csv',index=False,encoding='utf-8-sig')
    events.to_csv(OUT/'fertilizer_events_resolved.csv',index=False,encoding='utf-8-sig')
    data.to_csv(OUT/'daily_input_audit.csv',index=False,encoding='utf-8-sig')
    sources = [dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(RAW.iterdir()) if p.suffix in ['.csv','.xls']]
    dump(dict(seed=SEED,python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
              scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,lightgbm=lightgbm.__version__,
              threads=2,cuda=False,sources=sources,script_sha256=sha(__file__)),OUT/'run_environment.json')
    config = dict(models={m:candidates(m) for m in MODELS},loco_tests=YEARS,forward_tests=[2018,2019,2020,2021],
                  warmup=180,init_states=[0,0],fold_definition='Management cycles; both treatments held out together',
                  seeds=[SEED],inner_validation='Two predetermined whole cycles; strictly earlier training in forward splits',
                  primary='Mean across cycles of mean treatment RMSE in original g N2O-N ha-1 day-1',
                  target='Recorded daily flux; no imputed y; signed values retained',
                  note='Exploratory development. All existing seasons already inspected descriptively; no blind test claim.')
    dump(config,OUT/'frozen_config.json')
    dump({'script':sha(__file__),'config':sha(OUT/'frozen_config.json'),'manifest':sha(OUT/'season_manifest.csv')},OUT/'frozen_hashes.json')


def self_checks(data,manifest,events,arrays):
    rng=np.random.default_rng(SEED)
    age=np.arange(80,dtype=float)[:,None]
    ni=np.full_like(age,70.)
    no=np.full_like(age,30.)
    errors=[]
    for to,tm in [(90.,30.),(30.,90.),(20.,20.),(365.,2.)]:
        op,mp,c=recurrence_numpy(age,ni,no,to,tm)
        o,m=0.,0.
        expected=[]
        for j in range(80):
            o+=30. if j==0 else 0.
            m+=70. if j==0 else 0.
            r=o*(-np.expm1(-1/to));o-=r;m+=r
            loss=m*(-np.expm1(-1/tm));m-=loss
            expected.append(loss)
            assert abs(o+m+sum(expected)-100.)<1e-9
        err=float(np.max(np.abs(c-expected)))
        errors.append(err)
        assert err<1e-8
        net=MemoryNet(2,2,2,'LearnedMemory')
        net.log_tau.data=torch.log(torch.tensor([to,tm]))
        z=net.states(torch.tensor(age),torch.tensor(ni),torch.tensor(no))[2]
        assert np.max(np.abs(z.detach().numpy()-c))<1e-7
        z.sum().backward()
        assert torch.isfinite(net.log_tau.grad).all()
    assert len(data.loc[data.season.isin(YEARS)&data.y.notna()])>3000
    assert all(manifest.loc[manifest.included,'warmup_days']==180)
    assert not any('y'==s or 'stdev' in s or 'NDVI' in s or 'gap-filled' in s for s in FEATURES)
    assert events.loc[(events.treatment=='AN')&(events.identifier=='AN_26/3/19'),'date'].iloc[0]==pd.Timestamp('2019-03-26')
    for protocol,tests in [('loco',YEARS),('forward',[2018,2019,2020,2021])]:
        for test in tests:
            train=[y for y in YEARS if (y!=test if protocol=='loco' else y<test)]
            for inner,v in inner_splits(train,protocol):
                assert test not in inner and v!=test and v not in inner
                if protocol=='forward': assert max(inner)<v<test
    dump(dict(status='PASS',recurrence_max_errors=errors,observed_targets=int((data.season.isin(YEARS)&data.y.notna()).sum()),
              checks=['Exact discrete state identity','Tensor/numpy/iterative recurrence agreement including equal rates',
                      'Finite gradients','Both treatments share outer cycle','Forward inner order','180-day warmup',
                      'No target / SD / NDVI / supplied gap-filled inputs','2019 AN source date']),OUT/'self_checks.json')


def run(data,manifest,events,arrays, smoke=False):
    cache=ForceCache(data)
    if smoke:
        dd=cache.get([2014,2015,2016])
        then=time.perf_counter()
        for model in ['FixedMemory','LearnedMemory','GatedMemory','LightGBM','GAM']:
            p,_,info=fit_model(model,candidates(model)[0],dd,arrays,[2014,2015,2016])
            print(model,'fit_seconds',round(info['fit_seconds'],2),'train_RMSE',round(cycle_rmse(dd,p,[2014,2015,2016]),3),flush=True)
        print('SMOKE_SECONDS',time.perf_counter()-then,flush=True)
        return
    for protocol,tests in [('loco',YEARS),('forward',[2018,2019,2020,2021])]:
        for test in tests:
            years=tuple(y for y in YEARS if (y!=test if protocol=='loco' else y<test))
            folder=OUT/'folds'/f'{protocol}_{test}'
            folder.mkdir(exist_ok=True,parents=True)
            testmask=data.season.eq(test)
            for model in MODELS:
                target=folder/f'{model}_predictions.csv'
                if target.exists():
                    print('RESUME',protocol,test,model,flush=True)
                    continue
                t=time.perf_counter()
                p,states,info=select_fit(model,data,arrays,cache,years,protocol,folder)
                result=data.loc[testmask,['row_id','Date','season','treatment','y']].copy()
                result['prediction']=p[testmask]
                result['protocol']=protocol
                result['model']=model
                for tr in ['PS','AN']:
                    yy=data.loc[data.season.isin(years)&data.treatment.eq(tr)&data.y.notna(),'y']
                    result.loc[result.treatment.eq(tr),'train_Q90']=float(yy.quantile(.9))
                if states is not None:
                    for i,n in enumerate(['Opre','Mpre','C','gate']):
                        result[n]=states[testmask,i]
                dd=cache.get(years)
                result['T']=dd.loc[testmask,'T'].to_numpy()
                result['W']=dd.loc[testmask,'W'].to_numpy()
                result['T_imputed']=dd.loc[testmask,'T_imputed'].to_numpy()
                result['W_imputed']=dd.loc[testmask,'W_imputed'].to_numpy()
                result.to_csv(target,index=False,encoding='utf-8-sig')
                print('DONE',protocol,test,model,'test_macro_RMSE',round(cycle_rmse(data,p,[test]),3),'sec',round(time.perf_counter()-t,1),flush=True)
                pd.DataFrame(cache.audit).to_csv(OUT/'soil_driver_holdout_metrics.csv',index=False,encoding='utf-8-sig')
            with (OUT/'evaluation_ledger.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(protocol=protocol,test_cycle=test,train_cycles=years,models=MODELS,
                                             status='evaluated',timestamp=pd.Timestamp.now(tz='Pacific/Auckland').isoformat()))+'\n')
    print('ALL_FOLDS_COMPLETE',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    data,manifest,events,arrays=load_data()
    if args.prepare: prepare(data,manifest,events)
    self_checks(data,manifest,events,arrays)
    if args.smoke or args.run: run(data,manifest,events,arrays,smoke=args.smoke)
