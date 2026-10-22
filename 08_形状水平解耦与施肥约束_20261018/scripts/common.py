"""Shared paths and existing, audited data loader. Import has no experiment side effects."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
RES = OUT/'results'
CACHE = OUT/'cache'
FOLDS = OUT/'fold_data'
MODELS = OUT/'models'
PREV = ROOT/'07_周期损失与TabPFN_20261018'
YEARS = tuple(range(2014, 2022))
TAUS = (3., 7., 14., 30., 90.)
SEED = 20261018
sys.stdout.reconfigure(encoding='utf-8')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def save(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')

def read(path):
    return pd.read_csv(path, float_precision='round_trip')

def key_name(key):
    return '_'.join(map(str, key))

def kernel_basis(d, arrays, environment=True):
    A, IM, IO = arrays
    ids = d.row_id.to_numpy(dtype=int)
    raw = np.column_stack([(D[ids]*np.exp(-A[ids]/tau)).sum(axis=1)*(-np.expm1(-1/tau))
                           for D in [IM, IO] for tau in TAUS])
    if environment:
        g = 2.**np.clip((d['T'].to_numpy()-20.)/10., -3., 3.)
        g *= np.clip(d.W.to_numpy()/60., 0., 100./60.)
        raw *= g[:, None]
    assert np.isfinite(raw).all() and (raw >= 0).all()
    return raw

def groups(d):
    return (2*(d.season.to_numpy(dtype=int)-2014) + d.treatment.eq('PS').to_numpy(dtype=int))

def mean_broadcast(x, g, obs):
    x = np.asarray(x, dtype=float)
    one = x.ndim == 1
    if one: x = x[:, None]
    avg = np.empty_like(x)
    for k in np.unique(g):
        take = g == k
        assert (take & obs).any()
        avg[take] = x[take & obs].mean(axis=0)
    return avg[:, 0] if one else avg

