"""Crossfit residual heads with exact observed-support mean reconciliation."""
import copy
import time
from scipy.optimize import lsq_linear
from common import *

NAMES = ['Blend50','BlendCycle','MeanReconciled','BiasReconciled','AffineReconciled',
         'DecoupledNoPulse','DecoupledPulse','DecoupledUnconstrained','TiedPulse',
         'PulseShapeOnly','PulseLevelOnly','DecoupledPulseWeak','DecoupledPulseStrong',
         'DecoupledNoEnvironment']

def solve(X, y, w, lower, lam):
    X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
    assert np.isclose(w.sum(), 1.) or np.isclose(w.sum(), 2.)
    p = X.shape[1]
    aug = np.vstack([np.sqrt(w[:,None])*X, np.sqrt(lam)*np.eye(p)])
    target = np.r_[np.sqrt(w)*y, np.zeros(p)]
    fit = lsq_linear(aug, target, bounds=(np.array(lower), np.full(p,np.inf)),
                     tol=1e-12, max_iter=1000)
    assert fit.success, fit.message
    beta = fit.x
    grad = aug.T@(aug@beta-target)
    at_lower = np.isfinite(lower)&(beta-np.array(lower)<1e-7)
    projected = np.where(at_lower, np.minimum(grad,0), grad)
    kkt = float(np.max(np.abs(projected)))
    loss0 = float(np.sum(w*y*y)/2)
    loss = float(np.sum(w*(X@beta-y)**2)/2 + lam*np.sum(beta*beta)/2)
    assert kkt < 1e-6 and loss <= loss0+1e-9
    return beta, dict(loss_initial=loss0,loss_final=loss,kkt_max=kkt,
                      n_parameters=p,active_lower=int(at_lower.sum()),iterations=fit.nit)

def state(d, arrays, environment=True):
    obs = d.y.notna().to_numpy()
    g = groups(d)
    B = kernel_basis(d,arrays,environment)
    L, T = d.p_lgb.to_numpy(), d.p_tab.to_numpy()
    m = mean_broadcast(L,g,obs)
    mt = mean_broadcast(T,g,obs)
    mb = mean_broadcast(B,g,obs)
    first = np.array([np.flatnonzero((g==k)&obs)[0] for k in np.unique(g)])
    n = np.array([(g[obs]==k).sum() for k in g[obs]],dtype=float)
    w = 1/(len(first)*n)
    return dict(obs=obs,g=g,B=B,m=m,ct=T-mt,mb=mb,cb=B-mb,first=first,w=w)

def learn(d, arrays, lam=1., pulse=True, constrained=True, environment=True,
          level_mode='affine', shape=True):
    st = state(d,arrays,environment)
    obs, g, ii = st['obs'],st['g'],st['first']
    y = d.y.to_numpy()
    my = mean_broadcast(y,g,obs)
    cy = y-my
    sl = max(1.,float(np.std(my[ii])))
    sd = max(1.,float(np.sqrt(np.sum(st['w']*cy[obs]**2))))
    level_cols = [np.ones(len(ii))]
    lower_l = [-np.inf]
    if level_mode=='affine':
        level_cols.append(st['m'][ii]/sl)
        lower_l.append(-1.)
    shape_cols = [st['ct'][obs]/sd]
    lower_s = [-1.]
    norm_l, norm_s = np.array([]),np.array([])
    if pulse:
        norm_l = np.maximum(1e-8,np.sqrt(np.mean(st['mb'][ii]**2,axis=0)))*np.sqrt(10)
        norm_s = np.maximum(1e-8,np.sqrt(np.sum(st['w'][:,None]*st['cb'][obs]**2,axis=0)))*np.sqrt(10)
        level_cols.extend((st['mb'][ii]/norm_l).T)
        shape_cols.extend((st['cb'][obs]/norm_s).T)
        lower_l.extend([0. if constrained else -np.inf]*10)
        lower_s.extend([0. if constrained else -np.inf]*10)
    Xl, Xs = np.column_stack(level_cols),np.column_stack(shape_cols)
    bl, info_l = solve(Xl,(my[ii]-st['m'][ii])/sl,np.full(len(ii),1/len(ii)),lower_l,lam)
    if shape:
        bs, info_s = solve(Xs,(cy[obs]-st['ct'][obs])/sd,st['w'],lower_s,lam)
    else:
        bs=np.zeros(Xs.shape[1]); info_s=dict(fixed_shape=True)
    result = dict(kind='decoupled',lam=lam,pulse=pulse,constrained=constrained,
                  environment=environment,level_mode=level_mode,sl=sl,sd=sd,
                  norm_l=norm_l.tolist(),norm_s=norm_s.tolist(),
                  beta_l=bl.tolist(),beta_s=bs.tolist(),level_audit=info_l,shape_audit=info_s,
                  train_cycles=sorted(d.season.unique().tolist()),train_groups=len(ii),
                  train_observations=int(obs.sum()))
    if pulse and constrained:
        assert min(bl[-10:].min(),bs[-10:].min()) >= -1e-10
    return result

def predict(model,d,arrays):
    st=state(d,arrays,model['environment'])
    m=st['m']
    if model['kind']=='tied':
        b=np.array(model['beta'])
        pulse=model['sl']*(st['B']/np.array(model['norm']))@b[3:]
        p=m+model['sl']*b[0]+b[1]*m+(1+b[2])*st['ct']+pulse
        level=m+model['sl']*b[0]+b[1]*m+mean_broadcast(pulse,st['g'],st['obs'])
    else:
        xl=[np.ones(len(d))]
        if model['level_mode']=='affine': xl.append(m/model['sl'])
        xs=[st['ct']/model['sd']]
        if model['pulse']:
            xl.extend((st['mb']/np.array(model['norm_l'])).T)
            xs.extend((st['cb']/np.array(model['norm_s'])).T)
        dl=model['sl']*np.column_stack(xl)@np.array(model['beta_l'])
        ds=model['sd']*np.column_stack(xs)@np.array(model['beta_s'])
        if model.get('zero_level'): dl[:]=0
        if model.get('zero_shape'): ds[:]=0
        level=m+dl
        p=level+st['ct']+ds
    delta=mean_broadcast(p,st['g'],st['obs'])-level
    assert np.max(np.abs(delta))<1e-9
    assert np.isfinite(p).all()
    return p,level

def learn_tied(d,arrays):
    st=state(d,arrays,True)
    obs,g,ii=st['obs'],st['g'],st['first']
    y=d.y.to_numpy(); my=mean_broadcast(y,g,obs); cy=y-my
    sl=max(1.,float(np.std(my[ii])))
    sd=max(1.,float(np.sqrt(np.sum(st['w']*cy[obs]**2))))
    norm=np.maximum(1e-8,np.sqrt(np.sum(st['w'][:,None]*st['B'][obs]**2,axis=0)))*np.sqrt(10)
    xl=np.column_stack([np.ones(len(ii)),st['m'][ii]/sl,np.zeros(len(ii)),st['mb'][ii]/norm])
    xs=np.column_stack([np.zeros(obs.sum()),np.zeros(obs.sum()),st['ct'][obs]/sd,
                        sl/sd*st['cb'][obs]/norm])
    X=np.vstack([xl,xs])
    target=np.r_[(my[ii]-st['m'][ii])/sl,(cy[obs]-st['ct'][obs])/sd]
    w=np.r_[np.full(len(ii),1/len(ii)),st['w']]
    beta,info=solve(X,target,w,[-np.inf,-1.,-1.]+[0.]*10,1.)
    return dict(kind='tied',lam=1.,environment=True,sl=sl,sd=sd,norm=norm.tolist(),
                beta=beta.tolist(),audit=info,train_cycles=sorted(d.season.unique().tolist()))

def blend_weight(d):
    obs=d.y.notna().to_numpy(); g=groups(d)
    ii=np.array([np.flatnonzero((g==k)&obs)[0] for k in np.unique(g)])
    lc=mean_broadcast(d.p_lgb.to_numpy(),g,obs)[ii]
    tc=mean_broadcast(d.p_tab.to_numpy(),g,obs)[ii]
    yc=mean_broadcast(d.y.to_numpy(),g,obs)[ii]
    s2=max(1.,float(np.var(yc))); delta=lc-tc
    w=float(np.clip((np.mean(delta*(yc-tc))/s2+.05)/(np.mean(delta*delta)/s2+.1),0,1))
    return w

def assemble(held,outer_tab):
    train=tuple(y for y in YEARS if y!=held)
    parts=[]
    for val in train:
        key=tuple(y for y in train if y!=val)
        d=read(FOLDS/(key_name(key)+'.csv'))
        d=d.loc[d.season.eq(val)].copy()
        tab=read(CACHE/('tab_'+key_name(key)+'.csv'))[['row_id','p_tab']]
        d=d.merge(tab,on='row_id',how='left',validate='one_to_one')
        assert d.p_tab.notna().all() and held not in key and val not in key
        d['base_train_cycles']=';'.join(map(str,key))
        parts.append(d)
    tr=pd.concat(parts,ignore_index=True)
    te=read(FOLDS/(key_name(train)+'.csv'))
    te=te.loc[te.season.eq(held)].copy()
    te=te.merge(outer_tab[['row_id','p_tab']],on='row_id',how='left',validate='one_to_one')
    assert te.p_tab.notna().all()
    return tr,te

def structural_checks():
    # Validate the uncentred kernel; mean reconciliation is retrospective, not causal.
    age=np.arange(-10,1500,dtype=float)
    audits=[]
    for tau in TAUS:
        k=np.where(age>=0,(-np.expm1(-1/tau))*np.exp(-np.maximum(age,0)/tau),0)
        assert (k[age<0]==0).all() and (k>=0).all()
        assert np.allclose(2*k,k+k)
        mass=float(k.sum())
        assert abs(mass-1)<1e-6
        audits.append(dict(tau=tau,finite_1500_day_mass=mass,min_response=float(k.min())))
    return dict(kernels=audits,status='passed')

def main():
    if (RES/'new_daily_predictions.csv').exists():
        raise RuntimeError('Frozen predictions exist; do not overwrite.')
    start=time.perf_counter()
    assert json.loads((RES/'base_completion.json').read_text())['status']=='completed'
    assert json.loads((RES/'tabpfn_crossfit_completion.json').read_text())['status']=='completed'
    raw=np.load(CACHE/'event_arrays.npz'); arrays=(raw['A'],raw['IM'],raw['IO'])
    tab=read(PREV/'results/tabpfn_daily_predictions.csv')
    tab=tab.query("model=='TabPFN_DailyRaw'").rename(columns={'prediction':'p_tab'})
    checks=structural_checks()
    checks['prediction_checks']=[]
    records=[]; cycle_rows=[]; model_audit=[]; head_inputs=[]
    source_paths=[Path(__file__),OUT/'scripts/common.py',OUT/'protocol.md',
                  RES/'base_completion.json',RES/'tabpfn_crossfit_completion.json']
    dump(dict(source_hashes={str(p):sha(p) for p in source_paths},models=NAMES),RES/'head_frozen_run.json')
    for held in YEARS:
        tr,te=assemble(held,tab)
        # Keep model-input evidence, without copying full source data again.
        head_inputs.append(tr[['row_id','Date','season','treatment','y','p_lgb','p_tab','base_train_cycles']].assign(outer_test=held))
        main_model=learn(tr,arrays)
        models={
            'BiasReconciled':learn(tr,arrays,pulse=False,shape=False,level_mode='bias'),
            'AffineReconciled':learn(tr,arrays,pulse=False,shape=False),
            'DecoupledNoPulse':learn(tr,arrays,pulse=False),
            'DecoupledPulse':main_model,
            'DecoupledUnconstrained':learn(tr,arrays,constrained=False),
            'TiedPulse':learn_tied(tr,arrays),
            'PulseShapeOnly':dict(main_model,zero_level=True),
            'PulseLevelOnly':dict(main_model,zero_shape=True),
            'DecoupledPulseWeak':learn(tr,arrays,lam=.1),
            'DecoupledPulseStrong':learn(tr,arrays,lam=10.),
            'DecoupledNoEnvironment':learn(tr,arrays,environment=False)}
        w=blend_weight(tr)
        ob=te.y.notna().to_numpy(); gg=groups(te)
        l,t=te.p_lgb.to_numpy(),te.p_tab.to_numpy()
        mr=t+mean_broadcast(l-t,gg,ob)
        predictions={'Blend50':(.5*(l+t),None),
                     'BlendCycle':(w*l+(1-w)*t,None),
                     'MeanReconciled':(mr,mean_broadcast(l,gg,ob))}
        for name,model in models.items():
            path=MODELS/f'{held}_{name}.json'; dump(model,path)
            p,lev=predict(model,te,arrays)
            restored=json.loads(path.read_text(encoding='utf-8'))
            rp,rl=predict(restored,te,arrays)
            assert np.array_equal(p,rp) and np.array_equal(lev,rl)
            predictions[name]=(p,lev)
            model_audit.append(dict(held=held,model=name,parameters=model))
        # Test values may affect scoring, never prediction or fitted coefficients.
        changed=te.copy(); changed.loc[changed.y.notna(),'y']+=10000
        pc,_=predict(main_model,changed,arrays)
        assert np.array_equal(pc,predictions['DecoupledPulse'][0])
        all_data=pd.concat([tr,te],ignore_index=True)
        changed_all=all_data.copy()
        changed_all.loc[changed_all.season.eq(held)&changed_all.y.notna(),'y']-=9999
        refit=learn(changed_all.loc[changed_all.season.ne(held)],arrays)
        assert refit==main_model
        checks['prediction_checks'].append(dict(held=held,target_perturbation_max_change=0.,
            refit_training_target_isolation=True,serialization_exact=True,
            mean_consistency_error=float(np.max(np.abs(mean_broadcast(predictions['DecoupledPulse'][0],gg,ob)-predictions['DecoupledPulse'][1])))))
        model_audit.append(dict(held=held,model='BlendCycle',weight_lightgbm=w,weight_tabpfn=1-w))
        assert set(predictions)==set(NAMES)
        for name,(p,lev) in predictions.items():
            rr=te[['row_id','Date','season','treatment','y']].copy()
            rr['prediction']=p; rr['model']=name
            rr['level_prediction']=lev if lev is not None else mean_broadcast(p,gg,ob)
            records.append(rr)
            cr=rr.loc[rr.y.notna()].groupby(['season','treatment'],as_index=False).agg(
                y=('y','mean'),prediction=('prediction','mean'),n_obs=('y','size'))
            cr['model']=name; cycle_rows.append(cr)
        print(f'Outer {held}: all 14 fixed variants fitted and checked',flush=True)
    save(pd.concat(records,ignore_index=True),RES/'new_daily_predictions.csv')
    save(pd.concat(cycle_rows,ignore_index=True),RES/'new_cycle_predictions.csv')
    save(pd.concat(head_inputs,ignore_index=True),RES/'internal_head_inputs.csv')
    dump(model_audit,RES/'head_fit_audit.json')
    checks.update(models=len(NAMES),folds=len(YEARS),seconds=time.perf_counter()-start)
    dump(checks,RES/'head_verification.json')
    print(json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
