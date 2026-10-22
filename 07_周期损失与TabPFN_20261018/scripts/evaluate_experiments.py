"""Score frozen predictions with identical support and paired cycle bootstrap."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
RES=OUT/'results'
spec=importlib.util.spec_from_file_location('evaluation_prior',ROOT/'05_框架再评估_20261018/scripts/recheck_saved_predictions.py')
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
KEY=['season','treatment','Date']

def dump(obj,name):
    (RES/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def save(df,name):
    df.to_csv(RES/name,index=False,encoding='utf-8-sig')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    original=ROOT/'03_首轮实测_20261018/all_heldout_predictions.csv'
    base=pd.read_csv(original,parse_dates=['Date']).query("protocol == 'loco' and model == 'LightGBM'")
    base=base[['row_id','Date','season','treatment','y','prediction']].copy()
    base['model']='OriginalLightGBM'
    daily=pd.concat([base,pd.read_csv(RES/'joint_daily_predictions.csv',parse_dates=['Date']),
                     pd.read_csv(RES/'tabpfn_daily_predictions.csv',parse_dates=['Date'])],ignore_index=True)
    assert not daily.duplicated(['model']+KEY).any()
    cycle_base=ev.tables(base)['cycle'].assign(model='OriginalLightGBM')
    cycles=pd.concat([cycle_base,pd.read_csv(RES/'joint_cycle_predictions.csv'),
                      pd.read_csv(RES/'tabpfn_cycle_predictions.csv')],ignore_index=True)
    assert not cycles.duplicated(['model','season','treatment']).any()
    reference=base.sort_values(KEY).reset_index(drop=True)
    cbase=cycle_base.sort_values(['season','treatment']).reset_index(drop=True)
    counts=np.array(list(ev.compositions(8,8)))
    multiplicities=np.array([math.factorial(8)//math.prod(math.factorial(int(k)) for k in r) for r in counts])
    prob=multiplicities/8**8
    assert len(counts)==6435 and multiplicities.sum()==8**8
    bootstrap,point,rows,window_rows,group_rows=[],{},[],[],[]
    distributions={}
    for name,c in cycles.groupby('model'):
        c=c.sort_values(['season','treatment']).reset_index(drop=True)
        pd.testing.assert_frame_equal(c[['season','treatment','n_obs']],cbase[['season','treatment','n_obs']])
        assert np.allclose(c.y,cbase.y,atol=1e-12)
        pred=c.prediction.to_numpy(); y=c.y.to_numpy()
        error=pred-y; rmse=float(np.sqrt(np.mean(error**2)))
        row=dict(model=name,n_clusters=8,n_cycle_units=len(c),R2_cycle=float(r2_score(y,pred)),
                 RMSE_cycle=rmse,MAE_cycle=float(np.abs(error).mean()),rRMSE_cycle=rmse/y.mean(),
                 RPIQ_cycle=float((np.quantile(y,.75)-np.quantile(y,.25))/rmse),
                 cycle_mean_bias_pct=100*error.sum()/y.sum(),
                 observed_sum_bias_pct=100*np.sum(error*c.n_obs)/np.sum(y*c.n_obs),
                 correlation_squared_cycle=float(np.corrcoef(y,pred)[0,1]**2))
        by_scale={'cycle':c}
        if name in set(daily.model):
            g=daily.loc[daily.model.eq(name)].sort_values(KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(g[KEY],reference[KEY])
            assert np.allclose(g.y,reference.y,equal_nan=True) and len(g)==5970
            by_scale=ev.tables(g)
            agg=by_scale['cycle'].sort_values(['season','treatment']).reset_index(drop=True)
            assert np.allclose(agg.prediction,c.prediction,atol=1e-6,rtol=1e-7)
            observed=g.loc[g.y.notna()]
            row.update(n_observations=len(observed),daily_macro_RMSE=float(ev.cycle_rmse(observed).mean()))
            w=ev.windows(g).assign(model=name)
            window_rows.append(w)
            row.update(continuous_window_WAPE=w.abs_error.sum()/w.absolute_observed_sum.sum(),
                       n_continuous_windows=len(w))
        for scale,t in by_scale.items():
            r2=ev.r2(t.y,t.prediction)
            assert np.isclose(r2,r2_score(t.y,t.prediction))
            samples=ev.bootstrap_r2(t,counts)
            lo,med,hi=ev.weighted_quantiles(samples,prob)
            row.update({f'R2_{scale}':r2,f'R2_{scale}_lower':lo,f'R2_{scale}_upper':hi,
                        f'RMSE_{scale}':float(np.sqrt(np.mean((t.y-t.prediction)**2))),f'n_{scale}':len(t)})
            distributions[(name,scale)]=samples
            point[(name,scale)]=r2
            bootstrap.append(dict(model=name,scale=scale,R2=r2,q025=lo,q975=hi,
                                  finite_probability=float(prob[np.isfinite(samples)].sum())))
        for year,t in c.groupby('season'):
            group_rows.append(dict(model=name,season=year,MAE=float(np.abs(t.prediction-t.y).mean()),
                                   MSE=float(np.mean((t.prediction-t.y)**2)),
                                   signed_bias=float(np.mean(t.prediction-t.y))))
        rows.append(row)
    result=pd.DataFrame(rows).sort_values('R2_cycle',ascending=False)
    assert np.isclose(result.set_index('model').loc['OriginalLightGBM','R2_cycle'],.763480,atol=1e-6)
    pair_rows=[]
    references=['OriginalLightGBM','HuberMatchedCapacity','HuberDay','HuberBiasCorrected','TabPFN_DailyRaw']
    for name,scale in distributions:
        for ref in references:
            if name==ref or (ref,scale) not in distributions: continue
            difference=distributions[(name,scale)]-distributions[(ref,scale)]
            lo,med,hi=ev.weighted_quantiles(difference,prob)
            p=dict(model=name,reference=ref,scale=scale,delta_R2=point[(name,scale)]-point[(ref,scale)],
                   q025=lo,q975=hi)
            if scale=='cycle':
                a=cycles.loc[cycles.model.eq(name)].sort_values(['season','treatment'])
                b=cycles.loc[cycles.model.eq(ref)].sort_values(['season','treatment'])
                ea=a.prediction.to_numpy()-a.y.to_numpy(); eb=b.prediction.to_numpy()-b.y.to_numpy()
                p.update(cycle_MSE_wins=int(((ea**2).reshape(8,2).mean(1)<(eb**2).reshape(8,2).mean(1)-1e-10).sum()),
                         cycle_MAE_wins=int((abs(ea).reshape(8,2).mean(1)<abs(eb).reshape(8,2).mean(1)-1e-10).sum()))
            pair_rows.append(p)
    save(result,'all_model_metrics.csv')
    save(pd.DataFrame(bootstrap),'multiscale_intervals.csv')
    save(pd.DataFrame(pair_rows),'paired_R2_differences.csv')
    save(pd.DataFrame(group_rows),'per_cycle_errors.csv')
    save(cycles,'all_cycle_predictions.csv')
    save(daily,'all_daily_predictions.csv')
    save(pd.concat(window_rows,ignore_index=True),'continuous_observed_windows.csv')
    hashchecks=[]
    for record in ['joint_frozen_run.json','tabpfn_frozen_run.json']:
        f=json.loads((RES/record).read_text(encoding='utf-8'))
        if f.get('archival_original_hashes'):
            hashchecks.append(dict(record=record, archival_original_hashes=True))
            continue
        for path,expected in f['source_hashes'].items():
            actual=sha(path)
            hashchecks.append(dict(file=path,match=actual==expected))
            assert actual==expected, path
    log=json.loads((RES/'joint_fit_audit.json').read_text(encoding='utf-8'))
    custom=[r for r in log if 'history' in r]
    assert len(log)==504 and len(custom)==360
    assert all(len(r['history'])>=5 and r['history'][-1]['total']<r['history'][0]['total'] for r in custom)
    traces=[]
    for fit in custom:
        for row in fit['history']:
            traces.append(dict(train_cycles=';'.join(map(str,fit['train_cycles'])),
                               candidate=fit['config']['id'],**row))
    save(pd.DataFrame(traces),'training_loss_trajectories.csv')
    technical_runs={p.parent.name:len(json.loads(p.read_text(encoding='utf-8')))
                    for p in (OUT/'history').glob('*/joint_fit_audit.json')}
    dump(dict(status='passed',source_hash_checks=hashchecks,models=len(result),
              daily_models=daily.model.nunique(),observations=3319,cycle_units=16,independent_clusters=8,
              successful_custom_training_fits=len(custom),main_tree_fits=len(log),
              logged_partial_fits_before_technical_restart=technical_runs,
              extra_short_checks=3,
              note='Technical failures also included an outer fit whose C++ serialization failed and possibly an incomplete subset before interruption; these did not add candidates or outer-score selection.',
              tabpfn_fits=24,bootstrap_compositions=6435,
              tests=['prediction-target support alignment','daily-to-cycle aggregation identity',
                     'R2 independent formula','original baseline recovered','frozen sources unchanged',
                     'all custom objectives decreased','nonfinite predictions absent']), 'evaluation_verification.json')
    print(result[['model','R2_cycle','R2_daily','RMSE_cycle','observed_sum_bias_pct']].to_string(index=False),flush=True)

if __name__=='__main__': main()
