"""Identical-support, cycle-clustered evaluation of all frozen variants."""
import math
from common import *

def main():
    ev=module('audit05',ROOT/'05_框架再评估_20261018/scripts/recheck_saved_predictions.py')
    refs=['OriginalLightGBM','TabPFN_DailyRaw','JointSelected','Joint01']
    prior=read(PREV/'results/all_daily_predictions.csv')
    prior=prior.loc[prior.model.isin(refs)]
    daily=pd.concat([prior,read(RES/'new_daily_predictions.csv')],ignore_index=True)
    daily['Date']=pd.to_datetime(daily.Date)
    key=['season','treatment','Date']
    assert not daily.duplicated(['model']+key).any()
    reference=daily.loc[daily.model.eq('OriginalLightGBM')].sort_values(key).reset_index(drop=True)
    counts=np.array(list(ev.compositions(8,8)))
    multiplicity=np.array([math.factorial(8)//math.prod(math.factorial(int(k)) for k in row) for row in counts])
    probability=multiplicity/8**8
    assert len(counts)==6435 and multiplicity.sum()==8**8
    distributions,points={},{}
    metrics=[]; cycles=[]; per_cycle=[]; windows=[]; by_scale=[]; decompositions=[]
    for name,g in daily.groupby('model'):
        g=g.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(g[key],reference[key])
        assert len(g)==5970 and np.allclose(g.y,reference.y,equal_nan=True)
        assert np.isfinite(g.prediction).all()
        tables=ev.tables(g)
        c=tables['cycle'].copy().assign(model=name)
        cycles.append(c)
        error=c.prediction-c.y
        rmse=float(np.sqrt(np.mean(error**2)))
        obs=g.loc[g.y.notna()].copy()
        row=dict(model=name,n_clusters=8,n_cycle_units=len(c),n_daily=len(obs),
                 RMSE_cycle=rmse,MAE_cycle=float(np.abs(error).mean()),
                 rRMSE_cycle=rmse/c.y.mean(),RPIQ_cycle=(c.y.quantile(.75)-c.y.quantile(.25))/rmse,
                 observed_sum_bias_pct=100*(obs.prediction-obs.y).sum()/obs.y.sum(),
                 daily_macro_RMSE=float(ev.cycle_rmse(obs).mean()))
        w=ev.windows(g).assign(model=name); windows.append(w)
        row['continuous_window_WAPE']=w.abs_error.sum()/w.absolute_observed_sum.sum()
        for scale,t in tables.items():
            r2=ev.r2(t.y,t.prediction)
            samples=ev.bootstrap_r2(t,counts)
            lo,_,hi=ev.weighted_quantiles(samples,probability)
            distributions[name,scale]=samples; points[name,scale]=r2
            row.update({f'R2_{scale}':r2,f'R2_{scale}_lower':lo,f'R2_{scale}_upper':hi,
                        f'RMSE_{scale}':float(np.sqrt(np.mean((t.prediction-t.y)**2)))})
            by_scale.append(dict(model=name,scale=scale,R2=r2,q025=lo,q975=hi,n=len(t)))
        for yr,x in c.groupby('season'):
            per_cycle.append(dict(model=name,season=yr,MSE=float(np.mean((x.prediction-x.y)**2)),
                                  MAE=float(np.mean(np.abs(x.prediction-x.y)))))
        # Exact unweighted daily SSE decomposition; observed-day counts matter.
        ymean=obs.groupby(['season','treatment']).y.transform('mean')
        pmean=obs.groupby(['season','treatment']).prediction.transform('mean')
        level_sse=float(np.sum((ymean-pmean)**2))
        shape_sse=float(np.sum(((obs.y-ymean)-(obs.prediction-pmean))**2))
        total_sse=float(np.sum((obs.y-obs.prediction)**2))
        assert np.isclose(total_sse,level_sse+shape_sse,atol=1e-7)
        decompositions.append(dict(model=name,daily_sse=total_sse,level_sse=level_sse,
                                   shape_sse=shape_sse,level_fraction=level_sse/total_sse,
                                   pulse_2014AN_over300_sse=float(np.sum((obs.loc[(obs.season==2014)&(obs.treatment=='AN')&(obs.y>300),'y']-
                                       obs.loc[(obs.season==2014)&(obs.treatment=='AN')&(obs.y>300),'prediction'])**2))))
        metrics.append(row)
    metrics=pd.DataFrame(metrics).set_index('model')
    assert np.isclose(metrics.loc['OriginalLightGBM','R2_cycle'],.7634799697989234,atol=1e-12)
    assert np.isclose(metrics.loc['OriginalLightGBM','R2_daily'],.2777018390485084,atol=1e-12)
    assert np.isclose(metrics.loc['TabPFN_DailyRaw','R2_daily'],.3320908181064949,atol=1e-12)
    for name in ['MeanReconciled','PulseShapeOnly']:
        assert np.isclose(metrics.loc[name,'R2_cycle'],metrics.loc['OriginalLightGBM','R2_cycle'],atol=1e-12)
    metrics['primary_point_gate']=(metrics.R2_cycle>metrics.loc['OriginalLightGBM','R2_cycle']+1e-12)&(metrics.R2_daily>=metrics.loc['OriginalLightGBM','R2_daily']-1e-12)
    metrics['strong_daily_point_gate']=(metrics.R2_cycle>metrics.loc['OriginalLightGBM','R2_cycle']+1e-12)&(metrics.R2_daily>=metrics.loc['TabPFN_DailyRaw','R2_daily']-1e-12)
    pair=[]
    comparators=['OriginalLightGBM','TabPFN_DailyRaw','MeanReconciled','AffineReconciled','DecoupledNoPulse','DecoupledPulse']
    per=pd.DataFrame(per_cycle)
    for (name,scale),a in distributions.items():
        for ref in comparators:
            if ref==name: continue
            delta=a-distributions[ref,scale]
            lo,_,hi=ev.weighted_quantiles(delta,probability)
            row=dict(model=name,reference=ref,scale=scale,
                     delta_R2=points[name,scale]-points[ref,scale],q025=lo,q975=hi)
            if scale=='cycle':
                e=per.loc[per.model.eq(name)].sort_values('season').MSE.to_numpy()
                b=per.loc[per.model.eq(ref)].sort_values('season').MSE.to_numpy()
                row['cycle_MSE_wins']=int((e<b-1e-10).sum())
            pair.append(row)
    save(metrics.reset_index().sort_values('R2_cycle',ascending=False),RES/'all_model_metrics.csv')
    save(pd.concat(cycles,ignore_index=True),RES/'all_cycle_predictions.csv')
    save(daily,RES/'all_daily_predictions.csv')
    save(pd.DataFrame(pair),RES/'paired_R2_differences.csv')
    save(per,RES/'per_cycle_errors.csv')
    save(pd.concat(windows,ignore_index=True),RES/'continuous_observed_windows.csv')
    save(pd.DataFrame(by_scale),RES/'multiscale_intervals.csv')
    save(pd.DataFrame(decompositions),RES/'SSE_decomposition.csv')
    records=[]
    for filename in ['frozen_run.json','head_frozen_run.json','tabpfn_crossfit_frozen.json']:
        info=json.loads((RES/filename).read_text(encoding='utf-8'))
        if info.get('archival_original_hashes'):
            records.append(dict(record=filename, archival_original_hashes=True,
                                verification='public_manifest.json verifies the portable copy'))
            continue
        for path,expected in info['source_hashes'].items():
            current=sha(path)
            assert current==expected, path
            records.append(dict(path=path,sha256=current))
    dump(dict(status='passed',models=len(metrics),calendar_rows_per_model=5970,
              observed_rows_per_model=3319,cycle_units_per_model=16,bootstrap_cluster_count=8,
              unique_bootstrap_count_vectors=6435,bootstrap_ordered_samples=8**8,
              main_candidate='DecoupledPulse',primary_point_gate=bool(metrics.loc['DecoupledPulse','primary_point_gate']),
              stronger_point_gate=bool(metrics.loc['DecoupledPulse','strong_daily_point_gate']),
              verified_sources=records),RES/'evaluation_verification.json')
    print(metrics[['R2_cycle','R2_daily','primary_point_gate','strong_daily_point_gate']].sort_values('R2_cycle',ascending=False).to_string(),flush=True)

if __name__=='__main__': main()
