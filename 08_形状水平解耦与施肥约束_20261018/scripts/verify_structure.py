"""Synthetic numerical checks only; no scientific performance estimates."""
from common import *
from fit_decoupled import learn, learn_tied, predict, state, structural_checks

def main():
    rng=np.random.default_rng(SEED)
    rows=[]
    for yr in range(2014,2018):
        for treatment in ['AN','PS']:
            for day in range(40):
                rows.append(dict(season=yr,treatment=treatment,day=day,T=15+8*np.sin(day/15),
                                 W=50+20*np.cos(day/11),p_lgb=2+day/10,
                                 p_tab=3+2*np.sin(day/3),
                                 y=1+np.sin(day/3)+.15*max(day-10,0)+rng.normal(0,.3)))
    d=pd.DataFrame(rows);d['row_id']=np.arange(len(d))
    d.loc[d.day.mod(7).eq(0),'y']=np.nan
    A=np.zeros((len(d),5));A[:,0]=np.maximum(d.day.to_numpy()-10,0)
    IM=np.zeros_like(A);IO=np.zeros_like(A)
    IM[:,0]=np.where(d.day>=10,10.,0.)
    IO[:,0]=np.where((d.day>=10)&d.treatment.eq('PS'),15.,0.)
    arrays=A,IM,IO
    result=structural_checks();result['purpose']='Synthetic implementation verification; not research results'
    model=learn(d,arrays);p,m=predict(model,d,arrays)
    changed=d.copy();changed.loc[changed.y.notna(),'y']=rng.normal(1000,100,changed.y.notna().sum())
    p2,m2=predict(model,changed,arrays)
    assert np.array_equal(p,p2) and np.array_equal(m,m2)
    doubled=(A,IM*2,IO*2)
    b=kernel_basis(d,arrays);b2=kernel_basis(d,doubled)
    assert np.array_equal(b2,2*b) and np.max(np.abs(b[d.day<10]))==0
    g=groups(d);obs=d.y.notna().to_numpy()
    assert np.max(np.abs(mean_broadcast(p,g,obs)-m))<1e-10
    for kwargs in [dict(pulse=False),dict(constrained=False),dict(environment=False),
                   dict(shape=False,pulse=False,level_mode='bias'),dict(lam=.1),dict(lam=10.)]:
        alt=learn(d,arrays,**kwargs);predict(alt,d,arrays)
    tied=learn_tied(d,arrays);predict(tied,d,arrays)
    result.update(inference_label_perturbation_change=float(np.max(abs(p-p2))),
                  exact_dose_scaling=True,before_application_zero=True,
                  mean_consistency_error=float(np.max(abs(mean_broadcast(p,g,obs)-m))),
                  level_KKT=model['level_audit']['kkt_max'],shape_KKT=model['shape_audit']['kkt_max'],
                  tied_KKT=tied['audit']['kkt_max'])
    dump(result,RES/'synthetic_structure_verification.json')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
