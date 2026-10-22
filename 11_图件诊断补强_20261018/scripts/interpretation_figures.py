"""Data-rich interpretation panels from actual frozen-model perturbations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import generate_paper_figures as b

RES=b.OUT/'results'
GROUPS=['Temperature','Water','Atmosphere','Calendar','Crop','Fertilizer']
LABEL=['Temperature','Water and rainfall','Humidity, wind, radiation','Calendar position','Crop and tillage','Fertilizer history']

def prepare():
    complete=b.read(RES/'temporal_reliance_raw.csv')
    assert set(complete.season)==set(range(2014,2022)) and len(complete)==104
    base=complete.query("group=='Baseline'").set_index('season')
    rows=[];pooled=[]
    for group in GROUPS:
        g=complete.loc[complete.group.eq(group)]
        for component in ['level','shape','daily']:
            col=component+'_mse'
            for year,part in g.groupby('season'):
                val=(np.sqrt(part[col])-np.sqrt(base.loc[year,col])).mean()
                rows.append(dict(group=group,component=component,season=year,delta_rmse=val,
                    baseline_rmse=np.sqrt(base.loc[year,col])))
            vals=[]
            for shift,p in g.groupby('shift'):
                w=np.ones(len(p)) if component=='level' else p.n_obs.to_numpy()
                bb=base.loc[p.season]
                vals.append(float(np.sqrt(np.average(p[col],weights=w))-np.sqrt(np.average(bb[col],weights=w))))
            pooled.append(dict(group=group,component=component,pooled_delta_rmse=np.mean(vals),
                shift_min=min(vals),shift_max=max(vals)))
    folds=pd.DataFrame(rows);pool=pd.DataFrame(pooled)
    b.save(folds,RES/'temporal_reliance_by_cycle.csv');b.save(pool,RES/'temporal_reliance_pooled.csv')
    local=b.read(RES/'local_responses.csv').dropna(subset=['y'])
    assert local.groupby('perturbation').size().eq(3319).all()
    data={k:v.set_index('row_id') for k,v in local.groupby('perturbation')}
    interaction=data['TW'][['Date','season','treatment','y','T','W','actual_T_change','actual_W_change']].copy()
    for head,col in [('Level','level_row_response'),('Shape','shape_response')]:
        interaction[head]=data['TW'][col]-data['T'][col]-data['W'][col]
    # Require both local increments to have been available.
    interaction=interaction.loc[interaction.actual_T_change.gt(0)&interaction.actual_W_change.gt(0)]
    b.save(interaction.reset_index(),RES/'soil_nonadditivity_rows.csv')
    bins={k:np.unique(np.quantile(data['TW'][k],np.linspace(0,1,6))) for k in ['T','W']}
    rows=[]
    for k in ['T','W']:
        interaction[k+'_bin']=np.clip(np.searchsorted(bins[k],interaction[k],side='right')-1,0,len(bins[k])-2)
    for (wi,ti),g in interaction.groupby(['W_bin','T_bin']):
        for head in ['Level','Shape']:
            means=g.groupby('season')[head].mean()
            rows.append(dict(head=head,T_bin=ti,W_bin=wi,T_low=bins['T'][ti],T_high=bins['T'][ti+1],
                W_low=bins['W'][wi],W_high=bins['W'][wi+1],n=len(g),n_cycles=len(means),
                value=means.mean(),supported=len(g)>=20 and len(means)>=3))
    maps=pd.DataFrame(rows);b.save(maps,RES/'soil_nonadditivity_bins.csv');b.dump({k:v.tolist() for k,v in bins.items()},RES/'soil_map_edges.json')
    curve=[]
    for variable in ['T','W']:
        g=data[variable].copy();g=g.loc[g['actual_'+variable+'_change'].gt(0)]
        edges=np.unique(np.quantile(data[variable][variable],np.linspace(0,1,9)))
        g['bin']=np.clip(np.searchsorted(edges,g[variable],side='right')-1,0,len(edges)-2)
        for bi,part in g.groupby('bin'):
            for head,col in [('Level','level_row_response'),('Shape','shape_response')]:
                vals=part.groupby('season')[col].mean()
                for year,value in vals.items():
                    curve.append(dict(variable=variable,bin=bi,x=part[variable].mean(),head=head,
                        season=year,value=value,n=len(part),n_cycles=len(vals),supported=len(part)>=20 and len(vals)>=3))
    curves=pd.DataFrame(curve);b.save(curves,RES/'local_response_curves.csv')
    return folds,pool,maps,bins,curves

def reliance(ax,folds,pool,component,role,reg):
    p=folds.loc[folds.component.eq(component)];pooled=pool.loc[pool.component.eq(component)].set_index('group')
    for i,group in enumerate(GROUPS):
        y=5-i;values=p.loc[p.group.eq(group)].sort_values('season').delta_rmse.to_numpy()
        b.line(ax,[values.min(),values.max()],[y,y],role,reg,lw=1.3)
        b.point(ax,values,y+np.linspace(-.16,.16,8),role,reg,s=33,linewidths=.55,alpha=.8)
        b.point(ax,pooled.loc[group,'pooled_delta_rmse'],y,role,reg,marker='D',s=78,linewidths=.8)
    ax.axvline(0,color=b.INK,lw=.9,ls='--')
    ax.set_yticks(range(6),LABEL[::-1]);ax.set_ylim(-.6,5.6)
    ax.set_xscale('symlog',linthresh=1,linscale=1)
    maxval=max(abs(p.delta_rmse).max(),abs(pooled.pooled_delta_rmse).max())
    ticks=np.array([-30,-10,-3,-1,0,1,3,10,30,100]);ticks=ticks[(ticks>=min(-1,p.delta_rmse.min()*1.1))&(ticks<=max(1,p.delta_rmse.max()*1.1))]
    ax.set_xticks(ticks,[str(int(t)) for t in ticks]);ax.minorticks_off()
    b.style_axis(ax,f'Δ {component} RMSE ({b.UNIT})',tick_size=10.6,label_size=12.3)
    hs=[Line2D([],[],ls='',marker=m,color=b.COLOR[role],markeredgecolor=b.INK,ms=7) for m in ['o','D']]
    reg.legend(ax,hs,['Held-out cycle','Pooled'],[role,role],loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2,fontsize=10)

def fig5(preview=False):
    from extract_tree_attributions import main as extract
    extract()
    folds,pool,maps,bins,curves=prepare()
    name='Fig05_driver_reliance';folder=b.OUT/'figures'/name
    fig,axes=plt.subplots(3,2,figsize=(15.2,16.3));a,c,d,h,e,f=axes.flat;reg=b.Registry()
    reliance(a,folds,pool,'level','Level',reg);reliance(c,folds,pool,'shape','Shape',reg)
    shares=b.read(RES/'tree_shap_feature_shares.csv')
    group_shares=shares.groupby(['season','group'],as_index=False).share_pct.sum()
    for i,group in enumerate(GROUPS):
        vals=group_shares.loc[group_shares.group.eq(group)].sort_values('season').share_pct.to_numpy();yy=5-i
        reg.add(group,d.barh(yy,vals.mean(),height=.6,color=b.COLOR[group],edgecolor=b.INK,lw=.6))
        b.point(d,vals,yy+np.linspace(-.16,.16,8),group,reg,s=25,linewidths=.55,alpha=.85)
    d.set_yticks(range(6),LABEL[::-1]);d.set_ylim(-.6,5.6)
    b.style_axis(d,'LightGBM absolute attribution share (%)',tick_size=10.6,label_size=12.1);b.numeric_ticks(d,y=False,n=4)
    d.text(.98,1.035,'Fold-specific tree output',transform=d.transAxes,ha='right',fontsize=10.1)
    coverage=maps.loc[maps['head'].eq('Level')].copy()
    for row in coverage.itertuples():
        xx=(row.T_low+row.T_high)/2;yy=(row.W_low+row.W_high)/2
        role='Support' if row.supported else 'Missing'
        b.point(h,xx,yy,role,reg,s=row.n*1.45,linewidths=.6,alpha=.9)
        h.annotate(str(row.n_cycles),(xx,yy),xytext=(0,6 if row.n<60 else 0),textcoords='offset points',
                   fontsize=9.5,fontweight='bold',ha='center',va='bottom' if row.n<60 else 'center',zorder=5)
    h.set_xlim(bins['T'][0],bins['T'][-1]);h.set_ylim(bins['W'][0],bins['W'][-1])
    b.style_axis(h,'Soil temperature (°C)','WFPS (%)',tick_size=11,label_size=13);b.numeric_ticks(h,n=4)
    hs=[Line2D([],[],ls='',marker='o',ms=7,color=b.COLOR[r],markeredgecolor=b.INK) for r in ['Support','Missing']]
    reg.legend(h,hs,['Supported','Below support'],['Support','Missing'],loc='upper center',bbox_to_anchor=(.5,1.16),ncol=2,fontsize=9.6)
    from diagnostic_panels import shap_panels
    shap_panels(e,f,reg)
    b.save(folds,folder/'temporal_reliance_by_cycle.csv');b.save(pool,folder/'temporal_reliance_pooled.csv');b.save(maps,folder/'soil_nonadditivity_bins.csv')
    b.save(group_shares,folder/'tree_shap_group_shares.csv');b.save(coverage,folder/'soil_input_coverage.csv')
    b.finish(fig,axes,name,reg,[RES/'interpretation_completion.json',RES/'interpretation_frozen.json',
        RES/'temporal_reliance_raw.csv',RES/'local_responses.csv',RES/'tree_shap_feature_shares.csv',RES/'shap_frozen.json',
        RES/'shap_completion.json',Path(__file__),Path(__file__).with_name('extract_tree_attributions.py'),
        b.OUT/'interpretation_protocol.md',b.OUT/'shap_protocol.md',RES/'interaction_matrix_by_fold.csv',
        RES/'nitrogen_water_interactions.csv',RES/'interaction_completion.json'],preview=preview,gap=(.055,.075))


def figs6(preview=False):
    curves=b.read(RES/'local_response_curves.csv');name='FigS06_local_soil_responses';folder=b.OUT/'figures'/name
    fig,axes=plt.subplots(2,2,figsize=(14.5,10.2));reg=b.Registry()
    for ax,(head,variable) in zip(axes.flat,[('Level','T'),('Level','W'),('Shape','T'),('Shape','W')]):
        g=curves.loc[curves['head'].eq(head)&curves.variable.eq(variable)&curves.supported]
        allx=curves.loc[curves.variable.eq(variable)].groupby('bin').x.first()
        for year,part in g.groupby('season'):
            part=part.set_index('bin').reindex(allx.index)
            b.line(ax,allx.to_numpy(),part.value,head,reg,lw=.8,alpha=.45)
        v=g.groupby('bin').agg(x=('x','first'),mean=('value','mean'),low=('value','min'),high=('value','max'))
        v=v.reindex(allx.index);v['x']=allx
        reg.add(head,ax.fill_between(v.x,v.low,v.high,color=b.COLOR[head],alpha=.22))
        b.line(ax,v.x,v['mean'],head,reg,lw=2.6,marker='o',ms=5,outlined=True)
        ax.axhline(0,color=b.INK,ls='--',lw=.8)
        xlabel='Soil temperature (°C)' if variable=='T' else 'WFPS (%)'
        b.style_axis(ax,xlabel,f'{head} response ({b.UNIT})',tick_size=11,label_size=12.5);b.numeric_ticks(ax,n=4)
        ax.text(.03,.94,'≤ +1 °C' if variable=='T' else '≤ +5 WFPS points',transform=ax.transAxes,ha='left',va='top',fontsize=11)
    b.save(curves,folder/'local_response_curves.csv')
    b.finish(fig,axes,name,reg,[RES/'local_responses.csv',RES/'local_response_curves.csv',Path(__file__),b.OUT/'interpretation_protocol.md'],preview=preview)

def register():
    b.BUILDERS['Fig05']=fig5;b.BUILDERS['FigS06']=figs6;b.BUILDERS['FigS07']=figs7

def figs7(preview=False):
    from extract_tree_attributions import main as extract
    extract()
    shares=b.read(RES/'tree_shap_feature_shares.csv');data=b.read(RES/'tree_shap_heldout.csv')
    ranking=shares.groupby('feature').share_pct.mean().sort_values(ascending=False);top=ranking.head(12).index.tolist()
    groupmap=shares.drop_duplicates('feature').set_index('feature')['group'].to_dict()
    b.COLOR.update(dict(zip(GROUPS,[b.COL[i] for i in [1,0,4,6,7,2]])))
    names={'T':'Soil temperature','W':'WFPS','DSFA':'Days since fertilization','Ta_max':'Maximum air temperature',
        'Ta_min':'Minimum air temperature','crop_age':'Crop age','last_Ndose':'Last fertilizer dose',
        'doy_sin':'Day-of-year sine','doy_cos':'Day-of-year cosine','Rh_min':'Minimum humidity',
        'Rh_max':'Maximum humidity','Rainfall_sum':'Daily rainfall','Rg_sum':'Solar radiation','WS_mean':'Wind speed',
        'crop_MAIZE':'Maize indicator','crop_WHEAT':'Wheat indicator','crop_MUSTARD':'Mustard indicator',
        'tillage_14':'Tillage memory (14 d)'}
    def label(f):
        if f in names:return names[f]
        if f.startswith('rain_'):return f'{f.split("_")[-1]}-day rainfall'
        if f.startswith('temp_'):return f'{f.split("_")[-1]}-day air temperature'
        for prefix,kind in [('Imin','mineral'),('Iorg','organic')]:
            if f.startswith(prefix+'_sum'):return f'{f.split("sum")[-1]}-day {kind} N input'
            if f.startswith(prefix+'_exp'):return f'{kind.capitalize()} N memory ({f.split("exp")[-1]} d)'
        return f.replace('Imin','Mineral N').replace('Iorg','Organic N').replace('_',' ')
    name='FigS07_tree_attributions';folder=b.OUT/'figures'/name
    fig,axes=plt.subplots(1,2,figsize=(15.5,8.5));a,c=axes;reg=b.Registry()
    cmap=LinearSegmentedColormap.from_list('soft8_feature',[b.COL[0],'white',b.COL[1]])
    rng=np.random.default_rng(20261018)
    for i,feature in enumerate(top):
        yy=11-i;role=groupmap[feature]
        reg.add(role,a.barh(yy,ranking[feature],color=b.COLOR[role],height=.62,edgecolor=b.INK,lw=.6))
        vals=shares.loc[shares.feature.eq(feature)].sort_values('season').share_pct.to_numpy()
        b.point(a,vals,yy+np.linspace(-.17,.17,8),role,reg,s=19,linewidths=.5,alpha=.85)
        p=data.loc[data.feature.eq(feature)]
        x=p.shap_normalized.to_numpy();hist,edges=np.histogram(x,bins=45)
        ix=np.clip(np.searchsorted(edges,x,side='right')-1,0,len(hist)-1)
        jitter=rng.uniform(-1,1,len(x))*.34*hist[ix]/max(hist.max(),1)
        artist=c.scatter(x,yy+jitter,c=p.training_percentile,cmap=cmap,vmin=0,vmax=1,s=4.5,
                         linewidths=0,alpha=.8,rasterized=False)
    for ax in axes:
        ax.set_yticks(range(12),[label(f) for f in top[::-1]]);ax.set_ylim(-.65,11.65)
    b.style_axis(a,'Mean absolute attribution share (%)',tick_size=10.7,label_size=12.4);b.numeric_ticks(a,y=False,n=4)
    b.style_axis(c,'Normalized TreeSHAP contribution',tick_size=10.7,label_size=12.4);b.numeric_ticks(c,y=False,n=4)
    c.axvline(0,color=b.INK,ls='--',lw=.8)
    hs=[Line2D([],[],color=b.COLOR[g],marker='s',ls='',markeredgecolor=b.INK,ms=7) for g in GROUPS]
    reg.legend(a,hs,['Temperature','Water','Atmosphere','Calendar','Crop','Fertilizer'],GROUPS,
        loc='upper center',bbox_to_anchor=(.5,1.12),ncol=3,fontsize=9)
    cbax=c.inset_axes([1.035,.03,.035,.94]);cb=fig.colorbar(artist,cax=cbax,ticks=[0,.5,1]);cb.ax.set_yticklabels(['0','50','100'])
    cb.set_label('Input percentile in training data (%)',fontsize=11,fontweight='bold');cb.ax.tick_params(labelsize=10,width=1,length=4)
    for t in cb.ax.get_yticklabels():t.set_fontweight('bold')
    b.save(shares,folder/'all_42_feature_shares.csv');b.save(ranking.rename('mean_share_pct').reset_index(),folder/'feature_ranking.csv')
    b.finish(fig,axes,name,reg,[RES/'tree_shap_heldout.csv',RES/'tree_shap_feature_shares.csv',RES/'shap_completion.json',
        RES/'shap_frozen.json',b.OUT/'shap_protocol.md',Path(__file__),Path(__file__).with_name('extract_tree_attributions.py')],preview=preview,gap=(.055,.075))
    b.dump(dict(passed=True,kind='continuous_feature_value',endpoints=[b.COL[0],'#ffffff',b.COL[1]],
        range=[0,1],definition='empirical feature percentile in the outer-training observations'),folder/'continuous_color_audit.json')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
    b.configure_publication_style();fig5(args.preview);figs6(args.preview)
