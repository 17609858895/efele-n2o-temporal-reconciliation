"""Manuscript figures from frozen EFELE predictions; no model fitting.

Run with python. --preview builds quick review renders;
default exports 600 dpi PNG, vector PDF and independent panels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
from matplotlib.transforms import Bbox
import fitz

HERE = Path(__file__).resolve().parent
if __name__=='__main__':
    sys.modules['generate_paper_figures']=sys.modules[__name__]
OUT = HERE.parent
ROOT = OUT.parent
R08 = ROOT/'08_形状水平解耦与施肥约束_20261018'
RES = R08/'results'
R07 = ROOT/'07_周期损失与TabPFN_20261018/results'
HELPERS = ROOT/'portable_helpers'
sys.path.append(str(HELPERS))
from publication_style_local import configure_publication_style, style_axis, numeric_ticks, export_subfigures, save_figure_with_subfigures
from scientific_palettes import get_palette
from audit_palette_usage import audit_series_artists, validate_series_colors

PALETTE = 'au_nimno_soft_8'
COL = get_palette(PALETTE, n=8)
INK = '#262626'
CORE = ['OriginalLightGBM', 'TabPFN_DailyRaw', 'Joint01', 'BlendCycle',
        'MeanReconciled', 'DecoupledPulse', 'AffineReconciled']
LABEL = dict(zip(CORE, ['LightGBM', 'TabPFN', 'Joint loss', 'Learned blend',
                      'Mean reconciliation', 'Decoupled + pulse', 'Affine reconciliation']))
ABL = ['MeanReconciled', 'AffineReconciled', 'DecoupledNoPulse', 'DecoupledPulse',
       'DecoupledUnconstrained', 'TiedPulse', 'PulseShapeOnly', 'PulseLevelOnly']
ALABEL = dict(zip(ABL, ['Mean reconciliation', 'Affine reconciliation', 'Free shape / no pulse',
                      'Independent pulse heads', 'Unconstrained pulse', 'Shared pulse weights',
                      'Pulse in shape only', 'Pulse in level only']))
# Role colours are shared across every figure, including legend proxies.
COLOR = dict(zip(CORE, [COL[i] for i in [0,3,7,4,6,2,1]]))
COLOR.update(dict(zip(ABL, [COL[i] for i in [6,1,7,2,5,4,3,0]])))
COLOR.update(Observed=COL[2], Missing=COL[6], Level=COL[2], Shape=COL[0],
             Nmin=COL[2], Norg=COL[5], Drivers=COL[7], Support=COL[4], Training=COL[3])
COLOR.update(Temperature=COL[1],Water=COL[0],Atmosphere=COL[4],Calendar=COL[6],Crop=COL[7],Fertilizer=COL[2])
UNIT = 'g N$_2$O-N ha$^{-1}$ d$^{-1}$'
SUMUNIT = 'g N$_2$O-N ha$^{-1}$'
MODEL_MARKER = {'OriginalLightGBM':'o', 'TabPFN_DailyRaw':'D', 'MeanReconciled':'s',
                'AffineReconciled':'^', 'DecoupledPulse':'*'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(obj, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def save(data, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding='utf-8-sig')

def read(path):
    return pd.read_csv(path, float_precision='round_trip')

M = read(RES/'all_model_metrics.csv').set_index('model')
D = read(RES/'all_daily_predictions.csv')
D['Date'] = pd.to_datetime(D.Date)
C = read(RES/'all_cycle_predictions.csv')
W = read(RES/'continuous_observed_windows.csv')
SSE = read(RES/'SSE_decomposition.csv').set_index('model')
PAIR = read(RES/'paired_R2_differences.csv')
MAN = read(RES/'season_manifest.csv')
MAN = MAN.loc[MAN.included].set_index('season')
EVENTS = read(RES/'fertilizer_events.csv')
EVENTS['date'] = pd.to_datetime(EVENTS.date)
PRED = {m:g.sort_values(['season','treatment','Date']).copy() for m,g in D.groupby('model')}
SOURCE_FILES = [RES/f for f in ['all_model_metrics.csv','all_daily_predictions.csv',
    'all_cycle_predictions.csv','continuous_observed_windows.csv','SSE_decomposition.csv',
    'paired_R2_differences.csv','season_manifest.csv','fertilizer_events.csv']]
SOURCE_FILES += [R07/'all_model_metrics.csv']
SOURCE_FILES += sorted((R08/'models').glob('*_DecoupledPulse.json'))

class Registry:
    def __init__(self): self.artists={}
    def add(self, name, artist):
        self.artists.setdefault(name, []).append(artist)
        return artist
    def legend(self, ax, handles, labels, names, **kw):
        leg=ax.legend(handles, labels, **kw)
        for name, artist in zip(names,leg.legend_handles):
            if name is not None: self.add(name,artist)
        return leg
    def audit(self):
        colors={k:COLOR[k] for k in self.artists}
        return audit_series_artists(PALETTE,self.artists,colors), colors

def line(ax, x, y, name, reg, *, outlined=False, **kw):
    artist=ax.plot(x,y,color=COLOR[name],**kw)[0]
    if outlined:
        artist.set_path_effects([pe.Stroke(linewidth=artist.get_linewidth()+.7,foreground=INK),pe.Normal()])
    return reg.add(name,artist)

def point(ax, x, y, name, reg, **kw):
    defaults=dict(s=65,edgecolors=INK,linewidths=.7,zorder=4)
    defaults.update(kw)
    return reg.add(name,ax.scatter(x,y,c=COLOR[name],**defaults))

def letters(axes):
    for letter,ax in zip('abcdefghijklmnopqrstuvwxyz',np.asarray(axes).flat):
        text=ax.text(-.14,1.04,letter,transform=ax.transAxes,fontsize=max(17,7.5*ax.figure.get_figwidth()/(190/25.4)),fontweight='bold',va='bottom')
        text.set_gid('panel_letter')

def align_outer(fig, axes, *, bounds=(.025,.025,.985,.98), gap=(.045,.065)):
    """Refit axes to aligned outer boxes, measured afresh on every run.

    Includes tick labels, axis labels, panel letters and axes-owned legends.
    This keeps row/column *visual* boundaries aligned, not just plotting areas.
    """
    axes=np.asarray(axes)
    if axes.ndim==1: axes=axes[None,:]
    nr,nc=axes.shape
    x0,y0,x1,y1=bounds; gx,gy=gap
    cw=(x1-x0-(nc-1)*gx)/nc; ch=(y1-y0-(nr-1)*gy)/nr
    targets=[]
    for r in range(nr):
        for c in range(nc):
            left=x0+c*(cw+gx); bottom=y1-(r+1)*ch-r*gy
            targets.append(Bbox.from_extents(left,bottom,left+cw,bottom+ch))
    for iteration in range(24):
        fig.canvas.draw(); renderer=fig.canvas.get_renderer(); maxerr=0.
        for ax,target in zip(axes.flat,targets):
            tight=ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            pos=ax.get_position(); delta=np.array(target.extents)-np.array(tight.extents)
            maxerr=max(maxerr,float(np.abs(delta).max()))
            new=np.array(pos.extents)+delta*.85
            if new[2]-new[0]<.05 or new[3]-new[1]<.05:
                raise RuntimeError('Figure cell too small for its labels')
            ax.set_position(Bbox.from_extents(*new))
        if maxerr<.00045: break
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    # Put letters at the actual outer corner, including the legend/label extent.
    # Keeping them axes-owned preserves them in every standalone panel export.
    for ax,target in zip(axes.flat,targets):
        label=next(t for t in ax.texts if t.get_gid()=='panel_letter')
        label.set_transform(fig.transFigure);label.set_position((target.x0,target.y1))
        label.set_ha('left');label.set_va('top')
    # Shared observation rows need matching data-space heights as well.
    if getattr(fig,'_match_row_plot_heights',False):
        for row in axes:
            bottom=max(ax.get_position().y0 for ax in row)
            top=min(ax.get_position().y1 for ax in row)
            for ax in row:
                pos=ax.get_position();ax.set_position([pos.x0,bottom,pos.width,top-bottom])
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    boxes=np.array([a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).extents for a in axes.flat]).reshape(nr,nc,4)
    row_error=max([np.ptp(boxes[r,:,j]) for r in range(nr) for j in [1,3]] or [0])
    col_error=max([np.ptp(boxes[:,c,j]) for c in range(nc) for j in [0,2]] or [0])
    return dict(iterations=iteration+1,row_outer_misalignment=float(row_error),
                column_outer_misalignment=float(col_error),tolerance=.002,
                passed=bool(max(row_error,col_error)<.002),outer_boxes=boxes.tolist())

def finish(fig, axes, name, reg, sources, *, preview=False, framework=False, gap=(.055,.075)):
    folder=OUT/'figures'/name; folder.mkdir(parents=True,exist_ok=True)
    qa=OUT/'qa'/name; qa.mkdir(parents=True,exist_ok=True)
    from matplotlib.text import Text
    # Type is sized for a 190 mm final figure, before the alignment measurement.
    fig.canvas.draw()
    min_size=7.5*fig.get_figwidth()/(190/25.4)
    for text in fig.findobj(Text):
        if text.get_text().strip(): text.set_fontsize(max(text.get_fontsize(),min_size))
    if framework and len(axes)==3:
        letters(axes)
        alignment=align_outer(fig,np.asarray(axes[:2])[None,:],bounds=(.025,.60,.985,.985),gap=(.065,.06))
        label=next(t for t in axes[2].texts if t.get_gid()=='panel_letter')
        label.set_transform(fig.transFigure);label.set_position((.025,.555));label.set_ha('left');label.set_va('top')
        alignment['layout']='aligned geography/timeline row above a spanning vector framework'
    elif framework: alignment=dict(status='not_applicable_single_continuous_framework')
    else:
        letters(axes)
        alignment=align_outer(fig,axes,gap=gap,bounds=getattr(fig,'_outer_bounds',(.025,.025,.985,.98)))
        if not alignment['passed']: raise RuntimeError(f'Alignment failed: {name} {alignment}')
    audit,colors=reg.audit()
    dump(audit,folder/'palette_audit.json');dump(colors,folder/'series_colors.json')
    dump(alignment,folder/'alignment_check.json')
    fig.savefig(folder/'preview.png',dpi=120,bbox_inches='tight',facecolor='white')
    fig.savefig(qa/'review.pdf',bbox_inches='tight',facecolor='white',metadata={'Author':'CHONG LIU','Creator':'CHONG LIU'})
    if not preview:
        save_figure_with_subfigures(fig,folder,name,dpi=600,close=False)
        if framework:
            # Editable text and vector geometry, generated directly from code.
            fig.savefig(folder/f'{name}.svg',bbox_inches='tight',metadata={'Creator':'CHONG LIU'})
        for p in folder.rglob('*.pdf'):
            with fitz.open(p) as doc:
                metadata=doc.metadata;metadata.update(author='CHONG LIU',creator='CHONG LIU')
                doc.set_metadata(metadata);doc.saveIncr()
            with fitz.open(p) as doc:
                assert doc.metadata['author']==doc.metadata['creator']=='CHONG LIU'
                relative=p.relative_to(folder)
                out=qa/relative.with_suffix('.png');out.parent.mkdir(parents=True,exist_ok=True)
                doc[0].get_pixmap(dpi=100).save(out)
    else:
        # These are QA-only independent panels, explicitly not final 600-dpi files.
        export_subfigures(fig,qa,'review',dpi=110)
    plt.close(fig)
    font_record=[]
    # Text sizes are checked again against the actual PDF width including margins.
    with fitz.open(qa/'review.pdf') as doc: exported_width_mm=doc[0].rect.width/72*25.4
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip():
            font_record.append(dict(text=text.get_text(),source_pt=text.get_fontsize(),final_190mm_pt=text.get_fontsize()*190/exported_width_mm))
    dump(dict(target_width_mm=190,exported_width_mm=exported_width_mm,
        minimum_body_pt=min(x['final_190mm_pt'] for x in font_record),texts=font_record),folder/'font_size_audit.json')
    common=[Path(__file__),HERE/'diagnostic_panels.py',HERE/'compute_extensions.py',HERE/'publication_style_local.py',HELPERS/'scientific_palettes.py',HELPERS/'audit_palette_usage.py']
    dump(dict(figure=name,panels=len(np.asarray(axes).ravel()),preview_only=preview,
              palette=PALETTE,source_hashes={str(p):sha(p) for p in list(dict.fromkeys(sources+common))},
              author='CHONG LIU',creator='CHONG LIU',alignment=alignment),folder/'export_record.json')
    print(f'{name}: {"preview" if preview else "600 dpi + PDF + panels"}',flush=True)

def draw_framework(ax,reg):
    ax.set(xlim=(0,164),ylim=(0,97));ax.axis('off')
    def box(x,y,w,h,role,rounding=1.8):
        patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.02,rounding_size={rounding}',
                             fc=COLOR[role],ec=INK,lw=1.05)
        ax.add_patch(patch);reg.add(role,patch);return patch
    def txt(x,y,s,size=13,bold=False,**kw):
        return ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=INK,**kw)
    def arrow(start,end,style='-',connectionstyle='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,color=INK,lw=1.3,linestyle=style,
                                    connectionstyle=connectionstyle))
    box(2,28,28,45,'Drivers')
    txt(16,69,'Agricultural drivers',14,True)
    # A small vector plant communicates the agricultural inputs without generated imagery.
    ax.plot([8,8],[48,59],color=INK,lw=1.3)
    for xx,yy,angle in [(6.2,55,30),(9.8,57,-30)]:
        leaf=Ellipse((xx,yy),4.5,1.8,angle=angle,facecolor=COLOR['Observed'],edgecolor=INK,lw=.7)
        ax.add_patch(leaf);reg.add('Observed',leaf)
    ax.plot([4,12],[47,47],color=INK,lw=1.2)
    txt(20,55,'Weather\nSoil T and water',11.8)
    txt(16,40,'Crop and fertilizer\nAmount, form\nand timing',12)
    box(2,15,28,10,'Support');txt(16,20,'Observation dates  $O_c$',12.5)
    box(38,59,25,17,'OriginalLightGBM');txt(50.5,70,'LightGBM',15,True);txt(50.5,64,'Daily prediction  $L_t$',12.7)
    box(38,29,25,17,'TabPFN_DailyRaw');txt(50.5,40,'TabPFN',15,True);txt(50.5,34,'Daily prediction  $T_t$',12.7)
    box(73,57,37,22,'AffineReconciled')
    txt(91.5,74,'Cycle-level calibration',14,True)
    txt(91.5,65,r'$\widehat m_c=a_h+b_h A_c(L)$',17)
    txt(91.5,59.5,r'$b_h\geq 0$',12)
    box(73,27,37,22,'TabPFN_DailyRaw')
    txt(91.5,44,'Centred daily shape',14,True)
    txt(91.5,35,r'$C_c(T)_t=T_t-A_c(T)$',16.5)
    arrow((30,59),(38,67));arrow((30,41),(38,37))
    arrow((63,67.5),(73,67.5));arrow((63,37.5),(73,37.5))
    # Shared support is explicitly dates, never the held-out flux or its mean.
    ax.plot([16,16,68,68],[25,25.7,25.7,67.5],color=INK,lw=1,ls=':')
    arrow((68,67.5),(73,67.5),':');arrow((68,37.5),(73,37.5),':')
    box(38,83,72,12,'Training')
    txt(74,91.2,'Fit calibration inside seven training cycles',13.5,True)
    txt(74,86.2,'Internal held-out predictions  →  shrink toward $a_h=0$, $b_h=1$',11.8)
    arrow((91.5,83),(91.5,79),'--')
    merge=Circle((119,53),3.4,facecolor='white',edgecolor=INK,lw=1.5);ax.add_patch(merge)
    txt(119,53,'+',22)
    arrow((110,67),(116.6,55.4));arrow((110,38),(116.6,50.5))
    box(129,29,33,48,'AffineReconciled')
    txt(145.5,71,'Daily estimate',14,True)
    txt(145.5,63,r'$\widehat f_t=\widehat m_c+C_c(T)_t$',16)
    ax.plot([132,159],[56,56],color=INK,lw=.8)
    txt(145.5,51,'Cycle mean',13,True)
    txt(145.5,45,r'$A_c(\widehat f)=\widehat m_c$',15.5)
    txt(145.5,38,'Daily anomalies',13,True)
    txt(145.5,32.5,r'$C_c(\widehat f)=C_c(T)$',15)
    arrow((122.4,53),(129,53))
    txt(145,20,'Held-out cycle',12.7,True)
    txt(145,13,'Retrospective estimation\non known observation dates',11.4)
    txt(18,6.5,r'$A_c(v)=\frac{1}{|O_c|}\sum_{t\in O_c}v_t$',16)
    save(pd.DataFrame([dict(operator='support_mean',expression='A_c(v)=mean over known O_c'),
        dict(operator='centering',expression='C_c(v)=v-A_c(v)'),
        dict(operator='level',expression='m_c=a_h+b_h A_c(L)'),
        dict(operator='output',expression='f_t=m_c+C_c(T)_t')]),OUT/'figures/Fig01_framework/operators.csv')
    return

def fig1(preview=False):
    from diagnostic_panels import geography,timeline,GEO
    fig=plt.figure(figsize=(16,15.5));reg=Registry()
    a=fig.add_axes([.04,.63,.42,.33]);bb=fig.add_axes([.57,.63,.40,.33]);c=fig.add_axes([.025,.025,.96,.52])
    geography(a,reg);timeline(bb,reg);draw_framework(c,reg)
    finish(fig,[a,bb,c],'Fig01_framework',reg,[R08/'scripts/fit_decoupled.py',R08/'方法实现与创新边界.md',
        GEO/'site_locations.csv',GEO/'ne_110m_admin_0_countries.geojson',RES/'season_manifest.csv',
        RES/'fertilizer_events.csv',RES/'all_daily_predictions.csv'],preview=preview,framework=True)


def fig2(preview=False):
    name='Fig02_scales_and_errors';path=OUT/'figures'/name
    fig,axes=plt.subplots(3,2,figsize=(14.5,16.0));reg=Registry()
    a,b,c,d,e,f=axes.flat
    featured=['OriginalLightGBM','TabPFN_DailyRaw','AffineReconciled']
    trace=PRED['OriginalLightGBM'].query('season==2014 and treatment=="AN"').copy()
    trace=trace.loc[trace.Date.between('2014-04-20','2014-06-10')]
    point(a,trace.Date,trace.y,'Observed',reg,s=32,linewidths=.45,label='Observed')
    handles=[a.collections[-1]];names=['Observed'];labels=['Observed']
    for mod,ls in [('OriginalLightGBM','--'),('AffineReconciled','-')]:
        g=PRED[mod].query('season==2014 and treatment=="AN"').set_index('Date').loc[trace.Date]
        h=line(a,g.index,g.prediction,mod,reg,lw=2.2,ls=ls,outlined=mod=='AffineReconciled')
        handles.append(h);names.append(mod);labels.append(LABEL[mod])
        trace[mod]=g.prediction.to_numpy()
    # A structural event marker, not a third measured series.
    a.axvline(pd.Timestamp('2014-04-28'),color=INK,lw=1,ls=':')
    a.text(.97,.94,'2014 · ammonium nitrate',transform=a.transAxes,fontsize=11.5,va='top',ha='right')
    a.set_ylim(-35,590);a.set_xlim(pd.Timestamp('2014-04-19'),pd.Timestamp('2014-06-11'))
    a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2));a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    style_axis(a,'Date',f'Daily flux ({UNIT})',tick_size=11.5,label_size=13.5);numeric_ticks(a,x=False,n=4)
    reg.legend(a,handles,labels,names,loc='upper center',bbox_to_anchor=(.5,1.19),ncol=3,fontsize=9.2,handlelength=1.6,columnspacing=.9)
    save(trace,path/'pulse_trace.csv')
    scales=['daily','weekly','monthly','cycle']
    for mod in featured:
        line(b,np.arange(4),[M.loc[mod,f'R2_{s}'] for s in scales],mod,reg,lw=2,marker=MODEL_MARKER[mod],ms=7,
             markeredgecolor=INK,markeredgewidth=.7,outlined=mod=='AffineReconciled')
    b.set_xticks(range(4),['Day','Week','Month','Cycle']);b.set_ylim(.1,.9)
    style_axis(b,'Aggregation scale','Prediction R²',tick_size=11.5,label_size=13.5);numeric_ticks(b,x=False,n=4)
    hs=[Line2D([],[],color=COLOR[m],marker=MODEL_MARKER[m],markeredgecolor=INK,lw=2) for m in featured]
    reg.legend(b,hs,[LABEL[m] for m in featured],featured,loc='upper left',fontsize=10.3)
    save(M.loc[featured,[f'R2_{s}' for s in scales]].reset_index(),path/'scale_scores.csv')
    ranked=[]
    for mod in ['OriginalLightGBM','AffineReconciled']:
        g=PRED[mod].dropna(subset=['y']).sort_values('y',ascending=False).copy()
        g['rank']=np.arange(1,len(g)+1);g['squared_error']=(g.y-g.prediction)**2
        g['cumulative_sse_fraction']=g.squared_error.cumsum()/g.squared_error.sum();ranked.append(g)
        line(c,g['rank'],100*g.cumulative_sse_fraction,mod,reg,lw=2.3,ls='--' if mod=='OriginalLightGBM' else '-',outlined=mod=='AffineReconciled')
        fraction=g.iloc[4].cumulative_sse_fraction*100
        point(c,5,fraction,mod,reg,s=60,marker=MODEL_MARKER[mod])
        c.annotate(f'{fraction:.1f}%',(5,fraction),xytext=(12,7 if mod=='AffineReconciled' else -20),textcoords='offset points',fontsize=11)
    c.set_xscale('log');c.set_xlim(1,4000);c.set_ylim(0,105)
    c.axvline(5,color=INK,lw=1,ls=':');c.set_xticks([1,5,50,500,3319],['1','5','50','500','3319'])
    c.minorticks_off();numeric_ticks(c,x=False,n=4)
    style_axis(c,'Days ranked by observed flux','Cumulative daily SSE (%)',tick_size=11.5,label_size=13.5)
    hs=[Line2D([],[],color=COLOR[m],lw=2,ls=l) for m,l in [('OriginalLightGBM','--'),('AffineReconciled','-')]]
    reg.legend(c,hs,[LABEL[m] for m in ['OriginalLightGBM','AffineReconciled']],['OriginalLightGBM','AffineReconciled'],loc='lower right',fontsize=10.2)
    save(pd.concat(ranked),path/'ranked_error_concentration.csv')
    decomp=['OriginalLightGBM','TabPFN_DailyRaw','MeanReconciled','AffineReconciled','DecoupledPulse']
    for i,mod in enumerate(decomp):
        y=len(decomp)-1-i;ll=SSE.loc[mod,'level_sse']/1e6;ss=SSE.loc[mod,'shape_sse']/1e6
        reg.add('Level',d.barh(y,ll,color=COLOR['Level'],edgecolor=INK,lw=.6,height=.55,hatch='//'))
        reg.add('Shape',d.barh(y,ss,left=ll,color=COLOR['Shape'],edgecolor=INK,lw=.6,height=.55))
    d.set_yticks(range(len(decomp)),[LABEL[m] for m in decomp[::-1]])
    d.set_xlim(0,2.05);d.set_ylim(-.6,4.7)
    style_axis(d,'Daily SSE (×10$^6$)',tick_size=10.8,label_size=13.5);numeric_ticks(d,y=False,n=4)
    hs=[Rectangle((0,0),1,1,fc=COLOR[k],ec=INK,hatch='//' if k=='Level' else None) for k in ['Level','Shape']]
    reg.legend(d,hs,['Cycle level','Daily shape'],['Level','Shape'],loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,fontsize=10.3)
    save(SSE.loc[decomp].reset_index(),path/'error_decomposition.csv')
    from diagnostic_panels import taylor,conditional_bias
    taylor(e,reg);conditional_bias(f,reg)
    finish(fig,axes,name,reg,SOURCE_FILES[:6]+[RES/'fertilizer_events.csv',OUT/'tables/daily_dispersion_metrics.csv',
        OUT/'tables/conditional_bias.csv',OUT/'results/diagnostics_protocol.json'],preview=preview,gap=(.045,.08))

def forest(ax,names,labels,scale,reg,paired=False):
    for i,mod in enumerate(names):
        y=len(names)-1-i
        if paired:
            if mod=='MeanReconciled': value,lo,hi=0.,0.,0.
            else:
                row=PAIR.query('model==@mod and reference=="MeanReconciled" and scale==@scale').iloc[0]
                value,lo,hi=row.delta_R2,row.q025,row.q975
        else:
            row=M.loc[mod];value,lo,hi=row[f'R2_{scale}'],row[f'R2_{scale}_lower'],row[f'R2_{scale}_upper']
        line(ax,[lo,hi],[y,y],mod,reg,lw=3.2,outlined=True)
        point(ax,value,y,mod,reg,s=112 if mod=='DecoupledPulse' else 65,marker=MODEL_MARKER.get(mod,'o'))
    ax.set_yticks(range(len(names)),[labels[m] for m in names[::-1]])
    ax.set_ylim(-.65,len(names)-.35);ax.axvline(0,color=INK,lw=.9,ls='--')
    xlab=f'Δ {scale} prediction R²' if paired else f'{scale.capitalize()} prediction R²'
    style_axis(ax,xlab,tick_size=11,label_size=13.5);numeric_ticks(ax,y=False,n=4)

def fig3(preview=False):
    name='Fig03_model_comparison';path=OUT/'figures'/name
    fig,axes=plt.subplots(3,2,figsize=(14.6,16.7));reg=Registry();a,b,c,d,e,f=axes.flat
    numbered={m:f'{i+1}  {LABEL[m]}' for i,m in enumerate(CORE)}
    offsets=[(-8,12),(10,-17),(-12,12),(10,10),(-30,-15),(-49,-17),(9,12)]
    for i,(mod,offset) in enumerate(zip(CORE,offsets)):
        row=M.loc[mod]
        point(a,row.R2_daily,row.R2_cycle,mod,reg,s=150 if mod=='AffineReconciled' else 105,
              marker=MODEL_MARKER.get(mod,'o'))
        a.annotate(str(i+1),(row.R2_daily,row.R2_cycle),xytext=offset,textcoords='offset points',
                   fontsize=12,fontweight='bold',ha='center',arrowprops=dict(arrowstyle='-',color=INK,lw=.65))
    for src,dst in [('OriginalLightGBM','MeanReconciled'),('MeanReconciled','AffineReconciled')]:
        a.annotate('',(M.loc[dst,'R2_daily'],M.loc[dst,'R2_cycle']),
                   (M.loc[src,'R2_daily'],M.loc[src,'R2_cycle']),
                   arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.1,shrinkA=10,shrinkB=10))
    a.set_xlim(.266,.357);a.set_ylim(.605,.845)
    a.set_xticks([.28,.30,.32,.34]);a.set_yticks([.65,.70,.75,.80])
    style_axis(a,'Daily prediction R²','Cycle prediction R²',tick_size=11,label_size=13.5)
    a.text(.03,.03,'7 schemes · 8 held-out cycles',transform=a.transAxes,fontsize=10)
    forest(b,CORE,numbered,'cycle',reg);b.set_xlim(-.05,.98)
    models=['OriginalLightGBM','AffineReconciled']
    left=C.loc[C.model.eq(models[0])].set_index(['season','treatment'])
    right=C.loc[C.model.eq(models[1])].set_index(['season','treatment'])
    for key,row in left.iterrows():
        c.plot([row.y,row.y],[row.prediction,right.loc[key,'prediction']],color=INK,lw=.65,alpha=.5,zorder=1)
    for mod in models:
        g=C.loc[C.model.eq(mod)]
        for treatment,marker in [('AN','o'),('PS','^')]:
            p=g.loc[g.treatment.eq(treatment)]
            point(c,p.y,p.prediction,mod,reg,s=75,marker=marker,linewidths=.8)
    c.plot([0,30],[0,30],color=INK,ls='--',lw=1)
    c.set(xlim=(0,30),ylim=(0,30));c.set_xticks([0,10,20,30]);c.set_yticks([0,10,20,30])
    style_axis(c,f'Observed cycle mean\n({UNIT})',f'Predicted cycle mean\n({UNIT})',tick_size=11,label_size=12.8)
    hs=[Line2D([],[],color=COLOR[m],marker='o',ls='',markeredgecolor=INK,ms=7) for m in models]
    hs += [Line2D([],[],color=INK,marker=k,ls='',mfc='white',ms=6) for k in ['o','^']]
    reg.legend(c,hs,[LABEL[m] for m in models]+['Ammonium nitrate','Pig slurry'],models+[None,None],
               loc='upper left',fontsize=9.2,handletextpad=.5)
    c.text(.96,.05,'16 means per model',transform=c.transAxes,ha='right',fontsize=10.7)
    errors=read(RES/'per_cycle_errors.csv')
    featured=['OriginalLightGBM','DecoupledPulse','AffineReconciled']
    for mod,ls in zip(featured,['--',':','-']):
        g=errors.loc[errors.model.eq(mod)].sort_values('season')
        line(d,g.season,np.sqrt(g.MSE),mod,reg,lw=2.1,ls=ls,marker=MODEL_MARKER[mod],
             ms=9 if mod=='DecoupledPulse' else 6.5,markeredgecolor=INK,markeredgewidth=.65,
             outlined=mod=='AffineReconciled')
    d.set_xticks([2014,2016,2018,2020,2021]);d.set_ylim(0,5.6)
    style_axis(d,'Held-out management cycle',f'Cycle-mean RMSE ({UNIT})',tick_size=11,label_size=12.8);numeric_ticks(d,x=False,n=4)
    hs=[Line2D([],[],color=COLOR[m],marker=MODEL_MARKER[m],markeredgecolor=INK,lw=2) for m in featured]
    reg.legend(d,hs,[LABEL[m] for m in featured],featured,loc='upper center',bbox_to_anchor=(.5,1.27),fontsize=10,ncol=1)
    save(M.loc[CORE].reset_index(),path/'core_scores.csv')
    save(C.loc[C.model.isin(models)],path/'observed_vs_predicted_cycle_means.csv')
    save(errors.loc[errors.model.isin(featured)],path/'per_cycle_errors.csv')
    from diagnostic_panels import joint_density
    settings=[joint_density(e,'OriginalLightGBM',reg),joint_density(f,'AffineReconciled',reg)]
    dump(settings,path/'joint_density_settings.json')
    finish(fig,axes,name,reg,[RES/'all_model_metrics.csv',RES/'all_cycle_predictions.csv',RES/'per_cycle_errors.csv'],preview=preview,gap=(.055,.075))

def fig4(preview=False):
    name='Fig04_ablation_and_windows';path=OUT/'figures'/name
    fig,axes=plt.subplots(3,2,figsize=(14.7,16));reg=Registry();a,b,c,d,e,f=axes.flat
    forest(a,ABL,ALABEL,'cycle',reg,paired=True);forest(b,ABL,ALABEL,'daily',reg,paired=True)
    for i,mod in enumerate(ABL):
        y=len(ABL)-1-i
        for component,marker,offset in [('level_sse','o',.12),('shape_sse','s',-.12)]:
            ratio=SSE.loc[mod,component]/SSE.loc['MeanReconciled',component]
            point(c,ratio,y+offset,mod,reg,s=57,marker=marker)
        point(d,100*M.loc[mod,'continuous_window_WAPE'],y,mod,reg,s=70,marker=MODEL_MARKER.get(mod,'o'))
    for ax in [c,d]:
        ax.set_yticks(range(len(ABL)),[ALABEL[m] for m in ABL[::-1]]);ax.set_ylim(-.65,len(ABL)-.35)
    c.axvline(1,color=INK,ls='--',lw=.9)
    hs=[Line2D([],[],color=INK,marker=k,ls='',mfc='white',ms=7) for k in ['o','s']]
    c.legend(hs,['Cycle level','Daily shape'],loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,fontsize=10.2)
    style_axis(c,'SSE / mean-reconciliation SSE',tick_size=11,label_size=13);numeric_ticks(c,y=False,n=4)
    baseline=100*M.loc['OriginalLightGBM','continuous_window_WAPE']
    reg.add('OriginalLightGBM',d.axvline(baseline,color=COLOR['OriginalLightGBM'],ls='--',lw=1.4))
    d.set_xlim(54,62.5)
    d.set_ylim(-1.2,len(ABL)-.35)
    d.text(.97,.03,'147 observed windows',transform=d.transAxes,ha='right',va='bottom',fontsize=10.4)
    d.text(.02,1.04,'Dashed: LightGBM',transform=d.transAxes,fontsize=10.2)
    style_axis(d,'Window integral WAPE (%)',tick_size=11,label_size=13);numeric_ticks(d,y=False,n=4)
    # Changes in the estimator itself, separate from changes in prediction error.
    observed=D.dropna(subset=['y']).copy()
    unit=['season','treatment']
    base=observed.loc[observed.model.eq('TabPFN_DailyRaw')].set_index('row_id')
    base_shape=base.prediction-base.groupby(unit).prediction.transform('mean')
    base_level=C.loc[C.model.eq('OriginalLightGBM')].set_index(unit).prediction
    adjustments=[]
    for i,mod in enumerate(ABL):
        g=observed.loc[observed.model.eq(mod)].set_index('row_id').reindex(base.index)
        centered=g.prediction-g.groupby(unit).prediction.transform('mean')
        delta_shape=centered-base_shape
        levels=C.loc[C.model.eq(mod)].set_index(unit).prediction-base_level
        for year in range(2014,2022):
            adjustments.append(dict(model=mod,season=year,
                level_rms=float(np.sqrt(np.mean(levels.loc[year]**2))),
                shape_rms=float(np.sqrt(np.mean(delta_shape.loc[g.season.eq(year)]**2))),
                n_obs=int(g.season.eq(year).sum())))
        part=pd.DataFrame(adjustments).query('model==@mod').sort_values('season')
        for ax,component in [(e,'level'),(f,'shape')]:
            vals=part[component+'_rms'].to_numpy()
            weights=np.ones(8) if component=='level' else part.n_obs.to_numpy()
            pooled=np.sqrt(np.average(vals**2,weights=weights))
            yy=len(ABL)-1-i
            line(ax,[vals.min(),vals.max()],[yy,yy],mod,reg,lw=1.2)
            point(ax,vals,yy+np.linspace(-.16,.16,8),mod,reg,s=22,linewidths=.5,alpha=.75)
            point(ax,pooled,yy,mod,reg,marker='D',s=60)
    for ax,component,reference in [(e,'Cycle-level','LightGBM means'),(f,'Daily-shape','Centered TabPFN')]:
        ax.set_yticks(range(len(ABL)),[ALABEL[m] for m in ABL[::-1]]);ax.set_ylim(-.65,len(ABL)-.35)
        ax.axvline(0,color=INK,ls='--',lw=.8)
        ax.text(.02,1.04,f'Reference: {reference}',transform=ax.transAxes,fontsize=10.2)
        style_axis(ax,f'{component} adjustment RMS\n({UNIT})',tick_size=11,label_size=12.3)
        numeric_ticks(ax,y=False,n=4)
        ax.set_xlim(left=-.05*max(ax.get_xlim()[1],.1))
    save(pd.DataFrame(adjustments),path/'component_adjustments_by_cycle.csv')
    save(M.loc[ABL].reset_index(),path/'ablation_scores.csv')
    save(PAIR.loc[PAIR.model.isin(ABL)&PAIR.reference.eq('MeanReconciled')&PAIR.scale.isin(['cycle','daily'])],path/'paired_differences.csv')
    save(SSE.loc[ABL].reset_index(),path/'sse_ratios_source.csv')
    finish(fig,axes,name,reg,[RES/'all_model_metrics.csv',RES/'paired_R2_differences.csv',RES/'SSE_decomposition.csv',RES/'continuous_observed_windows.csv',RES/'all_daily_predictions.csv',RES/'all_cycle_predictions.csv'],preview=preview,gap=(.045,.065))

def figs1(preview=False):
    name='FigS01_observation_support';path=OUT/'figures'/name
    fig,axes=plt.subplots(1,2,figsize=(14,8.8));fig._match_row_plot_heights=True;reg=Registry();a,b=axes
    base=PRED['OriginalLightGBM'];rows=[];labels=[]
    for i,(year,treatment) in enumerate([(y,t) for y in range(2014,2022) for t in ['AN','PS']]):
        g=base.query('season==@year and treatment==@treatment').copy()
        g['cycle_day']=(g.Date-pd.Timestamp(MAN.loc[year,'start'])).dt.days
        yy=15-i;labels.append(f'{year} {treatment}')
        for state,mask in [('Missing',g.y.isna()),('Observed',g.y.notna())]:
            # One square per calendar day; no interpolation of targets.
            reg.add(state,a.scatter(g.loc[mask,'cycle_day'],np.full(mask.sum(),yy),c=COLOR[state],marker='s',s=12,edgecolors='none'))
        n=int(g.y.notna().sum());nd=len(g);fraction=100*n/nd
        point(b,fraction,yy,'Observed',reg,s=75)
        b.text(103,yy,f'{n}/{nd}',va='center',fontsize=10.4)
        rows.append(dict(season=year,treatment=treatment,n_observed=n,n_calendar=nd,coverage_pct=fraction))
    for ax in axes:
        ax.set_yticks(range(16),labels[::-1]);ax.set_ylim(-.7,15.7)
    a.set_xlim(-5,420);a.set_xticks([0,100,200,300,400])
    style_axis(a,'Days since cycle start',tick_size=11,label_size=14)
    hs=[Line2D([],[],ls='',marker='s',color=COLOR[k],ms=7) for k in ['Observed','Missing']]
    reg.legend(a,hs,['Observed','Missing'],['Observed','Missing'],loc='upper center',bbox_to_anchor=(.5,1.12),ncol=2,fontsize=11)
    b.set_xlim(0,122);b.set_xticks([0,25,50,75,100])
    style_axis(b,'Observed days (%)',tick_size=11,label_size=14)
    b.text(103,16.15,'n / days',ha='left',fontsize=10.5)
    save(pd.DataFrame(rows),path/'support_counts.csv')
    finish(fig,axes,name,reg,[RES/'all_daily_predictions.csv',RES/'season_manifest.csv'],preview=preview,gap=(.06,.05))

def figs2(preview=False):
    name='FigS02_daily_trajectories';path=OUT/'figures'/name
    fig,axes=plt.subplots(4,4,figsize=(17.5,15.8));fig._match_row_plot_heights=True;reg=Registry();records=[]
    for i,(year,treatment) in enumerate([(y,t) for y in range(2014,2022) for t in ['AN','PS']]):
        ax=axes.flat[i];start=pd.Timestamp(MAN.loc[year,'start'])
        g=PRED['OriginalLightGBM'].query('season==@year and treatment==@treatment').copy()
        days=(g.Date-start).dt.days
        point(ax,days,g.y,'Observed',reg,s=11,linewidths=.25)
        for mod,ls in [('OriginalLightGBM','--'),('AffineReconciled','-')]:
            gg=PRED[mod].query('season==@year and treatment==@treatment')
            line(ax,days,gg.prediction,mod,reg,lw=1.3,ls=ls,outlined=mod=='AffineReconciled')
            g[mod]=gg.prediction.to_numpy()
        g['day_since_start']=days;records.append(g)
        # Same vertical range for the two treatments of a given cycle.
        both=PRED['OriginalLightGBM'].query('season==@year')
        allvals=np.r_[both.y.dropna(),both.prediction,PRED['AffineReconciled'].query('season==@year').prediction]
        ymin=min(-5,float(np.min(allvals))-3);ymax=float(np.max(allvals))*1.12+5
        ax.set_ylim(ymin,ymax);ax.set_xlim(0,int(MAN.loc[year,'days'])-1)
        ax.text(.5,1.025,f'{year} · {treatment}',transform=ax.transAxes,ha='center',va='bottom',fontsize=11.5,fontweight='bold')
        ax.set_xticks([0,100,200,300,400] if len(g)>400 else [0,100,200,300])
        numeric_ticks(ax,x=False,n=3)
        style_axis(ax,'Cycle day','N$_2$O-N flux\n(g N ha$^{-1}$ d$^{-1}$)',tick_size=10.3,label_size=11.3)
    # Shared key has its own reserved band; it must not compress panel a.
    hs=[Line2D([],[],color=COLOR['Observed'],marker='o',ls='',ms=4,markeredgecolor=INK),
        Line2D([],[],color=COLOR['OriginalLightGBM'],ls='--',lw=1.7),
        Line2D([],[],color=COLOR['AffineReconciled'],lw=1.7)]
    leg=fig.legend(hs,['Observed','LightGBM','Affine'],loc='upper center',bbox_to_anchor=(.5,.995),
                   ncol=3,fontsize=14,handlelength=1.3,columnspacing=1)
    for role,artist in zip(['Observed','OriginalLightGBM','AffineReconciled'],leg.legend_handles):reg.add(role,artist)
    fig._outer_bounds=(.025,.025,.985,.945)
    save(pd.concat(records),path/'daily_trajectories.csv')
    finish(fig,axes,name,reg,[RES/'all_daily_predictions.csv',RES/'season_manifest.csv'],preview=preview,gap=(.028,.04))

def figs3(preview=False):
    name='FigS03_loss_and_shrinkage';path=OUT/'figures'/name
    fig,axes=plt.subplots(2,2,figsize=(12,9));reg=Registry();m07=read(R07/'all_model_metrics.csv').set_index('model')
    records=[]
    for row,(models,table,role,param) in enumerate([
        (['Joint01','Joint1','Joint10'],m07,'Joint01','Cycle-loss weight'),
        (['DecoupledPulseWeak','DecoupledPulse','DecoupledPulseStrong'],M,'DecoupledPulse','Shrinkage weight')]):
        for col,scale in enumerate(['cycle','daily']):
            ax=axes[row,col];xx=[.1,1,10];yy=[table.loc[m,f'R2_{scale}'] for m in models]
            line(ax,xx,yy,role,reg,lw=2.2,marker='o',ms=7,markeredgecolor=INK,markeredgewidth=.7,outlined=True)
            ref=M.loc['OriginalLightGBM',f'R2_{scale}']
            reg.add('OriginalLightGBM',ax.axhline(ref,color=COLOR['OriginalLightGBM'],lw=1.4,ls='--'))
            ax.set_xscale('log');ax.set_xticks(xx,['0.1','1','10']);ax.minorticks_off();ax.set_xlim(.07,14)
            if row==0: ax.set_ylim((.1,.85) if col==0 else (-.03,.4))
            else: ax.set_ylim((.70,.84) if col==0 else (.26,.365))
            numeric_ticks(ax,x=False,n=4);style_axis(ax,param,f'{scale.capitalize()} prediction R²',tick_size=12,label_size=14)
            ax.text(.04,.05,'Dashed: LightGBM',transform=ax.transAxes,fontsize=10.5)
            records.extend(dict(model=m,parameter=param,weight=x,scale=scale,R2=y) for m,x,y in zip(models,xx,yy))
    save(pd.DataFrame(records),path/'sensitivity_points.csv')
    finish(fig,axes,name,reg,[R07/'all_model_metrics.csv',RES/'all_model_metrics.csv'],preview=preview)

def figs4(preview=False):
    name='FigS04_pulse_response';path=OUT/'figures'/name
    fig,axes=plt.subplots(2,2,figsize=(12.5,9.6));reg=Registry()
    taus=np.array([3,7,14,30,90]);days=np.arange(181)
    kernel=(-np.expm1(-1/taus))[None,:]*np.exp(-days[:,None]/taus)
    weights=[];responses=[]
    for year in range(2014,2022):
        model=json.loads((R08/f'models/{year}_DecoupledPulse.json').read_text(encoding='utf-8'))
        for branch,offset,scale,norm in [('level',2,'sl','norm_l'),('shape',1,'sd','norm_s')]:
            beta=np.array(model['beta_l' if branch=='level' else 'beta_s'])[offset:]*model[scale]/np.array(model[norm])
            for form,j in [('Nmin',0),('Norg',5)]:
                weights.extend(dict(held=year,branch=branch,form=form,tau=int(tau),coefficient=float(v)) for tau,v in zip(taus,beta[j:j+5]))
                if branch=='shape':
                    rr=kernel@beta[j:j+5]
                    responses.extend(dict(held=year,form=form,day=int(day),response=float(v)) for day,v in zip(days,rr))
    weights=pd.DataFrame(weights);responses=pd.DataFrame(responses)
    for ax,form in zip(axes[0],['Nmin','Norg']):
        rr=responses.loc[responses.form.eq(form)].pivot(index='day',columns='held',values='response')
        line(ax,days,rr.median(axis=1),form,reg,lw=2.6,marker='o' if form=='Nmin' else 's',markevery=[0,30,60,90,120,150,180],ms=4,markeredgecolor=INK,markeredgewidth=.5,outlined=True)
        for values in [rr.min(axis=1),rr.max(axis=1)]: line(ax,days,values,form,reg,lw=1.5,ls=':',outlined=True)
        upper=float(rr.max().max());ax.set_xlim(0,180);ax.set_ylim(-.035*upper,1.1*upper);numeric_ticks(ax,n=4)
        style_axis(ax,'Days after application',f'Pulse increment ({UNIT})',tick_size=11.5,label_size=13)
        ax.text(.97,.95,('Mineral N' if form=='Nmin' else 'Organic N')+'\n1 kg N ha$^{-1}$\n20 °C · 60% WFPS',transform=ax.transAxes,ha='right',va='top',fontsize=10.5)
    for ax,branch in zip(axes[1],['level','shape']):
        for form,marker in [('Nmin','o'),('Norg','s')]:
            rr=weights.query('branch==@branch and form==@form').pivot(index='tau',columns='held',values='coefficient')
            line(ax,taus,rr.median(axis=1),form,reg,lw=2,marker=marker,ms=6,markeredgecolor=INK,markeredgewidth=.6,outlined=True)
            for xx,lo,hi in zip(taus,rr.min(axis=1),rr.max(axis=1)):
                line(ax,[xx,xx],[lo,hi],form,reg,lw=1.5,ls=':',outlined=True)
        ax.set_xscale('log');ax.set_xticks(taus,[str(v) for v in taus]);ax.minorticks_off()
        upper=float(weights.loc[weights.branch.eq(branch),'coefficient'].max());ax.set_ylim(-.035*upper,1.1*upper)
        numeric_ticks(ax,x=False,n=4)
        style_axis(ax,'Kernel time scale (days)','Kernel coefficient\n(g N$_2$O-N kg$^{-1}$ N d$^{-1}$)',tick_size=11.5,label_size=13)
        ax.text(.03,.96,f'{branch.capitalize()} branch',transform=ax.transAxes,fontsize=10.5,va='top')
        hs=[Line2D([],[],color=COLOR[k],marker=m,markeredgecolor=INK,lw=2) for k,m in [('Nmin','o'),('Norg','s')]]
        reg.legend(ax,hs,['Mineral N','Organic N'],['Nmin','Norg'],loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,fontsize=10.5)
    save(weights,path/'physical_coefficients.csv');save(responses,path/'unit_responses.csv')
    finish(fig,axes,name,reg,sorted((R08/'models').glob('*_DecoupledPulse.json'))+[R08/'scripts/common.py'],preview=preview)

def figs5(preview=False):
    name='FigS05_observed_windows';path=OUT/'figures'/name
    fig,axes=plt.subplots(1,2,figsize=(13,6.1));reg=Registry();a,b=axes
    models=['OriginalLightGBM','AffineReconciled']
    for mod,marker in zip(models,['o','^']):
        g=W.loc[W.model.eq(mod)]
        point(a,g.observed_integral,g.predicted_integral,mod,reg,s=35,marker=marker,alpha=.82,linewidths=.5)
    values=W.loc[W.model.isin(models),['observed_integral','predicted_integral']].to_numpy()
    assert values.min()>-160 and values.max()<5000
    a.plot([-160,5000],[-160,5000],color=INK,ls='--',lw=1)
    a.set_xscale('symlog',linthresh=50);a.set_yscale('symlog',linthresh=50)
    a.set_xlim(-160,5000);a.set_ylim(-160,5000)
    ticks=[-100,0,50,200,1000,4000];a.set_xticks(ticks,[str(v) for v in ticks]);a.set_yticks(ticks,[str(v) for v in ticks]);a.minorticks_off()
    style_axis(a,f'Observed window sum ({SUMUNIT})',f'Predicted window sum ({SUMUNIT})',tick_size=11.5,label_size=13)
    hs=[Line2D([],[],color=COLOR[m],marker=k,markeredgecolor=INK,ls='',ms=6) for m,k in zip(models,['o','^'])]
    reg.legend(a,hs,[LABEL[m] for m in models],models,loc='upper left',fontsize=10.5)
    keys=['season','treatment','start','end','n_days']
    left=W.loc[W.model.eq('OriginalLightGBM')].set_index(keys)
    right=W.loc[W.model.eq('AffineReconciled')].set_index(keys)
    pd.testing.assert_index_equal(left.index,right.index)
    delta=(right.abs_error-left.abs_error).rename('delta_absolute_integral_error').to_frame().reset_index()
    delta['delta_absolute_mean_error']=delta.delta_absolute_integral_error/delta.n_days
    point(b,delta.n_days,delta.delta_absolute_mean_error,'AffineReconciled',reg,s=45,alpha=.85,linewidths=.6)
    b.axhline(0,color=INK,ls='--',lw=1);b.set_xlim(4,92);numeric_ticks(b,n=4)
    style_axis(b,'Continuous observed days',f'Δ absolute window-mean error\n({UNIT})',tick_size=11.5,label_size=13)
    b.text(.98,.96,'Affine minus LightGBM',transform=b.transAxes,ha='right',va='top',fontsize=10.5)
    save(W.loc[W.model.isin(models)],path/'observed_window_sums.csv');save(delta,path/'paired_window_error_changes.csv')
    finish(fig,axes,name,reg,[RES/'continuous_observed_windows.csv'],preview=preview)

BUILDERS={'Fig01':fig1,'Fig02':fig2,'Fig03':fig3,'Fig04':fig4,'FigS01':figs1,'FigS02':figs2,'FigS03':figs3,'FigS04':figs4,'FigS05':figs5}

def verify_sources():
    """Recompute displayed metrics and algebra from saved predictions."""
    result=[]
    for mod,g in PRED.items():
        observed=g.dropna(subset=['y'])
        assert len(g)==5970 and len(observed)==3319
        r2=1-np.sum((observed.y-observed.prediction)**2)/np.sum((observed.y-observed.y.mean())**2)
        assert np.isclose(r2,M.loc[mod,'R2_daily'],rtol=0,atol=1e-12)
        means=observed.groupby(['season','treatment'])[['y','prediction']].mean()
        cr2=1-np.sum((means.y-means.prediction)**2)/np.sum((means.y-means.y.mean())**2)
        assert np.isclose(cr2,M.loc[mod,'R2_cycle'],rtol=0,atol=1e-12)
        w=W.loc[W.model.eq(mod)]
        assert len(w)==147
        wape=w.abs_error.sum()/w.absolute_observed_sum.sum()
        assert np.isclose(wape,M.loc[mod,'continuous_window_WAPE'],rtol=0,atol=1e-12)
        assert np.isclose(SSE.loc[mod,'level_sse']+SSE.loc[mod,'shape_sse'],np.sum((observed.y-observed.prediction)**2))
        result.append(dict(model=mod,daily_R2=r2,cycle_R2=cr2,window_WAPE=wape))
    a=PRED['AffineReconciled'].dropna(subset=['y']).copy();t=PRED['TabPFN_DailyRaw'].dropna(subset=['y']).copy()
    key=['season','treatment'];aa=a.prediction-a.groupby(key).prediction.transform('mean');tt=t.prediction-t.groupby(key).prediction.transform('mean')
    err=float(np.max(np.abs(aa.to_numpy()-tt.to_numpy())))
    assert err<1e-10
    high=PRED['OriginalLightGBM'].nlargest(5,'y')
    assert (high.season.eq(2014)&high.treatment.eq('AN')&high.y.gt(300)).all()
    dump(dict(passed=True,models=len(result),max_shape_identity_error=err,checked_metrics=result,
              extreme_observation_days=high.Date.dt.strftime('%Y-%m-%d').tolist(),
              source_hashes={str(p):sha(p) for p in SOURCE_FILES}),OUT/'qa/numerical_verification.json')
    save(M.reset_index(),OUT/'tables/all_18_model_metrics.csv')
    save(read(R07/'all_model_metrics.csv'),OUT/'tables/round07_loss_controls.csv')
    dump(validate_series_colors(PALETTE,COLOR),OUT/'qa/project_series_colors.json')

def main():
    from interpretation_figures import fig5,figs6,figs7
    BUILDERS.update(Fig05=fig5,FigS06=figs6,FigS07=figs7)
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--only',nargs='+',choices=list(BUILDERS))
    args=parser.parse_args();configure_publication_style();plt.rcParams.update({'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
    verify_sources()
    for name in args.only or BUILDERS: BUILDERS[name](preview=args.preview)

if __name__=='__main__': main()
