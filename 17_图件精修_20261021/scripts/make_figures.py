"""Round-17 refined manuscript and SI figures (no model fitting, frozen results only).

Run from this script or use reproduce_figures.py at repository root.
Data are read from rounds 07/08/11; nothing is written back to those folders.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle,
                                Wedge)
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent
R11 = ROOT / '11_图件诊断补强_20261018'
sys.path.insert(0, str(R11 / 'scripts'))
sys.path.insert(0, str(HERE))
import generate_paper_figures as b  # noqa: E402  (loads frozen round-08 predictions; draws nothing)
import figkit as K  # noqa: E402
from figkit import (BLUE, GOLD, GREEN, INK, LIGHT, MAUVE, MUTED, PINK, PURPLE, ROSE, TEAL, GRID,  # noqa: E402
                    COLOR, MARKER, BASE, TICK, style, nticks, bands, outlined, Layout, grid_cells)

M, D, C, W, SSE, PAIR, MAN, EVENTS, PRED = b.M, b.D, b.C, b.W, b.SSE, b.PAIR, b.MAN, b.EVENTS, b.PRED
RES8, R08, R07 = b.RES, b.R08, b.R07
RES11, TAB11, GEO = R11 / 'results', R11 / 'tables', R11 / 'evidence/geography'
UNIT = 'g N$_2$O-N ha$^{-1}$ d$^{-1}$'
SUMUNIT = 'g N$_2$O-N ha$^{-1}$'
CORE = b.CORE
LABEL = dict(b.LABEL)
ABL = b.ABL
ALABEL = dict(b.ALABEL)
ALABEL['DecoupledPulse'] = 'Decoupled + pulse'  # same label as Fig. 4 (round 23)
FIGDIR = Path(__import__('os').environ.get('EFELE_FIGURE_OUTPUT', str(OUT/'figures')))
QA = FIGDIR / '_qa'
PREVIEW = False


def read(p):
    return pd.read_csv(p, float_precision='round_trip')


def save(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def pt(ax, x, y, role, **kw):
    d = dict(s=60, edgecolors=INK, linewidths=.7, zorder=4)
    d.update(kw)
    return ax.scatter(x, y, color=COLOR[role], **d)


def proxy(role, marker='o', ls='', ms=8, lw=2, **kw):
    return Line2D([], [], color=COLOR[role], marker=marker, ls=ls, ms=ms, lw=lw,
                  markeredgecolor=INK, markeredgewidth=.7, **kw)


def finish(fig, lay, name, sources, *, legend_axes=None):
    folder = FIGDIR / name
    folder.mkdir(parents=True, exist_ok=True)
    al = lay.alignment()
    au = K.audit(fig)
    K.dump(dict(alignment=al, palette=au), folder / 'qa_record.json')
    if getattr(fig, '_free_alignment', False):
        al['note'] = 'Plot rectangles placed on an exact shared grid; outer boxes differ only by tick-label width.'
    elif not al['passed']:
        print('  ! alignment', al)
    if not au['passed']:
        print('  ! palette', au['off_palette'][:5])
    QA.mkdir(parents=True, exist_ok=True)
    fig.savefig(QA / f'{name}_preview.png', dpi=70, facecolor='white')
    if not PREVIEW:
        K.export(fig, lay, folder, name, legend_axes=legend_axes)
        K.dump(dict(figure=name, sources=[str(s) for s in sources], palette=K.PALETTE,
                    author='CHONG LIU', note='Restyled from frozen results; no refit.'), folder / 'export_record.json')
    plt.close(fig)
    print(f'{name}: min font at 190 mm = {au["min_font_pt_at_190mm"]:.2f} pt; '
          f'align {al["row_outer_misalignment_pt"]:.2f}/{al["column_outer_misalignment_pt"]:.2f} pt', flush=True)


# =========================================================================== Fig. 1
def icon_sun(ax, x, y, r=1.6):
    ax.add_patch(Circle((x, y), r, fc=GOLD, ec=INK, lw=.9, zorder=6))
    for ang in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([x + 1.45 * r * np.cos(ang), x + 2.0 * r * np.cos(ang)],
                [y + 1.45 * r * np.sin(ang), y + 2.0 * r * np.sin(ang)], color=INK, lw=.9, zorder=6)


def icon_cloud(ax, x, y, s=1.0):
    for dx, dy, r in [(-1.5, 0, 1.25), (0, .7, 1.6), (1.6, 0, 1.2)]:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, fc='white', ec=INK, lw=.9, zorder=6))
    ax.add_patch(Rectangle((x - 1.5 * s, y - 1.25 * s), 3.1 * s, 1.3 * s, fc='white', ec='none', zorder=7))
    ax.plot([x - 2.75 * s, x + 2.8 * s], [y - 1.2 * s, y - 1.2 * s], color=INK, lw=.9, zorder=7)
    for dx in [-1.4, 0, 1.4]:
        ax.plot([x + dx * s, x + (dx - .35) * s], [y - 1.9 * s, y - 2.9 * s], color=BLUE, lw=1.6, zorder=6,
                solid_capstyle='round')


def icon_soil(ax, x, y, w=8, h=5):
    cols = [GREEN, GOLD, '#EEF0F2']
    for i, c in enumerate(cols):
        ax.add_patch(Rectangle((x, y + h - (i + 1) * h / 3), w, h / 3, fc=c, ec=INK, lw=.7, zorder=5))
    # a plant on the surface
    ax.plot([x + w * .5, x + w * .5], [y + h, y + h + 3.2], color=INK, lw=1.0, zorder=6)
    for ang, dx in [(35, -1.1), (-35, 1.1)]:
        ax.add_patch(Ellipse((x + w * .5 + dx, y + h + 2.2), 2.4, .95, angle=ang, fc=GREEN, ec=INK, lw=.7, zorder=6))
    ax.plot([x + w * .25, x + w * .25], [y + h * .55, y + h * .9], color=BLUE, lw=1.6, zorder=6)


def icon_drop(ax, x, y, s=1.2, fc=PURPLE):
    th = np.linspace(0, 2 * np.pi, 60)
    xs = x + s * np.sin(th) * (1 - np.cos(th)) * .55
    ys = y + s * (-np.cos(th)) * 1.0
    ax.add_patch(Polygon(np.c_[xs, ys], closed=True, fc=fc, ec=INK, lw=.8, zorder=6))


def icon_tree(ax, x, y, s=1.0):
    nodes = {0: (0, 2.2), 1: (-1.6, .9), 2: (1.6, .9), 3: (-2.3, -.4), 4: (-.9, -.4), 5: (.9, -.4), 6: (2.3, -.4)}
    for a, bb in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        ax.plot([x + nodes[a][0] * s, x + nodes[bb][0] * s], [y + nodes[a][1] * s, y + nodes[bb][1] * s],
                color=INK, lw=.9, zorder=6)
    for k, (dx, dy) in nodes.items():
        ax.add_patch(Circle((x + dx * s, y + dy * s), .42 * s, fc='white' if k < 3 else BLUE, ec=INK, lw=.8, zorder=7))


def icon_tokens(ax, x, y, s=1.0):
    xs = np.linspace(-2.2, 2.2, 5)
    for i, xx in enumerate(xs):
        for yy in [-.8, 1.5]:
            ax.add_patch(FancyBboxPatch((x + xx * s - .35 * s, y + yy * s - .3 * s), .7 * s, .6 * s,
                                        boxstyle='round,pad=0.02,rounding_size=0.15', fc='white' if yy > 0 else TEAL,
                                        ec=INK, lw=.7, zorder=7))
    for i, xa in enumerate(xs):
        for xb in xs:
            ax.plot([x + xa * s, x + xb * s], [y - .5 * s, y + 1.2 * s], color=MUTED, lw=.35, zorder=6, alpha=.8)


def icon_calendar(ax, x, y, s=1.0):
    ax.add_patch(Rectangle((x - 2.6 * s, y - 1.8 * s), 5.2 * s, 3.6 * s, fc='white', ec=INK, lw=.8, zorder=6))
    ax.add_patch(Rectangle((x - 2.6 * s, y + 1.0 * s), 5.2 * s, .8 * s, fc=PURPLE, ec=INK, lw=.8, zorder=7))
    rng = np.random.default_rng(3)
    for i in range(6):
        for j in range(3):
            filled = rng.random() > .45
            ax.add_patch(Rectangle((x + (-2.25 + i * .78) * s, y + (-1.45 + j * .8) * s), .55 * s, .5 * s,
                                   fc=PURPLE if filled else 'white', ec=INK, lw=.45, zorder=7))


def glyph(ax, x0, y0, w, h, kind):
    """Schematic (non-data) curves inside operator boxes."""
    t = np.linspace(0, 1, 120)
    wave = .35 * np.sin(2 * np.pi * 2.2 * t) + .55 * np.exp(-((t - .62) / .07) ** 2)
    if kind == 'level':
        ax.plot([x0, x0 + w], [y0 + .25 * h] * 2, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=6)
        ax.plot([x0, x0 + w], [y0 + .72 * h] * 2, color=GOLD, lw=3.0, zorder=6, solid_capstyle='round')
        ax.add_patch(FancyArrowPatch((x0 + w * .5, y0 + .3 * h), (x0 + w * .5, y0 + .66 * h), arrowstyle='-|>',
                                     mutation_scale=10, color=INK, lw=1, zorder=7))
    elif kind == 'shape':
        ax.plot([x0, x0 + w], [y0 + .45 * h] * 2, color=INK, lw=.9, ls=(0, (3, 2)), zorder=6)
        ax.plot(x0 + w * t, y0 + .45 * h + .38 * h * (wave - wave.mean()) / .6, color=TEAL, lw=2.4, zorder=6)
    elif kind == 'output':
        ax.plot(x0 + w * t, y0 + .5 * h + .38 * h * (wave - wave.mean()) / .6, color=ROSE, lw=2.4, zorder=6)
        ax.plot([x0, x0 + w], [y0 + .5 * h] * 2, color=GOLD, lw=2.4, zorder=5)


def draw_framework(ax):
    ax.set(xlim=(0, 170), ylim=(1, 76)); ax.axis('off'); ax.set_aspect('equal')

    def box(x, y, w, h, fc, ec=INK, lw=1.0, r=2.2, alpha=1, z=2, ls='-'):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                           alpha=alpha, zorder=z, ls=ls)
        ax.add_patch(p); return p

    def txt(x, y, s, size=BASE, **kw):
        kw.setdefault('ha', 'center'); kw.setdefault('va', 'center'); kw.setdefault('color', INK)
        return ax.text(x, y, s, fontsize=size, zorder=9, **kw)

    def arrow(p0, p1, ls='-', rad=0.0, lw=1.4):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=15, color=INK, lw=lw, linestyle=ls,
                                     connectionstyle=f'arc3,rad={rad}', zorder=8, shrinkA=0, shrinkB=0))

    # 1 drivers ------------------------------------------------------------
    box(1, 22, 30, 41, '#FFFFFF', ec=INK)
    box(1, 56.5, 30, 6.5, GREEN, r=2.2)
    txt(16, 59.75, 'Agricultural drivers')
    icon_sun(ax, 6.3, 49, 1.45); icon_cloud(ax, 14.3, 49.8, 1.05)
    txt(24.2, 49, 'Weather', 13.5)
    icon_soil(ax, 2.6, 32.4, 9.5, 5.2)
    txt(22, 37.2, 'Soil T and\nwater', 13.5)
    icon_drop(ax, 7.2, 27.3, 1.9)
    txt(19.8, 26.6, 'Crop, N amount,\nform and timing', 13.5)
    # observation dates
    box(1, 3, 30, 14, '#FFFFFF')
    icon_calendar(ax, 8.3, 10, 1.2)
    txt(22, 10, 'Observation\ndates $O_c$', 13.5)

    # 2 learners -----------------------------------------------------------
    box(41, 41, 27, 21, '#FFFFFF')
    box(41, 55.5, 27, 6.5, BLUE)
    txt(54.5, 58.75, 'LightGBM')
    icon_tree(ax, 49, 46.2, 1.6)
    txt(62.5, 47, '$L_t$', 19)
    box(41, 12, 27, 21, '#FFFFFF')
    box(41, 26.5, 27, 6.5, TEAL)
    txt(54.5, 29.75, 'TabPFN')
    icon_tokens(ax, 49.5, 17.0, 1.55)
    txt(62.5, 18.6, '$T_t$', 19)
    txt(54.5, 37.2, 'Daily predictions', 13, style='italic', color=MUTED)

    # 3 operators -----------------------------------------------------------
    box(78, 41, 38, 21, '#FFFFFF')
    box(78, 55.5, 38, 6.5, GOLD)
    txt(97, 58.75, 'Cycle-level calibration')
    txt(97, 50.2, r'$\widehat{m}_c=a_h+b_h\,A_c(L)$', 15.5)
    txt(90, 44.6, r'$b_h\geq 0$', 14)
    glyph(ax, 102, 42.6, 11, 5.6, 'level')
    box(78, 12, 38, 21, '#FFFFFF')
    box(78, 26.5, 38, 6.5, TEAL)
    txt(97, 29.75, 'Centered daily shape')
    txt(92.5, 20.2, r'$C_c(T)_t=T_t-A_c(T)$', 15.5)
    glyph(ax, 104, 13.6, 10.5, 5.8, 'shape')
    # training box on top
    box(72, 64.5, 50, 11, '#FFFFFF', ls=(0, (4, 2)))
    txt(97, 72.2, r'Fit $a_h$, $b_h$ within seven training cycles', 13.5)
    txt(97, 67.6, r'shrink toward $a_h=0$, $b_h=1$', 13.5, color=MUTED)
    arrow((97, 64.5), (97, 62.1), ls=(0, (4, 2)))

    # merge + output --------------------------------------------------------
    ax.add_patch(Circle((126, 37), 3.2, fc='white', ec=INK, lw=1.4, zorder=8))
    txt(126, 37.2, '+', 20)
    box(136, 12, 33, 50, '#FFFFFF')
    box(136, 55.5, 33, 6.5, ROSE)
    txt(152.5, 58.75, 'Daily estimate')
    txt(152.5, 49.6, r'$\hat{f}_t=\widehat{m}_c+C_c(T)_t$', 15.5)
    glyph(ax, 141, 37.5, 23, 7.5, 'output')
    ax.plot([140, 165], [34.2, 34.2], color=GRID, lw=1)
    txt(152.5, 29.7, r'$A_c(\hat{f})=\widehat{m}_c$', 15)
    txt(152.5, 23.5, r'$C_c(\hat{f})=C_c(T)$', 15)
    txt(152.5, 16.3, 'level calibrated,\nshape preserved', 13, color=MUTED, style='italic')

    # arrows ----------------------------------------------------------------
    arrow((31, 47), (41, 51)); arrow((31, 37), (41, 23))
    arrow((68, 51), (78, 51)); arrow((68, 22.5), (78, 22.5))
    arrow((116, 51), (123.6, 39.4)); arrow((116, 22.5), (123.6, 34.6))
    arrow((129.2, 37), (136, 37))
    # observation-date support (dotted) to both operators
    ax.plot([31, 73, 73], [10, 10, 44], color=INK, lw=1.2, ls=':', zorder=7)
    arrow((73, 44), (78, 44), ls=':'); arrow((73, 15.5), (78, 15.5), ls=':')
    txt(152.5, 6.0, r'$A_c(v)$: mean of $v_t$ over $O_c$;  $h$: held-out cycle', 13)
    # key
    for i, (ls, lab) in enumerate([('-', 'Prediction'), (':', 'Date support'), ((0, (4, 2)), 'Fitted parameters')]):
        x = 38 + i * 26
        ax.add_patch(FancyArrowPatch((x, 6), (x + 5.2, 6), arrowstyle='-|>', mutation_scale=12, color=INK, lw=1.2,
                                     linestyle=ls, zorder=8))
        txt(x + 6.4, 6, lab, 13, ha='left')


def map_panel(ax):
    countries = json.loads((GEO / 'ne_110m_admin_0_countries.geojson').read_text(encoding='utf-8'))
    ax.set_facecolor('white')
    for f in countries['features']:
        g = f['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for rings in polys:
            ax.add_patch(Polygon(np.array(rings[0]), closed=True, fc='#EEF0F2', ec='#B7B7B7', lw=.6, zorder=1))
    ax.set_xlim(-6, 17); ax.set_ylim(42.4, 54)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.5)))
    sites = read(GEO / 'site_locations.csv')
    off = {'EFELE': (-2.6, 1.9), 'Nouzilly': (3.4, 1.3), 'Grabenegg': (-1.6, 1.9)}
    for r in sites.itertuples():
        dev = r.status == 'Development'
        ax.scatter(r.longitude, r.latitude, s=190 if dev else 120, marker='o' if dev else '^',
                   color=COLOR['Support'] if dev else COLOR['External'], edgecolors=INK, linewidths=1, zorder=6)
        dx, dy = off[r.site]
        ax.annotate(r.site, (r.longitude, r.latitude), xytext=(r.longitude + dx, r.latitude + dy), fontsize=BASE,
                    ha='center', va='center', zorder=7,
                    arrowprops=dict(arrowstyle='-', color=INK, lw=.7, shrinkA=2, shrinkB=6))
    for x, y, s in [(2.6, 45.6, 'France'), (10.3, 51.4, 'Germany'), (14.0, 46.6, 'Austria')]:
        ax.text(x, y, s, fontsize=TICK, color=MUTED, ha='center', style='italic', zorder=3)
    ax.set_xticks([-5, 0, 5, 10, 15], ['5°W', '0°', '5°E', '10°E', '15°E'])
    ax.set_yticks([44, 48, 52], ['44°N', '48°N', '52°N'])
    style(ax)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(True); ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(1.0)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.0)
    hs = [Line2D([], [], marker='o', ls='', color=COLOR['Support'], markeredgecolor=INK, ms=10),
          Line2D([], [], marker='^', ls='', color=COLOR['External'], markeredgecolor=INK, ms=9)]
    leg = ax.legend(hs, ['Development site', 'External candidate'], loc='lower left', fontsize=TICK,
                    frameon=True, framealpha=1, edgecolor='none', borderpad=.5)
    leg.set_zorder(10)
    # 100 km scale bar at 49°N
    km = 100 / (111.32 * np.cos(np.deg2rad(49)))
    ax.plot([12, 12 + km], [43.2, 43.2], color=INK, lw=2.2, solid_capstyle='butt', zorder=6)
    ax.text(12 + km / 2, 43.5, '100 km', ha='center', va='bottom', fontsize=TICK, zorder=6)


def timeline_panel(ax):
    key = '_'.join(map(str, range(2015, 2022)))
    fd = read(R08 / 'fold_data' / f'{key}.csv'); fd['Date'] = pd.to_datetime(fd.Date)
    crops = {'crop_WHEAT': 'Wheat', 'crop_MAIZE': 'Maize', 'crop_MUSTARD': 'Mustard'}
    seasons = list(MAN.iterrows())
    records = []
    for i, (season, m) in enumerate(seasons):
        y = 7 - i; start = pd.Timestamp(m.start); end = pd.Timestamp(m.end)
        if i % 2 == 0:
            ax.axhspan(y - .5, y + .5, color=LIGHT, lw=0, zorder=0)
        part = fd.loc[fd.season.eq(season) & fd.treatment.eq('PS')].sort_values('Date')
        for field, role in crops.items():
            active = part.loc[part[field].gt(.5), 'Date']
            if len(active):
                days = (active - start).dt.days.to_numpy()
                for run in np.split(days, np.where(np.diff(days) > 1)[0] + 1):
                    ax.add_patch(FancyBboxPatch((run[0], y + .14), len(run), .2, boxstyle='round,pad=0,rounding_size=.08',
                                                fc=COLOR[role], ec=INK, lw=.5, zorder=3, mutation_aspect=.02))
                    records.append(dict(season=season, crop=role, start_day=int(run[0]), end_day=int(run[-1])))
        for treatment, dy in [('PS', -.08), ('AN', -.32)]:
            obs = PRED['OriginalLightGBM'].query('season==@season and treatment==@treatment').dropna(subset=['y'])
            ax.scatter((obs.Date - start).dt.days, np.full(len(obs), y + dy), marker='|', s=30,
                       color=COLOR['Support'], linewidths=.55, zorder=3)
        ev = EVENTS.loc[EVENTS.date.between(start, end)]
        for e in ev.itertuples():
            ax.scatter((e.date - start).days, y + (-.08 if e.treatment == 'PS' else -.32), s=e.Ntotal * .55 + 12,
                       marker='v', color='white', edgecolors=INK, linewidths=.9, zorder=6)
    ax.set_xlim(-6, 425); ax.set_ylim(-.55, 7.55)
    ax.set_yticks(range(8), [f'{s}  {m.main_crop}' for s, m in seasons[::-1]])
    ax.set_xticks([0, 100, 200, 300, 400])
    style(ax, 'Days from cycle start')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    hs = [Rectangle((0, 0), 1, 1, fc=COLOR[r], ec=INK, lw=.5) for r in ['Wheat', 'Maize', 'Mustard']]
    hs += [Line2D([], [], color=COLOR['Support'], ls='', marker='|', ms=12, mew=1.2),
           Line2D([], [], color='white', ls='', marker='v', ms=10, markeredgecolor=INK)]
    ax.legend(hs, ['Wheat', 'Maize', 'Mustard', 'Observed dates', 'N input'], loc='lower center',
              bbox_to_anchor=(.45, 1.0), ncol=5, fontsize=TICK, handlelength=1.1, columnspacing=1.0)
    save(pd.DataFrame(records), FIGDIR / 'Fig1_framework' / 'timeline_crop_segments.csv')


def fig1():
    fig = plt.figure(figsize=(K.FIG_W, 10.2))
    lay = Layout(fig)
    rows = grid_cells(.045, .995, .985, [.45, .50], [[.47, .53], [1]], hgap=.07, vgap=.045)
    a = fig.add_axes([.1, .6, .3, .3]); bb = fig.add_axes([.55, .6, .4, .3])
    c = fig.add_axes([.03, .02, .95, .45])
    map_panel(a); timeline_panel(bb); draw_framework(c)
    lay.add('a', a, rows[0][0]); lay.add('b', bb, rows[0][1]); lay.add('c', c, rows[1][0])
    # The map keeps its cartographic aspect; its outer height then defines row 1.
    cell = rows[0][0]
    for _ in range(14):
        fig.canvas.draw(); o = lay.outer([a]); p = a.get_position()
        s = min((cell[2] - cell[0]) / o.width, (cell[3] - cell[1]) / o.height)
        a.set_position([p.x0, p.y0, p.width * s, p.height * s])
        fig.canvas.draw(); o = lay.outer([a]); p = a.get_position()
        a.set_position([p.x0 + cell[0] - o.x0, p.y0 + cell[3] - o.y1, p.width, p.height])
    fig.canvas.draw(); o = lay.outer([a])
    lay.panels[0]['cell'] = Bbox.from_extents(cell[0], o.y0, cell[2], cell[3])
    lay.panels[1]['cell'] = Bbox.from_extents(rows[0][1][0], o.y0, rows[0][1][2], rows[0][1][3])
    c_cell = rows[1][0]
    lay.panels[2]['cell'] = Bbox.from_extents(c_cell[0], c_cell[1], c_cell[2], o.y0 - .022)
    lay.fit(['b', 'c'])
    lay.panels[0]['fixed'] = True
    lay.letters(dx=-.006)
    finish(fig, lay, 'Fig1_framework', [GEO / 'site_locations.csv', RES8 / 'season_manifest.csv',
                                         RES8 / 'fertilizer_events.csv', RES8 / 'all_daily_predictions.csv'])


# =========================================================================== Fig. 2
def fig2():
    name = 'Fig2_daily_trajectories'
    fig = plt.figure(figsize=(K.FIG_W, 12.2))
    lay = Layout(fig)
    x0, x1, top, bottom = .085, .995, .928, .065
    inner, pair_gap, vgap = .024, .06, .056
    pw = (x1 - x0 - 2 * inner - pair_gap) / 4
    ph = (top - bottom - 3 * vgap) / 4
    records = []
    letters = 'abcdefghijklmnop'
    for i, (year, tr) in enumerate([(y, t) for y in range(2014, 2022) for t in ['AN', 'PS']]):
        r, cpos = divmod(i, 4)
        left = x0 + cpos * pw + (cpos >= 1) * inner + (cpos >= 2) * (pair_gap - inner) + (cpos >= 3) * inner
        y0 = top - (r + 1) * ph - r * vgap
        ax = fig.add_axes([left, y0, pw, ph])
        start = pd.Timestamp(MAN.loc[year, 'start'])
        g = PRED['OriginalLightGBM'].query('season==@year and treatment==@tr').copy()
        days = (g.Date - start).dt.days
        both = PRED['OriginalLightGBM'].query('season==@year')
        allv = np.r_[both.y.dropna(), both.prediction, PRED['AffineReconciled'].query('season==@year').prediction]
        ymin = min(-5, float(np.min(allv)) - 3); ymax = float(np.max(allv)) * 1.16 + 6
        for mod, ls, lw in [('OriginalLightGBM', (0, (3.2, 1.6)), 1.5), ('AffineReconciled', '-', 1.7)]:
            gg = PRED[mod].query('season==@year and treatment==@tr')
            ax.plot(days, gg.prediction, color=COLOR[mod], ls=ls, lw=lw, zorder=3)
            g[mod] = gg.prediction.to_numpy()
        ax.scatter(days, g.y, s=17, color=COLOR['Observed'], edgecolors=INK, linewidths=.4, zorder=4)
        ev = EVENTS.loc[EVENTS.treatment.eq(tr) & EVENTS.date.between(start, pd.Timestamp(MAN.loc[year, 'end']))]
        ax.scatter((ev.date - start).dt.days, np.full(len(ev), ymax - .045 * (ymax - ymin)), marker='v', s=46,
                   color='white', edgecolors=INK, linewidths=.8, zorder=6, clip_on=False)
        ax.axhline(0, color=GRID, lw=.8, zorder=1)
        ax.set_ylim(ymin, ymax); ax.set_xlim(0, int(MAN.loc[year, 'days']) - 1)
        ax.set_xticks([0, 100, 200, 300, 400] if len(g) > 400 else [0, 100, 200, 300])
        nticks(ax, x=False, n=3)
        style(ax)
        ax.text(.04, .97, f'{year} {tr}', transform=ax.transAxes, ha='left', va='top', fontsize=TICK, color=INK)
        restore = {'xlabel': 'Days from cycle start', 'ylabel': f'Flux ({UNIT})'}
        if tr == 'PS':
            ax.tick_params(axis='y', labelleft=False); restore['ylabels'] = True
        g['day_since_start'] = days; records.append(g)
        lay.add(letters[i], ax, (left, y0, left + pw, y0 + ph), letter_xy=(left - .004, y0 + ph + .028), restore=restore)
    fig.text(.5, .012, 'Days from cycle start', ha='center', va='bottom', fontsize=BASE)
    fig.text(.012, (top + bottom) / 2, f'N$_2$O-N flux ({UNIT})', rotation=90, ha='left', va='center', fontsize=BASE)
    hs = [Line2D([], [], color=COLOR['Observed'], marker='o', ls='', ms=6, markeredgecolor=INK, markeredgewidth=.5),
          Line2D([], [], color=COLOR['OriginalLightGBM'], ls=(0, (3.2, 1.6)), lw=2),
          Line2D([], [], color=COLOR['AffineReconciled'], lw=2),
          Line2D([], [], color='white', marker='v', ls='', ms=8, markeredgecolor=INK)]
    labels = ['Observed', 'LightGBM', 'Affine reconciliation', 'N application']
    fig.legend(hs, labels, loc='upper center', bbox_to_anchor=(.54, 1.0), ncol=4, fontsize=BASE, columnspacing=2.4)
    fig._free_alignment = True  # plot areas share exact edges; only tick-label widths differ
    lay.letters()
    for t in fig.texts:
        if t.get_gid() == 'panel_letter':
            t.set_fontsize(20)

    def legend_axes(ax):
        ax.legend(hs, labels, loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=TICK)
    save(pd.concat(records), FIGDIR / name / 'daily_trajectories.csv')
    finish(fig, lay, name, [RES8 / 'all_daily_predictions.csv', RES8 / 'season_manifest.csv', RES8 / 'fertilizer_events.csv'],
           legend_axes=legend_axes)


# =========================================================================== Fig. 3
LS = {'OriginalLightGBM': (0, (3.2, 1.6)), 'TabPFN_DailyRaw': (0, (1, 1.2)), 'AffineReconciled': '-',
      'MeanReconciled': (0, (5, 1.5, 1, 1.5)), 'DecoupledPulse': (0, (2, 1))}
TRIO = ['OriginalLightGBM', 'TabPFN_DailyRaw', 'AffineReconciled']


def model_handles(models, with_observed=False):
    hs, ls = [], []
    if with_observed:
        hs.append(Line2D([], [], color=COLOR['Observed'], marker='o', ls='', ms=8, markeredgecolor=INK, markeredgewidth=.6))
        ls.append('Observed')
    for m in models:
        hs.append(Line2D([], [], color=COLOR[m], ls=LS[m], lw=2.2, marker=MARKER[m], ms=8 if m != 'DecoupledPulse' else 11,
                         markeredgecolor=INK, markeredgewidth=.7))
        ls.append(LABEL[m])
    return hs, ls


def taylor_panel(ax):
    ax.set_aspect('equal'); ax.set_xlim(-.03, 1.42); ax.set_ylim(-.03, 1.42)
    th = np.linspace(0, np.pi / 2, 160); R = 1.3
    for sd in [.25, .5, .75, 1.0, 1.25]:
        ax.plot(sd * np.cos(th), sd * np.sin(th), color=GRID if sd != 1 else MUTED, lw=.8 if sd != 1 else 1.0, zorder=1)
    ax.plot(R * np.cos(th), R * np.sin(th), color=INK, lw=1.3, zorder=2)
    for r in [.2, .4, .6, .8, .9, .95, .99]:
        ang = np.arccos(r)
        ax.plot([0, R * np.cos(ang)], [0, R * np.sin(ang)], color=GRID, lw=.7, zorder=1)
        ax.plot([R * np.cos(ang), 1.03 * R * np.cos(ang)], [R * np.sin(ang), 1.03 * R * np.sin(ang)], color=INK, lw=1.1)
        ax.text(1.085 * R * np.cos(ang), 1.085 * R * np.sin(ang), f'{r:g}', fontsize=TICK, ha='center', va='center')
    ax.text(1.2 * R * np.cos(np.radians(47)), 1.2 * R * np.sin(np.radians(47)), 'Correlation', rotation=-43,
            fontsize=BASE, ha='center', va='center')
    xx, yy = np.meshgrid(np.linspace(0, 1.4, 220), np.linspace(0, 1.4, 220))
    rms = np.sqrt((xx - 1) ** 2 + yy ** 2); rms[np.hypot(xx, yy) > R] = np.nan
    cs = ax.contour(xx, yy, rms, levels=[.25, .5, .75, 1.0], colors=MUTED, linewidths=.8, linestyles=[(0, (3, 2))])
    ax.clabel(cs, fmt='%g', fontsize=TICK, inline_spacing=2)
    stats = read(TAB11 / 'daily_dispersion_metrics.csv').set_index('model')
    for m in TRIO:
        v = stats.loc[m]
        ax.scatter(v.sd_ratio * v.correlation, v.sd_ratio * np.sqrt(1 - v.correlation ** 2), marker=MARKER[m],
                   s=120, color=COLOR[m], edgecolors=INK, linewidths=.8, zorder=6)
    ax.scatter(1, 0, marker='*', s=260, color=COLOR['Observed'], edgecolors=INK, linewidths=.8, zorder=7, clip_on=False)
    ax.text(1.06, .035, 'Observed', ha='left', va='bottom', fontsize=TICK)
    ax.set_xticks([0, .5, 1], ['0', '0.5', '1'])
    ax.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    style(ax, 'Normalized SD', 'Normalized SD')
    ax.spines['left'].set_bounds(0, R); ax.spines['bottom'].set_bounds(0, R)


def fig3():
    name = 'Fig3_scales_and_errors'
    fig = plt.figure(figsize=(K.FIG_W, 12.67)); lay = Layout(fig)
    rows = grid_cells(.065, .995, .956, [.286, .297, .318], [[.6, .4], [.5, .5], [.42, .58]], hgap=.085, vgap=.025)
    a, bb, c, d, e, f = [fig.add_axes([.1, .1, .3, .3]) for _ in range(6)]
    # a: pulse trace
    trace = PRED['OriginalLightGBM'].query('season==2014 and treatment=="AN"').copy()
    trace = trace.loc[trace.Date.between('2014-04-20', '2014-06-10')]
    for mod in ['OriginalLightGBM', 'AffineReconciled']:
        g = PRED[mod].query('season==2014 and treatment=="AN"').set_index('Date').loc[trace.Date]
        a.plot(g.index, g.prediction, color=COLOR[mod], ls=LS[mod], lw=2.4, zorder=3)
        trace[mod] = g.prediction.to_numpy()
    a.scatter(trace.Date, trace.y, s=46, color=COLOR['Observed'], edgecolors=INK, linewidths=.6, zorder=5)
    a.axvline(pd.Timestamp('2014-04-28'), color=INK, lw=1, ls=':')
    a.text(pd.Timestamp('2014-04-29 12:00'), 575, ' N application', fontsize=TICK, va='top')
    a.text(.98, .96, '2014 ammonium nitrate', transform=a.transAxes, ha='right', va='top', fontsize=TICK)
    a.set_ylim(-25, 590); a.set_xlim(pd.Timestamp('2014-04-19'), pd.Timestamp('2014-06-11'))
    a.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    style(a, 'Date (2014)', f'Daily flux ({UNIT})'); nticks(a, x=False, n=4)
    save(trace, FIGDIR / name / 'pulse_trace.csv')
    # b: R2 across scales
    scales = ['daily', 'weekly', 'monthly', 'cycle']
    for m in TRIO:
        bb.plot(range(4), [M.loc[m, f'R2_{s}'] for s in scales], color=COLOR[m], ls=LS[m], lw=2.2, marker=MARKER[m],
                ms=9, markeredgecolor=INK, markeredgewidth=.7, zorder=4)
    bb.set_xticks(range(4), ['Day', 'Week', 'Month', 'Cycle']); bb.set_ylim(.15, .88); bb.set_xlim(-.3, 3.3)
    style(bb, 'Aggregation scale', 'Prediction R²', grid='y'); nticks(bb, x=False, n=4)
    save(M.loc[TRIO, [f'R2_{s}' for s in scales]].reset_index(), FIGDIR / name / 'scale_scores.csv')
    # c: cumulative SSE
    ranked = []
    for m in ['OriginalLightGBM', 'AffineReconciled']:
        g = PRED[m].dropna(subset=['y']).sort_values('y', ascending=False).copy()
        g['rank'] = np.arange(1, len(g) + 1); g['squared_error'] = (g.y - g.prediction) ** 2
        g['cumulative_sse_fraction'] = g.squared_error.cumsum() / g.squared_error.sum(); ranked.append(g)
        c.plot(g['rank'], 100 * g.cumulative_sse_fraction, color=COLOR[m], ls=LS[m], lw=2.4, zorder=3)
        fr = g.iloc[4].cumulative_sse_fraction * 100
        c.scatter(5, fr, marker=MARKER[m], s=80, color=COLOR[m], edgecolors=INK, linewidths=.8, zorder=5)
        c.annotate(f'{fr:.1f}%', (5, fr), xytext=(-12, 8) if m == 'AffineReconciled' else (10, -16),
                   textcoords='offset points', fontsize=TICK, ha='right' if m == 'AffineReconciled' else 'left')
    c.axvline(5, color=INK, lw=1, ls=':')
    c.text(5.6, 4, 'five largest\nobserved days', fontsize=TICK, va='bottom')
    c.set_xscale('log'); c.set_xlim(1, 4000); c.set_ylim(0, 103)
    c.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1000']); c.minorticks_off()
    style(c, 'Days ranked by observed flux', 'Cumulative daily SSE (%)', grid='y'); nticks(c, x=False, n=4)
    save(pd.concat(ranked), FIGDIR / name / 'ranked_error_concentration.csv')
    # d: SSE decomposition
    dec = ['OriginalLightGBM', 'TabPFN_DailyRaw', 'MeanReconciled', 'AffineReconciled', 'DecoupledPulse']
    for i, m in enumerate(dec):
        y = len(dec) - 1 - i; ll = SSE.loc[m, 'level_sse'] / 1e6; ss = SSE.loc[m, 'shape_sse'] / 1e6
        d.barh(y, ll, color=COLOR['Level'], edgecolor=INK, lw=.7, height=.58, hatch='////', zorder=3)
        d.barh(y, ss, left=ll, color=COLOR['Shape'], edgecolor=INK, lw=.7, height=.58, zorder=3)
        d.text(ll + ss + .03, y, f'{ll + ss:.2f}', va='center', fontsize=TICK)
    d.set_yticks(range(len(dec)), [LABEL[m] for m in dec[::-1]]); d.set_ylim(-.6, len(dec) - .4)
    d.set_xlim(0, 2.15); d.set_xticks([0, .5, 1, 1.5, 2])
    style(d, 'Daily SSE (×10$^6$)', grid='x'); d.tick_params(axis='y', length=0); d.spines['left'].set_visible(False)
    d.legend([Rectangle((0, 0), 1, 1, fc=COLOR['Level'], ec=INK, hatch='////'), Rectangle((0, 0), 1, 1, fc=COLOR['Shape'], ec=INK)],
             ['Cycle level', 'Within-cycle shape'], loc='lower center', bbox_to_anchor=(.42, 1.0), ncol=2, fontsize=TICK)
    save(SSE.loc[dec].reset_index(), FIGDIR / name / 'error_decomposition.csv')
    # e: Taylor
    taylor_panel(e)
    # f: conditional bias
    tab = read(TAB11 / 'conditional_bias.csv')
    for i, m in enumerate(TRIO):
        g = tab.loc[tab.model.eq(m)].sort_values('bin'); x = g.bin + (i - 1) * .17
        f.plot(x, g.bias, color=COLOR[m], ls=LS[m], lw=1.6, zorder=3)
        f.errorbar(x, g.bias, yerr=[g.bias - g.q025, g.q975 - g.bias], fmt='none', ecolor=COLOR[m], elinewidth=1.8,
                   capsize=0, zorder=3)
        f.scatter(x, g.bias, marker=MARKER[m], s=70, color=COLOR[m], edgecolors=INK, linewidths=.7, zorder=5)
    f.axhline(0, color=INK, lw=.9, ls=(0, (3, 2)))
    f.set_xticks(range(5), ['0–50', '50–75', '75–90', '90–95', '95–100'])
    style(f, 'Observed-flux percentile', f'Bias ({UNIT})', grid='y'); nticks(f, x=False, n=5)
    cells = [r for row in rows for r in row]
    for k, ax, cell in zip('abcdef', [a, bb, c, d, e, f], cells):
        lay.add(k, ax, cell)
    lay.fit(['a', 'b', 'c', 'd', 'f'])
    lay.fit(['e'], keep_aspect=('e',))
    hs, ls = model_handles(TRIO, with_observed=True)
    fig.legend(hs, ls, loc='upper center', bbox_to_anchor=(.53, 1.0), ncol=4, fontsize=BASE, columnspacing=2.2,
               handlelength=2.6)
    lay.letters(dx=-.022)

    def legend_axes(ax):
        if ax.get_xlabel().startswith('Daily SSE'):
            return
        h2, l2 = model_handles(TRIO, with_observed=True)
        ax.legend(h2, l2, loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=2, fontsize=TICK, handlelength=2.4)
    finish(fig, lay, name, [RES8 / 'all_daily_predictions.csv', RES8 / 'all_model_metrics.csv', RES8 / 'SSE_decomposition.csv',
                            TAB11 / 'daily_dispersion_metrics.csv', TAB11 / 'conditional_bias.csv'], legend_axes=legend_axes)


# =========================================================================== Fig. 4
def forest_ci(ax, models, labels, scale, *, paired=False, label_left=True):
    for i, m in enumerate(models):
        y = len(models) - 1 - i
        if paired:
            if m == 'MeanReconciled':
                v, lo, hi = 0., 0., 0.
            else:
                r = PAIR.query('model==@m and reference=="MeanReconciled" and scale==@scale').iloc[0]
                v, lo, hi = r.delta_R2, r.q025, r.q975
        else:
            r = M.loc[m]; v, lo, hi = r[f'R2_{scale}'], r[f'R2_{scale}_lower'], r[f'R2_{scale}_upper']
        ax.plot([lo, hi], [y, y], color=COLOR[m], lw=5.5, solid_capstyle='round', zorder=3, alpha=.9)
        ax.plot([lo, hi], [y, y], color=INK, lw=.6, zorder=2)
        ax.scatter(v, y, marker=MARKER.get(m, 'o'), s=150 if m == 'DecoupledPulse' else 85, color=COLOR[m],
                   edgecolors=INK, linewidths=.9, zorder=5)
    bands(ax, len(models))
    ax.set_yticks(range(len(models)), [labels[m] for m in models[::-1]])
    ax.set_ylim(-.6, len(models) - .4)
    ax.axvline(0, color=INK, lw=.9, ls=(0, (3, 2)))
    xl = f'Δ {scale} prediction R²' if paired else f'{scale.capitalize()} prediction R²'
    style(ax, xl); ax.tick_params(axis='y', length=0, labelleft=label_left); ax.spines['left'].set_visible(False)
    nticks(ax, y=False, n=4)


DENSITY_RAMP = {'OriginalLightGBM': [BLUE, PURPLE], 'AffineReconciled': [PINK, ROSE]}


def density_panel(fig, main, model):
    g = PRED[model].dropna(subset=['y'])
    tr = SymmetricalLogTransform(base=10, linthresh=5, linscale=1)
    x = tr.transform(g.y.to_numpy()); y = tr.transform(g.prediction.to_numpy())
    lim = tr.transform([-30, 600])
    cm = K.cmap('#FFFFFF', *DENSITY_RAMP[model], name='dens_' + model)
    hb = main.hexbin(x, y, gridsize=34, mincnt=1, cmap=cm, norm=LogNorm(vmin=1, vmax=400), edgecolors=INK,
                     linewidths=.2, extent=(*lim, *lim))
    hb.set_gid('continuous')
    main.plot(lim, lim, ls=(0, (3, 2)), color=INK, lw=1)
    main.set_xlim(lim); main.set_ylim(lim)
    vals = np.array([-10, 0, 10, 100, 500]); tk = tr.transform(vals)
    main.set_xticks(tk, [str(v) for v in vals]); main.set_yticks(tk, [str(v) for v in vals])
    style(main, f'Observed ({UNIT})', f'Predicted ({UNIT})')
    st = read(TAB11 / 'daily_dispersion_metrics.csv').set_index('model').loc[model]
    main.text(.97, .04, f'R² = {st.R2:.3f}\nRMSE = {st.RMSE:.1f}\nn = {len(g)}', transform=main.transAxes, ha='right',
              va='bottom', fontsize=TICK, linespacing=1.3)
    main.text(.04, .96, LABEL[model], transform=main.transAxes, ha='left', va='top', fontsize=BASE)
    p = main.get_position()
    top = fig.add_axes([p.x0, p.y1 + .01, p.width, p.height * .22])
    err = g.prediction - g.y
    top.hist(err, bins=np.linspace(-550, 250, 41), color=COLOR[model], edgecolor=INK, lw=.4, zorder=3)
    top.axvline(0, color=INK, ls=(0, (3, 2)), lw=.9, zorder=4)
    top.set_yscale('log'); top.set_xlim(-560, 260); top.set_ylim(.7, 4000)
    top.set_yticks([1, 100], ['1', '100'])
    style(top, None, 'Days'); top.set_xticks([-500, -250, 0, 250]); top.minorticks_off()
    top.set_xlabel('Prediction − observation', fontsize=TICK, labelpad=3)
    top.xaxis.set_label_position('top'); top.xaxis.tick_top()
    top.spines['top'].set_visible(True); top.spines['bottom'].set_visible(False)
    top.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    cax = fig.add_axes([p.x0 + .55 * p.width, p.y0 - .3 * p.height, .42 * p.width, .035 * p.height])
    cb = fig.colorbar(hb, cax=cax, orientation='horizontal', ticks=[1, 10, 100])
    cb.ax.set_xticklabels(['1', '10', '100']); cb.outline.set_linewidth(1.0); cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=TICK, length=4, color=INK)
    cb.set_label('Days per hexagon', fontsize=TICK, labelpad=3)
    return [main, top, cax], dict(transform='symlog base 10, linthresh 5, linscale 1', gridsize=34, count_range=[1, 400],
                                  colour_endpoints=['#FFFFFF', *DENSITY_RAMP[model]])


def fig4():
    name = 'Fig4_model_comparison'
    fig = plt.figure(figsize=(K.FIG_W, 14.6)); lay = Layout(fig)
    rows = grid_cells(.06, .995, .99, [.265, .265, .401], [[.46, .54], [.46, .54], [.5, .5]], hgap=.085, vgap=.025)
    a, bb, c, d = [fig.add_axes([.1, .1, .3, .3]) for _ in range(4)]
    numbered = {m: f'{i + 1}  {LABEL[m]}' for i, m in enumerate(CORE)}
    offs = [(-10, 12), (12, -14), (-13, 11), (12, 9), (-30, -14), (-34, 10), (12, 11)]
    for i, (m, off) in enumerate(zip(CORE, offs)):
        r = M.loc[m]
        a.scatter(r.R2_daily, r.R2_cycle, marker=MARKER[m], s=230 if m == 'DecoupledPulse' else 130, color=COLOR[m],
                  edgecolors=INK, linewidths=.9, zorder=5)
        a.annotate(str(i + 1), (r.R2_daily, r.R2_cycle), xytext=off, textcoords='offset points', fontsize=BASE,
                   ha='center', va='center', arrowprops=dict(arrowstyle='-', color=INK, lw=.6, shrinkA=0, shrinkB=6))
    for src, dst in [('OriginalLightGBM', 'MeanReconciled'), ('MeanReconciled', 'AffineReconciled')]:
        a.annotate('', (M.loc[dst, 'R2_daily'], M.loc[dst, 'R2_cycle']), (M.loc[src, 'R2_daily'], M.loc[src, 'R2_cycle']),
                   arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2, shrinkA=11, shrinkB=11, mutation_scale=14))
    a.set_xlim(.266, .356); a.set_ylim(.605, .845)
    a.set_xticks([.28, .30, .32, .34]); a.set_yticks([.65, .70, .75, .80])
    style(a, 'Daily prediction R²', 'Cycle prediction R²', grid='both')
    a.text(.03, .04, '7 schemes, 8 held-out cycles', transform=a.transAxes, fontsize=TICK)
    forest_ci(bb, CORE, numbered, 'cycle'); bb.set_xlim(-.05, 1.0)
    # c observed vs predicted cycle means
    models = ['OriginalLightGBM', 'AffineReconciled']
    L_ = C.loc[C.model.eq(models[0])].set_index(['season', 'treatment'])
    R_ = C.loc[C.model.eq(models[1])].set_index(['season', 'treatment'])
    for key, r in L_.iterrows():
        c.plot([r.y, r.y], [r.prediction, R_.loc[key, 'prediction']], color=MUTED, lw=.9, zorder=2)
    for m in models:
        g = C.loc[C.model.eq(m)]
        for tr, mk in [('AN', 'o'), ('PS', '^')]:
            q = g.loc[g.treatment.eq(tr)]
            c.scatter(q.y, q.prediction, marker=mk, s=85, color=COLOR[m], edgecolors=INK, linewidths=.8, zorder=4)
    c.plot([0, 30], [0, 30], color=INK, ls=(0, (3, 2)), lw=1)
    c.set(xlim=(0, 30), ylim=(0, 30)); c.set_xticks([0, 10, 20, 30]); c.set_yticks([0, 10, 20, 30]); c.set_aspect('auto')
    style(c, f'Observed cycle mean ({UNIT})', f'Predicted cycle mean ({UNIT})', grid='both')
    hs = [Line2D([], [], color=COLOR[m], marker='o', ls='', ms=9, markeredgecolor=INK) for m in models]
    hs += [Line2D([], [], color='white', marker=k, ls='', ms=8, markeredgecolor=INK) for k in ['o', '^']]
    c.legend(hs, [LABEL[m] for m in models] + ['Ammonium nitrate', 'Pig slurry'], loc='upper left', fontsize=TICK,
             handletextpad=.4)
    c.text(.97, .04, '16 means per model', transform=c.transAxes, ha='right', fontsize=TICK)
    # d RMSE per cycle
    errors = read(RES8 / 'per_cycle_errors.csv')
    feat = ['OriginalLightGBM', 'DecoupledPulse', 'AffineReconciled']
    for m in feat:
        g = errors.loc[errors.model.eq(m)].sort_values('season')
        d.plot(g.season, np.sqrt(g.MSE), color=COLOR[m], ls=LS[m], lw=2.2, marker=MARKER[m],
               ms=13 if m == 'DecoupledPulse' else 8.5, markeredgecolor=INK, markeredgewidth=.75, zorder=4 + (m == 'AffineReconciled'))
    d.set_xticks([2014, 2016, 2018, 2020]); d.set_xticks(range(2014, 2022), minor=True); d.set_ylim(0, 5.3); d.set_xlim(2013.6, 2021.4)
    style(d, 'Held-out management cycle', f'Cycle-mean RMSE ({UNIT})', grid='y'); nticks(d, x=False, n=4)
    d.tick_params(axis='x', labelrotation=0)
    hs, ls = model_handles(feat)
    d.legend(hs, ls, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=TICK, handlelength=2.4, columnspacing=1.2)
    save(M.loc[CORE].reset_index(), FIGDIR / name / 'core_scores.csv')
    save(C.loc[C.model.isin(models)], FIGDIR / name / 'observed_vs_predicted_cycle_means.csv')
    save(errors.loc[errors.model.isin(feat)], FIGDIR / name / 'per_cycle_errors.csv')
    cells = [r for row in rows for r in row]
    for k, ax, cell in zip('abcd', [a, bb, c, d], cells):
        lay.add(k, ax, cell)
    settings = []
    for k, m, cell in zip('ef', models, cells[4:]):
        main = fig.add_axes([cell[0] + .08, cell[1] + .1, cell[2] - cell[0] - .12, (cell[3] - cell[1]) * .55])
        axes, st = density_panel(fig, main, m); settings.append(st)
        lay.add(k, axes, cell)
    K.dump(settings, FIGDIR / name / 'joint_density_settings.json')
    rep = lay.fit(); print([ (r['panel'], round(r['max_residual_pt'], 2)) for r in rep])
    lay.letters(dx=-.022)
    finish(fig, lay, name, [RES8 / 'all_model_metrics.csv', RES8 / 'all_cycle_predictions.csv', RES8 / 'per_cycle_errors.csv',
                            RES8 / 'all_daily_predictions.csv', TAB11 / 'daily_dispersion_metrics.csv'])


# =========================================================================== Fig. 5
AMARK = {'MeanReconciled': 's', 'AffineReconciled': '^', 'DecoupledNoPulse': 'o', 'DecoupledPulse': '*',
         'DecoupledUnconstrained': 'o', 'TiedPulse': 'o', 'PulseShapeOnly': 'o', 'PulseLevelOnly': 'o'}


def cat_axis(ax, n, labels, *, label_left=True):
    bands(ax, n)
    ax.set_yticks(range(n), labels)
    ax.set_ylim(-.6, n - .4)
    ax.tick_params(axis='y', length=0, labelleft=label_left)
    ax.spines['left'].set_visible(False)


def fig5():
    name = 'Fig5_ablation_and_windows'
    fig = plt.figure(figsize=(K.FIG_W, 13.06)); lay = Layout(fig)
    rows = grid_cells(.04, .995, .99, [.281, .297, .323], [[.585, .415]] * 3, hgap=.045, vgap=.03)
    axs = [fig.add_axes([.1, .1, .3, .3]) for _ in range(6)]
    a, bb, c, d, e, f = axs
    labels = [ALABEL[m] for m in ABL[::-1]]
    n = len(ABL)
    for ax, scale, left in [(a, 'cycle', True), (bb, 'daily', False)]:
        for i, m in enumerate(ABL):
            y = n - 1 - i
            if m == 'MeanReconciled':
                v, lo, hi = 0., 0., 0.
            else:
                r = PAIR.query('model==@m and reference=="MeanReconciled" and scale==@scale').iloc[0]
                v, lo, hi = r.delta_R2, r.q025, r.q975
            ax.plot([lo, hi], [y, y], color=COLOR[m], lw=5.5, solid_capstyle='round', zorder=3, alpha=.9)
            ax.scatter(v, y, marker=AMARK[m], s=170 if m == 'DecoupledPulse' else 85, color=COLOR[m], edgecolors=INK,
                       linewidths=.9, zorder=5)
        ax.axvline(0, color=INK, lw=.9, ls=(0, (3, 2)))
        style(ax, f'Δ {scale} prediction R²'); cat_axis(ax, n, labels, label_left=left); nticks(ax, y=False, n=4)
    for i, m in enumerate(ABL):
        y = n - 1 - i
        for comp, mk, off in [('level_sse', 'o', .17), ('shape_sse', 's', -.17)]:
            ratio = SSE.loc[m, comp] / SSE.loc['MeanReconciled', comp]
            c.scatter(ratio, y + off, marker=mk, s=80, color=COLOR[m], edgecolors=INK, linewidths=.8, zorder=5)
        lv = SSE.loc[m, 'level_sse'] / SSE.loc['MeanReconciled', 'level_sse']
        sv = SSE.loc[m, 'shape_sse'] / SSE.loc['MeanReconciled', 'shape_sse']
        c.plot([lv, sv], [y + .17, y - .17], color=MUTED, lw=.8, zorder=2)
        d.scatter(100 * M.loc[m, 'continuous_window_WAPE'], y, marker=AMARK[m], s=170 if m == 'DecoupledPulse' else 95,
                  color=COLOR[m], edgecolors=INK, linewidths=.9, zorder=5)
    c.axvline(1, color=INK, ls=(0, (3, 2)), lw=.9)
    style(c, 'SSE / mean-reconciliation SSE'); cat_axis(c, n, labels); nticks(c, y=False, n=4)
    c.legend([Line2D([], [], color='white', marker=k, ls='', ms=9, markeredgecolor=INK) for k in ['o', 's']],
             ['Cycle level', 'Within-cycle shape'], loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=TICK)
    base = 100 * M.loc['OriginalLightGBM', 'continuous_window_WAPE']
    d.axvline(base, color=COLOR['OriginalLightGBM'], ls=(0, (3.2, 1.6)), lw=2)
    d.text(base + .1, n - .55, 'LightGBM', color=INK, fontsize=TICK, ha='left', va='center')
    d.set_xlim(55.5, 61)
    style(d, 'Window-integral WAPE (%)'); cat_axis(d, n, labels, label_left=False); nticks(d, y=False, n=4)
    d.text(.98, .985, '147 observed windows', transform=d.transAxes, ha='right', va='top', fontsize=TICK)
    # e/f: change in the estimator itself (unchanged algorithm from round 11)
    observed = D.dropna(subset=['y']).copy(); unit = ['season', 'treatment']
    base_t = observed.loc[observed.model.eq('TabPFN_DailyRaw')].set_index('row_id')
    base_shape = base_t.prediction - base_t.groupby(unit).prediction.transform('mean')
    base_level = C.loc[C.model.eq('OriginalLightGBM')].set_index(unit).prediction
    adjustments = []
    for i, m in enumerate(ABL):
        g = observed.loc[observed.model.eq(m)].set_index('row_id').reindex(base_t.index)
        centered = g.prediction - g.groupby(unit).prediction.transform('mean')
        dshape = centered - base_shape
        levels = C.loc[C.model.eq(m)].set_index(unit).prediction - base_level
        for year in range(2014, 2022):
            adjustments.append(dict(model=m, season=year, level_rms=float(np.sqrt(np.mean(levels.loc[year] ** 2))),
                                    shape_rms=float(np.sqrt(np.mean(dshape.loc[g.season.eq(year)] ** 2))),
                                    n_obs=int(g.season.eq(year).sum())))
        part = pd.DataFrame(adjustments).query('model==@m').sort_values('season')
        for ax, comp in [(e, 'level'), (f, 'shape')]:
            vals = part[comp + '_rms'].to_numpy()
            w = np.ones(8) if comp == 'level' else part.n_obs.to_numpy()
            pooled = np.sqrt(np.average(vals ** 2, weights=w)); yy = n - 1 - i
            ax.plot([vals.min(), vals.max()], [yy, yy], color=COLOR[m], lw=1.6, zorder=3)
            ax.scatter(vals, yy + np.linspace(-.2, .2, 8), s=26, color=COLOR[m], edgecolors=INK, linewidths=.45, zorder=4)
            ax.scatter(pooled, yy, marker='D', s=95, color=COLOR[m], edgecolors=INK, linewidths=.9, zorder=6)
    for ax, comp, left in [(e, 'Cycle-level', True), (f, 'Daily-shape', False)]:
        ax.axvline(0, color=INK, ls=(0, (3, 2)), lw=.9)
        style(ax, f'{comp} adjustment RMS\n({UNIT})'); cat_axis(ax, n, labels, label_left=left); nticks(ax, y=False, n=4)
        ax.set_xlim(left=-.05 * max(ax.get_xlim()[1], .1))
    e.text(.98, .985, 'Reference: LightGBM cycle means', transform=e.transAxes, ha='right', va='top', fontsize=TICK)
    f.text(.98, .985, 'Reference: centered TabPFN', transform=f.transAxes, ha='right', va='top', fontsize=TICK)
    hs = [Line2D([], [], color=MUTED, marker='o', ls='', ms=6, markerfacecolor='white', markeredgecolor=INK),
          Line2D([], [], color='white', marker='D', ls='', ms=9, markeredgecolor=INK)]
    e.legend(hs, ['Held-out cycle', 'Pooled'], loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=TICK)
    save(pd.DataFrame(adjustments), FIGDIR / name / 'component_adjustments_by_cycle.csv')
    save(M.loc[ABL].reset_index(), FIGDIR / name / 'ablation_scores.csv')
    save(PAIR.loc[PAIR.model.isin(ABL) & PAIR.reference.eq('MeanReconciled') & PAIR.scale.isin(['cycle', 'daily'])],
         FIGDIR / name / 'paired_differences.csv')
    cells = [r for row in rows for r in row]
    for k, ax, cell in zip('abcdef', axs, cells):
        lay.add(k, ax, cell, restore={} if k in 'ace' else {'ylabels': True})
    # Right-hand panels share the row labels of their left neighbour: equal plot widths.
    lay.fit(['a', 'c', 'e'])
    fig.canvas.draw()
    for left_ax, right_ax, cell in [(a, bb, cells[1]), (c, d, cells[3]), (e, f, cells[5])]:
        pl = left_ax.get_position(); pr = right_ax.get_position()
        right_ax.set_position([cell[2] - pl.width - .005, pl.y0, pl.width, pl.height])
    for k, cell in zip('bdf', [cells[1], cells[3], cells[5]]):
        p = next(q for q in lay.panels if q['key'] == k); p['fixed'] = True
        p['cell'] = Bbox.from_extents(*cell)
    lay.letters(dx=.0)
    for t in fig.texts:
        if t.get_gid() == 'panel_letter' and t.get_text() in 'bdf':
            ax_ = {'b': bb, 'd': d, 'f': f}[t.get_text()]
            t.set_position((ax_.get_position().x0 - .012, t.get_position()[1]))
    fig._free_alignment = True

    def legend_axes(ax):
        return
    finish(fig, lay, name, [RES8 / 'all_model_metrics.csv', RES8 / 'paired_R2_differences.csv', RES8 / 'SSE_decomposition.csv',
                            RES8 / 'continuous_observed_windows.csv', RES8 / 'all_daily_predictions.csv',
                            RES8 / 'all_cycle_predictions.csv'], legend_axes=legend_axes)


# =========================================================================== Fig. 6
GROUPS = ['Temperature', 'Water', 'Atmosphere', 'Calendar', 'Crop', 'Fertilizer']
GLABEL = ['Temperature', 'Water and rainfall', 'Humidity, wind, radiation', 'Calendar position', 'Crop and tillage',
          'Fertilizer history']
FNAME = {'W': 'WFPS', 'T': 'Soil T', 'crop_MAIZE': 'Maize indicator', 'Imin_sum60': 'Mineral N 60 d',
         'DSFA': 'Days since N input', 'crop_age': 'Crop age', 'Ta_max': 'Max air T', 'temp_30': 'Air T 30 d',
         'Imin_exp60': 'Mineral N exp. 60 d', 'Imin_exp30': 'Mineral N exp. 30 d', 'doy_cos': 'DOY cosine',
         'doy_sin': 'DOY sine', 'rain_90': 'Rain 90 d', 'rain_60': 'Rain 60 d', 'Rainfall_sum': 'Rain',
         'Rh_max': 'Max RH', 'Ta_min': 'Min air T'}


def reliance_panel(ax, folds, pool, component, role, label_left):
    p = folds.loc[folds.component.eq(component)]; pooled = pool.loc[pool.component.eq(component)].set_index('group')
    for i, g in enumerate(GROUPS):
        y = 5 - i; v = p.loc[p.group.eq(g)].sort_values('season').delta_rmse.to_numpy()
        ax.plot([v.min(), v.max()], [y, y], color=COLOR[role], lw=1.8, zorder=3)
        ax.scatter(v, y + np.linspace(-.2, .2, 8), s=36, color=COLOR[role], edgecolors=INK, linewidths=.5, zorder=4)
        ax.scatter(pooled.loc[g, 'pooled_delta_rmse'], y, marker='D', s=110, color=COLOR[role], edgecolors=INK,
                   linewidths=.9, zorder=6)
    ax.axvline(0, color=INK, lw=.9, ls=(0, (3, 2)))
    ax.set_xscale('symlog', linthresh=1, linscale=1)
    tk = np.array([-3, -1, 0, 1, 3, 10])
    tk = tk[(tk >= min(-1, p.delta_rmse.min() * 1.1)) & (tk <= max(1, p.delta_rmse.max() * 1.1))]
    ax.set_xticks(tk, [str(int(t)) for t in tk]); ax.minorticks_off()
    ax.set_xlim(min(-1.2, p.delta_rmse.min() * 1.25), max(1.2, p.delta_rmse.max() * 1.25))
    style(ax, f'Δ {"cycle-level" if component == "level" else "daily-shape"} RMSE ({UNIT})')
    cat_axis(ax, 6, GLABEL[::-1], label_left=label_left)


def fig6():
    name = 'Fig6_driver_reliance'
    folds = read(RES11 / 'temporal_reliance_by_cycle.csv'); pool = read(RES11 / 'temporal_reliance_pooled.csv')
    maps = read(RES11 / 'soil_nonadditivity_bins.csv'); edges = json.loads((RES11 / 'soil_map_edges.json').read_text())
    shares = read(RES11 / 'tree_shap_feature_shares.csv')
    fig = plt.figure(figsize=(K.FIG_W, 13.82)); lay = Layout(fig)
    rows = grid_cells(.045, .995, .985, [.25, .313, .333], [[.585, .415], [.585, .415], [.5, .5]], hgap=.06, vgap=.03)
    cells = [r for row in rows for r in row]
    a, bb, c, d, e, f = [fig.add_axes([.1, .1, .3, .3]) for _ in range(6)]
    reliance_panel(a, folds, pool, 'level', 'Level', True)
    reliance_panel(bb, folds, pool, 'shape', 'Shape', False)
    hs = [Line2D([], [], color='white', marker='o', ls='', ms=7, markeredgecolor=INK),
          Line2D([], [], color='white', marker='D', ls='', ms=9, markeredgecolor=INK)]
    a.legend(hs, ['Held-out cycle', 'Pooled'], loc='lower center', bbox_to_anchor=(.6, 1.0), ncol=2, fontsize=TICK)
    # c group attribution shares
    gs = shares.groupby(['season', 'group'], as_index=False).share_pct.sum()
    for i, g in enumerate(GROUPS):
        v = gs.loc[gs.group.eq(g)].sort_values('season').share_pct.to_numpy(); y = 5 - i
        c.barh(y, v.mean(), height=.62, color=COLOR[g], edgecolor=INK, lw=.7, zorder=3)
        c.scatter(v, y + np.linspace(-.18, .18, 8), s=28, color='white', edgecolors=INK, linewidths=.6, zorder=4)
    style(c, 'LightGBM absolute attribution share (%)'); cat_axis(c, 6, GLABEL[::-1]); nticks(c, y=False, n=4)
    c.set_xlim(0, 37)
    c.legend([Rectangle((0, 0), 1, 1, fc='white', ec=INK), Line2D([], [], color='white', marker='o', ls='', ms=7,
              markeredgecolor=INK)], ['Equal-cycle mean', 'Held-out cycle'], loc='lower center',
             bbox_to_anchor=(.6, 1.0), ncol=2, fontsize=TICK, handlelength=1.4)
    # d soil-input coverage
    cov = maps.loc[maps['head'].eq('Level')].copy()
    for r in cov.itertuples():
        d.add_patch(Rectangle((r.T_low, r.W_low), r.T_high - r.T_low, r.W_high - r.W_low,
                              fc=COLOR['Support'] if r.supported else 'white', alpha=.28 if r.supported else 1,
                              ec='white', lw=1.5, zorder=1))
        xx, yy = (r.T_low + r.T_high) / 2, (r.W_low + r.W_high) / 2
        role = 'Support' if r.supported else 'BelowSupport'
        d.scatter(xx, yy, s=r.n * 2.1, color=COLOR[role], edgecolors=INK, linewidths=.7, zorder=4)
        big = r.n >= 90
        d.annotate(str(r.n_cycles), (xx, yy), xytext=(0, 0 if big else 9), textcoords='offset points', fontsize=TICK,
                   ha='center', va='center' if big else 'bottom', zorder=5)
    d.set_xlim(edges['T'][0], edges['T'][-1]); d.set_ylim(edges['W'][0], edges['W'][-1])
    style(d, 'Soil temperature (°C)', 'WFPS (%)'); nticks(d, n=4)
    hs = [Line2D([], [], color=COLOR[r], marker='o', ls='', ms=9, markeredgecolor=INK) for r in ['Support', 'BelowSupport']]
    d.legend(hs, ['Supported', 'Below support'], loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=TICK)
    # e TreeSHAP interaction heatmap
    mat = read(RES11 / 'interaction_matrix_by_fold.csv')
    rank = shares.groupby('feature').share_pct.mean().sort_values(ascending=False)
    top = list(rank.index[:6])
    for req in ['Imin_sum60', 'W']:
        if req not in top:
            top[-1 if top[-1] != 'Imin_sum60' else -2] = req
    mean = mat.groupby(['feature', 'other_feature']).display_magnitude.mean()
    ncol = len(top); table = []
    hm = np.full((ncol, ncol), np.nan)
    for i, fi in enumerate(top):
        for j, fj in enumerate(top):
            if j >= i:
                hm[i, j] = mean.loc[(fi, fj)]
                table.append(dict(feature=fi, other_feature=fj, magnitude=hm[i, j], component='main' if i == j else 'pair'))
    cmh = K.cmap('#FFFFFF', TEAL, PURPLE, name='shap_int')
    im = e.imshow(hm, cmap=cmh, norm=PowerNorm(.5, vmin=0, vmax=np.nanmax(hm)), origin='upper', aspect='equal')
    im.set_gid('continuous')
    for i in range(ncol):
        for j in range(i, ncol):
            e.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK if i == j else 'white',
                                  lw=1.6 if i == j else 1.2, zorder=3))
            e.text(j, i, f'{hm[i, j]:.2f}', ha='center', va='center', fontsize=TICK, zorder=4)
    e.set_xticks(range(ncol), [FNAME.get(x, x) for x in top], rotation=40, ha='right', rotation_mode='anchor')
    e.set_yticks(range(ncol), [FNAME.get(x, x) for x in top])
    style(e); e.tick_params(length=0); e.spines['left'].set_visible(False); e.spines['bottom'].set_visible(False)
    e.set_xlim(-.5, ncol - .5); e.set_ylim(ncol - .5, -.5)
    cax = fig.add_axes([.2, .2, .02, .1])
    cb = fig.colorbar(im, cax=cax, orientation='vertical'); cb.outline.set_linewidth(1); cb.outline.set_edgecolor(INK)
    cb.set_ticks([0, .05, .1, .2]); cb.ax.tick_params(labelsize=TICK, length=4)
    cb.set_label('Mean |TreeSHAP| (model SD)', fontsize=TICK)
    save(pd.DataFrame(table), FIGDIR / name / 'displayed_shap_matrix.csv')
    # f mineral-N x WFPS pair contributions
    dd = read(RES11 / 'nitrogen_water_interactions.csv')
    for g in ['Low', 'Middle', 'High']:
        q = dd.loc[dd.WFPS_group.eq(g)]
        f.scatter(q.Imin_sum60, q.pair_interaction, s=18, color=COLOR[g], edgecolors=INK, linewidths=.2, alpha=.55, zorder=3)
    grp = dd.assign(dose=dd.Imin_sum60.round(5)).groupby(['WFPS_group', 'dose', 'season'], as_index=False).pair_interaction.mean()
    means = grp.groupby(['WFPS_group', 'dose']).agg(mean=('pair_interaction', 'mean'), n_cycles=('season', 'nunique')).reset_index()
    for g in ['Low', 'Middle', 'High']:
        q = means.loc[means.WFPS_group.eq(g) & means.n_cycles.ge(3)]
        f.scatter(q.dose, q['mean'], marker='D', s=110, color=COLOR[g], edgecolors=INK, linewidths=1, zorder=6)
    f.axhline(0, color=INK, lw=.9, ls=(0, (3, 2)))
    style(f, '60-day mineral-N input (kg N ha$^{-1}$)', 'Pair contribution (model SD)', grid='y'); nticks(f, n=4)
    hs = [Line2D([], [], color=COLOR[g], marker='o', ls='', ms=9, markeredgecolor=INK) for g in ['Low', 'Middle', 'High']]
    hs.append(Line2D([], [], color='white', marker='D', ls='', ms=9, markeredgecolor=INK))
    f.legend(hs, ['Low WFPS', 'Middle', 'High', 'Cycle mean'], loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4,
             fontsize=TICK, columnspacing=.9, handletextpad=.3)
    save(means, FIGDIR / name / 'nitrogen_water_dose_means.csv')
    save(gs, FIGDIR / name / 'tree_shap_group_shares.csv'); save(cov, FIGDIR / name / 'soil_input_coverage.csv')
    # layout ---------------------------------------------------------------
    lay.add('a', a, cells[0]); lay.add('b', bb, cells[1], restore={'ylabels': True})
    lay.add('c', c, cells[2]); lay.add('d', d, cells[3])
    lay.add('e', [e, cax], cells[4]); lay.add('f', f, cells[5])
    # heatmap: position colourbar next to the matrix before fitting the group
    fig.canvas.draw(); pe_ = e.get_position()
    cax.set_position([pe_.x1 + .015, pe_.y0, .015, pe_.height])
    lay.fit(['a', 'c', 'd', 'f']); lay.fit(['e'], keep_aspect=('e',))
    fig.canvas.draw(); pe_ = e.get_position(); cax.set_position([pe_.x1 + .012, pe_.y0 + .1 * pe_.height, .013, .8 * pe_.height])
    pl = a.get_position(); cell = cells[1]
    bb.set_position([cell[2] - pl.width - .005, pl.y0, pl.width, pl.height])
    lay.panels[1]['fixed'] = True
    lay.letters(dx=-.004)
    for t in fig.texts:
        if t.get_gid() == 'panel_letter' and t.get_text() == 'b':
            t.set_position((bb.get_position().x0 - .012, t.get_position()[1]))
        if t.get_gid() == 'panel_letter' and t.get_text() == 'e':
            t.set_position((t.get_position()[0] - .004, t.get_position()[1] + .018))
    fig._free_alignment = True

    def legend_axes(ax):
        return
    finish(fig, lay, name, [RES11 / 'temporal_reliance_by_cycle.csv', RES11 / 'temporal_reliance_pooled.csv',
                            RES11 / 'soil_nonadditivity_bins.csv', RES11 / 'tree_shap_feature_shares.csv',
                            RES11 / 'interaction_matrix_by_fold.csv', RES11 / 'nitrogen_water_interactions.csv'],
           legend_axes=legend_axes)


# =========================================================================== SI figures
def simple_grid(fig, rows, keys, *, keep=()):
    lay = Layout(fig); cells = [r for row in rows for r in row]
    axs = [fig.add_axes([.1, .1, .3, .3]) for _ in keys]
    return lay, cells, axs


def figs1():
    name = 'FigS1_observation_support'
    fig = plt.figure(figsize=(K.FIG_W, 7.4))
    rows = grid_cells(.045, .995, .94, [.9], [[.64, .36]], hgap=.06)
    lay, cells, (a, bb) = simple_grid(fig, rows, 'ab')
    base = PRED['OriginalLightGBM']; out = []
    labels = []
    for i, (year, tr) in enumerate([(y, t) for y in range(2014, 2022) for t in ['AN', 'PS']]):
        g = base.query('season==@year and treatment==@tr').copy()
        g['day'] = (g.Date - pd.Timestamp(MAN.loc[year, 'start'])).dt.days
        y = 15 - i; labels.append(f'{year} {tr}')
        if (i // 2) % 2 == 0:
            for ax in (a, bb):
                ax.axhspan(y - .5, y + .5, color=LIGHT, lw=0, zorder=0)
        a.broken_barh([(0, len(g))], (y - .3, .6), facecolors=COLOR['Missing'], edgecolor='none', zorder=2)
        obs = g.loc[g.y.notna(), 'day'].to_numpy()
        runs = np.split(obs, np.where(np.diff(obs) > 1)[0] + 1)
        a.broken_barh([(r[0], len(r)) for r in runs if len(r)], (y - .3, .6), facecolors=COLOR['Observed'],
                      edgecolor=INK, lw=.35, zorder=3)
        n = int(g.y.notna().sum()); nd = len(g); fr = 100 * n / nd
        bb.plot([0, fr], [y, y], color=MUTED, lw=1, zorder=2)
        bb.scatter(fr, y, s=80, color=COLOR['Observed'], edgecolors=INK, linewidths=.8, zorder=4)
        bb.text(103, y, f'{n}/{nd}', va='center', fontsize=TICK)
        out.append(dict(season=year, treatment=tr, n_observed=n, n_calendar=nd, coverage_pct=fr))
    for ax in (a, bb):
        ax.set_yticks(range(16), labels[::-1]); ax.set_ylim(-.6, 15.6)
    a.set_xlim(0, 420); a.set_xticks([0, 100, 200, 300, 400])
    style(a, 'Days from cycle start'); a.tick_params(axis='y', length=0); a.spines['left'].set_visible(False)
    a.legend([Rectangle((0, 0), 1, 1, fc=COLOR[k], ec=INK if k == 'Observed' else 'none', lw=.4) for k in ['Observed', 'Missing']],
             ['Observed', 'Not observed'], loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=TICK)
    bb.set_xlim(0, 125); bb.set_xticks([0, 25, 50, 75, 100])
    style(bb, 'Observed days (%)'); bb.tick_params(axis='y', length=0); bb.spines['left'].set_visible(False)
    bb.text(103, 15.9, 'n / days', fontsize=TICK, va='bottom')
    save(pd.DataFrame(out), FIGDIR / name / 'support_counts.csv')
    lay.add('a', a, cells[0]); lay.add('b', bb, cells[1]); lay.fit(); lay.letters(dx=-.012)
    finish(fig, lay, name, [RES8 / 'all_daily_predictions.csv', RES8 / 'season_manifest.csv'])


def figs2():
    name = 'FigS2_loss_and_shrinkage'
    fig = plt.figure(figsize=(K.FIG_W, 8.92))
    rows = grid_cells(.05, .995, .985, [.454, .454], [[.5, .5]] * 2, hgap=.08, vgap=.05)
    lay, cells, axs = simple_grid(fig, rows, 'abcd')
    m07 = read(R07 / 'all_model_metrics.csv').set_index('model'); rec = []
    for r_, (models, table, role, param) in enumerate([
            (['Joint01', 'Joint1', 'Joint10'], m07, 'Joint01', 'Cycle-loss weight'),
            (['DecoupledPulseWeak', 'DecoupledPulse', 'DecoupledPulseStrong'], M, 'DecoupledPulse', 'Shrinkage weight')]):
        for c_, scale in enumerate(['cycle', 'daily']):
            ax = axs[2 * r_ + c_]; xx = [.1, 1, 10]; yy = [table.loc[m, f'R2_{scale}'] for m in models]
            ref = M.loc['OriginalLightGBM', f'R2_{scale}']
            ax.axhline(ref, color=COLOR['OriginalLightGBM'], lw=2, ls=(0, (3.2, 1.6)), zorder=2)
            ax.text(13, ref, ' LightGBM', va='center', ha='left', fontsize=TICK)
            ax.plot(xx, yy, color=COLOR[role], lw=2.6, marker=MARKER[role], ms=12 if role == 'DecoupledPulse' else 10,
                    markeredgecolor=INK, markeredgewidth=.8, zorder=4)
            ax.set_xscale('log'); ax.set_xticks(xx, ['0.1', '1', '10']); ax.minorticks_off(); ax.set_xlim(.07, 13)
            if r_ == 0:
                ax.set_ylim((.1, .85) if c_ == 0 else (-.03, .4))
            else:
                ax.set_ylim((.70, .84) if c_ == 0 else (.26, .365))
            style(ax, param, f'{scale.capitalize()} prediction R²', grid='y'); nticks(ax, x=False, n=4)
            ax.text(.03, .04, LABEL[role] if role != 'DecoupledPulse' else 'Decoupled + pulse', transform=ax.transAxes, fontsize=TICK)
            rec.extend(dict(model=m, parameter=param, weight=x, scale=scale, R2=y) for m, x, y in zip(models, xx, yy))
    save(pd.DataFrame(rec), FIGDIR / name / 'sensitivity_points.csv')
    for k, ax, cell in zip('abcd', axs, cells):
        lay.add(k, ax, cell)
    lay.fit(); lay.letters(dx=-.012)
    finish(fig, lay, name, [R07 / 'all_model_metrics.csv', RES8 / 'all_model_metrics.csv'])


def figs3():
    name = 'FigS3_pulse_response'
    fig = plt.figure(figsize=(K.FIG_W, 9.36))
    rows = grid_cells(.05, .995, .95, [.42, .451], [[.5, .5]] * 2, hgap=.08, vgap=.05)
    lay, cells, axs = simple_grid(fig, rows, 'abcd')
    taus = np.array([3, 7, 14, 30, 90]); days = np.arange(181)
    kern = (-np.expm1(-1 / taus))[None, :] * np.exp(-days[:, None] / taus)
    wts, resp = [], []
    for year in range(2014, 2022):
        mdl = json.loads((R08 / f'models/{year}_DecoupledPulse.json').read_text(encoding='utf-8'))
        for branch, off, sc, nm in [('level', 2, 'sl', 'norm_l'), ('shape', 1, 'sd', 'norm_s')]:
            beta = np.array(mdl['beta_l' if branch == 'level' else 'beta_s'])[off:] * mdl[sc] / np.array(mdl[nm])
            for form, j in [('Nmin', 0), ('Norg', 5)]:
                wts.extend(dict(held=year, branch=branch, form=form, tau=int(t), coefficient=float(v))
                           for t, v in zip(taus, beta[j:j + 5]))
                if branch == 'shape':
                    rr = kern @ beta[j:j + 5]
                    resp.extend(dict(held=year, form=form, day=int(dd), response=float(v)) for dd, v in zip(days, rr))
    wts = pd.DataFrame(wts); resp = pd.DataFrame(resp)
    for ax, form in zip(axs[:2], ['Nmin', 'Norg']):
        rr = resp.loc[resp.form.eq(form)].pivot(index='day', columns='held', values='response')
        ax.fill_between(days, rr.min(axis=1), rr.max(axis=1), color=COLOR[form], alpha=.3, lw=0, zorder=2)
        ax.plot(days, rr.median(axis=1), color=COLOR[form], lw=2.8, zorder=4)
        ax.plot(days[::30], rr.median(axis=1).iloc[::30], ls='', marker='o' if form == 'Nmin' else 's', ms=7,
                color=COLOR[form], markeredgecolor=INK, zorder=5)
        up = float(rr.max().max()); ax.set_xlim(0, 180); ax.set_ylim(-.04 * up, 1.12 * up); nticks(ax, n=4)
        style(ax, 'Days after application', f'Pulse increment ({UNIT})')
        ax.text(.97, .95, ('Mineral N' if form == 'Nmin' else 'Organic N') + '\n1 kg N ha$^{-1}$, 20 °C, 60% WFPS',
                transform=ax.transAxes, ha='right', va='top', fontsize=TICK, linespacing=1.4)
    for ax, branch in zip(axs[2:], ['level', 'shape']):
        for k, (form, mk) in enumerate([('Nmin', 'o'), ('Norg', 's')]):
            rr = wts.query('branch==@branch and form==@form').pivot(index='tau', columns='held', values='coefficient')
            xs = taus * (1.06 if k else 1 / 1.06)
            ax.vlines(xs, rr.min(axis=1), rr.max(axis=1), color=COLOR[form], lw=4, alpha=.75, zorder=2)
            ax.plot(xs, rr.median(axis=1), color=COLOR[form], lw=2.2, marker=mk, ms=9, markeredgecolor=INK,
                    markeredgewidth=.8, zorder=4)
        ax.set_xscale('log'); ax.set_xticks(taus, [str(v) for v in taus]); ax.minorticks_off()
        up = float(wts.loc[wts.branch.eq(branch), 'coefficient'].max()); ax.set_ylim(-.04 * up, 1.1 * up)
        nticks(ax, x=False, n=4)
        style(ax, 'Kernel time scale (days)', 'Kernel coefficient\n(g N$_2$O-N kg$^{-1}$ N d$^{-1}$)', grid='y')
        ax.text(.03, .96, f'{branch.capitalize()} branch', transform=ax.transAxes, fontsize=TICK, va='top')
    hs = [Line2D([], [], color=COLOR[k], marker=m, markeredgecolor=INK, lw=2.4, ms=9) for k, m in [('Nmin', 'o'), ('Norg', 's')]]
    fig.legend(hs, ['Mineral N', 'Organic N'], loc='upper center', bbox_to_anchor=(.53, 1.0), ncol=2, fontsize=BASE)
    save(wts, FIGDIR / name / 'physical_coefficients.csv'); save(resp, FIGDIR / name / 'unit_responses.csv')
    for k, ax, cell in zip('abcd', axs, cells):
        lay.add(k, ax, cell)
    lay.fit(); lay.letters(dx=-.012)

    def legend_axes(ax):
        ax.legend(hs, ['Mineral N', 'Organic N'], loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=TICK)
    finish(fig, lay, name, sorted((R08 / 'models').glob('*_DecoupledPulse.json')), legend_axes=legend_axes)


def figs4():
    name = 'FigS4_observed_windows'
    fig = plt.figure(figsize=(K.FIG_W, 6.6))
    rows = grid_cells(.05, .995, .985, [.95], [[.5, .5]], hgap=.08)
    lay, cells, (a, bb) = simple_grid(fig, rows, 'ab')
    models = ['OriginalLightGBM', 'AffineReconciled']
    for m, mk in zip(models, ['o', '^']):
        g = W.loc[W.model.eq(m)]
        a.scatter(g.observed_integral, g.predicted_integral, marker=mk, s=48, color=COLOR[m], edgecolors=INK,
                  linewidths=.55, alpha=.9, zorder=4 if m == 'AffineReconciled' else 3)
    a.plot([-160, 5000], [-160, 5000], color=INK, ls=(0, (3, 2)), lw=1)
    a.set_xscale('symlog', linthresh=50); a.set_yscale('symlog', linthresh=50)
    a.set_xlim(-160, 5000); a.set_ylim(-160, 5000)
    tk = [-100, 0, 50, 200, 1000, 4000]
    a.set_xticks(tk, [str(v) for v in tk]); a.set_yticks(tk, [str(v) for v in tk]); a.minorticks_off()
    style(a, f'Observed window sum ({SUMUNIT})', f'Predicted window sum ({SUMUNIT})', grid='both')
    a.legend([Line2D([], [], color=COLOR[m], marker=k, markeredgecolor=INK, ls='', ms=9) for m, k in zip(models, ['o', '^'])],
             [LABEL[m] for m in models], loc='upper left', fontsize=TICK)
    keys = ['season', 'treatment', 'start', 'end', 'n_days']
    L_ = W.loc[W.model.eq('OriginalLightGBM')].set_index(keys); R_ = W.loc[W.model.eq('AffineReconciled')].set_index(keys)
    pd.testing.assert_index_equal(L_.index, R_.index)
    delta = (R_.abs_error - L_.abs_error).rename('delta_absolute_integral_error').to_frame().reset_index()
    delta['delta_absolute_mean_error'] = delta.delta_absolute_integral_error / delta.n_days
    worse = delta.delta_absolute_mean_error > 0
    bb.axhspan(0, 23, color=LIGHT, lw=0, zorder=0)
    bb.scatter(delta.n_days, delta.delta_absolute_mean_error, s=55, color=COLOR['AffineReconciled'], edgecolors=INK,
               linewidths=.6, alpha=.9, zorder=4)
    bb.axhline(0, color=INK, ls=(0, (3, 2)), lw=1); bb.set_xlim(3, 92); bb.set_ylim(-23, 19); nticks(bb, n=4)
    style(bb, 'Continuous observed days', f'Δ absolute window-mean error\n({UNIT})')
    bb.text(.97, .96, f'Affine larger error: {worse.sum()} of {len(delta)}', transform=bb.transAxes, ha='right', va='top',
            fontsize=TICK)
    bb.text(.97, .04, f'Affine smaller error: {(delta.delta_absolute_mean_error < 0).sum()} of {len(delta)}', transform=bb.transAxes, ha='right',
            va='bottom', fontsize=TICK)
    save(W.loc[W.model.isin(models)], FIGDIR / name / 'observed_window_sums.csv')
    save(delta, FIGDIR / name / 'paired_window_error_changes.csv')
    lay.add('a', a, cells[0]); lay.add('b', bb, cells[1]); lay.fit(); lay.letters(dx=-.012)
    finish(fig, lay, name, [RES8 / 'continuous_observed_windows.csv'])


def figs5():
    name = 'FigS5_local_soil_responses'
    curves = read(RES11 / 'local_response_curves.csv')
    fig = plt.figure(figsize=(K.FIG_W, 9.12))
    rows = grid_cells(.05, .995, .985, [.454, .454], [[.5, .5]] * 2, hgap=.08, vgap=.05)
    lay, cells, axs = simple_grid(fig, rows, 'abcd')
    for ax, (head, var) in zip(axs, [('Level', 'T'), ('Level', 'W'), ('Shape', 'T'), ('Shape', 'W')]):
        g = curves.loc[curves['head'].eq(head) & curves.variable.eq(var) & curves.supported]
        allx = curves.loc[curves.variable.eq(var)].groupby('bin').x.first()
        for _, part in g.groupby('season'):
            part = part.set_index('bin').reindex(allx.index)
            ax.plot(allx.to_numpy(), part.value, color=COLOR[head], lw=.9, alpha=.75, zorder=2)
        v = g.groupby('bin').agg(mean=('value', 'mean'), low=('value', 'min'), high=('value', 'max')).reindex(allx.index)
        ax.fill_between(allx.to_numpy(), v.low, v.high, color=COLOR[head], alpha=.25, lw=0, zorder=1)
        ax.plot(allx.to_numpy(), v['mean'], color=COLOR[head], lw=3, marker='o', ms=8, markeredgecolor=INK,
                markeredgewidth=.8, zorder=4)
        outlined(ax.lines[-1], 3)
        ax.axhline(0, color=INK, ls=(0, (3, 2)), lw=.9)
        style(ax, 'Soil temperature (°C)' if var == 'T' else 'WFPS (%)', f'{head} response ({UNIT})', grid='y')
        nticks(ax, n=4)
        ax.text(.03, .95, ('+1 °C' if var == 'T' else '+5 WFPS points') + (', calibrated LightGBM' if head == 'Level' else ', centered TabPFN'),
                transform=ax.transAxes, ha='left', va='top', fontsize=TICK)
    save(curves, FIGDIR / name / 'local_response_curves.csv')
    for k, ax, cell in zip('abcd', axs, cells):
        lay.add(k, ax, cell)
    lay.fit(); lay.letters(dx=-.012)
    finish(fig, lay, name, [RES11 / 'local_responses.csv', RES11 / 'local_response_curves.csv'])


def figs6():
    name = 'FigS6_tree_attributions'
    shares = read(RES11 / 'tree_shap_feature_shares.csv'); data = read(RES11 / 'tree_shap_heldout.csv')
    rank = shares.groupby('feature').share_pct.mean().sort_values(ascending=False); top = rank.head(12).index.tolist()
    gmap = shares.drop_duplicates('feature').set_index('feature')['group'].to_dict()
    names = {'T': 'Soil temperature', 'W': 'WFPS', 'DSFA': 'Days since fertilization', 'crop_age': 'Crop age',
             'doy_sin': 'Day-of-year sine', 'doy_cos': 'Day-of-year cosine', 'crop_MAIZE': 'Maize indicator',
             'crop_WHEAT': 'Wheat indicator', 'tillage_14': 'Tillage memory (14 d)', 'Rainfall_sum': 'Daily rainfall'}

    def lab(ft):
        if ft in names: return names[ft]
        if ft.startswith('rain_'): return f'{ft.split("_")[-1]}-day rainfall'
        if ft.startswith('temp_'): return f'{ft.split("_")[-1]}-day air temperature'
        for pre, kind in [('Imin', 'mineral'), ('Iorg', 'organic')]:
            if ft.startswith(pre + '_sum'): return f'{ft.split("sum")[-1]}-day {kind} N input'
            if ft.startswith(pre + '_exp'): return f'{kind.capitalize()} N memory ({ft.split("exp")[-1]} d)'
        return ft
    fig = plt.figure(figsize=(K.FIG_W, 8.2))
    rows = grid_cells(.04, .995, .94, [.9], [[.52, .48]], hgap=.05)
    lay, cells, (a, c) = simple_grid(fig, rows, 'ab')
    cm = K.cmap(BLUE, '#FFFFFF', ROSE, name='feature_pct')
    rng = np.random.default_rng(20261018)
    for i, ft in enumerate(top):
        y = 11 - i; role = gmap[ft]
        if i % 2 == 0:
            for ax in (a, c):
                ax.axhspan(y - .5, y + .5, color=LIGHT, lw=0, zorder=0)
        a.barh(y, rank[ft], color=COLOR[role], height=.62, edgecolor=INK, lw=.7, zorder=3)
        v = shares.loc[shares.feature.eq(ft)].sort_values('season').share_pct.to_numpy()
        a.scatter(v, y + np.linspace(-.18, .18, 8), s=24, color='white', edgecolors=INK, linewidths=.55, zorder=4)
        p = data.loc[data.feature.eq(ft)]; x = p.shap_normalized.to_numpy()
        h, ed = np.histogram(x, bins=45); ix = np.clip(np.searchsorted(ed, x, side='right') - 1, 0, len(h) - 1)
        jit = rng.uniform(-1, 1, len(x)) * .36 * h[ix] / max(h.max(), 1)
        sc = c.scatter(x, y + jit, c=p.training_percentile, cmap=cm, vmin=0, vmax=1, s=7, linewidths=0, zorder=3)
        sc.set_gid('continuous')
    for ax in (a, c):
        ax.set_yticks(range(12), [lab(f) for f in top[::-1]]); ax.set_ylim(-.6, 11.6)
        ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    c.tick_params(axis='y', labelleft=False)
    style(a, 'Mean absolute attribution share (%)'); nticks(a, y=False, n=4); a.tick_params(axis='y', length=0)
    a.spines['left'].set_visible(False)
    style(c, 'Normalized TreeSHAP contribution'); nticks(c, y=False, n=4); c.tick_params(axis='y', length=0, labelleft=False)
    c.spines['left'].set_visible(False)
    c.axvline(0, color=INK, ls=(0, (3, 2)), lw=.9)
    hs = [Rectangle((0, 0), 1, 1, fc=COLOR[g], ec=INK, lw=.6) for g in GROUPS]
    a.legend(hs, ['Temperature', 'Water', 'Atmosphere', 'Calendar', 'Crop', 'Fertilizer'], loc='lower center',
             bbox_to_anchor=(.45, 1.0), ncol=3, fontsize=TICK, handlelength=1.2, columnspacing=1.0)
    p0 = c.get_position()
    cax = fig.add_axes([p0.x0, p0.y1 + .02, p0.width * .5, .015])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cb.ax.set_xticklabels(['0', '50', '100']); cb.outline.set_linewidth(1); cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=TICK, length=4); cb.ax.xaxis.set_ticks_position('top'); cb.ax.xaxis.set_label_position('top')
    cb.set_label('Input percentile in training data (%)', fontsize=TICK, labelpad=4)
    save(shares, FIGDIR / name / 'all_42_feature_shares.csv')
    save(rank.rename('mean_share_pct').reset_index(), FIGDIR / name / 'feature_ranking.csv')
    lay.add('a', a, cells[0]); lay.add('b', [c, cax], cells[1], restore={'ylabels': True})
    fig.canvas.draw()
    lay.fit(['a'])
    # b shares row positions with a; colour bar sits above b.
    pa = a.get_position(); cell = cells[1]
    c.set_position([cell[0] + .005, pa.y0, cell[2] - cell[0] - .01, pa.height])
    pc = c.get_position(); cax.set_position([pc.x0 + .25 * pc.width, pa.y1 + .045, .5 * pc.width, .016])
    lay.panels[1]['fixed'] = True
    lay.letters(dx=-.006)
    for t in fig.texts:
        if t.get_gid() == 'panel_letter' and t.get_text() == 'b':
            t.set_position((pc.x0 - .01, t.get_position()[1]))
    fig._free_alignment = True

    def legend_axes(ax):
        if ax.collections and ax.collections[-1].get_gid() == 'continuous':
            ax.set_yticks(range(12), [lab(f) for f in top[::-1]])
    finish(fig, lay, name, [RES11 / 'tree_shap_heldout.csv', RES11 / 'tree_shap_feature_shares.csv'], legend_axes=legend_axes)


BUILDERS = {'Fig1': fig1, 'Fig2': fig2, 'Fig3': fig3, 'Fig4': fig4, 'Fig5': fig5, 'Fig6': fig6,
            'FigS1': figs1, 'FigS2': figs2, 'FigS3': figs3, 'FigS4': figs4, 'FigS5': figs5, 'FigS6': figs6}


def main():
    global PREVIEW
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='+'); ap.add_argument('--preview', action='store_true')
    args = ap.parse_args(); PREVIEW = args.preview
    K.setup()
    for k in args.only or BUILDERS:
        BUILDERS[k]()


if __name__ == '__main__':
    main()
