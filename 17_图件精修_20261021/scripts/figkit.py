"""Figure layout helpers for the EFELE N2O paper.

Uses a fixed publication palette, shared legends and aligned panel cells.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba_array
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'portable_helpers'))
from scientific_palettes import get_palette, STYLE_COLORS  # noqa: E402

PALETTE = 'nature_new'
PAL = get_palette(PALETTE)  # rose, blue, green, teal, purple, pink, gold, mauve
ROSE, BLUE, GREEN, TEAL, PURPLE, PINK, GOLD, MAUVE = PAL
INK, MUTED, LIGHT = STYLE_COLORS['ink'], STYLE_COLORS['muted'], STYLE_COLORS['light']
GRID = '#D9DCE0'
NEUTRAL = [INK, MUTED, LIGHT, GRID, '#FFFFFF', '#000000', '#B7B7B7', '#EEF0F2']

FIG_W = 13.0          # inches; printed at 190 mm => x0.574
BASE = 14.0           # body type, ~8.0 pt at 190 mm
TICK = 13.0           # ~7.5 pt
LETTER = 24.0

# One project-wide role map; legend proxies are drawn from the same entries.
COLOR = {
    'OriginalLightGBM': BLUE, 'TabPFN_DailyRaw': TEAL, 'Joint01': MAUVE, 'BlendCycle': PINK,
    'MeanReconciled': GOLD, 'DecoupledPulse': PURPLE, 'AffineReconciled': ROSE,
    'DecoupledNoPulse': GREEN, 'DecoupledUnconstrained': MAUVE, 'TiedPulse': PINK,
    'PulseShapeOnly': TEAL, 'PulseLevelOnly': BLUE,
    'Observed': GREEN, 'Missing': PINK, 'Support': PURPLE, 'BelowSupport': PINK,
    'Level': GOLD, 'Shape': TEAL, 'Nmin': PURPLE, 'Norg': MAUVE,
    'Wheat': GOLD, 'Maize': GREEN, 'Mustard': PINK, 'External': TEAL,
    'Temperature': ROSE, 'Water': BLUE, 'Atmosphere': TEAL, 'Calendar': GOLD,
    'Crop': GREEN, 'Fertilizer': PURPLE, 'Low': BLUE, 'Middle': GOLD, 'High': ROSE,
    'Drivers': GREEN, 'Training': PURPLE, 'Output': ROSE,
}
MARKER = {'OriginalLightGBM': 'o', 'TabPFN_DailyRaw': 'D', 'MeanReconciled': 's',
          'AffineReconciled': '^', 'DecoupledPulse': '*', 'Joint01': 'p', 'BlendCycle': 'h'}


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'axes.unicode_minus': False,
        'font.size': BASE, 'axes.labelsize': BASE, 'axes.titlesize': BASE,
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'legend.fontsize': TICK,
        'axes.labelweight': 'normal', 'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.linewidth': 1.3, 'xtick.major.width': 1.3, 'ytick.major.width': 1.3,
        'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 3.5, 'ytick.minor.size': 3.5,
        'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0, 'xtick.major.pad': 4.5, 'ytick.major.pad': 4.5,
        'axes.labelpad': 6, 'legend.frameon': False, 'legend.handletextpad': .5,
        'legend.columnspacing': 1.3, 'legend.borderaxespad': .3,
        'savefig.dpi': 600, 'figure.dpi': 100, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'hatch.color': INK, 'hatch.linewidth': .6,
    })


def style(ax, xlabel=None, ylabel=None, *, left=True, bottom=True, grid=None):
    ax.grid(False)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(1.3)
    ax.tick_params(axis='both', which='both', direction='out', colors=INK, labelsize=TICK, top=False, right=False)
    ax.tick_params(axis='x', bottom=bottom, labelbottom=bottom)
    ax.tick_params(axis='y', left=left)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=BASE, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=BASE, color=INK)
    if grid in ('x', 'y', 'both'):
        ax.grid(True, axis=grid, color=GRID, lw=.6, zorder=0)
        ax.set_axisbelow(True)


def nticks(ax, x=True, y=True, n=4):
    from matplotlib.ticker import MaxNLocator
    if x: ax.xaxis.set_major_locator(MaxNLocator(nbins=n))
    if y: ax.yaxis.set_major_locator(MaxNLocator(nbins=n))


def bands(ax, n, *, start=0):
    """Pale alternate row bands for long categorical rows (structural colour)."""
    for y in range(start, n, 2):
        ax.axhspan(y - .5, y + .5, color=LIGHT, lw=0, zorder=0)


def cmap(*colors, name='c'):
    return LinearSegmentedColormap.from_list(name, list(colors))


def outlined(artist, width):
    import matplotlib.patheffects as pe
    artist.set_path_effects([pe.Stroke(linewidth=width + .9, foreground=INK), pe.Normal()])
    return artist


# --------------------------------------------------------------------------- layout
class Layout:
    """Register panels (one or more axes) against target outer cells."""

    def __init__(self, fig):
        self.fig = fig
        self.panels = []

    def add(self, key, axes, cell, *, letter=True, letter_xy=None, restore=None):
        axes = list(np.atleast_1d(axes))
        self.panels.append(dict(key=key, axes=axes, cell=Bbox.from_extents(*cell), letter=letter,
                                letter_xy=letter_xy, restore=restore or {}))
        return axes

    def outer(self, axes):
        r = self.fig.canvas.get_renderer()
        boxes = [a.get_tightbbox(r) for a in axes if a.get_visible()]
        return Bbox.union([b for b in boxes if b is not None]).transformed(self.fig.transFigure.inverted())

    def fit(self, keys=None, *, scale_y=True, keep_aspect=()):
        fig = self.fig
        report = []
        for p in self.panels:
            if keys is not None and p['key'] not in keys:
                continue
            if p.get('fixed'):
                continue
            target = p['cell']
            for _ in range(30):
                fig.canvas.draw()
                o = self.outer(p['axes'])
                err = (np.array(o.extents) - np.array(target.extents)) * np.tile(fig.get_size_inches(), 2) * 72
                if np.abs(err).max() < .3:
                    break
                sx = target.width / o.width
                sy = target.height / o.height if scale_y else 1
                if p['key'] in keep_aspect:
                    sx = sy = min(sx, sy)
                ox = (target.width - o.width * sx) / 2 if p['key'] in keep_aspect else 0
                for a in p['axes']:
                    pos = a.get_position()
                    a.set_position([target.x0 + ox + (pos.x0 - o.x0) * sx, target.y1 - o.height * sy + (pos.y0 - o.y0) * sy,
                                    pos.width * sx, pos.height * sy])
            fig.canvas.draw()
            o = self.outer(p['axes'])
            res = (np.array(o.extents) - np.array(target.extents)) * np.tile(fig.get_size_inches(), 2) * 72
            report.append(dict(panel=p['key'], target=list(target.extents), outer=list(o.extents),
                               max_residual_pt=float(np.abs(res).max())))
        return report

    def letters(self, dx=.0, dy=.0):
        for p in self.panels:
            if not p['letter']:
                continue
            x, y = p['letter_xy'] or (p['cell'].x0 + dx, p['cell'].y1 + dy)
            t = self.fig.text(x, y, p['key'], fontsize=LETTER, ha='right', va='top', color=INK)
            t.set_gid('panel_letter')

    def alignment(self):
        """Measure outer boxes again: rows share y extents, columns share x extents."""
        self.fig.canvas.draw()
        rows, cols = {}, {}
        for p in self.panels:
            o = self.outer(p['axes']); c = p['cell']
            rows.setdefault((round(c.y0, 4), round(c.y1, 4)), []).append(o)
            cols.setdefault((round(c.x0, 4), round(c.x1, 4)), []).append(o)
        h, w = self.fig.get_size_inches() * 72
        rerr = max([max(np.ptp([b.y0 for b in v]) * w, np.ptp([b.y1 for b in v]) * w) for v in rows.values()] or [0])
        cerr = max([max(np.ptp([b.x0 for b in v]) * h, np.ptp([b.x1 for b in v]) * h) for v in cols.values()] or [0])
        return dict(row_outer_misalignment_pt=float(rerr), column_outer_misalignment_pt=float(cerr),
                    tolerance_pt=1.0, passed=bool(max(rerr, cerr) < 1.0))


def grid_cells(x0, x1, y_top, heights, widths_by_row, *, hgap=.035, vgap=.05):
    """Return rows of cells (figure fractions). widths_by_row: list of ratios lists."""
    rows, y = [], y_top
    for h, ratios in zip(heights, widths_by_row):
        total = x1 - x0 - hgap * (len(ratios) - 1)
        x, row = x0, []
        for r in ratios:
            w = total * r / sum(ratios)
            row.append((x, y - h, x + w, y))
            x += w + hgap
        rows.append(row)
        y -= h + vgap
    return rows


# --------------------------------------------------------------------------- audit
def audit(fig, extra_ok=()):
    approved = [to_rgb(c) for c in PAL]
    neutral = [to_rgb(c) for c in NEUTRAL]
    bad, inspected = [], 0
    from matplotlib.artist import Artist
    for art in fig.findobj(Artist):
        if not art.get_visible() or isinstance(art, Text) or art.get_gid() in ('continuous',) + tuple(extra_ok):
            continue
        if hasattr(art, 'get_array') and art.get_array() is not None and getattr(art, 'cmap', None) is not None:
            continue  # continuous scales are audited by their declared endpoints
        getters = ['get_color', 'get_facecolor', 'get_edgecolor']
        if isinstance(art, Line2D):
            if art.get_linestyle() in [None, 'None', '', 'none']:
                getters.remove('get_color')
            getters += ['get_markerfacecolor', 'get_markeredgecolor']
        for g in getters:
            if not hasattr(art, g):
                continue
            try:
                cols = to_rgba_array(getattr(art, g)())
            except (TypeError, ValueError):
                continue
            for rgba in cols:
                if rgba[3] == 0:
                    continue
                inspected += 1
                if not any(np.allclose(rgba[:3], c, atol=2e-3) for c in approved + neutral):
                    bad.append(dict(artist=type(art).__name__, getter=g, rgb=[round(v, 4) for v in rgba[:3]]))
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return dict(palette=PALETTE, palette_hex=PAL, inspected_colours=inspected, off_palette=bad,
                passed=not bad, min_font_pt_source=min(sizes), min_font_pt_at_190mm=min(sizes) * 190 / (fig.get_figwidth() * 25.4))


# --------------------------------------------------------------------------- export
def _set_pdf_meta(path):
    with fitz.open(path) as doc:
        meta = doc.metadata; meta.update(author='CHONG LIU', creator='CHONG LIU')
        doc.set_metadata(meta); doc.saveIncr()


def export(fig, layout, folder, stem, *, dpi=600, legend_axes=None):
    """Combined PNG/PDF plus independently re-rendered vector panels."""
    folder = Path(folder); sub = folder / 'subfigures'
    folder.mkdir(parents=True, exist_ok=True); sub.mkdir(exist_ok=True)
    for old in list(folder.glob(f'{stem}.*')) + list(sub.glob('*')):
        old.unlink()
    fig.savefig(folder / f'{stem}.png', dpi=dpi, facecolor='white')
    fig.savefig(folder / f'{stem}.pdf', facecolor='white')
    _set_pdf_meta(folder / f'{stem}.pdf')
    blob = pickle.dumps(fig)
    index = {id(a): i for i, a in enumerate(fig.axes)}
    lines = ['# Independent subfigures', '',
             'Each panel is re-rendered from its own vector artists on a separate canvas of the same physical',
             'size as its cell in the combined figure (type sizes unchanged). No raster cropping.', '']
    fig.canvas.draw()
    for p in layout.panels:
        keep = [index[id(a)] for a in p['axes']]
        o = layout.outer(p['axes'])
        single = pickle.loads(blob)
        kept = [single.axes[i] for i in keep]
        for a in list(single.axes):
            if a not in kept:
                single.delaxes(a)
        for t in list(single.texts):
            t.remove()
        for leg in list(single.legends):
            leg.remove()
        if legend_axes is not None:
            legend_axes(kept[0])
        for a in kept:
            for attr, value in p['restore'].items():
                if attr == 'ylabels':
                    a.tick_params(axis='y', labelleft=True)
                if attr == 'ylabel':
                    a.set_ylabel(value, fontsize=BASE)
                if attr == 'xlabel':
                    a.set_xlabel(value, fontsize=BASE)
                    a.tick_params(axis='x', labelbottom=True)
        W, H = fig.get_size_inches()
        w, h = o.width * W, o.height * H
        pad = .35
        single.set_size_inches(w + 2 * pad, h + 2 * pad, forward=True)
        for a in kept:
            pos = a.get_position()
            x0 = (pad + (pos.x0 - o.x0) * W) / (w + 2 * pad)
            y0 = (pad + (pos.y0 - o.y0) * H) / (h + 2 * pad)
            a.set_position([x0, y0, pos.width * W / (w + 2 * pad), pos.height * H / (h + 2 * pad)])
        png = sub / f'{stem}_{p["key"]}.png'; pdf = sub / f'{stem}_{p["key"]}.pdf'
        single.savefig(png, dpi=dpi, bbox_inches='tight', pad_inches=.08, facecolor='white')
        single.savefig(pdf, bbox_inches='tight', pad_inches=.08, facecolor='white')
        _set_pdf_meta(pdf)
        plt.close(single)
        lines.append(f'- {p["key"]}: `{png.name}`, `{pdf.name}`')
    (sub / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def dump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
