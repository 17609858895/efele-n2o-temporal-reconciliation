"""Repository-local numerical colors used by the EFELE figures."""
PALETTES = {
    'nature_new': ('#DF9E9B','#99BADF','#D8E7CA','#99CDCE','#999ACD','#FFD0E9','#F6D58B','#C7A3CC'),
    'au_nimno_soft_8': ('#79C1E4','#E68282','#B2D362','#BAE1F3','#D4EAF8','#EECDD5','#F8E6E4','#D1E4A6'),
}
STYLE_COLORS = {'ink':'#30343B','muted':'#6F7782','light':'#F3F5F7'}

def get_palette(name, n=None, reverse=False):
    colors=list(PALETTES[name])
    if reverse: colors.reverse()
    if n is None: return colors
    if n > len(colors): raise ValueError('Palette has insufficient distinct colors')
    return colors[:n]
