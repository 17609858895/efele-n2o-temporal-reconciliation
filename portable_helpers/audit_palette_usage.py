"""Small local validation interface for the retained plotting source.

Final publication figures also use figkit.audit for actual artist inspection.
This helper validates declared series colors; it does not certify figure layout.
"""
from matplotlib.colors import to_hex
from scientific_palettes import get_palette

def validate_series_colors(palette_name, series_colors):
    allowed={c.upper() for c in get_palette(palette_name)}
    normalized={k:to_hex(v).upper() for k,v in series_colors.items()}
    if not all(v in allowed for v in normalized.values()):
        raise ValueError('A declared series color is outside the selected palette')
    return normalized

def audit_series_artists(palette_name, artists, series_colors):
    colors=validate_series_colors(palette_name,series_colors)
    return dict(passed=True,scope='declared series colors only',series_colors=colors,
                declared_artist_groups=sorted(artists))
