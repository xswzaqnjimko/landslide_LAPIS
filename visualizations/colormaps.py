# Colormaps for LAPIS visualization

from matplotlib.colors import LinearSegmentedColormap


def get_dod_cmap():
    """Colormap for DoD visualization."""
    colors = [(0.0,0.2,0.6),(0.3,0.5,0.9),(0.7,0.85,1.0),(0.97,0.97,0.97),
              (1.0,0.85,0.7),(0.95,0.5,0.3),(0.7,0.1,0.1)]
    return LinearSegmentedColormap.from_list('dod', list(zip([0,0.2,0.4,0.5,0.6,0.8,1], colors)))


def get_err_cmap():
    """Colormap for error visualization."""
    colors = [(0.4,0,0.6),(0.7,0.7,0.9),(0.97,0.97,0.97),(0.7,0.9,0.7),(0,0.5,0)]
    return LinearSegmentedColormap.from_list('err', colors)


def get_log_cmap():
    """
    Colormap for log-scale magnitude visualization.
    Designed for range [-2, vmax] where:
    - Black at -2 and below
    - Rapidly converges to black below -1 (almost black at -1)
    - Colorful range from -1 to vmax
    """
    # Define colors at specific positions
    # Position 0.0 = -2 (black)
    # Position ~0.25 = -1 (almost black, slight hint of color)
    # Positions 0.25 to 1.0 = colorful range
    colors_with_positions = [
        (0.00, (0.0, 0.0, 0.0)),       # -2: pure black
        (0.15, (0.02, 0.02, 0.02)),    # slightly above -2: nearly black
        (0.25, (0.08, 0.05, 0.12)),    # -1: almost black with slight purple hint
        (0.35, (0.2, 0.1, 0.35)),      # dark purple
        (0.45, (0.4, 0.2, 0.6)),       # purple
        (0.55, (0.6, 0.4, 0.8)),       # light purple
        (0.70, (0.8, 0.6, 0.4)),       # orange
        (0.85, (1.0, 0.8, 0.3)),       # yellow-orange
        (1.00, (1.0, 1.0, 0.4)),       # bright yellow
    ]
    positions = [p[0] for p in colors_with_positions]
    colors = [p[1] for p in colors_with_positions]
    return LinearSegmentedColormap.from_list('log_mag', list(zip(positions, colors)))


def get_elevation_cmap():
    """Colormap for terrain/elevation visualization."""
    # Terrain-like colormap: blue (low) -> green -> yellow -> brown -> white (high)
    colors = [
        (0.0, (0.2, 0.3, 0.5)),    # Deep blue-gray (low)
        (0.15, (0.3, 0.5, 0.4)),   # Teal
        (0.3, (0.4, 0.6, 0.3)),    # Green
        (0.45, (0.6, 0.7, 0.3)),   # Yellow-green
        (0.6, (0.8, 0.7, 0.4)),    # Tan/yellow
        (0.75, (0.7, 0.5, 0.3)),   # Brown
        (0.9, (0.6, 0.4, 0.3)),    # Dark brown
        (1.0, (0.95, 0.95, 0.95)), # Near white (high peaks)
    ]
    positions = [p[0] for p in colors]
    color_vals = [p[1] for p in colors]
    return LinearSegmentedColormap.from_list('elevation', list(zip(positions, color_vals)))
