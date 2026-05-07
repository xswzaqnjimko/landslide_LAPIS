# Utility functions for LAPIS visualization

import numpy as np


def add_boundary(ax, mask, color='lime', lw=1.5):
    """Add contour boundary around masked region."""
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def get_bbox(mask, pad=10):
    """Get bounding box for masked region with padding."""
    H, W = mask.shape
    if not np.any(mask):
        return 0, H, 0, W
    r, c = np.where(mask)
    return max(0, r.min()-pad), min(H, r.max()+pad+1), max(0, c.min()-pad), min(W, c.max()+pad+1)


def plot_with_dark_inactive(ax, data, active_mask, cmap, vmin, vmax, inactive_color=0.15):
    """Plot data with dark gray for inactive regions."""
    norm_data = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
    rgba = cmap(norm_data)
    rgb = rgba[:, :, :3].copy()
    rgb[~active_mask] = inactive_color
    return ax.imshow(rgb, origin='upper', aspect='equal')


def plot_log_magnitude(ax, data, active_mask, cmap, vmin_log, vmax_log, inactive_color=0.15):
    """
    Plot log (log_e) of magnitude with dark inactive regions.
    Uses fixed vmin_log=-2 for colormap alignment (black at -2 and below).
    """
    # Compute log(|data| + epsilon) to handle zeros
    epsilon = 1e-3
    log_mag = np.log(np.abs(data) + epsilon)
    
    # Clamp to [-2, vmax_log] range - anything below -2 will be black
    log_mag_clamped = np.clip(log_mag, -2.0, vmax_log)
    
    # Normalize using fixed -2 as minimum
    norm_data = np.clip((log_mag_clamped - (-2.0)) / (vmax_log - (-2.0) + 1e-10), 0, 1)
    rgba = cmap(norm_data)
    rgb = rgba[:, :, :3].copy()
    rgb[~active_mask] = inactive_color
    return ax.imshow(rgb, origin='upper', aspect='equal')
