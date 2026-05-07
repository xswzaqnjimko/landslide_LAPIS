# Elevation visualization for LAPIS

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .colormaps import get_elevation_cmap
from .utils import add_boundary, get_bbox


def save_initial_elevation(dem, active_mask, sensors, config, filepath):
    """
    Save a figure showing the initial terrain elevation (ground topology).
    
    Args:
        dem: 2D array of elevation values
        active_mask: boolean mask of active region
        sensors: array of sensor locations [(r, c), ...]
        config: Config object
        filepath: output path
    """
    filepath = Path(filepath)
    
    elev_cmap = get_elevation_cmap()
    
    # Get elevation range (excluding nodata)
    nodata = -9999  # or get from metadata
    valid_mask = dem > nodata
    vmin = dem[valid_mask].min()
    vmax = dem[valid_mask].max()
    
    # Create figure with colorbar
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    # Plot elevation
    im = ax.imshow(dem, cmap=elev_cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
    
    # Add active region boundary
    add_boundary(ax, active_mask, color='red', lw=2)
    
    # Plot sensor locations
    if sensors is not None and len(sensors) > 0:
        sensor_rows = [s[0] for s in sensors]
        sensor_cols = [s[1] for s in sensors]
        ax.scatter(sensor_cols, sensor_rows, c=config.SENSOR_COLOR, 
                   marker=config.SENSOR_MARKER, s=config.SENSOR_SIZE, 
                   linewidths=1, label='Sensors', zorder=10)
    
    ax.set_title('Initial Terrain Elevation (Pre-Runout DEM)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Colorbar
    fig.colorbar(im, cax=cax, label='Elevation (m)')
    
    # Add legend
    if sensors is not None and len(sensors) > 0:
        ax.legend(loc='upper right')
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_initial_elevation_zoomed(dem, active_mask, sensors, config, filepath):
    """
    Save a zoomed figure showing the initial terrain elevation around active region.
    """
    filepath = Path(filepath)
    
    elev_cmap = get_elevation_cmap()
    
    # Get bounding box for zoom
    r_min, r_max, c_min, c_max = get_bbox(active_mask, config.ZOOM_PAD)
    
    # Crop to zoomed region
    dem_zoom = dem[r_min:r_max, c_min:c_max]
    active_zoom = active_mask[r_min:r_max, c_min:c_max]
    
    # Get elevation range for zoomed region (excluding nodata)
    nodata = -9999
    valid_mask = dem_zoom > nodata
    if valid_mask.any():
        vmin = dem_zoom[valid_mask].min()
        vmax = dem_zoom[valid_mask].max()
    else:
        vmin = dem_zoom.min()
        vmax = dem_zoom.max()
    
    # Create figure with colorbar
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    # Plot elevation
    im = ax.imshow(dem_zoom, cmap=elev_cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
    
    # Add active region boundary
    add_boundary(ax, active_zoom, color='red', lw=2)
    
    # Plot sensor locations (adjusted for zoom)
    if sensors is not None and len(sensors) > 0:
        sensor_rows = [s[0] - r_min for s in sensors if r_min <= s[0] < r_max and c_min <= s[1] < c_max]
        sensor_cols = [s[1] - c_min for s in sensors if r_min <= s[0] < r_max and c_min <= s[1] < c_max]
        if len(sensor_rows) > 0:
            ax.scatter(sensor_cols, sensor_rows, c=config.SENSOR_COLOR, 
                       marker=config.SENSOR_MARKER, s=config.SENSOR_SIZE * 1.5, 
                       linewidths=1, label='Sensors', zorder=10)
    
    ax.set_title('Initial Terrain Elevation (Zoomed)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Colorbar
    fig.colorbar(im, cax=cax, label='Elevation (m)')
    
    # Add legend
    if sensors is not None and len(sensors) > 0:
        ax.legend(loc='upper right')
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")
