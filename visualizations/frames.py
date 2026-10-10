# Frame and video visualization for LAPIS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from .colormaps import get_dod_cmap, get_err_cmap, get_log_cmap
from .utils import add_boundary, get_bbox, plot_with_dark_inactive, plot_log_magnitude


def save_animation_formats(anim, base_dir, filename, fps, dpi, generate_videos=True):
    """
    Save animation in GIF and optionally MP4 formats to separate folders.
    
    Args:
        anim: matplotlib animation object
        base_dir: base results directory
        filename: base filename (without extension)
        fps: frames per second
        dpi: dots per inch
        generate_videos: if True, also generate MP4 videos
    """
    base_dir = Path(base_dir)
    gif_dir = base_dir / "gifs"
    video_dir = base_dir / "videos"
    
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    # Always save GIF (pillow is always available)
    gif_path = gif_dir / f"{filename}.gif"
    try:
        anim.save(str(gif_path), writer='pillow', fps=fps, dpi=dpi//2)
        print(f"    Saved: gifs/{filename}.gif")
    except Exception as e:
        print(f"    Warning: Could not save GIF: {e}")
    
    # Save MP4 if enabled and FFmpeg is available
    if generate_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = video_dir / f"{filename}.mp4"
        try:
            import shutil
            if shutil.which('ffmpeg') is not None:
                writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='LAPIS'), bitrate=1800)
                anim.save(str(mp4_path), writer=writer, dpi=dpi)
                print(f"    Saved: videos/{filename}.mp4")
            else:
                print(f"    Note: FFmpeg not found, skipping MP4 output")
        except Exception as e:
            print(f"    Note: Could not save MP4: {e}")


def save_single_panel_video(data_sequence, active_mask, cmap, vmin, vmax, 
                            title_template, colorbar_label, config, base_dir, filename):
    """
    Save a single-panel video with colorbar.
    
    Args:
        data_sequence: (T, H, W) array of data to visualize
        active_mask: (H, W) boolean mask
        cmap: colormap to use
        vmin, vmax: color scale limits
        title_template: string template with {t} for frame number
        colorbar_label: label for colorbar
        config: Config object
        base_dir: base results directory
        filename: base filename (without extension)
    """
    T = data_sequence.shape[0]
    
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, cax=cax, label=colorbar_label)
    
    def animate(t):
        if t == 0:
            t = 1
        ax.clear()
        
        data_t = data_sequence[t]
        plot_with_dark_inactive(ax, data_t, active_mask, cmap, vmin, vmax, config.INACTIVE_COLOR)
        ax.set_title(title_template.format(t=t))
        add_boundary(ax, active_mask)
        ax.set_xticks([]); ax.set_yticks([])
        
        return [ax]
    
    anim = animation.FuncAnimation(fig, animate, frames=range(1, T), 
                                   interval=1000//config.VIDEO_FPS, blit=False)
    
    save_animation_formats(anim, base_dir, filename, config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)


def save_single_panel_video_logscale(data_sequence, active_mask, cmap, vmin_log, vmax_log,
                                      title_template, colorbar_label, config, base_dir, filename):
    """
    Save a single-panel video with log-scale magnitude and colorbar.
    """
    T = data_sequence.shape[0]
    
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin_log, vmax_log))
    fig.colorbar(sm, cax=cax, label=colorbar_label)
    
    def animate(t):
        if t == 0:
            t = 1
        ax.clear()
        
        data_t = data_sequence[t]
        plot_log_magnitude(ax, data_t, active_mask, cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        ax.set_title(title_template.format(t=t))
        add_boundary(ax, active_mask)
        ax.set_xticks([]); ax.set_yticks([])
        
        return [ax]
    
    anim = animation.FuncAnimation(fig, animate, frames=range(1, T), 
                                   interval=1000//config.VIDEO_FPS, blit=False)
    
    save_animation_formats(anim, base_dir, filename, config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)


def save_error_panel_video(gt, pred, active_mask, cmap, config, base_dir, filename):
    """Save error panel video with RMSE in title."""
    T = min(gt.shape[0], pred.shape[0])
    err_seq = pred - gt
    emax = max(np.abs(err_seq).max(), 0.1)
    
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-emax, emax))
    fig.colorbar(sm, cax=cax, label='Error (m)')
    
    def animate(t):
        if t == 0:
            t = 1
        ax.clear()
        
        err_t = err_seq[t]
        rmse = np.sqrt(np.mean((err_t[active_mask])**2)) if active_mask.sum() > 0 else 0
        
        plot_with_dark_inactive(ax, err_t, active_mask, cmap, -emax, emax, config.INACTIVE_COLOR)
        ax.set_title(f'Error (t={t}, RMSE={rmse:.3f}m)')
        add_boundary(ax, active_mask)
        ax.set_xticks([]); ax.set_yticks([])
        
        return [ax]
    
    anim = animation.FuncAnimation(fig, animate, frames=range(1, T), 
                                   interval=1000//config.VIDEO_FPS, blit=False)
    
    save_animation_formats(anim, base_dir, filename, config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)


def save_error_panel_video_logscale(gt, pred, active_mask, cmap, vmin_log, err_vmax_log, config, base_dir, filename):
    """Save error panel video with log-scale absolute error and RMSE in title."""
    T = min(gt.shape[0], pred.shape[0])
    err_seq = pred - gt
    abs_err_seq = np.abs(err_seq)
    
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin_log, err_vmax_log))
    fig.colorbar(sm, cax=cax, label='log(|Error| + ε)')
    
    def animate(t):
        if t == 0:
            t = 1
        ax.clear()
        
        abs_err_t = abs_err_seq[t]
        err_t = err_seq[t]
        rmse = np.sqrt(np.mean((err_t[active_mask])**2)) if active_mask.sum() > 0 else 0
        
        plot_log_magnitude(ax, abs_err_t, active_mask, cmap, vmin_log, err_vmax_log, config.INACTIVE_COLOR)
        ax.set_title(f'log|Error| (t={t}, RMSE={rmse:.3f}m)')
        add_boundary(ax, active_mask)
        ax.set_xticks([]); ax.set_yticks([])
        
        return [ax]
    
    anim = animation.FuncAnimation(fig, animate, frames=range(1, T), 
                                   interval=1000//config.VIDEO_FPS, blit=False)
    
    save_animation_formats(anim, base_dir, filename, config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)


def save_video(gt, pred, active_mask, sensors, config, base_dir, base_name="lapis_ensemble"):
    """
    Save comparison videos - one combined GIF and individual panel videos.
    """
    T = min(gt.shape[0], pred.shape[0])
    dod_cmap = get_dod_cmap()
    err_cmap = get_err_cmap()
    vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1.0)
    
    # Compute error sequence
    err_seq = pred[:T] - gt[:T]
    emax = max(np.abs(err_seq).max(), 0.1)
    
    print(f"  Saving {base_name} animations...")
    
    # Save combined GIF (3 panels)
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, left=0.02, right=0.95)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    
    sm = plt.cm.ScalarMappable(cmap=dod_cmap, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, cax=cax, label='DoD (m)')
    
    def animate(t):
        if t == 0:
            t = 1
        for ax in axes:
            ax.clear()
        
        gt_t = gt[t]
        pred_t = pred[t]
        err_t = err_seq[t]
        
        plot_with_dark_inactive(axes[0], gt_t, active_mask, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
        axes[0].set_title(f'Ground Truth (t={t})')
        add_boundary(axes[0], active_mask)
        
        plot_with_dark_inactive(axes[1], pred_t, active_mask, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
        axes[1].set_title(f'Prediction (t={t})')
        add_boundary(axes[1], active_mask)
        
        rmse = np.sqrt(np.mean((err_t[active_mask])**2)) if active_mask.sum() > 0 else 0
        plot_with_dark_inactive(axes[2], err_t, active_mask, err_cmap, -emax, emax, config.INACTIVE_COLOR)
        axes[2].set_title(f'Error (RMSE={rmse:.3f}m)')
        add_boundary(axes[2], active_mask)
        
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        
        return axes
    
    anim = animation.FuncAnimation(fig, animate, frames=range(1, T), 
                                   interval=1000//config.VIDEO_FPS, blit=False)
    
    save_animation_formats(anim, base_dir, f"{base_name}_combined", config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)
    
    # Save individual panel videos
    save_single_panel_video(gt[:T], active_mask, dod_cmap, -vmax, vmax,
                           'Ground Truth (t={t})', 'DoD (m)', config,
                           base_dir, f"{base_name}_gt")
    
    save_single_panel_video(pred[:T], active_mask, dod_cmap, -vmax, vmax,
                           'Prediction (t={t})', 'DoD (m)', config,
                           base_dir, f"{base_name}_pred")
    
    # For error, we need a custom function since it has dynamic RMSE in title
    save_error_panel_video(gt[:T], pred[:T], active_mask, err_cmap, config,
                          base_dir, f"{base_name}_error")


def save_video_logscale(gt, pred, active_mask, sensors, config, base_dir, base_name="lapis_ensemble_logscale"):
    """
    Save comparison videos with log-scale - one combined GIF and individual panel videos.
    Error shown as absolute error in log scale.
    """
    T = min(gt.shape[0], pred.shape[0])
    log_cmap = get_log_cmap()
    
    # Compute log scale bounds - fixed vmin at -2 for colormap alignment
    epsilon = 1e-3
    all_data = np.concatenate([gt.ravel(), pred.ravel()])
    vmin_log = -2.0  # Fixed for colormap alignment
    vmax_log = np.log(np.abs(all_data).max() + epsilon)
    
    # For error, compute max absolute error for vmax
    all_errors = pred[:T] - gt[:T]
    err_vmax_log = np.log(np.abs(all_errors).max() + epsilon)
    
    print(f"  Saving {base_name} animations...")
    
    # Save combined GIF (3 panels)
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, left=0.02, right=0.95)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    
    sm = plt.cm.ScalarMappable(cmap=log_cmap, norm=plt.Normalize(vmin_log, vmax_log))
    fig.colorbar(sm, cax=cax, label='log(|value| + ε)')
    
    def animate(t):
        if t == 0:
            t = 1
        for ax in axes:
            ax.clear()
        
        gt_t = gt[t]
        pred_t = pred[t]
        err_t = pred_t - gt_t
        abs_err_t = np.abs(err_t)
        
        plot_log_magnitude(axes[0], gt_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[0].set_title(f'GT log|DoD| (t={t})')
        add_boundary(axes[0], active_mask)
        
        plot_log_magnitude(axes[1], pred_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[1].set_title(f'Pred log|DoD| (t={t})')
        add_boundary(axes[1], active_mask)
        
        rmse = np.sqrt(np.mean((err_t[active_mask])**2)) if active_mask.sum() > 0 else 0
        plot_log_magnitude(axes[2], abs_err_t, active_mask, log_cmap, vmin_log, err_vmax_log, config.INACTIVE_COLOR)
        axes[2].set_title(f'log|Error| (RMSE={rmse:.3f}m)')
        add_boundary(axes[2], active_mask)
        
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        
        return axes
    
    anim = animation.FuncAnimation(fig, animate, frames=range(1, T), 
                                   interval=1000//config.VIDEO_FPS, blit=False)
    
    save_animation_formats(anim, base_dir, f"{base_name}_combined", config.VIDEO_FPS, config.VIDEO_DPI, config.GENERATE_VIDEOS)
    plt.close(fig)
    
    # Save individual panel videos
    save_single_panel_video_logscale(gt[:T], active_mask, log_cmap, vmin_log, vmax_log,
                                     'GT log|DoD| (t={t})', 'log(|DoD| + ε)', config,
                                     base_dir, f"{base_name}_gt")
    
    save_single_panel_video_logscale(pred[:T], active_mask, log_cmap, vmin_log, vmax_log,
                                     'Pred log|DoD| (t={t})', 'log(|DoD| + ε)', config,
                                     base_dir, f"{base_name}_pred")
    
    save_error_panel_video_logscale(gt[:T], pred[:T], active_mask, log_cmap, vmin_log, err_vmax_log, config,
                                    base_dir, f"{base_name}_error")


def save_zoomed_frame_figure(gt, pred, active_mask, frame_idx, title_prefix, config, filepath, 
                             include_colorbar=True):
    """
    Save a figure showing only the zoomed 2nd row (GT Zoomed, Prediction Zoomed, Error Zoomed).
    """
    filepath = Path(filepath)
    
    dod_cmap = get_dod_cmap()
    err_cmap = get_err_cmap()
    
    vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1.0)
    
    gt_t = gt[frame_idx]
    pred_t = pred[frame_idx]
    err_t = pred_t - gt_t
    emax = max(np.abs(err_t).max(), 0.1)
    
    # Get bounding box for zoom
    r_min, r_max, c_min, c_max = get_bbox(active_mask, config.ZOOM_PAD)
    
    # Crop to zoomed region
    gt_zoom = gt_t[r_min:r_max, c_min:c_max]
    pred_zoom = pred_t[r_min:r_max, c_min:c_max]
    err_zoom = err_t[r_min:r_max, c_min:c_max]
    active_zoom = active_mask[r_min:r_max, c_min:c_max]
    
    # Create figure
    if include_colorbar:
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # Plot zoomed GT
    plot_with_dark_inactive(axes[0], gt_zoom, active_zoom, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
    axes[0].set_title(f'GT (Zoomed)')
    add_boundary(axes[0], active_zoom)
    
    # Plot zoomed Prediction
    plot_with_dark_inactive(axes[1], pred_zoom, active_zoom, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
    axes[1].set_title(f'Prediction (Zoomed)')
    add_boundary(axes[1], active_zoom)
    
    # Plot zoomed Error
    rmse = np.sqrt(np.mean((err_zoom[active_zoom])**2)) if active_zoom.sum() > 0 else 0
    plot_with_dark_inactive(axes[2], err_zoom, active_zoom, err_cmap, -emax, emax, config.INACTIVE_COLOR)
    axes[2].set_title(f'Error (Zoomed)')
    add_boundary(axes[2], active_zoom)
    
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    
    if include_colorbar:
        sm = plt.cm.ScalarMappable(cmap=dod_cmap, norm=plt.Normalize(-vmax, vmax))
        fig.colorbar(sm, cax=cax, label='DoD (m)')
    
    fig.suptitle(f'{title_prefix} | Frame {frame_idx} | RMSE={rmse:.4f}m', fontsize=12, fontweight='bold')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_zoomed_frame_figure_logscale(gt, pred, active_mask, frame_idx, title_prefix, config, filepath,
                                      include_colorbar=True):
    """
    Save a zoomed frame figure with log-scale magnitude visualization.
    """
    filepath = Path(filepath)
    
    log_cmap = get_log_cmap()
    
    # Fixed vmin at -2 for colormap alignment
    epsilon = 1e-3
    all_data = np.concatenate([gt.ravel(), pred.ravel()])
    vmin_log = -2.0
    vmax_log = np.log(np.abs(all_data).max() + epsilon)
    
    # For error, compute max absolute error for vmax
    all_errors = pred - gt
    err_vmax_log = np.log(np.abs(all_errors).max() + epsilon)
    
    gt_t = gt[frame_idx]
    pred_t = pred[frame_idx]
    err_t = pred_t - gt_t
    abs_err_t = np.abs(err_t)
    
    # Get bounding box for zoom
    r_min, r_max, c_min, c_max = get_bbox(active_mask, config.ZOOM_PAD)
    
    # Crop to zoomed region
    gt_zoom = gt_t[r_min:r_max, c_min:c_max]
    pred_zoom = pred_t[r_min:r_max, c_min:c_max]
    abs_err_zoom = abs_err_t[r_min:r_max, c_min:c_max]
    err_zoom = err_t[r_min:r_max, c_min:c_max]
    active_zoom = active_mask[r_min:r_max, c_min:c_max]
    
    if include_colorbar:
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # Plot zoomed GT (log scale)
    plot_log_magnitude(axes[0], gt_zoom, active_zoom, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
    axes[0].set_title(f'GT log|DoD| (Zoomed)')
    add_boundary(axes[0], active_zoom)
    
    # Plot zoomed Prediction (log scale)
    plot_log_magnitude(axes[1], pred_zoom, active_zoom, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
    axes[1].set_title(f'Pred log|DoD| (Zoomed)')
    add_boundary(axes[1], active_zoom)
    
    # Plot zoomed absolute Error (log scale)
    rmse = np.sqrt(np.mean((err_zoom[active_zoom])**2)) if active_zoom.sum() > 0 else 0
    plot_log_magnitude(axes[2], abs_err_zoom, active_zoom, log_cmap, vmin_log, err_vmax_log, config.INACTIVE_COLOR)
    axes[2].set_title(f'log|Error| (Zoomed)')
    add_boundary(axes[2], active_zoom)
    
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    
    if include_colorbar:
        sm = plt.cm.ScalarMappable(cmap=log_cmap, norm=plt.Normalize(vmin_log, vmax_log))
        fig.colorbar(sm, cax=cax, label='log(|value| + ε)')
    
    fig.suptitle(f'{title_prefix} (Log Scale) | Frame {frame_idx} | RMSE={rmse:.4f}m', fontsize=12, fontweight='bold')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")
