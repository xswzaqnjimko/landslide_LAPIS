# Ensemble results visualization for LAPIS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from .colormaps import get_dod_cmap, get_err_cmap, get_log_cmap
from .utils import add_boundary, get_bbox, plot_with_dark_inactive, plot_log_magnitude
from .frames import (
    save_animation_formats,
    save_single_panel_video,
    save_single_panel_video_logscale,
    save_error_panel_video,
    save_error_panel_video_logscale
)


def save_ensemble_results(gt, pred, active_mask, test_theta,
                          metrics, config, filepath):
    """Save ensemble results figure with colorbars."""
    T = min(gt.shape[0], pred.shape[0])
    
    frames = [1, T//4, T//2, 3*T//4, T-1]
    
    dod_cmap = get_dod_cmap()
    err_cmap = get_err_cmap()
    
    vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1.0)
    
    crop_pad = getattr(config, 'ZOOM_PAD', 10)
    r0, r1, c0, c1 = get_bbox(active_mask, pad=crop_pad)
    
    fig = plt.figure(figsize=(5*len(frames) + 1.0, 14))
    gs = fig.add_gridspec(3, len(frames) + 1, width_ratios=[1]*len(frames) + [0.08], 
                          wspace=0.05, hspace=0.12)
    
    axes = [[fig.add_subplot(gs[row, col]) for col in range(len(frames))] for row in range(3)]
    cax_dod = fig.add_subplot(gs[0:2, len(frames)])  # Colorbar for DoD (rows 0-1)
    cax_err = fig.add_subplot(gs[2, len(frames)])     # Colorbar for Error (row 2)
    
    max_err = 0.1
    
    for i, t in enumerate(frames):
        gt_t = gt[t]
        pred_t = pred[t]
        err_t = pred_t - gt_t
        emax = max(np.abs(err_t).max(), 0.1)
        max_err = max(max_err, emax)
        
        plot_with_dark_inactive(axes[0][i], gt_t, active_mask, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
        axes[0][i].set_title(f'GT (t={t})', fontsize=20)
        add_boundary(axes[0][i], active_mask)
        axes[0][i].set_xlim(c0, c1); axes[0][i].set_ylim(r1, r0)
        axes[0][i].set_xticks([]); axes[0][i].set_yticks([])
        
        plot_with_dark_inactive(axes[1][i], pred_t, active_mask, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
        axes[1][i].set_title(f'Pred (t={t})', fontsize=20)
        add_boundary(axes[1][i], active_mask)
        axes[1][i].set_xlim(c0, c1); axes[1][i].set_ylim(r1, r0)
        axes[1][i].set_xticks([]); axes[1][i].set_yticks([])
        
        rmse = np.sqrt(np.mean((err_t[active_mask])**2)) if active_mask.sum() > 0 else 0
        plot_with_dark_inactive(axes[2][i], err_t, active_mask, err_cmap, -emax, emax, config.INACTIVE_COLOR)
        axes[2][i].set_title('Error', fontsize=20)
        axes[2][i].text(0.97, 0.97, f'RMSE={rmse:.3f}m', transform=axes[2][i].transAxes,
                        fontsize=11, color='white', ha='right', va='top')
        add_boundary(axes[2][i], active_mask)
        axes[2][i].set_xlim(c0, c1); axes[2][i].set_ylim(r1, r0)
        axes[2][i].set_xticks([]); axes[2][i].set_yticks([])
    
    axes[0][0].set_ylabel('Ground Truth', fontsize=16, fontweight='bold')
    axes[1][0].set_ylabel('LAPIS Ensemble', fontsize=16, fontweight='bold')
    axes[2][0].set_ylabel('Error', fontsize=16, fontweight='bold')
    
    sm_dod = plt.cm.ScalarMappable(cmap=dod_cmap, norm=plt.Normalize(-vmax, vmax))
    cb_dod = fig.colorbar(sm_dod, cax=cax_dod, label='DoD (m)')
    cb_dod.ax.tick_params(labelsize=13)
    cb_dod.set_label('DoD (m)', fontsize=14)
    
    sm_err = plt.cm.ScalarMappable(cmap=err_cmap, norm=plt.Normalize(-max_err, max_err))
    cb_err = fig.colorbar(sm_err, cax=cax_err, label='Error (m)')
    cb_err.ax.tick_params(labelsize=13)
    cb_err.set_label('Error (m)', fontsize=14)
    
    theta_str = f"Test θ: S_c={test_theta['S_c']:.3f}, q_c={test_theta['q_c']:.3f}, k={test_theta['k']:.4f}"
    metrics_str = f"RMSE_active={metrics['rmse_active']:.4f}m, Corr={metrics['corr_active']:.4f}"
    
    mode_label = "Forward" if getattr(config, "INFERENCE_MODE", "backward").lower() == "forward" else "Backward"
    plt.suptitle(f'LAPIS Ensemble: {mode_label} Reconstruction on Unseen θ  [{getattr(config, "CLIMATE_CLASS", "")}/{getattr(config, "THETA_GROUP", "")}]\n{theta_str}\n{metrics_str}', 
                 fontsize=18, fontweight='bold')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def save_ensemble_results_logscale(gt, pred, active_mask, sensors, train_thetas, test_theta, 
                                   metrics, config, filepath, row2_label='LAPIS Inference',
                                   n_columns=5):
    T = min(gt.shape[0], pred.shape[0])
    
    if n_columns == 3:
        frames = [1, T//2, T-1]
    else:
        frames = [1, T//4, T//2, 3*T//4, T-1]
    
    log_cmap = get_log_cmap()
    
    epsilon = 1e-3
    all_data = np.concatenate([gt.ravel(), pred.ravel()])
    vmin_log = -2.0  # Fixed for colormap alignment
    vmax_log = np.log(np.abs(all_data).max() + epsilon)
    
    crop_pad = getattr(config, 'ZOOM_PAD', 10)
    r0, r1, c0, c1 = get_bbox(active_mask, pad=crop_pad)
    
    panel_w = 3.5 if n_columns == 3 else 5
    ws = 0.01 if n_columns == 3 else 0.05
    fig = plt.figure(figsize=(panel_w*len(frames) + 1.0, 14))
    gs = fig.add_gridspec(3, len(frames) + 1, width_ratios=[1]*len(frames) + [0.06], 
                          wspace=ws, hspace=0.12)
    
    axes = [[fig.add_subplot(gs[row, col]) for col in range(len(frames))] for row in range(3)]
    cax = fig.add_subplot(gs[:, len(frames)])  # Single colorbar spanning all rows
    
    for i, t in enumerate(frames):
        gt_t = gt[t]
        pred_t = pred[t]
        err_t = pred_t - gt_t
        abs_err_t = np.abs(err_t)
        
        plot_log_magnitude(axes[0][i], gt_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[0][i].set_title(f'log|DoD| (t={t})', fontsize=20)
        add_boundary(axes[0][i], active_mask)
        axes[0][i].set_xlim(c0, c1); axes[0][i].set_ylim(r1, r0)
        axes[0][i].set_xticks([]); axes[0][i].set_yticks([])
        
        plot_log_magnitude(axes[1][i], pred_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[1][i].set_title(f'log|DoD| (t={t})', fontsize=20)
        add_boundary(axes[1][i], active_mask)
        axes[1][i].set_xlim(c0, c1); axes[1][i].set_ylim(r1, r0)
        axes[1][i].set_xticks([]); axes[1][i].set_yticks([])
        
        frame_gt_range = float(gt_t.max() - gt_t.min()) if active_mask.sum() > 0 else 1.0
        rmse = np.sqrt(np.mean((err_t[active_mask])**2)) if active_mask.sum() > 0 else 0
        nrmse = rmse / frame_gt_range if frame_gt_range > 0 else 0
        plot_log_magnitude(axes[2][i], abs_err_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[2][i].set_title(f'NRMSE={nrmse:.4f}', fontsize=20)
        add_boundary(axes[2][i], active_mask)
        axes[2][i].set_xlim(c0, c1); axes[2][i].set_ylim(r1, r0)
        axes[2][i].set_xticks([]); axes[2][i].set_yticks([])
    
    axes[0][0].set_ylabel('Ground Truth', fontsize=20, fontweight='bold')
    axes[1][0].set_ylabel(row2_label, fontsize=20, fontweight='bold')
    axes[2][0].set_ylabel('log|Error|', fontsize=20, fontweight='bold')
    
    sm = plt.cm.ScalarMappable(cmap=log_cmap, norm=plt.Normalize(vmin_log, vmax_log))
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=24)
    
    theta_str = f"Test θ: S_c={test_theta['S_c']:.3f}, q_c={test_theta['q_c']:.3f}, k={test_theta['k']:.4f}"
    overall_nrmse = metrics.get('overall_nrmse_active')
    if overall_nrmse is not None:
        metrics_str = f"Overall NRMSE={overall_nrmse:.4f}, Corr={metrics['corr_active']:.4f}"
        nse_val = metrics.get('overall_nse')
        kge_prime_abs_val = metrics.get('overall_kge_prime_abs')
        if nse_val is not None:
            metrics_str += f", NSE={nse_val:.4f}"
        if kge_prime_abs_val is not None:
            metrics_str += f", KGE'|DoD|={kge_prime_abs_val:.4f}"
    else:
        metrics_str = f"RMSE_active={metrics['rmse_active']:.4f}m, Corr={metrics['corr_active']:.4f}"
    
    mode_label = "Forward" if getattr(config, "INFERENCE_MODE", "backward").lower() == "forward" else "Backward"
    plt.suptitle(f'LAPIS Ensemble (Log Scale): {mode_label} Reconstruction on Unseen θ  [{getattr(config, "CLIMATE_CLASS", "")}/{getattr(config, "THETA_GROUP", "")}]\n{theta_str}\n{metrics_str}', 
                 fontsize=18, fontweight='bold')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def save_shred_results_logscale(gt, pred, active_mask, sensors, train_thetas, test_theta, 
                                metrics, config, filepath):
    """Save SHRED reconstruction results figure with log-scale magnitude (upper bound reference)."""
    T = min(gt.shape[0], pred.shape[0])
    
    frames = [1, T//4, T//2, 3*T//4, T-1]
    
    log_cmap = get_log_cmap()
    
    epsilon = 1e-3
    all_data = np.concatenate([gt.ravel(), pred.ravel()])
    vmin_log = -2.0
    vmax_log = np.log(np.abs(all_data).max() + epsilon)
    
    crop_pad = getattr(config, 'ZOOM_PAD', 10)
    r0, r1, c0, c1 = get_bbox(active_mask, pad=crop_pad)
    
    fig = plt.figure(figsize=(5*len(frames) + 1.0, 14))
    gs = fig.add_gridspec(3, len(frames) + 1, width_ratios=[1]*len(frames) + [0.06], 
                          wspace=0.05, hspace=0.12)
    
    axes = [[fig.add_subplot(gs[row, col]) for col in range(len(frames))] for row in range(3)]
    cax = fig.add_subplot(gs[:, len(frames)])
    
    for i, t in enumerate(frames):
        gt_t = gt[t]
        pred_t = pred[t]
        err_t = pred_t - gt_t
        abs_err_t = np.abs(err_t)
        
        plot_log_magnitude(axes[0][i], gt_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[0][i].set_title(f'GT log|DoD| (t={t})', fontsize=14)
        add_boundary(axes[0][i], active_mask)
        axes[0][i].set_xlim(c0, c1); axes[0][i].set_ylim(r1, r0)
        axes[0][i].set_xticks([]); axes[0][i].set_yticks([])
        
        plot_log_magnitude(axes[1][i], pred_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[1][i].set_title(f'Pred log|DoD| (t={t})', fontsize=14)
        add_boundary(axes[1][i], active_mask)
        axes[1][i].set_xlim(c0, c1); axes[1][i].set_ylim(r1, r0)
        axes[1][i].set_xticks([]); axes[1][i].set_yticks([])
        
        frame_gt_range = float(gt_t.max() - gt_t.min()) if active_mask.sum() > 0 else 1.0
        rmse = np.sqrt(np.mean((err_t[active_mask])**2)) if active_mask.sum() > 0 else 0
        nrmse = rmse / frame_gt_range if frame_gt_range > 0 else 0
        plot_log_magnitude(axes[2][i], abs_err_t, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
        axes[2][i].set_title(f'NRMSE={nrmse:.4f}', fontsize=14)
        add_boundary(axes[2][i], active_mask)
        axes[2][i].set_xlim(c0, c1); axes[2][i].set_ylim(r1, r0)
        axes[2][i].set_xticks([]); axes[2][i].set_yticks([])
    
    axes[0][0].set_ylabel('Ground Truth', fontsize=16, fontweight='bold')
    axes[1][0].set_ylabel('SHRED Reconstruction', fontsize=16, fontweight='bold')
    axes[2][0].set_ylabel('|Error|', fontsize=16, fontweight='bold')
    
    # Single colorbar with larger tick labels
    sm = plt.cm.ScalarMappable(cmap=log_cmap, norm=plt.Normalize(vmin_log, vmax_log))
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=14)
    
    theta_str = f"Test θ: S_c={test_theta['S_c']:.3f}, q_c={test_theta['q_c']:.3f}, k={test_theta['k']:.4f}"
    overall_nrmse = metrics.get('overall_nrmse_active')
    if overall_nrmse is not None:
        metrics_str = f"Overall NRMSE={overall_nrmse:.4f}, Corr={metrics['corr_active']:.4f}"
        nse_val = metrics.get('overall_nse')
        kge_prime_abs_val = metrics.get('overall_kge_prime_abs')
        if nse_val is not None:
            metrics_str += f", NSE={nse_val:.4f}"
        if kge_prime_abs_val is not None:
            metrics_str += f", KGE'|DoD|={kge_prime_abs_val:.4f}"
    else:
        metrics_str = f"RMSE_active={metrics['rmse_active']:.4f}m, Corr={metrics['corr_active']:.4f}"
    
    mode_label = "Forward" if getattr(config, "INFERENCE_MODE", "backward").lower() == "forward" else "Backward"
    plt.suptitle(f'SHRED (Log Scale): {mode_label} Reconstruction on θ_test (Upper Bound)  [{getattr(config, "CLIMATE_CLASS", "")}/{getattr(config, "THETA_GROUP", "")}]\n{theta_str}\n{metrics_str}', 
                 fontsize=18, fontweight='bold')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def save_combined_results_plot(
    gt, pred, active_mask, sensors, train_thetas, test_theta,
    metrics, config, filepath, title_prefix="LAPIS"
):
    """
    Save combined results figure with full domain, zoomed views, scatter plot, and metrics.
    """
    H, W = gt.shape
    error = pred - gt
    r0, r1, c0, c1 = get_bbox(active_mask, config.ZOOM_PAD)
    
    dod_cmap = get_dod_cmap()
    err_cmap = get_err_cmap()
    
    vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1.0)
    emax = max(np.abs(error).max(), 0.1)
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.08],
                          height_ratios=[1, 1, 0.8], hspace=0.25, wspace=0.15)
    
    # Row 1: Full domain
    ax = fig.add_subplot(gs[0, 0])
    plot_with_dark_inactive(ax, gt, active_mask, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
    ax.contour(active_mask.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.scatter(sensors[:, 1], sensors[:, 0], c=config.SENSOR_COLOR,
               s=config.SENSOR_SIZE, marker=config.SENSOR_MARKER, zorder=10)
    ax.add_patch(patches.Rectangle((c0, r0), c1-c0, r1-r0, lw=2, ec='cyan', fc='none', ls='--'))
    ax.set_title(f'GT (θ_test)', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[0, 1])
    plot_with_dark_inactive(ax, pred, active_mask, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
    ax.contour(active_mask.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.add_patch(patches.Rectangle((c0, r0), c1-c0, r1-r0, lw=2, ec='cyan', fc='none', ls='--'))
    ax.set_title('Ensemble Prediction', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[0, 2])
    plot_with_dark_inactive(ax, error, active_mask, err_cmap, -emax, emax, config.INACTIVE_COLOR)
    ax.contour(active_mask.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title(f'Error | Max: {emax:.2f}m', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    
    cax = fig.add_subplot(gs[0, 3])
    sm = plt.cm.ScalarMappable(cmap=dod_cmap, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, cax=cax, label='DoD (m)')
    
    # Row 2: Zoomed
    gt_z = gt[r0:r1, c0:c1]
    pred_z = pred[r0:r1, c0:c1]
    err_z = error[r0:r1, c0:c1]
    mask_z = active_mask[r0:r1, c0:c1]
    
    ax = fig.add_subplot(gs[1, 0])
    plot_with_dark_inactive(ax, gt_z, mask_z, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
    ax.contour(mask_z.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title('GT (Zoomed)', fontweight='bold', color='darkblue')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[1, 1])
    plot_with_dark_inactive(ax, pred_z, mask_z, dod_cmap, -vmax, vmax, config.INACTIVE_COLOR)
    ax.contour(mask_z.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title('Prediction (Zoomed)', fontweight='bold', color='darkblue')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[1, 2])
    plot_with_dark_inactive(ax, err_z, mask_z, err_cmap, -emax, emax, config.INACTIVE_COLOR)
    ax.contour(mask_z.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title('Error (Zoomed)', fontweight='bold', color='darkblue')
    ax.set_xticks([]); ax.set_yticks([])
    
    cax = fig.add_subplot(gs[1, 3])
    sm = plt.cm.ScalarMappable(cmap=err_cmap, norm=plt.Normalize(-emax, emax))
    fig.colorbar(sm, cax=cax, label='Error (m)')
    
    # Row 3: Overlap, Scatter, and Info
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(np.ones_like(gt) * config.INACTIVE_COLOR, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax.contour(np.abs(gt) > 0.1, colors='blue', linewidths=2)
    ax.contour(np.abs(pred) > 0.1, colors='red', linewidths=1.5, linestyles='--')
    ax.set_title('Overlap (Blue=GT, Red=Pred)')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[2, 1])
    active_flat = active_mask.ravel()
    if active_flat.sum() > 0:
        gt_a = gt.ravel()[active_flat]
        pred_a = pred.ravel()[active_flat]
        sc = ax.scatter(gt_a, pred_a, c=np.abs(gt_a), cmap='viridis', alpha=0.6, s=25)
        plt.colorbar(sc, ax=ax, label='|GT| (m)')
        lim = [min(gt_a.min(), pred_a.min()) - 0.5, max(gt_a.max(), pred_a.max()) + 0.5]
        ax.plot(lim, lim, 'r--', lw=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('GT (m)'); ax.set_ylabel('Pred (m)')
    corr = metrics.get('corr_active', 0)
    ax.set_title(f'Scatter | Corr: {corr:.4f}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Info text
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    train_str = "\n".join([f"    θ{i+1}: S_c={t['S_c']:.3f}, q_c={t['q_c']:.2f}, k={t['k']:.4f}"
                          for i, t in enumerate(train_thetas)])
    test_str = f"    θ_test: S_c={test_theta['S_c']:.3f}, q_c={test_theta['q_c']:.2f}, k={test_theta['k']:.4f}"
    
    txt = [
        "=" * 50,
        f"  {title_prefix} - Parameter Generalization",
        "=" * 50, "",
        f"  TRAINING PARAMETERS ({len(train_thetas)} sets):",
        train_str, "",
        "  TEST PARAMETER (unseen):",
        test_str, "",
        "-" * 50,
        f"  ACTIVE REGION ({active_mask.sum()} cells)",
        f"    RMSE: {metrics.get('rmse_active', 0):.4f} m",
        f"    Corr: {metrics.get('corr_active', 0):.4f}", "",
        f"  FULL DOMAIN ({H}×{W})",
        f"    SSIM: {metrics.get('ssim', 0):.4f}",
        "=" * 50
    ]
    ax.text(0.02, 0.98, '\n'.join(txt), transform=ax.transAxes, fontsize=9,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def save_combined_results_plot_logscale(
    gt, pred, active_mask, sensors, train_thetas, test_theta,
    metrics, config, filepath, title_prefix="LAPIS (Log Scale)"
):
    """
    Save combined results figure with log-scale values.
    Full domain, zoomed views, scatter plot, and metrics.
    """
    H, W = gt.shape
    error = pred - gt
    abs_error = np.abs(error)
    r0, r1, c0, c1 = get_bbox(active_mask, config.ZOOM_PAD)
    
    log_cmap = get_log_cmap()
    
    # Log scale parameters
    epsilon = 1e-3
    vmin_log = -2.0
    vmax_log = np.log(max(np.abs(gt).max(), np.abs(pred).max()) + epsilon)
    err_vmax_log = np.log(np.abs(error).max() + epsilon)
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.08],
                          height_ratios=[1, 1, 0.8], hspace=0.25, wspace=0.15)
    
    # Row 1: Full domain (log scale)
    ax = fig.add_subplot(gs[0, 0])
    plot_log_magnitude(ax, gt, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
    ax.contour(active_mask.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.scatter(sensors[:, 1], sensors[:, 0], c=config.SENSOR_COLOR,
               s=config.SENSOR_SIZE, marker=config.SENSOR_MARKER, zorder=10)
    ax.add_patch(patches.Rectangle((c0, r0), c1-c0, r1-r0, lw=2, ec='cyan', fc='none', ls='--'))
    ax.set_title(f'GT log|DoD| (θ_test)', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[0, 1])
    plot_log_magnitude(ax, pred, active_mask, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
    ax.contour(active_mask.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.add_patch(patches.Rectangle((c0, r0), c1-c0, r1-r0, lw=2, ec='cyan', fc='none', ls='--'))
    ax.set_title('Pred log|DoD|', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[0, 2])
    plot_log_magnitude(ax, abs_error, active_mask, log_cmap, vmin_log, err_vmax_log, config.INACTIVE_COLOR)
    ax.contour(active_mask.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title(f'log|Error| | Max: {np.abs(error).max():.2f}m', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    
    cax = fig.add_subplot(gs[0, 3])
    sm = plt.cm.ScalarMappable(cmap=log_cmap, norm=plt.Normalize(vmin_log, vmax_log))
    fig.colorbar(sm, cax=cax, label='log(|DoD| + ε)')
    
    # Row 2: Zoomed (log scale)
    gt_z = gt[r0:r1, c0:c1]
    pred_z = pred[r0:r1, c0:c1]
    err_z = abs_error[r0:r1, c0:c1]
    mask_z = active_mask[r0:r1, c0:c1]
    
    ax = fig.add_subplot(gs[1, 0])
    plot_log_magnitude(ax, gt_z, mask_z, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
    ax.contour(mask_z.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title('GT log|DoD| (Zoomed)', fontweight='bold', color='darkblue')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[1, 1])
    plot_log_magnitude(ax, pred_z, mask_z, log_cmap, vmin_log, vmax_log, config.INACTIVE_COLOR)
    ax.contour(mask_z.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title('Pred log|DoD| (Zoomed)', fontweight='bold', color='darkblue')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[1, 2])
    plot_log_magnitude(ax, err_z, mask_z, log_cmap, vmin_log, err_vmax_log, config.INACTIVE_COLOR)
    ax.contour(mask_z.astype(float), levels=[0.5], colors=['lime'], linewidths=2)
    ax.set_title('log|Error| (Zoomed)', fontweight='bold', color='darkblue')
    ax.set_xticks([]); ax.set_yticks([])
    
    cax = fig.add_subplot(gs[1, 3])
    sm = plt.cm.ScalarMappable(cmap=log_cmap, norm=plt.Normalize(vmin_log, err_vmax_log))
    fig.colorbar(sm, cax=cax, label='log(|Error| + ε)')
    
    # Row 3: Overlap, Scatter, and Info
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(np.ones_like(gt) * config.INACTIVE_COLOR, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax.contour(np.abs(gt) > 0.1, colors='blue', linewidths=2)
    ax.contour(np.abs(pred) > 0.1, colors='red', linewidths=1.5, linestyles='--')
    ax.set_title('Overlap (Blue=GT, Red=Pred)')
    ax.set_xticks([]); ax.set_yticks([])
    
    ax = fig.add_subplot(gs[2, 1])
    active_flat = active_mask.ravel()
    if active_flat.sum() > 0:
        gt_a = gt.ravel()[active_flat]
        pred_a = pred.ravel()[active_flat]
        log_gt = np.sign(gt_a) * np.log(np.abs(gt_a) + epsilon)
        log_pred = np.sign(pred_a) * np.log(np.abs(pred_a) + epsilon)
        sc = ax.scatter(log_gt, log_pred, c=np.abs(gt_a), cmap='viridis', alpha=0.6, s=25)
        plt.colorbar(sc, ax=ax, label='|GT| (m)')
        lim = [min(log_gt.min(), log_pred.min()) - 0.5, max(log_gt.max(), log_pred.max()) + 0.5]
        ax.plot(lim, lim, 'r--', lw=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('sign(GT)·log(|GT|+ε)'); ax.set_ylabel('sign(Pred)·log(|Pred|+ε)')
    corr = metrics.get('corr_active', 0)
    ax.set_title(f'Log Scatter | Corr: {corr:.4f}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Info text
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    train_str = "\n".join([f"    θ{i+1}: S_c={t['S_c']:.3f}, q_c={t['q_c']:.2f}, k={t['k']:.4f}"
                          for i, t in enumerate(train_thetas)])
    test_str = f"    θ_test: S_c={test_theta['S_c']:.3f}, q_c={test_theta['q_c']:.2f}, k={test_theta['k']:.4f}"
    
    txt = [
        "=" * 50,
        f"  {title_prefix} - Parameter Generalization",
        "=" * 50, "",
        f"  TRAINING PARAMETERS ({len(train_thetas)} sets):",
        train_str, "",
        "  TEST PARAMETER (unseen):",
        test_str, "",
        "-" * 50,
        f"  ACTIVE REGION ({active_mask.sum()} cells)",
        f"    RMSE: {metrics.get('rmse_active', 0):.4f} m",
        f"    Corr: {metrics.get('corr_active', 0):.4f}", "",
        f"  FULL DOMAIN ({H}×{W})",
        f"    SSIM: {metrics.get('ssim', 0):.4f}",
        "=" * 50
    ]
    ax.text(0.02, 0.98, '\n'.join(txt), transform=ax.transAxes, fontsize=9,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def save_shred_video(gt, pred, active_mask, theta_info, config, base_dir, base_name="shred_reconstruction"):
    """
    Save SHRED reconstruction videos - one combined GIF and individual panel videos.
    """
    T = min(gt.shape[0], pred.shape[0])
    dod_cmap = get_dod_cmap()
    err_cmap = get_err_cmap()
    vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1.0)
    
    # Compute error sequence
    err_seq = pred[:T] - gt[:T]
    emax = max(np.abs(err_seq).max(), 0.1)
    
    theta_str = f"θ: S_c={theta_info['S_c']:.3f}, q_c={theta_info['q_c']:.2f}, k={theta_info['k']:.4f}"
    
    print(f"  Saving {base_name} animations...")
    
    # Save combined GIF (3 panels)
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, left=0.02, right=0.95)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    
    sm = plt.cm.ScalarMappable(cmap=dod_cmap, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, cax=cax, label='DoD (m)')
    fig.suptitle(f'Forward SHRED Reconstruction ({theta_str})', fontsize=12, fontweight='bold')
    
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
        axes[1].set_title(f'SHRED Prediction (t={t})')
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
                           'SHRED Prediction (t={t})', 'DoD (m)', config,
                           base_dir, f"{base_name}_pred")
    
    save_error_panel_video(gt[:T], pred[:T], active_mask, err_cmap, config,
                          base_dir, f"{base_name}_error")


def save_shred_video_logscale(gt, pred, active_mask, theta_info, config, base_dir, base_name="shred_reconstruction_logscale"):
    """
    Save SHRED reconstruction videos with log-scale.
    """
    T = min(gt.shape[0], pred.shape[0])
    log_cmap = get_log_cmap()
    
    epsilon = 1e-3
    all_data = np.concatenate([gt.ravel(), pred.ravel()])
    vmin_log = -2.0
    vmax_log = np.log(np.abs(all_data).max() + epsilon)
    
    all_errors = pred[:T] - gt[:T]
    err_vmax_log = np.log(np.abs(all_errors).max() + epsilon)
    
    theta_str = f"θ: S_c={theta_info['S_c']:.3f}, q_c={theta_info['q_c']:.2f}, k={theta_info['k']:.4f}"
    
    print(f"  Saving {base_name} animations...")
    
    # Save combined GIF (3 panels)
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, left=0.02, right=0.95)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    
    sm = plt.cm.ScalarMappable(cmap=log_cmap, norm=plt.Normalize(vmin_log, vmax_log))
    fig.colorbar(sm, cax=cax, label='log(|value| + ε)')
    fig.suptitle(f'Forward SHRED Reconstruction - Log Scale ({theta_str})', fontsize=12, fontweight='bold')
    
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


def save_error_distribution_plot(gt, pred_lapis, pred_shred, active_mask, config, filepath, use_log_y=True):
    """
    Save distribution plot comparing GT DoD values vs reconstruction errors for the last frame.
    """
    # Last frame, active cells only
    gt_last = gt[-1][active_mask]
    err_lapis = (pred_lapis[-1] - gt[-1])[active_mask]
    err_shred = (pred_shred[-1] - gt[-1])[active_mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    all_vals = np.concatenate([gt_last, err_lapis, err_shred])
    bins = np.linspace(all_vals.min(), all_vals.max(), 120)

    ax.hist(gt_last, bins=bins, alpha=0.5, label='GT DoD values', color='steelblue',
            edgecolor='none')
    ax.hist(err_lapis, bins=bins, alpha=0.5, label='LAPIS error (pred − GT)', color='tomato',
            edgecolor='none')
    ax.hist(err_shred, bins=bins, alpha=0.5, label='SHRED error (pred − GT)', color='orange',
            edgecolor='none')

    if use_log_y:
        ax.set_yscale('log')
    ax.set_xlabel('Value (m)', fontsize=12)
    ax.set_ylabel('Count (log scale)' if use_log_y else 'Count', fontsize=12)
    ax.set_title(
        f'Last-Frame Distribution: GT DoD vs Reconstruction Errors\n'
        f'(active region, {int(active_mask.sum())} cells)',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    stats_txt = (
        f"GT:  μ={gt_last.mean():.2f}, σ={gt_last.std():.2f}, "
        f"range=[{gt_last.min():.1f}, {gt_last.max():.1f}]\n"
        f"LAPIS err: μ={err_lapis.mean():.2f}, σ={err_lapis.std():.2f}, "
        f"|max|={np.abs(err_lapis).max():.1f}\n"
        f"SHRED err: μ={err_shred.mean():.2f}, σ={err_shred.std():.2f}, "
        f"|max|={np.abs(err_shred).max():.1f}"
    )
    ax.text(0.98, 0.97, stats_txt, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")
