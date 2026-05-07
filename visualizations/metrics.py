# Metrics and latent trajectory visualization for LAPIS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def save_metrics_plot(train_losses, val_metrics, config, filepath):
    """Save training metrics plot."""
    filepath = Path(filepath)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    axes[0].plot(train_losses, 'b-', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Validation metrics
    if val_metrics:
        epochs = [m['epoch'] for m in val_metrics]
        rmse = [m['rmse_active'] for m in val_metrics]
        corr = [m['corr_active'] for m in val_metrics]
        
        ax1 = axes[1]
        ax2 = ax1.twinx()
        
        l1 = ax1.plot(epochs, rmse, 'b-', label='RMSE (active)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE (m)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        l2 = ax2.plot(epochs, corr, 'r-', label='Correlation')
        ax2.set_ylabel('Correlation', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        axes[1].set_title('Validation Metrics')
        ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_latent_trajectory_plot(latent_trajectory, config, filepath, title="Latent Trajectory"):
    """Save latent space trajectory visualization."""
    filepath = Path(filepath)
    
    T, latent_dim = latent_trajectory.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Top-left: First few dimensions over time
    n_dims_to_show = min(8, latent_dim)
    for i in range(n_dims_to_show):
        axes[0, 0].plot(latent_trajectory[:, i], label=f'z_{i}', alpha=0.7)
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Latent value')
    axes[0, 0].set_title(f'First {n_dims_to_show} latent dimensions')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-right: Heatmap of all dimensions
    im = axes[0, 1].imshow(latent_trajectory.T, aspect='auto', cmap='RdBu_r')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Latent dimension')
    axes[0, 1].set_title('Latent trajectory heatmap')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Bottom-left: Latent norm over time
    latent_norm = np.linalg.norm(latent_trajectory, axis=1)
    axes[1, 0].plot(latent_norm, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('||z||')
    axes[1, 0].set_title('Latent norm over time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: PCA projection (if enough dimensions)
    if latent_dim >= 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_trajectory)
        
        # Color by time
        colors = np.arange(T)
        scatter = axes[1, 1].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                                      c=colors, cmap='viridis', s=30, alpha=0.7)
        axes[1, 1].plot(latent_pca[:, 0], latent_pca[:, 1], 'k-', alpha=0.3, linewidth=0.5)
        axes[1, 1].scatter(latent_pca[0, 0], latent_pca[0, 1], c='green', s=100, 
                          marker='o', edgecolors='black', linewidths=2, label='Start', zorder=5)
        axes[1, 1].scatter(latent_pca[-1, 0], latent_pca[-1, 1], c='red', s=100, 
                          marker='s', edgecolors='black', linewidths=2, label='End', zorder=5)
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1, 1].set_title('PCA projection of latent trajectory')
        axes[1, 1].legend()
        plt.colorbar(scatter, ax=axes[1, 1], label='Time step')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")
