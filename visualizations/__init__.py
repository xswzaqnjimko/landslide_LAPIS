# Visualization module for LAPIS

from .colormaps import get_dod_cmap, get_err_cmap, get_log_cmap, get_elevation_cmap
from .utils import add_boundary, get_bbox, plot_with_dark_inactive, plot_log_magnitude
from .elevation import save_initial_elevation, save_initial_elevation_zoomed
from .frames import (
    save_animation_formats,
    save_single_panel_video,
    save_single_panel_video_logscale,
    save_error_panel_video,
    save_error_panel_video_logscale,
    save_video,
    save_video_logscale,
    save_zoomed_frame_figure,
    save_zoomed_frame_figure_logscale
)
from .ensemble import (
    save_ensemble_results,
    save_ensemble_results_logscale,
    save_shred_results_logscale,
    save_shred_video,
    save_shred_video_logscale,
    save_combined_results_plot,
    save_combined_results_plot_logscale,
    save_error_distribution_plot
)
from .metrics import save_metrics_plot, save_latent_trajectory_plot
