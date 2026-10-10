# LAPIS-SHRED Landslide

**Paper:** *Spatiotemporal Landslide Runout Reconstruction from Sparse Observations with LAPIS-SHRED* (Zhang et al., 2026)

## Overview

This repository applies the LAPIS-SHRED framework to landslide runout reconstruction using the [Mass Wasting Runout (MWR) model](https://github.com/landlab/landlab). The `shred_jax/` module is a JAX/Flax adaptation of the core [LAPIS-SHRED](https://github.com/xswzaqnjimko/LAPIS-SHRED) architecture, with application-specific modifications for the landslide reconstruction setting.

## Repository Structure

```
├── data_generation_scenarios.py   # Generate MWR simulation ensembles from scenario configs
├── lapis_landslide_bi.py          # Main training & inference pipeline (backward + forward)
├── scenarios_2009_split_16.json   # Demo parameter ranges and ground truth for 16 scenarios
├── shred_jax/                     # SHRED and temporal model implementations (JAX/Flax)
│   ├── shred.py                   # Seq2Seq SHRED encoder-decoder
│   ├── temporal_models.py         # Backward and forward temporal models
│   ├── training.py                # Training loops and loss functions
│   ├── inference.py               # LAPIS inference pipelines
│   └── datasets.py                # Data loading and preprocessing
└── visualizations/                # Plotting utilities
```

## Requirements

- Python ≥ 3.10
- JAX, Flax, Optax
- NumPy, Matplotlib, SciPy, scikit-learn
- Landlab (for MWR data generation)

## Data

Data generation in this demo reads three site input grids (ESRI ASCII) from `data/original/Slide_{year}/`, e.g.:

- `pre-runout_DEM_CM2009.asc`
- `landslide_nodes_CM2009.asc`
- `landslide_depth_CM2009.asc`

These files are not included in this repository and are available from the authors on request.

## Usage

**1. Generate simulation ensembles:**

```bash
python data_generation_scenarios.py --scenario_file scenarios_2009_split_16.json
```

**2. Train and evaluate:**

```bash
# Backward inference (terminal-only observation)
python lapis_landslide_bi.py --location mt_st_helens --year 2009 \
    --climate_class dry --theta_group dryer_soil1

# Forward inference (short initial window)
python lapis_landslide_bi.py --location mt_st_helens --year 2009 \
    --climate_class dry --theta_group dryer_soil1 --inference_mode forward
```

## License

[MIT](LICENSE)
