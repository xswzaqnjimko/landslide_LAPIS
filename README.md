# LAPIS-SHRED Landslide

**Paper:** *Spatiotemporal Landslide Runout Reconstruction from Sparse Observations with LAPIS-SHRED* (Zhang et al., 2026)

## Overview

This repository applies the LAPIS-SHRED framework to landslide runout reconstruction using the Mass Wasting Runout (MWR) model. The `shred_jax/` module is a JAX/Flax adaptation of the core [LAPIS-SHRED](https://github.com/xswzaqnjimko/LAPIS-SHRED) architecture, with application-specific modifications for the landslide reconstruction setting.

## Repository Structure

```
├── data_generation_scenarios.py   # Generate MWR simulation ensembles from scenario configs
├── lapis_landslide_bi.py          # Main training & inference pipeline (backward + forward)
├── shred_jax/                     # SHRED and temporal model implementations (JAX/Flax)
│   ├── shred.py                   # Seq2Seq SHRED encoder-decoder
│   ├── temporal_models.py         # Backward and forward temporal models
│   ├── training.py                # Training loops and loss functions
│   ├── inference.py               # LAPIS inference pipelines
│   ├── datasets.py                # Data loading and preprocessing
│   └── utils.py                   # Utilities
└── visualizations/                # Plotting utilities
```

## Requirements

- Python ≥ 3.10
- JAX, Flax, Optax
- NumPy, SciPy, scikit-learn, scikit-image
- Landlab (for MWR data generation)

## Usage

**1. Generate simulation ensembles:**

```bash
python data_generation_scenarios.py --scenario_file scenarios.json
```

**2. Train and evaluate:**

```bash
# Backward inference (terminal-only observation)
python lapis_landslide_bi.py --location mt_st_helens --year 2009 \
    --climate_class wet --theta_group dryer_soil2

# Forward inference (short initial window)
python lapis_landslide_bi.py --location mt_st_helens --year 2009 \
    --climate_class wet --theta_group dryer_soil2 --inference_mode forward
```

## License

[MIT](LICENSE)
