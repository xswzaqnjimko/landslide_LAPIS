#!/usr/bin/env python3
"""
MWR Landslide LAPIS-SHRED Data Generation Pipeline — Scenario-Based
=================================================

Generates MWR simulation ensembles for LAPIS training and evaluation.

Directory layout produced:
  data/<location>/<year>/<fixed_mwr_tag>/<climate_class>/<theta_group>/
      metadata.json
      topology/
      simulations/
          sim_theta_00/ ... sim_theta_14/   <- per-iteration DoD .npy files
          ground_truth/

Usage:
  python data_generation_scenarios.py --scenario_file scenarios_2009_split_16.json
  python data_generation_scenarios.py --scenario_file scenarios_2009_split_16.json --force
  python data_generation_scenarios.py --list
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# RUN CONFIGURATION

class RunConfig:
    MODE = "batch"
    SCENARIO_FILE = "scenarios_2009_split_16.json"
    ONLY = ["dry/dryer_soil1"] # None

    LOCATION       = "mt_st_helens"
    YEAR           = "2009"
    FIXED_MWR_TAG  = "2_FIXED_MWR_PARAMS"
    CLIMATE_CLASS  = "wet"
    THETA_GROUP    = "dryer_soil2"
    THETA_BASE     = {"S_c": 0.28, "q_c": 0.70, "k": 0.012, "soil_depth": 1.25, "depth_parameter": 1.25}
    THETA_TRUE     = {"S_c": 0.30, "q_c": 0.80, "k": 0.010, "soil_depth": 1.25, "depth_parameter": 1.25}
    N_ENSEMBLE     = 15
    PERTURBATION_RANGE = 0.80
    PERTURBATION_SEED  = 42

    FORCE = True # False


# PROJECT LAYOUT

class ProjectPaths:
    ROOT = Path(__file__).resolve().parent

    CODE   = ROOT
    DATA   = ROOT / "data"
    RESULT = ROOT / "results"

    ORIGINAL = DATA / "original"

    @classmethod
    def original_topo(cls, year: str) -> Path:
        return cls.ORIGINAL / f"Slide_{year}"

    @classmethod
    def scenario_dir(cls, location: str, year: str, fixed_mwr_tag: str,
                     climate_class: str, theta_group: str) -> Path:
        return cls.DATA / location / year / fixed_mwr_tag / climate_class / theta_group

    @classmethod
    def result_dir(cls, location: str, year: str, fixed_mwr_tag: str,
                   climate_class: str, theta_group: str) -> Path:
        return cls.RESULT / location / year / fixed_mwr_tag / climate_class / theta_group


# FIXED MWR PARAMETER SETS

FIXED_MWR_PARAM_SETS = {
    "1_FIXED_MWR_PARAMS": {
        "description": "2021 event",
        "typical_flow_thickness_of_erosion_zone": 2,
        "typical_slope_of_erosion_zone": 0.25,
        "max_flow_depth_observed_in_field": 5,
        "Dp": 0.316,
    },
    "2_FIXED_MWR_PARAMS": {
        "description": "2009 event",
        "typical_flow_thickness_of_erosion_zone": 4,
        "typical_slope_of_erosion_zone": 0.25,
        "max_flow_depth_observed_in_field": 6,
        "Dp": 0.316,
    },
}


# TOPOLOGY FILE MAPPING
# Input grids are ESRI ASCII files for the Landlab MassWastingRunout (MWR) model (https://github.com/landlab/landlab).

TOPO_FILE_MAP = {
    "dem.asc": [
        "pre-runout_DEM_CM2009.asc",
        "pre-runout_DEM_CM2021.asc",
        "pre_runout_DEM.asc",
        "dem.asc",
        "DEM.asc",
    ],
    "landslide_polygon.asc": [
        "landslide_nodes_CM2009.asc",
        "landslide_nodes_CM2021.asc",
        "landslide_polygon.asc",
        "mass__wasting_id.asc",
    ],
    "landslide_depth.asc": [
        "landslide_depth_CM2009.asc",
        "landslide_depth_CM2021.asc",
        "landslide_depth.asc",
    ],
    "observed_DoD.asc": [
        "observed_DoD_CM2009.asc",
        "observed_DoD_CM2021.asc",
        "observed_DoD.asc",
    ],
}


# UTILITY FUNCTIONS

def load_esri_ascii_meta(filepath: Path) -> dict:
    """Load only the header metadata from an ESRI ASCII grid file."""
    with open(filepath) as f:
        meta = {}
        for _ in range(6):
            parts = f.readline().strip().split()
            if len(parts) >= 2:
                key = parts[0].lower()
                try:
                    meta[key] = (int(float(parts[1]))
                                 if key in ('ncols', 'nrows')
                                 else float(parts[1]))
                except ValueError:
                    meta[key] = parts[1]
        if 'nodata_value' not in meta:
            meta['nodata_value'] = -9999
    return meta


# PERTURBATION

THETA_PARAMS = ["S_c", "q_c", "k", "soil_depth", "depth_parameter"]


def generate_perturbed_thetas(theta_base: Dict[str, float],
                              n: int = 15,
                              seed: int = 42,
                              perturbation_range: float = 0.80,
                              param_ranges: Optional[Dict[str, List[float]]] = None
                              ) -> List[Dict[str, float]]:
    """
    Generate *n* perturbed parameter dicts.
    """
    rng = np.random.RandomState(seed)
    result = []
    for _ in range(n):
        perturbed = {}
        for p in THETA_PARAMS:
            if param_ranges and p in param_ranges:
                lo, hi = param_ranges[p]
                perturbed[p] = rng.uniform(lo, hi)
            else:
                base = theta_base[p]
                factor = 1.0 + rng.uniform(-perturbation_range, perturbation_range)
                perturbed[p] = base * factor
        result.append(perturbed)
    return result


# MWR SIMULATION
# Following the Landlab MWR notebook.

def run_mwr_simulation(theta: Dict[str, float],
                       topology_dir: Path,
                       output_dir: Path,
                       fixed_params: Dict,
                       deep_seated: bool = True) -> Dict[int, np.ndarray]:
    """
    Run a single MWR simulation.
    """
    print(f"    Running MWR: S_c={theta['S_c']:.4f}, q_c={theta['q_c']:.4f}, "
          f"k={theta['k']:.5f}, soil={theta['soil_depth']:.2f}, "
          f"depth_param={theta['depth_parameter']:.2f}")

    try:
        from landlab.components import (FlowDirectorMFD, FlowAccumulator,
                                        DepressionFinderAndRouter)
        from landlab.components.mass_wasting_runout import MassWastingRunout
        from landlab.io import esri_ascii
    except ImportError as e:
        print(f"ERROR: Landlab not available: {e}")
        sys.exit(1)

    # GRID SETUP

    dem_path = topology_dir / "dem.asc"
    with open(dem_path) as fp:
        mg = esri_ascii.load(fp, name="topographic__elevation", at="node")
    z = mg.at_node["topographic__elevation"]

    nr = mg.number_of_node_rows
    nc = mg.number_of_node_columns

    poly_path = topology_dir / "landslide_polygon.asc"
    with open(poly_path) as fp:
        esri_ascii.load(fp, name="mass__wasting_id", at="node", out=mg)
    mg.at_node["mass__wasting_id"] = mg.at_node["mass__wasting_id"].astype(int)

    soil_depth_val = theta['soil_depth']
    mg.add_field("soil__thickness",
                 np.ones(mg.number_of_nodes) * soil_depth_val, at="node")

    if deep_seated:
        mw_id = 1
        ls_mask = mg.at_node['mass__wasting_id'] == mw_id
        if ls_mask.any():
            lsnode_y = mg.node_y[ls_mask]
            ymin = lsnode_y.min()
            yd = lsnode_y - ymin
            dp = theta['depth_parameter']
            lsd = dp + (9.0 / 200.0) * yd
            mg.at_node['soil__thickness'][ls_mask] = lsd
            print(f"      deep_seated: depth_param={dp:.2f}, "
                  f"soil range [{lsd.min():.2f}, {lsd.max():.2f}] m "
                  f"over {ls_mask.sum()} landslide nodes")
    else:
        depth_path = topology_dir / "landslide_depth.asc"
        if depth_path.exists():
            with open(depth_path) as fp:
                esri_ascii.load(fp, name="landslide__depth",
                                at="node", out=mg)
            ld = np.abs(mg.at_node["landslide__depth"])
            ld[ld == 0] = 3.0
            ls_mask = mg.at_node['mass__wasting_id'] == 1
            mg.at_node['soil__thickness'][ls_mask] = ld[ls_mask]

    mg.set_watershed_boundary_condition(z)

    mg.at_node["particle__diameter"] = (
        np.ones(mg.number_of_nodes) * fixed_params['Dp'])

    fa = FlowAccumulator(mg, "topographic__elevation",
                         flow_director="FlowDirectorD8")
    fa.run_one_step()

    df = DepressionFinderAndRouter(mg)
    df.map_depressions()

    mg.add_field("topographic__initial_elevation",
                 mg.at_node["topographic__elevation"].copy(),
                 at="node", clobber=True)

    # FLOW ROUTING

    for f in ['flow__sink_flag', 'flow__link_to_receiver_node',
              'flow__receiver_node', 'topographic__steepest_slope']:
        if f in mg.at_node:
            mg.delete_field(loc="node", name=f)

    fd = FlowDirectorMFD(mg, diagonals=True, partition_method="slope")
    fd.run_one_step()

    mwr = MassWastingRunout(
        grid=mg,
        critical_slope=theta['S_c'],
        threshold_flux=theta['q_c'],
        erosion_coefficient=theta['k'],
        tracked_attributes=["particle__diameter"],
        save=True,
        typical_flow_thickness_of_erosion_zone=(
            fixed_params['typical_flow_thickness_of_erosion_zone']),
        typical_slope_of_erosion_zone=(
            fixed_params['typical_slope_of_erosion_zone']),
        max_flow_depth_observed_in_field=(
            fixed_params['max_flow_depth_observed_in_field']),
    )
    mwr.run_one_step()

    # EXTRACT DoD AND SAVE

    init_elev = mg.at_node["topographic__initial_elevation"]

    dod_evolution = {}
    for i, evo_flat in mwr.saver.runout_evo_maps[0].items():
        dod_flat = evo_flat - init_elev

        # Reshape from Landlab 1-D (bottom-to-top rows) to 2-D grid
        landlab_grid = dod_flat.reshape((nr, nc))
        # Flip to geographic image order (row 0 = north)
        geo_grid = np.flipud(landlab_grid)
        dod_evolution[i] = geo_grid

    # Save .npy files
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, dod_frame in dod_evolution.items():
        np.save(output_dir / f"iter_{i:04d}.npy", dod_frame)

    # ---- Print final-frame stats ----
    iter_keys = sorted(dod_evolution.keys())
    final_dod_2d = dod_evolution[iter_keys[-1]]
    final_dod_flat = final_dod_2d.ravel()

    E_mask = final_dod_flat < 0
    A_mask = final_dod_flat > 0
    cell_area = mg.dx * mg.dy

    if E_mask.any():
        E_mean = final_dod_flat[E_mask].mean()
        E_area = E_mask.sum() * cell_area
        print(f"      Erosion:     mean {E_mean:.2f} m, "
              f"area {E_area:.0f} m2, vol {E_mean * E_area:.0f} m3")
    if A_mask.any():
        A_mean = final_dod_flat[A_mask].mean()
        A_area = A_mask.sum() * cell_area
        print(f"      Aggradation: mean {A_mean:.2f} m, "
              f"area {A_area:.0f} m2, vol {A_mean * A_area:.0f} m3")

    all_dod = np.stack([dod_evolution[k] for k in iter_keys])
    print(f"      -> {len(dod_evolution)} iterations saved to {output_dir.name}/")
    print(f"        Final DoD range: [{final_dod_2d.min():.2f}, {final_dod_2d.max():.2f}]")
    print(f"        All-frames |DoD| max: {np.abs(all_dod).max():.2f}")

    return dod_evolution


def load_cached_simulation(cache_dir: Path) -> Optional[Dict[int, np.ndarray]]:
    """Load previously computed simulation from cache."""
    if not (cache_dir / 'iter_0000.npy').exists():
        return None
    return {int(f.stem.split('_')[1]): np.load(f)
            for f in sorted(cache_dir.glob("iter_*.npy"))}


# TOPOLOGY SETUP

def setup_topology(topo_dest: Path, topo_source: Path) -> bool:
    """Copy canonical topology files from original data."""
    topo_dest.mkdir(parents=True, exist_ok=True)
    available = {f.name: f for f in topo_source.glob("*.asc")}
    all_ok = True

    for canonical, candidates in TOPO_FILE_MAP.items():
        found = None
        for cand in candidates:
            if cand in available:
                found = available[cand]
                break
        if found:
            dest_path = topo_dest / canonical
            if not dest_path.exists():
                shutil.copy(found, dest_path)
                tag = f" (<- {found.name})" if found.name != canonical else ""
                print(f"      Copied {canonical}{tag}")
        else:
            if canonical == "observed_DoD.asc":
                print(f"      Note: {canonical} not found (optional)")
            else:
                print(f"      WARNING: missing required {canonical}")
                all_ok = False
    return all_ok


# SINGLE-SCENARIO PIPELINE

def run_scenario(location: str, year: str, fixed_mwr_tag: str,
                 climate_class: str, theta_group: str,
                 theta_base: Dict[str, float],
                 theta_true: Dict[str, float],
                 n_ensemble: int = 15,
                 perturbation_range: float = 0.80,
                 perturbation_seed: int = 42,
                 deep_seated: bool = True,
                 description: str = "",
                 param_ranges: Optional[Dict[str, List[float]]] = None,
                 force: bool = False) -> bool:
    """Full pipeline for one scenario."""
    scenario_path = ProjectPaths.scenario_dir(
        location, year, fixed_mwr_tag, climate_class, theta_group)
    topo_dir = scenario_path / "topology"
    sim_dir  = scenario_path / "simulations"

    tag = f"{location}/{year}/{fixed_mwr_tag}/{climate_class}/{theta_group}"
    print(f"\n{'='*70}")
    print(f"  Scenario: {tag}")
    print(f"  {description}")
    print(f"{'='*70}")

    # ---- check cache ----
    if not force and _simulations_complete(sim_dir, n_ensemble):
        print(f"  Already complete (use --force to regenerate)")
        return True

    # ---- resolve fixed MWR params ----
    if fixed_mwr_tag not in FIXED_MWR_PARAM_SETS:
        print(f"  ERROR: unknown fixed_mwr_tag '{fixed_mwr_tag}'")
        print(f"         available: {list(FIXED_MWR_PARAM_SETS.keys())}")
        return False
    fixed_params = {k: v for k, v in FIXED_MWR_PARAM_SETS[fixed_mwr_tag].items()
                    if k != "description"}

    # ---- topology ----
    topo_source = ProjectPaths.original_topo(year)
    if not topo_source.exists():
        print(f"  ERROR: original topology not found at {topo_source}")
        return False

    print(f"\n  [1/4] Setting up topology from {topo_source.name}/")
    if not setup_topology(topo_dir, topo_source):
        print(f"  ERROR: missing required topology files")
        return False

    # ---- generate perturbed thetas ----
    if param_ranges:
        print(f"\n  [2/4] Generating {n_ensemble} ensemble thetas (bounded ranges)")
        for p, (lo, hi) in param_ranges.items():
            print(f"    {p}: [{lo}, {hi}]")
    else:
        print(f"\n  [2/4] Generating {n_ensemble} perturbed thetas "
              f"(+/-{perturbation_range*100:.0f}%)")
    perturbed = generate_perturbed_thetas(
        theta_base, n=n_ensemble, seed=perturbation_seed,
        perturbation_range=perturbation_range,
        param_ranges=param_ranges)

    for i, th in enumerate(perturbed):
        pct = {p: (th[p] - theta_base[p]) / theta_base[p] * 100
               for p in THETA_PARAMS}
        print(f"    t_{i:02d}: S_c={th['S_c']:.4f}({pct['S_c']:+.0f}%) "
              f"q_c={th['q_c']:.4f}({pct['q_c']:+.0f}%) "
              f"k={th['k']:.5f}({pct['k']:+.0f}%) "
              f"soil={th['soil_depth']:.2f}({pct['soil_depth']:+.0f}%) "
              f"dp={th['depth_parameter']:.2f}({pct['depth_parameter']:+.0f}%)")

    print(f"\n    t_true: S_c={theta_true['S_c']:.4f} "
          f"q_c={theta_true['q_c']:.4f} k={theta_true['k']:.5f} "
          f"soil={theta_true['soil_depth']:.2f} "
          f"dp={theta_true['depth_parameter']:.2f}")

    # ---- run ensemble simulations ----
    print(f"\n  [3/4] Running {n_ensemble} ensemble simulations...")
    sim_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for i, th in enumerate(perturbed):
        out = sim_dir / f"sim_theta_{i:02d}"
        if not force and (out / "iter_0000.npy").exists():
            print(f"    [sim_theta_{i:02d}] cached, skipping")
            continue
        print(f"\n    [sim_theta_{i:02d}]")
        run_mwr_simulation(th, topo_dir, out, fixed_params,
                           deep_seated=deep_seated)

    # ---- run ground truth ----
    gt_dir = sim_dir / "ground_truth"
    if not force and (gt_dir / "iter_0000.npy").exists():
        print(f"    [ground_truth] cached, skipping")
    else:
        print(f"\n    [ground_truth]")
        run_mwr_simulation(theta_true, topo_dir, gt_dir, fixed_params,
                           deep_seated=deep_seated)

    elapsed = time.time() - t0
    print(f"\n  Simulations completed in {elapsed:.1f}s")

    # ---- write metadata ----
    print(f"\n  [4/4] Writing metadata")
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "location": location,
        "year": year,
        "fixed_mwr_tag": fixed_mwr_tag,
        "fixed_mwr_params": FIXED_MWR_PARAM_SETS[fixed_mwr_tag],
        "climate_class": climate_class,
        "theta_group": theta_group,
        "description": description,
        "deep_seated": deep_seated,
        "n_ensemble": n_ensemble,
        "perturbation_range": perturbation_range,
        "perturbation_seed": perturbation_seed,
        "param_ranges": param_ranges,
        "theta_base": theta_base,
        "theta_true": theta_true,
        "perturbed_thetas": perturbed,
        "directory_layout": {
            "topology": "topology/",
            "ensemble_sims": f"simulations/sim_theta_00 .. sim_theta_{n_ensemble-1:02d}",
            "ground_truth_sim": "simulations/ground_truth/",
            "each_sim_contains": "iter_0000.npy, iter_0001.npy, ... "
                                 "(DoD = elevation - initial_elevation, per iteration)",
        },
        "grid": {
            "note": "read from DEM header at generation time",
        },
        "loader": "landlab.io.esri_ascii (native)",
        "npy_row_order": "geographic (row 0 = north, top of map)",
        "npy_contents": "DoD (evo_maps[i] - topographic__initial_elevation)",
    }

    try:
        dem_meta = load_esri_ascii_meta(topo_dir / "dem.asc")
        metadata["grid"]["nrows"] = dem_meta.get("nrows")
        metadata["grid"]["ncols"] = dem_meta.get("ncols")
        metadata["grid"]["cellsize"] = dem_meta.get("cellsize")
    except Exception:
        pass

    with open(scenario_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Scenario complete: {tag}")
    print(f"    Data at: {scenario_path}")
    return True


def _simulations_complete(sim_dir: Path, n_ensemble: int) -> bool:
    if not sim_dir.exists():
        return False
    for i in range(n_ensemble):
        if not (sim_dir / f"sim_theta_{i:02d}" / "iter_0000.npy").exists():
            return False
    if not (sim_dir / "ground_truth" / "iter_0000.npy").exists():
        return False
    return True


# BATCH MODE

def run_from_scenario_file(scenario_file: Path, force: bool = False,
                           scenarios_filter: Optional[List[str]] = None):
    with open(scenario_file) as f:
        spec = json.load(f)

    defaults = spec.get("global_defaults", {})
    scenarios = spec["scenarios"]

    print(f"\n{'#'*70}")
    print(f"  LAPIS Data Generation -- Batch Mode")
    print(f"  Scenario file: {scenario_file.name}")
    print(f"  {len(scenarios)} scenario(s) defined")
    print(f"{'#'*70}")

    results = {}
    for sc in scenarios:
        cc = sc["climate_class"]
        tg = sc["theta_group"]
        key = f"{cc}/{tg}"

        if scenarios_filter and key not in scenarios_filter:
            continue

        ok = run_scenario(
            location=sc.get("location", defaults.get("location", "mt_st_helens")),
            year=sc.get("year", defaults.get("year", "2009")),
            fixed_mwr_tag=sc.get("fixed_mwr_tag",
                                 defaults.get("fixed_mwr_tag", "2_FIXED_MWR_PARAMS")),
            climate_class=cc,
            theta_group=tg,
            theta_base=sc["theta_base"],
            theta_true=sc["theta_true"],
            n_ensemble=sc.get("n_ensemble", defaults.get("n_ensemble", 15)),
            perturbation_range=sc.get("perturbation_range",
                                      defaults.get("perturbation_range", 0.80)),
            perturbation_seed=sc.get("perturbation_seed",
                                     defaults.get("perturbation_seed", 42)),
            deep_seated=sc.get("deep_seated", defaults.get("deep_seated", True)),
            description=sc.get("description", ""),
            param_ranges=sc.get("param_ranges", None),
            force=force,
        )
        results[key] = ok

    print(f"\n\n{'#'*70}")
    print(f"  BATCH SUMMARY")
    print(f"{'#'*70}")
    for key, ok in results.items():
        status = "ok" if ok else "FAIL"
        print(f"  {status}  {key}")
    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} scenarios completed successfully")


# LIST MODE

def list_existing():
    data = ProjectPaths.DATA
    if not data.exists():
        print(f"  No data directory at {data}")
        return

    print(f"\n  Existing scenarios under {data}/\n")
    for location_dir in sorted(data.iterdir()):
        if not location_dir.is_dir() or location_dir.name == "original":
            continue
        for year_dir in sorted(location_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for fmwr_dir in sorted(year_dir.iterdir()):
                if not fmwr_dir.is_dir():
                    continue
                for cc_dir in sorted(fmwr_dir.iterdir()):
                    if not cc_dir.is_dir():
                        continue
                    for tg_dir in sorted(cc_dir.iterdir()):
                        if not tg_dir.is_dir():
                            continue
                        sim_dir = tg_dir / "simulations"
                        meta_file = tg_dir / "metadata.json"
                        has_meta = "M" if meta_file.exists() else " "
                        n_sims = len(list(sim_dir.glob("sim_theta_*"))) \
                            if sim_dir.exists() else 0
                        has_gt = (sim_dir / "ground_truth" /
                                  "iter_0000.npy").exists() \
                            if sim_dir.exists() else False
                        gt_tag = "GT" if has_gt else "  "

                        rel = tg_dir.relative_to(data)
                        desc = ""
                        if meta_file.exists():
                            try:
                                with open(meta_file) as f:
                                    m = json.load(f)
                                desc = m.get("description", "")[:50]
                            except Exception:
                                pass
                        print(f"  [{has_meta}] [{gt_tag}] {n_sims:2d} ens  "
                              f"{rel}  {desc}")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="LAPIS Data Generation -- Scenario-Based",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch: generate all scenarios from a file
  python data_generation_scenarios.py --scenario_file scenarios_2009_split_16.json

  # Batch: generate only selected scenarios
  python data_generation_scenarios.py --scenario_file scenarios_2009_split_16.json \\
      --only dry/dryer_soil1 dry/wetter_soil2

  # Force regeneration
  python data_generation_scenarios.py --scenario_file scenarios_2009_split_16.json --force

  # List existing
  python data_generation_scenarios.py --list
"""
    )

    parser.add_argument("--scenario_file", type=Path, default=None,
                        help="JSON file with scenario definitions")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run these scenarios (climate_class/theta_group)")

    parser.add_argument("--location", default=None)
    parser.add_argument("--year", default=None)
    parser.add_argument("--fixed_mwr_tag", default=None)
    parser.add_argument("--climate_class", default=None)
    parser.add_argument("--theta_group", default=None)
    parser.add_argument("--theta_base", type=str, default=None,
                        help="JSON string for theta_base")
    parser.add_argument("--theta_true", type=str, default=None,
                        help="JSON string for theta_true")
    parser.add_argument("--n_ensemble", type=int, default=None)
    parser.add_argument("--perturbation_range", type=float, default=None)
    parser.add_argument("--perturbation_seed", type=int, default=None)

    parser.add_argument("--force", action="store_true", default=False,
                        help="Force regeneration")
    parser.add_argument("--list", action="store_true",
                        help="List existing scenarios and exit")

    args = parser.parse_args()

    no_cli_args = (
        not args.list
        and args.scenario_file is None
        and args.climate_class is None
        and args.theta_group is None
    )

    if no_cli_args:
        print("  (No CLI args detected -- using RunConfig from script header)")
        if RunConfig.MODE == "list":
            list_existing()
            return
        elif RunConfig.MODE == "batch":
            sf = Path(RunConfig.SCENARIO_FILE)
            if not sf.is_absolute():
                sf = ProjectPaths.CODE / sf
            if not sf.exists():
                print(f"ERROR: scenario file not found: {sf}")
                sys.exit(1)
            run_from_scenario_file(sf, force=RunConfig.FORCE,
                                   scenarios_filter=RunConfig.ONLY)
            return
        elif RunConfig.MODE == "single":
            run_scenario(
                location=RunConfig.LOCATION,
                year=RunConfig.YEAR,
                fixed_mwr_tag=RunConfig.FIXED_MWR_TAG,
                climate_class=RunConfig.CLIMATE_CLASS,
                theta_group=RunConfig.THETA_GROUP,
                theta_base=RunConfig.THETA_BASE,
                theta_true=RunConfig.THETA_TRUE,
                n_ensemble=RunConfig.N_ENSEMBLE,
                perturbation_range=RunConfig.PERTURBATION_RANGE,
                perturbation_seed=RunConfig.PERTURBATION_SEED,
                force=RunConfig.FORCE,
            )
            return

    if args.list:
        list_existing()
        return

    if args.scenario_file:
        if not args.scenario_file.exists():
            alt = ProjectPaths.CODE / args.scenario_file
            if alt.exists():
                args.scenario_file = alt
            else:
                print(f"ERROR: scenario file not found: {args.scenario_file}")
                sys.exit(1)
        run_from_scenario_file(args.scenario_file,
                               force=args.force,
                               scenarios_filter=args.only)
        return

    if not args.climate_class or not args.theta_group:
        print("ERROR: provide --scenario_file for batch mode, or "
              "--climate_class + --theta_group for single mode")
        parser.print_help()
        sys.exit(1)

    if not args.theta_base or not args.theta_true:
        print("ERROR: --theta_base and --theta_true required in single mode "
              "(as JSON strings)")
        sys.exit(1)

    run_scenario(
        location=args.location or RunConfig.LOCATION,
        year=args.year or RunConfig.YEAR,
        fixed_mwr_tag=args.fixed_mwr_tag or RunConfig.FIXED_MWR_TAG,
        climate_class=args.climate_class,
        theta_group=args.theta_group,
        theta_base=json.loads(args.theta_base),
        theta_true=json.loads(args.theta_true),
        n_ensemble=args.n_ensemble or RunConfig.N_ENSEMBLE,
        perturbation_range=args.perturbation_range or RunConfig.PERTURBATION_RANGE,
        perturbation_seed=args.perturbation_seed or RunConfig.PERTURBATION_SEED,
        force=args.force,
    )


if __name__ == "__main__":
    main()
