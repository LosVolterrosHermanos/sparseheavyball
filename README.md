# sparseheavyball

Lightweight simulation workspace for sparse-momentum ODE scaling studies.

## What Is Here

- `simulations/master_ode.py`  
  Core master ODE model and linear ODE integrators.

- `simulations/run_region_panel.py`  
  Baseline panel over representative interior points of the scaling regions.

- `simulations/run_rescaled_limit_panel.py`  
  Regime-rescaled master ODE trajectories overlaid with simplified limit ODEs.

- `simulations/run_coherent_finite_size_sweep.py`  
  Coherent-regime finite-size sweep across multiple `d`.

- `simulations/run_last_iter_risk_heatmap.py`  
  Grid search over \((\kappa,\gamma)\), with per-point LR optimization and risk heatmaps.

- `simulations/export_last_iter_split_heatmaps.py`  
  Exports standalone optimal-LR and last-iterate-risk heatmaps from a CSV grid.

## Quick Start

```bash
source /opt/e-py/bin/activate
```

Generate a sample heatmap (active-iterates timescale):

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl python simulations/run_last_iter_risk_heatmap.py \
  --d 2000 \
  --sigma 1.2 \
  --time 100 \
  --time-rule active-iterates \
  --search-factor 1.1 \
  --binary-search-steps 6 \
  --max-dyadic-steps 20 \
  --grid-fineness 31 \
  --num-workers 8 \
  --figure-title "README sample: Active-Iterates" \
  --out-heatmaps simulations/outputs/readme_sample_heatmap.png \
  --out-eta-scaled simulations/outputs/readme_sample_eta_scaled.png \
  --out-csv simulations/outputs/readme_sample_grid.csv
```

Generate a sample efficiency heatmap (sample-timescale):

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl python simulations/run_last_iter_risk_heatmap.py \
  --d 2000 \
  --sigma 1.2 \
  --time 24 \
  --time-rule sample \
  --search-factor 1.1 \
  --binary-search-steps 6 \
  --max-dyadic-steps 20 \
  --grid-fineness 31 \
  --num-workers 8 \
  --figure-title "README sample: Sample Efficiency" \
  --out-heatmaps simulations/outputs/readme_sample_efficiency_heatmap.png \
  --out-eta-scaled /tmp/readme_sample_efficiency_eta_scaled.png \
  --out-csv /tmp/readme_sample_efficiency_grid.csv
```

## Sample Output

`simulations/outputs/readme_sample_heatmap.png`

![README sample heatmap](simulations/outputs/readme_sample_heatmap.png)

`simulations/outputs/readme_sample_efficiency_heatmap.png`

![README sample efficiency heatmap](simulations/outputs/readme_sample_efficiency_heatmap.png)
