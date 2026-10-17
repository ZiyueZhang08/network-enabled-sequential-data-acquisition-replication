# Reproduction Runbook

This runbook records the commands needed to reproduce the current artifacts.

## Environment

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Compile check:

```powershell
python -m compileall src experiments
```

Core tests:

```powershell
python -m unittest discover -s tests
```

## Data

Required local data:

- MovieLens 100K: `data/raw/ml-100k/`
- Flixster GCMC: `data/raw/gcmc/flixster/training_test_dataset.mat`
- Douban GCMC: `data/raw/gcmc/douban/training_test_dataset.mat`

## Main Three-Dataset Runs

MovieLens homogeneous response:

```powershell
python experiments\run_movielens.py --config configs\movielens_100k_svdpp_optimized_20seeds.json
```

MovieLens heterogeneous response:

```powershell
python experiments\run_movielens_robustness.py --config configs\movielens_100k_svdpp_robustness_20seeds.json
```

Flixster tuned 20 seeds:

```powershell
python experiments\run_movielens.py --config configs\flixster_svdpp_tuned_20seeds_20cycles.json
```

Douban tuned 20 seeds:

```powershell
python experiments\run_movielens.py --config configs\douban_svdpp_tuned_20seeds_20cycles.json
```

Generate Figure 4 / Figure 5 style MovieLens panels:

```powershell
python experiments\plot_movielens_figures.py
```

Generate the three-dataset main figure:

```powershell
python experiments\plot_three_dataset_main.py
```

The frozen deliverable snapshot is documented in:

```text
reports/final_snapshot.md
```

## Secondary Runs

MovieLens bandit baselines:

```powershell
python experiments\run_movielens.py --config configs\movielens_100k_svdpp_baselines_20seeds.json
```

MovieLens DLS ablation:

```powershell
python experiments\run_movielens.py --config configs\movielens_100k_svdpp_dls_ablation_20seeds.json
```

Flixster and Douban smoke tests:

```powershell
python experiments\run_movielens.py --config configs\flixster_svdpp_smoke.json
python experiments\run_movielens.py --config configs\douban_svdpp_smoke.json
```

## Tuning Examples

Lambda grid:

```powershell
python experiments\run_movielens_grid.py --config configs\douban_svdpp_lambda_grid_1seed.json
```

Diffusion grid:

```powershell
python experiments\run_movielens_grid.py --config configs\douban_svdpp_diffusion_grid_1seed.json
```

## Douban Mechanism Follow-Ups

Temperature-controlled exponential sampling:

```powershell
python experiments\run_movielens_grid.py --config configs\douban_svdpp_tau_grid_3seeds.json
```

Observational entropy scaling:

```powershell
python experiments\run_movielens_grid.py --config configs\douban_svdpp_entropy_lambda_grid_3seeds.json
python experiments\run_movielens_grid.py --config configs\douban_svdpp_entropy_coverage_grid_3seeds.json
```

20-seed dynamic sigma / no-DLS confirmation:

```powershell
python experiments\run_movielens.py --config configs\douban_svdpp_dynamic_no_dls_20seeds.json
```
