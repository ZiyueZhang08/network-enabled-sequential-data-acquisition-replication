# Paper-faithful MovieLens alignment and gap-reduction audit

Date: 2026-07-09

## Goal

Move the clean-room MovieLens Figure 4a reproduction closer to the paper without promoting engineering-only enhancements as the main line. The priority was to improve fidelity in mechanisms already present in the paper:

- Eq. 3.3-3.4 Gaussian DLS over network/user representation `u_sn`.
- Eq. 3.6 observational entropy with Schur-style observed-subspace correction.
- Section 3.5.1 dynamic tuning over `lambda_H`, `sigma`, and MovieLens `p=0`.
- Full-pool exponential acquisition sampling.

## Code alignment changes

| Paper requirement | Previous local state | Change made |
|---|---|---|
| Eq. 3.3 normalizes Gaussian DLS over all users | Paper faithful branch still used graph/top-k neighbor DLS | Added `paper_dls_scope` and set the main config to `full` |
| Acquisition and tuning should use the same DLS definition | Main loop and inner tuning both hard-coded neighbor DLS | Routed both through `scoped_gaussian_dls_user_factors` |
| MovieLens network representation should come from auxiliary user information when available | Loader built GC-MC/Monti user features but discarded them after KNN graph construction | Added `RatingDataset.user_features` and let the paper runner pass them as `network_embedding` |
| MovieLens has no network diffusion | Already `p=0` in main config and tuning grid | Kept unchanged |
| Paper uses exponential sampling | Already `selection: exp` full pool | Kept unchanged |
| Paper-style dynamic tuning uses observed train/acquisition/test validation | Already implemented every 5 cycles with paper metric | Kept unchanged for this pass |

Changed files:

- `src/dls_paper.py`
- `src/acquisition.py`
- `src/tuning.py`
- `src/data_loader.py`
- `experiments/run_paper_faithful_movielens.py`
- `configs/paper_faithful_movielens_v1.json`
- `tests/test_core.py`

## 20-seed results

All rows use the same MovieLens u1 fallback/Table-1-density protocol, 20 cycles, batch size 100, SVD++ rank 12/epochs 6, and paper-style tuning grid.

| Version | Main change | DU-Net PDF-literal RMSE | DU-Net standard RMSE | DU-Net reward | DU-Net local w-RMSE |
|---|---|---:|---:|---:|---:|
| v1 cleanroom | coherence + observational-lite proxy | 0.007680 | 1.086114 | 1354.526 | 0.986346 |
| v2 full Schur | Eq. 3.6-style observed correction | 0.007670 | 1.084746 | 1355.751 | 0.982636 |
| v3 full DLS over spectral embedding | Eq. 3.3 full user scope | 0.007671 | 1.084806 | 1353.176 | 0.985574 |
| v4 user-feature full DLS | full DLS over MovieLens auxiliary features | 0.007654 | 1.082427 | 1336.936 | 0.984814 |

Current v4 comparator gaps, using DU-Net minus comparator:

| Comparator | PDF-literal RMSE gap | Standard RMSE gap | Reward gap | Local w-RMSE gap |
|---|---:|---:|---:|---:|
| LinUCB concat | -0.000076 | -0.010759 | -22.326 | 0.024937 |
| LinUCB MF interaction | 0.000035 | 0.004975 | 182.202 | -0.090616 |

Negative RMSE gaps favor DU-Net; positive reward gaps favor DU-Net. The v4 change improves DU-Net's standard RMSE gap against MF interaction from about 0.00735 to about 0.00498, but it does not close it fully.

## Reflection

The best-performing faithful change in this pass was not full DLS alone. Full DLS over the old spectral graph embedding produced an effective neighborhood in the hundreds of users and slightly worsened the 20-seed mean. That is consistent with over-smoothing: MovieLens has 943 users, and the spectral proxy made the all-user Gaussian weights too diffuse.

Using the MovieLens auxiliary features as `u_sn` corrected that failure mode. It lowered the effective neighborhood scale on seed 0 from hundreds to tens of users for the selected sigma values, and the 20-seed RMSE improved. This is a stronger paper-faithful explanation than tuning constants: the paper's DLS depends on the network representation, so a weak proxy for `u_sn` contaminates both reward and entropy scoring.

There is a tradeoff. The v4 main line improves RMSE fidelity but lowers cumulative reward relative to v2/v3. I would not use reward-tail proxy as the validation objective to recover reward, because that would move away from the paper's main Figure 4 RMSE-style reporting. The clean interpretation is: v4 is a better RMSE-aligned paper-faithful main line, while reward acquisition still exposes a remaining objective mismatch.

## Verification

- `python -m py_compile src\dls_paper.py src\tuning.py src\acquisition.py`
- `python -m py_compile src\data_loader.py experiments\run_paper_faithful_movielens.py`
- `python -m pytest tests\test_core.py -q` - 28 passed
- `python -m pytest -q` - 58 passed
- 20-seed v4 metrics: `outputs/tables/paper_faithful_movielens_userfeat_full_dls_schur_v4_20seeds_metrics.csv`
- 20-seed v4 summary: `outputs/tables/paper_faithful_movielens_userfeat_full_dls_schur_v4_20seeds_final_summary.csv`

## Next step

The next faithful performance branch should be backend validation, but only under two constraints:

1. Use the same SVD++ backend for DU-Net and all MF-centered comparators.
2. Select rank/epochs by inner validation, not by test gap.

Recommended screen:

- `rank = [20, 30]`
- `epochs = [12, 20]`
- `learning_rate = [0.015, 0.02, 0.025]`
- `regularization = [0.04, 0.06]`
- Keep v4 acquisition: user-feature full DLS, Schur entropy, full-pool exponential sampling, MovieLens `p=0`.

Promotion rule: promote only if the same-backend 20-seed confirmation reduces DU-Net's standard RMSE gap to MF-interaction without sacrificing the paper protocol. If the backend improves all methods equally, keep v4 as the main line and report the remaining gap as likely source/split/backend-detail uncertainty rather than continuing parameter tuning.
