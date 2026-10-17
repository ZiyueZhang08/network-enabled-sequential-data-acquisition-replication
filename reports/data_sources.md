# Data Sources

## MovieLens 100K

- Status: downloaded.
- Local path: `data/raw/ml-100k/`
- Source URL: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- Files used: `u.data`, `u.user`.
- Reproduction use: real-data MovieLens run with 1% initial training observations, candidate acquisition set from the remaining train observations, and a hold-out test set.

## Flixster and Douban

- Status: downloaded.
- Paper requirement: the DU-Net paper evaluates Flixster, Douban, and MovieLens. It states that Flixster and Douban include social/follower network data, while MovieLens uses a K-nearest-neighbor similarity network from auxiliary user information.
- GCMC-style file: `training_test_dataset.mat`.
- Expected fields from `gcmc/preprocessing.py`: `M`, `Otraining`, `Otest`, `W_users`; Flixster also expects `W_movies`.
- Checked source: `https://github.com/riannevdberg/gc-mc`.
- Source paths:
  - `https://github.com/riannevdberg/gc-mc/blob/master/gcmc/data/flixster/training_test_dataset.mat`
  - `https://github.com/riannevdberg/gc-mc/blob/master/gcmc/data/douban/training_test_dataset.mat`
- Local paths:
  - `data/raw/gcmc/flixster/training_test_dataset.mat` (2,776,221 bytes)
  - `data/raw/gcmc/douban/training_test_dataset.mat` (2,778,554 bytes)
- Finding: the files are MATLAB v7.3/HDF5 `.mat` files. `h5py` was installed and an HDF5 loader was added in `src/data_loader.py`.

## Next Data Step

The loader reads `M` as ratings, `W_users` as the user network, and preserves `Otraining`/`Otest` as the official GCMC/Monti train/test split. Flixster and Douban smoke tests plus 20-cycle single-seed runs have completed; see `reports/gcmc_flixster_douban_reproduction.md`.
