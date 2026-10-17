# External data setup

Raw datasets and processed masks are excluded from this public export. The synthetic example in the root README does not require downloads.

## MovieLens 100K

Obtain the dataset from [GroupLens](https://grouplens.org/datasets/movielens/100k/), read its usage terms, and extract the archive into `data/raw/`:

```bash
mkdir -p data/raw
curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip -o data/raw/ml-100k.zip
unzip data/raw/ml-100k.zip -d data/raw
```

Expected files include `data/raw/ml-100k/u.data`, `u.user`, `u.item`, `u1.base`, and `u1.test`. The original dataset README requires separate permission for redistribution; this repository supplies download instructions instead of a copy of the ratings or demographics.

The general loader uses the available MovieLens data. The frozen v2 attempt uses a locally defined `u1` fallback with 943 users, 1681 items, and 1364 initial observations. This fallback is not a recovered author mask.

The fallback-preparation helper can be run after downloading:

```bash
python scripts/find_or_prepare_movielens_monti_split.py
```

## Flixster and Douban

Obtain the preprocessed [Flixster file](https://github.com/riannevdberg/gc-mc/blob/master/gcmc/data/flixster/training_test_dataset.mat) and [Douban file](https://github.com/riannevdberg/gc-mc/blob/master/gcmc/data/douban/training_test_dataset.mat) from the original [Graph Convolutional Matrix Completion repository](https://github.com/riannevdberg/gc-mc). Its [preprocessing implementation](https://github.com/riannevdberg/gc-mc/blob/master/gcmc/preprocessing.py) documents the MATLAB fields. Use these datasets under their applicable terms.

Place the files at:

```text
data/raw/gcmc/flixster/training_test_dataset.mat
data/raw/gcmc/douban/training_test_dataset.mat
```

`src/data_loader.py` expects MATLAB v7.3/HDF5 matrices named `M`, `Otraining`, `Otest`, and `W_users`, read with `h5py`. Generic ratings CSV files are not drop-in replacements. The legacy GCMC preprocessing code is not required to execute this project's own loader.

## Scope

The datasets, official preprocessing masks, locally generated fallback masks, and the DU-Net authors' exact masks are distinct artifacts. Downloading a public dataset does not establish that a local split matches the paper's split. See the [strict final audit](../reports/paper_exact_attempt_movielens_v2_final_exact_gate.md).
