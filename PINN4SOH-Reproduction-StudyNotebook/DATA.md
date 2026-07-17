# Data preparation

This repository does not redistribute third-party battery datasets or the upstream repository snapshot. Download data from the original providers and review their terms before use.

## Route A: reproduce the published PINN training workflow

The cleaned data loader consumes the 17-column, preprocessed CSV files distributed in the upstream PINN4SOH repository. This is the recommended route when the goal is to reproduce data loading, model training, and evaluation rather than feature extraction.

1. Open the [upstream PINN4SOH repository](https://github.com/wang-fujin/PINN4SOH).
2. Obtain the contents of its `data/XJTU data/` directory.
3. Copy the CSV files into this repository without changing their names:

   ```text
   PINN4SOH-Reproduction-StudyNotebook/
   └── data/
       └── XJTU data/
           ├── 2C_battery-1.csv
           ├── 2C_battery-2.csv
           ├── 3C_battery-1.csv
           └── ...
   ```

4. From the repository root, run:

   ```powershell
   python scripts/check_setup.py
   python src/verify_pipeline.py
   ```

The expected CSV schema is 16 statistical feature columns followed by `capacity`; the data loader inserts `cycle index` immediately before `capacity`. It normalizes all 17 model inputs, including `cycle index`, while converting `capacity` to SOH by dividing by nominal capacity.

The upstream project does not display a repository-wide license. Downloading the files yourself does not remove any usage restrictions that may apply; consult the upstream authors when redistribution or commercial use is planned.

## Route B: study feature extraction from raw data

The public raw datasets and preprocessing library can be used with `src/01_extract_features.py` and `notebooks/01_feature_extraction.ipynb`:

| Dataset or resource | Source |
|---|---|
| XJTU raw dataset | <https://zenodo.org/records/10963339> |
| TJU dataset | <https://zenodo.org/records/6405084> |
| HUST dataset | <https://data.mendeley.com/datasets/nsc7hnsg4s/2> |
| MIT dataset | <https://data.matr.io/1/projects/5c48dd2bc625d700019f3204> |
| Public preprocessing library | <https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library> |

The original 16-feature extraction source used to generate the upstream CSV files was not published. Therefore, raw-data extraction in this repository is a documented reverse implementation based on the paper and public resources; it is not expected to reproduce the upstream CSV values exactly. See `docs/差异分析.md` before comparing extracted features.

## Optional environment variables

The default data root is `data/` in the extracted repository. Paths can be overridden without editing source code:

```powershell
$env:PINN4SOH_DATA_ROOT='D:\path\to\data'
$env:PINN4SOH_XJTU_MAT_PATH='D:\path\to\2C_battery-1.mat'
$env:PINN4SOH_PREPROCESSING_LIB='D:\path\to\Battery-dataset-preprocessing-code-library'
```

`PINN4SOH_DATA_ROOT` must point to the directory that contains the `XJTU data` subdirectory, not to the subdirectory itself.

## Files that must remain local

Do not commit downloaded datasets, raw MAT files, upstream snapshots, machine-specific absolute paths, model checkpoints, or full repeated-run output directories. The provided `.gitignore` excludes the normal locations for these artifacts.
