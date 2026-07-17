# Repository file guide

[简体中文](README.zh-CN.md)

This document explains the purpose of the main directories and files in `PINN4SOH-Reproduction-StudyNotebook`. It is intended to help readers understand the project layout, module relationships, and recommended learning path.

## 1. Repository scope

This repository is an unofficial, study-oriented reproduction of *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis* (Nature Communications, 2024). It organizes the complete lithium-ion battery state-of-health (SOH) workflow, including feature extraction, data preprocessing, PINN construction, loss design, model training, and evaluation.

The repository provides two complementary forms of material:

- reusable and verifiable Python modules under `src/`;
- step-by-step learning notebooks under `notebooks/`.

Third-party battery datasets are not included. Follow `DATA.md` to prepare the required data before running the complete training workflow.

## 2. Repository layout

```text
PINN4SOH-Reproduction-StudyNotebook/
├── assets/          Figures used by the documentation and notebooks
├── docs/            Difference analyses and verification reports
├── experiments/     Repeated training and baseline experiment entry points
├── notebooks/       Step-by-step learning notebooks, numbered 01 to 06
├── scripts/         Lightweight user-facing training entry points
├── src/             Core data, model, loss, training, and evaluation code
├── DATA.md          Data sources, expected layout, and path configuration
├── README.md        English project overview
├── README.zh-CN.md  Chinese project overview
├── RESULTS.md       Verification status and reference results
├── requirements.txt Python dependency list
└── THIRD_PARTY_NOTICES.md  Third-party sources and rights notices
```

Local runs may also create `data/`, `outputs/`, `results/`, `experiments/results/`, and `experiments/images/`. These directories contain datasets, generated artifacts, and experiment results and are normally excluded from Git.

## 3. Root-level files

| File | Purpose |
|---|---|
| `README.md` | Main English project page covering scope, setup, data preparation, quick verification, experiment entry points, and reproduction boundaries. |
| `README.zh-CN.md` | Chinese version of the main project page. |
| `DATA.md` | Documents two data routes: reproducing training from upstream preprocessed CSV files or studying feature extraction from raw data. It also describes the expected directory layout and environment variables. |
| `RESULTS.md` | Summarizes the current verification status, deterministic minimal-run reference metrics, and the location of repeated-experiment reports. |
| `requirements.txt` | Lists the Python packages required by the project. Install them with `pip install -r requirements.txt`. |
| `THIRD_PARTY_NOTICES.md` | Records the sources of third-party papers, code, and datasets, together with relevant usage considerations. |
| `文件说明.md` | Chinese repository file guide corresponding to this document. |

## 4. `src/`: core implementation

### `src/config.py`

Centralizes repository paths for data, raw MAT files, the external preprocessing library, extracted features, and generated figures. The default paths can be overridden with the following environment variables:

- `PINN4SOH_DATA_ROOT`: data root containing the `XJTU data/` directory;
- `PINN4SOH_XJTU_MAT_PATH`: XJTU MAT file used by the feature-extraction example;
- `PINN4SOH_PREPROCESSING_LIB`: path to the external battery preprocessing library.

The `require_path()` helper checks required paths and raises a clear error when a resource is missing.

### `src/01_extract_features.py`

Demonstrates how statistical features can be derived from raw battery-cycle data and written to `outputs/features/`. The implementation is a documented reverse implementation based on the paper and public resources. It is useful for studying feature engineering but is not expected to reproduce the upstream 16-feature CSV values exactly because the original extraction source was not published.

### `src/02_dataloader.py`

Loads and preprocesses the training data. Its responsibilities include:

- reading preprocessed XJTU CSV files;
- applying a three-sigma outlier filter;
- converting capacity to SOH;
- inserting `cycle index`;
- normalizing the 16 statistical features together with `cycle index`;
- constructing adjacent-cycle time-step pairs;
- splitting training, validation, and test data;
- wrapping tensors in PyTorch `DataLoader` objects.

This module is the main data entry point for model training.

### `src/03_model.py`

Defines the PINN architecture. Its main components are:

- `Sin`: sine activation function;
- `MLP`: base multilayer perceptron;
- `Predictor`: data-driven prediction branch;
- `SolutionU`: solution network describing degradation evolution;
- `PINN`: top-level model combining the subnetworks and automatic differentiation;
- `count_parameters()`: helper for counting trainable parameters.

### `src/03_model_runner.py`

Provides a small model demonstration. It loads an example data batch, constructs the PINN, and checks model inputs, outputs, and forward propagation.

### `src/04_loss.py`

Defines the loss terms used for PINN training. It combines data-fitting, physics-informed, and monotonicity constraints and includes a small loss-computation demonstration. The paper description and public upstream implementation differ in the monotonicity term; read `docs/差异分析.md` before interpreting this implementation choice.

### `src/05_train.py`

Implements the reusable training loop, including:

- `LR_Scheduler` for validation-based learning-rate adjustment;
- `Train()` for batch training, validation, testing, history collection, and early stopping;
- best-model state tracking and restoration;
- aggregation of component losses and evaluation results.

This module supplies the training functionality. Normal user runs begin with a script under `scripts/` or `experiments/`.

### `src/06_eval.py`

Evaluates model predictions and generates figures. It calculates MAE, MAPE, MSE, and RMSE and plots predicted-versus-observed values and degradation curves. Default figure paths are configured in `config.py`.

### `src/module_loader.py`

Dynamically loads Python files whose names begin with numbers. Files such as `02_dataloader.py` and `03_model.py` cannot be imported through ordinary Python module syntax, so other scripts use `load_clean_module()` to load them.

### `src/verify_pipeline.py`

Runs an end-to-end pipeline check that connects data loading, model forward propagation, and loss calculation. Use it to confirm that the core modules work together before starting a longer training run.

## 5. `notebooks/`: guided learning material

| Notebook | Topic | Related source file |
|---|---|---|
| `01_feature_extraction.ipynb` | Raw cycle data and statistical feature extraction | `src/01_extract_features.py` |
| `02_data_preprocessing.ipynb` | Outlier handling, normalization, SOH conversion, and time-step pairing | `src/02_dataloader.py` |
| `03_forward_propagation.ipynb` | PINN subnetworks, activations, forward propagation, and automatic differentiation | `src/03_model.py` |
| `04_loss_design.ipynb` | Data, physics-informed, and monotonicity loss design | `src/04_loss.py` |
| `05_training.ipynb` | Optimizer use, learning-rate scheduling, early stopping, and training loops | `src/05_train.py` |
| `06_evaluation.ipynb` | Metrics, prediction comparison, and degradation-curve visualization | `src/06_eval.py` |

The notebooks are best read in numerical order. They emphasize explanation and interactive exploration, while `src/` contains the reusable implementation.

## 6. `scripts/`: lightweight entry points

### `scripts/run_minimal_training.py`

Runs a deterministic minimal end-to-end training experiment. The default 10-epoch run verifies the data, model, loss, training, evaluation, and plotting pipeline. It is a smoke test rather than a reproduction of the paper's full 200-epoch performance.

Outputs are written to `results/minimal_run/`.

```powershell
python scripts/run_minimal_training.py
```

## 7. `experiments/`: full experiment entry points

### `experiments/run_50epoch_training.py`

Runs a medium-scale 50-epoch experiment. It provides an intermediate validation stage between the minimal smoke test and the complete paper-scale experiments and produces metrics, training history, and figures.

### `experiments/run_XJTU_standard_2C.py`

Runs standard repeated experiments for the XJTU 2C condition. It supports command-line configuration, random seeds, repeated runs, per-run output, and aggregate statistics. Results are written to `experiments/results/`, and figures are written to `experiments/images/`.

### `experiments/run_xjtu_all_baselines.py`

Runs XJTU experiments across multiple conditions or baseline configurations and produces consistent summaries for comparison.

### `experiments/run_paper_60.py`

Runs the complete 60-run experiment suite used by the repository reports. This is a computationally expensive entry point and should be used only after the minimal run and a single-condition experiment succeed. A seed such as `--seed 42` can be supplied for more deterministic reruns.

## 8. `docs/`: reports and implementation notes

| File or directory | Purpose |
|---|---|
| `差异分析.md` | Compares the paper, the public upstream implementation, and this repository, including known differences in feature extraction, normalization, and loss definitions. |
| `最终验证报告.md` | Records the final verification of the code, environment, data flow, and core execution pipeline. |
| `完整60次实验报告.md` | Summarizes the configuration, statistics, and conclusions of the complete 60-run experiment. |
| `cycle_index归一化对训练结果的影响解读报告.md` | Analyzes how `cycle index` normalization affects training stability and evaluation metrics. |
| `results/cycle_normalized_summary.json` | Lightweight machine-readable summary for the current normalized-cycle implementation. |
| `results/cycle_unscaled_summary.json` | Historical baseline summary for the earlier unscaled-cycle implementation. |

Read `docs/差异分析.md` and the corresponding verification report before comparing metrics with the paper.

## 9. `assets/`: figure resources

The `assets/` directory stores figures referenced by the README files, reports, and notebooks:

- `02_特征提取/`: statistical features, CC/CV curves, capacity, and entropy figures;
- `03_数据预处理与成对构造/`: adjacent time-step pairing diagram;
- `04_PINN前向传播/`: activation comparison and forward-output figures;
- `06_训练循环/`: learning-rate, early-stopping, and training-history figures;
- `07_评估与可视化/`: prediction comparisons, degradation curves, MAPE comparisons, and smoke-test results.

These files support explanation and presentation and are not model inputs.

## 10. Core module flow

```text
config.py
   ├── 01_extract_features.py ──> feature CSV
   └── 02_dataloader.py ────────> DataLoader
                                      │
03_model.py ──────────────────────────┤
                                      v
04_loss.py ─────────────────────> 05_train.py
                                      │
                                      v
                                  06_eval.py
                                      │
                                      v
                             metrics, JSON, figures
```

`scripts/run_minimal_training.py` and the files under `experiments/` are top-level entry points. They use `module_loader.py` to access the numbered core modules.

## 11. Recommended workflow

1. Read `README.md` for the project scope and reproduction boundaries.
2. Follow `DATA.md` to prepare `data/XJTU data/*.csv`.
3. Install the dependencies listed in `requirements.txt`.
4. Read and run the notebooks in order from `01` to `06`.
5. Run `src/02_dataloader.py`, `src/03_model.py`, `src/04_loss.py`, and `src/verify_pipeline.py` to check the core modules.
6. Run `scripts/run_minimal_training.py` for a minimal end-to-end experiment.
7. Read `docs/差异分析.md` before running the formal or repeated experiments under `experiments/`.
8. Interpret the output using `RESULTS.md` and the reports under `docs/`.

## 12. Important notes

- Do not commit third-party datasets, upstream repository snapshots, model checkpoints, or large repeated-run output directories.
- The minimal training run verifies pipeline execution; it does not demonstrate reproduction of the paper's complete performance.
- Neural-network training is stochastic. Without a fixed seed, compare overall trends and aggregate statistics rather than individual decimal values.
- The original 16-feature extraction source was not published, so features derived from raw MAT files in this repository may not match the upstream CSV values exactly.
- Review `THIRD_PARTY_NOTICES.md` and the licenses or terms of the original sources before reusing or redistributing derived materials.
