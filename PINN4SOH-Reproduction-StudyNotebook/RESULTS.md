# Results and verification status

## Current release status

This file describes the release version in which the 16 statistical features and `cycle index` are normalized together. Metrics from the earlier unscaled-cycle implementation are retained only as a historical baseline in `docs/results/cycle_unscaled_summary.json`.

The following checks were completed in the Python 3.12 verification environment:

- Static compilation of the published Python files and Notebook code cells.
- Dependency consistency check with `pip check`.
- Six-condition XJTU data-layout and preprocessing check with `scripts/check_setup.py --full`.
- Data loader, model forward pass, three loss terms, and end-to-end training verification.
- Deterministic 10-epoch minimal run with seed 42, including metrics and figure generation in a headless environment.

## Deterministic minimal-run reference

The following values are a pipeline smoke test, not the paper's 200-epoch performance:

| Item | Value |
|---|---:|
| Training batteries | 2 |
| Test batteries | 1 |
| Training pairs | 579 |
| Validation pairs | 145 |
| Test pairs | 354 |
| Epochs | 10 |
| MAE | 0.0702587 |
| MAPE | 7.68917% |
| MSE | 0.00811384 |
| RMSE | 0.0900768 |
| Random seed | 42 |

The run can be regenerated after data preparation:

```powershell
python scripts/check_setup.py
python scripts/run_minimal_training.py
```

Outputs are written to `results/minimal_run/` and are intentionally ignored by Git.

## Complete repeated experiment

The current six-condition, 60-run summary is documented in [`docs/完整60次实验报告.md`](docs/完整60次实验报告.md). Lightweight machine-readable summaries are included under `docs/results/`; checkpoints, full prediction arrays, and per-run figures are excluded to keep the repository small.

Neural-network training is stochastic. An unfixed-seed rerun should reproduce the workflow and similar aggregate behavior, not every reported decimal. Use `python experiments/run_paper_60.py --seed 42` when deterministic reruns are required.
