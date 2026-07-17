"""Check whether an extracted repository ZIP is ready to run."""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import XJTU_ROOT


REQUIRED_PACKAGES = [
    "torch", "sklearn", "numpy", "pandas", "matplotlib", "scipy",
]
REQUIRED_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]


def load_data_module():
    """Load the numbered data-loader module without renaming it."""
    path = SRC_ROOT / "02_dataloader.py"
    spec = importlib.util.spec_from_file_location("setup_check_dataloader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load data module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate dependencies and XJTU CSV layout after ZIP extraction."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Require all six XJTU conditions used by the 60-run experiment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []
    warnings = []

    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ is required; found {sys.version.split()[0]}")
    elif sys.version_info[:2] != (3, 12):
        warnings.append(
            f"Python {sys.version.split()[0]} detected; Python 3.12 is the verified version."
        )

    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except Exception as exc:  # pragma: no cover - diagnostic entry point
            errors.append(f"Cannot import {package}: {exc}")

    if not XJTU_ROOT.exists():
        errors.append(
            f"Missing data directory: {XJTU_ROOT}\n"
            "Create data/XJTU data or set PINN4SOH_DATA_ROOT to its parent directory."
        )
        csv_files = []
    else:
        csv_files = sorted(XJTU_ROOT.glob("*.csv"))
        if not csv_files:
            errors.append(f"No CSV files found in: {XJTU_ROOT}")

    counts = {
        batch: sum(batch in path.name for path in csv_files)
        for batch in REQUIRED_BATCHES
    }
    if counts["2C"] < 1:
        errors.append("At least one 2C CSV is required for the quick verification.")
    missing_batches = [batch for batch, count in counts.items() if count == 0]
    if missing_batches:
        message = "Missing conditions for the complete experiment: " + ", ".join(missing_batches)
        if args.full:
            errors.append(message)
        else:
            warnings.append(message)

    if csv_files:
        sample = csv_files[0]
        try:
            raw = pd.read_csv(sample)
            if raw.shape[1] != 17:
                errors.append(
                    f"{sample.name} has {raw.shape[1]} columns; expected 17 before cycle-index insertion."
                )
            if not len(raw):
                errors.append(f"{sample.name} is empty.")
            if raw.columns[-1] != "capacity":
                errors.append(
                    f"The final column of {sample.name} must be 'capacity'; found {raw.columns[-1]!r}."
                )
            data_module = load_data_module()
            processed = data_module.process_one_csv(sample, nominal_capacity=2.0)
            if processed.shape[1] != 18:
                errors.append(
                    f"Processed data has {processed.shape[1]} columns; expected 18 including cycle index."
                )
            if not np.isfinite(processed.to_numpy()).all():
                errors.append(f"Processed data contains NaN or infinity: {sample.name}")
            if "cycle index" in processed:
                cycle_min = float(processed["cycle index"].min())
                cycle_max = float(processed["cycle index"].max())
                if not (np.isclose(cycle_min, -1.0) and np.isclose(cycle_max, 1.0)):
                    errors.append(
                        f"cycle index range is [{cycle_min}, {cycle_max}], expected [-1, 1]."
                    )
        except Exception as exc:
            errors.append(f"Failed to validate {sample.name}: {exc}")

    print(f"Repository: {REPO_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"XJTU data: {XJTU_ROOT}")
    print("CSV counts: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("SETUP_CHECK=FAIL")
        return 1
    print("SETUP_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
