"""PINN4SOH 项目路径配置。"""

import os
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPRO_ROOT
DATA_ROOT = Path(os.getenv("PINN4SOH_DATA_ROOT", REPRO_ROOT / "data"))
XJTU_ROOT = DATA_ROOT / "XJTU data"

PREPROCESSING_LIB = Path(os.getenv(
    "PINN4SOH_PREPROCESSING_LIB",
    REPRO_ROOT / "vendor" / "Battery-dataset-preprocessing-code-library",
))
XJTU_MAT_PATH = Path(os.getenv(
    "PINN4SOH_XJTU_MAT_PATH",
    DATA_ROOT / "raw" / "XJTU" / "Batch-1" / "2C_battery-1.mat",
))

FEATURE_DIR = REPRO_ROOT / "outputs" / "features"
FEATURE_CSV = FEATURE_DIR / "2C_battery-1_features.csv"
MODEL_IMAGE_DIR = REPRO_ROOT / "outputs" / "figures" / "model"
TRAIN_IMAGE_DIR = REPRO_ROOT / "outputs" / "figures" / "training"
EVAL_IMAGE_DIR = REPRO_ROOT / "outputs" / "figures" / "evaluation"


def require_path(path, label):
    """检查所需路径并返回 Path。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在: {path}")
    return path
