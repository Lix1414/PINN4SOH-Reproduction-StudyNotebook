"""执行可复现的10-epoch最小训练验证。"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pinn4soh-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import torch


REPRO = Path(__file__).resolve().parents[1]
SRC = REPRO / "src"
sys.path.insert(0, str(SRC))
from module_loader import load_clean_module


data_module = load_clean_module("02_dataloader.py", "minimal_data")
model_module = load_clean_module("03_model.py", "minimal_model")
train_module = load_clean_module("05_train.py", "minimal_train")
eval_module = load_clean_module("06_eval.py", "minimal_eval")


def set_seed(seed=42):
    """设置Python、NumPy与PyTorch随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_history(history):
    """将训练历史转换为JSON可序列化字典。"""
    return {
        key: [float(value) for value in values]
        for key, values in history.items()
    }


def main(epochs=10):
    """训练小样本模型并保存曲线、预测和指标。"""
    set_seed(42)
    output = REPRO / "results" / "minimal_run"
    train_image = output / "figures" / "training"
    eval_image = output / "figures" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    train_image.mkdir(parents=True, exist_ok=True)
    eval_image.mkdir(parents=True, exist_ok=True)
    train_files, test_files = data_module.split_xjtu_files()
    selected_train = train_files[:2]
    selected_test = test_files[:1]
    train_bundle = data_module.build_dataloaders(selected_train, 2.0, 128)
    test_bundle = data_module.build_dataloaders(selected_test, 2.0, 128)
    model = model_module.PINN()
    history = train_module.Train(
        model,
        train_bundle["train_2"],
        train_bundle["valid_2"],
        test_bundle["test_3"],
        epochs=epochs,
        warmup_epochs=2,
        warmup_lr=0.001,
        base_lr=0.005,
        final_lr=0.0005,
        lr_F=0.001,
        alpha=0.7,
        beta=0.2,
        early_stop_patience=None,
        save_folder=output,
    )
    true_label, pred_label = model.Test(test_bundle["test_3"])
    mae, mape, mse, rmse = eval_module.eval_metrix(true_label, pred_label)
    metrics = {
        "MAE": float(mae),
        "MAPE_percent": float(mape),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "train_batteries": selected_train,
        "test_batteries": selected_test,
        "train_pairs": len(train_bundle["train_2"].dataset),
        "valid_pairs": len(train_bundle["valid_2"].dataset),
        "test_pairs": len(test_bundle["test_3"].dataset),
        "epochs": len(history["epoch"]),
        "seed": 42,
    }
    np.save(output / "true_label.npy", true_label)
    np.save(output / "pred_label.npy", pred_label)
    (output / "history.json").write_text(
        json.dumps(serializable_history(history), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure = eval_module.plot_training_history(
        history, train_image / "minimal_training_history.png",
    )
    plt.close(figure)
    figure = eval_module.plot_pred_vs_true(
        true_label, pred_label, eval_image / "minimal_pred_vs_true.png",
    )
    plt.close(figure)
    figure = eval_module.plot_degradation_curves(
        true_label, pred_label, eval_image / "minimal_degradation_curve.png",
    )
    plt.close(figure)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
