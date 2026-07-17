"""PINN4SOH 模块6：回归指标与结果可视化。"""

import math
import os
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
from sklearn import metrics

from config import EVAL_IMAGE_DIR


def eval_metrix(true_label, pred_label):
    """返回MAE、MAPE百分数、MSE与RMSE。"""
    mae = metrics.mean_absolute_error(true_label, pred_label)
    mape = metrics.mean_absolute_percentage_error(true_label, pred_label) * 100
    mse = metrics.mean_squared_error(true_label, pred_label)
    return [mae, mape, mse, np.sqrt(mse)]


def eval_per_battery(true_label, pred_label, min_points=5):
    """按SOH序列重置位置拆分并评估各节电池。"""
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)
    split_points = np.where(np.diff(true_label) > 0.05)[0]
    boundaries = np.concatenate(([0], split_points + 1, [len(true_label)]))
    results = {
        "pred_list": [], "true_list": [],
        "MAE_list": [], "MAPE_list": [], "RMSE_list": [],
    }
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < min_points:
            continue
        true_values = true_label[start:end]
        predictions = pred_label[start:end]
        mae, mape, _, rmse = eval_metrix(true_values, predictions)
        results["pred_list"].append(predictions)
        results["true_list"].append(true_values)
        results["MAE_list"].append(mae)
        results["MAPE_list"].append(mape)
        results["RMSE_list"].append(rmse)
    return results


def print_eval_report(true_label, pred_label):
    """打印整体与逐电池评估报告。"""
    mae, mape, mse, rmse = eval_metrix(true_label, pred_label)
    per_battery = eval_per_battery(true_label, pred_label)
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"识别电池数: {len(per_battery['MAE_list'])}")
    return per_battery


def plot_pred_vs_true(true_label, pred_label, save_path=None):
    """绘制预测SOH与真实SOH散点图。"""
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(true_label, pred_label, alpha=0.4, s=8)
    limits = [min(true_label.min(), pred_label.min()) - 0.02,
              max(true_label.max(), pred_label.max()) + 0.02]
    axis.plot(limits, limits, "r--", linewidth=1.2)
    axis.set(xlabel="True SOH", ylabel="Predicted SOH",
             title="PINN4SOH: Predicted vs True SOH",
             xlim=limits, ylim=limits)
    axis.grid(True, alpha=0.3)
    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def plot_degradation_curves(true_label, pred_label, save_path=None,
                            max_batteries=6):
    """绘制逐电池SOH退化曲线。"""
    results = eval_per_battery(true_label, pred_label)
    count = min(len(results["true_list"]), max_batteries)
    if count == 0:
        raise ValueError("没有满足最小长度要求的电池序列")
    columns = min(3, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.set_visible(False)
            continue
        true_values = results["true_list"][index]
        predictions = results["pred_list"][index]
        axis.plot(true_values, label="True")
        axis.plot(predictions, "r--", label="Predicted")
        axis.set(xlabel="Cycle", ylabel="SOH", title=f"Battery {index + 1}")
        axis.legend()
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def plot_training_history(history, save_path=None):
    """绘制三项损失、学习率和验证MSE。"""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss1"], label="L_data")
    axes[0].plot(epochs, history["train_loss2"], label="L_PDE")
    axes[0].plot(epochs, history["train_loss3"], label="L_mono")
    axes[0].legend()
    axes[1].plot(epochs, history["lr"])
    axes[1].set_title("Learning rate used")
    if history["valid_mse"]:
        axes[2].plot(epochs[:len(history["valid_mse"])], history["valid_mse"])
    axes[2].set_title("Validation MSE")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("Epoch")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def main():
    """用模拟序列验证评估与绘图。"""
    true_label = np.linspace(1.0, 0.8, 100)
    pred_label = true_label + 0.01 * np.sin(np.linspace(0, 8, 100))
    EVAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print_eval_report(true_label, pred_label)
    plot_pred_vs_true(true_label, pred_label, EVAL_IMAGE_DIR / "eval_smoke.png")


if __name__ == "__main__":
    main()
