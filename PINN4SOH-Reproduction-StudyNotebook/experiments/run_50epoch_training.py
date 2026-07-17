"""运行独立的 50-epoch PINN4SOH 实验并保存全部结果。

本文件只负责组织实验；数据、模型、训练和评估实现均复用 clean_code。
默认设置与 minimal run 相似：2 节训练电池、1 节测试电池、最多 50 epoch。
默认启用 patience=20 的早停，因此实际训练轮数可能少于 50。
"""

import argparse
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


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
CLEAN = REPRO_ROOT / "src"
RESULT_DIR = HERE / "results"
IMAGE_DIR = HERE / "images"

if str(CLEAN) not in sys.path:
    sys.path.insert(0, str(CLEAN))

from module_loader import load_clean_module


data_module = load_clean_module("02_dataloader.py", "test50_data")
model_module = load_clean_module("03_model.py", "test50_model")
train_module = load_clean_module("05_train.py", "test50_train")
eval_module = load_clean_module("06_eval.py", "test50_eval")


def parse_args():
    """读取命令行实验参数。"""
    parser = argparse.ArgumentParser(description="PINN4SOH 50-epoch 独立实验")
    parser.add_argument("--epochs", type=int, default=50, help="最大训练 epoch 数")
    parser.add_argument("--warmup-epochs", type=int, default=8, help="warmup epoch 数")
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="早停 patience；源码使用 counter > patience")
    parser.add_argument("--disable-early-stop", action="store_true",
                        help="关闭早停，保证训练到设定的最大 epoch")
    parser.add_argument("--train-batteries", type=int, default=2,
                        help="minimal 模式使用的训练电池数量")
    parser.add_argument("--test-batteries", type=int, default=1,
                        help="minimal 模式使用的测试电池数量")
    parser.add_argument("--all-batteries", action="store_true",
                        help="使用当前 2C 数据的全部训练和测试电池")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    """固定 Python、NumPy 与 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_history(history):
    """把训练历史转换为可写入 JSON 的普通数值。"""
    return {
        key: [float(value) for value in values]
        for key, values in history.items()
    }


def main():
    """执行实验并把所有产物保存到 test_50epoch 下。"""
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs 必须大于等于 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs 必须满足 0 <= warmup_epochs < epochs")
    if args.train_batteries < 1 or args.test_batteries < 1:
        raise ValueError("训练和测试电池数量必须大于等于 1")

    set_seed(args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    train_files, test_files = data_module.split_xjtu_files()
    if args.all_batteries:
        selected_train = train_files
        selected_test = test_files
    else:
        selected_train = train_files[:args.train_batteries]
        selected_test = test_files[:args.test_batteries]

    if not selected_train or not selected_test:
        raise ValueError("没有选中训练或测试电池，请检查数据目录与命令行参数")

    train_bundle = data_module.build_dataloaders(
        selected_train, nominal_capacity=2.0, batch_size=args.batch_size,
    )
    test_bundle = data_module.build_dataloaders(
        selected_test, nominal_capacity=2.0, batch_size=args.batch_size,
    )

    early_stop_patience = (
        None if args.disable_early_stop else args.early_stop_patience
    )
    model = model_module.PINN()

    print("=" * 70)
    print("PINN4SOH 50-epoch experiment")
    print(f"device: {model_module.device}")
    print(f"最大 epochs: {args.epochs}")
    print(f"early stop patience: {early_stop_patience}")
    print(f"训练电池: {len(selected_train)}，测试电池: {len(selected_test)}")
    print(f"训练样本对: {len(train_bundle['train_2'].dataset)}")
    print(f"验证样本对: {len(train_bundle['valid_2'].dataset)}")
    print(f"测试样本对: {len(test_bundle['test_3'].dataset)}")
    print("训练期间可能暂时没有终端输出，请等待训练完成。")
    print("=" * 70)

    history = train_module.Train(
        pinn=model,
        trainloader=train_bundle["train_2"],
        validloader=train_bundle["valid_2"],
        testloader=test_bundle["test_3"],
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=0.001,
        base_lr=0.005,
        final_lr=0.0005,
        lr_F=0.001,
        alpha=0.7,
        beta=0.2,
        early_stop_patience=early_stop_patience,
        save_folder=RESULT_DIR,
    )

    true_label, pred_label = model.Test(test_bundle["test_3"])
    mae, mape, mse, rmse = eval_module.eval_metrix(true_label, pred_label)
    valid_mse = history["valid_mse"]
    best_epoch = int(np.argmin(valid_mse)) + 1 if valid_mse else None

    metrics = {
        "MAE": float(mae),
        "MAPE_percent": float(mape),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "max_epochs": args.epochs,
        "completed_epochs": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_valid_mse": float(min(valid_mse)) if valid_mse else None,
        "early_stop_patience": early_stop_patience,
        "warmup_epochs": args.warmup_epochs,
        "seed": args.seed,
        "device": model_module.device,
        "train_pairs": len(train_bundle["train_2"].dataset),
        "valid_pairs": len(train_bundle["valid_2"].dataset),
        "test_pairs": len(test_bundle["test_3"].dataset),
        "train_batteries": selected_train,
        "test_batteries": selected_test,
    }

    np.save(RESULT_DIR / "true_label.npy", true_label)
    np.save(RESULT_DIR / "pred_label.npy", pred_label)
    (RESULT_DIR / "history.json").write_text(
        json.dumps(serializable_history(history), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure = eval_module.plot_training_history(
        history, IMAGE_DIR / "training_history_50epoch.png",
    )
    plt.close(figure)
    figure = eval_module.plot_pred_vs_true(
        true_label, pred_label, IMAGE_DIR / "pred_vs_true_50epoch.png",
    )
    plt.close(figure)
    figure = eval_module.plot_degradation_curves(
        true_label, pred_label, IMAGE_DIR / "degradation_curve_50epoch.png",
    )
    plt.close(figure)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"结果目录: {RESULT_DIR}")
    print(f"图片目录: {IMAGE_DIR}")
    return metrics


if __name__ == "__main__":
    main()
