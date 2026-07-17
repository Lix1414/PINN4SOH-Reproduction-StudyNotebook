"""按原仓库标准连续运行 10 次 XJTU 2C 实验并汇总结果。"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# 某些受限 Windows 终端不注入 WINDIR，Matplotlib 扫描字体时仍需要它。
if sys.platform == "win32":
    os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
CLEAN = REPRO_ROOT / "src"
RESULT_ROOT = HERE / "results"
IMAGE_ROOT = HERE / "images"

if str(CLEAN) not in sys.path:
    sys.path.insert(0, str(CLEAN))

from module_loader import load_clean_module

data_module = load_clean_module("02_dataloader.py", "xjtu_standard_data")
model_module = load_clean_module("03_model.py", "xjtu_standard_model")
train_module = load_clean_module("05_train.py", "xjtu_standard_train")
eval_module = load_clean_module("06_eval.py", "xjtu_standard_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="PINN4SOH XJTU 2C 标准实验")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-lr", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--final-lr", type=float, default=0.0002)
    parser.add_argument("--lr-F", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=10,
                        help="独立重复实验次数；原文每个工况重复 10 次")
    parser.add_argument("--seed", type=int, default=None,
                        help="可选基础种子；默认不固定，与原始入口一致")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_history(history):
    return {key: [float(value) for value in values] for key, values in history.items()}


def run_experiment(args, experiment_number, experiment_seed):
    """运行一次独立实验，并写入自己的 ExperimentN 目录。"""
    if experiment_seed is not None:
        set_seed(experiment_seed)
    result_dir = RESULT_ROOT / f"Experiment{experiment_number}"
    image_dir = IMAGE_ROOT / f"Experiment{experiment_number}"
    result_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    # 原始 main_XJTU.py 的规则：2C 编号 4、8 测试，其余编号训练。
    train_files, test_files = data_module.split_xjtu_files(batch="2C")
    train_bundle = data_module.build_dataloaders(
        train_files, nominal_capacity=2.0, batch_size=args.batch_size,
    )
    test_bundle = data_module.build_dataloaders(
        test_files, nominal_capacity=2.0, batch_size=args.batch_size,
    )
    patience = None if args.disable_early_stop else args.early_stop_patience
    model = model_module.PINN()

    print("=" * 70)
    print(f"PINN4SOH XJTU standard 2C - Experiment {experiment_number}/{args.repeats}")
    print(f"device: {model_module.device}")
    print(f"最大 epochs: {args.epochs}; warmup: {args.warmup_epochs}")
    print(f"early stop patience: {patience}; batch size: {args.batch_size}")
    print(f"训练电池: {len(train_files)}; 测试电池: {len(test_files)}")
    print(f"训练/验证/测试样本对: {len(train_bundle['train_2'].dataset)}/"
          f"{len(train_bundle['valid_2'].dataset)}/{len(test_bundle['test_3'].dataset)}")
    print("=" * 70)

    history = train_module.Train(
        pinn=model,
        trainloader=train_bundle["train_2"],
        validloader=train_bundle["valid_2"],
        testloader=test_bundle["test_3"],
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        base_lr=args.lr,
        final_lr=args.final_lr,
        lr_F=args.lr_F,
        alpha=args.alpha,
        beta=args.beta,
        early_stop_patience=patience,
        save_folder=result_dir,
    )

    true_label, pred_label = model.Test(test_bundle["test_3"])
    mae, mape, mse, rmse = eval_module.eval_metrix(true_label, pred_label)
    valid_mse = history["valid_mse"]
    best_epoch = int(np.argmin(valid_mse)) + 1 if valid_mse else None
    metrics = {
        "MAE": float(mae), "MAPE_percent": float(mape),
        "MSE": float(mse), "RMSE": float(rmse),
        "max_epochs": args.epochs, "completed_epochs": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_valid_mse": float(min(valid_mse)) if valid_mse else None,
        "early_stop_patience": patience, "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size, "warmup_lr": args.warmup_lr,
        "lr": args.lr, "final_lr": args.final_lr, "lr_F": args.lr_F,
        "alpha": args.alpha, "beta": args.beta,
        "experiment": experiment_number, "seed": experiment_seed,
        "device": model_module.device,
        "train_pairs": len(train_bundle["train_2"].dataset),
        "valid_pairs": len(train_bundle["valid_2"].dataset),
        "test_pairs": len(test_bundle["test_3"].dataset),
        "train_batteries": train_files, "test_batteries": test_files,
    }

    np.save(result_dir / "true_label.npy", true_label)
    np.save(result_dir / "pred_label.npy", pred_label)
    (result_dir / "history.json").write_text(
        json.dumps(serializable_history(history), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    plots = [
        (eval_module.plot_training_history, (history,), "training_history_XJTU_standard_2C.png"),
        (eval_module.plot_pred_vs_true, (true_label, pred_label), "pred_vs_true_XJTU_standard_2C.png"),
        (eval_module.plot_degradation_curves, (true_label, pred_label), "degradation_curve_XJTU_standard_2C.png"),
    ]
    for plot_function, values, filename in plots:
        figure = plot_function(*values, image_dir / filename)
        plt.close(figure)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"结果目录: {result_dir}")
    print(f"图片目录: {image_dir}")
    return metrics


def build_summary(experiments, args):
    """计算 10 次实验核心指标的均值和样本标准差。"""
    metric_names = ["MAE", "MAPE_percent", "MSE", "RMSE"]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([item[name] for item in experiments], dtype=float)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "batch": "2C", "repeats": args.repeats,
        "random_seed_policy": (
            "unfixed" if args.seed is None else f"base_seed_plus_index ({args.seed})"
        ),
        "aggregate": aggregate, "experiments": experiments,
    }


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs 必须大于等于 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs 必须满足 0 <= warmup_epochs < epochs")
    if args.batch_size < 1 or args.repeats < 1:
        raise ValueError("--batch-size 和 --repeats 必须大于等于 1")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = []
    for index in range(args.repeats):
        experiment_seed = None if args.seed is None else args.seed + index
        experiments.append(run_experiment(args, index + 1, experiment_seed))

    summary = build_summary(experiments, args)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("=" * 70)
    print("全部重复实验完成，汇总结果：")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"汇总文件: {RESULT_ROOT / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
