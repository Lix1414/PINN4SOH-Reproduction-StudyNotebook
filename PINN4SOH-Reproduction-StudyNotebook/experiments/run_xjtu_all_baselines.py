"""运行原文 XJTU 六种工况，每种默认重复 10 次，并汇总结果。"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

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
BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]

if str(CLEAN) not in sys.path:
    sys.path.insert(0, str(CLEAN))

from module_loader import load_clean_module

data_module = load_clean_module("02_dataloader.py", "xjtu_all_data")
model_module = load_clean_module("03_model.py", "xjtu_all_model")
train_module = load_clean_module("05_train.py", "xjtu_all_train")
eval_module = load_clean_module("06_eval.py", "xjtu_all_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="PINN4SOH XJTU 六工况完整实验")
    parser.add_argument("--batches", nargs="+", choices=BATCHES, default=BATCHES,
                        help="默认运行全部六种工况")
    parser.add_argument("--repeats", type=int, default=10)
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
    parser.add_argument("--seed", type=int, default=None,
                        help="默认不固定；指定后按工况和重复编号生成不同种子")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_history(history):
    return {key: [float(value) for value in values] for key, values in history.items()}


def summarize(experiments):
    summary = {}
    for name in ["MAE", "MAPE_percent", "MSE", "RMSE"]:
        values = np.asarray([item[name] for item in experiments], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def run_one(args, batch, experiment_number, experiment_seed):
    if experiment_seed is not None:
        set_seed(experiment_seed)
    result_dir = RESULT_ROOT / batch / f"Experiment{experiment_number}"
    image_dir = IMAGE_ROOT / batch / f"Experiment{experiment_number}"
    result_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = data_module.split_xjtu_files(batch=batch)
    train_bundle = data_module.build_dataloaders(
        train_files, nominal_capacity=2.0, batch_size=args.batch_size,
    )
    test_bundle = data_module.build_dataloaders(
        test_files, nominal_capacity=2.0, batch_size=args.batch_size,
    )
    patience = None if args.disable_early_stop else args.early_stop_patience
    model = model_module.PINN()
    started = time.perf_counter()
    print("=" * 72)
    print(f"工况 {batch} | Experiment {experiment_number}/{args.repeats} | "
          f"训练/测试电池 {len(train_files)}/{len(test_files)}")
    print(f"训练/验证/测试样本对 {len(train_bundle['train_2'].dataset)}/"
          f"{len(train_bundle['valid_2'].dataset)}/{len(test_bundle['test_3'].dataset)}")

    history = train_module.Train(
        pinn=model, trainloader=train_bundle["train_2"],
        validloader=train_bundle["valid_2"], testloader=test_bundle["test_3"],
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr, base_lr=args.lr, final_lr=args.final_lr,
        lr_F=args.lr_F, alpha=args.alpha, beta=args.beta,
        early_stop_patience=patience, save_folder=result_dir,
    )
    true_label, pred_label = model.Test(test_bundle["test_3"])
    mae, mape, mse, rmse = eval_module.eval_metrix(true_label, pred_label)
    valid_mse = history["valid_mse"]
    metrics = {
        "batch": batch, "experiment": experiment_number, "seed": experiment_seed,
        "MAE": float(mae), "MAPE_percent": float(mape),
        "MSE": float(mse), "RMSE": float(rmse),
        "max_epochs": args.epochs, "completed_epochs": len(history["epoch"]),
        "best_epoch": int(np.argmin(valid_mse)) + 1 if valid_mse else None,
        "best_valid_mse": float(min(valid_mse)) if valid_mse else None,
        "elapsed_seconds": float(time.perf_counter() - started),
        "early_stop_patience": patience, "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size, "warmup_lr": args.warmup_lr,
        "lr": args.lr, "final_lr": args.final_lr, "lr_F": args.lr_F,
        "alpha": args.alpha, "beta": args.beta, "device": model_module.device,
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
        (eval_module.plot_training_history, (history,), "training_history.png"),
        (eval_module.plot_pred_vs_true, (true_label, pred_label), "pred_vs_true.png"),
        (eval_module.plot_degradation_curves, (true_label, pred_label), "degradation_curve.png"),
    ]
    for function, values, filename in plots:
        figure = function(*values, image_dir / filename)
        plt.close(figure)
    print(f"完成：MAPE={mape:.4f}%, RMSE={rmse:.6f}, epoch={len(history['epoch'])}, "
          f"耗时={metrics['elapsed_seconds']:.1f}s")
    return metrics


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.repeats < 1:
        raise ValueError("epochs、batch-size 和 repeats 必须大于等于 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs 必须满足 0 <= warmup_epochs < epochs")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_experiments = []
    batch_summaries = {}
    for batch_index, batch in enumerate(args.batches):
        experiments = []
        for repeat_index in range(args.repeats):
            seed = None if args.seed is None else args.seed + batch_index * args.repeats + repeat_index
            result = run_one(args, batch, repeat_index + 1, seed)
            experiments.append(result)
            all_experiments.append(result)
        batch_summary = {
            "batch": batch, "repeats": args.repeats,
            "aggregate": summarize(experiments), "experiments": experiments,
        }
        batch_summaries[batch] = batch_summary
        (RESULT_ROOT / batch / "summary.json").write_text(
            json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    total_seconds = time.perf_counter() - started
    overall = {
        "batches": args.batches, "repeats_per_batch": args.repeats,
        "total_experiments": len(all_experiments),
        "random_seed_policy": "unfixed" if args.seed is None else "deterministic_sequence",
        "elapsed_seconds": float(total_seconds),
        "per_batch": {batch: value["aggregate"] for batch, value in batch_summaries.items()},
        "all_experiments_aggregate": summarize(all_experiments),
    }
    (RESULT_ROOT / "summary_all.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("=" * 72)
    print(f"全部完成：{len(all_experiments)} 次实验，耗时 {total_seconds / 60:.2f} 分钟")
    print(f"总汇总：{RESULT_ROOT / 'summary_all.json'}")
    return overall


if __name__ == "__main__":
    main()
