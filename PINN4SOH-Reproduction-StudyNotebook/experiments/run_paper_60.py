"""运行 XJTU 六种工况，每种重复十次，并保存论文统计所需产物。"""

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
CLEAN_ROOT = REPRO_ROOT / "src"
RESULT_ROOT = HERE / "results"
IMAGE_ROOT = HERE / "images"
BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
METRIC_NAMES = ["MAE", "MAPE_percent", "MSE", "RMSE"]

if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from module_loader import load_clean_module

data_module = load_clean_module("02_dataloader.py", "paper_all_data")
model_module = load_clean_module("03_model.py", "paper_all_model")
train_module = load_clean_module("05_train.py", "paper_all_train")
eval_module = load_clean_module("06_eval.py", "paper_all_eval")


def parse_args(argv=None):
    """读取完整实验的命令行参数。"""
    parser = argparse.ArgumentParser(description="PINN4SOH XJTU 六工况论文统计实验")
    parser.add_argument("--batches", nargs="+", choices=BATCHES, default=BATCHES)
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
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def validate_args(args):
    """检查实验参数是否可用于训练。"""
    if args.epochs < 1 or args.batch_size < 1 or args.repeats < 1:
        raise ValueError("epochs、batch-size 和 repeats 必须大于等于 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs 必须满足 0 <= warmup_epochs < epochs")
    if args.early_stop_patience < 0:
        raise ValueError("early-stop-patience 必须大于等于 0")


def set_seed(seed):
    """为确定性调试设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_json_history(history):
    """把训练历史转换为 JSON 可序列化数值。"""
    return {key: [float(value) for value in values] for key, values in history.items()}


def aggregate_metrics(experiments):
    """计算重复实验指标的均值、样本标准差和范围。"""
    aggregate = {}
    for name in METRIC_NAMES:
        values = np.asarray([item[name] for item in experiments], dtype=float)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


def per_battery_metrics(test_files, true_label, pred_label):
    """根据每个测试 CSV 的清洗后长度计算逐电池指标。"""
    reports = []
    offset = 0
    for file_name in test_files:
        frame = data_module.process_one_csv(file_name, nominal_capacity=2.0)
        pair_count = max(len(frame) - 1, 0)
        end = offset + pair_count
        truth = true_label[offset:end]
        prediction = pred_label[offset:end]
        mae, mape, mse, rmse = eval_module.eval_metrix(truth, prediction)
        reports.append({
            "file": file_name, "pair_count": pair_count,
            "MAE": float(mae), "MAPE_percent": float(mape),
            "MSE": float(mse), "RMSE": float(rmse),
        })
        offset = end
    if offset != len(true_label):
        raise RuntimeError("逐电池样本数与测试标签总数不一致")
    return reports


def save_plots(history, true_label, pred_label, image_dir):
    """保存训练历史、散点预测和逐电池退化曲线。"""
    plots = [
        (eval_module.plot_training_history, (history,), "training_history.png"),
        (eval_module.plot_pred_vs_true, (true_label, pred_label), "pred_vs_true.png"),
        (eval_module.plot_degradation_curves, (true_label, pred_label), "degradation_curve.png"),
    ]
    for function, values, file_name in plots:
        figure = function(*values, image_dir / file_name)
        plt.close(figure)


def run_experiment(args, batch, experiment_number, experiment_seed):
    """运行一个工况的一次独立训练与测试。"""
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
    model = model_module.PINN()
    patience = None if args.disable_early_stop else args.early_stop_patience
    started = time.perf_counter()
    print("=" * 72, flush=True)
    print(f"{batch} | Experiment {experiment_number}/{args.repeats} | "
          f"训练/测试电池 {len(train_files)}/{len(test_files)}", flush=True)

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
        "best_epoch": int(np.argmin(valid_mse)) + 1,
        "best_valid_mse": float(min(valid_mse)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "early_stop_patience": patience, "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size, "warmup_lr": args.warmup_lr,
        "lr": args.lr, "final_lr": args.final_lr, "lr_F": args.lr_F,
        "alpha": args.alpha, "beta": args.beta, "device": model_module.device,
        "train_pairs": len(train_bundle["train_2"].dataset),
        "valid_pairs": len(train_bundle["valid_2"].dataset),
        "test_pairs": len(test_bundle["test_3"].dataset),
        "train_batteries": train_files, "test_batteries": test_files,
        "per_battery": per_battery_metrics(test_files, true_label, pred_label),
    }
    np.save(result_dir / "true_label.npy", true_label)
    np.save(result_dir / "pred_label.npy", pred_label)
    (result_dir / "history.json").write_text(
        json.dumps(to_json_history(history), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    save_plots(history, true_label, pred_label, image_dir)
    print(f"完成 | MAPE={mape:.4f}% | RMSE={rmse:.6f} | "
          f"epoch={len(history['epoch'])} | {metrics['elapsed_seconds']:.1f}s", flush=True)
    return metrics


def save_comparison_plot(batch_summaries):
    """保存六种工况的 MAPE 均值与标准差对比图。"""
    names = list(batch_summaries)
    means = [batch_summaries[name]["aggregate"]["MAPE_percent"]["mean"] for name in names]
    errors = [batch_summaries[name]["aggregate"]["MAPE_percent"]["std"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(names, means, yerr=errors, capsize=5)
    axis.set(xlabel="XJTU condition", ylabel="MAPE (%)",
             title="PINN4SOH XJTU: mean MAPE over repeated experiments")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(IMAGE_ROOT / "MAPE_comparison_all.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_suite(args):
    """执行选定工况的全部重复实验并生成分组与总汇总。"""
    validate_args(args)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_experiments = []
    batch_summaries = {}
    for batch_index, batch in enumerate(args.batches):
        experiments = []
        for repeat_index in range(args.repeats):
            seed = None if args.seed is None else args.seed + batch_index * args.repeats + repeat_index
            result = run_experiment(args, batch, repeat_index + 1, seed)
            experiments.append(result)
            all_experiments.append(result)
        batch_summary = {
            "batch": batch, "repeats": args.repeats,
            "aggregate": aggregate_metrics(experiments), "experiments": experiments,
        }
        batch_summaries[batch] = batch_summary
        (RESULT_ROOT / batch / "summary.json").write_text(
            json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    overall = {
        "protocol": "XJTU six conditions, ten stochastic repeats per condition",
        "batches": args.batches, "repeats_per_batch": args.repeats,
        "total_experiments": len(all_experiments),
        "random_seed_policy": "unfixed" if args.seed is None else "deterministic_sequence",
        "elapsed_seconds": float(time.perf_counter() - started),
        "per_batch": {name: value["aggregate"] for name, value in batch_summaries.items()},
        "all_experiments_aggregate": aggregate_metrics(all_experiments),
    }
    (RESULT_ROOT / "summary_all.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    save_comparison_plot(batch_summaries)
    print("=" * 72, flush=True)
    print(f"全部完成：{len(all_experiments)} 次，耗时 {overall['elapsed_seconds'] / 60:.2f} 分钟", flush=True)
    print(f"汇总文件：{RESULT_ROOT / 'summary_all.json'}", flush=True)
    return overall


def main(argv=None):
    """运行命令行指定的 XJTU 实验套件。"""
    return run_suite(parse_args(argv))


if __name__ == "__main__":
    main()
