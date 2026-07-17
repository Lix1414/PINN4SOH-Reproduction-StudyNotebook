"""PINN4SOH 模块5：学习率调度、训练、验证与最佳模型保存。"""

import copy
from pathlib import Path

import numpy as np
import torch

from module_loader import load_clean_module


model_module = load_clean_module("03_model.py", "train_model")
loss_module = load_clean_module("04_loss.py", "train_loss")
eval_module = load_clean_module("06_eval.py", "train_eval")
PINN = model_module.PINN
device = model_module.device
train_one_epoch = loss_module.train_one_epoch
eval_metrix = eval_module.eval_metrix


class LR_Scheduler:
    """生成线性warmup与余弦衰减学习率。"""

    def __init__(self, optimizer, warmup_epochs, warmup_lr,
                 num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )
        self.lr_schedule = np.concatenate((warmup, cosine))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = optimizer.param_groups[0]["lr"]

    def step(self):
        """更新一次优化器学习率。"""
        if self.iter >= len(self.lr_schedule):
            return self.current_lr
        next_lr = self.current_lr
        for group in self.optimizer.param_groups:
            if self.constant_predictor_lr and group.get("name") == "predictor":
                group["lr"] = self.base_lr
            else:
                group["lr"] = self.lr_schedule[self.iter]
                next_lr = group["lr"]
        self.iter += 1
        self.current_lr = next_lr
        return next_lr

    def get_lr(self):
        """返回最近设置的学习率。"""
        return self.current_lr


def Train(pinn, trainloader, validloader=None, testloader=None,
          epochs=200, warmup_epochs=30, warmup_lr=0.002,
          base_lr=0.01, final_lr=0.0002, lr_F=0.001,
          alpha=0.7, beta=0.2, early_stop_patience=20,
          save_folder=None):
    """执行与原始仓库一致的PINN训练流程。"""
    optimizer1 = torch.optim.Adam(pinn.solution_u.parameters(), lr=warmup_lr)
    optimizer2 = torch.optim.Adam(pinn.dynamical_F.parameters(), lr=lr_F)
    scheduler = LR_Scheduler(
        optimizer1, warmup_epochs, warmup_lr,
        epochs, base_lr, final_lr,
    )
    min_valid_mse = float("inf")
    early_stop_counter = 0
    best_model = None
    history = {
        "epoch": [], "lr": [], "next_lr": [],
        "train_loss1": [], "train_loss2": [], "train_loss3": [],
        "valid_mse": [], "test_mae": [], "test_mape": [], "test_rmse": [],
    }
    for epoch in range(1, epochs + 1):
        early_stop_counter += 1
        used_lr = optimizer1.param_groups[0]["lr"]
        loss1, loss2, loss3 = train_one_epoch(
            pinn, trainloader, optimizer1, optimizer2, alpha, beta,
        )
        next_lr = scheduler.step()
        history["epoch"].append(epoch)
        history["lr"].append(float(used_lr))
        history["next_lr"].append(float(next_lr))
        history["train_loss1"].append(loss1)
        history["train_loss2"].append(loss2)
        history["train_loss3"].append(loss3)
        valid_mse = None
        if validloader is not None:
            valid_mse = pinn.Valid(validloader)
            history["valid_mse"].append(valid_mse)
        if valid_mse is not None and valid_mse < min_valid_mse:
            min_valid_mse = valid_mse
            early_stop_counter = 0
            best_model = {
                "solution_u": copy.deepcopy(pinn.solution_u.state_dict()),
                "dynamical_F": copy.deepcopy(pinn.dynamical_F.state_dict()),
            }
            if testloader is not None:
                true_label, pred_label = pinn.Test(testloader)
                mae, mape, _, rmse = eval_metrix(true_label, pred_label)
                history["test_mae"].append(mae)
                history["test_mape"].append(mape)
                history["test_rmse"].append(rmse)
        if early_stop_patience is not None and early_stop_counter > early_stop_patience:
            break
    if best_model is not None:
        pinn.solution_u.load_state_dict(best_model["solution_u"])
        pinn.dynamical_F.load_state_dict(best_model["dynamical_F"])
    if save_folder is not None and best_model is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        torch.save(best_model, save_folder / "model.pth")
    return history


def main():
    """执行5个epoch的最小训练验证。"""
    dataloader_module = load_clean_module("02_dataloader.py", "train_data")
    train_files, test_files = dataloader_module.split_xjtu_files()
    train_bundle = dataloader_module.build_dataloaders(train_files[:2], 2.0, 128)
    test_bundle = dataloader_module.build_dataloaders(test_files[:1], 2.0, 128)
    pinn = PINN()
    history = Train(
        pinn, train_bundle["train_2"], train_bundle["valid_2"],
        test_bundle["test_3"], epochs=5, warmup_epochs=2,
        warmup_lr=0.001, base_lr=0.005, final_lr=0.0005,
        early_stop_patience=None,
    )
    true_label, pred_label = pinn.Test(test_bundle["test_3"])
    print(eval_metrix(true_label, pred_label))
    print(history)


if __name__ == "__main__":
    main()
