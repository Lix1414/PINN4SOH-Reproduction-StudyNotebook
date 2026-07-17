"""PINN4SOH 模块4：数据、PDE与单调性损失。"""

import torch

from module_loader import load_clean_module


model_module = load_clean_module("03_model.py", "loss_model")
PINN = model_module.PINN
device = model_module.device


class AverageMeter:
    """累计标量平均值。"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置累计状态。"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        """更新累计状态。"""
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count


def compute_data_loss(u1, u2, y1, y2, loss_func):
    """计算相邻时刻的数据损失。"""
    return 0.5 * loss_func(u1, y1) + 0.5 * loss_func(u2, y2)


def compute_pde_loss(f1, f2, loss_func):
    """计算相邻时刻的PDE残差损失。"""
    target = torch.zeros_like(f1)
    return 0.5 * loss_func(f1, target) + 0.5 * loss_func(f2, target)


def compute_mono_loss(u1, u2, y1, y2, relu):
    """按原始源码计算相邻时刻方向一致性损失。"""
    return relu(torch.mul(u2 - u1, y1 - y2)).sum()


def compute_total_loss(u1, u2, f1, f2, y1, y2,
                       loss_func, relu, alpha=0.7, beta=0.2):
    """汇总三项损失。"""
    data_loss = compute_data_loss(u1, u2, y1, y2, loss_func)
    pde_loss = compute_pde_loss(f1, f2, loss_func)
    mono_loss = compute_mono_loss(u1, u2, y1, y2, relu)
    total = data_loss + alpha * pde_loss + beta * mono_loss
    return total, data_loss, pde_loss, mono_loss


def train_one_epoch(pinn, dataloader, optimizer1, optimizer2,
                    alpha=0.7, beta=0.2):
    """训练一个epoch并返回三项平均损失。"""
    pinn.train()
    meters = [AverageMeter(), AverageMeter(), AverageMeter()]
    for x1, x2, y1, y2 in dataloader:
        x1, x2 = x1.to(device), x2.to(device)
        y1, y2 = y1.to(device), y2.to(device)
        u1, f1 = pinn(x1)
        u2, f2 = pinn(x2)
        total, data_loss, pde_loss, mono_loss = compute_total_loss(
            u1, u2, f1, f2, y1, y2,
            pinn.loss_func, pinn.relu, alpha, beta,
        )
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total.backward()
        optimizer1.step()
        optimizer2.step()
        for meter, value in zip(meters, [data_loss, pde_loss, mono_loss]):
            meter.update(value.item())
    return tuple(meter.avg for meter in meters)


def main():
    """验证三项损失计算。"""
    pinn = PINN()
    x1 = torch.randn(8, 17, device=device)
    x2 = torch.randn(8, 17, device=device)
    y1 = torch.rand(8, 1, device=device)
    y2 = torch.rand(8, 1, device=device)
    u1, f1 = pinn(x1)
    u2, f2 = pinn(x2)
    losses = compute_total_loss(
        u1, u2, f1, f2, y1, y2,
        pinn.loss_func, pinn.relu,
    )
    print([value.item() for value in losses])


if __name__ == "__main__":
    main()
