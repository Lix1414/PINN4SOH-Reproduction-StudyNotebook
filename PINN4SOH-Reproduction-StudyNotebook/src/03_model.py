"""PINN4SOH 模块3：PINN前向传播与自动微分。"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


device = "cuda" if torch.cuda.is_available() else "cpu"


class Sin(nn.Module):
    """正弦激活函数。"""

    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """使用正弦激活的多层感知机。"""

    def __init__(self, input_dim=17, output_dim=1, layers_num=4,
                 hidden_dim=50, dropout=0.2):
        super().__init__()
        if layers_num < 2:
            raise ValueError("layers_num必须大于等于2")
        layers = []
        for index in range(layers_num):
            if index == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), Sin()])
            elif index == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    Sin(),
                    nn.Dropout(p=dropout),
                ])
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        """初始化线性层权重。"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """将编码向量映射为SOH。"""

    def __init__(self, input_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 32),
            Sin(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class SolutionU(nn.Module):
    """实现特征到SOH的解网络。"""

    def __init__(self):
        super().__init__()
        self.encoder = MLP(17, 32, 3, 60, 0.2)
        self.predictor = Predictor(32)
        self._init()

    def _init(self):
        """初始化全部线性层。"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def get_embedding(self, x):
        """返回编码器输出。"""
        return self.encoder(x)

    def forward(self, x):
        return self.predictor(self.encoder(x))


class PINN(nn.Module):
    """组合SOH解网络与退化动力学网络。"""

    def __init__(self, F_layers_num=3, F_hidden_dim=60):
        super().__init__()
        self.solution_u = SolutionU().to(device)
        self.dynamical_f = MLP(35, 1, F_layers_num, F_hidden_dim, 0.2).to(device)
        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

    @property
    def dynamical_F(self):
        """保留与原始仓库一致的属性名。"""
        return self.dynamical_f

    def forward(self, xt):
        xt.requires_grad_(True)
        x = xt[:, :-1]
        t = xt[:, -1:]
        u = self.solution_u(torch.cat((x, t), dim=1))
        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True)[0]
        dynamics = self.dynamical_f(torch.cat([xt, u, u_x, u_t], dim=1))
        return u, u_t - dynamics

    def predict(self, xt):
        """预测SOH。"""
        return self.solution_u(xt)

    def test(self, dataloader):
        """返回真实标签与预测标签。"""
        self.eval()
        true_values, predictions = [], []
        with torch.no_grad():
            for x1, _, y1, _ in dataloader:
                predictions.append(self.predict(x1.to(device)).cpu().numpy())
                true_values.append(y1.numpy())
        return np.concatenate(true_values), np.concatenate(predictions)

    def valid(self, dataloader):
        """返回验证集MSE。"""
        true_values, predictions = self.test(dataloader)
        return self.loss_func(
            torch.from_numpy(predictions),
            torch.from_numpy(true_values),
        ).item()

    Test = test
    Valid = valid


Solution_u = SolutionU


def count_parameters(model):
    """返回模型可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    """验证模型结构与前向传播。"""
    model = PINN()
    inputs = torch.randn(4, 17, device=device)
    u, residual = model(inputs)
    print(f"solution_u参数: {count_parameters(model.solution_u)}")
    print(f"dynamical_F参数: {count_parameters(model.dynamical_F)}")
    print(f"u形状: {tuple(u.shape)}")
    print(f"残差形状: {tuple(residual.shape)}")


if __name__ == "__main__":
    main()
