"""PINN4SOH 模块2：数据清洗、归一化、时间步配对与DataLoader构造。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import XJTU_ROOT, require_path


def _3_sigma(series):
    """返回超出均值正负三倍标准差的行索引。"""
    rule = (series.mean() - 3 * series.std() > series) | \
           (series.mean() + 3 * series.std() < series)
    return np.arange(series.shape[0])[rule]


def delete_3_sigma(dataframe):
    """删除含无效值或任一列三倍标准差异常值的行。"""
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    outliers = []
    for column in dataframe.columns:
        outliers.extend(_3_sigma(dataframe[column]))
    return dataframe.drop(list(set(outliers)), axis=0).reset_index(drop=True)


def _normalize_inputs(dataframe, input_columns, method):
    """归一化全部模型输入列，包括公开源码中的 cycle index。"""
    features = dataframe[input_columns]
    if method == "min-max":
        scale = features.max() - features.min()
        safe_scale = scale.mask(scale == 0, 1.0)
        normalized = 2 * (features - features.min()) / safe_scale - 1
        normalized.loc[:, scale == 0] = 0.0
    elif method == "z-score":
        scale = features.std()
        safe_scale = scale.mask(scale == 0, 1.0)
        normalized = (features - features.mean()) / safe_scale
        normalized.loc[:, scale == 0] = 0.0
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    dataframe[input_columns] = normalized.to_numpy()
    return dataframe


def process_one_csv(file_name, nominal_capacity=None, norm_method="min-max"):
    """读取、清洗并归一化一节电池的CSV。"""
    dataframe = pd.read_csv(file_name)
    # float64 避免 pandas 3 将归一化浮点数写回整数列时报错；数值行为与上游一致。
    dataframe.insert(
        dataframe.shape[1] - 1,
        "cycle index",
        np.arange(dataframe.shape[0], dtype=np.float64),
    )
    dataframe = delete_3_sigma(dataframe)
    if nominal_capacity is not None:
        dataframe["capacity"] = dataframe["capacity"] / nominal_capacity
        # 上游使用 df.iloc[:, :-1]：16项统计特征和 cycle index 均参与归一化，
        # 仅最后一列 capacity（SOH）不参与。
        input_columns = [column for column in dataframe.columns if column != "capacity"]
        dataframe = _normalize_inputs(dataframe, input_columns, norm_method)
    return dataframe


def make_time_step_pairs(dataframe):
    """将连续循环构造成相邻时间步样本对。"""
    x = dataframe.iloc[:, :-1].to_numpy()
    y = dataframe.iloc[:, -1].to_numpy()
    return (x[:-1], y[:-1]), (x[1:], y[1:])


def _load_tensors(path_list, nominal_capacity, norm_method):
    """合并多节电池并转换为张量。"""
    if not path_list:
        raise ValueError("电池文件列表为空")
    x1_list, x2_list, y1_list, y2_list = [], [], [], []
    for path in path_list:
        dataframe = process_one_csv(path, nominal_capacity, norm_method)
        (x1, y1), (x2, y2) = make_time_step_pairs(dataframe)
        x1_list.append(x1)
        x2_list.append(x2)
        y1_list.append(y1)
        y2_list.append(y2)
    return (
        torch.from_numpy(np.concatenate(x1_list)).float(),
        torch.from_numpy(np.concatenate(x2_list)).float(),
        torch.from_numpy(np.concatenate(y1_list)).float().view(-1, 1),
        torch.from_numpy(np.concatenate(y2_list)).float().view(-1, 1),
    )


def _loader(tensors, batch_size, shuffle):
    """将四个张量封装成DataLoader。"""
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def build_dataloaders(path_list, nominal_capacity, batch_size=256,
                      norm_method="min-max"):
    """返回与原始仓库一致的三种DataLoader组合。"""
    x1, x2, y1, y2 = _load_tensors(path_list, nominal_capacity, norm_method)
    split = int(x1.shape[0] * 0.8)
    first_train = (x1[:split], x2[:split], y1[:split], y2[:split])
    first_test = (x1[split:], x2[split:], y1[split:], y2[split:])
    split_values = train_test_split(*first_train, test_size=0.2, random_state=420)
    train = (split_values[0], split_values[2], split_values[4], split_values[6])
    valid = (split_values[1], split_values[3], split_values[5], split_values[7])
    split_values_2 = train_test_split(x1, x2, y1, y2, test_size=0.2, random_state=420)
    train_2 = (split_values_2[0], split_values_2[2], split_values_2[4], split_values_2[6])
    valid_2 = (split_values_2[1], split_values_2[3], split_values_2[5], split_values_2[7])
    return {
        "train": _loader(train, batch_size, True),
        "valid": _loader(valid, batch_size, True),
        "test": _loader(first_test, batch_size, False),
        "train_2": _loader(train_2, batch_size, True),
        "valid_2": _loader(valid_2, batch_size, True),
        "test_3": _loader((x1, x2, y1, y2), batch_size, False),
    }


def split_xjtu_files(root=XJTU_ROOT, batch="2C"):
    """按原始XJTU入口规则划分训练电池和测试电池。"""
    root = require_path(root, "XJTU CSV目录")
    filenames = sorted(path.name for path in Path(root).iterdir() if batch in path.name)
    train_files = [str(Path(root) / name) for name in filenames if "4" not in name and "8" not in name]
    test_files = [str(Path(root) / name) for name in filenames if "4" in name or "8" in name]
    return train_files, test_files


def main():
    """验证XJTU 2C数据加载流程。"""
    train_files, test_files = split_xjtu_files()
    train_bundle = build_dataloaders(train_files, 2.0)
    test_bundle = build_dataloaders(test_files, 2.0)
    print(f"训练电池: {len(train_files)}")
    print(f"测试电池: {len(test_files)}")
    print(f"训练样本对: {len(train_bundle['train_2'].dataset)}")
    print(f"验证样本对: {len(train_bundle['valid_2'].dataset)}")
    print(f"测试样本对: {len(test_bundle['test_3'].dataset)}")


if __name__ == "__main__":
    main()
