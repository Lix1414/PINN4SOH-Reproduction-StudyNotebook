"""PINN4SOH 模块1：从XJTU原始MAT数据提取16个统计特征。"""

import sys

import numpy as np
import pandas as pd
from scipy import integrate as scipy_integrate
from scipy import stats as scipy_stats

from config import FEATURE_CSV, PREPROCESSING_LIB, XJTU_MAT_PATH, require_path


CC_VOLTAGE_RANGE = [4.0, 4.199]
CV_CURRENT_RANGE = [0.5, 0.1]


def load_battery_class():
    """加载XJTU电池预处理类。"""
    library = require_path(PREPROCESSING_LIB, "电池预处理库")
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    from XJTUBatteryClass import Battery
    return Battery


def compute_features(voltage_cc, current_cc, time_cc,
                     current_cv, voltage_cv, time_cv):
    """计算CC与CV片段的16个统计特征。"""
    features = {
        "voltage mean": np.mean(voltage_cc),
        "voltage std": np.std(voltage_cc),
        "voltage kurtosis": scipy_stats.kurtosis(voltage_cc, fisher=True),
        "voltage skewness": scipy_stats.skew(voltage_cc),
        "CC Q": scipy_integrate.trapezoid(current_cc, time_cc) / 60.0,
        "CC charge time": (time_cc[-1] - time_cc[0]) * 60.0,
    }
    slope_v_per_min, _ = np.polyfit(time_cc, voltage_cc, 1)
    features["voltage slope"] = slope_v_per_min / 60.0
    v_hist, _ = np.histogram(voltage_cc, bins=len(voltage_cc), density=True)
    features["voltage entropy"] = scipy_stats.entropy(v_hist[v_hist > 0])
    features.update({
        "current mean": np.mean(current_cv),
        "current std": np.std(current_cv),
        "current kurtosis": scipy_stats.kurtosis(current_cv, fisher=True),
        "current skewness": scipy_stats.skew(current_cv),
        "CV Q": scipy_integrate.trapezoid(current_cv, time_cv) / 60.0,
        "CV charge time": (time_cv[-1] - time_cv[0]) * 60.0,
    })
    slope_c_per_min, _ = np.polyfit(time_cv, current_cv, 1)
    features["current slope"] = slope_c_per_min / 60.0
    c_hist, _ = np.histogram(current_cv, bins=len(current_cv), density=True)
    features["current entropy"] = scipy_stats.entropy(c_hist[c_hist > 0])
    return features


def extract_all_features(battery):
    """提取一节电池全部有效循环的特征。"""
    capacities = battery.get_capacity()
    rows = []
    skipped = []
    for cycle in range(1, battery.data.shape[1] + 1):
        try:
            voltage_cc = battery.get_CC_value(cycle, "voltage_V", voltage_range=CC_VOLTAGE_RANGE)
            current_cc = battery.get_CC_value(cycle, "current_A", voltage_range=CC_VOLTAGE_RANGE)
            time_cc = battery.get_CC_value(cycle, "relative_time_min", voltage_range=CC_VOLTAGE_RANGE)
            current_cv = battery.get_CV_value(cycle, "current_A", current_range=CV_CURRENT_RANGE)
            voltage_cv = battery.get_CV_value(cycle, "voltage_V", current_range=CV_CURRENT_RANGE)
            time_cv = battery.get_CV_value(cycle, "relative_time_min", current_range=CV_CURRENT_RANGE)
            if len(voltage_cc) < 3 or len(current_cv) < 3:
                skipped.append((cycle, "insufficient data points"))
                continue
            features = compute_features(
                voltage_cc, current_cc, time_cc,
                current_cv, voltage_cv, time_cv,
            )
            features["capacity"] = capacities[cycle - 1]
            rows.append(features)
        except Exception as error:
            skipped.append((cycle, str(error)))
    columns = [
        "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
        "CC Q", "CC charge time", "voltage slope", "voltage entropy",
        "current mean", "current std", "current kurtosis", "current skewness",
        "CV Q", "CV charge time", "current slope", "current entropy", "capacity",
    ]
    return pd.DataFrame(rows, columns=columns), skipped


def main(mat_path=XJTU_MAT_PATH, output_csv=FEATURE_CSV):
    """执行特征提取并保存CSV。"""
    Battery = load_battery_class()
    battery = Battery(str(require_path(mat_path, "XJTU MAT文件")))
    dataframe, skipped = extract_all_features(battery)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    print(f"输出: {output_csv}")
    print(f"形状: {dataframe.shape}")
    print(f"跳过: {skipped}")
    return dataframe


if __name__ == "__main__":
    main()
