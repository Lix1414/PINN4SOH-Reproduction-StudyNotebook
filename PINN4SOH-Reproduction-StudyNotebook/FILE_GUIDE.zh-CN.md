# 仓库文件说明

本文档说明 `PINN4SOH-Reproduction-StudyNotebook` 仓库中各目录和主要文件的用途，帮助读者快速理解项目结构、代码调用关系和推荐阅读顺序。

## 1. 仓库定位

本仓库是论文 *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis*（Nature Communications，2024）的非官方、学习型复现项目。项目围绕锂离子电池健康状态（SOH）预测，整理了从特征提取、数据预处理、PINN 建模、损失设计、模型训练到结果评估的完整流程。

仓库同时提供两类内容：

- `src/` 中可直接复用、验证的 Python 模块；
- `notebooks/` 中按学习顺序编排的中文 Notebook。

需要注意：仓库不包含第三方电池数据集；完整训练前需按 `DATA.md` 准备数据。

## 2. 目录结构总览

```text
PINN4SOH-Reproduction-StudyNotebook/
├─ assets/          文档和 Notebook 使用的示意图、实验图
├─ docs/            差异分析、验证报告和完整实验报告
├─ experiments/     多轮训练、基线对比和论文规模实验入口
├─ notebooks/       01～06 分阶段中文学习 Notebook
├─ scripts/         面向用户的轻量训练入口
├─ src/             数据、模型、损失、训练、评估等核心实现
├─ DATA.md          数据来源、目录格式和环境变量说明
├─ README.md        英文项目主页
├─ README.zh-CN.md  中文项目主页
├─ RESULTS.md       已完成验证及参考结果
├─ requirements.txt Python 依赖列表
└─ THIRD_PARTY_NOTICES.md  第三方来源与权利说明
```

运行后还可能生成 `data/`、`outputs/`、`results/`、`experiments/results/` 和 `experiments/images/` 等本地目录。这些目录用于存放数据、模型输出和实验结果，通常不纳入 Git。

## 3. 根目录文件

| 文件 | 作用 |
|---|---|
| `README.md` | 英文版项目介绍，包括仓库定位、环境安装、数据准备、快速验证、实验入口和复现边界。 |
| `README.zh-CN.md` | 中文版项目介绍，适合中文读者作为总入口。 |
| `DATA.md` | 说明两条数据路线：使用上游预处理 CSV 复现训练，或从原始数据研究特征提取；同时给出数据目录结构及路径环境变量。 |
| `RESULTS.md` | 汇总当前版本的验证状态、固定随机种子的最小训练参考指标，以及完整重复实验报告的位置。 |
| `requirements.txt` | 项目所需的 Python 第三方依赖，用于通过 `pip install -r requirements.txt` 安装环境。 |
| `THIRD_PARTY_NOTICES.md` | 记录论文、上游代码和数据等第三方材料的来源及使用注意事项。 |

## 4. `src/`：核心实现

### `src/config.py`

统一管理仓库根目录、数据目录、原始 MAT 文件、外部预处理库以及输出图片目录。支持通过以下环境变量覆盖默认路径：

- `PINN4SOH_DATA_ROOT`：数据根目录，其下应包含 `XJTU data/`；
- `PINN4SOH_XJTU_MAT_PATH`：用于特征提取示例的 XJTU MAT 文件；
- `PINN4SOH_PREPROCESSING_LIB`：外部电池数据预处理代码库路径。

其中 `require_path()` 用于在运行前检查必需路径，并在路径不存在时给出明确错误。

### `src/01_extract_features.py`

特征提取模块，用于演示如何从原始电池循环数据构造统计特征，并将结果写入 `outputs/features/`。该实现依据论文及公开资源反向整理，适合学习特征工程流程，但不保证与上游未公开的原始 16 特征提取代码逐值一致。

### `src/02_dataloader.py`

负责训练数据的读取和预处理，主要包括：

- 读取 XJTU 预处理 CSV；
- 使用 3σ 方法处理异常值；
- 将容量转换为 SOH；
- 插入 `cycle index`；
- 对 16 项统计特征和 `cycle index` 进行归一化；
- 将相邻循环构造成时间步样本对；
- 划分训练集、验证集和测试集，并封装为 PyTorch `DataLoader`。

它是后续模型训练的数据入口。

### `src/03_model.py`

定义 PINN 网络结构，主要包含：

- `Sin`：正弦激活函数；
- `MLP`：基础多层感知机；
- `Predictor`：数据驱动预测分支；
- `SolutionU`：用于描述退化演化的解网络；
- `PINN`：组合各子网络、完成前向传播及自动微分计算的主模型；
- `count_parameters()`：统计模型可训练参数量。

### `src/03_model_runner.py`

模型演示辅助入口。它通过数据加载模块取得示例批次，创建 PINN，并用于快速检查模型输入、输出和前向传播是否正常。

### `src/04_loss.py`

定义 PINN 训练使用的损失项。该模块组合数据拟合误差、物理约束误差和单调性约束等目标，并提供损失计算示例。论文描述与公开上游代码在单调性约束上存在差异，本仓库的具体取舍应结合 `docs/差异分析.md` 阅读。

### `src/05_train.py`

实现通用训练循环，主要包含：

- `LR_Scheduler`：根据验证表现调整学习率；
- `Train()`：执行批次训练、验证、测试、历史记录和早停；
- 最优模型状态保存与恢复；
- 各项损失及评价结果的汇总。

该文件主要提供可复用的训练能力，日常运行通常从 `scripts/` 或 `experiments/` 中的入口脚本开始。

### `src/06_eval.py`

负责模型评估和可视化，计算 MAE、MAPE、MSE、RMSE 等回归指标，并绘制预测值与真实值、退化曲线等图片。默认图像输出位置由 `config.py` 管理。

### `src/module_loader.py`

动态加载以数字开头的 Python 文件。由于 `02_dataloader.py`、`03_model.py` 等文件名不能通过普通 `import` 语句直接作为模块名导入，其他脚本使用 `load_clean_module()` 加载它们。

### `src/verify_pipeline.py`

端到端流程检查入口，用于串联数据加载、模型前向传播和损失计算，确认核心模块能够协同运行。它适合在正式训练前做快速验证。

## 5. `notebooks/`：分阶段学习笔记

| Notebook | 学习内容 | 对应核心代码 |
|---|---|---|
| `01_feature_extraction.ipynb` | 原始循环数据与统计特征提取 | `src/01_extract_features.py` |
| `02_data_preprocessing.ipynb` | 异常值处理、归一化、SOH 转换和时间步配对 | `src/02_dataloader.py` |
| `03_forward_propagation.ipynb` | PINN 子网络、激活函数、前向传播和自动微分 | `src/03_model.py` |
| `04_loss_design.ipynb` | 数据损失、物理损失与单调性损失的设计 | `src/04_loss.py` |
| `05_training.ipynb` | 优化器、学习率调度、早停及训练循环 | `src/05_train.py` |
| `06_evaluation.ipynb` | 评价指标、预测结果和退化曲线可视化 | `src/06_eval.py` |

建议按编号顺序学习。Notebook 偏重解释和交互实验，`src/` 偏重可复用实现，两者可对照阅读。

## 6. `scripts/`：轻量运行入口

### `scripts/run_minimal_training.py`

执行固定随机种子的最小端到端训练。默认训练 10 个 epoch，用于确认数据、模型、损失、训练和绘图链路均可运行，不代表论文 200 epoch 的最终性能。结果写入 `results/minimal_run/`。

该脚本适合作为环境准备完成后的第一个训练命令：

```powershell
python scripts/run_minimal_training.py
```

## 7. `experiments/`：正式实验入口

### `experiments/run_50epoch_training.py`

执行 50 epoch 的中等规模训练，用于在最小冒烟实验与完整论文规模实验之间进行阶段性验证，并输出指标、训练历史和图片。

### `experiments/run_XJTU_standard_2C.py`

面向 XJTU 2C 工况的标准重复实验脚本。它支持命令行参数、随机种子、多次运行、单次结果保存和汇总统计，结果写入 `experiments/results/`，图片写入 `experiments/images/`。

### `experiments/run_xjtu_all_baselines.py`

用于运行 XJTU 多工况或多基线组合实验，并对各实验配置的表现进行统一汇总，适合进行横向比较。

### `experiments/run_paper_60.py`

完整 60 次重复实验的总入口，用于复现仓库报告中的多条件统计结果。该脚本运行成本较高，应在最小训练和单条件实验通过后再执行。可通过 `--seed 42` 等参数增强重复运行的确定性。

## 8. `docs/`：报告与差异说明

| 文件或目录 | 作用 |
|---|---|
| `差异分析.md` | 对照论文、上游公开实现和本仓库实现，说明特征提取、归一化、损失公式等已知差异。 |
| `最终验证报告.md` | 记录代码、环境、数据流和核心运行链路的最终验证情况。 |
| `完整60次实验报告.md` | 汇总完整 60 次实验的配置、统计结果和结论。 |
| `cycle_index归一化对训练结果的影响解读报告.md` | 分析是否归一化 `cycle index` 对训练稳定性与评价指标的影响。 |
| `results/cycle_normalized_summary.json` | 当前归一化方案的轻量机器可读汇总。 |
| `results/cycle_unscaled_summary.json` | 早期未缩放 `cycle index` 方案的历史基线汇总。 |

在解释模型指标或比较论文结果前，建议优先阅读 `差异分析.md` 和相关验证报告。

## 9. `assets/`：图片资源

`assets/` 按学习章节保存 README、文档和 Notebook 引用的图片：

- `02_特征提取/`：统计特征、恒流/恒压曲线、容量等特征示意图；
- `03_数据预处理与成对构造/`：相邻时间步样本配对示意图；
- `04_PINN前向传播/`：激活函数比较和模型输出示意图；
- `06_训练循环/`：学习率调度、早停和训练历史图；
- `07_评估与可视化/`：预测对比、退化曲线、MAPE 对比和冒烟测试结果图。

这些图片主要用于讲解和展示，不参与模型计算。

## 10. 核心调用关系

```text
config.py
   ├─ 01_extract_features.py ──> 特征 CSV
   └─ 02_dataloader.py ────────> DataLoader
                                      │
03_model.py ──────────────────────────┤
                                      v
04_loss.py ─────────────────────> 05_train.py
                                      │
                                      v
                                  06_eval.py
                                      │
                                      v
                              指标、JSON 和结果图
```

`scripts/run_minimal_training.py` 与 `experiments/*.py` 是上层运行入口，通过 `module_loader.py` 调用上述核心模块。

## 11. 推荐使用顺序

1. 阅读 `README.zh-CN.md`，了解项目定位和复现边界。
2. 阅读 `DATA.md`，准备 `data/XJTU data/*.csv`。
3. 安装 `requirements.txt` 中的依赖。
4. 按 `01`～`06` 的顺序阅读和运行 Notebook。
5. 分别运行 `src/02_dataloader.py`、`src/03_model.py`、`src/04_loss.py` 和 `src/verify_pipeline.py` 检查核心模块。
6. 运行 `scripts/run_minimal_training.py` 完成最小端到端训练。
7. 阅读 `docs/差异分析.md`，再执行 `experiments/` 下的正式或重复实验。
8. 对照 `RESULTS.md` 和 `docs/` 中的报告解释实验结果。

## 12. 使用注意事项

- 第三方数据集、上游仓库快照、模型检查点及大规模实验输出不应直接提交到本仓库。
- 最小训练用于验证流程是否可运行，不能视为已经复现论文完整性能。
- 神经网络训练具有随机性；未固定随机种子时，应比较总体趋势和统计量，而不是要求每个小数完全一致。
- 原始 16 项特征提取实现未公开，因此本仓库从原始 MAT 数据生成的特征不一定与上游 CSV 完全一致。
- 在复用或分发论文、代码和数据衍生内容前，应阅读 `THIRD_PARTY_NOTICES.md` 并核对相应来源的许可条款。
