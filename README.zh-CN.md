# PINN4SOH 论文复现与学习笔记

[English](README.md)

这是论文《Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis》（Nature Communications，2024）的一个**非官方、学习型复现项目**。

本仓库将公开的 PINN4SOH 流程整理为更容易阅读和运行的模块，并提供可执行 Notebook、验证记录及面向可复现性的实验脚本。它用于学习与工程验证，不代表原作者官方实现，也不宣称数值结果与论文完全一致。

## 为什么开源这些学习笔记

AI 与电池研究正处在从探索性研究走向工程应用的转型阶段。在这一过程中，难免会出现验证不充分、复现信息不完整或工程约束考虑不足的工作。但这并不意味着该方向本身缺乏价值，更值得关注的问题是：如何让算法与电池机理、数据质量、真实工况和可部署系统建立更扎实的联系。

与此同时，电池智能研究的对象也在从单体电芯逐步扩展到电池包、电池簇乃至完整储能系统。我认为这一演进仍将是未来的重要趋势，而开放、可检查的学习资料有助于让研究思路与实际工程问题更好地衔接。

PINN4SOH 是该方向中具有代表性、也相对适合作为入门案例的工作。我开源这些笔记，是希望帮助代码基础暂时不强的学习者，也能沿着数据流、模型结构、损失函数和实验验证逐步理解完整复现过程。如果这些内容能够帮助你入门、完成实验、发现错误，或者提出更有工程意义的问题，欢迎关注本仓库并参与交流。

## 适合哪些读者

- 希望了解 AI 与电池健康状态估计交叉方向的初学者。
- 希望从论文逐步过渡到可执行代码的学习者。
- 希望核对实现细节、复现边界和实验流程的研究人员或工程人员。
- 愿意交流错误修正、不同理解或工程扩展思路的读者。

## 仓库内容

- 特征提取、数据加载、PINN 前向传播、三项损失、训练和评估的整理版流程。
- 6 个可以从头运行的中文学习 Notebook。
- 最小端到端训练示例。
- XJTU 多次重复实验的启动脚本和实验报告。
- 论文公式、上游源码和本复现之间已知差异的说明。

## 目录结构

```text
src/          整理后的 Python 核心实现
scripts/      验证及训练入口
notebooks/    可执行中文学习 Notebook
docs/         复现说明、使用手册和验证报告
experiments/  实验启动脚本；生成结果不纳入 Git
assets/       精选说明图片
```

## 建议学习顺序

1. 将原论文与上游仓库对照阅读，先理解研究目标和原始实现。
2. 按 `01` 至 `06` 的顺序学习 Notebook：特征提取、数据预处理、前向传播、损失设计、训练和评估。
3. 运行下方快速验证命令，确认环境依赖和数据路径正确。
4. 在解释实验指标之前阅读 `docs/`，特别关注论文、上游源码与本复现之间已记录的差异。
5. 最小训练用于检查数据流与训练链路；只有在评估复现稳定性或报告性能时，才建议运行多次完整实验。

## 从下载的 ZIP 开始运行

当前验证环境使用 Python 3.12。ZIP 中包含代码和学习资料，但不包含第三方电池数据，因此运行前必须先完成数据准备。

1. 解压 ZIP，并在解压后的仓库根目录打开终端。
2. 创建独立环境并安装依赖。

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   如果使用 Windows 命令提示符，请运行 `.venv\Scripts\activate.bat`；Linux 或 macOS 使用 `source .venv/bin/activate`。

3. 获取上游预处理好的 XJTU CSV，并按以下结构放置：

   ```text
   PINN4SOH-Reproduction-StudyNotebook/
   └── data/
       └── XJTU data/
           ├── 2C_battery-1.csv
           ├── 2C_battery-2.csv
           └── ...
   ```

   若目标是复现模型训练，最直接的方式是使用上游 PINN4SOH 仓库提供的预处理 CSV。由于原作者没有公开完全一致的特征提取实现，从原始 MAT 文件重新提取特征属于另一条实验路线。两种路线及数据许可说明见 [DATA.md](DATA.md)。

4. 检查环境和数据目录，然后运行已经验证的核心流程。

   ```powershell
   python scripts/check_setup.py
   python src/02_dataloader.py
   python src/03_model.py
   python src/04_loss.py
   python src/verify_pipeline.py
   ```

5. 运行固定种子的 10 epoch 冒烟实验，或执行完整重复实验。

   ```powershell
   python scripts/run_minimal_training.py
   python experiments/run_paper_60.py
   ```

   最小实验输出到 `results/minimal_run/`；完整重复实验的汇总写入 `experiments/results/`，图片写入 `experiments/images/`。这些生成物默认不纳入 Git。

神经网络训练具有随机性。上述命令能够复现数据流、训练协议和同量级结果，但默认不固定种子的 60 次实验不保证逐位重现报告中的每个小数。若更重视确定性复跑，可使用 `python experiments/run_paper_60.py --seed 42`。

## 复现边界

- 上游仓库未公开原始 16 项特征提取源码；本项目的特征提取是依据论文和公开预处理资源进行的反向实现。
- 数据加载严格沿用公开源码行为：16 项统计特征和 `cycle index` 一起归一化，仅排除 `capacity`（SOH）；整理版只增加 float dtype 与常数列保护以兼容 pandas 3。
- 论文描述的单调性损失公式与公开源码存在差异；整理版实现忠于公开源码行为，并在文档中记录该差异。
- 神经网络训练具有随机性；流程一致不等于指标逐位一致。
- 最小训练只用于验证数据流和训练流程，不能代表已经复现论文完整 200 epoch 性能。

## 原项目与论文

- 上游项目：<https://github.com/wang-fujin/PINN4SOH>
- 论文：Wang 等，《Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis》，*Nature Communications* 15, 4332 (2024)。DOI：<https://doi.org/10.1038/s41467-024-48779-z>

## 双语策略

项目主页提供中英文版本。详细 Notebook 和学习笔记暂时保留中文；代码标识符尽量使用英文。这种方式可以兼顾国际读者与中文学习者，同时避免维护两套重复文档。

## 生成式 AI 使用说明与学习交流

本项目在代码整理、文档编写、一致性检查和复现验证过程中使用了生成式 AI，并结合了作者自编的复现学习 Skill。仓库中的实现选择、技术解释及发布内容均由作者审核。生成式 AI 仅作为辅助工具，不能替代原论文和上游源码；涉及研究结论时，请以原始资料为准。

如果你对复现过程有疑问、发现需要修正的内容，或者希望交流具体学习细节，欢迎通过邮件联系：

- 邮箱：`lixiangl1iiowo@gmail.com`

## 版权与许可

本仓库包含基于第三方论文和公开项目整理或衍生的学习材料。除非相应权利人另有说明，本仓库不会替第三方材料授予许可。详情见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。在确认衍生代码的授权状态之前，不建议为整个仓库直接添加统一的开源许可证。
