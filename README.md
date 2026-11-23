# PLG-MVS

PLG-MVS / PLG-SGRefiner 的代码、训练与测试配置，以及已完成实验的结果汇总。

本方法在 MonoMVSNet 主干上引入深度优化模块：PGE 编码初始深度、光度置信度、单目深度和边缘先验，
LRB 进行局部细化，SG-GRB 建模全局关系，最后预测残差深度；完整版本在四个阶段使用细化模块。
`MonoMVSNet`、`SGFormerDepthRefiner` 等历史类名保留，以兼容已有 checkpoint。

## 目录

- `models/`：完整 PLG 的 DTU 版本，含必需的主干和先验网络代码。
- `train_dtu.py`、`test_dtu_dypcd.py`：DTU 训练、深度推理和点云融合。
- `experiments/blended_tnt/`：PLG 的 BlendedMVS 微调和 TNT 推理版本。
- `scripts/`：使用环境变量配置路径的运行入口。
- `configs/`：历史训练/测试命令记录和模块参数。
- `results/`：本方法的 DTU 与 TNT 结果汇总。
- `docs/PROVENANCE.md`：代码版本、验证方式和分数来源。

## 已有结果

| 数据集 | Accuracy↓ | Completeness↓ | Overall↓ | F-score↑ |
|---|---:|---:|---:|---:|
| DTU | 0.305817 | 0.252810 | 0.279314 | — |
| TNT Intermediate | — | — | — | 66.40% |

DTU 数值来自现有评估文件，Overall 为 Accuracy 与 Completeness 的均值。
TNT 数值来自用户提供的官方评估邮件，逐场景分数见 `results/tnt_intermediate.json`。
这些是既有实验结果，本次代码上传没有重新训练或重新提交评估。

TNT 的该次提交采用原作者 backbone 与 v4 训练得到的 PLG refiner 组合权重，
构建工具是 `experiments/blended_tnt/make_author_v4_refiner_hybrid.py`。
DTU 与 TNT 两个版本的残差处理不同，应分别使用对应模型权重和入口。

## 准备

使用 Python 3.10，先按本机 CUDA 环境安装兼容的 PyTorch 与 torchvision，再安装其余依赖：

```bash
pip install -r requirements.txt
```

按 `pre_trained_weights/README.md` 准备 Depth Anything V2 与 TEED 的预训练先验权重。
按各数据集的原始协议准备 DTU、BlendedMVS 或 TNT 数据。
本次发布不包含权重、原始数据、深度图或点云大文件。

## 运行

在仓库根目录执行；将示例路径替换为自己的路径。脚本会实际启动训练或推理。

```bash
# DTU 完整 PLG 训练
DTU_ROOT=/path/to/dtu_training bash scripts/train_dtu_plg.sh

# DTU 深度推理与点云融合
DTU_TEST_ROOT=/path/to/dtu_testing PLG_CHECKPOINT=/path/to/finalmodel_14.ckpt \
  bash scripts/test_dtu_plg.sh

# BlendedMVS 微调
BLENDED_ROOT=/path/to/blendedmvs DTU_CHECKPOINT=/absolute/path/to/dtu_model.ckpt \
  bash scripts/finetune_blended_plg.sh

# TNT Intermediate 深度推理与点云融合
TNT_ROOT=/path/to/intermediate PLG_CHECKPOINT=/absolute/path/to/tnt_plg_model.ckpt \
  bash scripts/test_tnt_plg.sh
```

原始命令记录保存在 `configs/`。路径参数已替换为占位符，其他历史参数保留。
DTU 模块的逐阶段深度细化损失权重为 `[0.01, 0.01, 0.02, 0.05]`，
单目一致性项权重为 `0.01`。TNT 微调版本使用自身实现中的归一化细化损失。

## 来源与许可证

PLG-MVS 的基础主干来自 [MonoMVSNet](https://github.com/JianfeiJ/MonoMVSNet)，
并使用 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
及 [TEED](https://github.com/xavysp/TEED) 作为先验网络。
保留原始 `LICENSE` 及源文件中的作者声明；第三方预训练权重遵守各自的使用条款。
