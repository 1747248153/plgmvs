# 代码与结果来源

导出日期：2026-11-24。本仓库仅整理完整 PLG 方法及其必要依赖。

## DTU 完整方法

根目录的 `models/monomvsnet.py` 来自本地保存的完整方法历史文件
`MonoMVSNet-main/models/monomvsnet_backup_before_refiner_switch.py`，
保留四阶段细化和 `[0.01, 0.01, 0.02, 0.05]` 的细化损失权重。
`models/depth_refine_sgformer.py` 来自完整模块的兼容版本，
其参数含 `local_branch`、`global_attn` 和 `fuse`。

已在 CPU 上验证完整模型可以严格加载既有 DTU `finalmodel_14.ckpt`：
全部 814 个 state_dict 项匹配，其中 PLG refiner 有 42 项。
训练日志首步的损失按保存的权重重算约为 22.45539，与日志四舍五入后的 22.456 一致。
其余训练/测试与数据读取代码来自本地工作工程。此次整理没有重新训练，也不据此保证任意新环境的重训分数完全一致。

## BlendedMVS / TNT

`experiments/blended_tnt/` 保留本地 PLG 微调/推理快照。
该版本对残差采用深度区间归一化和置信度门控，与 DTU 原始残差公式不同。
TNT v4 提交使用原作者 backbone 和 v4 PLG refiner 组合权重。
对应的微调命令和 TNT 测试命令来自原始运行日志，记录在 `configs/`。
旧测试记录将 Family 与其余场景分开执行；当前脚本的场景列表覆盖全部八个 Intermediate 场景。

## 指标来源

- DTU：`MonoMVSNet-main/eval/dtu_dypcd_sgformer_refine_all/TotalStat_mvsnet_Eval_.txt`，
  原始文件一并保存；Overall 按两项均值计算。
- TNT：用户提供的 `MonoMVSNet RefineAll v4` 官方评估邮件中的八个场景分数，
  并非本次在本地重新计算的官方分数。显示场景分数的均值为 66.4025，官方显示为 66.40。

`configs/source_manifest.json` 保存各源码文件的原始相对位置及 SHA-256，方便追溯。
