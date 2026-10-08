
# 牙齿3D分割快速入门

本快速入门介绍了牙齿多类别3D分割的首个可运行集成流程。

> **当前状态（2026-10-08）**：生产配置 = **Dataset1001（去 x 轴镜像）5 折集成**（`tooth_predict.py` 推荐模型，ID assigner 默认 on 作安全网），66-case holdout **0.9281**、32/32 类 ≥0.80（弱类最低 0.856）。上一代 Dataset1000 + assigner（0.9094）留作对照/回退。终评四列对比：`docs/superpowers/plans/2026-09-20-dataset1000-optimization-results.md`。L/R ID 错误背景与三轨方案：`docs/superpowers/plans/2026-09-20-dataset1000-optimization.md`。

## 1）分析spacing（推荐用于目标spacing）

在`imagesTr`上运行spacing分析，获得中位数/p10/p90及推荐的目标spacing：

```bash
python dataset_conversion/analyze_tooth_spacing.py --images_dir RAW/images --image_suffix .mha --save_report .\spacing_stats.yaml
```

将`recommended_target_spacing`（或类似的四舍五入值，如`0.4,0.4,0.4`）用于`tooth_3d.py --target_spacing`参数。

> 注：`analyze_tooth_spacing.py` 为 CBIM 项目历史脚本（不在本仓库）；结论已固化——实测原生 spacing 0.3 = 目标 0.3，不重采样（见 §7）。

## 2）分析标签（可选但推荐）

对整个数据集标签进行审核，获得唯一ID及建议的连续映射：

```bash
python dataset_conversion/analyze_tooth_labels.py --labels_dir RAW/labels --label_suffix .mha --save_report .\label_stats.yaml --save_map .\label_map.yaml
```

> 注：`analyze_tooth_labels.py` 同为 CBIM 项目历史脚本（不在本仓库）；结论已固化为 §3 的 `label_map.yaml`。

## 3）准备标签映射

将原始牙齿ID映射为连续训练ID的YAML映射表。

重要规则：原始ID `1..10` 视为非牙齿/背景，需映射为`0`。

示例（`label_map.yaml`）：

```yaml
0: 0
1: 0
2: 0
3: 0
4: 0
5: 0
6: 0
7: 0
8: 0
9: 0
10: 0
11: 1
12: 2
13: 3
14: 4
15: 5
16: 6
17: 7
18: 8
21: 9
22: 10
23: 11
24: 12
25: 13
26: 14
27: 15
28: 16
31: 17
32: 18
33: 19
34: 20
35: 21
36: 22
37: 23
38: 24
41: 25
42: 26
43: 27
44: 28
45: 29
46: 30
47: 31
48: 32
```

## 4）将`.mha`转换为CBIM格式

期望的原始数据布局：

- `RAW/images/*.mha`
- `RAW/labels/*.mha`

命名匹配规则（重要）：

- 默认情况下，`tooth_3d.py`使用`--image_stem_suffix _0000`。
- 例如：图像`ToothFairy2F_001_0000.mha`会匹配标签`ToothFairy2F_001.mha`。
- 如果你的图像和标签stem已经一致，可用`--image_stem_suffix ""`禁用裁剪。
- 输出`case_name`跟随匹配到的标签stem，文件写为`case_name.nii.gz`和`case_name_gt.nii.gz`。

转换命令：

```bash
python dataset_conversion/tooth_3d.py --src_images RAW/images --src_labels RAW/labels --dst_root DATA/tooth_3d --label_map .\label_map.yaml --strict_unmapped --target_spacing 0.4,0.4,0.4
```

可选（推荐）：标签重映射后做前景裁剪（`label > 0`），并设置`z,y,x`方向的上下文：

```bash
python dataset_conversion/tooth_3d.py --src_images /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/imagesTr/ --src_labels /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/labelsTr/ -dst_root /home/share/clr/share/data/CBCT/tooth_fairy2 --label_map .\label_map.yaml --strict_unmapped --target_spacing 0.3,0.3,0.3 --crop_foreground --crop_context 10,30,30
```

裁剪行为说明：

- 裁剪bbox由每例映射标签（`> 0`）计算。
- `--crop_context 10,30,30`为经验默认值（体素单位，`z,y,x`），非唯一最佳。
- 若验证集出现边界漏分/欠分割，可适当增大context（如`15,40,40`）。
- 若验证集质量稳定，可适当减小context以提升速度/节省内存。

等价的显式命令（同默认行为）：

```bash
python dataset_conversion/tooth_3d.py --src_images RAW/images --src_labels RAW/labels --dst_root DATA/tooth_3d --label_map .\label_map.yaml --strict_unmapped --target_spacing 0.4,0.4,0.4 --image_stem_suffix _0000
```

### z/y 轴纠正开关（`--fix_z` / `--fix_y`，默认关闭）

原始 MHA 数组实际按 RPI 顺序存储（头顶在低 z、面部前部在低 y），但元数据声称 identity RAS（发布数据自身的元数据错误，非转换引入）。`--fix_z` / `--fix_y` 可单独或组合使用，在裁剪后把图像/标签数组沿对应轴翻转（spacing/origin/direction 不变）：`--fix_z` 使头顶在高 z；`--fix_y` 使面部前部在高 y；两者同用即真 RAS 存储，查看器正确显示朝向。

```bash
python dataset_conversion/tooth_3d.py --src_images /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/imagesTr/ --src_labels /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/labelsTr/ --dst_root <输出目录> --label_map /home/share/clr/share/data/CBCT/tooth_fairy2/list/label_map.yaml --strict_unmapped --target_spacing 0.3,0.3,0.3 --crop_foreground --crop_context 10,30,30 --fix_z --fix_y
```

已生成的纠正副本：
- 2026-09-03 两者同用 → `/home/share/clr/share/data/CBCT/tooth_fairy2_m`（真 RAS；480 case，case 集与裁剪同 `tooth_fairy2`，数组与对应文件严格 flip 对应；`scripts/verify_zy_flip.py --anatomy` 全量验证通过）。
- 2026-09-04 仅 `--fix_z` → `/home/share/clr/share/data/CBCT/tooth_fairy2_z`（头顶高 z，y 保持原帧；`scripts/verify_zy_flip.py --axes 0 --anatomy` 479/480，唯一例外为已知的 P_192 左右标注问题）。**注意：只翻转单轴必然得到镜像帧**——元数据仍为 identity，真实面部（低 y）会被查看器显示为"朝后"，按面部判定解剖左右时 1–8 会出现在患者左侧（数学上不可避免，非数据错误）；`_z` 仅适用于不涉及左右的场景。需要左右正确的显示请用 `_m` 并从前方（anterior）观察。

注意：在旧帧（`tooth_fairy2`）训练的模型与任何纠正帧数据不兼容，用新数据集训练需重新训练。

输出布局：

- `DATA/tooth_3d/case_xxx.nii.gz`
- `DATA/tooth_3d/case_xxx_gt.nii.gz`
- `DATA/tooth_3d/list/dataset.yaml`

## 5）先生成固定划分（必需）

在任何基于划分的patch索引前，先生成划分：

```bash
python dataset_conversion/tooth_split.py --data_root DATA/tooth_3d --k_fold 5 --seed 0
```

默认使用`holdout_ratio=0.15`，输出`DATA/tooth_3d/list/tooth_split_k5_seed0.yaml`，内容包括：

- 顶层独立`holdout_test`（15%）
- 每折内部`train/val`列表（无折内test）

## 6）可选：预生成训练patch索引

若3D训练启动慢或全量case预加载内存占用大，可生成缓存patch索引：

```bash
python dataset_conversion/tooth_patch_index.py --data_root DATA/tooth_3d --patch_size 64,160,160 --patches_per_case 8 --foreground_fraction 0.875 --foreground_threshold 0.01 --background_max_fg_ratio 0.0
```

推荐与固定划分结合（仅为某一折的train case生成索引）：

```bash
python dataset_conversion/tooth_patch_index.py --data_root DATA/tooth_3d --patch_size 64,160,160 --patches_per_case 8 --foreground_fraction 0.875 --foreground_threshold 0.01 --background_max_fg_ratio 0.0 --split_file DATA/tooth_3d/list/tooth_split_k5_seed0.yaml --fold 0 --k_fold 5 --split_seed 0
```

默认输出：

- `DATA/tooth_3d/list/tooth_patch_index.npz`
- `DATA/tooth_3d/list/tooth_patch_index_meta.yaml`

主要参数含义推荐：

- `--patch_size`：与`training_size`一致的`z,y,x`
- `--patches_per_case`：每个case缓存多少训练patch
- `--foreground_fraction`：前景优先patch比例（如`0.875`表示8个patch中7个优先采样前景，1个采样背景）
- `--foreground_threshold`：patch内前景体素比例阈值
- `--split_file` + `--fold`：仅为该折train case生成patch索引（避免val/test case生成索引）

patch数量建议：

- 保证`train_cases * patches_per_case`远大于`iter_per_epoch * batch_size`
- 默认配置（`iter_per_epoch: 300`），`patches_per_case=8`对每折几百个训练case已足够

## 7）nnUNet 环境与预处理

本项目现以 **nnUNetv2** 训练（不再是 MedFormer）。数据集已组装为 `tooth_3d_semantic/raw/Dataset999_ToothSemantic`（Dataset ID 999），全链路来龙去脉见 `tooth_data_pipeline_plan.md`。

```bash
# 设置 nnUNet 路径变量（raw / preprocessed / results 均在 tooth_3d_semantic/ 下）
source dataset_conversion/setup_nnunet_env.sh

# 规划 + 预处理（已执行；输出在 tooth_3d_semantic/preprocessed/）
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 999
```

plans 关键参数（见 `tooth_3d_semantic/results/Dataset999_ToothSemantic/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json`）：

- `batch_size: 8`（global；4×A10 DDP，每 GPU 2）
- `patch_size: [80, 128, 192]`（z,y,x）
- `spacing: [0.3, 0.3, 0.3]` = 原生 spacing → 不重采样
- 1000 epochs/fold

**RAS 帧重训（Dataset1000_ToothSemanticRAS，2026-09-05 起）**：用真 RAS 纠正帧 `tooth_fairy2_m` 组装新数据集并重新 5 折训练（旧 999 全部保留；splits 与 999 逐字节相同 → 同一 72 case holdout，结果与 0.818 基线可比）。训练用 GPU 1,2,3（4 卡留 0 空闲）。

```bash
# 组装 raw（已执行）
python dataset_conversion/ToothFairy2SemanticRAS.py
# 规划 + 预处理（已执行；自动 plan 的 3d_fullres batch_size=2，已手动改：fold_0 用 18 训完，fold_1–4 用 8，见 §8）
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 1000
```

**去 x 轴镜像重训（Dataset1001_ToothSemanticNoXMirror，2026-09-24 起）**：根因 = nnUNet 默认 x 轴镜像数据增强不做标签重映射 → 颌弓级 L/R ±8 ID 互换（见上文 plan 文档）。与 1000 同数据同划分（480 case、无剔除；全量审计证实零 GT 错误），仅 trainer 改为 `nnUNetTrainer_onlyMirror01`（只镜像 z/y 轴）。fold_0 门控 PASS（2026-09-30，worst20 +0.259），5 折已全部完成（2026-10），终评 66-case 0.9281 胜出并接替 1000 成为生产模型。

```bash
# 组装 raw（已执行；自 1000 硬链接拷贝，新 ID 1001，不覆盖 999/1000）
bash scripts/assemble_dataset1001.sh
# 规划 + 预处理（已执行）
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 1001
```

## 8）训练与 Holdout 评估

训练单折（Dataset999，内部 4 GPU DDP；**999 时代脚本 `train_fold.sh` / `train_folds_seq.sh` 已于 2026-10-02 删除**，命令留档）：

```bash
bash scripts/train_fold.sh 0        # [已删除]
```

5 折顺序训练：`bash scripts/train_folds_seq.sh` **[已删除]**。

RAS 帧（Dataset1000，2026-09-05 起；**1000 时代脚本，2026-10-02 加 1000 标识归档**）：

```bash
bash scripts/train_fold_ras_1000.sh 0     # 单折；GPU 1,2,3（留 0 空闲）
bash scripts/train_folds_seq_ras_1000.sh  # 5 折顺序，首折失败即停
```

去 x 轴镜像帧（Dataset1001，2026-09-24 起；**5 折已于 2026-10 完成，现为生产模型**）：

```bash
bash scripts/train_fold_noxmirror.sh 1    # 单折；GPU 1,2,3
bash scripts/train_folds_seq_noxmirror.sh # folds 1–4 顺序，首折失败即停
```

- Dataset1001 内部 val（同 split，vs 1000 括号内）：fold_0 **0.8185**（0.650）、fold_1 **0.7694**（0.6705）、fold_2 **0.8122**（0.7347）、fold_3 **0.8066**（0.7083）、fold_4 **0.8042**（0.6772）；均值 **0.8022** vs 0.6881，5/5 折全部提升。

- 脚本带 `--c`：fold 结果目录有 checkpoint 即自动恢复（final→latest→best），没有则警告后正常新训；要强制重训先删 `results/.../fold_<N>/`。已完成的折带 `--c` 重跑是安全空转（重存 final + 重跑内部验证后 exit 0）。
- 日志：`tooth_3d_semantic/logs/train_ras_fold<N>.log` 经 tee 为块缓冲，长时间"静默"属正常；**权威进度看 `results/.../fold_<N>/training_log_*.txt`（最新时间戳那份，每 epoch 落盘带时间戳）**。
- **当前配置（2026-09-09 起）：global batch 8 → per-GPU 3/3/2**，~145–150 s/epoch，~38 h/折；fold_0 已用 global 18 训完（保留，内部 val 0.650）。改 8 的原因：18 时每折总样本 450 万 = 2.25×999 的 200 万（nnUNet 总样本 = 250 迭代 × 1000 epoch × 全局 batch），改 8 与 999 样本预算对齐。
- batch 阶梯（A10 实测，patch 80×128×192）：每卡 8（global 24）首个 forward 即 OOM；每卡 6（global 18）稳定 22402/22731 MiB（98.5%）；每卡上限约 6（~3.3 GB/样本 + ~2.6 GB 固定）。若之后 OOM：把 `preprocessed/Dataset1000_ToothSemanticRAS/nnUNetPlans.json` 的 3d_fullres `batch_size` 改小（如 6 = 2/2/2）后跑同一命令从断点续训。
- **5 折已于 2026-09-15 22:09 全部完成**；内部 val：0.650/0.670/0.735/0.708/0.677（均值 0.688，999 为 0.685）。

Holdout 评估（Dataset999，72 case，5-fold logit 集成，`checkpoint_final`，mirroring+gaussian；**`eval_holdout.py` 已于 2026-10-02 删除**，命令留档）：

```bash
python scripts/eval_holdout.py        # [已删除] 全量；--smoke 为单 case 冒烟
```

输出：`tooth_3d_semantic/eval/holdout_ensemble/{predictions/, per_case_metrics.jsonl, holdout_metrics.json}`。

当前基线（2026-08-26）：72 case raw 均值 fg dice 0.784；69 个非空-GT case 0.818；再剔除 3 个模型侧低分 case（F_049/P_049/P_105）≈ 0.852。3 个低分 case 的分析（已排除 GT 左右标反与图像镜像，根因为缺牙下的 ID 混淆）见 `tooth_data_pipeline_plan.md` §3。

RAS 帧终评（2026-09-20，`scripts/eval_holdout_ras_1000.py`，GPU 1,2,3，19 min）：

```bash
python scripts/eval_holdout_ras_1000.py        # 全量；--smoke 为单 case 冒烟
```

结果：72 case raw 0.7837；69 非空 **0.8177**；66-case（剔 3 低分）0.8446（999 为 0.8524）——与旧模型质量持平（逐例 36 升/33 降，噪声级差异），朝向修正零代价。输出：`tooth_3d_semantic/eval/holdout_ensemble_ras/`。

**1000 + ID assigner（2026-09-23，原生产配置，已被 1001 取代）**：66-case **0.9094**（raw 0.8446 → +0.065）、69=0.899、72=0.8615；上颌 0.8945 / 下颌 0.9085；10 个弱类全 ≥0.80（最低 23/FDI37=0.8378）；零回退（>0.005）。输出：`tooth_3d_semantic/eval/holdout_ensemble_ras_idassigned_final/`。

**1001 去镜像终评（2026-10-08，当前生产配置）**：`scripts/eval_holdout_noxmirror.py`（5 折集成，TTA 仅 z/y）+ `scripts/eval_holdout_general.py` 计指标。**1001-raw 66-case 0.9281 / 69=0.9253 / 72=0.8868**（上颌 0.9237 / 下颌 0.9434）；+ID assigner 后逐位不变（72 例仅 1 次重映射，assigner 仅作安全网）。弱类 10 类全 ≥0.856（最低 4/FDI14=0.856），32/32 类 ≥0.80。四列对比与逐类/逐 case 明细见 `docs/superpowers/plans/2026-09-20-dataset1000-optimization-results.md`；输出：`tooth_3d_semantic/eval/holdout_ensemble_noxmirror{,_raw,_assigned}/`。

## 9）新图像预测

使用去 x 镜像 RAS 帧生产模型（Dataset1001，5 折 logit 集成，TTA 镜像仅 z/y；66-case holdout **0.9281**（裸分，1000 裸分为 0.8446、1000+assigner 为 0.9094），默认另开 ID assigner 作安全网——1001 上仅 1 次重映射、零指标损失）：

```bash
source dataset_conversion/setup_nnunet_env.sh
python inference/tooth_predict.py \
    --model_dir tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres \
    --input /path/to/new_scan_0000.nii.gz \
    --output /path/to/pred.nii.gz
```

说明：

- `--model_dir` 必须是 trainer 父目录（含 `plans.json` 与 `fold_N/`），不是某个 `fold_N` 子目录；
- `--folds` 默认自动检测全部可用折（5 折 logit 集成）；单折用 `--folds 0`；
- 输入文件名须以 `_0000.nii.gz` 结尾（nnUNet 约定，脚本强制校验）；可为整卷（nnUNet 推理时自动重采样与前景裁剪）；输入应为 RAS 朝向（与训练数据 `tooth_fairy2_m` 一致）；
- `--id_assign on/off`（默认 `on`）：确定性 L/R ID 一致性后处理（连通分量 + 中线 + 双孪生守卫，见 plan 文档）；当前生产配置 = 1001 集成，该后处理仅作安全网（1001 模型本身已 L/R 正确，72 例 holdout 仅 1 次重映射、逐位指标不变；`off` 一键关闭）；
- 输出标签空间 1–32：1–8 右上、9–16 左上、17–24 左下（FDI 31–38）、25–32 右下（FDI 41–48）（FDI 对照见 §3 的 label_map）；
- 将输出转回临床 FDI 编号（交付/展示用）见 `fdi_label_conversion.md`，程序为 `inference/tooth_relabel_fdi.py`；
- 上一代 RAS 模型（Dataset1000，66-case +assigner 0.9094，留作对照/回退）：`--model_dir` 改指 `tooth_3d_semantic/results/Dataset1000_ToothSemanticRAS/nnUNetTrainer__nnUNetPlans__3d_fullres`；
- 旧 RPI 帧模型（Dataset999，与旧朝向扫描配套）：`--model_dir` 改指 `tooth_3d_semantic/results/Dataset999_ToothSemantic/nnUNetTrainer__nnUNetPlans__3d_fullres` 即可。

## 10）验证与审计工具

```bash
python scripts/eval_holdout_ras_1000.py --smoke     # 单 case 冒烟（1000 集成）
```

一次性审计/实验脚本 `audit_mha_lr_all480.py` / `audit_midpoint_lr_all480.py` / `flip_input_test.py`（2026-09 L/R 审计与图像镜像检验）与 999 时代 `eval_holdout.py` 已于 2026-10-02 清理删除；结果保留在 `tooth_3d_semantic/logs/`、`tooth_3d_semantic/eval/`（磁盘），结论记录于 `tooth_data_pipeline_plan.md` 与 plan 文档。
