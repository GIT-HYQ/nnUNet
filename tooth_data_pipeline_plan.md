# CBCT 牙齿语义分割 — 数据处理链路梳理与方案

> 日期:2026-08-27 · 本文所有结论均基于字节级验证、480 case 全量审计与对照实验,无需再重复排查。
> 相关文档:`tooth_3d_quickstart.md`(复现命令)、`docs/superpowers/specs/tooth-semantic-segmentation-design.md`(设计文档)
> **2026-10-02 更新**:本文 §3.4 所述“模型侧失败”的机制已查明 = nnUNet 默认 x 轴镜像数据增强不做标签重映射 → 颌弓级 L/R ±8 ID 互换;三轨方案见 `docs/superpowers/plans/2026-09-20-dataset1000-optimization.md`(Track A 后处理已达标、Track B 1001 重训进行中、Track C 审计证实零 GT 错误)。§8 复现命令引用的一次性脚本已于 2026-10-02 清理删除/加 1000 标识重命名(已加注记)。

## 0. 结论(TL;DR)

1. **数据处理流程没有问题。** 重映射、裁剪、划分、nnUNet 预处理每一环都逐一验证,无静默标签变更、无几何失真、无训练/验证泄漏(§4)。
2. **不需要重新生成数据集。** 转换与划分脚本完全确定性(唯一随机源是 seed=0 的划分),重新生成只会得到字节级相同的数据集,收益为 0,代价是 5-fold 重训;且三个低分 case 是模型侧问题,重生成+重训也治不了。
3. **三个低分 case(F_049 / P_049 / P_105)是模型侧问题**:严重缺牙/不对称牙列下的逐牙 ID 混淆。GT 左右标反与图像镜像两个数据侧假设均已被实验排除(§3)。
4. 18 个空 GT case 是源数据特性(MHA 标注本身无牙),非流程 bug。
5. 评审中确定的 4 项决定均已执行:拷贝 3 个核心脚本(+1 个依赖)入本仓库、删除 7 个一次性调试脚本、异常处置改为"报告+标记"(因数据无错,"主动修复数据"分支不适用)、quickstart 最小修改。

## 1. 数据来龙去脉(字节级验证)

```
① 官方发布 Dataset112_ToothFairy2 (MHA)
   /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2
   480 case / 960 文件(imagesTr + labelsTr),CC-BY-SA 4.0,2024-04-20
   63 例 F 系列 + 417 例 P 系列(ID 稀疏)
   标签:0 背景;1/2 下/上颌骨;3/4 左/右下颌管;5/6 左/右颌窦;
        7 咽;8 桥体;9 冠;10 种植体;11–48 FDI 牙号;40 NA(480 文件中均未出现)
     │
     │ ② CBIM 转换脚本 tooth_3d.py(现已拷贝至本仓库 dataset_conversion/tooth_3d.py)
     │    a. 标签 LUT:0–10 → 0(全部非牙齿 → 背景)
     │       11–18→1–8(右上) 21–28→9–16(左上) 31–38→17–24(左下) 41–48→25–32(右下)
     │    b. 裁剪:bbox(重映射后标签 > 0) + context (10,30,30) z,y,x,双侧 clamp 到图像边界
     │    c. spacing:原生 0.3 == 目标 0.3 → 不重采样
     │    d. 方向:源即 identity → 不旋转
     ▼
tooth_fairy2 (nii.gz,1–32 标签空间)
   /home/share/clr/share/data/CBCT/tooth_fairy2/{case}.nii.gz + {case}_gt.nii.gz + list/
   (list/ 内含 label_map.yaml、dataset.yaml)
     │
     │ ③ tooth_split.py:random.Random(seed=0);15% = holdout_test(72 例:F=8/P=64)
     │    其余 408 例 → 5-fold(每折 train/val = 327/81)
     ▼
   list/tooth_split_k5_seed0.yaml
     │
     │ ④ ToothFairy2Semantic.py:原样拷贝(不重采样/不改标签;GT 最大值 > 32 时中止)
     │    图像改名 {case}_0000.nii.gz,标签 {case}.nii.gz;生成 dataset.json + splits_final.json
     ▼
tooth_3d_semantic/raw/Dataset999_ToothSemantic (Dataset ID 999)
     │
     │ ⑤ nnUNetv2 plan_and_preprocess -d 999(已验证:不做标签重映射;spacing 0.3 = 目标)
     ▼
tooth_3d_semantic/preprocessed/ → 5-fold 3d_fullres 训练
   (4×A10 DDP,global batch 8,patch 80×128×192,1000 epoch/fold)
     │
     │ ⑥ eval_holdout.py:5-fold logit 集成(checkpoint_final,mirroring+gaussian)
     ▼
72 case holdout:raw 均值 0.784 · 69 case(剔 3 空 GT)0.818(中位 0.897)
              · 66 case(再剔 3 个模型侧失败)≈ 0.852
```

**验证方式**:抽样 case 对 "MHA 标签 → LUT 重映射 → 裁剪" 与最终 nii.gz 做字节级对比,完全一致;裁剪公式先由形状差逆推(P_105 的 y 方向 2 体素偏移暴露了 `min(shape, ·)` 的 clamp 规则),再在 5 个 case 上字节级验证。nnUNet 预处理后的标签与 raw 标签逐字节一致(已验证)。

**历史事故(已修复)**:④ 曾对已在 1–32 空间的 GT 再套一次 FDI→1–32 LUT(双重重映射,会把大部分牙齿清零)。fold 1 训练前(2026-08-19)发现并修复;现行 `ToothFairy2Semantic.py` 为原样拷贝,并带 `max>32` 中止守卫(文件 102–105 行注释记录了此事)。

**方向问题与 tooth_fairy2_m(2026-09-03)**:原始 MHA 数组实际按 RPI 顺序存储(头顶在低 z、面部前部在低 y),但元数据声称 identity RAS —— 发布数据自身的元数据错误,非转换引入(证据:40/40 颌窦 case 上颌弓 z < 下颌弓 z、185/185 颌骨 case 下颌骨 z 更高、12/12 抽样 case 门齿 y < 磨牙 y;x 轴正确,左右标注不受影响)。训练/评估不受影响(模型在数组帧工作,图像与标签同帧),但遵循元数据的查看器(如 3D Slicer)会把体积显示成上下颠倒+前后翻转。
2026-09-03 给 `tooth_3d.py` 增加轴向纠正开关(默认关闭),2026-09-04 拆分为 `--fix_z` / `--fix_y`(可单独或组合使用):裁剪后把图像/标签数组沿对应轴翻转(spacing/origin/direction 不变)。已生成:两者同用(真 RAS) → `/home/share/clr/share/data/CBCT/tooth_fairy2_m`(480 case,case 集与裁剪同 `tooth_fairy2`,数组与对应文件严格 flip 对应;`scripts/verify_zy_flip.py --anatomy` 全量验证通过,见 §8 ②′);仅 `--fix_z`(头顶高 z,y 原帧) → `/home/share/clr/share/data/CBCT/tooth_fairy2_z`(`scripts/verify_zy_flip.py --axes 0 --anatomy` 479/480,唯一例外为已知 P_192 左右标注问题,见 §8 ②″)。注意:在旧帧(`tooth_fairy2`)训练的模型与新帧数据不兼容,用新数据集训练需重新训练。


## 2. 五个问题的回答

| # | 问题 | 回答 |
|---|------|------|
| 1 | 原始数据路径 | `/home/share/clr/share/data/CBCT/Dataset112_ToothFairy2` |
| 2 | mha 训练需转 nii.gz 吗 | **转换不是必须的** —— nnUNet 原生可读 `.mha`。真正的必要步骤是 ① FDI→1–32 重映射 ② 前景裁剪 ③ 划分;nii.gz 只是现有管线的历史产物 |
| 3 | 标签构成 | FDI 11–48(32 颗牙)+ 10 个非牙齿类(颌骨/下颌管/颌窦/咽/桥体/冠/种植体/NA);本项目只保留牙齿,非牙齿 → 0 |
| 4 | 体素值映射是否必须 | **必须。** nnUNet 要求连续类标签 0..N,FDI 编号稀疏(11–48)且与非牙齿标签混杂,不能直接训练。LUT 重映射为 1–32(同侧同名牙相差 ±8,便于模型学习同源结构) |
| 5 | 历史脚本哪些在用 | 见 §5:4 个核心脚本已拷入本仓库(3 核心 + utils.py 依赖),7 个一次性调试脚本已删除 |

## 3. 左右(L/R)问题 — 证据链

三个低分 case 的初始怀疑:(A) GT 左右标反,或 (B) 模型失败。**A 已排除,B 确认。**

### 3.1 GT 标注:无左右标反(480 case 全量审计,双方法)

审计全部在 MHA 原生数组坐标系内完成(不用 nibabel、不做重定向),规避方向矩阵陷阱。

- **方法 1(颌窦标签定方向)**:MHA 标签 5/6 本身即"左/右颌窦",可直接确定每例"患者右侧 = 数组哪个方向",不依赖任何方向元数据。42 例同时有 5/6 标注,**42/42 一致**:患者右 = 数组 i 较大方向(x);再核对各 FDI 象限(1x 右上 / 2x 左上 / 3x 左下 / 4x 右下)相对颌窦中线的侧向:**零偏差 case**。
- **方法 2(双侧对中线)**:不依赖颌窦标签,每牙弓中线 = 双侧同名 FDI 对(11↔21…18↔28、31↔41…38↔48)质心连线的中位数,再核对象限侧向。480 case 全量审计(18 例无牙自然不可审)。
- **两方法交叉验证**:42 例颌窦标注 case 上 1112/1113 个象限判断一致。
- **三个关键 case 双方法一致**:

| Case | GT 牙类数 | 方法 1 | 方法 2 | 结论 |
|---|---|---|---|---|
| F_049 | 29 | 0 mismatch | 0 mismatch | 标注与 FDI 一致 |
| P_049 | 17(上颌仅 4) | 0 mismatch | 0 mismatch | 标注与 FDI 一致 |
| P_105 | 28 | 0 mismatch | 0 mismatch | 标注与 FDI 一致 |

产物:`scripts/audit_mha_lr_all480.py` → `tooth_3d_semantic/logs/mha_lr_audit_480.json`;`scripts/audit_midpoint_lr_all480.py` → `tooth_3d_semantic/logs/mha_audit_midpoint.json`。

### 3.2 图像:未镜像(翻转输入对照实验)

5-fold 集成、**`use_mirroring=False`**(见 3.3 陷阱),输入沿 x 轴翻转、预测翻回后对比逐类 dice:

| Case | fg_dice 正常输入 | fg_dice 翻转输入 |
|---|---|---|
| F_049 | 0.0735 | 0.0959 |
| P_049 | 0.1207 | **0.0292(更差)** |
| P_105 | 0.0000 | 0.0252 |

若图像被镜像,翻转输入应**系统性**恢复 dice;实际 P_049 明显变差、F_049 混合模式(翻转输入改善上颌类 2–5,但下颌类 29–31 变差)。这种模式是**混乱的部分 ID 分配,不是干净的整体翻转** → 图像未镜像。产物:`scripts/flip_input_test.py` → `tooth_3d_semantic/eval/flip_input_test_nomirror.json`。

### 3.3 陷阱(重要,记录在案)

nnUNet 的 `use_mirroring=True`(TTA)使输出在数学上**翻转不变**:
`out(flip I) = flip(out(I))`(out 是两个镜像子网络预测的平均)。
因此任何"翻转输入测试"**必须关闭 TTA**,否则结果恒为"不变",不携带任何信息。此前"翻转图像重推理输出不变"的观察是 TTA 假象,不是模型证据。

### 3.4 结论与后果

- 根因 = **B(模型侧)**:5-fold 集成在这 3 位患者严重缺牙/不对称的牙列上确实做不对左右与 ID 分配(P_049 仅 17 颗牙、上颌仅 4 颗;P_105 缺 4 颗;F_049 缺 3 颗)。输出模式为"部分象限正确(dice 0.6–0.85)、多数类接近 0"。
- **2026-09-20 更新**:该模型侧失败的机制已查明 = nnUNet 默认 x 轴镜像数据增强不做标签重映射 → 颌弓级 L/R ±8 ID 互换(失败 case 空间 dice 0.90–0.99 / ID dice 0.14–0.48),见 plan 文档;本节的数据侧结论(GT 无错、图像未镜像)不变且仍成立。
- 原计划的"主动修复数据"分支**不适用**:数据(标注 + 图像)已验证无错,无对象可修。处置改为**报告 + 标记 + 干净的指标分层**(§0-3 的 66 case ≈ 0.852 才是反映模型真实水平的数字)。
- 性能改进方向在模型侧,见 §7。

## 4. 流程问题逐环节判断

| 环节 | 潜在风险 | 验证结果 | 结论 |
|------|---------|---------|------|
| ② FDI→1–32 重映射 | LUT 不全、非牙齿残留、标签 40 泄漏 | LUT 覆盖全部实际出现标签;40 在 480 文件中均未出现;`--strict_unmapped` 通过;抽样 case 字节级一致 | ✅ |
| ② 前景裁剪 | 裁掉解剖结构、丢失左右信息 | 值域驱动(label>0 bbox)+ 空间中性双侧 clamp,与左右无关;公式 5 case 字节级验证;18 例无牙 case 正常通过 | ✅ |
| ② 几何 | 重采样/旋转失真 | 原生 spacing 0.3 = 目标 → 不重采样;方向源即 identity → 不旋转 | ✅ |
| ③ 划分 | 训练/验证泄漏、不可复现 | seed=0 确定性;按 case 划分(图像与 GT 同折);holdout 永不进入任何 fold | ✅ |
| ④ 拷贝进 raw | 双重重映射(历史事故) | 已修复;原样拷贝 + `max>32` 中止守卫;GT 标签空间已验证为 {1..32} | ✅ |
| ⑤ nnUNet 预处理 | 静默改标签 | 预处理后标签与 raw 逐字节一致 | ✅ |
| GT 左右一致性 | 标反 | 480 case 双方法全量审计,3 个关键 case 零 mismatch(§3.1) | ✅ |
| 图像朝向 | 镜像 | 关 TTA 的翻转输入测试,未恢复 dice(§3.2) | ✅ |

**总判断:流程无问题,不需要重新生成数据集**(确定性管线 → 重生成字节级相同;低分 case 属模型侧,重生成亦无收益)。

## 5. 脚本归属(已执行)

**拷入本仓库 `dataset_conversion/`**(原在 CBIM 项目 `/home/share/clr/share/work/seg/CBIM-Medical-Image-Segmentation/dataset_conversion/`):

| 脚本 | 作用 |
|---|---|
| `tooth_3d.py` | MHA → nii.gz(LUT 重映射 + 前景裁剪 + spacing 检查) |
| `tooth_split.py` | 划分(seed=0,15% holdout + 5-fold) |
| `tooth_patch_index.py` | 历史 MedFormer patch 索引(留档) |
| `utils.py` | tooth_3d.py 的硬 import 依赖 |
| `ToothFairy2Semantic.py`、`setup_nnunet_env.sh` | 本仓库已有(nnUNet raw 组装 + 环境变量) |

**已删除**(一次性调试/验证脚本,问题定位完成后无保留价值):
`scripts/` 下 `check_labels_002.py`、`verify_fix_002.py`、`regen_labels_only.py`、`final_check_preprocessed.py`、`scan_all_gts.py`、`analyze_lr_flip.py`、`analyze_lr_flip2.py`(共 7 个)+ `dataset_conversion/__pycache__/`。

**脚本沿革(2026-10-02 更新)**:
- 已删除:999 时代入口 `scripts/train_fold.sh`、`scripts/train_folds_seq.sh`、`scripts/eval_holdout.py`;一次性审计/实验脚本 `scripts/audit_mha_lr_all480.py`、`scripts/audit_midpoint_lr_all480.py`、`scripts/flip_input_test.py`、`scripts/verify_zy_flip.py`(证据保留在 `tooth_3d_semantic/logs/`、`tooth_3d_semantic/eval/` 与本文结论)。
- 归档(加 1000 标识重命名):`scripts/train_fold_ras_1000.sh`、`scripts/train_folds_seq_ras_1000.sh`、`scripts/eval_holdout_ras_1000.py`。
- 在用:`inference/tooth_predict.py`(新图推理,ID assigner 默认开启);1001 轨道 `scripts/assemble_dataset1001.sh`、`scripts/train_fold_noxmirror.sh`、`scripts/train_folds_seq_noxmirror.sh`、`scripts/eval_subset.py`、`scripts/eval_holdout_idassigned.py`、`scripts/lr_census_66.py`;GT 审计 `scripts/audit_wholearch_mirror.py`。

## 6. 遗留事项(仅记录,无需处理)

| 事项 | 说明 | 处置 |
|---|---|---|
| 18 个空 GT case | MHA 标注本身无牙(holdout 3 例:P_113/P_136/P_356;训练 15 例) | 指标分层中已剔除;训练侧为纯背景样本,占比 ~3.7%,影响有限,不值得剔除重训 |
| P_192(训练 case) | 方法 2 下颌 4/4 侧向不符;可能是源标注问题,也可能是方法在缺牙 case 上的退化(单侧缺牙时中线估计不稳);1/480 | 对训练影响可忽略;如需确定,可对该例下颌在源标注工具中人工抽查一次(可选) |
| P_527(holdout) | 方法 2 有 3/5 不符,但实际 holdout dice 0.597 正常 | 中点法噪声(牙数少、中线不稳),非真标反 |
| `tooth_predict.py` 默认环境变量路径 | 内置 `setup_nnunet_env()` 默认指向 `/home/share/clr/share/data/CBCT/tooth_3d_semantic`(该目录不存在),docstring 中 model_dir 示例同样过时;实际 results 在本仓库 `tooth_3d_semantic/results/` | **已解决**(2026-10-02 核证):内置路径已指向本仓库 `tooth_3d_semantic/`,docstring 示例为当前生产数据集 Dataset1000 |
| 设计文档 "481 cases" | 笔误,实际 480 | 下次更新文档时顺手改 |
| quickstart §1–§2 引用的 `analyze_tooth_spacing.py` / `analyze_tooth_labels.py` | 这两个脚本不在本仓库(CBIM 项目历史产物;其结论已固化在 label_map.yaml 与 0.3 spacing 选择中) | **已解决**(2026-10-02):quickstart §1–§2 已加注记(脚本为 CBIM 历史产物、结论已固化) |
| 方向元数据问题(数组 RPI 存储,头声称 identity RAS) | 详见 §1;训练/评估不受影响;遵循元数据的查看器显示上下颠倒+前后翻转 | 2026-09-03 已生成纠正副本 `tooth_fairy2_m`(真 RAS),2026-09-04 生成 `tooth_fairy2_z`(仅纠正 z);旧数据集与已训模型保持不动 |

## 7. 模型侧改进方向(不动数据)

> **2026-09-20 更新**:根因已查明(x 轴镜像数据增强 → L/R ±8 ID 互换),方案按三轨执行:Track A 后处理 ID assigner(holdout 66-case 0.9094,达标)、Track B Dataset1001 去 x 轴镜像重训(训练中)、Track C GT 全量审计(证实零 GT 错误)。见 plan 文档;以下 4 条方向为根因查明前的背景。

三个低分 case 的共同特征是**严重缺牙 / 不对称牙列**;检测部分正常(空间 fg dice 0.78–0.97),失败的是"哪颗牙"的 ID 判别。方向:

1. 统计训练集缺牙样本占比(缺 ≥4 颗的 case 数)——若偏低,模型对缺牙场景欠训练;
2. 损失重加权(当前 fg_dice 为逐类均值,受大体量类主导;可考虑逐牙加权或 focal);
3. 两阶段"检测 → ID 分类"结构(单阶段 32 类头在缺牙下天然易混淆);
4. 上颌弱势(1–16:0.758 < 17–32:0.847)可并查(解剖对比度 / 拥挤遮挡)。

以上均为模型侧改动,**不需要重新生成数据集**。

## 8. 复现命令(实测,conda env `nnunet310`)

> 注(2026-10-02):以下命令中 `verify_zy_flip.py`(②′/②″ 验证)与 `train_fold.sh` / `train_folds_seq.sh` / `eval_holdout.py`(⑥/⑦)为已删除的一次性/999 时代脚本,命令留档,结果在 `tooth_3d_semantic/logs/` 与 `tooth_3d_semantic/eval/`;1000 时代脚本已加 1000 标识重命名(见 §8′)。

```bash
cd /home/share/clr/share/work/CBCT/nnUNet
PY=/root/miniconda3/envs/nnunet310/bin/python

# ② MHA → nii.gz(重映射 + 裁剪)  [已执行;输出 /home/share/clr/share/data/CBCT/tooth_fairy2]
$PY dataset_conversion/tooth_3d.py \
  --src_images /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/imagesTr/ \
  --src_labels /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/labelsTr/ \
  --dst_root /home/share/clr/share/data/CBCT/tooth_fairy2 \
  --label_map /home/share/clr/share/data/CBCT/tooth_fairy2/list/label_map.yaml \
  --strict_unmapped --target_spacing 0.3,0.3,0.3 --crop_foreground --crop_context 10,30,30

# ②′ MHA → nii.gz(带 y/z 纠正;输出 /home/share/clr/share/data/CBCT/tooth_fairy2_m)  [2026-09-03 已执行]
# 与 ② 相同参数,另加 --fix_z --fix_y:数组沿 z、y 轴 flip 为真 RAS(背景见 §1)
$PY dataset_conversion/tooth_3d.py \
  --src_images /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/imagesTr/ \
  --src_labels /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/labelsTr/ \
  --dst_root /home/share/clr/share/data/CBCT/tooth_fairy2_m \
  --label_map /home/share/clr/share/data/CBCT/tooth_fairy2/list/label_map.yaml \
  --strict_unmapped --target_spacing 0.3,0.3,0.3 --crop_foreground --crop_context 10,30,30 --fix_z --fix_y
# 验证(480 case 全量,含 RAS 解剖朝向检查):
$PY scripts/verify_zy_flip.py \
  --orig_root /home/share/clr/share/data/CBCT/tooth_fairy2 \
  --new_root /home/share/clr/share/data/CBCT/tooth_fairy2_m --anatomy

# ②″ MHA → nii.gz(仅纠正 z;输出 /home/share/clr/share/data/CBCT/tooth_fairy2_z)  [2026-09-04 已执行]
# 与 ② 相同参数,另加 --fix_z:头顶高 z,y 保持原帧
$PY dataset_conversion/tooth_3d.py \
  --src_images /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/imagesTr/ \
  --src_labels /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/labelsTr/ \
  --dst_root /home/share/clr/share/data/CBCT/tooth_fairy2_z \
  --label_map /home/share/clr/share/data/CBCT/tooth_fairy2/list/label_map.yaml \
  --strict_unmapped --target_spacing 0.3,0.3,0.3 --crop_foreground --crop_context 10,30,30 --fix_z
# 验证(480 case 全量;未翻转 y 故不含门齿/磨牙 y 检查;479/480,唯一例外为已知 P_192 左右标注问题):
$PY scripts/verify_zy_flip.py \
  --orig_root /home/share/clr/share/data/CBCT/tooth_fairy2 \
  --new_root /home/share/clr/share/data/CBCT/tooth_fairy2_z --axes 0 --anatomy
# 注意: 仅翻转单轴 = 镜像帧( chirality 反):元数据仍 identity,真实面部(低 y)被显示为朝后,
# 按面部判定左右时 1-8 出现在患者左侧。_z 仅适用于不涉及左右的场景;
# 左右正确的自洽纠正集只有 _m(z+y),配合从 anterior 侧观察。图: logs/frame_lr_check_3views.png

# ③ 划分(seed 0)  [已执行]
$PY dataset_conversion/tooth_split.py \
  --data_root /home/share/clr/share/data/CBCT/tooth_fairy2 --k_fold 5 --seed 0

# ④ 组装 nnUNet raw(Dataset999)  [已执行]
$PY dataset_conversion/ToothFairy2Semantic.py

# ⑤ 规划 + 预处理  [已执行]
source dataset_conversion/setup_nnunet_env.sh
$PY -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 999

# ⑥ 训练单折(5 折可顺序驱动)  [已训练]
bash scripts/train_fold.sh 0        # fold ∈ {0..4};内部 4×A10 DDP、global batch 8、1000 epoch

# ⑦ holdout 评估(72 case,5-fold 集成,4 GPU,~16 min)  [已执行]
$PY scripts/eval_holdout.py         # 或 --smoke 单 case 冒烟

# ⑧ 新图推理(单折 0,checkpoint_final)
$PY inference/tooth_predict.py \
  --model_dir tooth_3d_semantic/results/Dataset999_ToothSemantic/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 \
  --input /path/to/new_scan.nii.gz \
  --output /path/to/pred.nii.gz
```

注:②③④ 均为确定性脚本,重跑输出字节级相同 —— 这正是"无需重新生成数据集"的依据。

### 8′ RAS 帧重训(Dataset1000_ToothSemanticRAS,2026-09-05 起)

用真 RAS 纠正帧 `tooth_fairy2_m` 重训 5 折,新 ID 1000(旧 999 全部保留)。GPU:4×A10 用后 3 张(`CUDA_VISIBLE_DEVICES=1,2,3`)。

```bash
# ④′ 组装 nnUNet raw(Dataset1000)  [已执行]
$PY dataset_conversion/ToothFairy2SemanticRAS.py
#   480 case 标签逐例校验 0..32;splits_final.json 从 Dataset999 逐字节复制
#   → 5 折划分与 72 case holdout 和 999 完全相同,结果与 0.818 基线直接可比

# ⑤′ 规划 + 预处理  [已执行]
$PY -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 1000
#   自动 plan 的 3d_fullres batch_size=2,已手动改:fold_0 用 18 训完;
#   2026-09-09 起 fold_1–4 改用 8(与 999 样本预算对齐,见 ⑥′);
#   patch [80,128,192]、spacing 0.3、架构自动一致

# ⑥′ 训练单折(3×A10 DDP;global batch 8 → per-GPU 3/3/2)  [fold_0 完成,fold_1–4 进行中]
bash scripts/train_fold_ras_1000.sh 0      # 或顺序驱动 bash scripts/train_folds_seq_ras_1000.sh
#   脚本带 --c: fold 目录有 checkpoint 即自动恢复(final→latest→best),无则新训;
#   已完成折带 --c 重跑=安全空转(重存 final+重验证后 exit 0);强制重训先删 fold 目录
#   权威进度: results/.../fold_<N>/training_log_*.txt(最新时间戳,每 epoch 落盘);
#   logs/train_ras_fold<N>.log 经 tee 块缓冲,长时间静默属正常
#   batch 沿革: 24(8/GPU)OOM → 18(6/GPU,98.5% 显存)训完 fold_0 → 09-09 改 8:
#   nnUNet 总样本=250 迭代×1000 epoch×全局 batch,18 时每折 450 万=2.25×999(200 万),
#   改 8 对齐 999 预算;fold_1 在 18 下已训 ~150 epoch 已删除重跑;fold_0 保留 18 的结果
#   每折 ~38 h,5 折全部完成 ≈ 2026-09-16
#   注: 2026-09-05 夜一次误重启(脚本当时未带 --c)把 fold_0 epoch-41 checkpoint 覆盖丢失,
#       后自 epoch 8 续训完成;教训已固化进脚本(--c)与监控流程

# ⑦′ holdout 评估: 用 scripts/eval_holdout_ras_1000.py(2026-09-20 已执行,69 非空 0.8177,结果见 quickstart §8)
```
