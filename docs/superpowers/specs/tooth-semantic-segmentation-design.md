# 牙齿3D语义分割设计文档

## 目标

使用 nnUNetv2 官方工具链训练 32 类牙齿语义分割模型，支持对新的 CBCT 图像进行推理，输出每颗牙的独立 mask。

## 背景

- 数据集：ToothFairy2，481 个 case（F 系列 + P 系列）
- 标注格式：每个像素值对应一颗特定位置的牙齿（原始 ID 11-48），已映射为连续训练 ID 1-32
- 现有资产：`list/dataset.yaml`（label_map）、`list/tooth_split_k5_seed0.yaml`（5折划分）
- 硬件：4×A10 24GB

## 架构

```
ToothFairy2 原始数据 (扁平 .nii.gz)
        │
        ▼
  ToothFairy2Semantic.py  ──→  nnUNetv2 标准格式目录
                                    ├── imagesTr/ (case_0000.nii.gz)
                                    ├── labelsTr/ (case.nii.gz)
                                    ├── dataset.json
                                    └── splits_final.json
                                       │
                                       ▼  nnUNetv2_plan_and_preprocess
                                       │
                                       ▼  nnUNetv2_train -num_gpus 4
                                       │
                                       ▼  nnUNetv2_predict / inference.py
                                       │
                               新 CBCT → 32通道预测 mask
```

## 组件设计

### 1. 数据转换脚本 `dataset_conversion/ToothFairy2Semantic.py`

**职责：** 将现有扁平数据组织为 nnUNetv2 标准格式。

**输入：**
- `/home/share/clr/share/data/CBCT/tooth_fairy2/*.nii.gz`（图像）
- `/home/share/clr/share/data/CBCT/tooth_fairy2/*_gt.nii.gz`（标签）
- `/home/share/clr/share/data/CBCT/tooth_fairy2/list/dataset.yaml`（label_map）
- `/home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml`（划分）

**输出：**
```
/home/share/clr/share/data/CBCT/tooth_3d_semantic/
├── dataset.json          （含 name, labels(0-32), numTraining=481）
├── splits_final.json     （train/val 列表）
├── imagesTr/
│   ├── ToothFairy2F_001_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── ToothFairy2F_001.nii.gz
    └── ...
```

**关键逻辑：**
- 图像重命名为 `case_0000.nii.gz`（nnUNetv2 要求）
- 标签保持原文件名 `case.nii.gz`
- `dataset.json` 中 labels 映射：0=background, 1-32=各颗牙
- `splits_final.json` 从已有划分读取，格式为 nnUNetv2 期望的 `[{"train": [...], "val": [...]}]`

### 2. 训练流程（nnUNetv2 命令行）

使用默认配置 `Default` + `3d_fullres`：

```bash
# Step 1: 规划与预处理
nnUNetv2_plan_and_preprocess -d 1 --plans_name Default --config_name 3d_fullres

# Step 2: 训练（调试用，单折）
nnUNetv2_train -d 1 -f 0 -tr nnUNetTrainer_1000epochs -num_gpus 4

# Step 3: 全部5折
nnUNetv2_train -d 1 -f 0/1/2/3/4 -tr nnUNetTrainer_1000epochs -num_gpus 4
```

**默认配置说明：**
- `3d_fullres`：全分辨率 3D U-Net，适合 A10 24GB 显存
- DDP 自动处理多卡同步
- 早停 + best model checkpoint 机制

### 3. 推理脚本 `inference.py`

**职责：** 对新 CBCT 做预测。

**输入：** 单个 CBCT 图像（`.nii.gz`）
**输出：** 预测 mask（每个像素值 = 牙齿类别 ID，0=背景）

**实现方式：** 封装 nnUNetv2 的 `predict_single_datum` API，或直接调用 `nnUNetv2_predict` 命令行。

## 数据流

1. 原始 CBCT → (spacing 归一化) → 预处理图像
2. 预处理图像 + 映射后的标签 → DatasetBuilder → DataLoader
3. DataLoader → U-Net forward → CrossEntropy/Loss → backward → optimizer step
4. 推理时：新图像 → spacing 重采样 → inference padding/cropping → 模型输出 → softmax → argmax → mask

## 错误处理

- **缺失文件：** 转换脚本检查每个 case 的 image + label 配对，跳过不完整的
- **标签值异常：** 运行时校验 label 像素值在 [0, 32] 范围内
- **显存不足：** A10 24GB 对 3d_fullres 足够；如出现问题可回退到 `3d_lowres` + `2d` 级联

## 测试策略

1. **冒烟测试：** 用 2-3 个 case 跑完预处理 → 训练 5 epoch → 推理，验证 pipeline 无崩溃
2. **数据完整性：** 校验 dataset.json 的 numTraining 与实际文件数一致
3. **推理正确性：** 检查输出 mask 维度与输入图像一致，像素值在 [0, 32] 范围内
