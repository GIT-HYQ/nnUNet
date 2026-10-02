# 连续编号 → FDI 标签转换程序说明

程序：`inference/tooth_relabel_fdi.py`（2026-09-02 编写并验证）

## 1. 目的

- 模型（Dataset999_ToothSemantic, nnUNet v2）输出的标签空间是**连续编号 1–32**：数据转换时由 `dataset_conversion/tooth_3d.py` 的 `auto_build_label_map` 将原始 FDI 编号（11–48）升序排列后依次赋 1–32，原始非牙齿标签 0–10 合并为 0。
- 临床习惯使用 **FDI 编号**（11–18、21–28、31–38、41–48）。预测结果若要展示（3D Slicer / 影像软件 / 报告）或交付，必须转回 FDI。
- 转换是**输出侧后处理**：不改动模型、checkpoint、训练数据或 GT；所有评估 / 审计脚本继续用 1–32 空间对 GT 计算指标（holdout GT 也是 1–32）。

## 2. 标签空间与映射

### 2.1 标签空间定义

- `0` = 背景（含原始非牙齿结构 0–10，转换时已并入 0；模型不会输出 1–10 这些值）
- `1–32` = 连续编号（模型输出空间）
- 输出文件只需容纳 0 与 11–48，`uint8` 足够

### 2.2 映射表

| 连续 | FDI | 连续 | FDI | 连续 | FDI | 连续 | FDI |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 1   | 11  | 9   | 21  | 17  | 31  | 25  | 41  |
| 2   | 12  | 10  | 22  | 18  | 32  | 26  | 42  |
| 3   | 13  | 11  | 23  | 19  | 33  | 27  | 43  |
| 4   | 14  | 12  | 24  | 20  | 34  | 28  | 44  |
| 5   | 15  | 13  | 25  | 21  | 35  | 29  | 45  |
| 6   | 16  | 14  | 26  | 22  | 36  | 30  | 46  |
| 7   | 17  | 15  | 27  | 23  | 37  | 31  | 47  |
| 8   | 18  | 16  | 28  | 24  | 38  | 32  | 48  |

按象限汇总（患者自身左右）：

| 连续范围 | FDI 范围 | 解剖象限 |
|---------|---------|---------|
| 1–8     | 11–18   | 上颌右（UR） |
| 9–16    | 21–28   | 上颌左（UL） |
| 17–24   | 31–38   | 下颌左（LL） |
| 25–32   | 41–48   | 下颌右（LR） |

> 注意：FDI 象限号 1→2→3→4 从正面看**顺时针**编号，因此下颌是 **3x = 左、4x = 右**，与上颌的左右位置相反，容易记错（本数据集 480 case 鼻窦标签审计已确认 GT 符合该标准约定，见 `scripts/audit_mha_lr_all480.py` 的 `EXP_SIDE`）。

### 2.3 映射的权威来源

- 权威来源：`/home/share/clr/share/data/CBCT/tooth_fairy2/list/label_map.yaml` 的 `train_to_raw`（转换脚本生成，与训练所用 GT 逐字节一致）。
- 闭式公式（本数据集 GT 覆盖全部 32 颗牙时才成立）：

  ```
  fdi(i) = 10 * (1 + (i-1) // 8) + ((i-1) % 8 + 1)      i = 1..32
  ```

- 程序启动时用 `label_map.yaml` 构建 LUT，并与闭式公式逐值交叉核对，不一致即报错退出。

## 3. 程序说明（`inference/tooth_relabel_fdi.py`）

### 3.1 功能与流程

读入 1–32 标签图（`.nii.gz`）→ 用 33 元素 LUT 向量化重映射 → 写出 FDI 标签图，并完整保留原文件头（size / spacing / origin / direction）。纯值重映射，不改变任何体素的几何位置。

### 3.2 内置校验（任一失败即报错退出）

1. 输入标签值必须落在 0–32 内（防止误把图像或已 FDI 化的文件当输入）；
2. `label_map.yaml` 中每个训练 id 1–32 必须恰好对应一个 FDI 编号；
3. 映射后每个标签的体素数与映射前逐一相等（双射，无体素增减）；
4. 输出 size / spacing 与输入一致。

### 3.3 用法

```bash
# 单文件
python inference/tooth_relabel_fdi.py \
    --input  /path/to/pred.nii.gz \
    --output /path/to/pred_fdi.nii.gz

# 批量（目录，文件名保持不变）
python inference/tooth_relabel_fdi.py \
    --input_dir  /path/to/preds/ \
    --output_dir /path/to/preds_fdi/

# 指定其他映射文件（默认 /home/share/clr/share/data/CBCT/tooth_fairy2/list/label_map.yaml）
python inference/tooth_relabel_fdi.py --input a.nii.gz --output b.nii.gz \
    --label_map /path/to/other_dataset/list/label_map.yaml
```

环境：conda env `nnunet310`（依赖 numpy / SimpleITK / pyyaml，无需 GPU）。

### 3.4 核心代码

```python
lut = np.zeros(33, dtype=np.uint8)          # lut[0]=0
for i in range(1, 33):
    lut[i] = fdi_from_seq(i)                # 或取自 label_map.yaml 的 train_to_raw
out = sitk.GetImageFromArray(lut[arr])      # arr: 1-32 标签数组 (zyx)
out.CopyInformation(img)                    # 保留 spacing/origin/direction/size
```

## 4. 在推理流程中的接入位置

标准两阶段流程：

1. `inference/tooth_predict.py`：模型推理，输出 **1–32** 标签图（用于评估 / 审计，保留）；
2. `inference/tooth_relabel_fdi.py`：对预测文件做值重映射，输出 **FDI** 标签图（用于展示 / 交付）。

两个文件建议同时保留：1–32 原件是评估 / 复算指标的唯一依据，FDI 文件是展示层产物。

## 5. 验证记录（2026-09-02）

- `label_map.yaml` 与闭式公式：32/32 一致（程序启动时自动核对）。
- 真实样本 `ToothFairy2P_079`（holdout，5 折 ensemble 预测）：
  - 前景体素 247,672，背景 5,093,834；33 个标签映射前后体素数逐一相等；
  - size (237, 191, 118)、spacing (0.3, 0.3, 0.3)、identity direction、origin 均不变；
  - 解剖一致性抽查（RAS identity 下患者右侧 = 较大 x 索引）：FDI 11 x=124.0、FDI 41 x=125.6 在右侧（大 x）；FDI 21 x=96.9、FDI 31 x=109.1 在左侧（小 x），与 GT 象限审计结论一致。

## 6. 注意事项

1. 只转换**输出预测文件**；不要对输入图像、训练数据或评估 GT 应用本程序。
2. 若未来数据集缺牙（32 颗未全覆盖），闭式公式不再成立，必须使用**该数据集自己的** `label_map.yaml`（程序默认已如此，公式仅用于交叉核对）。
3. 3D Slicer 中"同一颗牙颜色不同"通常是两个 labelmap 各自分配了不同 color table；判断以"Show label value"显示的体素值为准。
4. 转换前后文件体积相近（均为压缩 nii.gz）；批量转换 72 个 holdout 文件实测约 20 秒（CPU，无需 GPU）。
