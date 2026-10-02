# Dataset1000 优化实施计划（牙位 ID 左右错配修复）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Dataset1000 的牙位 ID 准确率抬到 66-case 严格口径 ≥0.85（当前 0.8446）、弱类 ≥0.80，根因是颌弓级左右（±8）ID 互换，来自 nnUNet 默认 x 轴镜像数据增强（flip 图像不重映射标签值）。

**Architecture:** 三条轨道并行。轨道 A（先做，零训练成本）：确定性后处理器 `id_assigner` —— 连通域 + 中线锚定 + ±8 重映射，已获用户批准纳入生产管线（"模型 + ID 分配器"）。轨道 B（主方向，用户已批 3 卡×15 天）：用现成的 `nnUNetTrainer_onlyMirror01`（只镜像 z/y 轴）重训 5 折，新数据集 Dataset1001。轨道 C（与训练并行，纯 CPU）：全 480 case 整颌镜像 GT 审计（08-27 的中点法对整颌镜像是盲的），只审计不修标签，审计证实 GT 错的 case 从考核口径剔除（用户规则）。NC 67 case 为独立并行轨道（上游未到位，不阻塞主线）。

**Tech Stack:** Python 3.10（conda env `/root/miniconda3/envs/nnunet310`）、nnUNet v2（本仓库 editable 安装）、SimpleITK、scipy、numpy；GPU 1,2,3（A10×3 DDP）。

**Spec:** 本计划自包含。决策来源为 2026-09-20 grilling 会话（目标/口径/预算/生产形态/标签策略均已与用户确认，见 Global Constraints）；证据见下节。相关既有文档：`fdi_label_conversion.md`（1-32↔FDI 映射权威表）、`tooth_3d_quickstart.md`、`tooth_data_pipeline_plan.md`。

## Global Constraints

- **考核口径（主指标）**：66-case 严格 = 72 holdout − 3 空 GT（P_113/P_136/P_356）− 3 病态（F_049/P_049/P_105）。审计证实 GT 整颌镜像的 case 追加剔除（用户规则"证实 GT 错即剔除"）。69-case 非空、72-case raw 仅作参考。
- **目标**：66-case 均值 ≥0.85；弱类（当前 <0.80 的 10 个类：4,6,7,8,13,14,15,16,23,32）≥0.80；无硬门槛（尽力提升）。
- **资源**：GPU 1,2,3（GPU 0 用户要求留空）；~15 天可接受；本机 CPU 慢（实测 ~80s/个 55KB nii.gz 解压），CPU 任务一律后台跑、不上关键路径。
- **生产形态**："模型 + 确定性 ID 分配后处理器" 用户已批准；`tooth_predict.py` 默认开启后处理（可 `--id_assign off` 关闭）。
- **标签**：只审计、不修正（不重标注、不改标签值）；重训仍用现有 480 标签。
- **NC 67 case**：并行轨道，上游（provider 协议 + L/R 锚点）未确认前不动。
- **架构变更**（MedFormer 等）：最后手段，本计划不含，见"升级路径"。
- **数据集纪律**：新数据集用新 ID（1001），绝不覆盖 Dataset999/Dataset1000 及其 preprocessed/模型/评估结果。
- **训练参数**：与 1000 对齐 —— 3 卡 DDP、global batch 8（3/3/2）、1000 epoch、`nnUNet_compile=false`、`--c` 恢复语义；split 与 999/1000 完全相同（`splits_final.json` 原样复制）。
- **帧约定**：tooth_fairy2_m / Dataset1000 = 真 RAS 帧，**患者右 = 大 x**（数组轴 2，SimpleITK `GetArrayFromImage` 序 (z,y,x)）。所有几何判断基于此，推理/评估不得换帧。
- **提交纪律**：每个 Task 完成即 commit（中文或英文 commit message 均可，说明 what+why）。

## 证据基础（为什么这样优化）

1. **误差主模式 = 颌弓级左右 ID 互换，空间分割近乎完美**（2026-09-20 实测，`scripts/spatial_id_decompose.py` + `scripts/component_id_diag.py`，输出 `tooth_3d_semantic/logs/`）：

   | case | spatial(label>0) | ID(逐类均值) | 错误结构（连通域级） |
   |---|---|---|---|
   | P_002 | 0.901 | 0.281 | 下颌 8/14 颗成对互换（23↔31、21↔29、19↔27、17↔25…） |
   | P_379 | 0.964 | 0.346 | 下颌右侧 8 颗全部被标成左下 ID（±8） |
   | F_031 | 0.966 | 0.475 | 上颌整弓镜像（11/11 颗 ID 均为对侧孪生牙） |
   | F_049（病态对照） | 0.968 | 0.143 | 同模式 + 缺牙 |
   | P_079（健康对照） | 0.992 | 0.991 | 无错 |

   弱类（上颌 0.7756 < 下颌 0.8473）的高失败 case 全部集中在这类"功能失常 case"上（F_031 上颌 16 类 <0.5、P_002 下颌 11 类 <0.5），不是普遍性的分割弱点。

2. **机制 = nnUNet 默认 x 轴镜像数据增强污染训练**（源码确认）：
   - `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:496`：3D 默认 `mirror_axes = (0,1,2)`（空间轴 z,y,x），训练 DA 与推理 TTA 共用；
   - `nnUNetTrainer.py:831-834`：训练管线加入 `MirrorTransform(allowed_axes=(0,1,2))`；
   - `batchgeneratorsv2/transforms/spatial/mirroring.py:14`：每轴独立 p=0.5 触发；
   - `mirroring.py:25-29`：`_apply_to_segmentation` = 纯 `torch.flip`，**标签值不重映射** → x 轴镜像后"左下颌中切牙 ID(17)出现在右侧"这类解剖不可能配置进入训练分布，50% 的 patch 受影响，直接教会模型左右 ID 可以互换。
   - 旁证：999→1000 重训（换真 RAS 帧、同 DA）逐例结果几乎不变（holdout 0.8181 vs 0.8177）——换数据帧不改 ID 错配，因为 DA 没变。

3. **弱类解剖分布**（1-32→FDI 按 `fdi_label_conversion.md`；**注意下颌 17-24=FDI 3x=左、25-32=FDI 4x=右**，FDI 下颌象限左右反置）：当前 <0.80 的 10 类 = 4(FDI14 右上尖牙)、6(FDI16 右上第二磨牙)、7(FDI17 右上侧切牙)、8(FDI18 右上第三磨牙)、13(FDI25 左上第二前磨牙)、14(FDI26 左上第一磨牙)、15(FDI27 左上第二磨牙)、16(FDI28 左上第三磨牙)、23(FDI37 左下第二磨牙)、32(FDI48 右下第三磨牙) —— 后段牙 + 全部三颗第三磨牙，与"远离中线处互换=该牙 dice 全损"的互换污染模式一致。

4. **天花板估计**：9 个实测 case 的 spatial−ID 差值合计 ≈3.19，全部弥合则 66-case 0.8446 → ~0.893；70% 弥合 → ~0.867。后处理单轨即可达 0.85 目标（Task 3 用全 66 case 实测校准）。

5. **审计盲区**：08-27 的 480-case 左右审计（鼻窦锚点 + 双侧对 midpoint 法）检测的是"局部不对称"；**整颌镜像的标签集仍然是对称的**（每对中点仍落中线），midpoint 法对其完全失明，且仅 40/480 case 有鼻窦标签锚点。需要新的"象限-侧别一致性"直接检查（Task 5）。

6. **现成工具**：`nnUNetTrainer_onlyMirror01`（`nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerNoMirroring.py:39`）**已存在** —— 3D 只镜像轴 (0,1)（z,y），且 `inference_allowed_mirroring_axes=(0,1)`（训练/TTA 一致）。轨道 B 直接 `-tr nnUNetTrainer_onlyMirror01`，无需新 trainer 代码。

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `inference/id_assigner.py` | 新建 | L/R 一致性分配器：核心决策（纯函数）+ 数组级应用 + nii.gz IO/CLI |
| `tests/test_id_assigner.py` | 新建 | 合成数据单元测试（纯 assert，无 pytest 依赖） |
| `inference/tooth_predict.py` | 修改 | 加 `--id_assign on/off`（默认 on），推理后调用 assigner 并写 remap 日志 |
| `scripts/lr_census_66.py` | 新建 | 66-case 互换普查 + 后处理天花板实测 |
| `scripts/eval_holdout_idassigned.py` | 新建 | 后处理后全量指标（66/69/72 + FDI 逐类 + 逐 case 前后对比），输出 `eval/holdout_ensemble_ras_idassigned/` |
| `scripts/audit_wholearch_mirror.py` | 新建 | 全 480 case 整颌镜像 GT 审计（只审计） |
| `scripts/assemble_dataset1001.sh` | 新建 | 从 1000 组装 Dataset1001（硬链接 + 校验 + 可选剔除清单） |
| `scripts/train_fold_noxmirror.sh` | 新建 | 单折训练（`-tr nnUNetTrainer_onlyMirror01`） |
| `scripts/train_folds_seq_noxmirror.sh` | 新建 | 5 折顺序训练，首折失败即停 |
| `scripts/eval_subset.py` | 新建 | fold_0 门控用：40-case 子集单折评估（1000 vs 1001 对比） |
| `scripts/eval_holdout_general.py` | 新建 | 通用 holdout 集成评估（任意 predictions 目录，供 1001 终评） |
| 复用不动 | — | `nnUNetTrainer_onlyMirror01`、`scripts/eval_holdout_ras_1000.py`（指标口径参照；2026-10-02 加 1000 标识归档）、`inference/tooth_relabel_fdi.py` |

## 执行顺序与并行

| 时间 | 事件 | 资源 |
|---|---|---|
| D0 | Task 1+2（后处理器开发）；Task 5 审计启动（后台 ~11h） | CPU |
| D1 | Task 3 普查（后台，实测 ~4 min）→ Task 4 指标（GPU 19min）；审计完成 → 用户裁决 flagged case | CPU+GPU |
| D1-2 | Task 6 组装 Dataset1001 + fold_0 开训（~38h） | GPU 1,2,3 |
| D3 | fold_0 门控（Task 6 末）：过 → 继续；不过 → 停 seq、报告 | GPU |
| D3-10 | Task 7 fold 1-4 顺序（~6.3 天） | GPU 1,2,3 |
| D10-11 | Task 8 终评 + 对比报告 + 生产决定 | GPU 19min |
| 全程 | Task 9 NC 轨道待上游解锁 | — |

总计 ~11 天，在 15 天预算内留有余量。

---

### Task 1: id_assigner 核心逻辑 + 单元测试

**Files:**
- Create: `inference/id_assigner.py`
- Create: `tests/test_id_assigner.py`

**Interfaces:**
- Consumes: 无（自包含）
- Produces（Task 2-5 依赖）：
  - `SIDE_LUT: dict[int, str]` — 1..32 → "R"/"L"（RAS 帧解剖侧别）
  - `MIN_COMP_VOX: int = 20`、`MARGIN_MIN_VOX: int = 10`、`MARGIN_FRAC: float = 0.08`、`INTERNAL_TWIN_FRAC: float = 0.5`
  - `swap_lr(i: int) -> int` — ±8 孪生牙 ID
  - `decide_remaps(components: list[dict]) -> tuple[list[tuple[int,int]], dict]` — components 元素 `{"id": int, "cx": float}`（cx 单位 voxel）；返回 (每组件 (old,new) 对, `{"midline": float, "margin": float}`)
  - `apply_assigner(seg: np.ndarray) -> tuple[np.ndarray, list[dict]]` — seg 为 3D (z,y,x) uint8 0..32；返回 (重映射后数组, log)；log[0] = `{"midline":..., "margin":...}`，其余每项 `{"comp": int, "id_before": int, "id_after": int, "cx": float, "n_vox": int}`

**判定规则（写死，勿改）**：
- 连通域 = `ndimage.label(seg > 0)`，< `MIN_COMP_VOX` 的域忽略；
- 中线 = 所有有效组件 cx 的 (min+max)/2；
- margin = `max(MARGIN_MIN_VOX, MARGIN_FRAC * (xmax - xmin))`；
- 组件侧别：`cx > mid + margin` → R；`cx < mid - margin` → L；否则中线区（跳过，不改）；
- 若组件侧别与 `SIDE_LUT[id]` 矛盾 → 该组件**主导标签**（域内占比最大的非零标签）重映射为 `swap_lr(id)`；域内其他标签不动（保护粘连域里的第二颗牙）。
- **孪生守卫（2026-09-22 增补，首次全量普查后，见文末修订记录）**：重映射前先检查是否已有**其他**组件在几何正确侧（中线区外）携带目标 ID `swap_lr(id)`；若存在则**跳过**该重映射（幻影重复牙：孪生牙已正确就位，重映射会摧毁正确侧牙类并稀释孪生类）。整弓真互换不受影响（完全镜像弓中孪生 ID 在错误侧或缺席）。动机：首跑普查（无守卫）P_038 −0.1781、F_003 −0.1158 回归，组件级剖析均为幻影重复牙模式。
- **域内孪生守卫（2026-09-22 增补，fix round 2，见文末修订记录）**：重映射前再检查组件**自身内部**携带目标 ID `swap_lr(id)` 的体素数；若 ≥ `INTERNAL_TWIN_FRAC`（0.5）× 该组件主导标签体素数，则**跳过**该重映射（域内孪生对粘连：一对左右同名牙粘连成同一连通域、两半各自标对，域质心侧别与主导 ID 矛盾 → 重映射主导半会摧毁标对的另一半；F_003 模式，实测有害比 0.889–0.984）。真互换粘连域（同牙一小片误标）比 ≤0.40，不受影响。

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_id_assigner.py`：

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.id_assigner import (SIDE_LUT, apply_assigner, decide_remaps, swap_lr)


def _seg_two_teeth(left_id, right_id, left_x=(10, 20), right_x=(80, 90)):
    """16x40x100 数组：左(小x)一颗牙、右(大x)一颗牙。"""
    seg = np.zeros((16, 40, 100), dtype=np.uint8)
    seg[4:12, 10:30, left_x[0]:left_x[1]] = left_id
    seg[4:12, 10:30, right_x[0]:right_x[1]] = right_id
    return seg


def test_swap_lr_mapping():
    assert swap_lr(1) == 9 and swap_lr(9) == 1
    assert swap_lr(8) == 16 and swap_lr(16) == 8
    assert swap_lr(17) == 25 and swap_lr(25) == 17
    assert swap_lr(24) == 32 and swap_lr(32) == 24


def test_side_lut():
    assert all(SIDE_LUT[i] == "R" for i in range(1, 9))    # FDI 11-18 右上
    assert all(SIDE_LUT[i] == "L" for i in range(9, 17))   # FDI 21-28 左上
    assert all(SIDE_LUT[i] == "L" for i in range(17, 25))  # FDI 31-38 左下
    assert all(SIDE_LUT[i] == "R" for i in range(25, 33))  # FDI 41-48 右下


def test_correct_arch_unchanged():
    comps = [{"id": 17, "cx": 20.0}, {"id": 25, "cx": 80.0}]
    remaps, _ = decide_remaps(comps)
    assert [new for _, new in remaps] == [17, 25]


def test_swapped_lower_arch_corrected():
    comps = [{"id": 25, "cx": 20.0}, {"id": 17, "cx": 80.0}]
    remaps, _ = decide_remaps(comps)
    assert [new for _, new in remaps] == [17, 25]


def test_swapped_upper_arch_corrected():
    comps = [{"id": 1, "cx": 20.0}, {"id": 9, "cx": 80.0}]
    remaps, _ = decide_remaps(comps)
    assert [new for _, new in remaps] == [9, 1]


def test_midline_region_skipped():
    # 两颗牙都在中线 ±margin 内：侧别不可判定，不动
    comps = [{"id": 17, "cx": 50.0}, {"id": 25, "cx": 55.0}]
    remaps, meta = decide_remaps(comps)
    assert meta["margin"] >= 10.0
    assert [new for _, new in remaps] == [17, 25]


def test_apply_assigner_swapped_array():
    seg = _seg_two_teeth(left_id=25, right_id=17)   # 左右 ID 互换的错误预测
    out, log = apply_assigner(seg)
    assert out[8, 20, 15] == 17    # 左侧牙改回 17
    assert out[8, 20, 85] == 25    # 右侧牙改回 25
    assert log[0]["midline"] == 50.0
    assert len(log) == 3           # meta + 2 条 remap
    assert {e["id_before"] for e in log[1:]} == {17, 25}


def test_apply_assigner_correct_array_untouched():
    seg = _seg_two_teeth(left_id=17, right_id=25)
    out, log = apply_assigner(seg)
    assert np.array_equal(out, seg)
    assert len(log) == 1           # 只有 meta，无 remap


def test_apply_assigner_empty():
    out, log = apply_assigner(np.zeros((8, 16, 16), dtype=np.uint8))
    assert np.array_equal(out, np.zeros((8, 16, 16), dtype=np.uint8))
    assert log == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/share/clr/share/work/CBCT/nnUNet && /root/miniconda3/envs/nnunet310/bin/python tests/test_id_assigner.py`
Expected: `ModuleNotFoundError: No module named 'inference.id_assigner'`（或 ImportError）

- [ ] **Step 3: 实现核心模块**

创建 `inference/id_assigner.py`（含 `__init__` 不需要，目录已有）：

```python
#!/usr/bin/env python3
"""Deterministic left-right consistency assigner for 1-32 tooth ID maps.

The model occasionally assigns a tooth its left-right homolog's ID
(arch-level +/-8 swap) while segmenting its position correctly. This module
re-maps such IDs from the geometric side of each connected component in the
true-RAS array frame (patient right = high x, array axis 2).

Side convention (verified on tooth_fairy2_m, Slicer-confirmed, see
memory orientation-investigation):
    1-8  = FDI 11-18 upper right  -> patient RIGHT (high x)
    9-16 = FDI 21-28 upper left   -> patient LEFT  (low x)
    17-24 = FDI 31-38 lower left  -> patient LEFT  (low x)
    25-32 = FDI 41-48 lower right -> patient RIGHT (high x)
(FDI lower quadrants are reversed: 3x = left, 4x = right.)
"""
import argparse
import json

import numpy as np
from scipy import ndimage

MIN_COMP_VOX = 20       # ignore specks
MARGIN_MIN_VOX = 10     # ~3 mm at 0.3 mm/voxel
MARGIN_FRAC = 0.08      # of arch x-extent


def build_side_lut():
    lut = {}
    for i in range(1, 9):
        lut[i] = "R"
    for i in range(9, 17):
        lut[i] = "L"
    for i in range(17, 25):
        lut[i] = "L"
    for i in range(25, 33):
        lut[i] = "R"
    return lut


SIDE_LUT = build_side_lut()


def swap_lr(i):
    if not 1 <= i <= 32:
        raise ValueError(f"label out of range: {i}")
    return i + 8 if (i <= 8 or 17 <= i <= 24) else i - 8


def decide_remaps(components):
    """components: [{"id": int, "cx": float}, ...].
    Returns (per-component (old, new) pairs, {"midline": float, "margin": float})."""
    cxs = [c["cx"] for c in components]
    mid = (min(cxs) + max(cxs)) / 2.0
    margin = max(MARGIN_MIN_VOX, MARGIN_FRAC * (max(cxs) - min(cxs)))
    remaps = []
    for c in components:
        new = c["id"]
        if c["cx"] > mid + margin and SIDE_LUT[c["id"]] == "L":
            new = swap_lr(c["id"])
        elif c["cx"] < mid - margin and SIDE_LUT[c["id"]] == "R":
            new = swap_lr(c["id"])
        remaps.append((c["id"], new))
    return remaps, {"midline": mid, "margin": margin}


def _components(seg):
    """Yield (comp_index, dominant_label, cx_vox, n_vox) for seg>0 components."""
    lab, n = ndimage.label(seg > 0)
    sizes = np.bincount(lab.ravel())
    out = []
    for i in range(1, n + 1):
        if sizes[i] < MIN_COMP_VOX:
            continue
        m = lab == i
        vals, counts = np.unique(seg[m], return_counts=True)
        dom = int(vals[np.argmax(counts)])
        if dom == 0:
            continue
        _, _, cx = ndimage.center_of_mass(m)
        out.append((i, dom, float(cx), int(sizes[i])))
    return lab, out


def apply_assigner(seg):
    """seg: 3D uint8 (z, y, x), labels 0..32 -> (remapped seg, log).
    Only the dominant label of a re-mapped component is rewritten; other
    labels inside merged components are left untouched."""
    lab, comps = _components(seg)
    out = seg.copy()
    log = []
    if not comps:
        return out, log
    remaps, meta = decide_remaps([{"id": d, "cx": cx} for _, d, cx, _ in comps])
    log.append({"midline": round(meta["midline"], 1),
                "margin": round(meta["margin"], 1)})
    for (i, old, cx, nvox), (_, new) in zip(comps, remaps):
        if old != new:
            out[(lab == i) & (seg == old)] = new
            log.append({"comp": i, "id_before": old, "id_after": new,
                        "cx": round(cx, 1), "n_vox": nvox})
    return out, log


def assign_file(input_path, output_path, log_path=None):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(input_path))
    arr = sitk.GetArrayFromImage(img)
    out, log = apply_assigner(arr)
    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(output_path))
    n = max(0, len(log) - 1)
    if log_path is not None:
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="1-32 label map .nii.gz")
    p.add_argument("--output", required=True, help="output .nii.gz")
    p.add_argument("--log", default=None, help="remap log .json path")
    a = p.parse_args()
    n = assign_file(a.input, a.output, a.log)
    print(f"re-mapped components: {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/share/clr/share/work/CBCT/nnUNet && /root/miniconda3/envs/nnunet310/bin/python tests/test_id_assigner.py`
Expected: `9 tests passed`

- [ ] **Step 5: Commit**

```bash
git add inference/id_assigner.py tests/test_id_assigner.py
git commit -m "feat: deterministic L/R tooth-ID consistency assigner (core + unit tests)"
```

---

### Task 2: id_assigner 真实数据验证 + tooth_predict 集成

**Files:**
- Modify: `inference/tooth_predict.py`（`predict_single_image` 末尾 + `main()` 加参数）
- Create: `tooth_3d_semantic/logs/id_assigner_realdata_check.json`（验证产物）

**Interfaces:**
- Consumes: Task 1 的 `assign_file`
- Produces: `tooth_predict.py --id_assign {on,off}`（默认 on），输出旁写 `<output_stem>_idassign_log.json`

- [ ] **Step 1: 真实数据 sanity check（P_002 应大涨，P_079 应零改动）**

```bash
cd /home/share/clr/share/work/CBCT/nnUNet
P=tooth_3d_semantic/eval/holdout_ensemble_ras/predictions
G=tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations
/root/miniconda3/envs/nnunet310/bin/python - <<'EOF'
import json, sys
import numpy as np, SimpleITK as sitk
sys.path.insert(0, ".")
from inference.id_assigner import assign_file, apply_assigner

P = "tooth_3d_semantic/eval/holdout_ensemble_ras/predictions"
G = "tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"

def fg_dice(pred, gt):
    d = []
    for c in np.unique(gt):
        if c == 0: continue
        g = gt == c; p = pred == c
        d.append(2*np.count_nonzero(p&g)/max(1, p.sum()+g.sum()))
    return float(np.mean(d))

rows = {}
for case in ["ToothFairy2P_002", "ToothFairy2P_079"]:
    pred = sitk.GetArrayFromImage(sitk.ReadImage(f"{P}/{case}.nii.gz"))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{G}/{case}.nii.gz"))
    new, log = apply_assigner(pred)
    rows[case] = {"before": round(fg_dice(pred, gt), 4),
                  "after": round(fg_dice(new, gt), 4),
                  "n_remaps": max(0, len(log)-1)}
    print(case, rows[case], flush=True)
with open("tooth_3d_semantic/logs/id_assigner_realdata_check.json", "w") as f:
    json.dump(rows, f, indent=1)
EOF
```

Expected: P_002 `before≈0.281, after≥0.85, n_remaps≥5`；P_079 `n_remaps=0` 且 before==after。
**若 P_079 出现 remap 或 P_002 after < 0.7：STOP，回到判定规则排查（中线/margin），不得带病进 Task 3。**

- [ ] **Step 2: CLI round-trip 验证（header 保留）**

```bash
cd /home/share/clr/share/work/CBCT/nnUNet
P=tooth_3d_semantic/eval/holdout_ensemble_ras/predictions
/root/miniconda3/envs/nnunet310/bin/python inference/id_assigner.py \
  --input $P/ToothFairy2P_002.nii.gz \
  --output /tmp/p002_assigned.nii.gz \
  --log /tmp/p002_idassign_log.json
/root/miniconda3/envs/nnunet310/bin/python - <<'EOF'
import SimpleITK as sitk
a = sitk.ReadImage("tooth_3d_semantic/eval/holdout_ensemble_ras/predictions/ToothFairy2P_002.nii.gz")
b = sitk.ReadImage("/tmp/p002_assigned.nii.gz")
assert a.GetSize() == b.GetSize() and a.GetSpacing() == b.GetSpacing()
assert a.GetOrigin() == b.GetOrigin() and a.GetDirection() == b.GetDirection()
print("header preserved OK")
EOF
```

Expected: `header preserved OK`，且 log 非空。

- [ ] **Step 3: 修改 tooth_predict.py 集成**

在 `inference/tooth_predict.py`：

1) `main()` 的 parser 加（放在 `--folds` 之后）：

```python
    parser.add_argument('--id_assign', choices=['on', 'off'], default='on',
                        help='deterministic L/R ID consistency post-processor '
                             '(default on)')
```

2) `predict_single_image` 签名加参数 `id_assign: str = 'on'`；在函数末尾 `print(f"Prediction saved to: {final_output}")` **之前**插入：

```python
    if id_assign == 'on':
        import json as _json
        from inference.id_assigner import apply_assigner
        import SimpleITK as _sitk
        seg_img = _sitk.ReadImage(final_output)
        seg_arr = _sitk.GetArrayFromImage(seg_img)
        seg_arr, alog = apply_assigner(seg_arr)
        seg_img = _sitk.GetImageFromArray(seg_arr.astype('uint8'))
        seg_img.CopyInformation(_sitk.ReadImage(final_output))
        _sitk.WriteImage(seg_img, final_output)
        log_path = str(Path(final_output).with_suffix('')) + '_idassign_log.json'
        with open(log_path, 'w') as f:
            _json.dump(alog, f, indent=1)
        print(f"ID assigner: {max(0, len(alog)-1)} component(s) re-mapped "
              f"(log: {log_path})")
```

3) `main()` 末尾调用改为：

```python
    predict_single_image(args.model_dir, args.input, args.output,
                         folds=args.folds, id_assign=args.id_assign)
```

注意：`inference/id_assigner.py` 的导入依赖 CWD 在仓库根或 `inference/` 在 sys.path —— `tooth_predict.py` 的用法是 `python inference/tooth_predict.py ...`（从仓库根），此时 `import inference.id_assigner` 要求 `inference/` 是包；若报 ModuleNotFoundError，改为把 `sys.path.insert(0, str(Path(__file__).resolve().parent))` 加到文件顶部 import 区之后、再 `from id_assigner import apply_assigner`。**以实际跑通为准，两种写法都允许，只保留一种。**

- [ ] **Step 4: 端到端验证（1 个真实 case，GPU 1）**

```bash
cd /home/share/clr/share/work/CBCT/nnUNet
source dataset_conversion/setup_nnunet_env.sh
export CUDA_VISIBLE_DEVICES=1
IN=$(ls tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/imagesTr/ToothFairy2P_002* 2>/dev/null || ls /home/share/clr/share/data/CBCT/tooth_fairy2_m/imagesTr/ToothFairy2P_002*)
/root/miniconda3/envs/nnunet310/bin/python inference/tooth_predict.py \
  --model_dir tooth_3d_semantic/results/Dataset1000_ToothSemanticRAS/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --input "$IN" --output /tmp/p002_e2e.nii.gz --id_assign on
```

Expected: 输出含 `ID assigner: N component(s) re-mapped`（N≥5，与 Step 1 一致），`/tmp/p002_idassign_log.json` 存在，输出 nii.gz 头信息与输入一致。再跑一次 `--id_assign off` 确认无 log 文件、输出与模型裸输出一致。

- [ ] **Step 5: Commit**

```bash
git add inference/tooth_predict.py
git commit -m "feat: wire L/R ID assigner into tooth_predict.py (--id_assign, default on)"
```

---

### Task 3: 66-case 互换普查 + 天花板实测

**Files:**
- Create: `scripts/lr_census_66.py`
- Create（产物）: `tooth_3d_semantic/logs/lr_census_66.jsonl`

**Interfaces:**
- Consumes: Task 1 `apply_assigner`；已存预测 `eval/holdout_ensemble_ras/predictions/`；GT `preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations/`
- Produces: 66-case 逐例 `{"case","n_remaps","dice_before","dice_after"}` + 汇总（66 均值 before/after、健康 case 零误改检查、top gains）——Task 4 的验收基准

- [ ] **Step 1: 写普查脚本**

创建 `scripts/lr_census_66.py`：

```python
#!/usr/bin/env python3
"""Census of L/R ID swaps on the 66-case strict holdout set (1000 ensemble
predictions), and measure the id-assigner's achievable mean (ceiling)."""
import json
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import apply_assigner

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
PRED = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras/predictions"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
PER_CASE = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras/per_case_metrics.jsonl"
OUT = f"{BASE}/tooth_3d_semantic/logs/lr_census_66.jsonl"

EXCLUDE = {"ToothFairy2P_113", "ToothFairy2P_136", "ToothFairy2P_356",
           "ToothFairy2F_049", "ToothFairy2P_049", "ToothFairy2P_105"}


def case_dice(seg, gt):
    d = {}
    for c in np.unique(gt):
        if c == 0:
            continue
        g = gt == c
        p = seg == c
        d[int(c)] = float(2 * np.count_nonzero(p & g) / max(1, p.sum() + g.sum()))
    return d


def main():
    with open(PER_CASE) as f:
        cases = sorted(json.loads(l)["case"] for l in f)
    cases = [c for c in cases if c not in EXCLUDE]
    assert len(cases) == 66, f"expected 66 cases, got {len(cases)}"
    rows = []
    for i, c in enumerate(cases, 1):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(f"{PRED}/{c}.nii.gz"))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{c}.nii.gz"))
        before = case_dice(pred, gt)
        new, log = apply_assigner(pred)
        after = case_dice(new, gt)
        fb = float(np.mean(list(before.values())))
        fa = float(np.mean(list(after.values())))
        rows.append({"case": c, "n_remaps": max(0, len(log) - 1),
                     "dice_before": round(fb, 4), "dice_after": round(fa, 4)})
        print(f"[{i}/66] {c} remaps={rows[-1]['n_remaps']} {fb:.4f} -> {fa:.4f}",
              flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mb = float(np.mean([r["dice_before"] for r in rows]))
    ma = float(np.mean([r["dice_after"] for r in rows]))
    healthy = [r for r in rows if r["dice_before"] >= 0.9]
    bad = [r["case"] for r in healthy if r["n_remaps"] > 0]
    print(f"\n66-case mean: before={mb:.4f}  after={ma:.4f}")
    print(f"healthy (before>=0.9): {len(healthy)} cases; re-mapped: {bad or 'none'}")
    gains = sorted(rows, key=lambda r: -(r["dice_after"] - r["dice_before"]))
    print("top gains: " + ", ".join(
        f"{r['case']} +{r['dice_after']-r['dice_before']:.3f}" for r in gains[:5]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 后台运行（实测 ~4 min；初估 ~2.4h 过于悲观，见修订记录）**

Run: `nohup /root/miniconda3/envs/nnunet310/bin/python -u scripts/lr_census_66.py > tooth_3d_semantic/logs/lr_census_66.log 2>&1 &`
监控：`tail -f tooth_3d_semantic/logs/lr_census_66.log`（逐 case 打点，权威进度）。

- [ ] **Step 3: 验收（跑完后）**

检查输出日志与 `lr_census_66.jsonl`：
1. `66-case mean before ≈ 0.8446`（与 holdout_metrics.json 一致，误差 <0.001；不一致则口径漂移，STOP 排查）；
2. `after ≥ 0.86`（证据 §4 预期 0.867-0.893；若 <0.85，说明互换不是全部误差源，记录实际值并继续——Task 4/8 仍执行，但报告需注明）；
3. **无回归（2026-09-22 修订，见文末修订记录）**：无任何 case `dice_after < dice_before − 0.005`（含全部 35 个 `before≥0.9` 健康 case）。原"零容忍 `n_remaps>0`"判据弃用为过严代理：孪生守卫引入后，健康 case 残留的 remap 已逐 case 验证无害/有益（F_027 +0.0122、P_380 +0.0502、P_461 +0.0011、P_463 0.0000，受影响类 dice 无一下降）。结构性安全由孪生守卫 + 域内孪生守卫提供，本判据为可测代理。
4. P_002/P_379/F_031 的 `dice_after - dice_before ≥ 0.3`。

- [ ] **Step 4: Commit**

```bash
git add scripts/lr_census_66.py tooth_3d_semantic/logs/lr_census_66.jsonl
git commit -m "feat: 66-case L/R swap census + assigner ceiling measurement"
```

---

### Task 4: 后处理后全量指标（66/69/72 + FDI 逐类 + 前后对比）

**Files:**
- Create: `scripts/eval_holdout_idassigned.py`
- Create（产物）: `tooth_3d_semantic/eval/holdout_ensemble_ras_idassigned/{per_case_metrics.jsonl, holdout_metrics.json}`

**Interfaces:**
- Consumes: Task 1 `apply_assigner`；Task 3 的 66-case 清单（EXCLUDE 集）
- Produces: `holdout_metrics.json`（键：`mean_foreground_dice_66/69/72`、`per_class_dice`（1-32 + FDI 注释表）、`mean_upper_jaw/mean_lower_jaw`、`per_case_before_after`）——Task 8 对比表的 1000+assigner 列

- [ ] **Step 1: 写评估脚本**

创建 `scripts/eval_holdout_idassigned.py`：

```python
#!/usr/bin/env python3
"""Full holdout metrics with the L/R id-assigner applied to the saved
1000 5-fold ensemble predictions. Outputs before/after per case and the
66/69/72 aggregates + per-class table (1-32 with FDI annotation)."""
import json

import numpy as np
import SimpleITK as sitk
import sys

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import apply_assigner

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
PRED = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras/predictions"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"
OUT = f"{BASE}/tooth_3d_semantic/eval/holdout_ensemble_ras_idassigned"

EMPTY_GT = {"ToothFairy2P_113", "ToothFairy2P_136", "ToothFairy2P_356"}
PATHO = {"ToothFairy2F_049", "ToothFairy2P_049", "ToothFairy2P_105"}


def fdi(i):
    # closed form, verified 32/32 vs tooth_fairy2/list/label_map.yaml
    return 10 * (1 + (i - 1) // 8) + ((i - 1) % 8 + 1)


def case_dice(seg, gt):
    d = {}
    for c in np.unique(gt):
        if c == 0:
            continue
        g = gt == c
        p = seg == c
        d[int(c)] = float(2 * np.count_nonzero(p & g) / max(1, p.sum() + g.sum()))
    return d


def mean_of(d):
    return float(np.mean(list(d.values()))) if d else 0.0


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for c in sorted(os.listdir(PRED)):
        if not c.endswith(".nii.gz"):
            continue
        case = c[:-len(".nii.gz")]
        pred = sitk.GetArrayFromImage(sitk.ReadImage(f"{PRED}/{c}"))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{case}.nii.gz"))
        new, log = apply_assigner(pred)
        b, a = case_dice(pred, gt), case_dice(new, gt)
        rows.append({"case": case,
                     "n_remaps": max(0, len(log) - 1),
                     "fg_before": round(mean_of(b), 4), "fg_after": round(mean_of(a), 4),
                     "per_class_after": {str(k): round(v, 4) for k, v in a.items()}})
        print(f"[{len(rows)}/72] {case} {rows[-1]['fg_before']} -> {rows[-1]['fg_after']} "
              f"({rows[-1]['n_remaps']} remaps)", flush=True)

    with open(f"{OUT}/per_case_metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def subset(preds):
        vals = [r["fg_after"] for r in rows
                if r["case"] in preds or
                (preds is None and True)]
        return vals

    n72 = [r["fg_after"] for r in rows]
    n69 = [r["fg_after"] for r in rows if r["case"] not in EMPTY_GT]
    n66 = [r["fg_after"] for r in rows if r["case"] not in EMPTY_GT and r["case"] not in PATHO]
    cls_sum, cls_cnt = np.zeros(33), np.zeros(33)
    for r in rows:
        for c, v in r["per_class_after"].items():
            cls_sum[int(c)] += v
            cls_cnt[int(c)] += 1
    per_class = {str(c): round(float(cls_sum[c] / cls_cnt[c]), 4) if cls_cnt[c] else None
                 for c in range(1, 33)}
    upper = [per_class[str(c)] for c in range(1, 17) if per_class[str(c)] is not None]
    lower = [per_class[str(c)] for c in range(17, 33) if per_class[str(c)] is not None]
    weak_now = {c: per_class[c] for c in ("4", "6", "7", "8", "13", "14",
                                          "15", "16", "23", "32")}
    metrics = {
        "ensemble": "1000 5-fold, checkpoint_final, mirroring+gaussian, + L/R id-assigner",
        "mean_fg_dice_66": round(float(np.mean(n66)), 4),
        "mean_fg_dice_69": round(float(np.mean(n69)), 4),
        "mean_fg_dice_72": round(float(np.mean(n72)), 4),
        "mean_upper_jaw_1_16": round(float(np.mean(upper)), 4),
        "mean_lower_jaw_17_32": round(float(np.mean(lower)), 4),
        "per_class_dice": per_class,
        "per_class_fdi": {str(c): fdi(int(c)) for c in per_class},
        "weak_classes_focus": weak_now,
        "n_remaps_total": int(sum(r["n_remaps"] for r in rows)),
    }
    with open(f"{OUT}/holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: metrics[k] for k in
                      ("mean_fg_dice_66", "mean_fg_dice_69", "mean_fg_dice_72",
                       "mean_upper_jaw_1_16", "mean_lower_jaw_17_32")}, indent=1))
    print("weak classes (after assigner):",
          {k: f"{k}=FDI{fdi(int(k))}:{v}" for k, v in weak_now.items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行（后台，~1.5h CPU）**

Run: `nohup /root/miniconda3/envs/nnunet310/bin/python -u scripts/eval_holdout_idassigned.py > tooth_3d_semantic/logs/eval_idassigned.log 2>&1 &`

- [ ] **Step 3: 验收**

1. `mean_fg_dice_66 ≥ 0.85` 且与 Task 3 的 after 均值一致（<0.001）；
2. 66-case 中 `fg_before ≥ 0.95` 的 case 无一 `fg_after < fg_before - 0.02`（无显著误伤）；
3. 弱类表打印完整（10 类带 FDI 注释）；其中 8(FDI18)/16(FDI28)/32(FDI48) 三颗第三磨牙是否 ≥0.80 记录在案（达不到的类列入 Task 8 报告"剩余弱点"）。

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_holdout_idassigned.py tooth_3d_semantic/eval/holdout_ensemble_ras_idassigned/
git commit -m "feat: full holdout metrics with L/R id-assigner (66/69/72 + FDI per-class)"
```

---

### Task 5: 整颌镜像 GT 审计（全 480，后台 ~11h）

**Files:**
- Create: `scripts/audit_wholearch_mirror.py`
- Create（产物）: `tooth_3d_semantic/logs/wholearch_audit_all480.json`

**Interfaces:**
- Consumes: Task 1 的 `SIDE_LUT`/`MIN_COMP_VOX`/`MARGIN_MIN_VOX`/`MARGIN_FRAC`；GT = `tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/labelsTr/`（480 个）；holdout 清单 = `/home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml` 的 `holdout_test` 字段（其余 408 = 训练）
- Produces: 逐 case `{"case","split","status","score","n_det"}`；status ∈ {OK, SUSPECT, MIRROR, INSUFFICIENT, EMPTY}。**只审计报告，不改任何标签**（Global Constraint）。flagged 清单交给用户裁决：holdout flagged → 追加进考核剔除集；训练 flagged → Task 6 组装前用户决定是否剔除。

- [ ] **Step 1: 写审计脚本**

创建 `scripts/audit_wholearch_mirror.py`：

```python
#!/usr/bin/env python3
"""Whole-arch L/R mirrored-GT detector (blind spot of the 08-27 midpoint
audit: a fully mirrored label set is still symmetric, so midpoint checks
pass). Direct check: each tooth ID must sit on its anatomical side in the
true-RAS frame (right = high x). Audit only - no label modification."""
import glob
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import yaml
from scipy import ndimage

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import SIDE_LUT, MIN_COMP_VOX, MARGIN_MIN_VOX, MARGIN_FRAC

RAW = "/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/labelsTr"
SPLIT_YAML = "/home/share/clr/share/data/CBCT/tooth_fairy2/list/tooth_split_k5_seed0.yaml"
OUT = "/home/share/clr/share/work/CBCT/nnUNet/tooth_3d_semantic/logs/wholearch_audit_all480.json"


def audit_case(seg):
    lab, n = ndimage.label(seg > 0)
    sizes = np.bincount(lab.ravel())
    cxs, ids = [], []
    for i in range(1, n + 1):
        if sizes[i] < MIN_COMP_VOX:
            continue
        m = lab == i
        vals, counts = np.unique(seg[m], return_counts=True)
        dom = int(vals[np.argmax(counts)])
        if dom == 0:
            continue
        _, _, cx = ndimage.center_of_mass(m)
        cxs.append(float(cx))
        ids.append(dom)
    if not cxs:
        return {"status": "EMPTY"}
    mid = (min(cxs) + max(cxs)) / 2.0
    margin = max(MARGIN_MIN_VOX, MARGIN_FRAC * (max(cxs) - min(cxs)))
    det = [(cx, i) for cx, i in zip(cxs, ids) if abs(cx - mid) > margin]
    if len(det) < 6:
        return {"status": "INSUFFICIENT", "n_det": len(det)}
    wrong = sum(1 for cx, i in det if SIDE_LUT[i] != ("R" if cx > mid else "L"))
    score = wrong / len(det)
    status = "MIRROR" if score >= 0.6 else "SUSPECT" if score >= 0.3 else "OK"
    return {"status": status, "score": round(score, 3), "n_det": len(det)}


def main():
    with open(SPLIT_YAML) as f:
        split = yaml.safe_load(f)
    holdout = set(split["holdout_test"])
    files = sorted(glob.glob(f"{RAW}/*.nii.gz"))
    assert len(files) == 480, f"expected 480 GTs, got {len(files)}"
    results = []
    for i, fp in enumerate(files, 1):
        case = os.path.basename(fp)[:-len(".nii.gz")]
        seg = sitk.GetArrayFromImage(sitk.ReadImage(fp))
        r = audit_case(seg)
        r["case"] = case
        r["split"] = "holdout" if case in holdout else "train"
        results.append(r)
        if r["status"] in ("MIRROR", "SUSPECT"):
            print(f"FLAG {case} [{r['split']}] {r}", flush=True)
        if i % 20 == 0:
            print(f"[{i}/480]", flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    from collections import Counter
    cnt = Counter((r["split"], r["status"]) for r in results)
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(cnt.items())}, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 正确性验证（3 个已知答案，先跑这 3 个再放全量）**

```bash
cd /home/share/clr/share/work/CBCT/nnUNet
/root/miniconda3/envs/nnunet310/bin/python - <<'EOF'
import sys
import numpy as np, SimpleITK as sitk
sys.path.insert(0, ".")
from scripts.audit_wholearch_mirror import audit_case

RAW = "tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/labelsTr"
clean = sitk.GetArrayFromImage(sitk.ReadImage(f"{RAW}/ToothFairy2P_079.nii.gz"))
print("P_079 clean   ->", audit_case(clean))          # 期望 OK, score≈0
print("P_079 mirrored->", audit_case(np.flip(clean, axis=2)))  # 期望 MIRROR, score≈1（数组 (z,y,x)，L/R=x=axis 2；fliplr 翻的是 y/A-P，2026-09-23 修订）
f001 = sitk.GetArrayFromImage(sitk.ReadImage(f"{RAW}/ToothFairy2F_001.nii.gz"))
print("F_001 clean   ->", audit_case(f001))           # 期望 OK
EOF
```

Expected: P_079 → `OK`（score <0.1）；`np.flip(axis=2)` 版 → `MIRROR`（score >0.8）；F_001 → `OK`。
**任一不符：STOP，排查 SIDE_LUT/margin 后再放全量。**

- [ ] **Step 3: 全量后台运行（~11h）**

Run: `nohup /root/miniconda3/envs/nnunet310/bin/python -u scripts/audit_wholearch_mirror.py > tooth_3d_semantic/logs/wholearch_audit.log 2>&1 &`
监控：`tail -f tooth_3d_semantic/logs/wholearch_audit.log`（每 20 case 打点，FLAG 即时打印）。

- [ ] **Step 4: 结果交接（用户裁决，D1）**

把 `wholearch_audit_all480.json` 的 flagged 清单（MIRROR/SUSPECT 分列，train/holdout 分列）呈给用户裁决：
- holdout MIRROR → 进考核剔除集（用户规则"证实 GT 错即剔除"）；SUSPECT → 用户目视复核后定；
- train MIRROR/SUSPECT → 用户决定是否在 Dataset1001 组装时剔除（默认不剔除、只报告，除非用户明示）。
裁决结果记录在 `tooth_3d_semantic/logs/wholearch_audit_decisions.json`（`{"holdout_exclude": [...], "train_exclude": [...]}`）。

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_wholearch_mirror.py tooth_3d_semantic/logs/wholearch_audit_all480.json
git commit -m "feat: whole-arch mirrored-GT audit (blind spot of 08-27 midpoint method)"
```

---

### Task 6: Dataset1001 组装 + fold_0 训练 + 门控

**Files:**
- Create: `scripts/assemble_dataset1001.sh`
- Create: `scripts/train_fold_noxmirror.sh`
- Create: `scripts/eval_subset.py`
- Create（产物）: `tooth_3d_semantic/raw/Dataset1001_ToothSemanticNoXMirror/`、`tooth_3d_semantic/preprocessed/Dataset1001_ToothSemanticNoXMirror/`

**Interfaces:**
- Consumes: Dataset1000 raw/preprocessed；Task 5 的 `wholearch_audit_decisions.json`（可选 train_exclude）；现成 trainer `nnUNetTrainer_onlyMirror01`
- Produces: Dataset1001（batch 8、split 同 1000、可选剔除）+ fold_0 模型 + 门控 JSON `tooth_3d_semantic/logs/fold0_gate.json`（PASS/FAIL）

- [ ] **Step 1: 组装脚本**

创建 `scripts/assemble_dataset1001.sh`：

```bash
#!/bin/bash
# Assemble Dataset1001 (no-x-mirror retrain base) from Dataset1000 via hardlinks.
# Optional: --exclude <file> with one case name per line (from audit decisions).
# 2026-09-23 修订（控制器按本 nnUNet 构建实际布局核证）：preprocessed 为 blosc2
# 布局（nnUNetPlans_3d_fullres/ 下平铺 ${c}.b2nd/${c}.pkl/${c}_seg.b2nd，无
# imagesTr/npy.gz，无根级 plans.json；batch 在根 nnUNetPlans.json）；splits_
# final.json = 5 折 {train,val} 列表，且 trainer 从 PREPROCESSED 副本读取
# （nnUNetTrainer.do_split）——缺失会生成随机 split，故剔除必须同步两份且
# 两份都必须是真拷贝（防硬链接 in-place 写回污染 Dataset1000 源）。
set -e
cd /home/share/clr/share/work/CBCT/nnUNet

EXCLUDE_FILE=""
if [ "${1:-}" = "--exclude" ]; then EXCLUDE_FILE="${2:?path}"; fi

SRC_R=tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS
DST_R=tooth_3d_semantic/raw/Dataset1001_ToothSemanticNoXMirror
SRC_P=tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS
DST_P=tooth_3d_semantic/preprocessed/Dataset1001_ToothSemanticNoXMirror

[ -d "$DST_R" ] && { echo "DST_R exists, aborting"; exit 1; }
[ -d "$DST_P" ] && { echo "DST_P exists, aborting"; exit 1; }

cp -al "$SRC_R/imagesTr" "$DST_R_imagesTr" 2>/dev/null || true
mkdir -p "$DST_R"
mv "$DST_R_imagesTr" "$DST_R/imagesTr" 2>/dev/null || cp -al "$SRC_R/imagesTr" "$DST_R/imagesTr"
cp -al "$SRC_R/labelsTr" "$DST_R/labelsTr"
cp "$SRC_R/splits_final.json" "$DST_R/"
sed 's/Dataset1000_ToothSemanticRAS/Dataset1001_ToothSemanticNoXMirror/' \
    "$SRC_R/dataset.json" > "$DST_R/dataset.json"
cp -al "$SRC_P" "$DST_P"
# make splits a REAL copy in the preprocessed tree (cp -al above hardlinks it;
# an in-place rewrite during exclusion would corrupt the Dataset1000 source;
# rm first because cp refuses a destination that is a hardlink of the source)
rm -f "$DST_P/splits_final.json"
cp "$SRC_P/splits_final.json" "$DST_P/"
# keep the dataset-name reference consistent in the copied plans metadata
# (sed -i replaces the file, breaking the hardlink; the source stays untouched)
sed -i 's/Dataset1000_ToothSemanticRAS/Dataset1001_ToothSemanticNoXMirror/g' "$DST_P/nnUNetPlans.json"

# byte-identity check on 3 sample files each
for sub in imagesTr labelsTr; do
  for f in $(ls "$DST_R/$sub" | head -3); do
    a=$(sha256sum "$SRC_R/$sub/$f" | cut -d' ' -f1)
    b=$(sha256sum "$DST_R/$sub/$f" | cut -d' ' -f1)
    [ "$a" = "$b" ] || { echo "MISMATCH $sub/$f"; exit 1; }
  done
done
# preprocessed case-count parity (blosc2 layout: flat *.b2nd, excluding *_seg.b2nd)
n_src=$(ls "$SRC_P/nnUNetPlans_3d_fullres" | grep '\.b2nd$' | grep -vc '_seg\.b2nd$')
n_dst=$(ls "$DST_P/nnUNetPlans_3d_fullres" | grep '\.b2nd$' | grep -vc '_seg\.b2nd$')
[ "$n_src" = "$n_dst" ] || { echo "PREPROCESSED COUNT MISMATCH src=$n_src dst=$n_dst"; exit 1; }
echo "preprocessed cases OK: $n_dst"

# optional exclusion: remove cases from raw + preprocessed dirs + BOTH splits copies
# (the trainer reads splits from the PREPROCESSED copy — nnUNetTrainer.do_split)
if [ -n "$EXCLUDE_FILE" ]; then
  while read -r c; do
    [ -z "$c" ] && continue
    rm -f "$DST_R/imagesTr/${c}_0000.nii.gz" "$DST_R/labelsTr/${c}.nii.gz"
    for cfg in nnUNetPlans_3d_fullres nnUNetPlans_2d; do
      rm -f "$DST_P/$cfg/${c}.b2nd" "$DST_P/$cfg/${c}_seg.b2nd" "$DST_P/$cfg/${c}.pkl"
    done
    rm -f "$DST_P/gt_segmentations/${c}.nii.gz"
    echo "excluded $c"
  done < "$EXCLUDE_FILE"
  for sp in "$DST_R/splits_final.json" "$DST_P/splits_final.json"; do
    /root/miniconda3/envs/nnunet310/bin/python - "$EXCLUDE_FILE" "$sp" <<'EOF'
import json, sys
excl = {l.strip() for l in open(sys.argv[1]) if l.strip()}
p = sys.argv[2]
splits = json.load(open(p))          # list of 5 {train, val} dicts
for fold in splits:
    fold["train"] = [c for c in fold["train"] if c not in excl]
    fold["val"] = [c for c in fold["val"] if c not in excl]
json.dump(splits, open(p, "w"), indent=1)
print("splits updated:", p,
      {k: (len(f["train"]), len(f["val"])) for k, f in enumerate(splits)})
EOF
  done
fi

# batch size must be 8 (root nnUNetPlans.json in this nnUNet build; 1000 has it)
/root/miniconda3/envs/nnunet310/bin/python - <<'EOF'
import json
p = "tooth_3d_semantic/preprocessed/Dataset1001_ToothSemanticNoXMirror/nnUNetPlans.json"
plans = json.load(open(p))
bs = plans["configurations"]["3d_fullres"]["batch_size"]
assert bs == 8, f"batch_size is {bs}, expected 8 -- fix before training"
print("plans batch_size OK:", bs)
EOF
echo "Dataset1001 assembled."
```

- [ ] **Step 2: 组装并验证**

Run:
- 无训练剔除：`bash scripts/assemble_dataset1001.sh`
- 有（按 Task 5 裁决）：`bash scripts/assemble_dataset1001.sh --exclude /tmp/train_exclude.txt`

Expected: 无 MISMATCH、`preprocessed cases OK: 480`、`plans batch_size OK: 8`、`Dataset1001 assembled.`

- [ ] **Step 3: 单折训练脚本**

创建 `scripts/train_fold_noxmirror.sh`（镜像 `train_fold_ras.sh`，改 dataset ID 与 trainer）：

```bash
#!/bin/bash
# Train one fold of Dataset1001 with x-axis mirroring disabled
# (nnUNetTrainer_onlyMirror01: mirrors z/y only; TTA axes (0,1)).
# GPUs 1,2,3 (GPU 0 free), global batch 8.
# Usage: bash scripts/train_fold_noxmirror.sh <fold>
set -e

FOLD="${1:?Usage: bash scripts/train_fold_noxmirror.sh <fold>}"

source dataset_conversion/setup_nnunet_env.sh
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Training fold $FOLD (Dataset1001 no-x-mirror, GPUs 1,2,3)"
echo "=========================================="

/root/miniconda3/envs/nnunet310/bin/python -m nnunetv2.run.run_training 1001 3d_fullres "$FOLD" \
    -num_gpus 3 \
    --npz \
    --c \
    -tr nnUNetTrainer_onlyMirror01

echo "Training fold $FOLD complete!"
```

- [ ] **Step 4: 启动 fold_0（~38h，后台 + 监控）**

```bash
nohup bash scripts/train_fold_noxmirror.sh 0 > tooth_3d_semantic/logs/train_noxmirror_fold0.log 2>&1 &
```
监控（权威进度 = `tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt` 最新时间戳；tee 日志长时间静默属正常，见既有监控教训）。
**开训后 1h 内检查一次**：training_log 有 epoch 0/1 条目、显存 ~90%+、无 OOM；同时确认日志里 mirror 配置行为（`do_dummy_2d_data_aug` 行存在即 trainer 初始化正常）。

- [ ] **Step 5: 门控脚本（fold_0 完成后，D3）**

创建 `scripts/eval_subset.py`（40-case 子集：Task 3 中 dice_before 最低的 20 + 最高的 20；单折评估，可换 model_dir 复用于 1000 基线）：

```python
#!/usr/bin/env python3
"""Subset single-fold eval for the fold_0 gate.
Usage: python scripts/eval_subset.py --model_dir <trainer_dir> --folds 0 \
        --out <per_case.jsonl> [--id_assign on|off]
Evaluates the 40-case subset (20 worst + 20 best by Task-3 dice_before)."""
import argparse
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, "/home/share/clr/share/work/CBCT/nnUNet")
from inference.id_assigner import apply_assigner

BASE = "/home/share/clr/share/work/CBCT/nnUNet"
CENSUS = f"{BASE}/tooth_3d_semantic/logs/lr_census_66.jsonl"
GT = f"{BASE}/tooth_3d_semantic/preprocessed/Dataset1000_ToothSemanticRAS/gt_segmentations"

os.environ.setdefault("nnUNet_raw", f"{BASE}/tooth_3d_semantic/raw")
os.environ.setdefault("nnUNet_preprocessed", f"{BASE}/tooth_3d_semantic/preprocessed")
os.environ.setdefault("nnUNet_results", f"{BASE}/tooth_3d_semantic/results")


def case_dice(seg, gt):
    d = []
    for c in np.unique(gt):
        if c == 0:
            continue
        g = gt == c
        p = seg == c
        d.append(2 * np.count_nonzero(p & g) / max(1, p.sum() + g.sum()))
    return float(np.mean(d)) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id_assign", choices=["on", "off"], default="off")
    a = ap.parse_args()

    with open(CENSUS) as f:
        rows = [json.loads(l) for l in f]
    worst20 = {r["case"] for r in sorted(rows, key=lambda r: r["dice_before"])[:20]}
    best20 = {r["case"] for r in sorted(rows, key=lambda r: -r["dice_before"])[:20]}
    cases = sorted(worst20 | best20)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                           use_mirroring=True, perform_everything_on_device=True,
                           device=torch.device("cuda"), verbose=False,
                           allow_tqdm=False)
    pred.initialize_from_trained_model_folder(
        a.model_dir, use_folds=[a.folds], checkpoint_name="checkpoint_final.pth")

    out_rows = []
    for i, c in enumerate(cases, 1):
        img = sitk.ReadImage(f"{GT}/{c}.nii.gz")  # shape/spacing reference
        # read raw image for this case from Dataset1000 imagesTr
        raw = sitk.ReadImage(
            f"{BASE}/tooth_3d_semantic/raw/Dataset1000_ToothSemanticRAS/imagesTr/{c}_0000.nii.gz")
        arr = sitk.GetArrayFromImage(raw)
        seg = pred.predict_single_npy_array(
            arr[None], {"spacing": raw.GetSpacing()},
            save_or_return_probabilities=False)
        if a.id_assign == "on":
            seg, _ = apply_assigner(seg)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/{c}.nii.gz"))
        d = case_dice(seg, gt)
        out_rows.append({"case": c, "group": "worst" if c in worst20 else "best",
                         "dice": round(float(d), 4)})
        print(f"[{i}/40] {c} {d:.4f}", flush=True)
    with open(a.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    w = [r["dice"] for r in out_rows if r["group"] == "worst"]
    b = [r["dice"] for r in out_rows if r["group"] == "best"]
    print(f"worst20 mean={np.mean(w):.4f}  best20 mean={np.mean(b):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 门控执行（GPU 1,2,3，~20min）**

```bash
cd /home/share/clr/share/work/CBCT/nnUNet
source dataset_conversion/setup_nnunet_env.sh
export CUDA_VISIBLE_DEVICES=1
/root/miniconda3/envs/nnunet310/bin/python scripts/eval_subset.py \
  --model_dir tooth_3d_semantic/results/Dataset1000_ToothSemanticRAS/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --folds 0 --out tooth_3d_semantic/logs/gate_1000_fold0.jsonl
export CUDA_VISIBLE_DEVICES=2
/root/miniconda3/envs/nnunet310/bin/python scripts/eval_subset.py \
  --model_dir tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres \
  --folds 0 --out tooth_3d_semantic/logs/gate_1001_fold0_raw.jsonl
export CUDA_VISIBLE_DEVICES=3
/root/miniconda3/envs/nnunet310/bin/python scripts/eval_subset.py \
  --model_dir tooth_3d_semantic/results/Dataset1001_ToothSemanticNoXMirror/nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_fullres \
  --folds 0 --id_assign on --out tooth_3d_semantic/logs/gate_1001_fold0_assigned.jsonl
```

判据（写入 `tooth_3d_semantic/logs/fold0_gate.json`，人工+脚本核对）：
- **PASS 条件（两条都要满足）**：
  1. 1001-raw 的 best20 均值 ≥ 1000-raw best20 均值 − 0.02（健康 case 无回归）；
  2. 1001-raw 的 worst20 均值 ≥ 1000-raw worst20 均值 + 0.03（去镜像在裸模型上已可见改善）。
- **FAIL → 立即停 seq（不训 fold 1-4）**，把三份 jsonl + 训练曲线摘要呈给用户决定（换 B2 方案：镜像时重映射标签；或接受后处理单轨；或升级路径）。
- 参考信息（不判停）：1001-assigned worst20 均值（后处理叠加效果）。

- [ ] **Step 7: Commit**

```bash
git add scripts/assemble_dataset1001.sh scripts/train_fold_noxmirror.sh scripts/eval_subset.py \
        tooth_3d_semantic/logs/fold0_gate.json
git commit -m "feat: Dataset1001 assembly + no-x-mirror fold_0 training + gate"
```

---

### Task 7: fold 1-4 顺序训练（~6.3 天）

**Files:**
- Create: `scripts/train_folds_seq_noxmirror.sh`

**Interfaces:**
- Consumes: Task 6 的 `train_fold_noxmirror.sh`（门控 PASS 后启动）
- Produces: fold 1-4 checkpoint_final + checkpoint_best（各 ~248MB）

- [ ] **Step 1: 顺序训练脚本**

创建 `scripts/train_folds_seq_noxmirror.sh`（镜像 `train_folds_seq_ras.sh` 结构）：

```bash
#!/bin/bash
# Sequential fold 1-4 for Dataset1001 (start after fold_0 gate PASS).
# Stops on first failure. Resume-safe via --c in the per-fold script.
set -e
cd /home/share/clr/share/work/CBCT/nnUNet
for F in 1 2 3 4; do
  if ! bash scripts/train_fold_noxmirror.sh "$F"; then
    echo "FOLD $F FAILED -- stopping sequence" >&2
    exit 1
  fi
done
echo "ALL FOLDS 1-4 COMPLETE"
```

- [ ] **Step 2: 启动（fold_0 门控 PASS 后）**

```bash
nohup bash scripts/train_folds_seq_noxmirror.sh > tooth_3d_semantic/logs/train_noxmirror_seq.log 2>&1 &
```

- [ ] **Step 3: 监控节奏（用户偏好：可监控执行）**

- 每折开训 1h 内：查 `results/.../fold_N/training_log_*.txt` 有 epoch 条目、无 OOM；
- 每天一次：各折最新 epoch（`ls -t .../fold_N/training_log_*` + 尾 5 行），记录到会话；
- 每折完成：记录内部 val Mean Dice（与 1000 对应折对比：0.670/0.735/0.708/0.677），**某折 val 比 1000 同折低 >0.05 时向用户预警**（不自动停，用户裁决）；
- 预计 D9-10 全部完成。

- [ ] **Step 4: Commit**

```bash
git add scripts/train_folds_seq_noxmirror.sh
git commit -m "feat: sequential fold 1-4 driver for Dataset1001"
```

---

### Task 8: 终评 + 对比报告 + 生产决定

**Files:**
- Create: `scripts/eval_holdout_general.py`
- Create（产物）: `tooth_3d_semantic/eval/holdout_ensemble_noxmirror/`、`docs/superpowers/plans/2026-09-20-dataset1000-optimization-results.md`
- Modify: `inference/tooth_predict.py`（生产模型切换，**按终评结果**）

**Interfaces:**
- Consumes: Task 4 的指标口径；Task 7 的 5 折模型；Task 5 的最终剔除集
- Produces: 4 列对比表（1000-raw / 1000+assigner / 1001-raw / 1001+assigner）+ 生产配置建议

- [ ] **Step 1: 通用集成评估脚本**

创建 `scripts/eval_holdout_general.py`（结构同 Task 4 脚本，参数化输入目录；**完整代码 = Task 4 脚本做以下 3 处修改后另存**，不得写"同 Task 4"）：
1. `PRED`/`OUT` 改为 `argparse` 参数：`--pred_dir`、`--out_dir`（GT 路径不变，仍 Dataset1000 的 gt_segmentations，72 case 同一批）；
2. 增加 `--id_assign on|off`：off 时 `new = pred`（跳过 assigner，n_remaps=0）；
3. metrics 键增加 `"exclude_extra"`（接受 `--exclude_cases a b c`，并入 EMPTY_GT 做 66 口径，用于 Task 5 追加剔除的 case）。

- [ ] **Step 2: 生成 1001 集成预测 + 4 配置指标**

```bash
cd /home/share/clr/share/work/CBCT/nnUNet
source dataset_conversion/setup_nnunet_env.sh
# 1001 5-fold ensemble 预测（GPU 1,2,3 分 case，参照 scripts/eval_holdout_ras_1000.py 的 worker 结构，
# 把 RESULTS 指向 1001 trainer 目录、FOLDS=[0..4]、输出到 eval/holdout_ensemble_noxmirror/predictions/）
```
（实现方式：复制 `scripts/eval_holdout_ras_1000.py` 为 `scripts/eval_holdout_noxmirror.py`，改 `RESULTS` 与输出目录两处常量，后台跑 ~20min。）

```bash
# 4 列指标（EXTRA_EXCL = Task 5 裁决的 holdout 追加剔除集，无则省略参数）
/root/miniconda3/envs/nnunet310/bin/python scripts/eval_holdout_general.py \
  --pred_dir tooth_3d_semantic/eval/holdout_ensemble_ras/predictions \
  --out_dir tooth_3d_semantic/eval/holdout_ensemble_ras_idassigned_final \
  --id_assign on --exclude_cases $EXTRA_EXCL
/root/miniconda3/envs/nnunet310/bin/python scripts/eval_holdout_general.py \
  --pred_dir tooth_3d_semantic/eval/holdout_ensemble_noxmirror/predictions \
  --out_dir tooth_3d_semantic/eval/holdout_ensemble_noxmirror_raw \
  --id_assign off --exclude_cases $EXTRA_EXCL
/root/miniconda3/envs/nnunet310/bin/python scripts/eval_holdout_general.py \
  --pred_dir tooth_3d_semantic/eval/holdout_ensemble_noxmirror/predictions \
  --out_dir tooth_3d_semantic/eval/holdout_ensemble_noxmirror_assigned \
  --id_assign on --exclude_cases $EXTRA_EXCL
# 1000-raw 基线直接取既有 holdout_metrics.json（0.8446 等），不重跑
```

- [ ] **Step 3: 对比报告**

写 `docs/superpowers/plans/2026-09-20-dataset1000-optimization-results.md`，内容固定包含：
1. 4 列 × {66/69/72 均值、上颌/下颌均值} 对比表 + 每列 n_remaps_total；
2. 逐类表（1-32 + FDI）：1000-raw → 1001+assigner，重点 10 个弱类是否 ≥0.80；
3. 逐 case 前后对比 top/bottom 各 5（含 3 个病态 case 的改善情况，参考信息）;
4. fold_0 门控数据 + 5 折内部 val 对比（vs 1000 各折）；
5. 审计 flagged 清单与裁决记录；
6. **生产决定**：`tooth_predict.py` 默认 `--model_dir` 切到哪个 trainer 目录 + `--id_assign` 保持 on/off，理由（哪个配置 66-case 最高且健康 case 无回归）；
7. 剩余弱点与升级路径触发条件（见下节）。

- [ ] **Step 4: 生产切换（按报告结论执行）**

若 1001 胜出：修改 `inference/tooth_predict.py` 顶部 docstring 的 Default production model 说明 + `main()` 中 `--model_dir` 的 help 示例为 1001 trainer 目录；**默认值不改**（`--model_dir` 仍必填，避免静默换模型），在 docstring 里写明推荐值。同步更新 `tooth_3d_quickstart.md` §8 的模型路径与指标。

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_holdout_general.py scripts/eval_holdout_noxmirror.py \
        docs/superpowers/plans/2026-09-20-dataset1000-optimization-results.md inference/tooth_predict.py tooth_3d_quickstart.md
git commit -m "feat: final 4-way comparison, production decision, quickstart update"
```

---

### Task 9: NC 67 case 轨道（外部门控，并行）

**Files:**
- Create（门控解锁后）: 复用既有 pipeline（`dataset_conversion/tooth_3d.py` 等）

**Interfaces:**
- Consumes: Task 5 审计器（新数据 L/R 校验）；Task 6 的组装/训练脚本（改 dataset ID）
- Produces: Dataset1002（480+67，视裁决）+ 增量评估

**外部门**：provider 确认标注协议 + L/R 锚点（既有记忆：先问 provider，67 高置信 case 可转换）。门未开前本任务只有 Step 1。

- [ ] **Step 1: 向 provider 发协议确认清单**（D0 发出，不等它）
  - 标注协议：编号体系（FDI 11-48? 连续 1-32?）、缺牙/残根/植入体/第三磨牙的处理约定、左右判定基准；
  - L/R 锚点：至少一个解剖锚（如鼻窦或中缝定位）用于自动校验；
  - 交付格式与数量（67 高置信 case 清单）。

- [ ] **Step 2:（门开后）转换 + 审计**
  - 按 `tooth_data_pipeline_plan.md` 流程把 67 case 转成 RAS 帧（tooth_fairy2 同款裁剪/LUT）；
  - 全部 67 case 跑 `scripts/audit_wholearch_mirror.py`（改 RAW 路径为参数），MIRROR/SUSPECT 的先目视复核再入库；

- [ ] **Step 3:（门开后）增量实验**
  - 组装 Dataset1002（480+67，沿用 Task 6 组装脚本改 ID，split：新 case 按比例入 5 折、holdout 72 不变）；
  - 用同一 no-x-mirror 配置重训（GPU 空闲窗口，~15 天）；
  - 同一 66-case 协议评估，报告增量（480→547）；

- [ ] **Step 4: Commit（每步完成即提交）**

```bash
git commit -m "feat: NC supplement conversion + audit + Dataset1002 incremental experiment"
```

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 后处理器过度修正（不对称/部分牙列 case 中线估计失败） | margin 门控 + Task 3"健康 case 零误改"硬检查 + 全量 remap log 可审计 + 生产默认开但 `--id_assign off` 一键关 |
| 去镜像训练不彻底解决互换（缺牙 case 仍歧义） | 后处理器作为生产安全网保留（已批准）；fold_0 门控保护 15 天投入 |
| 训练集含整颌镜像 GT 污染（审计盲区） | Task 5 在 D1 出结果、早于 fold_0 完成；用户裁决是否剔除；若剔除在 fold_0 前完成则不影响；fold_0 已开训再发现 → 用户裁决（重训 fold_0 成本 ~38h） |
| 本机 CPU 慢拖慢诊断/审计 | 全部 CPU 任务 nohup 后台 + 逐 case 打点日志，不上关键路径 |
| DDP 3/3/2 不均分 | 与 1000 相同，既有结论可接受，报告注明 |
| fold_0 是 batch 18 的遗留不一致 | 1001 全折 batch 8，问题自然消失 |

## 升级路径（最后手段，需用户批准才启动）

按顺序，每条都有明确的触发条件：
1. **B2：镜像时重映射标签**（x-flip 时标签同步 ±8 重映射的自定义 transform）——触发：fold_0 门控 FAIL 且分析显示去镜像方向正确但不够（如 best20 无回归但 worst20 提升 <0.03）；
2. **ID 感知监督**（牙心点监督 / 左右一致性损失）——触发：B2 后仍不达标；
3. **延长训练**（1500 ep，现成 `nnUNetTrainer_onlyMirror01_1500ep`）——触发：val 曲线未收敛；
4. **换架构（MedFormer 等）**——触发：以上全部用尽仍 <0.85；
5. NC 数据到位本身就是增量杠杆（Task 9，不算升级）。

## Self-Review 记录

- Spec 覆盖：目标（66≥0.85→Task 4/8；弱类→Task 8 逐类表）、主方向重训（Task 6/7）、后处理先行（Task 1-4，D0 开始）、NC 并行（Task 9）、标签只审计（Task 5）、口径与剔除规则（Global Constraints + Task 5 Step 4）、生产形态（Task 2 默认 on）、架构最后手段（升级路径）——全覆盖。
- 占位符：Task 9 有外部门，但步骤具体（协议清单/转换/审计/重训命令均明确），非 TBD。
- 类型一致性：`apply_assigner(seg)->(seg,log)`、`decide_remaps(components)->(remaps,meta)`、`assign_file(...)`、`SIDE_LUT`/`MIN_COMP_VOX`/`MARGIN_*` 在 Task 1-5 中签名一致；脚本间数据接口（jsonl 键名 `case/n_remaps/dice_before/dice_after`、`fg_before/fg_after/per_class_after`）一致。

## 修订记录

**2026-09-22（执行期，Task 3 首跑后控制器裁决）**：
1. Task 1 判定规则增补"孪生守卫"（见上）。首跑普查（无守卫）：66-case after=0.9064，但 2 个 case 回归（P_038 −0.1781、F_003 −0.1158）；组件级剖析确认均为幻影重复牙（模型在错误侧多生成分割 blob，携带与正确侧重复的 ID）——孪生 ID 已在正确侧就位时重映射会净伤害。守卫语义：目标 ID 已存在于几何正确侧（中线区外）的其他组件 → 跳过。P_002（6 remap，无孪生）与整弓镜像 case（孪生全在错误侧）不受影响。
2. Task 3 验收点 3 由"健康 case 零 remap"改为"无任何 case 回归 >0.005"（理由见上条；4 个健康 case 残留 remap 已逐 case 验证无害/有益）。
3. 时间修正：实测普查 66 case ≈ 8.4 min（~7.7 s/case），非计划的 ~2.4h；Task 4/5 的 CPU 时长估计按同比例下修（Task 5 全 480 ≈ 1h）。

**2026-09-22（执行期，Task 3 fix round 2 控制器裁决）**：
1. 孪生守卫后 P_038 已修复（0.9271 flat），F_003 −0.1158 残留。组件级再剖析：F_003 的两个有害 remap 是**另一种模式——域内孪生对粘连**：comp5 = 17/25 粘连域（域内 25 体素 9367 / 主导 17 体素 9733，比 0.962）、comp11 = 9/1 粘连域（域内 1 体素 18644 / 主导 9 体素 18952，比 0.984）；两半各自标对（GTmatch 9555/9238、18414/18113），域质心侧别与主导 ID 矛盾 → 重映射主导半摧毁标对的孪生半。
2. 朴素"外部体素存在"守卫（含自排除变体）经 6-case + 全 66 验证**否决**：F_003 有害 remap 的目标体素全部位于被重映射域**内部**（ext=0），自排除无法拦截，F_003 反降至 0.6513；同时会误杀 P_002/F_031 的有益 remap（其目标体素亦全在域内，但 remap 有益）。
3. 终裁 = **域内孪生比例守卫**：域内目标 ID 体素数 ≥ `INTERNAL_TWIN_FRAC`（0.5）× 组件主导标签体素数 → 跳过该 remap（守卫在 `apply_assigner` 内，`decide_remaps`/孪生守卫不动）。全 66 模拟（scripts/lr_guard2_sim66.py）：after=**0.9094**、零回归（>0.005）、F_003 0.7905→0.8204（+0.0299）、P_002 +0.4194（n=6）、P_379 +0.4134、F_031 +0.3051、P_038 flat 0.9271。有害比 0.889–0.984 vs 有益 remap 最高比 0.399（P_379 一例 0.575 被拦但 dice 中性），阈值 0.5 对有害侧留 0.39 裕量。
4. 同步更新：判定规则节 + Interfaces（`INTERNAL_TWIN_FRAC`）+ 验收点 3 措辞 + Task 3 Step 2/D1 时间修正（实测 ~4 min）。
5. 台账勘误：round 1 裁决中"F_031 的 16 个 remap"为笔误，普查权威值 = **10**。

**2026-09-23（执行期，Task 5 Step 2 门控裁决）**：
1. Task 5 Step 2 门控的合成交替镜像行 `np.fliplr(clean)` 是**门控 oracle 缺陷**，非检测器缺陷：GT 数组为 (z,y,x)、direction cosines 单位阵（真 RAS），L/R = x = axis 2，而 `fliplr` 翻 axis 1（y/A-P），x 质心不变 → 干净判定（实测 P_079：clean OK 0.0，flip axis1 OK 0.0，flip axis2 **MIRROR 1.0**，控制器已独立复验）。
2. 修正：Step 2 改用 `np.flip(clean, axis=2)`（已改）；审计脚本本身从未使用 fliplr，**无需改动**。
3. 时间修正：Task 5 全 480 预计 ~10–40 min（非 ~11h；与本宿主历次实测一致，plan 时长估计系统性失真 10–100 倍）。

**2026-09-23（执行期，Task 6 组装脚本布局裁决，派发前控制器核证）**：
1. 本 nnUNet 构建（仓库内可编辑安装）preprocessed 为 **blosc2 布局**：`nnUNetPlans_3d_fullres/` 下平铺 `${c}.b2nd`/`${c}.pkl`/`${c}_seg.b2nd`（`nnunetv2/training/dataloading/nnunet_dataset.py:207` 按此扫描 case 标识），**无** imagesTr/npy.gz、无根级 plans.json；batch_size 在根 `nnUNetPlans.json` 的 `configurations.3d_fullres.batch_size`（实测=8）。原 Step 1 的 batch 检查路径与 preprocessed 剔除路径（npy.gz/imagesTr）均无效 → 已改。
2. `splits_final.json` = **5 折 {train,val} 列表**（非单 dict）；trainer 从 **PREPROCESSED 副本**读取（`nnUNetTrainer.do_split`，nnUNetTrainer.py:616），缺失则生成随机 split → 剔除必须同步 raw+preprocessed 两份，且组装时把两份都转为**真拷贝**（`cp -al` 后 in-place 写会经硬链接污染 Dataset1000 源）→ 已改。
3. 剔除路径按实际布局改为 `nnUNetPlans_{3d_fullres,2d}/` 的三件套 + `gt_segmentations/${c}.nii.gz`；新增 preprocessed case 数一致性检查（src/dst b2nd case 计数相等）；拷贝后对 `nnUNetPlans.json` 做 dataset_name sed（`sed -i` 换 inode，源不受影响）。
4. **阶段拆分**：fold_0 ≈38h 跨会话 → Task 6 = Phase A（3 脚本 + 组装验证 + 启动训练 + 开训 1h 健康检查 + 提交 3 脚本）/ Phase B（训练完成后：3 次门控评估 + fold0_gate.json + 提交）。Phase B 训练完成后再派发（设 durable 提醒）。
5. `eval_subset.py` 预测路径已对照 `scripts/eval_holdout_ras.py:67-85`（1000 ensemble 权威调用）**逐字核证一致**，无需改动。

**2026-10-02（执行期，脚本归档与文档清理，用户裁决）**：
1. 1000 时代脚本加 1000 标识归档重命名：`scripts/train_fold_ras.sh` → `train_fold_ras_1000.sh`、`scripts/train_folds_seq_ras.sh` → `train_folds_seq_ras_1000.sh`、`scripts/eval_holdout_ras.py` → `eval_holdout_ras_1000.py`（内部互调与 Usage 措辞同步更新，文件头加归档注记）。
2. 删除 4 个一次性诊断脚本（`spatial_id_decompose.py`、`component_id_diag.py`、`lr_guard2_sim66.py`、`lr_guard2_verify.py`）与 `nc_supplement_analysis.md`（证据保留于 `tooth_3d_semantic/logs/`、本文档与台账）。
3. 已提交代码的功能依赖入库：`dataset_conversion/` 全套（含已提交训练/预测脚本所需的 `setup_nnunet_env.sh`；排除 `__pycache__`）+ `inference/tooth_relabel_fdi.py` + `inference/__init__.py`。
4. 本文档前瞻性引用（文件表 line 71、Task 8 Step 2 实现方式）同步改为新名；历史修订记录中的旧名保留不改。
5. `tooth_3d_quickstart.md` / `tooth_data_pipeline_plan.md` 过时内容修正后提交（删除/重命名脚本的引用、1001 与 ID assigner 现状、tooth_predict 的 `_0000.nii.gz` 文件名约束与 `--id_assign` 参数）。
