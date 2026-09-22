import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.id_assigner import (SIDE_LUT, apply_assigner, decide_remaps, swap_lr)


def _seg_two_teeth(left_id, right_id, left_x=(10, 20), right_x=(81, 91)):
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


def test_twin_guard_blocks_phantom_duplicate():
    # 17 correctly on the left; a phantom 17 duplicate on the right; 25
    # correctly on the right. The phantom must NOT be remapped to 25 (the
    # twin is already on the correct side) — remapping would destroy the
    # correct 17 and dilute the 25.
    comps = [{"id": 17, "cx": 20.0}, {"id": 17, "cx": 80.0}, {"id": 25, "cx": 85.0}]
    remaps, _ = decide_remaps(comps)
    assert [new for _, new in remaps] == [17, 17, 25]


def test_twin_guard_allows_full_swap():
    # Whole-arch swap: each twin sits on the WRONG side, so the guard must
    # not block the remaps.
    comps = [{"id": 25, "cx": 20.0}, {"id": 17, "cx": 80.0}]
    remaps, _ = decide_remaps(comps)
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
