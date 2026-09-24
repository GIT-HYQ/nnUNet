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
