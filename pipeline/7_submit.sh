#!/usr/bin/env bash
# Package a checkpoint with the v76 test-time settings, upload the Kaggle datasets (new, or a new version), push the
# notebook (Kaggle runs it on the test set), and with SUBMIT=1 submit that run (docs/submission.md).
#   bash pipeline/7_submit.sh "<version message>" <checkpoint.pt>
# Environment: KNEE_KAGGLE_OWNER (your Kaggle user; default houjinghuang), KNEE_KAGGLE_PREFIX (the models dataset
# <prefix>-models; default rsna-knee-v76), KNEE_KAGGLE_CODE_PREFIX (the code and wheels datasets <code prefix>-code,
# -wheels; default the prefix; when it differs, these two are shared with another notebook and not uploaded again),
# KNEE_KAGGLE_KERNEL (notebook slug; default rsna-knee-v76-inference), PUBLIC=1 (new datasets and the notebook public;
# default private), SUBMIT=1. Our ablation D notebook, for example:
#   KNEE_KAGGLE_PREFIX=rsna-knee-ablation-d KNEE_KAGGLE_CODE_PREFIX=rsna-knee-v76 KNEE_KAGGLE_KERNEL=rsna-knee-ablation-d-inference \
#       PUBLIC=1 bash pipeline/7_submit.sh "ablation D" data/models/ablation_D_r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48_nojoint.pt
set -euo pipefail
cd "$(dirname "$0")/.."
MSG=$1; CKPT=$2
OWNER=${KNEE_KAGGLE_OWNER:-houjinghuang}
PREFIX=${KNEE_KAGGLE_PREFIX:-rsna-knee-v76}
CODE_PREFIX=${KNEE_KAGGLE_CODE_PREFIX:-$PREFIX}
KERNEL=${KNEE_KAGGLE_KERNEL:-rsna-knee-v76-inference}
K=.venv/bin/kaggle
COMP=rsna-knee-abnormality-detection
ls kaggle/datasets/knee-wheels/*.whl >/dev/null 2>&1 || bash pipeline/0_wheels.sh
# 1. the package: 16 windows per series, one view, 12 file threads, 2 GPUs x 6 loader workers
KNEE_KAGGLE_OWNER=$OWNER KNEE_KAGGLE_PREFIX=$PREFIX KNEE_KAGGLE_CODE_PREFIX=$CODE_PREFIX .venv/bin/python -m knee.submission.package \
    --models "$CKPT" --fp16 --k_slices 16 --threads 12 --workers 6 --gpus 2 --no_tta_mirror
# 2. the notebook metadata for this account
.venv/bin/python - "$OWNER" "$PREFIX" "$KERNEL" "${PUBLIC:-}" "$CODE_PREFIX" <<'PY'
import json, sys
owner, prefix, kernel, public, code_prefix = sys.argv[1:6]
p = 'kaggle/notebooks/knee-infer/kernel-metadata.json'
m = json.load(open(p))
m.update(id=f'{owner}/{kernel}', title=kernel, is_private=not public,
         dataset_sources=[f'{owner}/{code_prefix}-wheels', f'{owner}/{code_prefix}-code', f'{owner}/{prefix}-models'])
json.dump(m, open(p, 'w'), indent=2)
print(p, m['id'], 'private' if m['is_private'] else 'public', m['dataset_sources'])
PY
# 3. upload: a new dataset (public with PUBLIC=1) or a new version of an existing one; shared code and wheels datasets
#    (KNEE_KAGGLE_CODE_PREFIX differs from the prefix) are left as they are
PARTS="wheels code models"; [ "$CODE_PREFIX" = "$PREFIX" ] || PARTS=models
for part in $PARTS; do
    dir=kaggle/datasets/knee-$part; slug=$OWNER/$PREFIX-$part; [ $part = models ] || slug=$OWNER/$CODE_PREFIX-$part
    if $K datasets status "$slug" >/dev/null 2>&1; then
        $K datasets version -p "$dir" -m "$MSG" --dir-mode zip -q
        $K datasets metadata --update -p "$dir" "$slug" >/dev/null   # a new version keeps the old licence and description
    else
        $K datasets create -p "$dir" --dir-mode zip -q ${PUBLIC:+--public}
    fi
done
sleep 30                        # the status can still read "ready" for the previous version just after an upload
for part in $PARTS; do
    slug=$OWNER/$PREFIX-$part; [ $part = models ] || slug=$OWNER/$CODE_PREFIX-$part
    for i in $(seq 1 60); do
        s=$($K datasets status "$slug" 2>/dev/null | tail -1 || true)
        [ "$s" = "ready" ] && break
        sleep 10
    done
    echo "$slug: $s"
done
# 4. the notebook: Kaggle runs it on the hidden test set (about 5 minutes on 2 x T4 for v76)
VERSION=$($K kernels push -p kaggle/notebooks/knee-infer | tee /dev/stderr | sed -n 's/.*Kernel version \([0-9]*\) .*/\1/p')
echo "notebook version $VERSION"
sleep 60
for i in $(seq 1 1440); do      # up to 8 h: Kaggle can keep a run queued
    s=$($K kernels status "$OWNER/$KERNEL" 2>/dev/null | tail -1 || true)
    case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
    sleep 20
done
echo "notebook run: $s"
case "$s" in *COMPLETE*) ;; *) exit 1 ;; esac
# 5. the submission (a late submission after the deadline scores but does not rank)
if [ -n "${SUBMIT:-}" ]; then
    $K competitions submit "$COMP" -k "$OWNER/$KERNEL" -v "$VERSION" -m "$MSG"
fi
