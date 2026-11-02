#!/usr/bin/env bash
# Train one config on all training studies (docs/training.md) on one GPU, resumable (run it again to continue), then
# write the test-time copy of the model without its joint transformer branch (checkpoints/final_nojoint.pt).
#   bash pipeline/5_train.sh configs/<name>.yaml [key=value ...]
#   CUDA_VISIBLE_DEVICES=1 bash pipeline/5_train.sh configs/<name>.yaml train.epochs=1 train.max_train_studies=24   # a smoke run
set -euo pipefail
cd "$(dirname "$0")/.."
export NUMPY_MADVISE_HUGEPAGE=0     # NumPy huge pages slow large arrays down on fragmented memory
CONFIG=$1; shift || true
NAME=$(.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['name'])")
run=$(ls -d experiments/*_${NAME}_f-1 2>/dev/null | tail -1 || true)
if [ -n "$run" ] && [ -f "$run/checkpoints/final.pt" ]; then
    echo "finished run: $run"
elif [ -n "$run" ] && [ -f "$run/checkpoints/last.pt" ]; then
    .venv/bin/python -m knee.training.train --resume "$run"
else
    .venv/bin/python -m knee.training.train --config "$CONFIG" --fold -1 --set "$@"
fi
.venv/bin/python -m knee.submission.drop_joint --run "${NAME}_f-1"
