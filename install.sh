#!/usr/bin/env bash
# Create the environment of the pipeline with uv (details: docs/install.md).
#
#   bash install.sh                 # Python 3.12, torch 2.13.0 cu129 -> ./.venv
#   CUDA=cu130 bash install.sh      # another CUDA build of PyTorch
#   VENV=/path/to/env bash install.sh
#   REQUIREMENTS=requirements-lock.txt bash install.sh   # the exact package versions of our environment
#
# Idempotent: an existing $VENV is reused and only missing or changed packages are installed.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0}"
CUDA="${CUDA:-cu129}"
VENV="${VENV:-.venv}"
INDEX="https://download.pytorch.org/whl/${CUDA}"

command -v uv >/dev/null 2>&1 || { echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

# 1. Python and virtual environment
if [ -x "$VENV/bin/python" ]; then
    echo "Reusing existing environment $VENV ($("$VENV/bin/python" --version))"
else
    uv python install "$PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi
PY="$VENV/bin/python"

# 2. PyTorch (prebuilt wheel for the selected CUDA build), then the other packages
uv pip install --python "$PY" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$INDEX"
uv pip install --python "$PY" -r "${REQUIREMENTS:-requirements.txt}"   # REQUIREMENTS=requirements-lock.txt: the exact versions we used
# 3. this repository as an editable package (import knee)
uv pip install --python "$PY" -e .

# 4. Verification
"$PY" - <<'PYEOF'
import numpy, pandas, sklearn, timm, pydicom, pylibjpeg, torch, knee
print('imports OK; torch', torch.__version__, '| timm', timm.__version__, '| pydicom', pydicom.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU 0: {p.name} sm_{p.major}{p.minor} | kernels in this build: {torch.cuda.get_arch_list()}')
    (torch.ones(8, device='cuda') @ torch.ones(8, device='cuda')).item()   # fails if the build has no kernels for this GPU
PYEOF
