#!/usr/bin/env bash
# The DICOM codec wheels that the Kaggle notebook installs offline (no internet on Kaggle): one set for each Python
# version Kaggle may run (3.11, 3.12, 3.13). Made once; knee.submission.package writes their dataset metadata.
#   bash pipeline/0_wheels.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=kaggle/datasets/knee-wheels
mkdir -p "$D"
for v in 3.11 3.12 3.13; do
    uvx --from pip pip download --no-deps --only-binary=:all: --python-version "$v" \
        --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 -d "$D" \
        pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.4.0 pylibjpeg-openjpeg==2.6.0 python-gdcm==3.2.6
done
ls "$D"
