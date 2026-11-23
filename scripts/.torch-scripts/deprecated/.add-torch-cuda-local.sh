#!/bin/bash
# Call scripts with bash -i ./scripts/<scriptname>.sh

source scripts/.before-script.sh
source scripts/.functions.sh

read -p "Torch version (default to 2.5.1): " TORCH_VERSION
TORCH_VERSION=${TORCH_VERSION:-2.5.1}

read -p "Torch version (default to cu124): " CUDA_VERSION
CUDA_VERSION=${CUDA_VERSION:-cu124}

SMALL_PYTHON_VERSION=$(echo $PYTHON_VERSION | sed -e 's/\.//g' | head -c 3)

BASE_TORCH_URL=https://download.pytorch.org/whl/$CUDA_VERSION
TARGET_WHEEL_ONLINE=torch-$TORCH_VERSION%2B$CUDA_VERSION-cp$SMALL_PYTHON_VERSION-cp$SMALL_PYTHON_VERSION-win_amd64.whl
TARGET_WHEEL_LOCAL=torch-$TORCH_VERSION+$CUDA_VERSION-cp$SMALL_PYTHON_VERSION-cp$SMALL_PYTHON_VERSION-win_amd64.whl

source scripts/.export-proxy.sh
if [ ! -f $PACKAGES_DIR/$TARGET_WHEEL_LOCAL ]; then
    make_a_dir $PACKAGES_DIR
    add_gitkeep_in_folder $PACKAGES_DIR
    curl -k $BASE_TORCH_URL/$TARGET_WHEEL_ONLINE --output $PACKAGES_DIR/$TARGET_WHEEL_LOCAL
fi
source scripts/.unset-proxy.sh

uv add $PACKAGES_DIR/$TARGET_WHEEL_LOCAL