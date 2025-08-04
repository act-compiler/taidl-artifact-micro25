#!/bin/bash

docker stop taidl-main >/dev/null 2>&1 || true
docker stop taidl-baseline >/dev/null 2>&1 || true

SCRIPT_DIR=$(dirname "$0")

# Detect architecture
ARCH=$(uname -m)

if [[ "$ARCH" == "x86_64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:amd64"
    GPU_FLAG=$(command -v nvidia-smi >/dev/null 2>&1 && echo "--gpus all" || echo "")
elif [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:arm64"
    GPU_FLAG=""
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

echo "Kick-tires plotting - generating plots from saved data with image: $IMAGE_NAME"

docker run --rm --name taidl-main \
    -v "$SCRIPT_DIR/..:/taidl" \
    -w /taidl/plots \
    $IMAGE_NAME \
    bash -c "rm -rf /taidl/plots/csv/ && rm -rf /taidl/plots/pdf/ && bash saved_plots.sh"
