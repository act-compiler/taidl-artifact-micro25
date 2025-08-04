#!/bin/bash

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

echo "Starting interactive TAIDL container for $ARCH"
if [ -n "$GPU_FLAG" ]; then
    echo "Using GPU support: $GPU_FLAG"
fi

docker run --rm -it $GPU_FLAG \
    -v "$SCRIPT_DIR/..:/taidl" \
    -w /taidl \
    $IMAGE_NAME \
    bash
