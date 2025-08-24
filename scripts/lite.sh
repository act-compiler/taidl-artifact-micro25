#!/bin/bash

set -e

docker stop taidl-main >/dev/null 2>&1 || true
docker stop taidl-baseline >/dev/null 2>&1 || true

cd "$(dirname "$0")/../"
HOST_MOUNT="$(pwd)"

UID_N="$(id -u)"
GID_N="$(id -g)"

# Set up logging
LOG_NAME="lite_run_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$HOST_MOUNT/$LOG_NAME.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Detect architecture
ARCH=$(uname -m)

if [[ "$ARCH" == "x86_64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:amd64"
    GPU_FLAG=$(command -v nvidia-smi >/dev/null 2>&1 && echo "--gpus all" || echo "")

    if [ -n "$GPU_FLAG" ]; then
        if (! docker info 2>/dev/null | grep -q 'nvidia' || ! command -v nvidia-container-runtime >/dev/null 2>&1); then
            echo "Warning: NVIDIA GPU detected but the NVIDIA Container Toolkit is not set up properly."
            echo "Running without GPU support."
            GPU_FLAG=""
        else
            echo "NVIDIA GPU detected. Running with GPU support."
        fi
    else
        echo "No GPU detected. Running without GPU support."
    fi
elif [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:arm64"
    GPU_FLAG=""
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

echo "=== Lite Test Run Started at $(date) ==="
echo "Log file: $LOG_FILE"
echo "Running lite tests with image: $IMAGE_NAME"

echo
docker run --rm --name taidl-main $GPU_FLAG \
    -v "$HOST_MOUNT:/taidl" \
    -w /taidl/artifact-taidl \
    $IMAGE_NAME \
    bash -c "rm -rf /taidl/plots/csv/ && rm -rf /taidl/plots/pdf/ && ./run-tests.sh --trials 10 && \
    chown -R ${UID_N}:${GID_N} /taidl/*"

cd "$HOST_MOUNT/plots/"
mkdir "$LOG_NAME"
mv csv "$LOG_NAME"
mv pdf "$LOG_NAME"

echo
echo "=== Full Test Run Completed at $(date) ==="
echo "All results moved to $HOST_MOUNT/plots/$LOG_NAME/"
echo "Log file: $HOST_MOUNT/$LOG_NAME.log"
