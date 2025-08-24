#!/bin/bash

set -e

# Parse command line arguments
IBERT_FLAG=""
if [[ "$1" == "--spike-ibert" ]]; then
    IBERT_FLAG="--spike-ibert"
fi

docker stop taidl-main >/dev/null 2>&1 || true
docker stop taidl-baseline >/dev/null 2>&1 || true

cd "$(dirname "$0")/../"
HOST_MOUNT="$(pwd)"

UID_N="$(id -u)"
GID_N="$(id -g)"

# Set up logging
LOG_NAME="full_run_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$HOST_MOUNT/$LOG_NAME.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Detect architecture
ARCH=$(uname -m)

if [[ "$ARCH" == "x86_64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:amd64"
    BASELINE_IMAGE="devanshdvj/taidl-micro25-artifact:baseline-amd64"
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
    echo "Error: Full tests with baseline evaluations are only supported on x86_64/amd64 architecture"
    echo "Current architecture: $ARCH"
    exit 1
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

echo "=== Full Test Run Started at $(date) ==="
echo "Log file: $LOG_FILE"
echo "Running full tests with images: $IMAGE_NAME and $BASELINE_IMAGE"

# First run baseline in separate docker container
echo
echo "################################################################################"
echo "First, running baseline evaluation and golden data generation..."
echo "################################################################################"
echo

docker run --rm -t --name taidl-baseline \
    -v "$HOST_MOUNT:/workspace" \
    -w /workspace/artifact-baseline \
    $BASELINE_IMAGE \
    bash -c "rm -rf /workspace/plots/csv/ && rm -rf /workspace/plots/pdf/ && ./full.sh $IBERT_FLAG && \
    chown -R ${UID_N}:${GID_N} /workspace/*"

if [ $? -ne 0 ]; then
    echo "Error: Baseline execution failed"
    exit 1
fi

echo
echo "################################################################################"
echo "Baseline full run completed. Now, running TAIDL-TO full tests..."
echo "################################################################################"
echo

# Then run the main tests
docker run --rm --name taidl-main $GPU_FLAG \
    -v "$HOST_MOUNT:/taidl" \
    -w /taidl/artifact-taidl \
    $IMAGE_NAME \
    bash -c "./run-tests.sh --trials 100 && \
    chown -R ${UID_N}:${GID_N} /taidl/*"

echo
echo "#################################################################################"
echo "TAIDL-TO full run completed."
echo "#################################################################################"
echo

cd "$HOST_MOUNT/plots/"
mkdir "$LOG_NAME"
mv csv "$LOG_NAME"
mv pdf "$LOG_NAME"

echo "=== Full Test Run Completed at $(date) ==="
echo "All results moved to $HOST_MOUNT/plots/$LOG_NAME/"
echo "Log file: $HOST_MOUNT/$LOG_NAME.log"
