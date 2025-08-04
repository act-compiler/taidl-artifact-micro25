#!/bin/bash

docker stop taidl-main >/dev/null 2>&1 || true
docker stop taidl-baseline >/dev/null 2>&1 || true

SCRIPT_DIR=$(dirname "$0")

# Set up logging
LOG_FILE="$SCRIPT_DIR/../lite_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

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

echo "=== Lite Test Run Started at $(date) ==="
echo "Log file: $LOG_FILE"
echo "Running lite tests (10 trials) with image: $IMAGE_NAME"
if [ -n "$GPU_FLAG" ]; then
    echo "GPU detected. Using $GPU_FLAG"
else
    echo "No GPU detected. Running without GPU support."
fi

docker run --rm --name taidl-main $GPU_FLAG \
    -v "$SCRIPT_DIR/..:/taidl" \
    -w /taidl/artifact-taidl \
    $IMAGE_NAME \
    bash -c "rm -rf /taidl/plots/csv/ && rm -rf /taidl/plots/pdf/ && ./run-tests.sh --trials 10"

echo "=== Lite Test Run Completed at $(date) ==="
echo "All output has been saved to: $LOG_FILE"
