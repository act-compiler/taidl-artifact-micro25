#!/bin/bash

docker stop taidl-main >/dev/null 2>&1 || true
docker stop taidl-baseline >/dev/null 2>&1 || true

SCRIPT_DIR=$(dirname "$0")

# Set up logging
LOG_FILE="$SCRIPT_DIR/../full_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Detect architecture
ARCH=$(uname -m)

if [[ "$ARCH" == "x86_64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:amd64"
    BASELINE_IMAGE="devanshdvj/taidl-micro25-artifact:baseline-amd64"
    GPU_FLAG=$(command -v nvidia-smi >/dev/null 2>&1 && echo "--gpus all" || echo "")

    if [ -n "$GPU_FLAG" ] && (! docker info 2>/dev/null | grep -q 'nvidia' || ! command -v nvidia-container-runtime >/dev/null 2>&1); then
        echo "Warning: NVIDIA GPU detected but the NVIDIA Container Toolkit is not set up properly."
        echo "Running with CPU only."
        GPU_FLAG=""
    fi
elif [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    echo "Error: Full tests with baseline are only supported on x86_64/amd64 architecture"
    echo "Current architecture: $ARCH"
    exit 1
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

echo "=== Full Test Run Started at $(date) ==="
echo "Log file: $LOG_FILE"
echo "Running full tests (100 trials) with baseline on x86_64"
if [ -n "$GPU_FLAG" ]; then
    echo "NVIDIA GPU detected."
else
    echo "No NVIDIA GPU drivers detected (missing nvidia-smi). Skipping TAIDL-TO (GPU) results."
fi

# First run baseline in separate docker container
echo "Running artifact-baseline/full.sh in baseline Docker container..."
docker run --rm -t --name taidl-baseline \
    -v "$SCRIPT_DIR/..:/workspace" \
    -w /workspace/artifact-baseline \
    $BASELINE_IMAGE \
    bash -c "rm -rf /workspace/plots/csv/ && rm -rf /workspace/plots/pdf/ && ./full.sh"

if [ $? -ne 0 ]; then
    echo "Error: Baseline execution failed"
    exit 1
fi

echo "Baseline completed. Now running TAIDL full tests..."

# Then run the main tests
docker run --rm --name taidl-main $GPU_FLAG \
    -v "$SCRIPT_DIR/..:/taidl" \
    -w /taidl/artifact-taidl \
    $IMAGE_NAME \
    bash -c "./run-tests.sh --trials 100"

echo "=== Full Test Run Completed at $(date) ==="
echo "All output has been saved to: $LOG_FILE"
