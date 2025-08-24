#!/bin/bash

set -e

docker stop taidl-main >/dev/null 2>&1 || true
docker stop taidl-baseline >/dev/null 2>&1 || true

cd "$(dirname "$0")/../"
HOST_MOUNT="$(pwd)"

UID_N="$(id -u)"
GID_N="$(id -g)"

# Detect architecture
ARCH=$(uname -m)

if [[ "$ARCH" == "x86_64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:amd64"
elif [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:arm64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

echo "Running kick-tires plotting with image: $IMAGE_NAME"

docker run --rm --name taidl-main \
    -v "$HOST_MOUNT:/taidl" \
    -w /taidl/plots \
    $IMAGE_NAME \
    bash -c "rm -rf /taidl/plots/csv/ && rm -rf /taidl/plots/pdf/ && bash reference_plots.sh && \
    chown -R ${UID_N}:${GID_N} /taidl/*"
