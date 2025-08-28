#!/bin/bash

# Configuration (amd64 only)
IMAGE_NAME="devanshdvj/taidl-micro25-artifact:baseline-amd64"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo "Error: This build script only supports amd64/x86_64 architecture"
    echo "Detected architecture: $ARCH"
    exit 1
fi

cd $(dirname "$0")

echo "Building baseline image $IMAGE_NAME (includes Gemmini/Chipyard build - 30+ minutes)"
docker build -t "$IMAGE_NAME" .
