#!/bin/bash

# Detect architecture
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    DOCKERFILE="Dockerfile.amd64"
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    DOCKERFILE="Dockerfile.arm64"
    IMAGE_NAME="devanshdvj/taidl-micro25-artifact:arm64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

echo "Building TAIDL image for $ARCH using $DOCKERFILE"
docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" . 
