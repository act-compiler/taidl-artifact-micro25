#!/bin/bash
trap "exit" INT TERM
trap "kill 0" EXIT

set -e

# Parse command line arguments
TRIALS=10  # Default value
if [[ "$1" == "--trials" ]]; then
    TRIALS="$2"
fi

export PYTHONUNBUFFERED=1

echo "########## Accelerator 1: Gemmini ##########"
cd /taidl/accelerators/gemmini/

rm -rf sim*
python3 TAIDL_gemmini.py --size=16
python3 TAIDL_gemmini.py --size=64
python3 TAIDL_gemmini.py --size=256
python3 TAIDL_gemmini.py --size=1024

echo
echo "(1.1) Benchmarking all Gemmini(DIM=*) TAIDL-TOs for Tiled MatMul kernels averaged over $TRIALS trials..."
python3 tests/main.py --kernel_type tiled_matmul --trials $TRIALS

echo
echo "(1.2) Benchmarking Gemmini(DIM=16) TAIDL-TO for Exo kernels averaged over $TRIALS trials..."
python3 tests/main.py --kernel_type exo --trials $TRIALS

echo
echo "(1.3) Benchmarking Gemmini(DIM=256) TAIDL-TO for end-to-end I-BERT model averaged over 5 trials..."
python3 tests/ibert.py --trials 5

echo
echo "########## Accelerator 2: Intel AMX ##########"
cd /taidl/accelerators/amx/

rm -rf sim
python3 TAIDL_amx.py

echo
echo "(2.1) Benchmarking AMX TAIDL-TO for oneDNN kernels averaged over $TRIALS trials..."
python3 tests/main.py --trials $TRIALS

echo
echo "Running plotting scripts..."
cd /taidl/plots

python3 figure-16-gemmini-tiled-matmul.py
python3 figure-17-amx-oneDNN.py
python3 figure-18-gemmini-exo.py
