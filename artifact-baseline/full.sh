#!/bin/bash
trap "exit" INT TERM
trap "kill 0" EXIT

set -e

# Parse command line arguments
IBERT_FLAG=""
if [[ "$1" == "--spike-ibert" ]]; then
    IBERT_FLAG="--spike-ibert"
    echo "--spike-ibert flag detected. Gemmini Spike will be benchmarked for I-BERT model as well."
    echo "Note that this will significantly increase the runtime of the full tests (likely by over an hour)."
    echo
fi

export PYTHONUNBUFFERED=1

echo "########## Accelerator 1: Gemmini ##########"
cd /workspace/artifact-baseline/gemmini

echo
echo "(1.1A) Gemmini Spike for Tiled MatMul kernels: generating golden data..."
python tiled_matmul.py --mode gen

echo
echo "(1.1B) Gemmini Spike for Tiled MatMul kernels: benchmarking..."
python tiled_matmul.py --mode eval

echo
echo "(1.2) Gemmini Spike for Exo kernels: generating golden data..."
python exo.py

echo
echo "(1.3) Gemmini Spike for end-to-end I-BERT model: benchmarking..."
if [[ -n "$IBERT_FLAG" ]]; then
    python ibert.py
else
    echo "Disabled by default. Skipping..."
    echo "To enable, rerun with --spike-ibert flag."
fi

echo
echo "########## Accelerator 2: Intel AMX ##########"
cd /workspace/artifact-baseline/amx

echo
echo "(2.1) Benchmarking Intel SDE for oneDNN kernels averaged over 4 trials..."
python oneDNN_eval.py --trials 4

echo
echo "(2.2) Collecting instruction statistics for oneDNN kernels..."
python oneDNN_stats.py
