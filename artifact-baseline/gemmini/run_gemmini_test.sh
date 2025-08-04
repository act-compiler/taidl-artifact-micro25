#!/bin/bash -i

if [ -z "$1" ]; then
    echo "Usage: $0 <test_name>"
    exit 1
fi

export RISCV=/gemmini/chipyard/.conda-env/riscv-tools
export PATH=/gemmini/chipyard/.conda-env/riscv-tools/bin:$PATH

source /gemmini/chipyard/env.sh

cd /gemmini
echo "Starting spike execution..."
start_time=$(date +%s%3N)
./chipyard/generators/gemmini/scripts/run-spike.sh chipyard/generators/gemmini/software/gemmini-rocc-tests/build/local/$1-baremetal
end_time=$(date +%s%3N)
execution_time=$((end_time - start_time))
echo "Spike execution time: ${execution_time}ms"
