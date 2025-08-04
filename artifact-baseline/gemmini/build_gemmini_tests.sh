#!/bin/bash -i

export RISCV=/gemmini/chipyard/.conda-env/riscv-tools
export PATH=/gemmini/chipyard/.conda-env/riscv-tools/bin:$PATH

source /gemmini/chipyard/env.sh

cd /gemmini

echo "Set up tests"
rm -rf ./chipyard/generators/gemmini/software/gemmini-rocc-tests/local
ln -s /workspace/artifact-baseline/gemmini/tests/ ./chipyard/generators/gemmini/software/gemmini-rocc-tests/local -f
cp /workspace/artifact-baseline/gemmini/changes/Makefile.in ./chipyard/generators/gemmini/software/gemmini-rocc-tests/Makefile.in
cp /workspace/artifact-baseline/gemmini/changes/run-spike.sh ./chipyard/generators/gemmini/scripts/run-spike.sh
cp /workspace/artifact-baseline/gemmini/changes/run-verilator.sh ./chipyard/generators/gemmini/scripts/run-verilator.sh

echo "Building gemmini-rocc-tests"
cd /gemmini/chipyard/generators/gemmini/software/gemmini-rocc-tests
rm -rf ./build/
./build.sh
