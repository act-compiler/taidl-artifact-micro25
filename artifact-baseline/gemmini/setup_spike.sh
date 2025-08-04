#!/bin/bash -i

if [ -z "$1" ]; then
    echo "Usage: $0 <DIM>"
    exit 1
fi

export RISCV=/gemmini/chipyard/.conda-env/riscv-tools
export PATH=/gemmini/chipyard/.conda-env/riscv-tools/bin:$PATH

cd /gemmini/

echo "Copy Configs"
cp /workspace/artifact-baseline/gemmini/changes/CustomConfigs_$1.scala ./chipyard/generators/gemmini/src/main/scala/gemmini/CustomConfigs.scala
cp /workspace/artifact-baseline/gemmini/changes/gemmini_params_$1.h ./chipyard/generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h

cd /gemmini/chipyard/generators/gemmini

echo "Installing libgemmini"
make -C software/libgemmini install

./scripts/setup-paths.sh

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

cd /gemmini/chipyard/generators/gemmini/

echo "Building Spike"
#./scripts/build-verilator.sh
./scripts/build-spike.sh
