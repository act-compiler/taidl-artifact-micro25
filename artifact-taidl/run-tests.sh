#!/bin/bash
trap "exit" INT TERM
trap "kill 0" EXIT

set -e

# Parse command line arguments
TRIALS=10  # Default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Define lists for accelerators and their corresponding TAIDL files
accelerators=("AMX" "gemmini")
#accelerators=("AMX")

export PYTHONUNBUFFERED=1

cd /taidl/accelerators/gemmini/

python3 TAIDL_gemmini.py --size=16
python3 TAIDL_gemmini.py --size=64
python3 TAIDL_gemmini.py --size=256
python3 TAIDL_gemmini.py --size=1024

cd /taidl/accelerators/AMX/
python3 TAIDL_AMX.py


for accelerator in "${accelerators[@]}"; do
    echo "Running tests for $accelerator with $TRIALS trials"
    cd "/taidl/accelerators/$accelerator" && python3 -u tests/main.py --trials $TRIALS
done

echo
echo "Running plotting scripts..."

plot_scripts=("amx_plots.py" "gemmini_exo_plots.py" "gemmini_matmul_plots.py")
for plot_script in "${plot_scripts[@]}"; do
    cd /taidl/plots && python3 "$plot_script"
done
