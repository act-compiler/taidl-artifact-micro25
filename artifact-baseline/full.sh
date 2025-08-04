#!/bin/bash -i

cd /workspace/artifact-baseline/amx
python sde_eval.py

cd /workspace/artifact-baseline/gemmini
python gemmini_eval.py
python gemmini_eval.py --gen
python exo_gen.py
