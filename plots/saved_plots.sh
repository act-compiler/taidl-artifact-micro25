#!/bin/bash

# Script to copy saved CSVs, run all plotting scripts, and copy generated PDFs back.
# This script orchestrates the complete plotting workflow for saved data.

echo "Starting plotting workflow..."

# Copy CSVs from saved/ to csv/
echo "Copying saved CSVs to csv/..."
mkdir -p csv/
cp saved/*.csv csv/

# Run all three plotting scripts
echo "Running plotting scripts..."
python3 amx_plots.py

python3 gemmini_exo_plots.py

python3 gemmini_matmul_plots.py

# Copy generated PDFs from pdf/ to saved/
echo "Copying generated PDFs to saved/..."
mkdir -p saved/
cp pdf/*.pdf saved/


echo "Plotting workflow completed!"
