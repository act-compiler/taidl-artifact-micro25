#!/bin/bash

# Script to copy paper_data CSVs, run all plotting scripts, and copy generated PDFs back.
# This script orchestrates the complete plotting workflow for paper_data data.

echo "Loading pre-evaluated data from paper_data/..."
echo "Starting plotting workflow..."

# Copy CSVs from paper_data/ to csv/
mkdir -p csv/
cp paper_data/*.csv csv/

echo "Running plotting scripts..."
python3 figure-16-gemmini-tiled-matmul.py
python3 figure-17-amx-oneDNN.py
python3 figure-18-gemmini-exo.py

# Copy generated PDFs from pdf/ to paper_data/
mkdir -p paper_data/
cp pdf/*.pdf paper_data/

rm -r csv/ pdf/

echo "Completed plotting workflow..."
echo "Generated plots saved as plots/paper_data/*.pdf"
echo
echo "Note that this script does not rerun any benchmarks locally."
echo "It uses data from author's final benchmarking runs to regenerate plots in the paper."
