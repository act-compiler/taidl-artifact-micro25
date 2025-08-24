#!/bin/bash

# Delete all generated files and directories to clean the workspace.

# Remove compiled Python files
rm -rf **/__pycache__
rm -rf accelerators/*/tests/__pycache__
rm -rf idl/__pycache__

# Remove generated TAIDL-TO APIs
rm -rf accelerators/*/sim*
rm -rf accelerators/*/tests/data/*/[!0].txt

# Remove generated CSVs and PDFs
rm -rf plots/lite_run_*
rm -rf plots/full_run_*
rm -rf plots/paper_data/*.pdf

# Remove log files
rm -rf lite_run_*.log
rm -rf full_run_*.log
