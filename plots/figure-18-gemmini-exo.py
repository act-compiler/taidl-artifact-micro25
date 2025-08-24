import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Start: Common setup #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
PLOTS_DIR = os.path.join(BASE_DIR, "pdf")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-bright')
plt.rcParams.update({'font.size': 15})
# End: Common setup #


def beautify(ax):
    ax.grid(True, axis='y', linestyle='--', zorder=0)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=True, left=True)
    ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.4, zorder=0)


kernels = [
    'matmul_512',
    'matmul_4',
    'matmul_6',
    'matmul_14',
    'matmul_16',
    'matmul_27',
]
labels, cpu_values, gpu_values = [], [], []

# Read data files
data = pd.read_csv(os.path.join(CSV_DIR, "gemmini-exo.csv"))

for kernel in kernels:
    kernel_data = data[data['Kernel'] == kernel]
    if not kernel_data.empty:
        kernel_data = kernel_data.iloc[0]
    else:
        continue

    M = int(kernel_data['M'])
    N = int(kernel_data['N'])
    K = int(kernel_data['K'])
    labels.append(f'{N}x{M}x{K}')
    cpu_values.append(float(kernel_data['TAIDL-TO (CPU) (ms)']))
    gpu_values.append(float(kernel_data['TAIDL-TO (GPU) (ms)']))


fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(labels))
width = 0.35
ax.bar(x - width / 2, gpu_values, width, label='TAIDL-TO (GPU)', color='green', zorder=2)
ax.bar(x + width / 2, cpu_values, width, label='TAIDL-TO (CPU)', color='red', zorder=2)

beautify(ax)
ax.set_ylabel('Simulation time (in ms)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, fontsize=12)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure-18-gemmini-exo.pdf'),
            format='pdf', bbox_inches='tight', dpi=600)
