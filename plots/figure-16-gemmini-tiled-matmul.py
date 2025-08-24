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
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(which='major', axis='both', linewidth=0.3)
    ax.grid(which='minor', axis='x', linewidth=0.1)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=True, left=True)


dims = [16, 64, 256]
hbm_sizes, spike_times, cpu_times, gpu_times = [], [], [], []

# Read data files
data = pd.read_csv(os.path.join(CSV_DIR, 'gemmini-tiled-matmul.csv'))
data = data[data['Dataflow'] == 'WS']

for dim in dims:
    dim_data = data[data['DIM'] == dim]
    hbm_sizes.append(np.array(dim_data['HBM size'].astype(int)))
    cpu_times.append(np.array(dim_data['TAIDL-TO (CPU) (ms)'].astype(float).dropna()))
    gpu_times.append(np.array(dim_data['TAIDL-TO (GPU) (ms)'].astype(float).dropna()))
    spike_times.append(np.array(dim_data['Gemmini Spike (ms)'].astype(float).dropna()))

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 5))

for n in range(3):
    # Plot if available
    if len(spike_times[n]) > 0:  # kick-tires.sh and full.sh
        axes[n].plot(
            hbm_sizes[n],
            spike_times[n],
            'o-',
            color='blue',
            label='Gemmini Spike',
            linewidth=2.5,
            markersize=7)

    if len(gpu_times[n]) > 0:  # with GPU support
        axes[n].plot(
            hbm_sizes[n],
            gpu_times[n],
            'o-',
            color='green',
            label='TAIDL-TO (GPU)',
            linewidth=2.5,
            markersize=7)

    if len(cpu_times[n]) > 0:  # always available
        axes[n].plot(
            hbm_sizes[n],
            cpu_times[n],
            'o-',
            color='red',
            label='TAIDL-TO (CPU)',
            linewidth=2.5,
            markersize=7)

    beautify(axes[n])
    axes[n].set_title(f"({chr(ord('a') + n)}) DIM = {dims[n]}")
    axes[n].set_xlabel('Kernel size')

axes[0].set_ylabel('Simulation time (in ms)')

axes[0].set_xticks([1e3, 1e4, 1e5], labels=['1 KB', '10 KB', '100 KB'])
axes[1].set_xticks([2e4, 1e5, 8e5], labels=['20 KB', '100 KB', '800 KB'])
axes[2].set_xticks([3e5, 1e6, 3e6], labels=['300 KB', '1 MB', '3 MB'])

if len(spike_times[0]) > 0:
    fig.suptitle('TAIDL-TO for Gemmini ISA vs. Gemmini Spike', fontsize=24)
else:
    fig.suptitle('TAIDL-TO for Gemmini ISA', fontsize=24)
axes[0].legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure-16-gemmini-tiled-matmul.pdf'),
            format='pdf', bbox_inches='tight', dpi=600)
