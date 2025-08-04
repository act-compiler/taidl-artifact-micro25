import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Common setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, 'csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'pdf')
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-bright')
plt.rcParams.update({'font.size': 15})


def beautify(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(which='major', axis='both', linewidth=0.3)
    ax.grid(which='minor', axis='x', linewidth=0.1)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=True, left=True)


def gemmini_plot():
    dims = [16, 64, 256]
    hbm_sizes, spike_times, cpu_times, gpu_times = [], [], [], []

    # Load data files
    spike_file_path = os.path.join(CSV_DIR, 'spike-gemmini-tiled-matmul.csv')
    has_spike_data = os.path.exists(spike_file_path)

    if has_spike_data:
        spike_data = pd.read_csv(spike_file_path)
    cpu_gpu_data = pd.read_csv(os.path.join(CSV_DIR, 'taidl-gemmini-tiled-matmul.csv'))

    for dim in dims:
        cpu_gpu_dim_data = cpu_gpu_data[(cpu_gpu_data['DIM'] == dim)
                                        & (cpu_gpu_data['Dataflow'] == 'WS')]

        if has_spike_data:
            spike_dim_data = spike_data[spike_data['DIM'] == dim]
            # Find common I values between datasets
            common_i_values = sorted(set(spike_dim_data['I']) & set(cpu_gpu_dim_data['I']))
            # Filter both datasets to common I values
            spike_filtered = spike_dim_data[spike_dim_data['I'].isin(
                common_i_values)].sort_values('I')
            cpu_gpu_filtered = cpu_gpu_dim_data[cpu_gpu_dim_data['I'].isin(
                common_i_values)].sort_values('I')

            hbm_sizes.append(np.array(cpu_gpu_filtered['HBM Size']))
            spike_times.append(np.array(spike_filtered['OS Spike Time (ms)']))
            cpu_times.append(np.array(cpu_gpu_filtered['CPU Runtime (ms)']))
            gpu_times.append(np.array(cpu_gpu_filtered['GPU Runtime (ms)']))
        else:
            cpu_gpu_filtered = cpu_gpu_dim_data.sort_values('I')
            hbm_sizes.append(np.array(cpu_gpu_filtered['HBM Size']))
            spike_times.append(None)
            cpu_times.append(np.array(cpu_gpu_filtered['CPU Runtime (ms)']))
            gpu_times.append(np.array(cpu_gpu_filtered['GPU Runtime (ms)']))

    fig, axes = plt.subplots(1, len(dims), sharey=True, figsize=(20, 5))

    for n in range(len(dims)):
        if has_spike_data and spike_times[n] is not None:
            axes[n].plot(
                hbm_sizes[n],
                spike_times[n],
                'o-',
                color='blue',
                label='Gemmini Spike',
                linewidth=2.5,
                markersize=7)
        axes[n].plot(
            hbm_sizes[n],
            gpu_times[n],
            'o-',
            color='green',
            label='TAIDL-TO (GPU)',
            linewidth=2.5,
            markersize=7)
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

    if has_spike_data:
        fig.suptitle('TAIDL-TO for Gemmini ISA vs. Gemmini Spike', fontsize=24)
    else:
        fig.suptitle('TAIDL-TO for Gemmini ISA', fontsize=24)
    axes[0].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'figure-16-gemmini-tiled-matmul.pdf'),
                format='pdf', bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    gemmini_plot()
