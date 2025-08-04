import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Common setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
PLOTS_DIR = os.path.join(BASE_DIR, "pdf")
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


def sde_plot():
    kernels = ['cnn_inference_amx', 'rnn_inference_amx', 'cnn_inference_mix',
               'sgemm_avx', 'mem_format_prop_avx']
    titles = ['cnn_inf_amx', 'rnn_inf_amx', 'cnn_inf_mix', 'sgemm_avx', 'mem_format_avx']
    hbm_sizes, sde_times, cpu_times, gpu_times = [], [], [], []

    # Read data files
    data = pd.read_csv(os.path.join(CSV_DIR, 'taidl-oneDNN.csv'), na_values='---')
    data.dropna(inplace=True)

    sde_file = os.path.join(CSV_DIR, 'sde-oneDNN.csv')
    sde_data = pd.read_csv(sde_file) if os.path.exists(sde_file) else None

    for kernel in kernels:
        kernel_data = data[data['Kernel'] == kernel]
        hbm_sizes.append(np.array(kernel_data['HBM size']))
        cpu_times.append(np.array(kernel_data['CPU Runtime (ms)']))
        gpu_times.append(np.array(kernel_data['GPU Runtime (ms)']))

        # Get SDE times if available
        if sde_data is not None:
            sde_kernel_data = sde_data[sde_data['Kernel'] == kernel]
            if not sde_kernel_data.empty:
                sde_times.append(np.array(sde_kernel_data['SDE Time (ms)']))
            else:
                sde_times.append(np.array([]))
        else:
            sde_times.append(np.array([]))

    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(20, 8))

    for n in range(len(kernels)):
        # Plot SDE times if available
        if len(sde_times[n]) > 0:
            l1 = axes.flat[n].plot(
                hbm_sizes[n],
                sde_times[n],
                'o-',
                color='blue',
                linewidth=2.5,
                markersize=7)
        l2 = axes.flat[n].plot(
            hbm_sizes[n],
            gpu_times[n],
            'o-',
            color='green',
            linewidth=2.5,
            markersize=7)
        l3 = axes.flat[n].plot(
            hbm_sizes[n],
            cpu_times[n],
            'o-',
            color='red',
            linewidth=2.5,
            markersize=7)

        beautify(axes.flat[n])
        axes.flat[n].set_title(f"({chr(ord('a') + n)}) {titles[n]}")
        axes.flat[n].set_xlabel('Kernel size')

    axes.flat[-1].set_axis_off()

    axes[0][0].set_ylabel('Simulation time (in ms)')
    axes[1][0].set_ylabel('Simulation time (in ms)')

    axes[0][0].set_xticks([2e4, 1e5, 6e5], labels=['20 KB', '100 KB', '600 KB'])
    axes[0][1].set_xticks([2e4, 1e5, 6e5], labels=['20 KB', '100 KB', '600 KB'])
    axes[0][2].set_xticks([2e4, 1e5, 6e5], labels=['20 KB', '100 KB', '600 KB'])
    axes[1][0].set_xticks([5e3, 1e4, 5e4], labels=['5 KB', '10 KB', '50 KB'])
    axes[1][1].set_xticks([3e3, 1e4, 3e4], labels=['3 KB', '10 KB', '30 KB'])

    fig.suptitle('TAIDL-TO for Intel AMX & AVX-512', fontsize=24)

    # Create legend based on available data
    if sde_data is not None and len(sde_times[0]) > 0:
        fig.legend(
            handles=[
                l1[0],
                l2[0],
                l3[0]],
            labels=[
                'Intel SDE',
                'TAIDL-TO (GPU)',
                'TAIDL-TO (CPU)'],
            bbox_to_anchor=(
                0.83,
                0.30),
            loc='center')
    else:
        fig.legend(handles=[l2[0], l3[0]], labels=['TAIDL-TO (GPU)', 'TAIDL-TO (CPU)'],
                   bbox_to_anchor=(0.83, 0.25), loc='center')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'figure-17-oneDNN.pdf'),
                format='pdf', bbox_inches='tight', dpi=600)


sde_plot()
