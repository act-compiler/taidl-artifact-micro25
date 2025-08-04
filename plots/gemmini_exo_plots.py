import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Common setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
PLOTS_DIR = os.path.join(BASE_DIR, "pdf")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-bright')
plt.rcParams.update({'font.size': 15})


def beautify(ax):
    ax.grid(True, axis='y', linestyle='--', zorder=0)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=True, left=True)
    ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.4, zorder=0)


df = pd.read_csv(os.path.join(CSV_DIR, "taidl-gemmini-exo.csv"))

matmul_names = {
    "exo_512x512x512": "512 x 512 x 512",
    "exo_12544x256x64": "12544 x 256 x 64",
    "exo_12544x64x256": "12544 x 64 x 256",
    "exo_3136x512x128": "3136 x 512 x 128",
    "exo_3136x128x512": "3136 x 128 x 512",
    "exo_784x1024x256": "784 x 1024 x 256",
}

order = [
    "exo_512x512x512",
    "exo_12544x256x64",
    "exo_12544x64x256",
    "exo_3136x512x128",
    "exo_3136x128x512",
    "exo_784x1024x256"]

# Process data
runtime_data = {row['Kernel']: (row["CPU Runtime (ms)"], row["GPU Runtime (ms)"])
                for _, row in df.iterrows()}

labels, cpu_values, gpu_values = [], [], []
for kernel in order:
    cpu_time, gpu_time = runtime_data[kernel]
    labels.append(matmul_names[kernel])
    cpu_values.append(cpu_time)
    gpu_values.append(gpu_time)

# Create plot
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
beautify(ax)
ax.bar(x - width / 2, gpu_values, width, label='TAIDL-TO (GPU)', color='green', zorder=2)
ax.bar(x + width / 2, cpu_values, width, label='TAIDL-TO (CPU)', color='red', zorder=2)

ax.set_ylabel('Simulation time (in ms)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, fontsize=12)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure-18-gemmini-exo.pdf'),
            format='pdf', bbox_inches='tight', dpi=600)
