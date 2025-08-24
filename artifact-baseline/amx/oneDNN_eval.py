import argparse
import csv
import os
import subprocess

# List of kernels to benchmark (defined in *.c)
kernels = [
    'cnn_inference_amx',
    'cnn_inference_mix',
    'mem_format_prop_avx',
    'rnn_inference_amx',
    'sgemm_avx',
]
block_sizes = [2**i for i in range(2, 9)]

# Golden Input-output pairs to validate the simulations
data_path = os.path.join("accelerators", "amx", "tests", "data")

# Saving results as CSV
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_dir = os.path.join(repo_dir, "plots", "csv")
AMX_ONEDNN_CSV = "amx-oneDNN.csv"
csv_fields = [
    'kernel',
    'blocks',
    'hbm',
    'cpu_runtime',
    'gpu_runtime',
    'sde_runtime'
]
csv_header = {
    'kernel': 'Kernel',
    'blocks': 'Blocks',
    'hbm': 'HBM size',
    'cpu_runtime': 'TAIDL-TO (CPU) (ms)',
    'gpu_runtime': 'TAIDL-TO (GPU) (ms)',
    'sde_runtime': 'Intel SDE (ms)'
}


def run_kernel(kernel, trials):
    results = []

    cmd = ['/amx/sde64', '-spr', '--', f'./{kernel}']
    for block_size in block_sizes:
        subprocess.run(['make', f'blocks={block_size}', 'unroll=1', kernel], capture_output=True)
        hbm_size, total_time = 0, 0
        for _ in range(trials):
            output = subprocess.check_output(cmd).decode('utf-8')
            total_time += float(output.split()[-1])

        hbm_size = int(output.split()[-3])
        sde_runtime = total_time / trials

        results.append({
            'kernel': kernel,
            'blocks': block_size,
            'hbm': hbm_size,
            'sde_runtime': f'{sde_runtime:.3f}'
        })

        # Print timing in similar format to TAIDL-TO evaluation
        print(f"{kernel} SIZE:{block_size} | Runtime: {sde_runtime:.3f}ms")

        # Save data lines if any were captured
        data_lines = []
        for line in output.split('\n'):
            if line.startswith('DATA:'):
                data_lines.append(line[5:].strip())  # Remove "DATA:" prefix
        if data_lines:
            save_golden_data(f'{kernel}_{block_size}', '\n'.join(data_lines))
        else:
            print(f"Warning: No golden input-output pairs captured for {kernel}_{block_size}.")

        subprocess.run(['rm', f'./{kernel}'], capture_output=True)

    return results


def save_golden_data(name, data):
    kernel_validation_dir = os.path.join(repo_dir, data_path, name)
    os.makedirs(kernel_validation_dir, exist_ok=True)
    data_file = os.path.join(kernel_validation_dir, "0.txt")
    with open(data_file, "w") as f:
        f.write(data)
    print(f"Saved 1 golden input-output pair to {data_path}/{name}/")


def save_results(results):
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, AMX_ONEDNN_CSV)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, restval='NA', extrasaction='ignore')
        writer.writerow(csv_header)
        writer.writerows(results)

    print("\nCompleted benchmarking Intel SDE for oneDNN kernels.")
    print(f"Results saved to plots/csv/{AMX_ONEDNN_CSV}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=1, help="Number of trail runs")
    args = parser.parse_args()
    trials = args.trials

    results = []
    for kernel in kernels:
        results.extend(run_kernel(kernel, trials))

    save_results(results)
