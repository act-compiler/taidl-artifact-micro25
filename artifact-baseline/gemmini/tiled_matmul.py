import argparse
import csv
import os
import re
import subprocess

# Set environment variable for Tiled MatMul kernels
os.environ["TILED_MATMUL_TEST"] = "1"

os.environ["IBERT_TEST"] = "1"
os.environ.pop('IBERT_TEST', None)

# List of kernels (defined in tests/*.c)
kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
gen = 'tiled_matmul_data_gen.c'
eval = {
    'OS': 'tiled_matmul_timing_os.c',
    'WS': 'tiled_matmul_timing_ws.c',
}

# Golden Input-output pairs to validate the simulations
data_path = os.path.join("accelerators", "gemmini", "tests", "data")

# Saving results as CSV
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_dir = os.path.join(repo_dir, "plots", "csv")
GEMMINI_MATMUL_CSV = "gemmini-tiled-matmul.csv"
csv_fields = [
    'DIM',
    'I',
    'df',
    'hbm',
    'cpu_runtime',
    'gpu_runtime',
    'spike_runtime'
]
csv_header = {
    'DIM': 'DIM',
    'I': 'I',
    'df': 'Dataflow',
    'hbm': 'HBM size',
    'cpu_runtime': 'TAIDL-TO (CPU) (ms)',
    'gpu_runtime': 'TAIDL-TO (GPU) (ms)',
    'spike_runtime': 'Gemmini Spike (ms)'
}


def run_setup_spike(dim):
    result = subprocess.run(
        ["./setup_spike.sh", str(dim)],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def modify_defines(file_paths, value):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        with open(file_path, "r") as f:
            content = f.read()

        pattern = r"#define I \d+"
        new_content = re.sub(pattern, f"#define I {value}", content)

        with open(file_path, "w") as f:
            f.write(new_content)


def run_build():
    result = subprocess.run(
        ["./build_gemmini_tests.sh"],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_benchmark(benchmark):
    DIM = benchmark['DIM']
    I = benchmark['I']  # noqa: E741
    df = benchmark['df']

    kernel_file = os.path.join(kernel_dir, eval[df])
    kernel_name = os.path.splitext(eval[df])[0]

    modify_defines(kernel_file, benchmark['I'])

    if I == 1 and df == 'OS':
        print(f"Setting up Spike for DIM {DIM}...")
        if not run_setup_spike(DIM):
            print(f"Setup failed for DIM {DIM}")
            exit(1)

    if not run_build():
        print(f"Build failed for DIM {DIM}, size {I}")
        exit(1)

    result = subprocess.run(
        ["./run_gemmini_test.sh", kernel_name],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )

    modify_defines(kernel_file, 1)  # Reset I to 1 after run

    spike_runtime = None
    if result.returncode == 0:
        output = result.stdout + result.stderr
        time_match = re.search(r"Spike execution time: ([\d.]+)ms", output)
        if time_match:
            spike_runtime = float(time_match.group(1))

    if spike_runtime:
        print(f"matmul_{df} DIM:{DIM} I:{I} | Runtime: {spike_runtime:.3f}ms")
    else:
        print(f"Warning: No runtime captured for matmul_{df} DIM:{DIM} I:{I}.")

    return spike_runtime


def generate_data(benchmark):
    DIM = benchmark['DIM']
    I = benchmark['I']  # noqa: E741

    kernel_file = os.path.join(kernel_dir, gen)
    kernel_name = os.path.splitext(gen)[0]
    benchmark_name = f"matmul_{DIM}_{I}"

    modify_defines(kernel_file, benchmark['I'])

    if I == 1:
        print(f"Setting up Spike for DIM {DIM}...")
        if not run_setup_spike(DIM):
            print(f"Setup failed for DIM {DIM}")
            exit(1)

    if not run_build():
        print(f"Build failed for DIM {DIM}, size {I}")
        exit(1)

    result = subprocess.run(
        ["./run_gemmini_test.sh", kernel_name],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )

    modify_defines(kernel_file, 1)  # Reset I to 1 after run

    # Save data lines if any were captured
    data_lines = []
    if result.returncode == 0:
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if line.startswith('DATA:'):
                data_lines.append(line[5:].strip())  # Remove "DATA:" prefix
    if data_lines:
        dataset = []
        indices = [i for i, line in enumerate(data_lines) if line.startswith("A ")]
        indices.append(len(data_lines))  # Add end index for the last dataset
        for start, end in zip(indices, indices[1:]):
            dataset.append('\n'.join(data_lines[start:end]))
        save_golden_data(benchmark_name, dataset)
    else:
        print(f"Warning: No golden input-output pairs captured for {benchmark_name}.")

    return result.returncode == 0


def save_golden_data(name, dataset):
    kernel_validation_dir = os.path.join(repo_dir, data_path, name)
    os.makedirs(kernel_validation_dir, exist_ok=True)
    for i, data in enumerate(dataset):
        data_file = os.path.join(kernel_validation_dir, f"{i}.txt")
        with open(data_file, "w") as f:
            f.write(data)
    print(f"Saved {len(dataset)} golden input-output pairs to {data_path}/{name}/")


def save_results(results):
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, GEMMINI_MATMUL_CSV)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, restval='NA', extrasaction='ignore')
        writer.writerow(csv_header)
        writer.writerows(results)

    print("\nCompleted benchmarking Gemmini Spike for Tiled MatMul kernels.")
    print(f"Results saved to plots/csv/{GEMMINI_MATMUL_CSV}.")


def init(mode):
    DIMS = [16, 64, 256, 1024]
    log2I_max = {16: 8, 64: 6, 256: 4, 1024: 0}

    if mode == 'gen':
        dataflows = ['OS']  # Data generation only needs one dataflow
    else:  # mode == 'eval'
        dataflows = ['OS', 'WS']

    benchmarks = []
    for dim in DIMS:
        for df in dataflows:
            for i in range(0, log2I_max[dim] + 1):
                benchmarks.append({
                    'DIM': dim,
                    'I': 2**i,
                    'df': df,
                })

    return benchmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        choices=['gen', 'eval'],
        required=True,
        help="Mode: 'gen' for data generation, 'eval' for evaluation")
    args = parser.parse_args()
    mode = args.mode

    if mode == 'gen':
        benchmarks = init(mode)

        for benchmark in benchmarks:
            if not generate_data(benchmark):
                print(f"Data generation failed for matmul_{benchmark['DIM']}_{benchmark['I']}")

        exit(0)
    elif mode == 'eval':
        benchmarks = init(mode)

        for benchmark in benchmarks:
            spike_runtime = run_benchmark(benchmark)
            benchmark['spike_runtime'] = f'{spike_runtime:.3f}'

        benchmarks = sorted(benchmarks, key=lambda x: x['df'])
        save_results(benchmarks)

        exit(0)
