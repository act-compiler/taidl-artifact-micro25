import argparse
import csv
import os
import sys

import numpy as np
from kernels import *

MATMUL_CSV_FILE = "taidl-gemmini-tiled-matmul.csv"
EXO_CSV_FILE = "taidl-gemmini-exo.csv"

base_dir = os.path.dirname(os.path.abspath(__file__))

# Import decorator functions from sim_16 by default
target_dir = os.path.join(os.path.dirname(base_dir), "sim_16")
sys.path.append(target_dir)
from decorator import gen_arguments, set_simulation_backend, verifier, gpu_check

log_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "plots", "csv",
)

exo_kernels = {
    "exo_12544x64x256": (matmul_6, 12544, 256, 64),
    "exo_12544x256x64": (matmul_4, 12544, 64, 256),
    "exo_784x1024x256": (matmul_27, 784, 256, 1024),
    "exo_3136x512x128": (matmul_14, 3136, 128, 512),
    "exo_3136x128x512": (matmul_16, 3136, 512, 128),
    "exo_512x512x512": (matmul_512, 512, 512, 512),
}


def load_file_inputs(kernel_name):
    kernel_dir = os.path.join(base_dir, "data", kernel_name)
    if not os.path.exists(kernel_dir):
        print(f"Error: Kernel directory {kernel_dir} not found")
        return []

    data_files = [f for f in os.listdir(kernel_dir) if f.endswith('.txt')]
    if not data_files:
        print(f"Error: No data files found in {kernel_dir}")
        return []

    all_test_data = []

    for file_name in sorted(data_files):
        data = []
        try:
            file_path = os.path.join(kernel_dir, file_name)
            with open(file_path, "r") as f:
                lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print(f"Warning: Invalid line format at line {i + 1}: {line}")
                    i += 1
                    continue
                var_name = parts[0]
                shape = [int(dim) for dim in parts[1:]]
                total_elements = np.prod(shape)
                data_values = []
                i += 1
                while i < len(lines) and len(data_values) < total_elements:
                    line = lines[i].strip()
                    if line:
                        values = line.split()
                        data_values.extend([float(val) for val in values])
                    i += 1
                if len(data_values) < total_elements:
                    print(
                        f"Warning: Not enough data for variable {var_name}. Expected {total_elements}, got {len(data_values)}"
                    )
                    continue
                elif len(data_values) > total_elements:
                    print(
                        f"Warning: Too much data for variable {var_name}. Expected {total_elements}, got {len(data_values)}"
                    )
                    data_values = data_values[:total_elements]
                array = np.array(data_values, dtype=np.float64)
                array = array.reshape(shape)
                data.append(array)
        except FileNotFoundError:
            print(f"Error: File {file_name} not found")
            continue
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            continue

        if data:
            inputs = data[:-1]
            inputs = [i.astype(np.int8) for i in inputs]
            gold = data[-1]
            gold = [gold.astype(np.int32)]
            all_test_data.append((inputs, gold, file_name))

    return all_test_data


def run_gemmini_exo(name, iterations):
    gen_kernel, I, J, K = exo_kernels[name]

    # Load all inputs and outputs from kernel directory
    test_data = load_file_inputs(f"{name}")

    verified = 0

    if not test_data:
        print(f"No test data found for {kernel_func.__name__}_SIZE{block_size}, cannot verify")

    kernel = gen_kernel()
    compile_time = 0

    # Run verification on all test data files
    for file_inputs, gold, file_name in test_data:
        if compile_time == 0:  # Compile only once
            inputs, compile_time = kernel("fsim-compile")(*file_inputs)

        ans, tmp_time = kernel("fsim")(*file_inputs)

        if verifier(ans, gold):
            verified += 1

    res = 0
    for _ in range(iterations):
        # Random inputs after verification
        a = np.random.randint(low=-2, high=3, size=(I, J), dtype=np.int8)
        b = np.random.randint(low=-2, high=3, size=(J, K), dtype=np.int8)
        ans, tmp_time = kernel("fsim")(a, b)
        res += tmp_time

    avg_runtime = res / iterations
    verified_str = f"{verified}/{len(test_data)}" if test_data else "0/0"
    verified_status = f"VERIFIED" if verified == len(test_data) and test_data else "FAILED"
    print(f"{name} | Compile: {compile_time:.3f}ms | Runtime: {avg_runtime:.3f}ms | {verified_status} {verified_str}")

    return compile_time, avg_runtime


def run_gemmini_kernel(iterations=16, trials=1, xla_while=0, df=0, DIM=16):
    kernel, hbm_size = gemmini_kernel(DIM, iterations, df=df, xla_while=xla_while)
    inputs, compile_time = kernel("fsim-compile")()

    # Load all inputs and outputs from kernel directory
    test_data = load_file_inputs(f"matmul_{DIM}_{iterations}")

    if not test_data:
        print(f"No test data found for {kernel_func.__name__}_SIZE{block_size}, cannot verify")

    compile_time = 0
    verified = 0

    # Run verification on all test data files
    for file_inputs, gold, file_name in test_data:
        if compile_time == 0:  # Compile only once
            inputs, compile_time = kernel("fsim-compile")(*file_inputs)

        ans, tmp_time = kernel("fsim")(*file_inputs)

        if verifier(ans, gold):
            verified += 1

    res = 0
    for _ in range(trials):
        ans, tmp_time = kernel("fsim")(*gen_arguments(inputs))
        res += tmp_time

    avg_runtime = res / trials
    dataflow = "WS" if df == 1 else "OS"
    verified_str = f"{verified}/{len(test_data)}" if test_data else "0/0"
    verified_status = f"VERIFIED" if verified == len(test_data) and test_data else "FAILED"
    print(f"matmul_{dataflow} DIM:{DIM} I:{iterations} | Compile: {compile_time:.3f}ms | Runtime: {avg_runtime:.3f}ms | {verified_status} {verified_str}")

    return compile_time, avg_runtime, hbm_size


exo_csv_data = {}
matmul_csv_data = {}


def run_all(platform, trials):
    set_simulation_backend(platform)
    print("\nGemmini Exo")
    print(f"{platform} Backend", flush=True)
    # All exo kernels run with DIM=16 only (using sim_16 directory)
    for name, value in exo_kernels.items():
        compile_time, runtime = run_gemmini_exo(name, trials)
        if platform == "CPU":
            exo_csv_data[name] = [
                name,
                f"{compile_time:.3f}",
                0.0,
                f"{runtime:.3f}",
                0.0,
            ]
        elif platform == "GPU":
            exo_csv_data[name][2] = f"{compile_time:.3f}"
            exo_csv_data[name][4] = f"{runtime:.3f}"

    print("\nGemmini tiled matmul")
    print(f"{platform} Backend", flush=True)
    use_loop = 4

    # Run gemmini_kernel with DIM=16, 64, 256
    DIMs = [16, 64, 256, 1024]
    i_range = {
        "16": 8,
        "64": 6,
        "256": 4,
        "1024": 0
    }
    for DIM in DIMs:
        for i in range(0, i_range[str(DIM)] + 1):
            compile_time, runtime, hbm_size = run_gemmini_kernel(
                2**i, trials, (i > use_loop), df=1, DIM=DIM
            )
            name = f"WS{i}_DIM{DIM}"
            if platform == "CPU":
                matmul_csv_data[name] = [
                    hbm_size,
                    DIM,
                    2**i,
                    "WS",
                    f"{compile_time:.3f}",
                    0.0,
                    f"{runtime:.3f}",
                    0.0,
                ]
            elif platform == "GPU":
                matmul_csv_data[name][5] = f"{compile_time:.3f}"
                matmul_csv_data[name][7] = f"{runtime:.3f}"
        for i in range(0, i_range[str(DIM)] + 1):
            compile_time, runtime, hbm_size = run_gemmini_kernel(
                2**i, trials, (i > use_loop), df=0, DIM=DIM
            )
            name = f"OS{i}_DIM{DIM}"
            if platform == "CPU":
                matmul_csv_data[name] = [
                    hbm_size,
                    DIM,
                    2**i,
                    "OS",
                    f"{compile_time:.3f}",
                    0.0,
                    f"{runtime:.3f}",
                    0.0,
                ]
            elif platform == "GPU":
                matmul_csv_data[name][5] = f"{compile_time:.3f}"
                matmul_csv_data[name][7] = f"{runtime:.3f}"


def run_i_bert(platform, trials):
    set_simulation_backend(platform)
    print("\nGemmini i-bert")
    print(f"{platform} Backend")
    kernel = i_bert()
    inputs, compile_time = kernel("fsim-compile")()
    res = 0
    print(f"i-bert compile time: {compile_time:.3f} ms")
    for i in range(trials):
        ans, tmp_time = kernel("fsim")(*gen_arguments(inputs))
        res += tmp_time
    print(f"i-bert runtime: {res / trials:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()

    run_all("CPU", args.trials)

    if gpu_check():
        run_all("GPU", args.trials)

    # Generate EXO csv
    os.makedirs(log_path, exist_ok=True)
    exo_csv_path = os.path.join(log_path, EXO_CSV_FILE)
    matmul_csv_path = os.path.join(log_path, MATMUL_CSV_FILE)

    with open(exo_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Kernel', 'CPU Compile Time (ms)', 'GPU Compile Time (ms)',
            'CPU Runtime (ms)', 'GPU Runtime (ms)'
        ])
        for row in exo_csv_data.values():
            writer.writerow(row)
        print("Wrote gemmini-exo to", exo_csv_path)

    with open(matmul_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'HBM Size', 'DIM', 'I', 'Dataflow',
            'CPU Compile Time (ms)', 'GPU Compile Time (ms)',
            'CPU Runtime (ms)', 'GPU Runtime (ms)'
        ])
        for row in matmul_csv_data.values():
            writer.writerow(row)
        print("Wrote gemmini-matmul to", matmul_csv_path)
