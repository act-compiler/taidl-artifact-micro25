import sys
import csv
import os
import argparse
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(os.path.dirname(base_dir), "sim")
sys.path.append(target_dir)
from kernels import *
from decorator import set_simulation_backend, gen_arguments, verifier, gpu_check

log_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "plots/csv"
)

AMX_CSV_FILE = "taidl-oneDNN.csv"

amx_avx = [
    cnn_inference_amx,
    cnn_inference_mix,
    mem_format_prop_avx,
    rnn_inference_amx,
    sgemm_avx,
]


def parse_file_data(file_path):
    data = []
    var_names = []
    dtype_map = {'i8': np.int8, 'i32': np.int32, 'f32': np.float32, 'u8': np.uint8}

    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 3 and parts[-1] in dtype_map:
            var_name = parts[0]
            ndims = int(parts[1])
            dtype_str = parts[-1]
            shape = [int(x) for x in parts[2:2 + ndims]]
            numpy_dtype = dtype_map[dtype_str]

            # Read data from next line
            i += 1
            if i >= len(lines):
                continue

            values = lines[i].strip().split()
            if numpy_dtype in [np.int8, np.int32, np.uint8]:
                data_values = [int(val) for val in values]
            else:
                data_values = [float(val) for val in values]

            array = np.array(data_values, dtype=numpy_dtype).reshape(shape)
            data.append(array)
            var_names.append(var_name)

        i += 1

    return data, var_names


def load_test_data(kernel_name):
    kernel_dir = os.path.join(base_dir, "data", kernel_name)
    if not os.path.exists(kernel_dir):
        return []

    data_files = [f for f in os.listdir(kernel_dir) if f.endswith('.txt')]
    if not data_files:
        return []

    all_test_data = []
    for file_name in sorted(data_files):
        try:
            file_path = os.path.join(kernel_dir, file_name)
            data, var_names = parse_file_data(file_path)

            if data:
                inputs = [
                    array for array, var_name in zip(
                        data, var_names) if not var_name.startswith("out")]
                outputs = [
                    array for array, var_name in zip(
                        data, var_names) if var_name.startswith("out")]
                all_test_data.append((inputs, outputs, file_name))
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

    return all_test_data


def run_kernel(kernel_func, block_size, trials):
    kernel, hbm_size = kernel_func(block_size)
    inputs, compile_time = kernel('fsim-compile')()

    # Load and verify test data
    test_data = load_test_data(f"{kernel_func.__name__}_{block_size}")
    verified = 0

    if not test_data:
        print(f"No test data found for {kernel_func.__name__}_SIZE{block_size}, cannot verify")

    for file_inputs, outputs, file_name in test_data:
        ans, _ = kernel('fsim')(*file_inputs)
        if verifier(ans, outputs):
            verified += 1

    # Performance measurement
    total_time = 0
    for _ in range(trials):
        test_inputs = gen_arguments(inputs)
        _, run_time = kernel('fsim')(*test_inputs)
        total_time += run_time

    avg_runtime = total_time / trials
    verified_str = f"{verified}/{len(test_data)}" if test_data else "0/0"
    verified_status = f"VERIFIED" if verified == len(test_data) and test_data else "FAILED"
    print(f"{kernel_func.__name__} SIZE:{block_size} | Compile: {compile_time:.3f}ms | Runtime: {avg_runtime:.3f}ms | {verified_status} {verified_str}")

    return hbm_size, compile_time, avg_runtime


def run_all(trials):
    block_range = 9
    results = []

    print("\nAMX-AVX oneDNN Kernels")

    # CPU Backend
    print("CPU Backend")
    set_simulation_backend("CPU")
    for kernel_func in amx_avx:
        for block_size in range(2, block_range):
            hbm, compile_time, run_time = run_kernel(kernel_func, 2**block_size, trials)
            results.append({
                'kernel': kernel_func.__name__,
                'hbm': hbm,
                'blocks': 2**block_size,
                'cpu_compile': compile_time,
                'cpu_runtime': run_time,
                'gpu_compile': 0,
                'gpu_runtime': 0
            })

    # GPU Backend (if available)
    if gpu_check():
        print("GPU Backend")
        set_simulation_backend("GPU")
        result_idx = 0
        for kernel_func in amx_avx:
            for block_size in range(2, block_range):
                _, gpu_compile, gpu_runtime = run_kernel(kernel_func, 2**block_size, trials)
                results[result_idx]['gpu_compile'] = gpu_compile
                results[result_idx]['gpu_runtime'] = gpu_runtime
                result_idx += 1

    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, AMX_CSV_FILE)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Kernel', 'HBM size', 'Blocks',
                         'CPU Compile Time (ms)', 'GPU Compile Time (ms)',
                         'CPU Runtime (ms)', 'GPU Runtime (ms)'])

        for result in results:
            writer.writerow([
                result['kernel'], result['hbm'], result['blocks'],
                f"{result['cpu_compile']:.3f}", f"{result['gpu_compile']:.3f}",
                f"{result['cpu_runtime']:.3f}", f"{result['gpu_runtime']:.3f}"
            ])

    print(f"Wrote results to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    run_all(args.trials)
