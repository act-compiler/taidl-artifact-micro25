import argparse
import csv
import os
import re
import subprocess

os.environ["TILED_MATMUL_TEST"] = "1"

GEMMINI_CSV_FILE = "spike-gemmini-tiled-matmul.csv"


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


def run_test(test_name="matmul_timing", gen_mode=False):
    result = subprocess.run(
        ["./run_gemmini_test.sh", test_name],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )

    execution_time = None
    data_lines = []

    if result.returncode == 0:
        # Get full output
        output = result.stdout + result.stderr

        if gen_mode:
            # Extract DATA lines for generation mode
            for line in output.split('\n'):
                if line.startswith('DATA:'):
                    data_lines.append(line[6:])  # Remove "DATA: " prefix
        else:
            # Extract time from output for normal mode
            time_match = re.search(r"Spike execution time: ([\d.]+)ms", output)
            if time_match:
                execution_time = float(time_match.group(1))

    if gen_mode:
        return result.returncode == 0, data_lines
    else:
        return result.returncode == 0, execution_time


def run_setup_spike(dim):
    result = subprocess.run(
        ["./setup_spike.sh", str(dim)],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def save_data_lines(dim, i_value, all_data_lines):
    data_dir = f"/workspace/accelerators/gemmini/tests/data/matmul_{dim}_{i_value}"
    os.makedirs(data_dir, exist_ok=True)

    # Parse the data lines to separate the 5 datasets
    # Each data set starts with "A" matrix
    current_set = -1
    current_data = []

    for line in all_data_lines:
        if line.startswith("A "):
            # New data set starting, save previous if exists
            if current_set >= 0 and current_data:
                data_file = os.path.join(data_dir, f"{current_set}.txt")
                with open(data_file, "w") as f:
                    for data_line in current_data:
                        f.write(data_line + "\n")
                current_data = []
            current_set += 1
        current_data.append(line)

    # Save the last data set
    if current_set >= 0 and current_data:
        data_file = os.path.join(data_dir, f"{current_set}.txt")
        with open(data_file, "w") as f:
            for data_line in current_data:
                f.write(data_line + "\n")

    print(f"Saved {current_set + 1} datasets to {data_dir}/", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Gemmini evaluation script")
    parser.add_argument("--gen", action="store_true",
                        help="Generate test data instead of CSV")
    args = parser.parse_args()

    gen_test_files = ["/workspace/artifact-baseline/gemmini/tests/tiled_matmul_data_gen.c"]
    test_files = [
        "/workspace/artifact-baseline/gemmini/tests/tiled_matmul_timing_os.c",
        "/workspace/artifact-baseline/gemmini/tests/tiled_matmul_timing_ws.c"
    ]

    modify_defines(test_files, 1)
    modify_defines(gen_test_files, 1)

    if not args.gen:
        # Normal mode: generate CSV
        csv_dir = "/workspace/plots/csv"
        csv_file = os.path.join(csv_dir, GEMMINI_CSV_FILE)
        os.makedirs(csv_dir, exist_ok=True)

    results = []
    dims = [16, 64, 256, 1024]

    max_size = {
        "16": 8,
        "64": 6,
        "256": 4,
        "1024": 0,
    }

    for dim in dims:
        print(f"Setting up Spike for DIM {dim}...", flush=True)

        if not run_setup_spike(dim):
            print(f"Setup failed for DIM {dim}")
            continue

        for i in range(0, max_size[str(dim)] + 1):
            matrix_size = 2**i

            if args.gen:
                # Generation mode: generate all 5 datasets in one run
                modify_defines(gen_test_files, matrix_size)

                if not run_build():
                    print(f"Build failed for DIM {dim}, size {matrix_size}")
                    continue

                success, data_lines = run_test(test_name="tiled_matmul_data_gen", gen_mode=True)
                if success and data_lines:
                    save_data_lines(dim, matrix_size, data_lines)
                else:
                    print(f"Data generation failed for DIM {dim}, size {matrix_size}")
            else:
                # Timing mode: run both OS and WS tests
                modify_defines(test_files, matrix_size)

                if not run_build():
                    print(f"Build failed for DIM {dim}, size {matrix_size}")
                    continue

                # Run OS timing test
                success_os, exec_time_os = run_test(
                    test_name="tiled_matmul_timing_os", gen_mode=False)

                # Run WS timing test
                success_ws, exec_time_ws = run_test(
                    test_name="tiled_matmul_timing_ws", gen_mode=False)

                if success_os and exec_time_os is not None and success_ws and exec_time_ws is not None:
                    results.append([dim, matrix_size, exec_time_os, exec_time_ws])
                    print(
                        f"matmul_{dim}_{matrix_size} SIZE:{matrix_size} | OS Runtime: {exec_time_os:.3f}ms | WS Runtime: {exec_time_ws:.3f}ms",
                        flush=True)
                else:
                    print(
                        f"Test failed for DIM {dim}, size {matrix_size} - OS: {success_os}, WS: {success_ws}")

    if not args.gen:
        # Save CSV results only in normal mode
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["DIM", "I", "OS Spike Time (ms)", "WS Spike Time (ms)"])
            writer.writerows(results)

        print(f"Wrote results to {csv_file}")
    else:
        print("Data generation completed")


if __name__ == "__main__":
    main()
