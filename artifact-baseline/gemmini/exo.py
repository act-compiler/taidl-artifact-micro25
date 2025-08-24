import os
import subprocess

# Set environment variable for Exo kernels
os.environ["TILED_MATMUL_TEST"] = "1"
os.environ.pop('TILED_MATMUL_TEST', None)

os.environ["IBERT_TEST"] = "1"
os.environ.pop('IBERT_TEST', None)

# List of kernels to benchmark (defined in tests/*.c)
kernels = [
    ("matmul_6", 12544, 64, 256),
    ("matmul_4", 12544, 256, 64),
    ("matmul_27", 784, 1024, 256),
    ("matmul_14", 3136, 512, 128),
    ("matmul_16", 3136, 128, 512),
    ("matmul_512", 512, 512, 512),
]

# Golden Input-output pairs to validate the simulations
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join("accelerators", "gemmini", "tests", "data")


def run_setup_spike():
    result = subprocess.run(
        ["./setup_spike.sh", "16"],
        cwd="/workspace/artifact-baseline/gemmini",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_build():
    result = subprocess.run(
        ["./build_gemmini_tests.sh"],
        cwd="/workspace/artifact-baseline/gemmini",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_kernel(kernel):
    result = subprocess.run(
        ["./run_gemmini_test.sh", kernel],
        cwd="/workspace/artifact-baseline/gemmini",
        capture_output=True,
        text=True,
    )

    # Save data lines if any were captured
    data_lines = []
    if result.returncode == 0:
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if line.startswith('DATA:'):
                data_lines.append(line[5:].strip())  # Remove "DATA:" prefix
    if data_lines:
        save_golden_data(kernel, '\n'.join(data_lines))
    else:
        print(f"Warning: No golden input-output pairs captured for {kernel}.")

    return result.returncode == 0


def save_golden_data(name, data):
    kernel_validation_dir = os.path.join(repo_dir, data_path, name)
    os.makedirs(kernel_validation_dir, exist_ok=True)
    data_file = os.path.join(kernel_validation_dir, "0.txt")
    with open(data_file, "w") as f:
        f.write(data)
    print(f"Saved 1 golden input-output pair to {data_path}/{name}/")


if __name__ == "__main__":
    print("Setting up Spike for DIM 16...")
    if not run_setup_spike():
        print("Setup failed")
        exit(1)

    if not run_build():
        print("Build failed")
        exit(1)

    for kernel in kernels:
        kernel_name = f"exo_{kernel[1]}x{kernel[2]}x{kernel[3]}"
        if not run_kernel(kernel_name):
            print(f"Run failed for {kernel_name}")
            exit(1)
