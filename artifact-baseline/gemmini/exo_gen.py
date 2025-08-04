import os
import subprocess

os.environ["TILED_MATMUL_TEST"] = "1"
os.environ.pop('TILED_MATMUL_TEST', None)


def run_build():
    result = subprocess.run(
        ["./build_gemmini_tests.sh"],
        cwd="/workspace/artifact-baseline/gemmini",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_test(test_name):
    result = subprocess.run(
        ["./run_gemmini_test.sh", test_name],
        cwd="/workspace/artifact-baseline/gemmini",
        capture_output=True,
        text=True,
    )

    data_lines = []
    if result.returncode == 0:
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if line.startswith('DATA:'):
                data_lines.append(line[6:])

    return result.returncode == 0, data_lines


def run_setup_spike():
    result = subprocess.run(
        ["./setup_spike.sh", "16"],
        cwd="/workspace/artifact-baseline/gemmini",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def save_data_lines(test_name, data_lines):
    data_dir = f"/workspace/accelerators/gemmini/tests/data/{test_name}"
    os.makedirs(data_dir, exist_ok=True)

    data_file = os.path.join(data_dir, "0.txt")
    with open(data_file, "w") as f:
        for line in data_lines:
            f.write(line + "\n")

    print(f"Saved 1 dataset to {data_dir}")


def main():
    exo_tests = [
        "exo_12544x256x64",
        "exo_12544x64x256",
        "exo_3136x128x512",
        "exo_3136x512x128",
        "exo_512x512x512",
        "exo_784x1024x256"
    ]

    print("Setting up Spike for DIM 16...", flush=True)
    if not run_setup_spike():
        print("Setup failed")
        return

    print("Building...", flush=True)
    if not run_build():
        print("Build failed")
        return

    for test_name in exo_tests:
        print(f"Running {test_name}...", flush=True)
        success, data_lines = run_test(test_name)
        if success and data_lines:
            save_data_lines(test_name, data_lines)
        else:
            print(f"Failed to generate data for {test_name}")


if __name__ == "__main__":
    main()
