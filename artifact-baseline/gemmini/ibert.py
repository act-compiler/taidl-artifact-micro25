import os
import re
import subprocess

# Set environment variable for I-BERT model
os.environ["TILED_MATMUL_TEST"] = "1"
os.environ.pop('TILED_MATMUL_TEST', None)

os.environ["IBERT_TEST"] = "1"

# Saving results as CSV
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_dir = os.path.join(repo_dir, "plots", "csv")
SPIKE_IBERT_LOG = "spike-ibert.log"


def run_setup_spike(dim):
    result = subprocess.run(
        ["./setup_spike.sh", str(dim)],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_build():
    result = subprocess.run(
        ["./build_gemmini_tests.sh"],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_ibert():
    result = subprocess.run(
        ["./run_gemmini_test.sh", "ibert"],
        cwd="/workspace/artifact-baseline/gemmini/",
        capture_output=True,
        text=True,
    )

    spike_runtime = None
    if result.returncode == 0:
        output = result.stdout + result.stderr
        time_match = re.search(r"Spike execution time: ([\d.]+)ms", output)
        if time_match:
            spike_runtime = float(time_match.group(1))

    if spike_runtime:
        print(f"I-BERT | Runtime: {spike_runtime:.3f}ms")
    else:
        print("Warning: No runtime captured for I-BERT.")

    return spike_runtime


def save_result(result):
    os.makedirs(csv_dir, exist_ok=True)
    log_path = os.path.join(csv_dir, SPIKE_IBERT_LOG)
    with open(log_path, "w") as f:
        f.write(f"{result:.3f}\n")

    print("\nCompleted benchmarking Gemmini Spike for end-to-end I-BERT model.")
    print(f"Results saved to plots/csv/{SPIKE_IBERT_LOG}.")


if __name__ == "__main__":
    DIM = 256

    print(f"Setting up Spike for DIM {DIM}...")

    if not run_setup_spike(DIM):
        print(f"Setup failed for DIM {DIM}")
        exit(1)

    if not run_build():
        print(f"Build failed for DIM {DIM}")
        exit(1)

    spike_runtime = run_ibert()
    if spike_runtime:
        save_result(spike_runtime)
