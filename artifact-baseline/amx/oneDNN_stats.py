import csv
import os
import re
import subprocess

# List of kernels to analyze (defined in *.c)
kernels = [
    'cnn_inference_amx',
    'cnn_inference_mix',
    'mem_format_prop_avx',
    'rnn_inference_amx',
    'sgemm_avx',
]
block_sizes = [4, 256]

# regex patterns to count instructions
AMX_RE = r'tdpbusd|tileloadd|tilestored|tilezero'
AVX_RE = r'vmovups|vmovaps|vxorps|vmulps|vaddps|vmaxps|vminps|vbroadcastss|vcvtps|vpmovusdb|vfmadd'

# Saving results as CSV
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_dir = os.path.join(repo_dir, "plots", "csv")
ONEDNN_STATS_CSV = "oneDNN-stats.csv"
csv_fields = [
    'kernel',
    'amx_4',
    'avx_4',
    'amx_256',
    'avx_256',
]
csv_header = {
    'kernel': 'Kernel',
    'amx_4': '#AMX (#BLOCKS=4)',
    'avx_4': '#AVX-512 (#BLOCKS=4)',
    'amx_256': '#AMX (#BLOCKS=256)',
    'avx_256': '#AVX-512 (#BLOCKS=256)',
}


def save_results(results):
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, ONEDNN_STATS_CSV)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, restval='NA', extrasaction='ignore')
        writer.writerow(csv_header)
        writer.writerows(results)

    print("\nCompleted benchmarking Intel SDE for oneDNN kernels.")
    print(f"Results saved to plots/csv/{ONEDNN_STATS_CSV}.")


if __name__ == "__main__":
    results = []
    for kernel in kernels:
        subprocess.run(['make', 'blocks=4', 'unroll=1', kernel], capture_output=True)
        asm_4 = subprocess.run(
            ['objdump', '--disassemble=main', f'./{kernel}'],
            capture_output=True,
            text=True
        ).stdout
        amx_4 = len(re.findall(AMX_RE, asm_4))
        avx_4 = len(re.findall(AVX_RE, asm_4))
        subprocess.run(['rm', f'./{kernel}'], capture_output=True)

        subprocess.run(['make', 'blocks=256', 'unroll=1', kernel], capture_output=True)
        asm_256 = subprocess.run(
            ['objdump', '--disassemble=main', f'./{kernel}'],
            capture_output=True,
            text=True
        ).stdout
        amx_256 = len(re.findall(AMX_RE, asm_256))
        avx_256 = len(re.findall(AVX_RE, asm_256))
        subprocess.run(['rm', f'./{kernel}'], capture_output=True)

        results.append({
            'kernel': kernel,
            'amx_4': amx_4,
            'avx_4': avx_4,
            'amx_256': amx_256,
            'avx_256': avx_256,
        })

    save_results(results)
