import csv
import os
import subprocess

csv_dir = '/workspace/plots/csv'
output_file = 'sde-oneDNN.csv'

tests = ['cnn_inference_amx',
         'rnn_inference_amx',
         'sgemm_avx',
         'mem_format_prop_avx',
         'cnn_inference_mix',]

block_range = range(2, 9)
num_trials = 4


def time_kernel(test, cmd):
    hbm_sizes, avg_times = {}, {}
    for b in block_range:
        num_blocks = 2**b
        subprocess.run(['make', f'blocks={num_blocks}', f'unroll=1', test],
                       capture_output=True,)
        hbm_size, time = 0, 0
        data_lines = []
        for trial in range(num_trials):
            output = subprocess.check_output(cmd).decode('utf-8')
            hbm_size = int(output.split()[-3])
            time += float(output.split()[-1])
            if trial == 0:
                for line in output.split('\n'):
                    if line.startswith('DATA:'):
                        data_lines.append(line[5:])  # Remove "DATA: " prefix

        hbm_sizes[num_blocks] = hbm_size
        avg_time = time / num_trials
        avg_times[num_blocks] = avg_time

        # Print timing in similar format to AMX tests
        print(f"{test} SIZE:{num_blocks} | Runtime: {avg_time:.3f}ms")

        # Save data lines if any were captured
        if data_lines:
            save_data_lines(test, num_blocks, data_lines)

        subprocess.run(['rm', f'./{test}'])
    return hbm_sizes, avg_times


def save_data_lines(test_name, num_blocks, data_lines):
    data_dir = f"/workspace/accelerators/AMX/tests/data/{test_name}_{num_blocks}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f"0.txt")
    with open(data_file, "w") as f:
        for line in data_lines:
            f.write(line + "\n")
    print(f"Saved 1 dataset to {data_dir}")


def write_combined_csv(filename, all_results):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Kernel', 'HBM size', 'Blocks', 'SDE Time (ms)'])
        for test_name, (hbm_sizes, times) in all_results.items():
            for num_blocks in sorted(times.keys()):
                hbm_size = hbm_sizes[num_blocks]
                sde_time = times[num_blocks]
                writer.writerow([test_name, hbm_size, num_blocks, f'{sde_time:.3f}'])


def sde_test():
    all_results = {}
    for test in tests:
        print(f"Running {test}...", flush=True)
        cmd = ['/amx/sde64', '-spr', '--', f'./{test}']
        hbm_sizes, times = time_kernel(test, cmd)
        all_results[test] = (hbm_sizes, times)
    return all_results


def main():
    os.makedirs(csv_dir, exist_ok=True)
    all_results = sde_test()
    output_path = os.path.join(csv_dir, output_file)
    write_combined_csv(output_path, all_results)


if __name__ == '__main__':
    main()
