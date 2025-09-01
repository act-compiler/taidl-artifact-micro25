# TAIDL: Tensor Accelerator ISA Definition Language

![Badge: Artifact Available](badges/artifacts_available_dl.jpg)
![Badge: Artifact Evaluated - Functional](badges/artifacts_evaluated_functional_dl.jpg)
![Badge: Artifact Evaluated - Reusable](badges/results_reproduced_dl.jpg)

This is the artifact for our paper "TAIDL: Tensor Accelerator ISA Definition Language with Auto-generation of Scalable Test Oracles".
In our paper, we present an ISA specification language for tensor accelerators and auto-generated test oracles (a.k.a. functional simulators).

This artifact consists of TAIDL source code and the necessary scripts to reproduce the evaluation results.
To facilitate artifact evaluation, we have automated the entire environment setup and experimental processes as part of Docker images.
Our evaluation results were collected using Intel Xeon Platinum 8358 CPU and NVIDIA A100 GPU.
We recommend using a machine with an Intel CPU and an NVIDIA GPU to benchmark TAIDL-TO and its baselines.
Reproducing all simulation statistics takes approximately 30-45 minutes.

GitHub version of the artifact is present at [https://github.com/act-compiler/taidl-artifact-micro25](https://github.com/act-compiler/taidl-artifact-micro25). Note that this repository is now archived (read-only).  
For more recent releases, refer to [https://github.com/act-compiler/taidl](https://github.com/act-compiler/taidl)

## System Requirements

**Docker**

We use Docker images for environment setup of TAIDL and baselines.  
To run the TAIDL artifact, install Docker using the [installation guide](https://docs.docker.com/engine/install/).

**Architecture Support**

- **amd64/x86_64**: Full support for TAIDL including GPU acceleration and baselines.
- **arm64**: CPU-only support for TAIDL (no GPU support). The `full.sh` script cannot be run on arm64 since baselines are not supported on this architecture.

**GPU Support**

For NVIDIA GPU usage on amd64 systems, install the NVIDIA Container Toolkit using the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## TAIDL Artifact Overview

### Scripts

`scripts/` directory contains the following bash scripts to automate the experimental workflows:

- `setup.sh` -- load the necessary Docker images from local tarballs or [Docker Hub](https://hub.docker.com/r/devanshdvj/taidl-micro25-artifact).
- `kick-tires.sh` -- regenerate paper figures from pre-evaluated data; does not benchmark the simulators.
- `lite.sh` -- only benchmark TAIDL-TOs and verifies using pre-generated golden data; does not regenerate any golden data or run baselines.
- `full.sh` -- benchmark TAIDL-TOs along with baselines; regenerates all golden data to verify TAIDL-TOs.
- `launch.sh` -- start a TAIDL Docker environment for interactive use.
- `clean.sh` -- remove all files generated including the log files, CSV data files, and PDF plots.

### Docker images

| Docker image                                       |    Environment Installed | Architecture | GPU support | Dockerfile                          |
| -------------------------------------------------- | -----------------------: | ------------ | ----------- | ----------------------------------- |
| `devanshdvj/taidl-micro25-artifact:amd64`          |                    TAIDL | amd64        | Yes         | `./artifact-taidl/Dockerfile.amd64` |
| `devanshdvj/taidl-micro25-artifact:arm64`          |                    TAIDL | arm64        | No          | `./artifact-taidl/Dockerfile.arm64` |
| `devanshdvj/taidl-micro25-artifact:baseline-amd64` | Gemmini Spike, Intel SDE | amd64        | No          | `./artifact-baseline/Dockerfile`    |

### Generated CSV data files and PDF plots

| Filename                             |            Metric | ISA                 | Kernel suite         | Paper Reference                                                                            |
| ------------------------------------ | ----------------: | ------------------- | -------------------- | ------------------------------------------------------------------------------------------ |
| `gemmini-tiled-matmul.csv`           |   Simulation Time | Gemmini ISA         | Tiled MatMul kernels | Raw data for Figure 16                                                                     |
| `figure-16-gemmini-tiled-matmul.pdf` |   Simulation Time | Gemmini ISA         | Tiled MatMul kernels | Figure 16 (Scalability analysis of Auto-generated TAIDL-TOs against Gemmini Spike)         |
| `gemmini-exo.csv`                    |   Simulation Time | Gemmini ISA         | Exo-compiled kernels | Raw data for Figure 18                                                                     |
| `figure-18-gemmini-exo.pdf`          |   Simulation Time | Gemmini ISA         | Exo-compiled kernels | Figure 18 (Case Study: Integrating TAIDL-TO into Existing Compiler Testing Infrastructure) |
| `gemmini-ibert.csv`                  |   Simulation Time | Gemmini ISA         | I-BERT model         | Section 8.2 (Case Study: Simulating end-to-end Model)                                      |
| `amx-oneDNN.csv`                     |   Simulation Time | Intel AMX + AVX-512 | oneDNN kernels       | Raw data for Figure 17                                                                     |
| `figure-17-amx-oneDNN.pdf`           |   Simulation Time | Intel AMX + AVX-512 | oneDNN kernels       | Figure 17 (Scalability analysis of Auto-generated TAIDL-TOs against Intel SDE)             |
| `oneDNN-stats.csv`                   | Instruction Count | Intel AMX + AVX-512 | oneDNN kernels       | Table 4 (Statistics of the selected oneDNN kernels)                                        |

## Step-by-step Instructions to Reproduce Results

### (Step 0) Setup the Docker Environment

This will first try to load the necessary Docker images from tarballs locally present in `./` or `../` (i.e., current or parent directory).
These tarballs, if not present, can be downloaded from [Zenodo](https://zenodo.org/record/16971223).

If the tarballs are not found locally, it will try to pull the images from [Docker Hub](https://hub.docker.com/r/devanshdvj/taidl-micro25-artifact).

This may take a 1-2 minutes if loading from local tarballs, or 4-5 minutes if pulling from Docker Hub.

Run the setup script using:

```bash
./scripts/setup.sh
```

(Not recommended) Alternatively, if you have a machine with sufficient resources, you can build the Docker images locally using:

```bash
./scripts/setup.sh --build
```

This may take over an hour to build all images from scratch.

### (Step 1) Kick the Tires: Quick Plot Generation

This uses paper's benchmarking data (`./plots/paper_data/`) to quickly generate all figures without running any benchmarking experiments.
These statistics were collected using Intel Xeon Platinum 8358 CPU and NVIDIA A100 GPU.

Run the kick-tires script using:

```bash
./scripts/kick-tires.sh
```

The resulting figures can be found in `./plots/paper_data/`.

- `figure-16-gemmini-tiled-matmul.pdf` - Comparing simulation times of TAIDL-TO and Gemmini Spike
- `figure-17-oneDNN.pdf` - Comparing simulation times of TAIDL-TO and Intel SDE
- `figure-18-gemmini-exo.pdf` - Benchmarking TAIDL-TO for Exo-compiled Gemmini kernels

- `*.csv` - Paper's raw benchmarking data (already part of the artifact)

### (Step 2) Only Benchmark TAIDL-TOs using Pre-generated Data

This uses pre-generated inputs and golden outputs from Gemmini Spike and Intel SDE to benchmark TAIDL-TOs.
It does not regenerate any data or run the baselines.
This is useful for quickly verifying TAIDL-TO's correctness and performance.
This would take around 2-5 minutes to run.

Run the lite script using:

```bash
./scripts/lite.sh
```

For every lite run, a new folder is created as `./plots/lite_run_<timestamp>/` containing the generated PDF plots and CSV data files.

- `pdf/figure-16-gemmini-tiled-matmul.pdf` - Benchmarking TAIDL-TO for Gemmini's tiled matrix multiplication kernels
- `pdf/figure-17-oneDNN.pdf` - Benchmarking TAIDL-TO for oneDNN's Intel AMX kernels.
- `pdf/figure-18-gemmini-exo.pdf` - Benchmarking TAIDL-TO for Exo-compiled Gemmini kernels

- `csv/amx-oneDNN.csv` - Raw data for Figure 17
- `csv/gemmini-exo.csv` - Raw data for Figure 18
- `csv/gemmini-ibert.csv` - Benchmarking results for I-BERT model
- `csv/gemmini-tiled-matmul.csv` - Raw data for Figure 16

Not generated in this step:

- Baseline benchmarking data (set to NA in the CSVs)
- `csv/oneDNN-stats.csv` (it uses the baseline docker image)

### (Step 3) Regenerate All Test Data and Benchmarking Results

This will benchmark TAIDL-TO along with baselines (Gemmini Spike and Intel SDE).
The script will also generate new data files containing inputs and outputs from these tools, which are used to verify TAIDL-TO's output.
This would take around 30-45 minutes to run.

Run the full script using:

```bash
./scripts/full.sh
```

By default, this script will not simulate I-BERT model on Gemmini Spike since it takes a long time (around an hour).
To enable it, use the `--spike-ibert` flag:

```bash
./scripts/full.sh --spike-ibert
```

For every full run, a new folder is created as `./plots/full_run_<timestamp>/` containing the generated PDF plots and CSV data files.

- `pdf/figure-16-gemmini-tiled-matmul.pdf` - Comparing simulation times of TAIDL-TO and Gemmini Spike
- `pdf/figure-17-oneDNN.pdf` - Comparing simulation times of TAIDL-TO and Intel SDE
- `pdf/figure-18-gemmini-exo.pdf` - Benchmarking TAIDL-TO for Exo-compiled Gemmini kernels

- `csv/amx-oneDNN.csv` - Raw data for Figure 17
- `csv/gemmini-exo.csv` - Raw data for Figure 18
- `csv/gemmini-ibert.csv` - Benchmarking results for I-BERT model
- `csv/gemmini-tiled-matmul.csv` - Raw data for Figure 16
- `csv/oneDNN-stats.csv` - Instruction statistics for oneDNN kernels

### (Optional) Cleanup/Fresh Start

To remove all files generated including the log files, CSV data files, and PDF plots, run:

```bash
./scripts/clean.sh
```

## Observations

When running the `full.sh` scripts, you should observe the following trends in the generated plots:

### Relative performance of TAIDL-TOs vs Baselines

TAIDL-TOs show significant speedups over the baselines across all benchmarks.
The speedup is more pronounced for larger problem sizes due to better optimization and parallelism in TAIDL-TO. (More details in the paper)

For example, in Figure 16, the relative speedup of TAIDL-TO over Gemmini Spike increases as DIM increases.
Furthermore, the simulation time for end-to-end models like I-BERT are significantly lower for TAIDL-TO (a few seconds) compared to Gemmini Spike (around an hour).
Similar trends are observed in Figure 17 for oneDNN kernels, albeit slightly lower speedups due to an industrial-grade baseline (Intel SDE).

### Relative performance of TAIDL-TO (CPU) vs TAIDL-TO (GPU)

In general, CPU-GPU trends are difficult to predict due to various factors like data transfer overheads, kernel launch latencies, and hardware characteristics.
However, we observe that TAIDL-TO with GPU acceleration consistently outperforms CPU-only execution for larger problem sizes (large DIMs, batch sizes, etc.).
Whereas, TAIDL-TO on CPU can be faster for smaller problem sizes due to lower overheads.

For example, in Figure 16 for Tiled MatMul kernels, TAIDL-TO with GPU starts outperforming CPU-only execution around DIM=64 on our machine setup (Intel Xeon Platinum 8358 CPU + NVIDIA A100 GPU).
This crossover point may vary based on the CPU and GPU characteristics of the machine like memory bandwidths, CPU core counts, CPU generation, GPU compute capabilities, etc.

On one extreme, in Figure 17 for oneDNN kernels, TAIDL-TO on CPU is consistently faster than GPU execution for all benchmarks due to the small tensor shapes in AMX instruction semantics (16x64).

On the other extreme, for end-to-end models like I-BERT for Gemmini (DIM=256), TAIDL-TO with GPU is significantly faster (a few 100 ms) than CPU-only execution (a few seconds) due to the large problem size and high parallelization capability of the GPU.

## Project Structure

- `accelerators/` - TAIDL accelerator implementations
  - `*/` - Accelerator implementation directory
    - `TAIDL_*.py` - ISA definition using TAIDL
    - `tests/` - Kernel implementations and test runner
- `artifact-baseline/` - Baseline Docker environment and evaluation scripts
  - `amx/` - Intel AMX baseline kernels and benchmarking scripts
  - `gemmini/` - Gemmini baseline with Spike simulator integration
- `artifact-taidl/` - TAIDL Docker environment for multi-architecture support
  - `xla-debug/` - C++ XLA custom call for debugging tensor data
- `idl/` - TAIDL language infrastructure for generating simulation code
- `plots/` - Visualization scripts and output data
  - `lite_run_<timestamp>/` - Generated results from `lite.sh` runs
  - `full_run_<timestamp>/` - Generated results from `full.sh` runs
  - `paper_data/` - Paper's benchmarking data for quick plot generation
- `scripts/` - Automation scripts
  - `setup.sh` - Setup TAIDL and baseline Docker environments
  - `kick-tires.sh` - Quick plot generation using pre-evaluated data
  - `lite.sh` - Run tests with subset of data
  - `full.sh` - Complete test suite with data regeneration
  - `launch.sh` - Launch TAIDL Docker environment for interactive use
  - `clean.sh` - Cleanup generated files including logs, CSVs, and PDFs

# Getting Started with TAIDL

## Writing Custom ISAs in TAIDL

Here, we provide a step-by-step guide to defining a custom ISA using TAIDL and writing kernels to run on the generated TAIDL-TO simulator.

First, launch our provided TAIDL docker environment using

```bash
./scripts/launch.sh
```

The TAIDL environment is at `/taidl/` in the Docker.

#### 1. Define Your ISA

Create a new `toy/` directory in `accelerators/` and define your ISA in `TAIDL_toy.py`:

```python
# accelerators/toy/TAIDL_toy.py
import importlib
import os
import sys

# Start: Import the TAIDL API #
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(repo_dir))

Accelerator = importlib.import_module("idl.accelerator").Accelerator
# End: Import the TAIDL API #

# Initialize a new accelerator instance
acc = Accelerator("Toy")

# Define data model (memory space)
acc.add_data_model("regs", "32", "16xs8")  # 32 registers with 16 elements each
# s8 indicates 8-bit signed integers

# Define instruction: load from HBM to register
instr = acc.add_instruction("load", ["dst", "addr"])
instr.add_semantics("""
%data:16xs8 <- hbm[@a.addr:@a.addr + 16];
%reshaped:1x16xs8 = reshape(%data);
%reshaped:1x16xs8 -> regs[@a.dst, 0];
""")

# Define instruction: store from register to HBM
instr = acc.add_instruction("store", ["src", "addr"])
instr.add_semantics("""
%data:1x16xs8 <- regs[@a.src:@a.src+1, 0:16];
%flattened:16xs8 = reshape(%data);
%flattened:16xs8 -> hbm[@a.addr];
""")

# Define instruction: add two registers
instr = acc.add_instruction("add", ["dst", "src1", "src2"])
instr.add_semantics("""
%a:1x16xs8 <- regs[@a.src1:@a.src1+1, 0:16];
%b:1x16xs8 <- regs[@a.src2:@a.src2+1, 0:16];
%c:1x16xs8 = add(%a, %b);
%c:1x16xs8 -> regs[@a.dst, 0];
""")

# Generate TAIDL-TO library
acc.generate_sim()
```

**Current directory structure**:

```
accelerators/toy/
└── TAIDL_toy.py             # ISA definition
```

#### 2. Generate Simulation Code

Run your TAIDL definition to generate the TAIDL-TO simulation library:

```bash
cd /taidl/accelerators/toy && python3 TAIDL_toy.py
```

This creates the `sim/` directory with:

- `api.py` - Operation APIs for your ISA
- `decorator.py` - Kernel compilation framework
- `utils.py` - Helper functions

**Current directory structure**:

```
accelerators/toy/
├── sim                      # Generated simulation code
│   ├── api.py
│   ├── decorator.py
│   └── utils.py
└── TAIDL_toy.py
```

#### 3. Write Kernels

Now, let's write kernels in `kernels.py` using your generated TAIDL-TO library:

```python
# accelerators/toy/kernels.py
import numpy as np
from sim import api
from sim.decorator import kernel


@kernel(hbm=1024,
        input=[
            {'addr': 0, 'shape': (16,), 'dtype': np.int8},
            {'addr': 16, 'shape': (16,), 'dtype': np.int8},
        ],
        output=[
            {'addr': 32, 'shape': (16,), 'dtype': np.int8},
        ])
def add_kernel():
    api.load(dst=0, addr=0)
    api.load(dst=1, addr=16)
    api.add(dst=2, src1=0, src2=1)
    api.store(src=2, addr=32)
```

**Current directory structure**:

```
accelerators/toy/
├── kernels.py               # New kernel: add_kernel
├── sim
│   ├── api.py
│   ├── decorator.py
│   └── utils.py
└── TAIDL_toy.py
```

#### 4. Test Your Kernels

Create a kernel runner `run_add.py` to execute and verify your kernel:

```python
# accelerators/toy/run_add.py
import numpy as np
from kernels import add_kernel
from sim.decorator import set_simulation_backend

# Generate random input data
np.random.seed(0)
a = np.random.randint(-40, 40, size=16, dtype=np.int8)
b = np.random.randint(-40, 40, size=16, dtype=np.int8)

print("A: \t", a)
print("B: \t", b)

# Compile the simulation
set_simulation_backend("CPU")
_, _ = add_kernel("fsim-compile")()

# Run the simulation
outputs, _ = add_kernel("fsim")(a, b)
print("Sum: \t", outputs[0])
```

Run the kernel to see the output:

```bash
cd /taidl/accelerators/toy && python3 run_add.py
```

**Expected output**:

```
A:       [  4 -30 -28   7   2  28 -17  -2 -18  15  24  36  -7  27  11  38]
B:       [ -6  27 -29 -31  -4  -7  -9 -19  -3  -4  15 -35  18   3  36  30]
Sum:     [ -2  -3 -57 -24  -2  21 -26 -21 -21  11  39   1  11  30  47  68]
```

**Current directory structure**:

```
accelerators/toy/
├── kernels.py
├── run_add.py               # Kernel runner
├── sim
│   ├── api.py
│   ├── decorator.py
│   └── utils.py
└── TAIDL_toy.py
```

#### 5. Debugging

TAIDL-TO allows you to inspect register and memory contents during kernel execution using the `api.debug()` function.

Now, edit the `add_kernel` to include debugging statements.

```python
# accelerators/toy/kernels.py
@kernel(hbm=1024, input=[...], output=[...])
def add_kernel():
    api.load(dst=0, addr=0)
    api.load(dst=1, addr=16)
    api.add(dst=2, src1=0, src2=1)
    api.debug(prefix="reg0", data="regs[0]")  # New code
    api.debug(prefix="result(reg2)", data="regs[2]")  # New code
    api.store(src=2, addr=32)
```

Run the kernel again to see the output along with debug information:

```bash
cd /taidl/accelerators/toy && python3 run_add.py
```

**Expected output**:

```
A:       [  4 -30 -28   7   2  28 -17  -2 -18  15  24  36  -7  27  11  38]
B:       [ -6  27 -29 -31  -4  -7  -9 -19  -3  -4  15 -35  18   3  36  30]
reg0:
shape: (1, 16)
[  [4, -30, -28, 7, 2, 28, -17, -2, -18, 15, 24, 36, -7, 27, 11, 38]]
result(reg2):
shape: (1, 16)
[  [-2, -3, -57, -24, -2, 21, -26, -21, -21, 11, 39, 1, 11, 30, 47, 68]]
Sum:     [ -2  -3 -57 -24  -2  21 -26 -21 -21  11  39   1  11  30  47  68]
```

Note that while the input variables `A` and `B` are 1-D tensors of shape `(16,)`, the debug output shows the register contents as 2-D tensor of shape `(1, 16)`.  
A register file is represented as a 2-D tensor of shape `(num_registers, elements_per_register)` with the first dimension indexing the registers (more details in the paper), with its slice `regs[0]` being a 2-D tensor of shape `(1, elements_per_register)`.  
This behavior is intended since scratchpads often access multiple rows at once, unlike traditional register files that access one register at a time.

**Current directory structure**:

```accelerators/toy/
├── kernels.py               # Edited kernel: add_kernel
├── run_add.py
├── sim
│   ├── api.py
│   ├── decorator.py
│   └── utils.py
└── TAIDL_toy.py
```

#### 6. Python native loops

Create a new kernel `loop_kernel` in `kernels.py` to add 50 vectors using a for loop.

```python
# accelerators/toy/kernels.py
@kernel(hbm=1024,
        input=[  # 50 vectors of 16 elements
            {'addr': 0, 'shape': (50, 16), 'dtype': np.int8},
        ],
        output=[  # Sum of the 50 vectors
            {'addr': 800, 'shape': (16,), 'dtype': np.int8},
        ])
def loop_kernel():
    api.load(dst=0, addr=0)  # Load first vector to initialize reg[0]

    for i in range(1, 50):              # Native for loop
        api.load(dst=1, addr=16 * i)    # Load vector i
        api.add(dst=0, src1=0, src2=1)  # Accumulate into dst=0

    api.store(src=0, addr=800)  # Store final accumulated result
```

Next, create a corresponding kernel runner `run_loop.py`:

```python
# accelerators/toy/run_loop.py
import numpy as np
from kernels import loop_kernel
from sim.decorator import set_simulation_backend

# Generate random input data: 50 vectors of 16 elements
np.random.seed(0)
vectors = np.random.randint(-2, 2, size=(50, 16), dtype=np.int8)

# Compile the simulation
set_simulation_backend("CPU")
_, compile_time = loop_kernel("fsim-compile")()

# Run the simulation
outputs, runtime = loop_kernel("fsim")(vectors)

golden = np.sum(vectors, axis=0)
print("Golden: \t", golden)
print("Result: \t", outputs[0])
assert np.array_equal(golden, outputs[0]), "Result does not match golden!"
print("Test passed!")

print("\nBenchmarking statistics:")
print(f"Compilation time: {compile_time:.3f} ms")
print(f"Simulation time: {runtime:.3f} ms")
```

Run the kernel to see the output:

```bash
cd /taidl/accelerators/toy && python3 run_loop.py
```

**Expected output**: (benchmarking statistics may vary based on your machine)

```
Golden:          [-13 -28 -14 -37 -19 -12 -18 -14 -23 -27 -27 -18 -30 -24 -34 -17]
Result:          [-13 -28 -14 -37 -19 -12 -18 -14 -23 -27 -27 -18 -30 -24 -34 -17]
Test passed!

Benchmarking statistics:
Compilation time: 116.406 ms
Simulation time: 12.657 ms
```

**Current directory structure**:

```accelerators/toy/
├── kernels.py               # New kernel: loop_kernel
├── run_add.py
├── run_loop.py              # New kernel runner: run_loop.py
├── sim
│   ├── api.py
│   ├── decorator.py
│   └── utils.py
└── TAIDL_toy.py
```

#### 7. TAIDL-TO loop API for faster compilation

We observed that using native Python loops can lead to slow compilation times for larger loop counts since the entire loop body is unrolled during compilation of the simulation.  
To address this, TAIDL-TO provides a loop API (`api.start_loop`, `api.end_loop`) that allows you to define loops that are handled directly by the simulator without unrolling.

Curious how these loops are implemented? Hint: XLA-HLO IR has a `while` loop operator that you can read about [here](https://openxla.org/xla/operation_semantics#while).

Now, edit the `loop_kernel` in `kernels.py` to use the TAIDL-TO loop API:

```python
# accelerators/toy/kernels.py
@kernel(hbm=1024, inputs=[...], outputs=[...])
def loop_kernel():
    api.load(dst=0, addr=0)  # Load first vector to initialize reg[0]

    api.start_loop("i", 1, 50)       # (End value is exclusive)
    api.load(dst=1, addr="16 * %i + 0")  # Load vector i
    api.add(dst=0, src1=0, src2=1)   # Accumulate into dst=0
    api.end_loop()

    api.store(src=0, addr=800)  # Store final accumulated result
```

Run the kernel again to see the output:

```bash
cd /taidl/accelerators/toy && python3 run_loop.py
```

**Expected output**: (benchmarking statistics may vary based on your machine)

```
Golden:          [-13 -28 -14 -37 -19 -12 -18 -14 -23 -27 -27 -18 -30 -24 -34 -17]
Result:          [-13 -28 -14 -37 -19 -12 -18 -14 -23 -27 -27 -18 -30 -24 -34 -17]
Test passed!

Benchmarking statistics:
Compilation time: 17.955 ms
Simulation time: 12.668 ms
```

You'll likely notice a significant speedup in compilation time compared to using native Python loops, especially as the loop count increases with little to no change in simulation time.

**Current directory structure**:

```accelerators/toy/
├── kernels.py               # Edited kernel: loop_kernel
├── run_add.py
├── run_loop.py
├── sim
│   ├── api.py
│   ├── decorator.py
│   └── utils.py
└── TAIDL_toy.py
```

## TAIDL API Reference

### Supported Operations

**Arithmetic Operations:**

- `add(A, B)` - Element-wise addition
- `subtract(A, B)` - Element-wise subtraction
- `multiply(A, B)` - Element-wise multiplication
- `divide(A, B)` - Element-wise division

**Math Functions:**

- `exp(A)` - Element-wise exponential
- `tanh(A)` - Element-wise hyperbolic tangent
- `maximum(A, B)` - Element-wise maximum
- `minimum(A, B)` - Element-wise minimum

**Logic Operations:**

- `xor(A, B)` - Bitwise XOR

**Shape Operations:**

- `reshape(A)` - Reshape tensor
- `transpose(A, dimensions={...})` - Transpose tensor
- `concatenate(A)` - Concatenate tensors
- `slice(A, slice={...})` - Extract slice
- `dynamic_update_slice(A, B, dims)` - Update slice

**Data Type Operations:**

- `convert(A)` - Convert data type
- `bitcast_convert(A)` - Bitcast conversion

**Linear Algebra:**

- `dot(A, B, lhs_batch_dims={...}, lhs_contracting_dims={...}, rhs_batch_dims={...}, rhs_contracting_dims={...})` - Matrix multiplication

**Broadcast & Constants:**

- `broadcast(A)` - Broadcast tensor
- `broadcast_type(A)` - Type-aware broadcast
- `constant(value)` - Create constant tensor

**Reduction:**

- `reduce(A, B, dims, operation)` - Reduce along dimensions. Right now,
  the only options for `operation` are `add_f32`, `max_f32`. (ADD MORE)

**Conditionals:**

- `select_lt(A, B, C, D)` - Select based on less-than comparison
- `clamp(min, A, max)` - Clamp values to range

### Control Flow

**Conditionals:**

```
IF(condition)
{
    // statements
}
```

**Loops\*:**

```
REPEAT(variable, range)
{
    // statements using @l.variable
}
```

\* While REPEAT blocks are supported, it is _highly_ recommended for speed of compilation that you modify your tensor shapes and operations so a REPEAT is not necessary. We have an example of this in `TAIDL_amx.py` where we have two versions of the instruction `tdpbusd`, one with and without REPEAT.
