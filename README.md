# TAIDL: Tensor Accelerator ISA Definition Language

## Overview

This is the artifact for our paper "TAIDL: Tensor Accelerator ISA Definition Language with Auto-generation of Scalable Test Oracles".
In our paper, we present an ISA specification language for tensor accelerators and auto-generated test oracles (a.k.a. functional simulators).

This artifact consists of TAIDL source code and the necessary scripts to reproduce the evaluation results.
To facilitate artifact evaluation, we have automated the entire environment setup and experimental processes as part of Docker images.
Our evaluation results were collected using Intel Xeon Platinum 8358 CPU and NVIDIA A100 GPU.
We recommend using a machine with an Intel CPU and an NVIDIA GPU to benchmark TAIDL-TO and its baselines.
Reproducing all simulation statistics takes approximately 30-45 minutes.

## Running TAIDL Artifact

### Getting Started

We use Docker images for environment setup of TAIDL and baselines.
To run the TAIDL artifact, install Docker using the [installation guide](https://docs.docker.com/engine/install/).

All experimental workflows are encapsulated as bash scripts located in the `scripts/` directory.
These scripts automatically pull and use the appropriate Docker images:

- `devanshdvj/taidl-micro25-artifact:amd64` - TAIDL environment for amd64/x86-64
- `devanshdvj/taidl-micro25-artifact:arm64` - TAIDL environment for arm64
- `devanshdvj/taidl-micro25-artifact:baseline-amd64` - Baseline environment with Gemmini Spike and Intel SDE to generate data and log simulation times.

### System Requirements

**Architecture Support:**

- **amd64/x86_64**: Full support for TAIDL including GPU acceleration and baselines.
- **arm64**: CPU-only support for TAIDL (no GPU support). The `full.sh` script cannot be run on arm64 since baselines are not supported on this architecture.

**GPU Support:**
For NVIDIA GPU usage on amd64 systems, install the NVIDIA Container Toolkit using the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Kick the Tires: Quick Plot Generation

This script uses paper's benchmarking data (`plots/saved/`) to quickly generate all figures without running any experiments.
These statistics were collected using Intel Xeon Platinum 8358 CPU and NVIDIA A100 GPU.

```bash
./scripts/kick-tires.sh
```

The resulting figures can be found in `plots/saved/`.

- `figure-16-gemmini-tiled-matmul.pdf` - Comparing simulation times of TAIDL-TO and Gemmini Spike
- `figure-17-oneDNN.pdf` - Comparing simulation times of TAIDL-TO and Intel SDE
- `figure-18-gemmini-exo.pdf` - Benchmarking TAIDL-TO for Exo-generated Gemmini kernels

### Only Benchmark TAIDL-TO using Pre-generated Data

This uses pre-generated inputs and golden outputs from Gemmini Spike and Intel SDE to benchmark TAIDL-TO.
It does not regenerate any data or run the baselines.
This is useful for quickly verifying TAIDL-TO's correctness and performance.
This would take around 2-5 minutes to run.

Run using:

```bash
./scripts/lite.sh
```

The resulting figures can be found in `plots/pdf/`.
More detailed statistics can be found in `plots/csv/`.

- `figure-16-gemmini-tiled-matmul.pdf` - Benchmarking TAIDL-TO for Gemmini's tiled matrix multiplication kernels
- `figure-17-oneDNN.pdf` - Benchmarking TAIDL-TO for oneDNN's Intel AMX kernels.
- `figure-18-gemmini-exo.pdf` - Benchmarking TAIDL-TO for Exo-generated Gemmini kernels

### Regenerate All Test Data and Benchmarking Results

This will benchmark TAIDL-TO along with baselines Gemmini Spike and Intel SDE.
The script will also generate new data files containing inputs and outputs from these tools, which are used to verify TAIDL-TO's output.
This would take around 30-45 minutes to run.

Run using:

```bash
./scripts/full.sh
```

The resulting figures can be found in `plots/pdf/`.
More detailed statistics is available in `plots/csv/`.

- `figure-16-gemmini-tiled-matmul.pdf` - Comparing simulation times of TAIDL-TO and Gemmini Spike
- `figure-17-oneDNN.pdf` - Comparing simulation times of TAIDL-TO and Intel SDE
- `figure-18-gemmini-exo.pdf` - Benchmarking TAIDL-TO for Exo-generated Gemmini kernels

## Project Structure

- `accelerators/` - TAIDL accelerator implementations
  - `*/` - Accelerator implementation directory
    - `TAIDL_*.py` - ISA definition using TAIDL
    - `sim/` - Generated simulation code (API, decorator, utils)
    - `tests/` - Kernel implementations and test runner
- `artifact-baseline/` - Reference implementations for comparison
  - `amx/` - Intel AMX baseline kernels and benchmarking scripts
  - `gemmini/` - Gemmini baseline with Spike simulator integration
- `artifact-taidl/` - TAIDL Docker environment for multi-architecture support
  - `xla-debug/` - C++ XLA custom call for debugging tensor data
- `idl/` - TAIDL language infrastructure for generating simulation code
- `plots/` - Visualization scripts and output data
  - `csv/` - Benchmarking and verification data files
  - `pdf/` - Generated comparison plots
  - `saved/` - Paper's benchmarking data for quick plot generation
- `scripts/` - Automation scripts
  - `kick-tires.sh` - Quick plot generation from saved data
  - `lite.sh` - Run tests with subset of data
  - `full.sh` - Complete test suite with data regeneration
  - `launch.sh` - Launch TAIDL Docker environment

## Writing Custom ISAs in TAIDL

Here is a simple example of a TAIDL workflow.

First, launch our provided docker environment using

```
./scripts/launch.sh
```

The TAIDL environment is at `/taidl/` in the Docker.

#### 1. Define Your ISA

Create a new `toy/` directory in `accelerators/` and define your ISA in `TAIDL_toy.py`:

```python
import os, sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir = os.path.join(os.path.dirname(base_dir), "idl")
sys.path.append(target_dir)

from accelerator import Accelerator

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

acc.generate_api()
```

#### 2. Generate Simulation Code

Run your TAIDL definition to generate the simulation environment:

```bash
cd /taidl/accelerators/toy
python3 TAIDL_toy.py
```

This creates the `sim/` directory with:

- `api.py` - Operation APIs for your ISA
- `decorator.py` - Kernel compilation framework
- `utils.py` - Helper functions

#### Directory Structure

After completing the steps above, your `accelerators/toy/` directory should look like:

```
accelerators/toy/
├── TAIDL_toy.py           # ISA definition (step 1)
├── sim/                   # Generated simulation code (step 2)
│   ├── api.py
│   ├── decorator.py
│   └── utils.py
└── tests/                 # Your kernel implementations (steps 3-6)
    ├── kernels.py         # Kernel definitions
    └── main.py            # Test runner
```

#### 3. Write Kernels

Create `tests/kernels.py` to define kernels using your generated API:

```python
# Import the generated TAIDL-TO API
import os, sys
base_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(os.path.dirname(base_dir), "sim")
sys.path.append(target_dir)
from decorator import kernel
import api

import numpy as np

@kernel(hbm=1024,
        input=[
            {'addr': 0, 'shape': (16,), 'dtype': np.int8},
            {'addr': 16, 'shape': (16,), 'dtype': np.int8},
        ],
        output=[
            {'addr': 32, 'shape': (16,), 'dtype': np.int8},
        ])
def my_kernel():
    api.load(dst = 0, addr = 0)
    api.load(dst = 1, addr = 16)
    api.add(dst = 2, src1=0, src2=1)
    api.store(src = 2, addr = 32)
```

#### 4. Test Your Kernels

Create `tests/main.py` to run and verify your kernels:

```python
from kernels import my_kernel
from decorator import set_simulation_backend, verifier
import numpy as np

# Generate random input data
a = np.random.randint(-10, 10, size=16, dtype=np.int8)
b = np.random.randint(-10, 10, size=16, dtype=np.int8)
print("Input A:", a)
print("Input B:", b)

set_simulation_backend("CPU")
_, compile_time = my_kernel("fsim-compile")()
outputs, runtime = my_kernel("fsim")(a, b)
print("Sum: \t", outputs[0])
```

Run the test with:

```bash
cd /taidl/accelerators/toy/tests
python3 main.py
```

#### 5. Debugging

Modify `tests/kernels.py` to use `api.debug()` to inspect register and memory contents during execution:

```python
@kernel(hbm=1024, input=[...], output=[...])
def debug_kernel():
    api.load(dst=0, addr=0)
    api.load(dst=1, addr=16)
    api.add(dst=2, src1=0, src2=1)

    # Debug register contents
    api.debug(prefix="reg0", data="regs[0]")
    api.debug(prefix="result(reg2)", data="regs[2]")

    api.store(src=2, addr=32)
```

#### 6. Loops

Modify `tests/kernels.py` to use `api.start_loop("loop_var", start, end)` and `api.end_loop()` instead of native Python loops for faster compilation:

```python
@kernel(hbm=1024,
        input=[  # 4 vectors of 16 elements
            {'addr': 0, 'shape': (4, 16), 'dtype': np.int8},
        ],
        output=[  # Sum of the 4 vectors
            {'addr': 256, 'shape': (16,), 'dtype': np.int8},
        ])
def loop_kernel():
    api.load(dst=0, addr=0)  # Load first vector to initialize reg[0]

    api.start_loop("i", 1, 4)             # (End value is exclusive)
    api.load(dst=1, addr=f"16 * %i + 0")  # Load vector i
    api.add(dst=0, src1=0, src2=1)        # Accumulate into dst=0
    api.end_loop()

    api.store(src=0, addr=256)  # Store final accumulated result
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

\* While REPEAT blocks are supported, it is _highly_ recommended for speed of compilation that you modify your tensor shapes and operations so a REPEAT is not necessary. We have an example of this in TAIDL_AMX.py where we have two versions of the instruction `tdpbusd`, one with and without REPEAT.
