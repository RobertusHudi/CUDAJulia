# CUDAJulia: Performance Comparison of Naive Matrix Multiplication in C and Julia

This repository benchmarks the performance of naive matrix multiplication implementations using CUDA in both C and Julia. The goal is to compare execution speed and efficiency across different implementations.

## Overview
Matrix multiplication is a fundamental operation in scientific computing. This project implements and benchmarks naive GPU-accelerated matrix multiplication using CUDA in C and Julia.

## Repository Structure
```
CUDAJulia/
│── Results_and_Reports/      # Contains benchmarking results and reports
│── Source/                   # Source code for the project
│   │── .vscode/              # VSCode settings and configurations
│   │── cpuNaive              # Executable for CPU-based naive multiplication
│   │── cpuNaive.cpp          # C++ implementation for CPU naive multiplication
│   │── mmJNaive.jl           # Julia CUDA naive matrix multiplication
│   │── mmNaive               # Executable for CUDA C naive multiplication
│   │── mmNaive.cu            # CUDA C naive implementation
│── LICENSE                   # License file
│── README.md                 # Project documentation
```

## Prerequisites
- **Hardware**: NVIDIA GPU with CUDA support.
- **Software**:
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - [Julia](https://julialang.org/downloads/)
  - [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU computing in Julia.

## Setup Instructions

### C Implementation
1. **Navigate to the Source directory**:
   ```bash
   cd Source/
   ```
2. **Compile the CUDA C program**:
   ```bash
   nvcc -o mmNaive mmNaive.cu
   ```
3. **Run the compiled program**:
   ```bash
   ./mmNaive
   ```

### Julia Implementation
1. **Navigate to the Source directory**:
   ```bash
   cd Source/
   ```
2. **Install the required Julia packages**:
   ```julia
   using Pkg
   Pkg.add("CUDA")
   ```
3. **Run the Julia script**:
   ```julia
   include("mmJNaive.jl")
   ```

## Performance Evaluation
Each implementation prints the execution time for matrix multiplication, allowing for a direct performance comparison between CUDA C and Julia implementations.

### Profiling with NVIDIA Nsight Systems (nsys)
To perform a detailed performance analysis using Nsight Systems, run the following command to generate an `.nsys-rep` report:

#### Profiling CUDA C Implementation:
```bash
nsys profile -o results/nsys_cuda_report ./mmNaive
```

#### Profiling Julia Implementation:
```bash
nsys profile -o results/nsys_julia_report julia -e 'include("mmJNaive.jl")'
```

The profiling results are stored in the `Results_and_Reports` folder for further analysis.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

