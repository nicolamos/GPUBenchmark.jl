# GPUBenchmark.jl

A standalone Julia GPU benchmarking suite designed for **HPC node validation and burn-in**.

## Features

- **Modular Architecture**: Easy to extend with custom algorithms.
- **Scaling Analysis**: Measure performance across problem sizes and multiple GPUs.
- **Parallel Probing**: Identify system-wide bottlenecks.
- **Hybrid Plugin System**: Drop modules into a folder to add new benchmarks.
- **Standard Reports**: Generates JSON, Text, and Tabular (.dat) results.

## Quick Start

```bash
julia --project -m GPUBenchmark all
```

## Installation

```julia
using Pkg
Pkg.add("https://github.com/nicolamos/GPUBenchmark.jl")
```
