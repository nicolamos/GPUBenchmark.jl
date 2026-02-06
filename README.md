# GPUBenchmark.jl 🚀

A professional, standalone Julia GPU benchmarking suite designed for **HPC node validation and burn-in**.

This tool is used to verify that a freshly provisioned GPU node is stable under peak load, correctly cooled, and delivering its theoretical performance.

---

## ⚠️ Prerequisites & Caveats

- **Hardware:** Optimized for **NVIDIA GPUs** (Compute Capability 6.0+).
- **OS:** Linux (x86_64) is the primary target for HPC validation.
- **CUDA:** The package includes `CUDA.jl` and uses Julia-managed artifacts by default. For production-grade validation, using the **local system toolkit** is recommended (see Advanced Usage).
- **Core vs. Extended Features:** 
    - **Core:** Hardware audit (`sysinfo`) and raw compute (`matmul`). Zero extra dependencies.
    - **Extended:** Parallel burn-in and real-time telemetry (`gpuinspector`). Requires manual installation of optional extensions (see Setup).

---

## 💿 Deployment Options

Choose the installation mode that fits your operational requirements.

### Mode 1: Global CLI Tool (Recommended 🌟)
Installs a standalone `gpu_benchmark` command. Best for dedicated validation nodes.

1. **Install the App:**
   ```julia
   julia> ]
   pkg> app add https://github.com/nicolamos/GPUBenchmark.jl
   ```

2. **Configure PATH:** Add `export PATH="$PATH:$HOME/.julia/bin"` to your `~/.bashrc`.
3. **Run:** `gpu_benchmark all`

### Mode 2: Shared Environment (Portable)
Keeps the benchmark suite in a versioned Julia environment without a global binary. Ideal for multi-user systems.

1. **Activate shared environment:**
   ```julia
   julia> ]
   pkg> activate --shared gpu-test
   ```

2. **Add package:** `(gpu-test) pkg> add https://github.com/nicolamos/GPUBenchmark.jl`
3. **Run:** `julia --project=@gpu-test -m GPUBenchmark all`

---

## 📊 Usage Guide

The tool uses a task-based system. Running without arguments performs a quick hardware audit.

### Execution Examples
```bash
# Basic usage (Global App)
gpu_benchmark all

# Advanced tuning via the '--' delimiter
# (Flags before '--' are for Julia, flags after are for the benchmark)
gpu_benchmark --threads=auto -- --duration 60 all

# Headless mode (No terminal dashboard)
# Ideal for CI/CD, Cron, or Batch jobs (Slurm/PBS)
gpu_benchmark -- --quiet all
```

---

## 📋 Available Benchmark Tasks

| Task | Level | Description |
| :--- | :--- | :--- |
| `sysinfo` | **Core** | Hardware audit: GPU model, VRAM, PCI IDs. |
| `matmul` | **Core** | Raw compute: Peak FP32 TFLOPS via massive matrix ops. |
| `gpuinspector`| **Ext** | **Burn-in:** Parallel stress test with telemetry. |
| `all` | - | Runs all available tasks sequentially. |

---

## 📂 Understanding the Results

Results are saved to `results/<hostname>/<timestamp>/`.

### 1. Core Artifacts (Always generated)
- **`summary.txt`**: The "Health Certificate". Human-readable report of specs and scores.
- **`metrics.json`**: Structured data for CI/CD pipelines or database ingestion.
- **`benchmark.log`**: Detailed execution traces and hardware events.

### 2. Extended Artifacts (Requires `GPUInspector` & `CairoMakie`)
- **`dashboard.png`**: Visual chart of Power, Temp, and Utilization during the burn-in.
- **`telemetry.h5`**: High-frequency raw sensor data for scientific analysis.

---

## 🚀 Advanced Usage & Extensions

### Enabling the Full Burn-in Suite
For meaningful stability testing, the `gpuinspector` extension is highly recommended. It provides the parallel load necessary to verify cooling and power delivery.

**Trade-offs:** Adding these will increase disk usage (~300MB) and precompilation time.
```julia
# In your chosen environment (e.g. @gpu-test):
pkg> add GPUInspector CairoMakie
```

### Performance Tuning: System CUDA
To benchmark against the specific CUDA version installed on your host OS:
```julia
using CUDA
CUDA.set_runtime_version!(v"12.4", local_toolkit=true)
```

## ⚖️ License
MIT / Apache 2.0
