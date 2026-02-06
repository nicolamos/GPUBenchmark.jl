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

# Show the results of the latest run
gpu_benchmark -- --show-latest

# Show the results of a specific run
gpu_benchmark -- --show results/cloud-kvm-gpu-001/2026-02-06_224508
```

---

## 📂 Understanding & Viewing Results

Results are saved to `results/<hostname>/<timestamp>/`. You can view them in several ways:

### 1. The Terminal Dashboard (CLI)
If you have the **Visuals Extension** installed (`Term`, `UnicodePlots`), you can re-render the dashboard:
```bash
# Show latest run found in the results directory
gpu_benchmark -- --show-latest

# Show a specific run path
gpu_benchmark -- --show results/my-node/2026-02-06_120000
```

### 2. Julia Scripting & REPL
For more control, you can use the Julia API to inspect specific runs:
```julia
using GPUBenchmark

# Show latest run from default directory
GPUBenchmark.show_latest()

# Show a specific run by path
GPUBenchmark.show_results("results/my-node/2026-02-06_120000")
```

### 3. Filesystem Artifacts
- **`summary.txt`**: The "Health Certificate". Human-readable report of specs and scores.
- **`metrics.json`**: Structured data for CI/CD pipelines or database ingestion.
- **`dashboard.png`**: (Extended) Visual chart of Power, Temp, and Utilization.
- **`telemetry.h5`**: (Extended) High-frequency raw sensor data.

---

## 📋 Available Benchmark Tasks

| Task | Level | Description |
| :--- | :--- | :--- |
| `sysinfo` | **Core** | Hardware audit: GPU model, VRAM, PCI IDs. |
| `matmul` | **Core** | Raw compute: Peak FP32 TFLOPS via massive matrix ops. |
| `gpuinspector`| **Ext** | **Burn-in:** Parallel stress test with telemetry. |
| `all` | - | Runs all available tasks sequentially. |

---

## ⚙️ CLI Reference

| Flag | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `-d` | `--duration` | Stress test duration in seconds | `30` |
| `-f` | `--fraction` | Target VRAM usage fraction (0.0 to 1.0) | `0.4` |
| `-s` | `--size` | Manual matrix size (N). If 0, auto-calculates. | `0` |
| `-o` | `--output-dir` | Root directory for results | `results` |
| `-v` | `--verbose` | Enable debug logging (level DEBUG) | - |
| `-q` | `--quiet` | Suppress terminal dashboard (Batch mode) | - |
| `-S` | `--show` | Show results of the latest run and exit | - |
| `-l` | `--list` | List all available benchmark tasks | - |

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
