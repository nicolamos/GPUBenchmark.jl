# GPUBenchmark.jl 🚀

A standalone Julia GPU benchmarking suite designed for **HPC node validation and burn-in**.

This tool is used to verify that a freshly provisioned GPU node is stable under load and to measure its compute performance.

---

## ⚠️ Prerequisites & Caveats

- **Hardware:** Optimized for **NVIDIA GPUs** (Compute Capability 6.0+).
- **OS:** Linux (x86_64) is the primary target for HPC validation.
- **CUDA:** The package includes `CUDA.jl`. For production validation, using the **local system toolkit** is an option (see Advanced Tuning).

---

## 💿 Step 1: Deployment

Choose the installation mode that fits your environment.

### Mode 1: Global CLI Tool
Installs a standalone `gpu_benchmark` command.

1. **Install the App:**
   ```julia
   julia> ]
   pkg> app add https://github.com/nicolamos/GPUBenchmark.jl
   ```
2. **Configure PATH:** Add `export PATH="$PATH:$HOME/.julia/bin"` to your `~/.bashrc`.
3. **Update (Maintenance):**
   ```julia
   julia> ]
   pkg> app update GPUBenchmark
   ```

### Mode 2: Shared Environment
Keeps the suite in a versioned Julia environment without a global binary.

1. **Activate shared environment:**
   ```julia
   julia> ]
   pkg> activate --shared gpu-test
   ```
2. **Add package:** `(gpu-test) pkg> add https://github.com/nicolamos/GPUBenchmark.jl`

---

## 🚀 Step 2: Enable Dashboards & Stress Tests

By default, the tool is lightweight and only performs core tests. To enable the **Dashboard** (`Term.jl`) and **Parallel Burn-in** (`GPUInspector.jl`), you must add the feature extensions.

### For Global CLI Tool (Mode 1)
Run this command to add stress testing capabilities to the app environment (visuals are now included by default):
```bash
julia --project=$HOME/.julia/apps/GPUBenchmark -e 'using Pkg; Pkg.add(["GPUInspector", "CairoMakie"])'
```

### For Shared Environment (Mode 2)
```julia
(gpu-test) pkg> add GPUInspector CairoMakie
```

---

## 📊 Usage Guide

### Execution Examples
```bash
# Basic usage (Global App)
gpu_benchmark all

# Advanced tuning via the '--' delimiter
# (Flags before '--' are for Julia, flags after are for the benchmark)
gpu_benchmark --threads=auto -- --duration 60 all

# Headless mode (No terminal dashboard)
gpu_benchmark -- --quiet all

# Show the results of the latest run
gpu_benchmark -- --show-latest
```

### Available Benchmark Tasks

| Task | Level | Description |
| :--- | :--- | :--- |
| `sysinfo` | **Core** | Hardware audit: GPU model, VRAM, PCI IDs. |
| `matmul` | **Core** | Raw compute: FP32 TFLOPS via matrix operations. |
| `gpuinspector`| **Ext** | **Burn-in:** Parallel stress test with telemetry (Requires Step 2). |
| `all` | - | Runs all available tasks sequentially. |

---

## 📂 Understanding & Viewing Results

Results are saved to `results/<hostname>/<timestamp>/`.

### 1. The Terminal Dashboard (CLI)
If you enabled extensions in Step 2, you can re-render the dashboard:

**Via Global App:**
```bash
gpu_benchmark -- --show-latest
```

**Via Shared Environment:**
```bash
julia --project=@gpu-test -m GPUBenchmark --show-latest
```

### 2. Filesystem Artifacts
- **`summary.txt`**: A human-readable report of specs and scores.
- **`metrics.json`**: Structured data for CI/CD pipelines.
- **`dashboard.png`**: (Ext) Visual chart of Power, Temp, and Utilization.
- **`telemetry.h5`**: (Ext) Raw sensor data.

---

## ⚙️ Advanced Performance Tuning

### Using System CUDA
To benchmark against the specific CUDA version installed on your host OS:
```julia
using CUDA
CUDA.set_runtime_version!(v"12.4", local_toolkit=true)
```

### Batch & Non-interactive Use
Use the `--quiet` flag to suppress the terminal dashboard while still generating all file artifacts.

## ⚖️ License
MIT / Apache 2.0
