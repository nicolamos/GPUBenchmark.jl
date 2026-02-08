# GPUBenchmark.jl 🚀

A standalone Julia GPU benchmarking suite designed for **HPC node validation and burn-in**.

This tool verifies that a freshly provisioned GPU node is stable under load and measures its compute performance. It uses a **plugin-based architecture** via Julia extensions to keep the core lightweight while offering powerful telemetry and stress-testing capabilities.

---

## ⚠️ Prerequisites & Caveats

- **Hardware:** Optimized for **NVIDIA GPUs** (Compute Capability 6.0+).
- **OS:** Linux (x86_64) is the primary target for HPC validation.
- **CUDA:** The package includes `CUDA.jl`. For production validation, benchmarking against the **local system toolkit** is recommended (see HPC Considerations).

---

## 💿 Step 1: Installation & Setup

Choose the workflow that fits your needs. We recommend the **Shared Environment** for most HPC users.

### Path A: Shared Named Environment (Recommended)
Keeps your global environment clean while providing a dedicated test suite.

1.  **Open Julia** and enter **Pkg mode** by pressing `]`.
2.  **Setup the environment:**
    ```julia
    pkg> activate --shared gpu-test
    pkg> add https://github.com/nicolamos/GPUBenchmark.jl
    pkg> add GPUInspector CairoMakie
    ```
    *(Adding `GPUInspector` and `CairoMakie` enables the dashboard and burn-in features.)*
3.  **Run the suite:**
    ```bash
    julia --project=@gpu-test -m GPUBenchmark all
    ```

---

### Path B: Global CLI Tool (App Mode)
Installs a standalone `gpu_benchmark` command to your PATH.

1.  **Install the App:**
    ```julia
    pkg> app add https://github.com/nicolamos/GPUBenchmark.jl
    ```
2.  **Enable Plugins (Tweak the Private Env):**
    Apps have isolated environments. To enable dashboards, you must add the plugins to its private project:
    ```bash
    julia --project=$HOME/.julia/apps/GPUBenchmark -e 'using Pkg; Pkg.add(["GPUInspector", "CairoMakie"])'
    ```
3.  **Run the tool:**
    ```bash
    gpu_benchmark all
    ```

---

> [!IMPORTANT]
> **Standalone Environment (`--project=.`)**
> Running directly from a cloned directory using `julia --project=.` is **not recommended** for production health checks. This is because the extension dependencies (`GPUInspector`, `CairoMakie`) are "weak" and won't be triggered unless they are explicitly added to the environment. Use the `@gpu-test` method instead to keep your development environment clean while having a fully-featured test environment.

---

## 📊 Usage Guide

### Common Commands
```bash
# Run everything (System Audit + MatMul + Parallel Burn-in)
julia --project=@gpu-test -m GPUBenchmark all

# Stress test for a specific duration (seconds)
julia --project=@gpu-test -m GPUBenchmark --duration 120 gpuinspector

# Headless mode (Generates all files but skips terminal dashboard)
julia --project=@gpu-test -m GPUBenchmark --quiet all

# Re-view the results of the latest run
julia --project=@gpu-test -m GPUBenchmark --show-latest
```

### Available Tasks

| Task | Level | Description |
| :--- | :--- | :--- |
| `sysinfo` | **Core** | Hardware audit: GPU model, VRAM, PCI IDs. |
| `matmul` | **Core** | Raw compute: FP32 TFLOPS via matrix operations. |
| `gpuinspector`| **Ext** | **Burn-in:** Parallel stress test with telemetry (Requires Plugins). |
| `all` | - | Runs all available tasks sequentially. |

---

## 🏗️ HPC Deployment Considerations

### Threading on Multi-core Nodes
On nodes with high core counts (100+), `julia --threads auto` may cause excessive overhead. **Always prefer an explicit thread count** (e.g., `--threads 8`).
```bash
julia --project=@gpu-test --threads 8 -m GPUBenchmark all
```

### Julia Depot & Shared Filesystems
By default, packages are installed in `~/.julia`. If you are limited by disk quotas or want to use a shared installation, use `JULIA_DEPOT_PATH`:
```bash
# Example: Use a shared HPC software stack but keep your home for private settings
export JULIA_DEPOT_PATH="/apps/software/julia/depot:$HOME/.julia"
```
*Note: The first path in the list must be writable to install new packages.*

### Using System CUDA
To benchmark against the specific CUDA version installed on your host OS (instead of the artifacts downloaded by Julia):
```julia
using CUDA
# Must be set before running benchmarks
CUDA.set_runtime_version!(v"12.4", local_toolkit=true)
```

---

## 📂 Understanding Results

Results are saved to `results/<hostname>/<timestamp>/`.

- **`summary.txt`**: Human-readable report of specs and scores.
- **`metrics.json`**: Structured data for automation/CI.
- **`dashboard.png`**: (Ext) Visual chart of Power, Temp, and Utilization.
- **`telemetry.h5`**: (Ext) Raw sensor data from the burn-in.

---

## ⚖️ License
MIT / Apache 2.0
