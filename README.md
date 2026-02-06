# GPUBenchmark.jl 🚀

A professional, standalone Julia GPU benchmarking suite designed for **HPC node validation and burn-in**.

This tool is used to verify that a freshly provisioned GPU node is stable under peak load, correctly cooled, and delivering its theoretical performance.

---

## 🛠 Manual Installation (Standalone)

If you are not using the Ansible provisioning role and want to run this manually:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nicolamos/GPUBenchmark.jl.git
   cd GPUBenchmark.jl
   ```

2. **Initialize the Environment:**
   This will download all dependencies (CUDA, GPUInspector, CairoMakie) and solve the environment for your specific hardware.
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
   ```

---

## 📊 Usage Guide

The tool uses a task-based system. Running without arguments will perform a quick system check.

### 1. Quick Hardware Report (Default)
Check if GPUs are detected and show their basic capabilities.
```bash
julia --project -m GPUBenchmark
```

### 2. Full System Validation
Runs the hardware report, a raw TFLOPS benchmark, and a parallel monitored stress test on all GPUs.
```bash
julia --project -m GPUBenchmark all
```

### 3. Tuning the Stress Test
You can control the intensity and duration of the burn-in:
```bash
# Run for 2 minutes using 50% of available VRAM
julia --project -m GPUBenchmark --duration 120 --fraction 0.5 gpuinspector
```

---

## 📋 Available Benchmark Tasks

| Task | Description | Output |
| :--- | :--- | :--- |
| `sysinfo` | Hardware audit: GPU model, VRAM capacity, Compute Capability, PCI IDs. | `summary.txt` |
| `matmul` | Raw compute test: Performs massive FP32 Matrix Multiplications. | TFLOPS Score |
| `gpuinspector` | Parallel burn-in: Stress tests **all** GPUs simultaneously with real-time telemetry. | `dashboard.png`, `telemetry.h5` |
| `all` | Sequentially runs all tasks listed above. | Full Report |

---

## ⚙️ CLI Reference

| Flag | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `-d` | `--duration` | Stress test duration in seconds | `30` |
| `-f` | `--fraction` | Target VRAM usage fraction (0.0 to 1.0) | `0.4` |
| `-s` | `--size` | Manual matrix size (N). If 0, auto-calculates. | `0` |
| `-o` | `--output-dir` | Root directory for results | `results` |
| `-v` | `--verbose` | Enable debug logging (level DEBUG) | - |
| `-l` | `--list` | List all available benchmark tasks | - |

---

## 📂 Understanding the Results

Every run creates a timestamped folder: `results/<hostname>/<timestamp>/`

1. **`summary.txt`**: The "Health Certificate". A human-readable report of system info and performance scores.
2. **`dashboard.png`**: A visual chart showing Power, Temperature, and Utilization over time.
3. **`metrics.json`**: Structured data for automation or CI/CD pipelines (e.g., ARA).
4. **`telemetry.h5`**: Raw HDF5 time-series data for deep scientific analysis.
5. **`benchmark.log`**: Complete execution logs. If a task fails, check this for the stacktrace.

---

## 🏗 Developer Notes

To add a new benchmark task, create a file in `src/benchmarks/` and use the `register_benchmark` function:

```julia
register_benchmark("my_test", "Description of test", run_my_test_function)
```

## ⚖️ License
MIT / Apache 2.0