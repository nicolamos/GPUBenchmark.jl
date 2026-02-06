# GPUBenchmark.jl

A standalone Julia GPU benchmarking suite designed for HPC node validation and burn-in.

## 🚀 Key Features

- **Smart Auto-Sizing**: Automatically calculates matrix sizes based on available VRAM to ensure hardware saturation without OOM errors.
- **Deep Inspection**: Leverages `GPUInspector.jl` for high-fidelity performance metrics.
- **Visual Dashboard**: Automatically generates a tiled PNG dashboard showing:
  - GPU Utilization (Compute & Memory)
  - Power Usage (W)
  - Temperature (°C)
- **HPC Ready**: Designed to run in isolated environments with structured, timestamped output (JSON, HDF5, TXT).

## 🛠 Installation

```bash
git clone https://github.com/nicolamos/GPUBenchmark.jl.git
cd GPUBenchmark.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## 📊 Usage

### Run all benchmarks
```bash
julia --project -m GPUBenchmark all
```

### Run specific deep inspection
```bash
julia --project -m GPUBenchmark gpuinspector
```

### Manual Size Override
```bash
julia --project -m GPUBenchmark --size 16384 gpuinspector
```

## 📂 Output Structure

Results are saved by default in `results/<hostname>/<timestamp>/`:
- `summary.txt`: Human-readable health certificate.
- `metrics.json`: High-level aggregate scores (suitable for Ansible/ARA).
- `telemetry.h5`: Raw time-series telemetry data.
- `dashboard.png`: Visual overview of the run.

## ⚖️ License
MIT / Apache 2.0