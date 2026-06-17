# Advanced Benchmarking

This guide covers complex benchmarking scenarios including multi-GPU scaling and custom plugins.

## Scaling Analysis

The `scaling` task allows you to measure how performance changes as problem size increases or as you add more GPUs.

### Multi-GPU Parallel Probing
By using the `--parallel` flag, you can exercise all GPUs simultaneously. This is useful for identifying PCIe bandwidth bottlenecks or power delivery issues on a node.

```bash
julia --project -m GPUBenchmark scaling --parallel --devices all
```

## Plugin Development

GPUBenchmark.jl uses a modular architecture. You can add new algorithms by implementing the `AbstractAlgorithm` interface.

### Example Plugin
See the `README.md` for a complete example of a plugin module.
