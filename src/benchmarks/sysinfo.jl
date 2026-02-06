function run_sysinfo(args)
    @info "Collecting System Information..."
    
    results = Dict{String, Any}()
    
    if CUDA.functional()
        dev = CUDA.device()
        results["gpu_name"] = CUDA.name(dev)
        results["compute_capability"] = string(CUDA.capability(dev))
        # CUDA.jl v5+ uses totalmem(dev) and available_memory()
        results["vram_total"] = Base.format_bytes(Int(CUDA.totalmem(dev)))
        results["vram_free"] = Base.format_bytes(Int(CUDA.available_memory()))
        results["pci_bus_id"] = string(CUDA.uuid(dev))
    end
    
    return results
end

register_benchmark(
    "sysinfo",
    "Display system and GPU hardware information without running benchmarks",
    run_sysinfo
)