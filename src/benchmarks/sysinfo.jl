function run_sysinfo(args)
    @info "Collecting System Information..."
    
    # This task mainly returns info that is already collected in the main loop
    # but we can add more specific details here if needed.
    
    results = Dict{String, Any}()
    
    if CUDA.functional()
        dev = CUDA.device()
        results["gpu_name"] = CUDA.name(dev)
        results["compute_capability"] = string(CUDA.capability(dev))
        results["vram_total"] = Base.format_bytes(CUDA.total_memory(dev))
        results["vram_free"] = Base.format_bytes(CUDA.available_memory(dev))
        results["pci_bus_id"] = CUDA.uuid(dev) # or actual PCI ID if available
    end
    
    return results
end

register_benchmark(
    "sysinfo",
    "Display system and GPU hardware information without running benchmarks",
    run_sysinfo
)
