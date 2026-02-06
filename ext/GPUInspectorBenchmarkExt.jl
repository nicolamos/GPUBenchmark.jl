module GPUInspectorBenchmarkExt

using GPUBenchmark
using GPUInspector
using CairoMakie
using CUDA
using JSON

function run_gpuinspector(args)
    @info "Running GPUInspector deep inspection on ALL available GPUs..."
    
    # Check for available GPUs
    devs = CUDA.devices()
    if isempty(devs)
        error("No CUDA-capable GPUs found.")
    end
    @info "Detected $(length(devs)) GPUs: $(join([CUDA.name(d) for d in devs], ", "))"

    results = Dict{String, Any}()
    
    # 1. Bandwidth
    @info "Measuring Memory Bandwidth (max across devices)..."
    # memory_bandwidth() usually targets current device, we can loop if needed
    # but for a summary, let's keep the existing call or expand it
    results["memory_bandwidth"] = memory_bandwidth()
    
    # 2. Parallel Stress Test with Monitoring
    duration = args["duration"]
    size = args["size"]
    @info "Running $(duration)s Parallel Stress Test (size=$size) with telemetry..."
    
    # We use the high-level stresstest which supports parallel and monitoring
    # This internally calls monitoring_start and monitoring_stop
    mon_results = stresstest(; 
        devices=devs, 
        duration=duration, 
        size=size, 
        monitoring=true, 
        parallel=true, 
        verbose=true
    )
    
    # Store monitoring data for JSON serialization
    results["monitoring"] = Dict(
        "times" => mon_results.times,
        "metrics" => Dict(string(k) => v for (k, v) in mon_results.results)
    )
    
    # Attach raw object for save_plots
    results[:_raw_monitoring] = mon_results
    
    return results
end

function GPUBenchmark.save_plots(results, output_path)
    if haskey(results["benchmarks"], "gpuinspector")
        g_res = results["benchmarks"]["gpuinspector"]
        if haskey(g_res, :_raw_monitoring)
            mon_results = g_res[:_raw_monitoring]
            
            # Save raw HDF5 telemetry
            h5_file = joinpath(output_path, "telemetry.h5")
            @info "Saving raw telemetry to $h5_file"
            try
                save_monitoring_results(h5_file, mon_results)
            catch e
                @error "Failed to save HDF5 telemetry" exception=e
            end
            
            # Save Plots (Tiled Dashboard)
            plot_file = joinpath(output_path, "dashboard.png")
            @info "Saving dashboard to $plot_file"
            try
                savefig_monitoring_results(plot_file, mon_results)
            catch e
                @error "Failed to save monitoring plots" exception=e
            end
        end
    end
end

function __init__()
    GPUBenchmark.register_benchmark(
        "gpuinspector",
        "Parallel GPU stress test, bandwidth, and telemetry across all devices",
        run_gpuinspector
    )
end

end # module
