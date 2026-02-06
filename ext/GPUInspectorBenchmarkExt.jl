module GPUInspectorBenchmarkExt

using GPUBenchmark
using GPUInspector
using CairoMakie
using CUDA
using JSON

function run_gpuinspector(args)
    @info "Running GPUInspector deep inspection..."
    
    # We use stresstest with monitoring to get telemetry
    # and also run bandwidth tests
    
    results = Dict{String, Any}()
    
    # 1. Bandwidth
    @info "Measuring Memory Bandwidth..."
    results["memory_bandwidth"] = memory_bandwidth()
    
    # 2. Stress Test with Monitoring
    @info "Running $(args["duration"])s Stress Test with monitoring (size=$(args["size"]))..."
    # We capture the MonitoringResults object
    mon_results = stresstest(duration=args["duration"], monitoring=true, verbose=false, size=args["size"])
    
    # Store monitoring results in a way that can be serialized to JSON
    # (Extracting raw data from MonitoringResults)
    results["monitoring"] = Dict(
        "times" => mon_results.times,
        "metrics" => Dict(string(k) => v for (k, v) in mon_results.results)
    )
    
    # Attach the raw MonitoringResults object to the results dict 
    # using a special key that we'll use in save_plots
    # Note: This is stored in the object but won't be serialized by JSON.print
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
            
            # Save Plots
            plot_file = joinpath(output_path, "dashboard.png")
            @info "Saving dashboard to $plot_file"
            try
                # Use the new tiled summary dashboard
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
        "Deep GPU inspection, bandwidth tests, and telemetry using GPUInspector.jl",
        run_gpuinspector
    )
end

end # module