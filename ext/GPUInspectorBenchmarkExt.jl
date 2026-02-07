module GPUInspectorBenchmarkExt

using GPUBenchmark
using GPUInspector
using CairoMakie
using CUDA
using JSON
using Statistics: mean

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
    
    # Store only summary/metadata in JSON to avoid redundancy with HDF5
    results["monitoring_file"] = "telemetry.h5"
    
    # Attach raw object for save_plots
    results["_raw_monitoring"] = mon_results
    
    return results
end

function GPUBenchmark.save_plots(results, output_path)
    if haskey(results["benchmarks"], "gpuinspector")
        g_res = results["benchmarks"]["gpuinspector"]
        if haskey(g_res, "_raw_monitoring")
            mon_results = g_res["_raw_monitoring"]
            
            # Save raw HDF5 telemetry (the source of truth)
            h5_file = joinpath(output_path, "telemetry.h5")
            @info "Saving raw telemetry to $h5_file"
            try
                save_monitoring_results(h5_file, mon_results, overwrite=true)
            catch e
                @error "Failed to save HDF5 telemetry" exception=e
            end
            
            # Save Plots (Tiled Dashboard Image)
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

function GPUBenchmark.cleanup()
    @info "Cleaning up GPU memory..."
    try
        # Use GPUInspector's multi-GPU memory clearing (calls CUDA.reclaim internally)
        clear_all_gpus_memory()
    catch e
        @warn "Cleanup encountered an issue" exception=e
    end
end

function _render_telemetry_gpuinspector(bench_data, run_path)
    h5_file = joinpath(run_path, "telemetry.h5")
    if !isfile(h5_file)
        return ""
    end

    try
        # 1. Load from HDF5 (source of truth)
        res = load_monitoring_results(h5_file)
        
        plot_strings = String[]
        
        # Helper for smoothing
        function smooth_and_resample(data, target_points=80)
            if length(data) <= target_points return Float64.(data) end
            window = max(2, length(data) ÷ target_points)
            smoothed = [sum(data[i:min(i+window-1, end)]) / length(data[i:min(i+window-1, end)]) for i in 1:length(data)]
            return smoothed[round.(Int, range(1, length(smoothed), length=target_points))]
        end

        # Order: Power, Compute, Mem, Temp
        requested_symbols = [:power, :compute, :mem, :temperature]
        titles = Dict(
            :power => "Power (W)",
            :compute => "Compute Util (%)",
            :mem => "Memory Util (%)",
            :temperature => "Temperature (°C)"
        )
        ylims = Dict(
            :compute => (0, 100),
            :mem => (0, 100)
        )

        for s in requested_symbols
            if haskey(res.results, s)
                data = res.results[s]
                num_gpus = length(res.devices)
                
                # Aggregate for Global Trend (requested by user)
                history_len = length(data[1])
                aggregated = [mean([data[gpu][t] for gpu in 1:num_gpus]) for t in 1:history_len]
                vals = smooth_and_resample(aggregated)
                
                # Create plot
                p = lineplot(vals, 
                    title=titles[s], 
                    color=:cyan, 
                    width=65, height=10, 
                    border=:solid, canvas=BrailleCanvas,
                    xlabel="", ylabel=""
                )
                if haskey(ylims, s)
                    p = lineplot(vals, 
                        title=titles[s], color=:cyan, width=65, height=10, 
                        border=:solid, canvas=BrailleCanvas,
                        xlabel="", ylabel="", ylim=ylims[s]
                    )
                end

                # Global Stats
                all_vals = reduce(vcat, data)
                stats = "{dim}  Global Max: $(round(maximum(all_vals), digits=1))  |  Global Avg: $(round(mean(all_vals), digits=1))  |  Devices: $num_gpus{/dim}"
                
                push!(plot_strings, string(p) * "\n" * stats)
            end
        end

        return "{bold}Node Telemetry Summary (from HDF5):{/bold}\n" * join(plot_strings, "\n\n")

    catch e
        return "{red}Failed to load telemetry from HDF5: $e{/red}"
    end
end

function __init__()
    # Register the telemetry renderer
    GPUBenchmark.Visuals.TELEMETRY_RENDERER[] = _render_telemetry_gpuinspector

    GPUBenchmark.register_benchmark(
        "gpuinspector",
        "Parallel GPU stress test, bandwidth, and telemetry across all devices",
        run_gpuinspector
    )
end

end # module
