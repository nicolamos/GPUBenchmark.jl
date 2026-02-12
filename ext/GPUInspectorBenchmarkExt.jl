module GPUInspectorBenchmarkExt

using GPUBenchmark
using GPUInspector
using CairoMakie
using CUDA
using JSON
using Statistics: mean
using Printf

function run_gpuinspector(args)
    @info "Running GPUInspector deep inspection on ALL available GPUs..."

    devs = CUDA.devices()
    if isempty(devs)
        error("No CUDA-capable GPUs found.")
    end
    @info "Detected $(length(devs)) GPUs: $(join([CUDA.name(d) for d in devs], ", "))"

    results = Dict{String, Any}()

    @info "Measuring Memory Bandwidth (max across devices)..."
    results["memory_bandwidth"] = memory_bandwidth()

    duration = args["duration"]
    size = args["size"]
    @info "Running $(duration)s Parallel Stress Test (size=$size) with telemetry..."

    mon_results = stresstest(;
        devices=devs,
        duration=duration,
        size=size,
        monitoring=true,
        parallel=true,
        verbose=true
    )

    results["monitoring_file"] = "telemetry.h5"
    results["_raw_monitoring"] = mon_results
    
    # Calculate average power across all GPUs for efficiency reporting
    if haskey(mon_results.results, :power)
        all_pwr = reduce(vcat, mon_results.results[:power])
        results["avg_power_w"] = isempty(all_pwr) ? 0.0 : mean(all_pwr)
    end

    return results
end

function GPUBenchmark.save_plots(results, output_path, format="png")
    if haskey(results["benchmarks"], "gpuinspector")
        g_res = results["benchmarks"]["gpuinspector"]
        if haskey(g_res, "_raw_monitoring")
            mon_results = g_res["_raw_monitoring"]

            h5_file = joinpath(output_path, "telemetry.h5")
            @info "Saving raw telemetry to $h5_file"
            try
                save_monitoring_results(h5_file, mon_results, overwrite=true)
            catch e
                @error "Failed to save HDF5 telemetry" exception=e
            end

            plot_file = joinpath(output_path, "telemetry_dashboard.$format")
            @info "Saving telemetry dashboard to $plot_file"
            try
                savefig_monitoring_results(plot_file, mon_results)
            catch e
                @error "Failed to save monitoring plots" exception=e
            end
        end
    end

    # Performance plots
    dat_path = joinpath(output_path, "scaling.dat")
    if isfile(dat_path)
        try
            dat = GPUBenchmark.Visuals.parse_scaling_dat(dat_path)
            
            peak_gpu = 10.0; peak_cpu = 1.0; bw_gpu = 500.0; bw_cpu = 50.0
            if haskey(results["benchmarks"], "scaling")
                peak_gpu = get(results["benchmarks"]["scaling"], "peak_gpu_tflops", 10.0)
                peak_cpu = get(results["benchmarks"]["scaling"], "peak_cpu_tflops", 1.0)
            end
            if haskey(results["benchmarks"], "gpuinspector")
                bw_gpu = get(results["benchmarks"]["gpuinspector"], "memory_bandwidth", 500.0)
            end
            
            _save_scaling_plots(dat, output_path, format)
            _save_roofline_plots(dat, output_path, format, peak_gpu, peak_cpu, bw_gpu, bw_cpu)
        catch e
            @error "Failed to save performance plots" exception=e
        end
    end
end

function _save_scaling_plots(dat, output_path, format)
    if isempty(dat.cpu_ns) && isempty(dat.gpu_data)
        return
    end

    fig = Figure(size=(800, 800))
    
    # GPU Axis
    ax_gpu = Axis(fig[1, 1],
        title  = "GPU Performance Scaling: $(dat.alg)",
        xlabel = "Matrix Size N", ylabel = "TFLOPS",
        xgridvisible = true, ygridvisible = true)
    
    colors = Makie.wong_colors()
    i = 1
    for (dev_id, (ns, ts)) in dat.gpu_data
        isempty(ns) && continue
        ord = sortperm(ns)
        scatterlines!(ax_gpu, ns[ord], ts[ord]; label="GPU $dev_id", color=colors[mod1(i, length(colors))], marker=:circle)
        i += 1
    end
    axislegend(ax_gpu, position=:lt)

    # CPU Axis (Independent Y-scale)
    ax_cpu = Axis(fig[2, 1],
        title  = "CPU Performance Scaling: $(dat.alg)",
        xlabel = "Matrix Size N", ylabel = "TFLOPS",
        xgridvisible = true, ygridvisible = true)
    
    if !isempty(dat.cpu_ns)
        ord = sortperm(dat.cpu_ns)
        scatterlines!(ax_cpu, dat.cpu_ns[ord], dat.cpu_tflops[ord]; label="CPU", color=:gray, marker=:rect)
    end
    axislegend(ax_cpu, position=:lt)

    save(joinpath(output_path, "performance_scaling.$format"), fig)
end

function _save_roofline_plots(dat, output_path, format, peak_gpu, peak_cpu, bw_gpu, bw_cpu)
    fig = Figure(size=(800, 800))
    
    # GPU Roofline
    ax_gpu = Axis(fig[1, 1], title="GPU Roofline Model", xlabel="Intensity (FLOP/Byte)", ylabel="TFLOPS", xscale=log10, yscale=log10, xgridvisible=true, ygridvisible=true)
    ns_g = isempty(dat.gpu_data) ? Int[] : first(values(dat.gpu_data))[1]
    ts_g = isempty(dat.gpu_data) ? Float64[] : first(values(dat.gpu_data))[2]
    if !isempty(ns_g)
        is = ns_g ./ 6.0
        max_i = maximum(is) * 1.5
        roof_is = range(0.1, max_i, length=200)
        bw_T_s = (bw_gpu * 1024^3) / 1e12
        roof_ts = [min(peak_gpu, i * bw_T_s) for i in roof_is]
        ridge_i = peak_gpu / bw_T_s
        
        lines!(ax_gpu, roof_is, roof_ts, color=:black, linewidth=2, label="Theoretical Roof")
        vlines!(ax_gpu, [ridge_i], color=:red, linestyle=:dash, label="Ridge Point ($(@sprintf("%.2f", ridge_i)))")
        scatter!(ax_gpu, is, ts_g, color=:cyan, markersize=12, label="Measured Scaling")
    end
    axislegend(ax_gpu, position=:rb)

    # CPU Roofline
    ax_cpu = Axis(fig[2, 1], title="CPU Roofline Model", xlabel="Intensity (FLOP/Byte)", ylabel="TFLOPS", xscale=log10, yscale=log10, xgridvisible=true, ygridvisible=true)
    if !isempty(dat.cpu_ns)
        is_c = dat.cpu_ns ./ 6.0
        max_ic = maximum(is_c) * 1.5
        roof_isc = range(0.1, max_ic, length=200)
        bw_T_sc = (bw_cpu * 1024^3) / 1e12
        roof_tsc = [min(peak_cpu, i * bw_T_sc) for i in roof_isc]
        ridge_ic = peak_cpu / bw_T_sc
        
        lines!(ax_cpu, roof_isc, roof_tsc, color=:black, linewidth=2, label="Theoretical Roof")
        vlines!(ax_cpu, [ridge_ic], color=:red, linestyle=:dash, label="Ridge Point ($(@sprintf("%.2f", ridge_ic)))")
        scatter!(ax_cpu, is_c, dat.cpu_tflops, color=:blue, markersize=12, label="Measured Scaling")
    end
    axislegend(ax_cpu, position=:rb)

    save(joinpath(output_path, "performance_roofline.$format"), fig)
end

function GPUBenchmark.cleanup()
    @info "Cleaning up GPU memory..."
    try
        clear_all_gpus_memory()
    catch e
        @warn "Cleanup encountered an issue" exception=e
    end
end

function _render_telemetry_gpuinspector(bench_data, run_path)
    h5_file = joinpath(run_path, "telemetry.h5")
    !isfile(h5_file) && return

    try
        res = load_monitoring_results(h5_file)

        function smooth_and_resample(data, target_points=80)
            length(data) <= target_points && return Float64.(data)
            window = max(2, length(data) ÷ target_points)
            smoothed = [sum(data[i:min(i+window-1, end)]) / length(data[i:min(i+window-1, end)]) for i in 1:length(data)]
            return smoothed[round.(Int, range(1, length(smoothed), length=target_points))]
        end

        requested_symbols = [:power, :compute, :mem, :temperature]
        titles = Dict(
            :power       => "Power (W)",
            :compute     => "Compute Util (%)",
            :mem         => "Memory Util (%)",
            :temperature => "Temperature (°C)"
        )
        ylims = Dict(:compute => (0, 100), :mem => (0, 100))

        println("\n  Node Telemetry Summary (from HDF5):")
        for s in requested_symbols
            !haskey(res.results, s) && continue
            data = res.results[s]
            num_gpus = length(res.devices)

            history_len = length(data[1])
            aggregated = [mean([data[gpu][t] for gpu in 1:num_gpus]) for t in 1:history_len]
            vals = smooth_and_resample(aggregated)

            println("\n") # Space before plot
            GPUBenchmark.Visuals.render_lineplot(1:length(vals), vals;
                title=titles[s], color=:cyan, width=65, height=10,
                ylim=get(ylims, s, nothing))

            all_vals = reduce(vcat, data)
            @printf("  Global Max: %.1f  |  Global Avg: %.1f  |  Devices: %d\n",
                maximum(all_vals), mean(all_vals), num_gpus)
        end

    catch e
        println("  Failed to load telemetry from HDF5: $e")
    end
end

function __init__()
    GPUBenchmark.Visuals.TELEMETRY_RENDERER[] = _render_telemetry_gpuinspector

    GPUBenchmark.register_benchmark(
        "gpuinspector",
        "Parallel GPU stress test, bandwidth, and telemetry across all devices",
        run_gpuinspector
    )
end

end # module
