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

            plot_file = joinpath(output_path, "dashboard.$format")
            @info "Saving dashboard to $plot_file"
            try
                savefig_monitoring_results(plot_file, mon_results)
            catch e
                @error "Failed to save monitoring plots" exception=e
            end
        end
    end

    # Scaling plot (read from scaling.dat, same source as terminal plot)
    dat_path = joinpath(output_path, "scaling.dat")
    if isfile(dat_path)
        try
            dat = GPUBenchmark.Visuals.parse_scaling_dat(dat_path)
            _save_scaling_plot(dat, output_path, format)
        catch e
            @error "Failed to save scaling plot" exception=e
        end
    end
end

function _save_scaling_plot(dat, output_path, format)
    if isempty(dat.cpu_ns) && isempty(dat.gpu_data)
        return
    end

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
        title  = "Scaling Performance: $(dat.alg)",
        xlabel = "Matrix Size N",
        ylabel = "TFLOPS",
        xgridvisible = true,
        ygridvisible = true)

    # Use a nice color palette
    colors = Makie.wong_colors()
    
    i = 1
    for (dev_id, (ns, ts)) in dat.gpu_data
        isempty(ns) && continue
        ord = sortperm(ns)
        scatterlines!(ax, ns[ord], ts[ord]; 
            label="GPU $dev_id", 
            color=colors[mod1(i, length(colors))],
            marker=:circle, markersize=8)
        i += 1
    end

    # Mean trend line across all GPU devices (when >1)
    if length(dat.gpu_data) > 1
        n_vals = Dict{Int, Vector{Float64}}()
        for (_, (ns, ts)) in dat.gpu_data
            for (n, t) in zip(ns, ts)
                push!(get!(n_vals, n, Float64[]), t)
            end
        end
        mean_ns = sort(collect(keys(n_vals)))
        mean_ts = [mean(n_vals[n]) for n in mean_ns]
        lines!(ax, mean_ns, mean_ts;
            label="mean", color=:black, linestyle=:dot, linewidth=2)
    end

    if !isempty(dat.cpu_ns)
        ord = sortperm(dat.cpu_ns)
        scatterlines!(ax, dat.cpu_ns[ord], dat.cpu_tflops[ord];
            label="CPU", linestyle=:dash, color=:gray,
            marker=:rect, markersize=8)
    end

    axislegend(ax, position=:lt)
    plot_path = joinpath(output_path, "scaling_plot.$format")
    save(plot_path, fig)
    @info "Scaling plot saved to $plot_path"
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
