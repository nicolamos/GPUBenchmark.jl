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

    # Roofline plot (requires memory_bandwidth + at least matmul or scaling)
    try
        _save_roofline_plot(results["benchmarks"], output_path, format)
    catch e
        @error "Failed to save roofline plot" exception=e
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

function _save_roofline_plot(bench_data, output_path, format)
    gi = get(bench_data, "gpuinspector", nothing)
    bw_GiB_s = isnothing(gi) || haskey(gi, "error") ? nothing : get(gi, "memory_bandwidth", nothing)
    isnothing(bw_GiB_s) && return

    # Peak TFLOPS: prefer scaling result, fall back to matmul
    peak_tflops = nothing
    if haskey(bench_data, "scaling") && !haskey(bench_data["scaling"], "error")
        peak_tflops = get(bench_data["scaling"], "peak_gpu_tflops", nothing)
    end
    if isnothing(peak_tflops) && haskey(bench_data, "matmul") && !haskey(bench_data["matmul"], "error")
        peak_tflops = get(bench_data["matmul"], "tflops", nothing)
    end
    isnothing(peak_tflops) && return

    # Bandwidth in T-bytes/s so the slope y = bw_T_bytes_s * x gives TFLOPS
    bw_bytes_s   = bw_GiB_s * (1024^3)
    bw_T_bytes_s = bw_bytes_s / 1e12
    ridge_AI     = peak_tflops / bw_T_bytes_s   # FLOP/byte where lines cross

    # X range: one decade below lowest measured AI to one decade above ridge
    x_lo = 0.1
    x_hi = ridge_AI * 10
    xs   = exp10.(range(log10(x_lo), log10(x_hi), length=300))
    bw_line = min.(bw_T_bytes_s .* xs, peak_tflops)

    fig = Figure(size=(700, 500))
    ax  = Axis(fig[1, 1];
        title        = "Roofline Model",
        xlabel       = "Arithmetic Intensity (FLOP/byte)",
        ylabel       = "Attainable TFLOPS",
        xscale       = log10,
        yscale       = log10,
        xgridvisible = true,
        ygridvisible = true)

    lines!(ax, xs, bw_line;
        color=:steelblue, linewidth=2,
        label="Memory BW ($(round(bw_GiB_s, digits=1)) GiB/s)")
    hlines!(ax, [peak_tflops];
        color=:darkorange, linewidth=2, linestyle=:dash,
        label="Peak compute ($(round(peak_tflops, digits=1)) TFLOPS)")
    vlines!(ax, [ridge_AI]; color=:gray, linewidth=1, linestyle=:dot)

    # Measured points
    pt_colors = [:seagreen, :crimson, :purple]
    pt_idx = 1
    if haskey(bench_data, "matmul") && !haskey(bench_data["matmul"], "error")
        m = bench_data["matmul"]
        n, t = get(m, "matrix_size", nothing), get(m, "tflops", nothing)
        if !isnothing(n) && !isnothing(t)
            ai = n / 6.0   # FP32: 2N³ / (3 * N² * 4 bytes)
            scatter!(ax, [ai], [t];
                color=pt_colors[pt_idx], marker=:circle, markersize=14,
                label="MatMul FP32 (N=$n, AI=$(round(ai, digits=1)))")
            pt_idx += 1
        end
    end
    if haskey(bench_data, "tensorcore") && !haskey(bench_data["tensorcore"], "error")
        tc = bench_data["tensorcore"]
        n, t = get(tc, "matrix_size", nothing), get(tc, "tflops", nothing)
        if !isnothing(n) && !isnothing(t)
            ai = n / 3.0   # FP16: 2N³ / (3 * N² * 2 bytes)
            scatter!(ax, [ai], [t];
                color=pt_colors[pt_idx], marker=:diamond, markersize=14,
                label="TensorCore FP16 (N=$n, AI=$(round(ai, digits=1)))")
        end
    end

    axislegend(ax; position=:lt)
    plot_path = joinpath(output_path, "roofline.$format")
    save(plot_path, fig)
    @info "Roofline plot saved to $plot_path"
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

        # TFLOPS/Watt efficiency (requires both power telemetry and scaling peak)
        if haskey(res.results, :power) &&
                haskey(bench_data, "scaling") && !haskey(bench_data["scaling"], "error")
            peak_tflops = get(bench_data["scaling"], "peak_gpu_tflops", nothing)
            if !isnothing(peak_tflops) && peak_tflops > 0
                avg_power = mean(reduce(vcat, res.results[:power]))
                @printf("\n  Efficiency:  %.3f TFLOPS/W  (%.1f TFLOPS peak / %.1f W avg)\n",
                    peak_tflops / avg_power, peak_tflops, avg_power)
            end
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
