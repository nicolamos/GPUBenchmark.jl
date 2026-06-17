# Visual Reporting for GPUBenchmark.jl
module Visuals

using PrettyTables
using UnicodePlots
using JSON
using Printf
using Statistics: mean

export show_results, render_lineplot, parse_scaling_dat

# Dynamic hook for extensions to provide telemetry rendering
const TELEMETRY_RENDERER = Ref{Function}((bench_data, run_path) -> begin
    if haskey(bench_data, "gpuinspector") && haskey(bench_data["gpuinspector"], "monitoring_file")
        println("  Telemetry data available but GPUInspector extension not loaded.")
    end
end)

function render_telemetry(bench_data, run_path)
    TELEMETRY_RENDERER[](bench_data, run_path)
end

"""
    render_lineplot(xs, ys; title, xlabel, ylabel, color, width, height)

Render a UnicodePlots line plot to stdout using explicit MIME text/plain.
Ensures deterministic plain-text output on headless HPC nodes.
"""
function render_lineplot(xs, ys; title="", xlabel="", ylabel="", color=:cyan, width=60, height=10, ylim=nothing)
    kw = (title=title, xlabel=xlabel, ylabel=ylabel, color=color, width=width, height=height)
    p = isnothing(ylim) ? lineplot(xs, ys; kw...) : lineplot(xs, ys; kw..., ylim=ylim)
    show(stdout, MIME"text/plain"(), p)
    println()
end

function parse_scaling_dat(dat_path)
    cpu_ns = Int[]; cpu_tflops = Float64[]; cpu_min = Float64[]; cpu_max = Float64[]
    gpu_ns   = Dict{String, Vector{Int}}()
    gpu_tflops = Dict{String, Vector{Float64}}()
    gpu_min    = Dict{String, Vector{Float64}}()
    gpu_max    = Dict{String, Vector{Float64}}()
    alg = ""

    open(dat_path) do io
        for line in eachline(io)
            if startswith(line, "#")
                contains(line, "Algorithm:") && (alg = strip(split(line, "Algorithm:")[end]))
                continue
            end
            parts = split(line, '\t')
            length(parts) < 5 && continue
            n = parse(Int, parts[1])
            mode = parts[2]
            id = parts[3]
            tflops = parse(Float64, parts[5])
            tmin = length(parts) >= 6 ? parse(Float64, parts[6]) : 0.0
            tmax = length(parts) >= 7 ? parse(Float64, parts[7]) : 0.0
            
            if mode == "CPU"
                push!(cpu_ns, n); push!(cpu_tflops, tflops)
                push!(cpu_min, tmin); push!(cpu_max, tmax)
            else
                push!(get!(gpu_ns,     id, Int[]),     n)
                push!(get!(gpu_tflops, id, Float64[]), tflops)
                push!(get!(gpu_min,    id, Float64[]), tmin)
                push!(get!(gpu_max,    id, Float64[]), tmax)
            end
        end
    end
    gpu_data = Dict(id => (gpu_ns[id], gpu_tflops[id], gpu_min[id], gpu_max[id]) for id in keys(gpu_ns))
    return (alg=alg, cpu_ns=cpu_ns, cpu_tflops=cpu_tflops, cpu_min=cpu_min, cpu_max=cpu_max, gpu_data=gpu_data)
end

const _GPU_COLORS = [:green, :red, :cyan, :magenta, :yellow]

function _aggregate_per_n(gpu_data)
    n_vals = Dict{Int, Vector{Float64}}()
    for (_, (ns, ts)) in gpu_data
        for (n, t) in zip(ns, ts)
            push!(get!(n_vals, n, Float64[]), t)
        end
    end
    return n_vals
end

function _render_scaling_plots(dat)
    if isempty(dat.cpu_ns) && isempty(dat.gpu_data)
        return
    end

    num_devices = length(dat.gpu_data)
    println("\n  Performance Scaling (TFLOPS vs N):")

    # 1. GPU Scaling (Separate)
    if !isempty(dat.gpu_data)
        n_vals = _aggregate_per_n(dat.gpu_data)
        all_ns  = sort(collect(keys(n_vals)))
        mean_ts = [mean(n_vals[n]) for n in all_ns]

        if num_devices <= 4
            device_ids = sort(collect(keys(dat.gpu_data)))
            id1 = device_ids[1]
            ns1, ts1 = dat.gpu_data[id1]
            ord = sortperm(ns1)
            p = lineplot(ns1[ord], ts1[ord];
                title="GPU Scaling",
                xlabel="N", ylabel="TFLOPS",
                color=_GPU_COLORS[1], width=70, height=12,
                name="GPU $id1")
            for (i, id) in enumerate(device_ids[2:end])
                ns_i, ts_i = dat.gpu_data[id]
                ord_i = sortperm(ns_i)
                lineplot!(p, ns_i[ord_i], ts_i[ord_i];
                    color=_GPU_COLORS[mod1(i + 1, length(_GPU_COLORS))],
                    name="GPU $id")
            end
            num_devices > 1 && lineplot!(p, all_ns, mean_ts; color=:white, name="mean")
        else
            max_ts = [maximum(n_vals[n]) for n in all_ns]
            min_ts = [minimum(n_vals[n]) for n in all_ns]
            p = lineplot(all_ns, max_ts;
                title="GPU Scaling — $(num_devices) GPUs",
                xlabel="N", ylabel="TFLOPS",
                color=:cyan, width=70, height=12, name="max")
            lineplot!(p, all_ns, mean_ts; color=:white, name="mean")
            lineplot!(p, all_ns, min_ts;  color=:red,   name="min")
        end
        show(stdout, MIME"text/plain"(), p)
        println()
    end

    # 2. CPU Scaling (Separate)
    if !isempty(dat.cpu_ns)
        ord = sortperm(dat.cpu_ns)
        p_cpu = lineplot(dat.cpu_ns[ord], dat.cpu_tflops[ord];
            title="CPU Scaling", xlabel="N", ylabel="TFLOPS", 
            color=:blue, width=70, height=12)
        show(stdout, MIME"text/plain"(), p_cpu)
        println()
    end
end

# Plot a roofline model for the given (ns, ts) measurements.
function _render_roofline(ns, ts; title="Roofline", peak_tflops=10.0, peak_bw_gb=500.0, bw_source="default")
    # Arithmetic intensity for FP32 matmul NxN: 2N³ FLOPs / 12N² bytes = N/6 FLOP/byte
    isempty(ns) && return

    is = ns ./ 6.0
    valid = is .> 0
    is = is[valid]; ts_v = ts[valid]
    isempty(is) && return

    bw_t_s  = (peak_bw_gb * 1024^3) / 1e12   # bandwidth ceiling slope (TFLOPS per FLOP/Byte)
    ridge_i = peak_tflops / bw_t_s             # ridge point

    # Snap axis bounds to exact powers of 10 so UnicodePlots generates clean tick labels
    x_low  = exp10(floor(log10(max(0.01, ridge_i / 100))))
    x_high = exp10(ceil(log10(maximum(is))))
    y_low  = exp10(floor(log10(max(1e-3, x_low * bw_t_s))))
    y_high = exp10(ceil(log10(peak_tflops * 1.5)))

    n_pts   = 60
    roof_is = exp10.(range(log10(x_low), log10(x_high), length=n_pts))
    roof_ts = [min(peak_tflops, i * bw_t_s) for i in roof_is]

    ridge_xs = [ridge_i, ridge_i]
    ridge_ys = [y_low, y_high]

    bw_note = bw_source == "default" ?
        " [⚠ stima — installa GPUInspector.jl]" :
        " [via $bw_source]"

    println("\n  Roofline Analysis ($title) — log/log axes:")
    p = lineplot(roof_is, roof_ts;
        title="$title Roofline", xlabel="Intensity (FLOP/Byte)", ylabel="TFLOPS",
        color=:white, width=70, height=12,
        xscale=:log10, yscale=:log10,
        xlim=(x_low, x_high), ylim=(y_low, y_high),
        name="Theoretical")
    lineplot!(p, ridge_xs, ridge_ys;
        color=:red, name="Ridge ($(@sprintf("%.1f", ridge_i)) F/B)")
    scatterplot!(p, is, ts_v; color=:cyan, marker=:circle, name="Measured")

    show(stdout, MIME"text/plain"(), p)
    println()
    println("  Ridge: below $(@sprintf("%.1f", ridge_i)) FLOP/Byte → memory-bound; above → compute-bound")
    if bw_source == "default"
        println("  ⚠  Banda GPU non misurata: usando $(@sprintf("%.0f", peak_bw_gb)) GB/s come fallback — ridge point inaccurato.")
    end
end

function _render_bandwidth_util_plot(ns, ts; title="BW Utilization", peak_bw_gb=500.0)
    # BW achieved per matmul: BW = (6 * TFLOPS * 10^12) / N  (Bytes/s)
    isempty(ns) && return

    bw_achieved = (6.0 .* ts .* 1e12) ./ ns
    bw_gib = bw_achieved ./ (1024.0^3)
    util = (bw_gib ./ peak_bw_gb) .* 100.0

    println("\n  Memory Bandwidth Utilization ($title):")
    p = lineplot(ns, util,
        title="BW Utilization (%) vs N", xlabel="N", ylabel="Utilization (%)",
        color=:magenta, width=70, height=12)
    show(stdout, MIME"text/plain"(), p)
    println()
end

function _render_speedup_plot(dat)
    isempty(dat.cpu_ns) && return
    isempty(dat.gpu_data) && return

    cpu_by_n = Dict(zip(dat.cpu_ns, dat.cpu_tflops))

    # Max GPU TFLOPS at each N across all devices
    n_vals   = _aggregate_per_n(dat.gpu_data)
    gpu_by_n = Dict(n => maximum(ts) for (n, ts) in n_vals)

    shared_ns = sort(collect(intersect(keys(cpu_by_n), keys(gpu_by_n))))
    length(shared_ns) < 2 && return

    speedups = [gpu_by_n[n] / cpu_by_n[n] for n in shared_ns]

    println("\n  GPU vs CPU Speedup (comparison range):")
    p = lineplot(shared_ns, speedups;
        title="GPU/CPU Speedup vs N", xlabel="N", ylabel="Speedup (×)",
        color=:green, width=70, height=10,
        name="Speedup")
    lineplot!(p, [minimum(shared_ns), maximum(shared_ns)], [1.0, 1.0];
        color=:red, name="1× (parity)")
    show(stdout, MIME"text/plain"(), p)
    println()

    peak_sp  = maximum(speedups)
    peak_idx = argmax(speedups)
    xover    = findfirst(sp -> sp > 1.0, speedups)
    @printf("  Peak speedup: %.1f× at N=%d\n", peak_sp, shared_ns[peak_idx])
    !isnothing(xover) && @printf("  GPU > CPU from N=%d onwards\n", shared_ns[xover])
end

"""
    to_vector(x) -> Vector{Float64}

Normalize a JSON-parsed numeric value to a flat `Vector{Float64}`.
Handles scalars, 0-dimensional arrays, and any `AbstractArray`.
"""
to_vector(x::Number)::Vector{Float64}         = [Float64(x)]
to_vector(x::AbstractArray)::Vector{Float64}  = vec(Float64.(x))

function _render_latency_histogram(bench_data)
    if !haskey(bench_data, "matmul") || haskey(bench_data["matmul"], "error")
        return
    end

    m = bench_data["matmul"]
    if !haskey(m, "samples")
        return
    end

    samples_ms = to_vector(m["samples"]) .* 1000.0
    isempty(samples_ms) && return
    println("\n  Latency Distribution (MatMul FP32):")
    if length(samples_ms) < 2
        # UnicodePlots.histogram errors on a single-point sample (degenerate bin edges)
        @warn "Single sample, skipping histogram" latency_ms=samples_ms[1]
        return
    end
    p = histogram(samples_ms, nbins=15, title="Kernel Latency", xlabel="Time (ms)", color=:yellow, width=70, height=10)
    show(stdout, MIME"text/plain"(), p)
    println()
end

function show_results(path::String)
    metrics_path = joinpath(path, "metrics.json")
    if !isfile(metrics_path)
        println("Error: metrics.json not found in $path")
        return
    end

    results = JSON.parsefile(metrics_path)
    meta = results["metadata"]
    bench_data = results["benchmarks"]

    # Header
    sep = "─"^70
    println(sep)
    println("  GPUBenchmark.jl SUMMARY REPORT")
    println("  Node:  $(meta["hostname"])")
    println("  Time:  $(results["timestamp"])")
    println("  CUDA:  $(meta["cuda_runtime"]) (Driver: $(meta["cuda_driver"]))")
    println(sep)

    # Benchmark table
    tasks = String[]; perf = String[]; details = String[]

    function add_row!(t, p, d)
        push!(tasks, string(t)); push!(perf, string(p)); push!(details, string(d))
    end

    if haskey(bench_data, "sysinfo")
        si = bench_data["sysinfo"]
        if !haskey(si, "error")
            gpu_name = get(si, "gpu_name", "N/A")
            gpu_count = get(si, "gpu_count", 1)
            gpu_count > 1 && (gpu_name = "$(gpu_count)x $gpu_name")
            add_row!("GPU", gpu_name, get(si, "vram_total", ""))
            haskey(si, "cpu_model") && add_row!("CPU", si["cpu_model"], "$(get(si, "cpu_threads", "?")) threads")
            haskey(si, "ram_total") && add_row!("System RAM", si["ram_total"], "Free: $(get(si, "ram_free", "?"))")
        end
    end

    if haskey(bench_data, "bandwidth")
        bw = bench_data["bandwidth"]
        if !haskey(bw, "error")
            haskey(bw, "gpu_bandwidth_gibs") &&
                add_row!("BW GPU (STREAM)", @sprintf("%.1f GiB/s", bw["gpu_bandwidth_gibs"]), "triad A=B+s*C")
            haskey(bw, "cpu_bandwidth_gibs") &&
                add_row!("BW CPU (STREAM)", @sprintf("%.1f GiB/s", bw["cpu_bandwidth_gibs"]), "triad A=B+s*C")
        end
    end

    if haskey(bench_data, "matmul")
        m = bench_data["matmul"]
        !haskey(m, "error") && add_row!("MatMul (FP32)",
            @sprintf("%.2f TFLOPS", m["tflops"]),
            haskey(m, "matrix_size") ? "N=$(m["matrix_size"])" : "")
    end

    if haskey(bench_data, "scaling")
        s = bench_data["scaling"]
        if !haskey(s, "error")
            add_row!("Scaling",
                @sprintf("%.2f TFLOPS (GPU peak)", get(s, "peak_gpu_tflops", 0.0)),
                @sprintf("CPU: %.2f", get(s, "peak_cpu_tflops", 0.0)))

            peak_gpu_s = get(s, "peak_gpu_tflops", 0.0)
            peak_cpu_s = get(s, "peak_cpu_tflops", 0.0)
            if peak_gpu_s > 0 && peak_cpu_s > 0
                add_row!("Peak Speedup",
                    @sprintf("%.1f×  (GPU/CPU)", peak_gpu_s / peak_cpu_s),
                    "peak-over-peak (upper bound)")
            end

            # Power Efficiency if available
            if haskey(bench_data, "gpuinspector")
                gi = bench_data["gpuinspector"]
                if haskey(gi, "avg_power_w") && gi["avg_power_w"] > 0
                    eff = (peak_gpu_s * 1000) / gi["avg_power_w"]
                    add_row!("Efficiency", @sprintf("%.2f GFLOPS/W", eff), "Peak Perf/Avg Power")
                end
            end
        end
    end

    if !isempty(tasks)
        data = hcat(tasks, perf, details)
        pretty_table(data;
            column_labels = ["Task", "Performance / Model", "Details"],
            alignment = :l,
            table_format = TextTableFormat(borders = text_table_borders__compact),
            fit_table_in_display_vertically = false)
    else
        println("  No benchmark data found.")
    end

    # Latency histogram (after summary table)
    _render_latency_histogram(bench_data)

    # Scaling & Roofline (from scaling.dat)
    dat_path = joinpath(path, "scaling.dat")
    if isfile(dat_path)
        try
            dat = parse_scaling_dat(dat_path)
            _render_scaling_plots(dat)

            # Peaks and bandwidth
            peak_gpu = 10.0; peak_cpu = 1.0
            bw_gpu = 500.0; bw_source = "default"
            bw_cpu = 50.0

            if haskey(bench_data, "scaling")
                peak_gpu = get(bench_data["scaling"], "peak_gpu_tflops", 10.0)
                peak_cpu = get(bench_data["scaling"], "peak_cpu_tflops", 1.0)
            end

            # Bandwidth priority: built-in STREAM > GPUInspector > hardcoded default
            if haskey(bench_data, "bandwidth")
                bw = bench_data["bandwidth"]
                if haskey(bw, "gpu_bandwidth_gibs")
                    bw_gpu    = bw["gpu_bandwidth_gibs"]
                    bw_source = "STREAM triad"
                end
                if haskey(bw, "cpu_bandwidth_gibs")
                    bw_cpu = bw["cpu_bandwidth_gibs"]
                end
            end
            if haskey(bench_data, "gpuinspector")
                # GPUInspector is more thorough — takes priority over built-in STREAM
                gi_bw = get(bench_data["gpuinspector"], "memory_bandwidth", nothing)
                if !isnothing(gi_bw)
                    bw_gpu    = gi_bw
                    bw_source = "GPUInspector"
                end
            end

            # GPU: one roofline per device (sorted by device ID)
            sorted_gpu = sort(collect(dat.gpu_data), by = x -> tryparse(Int, x[1]) |> (v -> isnothing(v) ? 0 : v))
            for (gpu_id, (ns, ts, mins, maxs)) in sorted_gpu
                peak_i = isempty(ts) ? peak_gpu : maximum(ts)
                _render_roofline(ns, ts;
                    title="GPU $gpu_id", peak_tflops=peak_i,
                    peak_bw_gb=bw_gpu, bw_source=bw_source)
                _render_bandwidth_util_plot(ns, ts;
                    title="GPU $gpu_id", peak_bw_gb=bw_gpu)
            end

            # Speedup comparison (comparison range only)
            _render_speedup_plot(dat)

            # CPU roofline
            _render_roofline(dat.cpu_ns, dat.cpu_tflops;
                title="CPU", peak_tflops=peak_cpu, peak_bw_gb=bw_cpu, bw_source="estimated")
            _render_bandwidth_util_plot(dat.cpu_ns, dat.cpu_tflops;
                title="CPU", peak_bw_gb=bw_cpu)
            
        catch e
            @warn "Could not render scaling/roofline plots" exception=e
        end
    end

    # Telemetry plots (from extension)
    try
        render_telemetry(bench_data, path)
    catch e
        @warn "Could not render telemetry" exception=e
    end

    println(sep)
    println("  Path: $path")
    println(sep)
end

end # module Visuals
