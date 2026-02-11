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
    cpu_ns = Int[]; cpu_tflops = Float64[]
    gpu_ns   = Dict{String, Vector{Int}}()
    gpu_tflops = Dict{String, Vector{Float64}}()
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
            if mode == "CPU"
                push!(cpu_ns, n); push!(cpu_tflops, tflops)
            else
                push!(get!(gpu_ns,     id, Int[]),     n)
                push!(get!(gpu_tflops, id, Float64[]), tflops)
            end
        end
    end
    gpu_data = Dict(id => (gpu_ns[id], gpu_tflops[id]) for id in keys(gpu_ns))
    return (alg=alg, cpu_ns=cpu_ns, cpu_tflops=cpu_tflops, gpu_data=gpu_data)
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

function _render_roofline(dat; device="GPU", peak_tflops=10.0, peak_bw_gb=500.0)
    # Intensity I = FLOPs / Byte. For MatMul Float32: I = N/6.
    ns = device == "GPU" ? (isempty(dat.gpu_data) ? Int[] : first(values(dat.gpu_data))[1]) : dat.cpu_ns
    ts = device == "GPU" ? (isempty(dat.gpu_data) ? Float64[] : first(values(dat.gpu_data))[2]) : dat.cpu_tflops
    
    isempty(ns) && return
    
    is = ns ./ 6.0 
    max_i = maximum(is) * 1.2
    roof_is = range(0.1, max_i, length=100)
    roof_ts = [min(peak_tflops, i * peak_bw_gb / 1000.0) for i in roof_is]
    
    println("\n  Roofline Analysis ($(device)):")
    p = lineplot(roof_is, roof_ts, title="$(device) Roofline", xlabel="Intensity (FLOP/Byte)", ylabel="TFLOPS", color=:white, width=70, height=12)
    scatterplot!(p, is, ts, color=:cyan)
    show(stdout, MIME"text/plain"(), p)
    println()
    println("  Guide: Points on the slope are Bandwidth-Bound; points on the ceiling are Compute-Bound.")
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
    println("  GPUBenchmark.jl Summary Report")
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

    if haskey(bench_data, "matmul")
        m = bench_data["matmul"]
        !haskey(m, "error") && add_row!("MatMul (FP32)",
            @sprintf("%.2f TFLOPS", m["tflops"]),
            haskey(m, "matrix_size") ? "N=$(m["matrix_size"])" : "")
    end

    if haskey(bench_data, "scaling")
        s = bench_data["scaling"]
        !haskey(s, "error") && add_row!("Scaling",
            @sprintf("%.2f TFLOPS (GPU peak)", get(s, "peak_gpu_tflops", 0.0)),
            @sprintf("CPU: %.2f", get(s, "peak_cpu_tflops", 0.0)))

        # Power Efficiency if available
        if haskey(bench_data, "gpuinspector")
            gi = bench_data["gpuinspector"]
            if haskey(gi, "avg_power_w") && gi["avg_power_w"] > 0
                eff = (get(s, "peak_gpu_tflops", 0.0) * 1000) / gi["avg_power_w"]
                add_row!("Efficiency", @sprintf("%.2f GFLOPS/W", eff), "Peak Perf/Avg Power")
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

    # Scaling & Roofline (from scaling.dat)
    dat_path = joinpath(path, "scaling.dat")
    if isfile(dat_path)
        try
            dat = parse_scaling_dat(dat_path)
            _render_scaling_plots(dat)
            
            # Use results for roofs
            peak_gpu = 10.0; peak_cpu = 1.0; bw_gpu = 500.0; bw_cpu = 50.0
            if haskey(bench_data, "scaling")
                peak_gpu = get(bench_data["scaling"], "peak_gpu_tflops", 10.0)
                peak_cpu = get(bench_data["scaling"], "peak_cpu_tflops", 1.0)
            end
            if haskey(bench_data, "gpuinspector")
                bw_gpu = get(bench_data["gpuinspector"], "memory_bandwidth", 500.0)
            end
            
            _render_roofline(dat, device="GPU", peak_tflops=peak_gpu, peak_bw_gb=bw_gpu)
            _render_roofline(dat, device="CPU", peak_tflops=peak_cpu, peak_bw_gb=bw_cpu)
            
        catch e
            @warn "Could not render scaling/roofline plots" exception=e
        end
    end

    # Telemetry plots (from extension)
    render_telemetry(bench_data, path)

    println(sep)
    println("  Path: $path")
    println(sep)
end

end # module Visuals
