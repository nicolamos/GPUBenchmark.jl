# Visual Reporting for GPUBenchmark.jl
module Visuals

using PrettyTables
using UnicodePlots
using JSON
using Printf

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

function _render_scaling_plot(dat)
    if isempty(dat.cpu_ns) && isempty(dat.gpu_data)
        return
    end
    println("\n  Scaling: $(dat.alg)")
    # Collect all GPU points, take max per N for the terminal plot
    if !isempty(dat.gpu_data)
        n_tflops = Dict{Int,Float64}()
        for (_, (ns, ts)) in dat.gpu_data
            for (n, t) in zip(ns, ts)
                n_tflops[n] = max(get(n_tflops, n, 0.0), t)
            end
        end
        ns = sort(collect(keys(n_tflops)))
        ts = [n_tflops[n] for n in ns]
        render_lineplot(ns, ts; title="GPU Scaling TFLOPS vs N", xlabel="N", ylabel="TFLOPS", color=:green)
    end
    if !isempty(dat.cpu_ns)
        ord = sortperm(dat.cpu_ns)
        render_lineplot(dat.cpu_ns[ord], dat.cpu_tflops[ord];
            title="CPU Scaling TFLOPS vs N", xlabel="N", ylabel="TFLOPS", color=:blue)
    end
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

    if haskey(bench_data, "tensorcore")
        t = bench_data["tensorcore"]
        !haskey(t, "error") && add_row!("TensorCore",
            @sprintf("%.2f TFLOPS", t["tflops"]),
            haskey(t, "matrix_size") ? "N=$(t["matrix_size"]) Mixed Prec" : "Mixed Prec")
    end

    if haskey(bench_data, "scaling")
        s = bench_data["scaling"]
        !haskey(s, "error") && add_row!("Scaling",
            @sprintf("%.2f TFLOPS (GPU peak)", get(s, "peak_gpu_tflops", 0.0)),
            @sprintf("CPU: %.2f", get(s, "peak_cpu_tflops", 0.0)))
    end

    if haskey(bench_data, "gpuinspector")
        gi = bench_data["gpuinspector"]
        haskey(gi, "memory_bandwidth") && add_row!("Memory BW",
            @sprintf("%.2f GiB/s", gi["memory_bandwidth"]), "Burn-in")
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

    # Scaling terminal plot (from scaling.dat)
    dat_path = joinpath(path, "scaling.dat")
    if isfile(dat_path)
        try
            dat = parse_scaling_dat(dat_path)
            _render_scaling_plot(dat)
        catch e
            @warn "Could not render scaling plot" exception=e
        end
    end

    # Telemetry plots (from extension)
    render_telemetry(bench_data, path)

    println(sep)
    println("  Path: $path")
    println(sep)
end

end # module Visuals
