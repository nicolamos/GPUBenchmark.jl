# Visual Reporting for GPUBenchmark.jl
module Visuals

using Term: Table, Panel
using UnicodePlots
using JSON
using Statistics: mean

export show_results

# Stubs for extension overrides
function render_telemetry(bench_data, run_path)
    # Fallback if GPUInspector isn't available
    if haskey(bench_data, "gpuinspector") && haskey(bench_data["gpuinspector"], "monitoring_file")
        return "{dim}Telemetry data available but GPUInspector extension not loaded.{/dim}"
    end
    return ""
end

function show_results(path::String)
    metrics_path = joinpath(path, "metrics.json")
    if !isfile(metrics_path)
        println(Panel("{red}Error: metrics.json not found in $path{/red}", title="Result Viewer"))
        return
    end

    results = JSON.parsefile(metrics_path)
    
    # 1. Header with Metadata
    meta = results["metadata"]
    header_content = """
    {bold}Node:{/bold}      $(meta["hostname"])
    {bold}Time:{/bold}      $(results["timestamp"])
    {bold}CUDA:{/bold}      $(meta["cuda_runtime"]) (Driver: $(meta["cuda_driver"]))
    """
    
    # 2. Benchmark Table
    bench_data = results["benchmarks"]
    
    # Term.Table expects a vector of vectors where each vector is a COLUMN
    # or a Matrix. Let's use columns.
    tasks = String[]
    perf = String[]
    details = String[]

    function add_row!(t, p, d)
        push!(tasks, string(t))
        push!(perf, string(p))
        push!(details, string(d))
    end
    
    if haskey(bench_data, "sysinfo")
        si = bench_data["sysinfo"]
        add_row!("Hardware", si["gpu_name"], si["vram_total"])
    end
    
    if haskey(bench_data, "matmul")
        m = bench_data["matmul"]
        add_row!("MatMul (FP32)", "$(round(m["tflops"], digits=2)) TFLOPS", "N=$(m["matrix_size"])")
    end

    if haskey(bench_data, "tensorcore")
        t = bench_data["tensorcore"]
        add_row!("TensorCore", "$(round(t["tflops"], digits=2)) TFLOPS", "Mixed Prec")
    end
    
    if haskey(bench_data, "gpuinspector")
        gi = bench_data["gpuinspector"]
        if haskey(gi, "memory_bandwidth")
            add_row!("Memory BW", "$(round(gi["memory_bandwidth"], digits=2)) GiB/s", "Burn-in")
        end
    end

    if isempty(tasks)
        tbl = "No benchmark data found in metrics."
    else
        # Term.Table expects a Matrix or similar Tables.jl compatible object
        # hcat creates a Matrix where each argument is a column.
        data = hcat(tasks, perf, details)
        tbl = Table(
            data,
            header=["Task", "Performance / Model", "Details"],
            columns_justify=[:left, :left, :right],
            columns_widths=[15, 30, 15],
            box=:ROUNDED,
            style="blue"
        )
    end

    # 3. Telemetry Trends
    plots = render_telemetry(bench_data, path)

    # 4. Final Layout
    # Use safe concatenation to avoid crashes if tbl is nothing
    content = header_content / ""
    if !isnothing(tbl)
        content = content / tbl
    else
        content = content / "{red}Failed to generate results table.{/red}"
    end

    if !isempty(plots)
        content = content / "" / plots
    end
    
    println(Panel(
        content,
        title=" {bold blue}GPUBenchmark.jl Summary Report{/bold blue} ",
        subtitle="{dim}Path: $path{/dim}",
        style="blue",
        padding=(2, 2, 1, 1),
        fit=true
    ))
end

end # module Visuals
