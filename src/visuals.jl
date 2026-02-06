# Visual Reporting for GPUBenchmark.jl
module Visuals

using Term: Table, Panel
using UnicodePlots
using JSON

export show_results

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
    rows = []
    
    if haskey(bench_data, "sysinfo")
        si = bench_data["sysinfo"]
        push!(rows, ["Hardware", si["gpu_name"], si["vram_total"]])
    end
    
    if haskey(bench_data, "matmul")
        m = bench_data["matmul"]
        push!(rows, ["MatMul (FP32)", "$(round(m["tflops"], digits=2)) TFLOPS", "N=$(m["matrix_size"])"])
    end

    if haskey(bench_data, "tensorcore")
        t = bench_data["tensorcore"]
        push!(rows, ["TensorCore", "$(round(t["tflops"], digits=2)) TFLOPS", "Mixed Prec"])
    end
    
    if haskey(bench_data, "gpuinspector")
        gi = bench_data["gpuinspector"]
        if haskey(gi, "memory_bandwidth")
            push!(rows, ["Memory BW", "$(round(gi["memory_bandwidth"], digits=2)) GiB/s", "Burn-in"])
        end
    end

    tbl = Table(
        rows,
        header=["Task", "Performance / Model", "Details"],
        columns_justify=[:left, :left, :right],
        columns_widths=[15, 30, 15],
        box=:ROUNDED,
        style="blue"
    )

    # 3. Telemetry Visuals (Future: Sparklines with UnicodePlots)
    
    println(Panel(
        header_content / "" / tbl,
        title=" {bold blue}GPUBenchmark.jl Summary Report{/bold blue} ",
        subtitle="{dim}Path: $path{/dim}",
        style="blue",
        padding=(2, 2, 1, 1),
        fit=true
    ))
end

end # module Visuals
