# Visual Reporting for GPUBenchmark.jl
module Visuals

using Term: Table, Panel
using UnicodePlots
using JSON

export show_results

# Distinct colors for multi-GPU distinction
const GPU_COLORS = [:yellow, :cyan, :magenta, :blue, :green, :red, :white]

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

    # 3. Telemetry Trends (UnicodePlots)
    plots = ""
    if haskey(bench_data, "gpuinspector") && haskey(bench_data["gpuinspector"], "monitoring")
        mon = bench_data["gpuinspector"]["monitoring"]
        if haskey(mon, "metrics")
            metrics = mon["metrics"]
            
            # Smoothing and resampling helper
            function smooth_and_resample(data, target_points=80)
                if length(data) <= target_points
                    return Float64.(data)
                end
                
                # 1. Simple Moving Average (Smoothing)
                window = max(2, length(data) ÷ target_points)
                smoothed = [sum(data[i:min(i+window-1, end)]) / length(data[i:min(i+window-1, end)]) 
                           for i in 1:length(data)]
                
                # 2. Resample to target_points
                indices = round.(Int, range(1, length(smoothed), length=target_points))
                return smoothed[indices]
            end

            function create_sparkline(data, title, ylims=nothing)
                if isempty(data) return "" end
                
                # Data is Vector{Any} containing Vectors (one per GPU)
                all_vals = (data isa Vector || data isa AbstractVector) && !isempty(data) && first(data) isa AbstractVector ? data : [data]
                num_gpus = length(all_vals)
                
                # Create base plot with BrailleCanvas
                # Smooth and resample for professional curves
                first_vals = smooth_and_resample(all_vals[1])
                
                # Set limits if provided, otherwise auto
                kw = ylims === nothing ? NamedTuple() : (ylim=ylims,)
                
                p = lineplot(first_vals, 
                    title=title, 
                    name="GPU 0",
                    color=GPU_COLORS[1], 
                    width=65, 
                    height=10, 
                    border=:solid, 
                    canvas=BrailleCanvas,
                    xlabel="", ylabel="";
                    kw...
                )
                
                # Overlay other GPUs
                for i in 2:num_gpus
                    lineplot!(p, smooth_and_resample(all_vals[i]), 
                        name="GPU $(i-1)", 
                        color=GPU_COLORS[((i-1) % length(GPU_COLORS)) + 1]
                    )
                end
                
                # Summary Stats
                stats_lines = String[]
                for i in 1:num_gpus
                    v = Float64.(all_vals[i])
                    color = GPU_COLORS[((i-1) % length(GPU_COLORS)) + 1]
                    push!(stats_lines, "{$color}GPU $(i-1) Max: $(round(maximum(v), digits=1)){/$color}")
                end
                
                return string(p) * "\n  " * join(stats_lines, "  |  ")
            end

            available_plots = []
            haskey(metrics, "power") && push!(available_plots, create_sparkline(metrics["power"], "Power (W)"))
            haskey(metrics, "compute") && push!(available_plots, create_sparkline(metrics["compute"], "Compute Util (%)", (0, 100)))
            haskey(metrics, "mem") && push!(available_plots, create_sparkline(metrics["mem"], "Memory Util (%)", (0, 100)))
            haskey(metrics, "temperature") && push!(available_plots, create_sparkline(metrics["temperature"], "Temperature (°C)"))
            
            if !isempty(available_plots)
                plots = "{bold}Telemetry Trends (Burn-in Phase):{/bold}\n" * join(available_plots, "\n\n")
            end
        end
    end

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
