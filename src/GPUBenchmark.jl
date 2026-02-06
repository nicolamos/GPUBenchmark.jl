module GPUBenchmark

using CUDA
using JSON
using ArgParse
using Dates
using Printf
using InteractiveUtils
using Statistics
using Logging
using LoggingExtras
using Term
using UnicodePlots

# --- Registry System ---

struct BenchmarkTask
    name::String
    description::String
    run_func::Function
end

const REGISTRY = Dict{String, BenchmarkTask}()

"""
    register_benchmark(name, description, run_func)

Register a new benchmark task in the global registry.
"""
function register_benchmark(name::String, description::String, run_func::Function)
    REGISTRY[name] = BenchmarkTask(name, description, run_func)
end

export register_benchmark

# --- CLI Handling ---

function parse_commandline(args)
    s = ArgParseSettings(description = "Julia GPU Benchmark Suite - Production Health Check")

    @add_arg_table! s begin
        "--list", "-l"
            help = "List available benchmarks and exit"
            action = :store_true
        "--size", "-s"
            help = "Global matrix size (N). If 0, auto-scales to ~40% VRAM."
            arg_type = Int
            default = 0
        "--duration", "-d"
            help = "Stress test duration (seconds)."
            arg_type = Int
            default = 30
        "--fraction", "-f"
            help = "Target VRAM usage fraction for auto-sizing."
            arg_type = Float64
            default = 0.4
        "--output-dir", "-o"
            help = "Root directory for results."
            arg_type = String
            default = "results"
        "--verbose", "-v"
            help = "Enable debug logging."
            action = :store_true
        "--quiet", "-q"
            help = "Suppress terminal dashboard (useful for batch/ansible)."
            action = :store_true
        "benchmarks"
            help = "Tasks: sysinfo (default), matmul, gpuinspector, all."
            nargs = '*'
            default = ["sysinfo"]
    end

    return parse_args(args, s)
end

function list_benchmarks()
    println("\n📋 Registered Benchmark Tasks:")
    println("-"^40)
    for (name, task) in sort(collect(REGISTRY), by=x->x[1])
        @printf("  %-15s : %s\n", name, task.description)
    end
    println("-"^40)
end

function calculate_reasonable_size(fraction=0.4)
    if !CUDA.functional()
        @warn "CUDA not functional, defaulting to minimal size."
        return 2048
    end

    free_mem = CUDA.available_memory()
    # N = sqrt( (free_mem * fraction) / (3 matrices * 4 bytes per Float32) )
    N = isqrt(Int(floor((free_mem * fraction) / 12)))
    
    # Align to 128 for Tensor Core performance
    N = div(N, 128) * 128
    N = max(2048, N)
    
    @info "STEP: Auto-calculating matrix size..." 
    @info "  - VRAM Free: $(Base.format_bytes(free_mem))"
    @info "  - Target:    $(round(fraction*100))% usage"
    @info "  - Result:    N = $N"
    
    return N
end

# --- Result Visualization (Terminal Dashboard) ---

"""
    show_results(path::String)

Render a professional terminal dashboard for a specific benchmark run.
"""
function show_results(path::String)
    metrics_path = joinpath(path, "metrics.json")
    if !filemode(metrics_path) == 0 && !isfile(metrics_path)
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
        columns_width=[15, 30, 15],
        box=:ROUNDED,
        style="blue"
    )

    # 3. Telemetry Visuals (Sparklines)
    # Note: In a real scenario we'd load telemetry.h5, 
    # but for simplicity we'll check if averages are in metrics.json 
    # or just show a status panel.
    
    println(Panel(
        header_content / "" / tbl,
        title=" {bold blue}GPUBenchmark.jl Summary Report{/bold blue} ",
        subtitle="{dim}Path: $path{/dim}",
        style="blue",
        padding=(2, 2, 1, 1),
        fit=true
    ))
end

"""
    show_latest(output_dir="results")

Convenience helper to show the most recent benchmark run.
"""
function show_latest(output_dir="results")
    if !isdir(output_dir)
        @warn "Output directory '$output_dir' does not exist."
        return
    end
    
    # Find hostname subdirectories
    host_dirs = filter(isdir, [joinpath(output_dir, d) for d in readdir(output_dir)])
    if isempty(host_dirs) return end
    
    # Find all timestamped runs
    all_runs = String[]
    for hdir in host_dirs
        append!(all_runs, filter(isdir, [joinpath(hdir, d) for d in readdir(hdir)]))
    end
    
    if isempty(all_runs) return end
    
    # Sort by folder name (timestamped)
    latest_run = sort(all_runs)[end]
    show_results(latest_run)
end

function generate_text_report(results, output_dir)
    filename = joinpath(output_dir, "summary.txt")
    open(filename, "w") do io
        println(io, "================================================================")
        println(io, "             GPU HEALTH CERTIFICATE - $(results["timestamp"])")
        println(io, "================================================================")
        println(io, "")
        println(io, "[1. NODE ENVIRONMENT]")
        println(io, "  Hostname:      $(results["metadata"]["hostname"])")
        println(io, "  Julia:         v$(VERSION)")
        if results["cuda_functional"]
            println(io, "  CUDA Runtime:  $(results["metadata"]["cuda_runtime"])")
            println(io, "  CUDA Driver:   $(results["metadata"]["cuda_driver"])")
        else
            println(io, "  CUDA:          ⚠️ NOT FUNCTIONAL")
        end
        
        println(io, "")
        println(io, "[2. HARDWARE CAPABILITY]")
        if haskey(results["benchmarks"], "sysinfo")
            si = results["benchmarks"]["sysinfo"]
            if !haskey(si, "error")
                println(io, "  GPU Model:     $(si["gpu_name"])")
                println(io, "  Compute Cap:   $(si["compute_capability"])")
                println(io, "  Total VRAM:    $(si["vram_total"])")
                println(io, "  PCI UUID:      $(si["pci_bus_id"])")
            else
                println(io, "  Status:        ⚠️ Failed to collect capability info.")
            end
        end

        println(io, "")
        println(io, "[3. PERFORMANCE RESULTS]")
        
        if haskey(results["benchmarks"], "matmul")
            m = results["benchmarks"]["matmul"]
            if !haskey(m, "error")
                @printf(io, "  Raw Compute:   %.2f TFLOPS (FP32)\n", m["tflops"])
                @printf(io, "  Matrix Size:   %d x %d\n", m["matrix_size"], m["matrix_size"])
            end
        end

        if haskey(results["benchmarks"], "gpuinspector")
            r = results["benchmarks"]["gpuinspector"]
            if !haskey(r, "error")
                if haskey(r, "memory_bandwidth")
                    @printf(io, "  Memory BW:     %.2f GiB/s\n", r["memory_bandwidth"])
                end
                
                if haskey(r, "_raw_monitoring")
                    mon = r["_raw_monitoring"]
                    if haskey(mon.results, :power)
                        all_p = reduce(vcat, mon.results[:power])
                        @printf(io, "  Peak Power:    %.1f W (Avg: %.1f W)\n", maximum(all_p), mean(all_p))
                    end
                    if haskey(mon.results, :temperature)
                        all_t = reduce(vcat, mon.results[:temperature])
                        @printf(io, "  Peak Temp:     %d °C\n", Int(maximum(all_t)))
                    end
                end
            end
        end
        println(io, "")
        println(io, "================================================================")
        println(io, "Generated by GPUBenchmark.jl Suite")
    end
end

function discover_benchmarks()
    bench_dir = joinpath(@__DIR__, "benchmarks")
    if isdir(bench_dir)
        files = filter(f -> endswith(f, ".jl"), readdir(bench_dir))
        for file in files
            try
                include(joinpath(bench_dir, file))
            catch e
                @error "Failed to load benchmark module: $file" exception=e
            end
        end
    end
end

# Stubs for extension methods
function save_plots(args...) end
function cleanup(args...) end

"""
    load_extension_dependencies(to_run)

Forcing the loading of packages that trigger Pkg extensions.
"""
function load_extension_dependencies(to_run)
    # If "all" is requested, we definitely want the extensions
    needs_gpuinspector = "all" in to_run || "gpuinspector" in to_run
    
    if needs_gpuinspector
        @info "STEP: Loading extension dependencies (GPUInspector, CairoMakie)..."
        try
            # We must load them in Main to ensure they are visible globally
            Base.eval(Main, :(using GPUInspector))
            Base.eval(Main, :(using CairoMakie))
        catch e
            @warn "Could not load extension dependencies. Parallel stress test will be unavailable." exception=e
        end
    end
end

# --- Entry Points ---

function (@main)(ARGS)
    # 1. Discover static benchmarks
    discover_benchmarks()
    
    # 2. Parse initial command line to see what's requested
    parsed_args = parse_commandline(ARGS)

    # 3. Load extensions BEFORE building the final task list
    load_extension_dependencies(parsed_args["benchmarks"])

    if parsed_args["list"]
        Base.invokelatest(list_benchmarks)
        return 0
    end
    
    # 4. SETUP OUTPUT
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    full_hostname = get(ENV, "HOSTNAME", get(ENV, "COMPUTERNAME", "localhost"))
    hostname = split(full_hostname, '.') |> first
    run_dir = joinpath(parsed_args["output-dir"], hostname, timestamp)
    mkpath(run_dir)

    # 5. CONFIGURE LOGGING
    log_file = joinpath(run_dir, "benchmark.log")
    min_level = parsed_args["verbose"] ? Logging.Debug : Logging.Info
    tee_logger = MinLevelLogger(
        TeeLogger(
            ConsoleLogger(stdout),
            SimpleLogger(open(log_file, "w"))
        ),
        min_level
    )

    with_logger(tee_logger) do
        @info "🚀 GPU BENCHMARK SUITE STARTING" 
        @info "  - Node:      $hostname"
        @info "  - Time:      $timestamp"
        @info "  - Artifacts: $run_dir"
        println("-"^60)

        # 6. PRE-FLIGHT
        if parsed_args["size"] == 0
            parsed_args["size"] = calculate_reasonable_size(parsed_args["fraction"])
        end

        results = Dict{String, Any}()
        results["timestamp"] = timestamp
        results["cuda_functional"] = CUDA.functional()
        results["metadata"] = Dict(
            "hostname" => full_hostname,
            "cuda_runtime" => results["cuda_functional"] ? string(CUDA.runtime_version()) : "N/A",
            "cuda_driver" => results["cuda_functional"] ? string(CUDA.driver_version()) : "N/A"
        )
        results["benchmarks"] = Dict{String, Any}()

        # 7. EXECUTION
        to_run = parsed_args["benchmarks"]
        if "all" in to_run
            # Now that extensions are loaded, collect all keys from registry
            # We explicitly include "gpuinspector" if not yet registered but requested
            to_run = union(collect(keys(REGISTRY)), ["gpuinspector"])
        end

        for name in to_run
            # Use invokelatest to ensure we see tasks registered by extensions in this world age
            if Base.invokelatest(haskey, REGISTRY, name)
                @info "STEP: Running task '$name'..."
                try
                    results["benchmarks"][name] = Base.invokelatest(REGISTRY[name].run_func, parsed_args)
                    # Inline Summary
                    if name == "matmul" && !haskey(results["benchmarks"][name], "error")
                        @info "  - Result: $(round(results["benchmarks"][name]["tflops"], digits=2)) TFLOPS"
                    end
                    Base.invokelatest(cleanup)
                catch e
                    @error "Task '$name' failed!" exception=e
                    results["benchmarks"][name] = Dict("error" => string(e))
                end
            else
                @warn "Task '$name' not found in registry. Skipping."
            end
        end

        # 8. POST-FLIGHT
        @info "STEP: Finalizing reports..."
        try
            Base.invokelatest(cleanup)
            open(joinpath(run_dir, "metrics.json"), "w") do f JSON.print(f, results, 4) end
            generate_text_report(results, run_dir)
            # Extension-provided method
            Base.invokelatest(save_plots, results, run_dir)
            @info "✅ All reports saved successfully."
            
            # 9. CLI DASHBOARD
            if !parsed_args["quiet"]
                println("\n")
                show_results(run_dir)
            end
            
        catch e
            @error "Failed to save final reports" exception=e
        end
        
        println("-"^60)
        @info "🚀 BENCHMARK COMPLETE"
    end

    return 0
end

# Backward compatibility for julia -m GPUBenchmark
function main()
    (@main)(ARGS)
end

end # module
