module GPUBenchmark

using CUDA
using JSON
using ArgParse
using Dates
using Printf
using InteractiveUtils
using Statistics

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
    s = ArgParseSettings(description = "Julia GPU Benchmark Suite")

    @add_arg_table! s begin
        "--list", "-l"
            help = "List available benchmarks and exit"
            action = :store_true
        "--size", "-s"
            help = "Global size parameter (matrix size). If 0, auto-calculates based on VRAM."
            arg_type = Int
            default = 0
        "--duration", "-d"
            help = "Duration of the stress test in seconds."
            arg_type = Int
            default = 30
        "--fraction", "-f"
            help = "Fraction of VRAM to target during auto-sizing."
            arg_type = Float64
            default = 0.4
        "--output-dir", "-o"
            help = "Base directory to save results. A timestamped subdirectory will be created."
            arg_type = String
            default = "results"
        "benchmarks"
            help = "Names of benchmarks to run (space separated). Default: sysinfo"
            nargs = '*'
            default = ["sysinfo"]
    end

    return parse_args(args, s)
end

function list_benchmarks()
    println("Available Benchmarks:")
    for (name, task) in sort(collect(REGISTRY), by=x->x[1])
        println("  - $(rpad(name, 15)): $(task.description)")
    end
end

"""
    calculate_reasonable_size(fraction=0.4)

Calculate a matrix size N such that 3 matrices (A, B, C) of Float32
occupy `fraction` of the available VRAM.
"""
function calculate_reasonable_size(fraction=0.4)
    if !CUDA.functional()
        return 2048 # Fallback
    end
    
    # Get free memory in bytes
    free_mem = CUDA.available_memory()
    
    # We need 3 matrices (A, B, C) of Float32 (4 bytes)
    # Total bytes = 3 * N^2 * 4 = 12 * N^2
    # 12 * N^2 = free_mem * fraction
    # N = sqrt( (free_mem * fraction) / 12 )
    
    N = isqrt(Int(floor((free_mem * fraction) / 12)))
    
    # Round down to nearest multiple of 128 for tensor core alignment niceness
    N = div(N, 128) * 128
    
    # Clamp to reasonable bounds
    N = max(2048, N)
    
    @info "Auto-calculated matrix size: $N (targeting $(round(fraction*100))% of $(Base.format_bytes(free_mem)) free VRAM)"
    return N
end

function generate_text_report(results, output_dir)
    filename = joinpath(output_dir, "summary.txt")
    open(filename, "w") do io
        println(io, "================================================================")
        println(io, "             GPU BENCHMARK REPORT - $(results["timestamp"])")
        println(io, "================================================================")
        println(io, "")
        println(io, "[System Information]")
        println(io, "Hostname:      $(get(ENV, "HOSTNAME", "unknown"))")
        println(io, "Julia Version: $(VERSION)")
        if CUDA.functional()
            println(io, "CUDA Runtime:  $(CUDA.runtime_version())")
            println(io, "CUDA Driver:   $(CUDA.driver_version())")
            dev = CUDA.device()
            println(io, "Active GPU:    $(CUDA.name(dev))")
            println(io, "Compute Cap:   $(CUDA.capability(dev))")
            println(io, "Total VRAM:    $(Base.format_bytes(CUDA.totalmem(dev)))")
        else
            println(io, "CUDA:          Not Functional")
        end
        println(io, "")
        println(io, "[Hardware Capability]")
        if haskey(results["benchmarks"], "sysinfo")
            si = results["benchmarks"]["sysinfo"]
            if haskey(si, "gpu_name")
                println(io, "GPU Name:      $(si["gpu_name"])")
                println(io, "Compute Cap:   $(si["compute_capability"])")
                println(io, "VRAM Total:    $(si["vram_total"])")
                println(io, "VRAM Free:     $(si["vram_free"])")
                println(io, "PCI UUID:      $(si["pci_bus_id"])")
            end
        end
        
        println(io, "")
        println(io, "[Benchmark Results]")
        
        if haskey(results["benchmarks"], "gpuinspector")
            r = results["benchmarks"]["gpuinspector"]
            if haskey(r, "memory_bandwidth")
                bw = r["memory_bandwidth"]
                @printf(io, "Memory Bandwidth: %.2f GiB/s\n", bw)
            end
            
            if haskey(r, :_raw_monitoring)
                mon = r[:_raw_monitoring]
                # Calculate avg metrics
                if haskey(mon.results, :power)
                    # Average over all devices and samples
                    all_power = reduce(vcat, mon.results[:power])
                    avg_p = mean(all_power)
                    max_p = maximum(all_power)
                    @printf(io, "Avg Power Draw:   %.1f W (Peak: %.1f W)\n", avg_p, max_p)
                end
                if haskey(mon.results, :temperature)
                    max_t = maximum(reduce(vcat, mon.results[:temperature]))
                    @printf(io, "Peak Temp:        %d °C\n", Int(max_t))
                end
            end
        end
        println(io, "")
        println(io, "================================================================")
        println(io, "Generated by GPUBenchmark.jl")
    end
end

"""
    discover_benchmarks()

Dynamically load all benchmark modules from the benchmarks directory.
"""
function discover_benchmarks()
    bench_dir = joinpath(@__DIR__, "benchmarks")
    if isdir(bench_dir)
        for file in readdir(bench_dir)
            if endswith(file, ".jl")
                include(joinpath(bench_dir, file))
            end
        end
    end
end

"""
    save_plots(results, output_path)

Stub for saving plots, implemented in extensions.
"""
function save_plots(args...)
    # Default: do nothing if extension not loaded
    return nothing
end

"""
    load_extension_dependencies(to_run)

Try to load packages that trigger Pkg extensions for specific benchmarks.
"""
function load_extension_dependencies(to_run)
    # Map benchmark names to the packages that trigger their extensions
    ext_map = Dict(
        "gpuinspector" => [:GPUInspector, :CairoMakie]
    )

    for name in to_run
        if haskey(ext_map, name)
            pkg_names = ext_map[name]
            for pkg_name in pkg_names
                @info "Loading extension dependency for $name: $pkg_name"
                try
                    # Use Base.require to dynamically load the package if it's available
                    # This triggers the extension loading
                    Base.eval(Main, :(using $pkg_name))
                catch e
                    @warn "Failed to load extension dependency $pkg_name for benchmark $name. Ensure it is installed in the environment." exception=e
                end
            end
        end
    end
end

function (@main)(ARGS)
    discover_benchmarks()
    
    parsed_args = parse_commandline(ARGS)

    if parsed_args["list"]
        list_benchmarks()
        return 0
    end
    
    # Auto-calculate size if 0 (default)
    if parsed_args["size"] == 0
        parsed_args["size"] = calculate_reasonable_size(parsed_args["fraction"])
    end

    # Determine which benchmarks to run
    to_run = parsed_args["benchmarks"]
    if "all" in to_run
        # Add all registered benchmarks plus extension benchmarks
        to_run = union(collect(keys(REGISTRY)), ["gpuinspector"])
    end

    # Load dependencies for extensions if needed
    load_extension_dependencies(to_run)

    results = Dict{String, Any}()
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    results["timestamp"] = timestamp
    results["cuda_functional"] = CUDA.functional()
    results["benchmarks"] = Dict{String, Any}()

    @info "Starting GPU Benchmark Suite" benchmarks=to_run size=parsed_args["size"]
    
    for name in to_run
        if haskey(REGISTRY, name)
            @info "Running benchmark: $name"
            try
                # Use invokelatest to avoid world age issues with discover_benchmarks
                results["benchmarks"][name] = Base.invokelatest(REGISTRY[name].run_func, parsed_args)
            catch e
                @error "Benchmark $name failed" exception=e
                results["benchmarks"][name] = Dict("error" => string(e))
            end
        else
            @warn "Benchmark '$name' not found in registry. Skipping."
        end
    end

    # Create timestamped output directory
    base_dir = parsed_args["output-dir"]
    hostname = get(ENV, "HOSTNAME", "localhost")
    run_dir = joinpath(base_dir, hostname, timestamp)
    mkpath(run_dir)
    
    @info "Saving results to $run_dir"

    # 1. JSON (Metadata & High-level metrics)
    json_path = joinpath(run_dir, "metrics.json")
    open(json_path, "w") do f
        JSON.print(f, results, 4)
    end
    
    # 2. Text Report
    generate_text_report(results, run_dir)

    # 3. Plots & Telemetry (HDF5/PNG handled by extensions)
    # Use invokelatest for the extension-provided method
    Base.invokelatest(save_plots, results, run_dir)
    
    return 0
end

end # module