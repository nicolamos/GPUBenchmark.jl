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

function save_plots(args...) end
function cleanup(args...) end

function load_extension_dependencies(to_run)
    ext_map = Dict("gpuinspector" => [:GPUInspector, :CairoMakie])
    for name in to_run
        if haskey(ext_map, name)
            for pkg in ext_map[name]
                @info "STEP: Loading extension dependency..." package=pkg
                try
                    Base.eval(Main, :(using $pkg))
                catch e
                    @warn "Could not load optional package $pkg. Some features will be disabled."
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
    
    # 1. SETUP OUTPUT
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    full_hostname = get(ENV, "HOSTNAME", get(ENV, "COMPUTERNAME", "localhost"))
    hostname = split(full_hostname, '.') |> first
    run_dir = joinpath(parsed_args["output-dir"], hostname, timestamp)
    mkpath(run_dir)

    # 2. CONFIGURE LOGGING
    log_file = joinpath(run_dir, "benchmark.log")
    tee_logger = TeeLogger(
        ConsoleLogger(stdout),
        SimpleLogger(open(log_file, "w"))
    )

    with_logger(tee_logger) do
        @info "🚀 GPU BENCHMARK SUITE STARTING" 
        @info "  - Node:      $hostname"
        @info "  - Time:      $timestamp"
        @info "  - Artifacts: $run_dir"
        println("-"^60)

        # 3. PRE-FLIGHT
        if parsed_args["size"] == 0
            parsed_args["size"] = calculate_reasonable_size(parsed_args["fraction"])
        end
        load_extension_dependencies(parsed_args["benchmarks"])

        results = Dict{String, Any}()
        results["timestamp"] = timestamp
        results["cuda_functional"] = CUDA.functional()
        results["metadata"] = Dict(
            "hostname" => full_hostname,
            "cuda_runtime" => results["cuda_functional"] ? string(CUDA.runtime_version()) : "N/A",
            "cuda_driver" => results["cuda_functional"] ? string(CUDA.driver_version()) : "N/A"
        )
        results["benchmarks"] = Dict{String, Any}()

        # 4. EXECUTION
        to_run = parsed_args["benchmarks"]
        if "all" in to_run
            to_run = union(collect(keys(REGISTRY)), ["gpuinspector"])
        end

        for name in to_run
            if haskey(REGISTRY, name)
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

        # 5. POST-FLIGHT
        @info "STEP: Finalizing reports..."
        try
            Base.invokelatest(cleanup)
            open(joinpath(run_dir, "metrics.json"), "w") do f JSON.print(f, results, 4) end
            generate_text_report(results, run_dir)
            Base.invokelatest(save_plots, results, run_dir)
            @info "✅ All reports saved successfully."
        catch e
            @error "Failed to save final reports" exception=e
        end
        
        println("-"^60)
        @info "🚀 BENCHMARK COMPLETE"
    end

    return 0
end

end # module
