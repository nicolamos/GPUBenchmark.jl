module GPUBenchmark

using CUDA
using JSON
using Dates
using Printf
using Logging
using LoggingExtras

include("Core.jl")
using .Core

include("Visuals.jl")
using .Visuals

include("Reporting.jl")
using .Reporting

include("Engine.jl")
using .Engine

include("CLI.jl")
using .CLI

# Re-export public API
export register_benchmark, register_algorithm, AbstractAlgorithm, run_cpu, run_gpu

# Stubs for extension methods (overridden by extensions)
function save_plots(args...) end
function cleanup(args...) end

# --- Main Entry Point ---

function (@main)(ARGS)
    # 1. Parse command line
    parsed_args = parse_commandline(ARGS)
    isnothing(parsed_args) && return 0

    # 2. Discovery
    discover_benchmarks(parsed_args.plugin)

    # 3. Extensions
    load_extension_dependencies(parsed_args)

    if parsed_args.list
        Base.invokelatest(list_benchmarks)
        return 0
    end

    if parsed_args.show_latest
        Base.invokelatest(show_latest, parsed_args.output_dir)
        return 0
    end

    if !isnothing(parsed_args.show)
        Base.invokelatest(Visuals.show_results, parsed_args.show)
        return 0
    end

    # 4. SETUP OUTPUT
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    full_hostname = get(ENV, "HOSTNAME", get(ENV, "COMPUTERNAME", "localhost"))
    hostname = split(full_hostname, '.') |> first
    run_dir = joinpath(parsed_args.output_dir, hostname, timestamp)
    mkpath(run_dir)

    # 5. CONFIGURE LOGGING
    log_file = joinpath(run_dir, "benchmark.log")
    min_level = parsed_args.verbose ? Logging.Debug : Logging.Info
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
        size = parsed_args.size
        if size == 0
            size = calculate_reasonable_size(parsed_args.fraction)
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
        to_run = parsed_args.benchmarks
        if "all" in to_run
            to_run = union(collect(keys(REGISTRY)), ["gpuinspector"])
            if CUDA.functional() && capability(CUDA.device()) < v"7.0"
                to_run = filter(x -> x != "tensorcore", to_run)
            end
        end

        for name in to_run
            if Base.invokelatest(haskey, REGISTRY, name)
                @info "STEP: Running task '$name'..."
                try
                    # Legacy tasks expect a Dict of string keys
                    task_args = Dict(string(k) => v for (k, v) in pairs(parsed_args))
                    task_args["size"] = size

                    results["benchmarks"][name] = Base.invokelatest(REGISTRY[name].run_func, task_args)

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

            if haskey(results["benchmarks"], "scaling")
                scaling_raw = results["benchmarks"]["scaling"]
                export_scaling_dat(scaling_raw, run_dir)

                results["benchmarks"]["scaling"] = Dict(
                    "algorithm" => scaling_raw["algorithm"],
                    "peak_cpu_tflops" => maximum(r -> r["tflops"], scaling_raw["cpu_results"]),
                    "peak_gpu_tflops" => maximum([maximum(r -> r["tflops"], runs) for (_, runs) in scaling_raw["gpu_results"]])
                )
            end

            open(joinpath(run_dir, "metrics.json"), "w") do f JSON.print(f, results, 4) end
            generate_text_report(results, run_dir)
            Base.invokelatest(save_plots, results, run_dir)
            @info "✅ All reports saved successfully."

            if !parsed_args.quiet
                println("\n")
                Base.invokelatest(Visuals.show_results, run_dir)
            end
        catch e
            @error "Failed to save final reports" exception=e
        end

        println("-"^60)
        @info "🚀 BENCHMARK COMPLETE"
    end

    return 0
end

# Backward compatibility
function main()
    (@main)(ARGS)
end

end # module
