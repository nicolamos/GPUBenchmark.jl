module Engine

using CUDA
using Dates
using Logging
using ..Core
using ..Visuals # For show_results call in show_latest

export calculate_reasonable_size, calculate_scaling_sizes, discover_benchmarks, load_extension_dependencies, show_latest

function calculate_reasonable_size(fraction=0.4)
    if !CUDA.functional()
        @warn "CUDA not functional, defaulting to minimal size."
        return 2048
    end

    gpu_mem = CUDA.available_memory()
    sys_mem = Sys.free_memory()
    mem = min(gpu_mem, sys_mem)

    # N = sqrt( (mem * fraction) / (3 matrices * 4 bytes per Float32) )
    N = isqrt(Int(floor((mem * fraction) / 12)))

    # Align to 128 for Tensor Core performance
    N = div(N, 128) * 128
    N = max(2048, N)

    @debug "Auto-calculating matrix size" vram_free=Base.format_bytes(gpu_mem) ram_free=Base.format_bytes(sys_mem) target_fraction=fraction result_n=N
    return N
end

function calculate_scaling_sizes(fraction=0.6)
    max_n = calculate_reasonable_size(fraction)
    # Generate 5 logarithmic steps up to max_n
    return [max(1024, div(max_n, 2^i)) for i in 4:-1:0] |> unique |> sort
end

function discover_benchmarks(plugin_arg=nothing)
    # Built-in benchmarks are statically included in GPUBenchmark.jl (precompiled).
    # This function only handles external plugin discovery.

    # 1. Hybrid Plugin Discovery (LOAD_PATH + Module Import)
    local_plugins = abspath("plugins")
    if isdir(local_plugins)
        @info "STEP: Scanning local plugins/ directory..."
        if !(local_plugins in LOAD_PATH)
            push!(LOAD_PATH, local_plugins)
        end
        
        for file in filter(f -> endswith(f, ".jl"), readdir(local_plugins))
            mod_name = splitext(file)[1]
            try
                Base.eval(Main, :(using $(Symbol(mod_name))))
            catch e
                @error "Failed to load plugin module '$mod_name' from plugins/" exception=e
            end
        end
    end

    # 2. Explicit Plugin Loading (via --plugin flag)
    if !isnothing(plugin_arg)
        if isfile(plugin_arg)
            @warn "Loading plugin via file path is deprecated. Prefer placing modules in 'plugins/'."
            try
                Base.include(Main, plugin_arg)
            catch e
                @error "Failed to include plugin file: $plugin_arg" exception=e
            end
        else
            @info "STEP: Loading explicit plugin module: $plugin_arg"
            try
                Base.eval(Main, :(using $(Symbol(plugin_arg))))
            catch e
                @error "Failed to load plugin module: $plugin_arg" exception=e
            end
        end
    end
end

function load_extension_dependencies(parsed_args)
    to_run = parsed_args.benchmarks
    needs_gpuinspector = "all" in to_run || "gpuinspector" in to_run || parsed_args.show_latest || !isnothing(parsed_args.show)
    
    if needs_gpuinspector
        if !parsed_args.quiet
            @info "STEP: Loading extension dependencies (GPUInspector, CairoMakie)..."
        end
        try
            Base.eval(Main, :(using GPUInspector))
            Base.eval(Main, :(using CairoMakie))
        catch e
            if !parsed_args.quiet
                @warn "Could not load extension dependencies. Telemetry and plots will be limited."
            end
        end
    end
end

function show_latest(output_dir="results")
    if !isdir(output_dir)
        @warn "Output directory '$output_dir' not found."
        return
    end
    
    host_dirs = filter(isdir, [joinpath(output_dir, d) for d in readdir(output_dir)])
    all_runs = String[]
    for hdir in host_dirs
        append!(all_runs, filter(isdir, [joinpath(hdir, d) for d in readdir(hdir)]))
    end
    
    if isempty(all_runs)
        @warn "No benchmark results found."
        return
    end
    
    latest_run = sort(all_runs)[end]
    Base.invokelatest(Visuals.show_results, latest_run)
end

end # module
