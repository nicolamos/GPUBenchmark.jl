module Scaling

using ..Core: register_benchmark, ALGORITHM_REGISTRY, run_cpu, run_gpu, AbstractHardware, CUDAHardware, get_devices, set_device!
using ..Engine: calculate_scaling_sizes, calculate_scaling_sizes_cpu, calculate_scaling_sizes_gpu

function run_scaling(args; hardware::AbstractHardware = CUDAHardware())
    alg_name = get(args, "algorithm", "matmul")
    if !haskey(ALGORITHM_REGISTRY, alg_name)
        return Dict("error" => "Algorithm '$alg_name' not found.")
    end
    alg = ALGORITHM_REGISTRY[alg_name]

    no_cpu       = get(args, "no_cpu",       false)
    no_gpu       = get(args, "no_gpu",       false)
    cpu_fraction = get(args, "cpu_fraction", 0.4)
    gpu_fraction = get(args, "gpu_fraction", 0.6)

    # 1. Determine sizes
    # Manual --sizes override: same N for both (backward compat, enables direct comparison)
    sizes_str = get(args, "sizes", "")
    if !isempty(sizes_str)
        manual = parse.(Int, split(sizes_str, ","))
        cpu_sizes = no_cpu ? Int[] : manual
        gpu_sizes = no_gpu ? Int[] : manual
    else
        cpu_sizes = no_cpu ? Int[] : calculate_scaling_sizes_cpu(cpu_fraction)
        gpu_sizes = no_gpu ? Int[] : calculate_scaling_sizes_gpu(gpu_fraction)
    end

    # Comparison range: N values present in both CPU and GPU sets (direct speedup possible)
    comparison_sizes   = sort(intersect(cpu_sizes, gpu_sizes))
    # Extended GPU range: N values beyond CPU max (GPU-only, for roofline)
    cpu_max_n          = isempty(cpu_sizes) ? 0 : maximum(cpu_sizes)
    gpu_extended_sizes = sort([n for n in gpu_sizes if n > cpu_max_n])
    all_gpu_sizes      = sort(unique([comparison_sizes; gpu_extended_sizes]))

    # 2. Determine devices
    device_selection = get(args, "devices", "all")
    all_dev_ids = get_devices(hardware)
    dev_ids = if device_selection == "all"
        all_dev_ids
    else
        requested = parse.(Int, split(device_selection, ","))
        intersect(requested, all_dev_ids)
    end

    cpu_threads = get(args, "cpu_threads", Sys.CPU_THREADS)
    @info "Starting Scaling Benchmark" alg=alg_name cpu_sizes=cpu_sizes gpu_sizes=all_gpu_sizes comparison_sizes=comparison_sizes devices=dev_ids cpu_threads=cpu_threads

    results = Dict{String, Any}()
    results["algorithm"]           = alg_name
    results["cpu_sizes"]           = cpu_sizes
    results["gpu_sizes"]           = all_gpu_sizes
    results["comparison_sizes"]    = comparison_sizes
    results["gpu_extended_sizes"]  = gpu_extended_sizes
    results["cpu_results"]         = []
    results["gpu_results"]         = Dict{Int, Any}(id => [] for id in dev_ids)

    # --- CPU Loop ---
    if !no_cpu
        for n in cpu_sizes
            @info "  - Scaling CPU: n=$n"
            res = run_cpu(alg, n, cpu_threads)
            res["n"] = n
            push!(results["cpu_results"], res)
        end
    end

    # --- GPU Loop (Parallel or Sequential) ---
    if !no_gpu && !isempty(dev_ids)
        is_parallel = get(args, "parallel", false)

        for n in all_gpu_sizes
            @info "  - Scaling GPU: n=$n (Parallel: $is_parallel)"

            if is_parallel
                # Multi-GPU Parallel Probing — each @spawn gets its own task-local CUDA state
                temp_results = Vector{Any}(nothing, length(dev_ids))
                @sync for i in 1:length(dev_ids)
                    Threads.@spawn begin
                        id = dev_ids[i]
                        set_device!(hardware, id)
                        res = run_gpu(alg, n)
                        res["n"] = n
                        res["device_id"] = id
                        temp_results[i] = res
                    end
                end
                for i in 1:length(dev_ids)
                    push!(results["gpu_results"][dev_ids[i]], temp_results[i])
                end
            else
                # Sequential Probing
                for id in dev_ids
                    set_device!(hardware, id)
                    res = run_gpu(alg, n)
                    res["n"] = n
                    res["device_id"] = id
                    push!(results["gpu_results"][id], res)
                end
            end
        end
    end

    return results
end

register_benchmark(
    "scaling",
    "Benchmark algorithm scaling across problem sizes and multiple GPUs",
    run_scaling
)

end # module
