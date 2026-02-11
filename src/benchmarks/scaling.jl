module Scaling

using ..Core: register_benchmark, ALGORITHM_REGISTRY, run_cpu, run_gpu, AbstractHardware, CUDAHardware, get_devices, set_device!
using ..Engine: calculate_scaling_sizes

function run_scaling(args; hardware::AbstractHardware = CUDAHardware())
    alg_name = get(args, "algorithm", "matmul")
    if !haskey(ALGORITHM_REGISTRY, alg_name)
        return Dict("error" => "Algorithm '$alg_name' not found.")
    end
    alg = ALGORITHM_REGISTRY[alg_name]
    
    # 1. Determine sizes
    sizes_str = get(args, "sizes", "")
    sizes = isempty(sizes_str) ? calculate_scaling_sizes() : parse.(Int, split(sizes_str, ","))
    
    # 2. Determine devices via abstraction
    device_selection = get(args, "devices", "all")
    all_dev_ids = get_devices(hardware)
    
    dev_ids = if device_selection == "all"
        all_dev_ids
    else
        # Filter requested IDs against available IDs
        requested = parse.(Int, split(device_selection, ","))
        intersect(requested, all_dev_ids)
    end
    
    # 3. CPU execution
    cpu_threads = get(args, "cpu_threads", Sys.CPU_THREADS)
    @info "Starting Scaling Benchmark" alg=alg_name sizes=sizes devices=dev_ids cpu_threads=cpu_threads
    
    results = Dict{String, Any}()
    results["algorithm"] = alg_name
    results["sizes"] = sizes
    results["cpu_results"] = []
    results["gpu_results"] = Dict{Int, Any}(id => [] for id in dev_ids)
    
    # --- CPU Loop ---
    for n in sizes
        @info "  - Scaling CPU: n=$n"
        res = run_cpu(alg, n, cpu_threads)
        res["n"] = n
        push!(results["cpu_results"], res)
    end
    
    # --- GPU Loop (Parallel or Sequential) ---
    if !isempty(dev_ids)
        is_parallel = get(args, "parallel", false)
        
        for n in sizes
            @info "  - Scaling GPU: n=$n (Parallel: $is_parallel)"
            
            if is_parallel
                # Multi-GPU Parallel Probing
                temp_results = Vector{Any}(nothing, length(dev_ids))
                Threads.@threads for i in 1:length(dev_ids)
                    id = dev_ids[i]
                    set_device!(hardware, id)
                    res = run_gpu(alg, n)
                    res["n"] = n
                    res["device_id"] = id
                    temp_results[i] = res
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
