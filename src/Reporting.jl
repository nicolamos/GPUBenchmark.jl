module Reporting

using Printf
using Dates
using JSON
using Statistics: mean
using ..Core

export generate_text_report, export_scaling_dat

function generate_text_report(results, output_dir)
    filename = joinpath(output_dir, "summary.txt")
    open(filename, "w") do io
        println(io, "================================================================")
        println(io, "           GPU BENCHMARK ANALYSIS SUMMARY - $(results["timestamp"])")
        println(io, "================================================================")
        println(io, "")
        println(io, "[1. NODE ENVIRONMENT]")
        
        metadata = get(results, "metadata", Dict())
        println(io, "  Hostname:      $(get(metadata, "hostname", "unknown"))")
        println(io, "  Julia:         v$(VERSION)")
        
        if get(results, "cuda_functional", false)
            println(io, "  CUDA Runtime:  $(get(metadata, "cuda_runtime", "unknown"))")
            println(io, "  CUDA Driver:   $(get(metadata, "cuda_driver", "unknown"))")
        else
            println(io, "  CUDA:          ⚠️ NOT FUNCTIONAL")
        end
        
        println(io, "")
        println(io, "[2. HARDWARE CAPABILITY]")
        benchmarks = get(results, "benchmarks", Dict())
        if haskey(benchmarks, "sysinfo")
            si = benchmarks["sysinfo"]
            if !haskey(si, "error")
                # CPU / System
                println(io, "  CPU Model:     $(get(si, "cpu_model", "unknown"))")
                println(io, "  CPU Threads:   $(get(si, "cpu_threads", "unknown")) ($(get(si, "cpu_arch", "unknown")))")
                println(io, "  Total RAM:     $(get(si, "ram_total", "unknown"))")
                
                # GPU
                println(io, "  GPU Model:     $(get(si, "gpu_name", "unknown"))")
                println(io, "  Compute Cap:   $(get(si, "compute_capability", "unknown"))")
                println(io, "  Total VRAM:    $(get(si, "vram_total", "unknown"))")
                if get(si, "gpu_count", 1) > 1
                    println(io, "  GPU Count:     $(si["gpu_count"])")
                end
            end
        end

        println(io, "")
        println(io, "[3. PERFORMANCE & EFFICIENCY]")
        
        if haskey(benchmarks, "bandwidth")
            bw = benchmarks["bandwidth"]
            if !haskey(bw, "error")
                haskey(bw, "gpu_bandwidth_gibs") &&
                    @printf(io, "  BW GPU (STREAM): %.1f GiB/s\n", bw["gpu_bandwidth_gibs"])
                haskey(bw, "cpu_bandwidth_gibs") &&
                    @printf(io, "  BW CPU (STREAM): %.1f GiB/s\n", bw["cpu_bandwidth_gibs"])
            end
        end

        if haskey(benchmarks, "matmul")
            m = benchmarks["matmul"]
            if !haskey(m, "error")
                @printf(io, "  FP32 Compute:  %.2f TFLOPS\n", get(m, "tflops", 0.0))
            end
        end

        if haskey(benchmarks, "scaling")
            s = benchmarks["scaling"]
            if !haskey(s, "error")
                @printf(io, "  Peak (GPU):    %.2f TFLOPS\n", get(s, "peak_gpu_tflops", 0.0))
                @printf(io, "  Peak (CPU):    %.2f TFLOPS\n", get(s, "peak_cpu_tflops", 0.0))

                # CPU vs GPU comparison table (shared N values only)
                cpu_results = get(s, "cpu_results", [])
                gpu_results = get(s, "gpu_results", Dict())
                if !isempty(cpu_results) && !isempty(gpu_results)
                    cpu_by_n = Dict(r["n"] => r["tflops"] for r in cpu_results)
                    # Max GPU TFLOPS at each N across all devices
                    gpu_by_n = Dict{Int, Float64}()
                    for (_, runs) in gpu_results
                        for r in runs
                            n, t = r["n"], r["tflops"]
                            gpu_by_n[n] = max(get(gpu_by_n, n, 0.0), t)
                        end
                    end
                    shared_ns = sort(collect(intersect(keys(cpu_by_n), keys(gpu_by_n))))

                    if !isempty(shared_ns)
                        speedups = [gpu_by_n[n] / cpu_by_n[n] for n in shared_ns]
                        peak_sp  = maximum(speedups)
                        xover    = findfirst(sp -> sp > 1.0, speedups)

                        @printf(io, "  Peak Speedup:  %.1f× (GPU/CPU)\n", peak_sp)
                        !isnothing(xover) && @printf(io, "  Crossover N:   %d\n", shared_ns[xover])

                        println(io, "")
                        println(io, "[3b. CPU vs GPU COMPARISON]")
                        @printf(io, "  %-10s  %-12s  %-12s  %s\n", "N", "CPU TFLOPS", "GPU TFLOPS", "Speedup")
                        println(io, "  " * "─"^50)
                        for (n, sp) in zip(shared_ns, speedups)
                            @printf(io, "  %-10d  %-12.3f  %-12.3f  %.1f×\n",
                                n, cpu_by_n[n], gpu_by_n[n], sp)
                        end
                        gpu_extended = get(s, "gpu_extended_sizes", Int[])
                        gpu_ext_data = [n for n in gpu_extended if haskey(gpu_by_n, n)]
                        if !isempty(gpu_ext_data)
                            println(io, "  " * "─"^50)
                            println(io, "  GPU extended (no CPU baseline):")
                            for n in gpu_ext_data
                                @printf(io, "  %-10d  %-12s  %-12.3f  —\n", n, "—", gpu_by_n[n])
                            end
                        end
                    end
                end
            end
        end

        if haskey(benchmarks, "gpuinspector")
            r = benchmarks["gpuinspector"]
            if !haskey(r, "error")
                if haskey(r, "memory_bandwidth")
                    @printf(io, "  Memory BW:     %.2f GiB/s\n", get(r, "memory_bandwidth", 0.0))
                end
                
                # Power Efficiency
                if haskey(benchmarks, "scaling") && haskey(r, "avg_power_w")
                    peak_gpu = get(benchmarks["scaling"], "peak_gpu_tflops", 0.0)
                    avg_pwr = r["avg_power_w"]
                    if avg_pwr > 0
                        eff = (peak_gpu * 1000) / avg_pwr # GFLOPS/W
                        @printf(io, "  Energy Eff:    %.2f GFLOPS/Watt\n", eff)
                    end
                end
            end
        end

        println(io, "")
        println(io, "[4. INTERPRETATION GUIDE]")
        println(io, "  * Roofline: Points on the slope are Bandwidth-Bound (limited by memory).")
        println(io, "              Points on the flat ceiling are Compute-Bound (limited by cores).")
        println(io, "  * Efficiency: GFLOPS/Watt measures how much work is done per unit of energy.")
        println(io, "                Higher values indicate better sustainability and lower heat.")
        
        println(io, "")
        println(io, "================================================================")
        println(io, "Generated by GPUBenchmark.jl Suite")
    end
end

function export_scaling_dat(scaling_results, run_dir)
    dat_file = joinpath(run_dir, "scaling.dat")
    open(dat_file, "w") do io
        println(io, "# GPUBenchmark Scaling Data")
        println(io, "# Algorithm: ", get(scaling_results, "algorithm", "unknown"))
        println(io, "# N\tMode\tID/Threads\tTime(s)\tTFLOPS\tMinTime\tMaxTime")
        
        # CPU results
        cpu_res = get(scaling_results, "cpu_results", [])
        for res in cpu_res
            @printf(io, "%d\tCPU\t%d\t%.6f\t%.4f\t%.6f\t%.6f\n", 
                get(res, "n", 0), get(res, "threads", 0), 
                get(res, "time_s", 0.0), get(res, "tflops", 0.0),
                get(res, "min_time", 0.0), get(res, "max_time", 0.0))
        end
        
        # GPU results
        gpu_res = get(scaling_results, "gpu_results", Dict())
        for (id, runs) in gpu_res
            for res in runs
                @printf(io, "%d\tGPU\t%d\t%.6f\t%.4f\t%.6f\t%.6f\n", 
                    get(res, "n", 0), id, 
                    get(res, "time_s", 0.0), get(res, "tflops", 0.0),
                    get(res, "min_time", 0.0), get(res, "max_time", 0.0))
            end
        end
    end
end

end # module