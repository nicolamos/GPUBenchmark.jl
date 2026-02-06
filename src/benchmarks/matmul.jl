module MatMul

using CUDA
using ..GPUBenchmark: register_benchmark

function run_matmul(args)
    n = get(args, "size", 10000)
    results = Dict{String, Any}()
    
    if !CUDA.functional()
        return Dict("error" => "CUDA not functional")
    end

    dev = CUDA.device()
    results["gpu_name"] = name(dev)
    results["matrix_size"] = n
    
    # Matrix Multiplication Benchmark
    A = CUDA.rand(Float32, n, n)
    B = CUDA.rand(Float32, n, n)
    
    # Warmup
    CUDA.@sync A * B
    
    # Benchmark
    t = @elapsed CUDA.@sync A * B
    
    results["time_s"] = t
    results["tflops"] = (2.0 * n^3) / t / 1e12
    
    return results
end

# Register the benchmark
register_benchmark(
    "matmul", 
    "Standard Float32 matrix multiplication benchmark", 
    run_matmul
)

end # module
