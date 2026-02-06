module TensorCore

using CUDA
using ..GPUBenchmark: register_benchmark

function run_tensorcore(args)
    n = get(args, "size", 10000)
    results = Dict{String, Any}()
    
    if !CUDA.functional()
        return Dict("error" => "CUDA not functional")
    end

    dev = CUDA.device()
    cc = capability(dev)
    
    if cc < v"7.0"
        return Dict("error" => "Compute Capability $(cc) does not support Tensor Cores (7.0+ required)")
    end

    results["gpu_name"] = name(dev)
    results["compute_capability"] = string(cc)
    results["matrix_size"] = n
    
    # Tensor Core Benchmark (Mixed Precision: FP16 input, FP32 accumulate)
    # CUDA.jl uses cublasGemmEx internally which leverages Tensor Cores 
    # when using Float16 inputs.
    A = CUDA.rand(Float16, n, n)
    B = CUDA.rand(Float16, n, n)
    
    # Warmup
    CUDA.@sync A * B
    
    # Benchmark
    t = @elapsed CUDA.@sync begin
        C = A * B
        CUDA.unsafe_free!(C)
    end
    
    # Explicitly free input matrices
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    
    results["time_s"] = t
    results["tflops"] = (2.0 * n^3) / t / 1e12
    
    return results
end

# Register the benchmark
register_benchmark(
    "tensorcore", 
    "Mixed-precision Tensor Core benchmark (FP16/FP32)", 
    run_tensorcore
)

end # module
