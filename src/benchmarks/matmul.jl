module MatMul

using CUDA
using BenchmarkTools
using LinearAlgebra
import ..GPUBenchmark.Core: run_cpu, run_gpu
using ..GPUBenchmark.Core: register_benchmark, register_algorithm, AbstractAlgorithm

struct MatMulAlg <: AbstractAlgorithm end

function run_cpu(::MatMulAlg, n, threads)
    # Configure BLAS threads for this run
    LinearAlgebra.BLAS.set_num_threads(threads)
    
    A = rand(Float32, n, n)
    B = rand(Float32, n, n)
    
    # Use BenchmarkTools for accurate CPU timing
    t = @belapsed begin
        $A * $B
    end
    
    return Dict(
        "time_s" => t,
        "tflops" => (2.0 * n^3) / t / 1e12,
        "mode" => "cpu",
        "threads" => threads
    )
end

function run_gpu(::MatMulAlg, n)
    if !CUDA.functional()
        return Dict("error" => "CUDA not functional")
    end

    A = CUDA.rand(Float32, n, n)
    B = CUDA.rand(Float32, n, n)
    
    # Warmup
    CUDA.@sync A * B
    
    # Benchmark
    t = @elapsed CUDA.@sync begin
        C = A * B
        CUDA.unsafe_free!(C)
    end
    
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    
    return Dict(
        "time_s" => t,
        "tflops" => (2.0 * n^3) / t / 1e12,
        "mode" => "gpu"
    )
end

# Keep the legacy task
function run_matmul_task(args)
    n = get(args, "size", 10000)
    return run_gpu(MatMulAlg(), n)
end

register_benchmark(
    "matmul", 
    "Standard Float32 matrix multiplication benchmark", 
    run_matmul_task
)

register_algorithm("matmul", MatMulAlg())

end # module