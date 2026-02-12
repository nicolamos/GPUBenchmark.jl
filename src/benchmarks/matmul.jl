module MatMul

using CUDA
using BenchmarkTools
using Statistics: mean
using LinearAlgebra
using LinearAlgebra: mul!

import ..Core: run_cpu, run_gpu
using ..Core: register_benchmark, register_algorithm, AbstractAlgorithm

struct MatMulAlg <: AbstractAlgorithm end

function run_cpu(::MatMulAlg, n, threads)
    # Configure BLAS threads for this run
    LinearAlgebra.BLAS.set_num_threads(threads)

    A = rand(Float32, n, n)
    B = rand(Float32, n, n)
    C = zeros(Float32, n, n)

    trial = @benchmark mul!($C, $A, $B)
    t_min = minimum(trial.times) / 1e9
    t_mean = mean(trial.times) / 1e9
    t_max = maximum(trial.times) / 1e9

    return Dict(
        "time_s" => t_mean,
        "min_time" => t_min,
        "max_time" => t_max,
        "tflops" => (2.0 * n^3) / t_min / 1e12,
        "samples" => trial.times ./ 1e9,
        "mode" => "cpu",
        "threads" => threads,
        "matrix_size" => n
    )
end

function run_gpu(::MatMulAlg, n)
    if !CUDA.functional()
        return Dict("error" => "CUDA not functional")
    end

    A = CUDA.rand(Float32, n, n)
    B = CUDA.rand(Float32, n, n)
    C = CUDA.zeros(Float32, n, n)

    trial = @benchmark CUDA.@sync mul!($C, $A, $B)
    t_min = minimum(trial.times) / 1e9
    t_mean = mean(trial.times) / 1e9
    t_max = maximum(trial.times) / 1e9

    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)

    return Dict(
        "time_s" => t_mean,
        "min_time" => t_min,
        "max_time" => t_max,
        "tflops" => (2.0 * n^3) / t_min / 1e12,
        "samples" => trial.times ./ 1e9,
        "mode" => "gpu",
        "matrix_size" => n
    )
end

# Legacy task entry point
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
