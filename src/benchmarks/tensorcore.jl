module TensorCore

using CUDA
using Statistics: median
import ..Core: run_cpu, run_gpu
using ..Core: register_benchmark, register_algorithm, AbstractAlgorithm

struct TensorCoreAlg <: AbstractAlgorithm end

function run_cpu(::TensorCoreAlg, n, threads)
    return Dict("error" => "Tensor Cores are not available on CPU.")
end

function run_gpu(::TensorCoreAlg, n)
    if !CUDA.functional()
        return Dict("error" => "CUDA not functional")
    end

    dev = CUDA.device()
    cc = capability(dev)

    if cc < v"7.0"
        return Dict("error" => "Compute Capability $(cc) does not support Tensor Cores (7.0+ required)")
    end

    # Tensor Core Benchmark (Mixed Precision: FP16 input, FP32 accumulate)
    A = CUDA.rand(Float16, n, n)
    B = CUDA.rand(Float16, n, n)

    # Warmup
    C_warmup = CUDA.@sync A * B
    CUDA.unsafe_free!(C_warmup)

    # Benchmark: 5 iterations, take median
    times = Float64[]
    for _ in 1:5
        t = @elapsed CUDA.@sync begin
            C = A * B
        end
        CUDA.unsafe_free!(C)
        push!(times, t)
    end
    t = median(times)

    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)

    return Dict(
        "time_s" => t,
        "tflops" => (2.0 * n^3) / t / 1e12,
        "mode" => "gpu",
        "precision" => "mixed (FP16/FP32)",
        "matrix_size" => n
    )
end

# Legacy task entry point
function run_tensorcore_task(args)
    n = get(args, "size", 10000)
    return run_gpu(TensorCoreAlg(), n)
end

register_benchmark(
    "tensorcore",
    "Mixed-precision Tensor Core benchmark (FP16/FP32)",
    run_tensorcore_task
)

register_algorithm("tensorcore", TensorCoreAlg())

end # module
