module Bandwidth

using CUDA
using BenchmarkTools
using Printf

using ..Core: register_benchmark

# STREAM triad: A[i] = B[i] + s * C[i]
# Bytes accessed: 2 reads (B, C) + 1 write (A) = 3 × N × sizeof(Float32)
# Arithmetic intensity: 2 FLOP / 12 bytes ≈ 0.17 FLOP/Byte → always memory-bound

# Arrays must be large enough to overflow L2 cache (up to ~50 MB on H100).
# We cap at 256M elements (1 GiB per array) to keep runtime short.
const _MAX_ELEMENTS = 256 * 1024 * 1024

function _bw_n_gpu()
    !CUDA.functional() && return _MAX_ELEMENTS
    # Use at most 25% of free VRAM split across 3 arrays
    n = Int(CUDA.available_memory()) ÷ (4 * 3 * sizeof(Float32))
    return clamp(n, 32 * 1024 * 1024, _MAX_ELEMENTS)
end

function _bw_n_cpu()
    # Use at most 20% of free RAM split across 3 arrays
    n = Int(Sys.free_memory()) ÷ (5 * 3 * sizeof(Float32))
    return clamp(n, 8 * 1024 * 1024, _MAX_ELEMENTS ÷ 2)
end

function run_bandwidth(args)
    results = Dict{String, Any}()

    # --- GPU STREAM ---
    if CUDA.functional()
        n   = _bw_n_gpu()
        A   = CUDA.zeros(Float32, n)
        B   = CUDA.rand(Float32, n)
        C   = CUDA.rand(Float32, n)
        s   = 2.0f0

        CUDA.@sync @. A = B + s * C  # warm-up

        trial  = @benchmark CUDA.@sync @. $A = $B + $s * $C  samples=20 evals=1
        t_min  = minimum(trial.times) / 1e9           # seconds
        bytes  = 3 * n * sizeof(Float32)
        # Store in GiB/s (consistent with GPUInspector and the roofline formula)
        bw_gibs = bytes / t_min / (1024.0^3)

        CUDA.unsafe_free!(A); CUDA.unsafe_free!(B); CUDA.unsafe_free!(C)

        results["gpu_bandwidth_gibs"] = bw_gibs
        results["gpu_n_elements"]     = n
        @info "  - GPU STREAM bandwidth: $(@sprintf("%.1f", bw_gibs)) GiB/s (N=$(n÷1024^2)M)"
    end

    # --- CPU STREAM ---
    n_c = _bw_n_cpu()
    A_c = zeros(Float32, n_c)
    B_c = rand(Float32, n_c)
    C_c = rand(Float32, n_c)
    s_c = 2.0f0

    @. A_c = B_c + s_c * C_c  # warm-up

    trial_c = @benchmark @. $A_c = $B_c + $s_c * $C_c  samples=10 evals=1
    t_min_c  = minimum(trial_c.times) / 1e9
    bytes_c  = 3 * n_c * sizeof(Float32)
    bw_cpu   = bytes_c / t_min_c / (1024.0^3)

    results["cpu_bandwidth_gibs"] = bw_cpu
    results["cpu_n_elements"]     = n_c
    @info "  - CPU STREAM bandwidth: $(@sprintf("%.1f", bw_cpu)) GiB/s (N=$(n_c÷1024^2)M)"

    return results
end

register_benchmark(
    "bandwidth",
    "STREAM triad memory bandwidth (CPU + GPU), used by the roofline model",
    run_bandwidth
)

end # module
