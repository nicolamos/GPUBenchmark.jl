module Core

using CUDA

export AbstractAlgorithm, BenchmarkTask, REGISTRY, ALGORITHM_REGISTRY
export register_benchmark, register_algorithm, run_cpu, run_gpu
export AbstractHardware, CUDAHardware, get_devices, set_device!
export available_cpu_threads

abstract type AbstractAlgorithm end

"""
    available_cpu_threads()
Number of CPUs actually schedulable by this process. `Sys.CPU_THREADS` reports
every online CPU on the machine, ignoring cgroup/cpuset affinity restrictions
(e.g. under Slurm), which causes massive thread oversubscription. This reads
the process affinity mask on Linux, the same source `nproc` uses, and falls
back to `Sys.CPU_THREADS` elsewhere.
"""
function available_cpu_threads()
    if Sys.islinux()
        m = match(r"Cpus_allowed:\s*([0-9a-fA-F,]+)", read("/proc/self/status", String))
        if m !== nothing
            hexmask = replace(m.captures[1], "," => "")
            return count_ones(parse(BigInt, hexmask, base=16))
        end
    end
    return Sys.CPU_THREADS
end

# --- Hardware Abstraction ---

abstract type AbstractHardware end

struct CUDAHardware <: AbstractHardware end

"""
    get_devices(hw::AbstractHardware)
Return a list of available device IDs.
"""
function get_devices(::CUDAHardware)
    if CUDA.functional()
        return collect(0:length(CUDA.devices())-1)
    else
        return Int[]
    end
end

"""
    set_device!(hw::AbstractHardware, id::Int)
Switch context to the specified device.
"""
function set_device!(::CUDAHardware, id::Int)
    CUDA.device!(id)
end

# --- Algorithm Interface ---

"""
    run_cpu(alg, n, threads)
Implement CPU execution path for an algorithm.
"""
function run_cpu end

"""
    run_gpu(alg, n)
Implement GPU execution path for an algorithm.
"""
function run_gpu end

struct BenchmarkTask
    name::String
    description::String
    run_func::Function
end

const REGISTRY = Dict{String, BenchmarkTask}()
const ALGORITHM_REGISTRY = Dict{String, AbstractAlgorithm}()

"""
    register_benchmark(name, description, run_func)
Register a new benchmark task.
"""
function register_benchmark(name::String, description::String, run_func::Function)
    REGISTRY[name] = BenchmarkTask(name, description, run_func)
end

"""
    register_algorithm(name, alg)
Register a new algorithm for scaling benchmarks.
"""
function register_algorithm(name::String, alg::AbstractAlgorithm)
    ALGORITHM_REGISTRY[name] = alg
end

end # module
