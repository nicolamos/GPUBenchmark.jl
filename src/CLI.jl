module CLI

using ArgParse
using Printf
using PrettyTables
using ..Core: REGISTRY, ALGORITHM_REGISTRY, available_cpu_threads

export parse_commandline, list_benchmarks

function parse_commandline(args)
    s = ArgParseSettings(
        description = """Julia GPU Benchmark Suite — Production Health Check

  Tasks (positional): sysinfo, matmul, tensorcore, scaling, gpuinspector, all
    sysinfo      Print CPU, GPU, and memory info (fast, no computation)
    matmul       Single FP32 matmul at one size (--size or auto)
    tensorcore   Mixed-precision Tensor Core benchmark (requires SM ≥ 7.0)
    scaling      GEMM scaling across sizes: CPU comparison + GPU extended range
    gpuinspector Full hardware telemetry via GPUInspector.jl (requires extension)
    all          Run all of the above

  Examples:
    gpubenchmark scaling
    gpubenchmark scaling --no-cpu --gpu-fraction 0.7
    gpubenchmark scaling --sizes 1024,4096,16384
    gpubenchmark matmul --size 8192
    gpubenchmark all --quiet
    gpubenchmark --show results/mynode/2026-04-12_120000""",
        autofix_names = true,
        version = string(pkgversion(parentmodule(CLI))),
        add_version = true
    )

    @add_arg_table! s begin
        "--list", "-l"
            help = "List all registered benchmark tasks and algorithms, then exit."
            action = :store_true
        "--size", "-s"
            help = "Matrix size N for the 'matmul' task. If 0, auto-scales using --fraction. Example: --size 8192"
            arg_type = Int
            default = 0
        "--sizes"
            help = "Comma-separated matrix sizes for the 'scaling' task. Overrides auto-sizing for both CPU and GPU. Example: --sizes 1024,4096,16384,32768"
            arg_type = String
            default = ""
        "--algorithm", "-a"
            help = "BLAS algorithm for the 'scaling' task. Default: matmul. Use --list to see available algorithms."
            arg_type = String
            default = "matmul"
        "--devices"
            help = "Comma-separated GPU device IDs to benchmark. 'all' uses every functional GPU. Example: --devices 0,2"
            arg_type = String
            default = "all"
        "--cpu-threads"
            help = "Number of BLAS threads for CPU benchmarks. Default: CPUs schedulable by this process ($(available_cpu_threads())). Example: --cpu-threads 8"
            arg_type = Int
            default = available_cpu_threads()
        "--no-cpu"
            help = "Skip CPU benchmark in the 'scaling' task. Useful for GPU-only characterization."
            action = :store_true
        "--no-gpu"
            help = "Skip GPU benchmark in the 'scaling' task. Useful for CPU-only baseline."
            action = :store_true
        "--cpu-fraction"
            help = "Fraction of free system RAM used to auto-size the CPU scaling range [0.0–1.0]. Default: 0.4"
            arg_type = Float64
            default = 0.4
        "--gpu-fraction"
            help = "Fraction of free GPU VRAM used to auto-size the GPU scaling range [0.0–1.0]. Default: 0.6"
            arg_type = Float64
            default = 0.6
        "--fraction", "-f"
            help = "Memory fraction for single 'matmul' auto-sizing (see --size). For scaling use --cpu-fraction / --gpu-fraction. [0.0–1.0]"
            arg_type = Float64
            default = 0.4
        "--parallel", "-p"
            help = "Run the 'scaling' GPU loop in parallel across all selected devices."
            action = :store_true
        "--plugin", "-P"
            help = "Module name or file path to load as a benchmark plugin."
            arg_type = String
        "--duration", "-d"
            help = "Stress test duration in seconds."
            arg_type = Int
            default = 30
        "--output-dir", "-o"
            help = "Root directory for results. A timestamped sub-directory is created automatically. Default: results/"
            arg_type = String
            default = "results"
        "--verbose", "-v"
            help = "Enable debug-level logging."
            action = :store_true
        "--quiet", "-q"
            help = "Suppress the terminal dashboard at the end of the run."
            action = :store_true
        "--plot-format"
            help = "File format for saved plots when GPUInspector extension is loaded (png, pdf, svg). Default: png"
            arg_type = String
            default = "png"
        "--show-latest"
            help = "Show the dashboard for the most recent benchmark run in --output-dir, then exit."
            action = :store_true
        "--show", "-S"
            help = "Show the dashboard for a specific run directory, then exit. Example: --show results/mynode/2026-04-12_120000"
            arg_type = String
        "benchmarks"
            help = "One or more tasks to run (see Tasks above). Default: sysinfo."
            nargs = '*'
            default = ["sysinfo"]
    end

    parsed = parse_args(args, s, as_symbols=true)
    isnothing(parsed) && return nothing
    return NamedTuple(parsed)
end

function list_benchmarks()
    sep = "─"^60
    println(sep)
    println("  GPUBenchmark.jl Registry")
    println(sep)

    println("\n  Registered Benchmark Tasks\n")
    task_names = String[]
    task_descs = String[]
    for (name, task) in sort(collect(REGISTRY), by=x->x[1])
        push!(task_names, name)
        push!(task_descs, task.description)
    end
    pretty_table(hcat(task_names, task_descs);
        column_labels = ["Task", "Description"],
        alignment = :l,
        table_format = TextTableFormat(borders = text_table_borders__compact),
        fit_table_in_display_vertically = false)

    if !isempty(ALGORITHM_REGISTRY)
        println("\n  Registered Algorithms (for scaling)\n")
        alg_names = sort(collect(keys(ALGORITHM_REGISTRY)))
        pretty_table(reshape(alg_names, :, 1);
            column_labels = ["Algorithm"],
            alignment = :l,
            table_format = TextTableFormat(borders = text_table_borders__compact),
            fit_table_in_display_vertically = false)
    end

    println(sep)
end

end # module
