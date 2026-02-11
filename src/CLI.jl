module CLI

using ArgParse
using Printf
using PrettyTables
using ..Core

export parse_commandline, list_benchmarks

function parse_commandline(args)
    s = ArgParseSettings(
        description = "Julia GPU Benchmark Suite - Production Health Check",
        autofix_names = true,
        version = string(pkgversion(parentmodule(CLI))),
        add_version = true
    )

    @add_arg_table! s begin
        "--list", "-l"
            help = "List available benchmarks and algorithms"
            action = :store_true
        "--size", "-s"
            help = "Global matrix size (N). If 0, auto-scales to ~40% VRAM."
            arg_type = Int
            default = 0
        "--sizes"
            help = "Comma-separated list of sizes for scaling benchmark. If empty, auto-scales."
            arg_type = String
            default = ""
        "--algorithm", "-a"
            help = "Algorithm to use for scaling benchmark."
            arg_type = String
            default = "matmul"
        "--devices"
            help = "Comma-separated GPU IDs to use (e.g. '0,1'). 'all' uses all functional GPUs."
            arg_type = String
            default = "all"
        "--cpu-threads"
            help = "Number of CPU threads to use for CPU benchmarks."
            arg_type = Int
            default = Sys.CPU_THREADS
        "--parallel", "-p"
            help = "Run scaling benchmark in parallel across specified GPUs."
            action = :store_true
        "--plugin", "-P"
            help = "Module name or file path to load as a plugin."
            arg_type = String
        "--duration", "-d"
            help = "Stress test duration (seconds)."
            arg_type = Int
            default = 30
        "--fraction", "-f"
            help = "Target VRAM usage fraction for auto-sizing."
            arg_type = Float64
            default = 0.4
        "--output-dir", "-o"
            help = "Root directory for results."
            arg_type = String
            default = "results"
        "--verbose", "-v"
            help = "Enable debug logging."
            action = :store_true
        "--quiet", "-q"
            help = "Suppress terminal dashboard."
            action = :store_true
        "--plot-format"
            help = "Format for saved plots (png, pdf, svg)."
            arg_type = String
            default = "png"
        "--show-latest"
            help = "Show the dashboard for the latest benchmark run and exit."
            action = :store_true
        "--show", "-S"
            help = "Show the dashboard for a specific run path and exit."
            arg_type = String
        "benchmarks"
            help = "Tasks: sysinfo, matmul, tensorcore, scaling, gpuinspector, all. Default: sysinfo."
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
