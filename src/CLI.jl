module CLI

using ArgParse
using Printf
using Term: Table, Panel, RenderableText
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
        "--show-latest"
            help = "Show the dashboard for the latest benchmark run and exit."
            action = :store_true
        "--show", "-S"
            help = "Show the dashboard for a specific run path and exit."
            arg_type = String
        "benchmarks"
            help = "Tasks: sysinfo (default), matmul, scaling, all."
            nargs = '*'
            default = ["sysinfo"]
    end

    parsed = parse_args(args, s, as_symbols=true)
    isnothing(parsed) && return nothing
    return NamedTuple(parsed)
end

function list_benchmarks()
    # 1. Benchmark Tasks Table
    task_names = String[]
    task_descs = String[]
    for (name, task) in sort(collect(REGISTRY), by=x->x[1])
        push!(task_names, "{bold green}$name{/bold green}")
        push!(task_descs, task.description)
    end
    
    tasks_tbl = Table(
        hcat(task_names, task_descs),
        header=["Task", "Description"],
        columns_justify=[:left, :left],
        columns_widths=[15, 45],
        box=:ROUNDED,
        style="green"
    )
    
    content = RenderableText("{bold white}📋 Registered Benchmark Tasks{/bold white}") / tasks_tbl
    
    # 2. Scaling Algorithms Table
    if !isempty(ALGORITHM_REGISTRY)
        alg_names = ["{bold blue}$name{/bold blue}" for name in sort(collect(keys(ALGORITHM_REGISTRY)))]
        algs_tbl = Table(
            reshape(alg_names, :, 1),
            header=["Algorithm"],
            columns_justify=[:left],
            columns_widths=[20],
            box=:ROUNDED,
            style="blue"
        )
        content = content / "" / RenderableText("{bold white}🧮 Registered Algorithms (for scaling){/bold white}") / algs_tbl
    end
    
    println(Panel(
        content,
        title=" {bold yellow}GPUBenchmark.jl Registry{/bold yellow} ",
        style="yellow",
        fit=true,
        padding=(2, 2, 1, 1)
    ))
end

end # module
