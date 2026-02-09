using Documenter
using GPUBenchmark

makedocs(
    sitename = "GPUBenchmark.jl",
    format = Documenter.HTML(),
    modules = [GPUBenchmark, GPUBenchmark.Core],
    pages = [
        "Home" => "index.md",
        "Advanced Benchmarking" => "advanced.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/nicolamos/GPUBenchmark.jl.git",
)