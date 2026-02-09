using Documenter
using GPUBenchmark

makedocs(
    sitename = "GPUBenchmark.jl",
    format = Documenter.HTML(),
    modules = [GPUBenchmark],
    pages = [
        "Home" => "index.md",
        "Advanced Benchmarking" => "advanced.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/nicolamos/GPUBenchmark.jl.git",
)
