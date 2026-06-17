using Documenter
using GPUBenchmark

makedocs(
    sitename = "GPUBenchmark.jl",
    format = Documenter.HTML(
        edit_link = "main",
        repolink = "https://github.com/nicolamos/GPUBenchmark.jl"
    ),
    modules = [GPUBenchmark, GPUBenchmark.Core],
    pages = [
        "Home" => "index.md",
        "Advanced Benchmarking" => "advanced.md",
        "API Reference" => "api.md"
    ]
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/nicolamos/GPUBenchmark.jl.git",
        devbranch = "main"
    )
end