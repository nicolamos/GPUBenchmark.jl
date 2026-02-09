using GPUBenchmark
using GPUBenchmark.Core
using GPUBenchmark.CLI
using GPUBenchmark.Engine
using GPUBenchmark.Reporting
using GPUBenchmark.Visuals
using Test
using JSON
using Dates

# Include Scaling module to test its run function
include("../src/benchmarks/scaling.jl")
using .Scaling

# Include sysinfo
include("../src/benchmarks/sysinfo.jl")

@testset "GPUBenchmark.jl" begin

    @testset "Core Registry" begin
        struct TestAlg <: AbstractAlgorithm end
        register_algorithm("test_alg", TestAlg())
        @test haskey(ALGORITHM_REGISTRY, "test_alg")
        @test ALGORITHM_REGISTRY["test_alg"] isa TestAlg
        
        register_benchmark("test_bench", "Description", (args) -> Dict("ok" => true))
        @test haskey(REGISTRY, "test_bench")
    end

    @testset "CLI & Display" begin
        args = ["scaling", "--algorithm", "matmul", "--sizes", "1024,2048"]
        parsed = parse_commandline(args)
        @test parsed.benchmarks == ["scaling"]
        @test list_benchmarks() === nothing
    end

    @testset "Reporting & Visuals" begin
        mktempdir() do dir
            results = Dict(
                "timestamp" => "2026-02-09_120000",
                "cuda_functional" => false,
                "metadata" => Dict(
                    "hostname" => "test-host",
                    "cuda_runtime" => "N/A",
                    "cuda_driver" => "N/A"
                ),
                "benchmarks" => Dict(
                    "sysinfo" => Dict("gpu_name" => "Mock GPU", "vram_total" => "8GB"),
                    "matmul" => Dict("tflops" => 5.0, "matrix_size" => 1024)
                )
            )
            generate_text_report(results, dir)
            @test isfile(joinpath(dir, "summary.txt"))
            open(joinpath(dir, "metrics.json"), "w") do f JSON.print(f, results) end
            @test show_results(dir) === nothing
        end
    end

    @testset "Engine & Hardware Mocking" begin
        # Define Mock Hardware
        struct MockHardware <: AbstractHardware end
        GPUBenchmark.Core.get_devices(::MockHardware) = [0, 1, 2, 3] # 4 fake GPUs
        GPUBenchmark.Core.set_device!(::MockHardware, id::Int) = nothing
        
        @test get_devices(MockHardware()) == [0, 1, 2, 3]
        
        # Test sizing
        @test calculate_reasonable_size(0.1) >= 1024
    end

    @testset "Scaling Task Execution (Mocked)" begin
        struct MockAlg <: AbstractAlgorithm end
        GPUBenchmark.Core.run_cpu(::MockAlg, n, threads) = Dict("tflops" => 1.0, "time_s" => 0.1, "threads" => threads)
        GPUBenchmark.Core.run_gpu(::MockAlg, n) = Dict("tflops" => 2.0, "time_s" => 0.05)
        
        register_algorithm("mock_run", MockAlg())
        
        # Define Mock Hardware for this test
        struct MockHardware <: AbstractHardware end
        GPUBenchmark.Core.get_devices(::MockHardware) = [0, 1]
        GPUBenchmark.Core.set_device!(::MockHardware, id::Int) = nothing

        args = Dict(
            "algorithm" => "mock_run",
            "sizes" => "1024",
            "devices" => "all",
            "cpu_threads" => 1,
            "parallel" => false
        )
        
        # Call run_scaling with the mock hardware
        results = Scaling.run_scaling(args; hardware=MockHardware())
        
        @test results["algorithm"] == "mock_run"
        @test length(results["cpu_results"]) == 1
        @test length(results["gpu_results"]) == 2 # Should have results for both mock GPUs
        @test results["gpu_results"][0][1]["tflops"] == 2.0
    end

    @testset "Main Entry Point" begin
        original_stdout = stdout
        (rd, wr) = redirect_stdout()
        try
            @test GPUBenchmark.main(["--list"]) == 0
        finally
            redirect_stdout(original_stdout)
            close(wr)
        end
    end

end
