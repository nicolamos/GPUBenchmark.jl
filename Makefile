.PHONY: test docs clean coverage install

# Default: run tests
all: test

# Install/Setup the recommended shared HPC environment (@gpu-test)
install:
	julia -e 'using Pkg; Pkg.activate("--shared", "gpu-test"); Pkg.add(url="https://github.com/nicolamos/GPUBenchmark.jl"); Pkg.add(["GPUInspector", "CairoMakie"])'

# Run the standard test suite (Registry-compliant way)
test:
	julia --project -e 'using Pkg; Pkg.test()'

# Run tests with coverage and print summary using the dedicated test environment
coverage:
	julia --project=test --code-coverage=user test/runtests.jl
	julia --project=test -e 'using Coverage; LCOV.writefile("coverage-lcov.info", process_folder("src"))'
	@julia --project=test -e 'using Coverage; covered_lines, total_lines = get_summary(process_folder("src")); println("
📊 Coverage: ", round(covered_lines / total_lines * 100, digits=2), "%")'

# Build the documentation
docs:
	julia --project=docs docs/make.jl

# Clean up benchmark results and coverage files
clean:
	rm -rf results/*
	find . -name "*.jl.cov" -delete
	find . -name "*.jl.*.cov" -delete
	rm -f coverage-lcov.info
