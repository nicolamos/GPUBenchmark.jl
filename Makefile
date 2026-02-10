.PHONY: test docs clean coverage

# Default: run tests
all: test

# Run the standard test suite (Registry-compliant way)
test:
	julia --project -e 'using Pkg; Pkg.test()'

# Run tests with coverage (tracefile: no .cov files polluting src/)
coverage:
	@rm -f coverage.info
	julia --project=test --code-coverage=@src --code-coverage=coverage.info test/runtests.jl
	@julia --project=test -e ' \
		using Coverage; \
		cov = LCOV.readfile("coverage.info"); \
		covered, total = get_summary(cov); \
		pct = round(covered / total * 100, digits=1); \
		println("\n📊 Coverage: $$covered / $$total lines ($$pct%)\n"); \
		for fc in sort(cov, by=c->c.filename); \
			fc_cov, fc_tot = get_summary([fc]); \
			fc_pct = fc_tot > 0 ? round(fc_cov / fc_tot * 100, digits=1) : 0.0; \
			name = replace(fc.filename, r".*/src/" => ""); \
			println("  ", rpad(name, 30), lpad("$$fc_cov/$$fc_tot", 10), "  ($$fc_pct%)"); \
		end; \
		println()'

# Build the documentation
docs:
	julia --project=docs docs/make.jl

# Clean up benchmark results and coverage files
clean:
	rm -rf results/*
	rm -f coverage.info
