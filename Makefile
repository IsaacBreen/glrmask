.PHONY: ffi ffi-release bench-cfa-build bench-cfa-build-mt

BENCH_CASE_ENV = $(if $(CASE),GLRMASK_BENCH_CASE='$(CASE)',$(error set CASE, e.g. `make $@ CASE=github_trivial_o20469`))
BENCH_PROFILE_ENV = $(if $(PROFILE),GLRMASK_BENCH_PROFILE='$(PROFILE)')

ffi:
	maturin develop --release --manifest-path python/Cargo.toml

ffi-release:
	maturin develop --release --manifest-path python/Cargo.toml

bench-cfa-build:
	$(BENCH_CASE_ENV) $(BENCH_PROFILE_ENV) cargo bench --bench cfa_sweep_schema_build_single_threaded

bench-cfa-build-mt:
	$(BENCH_CASE_ENV) $(BENCH_PROFILE_ENV) cargo bench --bench cfa_sweep_schema_build_multithreaded
