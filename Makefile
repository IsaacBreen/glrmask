.PHONY: ffi ffi-release runtime-ffi runtime-ffi-release bench-cfa-build bench-cfa-build-mt cargo cargo-isolated test check clippy

MACHINE_GATE ?= python3 scripts/machine_gate.py
CARGO_ARGS ?=

BENCH_CASE_ENV = $(if $(CASE),GLRMASK_BENCH_CASE='$(CASE)',$(error set CASE, e.g. `make $@ CASE=github_trivial_o20469`))
BENCH_PROFILE_ENV = $(if $(PROFILE),GLRMASK_BENCH_PROFILE='$(PROFILE)')

ffi:
	$(MACHINE_GATE) shared -- maturin develop --release --manifest-path python/Cargo.toml

ffi-release:
	$(MACHINE_GATE) shared -- maturin develop --release --manifest-path python/Cargo.toml

runtime-ffi:
	$(MACHINE_GATE) shared -- maturin develop --manifest-path glrmask-runtime/python/Cargo.toml

runtime-ffi-release:
	$(MACHINE_GATE) shared -- maturin develop --release --manifest-path glrmask-runtime/python/Cargo.toml

cargo:
	$(MACHINE_GATE) shared -- cargo $(CARGO_ARGS)

cargo-isolated:
	$(MACHINE_GATE) isolated -- cargo $(CARGO_ARGS)

test:
	$(MACHINE_GATE) shared -- cargo test $(CARGO_ARGS)

check:
	$(MACHINE_GATE) shared -- cargo check $(CARGO_ARGS)

clippy:
	$(MACHINE_GATE) shared -- cargo clippy $(CARGO_ARGS)

bench-cfa-build:
	$(MACHINE_GATE) isolated -- env $(BENCH_CASE_ENV) $(BENCH_PROFILE_ENV) cargo bench --bench cfa_sweep_schema_build_single_threaded

bench-cfa-build-mt:
	$(MACHINE_GATE) isolated -- env $(BENCH_CASE_ENV) $(BENCH_PROFILE_ENV) cargo bench --bench cfa_sweep_schema_build_multithreaded
