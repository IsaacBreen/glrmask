.PHONY: ffi ffi-release runtime-ffi runtime-ffi-release show-grammar-glrmask

ffi:
	maturin develop --manifest-path python/Cargo.toml

ffi-release:
	maturin develop --release --manifest-path python/Cargo.toml

runtime-ffi:
	maturin develop --manifest-path glrmask-runtime/python/Cargo.toml

runtime-ffi-release:
	maturin develop --release --manifest-path glrmask-runtime/python/Cargo.toml

# Dump a JSON schema grammar in the GLRM format.
# Usage: make show-grammar-glrmask SCHEMA='{"type":"string"}'
show-grammar-glrmask:
	@cargo run --quiet --example show_grammar_glrmask -- '$(SCHEMA)'
