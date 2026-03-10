.PHONY: ffi ffi-release

ffi:
	maturin develop --manifest-path python/Cargo.toml

ffi-release:
	maturin develop --release --manifest-path python/Cargo.toml
