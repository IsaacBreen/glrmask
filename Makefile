.PHONY: ffi ffi-release runtime-ffi runtime-ffi-release show-grammar-glrmask

ffi:
	maturin develop --manifest-path python/Cargo.toml

ffi-release:
	maturin develop --release --manifest-path python/Cargo.toml

runtime-ffi:
	maturin develop --manifest-path glrmask-runtime/python/Cargo.toml

runtime-ffi-release:
	maturin develop --release --manifest-path glrmask-runtime/python/Cargo.toml
