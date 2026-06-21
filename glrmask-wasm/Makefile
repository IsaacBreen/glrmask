.PHONY: web clean-web demo-artifact

# Raw self-contained WebAssembly: no wasm-bindgen-generated JavaScript glue.
web:
	RUSTFLAGS='-C target-feature=-reference-types' cargo build --release --target wasm32-unknown-unknown

clean-web:
	cargo clean --target wasm32-unknown-unknown

# TOKENIZER must be the tokenizer.json shipped with the target browser model.
# Example:
# make demo-artifact TOKENIZER=/tmp/tinystories-1m-tokenizer.json OUT=/tmp/tinystories-story.glrmaskc
demo-artifact:
	cargo run -p glrmask-browser-artifact --release -- $(TOKENIZER) $(OUT)
