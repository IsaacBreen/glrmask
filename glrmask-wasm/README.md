# glrmask-wasm

Thin raw WebAssembly wrapper around `glrmask-runtime`. It exports no compiler or
schema importer and requires no generated JavaScript glue.

Build it from this directory:

```sh
make web
```

The resulting binary is:

```text
../target/wasm32-unknown-unknown/release/glrmask_wasm.wasm
```

## JavaScript ABI

The module exports linear memory plus these functions:

```text
glrmask_alloc(length) -> pointer
glrmask_dealloc(pointer, length)
glrmask_constraint_load(artifact_pointer, artifact_length) -> handle | 0
glrmask_constraint_free(handle)
glrmask_session_new_from_constraint(constraint_handle) -> handle | 0
glrmask_session_new(artifact_pointer, artifact_length) -> handle | 0
glrmask_session_free(handle)
glrmask_mask(handle) -> pointer
glrmask_mask_len(handle) -> u32 words
glrmask_commit(handle, token_id) -> 1 | 0
glrmask_is_finished(handle) -> 1 | 0
glrmask_reset(handle) -> 1 | 0
glrmask_last_error_ptr() -> pointer
glrmask_last_error_len() -> bytes
```

`glrmask_dealloc` must receive the same byte length passed to `glrmask_alloc`.
For several streams using the same grammar, load the artifact once with
`glrmask_constraint_load`, create one session per stream with
`glrmask_session_new_from_constraint`, then release the public constraint handle.
Existing sessions retain their shared executor. `glrmask_session_new` remains the
one-shot convenience form for a single stream.

`glrmask_mask` returns a pointer into WASM memory. Its packed `u32` buffer is
allocated once when the session is created and reused for subsequent mask calls.
Copy the `Uint32Array` before calling another runtime operation. The bit layout
is original vocabulary ID space:
token `id` is admitted when:

```js
const allowed = (maskWords[id >>> 5] & (1 << (id & 31))) !== 0;
```

The browser must apply the mask to logits before sampling `id`, then call
`glrmask_commit(handle, id)` after sampling. This is exact whole-token constrained
decoding; it is not a byte-level proxy.
