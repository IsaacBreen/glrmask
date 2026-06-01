# Performance model

Commit performance is driven by four quantities:

1. number of live tokenizer states,
2. number of normalized matches per scanner execution,
3. cost of parser stack-effect advance,
4. cost of merging/fusing GSS frontiers.

The source split reflects those costs:

- `acceptance.rs` controls match-list size;
- `parser_advance.rs` and `single_top.rs` control stack-effect cost;
- `queue.rs` controls frontier merge/fuse cost;
- `fast_path.rs` bypasses the general queue when the state shape is common and simple;
- `profiled.rs` measures the phases without changing the reference implementation.

The publication goal is not to hide optimization; it is to make optimization read as denotation-preserving specialization.
