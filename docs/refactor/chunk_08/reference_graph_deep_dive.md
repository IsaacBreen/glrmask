# Reference graph deep dive
References are graph edges between schema locations.  Loading should construct
or make discoverable a local reference graph; lowering should allocate grammar
rules for recursive references.  These responsibilities must not mix.  A future
`resolve/` or `load/references.rs` should expose: normalized pointers, alias
map, definition targets, non-definition local targets, and recursion detection
utilities.
