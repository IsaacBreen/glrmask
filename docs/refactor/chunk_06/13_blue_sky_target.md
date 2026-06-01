# Blue-sky target after this chunk

Chunk 06 is still incremental relative to the ideal architecture.  The ideal
future shape is cleaner:

```text
scan/
  relation.rs          // Scan, Completed, Partial, CanMatch vocabulary
  execute.rs           // one-fragment scanning
  trie.rs              // byte trie independent of compile/runtime

compile/
  scan_relation/
    collect.rs         // Q × trie -> grouped intervals
    quotient.rs        // token quotient by CanMatch signatures
    materialize.rs     // grouped intervals -> weights
    validate.rs        // slow oracle
    options.rs         // typed config
    profile.rs         // reporting
    build.rs           // phase orchestration
```

In that target, `VocabPrefixTree` may move out of `ds` into `scan`, because it is
not a generic data structure: it is the byte-domain representation of a token
vocabulary.  But that move should happen only after runtime and compile users are
both stabilized.

The deeper mathematical target is to state all compiled objects as quotient
relations:

- Terminal DWA quotient: equality of completed terminal-sequence behavior.
- Scan-relation quotient: equality of future-completion behavior from partial
  lexer states.
- Parser DWA quotient: equality of parser stack-prefix admissibility.
- Final shared ID space: the coarsest common refinement required by runtime mask
  and commit.

Once all four are named, the code becomes much easier to explain in the paper:
compilation is not a pile of optimizations; it is a sequence of relation
construction and quotient/refinement steps.
