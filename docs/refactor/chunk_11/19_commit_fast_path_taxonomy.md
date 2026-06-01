# Commit fast-path taxonomy

The remaining `runtime/commit/mod.rs` body contains several fast paths.  This
chunk does not split them, but it names the taxonomy for the next cleanup.

Known categories:

- initial commit scan special cases;
- terminal/actionable-match collection;
- pruning initial tokenizer states;
- future-terminal disallow propagation;
- single-top-action reductions;
- fast terminal-match advance;
- full-width fast path;
- small-queue fast path;
- direct-linear fast path;
- profiled versions of the above;
- reference `commit_bytes_impl` path.

Each category should eventually become either a file or a named section with a
clear precondition:

```text
Precondition: what shape of scan/parser state makes this path applicable?
Relation: what transition does it compute?
Fallback: when does it decline to handle the input?
Validation: what reference path can check it?
```

The new `parser_advance.rs` file makes that future split easier: all paths can
call the same parser-stack transition dispatch.
