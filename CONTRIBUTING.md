# Contributing

This repository is currently being prepared for publication. During the cleanup phase, prefer changes that make the implementation easier to compare with the paper.

## Cleanup principles

1. Keep public API changes explicit and documented.
2. Keep paper terminology visible in module names, rustdoc, and README prose.
3. Separate algorithmic refactors from mechanical moves where possible.
4. Do not commit generated caches, benchmark artifacts, local vocab dumps, or platform metadata.
5. Do not add library-side `println!` or `eprintln!` diagnostics in normal API paths.
6. Add narrow comments to any remaining `allow`, panic, unwrap, or environment-variable compatibility path.

## Preferred review order

1. Repository hygiene and metadata.
2. Public API boundary.
3. Paper terminology alignment.
4. Compile pipeline structure.
5. Runtime Mask and Commit separation.
6. Tests, docs, examples, and benchmarks.

## Validation policy

Large structural refactors may be staged before compiling, but no release branch should be merged until formatting, linting, tests, examples, Python bindings, and representative benchmarks have been run and documented.
