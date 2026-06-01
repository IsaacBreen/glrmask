# Graph-structured stack representation

`LeveledGSS<T, A>` denotes a finite set of parser stacks with accumulated values. It lives under `parser` because pop/push/merge/isolate/apply/prune are parser-stack language operations, not generic collection operations.
