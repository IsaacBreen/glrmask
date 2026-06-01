# Template DFA subsystem

A template DFA is a finite quotient of GLR stack effects. It is neither the parser nor the tokenizer; it is a precomputed shortcut for the Commit sub-relation `(template stack prefix, completed terminal) -> advanced parser frontier`.

Reading order: `characterize.rs`, then `compile_dfa.rs`, then `compile_bundle.rs`. The correctness invariant is denotational equality with direct GLR advancement for every represented template state.
