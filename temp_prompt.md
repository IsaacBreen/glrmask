
# Goal: Reproduce Transition Count Explosion on Epsilon Start Merge

The objective is to create a minimal, reproducible Rust test case that demonstrates a counter-intuitive phenomenon observed in the `TerminalDWA`:
Replacing the **initial transitions** of an NWA with **epsilon transitions** (conceptually merging the start states) causes the total number of transitions in the final **Determinized & Simplified** DWA to **INCREASE** significantly (e.g., in the real codebase, it jumped from 45k to 315k transitions).

## Context
- **Experiment:** We verified this behavior in `src/constraint.rs` by modifying the `TerminalDWA` construction.
- **Hypothesis:** This explosion is likely due to the interaction of **Weighted Automata** logic. The `TerminalNWA` uses dense *bitvector weights* (Token IDs). When start states are merged, the determinizer creates composite states that cannot be simplified/merged because their outgoing transitions have incompatible weights, leading to a combinatorial explosion of states/transitions deep in the graph.
- **Current Attempt:** tried fuzzing random graphs in `src/precompute4/weighted_automata/test_determinization_explosion.rs`.
    - **Simple Weights (Weight::all()):** Failed to reproduce (simplification always reduced size).
    - **Distinct Weights:** Preliminary fuzzing didn't find a case yet.

## Task
1. Analyze why the real `TerminalDWA` explodes. It is a **Trie** of tokens. Tokens share prefixes but have distinct validity masks (weights).
2. Construct a test case in `src/precompute4/weighted_automata/test_determinization_explosion.rs` that mimics this "Weighted Trie" structure.
    - Example: Two strings `abc` and `abd` with *disjoint weights* `W1` and `W2`.
    - Or a cyclic structure where weights interfere.
3. Validate that `modified.determinize().simplify()` results in MORE transitions than `original.determinize().simplify()`.
4. Ensure the test fails if the explosion is NOT observed (i.e., we want a passing test that proves the explosion exists).

## Relevant Files
- `src/precompute4/weighted_automata/test_determinization_explosion.rs` (Scratchpad for the test)
- `src/precompute4/weighted_automata/nwa.rs` (NWA implementation)
- `src/precompute4/weighted_automata/determinization.rs`
- `src/precompute4/weighted_automata/simplification.rs`
