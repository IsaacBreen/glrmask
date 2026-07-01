//! Regression: two terminals that merely share a suffix (`A="at"`, `B="bt"`)
//! must NOT be coalesced when the vocab byte set is only `{a}`. Restricted to
//! `a`, the reset / mid-A / mid-B states share the same *finalizers* (none) but
//! differ in their *future finalizers* (`{A,B}` / `{A}` / `{B}`). The
//! interchange criterion must characterize states by future finalizers too, so
//! A and B are not interchangeable here and no spurious `[B]` match is produced.
//!
//! Before the fix (finalizers only) this over-accepted `word=[B]` for token
//! `"a"`. See obsidian/.../MINIMAL-mre-at-bt-single-token.md.

use glrmask::{Constraint, Vocab};

#[test]
fn shared_suffix_terminals_are_not_interchangeable_under_prefix_byte_partition() {
    unsafe {
        std::env::set_var("GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY", "1");
        std::env::set_var("GLRMASK_ASSERT_L2P_TERMINAL_INTERCHANGEABILITY_EQUAL", "1");
        std::env::set_var("GLRMASK_FORCE_ALL_L2P", "1");
        std::env::set_var("GLRMASK_SPLIT_L2P_VOCAB", "0");
    }
    let grammar = "start: A | B\nA: \"at\"\nB: \"bt\"";
    let v = Vocab::new(vec![(0u32, b"a".to_vec())], None);
    // The assertion inside the build panics on candidate/baseline mismatch;
    // reaching here means the artifacts agree.
    let _ = Constraint::from_lark(grammar, &v).unwrap();
}
