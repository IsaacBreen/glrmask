//! Minimal TI-off reproducer for byte-restricted L2P state equivalence.
//!

use glrmask::{Constraint, Vocab};

fn contains(mask: &[u32], token: u32) -> bool {
    let word = token as usize / 32;
    let bit = token % 32;
    mask.get(word).is_some_and(|word| word & (1 << bit) != 0)
}

/// A single terminal needs three bytes, but the first vocabulary token ends
/// after its two punctuation bytes. `a` is in a different byte partition.
///
/// `FORCE_ALL_L2P` is solely to direct this one-terminal grammar through the
/// L2P code path. The ordinary path classifies a one-terminal grammar as L1,
/// where this specific partition-local continuation machinery is not used.
/// Terminal interchangeability is explicitly disabled.
#[test]
fn byte_restricted_state_equivalence_must_admit_a_cross_partition_partial_token() {
    unsafe {
        std::env::set_var("GLRMASK_FORCE_ALL_L2P", "1");
        std::env::set_var("GLRMASK_SPLIT_L2P_VOCAB", "0");
        std::env::set_var("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "1");
    }

    // Exactly one grammar terminal, and exactly the two necessary tokens.
    let vocab = Vocab::new(vec![(0, b"{\"".to_vec()), (1, b"a".to_vec())]);
    let constraint = Constraint::from_lark("start: A\nA: \"{\\\"a\"", &vocab).unwrap();

    let mut state = constraint.start();
    assert!(
        contains(&state.mask(), 0),
        "the token '{{\"' must be admitted because a later token 'a' completes A"
    );
    state.commit_token(0).unwrap();
    assert!(contains(&state.mask(), 1));
    state.commit_token(1).unwrap();
    assert!(state.is_finished());
}
