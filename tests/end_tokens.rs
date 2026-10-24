use glrmask::test_modes::MaskMode;
use glrmask::{DynamicConstraint, Grammar, Constraint, Vocab};

#[test]
fn termination_is_caller_policy_after_acceptance() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (64, Vec::new())]);
    let grammar = Grammar::ebnf(r#"start ::= "a""#);
    let static_constraint = Constraint::compile(grammar.clone(), &vocab).unwrap();
    let dynamic_constraint = DynamicConstraint::compile(grammar, &vocab).unwrap();

    let mut static_state = static_constraint.start();
    let mut dynamic_state = dynamic_constraint.start();
    assert!(static_state.commit_token(0));
    assert!(dynamic_state.commit_token(0));
    assert!(static_state.is_accepting());
    assert!(dynamic_state.is_accepting());

    // This constraint was built without configured end tokens. A caller stops
    // here (or applies its own generation policy) based only on acceptance;
    // token 64 is never handed to the constraint as an implicit terminator.
    let caller_should_stop = static_state.is_accepting() && dynamic_state.is_accepting();
    assert!(caller_should_stop);
}

#[test]
fn configured_eos_absent_from_vocab_supports_continuation_and_terminal_lifecycle_across_modes() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let grammar = Grammar::ebnf(r#"start ::= "a" "b"?"#);
    let allowed = |mask: &[u32], token: u32| -> bool {
        let word_idx = token as usize / 32;
        let bit = token % 32;
        mask.get(word_idx).map_or(false, |w| (w & (1 << bit)) != 0)
    };

    let modes = MaskMode::ALL;

    for mode in modes {
        let constraint = grammar
            .clone()
            .compile_with_build_options(
                &vocab,
                mode.build_options()
                    .end_tokens([99]),
            )
            .unwrap();

        // Path 1: Stop immediately after "a" with EOS 99
        {
            let mut state = constraint.start();
            assert!(!state.is_accepting());
            assert!(!(state.mask().iter().all(|word| *word == 0)));
            assert!(allowed(&state.mask(), 0));
            assert!(!allowed(&state.mask(), 1));
            assert!(!allowed(&state.mask(), 99));

            assert!(state.commit_token(0));
            // The grammar is complete after "a", but with end tokens configured
            // acceptance needs the EOS commit. Both "b" and EOS 99 are allowed.
            assert!(!state.is_accepting());
            assert!(allowed(&state.mask(), 1));
            assert!(allowed(&state.mask(), 99));

            // Commit EOS 99
            assert!(state.commit_token(99));
            assert!(state.is_accepting());
            assert!(state.mask().iter().all(|&w| w == 0));

            // Terminated clone behavior (clone before failing commits, which
            // reject the state).
            let mut clone = state.clone();
            assert!(clone.is_accepting());
            assert!(clone.mask().iter().all(|&w| w == 0));
            assert!(!state.commit_token(1));
            assert!(!state.commit_bytes(b"b"));
            assert!(!clone.commit_token(1));
            assert!(!clone.commit_bytes(b"b"));
        }

        // Path 2: Continue through "b" after "a", then commit EOS 99
        {
            let mut state = constraint.start();
            assert!(state.commit_token(0));
            assert!(!state.is_accepting());
            assert!(allowed(&state.mask(), 1));
            assert!(allowed(&state.mask(), 99));

            // Continue through "b"
            assert!(state.commit_token(1));
            assert!(!state.is_accepting());
            assert!(!(state.mask().iter().all(|word| *word == 0)));
            assert!(!allowed(&state.mask(), 0));
            assert!(!allowed(&state.mask(), 1));
            assert!(allowed(&state.mask(), 99));

            // Commit EOS 99
            assert!(state.commit_token(99));
            assert!(state.is_accepting());
            assert!(state.mask().iter().all(|&w| w == 0));

            // Terminated clone behavior (clone before failing commits, which
            // reject the state).
            let mut clone = state.clone();
            assert!(clone.is_accepting());
            assert!(clone.mask().iter().all(|&w| w == 0));
            assert!(!state.commit_token(0));
            assert!(!state.commit_token(1));
            assert!(!state.commit_bytes(b"a"));
            assert!(!clone.commit_token(0));
            assert!(!clone.commit_bytes(b"a"));
        }
    }
}
