use glrmask::{DynamicConstraint, Grammar, Constraint, Vocab};

#[test]
fn termination_is_caller_policy_after_acceptance() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (64, Vec::new())]);
    let grammar = Grammar::ebnf(r#"start ::= "a""#);
    let static_constraint = Constraint::compile(grammar.clone(), &vocab).unwrap();
    let dynamic_constraint = DynamicConstraint::compile(grammar, &vocab).unwrap();

    let mut static_state = static_constraint.start();
    let mut dynamic_state = dynamic_constraint.start();
    static_state.commit_token(0).unwrap();
    dynamic_state.commit_token(0).unwrap();
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
    use glrmask::{BuildOptions, Optimization};

    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let grammar = Grammar::ebnf(r#"start ::= "a" "b"?"#);
    let allowed = |mask: &[u32], token: u32| -> bool {
        let word_idx = token as usize / 32;
        let bit = token % 32;
        mask.get(word_idx).map_or(false, |w| (w & (1 << bit)) != 0)
    };

    let modes = [
        Optimization::Auto,
        Optimization::FastBuild,
        Optimization::Balanced,
        Optimization::FastRuntime,
    ];

    for mode in modes {
        let constraint = grammar
            .clone()
            .compile_with(
                &vocab,
                BuildOptions::default()
                    .optimization(mode)
                    .end_tokens([99]),
            )
            .unwrap();

        // Path 1: Stop immediately after "a" with EOS 99
        {
            let mut state = constraint.start();
            assert!(!state.is_accepting());
            assert!(!state.is_rejected());
            assert!(allowed(&state.mask(), 0));
            assert!(!allowed(&state.mask(), 1));
            assert!(!allowed(&state.mask(), 99));

            state.commit_token(0).unwrap();
            // Live acceptance after "a", which permits both continuation "b" and EOS 99
            assert!(state.is_accepting());
            assert!(!state.is_rejected());
            assert!(allowed(&state.mask(), 1));
            assert!(allowed(&state.mask(), 99));

            // Commit EOS 99
            state.commit_token(99).unwrap();
            assert!(state.is_accepting());
            assert!(!state.is_rejected());
            assert!(state.mask().iter().all(|&w| w == 0));
            assert!(state.commit_token(1).is_err());
            assert!(state.commit_bytes(b"b").is_err());

            // Terminated clone behavior
            let mut clone = state.clone();
            assert!(clone.is_accepting());
            assert!(!clone.is_rejected());
            assert!(clone.mask().iter().all(|&w| w == 0));
            assert!(clone.commit_token(1).is_err());
            assert!(clone.commit_bytes(b"b").is_err());
        }

        // Path 2: Continue through "b" after "a", then commit EOS 99
        {
            let mut state = constraint.start();
            state.commit_token(0).unwrap();
            assert!(state.is_accepting());
            assert!(allowed(&state.mask(), 1));
            assert!(allowed(&state.mask(), 99));

            // Continue through "b"
            state.commit_token(1).unwrap();
            assert!(state.is_accepting());
            assert!(!state.is_rejected());
            assert!(!allowed(&state.mask(), 0));
            assert!(!allowed(&state.mask(), 1));
            assert!(allowed(&state.mask(), 99));

            // Commit EOS 99
            state.commit_token(99).unwrap();
            assert!(state.is_accepting());
            assert!(!state.is_rejected());
            assert!(state.mask().iter().all(|&w| w == 0));
            assert!(state.commit_token(0).is_err());
            assert!(state.commit_token(1).is_err());
            assert!(state.commit_bytes(b"a").is_err());

            // Terminated clone behavior
            let mut clone = state.clone();
            assert!(clone.is_accepting());
            assert!(!clone.is_rejected());
            assert!(clone.mask().iter().all(|&w| w == 0));
            assert!(clone.commit_token(0).is_err());
            assert!(clone.commit_bytes(b"a").is_err());
        }
    }
}
