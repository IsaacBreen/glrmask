mod ti_mre_tests {
    use std::{env, ffi::OsString, sync::Mutex};

    use crate::{Constraint, Vocab};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = env::var_os(key);
            unsafe {
                env::set_var(key, value);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    env::set_var(self.key, value);
                },
                None => unsafe {
                    env::remove_var(self.key);
                },
            }
        }
    }

    #[test]
    fn p7_and_p8_use_terminal_interchangeability_by_default() {
        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");

        assert!(crate::compiler::stages::id_map_and_terminal_dwa::l2p::l2p_terminal_interchangeability_enabled_for_partition("p7"));
        assert!(crate::compiler::stages::id_map_and_terminal_dwa::l2p::l2p_terminal_interchangeability_enabled_for_partition("p8"));
    }

    #[test]
    fn terminal_interchangeability_policy_leaves_generic_partitions_unchanged() {
        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");

        assert!(crate::compiler::stages::id_map_and_terminal_dwa::l2p::l2p_terminal_interchangeability_enabled_for_partition("p0"));
        assert!(crate::compiler::stages::id_map_and_terminal_dwa::l2p::l2p_terminal_interchangeability_enabled_for_partition("p7"));
    }

    #[test]
    fn terminal_interchangeability_policy_defaults_enabled_and_honors_explicit_disable() {
        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let original = env::var_os("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY");
        unsafe {
            env::remove_var("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY");
        }
        let _restore = EnvVarGuard {
            key: "GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY",
            original,
        };

        assert!(crate::compiler::stages::id_map_and_terminal_dwa::l2p::l2p_terminal_interchangeability_enabled_for_partition("p0"));
        let _disabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "1");
        assert!(!crate::compiler::stages::id_map_and_terminal_dwa::l2p::l2p_terminal_interchangeability_enabled_for_partition("p0"));
    }

    #[test]
    fn p7_boundary_bypass_matches_forced_full_ti_reference() {
        let grammar = r#"
start S;
t TRUE ::= "true";
t FALSE ::= "false";
t NULL ::= "null";
nt S ::= TRUE | FALSE | NULL;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b" true".to_vec()),
                (1, b" false".to_vec()),
                (2, b" null".to_vec()),
                (3, b"[true".to_vec()),
                (4, b" -".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _structural = EnvVarGuard::set("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION", "1");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );
        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("P7 local TI bypass must match the forced full-TI artifact");
    }

    #[test]
    fn p8_boundary_bypass_matches_forced_full_ti_reference() {
        let grammar = r#"
start S;
t QUOTE ::= "\"";
t IDENT ::= /[A-Za-z_][A-Za-z0-9_]*/;
nt S ::= QUOTE IDENT;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b"\"A".to_vec()),
                (1, b"\"Z".to_vec()),
                (2, b"\"_".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _structural = EnvVarGuard::set("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION", "1");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );
        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("P8 local TI bypass must match the forced full-TI artifact");
    }

    #[test]
    fn forced_all_l2p_enum_skips_p8_first_byte_factorization() {
        let vocab = Vocab::new(
            vec![
                (0, b"\"red\"".to_vec()),
                (1, b"\"blue\"".to_vec()),
                (2, b"\"green\"".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
        let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );

        Constraint::from_json_schema(r#"{"enum":["red","blue"]}"#, &vocab)
            .expect("forced-all-L2P enum must match the strict TI reference");
    }

    #[test]
    fn partitioned_epsilon_ti_matches_strict_reference() {
        let grammar = r#"
start S;
lexer group a ::= A;
lexer group b ::= B;
lexer group c ::= C;
t A ::= "x";
t B ::= [xy] & [xz];
t C ::= "z";
nt S ::= A | B | C;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b"x".to_vec()),
                (1, b"xx".to_vec()),
                (2, b"z".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _adaptive = EnvVarGuard::set("GLRMASK_LEXER_ADAPTIVE", "0");
        let _force_l2p = EnvVarGuard::set("GLRMASK_FORCE_ALL_L2P", "1");
        let _disable_vocab_split = EnvVarGuard::set("GLRMASK_SPLIT_L2P_VOCAB", "0");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY_STRICT_REFERENCE",
            "1",
        );

        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("epsilon-NFA TI must match its TI-off symbolic reference");
    }

    #[test]
    fn partitioned_epsilon_p8_global_token_position_matches_strict_reference() {
        let grammar = r#"
start S;
lexer group quote ::= QUOTE;
lexer group ident ::= IDENT;
t QUOTE ::= "\"";
t IDENT ::= /[A-Za-z_][A-Za-z0-9_]*/;
nt S ::= QUOTE IDENT;
"#;
        let vocab = Vocab::new(
            vec![
                (0, b"\"A".to_vec()),
                (1, b"\"Z".to_vec()),
                (2, b"\"_".to_vec()),
            ]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _adaptive = EnvVarGuard::set("GLRMASK_LEXER_ADAPTIVE", "0");
        let _structural = EnvVarGuard::set("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION", "1");
        let _disable_ti =
            EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "1");
        let _strict = EnvVarGuard::set(
            "GLRMASK_L2P_GLOBAL_TOKEN_POSITION_STRICT_REFERENCE",
            "1",
        );

        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("epsilon-NFA C quotient must match its suppressed-C symbolic reference");
    }

    #[test]
    fn representative_only_vocab_equivalence_mre() {
        // `b" _"` completes SPACE then is a live prefix of CLASS. With TI
        // enabled, CLASS is hidden behind representative FROM during equivalence
        // analysis. Because FROM cannot follow SPACE but CLASS can, the
        // representative-labeled follow table must be COALESCED (a follow is
        // disallowed for the class only if disallowed for every member);
        // otherwise equivalence prunes the FROM-class continuation, merges
        // `b" !"`/`b" _"`, and the completed terminal DWA underaccepts
        // `[SPACE, CLASS]`. Regression guard for that coalescing fix.
        let grammar = r#"
start S;
t V ::= /.+/;
t SPACE ::= " ";
t FROM ::= /_a_/;
t CLASS ::= /_b_/;
nt S ::= FROM V | SPACE V SPACE CLASS;
"#;
        let vocab = Vocab::new(vec![(0, b" !".to_vec()), (1, b" _".to_vec())]);

        let _lock = ENV_LOCK.lock().expect("TI MRE env lock poisoned");
        let _enabled = EnvVarGuard::set("GLRMASK_DISABLE_L2P_TERMINAL_INTERCHANGEABILITY", "0");
        Constraint::from_glrm_grammar(grammar, &vocab)
            .expect("TI must preserve the completed terminal-DWA language");
    }
}


mod synthetic_state_facade_tests {
    use crate::{Constraint, DynamicConstraint, Vocab};

    struct LlGuidanceCompatGuard;

    impl LlGuidanceCompatGuard {
        fn enabled() -> Self {
            crate::set_test_compat_mode(true);
            Self
        }
    }

    impl Drop for LlGuidanceCompatGuard {
        fn drop(&mut self) {
            crate::set_test_compat_mode(false);
        }
    }

    #[test]
    fn synthesized_intersection_never_drops_long_live_tokens_after_min_length() {
        let _compat = LlGuidanceCompatGuard::enabled();
        let vocab = Vocab::new(vec![
            (0, b"{\"".to_vec()),
            (1, b"email".to_vec()),
            (2, b"\":".to_vec()),
            (3, b" \"".to_vec()),
            (4, b"john".to_vec()),
            (5, b".".to_vec()),
            (6, b"_____".to_vec()),
        ]);
        let schema = r#"{
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": "^\\S+@\\S+$",
                    "minLength": 5,
                    "maxLength": 255
                }
            },
            "required": ["email"]
        }"#;

        let constraint = Constraint::from_json_schema(schema, &vocab).expect("static constraint");
        let dynamic =
            DynamicConstraint::from_json_schema(schema, &vocab).expect("dynamic constraint");
        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();
        for token in 0..=5 {
            static_state.commit_token(token).expect("static prefix token");
            dynamic_state.commit_token(token).expect("dynamic prefix token");
        }

        let static_mask = static_state.mask();
        let dynamic_mask = dynamic_state.mask();
        assert_eq!(static_mask, dynamic_mask);
        assert_ne!(
            static_mask[0] & (1 << 6),
            0,
            "five-byte non-whitespace token must remain live after minLength is reached",
        );
        static_state
            .commit_token(6)
            .expect("a token admitted by the exact language must commit statically");
        dynamic_state
            .commit_token(6)
            .expect("a token admitted by the exact language must commit dynamically");
        assert_eq!(static_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn static_synthesized_pipeline_matches_exact_dynamic_runtime_through_full_bound() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaaa".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^a{1,80}$",
            "minLength": 1,
            "maxLength": 80
        }"#;
        let constraint = Constraint::from_json_schema(schema, &vocab).expect("static constraint");
        let dynamic =
            DynamicConstraint::from_json_schema(schema, &vocab).expect("dynamic constraint");
        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();

        assert_eq!(static_state.mask(), dynamic_state.mask());
        static_state.commit_token(0).expect("opening quote");
        dynamic_state.commit_token(0).expect("opening quote");
        for chunk in 0..20 {
            assert_eq!(
                static_state.mask(),
                dynamic_state.mask(),
                "mask mismatch before four-byte chunk {chunk}",
            );
            static_state.commit_token(3).expect("four a bytes");
            dynamic_state.commit_token(3).expect("four a bytes");
        }
        assert_eq!(static_state.mask(), dynamic_state.mask());
        let mask = static_state.mask();
        assert_ne!(mask[0] & (1 << 0), 0, "closing quote must be allowed");
        assert_eq!(mask[0] & (1 << 1), 0, "65th a must be rejected");
        assert_eq!(mask[0] & (1 << 2), 0, "66th a must be rejected");
        assert_eq!(mask[0] & (1 << 3), 0, "68th a must be rejected");
        static_state.commit_token(0).expect("closing quote");
        dynamic_state.commit_token(0).expect("closing quote");
        assert_eq!(static_state.is_finished(), dynamic_state.is_finished());
        assert_eq!(static_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn independently_synthesized_identical_terminals_preserve_different_full_lifetimes() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aaaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let schema = r#"{
            "anyOf": [
                {"type":"string","pattern":"^a{1,80}$","maxLength":80},
                {"type":"string","pattern":"^a{1,160}$","maxLength":160}
            ]
        }"#;
        let constraint = Constraint::from_json_schema(schema, &vocab).expect("static constraint");
        let dynamic =
            DynamicConstraint::from_json_schema(schema, &vocab).expect("dynamic constraint");
        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();

        static_state.commit_token(0).expect("opening quote");
        dynamic_state.commit_token(0).expect("opening quote");
        for chunk in 0..20 {
            assert_eq!(static_state.mask(), dynamic_state.mask(), "chunk {chunk}");
            static_state.commit_token(2).expect("four a bytes");
            dynamic_state.commit_token(2).expect("four a bytes");
        }

        let at_short_limit = static_state.mask();
        assert_eq!(at_short_limit, dynamic_state.mask());
        assert_ne!(at_short_limit[0] & (1 << 0), 0, "short terminal may close");
        assert_ne!(
            at_short_limit[0] & (1 << 2),
            0,
            "long terminal must remain alive after the short terminal expires",
        );

        for chunk in 20..40 {
            assert_eq!(static_state.mask(), dynamic_state.mask(), "chunk {chunk}");
            static_state.commit_token(2).expect("long terminal continuation");
            dynamic_state.commit_token(2).expect("long terminal continuation");
        }
        let at_long_limit = static_state.mask();
        assert_eq!(at_long_limit, dynamic_state.mask());
        assert_ne!(at_long_limit[0] & (1 << 0), 0, "long terminal may close");
        assert_eq!(
            at_long_limit[0] & (1 << 2),
            0,
            "long terminal must expire at its exact full bound",
        );
    }

    #[test]
    #[ignore = "profiling probe for the pathological nested-repeat/max-length product"]
    fn profile_pathological_nested_repeat_max_length() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"aabb".to_vec()),
            (5, b"aaaa".to_vec()),
            (6, b"bbbb".to_vec()),
        ]);
        let schema = r#"{
            "type":"string",
            "pattern":"^(?:a+b+){0,100}a+$",
            "minLength":2,
            "maxLength":500
        }"#;
        std::hint::black_box(
            Constraint::from_json_schema(schema, &vocab).expect("pathological exact constraint"),
        );
    }

}
