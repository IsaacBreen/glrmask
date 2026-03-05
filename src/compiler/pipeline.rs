//! Compilation pipeline.
//!
//! The main entry point for compiling a grammar + vocabulary into a [`Constraint`].
//!
//! # Pipeline stages
//!
//! 1. Build augmented GLR grammar (FIRST/FOLLOW/nullable)
//! 2. Build SLR(1) parse table
//! 3. Build tokenizer DFA from terminal patterns
//! 4. Compute vocabulary preprocessing (TSID mapping, possible_matches)
//! 5. Build parser NWA from terminal characterizations
//! 6. Determinize + minimize → parser DWA
//! 7. Package as `Constraint`

use crate::Vocab;
use crate::compiler::glr::grammar::GlrGrammar;
use crate::compiler::glr::table::GlrTable;
use crate::compiler::grammar_def::GrammarDef;
use crate::compiler::parser_dwa::build_parser_dwa;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::compiler::vocab_pre::VocabPreprocessing;
use crate::runtime::Constraint;

/// Compile a grammar definition and vocabulary into a `Constraint`.
///
/// This is the main compilation entry point:
/// ```text
/// GrammarDef + Vocab
///       │
///       ├── GlrGrammar::from_grammar_def()
///       │      │
///       │      └── GlrTable::build()
///       │
///       ├── TokenizerDfa::from_grammar_def()
///       │      │
///       │      └── VocabPreprocessing::compute()
///       │
///       └── build_parser_dwa()  ← uses table + grammar + vocab_pre
///              │
///              └── determinize + minimize → CompDwa
///                     │
///                     └── Constraint { parser_dwa, table, tokenizer, vocab_pre, ... }
/// ```
pub fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    // 1. Build augmented GLR grammar.
    let glr_grammar = GlrGrammar::from_grammar_def(grammar);

    // 2. Build SLR(1) parse table.
    let table = GlrTable::build(&glr_grammar);

    // 3. Build tokenizer DFA.
    let tokenizer = TokenizerDfa::from_grammar_def(grammar);

    // 4. Compute vocabulary preprocessing.
    let vocab_pre = VocabPreprocessing::compute(&tokenizer, vocab);

    // 5–6. Build parser DWA (NWA → determinize → minimize).
    let parser_dwa = build_parser_dwa(&table, &glr_grammar, &vocab_pre);

    // 7. Package as Constraint.
    let token_bytes: std::collections::BTreeMap<u32, Vec<u8>> = vocab
        .entries
        .iter()
        .map(|(id, bytes)| (*id, bytes.clone()))
        .collect();

    Constraint {
        parser_dwa,
        table,
        tokenizer,
        num_tsids: vocab_pre.num_tsids,
        state_to_tsid: vocab_pre.state_to_tsid.clone(),
        tsid_to_state: vocab_pre.tsid_to_state.clone(),
        possible_matches: vocab_pre.possible_matches,
        max_token: vocab_pre.max_token,
        eos_token_id: vocab.eos_token_id,
        token_bytes,
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar_def::tests::*;

    #[test]
    fn test_compile_simple_ab() {
        let gdef = simple_ab_grammar(); // S → a b
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.num_dwa_states() > 0);
        assert!(constraint.num_tsids() > 0);
    }

    #[test]
    fn test_compile_choice() {
        let gdef = choice_grammar(); // S → a | b
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.num_dwa_states() > 0);
    }

    #[test]
    fn test_compile_two_nt() {
        let gdef = two_nt_grammar(); // S → A b, A → a
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.num_dwa_states() > 0);
        assert!(constraint.num_parser_states() > 0);
    }

    #[test]
    fn test_end_to_end_simple_ab() {
        // Grammar: S → a b
        // Vocab: token 0 = "a", token 1 = "b"
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        // Initial mask: "a" allowed, "b" not.
        let mask = state.compute_mask(&constraint);
        assert!(mask.get(0), "token 'a' should be allowed initially");
        assert!(!mask.get(1), "token 'b' should NOT be allowed initially");

        // Commit "a".
        state
            .commit(&constraint, 0)
            .expect("commit 'a' should succeed");
        assert!(
            !state.is_accepting(&constraint),
            "not yet accepting after 'a'"
        );

        // After "a": "b" allowed, "a" not.
        let mask = state.compute_mask(&constraint);
        assert!(!mask.get(0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask.get(1), "token 'b' should be allowed after 'a'");

        // Commit "b".
        state
            .commit(&constraint, 1)
            .expect("commit 'b' should succeed");
        assert!(state.is_accepting(&constraint), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_choice() {
        // Grammar: S → a | b
        // Vocab: token 0 = "a", token 1 = "b"
        // Expected: initial mask allows both "a" and "b".
        let gdef = choice_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        // Initial mask: should allow both "a" and "b".
        let mask = state.compute_mask(&constraint);
        assert!(mask.get(0), "token 'a' should be allowed");
        assert!(mask.get(1), "token 'b' should be allowed");

        // Commit token "a".
        state
            .commit(&constraint, 0)
            .expect("commit 'a' should succeed");
        assert!(
            state.is_accepting(&constraint),
            "parse should accept after 'a'"
        );
    }

    #[test]
    fn test_end_to_end_two_nt() {
        // Grammar: S → A b, A → a
        // Same as simple_ab but with an intermediate nonterminal.
        let gdef = two_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        // Initial: "a" allowed, "b" not.
        let mask = state.compute_mask(&constraint);
        assert!(mask.get(0), "token 'a' should be allowed initially");
        assert!(!mask.get(1), "token 'b' should NOT be allowed initially");

        // Commit "a".
        state
            .commit(&constraint, 0)
            .expect("commit 'a' should succeed");
        assert!(
            !state.is_accepting(&constraint),
            "not yet accepting after 'a'"
        );

        // After "a": "b" allowed, "a" not.
        let mask = state.compute_mask(&constraint);
        assert!(!mask.get(0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask.get(1), "token 'b' should be allowed after 'a'");

        // Commit "b".
        state
            .commit(&constraint, 1)
            .expect("commit 'b' should succeed");
        assert!(state.is_accepting(&constraint), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_nested_nt() {
        // Grammar: S → A B, A → a, B → b
        // Same result as S → a b but with two nonterminal reductions.
        let gdef = nested_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        // Initial: "a" allowed, "b" not.
        let mask = state.compute_mask(&constraint);
        assert!(mask.get(0), "token 'a' should be allowed initially");
        assert!(!mask.get(1), "token 'b' should NOT be allowed initially");

        // Commit "a".
        state
            .commit(&constraint, 0)
            .expect("commit 'a' should succeed");
        assert!(!state.is_accepting(&constraint), "not accepting after 'a'");

        // After "a": "b" should be allowed (A reduced, now need B → b).
        let mask = state.compute_mask(&constraint);
        assert!(!mask.get(0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask.get(1), "token 'b' should be allowed after 'a'");

        // Commit "b".
        state
            .commit(&constraint, 1)
            .expect("commit 'b' should succeed");
        assert!(state.is_accepting(&constraint), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_three_terminals() {
        // Grammar: S → a b c
        let gdef = three_terminal_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        // Initial: only "a" allowed.
        let mask = state.compute_mask(&constraint);
        assert!(mask.get(0), "token 'a' should be allowed initially");
        assert!(!mask.get(1), "token 'b' should NOT be allowed initially");
        assert!(!mask.get(2), "token 'c' should NOT be allowed initially");

        // Commit "a".
        state.commit(&constraint, 0).expect("commit 'a'");

        // After "a": only "b" allowed.
        let mask = state.compute_mask(&constraint);
        assert!(!mask.get(0), "no 'a' after 'a'");
        assert!(mask.get(1), "'b' after 'a'");
        assert!(!mask.get(2), "no 'c' after 'a'");

        // Commit "b".
        state.commit(&constraint, 1).expect("commit 'b'");

        // After "ab": only "c" allowed.
        let mask = state.compute_mask(&constraint);
        assert!(!mask.get(0), "no 'a' after 'ab'");
        assert!(!mask.get(1), "no 'b' after 'ab'");
        assert!(mask.get(2), "'c' after 'ab'");

        // Commit "c".
        state.commit(&constraint, 2).expect("commit 'c'");
        assert!(state.is_accepting(&constraint), "should accept after 'abc'");
    }

    #[test]
    fn test_end_to_end_nested_two_rhs() {
        // Grammar: S → A c, A → a b
        // Exercises reduce with pop_count=2.
        let gdef = nested_two_rhs_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        // Initial: only "a" allowed.
        let mask = state.compute_mask(&constraint);
        assert!(mask.get(0), "token 'a' should be allowed initially");
        assert!(!mask.get(1), "token 'b' should NOT be allowed initially");
        assert!(!mask.get(2), "token 'c' should NOT be allowed initially");

        // Commit "a".
        state.commit(&constraint, 0).expect("commit 'a'");

        // After "a": only "b" allowed (still in A → a • b).
        let mask = state.compute_mask(&constraint);
        assert!(!mask.get(0), "no 'a' after 'a'");
        assert!(mask.get(1), "'b' after 'a'");
        assert!(!mask.get(2), "no 'c' after 'a'");

        // Commit "b".
        state.commit(&constraint, 1).expect("commit 'b'");

        // After "ab": "c" should be allowed (A reduced, S → A • c).
        let mask = state.compute_mask(&constraint);
        assert!(!mask.get(0), "no 'a' after 'ab'");
        assert!(!mask.get(1), "no 'b' after 'ab'");
        assert!(mask.get(2), "'c' after 'ab'");

        // Commit "c".
        state.commit(&constraint, 2).expect("commit 'c'");
        assert!(state.is_accepting(&constraint), "should accept after 'abc'");
    }
}
