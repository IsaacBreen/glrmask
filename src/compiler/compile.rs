//! Top-level compiler orchestration.
//!
//! This module owns the entrypoints that turn a grammar plus vocabulary into a
//! compiled [`Constraint`]. Stage implementations live under `compiler/stages/`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::debug::CompileDebug;
use crate::compiler::glr::analysis::GLRGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::ast::GrammarDef;
use crate::compiler::grammar::normalize::normalize_for_mask;
use crate::compiler::parser_dwa::build_parser_dwa;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, analyze_equivalences};
use crate::runtime::Constraint;

/// Compile a grammar definition and vocabulary into a `Constraint`.
///
/// This is the main compilation entry point:
/// ```text
/// GrammarDef + Vocab
///       │
///       ├── GLRGrammar::from_grammar_def()
///       │      │
///       │      └── GLRTable::build()
///       │
///       ├── Tokenizer::from_grammar_def()
///       │      │
///       │      └── analyze_equivalences() → InternalIdMap
///       │
///       └── build_parser_dwa()  ← uses table + grammar + tokenizer + vocab + id_map
///              │
///              └── determinize + minimize → DWA
///                     │
///                     └── Constraint { parser_dwa, table, tokenizer, id_map-derived metadata, ... }
/// ```
pub fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    unimplemented!()
}

/// Compile, returning a [`CompileDebug`] bundle alongside the [`Constraint`].
///
/// The debug bundle captures every intermediate automaton stage so callers can
/// inspect the terminal NWA before/after optimisations, the composed parser
/// NWA before/after resolve_negatives, the DWA pre/post minimisation, etc.
pub fn compile_with_debug(grammar: &GrammarDef, vocab: &Vocab) -> (Constraint, CompileDebug) {
    unimplemented!()
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::ast::tests::*;

    fn mask_has_token(mask: &[u32], token: u32) -> bool {
        let word = token as usize / 32;
        let bit = token as usize % 32;
        word < mask.len() && (mask[word] & (1u32 << bit)) != 0
    }

    #[test]
    fn test_compile_simple_ab() {
        let gdef = simple_ab_grammar(); // S → a b
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(!constraint.terminal_tokens_by_state.is_empty());
    }

    #[test]
    fn test_compile_choice() {
        let gdef = choice_grammar(); // S → a | b
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
    }

    #[test]
    fn test_compile_two_nt() {
        let gdef = two_nt_grammar(); // S → A b, A → a
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(constraint.table.num_states > 0);
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
        let mask = state.mask_view().mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        // Commit "a".
        state
            .commit_token(0);
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        // After "a": "b" allowed, "a" not.
        let mask = state.mask_view().mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        // Commit "b".
        state
            .commit_token(1);
        assert!(state.is_finished(), "should accept after 'ab'");
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
        let mask = state.mask_view().mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed");

        // Commit token "a".
        state
            .commit_token(0);
        assert!(
            state.is_finished(),
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
        let mask = state.mask_view().mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        // Commit "a".
        state
            .commit_token(0);
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        // After "a": "b" allowed, "a" not.
        let mask = state.mask_view().mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        // Commit "b".
        state
            .commit_token(1);
        assert!(state.is_finished(), "should accept after 'ab'");
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
        let mask = state.mask_view().mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        // Commit "a".
        state
            .commit_token(0);
        assert!(!state.is_finished(), "not accepting after 'a'");

        // After "a": "b" should be allowed (A reduced, now need B → b).
        let mask = state.mask_view().mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        // Commit "b".
        state
            .commit_token(1);
        assert!(state.is_finished(), "should accept after 'ab'");
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
        let mask = state.mask_view().mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        // Commit "a".
        state.commit_token(0);

        // After "a": only "b" allowed.
        let mask = state.mask_view().mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        // Commit "b".
        state.commit_token(1);

        // After "ab": only "c" allowed.
        let mask = state.mask_view().mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        // Commit "c".
        state.commit_token(2);
        assert!(state.is_finished(), "should accept after 'abc'");
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
        let mask = state.mask_view().mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        // Commit "a".
        state.commit_token(0);

        // After "a": only "b" allowed (still in A → a • b).
        let mask = state.mask_view().mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        // Commit "b".
        state.commit_token(1);

        // After "ab": "c" should be allowed (A reduced, S → A • c).
        let mask = state.mask_view().mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        // Commit "c".
        state.commit_token(2);
        assert!(state.is_finished(), "should accept after 'abc'");
    }
}
