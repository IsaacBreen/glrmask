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

use crate::compiler::glr::grammar::GlrGrammar;
use crate::compiler::glr::table::GlrTable;
use crate::compiler::grammar_def::GrammarDef;
use crate::compiler::parser_dwa::build_parser_dwa;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::compiler::vocab_pre::VocabPreprocessing;
use crate::runtime::Constraint;
use crate::Vocab;

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
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.num_dwa_states() > 0);
        assert!(constraint.num_tsids() > 0);
    }

    #[test]
    fn test_compile_choice() {
        let gdef = choice_grammar(); // S → a | b
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.num_dwa_states() > 0);
    }

    #[test]
    fn test_compile_two_nt() {
        let gdef = two_nt_grammar(); // S → A b, A → a
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.num_dwa_states() > 0);
        assert!(constraint.num_parser_states() > 0);
    }
}
