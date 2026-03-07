//! Vocabulary preprocessing.
//!
//! Runs each LLM token through the tokenizer DFA from every reachable
//! tokenizer state to discover which (state, terminal) pairs each token
//! produces. Compresses the reachable tokenizer states into compact
//! "TSID" (Tokenizer State ID) indices.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::dfa::DEAD;
use crate::automata::lexer::tokenizer::TokenizerDfa;
use crate::compiler::grammar::ast::TerminalId;

/// Result of vocabulary preprocessing.
#[derive(Debug, Clone)]
pub struct VocabPreprocessing {
    /// `possible_matches[tsid]` = map from terminal_id → token range-set of token positions.
    /// A "token position" is just the token index in the vocab.
    pub possible_matches: Vec<BTreeMap<TerminalId, RangeSetBlaze<u32>>>,
    /// Number of unique TSIDs.
    pub num_tsids: u32,
    /// `state_to_tsid[dfa_state]` = compacted TSID (u32::MAX if unreachable).
    pub state_to_tsid: Vec<u32>,
    /// `tsid_to_state[tsid]` = the DFA state for this TSID.
    pub tsid_to_state: Vec<u32>,
    /// Maximum token index (vocab size - 1), or 0 if empty.
    pub max_token: u32,
}

impl VocabPreprocessing {
    /// Compute vocabulary preprocessing.
    ///
    /// For each reachable tokenizer DFA state (TSID) and each LLM token,
    /// run the token's bytes through the tokenizer and record which
    /// terminals match.
    ///
    /// If `used_terminals` is provided, TSIDs whose start state cannot reach
    /// any terminal in the set are skipped in Phase 3 (their `possible_matches`
    /// entries remain empty). This can dramatically
    /// reduce Phase 3 cost when only a few terminals are actually used by
    /// the grammar's parse table.
    pub fn compute(
        tokenizer: &TokenizerDfa,
        vocab: &Vocab,
        used_terminals: Option<&BTreeSet<TerminalId>>,
    ) -> Self {
        unimplemented!()
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::TokenizerDfa;
    use crate::compiler::grammar::ast::{GrammarDef, Rule, Symbol, TerminalDef};

    #[test]
    fn test_vocab_preprocessing_basic() {
        // Grammar: S → a b
        // Vocab: ["a", "b", "c", "ab"]
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                TerminalDef {
                    id: 0,
                    name: "a".into(),
                    pattern: "a".into(),
                },
                TerminalDef {
                    id: 1,
                    name: "b".into(),
                    pattern: "b".into(),
                },
            ],
        };
        let tok = TokenizerDfa::from_grammar_def(&gdef);
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
            ],
            None,
        );
        let vp = VocabPreprocessing::compute(&tok, &vocab, None);

        assert!(vp.num_tsids >= 1);

        // From TSID 0 (start state):
        // Token 0 ("a") matches terminal 0 ("a")
        // Token 1 ("b") matches terminal 1 ("b")
        // Token 2 ("c") matches nothing
        // Token 3 ("ab") matches neither terminal alone (it's both a then b)
        let tsid_0 = vp.state_to_tsid[tok.start_state() as usize];
        let matches = &vp.possible_matches[tsid_0 as usize];

        // Terminal 0 should match token 0 ("a")
        assert!(matches.get(&0).is_some_and(|rs| rs.contains(0)));
        // Terminal 1 should match token 1 ("b")
        assert!(matches.get(&1).is_some_and(|rs| rs.contains(1)));
        // Token 2 ("c") should not match terminal 0 or 1
        assert!(!matches.get(&0).is_some_and(|rs| rs.contains(2)));
        assert!(!matches.get(&1).is_some_and(|rs| rs.contains(2)));
    }
}
