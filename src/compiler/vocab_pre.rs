//! Vocabulary preprocessing.
//!
//! Runs each LLM token through the tokenizer DFA from every reachable
//! tokenizer state to discover which (state, terminal) pairs each token
//! produces. Compresses the reachable tokenizer states into compact
//! "TSID" (Tokenizer State ID) indices.

use std::collections::{BTreeMap, BTreeSet};

use crate::Vocab;
use crate::automata::dfa::DEAD;
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::ds::rangeset::RangeSet;

/// Result of vocabulary preprocessing.
#[derive(Debug, Clone)]
pub struct VocabPreprocessing {
    /// `possible_matches[tsid]` = map from terminal_id → RangeSet of token positions.
    /// A "token position" is just the token index in the vocab.
    pub possible_matches: Vec<BTreeMap<TerminalId, RangeSet>>,
    /// `passthrough_tokens[tsid]` = RangeSet of tokens that reach a non-dead
    /// tokenizer state from this TSID but don't match any terminal.
    /// These tokens just advance the tokenizer without triggering parser actions.
    pub passthrough_tokens: Vec<RangeSet>,
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
    pub fn compute(tokenizer: &TokenizerDfa, vocab: &Vocab) -> Self {
        let num_dfa_states = tokenizer.num_states();
        let vocab_size = vocab.entries.len();
        let max_token = if vocab_size > 0 {
            vocab.entries.iter().map(|(id, _)| *id).max().unwrap_or(0)
        } else {
            0
        };

        // Phase 1: Find all reachable DFA states.
        // Start from state 0 (initial state), then find all states reachable
        // after running any token's bytes from any already-reachable state.
        let mut reachable: BTreeSet<u32> = BTreeSet::new();
        reachable.insert(tokenizer.start_state());

        // Fixed-point: keep discovering new end states.
        let mut changed = true;
        while changed {
            changed = false;
            let current_reachable: Vec<u32> = reachable.iter().copied().collect();
            for &start_state in &current_reachable {
                for (_token_id, token_bytes) in &vocab.entries {
                    let (end_state, _) = tokenizer.execute(token_bytes, start_state);
                    if end_state != DEAD && reachable.insert(end_state) {
                        changed = true;
                    }
                }
            }
        }

        // Phase 2: Assign TSID indices.
        let mut state_to_tsid = vec![u32::MAX; num_dfa_states as usize];
        let mut tsid_to_state: Vec<u32> = Vec::new();
        for &state in &reachable {
            let tsid = tsid_to_state.len() as u32;
            state_to_tsid[state as usize] = tsid;
            tsid_to_state.push(state);
        }
        let num_tsids = tsid_to_state.len() as u32;

        // Phase 3: For each TSID, run all tokens and collect matches.
        //
        // Use execute_all_matches to find ALL intermediate terminal matches,
        // not just the final-state match. This is critical because:
        // 1. Multi-terminal tokens (e.g., "[]") match terminal "[" at byte 1,
        //    then die. The commit function handles this by restarting the
        //    tokenizer after each match.
        // 2. Prefix tokens (e.g., "tr" for "true") don't match any terminal
        //    but their end state can reach valid terminals.
        let reachable = tokenizer.compute_reachable_terminals();
        let mut possible_matches: Vec<BTreeMap<TerminalId, RangeSet>> =
            vec![BTreeMap::new(); num_tsids as usize];
        let mut passthrough_tokens: Vec<RangeSet> =
            vec![RangeSet::new(); num_tsids as usize];

        for (tsid, &dfa_state) in tsid_to_state.iter().enumerate() {
            for &(token_id, ref token_bytes) in &vocab.entries {
                let result = tokenizer.execute_all_matches(token_bytes, dfa_state);

                // Collect ALL terminals matched at ANY intermediate position.
                // The commit function tries all matches, so a token is valid
                // if ANY of its matches produce a valid parser continuation.
                let mut all_matched = BTreeSet::new();
                for (_offset, terminals) in &result.matches {
                    for &terminal in terminals {
                        all_matched.insert(terminal);
                        possible_matches[tsid]
                            .entry(terminal)
                            .or_default()
                            .insert(token_id);
                    }
                }

                // Record prefix matches: the token reaches a non-dead end
                // state that can eventually lead to specific terminals.
                if result.end_state != DEAD {
                    if let Some(rt) = reachable.get(result.end_state as usize) {
                        for &reachable_terminal in rt {
                            if !all_matched.contains(&reachable_terminal) {
                                possible_matches[tsid]
                                    .entry(reachable_terminal)
                                    .or_default()
                                    .insert(token_id);
                            }
                        }
                    }
                    if all_matched.is_empty() {
                        passthrough_tokens[tsid].insert(token_id);
                    }
                }
            }
        }

        Self {
            possible_matches,
            passthrough_tokens,
            num_tsids,
            state_to_tsid,
            tsid_to_state,
            max_token,
        }
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar_def::{GrammarDef, Rule, Symbol, TerminalDef};
    use crate::compiler::tokenizer_dfa::TokenizerDfa;

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
        let vp = VocabPreprocessing::compute(&tok, &vocab);

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
