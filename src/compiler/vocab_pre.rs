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
use crate::automata::weighted::weight::TokenSet;

/// Result of vocabulary preprocessing.
#[derive(Debug, Clone)]
pub struct VocabPreprocessing {
    /// `possible_matches[tsid]` = map from terminal_id → TokenSet of token positions.
    /// A "token position" is just the token index in the vocab.
    pub possible_matches: Vec<BTreeMap<TerminalId, TokenSet>>,
    /// `passthrough_tokens[tsid]` = set of tokens that reach a non-dead
    /// tokenizer state from this TSID but don't match any terminal.
    /// These tokens just advance the tokenizer without triggering parser actions.
    pub passthrough_tokens: Vec<TokenSet>,
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
    /// and `passthrough_tokens` entries remain empty). This can dramatically
    /// reduce Phase 3 cost when only a few terminals are actually used by
    /// the grammar's parse table.
    pub fn compute(
        tokenizer: &TokenizerDfa,
        vocab: &Vocab,
        used_terminals: Option<&BTreeSet<TerminalId>>,
    ) -> Self {
        use std::time::Instant;
        let t_start = Instant::now();

        let num_dfa_states = tokenizer.num_states();
        let vocab_size = vocab.entries.len();
        let max_token = if vocab_size > 0 {
            vocab.entries.iter().map(|(id, _)| *id).max().unwrap_or(0)
        } else {
            0
        };

        // Phase 1: Find all reachable DFA states.
        //
        // When used_terminals is provided, we can use a fast byte-level
        // reachability analysis instead of running full tokens through the DFA.
        // The byte-level analysis over-approximates (some states may be reachable
        // by bytes but not by complete tokens), but extra TSIDs are harmless
        // because Phase 3 and token iteration skip useless TSIDs anyway.
        let t = Instant::now();
        let (reachable, phase1_method) = if used_terminals.is_some() {
            // Fast path: byte-level reachability via DFA transition table.
            let mut visited = vec![false; num_dfa_states as usize];
            let mut frontier = vec![tokenizer.start_state()];
            visited[tokenizer.start_state() as usize] = true;
            while let Some(state) = frontier.pop() {
                for byte in 0..=255u8 {
                    let next = tokenizer.dfa.get_transition(state, byte);
                    if next != DEAD && !(visited[next as usize]) {
                        visited[next as usize] = true;
                        frontier.push(next);
                    }
                }
            }
            let reachable: BTreeSet<u32> = (0..num_dfa_states)
                .filter(|&s| visited[s as usize])
                .collect();
            (reachable, "byte-reachable")
        } else {
            // Full BFS with token execution (original behavior for tests).
            let mut reachable: BTreeSet<u32> = BTreeSet::new();
            reachable.insert(tokenizer.start_state());
            let mut frontier: Vec<u32> = vec![tokenizer.start_state()];
            let mut _iterations = 0u32;
            while !frontier.is_empty() {
                _iterations += 1;
                let mut next_frontier = Vec::new();
                for &start_state in &frontier {
                    for (_token_id, token_bytes) in &vocab.entries {
                        let (end_state, _) = tokenizer.execute(token_bytes, start_state);
                        if end_state != DEAD && reachable.insert(end_state) {
                            next_frontier.push(end_state);
                        }
                    }
                }
                frontier = next_frontier;
            }
            (reachable, "token-BFS")
        };
        eprintln!("[vocab_pre] Phase 1 (reachable): {:.3}s ({} states, {}, {} DFA states)",
            t.elapsed().as_secs_f64(), reachable.len(), phase1_method, num_dfa_states);

        // Phase 2: Assign TSID indices.
        let t = Instant::now();
        let mut state_to_tsid = vec![u32::MAX; num_dfa_states as usize];
        let mut tsid_to_state: Vec<u32> = Vec::new();
        for &state in &reachable {
            let tsid = tsid_to_state.len() as u32;
            state_to_tsid[state as usize] = tsid;
            tsid_to_state.push(state);
        }
        let num_tsids = tsid_to_state.len() as u32;
        eprintln!("[vocab_pre] Phase 2 (TSID):      {:.3}s ({} TSIDs)", t.elapsed().as_secs_f64(), num_tsids);

        // Phase 3: For each TSID, run all tokens and collect matches.
        // Parallelized across TSIDs with rayon for ~8-10x speedup.
        // When used_terminals is provided, skip TSIDs whose start state
        // can never reach any used terminal (reachability check).
        let t = Instant::now();
        let reachable = tokenizer.compute_reachable_terminals();

        // Pre-compute which TSIDs to skip based on terminal reachability.
        let skip_tsid: Vec<bool> = if let Some(used) = used_terminals {
            tsid_to_state.iter().map(|&dfa_state| {
                let state_reachable = &reachable[dfa_state as usize];
                !state_reachable.iter().any(|t| used.contains(t))
            }).collect()
        } else {
            vec![false; num_tsids as usize]
        };
        let skipped_count = skip_tsid.iter().filter(|&&s| s).count();

        #[cfg(feature = "rayon")]
        let (possible_matches, passthrough_tokens) = {
            use rayon::prelude::*;
            let results: Vec<(BTreeMap<TerminalId, TokenSet>, TokenSet)> = tsid_to_state
                .par_iter()
                .enumerate()
                .map(|(tsid_idx, &dfa_state)| {
                    if skip_tsid[tsid_idx] {
                        return (BTreeMap::new(), TokenSet::new());
                    }
                    // Collect (terminal, token_id) pairs for batch TokenSet construction.
                    // Uses zero-allocation callback to avoid BTreeSet per match.
                    let mut pairs: Vec<(TerminalId, u32)> = Vec::new();
                    let mut pt_ids: Vec<u32> = Vec::new();
                    for &(token_id, ref token_bytes) in &vocab.entries {
                        let mut any_matched = false;
                        let end_state = tokenizer.execute_all_matches_cb(token_bytes, dfa_state, |_offset, finalizers| {
                            for &gid in finalizers {
                                any_matched = true;
                                pairs.push((gid as TerminalId, token_id));
                            }
                        });
                        if end_state != DEAD {
                            if let Some(rt) = reachable.get(end_state as usize) {
                                for &reachable_terminal in rt {
                                    // Duplicates handled by sort+dedup below.
                                    pairs.push((reachable_terminal, token_id));
                                }
                            }
                            if !any_matched {
                                pt_ids.push(token_id);
                            }
                        }
                    }
                    // Sort by (terminal, token_id), dedup, then batch-build TokenSets.
                    pairs.sort_unstable();
                    pairs.dedup();
                    let mut pm: BTreeMap<TerminalId, TokenSet> = BTreeMap::new();
                    let mut i = 0;
                    while i < pairs.len() {
                        let terminal = pairs[i].0;
                        let start = i;
                        while i < pairs.len() && pairs[i].0 == terminal {
                            i += 1;
                        }
                        let rs: TokenSet =
                            pairs[start..i].iter().map(|&(_, tid)| tid..=tid).collect();
                        pm.insert(terminal, rs);
                    }
                    // Batch-build passthrough TokenSet.
                    let pt: TokenSet = if !pt_ids.is_empty() {
                        pt_ids.sort_unstable();
                        pt_ids.dedup();
                        pt_ids.iter().map(|&tid| tid..=tid).collect()
                    } else {
                        TokenSet::new()
                    };
                    (pm, pt)
                })
                .collect();
            let mut possible_matches = Vec::with_capacity(num_tsids as usize);
            let mut passthrough_tokens = Vec::with_capacity(num_tsids as usize);
            for (pm, pt) in results {
                possible_matches.push(pm);
                passthrough_tokens.push(pt);
            }
            (possible_matches, passthrough_tokens)
        };

        #[cfg(not(feature = "rayon"))]
        let (possible_matches, passthrough_tokens) = {
            // Collect (terminal, token_id) pairs per TSID, then batch-build TokenSets.
            let mut possible_matches: Vec<BTreeMap<TerminalId, TokenSet>> =
                vec![BTreeMap::new(); num_tsids as usize];
            let mut passthrough_tokens: Vec<TokenSet> =
                vec![TokenSet::new(); num_tsids as usize];
            for (tsid, &dfa_state) in tsid_to_state.iter().enumerate() {
                if skip_tsid[tsid] {
                    continue;
                }
                // Collect (terminal, token_id) pairs for batch TokenSet construction.
                // Uses zero-allocation callback to avoid BTreeSet per match.
                let mut pairs: Vec<(TerminalId, u32)> = Vec::new();
                let mut pt_ids: Vec<u32> = Vec::new();
                for &(token_id, ref token_bytes) in &vocab.entries {
                    let mut any_matched = false;
                    let end_state = tokenizer.execute_all_matches_cb(token_bytes, dfa_state, |_offset, finalizers| {
                        for &gid in finalizers {
                            any_matched = true;
                            pairs.push((gid as TerminalId, token_id));
                        }
                    });
                    if end_state != DEAD {
                        if let Some(rt) = reachable.get(end_state as usize) {
                            for &reachable_terminal in rt {
                                // Duplicates handled by sort+dedup below.
                                pairs.push((reachable_terminal, token_id));
                            }
                        }
                        if !any_matched {
                            pt_ids.push(token_id);
                        }
                    }
                }
                // Sort by (terminal, token_id), dedup, then batch-build TokenSets.
                pairs.sort_unstable();
                pairs.dedup();
                let mut i = 0;
                while i < pairs.len() {
                    let terminal = pairs[i].0;
                    let start = i;
                    while i < pairs.len() && pairs[i].0 == terminal {
                        i += 1;
                    }
                    let rs: TokenSet =
                        pairs[start..i].iter().map(|&(_, tid)| tid..=tid).collect();
                    possible_matches[tsid].insert(terminal, rs);
                }
                // Batch-build passthrough TokenSet.
                if !pt_ids.is_empty() {
                    pt_ids.sort_unstable();
                    pt_ids.dedup();
                    passthrough_tokens[tsid] = pt_ids.iter().map(|&tid| tid..=tid).collect();
                }
            }
            (possible_matches, passthrough_tokens)
        };

        eprintln!("[vocab_pre] Phase 3 (matches):   {:.3}s (skipped {}/{} TSIDs)", t.elapsed().as_secs_f64(), skipped_count, num_tsids);
        eprintln!("[vocab_pre] Total:                {:.3}s", t_start.elapsed().as_secs_f64());

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
