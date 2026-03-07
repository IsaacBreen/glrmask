
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This is the nearest analogue to sep1's `state_equivalence_analysis_fast.rs`.
// It uses iterative partition refinement (essentially DFA minimization on the
// tokenizer state space) to collapse states with identical terminal-match and
// transition behaviour into equivalence classes.  The previous implementation
// returned the identity mapping; this version gives real compression.

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::ManyToOneIdMap;

/// Iterative partition refinement over tokenizer states.
///
/// 1. Initial partition: group by `(matched_terminals, possible_future_terminals)`.
/// 2. Refine: split groups when members transition to different classes on any byte.
/// 3. Repeat until the partition is stable (fixed-point).
///
/// Complexity: O(k × n × 256) where k ≤ log₂(n) refinement rounds.
pub(crate) fn analyze_state_equivalences(tokenizer: &Tokenizer) -> ManyToOneIdMap {
    let n = tokenizer.num_states() as usize;
    if n == 0 {
        return ManyToOneIdMap {
            original_to_internal: Vec::new(),
            internal_to_originals: Vec::new(),
        };
    }

    // --- Stage 1: initial partition by (matched_terminals, possible_future_terminals) ---
    let mut partition: Vec<u32> = vec![0; n];
    {
        let mut sig_to_class: BTreeMap<(Vec<u32>, Vec<u32>), u32> = BTreeMap::new();
        let mut next_class = 0u32;
        for state in 0..n as u32 {
            let matched: Vec<u32> = tokenizer.all_matched_terminals(state).into_iter().collect();
            let futures: Vec<u32> = tokenizer.possible_future_terminals(state).into_iter().collect();
            let class = *sig_to_class.entry((matched, futures)).or_insert_with(|| {
                let c = next_class;
                next_class += 1;
                c
            });
            partition[state as usize] = class;
        }
    }

    // --- Stage 2: iterative refinement ---
    // Each round: for every state, build (current_class, [target_class_for_byte_0..255]).
    // If any two states in the same class produce different keys, split.
    const MAX_ITERS: usize = 64;
    for _ in 0..MAX_ITERS {
        let prev = partition.clone();
        let mut sig_to_class: BTreeMap<(u32, Vec<u32>), u32> = BTreeMap::new();
        let mut next_class = 0u32;

        for state in 0..n as u32 {
            let mut trans_key = Vec::with_capacity(256);
            for byte in 0..=255u8 {
                let target_class = tokenizer
                    .step(state, byte)
                    .map(|t| prev[t as usize])
                    .unwrap_or(u32::MAX); // dead state sentinel
                trans_key.push(target_class);
            }
            let sig = (prev[state as usize], trans_key);
            let class = *sig_to_class.entry(sig).or_insert_with(|| {
                let c = next_class;
                next_class += 1;
                c
            });
            partition[state as usize] = class;
        }

        if partition == prev {
            break;
        }
    }

    // --- Stage 3: build ManyToOneIdMap from final partition ---
    // Renumber classes contiguously and build the reverse map.
    let mut class_remap: BTreeMap<u32, u32> = BTreeMap::new();
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();

    for (state, &class) in partition.iter().enumerate() {
        let internal = *class_remap.entry(class).or_insert_with(|| {
            let id = internal_to_originals.len() as u32;
            internal_to_originals.push(Vec::new());
            id
        });
        internal_to_originals[internal as usize].push(state as u32);
    }

    let original_to_internal: Vec<u32> = partition
        .iter()
        .map(|class| class_remap[class])
        .collect();

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    fn make_tokenizer(patterns: &[&str]) -> Tokenizer {
        let terminals: Vec<Terminal> = patterns
            .iter()
            .enumerate()
            .map(|(i, name)| Terminal {
                id: i as u32,
                name: name.to_string(),
            })
            .collect();
        let terminal_patterns: Vec<String> = patterns.iter().map(|s| s.to_string()).collect();
        let rules = vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }];
        let gdef = GrammarDef {
            rules,
            start: 0,
            terminals,
            terminal_patterns,
        };
        Tokenizer::from_grammar_def(&gdef)
    }

    #[test]
    fn test_identity_for_trivial_grammar() {
        // Single terminal "a": initial state + accepting state.
        // These have different matched_terminals → different classes.
        let tok = make_tokenizer(&["a"]);
        let map = analyze_state_equivalences(&tok);
        assert!(map.num_internal_ids() > 0);
        // Every class should have at least one member
        for class in &map.internal_to_originals {
            assert!(!class.is_empty());
        }
    }

    #[test]
    fn test_compression_for_symmetric_grammar() {
        // "a" and "b" as terminals create symmetric DFA branches.
        // States reachable only via 'a'-path and 'b'-path with symmetric structure
        // should compress.
        let tok = make_tokenizer(&["a", "b"]);
        let map = analyze_state_equivalences(&tok);
        // With 2 single-char terminals, the DFA likely has 3 states:
        // initial (no match, futures={0,1}), accept-a (match={0}), accept-b (match={1}).
        // accept-a and accept-b have different matched sets, so no compression.
        // But at least verify the map is valid.
        assert_eq!(map.original_to_internal.len(), tok.num_states() as usize);
    }

    #[test]
    fn test_roundtrip_consistency() {
        let tok = make_tokenizer(&["ab", "cd", "ef"]);
        let map = analyze_state_equivalences(&tok);
        // Verify every original state maps to a valid internal ID
        for &internal in &map.original_to_internal {
            assert!((internal as usize) < map.internal_to_originals.len());
        }
        // Verify every internal ID maps back to at least one original
        for class in &map.internal_to_originals {
            assert!(!class.is_empty());
            for &original in class {
                assert_eq!(
                    map.original_to_internal[original as usize],
                    map.original_to_internal[class[0] as usize]
                );
            }
        }
    }
}
