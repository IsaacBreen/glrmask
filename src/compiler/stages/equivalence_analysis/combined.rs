#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::equivalence_analysis::state_analysis::analyze_state_equivalences;

pub(crate) fn analyze_equivalences(tokenizer: &Tokenizer, vocab: &Vocab) -> InternalIdMap {
    let state_map = analyze_state_equivalences(tokenizer);
    let vocab_map = analyze_vocab_equivalences_combined(tokenizer, vocab, &state_map);
    InternalIdMap {
        tokenizer_states: state_map,
        vocab_tokens: vocab_map,
    }
}

/// Compute vocab equivalence by simulating each token through the tokenizer
/// from every representative state. Two tokens are equivalent if they produce
/// identical behavior (same terminal matches at each byte position, same
/// end-state equivalence class) across all representative states.
fn analyze_vocab_equivalences_combined(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    state_map: &ManyToOneIdMap,
) -> ManyToOneIdMap {
    let max_token_id = vocab
        .entries
        .iter()
        .map(|(token_id, _)| *token_id)
        .max()
        .unwrap_or(0);

    // Collect representative states (one per state equivalence class).
    let representative_states: Vec<u32> = state_map
        .internal_to_originals
        .iter()
        .filter_map(|originals| originals.first().copied())
        .collect();

    // For each vocab token, compute a behavioral signature across all representative states.
    let mut sig_to_internal: BTreeMap<Vec<u64>, u32> = BTreeMap::new();
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];

    for (&token_id, bytes) in &vocab.entries {
        let sig = compute_token_signature(tokenizer, bytes, &representative_states, &state_map.original_to_internal);

        let internal_id = if let Some(&existing) = sig_to_internal.get(&sig) {
            existing
        } else {
            let next = internal_to_originals.len() as u32;
            sig_to_internal.insert(sig, next);
            internal_to_originals.push(Vec::new());
            next
        };

        if let Some(slot) = original_to_internal.get_mut(token_id as usize) {
            *slot = internal_id;
        }
        internal_to_originals[internal_id as usize].push(token_id);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
    }
}

/// Compute a behavioral signature for a token across all representative states.
///
/// For each representative state, we simulate the token byte-by-byte and record:
/// - At each byte position: a hash of the matched terminals at that state
/// - The final state's equivalence class (or a sentinel for dead/no-transition)
///
/// The concatenation of per-state hashes forms the full signature.
///
/// Critically, this also captures "continuation after match" semantics:
/// when a terminal match occurs at an intermediate position within the
/// token, the NWA builder resets to the initial tokenizer state for the
/// remaining suffix. So we hash suffix behavior from the initial state
/// at each match point to distinguish tokens like " a" vs " b" that have
/// identical raw DFA walks but different suffix continuations.
fn compute_token_signature(
    tokenizer: &Tokenizer,
    token_bytes: &[u8],
    representative_states: &[u32],
    state_to_class: &[u32],
) -> Vec<u64> {
    let num_states = representative_states.len();
    let token_len = token_bytes.len();

    // Pre-compute suffix hashes from each representative state for each suffix position.
    // suffix_hashes[pos][state_idx] = hash of running bytes[pos..] from representative_states[state_idx].
    let suffix_hashes = precompute_suffix_hashes(tokenizer, token_bytes, representative_states, state_to_class);

    // Build the main per-state signature, incorporating suffix continuation info at match points.
    let mut signature = Vec::with_capacity(num_states);

    for (state_idx, &start_state) in representative_states.iter().enumerate() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        let mut state = start_state;
        let mut alive = true;

        for (pos, &byte) in token_bytes.iter().enumerate() {
            if !alive {
                u32::MAX.hash(&mut hasher);
                continue;
            }
            match tokenizer.step(state, byte) {
                Some(next) => {
                    state = next;
                    let matched = tokenizer.all_matched_terminals(state);
                    let has_match = !matched.is_empty();
                    for &tid in &matched {
                        tid.hash(&mut hasher);
                    }
                    // Separator
                    u32::MAX.hash(&mut hasher);

                    // At each match point, hash the suffix continuation from every
                    // representative state. This captures what happens after the
                    // NWA builder resets to the initial state for the remaining bytes.
                    if has_match && pos + 1 < token_len {
                        for suffix_state_hash in &suffix_hashes[pos + 1] {
                            suffix_state_hash.hash(&mut hasher);
                        }
                    }
                }
                None => {
                    alive = false;
                    u32::MAX.hash(&mut hasher);
                }
            }
        }

        // Hash the final state's equivalence class
        if alive {
            let final_class = state_to_class
                .get(state as usize)
                .copied()
                .unwrap_or(u32::MAX);
            final_class.hash(&mut hasher);
            let futures = tokenizer.possible_future_terminals(state);
            for &tid in &futures {
                tid.hash(&mut hasher);
            }
        } else {
            (u32::MAX - 1).hash(&mut hasher);
        }

        signature.push(hasher.finish());
    }

    signature
}

/// Pre-compute a hash for running each suffix of the token from each representative state.
/// Returns suffix_hashes[pos][state_idx] for pos in 0..=token_len.
fn precompute_suffix_hashes(
    tokenizer: &Tokenizer,
    token_bytes: &[u8],
    representative_states: &[u32],
    state_to_class: &[u32],
) -> Vec<Vec<u64>> {
    let token_len = token_bytes.len();
    let mut result = Vec::with_capacity(token_len + 1);

    for start_pos in 0..=token_len {
        let suffix = &token_bytes[start_pos..];
        let mut state_hashes = Vec::with_capacity(representative_states.len());

        for &start_state in representative_states {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            let mut state = start_state;
            let mut alive = true;

            for &byte in suffix {
                if !alive {
                    u32::MAX.hash(&mut hasher);
                    continue;
                }
                match tokenizer.step(state, byte) {
                    Some(next) => {
                        state = next;
                        let matched = tokenizer.all_matched_terminals(state);
                        for &tid in &matched {
                            tid.hash(&mut hasher);
                        }
                        u32::MAX.hash(&mut hasher);
                    }
                    None => {
                        alive = false;
                        u32::MAX.hash(&mut hasher);
                    }
                }
            }

            if alive {
                let final_class = state_to_class
                    .get(state as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                final_class.hash(&mut hasher);
                let futures = tokenizer.possible_future_terminals(state);
                for &tid in &futures {
                    tid.hash(&mut hasher);
                }
            } else {
                (u32::MAX - 1).hash(&mut hasher);
            }

            state_hashes.push(hasher.finish());
        }

        result.push(state_hashes);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile::build_tokenizer;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    #[test]
    fn test_internal_id_map_shape() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let tok = build_tokenizer(&gdef);
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
            ],
            None,
        );
        let id_map = analyze_equivalences(&tok, &vocab);

        assert!(id_map.num_tsids() >= 1);
        assert_eq!(id_map.max_token_id(), 2);
    }

        #[test]
        fn test_json_schema_equivalence_classes_port() {
            use crate::import::json_schema::json_schema_to_grammar;
            // Schema: {"type": "object", "properties": {"name": {"type": "string"}}}
            let schema = r#"{
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }"#;
            let grammar = json_schema_to_grammar(schema).expect("Schema should convert");
            let tok = build_tokenizer(&grammar);
            let vocab_strs = vec![
                "{", "}", "\"", ":", ",", "n", "a", "m", "e", "s", "t", "r", "i", "g", "{\"", "\":"
            ];
            let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs.iter().enumerate().map(|(i, s)| (i as u32, s.as_bytes().to_vec())).collect();
            let vocab = Vocab::new(vocab_entries, None);
            let id_map = analyze_equivalences(&tok, &vocab);
            let classes = &id_map.vocab_tokens.internal_to_originals;
            // Print for debugging
            for (i, class) in classes.iter().enumerate() {
                let content: Vec<&str> = class.iter().map(|&idx| vocab_strs[idx as usize]).collect();
                println!("  Class {}: {:?}", i, content);
            }
            // Combined state+vocab equivalence analysis groups tokens by their
            // behavior across all tokenizer states. Tokens "i" and "g" behave
            // identically (both are single-char string characters with the same
            // DFA transitions from every state), so they are merged into one class.
            let expected: Vec<Vec<usize>> = vec![
                vec![0],           // "{"
                vec![1],           // "}"
                vec![2],           // "\""
                vec![3],           // ":"
                vec![4],           // ","
                vec![5],           // "n"
                vec![6],           // "a"
                vec![7],           // "m"
                vec![8],           // "e"
                vec![9],           // "s"
                vec![10],          // "t"
                vec![11],          // "r"
                vec![12, 13],      // "i", "g" - equivalent string chars
                vec![14],          // "{\""
                vec![15],          // "\":"
            ];
            let mut expected_sorted: Vec<Vec<usize>> = expected.iter().map(|c| { let mut v = c.clone(); v.sort(); v }).collect();
            expected_sorted.sort();
            let mut actual_sorted: Vec<Vec<usize>> = classes.iter().map(|c| { let mut v: Vec<usize> = c.iter().map(|&id| id as usize).collect(); v.sort(); v }).collect();
            actual_sorted.sort();
            assert_eq!(actual_sorted, expected_sorted,
                "Equivalence classes don't match expected!\n\
                 Expected: {:?}\n\
                 Actual:   {:?}",
                expected_sorted, actual_sorted);
        }

        #[test]
        fn test_json_schema_equivalence_classes_simpler_port() {
            // Simple EBNF: root ::= '{' '}'
            let ebnf = "root ::= '{' '}'";
            let grammar = crate::import::ebnf::parse_ebnf(ebnf).expect("Grammar should build");
            let tok = build_tokenizer(&grammar);
            let vocab_strs = vec!["{", "}"];
            let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs.iter().enumerate().map(|(i, s)| (i as u32, s.as_bytes().to_vec())).collect();
            let vocab = Vocab::new(vocab_entries, None);
            let id_map = analyze_equivalences(&tok, &vocab);
            let classes = &id_map.vocab_tokens.internal_to_originals;
            for (i, class) in classes.iter().enumerate() {
                let content: Vec<&str> = class.iter().map(|&idx| vocab_strs[idx as usize]).collect();
                println!("  Class {}: {:?}", i, content);
            }
            let expected: Vec<Vec<usize>> = vec![vec![0], vec![1]];
            let mut expected_sorted: Vec<Vec<usize>> = expected.iter().map(|c| { let mut v = c.clone(); v.sort(); v }).collect();
            expected_sorted.sort();
            let mut actual_sorted: Vec<Vec<usize>> = classes.iter().map(|c| { let mut v: Vec<usize> = c.iter().map(|&id| id as usize).collect(); v.sort(); v }).collect();
            actual_sorted.sort();
            assert_eq!(actual_sorted, expected_sorted,
                "Equivalence classes don't match expected!\n\
                 Expected: {:?}\n\
                 Actual:   {:?}",
                expected_sorted, actual_sorted);
        }
}
