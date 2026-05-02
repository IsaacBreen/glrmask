use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

use super::RuntimePossibleMatchesByTerminal;

#[derive(Debug, Clone)]
pub(crate) struct BruteForcePossibleMatches {
    /// Possible matches keyed by terminal. Each Weight maps original tokenizer
    /// start states to token sets in `id_map`'s internal token space.
    pub(crate) possible_matches: RuntimePossibleMatchesByTerminal,
    /// Original vocab token id -> possible-matches internal token id.
    ///
    /// This is intentionally independent of parser-DWA vocab compaction. A later
    /// pipeline stage must merge/refine this id map with the parser-DWA id map
    /// before possible_matches and parser-DWA weights can be used together.
    pub(crate) id_map: ManyToOneIdMap,
}

/// Compute possible matches by direct simulation.
///
/// Mathematical definition: for every vocab token and every original tokenizer
/// start state, scan the token bytes with all terminals as terminals of
/// interest. For each terminal that finalizes while scanning those bytes, add
/// `(terminal, start_state, token)` to the result.
///
/// This is deliberately simple and useful as a correctness reference. It is not
/// intended as the default large-vocab implementation: the literal complexity is
/// `num_tokens * num_tokenizer_states * scan(token_bytes)`.
pub(crate) fn compute_possible_matches_bruteforce(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> BruteForcePossibleMatches {
    let id_map = build_vocab_token_id_map(vocab);
    let all_terminals = BitSet::all(tokenizer.num_terminals as usize);
    let mut terminal_state_tokens: BTreeMap<TerminalID, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();

    for (&original_token_id, bytes) in &vocab.entries {
        let Some(&internal_token_id) = id_map.original_to_internal.get(original_token_id as usize)
        else {
            continue;
        };
        if internal_token_id == u32::MAX {
            continue;
        }

        for start_state in 0..tokenizer.num_states() {
            let (matched_terminals, _end_state) =
                tokenizer.scan_terminal_matches_from_state(bytes, start_state, &all_terminals);

            for terminal in matched_terminals.iter() {
                terminal_state_tokens
                    .entry(terminal as TerminalID)
                    .or_default()
                    .entry(start_state)
                    .or_default()
                    .push(internal_token_id);
            }
        }
    }

    let possible_matches = terminal_state_tokens
        .into_iter()
        .map(|(terminal, state_tokens)| {
            let weight_entries = state_tokens.into_iter().map(|(state, mut token_ids)| {
                token_ids.sort_unstable();
                token_ids.dedup();
                let token_set =
                    RangeSetBlaze::from_iter(token_ids.into_iter().map(|id| id..=id));
                (state, token_set)
            });
            (terminal, Weight::from_per_tsid_token_sets(weight_entries))
        })
        .collect();

    BruteForcePossibleMatches {
        possible_matches,
        id_map,
    }
}

fn build_vocab_token_id_map(vocab: &Vocab) -> ManyToOneIdMap {
    let max_original_token_id = vocab.entries.keys().next_back().copied().unwrap_or(0);
    let mut original_to_internal = vec![u32::MAX; max_original_token_id as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(vocab.entries.len());
    let mut representative_original_ids = Vec::with_capacity(vocab.entries.len());

    for &original_token_id in vocab.entries.keys() {
        let internal_token_id = internal_to_originals.len() as u32;
        original_to_internal[original_token_id as usize] = internal_token_id;
        internal_to_originals.push(vec![original_token_id]);
        representative_original_ids.push(original_token_id);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::bytes;
    use crate::compiler::compile::build_tokenizer_from_exprs;

    fn internal_id(result: &BruteForcePossibleMatches, original_token_id: u32) -> u32 {
        result.id_map.original_to_internal[original_token_id as usize]
    }

    #[test]
    fn brute_force_records_terminal_state_token_relation() {
        let tokenizer =
            build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"ab"), bytes(b"b")]);
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"ab".to_vec()),
                (30, b"b".to_vec()),
                (40, b"c".to_vec()),
            ],
            None,
        );

        let result = compute_possible_matches_bruteforce(&tokenizer, &vocab);
        let start = tokenizer.start_state();
        let after_a = tokenizer.run(b"a");

        let token_a = internal_id(&result, 10);
        let token_ab = internal_id(&result, 20);
        let token_b = internal_id(&result, 30);
        let token_c = internal_id(&result, 40);

        // From start state, terminal 0 ("a") covers tokens "a" and "ab"
        // (both finalize while scanning from state 0).
        let terminal_a_from_start = result.possible_matches[&0].tokens_for_tsid(start);
        assert!(terminal_a_from_start.contains(token_a));
        assert!(terminal_a_from_start.contains(token_ab));
        assert!(!terminal_a_from_start.contains(token_b));
        assert!(!terminal_a_from_start.contains(token_c));

        // From start state, terminal 1 ("ab") covers token "ab".
        let terminal_ab_from_start = result.possible_matches[&1].tokens_for_tsid(start);
        assert!(terminal_ab_from_start.contains(token_ab));
        assert!(!terminal_ab_from_start.contains(token_a));

        // From state after "a", terminal 1 ("ab") covers token "b"
        // (starting from after_a and scanning "b" finalizes terminal 1).
        let terminal_ab_from_after_a = result.possible_matches[&1].tokens_for_tsid(after_a);
        assert!(terminal_ab_from_after_a.contains(token_b));

        // From start state, terminal 2 ("b") covers token "b".
        let terminal_b_from_start = result.possible_matches[&2].tokens_for_tsid(start);
        assert!(terminal_b_from_start.contains(token_b));
        assert!(!terminal_b_from_start.contains(token_a));
    }

    #[test]
    fn brute_force_id_map_is_compact_and_deterministic_in_sorted_order() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"x")]);
        let vocab = Vocab::new(
            vec![
                (100, b"x".to_vec()),
                (7, b"x".to_vec()),
                (42, b"y".to_vec()),
            ],
            None,
        );

        let result = compute_possible_matches_bruteforce(&tokenizer, &vocab);

        // Vocab entries are stored in a BTreeMap, so iteration is by sorted
        // key order: 7, 42, 100.
        assert_eq!(result.id_map.original_to_internal[7], 0);
        assert_eq!(result.id_map.original_to_internal[42], 1);
        assert_eq!(result.id_map.original_to_internal[100], 2);
        assert_eq!(
            result.id_map.internal_to_originals,
            vec![vec![7], vec![42], vec![100]]
        );
        assert_eq!(
            result.id_map.representative_original_ids,
            vec![7, 42, 100]
        );
    }

    #[test]
    fn brute_force_omits_tokens_that_match_no_terminal() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a")]);
        let vocab = Vocab::new(vec![(0, b"z".to_vec())], None);

        let result = compute_possible_matches_bruteforce(&tokenizer, &vocab);

        // No terminal matches "z", so all maps should be empty.
        assert!(result.possible_matches.is_empty());
        // The id_map still gives the token an internal id.
        assert_eq!(result.id_map.original_to_internal[0], 0);
    }
}
