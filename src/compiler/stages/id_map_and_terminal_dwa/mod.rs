//! Top-level id_map + terminal DWA builder.
//!
//! Splits the vocab into 3 character-type partitions, builds a per-partition
//! `(InternalIdMap, DWA)` for each via f1, then merges the 3 results via f4
//! to produce the final `(InternalIdMap, DWA)`.

pub(crate) mod classify;
pub(crate) mod grammar_helpers;
pub(crate) mod l1;
pub(crate) mod l2p;
pub(crate) mod merge;
mod monolithic;
pub(crate) mod partition;
pub(crate) mod types;

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::possible_matches::PossibleMatchesByState;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::ds::bitset::BitSet;
use crate::Vocab;

use classify::classify_vocab_char_type;
use types::TerminalColoring;

/// Build the global `(InternalIdMap, DWA)` for the full vocabulary.
///
/// 1. Splits vocab into 3 partitions by leading-byte character type.
/// 2. Builds each partition's `(InternalIdMap, DWA)` in parallel via
///    [`partition::build_partition_id_map_and_terminal_dwa`].
/// 3. Merges the 3 results via [`merge::merge_id_maps_and_terminal_dwas`].
pub(crate) fn build_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> (InternalIdMap, DWA) {
    // Split vocab into 3 partitions by character type.
    let mut partition_entries: [Vec<(u32, Vec<u8>)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (&token_id, bytes) in &vocab.entries {
        let idx = classify_vocab_char_type(bytes) as usize;
        partition_entries[idx].push((token_id, bytes.clone()));
    }
    let sub_vocabs: Vec<Vocab> = partition_entries
        .into_iter()
        .map(|entries| Vocab::new(entries, None))
        .collect();

    // Build each partition in parallel.
    let ((p0, p1), p2) = rayon::join(
        || {
            rayon::join(
                || {
                    partition::build_partition_id_map_and_terminal_dwa(
                        tokenizer,
                        &sub_vocabs[0],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        grammar,
                        disallowed_follows,
                    )
                },
                || {
                    partition::build_partition_id_map_and_terminal_dwa(
                        tokenizer,
                        &sub_vocabs[1],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        grammar,
                        disallowed_follows,
                    )
                },
            )
        },
        || {
            partition::build_partition_id_map_and_terminal_dwa(
                tokenizer,
                &sub_vocabs[2],
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                grammar,
                disallowed_follows,
            )
        },
    );

    // Collect non-None results.
    let mut pairs: Vec<(InternalIdMap, DWA)> = Vec::new();
    if let Some(pair) = p0 {
        pairs.push(pair);
    }
    if let Some(pair) = p1 {
        pairs.push(pair);
    }
    if let Some(pair) = p2 {
        pairs.push(pair);
    }

    if pairs.is_empty() {
        let num_states = tokenizer.num_states() as usize;
        let empty_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: vec![0u32; num_states],
                internal_to_originals: vec![(0..num_states as u32).collect()],
                representative_original_ids: vec![0],
            },
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: Vec::new(),
                internal_to_originals: Vec::new(),
                representative_original_ids: Vec::new(),
            },
        };
        return (empty_map, DWA::new(1, 0));
    }

    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    merge::merge_id_maps_and_terminal_dwas(pairs, num_tokenizer_states, max_token_id)
}

/// Build a terminal DWA for an already-chosen `InternalIdMap`.
///
/// This is the legacy fallback surface used when compile-time overrides bypass
/// the split id_map+terminal_dwa pipeline.
pub(crate) fn build_terminal_dwa_for_existing_id_map(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    monolithic::build_terminal_dwa(grammar, tokenizer, vocab, id_map, ignore_terminal)
}

/// Build a terminal DWA plus possible-matches data for an already-chosen
/// `InternalIdMap`.
pub(crate) fn build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
) -> (DWA, PossibleMatchesByState) {
    monolithic::build_terminal_dwa_with_possible_matches_and_coloring(
        grammar,
        tokenizer,
        vocab,
        id_map,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        disallowed_follows,
    )
}
