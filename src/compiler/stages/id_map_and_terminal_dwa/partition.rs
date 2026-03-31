//! Per-partition terminal DWA builder (f1).
//!
//! Given a (sub-)vocab and shared parameters, classifies terminals into L1 and
//! L2+, builds separate `(InternalIdMap, DWA)` pairs for each via f2 and f3,
//! then merges them via f4 to produce a single `(InternalIdMap, DWA)`.

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::terminal_dwa::{
    classify_terminal_path_lengths, TerminalColoring, TerminalPathLength,
};
use crate::ds::bitset::BitSet;
use crate::Vocab;

/// Build an id_map and terminal DWA for a single vocab partition.
///
/// 1. Classifies terminal path lengths → L1 / L2+ masks.
/// 2. Builds L1 and L2+ `(InternalIdMap, DWA)` pairs in parallel.
/// 3. Merges them via f4.
/// 4. Returns a single `(InternalIdMap, DWA)`.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_partition_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<(InternalIdMap, DWA)> {
    if vocab.is_empty() {
        return None;
    }

    let num_terminals = grammar.num_terminals as u32;

    // Classify terminal path lengths to determine L1 vs L2+ split.
    let terminal_path_lengths =
        classify_terminal_path_lengths(tokenizer, vocab, disallowed_follows, num_terminals);

    let mut l1_mask = vec![false; num_terminals as usize];
    let mut l2p_mask = vec![false; num_terminals as usize];
    let mut has_l1 = false;
    let mut has_l2p = false;
    for (i, len) in terminal_path_lengths.iter().enumerate() {
        match len {
            TerminalPathLength::Zero | TerminalPathLength::One => {
                l1_mask[i] = true;
                has_l1 = true;
            }
            TerminalPathLength::TwoPlus => {
                l2p_mask[i] = true;
                has_l2p = true;
            }
        }
    }

    // Build L1 and L2+ terminal DWAs in parallel.
    let (l1_result, l2p_result) = rayon::join(
        || {
            if has_l1 {
                super::l1::build_l1_id_map_and_terminal_dwa(
                    tokenizer,
                    vocab,
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    grammar,
                    &l1_mask,
                )
            } else {
                None
            }
        },
        || {
            if has_l2p {
                super::l2p::build_l2p_id_map_and_terminal_dwa(
                    tokenizer,
                    vocab,
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    grammar,
                    &l2p_mask,
                    disallowed_follows,
                )
            } else {
                None
            }
        },
    );

    // Collect non-None results and merge.
    let mut pairs: Vec<(InternalIdMap, DWA)> = Vec::new();
    if let Some(l1) = l1_result {
        pairs.push(l1);
    }
    if let Some(l2p) = l2p_result {
        pairs.push(l2p);
    }

    if pairs.is_empty() {
        return None;
    }

    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    Some(super::merge::merge_id_maps_and_terminal_dwas(
        pairs,
        num_tokenizer_states,
        max_token_id,
    ))
}
