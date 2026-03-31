//! L2+ equivalence analysis.
//!
//! For L2+ terminals (max path length ≥ 2), we use the full combined
//! equivalence analysis from the shared equivalence_analysis module.
//! This module provides a convenience wrapper that builds an InternalIdMap
//! using the full analysis pipeline, optionally with group filtering.

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::ds::bitset::BitSet;
use crate::Vocab;
use std::collections::BTreeMap;

/// Build an id_map using full combined equivalence analysis.
///
/// When `active_groups` is provided, only the specified terminal groups
/// contribute to vocab equivalence discrimination.
pub fn build_l2p_id_map(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
) -> InternalIdMap {
    InternalIdMap::build_with_group_filter(
        tokenizer, vocab, disallowed_follows, ignore_terminal, active_groups,
    )
}
