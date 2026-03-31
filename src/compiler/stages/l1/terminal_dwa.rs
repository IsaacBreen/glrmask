//! L1 terminal DWA: fast direct construction for terminals with max path length ≤ 1.
//!
//! Since L1 terminals never co-occur with another terminal in a single token,
//! the DWA can be built by walking each token from each state and checking
//! which terminal matches at the end.

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::terminal_dwa::{
    build_partition_terminal_nwa_l1_direct_filtered, PartitionTerminalNwaBuild,
};
use crate::Vocab;

/// Build an L1-only terminal NWA for the given partition.
///
/// Only terminals marked `true` in `active_terminals` are processed.
/// Typically these are L0/L1 terminals for this partition's vocab.
pub fn build_l1_terminal_nwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
    num_terminals: u32,
    partition_index: usize,
    active_terminals: &[bool],
) -> PartitionTerminalNwaBuild {
    build_partition_terminal_nwa_l1_direct_filtered(
        tokenizer,
        vocab,
        id_map,
        ignore_terminal,
        num_terminals,
        partition_index,
        Some(active_terminals),
    )
}

