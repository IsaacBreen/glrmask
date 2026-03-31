//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the existing NWA build → postprocess → determinize → minimize pipeline,
//! but only for L2+ terminal groups.

use std::collections::BTreeMap;

use crate::ds::bitset::BitSet;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::terminal_dwa::{
    build_partition_terminal_nwa, PartitionTerminalNwaBuild, TerminalColoring,
};
use crate::Vocab;

/// Build an L2+-only terminal NWA for the given partition.
///
/// Delegates to the full trie-walk NWA construction pipeline
/// (build → postprocess → determinize → minimize).
pub fn build_l2p_terminal_nwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    partition_index: usize,
    grammar: &AnalyzedGrammar,
) -> PartitionTerminalNwaBuild {
    build_partition_terminal_nwa(
        tokenizer,
        vocab,
        id_map,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        disallowed_follows,
        num_terminals,
        partition_index,
        grammar,
    )
}

