//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the same structure as the pre-partition/path-length code (commit 67146d8):
//! build vocab trie → compute possible_matches → seed root nodes → trie-walk
//! NWA build → postprocess → determinize → minimize.
//!
//! The only structural difference from the old code is `active_terminals`
//! filtering: terminals not in the L2+ set are skipped during the trie walk.

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::terminal_dwa::{
    build_l2p_partition_terminal_nwa, PartitionTerminalNwaBuild, TerminalColoring,
};
use crate::Vocab;

/// Build an L2+-only terminal NWA for the given partition.
///
/// Follows the old code shape (67146d8): trie walk → postprocess → det → min.
/// Only difference: `active_terminals` filtering excludes non-L2+ terminals.
pub fn build_l2p_terminal_nwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    partition_index: usize,
) -> PartitionTerminalNwaBuild {
    build_l2p_partition_terminal_nwa(
        tokenizer,
        vocab,
        id_map,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        grammar,
        active_terminals,
        partition_index,
    )
}

