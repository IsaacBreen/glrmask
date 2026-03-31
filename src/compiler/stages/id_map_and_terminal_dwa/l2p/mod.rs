//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the same structure as the pre-partition/path-length code (commit 67146d8):
//! build vocab trie → compute possible_matches → seed root nodes → trie-walk
//! NWA build → postprocess (always_allowed → collapse → disallowed → prune →
//! canonicalize) → determinize → minimize.
//!
//! The only structural difference from the old code is `active_terminals`
//! filtering: terminals not in the L2+ set are skipped during the trie walk.

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::terminal_dwa::TerminalColoring;
use crate::ds::bitset::BitSet;
use crate::Vocab;

/// Build an L2+ id_map and terminal DWA for the given vocab and terminal set.
///
/// Builds its own id_map via `InternalIdMap::build_with_group_filter` (full DFA-
/// based equivalence analysis restricted to L2+ terminal groups). Then builds
/// the terminal DWA using the old-shaped trie-walk NWA pipeline matching the
/// 67146d8 code shape.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_l2p_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<(InternalIdMap, DWA)> {
    if vocab.is_empty() {
        return None;
    }

    let id_map = InternalIdMap::build_with_group_filter(
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        Some(active_terminals),
    );

    let build = crate::compiler::stages::terminal_dwa::build_l2p_partition_terminal_nwa(
        tokenizer,
        vocab,
        &id_map,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        grammar,
        active_terminals,
        0, // partition_index (not meaningful here)
    );

    let nwa = build.nwa?;
    // The NWA came from dwa.to_nwa() inside the L2+ builder (which already
    // did postprocess + det + min). Re-determinizing is essentially a no-op.
    let det = determinize(&nwa).expect("L2+ terminal NWA determinization failed");
    let dwa = minimize(&det);

    Some((id_map, dwa))
}
