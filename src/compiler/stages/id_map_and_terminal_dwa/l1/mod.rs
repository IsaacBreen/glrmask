//! L1 terminal DWA: fast direct construction for terminals with max path length ≤ 1.
//!
//! Since L1 terminals never co-occur with another terminal in a single token,
//! the DWA can be built by walking each token from each state and checking
//! which terminal matches at the end.

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize_fast;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::terminal_dwa::TerminalColoring;
use crate::Vocab;

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// Builds its own id_map via `InternalIdMap::build_l1` (fast fingerprint-based
/// equivalence, no DFA walk). Then builds the terminal DWA using the L1 direct
/// path that walks (token, state) pairs without the full NWA trie-walk pipeline.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_l1_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _terminal_coloring: &TerminalColoring,
    _use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
) -> Option<(InternalIdMap, DWA)> {
    if vocab.is_empty() {
        return None;
    }

    let id_map = InternalIdMap::build_l1(tokenizer, vocab);
    let num_terminals = grammar.num_terminals as u32;

    let build = crate::compiler::stages::terminal_dwa::build_partition_terminal_nwa_l1_direct_filtered(
        tokenizer,
        vocab,
        &id_map,
        ignore_terminal,
        num_terminals,
        0, // partition_index (not meaningful here)
        Some(active_terminals),
    );

    let nwa = build.nwa?;
    // The NWA came from dwa.to_nwa() inside the L1 direct builder, so
    // re-determinizing is essentially a no-op. We need a DWA though.
    let det = determinize(&nwa).expect("L1 terminal NWA determinization failed");
    let dwa = minimize_fast(&det);

    Some((id_map, dwa))
}
