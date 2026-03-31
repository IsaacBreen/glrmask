//! L1 terminal DWA: fast direct construction for terminals with max path length ≤ 1.
//!
//! Since L1 terminals never co-occur with another terminal in a single token,
//! the DWA can be built by walking each token from each state and checking
//! which terminal matches at the end. No full NWA trie-walk pipeline needed.

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::terminal_dwa::{self, TerminalColoring};
use crate::Vocab;

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// 1. Build id_map via `InternalIdMap::build_l1` (fast fingerprint-based equiv).
/// 2. Build L1 terminal DWA via the direct walk path (no trie-walk NWA).
///
/// Returns `None` if the vocab is empty or no terminal matches exist.
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

    // 1. Build L1 id_map (fast fingerprint-based equivalence, no DFA walk).
    let id_map = InternalIdMap::build_l1(tokenizer, vocab);

    // 2. Build L1 terminal DWA via direct walk.
    //    Walks all (token, representative_state) pairs, builds a 2-state NWA,
    //    then determinizes + minimizes internally.
    let num_terminals = grammar.num_terminals as u32;
    let dwa = terminal_dwa::build_l1_terminal_dwa(
        tokenizer,
        vocab,
        &id_map,
        ignore_terminal,
        num_terminals,
        Some(active_terminals),
    )?;

    Some((id_map, dwa))
}
