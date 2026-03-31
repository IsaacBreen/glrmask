//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the same structure as the pre-partition/path-length code (commit 67146d8):
//! build vocab trie → compute possible_matches → seed root nodes → trie-walk
//! NWA build → postprocess (always_allowed → collapse → disallowed → prune →
//! canonicalize) → determinize → minimize.
//!
//! The only structural difference from the old code is `active_terminals`
//! filtering: terminals not in the L2+ set are skipped during the trie walk.

pub(crate) mod nwa_builder;
pub(crate) mod postprocess;

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesComputer, collect_possible_matches_by_internal_tsid,
};
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;
use crate::Vocab;

use super::grammar_helpers::compute_always_allowed_follows;
use super::types::TerminalColoring;
use nwa_builder::{build_nwa_via_trie_walk, internal_vocab_entries, seed_root_nodes};
use postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa, collapse_always_allowed,
    prune_non_coreachable_states,
};

/// Build an L2+ id_map and terminal DWA for the given vocab and terminal set.
///
/// Builds its own id_map via `InternalIdMap::build_with_group_filter` (full DFA-
/// based equivalence analysis restricted to L2+ terminal groups). Then builds
/// the terminal DWA using the old-shaped trie-walk NWA pipeline matching the
/// 67146d8 code shape:
///
/// 1. Build internal vocab entries
/// 2. Build vocab prefix trie
/// 3. Compute possible_matches (for root node seeding)
/// 4. Create NWA, seed root nodes
/// 5. Trie-walk NWA build
/// 6. Postprocess: always_allowed → collapse → disallowed → prune → canonicalize
/// 7. Determinize → minimize
///
/// `disallowed_follows` is threaded explicitly for id_map building.
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

    let id_map = InternalIdMap::build(
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
    );

    let internal_vocab = internal_vocab_entries(vocab, &id_map);
    if internal_vocab.is_empty() {
        return None;
    }
    let full_tree = VocabPrefixTree::build_owned(
        internal_vocab
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect(),
    );

    // 3. Compute possible_matches (needed for root node seeding).
    let mut pm_computer = PossibleMatchesComputer::new(tokenizer);
    let possible_matches_by_state = collect_possible_matches_by_internal_tsid(
        tokenizer,
        &full_tree.root,
        &mut pm_computer,
        &id_map.tokenizer_states,
    );

    // 4. Create NWA and seed root nodes.
    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let roots = seed_root_nodes(
        &mut nwa,
        start_state,
        tokenizer,
        &id_map,
        terminal_coloring,
        ignore_terminal,
        &possible_matches_by_state,
    );

    // 5. Trie-walk NWA build (active_terminals filtering skips non-L2+ terminals).
    let _build_profile = build_nwa_via_trie_walk(
        tokenizer,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        &mut nwa,
        leaf_state,
        id_map.num_tsids(),
        &full_tree.root,
        &roots,
        &mut pm_computer,
        Some(active_terminals),
    );

    // 6. Postprocess: same sequence as old code (67146d8).
    let always_allowed = compute_always_allowed_follows(grammar);
    collapse_always_allowed(&mut nwa, &always_allowed, grammar.num_terminals as usize);
    apply_disallowed_follow_constraints(&mut nwa, disallowed_follows, grammar.num_terminals as usize);
    prune_non_coreachable_states(&mut nwa);
    canonicalize_acyclic_nwa(&mut nwa);

    // 7. Determinize → minimize.
    let det = determinize(&nwa).expect("L2+ terminal NWA determinization failed");
    let dwa = minimize(&det);

    Some((id_map, dwa))
}
