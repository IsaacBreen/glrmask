//! L1 terminal DWA: direct 2-state construction for terminals with max path
//! length ≤ 1.

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::compact::compact_dwa_dimensions;
use crate::compiler::stages::equivalence_analysis::compat::TokenizerView;
use crate::compiler::stages::equivalence_analysis::state::max_length;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::Weight;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::Vocab;

use super::types::{TerminalColoring, debug_profile_enabled};

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// Uses max-length state equivalence and an identity vocab map, then traverses
/// the vocab tree to accumulate `terminal -> Weight` before building the final
/// 2-state DWA directly.
///
/// Returns `None` if the vocab is empty or no terminal matches exist.
pub(crate) fn build_l1_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _terminal_coloring: &TerminalColoring,
    _use_terminal_coloring: bool,
    _ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
) -> Option<(InternalIdMap, DWA)> {
    if vocab.is_empty() {
        return None;
    }

    let mut id_map = build_l1_id_map(tokenizer, vocab);
    let num_terminals = grammar.num_terminals as u32;
    let mut dwa = build_l1_terminal_dwa(
        tokenizer,
        vocab,
        &id_map,
        num_terminals,
        active_terminals,
    )?;

    compact_dwa_dimensions(&mut dwa, &mut id_map, false);

    Some((id_map, dwa))
}

fn build_l1_id_map(tokenizer: &Tokenizer, vocab: &Vocab) -> InternalIdMap {
    let tokenizer_view = TokenizerView::new(tokenizer);
    let token_bytes: Vec<&[u8]> = vocab.entries.values().map(|bytes| bytes.as_slice()).collect();
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let state_reps = max_length::find_state_equivalence_classes(&tokenizer_view, &token_bytes, &states);

    let mut rep_to_internal = FxHashMap::<usize, u32>::default();
    let mut state_original_to_internal = vec![u32::MAX; states.len()];
    let mut state_representatives = Vec::new();
    for (state_id, &representative_state) in state_reps.iter().enumerate() {
        let next_internal = state_representatives.len() as u32;
        let internal = *rep_to_internal.entry(representative_state).or_insert_with(|| {
            state_representatives.push(representative_state as u32);
            next_internal
        });
        state_original_to_internal[state_id] = internal;
    }

    let token_ids: Vec<u32> = vocab.entries.keys().copied().collect();
    let mut token_original_to_internal = vec![u32::MAX; vocab.max_token_id() as usize + 1];
    for (internal_token_id, token_id) in token_ids.iter().copied().enumerate() {
        token_original_to_internal[token_id as usize] = internal_token_id as u32;
    }

    InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
            state_original_to_internal,
            state_representatives.len() as u32,
            state_representatives,
        ),
        vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
            token_original_to_internal,
            token_ids.len() as u32,
            token_ids,
        ),
    }
}

fn build_l1_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    num_terminals: u32,
    active_terminals: &[bool],
) -> Option<DWA> {
    let total_started_at = std::time::Instant::now();

    let internal_vocab: Vec<(usize, Vec<u8>)> = id_map
        .vocab_tokens
        .iter_representative_ids()
        .enumerate()
        .filter_map(|(internal_token_id, representative)| {
            vocab
                .entries
                .get(&representative)
                .map(|bytes| (internal_token_id, bytes.clone()))
        })
        .collect();

    if internal_vocab.is_empty() {
        return None;
    }

    let tree = VocabPrefixTree::build_owned(internal_vocab);
    let mut states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
    for (internal_tsid, representative_state) in id_map.tokenizer_states.iter_representative_ids().enumerate() {
        states_to_initial_tsids
            .entry(representative_state)
            .or_default()
            .push(internal_tsid as u32);
    }

    let mut terminal_to_token_weights = vec![FxHashMap::<u32, RangeSetBlaze<u32>>::default(); num_terminals as usize];
    let mut profile = L1BuildProfile::default();
    collect_terminal_weights(
        tokenizer,
        &tree.root,
        &states_to_initial_tsids,
        active_terminals,
        &mut terminal_to_token_weights,
        &mut profile,
    );

    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());

    let mut num_transitions = 0usize;
    for (terminal_id, tsid_to_tokens) in terminal_to_token_weights.into_iter().enumerate() {
        if tsid_to_tokens.is_empty() {
            continue;
        }

        let weight = Weight::from_per_tsid_token_sets(tsid_to_tokens.into_iter());
        if weight.is_empty() {
            continue;
        }

        dwa.add_transition(dwa.start_state, terminal_id as i32, end_state, weight);
        num_transitions += 1;
    }

    if num_transitions == 0 {
        return None;
    }

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][terminal_dwa] partition_build_l1_assoc vocab={} tsids={} tree_nodes={} segment_execs={} live_segments={} terminal_hits={} transitions={} total_ms={:.1}",
            tree.root.reachable_token_ids().len(),
            id_map.num_tsids(),
            profile.tree_nodes,
            profile.segment_execs,
            profile.live_segments,
            profile.terminal_hits,
            num_transitions,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some(dwa)
}

#[derive(Default)]
struct L1BuildProfile {
    tree_nodes: usize,
    segment_execs: usize,
    live_segments: usize,
    terminal_hits: usize,
}

fn collect_terminal_weights(
    tokenizer: &Tokenizer,
    node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    states_to_initial_tsids: &FxHashMap<u32, Vec<u32>>,
    active_terminals: &[bool],
    terminal_to_token_weights: &mut [FxHashMap<u32, RangeSetBlaze<u32>>],
    profile: &mut L1BuildProfile,
) {
    profile.tree_nodes += 1;

    if node.has_token() {
        let internal_token_id = node.token_id() as u32;
        for (&end_state, initial_tsids) in states_to_initial_tsids {
            for terminal_id in tokenizer.dfa.finalizers(end_state).iter() {
                if !active_terminals.get(terminal_id).copied().unwrap_or(false) {
                    continue;
                }
                for &initial_tsid in initial_tsids {
                    terminal_to_token_weights[terminal_id]
                        .entry(initial_tsid)
                        .or_default()
                        .insert(internal_token_id);
                }
                profile.terminal_hits += initial_tsids.len();
            }
            for terminal_id in tokenizer.tokens_accessible_from_state(end_state).iter() {
                if !active_terminals.get(terminal_id).copied().unwrap_or(false) {
                    continue;
                }
                for &initial_tsid in initial_tsids {
                    terminal_to_token_weights[terminal_id]
                        .entry(initial_tsid)
                        .or_default()
                        .insert(internal_token_id);
                }
                profile.terminal_hits += initial_tsids.len();
            }
        }
    }

    for (segment, child) in node.iter_children() {
        let mut child_states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
        for (&start_state, initial_tsids) in states_to_initial_tsids {
            profile.segment_execs += 1;
            let Some(end_state) = tokenizer.execute_from_state_end_only(segment, start_state) else {
                continue;
            };
            profile.live_segments += 1;
            child_states_to_initial_tsids
                .entry(end_state)
                .or_default()
                .extend(initial_tsids.iter().copied());
        }

        if child_states_to_initial_tsids.is_empty() {
            continue;
        }

        collect_terminal_weights(
            tokenizer,
            child,
            &child_states_to_initial_tsids,
            active_terminals,
            terminal_to_token_weights,
            profile,
        );
    }
}
