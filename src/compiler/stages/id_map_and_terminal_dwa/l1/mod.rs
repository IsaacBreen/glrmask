//! L1 terminal DWA: direct 2-state construction for terminals with max path
//! length ≤ 1.

pub(crate) mod max_length;

use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::compact::compact_dwa_dimensions_fast;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::u8set::U8Set;
use crate::ds::weight::{Weight, shared_rangeset};
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::Vocab;

use super::l2p::equivalence_analysis::compat::TokenizerView;
use super::types::{TerminalColoring, compile_profile_enabled, debug_profile_enabled};

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// Uses max-length state equivalence and an identity vocab map, then traverses
/// the vocab tree to accumulate `terminal -> Weight` before building the final
/// 2-state DWA directly.
///
/// Returns `None` if the vocab is empty or no terminal matches exist.
pub(crate) fn build_l1_id_map_and_terminal_dwa(
    partition_label: &str,
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

    let total_started_at = Instant::now();
    let id_map_started_at = Instant::now();
    let (mut id_map, sorted_entries, state_to_rep, id_map_profile) = build_l1_id_map(tokenizer, vocab, active_terminals);
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_terminals = grammar.num_terminals as u32;
    let dwa_started_at = Instant::now();
    let (mut dwa, terminal_profile) = build_l1_terminal_dwa(
        tokenizer,
        sorted_entries,
        &id_map,
        &state_to_rep,
        num_terminals,
        active_terminals,
    )?;
    let terminal_build_ms = dwa_started_at.elapsed().as_secs_f64() * 1000.0;

    let compact_started_at = Instant::now();
    compact_dwa_dimensions_fast(&mut dwa, &mut id_map);
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l1] partition={} vocab_tokens={} tsids={} state_equiv_ms={:.3} token_identity_map_ms={:.3} id_map_ms={:.3} internal_vocab_ms={:.3} vocab_tree_build_ms={:.3} state_seed_ms={:.3} vocab_tree_traversal_ms={:.3} direct_terminal_dwa_ms={:.3} terminal_build_ms={:.3} compact_ms={:.3} determinize=none minimize=none prune=none total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            id_map.num_tsids(),
            id_map_profile.state_equiv_ms,
            id_map_profile.token_identity_map_ms,
            id_map_ms,
            terminal_profile.internal_vocab_ms,
            terminal_profile.vocab_tree_build_ms,
            terminal_profile.state_seed_ms,
            terminal_profile.vocab_tree_traversal_ms,
            terminal_profile.direct_terminal_dwa_ms,
            terminal_build_ms,
            compact_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    // Fast iteration: exit after L1 for a specific partition
    if let Ok(exit_label) = std::env::var("GLRMASK_EXIT_AFTER_L1") {
        if exit_label == partition_label {
            eprintln!("[glrmask/debug] EXIT_AFTER_L1={} triggered.", partition_label);
            std::process::exit(0);
        }
    }

    Some((id_map, dwa))
}

fn build_l1_id_map<'a>(tokenizer: &Tokenizer, vocab: &'a Vocab, active_terminals: &[bool]) -> (InternalIdMap, Vec<(u32, &'a [u8])>, Vec<u32>, L1IdMapProfile) {
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

    // Max-length bounded state equivalence: merge DFA states that behave
    // identically when only tokens up to the max vocab token length are
    // considered. Filtering by active_terminals lets us also merge states
    // that differ only by inactive terminal finalizers/futures.
    let state_equiv_started_at = Instant::now();
    let tokenizer_view = TokenizerView::new_filtered(tokenizer, active_terminals);
    let token_bytes: Vec<&[u8]> = vocab
        .entries
        .values()
        .map(|b| b.as_slice())
        .collect();
    let equiv_mapping = max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
    );
    // Build representative → internal_id mapping
    let mut rep_to_internal: FxHashMap<usize, u32> = FxHashMap::default();
    let mut state_original_to_internal = vec![u32::MAX; states.len()];
    let mut state_representatives = Vec::new();
    for (i, &rep) in equiv_mapping.iter().enumerate() {
        let state_id = states[i];
        let internal_id = *rep_to_internal.entry(rep).or_insert_with(|| {
            let id = state_representatives.len() as u32;
            state_representatives.push(rep as u32);
            id
        });
        state_original_to_internal[state_id] = internal_id;
    }
    // Build state_to_rep: original_state → representative_state (for trie traversal)
    let mut state_to_rep = vec![0u32; states.len()];
    for (i, &rep) in equiv_mapping.iter().enumerate() {
        state_to_rep[states[i]] = rep as u32;
    }
    let state_equiv_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    // Sort token IDs by byte content so internal IDs follow DFS traversal order
    // in the VocabPrefixTree. This makes reachable_token_ids() contiguous ranges,
    // enabling O(1) RangeSetBlaze unions during self-loop optimization.
    let token_identity_started_at = Instant::now();
    let mut token_id_bytes: Vec<(u32, &[u8])> = vocab
        .entries
        .iter()
        .map(|(&id, bytes)| (id, bytes.as_slice()))
        .collect();
    token_id_bytes.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));
    let mut token_original_to_internal = vec![u32::MAX; vocab.max_token_id() as usize + 1];
    let token_ids_sorted: Vec<u32> = token_id_bytes
        .iter()
        .enumerate()
        .map(|(internal_id, &(original_id, _))| {
            token_original_to_internal[original_id as usize] = internal_id as u32;
            original_id
        })
        .collect();
    let token_identity_map_ms = token_identity_started_at.elapsed().as_secs_f64() * 1000.0;

    (
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                state_original_to_internal,
                state_representatives.len() as u32,
                state_representatives,
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                token_original_to_internal,
                token_ids_sorted.len() as u32,
                token_ids_sorted,
            ),
        },
        token_id_bytes,
        state_to_rep,
        L1IdMapProfile {
            state_equiv_ms,
            token_identity_map_ms,
        },
    )
}

fn build_l1_terminal_dwa(
    tokenizer: &Tokenizer,
    sorted_entries: Vec<(u32, &[u8])>,
    id_map: &InternalIdMap,
    state_to_rep: &[u32],
    num_terminals: u32,
    active_terminals: &[bool],
) -> Option<(DWA, L1TerminalBuildProfile)> {
    let total_started_at = std::time::Instant::now();

    // Internal vocab is already in DFS (byte-sorted) order from build_l1_id_map
    let internal_vocab_started_at = Instant::now();
    let internal_vocab: Vec<(usize, &[u8])> = sorted_entries
        .iter()
        .enumerate()
        .map(|(internal_token_id, &(_original_id, bytes))| (internal_token_id, bytes))
        .collect();
    let internal_vocab_ms = internal_vocab_started_at.elapsed().as_secs_f64() * 1000.0;

    if internal_vocab.is_empty() {
        return None;
    }

    let vocab_tree_started_at = Instant::now();
    let tree = VocabPrefixTree::build_presorted(&internal_vocab);
    let vocab_tree_build_ms = vocab_tree_started_at.elapsed().as_secs_f64() * 1000.0;

    let state_seed_started_at = Instant::now();
    let mut states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
    for (internal_tsid, representative_state) in id_map.tokenizer_states.iter_representative_ids().enumerate() {
        states_to_initial_tsids
            .entry(representative_state)
            .or_default()
            .push(internal_tsid as u32);
    }
    let state_seed_ms = state_seed_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut profile = L1BuildProfile::default();
    let mut self_loop_cache = FxHashMap::<u32, U8Set>::default();

    // Fully deferred accumulation: end_state → (tsid → token_ids).
    // Both token nodes and self-loop subtrees accumulate here.
    // Terminal distribution happens once at the end via grouping.
    let num_dfa_states = tokenizer.num_states() as usize;
    let mut deferred: Vec<FxHashMap<u32, RangeSetBlaze<u32>>> =
        vec![FxHashMap::default(); num_dfa_states];

    let traversal_started_at = Instant::now();
    collect_terminal_weights(
        tokenizer,
        &tree.root,
        &states_to_initial_tsids,
        &mut deferred,
        &mut self_loop_cache,
        state_to_rep,
        &mut profile,
    );
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;

    // Pre-wrap deferred entries in Arc to enable cheap sharing during distribution.
    // Single-source (terminal, tsid) pairs just Arc::clone (~5ns) vs full clone (~200ns).
    let distribute_started_at = Instant::now();
    let deferred_arced: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> = deferred
        .into_iter()
        .map(|m| {
            m.into_iter()
                .map(|(tsid, rsb)| (tsid, shared_rangeset(rsb)))
                .collect()
        })
        .collect();
    let arc_wrap_ms = distribute_started_at.elapsed().as_secs_f64() * 1000.0;

    // Build terminal → sorted deduped set of active DFA states (mapped to representatives)
    let inverse_started_at = Instant::now();
    let mut terminal_to_active_states: Vec<Vec<u32>> = vec![Vec::new(); num_terminals as usize];
    for state in 0..num_dfa_states {
        let state_u32 = state as u32;
        let rep = state_to_rep[state];
        for tid in tokenizer.dfa.finalizers(state_u32).iter() {
            if active_terminals.get(tid).copied().unwrap_or(false) {
                terminal_to_active_states[tid].push(rep);
            }
        }
        for tid in tokenizer.tokens_accessible_from_state(state_u32).iter() {
            if active_terminals.get(tid).copied().unwrap_or(false) {
                terminal_to_active_states[tid].push(rep);
            }
        }
    }
    for states in &mut terminal_to_active_states {
        states.sort_unstable();
        states.dedup();
    }
    let inverse_map_ms = inverse_started_at.elapsed().as_secs_f64() * 1000.0;

    // Per-terminal distribution: collect deferred entries into per-tsid slots,
    // sort occupied tsids, merge per-tsid, build Weight.
    let merge_started_at = Instant::now();

    let num_tsids = id_map.num_tsids() as usize;
    let mut tsid_ranges: Vec<Vec<(u32, u32)>> = (0..num_tsids).map(|_| Vec::new()).collect();
    let mut tsid_single_arc: Vec<Option<Arc<RangeSetBlaze<u32>>>> = vec![None; num_tsids];
    let mut occupied_tsids: Vec<u32> = Vec::new();
    let mut weight_entries: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();

    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());
    let mut num_transitions = 0usize;

    for (tid, active_states) in terminal_to_active_states.iter().enumerate() {
        if active_states.is_empty() {
            continue;
        }

        occupied_tsids.clear();
        for &state in active_states {
            for &(tsid, ref arc) in &deferred_arced[state as usize] {
                let tsid_idx = tsid as usize;
                if tsid_ranges[tsid_idx].is_empty() && tsid_single_arc[tsid_idx].is_none() {
                    tsid_single_arc[tsid_idx] = Some(Arc::clone(arc));
                    occupied_tsids.push(tsid);
                } else if let Some(first_arc) = tsid_single_arc[tsid_idx].take() {
                    for r in first_arc.ranges() {
                        tsid_ranges[tsid_idx].push((*r.start(), *r.end()));
                    }
                    for r in arc.ranges() {
                        tsid_ranges[tsid_idx].push((*r.start(), *r.end()));
                    }
                } else {
                    for r in arc.ranges() {
                        tsid_ranges[tsid_idx].push((*r.start(), *r.end()));
                    }
                }
            }
        }

        if occupied_tsids.is_empty() {
            continue;
        }

        occupied_tsids.sort_unstable();

        weight_entries.clear();
        for &tsid in &occupied_tsids {
            let tsid_idx = tsid as usize;
            if let Some(arc) = tsid_single_arc[tsid_idx].take() {
                weight_entries.push((tsid, arc));
            } else {
                let slot = &mut tsid_ranges[tsid_idx];
                slot.sort_unstable();
                if !slot.is_empty() {
                    let mut write = 0;
                    for read in 1..slot.len() {
                        if slot[read].0 <= slot[write].1.saturating_add(1) {
                            slot[write].1 = slot[write].1.max(slot[read].1);
                        } else {
                            write += 1;
                            slot[write] = slot[read];
                        }
                    }
                    slot.truncate(write + 1);
                }
                let merged: RangeSetBlaze<u32> = slot.iter()
                    .map(|&(s, e)| s..=e)
                    .collect();
                weight_entries.push((tsid, shared_rangeset(merged)));
                slot.clear();
            }
        }

        let weight = Weight::from_per_tsid_shared(
            weight_entries.iter().map(|(tsid, arc)| (*tsid, Arc::clone(arc))),
        );
        if weight.is_empty() {
            continue;
        }

        dwa.add_transition(dwa.start_state, tid as i32, end_state, weight);
        num_transitions += 1;
    }

    if num_transitions == 0 {
        return None;
    }

    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    let direct_terminal_dwa_ms = merge_ms;
    let distribute_ms = distribute_started_at.elapsed().as_secs_f64() * 1000.0;
    let vocab_tree_traversal_ms = traversal_ms;

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][terminal_dwa] partition_build_l1_assoc vocab={} tsids={} tree_nodes={} token_nodes={} segment_execs={} live_segments={} terminal_hits={} states_at_token_nodes={} total_tsids_at_token_nodes={} max_tsids_per_state={} self_loop_skipped_subtrees={} self_loop_skipped_tokens={} transitions={} traversal_ms={:.1} arc_wrap_ms={:.1} inverse_map_ms={:.1} merge_ms={:.1} distribute_ms={:.1} total_ms={:.1}",
            tree.root.reachable_token_ids().len(),
            id_map.num_tsids(),
            profile.tree_nodes,
            profile.token_nodes,
            profile.segment_execs,
            profile.live_segments,
            profile.terminal_hits,
            profile.states_at_token_nodes,
            profile.total_tsids_at_token_nodes,
            profile.max_tsids_per_state,
            profile.self_loop_skipped_subtrees,
            profile.self_loop_skipped_tokens,
            num_transitions,
            traversal_ms,
            arc_wrap_ms,
            inverse_map_ms,
            merge_ms,
            distribute_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some((
        dwa,
        L1TerminalBuildProfile {
            internal_vocab_ms,
            vocab_tree_build_ms,
            state_seed_ms,
            vocab_tree_traversal_ms,
            direct_terminal_dwa_ms,
        },
    ))
}

struct L1IdMapProfile {
    state_equiv_ms: f64,
    token_identity_map_ms: f64,
}

struct L1TerminalBuildProfile {
    internal_vocab_ms: f64,
    vocab_tree_build_ms: f64,
    state_seed_ms: f64,
    vocab_tree_traversal_ms: f64,
    direct_terminal_dwa_ms: f64,
}

#[derive(Default)]
struct L1BuildProfile {
    tree_nodes: usize,
    segment_execs: usize,
    live_segments: usize,
    terminal_hits: usize,
    token_nodes: usize,
    states_at_token_nodes: usize,
    max_tsids_per_state: usize,
    total_tsids_at_token_nodes: usize,
    self_loop_skipped_subtrees: usize,
    self_loop_skipped_tokens: usize,
}

/// Compute the set of byte values that are self-loops for a DFA state.
fn compute_self_loop_bytes(tokenizer: &Tokenizer, state: u32) -> U8Set {
    let dfa_state = &tokenizer.dfa.states()[state as usize];
    let mut bytes = U8Set::empty();
    for (byte, &target) in dfa_state.transitions.iter() {
        if target == state {
            bytes.insert(byte);
        }
    }
    bytes
}

/// Compute equivalence-space self-loops: bytes where transitioning stays
/// in the same equivalence class (not necessarily the same DFA state).
fn compute_equiv_self_loop_bytes(tokenizer: &Tokenizer, state: u32, state_to_rep: &[u32]) -> U8Set {
    let dfa_state = &tokenizer.dfa.states()[state as usize];
    let rep = state_to_rep[state as usize];
    let mut bytes = U8Set::empty();
    for (byte, &target) in dfa_state.transitions.iter() {
        if target != u32::MAX && state_to_rep[target as usize] == rep {
            bytes.insert(byte);
        }
    }
    bytes
}

/// Check if all bytes in a subtree are self-loops for the given state.
fn is_self_loop_subtree(
    self_loop_cache: &mut FxHashMap<u32, U8Set>,
    tokenizer: &Tokenizer,
    node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    state: u32,
) -> bool {
    let self_loop_bytes = self_loop_cache
        .entry(state)
        .or_insert_with(|| compute_self_loop_bytes(tokenizer, state));
    U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes)
}

fn collect_terminal_weights(
    tokenizer: &Tokenizer,
    node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    states_to_initial_tsids: &FxHashMap<u32, Vec<u32>>,
    deferred: &mut Vec<FxHashMap<u32, RangeSetBlaze<u32>>>,
    self_loop_cache: &mut FxHashMap<u32, U8Set>,
    state_to_rep: &[u32],
    profile: &mut L1BuildProfile,
) {
    profile.tree_nodes += 1;

    if node.has_token() {
        let internal_token_id = node.token_id() as u32;
        profile.token_nodes += 1;
        profile.states_at_token_nodes += states_to_initial_tsids.len();
        for (&end_state, initial_tsids) in states_to_initial_tsids {
            profile.total_tsids_at_token_nodes += initial_tsids.len();
            if initial_tsids.len() > profile.max_tsids_per_state {
                profile.max_tsids_per_state = initial_tsids.len();
            }
            // Accumulate into deferred structure — no terminal iteration here
            let state_entry = &mut deferred[end_state as usize];
            for &initial_tsid in initial_tsids {
                state_entry.entry(initial_tsid).or_default().insert(internal_token_id);
            }
            profile.terminal_hits += initial_tsids.len();
        }
    }

    for (segment, child) in node.iter_children() {
        let mut child_states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
        let mut self_loop_states = FxHashMap::<u32, Vec<u32>>::default();

        // Pre-compute reachable_u32 once per child for self-loop optimization
        let reachable_u32: Option<RangeSetBlaze<u32>> = {
            let reachable = child.reachable_token_ids();
            if reachable.is_empty() {
                None
            } else {
                Some(
                    reachable
                        .ranges()
                        .map(|r| (*r.start() as u32)..=(*r.end() as u32))
                        .collect(),
                )
            }
        };
        // Pre-compute child's subtree byte set once (avoids repeated U8Set::from_words per state)
        let child_subtree_bytes = if reachable_u32.is_some() {
            Some(U8Set::from_words(*child.subtree_bytes()))
        } else {
            None
        };

        for (&start_state, initial_tsids) in states_to_initial_tsids {
            profile.segment_execs += 1;
            let Some(end_state) = tokenizer.execute_from_state_end_only(segment, start_state) else {
                continue;
            };
            profile.live_segments += 1;

            // Collapse end_state to its equivalence class representative
            let end_state_rep = state_to_rep[end_state as usize];

            let is_self_loop = child_subtree_bytes.as_ref().map_or(false, |csb| {
                let self_loop_bytes = self_loop_cache
                    .entry(end_state_rep)
                    .or_insert_with(|| compute_equiv_self_loop_bytes(tokenizer, end_state_rep, state_to_rep));
                csb.is_subset(self_loop_bytes)
            });

            if is_self_loop {
                self_loop_states
                    .entry(end_state_rep)
                    .or_default()
                    .extend(initial_tsids.iter().copied());
            } else {
                child_states_to_initial_tsids
                    .entry(end_state_rep)
                    .or_default()
                    .extend(initial_tsids.iter().copied());
            }
        }

        // Self-loop optimization: accumulate into the same deferred structure
        if let Some(ref reachable_u32) = reachable_u32 {
            if !self_loop_states.is_empty() {
                profile.self_loop_skipped_subtrees += self_loop_states.len();

                for (&end_state, initial_tsids) in &self_loop_states {
                    profile.self_loop_skipped_tokens += reachable_u32.len() as usize * initial_tsids.len();
                    let state_entry = &mut deferred[end_state as usize];
                    for &initial_tsid in initial_tsids {
                        *state_entry.entry(initial_tsid).or_default() |= reachable_u32;
                    }
                }
            }
        }

        if child_states_to_initial_tsids.is_empty() {
            continue;
        }

        collect_terminal_weights(
            tokenizer,
            child,
            &child_states_to_initial_tsids,
            deferred,
            self_loop_cache,
            state_to_rep,
            profile,
        );
    }
}
