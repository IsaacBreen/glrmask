//! Terminal DWA construction.
//!
//! This module builds the Terminal DWA from the tokenizer and LLM vocabulary.
//!
//! The Terminal DWA encodes which LLM tokens can be generated in each tokenizer state.
//! It's called "Terminal" because it handles the terminal symbols of the grammar -
//! specifically, how LLM tokens map to grammar terminals via the tokenizer.
//!
//! This is distinct from "Template DFAs" (in precompute4/template_dfa.rs) which encode
//! how each terminal type interacts with the parser stack.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::ops::BitOrAssign;
use std::sync::Arc;
use rayon::prelude::*;
use range_set_blaze::RangeSetBlaze;
use profiler_macro::{time_it, timeit};

use crate::constraint_vocab::LLMTokenBV;
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::dfa_u8::{Tokenizer, Regex};
use crate::glr::approximate_dfa::LazyApproximateDFA;
use crate::glr::parser::GLRParser;
use crate::dwa_i32::rangeset::RangeSet as WARangeSet;
use crate::dwa_i32::{DeterminizeAndMinimizeProfile, DWA, NWA, NWAStateID, Weight};
use crate::dwa_i32::weight_expansion::{expand_rsb, create_tsid_set_mask};
use crate::profiler::{self};
use crate::interface::{prune_dwa_with_suffix_cache, prune_nwa_with_suffix_cache, SuffixParserCache};

use crate::dfa_u8::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID as GrammarTokenID;
use crate::dwa_i32::common::Label;
use smallvec::SmallVec;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ChainCollapseStats {
    pub(crate) collapsed_states: usize,
    pub(crate) rewired_transitions: usize,
    pub(crate) blocked_repeat_transitions: usize,
    pub(crate) pruned_states: usize,
    pub(crate) iterations: usize,
}

pub(crate) fn collapse_self_extending_chains(
    dwa: &mut DWA,
    self_extending_labels: &HashSet<Label>,
) -> ChainCollapseStats {
    let mut stats = ChainCollapseStats::default();

    loop {
        let num_states = dwa.states.len();
        if num_states == 0 {
            break;
        }

        let mut incoming_total = vec![0usize; num_states];
        let mut incoming_by_label: HashMap<(usize, Label), usize> = HashMap::new();
        let mut incoming_sources: HashMap<(usize, Label), Vec<usize>> = HashMap::new();

        for (src, state) in dwa.states.0.iter().enumerate() {
            for (&label, &dst) in &state.transitions {
                incoming_total[dst] += 1;
                *incoming_by_label.entry((dst, label)).or_insert(0) += 1;
                incoming_sources.entry((dst, label)).or_default().push(src);
            }
        }

        let mut changed = false;
        for b in 0..num_states {
            if b == dwa.body.start_state {
                continue;
            }
            let state_b = &dwa.states[b];
            if state_b.transitions.len() != 1 {
                continue;
            }
            let (&label, &c) = match state_b.transitions.iter().next() {
                Some(entry) => entry,
                None => continue,
            };
            if !self_extending_labels.contains(&label) {
                continue;
            }
            if c == b {
                continue; // self-loop
            }
            let incoming_label = incoming_by_label.get(&(b, label)).copied().unwrap_or(0);
            if incoming_label == 0 || incoming_total[b] != incoming_label {
                continue; // other incoming labels exist
            }

            let preds = incoming_sources.get(&(b, label)).cloned().unwrap_or_default();
            if preds.is_empty() {
                continue;
            }

            let w_out = state_b
                .trans_weights
                .get(&label)
                .cloned()
                .unwrap_or_else(Weight::zeros);
            let final_weight_b = state_b.final_weight.clone();

            for p in preds {
                let w_in = dwa.states[p]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::zeros);
                let mut new_w = w_in;
                new_w |= &w_out;
                dwa.states[p].trans_weights.insert(label, new_w);
                dwa.states[p].transitions.insert(label, c);
                stats.rewired_transitions += 1;
            }

            if let Some(fw_b) = final_weight_b {
                let fw_c = dwa.states[c]
                    .final_weight
                    .get_or_insert_with(Weight::zeros);
                *fw_c |= &fw_b;
            }

            let state_b_mut = &mut dwa.states[b];
            state_b_mut.transitions.clear();
            state_b_mut.trans_weights.clear();
            state_b_mut.final_weight = None;

            stats.collapsed_states += 1;
            changed = true;
        }

        stats.iterations += 1;
        if !changed {
            break;
        }
    }

    if !self_extending_labels.is_empty() && !dwa.states.0.is_empty() {
        let mut incoming_label: HashSet<(usize, Label)> = HashSet::new();
        for state in dwa.states.0.iter() {
            for (&label, &dst) in &state.transitions {
                if self_extending_labels.contains(&label) {
                    incoming_label.insert((dst, label));
                }
            }
        }

        for (dst, label) in incoming_label {
            if dst == dwa.body.start_state {
                continue;
            }
            if let Some(state) = dwa.states.0.get_mut(dst) {
                if state.transitions.remove(&label).is_some() {
                    state.trans_weights.remove(&label);
                    stats.blocked_repeat_transitions += 1;
                }
            }
        }
    }

    let before = dwa.states.len();
    if dwa.prune_unreachable() {
        let after = dwa.states.len();
        if after < before {
            stats.pruned_states += before - after;
        }
    }

    stats
}

pub(crate) fn collapse_always_allowed(
    nwa: &mut NWA,
    always_allowed_by_label: &[Vec<Label>],
    terminals_count: usize,
) -> bool {
    if always_allowed_by_label.is_empty() || terminals_count == 0 {
        return false;
    }

    let num_states = nwa.states.len();
    if num_states == 0 {
        return false;
    }

    // Precompute incoming terminal labels and compute exact domains.
    let mut incoming: Vec<HashSet<Label>> = vec![HashSet::new(); num_states];
    let mut domain: Vec<Weight> = (0..num_states).map(|_| Weight::zeros()).collect();

    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    let mut in_queue: Vec<bool> = vec![false; num_states];

    for &start in &nwa.body.start_states {
        domain[start] = Weight::all();
        queue.push_back(start);
        in_queue[start] = true;
    }

    while let Some(state_id) = queue.pop_front() {
        in_queue[state_id] = false;
        let state = &nwa.states[state_id];
        let state_domain = domain[state_id].clone();
        let mut domain_empty = state_domain.clone();
        domain_empty &= Weight::all();
        if domain_empty.is_subset_of(&Weight::zeros()) {
            continue;
        }

        for (dst, w) in &state.epsilons {
            if domain[*dst].is_subset_of(&state_domain) {
                let src_labels: Vec<Label> = incoming[state_id].iter().copied().collect();
                incoming[*dst].extend(src_labels);
                continue;
            }
            domain[*dst] |= &state_domain;
            let src_labels: Vec<Label> = incoming[state_id].iter().copied().collect();
            incoming[*dst].extend(src_labels);
            if !in_queue[*dst] {
                in_queue[*dst] = true;
                queue.push_back(*dst);
            }
        }

        for (&label, targets) in &state.transitions {
            if label < 0 {
                continue;
            }
            let idx = label as usize;
            if idx >= terminals_count {
                continue;
            }
            for (dst, w) in targets {
                let mut contrib = state_domain.clone();
                contrib &= w;
                if !contrib.is_subset_of(&domain[*dst]) {
                    domain[*dst] |= &contrib;
                    if !in_queue[*dst] {
                        in_queue[*dst] = true;
                        queue.push_back(*dst);
                    }
                }
                if incoming[*dst].insert(label) {
                    if !in_queue[*dst] {
                        in_queue[*dst] = true;
                        queue.push_back(*dst);
                    }
                }
            }
        }
    }

    // Treat start states as having all terminals incoming.
    let all_labels: Vec<Label> = (0..(terminals_count as Label)).collect();
    for &start in &nwa.body.start_states {
        incoming[start].extend(all_labels.iter().copied());
    }
    for state_id in 0..num_states {
        if !nwa.states[state_id].epsilons.is_empty() {
            for &label in &all_labels {
                incoming[state_id].insert(label);
            }
        }
    }

    let mut allowed_by_state: Vec<HashSet<Label>> = vec![HashSet::new(); num_states];
    for state_id in 0..num_states {
        let incoming_labels = &incoming[state_id];
        if incoming_labels.is_empty() {
            continue;
        }
        let mut iter = incoming_labels.iter();
        let Some(&first) = iter.next() else {
            continue;
        };
        let first_idx = first as usize;
        if first_idx >= always_allowed_by_label.len() {
            continue;
        }
        let mut allowed: HashSet<Label> =
            always_allowed_by_label[first_idx].iter().copied().collect();
        for &label in iter {
            let idx = label as usize;
            if idx >= always_allowed_by_label.len() {
                continue;
            }
            let follow_set: HashSet<Label> =
                always_allowed_by_label[idx].iter().copied().collect();
            allowed.retain(|t| follow_set.contains(t));
            if allowed.is_empty() {
                break;
            }
        }
        allowed_by_state[state_id] = allowed;
    }

    let final_weights: Vec<Option<Weight>> =
        nwa.states.0.iter().map(|s| s.final_weight.clone()).collect();
    let mut changed = false;

    for state_id in 0..num_states {
        let allowed = &allowed_by_state[state_id];
        if allowed.is_empty() {
            continue;
        }
        let domain_state = &domain[state_id];
        let mut domain_empty = domain_state.clone();
        domain_empty &= Weight::all();
        if domain_empty.is_subset_of(&Weight::zeros()) {
            continue;
        }

        let state = &mut nwa.states[state_id];
        let mut labels_to_remove = Vec::new();

        for (&label, targets) in state.transitions.iter_mut() {
            if label < 0 {
                continue;
            }
            let idx = label as usize;
            if idx >= terminals_count {
                continue;
            }
            if !allowed.contains(&label) {
                continue;
            }

            let mut new_targets: Vec<(NWAStateID, Weight)> = Vec::new();
            for (dst, w) in targets.iter() {
                if let Some(fw) = final_weights[*dst].as_ref() {
                    let mut reach = domain_state.clone();
                    reach &= w;
                    if reach.is_subset_of(fw) {
                        let mut contrib = fw.clone();
                        contrib &= w;
                        let fw_state = state.final_weight.get_or_insert_with(Weight::zeros);
                        *fw_state |= &contrib;
                        changed = true;
                        continue;
                    }
                }
                new_targets.push((*dst, w.clone()));
            }

            if new_targets.is_empty() {
                labels_to_remove.push(label);
            } else {
                *targets = new_targets;
            }
        }

        for label in labels_to_remove {
            state.transitions.remove(&label);
        }
    }

    if nwa.prune_unreachable() {
        changed = true;
    }

    changed
}

/// Prune NWA transitions based on disallowed follow sets.
///
/// For an acyclic NWA, propagates "disallowed terminal" sets forward through
/// the graph in topological order:
///   - Start states begin with empty disallowed sets.
///   - Epsilon transitions transmit the source's disallowed set unchanged.
///   - Label transitions (terminal L) transmit disallowed_after[L] to the dest
///     (the set of terminals that can NEVER follow L in the grammar, i.e.,
///      complement of the union of follow sets across all occurrences of L).
///   - At each state, outgoing transitions whose label is in that state's
///     disallowed set are pruned.
///
/// `ever_allowed_by_label[L]` = union of per-occurrence follow sets for terminal L.
/// The complement gives terminals that can NEVER follow L — safe to prune.
///
/// Returns true if any transitions were removed.
pub(crate) fn prune_nwa_disallowed_follows(
    nwa: &mut NWA,
    ever_allowed_by_label: &[Vec<Label>],
    terminals_count: usize,
) -> bool {
    if ever_allowed_by_label.is_empty() || terminals_count == 0 {
        return false;
    }
    let num_states = nwa.states.len();
    if num_states == 0 {
        return false;
    }

    // Use a compact bitset for terminal sets (much faster than HashSet for 157 terminals)
    let words_needed = (terminals_count + 63) / 64;

    // Inline bitset operations
    let mut new_bitset = || vec![0u64; words_needed];
    let set_bit = |bs: &mut [u64], idx: usize| {
        if idx < terminals_count { bs[idx / 64] |= 1u64 << (idx % 64); }
    };
    let test_bit = |bs: &[u64], idx: usize| -> bool {
        idx < terminals_count && (bs[idx / 64] & (1u64 << (idx % 64))) != 0
    };
    let is_empty = |bs: &[u64]| -> bool { bs.iter().all(|&w| w == 0) };
    let union_into = |dst: &mut [u64], src: &[u64]| {
        for i in 0..dst.len() { dst[i] |= src[i]; }
    };
    let intersect_into = |dst: &mut [u64], src: &[u64]| {
        for i in 0..dst.len() { dst[i] &= src[i]; }
    };

    // Precompute disallowed_after[label] as bitsets
    let mut all_terminals_bs = new_bitset();
    for i in 0..terminals_count { set_bit(&mut all_terminals_bs, i); }

    let disallowed_after: Vec<Vec<u64>> = (0..terminals_count)
        .map(|idx| {
            if idx < ever_allowed_by_label.len() {
                let mut bs = all_terminals_bs.clone();
                for &allowed_label in &ever_allowed_by_label[idx] {
                    if (allowed_label as usize) < terminals_count {
                        bs[allowed_label as usize / 64] &= !(1u64 << (allowed_label as usize % 64));
                    }
                }
                bs
            } else {
                new_bitset() // no follow info → disallow nothing
            }
        })
        .collect();

    // Topological sort via Kahn's algorithm
    let mut in_degree = vec![0u32; num_states];
    for state in &nwa.states.0 {
        for (dst, _) in &state.epsilons {
            in_degree[*dst] += 1;
        }
        for (_, targets) in &state.transitions {
            for (dst, _) in targets {
                in_degree[*dst] += 1;
            }
        }
    }

    let mut topo_queue: VecDeque<NWAStateID> = VecDeque::new();
    for id in 0..num_states {
        if in_degree[id] == 0 {
            topo_queue.push_back(id);
        }
    }

    let mut topo_order: Vec<NWAStateID> = Vec::with_capacity(num_states);

    // First pass: union semantics (collect all disallowed)
    let mut disallowed: Vec<Option<Vec<u64>>> = vec![None; num_states];
    for &start in &nwa.body.start_states {
        disallowed[start] = Some(new_bitset());
    }

    while let Some(sid) = topo_queue.pop_front() {
        topo_order.push(sid);

        let state = &nwa.states[sid];
        let src_disallowed = disallowed[sid].clone().unwrap_or_else(|| new_bitset());

        for (dst, _) in &state.epsilons {
            let dst_set = disallowed[*dst].get_or_insert_with(|| new_bitset());
            union_into(dst_set, &src_disallowed);
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            let label_disallowed = &disallowed_after[label as usize];
            for (dst, _) in targets {
                let dst_set = disallowed[*dst].get_or_insert_with(|| new_bitset());
                union_into(dst_set, label_disallowed);
            }
        }

        // Decrement in-degree for successors
        for (dst, _) in &state.epsilons {
            in_degree[*dst] -= 1;
            if in_degree[*dst] == 0 {
                topo_queue.push_back(*dst);
            }
        }
        for (_, targets) in &state.transitions {
            for (dst, _) in targets {
                in_degree[*dst] -= 1;
                if in_degree[*dst] == 0 {
                    topo_queue.push_back(*dst);
                }
            }
        }
    }

    // Second pass: intersection semantics
    let mut disallowed_intersected: Vec<Option<Vec<u64>>> = vec![None; num_states];
    for &start in &nwa.body.start_states {
        disallowed_intersected[start] = Some(new_bitset());
    }

    for &sid in &topo_order {
        let state = &nwa.states[sid];
        let src_disallowed = disallowed_intersected[sid].clone().unwrap_or_else(|| new_bitset());

        for (dst, _) in &state.epsilons {
            let entry = &mut disallowed_intersected[*dst];
            match entry {
                None => *entry = Some(src_disallowed.clone()),
                Some(existing) => {
                    intersect_into(existing, &src_disallowed);
                }
            }
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            let label_disallowed = &disallowed_after[label as usize];
            for (dst, _) in targets {
                let entry = &mut disallowed_intersected[*dst];
                match entry {
                    None => *entry = Some(label_disallowed.clone()),
                    Some(existing) => {
                        intersect_into(existing, label_disallowed);
                    }
                }
            }
        }
    }

    // Now prune: for each state, remove outgoing transitions whose label is disallowed
    let mut changed = false;
    let mut total_pruned = 0usize;
    for sid in 0..num_states {
        let state_disallowed = match &disallowed_intersected[sid] {
            Some(d) if !is_empty(d) => d,
            _ => continue,
        };

        let state = &mut nwa.states[sid];
        let mut labels_to_remove: Vec<Label> = Vec::new();

        for (&label, targets) in state.transitions.iter() {
            if label >= 0 && (label as usize) < terminals_count && test_bit(state_disallowed, label as usize) {
                labels_to_remove.push(label);
                total_pruned += targets.len();
            }
        }

        if !labels_to_remove.is_empty() {
            changed = true;
            for label in labels_to_remove {
                state.transitions.remove(&label);
            }
        }
    }

    if changed {
        crate::debug!(
            4,
            "NWA disallowed-follows prune: removed {} transition targets",
            total_pruned,
        );
        // Prune states that became unreachable
        nwa.prune_unreachable();
    }

    changed
}

// No-op progress bar replacement
struct NoOpPb;
impl NoOpPb {
    fn inc(&self, _: u64) {}
    fn finish(&self) {}
}

const EXPANDED_RSB_CACHE_MAX_ENTRIES: usize = 100_000;
const EXPANDED_RSB_VALUE_CACHE_MAX_ENTRIES: usize = 50_000;

type SourceStates = SmallVec<[NWAStateID; 1]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DfsKey {
    tokenizer_state: TokenizerStateID,
}

impl DfsKey {
    #[inline]
    fn new(tokenizer_state: TokenizerStateID) -> Self {
        Self { tokenizer_state }
    }
}

#[derive(Clone)]
pub struct ApproximateDfaPruner {
    pub dfa: LazyApproximateDFA,
    pub orig_to_suffix_tid: Vec<Option<crate::types::TerminalID>>,
    pub ignored_terminals: Vec<bool>,
    pub reduce_fallback_terminals_by_state: Vec<Vec<usize>>,
}

pub fn build_reduce_fallback_terminals_by_state(
    parser: &crate::glr::parser::GLRParser,
) -> Vec<Vec<usize>> {
    use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;

    let num_terminals = parser.terminal_map.len();
    let num_states = parser.table.keys().map(|s| s.0).max().unwrap_or(0) + 1;
    let mut fallback_by_state: Vec<Vec<usize>> = vec![Vec::new(); num_states];

    for (_state_id, row) in parser.table.iter() {
        let state_idx = _state_id.0;
        let mut terms: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for (term_id, action) in row.get_shifts_and_reduces_map() {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce { .. } => {
                    terms.insert(term_id.0);
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                    if !reduces.is_empty() {
                        terms.insert(term_id.0);
                    }
                }
                Stage7ShiftsAndReducesLookaheadValue::Shift(_) => {}
            }
        }

        if let Some(default_reduce) = &row.default_reduce {
            if let Stage7ShiftsAndReducesLookaheadValue::Reduce { .. } = default_reduce {
                if let Some(lookaheads) = &row.default_reduce_lookaheads {
                    for term_id in lookaheads {
                        if term_id.0 < num_terminals {
                            terms.insert(term_id.0);
                        }
                    }
                } else {
                    // Wildcard default reduce: we only need to know that a reduce is
                    // possible for this state. Avoid O(num_terminals) work here since
                    // downstream only checks for non-empty fallback lists.
                    if terms.is_empty() {
                        if num_terminals > 0 {
                            terms.insert(0);
                        } else {
                            terms.insert(usize::MAX);
                        }
                    }
                }
            }
        }

        if !terms.is_empty() {
            fallback_by_state[state_idx] = terms.into_iter().collect();
        }
    }

    fallback_by_state
}

// ---------------------------------------------------------------------------
// PossibleMatchesComputer - thread-safe PM computation for parallel use
// ---------------------------------------------------------------------------

/// Standalone possible-matches computer that owns a per-thread cache.
/// Used by `compute_possible_matches_for_all_states` to parallelize PM computation
/// across representative tokenizer states. Each rayon thread gets its own instance.
struct PossibleMatchesComputer<'a> {
    tokenizer: &'a Tokenizer,
    terminal_to_greedy_group: &'a [Option<usize>],
    merge_possible_matches_by_group: bool,
    merge_possible_matches_globally: bool,
    cache: HashMap<usize, HashMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>,
}

impl<'a> PossibleMatchesComputer<'a> {
    fn new(
        tokenizer: &'a Tokenizer,
        terminal_to_greedy_group: &'a [Option<usize>],
        merge_possible_matches_by_group: bool,
        merge_possible_matches_globally: bool,
    ) -> Self {
        Self {
            tokenizer,
            terminal_to_greedy_group,
            merge_possible_matches_by_group,
            merge_possible_matches_globally,
            cache: HashMap::new(),
        }
    }

    fn possible_matches(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key = vocab_node as *const _ as usize;

        if let Some(cached_for_node) = self.cache.get(&cache_key) {
            if let Some(cached_result) = cached_for_node.get(&tokenizer_state_id) {
                return cached_result.clone();
            }
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();
        let mut blocked_by_terminal: HashMap<GrammarTokenID, LLMTokenBV> = HashMap::new();
        let tokenizer_dfa = self.tokenizer.dfa();
        let terminal_to_greedy_group = self.terminal_to_greedy_group;
        let has_future_in_same_greedy_group = |state_idx: usize, group_idx: usize| -> bool {
            tokenizer_dfa
                .states
                .get(state_idx)
                .map(|state| {
                    state
                        .possible_future_group_ids
                        .iter()
                        .any(|future_gid| {
                            terminal_to_greedy_group
                                .get(*future_gid)
                                .and_then(|group| *group)
                                == Some(group_idx)
                        })
                })
                .unwrap_or(false)
        };
        let build_state_by_width =
            |segment_bytes: &[u8], start_state: TokenizerStateID| -> Vec<Option<usize>> {
                let mut states_by_width = vec![None; segment_bytes.len() + 1];
                let mut curr_state = start_state.0;
                states_by_width[0] = Some(curr_state);
                for (idx, &byte) in segment_bytes.iter().enumerate() {
                    let Some(state) = tokenizer_dfa.states.get(curr_state) else {
                        break;
                    };
                    let Some(&next_state) = state.transitions.get(byte) else {
                        break;
                    };
                    curr_state = next_state;
                    states_by_width[idx + 1] = Some(curr_state);
                }
                states_by_width
            };

        let own_token_id = vocab_node.token_id();
        if vocab_node.reachable_token_ids().contains(own_token_id) {
            if let Some(state) = tokenizer_dfa.states.get(tokenizer_state_id.0) {
                for finalizer_gid in state.finalizers.iter() {
                    let terminal_id = GrammarTokenID(finalizer_gid);
                    if self.merge_possible_matches_by_group {
                        if let Some(Some(group_idx)) =
                            self.terminal_to_greedy_group.get(terminal_id.0)
                        {
                            if has_future_in_same_greedy_group(tokenizer_state_id.0, *group_idx) {
                                for (blocked_tid, blocked_group) in
                                    self.terminal_to_greedy_group.iter().enumerate()
                                {
                                    if *blocked_group == Some(*group_idx) {
                                        blocked_by_terminal
                                            .entry(GrammarTokenID(blocked_tid))
                                            .or_insert_with(LLMTokenBV::zeros)
                                            .set(own_token_id, true);
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    result_map
                        .entry(terminal_id)
                        .or_insert_with(LLMTokenBV::zeros)
                        .set(own_token_id, true);
                }
            }
        }

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result =
                self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            let states_by_width = if self.merge_possible_matches_by_group && !exec_result.matches.is_empty() {
                Some(build_state_by_width(&segment_bytes, tokenizer_state_id))
            } else {
                None
            };
            for token in &exec_result.matches {
                let applicable_tokens = child_vocab_node.reachable_token_ids();
                let applicable_tokens_bv: LLMTokenBV = RangeSet::from(applicable_tokens);
                if self.merge_possible_matches_by_group {
                    if let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(token.id) {
                        if let Some(state_idx) = states_by_width
                            .as_ref()
                            .and_then(|states| states.get(token.width))
                            .and_then(|sid| *sid)
                        {
                            if has_future_in_same_greedy_group(state_idx, *group_idx) {
                                for (blocked_tid, blocked_group) in
                                    self.terminal_to_greedy_group.iter().enumerate()
                                {
                                    if *blocked_group == Some(*group_idx) {
                                        *blocked_by_terminal
                                            .entry(GrammarTokenID(blocked_tid))
                                            .or_insert_with(LLMTokenBV::zeros) |=
                                            applicable_tokens_bv.clone();
                                    }
                                }
                                continue;
                            }
                        }
                    }
                }
                let grammar_token_id = GrammarTokenID(token.id);
                *result_map
                    .entry(grammar_token_id)
                    .or_insert_with(LLMTokenBV::zeros) |=
                    applicable_tokens_bv;
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: std::collections::BTreeSet<_> = self
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(final_state_val))
                    .into_iter()
                    .collect();
                let matches_here: std::collections::BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();
                let possible_new_matches =
                    &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(
                        child_vocab_node,
                        TokenizerStateID(final_state_val),
                    );
                    for (token, bv) in next_results {
                        *result_map
                            .entry(token)
                            .or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }

        if self.merge_possible_matches_by_group && !result_map.is_empty() {
            let mut group_merged: HashMap<usize, LLMTokenBV> = HashMap::new();
            for (&tid, bv) in &result_map {
                if let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(tid.0) {
                    group_merged
                        .entry(*group_idx)
                        .or_insert_with(LLMTokenBV::zeros)
                        .bitor_assign(bv.clone());
                }
            }
            for (tid, bv) in result_map.iter_mut() {
                if let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(tid.0) {
                    if let Some(merged) = group_merged.get(group_idx) {
                        *bv = merged.clone();
                    }
                }
                if let Some(blocked) = blocked_by_terminal.get(tid) {
                    *bv -= blocked;
                }
            }
            result_map.retain(|_, bv| !bv.is_empty());
        }

        if self.merge_possible_matches_globally && !result_map.is_empty() {
            let mut merged = LLMTokenBV::zeros();
            for bv in result_map.values().cloned() {
                merged |= bv;
            }
            result_map.clear();
            result_map.insert(MERGED_POSSIBLE_MATCH_KEY, merged);
        }

        self.cache
            .entry(cache_key)
            .or_default()
            .insert(tokenizer_state_id, result_map.clone());

        result_map
    }
}

// ---------------------------------------------------------------------------
// Precomputer1
// ---------------------------------------------------------------------------

const MERGED_POSSIBLE_MATCH_KEY: GrammarTokenID = GrammarTokenID(usize::MAX);

pub(crate) struct Precomputer1<'r> {
    pub(crate) tokenizer: &'r Tokenizer,
    pub(crate) vocab: VocabPrefixTree,
    pub(crate) roots: BTreeMap<DfsKey, SourceStates>,
    pub(crate) state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
    pub(crate) possible_matches: RefCell<
        BTreeMap<
            *const VocabPrefixTreeNode,
            BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>,
        >,
    >,
    terminal_to_greedy_group: Vec<Option<usize>>,
    merge_possible_matches_globally: bool,
    merge_possible_matches_by_group: bool,
    pub(crate) all_llm_tokens: RangeSetBlaze<usize>,
    pub(crate) pb: NoOpPb,
    pub(crate) leaf_state: NWAStateID,
    pub(crate) nwa: NWA,
    pub(crate) terminals_count: usize,
    pub(crate) pending_transitions: HashMap<NWAStateID, HashMap<Label, HashMap<NWAStateID, Weight>>>,
    pub(crate) pending_epsilons: HashMap<NWAStateID, HashMap<NWAStateID, Weight>>,
    pub(crate) pending_token_ids: Vec<Vec<Vec<usize>>>,  // [src_nwa_state][label_idx] → [token_ids]
    pub(crate) live_tokens: HashMap<NWAStateID, Weight>,
    // Cache for tokens_accessible_from_state - only 389 unique states but called 700k+ times
    accessible_terminals_cache: HashMap<TokenizerStateID, std::rc::Rc<Vec<Label>>>,
    // Cache for expanded single-token weights (indexed by token id)
    expanded_item_cache: Vec<Option<Weight>>,
    // Cache for expanded RangeSetBlaze weights (pointer-keyed, stable sets only)
    expanded_rsb_cache: HashMap<usize, Weight>,
    expanded_rsb_value_cache: HashMap<RangeSetBlaze<usize>, Weight>,
    // Weight-heavy mode: number of internal tsids (representative count)
    pub(crate) num_tsids: usize,
    // Max LLM token ID for creating tsid masks
    pub(crate) internal_max_llm_token: usize,
    // Mapping from raw tokenizer state ID -> internal tsid index (0..num_tsids-1)
    // Used in weight-heavy mode to encode tsid dimension in DWA weights.
    pub(crate) state_to_internal_tsid: Vec<usize>,
    expanded_all_weight: Weight,
    direct_insert: bool,
    suffix_prune_cache: Option<Arc<SuffixParserCache>>,
    self_extending_labels_for_collapse: Option<Arc<HashSet<Label>>>,
    ignored_terminals: Arc<Vec<bool>>,
    always_allowed_by_label: Arc<Vec<Vec<Label>>>,
    ever_allowed_by_label: Arc<Vec<Vec<Label>>>,
    nwa_rep_stats_enabled: bool,
    nwa_states_by_rep: BTreeMap<TokenizerStateID, usize>,
    nwa_states_by_rep_depth: BTreeMap<TokenizerStateID, BTreeMap<usize, usize>>,
    // DFS profiling counters
    dfs_profile_execute: std::time::Duration,
    dfs_profile_weight: std::time::Duration,
    dfs_profile_pending: std::time::Duration,
    dfs_profile_clone: std::time::Duration,
    dfs_profile_match: std::time::Duration,
    dfs_profile_endstate: std::time::Duration,
    dfs_profile_nodes: usize,
    dfs_profile_segments: usize,
    dfs_profile_iters: usize,
    dfs_profile_state_key_iters: usize,
    dfs_profile_matches: usize,
    dfs_profile_endstate_adds: usize,
    dfs_profile_endstate_events: usize,
    dfs_profile_enabled: bool,
    // Fast DFS execution tables
    dfs_flat_transitions: Vec<[u32; 256]>,
    dfs_finalizers: Vec<Vec<usize>>,
    dfs_non_greedy: Vec<bool>,
    dfs_num_groups: usize,
    dfs_match_gen: Vec<u32>,
    dfs_match_widths: Vec<usize>,
    dfs_current_gen: u32,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    /// Used for early-exit optimization in execute_from_state_fast.
    dfs_self_loop_bytes: Vec<[u64; 4]>,
    /// Reusable buffer for suffix byte sets in execute_from_state_fast.
    dfs_suffix_bytes: Vec<[u64; 4]>,
    // Batched leaf_state token IDs for deferred update_live_tokens
    dfs_leaf_token_ids: RangeSetBlaze<usize>,
}

impl<'r> Precomputer1<'r> {
    fn new(
        tokenizer: &'r Tokenizer,
        internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        terminals_count: usize,
        state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
        num_tsids: usize,
        state_to_internal_tsid: Vec<usize>,
        suffix_prune_cache: Option<Arc<SuffixParserCache>>,
        self_extending_labels_for_collapse: Option<Arc<HashSet<Label>>>,
        ignored_terminals: Arc<Vec<bool>>,
        always_allowed_by_label: Arc<Vec<Vec<Label>>>,
        ever_allowed_by_label: Arc<Vec<Vec<Label>>>,
        terminal_to_greedy_group: Vec<Option<usize>>,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!(
                "Precompute1 tokens: internal_llm_token_map entries={}, internal_max_llm_token={}, num_tsids={}",
                internal_llm_token_map.len(),
                internal_max_llm_token,
                num_tsids,
            );
        }

        crate::debug!(6, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(6, "Done building vocab prefix tree");

        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Clear default start state

        let mut roots = BTreeMap::new();
        for &rep_sid in state_to_rep.values() {
            let key = DfsKey::new(rep_sid);
            let entry = roots.entry(key).or_insert_with(SmallVec::new);
            if entry.is_empty() {
                let root_state = nwa.add_state();
                entry.push(root_state);
            }
        }
        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!(
                "Created trie1 roots ({} states for {} total tsids)",
                roots.len(),
                state_to_rep.len()
            );
        }

        let nwa_rep_stats_enabled = std::env::var("PRECOMPUTE1_NWA_REP_STATS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || crate::r#macro::is_debug_level_enabled(4);
        let mut nwa_states_by_rep = BTreeMap::new();
        let mut nwa_states_by_rep_depth = BTreeMap::new();
        if nwa_rep_stats_enabled {
            for key in roots.keys() {
                *nwa_states_by_rep.entry(key.tokenizer_state).or_insert(0) += 1;
                let depth_map = nwa_states_by_rep_depth
                    .entry(key.tokenizer_state)
                    .or_insert_with(BTreeMap::new);
                *depth_map.entry(0).or_insert(0) += 1;
            }
        }

        let pb = NoOpPb;

        let leaf_state = nwa.add_state();
        // Final weight - expanded in weight-heavy mode, simple in symbol-heavy mode
        // IMPORTANT: Use [0..=...] to create from ONE range, not iterate over all integers!
        let final_weight = if num_tsids == 0 {
            // Symbol-heavy mode: all tokens in N-space
            Weight::from_rsb(RangeSetBlaze::from_iter([0..=internal_max_llm_token]))
        } else {
            // Weight-heavy mode: all tokens in N×M-space
            Weight::from_rsb(expand_rsb(&RangeSetBlaze::from_iter([0..=internal_max_llm_token]), num_tsids))
        };
        nwa.states[leaf_state].final_weight = Some(final_weight);
        crate::debug!(6, "Created trie1 leaf state with final weight (num_tsids={})", num_tsids);

        let expanded_all_weight = if num_tsids == 0 {
            // Symbol-heavy mode: all tokens in N-space
            Weight::from_rsb(RangeSetBlaze::from_iter([0..=internal_max_llm_token]))
        } else {
            // Weight-heavy mode: All tokens in N×M space
            let max_pos = internal_max_llm_token * num_tsids + num_tsids - 1;
            // IMPORTANT: Use [0..=max_pos] to create from ONE range, not iterate over all integers!
            Weight::from_rsb(RangeSetBlaze::from_iter([0..=max_pos]))
        };

        let direct_insert = std::env::var("PRECOMPUTE1_DIRECT_INSERT")
            .map(|v| v == "1")
            .unwrap_or(false);
        let merge_possible_matches_globally = std::env::var("MERGE_POSSIBLE_MATCHES").is_ok();
        let disable_greedy_merge = std::env::var("NO_GREEDY_MERGE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let merge_possible_matches_by_group =
            !merge_possible_matches_globally
                && !disable_greedy_merge
                && terminal_to_greedy_group.iter().any(|group| group.is_some());

        Self {
            tokenizer,
            vocab,
            roots,
            state_to_rep,
            possible_matches: RefCell::new(BTreeMap::new()),
            terminal_to_greedy_group,
            merge_possible_matches_globally,
            merge_possible_matches_by_group,
            // IMPORTANT: Use [0..=...] to create from ONE range, not iterate over all integers!
            all_llm_tokens: RangeSetBlaze::from_iter([0..=internal_max_llm_token]),
            pb,
            leaf_state,
            nwa,
            terminals_count,
            pending_transitions: HashMap::new(),
            pending_epsilons: HashMap::new(),
            pending_token_ids: Vec::new(),  // Will be resized as NWA states are added
            live_tokens: HashMap::new(),
            accessible_terminals_cache: HashMap::new(),
            expanded_item_cache: vec![None; internal_max_llm_token.saturating_add(1)],
            expanded_rsb_cache: HashMap::new(),
            expanded_rsb_value_cache: HashMap::new(),
            num_tsids,
            internal_max_llm_token,
            state_to_internal_tsid,
            expanded_all_weight,
            direct_insert,
            suffix_prune_cache,
            self_extending_labels_for_collapse,
            ignored_terminals,
            always_allowed_by_label,
            ever_allowed_by_label,
            nwa_rep_stats_enabled,
            nwa_states_by_rep,
            nwa_states_by_rep_depth,
            dfs_profile_execute: std::time::Duration::ZERO,
            dfs_profile_weight: std::time::Duration::ZERO,
            dfs_profile_pending: std::time::Duration::ZERO,
            dfs_profile_clone: std::time::Duration::ZERO,
            dfs_profile_match: std::time::Duration::ZERO,
            dfs_profile_endstate: std::time::Duration::ZERO,
            dfs_profile_nodes: 0,
            dfs_profile_segments: 0,
            dfs_profile_iters: 0,
            dfs_profile_state_key_iters: 0,
            dfs_profile_matches: 0,
            dfs_profile_endstate_adds: 0,
            dfs_profile_endstate_events: 0,
            dfs_profile_enabled: false,
            // Initialize fast DFS tables from DFA
            dfs_flat_transitions: {
                let dfa = tokenizer.dfa();
                dfa.states.iter().map(|state| {
                    let mut table = [u32::MAX; 256];
                    for (byte, &target) in state.transitions.iter() {
                        table[byte as usize] = target as u32;
                    }
                    table
                }).collect()
            },
            dfs_finalizers: {
                let dfa = tokenizer.dfa();
                dfa.states.iter().map(|state| {
                    state.finalizers.iter().collect::<Vec<_>>()
                }).collect()
            },
            dfs_non_greedy: {
                let dfa = tokenizer.dfa();
                let max_gid = dfa.states.iter()
                    .flat_map(|s| s.finalizers.iter())
                    .max().unwrap_or(0);
                let mut flags = vec![false; max_gid + 1];
                for &gid in &dfa.non_greedy_finalizers {
                    if gid < flags.len() { flags[gid] = true; }
                }
                flags
            },
            dfs_num_groups: {
                let dfa = tokenizer.dfa();
                dfa.states.iter()
                    .flat_map(|s| s.finalizers.iter())
                    .max().map(|m| m + 1).unwrap_or(0)
            },
            dfs_match_gen: {
                let dfa = tokenizer.dfa();
                let num_groups = dfa.states.iter()
                    .flat_map(|s| s.finalizers.iter())
                    .max().map(|m| m + 1).unwrap_or(0);
                vec![0u32; num_groups]
            },
            dfs_match_widths: {
                let dfa = tokenizer.dfa();
                let num_groups = dfa.states.iter()
                    .flat_map(|s| s.finalizers.iter())
                    .max().map(|m| m + 1).unwrap_or(0);
                vec![0usize; num_groups]
            },
            dfs_current_gen: 0,
            dfs_self_loop_bytes: {
                let dfa = tokenizer.dfa();
                dfa.states.iter().enumerate().map(|(s, state)| {
                    let mut bits = [0u64; 4];
                    for (byte, &target) in state.transitions.iter() {
                        if target == s {
                            bits[byte as usize >> 6] |= 1u64 << (byte & 63);
                        }
                    }
                    bits
                }).collect()
            },
            dfs_suffix_bytes: Vec::new(),
            dfs_leaf_token_ids: RangeSetBlaze::new(),
        }
    }

    #[time_it("Precompute1::finish")]
    fn finish(mut self) -> DWA {
        let flush_start = std::time::Instant::now();
        self.flush_pending_token_ids();
        crate::timing!(
            "TIMING: precompute1::flush_pending_token_ids {:?}",
            flush_start.elapsed()
        );
        let run_debug_scan = std::env::var("PRECOMPUTE1_DEBUG_SCAN")
            .map(|v| v == "1")
            .unwrap_or(false)
            || crate::r#macro::is_debug_level_enabled(7);
        if run_debug_scan && !self.direct_insert {
            timeit!("precompute1::debug_scan", {
                let debug_scan_start = std::time::Instant::now();
                // Debug: print all states and transitions before processing
                crate::debug!(7, "=== NWA before flush (leaf_state={}, roots={:?}) ===", self.leaf_state, self.roots);
                for (i, state) in self.nwa.states.0.iter().enumerate() {
                    let trans_count = state.transitions.values().map(|v| v.len()).sum::<usize>();
                    let eps_count = state.epsilons.len();
                    let is_final = state.final_weight.is_some();
                    crate::debug!(7, "State {}: {} transitions, {} epsilons, final={}", i, trans_count, eps_count, is_final);
                }
                for (src, labels) in &self.pending_transitions {
                    for (label, dsts) in labels {
                        for (dst, weight) in dsts {
                            crate::debug!(7, "  {} --{}--> {} (weight: {:?})", src, label, dst, weight);
                        }
                    }
                }
                crate::debug!(7, "Pending epsilons:");
                for (src, dsts) in &self.pending_epsilons {
                    for (dst, weight) in dsts {
                        crate::debug!(7, "  {} --eps--> {} (weight: {:?})", src, dst, weight);
                    }
                }
                
                // Debug: Count transitions
                let mut total_transitions = 0;
                let mut transitions_to_leaf = 0;
                for (src, labels) in &self.pending_transitions {
                    for (label, dsts) in labels {
                        for (dst, weight) in dsts {
                            total_transitions += 1;
                            if *dst == self.leaf_state {
                                transitions_to_leaf += 1;
                                // Check if token 6 and 31 are in the same weight
                                if weight.contains(6) && weight.contains(31) {
                                    // Good - merged
                                } else if weight.contains(6) || weight.contains(31) {
                                    // crate::debug!(7, "SEPARATE: transition from {} on label {} has weight with 6={} 31={}",
                                    //     src, label, weight.contains(6), weight.contains(31));
                                }
                            }
                        }
                    }
                }
                // crate::debug!(5, "Pending transitions: {} total, {} to leaf", total_transitions, transitions_to_leaf);

                crate::debug!(5, "Precompute1 finish: debug scans in {:?}", debug_scan_start.elapsed());
            });
        }
        
        // Flush pending transitions and epsilons into the NWA
        if !self.direct_insert {
            timeit!("precompute1::flush_pending", {
                let flush_start = std::time::Instant::now();
                for (src, labels) in std::mem::take(&mut self.pending_transitions) {
                    let state = &mut self.nwa.states[src];
                    for (label, dsts) in labels {
                        let targets = state.transitions.entry(label).or_default();
                        targets.reserve(dsts.len());
                        targets.extend(dsts.into_iter());
                    }
                }
                for (src, dsts) in std::mem::take(&mut self.pending_epsilons) {
                    let state = &mut self.nwa.states[src];
                    state.epsilons.reserve(dsts.len());
                    state.epsilons.extend(dsts.into_iter());
                }
                crate::debug!(4, "Precompute1 finish: flushed pending transitions/epsilons in {:?}", flush_start.elapsed());
            });
        }

        // Convert ignored terminal transitions into epsilons.
        if !self.ignored_terminals.is_empty() && self.terminals_count > 0 {
            timeit!("precompute1::ignored_to_epsilon", {
                for state in &mut self.nwa.states.0 {
                    let mut moved_eps: HashMap<NWAStateID, Weight> = HashMap::new();
                    let mut labels_to_remove: Vec<Label> = Vec::new();
                    for (&label, targets) in &state.transitions {
                        if label >= 0 {
                            let idx = label as usize;
                            if idx < self.terminals_count
                                && self.ignored_terminals.get(idx).copied().unwrap_or(false)
                            {
                                labels_to_remove.push(label);
                                for (dst, w) in targets {
                                    let entry = moved_eps.entry(*dst).or_insert_with(Weight::zeros);
                                    if !w.is_subset_of(entry) {
                                        *entry |= w;
                                    }
                                }
                            }
                        }
                    }
                    for label in labels_to_remove {
                        state.transitions.remove(&label);
                    }
                    if !moved_eps.is_empty() {
                        state.epsilons.extend(moved_eps);
                    }
                }
            });
        }

        if let Ok(token_str) = std::env::var("DEBUG_PRECOMPUTE1_NWA_TOKEN") {
            if let Ok(token_id) = token_str.parse::<usize>() {
                let token_len = std::env::var("DEBUG_PRECOMPUTE1_NWA_TOKEN_LEN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);

                let weight_contains_token = |weight: &Weight| -> bool {
                    if self.num_tsids == 0 {
                        weight.contains(token_id)
                    } else {
                        let start = token_id.saturating_mul(self.num_tsids);
                        let end = start.saturating_add(self.num_tsids.saturating_sub(1));
                        for range in weight.ranges() {
                            let r_start = *range.start();
                            let r_end = *range.end();
                            if r_start > end {
                                break;
                            }
                            if r_end >= start {
                                return true;
                            }
                        }
                        false
                    }
                };

                let mut best_seen: HashMap<NWAStateID, usize> = HashMap::new();
                let mut found_path: Option<Vec<(NWAStateID, Option<Label>, NWAStateID)>> = None;

                fn dfs_nwa(
                    nwa: &NWA,
                    terminals_count: usize,
                    state: NWAStateID,
                    term_len: usize,
                    token_len: usize,
                    weight_contains_token: &impl Fn(&Weight) -> bool,
                    best_seen: &mut HashMap<NWAStateID, usize>,
                    path: &mut Vec<(NWAStateID, Option<Label>, NWAStateID)>,
                    found_path: &mut Option<Vec<(NWAStateID, Option<Label>, NWAStateID)>>,
                ) {
                    if found_path.is_some() {
                        return;
                    }
                    if term_len > token_len {
                        *found_path = Some(path.clone());
                        return;
                    }
                    if let Some(best) = best_seen.get(&state) {
                        if *best >= term_len {
                            return;
                        }
                    }
                    best_seen.insert(state, term_len);

                    // Epsilon transitions
                    for (next_state, w) in &nwa.states[state].epsilons {
                        if !weight_contains_token(w) {
                            continue;
                        }
                        path.push((state, None, *next_state));
                        dfs_nwa(nwa, terminals_count, *next_state, term_len, token_len, weight_contains_token, best_seen, path, found_path);
                        path.pop();
                        if found_path.is_some() {
                            return;
                        }
                    }

                    // Terminal transitions
                    for (&label, targets) in &nwa.states[state].transitions {
                        for (next_state, w) in targets {
                            if !weight_contains_token(w) {
                                continue;
                            }
                            let label_usize = label as usize;
                            let add = if label_usize < terminals_count { 1 } else { 0 };
                            path.push((state, Some(label), *next_state));
                            dfs_nwa(nwa, terminals_count, *next_state, term_len + add, token_len, weight_contains_token, best_seen, path, found_path);
                            path.pop();
                            if found_path.is_some() {
                                return;
                            }
                        }
                    }
                }

                for &start in &self.nwa.body.start_states {
                    let mut path = Vec::new();
                    dfs_nwa(
                        &self.nwa,
                        self.terminals_count,
                        start,
                        0,
                        token_len,
                        &weight_contains_token,
                        &mut best_seen,
                        &mut path,
                        &mut found_path,
                    );
                    if found_path.is_some() {
                        break;
                    }
                }

                if let Some(path) = found_path {
                    eprintln!("DEBUG_NWA_TOKEN_PATH token_id={} token_len={} steps={}", token_id, token_len, path.len());
                    for (idx, (src, label, dst)) in path.iter().enumerate() {
                        if let Some(label) = label {
                            let label_usize = *label as usize;
                            let kind = if label_usize < self.terminals_count { "terminal" } else { "tsid" };
                            eprintln!("DEBUG_NWA[{}] {} --{} {}--> {}", idx, src, kind, label, dst);
                        } else {
                            eprintln!("DEBUG_NWA[{}] {} --eps--> {}", idx, src, dst);
                        }
                    }
                } else {
                    eprintln!("DEBUG_NWA_TOKEN_PATH token_id={} token_len={} not found", token_id, token_len);
                }
            }
        }

        // Create start state with transitions to root states
        let new_start_state = timeit!("precompute1::start_state", {
            let start_state_start = std::time::Instant::now();
            let new_start_state = self.nwa.add_state();
            
            if self.num_tsids == 0 {
                // Symbol-heavy mode: create labeled transitions with Weight::all()
                // Label = tsid + terminals_count
                // Important: We need to create labels for ALL tsids (not just representatives),
                // because at runtime we'll look up by the raw tokenizer state ID.
                // All tsids that map to the same representative get their own label but point
                // to the same root state.
                let mut transitions_added = 0;
                let mut add_transition_time = std::time::Duration::ZERO;
                let mut unique_targets = std::collections::HashSet::new();
                for (tsid, rep_tsid) in &self.state_to_rep {
                    let root_key = DfsKey::new(*rep_tsid);
                    if let Some(states) = self.roots.get(&root_key) {
                        for &state in states {
                            let label = (tsid.0 + self.terminals_count) as Label;
                            let weight = Weight::from_rsb(RangeSetBlaze::from_iter([0..=self.internal_max_llm_token]));
                            let add_start = std::time::Instant::now();
                            self.nwa.add_transition(new_start_state, label, state, weight).unwrap();
                            add_transition_time += add_start.elapsed();
                            transitions_added += 1;
                            unique_targets.insert(state);
                        }
                    }
                }
                crate::debug!(4, "Precompute1 start-state breakdown (symbol-heavy): add_transition={:?}", add_transition_time);
                crate::debug!(3, "Symbol-heavy mode: added {} tsid transitions to {} unique root states", 
                    transitions_added, unique_targets.len());
            } else {
                // Weight-heavy mode: create epsilon transitions with tsid-masked weights.
                // Use INTERNAL tsid indices (0..num_tsids-1) for the weight encoding,
                // not original/raw tokenizer state IDs.
                // Group raw tsids by representative, then use the single internal index per group.
                let group_start = std::time::Instant::now();
                let mut rep_to_raw_tsids: BTreeMap<TokenizerStateID, Vec<usize>> = BTreeMap::new();
                for (tsid, rep_tsid) in &self.state_to_rep {
                    rep_to_raw_tsids.entry(*rep_tsid).or_default().push(tsid.0);
                }
                let group_time = group_start.elapsed();

                let mut mask_time = std::time::Duration::ZERO;
                let mut add_eps_time = std::time::Duration::ZERO;
                let mut group_count = 0usize;
                let mut tsid_count = 0usize;

                // Create one epsilon transition per representative with internal-tsid mask.
                // All raw states in the same equivalence class map to the same internal index.
                for (rep_tsid, raw_tsids) in rep_to_raw_tsids {
                    debug_assert!(raw_tsids.contains(&rep_tsid.0));
                    let root_key = DfsKey::new(rep_tsid);
                    if let Some(states) = self.roots.get(&root_key) {
                        group_count += 1;
                        tsid_count += raw_tsids.len();
                        // Use the internal tsid index (same for all raw tsids in this group)
                        let internal_idx = self.state_to_internal_tsid[rep_tsid.0];
                        let mask_start = std::time::Instant::now();
                        let tsid_mask = create_tsid_set_mask(
                            std::iter::once(internal_idx),
                            self.num_tsids,
                            self.internal_max_llm_token,
                        );
                        mask_time += mask_start.elapsed();
                        for &state in states {
                            let add_eps_start = std::time::Instant::now();
                            self.nwa.add_epsilon(new_start_state, state, tsid_mask.clone());
                            add_eps_time += add_eps_start.elapsed();
                        }
                    }
                }
                crate::debug!(
                    4,
                    "Precompute1 start-state breakdown: group_build={:?}, mask_build={:?}, add_epsilon={:?}, groups={}, tsids={}",
                    group_time,
                    mask_time,
                    add_eps_time,
                    group_count,
                    tsid_count,
                );
            }
            crate::debug!(4, "Precompute1 finish: added start state transitions in {:?}", start_state_start.elapsed());
            new_start_state
        });
        self.nwa.body.start_states = vec![new_start_state];

        // Stats
        // Find cases where there's multiple instances of same transition - incl symbol/epsilon transition - from one state to another, regardless of weight.
        let run_duplicate_scan = std::env::var("PRECOMPUTE1_DUPLICATE_SCAN")
            .map(|v| v == "1")
            .unwrap_or(false)
            || crate::r#macro::is_debug_level_enabled(6);
        if run_duplicate_scan {
            timeit!("precompute1::duplicate_scan", {
                let mut duplicate_transitions = 0;
                let duplicate_start = std::time::Instant::now();
                for state in &self.nwa.states.0 {
                    let mut dst_counts = HashMap::new();
                    for (dst, _) in &state.epsilons {
                        *dst_counts.entry(*dst).or_insert(0) += 1;
                    }
                    for count in dst_counts.values() {
                        if *count > 1 {
                            duplicate_transitions += count - 1;
                        }
                    }

                    for targets in state.transitions.values() {
                        let mut dst_counts = HashMap::new();
                        for (dst, _) in targets {
                            *dst_counts.entry(*dst).or_insert(0) += 1;
                        }
                        for count in dst_counts.values() {
                            if *count > 1 {
                                duplicate_transitions += count - 1;
                            }
                        }
                    }
                }
                if duplicate_transitions > 0 {
                    crate::debug!(6, "NWA: Found {} duplicate transitions (same src, dst, label)", duplicate_transitions);
                }
                crate::debug!(4, "Precompute1 finish: duplicate transition scan in {:?}", duplicate_start.elapsed());
            });
        }

        // Find cases where there's multiple instances of same transition - regardless of symbol/epsilon transition - from one state to another, regardless of weight.
        let run_parallel_scan = std::env::var("PRECOMPUTE1_PARALLEL_SCAN")
            .map(|v| v == "1")
            .unwrap_or(false)
            || crate::r#macro::is_debug_level_enabled(6);
        if run_parallel_scan {
            timeit!("precompute1::parallel_scan", {
                let mut parallel_connections = 0;
                let parallel_start = std::time::Instant::now();
                for state in &self.nwa.states.0 {
                    let mut dst_counts = HashMap::new();
                    for (dst, _) in &state.epsilons {
                        *dst_counts.entry(*dst).or_insert(0) += 1;
                    }
                    for targets in state.transitions.values() {
                        for (dst, _) in targets {
                            *dst_counts.entry(*dst).or_insert(0) += 1;
                        }
                    }

                    for count in dst_counts.values() {
                        if *count > 1 {
                            parallel_connections += 1;
                        }
                    }
                }
                if parallel_connections > 0 {
                    crate::debug!(5, "NWA: Found {} pairs of states connected by multiple transitions", parallel_connections);
                }
                crate::debug!(4, "Precompute1 finish: parallel transition scan in {:?}", parallel_start.elapsed());
            });
        }

        crate::debug!(3, "Terminal NWA: {}, num_tsids={}", 
                  self.nwa.stats(), self.num_tsids);

        let dump_terminal_automata = match std::env::var("PROFILE_TERMINAL_DWA") {
            Ok(value) => {
                let normalized = value.trim();
                normalized.is_empty()
                    || !(normalized == "0"
                        || normalized.eq_ignore_ascii_case("false")
                        || normalized.eq_ignore_ascii_case("off"))
            }
            Err(_) => false,
        };

        if dump_terminal_automata {
            println!("\n--- Terminal NWA (raw, before follow pruning / minimize / determinize) ---");
            println!("{}", self.nwa);
        }

        if self.nwa_rep_stats_enabled && !self.nwa_states_by_rep.is_empty() {
            let mut counts: Vec<(TokenizerStateID, usize)> = self
                .nwa_states_by_rep
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            counts.sort_by(|a, b| b.1.cmp(&a.1));

            let total_states: usize = counts.iter().map(|(_, v)| *v).sum();
            let reps = counts.len();
            let max = counts.first().map(|(_, v)| *v).unwrap_or(0);
            let min = counts.last().map(|(_, v)| *v).unwrap_or(0);
            let avg = total_states as f64 / reps as f64;

            let mut values: Vec<usize> = counts.iter().map(|(_, v)| *v).collect();
            values.sort_unstable();
            let p50 = values[reps / 2];
            let p90 = values[(reps * 90) / 100];
            let p99 = values[(reps * 99) / 100];

            let top: Vec<(usize, usize)> = counts
                .iter()
                .take(10)
                .map(|(k, v)| (k.0, *v))
                .collect();

            crate::debug!(
                4,
                "Terminal NWA per-rep states (created): reps={}, total_states={}, avg={:.2}, min={}, p50={}, p90={}, p99={}, max={}",
                reps,
                total_states,
                avg,
                min,
                p50,
                p90,
                p99,
                max
            );
            crate::debug!(4, "Terminal NWA per-rep top10 (tsid -> states): {:?}", top);

            let dfa = self.tokenizer.dfa();
            let mut top_details: Vec<(usize, usize, usize, usize, usize, usize)> = Vec::new();
            for (tsid, count) in &top {
                if let Some(state) = dfa.states.get(*tsid) {
                    let transitions = state.transitions.len();
                    let finalizers = state.finalizers.len();
                    let futures = state.possible_future_group_ids.len();
                    let accessible_terms = self
                        .tokenizer
                        .tokens_accessible_from_state(TokenizerStateID(*tsid))
                        .len();
                    top_details.push((*tsid, *count, transitions, finalizers, futures, accessible_terms));
                }
            }
            crate::debug!(
                4,
                "Terminal NWA per-rep top10 details (tsid, states, transitions, finalizers, futures, accessible_terms): {:?}",
                top_details
            );

            if let Some((top_tsid, _)) = top.first() {
                if let Some(depths) = self.nwa_states_by_rep_depth.get(&TokenizerStateID(*top_tsid)) {
                    let mut depth_counts: Vec<(usize, usize)> =
                        depths.iter().map(|(d, c)| (*d, *c)).collect();
                    depth_counts.sort_by_key(|(d, _)| *d);
                    let max_depth = depth_counts.last().map(|(d, _)| *d).unwrap_or(0);
                    let sample_len = depth_counts.len().min(20);
                    let sample = depth_counts[..sample_len].to_vec();
                    crate::debug!(
                        4,
                        "Terminal NWA rep {} depth histogram (first {}, max_depth={}): {:?}",
                        top_tsid,
                        sample_len,
                        max_depth,
                        sample
                    );
                }
            }
        }

        if std::env::var("DWA_DUMP_NWA").map(|v| v == "1").unwrap_or(false) {
            crate::debug!(5, "Dumping NWA to nwa_dump.json");
            let json = serde_json::to_string(&self.nwa).unwrap();
            std::fs::write("nwa_dump.json", json).unwrap();
        }

        // NWA-level always-allowed collapse only (follow pruning removed).
        if !self.always_allowed_by_label.is_empty() {
            let before_stats = self.nwa.stats();
            let collapse_start = std::time::Instant::now();
            let changed = collapse_always_allowed(
                &mut self.nwa,
                &self.always_allowed_by_label,
                self.terminals_count,
            );
            let after_stats = self.nwa.stats();
            crate::debug!(
                4,
                "Terminal NWA always-allowed collapse: changed={}, {} -> {}",
                changed,
                before_stats,
                after_stats,
            );
            crate::timing!(
                "TIMING: terminal_nwa_always_allowed_collapse {:?}",
                collapse_start.elapsed()
            );
        }

        // NWA disallowed-follows prune: prune transitions based on follow sets.
        // On by default. Disable via NWA_FOLLOW_PRUNE=0
        let nwa_follow_prune_disabled = std::env::var("NWA_FOLLOW_PRUNE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        if !nwa_follow_prune_disabled && !self.ever_allowed_by_label.is_empty() {
            let before_stats = self.nwa.stats();
            let prune_start = std::time::Instant::now();
            let changed = prune_nwa_disallowed_follows(
                &mut self.nwa,
                &self.ever_allowed_by_label,
                self.terminals_count,
            );
            let after_stats = self.nwa.stats();
            crate::debug!(
                4,
                "Terminal NWA disallowed-follows prune: changed={}, {} -> {}",
                changed,
                before_stats,
                after_stats,
            );
            crate::timing!(
                "TIMING: terminal_nwa_disallowed_follows_prune {:?}",
                prune_start.elapsed()
            );
        }

        let suffix_prune_enabled = false;
        let do_nwa_suffix_prune = false;
        if do_nwa_suffix_prune {
            if let Some(cache) = self.suffix_prune_cache.as_ref() {
                crate::debug!(4, "Terminal NWA (before suffix pruning): {}", self.nwa.stats());
                let prune_start = std::time::Instant::now();
                let (kept, pruned) = prune_nwa_with_suffix_cache(
                    &mut self.nwa,
                    cache,
                    self.terminals_count,
                );
                crate::debug!(4, "Terminal NWA suffix pruning complete. Kept={}, pruned={}", kept, pruned);
                crate::debug!(4, "Terminal NWA (after suffix pruning): {}", self.nwa.stats());
                crate::timing!(
                    "TIMING: terminal_nwa_suffix_prune {:?}",
                    prune_start.elapsed()
                );
            } else {
                crate::debug!(4, "NWA_SUFFIX_PRUNE set but missing suffix parser cache; skipping");
            }
        }

        let do_dwa_suffix_prune = false;
        let pre_dwa_suffix_prune = if do_dwa_suffix_prune {
            let suffix_prune_cache = self.suffix_prune_cache.clone();
            let terminals_count = self.terminals_count;
            Some(move |dwa: &mut DWA| {
                if let Some(cache) = suffix_prune_cache.as_ref() {
                    crate::debug!(4, "Terminal DWA (before pre-min suffix pruning): {}", dwa.stats());
                    let prune_start = std::time::Instant::now();
                    let (kept, pruned) = prune_dwa_with_suffix_cache(
                        dwa,
                        cache,
                        terminals_count,
                    );
                    crate::debug!(4, "Terminal DWA pre-min suffix pruning complete. Kept={}, pruned={}", kept, pruned);
                    crate::debug!(4, "Terminal DWA (after pre-min suffix pruning): {}", dwa.stats());
                    crate::timing!(
                        "TIMING: terminal_dwa_suffix_prune_pre_min {:?}",
                        prune_start.elapsed()
                    );
                } else {
                    crate::debug!(4, "DWA_SUFFIX_PRUNE set but missing suffix parser cache; skipping");
                }
            })
        } else {
            None
        };

        let collapse_labels = self.self_extending_labels_for_collapse.clone();
        let pre_dwa_hook = match (pre_dwa_suffix_prune, collapse_labels) {
            (None, None) => None,
            (prune_opt, collapse_opt) => Some(move |dwa: &mut DWA| {
                if let Some(labels) = collapse_opt.as_ref() {
                    let before_stats = dwa.stats();
                    let collapse_start = std::time::Instant::now();
                    let stats = collapse_self_extending_chains(dwa, labels);
                    crate::timing!(
                        "TIMING: terminal_dwa_self_ext_chain_collapse {:?}",
                        collapse_start.elapsed()
                    );
                    crate::debug!(
                        4,
                        "Terminal DWA self-ext chain collapse: {} -> {} (collapsed_states={}, rewired={}, blocked_repeats={}, pruned={}, iterations={})",
                        before_stats,
                        dwa.stats(),
                        stats.collapsed_states,
                        stats.rewired_transitions,
                        stats.blocked_repeat_transitions,
                        stats.pruned_states,
                        stats.iterations,
                    );
                }
                if let Some(prune) = prune_opt {
                    prune(dwa);
                }
            }),
        };

        // Use unified determinize_and_minimize with "Terminal" profile
        // Pipeline: NWA minimize → compress → rm_epsilon → determinize → DWA minimize
        // Expected results: 14647 → 5904 → 5904 → 889 → 189 states
        let profile_minimize_only = std::env::var("PROFILE_FACTORIZED_WEIGHT_MINIMIZE_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if profile_minimize_only {
            crate::datastructures::factorized_weight::set_factorized_weight_profile_active(true);
            crate::datastructures::factorized_weight::reset_factorized_weight_profile();
        }
        crate::debug!(4, "Terminal NWA (pre-minimize): {}", self.nwa.stats());
        crate::debug!(5, "precompute1::determinize_and_minimize start");
        let dwa = timeit!("precompute1::determinize_and_minimize", {
            self.nwa.determinize_and_minimize_with_hook(
                DeterminizeAndMinimizeProfile::Terminal,
                pre_dwa_hook,
            )
        });
        crate::debug!(5, "precompute1::determinize_and_minimize end");
        if profile_minimize_only {
            crate::datastructures::factorized_weight::flush_factorized_weight_profile("terminal_dwa_minimize");
            crate::datastructures::factorized_weight::set_factorized_weight_profile_active(false);
        }
        
        // NOTE: Stats are printed AFTER suffix grammar pruning in constraint.rs
        // This includes path counts, average path lengths, and sample paths.
        crate::debug!(4, "Terminal DWA (before suffix pruning): {}", 
                  dwa.stats());

        dwa
    }

    fn possible_matches(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key_ptr = vocab_node as *const VocabPrefixTreeNode;

        if let Some(cached_for_vocab_node) =
            self.possible_matches.borrow().get(&cache_key_ptr)
        {
            if let Some(cached_result) =
                cached_for_vocab_node.get(&tokenizer_state_id)
            {
                return cached_result.clone();
            }
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();
        let mut blocked_by_terminal: HashMap<GrammarTokenID, LLMTokenBV> = HashMap::new();
        let tokenizer_dfa = self.tokenizer.dfa();
        let has_future_in_same_greedy_group = |state_idx: usize, group_idx: usize| -> bool {
            tokenizer_dfa
                .states
                .get(state_idx)
                .map(|state| {
                    state
                        .possible_future_group_ids
                        .iter()
                        .any(|future_gid| {
                            self.terminal_to_greedy_group
                                .get(*future_gid)
                                .and_then(|group| *group)
                                == Some(group_idx)
                        })
                })
                .unwrap_or(false)
        };
        let build_state_by_width =
            |segment_bytes: &[u8], start_state: TokenizerStateID| -> Vec<Option<usize>> {
                let mut states_by_width = vec![None; segment_bytes.len() + 1];
                let mut curr_state = start_state.0;
                states_by_width[0] = Some(curr_state);
                for (idx, &byte) in segment_bytes.iter().enumerate() {
                    let Some(state) = tokenizer_dfa.states.get(curr_state) else {
                        break;
                    };
                    let Some(&next_state) = state.transitions.get(byte) else {
                        break;
                    };
                    curr_state = next_state;
                    states_by_width[idx + 1] = Some(curr_state);
                }
                states_by_width
            };

        // Include this node's own token only for terminals finalized at the current state.
        // Suppress early emissions inside greedy groups when the same group can continue.
        let own_token_id = vocab_node.token_id();
        if vocab_node.reachable_token_ids().contains(own_token_id) {
            if let Some(state) = tokenizer_dfa.states.get(tokenizer_state_id.0) {
                for finalizer_gid in state.finalizers.iter() {
                    let terminal_id = GrammarTokenID(finalizer_gid);
                    if self.merge_possible_matches_by_group {
                        if let Some(Some(group_idx)) =
                            self.terminal_to_greedy_group.get(terminal_id.0)
                        {
                            if has_future_in_same_greedy_group(tokenizer_state_id.0, *group_idx) {
                                for (blocked_tid, blocked_group) in
                                    self.terminal_to_greedy_group.iter().enumerate()
                                {
                                    if *blocked_group == Some(*group_idx) {
                                        blocked_by_terminal
                                            .entry(GrammarTokenID(blocked_tid))
                                            .or_insert_with(LLMTokenBV::zeros)
                                            .set(own_token_id, true);
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    result_map
                        .entry(terminal_id)
                        .or_insert_with(LLMTokenBV::zeros)
                        .set(own_token_id, true);
                }
            }
        }

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result =
                self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            let states_by_width = if self.merge_possible_matches_by_group && !exec_result.matches.is_empty() {
                Some(build_state_by_width(&segment_bytes, tokenizer_state_id))
                } else {
                    None
                };
                for token in &exec_result.matches {
                    let applicable_tokens = child_vocab_node.reachable_token_ids();
                    let applicable_tokens_bv: LLMTokenBV = RangeSet::from(applicable_tokens);
                    if self.merge_possible_matches_by_group {
                        if let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(token.id) {
                            if let Some(state_idx) = states_by_width
                                .as_ref()
                                .and_then(|states| states.get(token.width))
                                .and_then(|sid| *sid)
                            {
                                if has_future_in_same_greedy_group(state_idx, *group_idx) {
                                    for (blocked_tid, blocked_group) in
                                        self.terminal_to_greedy_group.iter().enumerate()
                                    {
                                        if *blocked_group == Some(*group_idx) {
                                            *blocked_by_terminal
                                                .entry(GrammarTokenID(blocked_tid))
                                                .or_insert_with(LLMTokenBV::zeros) |=
                                                applicable_tokens_bv.clone();
                                        }
                                    }
                                    continue;
                                }
                            }
                        }
                    }
                    let grammar_token_id = GrammarTokenID(token.id);
                    *result_map
                        .entry(grammar_token_id)
                        .or_insert_with(LLMTokenBV::zeros) |=
                        applicable_tokens_bv;
                }
                if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: std::collections::BTreeSet<_> = self
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(final_state_val))
                    .into_iter()
                    .collect();
                let matches_here: std::collections::BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();
                let possible_new_matches =
                    &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(
                        child_vocab_node,
                        TokenizerStateID(final_state_val),
                    );
                    for (token, bv) in next_results {
                        *result_map
                            .entry(token)
                            .or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }

        if self.merge_possible_matches_by_group && !result_map.is_empty() {
            let mut group_merged: HashMap<usize, LLMTokenBV> = HashMap::new();
            for (&tid, bv) in &result_map {
                if let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(tid.0) {
                    group_merged
                        .entry(*group_idx)
                        .or_insert_with(LLMTokenBV::zeros)
                        .bitor_assign(bv.clone());
                }
            }
            for (tid, bv) in result_map.iter_mut() {
                if let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(tid.0) {
                    if let Some(merged) = group_merged.get(group_idx) {
                        *bv = merged.clone();
                    }
                }
                if let Some(blocked) = blocked_by_terminal.get(tid) {
                    *bv -= blocked;
                }
            }
            result_map.retain(|_, bv| !bv.is_empty());
        }

        if self.merge_possible_matches_globally && !result_map.is_empty() {
            let mut merged = LLMTokenBV::zeros();
            for bv in result_map.values().cloned() {
                merged |= bv;
            }
            result_map.clear();
            result_map.insert(MERGED_POSSIBLE_MATCH_KEY, merged);
        }

        self.possible_matches
            .borrow_mut()
            .entry(cache_key_ptr)
            .or_default()
            .insert(tokenizer_state_id, result_map.clone());

        result_map
    }

    fn compute_possible_matches_for_all_states(
        &self,
    ) -> BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>> {
        // Collect unique representative states
        let unique_reps: Vec<TokenizerStateID> = self.state_to_rep.values()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        // Extract references to shared data before the parallel section
        // to avoid capturing `self` (which contains non-Send types like RefCell/Rc)
        let tokenizer = self.tokenizer;
        let terminal_to_greedy_group = &self.terminal_to_greedy_group;
        let merge_by_group = self.merge_possible_matches_by_group;
        let merge_globally = self.merge_possible_matches_globally;
        let vocab_root = &self.vocab.root;

        // Compute PM for each unique rep in parallel using rayon
        // Each thread gets its own PossibleMatchesComputer with a per-thread cache
        let rep_results: Vec<(TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>)> = unique_reps
            .par_iter()
            .map_init(
                || PossibleMatchesComputer::new(
                    tokenizer,
                    terminal_to_greedy_group,
                    merge_by_group,
                    merge_globally,
                ),
                |computer, &rep_state| {
                    let pm = computer.possible_matches(vocab_root, rep_state);
                    (rep_state, pm)
                },
            )
            .collect();

        // Build rep_possible_matches from parallel results
        let rep_possible_matches: BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>> =
            rep_results.into_iter().collect();

        // Clone from reps to all states
        let mut all_possible_matches: BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>> =
            BTreeMap::new();
        for (state, rep_state) in &self.state_to_rep {
            if let Some(rep_map) = rep_possible_matches.get(rep_state) {
                all_possible_matches.insert(*state, rep_map.clone());
            }
        }

        all_possible_matches
    }

    /// Create an expanded weight from a single token ID.
    /// Expands from N-space to N×M-space where M = num_tsids.
    /// If num_tsids == 0 (symbol-heavy mode), returns the token ID directly in N-space.
    #[inline]
    fn expanded_weight_from_item(&mut self, token_id: usize) -> Weight {
        if let Some(Some(cached)) = self.expanded_item_cache.get(token_id) {
            return cached.clone();
        }

        let rsb = if self.num_tsids == 0 {
            // Symbol-heavy mode: just use the token ID directly
            RangeSetBlaze::from_iter([token_id..=token_id])
        } else {
            // Weight-heavy mode: A single token ID in N-space becomes a range in N×M-space
            // Token i becomes positions [i*M, i*M + M - 1]
            let start = token_id * self.num_tsids;
            let end = start + self.num_tsids - 1;
            // IMPORTANT: Use [start..=end] to create from ONE range, not iterate over all integers!
            RangeSetBlaze::from_iter([start..=end])
        };
        let weight = Weight::from_rsb(rsb);
        if let Some(slot) = self.expanded_item_cache.get_mut(token_id) {
            *slot = Some(weight.clone());
        }
        weight
    }

    /// Create an expanded weight from a RangeSetBlaze of token IDs.
    /// If num_tsids <= 1 (symbol-heavy or degenerate single-tsid mode), returns the rsb directly.
    #[inline]
    fn expanded_weight_from_rsb(&mut self, rsb: &RangeSetBlaze<usize>, cache_key: Option<usize>) -> Weight {
        if rsb.is_empty() {
            return Weight::zeros();
        }
        if std::ptr::eq(rsb, &self.all_llm_tokens) {
            return self.expanded_all_weight.clone();
        }
        if let Some(key) = cache_key {
            if let Some(cached) = self.expanded_rsb_cache.get(&key) {
                return cached.clone();
            }
        }
        let expanded = if self.num_tsids <= 1 {
            rsb.clone()
        } else {
            expand_rsb(rsb, self.num_tsids)
        };
        let weight = Weight::from_rsb(expanded);
        if let Some(key) = cache_key {
            if self.expanded_rsb_cache.len() < EXPANDED_RSB_CACHE_MAX_ENTRIES {
                self.expanded_rsb_cache.insert(key, weight.clone());
            }
        }
        weight
    }

    /// Create an expanded weight from an owned RangeSetBlaze of token IDs.
    /// Uses a value cache keyed by the full RangeSetBlaze (avoids pointer-only caching).
    #[inline]
    fn expanded_weight_from_rsb_owned(&mut self, rsb: RangeSetBlaze<usize>) -> Weight {
        if rsb.is_empty() {
            return Weight::zeros();
        }
        if rsb == self.all_llm_tokens {
            return self.expanded_all_weight.clone();
        }

        if let Some(cached) = self.expanded_rsb_value_cache.get(&rsb) {
            return cached.clone();
        }
        let expanded = if self.num_tsids <= 1 {
            rsb.clone()
        } else {
            expand_rsb(&rsb, self.num_tsids)
        };
        let weight = Weight::from_rsb(expanded);
        if self.expanded_rsb_value_cache.len() < EXPANDED_RSB_VALUE_CACHE_MAX_ENTRIES {
            self.expanded_rsb_value_cache.insert(rsb, weight.clone());
        }
        weight
    }

    /// Create an expanded "all" weight (all tokens for all tsids).
    /// If num_tsids == 0 (symbol-heavy mode), returns Weight::all().
    #[inline]
    fn expanded_weight_all(&mut self) -> Weight {
        self.expanded_all_weight.clone()
    }

    #[inline]
    fn record_rep_depth_state(&mut self, rep: TokenizerStateID, depth: usize) {
        if !self.nwa_rep_stats_enabled {
            return;
        }
        let depth_map = self
            .nwa_states_by_rep_depth
            .entry(rep)
            .or_insert_with(BTreeMap::new);
        *depth_map.entry(depth).or_insert(0) += 1;
    }



    #[inline]
    fn update_live_tokens(&mut self, dst: NWAStateID, weight: &Weight) {
        *self.live_tokens.entry(dst).or_insert_with(Weight::zeros) |= weight;
    }

    /// Fast DFS-specialized tokenizer execution using flat transition tables.
    /// Avoids BTreeMap/HashMap allocation and should_terminate_early overhead.
    /// Uses generation-based match tracking to avoid per-call clearing.
    #[inline]
    fn execute_from_state_fast(
        &mut self,
        text: &[u8],
        start_state: TokenizerStateID,
    ) -> crate::dfa_u8::tokenizer_ops::ExecuteResult {
        use crate::dfa_u8::tokenizer_ops::{ExecuteResult, Token};
        let start = start_state.0;
        self.dfs_current_gen = self.dfs_current_gen.wrapping_add(1);
        let gen = self.dfs_current_gen;
        let mut current_state = start as u32;
        let mut done = false;

        // Precompute suffix byte sets for O(1) self-loop subset checks.
        // suffix_bytes[i] = bitset of all bytes occurring in text[i..].
        let len = text.len();
        self.dfs_suffix_bytes.resize(len + 1, [0u64; 4]);
        self.dfs_suffix_bytes[len] = [0u64; 4];
        for i in (0..len).rev() {
            self.dfs_suffix_bytes[i] = self.dfs_suffix_bytes[i + 1];
            let b = text[i] as usize;
            self.dfs_suffix_bytes[i][b >> 6] |= 1u64 << (b & 63);
        }

        for (pos, &byte) in text.iter().enumerate() {
            let next = self.dfs_flat_transitions[current_state as usize][byte as usize];
            if next == u32::MAX {
                done = true;
                break;
            }
            current_state = next;
            let width = pos + 1;
            // Check finalizers for this state
            for &gid in &self.dfs_finalizers[current_state as usize] {
                if gid < self.dfs_num_groups {
                    if !self.dfs_non_greedy[gid] || self.dfs_match_gen[gid] != gen {
                        self.dfs_match_widths[gid] = width;
                        self.dfs_match_gen[gid] = gen;
                    }
                }
            }

            // Self-loop early exit: if the current state self-loops on every
            // remaining byte in the text, the state won't change and greedy
            // match positions advance to text.len().
            if width < len {
                let sl = &self.dfs_self_loop_bytes[current_state as usize];
                let suffix = &self.dfs_suffix_bytes[width];
                // Check: suffix ⊆ self_loop_bytes  (i.e., no bit set in suffix that isn't in sl)
                let is_subset = (suffix[0] & !sl[0]) == 0
                    && (suffix[1] & !sl[1]) == 0
                    && (suffix[2] & !sl[2]) == 0
                    && (suffix[3] & !sl[3]) == 0;
                if is_subset {
                    let final_width = len;
                    for &gid in &self.dfs_finalizers[current_state as usize] {
                        if gid < self.dfs_num_groups {
                            if !self.dfs_non_greedy[gid] || self.dfs_match_gen[gid] != gen {
                                self.dfs_match_widths[gid] = final_width;
                                self.dfs_match_gen[gid] = gen;
                            }
                        }
                    }
                    // State doesn't change (self-loop), so end_state is current_state
                    break;
                }
            }
        }

        // Collect matches (only non-zero width, matching original filter)
        let mut matches = Vec::new();
        for gid in 0..self.dfs_num_groups {
            if self.dfs_match_gen[gid] == gen && self.dfs_match_widths[gid] > 0 {
                matches.push(Token { id: gid, width: self.dfs_match_widths[gid] });
            }
        }

        let end_state = if done { None } else { Some(current_state as usize) };
        ExecuteResult { matches, end_state }
    }

    #[inline]
    fn add_pending_token_id(
        &mut self,
        src: NWAStateID,
        label: Label,
        _dst: NWAStateID,  // Always leaf_state, stored implicitly
        token_id: usize,
    ) {
        let label_idx = label as usize;
        // Ensure outer Vec is large enough for this src state
        if src >= self.pending_token_ids.len() {
            self.pending_token_ids.resize_with(src + 1, Vec::new);
        }
        // Ensure inner Vec is large enough for this label
        let inner = &mut self.pending_token_ids[src];
        if label_idx >= inner.len() {
            inner.resize_with(label_idx + 1, Vec::new);
        }
        inner[label_idx].push(token_id);
    }

    fn flush_pending_token_ids(&mut self) {
        let dst = self.leaf_state;
        let all_empty = self.pending_token_ids.iter().all(|v| v.is_empty());
        if all_empty {
            return;
        }
        let pending = std::mem::take(&mut self.pending_token_ids);
        for (src, labels_vec) in pending.into_iter().enumerate() {
            for (label_idx, mut token_ids) in labels_vec.into_iter().enumerate() {
                if token_ids.is_empty() { continue; }
                let label = label_idx as Label;
                token_ids.sort_unstable();
                token_ids.dedup();
                let rsb = RangeSetBlaze::from_iter(token_ids.into_iter().map(|t| t..=t));
                let weight = self.expanded_weight_from_rsb_owned(rsb);
                if self.direct_insert {
                    let state = &mut self.nwa.states[src];
                    state.transitions.entry(label).or_default().push((dst, weight));
                    continue;
                }
                self.pending_transitions
                    .entry(src)
                    .or_default()
                    .entry(label)
                    .or_default()
                    .entry(dst)
                    .and_modify(|w| *w |= &weight)
                    .or_insert(weight);
            }
        }
    }

    #[inline]
    fn add_pending_transition_no_live(
        &mut self,
        src: NWAStateID,
        label: Label,
        dst: NWAStateID,
        weight: Weight,
    ) {
        if self.direct_insert {
            let state = &mut self.nwa.states[src];
            state.transitions.entry(label).or_default().push((dst, weight));
            return;
        }
        self.pending_transitions
            .entry(src)
            .or_default()
            .entry(label)
            .or_default()
            .entry(dst)
            .and_modify(|w| *w |= &weight)
            .or_insert(weight);
    }

    fn add_pending_transition(&mut self, src: NWAStateID, label: Label, dst: NWAStateID, weight: Weight) {
        self.update_live_tokens(dst, &weight);
        self.add_pending_transition_no_live(src, label, dst, weight);
    }

    fn add_pending_epsilon(&mut self, src: NWAStateID, dst: NWAStateID, weight: Weight) {
        self.update_live_tokens(dst, &weight);
        if self.direct_insert {
            let state = &mut self.nwa.states[src];
            state.epsilons.push((dst, weight));
            return;
        }
        self.pending_epsilons
            .entry(src)
            .or_default()
            .entry(dst)
            .and_modify(|w| *w |= &weight)
            .or_insert(weight);
    }

    fn run_dfs(&mut self) {
        let assoc = self.roots.clone();
        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!("Starting precompute DFS for {} tokenizer states", self.roots.len());
        }
        let vocab = std::mem::replace(&mut self.vocab, VocabPrefixTree::new());
        
        // Count vocab nodes for progress tracking
        let vocab_node_count = count_vocab_nodes(&vocab.root);
        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!("Vocab tree has {} nodes", vocab_node_count);
        }
        
        // Reset DFS profiling counters
        self.dfs_profile_execute = std::time::Duration::ZERO;
        self.dfs_profile_weight = std::time::Duration::ZERO;
        self.dfs_profile_pending = std::time::Duration::ZERO;
        self.dfs_profile_clone = std::time::Duration::ZERO;
        self.dfs_profile_match = std::time::Duration::ZERO;
        self.dfs_profile_endstate = std::time::Duration::ZERO;
        self.dfs_profile_nodes = 0;
        self.dfs_profile_segments = 0;
        self.dfs_profile_iters = 0;
        self.dfs_profile_state_key_iters = 0;
        self.dfs_profile_matches = 0;
        self.dfs_profile_endstate_adds = 0;
        self.dfs_profile_endstate_events = 0;
        self.dfs_profile_enabled = std::env::var("PROFILE_DFS").is_ok() || crate::r#macro::is_debug_level_enabled(3);

        let dfs_start = std::time::Instant::now();
        self.dfs(&vocab.root, assoc);

        // Finalize batched leaf_state live_tokens update
        if !self.dfs_leaf_token_ids.is_empty() {
            let leaf_ids = std::mem::take(&mut self.dfs_leaf_token_ids);
            let weight = self.expanded_weight_from_rsb_owned(leaf_ids);
            self.update_live_tokens(self.leaf_state, &weight);
        }

        let dfs_time = dfs_start.elapsed();
        self.vocab = vocab;
        self.pb.finish();

        if std::env::var("PROFILE_DFS").is_ok() || crate::r#macro::is_debug_level_enabled(3) {
            let other = dfs_time.saturating_sub(
                self.dfs_profile_execute + self.dfs_profile_weight +
                self.dfs_profile_pending + self.dfs_profile_clone +
                self.dfs_profile_match + self.dfs_profile_endstate
            );
            eprintln!("DFS PROFILE: total={:?}", dfs_time);
            eprintln!("  execute_from_state: {:?}", self.dfs_profile_execute);
            eprintln!("  match_processing: {:?}", self.dfs_profile_match);
            eprintln!("  end_state_handling: {:?}", self.dfs_profile_endstate);
            eprintln!("  weight_ops: {:?}", self.dfs_profile_weight);
            eprintln!("  pending_ops: {:?}", self.dfs_profile_pending);
            eprintln!("  clone_assoc: {:?}", self.dfs_profile_clone);
            eprintln!("  other: {:?}", other);
            eprintln!("  nodes={}, segments={}, iters={}, state_key_iters={}, matches={}, endstate_events={}, endstate_adds={}",
                self.dfs_profile_nodes, self.dfs_profile_segments, self.dfs_profile_iters,
                self.dfs_profile_state_key_iters, self.dfs_profile_matches,
                self.dfs_profile_endstate_events, self.dfs_profile_endstate_adds);
        }

        crate::timing!("TIMING: precompute1::run_dfs::dfs {:?}", dfs_time);
        crate::debug!(5, "Precomputation complete");
    }

    fn dfs(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<DfsKey, SourceStates>,
    ) {
        let mut total_pending_iters = 0usize;
        let base_depth = vocab_node.prefix_length();
        let skip_pm_prune = true; // Always skip PM during DFS; computed in parallel post-DFS
        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            crate::debug!(7, "=== Processing vocab segment: {:?} (token_id={}) ===",
                String::from_utf8_lossy(segment_bytes), child_vocab_node.token_id());
            crate::debug!(7, "Initial assoc_by_state: {:?}", assoc_by_state);
            
            let mut next_level_assoc: BTreeMap<DfsKey, SourceStates> =
                BTreeMap::new();

            // Queue: pos -> TokenizerState -> (NWAState -> ContextTokens)
            let mut pending: BTreeMap<usize, BTreeMap<DfsKey, SourceStates>> = BTreeMap::new();
            let clone_start = if self.dfs_profile_enabled { Some(std::time::Instant::now()) } else { None };
            pending.insert(0, assoc_by_state.clone());
            if let Some(t) = clone_start { self.dfs_profile_clone += t.elapsed(); }
            self.dfs_profile_segments += 1;

            let child_reachable = child_vocab_node.reachable_token_ids();
            let child_token_id = child_vocab_node.token_id();

            // Caches possible matches for end states to prune edge_bv
            let mut possible_matches_at_end_cache: HashMap<
                TokenizerStateID,
                BTreeMap<GrammarTokenID, LLMTokenBV>,
            > = HashMap::new();

            let mut segment_pending_iters = 0usize;
            self.dfs_profile_nodes += 1;
            // Pre-allocate Vecs outside inner loop to avoid 1.98M allocations
            let mut leaf_labels: Vec<Label> = Vec::new();
            let mut cont_transitions: Vec<(Label, NWAStateID, Weight)> = Vec::new();
            loop {
                let Some((pos, states_at_pos)) = pending.pop_first() else {
                    break;
                };
                segment_pending_iters += 1;
                total_pending_iters += 1;
                self.dfs_profile_iters += 1;
                crate::debug!(7, "--- Position {} (segment len={}) ---", pos, segment_bytes.len());
                crate::debug!(7, "States at pos: {:?}", states_at_pos);
                
                // If we reached the end of the segment, these states are ready for the next vocab node
                if pos == segment_bytes.len() {
                    crate::debug!(7, "  -> End of segment, propagating to next level");
                    for (state_key, nodes) in states_at_pos {
                        let entry = next_level_assoc.entry(state_key).or_insert_with(SmallVec::new);
                        entry.extend(nodes);
                    }
                    continue;
                }

                for (state_key, nodes) in states_at_pos {
                    let tokenizer_state_id = state_key.tokenizer_state;
                    let num_sources = nodes.len();

                    let slice = &segment_bytes[pos..];
                      let exec_start = if self.dfs_profile_enabled { Some(std::time::Instant::now()) } else { None };
                      let exec_result = self.execute_from_state_fast(slice, tokenizer_state_id);
                      if let Some(t) = exec_start { self.dfs_profile_execute += t.elapsed(); }
                      // Only compute states_by_width when needed for greedy group suppression
                      let states_by_width = if !self.terminal_to_greedy_group.is_empty() && !exec_result.matches.is_empty() {
                          let tokenizer_dfa = self.tokenizer.dfa();
                          let mut by_width = vec![None; slice.len() + 1];
                          let mut curr_state = tokenizer_state_id.0;
                          by_width[0] = Some(curr_state);
                          for (idx, &byte) in slice.iter().enumerate() {
                              let Some(state) = tokenizer_dfa.states.get(curr_state) else {
                                  break;
                              };
                              let Some(&next_state) = state.transitions.get(byte) else {
                                  break;
                              };
                              curr_state = next_state;
                              by_width[idx + 1] = Some(curr_state);
                          }
                          Some(by_width)
                      } else {
                          None
                      };

                      crate::debug!(
                          7,
                          "  Tokenizer on {:?} from state {:?} (sources={}): matches={:?}, end_state={:?}",
                        String::from_utf8_lossy(slice),
                        tokenizer_state_id,
                        num_sources,
                        exec_result.matches,
                        exec_result.end_state,
                    );

                    let possible_matches_at_end = if skip_pm_prune {
                        possible_matches_at_end_cache
                            .entry(TokenizerStateID(usize::MAX))
                            .or_default()
                    } else if let Some(end_val) = exec_result.end_state {
                        let ts = TokenizerStateID(end_val);
                        possible_matches_at_end_cache
                            .entry(ts)
                            .or_insert_with(|| {
                                self.possible_matches(child_vocab_node, ts)
                            })
                    } else {
                        // Dummy empty map
                        possible_matches_at_end_cache
                            .entry(TokenizerStateID(usize::MAX)) // Arbitrary key that won't be hit
                            .or_default()
                    };

                    leaf_labels.clear();
                    cont_transitions.clear();
                      let mut leaf_weight: Option<Weight> = None;
                      self.dfs_profile_state_key_iters += 1;
                      self.dfs_profile_matches += exec_result.matches.len();

                      let match_start = if self.dfs_profile_enabled { Some(std::time::Instant::now()) } else { None };
                      // 1. Handle Matches -> Transitions to Initial State (per state_key)
                      for match_info in &exec_result.matches {
                          let terminal_id = GrammarTokenID(match_info.id);
                            let next_pos = pos + match_info.width;
                            if next_pos < segment_bytes.len() {
                                if let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(terminal_id.0) {
                                    if let Some(state_idx) = states_by_width
                                        .as_ref()
                                        .and_then(|sbw| sbw.get(match_info.width))
                                        .and_then(|sid| *sid)
                                    {
                                        let should_suppress = self
                                            .tokenizer
                                            .dfa()
                                            .states
                                            .get(state_idx)
                                            .map(|state| {
                                                state.possible_future_group_ids.iter().any(|future_gid| {
                                                    self.terminal_to_greedy_group
                                                        .get(*future_gid)
                                                        .and_then(|group| *group)
                                                        == Some(*group_idx)
                                                })
                                            })
                                            .unwrap_or(false);
                                        if should_suppress {
                                            crate::debug!(
                                                7,
                                                "      -> Skip match (greedy continuation): terminal_id={}, width={}, state_after_match={}",
                                                terminal_id.0,
                                                match_info.width,
                                                state_idx
                                            );
                                            continue;
                                        }
                                    }
                                }
                            }

                        // Leaf check: if match consumes remainder of segment
                        if next_pos == segment_bytes.len() {
                            let leaf = self.leaf_state;
                            let weight = leaf_weight
                                .get_or_insert_with(|| self.expanded_weight_from_item(child_token_id));
                            crate::debug!(
                                7,
                                "      -> LEAF transition ({} sources): --{}--> {} (leaf_state), weight={:?}",
                                num_sources,
                                terminal_id.0,
                                leaf,
                                weight,
                            );
                            leaf_labels.push(terminal_id.0 as Label);
                        }

                        // Continuation logic
                        let final_bv: std::borrow::Cow<RangeSetBlaze<usize>> = if next_pos == segment_bytes.len() {
                            let mut edge_bv = child_reachable.clone();
                            edge_bv.remove(child_token_id);
                            let possible_match = possible_matches_at_end
                                .get(&terminal_id)
                                .or_else(|| possible_matches_at_end.get(&MERGED_POSSIBLE_MATCH_KEY));
                            if let Some(pm) = possible_match {
                                edge_bv = &edge_bv - pm.inner.as_ref();
                            }
                            crate::debug!(7, "      Continuation at end of segment: edge_bv={:?} (removed child_token_id={}, pm={:?})",
                                edge_bv.iter().collect::<Vec<_>>(), child_token_id, possible_match.map(|pm| &pm.inner));
                            std::borrow::Cow::Owned(edge_bv)
                        } else {
                            crate::debug!(7, "      Continuation (not end): using child_reachable={:?}", child_reachable.iter().collect::<Vec<_>>());
                            std::borrow::Cow::Borrowed(child_reachable)
                        };

                        if final_bv.is_empty() {
                            crate::debug!(7, "      -> Skip continuation (empty edge_bv)");
                            continue;
                        }

                        let dest_map = pending.entry(next_pos).or_default();

                        let initial_tsid = self.tokenizer.initial_state_id();
                        // Use expanded weight from rsb
                        let weight_start = if self.dfs_profile_enabled { Some(std::time::Instant::now()) } else { None };
                        let weight = match final_bv {
                            std::borrow::Cow::Borrowed(rsb) => {
                                self.expanded_weight_from_rsb(rsb, Some(rsb as *const _ as usize))
                            }
                            std::borrow::Cow::Owned(rsb) => self.expanded_weight_from_rsb_owned(rsb),
                        };
                        if let Some(t) = weight_start { self.dfs_profile_weight += t.elapsed(); }

                        let target_entry = dest_map.entry(DfsKey::new(initial_tsid));
                        let target = match target_entry {
                            std::collections::btree_map::Entry::Occupied(mut o) => {
                                if let Some(&existing) = o.get().first() {
                                    crate::debug!(7, "      -> Continuation to existing state: target={}", existing);
                                    existing
                                } else {
                                    let t = self.nwa.add_state();
                                    if self.nwa_rep_stats_enabled {
                                        *self.nwa_states_by_rep.entry(initial_tsid).or_insert(0) += 1;
                                    }
                                    self.record_rep_depth_state(initial_tsid, base_depth + next_pos);
                                    crate::debug!(7, "      -> Created new continuation state: target={}", t);
                                    o.get_mut().push(t);
                                    t
                                }
                            }
                            std::collections::btree_map::Entry::Vacant(v) => {
                                let t = self.nwa.add_state();
                                if self.nwa_rep_stats_enabled {
                                    *self.nwa_states_by_rep.entry(initial_tsid).or_insert(0) += 1;
                                }
                                self.record_rep_depth_state(initial_tsid, base_depth + next_pos);
                                crate::debug!(7, "      -> Created new continuation state: target={}", t);
                                let mut set = SmallVec::new();
                                set.push(t);
                                v.insert(set);
                                t
                            }
                        };


                        crate::debug!(
                            7,
                            "      -> CONT transition ({} sources): --{}--> {}, weight={:?}",
                            num_sources,
                            terminal_id.0,
                            target,
                            weight,
                        );
                        cont_transitions.push((terminal_id.0 as Label, target, weight));
                    }
                    if let Some(t) = match_start { self.dfs_profile_match += t.elapsed(); }

                    if leaf_weight.is_some() {
                        self.dfs_leaf_token_ids.insert(child_token_id);
                    }

                    let pending_start = if self.dfs_profile_enabled { Some(std::time::Instant::now()) } else { None };
                    for &src_node in nodes.iter() {
                        for label in &leaf_labels {
                            self.add_pending_token_id(src_node, *label, self.leaf_state, child_token_id);
                        }
                        for (label, dst, weight) in &cont_transitions {
                            self.add_pending_transition(src_node, *label, *dst, weight.clone());
                        }
                    }
                    if let Some(t) = pending_start { self.dfs_profile_pending += t.elapsed(); }

                    // 2. Handle End State -> Continuation
                    crate::debug!(7, "  End state handling: end_state={:?}", exec_result.end_state);
                    let endstate_start = if self.dfs_profile_enabled { Some(std::time::Instant::now()) } else { None };
                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state = TokenizerStateID(end_state_val);

                        // Cache accessible terminal labels directly (avoids 1.93M Vec<Label> allocations)
                        let end_labels: std::rc::Rc<Vec<Label>> = if let Some(cached) = self.accessible_terminals_cache.get(&final_tokenizer_state) {
                            cached.clone()
                        } else {
                            let result = std::rc::Rc::new(self.tokenizer.tokens_accessible_from_state(final_tokenizer_state)
                                .into_iter().map(|t| t.0 as Label).collect::<Vec<_>>());
                            self.accessible_terminals_cache.insert(final_tokenizer_state, result.clone());
                            result
                        };

                        if !end_labels.is_empty() {
                            self.dfs_profile_endstate_events += 1;
                            self.dfs_profile_endstate_adds += nodes.len() * end_labels.len();
                            // Batch: collect token ID for deferred update_live_tokens
                            self.dfs_leaf_token_ids.insert(child_token_id);
                            // Pre-size outer Vec to avoid repeated resize checks
                            let max_src = nodes.iter().copied().max().unwrap_or(0);
                            if max_src >= self.pending_token_ids.len() {
                                self.pending_token_ids.resize_with(max_src + 1, Vec::new);
                            }
                            let max_label = *end_labels.iter().max().unwrap() as usize;
                            for &src_node in nodes.iter() {
                                let inner = &mut self.pending_token_ids[src_node];
                                if max_label >= inner.len() {
                                    inner.resize_with(max_label + 1, Vec::new);
                                }
                                for &label in end_labels.iter() {
                                    inner[label as usize].push(child_token_id);
                                }
                            }
                        }

                        let entry = next_level_assoc
                            .entry(DfsKey::new(final_tokenizer_state))
                            .or_insert_with(SmallVec::new);
                        entry.extend(nodes);
                    }
                    if let Some(t) = endstate_start { self.dfs_profile_endstate += t.elapsed(); }
                }
            }

            if crate::r#macro::is_debug_level_enabled(6)
                && std::env::var("SUPPRESS_PRECOMPUTE1_DFS_LOGS").is_err()
            {
                eprintln!(
                    "DFS segment done: segment_len={}, pending_iters={}, next_level_assoc={}",
                    segment_bytes.len(),
                    segment_pending_iters,
                    next_level_assoc.len()
                );
            }

            crate::debug!(7, "=== Done processing segment {:?}, next_level_assoc={:?} ===",
                String::from_utf8_lossy(segment_bytes), next_level_assoc);

            if !next_level_assoc.is_empty() {
                self.dfs(child_vocab_node, next_level_assoc);
            }
        }

        if crate::r#macro::is_debug_level_enabled(6)
            && std::env::var("SUPPRESS_PRECOMPUTE1_DFS_LOGS").is_err()
        {
            eprintln!("DFS total pending iterations: {}", total_pending_iters);
        }
    }
}

pub(crate) fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
}

/// Check if weight-heavy mode is enabled via environment variable.
/// Returns true (weight-heavy enabled) unless DISABLE_WEIGHT_HEAVY=1 is set.
pub fn is_weight_heavy_enabled() -> bool {
    std::env::var("DISABLE_WEIGHT_HEAVY").map(|v| v != "1").unwrap_or(true)
}

// Public entry point wrapper
#[time_it("run_precompute1")]
pub fn run_precompute1(
    tokenizer: &Tokenizer,
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminals_count: usize,
    state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
    suffix_prune_cache: Option<Arc<SuffixParserCache>>,
    self_extending_labels_for_collapse: Option<Arc<HashSet<Label>>>,
    ignored_terminals: Arc<Vec<bool>>,
    allowed_follows_by_label: Arc<Vec<Vec<Label>>>,
    always_allowed_by_label: Arc<Vec<Vec<Label>>>,
    ever_allowed_by_label: Arc<Vec<Vec<Label>>>,
    terminal_to_greedy_group: Vec<Option<usize>>,
) -> DWA {
    // Compute num_tsids and state_to_internal_tsid from state_to_rep
    let (num_tsids, state_to_internal_tsid) = if is_weight_heavy_enabled() {
        let mut representatives: Vec<usize> = state_to_rep.values().map(|r| r.0).collect();
        representatives.sort_unstable();
        representatives.dedup();
        let rep_count = representatives.len();
        let rep_to_internal: std::collections::HashMap<usize, usize> = representatives
            .iter().enumerate().map(|(i, &r)| (r, i)).collect();
        let num_raw_states = tokenizer.dfa().states.len();
        let mut mapping = vec![0usize; num_raw_states];
        for (&state, &rep) in &state_to_rep {
            if let Some(&internal) = rep_to_internal.get(&rep.0) {
                if state.0 < num_raw_states {
                    mapping[state.0] = internal;
                }
            }
        }
        (rep_count, mapping)
    } else {
        (0, vec![])
    };

    run_precompute1_with_possible_matches(
        tokenizer,
        internal_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        state_to_rep,
        num_tsids,
        state_to_internal_tsid,
        suffix_prune_cache,
        self_extending_labels_for_collapse,
        ignored_terminals,
        allowed_follows_by_label,
        always_allowed_by_label,
        ever_allowed_by_label,
        terminal_to_greedy_group,
    )
    .0
}

#[time_it("run_precompute1_with_possible_matches")]
pub fn run_precompute1_with_possible_matches(
    tokenizer: &Tokenizer,
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminals_count: usize,
    state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
    num_tsids: usize,
    state_to_internal_tsid: Vec<usize>,
    suffix_prune_cache: Option<Arc<SuffixParserCache>>,
    self_extending_labels_for_collapse: Option<Arc<HashSet<Label>>>,
    ignored_terminals: Arc<Vec<bool>>,
    allowed_follows_by_label: Arc<Vec<Vec<Label>>>,
    always_allowed_by_label: Arc<Vec<Vec<Label>>>,
    ever_allowed_by_label: Arc<Vec<Vec<Label>>>,
    terminal_to_greedy_group: Vec<Option<usize>>,
) -> (
    DWA,
    BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>,
) {
    let _ = allowed_follows_by_label;

    // Ensure global dimensions are set when run_precompute1 is called directly (e.g., tests).
    crate::datastructures::set_global_dims_all_threads(
        internal_max_llm_token,
        if num_tsids > 0 { num_tsids } else { 1 },
    );

    let profile_minimize_only = std::env::var("PROFILE_FACTORIZED_WEIGHT_MINIMIZE_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if profile_minimize_only {
        crate::datastructures::factorized_weight::set_factorized_weight_profile_active(false);
    }

    let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut seen_internal_ids = std::collections::HashSet::new();

    for (bytes, id) in internal_llm_token_map {
        if seen_internal_ids.insert(id.0) {
            representative_llm_token_map.insert(bytes.clone(), *id);
        }
    }

    let setup_start = std::time::Instant::now();
    let mut helper = timeit!("precompute1::setup", {
        Precomputer1::new(
            tokenizer,
            &representative_llm_token_map,
            internal_max_llm_token,
            terminals_count,
            state_to_rep,
            num_tsids,
            state_to_internal_tsid,
            suffix_prune_cache,
            self_extending_labels_for_collapse,
            ignored_terminals,
            always_allowed_by_label,
            ever_allowed_by_label,
            terminal_to_greedy_group,
        )
    });
    crate::timing!("PHASE_TIMING: precompute1::setup = {:?}", setup_start.elapsed());

    let dfs_start = std::time::Instant::now();
    timeit!("precompute1::dfs", {
        helper.run_dfs();
    });
    crate::timing!("PHASE_TIMING: precompute1::dfs = {:?}", dfs_start.elapsed());

    let possible_matches_start = std::time::Instant::now();
    let possible_matches = helper.compute_possible_matches_for_all_states();
    crate::timing!("PHASE_TIMING: precompute1::possible_matches = {:?}", possible_matches_start.elapsed());

    let finish_start = std::time::Instant::now();
    let result = timeit!("precompute1::finish", {
        helper.finish()
    });
    crate::timing!("PHASE_TIMING: precompute1::finish = {:?}", finish_start.elapsed());
    (result, possible_matches)
}

#[cfg(test)]
pub(crate) fn run_precompute1_nwa_for_tests(
    tokenizer: &Tokenizer,
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminals_count: usize,
    state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
) -> NWA {
    // Compute num_tsids and state_to_internal_tsid from state_to_rep
    let (num_tsids, state_to_internal_tsid) = if is_weight_heavy_enabled() {
        let mut representatives: Vec<usize> = state_to_rep.values().map(|r| r.0).collect();
        representatives.sort_unstable();
        representatives.dedup();
        let rep_count = representatives.len();
        let rep_to_internal: std::collections::HashMap<usize, usize> = representatives
            .iter().enumerate().map(|(i, &r)| (r, i)).collect();
        let num_raw_states = tokenizer.dfa().states.len();
        let mut mapping = vec![0usize; num_raw_states];
        for (&state, &rep) in &state_to_rep {
            if let Some(&internal) = rep_to_internal.get(&rep.0) {
                if state.0 < num_raw_states {
                    mapping[state.0] = internal;
                }
            }
        }
        (rep_count, mapping)
    } else {
        (0, vec![])
    };

    crate::datastructures::set_global_dims_all_threads(
        internal_max_llm_token,
        if num_tsids > 0 { num_tsids } else { 1 },
    );

    let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut seen_internal_ids = std::collections::HashSet::new();

    for (bytes, id) in internal_llm_token_map {
        if seen_internal_ids.insert(id.0) {
            representative_llm_token_map.insert(bytes.clone(), *id);
        }
    }

    let ignored_terminals = Arc::new(vec![false; terminals_count]);
    let always_allowed_by_label = Arc::new(Vec::new());
    let ever_allowed_by_label = Arc::new(Vec::new());

    let mut helper = Precomputer1::new(
        tokenizer,
        &representative_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        state_to_rep,
        num_tsids,
        state_to_internal_tsid,
        None,
        None,
        ignored_terminals,
        always_allowed_by_label,
        ever_allowed_by_label,
        vec![None; terminals_count],
    );

    helper.run_dfs();

    helper.flush_pending_token_ids();

    if !helper.direct_insert {
        for (src, labels) in std::mem::take(&mut helper.pending_transitions) {
            let state = &mut helper.nwa.states[src];
            for (label, dsts) in labels {
                let targets = state.transitions.entry(label).or_default();
                targets.reserve(dsts.len());
                targets.extend(dsts.into_iter());
            }
        }
        for (src, dsts) in std::mem::take(&mut helper.pending_epsilons) {
            let state = &mut helper.nwa.states[src];
            state.epsilons.reserve(dsts.len());
            state.epsilons.extend(dsts.into_iter());
        }
    }

    helper.nwa
}
