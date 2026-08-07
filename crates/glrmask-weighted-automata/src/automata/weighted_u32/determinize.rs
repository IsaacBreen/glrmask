//! Weighted determinization for acyclic NWAs.
//!
//! Cyclic inputs are rejected and must be handled by the caller.

use std::collections::{BTreeMap, VecDeque, hash_map::Entry as HashMapEntry};
use std::time::Instant;
use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::dwa::{DWA, DWAState};
use super::equivalence::find_difference;
use super::nwa::NWA;
use crate::ds::weight::{SharedTokenSet, ScopedWeightOpCache, Weight, WeightIntersectionIndex};
use crate::GlrMaskError;

const MAX_INDEXED_FINAL_PATH_RANGES: usize = 8;
const MIN_INDEXED_FINAL_WEIGHT_RANGES: usize = 32;

/// Outgoing edges from one NWA state, grouped by the exact shared `Weight`.
///
/// For a path weight `p`, every edge in a group contributes `p ∩ w`, so this
/// computes that intersection once and distributes the immutable result to the
/// group's distinct labels and destinations.
#[derive(Clone)]
struct WeightGroupedEdge {
    label: i32,
    dst: u32,
    /// This source label has exactly one target and that target is epsilon-free.
    /// A one-entry determinized subset can emit it without the label staging map.
    direct_singleton: bool,
}

#[derive(Clone)]
struct WeightGroupedTransitions {
    weight: Weight,
    edges: Vec<WeightGroupedEdge>,
}

fn build_weight_grouped_transitions(nwa: &NWA) -> Vec<Vec<WeightGroupedTransitions>> {
    nwa.states()
        .iter()
        .map(|state| {
            let mut groups = FxHashMap::<usize, usize>::default();
            let mut result = Vec::<WeightGroupedTransitions>::new();
            for (&label, targets) in &state.transitions {
                let label_has_single_target = targets.len() == 1;
                for (dst, weight) in targets {
                    let direct_singleton_edge = label_has_single_target
                        && nwa.states()[*dst as usize].epsilons.is_empty();
                    let key = weight.ptr_key();
                    let index = if let Some(&existing) = groups.get(&key) {
                        existing
                    } else {
                        let created = result.len();
                        groups.insert(key, created);
                        result.push(WeightGroupedTransitions {
                            weight: weight.clone(),
                            edges: Vec::new(),
                        });
                        created
                    };
                    result[index].edges.push(WeightGroupedEdge {
                        label,
                        dst: *dst,
                        direct_singleton: direct_singleton_edge,
                    });
                }
            }
            result
        })
        .collect()
}

fn union_state_weight(weights: &mut FxHashMap<u32, Weight>, state_id: u32, add: Weight) {
    if add.is_empty() {
        return;
    }

    match weights.entry(state_id) {
        HashMapEntry::Occupied(mut occupied) => {
            let existing = occupied.get_mut();
            *existing = existing.union(&add);
        }
        HashMapEntry::Vacant(vacant) => {
            vacant.insert(add);
        }
    }
}

fn final_weight_intersection(
    path_weight: &Weight,
    state_final: &Weight,
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> Weight {
    if path_weight.outer_range_count() <= max_indexed_path_ranges {
        if let Some(index) = final_weight_indices.get(&state_final.ptr_key()) {
            return path_weight.intersection_with_index(index);
        }
    }
    path_weight.intersection(state_final)
}

fn subset_final_weight(
    nwa: &NWA,
    subset_entries: &[(u32, Weight)],
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> Weight {
    subset_entries.iter().fold(Weight::empty(), |final_weight, (state_id, path_weight)| {
        let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
            return final_weight;
        };

        final_weight.union(&final_weight_intersection(
            path_weight,
            state_final,
            final_weight_indices,
            max_indexed_path_ranges,
        ))
    })
}

#[derive(Default)]
struct FinalWeightProfile {
    subsets: usize,
    subset_entries: usize,
    final_entries: usize,
    nonempty_contributions: usize,
    max_final_entries: usize,
    intersection_ms: f64,
    union_ms: f64,
}

impl FinalWeightProfile {
    fn merge(&mut self, other: Self) {
        self.subsets += other.subsets;
        self.subset_entries += other.subset_entries;
        self.final_entries += other.final_entries;
        self.nonempty_contributions += other.nonempty_contributions;
        self.max_final_entries = self.max_final_entries.max(other.max_final_entries);
        self.intersection_ms += other.intersection_ms;
        self.union_ms += other.union_ms;
    }
}

fn subset_final_weight_profiled(
    nwa: &NWA,
    subset_entries: &[(u32, Weight)],
    final_weight_indices: &FxHashMap<usize, WeightIntersectionIndex>,
    max_indexed_path_ranges: usize,
) -> (Weight, FinalWeightProfile) {
    let mut profile = FinalWeightProfile {
        subsets: 1,
        subset_entries: subset_entries.len(),
        ..FinalWeightProfile::default()
    };
    let mut final_weight = Weight::empty();

    for (state_id, path_weight) in subset_entries {
        let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
            continue;
        };
        profile.final_entries += 1;
        profile.max_final_entries = profile.max_final_entries.max(profile.final_entries);

        let intersection_started_at = Instant::now();
        let contribution = final_weight_intersection(
            path_weight,
            state_final,
            final_weight_indices,
            max_indexed_path_ranges,
        );
        profile.intersection_ms += intersection_started_at.elapsed().as_secs_f64() * 1000.0;
        if contribution.is_empty() {
            continue;
        }
        profile.nonempty_contributions += 1;

        let union_started_at = Instant::now();
        final_weight = final_weight.union(&contribution);
        profile.union_ms += union_started_at.elapsed().as_secs_f64() * 1000.0;
    }

    (final_weight, profile)
}

fn seed_start_subset(nwa: &NWA) -> FxHashMap<u32, Weight> {
    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        union_state_weight(&mut start_subset, state_id, Weight::all());
    }
    start_subset
}

/// Exact determinization for an NWA whose structural accepted language has
/// labelled depth at most two.
///
/// Rather than constructing residual subsets, evaluate the accepted weight of
/// every word of length 0, 1, or 2 directly. The resulting prefix DWA stores
/// the exact one-label word weight as the depth-1 final weight, and the union
/// of all accepted words below a first label on its start edge. Two-label word
/// weights live on the second edge into one shared all-final leaf.
///
/// The caller must establish the structural depth bound. This function keeps
/// an acyclicity check because the reverse language DP requires a DAG.
pub fn determinize_depth2(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "depth-2 weighted determinization supports only acyclic NWAs".into(),
        ));
    }

    let profile = determinize_profile_enabled();
    let total_started_at = profile.then(Instant::now);
    type Word = SmallVec<[i32; 2]>;

    fn union_word_weight(language: &mut BTreeMap<Word, Weight>, word: Word, add: Weight) {
        if add.is_empty() {
            return;
        }
        match language.entry(word) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                *occupied.get_mut() = occupied.get().union(&add);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(add);
            }
        }
    }

    let mut in_degree = vec![0u32; nwa.states().len()];
    for state in nwa.states() {
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state_id, degree) in in_degree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }
    let mut topo_order = Vec::with_capacity(nwa.states().len());
    while let Some(state_id) = queue.pop_front() {
        topo_order.push(state_id);
        let state = &nwa.states()[state_id];
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] -= 1;
            if in_degree[*dst as usize] == 0 {
                queue.push_back(*dst as usize);
            }
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] -= 1;
                if in_degree[*dst as usize] == 0 {
                    queue.push_back(*dst as usize);
                }
            }
        }
    }
    debug_assert_eq!(topo_order.len(), nwa.states().len());

    let dp_started_at = profile.then(Instant::now);
    let mut languages = vec![BTreeMap::<Word, Weight>::new(); nwa.states().len()];
    let mut weight_cache = ScopedWeightOpCache::default();
    let mut dp_word_contributions = 0usize;
    let mut max_state_words = 0usize;
    for state_id in topo_order.into_iter().rev() {
        let state = &nwa.states()[state_id];
        let mut contributions = BTreeMap::<Word, SmallVec<[Weight; 4]>>::new();
        if let Some(final_weight) = &state.final_weight {
            contributions
                .entry(Word::new())
                .or_default()
                .push(final_weight.clone());
        }
        for (dst, edge_weight) in &state.epsilons {
            for (word, suffix_weight) in &languages[*dst as usize] {
                dp_word_contributions += 1;
                let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                if !contribution.is_empty() {
                    contributions
                        .entry(word.clone())
                        .or_default()
                        .push(contribution);
                }
            }
        }
        for (&label, targets) in &state.transitions {
            for (dst, edge_weight) in targets {
                for (suffix, suffix_weight) in &languages[*dst as usize] {
                    if suffix.len() >= 2 {
                        let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                        if !contribution.is_empty() {
                            return Err(GlrMaskError::Compilation(
                                "depth-2 determinization received an accepted path deeper than two labels".into(),
                            ));
                        }
                        continue;
                    }
                    dp_word_contributions += 1;
                    let contribution = weight_cache.intersection(edge_weight, suffix_weight);
                    if contribution.is_empty() {
                        continue;
                    }
                    let mut word = Word::with_capacity(suffix.len() + 1);
                    word.push(label);
                    word.extend_from_slice(suffix);
                    contributions.entry(word).or_default().push(contribution);
                }
            }
        }
        let language = contributions
            .into_iter()
            .filter_map(|(word, weights)| {
                let weight = Weight::union_all(weights.iter());
                (!weight.is_empty()).then_some((word, weight))
            })
            .collect::<BTreeMap<_, _>>();
        max_state_words = max_state_words.max(language.len());
        languages[state_id] = language;
    }
    let dp_ms = dp_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let start_merge_started_at = profile.then(Instant::now);
    let mut accepted = BTreeMap::<Word, Weight>::new();
    for &start_state in nwa.start_states() {
        for (word, weight) in &languages[start_state as usize] {
            union_word_weight(&mut accepted, word.clone(), weight.clone());
        }
    }
    let start_merge_ms = start_merge_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let accepted_word_count = accepted.len();
    let build_started_at = profile.then(Instant::now);
    let mut by_first = BTreeMap::<i32, (Weight, BTreeMap<i32, Weight>)>::new();
    let mut empty_weight = Weight::empty();
    for (word, weight) in accepted {
        match word.as_slice() {
            [] => empty_weight = empty_weight.union(&weight),
            [first] => {
                let entry = by_first
                    .entry(*first)
                    .or_insert_with(|| (Weight::empty(), BTreeMap::new()));
                entry.0 = entry.0.union(&weight);
            }
            [first, second] => {
                let entry = by_first
                    .entry(*first)
                    .or_insert_with(|| (Weight::empty(), BTreeMap::new()));
                entry
                    .1
                    .entry(*second)
                    .and_modify(|existing| *existing = existing.union(&weight))
                    .or_insert(weight);
            }
            _ => unreachable!("depth-2 language contains an overlong word"),
        }
    }

    let mut dwa = DWA::new(0, 0);
    if !empty_weight.is_empty() {
        dwa.set_final_weight(dwa.start_state(), empty_weight);
    }
    let mut shared_leaf = None::<u32>;
    for (first_label, (first_final, second_words)) in by_first {
        let prefix_weight = Weight::union_all(
            std::iter::once(&first_final).chain(second_words.values()),
        );
        let first_state = dwa.add_state();
        dwa.add_transition(dwa.start_state(), first_label, first_state, prefix_weight);
        if !first_final.is_empty() {
            dwa.set_final_weight(first_state, first_final);
        }
        if !second_words.is_empty() {
            let leaf = *shared_leaf.get_or_insert_with(|| {
                let state = dwa.add_state();
                dwa.set_final_weight(state, Weight::all());
                state
            });
            for (second_label, word_weight) in second_words {
                dwa.add_transition(first_state, second_label, leaf, word_weight);
            }
        }
    }
    let build_ms = build_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][determinize_depth2] nwa_states={} dp_word_contributions={} max_state_words={} accepted_words={} intersection_cache_entries={} dp_ms={:.3} start_merge_ms={:.3} build_ms={:.3} total_ms={:.3}",
            nwa.states().len(),
            dp_word_contributions,
            max_state_words,
            accepted_word_count,
            weight_cache.intersection_entry_count(),
            dp_ms,
            start_merge_ms,
            build_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok(dwa)
}

/// Exact determinization by explicit weighted-language dynamic programming for
/// shallow acyclic NWAs.
///
/// Returns `Ok(None)` when the finite language exceeds `max_words`; callers can
/// then use ordinary subset construction. This is a resource guard only and
/// never changes the accepted language.
pub fn determinize_bounded_language(
    nwa: &NWA,
    max_depth: usize,
    max_words: usize,
) -> Result<Option<DWA>, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "bounded-language weighted determinization supports only acyclic NWAs".into(),
        ));
    }
    if max_depth == 0 || max_depth > 8 {
        return Ok(None);
    }

    let profile = determinize_profile_enabled();
    let total_started_at = profile.then(Instant::now);
    type Word = SmallVec<[i32; 4]>;

    fn union_word_weight(language: &mut BTreeMap<Word, Weight>, word: Word, add: Weight) {
        if add.is_empty() {
            return;
        }
        match language.entry(word) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                *occupied.get_mut() = occupied.get().union(&add);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(add);
            }
        }
    }

    let mut indegree = vec![0usize; nwa.states().len()];
    for state in nwa.states() {
        for (target, _) in &state.epsilons {
            indegree[*target as usize] += 1;
        }
        for branches in state.transitions.values() {
            for (target, _) in branches {
                indegree[*target as usize] += 1;
            }
        }
    }
    let mut queue = indegree
        .iter()
        .enumerate()
        .filter_map(|(state, &degree)| (degree == 0).then_some(state))
        .collect::<VecDeque<_>>();
    let mut topo = Vec::with_capacity(nwa.states().len());
    while let Some(state) = queue.pop_front() {
        topo.push(state);
        for (target, _) in &nwa.states()[state].epsilons {
            indegree[*target as usize] -= 1;
            if indegree[*target as usize] == 0 {
                queue.push_back(*target as usize);
            }
        }
        for branches in nwa.states()[state].transitions.values() {
            for (target, _) in branches {
                indegree[*target as usize] -= 1;
                if indegree[*target as usize] == 0 {
                    queue.push_back(*target as usize);
                }
            }
        }
    }
    if topo.len() != nwa.states().len() {
        return Err(GlrMaskError::Compilation(
            "bounded-language weighted determinization requires an acyclic NWA".into(),
        ));
    }

    let dp_started_at = profile.then(Instant::now);
    let mut languages = vec![BTreeMap::<Word, Weight>::new(); nwa.states().len()];
    let mut cache = ScopedWeightOpCache::default();
    let mut contribution_count = 0usize;
    let mut max_state_words = 0usize;
    for state_id in topo.into_iter().rev() {
        let state = &nwa.states()[state_id];
        let mut contributions = BTreeMap::<Word, SmallVec<[Weight; 4]>>::new();
        if let Some(final_weight) = &state.final_weight
            && !final_weight.is_empty()
        {
            contributions
                .entry(Word::new())
                .or_default()
                .push(final_weight.clone());
        }
        for (target, edge_weight) in &state.epsilons {
            for (word, suffix_weight) in &languages[*target as usize] {
                contribution_count += 1;
                let contribution = cache.intersection(edge_weight, suffix_weight);
                if !contribution.is_empty() {
                    contributions
                        .entry(word.clone())
                        .or_default()
                        .push(contribution);
                }
            }
        }
        for (&label, branches) in &state.transitions {
            for (target, edge_weight) in branches {
                for (suffix, suffix_weight) in &languages[*target as usize] {
                    if suffix.len() >= max_depth {
                        let contribution = cache.intersection(edge_weight, suffix_weight);
                        if !contribution.is_empty() {
                            return Err(GlrMaskError::Compilation(format!(
                                "bounded-language determinization received an accepted path deeper than {max_depth} labels"
                            )));
                        }
                        continue;
                    }
                    contribution_count += 1;
                    let contribution = cache.intersection(edge_weight, suffix_weight);
                    if contribution.is_empty() {
                        continue;
                    }
                    let mut word = Word::with_capacity(suffix.len() + 1);
                    word.push(label);
                    word.extend_from_slice(suffix);
                    contributions.entry(word).or_default().push(contribution);
                }
            }
        }
        if contributions.len() > max_words {
            if profile {
                eprintln!(
                    "[glrmask/profile][determinize_bounded_language_abort] state={} contributions={} max_words={}",
                    state_id,
                    contributions.len(),
                    max_words,
                );
            }
            return Ok(None);
        }
        let language = contributions
            .into_iter()
            .filter_map(|(word, weights)| {
                let weight = Weight::union_all(weights.iter());
                (!weight.is_empty()).then_some((word, weight))
            })
            .collect::<BTreeMap<_, _>>();
        max_state_words = max_state_words.max(language.len());
        languages[state_id] = language;
    }
    let dp_ms = dp_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let merge_started_at = profile.then(Instant::now);
    let mut accepted = BTreeMap::<Word, Weight>::new();
    for &start in nwa.start_states() {
        for (word, weight) in &languages[start as usize] {
            union_word_weight(&mut accepted, word.clone(), weight.clone());
        }
    }
    if accepted.len() > max_words {
        return Ok(None);
    }
    let accepted_word_count = accepted.len();
    let merge_ms = merge_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    struct TrieNode {
        final_weight: Weight,
        children: BTreeMap<i32, usize>,
        subtree_weight: Weight,
    }

    impl Default for TrieNode {
        fn default() -> Self {
            Self {
                final_weight: Weight::empty(),
                children: BTreeMap::new(),
                subtree_weight: Weight::empty(),
            }
        }
    }

    let trie_started_at = profile.then(Instant::now);
    let mut trie = vec![TrieNode::default()];
    for (word, weight) in accepted {
        let mut node = 0usize;
        for label in word {
            let next = if let Some(&child) = trie[node].children.get(&label) {
                child
            } else {
                let child = trie.len();
                trie.push(TrieNode::default());
                trie[node].children.insert(label, child);
                child
            };
            node = next;
        }
        trie[node].final_weight = trie[node].final_weight.union(&weight);
    }
    for node in (0..trie.len()).rev() {
        let mut parts = SmallVec::<[Weight; 8]>::new();
        if !trie[node].final_weight.is_empty() {
            parts.push(trie[node].final_weight.clone());
        }
        for &child in trie[node].children.values() {
            if !trie[child].subtree_weight.is_empty() {
                parts.push(trie[child].subtree_weight.clone());
            }
        }
        trie[node].subtree_weight = Weight::union_all(parts.iter());
    }
    let trie_ms = trie_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    #[derive(Clone, Eq, PartialEq, Hash)]
    struct ResidualSignature {
        final_weight: Weight,
        transitions: Vec<(i32, u32, Weight)>,
    }

    let hash_cons_started_at = profile.then(Instant::now);
    let mut signatures = FxHashMap::<ResidualSignature, u32>::default();
    let mut states = Vec::<DWAState>::new();
    let mut trie_to_state = vec![u32::MAX; trie.len()];
    for node in (0..trie.len()).rev() {
        let transitions = trie[node]
            .children
            .iter()
            .map(|(&label, &child)| {
                (
                    label,
                    trie_to_state[child],
                    trie[child].subtree_weight.clone(),
                )
            })
            .collect::<Vec<_>>();
        let signature = ResidualSignature {
            final_weight: trie[node].final_weight.clone(),
            transitions,
        };
        let state = if let Some(&existing) = signatures.get(&signature) {
            existing
        } else {
            let id = states.len() as u32;
            let mut dwa_state = DWAState::default();
            if !signature.final_weight.is_empty() {
                dwa_state.final_weight = Some(signature.final_weight.clone());
            }
            for (label, target, weight) in &signature.transitions {
                dwa_state
                    .transitions
                    .insert(*label, (*target, weight.clone()));
            }
            states.push(dwa_state);
            signatures.insert(signature, id);
            id
        };
        trie_to_state[node] = state;
    }
    let dwa = DWA::from_parts(states, trie_to_state[0]);
    let hash_cons_ms = hash_cons_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][determinize_bounded_language] nwa_states={} max_depth={} contributions={} max_state_words={} accepted_words={} trie_states={} hashcons_states={} intersection_cache_entries={} dp_ms={:.3} merge_ms={:.3} trie_ms={:.3} hash_cons_ms={:.3} total_ms={:.3}",
            nwa.states().len(),
            max_depth,
            contribution_count,
            max_state_words,
            accepted_word_count,
            trie.len(),
            dwa.num_states(),
            cache.intersection_entry_count(),
            dp_ms,
            merge_ms,
            trie_ms,
            hash_cons_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(Some(dwa))
}

/// Aggregate direct epsilon-free point-entry contributions without repeatedly
/// rebuilding whole weight maps. Each contribution carries exactly one TSID;
/// sorting lets us reduce token sets locally for a `(destination, TSID)` pair
/// and construct the canonical per-destination and edge weights in one pass.
fn aggregate_direct_point_entries(
    target_contributions: SmallVec<[(u32, Weight); 1]>,
) -> (Vec<(u32, Weight)>, Weight) {
    let mut points: Vec<(u32, u32, SharedTokenSet)> = target_contributions
        .into_iter()
        .map(|(dst, weight)| {
            let (tsid, tokens) = weight
                .single_tsid_shared_entry()
                .expect("point-entry aggregation requires point weights");
            (dst, tsid, tokens)
        })
        .collect();

    let mut edge_entries: Vec<(u32, SharedTokenSet)> = points
        .iter()
        .map(|(_, tsid, tokens)| (*tsid, std::sync::Arc::clone(tokens)))
        .collect();
    edge_entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    let edge_weight = Weight::union_sorted_point_entries(edge_entries);

    points.sort_unstable_by_key(|(dst, tsid, _)| (*dst, *tsid));
    let mut next_key = Vec::new();
    let mut group_start = 0usize;
    while group_start < points.len() {
        let dst = points[group_start].0;
        let mut group_end = group_start + 1;
        while group_end < points.len() && points[group_end].0 == dst {
            group_end += 1;
        }
        let weight = Weight::union_sorted_point_entries(
            points[group_start..group_end]
                .iter()
                .map(|(_, tsid, tokens)| (*tsid, std::sync::Arc::clone(tokens))),
        );
        debug_assert!(!weight.is_empty());
        next_key.push((dst, weight));
        group_start = group_end;
    }

    (next_key, edge_weight)
}

fn canonicalize_subset_live_domain(
    next_key: Vec<(u32, Weight)>,
    future_domains: Option<&[Weight]>,
    cache: &mut ScopedWeightOpCache,
) -> (Vec<(u32, Weight)>, Option<Weight>) {
    let Some(future_domains) = future_domains else {
        return (next_key, None);
    };

    // For every suffix word z, L_q(z) is a subset of F(q), the union of all
    // accepting suffix weights from q. Therefore
    //
    //   w_q ∩ L_q(z) = (w_q ∩ F(q)) ∩ L_q(z).
    //
    // Replacing each subset entry by this canonical live restriction preserves
    // the complete residual weighted language, entrywise and hence under union.
    let mut canonical = Vec::with_capacity(next_key.len());
    for (state, weight) in next_key {
        let Some(future) = future_domains.get(state as usize) else {
            continue;
        };
        let live = if future.is_full() || weight.storage_ptr_eq(future) || weight.is_subset(future) {
            weight
        } else {
            cache.intersection(&weight, future)
        };
        if !live.is_empty() {
            canonical.push((state, live));
        }
    }
    let live_domain = Weight::union_all(canonical.iter().map(|(_, weight)| weight));
    (canonical, Some(live_domain))
}

fn intern_determinized_subset(
    next_key: Vec<(u32, Weight)>,
    future_domains: Option<&[Weight]>,
    canonicalization_cache: &mut ScopedWeightOpCache,
    state_live_domains: &mut Vec<Weight>,
    subset_map: &mut FxHashMap<Arc<[(u32, Weight)]>, u32>,
    worklist: &mut VecDeque<(u32, Arc<[(u32, Weight)]>)>,
    dwa: &mut DWA,
) -> Option<(u32, Option<Weight>)> {
    let (next_key, live_domain) =
        canonicalize_subset_live_domain(next_key, future_domains, canonicalization_cache);
    if next_key.is_empty() {
        return None;
    }
    if let Some(existing) = subset_map.get(next_key.as_slice()).copied() {
        let live_domain = future_domains.map(|_| {
            state_live_domains
                .get(existing as usize)
                .expect("interned determinized state must retain its live domain")
                .clone()
        });
        return Some((existing, live_domain));
    }

    let new_id = dwa.add_state();
    if let Some(live_domain) = &live_domain {
        debug_assert_eq!(new_id as usize, state_live_domains.len());
        state_live_domains.push(live_domain.clone());
    }
    let shared_key: Arc<[(u32, Weight)]> = next_key.into();
    subset_map.insert(Arc::clone(&shared_key), new_id);
    worklist.push_back((new_id, shared_key));
    Some((new_id, live_domain))
}

/// Intern a one-entry subset through a pointer-identity front cache. The
/// structural `subset_map` remains authoritative on every cache miss.
fn intern_determinized_singleton(
    dst: u32,
    weight: &Weight,
    future_domains: Option<&[Weight]>,
    canonicalization_cache: &mut ScopedWeightOpCache,
    state_live_domains: &mut Vec<Weight>,
    singleton_subsets: &mut FxHashMap<(u32, usize), u32>,
    subset_map: &mut FxHashMap<Arc<[(u32, Weight)]>, u32>,
    worklist: &mut VecDeque<(u32, Arc<[(u32, Weight)]>)>,
    dwa: &mut DWA,
) -> Option<(u32, Option<Weight>, bool)> {
    let (canonical, live_domain) = canonicalize_subset_live_domain(
        vec![(dst, weight.clone())],
        future_domains,
        canonicalization_cache,
    );
    let (_, canonical_weight) = canonical.first()?;
    let singleton_key = (dst, canonical_weight.ptr_key());
    if let Some(existing) = singleton_subsets.get(&singleton_key).copied() {
        let live_domain = future_domains.map(|_| {
            state_live_domains
                .get(existing as usize)
                .expect("interned singleton state must retain its live domain")
                .clone()
        });
        return Some((existing, live_domain, true));
    }

    if let Some(existing) = subset_map.get(canonical.as_slice()).copied() {
        singleton_subsets.insert(singleton_key, existing);
        let existing_live = future_domains.map(|_| {
            state_live_domains
                .get(existing as usize)
                .expect("interned singleton state must retain its live domain")
                .clone()
        });
        return Some((existing, existing_live, true));
    }

    let state = dwa.add_state();
    if let Some(domain) = &live_domain {
        debug_assert_eq!(state as usize, state_live_domains.len());
        state_live_domains.push(domain.clone());
    }
    let shared_key: Arc<[(u32, Weight)]> = canonical.into();
    subset_map.insert(Arc::clone(&shared_key), state);
    worklist.push_back((state, shared_key));
    singleton_subsets.insert(singleton_key, state);
    Some((state, live_domain, false))
}

#[inline]
fn restrict_edge_to_target_live_domain(
    edge_weight: Weight,
    target_live_domain: Option<&Weight>,
    cache: &mut ScopedWeightOpCache,
) -> Weight {
    let Some(target_live_domain) = target_live_domain else {
        return edge_weight;
    };
    if target_live_domain.is_empty() {
        return Weight::empty();
    }
    if target_live_domain.is_full()
        || edge_weight.storage_ptr_eq(target_live_domain)
        || edge_weight.is_subset(target_live_domain)
    {
        edge_weight
    } else {
        cache.intersection(&edge_weight, target_live_domain)
    }
}

fn determinize_profile_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_DETERMINIZE")
        .map(|value| value == "1")
        .unwrap_or(false)
}

struct DeterminizeImplOutput {
    dwa: DWA,
    subsets: Option<Vec<Arc<[(u32, Weight)]>>>,
    live_domains: Option<Vec<Weight>>,
}

pub struct DeterminizedWithLiveDomains {
    pub dwa: DWA,
    pub live_domains: Vec<Weight>,
    /// Every transition weight has already been restricted to the live domain
    /// of its target state.
    pub productive_edges: bool,
}

pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {
    let profile = determinize_profile_enabled();
    let dwa = determinize_impl_with_options(nwa, true, true, profile, false, None)?.dwa;

    if std::env::var_os("GLRMASK_ASSERT_GROUPED_DETERMINIZE_EQUIVALENCE").is_some() {
        let reference =
            determinize_impl_with_options(nwa, true, false, false, false, None)?.dwa;
        match find_difference(&dwa, &reference)? {
            Some(word) => {
                return Err(GlrMaskError::Compilation(format!(
                    "grouped-weight determinization differs from the ordinary path on labels {word:?}"
                )));
            }
            None if profile => eprintln!(
                "[glrmask/profile][determinize_grouped_weight_equivalence] result=equivalent"
            ),
            None => {}
        }
    }

    Ok(dwa)
}

/// Determinize an acyclic weighted NWA and derive each DWA state's intrinsic
/// live token domain directly from the NWA subset representation.
///
/// For a determinized subset `S = {(q, w_q)}`, the exact future-acceptance
/// domain is `union_q (w_q intersection F(q))`, where `F(q)` is the weighted
/// future language domain of NWA state `q`. Computing `F` once on the small NWA
/// avoids a second backward fixed-point over the usually much larger DWA.
pub fn determinize_with_live_domains(
    nwa: &NWA,
) -> Result<DeterminizedWithLiveDomains, GlrMaskError> {
    let profile = determinize_profile_enabled();
    let future = nwa_future_domains(nwa)?;
    let output = determinize_impl_with_options(nwa, true, true, profile, false, Some(&future))?;
    let live_domains = output
        .live_domains
        .expect("live-canonical determinization must retain state live domains");

    if profile {
        eprintln!(
            "[glrmask/profile][determinize_live_domains] nwa_states={} dwa_states={} future_ms_included=false canonicalized_during_subset_intern=true",
            nwa.states().len(),
            output.dwa.states().len(),
        );
    }
    Ok(DeterminizedWithLiveDomains {
        dwa: output.dwa,
        live_domains,
        productive_edges: true,
    })
}

fn nwa_future_domains(nwa: &NWA) -> Result<Vec<Weight>, GlrMaskError> {
    let state_count = nwa.states().len();
    let mut indegree = vec![0usize; state_count];
    for state in nwa.states() {
        for (target, _) in &state.epsilons {
            if (*target as usize) >= state_count {
                return Err(GlrMaskError::Compilation(
                    "weighted NWA contains an out-of-range epsilon target".into(),
                ));
            }
            indegree[*target as usize] += 1;
        }
        for branches in state.transitions.values() {
            for (target, _) in branches {
                if (*target as usize) >= state_count {
                    return Err(GlrMaskError::Compilation(
                        "weighted NWA contains an out-of-range transition target".into(),
                    ));
                }
                indegree[*target as usize] += 1;
            }
        }
    }
    let mut queue = indegree
        .iter()
        .enumerate()
        .filter_map(|(state, &degree)| (degree == 0).then_some(state))
        .collect::<Vec<_>>();
    let mut head = 0usize;
    let mut topo = Vec::with_capacity(state_count);
    while head < queue.len() {
        let state = queue[head];
        head += 1;
        topo.push(state);
        for (target, _) in &nwa.states()[state].epsilons {
            indegree[*target as usize] -= 1;
            if indegree[*target as usize] == 0 {
                queue.push(*target as usize);
            }
        }
        for branches in nwa.states()[state].transitions.values() {
            for (target, _) in branches {
                indegree[*target as usize] -= 1;
                if indegree[*target as usize] == 0 {
                    queue.push(*target as usize);
                }
            }
        }
    }
    if topo.len() != state_count {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }

    let started_at = determinize_profile_enabled().then(Instant::now);
    let mut future = vec![Weight::empty(); state_count];
    let mut cache = ScopedWeightOpCache::default();
    for state_id in topo.into_iter().rev() {
        let state = &nwa.states()[state_id];
        let mut parts = SmallVec::<[Weight; 8]>::new();
        if let Some(final_weight) = &state.final_weight {
            if !final_weight.is_empty() {
                parts.push(final_weight.clone());
            }
        }
        for (target, edge_weight) in &state.epsilons {
            let contribution = cache.intersection(edge_weight, &future[*target as usize]);
            if !contribution.is_empty() {
                parts.push(contribution);
            }
        }
        for branches in state.transitions.values() {
            for (target, edge_weight) in branches {
                let contribution = cache.intersection(edge_weight, &future[*target as usize]);
                if !contribution.is_empty() {
                    parts.push(contribution);
                }
            }
        }
        future[state_id] = Weight::union_all(parts.iter());
    }
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][nwa_future_domains] states={} transitions={} total_ms={:.3}",
            state_count,
            nwa.num_transitions(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(future)
}

fn determinize_impl(
    nwa: &NWA,
    direct_single_target_enabled: bool,
) -> Result<DWA, GlrMaskError> {
    Ok(determinize_impl_with_options(
        nwa,
        direct_single_target_enabled,
        true,
        determinize_profile_enabled(),
        false,
        None,
    )?.dwa)
}

fn determinize_impl_with_options(
    nwa: &NWA,
    direct_single_target_enabled: bool,
    group_transition_weights: bool,
    profile: bool,
    retain_subsets: bool,
    future_domains: Option<&[Weight]>,
) -> Result<DeterminizeImplOutput, GlrMaskError> {
    if !nwa.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "weighted determinization currently supports only acyclic NWAs".into(),
        ));
    }

    let weight_group_build_started_at = profile.then(Instant::now);
    let weight_grouped_transitions = group_transition_weights.then(|| build_weight_grouped_transitions(nwa));
    let weight_group_build_ms = weight_group_build_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    fn canonicalize(subset: &FxHashMap<u32, Weight>) -> Vec<(u32, Weight)> {
        let mut entries: Vec<_> = subset
            .iter()
            .filter_map(|(&state_id, weight)| (!weight.is_empty()).then_some((state_id, weight.clone())))
            .collect();
        entries.sort_by_key(|(state_id, _)| *state_id);
        entries
    }

    fn epsilon_closure(nwa: &NWA, seed: FxHashMap<u32, Weight>) -> FxHashMap<u32, Weight> {
        // Fast path: single-state seed with no epsilon transitions (99.6% of calls)
        if seed.len() == 1 {
            let (&state_id, _) = seed.iter().next().unwrap();
            if let Some(state) = nwa.states().get(state_id as usize) {
                if state.epsilons.is_empty() {
                    return seed;
                }
            }
        }

        let mut closure = seed;
        let mut queue: VecDeque<u32> = closure.keys().copied().collect();

        while let Some(state_id) = queue.pop_front() {
            let Some(current_weight) = closure.get(&state_id).cloned() else {
                continue;
            };
            let Some(state) = nwa.states().get(state_id as usize) else {
                continue;
            };
            for (dst, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let existing = closure.get(dst).cloned().unwrap_or_else(Weight::empty);
                if !contribution.is_subset(&existing) {
                    closure.insert(*dst, existing.union(&contribution));
                    queue.push_back(*dst);
                }
            }
        }

        closure
    }

    let mut dwa = DWA::new(0, 0);
    let start_id = dwa.start_state();

    let start_subset = epsilon_closure(nwa, seed_start_subset(nwa));

    if start_subset.is_empty() {
        return Ok(DeterminizeImplOutput {
            dwa,
            subsets: retain_subsets.then(Vec::new),
            live_domains: future_domains.map(|_| vec![Weight::empty()]),
        });
    }

    let mut canonicalization_cache = ScopedWeightOpCache::default();
    let mut subset_map: FxHashMap<Arc<[(u32, Weight)]>, u32> = FxHashMap::default();
    // This is an identity-only front cache. A miss always falls back through
    // `subset_map`, which retains structural equality as the source of truth.
    let mut singleton_subsets: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Arc<[(u32, Weight)]>)> = VecDeque::new();
    let (start_entries, start_live_domain) = canonicalize_subset_live_domain(
        canonicalize(&start_subset),
        future_domains,
        &mut canonicalization_cache,
    );
    if start_entries.is_empty() {
        return Ok(DeterminizeImplOutput {
            dwa,
            subsets: retain_subsets.then(Vec::new),
            live_domains: future_domains.map(|_| vec![Weight::empty()]),
        });
    }
    let start_entries: Arc<[(u32, Weight)]> = start_entries.into();
    subset_map.insert(Arc::clone(&start_entries), start_id);
    worklist.push_back((start_id, start_entries));
    let mut state_live_domains = start_live_domain.into_iter().collect::<Vec<_>>();

    // Almost every label has one surviving destination. Keep that common
    // case inline instead of allocating a nested hash map and a Vec per label.
    let mut raw_targets: FxHashMap<i32, SmallVec<[(u32, Weight); 1]>> = FxHashMap::default();
    // All operands in the expansion loop remain owned by the NWA or subset_map
    // for the lifetime of this determinization, so a local exact cache avoids
    // thread-local memo overhead on the heavily reused intersection pairs.
    let mut scoped_determinize_weight_cache = ScopedWeightOpCache::default();
    let mut profile_subset_entries = 0usize;
    let mut profile_max_subset_entries = 0usize;
    let mut profile_raw_transition_visits = 0usize;
    let mut profile_weight_group_visits = 0usize;
    let mut profile_labels = 0usize;
    let mut profile_target_contributions = 0usize;
    let mut profile_single_contribution_labels = 0usize;
    let mut profile_single_contribution_no_epsilon_labels = 0usize;
    let mut profile_direct_single_target_labels = 0usize;
    let mut profile_direct_singleton_cache_hits = 0usize;
    let mut profile_direct_singleton_cache_misses = 0usize;
    let mut profile_direct_singleton_fast_path_groups = 0usize;
    let mut profile_direct_singleton_fast_path_labels = 0usize;
    let mut profile_multi_contribution_single_target_no_epsilon_labels = 0usize;
    let mut profile_multi_contribution_single_target_no_epsilon_contributions = 0usize;
    let mut profile_direct_multi_target_labels = 0usize;
    let mut profile_multi_contribution_all_no_epsilon_labels = 0usize;
    let mut profile_multi_contribution_all_no_epsilon_contributions = 0usize;
    let mut profile_expand_ms = 0.0;
    let mut profile_combine_ms = 0.0;
    let mut profile_edge_union_ms = 0.0;
    let mut profile_closure_ms = 0.0;
    let mut profile_normalize_ms = 0.0;
    let mut profile_canonicalize_ms = 0.0;
    let mut profile_subset_lookup_ms = 0.0;

    while let Some((from_state, subset_entries)) = worklist.pop_front() {
        if profile {
            profile_subset_entries += subset_entries.len();
            profile_max_subset_entries = profile_max_subset_entries.max(subset_entries.len());
        }
        // Final weight computation is deferred to after the main loop
        // and parallelized across all states.
        let expand_started_at = profile.then(Instant::now);

        if let Some(weight_grouped_transitions) = &weight_grouped_transitions {
            if direct_single_target_enabled && subset_entries.len() == 1 {
                let (nwa_state_id, path_weight) = &subset_entries[0];
                for group in &weight_grouped_transitions[*nwa_state_id as usize] {
                    if profile {
                        profile_weight_group_visits += 1;
                        profile_raw_transition_visits += group.edges.len();
                    }
                    let next_weight = scoped_determinize_weight_cache
                        .intersection(path_weight, &group.weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    let mut emitted_direct_singleton = false;
                    for edge in &group.edges {
                        if edge.direct_singleton {
                            emitted_direct_singleton = true;
                            if profile {
                                profile_labels += 1;
                                profile_target_contributions += 1;
                                profile_single_contribution_labels += 1;
                                profile_single_contribution_no_epsilon_labels += 1;
                                profile_direct_single_target_labels += 1;
                                profile_direct_singleton_fast_path_labels += 1;
                            }
                            let subset_lookup_started_at = profile.then(Instant::now);
                            let Some((to_state, target_live_domain, cache_hit)) =
                                intern_determinized_singleton(
                                edge.dst,
                                &next_weight,
                                future_domains,
                                &mut canonicalization_cache,
                                &mut state_live_domains,
                                &mut singleton_subsets,
                                &mut subset_map,
                                &mut worklist,
                                &mut dwa,
                            ) else {
                                continue;
                            };
                            if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                                profile_subset_lookup_ms +=
                                    subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                            }
                            if profile {
                                if cache_hit {
                                    profile_direct_singleton_cache_hits += 1;
                                } else {
                                    profile_direct_singleton_cache_misses += 1;
                                }
                            }
                            let productive_edge = restrict_edge_to_target_live_domain(
                                next_weight.clone(),
                                target_live_domain.as_ref(),
                                &mut scoped_determinize_weight_cache,
                            );
                            if !productive_edge.is_empty() {
                                dwa.add_transition(
                                    from_state,
                                    edge.label,
                                    to_state,
                                    productive_edge,
                                );
                            }
                        } else {
                            raw_targets
                                .entry(edge.label)
                                .or_default()
                                .push((edge.dst, next_weight.clone()));
                        }
                    }
                    if emitted_direct_singleton && profile {
                        profile_direct_singleton_fast_path_groups += 1;
                    }
                }
            } else {
                for (nwa_state_id, path_weight) in subset_entries.iter() {
                    for group in &weight_grouped_transitions[*nwa_state_id as usize] {
                        if profile {
                            profile_weight_group_visits += 1;
                            profile_raw_transition_visits += group.edges.len();
                        }
                        let next_weight = scoped_determinize_weight_cache
                            .intersection(path_weight, &group.weight);
                        if next_weight.is_empty() {
                            continue;
                        }
                        for edge in &group.edges {
                            raw_targets
                                .entry(edge.label)
                                .or_default()
                                .push((edge.dst, next_weight.clone()));
                        }
                    }
                }
            }
        } else {
            for (nwa_state_id, path_weight) in subset_entries.iter() {
                let state = &nwa.states()[*nwa_state_id as usize];
                for (&label, targets) in &state.transitions {
                    for (dst, trans_weight) in targets {
                        if profile {
                            profile_raw_transition_visits += 1;
                        }
                        let next_weight = scoped_determinize_weight_cache
                            .intersection(path_weight, trans_weight);
                        if next_weight.is_empty() {
                            continue;
                        }
                        raw_targets.entry(label).or_default().push((*dst, next_weight));
                    }
                }
            }
        }

        if let Some(expand_started_at) = expand_started_at {
            profile_expand_ms += expand_started_at.elapsed().as_secs_f64() * 1000.0;
        }

        for (label, target_contributions) in raw_targets.drain() {
            if profile {
                profile_labels += 1;
                profile_target_contributions += target_contributions.len();
                if target_contributions.len() == 1 {
                    profile_single_contribution_labels += 1;
                    let (dst, _) = &target_contributions[0];
                    if nwa
                        .states()
                        .get(*dst as usize)
                        .is_some_and(|state| state.epsilons.is_empty())
                    {
                        profile_single_contribution_no_epsilon_labels += 1;
                    }
                } else {
                    let dst = target_contributions[0].0;
                    let all_no_epsilon = target_contributions.iter().all(|(candidate, _)| {
                        nwa
                            .states()
                            .get(*candidate as usize)
                            .is_some_and(|state| state.epsilons.is_empty())
                    });
                    if all_no_epsilon {
                        profile_multi_contribution_all_no_epsilon_labels += 1;
                        profile_multi_contribution_all_no_epsilon_contributions +=
                            target_contributions.len();
                    }
                    if target_contributions.iter().all(|(candidate, _)| *candidate == dst)
                        && all_no_epsilon
                    {
                        profile_multi_contribution_single_target_no_epsilon_labels += 1;
                        profile_multi_contribution_single_target_no_epsilon_contributions +=
                            target_contributions.len();
                    }
                }
            }
            if target_contributions.is_empty() {
                continue;
            }

            let direct_no_epsilon_targets = direct_single_target_enabled
                && target_contributions.iter().all(|(dst, _)| {
                    nwa
                        .states()
                        .get(*dst as usize)
                        .is_some_and(|state| state.epsilons.is_empty())
                });
            if direct_no_epsilon_targets {
                if target_contributions.len() == 1 {
                    if profile {
                        profile_direct_single_target_labels += 1;
                    }
                    let (dst, edge_weight) = target_contributions.into_iter().next().unwrap();
                    let subset_lookup_started_at = profile.then(Instant::now);
                    let Some((to_state, target_live_domain, cache_hit)) =
                        intern_determinized_singleton(
                        dst,
                        &edge_weight,
                        future_domains,
                        &mut canonicalization_cache,
                        &mut state_live_domains,
                        &mut singleton_subsets,
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    ) else {
                        continue;
                    };
                    if profile {
                        if cache_hit {
                            profile_direct_singleton_cache_hits += 1;
                        } else {
                            profile_direct_singleton_cache_misses += 1;
                        }
                    }
                    if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                        profile_subset_lookup_ms +=
                            subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    let productive_edge = restrict_edge_to_target_live_domain(
                        edge_weight,
                        target_live_domain.as_ref(),
                        &mut scoped_determinize_weight_cache,
                    );
                    if !productive_edge.is_empty() {
                        dwa.add_transition(from_state, label, to_state, productive_edge);
                    }
                    continue;
                }

                let direct_point_entries = target_contributions
                    .iter()
                    .all(|(_, weight)| weight.single_tsid_shared_entry().is_some());
                if direct_point_entries {
                    let (next_key, edge_weight) = aggregate_direct_point_entries(target_contributions);
                    debug_assert!(!edge_weight.is_empty());
                    let subset_lookup_started_at = profile.then(Instant::now);
                    let Some((to_state, target_live_domain)) = intern_determinized_subset(
                        next_key,
                        future_domains,
                        &mut canonicalization_cache,
                        &mut state_live_domains,
                        &mut subset_map,
                        &mut worklist,
                        &mut dwa,
                    ) else {
                        continue;
                    };
                    if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                        profile_subset_lookup_ms +=
                            subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    let productive_edge = restrict_edge_to_target_live_domain(
                        edge_weight,
                        target_live_domain.as_ref(),
                        &mut scoped_determinize_weight_cache,
                    );
                    if !productive_edge.is_empty() {
                        dwa.add_transition(from_state, label, to_state, productive_edge);
                    }
                    continue;
                }

                if profile {
                    profile_direct_multi_target_labels += 1;
                }
                let mut sorted_targets = target_contributions;
                sorted_targets.sort_unstable_by_key(|(dst, _)| *dst);
                let mut next_key: Vec<(u32, Weight)> =
                    Vec::with_capacity(sorted_targets.len());
                for (dst, weight) in sorted_targets {
                    if let Some((last_dst, last_weight)) = next_key.last_mut() {
                        if *last_dst == dst {
                            *last_weight = last_weight.union(&weight);
                            continue;
                        }
                    }
                    next_key.push((dst, weight));
                }
                let edge_weight = Weight::union_all(next_key.iter().map(|(_, weight)| weight));
                debug_assert!(!edge_weight.is_empty());

                let subset_lookup_started_at = profile.then(Instant::now);
                let Some((to_state, target_live_domain)) = intern_determinized_subset(
                    next_key,
                    future_domains,
                    &mut canonicalization_cache,
                    &mut state_live_domains,
                    &mut subset_map,
                    &mut worklist,
                    &mut dwa,
                ) else {
                    continue;
                };
                if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                    profile_subset_lookup_ms +=
                        subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
                }
                let productive_edge = restrict_edge_to_target_live_domain(
                    edge_weight,
                    target_live_domain.as_ref(),
                    &mut scoped_determinize_weight_cache,
                );
                if !productive_edge.is_empty() {
                    dwa.add_transition(from_state, label, to_state, productive_edge);
                }
                continue;
            }

            let combine_started_at = profile.then(Instant::now);
            let mut target_subset: FxHashMap<u32, Weight> = FxHashMap::default();
            if target_contributions.len() == 1 {
                let (dst, weight) = target_contributions.into_iter().next().unwrap();
                target_subset.insert(dst, weight);
            } else {
                // Keep the first contribution per destination in the compact
                // target map. On the first repeat, move that destination's
                // operands into a side bucket and union them once at the end.
                // This preserves the exact subset construction while avoiding
                // quadratic growth from repeatedly rebuilding wide weights.
                let mut duplicate_weights: Option<FxHashMap<u32, SmallVec<[Weight; 2]>>> = None;
                for (dst, weight) in target_contributions {
                    if let Some(duplicates) = duplicate_weights.as_mut() {
                        if let Some(weights) = duplicates.get_mut(&dst) {
                            weights.push(weight);
                            continue;
                        }
                    }

                    match target_subset.entry(dst) {
                        HashMapEntry::Vacant(vacant) => {
                            vacant.insert(weight);
                        }
                        HashMapEntry::Occupied(occupied) => {
                            let previous = occupied.remove();
                            let mut weights = SmallVec::<[Weight; 2]>::new();
                            weights.push(previous);
                            weights.push(weight);
                            let duplicates = duplicate_weights.get_or_insert_with(FxHashMap::default);
                            let replaced = duplicates.insert(dst, weights);
                            debug_assert!(replaced.is_none());
                        }
                    }
                }
                if let Some(duplicates) = duplicate_weights {
                    for (dst, weights) in duplicates {
                        target_subset.insert(dst, Weight::union_all(weights.iter()));
                    }
                }
            }

            if let Some(combine_started_at) = combine_started_at {
                profile_combine_ms += combine_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if target_subset.is_empty() {
                continue;
            }

            let edge_union_started_at = profile.then(Instant::now);
            let edge_weight = Weight::union_all(target_subset.values());
            if let Some(edge_union_started_at) = edge_union_started_at {
                profile_edge_union_ms += edge_union_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if edge_weight.is_empty() {
                continue;
            }

            let closure_started_at = profile.then(Instant::now);
            let expanded = epsilon_closure(nwa, target_subset);
            if let Some(closure_started_at) = closure_started_at {
                profile_closure_ms += closure_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if expanded.is_empty() {
                continue;
            }

            let normalize_started_at = profile.then(Instant::now);
            let edge_complement = edge_weight.complement();
            let normalized: FxHashMap<u32, Weight> = if edge_complement.is_empty() {
                expanded
            } else {
                expanded
                    .into_iter()
                    .filter_map(|(state_id, weight)| {
                        let normalized_weight = weight.union(&edge_complement);
                        (!normalized_weight.is_empty()).then_some((state_id, normalized_weight))
                    })
                    .collect()
            };
            if let Some(normalize_started_at) = normalize_started_at {
                profile_normalize_ms += normalize_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let canonicalize_started_at = profile.then(Instant::now);
            let next_key = canonicalize(&normalized);
            if let Some(canonicalize_started_at) = canonicalize_started_at {
                profile_canonicalize_ms += canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if next_key.is_empty() {
                continue;
            }
            let subset_lookup_started_at = profile.then(Instant::now);
            let Some((to_state, target_live_domain)) = intern_determinized_subset(
                next_key,
                future_domains,
                &mut canonicalization_cache,
                &mut state_live_domains,
                &mut subset_map,
                &mut worklist,
                &mut dwa,
            ) else {
                continue;
            };
            if let Some(subset_lookup_started_at) = subset_lookup_started_at {
                profile_subset_lookup_ms += subset_lookup_started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let productive_edge = restrict_edge_to_target_live_domain(
                edge_weight,
                target_live_domain.as_ref(),
                &mut scoped_determinize_weight_cache,
            );
            if !productive_edge.is_empty() {
                dwa.add_transition(from_state, label, to_state, productive_edge);
            }
        }
    }

    if profile {
        eprintln!(
            "[glrmask/profile][determinize_scoped_weight_cache] intersections={}",
            scoped_determinize_weight_cache.intersection_entry_count(),
        );
    }

    // Compute final weights in parallel after the main loop.
    // The subset_map already stores all (entries, state_id) pairs.
    // Sparse path weights recur against a few wide final weights. Index those
    // wide maps once so each sparse range can seek directly to its overlaps.
    let final_weight_indices: FxHashMap<usize, WeightIntersectionIndex> = nwa
        .states()
        .iter()
        .filter_map(|state| state.final_weight.as_ref())
        .filter(|weight| weight.outer_range_count() >= MIN_INDEXED_FINAL_WEIGHT_RANGES)
        .map(|weight| (weight.ptr_key(), weight.intersection_index()))
        .collect();
    let final_weights_started_at = profile.then(Instant::now);
    let mut final_weight_profile = FinalWeightProfile::default();
    let final_weights: Vec<(u32, Weight)> = if profile {
        let profiled: Vec<(u32, Weight, FinalWeightProfile)> = subset_map
            .par_iter()
            .map(|(entries, &state_id)| {
                let (fw, stats) = subset_final_weight_profiled(
                    nwa,
                    entries,
                    &final_weight_indices,
                    MAX_INDEXED_FINAL_PATH_RANGES,
                );
                (state_id, fw, stats)
            })
            .collect();
        profiled
            .into_iter()
            .filter_map(|(state_id, fw, stats)| {
                final_weight_profile.merge(stats);
                (!fw.is_empty()).then_some((state_id, fw))
            })
            .collect()
    } else {
        subset_map
            .par_iter()
            .filter_map(|(entries, &state_id)| {
                let fw = subset_final_weight(
                    nwa,
                    entries,
                    &final_weight_indices,
                    MAX_INDEXED_FINAL_PATH_RANGES,
                );
                (!fw.is_empty()).then_some((state_id, fw))
            })
            .collect()
    };
    for (state_id, fw) in final_weights {
        dwa.set_final_weight(state_id, fw);
    }
    let final_weights_ms = final_weights_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if profile {
        let mut final_group_entries = 0usize;
        let mut final_group_count = 0usize;
        let mut final_subsets_with_reused_group = 0usize;
        let mut max_final_groups_per_subset = 0usize;
        let mut max_final_entries_per_group = 0usize;
        for entries in subset_map.keys() {
            let mut groups = FxHashMap::<usize, usize>::default();
            for (state_id, _) in entries.iter() {
                let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
                    continue;
                };
                *groups.entry(state_final.ptr_key()).or_default() += 1;
            }
            let final_entry_count: usize = groups.values().sum();
            if final_entry_count > 0 {
                final_group_entries += final_entry_count;
                final_group_count += groups.len();
                max_final_groups_per_subset = max_final_groups_per_subset.max(groups.len());
                let max_group = groups.values().copied().max().unwrap_or(0);
                max_final_entries_per_group = max_final_entries_per_group.max(max_group);
                final_subsets_with_reused_group += usize::from(max_group > 1);
            }
        }
        eprintln!(
            "[glrmask/profile][determinize_final_groups] entries={} groups={} saved_intersections={} subsets_with_reused_group={} max_groups_per_subset={} max_entries_per_group={}",
            final_group_entries,
            final_group_count,
            final_group_entries.saturating_sub(final_group_count),
            final_subsets_with_reused_group,
            max_final_groups_per_subset,
            max_final_entries_per_group,
        );

        let max_weight_dim = dwa.states().iter()
            .filter_map(|s| s.final_weight.as_ref())
            .map(Weight::num_ranges)
            .max()
            .unwrap_or(0);
        let mut final_pairs = FxHashSet::default();
        let mut final_path_weights = FxHashSet::default();
        let mut final_state_weights = FxHashSet::default();
        let mut final_path_outer_ranges = 0usize;
        let mut final_state_outer_ranges = 0usize;
        let mut max_final_path_outer_ranges = 0usize;
        let mut max_final_state_outer_ranges = 0usize;
        for entries in subset_map.keys() {
            for (state_id, path_weight) in entries.iter() {
                let Some(state_final) = nwa.states()[*state_id as usize].final_weight.as_ref() else {
                    continue;
                };
                let path_key = path_weight.ptr_key();
                let state_key = state_final.ptr_key();
                final_pairs.insert((path_key, state_key));
                final_path_weights.insert(path_key);
                final_state_weights.insert(state_key);
                let path_ranges = path_weight.outer_range_count();
                let state_ranges = state_final.outer_range_count();
                final_path_outer_ranges += path_ranges;
                final_state_outer_ranges += state_ranges;
                max_final_path_outer_ranges = max_final_path_outer_ranges.max(path_ranges);
                max_final_state_outer_ranges = max_final_state_outer_ranges.max(state_ranges);
            }
        }
        eprintln!(
            "[glrmask/profile][determinize_final_shape] pairs={} unique_pairs={} unique_path_weights={} unique_state_weights={} path_outer_ranges={} state_outer_ranges={} max_path_outer_ranges={} max_state_outer_ranges={}",
            final_weight_profile.final_entries,
            final_pairs.len(),
            final_path_weights.len(),
            final_state_weights.len(),
            final_path_outer_ranges,
            final_state_outer_ranges,
            max_final_path_outer_ranges,
            max_final_state_outer_ranges,
        );

        eprintln!(
            "[glrmask/profile][determinize] nwa_states={} dwa_states={} subset_map_entries={} max_weight_dim={} subset_entries={} max_subset_entries={} raw_transition_visits={} weight_grouped_transitions={} weight_group_build_ms={:.3} weight_group_visits={} labels={} target_contributions={} single_contribution_labels={} single_contribution_no_epsilon_labels={} direct_single_target_labels={} direct_singleton_cache_hits={} direct_singleton_cache_misses={} direct_singleton_fast_path_groups={} direct_singleton_fast_path_labels={} multi_contribution_single_target_no_epsilon_labels={} multi_contribution_single_target_no_epsilon_contributions={} direct_multi_target_labels={} multi_contribution_all_no_epsilon_labels={} multi_contribution_all_no_epsilon_contributions={} expand_ms={:.3} combine_ms={:.3} edge_union_ms={:.3} closure_ms={:.3} normalize_ms={:.3} canonicalize_ms={:.3} subset_lookup_ms={:.3} final_weights_ms={:.3} final_subsets={} final_subset_entries={} final_entries={} final_nonempty_contributions={} final_max_entries={} final_intersection_ms={:.3} final_union_ms={:.3}",
            nwa.states().len(),
            dwa.states().len(),
            subset_map.len(),
            max_weight_dim,
            profile_subset_entries,
            profile_max_subset_entries,
            profile_raw_transition_visits,
            group_transition_weights,
            weight_group_build_ms,
            profile_weight_group_visits,
            profile_labels,
            profile_target_contributions,
            profile_single_contribution_labels,
            profile_single_contribution_no_epsilon_labels,
            profile_direct_single_target_labels,
            profile_direct_singleton_cache_hits,
            profile_direct_singleton_cache_misses,
            profile_direct_singleton_fast_path_groups,
            profile_direct_singleton_fast_path_labels,
            profile_multi_contribution_single_target_no_epsilon_labels,
            profile_multi_contribution_single_target_no_epsilon_contributions,
            profile_direct_multi_target_labels,
            profile_multi_contribution_all_no_epsilon_labels,
            profile_multi_contribution_all_no_epsilon_contributions,
            profile_expand_ms,
            profile_combine_ms,
            profile_edge_union_ms,
            profile_closure_ms,
            profile_normalize_ms,
            profile_canonicalize_ms,
            profile_subset_lookup_ms,
            final_weights_ms,
            final_weight_profile.subsets,
            final_weight_profile.subset_entries,
            final_weight_profile.final_entries,
            final_weight_profile.nonempty_contributions,
            final_weight_profile.max_final_entries,
            final_weight_profile.intersection_ms,
            final_weight_profile.union_ms,
        );
    }

    let subsets = if retain_subsets {
        let empty: Arc<[(u32, Weight)]> = Arc::from([]);
        let mut by_state = vec![empty; dwa.states().len()];
        for (subset, &state) in &subset_map {
            by_state[state as usize] = Arc::clone(subset);
        }
        Some(by_state)
    } else {
        None
    };
    Ok(DeterminizeImplOutput {
        dwa,
        subsets,
        live_domains: future_domains.map(|_| state_live_domains),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weighted_u32::equivalence::find_difference;
    use range_set_blaze::RangeSetBlaze;

    fn tokens(values: impl IntoIterator<Item = u32>) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(values.into_iter().map(|value| value..=value)),
        )))
    }

    #[test]
    fn determinize_unions_duplicate_label_target_contributions() {
        let mut nwa = NWA::new(1, 2);
        let start = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, accept, tokens([0]));
        nwa.add_transition(start, 7, accept, tokens([1]));
        nwa.set_final_weight(accept, tokens([0, 1]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();
        let grouped = determinize_impl_with_options(&nwa, true, true, false, false, None).unwrap().dwa;
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[7]), tokens([0, 1]));
    }

    #[test]
    fn depth2_language_dp_matches_generic_with_epsilon_paths() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        let middle = nwa.add_state();
        let accept = nwa.add_state();
        nwa.set_start_states(vec![start]);

        nwa.add_epsilon(start, left, tokens([0, 1, 2]));
        nwa.add_epsilon(start, right, tokens([1, 2, 3]));
        nwa.add_transition(left, 7, middle, tokens([0, 1]));
        nwa.add_transition(right, 7, middle, tokens([2, 3]));
        nwa.set_final_weight(middle, tokens([0, 2]));
        nwa.add_transition(middle, 8, accept, tokens([1, 2, 3]));
        nwa.set_final_weight(accept, tokens([1, 2]));

        let depth2 = determinize_depth2(&nwa).unwrap();
        let generic = determinize(&nwa).unwrap();
        assert_eq!(find_difference(&depth2, &generic).unwrap(), None);
        assert_eq!(depth2.eval_word(&[7]), tokens([0, 2]));
        assert_eq!(depth2.eval_word(&[7, 8]), tokens([1, 2]));
    }

    #[test]
    fn depth2_language_dp_rejects_accepted_three_label_path() {
        let mut nwa = NWA::new(1, 1);
        let states: Vec<u32> = (0..4).map(|_| nwa.add_state()).collect();
        nwa.set_start_states(vec![states[0]]);
        nwa.add_transition(states[0], 1, states[1], tokens([0]));
        nwa.add_transition(states[1], 2, states[2], tokens([0]));
        nwa.add_transition(states[2], 3, states[3], tokens([0]));
        nwa.set_final_weight(states[3], tokens([0]));

        let error = determinize_depth2(&nwa).unwrap_err();
        assert!(error.to_string().contains("deeper than two labels"), "{error}");
    }

    #[test]
    fn determinize_batches_many_repeated_destinations_exactly() {
        let mut nwa = NWA::new(1, 7);
        let start = nwa.add_state();
        let left = nwa.add_state();
        let right = nwa.add_state();
        nwa.set_start_states(vec![start]);

        for token in 0..=4 {
            nwa.add_transition(start, 7, left, tokens([token]));
        }
        for token in 5..=6 {
            nwa.add_transition(start, 7, right, tokens([token]));
        }
        nwa.set_final_weight(left, tokens(0..=4));
        nwa.set_final_weight(right, tokens(5..=6));

        let dwa = determinize(&nwa).unwrap();
        assert_eq!(dwa.eval_word(&[7]), tokens(0..=6));
    }

    #[test]
    fn direct_epsilon_free_subsets_match_generic_across_acyclic_cases() {
        for case in 0u32..32 {
            let mut nwa = NWA::new(1, 8);
            let states: Vec<u32> = (0..5).map(|_| nwa.add_state()).collect();
            nwa.set_start_states(vec![states[0]]);

            for from in 0..4usize {
                for to in (from + 1)..5usize {
                    if (case + (from * 7 + to * 11) as u32) % 3 == 0 {
                        continue;
                    }
                    let label = ((case + (from * 3 + to) as u32) % 4) as i32;
                    let first = (case + (from * 5 + to) as u32) % 6;
                    let second = (first + 1 + case % 3) % 7;
                    nwa.add_transition(states[from], label, states[to], tokens([first, second]));
                    if (case + from as u32 + to as u32) % 5 == 0 {
                        let extra = (second + 2) % 8;
                        nwa.add_transition(states[from], label, states[to], tokens([extra]));
                    }
                }
            }

            for state in 0..5usize {
                if (case + state as u32) % 2 == 0 {
                    let first = (case + state as u32) % 7;
                    nwa.set_final_weight(states[state], tokens([first]));
                }
            }

            // Exercise generic fallback as well: a transition into this state
            // must retain epsilon closure, while the remaining direct targets
            // may use the optimized path.
            if case % 4 == 0 {
                nwa.add_epsilon(states[1], states[4], tokens([case % 6]));
            }

            let fast = determinize_impl(&nwa, true).unwrap();
            let generic = determinize_impl(&nwa, false).unwrap();
            let grouped = determinize_impl_with_options(&nwa, true, true, false, false, None).unwrap().dwa;
            assert_eq!(
                find_difference(&fast, &generic).unwrap(),
                None,
                "case {case}",
            );
            assert_eq!(
                find_difference(&grouped, &generic).unwrap(),
                None,
                "grouped case {case}",
            );
        }
    }

    #[test]
    fn point_entry_aggregation_matches_generic_determinization() {
        let mut nwa = NWA::new(1, 8);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 5, first_accept, tokens([0, 2]));
        nwa.add_transition(start, 5, first_accept, tokens([1, 3]));
        nwa.add_transition(start, 5, second_accept, tokens([2, 4]));
        nwa.add_transition(start, 5, second_accept, tokens([5]));
        nwa.set_final_weight(first_accept, tokens([0, 1, 2, 3]));
        nwa.set_final_weight(second_accept, tokens([2, 4, 5]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();
        let grouped = determinize_impl_with_options(&nwa, true, true, false, false, None).unwrap().dwa;
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[5]), tokens([0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn grouped_singleton_edges_mix_direct_and_epsilon_fallback_exactly() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        let epsilon_source = nwa.add_state();
        let epsilon_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);

        // The three transition labels share one exact Weight group. Two are
        // directly emit-able; the third must keep the epsilon-closure fallback.
        let shared = tokens([0, 1]);
        nwa.add_transition(start, 7, first_accept, shared.clone());
        nwa.add_transition(start, 8, second_accept, shared.clone());
        nwa.add_transition(start, 9, epsilon_source, shared);
        nwa.add_epsilon(epsilon_source, epsilon_accept, tokens([0, 1]));
        nwa.set_final_weight(first_accept, tokens([0, 1]));
        nwa.set_final_weight(second_accept, tokens([0, 1]));
        nwa.set_final_weight(epsilon_accept, tokens([0, 1]));

        let grouped = determinize_impl_with_options(&nwa, true, true, false, false, None).unwrap().dwa;
        let generic = determinize_impl_with_options(&nwa, false, false, false, false, None).unwrap().dwa;
        assert_eq!(find_difference(&grouped, &generic).unwrap(), None);
        assert_eq!(grouped.eval_word(&[7]), tokens([0, 1]));
        assert_eq!(grouped.eval_word(&[8]), tokens([0, 1]));
        assert_eq!(grouped.eval_word(&[9]), tokens([0, 1]));
    }

    #[test]
    fn direct_single_target_path_matches_generic_determinization() {
        let mut nwa = NWA::new(1, 4);
        let start = nwa.add_state();
        let first_accept = nwa.add_state();
        let second_accept = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 7, first_accept, tokens([0, 1]));
        nwa.add_transition(start, 8, second_accept, tokens([2]));
        nwa.set_final_weight(first_accept, tokens([0, 1]));
        nwa.set_final_weight(second_accept, tokens([2]));

        let fast = determinize_impl(&nwa, true).unwrap();
        let generic = determinize_impl(&nwa, false).unwrap();

        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[7]), tokens([0, 1]));
        assert_eq!(fast.eval_word(&[8]), tokens([2]));

        let mut multi_destination = NWA::new(1, 4);
        let start = multi_destination.add_state();
        let first_accept = multi_destination.add_state();
        let second_accept = multi_destination.add_state();
        multi_destination.set_start_states(vec![start]);
        multi_destination.add_transition(start, 9, first_accept, tokens([0]));
        multi_destination.add_transition(start, 9, second_accept, tokens([1]));
        multi_destination.set_final_weight(first_accept, tokens([0]));
        multi_destination.set_final_weight(second_accept, tokens([1]));

        let fast = determinize_impl(&multi_destination, true).unwrap();
        let generic = determinize_impl(&multi_destination, false).unwrap();
        assert_eq!(find_difference(&fast, &generic).unwrap(), None);
        assert_eq!(fast.eval_word(&[9]), tokens([0, 1]));
    }
}
