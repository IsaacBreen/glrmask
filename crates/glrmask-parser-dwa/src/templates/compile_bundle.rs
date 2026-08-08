//! Template bundle assembly into a weighted NWA.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::sync::Arc;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::automata::unweighted_u32::determinize::determinize as unweighted_determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as unweighted_minimize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::compiler::glr::labels::{encode_negative_label, is_negative_label, negative_to_positive_label};
use crate::ds::weight::{SharedTokenSet, Weight};

type SubsetKey = SmallVec<[u64; 4]>;
type LabelTargets = SmallVec<[(i32, u32, u32); 8]>;
pub(crate) type BundleTopologySignature = Vec<Vec<TerminalID>>;
const SUBSET_BLOCK_BITS: usize = 8;
const SUBSET_BLOCK_MASK: u64 = (1u64 << SUBSET_BLOCK_BITS) - 1;

#[derive(Clone, Debug)]
struct BundleSkeletonState {
    final_groups: SmallVec<[u32; 8]>,
    transitions: BTreeMap<i32, (u32, SmallVec<[u32; 8]>)>,
}

#[derive(Clone, Debug)]
pub(crate) struct BundleSkeleton {
    states: Vec<BundleSkeletonState>,
    group_count: usize,
}


#[derive(Clone, Debug)]
enum PrepushBundleTarget {
    Core(u32),
    /// Weighted-union alternatives after the input-reading phase has ended.
    /// Each pair is `(weight_group, residual_group_dfa_state)`.
    Outputs(SmallVec<[(u32, u32); 8]>),
}

#[derive(Clone, Debug)]
struct PrepushBundleTransition {
    /// Groups contributing to this label while the deterministic input core is
    /// still live. Used only for `Core`; output alternatives carry their own
    /// group weight individually.
    groups: SmallVec<[u32; 8]>,
    target: PrepushBundleTarget,
}

#[derive(Clone, Debug)]
struct PrepushBundleState {
    final_groups: SmallVec<[u32; 8]>,
    transitions: BTreeMap<i32, PrepushBundleTransition>,
}

#[derive(Clone, Debug)]
struct PrepushBundleSkeleton {
    states: Vec<PrepushBundleState>,
    group_count: usize,
}


pub(crate) type PushSequence = SmallVec<[u32; 4]>;

#[derive(Clone, Debug)]
pub(crate) struct PrepushOutput {
    pub pushes: PushSequence,
    pub weight: Weight,
}

#[derive(Clone, Debug)]
pub(crate) enum WeightedPrepushTarget {
    Core { target: u32, weight: Weight },
    Outputs(Vec<PrepushOutput>),
}

#[derive(Clone, Debug)]
pub(crate) struct WeightedPrepushState {
    pub final_weight: Weight,
    pub outputs: Vec<PrepushOutput>,
    pub transitions: BTreeMap<i32, WeightedPrepushTarget>,
}

#[derive(Clone, Debug)]
pub(crate) struct WeightedPrepushBundle {
    pub states: Vec<WeightedPrepushState>,
}

#[derive(Clone, Debug)]
pub(crate) struct LazyWeightedPrepushBundle {
    skeleton: Arc<PrepushBundleSkeleton>,
    terminals: Vec<TerminalID>,
    weights: Vec<Weight>,
    group_weight_cache: FxHashMap<SmallVec<[u32; 8]>, Weight>,
}

#[derive(Debug)]
pub(crate) struct LazyWeightedPrepushBundleSet {
    bundles: Vec<Option<LazyWeightedPrepushBundle>>,
    decorated: Vec<Vec<Option<Arc<WeightedPrepushState>>>>,
    shared_sequences: FxHashMap<(usize, u32), Arc<[PushSequence]>>,
    profile: bool,
    decorate_calls: usize,
    decorate_hits: usize,
    decorate_misses: usize,
    decorate_ms: f64,
    decorate_group_view_ms: f64,
    decorate_sequence_memo_alloc_ms: f64,
    decorate_instantiate_ms: f64,
    decorate_final_weight_ms: f64,
    decorate_core_weight_ms: f64,
    decorate_output_expand_ms: f64,
    decorate_output_sequence_ms: f64,
    decorate_output_group_ms: f64,
    decorate_transition_insert_ms: f64,
    decorate_output_sort_ms: f64,
    decorate_residual_calls: usize,
    decorate_residual_hits: usize,
    decorate_residual_misses: usize,
    decorate_output_program_refs: usize,
    decorate_output_sequence_refs: usize,
    decorate_output_unique_sequences: usize,
    decorate_core_weight_unions: usize,
    decorate_output_weight_unions: usize,
}

/// A per-weight terminal group either reuses an immutable template DFA or owns
/// the union required for a multi-terminal group. Keeping singleton groups
/// borrowed avoids cloning their full DFA only to read it once while building a
/// deterministic bundle.
enum BundleGroupDfa<'a> {
    Borrowed(&'a UnweightedDfa),
    Owned(UnweightedDfa),
}

impl BundleGroupDfa<'_> {
    #[inline]
    fn dfa(&self) -> &UnweightedDfa {
        match self {
            Self::Borrowed(dfa) => dfa,
            Self::Owned(dfa) => dfa,
        }
    }
}

fn checked_usize_to_u32(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{what} exceeds u32::MAX"))
}

fn checked_u32_add(lhs: u32, rhs: u32, what: &str) -> u32 {
    lhs.checked_add(rhs)
        .unwrap_or_else(|| panic!("{what} exceeds u32::MAX"))
}

fn empty_bundle_nwa() -> NWA {
    let mut nwa = NWA::new(0, 0);
    let start_state = nwa.add_state();
    nwa.start_states_mut().push(start_state);
    nwa
}

fn instantiate_weighted_nwa_from_skeleton(skeleton: &NWA, weight: &Weight) -> NWA {
    let mut bundle = skeleton.clone();
    for state in  bundle.states_mut() {
        if state.final_weight.is_some() {
            state.final_weight = Some(weight.clone());
        }
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets {
                *edge_weight = weight.clone();
            }
        }
        for (_, epsilon_weight) in &mut state.epsilons {
            *epsilon_weight = weight.clone();
        }
    }

    bundle
}

fn clear_subset_key(key: &mut SubsetKey) {
    for word in key.iter_mut() {
        *word = 0;
    }
}

fn collect_label_targets(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    product_state: &[(u32, u32)],
    label_targets: &mut LabelTargets,
) {
    label_targets.clear();
    for &(group_id, dfa_state) in product_state {
        let dfa = groups[group_id as usize].1.dfa();
        for (&label, &target) in &dfa.states[dfa_state as usize].transitions {
            label_targets.push((label, group_id, target));
        }
    }
    // Preserve the former BTreeMap's ascending-label traversal, while retaining
    // each product state's canonical group order within a label.
    label_targets.sort_unstable_by_key(|&(label, group_id, _)| (label, group_id));
}

fn set_subset_key_bit(key: &mut SubsetKey, index: usize) {
    let word_index = index / 64;
    let bit_index = index % 64;
    key[word_index] |= 1u64 << bit_index;
}

fn cached_subset_union(
    cache: &mut FxHashMap<SubsetKey, Weight>,
    subset_key: &SubsetKey,
    subset: &[usize],
    group_weights: &[Weight],
    subset_unions: Option<&SubsetUnionIndex>,
    single_tsid_entries: Option<&[(u32, SharedTokenSet)]>,
) -> Weight {
    match subset {
        [] => return Weight::empty(),
        [index] => return group_weights[*index].clone(),
        _ => {}
    }

    if let Some(existing) = cache.get(subset_key) {
        return existing.clone();
    }

    let result = if let Some(single_tsid_entries) = single_tsid_entries {
        Weight::union_single_tsid_shared_entries(
            subset
                .iter()
                .map(|&index| {
                    let (tsid, tokens) = &single_tsid_entries[index];
                    (*tsid, Arc::clone(tokens))
                }),
        )
    } else if subset.len() >= SUBSET_BLOCK_BITS {
        if let Some(subset_unions) = subset_unions {
            subset_unions.union(subset_key)
        } else {
            Weight::union_all(subset.iter().map(|&index| &group_weights[index]))
        }
    } else {
        Weight::union_all(subset.iter().map(|&index| &group_weights[index]))
    };
    cache.insert(subset_key.clone(), result.clone());
    result
}

struct SubsetUnionIndex {
    block_unions: Vec<Box<[Weight]>>,
    segment_base: usize,
    segment_unions: Vec<Weight>,
}

impl SubsetUnionIndex {
    fn new(group_weights: &[Weight]) -> Self {
        Self {
            block_unions: build_subset_block_unions(group_weights),
            segment_base: group_weights.len().next_power_of_two(),
            segment_unions: build_subset_segment_unions(group_weights),
        }
    }

    fn union(&self, subset_key: &SubsetKey) -> Weight {
        let block_parts = self.block_parts(subset_key);
        let segment_parts = self.segment_parts(subset_key);
        if segment_parts.len() < block_parts.len() {
            Weight::union_all(segment_parts)
        } else {
            Weight::union_all(block_parts)
        }
    }

    fn block_parts<'a>(&'a self, subset_key: &SubsetKey) -> SmallVec<[&'a Weight; 8]> {
        let mut parts = SmallVec::<[&Weight; 8]>::new();
        for (block_index, block_table) in self.block_unions.iter().enumerate() {
            let bit_offset = block_index * SUBSET_BLOCK_BITS;
            let word_index = bit_offset / 64;
            let bit_index = bit_offset % 64;
            let mask = ((subset_key[word_index] >> bit_index) & SUBSET_BLOCK_MASK) as usize;
            if mask != 0 {
                parts.push(&block_table[mask]);
            }
        }
        parts
    }

    fn segment_parts<'a>(&'a self, subset_key: &SubsetKey) -> SmallVec<[&'a Weight; 8]> {
        let mut parts = SmallVec::<[&Weight; 8]>::new();
        for (word_index, &word) in subset_key.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let start_bit = remaining.trailing_zeros() as usize;
                let shifted = remaining >> start_bit;
                let run_len = (!shifted).trailing_zeros().min(64 - start_bit as u32) as usize;
                let start = word_index * 64 + start_bit;
                self.push_segment_range_parts(start, start + run_len, &mut parts);
                let run_mask = if run_len == 64 {
                    u64::MAX
                } else {
                    ((1u64 << run_len) - 1) << start_bit
                };
                remaining &= !run_mask;
            }
        }
        parts
    }

    fn push_segment_range_parts<'a>(
        &'a self,
        mut start: usize,
        mut end: usize,
        parts: &mut SmallVec<[&'a Weight; 8]>,
    ) {
        start += self.segment_base;
        end += self.segment_base;
        while start < end {
            if start % 2 == 1 {
                let weight = &self.segment_unions[start];
                if !weight.is_empty() {
                    parts.push(weight);
                }
                start += 1;
            }
            if end % 2 == 1 {
                end -= 1;
                let weight = &self.segment_unions[end];
                if !weight.is_empty() {
                    parts.push(weight);
                }
            }
            start /= 2;
            end /= 2;
        }
    }
}

fn build_subset_block_unions(group_weights: &[Weight]) -> Vec<Box<[Weight]>> {
    group_weights
        .chunks(SUBSET_BLOCK_BITS)
        .map(|chunk| {
            let table_len = 1usize << chunk.len();
            let mut unions = vec![Weight::empty(); table_len];
            for mask in 1..table_len {
                let bit = mask.trailing_zeros() as usize;
                let prev = mask & (mask - 1);
                unions[mask] = unions[prev].union(&chunk[bit]);
            }
            unions.into_boxed_slice()
        })
        .collect()
}

fn build_subset_segment_unions(group_weights: &[Weight]) -> Vec<Weight> {
    let base = group_weights.len().next_power_of_two();
    let mut tree = vec![Weight::empty(); base * 2];
    for (index, weight) in group_weights.iter().enumerate() {
        tree[base + index] = weight.clone();
    }
    for index in (1..base).rev() {
        tree[index] = tree[index * 2].union(&tree[index * 2 + 1]);
    }
    tree
}

#[derive(Clone, Debug, Default)]
pub struct BundleBuildProfile {
    pub input_terminals: usize,
    pub nonempty_terminals: usize,
    pub weight_groups: usize,
    pub overlap_components: usize,
    pub largest_overlap_component: usize,
    pub single_entry_weights: usize,
    pub single_tsid_weights: usize,
    pub total_weight_outer_ranges: usize,
    pub singleton_groups: usize,
    pub multi_terminal_groups: usize,
    pub largest_weight_group: usize,
    pub build_group_dfas_ms: f64,
    pub union_groups_ms: f64,
    pub slowest_group_terminals: usize,
    pub slowest_group_dfa_states: usize,
    pub slowest_group_dfa_transitions: usize,
    pub slowest_group_ms: f64,
    pub determinize_bundle_ms: f64,
    pub determinize_pop_state_ms: f64,
    pub determinize_alive_groups_ms: f64,
    pub determinize_effective_weights_ms: f64,
    pub determinize_final_weight_ms: f64,
    pub determinize_collect_labels_ms: f64,
    pub determinize_next_state_ms: f64,
    pub determinize_edge_weight_ms: f64,
    pub determinize_state_lookup_ms: f64,
    pub determinize_add_transition_ms: f64,
    pub determinize_states_visited: usize,
    pub determinize_labels_processed: usize,
    pub determinize_transitions_added: usize,
    pub determinize_worklist_peak: usize,
    pub determinize_cache_entries: usize,
    pub determinize_edge_subset_total: usize,
    pub determinize_edge_subset_max: usize,
    pub determinize_edge_cache_hits: usize,
    pub determinize_edge_cache_hit_subset_total: usize,
    pub determinize_edge_cache_misses: usize,
    pub determinize_edge_cache_miss_subset_total: usize,
    pub minimize_ms: f64,
    pub dwa_to_nwa_ms: f64,
    pub result_dwa_states: usize,
    pub result_dwa_transitions: usize,
    pub result_nwa_states: usize,
    pub result_nwa_transitions: usize,
    pub result_negative_only_states: usize,
    pub result_positive_only_states: usize,
    pub result_mixed_label_states: usize,
    pub result_unlabeled_states: usize,
    pub result_negative_transitions: usize,
    pub result_positive_transitions: usize,
    pub truncated_reachable_states: usize,
    pub truncated_push_frontier_states: usize,
    pub truncated_edges_traversed: usize,
    pub prepush_states: usize,
    pub prepush_input_transitions: usize,
    pub prepush_output_edges: usize,
    pub prepush_output_sites: usize,
    pub prepush_output_programs: usize,
    pub prepush_core_states: usize,
    pub prepush_frontier_payloads: usize,
    pub prepush_frontier_final_payloads: usize,
    pub prepush_frontier_push_edges: usize,
    pub prepush_census_ms: f64,
    pub prepush_program_sequences: usize,
    pub prepush_programs_multisequence: usize,
    pub prepush_max_sequences_per_program: usize,
    pub prepush_max_push_depth: usize,
    pub total_ms: f64,
    pub used_single_terminal_fast_path: bool,
    pub minimize_skipped: bool,
}

fn weight_overlap_components(weight_groups: &[(&Weight, Vec<TerminalID>)]) -> Vec<Vec<usize>> {
    let n = weight_groups.len();
    if n == 0 {
        return Vec::new();
    }
    let mut parent = (0..n).collect::<Vec<_>>();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for left in 0..n {
        for right in left + 1..n {
            if weight_groups[left]
                .0
                .intersection(weight_groups[right].0)
                .is_empty()
            {
                continue;
            }
            let left_root = find(&mut parent, left);
            let right_root = find(&mut parent, right);
            if left_root != right_root {
                parent[right_root] = left_root;
            }
        }
    }

    let mut components = FxHashMap::<usize, Vec<usize>>::default();
    for index in 0..n {
        let root = find(&mut parent, index);
        components.entry(root).or_default().push(index);
    }
    let mut result = components.into_values().collect::<Vec<_>>();
    result.sort_unstable_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left[0].cmp(&right[0]))
    });
    result
}

fn template_bundle_overlap_components_enabled() -> bool {
    std::env::var("GLRMASK_TEMPLATE_BUNDLE_OVERLAP_COMPONENTS")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn minimize_template_bundles_enabled() -> bool {
    std::env::var("GLRMASK_MINIMIZE_TEMPLATE_BUNDLES")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn template_bundle_subset_union_index_enabled() -> bool {
    std::env::var("GLRMASK_TEMPLATE_BUNDLE_SUBSET_UNION_INDEX")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn compute_unweighted_dfa_transition_count(dfa: &UnweightedDfa) -> usize {
    dfa.states.iter().map(|state| state.transitions.len()).sum()
}

impl Templates {
    fn build_single_terminal_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> Option<NWA> {
        let (&terminal, weight) = terminal_weights.iter().next()?;
        if terminal_weights.len() != 1 {
            return None;
        }
        if weight.is_empty() {
            return Some(empty_bundle_nwa());
        }
        Some(
            self.by_terminal_nwa
                .get(&terminal)
                .map(|template_nwa| instantiate_weighted_nwa_from_skeleton(template_nwa, weight))
                .unwrap_or_else(empty_bundle_nwa),
        )
    }

    fn group_terminals_by_weight<'a>(
        &'a self,
        terminal_weights: &'a BTreeMap<TerminalID, Weight>,
    ) -> Vec<(&'a Weight, Vec<TerminalID>)> {
        let mut weight_groups: HashMap<&Weight, Vec<TerminalID>> = HashMap::new();
        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() || !self.by_terminal.contains_key(&terminal) {
                continue;
            }
            weight_groups.entry(weight).or_default().push(terminal);
        }

        // Terminal IDs come from a BTreeMap, so each group's first terminal is
        // stable. This gives the product construction a deterministic order
        // without traversing every group DFA to allocate a structural sort key.
        let mut groups = weight_groups.into_iter().collect::<Vec<_>>();
        groups.sort_unstable_by_key(|(_, terminals)| terminals[0]);
        groups
    }

    pub(crate) fn bundle_topology_signature(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> BundleTopologySignature {
        self.group_terminals_by_weight(terminal_weights)
            .into_iter()
            .map(|(_, terminals)| terminals)
            .collect()
    }

    pub(crate) fn build_bundle_skeleton(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> Option<BundleSkeleton> {
        if terminal_weights.len() <= 1 {
            return None;
        }
        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        if weight_groups.is_empty() {
            return None;
        }
        let group_dfas = self.build_group_dfas(&weight_groups);
        Some(determinize_bundle_groups_skeleton(&group_dfas))
    }

    pub(crate) fn instantiate_bundle_skeleton(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        skeleton: &BundleSkeleton,
    ) -> NWA {
        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        assert_eq!(
            weight_groups.len(),
            skeleton.group_count,
            "bundle skeleton group count does not match concrete bundle",
        );
        let group_weights = weight_groups
            .into_iter()
            .map(|(weight, _)| weight.clone())
            .collect::<Vec<_>>();
        let bundle_dwa = instantiate_bundle_skeleton_dwa(skeleton, &group_weights);
        let bundle_dwa = if group_weights.len() > 1 && minimize_template_bundles_enabled() {
            minimize(&bundle_dwa)
        } else {
            bundle_dwa
        };
        dwa_to_nwa(&bundle_dwa)
    }

    fn build_group_dfas_profiled<'a>(
        &'a self,
        weight_groups: &'a [(&'a Weight, Vec<TerminalID>)],
        profile: &mut BundleBuildProfile,
    ) -> Vec<(&'a Weight, BundleGroupDfa<'a>)> {
        let build_started_at = Instant::now();
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            profile.nonempty_terminals += terminals.len();
            profile.largest_weight_group = profile.largest_weight_group.max(terminals.len());
            if terminals.len() == 1 {
                profile.singleton_groups += 1;
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, BundleGroupDfa::Borrowed(template)));
                }
                continue;
            }

            profile.multi_terminal_groups += 1;
            let group_started_at = Instant::now();
            let merged = union_unweighted_dfas(
                terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
            );
            let group_ms = elapsed_ms(group_started_at);
            profile.union_groups_ms += group_ms;

            if group_ms > profile.slowest_group_ms {
                profile.slowest_group_ms = group_ms;
                profile.slowest_group_terminals = terminals.len();
                profile.slowest_group_dfa_states = merged.states.len();
                profile.slowest_group_dfa_transitions = compute_unweighted_dfa_transition_count(&merged);
            }

            group_dfas.push((*weight, BundleGroupDfa::Owned(merged)));
        }
        profile.build_group_dfas_ms = elapsed_ms(build_started_at);
        group_dfas
    }

    fn build_group_dfas<'a>(
        &'a self,
        weight_groups: &'a [(&'a Weight, Vec<TerminalID>)],
    ) -> Vec<(&'a Weight, BundleGroupDfa<'a>)> {
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            if terminals.len() == 1 {
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, BundleGroupDfa::Borrowed(template)));
                }
            } else {
                let merged = union_unweighted_dfas(
                    terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
                );
                group_dfas.push((*weight, BundleGroupDfa::Owned(merged)));
            }
        }
        group_dfas
    }

    fn build_bundle_by_overlap_components(
        &self,
        weight_groups: &[(&Weight, Vec<TerminalID>)],
    ) -> Option<NWA> {
        let components = weight_overlap_components(weight_groups);
        if components.len() <= 1 {
            return None;
        }

        let mut union = NWA::new(0, 0);
        for component in components {
            let component_weight_groups = component
                .into_iter()
                .map(|index| (weight_groups[index].0, weight_groups[index].1.clone()))
                .collect::<Vec<_>>();
            let component_group_dfas = self.build_group_dfas(&component_weight_groups);
            let component_dwa = determinize_bundle_groups(&component_group_dfas);
            let component_dwa = if component_weight_groups.len() > 1
                && minimize_template_bundles_enabled()
            {
                minimize(&component_dwa)
            } else {
                component_dwa
            };
            let component_nwa = dwa_to_nwa(&component_dwa);
            let component_body = union.append_with_body(&component_nwa);
            union
                .start_states_mut()
                .extend(component_body.start_states.into_iter());
        }
        Some(union)
    }

    pub(crate) fn build_lazy_weighted_prepush_bundles_cached(
        &self,
        bundles: &[BTreeMap<TerminalID, Weight>],
        used: &[bool],
    ) -> LazyWeightedPrepushBundleSet {
        assert_eq!(bundles.len(), used.len());
        let profile = std::env::var_os("GLRMASK_PROFILE_PREPUSH_LAZY_DECORATION").is_some();
        let started = Instant::now();
        let mut skeletons = FxHashMap::<Vec<TerminalID>, Arc<PrepushBundleSkeleton>>::default();
        let mut plans = Vec::with_capacity(bundles.len());
        let mut decorated = Vec::with_capacity(bundles.len());
        let mut skeleton_hits = 0usize;
        let mut skeleton_misses = 0usize;
        for (bundle, used) in bundles.iter().zip(used.iter().copied()) {
            if !used {
                plans.push(None);
                decorated.push(Vec::new());
                continue;
            }
            let entries = bundle
                .iter()
                .filter(|(_, weight)| !weight.is_empty())
                .map(|(&terminal, weight)| (terminal, weight.clone()))
                .collect::<Vec<_>>();
            let key = entries.iter().map(|(terminal, _)| *terminal).collect::<Vec<_>>();
            let skeleton = if let Some(existing) = skeletons.get(&key) {
                skeleton_hits += 1;
                Arc::clone(existing)
            } else {
                skeleton_misses += 1;
                let group_dfas = entries
                    .iter()
                    .filter_map(|(terminal, weight)| {
                        self.by_terminal
                            .get(terminal)
                            .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                    })
                    .collect::<Vec<_>>();
                let built = Arc::new(build_prepush_bundle_skeleton(&group_dfas));
                skeletons.insert(key, Arc::clone(&built));
                built
            };
            decorated.push(vec![None; skeleton.states.len()]);
            plans.push(Some(LazyWeightedPrepushBundle {
                skeleton,
                terminals: entries.iter().map(|(terminal, _)| *terminal).collect(),
                weights: entries.into_iter().map(|(_, weight)| weight).collect(),
                group_weight_cache: FxHashMap::default(),
            }));
        }
        if profile {
            eprintln!(
                "[glrmask/profile][prepush_lazy_plan] used={} unique_skeletons={} skeleton_hits={} skeleton_misses={} plan_ms={:.3}",
                used.iter().filter(|&&value| value).count(),
                skeletons.len(),
                skeleton_hits,
                skeleton_misses,
                elapsed_ms(started),
            );
        }
        LazyWeightedPrepushBundleSet {
            bundles: plans,
            decorated,
            shared_sequences: FxHashMap::default(),
            profile,
            decorate_calls: 0,
            decorate_hits: 0,
            decorate_misses: 0,
            decorate_ms: 0.0,
            decorate_group_view_ms: 0.0,
            decorate_sequence_memo_alloc_ms: 0.0,
            decorate_instantiate_ms: 0.0,
            decorate_final_weight_ms: 0.0,
            decorate_core_weight_ms: 0.0,
            decorate_output_expand_ms: 0.0,
            decorate_output_sequence_ms: 0.0,
            decorate_output_group_ms: 0.0,
            decorate_transition_insert_ms: 0.0,
            decorate_output_sort_ms: 0.0,
            decorate_residual_calls: 0,
            decorate_residual_hits: 0,
            decorate_residual_misses: 0,
            decorate_output_program_refs: 0,
            decorate_output_sequence_refs: 0,
            decorate_output_unique_sequences: 0,
            decorate_core_weight_unions: 0,
            decorate_output_weight_unions: 0,
        }
    }

    pub(crate) fn lazy_weighted_prepush_bundle_is_empty(
        &self,
        set: &LazyWeightedPrepushBundleSet,
        bundle_id: usize,
    ) -> bool {
        set.bundles
            .get(bundle_id)
            .and_then(Option::as_ref)
            .is_none_or(|bundle| bundle.skeleton.states.is_empty())
    }

    pub(crate) fn lazy_weighted_prepush_state(
        &self,
        set: &mut LazyWeightedPrepushBundleSet,
        bundle_id: usize,
        state_id: u32,
    ) -> Option<Arc<WeightedPrepushState>> {
        set.decorate_calls += 1;
        let slot = set.decorated.get(bundle_id)?.get(state_id as usize)?;
        if let Some(existing) = slot {
            set.decorate_hits += 1;
            return Some(Arc::clone(existing));
        }
        set.decorate_misses += 1;
        let started = set.profile.then(Instant::now);
        let (bundles, shared_sequences) = (&mut set.bundles, &mut set.shared_sequences);
        let plan = bundles.get_mut(bundle_id)?.as_mut()?;
        let LazyWeightedPrepushBundle {
            skeleton,
            terminals,
            weights,
            group_weight_cache,
        } = plan;
        let group_view_started = set.profile.then(Instant::now);
        let group_dfas = terminals
            .iter()
            .zip(weights.iter())
            .filter_map(|(terminal, weight)| {
                self.by_terminal
                    .get(terminal)
                    .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
            })
            .collect::<Vec<_>>();
        if let Some(started) = group_view_started {
            set.decorate_group_view_ms += elapsed_ms(started);
        }
        if group_dfas.len() != weights.len() {
            return None;
        }
        let skeleton_state = skeleton.states.get(state_id as usize)?;
        let memo_started = set.profile.then(Instant::now);
        let mut sequence_memos = (0..group_dfas.len())
            .map(|_| FxHashMap::<u32, Arc<[PushSequence]>>::default())
            .collect::<Vec<_>>();
        if let Some(started) = memo_started {
            set.decorate_sequence_memo_alloc_ms += elapsed_ms(started);
        }
        let mut detail = PrepushInstantiateDetail::default();
        let instantiate_started = set.profile.then(Instant::now);
        let state = instantiate_weighted_prepush_state_profiled(
            skeleton_state,
            &group_dfas,
            &mut sequence_memos,
            shared_sequences,
            Some(group_weight_cache),
            &mut detail,
        );
        if let Some(started) = instantiate_started {
            set.decorate_instantiate_ms += elapsed_ms(started);
        }
        if set.profile {
            set.decorate_final_weight_ms += detail.final_weight_ms;
            set.decorate_core_weight_ms += detail.core_weight_ms;
            set.decorate_output_expand_ms += detail.output_expand_ms;
            set.decorate_output_sequence_ms += detail.output_sequence_ms;
            set.decorate_output_group_ms += detail.output_group_ms;
            set.decorate_transition_insert_ms += detail.transition_insert_ms;
            set.decorate_output_sort_ms += detail.output_sort_ms;
            set.decorate_residual_calls += detail.residual_calls;
            set.decorate_residual_hits += detail.residual_cache_hits;
            set.decorate_residual_misses += detail.residual_cache_misses;
            set.decorate_output_program_refs += detail.output_program_refs;
            set.decorate_output_sequence_refs += detail.output_sequence_refs;
            set.decorate_output_unique_sequences += detail.output_unique_sequences;
            set.decorate_core_weight_unions += detail.core_weight_unions;
            set.decorate_output_weight_unions += detail.output_weight_unions;
        }
        let state = Arc::new(state);
        set.decorated[bundle_id][state_id as usize] = Some(Arc::clone(&state));
        if let Some(started) = started {
            set.decorate_ms += elapsed_ms(started);
        }
        Some(state)
    }

    pub(crate) fn emit_lazy_weighted_prepush_profile(
        &self,
        set: &LazyWeightedPrepushBundleSet,
    ) {
        if !set.profile {
            return;
        }
        let cached_states = set
            .decorated
            .iter()
            .map(|states| states.iter().filter(|state| state.is_some()).count())
            .sum::<usize>();
        let mut multi_group_union_sites = 0usize;
        let mut distinct_multi_group_unions = 0usize;
        for (plan, decorated) in set.bundles.iter().zip(&set.decorated) {
            let Some(plan) = plan else { continue };
            let mut signatures = FxHashSet::<SmallVec<[u32; 8]>>::default();
            for (state_id, cached) in decorated.iter().enumerate() {
                if cached.is_none() {
                    continue;
                }
                let Some(state) = plan.skeleton.states.get(state_id) else { continue };
                if state.final_groups.len() > 1 {
                    multi_group_union_sites += 1;
                    signatures.insert(state.final_groups.clone());
                }
                for transition in state.transitions.values() {
                    if matches!(transition.target, PrepushBundleTarget::Core(_))
                        && transition.groups.len() > 1
                    {
                        multi_group_union_sites += 1;
                        signatures.insert(transition.groups.clone());
                    }
                }
            }
            distinct_multi_group_unions += signatures.len();
        }
        eprintln!(
            "[glrmask/profile][prepush_lazy_decorate] calls={} hits={} misses={} cached_states={} shared_sequence_entries={} decorate_ms={:.3} group_view_ms={:.3} sequence_memo_alloc_ms={:.3} instantiate_ms={:.3} final_weight_ms={:.3} core_weight_ms={:.3} output_expand_ms={:.3} output_sequence_ms={:.3} output_group_ms={:.3} transition_insert_ms={:.3} output_sort_ms={:.3} residual_calls={} residual_hits={} residual_misses={} output_program_refs={} output_sequence_refs={} output_unique_sequences={} core_weight_unions={} output_weight_unions={} multi_group_union_sites={} distinct_multi_group_unions={}",
            set.decorate_calls,
            set.decorate_hits,
            set.decorate_misses,
            cached_states,
            set.shared_sequences.len(),
            set.decorate_ms,
            set.decorate_group_view_ms,
            set.decorate_sequence_memo_alloc_ms,
            set.decorate_instantiate_ms,
            set.decorate_final_weight_ms,
            set.decorate_core_weight_ms,
            set.decorate_output_expand_ms,
            set.decorate_output_sequence_ms,
            set.decorate_output_group_ms,
            set.decorate_transition_insert_ms,
            set.decorate_output_sort_ms,
            set.decorate_residual_calls,
            set.decorate_residual_hits,
            set.decorate_residual_misses,
            set.decorate_output_program_refs,
            set.decorate_output_sequence_refs,
            set.decorate_output_unique_sequences,
            set.decorate_core_weight_unions,
            set.decorate_output_weight_unions,
            multi_group_union_sites,
            distinct_multi_group_unions,
        );
    }

    pub(crate) fn build_weighted_prepush_bundles_cached(
        &self,
        bundles: &[BTreeMap<TerminalID, Weight>],
        used: &[bool],
    ) -> Vec<Option<Arc<WeightedPrepushBundle>>> {
        assert_eq!(bundles.len(), used.len());
        if std::env::var_os("GLRMASK_PREPUSH_NO_WEIGHT_GROUP_UNION").is_none() {
            return bundles
                .iter()
                .zip(used.iter().copied())
                .map(|(bundle, used)| used.then(|| Arc::new(self.build_weighted_prepush_bundle(bundle))))
                .collect();
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_PREPUSH_BUNDLE_BATCH").is_some();
        let total_started = Instant::now();
        let mut skeleton_ms = 0.0;
        let mut instantiate_ms = 0.0;
        let mut skeleton_hits = 0usize;
        let mut skeleton_misses = 0usize;
        let mut output_states = 0usize;
        let mut output_transitions = 0usize;
        let mut output_programs = 0usize;
        let mut detail_final_weight_ms = 0.0;
        let mut detail_core_weight_ms = 0.0;
        let mut detail_output_expand_ms = 0.0;
        let mut detail_output_sequence_ms = 0.0;
        let mut detail_output_group_ms = 0.0;
        let mut detail_output_weight_ms = 0.0;
        let mut detail_transition_insert_ms = 0.0;
        let mut detail_output_sort_ms = 0.0;
        let mut detail_residual_calls = 0usize;
        let mut detail_residual_hits = 0usize;
        let mut detail_residual_misses = 0usize;
        let mut detail_output_program_refs = 0usize;
        let mut detail_output_sequence_refs = 0usize;
        let mut detail_output_unique_sequences = 0usize;
        let mut detail_core_weight_unions = 0usize;
        let mut detail_output_weight_unions = 0usize;
        let mut skeletons = FxHashMap::<Vec<TerminalID>, Arc<PrepushBundleSkeleton>>::default();
        let mut result = Vec::with_capacity(bundles.len());
        for (bundle, used) in bundles.iter().zip(used.iter().copied()) {
            if !used {
                result.push(None);
                continue;
            }
            let group_dfas = bundle
                .iter()
                .filter_map(|(&terminal, weight)| {
                    if weight.is_empty() {
                        return None;
                    }
                    self.by_terminal
                        .get(&terminal)
                        .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                })
                .collect::<Vec<_>>();
            let key = bundle
                .iter()
                .filter_map(|(&terminal, weight)| (!weight.is_empty()).then_some(terminal))
                .collect::<Vec<_>>();
            let skeleton_started = profile.then(Instant::now);
            let skeleton = if let Some(existing) = skeletons.get(&key) {
                skeleton_hits += 1;
                Arc::clone(existing)
            } else {
                skeleton_misses += 1;
                let built = Arc::new(build_prepush_bundle_skeleton(&group_dfas));
                skeletons.insert(key, Arc::clone(&built));
                built
            };
            if let Some(started) = skeleton_started {
                skeleton_ms += elapsed_ms(started);
            }
            let instantiate_started = profile.then(Instant::now);
            // `group_dfas` may contain temporary `Owned` DFAs. Residual-sequence
            // cache keys currently use the DFA address, so carrying this cache
            // across bundle iterations is unsound: an allocator can reuse an
            // owned DFA's address for a different DFA in a later bundle (ABA),
            // causing us to reuse the wrong WRITE suffix language. Keep sharing
            // within one bundle, where every referenced DFA remains alive.
            let mut shared_sequences = SharedResidualSequenceCache::default();
            let (weighted, inst_detail) = if profile {
                instantiate_weighted_prepush_bundle_profiled(
                    skeleton.as_ref(),
                    &group_dfas,
                    &mut shared_sequences,
                )
            } else {
                (
                    instantiate_weighted_prepush_bundle_profiled(
                        skeleton.as_ref(),
                        &group_dfas,
                        &mut shared_sequences,
                    )
                    .0,
                    PrepushInstantiateDetail::default(),
                )
            };
            if let Some(started) = instantiate_started {
                instantiate_ms += elapsed_ms(started);
            }
            if profile {
                detail_final_weight_ms += inst_detail.final_weight_ms;
                detail_core_weight_ms += inst_detail.core_weight_ms;
                detail_output_expand_ms += inst_detail.output_expand_ms;
                detail_output_sequence_ms += inst_detail.output_sequence_ms;
                detail_output_group_ms += inst_detail.output_group_ms;
                detail_output_weight_ms += inst_detail.output_weight_ms;
                detail_transition_insert_ms += inst_detail.transition_insert_ms;
                detail_output_sort_ms += inst_detail.output_sort_ms;
                detail_residual_calls += inst_detail.residual_calls;
                detail_residual_hits += inst_detail.residual_cache_hits;
                detail_residual_misses += inst_detail.residual_cache_misses;
                detail_output_program_refs += inst_detail.output_program_refs;
                detail_output_sequence_refs += inst_detail.output_sequence_refs;
                detail_output_unique_sequences += inst_detail.output_unique_sequences;
                detail_core_weight_unions += inst_detail.core_weight_unions;
                detail_output_weight_unions += inst_detail.output_weight_unions;
            }
            if profile {
                output_states += weighted.states.len();
                for state in &weighted.states {
                    output_transitions += state.transitions.len();
                    output_programs += state.outputs.len();
                    for transition in state.transitions.values() {
                        if let WeightedPrepushTarget::Outputs(outputs) = transition {
                            output_programs += outputs.len();
                        }
                    }
                }
            }
            result.push(Some(Arc::new(weighted)));
        }
        if profile {
            eprintln!(
                "[glrmask/profile][prepush_bundle_batch] used={} unique_skeletons={} skeleton_hits={} skeleton_misses={} output_states={} output_transitions={} output_programs={} skeleton_ms={:.3} instantiate_ms={:.3} final_weight_ms={:.3} core_weight_ms={:.3} output_expand_ms={:.3} output_sequence_ms={:.3} output_group_ms={:.3} output_weight_ms={:.3} transition_insert_ms={:.3} output_sort_ms={:.3} residual_calls={} residual_hits={} residual_misses={} output_program_refs={} output_sequence_refs={} output_unique_sequences={} core_weight_unions={} output_weight_unions={} total_ms={:.3}",
                used.iter().filter(|&&value| value).count(),
                skeletons.len(),
                skeleton_hits,
                skeleton_misses,
                output_states,
                output_transitions,
                output_programs,
                skeleton_ms,
                instantiate_ms,
                detail_final_weight_ms,
                detail_core_weight_ms,
                detail_output_expand_ms,
                detail_output_sequence_ms,
                detail_output_group_ms,
                detail_output_weight_ms,
                detail_transition_insert_ms,
                detail_output_sort_ms,
                detail_residual_calls,
                detail_residual_hits,
                detail_residual_misses,
                detail_output_program_refs,
                detail_output_sequence_refs,
                detail_output_unique_sequences,
                detail_core_weight_unions,
                detail_output_weight_unions,
                elapsed_ms(total_started),
            );
        }
        result
    }

    pub(crate) fn build_prepush_frontier_write_trie_bundle(
        &self,
        bundle: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if bundle.is_empty() {
            return empty_bundle_nwa();
        }
        if std::env::var_os("GLRMASK_PREPUSH_NO_WEIGHT_GROUP_UNION").is_some() {
            let group_dfas = bundle
                .iter()
                .filter_map(|(&terminal, weight)| {
                    if weight.is_empty() {
                        return None;
                    }
                    self.by_terminal
                        .get(&terminal)
                        .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                })
                .collect::<Vec<_>>();
            let skeleton = build_prepush_bundle_skeleton(&group_dfas);
            let weighted = instantiate_weighted_prepush_bundle(&skeleton, &group_dfas);
            return instantiate_weighted_prepush_frontier_write_trie_nwa(&weighted);
        }
        let weight_groups = self.group_terminals_by_weight(bundle);
        let mut profile = BundleBuildProfile::default();
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);
        let skeleton = build_prepush_bundle_skeleton(&group_dfas);
        let weighted = instantiate_weighted_prepush_bundle(&skeleton, &group_dfas);
        instantiate_weighted_prepush_frontier_write_trie_nwa(&weighted)
    }

    pub(crate) fn build_prepush_compact_write_trie_bundle(
        &self,
        bundle: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if bundle.is_empty() {
            return empty_bundle_nwa();
        }
        if std::env::var_os("GLRMASK_PREPUSH_NO_WEIGHT_GROUP_UNION").is_some() {
            let group_dfas = bundle
                .iter()
                .filter_map(|(&terminal, weight)| {
                    if weight.is_empty() {
                        return None;
                    }
                    self.by_terminal
                        .get(&terminal)
                        .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                })
                .collect::<Vec<_>>();
            let skeleton = build_prepush_bundle_skeleton(&group_dfas);
            let weighted = instantiate_weighted_prepush_bundle(&skeleton, &group_dfas);
            return instantiate_weighted_prepush_compact_write_trie_nwa(&weighted);
        }
        let weight_groups = self.group_terminals_by_weight(bundle);
        let mut profile = BundleBuildProfile::default();
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);
        let skeleton = build_prepush_bundle_skeleton(&group_dfas);
        let weighted = instantiate_weighted_prepush_bundle(&skeleton, &group_dfas);
        instantiate_weighted_prepush_compact_write_trie_nwa(&weighted)
    }

    pub(crate) fn build_prepush_reconstructed_bundle(
        &self,
        bundle: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if bundle.is_empty() {
            return empty_bundle_nwa();
        }
        if std::env::var_os("GLRMASK_PREPUSH_NO_WEIGHT_GROUP_UNION").is_some() {
            let group_dfas = bundle
                .iter()
                .filter_map(|(&terminal, weight)| {
                    if weight.is_empty() {
                        return None;
                    }
                    self.by_terminal
                        .get(&terminal)
                        .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                })
                .collect::<Vec<_>>();
            let skeleton = build_prepush_bundle_skeleton(&group_dfas);
            return instantiate_prepush_bundle_nwa(&skeleton, &group_dfas);
        }
        let weight_groups = self.group_terminals_by_weight(bundle);
        let mut profile = BundleBuildProfile::default();
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);
        let skeleton = build_prepush_bundle_skeleton(&group_dfas);
        instantiate_prepush_bundle_nwa(&skeleton, &group_dfas)
    }

    pub(crate) fn census_prepush_reconstructed_bundle(
        &self,
        bundle: &BTreeMap<TerminalID, Weight>,
    ) -> PrepushBundleCensus {
        if bundle.is_empty() {
            return PrepushBundleCensus::default();
        }
        if std::env::var_os("GLRMASK_PREPUSH_NO_WEIGHT_GROUP_UNION").is_some() {
            let group_dfas = bundle
                .iter()
                .filter_map(|(&terminal, weight)| {
                    if weight.is_empty() {
                        return None;
                    }
                    self.by_terminal
                        .get(&terminal)
                        .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                })
                .collect::<Vec<_>>();
            return census_prepush_bundle_groups(&group_dfas);
        }
        let weight_groups = self.group_terminals_by_weight(bundle);
        let mut profile = BundleBuildProfile::default();
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);
        census_prepush_bundle_groups(&group_dfas)
    }

    pub(crate) fn build_weighted_prepush_bundle(
        &self,
        bundle: &BTreeMap<TerminalID, Weight>,
    ) -> WeightedPrepushBundle {
        if bundle.is_empty() {
            return WeightedPrepushBundle { states: Vec::new() };
        }
        if std::env::var_os("GLRMASK_PREPUSH_NO_WEIGHT_GROUP_UNION").is_some() {
            let group_dfas = bundle
                .iter()
                .filter_map(|(&terminal, weight)| {
                    if weight.is_empty() {
                        return None;
                    }
                    self.by_terminal
                        .get(&terminal)
                        .map(|dfa| (weight, BundleGroupDfa::Borrowed(dfa)))
                })
                .collect::<Vec<_>>();
            let skeleton = build_prepush_bundle_skeleton(&group_dfas);
            return instantiate_weighted_prepush_bundle(&skeleton, &group_dfas);
        }
        let weight_groups = self.group_terminals_by_weight(bundle);
        let mut profile = BundleBuildProfile::default();
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);
        let skeleton = build_prepush_bundle_skeleton(&group_dfas);
        instantiate_weighted_prepush_bundle(&skeleton, &group_dfas)
    }

    pub fn build_bundle_profiled(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> (NWA, BundleBuildProfile) {
        let total_started_at = Instant::now();
        let mut profile = BundleBuildProfile {
            input_terminals: terminal_weights.len(),
            ..BundleBuildProfile::default()
        };

        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            profile.used_single_terminal_fast_path = true;
            profile.result_nwa_states = bundle.states().len();
            profile.result_nwa_transitions = NWA::num_transitions(&bundle);
            profile.total_ms = elapsed_ms(total_started_at);
            return (bundle, profile);
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        profile.weight_groups = weight_groups.len();
        let overlap_components = weight_overlap_components(&weight_groups);
        profile.overlap_components = overlap_components.len();
        profile.largest_overlap_component =
            overlap_components.first().map_or(0, |component| component.len());
        for (weight, _) in &weight_groups {
            profile.total_weight_outer_ranges += weight.outer_range_count();
            if weight.single_compact_entry_parts().is_some() {
                profile.single_entry_weights += 1;
            }
            if weight.single_tsid_shared_entry().is_some() {
                profile.single_tsid_weights += 1;
            }
        }
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);
        if std::env::var_os("GLRMASK_PROFILE_PREPUSH_BUNDLE_CENSUS").is_some() {
            let census_started_at = Instant::now();
            let census = census_prepush_bundle_groups(&group_dfas);
            profile.prepush_census_ms = elapsed_ms(census_started_at);
            profile.prepush_states = census.states;
            profile.prepush_input_transitions = census.input_transitions;
            profile.prepush_output_edges = census.output_edges;
            profile.prepush_output_sites = census.output_sites;
            profile.prepush_output_programs = census.output_programs;
            profile.prepush_core_states = census.core_states;
            profile.prepush_frontier_payloads = census.frontier_payloads;
            profile.prepush_frontier_final_payloads = census.frontier_final_payloads;
            profile.prepush_frontier_push_edges = census.frontier_push_edges;
            profile.prepush_program_sequences = census.program_sequences;
            profile.prepush_programs_multisequence = census.programs_multisequence;
            profile.prepush_max_sequences_per_program = census.max_sequences_per_program;
            profile.prepush_max_push_depth = census.max_push_depth;
        }

        // STICKY NOTE: NEVER REMOVE THIS NOTE.
        // These parser bundles must be determinized before they are converted
        // back to NWAs and stitched into the parser DWA. Negative-resolution
        // relies on this determinized boundary: if bundles remain nondeterministic
        // or merely factored, the next bundle has to combine with all unresolved
        // alternatives from the previous bundle, and adjacent bundles explode
        // combinatorially. If a large grammar cannot determinize the first bundle,
        // that is a compiler-scaling problem to solve by reducing/quotienting the
        // deterministic product, not by skipping this determinization step.
        let determinize_started_at = Instant::now();
        let (bundle_dwa, determinize_profile) = determinize_bundle_groups_profiled(&group_dfas);
        profile.determinize_bundle_ms = elapsed_ms(determinize_started_at);
        profile.determinize_pop_state_ms = determinize_profile.pop_state_ms;
        profile.determinize_alive_groups_ms = determinize_profile.alive_groups_ms;
        profile.determinize_effective_weights_ms = determinize_profile.effective_weights_ms;
        profile.determinize_final_weight_ms = determinize_profile.final_weight_ms;
        profile.determinize_collect_labels_ms = determinize_profile.collect_labels_ms;
        profile.determinize_next_state_ms = determinize_profile.next_state_ms;
        profile.determinize_edge_weight_ms = determinize_profile.edge_weight_ms;
        profile.determinize_state_lookup_ms = determinize_profile.state_lookup_ms;
        profile.determinize_add_transition_ms = determinize_profile.add_transition_ms;
        profile.determinize_states_visited = determinize_profile.states_visited;
        profile.determinize_labels_processed = determinize_profile.labels_processed;
        profile.determinize_transitions_added = determinize_profile.transitions_added;
        profile.determinize_worklist_peak = determinize_profile.worklist_peak;
        profile.determinize_cache_entries = determinize_profile.cache_entries;
        profile.determinize_edge_subset_total = determinize_profile.edge_subset_total;
        profile.determinize_edge_subset_max = determinize_profile.edge_subset_max;
        profile.determinize_edge_cache_hits = determinize_profile.edge_cache_hits;
        profile.determinize_edge_cache_hit_subset_total = determinize_profile.edge_cache_hit_subset_total;
        profile.determinize_edge_cache_misses = determinize_profile.edge_cache_misses;
        profile.determinize_edge_cache_miss_subset_total = determinize_profile.edge_cache_miss_subset_total;
        profile.result_dwa_states = bundle_dwa.states().len();
        profile.result_dwa_transitions = DWA::num_transitions(&bundle_dwa);
        if std::env::var_os("GLRMASK_VALIDATE_PREPUSH_BUNDLE").is_some() {
            use crate::automata::weighted::equivalence::find_difference;
            use crate::automata::weighted_u32::determinize::determinize as weighted_determinize;
            let prepush = build_prepush_bundle_skeleton(&group_dfas);
            let reconstructed = instantiate_prepush_bundle_nwa(&prepush, &group_dfas);
            let reconstructed = weighted_determinize(&reconstructed)
                .expect("pre-push bundle reconstruction must be finite and acyclic");
            assert!(
                find_difference(&bundle_dwa, &reconstructed)
                    .expect("pre-push bundle validation requires finite acyclic bundles")
                    .is_none(),
                "pre-push bundle transducer boundary changed weighted bundle language",
            );
        }
        for state in bundle_dwa.states() {
            let mut has_negative = false;
            let mut has_positive = false;
            for (&label, targets) in &state.transitions {
                if crate::compiler::glr::labels::is_negative_label(label) {
                    has_negative = true;
                    profile.result_negative_transitions += 1;
                } else {
                    has_positive = true;
                    profile.result_positive_transitions += 1;
                }
            }
            match (has_negative, has_positive) {
                (true, false) => profile.result_negative_only_states += 1,
                (false, true) => profile.result_positive_only_states += 1,
                (true, true) => profile.result_mixed_label_states += 1,
                (false, false) => profile.result_unlabeled_states += 1,
            }
        }

        {
            let mut seen = vec![false; bundle_dwa.states().len()];
            let mut queue = VecDeque::from([0u32]);
            while let Some(state_id) = queue.pop_front() {
                let state_index = state_id as usize;
                if seen[state_index] {
                    continue;
                }
                seen[state_index] = true;
                profile.truncated_reachable_states += 1;
                let state = &bundle_dwa.states()[state_index];
                let has_negative = state
                    .transitions
                    .keys()
                    .any(|&label| crate::compiler::glr::labels::is_negative_label(label));
                let has_positive = state
                    .transitions
                    .keys()
                    .any(|&label| !crate::compiler::glr::labels::is_negative_label(label));
                if has_negative && !has_positive {
                    profile.truncated_push_frontier_states += 1;
                    continue;
                }
                for (_, (target, _)) in &state.transitions {
                    profile.truncated_edges_traversed += 1;
                    queue.push_back(*target);
                }
            }
        }

        let minimize_started_at = Instant::now();
        profile.minimize_skipped = !minimize_template_bundles_enabled();
        let minimized = if profile.weight_groups > 1 && !profile.minimize_skipped {
            minimize(&bundle_dwa)
        } else {
            bundle_dwa
        };
        profile.minimize_ms = elapsed_ms(minimize_started_at);
        profile.result_dwa_states = minimized.states().len();
        profile.result_dwa_transitions = DWA::num_transitions(&minimized);

        let to_nwa_started_at = Instant::now();
        let nwa = dwa_to_nwa(&minimized);
        profile.dwa_to_nwa_ms = elapsed_ms(to_nwa_started_at);
        profile.result_nwa_states = nwa.states().len();
        profile.result_nwa_transitions = NWA::num_transitions(&nwa);
        profile.total_ms = elapsed_ms(total_started_at);

        (nwa, profile)
    }

    /// Assemble a weighted NWA for one bundle of (terminal, weight) entries.
    ///
    /// Pipeline: group by weight, merge each group, determinize the product,
    /// then convert back to an NWA. This determinization is correctness-critical
    /// for keeping negative-resolution local between bundle boundaries; see the
    /// sticky note in `build_bundle_profiled` above before changing it.
    /// Bundle minimization is skipped by default
    /// because parser-DWA composition reuses these fragments directly and the
    /// minimization cost dominates compile time on large schemas. Set
    /// `GLRMASK_MINIMIZE_TEMPLATE_BUNDLES=1` to restore the old behavior.
    pub fn build_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            return bundle;
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        if template_bundle_overlap_components_enabled()
            && let Some(bundle) = self.build_bundle_by_overlap_components(&weight_groups)
        {
            return bundle;
        }
        let group_dfas = self.build_group_dfas(&weight_groups);
        let bundle_dwa = determinize_bundle_groups(&group_dfas);
        let minimized = if weight_groups.len() > 1 && minimize_template_bundles_enabled() {
            minimize(&bundle_dwa)
        } else {
            bundle_dwa
        };
        dwa_to_nwa(&minimized)
    }
}

#[derive(Clone, Debug, Default)]
struct DeterminizeBundleProfile {
    pop_state_ms: f64,
    alive_groups_ms: f64,
    effective_weights_ms: f64,
    final_weight_ms: f64,
    collect_labels_ms: f64,
    next_state_ms: f64,
    edge_weight_ms: f64,
    state_lookup_ms: f64,
    add_transition_ms: f64,
    states_visited: usize,
    labels_processed: usize,
    transitions_added: usize,
    worklist_peak: usize,
    cache_entries: usize,
    edge_subset_total: usize,
    edge_subset_max: usize,
    edge_cache_hits: usize,
    edge_cache_hit_subset_total: usize,
    edge_cache_misses: usize,
    edge_cache_miss_subset_total: usize,
}



fn build_prepush_bundle_skeleton(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> PrepushBundleSkeleton {
    let n = groups.len();
    if n == 0 {
        return PrepushBundleSkeleton {
            states: Vec::new(),
            group_count: 0,
        };
    }

    let start_key = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| (group_id as u32, dfa.dfa().start_state))
        .collect::<Vec<_>>();
    let mut states = vec![PrepushBundleState {
        final_groups: SmallVec::new(),
        transitions: BTreeMap::new(),
    }];
    let mut state_map = FxHashMap::<Vec<(u32, u32)>, u32>::default();
    let mut singleton_state_map = FxHashMap::<(u32, u32), u32>::default();
    let mut worklist = VecDeque::<(u32, Vec<(u32, u32)>)>::new();
    if let [single] = start_key.as_slice() {
        singleton_state_map.insert(*single, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    worklist.push_back((0, start_key));
    let mut label_targets = LabelTargets::new();

    while let Some((state_id, product_state)) = worklist.pop_front() {
        let mut final_groups = SmallVec::<[u32; 8]>::new();
        for &(group_id, dfa_state) in &product_state {
            if groups[group_id as usize].1.dfa().states[dfa_state as usize].is_accepting {
                final_groups.push(group_id);
            }
        }
        states[state_id as usize].final_groups = final_groups;

        collect_label_targets(groups, &product_state, &mut label_targets);
        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }
            let slice = &label_targets[label_start..label_end];
            let mut edge_groups = SmallVec::<[u32; 8]>::new();
            for &(_, group_id, _) in slice {
                edge_groups.push(group_id);
            }

            let output_now = crate::compiler::glr::labels::is_negative_label(label);
            let target = if output_now {
                PrepushBundleTarget::Outputs(
                    slice
                        .iter()
                        .map(|&(_, group_id, target)| (group_id, target))
                        .collect(),
                )
            } else {
                let next_state = slice
                    .iter()
                    .map(|&(_, group_id, target)| (group_id, target))
                    .collect::<Vec<_>>();
                if product_state_has_input_labels(groups, &next_state) {
                    let target_id = if let [single] = next_state.as_slice() {
                        if let Some(&existing) = singleton_state_map.get(single) {
                            existing
                        } else {
                            let id = states.len() as u32;
                            states.push(PrepushBundleState {
                                final_groups: SmallVec::new(),
                                transitions: BTreeMap::new(),
                            });
                            singleton_state_map.insert(*single, id);
                            worklist.push_back((id, next_state));
                            id
                        }
                    } else if let Some(&existing) = state_map.get(&next_state) {
                        existing
                    } else {
                        let id = states.len() as u32;
                        states.push(PrepushBundleState {
                            final_groups: SmallVec::new(),
                            transitions: BTreeMap::new(),
                        });
                        state_map.insert(next_state.clone(), id);
                        worklist.push_back((id, next_state));
                        id
                    };
                    PrepushBundleTarget::Core(target_id)
                } else {
                    PrepushBundleTarget::Outputs(
                        slice
                            .iter()
                            .map(|&(_, group_id, target)| (group_id, target))
                            .collect(),
                    )
                }
            };

            states[state_id as usize].transitions.insert(
                label,
                PrepushBundleTransition {
                    groups: edge_groups,
                    target,
                },
            );
            label_start = label_end;
        }
    }

    PrepushBundleSkeleton {
        states,
        group_count: n,
    }
}

#[derive(Default)]
struct PrepushInstantiateDetail {
    final_weight_ms: f64,
    core_weight_ms: f64,
    output_expand_ms: f64,
    output_sequence_ms: f64,
    output_group_ms: f64,
    output_weight_ms: f64,
    transition_insert_ms: f64,
    output_sort_ms: f64,
    residual_calls: usize,
    residual_cache_hits: usize,
    residual_cache_misses: usize,
    output_program_refs: usize,
    output_sequence_refs: usize,
    output_unique_sequences: usize,
    core_weight_unions: usize,
    output_weight_unions: usize,
}
type SharedResidualSequenceCache = FxHashMap<(usize, u32), Arc<[PushSequence]>>;


fn residual_push_sequences(
    dfa: &UnweightedDfa,
    state: u32,
    memo: &mut FxHashMap<u32, Arc<[PushSequence]>>,
    shared: &mut SharedResidualSequenceCache,
    detail: &mut PrepushInstantiateDetail,
) -> Arc<[PushSequence]> {
    detail.residual_calls += 1;
    if let Some(cached) = memo.get(&state) {
        detail.residual_cache_hits += 1;
        return Arc::clone(cached);
    }
    let shared_key = (dfa as *const UnweightedDfa as usize, state);
    if let Some(cached) = shared.get(&shared_key) {
        detail.residual_cache_hits += 1;
        memo.insert(state, Arc::clone(cached));
        return Arc::clone(cached);
    }
    detail.residual_cache_misses += 1;
    let current = &dfa.states[state as usize];
    let mut out = Vec::<PushSequence>::new();
    if current.is_accepting {
        out.push(PushSequence::new());
    }
    for (&label, &target) in &current.transitions {
        assert!(
            is_negative_label(label),
            "pre-push residual program returned to input label"
        );
        let pushed = negative_to_positive_label(label) as u32;
        for suffix in residual_push_sequences(dfa, target, memo, shared, detail).iter() {
            let mut sequence = PushSequence::with_capacity(1 + suffix.len());
            sequence.push(pushed);
            sequence.extend_from_slice(suffix);
            out.push(sequence);
        }
    }
    out.sort_unstable();
    out.dedup();
    let out: Arc<[PushSequence]> = Arc::from(out.into_boxed_slice());
    memo.insert(state, Arc::clone(&out));
    shared.insert(shared_key, Arc::clone(&out));
    out
}

fn weighted_outputs_for_programs(
    programs: &[(u32, u32)],
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    sequence_memos: &mut [FxHashMap<u32, Arc<[PushSequence]>>],
    shared_sequences: &mut SharedResidualSequenceCache,
    prefix_push: Option<u32>,
    detail: &mut PrepushInstantiateDetail,
) -> Vec<PrepushOutput> {
    let sequence_started = Instant::now();
    detail.output_program_refs += programs.len();
    if let [(group, residual)] = programs {
        let sequences = residual_push_sequences(
            groups[*group as usize].1.dfa(),
            *residual,
            &mut sequence_memos[*group as usize],
            shared_sequences,
            detail,
        );
        detail.output_sequence_refs += sequences.len();
        detail.output_unique_sequences += sequences.len();
        let weight = groups[*group as usize].0;
        let outputs = sequences
            .iter()
            .map(|suffix| {
                let mut pushes =
                    PushSequence::with_capacity(prefix_push.is_some() as usize + suffix.len());
                if let Some(push) = prefix_push {
                    pushes.push(push);
                }
                pushes.extend_from_slice(suffix);
                PrepushOutput {
                    pushes,
                    weight: weight.clone(),
                }
            })
            .collect();
        detail.output_sequence_ms += elapsed_ms(sequence_started);
        return outputs;
    }
    let mut outputs = Vec::<PrepushOutput>::new();
    for &(group, residual) in programs {
        let sequences = residual_push_sequences(
            groups[group as usize].1.dfa(),
            residual,
            &mut sequence_memos[group as usize],
            shared_sequences,
            detail,
        );
        detail.output_sequence_refs += sequences.len();
        let weight = groups[group as usize].0;
        for suffix in sequences.iter() {
            let mut pushes = PushSequence::with_capacity(prefix_push.is_some() as usize + suffix.len());
            if let Some(push) = prefix_push {
                pushes.push(push);
            }
            pushes.extend_from_slice(suffix);
            outputs.push(PrepushOutput {
                pushes,
                weight: weight.clone(),
            });
        }
    }
    detail.output_sequence_ms += elapsed_ms(sequence_started);

    let group_started = Instant::now();
    outputs.sort_unstable_by(|left, right| left.pushes.cmp(&right.pushes));
    let mut merged = Vec::<PrepushOutput>::with_capacity(outputs.len());
    for output in outputs {
        if let Some(previous) = merged.last_mut()
            && previous.pushes == output.pushes
        {
            detail.output_weight_unions += 1;
            previous.weight = previous.weight.union(&output.weight);
        } else {
            merged.push(output);
        }
    }
    detail.output_unique_sequences += merged.len();
    detail.output_group_ms += elapsed_ms(group_started);
    merged
}

fn concrete_weight_for_groups(
    group_ids: &SmallVec<[u32; 8]>,
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    detail: &mut PrepushInstantiateDetail,
    cache: Option<&mut FxHashMap<SmallVec<[u32; 8]>, Weight>>,
) -> Weight {
    match group_ids.as_slice() {
        [] => Weight::empty(),
        [group] => groups[*group as usize].0.clone(),
        _ => {
            detail.core_weight_unions += 1;
            if let Some(cache) = cache {
                if let Some(weight) = cache.get(group_ids) {
                    return weight.clone();
                }
                let weight = Weight::union_all(
                    group_ids
                        .iter()
                        .map(|&group| groups[group as usize].0),
                );
                cache.insert(group_ids.clone(), weight.clone());
                weight
            } else {
                Weight::union_all(
                    group_ids
                        .iter()
                        .map(|&group| groups[group as usize].0),
                )
            }
        }
    }
}

fn instantiate_weighted_prepush_state_profiled(
    state: &PrepushBundleState,
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    sequence_memos: &mut [FxHashMap<u32, Arc<[PushSequence]>>],
    shared_sequences: &mut SharedResidualSequenceCache,
    mut group_weight_cache: Option<&mut FxHashMap<SmallVec<[u32; 8]>, Weight>>,
    detail: &mut PrepushInstantiateDetail,
) -> WeightedPrepushState {
    let final_started = Instant::now();
    let final_weight = concrete_weight_for_groups(
        &state.final_groups,
        groups,
        detail,
        group_weight_cache.as_deref_mut(),
    );
    detail.final_weight_ms += elapsed_ms(final_started);
    let mut outputs = Vec::<PrepushOutput>::new();
    let mut transitions = BTreeMap::<i32, WeightedPrepushTarget>::new();
    for (&label, transition) in &state.transitions {
        match &transition.target {
            PrepushBundleTarget::Core(target) => {
                debug_assert!(!is_negative_label(label));
                let weight_started = Instant::now();
                let weight = concrete_weight_for_groups(
                    &transition.groups,
                    groups,
                    detail,
                    group_weight_cache.as_deref_mut(),
                );
                detail.core_weight_ms += elapsed_ms(weight_started);
                if !weight.is_empty() {
                    let insert_started = Instant::now();
                    transitions.insert(
                        label,
                        WeightedPrepushTarget::Core {
                            target: *target,
                            weight,
                        },
                    );
                    detail.transition_insert_ms += elapsed_ms(insert_started);
                }
            }
            PrepushBundleTarget::Outputs(programs) => {
                let expand_started = Instant::now();
                let prefix_push =
                    is_negative_label(label).then(|| negative_to_positive_label(label) as u32);
                let weighted = weighted_outputs_for_programs(
                    programs,
                    groups,
                    sequence_memos,
                    shared_sequences,
                    prefix_push,
                    detail,
                );
                detail.output_expand_ms += elapsed_ms(expand_started);
                if prefix_push.is_some() {
                    outputs.extend(weighted);
                } else if !weighted.is_empty() {
                    let insert_started = Instant::now();
                    transitions.insert(label, WeightedPrepushTarget::Outputs(weighted));
                    detail.transition_insert_ms += elapsed_ms(insert_started);
                }
            }
        }
    }
    let sort_started = Instant::now();
    outputs.sort_by(|left, right| left.pushes.cmp(&right.pushes));
    detail.output_sort_ms += elapsed_ms(sort_started);
    WeightedPrepushState {
        final_weight,
        outputs,
        transitions,
    }
}

fn instantiate_weighted_prepush_bundle_profiled(
    skeleton: &PrepushBundleSkeleton,
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    shared_sequences: &mut SharedResidualSequenceCache,
) -> (WeightedPrepushBundle, PrepushInstantiateDetail) {
    assert_eq!(skeleton.group_count, groups.len());
    let mut detail = PrepushInstantiateDetail::default();
    let mut sequence_memos = (0..groups.len())
        .map(|_| FxHashMap::<u32, Arc<[PushSequence]>>::default())
        .collect::<Vec<_>>();
    let mut states = Vec::with_capacity(skeleton.states.len());
    for state in &skeleton.states {
        states.push(instantiate_weighted_prepush_state_profiled(
            state,
            groups,
            &mut sequence_memos,
            shared_sequences,
            None,
            &mut detail,
        ));
    }
    (WeightedPrepushBundle { states }, detail)
}

fn instantiate_weighted_prepush_bundle(
    skeleton: &PrepushBundleSkeleton,
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> WeightedPrepushBundle {
    let mut shared_sequences = SharedResidualSequenceCache::default();
    instantiate_weighted_prepush_bundle_profiled(skeleton, groups, &mut shared_sequences).0
}



#[derive(Default)]
struct WeightedWriteTrieNode {
    final_weight: Option<Weight>,
    children: BTreeMap<u32, (usize, Weight)>,
}

fn instantiate_weighted_prepush_frontier_write_trie_nwa(
    bundle: &WeightedPrepushBundle,
) -> NWA {
    if bundle.states.is_empty() {
        return empty_bundle_nwa();
    }
    let mut nwa = NWA::new(0, 0);
    for _ in 0..bundle.states.len() {
        nwa.add_state();
    }
    nwa.start_states_mut().push(0);

    let mut root_by_signature = FxHashMap::<Vec<(PushSequence, usize)>, u32>::default();
    let get_root = |outputs: &[PrepushOutput],
                    nwa: &mut NWA,
                    root_by_signature: &mut FxHashMap<Vec<(PushSequence, usize)>, u32>| {
        let mut signature = outputs
            .iter()
            .filter(|output| !output.weight.is_empty())
            .map(|output| (output.pushes.clone(), output.weight.ptr_key()))
            .collect::<Vec<_>>();
        signature.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        if let Some(&root) = root_by_signature.get(&signature) {
            return root;
        }
        let mut nodes = vec![WeightedWriteTrieNode::default()];
        for output in outputs {
            if output.weight.is_empty() {
                continue;
            }
            let mut node = 0usize;
            for &push in &output.pushes {
                let child = if let Some((child, edge_weight)) = nodes[node].children.get_mut(&push) {
                    *edge_weight = edge_weight.union(&output.weight);
                    *child
                } else {
                    let child = nodes.len();
                    nodes.push(WeightedWriteTrieNode::default());
                    nodes[node].children.insert(push, (child, output.weight.clone()));
                    child
                };
                node = child;
            }
            nodes[node].final_weight = Some(match nodes[node].final_weight.take() {
                Some(existing) => existing.union(&output.weight),
                None => output.weight.clone(),
            });
        }
        let offset = nwa.states().len() as u32;
        for _ in 0..nodes.len() {
            nwa.add_state();
        }
        for (node_id, node) in nodes.into_iter().enumerate() {
            let source = offset + node_id as u32;
            if let Some(final_weight) = node.final_weight.filter(|weight| !weight.is_empty()) {
                nwa.set_final_weight(source, final_weight);
            }
            for (push, (child, weight)) in node.children {
                if !weight.is_empty() {
                    nwa.add_transition(
                        source,
                        encode_negative_label(push),
                        offset + child as u32,
                        weight,
                    );
                }
            }
        }
        root_by_signature.insert(signature, offset);
        offset
    };

    for (state_id, state) in bundle.states.iter().enumerate() {
        let source = state_id as u32;
        if !state.final_weight.is_empty() {
            nwa.set_final_weight(source, state.final_weight.clone());
        }
        if !state.outputs.is_empty() {
            let root = get_root(&state.outputs, &mut nwa, &mut root_by_signature);
            nwa.add_epsilon(source, root, Weight::all());
        }
        for (&label, transition) in &state.transitions {
            match transition {
                WeightedPrepushTarget::Core { target, weight } => {
                    if !weight.is_empty() {
                        nwa.add_transition(source, label, *target, weight.clone());
                    }
                }
                WeightedPrepushTarget::Outputs(outputs) => {
                    if outputs.is_empty() {
                        continue;
                    }
                    let root = get_root(outputs, &mut nwa, &mut root_by_signature);
                    nwa.add_transition(source, label, root, Weight::all());
                }
            }
        }
    }
    nwa
}

fn instantiate_weighted_prepush_compact_write_trie_nwa(
    bundle: &WeightedPrepushBundle,
) -> NWA {
    if bundle.states.is_empty() {
        return empty_bundle_nwa();
    }
    let mut nwa = NWA::new(0, 0);
    for _ in 0..bundle.states.len() {
        nwa.add_state();
    }
    nwa.start_states_mut().push(0);
    let accept = nwa.add_state();
    nwa.set_final_weight(accept, Weight::all());
    let mut suffix_state = FxHashMap::<(u32, u32), u32>::default();
    let mut root_by_sequence = FxHashMap::<PushSequence, u32>::default();
    root_by_sequence.insert(PushSequence::new(), accept);

    let get_root = |pushes: &PushSequence,
                    nwa: &mut NWA,
                    suffix_state: &mut FxHashMap<(u32, u32), u32>,
                    root_by_sequence: &mut FxHashMap<PushSequence, u32>| {
        if let Some(&root) = root_by_sequence.get(pushes) {
            return root;
        }
        let mut cur = accept;
        for &push in pushes.iter().rev() {
            let key = (cur, push);
            cur = if let Some(&existing) = suffix_state.get(&key) {
                existing
            } else {
                let state = nwa.add_state();
                nwa.add_transition(
                    state,
                    encode_negative_label(push),
                    cur,
                    Weight::all(),
                );
                suffix_state.insert(key, state);
                state
            };
        }
        root_by_sequence.insert(pushes.clone(), cur);
        cur
    };

    for (state_id, state) in bundle.states.iter().enumerate() {
        let source = state_id as u32;
        if !state.final_weight.is_empty() {
            nwa.set_final_weight(source, state.final_weight.clone());
        }
        for output in &state.outputs {
            if output.weight.is_empty() {
                continue;
            }
            let root = get_root(
                &output.pushes,
                &mut nwa,
                &mut suffix_state,
                &mut root_by_sequence,
            );
            nwa.add_epsilon(source, root, output.weight.clone());
        }
        for (&label, transition) in &state.transitions {
            match transition {
                WeightedPrepushTarget::Core { target, weight } => {
                    if !weight.is_empty() {
                        nwa.add_transition(source, label, *target, weight.clone());
                    }
                }
                WeightedPrepushTarget::Outputs(outputs) => {
                    for output in outputs {
                        if output.weight.is_empty() {
                            continue;
                        }
                        let root = get_root(
                            &output.pushes,
                            &mut nwa,
                            &mut suffix_state,
                            &mut root_by_sequence,
                        );
                        nwa.add_transition(source, label, root, output.weight.clone());
                    }
                }
            }
        }
    }
    nwa
}

fn instantiate_prepush_bundle_nwa(
    skeleton: &PrepushBundleSkeleton,
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> NWA {
    assert_eq!(skeleton.group_count, groups.len());
    if skeleton.states.is_empty() {
        return empty_bundle_nwa();
    }
    let group_weights = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect::<Vec<_>>();
    let mut subset_cache = FxHashMap::<SmallVec<[u32; 8]>, Weight>::default();
    let mut union_groups = |ids: &SmallVec<[u32; 8]>| -> Weight {
        match ids.as_slice() {
            [] => Weight::empty(),
            [group] => group_weights[*group as usize].clone(),
            _ => {
                if let Some(existing) = subset_cache.get(ids) {
                    return existing.clone();
                }
                let result = Weight::union_all(
                    ids.iter().map(|&group| &group_weights[group as usize]),
                );
                subset_cache.insert(ids.clone(), result.clone());
                result
            }
        }
    };

    let mut nwa = NWA::new(0, 0);
    for _ in 0..skeleton.states.len() {
        nwa.add_state();
    }
    nwa.start_states_mut().push(0);

    // Allocate residual push-program states lazily and then fill them after all
    // core transitions have registered their roots.
    let mut residual_states = FxHashMap::<(u32, u32), u32>::default();
    let mut residual_queue = VecDeque::<(u32, u32)>::new();
    let get_residual_state = |key: (u32, u32),
                              nwa: &mut NWA,
                              residual_states: &mut FxHashMap<(u32, u32), u32>,
                              residual_queue: &mut VecDeque<(u32, u32)>| {
        if let Some(&existing) = residual_states.get(&key) {
            return existing;
        }
        let state = nwa.add_state();
        residual_states.insert(key, state);
        residual_queue.push_back(key);
        state
    };

    for (state_id, state) in skeleton.states.iter().enumerate() {
        let final_weight = union_groups(&state.final_groups);
        if !final_weight.is_empty() {
            nwa.set_final_weight(state_id as u32, final_weight);
        }
        for (&label, transition) in &state.transitions {
            match &transition.target {
                PrepushBundleTarget::Core(target) => {
                    let weight = union_groups(&transition.groups);
                    if !weight.is_empty() {
                        nwa.add_transition(state_id as u32, label, *target, weight);
                    }
                }
                PrepushBundleTarget::Outputs(outputs) => {
                    for &(group, residual) in outputs {
                        let target = get_residual_state(
                            (group, residual),
                            &mut nwa,
                            &mut residual_states,
                            &mut residual_queue,
                        );
                        nwa.add_transition(
                            state_id as u32,
                            label,
                            target,
                            group_weights[group as usize].clone(),
                        );
                    }
                }
            }
        }
    }

    while let Some((group, dfa_state)) = residual_queue.pop_front() {
        let nwa_state = residual_states[&(group, dfa_state)];
        let dfa_state_ref = &groups[group as usize].1.dfa().states[dfa_state as usize];
        let group_weight = &group_weights[group as usize];
        if dfa_state_ref.is_accepting {
            nwa.set_final_weight(nwa_state, group_weight.clone());
        }
        for (&label, &target) in &dfa_state_ref.transitions {
            assert!(
                crate::compiler::glr::labels::is_negative_label(label),
                "pre-push residual program unexpectedly returned to an input label: group={group} state={dfa_state} label={label}",
            );
            let target_state = get_residual_state(
                (group, target),
                &mut nwa,
                &mut residual_states,
                &mut residual_queue,
            );
            nwa.add_transition(nwa_state, label, target_state, group_weight.clone());
        }
    }

    nwa
}

#[derive(Default)]
pub(crate) struct PrepushBundleCensus {
    pub(crate) states: usize,
    pub(crate) input_transitions: usize,
    pub(crate) output_edges: usize,
    pub(crate) output_sites: usize,
    pub(crate) output_programs: usize,
    pub(crate) core_states: usize,
    pub(crate) residual_states: usize,
    pub(crate) frontier_payloads: usize,
    pub(crate) frontier_final_payloads: usize,
    pub(crate) frontier_push_edges: usize,
    pub(crate) program_sequences: usize,
    pub(crate) programs_multisequence: usize,
    pub(crate) max_sequences_per_program: usize,
    pub(crate) max_push_depth: usize,
}

impl PrepushBundleCensus {
    pub(crate) fn reconstructed_states(&self) -> usize {
        self.core_states.saturating_add(self.residual_states)
    }
}

/// Determinize only the stack-reading prefix of a weighted template bundle.
/// Negative labels are stack *writes*, not further input. At the first write,
/// record the residual group-DFA state as an output program and stop traversing
/// that alternative. This is an exact census of the finite-state transducer
/// boundary we would use instead of materializing push tails as input-NWA states.
fn product_state_has_input_labels(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    product_state: &[(u32, u32)],
) -> bool {
    product_state.iter().any(|&(group_id, state_id)| {
        groups[group_id as usize].1.dfa().states[state_id as usize]
            .transitions
            .keys()
            .any(|&label| !crate::compiler::glr::labels::is_negative_label(label))
    })
}

fn observe_frontier_payload(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
    product_state: &[(u32, u32)],
    census: &mut PrepushBundleCensus,
    output_programs: &mut FxHashMap<(u32, u32), ()>,
    residual_roots: &mut FxHashMap<(u32, u32), ()>,
) {
    census.frontier_payloads += 1;
    let mut accepting = false;
    for &(group_id, state_id) in product_state {
        // Reconstruction enters this residual state directly on the last READ
        // edge, before consuming any WRITE label. Count the whole negative-only
        // suffix rooted here, not only the states after the first WRITE.
        residual_roots.insert((group_id, state_id), ());
        let state = &groups[group_id as usize].1.dfa().states[state_id as usize];
        accepting |= state.is_accepting;
        for (&label, &target) in &state.transitions {
            if crate::compiler::glr::labels::is_negative_label(label) {
                census.frontier_push_edges += 1;
                output_programs.insert((group_id, target), ());
            }
        }
    }
    census.frontier_final_payloads += usize::from(accepting);
}

fn residual_push_path_stats(
    dfa: &UnweightedDfa,
    state: u32,
    memo: &mut FxHashMap<u32, (usize, usize)>,
) -> (usize, usize) {
    if let Some(&cached) = memo.get(&state) {
        return cached;
    }
    let current = &dfa.states[state as usize];
    let mut paths = usize::from(current.is_accepting);
    let mut max_depth = 0usize;
    for (&label, &target) in &current.transitions {
        assert!(
            crate::compiler::glr::labels::is_negative_label(label),
            "residual push program returned to input label while profiling",
        );
        let (child_paths, child_depth) = residual_push_path_stats(dfa, target, memo);
        paths = paths.saturating_add(child_paths);
        max_depth = max_depth.max(1 + child_depth);
    }
    memo.insert(state, (paths, max_depth));
    (paths, max_depth)
}

fn census_prepush_bundle_groups(groups: &[(&Weight, BundleGroupDfa<'_>)]) -> PrepushBundleCensus {
    let n = groups.len();
    if n == 0 {
        return PrepushBundleCensus::default();
    }
    let start_key = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| (group_id as u32, dfa.dfa().start_state))
        .collect::<Vec<_>>();
    let mut state_map = FxHashMap::<Vec<(u32, u32)>, u32>::default();
    let mut singleton_state_map = FxHashMap::<(u32, u32), u32>::default();
    let mut worklist = VecDeque::<Vec<(u32, u32)>>::new();
    if let [single] = start_key.as_slice() {
        singleton_state_map.insert(*single, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    worklist.push_back(start_key);
    let mut census = PrepushBundleCensus::default();
    let mut label_targets = LabelTargets::new();
    let mut output_programs = FxHashMap::<(u32, u32), ()>::default();
    let mut residual_roots = FxHashMap::<(u32, u32), ()>::default();

    while let Some(product_state) = worklist.pop_front() {
        census.states += 1;
        census.core_states += 1;
        collect_label_targets(groups, &product_state, &mut label_targets);
        let mut label_start = 0usize;
        let mut site_has_output = false;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }
            if crate::compiler::glr::labels::is_negative_label(label) {
                site_has_output = true;
                census.output_edges += label_end - label_start;
                for &(_, group_id, target) in &label_targets[label_start..label_end] {
                    output_programs.insert((group_id, target), ());
                    residual_roots.insert((group_id, target), ());
                }
                label_start = label_end;
                continue;
            }

            census.input_transitions += 1;
            let singleton_target = (label_end == label_start + 1).then(|| {
                let (_, group_id, target) = label_targets[label_start];
                (group_id, target)
            });
            if let Some(singleton_target) = singleton_target {
                let next_state = [singleton_target];
                if product_state_has_input_labels(groups, &next_state) {
                    if !singleton_state_map.contains_key(&singleton_target) {
                        let id = (state_map.len() + singleton_state_map.len()) as u32;
                        singleton_state_map.insert(singleton_target, id);
                        worklist.push_back(vec![singleton_target]);
                    }
                } else {
                    observe_frontier_payload(
                        groups,
                        &next_state,
                        &mut census,
                        &mut output_programs,
                        &mut residual_roots,
                    );
                }
            } else {
                let next_state = label_targets[label_start..label_end]
                    .iter()
                    .map(|&(_, group_id, target)| (group_id, target))
                    .collect::<Vec<_>>();
                if product_state_has_input_labels(groups, &next_state) {
                    if !state_map.contains_key(&next_state) {
                        let id = (state_map.len() + singleton_state_map.len()) as u32;
                        state_map.insert(next_state.clone(), id);
                        worklist.push_back(next_state);
                    }
                } else {
                    observe_frontier_payload(
                        groups,
                        &next_state,
                        &mut census,
                        &mut output_programs,
                        &mut residual_roots,
                    );
                }
            }
            label_start = label_end;
        }
        census.output_sites += usize::from(site_has_output);
    }
    census.output_programs = output_programs.len();
    let mut memos = (0..groups.len())
        .map(|_| FxHashMap::<u32, (usize, usize)>::default())
        .collect::<Vec<_>>();
    for &(group, state) in residual_roots.keys() {
        residual_push_path_stats(
            groups[group as usize].1.dfa(),
            state,
            &mut memos[group as usize],
        );
    }
    census.residual_states = memos.iter().map(FxHashMap::len).sum();
    for &(group, state) in output_programs.keys() {
        let (paths, depth) = residual_push_path_stats(
            groups[group as usize].1.dfa(),
            state,
            &mut memos[group as usize],
        );
        census.program_sequences = census.program_sequences.saturating_add(paths);
        census.programs_multisequence += usize::from(paths > 1);
        census.max_sequences_per_program = census.max_sequences_per_program.max(paths);
        census.max_push_depth = census.max_push_depth.max(depth);
    }
    census
}

fn determinize_bundle_groups_profiled(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> (DWA, DeterminizeBundleProfile) {
    use crate::automata::weighted_u32::dwa::DWA;

    let mut profile = DeterminizeBundleProfile::default();

    let n = groups.len();
    if n == 0 {
        return (DWA::new(0, 0), profile);
    }

    let group_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();
    let single_tsid_entries = group_weights
        .iter()
        .map(Weight::single_tsid_shared_entry)
        .collect::<Option<Vec<_>>>();

    let mut subset_union_cache: FxHashMap<SubsetKey, Weight> = FxHashMap::default();
    let subset_unions = template_bundle_subset_union_index_enabled().then(|| SubsetUnionIndex::new(&group_weights));

    let start_key: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| {
            (
                checked_usize_to_u32(group_id, "bundle group id"),
                dfa.dfa().start_state,
            )
        })
        .collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
    let mut singleton_state_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Vec<(u32, u32)>)> = VecDeque::new();

    if let [singleton] = start_key.as_slice() {
        singleton_state_map.insert(*singleton, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    worklist.push_back((0, start_key));
    profile.worklist_peak = worklist.len();

    let mut label_targets = LabelTargets::new();
    let key_words = n.div_ceil(64);
    let mut final_groups = SmallVec::<[usize; 8]>::new();
    let mut final_key = SubsetKey::from_elem(0, key_words);
    let mut edge_groups = SmallVec::<[usize; 8]>::new();
    let mut edge_key = SubsetKey::from_elem(0, key_words);

    while let Some((dwa_state, product_state)) = worklist.pop_front() {
        profile.states_visited += 1;
        let state_started_at = Instant::now();
        profile.pop_state_ms += elapsed_ms(state_started_at);

        let alive_started_at = Instant::now();
        let _alive_groups = product_state.len();
        profile.alive_groups_ms += elapsed_ms(alive_started_at);

        let effective_started_at = Instant::now();
        profile.effective_weights_ms += elapsed_ms(effective_started_at);

        let final_started_at = Instant::now();
        final_groups.clear();
        clear_subset_key(&mut final_key);
        for &(group_id, dfa_state) in &product_state {
            let group_id = group_id as usize;
            if groups[group_id].1.dfa().states[dfa_state as usize].is_accepting {
                final_groups.push(group_id);
                set_subset_key_bit(&mut final_key, group_id);
            }
        }
        let final_w = cached_subset_union(
            &mut subset_union_cache,
            &final_key,
            &final_groups,
            &group_weights,
            subset_unions.as_ref(),
            single_tsid_entries.as_deref(),
        );
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }
        profile.final_weight_ms += elapsed_ms(final_started_at);

        let labels_started_at = Instant::now();
        collect_label_targets(groups, &product_state, &mut label_targets);
        profile.collect_labels_ms += elapsed_ms(labels_started_at);

        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }
            profile.labels_processed += 1;

            let next_state_started_at = Instant::now();
            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            let singleton_target = (label_end == label_start + 1).then(|| {
                let (_, group_id, target) = label_targets[label_start];
                (group_id, target)
            });
            let mut next_state = singleton_target
                .is_none()
                .then(|| Vec::with_capacity(label_end - label_start));
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                let group_id = group_id as usize;
                edge_groups.push(group_id);
                set_subset_key_bit(&mut edge_key, group_id);
                if let Some(next_state) = next_state.as_mut() {
                    next_state.push((group_id as u32, target));
                }
            }
            profile.next_state_ms += elapsed_ms(next_state_started_at);

            let edge_subset_len = edge_groups.len();
            profile.edge_subset_total += edge_subset_len;
            profile.edge_subset_max = profile.edge_subset_max.max(edge_subset_len);
            if subset_union_cache.contains_key(&edge_key) {
                profile.edge_cache_hits += 1;
                profile.edge_cache_hit_subset_total += edge_subset_len;
            } else {
                profile.edge_cache_misses += 1;
                profile.edge_cache_miss_subset_total += edge_subset_len;
            }

            let edge_weight_started_at = Instant::now();
            let edge_w = cached_subset_union(
                &mut subset_union_cache,
                &edge_key,
                &edge_groups,
                &group_weights,
                subset_unions.as_ref(),
                single_tsid_entries.as_deref(),
            );
            if edge_w.is_empty() {
                profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);
                label_start = label_end;
                continue;
            }
            profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);

            let lookup_started_at = Instant::now();
            let to_dwa = if let Some(singleton_target) = singleton_target {
                if let Some(&existing) = singleton_state_map.get(&singleton_target) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    singleton_state_map.insert(singleton_target, new_id);
                    worklist.push_back((new_id, vec![singleton_target]));
                    profile.worklist_peak = profile.worklist_peak.max(worklist.len());
                    new_id
                }
            } else {
                let next_state = next_state.expect("non-singleton bundle state is populated");
                if let Some(&existing) = state_map.get(&next_state) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    state_map.insert(next_state.clone(), new_id);
                    worklist.push_back((new_id, next_state));
                    profile.worklist_peak = profile.worklist_peak.max(worklist.len());
                    new_id
                }
            };
            profile.state_lookup_ms += elapsed_ms(lookup_started_at);

            let add_transition_started_at = Instant::now();
            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
            profile.add_transition_ms += elapsed_ms(add_transition_started_at);
            profile.transitions_added += 1;
            label_start = label_end;
        }
    }

    profile.cache_entries = subset_union_cache.len();

    (dwa, profile)
}

fn determinize_bundle_groups_skeleton(
    groups: &[(&Weight, BundleGroupDfa<'_>)],
) -> BundleSkeleton {
    let n = groups.len();
    if n == 0 {
        return BundleSkeleton {
            states: Vec::new(),
            group_count: 0,
        };
    }

    let start_key: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| {
            (
                checked_usize_to_u32(group_id, "bundle group id"),
                dfa.dfa().start_state,
            )
        })
        .collect();

    let mut states = vec![BundleSkeletonState {
        final_groups: SmallVec::new(),
        transitions: BTreeMap::new(),
    }];
    let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
    let mut singleton_state_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Vec<(u32, u32)>)> = VecDeque::new();
    if let [singleton] = start_key.as_slice() {
        singleton_state_map.insert(*singleton, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    worklist.push_back((0, start_key));

    let mut label_targets = LabelTargets::new();
    while let Some((state_id, product_state)) = worklist.pop_front() {
        let mut final_groups = SmallVec::<[u32; 8]>::new();
        for &(group_id, dfa_state) in &product_state {
            if groups[group_id as usize].1.dfa().states[dfa_state as usize].is_accepting {
                final_groups.push(group_id);
            }
        }
        states[state_id as usize].final_groups = final_groups;

        collect_label_targets(groups, &product_state, &mut label_targets);
        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }

            let singleton_target = (label_end == label_start + 1).then(|| {
                let (_, group_id, target) = label_targets[label_start];
                (group_id, target)
            });
            let mut edge_groups = SmallVec::<[u32; 8]>::new();
            let mut next_state = singleton_target
                .is_none()
                .then(|| Vec::with_capacity(label_end - label_start));
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                edge_groups.push(group_id);
                if let Some(next_state) = next_state.as_mut() {
                    next_state.push((group_id, target));
                }
            }

            let target_state = if let Some(singleton_target) = singleton_target {
                if let Some(&existing) = singleton_state_map.get(&singleton_target) {
                    existing
                } else {
                    let new_id = states.len() as u32;
                    states.push(BundleSkeletonState {
                        final_groups: SmallVec::new(),
                        transitions: BTreeMap::new(),
                    });
                    singleton_state_map.insert(singleton_target, new_id);
                    worklist.push_back((new_id, vec![singleton_target]));
                    new_id
                }
            } else {
                let next_state = next_state.expect("non-singleton bundle state is populated");
                if let Some(&existing) = state_map.get(&next_state) {
                    existing
                } else {
                    let new_id = states.len() as u32;
                    states.push(BundleSkeletonState {
                        final_groups: SmallVec::new(),
                        transitions: BTreeMap::new(),
                    });
                    state_map.insert(next_state.clone(), new_id);
                    worklist.push_back((new_id, next_state));
                    new_id
                }
            };
            states[state_id as usize]
                .transitions
                .insert(label, (target_state, edge_groups));
            label_start = label_end;
        }
    }

    BundleSkeleton {
        states,
        group_count: n,
    }
}

fn instantiate_bundle_skeleton_dwa(
    skeleton: &BundleSkeleton,
    group_weights: &[Weight],
) -> DWA {
    assert_eq!(skeleton.group_count, group_weights.len());
    if skeleton.states.is_empty() {
        return DWA::new(0, 0);
    }

    let mut dwa = DWA::new(0, 0);
    for _ in 1..skeleton.states.len() {
        dwa.add_state();
    }
    let mut subset_cache = FxHashMap::<SmallVec<[u32; 8]>, Weight>::default();
    let mut union_subset = |groups: &SmallVec<[u32; 8]>| -> Weight {
        match groups.as_slice() {
            [] => Weight::empty(),
            [group] => group_weights[*group as usize].clone(),
            _ => {
                if let Some(weight) = subset_cache.get(groups) {
                    return weight.clone();
                }
                let weight = Weight::union_all(
                    groups
                        .iter()
                        .map(|&group| &group_weights[group as usize]),
                );
                subset_cache.insert(groups.clone(), weight.clone());
                weight
            }
        }
    };

    for (state_id, state) in skeleton.states.iter().enumerate() {
        let final_weight = union_subset(&state.final_groups);
        if !final_weight.is_empty() {
            dwa.set_final_weight(state_id as u32, final_weight);
        }
        for (&label, (target, groups)) in &state.transitions {
            let edge_weight = union_subset(groups);
            if !edge_weight.is_empty() {
                dwa.add_transition(state_id as u32, label, *target, edge_weight);
            }
        }
    }
    dwa
}

fn determinize_bundle_groups(groups: &[(&Weight, BundleGroupDfa<'_>)]) -> DWA {
    use crate::automata::weighted_u32::dwa::DWA;

    let n = groups.len();
    if n == 0 {
        return DWA::new(0, 0);
    }

    let group_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();
    let single_tsid_entries = group_weights
        .iter()
        .map(Weight::single_tsid_shared_entry)
        .collect::<Option<Vec<_>>>();

    let mut subset_union_cache: FxHashMap<SubsetKey, Weight> = FxHashMap::default();
    let subset_unions = template_bundle_subset_union_index_enabled().then(|| SubsetUnionIndex::new(&group_weights));

    let start_key: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| {
            (
                checked_usize_to_u32(group_id, "bundle group id"),
                dfa.dfa().start_state,
            )
        })
        .collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
    let mut singleton_state_map: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Vec<(u32, u32)>)> = VecDeque::new();

    if let [singleton] = start_key.as_slice() {
        singleton_state_map.insert(*singleton, 0);
    } else {
        state_map.insert(start_key.clone(), 0);
    }
    worklist.push_back((0, start_key));

    let mut label_targets = LabelTargets::new();
    let key_words = n.div_ceil(64);
    let mut final_groups = SmallVec::<[usize; 8]>::new();
    let mut final_key = SubsetKey::from_elem(0, key_words);
    let mut edge_groups = SmallVec::<[usize; 8]>::new();
    let mut edge_key = SubsetKey::from_elem(0, key_words);

    while let Some((dwa_state, product_state)) = worklist.pop_front() {

        final_groups.clear();
        clear_subset_key(&mut final_key);
        for &(group_id, dfa_state) in &product_state {
            let group_id = group_id as usize;
            if groups[group_id].1.dfa().states[dfa_state as usize].is_accepting {
                final_groups.push(group_id);
                set_subset_key_bit(&mut final_key, group_id);
            }
        }
        let final_w = cached_subset_union(
            &mut subset_union_cache,
            &final_key,
            &final_groups,
            &group_weights,
            subset_unions.as_ref(),
            single_tsid_entries.as_deref(),
        );
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }

        collect_label_targets(groups, &product_state, &mut label_targets);

        let mut label_start = 0usize;
        while label_start < label_targets.len() {
            let label = label_targets[label_start].0;
            let mut label_end = label_start + 1;
            while label_end < label_targets.len() && label_targets[label_end].0 == label {
                label_end += 1;
            }

            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            let singleton_target = (label_end == label_start + 1).then(|| {
                let (_, group_id, target) = label_targets[label_start];
                (group_id, target)
            });
            let mut next_state = singleton_target
                .is_none()
                .then(|| Vec::with_capacity(label_end - label_start));
            for &(_, group_id, target) in &label_targets[label_start..label_end] {
                let group_id = group_id as usize;
                edge_groups.push(group_id);
                set_subset_key_bit(&mut edge_key, group_id);
                if let Some(next_state) = next_state.as_mut() {
                    next_state.push((group_id as u32, target));
                }
            }

            let edge_w = cached_subset_union(
                &mut subset_union_cache,
                &edge_key,
                &edge_groups,
                &group_weights,
                subset_unions.as_ref(),
                single_tsid_entries.as_deref(),
            );
            if edge_w.is_empty() {
                label_start = label_end;
                continue;
            }

            let to_dwa = if let Some(singleton_target) = singleton_target {
                if let Some(&existing) = singleton_state_map.get(&singleton_target) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    singleton_state_map.insert(singleton_target, new_id);
                    worklist.push_back((new_id, vec![singleton_target]));
                    new_id
                }
            } else {
                let next_state = next_state.expect("non-singleton bundle state is populated");
                if let Some(&existing) = state_map.get(&next_state) {
                    existing
                } else {
                    let new_id = dwa.add_state();
                    state_map.insert(next_state.clone(), new_id);
                    worklist.push_back((new_id, next_state));
                    new_id
                }
            };

            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
            label_start = label_end;
        }
    }

    dwa
}

/// Union multiple unweighted DFAs into one DFA via NFA union + determinize + minimize.
fn union_unweighted_dfas<'a>(dfas: impl Iterator<Item = &'a UnweightedDfa>) -> UnweightedDfa {
    let mut nfa = UnweightedNfa::new_empty();
    let shared_start = nfa.add_state();
    nfa.start_states.push(shared_start);

    for dfa in dfas {
        if dfa.states.is_empty() {
            continue;
        }
        let offset = checked_usize_to_u32(nfa.states.len(), "bundle-union NFA offset");
        for _ in &dfa.states {
            nfa.add_state();
        }
        // Epsilon from shared start to this DFA's start.
        nfa.add_epsilon(
            shared_start,
            checked_u32_add(offset, dfa.start_state, "bundle-union start state"),
        );
        for (state_id, state) in dfa.states.iter().enumerate() {
            let from = checked_u32_add(
                offset,
                checked_usize_to_u32(state_id, "bundle-union DFA state id"),
                "bundle-union source state",
            );
            if state.is_accepting {
                nfa.set_accepting(from);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(
                    from,
                    label,
                    checked_u32_add(offset, target, "bundle-union target state"),
                );
            }
        }
    }

    let det = unweighted_determinize(&nfa);
    unweighted_minimize(&det)
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let states = dwa
        .states()
        .iter()
        .map(|state| NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| (label, vec![(*target, weight.clone())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();

    NWA::from_parts(
        states,
        vec![dwa.start_state()],
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use range_set_blaze::RangeSetBlaze;

    use super::*;
    use crate::automata::weighted::equivalence::find_difference;
    use crate::automata::weighted_u32::determinize::determinize as weighted_determinize;

    fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
    }

    fn eval_bundle_product(
        groups: &[(&Weight, BundleGroupDfa<'_>)],
        word: &[i32],
    ) -> Weight {
        let mut product_state = groups
            .iter()
            .enumerate()
            .map(|(group_id, (_, dfa))| (group_id, dfa.dfa().start_state))
            .collect::<Vec<_>>();
        let mut accumulated = Weight::all();

        for &label in word {
            let mut next_state = Vec::new();
            let mut edge_weights = SmallVec::<[&Weight; 4]>::new();
            for &(group_id, dfa_state) in &product_state {
                let dfa = groups[group_id].1.dfa();
                let Some(&target) = dfa.states[dfa_state as usize].transitions.get(&label) else {
                    continue;
                };
                next_state.push((group_id, target));
                edge_weights.push(groups[group_id].0);
            }
            let edge_weight = Weight::union_all(edge_weights);
            accumulated = accumulated.intersection(&edge_weight);
            if accumulated.is_empty() {
                return accumulated;
            }
            product_state = next_state;
        }

        let final_weight = Weight::union_all(product_state.iter().filter_map(
            |&(group_id, dfa_state)| {
                groups[group_id].1.dfa().states[dfa_state as usize]
                    .is_accepting
                    .then_some(groups[group_id].0)
            },
        ));
        accumulated.intersection(&final_weight)
    }

    fn visit_words(
        alphabet: &[i32],
        remaining: usize,
        word: &mut Vec<i32>,
        visit: &mut impl FnMut(&[i32]),
    ) {
        visit(word);
        if remaining == 0 {
            return;
        }
        for &label in alphabet {
            word.push(label);
            visit_words(alphabet, remaining - 1, word, visit);
            word.pop();
        }
    }

    #[test]
    fn collect_label_targets_is_sorted() {
        let mut first = UnweightedDfa::new();
        first.add_transition(0, 2, 20);
        first.add_transition(0, 1, 10);
        let mut second = UnweightedDfa::new();
        second.add_transition(0, 3, 31);
        second.add_transition(0, 1, 11);
        let weight = Weight::all();
        let groups = vec![
            (&weight, BundleGroupDfa::Owned(first)),
            (&weight, BundleGroupDfa::Owned(second)),
        ];
        let state = vec![(0, 0), (1, 0)];
        let mut targets = LabelTargets::new();

        collect_label_targets(&groups, &state, &mut targets);

        assert_eq!(targets.as_slice(), &[(1, 0, 10), (1, 1, 11), (2, 0, 20), (3, 1, 31)]);
    }

    #[test]
    fn singleton_bundle_state_interner_preserves_weighted_product() {
        let mut first = UnweightedDfa::new();
        let first_after_shared = first.add_state();
        let first_after_singleton = first.add_state();
        first.add_transition(0, 1, first_after_shared);
        first.add_transition(first_after_shared, 2, first_after_singleton);
        first.add_transition(first_after_shared, 5, first_after_shared);
        first.set_accepting(first_after_shared, true);
        first.set_accepting(first_after_singleton, true);

        let mut second = UnweightedDfa::new();
        let second_after_shared = second.add_state();
        let second_after_singleton = second.add_state();
        second.add_transition(0, 1, second_after_shared);
        second.add_transition(second_after_shared, 3, second_after_singleton);
        second.add_transition(second_after_shared, 5, second_after_shared);
        second.set_accepting(second_after_singleton, true);

        let mut third = UnweightedDfa::new();
        let third_after_singleton = third.add_state();
        third.add_transition(0, 4, third_after_singleton);
        third.set_accepting(third_after_singleton, true);

        let first_weight = weight(0..=15);
        let second_weight = weight(8..=23);
        let third_weight = weight(20..=31);
        let groups = vec![
            (&first_weight, BundleGroupDfa::Owned(first)),
            (&second_weight, BundleGroupDfa::Owned(second)),
            (&third_weight, BundleGroupDfa::Owned(third)),
        ];

        let optimized = determinize_bundle_groups(&groups);
        let (profiled, _) = determinize_bundle_groups_profiled(&groups);
        let mut word = Vec::new();
        visit_words(&[1, 2, 3, 4, 5], 4, &mut word, &mut |word| {
            let expected = eval_bundle_product(&groups, word);
            assert_eq!(optimized.eval_word(word), expected, "optimized word={word:?}");
            assert_eq!(profiled.eval_word(word), expected, "profiled word={word:?}");
        });
    }

    #[test]
    fn overlap_component_bundle_union_matches_full_product() {
        fn literal(labels: &[i32]) -> UnweightedDfa {
            let mut dfa = UnweightedDfa::new();
            let mut state = dfa.start_state;
            for &label in labels {
                let next = dfa.add_state();
                dfa.add_transition(state, label, next);
                state = next;
            }
            dfa.set_accepting(state, true);
            dfa
        }

        let mut templates = Templates::default();
        templates.by_terminal.insert(0, literal(&[1, 2]));
        templates.by_terminal.insert(1, literal(&[1, 3]));
        templates.by_terminal.insert(2, literal(&[1, 4]));
        templates.by_terminal.insert(3, literal(&[5, 6]));

        // The first two groups overlap and therefore must be determinized
        // together. The last two are support-disjoint from that component and
        // from each other, so they may remain separate weighted alternatives.
        let mut terminal_weights = BTreeMap::new();
        terminal_weights.insert(0, weight(0..=15));
        terminal_weights.insert(1, weight(8..=23));
        terminal_weights.insert(2, weight(32..=47));
        terminal_weights.insert(3, weight(64..=79));

        let weight_groups = templates.group_terminals_by_weight(&terminal_weights);
        let components = weight_overlap_components(&weight_groups);
        assert_eq!(components.len(), 3);
        assert_eq!(components[0].len(), 2);

        let full_groups = templates.build_group_dfas(&weight_groups);
        let full = determinize_bundle_groups(&full_groups);
        let split_nwa = templates
            .build_bundle_by_overlap_components(&weight_groups)
            .expect("three overlap components should take the split path");
        let split = weighted_determinize(&split_nwa).expect("split bundle determinizes");

        assert_eq!(
            find_difference(&full, &split).expect("acyclic bundle comparison"),
            None,
            "splitting pairwise support-disjoint weight components changed the weighted bundle language",
        );
    }

    #[test]
    fn prepush_census_exactly_predicts_reconstructed_state_count() {
        fn read_then_write(read: i32, writes: &[i32]) -> UnweightedDfa {
            let mut dfa = UnweightedDfa::new();
            let mut state = dfa.add_state();
            dfa.add_transition(dfa.start_state, read, state);
            for &write in writes {
                assert!(crate::compiler::glr::labels::is_negative_label(write));
                let next = dfa.add_state();
                dfa.add_transition(state, write, next);
                state = next;
            }
            dfa.set_accepting(state, true);
            dfa
        }

        fn write_only(writes: &[i32]) -> UnweightedDfa {
            let mut dfa = UnweightedDfa::new();
            let mut state = dfa.start_state;
            for &write in writes {
                assert!(crate::compiler::glr::labels::is_negative_label(write));
                let next = dfa.add_state();
                dfa.add_transition(state, write, next);
                state = next;
            }
            dfa.set_accepting(state, true);
            dfa
        }

        let mut templates = Templates::default();
        templates.by_terminal.insert(0, read_then_write(1, &[-10, -11]));
        templates.by_terminal.insert(1, read_then_write(1, &[-10]));
        templates.by_terminal.insert(2, write_only(&[-20, -21]));

        let mut bundle = BTreeMap::new();
        bundle.insert(0, weight(0..=15));
        bundle.insert(1, weight(16..=31));
        bundle.insert(2, weight(32..=47));

        let census = templates.census_prepush_reconstructed_bundle(&bundle);
        let reconstructed = templates.build_prepush_reconstructed_bundle(&bundle);

        assert!(census.core_states > 0);
        assert!(census.residual_states > 0);
        assert_eq!(
            census.reconstructed_states(),
            reconstructed.states().len(),
            "the cheap census must count every READ-core and residual WRITE state materialized by reconstruction",
        );
    }

    #[test]
    fn reusable_bundle_skeleton_matches_direct_determinization_after_reweighting() {
        fn literal(labels: &[i32]) -> UnweightedDfa {
            let mut dfa = UnweightedDfa::new();
            let mut state = dfa.start_state;
            for &label in labels {
                let next = dfa.add_state();
                dfa.add_transition(state, label, next);
                state = next;
            }
            dfa.set_accepting(state, true);
            dfa
        }

        let mut templates = Templates::default();
        templates.by_terminal.insert(0, literal(&[1, 2]));
        templates.by_terminal.insert(1, literal(&[1, 3]));
        templates.by_terminal.insert(2, literal(&[1, 4]));
        templates.by_terminal.insert(3, literal(&[5, 6]));

        let mut representative = BTreeMap::new();
        representative.insert(0, weight(0..=15));
        representative.insert(1, weight(0..=15));
        representative.insert(2, weight(16..=31));
        representative.insert(3, weight(32..=47));
        let skeleton = templates
            .build_bundle_skeleton(&representative)
            .expect("multi-terminal representative has a skeleton");

        // Preserve exactly the same equality partition of terminals, but
        // replace every concrete weight. The deterministic product topology is
        // unchanged; only the subset unions decorating its states and edges
        // should change.
        let mut reweighted = BTreeMap::new();
        reweighted.insert(0, weight(8..=23));
        reweighted.insert(1, weight(8..=23));
        reweighted.insert(2, weight(20..=39));
        reweighted.insert(3, weight(64..=95));

        assert_eq!(
            templates.bundle_topology_signature(&representative),
            templates.bundle_topology_signature(&reweighted),
        );
        let direct = templates.build_bundle(&reweighted);
        let reused = templates.instantiate_bundle_skeleton(&reweighted, &skeleton);
        let direct = weighted_determinize(&direct).expect("direct bundle determinizes");
        let reused = weighted_determinize(&reused).expect("reused bundle determinizes");
        assert_eq!(
            find_difference(&direct, &reused).expect("acyclic bundle comparison"),
            None,
            "reusing deterministic bundle topology changed the weighted language",
        );
    }
}
