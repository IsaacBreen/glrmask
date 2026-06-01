//! Template bundle assembly into a weighted NWA.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;
use rustc_hash::FxHashMap;
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
use crate::ds::weight::{SharedTokenSet, Weight};

type SubsetKey = SmallVec<[u64; 4]>;
const SUBSET_BLOCK_BITS: usize = 8;
const SUBSET_BLOCK_MASK: u64 = (1u64 << SUBSET_BLOCK_BITS) - 1;

pub(crate) type BundleShapeKey = Vec<Vec<u32>>;

#[derive(Clone, Debug)]
pub(crate) struct BundleSkeletonTransition {
    pub(crate) label: i32,
    pub(crate) target: u32,
    pub(crate) group_ids: Box<[u32]>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BundleSkeletonState {
    pub(crate) final_group_ids: Box<[u32]>,
    pub(crate) transitions: Box<[BundleSkeletonTransition]>,
}

#[derive(Clone, Debug)]
pub(crate) struct BundleSkeleton {
    pub(crate) states: Box<[BundleSkeletonState]>,
    pub(crate) start_state: u32,
    pub(crate) group_lex_order: Box<[u32]>,
}

#[derive(Clone)]
struct ShapeGroup {
    lex_index: u32,
    template_ids: Vec<u32>,
    dfa: UnweightedDfa,
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
pub(crate) struct BundleBuildProfile {
    pub(crate) input_terminals: usize,
    pub(crate) nonempty_terminals: usize,
    pub(crate) weight_groups: usize,
    pub(crate) single_entry_weights: usize,
    pub(crate) single_tsid_weights: usize,
    pub(crate) total_weight_outer_ranges: usize,
    pub(crate) singleton_groups: usize,
    pub(crate) multi_terminal_groups: usize,
    pub(crate) largest_weight_group: usize,
    pub(crate) build_group_dfas_ms: f64,
    pub(crate) union_groups_ms: f64,
    pub(crate) slowest_group_terminals: usize,
    pub(crate) slowest_group_dfa_states: usize,
    pub(crate) slowest_group_dfa_transitions: usize,
    pub(crate) slowest_group_ms: f64,
    pub(crate) determinize_bundle_ms: f64,
    pub(crate) determinize_pop_state_ms: f64,
    pub(crate) determinize_alive_groups_ms: f64,
    pub(crate) determinize_effective_weights_ms: f64,
    pub(crate) determinize_final_weight_ms: f64,
    pub(crate) determinize_collect_labels_ms: f64,
    pub(crate) determinize_next_state_ms: f64,
    pub(crate) determinize_edge_weight_ms: f64,
    pub(crate) determinize_state_lookup_ms: f64,
    pub(crate) determinize_add_transition_ms: f64,
    pub(crate) determinize_states_visited: usize,
    pub(crate) determinize_labels_processed: usize,
    pub(crate) determinize_transitions_added: usize,
    pub(crate) determinize_worklist_peak: usize,
    pub(crate) determinize_cache_entries: usize,
    pub(crate) determinize_edge_subset_total: usize,
    pub(crate) determinize_edge_subset_max: usize,
    pub(crate) determinize_edge_cache_hits: usize,
    pub(crate) determinize_edge_cache_hit_subset_total: usize,
    pub(crate) determinize_edge_cache_misses: usize,
    pub(crate) determinize_edge_cache_miss_subset_total: usize,
    pub(crate) minimize_ms: f64,
    pub(crate) dwa_to_nwa_ms: f64,
    pub(crate) result_dwa_states: usize,
    pub(crate) result_dwa_transitions: usize,
    pub(crate) result_nwa_states: usize,
    pub(crate) result_nwa_transitions: usize,
    pub(crate) total_ms: f64,
    pub(crate) used_single_terminal_fast_path: bool,
    pub(crate) minimize_skipped: bool,
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn minimize_template_bundles_enabled() -> bool {
    std::env::var("GLRMASK_MINIMIZE_TEMPLATE_BUNDLES")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn count_unweighted_dfa_transitions(dfa: &UnweightedDfa) -> usize {
    dfa.states.iter().map(|state| state.transitions.len()).sum()
}

fn dfa_order_key(dfa: &UnweightedDfa) -> Vec<i32> {
    let mut key = Vec::new();
    let mut stack = vec![dfa.start_state];
    let mut seen = vec![false; dfa.states.len()];

    while let Some(state_id) = stack.pop() {
        let index = state_id as usize;
        if index >= seen.len() || seen[index] {
            continue;
        }
        seen[index] = true;

        let state = &dfa.states[index];
        key.push(if state.is_accepting { i32::MIN } else { i32::MIN + 1 });
        for (&label, &target) in state.transitions.iter().rev() {
            key.push(label);
            stack.push(target);
        }
    }

    key
}

fn count_weighted_dwa_transitions(dwa: &DWA) -> usize {
    dwa.states().iter().map(|state| state.transitions.len()).sum()
}

fn count_nwa_transitions(nwa: &NWA) -> usize {
    nwa.states()
        .iter()
        .map(|state| state.transitions.values().map(|targets| targets.len()).sum::<usize>() + state.epsilons.len())
        .sum()
}

fn union_weights_by_group_ids(group_weights: &[Weight], group_ids: &[u32]) -> Weight {
    match group_ids {
        [] => Weight::empty(),
        [group_id] => group_weights[*group_id as usize].clone(),
        _ => Weight::union_all(
            group_ids
                .iter()
                .map(|group_id| &group_weights[*group_id as usize]),
        ),
    }
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
    ) -> HashMap<&'a Weight, Vec<TerminalID>> {
        let mut weight_groups: HashMap<&Weight, Vec<TerminalID>> = HashMap::new();
        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() || !self.by_terminal.contains_key(&terminal) {
                continue;
            }
            weight_groups.entry(weight).or_default().push(terminal);
        }
        weight_groups
    }

    fn build_group_dfas_profiled<'a>(
        &'a self,
        weight_groups: &'a HashMap<&'a Weight, Vec<TerminalID>>,
        profile: &mut BundleBuildProfile,
    ) -> Vec<(&'a Weight, UnweightedDfa)> {
        let build_started_at = Instant::now();
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            profile.nonempty_terminals += terminals.len();
            profile.largest_weight_group = profile.largest_weight_group.max(terminals.len());
            if terminals.len() == 1 {
                profile.singleton_groups += 1;
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, template.clone()));
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
                profile.slowest_group_dfa_transitions = count_unweighted_dfa_transitions(&merged);
            }

            group_dfas.push((*weight, merged));
        }
        group_dfas.sort_by_cached_key(|(_, dfa)| dfa_order_key(dfa));
        profile.build_group_dfas_ms = elapsed_ms(build_started_at);
        group_dfas
    }

    fn build_group_dfas<'a>(
        &'a self,
        weight_groups: &'a HashMap<&'a Weight, Vec<TerminalID>>,
    ) -> Vec<(&'a Weight, UnweightedDfa)> {
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            if terminals.len() == 1 {
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, template.clone()));
                }
            } else {
                let merged = union_unweighted_dfas(
                    terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
                );
                group_dfas.push((*weight, merged));
            }
        }
        group_dfas.sort_by_cached_key(|(_, dfa)| dfa_order_key(dfa));
        group_dfas
    }

    fn build_shape_groups_profiled(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        profile: &mut BundleBuildProfile,
    ) -> Vec<ShapeGroup> {
        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        profile.weight_groups = weight_groups.len();
        for weight in weight_groups.keys() {
            profile.total_weight_outer_ranges += weight.outer_range_count();
            if weight.single_compact_entry_parts().is_some() {
                profile.single_entry_weights += 1;
            }
            if weight.single_tsid_shared_entry().is_some() {
                profile.single_tsid_weights += 1;
            }
        }

        let build_started_at = Instant::now();
        let mut lex_groups = Vec::with_capacity(weight_groups.len());
        for terminals in weight_groups.values() {
            profile.nonempty_terminals += terminals.len();
            profile.largest_weight_group = profile.largest_weight_group.max(terminals.len());

            let mut template_ids = terminals
                .iter()
                .filter_map(|terminal| self.template_id_by_terminal.get(*terminal as usize).copied())
                .filter(|template_id| *template_id != u32::MAX)
                .collect::<Vec<_>>();
            template_ids.sort_unstable();

            let dfa = if terminals.len() == 1 {
                profile.singleton_groups += 1;
                self.by_terminal
                    .get(&terminals[0])
                    .cloned()
                    .unwrap_or_default()
            } else {
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
                    profile.slowest_group_dfa_transitions = count_unweighted_dfa_transitions(&merged);
                }
                merged
            };

            lex_groups.push((template_ids, dfa));
        }
        profile.build_group_dfas_ms = elapsed_ms(build_started_at);

        let mut order: Vec<usize> = (0..lex_groups.len()).collect();
        order.sort_by_cached_key(|&index| dfa_order_key(&lex_groups[index].1));

        order
            .into_iter()
            .map(|index| {
                let (template_ids, dfa) = lex_groups[index].clone();
                ShapeGroup {
                    lex_index: index as u32,
                    template_ids,
                    dfa,
                }
            })
            .collect()
    }

    fn build_shape_groups(&self, terminal_weights: &BTreeMap<TerminalID, Weight>) -> Vec<ShapeGroup> {
        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        let mut lex_groups = Vec::with_capacity(weight_groups.len());
        for terminals in weight_groups.values() {
            let mut template_ids = terminals
                .iter()
                .filter_map(|terminal| self.template_id_by_terminal.get(*terminal as usize).copied())
                .filter(|template_id| *template_id != u32::MAX)
                .collect::<Vec<_>>();
            template_ids.sort_unstable();

            let dfa = if terminals.len() == 1 {
                self.by_terminal
                    .get(&terminals[0])
                    .cloned()
                    .unwrap_or_default()
            } else {
                union_unweighted_dfas(
                    terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
                )
            };

            lex_groups.push((template_ids, dfa));
        }

        let mut order: Vec<usize> = (0..lex_groups.len()).collect();
        order.sort_by_cached_key(|&index| dfa_order_key(&lex_groups[index].1));
        order
            .into_iter()
            .map(|index| {
                let (template_ids, dfa) = lex_groups[index].clone();
                ShapeGroup {
                    lex_index: index as u32,
                    template_ids,
                    dfa,
                }
            })
            .collect()
    }

    pub(crate) fn bundle_shape_key(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> BundleShapeKey {
        let mut key: BundleShapeKey = self
            .build_shape_groups(terminal_weights)
            .into_iter()
            .map(|group| group.template_ids)
            .collect();
        key.sort_unstable();
        key
    }

    fn build_bundle_skeleton_from_groups_profiled(
        groups: &[ShapeGroup],
    ) -> (BundleSkeleton, DeterminizeBundleProfile) {
        let mut profile = DeterminizeBundleProfile::default();
        if groups.is_empty() {
            return (
                BundleSkeleton {
                    states: vec![BundleSkeletonState::default()].into_boxed_slice(),
                    start_state: 0,
                    group_lex_order: Box::default(),
                },
                profile,
            );
        }

        let start_key: Vec<(u32, u32)> = groups
            .iter()
            .enumerate()
            .map(|(group_id, group)| (group_id as u32, group.dfa.start_state))
            .collect();

        let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
        let mut worklist: VecDeque<Vec<(u32, u32)>> = VecDeque::new();
        let mut states = vec![BundleSkeletonState::default()];
        state_map.insert(start_key.clone(), 0);
        worklist.push_back(start_key);
        profile.worklist_peak = 1;

        let mut label_targets: BTreeMap<i32, Vec<(u32, u32)>> = BTreeMap::new();

        while let Some(product_state) = worklist.pop_front() {
            profile.states_visited += 1;
            let state_started_at = Instant::now();
            let state_id = state_map[&product_state] as usize;
            profile.pop_state_ms += elapsed_ms(state_started_at);

            let alive_started_at = Instant::now();
            let _alive_groups = product_state.len();
            profile.alive_groups_ms += elapsed_ms(alive_started_at);

            let final_started_at = Instant::now();
            let mut final_groups = Vec::new();
            for &(group_id, dfa_state) in &product_state {
                if groups[group_id as usize].dfa.states[dfa_state as usize].is_accepting {
                    final_groups.push(group_id);
                }
            }
            states[state_id].final_group_ids = final_groups.into_boxed_slice();
            profile.final_weight_ms += elapsed_ms(final_started_at);

            let labels_started_at = Instant::now();
            label_targets.clear();
            for &(group_id, dfa_state) in &product_state {
                for (&label, &target) in &groups[group_id as usize].dfa.states[dfa_state as usize].transitions {
                    label_targets.entry(label).or_default().push((group_id, target));
                }
            }
            profile.collect_labels_ms += elapsed_ms(labels_started_at);
            profile.labels_processed += label_targets.len();

            let mut transitions = Vec::with_capacity(label_targets.len());
            for (&label, next_state) in &label_targets {
                let next_state_started_at = Instant::now();
                let mut group_ids = next_state.iter().map(|(group_id, _)| *group_id).collect::<Vec<_>>();
                group_ids.sort_unstable();
                profile.next_state_ms += elapsed_ms(next_state_started_at);

                let lookup_started_at = Instant::now();
                let target = if let Some(&existing) = state_map.get(next_state) {
                    existing
                } else {
                    let key = next_state.clone();
                    let new_id = states.len() as u32;
                    state_map.insert(key.clone(), new_id);
                    worklist.push_back(key);
                    profile.worklist_peak = profile.worklist_peak.max(worklist.len());
                    states.push(BundleSkeletonState::default());
                    new_id
                };
                profile.state_lookup_ms += elapsed_ms(lookup_started_at);

                let add_transition_started_at = Instant::now();
                transitions.push(BundleSkeletonTransition {
                    label,
                    target,
                    group_ids: group_ids.into_boxed_slice(),
                });
                profile.add_transition_ms += elapsed_ms(add_transition_started_at);
                profile.transitions_added += 1;
            }
            states[state_id].transitions = transitions.into_boxed_slice();
        }
        profile.cache_entries = state_map.len();

        (
            BundleSkeleton {
                states: states.into_boxed_slice(),
                start_state: 0,
                group_lex_order: groups.iter().map(|group| group.lex_index).collect(),
            },
            profile,
        )
    }

    pub(crate) fn build_bundle_skeleton_profiled(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> (BundleShapeKey, BundleSkeleton, BundleBuildProfile) {
        let total_started_at = Instant::now();
        let mut profile = BundleBuildProfile {
            input_terminals: terminal_weights.len(),
            ..BundleBuildProfile::default()
        };

        let groups = self.build_shape_groups_profiled(terminal_weights, &mut profile);
        let mut shape_key: BundleShapeKey = groups.iter().map(|group| group.template_ids.clone()).collect();
        shape_key.sort_unstable();

        let determinize_started_at = Instant::now();
        let (skeleton, determinize_profile) = Self::build_bundle_skeleton_from_groups_profiled(&groups);
        profile.determinize_bundle_ms = elapsed_ms(determinize_started_at);
        profile.determinize_pop_state_ms = determinize_profile.pop_state_ms;
        profile.determinize_alive_groups_ms = determinize_profile.alive_groups_ms;
        profile.determinize_final_weight_ms = determinize_profile.final_weight_ms;
        profile.determinize_collect_labels_ms = determinize_profile.collect_labels_ms;
        profile.determinize_next_state_ms = determinize_profile.next_state_ms;
        profile.determinize_state_lookup_ms = determinize_profile.state_lookup_ms;
        profile.determinize_add_transition_ms = determinize_profile.add_transition_ms;
        profile.determinize_states_visited = determinize_profile.states_visited;
        profile.determinize_labels_processed = determinize_profile.labels_processed;
        profile.determinize_transitions_added = determinize_profile.transitions_added;
        profile.determinize_worklist_peak = determinize_profile.worklist_peak;
        profile.determinize_cache_entries = determinize_profile.cache_entries;

        (shape_key, skeleton, profile)
    }

    fn ordered_group_weights_from_shape(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        skeleton: &BundleSkeleton,
    ) -> Vec<Weight> {
        let mut by_lex_group = self
            .build_shape_groups(terminal_weights)
            .into_iter()
            .map(|group| {
                let mut lex_weight = Weight::empty();
                if let Some(weight) = terminal_weights
                    .iter()
                    .find_map(|(terminal, weight)| {
                        let template_id = self.template_id_by_terminal.get(*terminal as usize).copied()?;
                        (template_id != u32::MAX && group.template_ids.binary_search(&template_id).is_ok())
                            .then(|| weight.clone())
                    })
                {
                    lex_weight = weight;
                }
                (group.lex_index as usize, lex_weight)
            })
            .collect::<Vec<_>>();
        by_lex_group.sort_by_key(|(lex_index, _)| *lex_index);
        let lex_weights: Vec<Weight> = by_lex_group.into_iter().map(|(_, weight)| weight).collect();

        skeleton
            .group_lex_order
            .iter()
            .map(|lex_index| lex_weights[*lex_index as usize].clone())
            .collect()
    }

    pub(crate) fn instantiate_bundle_from_skeleton_profiled(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
        skeleton: &BundleSkeleton,
    ) -> (NWA, BundleBuildProfile) {
        let total_started_at = Instant::now();
        let mut profile = BundleBuildProfile {
            input_terminals: terminal_weights.len(),
            ..BundleBuildProfile::default()
        };
        let group_weights = self.ordered_group_weights_from_shape(terminal_weights, skeleton);
        profile.weight_groups = group_weights.len();
        for weight in &group_weights {
            profile.total_weight_outer_ranges += weight.outer_range_count();
            if weight.single_compact_entry_parts().is_some() {
                profile.single_entry_weights += 1;
            }
            if weight.single_tsid_shared_entry().is_some() {
                profile.single_tsid_weights += 1;
            }
        }

        let mut dwa = DWA::new(0, 0);
        while dwa.states().len() < skeleton.states.len() {
            dwa.add_state();
        }
        dwa.set_start_state(skeleton.start_state);

        for (state_id, state) in skeleton.states.iter().enumerate() {
            let final_weight = union_weights_by_group_ids(&group_weights, &state.final_group_ids);
            if !final_weight.is_empty() {
                dwa.set_final_weight(state_id as u32, final_weight);
            }
            for transition in state.transitions.iter() {
                let edge_weight = union_weights_by_group_ids(&group_weights, &transition.group_ids);
                if edge_weight.is_empty() {
                    continue;
                }
                dwa.add_transition(state_id as u32, transition.label, transition.target, edge_weight);
            }
        }
        profile.result_dwa_states = dwa.states().len();
        profile.result_dwa_transitions = count_weighted_dwa_transitions(&dwa);

        let minimize_started_at = Instant::now();
        profile.minimize_skipped = !minimize_template_bundles_enabled();
        let minimized = if profile.weight_groups > 1 && !profile.minimize_skipped {
            minimize(&dwa)
        } else {
            dwa
        };
        profile.minimize_ms = elapsed_ms(minimize_started_at);
        profile.result_dwa_states = minimized.states().len();
        profile.result_dwa_transitions = count_weighted_dwa_transitions(&minimized);

        let to_nwa_started_at = Instant::now();
        let nwa = dwa_to_nwa(&minimized);
        profile.dwa_to_nwa_ms = elapsed_ms(to_nwa_started_at);
        profile.result_nwa_states = nwa.states().len();
        profile.result_nwa_transitions = count_nwa_transitions(&nwa);
        profile.total_ms = elapsed_ms(total_started_at);
        (nwa, profile)
    }

    pub(crate) fn build_bundle_profiled(
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
            profile.result_nwa_transitions = count_nwa_transitions(&bundle);
            profile.total_ms = elapsed_ms(total_started_at);
            return (bundle, profile);
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        profile.weight_groups = weight_groups.len();
        for weight in weight_groups.keys() {
            profile.total_weight_outer_ranges += weight.outer_range_count();
            if weight.single_compact_entry_parts().is_some() {
                profile.single_entry_weights += 1;
            }
            if weight.single_tsid_shared_entry().is_some() {
                profile.single_tsid_weights += 1;
            }
        }
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);

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
        profile.result_dwa_transitions = count_weighted_dwa_transitions(&bundle_dwa);

        let minimize_started_at = Instant::now();
        profile.minimize_skipped = !minimize_template_bundles_enabled();
        let minimized = if profile.weight_groups > 1 && !profile.minimize_skipped {
            minimize(&bundle_dwa)
        } else {
            bundle_dwa
        };
        profile.minimize_ms = elapsed_ms(minimize_started_at);
        profile.result_dwa_states = minimized.states().len();
        profile.result_dwa_transitions = count_weighted_dwa_transitions(&minimized);

        let to_nwa_started_at = Instant::now();
        let nwa = dwa_to_nwa(&minimized);
        profile.dwa_to_nwa_ms = elapsed_ms(to_nwa_started_at);
        profile.result_nwa_states = nwa.states().len();
        profile.result_nwa_transitions = count_nwa_transitions(&nwa);
        profile.total_ms = elapsed_ms(total_started_at);

        (nwa, profile)
    }

    /// Assemble a weighted NWA for one bundle of (terminal, weight) entries.
    ///
    /// Pipeline: group by weight, merge each group, determinize the product,
    /// then convert back to an NWA. Bundle minimization is skipped by default
    /// because parser-DWA composition reuses these fragments directly and the
    /// minimization cost dominates compile time on large schemas. Set
    /// `GLRMASK_MINIMIZE_TEMPLATE_BUNDLES=1` to restore the old behavior.
    pub(crate) fn build_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            return bundle;
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
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

fn determinize_bundle_groups_profiled(
    groups: &[(&Weight, UnweightedDfa)],
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
    let subset_unions = (n >= 32).then(|| SubsetUnionIndex::new(&group_weights));

    let start_key: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| (group_id as u32, dfa.start_state))
        .collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<(u32, u32)>> = VecDeque::new();

    state_map.insert(start_key.clone(), 0);
    worklist.push_back(start_key.clone());
    profile.worklist_peak = worklist.len();

    let mut label_targets: BTreeMap<i32, Vec<(u32, u32)>> = BTreeMap::new();
    let key_words = n.div_ceil(64);
    let mut final_groups = SmallVec::<[usize; 8]>::new();
    let mut final_key = SubsetKey::from_elem(0, key_words);
    let mut edge_groups = SmallVec::<[usize; 8]>::new();
    let mut edge_key = SubsetKey::from_elem(0, key_words);

    while let Some(product_state) = worklist.pop_front() {
        profile.states_visited += 1;
        let state_started_at = Instant::now();
        let dwa_state = state_map[&product_state];
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
            if groups[group_id].1.states[dfa_state as usize].is_accepting {
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
        label_targets.clear();
        for &(group_id, dfa_state) in &product_state {
            for (&label, &target) in &groups[group_id as usize].1.states[dfa_state as usize]
                .transitions
            {
                label_targets
                    .entry(label)
                    .or_default()
                    .push((group_id, target));
            }
        }
        profile.collect_labels_ms += elapsed_ms(labels_started_at);
        profile.labels_processed += label_targets.len();

        for (&label, next_state) in &label_targets {
            let next_state_started_at = Instant::now();
            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            for &(group_id, _) in next_state {
                let group_id = group_id as usize;
                edge_groups.push(group_id);
                set_subset_key_bit(&mut edge_key, group_id);
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
                continue;
            }
            profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);

            let lookup_started_at = Instant::now();
            let to_dwa = if let Some(&existing) = state_map.get(next_state) {
                existing
            } else {
                let key = next_state.clone();
                let new_id = dwa.add_state();
                state_map.insert(key.clone(), new_id);
                worklist.push_back(key);
                profile.worklist_peak = profile.worklist_peak.max(worklist.len());
                new_id
            };
            profile.state_lookup_ms += elapsed_ms(lookup_started_at);

            let add_transition_started_at = Instant::now();
            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
            profile.add_transition_ms += elapsed_ms(add_transition_started_at);
            profile.transitions_added += 1;
        }
    }

    profile.cache_entries = subset_union_cache.len();

    (dwa, profile)
}

fn determinize_bundle_groups(groups: &[(&Weight, UnweightedDfa)]) -> DWA {
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
    let subset_unions = (n >= 32).then(|| SubsetUnionIndex::new(&group_weights));

    let start_key: Vec<(u32, u32)> = groups
        .iter()
        .enumerate()
        .map(|(group_id, (_, dfa))| (group_id as u32, dfa.start_state))
        .collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<(u32, u32)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<(u32, u32)>> = VecDeque::new();

    state_map.insert(start_key.clone(), 0);
    worklist.push_back(start_key);

    let mut label_targets: BTreeMap<i32, Vec<(u32, u32)>> = BTreeMap::new();
    let key_words = n.div_ceil(64);
    let mut final_groups = SmallVec::<[usize; 8]>::new();
    let mut final_key = SubsetKey::from_elem(0, key_words);
    let mut edge_groups = SmallVec::<[usize; 8]>::new();
    let mut edge_key = SubsetKey::from_elem(0, key_words);

    while let Some(product_state) = worklist.pop_front() {
        let dwa_state = state_map[&product_state];

        final_groups.clear();
        clear_subset_key(&mut final_key);
        for &(group_id, dfa_state) in &product_state {
            let group_id = group_id as usize;
            if groups[group_id].1.states[dfa_state as usize].is_accepting {
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

        label_targets.clear();
        for &(group_id, dfa_state) in &product_state {
            for (&label, &target) in &groups[group_id as usize].1.states[dfa_state as usize]
                .transitions
            {
                label_targets
                    .entry(label)
                    .or_default()
                    .push((group_id, target));
            }
        }

        for (&label, next_state) in &label_targets {
            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            for &(group_id, _) in next_state {
                let group_id = group_id as usize;
                edge_groups.push(group_id);
                set_subset_key_bit(&mut edge_key, group_id);
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
                continue;
            }

            let to_dwa = if let Some(&existing) = state_map.get(next_state) {
                existing
            } else {
                let key = next_state.clone();
                let new_id = dwa.add_state();
                state_map.insert(key.clone(), new_id);
                worklist.push_back(key);
                new_id
            };

            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
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
        let offset = nfa.states.len() as u32;
        for _ in &dfa.states {
            nfa.add_state();
        }
        // Epsilon from shared start to this DFA's start.
        nfa.add_epsilon(shared_start, offset + dfa.start_state);
        for (state_id, state) in dfa.states.iter().enumerate() {
            let from = offset + state_id as u32;
            if state.is_accepting {
                nfa.set_accepting(from);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(from, label, offset + target);
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
