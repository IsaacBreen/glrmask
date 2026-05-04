//! Template bundle assembly into a weighted NWA.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::Instant;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::automata::unweighted_u32::determinize::determinize as unweighted_determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as unweighted_minimize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_fast;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::ds::weight::Weight;

type SubsetKey = SmallVec<[u64; 4]>;
const SUBSET_BLOCK_BITS: usize = 8;

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
    block_unions: Option<&[Box<[Weight]>]>,
) -> Weight {
    match subset {
        [] => return Weight::empty(),
        [index] => return group_weights[*index].clone(),
        _ => {}
    }

    if let Some(existing) = cache.get(subset_key) {
        return existing.clone();
    }

    let result = if subset.len() >= SUBSET_BLOCK_BITS {
        if let Some(block_unions) = block_unions {
            subset_union_from_blocks(subset_key, block_unions)
        } else {
            Weight::union_all(subset.iter().map(|&index| &group_weights[index]))
        }
    } else {
        Weight::union_all(subset.iter().map(|&index| &group_weights[index]))
    };
    cache.insert(subset_key.clone(), result.clone());
    result
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

fn subset_union_from_blocks(subset_key: &SubsetKey, block_unions: &[Box<[Weight]>]) -> Weight {
    let mut parts = SmallVec::<[&Weight; 8]>::new();
    for (block_index, block_table) in block_unions.iter().enumerate() {
        let bit_offset = block_index * SUBSET_BLOCK_BITS;
        let word_index = bit_offset / 64;
        let bit_index = bit_offset % 64;
        let mask = ((subset_key[word_index] >> bit_index) & 0xff) as usize;
        if mask != 0 {
            parts.push(&block_table[mask]);
        }
    }
    Weight::union_all(parts)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BundleBuildProfile {
    pub(crate) input_terminals: usize,
    pub(crate) nonempty_terminals: usize,
    pub(crate) weight_groups: usize,
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
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn count_unweighted_dfa_transitions(dfa: &UnweightedDfa) -> usize {
    dfa.states.iter().map(|state| state.transitions.len()).sum()
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
                continue;
            }

            let merged = union_unweighted_dfas(
                terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
            );
            group_dfas.push((*weight, merged));
        }
        group_dfas
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
        profile.build_group_dfas_ms = elapsed_ms(build_started_at);
        group_dfas
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
        let minimized = if profile.weight_groups > 1 {
            minimize_fast(&bundle_dwa)
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
    /// optionally minimize it, then convert back to an NWA.
    pub(crate) fn build_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            return bundle;
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        let num_groups = weight_groups.len();
        let group_dfas = self.build_group_dfas(&weight_groups);
        let bundle_dwa = determinize_bundle_groups(&group_dfas);

        let minimized = if num_groups > 1 {
            minimize_fast(&bundle_dwa)
        } else {
            bundle_dwa
        };

        dwa_to_nwa(&minimized)
    }
}

/// Specialized weighted determinize for bundles.
fn determinize_bundle_groups(groups: &[(&Weight, UnweightedDfa)]) -> DWA {
    use crate::automata::weighted_u32::dwa::DWA;

    let n = groups.len();
    if n == 0 {
        return DWA::new(0, 0);
    }

    const DEAD: u32 = u32::MAX;

    let group_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();

    let mut subset_union_cache: FxHashMap<SubsetKey, Weight> = FxHashMap::default();
    let block_unions = (n >= 32).then(|| build_subset_block_unions(&group_weights));

    let start_key: Vec<u32> = groups.iter().map(|(_, dfa)| dfa.start_state).collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    state_map.insert(start_key.clone(), 0);
    worklist.push_back(start_key.clone());

    let mut all_labels: BTreeSet<i32> = BTreeSet::new();
    let mut next_state: Vec<u32> = vec![DEAD; n];
    let key_words = n.div_ceil(64);
    let mut final_groups = SmallVec::<[usize; 8]>::new();
    let mut final_key = SubsetKey::from_elem(0, key_words);
    let mut edge_groups = SmallVec::<[usize; 8]>::new();
    let mut edge_key = SubsetKey::from_elem(0, key_words);

    while let Some(product_state) = worklist.pop_front() {
        let dwa_state = state_map[&product_state];

        let alive_groups: Vec<usize> = (0..n)
            .filter(|&i| product_state[i] != DEAD)
            .collect();

        final_groups.clear();
        clear_subset_key(&mut final_key);
        for &i in &alive_groups {
            if groups[i].1.states[product_state[i] as usize].is_accepting {
                final_groups.push(i);
                set_subset_key_bit(&mut final_key, i);
            }
        }
        let final_w = cached_subset_union(
            &mut subset_union_cache,
            &final_key,
            &final_groups,
            &group_weights,
            block_unions.as_deref(),
        );
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }

        all_labels.clear();
        for &i in &alive_groups {
            for &label in groups[i].1.states[product_state[i] as usize]
                .transitions
                .keys()
            {
                all_labels.insert(label);
            }
        }

        for &label in &all_labels {
            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            for i in 0..n {
                let target = if product_state[i] == DEAD {
                    DEAD
                } else if let Some(&target) = groups[i]
                    .1
                    .states[product_state[i] as usize]
                    .transitions
                    .get(&label)
                {
                    target
                } else {
                    DEAD
                };
                next_state[i] = target;
                if target != DEAD {
                    edge_groups.push(i);
                    set_subset_key_bit(&mut edge_key, i);
                }
            }

            let edge_w = cached_subset_union(
                &mut subset_union_cache,
                &edge_key,
                &edge_groups,
                &group_weights,
                block_unions.as_deref(),
            );
            if edge_w.is_empty() {
                continue;
            }

            let to_dwa = if let Some(&existing) = state_map.get(&*next_state) {
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

    const DEAD: u32 = u32::MAX;

    let group_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();

    let mut subset_union_cache: FxHashMap<SubsetKey, Weight> = FxHashMap::default();
    let block_unions = (n >= 32).then(|| build_subset_block_unions(&group_weights));

    let start_key: Vec<u32> = groups.iter().map(|(_, dfa)| dfa.start_state).collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    state_map.insert(start_key.clone(), 0);
    worklist.push_back(start_key.clone());
    profile.worklist_peak = worklist.len();

    let mut all_labels: BTreeSet<i32> = BTreeSet::new();
    let mut next_state: Vec<u32> = vec![DEAD; n];
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
        let alive_groups: Vec<usize> = (0..n)
            .filter(|&i| product_state[i] != DEAD)
            .collect();
        profile.alive_groups_ms += elapsed_ms(alive_started_at);

        let effective_started_at = Instant::now();
        profile.effective_weights_ms += elapsed_ms(effective_started_at);

        let final_started_at = Instant::now();
        final_groups.clear();
        clear_subset_key(&mut final_key);
        for &i in &alive_groups {
            if groups[i].1.states[product_state[i] as usize].is_accepting {
                final_groups.push(i);
                set_subset_key_bit(&mut final_key, i);
            }
        }
        let final_w = cached_subset_union(
            &mut subset_union_cache,
            &final_key,
            &final_groups,
            &group_weights,
            block_unions.as_deref(),
        );
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }
        profile.final_weight_ms += elapsed_ms(final_started_at);

        let labels_started_at = Instant::now();
        all_labels.clear();
        for &i in &alive_groups {
            for &label in groups[i].1.states[product_state[i] as usize]
                .transitions
                .keys()
            {
                all_labels.insert(label);
            }
        }
        profile.collect_labels_ms += elapsed_ms(labels_started_at);
        profile.labels_processed += all_labels.len();

        for &label in &all_labels {
            let next_state_started_at = Instant::now();
            edge_groups.clear();
            clear_subset_key(&mut edge_key);
            for i in 0..n {
                let target = if product_state[i] == DEAD {
                    DEAD
                } else if let Some(&target) = groups[i]
                    .1
                    .states[product_state[i] as usize]
                    .transitions
                    .get(&label)
                {
                    target
                } else {
                    DEAD
                };
                next_state[i] = target;
                if target != DEAD {
                    edge_groups.push(i);
                    set_subset_key_bit(&mut edge_key, i);
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
                block_unions.as_deref(),
            );
            if edge_w.is_empty() {
                profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);
                continue;
            }
            profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);

            let lookup_started_at = Instant::now();
            let to_dwa = if let Some(&existing) = state_map.get(&*next_state) {
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
