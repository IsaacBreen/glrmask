use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use profiler_macro::{time_it, timeit};



use crate::glr::parser::{ExpectElse, GLRParser};
use crate::glr::table::{NonTerminalID, StateID as ParserStateID};
use crate::precompute4::characterize::{compute_all_characterizations, TerminalCharacterization};
use crate::precompute4::resolve_negatives::{
    apply_cancellations, apply_finality_fixpoint, remove_negative_transitions,
    is_negative_symbol,
    apply_cancellations_range,
    apply_finality_fixpoint_range, remove_negative_transitions_range,
    // Note: remove_redundant_default_transitions is only run in a global pass,
    // not per-range here, since it requires a global pass over all states.
};
use crate::precompute4::template_dfa::{build_ignore_terminal_dwa, build_template_dwas};
use crate::dwa_i32::{
    common::Label, DeterminizeAndMinimizeProfile, DwaOptimizeConfig, DWA, NWA, NWABody,
    NWAStateID, NWAStates, StateID, Weight,
};
use crate::dfa_u8::TokenizerStateID;
use crate::types::TerminalID;
use crate::datastructures::abstract_weight::{BackendChoice, override_backend, restore_backend};

fn validate_nwa_weight_dims(nwa: &NWA, expected_num_tsids: usize) {
    let mut mismatches = 0usize;
    let mut total = 0usize;
    let mut rangeset_weights = 0usize;
    let mut examples: Vec<String> = Vec::new();

    let mut check_weight = |w: &Weight| {
        total += 1;
    };

    for state in &nwa.states.0 {
        if let Some(w) = &state.final_weight {
            check_weight(w);
        }
        for (_, w) in &state.epsilons {
            check_weight(w);
        }
        for targets in state.transitions.values() {
            for (_, w) in targets {
                check_weight(w);
            }
        }
    }

    if mismatches > 0 {
        panic!(
            "Parser NWA weight dims mismatch: expected_num_tsids={}, mismatches={}, total_weights={}, rangeset_weights={}, examples={:?}",
            expected_num_tsids,
            mismatches,
            total,
            rangeset_weights,
            examples,
        );
    }

    crate::debug!(
        4,
        "Parser NWA weight dims OK: expected_num_tsids={}, total_weights={}, rangeset_weights={}",
        expected_num_tsids,
        total,
        rangeset_weights,
    );
}

pub use crate::precompute4::template_dfa::FullDWABuildError;

/// The Parser DWA - the final precomputed artifact used for get_mask queries.
/// This is a deterministic weighted automaton where weights are sparse bitvectors
/// over LLM token equivalence classes.
pub type ParserDWA = DWA;

/// Type alias for backward compatibility
#[deprecated(since = "0.3.0", note = "Use ParserDWA instead")]
pub type Precomputed4 = DWA;


#[derive(Default)]
struct GroupedCharacterization {
    initial_shifts: BTreeMap<(ParserStateID, ParserStateID), BTreeSet<TerminalID>>,
    initial_reduces: BTreeMap<(ParserStateID, usize, NonTerminalID), BTreeSet<TerminalID>>,
    per_nt: BTreeMap<NonTerminalID, GroupedNTCharacterization>,
}

#[derive(Default)]
struct GroupedNTCharacterization {
    escape_shifts: BTreeMap<(ParserStateID, ParserStateID, ParserStateID), BTreeSet<TerminalID>>,
    reveal_and_rereduces: BTreeMap<(ParserStateID, usize, NonTerminalID), BTreeSet<TerminalID>>,
}

impl GroupedCharacterization {
    fn from_terminals(
        chars: &BTreeMap<TerminalID, TerminalCharacterization>,
        used_terms: &BTreeSet<TerminalID>,
    ) -> Self {
        let mut grouped = GroupedCharacterization::default();
        for (term, tc) in chars {
            if !used_terms.contains(term) {
                continue;
            }
            for &(state, shift_state) in &tc.initial_shifts {
                grouped
                    .initial_shifts
                    .entry((state, shift_state))
                    .or_default()
                    .insert(*term);
            }
            for &(state, len, nt) in &tc.initial_reduces {
                grouped
                    .initial_reduces
                    .entry((state, len, nt))
                    .or_default()
                    .insert(*term);
            }
            for (nt, rc) in &tc.reduce_characterizations {
                let nt_group = grouped.per_nt.entry(*nt).or_default();
                for &(revealed, goto, shift) in &rc.reveal_goto_shift_escapes {
                    nt_group
                        .escape_shifts
                        .entry((revealed, goto, shift))
                        .or_default()
                        .insert(*term);
                }
                for &(revealed, remaining_len, target_nt) in &rc.reveal_and_rereduces {
                    nt_group
                        .reveal_and_rereduces
                        .entry((revealed, remaining_len, target_nt))
                        .or_default()
                        .insert(*term);
                }
            }
        }
        grouped
    }
}

struct Pass2Profile {
    process_total_us: AtomicU64,
    process_count: AtomicU64,
    template_count: AtomicU64,
    process_other_us: AtomicU64,
    canonicalize_us: AtomicU64,
    cache_lookup_us: AtomicU64,
    cache_insert_us: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    dynamic_derive_us: AtomicU64,
    instantiate_us: AtomicU64,
    apply_cancellations_us: AtomicU64,
    apply_finality_us: AtomicU64,
    remove_negative_us: AtomicU64,
    union_us: AtomicU64,
    tsid_collect_us: AtomicU64,
    final_collect_us: AtomicU64,
}

impl Pass2Profile {
    fn new() -> Self {
        Self {
            process_total_us: AtomicU64::new(0),
            process_count: AtomicU64::new(0),
            template_count: AtomicU64::new(0),
            process_other_us: AtomicU64::new(0),
            canonicalize_us: AtomicU64::new(0),
            cache_lookup_us: AtomicU64::new(0),
            cache_insert_us: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            dynamic_derive_us: AtomicU64::new(0),
            instantiate_us: AtomicU64::new(0),
            apply_cancellations_us: AtomicU64::new(0),
            apply_finality_us: AtomicU64::new(0),
            remove_negative_us: AtomicU64::new(0),
            union_us: AtomicU64::new(0),
            tsid_collect_us: AtomicU64::new(0),
            final_collect_us: AtomicU64::new(0),
        }
    }

    fn log(&self) {
        let process_count = self.process_count.load(Ordering::Relaxed);
        if process_count == 0 {
            return;
        }
        let template_count = self.template_count.load(Ordering::Relaxed);
        let process_total_us = self.process_total_us.load(Ordering::Relaxed);
        let process_other_us = self.process_other_us.load(Ordering::Relaxed);
        let canonicalize_us = self.canonicalize_us.load(Ordering::Relaxed);
        let cache_lookup_us = self.cache_lookup_us.load(Ordering::Relaxed);
        let cache_insert_us = self.cache_insert_us.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let dynamic_derive_us = self.dynamic_derive_us.load(Ordering::Relaxed);
        let instantiate_us = self.instantiate_us.load(Ordering::Relaxed);
        let apply_cancellations_us = self.apply_cancellations_us.load(Ordering::Relaxed);
        let apply_finality_us = self.apply_finality_us.load(Ordering::Relaxed);
        let remove_negative_us = self.remove_negative_us.load(Ordering::Relaxed);
        let union_us = self.union_us.load(Ordering::Relaxed);
        let tsid_collect_us = self.tsid_collect_us.load(Ordering::Relaxed);
        let final_collect_us = self.final_collect_us.load(Ordering::Relaxed);

        let cache_total = cache_hits + cache_misses;
        let cache_hit_rate = if cache_total == 0 {
            0.0
        } else {
            (cache_hits as f64) * 100.0 / (cache_total as f64)
        };

        crate::debug!(
            4,
            "Pass2 profile: process_total={:?} ({} calls), templates={}, canonicalize={:?}, cache_lookup={:?}, cache_insert={:?}, cache_hits={}, cache_misses={}, cache_hit_rate={:.2}%, dynamic_derive={:?}, instantiate={:?}, cancellations={:?}, finality_fixpoint={:?}, remove_negative={:?}, union={:?}, tsid_collect={:?}, final_collect={:?}, other={:?}",
            std::time::Duration::from_micros(process_total_us),
            process_count,
            template_count,
            std::time::Duration::from_micros(canonicalize_us),
            std::time::Duration::from_micros(cache_lookup_us),
            std::time::Duration::from_micros(cache_insert_us),
            cache_hits,
            cache_misses,
            cache_hit_rate,
            std::time::Duration::from_micros(dynamic_derive_us),
            std::time::Duration::from_micros(instantiate_us),
            std::time::Duration::from_micros(apply_cancellations_us),
            std::time::Duration::from_micros(apply_finality_us),
            std::time::Duration::from_micros(remove_negative_us),
            std::time::Duration::from_micros(union_us),
            std::time::Duration::from_micros(tsid_collect_us),
            std::time::Duration::from_micros(final_collect_us),
            std::time::Duration::from_micros(process_other_us),
        );
    }
}

struct NwaSpecialMapProfile {
    process_us: AtomicU64,
    step_us: AtomicU64,
    merge_us: AtomicU64,
    process_calls: AtomicU64,
    step_calls: AtomicU64,
    merge_calls: AtomicU64,
}

impl NwaSpecialMapProfile {
    fn new() -> Self {
        Self {
            process_us: AtomicU64::new(0),
            step_us: AtomicU64::new(0),
            merge_us: AtomicU64::new(0),
            process_calls: AtomicU64::new(0),
            step_calls: AtomicU64::new(0),
            merge_calls: AtomicU64::new(0),
        }
    }

    fn log(&self, label: &str) {
        let process_us = self.process_us.load(Ordering::Relaxed);
        let step_us = self.step_us.load(Ordering::Relaxed);
        let merge_us = self.merge_us.load(Ordering::Relaxed);
        let process_calls = self.process_calls.load(Ordering::Relaxed);
        let step_calls = self.step_calls.load(Ordering::Relaxed);
        let merge_calls = self.merge_calls.load(Ordering::Relaxed);
        crate::timing!(
            "TIMING: nwa_special_map::{} process={:?} ({} calls), step={:?} ({} calls), merge={:?} ({} calls)",
            label,
            std::time::Duration::from_micros(process_us),
            process_calls,
            std::time::Duration::from_micros(step_us),
            step_calls,
            std::time::Duration::from_micros(merge_us),
            merge_calls,
        );
    }
}

struct WeightBackendOverride {
    previous: Option<BackendChoice>,
}

impl WeightBackendOverride {
    fn new(backend: &str) -> Self {
        let choice = match backend {
            "rangeset" | "rsb" => BackendChoice::RangeSet,
            "rangemap" | "map" => BackendChoice::RangeMap,
            _ => BackendChoice::Factorized,
        };
        let previous = override_backend(choice);
        Self { previous }
    }

    fn from_choice(choice: BackendChoice) -> Self {
        let previous = override_backend(choice);
        Self { previous }
    }
}

impl Drop for WeightBackendOverride {
    fn drop(&mut self) {
        restore_backend(self.previous);
    }
}

pub struct NwaTraversalData {
    pub comp_id: Vec<usize>,
    pub sccs: Vec<Vec<usize>>,
    pub topo: Vec<usize>,
}

impl NWA {
    pub fn compute_traversal_data(&self) -> NwaTraversalData {
        let (sccs, comp_id) = compute_sccs(self);
        let scc_count = sccs.len();
        let mut scc_adj = vec![HashSet::new(); scc_count];
        let mut indeg = vec![0; scc_count];

        for u in 0..self.states.len() {
            let u_scc = comp_id[u];
            let state = &self.states[u];
            let mut neighbors = Vec::new();
            for targets in state.transitions.values() {
                for (v, _) in targets { neighbors.push(*v); }
            }
            for (v, _) in &state.epsilons { neighbors.push(*v); }

            for v in neighbors {
                let v_scc = comp_id[v];
                if u_scc != v_scc && !scc_adj[u_scc].contains(&v_scc) {
                    scc_adj[u_scc].insert(v_scc);
                    indeg[v_scc] += 1;
                }
            }
        }

        let mut topo = Vec::with_capacity(scc_count);
        let mut q = VecDeque::new();
        for i in 0..scc_count { if indeg[i] == 0 { q.push_back(i); } }

        while let Some(u) = q.pop_front() {
            topo.push(u);
            for &v in &scc_adj[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 { q.push_back(v); }
            }
        }
        NwaTraversalData { comp_id, sccs, topo }
    }
}

fn compute_sccs(nwa: &NWA) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = nwa.states.len();
    let mut adj = vec![Vec::new(); n];
    let mut rev_adj = vec![Vec::new(); n];

    for (u, state) in nwa.states.0.iter().enumerate() {
        let mut neighbors = Vec::new();
        for targets in state.transitions.values() {
            for (v, _) in targets { neighbors.push(*v); }
        }
        for (v, _) in &state.epsilons { neighbors.push(*v); }

        for v in neighbors {
            adj[u].push(v);
            rev_adj[v].push(u);
        }
    }

    let mut order = Vec::new();
    let mut visited = vec![false; n];
    for i in 0..n {
        if !visited[i] {
            let mut stack = vec![(i, false)];
            while let Some((u, processed)) = stack.pop() {
                if processed { order.push(u); } else {
                    if visited[u] { continue; }
                    visited[u] = true;
                    stack.push((u, true));
                    for &v in &adj[u] { if !visited[v] { stack.push((v, false)); } }
                }
            }
        }
    }

    let mut comp_id = vec![usize::MAX; n];
    let mut sccs = Vec::new();
    let mut current_scc_id = 0;

    for &u in order.iter().rev() {
        if comp_id[u] == usize::MAX {
            let mut component = Vec::new();
            let mut stack = vec![u];
            comp_id[u] = current_scc_id;
            while let Some(curr) = stack.pop() {
                component.push(curr);
                for &v in &rev_adj[curr] {
                    if comp_id[v] == usize::MAX {
                        comp_id[v] = current_scc_id;
                        stack.push(v);
                    }
                }
            }
            sccs.push(component);
            current_scc_id += 1;
        }
    }
    (sccs, comp_id)
}

pub fn nwa_special_map<V, U, I>(
    nwa: &NWA,
    traversal_data: &NwaTraversalData,
    initial_values: Vec<(StateID, V)>,
    mut step: impl FnMut(&U, Option<Label>, &[(StateID, Weight)]) -> I,
    mut merge: impl FnMut(&mut V, V) -> bool,
    mut process: impl FnMut(StateID, V) -> Option<U>,
    profile: Option<&NwaSpecialMapProfile>,
) where
    V: Clone,
    I: IntoIterator<Item = (StateID, V)>,
{
    let mut values: FxHashMap<StateID, V> = FxHashMap::default();
    let mut stopped_nodes: FxHashSet<StateID> = FxHashSet::default();

    for (state, v) in initial_values {
        match values.entry(state) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if let Some(profile) = profile {
                    let start = Instant::now();
                    merge(entry.get_mut(), v);
                    profile
                        .merge_us
                        .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    profile.merge_calls.fetch_add(1, Ordering::Relaxed);
                } else {
                    merge(entry.get_mut(), v);
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(v);
            }
        }
    }

    let mut in_queue = FxHashSet::default();


    for &scc_idx in &traversal_data.topo {
        let scc_nodes = &traversal_data.sccs[scc_idx];
        let mut local_queue: VecDeque<StateID> = VecDeque::new();

        for &u in scc_nodes {
            if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                local_queue.push_back(u);
                in_queue.insert(u);
            }
        }
        if local_queue.is_empty() { continue; }

        while let Some(u) = local_queue.pop_front() {
            in_queue.remove(&u);

            if stopped_nodes.contains(&u) { continue; }

            let agg_v = match values.get(&u) { Some(v) => v.clone(), None => continue };
            let proceed_val = if let Some(profile) = profile {
                let start = Instant::now();
                let result = process(u, agg_v.clone());
                profile
                    .process_us
                    .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                profile.process_calls.fetch_add(1, Ordering::Relaxed);
                match result {
                    Some(val) => val,
                    None => { stopped_nodes.insert(u); continue; }
                }
            } else {
                match process(u, agg_v.clone()) {
                    Some(val) => val,
                    None => { stopped_nodes.insert(u); continue; }
                }
            };
            let state = &nwa.states[u];

            if !state.epsilons.is_empty() {
                let step_start = profile.map(|_| Instant::now());
                for (v, new_v) in step(&proceed_val, None, &state.epsilons) {
                    if stopped_nodes.contains(&v) { continue; }
                    let changed = match values.entry(v) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            if let Some(profile) = profile {
                                let start = Instant::now();
                                let changed = merge(entry.get_mut(), new_v);
                                profile
                                    .merge_us
                                    .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                                profile.merge_calls.fetch_add(1, Ordering::Relaxed);
                                changed
                            } else {
                                merge(entry.get_mut(), new_v)
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(new_v);
                            true
                        }
                    };
                    if changed && traversal_data.comp_id[v] == scc_idx && !in_queue.contains(&v) {
                        local_queue.push_back(v);
                        in_queue.insert(v);
                    }
                }
                if let (Some(profile), Some(step_start)) = (profile, step_start) {
                    profile
                        .step_us
                        .fetch_add(step_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    profile.step_calls.fetch_add(1, Ordering::Relaxed);
                }
            }
            for (&label, targets) in &state.transitions {
                let step_start = profile.map(|_| Instant::now());
                for (v, new_v) in step(&proceed_val, Some(label), targets) {
                    if stopped_nodes.contains(&v) { continue; }
                    let changed = match values.entry(v) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            if let Some(profile) = profile {
                                let start = Instant::now();
                                let changed = merge(entry.get_mut(), new_v);
                                profile
                                    .merge_us
                                    .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                                profile.merge_calls.fetch_add(1, Ordering::Relaxed);
                                changed
                            } else {
                                merge(entry.get_mut(), new_v)
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(new_v);
                            true
                        }
                    };
                    if changed && traversal_data.comp_id[v] == scc_idx && !in_queue.contains(&v) {
                        local_queue.push_back(v);
                        in_queue.insert(v);
                    }
                }
                if let (Some(profile), Some(step_start)) = (profile, step_start) {
                    profile
                        .step_us
                        .fetch_add(step_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    profile.step_calls.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

fn weight_from_terminals(
    terminals: &BTreeSet<TerminalID>,
    term_to_bit: &BTreeMap<Option<TerminalID>, usize>,
) -> Weight {
    let mut w = Weight::zeros();
    for term in terminals {
        if let Some(bit) = term_to_bit.get(&Some(*term)) {
            w.set(*bit, true);
        }
    }
    w
}

fn ensure_nt_stack_state(
    nwa: &mut NWA,
    nt_stacks: &mut BTreeMap<NonTerminalID, Vec<StateID>>,
    nt: NonTerminalID,
    depth: usize,
) -> Result<StateID, FullDWABuildError> {
    let stack = nt_stacks.get_mut(&nt).expect("nt stack must exist");
    while stack.len() <= depth {
        let new_state = nwa.add_state();
        let prev_state = *stack.last().expect("stack must be non-empty");
        nwa.add_transition(
            new_state,
            crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL,
            prev_state,
            Weight::all(),
        )?;
        stack.push(new_state);
    }
    Ok(stack[depth])
}

fn build_super_nwa_from_grouped(
    grouped: &GroupedCharacterization,
    all_nts: &BTreeSet<NonTerminalID>,
    term_to_bit: &BTreeMap<Option<TerminalID>, usize>,
    ignore_terms: &BTreeSet<TerminalID>,
) -> Result<NWA, FullDWABuildError> {
    let initial_shift_count = grouped.initial_shifts.len();
    let initial_shift_prefix_count = grouped
        .initial_shifts
        .keys()
        .map(|(state, _)| *state)
        .collect::<BTreeSet<_>>()
        .len();
    let initial_reduce_count = grouped.initial_reduces.len();
    let escape_shift_count: usize = grouped.per_nt.values().map(|g| g.escape_shifts.len()).sum();
    let escape_revealed_prefix_count: usize = grouped
        .per_nt
        .values()
        .map(|g| {
            let mut reveals = BTreeSet::new();
            for ((revealed, _, _), _) in &g.escape_shifts {
                reveals.insert(*revealed);
            }
            reveals.len()
        })
        .sum();
    let escape_goto_prefix_count: usize = grouped
        .per_nt
        .values()
        .map(|g| {
            let mut prefixes = BTreeSet::new();
            for ((revealed, goto, _), _) in &g.escape_shifts {
                prefixes.insert((*revealed, *goto));
            }
            prefixes.len()
        })
        .sum();
    let rereduce_count: usize = grouped.per_nt.values().map(|g| g.reveal_and_rereduces.len()).sum();
    let mut max_depth_per_nt: BTreeMap<NonTerminalID, usize> = BTreeMap::new();
    for ((_, len, nt), _) in &grouped.initial_reduces {
        max_depth_per_nt
            .entry(*nt)
            .and_modify(|d| *d = (*d).max(*len))
            .or_insert(*len);
    }
    for nt_group in grouped.per_nt.values() {
        for ((_, remaining_len, target_nt), _) in &nt_group.reveal_and_rereduces {
            max_depth_per_nt
                .entry(*target_nt)
                .and_modify(|d| *d = (*d).max(*remaining_len))
                .or_insert(*remaining_len);
        }
    }
    let nt_stack_extra: usize = max_depth_per_nt.values().copied().sum();
    let mut nwa = NWA::new_empty();
    let start = nwa.add_state();
    let end = nwa.add_state();
    nwa.body.start_states = vec![start];
    nwa.states[end].final_weight = Some(Weight::all());

    let mut nt_states: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    let mut nt_stacks: BTreeMap<NonTerminalID, Vec<StateID>> = BTreeMap::new();
    for nt in all_nts {
        let state = nwa.add_state();
        nt_states.insert(*nt, state);
        nt_stacks.insert(*nt, vec![state]);
    }

    // None + ignore terminals: allow epsilon from start to end.
    let mut epsilon_weight = Weight::zeros();
    epsilon_weight.set(0, true);
    for term in ignore_terms {
        if let Some(bit) = term_to_bit.get(&Some(*term)) {
            epsilon_weight.set(*bit, true);
        }
    }
    if !epsilon_weight.is_empty() {
        nwa.add_epsilon(start, end, epsilon_weight);
    }

    let mut escape_goto_unique_count = 0usize;

    let mut initial_shift_edges: Vec<(ParserStateID, ParserStateID, Weight)> = Vec::new();
    let mut initial_prefix_weights: BTreeMap<ParserStateID, Weight> = BTreeMap::new();
    for ((state, shift_state), terminals) in &grouped.initial_shifts {
        let weight = weight_from_terminals(terminals, term_to_bit);
        if weight.is_empty() {
            continue;
        }
        initial_shift_edges.push((*state, *shift_state, weight.clone()));
        if let Some(existing) = initial_prefix_weights.get_mut(state) {
            *existing |= &weight;
        } else {
            initial_prefix_weights.insert(*state, weight.clone());
        }
    }
    let mut initial_prefix_states: BTreeMap<ParserStateID, (StateID, StateID)> = BTreeMap::new();
    for (state, weight) in &initial_prefix_weights {
        let pop_label = crate::precompute4::utils::encode_symbol_i16(*state)?;
        let push_label_state = crate::precompute4::utils::encode_negative_i16(*state)?;
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(start, pop_label, s1, weight.clone())?;
        nwa.add_transition(s1, push_label_state, s2, weight.clone())?;
        initial_prefix_states.insert(*state, (s1, s2));
    }
    for (state, shift_state, weight) in initial_shift_edges {
        let (_, s2) = initial_prefix_states
            .get(&state)
            .copied()
            .expect("initial shift prefix must exist");
        let push_label_shift = crate::precompute4::utils::encode_negative_i16(shift_state)?;
        nwa.add_transition(s2, push_label_shift, end, weight)?;
    }

    for ((state, len, nt), terminals) in &grouped.initial_reduces {
        let weight = weight_from_terminals(terminals, term_to_bit);
        if weight.is_empty() {
            continue;
        }
        let pop_label = crate::precompute4::utils::encode_symbol_i16(*state)?;
        let target = ensure_nt_stack_state(&mut nwa, &mut nt_stacks, *nt, *len)?;
        nwa.add_transition(start, pop_label, target, weight)?;
    }

    for (nt, nt_group) in &grouped.per_nt {
        let nt_state = *nt_states.get(nt).expect("nt state must exist");

        let mut escape_goto_shift_edges: BTreeMap<(ParserStateID, ParserStateID), BTreeMap<ParserStateID, Weight>> =
            BTreeMap::new();
        let mut escape_goto_weights: BTreeMap<(ParserStateID, ParserStateID), Weight> =
            BTreeMap::new();
        let mut escape_revealed_weights: BTreeMap<ParserStateID, Weight> = BTreeMap::new();
        for ((revealed, goto, shift), terminals) in &nt_group.escape_shifts {
            let weight = weight_from_terminals(terminals, term_to_bit);
            if weight.is_empty() {
                continue;
            }
            let goto_key = (*revealed, *goto);
            let shift_map = escape_goto_shift_edges.entry(goto_key).or_default();
            if let Some(existing) = shift_map.get_mut(shift) {
                *existing |= &weight;
            } else {
                shift_map.insert(*shift, weight.clone());
            }
            if let Some(existing) = escape_goto_weights.get_mut(&goto_key) {
                *existing |= &weight;
            } else {
                escape_goto_weights.insert(goto_key, weight.clone());
            }
            if let Some(existing) = escape_revealed_weights.get_mut(revealed) {
                *existing |= &weight;
            } else {
                escape_revealed_weights.insert(*revealed, weight.clone());
            }
        }
        let mut escape_revealed_states: BTreeMap<ParserStateID, (StateID, StateID)> =
            BTreeMap::new();
        for (revealed, weight) in &escape_revealed_weights {
            let pop_label = crate::precompute4::utils::encode_symbol_i16(*revealed)?;
            let push_revealed = crate::precompute4::utils::encode_negative_i16(*revealed)?;
            let s1 = nwa.add_state();
            let s2 = nwa.add_state();
            nwa.add_transition(nt_state, pop_label, s1, weight.clone())?;
            nwa.add_transition(s1, push_revealed, s2, weight.clone())?;
            escape_revealed_states.insert(*revealed, (s1, s2));
        }
        let mut escape_goto_states: BTreeMap<(ParserStateID, ParserStateID), StateID> = BTreeMap::new();
        let mut goto_state_cache: HashMap<Vec<(ParserStateID, Weight)>, StateID> = HashMap::new();
        for ((revealed, goto), shifts) in &escape_goto_shift_edges {
            let shift_edges: Vec<(ParserStateID, Weight)> = shifts
                .iter()
                .map(|(shift, weight)| (*shift, weight.clone()))
                .collect();
            let cached_state = goto_state_cache.get(&shift_edges).copied();
            let s3 = if let Some(existing) = cached_state {
                existing
            } else {
                let new_state = nwa.add_state();
                for (shift, weight) in &shift_edges {
                    let push_shift = crate::precompute4::utils::encode_negative_i16(*shift)?;
                    nwa.add_transition(new_state, push_shift, end, weight.clone())?;
                }
                goto_state_cache.insert(shift_edges.clone(), new_state);
                new_state
            };
            let (_, s2) = escape_revealed_states
                .get(revealed)
                .copied()
                .expect("escape revealed prefix must exist");
            let push_goto = crate::precompute4::utils::encode_negative_i16(*goto)?;
            let weight = escape_goto_weights
                .get(&(*revealed, *goto))
                .expect("escape goto weight must exist");
            nwa.add_transition(s2, push_goto, s3, weight.clone())?;
            escape_goto_states.insert((*revealed, *goto), s3);
        }
        escape_goto_unique_count += goto_state_cache.len();
        for ((revealed, remaining_len, target_nt), terminals) in &nt_group.reveal_and_rereduces {
            let weight = weight_from_terminals(terminals, term_to_bit);
            if weight.is_empty() {
                continue;
            }
            let pop_label = crate::precompute4::utils::encode_symbol_i16(*revealed)?;
            let target = ensure_nt_stack_state(&mut nwa, &mut nt_stacks, *target_nt, *remaining_len)?;
            nwa.add_transition(nt_state, pop_label, target, weight)?;
        }
    }

    let predicted_states = 2
        + all_nts.len()
        + nt_stack_extra
        + 2 * initial_shift_prefix_count
        + 2 * escape_revealed_prefix_count
        + escape_goto_unique_count;

    crate::debug!(
        5,
        "Super NWA size breakdown: start_end=2 nt_states={} nt_stack_extra={} initial_shifts={} initial_shift_prefixes={} escape_shifts={} escape_revealed_prefixes={} escape_goto_prefixes={} escape_goto_unique={} initial_reduces={} reveal_rereduces={} predicted_states={} actual_states={}",
        all_nts.len(),
        nt_stack_extra,
        initial_shift_count,
        initial_shift_prefix_count,
        escape_shift_count,
        escape_revealed_prefix_count,
        escape_goto_prefix_count,
        escape_goto_unique_count,
        initial_reduce_count,
        rereduce_count,
        predicted_states,
        nwa.states.len(),
    );

    Ok(nwa)
}

fn make_template_bundle(
    terminal_to_weight: &BTreeMap<Option<TerminalID>, Weight>,
    template_dwas: &BTreeMap<TerminalID, DWA>,
    ignore_dwa: &DWA,
    ignore_terminal_ids: &HashSet<TerminalID>,
    super_nwa: &NWA,
    bit_to_term: &[Option<TerminalID>],
) -> NWA {
    if terminal_to_weight.is_empty() {
        return NWA::new_empty();
    }

    let _backend_override = terminal_to_weight.values().next().map(|w| {
        let choice = match w {
            Weight::RangeSet(_) => BackendChoice::RangeSet,
            Weight::Factorized(_) => BackendChoice::Factorized,
            Weight::RangeMap(_) => BackendChoice::RangeMap,
        };
        WeightBackendOverride::from_choice(choice)
    });

    if terminal_to_weight.len() == 1 {
        let (term_opt, weight) = terminal_to_weight
            .iter()
            .next()
            .expect("terminal_to_weight is non-empty");
        let base_dwa = match term_opt {
            Some(term_id) if ignore_terminal_ids.contains(term_id) => ignore_dwa,
            Some(term_id) => template_dwas.get(term_id).unwrap_or(ignore_dwa),
            None => ignore_dwa,
        };
        if weight.is_empty() {
            return NWA::new_empty();
        }
        let mut nwa = NWA::from_dwa(base_dwa);
        for state in &mut nwa.states.0 {
            for targets in state.transitions.values_mut() {
                for (_, w) in targets {
                    *w = Weight::all();
                }
            }
            for (_, w) in &mut state.epsilons {
                *w = Weight::all();
            }
            if state.final_weight.is_some() {
                state.final_weight = if weight.is_empty() {
                    None
                } else {
                    Some(weight.clone())
                };
            }
        }
        return nwa;
    }
    let mut combined: Option<NWA> = None;

    for (term_opt, weight) in terminal_to_weight {
        if weight.is_empty() {
            continue;
        }
        let base_dwa = match term_opt {
            Some(term_id) if ignore_terminal_ids.contains(term_id) => ignore_dwa,
            Some(term_id) => template_dwas.get(term_id).unwrap_or(ignore_dwa),
            None => ignore_dwa,
        };
        let mut single_nwa = NWA::from_dwa(base_dwa);
        for state in &mut single_nwa.states.0 {
            for targets in state.transitions.values_mut() {
                for (_, w) in targets {
                    *w = Weight::all();
                }
            }
            for (_, w) in &mut state.epsilons {
                *w = Weight::all();
            }
            if state.final_weight.is_some() {
                state.final_weight = Some(weight.clone());
            }
        }
        if let Some(existing) = &mut combined {
            existing.union_assign(&single_nwa);
        } else {
            combined = Some(single_nwa);
        }
    }

    let combined = match combined {
        Some(nwa) => nwa,
        None => return NWA::new_empty(),
    };

    let dwa = {
        let _restore = crate::r#macro::set_silence_debug_timing(true);
        let res = combined.determinize_and_minimize(DeterminizeAndMinimizeProfile::SpecializedSuper);
        crate::r#macro::set_silence_debug_timing(_restore);
        res
    };
    NWA::from_dwa(&dwa)
}

/// Build the Parser DWA from the GLR parser and lexical NWA.
/// 
/// This is the main precomputation function that:
/// 1. Builds template DWAs from terminal characterizations (one per terminal group)
/// 2. Composes them with the lexical NWA
/// 3. Determinizes the result into the final Parser DWA
/// 
/// The resulting DWA is used at runtime for O(1) mask queries.
#[time_it("build_parser_dwa")]
pub fn build_parser_dwa(parser: &GLRParser, terminal_nwa: &NWA) -> DWA {
    crate::debug!(5, "build_parser_dwa: start");
    crate::debug!(3, "Starting Parser DWA construction. Input terminal_nwa: {}", 
        terminal_nwa.stats());
    let parser_dwa_total_start = Instant::now();

    // Handle empty terminal NWA (no valid tokens for this grammar/vocabulary combination)
    // Return a minimal DWA with one state and no transitions (always returns empty mask)
    if terminal_nwa.states.0.is_empty() || terminal_nwa.body.start_states.is_empty() {
        crate::debug!(3, "Terminal NWA is empty - returning empty Parser DWA");
        let mut empty_dwa = DWA::new_empty();
        // Add a single start state with no final weight (no tokens are valid)
        let start_state = empty_dwa.states.add_state();
        empty_dwa.body.start_state = start_state;
        crate::debug!(5, "build_parser_dwa: end (empty terminal NWA)");
        return empty_dwa;
    }
    
    let template_dwas_start = Instant::now();
    let template_dwas = timeit!("build_template_dwas", {
        let _restore = crate::r#macro::set_silence_debug_timing(true);
        let res = match build_template_dwas(parser) {
            Ok(m) => m,
            Err(e) => panic!("Failed to build template DWAs: {:?}", e),
        };
        crate::r#macro::set_silence_debug_timing(_restore);
        res
    });
    crate::timing!(
        "TIMING: parser_dwa::build_template_dwas {:?}",
        template_dwas_start.elapsed()
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::build_template_dwas = {:?}",
        template_dwas_start.elapsed()
    );

    let ignore_dwa_start = Instant::now();
    let ignore_dwa = timeit!("build_ignore_terminal_dwa", {
        build_ignore_terminal_dwa()
    });
    crate::timing!(
        "TIMING: parser_dwa::build_ignore_terminal_dwa {:?}",
        ignore_dwa_start.elapsed()
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::build_ignore_terminal_dwa = {:?}",
        ignore_dwa_start.elapsed()
    );

    // Check if we're in symbol-heavy mode (tsid encoded as labels, not weights)
    let is_symbol_heavy = !crate::constraint_precompute::is_weight_heavy_enabled();
    let terminals_count = parser.terminal_map.len();
    
    // In symbol-heavy mode, identify the original start state and tsid-labeled incoming edges
    // These will be used to reconstruct tsid-labeled transitions at the end
    let original_start_state = terminal_nwa.body.start_states[0];
    let tsid_to_root: BTreeMap<Label, StateID> = if is_symbol_heavy {
        let start_transitions = &terminal_nwa.states[original_start_state].transitions;
        
        // Collect tsid-labeled transitions (labels >= terminals_count)
        let mut mapping = BTreeMap::new();
        for (&label, targets) in start_transitions {
            if label as usize >= terminals_count {
                // This is a tsid transition: start --[tsid_label]--> root
                for &(target, _) in targets {
                    mapping.insert(label, target);
                }
            }
        }
        crate::debug!(4, "Symbol-heavy mode: found {} tsid transitions from original start state", mapping.len());
        mapping
    } else {
        BTreeMap::new()
    };

    // Debug: dump input terminal NWA
    crate::debug!(5, "Input terminal NWA: {}, start_states={:?}", terminal_nwa.stats(), terminal_nwa.body.start_states);
    for (i, state) in terminal_nwa.states.0.iter().enumerate() {
        crate::debug!(6, "  Input State {}: final_weight={:?}, epsilons={}, transitions={:?}", 
            i, 
            state.final_weight.as_ref().map(|w| format!("len={}, ranges={}", w.len(), w.ranges_len())),
            state.epsilons.len(),
            state.transitions.iter().map(|(&l, targets)| format!("{}:{}", l, targets.len())).collect::<Vec<_>>()
        );
    }

    let reverse_start = Instant::now();
    let reversed_nwa = timeit!("parser_dwa::reverse_nwa", {
        terminal_nwa.reverse()
    });
    crate::timing!("TIMING: parser_dwa::reverse_nwa {:?}", reverse_start.elapsed());
    crate::timing!(
        "PHASE_TIMING: parser_dwa::reverse_nwa = {:?}",
        reverse_start.elapsed()
    );
    crate::debug!(5, "Reversed NWA: {}, start_states={:?}", reversed_nwa.stats(), reversed_nwa.body.start_states);
    for (i, state) in reversed_nwa.states.0.iter().enumerate() {
        crate::debug!(6, "  State {}: final_weight={:?}, epsilons={:?}, transitions={}", 
            i, 
            state.final_weight.as_ref().map(|w| format!("len={}, ranges={}", w.len(), w.ranges_len())),
            state.epsilons.iter().take(3).map(|(v, w)| format!("->{}(len={}, ranges={})", v, w.len(), w.ranges_len())).collect::<Vec<_>>(),
            state.transitions.len()
        );
    }
    let traversal_start = Instant::now();
    let traversal_data = timeit!("parser_dwa::compute_traversal_data", {
        reversed_nwa.compute_traversal_data()
    });
    crate::timing!(
        "TIMING: parser_dwa::compute_traversal_data {:?}",
        traversal_start.elapsed()
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::compute_traversal_data = {:?}",
        traversal_start.elapsed()
    );

    
    // In symbol-heavy mode, build a map of OUTGOING tsid-labeled edges FROM each root state
    // In the reversed NWA, root --[tsid_label]--> original_start
    // We need: root -> [(tsid_label, edge_weight), ...]
    let outgoing_tsid_edges_start = Instant::now();
    let outgoing_tsid_edges: BTreeMap<StateID, Vec<(Label, Weight)>> = if is_symbol_heavy {
        timeit!("parser_dwa::outgoing_tsid_edges", {
            let mut outgoing: BTreeMap<StateID, Vec<(Label, Weight)>> = BTreeMap::new();
            for (src, state) in reversed_nwa.states.0.iter().enumerate() {
                for (&label, targets) in &state.transitions {
                    if label as usize >= terminals_count {
                        // This is a tsid-labeled transition
                        for (dst, weight) in targets {
                            if *dst == original_start_state {
                                outgoing.entry(src).or_default().push((label, weight.clone()));
                            }
                        }
                    }
                }
            }
            crate::debug!(5, "Symbol-heavy mode: {} root states with tsid edges", outgoing.len());
            for (src, edges) in &outgoing {
                crate::debug!(6, "  Root state {} has tsid edges: {:?}", src, edges.iter().map(|(l,_)|*l).collect::<Vec<_>>());
            }
            outgoing
        })
    } else {
        BTreeMap::new()
    };
    if is_symbol_heavy {
        crate::timing!(
            "TIMING: parser_dwa::outgoing_tsid_edges {:?}",
            outgoing_tsid_edges_start.elapsed()
        );
        crate::timing!(
            "PHASE_TIMING: parser_dwa::outgoing_tsid_edges = {:?}",
            outgoing_tsid_edges_start.elapsed()
        );
    }

    let super_nwa_start = Instant::now();
    let used_terminals: BTreeSet<TerminalID> =
        parser.terminal_map.right_values().cloned().collect();
    let ignore_terms: BTreeSet<TerminalID> = used_terminals
        .iter()
        .filter(|term| parser.ignore_terminal_ids.contains(term))
        .cloned()
        .collect();
    let used_nonignore_terms: BTreeSet<TerminalID> = used_terminals
        .difference(&ignore_terms)
        .cloned()
        .collect();

    let mut term_to_bit = BTreeMap::new();
    let mut bit_to_term: Vec<Option<TerminalID>> = Vec::new();
    term_to_bit.insert(None, 0);
    bit_to_term.push(None);
    for (i, term_id) in used_terminals.iter().enumerate() {
        term_to_bit.insert(Some(*term_id), i + 1);
        bit_to_term.push(Some(*term_id));
    }

    let super_nwa = {
        let _rangeset_backend = WeightBackendOverride::new("rangeset");
        let all_chars = compute_all_characterizations(parser);
        let grouped = GroupedCharacterization::from_terminals(&all_chars, &used_nonignore_terms);
        let all_nts: BTreeSet<NonTerminalID> =
            parser.non_terminal_map.right_values().cloned().collect();

        build_super_nwa_from_grouped(
            &grouped,
            &all_nts,
            &term_to_bit,
            &ignore_terms,
        )
            .expect("Failed to build Super NWA")
    };
    crate::debug!(5, "  Super NWA: {}", super_nwa.stats());
    crate::timing!("TIMING: parser_dwa::build_super_nwa {:?}", super_nwa_start.elapsed());
    crate::timing!(
        "PHASE_TIMING: parser_dwa::build_super_nwa = {:?}",
        super_nwa_start.elapsed()
    );
    let super_nwa = Arc::new(super_nwa);
    let bit_to_term = Arc::new(bit_to_term);

    crate::debug!(4, "Super NWA ready for template bundling");

    let states_arena = RefCell::new(NWAStates::default());
    let template_ranges: RefCell<Vec<std::ops::Range<usize>>> = RefCell::new(Vec::new());
    let initial_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_states: vec![start] }
    };
    let initial_term_map: BTreeMap<Option<TerminalID>, Weight> = BTreeMap::from([(None, Weight::all())]);
    let initial_values_full: Vec<(usize, (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, Weight))> =
        reversed_nwa.body.start_states.iter().map(|&s| (s, (BTreeMap::from([(initial_body.clone(), initial_term_map.clone())]), Weight::all()))).collect();

    // Store (NWABody, Weight, node_id) - node_id is the state ID in reversed NWA where we collected this body
    // In symbol-heavy mode, we also collect tsid-specific bodies separately
    let final_bodies_arc: Arc<Mutex<Vec<(NWABody, Weight, StateID)>>> = Arc::new(Mutex::new(Vec::new()));
    
    // For symbol-heavy mode: collect (NWABody, Weight, tsid_label) for each tsid-labeled transition
    // These are transitions from root states to the original start state in the reversed NWA
    let tsid_bodies_arc: Arc<Mutex<Vec<(NWABody, Weight, Label)>>> = Arc::new(Mutex::new(Vec::new()));

    crate::debug!(4, "Beginning NWA traversal");

    let pass2_start = Instant::now();
    let pass2_profile = Arc::new(Pass2Profile::new());
    let pass2_traversal_profile_enabled = std::env::var("PROFILE_PARSER_DWA_PASS2_TRAVERSAL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let pass2_traversal_profile = pass2_traversal_profile_enabled.then(NwaSpecialMapProfile::new);
    let pass2_traversal_profile_ref = pass2_traversal_profile.as_ref();

    let template_bundle_cache: RefCell<FxHashMap<BTreeMap<Option<TerminalID>, Weight>, Arc<NWA>>> =
        RefCell::new(FxHashMap::default());

    // Clone references for use in closures
    let tsid_bodies_for_process = tsid_bodies_arc.clone();
    let pass2_profile_for_process = pass2_profile.clone();
    let super_nwa_ref = &super_nwa;
    let bit_to_term_ref = &bit_to_term;
    let template_bundle_cache_ref = &template_bundle_cache;

    timeit!("parser_dwa::pass2_traversal", {
        nwa_special_map(
            &reversed_nwa, &traversal_data, initial_values_full,
                |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, Weight), edge_label, transitions| {
                    let (current_bodies, current_tokens) = current_val;
                    let mut results = Vec::new();
                    
                    // In symbol-heavy mode, skip tsid-labeled transitions in normal traversal
                    // These will be handled in the process callback when we're at a root state
                    if is_symbol_heavy {
                        if let Some(label) = edge_label {
                            if label as usize >= terminals_count {
                                // This is a tsid-labeled transition - skip it
                                return results;
                            }
                        }
                    }
                    
                    let terminal_id = edge_label.map(|l| TerminalID(l as usize));
                    for (dest_id, weight) in transitions {
                        let next_tokens = current_tokens & weight;
                        if next_tokens.is_empty() { continue; }
                        let mut terminal_map = BTreeMap::new();
                        terminal_map.insert(terminal_id, weight.clone());
                        let mut body_map = BTreeMap::new();
                        for body in current_bodies.keys() { body_map.insert(body.clone(), terminal_map.clone()); }
                        results.push((*dest_id, (body_map, next_tokens)));
                    }
                    results
                },
                |val1, val2| {
                    let (bodies1, tokens1) = val1;
                    let (bodies2, tokens2) = val2;
                    let mut changed = false;
                    for (right_body, term_map2) in bodies2 {
                        let term_map1 = bodies1.entry(right_body.clone()).or_default();
                        for (term, weight2) in term_map2 {
                            let entry = term_map1.entry(term).or_insert_with(Weight::zeros);
                            if !weight2.is_subset_of(entry) {
                                *entry |= &weight2;
                                changed = true;
                            }
                        }
                    }
                    if !tokens2.is_subset_of(tokens1) {
                        *tokens1 |= &tokens2;
                        changed = true;
                    }
                    changed
                },
                |node_id, val| {
                static PROCESS_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                static INSTANTIATE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                static TOTAL_TEMPLATE_STATES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                
                let proc_count = PROCESS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if proc_count % 10000 == 0 {
                    crate::debug!(4, "Process callback #{}, instantiate count: {}, total_states: {}", proc_count, 
                        INSTANTIATE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                        TOTAL_TEMPLATE_STATES.load(std::sync::atomic::Ordering::Relaxed));
                }
                
                let process_start = Instant::now();
                pass2_profile_for_process
                    .process_count
                    .fetch_add(1, Ordering::Relaxed);
                let mut accounted_us: u64 = 0;

                let (nwa_bodies_map, tokens) = val;
                let bodies_count = nwa_bodies_map.len();
                let mut nwa_body = NWABody { start_states: vec![] };
                for (right_body, terminal_map) in &nwa_bodies_map {
                    let cache_lookup_start = Instant::now();
                    let cached = template_bundle_cache_ref
                        .borrow()
                        .get(terminal_map)
                        .cloned();
                    let cache_lookup_us = cache_lookup_start.elapsed().as_micros() as u64;
                    pass2_profile_for_process
                        .cache_lookup_us
                        .fetch_add(cache_lookup_us, Ordering::Relaxed);
                    accounted_us = accounted_us.saturating_add(cache_lookup_us);

                    let template_nwa = if let Some(nwa) = cached {
                        pass2_profile_for_process
                            .cache_hits
                            .fetch_add(1, Ordering::Relaxed);
                        nwa
                    } else {
                        pass2_profile_for_process
                            .cache_misses
                            .fetch_add(1, Ordering::Relaxed);
                        let bundle_start = Instant::now();
                        let template_nwa = make_template_bundle(
                            terminal_map,
                            &template_dwas,
                            &ignore_dwa,
                            &parser.ignore_terminal_ids,
                            super_nwa_ref.as_ref(),
                            bit_to_term_ref.as_ref(),
                        );
                        let bundle_us = bundle_start.elapsed().as_micros() as u64;
                        pass2_profile_for_process
                            .instantiate_us
                            .fetch_add(bundle_us, Ordering::Relaxed);
                        accounted_us = accounted_us.saturating_add(bundle_us);

                        let template_nwa = Arc::new(template_nwa);
                        let cache_insert_start = Instant::now();
                        template_bundle_cache_ref
                            .borrow_mut()
                            .insert(terminal_map.clone(), template_nwa.clone());
                        let cache_insert_us = cache_insert_start.elapsed().as_micros() as u64;
                        pass2_profile_for_process
                            .cache_insert_us
                            .fetch_add(cache_insert_us, Ordering::Relaxed);
                        accounted_us = accounted_us.saturating_add(cache_insert_us);
                        template_nwa
                    };

                    pass2_profile_for_process
                        .template_count
                        .fetch_add(1, Ordering::Relaxed);

                    let template_nwa = template_nwa.as_ref();
                    let template_size = template_nwa.states.len();
                    let count = INSTANTIATE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    TOTAL_TEMPLATE_STATES.fetch_add(template_size, std::sync::atomic::Ordering::Relaxed);
                    if count % 10000 == 0 {
                        crate::debug!(4, "Template instantiation #{}: {} total states so far, template size {}, bodies_count: {}", 
                            count, TOTAL_TEMPLATE_STATES.load(std::sync::atomic::Ordering::Relaxed), template_size, bodies_count);
                    }

                    let mut states = states_arena.borrow_mut();
                    let template_offset = states.len();
                    let composed_body = states.concatenate_in_place(template_nwa, right_body);
                    let template_end = template_offset + template_nwa.states.len();
                    // Collect template range for deferred multi-range cancellation.
                    // Cancellation is done as a single batch after pass2 for efficiency.
                    template_ranges.borrow_mut().push(template_offset..template_end);

                    let union_start = Instant::now();
                    nwa_body = NWABody::union(&nwa_body, &composed_body);
                    let union_us = union_start.elapsed().as_micros() as u64;
                    pass2_profile_for_process.union_us.fetch_add(
                        union_us,
                        Ordering::Relaxed,
                    );
                    accounted_us = accounted_us.saturating_add(union_us);
                }
                
                // In symbol-heavy mode, check if this is a root state (has tsid-labeled edges to original_start_state)
                // If so, collect the body for each tsid
                if is_symbol_heavy {
                    if let Some(tsid_edges) = outgoing_tsid_edges.get(&node_id) {
                        for (tsid_label, edge_weight) in tsid_edges {
                            let intersection_w = &tokens & edge_weight;
                            if !intersection_w.is_empty() && !nwa_body.start_states.is_empty() {
                                let final_w = intersection_w;
                                crate::debug!(6, "Collecting tsid body at root {} for label {} with {} tokens", 
                                    node_id, tsid_label, final_w.len());
                                let tsid_collect_start = Instant::now();
                                let mut tb = tsid_bodies_for_process.lock().unwrap();
                                tb.push((nwa_body.clone(), final_w, *tsid_label));
                                let tsid_collect_us = tsid_collect_start.elapsed().as_micros() as u64;
                                pass2_profile_for_process.tsid_collect_us.fetch_add(
                                    tsid_collect_us,
                                    Ordering::Relaxed,
                                );
                                accounted_us = accounted_us.saturating_add(tsid_collect_us);
                            }
                        }
                    }
                }
                
                // Check if this is a final state in the reversed NWA (original start state or root state)
                // In symbol-heavy mode, we handle the original_start_state specially via tsid_bodies
                // so don't collect it here
                let has_final_weight = reversed_nwa.states[node_id].final_weight.is_some();
                crate::debug!(7, "Process node_id={}, is_symbol_heavy={}, has_final_weight={}", node_id, is_symbol_heavy, has_final_weight);
                let should_collect = if is_symbol_heavy {
                    // In symbol-heavy mode, only collect for states OTHER than original_start_state
                    // (the original start is handled via tsid-labeled transitions)
                    node_id != original_start_state && reversed_nwa.states[node_id].final_weight.is_some()
                } else {
                    reversed_nwa.states[node_id].final_weight.is_some()
                };
                
                if should_collect {
                    if let Some(fw) = &reversed_nwa.states[node_id].final_weight {
                        let intersection_w = &tokens & fw;
                        crate::debug!(7, "Final body candidate: node_id={}, tokens_len={}, tokens_ranges={}, fw_len={}, fw_ranges={}, intersection_len={}, intersection_ranges={}", 
                            node_id, tokens.len(), tokens.ranges_len(), fw.len(), fw.ranges_len(), intersection_w.len(), intersection_w.ranges_len());
                        if !intersection_w.is_empty() {
                            let final_w = intersection_w;
                            let final_collect_start = Instant::now();
                            let mut fb = final_bodies_arc.lock().unwrap();
                            fb.push((nwa_body.clone(), final_w, node_id));
                            let final_collect_us = final_collect_start.elapsed().as_micros() as u64;
                            pass2_profile_for_process.final_collect_us.fetch_add(
                                final_collect_us,
                                Ordering::Relaxed,
                            );
                            accounted_us = accounted_us.saturating_add(final_collect_us);
                        }
                    }
                }
                
                let process_elapsed = process_start.elapsed();
                let process_elapsed_us = process_elapsed.as_micros() as u64;
                pass2_profile_for_process.process_total_us.fetch_add(
                    process_elapsed_us,
                    Ordering::Relaxed,
                );
                let other_us = process_elapsed_us.saturating_sub(accounted_us);
                pass2_profile_for_process.process_other_us.fetch_add(
                    other_us,
                    Ordering::Relaxed,
                );

                if !tokens.is_empty() {
                    let mut next_body_map = BTreeMap::new(); next_body_map.insert(nwa_body, BTreeMap::new());
                    Some((next_body_map, tokens))
                } else { None }
                },
                pass2_traversal_profile_ref,
            );
        // Incremental cancellations/finality/negative removal disabled.
        // Batch resolution runs after pass 2 completes.
        pass2_profile.log();
        if let Some(profile) = pass2_traversal_profile_ref {
            profile.log("pass2_traversal");
        }
        let cache_hits = pass2_profile.cache_hits.load(Ordering::Relaxed);
        let cache_misses = pass2_profile.cache_misses.load(Ordering::Relaxed);
        let cache_total = cache_hits + cache_misses;
        let cache_hit_rate = if cache_total == 0 {
            0.0
        } else {
            (cache_hits as f64) * 100.0 / (cache_total as f64)
        };
        crate::timing!(
            "TIMING: parser_dwa::pass2_profile process_total={:?}, canonicalize={:?}, cache_lookup={:?}, cache_insert={:?}, cache_hits={}, cache_misses={}, cache_hit_rate={:.2}%, dynamic_derive={:?}, instantiate={:?}, cancellations={:?}, finality_fixpoint={:?}, remove_negative={:?}, union={:?}, tsid_collect={:?}, final_collect={:?}, other={:?}, templates={}",
            std::time::Duration::from_micros(pass2_profile.process_total_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.canonicalize_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.cache_lookup_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.cache_insert_us.load(Ordering::Relaxed)),
            cache_hits,
            cache_misses,
            cache_hit_rate,
            std::time::Duration::from_micros(pass2_profile.dynamic_derive_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.instantiate_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.apply_cancellations_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.apply_finality_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.remove_negative_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.union_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.tsid_collect_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.final_collect_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.process_other_us.load(Ordering::Relaxed)),
            pass2_profile.template_count.load(Ordering::Relaxed),
        );
        crate::debug!(4, "Pass 2 (nwa_special_map) in {:?}", pass2_start.elapsed());
        // Drop the process closure's reference to tsid_bodies
        drop(tsid_bodies_for_process);

        crate::debug!(4, "Finished Pass 2");
    });
    crate::timing!("TIMING: parser_dwa::pass2_traversal {:?}", pass2_start.elapsed());
    crate::timing!(
        "PHASE_TIMING: parser_dwa::pass2_traversal = {:?}",
        pass2_start.elapsed()
    );

    // Batch negative resolution over the full arena after pass2 completes.
    // Multi-range cancellation: seeds from all template ranges in a single call.
    let batch_negatives_start = Instant::now();
    let skip_batch_neg = std::env::var("PARSER_DWA_SKIP_BATCH_NEGATIVE_RESOLUTION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !skip_batch_neg {
        let mut states = states_arena.borrow_mut();
        let arena_len = states.len();
        let ranges = template_ranges.into_inner();
        crate::timing!(
            "TIMING: parser_dwa::batch_negative_resolution arena_len={} template_ranges={}",
            arena_len,
            ranges.len()
        );

        if arena_len > 0 {
            let mut neg_transition_edges = 0usize;
            let mut default_transition_edges = 0usize;
            let mut epsilon_edges = 0usize;
            for state_id in 0..arena_len {
                let state = &states[state_id];
                epsilon_edges += state.epsilons.len();
                for (&label, targets) in &state.transitions {
                    if is_negative_symbol(label) {
                        neg_transition_edges += targets.len();
                    }
                    if label == crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL {
                        default_transition_edges += targets.len();
                    }
                }
            }
            crate::timing!(
                "TIMING: parser_dwa::batch_neg_input_sizes neg_edges={} default_edges={} epsilon_edges={}",
                neg_transition_edges,
                default_transition_edges,
                epsilon_edges
            );

            // Cancellation must account for interactions across template ranges.
            // Always resolve across the full arena for correctness.
            let cancellations_start = Instant::now();
            apply_cancellations_range(&mut states, 0..arena_len);
            crate::timing!(
                "TIMING: parser_dwa::apply_cancellations {:?}",
                cancellations_start.elapsed()
            );
            crate::timing!(
                "PHASE_TIMING: parser_dwa::apply_cancellations = {:?}",
                cancellations_start.elapsed()
            );

            // Finality fixpoint needs negatives present for correct propagation.
            let finality_start = Instant::now();
            apply_finality_fixpoint_range(&mut states, 0..arena_len);
            crate::timing!(
                "TIMING: parser_dwa::apply_finality_fixpoint {:?}",
                finality_start.elapsed()
            );
            crate::timing!(
                "PHASE_TIMING: parser_dwa::apply_finality_fixpoint = {:?}",
                finality_start.elapsed()
            );

            let remove_start = Instant::now();
            remove_negative_transitions_range(&mut states, 0..arena_len);
            crate::timing!(
                "TIMING: parser_dwa::remove_negative_transitions {:?}",
                remove_start.elapsed()
            );
            crate::timing!(
                "PHASE_TIMING: parser_dwa::remove_negative_transitions = {:?}",
                remove_start.elapsed()
            );
        }
    }
    crate::timing!(
        "TIMING: parser_dwa::batch_negative_resolution {:?}",
        batch_negatives_start.elapsed()
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::batch_negative_resolution = {:?}",
        batch_negatives_start.elapsed()
    );
    let final_bodies = Arc::try_unwrap(final_bodies_arc).unwrap().into_inner().unwrap();
    let tsid_bodies = Arc::try_unwrap(tsid_bodies_arc).unwrap().into_inner().unwrap();
    let avg_template_size = states_arena.borrow().len() as f64 / (final_bodies.len() + tsid_bodies.len()).max(1) as f64;
    crate::debug!(4, "Collected {} final bodies, {} tsid bodies, states_arena has {} states (avg {:.0} states/body)", 
        final_bodies.len(), tsid_bodies.len(), states_arena.borrow().len(), avg_template_size);
    let combine_bodies_start = Instant::now();
    let combined_nwa = timeit!("parser_dwa::combine_bodies", {
        let mut combined_nwa_states = states_arena.into_inner();
        let combined_start_state = combined_nwa_states.add_state();

        if is_symbol_heavy && !tsid_bodies.is_empty() {
            // Symbol-heavy mode: add labeled transitions with tsid labels
            // Use the tsid_bodies collected during traversal
            for (body, weight, tsid_label) in tsid_bodies {
                crate::debug!(6, "Adding tsid body with label={}, weight len={}", tsid_label, weight.len());
                for &s in &body.start_states {
                    combined_nwa_states.add_transition(combined_start_state, tsid_label, s, weight.clone()).unwrap();
                }
            }
            crate::debug!(4, "Symbol-heavy mode: added {} tsid-labeled transitions", 
                combined_nwa_states[combined_start_state].transitions.values().map(|v| v.len()).sum::<usize>());
        } else {
            // Weight-heavy mode: no tsid labels, just add epsilon transitions with weights
            // The weights encode tsid info (positions in N×M space)
            for (body, weight, _node_id) in final_bodies {
                for &s in &body.start_states {
                    combined_nwa_states.add_epsilon(combined_start_state, s, weight.clone());
                }
            }
        }

        NWA { states: combined_nwa_states, body: NWABody { start_states: vec![combined_start_state] } }
    });
    crate::timing!(
        "TIMING: parser_dwa::combine_bodies {:?}",
        combine_bodies_start.elapsed()
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::combine_bodies = {:?}",
        combine_bodies_start.elapsed()
    );

    let macro_level = std::env::var("MACRO_DEBUG_LEVEL")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    if macro_level >= 4 {
        let expected_num_tsids = crate::datastructures::abstract_weight::current_num_tsids();
        validate_nwa_weight_dims(&combined_nwa, expected_num_tsids);
    }
    crate::debug!(3, "Combined NWA before determinization: states={}, is_symbol_heavy={}", 
        combined_nwa.states.len(), is_symbol_heavy);
    let finalize_start = Instant::now();
    let mut final_dwa = timeit!("parser_dwa::finalize_and_determinize", {
        finalize_and_optimize_and_determinize(parser, combined_nwa)
    });
    crate::timing!(
        "TIMING: parser_dwa::finalize_and_determinize {:?}",
        finalize_start.elapsed()
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::finalize_and_determinize = {:?}",
        finalize_start.elapsed()
    );
    // SKIP final minimization to test performance impact
    // final_dwa.minimize();
    crate::debug!(4, "Parser DWA construction complete. Stats: {}", final_dwa.stats());

    if let Some(avg_path_len) = final_dwa.average_path_length() {
        crate::debug!(4, "Parser DWA average path length: {:.2}", avg_path_len);
    }

    crate::timing!(
        "PHASE_TIMING: parser_dwa::total = {:?}",
        parser_dwa_total_start.elapsed()
    );
    crate::debug!(5, "build_parser_dwa: end");
    final_dwa
}

/// Deprecated alias for build_parser_dwa
#[deprecated(since = "0.3.0", note = "Use build_parser_dwa instead")]
pub fn precompute4(parser: &GLRParser, terminal_nwa: &NWA) -> DWA {
    build_parser_dwa(parser, terminal_nwa)
}

pub fn finalize_and_optimize_and_determinize(parser: &GLRParser, mut combined_nwa: NWA) -> DWA {
    crate::debug!(4, "Finalizing NWA (skipping stats on large arena)...");
    let finalize_total_start = std::time::Instant::now();
    
    // Prune unreachable states FIRST (forward BFS from starts - fast, removes 97% of states).
    // This dramatically reduces the state count before subtract_final_weights_from_outgoing,
    // which would otherwise iterate all 1.75M states.
    let before_prune_len = combined_nwa.states.len();
    let prune_unreachable_start = std::time::Instant::now();
    let unreach_changed = combined_nwa.prune_unreachable();
    let prune_unreachable_time = prune_unreachable_start.elapsed();
    crate::timing!(
        "TIMING: parser_dwa::finalize::prune_unreachable {:?} changed={}",
        prune_unreachable_time,
        unreach_changed
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::finalize::prune_unreachable = {:?}",
        prune_unreachable_time
    );
    let after_unreachable_len = combined_nwa.states.len();

    // Now subtract final weights from outgoing transitions on the much smaller NWA (~45K states).
    let prune_final_start = std::time::Instant::now();
    let skip_final_subtract = std::env::var("PARSER_DWA_SKIP_FINAL_SUBTRACTION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !skip_final_subtract {
        combined_nwa.subtract_final_weights_from_outgoing();
    }
    crate::debug!(5, "subtract_final_weights_from_outgoing in {:?}", prune_final_start.elapsed());
    crate::timing!(
        "TIMING: parser_dwa::finalize::subtract_final_weights_from_outgoing {:?}",
        prune_final_start.elapsed()
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::finalize::subtract_final_weights_from_outgoing = {:?}",
        prune_final_start.elapsed()
    );
    if skip_final_subtract {
        crate::debug!(4, "Skipped pruning continuations from final states (PARSER_DWA_SKIP_FINAL_SUBTRACTION=1). NWA now {}.", combined_nwa.stats());
    } else {
        crate::debug!(4, "Pruned continuations from final states. NWA now {}.", combined_nwa.stats());
    }

    // After weight subtraction, prune dead ends (states that can't reach any final state).
    let prune_start = std::time::Instant::now();
    let skip_prune_dead_ends = std::env::var("PARSER_DWA_SKIP_PRUNE_DEAD_ENDS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let dead_changed = if skip_prune_dead_ends {
        false
    } else {
        combined_nwa.prune_dead_ends()
    };
    let prune_dead_time = prune_start.elapsed();
    crate::timing!(
        "TIMING: parser_dwa::finalize::prune_dead_ends {:?} changed={}",
        prune_dead_time,
        dead_changed
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::finalize::prune_dead_ends = {:?}",
        prune_dead_time
    );
    crate::debug!(5, "prune_unreachable in {:?}, prune_dead_ends in {:?}", prune_unreachable_time, prune_dead_time);
    crate::debug!(4, "After pruning state counts: {} -> {} -> {}",
        before_prune_len, after_unreachable_len, combined_nwa.states.len());

    // Always minimize NWA before determinization.
    
    let disable_minimize = std::env::var("PARSER_DWA_MINIMIZE")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);

    if disable_minimize {
        crate::debug!(4, "Parser DWA minimize disabled (PARSER_DWA_MINIMIZE=0); running NWA minimization before determinization");
        let before_minimize = combined_nwa.stats();
        let minimize_start = Instant::now();
        combined_nwa.minimize();
        let minimize_elapsed = minimize_start.elapsed();
        crate::debug!(4, "Parser NWA minimization: {} -> {} in {:?}", before_minimize, combined_nwa.stats(), minimize_elapsed);
        crate::timing!("TIMING: parser_dwa::finalize::nwa_minimize {:?}", minimize_elapsed);
        crate::timing!(
            "PHASE_TIMING: parser_dwa::finalize::nwa_minimize = {:?}",
            minimize_elapsed
        );
        let det_start = std::time::Instant::now();
        let dwa = combined_nwa.determinize();
        let det_elapsed = det_start.elapsed();
        crate::debug!(5, "determinize(Parser) in {:?}", det_elapsed);
        crate::timing!("TIMING: parser_dwa::finalize::determinize {:?}", det_elapsed);
        crate::timing!(
            "PHASE_TIMING: parser_dwa::finalize::determinize = {:?}",
            det_elapsed
        );
        crate::debug!(4, "Parser DWA determinize complete. {}", dwa.stats());
        crate::timing!(
            "PHASE_TIMING: parser_dwa::finalize::total = {:?}",
            finalize_total_start.elapsed()
        );
        return dwa;
    }

    crate::debug!(4, "Running parser DWA minimize");
    // Use unified determinize_and_minimize with "Parser" profile
    // Pipeline: determinize → prune_dead_ends → minimize
    let det_min_start = std::time::Instant::now();
    let _dwa_type_guard = crate::dwa_i32::minimization::graph_coloring::set_current_dwa_type(
        Some("parser"),
    );
    let mut dwa = combined_nwa.determinize_and_minimize(DeterminizeAndMinimizeProfile::Parser);
    let det_min_elapsed = det_min_start.elapsed();
    crate::debug!(5, "determinize_and_minimize(Parser) in {:?}", det_min_elapsed);

    // Propagate final_weights through default transitions.
    // This fixes a correctness bug where deeper recursion depths in rules like
    // ap_extra_0_c don't inherit the accepting capability from shallower depths.
    let propagated = dwa.propagate_final_weights_through_defaults();
    if propagated > 0 {
        crate::debug!(5, "Parser DWA: propagated final_weights through {} default transitions", propagated);
    }
    crate::timing!(
        "TIMING: parser_dwa::finalize::determinize_and_minimize {:?}",
        det_min_elapsed
    );
    crate::timing!(
        "PHASE_TIMING: parser_dwa::finalize::determinize_and_minimize = {:?}",
        det_min_elapsed
    );
    crate::debug!(4, "Parser DWA minimization complete. {}", dwa.stats());
    crate::timing!(
        "PHASE_TIMING: parser_dwa::finalize::total = {:?}",
        finalize_total_start.elapsed()
    );
    dwa
}
