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
    apply_cancellations_range, apply_finality_fixpoint_range, remove_negative_transitions_range,
    // Note: remove_redundant_default_transitions is only run in a global pass,
    // not per-range here, since it requires a global pass over all states.
};
use crate::precompute4::template_dfa::{build_ignore_terminal_dwa, build_template_dwas};
use crate::dwa_i32::{
    common::Label, DeterminizeAndMinimizeProfile, DWA, NWA, NWABody, NWAStateID, NWAStates,
    StateID, Weight,
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

pub type Signature = Vec<Vec<Option<TerminalID>>>;

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
        let dynamic_derive_us = self.dynamic_derive_us.load(Ordering::Relaxed);
        let instantiate_us = self.instantiate_us.load(Ordering::Relaxed);
        let apply_cancellations_us = self.apply_cancellations_us.load(Ordering::Relaxed);
        let apply_finality_us = self.apply_finality_us.load(Ordering::Relaxed);
        let remove_negative_us = self.remove_negative_us.load(Ordering::Relaxed);
        let union_us = self.union_us.load(Ordering::Relaxed);
        let tsid_collect_us = self.tsid_collect_us.load(Ordering::Relaxed);
        let final_collect_us = self.final_collect_us.load(Ordering::Relaxed);

        crate::debug!(
            4,
            "Pass2 profile: process_total={:?} ({} calls), templates={}, canonicalize={:?}, cache_lookup={:?}, cache_insert={:?}, dynamic_derive={:?}, instantiate={:?}, cancellations={:?}, finality_fixpoint={:?}, remove_negative={:?}, union={:?}, tsid_collect={:?}, final_collect={:?}, other={:?}",
            std::time::Duration::from_micros(process_total_us),
            process_count,
            template_count,
            std::time::Duration::from_micros(canonicalize_us),
            std::time::Duration::from_micros(cache_lookup_us),
            std::time::Duration::from_micros(cache_insert_us),
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
        eprintln!(
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
            _ => BackendChoice::Factorized,
        };
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

struct SignatureIndex {
    term_to_group: HashMap<Option<TerminalID>, usize>,
    total_terms: usize,
}

impl SignatureIndex {
    fn new(sig: &Signature) -> Self {
        let mut map = HashMap::new();
        let mut count = 0;
        for (g_idx, group) in sig.iter().enumerate() {
            for term in group { map.insert(*term, g_idx); count += 1; }
        }
        Self { term_to_group: map, total_terms: count }
    }
    fn get_group(&self, term: &Option<TerminalID>) -> Option<usize> { self.term_to_group.get(term).cloned() }
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

fn add_shift_chain(
    nwa: &mut NWA,
    end: StateID,
    from: StateID,
    pop_state: ParserStateID,
    push_state: ParserStateID,
    push_shift: ParserStateID,
    weight: Weight,
) -> Result<(), FullDWABuildError> {
    let pop_label = crate::precompute4::utils::encode_symbol_i16(pop_state)?;
    let push_label_state = crate::precompute4::utils::encode_negative_i16(push_state)?;
    let push_label_shift = crate::precompute4::utils::encode_negative_i16(push_shift)?;
    let s1 = nwa.add_state();
    let s2 = nwa.add_state();
    nwa.add_transition(from, pop_label, s1, weight.clone())?;
    nwa.add_transition(s1, push_label_state, s2, weight.clone())?;
    nwa.add_transition(s2, push_label_shift, end, weight)?;
    Ok(())
}

fn add_escape_shift_chain(
    nwa: &mut NWA,
    end: StateID,
    from: StateID,
    revealed: ParserStateID,
    goto: ParserStateID,
    shift: ParserStateID,
    weight: Weight,
) -> Result<(), FullDWABuildError> {
    let pop_label = crate::precompute4::utils::encode_symbol_i16(revealed)?;
    let push_revealed = crate::precompute4::utils::encode_negative_i16(revealed)?;
    let push_goto = crate::precompute4::utils::encode_negative_i16(goto)?;
    let push_shift = crate::precompute4::utils::encode_negative_i16(shift)?;
    let s1 = nwa.add_state();
    let s2 = nwa.add_state();
    let s3 = nwa.add_state();
    nwa.add_transition(from, pop_label, s1, weight.clone())?;
    nwa.add_transition(s1, push_revealed, s2, weight.clone())?;
    nwa.add_transition(s2, push_goto, s3, weight.clone())?;
    nwa.add_transition(s3, push_shift, end, weight)?;
    Ok(())
}

fn build_super_nwa_from_grouped(
    grouped: &GroupedCharacterization,
    all_nts: &BTreeSet<NonTerminalID>,
    term_to_bit: &BTreeMap<Option<TerminalID>, usize>,
    ignore_terms: &BTreeSet<TerminalID>,
) -> Result<NWA, FullDWABuildError> {
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

    for ((state, shift_state), terminals) in &grouped.initial_shifts {
        let weight = weight_from_terminals(terminals, term_to_bit);
        if weight.is_empty() {
            continue;
        }
        add_shift_chain(&mut nwa, end, start, *state, *state, *shift_state, weight)?;
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

        for ((revealed, goto, shift), terminals) in &nt_group.escape_shifts {
            let weight = weight_from_terminals(terminals, term_to_bit);
            if weight.is_empty() {
                continue;
            }
            add_escape_shift_chain(&mut nwa, end, nt_state, *revealed, *goto, *shift, weight)?;
        }

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

    Ok(nwa)
}

fn collect_nwa_unique_weights(nwa: &NWA) -> Vec<Weight> {
    let mut weights = HashSet::new();
    for state in &nwa.states.0 {
        if let Some(w) = &state.final_weight {
            weights.insert(w.clone());
        }
        for targets in state.transitions.values() {
            for (_, w) in targets {
                weights.insert(w.clone());
            }
        }
        for (_, w) in &state.epsilons {
            weights.insert(w.clone());
        }
    }
    weights.into_iter().collect()
}

fn build_weight_map_for_signature(
    super_weights: &[Weight],
    sig_index: &SignatureIndex,
    bit_to_term: &[Option<TerminalID>],
) -> HashMap<Weight, Weight> {
    super_weights
        .iter()
        .map(|w| {
            if w.is_all_fast() {
                return (w.clone(), Weight::all());
            }
            let mut accumulator = Weight::zeros();
            for bit in w.iter_up_to_allow_expansion(bit_to_term.len()) {
                if let Some(term) = bit_to_term.get(bit) {
                    if let Some(group_idx) = sig_index.get_group(term) {
                        accumulator.set(group_idx, true);
                    }
                }
            }
            (w.clone(), accumulator)
        })
        .collect()
}

fn specialize_nwa_with_map(parent_nwa: &NWA, weight_map: &HashMap<Weight, Weight>) -> NWA {
    let mut nwa = NWA::new_empty();
    for _ in 0..parent_nwa.states.len() {
        nwa.add_state();
    }
    nwa.body = parent_nwa.body.clone();

    for (idx, state) in parent_nwa.states.0.iter().enumerate() {
        if let Some(fw) = &state.final_weight {
            if let Some(new_fw) = weight_map.get(fw).cloned() {
                if !new_fw.is_empty() {
                    nwa.states[idx].final_weight = Some(new_fw);
                }
            }
        }
        for (label, targets) in &state.transitions {
            for (dest, w) in targets {
                if let Some(new_w) = weight_map.get(w).cloned() {
                    if !new_w.is_empty() {
                        nwa.states[idx]
                            .transitions
                            .entry(*label)
                            .or_default()
                            .push((*dest, new_w));
                    }
                }
            }
        }
        for (dest, w) in &state.epsilons {
            if let Some(new_w) = weight_map.get(w).cloned() {
                if !new_w.is_empty() {
                    nwa.states[idx].epsilons.push((*dest, new_w));
                }
            }
        }
    }

    nwa
}

pub fn canonicalize_bundle(terminal_map: BTreeMap<Option<TerminalID>, Weight>) -> (Signature, Vec<Weight>) {
    let mut weight_groups: HashMap<Weight, Vec<Option<TerminalID>>> = HashMap::new();
    for (term, weight) in terminal_map {
        if !weight.is_empty() { weight_groups.entry(weight).or_default().push(term); }
    }
    let mut groups_vec: Vec<(Weight, Vec<Option<TerminalID>>)> = weight_groups.into_iter().collect();
    for (_, terms) in &mut groups_vec { terms.sort(); }
    groups_vec.sort_by(|a, b| a.1.cmp(&b.1));
    (groups_vec.iter().map(|(_, terms)| terms.clone()).collect(), groups_vec.into_iter().map(|(w, _)| w).collect())
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
        match build_template_dwas(parser) {
            Ok(m) => m,
            Err(e) => panic!("Failed to build template DWAs: {:?}", e),
        }
    });
    eprintln!("TIMING: parser_dwa::build_template_dwas {:?}", template_dwas_start.elapsed());

    let ignore_dwa_start = Instant::now();
    let ignore_dwa = timeit!("build_ignore_terminal_dwa", {
        build_ignore_terminal_dwa()
    });
    eprintln!("TIMING: parser_dwa::build_ignore_terminal_dwa {:?}", ignore_dwa_start.elapsed());

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
    eprintln!("TIMING: parser_dwa::reverse_nwa {:?}", reverse_start.elapsed());
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
    eprintln!("TIMING: parser_dwa::compute_traversal_data {:?}", traversal_start.elapsed());

    
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
        eprintln!("TIMING: parser_dwa::outgoing_tsid_edges {:?}", outgoing_tsid_edges_start.elapsed());
    }

    let initial_tokens = Weight::all();
    let mut initial_values_bv = Vec::new();
    for &start in &reversed_nwa.body.start_states {
        initial_values_bv.push((start, initial_tokens.clone()));
    }

    let start_pass1 = Instant::now();
    let (node_tokens, mut unique_signatures) = timeit!("parser_dwa::pass1_precompute", {
        precompute_token_bvs_and_signatures(&reversed_nwa, &traversal_data, initial_values_bv)
    });
    eprintln!("TIMING: parser_dwa::pass1_precompute {:?}", start_pass1.elapsed());
    unique_signatures.insert(vec![vec![None]]);
    crate::debug!(4, "Pass 1: Tokens & Signatures ({} sigs, {:.2?})", unique_signatures.len(), start_pass1.elapsed());
    let mut unique_term_ids_in_sigs = BTreeSet::new();
    for sig in &unique_signatures {
        for terms in sig {
            for term in terms {
                if let Some(term_id) = term {
                    unique_term_ids_in_sigs.insert(term_id.0);
                }
            }
        }
    }


    let mut super_nwa_opt: Option<Arc<NWA>> = None;
    let mut super_bit_to_term_opt: Option<Vec<Option<TerminalID>>> = None;
    let mut super_nwa_unique_weights_opt: Option<Vec<Weight>> = None;

    let template_cache: RefCell<FxHashMap<Signature, Arc<NWA>>> = RefCell::new(FxHashMap::default());

    // OPTIMIZATION START: Split signatures into Simple (Direct Union) and Complex (Bitvector Derivation)
    let mut simple_signatures = Vec::new();
    let mut complex_signatures = Vec::new();

    for sig in unique_signatures {
        if sig.len() == 1 {
            simple_signatures.push(sig);
        } else {
            complex_signatures.push(sig);
        }
    }

    crate::debug!(4, "Optimization: {} simple signatures (direct build), {} complex signatures (derivation)",
        simple_signatures.len(), complex_signatures.len());

    // 1. FAST PATH: Handle simple signatures via direct Union
    // A signature of length 1 means all terminals in it map to the same logical state transition.
    // We don't need bitmasks; we just Union the Templates.
    // NOTE: Parallelizing this was tested but memory contention makes serial faster (143-169ms vs 121ms serial).

    let ignore_nwa = Arc::new(NWA::from_dwa(&ignore_dwa));
    let mut term_nwa_cache: FxHashMap<TerminalID, Arc<NWA>> = FxHashMap::default();

    let simple_signatures_start = Instant::now();
    timeit!("parser_dwa::simple_signatures", {
        for sig in simple_signatures {
            let terminals = &sig[0];
            let mut combined_nwa = NWA::new_empty();

            // If there are many terminals, this might look expensive, but NWA union is cheap (just adding edges/start states).
            // Determinization handles the complexity.
            for term_opt in terminals {
                let term_nwa = match term_opt {
                    Some(term_id) => {
                        if parser.ignore_terminal_ids.contains(term_id) {
                            Arc::clone(&ignore_nwa)
                        } else {
                            term_nwa_cache
                                .entry(*term_id)
                                .or_insert_with(|| {
                                    Arc::new(NWA::from_dwa(
                                        template_dwas.get(term_id).unwrap_or(&ignore_dwa)
                                    ))
                                })
                                .clone()
                        }
                    }
                    None => Arc::clone(&ignore_nwa),
                };
                NWA::union_assign(&mut combined_nwa, term_nwa.as_ref());
            }

            // Always minimize NWA before determinization.
            let minimize_start = Instant::now();
            combined_nwa.minimize();
            crate::debug!(5, "Simple signature NWA minimize in {:?}", minimize_start.elapsed());
            let mut dwa = combined_nwa.determinize();
            dwa.prune_basic();

            template_cache.borrow_mut().insert(sig, Arc::new(NWA::from_dwa(&dwa)));
        }
    });
    eprintln!("TIMING: parser_dwa::simple_signatures {:?}", simple_signatures_start.elapsed());

    // 2. SLOW PATH: Handle complex signatures via Super NWA, then det/min.
    if !complex_signatures.is_empty() {
        let complex_signatures_start = Instant::now();
        timeit!("parser_dwa::complex_signatures", {
            crate::debug!(4, "Building Super NWA for {} complex signatures", complex_signatures.len());

            let mut used_terminals: BTreeSet<TerminalID> = BTreeSet::new();
            for sig in &complex_signatures {
                for group in sig {
                    for term in group {
                        if let Some(term) = term {
                            used_terminals.insert(*term);
                        }
                    }
                }
            }
            crate::debug!(5, "  Used terminals: {}", used_terminals.len());

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

            let _rangeset_backend = WeightBackendOverride::new("rangeset");

            let all_chars = compute_all_characterizations(parser);
            let grouped = GroupedCharacterization::from_terminals(&all_chars, &used_nonignore_terms);
            let all_nts: BTreeSet<NonTerminalID> =
                parser.non_terminal_map.right_values().cloned().collect();

            let super_nwa = build_super_nwa_from_grouped(
                &grouped,
                &all_nts,
                &term_to_bit,
                &ignore_terms,
            )
                .expect("Failed to build Super NWA");
            crate::debug!(5, "  Super NWA: {}", super_nwa.stats());

            let super_nwa_unique_weights = collect_nwa_unique_weights(&super_nwa);
            let super_nwa = Arc::new(super_nwa);
            let super_nwa_ref = super_nwa.as_ref();

            let mut results = Vec::new();
            for sig in &complex_signatures {
                let sig_index = SignatureIndex::new(sig);
                let weight_map = build_weight_map_for_signature(
                    &super_nwa_unique_weights,
                    &sig_index,
                    &bit_to_term,
                );
                let nwa = specialize_nwa_with_map(super_nwa_ref, &weight_map);
                let dwa = nwa.determinize_and_minimize(
                    DeterminizeAndMinimizeProfile::SpecializedSuper,
                );
                let nwa = NWA::from_dwa(&dwa);
                results.push((sig.clone(), nwa));
            }

            for (sig, nwa) in results {
                template_cache.borrow_mut().insert(sig, Arc::new(nwa));
            }

            super_nwa_opt = Some(Arc::clone(&super_nwa));
            super_bit_to_term_opt = Some(bit_to_term);
            super_nwa_unique_weights_opt = Some(super_nwa_unique_weights);
        });
        eprintln!("TIMING: parser_dwa::complex_signatures {:?}", complex_signatures_start.elapsed());
    }
    // OPTIMIZATION END

    // Log template cache stats
    let template_cache_snapshot = template_cache.borrow();
    let template_sizes: Vec<usize> = template_cache_snapshot.values().map(|nwa| nwa.states.len()).collect();
    let total_template_states: usize = template_sizes.iter().sum();
    let max_template: usize = template_sizes.iter().copied().max().unwrap_or(0);
    let avg_template: f64 = total_template_states as f64 / template_sizes.len().max(1) as f64;
    crate::debug!(4, "Template cache: {} templates, {} total states, max={}, avg={:.1}", 
        template_cache_snapshot.len(), total_template_states, max_template, avg_template);
    drop(template_cache_snapshot);

    crate::debug!(4, "Finished DWA specialization");

    let states_arena = RefCell::new(NWAStates::default());
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

    // Clone references for use in closures
    let tsid_bodies_for_process = tsid_bodies_arc.clone();
    let pass2_profile_for_process = pass2_profile.clone();
    let template_cache_ref = &template_cache;
    let super_nwa_opt_ref = &super_nwa_opt;
    let super_bit_to_term_opt_ref = &super_bit_to_term_opt;
    let super_nwa_unique_weights_opt_ref = &super_nwa_unique_weights_opt;
    let states_arena_ref = &states_arena;

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
                    let canon_start = Instant::now();
                    let (signature, concrete_weights) = canonicalize_bundle(terminal_map.clone());
                    let canon_us = canon_start.elapsed().as_micros() as u64;
                    pass2_profile_for_process
                        .canonicalize_us
                        .fetch_add(canon_us, Ordering::Relaxed);
                    accounted_us = accounted_us.saturating_add(canon_us);

                    let cached_nwa = {
                        let cache_lookup_start = Instant::now();
                        let cached = template_cache_ref.borrow().get(&signature).cloned();
                        let cache_lookup_us = cache_lookup_start.elapsed().as_micros() as u64;
                        pass2_profile_for_process.cache_lookup_us.fetch_add(
                            cache_lookup_us,
                            Ordering::Relaxed,
                        );
                        accounted_us = accounted_us.saturating_add(cache_lookup_us);
                        if let Some(nwa) = cached { nwa } else {
                        let dynamic_start = Instant::now();
                        let _rangeset_backend = WeightBackendOverride::new("rangeset");
                        crate::debug!(5, "Dynamic derivation for signature {:?}", signature);

                        let super_nwa = super_nwa_opt_ref
                            .as_ref()
                            .expect("Super NWA missing for dynamic derivation");
                        let bit_to_term = super_bit_to_term_opt_ref
                            .as_ref()
                            .expect("Super NWA bit mapping missing for dynamic derivation");
                        let super_nwa_unique_weights = super_nwa_unique_weights_opt_ref
                            .as_ref()
                            .expect("Super NWA weights missing for dynamic derivation");

                        let sig_index = SignatureIndex::new(&signature);
                        let weight_map = build_weight_map_for_signature(
                            super_nwa_unique_weights,
                            &sig_index,
                            bit_to_term,
                        );

                        let nwa = Arc::new(specialize_nwa_with_map(super_nwa.as_ref(), &weight_map));
                        let dynamic_us = dynamic_start.elapsed().as_micros() as u64;
                        pass2_profile_for_process.dynamic_derive_us.fetch_add(
                            dynamic_us,
                            Ordering::Relaxed,
                        );
                        accounted_us = accounted_us.saturating_add(dynamic_us);
                        let cache_insert_start = Instant::now();
                        template_cache_ref
                            .borrow_mut()
                            .insert(signature.clone(), nwa.clone());
                        let cache_insert_us = cache_insert_start.elapsed().as_micros() as u64;
                        pass2_profile_for_process.cache_insert_us.fetch_add(
                            cache_insert_us,
                            Ordering::Relaxed,
                        );
                        accounted_us = accounted_us.saturating_add(cache_insert_us);
                        nwa
                        }
                    };
                    let cached_nwa = cached_nwa.as_ref();


                    pass2_profile_for_process
                        .template_count
                        .fetch_add(1, Ordering::Relaxed);
                    
                    let template_size = cached_nwa.states.len();
                    let count = INSTANTIATE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    TOTAL_TEMPLATE_STATES.fetch_add(template_size, std::sync::atomic::Ordering::Relaxed);
                    if count % 10000 == 0 {
                        crate::debug!(4, "Template instantiation #{}: {} total states so far, template size {}, bodies_count: {}", 
                            count, TOTAL_TEMPLATE_STATES.load(std::sync::atomic::Ordering::Relaxed), template_size, bodies_count);
                    }
                    
                    let mut states = states_arena.borrow_mut();
                    let arena_start = states.len();
                    let instantiate_start = Instant::now();
                    let composed_body = instantiate_nwa_template_into(cached_nwa, &concrete_weights, &mut states, right_body);
                    let instantiate_us = instantiate_start.elapsed().as_micros() as u64;
                    pass2_profile_for_process.instantiate_us.fetch_add(
                        instantiate_us,
                        Ordering::Relaxed,
                    );
                    accounted_us = accounted_us.saturating_add(instantiate_us);
                    // Batch negative resolution runs after pass2 completes.
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
        eprintln!(
            "TIMING: parser_dwa::pass2_profile process_total={:?}, canonicalize={:?}, cache_lookup={:?}, cache_insert={:?}, dynamic_derive={:?}, instantiate={:?}, cancellations={:?}, finality_fixpoint={:?}, remove_negative={:?}, union={:?}, tsid_collect={:?}, final_collect={:?}, other={:?}, templates={}",
            std::time::Duration::from_micros(pass2_profile.process_total_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.canonicalize_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.cache_lookup_us.load(Ordering::Relaxed)),
            std::time::Duration::from_micros(pass2_profile.cache_insert_us.load(Ordering::Relaxed)),
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
    eprintln!("TIMING: parser_dwa::pass2_traversal {:?}", pass2_start.elapsed());

    // Batch negative resolution over the full arena after pass2 completes.
    let batch_negatives_start = Instant::now();
    {
        let mut states = states_arena.borrow_mut();
        let arena_len = states.len();
        if arena_len > 0 {
            let has_negatives = states.0.iter().any(|state| {
                state
                    .transitions
                    .keys()
                    .any(|&label| label < 0 && label != crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL)
            });
            if has_negatives {
                let cancel_start = Instant::now();
                apply_cancellations_range(&mut states, 0..arena_len);
                let cancel_us = cancel_start.elapsed().as_micros() as u64;
                pass2_profile
                    .apply_cancellations_us
                    .fetch_add(cancel_us, Ordering::Relaxed);

                let finality_start = Instant::now();
                apply_finality_fixpoint_range(&mut states, 0..arena_len);
                let finality_us = finality_start.elapsed().as_micros() as u64;
                pass2_profile
                    .apply_finality_us
                    .fetch_add(finality_us, Ordering::Relaxed);

                let remove_start = Instant::now();
                remove_negative_transitions_range(&mut states, 0..arena_len);
                let remove_us = remove_start.elapsed().as_micros() as u64;
                pass2_profile
                    .remove_negative_us
                    .fetch_add(remove_us, Ordering::Relaxed);
            }
        }
    }
    eprintln!(
        "TIMING: parser_dwa::batch_negative_resolution {:?}",
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
    eprintln!("TIMING: parser_dwa::combine_bodies {:?}", combine_bodies_start.elapsed());

    let macro_level = std::env::var("MACRO_DEBUG_LEVEL")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    if macro_level >= 4 {
        let expected_num_tsids = crate::datastructures::abstract_weight::current_num_tsids();
        validate_nwa_weight_dims(&combined_nwa, expected_num_tsids);
    }
    crate::debug!(3, "Combined NWA before determinization: {}, is_symbol_heavy={}", 
        combined_nwa.stats(), is_symbol_heavy);
    let finalize_start = Instant::now();
    let mut final_dwa = timeit!("parser_dwa::finalize_and_determinize", {
        finalize_and_optimize_and_determinize(parser, combined_nwa)
    });
    eprintln!("TIMING: parser_dwa::finalize_and_determinize {:?}", finalize_start.elapsed());
    // SKIP final minimization to test performance impact
    // final_dwa.minimize();
    crate::debug!(4, "Parser DWA construction complete. Stats: {}", final_dwa.stats());
    if let Some(avg_path_len) = final_dwa.average_path_length() {
        crate::debug!(4, "Parser DWA average path length: {:.2}", avg_path_len);
    }

    crate::debug!(5, "build_parser_dwa: end");
    final_dwa
}

/// Deprecated alias for build_parser_dwa
#[deprecated(since = "0.3.0", note = "Use build_parser_dwa instead")]
pub fn precompute4(parser: &GLRParser, terminal_nwa: &NWA) -> DWA {
    build_parser_dwa(parser, terminal_nwa)
}

pub fn precompute_token_bvs_and_signatures(reversed_nwa: &NWA, traversal_data: &NwaTraversalData, initial_values: Vec<(StateID, Weight)>) -> (HashMap<StateID, Weight>, HashSet<Signature>) {
    let node_tokens: Arc<Mutex<HashMap<StateID, Weight>>> = Arc::new(Mutex::new(HashMap::new()));
    let signatures: Arc<Mutex<HashSet<Signature>>> = Arc::new(Mutex::new(HashSet::new()));

    let node_tokens_clone = node_tokens.clone();
    let signatures_clone = signatures.clone();

    nwa_special_map(reversed_nwa, traversal_data, initial_values,
        move |tokens: &Weight, _edge_label, transitions| {
            let mut results = Vec::new();
            for (dest_id, weight) in transitions {
                let next = tokens & weight;
                if !next.is_empty() { results.push((*dest_id, next)); }
            }
            results
        },
        |t1, t2| {
            if t2.is_subset_of(t1) {
                false
            } else {
                *t1 |= &t2;
                true
            }
        },
        move |node_id, tokens| {
            node_tokens_clone.lock().unwrap().insert(node_id, tokens.clone());
            let mut bundles_by_dest: HashMap<StateID, BTreeMap<Option<TerminalID>, Weight>> = HashMap::new();
            let state = &reversed_nwa.states[node_id];
            for (label, targets) in &state.transitions {
                let term = Some(TerminalID(*label as usize));
                for (v, w) in targets {
                    let combined = &tokens & w;
                    if !combined.is_empty() {
                        bundles_by_dest.entry(*v).or_default().insert(term, w.clone());
                    }
                }
            }
            for (v, w) in &state.epsilons {
                let combined = &tokens & w;
                if !combined.is_empty() {
                    bundles_by_dest.entry(*v).or_default().insert(None, w.clone());
                }
            }
            let mut sigs = signatures_clone.lock().unwrap();
            for (_, bundle) in bundles_by_dest {
                let (sig, _) = canonicalize_bundle(bundle);
                sigs.insert(sig);
            }
            Some(tokens)
        },
        None,
    );
    (Arc::try_unwrap(node_tokens).unwrap().into_inner().unwrap(), Arc::try_unwrap(signatures).unwrap().into_inner().unwrap())
}

pub fn finalize_and_optimize_and_determinize(parser: &GLRParser, mut combined_nwa: NWA) -> DWA {
    crate::debug!(4, "Pruning continuations from final states for NWA with {}...", combined_nwa.stats());
    let prune_final_start = std::time::Instant::now();
    combined_nwa.subtract_final_weights_from_outgoing();
    crate::debug!(5, "subtract_final_weights_from_outgoing in {:?}", prune_final_start.elapsed());
    eprintln!("TIMING: parser_dwa::finalize::subtract_final_weights_from_outgoing {:?}", prune_final_start.elapsed());
    crate::debug!(4, "Pruned continuations from final states. NWA now {}.", combined_nwa.stats());
    
    // After pruning continuations, some transitions may become empty and states may become unreachable.
    // Prune dead ends before determinization to reduce the NWA size significantly.
    let before_prune = combined_nwa.stats();
    let prune_start = std::time::Instant::now();
    combined_nwa.prune_dead_ends();
    let prune_dead_time = prune_start.elapsed();
    eprintln!("TIMING: parser_dwa::finalize::prune_dead_ends {:?}", prune_dead_time);
    let prune_unreachable_start = std::time::Instant::now();
    combined_nwa.prune_unreachable();
    let prune_unreachable_time = prune_unreachable_start.elapsed();
    eprintln!("TIMING: parser_dwa::finalize::prune_unreachable {:?}", prune_unreachable_time);
    crate::debug!(5, "prune_dead_ends in {:?}, prune_unreachable in {:?}", prune_dead_time, prune_unreachable_time);
    crate::debug!(4, "After pruning dead ends: NWA {} -> {}", 
        before_prune, combined_nwa.stats());

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
        eprintln!("TIMING: parser_dwa::finalize::nwa_minimize {:?}", minimize_elapsed);
        let det_start = std::time::Instant::now();
        let dwa = combined_nwa.determinize();
        let det_elapsed = det_start.elapsed();
        crate::debug!(5, "determinize(Parser) in {:?}", det_elapsed);
        eprintln!("TIMING: parser_dwa::finalize::determinize {:?}", det_elapsed);
        crate::debug!(4, "Parser DWA determinize complete. {}", dwa.stats());
        return dwa;
    }

    crate::debug!(4, "Running parser DWA minimize");
    // Use unified determinize_and_minimize with "Parser" profile
    // Pipeline: determinize → prune_dead_ends → minimize
    let det_min_start = std::time::Instant::now();
    let _dwa_type_guard = crate::dwa_i32::minimization::graph_coloring::set_current_dwa_type(
        Some("parser"),
    );
    let dwa = combined_nwa.determinize_and_minimize(DeterminizeAndMinimizeProfile::Parser);
    let det_min_elapsed = det_min_start.elapsed();
    crate::debug!(5, "determinize_and_minimize(Parser) in {:?}", det_min_elapsed);
    eprintln!("TIMING: parser_dwa::finalize::determinize_and_minimize {:?}", det_min_elapsed);
    crate::debug!(4, "Parser DWA minimization complete. {}", dwa.stats());
    dwa
}

pub fn instantiate_nwa_template_into(
    template: &NWA,
    ordered_weights: &[Weight],
    states: &mut NWAStates,
    right_body: &NWABody,
) -> NWABody {
    let offset = states.len();
    states.0.reserve(template.states.len());

    let mut union_cache: HashMap<Weight, Weight> = HashMap::new();
    let mut map_abstract_weight = |w: &Weight| -> Weight {
        if w.is_empty() { return Weight::zeros(); }
        if let Some(res) = union_cache.get(w) { return res.clone(); }
        let mut concrete = Weight::zeros();
        for idx in w.iter_up_to_allow_expansion(ordered_weights.len()) {
            if let Some(concrete_w) = ordered_weights.get(idx) {
                if matches!(concrete, Weight::Factorized(_)) {
                    if let Weight::RangeSet(rsb) = concrete_w {
                        let converted = Weight::from_rsb(rsb.inner().clone());
                        concrete |= &converted;
                        continue;
                    }
                }
                concrete |= concrete_w;
            }
        }
        union_cache.insert(w.clone(), concrete.clone());
        concrete
    };

    for old_state in &template.states.0 {
        let mut new_state = crate::dwa_i32::nwa::NWAState::default();
        
        // Transitions
        for (lbl, targets) in &old_state.transitions {
            let mut new_targets = Vec::with_capacity(targets.len());
            for (target, w) in targets {
                let concrete = map_abstract_weight(w);
                if !concrete.is_empty() {
                    new_targets.push((*target + offset, concrete));
                }
            }
            if !new_targets.is_empty() {
                new_state.transitions.insert(*lbl, new_targets);
            }
        }

        // Epsilons
        for (target, w) in &old_state.epsilons {
            let concrete = map_abstract_weight(w);
            if !concrete.is_empty() {
                new_state.epsilons.push((*target + offset, concrete));
            }
        }

        // Final Weight -> Epsilon to right_body starts
        if let Some(fw) = &old_state.final_weight {
            let concrete = map_abstract_weight(fw);
            if !concrete.is_empty() {
                for &r_start in &right_body.start_states {
                    new_state.epsilons.push((r_start, concrete.clone()));
                }
            }
        }

        states.0.push(new_state);
    }

    NWABody {
        start_states: template.body.start_states.iter().map(|s| s + offset).collect()
    }
}

fn minimize_remove_epsilon(nwa: &mut NWA) {
    nwa.minimize()
}