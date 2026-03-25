use std::collections::{BTreeMap, BTreeSet, HashMap};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::ds::{bitset::BitSet, u8set::U8Set};

use super::ast::Expr;
use super::dfa::{DFA, DEAD};
use super::nfa::NFA;

fn common_prefix_factor(exprs: &[Expr]) -> Option<(Expr, Vec<Expr>)> {
    fn candidate_prefix(expr: &Expr) -> Option<&Expr> {
        match expr {
            Expr::Seq(parts) if !parts.is_empty() => Some(&parts[0]),
            Expr::Shared(inner) => candidate_prefix(inner),
            _ => None,
        }
    }

    let prefix = candidate_prefix(exprs.first()?)?.clone();
    let mut remainders = Vec::with_capacity(exprs.len());
    for expr in exprs {
        remainders.push(expr.strip_prefix(&prefix)?);
    }
    Some((prefix, remainders))
}

fn expr_contains_exclude(expr: &Expr) -> bool {
    match expr {
        Expr::Exclude { .. } => true,
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_contains_exclude),
        Expr::Repeat { expr, .. } => expr_contains_exclude(expr),
        Expr::Shared(inner) => expr_contains_exclude(inner),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
    }
}

fn split_top_level_exclusions(expr: &Expr) -> (Expr, Vec<Expr>) {
    match expr {
        Expr::Exclude { expr, exclude } => {
            let (base, mut excluded) = split_top_level_exclusions(expr);
            excluded.push((**exclude).clone());
            (base, excluded)
        }
        Expr::Shared(inner) if matches!(inner.as_ref(), Expr::Exclude { .. }) => {
            split_top_level_exclusions(inner.as_ref())
        }
        _ => (expr.clone(), Vec::new()),
    }
}

struct ExclusionCompilePlan {
    compiled_exprs: Vec<Expr>,
    exclusions: BTreeMap<u32, BTreeSet<u32>>,
    visible_groups: usize,
}

fn build_exclusion_compile_plan(exprs: &[Expr]) -> ExclusionCompilePlan {
    let visible_groups = exprs.len();
    let mut compiled_exprs = Vec::with_capacity(visible_groups);
    let mut deferred_exclusions = Vec::<Vec<Expr>>::with_capacity(visible_groups);

    for expr in exprs {
        let (base, excluded) = split_top_level_exclusions(expr);
        assert!(
            !expr_contains_exclude(&base),
            "Expr::Exclude is currently only supported at the top level of a terminal expression"
        );
        for excluded_expr in &excluded {
            assert!(
                !expr_contains_exclude(excluded_expr),
                "nested Expr::Exclude inside an exclusion branch is not supported"
            );
        }
        compiled_exprs.push(base);
        deferred_exclusions.push(excluded);
    }

    let mut exclusions = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut next_group = visible_groups as u32;
    for (group_id, excluded_exprs) in deferred_exclusions.into_iter().enumerate() {
        if excluded_exprs.is_empty() {
            continue;
        }
        let entry = exclusions.entry(group_id as u32).or_default();
        for excluded_expr in excluded_exprs {
            compiled_exprs.push(excluded_expr);
            entry.insert(next_group);
            next_group += 1;
        }
    }

    ExclusionCompilePlan {
        compiled_exprs,
        exclusions,
        visible_groups,
    }
}

fn expr_accepts_empty(expr: &Expr) -> bool {
    match expr {
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::U8Class(_) => false,
        Expr::Dfa(dfa) => !dfa.finalizers(0).is_empty(),
        Expr::Seq(parts) => parts.iter().all(expr_accepts_empty),
        Expr::Choice(options) => options.iter().any(expr_accepts_empty),
        Expr::Exclude { expr, exclude } => expr_accepts_empty(expr) && !expr_accepts_empty(exclude),
        Expr::Repeat { expr: _, min, .. } => *min == 0,
        Expr::Shared(inner) => expr_accepts_empty(inner),
        Expr::Epsilon => true,
    }
}

fn expr_u8set(expr: &Expr) -> U8Set {
    match expr {
        Expr::U8Seq(bytes) => U8Set::from_bytes(bytes),
        Expr::U8Class(set) => *set,
        Expr::Dfa(dfa) => dfa.get_u8set(0),
        Expr::Seq(parts) | Expr::Choice(parts) => parts
            .iter()
            .fold(U8Set::empty(), |acc, part| acc | expr_u8set(part)),
        Expr::Exclude { expr, .. } => expr_u8set(expr),
        Expr::Repeat { expr, .. } => expr_u8set(expr),
        Expr::Shared(inner) => expr_u8set(inner),
        Expr::Epsilon => U8Set::empty(),
    }
}

fn highest_power_of_two_leq(value: usize) -> usize {
    debug_assert!(value > 0);
    1usize << (usize::BITS - value.leading_zeros() - 1)
}

struct RepeatCompiler<'expr, 'nfa> {
    expr: &'expr Expr,
    nfa: &'nfa mut NFA,
    power_cache: HashMap<(usize, u32), u32>,
    upto_cache: HashMap<(usize, u32), u32>,
}

impl<'expr, 'nfa> RepeatCompiler<'expr, 'nfa> {
    fn new(expr: &'expr Expr, nfa: &'nfa mut NFA) -> Self {
        Self {
            expr,
            nfa,
            power_cache: HashMap::new(),
            upto_cache: HashMap::new(),
        }
    }

    fn compile_power(&mut self, copies: usize, end: u32) -> u32 {
        debug_assert!(copies.is_power_of_two());

        if let Some(&start) = self.power_cache.get(&(copies, end)) {
            return start;
        }

        let start = if copies == 1 {
            let start = self.nfa.add_state();
            append_compiled_expr(self.expr, self.nfa, start, end);
            start
        } else {
            let half = copies / 2;
            let suffix_start = self.compile_power(half, end);
            self.compile_power(half, suffix_start)
        };

        self.power_cache.insert((copies, end), start);
        start
    }

    fn compile_exact(&mut self, copies: usize, end: u32) -> u32 {
        if copies == 0 {
            return end;
        }

        let largest_power = highest_power_of_two_leq(copies);
        let suffix_start = self.compile_exact(copies - largest_power, end);
        self.compile_power(largest_power, suffix_start)
    }

    fn compile_upto(&mut self, copies: usize, end: u32) -> u32 {
        if copies == 0 {
            return end;
        }

        if let Some(&start) = self.upto_cache.get(&(copies, end)) {
            return start;
        }

        let largest_power = highest_power_of_two_leq(copies);
        let split = self.nfa.add_state();

        let smaller_start = self.compile_upto(largest_power - 1, end);
        self.nfa.add_epsilon(split, smaller_start);

        let suffix_start = self.compile_upto(copies - largest_power, end);
        let power_start = self.compile_power(largest_power, suffix_start);
        self.nfa.add_epsilon(split, power_start);

        self.upto_cache.insert((copies, end), split);
        split
    }
}

fn append_byte_sequence_expr(bytes: &[u8], nfa: &mut NFA, start: u32, end: u32) {
    let mut state = start;
    for (index, &byte) in bytes.iter().enumerate() {
        let next = if index + 1 == bytes.len() {
            end
        } else {
            nfa.add_state()
        };
        nfa.add_transition(state, byte, next);
        state = next;
    }

    if bytes.is_empty() {
        nfa.add_epsilon(start, end);
    }
}

fn append_dfa_expr(dfa: &DFA, nfa: &mut NFA, start: u32, end: u32) {
    let mut state_map = Vec::with_capacity(dfa.num_states());
    for _ in 0..dfa.num_states() {
        state_map.push(nfa.add_state());
    }
    nfa.add_epsilon(start, state_map[0]);

    for (state_id, state) in dfa.states().iter().enumerate() {
        let mapped_state = state_map[state_id];
        for (byte, &target) in state.transitions.iter() {
            nfa.add_transition(mapped_state, byte, state_map[target as usize]);
        }
        if !state.finalizers.is_empty() {
            nfa.add_epsilon(mapped_state, end);
        }
    }
}

fn append_group_dfa_expr(dfa: &DFA, nfa: &mut NFA, start: u32, group_id: u32) {
    let mut state_map = Vec::with_capacity(dfa.num_states());
    state_map.push(start);
    for _ in 1..dfa.num_states() {
        state_map.push(nfa.add_state());
    }

    for (state_id, state) in dfa.states().iter().enumerate() {
        let mapped_state = state_map[state_id];
        for (byte, &target) in state.transitions.iter() {
            nfa.add_transition(mapped_state, byte, state_map[target as usize]);
        }
        if !state.finalizers.is_empty() {
            nfa.add_finalizer(mapped_state, group_id);
        }
    }
}

fn dfa_start_is_entry_only(dfa: &DFA) -> bool {
    dfa.states()
        .iter()
        .all(|state| state.transitions.iter().all(|(_, &target)| target != 0))
}

fn append_sequence_expr(parts: &[Expr], nfa: &mut NFA, start: u32, end: u32) {
    let mut state = start;
    for (index, part) in parts.iter().enumerate() {
        let next = if index + 1 == parts.len() {
            end
        } else {
            nfa.add_state()
        };
        append_compiled_expr(part, nfa, state, next);
        state = next;
    }

    if parts.is_empty() {
        nfa.add_epsilon(start, end);
    }
}

fn append_choice_expr(options: &[Expr], nfa: &mut NFA, start: u32, end: u32) {
    if options.is_empty() {
        nfa.add_epsilon(start, end);
        return;
    }

    for option in options {
        append_compiled_expr(option, nfa, start, end);
    }
}

const DIRECT_BOUNDED_REPEAT_THRESHOLD: usize = 32;

fn compile_expr_to_dfa(expr: &Expr) -> DFA {
    let mut nfa = build_regex_nfa_impl(std::slice::from_ref(expr), false);
    nfa.condense_epsilon_sccs();
    nfa.to_dfa().minimize()
}

fn productive_dfa_states(dfa: &DFA) -> Vec<bool> {
    let mut reverse_edges = vec![Vec::new(); dfa.num_states()];
    for (state_id, state) in dfa.states().iter().enumerate() {
        for (_, &target) in state.transitions.iter() {
            reverse_edges[target as usize].push(state_id as u32);
        }
    }

    let mut productive = vec![false; dfa.num_states()];
    let mut stack = Vec::new();
    for state_id in 0..dfa.num_states() as u32 {
        if !dfa.finalizers(state_id).is_empty() {
            productive[state_id as usize] = true;
            stack.push(state_id);
        }
    }

    while let Some(state_id) = stack.pop() {
        for &pred in &reverse_edges[state_id as usize] {
            if !productive[pred as usize] {
                productive[pred as usize] = true;
                stack.push(pred);
            }
        }
    }

    productive
}

fn dfa_is_nonnullable_and_prefix_free(dfa: &DFA) -> bool {
    if !dfa.finalizers(0).is_empty() {
        return false;
    }

    let productive = productive_dfa_states(dfa);
    for state in dfa.states() {
        if state.finalizers.is_empty() {
            continue;
        }
        for (_, &target) in state.transitions.iter() {
            if productive[target as usize] {
                return false;
            }
        }
    }

    true
}

fn build_bounded_repeat_dfa(expr: &Expr, min: usize, max: usize) -> Option<DFA> {
    if max < DIRECT_BOUNDED_REPEAT_THRESHOLD {
        return None;
    }

    let base_dfa = compile_expr_to_dfa(expr);
    if base_dfa.num_states() == 0 || !dfa_is_nonnullable_and_prefix_free(&base_dfa) {
        return None;
    }

    let base_states = base_dfa.states();
    let base_state_count = base_states.len();
    let total_states = (max + 1).checked_mul(base_state_count)?;
    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(1);

    for copies_done in 0..=max {
        for (state_id, state) in base_states.iter().enumerate() {
            let mapped_state = (copies_done * base_state_count + state_id) as u32;
            let mut finalizers = crate::ds::bitset::BitSet::new(1);
            if state_id == 0 && copies_done >= min {
                finalizers.set(0);
            }
            dfa.overwrite_state_metadata(mapped_state, finalizers, crate::ds::bitset::BitSet::new(1));

            if copies_done == max || !base_dfa.finalizers(state_id as u32).is_empty() {
                continue;
            }

            for (byte, &target) in state.transitions.iter() {
                let mapped_target = if !base_dfa.finalizers(target).is_empty() {
                    ((copies_done + 1) * base_state_count) as u32
                } else {
                    (copies_done * base_state_count + target as usize) as u32
                };
                dfa.add_transition(mapped_state, byte, mapped_target);
            }
        }
    }

    Some(dfa)
}

fn append_bounded_repeat_expr(expr: &Expr, min: usize, max: usize, nfa: &mut NFA, start: u32, end: u32) {
    if max < min {
        return;
    }

    if let Some(dfa) = build_bounded_repeat_dfa(expr, min, max) {
        append_dfa_expr(&dfa, nfa, start, end);
        return;
    }

    let mut repeat_compiler = RepeatCompiler::new(expr, nfa);
    let optional = max - min;
    let tail_start = repeat_compiler.compile_upto(optional, end);
    let repeat_start = repeat_compiler.compile_exact(min, tail_start);
    repeat_compiler.nfa.add_epsilon(start, repeat_start);
}

fn append_unbounded_repeat_expr(
    expr: &Expr,
    min: usize,
    nfa: &mut NFA,
    start: u32,
    end: u32,
) {
    let mut current = start;
    for _ in 0..min {
        let next = nfa.add_state();
        append_compiled_expr(expr, nfa, current, next);
        current = next;
    }

    if current == start {
        let fresh = nfa.add_state();
        nfa.add_epsilon(start, fresh);
        current = fresh;
    }

    nfa.add_epsilon(current, end);
    let loop_state = nfa.add_state();
    append_compiled_expr(expr, nfa, current, loop_state);
    nfa.add_epsilon(loop_state, current);
    if expr_accepts_empty(expr) {
        nfa.add_epsilon(loop_state, end);
    }
}

fn append_compiled_expr(expr: &Expr, nfa: &mut NFA, start: u32, end: u32) {
    match expr {
        Expr::U8Seq(bytes) => append_byte_sequence_expr(bytes, nfa, start, end),
        Expr::U8Class(set) => {
            nfa.add_u8set_transition(start, *set, end);
        }
        Expr::Dfa(dfa) => append_dfa_expr(dfa, nfa, start, end),
        Expr::Seq(parts) => append_sequence_expr(parts, nfa, start, end),
        Expr::Choice(options) => append_choice_expr(options, nfa, start, end),
        Expr::Exclude { .. } => {
            unreachable!("nested Expr::Exclude must be lowered before NFA compilation")
        }
        Expr::Repeat { expr, min, max } => match max {
            Some(max) => append_bounded_repeat_expr(expr, *min, *max, nfa, start, end),
            None => append_unbounded_repeat_expr(expr, *min, nfa, start, end),
        },
        Expr::Shared(inner) => append_compiled_expr(inner, nfa, start, end),
        Expr::Epsilon => nfa.add_epsilon(start, end),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regex {
    pub(crate) dfa: DFA,
}

impl Regex {
    pub fn num_states(&self) -> usize {
        self.dfa.num_states()
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    pub fn get_u8set(&self, state: u32) -> U8Set {
        self.dfa.get_u8set(state)
    }
}

impl Expr {
    pub fn build(self) -> Regex {
        build_regex(&[self])
    }
}

/// Compile multiple expressions into a single multi-group [`Regex`].
///
/// Each expression's index becomes its group ID in the resulting DFA.
pub fn build_regex(exprs: &[Expr]) -> Regex {
    build_regex_with_profile_label(exprs, "default")
}

fn product_state_metadata(component_dfas: &[DFA], state_tuple: &[u32]) -> (BitSet, BitSet) {
    let num_groups = component_dfas.len();
    let mut finalizers = BitSet::new(num_groups);
    let mut future = BitSet::new(num_groups);

    for (group_id, (&state, dfa)) in state_tuple.iter().zip(component_dfas.iter()).enumerate() {
        if state == DEAD {
            continue;
        }
        if dfa.finalizers(state).contains(0) {
            finalizers.set(group_id);
        }
        if dfa.possible_future_group_ids(state).contains(0) {
            future.set(group_id);
        }
    }

    (finalizers, future)
}

fn build_product_dfa(exprs: &[Expr], profile_label: &str, debug_profile: bool) -> DFA {
    let component_dfas: Vec<DFA> = exprs
        .iter()
        .enumerate()
        .map(|(group_id, expr)| {
            let dfa = build_regex_with_profile_label(std::slice::from_ref(expr), profile_label).dfa;
            if debug_profile {
                eprintln!(
                    "[glrmask/debug][product] group={} dfa_states={}",
                    group_id,
                    dfa.num_states(),
                );
            }
            dfa
        })
        .collect();
    let num_groups = component_dfas.len();
    let (class_map, class_members) = compute_product_equivalence_classes(&component_dfas);
    let num_classes = class_members.len();
    let component_class_transitions = build_product_class_transitions(&component_dfas, &class_map);

    if debug_profile {
        eprintln!(
            "[glrmask/debug][product] alphabet_classes={}",
            num_classes,
        );
    }

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(num_groups);

    let start_tuple = vec![0u32; num_groups];
    let (start_finalizers, start_future) = product_state_metadata(&component_dfas, &start_tuple);
    dfa.overwrite_state_metadata(0, start_finalizers, start_future);

    let mut state_map = FxHashMap::<Vec<u32>, u32>::default();
    let mut worklist = Vec::new();
    let mut transitions_by_class = vec![None::<Vec<u32>>; num_classes];
    let mut used_classes = Vec::<usize>::new();
    let mut live_state_counts = vec![0u64; num_groups];
    let mut processed_product_states = 0u64;
    let mut total_live_groups = 0u64;
    let mut max_live_groups = 0usize;
    state_map.insert(start_tuple.clone(), 0);
    worklist.push((0, start_tuple));

    while let Some((current_state, state_tuple)) = worklist.pop() {
        processed_product_states += 1;
        let mut live_groups = 0usize;
        for (group_id, &component_state) in state_tuple.iter().enumerate() {
            if component_state == DEAD {
                continue;
            }
            live_state_counts[group_id] += 1;
            live_groups += 1;

            for &(class_id, target) in &component_class_transitions[group_id][component_state as usize] {
                let class_index = class_id as usize;
                if transitions_by_class[class_index].is_none() {
                    transitions_by_class[class_index] = Some(vec![DEAD; num_groups]);
                    used_classes.push(class_index);
                }
                let next_tuple = transitions_by_class[class_index]
                    .as_mut()
                    .expect("class transition bucket initialized");
                next_tuple[group_id] = target;
            }
        }
        total_live_groups += live_groups as u64;
        max_live_groups = max_live_groups.max(live_groups);

        let mut transitions = Vec::new();
        for &class_index in &used_classes {
            let next_tuple = transitions_by_class[class_index]
                .take()
                .expect("used class transition bucket populated");
            let next_state = if let Some(&existing) = state_map.get(&next_tuple) {
                existing
            } else {
                let new_state = dfa.add_state();
                let (finalizers, future) = product_state_metadata(&component_dfas, &next_tuple);
                dfa.overwrite_state_metadata(new_state, finalizers, future);
                state_map.insert(next_tuple.clone(), new_state);
                worklist.push((new_state, next_tuple));
                new_state
            };

            for &byte in &class_members[class_index] {
                transitions.push((byte, next_state));
            }
        }
        used_classes.clear();

        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|entry| entry.0);
        }
        dfa.set_transitions_from_sorted_entries(current_state, transitions);
    }

    if debug_profile {
        let avg_live_groups = if processed_product_states == 0 {
            0.0
        } else {
            total_live_groups as f64 / processed_product_states as f64
        };
        eprintln!(
            "[glrmask/debug][product] reachable_states={} avg_live_groups={:.2} max_live_groups={}",
            processed_product_states,
            avg_live_groups,
            max_live_groups,
        );
        for (group_id, alive_states) in live_state_counts.iter().enumerate() {
            let alive_ratio = if processed_product_states == 0 {
                0.0
            } else {
                *alive_states as f64 / processed_product_states as f64
            };
            eprintln!(
                "[glrmask/debug][product] group={} alive_states={} alive_ratio={:.4}",
                group_id,
                alive_states,
                alive_ratio,
            );
        }
    }

    dfa
}

fn compute_product_equivalence_classes(component_dfas: &[DFA]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for dfa in component_dfas {
        for state in dfa.states() {
            let mut bytes_by_target = FxHashMap::<u32, U8Set>::default();
            for (byte, &target) in state.transitions.iter() {
                bytes_by_target
                    .entry(target)
                    .and_modify(|set| {
                        set.insert(byte);
                    })
                    .or_insert_with(|| U8Set::single(byte));
            }

            for byte_set in bytes_by_target.into_values() {
                if seen_sets.insert(byte_set) {
                    partitions = refine_u8_partitions(partitions, byte_set);
                }
            }
        }
    }

    let mut class_map = vec![0u8; 256];
    let mut class_members = vec![Vec::new(); partitions.len()];
    for (class_id, partition) in partitions.iter().enumerate() {
        for byte in partition.iter() {
            class_map[byte as usize] = class_id as u8;
            class_members[class_id].push(byte);
        }
    }

    (class_map, class_members)
}

fn build_product_class_transitions(component_dfas: &[DFA], class_map: &[u8]) -> Vec<Vec<Vec<(u8, u32)>>> {
    component_dfas
        .iter()
        .map(|dfa| {
            dfa.states()
                .iter()
                .map(|state| {
                    let mut target_by_class = FxHashMap::<u8, u32>::default();
                    for (byte, &target) in state.transitions.iter() {
                        target_by_class.insert(class_map[byte as usize], target);
                    }
                    let mut entries: Vec<(u8, u32)> = target_by_class.into_iter().collect();
                    entries.sort_unstable_by_key(|entry| entry.0);
                    entries
                })
                .collect()
        })
        .collect()
}

fn refine_u8_partitions(partitions: Vec<U8Set>, split: U8Set) -> Vec<U8Set> {
    let mut next_partitions = Vec::with_capacity(partitions.len() * 2);
    for partition in partitions {
        let intersection = partition.intersection(&split);
        let difference = partition.difference(&split);
        if !intersection.is_empty() {
            next_partitions.push(intersection);
        }
        if !difference.is_empty() {
            next_partitions.push(difference);
        }
    }
    next_partitions
}

pub fn build_regex_with_profile_label(exprs: &[Expr], _profile_label: &str) -> Regex {
    let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false);
    let total_started_at = std::time::Instant::now();
    let plan = build_exclusion_compile_plan(exprs);
    let group_sets: Vec<U8Set> = plan
        .compiled_exprs
        .iter()
        .map(|expr| expr_u8set(expr))
        .collect();
    let used_product_dfa = plan.compiled_exprs.len() > 1;
    let determinize_started_at = std::time::Instant::now();

    let (mut dfa, build_nfa_ms, nfa_states_after_build, condense_ms, nfa_states_after_condense) =
        if used_product_dfa {
            (build_product_dfa(&plan.compiled_exprs, _profile_label, debug_profile), 0.0, 0, 0.0, 0)
        } else {
            let build_nfa_started_at = std::time::Instant::now();
            let mut nfa = build_regex_nfa(&plan.compiled_exprs);
            let build_nfa_ms = build_nfa_started_at.elapsed().as_secs_f64() * 1000.0;
            let nfa_states_after_build = nfa.states.len();
            if debug_profile {
                eprintln!(
                    "[glrmask/debug][prepare_regex] label={} stage=build_nfa num_exprs={} ms={:.3} nfa_states={}",
                    _profile_label,
                    plan.compiled_exprs.len(),
                    build_nfa_ms,
                    nfa_states_after_build,
                );
            }

            let condense_started_at = std::time::Instant::now();
            nfa.condense_epsilon_sccs();
            let condense_ms = condense_started_at.elapsed().as_secs_f64() * 1000.0;
            let nfa_states_after_condense = nfa.states.len();
            if debug_profile {
                eprintln!(
                    "[glrmask/debug][prepare_regex] label={} stage=condense num_exprs={} ms={:.3} nfa_states={}",
                    _profile_label,
                    plan.compiled_exprs.len(),
                    condense_ms,
                    nfa_states_after_condense,
                );
            }

            (nfa.to_dfa(), build_nfa_ms, nfa_states_after_build, condense_ms, nfa_states_after_condense)
        };

    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let dfa_states_after_determinize = dfa.num_states();
    if debug_profile {
        eprintln!(
            "[glrmask/debug][prepare_regex] label={} stage=determinize num_exprs={} ms={:.3} dfa_states={}",
            _profile_label,
            plan.compiled_exprs.len(),
            determinize_ms,
            dfa_states_after_determinize,
        );
    }

    dfa.ensure_group_capacity(group_sets.len());
    for (group_id, set) in group_sets.into_iter().enumerate() {
        dfa.set_group_u8set(group_id as u32, set);
    }

    if !plan.exclusions.is_empty() {
        dfa.apply_group_exclusions(&plan.exclusions);
    }

    let dfa = if plan.visible_groups < plan.compiled_exprs.len() {
        dfa.project_groups(plan.visible_groups)
    } else {
        dfa
    };

    let minimize_started_at = std::time::Instant::now();
    let skip_minimize_for_product = used_product_dfa
        && plan.exclusions.is_empty()
        && plan.visible_groups == plan.compiled_exprs.len();
    let mut dfa = if skip_minimize_for_product {
        // A reachable product of independently minimized component DFAs is already
        // minimal: any difference in a component state yields a distinguishing
        // suffix in that component, which also distinguishes the full product tuple.
        dfa
    } else {
        dfa.minimize()
    };
    if skip_minimize_for_product {
        dfa.recompute_possible_futures();
    }
    let minimize_ms = if skip_minimize_for_product {
        0.0
    } else {
        minimize_started_at.elapsed().as_secs_f64() * 1000.0
    };
    if debug_profile {
        eprintln!(
            "[glrmask/debug][prepare_regex] label={} stage=minimize num_exprs={} ms={:.3} dfa_states={}",
            _profile_label,
            plan.compiled_exprs.len(),
            minimize_ms,
            dfa.num_states(),
        );
    }

    if debug_profile {
        eprintln!(
            "[glrmask/debug][prepare_regex] label={} build_nfa_ms={:.3} nfa_states_build={} condense_ms={:.3} nfa_states_condensed={} determinize_ms={:.3} dfa_states_determinized={} minimize_ms={:.3} dfa_states_minimized={} total_ms={:.3}",
            _profile_label,
            build_nfa_ms,
            nfa_states_after_build,
            condense_ms,
            nfa_states_after_condense,
            determinize_ms,
            dfa_states_after_determinize,
            minimize_ms,
            dfa.num_states(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Regex { dfa }
}

fn debug_profile_enabled() -> bool {
    std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

/// Compile multiple expressions into a single NFA (without determinization).
///
/// Each expression's index becomes its group ID.
pub fn build_regex_nfa(exprs: &[Expr]) -> NFA {
    build_regex_nfa_impl(exprs, true)
}

fn build_regex_nfa_impl(exprs: &[Expr], probe_single_exprs: bool) -> NFA {
    let optimized_exprs: Vec<Expr> = exprs.iter().cloned().map(Expr::optimize).collect();

    if probe_single_exprs && debug_profile_enabled() {
        for (index, expr) in optimized_exprs.iter().enumerate() {
            let single_nfa = build_regex_nfa_impl(std::slice::from_ref(expr), false);
            eprintln!(
                "[glrmask/debug][regex_nfa] expr={} nfa_states={}",
                index,
                single_nfa.states.len(),
            );
        }
    }

    let mut nfa = NFA::new(1);

    if let Some((prefix, remainders)) = common_prefix_factor(&optimized_exprs) {
        let split = nfa.add_state();
        append_compiled_expr(&prefix, &mut nfa, 0, split);

        for (group_id, remainder) in remainders.iter().enumerate() {
            match remainder {
                Expr::Dfa(dfa) if dfa_start_is_entry_only(dfa) => {
                    append_group_dfa_expr(dfa, &mut nfa, split, group_id as u32)
                }
                Expr::Dfa(dfa) => {
                    let accept = nfa.add_state();
                    append_dfa_expr(dfa, &mut nfa, split, accept);
                    nfa.add_finalizer(accept, group_id as u32);
                }
                _ => {
                    let accept = nfa.add_state();
                    append_compiled_expr(remainder, &mut nfa, split, accept);
                    nfa.add_finalizer(accept, group_id as u32);
                }
            }
        }
        return nfa;
    }

    for (group_id, expr) in optimized_exprs.iter().enumerate() {
        match expr {
            Expr::Dfa(dfa) if dfa_start_is_entry_only(dfa) => {
                append_group_dfa_expr(dfa, &mut nfa, 0, group_id as u32)
            }
            Expr::Dfa(dfa) => {
                let accept = nfa.add_state();
                append_dfa_expr(dfa, &mut nfa, 0, accept);
                nfa.add_finalizer(accept, group_id as u32);
            }
            _ => {
                let accept = nfa.add_state();
                append_compiled_expr(expr, &mut nfa, 0, accept);
                nfa.add_finalizer(accept, group_id as u32);
            }
        }
    }
    nfa
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::regex::{byte, bytes, choice, class, exclude, repeat};

    fn accepts(regex: &Regex, input: &[u8]) -> bool {
        let mut state = 0;
        for &byte in input {
            let Some(next) = regex.step(state, byte) else {
                return false;
            };
            state = next;
        }
        regex.dfa.finalizers(state).contains(0)
    }

    #[test]
    fn test_bounded_repeat_accepts_only_exact_count() {
        let regex = repeat(bytes(b"ab"), 4, Some(4)).build();

        assert!(!accepts(&regex, b""));
        assert!(!accepts(&regex, b"ababab"));
        assert!(accepts(&regex, b"abababab"));
        assert!(!accepts(&regex, b"ababababab"));
    }

    #[test]
    fn test_bounded_repeat_accepts_required_and_optional_range() {
        let regex = repeat(choice(vec![bytes(b"ab"), bytes(b"cd")]), 2, Some(5)).build();

        assert!(!accepts(&regex, b""));
        assert!(!accepts(&regex, b"ab"));
        assert!(accepts(&regex, b"abcd"));
        assert!(accepts(&regex, b"ababcd"));
        assert!(accepts(&regex, b"abcdabcd"));
        assert!(accepts(&regex, b"abcdababcd"));
        assert!(!accepts(&regex, b"abcdababcdab"));
    }

    #[test]
    fn test_bounded_repeat_zero_to_range_accepts_expected_lengths() {
        let regex = repeat(byte(b'a'), 0, Some(7)).build();

        for len in 0..=7 {
            assert!(accepts(&regex, &vec![b'a'; len]), "expected len={} to match", len);
        }
        assert!(!accepts(&regex, b"aaaaaaaa"));
        assert!(!accepts(&regex, b"aaaab"));
    }

    #[test]
    fn test_large_bounded_repeat_accepts_expected_lengths() {
        let regex = repeat(byte(b'a'), 0, Some(130)).build();

        assert!(accepts(&regex, b""));
        assert!(accepts(&regex, &vec![b'a'; 130]));
        assert!(!accepts(&regex, &vec![b'a'; 131]));
    }

    #[test]
    fn test_top_level_dfa_expr_avoids_boundary_epsilons() {
        let base = byte(b'a').build();
        let nfa = build_regex_nfa(&[Expr::Dfa(base.dfa.clone())]);

        assert!(nfa
            .states
            .iter()
            .all(|state| state.epsilon_transitions.is_empty()));
        assert!(nfa.states.iter().any(|state| !state.finalizers.is_empty()));
    }

    #[test]
    fn test_possible_final_matches_ab() {
        let regex = build_regex(&[bytes(b"a"), bytes(b"b")]);
        // Only initial state should have possible final matches
        assert_eq!(regex.dfa.possible_future_group_ids(0).iter().collect::<Vec<_>>(), [0, 1]);
        assert!(regex.dfa.possible_future_group_ids(1).is_empty());
        assert!(regex.dfa.possible_future_group_ids(2).is_empty());
    }

    #[test]
    fn test_top_level_exclude_blocks_same_length_match() {
        let regex = build_regex(&[exclude(class(U8Set::from_range(0, 255)), byte(b'a'))]);

        assert!(!accepts(&regex, b"a"));
        assert!(accepts(&regex, b"b"));
    }

    #[test]
    fn test_top_level_exclude_chain_blocks_multiple_literals() {
        let regex = build_regex(&[exclude(
            exclude(class(U8Set::from_range(0, 255)), byte(b'a')),
            byte(b'b'),
        )]);

        assert!(!accepts(&regex, b"a"));
        assert!(!accepts(&regex, b"b"));
        assert!(accepts(&regex, b"c"));
    }
}
