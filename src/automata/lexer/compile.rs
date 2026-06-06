use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::env;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ds::{bitset::BitSet, u8set::U8Set};

use super::ast::Expr;
use super::dfa::DFA;
use super::nfa::NFA;

type ProductStateTuple = SmallVec<[(u32, u32); 12]>;

fn glrmask_debug_regex_suffix() -> bool {
    env::var_os("GLRMASK_DEBUG_REGEX_SUFFIX").is_some()
}

fn debug_byte_repr(x: u8) -> String {
    match x {
        b'\n' => "\\n".to_string(),
        b'\r' => "\\r".to_string(),
        b'\t' => "\\t".to_string(),
        b'\\' => "\\\\".to_string(),
        b'"' => "\\\"".to_string(),
        0x20..=0x7e => (x as char).to_string(),
        _ => format!("\\x{x:02X}"),
    }
}

fn debug_bytes_repr(xs: &[u8]) -> String {
    let mut out = String::new();
    for &x in xs {
        out.push_str(&debug_byte_repr(x));
    }
    out
}

fn debug_env_token_bytes(name: &str) -> Option<Vec<u8>> {
    let raw = env::var(name).ok()?;
    let mut out = Vec::new();
    let mut it = raw.as_bytes().iter().copied().peekable();
    while let Some(b) = it.next() {
        if b == b'\\' {
            match it.peek().copied() {
                Some(b'n') => {
                    it.next();
                    out.push(b'\n');
                }
                Some(b't') => {
                    it.next();
                    out.push(b'\t');
                }
                Some(b'r') => {
                    it.next();
                    out.push(b'\r');
                }
                Some(b'\\') => {
                    it.next();
                    out.push(b'\\');
                }
                Some(b'"') => {
                    it.next();
                    out.push(b'"');
                }
                _ => out.push(b),
            }
        } else {
            out.push(b);
        }
    }
    Some(out)
}

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

fn expr_contains_group_op(expr: &Expr) -> bool {
    match expr {
        Expr::Exclude { .. } | Expr::Intersect { .. } => true,
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_contains_group_op),
        Expr::Repeat { expr, .. } => expr_contains_group_op(expr),
        Expr::Shared(inner) => expr_contains_group_op(inner),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
    }
}

fn split_top_level_group_ops(expr: &Expr) -> (Expr, Vec<Expr>, Vec<Expr>) {
    match expr {
        Expr::Exclude { expr, exclude } => {
            let (base, mut excluded, intersections) = split_top_level_group_ops(expr);
            excluded.push((**exclude).clone());
            (base, excluded, intersections)
        }
        Expr::Intersect { expr, intersect } => {
            let (base, excluded, mut intersections) = split_top_level_group_ops(expr);
            intersections.push((**intersect).clone());
            (base, excluded, intersections)
        }
        Expr::Shared(inner)
            if matches!(inner.as_ref(), Expr::Exclude { .. } | Expr::Intersect { .. }) => {
            split_top_level_group_ops(inner.as_ref())
        }
        _ => (expr.clone(), Vec::new(), Vec::new()),
    }
}

fn materialize_nested_group_ops(expr: Expr) -> Expr {
    match expr {
        Expr::Exclude { .. } | Expr::Intersect { .. } => Expr::Dfa(Arc::new(compile_with_plan(
            build_exclusion_compile_plan(std::slice::from_ref(&expr)),
        ))),
        Expr::Seq(parts) => Expr::Seq(parts.into_iter().map(materialize_nested_group_ops).collect()),
        Expr::Choice(options) => Expr::Choice(
            options
                .into_iter()
                .map(materialize_nested_group_ops)
                .collect(),
        ),
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(materialize_nested_group_ops(*expr)),
            min,
            max,
        },
        Expr::Shared(inner) => {
            let rewritten = materialize_nested_group_ops((*inner).clone());
            if rewritten == *inner {
                Expr::Shared(inner)
            } else {
                rewritten
            }
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => expr,
    }
}

struct ExclusionCompilePlan {
    compiled_exprs: Vec<Expr>,
    exclusions: BTreeMap<u32, BTreeSet<u32>>,
    intersections: BTreeMap<u32, BTreeSet<u32>>,
    visible_groups: usize,
    profile_labels: Option<Vec<ProductComponentProfileLabel>>,
}

struct ProductComponentProfileLabel {
    name: String,
    origin: &'static str,
    shared: bool,
}

struct ProductGrowthTrieNode {
    children: HashMap<u32, usize>,
}

impl ProductGrowthTrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
        }
    }
}

struct ProductGrowthRecorder {
    nodes: Vec<ProductGrowthTrieNode>,
    prefix_counts: Vec<usize>,
    dense_states: Vec<u32>,
}

impl ProductGrowthRecorder {
    fn new(num_groups: usize) -> Self {
        Self {
            nodes: vec![ProductGrowthTrieNode::new()],
            prefix_counts: vec![0; num_groups],
            dense_states: vec![0; num_groups],
        }
    }

    fn record(&mut self, num_groups: usize, state_tuple: &ProductStateTuple) {
        self.dense_states.fill(0);
        for &(group_id, state) in state_tuple {
            let group_index = group_id as usize;
            if group_index < num_groups {
                self.dense_states[group_index] = state.saturating_add(1);
            }
        }

        let mut node_index = 0usize;
        for (depth, &state) in self.dense_states.iter().enumerate() {
            let next_index = if let Some(&existing) = self.nodes[node_index].children.get(&state) {
                existing
            } else {
                let new_index = self.nodes.len();
                self.nodes.push(ProductGrowthTrieNode::new());
                self.nodes[node_index].children.insert(state, new_index);
                self.prefix_counts[depth] += 1;
                new_index
            };
            node_index = next_index;
        }
    }

    fn prefix_counts(&self) -> &[usize] {
        &self.prefix_counts
    }
}

fn expr_is_shared(expr: &Expr) -> bool {
    match expr {
        Expr::Shared(_) => true,
        Expr::Exclude { expr, exclude } => expr_is_shared(expr) || expr_is_shared(exclude),
        Expr::Intersect { expr, intersect } => expr_is_shared(expr) || expr_is_shared(intersect),
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_is_shared),
        Expr::Repeat { expr, .. } => expr_is_shared(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
    }
}

fn expr_profile_summary(expr: &Expr) -> String {
    const MAX_LEN: usize = 80;
    let mut summary = format!("{:?}", expr);
    if summary.len() > MAX_LEN {
        summary.truncate(MAX_LEN - 3);
        summary.push_str("...");
    }
    summary
}

fn build_exclusion_compile_plan_with_labels(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
) -> ExclusionCompilePlan {
    let visible_groups = exprs.len();
    let mut compiled_exprs = Vec::with_capacity(visible_groups);
    let mut deferred_exclusions = Vec::<Vec<Expr>>::with_capacity(visible_groups);
    let mut deferred_intersections = Vec::<Vec<Expr>>::with_capacity(visible_groups);
    let mut profile_labels = visible_labels.map(|_| Vec::with_capacity(visible_groups));

    if let Some(labels) = visible_labels {
        assert_eq!(
            labels.len(),
            visible_groups,
            "visible profile labels must match expression count"
        );
    }

    for (index, expr) in exprs.iter().enumerate() {
        let (base, excluded, intersections) = split_top_level_group_ops(expr);
        let base = materialize_nested_group_ops(base);
        let excluded = excluded
            .into_iter()
            .map(materialize_nested_group_ops)
            .collect::<Vec<_>>();
        let intersections = intersections
            .into_iter()
            .map(materialize_nested_group_ops)
            .collect::<Vec<_>>();
        assert!(
            !expr_contains_group_op(&base),
            "Expr::Exclude and Expr::Intersect are currently only supported at the top level of a terminal expression"
        );
        for excluded_expr in &excluded {
            assert!(
                !expr_contains_group_op(excluded_expr),
                "nested Expr::Exclude/Expr::Intersect inside an exclusion branch is not supported"
            );
        }
        for intersection_expr in &intersections {
            assert!(
                !expr_contains_group_op(intersection_expr),
                "nested Expr::Exclude/Expr::Intersect inside an intersection branch is not supported"
            );
        }
        compiled_exprs.push(base);
        if let (Some(labels), Some(profile_labels)) = (visible_labels, profile_labels.as_mut()) {
            profile_labels.push(ProductComponentProfileLabel {
                name: labels[index].clone(),
                origin: "visible",
                shared: expr_is_shared(expr),
            });
        }
        deferred_exclusions.push(excluded);
        deferred_intersections.push(intersections);
    }

    let mut exclusions = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut intersections = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut next_group = visible_groups as u32;
    for (group_id, (excluded_exprs, intersection_exprs)) in deferred_exclusions
        .into_iter()
        .zip(deferred_intersections.into_iter())
        .enumerate()
    {
        let exclusion_entry = exclusions.entry(group_id as u32).or_default();
        for (excluded_index, excluded_expr) in excluded_exprs.into_iter().enumerate() {
            let is_shared = expr_is_shared(&excluded_expr);
            compiled_exprs.push(excluded_expr);
            exclusion_entry.insert(next_group);
            if let Some(profile_labels) = profile_labels.as_mut() {
                let base_name = profile_labels[group_id].name.clone();
                profile_labels.push(ProductComponentProfileLabel {
                    name: format!("{}::exclude#{}", base_name, excluded_index),
                    origin: "internal_exclusion",
                    shared: is_shared,
                });
            }
            next_group += 1;
        }

        let intersection_entry = intersections.entry(group_id as u32).or_default();
        for (intersection_index, intersection_expr) in intersection_exprs.into_iter().enumerate() {
            let is_shared = expr_is_shared(&intersection_expr);
            compiled_exprs.push(intersection_expr);
            intersection_entry.insert(next_group);
            if let Some(profile_labels) = profile_labels.as_mut() {
                let base_name = profile_labels[group_id].name.clone();
                profile_labels.push(ProductComponentProfileLabel {
                    name: format!("{}::intersect#{}", base_name, intersection_index),
                    origin: "internal_intersection",
                    shared: is_shared,
                });
            }
            next_group += 1;
        }
    }

    exclusions.retain(|_, v| !v.is_empty());
    intersections.retain(|_, v| !v.is_empty());

    ExclusionCompilePlan {
        compiled_exprs,
        exclusions,
        intersections,
        visible_groups,
        profile_labels,
    }
}

fn build_exclusion_compile_plan(exprs: &[Expr]) -> ExclusionCompilePlan {
    build_exclusion_compile_plan_with_labels(exprs, None)
}

fn expr_accepts_empty(expr: &Expr) -> bool {
    match expr {
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::U8Class(_) => false,
        Expr::Dfa(dfa) => !dfa.finalizers(0).is_empty(),
        Expr::Intersect { expr, intersect } => {
            expr_accepts_empty(expr) && expr_accepts_empty(intersect)
        }
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
        Expr::Dfa(dfa) => {
            let mut set = U8Set::empty();
            for state in dfa.states() {
                for (byte, _) in state.transitions.iter() {
                    set.insert(byte);
                }
            }
            set
        }
        Expr::Seq(parts) | Expr::Choice(parts) => parts
            .iter()
            .fold(U8Set::empty(), |acc, part| acc | expr_u8set(part)),
        Expr::Intersect { expr, intersect } => expr_u8set(expr).intersection(&expr_u8set(intersect)),
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
    let mut nfa = build_regex_nfa_impl(std::slice::from_ref(expr));
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

fn compile_direct_bounded_repeat_base_dfa_unconditionally(expr: &Expr) -> Option<DFA> {
    let base_dfa = compile_expr_to_dfa(expr);
    if base_dfa.num_states() == 0 || !dfa_is_nonnullable_and_prefix_free(&base_dfa) {
        return None;
    }

    Some(base_dfa)
}

fn compile_direct_bounded_repeat_base_dfa(expr: &Expr, max: usize) -> Option<DFA> {
    if max < DIRECT_BOUNDED_REPEAT_THRESHOLD {
        return None;
    }
    compile_direct_bounded_repeat_base_dfa_unconditionally(expr)
}

fn build_bounded_repeat_dfa(expr: &Expr, min: usize, max: usize) -> Option<DFA> {
    let base_dfa = compile_direct_bounded_repeat_base_dfa(expr, max)?;

    let base_states = base_dfa.states();
    let base_state_count = base_states.len();
    let total_states = (max + 1).checked_mul(base_state_count)?;
    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(1);

    for copies_done in 0..=max {
        for (state_id, state) in base_states.iter().enumerate() {
            let mapped_state = (copies_done * base_state_count + state_id) as u32;
            let mut finalizers = crate::ds::bitset::BitSet::new(1);
            let mut future = crate::ds::bitset::BitSet::new(1);
            if state_id == 0 && copies_done >= min {
                finalizers.set(0);
            }
            if copies_done < max {
                future.set(0);
            }
            dfa.overwrite_state_metadata(mapped_state, finalizers, future);

            if copies_done == max || !base_dfa.finalizers(state_id as u32).is_empty() {
                continue;
            }

            let mut transitions = Vec::with_capacity(state.transitions.len());
            for (byte, &target) in state.transitions.iter() {
                let mapped_target = if !base_dfa.finalizers(target).is_empty() {
                    ((copies_done + 1) * base_state_count) as u32
                } else {
                    (copies_done * base_state_count + target as usize) as u32
                };
                transitions.push((byte, mapped_target));
            }
            dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
        }
    }

    Some(dfa)
}

/// Collects all bytes from a slice of suffix expressions that are all U8Seq.
/// Returns None if any expression is not a simple byte sequence.
fn collect_suffix_bytes(exprs: &[Expr]) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    for expr in exprs {
        match expr {
            Expr::U8Seq(b) => bytes.extend_from_slice(b),
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::U8Seq(b) => bytes.extend_from_slice(b),
                _ => return None,
            },
            _ => return None,
        }
    }
    if bytes.is_empty() { None } else { Some(bytes) }
}

/// Builds a DFA for `Seq([Repeat{expr, min, max}, suffix_bytes...])` directly,
/// avoiding NFA→DFA determinization. Works when the first suffix byte does not
/// overlap with the repeat expression's start-state transitions (e.g., closing
/// quote `"` after JSON string chars that exclude `"`).
fn build_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {
    if parts.len() < 2 {
        return None;
    }

    // Extract repeat parameters, unwrapping Shared if needed.
    let first = match &parts[0] {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    let (repeat_expr, min, max) = match first {
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => (expr.as_ref(), *min, *max),
        _ => return None,
    };

    let suffix_bytes = collect_suffix_bytes(&parts[1..])?;
    let base_dfa = compile_direct_bounded_repeat_base_dfa_unconditionally(repeat_expr)?;

    let base_states = base_dfa.states();
    let base_state_count = base_states.len();
    let repeat_state_count = (max + 1).checked_mul(base_state_count)?;
    let suffix_len = suffix_bytes.len();
    let total_states = repeat_state_count + suffix_len;

    // Safety check: first suffix byte must NOT appear in start-state transitions
    // of the base DFA, otherwise the DFA would be nondeterministic at accepting
    // positions (ambiguity between continuing the repeat and starting the suffix).
    if base_states[0].transitions.get(suffix_bytes[0]).is_some() {
        return None;
    }

    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(1);
    let first_suffix_state = repeat_state_count as u32;

    for copies_done in 0..=max {
        for (state_id, state) in base_states.iter().enumerate() {
            let mapped_state = (copies_done * base_state_count + state_id) as u32;
            // No finalizers on repeat states — only the suffix chain end finalizes.
            let finalizers = crate::ds::bitset::BitSet::new(1);
            let mut future = crate::ds::bitset::BitSet::new(1);

            let is_accepting_pos = state_id == 0 && copies_done >= min;
            if copies_done < max || is_accepting_pos {
                future.set(0);
            }
            dfa.overwrite_state_metadata(mapped_state, finalizers, future);

            // At max copies or at a base-DFA finalizer state: no repeat transitions,
            // but accepting positions still get the suffix entry transition.
            if copies_done == max || !base_dfa.finalizers(state_id as u32).is_empty() {
                if is_accepting_pos {
                    dfa.set_transitions_from_sorted_entries(
                        mapped_state,
                        vec![(suffix_bytes[0], first_suffix_state)],
                    );
                }
                continue;
            }

            // Build transitions: repeat transitions + optional suffix entry.
            let extra = if is_accepting_pos { 1 } else { 0 };
            let mut transitions = Vec::with_capacity(state.transitions.len() + extra);
            for (byte, &target) in state.transitions.iter() {
                let mapped_target = if !base_dfa.finalizers(target).is_empty() {
                    ((copies_done + 1) * base_state_count) as u32
                } else {
                    (copies_done * base_state_count + target as usize) as u32
                };
                transitions.push((byte, mapped_target));
            }
            if is_accepting_pos {
                let pos = transitions.partition_point(|&(b, _)| b < suffix_bytes[0]);
                transitions.insert(pos, (suffix_bytes[0], first_suffix_state));
            }
            dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
        }
    }

    // Build suffix chain: each state transitions on the NEXT suffix byte.
    for i in 0..suffix_len {
        let suffix_state = (repeat_state_count + i) as u32;
        if i + 1 < suffix_len {
            let next_suffix = (repeat_state_count + i + 1) as u32;
            let mut future = crate::ds::bitset::BitSet::new(1);
            future.set(0);
            dfa.overwrite_state_metadata(
                suffix_state,
                crate::ds::bitset::BitSet::new(1),
                future,
            );
            dfa.set_transitions_from_sorted_entries(
                suffix_state,
                vec![(suffix_bytes[i + 1], next_suffix)],
            );
        } else {
            // Last suffix state: finalizer, no transitions, no future.
            let mut finalizers = crate::ds::bitset::BitSet::new(1);
            finalizers.set(0);
            dfa.overwrite_state_metadata(
                suffix_state,
                finalizers,
                crate::ds::bitset::BitSet::new(1),
            );
        }
    }

    Some((dfa, false))
}

/// Builds a DFA for `Seq([Repeat{body, min, max}, suffix_exprs...])` using a
/// product construction of body_DFA × suffix_DFA × completion_counter.
///
/// Handles cases where the suffix is a regex (not just bytes) and/or the body
/// is not prefix-free, which `build_bounded_repeat_with_suffix_dfa` cannot handle.
/// Avoids the exponential NFA→DFA blowup that occurs with unrolled bounded repeats.
///
/// The product state is `(body_state, suffix_state, counter)`:
///   - body tracks progress through the repeat body expression
///   - suffix tracks the suffix match (started at body boundaries when counter >= min)
///   - counter tracks completed body repetitions (0..max)
///
/// At body completion (body transitions from accept to dead), counter increments
/// and both body and suffix restart. If both old and fresh suffix are alive but
/// diverge, the function falls back to None (would need NFA-like tracking).
fn build_bounded_repeat_with_regex_suffix(parts: &[Expr]) -> Option<(DFA, bool)> {
    if parts.len() < 2 {
        return None;
    }

    let debug = glrmask_debug_regex_suffix();

    // Flatten one level of nested Seq: Seq([Seq([a, b]), c]) → [a, b, c]
    let flat_parts: Vec<&Expr>;
    let parts_ref: &[&Expr] = {
        let first_unwrapped = match &parts[0] {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        if let Expr::Seq(inner_parts) = first_unwrapped {
            flat_parts = inner_parts.iter().chain(parts[1..].iter()).collect();
            &flat_parts
        } else {
            flat_parts = parts.iter().collect();
            &flat_parts
        }
    };

    if parts_ref.len() < 2 {
        return None;
    }

    let first = match parts_ref[0] {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    let (repeat_expr, min, max) = match first {
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => (expr.as_ref(), *min, *max),
        _ => return None,
    };
    let body_dfa = compile_expr_to_dfa(repeat_expr);
    if body_dfa.num_states() == 0 || !body_dfa.finalizers(0).is_empty() {
        if debug {
            eprintln!(
                "[regex-suffix] reject candidate: body_dfa states={}, start_finalizers={:?}",
                body_dfa.num_states(),
                body_dfa.finalizers(0),
            );
        }
        return None;
    }

    let suffix_expr = if parts_ref.len() == 2 {
        parts_ref[1].clone()
    } else {
        Expr::Seq(parts_ref[1..].iter().map(|e| (*e).clone()).collect())
    };
    let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
    if suffix_dfa.num_states() == 0 {
        if debug {
            eprintln!("[regex-suffix] reject candidate: suffix_dfa has zero states");
        }
        return None;
    }

    let max_product =
        (body_dfa.num_states() + 1) * (suffix_dfa.num_states() + 1) * (max + 1);
    if max_product > 500_000 {
        if debug {
            eprintln!(
                "[regex-suffix] reject candidate: max_product={} body_states={} suffix_states={} max={}",
                max_product,
                body_dfa.num_states(),
                suffix_dfa.num_states(),
                max,
            );
        }
        return None;
    }

    let body_dead = body_dfa.num_states() as u32;
    let suffix_dead = suffix_dfa.num_states() as u32;

    let mut state_map: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    let mut state_keys: Vec<(u32, u32, u32)> = Vec::new();
    let mut worklist: VecDeque<(u32, (u32, u32, u32))> = VecDeque::new();
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);

    let start_suffix = if min == 0 { 0u32 } else { suffix_dead };
    let start_key = (0u32, start_suffix, 0u32);
    state_map.insert(start_key, 0);
    state_keys.push(start_key);
    worklist.push_back((0, start_key));

    if debug {
        eprintln!();
        eprintln!("[regex-suffix] ENTER build_bounded_repeat_with_regex_suffix");
        eprintln!("[regex-suffix] full parts = {parts_ref:#?}");
        eprintln!("[regex-suffix] repeat_expr = {repeat_expr:#?}");
        eprintln!("[regex-suffix] suffix_expr = {suffix_expr:#?}");
        eprintln!(
            "[regex-suffix] min={min} max={max} body_states={} suffix_states={} body_dead={} suffix_dead={} start_key={:?}",
            body_dfa.num_states(),
            suffix_dfa.num_states(),
            body_dead,
            suffix_dead,
            start_key,
        );
    }

    {
        let is_accept = start_suffix < suffix_dead
            && !suffix_dfa.finalizers(start_suffix).is_empty();
        let mut finalizers = BitSet::new(1);
        let mut future = BitSet::new(1);
        if is_accept {
            finalizers.set(0);
        }
        future.set(0);
        dfa.overwrite_state_metadata(0, finalizers, future);
    }

    while let Some((dfa_state, (b, s, c))) = worklist.pop_front() {
        let body_is_accept = b < body_dead && !body_dfa.finalizers(b).is_empty();

        if debug {
            eprintln!(
                "[regex-suffix] POP dfa_state={dfa_state} key=(body={b}, suffix={s}, count={c}) body_is_accept={body_is_accept} body_finalizers={:?} suffix_finalizers={:?}",
                if b < body_dead { Some(body_dfa.finalizers(b)) } else { None },
                if s < suffix_dead { Some(suffix_dfa.finalizers(s)) } else { None },
            );
        }

        let mut transitions = Vec::new();
        for byte_val in 0u16..=255 {
            let x = byte_val as u8;

            let b_next = if b < body_dead {
                body_dfa.step(b, x).map_or(body_dead, |t| t)
            } else {
                body_dead
            };

            if debug && body_is_accept {
                let boundary_c = c + 1;
                let boundary_new_b = if boundary_c < max as u32 {
                    body_dfa.step(0, x).map_or(body_dead, |t| t)
                } else {
                    body_dead
                };
                let old_s_next = if s < suffix_dead {
                    suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                } else {
                    suffix_dead
                };
                let boundary_fresh_s = if boundary_c >= min as u32 {
                    suffix_dfa.step(0, x).map_or(suffix_dead, |t| t)
                } else {
                    suffix_dead
                };

                if b_next != body_dead && boundary_fresh_s != suffix_dead {
                    eprintln!(
                        "[regex-suffix][AMBIG_BOUNDARY] from dfa_state={dfa_state} key=(body={b}, suffix={s}, count={c}) byte=0x{x:02X} '{}' \
                         body_can_continue: {b}->{b_next}; finish_body_first: count {c}->{boundary_c}, restart_body_on_byte={boundary_new_b}, fresh_suffix_on_byte={boundary_fresh_s}; old_suffix_on_byte={old_s_next}",
                        debug_byte_repr(x),
                    );
                }

                if b_next != body_dead
                    && old_s_next != suffix_dead
                    && boundary_fresh_s != suffix_dead
                    && old_s_next != boundary_fresh_s
                {
                    eprintln!(
                        "[regex-suffix][THREE_WAY_DIVERGE_POSSIBLE] from dfa_state={dfa_state} byte=0x{x:02X} '{}': body_next={b_next}, old_s_next={old_s_next}, boundary_fresh_s={boundary_fresh_s}",
                        debug_byte_repr(x),
                    );
                }
            }

            let (final_b, final_s, final_c) =
                if body_is_accept && b_next == body_dead {
                    let new_c = c + 1;
                    let new_b = if new_c < max as u32 {
                        body_dfa.step(0, x).map_or(body_dead, |t| t)
                    } else {
                        body_dead
                    };
                    let old_s_next = if s < suffix_dead {
                        suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    let fresh_s = if new_c >= min as u32 {
                        suffix_dfa.step(0, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    let new_s = match (old_s_next < suffix_dead, fresh_s < suffix_dead) {
                        (true, true) if old_s_next != fresh_s => {
                            if debug {
                                eprintln!(
                                    "[regex-suffix][FALLBACK_DIVERGE] from dfa_state={dfa_state} key=(body={b}, suffix={s}, count={c}) byte=0x{x:02X} '{}' old_s_next={old_s_next} fresh_s={fresh_s}",
                                    debug_byte_repr(x),
                                );
                            }
                            return None;
                        }
                        (true, _) => old_s_next,
                        (_, true) => fresh_s,
                        _ => suffix_dead,
                    };
                    if debug && (old_s_next < suffix_dead || fresh_s < suffix_dead) {
                        eprintln!(
                            "[regex-suffix][FINISH_BODY] from dfa_state={dfa_state} key=(body={b}, suffix={s}, count={c}) byte=0x{x:02X} '{}' \
                             because b_next=DEAD; new_key=(body={new_b}, suffix={new_s}, count={new_c}); old_s_next={old_s_next}, fresh_s={fresh_s}",
                            debug_byte_repr(x),
                        );
                    }
                    (new_b, new_s, new_c)
                } else {
                    let s_next = if s < suffix_dead {
                        suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    if debug && body_is_accept && b_next != body_dead {
                        let fresh_s_if_finished = if c + 1 >= min as u32 {
                            suffix_dfa.step(0, x).map_or(suffix_dead, |t| t)
                        } else {
                            suffix_dead
                        };
                        if fresh_s_if_finished != suffix_dead {
                            eprintln!("[regex-suffix][GREEDY_KEEP_BODY] from dfa_state={dfa_state} byte=0x{x:02X} '{}' kept key=(body={b_next}, suffix={s_next}, count={c}); DROPPED finish-before-byte suffix_state={fresh_s_if_finished}", debug_byte_repr(x));
                        }
                    }
                    (b_next, s_next, c)
                };

            if final_b == body_dead && final_s == suffix_dead {
                continue;
            }

            let target_key = (final_b, final_s, final_c);
            let target_dfa_state =
                if let Some(&existing) = state_map.get(&target_key) {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    let accept = final_s < suffix_dead
                        && !suffix_dfa.finalizers(final_s).is_empty()
                        && final_c >= min as u32;
                    let has_future = final_b < body_dead || final_s < suffix_dead;
                    let mut finalizers = BitSet::new(1);
                    let mut future = BitSet::new(1);
                    if accept {
                        finalizers.set(0);
                    }
                    if has_future {
                        future.set(0);
                    }
                    dfa.overwrite_state_metadata(new_state, finalizers, future);
                    state_map.insert(target_key, new_state);
                    if new_state as usize >= state_keys.len() {
                        state_keys.resize(new_state as usize + 1, (u32::MAX, u32::MAX, u32::MAX));
                    }
                    state_keys[new_state as usize] = target_key;
                    worklist.push_back((new_state, target_key));
                    new_state
                };

            transitions.push((x, target_dfa_state));
        }

        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|e| e.0);
        }
        dfa.set_transitions_from_sorted_entries(dfa_state, transitions);
    }

    if debug {
        eprintln!(
            "[regex-suffix] BUILT raw product DFA states={} transitions_done=true",
            dfa.num_states(),
        );
        for sid in 0..dfa.num_states() {
            let key = state_keys
                .get(sid)
                .copied()
                .unwrap_or((u32::MAX, u32::MAX, u32::MAX));
            eprintln!(
                "[regex-suffix][STATE] dfa_state={sid} key={key:?} finalizers={:?} future={:?}",
                dfa.finalizers(sid as u32),
                dfa.possible_future_group_ids(sid as u32),
            );
        }
    }

    if debug {
        if let Some(token) = debug_env_token_bytes("GLRMASK_DEBUG_REGEX_SUFFIX_TOKEN") {
            eprintln!(
                "[regex-suffix][TRACE_TOKEN] token bytes len={} repr='{}' raw={:?}",
                token.len(),
                debug_bytes_repr(&token),
                token,
            );
            let mut st = 0u32;
            eprintln!(
                "[regex-suffix][TRACE_TOKEN] start state={st} key={:?} finalizers={:?} future={:?}",
                state_keys.get(st as usize),
                dfa.finalizers(st),
                dfa.possible_future_group_ids(st),
            );
            for (i, &x) in token.iter().enumerate() {
                let next = dfa.step(st, x);
                eprintln!(
                    "[regex-suffix][TRACE_TOKEN] byte[{i}]=0x{x:02X} '{}' state={st} key={:?} -> {:?}",
                    debug_byte_repr(x),
                    state_keys.get(st as usize),
                    next,
                );
                match next {
                    Some(n) => st = n,
                    None => {
                        eprintln!("[regex-suffix][TRACE_TOKEN] DEAD after byte[{i}]");
                        break;
                    }
                }
            }
            if (st as usize) < dfa.num_states() {
                eprintln!(
                    "[regex-suffix][TRACE_TOKEN] end state={st} key={:?} finalizers={:?} future={:?}",
                    state_keys.get(st as usize),
                    dfa.finalizers(st),
                    dfa.possible_future_group_ids(st),
                );
            }
        }
    }

    let dfa = dfa.minimize();
    if debug {
        eprintln!("[regex-suffix] minimized states={}", dfa.num_states());
        eprintln!("[regex-suffix] EXIT build_bounded_repeat_with_regex_suffix");
        eprintln!();
    }
    Some((dfa, false))
}

fn prepend_literal_prefix_to_dfa(prefix_bytes: &[u8], tail_dfa: DFA) -> Option<DFA> {
    if prefix_bytes.is_empty() {
        return Some(tail_dfa);
    }

    let total_states = prefix_bytes.len().checked_add(tail_dfa.num_states())?;
    let tail_offset = prefix_bytes.len() as u32;
    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(tail_dfa.num_groups());

    for (i, &byte) in prefix_bytes.iter().enumerate() {
        let mut future = BitSet::new(tail_dfa.num_groups());
        if tail_dfa.num_groups() > 0 {
            future.set(0);
        }
        dfa.overwrite_state_metadata(i as u32, BitSet::new(tail_dfa.num_groups()), future);
        let target = if i + 1 == prefix_bytes.len() {
            tail_offset
        } else {
            (i + 1) as u32
        };
        dfa.set_transitions_from_sorted_entries(i as u32, vec![(byte, target)]);
    }

    for state_id in 0..tail_dfa.num_states() {
        let mapped_state = tail_offset + state_id as u32;
        dfa.overwrite_state_metadata(
            mapped_state,
            tail_dfa.finalizers(state_id as u32).clone(),
            tail_dfa.possible_future_group_ids(state_id as u32).clone(),
        );
        let transitions = tail_dfa.states()[state_id]
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, tail_offset + target))
            .collect();
        dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
    }

    Some(dfa)
}

fn build_prefixed_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {
    let mut flat_parts = Vec::new();
    for part in parts {
        match part {
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter().cloned()),
                _ => flat_parts.push(part.clone()),
            },
            Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter().cloned()),
            _ => flat_parts.push(part.clone()),
        }
    }

    let parts = flat_parts.as_slice();
    if parts.len() < 2 {
        return None;
    }

    for repeat_index in 1..parts.len() - 1 {
        let repeat_expr = match &parts[repeat_index] {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        let Expr::Repeat { .. } = repeat_expr else {
            continue;
        };

        let prefix_bytes = collect_suffix_bytes(&parts[..repeat_index])?;
        let tail_parts: Vec<Expr> = parts[repeat_index..].to_vec();
        let (tail_dfa, needs_future_recompute) =
            build_bounded_repeat_with_suffix_dfa(&tail_parts)
                .or_else(|| build_bounded_repeat_with_regex_suffix(&tail_parts))?;
        let dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
        return Some((dfa, needs_future_recompute));
    }

    if parts.len() == 2 {
        let prefix_bytes = collect_suffix_bytes(&parts[..1])?;
        let tail_parts = optional_tail_parts(&parts[1])?;
        if tail_parts.len() >= 2 {
            let (tail_dfa, needs_future_recompute) =
                build_bounded_repeat_with_suffix_dfa(&tail_parts)
                    .or_else(|| build_bounded_repeat_with_regex_suffix(&tail_parts))?;
            let mut dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
            mark_state_accepting(&mut dfa, prefix_bytes.len() as u32);
            return Some((dfa, needs_future_recompute));
        }
    }

    None
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
        Expr::Intersect { .. } => {
            unreachable!("nested Expr::Intersect must be lowered before NFA compilation")
        }
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

    pub fn num_transitions(&self) -> usize {
        dfa_transition_count(&self.dfa)
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    pub fn get_u8set(&self, state: u32) -> U8Set {
        self.dfa.get_u8set(state)
    }
}

fn dfa_transition_count(dfa: &DFA) -> usize {
    dfa.states()
        .iter()
        .map(|state| state.transitions.len())
        .sum()
}

impl Expr {
    pub fn build(self) -> Regex {
        build_regex(&[self])
    }
}

/// Compile multiple expressions into a single multi-group [`Regex`].
///
/// Each expression's index becomes its group ID in the resulting DFA.
fn compile_single_expr_dfa(expr: &Expr) -> DFA {
    if let Some((mut dfa, needs_future_recompute)) = compile_product_component_dfa_direct(expr) {
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        if needs_future_recompute {
            dfa.recompute_possible_futures();
        }
        return dfa;
    }

    let mut nfa = build_regex_nfa(std::slice::from_ref(expr));
    nfa.condense_epsilon_sccs();
    nfa.to_dfa()
}

fn compile_multi_expr_dfa_via_nfa(exprs: &[Expr]) -> DFA {
    let mut nfa = build_regex_nfa(exprs);
    nfa.condense_epsilon_sccs();
    nfa.to_dfa()
}

fn should_use_shared_multi_nfa(plan: &ExclusionCompilePlan) -> bool {
    if std::env::var_os("GLRMASK_DISABLE_SHARED_MULTI_NFA").is_some() {
        return false;
    }

    if plan.compiled_exprs.len() != plan.visible_groups {
        return false;
    }

    if plan.compiled_exprs.len() < 16 {
        return false;
    }

    let Some(labels) = plan.profile_labels.as_ref() else {
        return false;
    };

    labels
        .iter()
        .take(plan.visible_groups)
        .filter(|label| label.name.starts_with("json_string_constrained"))
        .count()
        >= 8
}

fn compile_with_plan(plan: ExclusionCompilePlan) -> DFA {
    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let group_sets: Vec<U8Set> = plan
        .compiled_exprs
        .iter()
        .map(|expr| expr_u8set(expr))
        .collect();
    let use_shared_multi_nfa = should_use_shared_multi_nfa(&plan);
    let used_product_dfa = plan.compiled_exprs.len() > 1 && !use_shared_multi_nfa;

    let mut dfa = if used_product_dfa {
        build_product_dfa(&plan.compiled_exprs, plan.profile_labels.as_deref())
    } else if plan.compiled_exprs.len() > 1 {
        compile_multi_expr_dfa_via_nfa(&plan.compiled_exprs)
    } else {
        compile_single_expr_dfa(&plan.compiled_exprs[0])
    };

    dfa.ensure_group_capacity(group_sets.len());
    for (group_id, set) in group_sets.into_iter().enumerate() {
        dfa.set_group_u8set(group_id as u32, set);
    }

    let mut group_ops_changed = false;
    if !plan.exclusions.is_empty() {
        group_ops_changed |= dfa.apply_group_exclusions(&plan.exclusions);
    }
    if !plan.intersections.is_empty() {
        group_ops_changed |= dfa.apply_group_intersections(&plan.intersections);
    }
    if group_ops_changed {
        dfa.recompute_possible_futures();
    }

    let dfa = if plan.visible_groups < plan.compiled_exprs.len() {
        dfa.project_groups(plan.visible_groups)
    } else {
        dfa
    };

    let pre_minimize_states = dfa.num_states();
    let pre_minimize_transitions = dfa_transition_count(&dfa);
    let force_tokenizer_minimize = std::env::var_os("GLRMASK_FORCE_TOKENIZER_MINIMIZE").is_some();
    let final_dfa = if used_product_dfa && !force_tokenizer_minimize {
        dfa
    } else {
        dfa.minimize()
    };
    let forced_minimized_states = if profile_detail {
        if used_product_dfa && !force_tokenizer_minimize {
            Some(final_dfa.minimize().num_states())
        } else {
            Some(final_dfa.num_states())
        }
    } else {
        None
    };
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] combined groups={} visible_groups={} product_dfa={} shared_multi_nfa={} pre_minimize_states={} pre_minimize_transitions={} final_states={} final_transitions={} forced_minimized_states={}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            use_shared_multi_nfa,
            pre_minimize_states,
            pre_minimize_transitions,
            final_dfa.num_states(),
            dfa_transition_count(&final_dfa),
            forced_minimized_states.unwrap_or(final_dfa.num_states())
        );
    }
    final_dfa
}

pub fn build_regex(exprs: &[Expr]) -> Regex {
    let dfa = compile_with_plan(build_exclusion_compile_plan(exprs));

    Regex { dfa }
}

pub fn build_regex_with_profile_labels(exprs: &[Expr], visible_labels: &[String]) -> Regex {
    let dfa = compile_with_plan(build_exclusion_compile_plan_with_labels(
        exprs,
        Some(visible_labels),
    ));

    Regex { dfa }
}

fn product_state_metadata(
    components: &[ProductComponent],
    state_tuple: &ProductStateTuple,
) -> (BitSet, BitSet) {
    let num_groups = components.len();
    let mut finalizers = BitSet::new(num_groups);
    let mut future = BitSet::new(num_groups);

    for &(group_id, state) in state_tuple {
        let group_id = group_id as usize;
        match &components[group_id] {
            ProductComponent::Materialized(dfa) => {
                if dfa.finalizers(state).contains(0) {
                    finalizers.set(group_id);
                }
                if dfa.possible_future_group_ids(state).contains(0) {
                    future.set(group_id);
                }
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, min, max } => {
                let base_state_count = base_dfa.num_states() as u32;
                let copy_count = state / base_state_count;
                let base_state = state % base_state_count;
                if base_state == 0 && copy_count >= *min {
                    finalizers.set(group_id);
                }
                if copy_count < *max {
                    future.set(group_id);
                }
            }
        }
    }

    (finalizers, future)
}

fn explicit_dead_sink_state(dfa: &DFA) -> Option<u32> {
    for (state_id, state) in dfa.states().iter().enumerate() {
        if !state.finalizers.is_empty() {
            continue;
        }

        let mut transition_count = 0usize;
        let mut loops_to_self = true;
        for (_, &target) in state.transitions.iter() {
            transition_count += 1;
            if target != state_id as u32 {
                loops_to_self = false;
                break;
            }
        }

        if loops_to_self && transition_count == 256 {
            return Some(state_id as u32);
        }
    }

    None
}

fn expr_is_epsilon_only(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::Seq(parts) => parts.iter().all(expr_is_epsilon_only),
        Expr::Shared(inner) => expr_is_epsilon_only(inner),
        Expr::U8Class(_)
        | Expr::Dfa(_)
        | Expr::Choice(_)
        | Expr::Exclude { .. }
        | Expr::Intersect { .. }
        | Expr::Repeat { .. } => false,
    }
}

fn optional_choice_non_epsilon(expr: &Expr) -> Option<&Expr> {
    let options = match expr {
        Expr::Shared(inner) => return optional_choice_non_epsilon(inner),
        Expr::Choice(options) if options.len() == 2 => options,
        _ => return None,
    };

    if expr_is_epsilon_only(&options[0]) {
        Some(&options[1])
    } else if expr_is_epsilon_only(&options[1]) {
        Some(&options[0])
    } else {
        None
    }
}

fn optional_tail_parts(expr: &Expr) -> Option<Vec<Expr>> {
    let non_epsilon = optional_choice_non_epsilon(expr)?;
    match non_epsilon {
        Expr::Shared(inner) => optional_tail_parts(inner).or_else(|| Some(vec![inner.as_ref().clone()])),
        Expr::Seq(parts) => Some(parts.clone()),
        other => Some(vec![other.clone()]),
    }
}

fn mark_state_accepting(dfa: &mut DFA, state_id: u32) {
    dfa.ensure_group_capacity(1);

    let mut finalizers = dfa.finalizers(state_id).clone();
    finalizers.set(0);
    let mut future = dfa.possible_future_group_ids(state_id).clone();
    future.set(0);
    dfa.overwrite_state_metadata(state_id, finalizers, future);
}

fn compile_product_component_dfa_direct(expr: &Expr) -> Option<(DFA, bool)> {
    match expr {
        Expr::Shared(inner) => compile_product_component_dfa_direct(inner),
        Expr::Dfa(dfa) => Some((dfa.as_ref().clone(), true)),
        Expr::Choice(_) => {
            let non_epsilon = optional_choice_non_epsilon(expr)?;
            let (mut dfa, needs_future_recompute) =
                compile_product_component_dfa_direct(non_epsilon)?;
            mark_state_accepting(&mut dfa, 0);
            Some((dfa, needs_future_recompute))
        }
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => build_bounded_repeat_dfa(expr, *min, *max).map(|dfa| (dfa, false)),
        Expr::Seq(parts) => build_bounded_repeat_with_suffix_dfa(parts)
            .or_else(|| build_bounded_repeat_with_regex_suffix(parts))
            .or_else(|| build_prefixed_bounded_repeat_with_suffix_dfa(parts)),
        _ => None,
    }
}

fn compile_product_component_dfa(expr: &Expr) -> DFA {
    compile_with_plan(build_exclusion_compile_plan(std::slice::from_ref(expr)))
}

enum ProductComponent {
    Materialized(DFA),
    VirtualBoundedRepeat {
        base_dfa: DFA,
        min: u32,
        max: u32,
    },
}

enum ProductComponentClassTransitions {
    Materialized(Vec<Vec<(u8, u32)>>),
    VirtualBoundedRepeat(Vec<Vec<(u8, u32)>>),
}

impl ProductComponent {
    fn partition_dfa(&self) -> &DFA {
        match self {
            ProductComponent::Materialized(dfa) => dfa,
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => base_dfa,
        }
    }

    fn dead_state(&self) -> Option<u32> {
        match self {
            ProductComponent::Materialized(dfa) => explicit_dead_sink_state(dfa),
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => explicit_dead_sink_state(base_dfa),
        }
    }
}

fn compile_product_component(expr: &Expr) -> ProductComponent {
    match expr {
        Expr::Shared(inner) => compile_product_component(inner),
        Expr::Repeat {
            expr: repeat_expr,
            min,
            max: Some(max),
        } => {
            if let Some(base_dfa) = compile_direct_bounded_repeat_base_dfa(repeat_expr, *max) {
                return ProductComponent::VirtualBoundedRepeat {
                    base_dfa,
                    min: *min as u32,
                    max: *max as u32,
                };
            }

            ProductComponent::Materialized(compile_product_component_dfa(expr))
        }
        _ => ProductComponent::Materialized(compile_product_component_dfa(expr)),
    }
}

fn build_product_dfa(exprs: &[Expr], profile_labels: Option<&[ProductComponentProfileLabel]>) -> DFA {
    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_started_at = Instant::now();
    let components: Vec<ProductComponent> = if profile_detail {
        let mut components = Vec::with_capacity(exprs.len());
        for (index, expr) in exprs.iter().enumerate() {
            let component_started_at = Instant::now();
            let component = compile_product_component(expr);
            let states = component.partition_dfa().num_states();
            let transitions = dfa_transition_count(component.partition_dfa());
            let label = profile_labels
                .and_then(|labels| labels.get(index))
                .map(|label| {
                    format!(
                        " name={:?} origin={} shared={}",
                        label.name,
                        label.origin,
                        label.shared
                    )
                })
                .unwrap_or_else(|| format!(" expr={:?}", expr_profile_summary(expr)));
            eprintln!(
                "[glrmask/profile][tokenizer] product_component_compiled index={} states={} transitions={} compile_ms={:.3}{}",
                index,
                states,
                transitions,
                component_started_at.elapsed().as_secs_f64() * 1000.0,
                label
            );
            components.push(component);
        }
        components
    } else {
        exprs.par_iter().map(compile_product_component).collect()
    };
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] product_components groups={} compile_components_ms={:.3}",
            components.len(),
            profile_started_at.elapsed().as_secs_f64() * 1000.0
        );
        for (index, component) in components.iter().enumerate() {
            let states = component.partition_dfa().num_states();
            let transitions = dfa_transition_count(component.partition_dfa());
            let label = profile_labels
                .and_then(|labels| labels.get(index))
                .map(|label| {
                    format!(
                        " name={:?} origin={} shared={}",
                        label.name,
                        label.origin,
                        label.shared
                    )
                })
                .unwrap_or_else(|| format!(" expr={:?}", expr_profile_summary(&exprs[index])));
            match component {
                ProductComponent::Materialized(_) => {
                    eprintln!(
                        "[glrmask/profile][tokenizer] component index={} kind=materialized states={} transitions={}{}",
                        index, states, transitions, label
                    );
                }
                ProductComponent::VirtualBoundedRepeat { min, max, .. } => {
                    eprintln!(
                        "[glrmask/profile][tokenizer] component index={} kind=virtual_bounded_repeat base_states={} base_transitions={} min={} max={}{}",
                        index, states, transitions, min, max, label
                    );
                }
            }
        }
    }
    let num_groups = components.len();
    let component_dead_states: Vec<Option<u32>> = components
        .iter()
        .map(ProductComponent::dead_state)
        .collect();
    let (class_map, class_members) = compute_product_equivalence_classes(&components);
    let num_classes = class_members.len();
    let component_class_transitions = build_product_class_transitions(&components, &class_map);
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(num_groups);

    assert!(num_groups <= u32::MAX as usize, "too many product DFA groups");
    let mut start_tuple = ProductStateTuple::with_capacity(num_groups);
    for group_id in 0..num_groups {
        start_tuple.push((group_id as u32, 0u32));
    }
    let (start_finalizers, start_future) = product_state_metadata(&components, &start_tuple);
    dfa.overwrite_state_metadata(0, start_finalizers, start_future);

    let mut state_map = FxHashMap::<ProductStateTuple, u32>::default();
    let mut worklist = VecDeque::new();
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    // Pre-allocated buffers for class transition tuples (reused across states)
    let mut class_buffers: Vec<ProductStateTuple> = (0..num_classes)
        .map(|_| ProductStateTuple::new())
        .collect();
    let mut class_active = vec![false; num_classes];
    let mut used_classes = Vec::<usize>::new();
    let mut growth_recorder = profile_detail.then(|| ProductGrowthRecorder::new(num_groups));
    state_map.insert(start_tuple.clone(), 0);
    if let Some(recorder) = growth_recorder.as_mut() {
        recorder.record(num_groups, &start_tuple);
    }
    worklist.push_back((0, start_tuple));

    while let Some((current_state, state_tuple)) = worklist.pop_front() {
        for &(group_id, component_state) in &state_tuple {
            let group_index = group_id as usize;

            match (&components[group_index], &component_class_transitions[group_index]) {
                (
                    ProductComponent::Materialized(_),
                    ProductComponentClassTransitions::Materialized(class_transitions),
                ) => {
                    for &(class_id, target) in &class_transitions[component_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                        }
                        if component_dead_states[group_index] == Some(target) {
                            continue;
                        }

                        class_buffers[class_index].push((group_id, target));
                    }
                }
                (
                    ProductComponent::VirtualBoundedRepeat { base_dfa, max, .. },
                    ProductComponentClassTransitions::VirtualBoundedRepeat(base_class_transitions),
                ) => {
                    let base_state_count = base_dfa.num_states() as u32;
                    let copy_count = component_state / base_state_count;
                    if copy_count >= *max {
                        continue;
                    }

                    let base_state = component_state % base_state_count;
                    if base_dfa.finalizers(base_state).contains(0) {
                        continue;
                    }
                    for &(class_id, target_base) in &base_class_transitions[base_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                        }
                        if component_dead_states[group_index] == Some(target_base) {
                            continue;
                        }

                        let target = if base_dfa.finalizers(target_base).contains(0) {
                            (copy_count + 1) * base_state_count
                        } else {
                            copy_count * base_state_count + target_base
                        };

                        class_buffers[class_index].push((group_id, target));
                    }
                }
                _ => unreachable!("component and class-transition kinds must match"),
            }
        }

        let mut class_transitions = Vec::with_capacity(used_classes.len());
        for &class_index in &used_classes {
            let next_tuple = &class_buffers[class_index];
            let next_state = if let Some(&existing) = state_map.get(next_tuple) {
                existing
            } else {
                let new_state = dfa.add_state();
                let (finalizers, future) = product_state_metadata(&components, next_tuple);
                dfa.overwrite_state_metadata(new_state, finalizers, future);
                state_map.insert(next_tuple.clone(), new_state);
                if let Some(recorder) = growth_recorder.as_mut() {
                    recorder.record(num_groups, next_tuple);
                }
                pending_class_transitions.push(Vec::new());
                worklist.push_back((new_state, next_tuple.clone()));
                new_state
            };
            class_transitions.push((class_index as u8, next_state));
            class_buffers[class_index].clear();
            class_active[class_index] = false;
        }
        used_classes.clear();
        pending_class_transitions[current_state as usize] = class_transitions;
    }

    if profile_detail {
        if let Some(recorder) = growth_recorder.as_ref() {
            let mut states_before = 0usize;
            for (index, states_after) in recorder.prefix_counts().iter().copied().enumerate() {
                let label = profile_labels
                    .and_then(|labels| labels.get(index))
                    .map(|label| {
                        format!(
                            " name={:?} origin={} shared={}",
                            label.name,
                            label.origin,
                            label.shared
                        )
                    })
                    .unwrap_or_else(|| format!(" expr={:?}", expr_profile_summary(&exprs[index])));
                eprintln!(
                    "[glrmask/profile][tokenizer/product-growth] component_index={} states_before={} states_after={} delta_states={}{}",
                    index,
                    states_before,
                    states_after,
                    states_after.saturating_sub(states_before),
                    label
                );
                states_before = states_after;
            }
        }
        eprintln!(
            "[glrmask/profile][tokenizer] product_reachable states={} classes={} construct_ms={:.3}",
            dfa.num_states(),
            num_classes,
            profile_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> = pending_class_transitions
        .into_par_iter()
        .map(|class_transitions| {
            let byte_capacity: usize = class_transitions
                .iter()
                .map(|(class_id, _)| class_members[*class_id as usize].len())
                .sum();
            let mut transitions = Vec::with_capacity(byte_capacity);
            for (class_id, target) in class_transitions {
                for &byte in &class_members[class_id as usize] {
                    transitions.push((byte, target));
                }
            }
            if transitions.len() > 1 {
                transitions.sort_unstable_by_key(|entry| entry.0);
            }
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
        })
        .collect();

    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }

    dfa
}

fn compute_product_equivalence_classes(components: &[ProductComponent]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for component in components {
        let dfa = component.partition_dfa();
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

fn build_product_class_transitions_for_dfa(dfa: &DFA, class_map: &[u8]) -> Vec<Vec<(u8, u32)>> {
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
}

fn build_product_class_transitions(
    components: &[ProductComponent],
    class_map: &[u8],
) -> Vec<ProductComponentClassTransitions> {
    components
        .iter()
        .map(|component| match component {
            ProductComponent::Materialized(dfa) => {
                ProductComponentClassTransitions::Materialized(build_product_class_transitions_for_dfa(
                    dfa, class_map,
                ))
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => {
                ProductComponentClassTransitions::VirtualBoundedRepeat(
                    build_product_class_transitions_for_dfa(base_dfa, class_map),
                )
            }
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

/// Compile multiple expressions into a single NFA (without determinization).
///
/// Each expression's index becomes its group ID.
pub fn build_regex_nfa(exprs: &[Expr]) -> NFA {
    build_regex_nfa_impl(exprs)
}

fn build_regex_nfa_impl(exprs: &[Expr]) -> NFA {
    let optimized_exprs: Vec<Expr> = exprs.iter().cloned().map(Expr::optimize).collect();

    let mut nfa = NFA::new(1);

    if let Some((prefix, remainders)) = common_prefix_factor(&optimized_exprs) {
        let split = nfa.add_state();
        append_compiled_expr(&prefix, &mut nfa, 0, split);

        for (group_id, remainder) in remainders.iter().enumerate() {
            match remainder {
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
    use super::build_regex;
    use super::compile_product_component_dfa_direct;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::regex::parse_regex;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::ds::u8set::U8Set;
    use std::sync::Arc;

    fn byte_expr(byte: u8) -> Expr {
        Expr::U8Seq(vec![byte])
    }

    fn byte_choice(bytes: &[u8]) -> Expr {
        Expr::Choice(bytes.iter().copied().map(byte_expr).collect())
    }

    fn terminal_matches(expr: Expr, input: &[u8]) -> bool {
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
        };
        let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
        exec.matches
            .iter()
            .any(|matched| matched.id == 0 && matched.width == input.len())
    }

    #[test]
    fn nested_exclude_in_exclusion_branch_compiles() {
        let nested_residual = Expr::Exclude {
            expr: Box::new(byte_choice(b"ab")),
            exclude: Box::new(byte_expr(b'a')),
        };
        assert!(!terminal_matches(nested_residual.clone(), b"a"));
        assert!(terminal_matches(nested_residual.clone(), b"b"));
        assert!(!terminal_matches(nested_residual.clone(), b"c"));

        let expr = Expr::Exclude {
            expr: Box::new(byte_choice(b"bc")),
            exclude: Box::new(nested_residual),
        };

        assert!(!terminal_matches(expr.clone(), b"a"));
        assert!(!terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"c"));
    }

    #[test]
    fn nested_intersect_in_exclusion_branch_compiles() {
        let nested_intersection = Expr::Intersect {
            expr: Box::new(byte_choice(b"ab")),
            intersect: Box::new(byte_expr(b'b')),
        };
        assert!(!terminal_matches(nested_intersection.clone(), b"a"));
        assert!(terminal_matches(nested_intersection.clone(), b"b"));
        assert!(!terminal_matches(nested_intersection.clone(), b"c"));

        let expr = Expr::Exclude {
            expr: Box::new(byte_choice(b"bc")),
            exclude: Box::new(nested_intersection),
        };

        assert!(!terminal_matches(expr.clone(), b"a"));
        assert!(!terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"c"));
    }

    #[test]
    fn standalone_exact_repeat_matches_only_at_full_length() {
        let expr = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 0),
                "exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 0 && matched.width == 16),
            "exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn product_exact_repeat_matches_only_at_full_length() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let exact_repeat = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };

        let regex = build_regex(&[space.clone(), exact_repeat.clone()]);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 2,
            exprs: Some(Arc::from(vec![space, exact_repeat].into_boxed_slice())),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "product exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 16),
            "product exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn product_vbr_exact_repeat_matches_only_at_full_length() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let exact_repeat = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 32,
            max: Some(32),
        };

        let regex = build_regex(&[space.clone(), exact_repeat.clone()]);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 2,
            exprs: Some(Arc::from(vec![space, exact_repeat].into_boxed_slice())),
        };

        for len in [1usize, 2, 31] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "product VBR exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 32];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 32),
            "product VBR exact repeat did not match at len 32: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn glrm_chunk16_terminal_family_keeps_exact_repeat_nonfinal_until_16() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let quote = Expr::U8Seq(vec![b'"']);
        let exact_16 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };
        let upto_16 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(16),
        };
        let upto_close_16 = Expr::Seq(vec![upto_16.clone(), quote.clone()]);
        let upto_3 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(3),
        };
        let upto_close_3 = Expr::Seq(vec![upto_3.clone(), quote.clone()]);

        let exprs = vec![
            space.clone(),
            exact_16.clone(),
            upto_16,
            upto_close_16,
            upto_3,
            upto_close_3,
            quote,
        ];
        let regex = build_regex(&exprs);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: exprs.len() as u32,
            exprs: Some(Arc::from(exprs.into_boxed_slice())),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "GLRM family exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 16),
            "GLRM family exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn product_vbr_with_literal_prefix_uses_direct_bounded_repeat_tail() {
        let quote = Expr::U8Seq(vec![b'"']);
        let spaces = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(32),
        };
        let expr = Expr::Seq(vec![quote.clone(), spaces, quote]);

        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat did not use direct product component path");
        };
        assert!(
            dfa.num_states() <= 80,
            "direct prefixed bounded repeat DFA unexpectedly large: {} states",
            dfa.num_states(),
        );

        let tokenizer = Tokenizer {
            dfa,
            num_terminals: 1,
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
        };

        for len in [0usize, 1, 31, 32] {
            let mut input = Vec::with_capacity(len + 2);
            input.push(b'"');
            input.extend(std::iter::repeat(b' ').take(len));
            input.push(b'"');
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed bounded repeat did not match length {len}: {:?}",
                exec.matches,
            );
        }
    }

    #[test]
    fn product_vbr_with_literal_prefix_and_regex_suffix_matches() {
        let quote = Expr::U8Seq(vec![b'"']);
        let word = Expr::U8Class(U8Set::single(b'a'));
        let space = Expr::U8Class(U8Set::single(b' '));
        let word_run = Expr::Repeat {
            expr: Box::new(word.clone()),
            min: 1,
            max: None,
        };
        let space_run = Expr::Repeat {
            expr: Box::new(space),
            min: 1,
            max: None,
        };
        let pair = Expr::Seq(vec![word_run.clone(), space_run]);
        let repeated_pairs = Expr::Repeat {
            expr: Box::new(pair),
            min: 0,
            max: Some(49),
        };
        let expr = Expr::Seq(vec![quote, repeated_pairs, word_run]);

        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 400,
            "direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );

        let tokenizer = Tokenizer {
            dfa,
            num_terminals: 1,
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
        };

        for input in [b"\"a".as_slice(), b"\"aa", b"\"a a", b"\"aa  aaa"] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed bounded repeat with suffix did not match {:?}: {:?}",
                std::str::from_utf8(input).unwrap(),
                exec.matches,
            );
        }

        let exec = tokenizer.execute_from_state(b"\"a ", tokenizer.initial_state());
        assert!(
            !exec
                .matches
                .iter()
                .any(|matched| matched.id == 0 && matched.width == 3),
            "prefixed bounded repeat with suffix matched trailing space: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn prefixed_bounded_repeat_with_regex_suffix_uses_direct_path_without_repeat_cutoff() {
        let quote = Expr::U8Seq(vec![b'"']);
        let word = Expr::U8Class(U8Set::single(b'a'));
        let space = Expr::U8Class(U8Set::single(b' '));
        let word_run = Expr::Repeat {
            expr: Box::new(word.clone()),
            min: 1,
            max: None,
        };
        let space_run = Expr::Repeat {
            expr: Box::new(space),
            min: 1,
            max: None,
        };
        let pair = Expr::Seq(vec![word_run.clone(), space_run]);
        let repeated_pairs = Expr::Repeat {
            expr: Box::new(pair),
            min: 0,
            max: Some(29),
        };
        let expr = Expr::Seq(vec![quote, repeated_pairs, word_run]);

        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 300,
            "direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );

        let repeated_pairs = Expr::Repeat {
            expr: Box::new(Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
                    min: 1,
                    max: None,
                },
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
                    min: 1,
                    max: None,
                },
            ])),
            min: 0,
            max: Some(2),
        };
        let expr = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            repeated_pairs,
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
                min: 1,
                max: None,
            },
        ]);
        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("small prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 40,
            "small direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );
    }

    fn prefixed_optional_word_list_expr(max_pairs: usize) -> Expr {
        let nonspace_plus = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
            min: 1,
            max: None,
        };
        let space_plus = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 1,
            max: None,
        };
        let body = Expr::Seq(vec![nonspace_plus.clone(), space_plus]);
        let repeated = Expr::Repeat {
            expr: Box::new(body),
            min: 0,
            max: Some(max_pairs),
        };

        Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::Seq(vec![repeated, nonspace_plus]),
            ]),
        ])
    }

    #[test]
    fn prefixed_optional_choice_uses_direct_component_path_for_bounded_repeat_suffix() {
        let expr = prefixed_optional_word_list_expr(199);

        let Some((dfa, _)) = compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed optional wrapper did not use direct product component path");
        };

        assert!(
            dfa.num_states() < 10_000,
            "prefixed optional direct-path DFA unexpectedly large: {} states",
            dfa.num_states(),
        );
        assert!(dfa.finalizers(1).contains(0));
    }

    #[test]
    fn prefixed_optional_word_list_semantics() {
        let expr = prefixed_optional_word_list_expr(2);
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
        };

        for input in [b"\"".as_slice(), b"\"a", b"\"a a", b"\"a  a"] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed optional word-list did not match {:?}: {:?}",
                std::str::from_utf8(input).unwrap(),
                exec.matches,
            );
        }

        let exec = tokenizer.execute_from_state(b"\" a", tokenizer.initial_state());
        assert!(
            !exec
                .matches
                .iter()
                .any(|matched| matched.id == 0 && matched.width == 3),
            "prefixed optional word-list matched leading space unexpectedly: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn direct_tokenizer_states_for_o35155_regex_groups() {
        let expr_1 = parse_regex(r"(\w+\.)+\d+", false);
        let expr_2 = parse_regex(r"\w+_(\w_)?\d+", false);
        let expr_3 = parse_regex(r"(\w|-){12}", false);
        let expr_5 = parse_regex(r"\d{7,9}", false);

        let regex_1235 = build_regex(&[
            expr_1.clone(),
            expr_2.clone(),
            expr_3.clone(),
            expr_5.clone(),
        ]);
        let regex_125 = build_regex(&[expr_1, expr_2, expr_5]);

        eprintln!(
            "o35155 direct tokenizer states for regex groups 1,2,3,5: states={} transitions={}",
            regex_1235.num_states(),
            regex_1235.num_transitions()
        );
        eprintln!(
            "o35155 direct tokenizer states for regex groups 1,2,5: states={} transitions={}",
            regex_125.num_states(),
            regex_125.num_transitions()
        );
    }

}
