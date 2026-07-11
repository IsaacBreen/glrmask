use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ds::{bitset::BitSet, u8set::U8Set};

use super::ast::Expr;
use super::tokenizer::Tokenizer;
use super::dfa::{DEAD, DFA};
use super::nfa::NFA;

type ProductStateTuple = SmallVec<[(u32, u32); 12]>;

fn unwrap_shared(expr: &Expr) -> &Expr {
    match expr {
        Expr::Shared(inner) => unwrap_shared(inner),
        other => other,
    }
}

fn seq_from_parts(mut parts: Vec<Expr>) -> Expr {
    match parts.len() {
        0 => Expr::Epsilon,
        1 => parts.pop().unwrap(),
        _ => Expr::Seq(parts),
    }
}

fn choice_first_part(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Shared(inner) => choice_first_part(inner),
        Expr::Seq(parts) => parts.first(),
        Expr::Epsilon => None,
        other => Some(other),
    }
}

fn choice_without_first_part(expr: &Expr) -> Expr {
    match expr {
        Expr::Shared(inner) => choice_without_first_part(inner),
        Expr::Seq(parts) => seq_from_parts(parts[1..].to_vec()),
        Expr::Epsilon => Expr::Epsilon,
        _ => Expr::Epsilon,
    }
}

fn choice_last_part(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Shared(inner) => choice_last_part(inner),
        Expr::Seq(parts) => parts.last(),
        Expr::Epsilon => None,
        other => Some(other),
    }
}

fn choice_without_last_part(expr: &Expr) -> Expr {
    match expr {
        Expr::Shared(inner) => choice_without_last_part(inner),
        Expr::Seq(parts) => seq_from_parts(parts[..parts.len() - 1].to_vec()),
        Expr::Epsilon => Expr::Epsilon,
        _ => Expr::Epsilon,
    }
}

fn factor_choice_common_prefix(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let prefix = choice_first_part(options.first()?)?.clone();
    if !options
        .iter()
        .all(|option| choice_first_part(option) == Some(&prefix))
    {
        return None;
    }

    let remainders = options
        .iter()
        .map(choice_without_first_part)
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        prefix,
        factor_regex_expr(Expr::Choice(remainders)),
    ]))
}

fn factor_choice_common_suffix(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let suffix = choice_last_part(options.first()?)?.clone();
    if !options
        .iter()
        .all(|option| choice_last_part(option) == Some(&suffix))
    {
        return None;
    }

    let prefixes = options
        .iter()
        .map(choice_without_last_part)
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        factor_regex_expr(Expr::Choice(prefixes)),
        suffix,
    ]))
}

fn factor_choice_literals(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let first_byte = match unwrap_shared(options.first()?) {
        Expr::U8Seq(bytes) if !bytes.is_empty() => bytes[0],
        _ => return None,
    };
    for option in options {
        match unwrap_shared(option) {
            Expr::U8Seq(bytes) if !bytes.is_empty() && bytes[0] == first_byte => {}
            _ => return None,
        }
    }

    let remainders = options
        .iter()
        .map(|option| match unwrap_shared(option) {
            Expr::U8Seq(bytes) => {
            if bytes.len() == 1 {
                Expr::Epsilon
            } else {
                Expr::U8Seq(bytes[1..].to_vec())
            }
            }
            _ => unreachable!("literal choice was validated above"),
        })
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        Expr::U8Seq(vec![first_byte]),
        factor_regex_expr(Expr::Choice(remainders)),
    ]))
}

pub(crate) fn factor_regex_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Seq(parts) => {
            let mut out = Vec::new();
            for part in parts {
                match factor_regex_expr(part) {
                    Expr::Seq(inner) => out.extend(inner),
                    Expr::Epsilon => {}
                    other => out.push(other),
                }
            }
            seq_from_parts(out)
        }
        Expr::Choice(options) => {
            let mut factored_options = options.into_iter().map(factor_regex_expr).collect::<Vec<_>>();

            if factored_options.len() == 1 {
                return factored_options.pop().unwrap();
            }

            // Prefix first handles A B1 C | A B2 C; suffix then handles
            // B1 C | B2 C. Each helper probes through references and only
            // clones a choice when it actually finds a factor.
            if let Some(factored) = factor_choice_literals(&factored_options) {
                return factored;
            }
            if let Some(factored) = factor_choice_common_prefix(&factored_options) {
                return factored;
            }
            if let Some(factored) = factor_choice_common_suffix(&factored_options) {
                return factored;
            }

            Expr::Choice(factored_options)
        }
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(factor_regex_expr(*expr)),
            min,
            max,
        },
        Expr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(factor_regex_expr(*expr)),
            exclude: Box::new(factor_regex_expr(*exclude)),
        },
        Expr::Intersect { expr, intersect } => Expr::Intersect {
            expr: Box::new(factor_regex_expr(*expr)),
            intersect: Box::new(factor_regex_expr(*intersect)),
        },
        Expr::Shared(inner) => factor_regex_expr((*inner).clone()),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => expr,
    }
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

#[derive(Default)]
struct NestedGroupOpCache {
    compiled: FxHashMap<Expr, Arc<DFA>>,
    cache_hits: usize,
    cache_misses: usize,
    compiled_ms: f64,
    max_compile_ms: f64,
}

fn materialize_nested_group_ops(expr: Expr, cache: &mut NestedGroupOpCache) -> Expr {
    match expr {
        expr @ (Expr::Exclude { .. } | Expr::Intersect { .. }) => {
            if let Some(compiled) = cache.compiled.get(&expr) {
                cache.cache_hits += 1;
                return Expr::Dfa(compiled.clone());
            }

            cache.cache_misses += 1;
            let started_at = Instant::now();
            let compiled = Arc::new(compile_with_plan(
                build_exclusion_compile_plan_with_labels_and_cache(
                    std::slice::from_ref(&expr),
                    None,
                    cache,
                ),
            ));
            let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            cache.compiled_ms += elapsed_ms;
            cache.max_compile_ms = cache.max_compile_ms.max(elapsed_ms);
            cache.compiled.insert(expr, compiled.clone());
            Expr::Dfa(compiled)
        }
        Expr::Seq(parts) => Expr::Seq(
            parts
                .into_iter()
                .map(|part| materialize_nested_group_ops(part, cache))
                .collect(),
        ),
        Expr::Choice(options) => Expr::Choice(
            options
                .into_iter()
                .map(|option| materialize_nested_group_ops(option, cache))
                .collect(),
        ),
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(materialize_nested_group_ops(*expr, cache)),
            min,
            max,
        },
        Expr::Shared(inner) => {
            let rewritten = materialize_nested_group_ops((*inner).clone(), cache);
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
    let mut nested_group_op_cache = NestedGroupOpCache::default();
    let plan = build_exclusion_compile_plan_with_labels_and_cache(
        exprs,
        visible_labels,
        &mut nested_group_op_cache,
    );
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        && nested_group_op_cache.cache_misses > 0
    {
        eprintln!(
            "[glrmask/profile][tokenizer] nested_group_ops cache_entries={} cache_hits={} cache_misses={} compiled_ms={:.3} max_compile_ms={:.3}",
            nested_group_op_cache.compiled.len(),
            nested_group_op_cache.cache_hits,
            nested_group_op_cache.cache_misses,
            nested_group_op_cache.compiled_ms,
            nested_group_op_cache.max_compile_ms,
        );
    }
    plan
}

fn build_exclusion_compile_plan_with_labels_and_cache(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    nested_group_op_cache: &mut NestedGroupOpCache,
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
        let base = materialize_nested_group_ops(base, nested_group_op_cache);
        let excluded = excluded
            .into_iter()
            .map(|expr| materialize_nested_group_ops(expr, nested_group_op_cache))
            .collect::<Vec<_>>();
        let intersections = intersections
            .into_iter()
            .map(|expr| materialize_nested_group_ops(expr, nested_group_op_cache))
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

/// TODO: replace this compact product with an exact subset construction if we
/// need to support ambiguous body/suffix boundaries or nullable suffixes in the
/// fast path. Until then, this function must return None for any case requiring
/// multiple simultaneous boundary choices.
///
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
/// This is a fast path, not a general NFA subset construction. It must return
/// `None` whenever the compact state would need to represent multiple live
/// body/suffix boundary choices.
///
/// At body completion, the counter increments and the suffix may start. If two
/// live paths cannot be represented by one `(body_state, suffix_state, counter)`
/// tuple, this function falls back to the general compiler.
fn build_bounded_repeat_with_regex_suffix(parts: &[Expr]) -> Option<(DFA, bool)> {
    if parts.len() < 2 {
        return None;
    }

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

    // With max == 0 the body is not allowed to consume anything. The compact
    // product construction starts with a live body state, so it would otherwise
    // permit one body occurrence before the suffix. Let the general path handle
    // this exact zero-repeat case.
    if max == 0 {
        return None;
    }

    let body_dfa = compile_expr_to_dfa(repeat_expr);
    if body_dfa.num_states() == 0 || !body_dfa.finalizers(0).is_empty() {
        return None;
    }

    let suffix_expr = if parts_ref.len() == 2 {
        parts_ref[1].clone()
    } else {
        Expr::Seq(parts_ref[1..].iter().map(|e| (*e).clone()).collect())
    };
    let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
    if suffix_dfa.num_states() == 0 {
        return None;
    }

    // Nullable suffixes require considering a body/suffix boundary without
    // consuming another byte. This compact construction only starts fresh
    // suffix paths while processing a byte, so it can miss finalization at
    // end-of-input. Fall back to the general construction.
    if !suffix_dfa.finalizers(0).is_empty() {
        return None;
    }

    let max_product =
        (body_dfa.num_states() + 1) * (suffix_dfa.num_states() + 1) * (max + 1);
    if max_product > 500_000 {
        return None;
    }

    let body_dead = body_dfa.num_states() as u32;
    let suffix_dead = suffix_dfa.num_states() as u32;

    let mut state_map: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, (u32, u32, u32))> = VecDeque::new();
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);

    let start_suffix = if min == 0 { 0u32 } else { suffix_dead };
    let start_key = (0u32, start_suffix, 0u32);
    state_map.insert(start_key, 0);
    worklist.push_back((0, start_key));

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

        let mut transitions = Vec::new();
        for byte_val in 0u16..=255 {
            let x = byte_val as u8;

            let b_next = if b < body_dead {
                body_dfa.step(b, x).map_or(body_dead, |t| t)
            } else {
                body_dead
            };

            // If the body is accepting before this byte, there is an implicit
            // boundary here: either continue the repeated body with `x`, or
            // finish the body and let the suffix consume `x`.
            //
            // This product state tracks only one body state and one suffix
            // state. If both paths are live, keeping only the body path is a
            // lossy greedy approximation.
            //
            // Minimal counterexample:
            //
            //   ("a"+)? "a"
            //
            // on input "aa". At the second "a", the body can continue, but the
            // suffix can also start. Dropping the suffix path falsely rejects.
            if body_is_accept && b_next != body_dead {
                let new_c = c + 1;
                if new_c >= min as u32 && new_c <= max as u32 {
                    let fresh_s = suffix_dfa
                        .step(0, x)
                        .map_or(suffix_dead, |t| t);
                    if fresh_s != suffix_dead {
                        return None;
                    }
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
                        (true, true) if old_s_next != fresh_s => return None,
                        (true, _) => old_s_next,
                        (_, true) => fresh_s,
                        _ => suffix_dead,
                    };
                    (new_b, new_s, new_c)
                } else {
                    let s_next = if s < suffix_dead {
                        suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
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

    let dfa = dfa.minimize();
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
    pub(super) dfa: DFA,
}

impl Regex {
    pub(crate) fn into_tokenizer(self, num_terminals: u32, exprs: Option<std::sync::Arc<[Expr]>>) -> Tokenizer {
        Tokenizer {
            dfa: self.dfa,
            num_terminals,
            exprs,
        }
    }

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
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_started_at = Instant::now();
    let group_set_started_at = Instant::now();
    let group_sets: Vec<U8Set> = plan
        .compiled_exprs
        .iter()
        .map(|expr| expr_u8set(expr))
        .collect();
    let group_set_ms = profile_timing
        .then(|| group_set_started_at.elapsed().as_secs_f64() * 1000.0);
    let use_shared_multi_nfa = should_use_shared_multi_nfa(&plan);
    let used_product_dfa = plan.compiled_exprs.len() > 1 && !use_shared_multi_nfa;

    let dfa_build_started_at = Instant::now();
    let mut dfa = if plan.compiled_exprs.is_empty() {
        // A grammar can lower to the empty language, for example when a const
        // literal conflicts with sibling assertions. Keep a single non-final
        // start state so tokenizer users can still query/step the DFA safely.
        DFA::new(1)
    } else if used_product_dfa {
        build_product_dfa(&plan.compiled_exprs, plan.profile_labels.as_deref())
    } else if plan.compiled_exprs.len() > 1 {
        compile_multi_expr_dfa_via_nfa(&plan.compiled_exprs)
    } else {
        compile_single_expr_dfa(&plan.compiled_exprs[0])
    };
    let dfa_build_ms = profile_timing
        .then(|| dfa_build_started_at.elapsed().as_secs_f64() * 1000.0);

    let metadata_started_at = Instant::now();
    dfa.ensure_group_capacity(group_sets.len());
    for (group_id, set) in group_sets.into_iter().enumerate() {
        dfa.set_group_u8set(group_id as u32, set);
    }
    let metadata_ms = profile_timing
        .then(|| metadata_started_at.elapsed().as_secs_f64() * 1000.0);

    let group_ops_started_at = Instant::now();
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
    let group_ops_ms = profile_timing
        .then(|| group_ops_started_at.elapsed().as_secs_f64() * 1000.0);

    let project_started_at = Instant::now();
    let dfa = if plan.visible_groups < plan.compiled_exprs.len() {
        dfa.project_groups(plan.visible_groups)
    } else {
        dfa
    };
    let project_ms = profile_timing
        .then(|| project_started_at.elapsed().as_secs_f64() * 1000.0);

    let pre_minimize_states = dfa.num_states();
    let pre_minimize_transitions = dfa_transition_count(&dfa);
    let force_tokenizer_minimize = std::env::var_os("GLRMASK_FORCE_TOKENIZER_MINIMIZE").is_some();
    let minimize_started_at = Instant::now();
    let final_dfa = if used_product_dfa && !force_tokenizer_minimize {
        dfa
    } else {
        dfa.minimize()
    };
    let minimize_ms = profile_timing
        .then(|| minimize_started_at.elapsed().as_secs_f64() * 1000.0);
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
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] compile_plan groups={} visible_groups={} product_dfa={} shared_multi_nfa={} group_set_ms={:.3} dfa_build_ms={:.3} metadata_ms={:.3} group_ops_ms={:.3} project_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            use_shared_multi_nfa,
            group_set_ms.unwrap_or_default(),
            dfa_build_ms.unwrap_or_default(),
            metadata_ms.unwrap_or_default(),
            group_ops_ms.unwrap_or_default(),
            project_ms.unwrap_or_default(),
            minimize_ms.unwrap_or_default(),
            profile_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    final_dfa
}

pub fn build_regex(exprs: &[Expr]) -> Regex {
    build_regex_monolithic(exprs)
}

/// Compile all expressions into one traditional deterministic lexer.
pub(crate) fn build_regex_monolithic(exprs: &[Expr]) -> Regex {
    Regex {
        dfa: compile_with_plan(build_exclusion_compile_plan(exprs)),
    }
}

pub fn build_regex_with_profile_labels(exprs: &[Expr], visible_labels: &[String]) -> Regex {
    Regex {
        dfa: compile_with_plan(build_exclusion_compile_plan_with_labels(
            exprs,
            Some(visible_labels),
        )),
    }
}

/// Compile terminals in caller-selected deterministic partitions, then join
/// those partitions with epsilon edges from one global start state. Terminals
/// sharing a partition may be jointly determinized; terminals in different
/// partitions can never cause a cross-partition subset/product blow-up.
pub(crate) fn build_regex_partitioned(exprs: &[Expr], partitions: &[u32]) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(exprs, None, partitions, None, false),
    }
}

pub(crate) fn build_regex_partitioned_adaptive(
    exprs: &[Expr],
    partitions: &[u32],
    adaptive_partitions: &BTreeSet<u32>,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            None,
            partitions,
            Some(adaptive_partitions),
            true,
        ),
    }
}

pub(crate) fn build_regex_partitioned_with_profile_labels(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(exprs, Some(visible_labels), partitions, None, false),
    }
}

pub(crate) fn build_regex_partitioned_with_profile_labels_adaptive(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
    adaptive_partitions: &BTreeSet<u32>,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            Some(visible_labels),
            partitions,
            Some(adaptive_partitions),
            true,
        ),
    }
}

fn adaptive_lexer_state_limit() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(8_192)
}

fn adaptive_lexer_component_state_limit() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_COMPONENT_MAX_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(32_768)
}

fn adaptive_lexer_growth_percent() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_GROWTH_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(100)
}

fn adaptive_lexer_max_trials_per_terminal() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_TRIALS_PER_TERMINAL")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1)
}

fn adaptive_lexer_max_trials_per_component() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_TRIALS_PER_COMPONENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(2)
}

fn adaptive_lexer_max_component_batches() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_COMPONENT_BATCHES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4)
}

fn compile_terminal_ids(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: &[usize],
) -> DFA {
    let local_exprs = terminal_ids
        .iter()
        .map(|&terminal| exprs[terminal].clone())
        .collect::<Vec<_>>();
    let local_labels = visible_labels.map(|labels| {
        terminal_ids
            .iter()
            .map(|&terminal| labels[terminal].clone())
            .collect::<Vec<_>>()
    });
    compile_with_plan(build_exclusion_compile_plan_with_labels(
        &local_exprs,
        local_labels.as_deref(),
    ))
}

struct AdaptiveTerminalBatch {
    terminal_ids: Vec<usize>,
    dfa: DFA,
}

fn product_union_metadata(
    left: &DFA,
    left_state: u32,
    right: &DFA,
    right_state: u32,
) -> (BitSet, BitSet) {
    let total_groups = left.num_groups() + right.num_groups();
    let mut finalizers = BitSet::new(total_groups);
    let mut futures = BitSet::new(total_groups);

    if left_state != DEAD {
        for group in left.finalizers(left_state).iter() {
            finalizers.set(group);
        }
        for group in left.possible_future_group_ids(left_state).iter() {
            futures.set(group);
        }
    }
    if right_state != DEAD {
        let offset = left.num_groups();
        for group in right.finalizers(right_state).iter() {
            finalizers.set(offset + group);
        }
        for group in right.possible_future_group_ids(right_state).iter() {
            futures.set(offset + group);
        }
    }

    (finalizers, futures)
}

/// Jointly determinize two already-deterministic lexer components. This is an
/// incremental subset construction over a pair of component states. Unlike
/// recompiling the source regexes for every trial, it reuses all earlier work
/// and can stop before allocating the first state beyond `state_limit`.
fn try_product_union_dfas(left: &DFA, right: &DFA, state_limit: usize) -> Option<DFA> {
    assert!(state_limit > 0, "adaptive lexer state limit must be positive");
    debug_assert!(!left.has_epsilon_transitions());
    debug_assert!(!right.has_epsilon_transitions());

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(left.num_groups() + right.num_groups());
    for group in 0..left.num_groups() {
        combined.set_group_u8set(group as u32, *left.group_id_to_u8set(group as u32));
    }
    for group in 0..right.num_groups() {
        combined.set_group_u8set(
            (left.num_groups() + group) as u32,
            *right.group_id_to_u8set(group as u32),
        );
    }

    let start = (0u32, 0u32);
    let (finalizers, futures) = product_union_metadata(left, start.0, right, start.1);
    combined.overwrite_state_metadata(0, finalizers, futures);

    let mut state_map = FxHashMap::<(u32, u32), u32>::default();
    state_map.insert(start, 0);
    let mut worklist = VecDeque::from([(0u32, start)]);

    while let Some((combined_state, (left_state, right_state))) = worklist.pop_front() {
        let mut transitions = Vec::new();
        for byte in 0u8..=u8::MAX {
            let left_target = if left_state == DEAD {
                DEAD
            } else {
                left.states()[left_state as usize]
                    .transitions
                    .get(byte)
                    .copied()
                    .unwrap_or(DEAD)
            };
            let right_target = if right_state == DEAD {
                DEAD
            } else {
                right.states()[right_state as usize]
                    .transitions
                    .get(byte)
                    .copied()
                    .unwrap_or(DEAD)
            };
            if left_target == DEAD && right_target == DEAD {
                continue;
            }

            let key = (left_target, right_target);
            let target = if let Some(&existing) = state_map.get(&key) {
                existing
            } else {
                if combined.num_states() >= state_limit {
                    return None;
                }
                let new_state = combined.add_state();
                let (finalizers, futures) =
                    product_union_metadata(left, left_target, right, right_target);
                combined.overwrite_state_metadata(new_state, finalizers, futures);
                state_map.insert(key, new_state);
                worklist.push_back((new_state, key));
                new_state
            };
            transitions.push((byte, target));
        }
        combined.set_transitions_from_sorted_entries(combined_state, transitions);
    }

    Some(combined)
}

fn coalesce_adaptive_batches(
    inputs: Vec<AdaptiveTerminalBatch>,
    max_trials: usize,
    state_limit: usize,
    profile_level: &str,
) -> Vec<AdaptiveTerminalBatch> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = Instant::now();
    let growth_percent = adaptive_lexer_growth_percent();
    let input_states = inputs.iter().map(|batch| batch.dfa.num_states()).sum::<usize>();
    let input_transitions = inputs
        .iter()
        .map(|batch| dfa_transition_count(&batch.dfa))
        .sum::<usize>();
    let input_batches = inputs.len();
    let terminals = inputs
        .iter()
        .map(|batch| batch.terminal_ids.len())
        .sum::<usize>();

    let mut batches = Vec::<AdaptiveTerminalBatch>::new();
    let mut attempted_merges = 0usize;
    let mut accepted_merges = 0usize;
    let mut budget_rejections = 0usize;

    for input in inputs {
        let mut batch_order = (0..batches.len()).collect::<Vec<_>>();
        batch_order.sort_unstable_by_key(|&index| (batches[index].dfa.num_states(), index));

        let mut best: Option<(usize, Vec<usize>, DFA)> = None;
        for batch_index in batch_order.into_iter().take(max_trials) {
            attempted_merges += 1;
            let batch = &batches[batch_index];
            let separate_states = batch
                .dfa
                .num_states()
                .saturating_add(input.dfa.num_states());
            let growth_limit = separate_states
                .saturating_mul(growth_percent)
                .saturating_add(99)
                / 100;
            let mut candidate_ids = batch.terminal_ids.clone();
            candidate_ids.extend(input.terminal_ids.iter().copied());

            let Some(candidate) =
                try_product_union_dfas(&batch.dfa, &input.dfa, state_limit)
            else {
                budget_rejections += 1;
                continue;
            };
            let candidate = candidate.minimize();
            if candidate.num_states() > growth_limit {
                budget_rejections += 1;
                continue;
            }

            let replace = best.as_ref().is_none_or(|(_, _, best_dfa)| {
                (candidate.num_states(), dfa_transition_count(&candidate))
                    < (best_dfa.num_states(), dfa_transition_count(best_dfa))
            });
            if replace {
                best = Some((batch_index, candidate_ids, candidate));
            }
        }

        if let Some((batch_index, terminal_ids, dfa)) = best {
            batches[batch_index] = AdaptiveTerminalBatch { terminal_ids, dfa };
            accepted_merges += 1;
        } else {
            batches.push(input);
        }
    }

    if profile {
        let output_states = batches.iter().map(|batch| batch.dfa.num_states()).sum::<usize>();
        let output_transitions = batches
            .iter()
            .map(|batch| dfa_transition_count(&batch.dfa))
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_merge level={} terminals={} input_batches={} output_batches={} input_states={} output_states={} input_transitions={} output_transitions={} attempted_merges={} accepted_merges={} budget_rejections={} max_states={} max_growth_percent={} max_trials={} total_ms={:.3}",
            profile_level,
            terminals,
            input_batches,
            batches.len(),
            input_states,
            output_states,
            input_transitions,
            output_transitions,
            attempted_merges,
            accepted_merges,
            budget_rejections,
            state_limit,
            growth_percent,
            max_trials,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    batches
}

fn compile_adaptive_terminal_partition(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: Vec<usize>,
) -> Vec<(Vec<usize>, DFA)> {
    let max_trials = adaptive_lexer_max_trials_per_terminal();

    let singletons = terminal_ids
        .into_par_iter()
        .map(|terminal| {
            let ids = vec![terminal];
            let dfa = compile_terminal_ids(exprs, visible_labels, &ids);
            AdaptiveTerminalBatch {
                terminal_ids: ids,
                dfa,
            }
        })
        .collect::<Vec<_>>();
    coalesce_adaptive_batches(
        singletons,
        max_trials,
        adaptive_lexer_state_limit(),
        "pattern_terminals",
    )
        .into_iter()
        .map(|batch| (batch.terminal_ids, batch.dfa))
        .collect()
}

fn remap_component_groups(
    component: &DFA,
    terminal_ids: &[usize],
    total_groups: usize,
) -> DFA {
    debug_assert_eq!(component.num_groups(), terminal_ids.len());
    let mut remapped = DFA::new(component.num_states());
    remapped.ensure_group_capacity(total_groups);

    for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
        remapped.set_group_u8set(
            terminal_id as u32,
            *component.group_id_to_u8set(local_group as u32),
        );
    }

    for (state_index, state) in component.states().iter().enumerate() {
        remapped.set_transitions_from_sorted_entries(
            state_index as u32,
            state.transitions.iter().map(|(byte, &target)| (byte, target)).collect(),
        );
        for &target in &state.epsilon_transitions {
            remapped.add_epsilon_transition(state_index as u32, target);
        }

        let mut finalizers = BitSet::new(total_groups);
        let mut futures = BitSet::new(total_groups);
        for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
            if state.finalizers.contains(local_group) {
                finalizers.set(terminal_id);
            }
            if component
                .possible_future_group_ids(state_index as u32)
                .contains(local_group)
            {
                futures.set(terminal_id);
            }
        }
        remapped.overwrite_state_metadata(state_index as u32, finalizers, futures);
    }

    remapped
}

fn compile_terminal_partitions(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    adaptive_partitions: Option<&BTreeSet<u32>>,
    adaptive_component_merge: bool,
) -> DFA {
    assert_eq!(exprs.len(), partitions.len(), "one lexer partition id is required per terminal");
    if let Some(labels) = visible_labels {
        assert_eq!(exprs.len(), labels.len(), "one profile label is required per terminal");
    }
    if exprs.is_empty() {
        return compile_with_plan(build_exclusion_compile_plan(exprs));
    }

    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }
    let only_partition_is_adaptive = grouped
        .keys()
        .next()
        .is_some_and(|partition| adaptive_partitions.is_some_and(|set| set.contains(partition)));
    if grouped.len() == 1 && !only_partition_is_adaptive && !adaptive_component_merge {
        return compile_with_plan(build_exclusion_compile_plan_with_labels(exprs, visible_labels));
    }

    let groups = grouped.into_iter().collect::<Vec<_>>();
    let mut components: Vec<AdaptiveTerminalBatch> = groups
        .into_par_iter()
        .flat_map_iter(|(partition, terminal_ids)| {
            if adaptive_partitions.is_some_and(|partitions| partitions.contains(&partition))
                && terminal_ids.len() > 1
            {
                compile_adaptive_terminal_partition(exprs, visible_labels, terminal_ids)
                    .into_iter()
                    .map(|(terminal_ids, dfa)| AdaptiveTerminalBatch { terminal_ids, dfa })
                    .collect::<Vec<_>>()
            } else {
                let component = compile_terminal_ids(exprs, visible_labels, &terminal_ids);
                vec![AdaptiveTerminalBatch {
                    terminal_ids,
                    dfa: component,
                }]
            }
        })
        .collect();

    let max_component_batches = adaptive_lexer_max_component_batches();
    if adaptive_component_merge
        && components.len() > 1
        && components.len() <= max_component_batches
    {
        components = coalesce_adaptive_batches(
            components,
            adaptive_lexer_max_trials_per_component(),
            adaptive_lexer_component_state_limit(),
            "compiled_components",
        );
    } else if adaptive_component_merge
        && components.len() > max_component_batches
        && (std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
            || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some())
    {
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_merge level=compiled_components skipped=true input_batches={} max_component_batches={}",
            components.len(),
            max_component_batches,
        );
    }

    if let [component] = components.as_slice() {
        return remap_component_groups(&component.dfa, &component.terminal_ids, exprs.len());
    }

    let total_states = 1usize
        + components
            .iter()
            .map(|component| component.dfa.num_states())
            .sum::<usize>();
    let mut combined = DFA::new(total_states);
    combined.ensure_group_capacity(exprs.len());

    let mut offset = 1u32;
    for batch in &components {
        let terminal_ids = &batch.terminal_ids;
        let component = &batch.dfa;
        debug_assert_eq!(component.num_groups(), terminal_ids.len());
        combined.add_epsilon_transition(0, offset);

        for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
            combined.set_group_u8set(
                terminal_id as u32,
                *component.group_id_to_u8set(local_group as u32),
            );
        }

        for (state_index, state) in component.states().iter().enumerate() {
            let mapped_state = offset + state_index as u32;
            let transitions = state
                .transitions
                .iter()
                .map(|(byte, &target)| (byte, offset + target))
                .collect();
            combined.set_transitions_from_sorted_entries(mapped_state, transitions);
            for &target in &state.epsilon_transitions {
                combined.add_epsilon_transition(mapped_state, offset + target);
            }

            let mut finalizers = BitSet::new(exprs.len());
            let mut futures = BitSet::new(exprs.len());
            for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
                if state.finalizers.contains(local_group) {
                    finalizers.set(terminal_id);
                }
                if component
                    .possible_future_group_ids(state_index as u32)
                    .contains(local_group)
                {
                    futures.set(terminal_id);
                }
            }
            combined.overwrite_state_metadata(mapped_state, finalizers, futures);
        }
        offset += component.num_states() as u32;
    }

    combined.recompute_possible_futures();
    combined
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

#[derive(Clone)]
enum ProductComponent {
    Materialized(Arc<DFA>),
    VirtualBoundedRepeat {
        base_dfa: Arc<DFA>,
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
                    base_dfa: Arc::new(base_dfa),
                    min: *min as u32,
                    max: *max as u32,
                };
            }

            ProductComponent::Materialized(Arc::new(compile_product_component_dfa(expr)))
        }
        _ => ProductComponent::Materialized(Arc::new(compile_product_component_dfa(expr))),
    }
}

fn compile_product_components(exprs: &[Expr]) -> (Vec<ProductComponent>, usize) {
    let mut unique_exprs = Vec::<&Expr>::new();
    let mut component_indices = Vec::with_capacity(exprs.len());
    let mut index_by_expr = FxHashMap::<&Expr, usize>::default();

    for expr in exprs {
        let expr = unwrap_shared(expr);
        let index = if let Some(&index) = index_by_expr.get(expr) {
            index
        } else {
            let index = unique_exprs.len();
            unique_exprs.push(expr);
            index_by_expr.insert(expr, index);
            index
        };
        component_indices.push(index);
    }

    let unique_components: Vec<ProductComponent> = unique_exprs
        .par_iter()
        .map(|expr| compile_product_component(expr))
        .collect();
    let cache_hits = exprs.len() - unique_components.len();
    let components = component_indices
        .into_iter()
        .map(|index| unique_components[index].clone())
        .collect();
    (components, cache_hits)
}

fn build_product_dfa(exprs: &[Expr], profile_labels: Option<&[ProductComponentProfileLabel]>) -> DFA {
    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_started_at = Instant::now();
    let component_compile_started_at = Instant::now();
    let (components, component_cache_hits): (Vec<ProductComponent>, usize) = if profile_detail {
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
        (components, 0)
    } else {
        compile_product_components(exprs)
    };
    let component_compile_ms = profile_timing
        .then(|| component_compile_started_at.elapsed().as_secs_f64() * 1000.0);
    if profile_timing && !profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] product_component_cache groups={} unique_components={} cache_hits={}",
            exprs.len(),
            exprs.len() - component_cache_hits,
            component_cache_hits,
        );
    }
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
    let equivalence_classes_started_at = Instant::now();
    let (class_map, class_members) = compute_product_equivalence_classes(&components);
    let equivalence_classes_ms = profile_timing
        .then(|| equivalence_classes_started_at.elapsed().as_secs_f64() * 1000.0);
    let num_classes = class_members.len();
    let class_transition_started_at = Instant::now();
    let component_class_transitions = build_product_class_transitions(&components, &class_map);
    let class_transition_ms = profile_timing
        .then(|| class_transition_started_at.elapsed().as_secs_f64() * 1000.0);
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

    let product_state_expand_started_at = Instant::now();
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
    let product_state_expand_ms = profile_timing
        .then(|| product_state_expand_started_at.elapsed().as_secs_f64() * 1000.0);

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

    let byte_expand_started_at = Instant::now();
    let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> = pending_class_transitions
        .into_par_iter()
        .map(|class_transitions| {
            let byte_capacity: usize = class_transitions
                .iter()
                .map(|(class_id, _)| class_members[*class_id as usize].len())
                .sum();
            const DENSE_BYTE_EXPANSION_THRESHOLD: usize = 96;

            let transitions = if byte_capacity >= DENSE_BYTE_EXPANSION_THRESHOLD {
                // Byte-equivalence classes are disjoint but need not be
                // contiguous, so expanding classes in class-ID order does
                // not generally produce byte-sorted output.  For dense rows,
                // scatter targets into the fixed byte alphabet and scan it
                // once. This avoids a large per-state comparison sort.
                let mut target_by_byte = [u32::MAX; 256];
                for (class_id, target) in class_transitions {
                    for &byte in &class_members[class_id as usize] {
                        target_by_byte[byte as usize] = target;
                    }
                }
                target_by_byte
                    .into_iter()
                    .enumerate()
                    .filter_map(|(byte, target)| {
                        (target != u32::MAX).then_some((byte as u8, target))
                    })
                    .collect()
            } else {
                let mut transitions = Vec::with_capacity(byte_capacity);
                for (class_id, target) in class_transitions {
                    for &byte in &class_members[class_id as usize] {
                        transitions.push((byte, target));
                    }
                }
                if transitions.len() > 1 {
                    transitions.sort_unstable_by_key(|entry| entry.0);
                }
                transitions
            };
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
        })
        .collect();
    let byte_expand_ms = profile_timing
        .then(|| byte_expand_started_at.elapsed().as_secs_f64() * 1000.0);

    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }

    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] product_phases groups={} classes={} component_compile_ms={:.3} equivalence_classes_ms={:.3} class_transition_ms={:.3} product_state_expand_ms={:.3} byte_expand_ms={:.3}",
            components.len(),
            num_classes,
            component_compile_ms.unwrap_or_default(),
            equivalence_classes_ms.unwrap_or_default(),
            class_transition_ms.unwrap_or_default(),
            product_state_expand_ms.unwrap_or_default(),
            byte_expand_ms.unwrap_or_default(),
        );
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
fn build_regex_nfa(exprs: &[Expr]) -> NFA {
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
    use super::super::Lexer;
    use super::{
        build_regex,
        build_regex_monolithic,
        build_regex_partitioned,
        build_regex_partitioned_adaptive,
        try_product_union_dfas,
    };
    use super::compile_product_component_dfa_direct;
    use super::factor_regex_expr;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::regex::parse_regex;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::ds::u8set::U8Set;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
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

    fn enumerate_inputs(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        fn extend(out: &mut Vec<Vec<u8>>, prefix: &mut Vec<u8>, alphabet: &[u8], max_len: usize) {
            out.push(prefix.clone());
            if prefix.len() == max_len {
                return;
            }
            for &byte in alphabet {
                prefix.push(byte);
                extend(out, prefix, alphabet, max_len);
                prefix.pop();
            }
        }

        let mut out = Vec::new();
        extend(&mut out, &mut Vec::new(), alphabet, max_len);
        out
    }

    fn tokenizer_observation(
        tokenizer: &Tokenizer,
        input: &[u8],
    ) -> (Vec<(u32, usize)>, Vec<u32>, bool) {
        let execution = tokenizer.execute_from_state_all_widths(input, tokenizer.initial_state());
        let mut matches = execution
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        matches.dedup();

        let mut futures = execution
            .end_state
            .iter()
            .flat_map(|&state| tokenizer.possible_future_terminals_iter(state))
            .collect::<Vec<_>>();
        futures.sort_unstable();
        futures.dedup();
        (matches, futures, execution.end_state.is_empty())
    }

    fn execute_state_set_observation(
        tokenizer: &Tokenizer,
        roots: &[u32],
        input: &[u8],
    ) -> (Vec<(u32, usize)>, Vec<u32>) {
        let mut matches = Vec::new();
        let mut end_states = Vec::new();
        for &root in roots {
            let execution = tokenizer.execute_from_state_all_widths(input, root);
            matches.extend(
                execution
                    .matches
                    .into_iter()
                    .map(|matched| (matched.id, matched.width)),
            );
            end_states.extend(execution.end_state);
        }
        matches.sort_unstable();
        matches.dedup();
        end_states.sort_unstable();
        end_states.dedup();

        let mut futures = end_states
            .iter()
            .flat_map(|&state| tokenizer.possible_future_terminals_iter(state))
            .collect::<Vec<_>>();
        futures.sort_unstable();
        futures.dedup();
        (matches, futures)
    }

    fn random_small_expr(rng: &mut StdRng, depth: usize) -> Expr {
        let atom = |rng: &mut StdRng| match rng.gen_range(0..4) {
            0 => Expr::U8Seq(vec![b'a' + rng.gen_range(0..3)]),
            1 => {
                let len = rng.gen_range(1..=3);
                Expr::U8Seq(
                    (0..len)
                        .map(|_| b'a' + rng.gen_range(0..3))
                        .collect(),
                )
            }
            2 => {
                let mut bytes = Vec::new();
                for byte in b'a'..=b'c' {
                    if rng.gen_bool(0.5) {
                        bytes.push(byte);
                    }
                }
                if bytes.is_empty() {
                    bytes.push(b'a' + rng.gen_range(0..3));
                }
                Expr::U8Class(U8Set::from_bytes(&bytes))
            }
            _ => Expr::Epsilon,
        };

        if depth == 0 {
            return atom(rng);
        }
        match rng.gen_range(0..9) {
            0..=2 => atom(rng),
            3 => Expr::Choice(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            4 => Expr::Seq(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            5 => Expr::Repeat {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                min: rng.gen_range(0..=1),
                max: Some(rng.gen_range(1..=3)),
            },
            6 => Expr::Repeat {
                expr: Box::new(atom(rng)),
                min: rng.gen_range(0..=1),
                max: None,
            },
            7 => Expr::Exclude {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                exclude: Box::new(random_small_expr(rng, depth - 1)),
            },
            _ => Expr::Intersect {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                intersect: Box::new(random_small_expr(rng, depth - 1)),
            },
        }
    }

    #[test]
    fn partitioned_lexer_matches_monolithic_semantics_exhaustively() {
        let shared_tail = Arc::new(Expr::Choice(vec![
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ]));
        let exprs = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"ac".to_vec()),
            ]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                    min: 0,
                    max: Some(2),
                },
                Expr::Shared(Arc::clone(&shared_tail)),
            ]),
            Expr::Exclude {
                expr: Box::new(byte_choice(b"abc")),
                exclude: Box::new(byte_expr(b'b')),
            },
            Expr::Intersect {
                expr: Box::new(byte_choice(b"ab")),
                intersect: Box::new(byte_choice(b"bc")),
            },
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Shared(shared_tail),
            ]),
        ];
        let monolithic = build_regex_monolithic(&exprs).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.clone().into_boxed_slice())),
        );
        let partitionings = [
            (0..exprs.len() as u32).collect::<Vec<_>>(),
            vec![0, 0, 1, 1, 2, 2, 1],
        ];
        let inputs = enumerate_inputs(b"abc ", 5);

        for partitions in partitionings {
            let partitioned = build_regex_partitioned(&exprs, &partitions).into_tokenizer(
                exprs.len() as u32,
                Some(Arc::from(exprs.clone().into_boxed_slice())),
            );
            for input in &inputs {
                assert_eq!(
                    tokenizer_observation(&partitioned, input),
                    tokenizer_observation(&monolithic, input),
                    "partitioned lexer differed for partitions={partitions:?} input={input:?}",
                );
            }
        }
    }

    #[test]
    fn seeded_partitioned_lexer_differential_fuzz() {
        let mut rng = StdRng::seed_from_u64(0xE051_10FA_2026_0710);
        let prefixes = enumerate_inputs(b"abc", 3);
        let suffixes = enumerate_inputs(b"abc", 3);

        for case in 0..48 {
            let expr_count = rng.gen_range(2..=6);
            let exprs = (0..expr_count)
                .map(|_| random_small_expr(&mut rng, 2))
                .collect::<Vec<_>>();
            let monolithic = build_regex_monolithic(&exprs).into_tokenizer(
                exprs.len() as u32,
                Some(Arc::from(exprs.clone().into_boxed_slice())),
            );

            for partition_case in 0..3 {
                let partitions = match partition_case {
                    0 => (0..expr_count as u32).collect::<Vec<_>>(),
                    1 => (0..expr_count)
                        .map(|_| rng.gen_range(0..3))
                        .collect::<Vec<_>>(),
                    _ => vec![0; expr_count],
                };
                let partitioned = build_regex_partitioned(&exprs, &partitions).into_tokenizer(
                    exprs.len() as u32,
                    Some(Arc::from(exprs.clone().into_boxed_slice())),
                );
                let adaptive_partitions = partitions.iter().copied().collect();
                let adaptive = build_regex_partitioned_adaptive(
                    &exprs,
                    &partitions,
                    &adaptive_partitions,
                )
                .into_tokenizer(
                    exprs.len() as u32,
                    Some(Arc::from(exprs.clone().into_boxed_slice())),
                );

                for prefix in &prefixes {
                    assert_eq!(
                        tokenizer_observation(&partitioned, prefix),
                        tokenizer_observation(&monolithic, prefix),
                        "top-level mismatch case={case} partitions={partitions:?} prefix={prefix:?} exprs={exprs:?}",
                    );
                    assert_eq!(
                        tokenizer_observation(&adaptive, prefix),
                        tokenizer_observation(&monolithic, prefix),
                        "adaptive top-level mismatch case={case} partitions={partitions:?} prefix={prefix:?} exprs={exprs:?}",
                    );

                    let partitioned_roots = partitioned.execute_from_state_end_only(
                        prefix,
                        partitioned.initial_state(),
                    );
                    let adaptive_roots = adaptive.execute_from_state_end_only(
                        prefix,
                        adaptive.initial_state(),
                    );
                    let monolithic_roots = monolithic.execute_from_state_end_only(
                        prefix,
                        monolithic.initial_state(),
                    );
                    for suffix in &suffixes {
                        assert_eq!(
                            execute_state_set_observation(
                                &partitioned,
                                &partitioned_roots,
                                suffix,
                            ),
                            execute_state_set_observation(
                                &monolithic,
                                &monolithic_roots,
                                suffix,
                            ),
                            "residual mismatch case={case} partitions={partitions:?} prefix={prefix:?} suffix={suffix:?} exprs={exprs:?}",
                        );
                        assert_eq!(
                            execute_state_set_observation(
                                &adaptive,
                                &adaptive_roots,
                                suffix,
                            ),
                            execute_state_set_observation(
                                &monolithic,
                                &monolithic_roots,
                                suffix,
                            ),
                            "adaptive residual mismatch case={case} partitions={partitions:?} prefix={prefix:?} suffix={suffix:?} exprs={exprs:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nested_group_op_materialization_reuses_structurally_equal_dfas() {
        let nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
            exclude: Box::new(Expr::U8Seq(vec![b'b'])),
        };
        let mut cache = super::NestedGroupOpCache::default();
        let first = super::materialize_nested_group_ops(nested.clone(), &mut cache);
        let second = super::materialize_nested_group_ops(nested, &mut cache);

        assert_eq!(cache.cache_misses, 1);
        assert_eq!(cache.cache_hits, 1);
        match (first, second) {
            (Expr::Dfa(first), Expr::Dfa(second)) => assert!(Arc::ptr_eq(&first, &second)),
            _ => panic!("nested group operation was not materialized to a DFA"),
        }
    }

    #[test]
    fn duplicate_product_components_share_the_same_dfa() {
        let expr = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
                min: 0,
                max: None,
            },
        ]);
        let shared_expr = Expr::Shared(Arc::new(expr.clone()));
        let (components, cache_hits) = super::compile_product_components(&[expr, shared_expr]);

        assert_eq!(cache_hits, 1);
        match (&components[0], &components[1]) {
            (
                super::ProductComponent::Materialized(first),
                super::ProductComponent::Materialized(second),
            ) => assert!(Arc::ptr_eq(first, second)),
            _ => panic!("expected materialized product components"),
        }
    }

    #[test]
    fn factors_contains_literal_choice_with_common_json_string_shell() {
        let char = Expr::U8Class(U8Set::from_bytes(br#"ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"#));
        let mk = |s: &[u8]| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(char.clone()),
                    min: 0,
                    max: None,
                },
                Expr::U8Seq(s.to_vec()),
                Expr::Repeat {
                    expr: Box::new(char.clone()),
                    min: 0,
                    max: None,
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let expr = Expr::Seq(vec![
            Expr::U8Seq(b"\"interval\": ".to_vec()),
            Expr::Choice(vec![
                mk(b"INTERVAL_TICK"),
                mk(b"INTERVAL_M1"),
                mk(b"INTERVAL_M2"),
                mk(b"INTERVAL_M3"),
                mk(b"INTERVAL_M4"),
                mk(b"INTERVAL_M5"),
                mk(b"INTERVAL_M6"),
                mk(b"INTERVAL_M10"),
                mk(b"INTERVAL_M15"),
                mk(b"INTERVAL_M20"),
                mk(b"INTERVAL_M30"),
                mk(b"INTERVAL_H1"),
                mk(b"INTERVAL_H2"),
                mk(b"INTERVAL_H4"),
                mk(b"INTERVAL_D1"),
                mk(b"INTERVAL_W1"),
                mk(b"INTERVAL_MN1"),
            ]),
        ]);
        let factored = factor_regex_expr(expr);
        let regex = build_regex(&[factored]);
        assert!(
            regex.num_states() < 500,
            "factored regex should not construct a huge terminal DFA; states={}",
            regex.num_states(),
        );
        let accept = |bytes: &[u8]| {
            let mut state = 0;
            for &b in bytes {
                let Some(next) = regex.step(state, b) else {
                    return false;
                };
                state = next;
            }
            !regex.dfa.finalizers(state).is_empty()
        };
        assert!(accept(br#""interval": "INTERVAL_M1""#));
        assert!(accept(br#""interval": "XXXINTERVAL_M1YYY""#));
        assert!(accept(br#""interval": "INTERVAL_TICK""#));
        assert!(!accept(br#""interval": "NOPE""#));
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

    #[test]
    fn bounded_repeat_regex_suffix_must_fork_at_ambiguous_boundary() {
        // ("a"+)? "a" matches "aa":
        //
        //   optional body "a"+ consumes the first "a"
        //   suffix "a" consumes the second "a"
        //
        // The regex-suffix fast path used to greedily continue the body on the
        // second "a" and drop the valid suffix path.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(vec![b'a'])),
                    min: 1,
                    max: None,
                }),
                min: 0,
                max: Some(1),
            },
            Expr::U8Seq(vec![b'a']),
        ]);
        assert!(terminal_matches(expr, b"aa"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_nullable_class_suffix_finalizes_after_body() {
        // "a" [b]? matches both "a" and "ab".
        //
        // The regex-suffix fast path used to miss the body/suffix boundary at
        // end-of-input when the suffix was nullable.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 1,
                max: Some(1),
            },
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::U8Class(U8Set::single(b'b')),
            ]),
        ]);
        assert!(terminal_matches(expr.clone(), b"a"));
        assert!(terminal_matches(expr, b"ab"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_nullable_suffix_after_optional_body() {
        // ("a")? [b]? matches "a", "b", and "ab".
        //
        // Do not assert the empty string here: zero-length terminals are not a
        // useful lexer regression target and may be intentionally unsupported by
        // terminal_matches / terminal DFA metadata. This test is about preserving
        // non-empty matches when both the repeated body and regex suffix are
        // nullable.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: Some(1),
            },
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::U8Class(U8Set::single(b'b')),
            ]),
        ]);
        assert!(terminal_matches(expr.clone(), b"a"));
        assert!(terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"ab"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_zero_max_must_not_consume_body() {
        // "a"{0,0} [b] is just [b]. It must not match "ab".
        //
        // The regex-suffix fast path used to start with a live body state even when
        // max == 0.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: Some(0),
            },
            Expr::U8Class(U8Set::single(b'b')),
        ]);
        assert!(terminal_matches(expr.clone(), b"b"));
        assert!(!terminal_matches(expr, b"ab"));
    }

    #[test]
    fn build_regex_defaults_to_one_monolithic_dfa() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        assert!(!tokenizer.has_epsilon_transitions());
    }

    #[test]
    fn separate_terminal_partitions_preserve_multiple_live_end_states() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let tokenizer = build_regex_partitioned(&expressions, &[0, 1]).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        assert!(tokenizer.has_epsilon_transitions());
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert_eq!(result.end_state.len(), 2, "both terminal components must remain live");
        assert_eq!(
            result.matches.iter().map(|matched| matched.id).collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1]),
        );

        let continued = tokenizer.execute_from_state(b"a", result.end_state[1]);
        assert!(continued.matches.iter().any(|matched| matched.id == 1));
    }

    #[test]
    fn explicit_partition_ids_control_joint_determinization() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"z".to_vec()),
        ];
        let regex = super::build_regex_partitioned(&expressions, &[7, 7, 9]);
        assert_eq!(
            regex.dfa.states()[0].epsilon_transitions.len(),
            2,
            "two declared partitions must produce two epsilon branches",
        );
        let tokenizer = regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert!(result.matches.iter().any(|matched| matched.id == 0));
        assert!(!result.end_state.is_empty());
    }

    #[test]
    fn bounded_product_trial_stops_before_cross_pattern_blowup() {
        let left = build_regex_monolithic(&[parse_regex(r"\w+_(\w_)?\d+", false)]).dfa;
        let right = build_regex_monolithic(&[parse_regex(r"(\w|-){12}", false)]).dfa;
        assert!(
            try_product_union_dfas(&left, &right, 32).is_none(),
            "the bounded trial unexpectedly completed within 32 product states",
        );
    }

    #[test]
    fn adaptive_partitioning_preserves_singleton_component_semantics() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let singleton = build_regex_partitioned(&expressions, &[0, 1, 2]).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        let adaptive = build_regex_partitioned_adaptive(
            &expressions,
            &[7, 7, 7],
            &std::collections::BTreeSet::from([7]),
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b"ab", 4) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&singleton, &input),
                "adaptive partitioning differed for input {input:?}",
            );
        }
    }

    #[test]
    fn epsilon_partitioned_tokenizer_round_trips_through_serde() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = build_regex_partitioned(&expressions, &[0, 1]).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let encoded = bincode::serialize(&tokenizer).unwrap();
        let decoded: Tokenizer = bincode::deserialize(&encoded).unwrap();
        assert!(decoded.has_epsilon_transitions());
        assert_eq!(
            tokenizer.execute_from_state(b"a", tokenizer.initial_state()),
            decoded.execute_from_state(b"a", decoded.initial_state()),
        );
    }

}
