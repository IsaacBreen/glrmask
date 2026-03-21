#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;
use crate::compiler::compile::compile_profile_enabled;

use super::ast::Expr;
use super::dfa::DFA;
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
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
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

fn compile_repeat_power_cps(
    expr: &Expr,
    copies: usize,
    nfa: &mut NFA,
    end: u32,
    cache: &mut HashMap<(usize, u32), u32>,
) -> u32 {
    debug_assert!(copies.is_power_of_two());

    if let Some(&start) = cache.get(&(copies, end)) {
        return start;
    }

    let start = if copies == 1 {
        let start = nfa.add_state();
        compile_expr(expr, nfa, start, end);
        start
    } else {
        let half = copies / 2;
        let suffix_start = compile_repeat_power_cps(expr, half, nfa, end, cache);
        compile_repeat_power_cps(expr, half, nfa, suffix_start, cache)
    };

    cache.insert((copies, end), start);
    start
}

fn compile_repeat_exact_cps(
    expr: &Expr,
    copies: usize,
    nfa: &mut NFA,
    end: u32,
    power_cache: &mut HashMap<(usize, u32), u32>,
) -> u32 {
    if copies == 0 {
        return end;
    }

    let largest_power = highest_power_of_two_leq(copies);
    let suffix_start = compile_repeat_exact_cps(expr, copies - largest_power, nfa, end, power_cache);
    compile_repeat_power_cps(expr, largest_power, nfa, suffix_start, power_cache)
}

fn compile_repeat_upto_cps(
    expr: &Expr,
    copies: usize,
    nfa: &mut NFA,
    end: u32,
    power_cache: &mut HashMap<(usize, u32), u32>,
    upto_cache: &mut HashMap<(usize, u32), u32>,
) -> u32 {
    if copies == 0 {
        return end;
    }

    if let Some(&start) = upto_cache.get(&(copies, end)) {
        return start;
    }

    let largest_power = highest_power_of_two_leq(copies);
    let split = nfa.add_state();

    let smaller_start = compile_repeat_upto_cps(
        expr,
        largest_power - 1,
        nfa,
        end,
        power_cache,
        upto_cache,
    );
    nfa.add_epsilon(split, smaller_start);

    let suffix_start = compile_repeat_upto_cps(
        expr,
        copies - largest_power,
        nfa,
        end,
        power_cache,
        upto_cache,
    );
    let power_start = compile_repeat_power_cps(expr, largest_power, nfa, suffix_start, power_cache);
    nfa.add_epsilon(split, power_start);

    upto_cache.insert((copies, end), split);
    split
}

fn compile_expr(expr: &Expr, nfa: &mut NFA, start: u32, end: u32) {
    match expr {
        Expr::U8Seq(bytes) => {
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
        Expr::U8Class(set) => {
            nfa.add_u8set_transition(start, *set, end);
        }
        Expr::Seq(parts) => {
            let mut state = start;
            for (index, part) in parts.iter().enumerate() {
                let next = if index + 1 == parts.len() {
                    end
                } else {
                    nfa.add_state()
                };
                compile_expr(part, nfa, state, next);
                state = next;
            }
            if parts.is_empty() {
                nfa.add_epsilon(start, end);
            }
        }
        Expr::Choice(options) => {
            if options.is_empty() {
                nfa.add_epsilon(start, end);
            }
            for option in options {
                compile_expr(option, nfa, start, end);
            }
        }
        Expr::Exclude { .. } => {
            unreachable!("nested Expr::Exclude must be lowered before NFA compilation")
        }
        Expr::Repeat { expr, min, max } => {
            match max {
                Some(max) => {
                    if *max < *min {
                        return;
                    }

                    let optional = max - min;
                    let mut power_cache = HashMap::new();
                    let mut upto_cache = HashMap::new();
                    let tail_start = compile_repeat_upto_cps(
                        expr,
                        optional,
                        nfa,
                        end,
                        &mut power_cache,
                        &mut upto_cache,
                    );
                    let repeat_start =
                        compile_repeat_exact_cps(expr, *min, nfa, tail_start, &mut power_cache);
                    nfa.add_epsilon(start, repeat_start);
                }
                None => {
                    let mut current = start;
                    for _ in 0..*min {
                        let next = nfa.add_state();
                        compile_expr(expr, nfa, current, next);
                        current = next;
                    }

                    // When min=0, current is still the shared `start` state.
                    // The loop-back edge (loop_state → current) must NOT point
                    // at state 0 (the NFA initial state), because that would
                    // make every terminal reachable from inside the loop,
                    // polluting `possible_future_group_ids`.  Insert a fresh
                    // intermediate so the loop is self-contained.
                    if current == start {
                        let fresh = nfa.add_state();
                        nfa.add_epsilon(start, fresh);
                        current = fresh;
                    }

                    nfa.add_epsilon(current, end);
                    let loop_state = nfa.add_state();
                    compile_expr(expr, nfa, current, loop_state);
                    nfa.add_epsilon(loop_state, current);
                    if expr_accepts_empty(expr) {
                        nfa.add_epsilon(loop_state, end);
                    }
                }
            }
        }
        Expr::Shared(inner) => compile_expr(inner, nfa, start, end),
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
    let plan = build_exclusion_compile_plan(exprs);
    let profile_enabled = compile_profile_enabled();

    let phase_started_at = std::time::Instant::now();
    let group_sets: Vec<U8Set> = plan
        .compiled_exprs
        .iter()
        .map(|expr| expr_u8set(expr))
        .collect();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][tokenizer] compute_group_sets_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    let phase_started_at = std::time::Instant::now();
    let mut nfa = build_regex_nfa(&plan.compiled_exprs);
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][tokenizer] build_regex_nfa_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    let phase_started_at = std::time::Instant::now();
    nfa.condense_epsilon_sccs();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][tokenizer] condense_epsilon_sccs_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    let phase_started_at = std::time::Instant::now();
    let mut dfa = nfa.to_dfa();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][tokenizer] determinize_regex_nfa_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    let phase_started_at = std::time::Instant::now();
    dfa.ensure_group_capacity(group_sets.len());
    for (group_id, set) in group_sets.into_iter().enumerate() {
        dfa.set_group_u8set(group_id as u32, set);
    }
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][tokenizer] assign_group_sets_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    if !plan.exclusions.is_empty() {
        let phase_started_at = std::time::Instant::now();
        let _ = dfa.apply_group_exclusions(&plan.exclusions);
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][tokenizer] apply_exclusions_ms={:.3}",
                phase_started_at.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    let mut dfa = if plan.visible_groups < plan.compiled_exprs.len() {
        dfa.project_groups(plan.visible_groups)
    } else {
        dfa
    };

    let phase_started_at = std::time::Instant::now();
    let dfa = dfa.minimize();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][tokenizer] minimize_regex_dfa_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }
    Regex { dfa }
}

/// Compile multiple expressions into a single NFA (without determinization).
///
/// Each expression's index becomes its group ID.
pub fn build_regex_nfa(exprs: &[Expr]) -> NFA {
    let optimized_exprs: Vec<Expr> = exprs.iter().cloned().map(Expr::optimize).collect();
    let mut nfa = NFA::new(1);

    if let Some((prefix, remainders)) = common_prefix_factor(&optimized_exprs) {
        let split = nfa.add_state();
        compile_expr(&prefix, &mut nfa, 0, split);

        for (group_id, remainder) in remainders.iter().enumerate() {
            let accept = nfa.add_state();
            compile_expr(remainder, &mut nfa, split, accept);
            nfa.add_finalizer(accept, group_id as u32);
        }
        return nfa;
    }

    for (group_id, expr) in optimized_exprs.iter().enumerate() {
        let accept = nfa.add_state();
        compile_expr(expr, &mut nfa, 0, accept);
        nfa.add_finalizer(accept, group_id as u32);
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
