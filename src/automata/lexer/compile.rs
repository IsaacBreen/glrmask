use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ds::{bitset::BitSet, u8set::U8Set};

use super::ast::Expr;
use super::tokenizer::Tokenizer;
use super::dfa::DFA;
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

/// Expose one intersection nested under a common sequence shell. This is an
/// exact distributive rewrite:
///
///     prefix · (left ∩ right) · suffix
///       = (prefix · left · suffix) ∩ (prefix · right · suffix)
///
/// Keeping the operands visible lets structural full/stencil compilation map
/// their residual coordinates independently and materialize missing correlated
/// product tuples. We deliberately handle exactly one nested intersection to
/// avoid an exponential distribution over multiple group operations.
fn lift_single_nested_intersection(expr: &Expr) -> Option<Expr> {
    let Expr::Seq(parts) = expr else {
        return None;
    };
    let mut intersection = None;
    for (index, part) in parts.iter().enumerate() {
        let part = match part {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        if let Expr::Intersect { expr, intersect } = part {
            if intersection.is_some() {
                return None;
            }
            intersection = Some((index, expr.as_ref(), intersect.as_ref()));
        }
    }
    let (index, left, right) = intersection?;
    let branch = |replacement: &Expr| {
        let mut branch = parts.to_vec();
        branch[index] = replacement.clone();
        Expr::Seq(branch).optimize()
    };
    Some(Expr::Intersect {
        expr: Box::new(branch(left)),
        intersect: Box::new(branch(right)),
    })
}

#[derive(Default)]
struct NestedGroupOpCache {
    compiled: FxHashMap<Expr, Arc<DFA>>,
    shared_duplicates: Option<Arc<SharedDuplicateNestedGroupOpCache>>,
    allow_shared_initialization: bool,
    cache_hits: usize,
    cache_misses: usize,
    compiled_ms: f64,
    max_compile_ms: f64,
}

struct SharedDuplicateNestedGroupOpCache {
    duplicated: FxHashSet<Expr>,
    compiled: Mutex<FxHashMap<Expr, Arc<OnceLock<Arc<DFA>>>>>,
}

impl SharedDuplicateNestedGroupOpCache {
    fn cell_if_duplicated(&self, expr: &Expr) -> Option<Arc<OnceLock<Arc<DFA>>>> {
        if !self.duplicated.contains(expr) {
            return None;
        }
        let mut compiled = self
            .compiled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Some(Arc::clone(
            compiled
                .entry(expr.clone())
                .or_insert_with(|| Arc::new(OnceLock::new())),
        ))
    }

    #[cfg(test)]
    fn all_entries_initialized(&self) -> bool {
        let compiled = self
            .compiled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        compiled.len() == self.duplicated.len()
            && compiled.values().all(|cell| cell.get().is_some())
    }
}

fn materialize_nested_group_ops(expr: Expr, cache: &mut NestedGroupOpCache) -> Expr {
    match expr {
        expr @ (Expr::Exclude { .. } | Expr::Intersect { .. }) => {
            if let Some(compiled) = cache.compiled.get(&expr) {
                cache.cache_hits += 1;
                return Expr::Dfa(compiled.clone());
            }

            if let Some(shared) = cache.shared_duplicates.clone()
                && let Some(cell) = shared.cell_if_duplicated(&expr)
            {
                let started_at = Instant::now();
                let was_ready = cell.get().is_some();
                let compiled = if cache.allow_shared_initialization {
                    Arc::clone(cell.get_or_init(|| {
                        let mut nested_cache = NestedGroupOpCache {
                            shared_duplicates: Some(Arc::clone(&shared)),
                            allow_shared_initialization: true,
                            ..NestedGroupOpCache::default()
                        };
                        Arc::new(compile_with_plan(
                            build_exclusion_compile_plan_with_labels_and_cache(
                                std::slice::from_ref(&expr),
                                None,
                                &mut nested_cache,
                            ),
                        ))
                    }))
                } else {
                    Arc::clone(cell.get().expect(
                        "shared nested group-op cache must be prewarmed before parallel compilation",
                    ))
                };
                if was_ready {
                    cache.cache_hits += 1;
                } else {
                    cache.cache_misses += 1;
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    cache.compiled_ms += elapsed_ms;
                    cache.max_compile_ms = cache.max_compile_ms.max(elapsed_ms);
                }
                cache.compiled.insert(expr, Arc::clone(&compiled));
                return Expr::Dfa(compiled);
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

fn count_nested_group_ops(expr: &Expr, counts: &mut FxHashMap<Expr, usize>) {
    match expr {
        Expr::Exclude { expr: inner, exclude } => {
            *counts.entry(expr.clone()).or_default() += 1;
            count_nested_group_ops(inner, counts);
            count_nested_group_ops(exclude, counts);
        }
        Expr::Intersect { expr: inner, intersect } => {
            *counts.entry(expr.clone()).or_default() += 1;
            count_nested_group_ops(inner, counts);
            count_nested_group_ops(intersect, counts);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            for part in parts {
                count_nested_group_ops(part, counts);
            }
        }
        Expr::Repeat { expr, .. } => count_nested_group_ops(expr, counts),
        Expr::Shared(expr) => count_nested_group_ops(expr, counts),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
    }
}

fn count_all_subexpressions(expr: &Expr, counts: &mut FxHashMap<Expr, usize>) {
    *counts.entry(expr.clone()).or_default() += 1;
    match expr {
        Expr::Exclude { expr, exclude } => {
            count_all_subexpressions(expr, counts);
            count_all_subexpressions(exclude, counts);
        }
        Expr::Intersect { expr, intersect } => {
            count_all_subexpressions(expr, counts);
            count_all_subexpressions(intersect, counts);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            for part in parts {
                count_all_subexpressions(part, counts);
            }
        }
        Expr::Repeat { expr, .. } => {
            count_all_subexpressions(expr, counts);
        }
        Expr::Shared(expr) => {
            count_all_subexpressions(expr, counts);
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
    }
}

fn collect_maximal_repeated_subexpressions(
    expr: &Expr,
    candidates: &FxHashSet<Expr>,
    selected: &mut FxHashSet<Expr>,
) {
    if candidates.contains(expr) {
        selected.insert(expr.clone());
        return;
    }
    match expr {
        Expr::Exclude { expr, exclude } => {
            collect_maximal_repeated_subexpressions(expr, candidates, selected);
            collect_maximal_repeated_subexpressions(exclude, candidates, selected);
        }
        Expr::Intersect { expr, intersect } => {
            collect_maximal_repeated_subexpressions(expr, candidates, selected);
            collect_maximal_repeated_subexpressions(intersect, candidates, selected);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            for part in parts {
                collect_maximal_repeated_subexpressions(part, candidates, selected);
            }
        }
        Expr::Repeat { expr, .. } => {
            collect_maximal_repeated_subexpressions(expr, candidates, selected);
        }
        Expr::Shared(expr) => {
            collect_maximal_repeated_subexpressions(expr, candidates, selected);
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
    }
}

fn replace_compiled_subexpressions(
    expr: Expr,
    compiled: &FxHashMap<Expr, Arc<DFA>>,
    replace_root: bool,
) -> Expr {
    if replace_root && let Some(dfa) = compiled.get(&expr) {
        return Expr::Dfa(Arc::clone(dfa));
    }
    match expr {
        Expr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            exclude: Box::new(replace_compiled_subexpressions(*exclude, compiled, true)),
        },
        Expr::Intersect { expr, intersect } => Expr::Intersect {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            intersect: Box::new(replace_compiled_subexpressions(*intersect, compiled, true)),
        },
        Expr::Seq(parts) => Expr::Seq(
            parts
                .into_iter()
                .map(|part| replace_compiled_subexpressions(part, compiled, true))
                .collect(),
        ),
        Expr::Choice(parts) => Expr::Choice(
            parts
                .into_iter()
                .map(|part| replace_compiled_subexpressions(part, compiled, true))
                .collect(),
        ),
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            min,
            max,
        },
        Expr::Shared(inner) => {
            let rewritten = replace_compiled_subexpressions((*inner).clone(), compiled, true);
            if rewritten == *inner {
                Expr::Shared(inner)
            } else {
                rewritten
            }
        }
        leaf @ (Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon) => leaf,
    }
}

fn materialize_repeated_subexpression_dfas_with_limits(
    exprs: &[Expr],
    min_size: usize,
    min_occurrences: usize,
) -> Option<Vec<Expr>> {
    debug_assert!(min_size > 1);
    debug_assert!(min_occurrences > 1);
    let started_at = Instant::now();
    let mut counts = FxHashMap::<Expr, usize>::default();
    for expr in exprs {
        count_all_subexpressions(expr, &mut counts);
    }
    let occurrences = counts.clone();
    let candidates = counts
        .into_iter()
        .filter_map(|(expr, count)| {
            (count >= min_occurrences
                && expr_structural_size(&expr) >= min_size
                && !expr_contains_group_op(&expr)
                && !matches!(expr, Expr::Dfa(_)))
            .then_some(expr)
        })
        .collect::<FxHashSet<_>>();
    let mut selected = FxHashSet::default();
    for expr in exprs {
        collect_maximal_repeated_subexpressions(expr, &candidates, &mut selected);
    }
    if selected.is_empty() {
        return None;
    }

    let selected = selected.into_iter().collect::<Vec<_>>();
    let compile_started_at = Instant::now();
    let compiled_entries = selected
        .into_par_iter()
        .map(|expr| {
            let expr_size = expr_structural_size(&expr);
            let occurrence_count = occurrences.get(&expr).copied().unwrap_or(0);
            let entry_started_at = Instant::now();
            let dfa = Arc::new(compile_expr_to_dfa(&expr));
            let entry_compile_ms = entry_started_at.elapsed().as_secs_f64() * 1000.0;
            (expr, dfa, expr_size, occurrence_count, entry_compile_ms)
        })
        .collect::<Vec<_>>();
    let compile_ms = compile_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut compiled = FxHashMap::<Expr, Arc<DFA>>::default();
    for (expr, dfa, expr_size, occurrence_count, entry_compile_ms) in compiled_entries {
        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][tokenizer] shared_subexpr_dfa_entry size={} occurrences={} states={} transitions={} compile_ms={:.3}",
                expr_size,
                occurrence_count,
                dfa.num_states(),
                dfa_transition_count(&dfa),
                entry_compile_ms,
            );
        }
        compiled.insert(expr, dfa);
    }
    let rewritten = exprs
        .iter()
        .cloned()
        .map(|expr| replace_compiled_subexpressions(expr, &compiled, true))
        .collect::<Vec<_>>();
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_subexpr_dfas entries={} compile_ms={:.3} total_ms={:.3}",
            compiled.len(),
            compile_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(rewritten)
}

fn materialize_repeated_subexpression_dfas(exprs: &[Expr]) -> Option<Vec<Expr>> {
    if std::env::var_os("GLRMASK_DISABLE_SHARED_SUBEXPR_DFA_CACHE").is_some() {
        return None;
    }
    let forced = std::env::var_os("GLRMASK_FORCE_SHARED_SUBEXPR_DFA_CACHE").is_some();
    let min_total_size = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_TOTAL_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(40_000);
    let total_size = exprs.iter().map(expr_structural_size).sum::<usize>();
    if !forced && total_size < min_total_size {
        return None;
    }
    let min_size = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(128);
    let min_occurrences = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_OCCURRENCES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(8);
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_subexpr_scan forced={} total_size={} min_total_size={} min_size={} min_occurrences={}",
            forced, total_size, min_total_size, min_size, min_occurrences,
        );
    }
    materialize_repeated_subexpression_dfas_with_limits(exprs, min_size, min_occurrences)
}

fn shared_duplicate_nested_group_op_cache(
    exprs: &[Expr],
    grouped: &BTreeMap<u32, Vec<usize>>,
) -> Option<Arc<SharedDuplicateNestedGroupOpCache>> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let singleton_partitions = grouped
        .values()
        .filter(|terminal_ids| terminal_ids.len() == 1)
        .count();
    if grouped.len() < 64 || singleton_partitions * 4 < grouped.len() * 3 {
        return None;
    }

    let mut counts = FxHashMap::<Expr, usize>::default();
    for terminal_ids in grouped.values() {
        for &terminal_id in terminal_ids {
            let (base, excluded, intersections) = split_top_level_group_ops(&exprs[terminal_id]);
            count_nested_group_ops(&base, &mut counts);
            for expr in &excluded {
                count_nested_group_ops(expr, &mut counts);
            }
            for expr in &intersections {
                count_nested_group_ops(expr, &mut counts);
            }
        }
    }
    let duplicated = counts
        .into_iter()
        .filter_map(|(expr, count)| (count > 1).then_some(expr))
        .collect::<FxHashSet<_>>();
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_duplicate_nested_ops partitions={} singleton_partitions={} duplicated={} scan_ms={:.3}",
            grouped.len(),
            singleton_partitions,
            duplicated.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (!duplicated.is_empty()).then(|| {
        Arc::new(SharedDuplicateNestedGroupOpCache {
            duplicated,
            compiled: Mutex::new(FxHashMap::default()),
        })
    })
}

fn expr_structural_size(expr: &Expr) -> usize {
    match expr {
        Expr::Exclude { expr, exclude } => {
            1 + expr_structural_size(expr) + expr_structural_size(exclude)
        }
        Expr::Intersect { expr, intersect } => {
            1 + expr_structural_size(expr) + expr_structural_size(intersect)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            1 + parts.iter().map(expr_structural_size).sum::<usize>()
        }
        Expr::Repeat { expr, .. } => 1 + expr_structural_size(expr),
        Expr::Shared(expr) => 1 + expr_structural_size(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => 1,
    }
}

/// Populate every cross-partition nested-group-op cache entry before the
/// partition compilation itself fans out over Rayon.
///
/// Lazy `OnceLock` initialization inside the parallel partition loop can
/// deadlock: several workers may block on one cache entry while the worker
/// initializing it launches nested Rayon work that requires those same
/// workers. Compile proper subexpressions first, then their parents, so nested
/// shared entries are already available while a parent is materialized.
fn prewarm_shared_duplicate_nested_group_ops(
    shared: &Arc<SharedDuplicateNestedGroupOpCache>,
) {
    let mut duplicated = shared.duplicated.iter().cloned().collect::<Vec<_>>();
    duplicated.sort_unstable_by_key(expr_structural_size);

    let mut cache = NestedGroupOpCache {
        shared_duplicates: Some(Arc::clone(shared)),
        allow_shared_initialization: true,
        ..NestedGroupOpCache::default()
    };
    for expr in duplicated {
        let _ = materialize_nested_group_ops(expr, &mut cache);
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
    build_bounded_repeat_dfa_from_base(&base_dfa, min, max)
}

fn build_bounded_repeat_dfa_from_base(base_dfa: &DFA, min: usize, max: usize) -> Option<DFA> {

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
#[derive(Clone, PartialEq, Eq, Hash)]
struct ZeroMinRepeatSuffixState {
    /// Minimum completed-copy count reaching each body DFA residual. A smaller
    /// count dominates every larger count at the same residual.
    body_min_counts: Box<[u32]>,
    /// Exact live suffix residuals. Count is irrelevant once the suffix starts.
    suffix_states: Box<[u32]>,
}

fn close_zero_min_repeat_suffix_state(
    body_min_counts: &mut [u32],
    suffix_states: &mut Vec<u32>,
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
) {
    let mut completed_boundary = u32::MAX;
    for (body_state, &completed) in body_min_counts.iter().enumerate() {
        if completed == u32::MAX {
            continue;
        }
        if body_dfa.finalizers(body_state as u32).contains(0) {
            completed_boundary = completed_boundary.min(completed.saturating_add(1));
        }
    }

    if completed_boundary != u32::MAX {
        suffix_states.push(0);
        if completed_boundary < max as u32 {
            body_min_counts[0] = body_min_counts[0].min(completed_boundary);
        }
    }

    // Keep only residuals that can consume at least one more byte toward a
    // body match. Acceptance at the current body state has already spawned the
    // repeat/suffix boundary above.
    for (body_state, completed) in body_min_counts.iter_mut().enumerate() {
        if *completed != u32::MAX
            && !body_dfa
                .possible_future_group_ids(body_state as u32)
                .contains(0)
        {
            *completed = u32::MAX;
        }
    }

    suffix_states.retain(|&state| {
        suffix_dfa.finalizers(state).contains(0)
            || suffix_dfa.possible_future_group_ids(state).contains(0)
    });
    suffix_states.sort_unstable();
    suffix_states.dedup();
}

fn set_zero_min_repeat_suffix_metadata(
    dfa: &mut DFA,
    state_id: u32,
    state: &ZeroMinRepeatSuffixState,
    suffix_dfa: &DFA,
) {
    let mut finalizers = BitSet::new(1);
    if state
        .suffix_states
        .iter()
        .any(|&suffix_state| suffix_dfa.finalizers(suffix_state).contains(0))
    {
        finalizers.set(0);
    }
    let mut future = BitSet::new(1);
    if state.body_min_counts.iter().any(|&count| count != u32::MAX)
        || state
            .suffix_states
            .iter()
            .any(|&suffix_state| {
                suffix_dfa
                    .possible_future_group_ids(suffix_state)
                    .contains(0)
            })
    {
        future.set(0);
    }
    dfa.overwrite_state_metadata(state_id, finalizers, future);
}

fn build_zero_min_repeat_suffix_dominance_dfa(
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
) -> Option<DFA> {
    if max == 0
        || body_dfa.num_states() == 0
        || suffix_dfa.num_states() == 0
        || body_dfa.finalizers(0).contains(0)
        || suffix_dfa.finalizers(0).contains(0)
    {
        return None;
    }

    let mut start_body = vec![u32::MAX; body_dfa.num_states()];
    start_body[0] = 0;
    let mut start_suffix = vec![0u32];
    close_zero_min_repeat_suffix_state(
        &mut start_body,
        &mut start_suffix,
        body_dfa,
        suffix_dfa,
        max,
    );
    let start = ZeroMinRepeatSuffixState {
        body_min_counts: start_body.into_boxed_slice(),
        suffix_states: start_suffix.into_boxed_slice(),
    };

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    set_zero_min_repeat_suffix_metadata(&mut dfa, 0, &start, suffix_dfa);
    let mut state_map = FxHashMap::<ZeroMinRepeatSuffixState, u32>::default();
    state_map.insert(start.clone(), 0);
    let mut worklist = VecDeque::from([(0u32, start)]);

    while let Some((state_id, state)) = worklist.pop_front() {
        let mut transitions = Vec::new();
        for byte_value in 0u16..=255 {
            let byte = byte_value as u8;
            let mut next_body = vec![u32::MAX; body_dfa.num_states()];
            for (body_state, &completed) in state.body_min_counts.iter().enumerate() {
                if completed == u32::MAX {
                    continue;
                }
                if let Some(target) = body_dfa.step(body_state as u32, byte) {
                    let target_count = &mut next_body[target as usize];
                    *target_count = (*target_count).min(completed);
                }
            }

            let mut next_suffix = Vec::with_capacity(state.suffix_states.len());
            for &suffix_state in state.suffix_states.iter() {
                if let Some(target) = suffix_dfa.step(suffix_state, byte) {
                    next_suffix.push(target);
                }
            }

            close_zero_min_repeat_suffix_state(
                &mut next_body,
                &mut next_suffix,
                body_dfa,
                suffix_dfa,
                max,
            );
            if next_body.iter().all(|&count| count == u32::MAX) && next_suffix.is_empty() {
                continue;
            }

            let next = ZeroMinRepeatSuffixState {
                body_min_counts: next_body.into_boxed_slice(),
                suffix_states: next_suffix.into_boxed_slice(),
            };
            let target = if let Some(&target) = state_map.get(&next) {
                target
            } else {
                let target = dfa.add_state();
                set_zero_min_repeat_suffix_metadata(&mut dfa, target, &next, suffix_dfa);
                state_map.insert(next.clone(), target);
                worklist.push_back((target, next));
                target
            };
            transitions.push((byte, target));
        }
        dfa.set_transitions_from_sorted_entries(state_id, transitions);
    }

    // Hopcroft-style minimization is counterproductive for broad direct
    // residual DFAs: a few thousand states with dense byte rows can cost
    // seconds even when almost no states merge. Downstream product/DWA
    // construction is already designed to consume unminimized deterministic
    // components. Keep minimization for compact results where it is cheap and
    // useful, but preserve the exact direct DFA as-is above that threshold.
    let transitions = dfa_transition_count(&dfa);
    if dfa.num_states() <= 2_048 && transitions <= 100_000 {
        Some(dfa.minimize())
    } else {
        Some(dfa)
    }
}

fn build_bounded_repeat_with_regex_suffix(parts: &[Expr]) -> Option<(DFA, bool)> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile_timing.then(Instant::now);
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

    let body_started_at = profile_timing.then(Instant::now);
    let body_dfa = compile_expr_to_dfa(repeat_expr);
    let body_ms = body_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if body_dfa.num_states() == 0 || !body_dfa.finalizers(0).is_empty() {
        return None;
    }

    let suffix_expr = if parts_ref.len() == 2 {
        parts_ref[1].clone()
    } else {
        Expr::Seq(parts_ref[1..].iter().map(|e| (*e).clone()).collect())
    };
    let suffix_started_at = profile_timing.then(Instant::now);
    let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
    let suffix_ms = suffix_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
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

    // For a zero-minimum repeat, paths that reach the same body residual with
    // different completed-copy counts are ordered by language inclusion. The
    // smallest count has at least as much remaining repeat budget and may take
    // every suffix boundary available to a larger count, so larger counts are
    // redundant. Track only that minimum count per body state, plus the exact
    // set of live suffix states. This is an exact quotient of the ordinary NFA
    // subset construction and handles ambiguous body/suffix boundaries without
    // materializing the enormous unrolled-repeat powerset.
    // The dense minimum-count vector makes this quotient exceptionally fast
    // for the small repeat bodies that cause large counter products, but its
    // per-state work is quadratic in a very large body DFA. Large-body,
    // low-repeat wrappers are better served by the existing compact product
    // below (or the generic fallback when genuinely ambiguous).
    const DOMINANCE_MAX_BODY_STATES: usize = 256;
    if min == 0
        && body_dfa.num_states() <= DOMINANCE_MAX_BODY_STATES
        && let Some(dfa) = build_zero_min_repeat_suffix_dominance_dfa(
            &body_dfa,
            &suffix_dfa,
            max,
        )
    {
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] bounded_repeat_regex_suffix_dominance body_states={} suffix_states={} max={} final_states={} final_transitions={} body_ms={:.3} suffix_ms={:.3} total_ms={:.3}",
                body_dfa.num_states(),
                suffix_dfa.num_states(),
                max,
                dfa.num_states(),
                dfa_transition_count(&dfa),
                body_ms,
                suffix_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some((dfa, false));
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

    let construct_started_at = profile_timing.then(Instant::now);
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

    let construct_ms = construct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let pre_minimize_states = dfa.num_states();
    let pre_minimize_transitions = dfa_transition_count(&dfa);
    let minimize_started_at = profile_timing.then(Instant::now);
    let dfa = dfa.minimize();
    let minimize_ms = minimize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] bounded_repeat_regex_suffix body_states={} suffix_states={} max={} pre_minimize_states={} pre_minimize_transitions={} final_states={} final_transitions={} body_ms={:.3} suffix_ms={:.3} construct_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            body_dfa.num_states(),
            suffix_dfa.num_states(),
            max,
            pre_minimize_states,
            pre_minimize_transitions,
            dfa.num_states(),
            dfa_transition_count(&dfa),
            body_ms,
            suffix_ms,
            construct_ms,
            minimize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
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
    pub(super) dfa: DFA,
}

impl Regex {
    pub(crate) fn into_tokenizer(self, num_terminals: u32, exprs: Option<std::sync::Arc<[Expr]>>) -> Tokenizer {
        Tokenizer {
            dfa: self.dfa,
            num_terminals,
            exprs,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let direct_started_at = profile_timing.then(Instant::now);
    if let Some((mut dfa, needs_future_recompute)) = compile_product_component_dfa_direct(expr) {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        if needs_future_recompute {
            dfa.recompute_possible_futures();
        }
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] single_expr_path path=direct states={} transitions={} direct_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
            );
        }
        return dfa;
    }

    let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let nfa_started_at = profile_timing.then(Instant::now);
    let mut nfa = build_regex_nfa(std::slice::from_ref(expr));
    let nfa_ms = nfa_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let condense_started_at = profile_timing.then(Instant::now);
    nfa.condense_epsilon_sccs();
    let condense_ms =
        condense_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let determinize_started_at = profile_timing.then(Instant::now);
    let dfa = nfa.to_dfa();
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] single_expr_path path=nfa_dfa states={} transitions={} direct_attempt_ms={:.3} nfa_ms={:.3} condense_ms={:.3} determinize_ms={:.3}",
            dfa.num_states(),
            dfa_transition_count(&dfa),
            direct_ms,
            nfa_ms,
            condense_ms,
            determinize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    dfa
}

fn compile_with_plan(plan: ExclusionCompilePlan) -> DFA {
    compile_with_plan_internal(plan, false).0
}

fn compile_with_plan_internal(
    plan: ExclusionCompilePlan,
    capture_product_trace: bool,
) -> (DFA, Option<ProductBuildTrace>) {
    let profile_trace = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some();
    let profile_detail = profile_trace
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    // Product construction compiles many one-group component DFAs internally.
    // Timing/detail mode should describe the enclosing lexer build, not emit a
    // line (and take timestamps) for every nested leaf compile. Exhaustive trace
    // mode remains available when that low-level view is explicitly requested.
    let profile_plan = profile_trace || (profile_timing && plan.compiled_exprs.len() > 1);
    let profile_started_at = Instant::now();
    let group_set_started_at = Instant::now();
    let group_sets: Vec<U8Set> = plan
        .compiled_exprs
        .iter()
        .map(expr_u8set)
        .collect();
    let group_set_ms = profile_plan
        .then(|| group_set_started_at.elapsed().as_secs_f64() * 1000.0);
    let used_product_dfa = plan.compiled_exprs.len() > 1;

    let dfa_build_started_at = Instant::now();
    let (mut dfa, product_group_ops_applied, mut product_trace) = if plan.compiled_exprs.is_empty() {
        // A grammar can lower to the empty language, for example when a const
        // literal conflicts with sibling assertions. Keep a single non-final
        // start state so tokenizer users can still query/step the DFA safely.
        (DFA::new(1), false, None)
    } else if used_product_dfa {
        build_product_dfa(
            &plan.compiled_exprs,
            plan.profile_labels.as_deref(),
            plan.visible_groups,
            &plan.exclusions,
            &plan.intersections,
            capture_product_trace,
        )
    } else {
        (compile_single_expr_dfa(&plan.compiled_exprs[0]), false, None)
    };
    let dfa_build_ms = profile_plan
        .then(|| dfa_build_started_at.elapsed().as_secs_f64() * 1000.0);

    let metadata_started_at = Instant::now();
    let output_group_count = if product_group_ops_applied {
        plan.visible_groups
    } else {
        group_sets.len()
    };
    dfa.ensure_group_capacity(output_group_count);
    for (group_id, set) in group_sets.into_iter().take(output_group_count).enumerate() {
        dfa.set_group_u8set(group_id as u32, set);
    }
    let metadata_ms = profile_plan
        .then(|| metadata_started_at.elapsed().as_secs_f64() * 1000.0);

    let group_ops_started_at = Instant::now();
    let mut group_ops_changed = false;
    if !product_group_ops_applied && !plan.exclusions.is_empty() {
        group_ops_changed |= dfa.apply_group_exclusions(&plan.exclusions);
    }
    if !product_group_ops_applied && !plan.intersections.is_empty() {
        group_ops_changed |= dfa.apply_group_intersections(&plan.intersections);
    }
    if group_ops_changed {
        dfa.recompute_possible_futures();
    }
    let group_ops_ms = profile_plan
        .then(|| group_ops_started_at.elapsed().as_secs_f64() * 1000.0);

    let project_started_at = Instant::now();
    let dfa = if !product_group_ops_applied && plan.visible_groups < plan.compiled_exprs.len() {
        dfa.project_groups(plan.visible_groups)
    } else {
        dfa
    };
    let project_ms = profile_plan
        .then(|| project_started_at.elapsed().as_secs_f64() * 1000.0);

    let pre_minimize_states = profile_plan.then(|| dfa.num_states());
    let pre_minimize_transitions = profile_plan.then(|| dfa_transition_count(&dfa));
    let force_tokenizer_minimize = std::env::var_os("GLRMASK_FORCE_TOKENIZER_MINIMIZE").is_some();
    let minimize_started_at = Instant::now();
    let final_dfa = if used_product_dfa && !force_tokenizer_minimize {
        dfa
    } else {
        product_trace = None;
        dfa.minimize()
    };
    let minimize_ms = profile_plan
        .then(|| minimize_started_at.elapsed().as_secs_f64() * 1000.0);
    if profile_detail && profile_plan {
        let minimized_states = if used_product_dfa && !force_tokenizer_minimize {
            "not_run".to_string()
        } else {
            final_dfa.num_states().to_string()
        };
        eprintln!(
            "[glrmask/profile][tokenizer] combined groups={} visible_groups={} product_dfa={} pre_minimize_states={} pre_minimize_transitions={} final_states={} final_transitions={} minimized_states={}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            pre_minimize_states.unwrap_or_default(),
            pre_minimize_transitions.unwrap_or_default(),
            final_dfa.num_states(),
            dfa_transition_count(&final_dfa),
            minimized_states,
        );
    }
    if profile_plan {
        eprintln!(
            "[glrmask/profile][tokenizer] compile_plan groups={} visible_groups={} product_dfa={} group_set_ms={:.3} dfa_build_ms={:.3} metadata_ms={:.3} group_ops_ms={:.3} project_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            group_set_ms.unwrap_or_default(),
            dfa_build_ms.unwrap_or_default(),
            metadata_ms.unwrap_or_default(),
            group_ops_ms.unwrap_or_default(),
            project_ms.unwrap_or_default(),
            minimize_ms.unwrap_or_default(),
            profile_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (final_dfa, product_trace)
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
    build_regex_partitioned_with_adaptive(exprs, partitions, adaptive_lexer_enabled())
}

pub(crate) fn build_regex_partitioned_with_residual_isolation(
    exprs: &[Expr],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
) -> Regex {
    build_regex_partitioned_with_adaptive_and_residual_isolation(
        exprs,
        partitions,
        residual_isolation_classes,
        adaptive_lexer_enabled(),
    )
}

pub(crate) fn build_regex_partitioned_with_adaptive(
    exprs: &[Expr],
    partitions: &[u32],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(exprs, None, partitions, None, adaptive),
    }
}

pub(crate) fn build_regex_partitioned_with_adaptive_and_residual_isolation(
    exprs: &[Expr],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            None,
            partitions,
            Some(residual_isolation_classes),
            adaptive,
        ),
    }
}

pub(crate) fn build_regex_partitioned_with_profile_labels(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
) -> Regex {
    build_regex_partitioned_with_profile_labels_and_adaptive(
        exprs,
        visible_labels,
        partitions,
        adaptive_lexer_enabled(),
    )
}


pub(crate) fn build_regex_partitioned_with_profile_labels_and_residual_isolation(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
) -> Regex {
    build_regex_partitioned_with_profile_labels_and_adaptive_and_residual_isolation(
        exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive_lexer_enabled(),
    )
}

pub(crate) fn build_regex_partitioned_with_profile_labels_and_adaptive(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            Some(visible_labels),
            partitions,
            None,
            adaptive,
        ),
    }
}


pub(crate) fn build_regex_partitioned_with_profile_labels_and_adaptive_and_residual_isolation(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            Some(visible_labels),
            partitions,
            Some(residual_isolation_classes),
            adaptive,
        ),
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "invalid {name}={other:?}; expected one of 1/0, true/false, yes/no, or on/off"
            ),
        },
        Err(_) => default,
    }
}

fn adaptive_lexer_enabled() -> bool {
    env_flag("GLRMASK_LEXER_ADAPTIVE", true)
}

fn adaptive_lexer_state_limit() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(32_768)
}

fn adaptive_lexer_max_depth() -> Option<usize> {
    let Ok(value) = std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH") else {
        return Some(1);
    };
    let value = value.trim();
    if matches!(value.to_ascii_lowercase().as_str(), "full" | "unbounded") {
        return None;
    }
    Some(value.parse::<usize>().unwrap_or_else(|_| {
        panic!(
            "invalid GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH={value:?}; expected a byte depth or full"
        )
    }))
}

fn adaptive_lexer_growth_percent() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_GROWTH_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(100)
}

fn adaptive_lexer_transition_growth_percent() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_TRANSITION_GROWTH_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(600)
}

fn adaptive_transition_growth_is_acceptable(
    input_transitions: usize,
    output_transitions: usize,
    growth_percent: usize,
) -> bool {
    output_transitions.saturating_mul(100)
        <= input_transitions.saturating_mul(growth_percent)
}

fn compile_terminal_ids(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: &[usize],
) -> DFA {
    compile_terminal_ids_with_shared_duplicate_cache(exprs, visible_labels, terminal_ids, None)
}

fn compile_terminal_ids_with_shared_duplicate_cache(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: &[usize],
    shared_duplicates: Option<&Arc<SharedDuplicateNestedGroupOpCache>>,
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
    let mut nested_group_op_cache = NestedGroupOpCache {
        shared_duplicates: shared_duplicates.cloned(),
        ..NestedGroupOpCache::default()
    };
    compile_with_plan(build_exclusion_compile_plan_with_labels_and_cache(
        &local_exprs,
        local_labels.as_deref(),
        &mut nested_group_op_cache,
    ))
}

struct LexerComponent {
    terminal_ids: Vec<usize>,
    dfa: DFA,
    protected_residual: bool,
}

struct LexerComponentPair {
    terminal_ids: Vec<usize>,
    synthesized: DFA,
    full: DFA,
    full_to_synthesized: Vec<u32>,
    synthesized_state_representatives_by_horizon: Vec<(usize, Vec<u32>)>,
    analysis_horizons: Vec<LexerComponentAnalysisHorizon>,
    protected_residual: bool,
}

struct LexerComponentAnalysisHorizon {
    horizon: usize,
    tokenizer_dfa: DFA,
    source_to_analysis: Vec<u32>,
}

pub(crate) struct CompiledPartitionedExpressionPair {
    pub(crate) synthesized: Regex,
    pub(crate) full: Regex,
    pub(crate) full_to_synthesized: Vec<u32>,
    pub(crate) synthesized_state_representatives_by_horizon: Vec<(usize, Vec<u32>)>,
    pub(crate) component_quotient_plans: Vec<StructuralComponentQuotientPlan>,
}

#[derive(Clone)]
pub(crate) struct StructuralComponentQuotientPlan {
    pub(crate) global_offset: u32,
    pub(crate) source_state_count: usize,
    /// Local terminal id `i` in each horizon tokenizer corresponds to
    /// `terminal_ids[i]` in the grammar-wide terminal domain.
    pub(crate) terminal_ids: Vec<u32>,
    pub(crate) horizons: Vec<StructuralComponentHorizonPlan>,
}

#[derive(Clone)]
pub(crate) struct StructuralComponentHorizonPlan {
    pub(crate) horizon: usize,
    pub(crate) tokenizer: Tokenizer,
    /// Source component state -> analysis-tokenizer state. The map is
    /// certified for every byte string up to `horizon` bytes.
    pub(crate) source_to_analysis: Vec<u32>,
}

fn compile_partition_components(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
) -> Vec<LexerComponent> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let rewritten_exprs = materialize_repeated_subexpression_dfas(exprs);
    let exprs = rewritten_exprs.as_deref().unwrap_or(exprs);
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }
    if let Some(classes) = residual_isolation_classes {
        assert_eq!(
            classes.len(),
            exprs.len(),
            "one residual isolation class entry is required per terminal",
        );
        for terminal_ids in grouped.values() {
            let mut class = None;
            let mut has_unprotected = false;
            for &terminal in terminal_ids {
                match classes[terminal] {
                    Some(current) => match class {
                        Some(previous) => assert_eq!(
                            previous, current,
                            "one lexer partition cannot contain distinct protected residual classes",
                        ),
                        None => class = Some(current),
                    },
                    None => has_unprotected = true,
                }
            }
            assert!(
                class.is_none() || !has_unprotected,
                "one lexer partition cannot mix protected and unprotected residual coordinates",
            );
        }
    }
    let shared_duplicates = shared_duplicate_nested_group_op_cache(exprs, &grouped);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }

    grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(partition, terminal_ids)| {
            let started_at = Instant::now();
            let dfa = compile_terminal_ids_with_shared_duplicate_cache(
                exprs,
                visible_labels,
                &terminal_ids,
                shared_duplicates.as_ref(),
            );
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] partition_compile partition={} terminals={} states={} transitions={} total_ms={:.3}",
                    partition,
                    terminal_ids.len(),
                    dfa.num_states(),
                    dfa_transition_count(&dfa),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let protected_residual = residual_isolation_classes.is_some_and(|classes| {
                terminal_ids
                    .first()
                    .is_some_and(|&terminal| classes[terminal].is_some())
            });
            LexerComponent {
                terminal_ids,
                dfa,
                protected_residual,
            }
        })
        .collect()
}

fn lexer_component_product_metadata(
    components: &[LexerComponent],
    group_offsets: &[usize],
    state_tuple: &ProductStateTuple,
    total_groups: usize,
) -> (BitSet, BitSet) {
    let mut finalizers = BitSet::new(total_groups);
    let mut futures = BitSet::new(total_groups);

    for &(component_id, component_state) in state_tuple {
        let component_index = component_id as usize;
        let component = &components[component_index].dfa;
        let offset = group_offsets[component_index];
        for group in component.finalizers(component_state).iter() {
            finalizers.set(offset + group);
        }
        for group in component
            .possible_future_group_ids(component_state)
            .iter()
        {
            futures.set(offset + group);
        }
    }

    (finalizers, futures)
}

fn compute_lexer_component_equivalence_classes(
    components: &[LexerComponent],
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for component in components {
        for state in component.dfa.states() {
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

/// Attempt one exact prefix determinization of the final union of independently
/// compiled lexer partitions. The sparse product tuple contains only component
/// states still live after the consumed bytes. Product construction stops at
/// `max_depth` consumed bytes and reconnects each frontier tuple to exact copies
/// of its live component states with epsilon edges. Thus adaptive
/// determinization can coalesce a bounded prefix without forcing unrelated
/// long-running terminals into the product for their entire lifetime.
///
/// Construction also stops before allocating the first product state beyond
/// `state_limit`; callers can then preserve the original epsilon-NFA unchanged.
/// `None` retains the historical full-product behavior.
fn try_product_union_components(
    components: &[LexerComponent],
    state_limit: usize,
    transition_limit: usize,
    max_depth: Option<usize>,
) -> Option<DFA> {
    assert!(state_limit > 0, "adaptive lexer state limit must be positive");
    assert!(transition_limit > 0, "adaptive lexer transition limit must be positive");
    debug_assert!(components
        .iter()
        .all(|component| !component.dfa.has_epsilon_transitions()));

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let setup_started_at = Instant::now();
    let total_groups = components
        .iter()
        .map(|component| component.dfa.num_groups())
        .sum::<usize>();
    let mut group_offsets = Vec::with_capacity(components.len());
    let mut group_offset = 0usize;
    for component in components {
        group_offsets.push(group_offset);
        group_offset += component.dfa.num_groups();
    }
    let component_dead_states = components
        .iter()
        .map(|component| explicit_dead_sink_state(&component.dfa))
        .collect::<Vec<_>>();
    let (class_map, class_members) = compute_lexer_component_equivalence_classes(components);
    let component_class_transitions = components
        .iter()
        .map(|component| build_product_class_transitions_for_dfa(&component.dfa, &class_map))
        .collect::<Vec<_>>();
    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_classes = class_members.len();

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(total_groups);
    for (component_index, component) in components.iter().enumerate() {
        let offset = group_offsets[component_index];
        for local_group in 0..component.dfa.num_groups() {
            combined.set_group_u8set(
                (offset + local_group) as u32,
                *component.dfa.group_id_to_u8set(local_group as u32),
            );
        }
    }

    let mut start = ProductStateTuple::with_capacity(components.len());
    for component_id in 0..components.len() {
        start.push((component_id as u32, 0));
    }
    let (finalizers, futures) =
        lexer_component_product_metadata(components, &group_offsets, &start, total_groups);
    combined.overwrite_state_metadata(0, finalizers, futures);

    #[derive(Clone, PartialEq, Eq, Hash)]
    enum ProductStateKey {
        Full(ProductStateTuple),
        Bounded(usize, ProductStateTuple),
    }
    let state_key = |depth: usize, tuple: ProductStateTuple| match max_depth {
        Some(_) => ProductStateKey::Bounded(depth, tuple),
        None => ProductStateKey::Full(tuple),
    };

    let mut state_map = FxHashMap::<ProductStateKey, u32>::default();
    state_map.insert(state_key(0, start.clone()), 0);
    let mut worklist = VecDeque::from([(0u32, start, 0usize)]);
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    let mut frontier_states = Vec::<(u32, ProductStateTuple)>::new();
    let mut class_buffers = (0..num_classes)
        .map(|_| ProductStateTuple::new())
        .collect::<Vec<_>>();
    let mut class_active = vec![false; num_classes];
    let mut used_classes = Vec::<usize>::new();
    let mut projected_byte_transitions = 0usize;

    let state_expand_started_at = Instant::now();
    while let Some((combined_state, state_tuple, depth)) = worklist.pop_front() {
        if max_depth.is_some_and(|max_depth| depth >= max_depth) {
            frontier_states.push((combined_state, state_tuple));
            continue;
        }
        for &(component_id, component_state) in &state_tuple {
            let component_index = component_id as usize;
            for &(class_id, target) in
                &component_class_transitions[component_index][component_state as usize]
            {
                let class_index = class_id as usize;
                if !class_active[class_index] {
                    class_active[class_index] = true;
                    used_classes.push(class_index);
                }
                if component_dead_states[component_index] == Some(target) {
                    continue;
                }
                class_buffers[class_index].push((component_id, target));
            }
        }

        let mut transitions = Vec::with_capacity(used_classes.len());
        for &class_index in &used_classes {
            projected_byte_transitions = projected_byte_transitions
                .saturating_add(class_members[class_index].len());
            if projected_byte_transitions > transition_limit {
                return None;
            }
            let next_tuple = &class_buffers[class_index];
            let next_depth = depth + 1;
            let next_key = state_key(next_depth, next_tuple.clone());
            let target = if let Some(&existing) = state_map.get(&next_key) {
                existing
            } else {
                if combined.num_states() >= state_limit {
                    return None;
                }
                let new_state = combined.add_state();
                let (finalizers, futures) = lexer_component_product_metadata(
                    components,
                    &group_offsets,
                    next_tuple,
                    total_groups,
                );
                combined.overwrite_state_metadata(new_state, finalizers, futures);
                state_map.insert(next_key, new_state);
                pending_class_transitions.push(Vec::new());
                worklist.push_back((new_state, next_tuple.clone(), next_depth));
                new_state
            };
            transitions.push((class_index as u8, target));
            class_buffers[class_index].clear();
            class_active[class_index] = false;
        }
        used_classes.clear();
        pending_class_transitions[combined_state as usize] = transitions;
    }
    let state_expand_ms = state_expand_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_expand_started_at = Instant::now();
    let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
        pending_class_transitions
            .into_par_iter()
            .map(|class_transitions| {
                let byte_capacity = class_transitions
                    .iter()
                    .map(|(class_id, _)| class_members[*class_id as usize].len())
                    .sum::<usize>();
                const DENSE_BYTE_EXPANSION_THRESHOLD: usize = 96;
                let transitions = if byte_capacity >= DENSE_BYTE_EXPANSION_THRESHOLD {
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
    for (state, transitions) in combined.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }

    if max_depth.is_some() {
        // Append one exact copy of every independently compiled component. A
        // product frontier can then resume from the precise component states in
        // its sparse tuple instead of continuing cross-component
        // determinization.
        let mut component_offsets = Vec::with_capacity(components.len());
        for (component_index, component) in components.iter().enumerate() {
            let component_offset = combined.num_states() as u32;
            component_offsets.push(component_offset);
            for _ in 0..component.dfa.num_states() {
                combined.add_state();
            }

            let group_offset = group_offsets[component_index];
            for (state_index, state) in component.dfa.states().iter().enumerate() {
                let mapped_state = component_offset + state_index as u32;
                combined.set_transitions_from_sorted_entries(
                    mapped_state,
                    state
                        .transitions
                        .iter()
                        .map(|(byte, &target)| (byte, component_offset + target))
                        .collect(),
                );
                for &target in &state.epsilon_transitions {
                    combined.add_epsilon_transition(mapped_state, component_offset + target);
                }

                let mut finalizers = BitSet::new(total_groups);
                let mut futures = BitSet::new(total_groups);
                for local_group in state.finalizers.iter() {
                    finalizers.set(group_offset + local_group);
                }
                for local_group in component
                    .dfa
                    .possible_future_group_ids(state_index as u32)
                    .iter()
                {
                    futures.set(group_offset + local_group);
                }
                combined.overwrite_state_metadata(mapped_state, finalizers, futures);
            }
        }

        for (frontier_state, state_tuple) in frontier_states {
            // The frontier is an epsilon branching state, like the root of the
            // ordinary partition union. Its exact component children carry
            // acceptance. Duplicating finalizers on the branch state creates a
            // second accepting path for the same terminal and is observably
            // different to the original epsilon-NFA for downstream state-set
            // analyses. Strict futures may remain cached on the branch state,
            // just as they are on the ordinary epsilon-union root.
            let futures = combined.possible_future_group_ids(frontier_state).clone();
            combined.overwrite_state_metadata(
                frontier_state,
                BitSet::new(total_groups),
                futures,
            );
            for (component_id, component_state) in state_tuple {
                combined.add_epsilon_transition(
                    frontier_state,
                    component_offsets[component_id as usize] + component_state,
                );
            }
        }
    }
    let byte_expand_ms = byte_expand_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_product states={} classes={} max_depth={:?} setup_ms={:.3} state_expand_ms={:.3} byte_expand_ms={:.3}",
            combined.num_states(),
            num_classes,
            max_depth,
            setup_ms,
            state_expand_ms,
            byte_expand_ms,
        );
    }

    Some(combined)
}

fn adaptively_determinize_components(
    inputs: Vec<LexerComponent>,
    state_limit: usize,
) -> Vec<LexerComponent> {
    adaptively_determinize_components_with_limits(
        inputs,
        state_limit,
        adaptive_lexer_growth_percent(),
        adaptive_lexer_transition_growth_percent(),
        adaptive_lexer_max_depth(),
    )
}

fn adaptively_determinize_components_with_limits(
    inputs: Vec<LexerComponent>,
    state_limit: usize,
    growth_percent: usize,
    transition_growth_percent: usize,
    max_depth: Option<usize>,
) -> Vec<LexerComponent> {
    if inputs.iter().any(|component| component.protected_residual) {
        let mut protected = Vec::new();
        let mut ordinary = Vec::new();
        for component in inputs {
            if component.protected_residual {
                protected.push(component);
            } else {
                ordinary.push(component);
            }
        }

        let mut output = if ordinary.len() >= 2 {
            adaptively_determinize_components_with_limits(
                ordinary,
                state_limit,
                growth_percent,
                transition_growth_percent,
                max_depth,
            )
        } else {
            ordinary
        };
        output.append(&mut protected);
        output.sort_unstable_by_key(|component| {
            component.terminal_ids.first().copied().unwrap_or(usize::MAX)
        });
        return output;
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = Instant::now();
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

    let growth_limit = input_states
        .saturating_mul(growth_percent)
        .saturating_add(99)
        / 100;
    let effective_state_limit = state_limit.min(growth_limit).max(1);
    let attempt_started_at = Instant::now();
    let transition_limit = input_transitions
        .saturating_mul(transition_growth_percent)
        / 100;
    let candidate = try_product_union_components(
        &inputs,
        effective_state_limit,
        transition_limit.max(1),
        max_depth,
    );
    let attempt_ms = attempt_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_transitions = candidate.as_ref().map(dfa_transition_count);
    let accepted = candidate_transitions.is_some_and(|output_transitions| {
        adaptive_transition_growth_is_acceptable(
            input_transitions,
            output_transitions,
            transition_growth_percent,
        )
    });
    let terminal_ids = inputs
        .iter()
        .flat_map(|component| component.terminal_ids.iter().copied())
        .collect::<Vec<_>>();
    let output_states = candidate.as_ref().map_or(input_states, DFA::num_states);
    let output_transitions = candidate_transitions.unwrap_or(input_transitions);

    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_determinize partitions={} terminals={} output_components={} input_states={} output_states={} input_transitions={} output_transitions={} attempted=1 accepted={} max_states={} effective_state_limit={} max_depth={:?} max_growth_percent={} max_transition_growth_percent={} attempt_ms={:.3} total_ms={:.3}",
            input_batches,
            terminals,
            if accepted { 1 } else { input_batches },
            input_states,
            output_states,
            input_transitions,
            output_transitions,
            accepted,
            state_limit,
            effective_state_limit,
            max_depth,
            growth_percent,
            transition_growth_percent,
            attempt_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    match candidate {
        Some(dfa) if accepted => vec![LexerComponent {
            terminal_ids,
            dfa,
            protected_residual: false,
        }],
        _ => inputs,
    }
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
    residual_isolation_classes: Option<&[Option<u32>]>,
    adaptive: bool,
) -> DFA {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = Instant::now();
    assert_eq!(exprs.len(), partitions.len(), "one lexer partition id is required per terminal");
    if let Some(classes) = residual_isolation_classes {
        assert_eq!(
            exprs.len(),
            classes.len(),
            "one residual isolation class entry is required per terminal",
        );
    }
    if let Some(labels) = visible_labels {
        assert_eq!(exprs.len(), labels.len(), "one profile label is required per terminal");
    }
    if exprs.is_empty() {
        return compile_with_plan(build_exclusion_compile_plan(exprs));
    }

    let num_partitions = partitions.iter().copied().collect::<BTreeSet<_>>().len();
    if num_partitions == 1 {
        return compile_with_plan(build_exclusion_compile_plan_with_labels(exprs, visible_labels));
    }

    // Every partition is compiled exactly as declared, independently of the
    // adaptive policy. Adaptive determinization is a second, generic step over
    // the disjoint deterministic components of the combined epsilon-NFA.
    let partition_compile_started_at = Instant::now();
    let mut components = compile_partition_components(
        exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
    );
    let partition_compile_ms = partition_compile_started_at.elapsed().as_secs_f64() * 1000.0;

    let adaptive_started_at = Instant::now();
    if adaptive {
        components =
            adaptively_determinize_components(components, adaptive_lexer_state_limit());
    }
    let adaptive_ms = adaptive_started_at.elapsed().as_secs_f64() * 1000.0;

    if let [component] = components.as_slice() {
        return remap_component_groups(&component.dfa, &component.terminal_ids, exprs.len());
    }

    let combine_started_at = Instant::now();
    let total_states = 1usize
        + components
            .iter()
            .map(|component| component.dfa.num_states())
            .sum::<usize>();
    let mut combined = DFA::new(total_states);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());

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
        for local_group in component.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
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
            for local_group in state.finalizers.iter() {
                finalizers.set(terminal_ids[local_group]);
            }
            for local_group in component.possible_future_group_ids(state_index as u32).iter() {
                futures.set(terminal_ids[local_group]);
            }
            combined.overwrite_state_metadata(mapped_state, finalizers, futures);
        }
        offset += component.num_states() as u32;
    }
    let combine_ms = combine_started_at.elapsed().as_secs_f64() * 1000.0;

    let futures_started_at = Instant::now();
    // Components are disjoint below a new epsilon-only root. Their strict
    // possible-future sets remain exact after local->global terminal remapping.
    // The root's strict futures are exactly the union of the component start
    // states' strict futures; no generic epsilon fixpoint is needed.
    combined.set_possible_future_group_ids(0, root_futures);
    let futures_ms = futures_started_at.elapsed().as_secs_f64() * 1000.0;
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] partitioned_build terminals={} partitions={} partition_compile_ms={:.3} adaptive_ms={:.3} combine_ms={:.3} futures_ms={:.3} total_ms={:.3}",
            exprs.len(),
            num_partitions,
            partition_compile_ms,
            adaptive_ms,
            combine_ms,
            futures_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    combined
}

fn combine_component_pairs_under_epsilon_root(
    components: &[LexerComponentPair],
    total_groups: usize,
    full_side: bool,
) -> (DFA, Vec<u32>) {
    let total_states = 1usize
        + components
            .iter()
            .map(|component| {
                if full_side {
                    component.full.num_states()
                } else {
                    component.synthesized.num_states()
                }
            })
            .sum::<usize>();
    let mut combined = DFA::new(total_states);
    combined.ensure_group_capacity(total_groups);
    let mut root_futures = BitSet::new(total_groups);
    let mut offsets = Vec::with_capacity(components.len());

    let mut offset = 1u32;
    for component_pair in components {
        offsets.push(offset);
        let terminal_ids = &component_pair.terminal_ids;
        let component = if full_side {
            &component_pair.full
        } else {
            &component_pair.synthesized
        };
        debug_assert_eq!(component.num_groups(), terminal_ids.len());
        combined.add_epsilon_transition(0, offset);

        for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
            combined.set_group_u8set(
                terminal_id as u32,
                *component.group_id_to_u8set(local_group as u32),
            );
        }
        for local_group in component.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }

        for (state_index, state) in component.states().iter().enumerate() {
            let mapped_state = offset + state_index as u32;
            combined.set_transitions_from_sorted_entries(
                mapped_state,
                state
                    .transitions
                    .iter()
                    .map(|(byte, &target)| (byte, offset + target))
                    .collect(),
            );
            for &target in &state.epsilon_transitions {
                combined.add_epsilon_transition(mapped_state, offset + target);
            }

            let mut finalizers = BitSet::new(total_groups);
            let mut futures = BitSet::new(total_groups);
            for local_group in state.finalizers.iter() {
                finalizers.set(terminal_ids[local_group]);
            }
            for local_group in component.possible_future_group_ids(state_index as u32).iter() {
                futures.set(terminal_ids[local_group]);
            }
            combined.overwrite_state_metadata(mapped_state, finalizers, futures);
        }
        offset += component.num_states() as u32;
    }
    combined.set_possible_future_group_ids(0, root_futures);
    (combined, offsets)
}

/// Compile independently protected terminal partitions as exact/synthesized
/// pairs while compiling every unchanged partition only once. The returned
/// state map is structural: the global epsilon root maps to the global root,
/// unchanged components map identically, and protected components use their
/// certified local product maps. Adaptive prefix determinization remains
/// available for the ordinary components but never crosses a protected
/// residual coordinate.
pub(crate) fn compile_partitioned_expression_pair_with_structural_maps(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    partition_stencil: Option<(usize, &[Expr])>,
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledPartitionedExpressionPair> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    if full_exprs.len() != synthesized_exprs.len()
        || partition_stencil.is_some_and(|(_, exprs)| exprs.len() != full_exprs.len())
        || full_exprs.len() != partitions.len()
        || full_exprs.len() != residual_isolation_classes.len()
        || visible_labels.is_some_and(|labels| labels.len() != full_exprs.len())
        || full_exprs.is_empty()
    {
        return None;
    }

    let changed = full_exprs
        .iter()
        .zip(synthesized_exprs)
        .zip(residual_isolation_classes)
        .map(|((full, synthesized), isolation_class)| {
            isolation_class.is_some() && full != synthesized
        })
        .collect::<Vec<_>>();
    if !changed.iter().any(|&changed| changed) {
        return None;
    }

    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }

    // Only ordinary terminals participate in repeated-subexpression sharing.
    // Replacing changed expressions with epsilon avoids compiling their large
    // exact bodies merely to prepare a cache they cannot use.
    let mut ordinary_exprs = synthesized_exprs.to_vec();
    for (terminal, &is_changed) in changed.iter().enumerate() {
        if is_changed {
            ordinary_exprs[terminal] = Expr::Epsilon;
        }
    }
    let rewritten_ordinary = materialize_repeated_subexpression_dfas(&ordinary_exprs);
    let ordinary_exprs = rewritten_ordinary.as_deref().unwrap_or(&ordinary_exprs);
    let ordinary_groups = grouped
        .iter()
        .filter_map(|(&partition, terminals)| {
            terminals
                .iter()
                .all(|&terminal| !changed[terminal])
                .then_some((partition, terminals.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let shared_duplicates = shared_duplicate_nested_group_op_cache(ordinary_exprs, &ordinary_groups);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }

    let compiled = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_partition, terminal_ids)| {
            let changed_terminals = terminal_ids
                .iter()
                .copied()
                .filter(|&terminal| changed[terminal])
                .collect::<Vec<_>>();
            if changed_terminals.is_empty() {
                // A bound can be visible to the longest token in the complete
                // vocabulary while remaining unobservable inside the dominant
                // <=64-byte partitions. For a large singleton component, build
                // one exact/stencil pair now and retain the exact DFA as the
                // compile/runtime source coordinate. The stencil is analysis
                // only and is selected later solely for partitions within its
                // certified horizon.
                if terminal_ids.len() == 1
                    && let Some((stencil_horizon, stencil_exprs)) = partition_stencil
                {
                    let terminal = terminal_ids[0];
                    if stencil_exprs[terminal] != full_exprs[terminal] {
                        let pair = compile_terminal_expression_pair_with_structural_map(
                            &full_exprs[terminal],
                            &stencil_exprs[terminal],
                            stencil_horizon,
                            relevant_bytes,
                        );
                        if pair.is_none() && profile {
                            eprintln!(
                                "[glrmask/profile][tokenizer] partition_analysis_stencil_rejected terminal={} horizon={} reason=structural_pair_failed full={:?} stencil={:?}",
                                terminal,
                                stencil_horizon,
                                expr_profile_summary(&full_exprs[terminal]),
                                expr_profile_summary(&stencil_exprs[terminal]),
                            );
                        }
                        if let Some(pair) = pair {
                            let exact_states = pair.full.dfa.num_states();
                            let stencil_states = pair.synthesized.dfa.num_states();
                            let profitable = exact_states >= 10_000
                                && stencil_states < exact_states
                                && stencil_states.saturating_mul(4)
                                    <= exact_states.saturating_mul(3);
                            if profitable {
                                let exact_dfa = pair.full.dfa;
                                let stencil_dfa = pair.synthesized.dfa;
                                let full_to_stencil = pair.full_to_synthesized;
                                let stencil_quotients =
                                    pair.synthesized_state_representatives_by_horizon;
                                let state_count = exact_dfa.num_states() as u32;
                                let horizons = structural_quotient_horizons(max_token_len);
                                let identity_horizons = horizons
                                    .iter()
                                    .copied()
                                    .map(|horizon| (horizon, (0..state_count).collect()))
                                    .collect();
                                let mut analysis_horizons = stencil_quotients
                                    .into_iter()
                                    .filter(|(horizon, _)| *horizon <= stencil_horizon)
                                    .map(|(horizon, stencil_to_representative)| {
                                        let source_to_analysis = full_to_stencil
                                            .iter()
                                            .map(|&stencil_state| {
                                                stencil_to_representative
                                                    [stencil_state as usize]
                                            })
                                            .collect();
                                        LexerComponentAnalysisHorizon {
                                            horizon,
                                            tokenizer_dfa: stencil_dfa.clone(),
                                            source_to_analysis,
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                if stencil_horizon < max_token_len {
                                    analysis_horizons.push(LexerComponentAnalysisHorizon {
                                        horizon: max_token_len,
                                        tokenizer_dfa: exact_dfa.clone(),
                                        source_to_analysis: (0..state_count).collect(),
                                    });
                                }
                                if profile {
                                    eprintln!(
                                        "[glrmask/profile][tokenizer] partition_analysis_stencil terminal={} horizon={} exact_states={} stencil_states={} saving={}",
                                        terminal,
                                        stencil_horizon,
                                        exact_states,
                                        stencil_states,
                                        exact_states.saturating_sub(stencil_states),
                                    );
                                }
                                return Some(LexerComponentPair {
                                    terminal_ids,
                                    synthesized: exact_dfa.clone(),
                                    full: exact_dfa,
                                    full_to_synthesized: (0..state_count).collect(),
                                    synthesized_state_representatives_by_horizon: identity_horizons,
                                    analysis_horizons,
                                    // Preserve this component boundary so adaptive
                                    // determinization cannot erase the source
                                    // coordinate used by the horizon maps.
                                    protected_residual: true,
                                });
                            }
                            if profile {
                                eprintln!(
                                    "[glrmask/profile][tokenizer] partition_analysis_stencil_rejected terminal={} horizon={} reason=insufficient_reduction exact_states={} stencil_states={}",
                                    terminal,
                                    stencil_horizon,
                                    exact_states,
                                    stencil_states,
                                );
                            }
                        }
                    } else if profile {
                        eprintln!(
                            "[glrmask/profile][tokenizer] partition_analysis_stencil_rejected terminal={} horizon={} reason=expression_unchanged",
                            terminal,
                            stencil_horizon,
                        );
                    }
                }

                let dfa = compile_terminal_ids_with_shared_duplicate_cache(
                    ordinary_exprs,
                    visible_labels,
                    &terminal_ids,
                    shared_duplicates.as_ref(),
                );
                let minimize_started_at = profile.then(Instant::now);
                let before_states = dfa.num_states();
                let dfa = dfa.minimize();
                if profile {
                    let labels = visible_labels
                        .map(|labels| {
                            terminal_ids
                                .iter()
                                .filter_map(|&terminal| labels.get(terminal))
                                .take(4)
                                .cloned()
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    eprintln!(
                        "[glrmask/profile][tokenizer] paired_ordinary_minimize terminals={} terminal_ids={:?} labels={:?} states_before={} states_after={} elapsed_ms={:.3}",
                        terminal_ids.len(),
                        terminal_ids,
                        labels,
                        before_states,
                        dfa.num_states(),
                        minimize_started_at
                            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    );
                }
                let state_count = dfa.num_states() as u32;
                let analysis_dfa = dfa.clone();
                return Some(LexerComponentPair {
                    terminal_ids,
                    synthesized: dfa.clone(),
                    full: dfa,
                    full_to_synthesized: (0..state_count).collect(),
                    synthesized_state_representatives_by_horizon:
                        structural_quotient_horizons(max_token_len)
                            .into_iter()
                            .map(|horizon| (horizon, (0..state_count).collect()))
                            .collect(),
                    analysis_horizons: vec![LexerComponentAnalysisHorizon {
                        horizon: max_token_len,
                        tokenizer_dfa: analysis_dfa,
                        source_to_analysis: (0..state_count).collect(),
                    }],
                    protected_residual: false,
                });
            }

            if changed_terminals.len() != 1 || terminal_ids.len() != 1 {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=non_singleton_partition terminals={:?} changed={:?}",
                        terminal_ids,
                        changed_terminals,
                    );
                }
                return None;
            }
            let terminal = changed_terminals[0];
            residual_isolation_classes[terminal]?;
            let pair = compile_terminal_expression_pair_with_structural_map(
                &full_exprs[terminal],
                &synthesized_exprs[terminal],
                max_token_len,
                relevant_bytes,
            );
            let Some(pair) = pair else {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=local_pair_failed terminal={} full_expr={:?} synthesized_expr={:?}",
                        terminal,
                        expr_profile_summary(&full_exprs[terminal]),
                        expr_profile_summary(&synthesized_exprs[terminal]),
                    );
                }
                return None;
            };
            let synthesized_dfa = pair.synthesized.dfa;
            let analysis_horizons = pair
                .synthesized_state_representatives_by_horizon
                .iter()
                .map(|(horizon, representatives)| LexerComponentAnalysisHorizon {
                    horizon: *horizon,
                    tokenizer_dfa: synthesized_dfa.clone(),
                    source_to_analysis: representatives.clone(),
                })
                .collect();
            Some(LexerComponentPair {
                terminal_ids,
                synthesized: synthesized_dfa,
                full: pair.full.dfa,
                full_to_synthesized: pair.full_to_synthesized,
                synthesized_state_representatives_by_horizon:
                    pair.synthesized_state_representatives_by_horizon,
                analysis_horizons,
                protected_residual: true,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let mut protected = Vec::new();
    let mut ordinary = Vec::new();
    for pair in compiled {
        if pair.protected_residual {
            protected.push(pair);
        } else {
            ordinary.push(LexerComponent {
                terminal_ids: pair.terminal_ids,
                dfa: pair.full,
                protected_residual: false,
            });
        }
    }

    let ordinary = if adaptive && ordinary.len() >= 2 {
        adaptively_determinize_components(ordinary, adaptive_lexer_state_limit())
    } else {
        ordinary
    };
    let mut pairs = ordinary
        .into_iter()
        .map(|component| {
            let state_count = component.dfa.num_states() as u32;
            let analysis_dfa = component.dfa.clone();
            LexerComponentPair {
                terminal_ids: component.terminal_ids,
                synthesized: component.dfa.clone(),
                full: component.dfa,
                full_to_synthesized: (0..state_count).collect(),
                synthesized_state_representatives_by_horizon:
                    structural_quotient_horizons(max_token_len)
                        .into_iter()
                        .map(|horizon| (horizon, (0..state_count).collect()))
                        .collect(),
                analysis_horizons: vec![LexerComponentAnalysisHorizon {
                    horizon: max_token_len,
                    tokenizer_dfa: analysis_dfa,
                    source_to_analysis: (0..state_count).collect(),
                }],
                protected_residual: false,
            }
        })
        .collect::<Vec<_>>();
    pairs.append(&mut protected);
    pairs.sort_unstable_by_key(|pair| {
        pair.terminal_ids.first().copied().unwrap_or(usize::MAX)
    });

    let (synthesized, synthesized_offsets) =
        combine_component_pairs_under_epsilon_root(&pairs, full_exprs.len(), false);
    let (full, full_offsets) =
        combine_component_pairs_under_epsilon_root(&pairs, full_exprs.len(), true);
    let mut full_to_synthesized = Vec::with_capacity(full.num_states());
    let horizons = structural_quotient_horizons(max_token_len);
    let mut synthesized_state_representatives_by_horizon = horizons
        .iter()
        .copied()
        .map(|horizon| (horizon, vec![0u32]))
        .collect::<Vec<_>>();
    full_to_synthesized.push(0);
    for (component_index, pair) in pairs.iter().enumerate() {
        let synthesized_offset = synthesized_offsets[component_index];
        let full_offset = full_offsets[component_index];
        debug_assert_eq!(full_to_synthesized.len(), full_offset as usize);
        if pair.full_to_synthesized.len() != pair.full.num_states() {
            return None;
        }
        full_to_synthesized.extend(
            pair.full_to_synthesized
                .iter()
                .map(|&state| synthesized_offset + state),
        );
        if pair.synthesized_state_representatives_by_horizon.len() != horizons.len() {
            return None;
        }
        for ((expected_horizon, global), (pair_horizon, local)) in
            synthesized_state_representatives_by_horizon
                .iter_mut()
                .zip(&pair.synthesized_state_representatives_by_horizon)
        {
            if expected_horizon != pair_horizon || local.len() != pair.synthesized.num_states() {
                return None;
            }
            global.extend(local.iter().map(|&state| synthesized_offset + state));
        }
    }
    if full_to_synthesized.len() != full.num_states()
        || synthesized_state_representatives_by_horizon
            .iter()
            .any(|(_, representatives)| representatives.len() != synthesized.num_states())
    {
        return None;
    }
    let component_quotient_plans = pairs
        .iter()
        .enumerate()
        .map(|(component_index, pair)| StructuralComponentQuotientPlan {
            global_offset: synthesized_offsets[component_index],
            source_state_count: pair.synthesized.num_states(),
            terminal_ids: pair.terminal_ids.iter().map(|&terminal| terminal as u32).collect(),
            horizons: pair
                .analysis_horizons
                .iter()
                .map(|analysis| StructuralComponentHorizonPlan {
                    horizon: analysis.horizon,
                    tokenizer: Regex {
                        dfa: analysis.tokenizer_dfa.clone(),
                    }
                    .into_tokenizer(pair.terminal_ids.len() as u32, None),
                    source_to_analysis: analysis.source_to_analysis.clone(),
                })
                .collect(),
        })
        .collect();

    Some(CompiledPartitionedExpressionPair {
        synthesized: Regex { dfa: synthesized },
        full: Regex { dfa: full },
        full_to_synthesized,
        synthesized_state_representatives_by_horizon,
        component_quotient_plans,
    })
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

fn product_state_single_visible_finalizer(
    components: &[ProductComponent],
    state_tuple: &ProductStateTuple,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> BitSet {
    let mut accepting = vec![false; components.len()];
    for &(group_id, state) in state_tuple {
        let group = group_id as usize;
        accepting[group] = match &components[group] {
            ProductComponent::Materialized(dfa) => dfa.finalizers(state).contains(0),
            ProductComponent::VirtualBoundedRepeat { base_dfa, min, .. } => {
                let base_state_count = base_dfa.num_states() as u32;
                state % base_state_count == 0 && state / base_state_count >= *min
            }
        };
    }

    let mut visible_accepting = accepting.first().copied().unwrap_or(false);
    if visible_accepting
        && exclusions
            .get(&0)
            .is_some_and(|blocked| blocked.iter().any(|&group| accepting[group as usize]))
    {
        visible_accepting = false;
    }
    if visible_accepting
        && intersections
            .get(&0)
            .is_some_and(|required| required.iter().any(|&group| !accepting[group as usize]))
    {
        visible_accepting = false;
    }

    let mut finalizers = BitSet::new(1);
    if visible_accepting {
        finalizers.set(0);
    }
    finalizers
}

fn set_single_group_futures_from_class_graph(
    dfa: &mut DFA,
    class_transitions: &[Vec<(u8, u32)>],
) {
    let states = dfa.num_states();
    debug_assert_eq!(states, class_transitions.len());
    let edge_count = class_transitions.iter().map(Vec::len).sum::<usize>();
    let mut predecessor_counts = vec![0u32; states];
    for transitions in class_transitions {
        for &(_, target) in transitions {
            predecessor_counts[target as usize] += 1;
        }
    }

    let mut offsets = vec![0usize; states + 1];
    for state in 0..states {
        offsets[state + 1] = offsets[state] + predecessor_counts[state] as usize;
    }
    debug_assert_eq!(offsets[states], edge_count);
    let mut write_offsets = offsets[..states].to_vec();
    let mut predecessors = vec![0u32; edge_count];
    for (source, transitions) in class_transitions.iter().enumerate() {
        for &(_, target) in transitions {
            let slot = &mut write_offsets[target as usize];
            predecessors[*slot] = source as u32;
            *slot += 1;
        }
    }

    let mut can_reach_final = vec![false; states];
    let mut queue = VecDeque::<u32>::new();
    for state in 0..states {
        if dfa.finalizers(state as u32).contains(0) {
            can_reach_final[state] = true;
            queue.push_back(state as u32);
        }
    }
    while let Some(target) = queue.pop_front() {
        let target = target as usize;
        for &source in &predecessors[offsets[target]..offsets[target + 1]] {
            if !can_reach_final[source as usize] {
                can_reach_final[source as usize] = true;
                queue.push_back(source);
            }
        }
    }

    for (source, transitions) in class_transitions.iter().enumerate() {
        let mut future = BitSet::new(1);
        if transitions
            .iter()
            .any(|&(_, target)| can_reach_final[target as usize])
        {
            future.set(0);
        }
        dfa.set_possible_future_group_ids(source as u32, future);
    }
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
        Expr::U8Seq(bytes) => {
            let mut dfa = DFA::new(bytes.len() + 1);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, U8Set::from_bytes(bytes));
            for (index, &byte) in bytes.iter().enumerate() {
                dfa.add_transition(index as u32, byte, index as u32 + 1);
            }
            mark_state_accepting(&mut dfa, bytes.len() as u32);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::U8Class(bytes) => {
            let mut dfa = DFA::new(2);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, *bytes);
            dfa.set_transitions_from_sorted_entries(
                0,
                bytes.iter().map(|byte| (byte, 1)).collect(),
            );
            mark_state_accepting(&mut dfa, 1);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::Epsilon => {
            let mut dfa = DFA::new(1);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, U8Set::empty());
            mark_state_accepting(&mut dfa, 0);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
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

fn compile_product_component_materialized_dfa(expr: &Expr) -> DFA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let direct_started_at = profile_timing.then(Instant::now);
    if let Some((mut dfa, needs_future_recompute)) = compile_product_component_dfa_direct(expr) {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        if needs_future_recompute {
            dfa.recompute_possible_futures();
        }
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] product_component_path path=direct states={} transitions={} direct_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
            );
        }
        dfa
    } else {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        // The generic NFA->DFA path may still carry conservative future-group
        // metadata until the ordinary single-expression compile/minimize pass
        // recomputes it. Product construction consumes that metadata directly,
        // so retain the exact old fallback rather than using the raw DFA here.
        let fallback_started_at = profile_timing.then(Instant::now);
        let dfa = compile_product_component_dfa(expr);
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] product_component_path path=fallback states={} transitions={} direct_attempt_ms={:.3} fallback_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
                fallback_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        dfa
    }
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

struct ProductBuildTrace {
    components: Vec<ProductComponent>,
    state_tuples: Vec<ProductStateTuple>,
    state_by_tuple: FxHashMap<ProductStateTuple, u32>,
    direct_single_visible_group: bool,
}

fn product_component_mapping_dfa(component: &ProductComponent) -> Option<DFA> {
    match component {
        ProductComponent::Materialized(dfa) => Some(dfa.as_ref().clone()),
        ProductComponent::VirtualBoundedRepeat {
            base_dfa,
            min,
            max,
        } => build_bounded_repeat_dfa_from_base(base_dfa, *min as usize, *max as usize),
    }
}

fn exact_combined_dfa_byte_representatives(
    full: &DFA,
    synthesized: &DFA,
    relevant_bytes: &[u8],
) -> Vec<u8> {
    const HASH_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;
    let mut bytes = relevant_bytes.to_vec();
    bytes.sort_unstable();
    bytes.dedup();
    if bytes.len() <= 1 {
        return bytes;
    }

    let mut hashes = FxHashMap::<u64, Vec<u8>>::default();
    for &byte in &bytes {
        let mut hash = 0u64;
        for dfa in [synthesized, full] {
            for state in 0..dfa.num_states() as u32 {
                hash = hash
                    .wrapping_mul(HASH_MULTIPLIER)
                    .wrapping_add(dfa.step(state, byte).unwrap_or(u32::MAX) as u64);
            }
        }
        hashes.entry(hash).or_default().push(byte);
    }

    let columns_equal = |left: u8, right: u8| {
        [synthesized, full].into_iter().all(|dfa| {
            (0..dfa.num_states() as u32)
                .all(|state| dfa.step(state, left) == dfa.step(state, right))
        })
    };
    let mut representatives = Vec::new();
    for candidates in hashes.into_values() {
        let mut local = Vec::<u8>::new();
        for byte in candidates {
            if local
                .iter()
                .copied()
                .any(|representative| columns_equal(byte, representative))
            {
                continue;
            }
            local.push(byte);
        }
        representatives.extend(local);
    }
    representatives.sort_unstable();
    representatives
}

fn exact_kbounded_single_group_state_map(
    full: &DFA,
    synthesized: &DFA,
    depth: usize,
    relevant_bytes: &[u8],
) -> Option<Vec<u32>> {
    if full.num_groups() != synthesized.num_groups() {
        return None;
    }
    let relevant_bytes =
        exact_combined_dfa_byte_representatives(full, synthesized, relevant_bytes);
    let synthesized_states = synthesized.num_states();
    let full_states = full.num_states();
    let label = |dfa: &DFA, state: u32| {
        (
            dfa.finalizers(state).clone(),
            dfa.possible_future_group_ids(state).clone(),
        )
    };
    let synthesized_labels = (0..synthesized_states as u32)
        .map(|state| label(synthesized, state))
        .collect::<Vec<_>>();
    let full_labels = (0..full_states as u32)
        .map(|state| label(full, state))
        .collect::<Vec<_>>();

    let mut label_classes = FxHashMap::<(BitSet, BitSet), u32>::default();
    let mut synthesized_classes = synthesized_labels
        .iter()
        .map(|state_label| {
            let next = label_classes.len() as u32;
            *label_classes.entry(state_label.clone()).or_insert(next)
        })
        .collect::<Vec<_>>();
    let mut full_classes = full_labels
        .iter()
        .map(|state_label| label_classes.get(state_label).copied())
        .collect::<Option<Vec<_>>>()?;

    let mut synthesized_signature = vec![0u32; 1 + relevant_bytes.len()];
    for _ in 0..depth {
        let mut signature_to_class = FxHashMap::<Vec<u32>, u32>::default();
        let mut next_synthesized = vec![0u32; synthesized_states];
        for state in 0..synthesized_states as u32 {
            synthesized_signature[0] = label_classes[&synthesized_labels[state as usize]];
            for (slot, &byte) in relevant_bytes.iter().enumerate() {
                synthesized_signature[slot + 1] = synthesized
                    .step(state, byte)
                    .map_or(u32::MAX, |target| synthesized_classes[target as usize]);
            }
            let next = signature_to_class.len() as u32;
            next_synthesized[state as usize] = *signature_to_class
                .entry(synthesized_signature.clone())
                .or_insert(next);
        }

        let next_full = (0..full_states as u32)
            .into_par_iter()
            .map_init(
                || vec![0u32; 1 + relevant_bytes.len()],
                |signature, state| {
                    signature[0] = label_classes[&full_labels[state as usize]];
                    for (slot, &byte) in relevant_bytes.iter().enumerate() {
                        signature[slot + 1] = full
                            .step(state, byte)
                            .map_or(u32::MAX, |target| full_classes[target as usize]);
                    }
                    signature_to_class.get(signature).copied()
                },
            )
            .collect::<Option<Vec<_>>>()?;
        synthesized_classes = next_synthesized;
        full_classes = next_full;
    }

    let class_count = synthesized_classes
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut synthesized_for_class = vec![u32::MAX; class_count];
    for (state, &class) in synthesized_classes.iter().enumerate() {
        synthesized_for_class[class as usize] = state as u32;
    }
    full_classes
        .into_iter()
        .map(|class| {
            synthesized_for_class
                .get(class as usize)
                .copied()
                .filter(|&state| state != u32::MAX)
        })
        .collect()
}

struct DirectBoundedSuffixShape<'a> {
    prefix: Vec<u8>,
    body: &'a Expr,
    min: usize,
    max: usize,
    suffix: Vec<u8>,
}

struct DirectBoundedSuffixStateMap {
    primary: Vec<u32>,
    prefix_states: usize,
    full_base_states: usize,
    synthesized_base_states: usize,
    full_max: usize,
    synthesized_max: usize,
    min: usize,
    crossed_boundaries: usize,
    base_state_map: Vec<u32>,
    full_suffix_start: usize,
    synthesized_suffix_start: usize,
}

impl DirectBoundedSuffixStateMap {
    fn primary(&self) -> &[u32] {
        &self.primary
    }

    fn visit_candidates(&self, full_state: u32, mut visit: impl FnMut(u32) -> bool) -> bool {
        let full_state = full_state as usize;
        let primary = self.primary[full_state];
        if visit(primary) {
            return true;
        }
        if full_state < self.prefix_states || full_state >= self.full_suffix_start {
            return false;
        }

        let local = full_state - self.prefix_states;
        let completed = local / self.full_base_states;
        let body_state = local % self.full_base_states;
        let distance_to_upper = self.full_max - completed;
        if completed < self.min || distance_to_upper <= self.crossed_boundaries {
            return false;
        }

        let Some(last_interior) = self
            .synthesized_max
            .checked_sub(self.crossed_boundaries.saturating_add(1))
        else {
            return false;
        };
        let mapped_body_state = self.base_state_map[body_state] as usize;
        for mapped_completed in self.min..=last_interior {
            let candidate = (self.prefix_states
                + mapped_completed * self.synthesized_base_states
                + mapped_body_state) as u32;
            if candidate != primary && visit(candidate) {
                return true;
            }
        }
        false
    }
}

enum ProductComponentStateMap {
    Fixed(Vec<u32>),
    Layered(DirectBoundedSuffixStateMap),
}

impl ProductComponentStateMap {
    fn primary(&self) -> &[u32] {
        match self {
            Self::Fixed(mapping) => mapping,
            Self::Layered(mapping) => mapping.primary(),
        }
    }

    fn visit_candidates(&self, full_state: u32, visit: impl FnMut(u32) -> bool) -> bool {
        match self {
            Self::Fixed(mapping) => {
                let mut visit = visit;
                visit(mapping[full_state as usize])
            }
            Self::Layered(mapping) => mapping.visit_candidates(full_state, visit),
        }
    }

    fn is_flexible(&self) -> bool {
        matches!(self, Self::Layered(_))
    }
}

fn direct_bounded_suffix_shape(expr: &Expr) -> Option<DirectBoundedSuffixShape<'_>> {
    let expr = match expr {
        Expr::Shared(inner) => inner.as_ref(),
        expr => expr,
    };
    let Expr::Seq(parts) = expr else {
        return None;
    };
    let mut flat_parts = Vec::new();
    for part in parts {
        match part {
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter()),
                _ => flat_parts.push(part),
            },
            Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter()),
            _ => flat_parts.push(part),
        }
    }
    for repeat_index in 0..flat_parts.len() {
        let repeat = match flat_parts[repeat_index] {
            Expr::Shared(inner) => inner.as_ref(),
            expr => expr,
        };
        let Expr::Repeat {
            expr: body,
            min,
            max: Some(max),
        } = repeat
        else {
            continue;
        };
        let prefix = if repeat_index == 0 {
            Vec::new()
        } else {
            collect_suffix_bytes(
                &flat_parts[..repeat_index]
                    .iter()
                    .map(|expr| (*expr).clone())
                    .collect::<Vec<_>>(),
            )?
        };
        let suffix = collect_suffix_bytes(
            &flat_parts[repeat_index + 1..]
                .iter()
                .map(|expr| (*expr).clone())
                .collect::<Vec<_>>(),
        )?;
        return Some(DirectBoundedSuffixShape {
            prefix,
            body: body.as_ref(),
            min: *min,
            max: *max,
            suffix,
        });
    }
    None
}

/// Exact finite-token-horizon transport for the direct DFA emitted for
/// `Repeat(body, min..=max) + literal_suffix`.
///
/// The direct compiler numbers repeat states by `(completed_copies,
/// body_state)`, followed by the literal suffix chain. Once the minimum has
/// been crossed, completed-copy counts differ observably only through their
/// remaining distance to the upper bound. A token of `K` bytes can cross at
/// most `ceil(K / minimum_body_width) + 1` repetition boundaries. We therefore
/// retain low counts exactly, retain that many upper-bound layers exactly, and
/// map every deeper interior layer to one synthesized interior layer.
fn direct_bounded_suffix_state_map(
    full_expr: &Expr,
    synthesized_expr: &Expr,
    full: &DFA,
    synthesized: &DFA,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<DirectBoundedSuffixStateMap> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let Some(full_shape) = direct_bounded_suffix_shape(full_expr)
    else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=full_shape expr={:?}",
                expr_profile_summary(full_expr),
            );
        }
        return None;
    };
    let Some(synthesized_shape) = direct_bounded_suffix_shape(synthesized_expr)
    else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=synthesized_shape expr={:?}",
                expr_profile_summary(synthesized_expr),
            );
        }
        return None;
    };
    if full_shape.prefix != synthesized_shape.prefix
        || full_shape.min != synthesized_shape.min
        || full_shape.suffix != synthesized_shape.suffix
        || full_shape.max < synthesized_shape.max
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=shape_mismatch full_prefix={} synthesized_prefix={} full_min={} synthesized_min={} full_max={} synthesized_max={} full_suffix={} synthesized_suffix={}",
                full_shape.prefix.len(),
                synthesized_shape.prefix.len(),
                full_shape.min,
                synthesized_shape.min,
                full_shape.max,
                synthesized_shape.max,
                full_shape.suffix.len(),
                synthesized_shape.suffix.len(),
            );
        }
        return None;
    }

    let base = compile_direct_bounded_repeat_base_dfa_unconditionally(full_shape.body)?;
    let synthesized_base =
        compile_direct_bounded_repeat_base_dfa_unconditionally(synthesized_shape.body)?;
    let base_state_map = exact_kbounded_single_group_state_map(
        &base,
        &synthesized_base,
        max_token_len,
        relevant_bytes,
    );
    let Some(base_state_map) = base_state_map else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=base_dfa_map full_base_states={} synthesized_base_states={}",
                base.num_states(),
                synthesized_base.num_states(),
            );
        }
        return None;
    };
    let full_base_states = base.num_states();
    let synthesized_base_states = synthesized_base.num_states();
    let prefix_states = full_shape.prefix.len();
    let suffix_states = full_shape.suffix.len();
    let expected_full_states =
        prefix_states + (full_shape.max + 1).checked_mul(full_base_states)? + suffix_states;
    let expected_synthesized_states =
        prefix_states
            + (synthesized_shape.max + 1).checked_mul(synthesized_base_states)?
            + suffix_states;
    let suffix_overlaps = base.states()[0]
        .transitions
        .get(full_shape.suffix[0])
        .is_some();
    if suffix_states == 0
        || suffix_overlaps
        || full.num_states() != expected_full_states
        || synthesized.num_states() != expected_synthesized_states
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected prefix_states={} base_states={} suffix_states={} full_max={} synthesized_max={} expected_full_states={} actual_full_states={} expected_synthesized_states={} actual_synthesized_states={} suffix_overlaps={}",
                prefix_states,
                full_base_states,
                suffix_states,
                full_shape.max,
                synthesized_shape.max,
                expected_full_states,
                full.num_states(),
                expected_synthesized_states,
                synthesized.num_states(),
                suffix_overlaps,
            );
        }
        return None;
    }

    let minimum_body_width = base.min_match_byte_len()?.max(1);
    let crossed_boundaries = max_token_len
        .div_ceil(minimum_body_width)
        .saturating_add(1);
    // The synthesized interior representative must itself remain farther than
    // one token horizon from the upper boundary.
    if synthesized_shape.max.saturating_sub(synthesized_shape.min) <= crossed_boundaries {
        return None;
    }
    let interior_representative = synthesized_shape
        .max
        .checked_sub(crossed_boundaries.saturating_add(1))?;
    if interior_representative < synthesized_shape.min {
        return None;
    }

    let full_suffix_start = prefix_states + (full_shape.max + 1) * full_base_states;
    let synthesized_suffix_start =
        prefix_states + (synthesized_shape.max + 1) * synthesized_base_states;
    let mut mapping = Vec::with_capacity(full.num_states());
    mapping.extend((0..prefix_states).map(|state| state as u32));
    for completed in 0..=full_shape.max {
        let distance_to_upper = full_shape.max - completed;
        let mapped_completed = if completed < full_shape.min {
            completed
        } else if distance_to_upper <= crossed_boundaries {
            synthesized_shape.max - distance_to_upper
        } else {
            interior_representative
        };
        for &mapped_body_state in &base_state_map {
            mapping.push(
                (prefix_states
                    + mapped_completed * synthesized_base_states
                    + mapped_body_state as usize) as u32,
            );
        }
    }
    for suffix_state in 0..suffix_states {
        debug_assert_eq!(mapping.len(), full_suffix_start + suffix_state);
        mapping.push((synthesized_suffix_start + suffix_state) as u32);
    }
    Some(DirectBoundedSuffixStateMap {
        primary: mapping,
        prefix_states,
        full_base_states,
        synthesized_base_states,
        full_max: full_shape.max,
        synthesized_max: synthesized_shape.max,
        min: full_shape.min,
        crossed_boundaries,
        base_state_map,
        full_suffix_start,
        synthesized_suffix_start,
    })
}

fn add_product_tuple_state(
    dfa: &mut DFA,
    trace: &mut ProductBuildTrace,
    tuple: ProductStateTuple,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> (u32, bool) {
    if let Some(&state) = trace.state_by_tuple.get(&tuple) {
        return (state, false);
    }
    let state = dfa.add_state();
    let (finalizers, futures) = if trace.direct_single_visible_group {
        (
            product_state_single_visible_finalizer(
                &trace.components,
                &tuple,
                exclusions,
                intersections,
            ),
            BitSet::new(1),
        )
    } else {
        product_state_metadata(&trace.components, &tuple)
    };
    dfa.overwrite_state_metadata(state, finalizers, futures);
    trace.state_by_tuple.insert(tuple.clone(), state);
    trace.state_tuples.push(tuple);
    (state, true)
}

fn augment_product_dfa_from_seed_tuples(
    dfa: &mut DFA,
    trace: &mut ProductBuildTrace,
    seeds: &[ProductStateTuple],
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) {
    let (class_map, class_members) = compute_product_equivalence_classes(&trace.components);
    let component_class_transitions =
        build_product_class_transitions(&trace.components, &class_map);
    let component_dead_states = trace
        .components
        .iter()
        .map(ProductComponent::dead_state)
        .collect::<Vec<_>>();
    let mut worklist = VecDeque::<(u32, ProductStateTuple)>::new();
    for tuple in seeds {
        let (state, inserted) = add_product_tuple_state(
            dfa,
            trace,
            tuple.clone(),
            exclusions,
            intersections,
        );
        if inserted {
            worklist.push_back((state, tuple.clone()));
        }
    }

    let mut class_buffers = (0..class_members.len())
        .map(|_| ProductStateTuple::new())
        .collect::<Vec<_>>();
    let mut class_active = vec![false; class_members.len()];
    let mut used_classes = Vec::<usize>::new();
    while let Some((state, tuple)) = worklist.pop_front() {
        for &(component_id, component_state) in &tuple {
            let component = component_id as usize;
            match (
                &trace.components[component],
                &component_class_transitions[component],
            ) {
                (
                    ProductComponent::Materialized(_),
                    ProductComponentClassTransitions::Materialized(transitions),
                ) => {
                    for &(class, target) in &transitions[component_state as usize] {
                        let class = class as usize;
                        if !class_active[class] {
                            class_active[class] = true;
                            used_classes.push(class);
                        }
                        if component_dead_states[component] != Some(target) {
                            class_buffers[class].push((component_id, target));
                        }
                    }
                }
                (
                    ProductComponent::VirtualBoundedRepeat { base_dfa, max, .. },
                    ProductComponentClassTransitions::VirtualBoundedRepeat(transitions),
                ) => {
                    let base_states = base_dfa.num_states() as u32;
                    let copies = component_state / base_states;
                    if copies >= *max {
                        continue;
                    }
                    let base_state = component_state % base_states;
                    if base_dfa.finalizers(base_state).contains(0) {
                        continue;
                    }
                    for &(class, target_base) in &transitions[base_state as usize] {
                        let class = class as usize;
                        if !class_active[class] {
                            class_active[class] = true;
                            used_classes.push(class);
                        }
                        if component_dead_states[component] == Some(target_base) {
                            continue;
                        }
                        let target = if base_dfa.finalizers(target_base).contains(0) {
                            (copies + 1) * base_states
                        } else {
                            copies * base_states + target_base
                        };
                        class_buffers[class].push((component_id, target));
                    }
                }
                _ => unreachable!("component and transition representations must align"),
            }
        }

        let mut byte_transitions = Vec::new();
        for &class in &used_classes {
            let next_tuple = class_buffers[class].clone();
            let (target, inserted) = add_product_tuple_state(
                dfa,
                trace,
                next_tuple.clone(),
                exclusions,
                intersections,
            );
            if inserted {
                worklist.push_back((target, next_tuple));
            }
            byte_transitions.extend(
                class_members[class]
                    .iter()
                    .copied()
                    .map(|byte| (byte, target)),
            );
            class_buffers[class].clear();
            class_active[class] = false;
        }
        used_classes.clear();
        byte_transitions.sort_unstable_by_key(|entry| entry.0);
        dfa.set_transitions_from_sorted_entries(state, byte_transitions);
    }
    if trace.direct_single_visible_group {
        dfa.recompute_possible_futures();
    }
}

pub(crate) struct CompiledTerminalExpressionPair {
    pub(crate) synthesized: Regex,
    pub(crate) full: Regex,
    pub(crate) full_to_synthesized: Vec<u32>,
    pub(crate) synthesized_state_representatives_by_horizon: Vec<(usize, Vec<u32>)>,
    pub(crate) component_quotient_plans: Vec<StructuralComponentQuotientPlan>,
}

fn structural_quotient_horizons(max_token_len: usize) -> Vec<usize> {
    let mut horizons = [4usize, 8, 16, 32, 64]
        .into_iter()
        .filter(|&horizon| horizon < max_token_len)
        .collect::<Vec<_>>();
    horizons.push(max_token_len);
    horizons.sort_unstable();
    horizons.dedup();
    horizons
}

#[derive(Hash, PartialEq, Eq)]
struct ProductStateKBoundedKey {
    component_representatives: ProductStateTuple,
    finalizers: BitSet,
    possible_futures: BitSet,
}

fn product_trace_kbounded_state_representatives(
    trace: &ProductBuildTrace,
    dfa: &DFA,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<Vec<u32>> {
    if trace.state_tuples.len() != dfa.num_states() {
        return None;
    }
    let component_maps = trace
        .components
        .par_iter()
        .map(|component| {
            let dfa = product_component_mapping_dfa(component)?;
            exact_kbounded_single_group_state_map(
                &dfa,
                &dfa,
                max_token_len,
                relevant_bytes,
            )
        })
        .collect::<Option<Vec<_>>>()?;

    let mut classes = FxHashMap::<ProductStateKBoundedKey, u32>::default();
    let mut representatives = Vec::<u32>::new();
    let mut state_representatives = Vec::with_capacity(dfa.num_states());
    for (state, tuple) in trace.state_tuples.iter().enumerate() {
        let mut component_representatives = ProductStateTuple::new();
        for &(component, component_state) in tuple {
            component_representatives.push((
                component,
                component_maps[component as usize][component_state as usize],
            ));
        }
        let key = ProductStateKBoundedKey {
            component_representatives,
            finalizers: dfa.finalizers(state as u32).clone(),
            possible_futures: dfa.possible_future_group_ids(state as u32).clone(),
        };
        let class = match classes.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let class = representatives.len() as u32;
                representatives.push(state as u32);
                entry.insert(class);
                class
            }
        };
        state_representatives.push(representatives[class as usize]);
    }
    Some(state_representatives)
}

pub(crate) fn compile_terminal_expression_pair_with_structural_map(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledTerminalExpressionPair> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile.then(Instant::now);
    let normalized_full = lift_single_nested_intersection(full_expression);
    let normalized_synthesized = lift_single_nested_intersection(synthesized_expression);
    let full_expression = normalized_full.as_ref().unwrap_or(full_expression);
    let synthesized_expression = normalized_synthesized
        .as_ref()
        .unwrap_or(synthesized_expression);
    let full_plan = build_exclusion_compile_plan(std::slice::from_ref(full_expression));
    let synthesized_plan =
        build_exclusion_compile_plan(std::slice::from_ref(synthesized_expression));
    if full_plan.visible_groups != 1
        || synthesized_plan.visible_groups != 1
        || full_plan.compiled_exprs.len() != synthesized_plan.compiled_exprs.len()
        || full_plan.exclusions != synthesized_plan.exclusions
        || full_plan.intersections != synthesized_plan.intersections
        || full_plan.compiled_exprs.is_empty()
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=plan_shape full_visible={} synthesized_visible={} full_components={} synthesized_components={} exclusions_equal={} intersections_equal={}",
                full_plan.visible_groups,
                synthesized_plan.visible_groups,
                full_plan.compiled_exprs.len(),
                synthesized_plan.compiled_exprs.len(),
                full_plan.exclusions == synthesized_plan.exclusions,
                full_plan.intersections == synthesized_plan.intersections,
            );
        }
        return None;
    }
    let exclusions = synthesized_plan.exclusions.clone();
    let intersections = synthesized_plan.intersections.clone();
    let full_component_expressions = full_plan.compiled_exprs.clone();
    let synthesized_component_expressions = synthesized_plan.compiled_exprs.clone();

    // A nested intersection/exclusion can already have been resolved into one
    // visible expression by the grammar lowerer. The previous implementation
    // unnecessarily required at least two hidden product components, even
    // though one deterministic component admits the same exact finite-horizon
    // refinement directly.
    if full_component_expressions.len() == 1 {
        let product_build_started_at = profile.then(Instant::now);
        let (full_dfa, synthesized_dfa) = rayon::join(
            || compile_with_plan(full_plan),
            || compile_with_plan(synthesized_plan),
        );
        let product_build_ms = product_build_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let map_started_at = profile.then(Instant::now);
        let homomorphism = deterministic_component_homomorphism_state_map(
            &full_dfa,
            &synthesized_dfa,
        );
        let used_homomorphism = homomorphism.is_some();
        let full_to_synthesized = homomorphism.or_else(|| {
            exact_kbounded_single_group_state_map(
                &full_dfa,
                &synthesized_dfa,
                max_token_len,
                relevant_bytes,
            )
        });
        let Some(full_to_synthesized) = full_to_synthesized else {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] structural_pair_rejected stage=single_component_map full_states={} synthesized_states={} depth={}",
                    full_dfa.num_states(),
                    synthesized_dfa.num_states(),
                    max_token_len,
                );
            }
            return None;
        };
        let map_ms = map_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let quotient_started_at = profile.then(Instant::now);
        let synthesized_state_representatives_by_horizon =
            structural_quotient_horizons(max_token_len)
                .into_iter()
                .map(|horizon| {
                    exact_kbounded_single_group_state_map(
                        &synthesized_dfa,
                        &synthesized_dfa,
                        horizon,
                        relevant_bytes,
                    )
                    .map(|representatives| (horizon, representatives))
                })
                .collect::<Option<Vec<_>>>()?;
        let quotient_ms = quotient_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_single_component_pair full_states={} synthesized_states={} depth={} path={} build_ms={:.3} map_ms={:.3} quotient_ms={:.3}",
                full_dfa.num_states(),
                synthesized_dfa.num_states(),
                max_token_len,
                if used_homomorphism { "deterministic_homomorphism" } else { "moore" },
                product_build_ms,
                map_ms,
                quotient_ms,
            );
        }
        let component_quotient_plans = vec![StructuralComponentQuotientPlan {
            global_offset: 0,
            source_state_count: synthesized_dfa.num_states(),
            terminal_ids: vec![0],
            horizons: synthesized_state_representatives_by_horizon
                .iter()
                .map(|(horizon, representatives)| StructuralComponentHorizonPlan {
                    horizon: *horizon,
                    tokenizer: Regex {
                        dfa: synthesized_dfa.clone(),
                    }
                    .into_tokenizer(1, None),
                    source_to_analysis: representatives.clone(),
                })
                .collect(),
        }];
        return Some(CompiledTerminalExpressionPair {
            synthesized: Regex {
                dfa: synthesized_dfa,
            },
            full: Regex { dfa: full_dfa },
            full_to_synthesized,
            synthesized_state_representatives_by_horizon,
            component_quotient_plans,
        });
    }

    let product_build_started_at = profile.then(Instant::now);
    let ((full_dfa, full_trace), (mut synthesized_dfa, synthesized_trace)) = rayon::join(
        || compile_with_plan_internal(full_plan, true),
        || compile_with_plan_internal(synthesized_plan, true),
    );
    let product_build_ms = product_build_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let Some(full_trace) = full_trace else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=missing_full_trace full_states={} synthesized_states={}",
                full_dfa.num_states(),
                synthesized_dfa.num_states(),
            );
        }
        return None;
    };
    let Some(mut synthesized_trace) = synthesized_trace else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=missing_synthesized_trace full_states={} synthesized_states={}",
                full_dfa.num_states(),
                synthesized_dfa.num_states(),
            );
        }
        return None;
    };
    if full_trace.components.len() != synthesized_trace.components.len() {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=component_count full_components={} synthesized_components={}",
                full_trace.components.len(),
                synthesized_trace.components.len(),
            );
        }
        return None;
    }

    let component_maps_started_at = profile.then(Instant::now);
    let component_maps = full_trace
        .components
        .par_iter()
        .zip(&synthesized_trace.components)
        .zip(
            full_component_expressions
                .par_iter()
                .zip(&synthesized_component_expressions),
        )
        .enumerate()
        .map(
            |(
                component_index,
                ((full_component, synthesized_component), (full_expr, synthesized_expr)),
            )| {
            let started_at = profile.then(Instant::now);
            let full = product_component_mapping_dfa(full_component)?;
            let synthesized = product_component_mapping_dfa(synthesized_component)?;
            let homomorphism_mapping =
                deterministic_component_homomorphism_state_map(&full, &synthesized);
            let used_homomorphism_mapping = homomorphism_mapping.is_some();
            let layered_mapping = if used_homomorphism_mapping {
                None
            } else {
                direct_bounded_suffix_state_map(
                    full_expr,
                    synthesized_expr,
                    &full,
                    &synthesized,
                    max_token_len,
                    relevant_bytes,
                )
            };
            let used_layered_mapping = layered_mapping.is_some();
            let mapping = if let Some(mapping) = homomorphism_mapping {
                Some(ProductComponentStateMap::Fixed(mapping))
            } else if let Some(mapping) = layered_mapping {
                Some(ProductComponentStateMap::Layered(mapping))
            } else {
                exact_kbounded_single_group_state_map(
                    &full,
                    &synthesized,
                    max_token_len,
                    relevant_bytes,
                )
                .map(ProductComponentStateMap::Fixed)
            };
            if profile {
                let kind = |component: &ProductComponent| match component {
                    ProductComponent::Materialized(_) => "materialized",
                    ProductComponent::VirtualBoundedRepeat { .. } => "virtual_bounded_repeat",
                };
                eprintln!(
                    "[glrmask/profile][tokenizer] structural_component_map component={} full_kind={} synthesized_kind={} full_states={} synthesized_states={} depth={} path={} success={} elapsed_ms={:.3} expr={:?}",
                    component_index,
                    kind(full_component),
                    kind(synthesized_component),
                    full.num_states(),
                    synthesized.num_states(),
                    max_token_len,
                    if used_homomorphism_mapping {
                        "deterministic_homomorphism"
                    } else if used_layered_mapping {
                        "layered_bounded_suffix"
                    } else {
                        "moore"
                    },
                    mapping.is_some(),
                    started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    expr_profile_summary(full_expr),
                );
            }
            mapping
        },
        )
        .collect::<Option<Vec<_>>>();
    let Some(component_maps) = component_maps else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=component_map full_states={} synthesized_states={} components={}",
                full_dfa.num_states(),
                synthesized_dfa.num_states(),
                full_trace.components.len(),
            );
        }
        return None;
    };
    let component_maps_ms = component_maps_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let synthesized_component_dead_states = synthesized_trace
        .components
        .iter()
        .map(ProductComponent::dead_state)
        .collect::<Vec<_>>();

    let tuple_map_started_at = profile.then(Instant::now);
    const DENSE_PRODUCT_LOOKUP_MAX_CELLS: usize = 16 * 1024 * 1024;
    let component_extents = component_maps
        .iter()
        .map(|mapping| {
            mapping
                .primary()
                .iter()
                .copied()
                .max()
                .map_or(1usize, |state| state as usize + 2)
        })
        .collect::<Vec<_>>();
    let dense_two_component_cells = (component_maps.len() == 2)
        .then(|| component_extents[0].checked_mul(component_extents[1]))
        .flatten()
        .filter(|&cells| cells <= DENSE_PRODUCT_LOOKUP_MAX_CELLS);

    let mut full_to_synthesized = if let Some(cells) = dense_two_component_cells {
        let right_extent = component_extents[1];
        let mut state_by_key = vec![u32::MAX; cells];
        for (state, tuple) in synthesized_trace.state_tuples.iter().enumerate() {
            let mut coordinates = [0usize; 2];
            for &(component, component_state) in tuple {
                coordinates[component as usize] = component_state as usize + 1;
            }
            state_by_key[coordinates[0] * right_extent + coordinates[1]] = state as u32;
        }
        full_trace
            .state_tuples
            .par_iter()
            .map(|tuple| {
                let mut coordinates = [0usize; 2];
                let mut full_states = [u32::MAX; 2];
                for &(component_id, full_state) in tuple {
                    let component = component_id as usize;
                    full_states[component] = full_state;
                    let synthesized_state =
                        component_maps[component].primary()[full_state as usize];
                    if synthesized_component_dead_states[component] != Some(synthesized_state) {
                        coordinates[component] = synthesized_state as usize + 1;
                    }
                }
                let primary = state_by_key[coordinates[0] * right_extent + coordinates[1]];
                if primary != u32::MAX {
                    return primary;
                }

                for component in 0..2 {
                    let full_state = full_states[component];
                    if full_state == u32::MAX || !component_maps[component].is_flexible() {
                        continue;
                    }
                    let original_coordinate = coordinates[component];
                    let mut found = u32::MAX;
                    component_maps[component].visit_candidates(full_state, |candidate| {
                        coordinates[component] = if synthesized_component_dead_states[component]
                            == Some(candidate)
                        {
                            0
                        } else {
                            candidate as usize + 1
                        };
                        let state = state_by_key[coordinates[0] * right_extent + coordinates[1]];
                        if state != u32::MAX {
                            found = state;
                            true
                        } else {
                            false
                        }
                    });
                    coordinates[component] = original_coordinate;
                    if found != u32::MAX {
                        return found;
                    }
                }
                u32::MAX
            })
            .collect::<Vec<_>>()
    } else {
        full_trace
            .state_tuples
            .par_iter()
            .map(|tuple| {
                let mut mapped = ProductStateTuple::new();
                for &(component_id, full_state) in tuple {
                    let component = component_id as usize;
                    let synthesized_state =
                        component_maps[component].primary()[full_state as usize];
                    if synthesized_component_dead_states[component] != Some(synthesized_state) {
                        mapped.push((component_id, synthesized_state));
                    }
                }
                synthesized_trace
                    .state_by_tuple
                    .get(&mapped)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect::<Vec<_>>()
    };
    let tuple_map_ms = tuple_map_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let missing_positions = full_to_synthesized
        .iter()
        .enumerate()
        .filter_map(|(position, &state)| (state == u32::MAX).then_some(position))
        .collect::<Vec<_>>();
    let missing_before = missing_positions.len();
    let missing_tuples = missing_positions
        .par_iter()
        .map(|&position| {
            let mut mapped = ProductStateTuple::new();
            for &(component_id, full_state) in &full_trace.state_tuples[position] {
                let component = component_id as usize;
                let synthesized_state = component_maps[component].primary()[full_state as usize];
                if synthesized_component_dead_states[component] != Some(synthesized_state) {
                    mapped.push((component_id, synthesized_state));
                }
            }
            mapped
        })
        .collect::<Vec<_>>();
    let states_before_augment = synthesized_dfa.num_states();
    let augment_started_at = profile.then(Instant::now);
    augment_product_dfa_from_seed_tuples(
        &mut synthesized_dfa,
        &mut synthesized_trace,
        &missing_tuples,
        &exclusions,
        &intersections,
    );
    let augment_ms = augment_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let lookup_started_at = profile.then(Instant::now);
    for (&position, tuple) in missing_positions.iter().zip(&missing_tuples) {
        full_to_synthesized[position] = *synthesized_trace.state_by_tuple.get(tuple)?;
    }
    let lookup_ms = lookup_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let quotient_started_at = profile.then(Instant::now);
    let synthesized_state_representatives_by_horizon = structural_quotient_horizons(max_token_len)
        .into_iter()
        .map(|horizon| {
            product_trace_kbounded_state_representatives(
                &synthesized_trace,
                &synthesized_dfa,
                horizon,
                relevant_bytes,
            )
            .map(|representatives| (horizon, representatives))
        })
        .collect::<Option<Vec<_>>>()?;
    let quotient_ms = quotient_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let synthesized_kbounded_reps = synthesized_state_representatives_by_horizon
        .last()
        .map(|(_, representatives)| {
            representatives.iter().copied().collect::<FxHashSet<_>>().len()
        })
        .unwrap_or(0);
    if profile {
        let counts = synthesized_state_representatives_by_horizon
            .iter()
            .map(|(horizon, representatives)| {
                format!(
                    "{}:{}",
                    horizon,
                    representatives.iter().copied().collect::<FxHashSet<_>>().len()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[glrmask/profile][tokenizer] structural_quotient_horizons states={} reps={}",
            synthesized_dfa.num_states(),
            counts,
        );
    }

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair full_states={} synthesized_states_before={} synthesized_states_after={} synthesized_kbounded_reps={} mapped_tuples={} missing_before={} product_build_ms={:.3} component_maps_ms={:.3} tuple_map_ms={:.3} augment_ms={:.3} lookup_ms={:.3} quotient_ms={:.3} total_ms={:.3}",
            full_dfa.num_states(),
            states_before_augment,
            synthesized_dfa.num_states(),
            synthesized_kbounded_reps,
            full_to_synthesized.len(),
            missing_before,
            product_build_ms,
            component_maps_ms,
            tuple_map_ms,
            augment_ms,
            lookup_ms,
            quotient_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let component_quotient_plans = vec![StructuralComponentQuotientPlan {
        global_offset: 0,
        source_state_count: synthesized_dfa.num_states(),
        terminal_ids: vec![0],
        horizons: synthesized_state_representatives_by_horizon
            .iter()
            .map(|(horizon, representatives)| StructuralComponentHorizonPlan {
                horizon: *horizon,
                tokenizer: Regex {
                    dfa: synthesized_dfa.clone(),
                }
                .into_tokenizer(1, None),
                source_to_analysis: representatives.clone(),
            })
            .collect(),
    }];
    Some(CompiledTerminalExpressionPair {
        synthesized: Regex {
            dfa: synthesized_dfa,
        },
        full: Regex { dfa: full_dfa },
        full_to_synthesized,
        synthesized_state_representatives_by_horizon,
        component_quotient_plans,
    })
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

            ProductComponent::Materialized(Arc::new(compile_product_component_materialized_dfa(
                expr,
            )))
        }
        _ => ProductComponent::Materialized(Arc::new(
            compile_product_component_materialized_dfa(expr),
        )),
    }
}

fn deterministic_component_homomorphism_state_map(
    full: &DFA,
    synthesized: &DFA,
) -> Option<Vec<u32>> {
    if full.num_groups() != synthesized.num_groups() {
        return None;
    }
    let mut mapping = vec![u32::MAX; full.num_states()];
    let mut worklist = VecDeque::from([(0u32, 0u32)]);
    mapping[0] = 0;

    while let Some((full_state, synthesized_state)) = worklist.pop_front() {
        if full.finalizers(full_state) != synthesized.finalizers(synthesized_state)
            || full.possible_future_group_ids(full_state)
                != synthesized.possible_future_group_ids(synthesized_state)
        {
            return None;
        }
        for byte in 0u16..=255 {
            let full_target = full.step(full_state, byte as u8);
            let synthesized_target = synthesized.step(synthesized_state, byte as u8);
            match (full_target, synthesized_target) {
                (None, None) => {}
                (Some(full_target), Some(synthesized_target)) => {
                    let slot = &mut mapping[full_target as usize];
                    if *slot == u32::MAX {
                        *slot = synthesized_target;
                        worklist.push_back((full_target, synthesized_target));
                    } else if *slot != synthesized_target {
                        return None;
                    }
                }
                _ => return None,
            }
        }
    }
    mapping.iter().all(|&state| state != u32::MAX).then_some(mapping)
}

#[derive(Debug)]
struct ProductComponentCompileProfile {
    first_group_index: usize,
    uses: usize,
    compile_ms: f64,
    states: usize,
    transitions: usize,
}

fn compile_product_components_profiled(
    exprs: &[Expr],
    profile_detail: bool,
) -> (
    Vec<ProductComponent>,
    usize,
    Option<Vec<ProductComponentCompileProfile>>,
) {
    let mut unique_exprs = Vec::<&Expr>::new();
    let mut unique_first_group_indices = Vec::<usize>::new();
    let mut component_indices = Vec::with_capacity(exprs.len());
    let mut index_by_expr = FxHashMap::<&Expr, usize>::default();

    for (group_index, expr) in exprs.iter().enumerate() {
        let expr = unwrap_shared(expr);
        let index = if let Some(&index) = index_by_expr.get(expr) {
            index
        } else {
            let index = unique_exprs.len();
            unique_exprs.push(expr);
            unique_first_group_indices.push(group_index);
            index_by_expr.insert(expr, index);
            index
        };
        component_indices.push(index);
    }

    let compiled: Vec<(ProductComponent, Option<f64>)> = unique_exprs
        .par_iter()
        .map(|expr| {
            if profile_detail {
                let started_at = Instant::now();
                let component = compile_product_component(expr);
                (
                    component,
                    Some(started_at.elapsed().as_secs_f64() * 1000.0),
                )
            } else {
                (compile_product_component(expr), None)
            }
        })
        .collect();
    let (unique_components, compile_times): (Vec<_>, Vec<_>) = compiled.into_iter().unzip();
    let cache_hits = exprs.len() - unique_components.len();
    let profiles = profile_detail.then(|| {
        let mut uses = vec![0usize; unique_components.len()];
        for &index in &component_indices {
            uses[index] += 1;
        }
        unique_components
            .iter()
            .zip(&compile_times)
            .enumerate()
            .map(|(index, (component, compile_ms))| ProductComponentCompileProfile {
                first_group_index: unique_first_group_indices[index],
                uses: uses[index],
                compile_ms: compile_ms.unwrap_or_default(),
                states: component.partition_dfa().num_states(),
                transitions: dfa_transition_count(component.partition_dfa()),
            })
            .collect::<Vec<_>>()
    });
    let components = component_indices
        .into_iter()
        .map(|index| unique_components[index].clone())
        .collect();
    (components, cache_hits, profiles)
}

fn compile_product_components(exprs: &[Expr]) -> (Vec<ProductComponent>, usize) {
    let (components, cache_hits, _) = compile_product_components_profiled(exprs, false);
    (components, cache_hits)
}

fn build_product_dfa(
    exprs: &[Expr],
    profile_labels: Option<&[ProductComponentProfileLabel]>,
    visible_groups: usize,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
    capture_trace: bool,
) -> (DFA, bool, Option<ProductBuildTrace>) {
    let profile_trace = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some();
    let profile_detail = profile_trace
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_started_at = Instant::now();
    let component_compile_started_at = Instant::now();
    let (components, component_cache_hits, component_profiles) =
        compile_product_components_profiled(exprs, profile_detail);
    let component_compile_ms = profile_timing
        .then(|| component_compile_started_at.elapsed().as_secs_f64() * 1000.0);
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] product_component_cache groups={} unique_components={} cache_hits={}",
            exprs.len(),
            exprs.len() - component_cache_hits,
            component_cache_hits,
        );
    }
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] product_components groups={} unique_components={} cache_hits={} compile_components_ms={:.3}",
            components.len(),
            components.len() - component_cache_hits,
            component_cache_hits,
            profile_started_at.elapsed().as_secs_f64() * 1000.0
        );
        let mut ranked = component_profiles
            .as_ref()
            .expect("detail profiling records unique component profiles")
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|(_, left), (_, right)| {
            right.compile_ms.total_cmp(&left.compile_ms)
        });
        let report_count = if profile_trace {
            ranked.len()
        } else {
            ranked.len().min(20)
        };
        for (rank, (unique_index, component)) in ranked.into_iter().take(report_count).enumerate() {
            let index = component.first_group_index;
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
                "[glrmask/profile][tokenizer/component-rank] rank={} unique_index={} first_group_index={} uses={} states={} transitions={} compile_ms={:.3}{}",
                rank + 1,
                unique_index,
                index,
                component.uses,
                component.states,
                component.transitions,
                component.compile_ms,
                label,
            );
        }
        let omitted = component_profiles
            .as_ref()
            .map_or(0, |profiles| profiles.len().saturating_sub(report_count));
        if omitted > 0 {
            eprintln!(
                "[glrmask/profile][tokenizer/component-rank] omitted={} set GLRMASK_PROFILE_TOKENIZER_TRACE=1 for exhaustive component output",
                omitted,
            );
        }
    }
    let num_groups = components.len();
    let direct_single_visible_group = visible_groups == 1
        && num_groups > 1
        && (!exclusions.is_empty() || !intersections.is_empty());
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
    dfa.ensure_group_capacity(if direct_single_visible_group {
        1
    } else {
        num_groups
    });

    assert!(num_groups <= u32::MAX as usize, "too many product DFA groups");
    let mut start_tuple = ProductStateTuple::with_capacity(num_groups);
    for group_id in 0..num_groups {
        start_tuple.push((group_id as u32, 0u32));
    }
    let (start_finalizers, start_future) = if direct_single_visible_group {
        (
            product_state_single_visible_finalizer(
                &components,
                &start_tuple,
                exclusions,
                intersections,
            ),
            BitSet::new(1),
        )
    } else {
        product_state_metadata(&components, &start_tuple)
    };
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
    let mut growth_recorder = profile_trace.then(|| ProductGrowthRecorder::new(num_groups));
    let mut state_tuples = capture_trace.then(|| vec![start_tuple.clone()]);
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
                let (finalizers, future) = if direct_single_visible_group {
                    (
                        product_state_single_visible_finalizer(
                            &components,
                            next_tuple,
                            exclusions,
                            intersections,
                        ),
                        BitSet::new(1),
                    )
                } else {
                    product_state_metadata(&components, next_tuple)
                };
                dfa.overwrite_state_metadata(new_state, finalizers, future);
                state_map.insert(next_tuple.clone(), new_state);
                if let Some(state_tuples) = state_tuples.as_mut() {
                    debug_assert_eq!(state_tuples.len(), new_state as usize);
                    state_tuples.push(next_tuple.clone());
                }
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

    let direct_future_started_at = Instant::now();
    if direct_single_visible_group {
        set_single_group_futures_from_class_graph(&mut dfa, &pending_class_transitions);
    }
    let direct_future_ms = profile_timing
        .then(|| direct_future_started_at.elapsed().as_secs_f64() * 1000.0);

    if profile_trace {
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
            "[glrmask/profile][tokenizer] product_phases groups={} classes={} direct_single_visible_group={} component_compile_ms={:.3} equivalence_classes_ms={:.3} class_transition_ms={:.3} product_state_expand_ms={:.3} direct_future_ms={:.3} byte_expand_ms={:.3}",
            components.len(),
            num_classes,
            direct_single_visible_group,
            component_compile_ms.unwrap_or_default(),
            equivalence_classes_ms.unwrap_or_default(),
            class_transition_ms.unwrap_or_default(),
            product_state_expand_ms.unwrap_or_default(),
            direct_future_ms.unwrap_or_default(),
            byte_expand_ms.unwrap_or_default(),
        );
    }

    let trace = state_tuples.map(|state_tuples| ProductBuildTrace {
        components,
        state_tuples,
        state_by_tuple: state_map,
        direct_single_visible_group,
    });
    (dfa, direct_single_visible_group, trace)
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
    use super::super::{Lexer, DFA};
    use super::{
        build_regex,
        build_regex_monolithic,
        build_regex_partitioned_with_adaptive,
        try_product_union_components,
    };
    use super::{compile_product_component_dfa, compile_product_component_dfa_direct};
    use super::factor_regex_expr;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::regex::parse_regex;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::ds::u8set::U8Set;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeSet;
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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

    fn dfa_state_observation(
        dfa: &super::DFA,
        mut state: u32,
        input: &[u8],
    ) -> (bool, bool, bool) {
        for &byte in input {
            let Some(next) = dfa.step(state, byte) else {
                return (true, false, false);
            };
            state = next;
        }
        (
            false,
            dfa.finalizers(state).contains(0),
            dfa.possible_future_group_ids(state).contains(0),
        )
    }

    #[test]
    fn layered_bounded_suffix_transport_matches_all_short_observations() {
        const HORIZON: usize = 4;
        let alphabet = b"[]abx";
        let inputs = enumerate_inputs(alphabet, HORIZON);
        let bodies = [
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::U8Seq(b"ab".to_vec()),
        ];

        for body in bodies {
            let make_expr = |max| {
                factor_regex_expr(Expr::Seq(vec![
                    Expr::U8Seq(b"[".to_vec()),
                    Expr::Repeat {
                        expr: Box::new(body.clone()),
                        min: 2,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"]".to_vec()),
                ]))
            };
            let full_expr = make_expr(24);
            let synthesized_expr = make_expr(16);
            let full = super::compile_product_component_materialized_dfa(&full_expr);
            let synthesized =
                super::compile_product_component_materialized_dfa(&synthesized_expr);
            let mapping = super::direct_bounded_suffix_state_map(
                &full_expr,
                &synthesized_expr,
                &full,
                &synthesized,
                HORIZON,
                alphabet,
            )
            .expect("direct layered transport");
            assert_eq!(mapping.primary().len(), full.num_states());

            for full_state in 0..full.num_states() as u32 {
                let synthesized_state = mapping.primary()[full_state as usize];
                for input in &inputs {
                    assert_eq!(
                        dfa_state_observation(&full, full_state, input),
                        dfa_state_observation(&synthesized, synthesized_state, input),
                        "body={body:?} full_state={full_state} synthesized_state={synthesized_state} input={input:?}",
                    );
                }
            }
        }
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

    fn random_group_free_expr(rng: &mut StdRng, depth: usize) -> Expr {
        let atom = |rng: &mut StdRng| match rng.gen_range(0..4) {
            0 => Expr::U8Seq(vec![b'a' + rng.gen_range(0..3)]),
            1 => Expr::U8Seq(
                (0..rng.gen_range(1..=3))
                    .map(|_| b'a' + rng.gen_range(0..3))
                    .collect(),
            ),
            2 => Expr::U8Class(U8Set::from_bytes(match rng.gen_range(0..3) {
                0 => b"ab",
                1 => b"bc",
                _ => b"abc",
            })),
            _ => Expr::Epsilon,
        };

        if depth == 0 {
            return atom(rng);
        }
        match rng.gen_range(0..7) {
            0..=2 => atom(rng),
            3 => Expr::Choice(vec![
                random_group_free_expr(rng, depth - 1),
                random_group_free_expr(rng, depth - 1),
            ]),
            4 => Expr::Seq(vec![
                random_group_free_expr(rng, depth - 1),
                random_group_free_expr(rng, depth - 1),
            ]),
            5 => Expr::Repeat {
                expr: Box::new(random_group_free_expr(rng, depth - 1)),
                min: rng.gen_range(0..=1),
                max: Some(rng.gen_range(1..=3)),
            },
            _ => Expr::Shared(Arc::new(random_group_free_expr(rng, depth - 1))),
        }
    }

    fn expr_contains_dfa(expr: &Expr) -> bool {
        match expr {
            Expr::Dfa(_) => true,
            Expr::Exclude { expr, exclude } => {
                expr_contains_dfa(expr) || expr_contains_dfa(exclude)
            }
            Expr::Intersect { expr, intersect } => {
                expr_contains_dfa(expr) || expr_contains_dfa(intersect)
            }
            Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_contains_dfa),
            Expr::Repeat { expr, .. } => expr_contains_dfa(expr),
            Expr::Shared(expr) => expr_contains_dfa(expr),
            Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
        }
    }

    fn tokenizer_from_partitioned_exprs(exprs: &[Expr]) -> Tokenizer {
        let partitions = (0..exprs.len() as u32).collect::<Vec<_>>();
        build_regex_partitioned_with_adaptive(exprs, &partitions, false).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.to_vec().into_boxed_slice())),
        )
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
            let partitioned = build_regex_partitioned_with_adaptive(
                &exprs,
                &partitions,
                false,
            )
            .into_tokenizer(
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
                let partitioned = build_regex_partitioned_with_adaptive(
                    &exprs,
                    &partitions,
                    false,
                )
                .into_tokenizer(
                    exprs.len() as u32,
                    Some(Arc::from(exprs.clone().into_boxed_slice())),
                );
                let adaptive = build_regex_partitioned_with_adaptive(
                    &exprs,
                    &partitions,
                    true,
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
    fn repeated_subexpression_dfa_materialization_preserves_observations_exhaustively() {
        let repeated = Expr::Seq(vec![
            Expr::Choice(vec![byte_expr(b'a'), byte_expr(b'b')]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 1,
                max: Some(2),
            },
        ]);
        let exprs = vec![
            Expr::Seq(vec![byte_expr(b'a'), repeated.clone(), byte_expr(b'c')]),
            Expr::Choice(vec![repeated.clone(), Expr::U8Seq(b"cc".to_vec())]),
            Expr::Repeat {
                expr: Box::new(repeated.clone()),
                min: 0,
                max: Some(2),
            },
            Expr::Repeat {
                expr: Box::new(repeated.clone()),
                min: 1,
                max: None,
            },
            Expr::Exclude {
                expr: Box::new(Expr::Choice(vec![repeated.clone(), byte_expr(b'c')])),
                exclude: Box::new(byte_expr(b'c')),
            },
            Expr::Intersect {
                expr: Box::new(Expr::Choice(vec![repeated.clone(), Expr::U8Seq(b"ab".to_vec())])),
                intersect: Box::new(Expr::Choice(vec![repeated, byte_expr(b'b')])),
            },
        ];
        let rewritten = super::materialize_repeated_subexpression_dfas_with_limits(&exprs, 2, 2)
            .expect("the repeated subtree should be materialized");
        assert!(rewritten.iter().any(expr_contains_dfa));

        let original = tokenizer_from_partitioned_exprs(&exprs);
        let materialized = tokenizer_from_partitioned_exprs(&rewritten);
        for input in enumerate_inputs(b"abc", 6) {
            assert_eq!(
                tokenizer_observation(&materialized, &input),
                tokenizer_observation(&original, &input),
                "materialized repeated subtree changed tokenizer observation for input={input:?}",
            );
        }
    }

    #[test]
    fn repeated_subexpression_dfa_materialization_seeded_differential() {
        let mut rng = StdRng::seed_from_u64(0xC5E0_2026_0718);
        let inputs = enumerate_inputs(b"abc", 4);

        for case in 0..32 {
            let repeated = Expr::Seq(vec![
                random_group_free_expr(&mut rng, 2),
                random_group_free_expr(&mut rng, 2),
            ]);
            let exprs = vec![
                Expr::Seq(vec![byte_expr(b'a'), repeated.clone(), byte_expr(b'b')]),
                Expr::Choice(vec![repeated.clone(), random_group_free_expr(&mut rng, 1)]),
                Expr::Repeat {
                    expr: Box::new(repeated.clone()),
                    min: rng.gen_range(0..=1),
                    max: Some(rng.gen_range(1..=3)),
                },
                Expr::Shared(Arc::new(repeated)),
            ];
            let rewritten =
                super::materialize_repeated_subexpression_dfas_with_limits(&exprs, 2, 2)
                    .expect("the seeded repeated subtree should be materialized");
            let original = tokenizer_from_partitioned_exprs(&exprs);
            let materialized = tokenizer_from_partitioned_exprs(&rewritten);

            for input in &inputs {
                assert_eq!(
                    tokenizer_observation(&materialized, input),
                    tokenizer_observation(&original, input),
                    "seeded CSE mismatch case={case} input={input:?} exprs={exprs:?} rewritten={rewritten:?}",
                );
            }
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
    fn factors_contains_literal_choice_with_common_quoted_string_shell() {
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
            singleton_epsilon_closures: std::sync::OnceLock::new(),
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
        let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
            .into_tokenizer(
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
        let regex = super::build_regex_partitioned_with_adaptive(
            &expressions,
            &[7, 7, 9],
            false,
        );
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
    fn protected_residual_components_survive_adaptive_determinization() {
        let expressions = vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"ordinary-a".to_vec()),
            Expr::U8Seq(b"ordinary-b".to_vec()),
        ];
        let isolation = [Some(100), Some(101), None, None];
        let components = super::compile_partition_components(
            &expressions,
            None,
            &[0, 1, 2, 3],
            Some(&isolation),
        );

        let output = super::adaptively_determinize_components_with_limits(
            components,
            32_768,
            1_000,
            1_000,
            Some(1),
        );

        for protected_terminal in [0usize, 1] {
            let component = output
                .iter()
                .find(|component| component.terminal_ids.contains(&protected_terminal))
                .expect("protected terminal must remain present");
            assert!(component.protected_residual);
            assert_eq!(component.terminal_ids, vec![protected_terminal]);
        }
    }

    #[test]
    fn protected_identical_terminals_keep_independent_live_states() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(4),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(4),
            },
        ];
        let regex = super::build_regex_partitioned_with_adaptive_and_residual_isolation(
            &expressions,
            &[0, 1],
            &[Some(200), Some(201)],
            true,
        );
        assert_eq!(
            regex.dfa.states()[0].epsilon_transitions.len(),
            2,
            "protected coordinates must not collapse into one adaptive product",
        );
        let tokenizer = regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert_eq!(result.end_state.len(), 2);
        assert_eq!(
            result
                .matches
                .iter()
                .map(|matched| matched.id)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1]),
        );
    }

    #[test]
    fn shared_nested_group_ops_are_initialized_before_parallel_partition_compile() {
        let shared_nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"))),
            exclude: Box::new(Expr::U8Seq(b"x".to_vec())),
        };
        let expressions = (0..64u8)
            .map(|suffix| {
                Expr::Seq(vec![
                    shared_nested.clone(),
                    Expr::U8Seq(vec![b'0' + suffix % 10]),
                ])
            })
            .collect::<Vec<_>>();
        let partitions = (0..expressions.len() as u32).collect::<Vec<_>>();
        let mut grouped = std::collections::BTreeMap::<u32, Vec<usize>>::new();
        for (terminal, &partition) in partitions.iter().enumerate() {
            grouped.entry(partition).or_default().push(terminal);
        }

        let shared = super::shared_duplicate_nested_group_op_cache(&expressions, &grouped)
            .expect("the repeated nested exclusion should use the shared cache");
        super::prewarm_shared_duplicate_nested_group_ops(&shared);
        assert!(shared.all_entries_initialized());

        let components =
            super::compile_partition_components(&expressions, None, &partitions, None);
        assert_eq!(components.len(), expressions.len());
    }

    #[test]
    #[should_panic(
        expected = "shared nested group-op cache must be prewarmed before parallel compilation"
    )]
    fn shared_nested_group_ops_cannot_initialize_from_partition_workers() {
        let shared_nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"))),
            exclude: Box::new(Expr::U8Seq(b"x".to_vec())),
        };
        let expressions = (0..64u8)
            .map(|suffix| {
                Expr::Seq(vec![
                    shared_nested.clone(),
                    Expr::U8Seq(vec![b'0' + suffix % 10]),
                ])
            })
            .collect::<Vec<_>>();
        let partitions = (0..expressions.len() as u32).collect::<Vec<_>>();
        let mut grouped = std::collections::BTreeMap::<u32, Vec<usize>>::new();
        for (terminal, &partition) in partitions.iter().enumerate() {
            grouped.entry(partition).or_default().push(terminal);
        }
        let shared = super::shared_duplicate_nested_group_op_cache(&expressions, &grouped)
            .expect("the repeated nested exclusion should use the shared cache");

        let _ = super::compile_terminal_ids_with_shared_duplicate_cache(
            &expressions,
            None,
            &[0],
            Some(&shared),
        );
    }

    #[test]
    fn partitioned_union_transports_exact_possible_futures_without_recompute() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Epsilon,
        ];
        let regex = build_regex_partitioned_with_adaptive(
            &expressions,
            &[7, 7, 9, 11],
            false,
        );
        let exact = regex.dfa;
        let mut recomputed = exact.clone();
        recomputed.recompute_possible_futures();

        assert_eq!(
            exact, recomputed,
            "transported component futures and epsilon-root union must match the generic fixpoint",
        );
    }

    #[test]
    fn bounded_product_trial_stops_before_cross_pattern_blowup() {
        let expressions = vec![
            parse_regex(r"\w+_(\w_)?\d+", false),
            parse_regex(r"(\w|-){12}", false),
        ];
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1], None);
        assert!(
            try_product_union_components(&components, 32, usize::MAX, None).is_none(),
            "the bounded trial unexpectedly completed within 32 product states",
        );
    }

    #[test]
    fn adaptive_transition_growth_limit_is_inclusive() {
        assert!(super::adaptive_transition_growth_is_acceptable(100, 600, 600));
        assert!(!super::adaptive_transition_growth_is_acceptable(100, 601, 600));
    }

    #[test]
    fn adaptive_transition_growth_rejection_keeps_partition_components() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1, 2], None);
        let original_terminal_ids = components
            .iter()
            .map(|component| component.terminal_ids.clone())
            .collect::<Vec<_>>();

        let retained = super::adaptively_determinize_components_with_limits(
            components,
            32_768,
            100,
            1,
            None,
        );

        assert_eq!(retained.len(), original_terminal_ids.len());
        assert_eq!(
            retained
                .iter()
                .map(|component| component.terminal_ids.clone())
                .collect::<Vec<_>>(),
            original_terminal_ids,
        );
    }

    #[test]
    fn adaptive_policy_does_not_change_per_partition_compilation() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"c".to_vec()),
        ];
        let partitions = [7, 7, 9, 11];
        let components =
            super::compile_partition_components(&expressions, None, &partitions, None);
        let expected_terminal_ids = [vec![0, 1], vec![2], vec![3]];

        assert_eq!(components.len(), expected_terminal_ids.len());
        for (component, expected_ids) in components.iter().zip(expected_terminal_ids) {
            assert_eq!(component.terminal_ids, expected_ids);
            let expected = super::compile_terminal_ids(&expressions, None, &expected_ids);
            assert_eq!(component.dfa, expected);
        }
    }

    #[test]
    fn adaptive_prefix_depth_preserves_partitioned_semantics() {
        let expressions = vec![
            Expr::U8Seq(b"abcd".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(8),
            },
        ];
        let baseline = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
            .into_tokenizer(
                expressions.len() as u32,
                Some(Arc::from(expressions.clone().into_boxed_slice())),
            );
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1], None);
        let prefix_dfa = try_product_union_components(&components, 32_768, usize::MAX, Some(1))
            .expect("depth-one adaptive product should fit");
        assert!(prefix_dfa.states()[0].epsilon_transitions.is_empty());
        assert!(
            prefix_dfa.states()[0]
                .transitions
                .iter()
                .any(|(_, &target)| !prefix_dfa.states()[target as usize].epsilon_transitions.is_empty()),
            "the depth-one product should resume exact component states at its frontier",
        );
        let adaptive = super::Regex { dfa: prefix_dfa }.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b"abcdx", 8) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&baseline, &input),
                "depth-one adaptive prefix differed for input {input:?}",
            );
        }
    }

    #[test]
    fn adaptive_final_nfa_determinization_preserves_partitioned_semantics() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let singleton = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        let adaptive = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            true,
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
    fn direct_trivial_product_components_match_generic_compilation() {
        let expressions = [
            Expr::U8Seq(b"literal".to_vec()),
            Expr::U8Seq(Vec::new()),
            Expr::U8Class(U8Set::from_bytes(b"abc")),
            Expr::Epsilon,
        ];

        for expr in expressions {
            let direct = compile_product_component_dfa_direct(&expr)
                .expect("trivial expression must have a direct DFA")
                .0;
            let generic = compile_product_component_dfa(&expr);
            assert_eq!(
                direct.group_id_to_u8set(0),
                generic.group_id_to_u8set(0),
                "group byte set differed for {expr:?}",
            );

            let mut inputs = vec![Vec::new()];
            match &expr {
                Expr::U8Seq(bytes) => {
                    for prefix_len in 0..=bytes.len() {
                        let prefix = bytes[..prefix_len].to_vec();
                        inputs.push(prefix.clone());
                        for byte in 0..=u8::MAX {
                            let mut deviated = prefix.clone();
                            deviated.push(byte);
                            inputs.push(deviated);
                        }
                    }
                }
                Expr::U8Class(_) => {
                    inputs.extend((0..=u8::MAX).map(|byte| vec![byte]));
                    inputs.extend((0..=u8::MAX).map(|byte| vec![byte, byte]));
                }
                Expr::Epsilon => inputs.push(vec![0]),
                _ => unreachable!(),
            }

            let observe = |dfa: DFA, input: &[u8]| {
                let tokenizer = Tokenizer::from_parts(dfa, 1, None);
                let states = tokenizer.run(input);
                let mut matched = BTreeSet::new();
                let mut future = BTreeSet::new();
                for state in states {
                    matched.extend(tokenizer.matched_terminals_iter(state));
                    future.extend(tokenizer.possible_future_terminals_iter(state));
                }
                (matched, future)
            };
            for input in inputs {
                assert_eq!(
                    observe(direct.clone(), &input),
                    observe(generic.clone(), &input),
                    "direct DFA behavior differed for {expr:?} on {input:?}",
                );
            }
        }
    }

    #[test]
    fn zero_min_repeat_suffix_dominance_matches_generic_ambiguous_boundaries() {
        let expressions = [
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("a+", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("a+", false),
            ]),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("a+b+", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("a+", false),
            ]),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("(?:ab|a)", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("b+", false),
            ]),
        ];

        for expression in expressions {
            let direct_dfa = compile_product_component_dfa_direct(&expression)
                .expect("zero-minimum bounded repeat with non-nullable suffix must compile directly")
                .0;
            let direct = super::Regex { dfa: direct_dfa }.into_tokenizer(
                1,
                Some(Arc::from(vec![expression.clone()].into_boxed_slice())),
            );

            let mut nfa = super::build_regex_nfa(std::slice::from_ref(&expression));
            nfa.condense_epsilon_sccs();
            let generic = super::Regex {
                dfa: nfa.to_dfa().minimize(),
            }
            .into_tokenizer(
                1,
                Some(Arc::from(vec![expression.clone()].into_boxed_slice())),
            );

            for input in enumerate_inputs(b"abx", 8) {
                assert_eq!(
                    tokenizer_observation(&direct, &input),
                    tokenizer_observation(&generic, &input),
                    "dominance quotient differed from generic determinization for expression {expression:?}, input {input:?}",
                );
            }
        }
    }

    #[test]
    fn generic_product_component_fallback_preserves_strict_future_metadata() {
        let expr = Expr::Repeat {
            expr: Box::new(Expr::Epsilon),
            min: 0,
            max: None,
        };
        let product = super::compile_product_component(&expr);
        let product = product.partition_dfa();
        let generic = compile_product_component_dfa(&expr);

        assert_eq!(product.finalizers(0), generic.finalizers(0));
        assert_eq!(
            product.possible_future_group_ids(0),
            generic.possible_future_group_ids(0),
        );
        assert!(product.finalizers(0).contains(0));
        assert!(
            !product.possible_future_group_ids(0).contains(0),
            "an epsilon-only terminal is final now but is not reachable after another byte",
        );
    }

    #[test]
    fn adaptive_final_representation_matches_monolithic_for_ignore_and_repeated_terminals() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ];
        let monolithic = build_regex_monolithic(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        let adaptive = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2, 3],
            true,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b" abc", 6) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&monolithic, &input),
                "adaptive tokenizer differed for input {input:?}",
            );
        }
    }

    #[test]
    fn epsilon_partitioned_tokenizer_round_trips_through_serde() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1],
            false,
        )
        .into_tokenizer(
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
