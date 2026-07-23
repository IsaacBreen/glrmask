use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ds::{bitset::BitSet, u8set::U8Set};
use crate::Vocab;
use crate::compiler::stages::id_map_and_terminal_dwa::synthetic_state_map::{
    CertifiedVocabularyExactStateCandidates, certify_vocabulary_exact_state_candidates,
};

use super::ast::Expr;
use super::tokenizer::{CompressedTransitionSegment, Tokenizer};
use super::dfa::DFA;
use super::nfa::NFA;

type ProductStateTuple = SmallVec<[(u32, u32); 12]>;

const DEAD_REPEAT_TRANSLATION_STATE: u32 = u32::MAX;

#[derive(Clone, PartialEq, Eq, Hash)]
struct RepeatTranslationState(Box<[u32]>);

#[derive(Clone, Copy)]
struct RepeatTranslationEdge {
    target: u32,
    completed: u16,
}

impl RepeatTranslationEdge {
    const DEAD: Self = Self {
        target: DEAD_REPEAT_TRANSLATION_STATE,
        completed: 0,
    };
}

struct RepeatTranslationAutomaton {
    transitions: Vec<Box<[RepeatTranslationEdge; 256]>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct RepeatBodyLanguageState {
    accepting: bool,
    transitions: Box<[(u8, u32)]>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct RepeatBodyLanguageKey(Box<[RepeatBodyLanguageState]>);

impl RepeatBodyLanguageKey {
    fn from_dfa(body: &DFA) -> Self {
        if body.num_states() == 0 {
            return Self(Box::new([]));
        }

        // Minimized equivalent DFAs may still carry different raw state ids
        // depending on how their expressions were factored. Canonical BFS
        // numbering from the start state makes the cache language-shaped rather
        // than construction-shaped. Byte transitions are visited in sorted
        // order, so the resulting key is deterministic.
        let mut canonical = vec![u32::MAX; body.num_states()];
        canonical[0] = 0;
        let mut next_id = 1u32;
        let mut queue = VecDeque::from([0u32]);
        let mut states = Vec::with_capacity(body.num_states());
        while let Some(state) = queue.pop_front() {
            let mut transitions = Vec::new();
            for (byte, &target) in body.states()[state as usize].transitions.iter() {
                let mapped = if canonical[target as usize] == u32::MAX {
                    let mapped = next_id;
                    next_id += 1;
                    canonical[target as usize] = mapped;
                    queue.push_back(target);
                    mapped
                } else {
                    canonical[target as usize]
                };
                transitions.push((byte, mapped));
            }
            states.push(RepeatBodyLanguageState {
                accepting: body.finalizers(state).contains(0),
                transitions: transitions.into_boxed_slice(),
            });
        }
        Self(states.into_boxed_slice())
    }
}

#[derive(Default)]
pub(crate) struct VocabularyRepeatHorizonCache {
    horizons: Mutex<
        FxHashMap<RepeatBodyLanguageKey, Arc<OnceLock<Option<usize>>>>,
    >,
}

impl VocabularyRepeatHorizonCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn horizon_for_dfa(&self, body: &DFA, vocab: &Vocab) -> Option<usize> {
        let key = RepeatBodyLanguageKey::from_dfa(body);
        let cell = self
            .horizons
            .lock()
            .ok()
            .map(|mut horizons| {
                Arc::clone(
                    horizons
                        .entry(key)
                        .or_insert_with(|| Arc::new(OnceLock::new())),
                )
            })?;
        *cell.get_or_init(|| vocabulary_repeat_boundary_horizon_for_dfa_uncached(body, vocab))
    }

    pub(crate) fn horizon_for_expr(&self, body: &Expr, vocab: &Vocab) -> Option<usize> {
        // The ordinary expression-to-DFA helper requires nested language
        // operations to be lowered before NFA compilation. Repeat synthesis is
        // an optimization, so unsupported body shapes must fail closed rather
        // than reaching that internal invariant.
        if expr_contains_group_op(body) {
            return None;
        }
        self.horizon_for_dfa(&compile_expr_to_dfa(body), vocab)
    }
}

fn normalize_repeat_translation_counts(counts: &mut [u32]) -> Option<u32> {
    let shift = counts
        .iter()
        .copied()
        .filter(|&count| count != u32::MAX)
        .min()?;
    for count in counts {
        if *count != u32::MAX {
            *count -= shift;
        }
    }
    Some(shift)
}

/// Advance the translation-invariant interior control state of `body*` by one
/// byte. Counts are minimum completed-copy counts for each live body residual.
/// A uniform count translation is factored out and returned as the edge weight.
///
/// This is the same dominance law used by the bounded-repeat/suffix compiler,
/// but with no finite upper boundary. It therefore captures exactly how many
/// repeat layers a continuation can move while remaining independent of the
/// absolute bounded-repeat count.
fn step_repeat_translation_state(
    body: &DFA,
    state: &RepeatTranslationState,
    byte: u8,
) -> Option<(RepeatTranslationState, u32)> {
    let mut next = vec![u32::MAX; body.num_states()];
    for (body_state, &completed) in state.0.iter().enumerate() {
        if completed == u32::MAX {
            continue;
        }
        let Some(target) = body.step(body_state as u32, byte) else {
            continue;
        };

        if body.finalizers(target).contains(0) {
            next[0] = next[0].min(completed.saturating_add(1));
        }
        if body.possible_future_group_ids(target).contains(0) {
            next[target as usize] = next[target as usize].min(completed);
        }
    }
    let shift = normalize_repeat_translation_counts(&mut next)?;
    Some((
        RepeatTranslationState(next.into_boxed_slice()),
        shift,
    ))
}

fn build_repeat_translation_automaton(
    body: &DFA,
    relevant_bytes: &[u8],
) -> Option<RepeatTranslationAutomaton> {
    const MAX_TRANSLATION_STATES: usize = 8_192;
    const MAX_TRANSLATION_EDGES: usize = 1_000_000;

    if body.num_states() == 0
        || body.num_groups() != 1
        || body.finalizers(0).contains(0)
    {
        return None;
    }

    let mut start = vec![u32::MAX; body.num_states()];
    start[0] = 0;
    let start = RepeatTranslationState(start.into_boxed_slice());
    let mut state_ids = FxHashMap::<RepeatTranslationState, u32>::default();
    state_ids.insert(start.clone(), 0);
    let mut states = vec![start.clone()];
    let mut worklist = VecDeque::from([start]);
    let mut transitions = Vec::<Box<[RepeatTranslationEdge; 256]>>::new();
    let mut edge_count = 0usize;

    while let Some(state) = worklist.pop_front() {
        let mut row = Box::new([RepeatTranslationEdge::DEAD; 256]);
        for &byte in relevant_bytes {
            let Some((target_state, completed)) =
                step_repeat_translation_state(body, &state, byte)
            else {
                continue;
            };
            let target = if let Some(&target) = state_ids.get(&target_state) {
                target
            } else {
                if states.len() >= MAX_TRANSLATION_STATES {
                    return None;
                }
                let target = states.len() as u32;
                state_ids.insert(target_state.clone(), target);
                states.push(target_state.clone());
                worklist.push_back(target_state);
                target
            };
            let completed = u16::try_from(completed).ok()?;
            row[byte as usize] = RepeatTranslationEdge { target, completed };
            edge_count += 1;
            if edge_count > MAX_TRANSLATION_EDGES {
                return None;
            }
        }
        transitions.push(row);
    }
    debug_assert_eq!(transitions.len(), states.len());
    Some(RepeatTranslationAutomaton { transitions })
}

/// Exact maximum uniform repeat-count displacement observable within the
/// suffix of one vocabulary token, starting from any translation-invariant
/// repeat-body residual.
///
/// Injecting every control state before each byte makes the analyzed language
/// suffix-closed. This is required because a token may consume a literal or
/// another product coordinate before entering the bounded repeat. The result
/// is vocabulary-relative but grammar-independent: it applies to every bounded
/// repeat whose body compiles to the supplied deterministic automaton.
fn max_repeat_translation_over_vocab_suffixes(
    automaton: &RepeatTranslationAutomaton,
    vocab: &Vocab,
) -> usize {
    let state_count = automaton.transitions.len();
    if state_count == 0 {
        return 0;
    }

    vocab
        .entries
        .values()
        .collect::<Vec<_>>()
        .par_iter()
        .map_init(
            || {
                (
                    vec![i32::MIN; state_count],
                    vec![i32::MIN; state_count],
                )
            },
            |(current, next), token| {
                current.fill(i32::MIN);
                for &byte in token.iter() {
                    next.fill(i32::MIN);
                    for (state, row) in automaton.transitions.iter().enumerate() {
                        // Start a new suffix at this byte from any reachable
                        // translation control state, or continue an earlier
                        // suffix when that has accumulated more completions.
                        let score = current[state].max(0);
                        let edge = row[byte as usize];
                        if edge.target == DEAD_REPEAT_TRANSLATION_STATE {
                            continue;
                        }
                        let candidate = score.saturating_add(i32::from(edge.completed));
                        let slot = &mut next[edge.target as usize];
                        *slot = (*slot).max(candidate);
                    }
                    std::mem::swap(current, next);
                }
                current
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    .max(0) as usize
            },
        )
        .max()
        .unwrap_or(0)
}

/// Compute a vocabulary-exact repeat-boundary horizon for one repeat body.
/// Returns `None` only when the body's translation control automaton exceeds a
/// conservative proof budget, in which case callers must retain their existing
/// byte-length upper bound or reject the synthesis candidate.
fn vocabulary_repeat_boundary_horizon_for_dfa_uncached(
    body: &DFA,
    vocab: &Vocab,
) -> Option<usize> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let relevant_bytes = vocab.relevant_bytes();
    let automaton = build_repeat_translation_automaton(body, &relevant_bytes)?;
    let horizon = max_repeat_translation_over_vocab_suffixes(&automaton, vocab);
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] vocabulary_repeat_horizon body_states={} translation_states={} relevant_bytes={} horizon={} elapsed_ms={:.3}",
            body.num_states(),
            automaton.transitions.len(),
            relevant_bytes.len(),
            horizon,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(horizon)
}

pub(crate) fn vocabulary_repeat_boundary_horizon(
    body_expr: &Expr,
    vocab: &Vocab,
) -> Option<usize> {
    VocabularyRepeatHorizonCache::new().horizon_for_expr(body_expr, vocab)
}

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

fn rebuild_single_visible_group_expression(
    compiled_exprs: &[Expr],
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> Option<Expr> {
    let mut used = vec![false; compiled_exprs.len()];
    let mut expression = compiled_exprs.first()?.clone();
    used[0] = true;

    let mut apply_hidden = |hidden: u32, intersect: bool| -> Option<()> {
        let hidden = hidden as usize;
        if hidden == 0 || hidden >= compiled_exprs.len() || used[hidden] {
            return None;
        }
        used[hidden] = true;
        let hidden_expression = compiled_exprs[hidden].clone();
        expression = if intersect {
            Expr::Intersect {
                expr: Box::new(expression.clone()),
                intersect: Box::new(hidden_expression),
            }
        } else {
            Expr::Exclude {
                expr: Box::new(expression.clone()),
                exclude: Box::new(hidden_expression),
            }
        };
        Some(())
    };

    if exclusions.keys().any(|&visible| visible != 0)
        || intersections.keys().any(|&visible| visible != 0)
    {
        return None;
    }
    for &hidden in exclusions.get(&0).into_iter().flatten() {
        apply_hidden(hidden, false)?;
    }
    for &hidden in intersections.get(&0).into_iter().flatten() {
        apply_hidden(hidden, true)?;
    }
    used.into_iter().all(|used| used).then(|| expression.optimize())
}

/// Expose one intersection nested under a common sequence shell. This is an
/// exact distributive rewrite:
///
/// ```text
/// prefix · (left ∩ right) · suffix
///   = (prefix · left · suffix) ∩ (prefix · right · suffix)
/// ```
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

struct ZeroMinRepeatSuffixBuild {
    dfa: DFA,
    states: Vec<ZeroMinRepeatSuffixState>,
    state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
}

fn compute_dfa_byte_equivalence_classes(dfas: &[&DFA]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for dfa in dfas {
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

fn build_zero_min_repeat_suffix_dominance_dfa_internal(
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
    preserve_coordinates: bool,
) -> Option<ZeroMinRepeatSuffixBuild> {
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
    let mut states = vec![start.clone()];
    let mut worklist = VecDeque::from([(0u32, start)]);
    let (byte_to_class, class_members) =
        compute_dfa_byte_equivalence_classes(&[body_dfa, suffix_dfa]);

    while let Some((state_id, state)) = worklist.pop_front() {
        let mut target_by_class = vec![u32::MAX; class_members.len()];
        for (class, members) in class_members.iter().enumerate() {
            let byte = members[0];
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
                states.push(next.clone());
                worklist.push_back((target, next));
                target
            };
            target_by_class[class] = target;
        }
        let transitions = byte_to_class
            .iter()
            .enumerate()
            .filter_map(|(byte, &class)| {
                let target = target_by_class[class as usize];
                (target != u32::MAX).then_some((byte as u8, target))
            })
            .collect();
        dfa.set_transitions_from_sorted_entries(state_id, transitions);
    }

    // Hopcroft-style minimization is counterproductive for broad direct
    // residual DFAs: a few thousand states with dense byte rows can cost
    // seconds even when almost no states merge. Downstream product/DWA
    // construction is already designed to consume unminimized deterministic
    // components. Keep minimization for compact results where it is cheap and
    // useful, but preserve the exact direct DFA as-is above that threshold.
    let transitions = dfa_transition_count(&dfa);
    if !preserve_coordinates && dfa.num_states() <= 2_048 && transitions <= 100_000 {
        dfa = dfa.minimize();
        states.clear();
        state_map.clear();
    }
    Some(ZeroMinRepeatSuffixBuild {
        dfa,
        states,
        state_by_key: state_map,
    })
}

fn build_zero_min_repeat_suffix_dominance_dfa(
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
) -> Option<DFA> {
    build_zero_min_repeat_suffix_dominance_dfa_internal(
        body_dfa,
        suffix_dfa,
        max,
        false,
    )
    .map(|built| built.dfa)
}

fn build_bounded_repeat_with_regex_suffix_with_options(
    parts: &[Expr],
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
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
        && let Some(built) = build_zero_min_repeat_suffix_dominance_dfa_internal(
            &body_dfa,
            &suffix_dfa,
            max,
            preserve_coordinates,
        )
    {
        let dfa = built.dfa;
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

fn build_bounded_repeat_with_regex_suffix(parts: &[Expr]) -> Option<(DFA, bool)> {
    build_bounded_repeat_with_regex_suffix_with_options(parts, false)
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

fn build_prefixed_bounded_repeat_with_suffix_dfa_with_options(
    parts: &[Expr],
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
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
                .or_else(|| {
                    build_bounded_repeat_with_regex_suffix_with_options(
                        &tail_parts,
                        preserve_coordinates,
                    )
                })?;
        let dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
        return Some((dfa, needs_future_recompute));
    }

    if parts.len() == 2 {
        let prefix_bytes = collect_suffix_bytes(&parts[..1])?;
        let tail_parts = optional_tail_parts(&parts[1])?;
        if tail_parts.len() >= 2 {
            let (tail_dfa, needs_future_recompute) =
                build_bounded_repeat_with_suffix_dfa(&tail_parts)
                    .or_else(|| {
                        build_bounded_repeat_with_regex_suffix_with_options(
                            &tail_parts,
                            preserve_coordinates,
                        )
                    })?;
            let mut dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
            mark_state_accepting(&mut dfa, prefix_bytes.len() as u32);
            return Some((dfa, needs_future_recompute));
        }
    }

    None
}

fn build_prefixed_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {
    build_prefixed_bounded_repeat_with_suffix_dfa_with_options(parts, false)
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
            compressed_transition_segments: Arc::from([]),
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
    full: DeferredDfa,
    full_to_synthesized: Vec<u32>,
    protected_residual: bool,
}

enum DeferredDfa {
    Ready(DFA),
    DenseBinary(DeferredDenseBinaryIntersectionProduct),
}

impl DeferredDfa {
    fn dfa(&self) -> &DFA {
        match self {
            Self::Ready(dfa) => dfa,
            Self::DenseBinary(product) => product.dfa(),
        }
    }

    fn dfa_mut(&mut self) -> &mut DFA {
        match self {
            Self::Ready(dfa) => dfa,
            Self::DenseBinary(product) => product.dfa_mut(),
        }
    }

    fn num_states(&self) -> usize {
        self.dfa().num_states()
    }

    fn finish(self) -> DFA {
        match self {
            Self::Ready(dfa) => dfa,
            Self::DenseBinary(product) => product.finish(),
        }
    }

    fn finish_runtime(self) -> (DFA, Option<CompressedTransitionSegment>) {
        match self {
            Self::Ready(dfa) => (dfa, None),
            Self::DenseBinary(product) => {
                let (dfa, segment) = product.finish_compressed();
                (dfa, Some(segment))
            }
        }
    }
}

fn isolate_component_nullable_start(dfa: DFA, num_terminals: usize) -> (DFA, BTreeSet<u32>) {
    let mut tokenizer = Regex { dfa }.into_tokenizer(num_terminals as u32, None);
    let nullable = tokenizer.isolate_start_state_and_drain_nullable_terminals();
    (tokenizer.dfa, nullable)
}

pub(crate) struct CompiledPartitionedExpressionPair {
    pub(crate) synthesized: Regex,
    pub(crate) full: Regex,
    pub(crate) full_to_synthesized: Vec<u32>,
}

pub(crate) struct PreparedPartitionedExpressionPair {
    pub(crate) synthesized: Regex,
    full: DeferredPartitionedRegex,
    pub(crate) full_to_synthesized: Vec<u32>,
    pub(crate) synthesized_expressions: Vec<Expr>,
}

impl PreparedPartitionedExpressionPair {
    pub(crate) fn full_num_states(&self) -> usize {
        self.full.num_states()
    }

    pub(crate) fn finish_full(self) -> CompiledPartitionedExpressionPair {
        CompiledPartitionedExpressionPair {
            synthesized: self.synthesized,
            full: self.full.finish(),
            full_to_synthesized: self.full_to_synthesized,
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (Regex, DeferredPartitionedRegex, Vec<u32>, Vec<Expr>) {
        (
            self.synthesized,
            self.full,
            self.full_to_synthesized,
            self.synthesized_expressions,
        )
    }
}

pub(crate) struct DeferredPartitionedRegex {
    components: Vec<LexerComponentPair>,
    total_groups: usize,
}

impl DeferredPartitionedRegex {
    pub(crate) fn num_states(&self) -> usize {
        1 + self
            .components
            .iter()
            .map(|component| component.full.num_states())
            .sum::<usize>()
    }

    pub(crate) fn finish(self) -> Regex {
        let components = self
            .components
            .into_iter()
            .map(|component| LexerComponent {
                terminal_ids: component.terminal_ids,
                dfa: component.full.finish(),
                protected_residual: component.protected_residual,
            })
            .collect::<Vec<_>>();
        Regex {
            dfa: combine_lexer_components_under_epsilon_root(components, self.total_groups),
        }
    }

    pub(crate) fn finish_runtime_tokenizer(
        self,
        num_terminals: u32,
        expressions: Arc<[Expr]>,
    ) -> Tokenizer {
        let mut combined = DFA::new(1);
        combined.ensure_group_capacity(self.total_groups);
        let mut root_futures = BitSet::new(self.total_groups);
        let mut compressed_segments = Vec::new();

        for component in self.components {
            let terminal_ids = component.terminal_ids;
            let (component_dfa, compressed) = component.full.finish_runtime();
            debug_assert_eq!(component_dfa.num_groups(), terminal_ids.len());
            for local_group in component_dfa.possible_future_group_ids(0).iter() {
                root_futures.set(terminal_ids[local_group]);
            }
            let offset = combined.append_rebased_component(component_dfa, &terminal_ids);
            combined.add_epsilon_transition(0, offset);
            if let Some(mut segment) = compressed {
                segment.state_offset = offset;
                compressed_segments.push(segment);
            }
        }
        combined.set_possible_future_group_ids(0, root_futures);
        Tokenizer::from_parts_with_compressed_transitions(
            combined,
            num_terminals,
            Some(expressions),
            compressed_segments,
        )
    }
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

fn combine_synthesized_component_pairs_under_epsilon_root(
    components: &[LexerComponentPair],
    total_groups: usize,
) -> (DFA, Vec<u32>) {
    let total_states = 1usize
        + components
            .iter()
            .map(|component| component.synthesized.num_states())
            .sum::<usize>();
    let mut combined = DFA::new(total_states);
    combined.ensure_group_capacity(total_groups);
    let mut root_futures = BitSet::new(total_groups);
    let mut offsets = Vec::with_capacity(components.len());

    let mut offset = 1u32;
    for component_pair in components {
        offsets.push(offset);
        let terminal_ids = &component_pair.terminal_ids;
        let component = &component_pair.synthesized;
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

fn compose_partitioned_component_state_maps(
    components: &[LexerComponentPair],
    synthesized_offsets: &[u32],
) -> Option<Vec<u32>> {
    if components.len() != synthesized_offsets.len() {
        return None;
    }
    let full_state_count = 1usize.checked_add(
        components
            .iter()
            .map(|component| component.full.num_states())
            .sum::<usize>(),
    )?;
    let mut full_to_synthesized = Vec::with_capacity(full_state_count);
    full_to_synthesized.push(0);
    let mut expected_synthesized_offset = 1u32;

    for (component, &synthesized_offset) in components.iter().zip(synthesized_offsets) {
        if synthesized_offset != expected_synthesized_offset
            || component.full_to_synthesized.len() != component.full.num_states()
            || component
                .full_to_synthesized
                .iter()
                .any(|&state| state as usize >= component.synthesized.num_states())
        {
            return None;
        }
        full_to_synthesized.extend(
            component
                .full_to_synthesized
                .iter()
                .map(|&state| synthesized_offset.checked_add(state))
                .collect::<Option<Vec<_>>>()?,
        );
        expected_synthesized_offset = expected_synthesized_offset
            .checked_add(component.synthesized.num_states() as u32)?;
    }

    (full_to_synthesized.len() == full_state_count).then_some(full_to_synthesized)
}

fn combine_lexer_components_under_epsilon_root(
    components: Vec<LexerComponent>,
    total_groups: usize,
) -> DFA {
    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(total_groups);
    let mut root_futures = BitSet::new(total_groups);

    for lexer_component in components {
        let terminal_ids = lexer_component.terminal_ids;
        let component = lexer_component.dfa;
        debug_assert_eq!(component.num_groups(), terminal_ids.len());
        for local_group in component.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(component, &terminal_ids);
        combined.add_epsilon_transition(0, offset);
    }
    combined.set_possible_future_group_ids(0, root_futures);
    combined
}

struct ExtractedDispatchComponent {
    terminal_ids: Vec<usize>,
    source_states: Vec<u32>,
    dfa: DFA,
}

pub(crate) struct PrecompiledFurtherSynthesisPairs {
    pairs: Mutex<BTreeMap<usize, CompiledTerminalExpressionPair>>,
    pub(crate) build_ms: f64,
}

impl PrecompiledFurtherSynthesisPairs {
    fn take(&self, terminal: usize) -> Option<CompiledTerminalExpressionPair> {
        self.pairs
            .lock()
            .expect("precompiled synthesis-pair cache poisoned")
            .remove(&terminal)
    }
}

pub(crate) fn precompile_further_synthesis_pairs(
    source_expressions: &[Expr],
    synthesized_expressions: &[Expr],
    protected_terminal_ids: &[u32],
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<Arc<PrecompiledFurtherSynthesisPairs>> {
    if source_expressions.len() != synthesized_expressions.len() {
        return None;
    }
    let protected = protected_terminal_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let changed = source_expressions
        .iter()
        .zip(synthesized_expressions)
        .enumerate()
        .filter_map(|(terminal, (source, synthesized))| {
            (source != synthesized).then_some(terminal)
        })
        .collect::<Vec<_>>();
    if changed.is_empty()
        || changed
            .iter()
            .any(|&terminal| !protected.contains(&(terminal as u32)))
    {
        return None;
    }
    let started_at = Instant::now();
    let pairs = changed
        .par_iter()
        .map(|&terminal| {
            compile_terminal_expression_pair_with_structural_map(
                &source_expressions[terminal],
                &synthesized_expressions[terminal],
                vocab,
                repeat_horizons,
                max_token_len,
                relevant_bytes,
            )
            .map(|pair| (terminal, pair))
        })
        .collect::<Option<BTreeMap<_, _>>>()?;
    Some(Arc::new(PrecompiledFurtherSynthesisPairs {
        pairs: Mutex::new(pairs),
        build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    }))
}

fn extract_dispatch_components(tokenizer: &Tokenizer) -> Option<Vec<ExtractedDispatchComponent>> {
    let roots = tokenizer.deterministic_dispatch_roots()?;
    let components = tokenizer.disjoint_dispatch_components()?;
    if roots.len() != components.len() {
        return None;
    }

    roots
        .iter()
        .copied()
        .zip(components)
        .map(|(root, mut source_states)| {
            source_states.sort_unstable();
            let root_position = source_states.iter().position(|&state| state == root)?;
            source_states.swap(0, root_position);

            let terminal_ids = tokenizer
                .dfa
                .possible_future_group_ids(root)
                .iter()
                .collect::<Vec<_>>();
            if terminal_ids.is_empty() {
                return None;
            }
            let mut local_group_by_terminal = vec![usize::MAX; tokenizer.num_terminals as usize];
            for (local_group, &terminal) in terminal_ids.iter().enumerate() {
                *local_group_by_terminal.get_mut(terminal)? = local_group;
            }
            let mut local_state_by_source = vec![u32::MAX; tokenizer.dfa.num_states()];
            for (local_state, &source_state) in source_states.iter().enumerate() {
                local_state_by_source[source_state as usize] = local_state as u32;
            }

            let mut dfa = DFA::new(source_states.len());
            dfa.ensure_group_capacity(terminal_ids.len());
            for (local_group, &terminal) in terminal_ids.iter().enumerate() {
                dfa.set_group_u8set(
                    local_group as u32,
                    *tokenizer.dfa.group_id_to_u8set(terminal as u32),
                );
            }
            for (local_state, &source_state) in source_states.iter().enumerate() {
                let source_dfa_state = tokenizer.dfa.states().get(source_state as usize)?;
                let transitions = source_dfa_state
                    .transitions
                    .iter()
                    .map(|(byte, &target)| {
                        let target = *local_state_by_source.get(target as usize)?;
                        (target != u32::MAX).then_some((byte, target))
                    })
                    .collect::<Option<Vec<_>>>()?;
                dfa.set_transitions_from_sorted_entries(local_state as u32, transitions);
                for &target in &source_dfa_state.epsilon_transitions {
                    let target = *local_state_by_source.get(target as usize)?;
                    if target == u32::MAX {
                        return None;
                    }
                    dfa.add_epsilon_transition(local_state as u32, target);
                }

                let mut finalizers = BitSet::new(terminal_ids.len());
                let mut futures = BitSet::new(terminal_ids.len());
                for terminal in tokenizer.dfa.finalizers(source_state).iter() {
                    let local_group = *local_group_by_terminal.get(terminal)?;
                    if local_group == usize::MAX {
                        return None;
                    }
                    finalizers.set(local_group);
                }
                for terminal in tokenizer
                    .dfa
                    .possible_future_group_ids(source_state)
                    .iter()
                {
                    let local_group = *local_group_by_terminal.get(terminal)?;
                    if local_group == usize::MAX {
                        return None;
                    }
                    futures.set(local_group);
                }
                dfa.overwrite_state_metadata(local_state as u32, finalizers, futures);
            }
            Some(ExtractedDispatchComponent {
                terminal_ids,
                source_states,
                dfa,
            })
        })
        .collect()
}

fn augment_component_from_verified_prefix(
    source: &DFA,
    rebuilt: &DFA,
    synthesized: &mut DFA,
    rebuilt_to_synthesized: &[u32],
) -> Option<Vec<u32>> {
    if source.num_groups() != rebuilt.num_groups()
        || source.num_groups() != synthesized.num_groups()
        || rebuilt_to_synthesized.len() != rebuilt.num_states()
        || rebuilt.num_states() > source.num_states()
    {
        return None;
    }
    for state in 0..rebuilt.num_states() {
        let state = state as u32;
        if source.finalizers(state) != rebuilt.finalizers(state)
            || source.possible_future_group_ids(state)
                != rebuilt.possible_future_group_ids(state)
            || source.states()[state as usize].epsilon_transitions
                != rebuilt.states()[state as usize].epsilon_transitions
            || source.states()[state as usize].transitions
                != rebuilt.states()[state as usize].transitions
        {
            return None;
        }
    }

    let mut source_to_synthesized = vec![u32::MAX; source.num_states()];
    source_to_synthesized[..rebuilt.num_states()].copy_from_slice(rebuilt_to_synthesized);
    for source_state in rebuilt.num_states()..source.num_states() {
        source_to_synthesized[source_state] = synthesized.add_state();
    }
    for source_state in rebuilt.num_states()..source.num_states() {
        let source_state_u32 = source_state as u32;
        let target_state = source_to_synthesized[source_state];
        let source_dfa_state = &source.states()[source_state];
        if !source_dfa_state.epsilon_transitions.is_empty() {
            return None;
        }
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| {
                Some((byte, *source_to_synthesized.get(target as usize)?))
            })
            .collect::<Option<Vec<_>>>()?;
        if transitions.iter().any(|&(_, target)| target == u32::MAX) {
            return None;
        }
        synthesized.set_transitions_from_sorted_entries(target_state, transitions);
        synthesized.overwrite_state_metadata(
            target_state,
            source.finalizers(source_state_u32).clone(),
            source.possible_future_group_ids(source_state_u32).clone(),
        );
    }
    Some(source_to_synthesized)
}

/// Further synthesize protected singleton components of an already-built
/// partitioned tokenizer without recompiling its ordinary components.
///
/// Every unchanged dispatch component is cloned exactly. Changed protected
/// components use the same structural terminal-pair proof as the global
/// synthesis, then clone any externally-entered residual states appended to
/// the actual source component. The returned map covers the actual source
/// tokenizer's raw-state domain.
pub(crate) fn compile_further_synthesized_tokenizer_with_structural_map(
    source: &Tokenizer,
    source_expressions: &[Expr],
    synthesized_expressions: &[Expr],
    protected_terminal_ids: &[u32],
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    precompiled_pairs: Option<&PrecompiledFurtherSynthesisPairs>,
) -> Option<(Tokenizer, Vec<u32>)> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let started_at = profile.then(Instant::now);
    let reject = |stage: &str| {
        if profile {
            eprintln!(
                "[glrmask/profile][partition_local_component_reuse] selected=false stage={} elapsed_ms={:.3}",
                stage,
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        None
    };
    if source_expressions.len() != synthesized_expressions.len()
        || source_expressions.len() != source.num_terminals as usize
    {
        return reject("input_shape");
    }
    let protected = protected_terminal_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let changed = source_expressions
        .iter()
        .zip(synthesized_expressions)
        .map(|(source, synthesized)| source != synthesized)
        .collect::<Vec<_>>();
    if !changed.iter().any(|&changed| changed) {
        return reject("unchanged");
    }

    let Some(extracted) = extract_dispatch_components(source) else {
        return reject("extract_dispatch");
    };
    let mut output_components = Vec::<LexerComponent>::with_capacity(extracted.len());
    let mut component_maps = Vec::<(Vec<u32>, Vec<u32>)>::with_capacity(extracted.len());
    let mut handled_changed = vec![false; changed.len()];
    let mut effective_synthesized_expressions = synthesized_expressions.to_vec();

    for component in extracted {
        let changed_terminals = component
            .terminal_ids
            .iter()
            .copied()
            .filter(|&terminal| changed.get(terminal).copied().unwrap_or(false))
            .collect::<Vec<_>>();
        if changed_terminals.is_empty() {
            let state_count = component.dfa.num_states() as u32;
            component_maps.push((
                component.source_states,
                (0..state_count).collect::<Vec<_>>(),
            ));
            output_components.push(LexerComponent {
                terminal_ids: component.terminal_ids,
                dfa: component.dfa,
                protected_residual: false,
            });
            continue;
        }
        if changed_terminals.len() != 1
            || component.terminal_ids.len() != 1
            || !protected.contains(&(changed_terminals[0] as u32))
        {
            return reject("changed_component_shape");
        }
        let terminal = changed_terminals[0];
        handled_changed[terminal] = true;
        let Some(pair) = precompiled_pairs
            .and_then(|pairs| pairs.take(terminal))
            .or_else(|| {
                compile_terminal_expression_pair_with_structural_map(
                    &source_expressions[terminal],
                    &synthesized_expressions[terminal],
                    vocab,
                    repeat_horizons,
                    max_token_len,
                    relevant_bytes,
                )
            })
        else {
            return reject("protected_pair");
        };
        effective_synthesized_expressions[terminal] = pair.synthesized_expression.clone();
        let (mut synthesized, synthesized_nullable) =
            isolate_component_nullable_start(pair.synthesized.dfa, 1);
        let (rebuilt, rebuilt_nullable) = isolate_component_nullable_start(pair.full.dfa, 1);
        if !synthesized_nullable.is_empty() || !rebuilt_nullable.is_empty() {
            return reject("protected_nullable");
        }
        let Some(source_to_synthesized) = augment_component_from_verified_prefix(
            &component.dfa,
            &rebuilt,
            &mut synthesized,
            &pair.full_to_synthesized,
        ) else {
            if profile {
                eprintln!(
                    "[glrmask/profile][partition_local_component_reuse_detail] terminal={} source_states={} rebuilt_states={} synthesized_states={}",
                    terminal,
                    component.dfa.num_states(),
                    rebuilt.num_states(),
                    synthesized.num_states(),
                );
            }
            return reject("protected_prefix");
        };
        component_maps.push((component.source_states, source_to_synthesized));
        output_components.push(LexerComponent {
            terminal_ids: component.terminal_ids,
            dfa: synthesized,
            protected_residual: true,
        });
    }
    if changed
        .iter()
        .enumerate()
        .any(|(terminal, &changed)| changed && !handled_changed[terminal])
    {
        return reject("unhandled_changed");
    }

    let mut source_to_synthesized = vec![u32::MAX; source.dfa.num_states()];
    source_to_synthesized[0] = 0;
    let mut offset = 1u32;
    for (component, (source_states, local_map)) in output_components.iter().zip(&component_maps) {
        if source_states.len() != local_map.len() {
            return reject("component_map_length");
        }
        for (&source_state, &local_state) in source_states.iter().zip(local_map) {
            source_to_synthesized[source_state as usize] = offset + local_state;
        }
        offset = offset.checked_add(component.dfa.num_states() as u32)?;
    }
    let mut dfa = combine_lexer_components_under_epsilon_root(
        output_components,
        source_expressions.len(),
    );
    // Nullable-start isolation and prior structural augmentation may retain
    // unreachable raw states outside the live dispatch components. They remain
    // part of the compiler's raw-state coordinate, so clone them after the live
    // component layout is fixed and redirect every edge through the completed
    // source-state map.
    let extra_source_states = source_to_synthesized
        .iter()
        .enumerate()
        .filter_map(|(state, &mapped)| (mapped == u32::MAX).then_some(state))
        .collect::<Vec<_>>();
    for &source_state in &extra_source_states {
        source_to_synthesized[source_state] = dfa.add_state();
    }
    for source_state in extra_source_states {
        let target_state = source_to_synthesized[source_state];
        let source_dfa_state = &source.dfa.states()[source_state];
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, source_to_synthesized[target as usize]))
            .collect::<Vec<_>>();
        dfa.set_transitions_from_sorted_entries(target_state, transitions);
        for &target in &source_dfa_state.epsilon_transitions {
            dfa.add_epsilon_transition(target_state, source_to_synthesized[target as usize]);
        }
        dfa.overwrite_state_metadata(
            target_state,
            source.dfa.finalizers(source_state as u32).clone(),
            source
                .dfa
                .possible_future_group_ids(source_state as u32)
                .clone(),
        );
    }
    let tokenizer = Regex { dfa }.into_tokenizer(
        source_expressions.len() as u32,
        Some(Arc::from(
            effective_synthesized_expressions.into_boxed_slice(),
        )),
    );
    if profile {
        eprintln!(
            "[glrmask/profile][partition_local_component_reuse] selected=true source_states={} synthesized_states={} elapsed_ms={:.3}",
            source.dfa.num_states(),
            tokenizer.dfa.num_states(),
            started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some((tokenizer, source_to_synthesized))
}

/// Compile independently protected terminal partitions as exact/synthesized
/// pairs while compiling every unchanged partition only once. The returned
/// state map is structural: the global epsilon root maps to the global root,
/// unchanged components map identically, and protected components use their
/// certified local product maps. Adaptive prefix determinization remains
/// available for the ordinary components but never crosses a protected
/// residual coordinate.
pub(crate) fn prepare_partitioned_expression_pair_with_structural_map(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<PreparedPartitionedExpressionPair> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    if full_exprs.len() != synthesized_exprs.len()
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
                return Some((
                    LexerComponentPair {
                        terminal_ids,
                        synthesized: dfa.clone(),
                        full: DeferredDfa::Ready(dfa),
                        full_to_synthesized: (0..state_count).collect(),
                        protected_residual: false,
                    },
                    None,
                ));
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
            let pair = prepare_terminal_expression_pair_with_structural_map(
                &full_exprs[terminal],
                &synthesized_exprs[terminal],
                vocab,
                repeat_horizons,
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
            let (synthesized, synthesized_nullable) =
                isolate_component_nullable_start(pair.synthesized.dfa, terminal_ids.len());
            let mut full = pair.full;
            let full_nullable = if full.dfa().finalizers(0).is_empty()
                && !full.dfa().has_epsilon_transitions()
            {
                BTreeSet::new()
            } else {
                let (dfa, nullable) =
                    isolate_component_nullable_start(full.finish(), terminal_ids.len());
                full = DeferredDfa::Ready(dfa);
                nullable
            };
            if !synthesized_nullable.is_empty() || !full_nullable.is_empty() {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=nullable_protected_terminal terminal={} synthesized_nullable={:?} full_nullable={:?}",
                        terminal, synthesized_nullable, full_nullable,
                    );
                }
                return None;
            }
            Some((
                LexerComponentPair {
                    terminal_ids,
                    synthesized,
                    full,
                    full_to_synthesized: pair.full_to_synthesized,
                    protected_residual: true,
                },
                Some((terminal, pair.synthesized_expression)),
            ))
        })
        .collect::<Option<Vec<_>>>()?;

    let mut protected = Vec::new();
    let mut ordinary = Vec::new();
    let mut effective_synthesized_expressions = synthesized_exprs.to_vec();
    for (pair, effective_expression) in compiled {
        if let Some((terminal, expression)) = effective_expression {
            effective_synthesized_expressions[terminal] = expression;
        }
        if pair.protected_residual {
            protected.push(pair);
        } else {
            let dfa = match pair.full {
                DeferredDfa::Ready(dfa) => dfa,
                DeferredDfa::DenseBinary(_) => {
                    unreachable!("ordinary lexer components are never deferred")
                }
            };
            ordinary.push(LexerComponent {
                terminal_ids: pair.terminal_ids,
                dfa,
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
            let terminal_ids = component.terminal_ids;
            let (dfa, _) = isolate_component_nullable_start(component.dfa, terminal_ids.len());
            let state_count = dfa.num_states() as u32;
            LexerComponentPair {
                terminal_ids,
                synthesized: dfa.clone(),
                full: DeferredDfa::Ready(dfa),
                full_to_synthesized: (0..state_count).collect(),
                protected_residual: false,
            }
        })
        .collect::<Vec<_>>();
    pairs.append(&mut protected);
    pairs.sort_unstable_by_key(|pair| {
        pair.terminal_ids.first().copied().unwrap_or(usize::MAX)
    });

    let (synthesized, synthesized_offsets) =
        combine_synthesized_component_pairs_under_epsilon_root(&pairs, full_exprs.len());
    let full_to_synthesized =
        compose_partitioned_component_state_maps(&pairs, &synthesized_offsets)?;

    Some(PreparedPartitionedExpressionPair {
        synthesized: Regex { dfa: synthesized },
        full: DeferredPartitionedRegex {
            components: pairs,
            total_groups: full_exprs.len(),
        },
        full_to_synthesized,
        synthesized_expressions: effective_synthesized_expressions,
    })
}

pub(crate) fn compile_partitioned_expression_pair_with_structural_map(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledPartitionedExpressionPair> {
    prepare_partitioned_expression_pair_with_structural_map(
        full_exprs,
        synthesized_exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
    )
    .map(PreparedPartitionedExpressionPair::finish_full)
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
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
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
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                dfa.finalizers(state).contains(0)
            }
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
    let mut predecessor_counts = vec![0u32; states];
    let mut seen_target_epoch = vec![0u32; states];
    let mut epoch = 1u32;
    let mut edge_count = 0usize;
    for transitions in class_transitions {
        for &(_, target) in transitions {
            if seen_target_epoch[target as usize] == epoch {
                continue;
            }
            seen_target_epoch[target as usize] = epoch;
            predecessor_counts[target as usize] += 1;
            edge_count += 1;
        }
        epoch = epoch.checked_add(1).expect("product state epoch overflow");
    }

    let mut offsets = vec![0usize; states + 1];
    for state in 0..states {
        offsets[state + 1] = offsets[state] + predecessor_counts[state] as usize;
    }
    debug_assert_eq!(offsets[states], edge_count);
    let mut write_offsets = offsets[..states].to_vec();
    let mut predecessors = vec![0u32; edge_count];
    seen_target_epoch.fill(0);
    epoch = 1;
    for (source, transitions) in class_transitions.iter().enumerate() {
        for &(_, target) in transitions {
            if seen_target_epoch[target as usize] == epoch {
                continue;
            }
            seen_target_epoch[target as usize] = epoch;
            let slot = &mut write_offsets[target as usize];
            predecessors[*slot] = source as u32;
            *slot += 1;
        }
        epoch = epoch.checked_add(1).expect("product state epoch overflow");
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

/// Compute single-group futures from a class graph without first rebuilding a
/// CSR predecessor graph. The deferred dense intersection already stores one
/// compact transition row per state. A linked reverse graph can be assembled
/// in one pass, and the reverse reachability walk itself identifies exactly
/// the states with an outgoing edge to a final-reachable state.
fn set_single_group_futures_from_class_graph_csr(
    dfa: &mut DFA,
    transition_offsets: &[u32],
    class_transitions: &[(u8, u32)],
) {
    let states = dfa.num_states();
    debug_assert_eq!(transition_offsets.len(), states + 1);

    let mut predecessor_heads = vec![u32::MAX; states];
    let mut predecessor_sources = Vec::<u32>::new();
    let mut predecessor_next = Vec::<u32>::new();
    let mut seen_target_source = vec![u32::MAX; states];

    for source in 0..states {
        let source_u32 = source as u32;
        let row_start = transition_offsets[source] as usize;
        let row_end = transition_offsets[source + 1] as usize;
        for &(_, target) in &class_transitions[row_start..row_end] {
            let target = target as usize;
            if seen_target_source[target] == source_u32 {
                continue;
            }
            seen_target_source[target] = source_u32;
            let edge = predecessor_sources.len() as u32;
            predecessor_sources.push(source_u32);
            predecessor_next.push(predecessor_heads[target]);
            predecessor_heads[target] = edge;
        }
    }

    let mut can_reach_final = vec![false; states];
    let mut has_future = vec![false; states];
    let mut queue = VecDeque::<u32>::new();
    for state in 0..states {
        if dfa.finalizers(state as u32).contains(0) {
            can_reach_final[state] = true;
            queue.push_back(state as u32);
        }
    }

    while let Some(target) = queue.pop_front() {
        let mut edge = predecessor_heads[target as usize];
        while edge != u32::MAX {
            let source = predecessor_sources[edge as usize] as usize;
            has_future[source] = true;
            if !can_reach_final[source] {
                can_reach_final[source] = true;
                queue.push_back(source as u32);
            }
            edge = predecessor_next[edge as usize];
        }
    }

    for (state, &future) in has_future.iter().enumerate() {
        let mut future_groups = BitSet::new(1);
        if future {
            future_groups.set(0);
        }
        dfa.set_possible_future_group_ids(state as u32, future_groups);
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

fn compile_product_component_dfa_direct_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
    match expr {
        Expr::Shared(inner) => {
            compile_product_component_dfa_direct_with_options(inner, preserve_coordinates)
        }
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
                compile_product_component_dfa_direct_with_options(
                    non_epsilon,
                    preserve_coordinates,
                )?;
            mark_state_accepting(&mut dfa, 0);
            Some((dfa, needs_future_recompute))
        }
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => build_bounded_repeat_dfa(expr, *min, *max).map(|dfa| (dfa, false)),
        Expr::Seq(parts) => build_bounded_repeat_with_suffix_dfa(parts)
            .or_else(|| {
                build_bounded_repeat_with_regex_suffix_with_options(
                    parts,
                    preserve_coordinates,
                )
            })
            .or_else(|| {
                build_prefixed_bounded_repeat_with_suffix_dfa_with_options(
                    parts,
                    preserve_coordinates,
                )
            }),
        _ => None,
    }
}

fn compile_product_component_dfa_direct(expr: &Expr) -> Option<(DFA, bool)> {
    compile_product_component_dfa_direct_with_options(expr, false)
}

fn compile_product_component_dfa(expr: &Expr) -> DFA {
    compile_with_plan(build_exclusion_compile_plan(std::slice::from_ref(expr)))
}

fn compile_product_component_materialized_dfa_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
) -> DFA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let direct_started_at = profile_timing.then(Instant::now);
    if let Some((mut dfa, needs_future_recompute)) =
        compile_product_component_dfa_direct_with_options(expr, preserve_coordinates)
    {
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

fn compile_product_component_materialized_dfa(expr: &Expr) -> DFA {
    compile_product_component_materialized_dfa_with_options(expr, false)
}

#[derive(Clone)]
enum ProductComponent {
    Materialized(Arc<DFA>),
    MaterializedZeroMinRepeatSuffix {
        dfa: Arc<DFA>,
        trace: Arc<ZeroMinRepeatSuffixComponentTrace>,
    },
    VirtualBoundedRepeat {
        base_dfa: Arc<DFA>,
        min: u32,
        max: u32,
    },
}

impl ProductComponent {
    fn materialized_dfa(&self) -> Option<&DFA> {
        match self {
            Self::Materialized(dfa) | Self::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                Some(dfa)
            }
            Self::VirtualBoundedRepeat { .. } => None,
        }
    }

    fn zero_min_repeat_suffix_trace(&self) -> Option<Arc<ZeroMinRepeatSuffixComponentTrace>> {
        match self {
            Self::MaterializedZeroMinRepeatSuffix { trace, .. } => Some(Arc::clone(trace)),
            _ => None,
        }
    }
}

struct ProductBuildTrace {
    components: Vec<ProductComponent>,
    state_tuples: ProductStateTuples,
    state_lookup: ProductStateLookup,
    direct_single_visible_group: bool,
}

enum ProductStateTuples {
    Generic(Vec<ProductStateTuple>),
    DenseBinary(Vec<(u32, u32)>),
}

impl ProductStateTuples {
    fn len(&self) -> usize {
        match self {
            Self::Generic(tuples) => tuples.len(),
            Self::DenseBinary(pairs) => pairs.len(),
        }
    }

    fn tuple(&self, state: usize) -> ProductStateTuple {
        match self {
            Self::Generic(tuples) => tuples[state].clone(),
            Self::DenseBinary(pairs) => {
                let (left, right) = pairs[state];
                let mut tuple = ProductStateTuple::new();
                tuple.push((0, left));
                tuple.push((1, right));
                tuple
            }
        }
    }

    fn push(&mut self, tuple: ProductStateTuple) {
        match self {
            Self::Generic(tuples) => tuples.push(tuple),
            Self::DenseBinary(pairs)
                if tuple.len() == 2 && tuple[0].0 == 0 && tuple[1].0 == 1 =>
            {
                pairs.push((tuple[0].1, tuple[1].1));
            }
            Self::DenseBinary(pairs) => {
                let mut tuples = Vec::with_capacity(pairs.len() + 1);
                tuples.extend(pairs.drain(..).map(|(left, right)| {
                    let mut tuple = ProductStateTuple::new();
                    tuple.push((0, left));
                    tuple.push((1, right));
                    tuple
                }));
                tuples.push(tuple);
                *self = Self::Generic(tuples);
            }
        }
    }
}

enum ProductStateLookup {
    Hash(FxHashMap<ProductStateTuple, u32>),
    DenseBinary {
        right_states: usize,
        state_by_pair: Vec<u32>,
        overflow: FxHashMap<ProductStateTuple, u32>,
    },
}

impl ProductStateLookup {
    fn dense_pair_index(tuple: &ProductStateTuple, right_states: usize) -> Option<usize> {
        if tuple.len() != 2 || tuple[0].0 != 0 || tuple[1].0 != 1 {
            return None;
        }
        (tuple[0].1 as usize)
            .checked_mul(right_states)?
            .checked_add(tuple[1].1 as usize)
    }

    fn get(&self, tuple: &ProductStateTuple) -> Option<u32> {
        match self {
            Self::Hash(states) => states.get(tuple).copied(),
            Self::DenseBinary {
                right_states,
                state_by_pair,
                overflow,
            } => Self::dense_pair_index(tuple, *right_states)
                .and_then(|index| state_by_pair.get(index).copied())
                .filter(|&state| state != u32::MAX)
                .or_else(|| overflow.get(tuple).copied()),
        }
    }

    fn insert(&mut self, tuple: ProductStateTuple, state: u32) {
        match self {
            Self::Hash(states) => {
                states.insert(tuple, state);
            }
            Self::DenseBinary {
                right_states,
                state_by_pair,
                overflow,
            } => {
                if let Some(index) = Self::dense_pair_index(&tuple, *right_states)
                    && let Some(slot) = state_by_pair.get_mut(index)
                {
                    *slot = state;
                } else {
                    overflow.insert(tuple, state);
                }
            }
        }
    }
}

fn product_component_mapping_dfa(component: &ProductComponent) -> Option<DFA> {
    match component {
        ProductComponent::Materialized(dfa)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
            Some(dfa.as_ref().clone())
        }
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

struct ZeroMinRepeatSuffixComponentTrace {
    dfa: Arc<DFA>,
    prefix_len: usize,
    body_dfa: DFA,
    suffix_dfa: DFA,
    max: usize,
    tail_states: Vec<ZeroMinRepeatSuffixState>,
    tail_state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
}

fn dfa_has_same_numbered_layout(left: &DFA, right: &DFA) -> bool {
    left.num_groups() == right.num_groups()
        && left.num_states() == right.num_states()
        && (0..left.num_groups()).all(|group| {
            left.group_id_to_u8set(group as u32)
                == right.group_id_to_u8set(group as u32)
        })
        && left
            .states()
            .iter()
            .zip(right.states())
            .enumerate()
            .all(|(state, (left_state, right_state))| {
                left_state.transitions == right_state.transitions
                    && left_state.epsilon_transitions == right_state.epsilon_transitions
                    && left_state.finalizers == right_state.finalizers
                    && left.possible_future_group_ids(state as u32)
                        == right.possible_future_group_ids(state as u32)
            })
}

fn zero_min_repeat_suffix_component_trace(
    expr: &Expr,
) -> Option<ZeroMinRepeatSuffixComponentTrace> {
    let expr = unwrap_shared(expr);
    let Expr::Seq(parts) = expr else {
        return None;
    };
    let mut flat_parts = Vec::<Expr>::new();
    for part in parts {
        match unwrap_shared(part) {
            Expr::Seq(inner) => flat_parts.extend(inner.iter().cloned()),
            _ => flat_parts.push(part.clone()),
        }
    }

    for repeat_index in 0..flat_parts.len().saturating_sub(1) {
        let Expr::Repeat {
            expr: body,
            min: 0,
            max: Some(max),
        } = unwrap_shared(&flat_parts[repeat_index])
        else {
            continue;
        };
        if *max == 0 {
            continue;
        }
        let prefix = collect_suffix_bytes(&flat_parts[..repeat_index])?;
        let suffix_expr = seq_from_parts(flat_parts[repeat_index + 1..].to_vec());
        let body_dfa = compile_expr_to_dfa(body);
        let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
        if body_dfa.num_states() == 0
            || body_dfa.num_states() > 256
            || body_dfa.finalizers(0).contains(0)
            || suffix_dfa.num_states() == 0
            || suffix_dfa.finalizers(0).contains(0)
        {
            continue;
        }
        let built = build_zero_min_repeat_suffix_dominance_dfa_internal(
            &body_dfa,
            &suffix_dfa,
            *max,
            true,
        )?;
        let mut dfa = prepend_literal_prefix_to_dfa(&prefix, built.dfa)?;
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        return Some(ZeroMinRepeatSuffixComponentTrace {
            dfa: Arc::new(dfa),
            prefix_len: prefix.len(),
            body_dfa,
            suffix_dfa,
            max: *max,
            tail_states: built.states,
            tail_state_by_key: built.state_by_key,
        });
    }
    None
}

struct ZeroMinRepeatSuffixStateMap {
    primary: Vec<u32>,
    full_trace: Arc<ZeroMinRepeatSuffixComponentTrace>,
    synthesized_trace: Arc<ZeroMinRepeatSuffixComponentTrace>,
    body_state_map: Arc<[u32]>,
    suffix_state_map: Arc<[u32]>,
    crossed_boundaries: u32,
    interior_representative: u32,
    full_max: u32,
    synthesized_max: u32,
}

impl ZeroMinRepeatSuffixStateMap {
    fn primary(&self) -> &[u32] {
        &self.primary
    }

    fn visit_candidates(&self, full_state: u32, mut visit: impl FnMut(u32) -> bool) -> bool {
        let primary = self.primary[full_state as usize];
        if visit(primary) {
            return true;
        }
        let Some(state) = (full_state as usize)
            .checked_sub(self.full_trace.prefix_len)
            .and_then(|index| self.full_trace.tail_states.get(index))
        else {
            return false;
        };
        let Some(minimum_completed) = state
            .body_min_counts
            .iter()
            .copied()
            .filter(|&count| count != u32::MAX)
            .min()
        else {
            return false;
        };
        let Some(distance_to_upper) = self.full_max.checked_sub(minimum_completed) else {
            return false;
        };
        if distance_to_upper <= self.crossed_boundaries {
            return false;
        }

        let mapped_minimum = self.interior_representative;
        let mut seen = SmallVec::<[u32; 8]>::from_slice(&[primary]);
        for alternative_minimum in 0..=self.synthesized_max {
            if alternative_minimum == mapped_minimum {
                continue;
            }
            let Some(alternative) = zero_min_repeat_suffix_candidate(
                state,
                Some(minimum_completed),
                Some(alternative_minimum),
                &self.body_state_map,
                &self.suffix_state_map,
                &self.synthesized_trace,
                self.synthesized_max,
            ) else {
                continue;
            };
            if self.full_trace.dfa.finalizers(full_state)
                != self.synthesized_trace.dfa.finalizers(alternative)
                || self.full_trace.dfa.possible_future_group_ids(full_state)
                    != self
                        .synthesized_trace
                        .dfa
                        .possible_future_group_ids(alternative)
                || seen.contains(&alternative)
            {
                continue;
            }
            seen.push(alternative);
            if visit(alternative) {
                return true;
            }
        }
        false
    }
}

fn zero_min_repeat_suffix_candidate(
    state: &ZeroMinRepeatSuffixState,
    minimum_completed: Option<u32>,
    mapped_minimum: Option<u32>,
    body_state_map: &[u32],
    suffix_state_map: &[u32],
    synthesized_trace: &ZeroMinRepeatSuffixComponentTrace,
    synthesized_max: u32,
) -> Option<u32> {
    let mut body_min_counts = vec![u32::MAX; synthesized_trace.body_dfa.num_states()];
    for (full_body_state, &completed) in state.body_min_counts.iter().enumerate() {
        if completed == u32::MAX {
            continue;
        }
        let minimum = minimum_completed?;
        let mapped_completed = mapped_minimum?.checked_add(completed.checked_sub(minimum)?)?;
        if mapped_completed > synthesized_max {
            return None;
        }
        let mapped_body_state = body_state_map[full_body_state] as usize;
        body_min_counts[mapped_body_state] =
            body_min_counts[mapped_body_state].min(mapped_completed);
    }
    let mut suffix_states = state
        .suffix_states
        .iter()
        .map(|&state| suffix_state_map[state as usize])
        .collect::<Vec<_>>();
    close_zero_min_repeat_suffix_state(
        &mut body_min_counts,
        &mut suffix_states,
        &synthesized_trace.body_dfa,
        &synthesized_trace.suffix_dfa,
        synthesized_trace.max,
    );
    let mapped_key = ZeroMinRepeatSuffixState {
        body_min_counts: body_min_counts.into_boxed_slice(),
        suffix_states: suffix_states.into_boxed_slice(),
    };
    synthesized_trace
        .tail_state_by_key
        .get(&mapped_key)
        .copied()
        .map(|tail| synthesized_trace.prefix_len as u32 + tail)
}

fn zero_min_repeat_suffix_state_map(
    full_expr: &Expr,
    synthesized_expr: &Expr,
    full: &DFA,
    synthesized: &DFA,
    full_trace: Option<Arc<ZeroMinRepeatSuffixComponentTrace>>,
    synthesized_trace: Option<Arc<ZeroMinRepeatSuffixComponentTrace>>,
    max_token_len: usize,
    vocab: Option<&Vocab>,
    repeat_horizons: Option<&VocabularyRepeatHorizonCache>,
) -> Option<ZeroMinRepeatSuffixStateMap> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let full_trace = full_trace
        .or_else(|| zero_min_repeat_suffix_component_trace(full_expr).map(Arc::new))?;
    let synthesized_trace = if let Some(trace) = synthesized_trace {
        trace
    } else if let Some(trace) = zero_min_repeat_suffix_component_trace(synthesized_expr) {
        Arc::new(trace)
    } else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=synthesized_trace full_states={} synthesized_states={}",
                full.num_states(), synthesized.num_states(),
            );
        }
        return None;
    };
    let full_layout = dfa_has_same_numbered_layout(&full_trace.dfa, full);
    let synthesized_layout =
        dfa_has_same_numbered_layout(&synthesized_trace.dfa, synthesized);
    let body_state_map = deterministic_component_homomorphism_state_map(
        &full_trace.body_dfa,
        &synthesized_trace.body_dfa,
    );
    let suffix_state_map = deterministic_component_homomorphism_state_map(
        &full_trace.suffix_dfa,
        &synthesized_trace.suffix_dfa,
    );
    if full_trace.prefix_len != synthesized_trace.prefix_len
        || full_trace.max < synthesized_trace.max
        || !full_layout
        || !synthesized_layout
        || body_state_map.is_none()
        || suffix_state_map.is_none()
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=shape prefix={}/{} max={}/{} trace_states={}/{} actual_states={}/{} full_layout={} synthesized_layout={} body_map={} suffix_map={}",
                full_trace.prefix_len,
                synthesized_trace.prefix_len,
                full_trace.max,
                synthesized_trace.max,
                full_trace.dfa.num_states(),
                synthesized_trace.dfa.num_states(),
                full.num_states(),
                synthesized.num_states(),
                full_layout,
                synthesized_layout,
                body_state_map.is_some(),
                suffix_state_map.is_some(),
            );
        }
        return None;
    }
    let body_state_map = body_state_map.unwrap();
    let suffix_state_map = suffix_state_map.unwrap();

    let minimum_body_width = full_trace.body_dfa.min_match_byte_len()?.max(1);
    let fallback_crossed_boundaries = max_token_len
        .div_ceil(minimum_body_width)
        .saturating_add(1);
    let crossed_boundaries = vocab
        .zip(repeat_horizons)
        .and_then(|(vocab, horizons)| horizons.horizon_for_dfa(&full_trace.body_dfa, vocab))
        .unwrap_or(fallback_crossed_boundaries);
    if synthesized_trace.max <= crossed_boundaries {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=horizon full_max={} synthesized_max={} body_width={} crossed_boundaries={}",
                full_trace.max,
                synthesized_trace.max,
                minimum_body_width,
                crossed_boundaries,
            );
        }
        return None;
    }
    let interior_representative = synthesized_trace
        .max
        .checked_sub(crossed_boundaries.saturating_add(1))? as u32;
    let full_max = full_trace.max as u32;
    let synthesized_max = synthesized_trace.max as u32;

    let mut mapping = Vec::with_capacity(full.num_states());
    for state in 0..full_trace.prefix_len {
        mapping.push(state as u32);
    }
    for state in &full_trace.tail_states {
        let minimum_completed = state
            .body_min_counts
            .iter()
            .copied()
            .filter(|&count| count != u32::MAX)
            .min();
        let mapped_minimum = if let Some(minimum) = minimum_completed {
            let distance_to_upper = full_max.checked_sub(minimum)?;
            if distance_to_upper <= crossed_boundaries as u32 {
                Some(synthesized_max.checked_sub(distance_to_upper)?)
            } else {
                Some(interior_representative)
            }
        } else {
            None
        };
        let Some(mapped) = zero_min_repeat_suffix_candidate(
            state,
            minimum_completed,
            mapped_minimum,
            &body_state_map,
            &suffix_state_map,
            &synthesized_trace,
            synthesized_max,
        ) else {
            if profile {
                let full_counts = state
                    .body_min_counts
                    .iter()
                    .enumerate()
                    .filter_map(|(body_state, &count)| {
                        (count != u32::MAX).then_some((body_state, count))
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=missing_key full_state={} full_max={} synthesized_max={} body_width={} crossed_boundaries={} full_counts={:?} mapped_minimum={:?} suffix_states={:?}",
                    mapping.len(),
                    full_trace.max,
                    synthesized_trace.max,
                    minimum_body_width,
                    crossed_boundaries,
                    full_counts,
                    mapped_minimum,
                    state.suffix_states,
                );
            }
            return None;
        };
        let full_state = mapping.len() as u32;
        if full.finalizers(full_state) != synthesized.finalizers(mapped)
            || full.possible_future_group_ids(full_state)
                != synthesized.possible_future_group_ids(mapped)
        {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=metadata full_state={} synthesized_state={}",
                    full_state, mapped,
                );
            }
            return None;
        }
        mapping.push(mapped);
    }
    if mapping.len() != full.num_states() {
        return None;
    }
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] dominance_state_map_accepted full_states={} synthesized_states={} full_max={} synthesized_max={} body_width={} crossed_boundaries={}",
            full.num_states(),
            synthesized.num_states(),
            full_trace.max,
            synthesized_trace.max,
            minimum_body_width,
            crossed_boundaries,
        );
    }
    Some(ZeroMinRepeatSuffixStateMap {
        primary: mapping,
        full_trace,
        synthesized_trace,
        body_state_map: Arc::from(body_state_map.into_boxed_slice()),
        suffix_state_map: Arc::from(suffix_state_map.into_boxed_slice()),
        crossed_boundaries: crossed_boundaries as u32,
        interior_representative,
        full_max,
        synthesized_max,
    })
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
    Dominance(ZeroMinRepeatSuffixStateMap),
    Vocabulary(CertifiedVocabularyExactStateCandidates),
}

impl ProductComponentStateMap {
    fn primary(&self) -> &[u32] {
        match self {
            Self::Fixed(mapping) => mapping,
            Self::Layered(mapping) => mapping.primary(),
            Self::Dominance(mapping) => mapping.primary(),
            Self::Vocabulary(mapping) => mapping.primary(),
        }
    }

    fn visit_candidates(&self, full_state: u32, visit: impl FnMut(u32) -> bool) -> bool {
        match self {
            Self::Fixed(mapping) => {
                let mut visit = visit;
                visit(mapping[full_state as usize])
            }
            Self::Layered(mapping) => mapping.visit_candidates(full_state, visit),
            Self::Dominance(mapping) => mapping.visit_candidates(full_state, visit),
            Self::Vocabulary(mapping) => mapping.visit_candidates(full_state, visit),
        }
    }

    fn is_flexible(&self) -> bool {
        matches!(
            self,
            Self::Layered(_) | Self::Dominance(_) | Self::Vocabulary(_)
        )
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
    vocab: Option<&Vocab>,
    repeat_horizons: Option<&VocabularyRepeatHorizonCache>,
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
    let fallback_crossed_boundaries = max_token_len
        .div_ceil(minimum_body_width)
        .saturating_add(1);
    let crossed_boundaries = vocab
        .zip(repeat_horizons)
        .and_then(|(vocab, horizons)| horizons.horizon_for_dfa(&base, vocab))
        .unwrap_or(fallback_crossed_boundaries);
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
    if let Some(state) = trace.state_lookup.get(&tuple) {
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
    trace.state_lookup.insert(tuple.clone(), state);
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
                    ProductComponent::Materialized(_)
                    | ProductComponent::MaterializedZeroMinRepeatSuffix { .. },
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
    pub(crate) synthesized_expression: Expr,
}

struct PreparedTerminalExpressionPair {
    synthesized: Regex,
    full: DeferredDfa,
    full_to_synthesized: Vec<u32>,
    synthesized_expression: Expr,
}

/// Return the number of structurally aligned product components available to
/// the paired terminal compiler without constructing either DFA.
///
/// Vocabulary-only repeat reductions currently rely on the explicit
/// component maps below. A single-component pair would instead require the
/// generic finite-horizon equivalence search, which is exact but can create a
/// large and input-sensitive planning cliff. Callers may therefore use this as
/// a cheap fail-closed capability check before selecting a new optimization
/// candidate.
pub(crate) fn structural_pair_component_count(
    full_expression: &Expr,
    synthesized_expression: &Expr,
) -> Option<usize> {
    let normalized_full = lift_single_nested_intersection(full_expression);
    let normalized_synthesized = lift_single_nested_intersection(synthesized_expression);
    let full_expression = normalized_full.as_ref().unwrap_or(full_expression);
    let synthesized_expression = normalized_synthesized
        .as_ref()
        .unwrap_or(synthesized_expression);
    let full_plan = build_exclusion_compile_plan(std::slice::from_ref(full_expression));
    let synthesized_plan =
        build_exclusion_compile_plan(std::slice::from_ref(synthesized_expression));
    (full_plan.visible_groups == 1
        && synthesized_plan.visible_groups == 1
        && full_plan.compiled_exprs.len() == synthesized_plan.compiled_exprs.len()
        && full_plan.exclusions == synthesized_plan.exclusions
        && full_plan.intersections == synthesized_plan.intersections
        && !full_plan.compiled_exprs.is_empty())
    .then_some(full_plan.compiled_exprs.len())
}

fn prepare_terminal_expression_pair_with_structural_map(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<PreparedTerminalExpressionPair> {
    prepare_terminal_expression_pair_with_structural_map_inner(
        full_expression,
        synthesized_expression,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        true,
    )
}

fn prepare_terminal_expression_pair_with_structural_map_inner(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    allow_component_identity_fallback: bool,
) -> Option<PreparedTerminalExpressionPair> {
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
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_single_component_pair full_states={} synthesized_states={} depth={} path={} build_ms={:.3} map_ms={:.3}",
                full_dfa.num_states(),
                synthesized_dfa.num_states(),
                max_token_len,
                if used_homomorphism { "deterministic_homomorphism" } else { "moore" },
                product_build_ms,
                map_ms,
            );
        }
        return Some(PreparedTerminalExpressionPair {
            synthesized: Regex {
                dfa: synthesized_dfa,
            },
            full: DeferredDfa::Ready(full_dfa),
            full_to_synthesized,
            synthesized_expression: synthesized_expression.clone(),
        });
    }

    let product_build_started_at = profile.then(Instant::now);
    let ((full, full_trace, full_build_ms), (mut synthesized_dfa, synthesized_trace, synthesized_build_ms)) = rayon::join(
        || {
            let started_at = profile.then(Instant::now);
            let (full, trace) = match try_compile_with_plan_deferred_dense(full_plan) {
                Ok(prepared) => prepared,
                Err(full_plan) => {
                    let (dfa, trace) = compile_with_plan_internal(full_plan, true);
                    (DeferredDfa::Ready(dfa), trace)
                }
            };
            (
                full,
                trace,
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            )
        },
        || {
            let started_at = profile.then(Instant::now);
            let (dfa, trace) = compile_with_plan_internal(synthesized_plan, true);
            (
                dfa,
                trace,
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            )
        },
    );
    let product_build_ms = product_build_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair_product_build full_ms={:.3} synthesized_ms={:.3} wall_ms={:.3}",
            full_build_ms,
            synthesized_build_ms,
            product_build_ms,
        );
    }
    let full_dfa = full.dfa();
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
            let identical_mapping = (full == synthesized)
                .then(|| (0..full.num_states() as u32).collect::<Vec<_>>());
            let used_identical_mapping = identical_mapping.is_some();
            let homomorphism_mapping = if used_identical_mapping {
                None
            } else {
                deterministic_component_homomorphism_state_map(&full, &synthesized)
            };
            let used_homomorphism_mapping = homomorphism_mapping.is_some();
            if profile && !used_homomorphism_mapping && full.num_states() == synthesized.num_states() {
                let reachable_count = |dfa: &DFA| {
                    let mut seen = vec![false; dfa.num_states()];
                    let mut stack = vec![0u32];
                    while let Some(state) = stack.pop() {
                        if seen[state as usize] {
                            continue;
                        }
                        seen[state as usize] = true;
                        stack.extend(dfa.states()[state as usize].transitions.iter().map(|(_, &target)| target));
                    }
                    seen.into_iter().filter(|seen| *seen).count()
                };
                eprintln!(
                    "[glrmask/profile][tokenizer] same_size_component_homomorphism_failed expr_equal={} full_dfa_equal={} full_states={} synthesized_states={} full_reachable={} synthesized_reachable={} full_expr={:?} synthesized_expr={:?}",
                    full_expr == synthesized_expr,
                    full == synthesized,
                    full.num_states(),
                    synthesized.num_states(),
                    reachable_count(&full),
                    reachable_count(&synthesized),
                    expr_profile_summary(full_expr),
                    expr_profile_summary(synthesized_expr),
                );
            }
            let layered_mapping = if used_identical_mapping || used_homomorphism_mapping {
                None
            } else {
                direct_bounded_suffix_state_map(
                    full_expr,
                    synthesized_expr,
                    &full,
                    &synthesized,
                    max_token_len,
                    relevant_bytes,
                    Some(vocab),
                    Some(repeat_horizons),
                )
            };
            let dominance_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
            {
                None
            } else {
                zero_min_repeat_suffix_state_map(
                    full_expr,
                    synthesized_expr,
                    &full,
                    &synthesized,
                    full_component.zero_min_repeat_suffix_trace(),
                    synthesized_component.zero_min_repeat_suffix_trace(),
                    max_token_len,
                    Some(vocab),
                    Some(repeat_horizons),
                )
            };
            let unsafe_override_horizon = std::env::var(
                "GLRMASK_UNSAFE_STRUCTURAL_MAP_HORIZON",
            )
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&horizon| horizon < max_token_len);
            let override_layered_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
                || dominance_mapping.is_some()
            {
                None
            } else {
                unsafe_override_horizon.and_then(|horizon| {
                    direct_bounded_suffix_state_map(
                        full_expr,
                        synthesized_expr,
                        &full,
                        &synthesized,
                        horizon,
                        relevant_bytes,
                        None,
                        None,
                    )
                })
            };
            let override_dominance_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
                || dominance_mapping.is_some()
                || override_layered_mapping.is_some()
            {
                None
            } else {
                unsafe_override_horizon.and_then(|horizon| {
                    zero_min_repeat_suffix_state_map(
                        full_expr,
                        synthesized_expr,
                        &full,
                        &synthesized,
                        full_component.zero_min_repeat_suffix_trace(),
                        synthesized_component.zero_min_repeat_suffix_trace(),
                        horizon,
                        None,
                        None,
                    )
                })
            };
            let vocabulary_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
                || dominance_mapping.is_some()
                || override_layered_mapping.is_some()
                || override_dominance_mapping.is_some()
            {
                None
            } else {
                const MAX_LOCAL_FULL_STATES: usize = 4_096;
                const MAX_LOCAL_SYNTHESIZED_STATES: usize = 2_048;
                const MAX_LOCAL_PAIR_CELLS: usize = 2_000_000;
                (full.num_states() <= MAX_LOCAL_FULL_STATES
                    && synthesized.num_states() <= MAX_LOCAL_SYNTHESIZED_STATES
                    && full
                        .num_states()
                        .checked_mul(synthesized.num_states())
                        .is_some_and(|cells| cells <= MAX_LOCAL_PAIR_CELLS))
                .then(|| {
                    let full_tokenizer = Regex { dfa: full.clone() }.into_tokenizer(1, None);
                    let synthesized_tokenizer =
                        Regex { dfa: synthesized.clone() }.into_tokenizer(1, None);
                    certify_vocabulary_exact_state_candidates(
                        &full_tokenizer,
                        &synthesized_tokenizer,
                        vocab,
                        Some(&[true]),
                    )
                })
                .flatten()
            };
            let used_layered_mapping = layered_mapping.is_some();
            let used_dominance_mapping = dominance_mapping.is_some();
            let used_override_layered_mapping = override_layered_mapping.is_some();
            let used_override_dominance_mapping = override_dominance_mapping.is_some();
            let used_vocabulary_mapping = vocabulary_mapping.is_some();
            let mapping = if let Some(mapping) = identical_mapping {
                Some(ProductComponentStateMap::Fixed(mapping))
            } else if let Some(mapping) = homomorphism_mapping {
                Some(ProductComponentStateMap::Fixed(mapping))
            } else if let Some(mapping) = layered_mapping {
                Some(ProductComponentStateMap::Layered(mapping))
            } else if let Some(mapping) = dominance_mapping {
                Some(ProductComponentStateMap::Dominance(mapping))
            } else if let Some(mapping) = override_layered_mapping {
                Some(ProductComponentStateMap::Layered(mapping))
            } else if let Some(mapping) = override_dominance_mapping {
                Some(ProductComponentStateMap::Dominance(mapping))
            } else if let Some(mapping) = vocabulary_mapping {
                Some(ProductComponentStateMap::Vocabulary(mapping))
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
                    ProductComponent::Materialized(_)
                    | ProductComponent::MaterializedZeroMinRepeatSuffix { .. } => "materialized",
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
                    if used_identical_mapping {
                        "identical_dfa"
                    } else if used_homomorphism_mapping {
                        "deterministic_homomorphism"
                    } else if used_layered_mapping {
                        "layered_bounded_suffix"
                    } else if used_dominance_mapping {
                        "zero_min_repeat_suffix_dominance"
                    } else if used_override_layered_mapping {
                        "UNSAFE_layered_bounded_suffix_override"
                    } else if used_override_dominance_mapping {
                        "UNSAFE_zero_min_repeat_suffix_dominance_override"
                    } else if used_vocabulary_mapping {
                        "bounded_component_vocabulary_exact"
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
        .collect::<Vec<_>>();
    let failed_components = component_maps
        .iter()
        .enumerate()
        .filter_map(|(component, mapping)| mapping.is_none().then_some(component))
        .collect::<Vec<_>>();
    if !failed_components.is_empty() && allow_component_identity_fallback {
        let mut effective_components = synthesized_component_expressions.clone();
        for &component in &failed_components {
            effective_components[component] = full_component_expressions[component].clone();
        }
        let effective_expression = rebuild_single_visible_group_expression(
            &effective_components,
            &exclusions,
            &intersections,
        )?;
        if effective_expression == *synthesized_expression {
            return None;
        }
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_component_identity_fallback components={:?} full_states={} synthesized_states={} total_components={}",
                failed_components,
                full_dfa.num_states(),
                synthesized_dfa.num_states(),
                full_trace.components.len(),
            );
            if std::env::var_os("GLRMASK_PROFILE_FAILED_COMPONENT_EXPR").is_some() {
                for &component in &failed_components {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_component_identity_fallback_expr component={} full={:#?} synthesized={:#?}",
                        component,
                        full_component_expressions[component],
                        synthesized_component_expressions[component],
                    );
                }
            }
        }
        return prepare_terminal_expression_pair_with_structural_map_inner(
            full_expression,
            &effective_expression,
            vocab,
            repeat_horizons,
            max_token_len,
            relevant_bytes,
            false,
        );
    }
    let Some(component_maps) = component_maps.into_iter().collect::<Option<Vec<_>>>() else {
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
    let component_extents = synthesized_trace
        .components
        .iter()
        .map(|component| component.partition_dfa().num_states().saturating_add(1))
        .collect::<Vec<_>>();
    let dense_two_component_cells = (component_maps.len() == 2)
        .then(|| component_extents[0].checked_mul(component_extents[1]))
        .flatten()
        .filter(|&cells| cells <= DENSE_PRODUCT_LOOKUP_MAX_CELLS);

    let mut full_to_synthesized = if let Some(cells) = dense_two_component_cells {
        let right_extent = component_extents[1];
        let mut state_by_key = vec![u32::MAX; cells];
        match &synthesized_trace.state_tuples {
            ProductStateTuples::Generic(tuples) => {
                for (state, tuple) in tuples.iter().enumerate() {
                    let mut coordinates = [0usize; 2];
                    for &(component, component_state) in tuple {
                        coordinates[component as usize] = component_state as usize + 1;
                    }
                    state_by_key[coordinates[0] * right_extent + coordinates[1]] = state as u32;
                }
            }
            ProductStateTuples::DenseBinary(pairs) => {
                for (state, &(left, right)) in pairs.iter().enumerate() {
                    state_by_key[(left as usize + 1) * right_extent + right as usize + 1] =
                        state as u32;
                }
            }
        }
        let map_full_states = |full_states: [u32; 2]| {
                let mut coordinates = [0usize; 2];
                for component in 0..2 {
                    let full_state = full_states[component];
                    if full_state == u32::MAX {
                        continue;
                    }
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

                if full_states.iter().all(|&state| state != u32::MAX)
                    && component_maps.iter().all(ProductComponentStateMap::is_flexible)
                {
                    let mut left_candidates = SmallVec::<[u32; 8]>::new();
                    let mut right_candidates = SmallVec::<[u32; 8]>::new();
                    component_maps[0].visit_candidates(full_states[0], |candidate| {
                        left_candidates.push(candidate);
                        false
                    });
                    component_maps[1].visit_candidates(full_states[1], |candidate| {
                        right_candidates.push(candidate);
                        false
                    });
                    for &left in &left_candidates {
                        coordinates[0] = if synthesized_component_dead_states[0] == Some(left) {
                            0
                        } else {
                            left as usize + 1
                        };
                        for &right in &right_candidates {
                            coordinates[1] =
                                if synthesized_component_dead_states[1] == Some(right) {
                                    0
                                } else {
                                    right as usize + 1
                                };
                            let state = state_by_key
                                [coordinates[0] * right_extent + coordinates[1]];
                            if state != u32::MAX {
                                return state;
                            }
                        }
                    }
                }
                u32::MAX
            };
        match &full_trace.state_tuples {
            ProductStateTuples::Generic(tuples) => tuples
                .par_iter()
                .map(|tuple| {
                    let mut full_states = [u32::MAX; 2];
                    for &(component_id, full_state) in tuple {
                        full_states[component_id as usize] = full_state;
                    }
                    map_full_states(full_states)
                })
                .collect::<Vec<_>>(),
            ProductStateTuples::DenseBinary(pairs) => pairs
                .par_iter()
                .map(|&(left, right)| map_full_states([left, right]))
                .collect::<Vec<_>>(),
        }
    } else {
        let map_tuple = |tuple: &[(u32, u32)]| {
            let mut mapped = ProductStateTuple::new();
            for &(component_id, full_state) in tuple {
                let component = component_id as usize;
                let synthesized_state = component_maps[component].primary()[full_state as usize];
                if synthesized_component_dead_states[component] != Some(synthesized_state) {
                    mapped.push((component_id, synthesized_state));
                }
            }
            synthesized_trace.state_lookup.get(&mapped).unwrap_or(u32::MAX)
        };
        match &full_trace.state_tuples {
            ProductStateTuples::Generic(tuples) => {
                tuples.par_iter().map(|tuple| map_tuple(tuple)).collect()
            }
            ProductStateTuples::DenseBinary(pairs) => pairs
                .par_iter()
                .map(|&(left, right)| map_tuple(&[(0, left), (1, right)]))
                .collect(),
        }
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
            for &(component_id, full_state) in &full_trace.state_tuples.tuple(position) {
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
        full_to_synthesized[position] = synthesized_trace.state_lookup.get(tuple)?;
    }
    let lookup_ms = lookup_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let augmented_state_count = synthesized_dfa
        .num_states()
        .saturating_sub(states_before_augment);
    if std::env::var_os("GLRMASK_MINIMIZE_SYNTHETIC_PRODUCT").is_some()
        && augmented_state_count == 0
    {
        let minimize_started_at = Instant::now();
        let states_before = synthesized_dfa.num_states();
        let (minimized, old_to_new) = synthesized_dfa.minimize_with_state_mapping();
        for state in &mut full_to_synthesized {
            let mapped = old_to_new[*state as usize];
            if mapped == u32::MAX {
                return None;
            }
            *state = mapped;
        }
        synthesized_dfa = minimized;
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_minimize states_before={} states_after={} elapsed_ms={:.3}",
                states_before,
                synthesized_dfa.num_states(),
                minimize_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    } else if profile
        && std::env::var_os("GLRMASK_MINIMIZE_SYNTHETIC_PRODUCT").is_some()
    {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair_minimize_skipped reason=augmented_residual_roots augmented_states={}",
            augmented_state_count,
        );
    }

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair full_states={} synthesized_states_before={} synthesized_states_after={} mapped_tuples={} missing_before={} product_build_ms={:.3} component_maps_ms={:.3} tuple_map_ms={:.3} augment_ms={:.3} lookup_ms={:.3} total_ms={:.3}",
            full_dfa.num_states(),
            states_before_augment,
            synthesized_dfa.num_states(),
            full_to_synthesized.len(),
            missing_before,
            product_build_ms,
            component_maps_ms,
            tuple_map_ms,
            augment_ms,
            lookup_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some(PreparedTerminalExpressionPair {
        synthesized: Regex {
            dfa: synthesized_dfa,
        },
        full,
        full_to_synthesized,
        synthesized_expression: synthesized_expression.clone(),
    })
}

pub(crate) fn compile_terminal_expression_pair_with_structural_map(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledTerminalExpressionPair> {
    let prepared = prepare_terminal_expression_pair_with_structural_map(
        full_expression,
        synthesized_expression,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
    )?;
    Some(CompiledTerminalExpressionPair {
        synthesized: prepared.synthesized,
        full: Regex {
            dfa: prepared.full.finish(),
        },
        full_to_synthesized: prepared.full_to_synthesized,
        synthesized_expression: prepared.synthesized_expression,
    })
}

enum ProductComponentClassTransitions {
    Materialized(Vec<Vec<(u8, u32)>>),
    VirtualBoundedRepeat(Vec<Vec<(u8, u32)>>),
}

impl ProductComponent {
    fn partition_dfa(&self) -> &DFA {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => dfa,
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => base_dfa,
        }
    }

    fn dead_state(&self) -> Option<u32> {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                explicit_dead_sink_state(dfa)
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => explicit_dead_sink_state(base_dfa),
        }
    }
}

fn compile_product_component_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
) -> ProductComponent {
    if preserve_coordinates
        && let Some(trace) = zero_min_repeat_suffix_component_trace(expr)
    {
        let dfa = Arc::clone(&trace.dfa);
        return ProductComponent::MaterializedZeroMinRepeatSuffix {
            dfa,
            trace: Arc::new(trace),
        };
    }
    match expr {
        Expr::Shared(inner) => {
            compile_product_component_with_options(inner, preserve_coordinates)
        }
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

            ProductComponent::Materialized(Arc::new(
                compile_product_component_materialized_dfa_with_options(
                    expr,
                    preserve_coordinates,
                ),
            ))
        }
        _ => ProductComponent::Materialized(Arc::new(
            compile_product_component_materialized_dfa_with_options(
                expr,
                preserve_coordinates,
            ),
        )),
    }
}

fn compile_product_component(expr: &Expr) -> ProductComponent {
    compile_product_component_with_options(expr, false)
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
    preserve_coordinates: bool,
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
                let component =
                    compile_product_component_with_options(expr, preserve_coordinates);
                (
                    component,
                    Some(started_at.elapsed().as_secs_f64() * 1000.0),
                )
            } else {
                (
                    compile_product_component_with_options(expr, preserve_coordinates),
                    None,
                )
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
    let (components, cache_hits, _) =
        compile_product_components_profiled(exprs, false, false);
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
        compile_product_components_profiled(exprs, profile_detail, capture_trace);
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
    if direct_single_visible_group
        && pure_binary_intersection(exclusions, intersections)
        && let Some(result) = try_build_dense_binary_intersection_product(
            &components,
            &class_members,
            &component_class_transitions,
            capture_trace,
            profile_timing,
        )
    {
        return result;
    }
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
                    ProductComponent::Materialized(_)
                    | ProductComponent::MaterializedZeroMinRepeatSuffix { .. },
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
        state_tuples: ProductStateTuples::Generic(state_tuples),
        state_lookup: ProductStateLookup::Hash(state_map),
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
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
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

fn pure_binary_intersection(
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> bool {
    exclusions.is_empty()
        && intersections.len() == 1
        && intersections
            .get(&0)
            .is_some_and(|required| required.len() == 1 && required.contains(&1))
}

struct DeferredDenseBinaryIntersectionProduct {
    dfa: DFA,
    pending_class_transition_offsets: Vec<u32>,
    pending_class_transitions: Vec<(u8, u32)>,
    class_members: Vec<Vec<u8>>,
    left_states: usize,
    right_states: usize,
    pair_cells: usize,
    num_classes: usize,
    discovery_ms: f64,
    started_at: Option<Instant>,
}

impl DeferredDenseBinaryIntersectionProduct {
    fn dfa(&self) -> &DFA {
        &self.dfa
    }

    fn dfa_mut(&mut self) -> &mut DFA {
        &mut self.dfa
    }

    fn finish(mut self) -> DFA {
        let profile_timing = self.started_at.is_some();
        let future_started_at = profile_timing.then(Instant::now);
        set_single_group_futures_from_class_graph_csr(
            &mut self.dfa,
            &self.pending_class_transition_offsets,
            &self.pending_class_transitions,
        );
        let future_ms = future_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let expansion_started_at = profile_timing.then(Instant::now);
        let parallel_expansion = std::env::var("GLRMASK_PARALLEL_DEFERRED_BYTE_EXPANSION")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false);
        let dense_expansion_threshold = std::env::var(
            "GLRMASK_DEFERRED_DENSE_BYTE_EXPANSION_THRESHOLD",
        )
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
        let mut byte_to_class = [0u8; 256];
        for (class, bytes) in self.class_members.iter().enumerate() {
            for &byte in bytes {
                byte_to_class[byte as usize] = class as u8;
            }
        }
        let expand = |class_transitions: &[(u8, u32)], target_by_class: &mut [u32]| {
            let byte_capacity: usize = class_transitions
                .iter()
                .map(|(class, _)| self.class_members[*class as usize].len())
                .sum();
            let entries = if byte_capacity >= dense_expansion_threshold {
                target_by_class.fill(u32::MAX);
                for &(class, target) in class_transitions {
                    target_by_class[class as usize] = target;
                }
                let mut entries = Vec::with_capacity(byte_capacity);
                for (byte, &class) in byte_to_class.iter().enumerate() {
                    let target = target_by_class[class as usize];
                    if target != u32::MAX {
                        entries.push((byte as u8, target));
                    }
                }
                entries
            } else {
                let mut entries = Vec::with_capacity(byte_capacity);
                for &(class, target) in class_transitions {
                    for &byte in &self.class_members[class as usize] {
                        entries.push((byte, target));
                    }
                }
                if entries.len() > 1 {
                    entries.sort_unstable_by_key(|entry| entry.0);
                }
                entries
            };
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(entries)
        };
        let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
            if parallel_expansion {
                (0..self.dfa.num_states())
                    .into_par_iter()
                    .map_init(
                        || vec![u32::MAX; self.num_classes],
                        |target_by_class, state| {
                            let row_start = self.pending_class_transition_offsets[state] as usize;
                            let row_end =
                                self.pending_class_transition_offsets[state + 1] as usize;
                            expand(
                                &self.pending_class_transitions[row_start..row_end],
                                target_by_class,
                            )
                        },
                    )
                    .collect()
            } else {
                let mut target_by_class = vec![u32::MAX; self.num_classes];
                (0..self.dfa.num_states())
                    .map(|state| {
                        let row_start = self.pending_class_transition_offsets[state] as usize;
                        let row_end = self.pending_class_transition_offsets[state + 1] as usize;
                        expand(
                            &self.pending_class_transitions[row_start..row_end],
                            &mut target_by_class,
                        )
                    })
                    .collect()
            };
        for (state, transitions) in self.dfa.states_mut().iter_mut().zip(expanded_transitions) {
            state.transitions = transitions;
        }
        let expansion_ms = expansion_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(started_at) = self.started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] dense_binary_intersection left_states={} right_states={} pair_cells={} reachable_states={} classes={} discovery_ms={:.3} future_ms={:.3} expansion_ms={:.3} total_ms={:.3}",
                self.left_states,
                self.right_states,
                self.pair_cells,
                self.dfa.num_states(),
                self.num_classes,
                self.discovery_ms,
                future_ms,
                expansion_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        self.dfa
    }

    fn finish_compressed(mut self) -> (DFA, CompressedTransitionSegment) {
        let profile_timing = self.started_at.is_some();
        let future_started_at = profile_timing.then(Instant::now);
        set_single_group_futures_from_class_graph_csr(
            &mut self.dfa,
            &self.pending_class_transition_offsets,
            &self.pending_class_transitions,
        );
        let future_ms = future_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut byte_to_class = [0u8; 256];
        let class_members = self
            .class_members
            .into_iter()
            .enumerate()
            .map(|(class, bytes)| {
                for &byte in &bytes {
                    byte_to_class[byte as usize] = class as u8;
                }
                bytes.into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let expanded_transition_count = self
            .pending_class_transitions
            .iter()
            .map(|(class, _)| class_members[*class as usize].len())
            .sum();
        if let Some(started_at) = self.started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] dense_binary_intersection left_states={} right_states={} pair_cells={} reachable_states={} classes={} discovery_ms={:.3} future_ms={:.3} expansion_ms=0.000 compressed=true total_ms={:.3}",
                self.left_states,
                self.right_states,
                self.pair_cells,
                self.dfa.num_states(),
                self.num_classes,
                self.discovery_ms,
                future_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let segment = CompressedTransitionSegment {
            state_offset: 0,
            state_count: self.dfa.num_states() as u32,
            byte_to_class: Arc::from(byte_to_class.to_vec().into_boxed_slice()),
            class_members: Arc::from(class_members),
            row_offsets: Arc::from(self.pending_class_transition_offsets),
            entries: Arc::from(self.pending_class_transitions),
            expanded_transition_count,
        };
        (self.dfa, segment)
    }
}

fn try_discover_dense_binary_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    profile_timing: bool,
) -> Option<(
    DeferredDenseBinaryIntersectionProduct,
    Option<ProductBuildTrace>,
)> {
    const MAX_DENSE_PAIR_CELLS: usize = 40_000_000;

    if components.len() != 2 || component_class_transitions.len() != 2 {
        return None;
    }
    let left = components[0].materialized_dfa()?;
    let right = components[1].materialized_dfa()?;
    let left_states = left.num_states();
    let right_states = right.num_states();
    let pair_cells = left_states.checked_mul(right_states)?;
    if pair_cells == 0 || pair_cells > MAX_DENSE_PAIR_CELLS {
        return None;
    }
    let num_classes = class_members.len();
    let ProductComponentClassTransitions::Materialized(left_transitions) =
        &component_class_transitions[0]
    else {
        return None;
    };
    let ProductComponentClassTransitions::Materialized(right_transitions) =
        &component_class_transitions[1]
    else {
        return None;
    };
    let left_dead = components[0].dead_state();
    let right_dead = components[1].dead_state();

    let started_at = profile_timing.then(Instant::now);
    let discovery_started_at = profile_timing.then(Instant::now);
    let mut state_by_pair = vec![u32::MAX; pair_cells];
    state_by_pair[0] = 0;
    let mut pairs = vec![(0u32, 0u32)];
    let mut pending_class_transition_offsets = Vec::<u32>::new();
    pending_class_transition_offsets.push(0);
    let mut pending_class_transitions = Vec::<(u8, u32)>::new();
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    let set_metadata = |dfa: &mut DFA, state: u32, left_state: u32, right_state: u32| {
        let mut finalizers = BitSet::new(1);
        if left.finalizers(left_state).contains(0)
            && right.finalizers(right_state).contains(0)
        {
            finalizers.set(0);
        }
        dfa.overwrite_state_metadata(state, finalizers, BitSet::new(1));
    };
    set_metadata(&mut dfa, 0, 0, 0);

    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (left_state, right_state) = pairs[cursor];
        let left_row = &left_transitions[left_state as usize];
        let right_row = &right_transitions[right_state as usize];
        let mut left_index = 0usize;
        let mut right_index = 0usize;
        while left_index < left_row.len() && right_index < right_row.len() {
            let (left_class, left_target) = left_row[left_index];
            let (right_class, right_target) = right_row[right_index];
            if left_class < right_class {
                left_index += 1;
                continue;
            }
            if right_class < left_class {
                right_index += 1;
                continue;
            }
            left_index += 1;
            right_index += 1;
            if left_target == u32::MAX
                || right_target == u32::MAX
                || left_dead == Some(left_target)
                || right_dead == Some(right_target)
            {
                continue;
            }
            let pair_index = (left_target as usize)
                .checked_mul(right_states)?
                .checked_add(right_target as usize)?;
            let target = if state_by_pair[pair_index] != u32::MAX {
                state_by_pair[pair_index]
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_pair[pair_index] = target;
                pairs.push((left_target, right_target));
                let added = dfa.add_state();
                debug_assert_eq!(added, target);
                set_metadata(&mut dfa, target, left_target, right_target);
                target
            };
            pending_class_transitions.push((left_class, target));
        }
        pending_class_transition_offsets
            .push(u32::try_from(pending_class_transitions.len()).ok()?);
        cursor += 1;
    }
    let discovery_ms = discovery_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let trace = capture_trace.then(|| ProductBuildTrace {
            components: components.to_vec(),
            state_tuples: ProductStateTuples::DenseBinary(pairs),
            state_lookup: ProductStateLookup::DenseBinary {
                right_states,
                state_by_pair,
                overflow: FxHashMap::default(),
            },
            direct_single_visible_group: true,
    });

    Some((
        DeferredDenseBinaryIntersectionProduct {
            dfa,
            pending_class_transition_offsets,
            pending_class_transitions,
            class_members: class_members.to_vec(),
            left_states,
            right_states,
            pair_cells,
            num_classes,
            discovery_ms,
            started_at,
        },
        trace,
    ))
}

/// Exact dense compiler for a single visible terminal defined as the
/// intersection of two materialized deterministic components.
///
/// The ordinary product builder stores each reachable tuple in a hash table.
/// For a binary intersection every live product state is exactly one pair and
/// any transition with a dead component is dead for the logical terminal. A
/// dense `(left_state, right_state) -> product_state` table therefore provides
/// the same construction with constant-time array lookup and no tuple hashing.
fn try_build_dense_binary_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    profile_timing: bool,
) -> Option<(DFA, bool, Option<ProductBuildTrace>)> {
    let (deferred, trace) = try_discover_dense_binary_intersection_product(
        components,
        class_members,
        component_class_transitions,
        capture_trace,
        profile_timing,
    )?;
    Some((deferred.finish(), true, trace))
}

fn try_compile_with_plan_deferred_dense(
    plan: ExclusionCompilePlan,
) -> Result<(DeferredDfa, Option<ProductBuildTrace>), ExclusionCompilePlan> {
    if plan.visible_groups != 1
        || plan.compiled_exprs.len() != 2
        || !pure_binary_intersection(&plan.exclusions, &plan.intersections)
    {
        return Err(plan);
    }

    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let (components, component_cache_hits, _) =
        compile_product_components_profiled(&plan.compiled_exprs, profile_detail, true);
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] deferred_product_component_cache groups={} unique_components={} cache_hits={}",
            plan.compiled_exprs.len(),
            plan.compiled_exprs.len() - component_cache_hits,
            component_cache_hits,
        );
    }
    let (class_map, class_members) = compute_product_equivalence_classes(&components);
    let component_class_transitions = build_product_class_transitions(&components, &class_map);
    let Some((mut deferred, trace)) = try_discover_dense_binary_intersection_product(
        &components,
        &class_members,
        &component_class_transitions,
        true,
        profile_timing,
    ) else {
        return Err(plan);
    };

    deferred.dfa_mut().ensure_group_capacity(1);
    deferred
        .dfa_mut()
        .set_group_u8set(0, expr_u8set(&plan.compiled_exprs[0]));
    Ok((DeferredDfa::DenseBinary(deferred), trace))
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
    use crate::Vocab;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::{BTreeMap, BTreeSet};
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
            compressed_transition_segments: Arc::from([]),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            singleton_epsilon_closures: std::sync::OnceLock::new(),
        };
        let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
        exec.matches
            .iter()
            .any(|matched| matched.id == 0 && matched.width == input.len())
    }

    fn dfa_accepts(dfa: &DFA, input: &[u8]) -> bool {
        let mut state = 0u32;
        for &byte in input {
            let Some(next) = dfa.step(state, byte) else {
                return false;
            };
            state = next;
        }
        dfa.finalizers(state).contains(0)
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

    fn component_pair_for_map_test(
        full_states: usize,
        synthesized_states: usize,
        full_to_synthesized: Vec<u32>,
        protected_residual: bool,
    ) -> super::LexerComponentPair {
        super::LexerComponentPair {
            terminal_ids: vec![0],
            synthesized: DFA::new(synthesized_states),
            full: super::DeferredDfa::Ready(DFA::new(full_states)),
            full_to_synthesized,
            protected_residual,
        }
    }

    #[test]
    fn partitioned_component_maps_compose_identity_and_protected_offsets() {
        let components = vec![
            component_pair_for_map_test(2, 2, vec![0, 1], false),
            component_pair_for_map_test(3, 2, vec![0, 1, 1], true),
        ];
        let map = super::compose_partitioned_component_state_maps(&components, &[1, 3])
            .expect("complete component maps compose");

        assert_eq!(map, vec![0, 1, 2, 3, 4, 4]);
        assert_eq!(map.len(), 1 + 2 + 3);
    }

    #[test]
    fn partitioned_component_map_composition_rejects_incomplete_or_invalid_maps() {
        let incomplete = vec![component_pair_for_map_test(3, 2, vec![0, 1], true)];
        assert!(super::compose_partitioned_component_state_maps(&incomplete, &[1]).is_none());

        let invalid_target = vec![component_pair_for_map_test(2, 2, vec![0, 2], true)];
        assert!(
            super::compose_partitioned_component_state_maps(&invalid_target, &[1]).is_none()
        );

        let wrong_offset = vec![component_pair_for_map_test(2, 2, vec![0, 1], false)];
        assert!(super::compose_partitioned_component_state_maps(&wrong_offset, &[2]).is_none());
    }

    #[test]
    fn rebuilt_mixed_product_expression_preserves_group_operation_semantics() {
        let components = vec![byte_choice(b"ab"), byte_expr(b'b'), byte_expr(b'a')];
        let exclusions = BTreeMap::from([(0, BTreeSet::from([1]))]);
        let intersections = BTreeMap::from([(0, BTreeSet::from([2]))]);
        let rebuilt = super::rebuild_single_visible_group_expression(
            &components,
            &exclusions,
            &intersections,
        )
        .expect("single visible product reconstruction");

        for input in enumerate_inputs(b"ab", 2) {
            assert_eq!(terminal_matches(rebuilt.clone(), &input), input == b"a");
        }
    }

    #[test]
    fn terminal_structural_map_matches_generic_certifier_on_small_repeat() {
        let full_expression = Expr::Repeat {
            expr: Box::new(byte_expr(b'a')),
            min: 1,
            max: Some(64),
        };
        let synthesized_expression = Expr::Repeat {
            expr: Box::new(byte_expr(b'a')),
            min: 1,
            max: Some(6),
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let horizons = super::VocabularyRepeatHorizonCache::new();
        let pair = super::compile_terminal_expression_pair_with_structural_map(
            &full_expression,
            &synthesized_expression,
            &vocab,
            &horizons,
            vocab.max_token_byte_len(),
            &vocab.relevant_bytes(),
        )
        .expect("structural repeat map");
        let structural_map = pair.full_to_synthesized.clone();
        let full = pair.full.into_tokenizer(
            1,
            Some(Arc::from(vec![full_expression].into_boxed_slice())),
        );
        let synthesized = pair.synthesized.into_tokenizer(
            1,
            Some(Arc::from(
                vec![pair.synthesized_expression].into_boxed_slice(),
            )),
        );
        let generic = crate::compiler::stages::id_map_and_terminal_dwa::synthetic_state_map::certify_full_to_synthesized_state_map(
            &full,
            &synthesized,
            &vocab,
            Some(&[true]),
        )
        .expect("generic certification");

        assert_eq!(structural_map, generic.full_to_synthesized);
    }

    #[test]
    fn vocabulary_repeat_horizon_is_suffix_closed_and_residual_aware() {
        let body = Expr::U8Seq(b"ab".to_vec());
        let vocab = Vocab::new(vec![(0, b"xabab".to_vec())]);

        // The token suffix "abab" crosses two body boundaries from the body
        // start state. The analysis must consider that suffix even though it is
        // not itself a vocabulary entry.
        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(2),
        );

        let vocab = Vocab::new(vec![(0, b"babab".to_vec())]);
        // Starting from the reachable residual after an initial 'a', this token
        // completes one pending copy and two further copies.
        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(3),
        );
    }

    #[test]
    fn vocabulary_repeat_horizon_counts_boundaries_before_token_exit() {
        let body = Expr::U8Seq(b"a".to_vec());
        let vocab = Vocab::new(vec![(0, b"aaaaX".to_vec())]);

        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(4),
        );
    }

    #[test]
    fn vocabulary_repeat_horizon_respects_dominance_not_path_count() {
        let body = parse_regex("a+", false);
        let vocab = Vocab::new(vec![(0, b"aaaaaaaa".to_vec())]);

        // Although there are paths that choose a repeat boundary after every
        // byte, the zero-boundary path at the same residual dominates all of
        // them. Consequently an upper repetition counter is unobservable for
        // this body language and the translation displacement is exactly zero.
        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(0),
        );
    }

    #[test]
    fn dense_binary_intersection_matches_component_conjunction_exhaustively() {
        let cases = vec![
            (
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(byte_choice(b"ab")),
                        min: 1,
                        max: Some(4),
                    },
                    Expr::U8Seq(b"b".to_vec()),
                ]),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                    min: 0,
                    max: Some(3),
                },
            ),
            (
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab "))),
                    min: 0,
                    max: Some(5),
                },
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(byte_choice(b"ab")),
                        min: 0,
                        max: Some(3),
                    },
                    Expr::U8Seq(b" ".to_vec()),
                ]),
            ),
            (
                Expr::Choice(vec![
                    Expr::U8Seq(b"abba".to_vec()),
                    Expr::U8Seq(b"baab".to_vec()),
                    Expr::U8Seq(b" ".to_vec()),
                ]),
                Expr::Repeat {
                    expr: Box::new(byte_choice(b"ab ")),
                    min: 1,
                    max: Some(4),
                },
            ),
        ];
        let inputs = enumerate_inputs(b"ab ", 6);

        for (left_expr, right_expr) in cases {
            let components = vec![
                super::compile_product_component(&left_expr),
                super::compile_product_component(&right_expr),
            ];
            let left = components[0].partition_dfa().clone();
            let right = components[1].partition_dfa().clone();
            let (class_map, class_members) =
                super::compute_product_equivalence_classes(&components);
            let class_transitions =
                super::build_product_class_transitions(&components, &class_map);
            let (product, _, trace) = super::try_build_dense_binary_intersection_product(
                &components,
                &class_members,
                &class_transitions,
                true,
                false,
            )
            .expect("dense binary intersection");
            let trace = trace.expect("captured dense trace");
            assert_eq!(trace.state_tuples.len(), product.num_states());

            for input in &inputs {
                assert_eq!(
                    dfa_accepts(&product, input),
                    dfa_accepts(&left, input) && dfa_accepts(&right, input),
                    "left={left_expr:?} right={right_expr:?} input={input:?}",
                );
            }
        }
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
                None,
                None,
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
            compressed_transition_segments: Arc::from([]),
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
            compressed_transition_segments: Arc::from([]),
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
            compressed_transition_segments: Arc::from([]),
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
            compressed_transition_segments: Arc::from([]),
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
            compressed_transition_segments: Arc::from([]),
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
            compressed_transition_segments: Arc::from([]),
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
            compressed_transition_segments: Arc::from([]),
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
    fn deferred_dense_binary_intersection_matches_eager_product() {
        let bounded_string = |max| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                    min: 0,
                    max: Some(max),
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let expression = Expr::Intersect {
            expr: Box::new(bounded_string(31)),
            intersect: Box::new(bounded_string(17)),
        };

        let eager = super::compile_with_plan(super::build_exclusion_compile_plan(
            std::slice::from_ref(&expression),
        ));
        let (deferred, trace) = match super::try_compile_with_plan_deferred_dense(
            super::build_exclusion_compile_plan(std::slice::from_ref(&expression)),
        ) {
            Ok(prepared) => prepared,
            Err(_) => panic!("binary intersection should admit deferred dense construction"),
        };
        assert!(trace.is_some(), "deferred construction must retain its state tuples");
        let finished = deferred.finish();

        assert_eq!(finished, eager);
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
