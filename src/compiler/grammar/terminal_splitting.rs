//! Exact, cost-gated decomposition of expensive regular terminals.
//!
//! This pass never recognizes source regex spellings. It operates on resolved
//! [`Expr`] structure and currently uses one algebraic identity. For languages
//! `P`, `A`, and `S`, with `A` non-nullable, it rewrites
//!
//! ```text
//! P A{0,N} S
//! ```
//!
//! into a finite grammar over wide regular-language chunks. If `K` is the
//! selected block size, `N = qK + r`, and
//!
//! ```text
//! C = A^K
//! D = A{0,K-1} S
//! E = A{0,r} S
//! L = A{0,K-1} S
//! ```
//!
//! then the replacement alternatives are
//!
//! ```text
//! P L
//! P C^(m + 1) D     for 0 <= m < q - 1
//! P C^q E
//! ```
//!
//! Their body-count intervals are exactly `0..=N`. Prefix and suffix occur only
//! at the edges, and every repeated grammar terminal covers a whole `A^K`
//! chunk, so the pass never emits a repeated character-sized terminal. The
//! block size is chosen from the vocabulary's maximum token length and a cap on
//! the number of grammar alternatives, bounding both token-boundary depth and
//! grammar growth.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashSet;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::grammar::flat::{GrammarDef, NonterminalID, Rule, Symbol, Terminal, TerminalID};

const SPLIT_COMPLEX_TERMINALS_ENV: &str = "GLRMASK_SPLIT_COMPLEX_TERMINALS";
const SPLIT_BLOCK_SIZE_ENV: &str = "GLRMASK_SPLIT_COMPLEX_TERMINAL_BLOCK_SIZE";
const SPLIT_MIN_REPEAT_ENV: &str = "GLRMASK_SPLIT_COMPLEX_TERMINAL_MIN_REPEAT";
const SPLIT_MIN_SCORE_ENV: &str = "GLRMASK_SPLIT_COMPLEX_TERMINAL_MIN_SCORE";
const SPLIT_MAX_FULL_MIDDLES_ENV: &str =
    "GLRMASK_SPLIT_COMPLEX_TERMINAL_MAX_FULL_MIDDLES_PER_TOKEN";
const SPLIT_MAX_GROUPS_ENV: &str = "GLRMASK_SPLIT_COMPLEX_TERMINAL_MAX_GROUPS";
const SPLIT_FUSE_PREFIX_ENV: &str = "GLRMASK_SPLIT_COMPLEX_TERMINAL_FUSE_PREFIX";
const SPLIT_SUBSET_MAX_PAIRS_ENV: &str =
    "GLRMASK_SPLIT_COMPLEX_TERMINAL_SUBSET_MAX_STATE_PAIRS";
pub(crate) const GENERATED_TERMINAL_NAME_PREFIX: &str = "__glrmask_split_regular__";

const DEFAULT_ENABLED: bool = false;
const DEFAULT_MIN_REPEAT: usize = 24;
const DEFAULT_MIN_SCORE: usize = 4_096;
const DEFAULT_MAX_FULL_MIDDLES_PER_TOKEN: usize = 16;
const DEFAULT_MAX_GROUPS: usize = 32;
const DEFAULT_MIN_BLOCK_SIZE: usize = 8;
const DEFAULT_SUBSET_MAX_PAIRS: usize = 250_000;

#[derive(Debug, Clone, Copy)]
struct SplitConfig {
    enabled: bool,
    min_repeat: usize,
    min_score: usize,
    max_full_middles_per_token: usize,
    max_groups: usize,
    block_override: Option<usize>,
    fuse_prefix: bool,
    subset_max_pairs: usize,
}

impl SplitConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool(SPLIT_COMPLEX_TERMINALS_ENV, DEFAULT_ENABLED),
            min_repeat: env_usize(SPLIT_MIN_REPEAT_ENV).unwrap_or(DEFAULT_MIN_REPEAT).max(2),
            min_score: env_usize(SPLIT_MIN_SCORE_ENV).unwrap_or(DEFAULT_MIN_SCORE),
            max_full_middles_per_token: env_usize(SPLIT_MAX_FULL_MIDDLES_ENV)
                .unwrap_or(DEFAULT_MAX_FULL_MIDDLES_PER_TOKEN)
                .max(1),
            max_groups: env_usize(SPLIT_MAX_GROUPS_ENV)
                .unwrap_or(DEFAULT_MAX_GROUPS)
                .max(1),
            block_override: env_usize(SPLIT_BLOCK_SIZE_ENV).filter(|value| *value > 0),
            fuse_prefix: env_bool(SPLIT_FUSE_PREFIX_ENV, false),
            subset_max_pairs: env_usize(SPLIT_SUBSET_MAX_PAIRS_ENV)
                .unwrap_or(DEFAULT_SUBSET_MAX_PAIRS)
                .max(1),
        }
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(default)
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TerminalSplitProfile {
    pub(crate) candidate_terminals: usize,
    pub(crate) split_terminals: usize,
    pub(crate) generated_terminals: usize,
    pub(crate) generated_rules: usize,
    pub(crate) maximum_internal_path_bound: usize,
    pub(crate) minimum_block_size: usize,
    pub(crate) maximum_block_size: usize,
    pub(crate) certified_intersections: usize,
    pub(crate) subset_certificate_pairs: usize,
    pub(crate) subset_certificate_ms: f64,
}

#[derive(Debug, Clone)]
struct RepeatContext {
    prefix: Expr,
    body: Expr,
    max_repeat: usize,
    suffix: Expr,
    passthrough: Vec<Expr>,
}

#[derive(Debug, Clone)]
enum ExtractResult {
    None,
    One(RepeatContext),
    Multiple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubsetOutcome {
    Holds { pairs: usize },
    NotProved { pairs: usize },
    BudgetExceeded { pairs: usize },
}

#[derive(Default)]
struct SubsetCertificateCache {
    tokenizers: HashMap<Expr, Arc<Tokenizer>>,
    completed: HashMap<(Expr, Expr), bool>,
}

impl SubsetCertificateCache {
    fn tokenizer(&mut self, expr: &Expr) -> Arc<Tokenizer> {
        if let Some(tokenizer) = self.tokenizers.get(expr) {
            return Arc::clone(tokenizer);
        }
        let tokenizer = Arc::new(
            build_regex(std::slice::from_ref(expr)).into_tokenizer(1, None),
        );
        self.tokenizers
            .insert(expr.clone(), Arc::clone(&tokenizer));
        tokenizer
    }
}

impl SubsetOutcome {
    fn pairs(self) -> usize {
        match self {
            Self::Holds { pairs }
            | Self::NotProved { pairs }
            | Self::BudgetExceeded { pairs } => pairs,
        }
    }

    fn with_additional_pairs(self, additional: usize) -> Self {
        let pairs = self.pairs().saturating_add(additional);
        match self {
            Self::Holds { .. } => Self::Holds { pairs },
            Self::NotProved { .. } => Self::NotProved { pairs },
            Self::BudgetExceeded { .. } => Self::BudgetExceeded { pairs },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClosureProof {
    Holds,
    Unknown,
    BudgetExceeded,
}

#[derive(Debug, Clone)]
struct CandidateSelection<'a> {
    expr: Option<&'a Expr>,
    certified_intersections: usize,
    subset_pairs: usize,
    subset_ms: f64,
}

fn seq(parts: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::make_seq(parts.into_iter().collect())
}

fn choice(parts: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::make_choice(parts.into_iter().collect())
}

fn repeat(expr: Expr, min: usize, max: usize) -> Expr {
    if min == 1 && max == 1 {
        expr
    } else if min == 0 && max == 0 {
        Expr::Epsilon
    } else {
        Expr::Repeat {
            expr: Box::new(expr),
            min,
            max: Some(max),
        }
    }
}

fn shared(expr: Expr) -> Expr {
    match expr {
        Expr::Shared(_) => expr,
        other => Expr::Shared(Arc::new(other)),
    }
}

fn append_sequence_atoms(expr: &Expr, atoms: &mut Vec<Expr>) {
    match expr {
        Expr::Seq(parts) => {
            for part in parts {
                append_sequence_atoms(part, atoms);
            }
        }
        Expr::Shared(inner) => append_sequence_atoms(inner, atoms),
        Expr::Epsilon => {}
        other => atoms.push(other.clone()),
    }
}

/// Return a smaller sufficient inclusion problem by removing identical
/// concatenation context from both sides.
///
/// If `L(left_core) ⊆ L(right_core)`, then for every common `P` and `S`,
/// `L(P left_core S) ⊆ L(P right_core S)` by monotonicity of concatenation.
/// This is only a proof reduction: failure on the residual problem is not used
/// to claim that the original inclusion fails.
fn reduce_subset_problem(left: &Expr, right: &Expr) -> (Expr, Expr) {
    let mut left_atoms = Vec::new();
    let mut right_atoms = Vec::new();
    append_sequence_atoms(left, &mut left_atoms);
    append_sequence_atoms(right, &mut right_atoms);

    let mut prefix = 0usize;
    while prefix < left_atoms.len()
        && prefix < right_atoms.len()
        && left_atoms[prefix] == right_atoms[prefix]
    {
        prefix += 1;
    }

    let mut suffix = 0usize;
    while suffix < left_atoms.len().saturating_sub(prefix)
        && suffix < right_atoms.len().saturating_sub(prefix)
        && left_atoms[left_atoms.len() - 1 - suffix]
            == right_atoms[right_atoms.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let left_end = left_atoms.len().saturating_sub(suffix);
    let right_end = right_atoms.len().saturating_sub(suffix);
    (
        seq(left_atoms[prefix..left_end].iter().cloned()),
        seq(right_atoms[prefix..right_end].iter().cloned()),
    )
}

fn extract_repeat_context(expr: &Expr, min_repeat: usize) -> ExtractResult {
    match expr {
        Expr::Repeat {
            expr,
            min: 0,
            max: Some(max),
        } if *max >= min_repeat && !expr.is_nullable() => ExtractResult::One(RepeatContext {
            prefix: Expr::Epsilon,
            body: expr.as_ref().clone(),
            max_repeat: *max,
            suffix: Expr::Epsilon,
            passthrough: Vec::new(),
        }),
        Expr::Seq(parts) => {
            let mut found: Option<(usize, RepeatContext)> = None;
            for (index, part) in parts.iter().enumerate() {
                match extract_repeat_context(part, min_repeat) {
                    ExtractResult::None => {}
                    ExtractResult::Multiple => return ExtractResult::Multiple,
                    ExtractResult::One(context) => {
                        if found.is_some() {
                            return ExtractResult::Multiple;
                        }
                        found = Some((index, context));
                    }
                }
            }
            let Some((index, mut context)) = found else {
                return ExtractResult::None;
            };
            let before = seq(parts[..index].iter().cloned());
            let after = seq(parts[index + 1..].iter().cloned());
            context.prefix = seq([before.clone(), context.prefix]);
            context.suffix = seq([context.suffix, after.clone()]);
            context.passthrough = context
                .passthrough
                .into_iter()
                .map(|alternative| seq([before.clone(), alternative, after.clone()]))
                .collect();
            ExtractResult::One(context)
        }
        Expr::Choice(options) => {
            let mut context = None;
            let mut other_options = Vec::new();
            for option in options {
                match extract_repeat_context(option, min_repeat) {
                    ExtractResult::None => other_options.push(option.clone()),
                    ExtractResult::Multiple => return ExtractResult::Multiple,
                    ExtractResult::One(found) => {
                        if context.is_some() {
                            return ExtractResult::Multiple;
                        }
                        context = Some(found);
                    }
                }
            }
            let Some(mut context) = context else {
                return ExtractResult::None;
            };
            context.passthrough.extend(other_options);
            ExtractResult::One(context)
        }
        Expr::Shared(inner) => extract_repeat_context(inner, min_repeat),
        Expr::U8Seq(_)
        | Expr::U8Class(_)
        | Expr::Dfa(_)
        | Expr::Intersect { .. }
        | Expr::Exclude { .. }
        | Expr::Repeat { .. }
        | Expr::Epsilon => ExtractResult::None,
    }
}

/// Check one already-reduced inclusion problem by exact DFA-product search.
/// `NotProved` means the supplied reduced problem has a counterexample; callers
/// deliberately expose it only as absence of a positive certificate because
/// context cancellation is sound in the positive direction only.
fn certify_subset_exact(
    cache: &mut SubsetCertificateCache,
    left: &Expr,
    right: &Expr,
    max_pairs: usize,
) -> SubsetOutcome {
    if left == right {
        return SubsetOutcome::Holds { pairs: 0 };
    }
    let key = (left.clone(), right.clone());
    if let Some(&holds) = cache.completed.get(&key) {
        return if holds {
            SubsetOutcome::Holds { pairs: 0 }
        } else {
            SubsetOutcome::NotProved { pairs: 0 }
        };
    }

    let left = cache.tokenizer(left);
    let right = cache.tokenizer(right);
    let start = (left.start_state(), Some(right.start_state()));
    let mut queue = VecDeque::from([start]);
    let mut seen = FxHashSet::<(u32, Option<u32>)>::default();
    let mut right_rows = (0..right.num_states())
        .map(|_| None::<Box<[u32; 256]>>)
        .collect::<Vec<_>>();
    seen.insert(start);

    while let Some((left_state, right_state)) = queue.pop_front() {
        let left_accepts = left.matched_terminal_bitset(left_state).contains(0);
        let right_accepts = right_state
            .is_some_and(|state| right.matched_terminal_bitset(state).contains(0));
        if left_accepts && !right_accepts {
            cache.completed.insert(key, false);
            return SubsetOutcome::NotProved { pairs: seen.len() };
        }

        for (byte, next_left) in left.transitions_from(left_state) {
            let next_right = right_state.and_then(|state| {
                let row = right_rows[state as usize]
                    .get_or_insert_with(|| right.transition_row(state));
                let target = row[byte as usize];
                (target != u32::MAX).then_some(target)
            });
            if seen.insert((next_left, next_right)) {
                if seen.len() > max_pairs {
                    return SubsetOutcome::BudgetExceeded { pairs: seen.len() };
                }
                queue.push_back((next_left, next_right));
            }
        }
    }

    cache.completed.insert(key, true);
    SubsetOutcome::Holds { pairs: seen.len() }
}

/// Prove that `L(expr)` lies in the Kleene closure of `L(atom)`.
///
/// Concatenation and repetition preserve membership in a Kleene closure. For
/// every other expression form, use the exact DFA inclusion checker against
/// `atom*`. A negative result for one factor is only `Unknown`: surrounding
/// concatenation can still make the complete language a subset.
fn prove_within_kleene_star(
    cache: &mut SubsetCertificateCache,
    expr: &Expr,
    atom_star: &Expr,
    max_pairs: usize,
    pairs_used: &mut usize,
) -> ClosureProof {
    match expr {
        Expr::Epsilon => ClosureProof::Holds,
        Expr::Seq(parts) => {
            for part in parts {
                match prove_within_kleene_star(
                    cache,
                    part,
                    atom_star,
                    max_pairs,
                    pairs_used,
                ) {
                    ClosureProof::Holds => {}
                    other => return other,
                }
            }
            ClosureProof::Holds
        }
        Expr::Repeat {
            max: Some(0), ..
        } => ClosureProof::Holds,
        Expr::Repeat { expr, .. } => {
            prove_within_kleene_star(cache, expr, atom_star, max_pairs, pairs_used)
        }
        Expr::Shared(expr) => {
            prove_within_kleene_star(cache, expr, atom_star, max_pairs, pairs_used)
        }
        _ => {
            let remaining = max_pairs.saturating_sub(*pairs_used);
            if remaining == 0 {
                return ClosureProof::BudgetExceeded;
            }
            let (expr, atom_star) = reduce_subset_problem(expr, atom_star);
            let outcome = certify_subset_exact(cache, &expr, &atom_star, remaining);
            *pairs_used = pairs_used.saturating_add(outcome.pairs());
            match outcome {
                SubsetOutcome::Holds { .. } => ClosureProof::Holds,
                SubsetOutcome::NotProved { .. } => ClosureProof::Unknown,
                SubsetOutcome::BudgetExceeded { .. } => ClosureProof::BudgetExceeded,
            }
        }
    }
}

fn strip_shared(mut expr: &Expr) -> &Expr {
    while let Expr::Shared(inner) = expr {
        expr = inner;
    }
    expr
}

/// Try the compositional theorem `X ⊆ A*` (and, when non-nullable, `X ⊆ A+`).
fn certify_against_repetition(
    cache: &mut SubsetCertificateCache,
    left: &Expr,
    right: &Expr,
    max_pairs: usize,
) -> (Option<SubsetOutcome>, usize) {
    let Expr::Repeat {
        expr: atom,
        min,
        max: None,
    } = strip_shared(right)
    else {
        return (None, 0);
    };
    if *min > 1 {
        return (None, 0);
    }
    if *min == 1 && !atom.is_nullable() && left.is_nullable() {
        return (None, 0);
    }

    let atom_star = Expr::Repeat {
        expr: Box::new(atom.as_ref().clone()),
        min: 0,
        max: None,
    };
    let mut pairs_used = 0usize;
    match prove_within_kleene_star(
        cache,
        left,
        &atom_star,
        max_pairs,
        &mut pairs_used,
    ) {
        ClosureProof::Holds => (
            Some(SubsetOutcome::Holds {
                pairs: pairs_used,
            }),
            pairs_used,
        ),
        ClosureProof::BudgetExceeded => (
            Some(SubsetOutcome::BudgetExceeded {
                pairs: pairs_used,
            }),
            pairs_used,
        ),
        ClosureProof::Unknown => (None, pairs_used),
    }
}

/// Prove `L(left) ⊆ L(right)`, subject to a resource budget.
///
/// Every `Holds` result is a sound positive certificate. The common-context
/// reduction is intentionally one-way: a failed residual inclusion is reported
/// only as `NotProved`, because concatenation languages are not cancellative in
/// general. Otherwise the fallback walks the reachable product of deterministic
/// byte DFAs. Budget exhaustion is never interpreted as a positive certificate.
fn certify_subset(
    cache: &mut SubsetCertificateCache,
    left: &Expr,
    right: &Expr,
    max_pairs: usize,
) -> SubsetOutcome {
    let (left, right) = reduce_subset_problem(left, right);
    let (structural, pairs_used) =
        certify_against_repetition(cache, &left, &right, max_pairs);
    if let Some(outcome) = structural {
        return outcome;
    }
    let remaining = max_pairs.saturating_sub(pairs_used);
    if remaining == 0 {
        return SubsetOutcome::BudgetExceeded { pairs: pairs_used };
    }
    certify_subset_exact(cache, &left, &right, remaining).with_additional_pairs(pairs_used)
}

fn select_candidate_expression<'a>(
    cache: &mut SubsetCertificateCache,
    original: &'a Expr,
    min_repeat: usize,
    subset_max_pairs: usize,
) -> CandidateSelection<'a> {
    if matches!(extract_repeat_context(original, min_repeat), ExtractResult::One(_)) {
        return CandidateSelection {
            expr: Some(original),
            certified_intersections: 0,
            subset_pairs: 0,
            subset_ms: 0.0,
        };
    }

    let Expr::Intersect { expr, intersect } = original else {
        return CandidateSelection {
            expr: None,
            certified_intersections: 0,
            subset_pairs: 0,
            subset_ms: 0.0,
        };
    };

    let left_is_candidate = matches!(
        extract_repeat_context(expr, min_repeat),
        ExtractResult::One(_)
    );
    let right_is_candidate = matches!(
        extract_repeat_context(intersect, min_repeat),
        ExtractResult::One(_)
    );
    if !left_is_candidate && !right_is_candidate {
        return CandidateSelection {
            expr: None,
            certified_intersections: 0,
            subset_pairs: 0,
            subset_ms: 0.0,
        };
    }

    let started_at = Instant::now();
    let mut subset_pairs = 0usize;
    if left_is_candidate {
        let outcome = certify_subset(cache, expr, intersect, subset_max_pairs);
        subset_pairs = subset_pairs.saturating_add(outcome.pairs());
        if matches!(outcome, SubsetOutcome::Holds { .. }) {
            return CandidateSelection {
                expr: Some(expr),
                certified_intersections: 1,
                subset_pairs,
                subset_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            };
        }
    }
    if right_is_candidate {
        let remaining = subset_max_pairs.saturating_sub(subset_pairs).max(1);
        let outcome = certify_subset(cache, intersect, expr, remaining);
        subset_pairs = subset_pairs.saturating_add(outcome.pairs());
        if matches!(outcome, SubsetOutcome::Holds { .. }) {
            return CandidateSelection {
                expr: Some(intersect),
                certified_intersections: 1,
                subset_pairs,
                subset_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            };
        }
    }

    CandidateSelection {
        expr: None,
        certified_intersections: 0,
        subset_pairs,
        subset_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    }
}

fn expr_min_byte_len(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::U8Seq(bytes) => Some(bytes.len()),
        Expr::U8Class(_) => Some(1),
        Expr::Dfa(dfa) => dfa.min_match_byte_len(),
        Expr::Intersect { expr, intersect } => {
            Some(expr_min_byte_len(expr)?.max(expr_min_byte_len(intersect)?))
        }
        Expr::Seq(parts) => parts.iter().try_fold(0usize, |total, part| {
            total.checked_add(expr_min_byte_len(part)?)
        }),
        Expr::Choice(options) => options.iter().map(expr_min_byte_len).min().flatten(),
        Expr::Exclude { expr, .. } => expr_min_byte_len(expr),
        Expr::Repeat { expr, min, .. } => expr_min_byte_len(expr)?.checked_mul(*min),
        Expr::Shared(inner) => expr_min_byte_len(inner),
        Expr::Epsilon => Some(0),
    }
}

fn expr_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::U8Seq(bytes) => 1usize.saturating_add(bytes.len()),
        Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => 1,
        Expr::Intersect { expr, intersect } | Expr::Exclude { expr, exclude: intersect } => 1usize
            .saturating_add(expr_node_count(expr))
            .saturating_add(expr_node_count(intersect)),
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().fold(1usize, |total, part| {
            total.saturating_add(expr_node_count(part))
        }),
        Expr::Repeat { expr, .. } => {
            1usize.saturating_add(expr_node_count(expr))
        }
        Expr::Shared(expr) => {
            1usize.saturating_add(expr_node_count(expr))
        }
    }
}

fn terminal_expr(terminal: &Terminal) -> Option<Expr> {
    match terminal {
        Terminal::Literal { bytes, .. } => Some(Expr::U8Seq(bytes.clone())),
        Terminal::Pattern { pattern, utf8, .. } => Some(parse_regex(pattern, *utf8)),
        Terminal::Expr { expr, .. } => Some(expr.clone()),
        Terminal::SpecialToken { .. } => None,
    }
}

fn terminal_candidate_expr(terminal: &Terminal) -> Option<Cow<'_, Expr>> {
    match terminal {
        Terminal::Pattern { pattern, utf8, .. } => {
            Some(Cow::Owned(parse_regex(pattern, *utf8)))
        }
        Terminal::Expr { expr, .. } => Some(Cow::Borrowed(expr)),
        Terminal::Literal { .. } | Terminal::SpecialToken { .. } => None,
    }
}

fn maximum_token_len(vocab: &Vocab) -> usize {
    vocab
        .entries
        .values()
        .map(Vec::len)
        .max()
        .unwrap_or(0)
}

fn choose_block_size(
    max_repeat: usize,
    body_min_len: usize,
    max_token_len: usize,
    config: SplitConfig,
) -> Option<usize> {
    if let Some(block) = config.block_override {
        return (block <= max_repeat).then_some(block.max(2));
    }
    let path_threshold = max_token_len / (config.max_full_middles_per_token + 1);
    let path_block = path_threshold / body_min_len + 1;
    let group_block = max_repeat.div_ceil(config.max_groups);
    let block = path_block.max(group_block).max(DEFAULT_MIN_BLOCK_SIZE);
    (block <= max_repeat).then_some(block)
}

#[derive(Debug, Clone)]
struct CountedSplitPlan {
    passthrough: Vec<Expr>,
    prefix: Expr,
    body: Expr,
    suffix: Expr,
    max_repeat: usize,
    block: usize,
}

impl CountedSplitPlan {
    fn source_expr(&self) -> Expr {
        let counted = seq([
            self.prefix.clone(),
            repeat(self.body.clone(), 0, self.max_repeat),
            self.suffix.clone(),
        ]);
        choice(self.passthrough.iter().cloned().chain(std::iter::once(counted)))
    }

    fn counted_alternatives(&self, fuse_prefix: bool) -> Vec<Vec<Expr>> {
        let q = self.max_repeat / self.block;
        let r = self.max_repeat % self.block;
        debug_assert!(q >= 1);

        let low_body = seq([
            repeat(self.body.clone(), 0, self.block - 1),
            self.suffix.clone(),
        ]);
        let chunk = repeat(self.body.clone(), self.block, self.block);
        let low = if fuse_prefix {
            vec![seq([self.prefix.clone(), low_body])]
        } else {
            vec![self.prefix.clone(), low_body]
        };
        let first = if fuse_prefix {
            seq([self.prefix.clone(), chunk.clone()])
        } else {
            chunk.clone()
        };
        let middle = repeat(self.body.clone(), self.block, self.block);
        let full_tail = seq([
            repeat(self.body.clone(), 0, self.block - 1),
            self.suffix.clone(),
        ]);
        let final_tail = seq([
            repeat(self.body.clone(), 0, r),
            self.suffix.clone(),
        ]);

        let mut alternatives = vec![low];
        for middle_count in 0..q.saturating_sub(1) {
            let mut parts = Vec::with_capacity(middle_count + usize::from(!fuse_prefix) + 2);
            if !fuse_prefix {
                parts.push(self.prefix.clone());
            }
            parts.push(first.clone());
            parts.extend(std::iter::repeat_n(middle.clone(), middle_count));
            parts.push(full_tail.clone());
            alternatives.push(parts);
        }
        let mut final_parts = Vec::with_capacity(q + usize::from(!fuse_prefix) + 1);
        if !fuse_prefix {
            final_parts.push(self.prefix.clone());
        }
        final_parts.push(first);
        final_parts.extend(std::iter::repeat_n(middle, q - 1));
        final_parts.push(final_tail);
        alternatives.push(final_parts);
        alternatives
    }

    fn replacement_expr(&self, fuse_prefix: bool) -> Expr {
        choice(
            self.passthrough
                .iter()
                .cloned()
                .chain(self.counted_alternatives(fuse_prefix).into_iter().map(seq)),
        )
    }

    fn count_intervals(&self) -> Vec<(usize, usize)> {
        let q = self.max_repeat / self.block;
        let r = self.max_repeat % self.block;
        let mut intervals = vec![(0, self.block - 1)];
        for middle_count in 0..q.saturating_sub(1) {
            let start = self.block.saturating_mul(middle_count + 1);
            intervals.push((start, start + self.block - 1));
        }
        intervals.push((q * self.block, q * self.block + r));
        intervals
    }

    fn count_cover_is_exact(&self) -> bool {
        let intervals = self.count_intervals();
        let mut expected_start = 0usize;
        for (index, (start, end)) in intervals.iter().copied().enumerate() {
            if start != expected_start || start > end || end > self.max_repeat {
                return false;
            }
            if end == self.max_repeat {
                return index + 1 == intervals.len();
            }
            let Some(next) = end.checked_add(1) else {
                return false;
            };
            expected_start = next;
        }
        false
    }
}

fn split_family_partition(body: &Expr) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    body.hash(&mut hasher);
    format!("__split_regular_body_{:016x}", hasher.finish())
}

#[derive(Default)]
struct GeneratedTerminalRegistry {
    by_family_and_expr: HashMap<(String, Expr), TerminalID>,
}

impl GeneratedTerminalRegistry {
    fn register(
        &mut self,
        grammar: &mut GrammarDef,
        expr: Expr,
        partition: &str,
        source_name: &str,
        role: &str,
    ) -> TerminalID {
        let key = (partition.to_string(), expr.clone());
        if let Some(&terminal) = self.by_family_and_expr.get(&key) {
            return terminal;
        }
        let id = grammar.terminals.len() as TerminalID;
        grammar.terminals.push(Terminal::Expr {
            id,
            expr: expr.clone(),
        });
        grammar
            .terminal_names
            .insert(
                id,
                format!("{GENERATED_TERMINAL_NAME_PREFIX}{source_name}__{role}"),
            );
        grammar.lexer_partitions.insert(id, partition.to_string());
        self.by_family_and_expr.insert(key, id);
        id
    }
}

struct PlannedTerminalSplit {
    terminal_id: TerminalID,
    source_name: String,
    plan: CountedSplitPlan,
    body_min_len: usize,
}

fn split_with_config(
    grammar: &mut GrammarDef,
    vocab: &Vocab,
    config: SplitConfig,
) -> TerminalSplitProfile {
    if !config.enabled {
        return TerminalSplitProfile::default();
    }

    let max_token_len = maximum_token_len(vocab);
    let original_terminal_count = grammar.terminals.len();
    let original_ignore = grammar.ignore_terminal;
    let mut profile = TerminalSplitProfile::default();
    let mut subset_cache = SubsetCertificateCache::default();
    let mut planned = Vec::<PlannedTerminalSplit>::new();

    for terminal in grammar.terminals.iter().take(original_terminal_count) {
        let terminal_id = terminal.id();
        if original_ignore == Some(terminal_id) {
            continue;
        }
        let Some(original_expr) = terminal_candidate_expr(terminal) else {
            continue;
        };
        let selection = select_candidate_expression(
            &mut subset_cache,
            original_expr.as_ref(),
            config.min_repeat,
            config.subset_max_pairs,
        );
        profile.certified_intersections = profile
            .certified_intersections
            .saturating_add(selection.certified_intersections);
        profile.subset_certificate_pairs = profile
            .subset_certificate_pairs
            .saturating_add(selection.subset_pairs);
        profile.subset_certificate_ms += selection.subset_ms;
        let Some(candidate_expr) = selection.expr else {
            continue;
        };
        let ExtractResult::One(context) =
            extract_repeat_context(candidate_expr, config.min_repeat)
        else {
            continue;
        };
        profile.candidate_terminals += 1;

        let Some(prefix_min_len) = expr_min_byte_len(&context.prefix) else {
            continue;
        };
        let Some(body_min_len) = expr_min_byte_len(&context.body) else {
            continue;
        };
        let Some(suffix_min_len) = expr_min_byte_len(&context.suffix) else {
            continue;
        };
        if prefix_min_len == 0
            || body_min_len == 0
            || suffix_min_len == 0
            || context
                .passthrough
                .iter()
                .any(|alternative| expr_min_byte_len(alternative).unwrap_or(0) == 0)
        {
            continue;
        }
        let score = context
            .max_repeat
            .saturating_mul(expr_node_count(&context.body));
        if score < config.min_score {
            continue;
        }
        let Some(block) = choose_block_size(
            context.max_repeat,
            body_min_len,
            max_token_len,
            config,
        ) else {
            continue;
        };
        let plan = CountedSplitPlan {
            passthrough: context.passthrough,
            prefix: context.prefix,
            // Every generated chunk reuses the same potentially enormous body
            // expression. Preserve that structural identity explicitly instead
            // of deep-cloning the AST once per alternative.
            body: shared(context.body),
            suffix: context.suffix,
            max_repeat: context.max_repeat,
            block,
        };
        if !plan.count_cover_is_exact() {
            continue;
        }
        if profile.minimum_block_size == 0 {
            profile.minimum_block_size = block;
        } else {
            profile.minimum_block_size = profile.minimum_block_size.min(block);
        }
        profile.maximum_block_size = profile.maximum_block_size.max(block);

        let source_name = grammar.terminal_display_name(terminal_id);
        planned.push(PlannedTerminalSplit {
            terminal_id,
            source_name,
            plan,
            body_min_len,
        });
    }

    if planned.is_empty() {
        return profile;
    }

    let mut next_nonterminal = grammar.num_nonterminals();
    let mut registry = GeneratedTerminalRegistry::default();
    let mut replacements = BTreeMap::<TerminalID, NonterminalID>::new();
    let mut generated_rules = Vec::new();

    for planned_split in planned {
        let PlannedTerminalSplit {
            terminal_id,
            source_name,
            plan,
            body_min_len,
        } = planned_split;
        let partition = split_family_partition(&plan.body);
        let split_nonterminal = next_nonterminal;
        next_nonterminal += 1;
        grammar.nonterminal_names.insert(
            split_nonterminal,
            format!("{source_name}__split_regular"),
        );

        for (index, alternative) in plan.passthrough.iter().cloned().enumerate() {
            let terminal = registry.register(
                grammar,
                alternative,
                &partition,
                &source_name,
                &format!("passthrough_{index}"),
            );
            generated_rules.push(Rule {
                lhs: split_nonterminal,
                rhs: vec![Symbol::Terminal(terminal)],
            });
        }
        for (alternative_index, alternative) in
            plan.counted_alternatives(config.fuse_prefix)
                .into_iter()
                .enumerate()
        {
            let rhs = alternative
                .into_iter()
                .enumerate()
                .map(|(part_index, expr)| {
                    Symbol::Terminal(registry.register(
                        grammar,
                        expr,
                        &partition,
                        &source_name,
                        &format!("alt_{alternative_index}_part_{part_index}"),
                    ))
                })
                .collect();
            generated_rules.push(Rule {
                lhs: split_nonterminal,
                rhs,
            });
        }
        replacements.insert(terminal_id, split_nonterminal);
        profile.split_terminals += 1;
        let middle_min_len = plan.block.saturating_mul(body_min_len).max(1);
        profile.maximum_internal_path_bound = profile
            .maximum_internal_path_bound
            .max(max_token_len / middle_min_len + 2);
    }

    for rule in &mut grammar.rules {
        for symbol in &mut rule.rhs {
            if let Symbol::Terminal(terminal) = symbol {
                if let Some(&replacement) = replacements.get(terminal) {
                    *symbol = Symbol::Nonterminal(replacement);
                }
            }
        }
    }
    profile.generated_terminals = grammar
        .terminals
        .len()
        .saturating_sub(original_terminal_count);
    profile.generated_rules = generated_rules.len();
    grammar.rules.extend(generated_rules);
    profile
}

pub(crate) fn split_complex_terminals(
    grammar: &mut GrammarDef,
    vocab: &Vocab,
) -> TerminalSplitProfile {
    split_with_config(grammar, vocab, SplitConfig::from_env())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet, VecDeque};

    use super::*;
    use crate::automata::lexer::Lexer;
    use crate::automata::lexer::compile::build_regex;

    fn exact_language_equivalent(left: &Expr, right: &Expr) -> bool {
        let left = build_regex(std::slice::from_ref(left)).into_tokenizer(1, None);
        let right = build_regex(std::slice::from_ref(right)).into_tokenizer(1, None);
        let mut queue = VecDeque::from([(Some(left.start_state()), Some(right.start_state()))]);
        let mut seen = HashSet::new();
        while let Some((left_state, right_state)) = queue.pop_front() {
            if !seen.insert((left_state, right_state)) {
                continue;
            }
            let left_accepts = left_state
                .is_some_and(|state| left.matched_terminal_bitset(state).contains(0));
            let right_accepts = right_state
                .is_some_and(|state| right.matched_terminal_bitset(state).contains(0));
            if left_accepts != right_accepts {
                return false;
            }
            for byte in 0u8..=255 {
                let next_left = left_state.and_then(|state| left.step(state, byte));
                let next_right = right_state.and_then(|state| right.step(state, byte));
                if next_left.is_some() || next_right.is_some() {
                    queue.push_back((next_left, next_right));
                }
            }
        }
        true
    }

    fn exact_language_subset(left: &Expr, right: &Expr) -> bool {
        let left = build_regex(std::slice::from_ref(left)).into_tokenizer(1, None);
        let right = build_regex(std::slice::from_ref(right)).into_tokenizer(1, None);
        let mut queue = VecDeque::from([(Some(left.start_state()), Some(right.start_state()))]);
        let mut seen = HashSet::new();
        while let Some((left_state, right_state)) = queue.pop_front() {
            if !seen.insert((left_state, right_state)) {
                continue;
            }
            let left_accepts = left_state
                .is_some_and(|state| left.matched_terminal_bitset(state).contains(0));
            let right_accepts = right_state
                .is_some_and(|state| right.matched_terminal_bitset(state).contains(0));
            if left_accepts && !right_accepts {
                return false;
            }
            for byte in 0u8..=255 {
                let next_left = left_state.and_then(|state| left.step(state, byte));
                let next_right = right_state.and_then(|state| right.step(state, byte));
                if next_left.is_some() {
                    queue.push_back((next_left, next_right));
                }
            }
        }
        true
    }

    fn candidate(body: Expr, max_repeat: usize) -> Expr {
        seq([
            Expr::U8Seq(b"prefix:".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"empty".to_vec()),
                seq([
                    Expr::Repeat {
                        expr: Box::new(body),
                        min: 0,
                        max: Some(max_repeat),
                    },
                    Expr::U8Seq(b"tail".to_vec()),
                ]),
            ]),
            Expr::U8Seq(b";".to_vec()),
        ])
    }

    #[test]
    fn block_formula_covers_every_and_only_original_count() {
        for max_repeat in 2..=200 {
            for block in 2..=max_repeat {
                let plan = CountedSplitPlan {
                    passthrough: Vec::new(),
                    prefix: Expr::U8Seq(b"p".to_vec()),
                    body: Expr::U8Seq(b"a".to_vec()),
                    suffix: Expr::U8Seq(b"s".to_vec()),
                    max_repeat,
                    block,
                };
                assert!(plan.count_cover_is_exact(), "max={max_repeat} block={block}");
            }
        }
    }

    #[test]
    fn replacement_is_exact_for_non_code_body_language() {
        let original = candidate(
            Expr::Choice(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::U8Seq(b"aa".to_vec()),
            ]),
            7,
        );
        let ExtractResult::One(context) = extract_repeat_context(&original, 2) else {
            panic!("candidate should extract");
        };
        let plan = CountedSplitPlan {
            passthrough: context.passthrough,
            prefix: context.prefix,
            body: context.body,
            suffix: context.suffix,
            max_repeat: context.max_repeat,
            block: 3,
        };
        assert!(exact_language_equivalent(&original, &plan.source_expr()));
        assert!(exact_language_equivalent(
            &original,
            &plan.replacement_expr(true)
        ));
        assert!(exact_language_equivalent(
            &original,
            &plan.replacement_expr(false)
        ));
    }

    #[test]
    fn counted_replacement_matches_source_across_small_parameter_grid() {
        let bodies = [
            Expr::U8Seq(b"a".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::U8Seq(b"aa".to_vec()),
            ]),
            seq([
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                    min: 1,
                    max: Some(2),
                },
                Expr::U8Seq(b"y".to_vec()),
            ]),
        ];
        for body in bodies {
            for max_repeat in 2..=12 {
                let original = candidate(body.clone(), max_repeat);
                let ExtractResult::One(context) = extract_repeat_context(&original, 2) else {
                    panic!("candidate should extract");
                };
                for block in 2..=max_repeat {
                    let plan = CountedSplitPlan {
                        passthrough: context.passthrough.clone(),
                        prefix: context.prefix.clone(),
                        body: context.body.clone(),
                        suffix: context.suffix.clone(),
                        max_repeat,
                        block,
                    };
                    assert!(plan.count_cover_is_exact());
                    assert!(exact_language_equivalent(
                        &original,
                        &plan.replacement_expr(false)
                    ));
                    assert!(exact_language_equivalent(
                        &original,
                        &plan.replacement_expr(true)
                    ));
                }
            }
        }
    }

    #[test]
    fn every_positive_subset_certificate_matches_exact_reference() {
        let a = Expr::U8Seq(b"a".to_vec());
        let b = Expr::U8Seq(b"b".to_vec());
        let expressions = vec![
            Expr::Epsilon,
            a.clone(),
            b.clone(),
            choice([a.clone(), b.clone()]),
            seq([a.clone(), a.clone()]),
            choice([a.clone(), seq([a.clone(), a.clone()])]),
            Expr::Repeat {
                expr: Box::new(a.clone()),
                min: 0,
                max: None,
            },
            Expr::Repeat {
                expr: Box::new(a.clone()),
                min: 1,
                max: None,
            },
            Expr::Repeat {
                expr: Box::new(choice([a.clone(), b.clone()])),
                min: 0,
                max: None,
            },
            seq([
                Expr::Repeat {
                    expr: Box::new(a.clone()),
                    min: 0,
                    max: None,
                },
                Expr::Repeat {
                    expr: Box::new(b.clone()),
                    min: 0,
                    max: None,
                },
            ]),
            Expr::Repeat {
                expr: Box::new(choice([a.clone(), seq([a.clone(), a.clone()])])),
                min: 0,
                max: Some(3),
            },
        ];
        let mut positive = 0usize;
        for left in &expressions {
            for right in &expressions {
                let mut cache = SubsetCertificateCache::default();
                if matches!(
                    certify_subset(&mut cache, left, right, 100_000),
                    SubsetOutcome::Holds { .. }
                ) {
                    positive += 1;
                    assert!(
                        exact_language_subset(left, right),
                        "unsound positive certificate: left={left:?} right={right:?}"
                    );
                }
            }
        }
        assert!(positive > expressions.len());
    }

    #[test]
    fn transform_replaces_terminal_with_wide_nonterminal_alternatives() {
        let original = candidate(
            seq([
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                    min: 1,
                    max: None,
                },
                Expr::U8Seq(b" ".to_vec()),
            ]),
            9,
        );
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: original.clone(),
            }],
            nonterminal_names: BTreeMap::from([(0, "start".to_string())]),
            terminal_names: BTreeMap::from([(0, "complex".to_string())]),
            ignore_terminal: None,
            lexer_partitions: BTreeMap::new(),
        };
        let vocab = Vocab::new(vec![(0, b"prefix:xx xx tail;".to_vec())]);
        let profile = split_with_config(
            &mut grammar,
            &vocab,
            SplitConfig {
                enabled: true,
                min_repeat: 2,
                min_score: 0,
                max_full_middles_per_token: 2,
                max_groups: 6,
                block_override: Some(3),
                fuse_prefix: true,
                subset_max_pairs: 10_000,
            },
        );
        assert_eq!(profile.split_terminals, 1);
        assert!(matches!(grammar.rules[0].rhs.as_slice(), [Symbol::Nonterminal(_)]));
        assert!(grammar.rules.len() > 2);
        assert!(grammar.terminals.iter().skip(1).all(|terminal| {
            terminal_expr(terminal).is_some_and(|expr| expr_min_byte_len(&expr).unwrap_or(0) > 0)
        }));
    }

    #[test]
    fn split_future_terminal_does_not_leak_tokens_into_current_bounded_terminal() {
        fn bounded(prefix: u8, suffix: u8, max: usize) -> Expr {
            seq([
                Expr::U8Seq(vec![prefix]),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                    min: 0,
                    max: Some(max),
                },
                Expr::U8Seq(vec![suffix]),
            ])
        }

        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: bounded(b'a', b'b', 100),
                },
                Terminal::Expr {
                    id: 1,
                    expr: bounded(b'c', b'd', 199),
                },
            ],
            nonterminal_names: BTreeMap::from([(0, "start".to_string())]),
            terminal_names: BTreeMap::from([
                (0, "current_bounded".to_string()),
                (1, "future_split".to_string()),
            ]),
            ignore_terminal: None,
            lexer_partitions: BTreeMap::new(),
        };
        let mut first = b"a".to_vec();
        first.extend(std::iter::repeat_n(b'x', 80));
        let vocab = Vocab::new(vec![
            (0, first),
            (1, std::iter::repeat_n(b'x', 64).collect()),
            (2, b"b".to_vec()),
            (3, b"c".to_vec()),
            (4, b"d".to_vec()),
        ]);

        let baseline = crate::compiler::pipeline::compile_prepared(
            crate::compiler::grammar::transforms::prepare_grammar_transforms_only(
                grammar.clone(),
            ),
            &vocab,
        );
        let mut transformed = grammar;
        let profile = split_with_config(
            &mut transformed,
            &vocab,
            SplitConfig {
                enabled: true,
                min_repeat: 150,
                min_score: 0,
                max_full_middles_per_token: 2,
                max_groups: 6,
                block_override: Some(8),
                fuse_prefix: false,
                subset_max_pairs: 10_000,
            },
        );
        assert_eq!(profile.split_terminals, 1);
        let split = crate::compiler::pipeline::compile_prepared(
            crate::compiler::grammar::transforms::prepare_grammar_transforms_only(transformed),
            &vocab,
        );

        let mut baseline_state = baseline.start();
        let mut split_state = split.start();
        assert_eq!(baseline_state.mask(), split_state.mask());
        baseline_state.commit_token(0).unwrap();
        split_state.commit_token(0).unwrap();
        let baseline_mask = baseline_state.mask();
        let split_mask = split_state.mask();
        assert_eq!(baseline_mask, split_mask);
        assert_eq!(baseline_mask[0] & (1 << 1), 0);
    }

    #[test]
    fn split_future_json_string_does_not_leak_after_escaped_quote() {
        fn json_char() -> Expr {
            let raw = Expr::U8Class(crate::ds::u8set::U8Set::from_predicate(|byte| {
                byte >= 0x20 && byte != b'"' && byte != b'\\'
            }));
            let escape = seq([
                Expr::U8Seq(b"\\".to_vec()),
                Expr::U8Class(crate::ds::u8set::U8Set::from_bytes(b"\"\\/bfnrt")),
            ]);
            choice([raw, escape])
        }

        fn bounded_json_string(max: usize) -> Expr {
            seq([
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(json_char()),
                    min: 0,
                    max: Some(max),
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        }

        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: bounded_json_string(100),
                },
                Terminal::Expr {
                    id: 1,
                    expr: bounded_json_string(199),
                },
            ],
            nonterminal_names: BTreeMap::from([(0, "start".to_string())]),
            terminal_names: BTreeMap::from([
                (0, "current_json_string".to_string()),
                (1, "future_split_json_string".to_string()),
            ]),
            ignore_terminal: None,
            lexer_partitions: BTreeMap::new(),
        };
        let mut first = b"\"\\\",".to_vec();
        first.extend(std::iter::repeat_n(b'x', 38));
        let vocab = Vocab::new(vec![
            (0, first),
            (1, std::iter::repeat_n(b'_', 64).collect()),
            (2, b"\"".to_vec()),
        ]);

        let baseline = crate::compiler::pipeline::compile_prepared(
            crate::compiler::grammar::transforms::prepare_grammar_transforms_only(
                grammar.clone(),
            ),
            &vocab,
        );
        let mut transformed = grammar;
        let profile = split_with_config(
            &mut transformed,
            &vocab,
            SplitConfig {
                enabled: true,
                min_repeat: 150,
                min_score: 0,
                max_full_middles_per_token: 2,
                max_groups: 6,
                block_override: Some(8),
                fuse_prefix: false,
                subset_max_pairs: 10_000,
            },
        );
        assert_eq!(profile.split_terminals, 1);
        let split = crate::compiler::pipeline::compile_prepared(
            crate::compiler::grammar::transforms::prepare_grammar_transforms_only(transformed),
            &vocab,
        );

        let mut baseline_state = baseline.start();
        let mut split_state = split.start();
        assert_eq!(baseline_state.mask(), split_state.mask());
        baseline_state.commit_token(0).unwrap();
        split_state.commit_token(0).unwrap();
        let baseline_mask = baseline_state.mask();
        let split_mask = split_state.mask();
        assert_eq!(baseline_mask, split_mask);
        assert_eq!(baseline_mask[0] & (1 << 1), 0);
    }

    #[test]
    fn duplicate_generated_expr_across_families_preserves_partitions() {
        let first = candidate(Expr::U8Seq(b"a".to_vec()), 24);
        let second = candidate(Expr::U8Seq(b"b".to_vec()), 24);
        let mut grammar = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Expr { id: 0, expr: first },
                Terminal::Expr { id: 1, expr: second },
            ],
            nonterminal_names: BTreeMap::from([(0, "start".to_string())]),
            terminal_names: BTreeMap::from([(0, "first".to_string()), (1, "second".to_string())]),
            ignore_terminal: None,
            lexer_partitions: BTreeMap::new(),
        };
        let vocab = Vocab::new(vec![(0, b"prefix:empty;".to_vec())]);
        let profile = split_with_config(
            &mut grammar,
            &vocab,
            SplitConfig {
                enabled: true,
                min_repeat: 2,
                min_score: 0,
                max_full_middles_per_token: 16,
                max_groups: 32,
                block_override: Some(8),
                fuse_prefix: false,
                subset_max_pairs: 10_000,
            },
        );
        assert_eq!(profile.split_terminals, 2);
        let normalized =
            crate::compiler::grammar::transforms::prepare_grammar_transforms_only(grammar);
        let shared_passthroughs = normalized
            .terminals
            .iter()
            .filter(|terminal| {
                terminal_expr(terminal)
                    == Some(Expr::U8Seq(b"prefix:empty;".to_vec()))
            })
            .collect::<Vec<_>>();
        assert_eq!(shared_passthroughs.len(), 2);
        let partitions = shared_passthroughs
            .iter()
            .map(|terminal| {
                normalized
                    .lexer_partitions
                    .get(&terminal.id())
                    .expect("generated terminal should retain its partition")
            })
            .collect::<HashSet<_>>();
        assert_eq!(partitions.len(), 2);
    }

    #[test]
    fn intersections_are_not_split_without_a_compositional_certificate() {
        let expr = Expr::Intersect {
            expr: Box::new(candidate(Expr::U8Seq(b"a".to_vec()), 20)),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(crate::ds::u8set::U8Set::all())),
                min: 1,
                max: None,
            }),
        };
        assert!(matches!(
            extract_repeat_context(&expr, 2),
            ExtractResult::None
        ));
    }

    #[test]
    fn subset_certificate_is_exact_and_budgeted() {
        let mut cache = SubsetCertificateCache::default();
        let left = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"aa".to_vec()),
        ]);
        let right = Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
            min: 1,
            max: None,
        };
        assert!(matches!(
            certify_subset(&mut cache, &left, &right, 100),
            SubsetOutcome::Holds { .. }
        ));
        assert!(matches!(
            certify_subset(&mut cache, &right, &left, 100),
            SubsetOutcome::NotProved { .. }
        ));
        assert!(matches!(
            certify_subset(
                &mut cache,
                &Expr::Choice(vec![
                    Expr::U8Seq(b"a".to_vec()),
                    Expr::U8Seq(b"b".to_vec()),
                ]),
                &Expr::Choice(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 1,
                        max: None,
                    },
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min: 1,
                        max: None,
                    },
                ]),
                1,
            ),
            SubsetOutcome::BudgetExceeded { .. }
        ));
    }

    #[test]
    fn certified_redundant_intersection_can_be_split_exactly() {
        let candidate = candidate(
            Expr::Choice(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::U8Seq(b"aa".to_vec()),
            ]),
            7,
        );
        let original = Expr::Intersect {
            expr: Box::new(candidate.clone()),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(crate::ds::u8set::U8Set::all())),
                min: 1,
                max: None,
            }),
        };
        let mut cache = SubsetCertificateCache::default();
        let selection = select_candidate_expression(&mut cache, &original, 2, 10_000);
        assert_eq!(selection.certified_intersections, 1);
        let selected = selection.expr.expect("subset certificate should select candidate");
        assert!(exact_language_equivalent(&original, &selected));
        let ExtractResult::One(context) = extract_repeat_context(&selected, 2) else {
            panic!("certified side should remain splittable");
        };
        let plan = CountedSplitPlan {
            passthrough: context.passthrough,
            prefix: context.prefix,
            body: context.body,
            suffix: context.suffix,
            max_repeat: context.max_repeat,
            block: 3,
        };
        assert!(exact_language_equivalent(
            &original,
            &plan.replacement_expr(false)
        ));
    }
}
