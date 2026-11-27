use crate::runtime::markers::{CommitObserver, NoMarkers, LiveMarkers};
use crate::automata::lexer::Lexer;
pub(crate) mod profile;
pub(crate) mod bounded;
pub(crate) mod forcing_reference;
pub(crate) mod template_advance;
pub(crate) mod push_suffixes;
#[cfg(test)]
mod flat_template;
pub(crate) mod simple_read_shift;
pub(crate) mod push_dag;
pub(crate) mod phase_dag;
pub(crate) mod single_cursor;
pub(crate) mod template_prepare;
pub(crate) use template_advance::advance_stacks_template_dfa;
pub(crate) mod tokenizer_scan;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerExecResult, TokenizerMatch, TokenizerStateSet};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    ParserGSS,
    apply_guarded_stack_shifts_fast,
    advance_control_closed_stacks,
    advance_control_closed_stacks_owned,
    advance_stacks,
    advance_stacks_profiled,
    advance_stacks_owned,
    advance_stacks_disjoint_top_terminals_bounded,
    normalize_lookahead_invariant_reductions,
    AdvanceProfile,
    stack_may_advance_on,
    stack_may_advance_on_control_closed,
    stack_may_advance_on_any,
    stack_may_advance_on_any_control_closed,
    stack_may_advance_disjoint_top_terminals_bounded,
    stack_admissible_terminals,
};
use crate::compiler::glr::table::{Action, AdmissionPolicy, GLRTable};
use crate::compiler::glr::table::row::ActionRow;
use crate::runtime::artifact::DynamicMaskVocab;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::{
    CommitBuffers, ConstraintState, INLINE_PARSER_STATE_CAPACITY, LINEAR_STACK_RESERVE,
    ParserAdmissionCacheEntry, ParserRelativeMaskEqCacheEntry, ParserStateMap,
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use self::profile::{
    apply_advance_profile,
    CommitProfile,
    PerAdvanceEntry,
};
use self::template_advance::advance_stacks_template_dfa_owned;
pub(crate) use self::template_advance::TemplateAdvanceRuntime;
use self::tokenizer_scan::{
    execute_recursive_tokenizer_reusable, execute_tokenizer_from_state_small,
    execute_tokenizer_reusable, execute_tokenizer_reusable_from_states,
};

/// An offset bucket is usually one or two lexer lanes. Larger frontiers
/// promote to a retained hash map without changing their merge semantics.
#[derive(Debug)]
pub(crate) enum CommitBucket {
    Inline(SmallVec<[(u32, ParserGSS); 4]>),
    Hashed(FxHashMap<u32, ParserGSS>),
}

impl Default for CommitBucket {
    fn default() -> Self { Self::Inline(SmallVec::new()) }
}

impl CommitBucket {
    pub(crate) fn clear(&mut self) {
        match self {
            Self::Inline(rows) => rows.clear(),
            Self::Hashed(rows) => rows.clear(),
        }
    }

    fn merge(&mut self, key: u32, gss: ParserGSS) {
        if gss.is_empty() { return; }
        if let Self::Inline(rows) = self {
            if let Some((_, existing)) = rows.iter_mut().find(|(k, _)| *k == key) {
                *existing = existing.merge(&gss);
                return;
            }
            if rows.len() < 4 {
                rows.push((key, gss));
                return;
            }
            let mut hashed: FxHashMap<_, _> = std::mem::take(rows).into_iter().collect();
            hashed.insert(key, gss);
            *self = Self::Hashed(hashed);
        } else if let Self::Hashed(rows) = self {
            rows.entry(key).and_modify(|existing| *existing = existing.merge(&gss)).or_insert(gss);
        }
    }

    fn take_entries(&mut self) -> SmallVec<[(u32, ParserGSS); 4]> {
        match self {
            Self::Inline(rows) => std::mem::take(rows),
            Self::Hashed(rows) => rows.drain().collect(),
        }
    }
}

type ParserStatesByTokenizer = CommitBucket;

#[cold]
#[inline(never)]
fn try_terminal_observation_quotient_commit_reuse(
    buffers: &mut CommitBuffers,
    generation: u64,
    vocab: &DynamicMaskVocab,
    tokenizer: &Tokenizer,
    left_source: u32,
    right_source: u32,
    admitted: &crate::ds::bitset::BitSet,
    cache_left: u32,
    cache_right: u32,
    visited_nodes: usize,
    byte_steps: usize,
    started: Option<std::time::Instant>,
) -> bool {
    let Some(quotient_relevant) = vocab.terminal_observation_equivalent_for_live_admitted(
        left_source,
        right_source,
        admitted,
        tokenizer.matched_terminal_bitset(left_source),
        tokenizer.possible_future_terminals(left_source),
        tokenizer.matched_terminal_bitset(right_source),
        tokenizer.possible_future_terminals(right_source),
    ) else {
        return false;
    };

    if let Some(started) = started {
        eprintln!(
            "[glrmask/profile][parser_relative_commit_reuse] generation={} result=terminal_observation_quotient left={} right={} admitted={} relevant={} nodes={} bytes={} elapsed_us={:.1}",
            generation,
            left_source,
            right_source,
            admitted.count_ones(),
            quotient_relevant,
            visited_nodes,
            byte_steps,
            started.elapsed().as_secs_f64() * 1e6,
        );
    }

    const CACHE_CAPACITY: usize = 8;
    if buffers.parser_relative_mask_eq_cache.len() == CACHE_CAPACITY {
        buffers.parser_relative_mask_eq_cache.remove(0);
    }
    buffers
        .parser_relative_mask_eq_cache
        .push(ParserRelativeMaskEqCacheEntry {
            left: cache_left,
            right: cache_right,
            admitted: admitted.clone(),
        });
    true
}

#[cfg(test)]
const FLAT_ACTION_MAX_BRANCHES: usize = 16;
#[cfg(test)]
const FLAT_ACTION_MAX_STEPS: usize = 256;

#[cfg(test)]
type FlatInlineStack = SmallVec<[u32; LINEAR_STACK_RESERVE]>;

#[cfg(test)]
#[derive(Debug, Default)]
struct FlatActionScratch {
    pending: SmallVec<[FlatInlineStack; FLAT_ACTION_MAX_BRANCHES]>,
    complete: SmallVec<[FlatInlineStack; FLAT_ACTION_MAX_BRANCHES]>,
}

#[cfg(test)]
impl FlatActionScratch {
    fn clear(&mut self) {
        self.pending.clear();
        self.complete.clear();
    }

    fn push_pending(&mut self, stack: FlatInlineStack) -> bool {
        if self.pending.iter().any(|existing| existing == &stack) {
            return true;
        }
        if self.pending.len() == self.pending.capacity() {
            return false;
        }
        self.pending.push(stack);
        true
    }

    fn push_complete(&mut self, stack: FlatInlineStack) -> bool {
        if self.complete.iter().any(|existing| existing == &stack) {
            return true;
        }
        if self.complete.len() == self.complete.capacity() {
            return false;
        }
        self.complete.push(stack);
        true
    }
}
const SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX: usize = 16;

#[derive(Clone, Copy)]
struct NormalizedMatch {
    terminal_id: u32,
    width: usize,
    ignored: bool,
}

const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 256;
fn advance_parser_stacks(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if let Some(parser) = constraint.template_parser() {
        parser.record_advance();
        return advance_stacks_template_dfa(constraint, stack, terminal)
            .expect("complete template program must contain this terminal");
    }

    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        return advanced;
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(stack, terminal) {
        return cached;
    }
    if !constraint.parser_has_controls() {
        advance_stacks(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks(&constraint.table, stack, terminal)
    }
}

fn advance_parser_stacks_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if let Some(parser) = constraint.template_parser() {
        parser.record_advance();
        return advance_stacks_template_dfa_owned(constraint, stack, terminal)
            .expect("complete template program must contain this terminal");
    }

    if let Some(advanced) = constraint.advance_compact_segmented_parser(&stack, terminal) {
        return advanced;
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(&stack, terminal) {
        return cached;
    }
    if !constraint.parser_has_controls() {
        advance_stacks_owned(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks_owned(&constraint.table, stack, terminal)
    }
}

fn advance_parser_stacks_profiled(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> (ParserGSS, AdvanceProfile) {
    let started = std::time::Instant::now();
    let advanced = if constraint.has_template_parser()
        || constraint.uses_compact_segmented_parser_runtime()
        || constraint.parser_has_controls()
    {
        advance_parser_stacks(constraint, stack, terminal)
    } else if let Some(cached) = constraint.direct_regular_cached_advance(stack, terminal) {
        cached
    } else {
        return advance_stacks_profiled(&constraint.table, stack, terminal);
    };
    let elapsed = started.elapsed().as_nanos() as u64;
    (advanced, AdvanceProfile {
        total_ns: elapsed,
        fast_path_ns: elapsed,
        top_states: stack.top_value_count() as u32,
        gss_depth: stack.max_depth(),
        ..AdvanceProfile::default()
    })
}

/// Advance once when admission requires exact simulation. Row-presence tables
/// retain their cheap precheck; exact-simulation tables must not execute the
/// same reduction closure once for admission and again for the actual advance.
#[inline]
pub(crate) fn parser_may_advance_on(constraint: &Constraint, stack: &ParserGSS, terminal: u32) -> bool {
    if let Some(parser) = constraint.template_parser() {
        return parser.admits(stack, terminal);
    }

    if let Some(result) = constraint.compact_segmented_parser_may_advance_on(stack, terminal) {
        return result;
    }
    constraint
        .direct_regular_admissible_terminals(stack)
        .map_or_else(
            || {
                if !constraint.parser_has_controls() {
                    stack_may_advance_on(&constraint.table, stack, terminal)
                } else {
                    stack_may_advance_on_control_closed(&constraint.table, stack, terminal)
                }
            },
            |terminals| terminals.contains(terminal as usize),
        )
}

#[inline]
fn bitset_prefix_intersects(
    left: &crate::ds::bitset::BitSet,
    right: &crate::ds::bitset::BitSet,
) -> bool {
    left.words()
        .iter()
        .zip(right.words())
        .any(|(left, right)| (*left & *right) != 0)
}

#[inline]
fn bitset_union_intersection_prefix(
    dst: &mut crate::ds::bitset::BitSet,
    left: &crate::ds::bitset::BitSet,
    right: &crate::ds::bitset::BitSet,
) {
    for ((dst, left), right) in dst
        .words_mut()
        .iter_mut()
        .zip(left.words())
        .zip(right.words())
    {
        *dst |= *left & *right;
    }
}

/// Return the sole candidate in `actions` that is requested by `terminals`
/// and not already covered by the unconditional-advance row.  `Err(())`
/// means there are at least two candidates, so the single-terminal bounded
/// shortcut is not applicable.
#[inline]
fn single_conditional_candidate(
    actions: &ActionRow,
    unconditional: &crate::ds::bitset::BitSet,
    terminals: &crate::ds::bitset::BitSet,
) -> Result<Option<u32>, ()> {
    let mut selected = None;
    for (terminal, _action) in actions.iter() {
        let bit = terminal as usize;
        if bit >= terminals.len()
            || !terminals.contains(bit)
            || unconditional.contains(bit)
        {
            continue;
        }
        if selected.is_some() {
            return Err(());
        }
        selected = Some(terminal);
    }
    Ok(selected)
}

fn exact_simulation_prefilter_may_advance_on_any(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminals: &crate::ds::bitset::BitSet,
) -> Option<bool> {
    if constraint.has_template_parser() { return None; }

    if constraint.parser_admission_policy() != AdmissionPolicy::ExactSimulation
        || constraint.parser_has_controls()
        || constraint.table.unconditional_advance.len() != constraint.parser_symbol_count() as usize
    {
        return None;
    }

    let tops = stack.peek_values();
    let mut any_relevant = false;
    for &state in &tops {
        let advance = constraint.table.advance.get(state as usize)?;
        let unconditional = constraint.parser_unconditional_row(state)?;
        if bitset_prefix_intersects(unconditional, terminals) {
            return Some(true);
        }
        any_relevant |= bitset_prefix_intersects(advance, terminals);
    }
    if !any_relevant {
        return Some(false);
    }

    // The unconditional portion is already known empty.  Try the bounded
    // concrete-stack exact path when each current top has at most one remaining
    // candidate terminal; otherwise fall through to the general exact closure.
    let mut terminal_by_top = SmallVec::<[(u32, u32); 8]>::new();
    let mut bounded_applicable = true;
    'tops: for &top in &tops {
        let actions = constraint.table.action.get(top as usize)?;
        let unconditional = constraint.parser_unconditional_row(top)?;
        let selected = match single_conditional_candidate(actions, unconditional, terminals) {
            Ok(selected) => selected,
            Err(()) => {
                bounded_applicable = false;
                break 'tops;
            }
        };
        if let Some(terminal) = selected {
            terminal_by_top.push((top, terminal));
        }
    }
    if bounded_applicable
        && !terminal_by_top.is_empty()
        && let Some(result) = stack_may_advance_disjoint_top_terminals_bounded(
            &constraint.table,
            stack,
            &terminal_by_top,
        )
    {
        return Some(result);
    }
    Some(stack_may_advance_on_any(&constraint.table, stack, terminals))
}

#[inline]
fn parser_may_advance_on_any(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminals: &crate::ds::bitset::BitSet,
) -> bool {
    if let Some(parser) = constraint.template_parser() {
        return parser.admits_any(stack, terminals);
    }

    if let Some(result) = constraint.compact_segmented_parser_may_advance_on_any(stack, terminals) {
        return result;
    }
    if let Some(result) = exact_simulation_prefilter_may_advance_on_any(
        constraint,
        stack,
        terminals,
    ) {
        return result;
    }
    constraint
        .direct_regular_admissible_terminals(stack)
        .map_or_else(
            || {
                if !constraint.parser_has_controls() {
                    stack_may_advance_on_any(&constraint.table, stack, terminals)
                } else {
                    stack_may_advance_on_any_control_closed(&constraint.table, stack, terminals)
                }
            },
            |admitted| {
                admitted
                    .words()
                    .iter()
                    .zip(terminals.words())
                    .any(|(left, right)| (*left & *right) != 0)
            },
        )
}

pub(crate) fn advance_parser_stacks_if_possible(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if constraint.parser_admission_policy() == AdmissionPolicy::RowPresenceExact
        && !parser_may_advance_on(constraint, stack, terminal)
    {
        return None;
    }
    let advanced = advance_parser_stacks(constraint, stack, terminal);
    (!advanced.is_empty()).then_some(advanced)
}

/// Advance against the authoritative composed GLR table without consulting
/// parser-DWA/direct-regular admission caches.
///
/// Segmented boundary B deliberately represents language that is absent from
/// retained component/static A. After loading a hybrid, the constraint's
/// parser DWA may therefore be exactly A; using the ordinary admission
/// prefilter here would circularly reject B-only cross-component terminals.
pub(crate) fn advance_parser_stacks_table_exact(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if constraint.has_template_parser() {
        return advance_parser_stacks_if_possible(constraint, stack, terminal);
    }

    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        return (!advanced.is_empty()).then_some(advanced);
    }
    let advanced = if !constraint.parser_has_controls() {
        advance_stacks(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks(&constraint.table, stack, terminal)
    };
    (!advanced.is_empty()).then_some(advanced)
}

struct ProfiledAdvanceAttempt {
    advanced: ParserGSS,
    profile: AdvanceProfile,
    may_ns: u64,
    core_ns: u64,
}

fn advance_parser_stacks_profiled_if_possible(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ProfiledAdvanceAttempt {
    use std::time::Instant;

    let mut may_ns = 0;
    if constraint.parser_admission_policy() == AdmissionPolicy::RowPresenceExact {
        let may_started_at = Instant::now();
        let admitted = parser_may_advance_on(constraint, stack, terminal);
        may_ns = may_started_at.elapsed().as_nanos() as u64;
        if !admitted {
            return ProfiledAdvanceAttempt {
                advanced: ParserGSS::empty(),
                profile: AdvanceProfile::default(),
                may_ns,
                core_ns: 0,
            };
        }
    }

    let core_started_at = Instant::now();
    let (advanced, profile) = advance_parser_stacks_profiled(constraint, stack, terminal);
    let core_ns = core_started_at.elapsed().as_nanos() as u64;
    ProfiledAdvanceAttempt {
        advanced,
        profile,
        may_ns,
        core_ns,
    }
}

/// Cache for `advance_stacks` results, keyed by (GSS pointer, terminal).
/// Stores the key GSS alongside the result to keep its Arc alive and prevent
/// address reuse (ABA problem) within a single `commit_bytes_impl` call.
type AdvanceResultCache = FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>;

fn state_has_nonempty_accumulators(state: &ParserStateMap) -> bool {
    state
        .values()
        .any(|gss| !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
}


fn parser_stacks_only(gss: &ParserGSS) -> Vec<Vec<u32>> {
    gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit").into_iter().map(|(stack, _)| stack).collect()
}


const COMMIT_ASSERT_MASK_EQUIVALENCE: u8 = 1 << 0;
const COMMIT_ASSERT_FAST_PATH_EQUIVALENCE: u8 = 1 << 1;

fn commit_assertion_flags() -> u8 {
    static FLAGS: OnceLock<u8> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        let mut flags = 0;
        if cfg!(debug_assertions)
            || crate::env::var("GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE")
                .map(|value| {
                    let normalized = value.trim().to_ascii_lowercase();
                    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false)
        {
            flags |= COMMIT_ASSERT_MASK_EQUIVALENCE;
        }
        if crate::env::var("GLRMASK_ASSERT_COMMIT_FAST_PATH_EQUIVALENCE")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
        {
            flags |= COMMIT_ASSERT_FAST_PATH_EQUIVALENCE;
        }
        flags
    })
}

fn canonical_commit_state_for_equivalence_assert(
    state: &ParserStateMap,
) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
    let mut grouped = BTreeMap::<u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>>::new();
    for (&tokenizer_state, gss) in state.iter() {
        let out = grouped.entry(tokenizer_state).or_default();
        out.extend(
            gss.to_stacks(100_000)
                .expect("stack enumeration exceeded explicit limit")
                .into_iter()
                .map(|(stack, terminals_disallowed)| {
                    let disallowed = terminals_disallowed
                        .iter()
                        .map(|(&state, terminals)| {
                            (state, terminals.iter().copied().collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>();
                    (stack, disallowed)
                }),
        );
    }
    for stacks in grouped.values_mut() {
        stacks.sort();
        stacks.dedup();
    }
    grouped.into_iter().collect()
}

fn profile_allow_fast_paths() -> bool {
    crate::env::var("GLRMASK_PROFILE_ALLOW_FAST_PATHS")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub(crate) fn initialize_runtime_config() {
    let _ = commit_assertion_flags();
}

fn token_in_mask(mask: &[u32], token_id: u32) -> bool {
    let word_idx = token_id as usize / 32;
    let bit_idx = token_id as usize % 32;
    word_idx < mask.len() && ((mask[word_idx] >> bit_idx) & 1) != 0
}

fn snapshot_mask_membership(
    state: &ConstraintState,
    token_id: u32,
    assertion_flags: u8,
) -> Option<bool> {
    if assertion_flags & COMMIT_ASSERT_MASK_EQUIVALENCE == 0 {
        return None;
    }
    let mut mask = vec![0u32; state.constraint.mask_bit_len().div_ceil(32)];
    state.fill_mask(&mut mask);
    Some(token_in_mask(&mask, token_id))
}

fn format_token_bytes(token_bytes: &[u8]) -> String {
    let mut escaped = String::new();
    for byte in token_bytes {
        for ch in std::ascii::escape_default(*byte) {
            escaped.push(ch as char);
        }
    }
    format!("b\"{}\"", escaped)
}

fn format_optional_token_bytes(token_bytes: Option<&[u8]>) -> String {
    token_bytes
        .map(format_token_bytes)
        .unwrap_or_else(|| "<no vocabulary bytes>".to_owned())
}

#[inline]
fn assert_commit_oracles(
    constraint: &Constraint,
    token_id: u32,
    token_bytes: Option<&[u8]>,
    was_in_mask: Option<bool>,
    fast_path_reference: Option<ParserStateMap>,
    actual_state: &ParserStateMap,
    commit_succeeded: bool,
) {
    if let Some(was_in_mask) = was_in_mask {
        assert!(
            commit_succeeded == was_in_mask,
            "commit/mask mismatch for token_id {} bytes {}: token_in_mask={} commit_succeeded={}",
            token_id,
            format_optional_token_bytes(token_bytes),
            was_in_mask,
            commit_succeeded,
        );
    }
    if let Some(reference_state) = fast_path_reference {
        assert_commit_fast_path_equivalence(
            constraint,
            reference_state,
            token_id,
            actual_state,
            commit_succeeded,
        );
    }
}

// The read-only core of `advance_special_terminal_paths`: commit one
// zero-width special terminal from one reset-state entry, if it is admissible.
fn advance_single_special_terminal(
    constraint: &Constraint,
    tokenizer_state: u32,
    gss: &ParserGSS,
    terminal_id: u32,
) -> Option<ParserGSS> {
    let pruned = prune_single_initial_state_for_terminal(
        gss.clone(), tokenizer_state, terminal_id, None,
    );
    if pruned.is_empty() { return None; }
    advance_parser_stacks_if_possible(constraint, &pruned, terminal_id)
}

fn special_terminal_admissible(
    constraint: &Constraint,
    tokenizer_state: u32,
    gss: &ParserGSS,
    terminal_id: u32,
) -> bool {
    advance_single_special_terminal(constraint, tokenizer_state, gss, terminal_id).is_some()
}

pub(super) fn advance_special_terminal_paths(
    constraint: &Constraint,
    state: &ParserStateMap,
    terminal_id: u32,
) -> Option<ParserGSS> {
    let mut merged = None::<ParserGSS>;
    for (&initial_state, initial_gss) in state.iter() {
        if !runtime_tokenizer_is_reset_state(constraint, initial_state) { continue; }
        let Some(advanced) =
            advance_single_special_terminal(constraint, initial_state, initial_gss, terminal_id)
        else { continue; };
        merged = Some(match merged.take() {
            Some(existing) => existing.merge(&advanced),
            None => advanced,
        });
    }
    merged
}

pub(super) fn advance_special_token_paths(
    constraint: &Constraint,
    state: &ParserStateMap,
    token_id: u32,
) -> Option<ParserGSS> {
    let mut merged = None::<ParserGSS>;
    for special in constraint.special_token_terminals.iter().filter(|special| special.contains(token_id)) {
        let Some(advanced) = advance_special_terminal_paths(constraint, state, special.terminal_id)
            else { continue; };
        merged = Some(match merged.take() {
            Some(existing) => existing.merge(&advanced),
            None => advanced,
        });
    }
    merged
}

#[derive(Default)]
struct SpecialTokenAdvanceProfile {
    paths: Option<ParserGSS>,
    prune_ns: u64,
    may_check_ns: u64,
    advance_ns: u64,
    summary_ns: u64,
    advances: Vec<AdvanceProfile>,
}

fn advance_special_token_paths_profiled(
    constraint: &Constraint,
    state: &ParserStateMap,
    token_id: u32,
    mut per_advance: Option<&mut Vec<PerAdvanceEntry>>,
) -> SpecialTokenAdvanceProfile {
    use std::time::Instant;

    let mut result = SpecialTokenAdvanceProfile::default();

    for (&initial_state, initial_gss) in state.iter() {
        if !runtime_tokenizer_is_reset_state(constraint, initial_state) {
            continue;
        }
        for special in constraint
            .special_token_terminals
            .iter()
            .filter(|special| special.contains(token_id))
        {
            let prune_started_at = Instant::now();
            let pruned = prune_single_initial_state_for_terminal(
                initial_gss.clone(),
                initial_state,
                special.terminal_id,
                None,
            );
            result.prune_ns += prune_started_at.elapsed().as_nanos() as u64;
            if pruned.is_empty() {
                continue;
            }

            let attempt =
                advance_parser_stacks_profiled_if_possible(constraint, &pruned, special.terminal_id);
            result.may_check_ns += attempt.may_ns;
            result.advance_ns += attempt.core_ns;
            if attempt.advanced.is_empty() {
                continue;
            }
            let advanced = attempt.advanced;
            let advance_profile = attempt.profile;

            if let Some(entries) = per_advance.as_deref_mut() {
                result.summary_ns += record_per_advance_entry(
                    entries,
                    initial_state,
                    special.terminal_id,
                    &pruned,
                    &advanced,
                    0,
                    0,
                    0,
                    &[],
                    advance_profile.clone(),
                );
            }
            result.advances.push(advance_profile);
            result.paths = Some(match result.paths.take() {
                Some(existing) => existing.merge(&advanced),
                None => advanced,
            });
        }
    }

    result
}

fn apply_special_token_advance_profile(
    profile: &mut CommitProfile,
    special: &SpecialTokenAdvanceProfile,
) {
    profile.prune_ns += special.prune_ns;
    profile.advance_may_check_ns += special.may_check_ns;
    profile.may_advance_ns += special.may_check_ns;
    profile.advance_core_ns += special.advance_ns;
    profile.advance_ns += special.advance_ns;
    profile.adv_summary_ns += special.summary_ns;
    profile.n_advances += special.advances.len() as u64;
    for advance in &special.advances {
        apply_advance_profile(profile, advance);
    }
}

fn merge_special_token_paths(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    special_paths: Option<ParserGSS>,
) -> Result<(), String> {
    let Some(gss) = special_paths.filter(|gss| !gss.is_empty()) else {
        return Ok(());
    };
    if !constraint.uses_compact_segmented_parser_runtime() {
        state.merge_insert(constraint.runtime_commit_initial_state(), gss);
        return Ok(());
    }
    let partitions = constraint
        .partition_recursive_parser_gss_by_active_leaf(&gss)
        .ok_or_else(|| "recursive special-token reset routing failed".to_owned())?;
    for (leaf_index, partition) in partitions {
        let reset_state = constraint
            .recursive_tokenizer_reset_state(leaf_index)
            .ok_or_else(|| {
                format!("recursive special-token leaf {leaf_index} has no tokenizer reset")
            })?;
        state.merge_insert(reset_state, partition);
    }
    Ok(())
}

fn finish_token_commit(state: &ParserStateMap) -> Result<(), String> {
    if state.is_empty() {
        Err("commit rejected: no valid parser states remain".to_owned())
    } else {
        Ok(())
    }
}

#[inline]
fn commit_token_impl(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    buffers: &mut CommitBuffers,
    token_id: u32,
) -> Result<(), String> {
    commit_token_impl_observed(constraint, state, buffers, token_id, &mut NoMarkers)
}

fn commit_token_impl_observed<O: CommitObserver>(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    buffers: &mut CommitBuffers,
    token_id: u32,
    observer: &mut O,
) -> Result<(), String> {
    let bytes = constraint.token_bytes_for_id(token_id);
    let has_special = constraint.has_special_token_id(token_id);
    if bytes.is_none() && !has_special {
        return Err(format!(
            "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
        ));
    }
    let special_paths = has_special
        .then(|| advance_special_token_paths(constraint, state, token_id))
        .flatten();
    // Native, dynamic and composed masks admit an empty-spelled ID only through its
    // exact special-token action. Do not turn a rejected ordinary empty token
    // into a successful no-op commit; live special paths are merged below.
    let bytes = bytes.filter(|bytes| !bytes.is_empty()
        || !(constraint.has_template_parser() || constraint.uses_dynamic_runtime()
            || constraint.uses_compact_segmented_parser_runtime()));
    if let Some(bytes) = bytes {
        if commit_bytes_impl_observed(constraint, state, bytes, buffers, observer).is_err() {
            state.clear();
            buffers.reset_all();
        }
    } else {
        state.clear();
    }
    merge_special_token_paths(constraint, state, special_paths)?;
    if has_special { observer.frontier(constraint, state, bytes.map_or(0, |b| b.len())); }
    maybe_normalize_lookahead_invariant_reductions(constraint, state);
    finish_token_commit(state)
}

/// Exact one-token admission probe over an existing runtime state.
///
/// This is intentionally the ordinary authoritative commit implementation,
/// not a second recognizer. Recursive boundary-B masking uses it to validate
/// its small composition-owned candidate domain without translating the live
/// leaf-scoped tokenizer/parser state back into the transitional outer lexer
/// coordinate. `buffers` is reusable across probes; state-dependent caches are
/// reset between candidates while their allocated capacity is retained.
pub(crate) fn token_admissible_from_state_exact(
    constraint: &Constraint,
    source: &ParserStateMap,
    buffers: &mut CommitBuffers,
    token_id: u32,
) -> bool {
    buffers.reset_all();
    let mut probe = source.clone();
    commit_token_impl(constraint, &mut probe, buffers, token_id).is_ok()
}

/// Advance an exact runtime state through one byte prefix without committing a
/// model-token boundary. Recursive composition masking retains the returned
/// frontier at vocabulary-trie nodes and reuses it for all child edges.
pub(crate) fn advance_bytes_from_state_exact(
    constraint: &Constraint,
    source: &ParserStateMap,
    buffers: &mut CommitBuffers,
    bytes: &[u8],
) -> Option<ParserStateMap> {
    if bytes.is_empty() {
        return Some(source.clone());
    }
    buffers.reset_all();
    let mut next = source.clone();
    commit_bytes_impl(constraint, &mut next, bytes, buffers).ok()?;
    (!next.is_empty()).then_some(next)
}

/// Exact batched admission for ordinary byte-backed model tokens.
///
/// Candidates are sorted by byte string and evaluated as a radix tree.  Each
/// shared byte prefix is committed once through the ordinary authoritative
/// commit engine, then the resulting `ParserStateMap` is cheaply cloned for
/// its child branches.  A dead prefix rejects the complete subtree below it.
///
/// This helper deliberately handles byte semantics only.  Callers must keep
/// tokens with special-token semantics on `token_admissible_from_state_exact`,
/// because those tokens additionally fork parser paths that are not represented
/// by their byte spelling.
pub(crate) fn admissible_byte_token_candidates_from_state_exact(
    constraint: &Constraint,
    source: &ParserStateMap,
    buffers: &mut CommitBuffers,
    candidates: &mut Vec<(u32, &[u8])>,
    admitted: &mut Vec<u32>,
) {
    if candidates.is_empty() || source.is_empty() {
        return;
    }
    candidates.sort_unstable_by(|(left_id, left), (right_id, right)| {
        left.cmp(right).then_with(|| left_id.cmp(right_id))
    });

    fn walk(
        constraint: &Constraint,
        source: &ParserStateMap,
        buffers: &mut CommitBuffers,
        entries: &[(u32, &[u8])],
        consumed: usize,
        admitted: &mut Vec<u32>,
    ) {
        let Some((_, first)) = entries.first() else {
            return;
        };
        let last = entries.last().expect("nonempty radix candidate group").1;
        let mut common_end = consumed;
        let common_limit = first.len().min(last.len());
        while common_end < common_limit && first[common_end] == last[common_end] {
            common_end += 1;
        }

        let Some(frontier) = advance_bytes_from_state_exact(
            constraint,
            source,
            buffers,
            &first[consumed..common_end],
        ) else {
            return;
        };

        // Every candidate ending at this prefix has exactly the byte language
        // already consumed into `frontier`.  Ordinary token commit rejects iff
        // that frontier is empty; token-end normalization is non-destructive in
        // the recursive runtime, and special-token alternatives are excluded by
        // the caller.
        let mut index = 0usize;
        while index < entries.len() && entries[index].1.len() == common_end {
            admitted.push(entries[index].0);
            index += 1;
        }

        while index < entries.len() {
            debug_assert!(entries[index].1.len() > common_end);
            let branch_byte = entries[index].1[common_end];
            let mut end = index + 1;
            while end < entries.len()
                && entries[end].1.len() > common_end
                && entries[end].1[common_end] == branch_byte
            {
                end += 1;
            }
            walk(
                constraint,
                &frontier,
                buffers,
                &entries[index..end],
                common_end,
                admitted,
            );
            index = end;
        }
    }

    walk(constraint, source, buffers, candidates, 0, admitted);
}

#[cold]
pub(crate) fn prime_initial_commits(
    constraint: &Constraint,
    initial_state: &ParserStateMap,
    buffers: &mut CommitBuffers,
    token_ids: &[u32],
) {
    for &token_id in token_ids {
        let mut state = initial_state.clone();
        let _ = commit_token_impl(constraint, &mut state, buffers, token_id);
    }
}

#[cold]
fn commit_token_no_fast_path_reference(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    token_id: u32,
) -> Result<(), String> {
    let bytes = constraint.token_bytes_for_id(token_id);
    let has_special = constraint.has_special_token_id(token_id);
    if bytes.is_none() && !has_special {
        return Err(format!(
            "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
        ));
    }
    let special_paths = has_special
        .then(|| advance_special_token_paths(constraint, state, token_id))
        .flatten();
    if let Some(bytes) = bytes {
        let mut buffers = CommitBuffers::default();
        if commit_bytes_impl_profiled(
            constraint,
            state,
            bytes,
            &mut buffers,
            None,
            false,
        )
        .is_err()
        {
            state.clear();
        }
    } else {
        state.clear();
    }
    merge_special_token_paths(constraint, state, special_paths)?;
    finish_token_commit(state)
}

#[cold]
fn assert_commit_fast_path_equivalence(
    constraint: &Constraint,
    mut reference_state: ParserStateMap,
    token_id: u32,
    actual_state: &ParserStateMap,
    actual_succeeded: bool,
) {
    let reference_result =
        commit_token_no_fast_path_reference(constraint, &mut reference_state, token_id);
    assert_eq!(
        actual_succeeded,
        reference_result.is_ok(),
        "commit fast-path result mismatch for token_id {token_id}: actual_succeeded={} reference={:?}",
        actual_succeeded,
        reference_result,
    );
    let actual_canonical = canonical_commit_state_for_equivalence_assert(actual_state);
    let reference_canonical = canonical_commit_state_for_equivalence_assert(&reference_state);
    if actual_canonical != reference_canonical {
        // Bounded flat frontiers deliberately preserve lexer/parser
        // correlation that the map-only reference normalizes away. The fast
        // state can therefore be a strict internal refinement of the reference
        // without changing the accepted continuation language. Keep exact
        // state equality as the strongest oracle, but fall back to the public
        // semantic observations when the reference has merged correlations.
        let mut actual_semantic = constraint.start();
        actual_semantic.state = actual_state.clone();
        let mut reference_semantic = constraint.start();
        reference_semantic.state = reference_state;
        assert_eq!(
            actual_semantic.is_accepting(),
            reference_semantic.is_accepting(),
            "commit fast-path completion mismatch for token_id {token_id} bytes {}\nactual={actual_canonical:?}\nreference={reference_canonical:?}",
            format_optional_token_bytes(constraint.token_bytes_for_id(token_id)),
        );
        assert_eq!(
            actual_semantic.mask(),
            reference_semantic.mask(),
            "commit fast-path successor-mask mismatch for token_id {token_id} bytes {}\nactual={actual_canonical:?}\nreference={reference_canonical:?}",
            format_optional_token_bytes(constraint.token_bytes_for_id(token_id)),
        );
    }
}

#[inline]
pub(crate) fn runtime_tokenizer_is_reset_state(constraint: &Constraint, state: u32) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_tokenizer_is_reset_state(state)
    } else {
        state == constraint.runtime_commit_initial_state()
    }
}

#[inline]
fn runtime_tokenizer_future_terminals(
    constraint: &Constraint,
    state: u32,
) -> Option<Cow<'_, crate::ds::bitset::BitSet>> {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_tokenizer_future_scoped_terminals(state)
    } else {
        Some(Cow::Borrowed(
            constraint.tokenizer.possible_future_terminals(state),
        ))
    }
}

#[inline]
fn runtime_terminal_count(constraint: &Constraint) -> usize {
    constraint
        .uses_compact_segmented_parser_runtime()
        .then(|| constraint.recursive_runtime_terminal_count())
        .flatten()
        .unwrap_or(constraint.parser_terminal_count() as usize)
}

/// An unfinished ignore lexeme is a valid byte prefix even though completing
/// it will not shift a parser terminal. Lexer-continuation tests must retain
/// that route independently of the parser's ordinary terminal admission.
#[inline]
fn runtime_future_contains_ignore(
    constraint: &Constraint,
    future: &crate::ds::bitset::BitSet,
) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        future.iter_ones().any(|terminal| {
            constraint.recursive_terminal_is_ignore(terminal as u32)
        })
    } else {
        constraint.ignore_terminal.is_some_and(|terminal| future.contains(terminal as usize))
    }
}

#[inline]
fn end_state_may_advance(constraint: &Constraint, gss: &ParserGSS, end_state: u32) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    end_state_may_advance_exact(constraint, gss, end_state)
}

// `end_state_may_advance` without the reset-state shortcut: can any terminal
// the lexer still matches from `end_state` advance the parser?
#[inline]
fn end_state_may_advance_exact(constraint: &Constraint, gss: &ParserGSS, end_state: u32) -> bool {
    let future = runtime_tokenizer_future_terminals(constraint, end_state)
        .expect("runtime tokenizer end state must belong to its active coordinate");
    parser_may_advance_on_any(constraint, gss, future.as_ref())
        || runtime_future_contains_ignore(constraint, future.as_ref())
}

// Whether one frontier entry can still commit input: its lexer can extend, or
// a special token can be committed from it.
pub(crate) fn entry_can_continue(
    constraint: &Constraint,
    tokenizer_state: u32,
    gss: &ParserGSS,
) -> bool {
    if !runtime_tokenizer_is_reset_state(constraint, tokenizer_state) {
        // Commit admits a non-reset entry only after `end_state_may_advance`
        // and the exact-liveness post-pass, so it is already known to extend.
        return true;
    }
    if end_state_may_advance_exact(constraint, gss, tokenizer_state) {
        return true;
    }
    constraint.special_token_terminals.iter().any(|special| {
        !constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
            && special_terminal_admissible(constraint, tokenizer_state, gss, special.terminal_id)
    })
}

/// For a tokenizer execution that produced several end states against the same
/// parser GSS, compute exact parser admission once over the union of their
/// possible-future terminal sets.  Individual end-state viability is then an
/// intersection with this admitted set.
///
/// This is exactly equivalent to repeated `end_state_may_advance` because
///
///   exists t in F_i: can_advance(G, t)
///
/// iff `F_i` intersects the exact admitted set over `union_i F_i`.
///
/// Single non-initial end states deliberately keep the old boolean path: the
/// exact-set computation cannot beat its early-exit simulation in that case.
fn batched_end_state_admitted_terminals(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
) -> Option<crate::ds::bitset::BitSet> {
    // Template admission is an existential domain query: its exact boolean
    // path can stop at a suffix-independent witness. Computing the full set
    // here would classify unrelated conditional terminals before intersecting
    // each lexer future, defeating that shortcut.
    if constraint.has_template_parser() { return None; }
    let mut candidates = crate::ds::bitset::BitSet::new(runtime_terminal_count(constraint));
    let mut non_initial = 0usize;
    for &end_state in end_states {
        if runtime_tokenizer_is_reset_state(constraint, end_state) {
            continue;
        }
        non_initial += 1;
        let future = runtime_tokenizer_future_terminals(constraint, end_state)?;
        candidates.union_with_prefix(future.as_ref());
    }
    if non_initial <= 1 {
        return None;
    }
    if let Some(admitted) = constraint.compact_segmented_parser_admitted_terminals(gss, &candidates) {
        return Some(admitted);
    }
    if let Some(direct) = constraint.direct_regular_admissible_terminals(gss) {
        let mut admitted = candidates;
        admitted.intersect_with(&direct);
        Some(admitted)
    } else {
        Some(exact_simulation_prefiltered_admitted_terminals(
            constraint,
            gss,
            &candidates,
        ))
    }
}


const PARSER_ADMISSION_CACHE_CAPACITY: usize = 8;
const PARSER_ADMISSION_BOOLEAN_CACHE_CAPACITY: usize = 8;

fn exact_simulation_prefiltered_admitted_terminals(
    constraint: &Constraint,
    gss: &ParserGSS,
    candidates: &crate::ds::bitset::BitSet,
) -> crate::ds::bitset::BitSet {
    if let Some(parser) = constraint.template_parser() {
        return parser.admitted(gss, candidates);
    }

    if constraint.parser_admission_policy() != AdmissionPolicy::ExactSimulation
        || constraint.parser_has_controls()
        || constraint.table.unconditional_advance.len() != constraint.parser_symbol_count() as usize
    {
        return stack_admissible_terminals(&constraint.table, gss, candidates);
    }

    let mut guaranteed = crate::ds::bitset::BitSet::new(candidates.len());
    let mut unresolved = crate::ds::bitset::BitSet::new(candidates.len());
    for state in gss.peek_values() {
        let Some(advance) = constraint.table.advance.get(state as usize) else {
            return stack_admissible_terminals(&constraint.table, gss, candidates);
        };
        let Some(unconditional) = constraint.parser_unconditional_row(state) else {
            return stack_admissible_terminals(&constraint.table, gss, candidates);
        };
        bitset_union_intersection_prefix(&mut guaranteed, unconditional, candidates);
        bitset_union_intersection_prefix(&mut unresolved, advance, candidates);
    }
    for (unresolved_word, guaranteed_word) in unresolved
        .words_mut()
        .iter_mut()
        .zip(guaranteed.words())
    {
        *unresolved_word &= !*guaranteed_word;
    }
    if unresolved.is_empty() {
        return guaranteed;
    }
    let simulated = stack_admissible_terminals(&constraint.table, gss, &unresolved);
    guaranteed.union_with(&simulated);
    guaranteed
}

#[inline]
pub(crate) fn exact_admitted_terminals_for_candidates(
    constraint: &Constraint,
    gss: &ParserGSS,
    candidates: &crate::ds::bitset::BitSet,
) -> crate::ds::bitset::BitSet {
    if let Some(parser) = constraint.template_parser() {
        return parser.admitted(gss, candidates);
    }

    if let Some(admitted) = constraint.compact_segmented_parser_admitted_terminals(gss, candidates) {
        return admitted;
    }
    if let Some(direct) = constraint.direct_regular_admissible_terminals(gss) {
        let mut admitted = candidates.clone();
        admitted.intersect_with(&direct);
        admitted
    } else {
        exact_simulation_prefiltered_admitted_terminals(constraint, gss, candidates)
    }
}

fn admission_cache_entry_index(
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
    gss: &ParserGSS,
    terminal_count: usize,
) -> usize {
    if let Some(index) = cache
        .iter()
        .position(|entry| entry.gss.ptr_eq(gss))
    {
        return index;
    }
    if cache.len() >= PARSER_ADMISSION_CACHE_CAPACITY {
        cache.remove(0);
    }
    cache.push(ParserAdmissionCacheEntry {
        gss: gss.clone(),
        tested: crate::ds::bitset::BitSet::new(terminal_count),
        admitted: crate::ds::bitset::BitSet::new(terminal_count),
        boolean_queries: SmallVec::new(),
    });
    cache.len() - 1
}

fn try_local_row_presence_admission_words(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
) -> Option<[u64; 32]> {
    const WORDS: usize = 32;
    if constraint.uses_compact_segmented_parser_runtime()
        || constraint.parser_admission_policy() != AdmissionPolicy::ExactSimulation
        || constraint.parser_has_controls()
        || constraint.tokenizer.num_terminals() as usize > WORDS * 64
        || (!constraint.has_template_parser()
            && (constraint.table.advance.len() != constraint.parser_symbol_count() as usize
                || constraint.table.unconditional_advance.len()
                    != constraint.parser_symbol_count() as usize))
    {
        return None;
    }
    let tops = gss.peek_values();
    if tops.is_empty() {
        return None;
    }

    // Only terminals reachable from the tokenizer continuation matter. Both
    // backends provide a sound upper bound P and a suffix-independent lower
    // bound U. If (P - U) intersects none of these candidates, U is exact for
    // this query. Template rows are derived from the input-domain automata;
    // this shortcut neither reads an LR action nor materializes output stacks.
    let initial = constraint.runtime_commit_initial_state();
    let mut candidates = [0u64; WORDS];
    for &end_state in end_states {
        if end_state == initial {
            continue;
        }
        for (index, &word) in constraint
            .tokenizer
            .possible_future_terminals(end_state)
            .words()
            .iter()
            .enumerate()
        {
            if index >= WORDS {
                return None;
            }
            candidates[index] |= word;
        }
    }

    let mut admitted = [0u64; WORDS];
    for state in tops {
        let advance = constraint.parser_advance_row(state)?;
        let unconditional = constraint.parser_unconditional_row(state)?;
        for index in 0..advance.words().len().min(WORDS) {
            let candidate_word = candidates[index];
            if candidate_word == 0 {
                continue;
            }
            let advance_word = advance.words()[index];
            let unconditional_word = unconditional.words().get(index).copied().unwrap_or(0);
            if (advance_word & !unconditional_word & candidate_word) != 0 {
                return None;
            }
            admitted[index] |= unconditional_word & candidate_word;
        }
    }
    Some(admitted)
}

#[inline]
fn end_state_may_advance_from_row_words(
    constraint: &Constraint,
    end_state: u32,
    admitted_words: &[u64; 32],
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    let future = runtime_tokenizer_future_terminals(constraint, end_state)
        .expect("runtime tokenizer end state must belong to its active coordinate");
    future
        .words()
        .iter()
        .zip(admitted_words.iter())
        .any(|(&future, admitted)| (future & *admitted) != 0)
        || runtime_future_contains_ignore(constraint, future.as_ref())
}

/// Cached exact-set version of `batched_end_state_admitted_terminals`.
/// Returns the cache entry containing exact admission facts for every terminal
/// occurring in these end-state future sets. Single-end-state callers use the
/// cheaper boolean cache instead.
fn cached_batched_end_state_admission(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_states: &[u32],
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
) -> Option<usize> {
    if constraint.has_template_parser() { return None; }
    let mut candidates = crate::ds::bitset::BitSet::new(runtime_terminal_count(constraint));
    let mut non_initial = 0usize;
    for &end_state in end_states {
        if runtime_tokenizer_is_reset_state(constraint, end_state) {
            continue;
        }
        non_initial += 1;
        let future = runtime_tokenizer_future_terminals(constraint, end_state)?;
        candidates.union_with_prefix(future.as_ref());
    }
    if non_initial <= 1 {
        return None;
    }
    let index = admission_cache_entry_index(cache, gss, candidates.len());
    let delta = candidates.difference(&cache[index].tested);
    if !delta.is_empty() {
        let newly_admitted = exact_admitted_terminals_for_candidates(constraint, gss, &delta);
        let entry = &mut cache[index];
        entry.tested.union_with(&delta);
        entry.admitted.union_with(&newly_admitted);
    }
    Some(index)
}

/// Exact cached boolean admission for one tokenizer end state. If a prior
/// batched query has already classified all future terminals, answer directly
/// from the pointwise cache. Otherwise cache the old exact existential result
/// for this complete future set.
fn cached_single_end_state_may_advance(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_state: u32,
    cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    let Some(future) = runtime_tokenizer_future_terminals(constraint, end_state) else {
        return false;
    };
    let future = future.as_ref();
    let index = admission_cache_entry_index(cache, gss, future.len());
    {
        let entry = &cache[index];
        if future.is_subset_of_extended(&entry.tested) {
            return !future.is_disjoint(&entry.admitted)
                || runtime_future_contains_ignore(constraint, future);
        }
        if let Some((_, result)) = entry
            .boolean_queries
            .iter()
            .find(|(query, _)| query == future)
        {
            return *result;
        }
    }
    let result = parser_may_advance_on_any(constraint, gss, future)
        || runtime_future_contains_ignore(constraint, future);
    let entry = &mut cache[index];
    if entry.boolean_queries.len() >= PARSER_ADMISSION_BOOLEAN_CACHE_CAPACITY {
        entry.boolean_queries.remove(0);
    }
    entry.boolean_queries.push((future.clone(), result));
    result
}

#[inline]
fn end_state_may_advance_from_cache_entry(
    constraint: &Constraint,
    end_state: u32,
    entry: &ParserAdmissionCacheEntry,
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    let future = runtime_tokenizer_future_terminals(constraint, end_state)
        .expect("runtime tokenizer end state must belong to its active coordinate");
    !entry.admitted.is_disjoint_prefix(future.as_ref())
        || runtime_future_contains_ignore(constraint, future.as_ref())
}


#[inline]
fn end_state_may_advance_with_batch(
    constraint: &Constraint,
    gss: &ParserGSS,
    end_state: u32,
    admitted: Option<&crate::ds::bitset::BitSet>,
) -> bool {
    if runtime_tokenizer_is_reset_state(constraint, end_state) {
        return true;
    }
    match admitted {
        Some(admitted) => {
            let future = runtime_tokenizer_future_terminals(constraint, end_state)
                .expect("runtime tokenizer end state must belong to its active coordinate");
            !admitted.is_disjoint_prefix(future.as_ref())
                || runtime_future_contains_ignore(constraint, future.as_ref())
        }
        None => end_state_may_advance(constraint, gss, end_state),
    }
}

#[inline]
fn wide_frontier_end_state_may_advance(
    constraint: &Constraint,
    summary: &crate::runtime::artifact::DirectRegularWideFrontierAcceptance,
    end_state: u32,
) -> bool {
    if end_state == constraint.tokenizer.initial_state() {
        return true;
    }
    summary
        .actionable_terminals
        .words()
        .iter()
        .zip(constraint.tokenizer.possible_future_terminals(end_state).words())
        .any(|(actionable, future)| (*actionable & *future) != 0)
        || runtime_future_contains_ignore(constraint, constraint.tokenizer.possible_future_terminals(end_state))
}

enum ActionableTerminals {
    DirectDynamic(crate::ds::bitset::BitSet),
    SingleState(u32),
    WideFrontier(usize),
    ManyStates(SmallVec<[u32; 8]>),
}

impl ActionableTerminals {
    fn from_gss(constraint: &Constraint, gss: &ParserGSS) -> Option<Self> {
        if constraint.uses_compact_segmented_parser_runtime() {
            return None;
        }
        if let Some(terminals) = constraint.direct_regular_admissible_terminals(gss) {
            return Some(Self::DirectDynamic(terminals));
        }
        if let Some(state_id) = gss.single_top_value() {
            return Some(Self::SingleState(state_id));
        }
        if let Some(index) = constraint.direct_regular_wide_frontier_index_for_gss(gss) {
            return Some(Self::WideFrontier(index));
        }

        let states = gss.peek_values();
        if states.is_empty() {
            None
        } else {
            Some(Self::ManyStates(states))
        }
    }

    fn bitset<'a>(&'a self, constraint: &'a Constraint) -> Option<&'a crate::ds::bitset::BitSet> {
        match self {
            Self::DirectDynamic(terminals) => Some(terminals),
            Self::SingleState(state_id) => constraint.parser_advance_row(*state_id),
            Self::WideFrontier(index) => constraint
                .direct_regular_wide_frontier_acceptance
                .get(*index)
                .map(|summary| &summary.actionable_terminals),
            Self::ManyStates(_) => None,
        }
    }

    fn contains(&self, constraint: &Constraint, terminal: u32) -> bool {
        match self {
            Self::DirectDynamic(terminals) => terminals.contains(terminal as usize),
            Self::SingleState(state_id) => constraint.parser_advance_row_allows(*state_id, terminal),
            Self::WideFrontier(index) => constraint
                .direct_regular_wide_frontier_acceptance
                .get(*index)
                .is_some_and(|summary| summary.actionable_terminals.contains(terminal as usize)),
            Self::ManyStates(states) => states
                .iter()
                .any(|state_id| constraint.parser_advance_row_allows(*state_id, terminal)),
        }
    }
}

fn is_ignored_terminal(ignore_terminal: Option<u32>, terminal: u32) -> bool {
    Some(terminal) == ignore_terminal
}

#[inline]
fn runtime_is_ignored_terminal(
    constraint: &Constraint,
    ignore_terminal: Option<u32>,
    terminal: u32,
) -> bool {
    if constraint.uses_compact_segmented_parser_runtime() {
        constraint.recursive_terminal_is_ignore(terminal)
    } else {
        is_ignored_terminal(ignore_terminal, terminal)
    }
}

fn is_actionable_terminal(
    actionable_terminals: Option<&ActionableTerminals>,
    constraint: &Constraint,
    terminal: u32,
) -> bool {
    actionable_terminals.is_none_or(|actionable| actionable.contains(constraint, terminal))
}


fn for_each_relevant_matched_terminal(
    constraint: &Constraint,
    tokenizer_state: u32,
    actionable_terminals: Option<&ActionableTerminals>,
    mut visit: impl FnMut(u32, bool),
) {
    let matched = constraint.tokenizer.matched_terminal_bitset(tokenizer_state);
    if let Some(actionable) = actionable_terminals.and_then(|value| value.bitset(constraint)) {
        let ignored = constraint.ignore_terminal;
        for (word_index, (&matched_word, &actionable_word)) in
            matched.words().iter().zip(actionable.words()).enumerate()
        {
            let mut intersection = matched_word & actionable_word;
            while intersection != 0 {
                let bit = intersection.trailing_zeros() as usize;
                let terminal = (word_index * 64 + bit) as u32;
                if Some(terminal) != ignored {
                    visit(terminal, false);
                }
                intersection &= intersection - 1;
            }
        }
        if let Some(ignored) = ignored
            && matched.contains(ignored as usize)
        {
            visit(ignored, true);
        }
        return;
    }

    for terminal in constraint.tokenizer.matched_terminals_iter(tokenizer_state) {
        let ignored = is_ignored_terminal(constraint.ignore_terminal, terminal);
        if ignored || is_actionable_terminal(actionable_terminals, constraint, terminal) {
            visit(terminal, ignored);
        }
    }
}

fn scan_wide_frontier_lexer_only(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
    summary: &crate::runtime::artifact::DirectRegularWideFrontierAcceptance,
) -> Option<u32> {
    if constraint.tokenizer_has_epsilon_transitions {
        return None;
    }
    let mut state = start_state;
    for &byte in bytes {
        state = constraint.tokenizer_fast_transitions.transition(
            &constraint.tokenizer,
            state,
            byte,
        );
        if state == u32::MAX {
            return None;
        }
        let matched = constraint.tokenizer.matched_terminal_bitset(state);
        if matched
            .words()
            .iter()
            .zip(summary.actionable_terminals.words())
            .any(|(left, right)| (*left & *right) != 0)
            || constraint
                .ignore_terminal
                .is_some_and(|terminal| matched.contains(terminal as usize))
        {
            return None;
        }
    }
    Some(state)
}

fn advance_uniform_disallowed_interest_only(
    constraint: &Constraint,
    terminals_disallowed: &TerminalsDisallowed,
    bytes: &[u8],
) -> Option<TerminalsDisallowed> {
    if terminals_disallowed.is_empty() {
        return Some(TerminalsDisallowed::new());
    }
    if constraint.tokenizer_has_epsilon_transitions {
        return None;
    }

    let mut remapped = BTreeMap::<u32, BTreeSet<u32>>::new();
    for (&continuation_state, disallowed) in terminals_disallowed.iter() {
        let mut state = continuation_state;
        let mut alive = true;
        for &byte in bytes {
            state = constraint.tokenizer_fast_transitions.transition(
                &constraint.tokenizer,
                state,
                byte,
            );
            if state == u32::MAX {
                alive = false;
                break;
            }
            let matched = constraint.tokenizer.matched_terminal_bitset(state);
            if disallowed
                .iter()
                .any(|terminal| matched.contains(*terminal as usize))
            {
                return None;
            }
        }
        if !alive {
            continue;
        }
        let future = constraint.tokenizer.possible_future_terminals(state);
        for &terminal in disallowed {
            if future.contains(terminal as usize) {
                remapped.entry(state).or_default().insert(terminal);
            }
        }
    }
    Some(TerminalsDisallowed::from_map(remapped))
}

fn collect_unique_actionable_reusable_matches(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    ignore_terminal: Option<u32>,
    matches: &[TokenizerMatch],
) -> SmallVec<[NormalizedMatch; 16]> {
    // `execute_tokenizer_reusable*` canonicalizes to one record per terminal at
    // its longest width, so no second duplicate-removal pass is needed here.
    let mut normalized = SmallVec::<[NormalizedMatch; 16]>::new();
    for matched in matches {
        let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
        if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
            continue;
        }
        normalized.push(NormalizedMatch {
            terminal_id: matched.id,
            width: matched.width,
            ignored,
        });
    }
    normalized
}

fn collect_unique_actionable_matches(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    ignore_terminal: Option<u32>,
    matches: &[TokenizerMatch],
    reusable_seen_matches: Option<&mut FxHashSet<(usize, u32)>>,
) -> SmallVec<[NormalizedMatch; 16]> {
    let mut normalized = SmallVec::<[NormalizedMatch; 16]>::new();

    if matches.len() <= SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX {
        'matches: for matched in matches {
            let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            for existing in &normalized {
                if existing.width == matched.width && existing.terminal_id == matched.id {
                    continue 'matches;
                }
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    if let Some(seen_matches) = reusable_seen_matches {
        seen_matches.clear();
        for matched in matches {
            let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            if !seen_matches.insert((matched.width, matched.id)) {
                continue;
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    let mut seen_matches = FxHashSet::default();
    for matched in matches {
        let ignored = runtime_is_ignored_terminal(constraint, ignore_terminal, matched.id);
        if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
            continue;
        }
        if !seen_matches.insert((matched.width, matched.id)) {
            continue;
        }
        normalized.push(NormalizedMatch {
            terminal_id: matched.id,
            width: matched.width,
            ignored,
        });
    }
    normalized
}

fn prune_single_initial_state_for_exec(
    constraint: &Constraint,
    gss: ParserGSS,
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
    bytes: &[u8],
) -> ParserGSS {
    prune_single_initial_state_for_parts(
        constraint,
        gss,
        tokenizer_state,
        &exec_result.end_state,
        &exec_result.matches,
        bytes,
    )
}

fn advance_terminals_disallowed_over_bytes(
    constraint: &Constraint,
    terminals_disallowed: &TerminalsDisallowed,
    bytes: &[u8],
    reusable_execution: Option<(u32, &[u32], &[TokenizerMatch])>,
) -> Option<TerminalsDisallowed> {
    if terminals_disallowed.is_empty() {
        return Some(TerminalsDisallowed::new());
    }

    let mut remapped = BTreeMap::new();
    for (&continuation_tokenizer_state, disallowed) in terminals_disallowed.iter() {
        let owned_execution;
        let (end_states, matches) = match reusable_execution {
            Some((state, end_states, matches)) if state == continuation_tokenizer_state => {
                (end_states, matches)
            }
            _ => {
                owned_execution = execute_tokenizer_from_state_small(
                    constraint,
                    bytes,
                    continuation_tokenizer_state,
                );
                (&owned_execution.end_state[..], &owned_execution.matches[..])
            }
        };
        if matches.iter()
            .any(|matched| disallowed.contains(&matched.id))
        {
            return None;
        }
        for &end_state in end_states {
            let future = runtime_tokenizer_future_terminals(constraint, end_state)?;
            for &terminal in disallowed.iter() {
                if future.contains(terminal as usize) {
                    remapped
                        .entry(end_state)
                        .or_insert_with(BTreeSet::new)
                        .insert(terminal);
                }
            }
        }
    }
    Some(TerminalsDisallowed::from_map(remapped))
}

fn single_disallowed_pair(acc: &TerminalsDisallowed) -> Option<(u32, u32)> {
    if acc.len() != 1 {
        return None;
    }
    let mut states = acc.iter();
    let (state, terminals) = states.next()?;
    if states.next().is_some() || terminals.len() != 1 {
        return None;
    }
    Some((*state, *terminals.iter().next()?))
}

fn try_prune_single_initial_state_batched_accumulators(
    constraint: &Constraint,
    gss: &ParserGSS,
    bytes: &[u8],
    scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    cached_starts: &mut SmallVec<[u32; 8]>,
) -> Option<ParserGSS> {
    let mut accumulators = SmallVec::<[TerminalsDisallowed; 8]>::new();
    let mut overflow = false;
    gss.for_each_acc(|acc| {
        if overflow || acc.is_empty() || accumulators.contains(acc) {
            return;
        }
        if accumulators.len() == accumulators.capacity() {
            overflow = true;
            return;
        }
        accumulators.push(acc.clone());
    });
    if overflow || accumulators.len() < 2 {
        return None;
    }

    let mut pairs = SmallVec::<[(u32, u32); 8]>::new();
    for acc in &accumulators {
        pairs.push(single_disallowed_pair(acc)?);
    }
    // Each logical exclusion must belong to a disjoint lexer lane.  Then one
    // union execution preserves lane-local longest-match semantics because no
    // terminal ID can be observed from two starts.
    for i in 0..pairs.len() {
        let left = constraint.tokenizer.possible_future_terminals(pairs[i].0);
        for &(right_state, _) in &pairs[..i] {
            if !left.is_disjoint(
                constraint.tokenizer.possible_future_terminals(right_state),
            ) {
                return None;
            }
        }
    }
    let starts = pairs
        .iter()
        .map(|&(state, _)| state)
        .collect::<SmallVec<[u32; 8]>>();
    if !execute_tokenizer_reusable_from_states(constraint, bytes, &starts, scratch) {
        return None;
    }
    cached_starts.clear();
    cached_starts.extend(starts.iter().copied());

    let mut remapped = SmallVec::<[(TerminalsDisallowed, Option<TerminalsDisallowed>); 8]>::new();
    for (acc, &(_, terminal)) in accumulators.iter().zip(&pairs) {
        if scratch.matches.iter().any(|matched| matched.id == terminal) {
            remapped.push((acc.clone(), None));
            continue;
        }
        let mut next = TerminalsDisallowed::new();
        for &end_state in &scratch.states {
            if constraint
                .tokenizer
                .possible_future_terminals(end_state)
                .contains(terminal as usize)
            {
                next = next.with_insert(end_state, terminal);
            }
        }
        remapped.push((acc.clone(), Some(next)));
    }

    Some(gss.apply_and_prune_no_promote(|acc| {
        if acc.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        remapped
            .iter()
            .find(|(source, _)| source == acc)
            .and_then(|(_, result)| result.clone())
    }))
}

fn prune_single_initial_state_for_parts(
    constraint: &Constraint,
    gss: ParserGSS,
    tokenizer_state: u32,
    end_states: &[u32],
    matches: &[TokenizerMatch],
    bytes: &[u8],
) -> ParserGSS {
    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        advance_terminals_disallowed_over_bytes(
            constraint,
            terminals_disallowed,
            bytes,
            Some((tokenizer_state, end_states, matches)),
        )
    })
}

fn prune_single_initial_state_for_terminal(
    gss: ParserGSS,
    tokenizer_state: u32,
    terminal: u32,
    end_state: Option<u32>,
) -> ParserGSS {
    if end_state.is_none()
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| {
            td.get(&tokenizer_state)
                .is_none_or(|disallowed| !disallowed.contains(&terminal))
        })
    {
        return gss.apply(|_: &TerminalsDisallowed| TerminalsDisallowed::new());
    }

    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        if terminals_disallowed.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state)
            && disallowed.contains(&terminal) {
                return None;
            }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = end_state
            && let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
                remapped
                    .entry(end_state)
                    .or_insert_with(BTreeSet::new)
                    .extend(disallowed.iter().copied());
            }
        Some(TerminalsDisallowed::from_map(remapped))
    })
}

fn merge_parser_state(states: &mut ParserStatesByTokenizer, tokenizer_state: u32, gss: ParserGSS) {
    states.merge(tokenizer_state, gss);
}

fn queue_parser_state(
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    if new_offset == total_len {
        merge_parser_state(pending_state, tokenizer_state, gss);
    } else {
        merge_parser_state(&mut processing_queue[new_offset], tokenizer_state, gss);
    }
}

fn queue_parser_reset_state(
    constraint: &Constraint,
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    gss: ParserGSS,
) -> bool {
    if !constraint.uses_compact_segmented_parser_runtime() {
        queue_parser_state(
            processing_queue,
            pending_state,
            new_offset,
            total_len,
            constraint.runtime_commit_initial_state(),
            gss,
        );
        return true;
    }
    let Some(partitions) = constraint.partition_recursive_parser_gss_by_active_leaf(&gss) else {
        return false;
    };
    for (leaf_index, partition) in partitions {
        let Some(reset_state) = constraint.recursive_tokenizer_reset_state(leaf_index) else {
            return false;
        };
        queue_parser_state(
            processing_queue,
            pending_state,
            new_offset,
            total_len,
            reset_state,
            partition,
        );
    }
    true
}

fn queue_parser_ignored_reset_state(
    constraint: &Constraint,
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    gss: ParserGSS,
) -> bool {
    let gss = if constraint.uses_compact_segmented_parser_runtime() {
        constraint
            .close_compact_segmented_parser(&gss)
            .unwrap_or(gss)
    } else {
        gss
    };
    queue_parser_reset_state(
        constraint,
        processing_queue,
        pending_state,
        new_offset,
        total_len,
        gss,
    )
}

fn finalize_pending_state(pending_state: &mut ParserStatesByTokenizer) -> ParserStateMap {
    let mut state: ParserStateMap = pending_state.take_entries().into_iter().collect();
    for gss in state.values_mut() {
        *gss = gss.fuse(Some(1));
    }
    state.retain(|_, gss| !gss.is_empty());
    state
}

fn apply_future_terminal_disallow(
    constraint: &Constraint,
    exec_result: &TokenizerExecResult,
    terminal: u32,
    gss: ParserGSS,
) -> ParserGSS {
    apply_future_terminal_disallow_for_states(
        constraint,
        &exec_result.end_state,
        terminal,
        gss,
    )
}

fn apply_future_terminal_disallow_for_states(
    constraint: &Constraint,
    end_states: &[u32],
    terminal: u32,
    gss: ParserGSS,
) -> ParserGSS {
    if gss.is_empty() || end_states.is_empty() {
        return gss;
    }
    let relevant: SmallVec<[u32; INLINE_PARSER_STATE_CAPACITY]> = end_states
        .iter()
        .copied()
        .filter(|&end_state| {
            runtime_tokenizer_future_terminals(constraint, end_state)
                .is_some_and(|future| future.contains(terminal as usize))
        })
        .collect();
    if relevant.is_empty() {
        return gss;
    }

    gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
        let mut updated = terminals_disallowed.clone();
        for &end_state in &relevant {
            updated = updated.with_insert(end_state, terminal);
        }
        updated
    })
}

#[inline]
fn try_apply_single_top_action_in_place(gss: &mut ParserGSS, action: &Action) -> bool {
    match action {
        Action::Skip => true,
        Action::Shift(target, replace) => {
            let pushes = [*target];
            gss.try_apply_single_segment_stack_effect_in_place(usize::from(*replace), &pushes)
        }
        Action::ReplaceShifts(targets) if targets.len() == 1 => {
            gss.try_apply_single_segment_stack_effect_in_place(1, targets)
        }
        Action::StackShifts(shifts) => {
            let [shift] = shifts.as_slice() else {
                return false;
            };
            gss.try_apply_single_segment_stack_effect_in_place(
                shift.pop as usize,
                &shift.pushes,
            )
        }
        _ => false,
    }
}

#[inline]
fn apply_single_top_action_fast(
    constraint: &Constraint,
    gss: &ParserGSS,
    state: u32,
    terminal: u32,
    action: &Action,
) -> Option<ParserGSS> {
    if let Some(cached) = constraint.direct_regular_cached_advance(gss, terminal) {
        return Some(cached);
    }
    let table = &constraint.table;
    match action {
        Action::Skip => Some(gss.clone()),
        Action::Shift(target, is_replace) => {
            if let Some(mut stack) = gss.try_virtual_stack() {
                if *is_replace && stack.pop(1) != 0 {
                    return Some(gss.popn(1).push(*target));
                }
                stack.push(*target);
                Some(stack.into_gss())
            } else {
                Some(if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                })
            }
        }
        Action::ReplaceShifts(targets) => {
            let stack = gss.try_virtual_stack()?;
            stack.into_gss_after_popping_and_pushing_unique_single_branches(
                1,
                targets.iter(),
            )
        }
        Action::StackShifts(shifts) => {
            if let [shift] = shifts.as_slice() {
                let mut branch = gss.try_virtual_stack()?;
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                return Some(branch.into_gss());
            }
            if let Some(stack) = gss.try_virtual_stack()
                && let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = stack.into_gss_after_popping_and_pushing_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }
            if let Some(shifted) = gss.apply_stack_effects_to_single_concrete_path(
                shifts
                    .iter()
                    .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
            ) {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && shift.pushes.len() == 1)
                && let Some(shifted) = gss.apply_shared_pop_push_single_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| &shift.pushes[0]),
                )
            {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = gss.apply_shared_pop_push_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }

            let stack = gss.try_virtual_stack()?;
            let mut shifted = ParserGSS::empty();
            for shift in shifts {
                let mut branch = stack.clone();
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                let branch = branch.into_gss();
                shifted = if shifted.is_empty() {
                    branch
                } else {
                    shifted.merge(&branch)
                };
            }
            Some(shifted)
        }
        Action::GuardedStackShifts(shifts) => {
            apply_guarded_stack_shifts_fast(gss, shifts, table.guarded_shift_index(state, terminal))
        }
        Action::Reduce(..) => apply_single_path_reduce_chain_fast(table, gss, terminal),
        _ => None,
    }
}

fn try_apply_action_to_carried_virtual_stack(
    stack: &mut crate::ds::leveled_gss::VirtualStack<u32, TerminalsDisallowed>,
    action: &Action,
) -> bool {
    match action {
        Action::Skip => true,
        Action::Shift(target, is_replace) => {
            if *is_replace {
                stack.replace_top(*target)
            } else {
                stack.push(*target);
                true
            }
        }
        Action::StackShifts(shifts) => {
            let [shift] = shifts.as_slice() else {
                return false;
            };
            let mut candidate = stack.clone();
            if candidate.pop(shift.pop as usize) != 0 {
                return false;
            }
            for &target in &shift.pushes {
                candidate.push(target);
            }
            if candidate.top().is_none() {
                // The next parser/top-row decision requires a concrete top
                // state. If the visible prefix has been exhausted, continuing
                // to carry this virtual stack would make the stale materialized
                // GSS an invalid proxy for the current parser frontier.
                return false;
            }
            *stack = candidate;
            true
        }
        _ => false,
    }
}

/// Preserve a uniform reduction result's shared bottom-of-stack prefix.
/// Rebuilding the entire concrete-stack trie repeats work for each retained
/// value, although every output has the same accumulator and usually differs
/// only in its final one or two states. Mixed accumulators retain the general
/// builder: exclusions must never be detached from their particular paths.
fn materialize_reduce_chain_outputs(
    mut outputs: Vec<(Vec<u32>, TerminalsDisallowed)>,
) -> ParserGSS {
    if outputs.len() == 1 {
        let (stack, accumulator) = outputs.pop().unwrap();
        return ParserGSS::from_single_stack(stack, accumulator);
    }
    let Some((first, accumulator)) = outputs.first() else {
        return ParserGSS::empty();
    };
    if outputs.iter().any(|(_, other)| other != accumulator) {
        return ParserGSS::from_stacks(&outputs);
    }
    let mut shared = first.len();
    for (stack, _) in outputs.iter().skip(1) {
        shared = first[..shared].iter().zip(stack).take_while(|(a, b)| a == b).count();
    }
    if shared == 0 {
        return ParserGSS::from_stacks(&outputs);
    }

    let base = ParserGSS::from_single_stack(first[..shared].to_vec(), accumulator.clone());
    let accepts_prefix = outputs.iter().any(|(stack, _)| stack.len() == shared);
    if outputs.iter().all(|(stack, _)| stack.len() == shared) {
        return base;
    }
    let branches = base.apply_shared_pop_push_branches(
        0,
        outputs.iter().filter(|(stack, _)| stack.len() > shared)
            .map(|(stack, _)| &stack[shared..]),
    ).expect("a nonempty single-stack prefix admits every nonempty suffix");
    if accepts_prefix { base.merge(&branches) } else { branches }
}

fn apply_single_path_reduce_chain_fast(
    table: &GLRTable,
    gss: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    let (mut stack, acc) =
        gss.try_single_stack_bounded(SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH)?;

    loop {
        let state = *stack.last()?;
        match table.action(state, terminal)? {
            Action::Skip => {
                return Some(ParserGSS::from_single_stack(stack, acc));
            }
            Action::Reduce(nt, len) => {
                let rhs_len = *len as usize;
                if rhs_len >= stack.len() {
                    return None;
                }
                stack.truncate(stack.len() - rhs_len);
                let goto_from = *stack.last()?;
                let (target, is_replace) = table.goto_target(goto_from, *nt)?;
                if is_replace {
                    *stack.last_mut()? = target;
                } else {
                    stack.push(target);
                }
            }
            Action::Shift(target, is_replace) => {
                if *is_replace {
                    *stack.last_mut()? = *target;
                } else {
                    stack.push(*target);
                }
                return Some(ParserGSS::from_single_stack(stack, acc));
            }
            Action::StackShifts(shifts) => {
                return ParserGSS::from_single_stack(stack, acc)
                    .apply_stack_effects_to_single_concrete_path(
                        shifts
                            .iter()
                            .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                        SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                    );
            }
            Action::Split {
                shift,
                reduces,
                accept: false,
            } => {
                let mut out: Vec<(Vec<u32>, TerminalsDisallowed)> = Vec::new();

                if let Some((target, is_replace)) = shift {
                    let mut branch = stack.clone();
                    if *is_replace {
                        *branch.last_mut()? = *target;
                    } else {
                        branch.push(*target);
                    }
                    out.push((branch, acc.clone()));
                }

                for &(nt, len) in reduces {
                    let mut branch = stack.clone();
                    let rhs_len = len as usize;
                    if rhs_len >= branch.len() {
                        return None;
                    }
                    branch.truncate(branch.len() - rhs_len);
                    let goto_from = *branch.last()?;
                    let (target, is_replace) = table.goto_target(goto_from, nt)?;
                    if is_replace {
                        *branch.last_mut()? = target;
                    } else {
                        branch.push(target);
                    }

                    let follow_state = *branch.last()?;
                    match table.action(follow_state, terminal)? {
                        Action::Skip => {
                            out.push((branch, acc.clone()));
                        }
                        Action::Shift(target, is_replace) => {
                            if *is_replace {
                                *branch.last_mut()? = *target;
                            } else {
                                branch.push(*target);
                            }
                            out.push((branch, acc.clone()));
                        }
                        Action::StackShifts(shifts) => {
                            let shifted = ParserGSS::from_single_stack(branch, acc.clone())
                                .apply_stack_effects_to_single_concrete_path(
                                    shifts
                                        .iter()
                                        .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                                    SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                                )?;
                            let shifted_stacks = shifted
                                .to_stacks(shifts.len())
                                .expect("stack-shift result exceeded its effect count");
                            out.extend(shifted_stacks);
                        }
                        _ => return None,
                    }
                }

                return (!out.is_empty()).then(|| materialize_reduce_chain_outputs(out));
            }
            _ => return None,
        }
    }
}

fn advance_terminal_match(
    constraint: &Constraint,
    gss_at_offset: &ParserGSS,
    terminal: u32,
    exec_result: &TokenizerExecResult,
    advance_result_cache: &mut AdvanceResultCache,
    terminal_result_cache: &mut FxHashMap<u32, ParserGSS>,
) -> Option<ParserGSS> {
    if let Some(cached) = terminal_result_cache.get(&terminal) {
        return (!cached.is_empty()).then(|| cached.clone());
    }

    let advance_cache_key = (gss_at_offset.ptr_key(), terminal);
    let advanced = if let Some((_, cached)) = advance_result_cache.get(&advance_cache_key) {
        cached.clone()
    } else {
        let advanced = try_advance_unique_actionable_top_fast(constraint, gss_at_offset, terminal)
            .unwrap_or_else(|| advance_parser_stacks(constraint, gss_at_offset, terminal));
        advance_result_cache.insert(advance_cache_key, (gss_at_offset.clone(), advanced.clone()));
        advanced
    };

    let advanced = apply_future_terminal_disallow(constraint, exec_result, terminal, advanced);
    terminal_result_cache.insert(terminal, advanced.clone());
    (!advanced.is_empty()).then_some(advanced)
}

fn try_advance_unique_actionable_top_fast(
    constraint: &Constraint,
    gss: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if constraint.parser_has_controls() || constraint.has_template_parser() || constraint.uses_compact_segmented_parser_runtime() {
        return None;
    }
    // If every nonempty path has this top, isolate(Some(top)) is precisely
    // the original immutable GSS. Avoid enumerating/copying the top set and
    // retaining an extra Arc. The caller must not retry a declined action:
    // repeating it on this same GSS cannot make the shortcut more applicable.
    if let Some(top) = gss.single_exclusive_top_value() {
        let action = constraint.table.action(top, terminal)?;
        return apply_single_top_action_fast(constraint, gss, top, terminal, action);
    }
    let mut selected = None;
    for top in gss.peek_values() {
        let Some(action) = constraint.table.action(top, terminal) else {
            continue;
        };
        if selected.is_some() {
            return None;
        }
        selected = Some((top, action));
    }
    let (top, action) = selected?;
    let isolated = gss.isolate(Some(top));
    (!isolated.is_empty())
        .then(|| apply_single_top_action_fast(constraint, &isolated, top, terminal, action))
        .flatten()
}

fn record_per_advance_entry(
    advances: &mut Vec<PerAdvanceEntry>,
    tokenizer_state: u32,
    terminal_id: u32,
    before_gss: &ParserGSS,
    after_gss: &ParserGSS,
    match_start: usize,
    match_end: usize,
    token_bound: usize,
    match_bytes: &[u8],
    profile: AdvanceProfile,
) -> u64 {
    use std::time::Instant;

    let summary_start = Instant::now();
    let gss_stacks_before = parser_stacks_only(before_gss);
    let gss_stacks_after = parser_stacks_only(after_gss);
    let gss_summary_before = before_gss.summary();
    let gss_summary_after = after_gss.summary();
    let match_bytes = match_bytes.to_vec();
    let summary_ns = summary_start.elapsed().as_nanos() as u64;
    advances.push(PerAdvanceEntry {
        terminal_id,
        tokenizer_state,
        gss_stacks_before,
        gss_stacks_after,
        gss_summary_before,
        gss_summary_after,
        match_start,
        match_end,
        token_bound,
        match_bytes,
        profile,
        summary_ns,
    });
    summary_ns
}

fn commit_bytes_impl_profiled(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    advances: Option<&mut Vec<PerAdvanceEntry>>,
    allow_fast_paths: bool,
) -> Result<CommitProfile, String> {
    let started = std::time::Instant::now();
    let mut profile = CommitProfile {
        n_tokenizer_states: state.len() as u64,
        ..CommitProfile::default()
    };
    commit_worklist(
        constraint, state, bytes, bufs, &mut NoMarkers,
        Some(&mut profile), advances, !allow_fast_paths,
    )?;
    finish_commit_frontier(constraint, state, bytes.len(), &mut NoMarkers)?;
    profile.total_ns = started.elapsed().as_nanos() as u64;
    Ok(profile)
}

fn final_stacks(state: &ParserStateMap) -> Vec<(u32, Vec<Vec<u32>>)> {
    let mut grouped = BTreeMap::<u32, Vec<Vec<u32>>>::new();
    for (&tokenizer_state, gss) in state.iter() {
        grouped
            .entry(tokenizer_state)
            .or_default()
            .extend(parser_stacks_only(gss));
    }
    for stacks in grouped.values_mut() {
        stacks.sort();
        stacks.dedup();
    }
    grouped.into_iter().collect()
}

#[cfg(test)]
fn apply_flat_stack_effect(
    stack: &mut FlatInlineStack,
    pop: u32,
    pushes: &[u32],
) -> Option<bool> {
    let pop = pop as usize;
    // Stack effects are atomic: popping the final visible state is valid when
    // the same effect pushes a replacement. Only a true underflow is invalid.
    if pop > stack.len() {
        return Some(false);
    }
    let new_len = stack.len() - pop + pushes.len();
    if new_len > LINEAR_STACK_RESERVE {
        return None;
    }
    stack.truncate(stack.len() - pop);
    stack.extend_from_slice(pushes);
    Some(true)
}

#[inline]
fn maybe_normalize_lookahead_invariant_reductions(
    constraint: &Constraint,
    state: &mut ParserStateMap,
) {
    if constraint.has_template_parser() { return; }

    if constraint.static_dynamic_overlay.is_none()
        || constraint.uses_compact_segmented_parser_runtime()
        || crate::env::var_os("GLRMASK_EXPERIMENT_EAGER_INVARIANT_REDUCTIONS").is_none()
    {
        return;
    }
    for gss in state.values_mut() {
        *gss = normalize_lookahead_invariant_reductions(&constraint.table, gss);
    }
}

#[inline]
fn commit_bytes_impl(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
) -> Result<(), String> {
    commit_bytes_impl_observed(constraint, state, bytes, bufs, &mut NoMarkers)
}

fn commit_bytes_impl_observed<O: CommitObserver>(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    observer: &mut O,
) -> Result<(), String> {
    commit_bytes_impl_inner_observed(constraint, state, bytes, bufs, observer)?;
    finish_commit_frontier(constraint, state, bytes.len(), observer)
}

fn finish_commit_frontier<O: CommitObserver>(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    byte_offset: usize,
    observer: &mut O,
) -> Result<(), String> {
    maybe_normalize_lookahead_invariant_reductions(constraint, state);

    // General symbolic residuals expose a conservative infallible future bit
    // during the inner commit so ordinary parser/lexer machinery can remain
    // unchanged. Before accepting the token, resolve every non-reset lexer key
    // through the bounded exact residual-liveness query. A hard Boolean case
    // that exceeds the work ceiling is an error, never a false live/dead bit.
    if constraint.tokenizer_requires_exact_liveness() {
        let mut dead = Vec::<u32>::new();
        for &tokenizer_state in state.keys() {
            // A recursive state belongs to its leaf lexer, not the outer
            // coordinator's superficially similar numeric state alphabet.
            if !constraint.exact_runtime_tokenizer_state_has_future(tokenizer_state)? {
                dead.push(tokenizer_state);
            }
        }
        if !dead.is_empty() {
            dead.sort_unstable();
            state.retain(|tokenizer_state, _| dead.binary_search(tokenizer_state).is_err());
        }
        if state.is_empty() {
            return Err("commit rejected: no valid parser states remain".to_owned());
        }
    }

    observer.frontier(constraint, state, byte_offset);
    Ok(())
}

fn commit_bytes_impl_inner_observed<O: CommitObserver>(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    observer: &mut O,
) -> Result<(), String> {
    if bytes.is_empty() { return Ok(()); }
    let has_linker_controls = constraint.parser_has_controls()
        || constraint.uses_compact_segmented_parser_runtime();
    // A wide direct-regular frontier already carries its exact actionable
    // terminal support. Scan only that support instead of materializing every
    // tokenizer finalizer, which can be proportional to the remaining source
    // file in project-scale diff grammars.
    if !has_linker_controls
        && state.len() == 1
        && !constraint.tokenizer_has_epsilon_transitions
    {
        let (&start_tokenizer_state, gss) = state.iter().next().unwrap();
        if let Some(summary) = constraint.direct_regular_wide_frontier_for_gss(gss)
            && let Some(accumulator) = gss.uniform_accumulator()
            && let Some(end_state) = scan_wide_frontier_lexer_only(
                constraint,
                bytes,
                start_tokenizer_state,
                summary,
            )
        {
            let Some(updated_accumulator) = advance_uniform_disallowed_interest_only(
                constraint,
                &accumulator,
                bytes,
            ) else {
                state.clear();
                return Err("commit rejected: no valid parser states remain".to_string());
            };
            if !wide_frontier_end_state_may_advance(constraint, summary, end_state) {
                state.clear();
                return Err("commit rejected: no valid parser states remain".to_string());
            }
            if let Some(updated) = gss.with_uniform_accumulator(updated_accumulator) {
                state.clear();
                state.insert(end_state, updated);
                return Ok(());
            }
        }
    }

    commit_worklist(constraint, state, bytes, bufs, observer, None, None, false)
}

/// The correlated frontier is a worklist over byte offset and lexer key.
/// Each event advances the parser and routes its reset. An unfinished lexeme
/// instead retains the input parser language, including its delayed guards.
fn commit_worklist<O: CommitObserver>(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    observer: &mut O,
    mut profile: Option<&mut CommitProfile>,
    mut advances: Option<&mut Vec<PerAdvanceEntry>>,
    reference: bool,
) -> Result<(), String> {
    if bytes.is_empty() { return Ok(()); }
    state.normalize_duplicate_keys();
    bufs.clear_all();
    let queue_started = profile.as_ref().map(|_| std::time::Instant::now());
    let mut processing_queue = std::mem::take(&mut bufs.processing_queue);
    processing_queue.resize_with(bytes.len() + 1, ParserStatesByTokenizer::default);
    for bucket in &mut processing_queue { bucket.clear(); }
    for (key, gss) in std::mem::take(state) { processing_queue[0].merge(key, gss); }
    let result = (|| {
        for offset in 0..=bytes.len() {
            let states_to_process = processing_queue[offset].take_entries();
            for (tokenizer_state, mut gss) in states_to_process {
                if let Some(p) = profile.as_deref_mut() { p.n_queue_entries += 1; }
                let started = profile.as_ref().map(|_| std::time::Instant::now());
                let exec = execute_tokenizer_from_state_small(
                    constraint, &bytes[offset..], tokenizer_state,
                );
                if let (Some(p), Some(started)) = (profile.as_deref_mut(), started) {
                    p.exec_ns += started.elapsed().as_nanos() as u64;
                }
                if offset == 0 && !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
                    gss = prune_single_initial_state_for_exec(
                        constraint, gss, tokenizer_state, &exec, bytes,
                    );
                    if gss.is_empty() { continue; }
                }
                let actionable = if reference { None } else {
                    ActionableTerminals::from_gss(constraint, &gss)
                };
                bufs.terminal_result_cache.clear();
                let matches = collect_unique_actionable_matches(
                    constraint, actionable.as_ref(), constraint.ignore_terminal,
                    &exec.matches, Some(&mut bufs.seen_matches),
                );
                // Admission is a query on the input parser, not the result of
                // an event. Decide all continuation destinations first.
                let admission = if reference { None } else {
                    cached_batched_end_state_admission(
                        constraint, &gss, &exec.end_state, &mut bufs.admission_cache,
                    )
                };
                let mut continuation = SmallVec::<[u32; 4]>::new();
                for &end_state in &exec.end_state {
                    let allowed = if reference {
                        end_state_may_advance(constraint, &gss, end_state)
                    } else if let Some(index) = admission {
                        end_state_may_advance_from_cache_entry(
                            constraint, end_state, &bufs.admission_cache[index],
                        )
                    } else {
                        cached_single_end_state_may_advance(
                            constraint, &gss, end_state, &mut bufs.admission_cache,
                        )
                    };
                    if allowed { continuation.push(end_state); }
                }
                // A single successor needs no retained source GSS. The same
                // worklist naturally becomes a linear execution in this case.
                if !reference && profile.is_none() && matches.len() == 1 && continuation.is_empty() {
                    let matched = &matches[0];
                    let end = offset + matched.width;
                    if matched.ignored {
                        observer.stack(constraint, &gss, end);
                        if !queue_parser_ignored_reset_state(
                            constraint, &mut processing_queue, &mut bufs.pending_state,
                            end, bytes.len(), gss,
                        ) {
                            return Err("recursive tokenizer reset routing failed".to_owned());
                        }
                    } else {
                        let mut input = gss;
                        let advanced = if !constraint.parser_has_controls()
                            && !constraint.has_template_parser()
                            && !constraint.uses_compact_segmented_parser_runtime()
                            && input.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
                            && let Some(top) = input.single_exclusive_top_value()
                            && let Some(action) = constraint.table.action(top, matched.terminal_id)
                            && try_apply_single_top_action_in_place(&mut input, action)
                        {
                            input
                        } else {
                            advance_parser_stacks_owned(constraint, input, matched.terminal_id)
                        };
                        let advanced = apply_future_terminal_disallow(
                            constraint, &exec, matched.terminal_id, advanced,
                        );
                        if !advanced.is_empty() {
                            observer.stack(constraint, &advanced, end);
                            if !queue_parser_reset_state(
                                constraint, &mut processing_queue, &mut bufs.pending_state,
                                end, bytes.len(), advanced,
                            ) {
                                return Err("recursive tokenizer reset routing failed".to_owned());
                            }
                        }
                    }
                    continue;
                }
                for matched in matches {
                    let end = offset + matched.width;
                    if matched.ignored {
                        observer.stack(constraint, &gss, end);
                        if !queue_parser_ignored_reset_state(
                            constraint, &mut processing_queue, &mut bufs.pending_state,
                            end, bytes.len(), gss.clone(),
                        ) {
                            return Err("recursive tokenizer reset routing failed".to_owned());
                        }
                        continue;
                    }
                    let advanced = if let Some(p) = profile.as_deref_mut() {
                        let attempt = advance_parser_stacks_profiled_if_possible(
                            constraint, &gss, matched.terminal_id,
                        );
                        p.advance_may_check_ns += attempt.may_ns;
                        p.advance_core_ns += attempt.core_ns;
                        if attempt.advanced.is_empty() { continue; }
                        apply_advance_profile(p, &attempt.profile);
                        if let Some(entries) = advances.as_deref_mut() {
                            p.adv_summary_ns += record_per_advance_entry(
                                entries, tokenizer_state, matched.terminal_id, &gss,
                                &attempt.advanced, offset, end, bytes.len(),
                                &bytes[offset..end], attempt.profile,
                            );
                        }
                        let started = std::time::Instant::now();
                        let advanced = apply_future_terminal_disallow(
                            constraint, &exec, matched.terminal_id, attempt.advanced,
                        );
                        let elapsed = started.elapsed().as_nanos() as u64;
                        p.advance_future_disallow_ns += elapsed;
                        p.advance_ns += attempt.may_ns + attempt.core_ns + elapsed;
                        p.n_advances += 1;
                        advanced
                    } else {
                        let Some(advanced) = advance_terminal_match(
                            constraint, &gss, matched.terminal_id, &exec,
                            &mut bufs.advance_result_cache, &mut bufs.terminal_result_cache,
                        ) else { continue; };
                        advanced
                    };
                    if advanced.is_empty() { continue; }
                    observer.stack(constraint, &advanced, end);
                    if !queue_parser_reset_state(
                        constraint, &mut processing_queue, &mut bufs.pending_state,
                        end, bytes.len(), advanced,
                    ) {
                        return Err("recursive tokenizer reset routing failed".to_owned());
                    }
                }
                for end_state in continuation {
                    queue_parser_state(
                        &mut processing_queue, &mut bufs.pending_state,
                        bytes.len(), bytes.len(), end_state, gss.clone(),
                    );
                }
            }
        }
        let started = profile.as_ref().map(|_| std::time::Instant::now());
        *state = finalize_pending_state(&mut bufs.pending_state);
        if let (Some(p), Some(started)) = (profile.as_deref_mut(), started) {
            p.fuse_ns += started.elapsed().as_nanos() as u64;
        }
        if state.is_empty() {
            return Err("commit rejected: no valid parser states remain".to_string());
        }
        Ok(())
    })();
    bufs.processing_queue = processing_queue;
    if let (Some(p), Some(started)) = (profile, queue_started) {
        p.queue_ns += started.elapsed().as_nanos() as u64;
        p.queue_bookkeeping_ns = p.queue_ns.saturating_sub(
            p.exec_ns + p.advance_ns + p.fuse_ns
        );
    }
    result
}

impl ConstraintState {
    /// Run one commit. A failed commit rejects the state: the frontier becomes
    /// empty, the mask is empty, and every later commit errors.
    fn commit_transaction<T>(
        &mut self,
        commit: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        match commit(self) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.reject_after_failed_commit();
                Err(error)
            }
        }
    }

    pub(crate) fn reject_after_failed_commit(&mut self) {
        self.state.clear();
        self.terminated = false;
        if let Some(markers) = self.markers.as_deref_mut() { markers.begin(); }
        self.buffers.reset_all();
        self.generation = self.generation.wrapping_add(1);
        *self.mask_cache.lock().unwrap() = None;
    }

    #[inline]
    fn commit_token_actual(&mut self, token_id: u32) -> Result<(), String> {
        self.commit_transaction(|this| this.commit_token_actual_inner(token_id))
    }

    fn commit_token_actual_inner(&mut self, token_id: u32) -> Result<(), String> {
        let constraint = &self.constraint;
        let Some(markers) = self.markers.as_deref_mut() else {
            return commit_token_impl(constraint, &mut self.state, &mut self.buffers, token_id);
        };
        markers.begin();
        let queries = constraint.marker_queries.as_deref().expect("marker-enabled state has immutable queries");
        let base = markers.byte_offset;
        let width = constraint.token_bytes_for_id(token_id).map_or(0, |bytes| bytes.len());
        let next = base.checked_add(width as u64).ok_or("marker byte position overflow")?;
        let result = commit_token_impl_observed(constraint, &mut self.state, &mut self.buffers, token_id,
            &mut LiveMarkers { queries, state: markers, base });
        markers.finish(queries, next, result.is_ok());
        result
    }

    #[inline]
    fn commit_bytes_actual(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.commit_transaction(|this| this.commit_bytes_actual_inner(bytes))
    }

    fn commit_bytes_actual_inner(&mut self, bytes: &[u8]) -> Result<(), String> {
        let constraint = self.constraint.clone();
        let Some(markers) = self.markers.as_deref_mut() else {
            return commit_bytes_impl(&constraint, &mut self.state, bytes, &mut self.buffers);
        };
        markers.begin();
        let queries = constraint.marker_queries.as_deref().expect("marker-enabled state has immutable queries");
        let base = markers.byte_offset;
        let next = base.checked_add(bytes.len() as u64).ok_or("marker byte position overflow")?;
        let result = commit_bytes_impl_observed(&constraint, &mut self.state, bytes, &mut self.buffers,
            &mut LiveMarkers { queries, state: markers, base });
        markers.finish(queries, next, result.is_ok());
        result
    }

    pub(crate) fn knows_token_id(&self, token_id: u32) -> bool {
        self.constraint.token_bytes_for_id(token_id).is_some()
            || self.constraint.has_special_token_id(token_id)
    }

    /// Snapshot the exact runtime state only when a reusable mask is currently
    /// cached. `ParserGSS` is Arc-backed, so cloning this small map keeps the
    /// old GSS identities alive without copying parser stacks.
    #[inline]
    fn snapshot_current_mask_state(&self) -> Option<ParserStateMap> {
        let cache = self.mask_cache.lock().unwrap();
        cache
            .as_ref()
            .is_some_and(|cache_data| cache_data.generation == self.generation)
            .then(|| self.state.clone())
    }

    /// Advance the commit generation while retaining a cached mask when commit
    /// left the complete semantic runtime state unchanged. `ParserStateMap`
    /// equality compares tokenizer keys and exact immutable GSS structure;
    /// `LeveledGSS` itself checks Arc identity first, so the usual unchanged
    /// object case remains a pointer comparison. `fill_mask` is a pure
    /// function of this state and the immutable constraint.
    #[inline]
    fn finish_commit_generation(
        &mut self,
        previous_state: Option<ParserStateMap>,
        commit_succeeded: bool,
    ) {
        // A failed commit leaves the state exactly as it was, so the cached mask
        // still describes it and the generation must not advance.
        if !commit_succeeded {
            return;
        }
        let previous_generation = self.generation;
        self.generation += 1;
        let Some(previous_state) = previous_state else {
            return;
        };
        let mask_state_unchanged = previous_state == self.state
            || (self.constraint.uses_dynamic_runtime()
                // Ordinary tokenizer quotients cannot interpret scoped
                // recursive composition IDs. Only exact state equality above
                // is currently proved for retaining a composed result.
                && !self.constraint.uses_compact_segmented_parser_runtime()
                && (self.dynamic_mask_projection_state_eq(&previous_state, &self.state)
                    || (crate::env::var_os(
                        "GLRMASK_DISABLE_DYNAMIC_MASK_PARSER_RELATIVE_COMMIT_REUSE",
                    )
                    .is_none()
                        && self.dynamic_parser_relative_vocab_mask_eq(&previous_state))));
        if !mask_state_unchanged {
            return;
        }

        let mut cache = self.mask_cache.lock().unwrap();
        if let Some(cache_data) = cache.as_mut()
            && cache_data.generation == previous_generation
        {
            cache_data.generation = self.generation;
        }
    }

    /// Exact equality in the lexer coordinate actually consumed by dynamic
    /// mask generation. A commit may advance the source/runtime lexer through
    /// states that are distinct only outside the model-vocabulary horizon; if
    /// every correlated parser GSS is unchanged and those source states map to
    /// the same mask-execution state, the next-token mask is identical and the
    /// already-cached mask can survive the generation bump.
    ///
    /// This is deliberately stricter than general semantic cache lookup: it
    /// does not merge parser alternatives or rely on observation classes. It
    /// only replaces the exact tokenizer-state coordinate with the exact
    /// finite mask coordinate selected by the constraint.
    fn dynamic_mask_projection_state_eq(
        &self,
        previous: &ParserStateMap,
        current: &ParserStateMap,
    ) -> bool {
        if previous.len() != current.len() {
            return false;
        }
        let vocab = self.constraint.dynamic_mask_vocab_for_runtime();
        let exact_initial = self.constraint.tokenizer.initial_state();

        // The common case is one correlated lexer/parser branch. Keep that
        // path allocation-free and constant-time.
        if let ([(previous_state, previous_gss)], [(current_state, current_gss)]) =
            (previous.entries.as_slice(), current.entries.as_slice())
        {
            return previous_gss == current_gss
                && (*previous_state == exact_initial) == (*current_state == exact_initial)
                && vocab.mask_runtime_state(*previous_state)
                    == vocab.mask_runtime_state(*current_state);
        }

        // Flat GLR frontiers can contain duplicate exact tokenizer keys. Match
        // correlated branches as a multiset in mask-coordinate space rather
        // than assuming source-state ordering survives quotienting.
        let mut matched = vec![false; current.len()];
        'previous: for (previous_state, previous_gss) in &previous.entries {
            let projected = vocab.mask_runtime_state(*previous_state);
            let initial = *previous_state == exact_initial;
            for (index, (current_state, current_gss)) in current.entries.iter().enumerate() {
                if !matched[index]
                    && initial == (*current_state == exact_initial)
                    && projected == vocab.mask_runtime_state(*current_state)
                    && previous_gss == current_gss
                {
                    matched[index] = true;
                    continue 'previous;
                }
            }
            return false;
        }
        true
    }

    /// Prove equality of the next-token mask for the common singleton parser
    /// frontier even when the exact lexer residual changed.
    ///
    /// This proof is deliberately vocabulary- and parser-relative. Two lexer
    /// states may differ because one still carries futures for terminals the
    /// unchanged parser frontier cannot consume. Those differences do not
    /// affect the next model-token mask. We walk the model vocabulary trie in
    /// lockstep and compare only matched/future observations for terminals
    /// admitted by this parser GSS. Once the lexer states converge, or both
    /// no-finalization branches become parser-dead, the complete suffix
    /// subtree is equal and no further work is needed.
    ///
    /// Equal admitted finalizers also need no recursive parser simulation:
    /// both sides advance the same GSS on the same terminal and reset to the
    /// same lexer initial state, so every post-finalization suffix branch is
    /// identical by construction.
    fn dynamic_parser_relative_vocab_mask_eq(
        &mut self,
        previous: &ParserStateMap,
    ) -> bool {
        let current = &self.state;
        if self.constraint.static_dynamic_overlay.is_some()
            || self.constraint.uses_compact_segmented_parser_runtime()
            || self.constraint.tokenizer.has_any_virtual_runtime()
        {
            return false;
        }
        let (left_source, left_gss, right_source, right_gss) =
            if let ([(left_source, left_gss)], [(right_source, right_gss)]) =
                (previous.entries.as_slice(), current.entries.as_slice())
            {
                (left_source, left_gss, right_source, right_gss)
            } else {
                // Conservative multi-branch extension: first pair off every
                // branch whose exact correlated lexer/GSS state is unchanged.
                // Only if exactly one branch remains on each side do we apply
                // the parser-relative lexer proof below.  This covers the
                // common GLR shape where one guarded/reset branch is stable
                // and one ordinary lexer residual advances, without needing a
                // general assignment search over unrelated alternatives.
                if previous.len() != current.len() {
                    return false;
                }
                let mut matched = vec![false; current.len()];
                let mut unmatched_previous = None::<(&u32, &ParserGSS)>;
                for (previous_state, previous_gss) in &previous.entries {
                    let exact = current
                        .entries
                        .iter()
                        .enumerate()
                        .find(|(index, (current_state, current_gss))| {
                            !matched[*index]
                                && previous_state == current_state
                                && previous_gss == current_gss
                        })
                        .map(|(index, _)| index);
                    if let Some(index) = exact {
                        matched[index] = true;
                        continue;
                    }
                    if unmatched_previous
                        .replace((previous_state, previous_gss))
                        .is_some()
                    {
                        return false;
                    }
                }
                let Some((left_source, left_gss)) = unmatched_previous else {
                    return false;
                };
                let mut unmatched_current = current
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !matched[*index])
                    .map(|(_, pair)| pair);
                let Some((right_source, right_gss)) = unmatched_current.next() else {
                    return false;
                };
                if unmatched_current.next().is_some() {
                    return false;
                }
                (left_source, left_gss, right_source, right_gss)
            };
        if left_gss != right_gss
            || !left_gss.all_accs_satisfy(|acc: &TerminalsDisallowed| acc.is_empty())
        {
            return false;
        }

        let tokenizer = &self.constraint.tokenizer;
        let initial = tokenizer.initial_state();
        if (*left_source == initial) != (*right_source == initial)
            || tokenizer.state_has_epsilon_transitions(*left_source)
            || tokenizer.state_has_epsilon_transitions(*right_source)
        {
            return false;
        }

        let (cache_left, cache_right) = if left_source <= right_source {
            (*left_source, *right_source)
        } else {
            (*right_source, *left_source)
        };
        let candidates = crate::ds::bitset::BitSet::all(
            self.constraint.parser_terminal_count() as usize,
        );
        let mut admitted = exact_admitted_terminals_for_candidates(
            &self.constraint,
            left_gss,
            &candidates,
        );
        // IGNORE is parser-transparent but still lexer-observable: a live
        // ignore future permits a token boundary, and an ignore match resets
        // the lexer while leaving the current GSS unchanged.  Treat it as an
        // always-observed terminal in this equivalence proof. Equal ignore
        // events therefore converge exactly just like equal parser-admitted
        // finalizers, except without a parser advance.
        if let Some(ignore) = self.constraint.ignore_terminal {
            admitted.set(ignore as usize);
        }
        if admitted.count_ones() == 0 {
            return true;
        }

        if self
            .buffers
            .parser_relative_mask_eq_cache
            .iter()
            .any(|entry| {
                entry.left == cache_left
                    && entry.right == cache_right
                    && entry.admitted == admitted
            })
        {
            if crate::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK_COMMIT_REUSE").is_some() {
                eprintln!(
                    "[glrmask/profile][parser_relative_commit_reuse] generation={} result=cache_hit left={} right={} admitted={}",
                    self.generation,
                    left_source,
                    right_source,
                    admitted.count_ones(),
                );
            }
            return true;
        }
        // A new relation certificate can cost tens of microseconds. If the
        // destination state already has a materialized global dynamic-mask
        // cache entry, proving equality with the previous state cannot beat
        // simply letting the next fill copy that existing result. Keep this
        // lookup below all cheap structural/relation-cache gates so ordinary
        // commits never pay dynamic-mask key construction merely to decline.
        if super::dynamic_mask::dynamic_mask_state_has_cached_result(self) {
            return false;
        }

        #[inline(always)]
        fn equal_under_mask(
            left: &crate::ds::bitset::BitSet,
            right: &crate::ds::bitset::BitSet,
            mask: &crate::ds::bitset::BitSet,
        ) -> bool {
            left.words()
                .iter()
                .zip(right.words())
                .zip(mask.words())
                .all(|((&left, &right), &mask)| ((left ^ right) & mask) == 0)
        }

        #[inline(always)]
        fn any_under_mask(
            value: &crate::ds::bitset::BitSet,
            mask: &crate::ds::bitset::BitSet,
        ) -> bool {
            value
                .words()
                .iter()
                .zip(mask.words())
                .any(|(&value, &mask)| value & mask != 0)
        }

        let compare = |left: u32, right: u32| -> Option<(bool, bool)> {
            const DEAD: u32 = u32::MAX;
            if left == DEAD || right == DEAD {
                let live = |state: u32| {
                    state != DEAD
                        && (state == initial
                            || any_under_mask(
                                tokenizer.matched_terminal_bitset(state),
                                &admitted,
                            )
                            || any_under_mask(
                                tokenizer.possible_future_terminals(state),
                                &admitted,
                            ))
                };
                let left_live = live(left);
                let right_live = live(right);
                return Some((left_live == right_live, left_live && right_live));
            }
            if tokenizer.state_has_epsilon_transitions(left)
                || tokenizer.state_has_epsilon_transitions(right)
                || (left == initial) != (right == initial)
            {
                return None;
            }
            let matched_equal = equal_under_mask(
                tokenizer.matched_terminal_bitset(left),
                tokenizer.matched_terminal_bitset(right),
                &admitted,
            );
            let futures_equal = equal_under_mask(
                tokenizer.possible_future_terminals(left),
                tokenizer.possible_future_terminals(right),
                &admitted,
            );
            if !matched_equal || !futures_equal {
                return Some((false, true));
            }
            let live = left == initial
                || any_under_mask(tokenizer.matched_terminal_bitset(left), &admitted)
                || any_under_mask(tokenizer.possible_future_terminals(left), &admitted);
            Some((true, live))
        };

        let Some((root_equal, root_live)) = compare(*left_source, *right_source) else {
            return false;
        };
        if !root_equal {
            return false;
        }
        if !root_live || left_source == right_source {
            return true;
        }

        let vocab = self.constraint.dynamic_mask_vocab_for_runtime();
        let trie = vocab.trie.as_ref();
        let max_byte_steps = crate::env::var(
            "GLRMASK_DYNAMIC_MASK_PARSER_RELATIVE_COMMIT_REUSE_MAX_STEPS",
        )
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        // Speculative commit-time proof work must stay well below the cost of
        // an ordinary full vocabulary mask. Successful broad-interior
        // equivalences converge rapidly because most first-byte branches either
        // die under the parser observation or enter the same lexer state. A
        // fixed transition-work budget keeps adversarial/non-equivalent pairs
        // from degenerating into a second full vocabulary traversal.
        .unwrap_or(2_048);
        let started = crate::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK_COMMIT_REUSE")
            .is_some()
            .then(std::time::Instant::now);
        let mut stack = vec![(0u32, *left_source, *right_source)];
        let mut visited_nodes = 0usize;
        let mut byte_steps = 0usize;
        let mut converged_subtrees = 0usize;
        let mut dead_subtrees = 0usize;
        // Reuse the existing byte-budget branch as the one-shot checkpoint for
        // the stronger terminal-observation quotient. Grammars without a
        // prepared quotient retain the historical 2,048-step budget exactly.
        // With a prepared quotient, pause once at 256 steps; if the quotient
        // cannot certify equality, restore the historical budget and continue.
        const TERMINAL_OBSERVATION_QUOTIENT_CHECKPOINT: usize = 256;
        let mut byte_step_budget = if max_byte_steps > TERMINAL_OBSERVATION_QUOTIENT_CHECKPOINT
            && vocab.shares_terminal_observation_class(*left_source, *right_source)
        {
            TERMINAL_OBSERVATION_QUOTIENT_CHECKPOINT
        } else {
            max_byte_steps
        };

        while let Some((node, left_state, right_state)) = stack.pop() {
            visited_nodes += 1;
            for edge in trie.children(node) {
                let mut left = left_state;
                let mut right = right_state;
                let mut subtree_done = false;
                for &byte in trie.edge_bytes(edge) {
                    if left == right {
                        converged_subtrees += 1;
                        subtree_done = true;
                        break;
                    }
                    byte_steps += 1;
                    if byte_steps > byte_step_budget
                        || tokenizer.state_has_epsilon_transitions(left)
                        || tokenizer.state_has_epsilon_transitions(right)
                    {
                        if byte_steps > byte_step_budget && byte_step_budget < max_byte_steps {
                            byte_step_budget = max_byte_steps;
                            if try_terminal_observation_quotient_commit_reuse(
                                &mut self.buffers,
                                self.generation,
                                vocab,
                                tokenizer,
                                *left_source,
                                *right_source,
                                &admitted,
                                cache_left,
                                cache_right,
                                visited_nodes,
                                byte_steps,
                                started,
                            ) {
                                return true;
                            }
                        }
                        if byte_steps > byte_step_budget
                            || tokenizer.state_has_epsilon_transitions(left)
                            || tokenizer.state_has_epsilon_transitions(right)
                        {
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][parser_relative_commit_reuse] generation={} result=decline reason={} nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                    self.generation,
                                    if byte_steps > byte_step_budget { "budget" } else { "epsilon" },
                                    visited_nodes,
                                    byte_steps,
                                    converged_subtrees,
                                    dead_subtrees,
                                    started.elapsed().as_secs_f64() * 1e6,
                                );
                            }
                            return false;
                        }
                    }
                    let left_next = tokenizer.get_transition(left, byte);
                    let right_next = tokenizer.get_transition(right, byte);
                    let Some((equal, live)) = compare(left_next, right_next) else {
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=decline reason=epsilon_target nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                self.generation,
                                visited_nodes,
                                byte_steps,
                                converged_subtrees,
                                dead_subtrees,
                                started.elapsed().as_secs_f64() * 1e6,
                            );
                        }
                        return false;
                    };
                    if !equal {
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=different nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                self.generation,
                                visited_nodes,
                                byte_steps,
                                converged_subtrees,
                                dead_subtrees,
                                started.elapsed().as_secs_f64() * 1e6,
                            );
                        }
                        return false;
                    }
                    if !live {
                        dead_subtrees += 1;
                        subtree_done = true;
                        break;
                    }
                    left = left_next;
                    right = right_next;
                }
                if subtree_done {
                    continue;
                }
                if left == right {
                    converged_subtrees += 1;
                    continue;
                }
                stack.push((edge.child, left, right));
            }
        }

        if let Some(started) = started {
            eprintln!(
                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=equal left={} right={} admitted={} nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                self.generation,
                left_source,
                right_source,
                admitted.count_ones(),
                visited_nodes,
                byte_steps,
                converged_subtrees,
                dead_subtrees,
                started.elapsed().as_secs_f64() * 1e6,
            );
        }
        const CACHE_CAPACITY: usize = 8;
        if self.buffers.parser_relative_mask_eq_cache.len() == CACHE_CAPACITY {
            self.buffers.parser_relative_mask_eq_cache.remove(0);
        }
        self.buffers
            .parser_relative_mask_eq_cache
            .push(ParserRelativeMaskEqCacheEntry {
                left: cache_left,
                right: cache_right,
                admitted,
            });
        true
    }

    /// Advances the state by the token. If it isn't allowed, returns `false` and terminates the state.
    #[must_use]
    pub fn commit_token(
        &mut self,
        token_id: u32,
    ) -> bool {
        self.commit_token_raw(token_id).is_ok()
    }

    /// Advances the state by each token in order, stopping at the first one that isn't allowed (which terminates the state). Returns how many were committed.
    #[must_use]
    pub fn commit_tokens(&mut self, token_ids: &[u32]) -> usize {
        for (i, &token_id) in token_ids.iter().enumerate() {
            if !self.commit_token(token_id) {
                return i;
            }
        }
        token_ids.len()
    }

    /// Return Some when root termination policy handles this commitment.
    fn commit_end_token(&mut self, token_id: u32) -> Option<Result<(), String>> {
        if self.terminated {
            self.reject_after_failed_commit();
            return Some(Err("sequence has already terminated".to_owned()));
        }
        if self.constraint.end_tokens.binary_search(&token_id).is_err() {
            return None;
        }
        if !self.frontier_complete() {
            // An end token before a complete match is a disallowed commit. The
            // state is rejected, mirroring every other failed commit.
            self.reject_after_failed_commit();
            return Some(Err(format!(
                "commit_token: end token {token_id} is not allowed for this prefix"
            )));
        }
        if let Some(markers) = self.markers.as_deref_mut() { markers.begin(); }
        self.terminated = true;
        self.generation = self.generation.wrapping_add(1);
        *self.mask_cache.lock().unwrap() = None;
        Some(Ok(()))
    }

    pub(crate) fn commit_token_raw(&mut self, token_id: u32) -> Result<(), String> {
        if !self.constraint.parser_has_controls() { return self.commit_token_raw_inner(token_id); }
        match glrmask_invariant::__private::catch_compilation_resource_limit(|| self.commit_token_raw_inner(token_id)) {
            Ok(result) => result,
            Err(message) => {
                self.reject_after_control_resource_limit();
                Err(format!("parser resource limit: {message}"))
            }
        }
    }

    fn reject_after_control_resource_limit(&mut self) {
        self.reject_after_failed_commit();
        // An unwind may have crossed a diagnostic mask guard. Do not retain
        // a poisoned mutex or a cached mask that was computed mid-unwind.
        self.mask_cache = std::sync::Mutex::new(None);
        self.mask_scratch = std::sync::Arc::new(std::sync::Mutex::new(
            crate::runtime::state::MaskScratch::for_constraint(&self.constraint)));
    }

    fn commit_token_raw_inner(&mut self, token_id: u32) -> Result<(), String> {
        if let Some(result) = self.commit_end_token(token_id) { return result; }
        let constraint = self.constraint.clone();
        let bytes = constraint.token_bytes_for_id(token_id);
        if !self.knows_token_id(token_id) {
            self.reject_after_failed_commit();
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let result = self.commit_token_actual(token_id);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            &constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result
    }

    pub(crate) fn commit_token_dynamic(&mut self, token_id: u32) -> Result<(), String> {
        if let Some(result) = self.commit_end_token(token_id) { return result; }
        let constraint = self.constraint.clone();
        let bytes = constraint.token_bytes_for_id(token_id);
        if !self.knows_token_id(token_id) {
            self.reject_after_failed_commit();
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = if assertion_flags & COMMIT_ASSERT_MASK_EQUIVALENCE != 0 {
            let mut mask = vec![0u32; constraint.mask_bit_len().div_ceil(32)];
            self.fill_mask_dynamic(&mut mask);
            Some(token_in_mask(&mask, token_id))
        } else {
            None
        };
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let result = self.commit_token_actual(token_id);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            &constraint,
            token_id,
            bytes,
            was_in_mask,
            None,
            &self.state,
            result.is_ok(),
        );
        result
    }

    pub(crate) fn commit_token_timed_ns(&mut self, token_id: u32) -> Result<u64, String> {
        if self.terminated || !self.constraint.end_tokens.is_empty() {
            let policy_started = std::time::Instant::now();
            if let Some(result) = self.commit_end_token(token_id) {
                return result.map(|()| policy_started.elapsed().as_nanos() as u64);
            }
        }
        use std::time::Instant;

        let constraint = self.constraint.clone();
        let bytes = constraint.token_bytes_for_id(token_id);
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let start = Instant::now();
        let result = self.commit_token_actual(token_id);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        // Cache-preservation proofs are part of commit's runtime cost. Keep
        // them inside the timed interval so TBM/commit reporting cannot make a
        // mask optimization look free merely by shifting work after the timer.
        let total_ns = start.elapsed().as_nanos() as u64;
        assert_commit_oracles(
            &constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result.map(|()| total_ns)
    }

    pub(crate) fn commit_token_profiled(&mut self, token_id: u32) -> Result<CommitProfile, String> {
        if let Some(result) = self.commit_end_token(token_id) {
            return result.map(|()| CommitProfile::default());
        }
        let constraint = self.constraint.clone();
        let bytes = constraint.token_bytes_for_id(token_id);
        let has_special = constraint.has_special_token_id(token_id);
        if bytes.is_none() && !has_special {
            self.reject_after_failed_commit();
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let total_started_at = std::time::Instant::now();
        let result = self.commit_transaction(|this| {
            let special = if has_special {
                advance_special_token_paths_profiled(&constraint, &this.state, token_id, None)
            } else {
                SpecialTokenAdvanceProfile::default()
            };
            let mut profile = if let Some(bytes) = bytes {
                match commit_bytes_impl_profiled(
                    &constraint,
                    &mut this.state,
                    bytes,
                    &mut this.buffers,
                    None,
                    true,
                ) {
                    Ok(profile) => profile,
                    Err(_) => {
                        this.state.clear();
                        this.buffers.reset_all();
                        CommitProfile::default()
                    }
                }
            } else {
                this.state.clear();
                CommitProfile::default()
            };
            apply_special_token_advance_profile(&mut profile, &special);
            let special_merge_result =
                merge_special_token_paths(&constraint, &mut this.state, special.paths);
            profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
            special_merge_result
                .and_then(|()| finish_token_commit(&this.state))
                .map(|()| profile)
        });
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            &constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result
    }

    pub(crate) fn commit_token_per_advance(
        &mut self,
        token_id: u32,
    ) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile), String> {
        if let Some(result) = self.commit_end_token(token_id) {
            return result.map(|()| (Vec::new(), final_stacks(&self.state), CommitProfile::default()));
        }
        let constraint = self.constraint.clone();
        let bytes = constraint.token_bytes_for_id(token_id);
        let has_special = constraint.has_special_token_id(token_id);
        if bytes.is_none() && !has_special {
            self.reject_after_failed_commit();
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let assertion_flags = commit_assertion_flags();
        let was_in_mask = snapshot_mask_membership(self, token_id, assertion_flags);
        let equivalence_reference = (assertion_flags & COMMIT_ASSERT_FAST_PATH_EQUIVALENCE != 0)
            .then(|| self.state.clone());
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let total_started_at = std::time::Instant::now();
        let mut advances = Vec::new();
        let result = self.commit_transaction(|this| {
            let special = if has_special {
                advance_special_token_paths_profiled(
                    &constraint,
                    &this.state,
                    token_id,
                    Some(&mut advances),
                )
            } else {
                SpecialTokenAdvanceProfile::default()
            };
            let mut profile = if let Some(bytes) = bytes {
                match commit_bytes_impl_profiled(
                    &constraint,
                    &mut this.state,
                    bytes,
                    &mut this.buffers,
                    Some(&mut advances),
                    profile_allow_fast_paths(),
                ) {
                    Ok(profile) => profile,
                    Err(_) => {
                        this.state.clear();
                        this.buffers.clear_all();
                        advances.clear();
                        CommitProfile::default()
                    }
                }
            } else {
                this.state.clear();
                CommitProfile::default()
            };
            apply_special_token_advance_profile(&mut profile, &special);
            let special_merge_result =
                merge_special_token_paths(&constraint, &mut this.state, special.paths);
            profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
            special_merge_result
                .and_then(|()| finish_token_commit(&this.state))
                .map(|()| (advances, final_stacks(&this.state), profile))
        });
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        assert_commit_oracles(
            &constraint,
            token_id,
            bytes,
            was_in_mask,
            equivalence_reference,
            &self.state,
            result.is_ok(),
        );
        result
    }

    /// Advances the state by the bytes. If they aren't allowed, returns `false` and terminates the state.
    #[must_use]
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return !self.is_terminated() || self.is_accepting();
        }
        self.commit_bytes_raw(bytes).is_ok()
    }

    pub(crate) fn commit_bytes_raw(&mut self, bytes: &[u8]) -> Result<(), String> {
        if !self.constraint.parser_has_controls() { return self.commit_bytes_raw_inner(bytes); }
        match glrmask_invariant::__private::catch_compilation_resource_limit(|| self.commit_bytes_raw_inner(bytes)) {
            Ok(result) => result,
            Err(message) => {
                self.reject_after_control_resource_limit();
                Err(format!("parser resource limit: {message}"))
            }
        }
    }

    fn commit_bytes_raw_inner(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.terminated {
            self.reject_after_failed_commit();
            return Err("sequence has already terminated".to_owned());
        }
        if let Some(markers) = self.markers.as_deref_mut() { markers.begin(); }
        let mask_state_before_commit = self.snapshot_current_mask_state();
        let result = self.commit_bytes_actual(bytes);
        self.finish_commit_generation(mask_state_before_commit, result.is_ok());
        result
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{runtime::Constraint as Constraint, DynamicConstraint, Grammar, Vocab};
    use std::collections::BTreeSet;

    type CanonicalCommitState =
        Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)>;

    #[test]
    fn unique_actionable_top_matches_isolation_reference_including_empty_paths() {
        fn reference(constraint: &Constraint, gss: &ParserGSS, terminal: u32) -> Option<ParserGSS> {
            if constraint.parser_has_controls() || constraint.has_template_parser() {
                return None;
            }
            let mut selected = None;
            for top in gss.peek_values() {
                let Some(action) = constraint.table.action(top, terminal) else { continue; };
                if selected.is_some() { return None; }
                selected = Some((top, action));
            }
            let (top, action) = selected?;
            let isolated = gss.isolate(Some(top));
            (!isolated.is_empty())
                .then(|| apply_single_top_action_fast(constraint, &isolated, top, terminal, action))
                .flatten()
        }
        let vocab = Vocab::from_entries_unchecked(["a", "b", "ab", "ba", ",", " ", "(", ")"]
            .into_iter().enumerate().map(|(i, token)| (i as u32, token.as_bytes().to_vec())).collect());
        let grammars = [
            r#"start start; t A = "a" | "ab"; t B = "a" | "ba";
                nt start = A B? | B A?;"#,
            r#"start start; ignore WS; t WS = " "+; t A = "a"+; t B = "a"+ "b"?;
                nt item = A | B; nt start = item ("," item)*;"#,
            r#"start start; t A = "a"; nt start = A? | "(" `start` ")";"#,
        ];
        let empty = ParserGSS::from_single_stack(Vec::new(), TerminalsDisallowed::new());
        let mut exclusive = 0usize;
        let mut with_empty = 0usize;
        let mut comparisons = 0usize;
        for source in grammars {
            let built = Constraint::compile(Grammar::glrm(source), &vocab).unwrap();
            let loaded = Constraint::load(built.save(), &vocab).unwrap();
            for constraint in [&built, &loaded] {
                let mut frontier = vec![constraint.start()];
                let mut seen = BTreeSet::new();
                for depth in 0..=3 {
                    let mut next = Vec::new();
                    for state in frontier {
                        if !seen.insert(canonical_commit_state(&state.state)) { continue; }
                        for gss in state.state.values() {
                            for input in [gss.clone(), gss.merge(&empty)] {
                                if let Some(top) = input.single_exclusive_top_value() {
                                    exclusive += 1;
                                    assert!(input.ptr_eq(&input.isolate(Some(top))));
                                } else {
                                    with_empty += 1;
                                }
                                for terminal in 0..constraint.parser_terminal_count() {
                                    let expected = reference(constraint, &input, terminal);
                                    let actual = try_advance_unique_actionable_top_fast(constraint, &input, terminal);
                                    assert_eq!(actual.as_ref().map(canonical_gss), expected.as_ref().map(canonical_gss),
                                        "terminal={terminal} depth={depth} source={source}");
                                    comparisons += 1;
                                }
                            }
                        }
                        if depth < 3 {
                            let mask = state.mask();
                            for (token, _) in constraint.token_bytes_iter() {
                                if !token_in_mask(&mask, token) { continue; }
                                let mut advanced = state.clone();
                                assert!(advanced.commit_token(token));
                                next.push(advanced);
                            }
                        }
                    }
                    frontier = next;
                }
            }
        }
        assert!(exclusive > 0 && with_empty > 0 && comparisons > 20);
    }

    #[test]
    fn reduce_chain_output_factoring_preserves_stacks_and_exclusions() {
        let plain = TerminalsDisallowed::new();
        let guarded = plain.clone().with_insert(7, 3);
        for depth in [1usize, 2, 14, 64, 256] {
            let prefix: Vec<u32> = (0..depth as u32).collect();
            let extend = |suffix: &[u32]| {
                let mut stack = prefix.clone();
                stack.extend_from_slice(suffix);
                stack
            };
            let cases = vec![
                vec![],
                vec![(prefix.clone(), plain.clone())],
                vec![(prefix.clone(), plain.clone()), (prefix.clone(), plain.clone())],
                vec![(extend(&[500]), plain.clone()), (extend(&[501]), plain.clone())],
                vec![(extend(&[500, 502]), plain.clone()), (extend(&[501, 502]), plain.clone())],
                vec![(prefix.clone(), plain.clone()), (extend(&[501]), plain.clone())],
                vec![(vec![], plain.clone()), (prefix.clone(), plain.clone())],
                vec![(prefix.clone(), plain.clone()), (prefix.clone(), guarded.clone())],
                vec![(extend(&[500]), guarded.clone()), (extend(&[501]), plain.clone())],
            ];
            for outputs in cases {
                let reference = ParserGSS::from_stacks(&outputs);
                let actual = materialize_reduce_chain_outputs(outputs);
                assert!(actual.semantically_eq(&reference, 32).unwrap(), "depth={depth}");
            }
        }
    }

    #[test]
    fn lexer_only_actionable_cache_keeps_distinct_lexer_lanes() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"ab".to_vec()), (1, b"cd".to_vec()),
            (2, b"abcd".to_vec()), (3, b"abx".to_vec()),
            (4, b"zabc".to_vec()),
        ]);
        let compiled = Constraint::compile(Grammar::glrm(r#"
            start document;
            pragma glrmask {
                lexer group first = A, C;
                lexer group second = B;
            }
            t A = "abx";
            t C = "abc";
            t B = "abcd";
            nt document = A | B | "z" C;
        "#), &vocab).unwrap();
        for constraint in [&compiled, &Constraint::load(compiled.save(), &vocab).unwrap()] {
            let mut state = constraint.start();
            assert!(state.commit_bytes(b"ab"));
            let mut reference = state.state.clone();
            assert!(commit_token_no_fast_path_reference(constraint, &mut reference, 1).is_ok());
            let mut actual = state.state.clone();
            let mut scratch = CommitBuffers::default();
            assert!(commit_token_impl(constraint, &mut actual, &mut scratch, 1).is_ok(),
                "lexer lanes sharing parser stacks must not share actionable-match answers");
            assert_commit_fast_path_equivalence(constraint, state.state.clone(), 1, &actual, true);
        }
    }

    #[test]
    fn unfinished_ignore_prefixes_survive_all_commit_paths() {
        let fragments: &[&[u8]] = &[
            b"a", b"+", b"b", b"//", b"/////", b"line", b"\n",
            b"/*", b"/***", b"body", b"*/", b" /*unfinished", b"//line\n+b",
        ];
        let vocab = Vocab::from_entries_unchecked(fragments.iter().enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec())).collect());
        for partitions in [
            "",
            "pragma glrmask { lexer group words = WORD; lexer group trivia = WS; }",
        ] {
            let source = format!(r#"
                start document;
                {partitions}
                ignore WS;
                t WS = " "+ | "//" b/[^\n]/* "\n" | "/*" (b/[^*]/ | "*" b/[^\/]/)* "*/";
                t WORD = /[a-z]+/;
                nt document = WORD ("+" WORD)*;
            "#);
            let built = Constraint::compile(Grammar::glrm(&source), &vocab).unwrap();
            let loaded = Constraint::load(built.save(), &vocab).unwrap();
            for constraint in [&built, &loaded] {
                for ids in [
                    vec![0,4,5,6,1,2],
                    vec![0,7,9,10,1,2],
                    vec![0,8,9,10,1,2],
                    vec![0,11,10,1,2],
                    vec![0,12],
                ] {
                    let mut actual = constraint.start();
                    let mut bytewise = constraint.start();
                    for token in ids {
                        let before = actual.state.clone();
                        assert!(token_in_mask(&actual.mask(), token),
                            "ignore prefix absent from mask partitions={partitions} token={token}");
                        let mut reference = before.clone();
                        assert!(commit_token_no_fast_path_reference(constraint, &mut reference, token).is_ok(),
                            "general queue must retain unfinished ignore token={token}");
                        assert!(actual.commit_token(token));
                        assert_commit_fast_path_equivalence(constraint, before, token, &actual.state, true);
                        for byte in fragments[token as usize] { assert!(bytewise.commit_bytes(&[*byte])); }
                        assert_eq!(actual.mask(), bytewise.mask(), "token-vs-byte prefix token={token}");
                    }
                    assert!(actual.is_accepting());
                    assert!(bytewise.is_accepting());
                }
            }
        }
    }

    #[test]
    fn recursive_batched_admission_matches_scalar_provider_reference() {
        use crate::ds::bitset::BitSet;
        let vocab = Vocab::from_entries_unchecked(b"[ab]! ".iter().enumerate()
            .map(|(id,byte)|(id as u32,vec![*byte])).collect());
        for source in [
            r#"start child; t WORD = /[ab]+/; nt child = WORD;"#,
            r#"start child; ignore WS; t WS = " "+; t WORD = /[ab]+/; nt child = WORD?;"#,
            r#"start child; t A = "a"; t AB = "ab"; nt child = (A | AB)*;"#,
        ] {
            let child = Constraint::compile(Grammar::glrm(source), &vocab).unwrap();
            let middle = Constraint::compile(Grammar::glrm(
                r#"start middle; extern grammar child; nt middle = "[" child "]";"#,
            ), &vocab).unwrap().bind_grammar_dynamic_boundary("child", child).unwrap();
            let outer = Constraint::compile(Grammar::glrm(
                r#"start outer; extern grammar middle; nt outer = middle "!";"#,
            ), &vocab).unwrap().bind_grammar_dynamic_boundary("middle", middle).unwrap();
            let loaded = Constraint::load(outer.save(), &vocab).unwrap();
            for constraint in [&outer, &loaded] {
                for prefix in ["", "[", "[a", "[ab", "[ab]", "[ab]!", "[ ", "[]"] {
                    let mut state = constraint.start();
                    if !state.commit_bytes(prefix.as_bytes()) {continue;}
                    let runtime = &state.constraint;
                    let count = runtime_terminal_count(runtime);
                    for (_, gss) in state.state.iter() {
                        for stride in [1,2,3] {
                            let mut candidates=BitSet::new(count);
                            for terminal in (0..count).step_by(stride) {candidates.set(terminal);}
                            let actual=runtime.compact_segmented_parser_admitted_terminals(gss,&candidates).unwrap();
                            let mut expected=BitSet::new(count);
                            for terminal in candidates.iter_ones() {
                                if runtime.compact_segmented_parser_may_advance_on(gss,terminal as u32).unwrap() {
                                    expected.set(terminal);
                                }
                            }
                            assert_eq!(actual,expected,"prefix={prefix} stride={stride} source={source}");
                            assert_eq!(runtime.compact_segmented_parser_may_advance_on_any(gss,&candidates),Some(!expected.is_empty()));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unchanged_runtime_state_preserves_fill_mask_cache() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
        ]);
        let constraint = Constraint::compile(
            Grammar::glrm(
                r#"glrm 1;
                start start;
                t A = /a+/;
                t B = "b";
                nt start = A B;
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let mut state = constraint.start();
        assert!(state.commit_token(0));

        // Populate the ordinary per-sequence mask cache while inside A. A
        // further `a` is an exact lexer self-loop: no parser terminal is
        // consumed and the complete runtime state remains identical.
        let cached_mask = state.mask();
        let before_generation = state.generation;
        let before_state = state.state.clone();
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_eq!(cache.as_ref().unwrap().generation, before_generation);
        }

        assert!(state.commit_token(0));
        assert_eq!(state.generation, before_generation + 1);
        assert_eq!(before_state, state.state);
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_eq!(cache.as_ref().unwrap().generation, state.generation);
            assert_eq!(cache.as_ref().unwrap().mask, cached_mask);
        }
        assert_eq!(state.mask(), cached_mask);

        // Completing A and advancing the parser must invalidate the cached
        // generation. The following mask is therefore recomputed for B.
        assert!(state.commit_token(2));
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_ne!(cache.as_ref().unwrap().generation, state.generation);
        }
    }

    #[test]
    fn unchanged_dynamic_mask_projection_preserves_fill_mask_cache() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_json_schema(
            r#"{"type":"string","maxLength":1000000000}"#,
            &vocab,
        )
        .unwrap();
        let mut state = dynamic.inner.start();

        // Enter the quoted lazy bounded-string residual, consume one interior
        // byte, then cache its mask. Far from the upper bound, another interior
        // byte changes the exact symbolic residual coordinate while the parser
        // has not seen a terminal boundary and the finite vocabulary-horizon
        // mask coordinate remains unchanged.
        assert!(state.commit_token(0)); // opening quote
        assert!(state.commit_token(1)); // first interior 'a'
        let cached_mask = state.mask();
        let before_generation = state.generation;
        let before_state = state.state.clone();

        assert!(state.commit_token(1)); // second interior 'a'
        assert_ne!(before_state, state.state, "test must exercise a changed exact lexer state");
        assert!(
            state.dynamic_mask_projection_state_eq(&before_state, &state.state),
            "changed exact state should remain equal in the mask-only coordinate",
        );
        {
            let cache = state.mask_cache.lock().unwrap();
            assert_eq!(cache.as_ref().unwrap().generation, before_generation + 1);
            assert_eq!(cache.as_ref().unwrap().mask, cached_mask);
        }
        assert_eq!(state.mask(), cached_mask);
    }

    #[test]
    fn recursive_radix_candidate_admission_matches_pointwise_exact_commit() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"X[".to_vec()),
            (2, b"X[a".to_vec()),
            (3, b"X[a]".to_vec()),
            (4, b"X[a]!".to_vec()),
            (5, b"X[a]!".to_vec()),
            (6, b"[".to_vec()),
            (7, b"[a".to_vec()),
            (8, b"[a]".to_vec()),
            (9, b"[a]!".to_vec()),
            (10, b"a".to_vec()),
            (11, b"a]".to_vec()),
            (12, b"a]!".to_vec()),
            (13, b"]".to_vec()),
            (14, b"]!".to_vec()),
            (15, b"!".to_vec()),
        ]);
        let child = Constraint::compile(
            Grammar::glrm("glrm 1; start child; nt child = \"[\" \"a\" \"]\";"),
            &vocab,
        )
        .unwrap();
        let parent = Constraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());

        for path in [
            &[][..],
            &[0][..],
            &[0, 6][..],
            &[0, 6, 10][..],
            &[0, 6, 10, 13][..],
        ] {
            let mut state = bound.start();
            for &token in path {
                assert!(state.commit_token(token));
            }

            let mut batch_candidates = bound.token_bytes_iter().collect::<Vec<_>>();
            let mut batch_buffers = CommitBuffers::default();
            let mut batched = Vec::new();
            admissible_byte_token_candidates_from_state_exact(
                &bound,
                &state.state,
                &mut batch_buffers,
                &mut batch_candidates,
                &mut batched,
            );
            batched.sort_unstable();

            let mut pointwise_buffers = CommitBuffers::default();
            let mut pointwise = bound
                .token_bytes_iter()
                .filter_map(|(token_id, _)| {
                    token_admissible_from_state_exact(
                        &bound,
                        &state.state,
                        &mut pointwise_buffers,
                        token_id,
                    )
                    .then_some(token_id)
                })
                .collect::<Vec<_>>();
            pointwise.sort_unstable();
            assert_eq!(batched, pointwise, "candidate mismatch after path {path:?}");
        }
    }

    #[test]
    fn recursive_scoped_tokenizer_exec_uses_only_the_active_leaf() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xa!".to_vec()),
        ]);
        let child = Constraint::compile(
            Grammar::glrm("glrm 1; start child; t A = \"a\"; nt child = A;"),
            &vocab,
        )
        .unwrap();
        let parent = Constraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; t X = \"X\"; t BANG = \"!\"; nt document = X child BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let local_x = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "X")
            .unwrap() as u32;
        let local_a = child
            .terminal_display_names
            .iter()
            .position(|name| name == "A")
            .unwrap() as u32;
        let bound = parent.bind_grammar("child", child).unwrap();
        let loaded = Constraint::load(bound.save(), &vocab).unwrap();
        for constraint in [&bound, &loaded] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            assert_eq!(layout.leaves.len(), 2);
            let scoped_x = constraint.recursive_terminal_scoped_id(0, local_x).unwrap();
            let scoped_a = constraint.recursive_terminal_scoped_id(1, local_a).unwrap();
            assert!(scoped_x >= constraint.parser_terminal_count());
            assert!(scoped_a >= constraint.parser_terminal_count());

            let root_reset = constraint.recursive_tokenizer_reset_state(0).unwrap();
            let root_x = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
                constraint,
                b"X",
                root_reset,
            )
            .unwrap();
            assert!(root_x.matches.iter().any(|matched| matched.id == scoped_x));
            assert!(root_x.end_state.iter().all(|&state| {
                constraint.recursive_tokenizer_leaf_state(state).unwrap().0 == 0
            }));
            let root_a = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
                constraint,
                b"a",
                root_reset,
            )
            .unwrap();
            assert!(
                root_a.matches.iter().all(|matched| matched.id != scoped_a),
                "child terminal leaked from inactive leaf: {:?}",
                root_a.matches,
            );

            let child_reset = constraint.recursive_tokenizer_reset_state(1).unwrap();
            let child_a = tokenizer_scan::execute_recursive_tokenizer_from_state_small(
                constraint,
                b"a",
                child_reset,
            )
            .unwrap();
            assert!(child_a.matches.iter().any(|matched| matched.id == scoped_a));
            assert!(child_a.end_state.iter().all(|&state| {
                constraint.recursive_tokenizer_leaf_state(state).unwrap().0 == 1
            }));
        }
    }

    fn canonical_commit_state(
        state: &ParserStateMap,
    ) -> CanonicalCommitState {
        canonical_commit_state_for_equivalence_assert(state)
    }

    #[test]
    fn exact_prefilter_rejects_multiple_conditional_candidates_independent_of_row_order() {
        let mut terminals = crate::ds::bitset::BitSet::new(8);
        terminals.set(1);
        terminals.set(2);
        let unconditional = crate::ds::bitset::BitSet::new(8);

        let row_a = ActionRow::from_iter([
            (1, Action::Reduce(0, 1)),
            (2, Action::Reduce(0, 1)),
        ]);
        let row_b = ActionRow::from_iter([
            (2, Action::Reduce(0, 1)),
            (1, Action::Reduce(0, 1)),
        ]);

        assert_eq!(
            single_conditional_candidate(&row_a, &unconditional, &terminals),
            Err(()),
        );
        assert_eq!(
            single_conditional_candidate(&row_b, &unconditional, &terminals),
            Err(()),
        );

        let mut one_terminal = crate::ds::bitset::BitSet::new(8);
        one_terminal.set(2);
        assert_eq!(
            single_conditional_candidate(&row_a, &unconditional, &one_terminal),
            Ok(Some(2)),
        );
    }

    #[test]
    fn batched_and_cached_end_state_admission_match_pointwise_simulation() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a";
                t AB = "ab";
                t B = "b";
                t C = "c";
                nt item = A | AB | B;
                nt start = item C;
            "#,
            &Vocab::from_entries_unchecked(vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"c".to_vec()),
                (4, b"ac".to_vec()),
                (5, b"abc".to_vec()),
            ]),
        )
        .unwrap();
        let state = constraint.start();
        let gss = state.state.values().next().unwrap();
        let initial = constraint.runtime_commit_initial_state();
        let non_initial = (0..constraint.tokenizer.num_states())
            .find(|&end_state| {
                end_state != initial
                    && !constraint
                        .tokenizer
                        .possible_future_terminals(end_state)
                        .is_empty()
            })
            .expect("test tokenizer should have a live non-initial state");
        // Repeating one exact continuation is enough to exercise batching and
        // cache reuse; production callers usually provide many distinct states,
        // but the theorem does not depend on their distinctness.
        let end_states = vec![initial, non_initial, non_initial];

        let admitted = batched_end_state_admitted_terminals(&constraint, gss, &end_states);
        assert!(admitted.is_none(), "native relations decline the LR batching shortcut");
        for &end_state in &end_states {
            assert_eq!(end_state_may_advance_with_batch(&constraint, gss, end_state, admitted.as_ref()),
                end_state_may_advance(&constraint, gss, end_state));
        }
        let mut cache = SmallVec::<[ParserAdmissionCacheEntry; 8]>::new();
        for _ in 0..2 {
            assert!(cached_batched_end_state_admission(&constraint,gss,&end_states,&mut cache).is_none());
            assert!(cache.is_empty(), "declined batching must leave the cache unchanged");
        }
        for (token, _) in constraint.token_bytes_iter() {
            let admitted = state.mask()[token as usize / 32] & (1 << (token % 32)) != 0;
            assert_eq!(state.clone().commit_token(token), admitted, "token={token}");
        }
    }

    #[test]
    fn admission_cache_reset_drops_persistent_gss_references() {
        let gss = ParserGSS::from_single_stack(
            vec![0_u32, 1],
            TerminalsDisallowed::new(),
        );
        let mut buffers = CommitBuffers::default();
        buffers.admission_cache.push(ParserAdmissionCacheEntry {
            gss,
            tested: crate::ds::bitset::BitSet::new(2),
            admitted: crate::ds::bitset::BitSet::new(2),
            boolean_queries: SmallVec::new(),
        });
        buffers.clear_all();
        assert_eq!(
            buffers.admission_cache.len(),
            1,
            "ordinary scratch reuse should preserve admission facts",
        );
        buffers.reset_all();
        assert!(buffers.admission_cache.is_empty());
    }


    #[test]
    #[ignore]
    fn debug_selected10_completed_child_lookahead_actions() {
        use crate::compiler::glr::analysis::EOF;
        use crate::compiler::glr::parser::stack_admissible_terminals;
        use crate::compiler::glr::table::Action;
        use crate::ds::bitset::BitSet;
        use std::collections::BTreeMap;

        let path = crate::env::var("GLRMASK_DEBUG_ARTIFACT").expect("GLRMASK_DEBUG_ARTIFACT");
        let bytes = std::fs::read(path).unwrap();
        let vocab = crate::compiler::constraint_compose::load_vocab(
            &crate::env::var("GLRMASK_DEBUG_VOCAB").expect("GLRMASK_DEBUG_VOCAB"));
        let constraint = Constraint::load(&bytes, &vocab).unwrap();
        let mut state = constraint.start();
        assert!(state.commit_bytes(b"const x = tools.tool_0({})"));
        eprintln!("STACKS {:?}", state.debug_parser_stacks());
        for (_, gss) in state.state.iter() {
            let all = BitSet::all(constraint.parser_terminal_count() as usize + 1);
            let admitted = stack_admissible_terminals(&constraint.table, gss, &all);
            let mut factor = gss.clone();
            let mut blockers = Vec::<u32>::new();
            for depth in 0..16 {
                let top = factor.single_exclusive_top_value();
                let admitted_now =
                    stack_admissible_terminals(&constraint.table, &factor, &all);
                eprintln!(
                    "FACTOR_DEPTH {depth} top={top:?} stack={:?} admitted={:?} blockers={:?}",
                    factor.to_stacks(64),
                    admitted_now.iter_ones().collect::<Vec<_>>(),
                    blockers,
                );
                let Some((next, blocked)) =
                    crate::compiler::glr::parser::lookahead_reduction_factor(
                        &constraint.table,
                        &factor,
                    )
                else {
                    break;
                };
                for terminal in blocked {
                    if !blockers.contains(&terminal) {
                        blockers.push(terminal);
                    }
                }
                factor = next;
            }
            for top in gss.peek_values() {
                let mut kinds = BTreeMap::<String, Vec<u32>>::new();
                for bit in admitted.iter_ones() {
                    let terminal = if bit == constraint.parser_terminal_count() as usize {
                        EOF
                    } else {
                        bit as u32
                    };
                    let desc = match constraint.table.action(top, terminal) {
                        Some(Action::Reduce(nt, len)) => format!("reduce:{nt}:{len}"),
                        Some(Action::Shift(target, replace)) => format!("shift:{target}:{replace}"),
                        Some(Action::StackShifts(shifts)) => format!("stackshifts:{}", shifts.len()),
                        Some(Action::GuardedStackShifts(shifts)) => format!("guarded:{}", shifts.len()),
                        Some(Action::ReplaceShifts(targets)) => format!("replace:{}", targets.len()),
                        Some(Action::Split { shift, reduces, accept }) => format!("split:{shift:?}:{reduces:?}:{accept}"),
                        Some(Action::Accept) => "accept".into(),
                        Some(Action::Skip) => "skip".into(),
                        None => "none".into(),
                    };
                    kinds.entry(desc).or_default().push(terminal);
                }
                eprintln!("TOP {top} ADMITTED {} KINDS {}", admitted.count_ones(), kinds.len());
                for (kind, terminals) in kinds {
                    eprintln!("  {kind} count={} sample={:?}", terminals.len(), &terminals[..terminals.len().min(24)]);
                }
            }
        }
    }


    #[test]
    #[ignore]
    fn debug_selected10_core_fast_weight_transport() {
        use crate::compiler::glr::labels::DEFAULT_LABEL;
        use crate::ds::weight::Weight;

        fn accepted_weight(constraint: &Constraint, labels: &[u32]) -> Weight {
            let parser_dwa = constraint.parser_dwa.for_compilation().unwrap();
            let mut state = parser_dwa.start_state();
            let mut weight = Weight::all();
            for &label in labels {
                let row = &parser_dwa.states()[state as usize];
                let Some((target, edge_weight)) = row
                    .transitions
                    .get(&(label as i32))
                    .or_else(|| row.transitions.get(&DEFAULT_LABEL))
                else {
                    return Weight::empty();
                };
                weight = weight.intersection(edge_weight);
                state = *target;
            }
            let Some(final_weight) = parser_dwa.states()[state as usize]
                .final_weight
                .as_ref()
            else {
                return Weight::empty();
            };
            weight.intersection(final_weight)
        }

        fn dump(name: &str, constraint: &Constraint, tokenizer_state: u32, token: u32) {
            let internal_token = constraint
                .original_token_internal_at(token)
                .unwrap_or(u32::MAX);
            let tsids = constraint.internal_tsids_for_state(tokenizer_state).to_vec();
            eprintln!("WEIGHT_TRANSPORT {name} tokenizer_state={tokenizer_state} tsids={tsids:?} token={token} internal_token={internal_token}");
            for labels in [&[22_u32][..], &[22_u32, 0][..]] {
                let weight = accepted_weight(constraint, labels);
                let memberships = tsids
                    .iter()
                    .map(|&tsid| (tsid, internal_token != u32::MAX && weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len()))
                    .collect::<Vec<_>>();
                eprintln!("WEIGHT_TRANSPORT {name} labels={labels:?} empty={} memberships={memberships:?}", weight.is_empty());
            }
            for key in [22_i32, DEFAULT_LABEL] {
                if let Some(weight) = constraint.non_dwa_weights.materialized().and_then(|w| w.parser_top_accept.get(&key)) {
                    let memberships = tsids.iter().map(|&tsid| (tsid, weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len())).collect::<Vec<_>>();
                    eprintln!("WEIGHT_TRANSPORT {name} top_accept key={key} memberships={memberships:?}");
                }
                if let Some(parts) = constraint.non_dwa_weights.materialized().and_then(|w| w.parser_top_accept_parts.get(&key)) {
                    for (part, weight) in parts.iter().enumerate() {
                        let memberships = tsids.iter().map(|&tsid| (tsid, weight.tokens_for_tsid(tsid).contains(internal_token), weight.tokens_for_tsid(tsid).len())).collect::<Vec<_>>();
                        eprintln!("WEIGHT_TRANSPORT {name} top_part key={key} part={part} memberships={memberships:?}");
                    }
                }
            }
            let mut l1 = Vec::new();
            constraint.for_each_direct_regular_l1_acceptance(22, |weight| {
                l1.push(tsids.iter().map(|&tsid| { let tokens = weight.token_set_for_tsid(tsid).map(|tokens| tokens.to_range_set()).unwrap_or_default(); (tsid, tokens.contains(internal_token), tokens.len()) }).collect::<Vec<_>>());
            });
            eprintln!("WEIGHT_TRANSPORT {name} l1={l1:?}");
        }

        let core_path = crate::env::var("GLRMASK_DEBUG_CORE_ARTIFACT").unwrap();
        let fast_path = crate::env::var("GLRMASK_DEBUG_ARTIFACT").unwrap();
        let vocab = crate::compiler::constraint_compose::load_vocab(
            &crate::env::var("GLRMASK_DEBUG_VOCAB").expect("GLRMASK_DEBUG_VOCAB"));
        let core = Constraint::load(std::fs::read(core_path).unwrap(), &vocab).unwrap();
        let fast = Constraint::load(std::fs::read(fast_path).unwrap(), &vocab).unwrap();
        dump("core", &core, 38, 443);
        dump("fast", &fast, 39, 443);
    }

    #[test]
    #[ignore]
    fn debug_selected10_boundary_shallow_acceptance() {
        use crate::compiler::glr::labels::DEFAULT_LABEL;
        use crate::ds::weight::Weight;

        fn accepted_weight(constraint: &Constraint, labels: &[u32]) -> Weight {
            let parser_dwa = constraint.parser_dwa.for_compilation().unwrap();
            let mut state = parser_dwa.start_state();
            let mut weight = Weight::all();
            for &label in labels {
                let row = &parser_dwa.states()[state as usize];
                let Some((target, edge_weight)) = row
                    .transitions
                    .get(&(label as i32))
                    .or_else(|| row.transitions.get(&DEFAULT_LABEL))
                else { return Weight::empty(); };
                weight = weight.intersection(edge_weight);
                state = *target;
            }
            let Some(final_weight) = parser_dwa.states()[state as usize].final_weight.as_ref()
            else { return Weight::empty(); };
            weight.intersection(final_weight)
        }

        fn contains(constraint: &Constraint, tokenizer_state: u32, token: u32, weight: &Weight) -> bool {
            let internal = constraint.original_token_internal_at(token).unwrap();
            constraint.internal_tsids_for_state(tokenizer_state)
                .iter()
                .any(|&tsid| weight.tokens_for_tsid(tsid).contains(internal))
        }

        fn dump_case(constraint: &Constraint, name: &str, tokenizer_state: u32, stack: &[u32], tokens: &[u32]) {
            let reversed = stack.iter().rev().copied().collect::<Vec<_>>();
            eprintln!("SHALLOW {name} tokenizer_state={tokenizer_state} stack={stack:?} tsids={:?}", constraint.internal_tsids_for_state(tokenizer_state));
            let start_final = constraint.parser_dwa.for_compilation().unwrap().states()[constraint.parser_dwa.start_state() as usize]
                .final_weight.clone().unwrap_or_else(Weight::empty);
            for &token in tokens {
                let mut first = contains(constraint, tokenizer_state, token, &start_final).then_some(0usize);
                for depth in 1..=reversed.len() {
                    let w = accepted_weight(constraint, &reversed[..depth]);
                    if contains(constraint, tokenizer_state, token, &w) {
                        first = Some(depth);
                        break;
                    }
                }
                eprintln!("SHALLOW {name} token={token} first_depth={first:?}");
            }
        }

        let full_path = crate::env::var("GLRMASK_DEBUG_FULL_ARTIFACT").unwrap();
        let vocab = crate::compiler::constraint_compose::load_vocab(
            &crate::env::var("GLRMASK_DEBUG_VOCAB").expect("GLRMASK_DEBUG_VOCAB"));
        let full = Constraint::load(std::fs::read(full_path).unwrap(), &vocab).unwrap();
        dump_case(&full, "tools_reset_a", 0, &[0,22,198,18], &[739,2446,7255,21966]);
        dump_case(&full, "tools_reset_b", 0, &[0,22,198,167], &[739,2446,7255,21966]);
        dump_case(&full, "tools_residual", 444, &[0,22,198], &[739,2446,7255,21966]);
        dump_case(&full, "tool0", 0, &[0,22,198,193,862], &[49209,69906]);
        dump_case(&full, "inside", 0, &[0,22,198,193,873,903], &[3033,3602,4649,9000,14419,16638,17041,29448,31893,32988,35183,36199,39942,44160,53511,71741,79237,82274,95445]);
        dump_case(&full, "await_tools", 0, &[0,22,198,109,193], &[3324,32809]);
    }

    #[test]
    #[ignore]
    fn debug_selected10_iterated_mask_factor() {
        use std::time::Instant;

        let fast_path = crate::env::var("GLRMASK_DEBUG_ARTIFACT").unwrap();
        let full_path = crate::env::var("GLRMASK_DEBUG_FULL_ARTIFACT").unwrap();
        let vocab = crate::compiler::constraint_compose::load_vocab(
            &crate::env::var("GLRMASK_DEBUG_VOCAB").expect("GLRMASK_DEBUG_VOCAB"));
        let fast = Constraint::load(std::fs::read(fast_path).unwrap(), &vocab).unwrap();
        let full = Constraint::load(std::fs::read(full_path).unwrap(), &vocab).unwrap();
        unsafe {
            crate::__private::set_env_var("GLRMASK_EXPERIMENT_STATIC_DYNAMIC_OVERLAY", None);
        }
        let prefixes: &[&[u8]] = &[
            b"",
            b"const x",
            b"const x =",
            b"const x = tools",
            b"const x = tools.tool_0",
            b"const x = tools.tool_0({",
            b"const x = tools.tool_0({})",
            b"const x = tools.tool_0({});",
            b"const r = await tools.",
        ];
        for &prefix in prefixes {
            let mut fs = full.start();
            let mut hs = fast.start();
            if !prefix.is_empty() {
                assert!(fs.commit_bytes(prefix));
                assert!(hs.commit_bytes(prefix));
            }
            let fm = fs.mask();
            let started = Instant::now();
            let hm = hs.mask();
            let elapsed = started.elapsed().as_micros();
            let mut extra = 0usize;
            let mut missing = 0usize;
            for (&left, &right) in hm.iter().zip(&fm) {
                extra += (left & !right).count_ones() as usize;
                missing += (right & !left).count_ones() as usize;
            }
            eprintln!(
                "IGNORE_PM prefix={:?} us={} extra={} missing={}",
                String::from_utf8_lossy(prefix), elapsed, extra, missing
            );
        }
        std::mem::forget(fast);
        std::mem::forget(full);
    }

    #[test]
    fn flat_stack_effect_can_replace_the_only_state() {
        let mut stack = FlatInlineStack::new();
        stack.push(36);

        assert_eq!(apply_flat_stack_effect(&mut stack, 1, &[37]), Some(true));
        assert_eq!(stack.as_slice(), &[37]);
    }

    #[test]
    fn compile_time_initial_commit_priming_preserves_runtime_semantics() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a";
                t B = "b";
                nt start = A B;
            "#,
            &Vocab::from_entries_unchecked(vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"x".to_vec()),
            ]),
        )
        .unwrap();

        let mut state = constraint.start();
        assert_eq!(
            state
                .mask()
                .iter()
                .enumerate()
                .flat_map(|(word, &bits)| {
                    (0..32).filter_map(move |bit| {
                        ((bits >> bit) & 1 == 1).then_some((word * 32 + bit) as u32)
                    })
                })
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1]),
        );
        assert!(state.commit_token(0));
        assert!(state.commit_token(2));
        assert!(state.is_accepting());

        let loaded = Constraint::load(constraint.save(), &crate::public_api::constraint_vocab(&constraint)).unwrap();
        let mut loaded_state = loaded.start();
        assert!(loaded_state.commit_token(1));
        assert!(loaded_state.is_accepting());
    }

    #[test]
    fn merged_key_frontier_reports_completion() {
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a";
                t B = "a" | "ab";
                nt item = A | B;
                nt start = item;
            "#,
            &Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]),
        )
        .unwrap();

        let mut state = constraint.start();
        assert!(state.commit_token(0));
        assert!(!state.state.has_duplicate_keys());
        assert!(state.is_accepting());

        let mut normalized = state.clone();
        normalized.state.normalize_duplicate_keys();
        assert!(normalized.is_accepting());
        assert_eq!(
            canonical_commit_state(&state.state),
            canonical_commit_state(&normalized.state),
        );
    }

    #[test]
    fn special_token_advance_matches_normalized_duplicate_key_state() {
        let constraint = Constraint::from_glrm_with_token_ids(
            r#"
                start start;
                t A = "a";
                t B = "a" | "ab";
                nt item = A | B;
                extern token EXACT_100;
                nt start = item EXACT_100;
            "#,
            &Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]),
            &[("EXACT_100", 100)],
            &[],
        )
        .unwrap();

        let mut flat = constraint.start();
        assert!(flat.commit_token(0));
        assert!(!flat.state.has_duplicate_keys());

        let mut normalized = flat.clone();
        normalized.state.normalize_duplicate_keys();
        assert_eq!(flat.mask(), normalized.mask());

        assert!(flat.commit_token(100));
        assert!(normalized.commit_token(100));
        assert!(flat.is_accepting());
        assert!(normalized.is_accepting());
        assert_eq!(
            canonical_commit_state(&flat.state),
            canonical_commit_state(&normalized.state),
        );
    }

    #[test]
    fn carried_virtual_stack_stops_when_stack_effect_exposes_branched_floor() {
        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 1, 10], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 2, 10], acc);
        let merged = left.merge(&right);

        let mut exhausted = merged
            .try_virtual_stack()
            .expect("merged common top should form a virtual stack");
        assert!(!try_apply_action_to_carried_virtual_stack(
            &mut exhausted,
            &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
                pop: 1,
                pushes: Vec::new(),
            }]),
        ));
        assert_eq!(exhausted.top(), Some(&10));

        let mut restored_common_top = merged
            .try_virtual_stack()
            .expect("merged common top should form a virtual stack");
        assert!(try_apply_action_to_carried_virtual_stack(
            &mut restored_common_top,
            &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
                pop: 1,
                pushes: vec![40],
            }]),
        ));
        assert_eq!(restored_common_top.top(), Some(&40));
        let mut stacks = restored_common_top.into_gss().to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(
            stacks.into_iter().map(|(stack, _)| stack).collect::<Vec<_>>(),
            vec![vec![0, 1, 40], vec![0, 2, 40]],
        );

        let mut emptied = ParserGSS::from_single_stack(
            vec![10],
            TerminalsDisallowed::new(),
        )
        .try_virtual_stack()
        .expect("single stack should form a virtual stack");
        assert!(!try_apply_action_to_carried_virtual_stack(
            &mut emptied,
            &Action::StackShifts(vec![crate::compiler::glr::table::StackShift {
                pop: 1,
                pushes: Vec::new(),
            }]),
        ));
        assert_eq!(emptied.top(), Some(&10));
    }

    fn canonical_gss(gss: &ParserGSS) -> Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)> {
        canonical_commit_state(&ParserStateMap::singleton(0, gss.clone()))
            .pop()
            .unwrap()
            .1
    }

    #[test]
    fn fused_exact_admission_matches_legacy_two_pass_reference() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a";
                t AB = "ab";
                t B = "b";
                nt item = A | AB | A B;
                nt start = item item?;
            "#,
            &vocab,
        )
        .unwrap();
        assert_eq!(
            constraint.parser_admission_policy(),
            AdmissionPolicy::ExactSimulation,
        );

        // Own compiler reference data directly; never restore a table in the
        // native Constraint whose commits and admission are under test.
        let grammar = constraint.template_parser().unwrap().link_grammar.as_ref().unwrap();
        let table = crate::compiler::glr::table::GLRTable::build_with_default_construction(
            &grammar.analyze(constraint.terminal_display_names.clone()),
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        let mut states = vec![constraint.start()];
        let mut after_a = constraint.start();
        assert!(after_a.commit_token(0));
        states.push(after_a);

        for state in states {
            for gss in state.state.values() {
                for terminal in 0..constraint.parser_terminal_count() {
                    let legacy = if stack_may_advance_on(&table, gss, terminal) {
                        let advanced = advance_parser_stacks(&constraint, gss, terminal);
                        (!advanced.is_empty()).then_some(advanced)
                    } else {
                        None
                    };
                    let fused = advance_parser_stacks_if_possible(&constraint, gss, terminal);
                    assert_eq!(
                        fused.as_ref().map(canonical_gss),
                        legacy.as_ref().map(canonical_gss),
                        "terminal={terminal} source={:#?}",
                        canonical_gss(gss),
                    );

                    let profiled =
                        advance_parser_stacks_profiled_if_possible(&constraint, gss, terminal);
                    assert_eq!(
                        (!profiled.advanced.is_empty())
                            .then(|| canonical_gss(&profiled.advanced)),
                        legacy.as_ref().map(canonical_gss),
                        "profiled terminal={terminal} source={:#?}",
                        canonical_gss(gss),
                    );
                    assert_eq!(profiled.may_ns, 0);
                }
            }
        }
    }

    fn top_local_prune_reference(
        constraint: &Constraint,
        gss: &ParserGSS,
        bytes: &[u8],
    ) -> ParserGSS {
        let prune_partition = |partition: ParserGSS| {
            partition.apply_and_prune_no_promote(
                |terminals_disallowed: &TerminalsDisallowed| {
                    if terminals_disallowed.is_empty() {
                        return Some(TerminalsDisallowed::new());
                    }

                    let mut remapped = BTreeMap::new();
                    for (&continuation_tokenizer_state, disallowed) in
                        terminals_disallowed.iter()
                    {
                        let execution = execute_tokenizer_from_state_small(
                            constraint,
                            bytes,
                            continuation_tokenizer_state,
                        );
                        if execution
                            .matches
                            .iter()
                            .any(|matched| disallowed.contains(&matched.id))
                        {
                            return None;
                        }
                        for end_state in execution.end_state {
                            let future = constraint
                                .tokenizer
                                .possible_future_terminals(end_state);
                            for &terminal in disallowed.iter() {
                                if future.contains(terminal as usize) {
                                    remapped
                                        .entry(end_state)
                                        .or_insert_with(BTreeSet::new)
                                        .insert(terminal);
                                }
                            }
                        }
                    }
                    Some(TerminalsDisallowed::from_map(remapped))
                },
            )
        };

        let mut partitions = Vec::new();
        let root = gss.isolate(None);
        if !root.is_empty() {
            let root = prune_partition(root);
            if !root.is_empty() {
                partitions.push(root);
            }
        }
        for parser_state in gss.peek_values() {
            let partition = prune_partition(gss.isolate(Some(parser_state)));
            if !partition.is_empty() {
                partitions.push(partition);
            }
        }

        let mut iter = partitions.into_iter();
        let Some(mut merged) = iter.next() else {
            return ParserGSS::empty();
        };
        for partition in iter {
            merged = merged.merge(&partition);
        }
        merged
    }

    #[test]
    fn initial_prune_advances_each_continuation_tokenizer_state() {
        let vocab = Vocab::from_entries_unchecked(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
            ]);
        let grammar = r#"
start start;
t A = "a";
t B = "a" | "ab";
nt item = A | B;
nt start = item item? item?;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        assert!(state.commit_token(0));

        let tokenizer_state = constraint.runtime_commit_initial_state();
        // This test exercises the GSS pruning primitive itself, so cross the
        // bounded flat-state boundary explicitly before inspecting one map value.
        state.state.normalize_duplicate_keys();
        let gss = state
            .state
            .get(&tokenizer_state)
            .expect("ambiguous a must retain a reset lexer branch");
        assert!(
            !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()),
            "MRE requires a stale residual exclusion on the reset branch"
        );

        let exec_result =
            execute_tokenizer_from_state_small(&constraint, b"b", tokenizer_state);
        assert!(exec_result.matches.is_empty());
        assert!(exec_result.end_state.is_empty());

        let pruned = prune_single_initial_state_for_exec(
            &constraint,
            gss.clone(),
            tokenizer_state,
            &exec_result,
            b"b",
        );
        assert!(
            !pruned.is_empty()
                && pruned.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()),
            "the valid A=a branch must survive while the provisional B=a branch is invalidated by B=ab"
        );
    }

    #[test]
    fn delayed_exclusion_survives_across_model_token_boundaries() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"@".to_vec()),
            (1, b" double".to_vec()),
            (2, b"Quote".to_vec()),
            (3, b"x".to_vec()),
            (4, b"=".to_vec()),
            (5, b"%".to_vec()),
            (6, b"0".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                nt start = declaration expression;
                nt declaration = "@" ID;
                nt expression = ID OP "0";
                t WS = b/[ \t\r\n]/+;
                t OP = "=" | "%=";
                t ID = b/[A-Za-z_$]/ b/[A-Za-z0-9_$]/*;
            "#,
            &vocab,
        )
        .unwrap();

        let mut state = constraint.start();
        assert!(state.commit_token(0));
        assert!(state.commit_token(1));
        assert!(state.commit_token(2));

        let mask = state.mask();
        let allowed = |token: u32| {
            ((mask[token as usize / 32] >> (token % 32)) & 1) != 0
        };
        assert!(allowed(3), "continuing the current ID remains viable");
        assert!(!allowed(4), "the consumed ID cannot also satisfy the next ID");
        assert!(!allowed(5), "a prefix of %= cannot follow until a new ID is consumed");
        let mut probe = state.clone();
        assert!(!probe.commit_token(4));
        let mut probe = state.clone();
        assert!(!probe.commit_token(5));
    }

    #[test]
    fn recursive_delayed_exclusion_stays_in_leaf_tokenizer_terminal_coordinate() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"@".to_vec()),
            (1, b" double".to_vec()),
            (2, b"Quote".to_vec()),
            (3, b"x".to_vec()),
            (4, b"=".to_vec()),
            (5, b"%".to_vec()),
            (6, b"0".to_vec()),
        ]);
        let body = Constraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                nt start = declaration expression;
                nt declaration = "@" ID;
                nt expression = ID OP "0";
                t WS = b/[ \t\r\n]/+;
                t OP = "=" | "%=";
                t ID = b/[A-Za-z_$]/ b/[A-Za-z0-9_$]/*;
            "#,
            &vocab,
        )
        .unwrap();
        let parent = Constraint::compile(
            Grammar::glrm("glrm 1; start document; extern grammar body; nt document = body;"),
            &vocab,
        )
        .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("body", body.clone())
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        let loaded = Constraint::load(bound.save(), &vocab).unwrap();

        let mut expected = body.start();
        for token in [0, 1, 2] {
            assert!(expected.commit_token(token));
        }
        let expected_mask = expected.mask();
        for constraint in [&bound, &loaded] {
            let mut state = constraint.start();
            for token in [0, 1, 2] {
                assert!(state.commit_token(token));
            }
            assert_eq!(state.mask(), expected_mask);
            assert!(state.state.iter().any(|(_, gss)| {
                !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            }));
            let mut probe = state.clone();
            assert!(!probe.commit_token(4));
            let mut probe = state.clone();
            assert!(!probe.commit_token(5));
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~28s); run with --features slow-tests"
    )]
    fn initial_prune_matches_top_local_reference_on_generated_small_languages() {
        const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
        let vocab = Vocab::from_entries_unchecked(
            [
                "a", "b", "ab", "ba", " ", " a", "a ", " b", "b ", " a ", " b ",
            ]
            .into_iter()
            .enumerate()
            .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
            .collect());
        let languages = (1u32..1u32 << WORDS.len())
            .filter(|mask| mask.count_ones() <= 2)
            .collect::<Vec<_>>();
        let rule = |name: &str, mask: u32| {
            let rhs = WORDS
                .iter()
                .enumerate()
                .filter(|&(index, word)| mask & (1 << index) != 0).map(|(index, word)| format!("\"{word}\""))
                .collect::<Vec<_>>()
                .join(" | ");
            format!("t {name} = {rhs};\n")
        };

        for &a in &languages {
            for &b in &languages {
                for start_rule in [
                    "nt item = A | B;\nnt start = item item? item?;",
                    "nt start = A A | B B;",
                    "nt start = A B | B A;",
                ] {
                    let grammar = format!(
                        "start start;\nignore WS;\nt WS = \" \"+;\n{}{}{start_rule}\n",
                        rule("A", a),
                        rule("B", b),
                    );
                    let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                    let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
                    let mut seen = BTreeSet::new();

                    for depth in 0..=4 {
                        let mut next = Vec::new();
                        for (state, path) in frontier {
                            let state_key = canonical_commit_state(&state.state);
                            if !seen.insert(state_key) {
                                continue;
                            }
                            for (token_id, bytes) in constraint.token_bytes_iter() {
                                for (&tokenizer_state, gss) in &state.state {
                                    if gss
                                        .all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
                                    {
                                        continue;
                                    }
                                    let exec_result = execute_tokenizer_from_state_small(
                                        &constraint,
                                        bytes,
                                        tokenizer_state,
                                    );
                                    let actual = prune_single_initial_state_for_exec(
                                        &constraint,
                                        gss.clone(),
                                        tokenizer_state,
                                        &exec_result,
                                        bytes,
                                    );
                                    let expected = top_local_prune_reference(
                                        &constraint,
                                        gss,
                                        bytes,
                                    );
                                    assert_eq!(
                                        canonical_gss(&actual),
                                        canonical_gss(&expected),
                                        "initial prune crossed parser-top correlations: A={a:#06b} B={b:#06b} depth={depth} path={path:?} token={token_id} bytes={bytes:?} tokenizer_state={tokenizer_state}\ngrammar:\n{grammar}\nsource={:#?}\nactual={:#?}\nexpected={:#?}",
                                        canonical_gss(gss),
                                        canonical_gss(&actual),
                                        canonical_gss(&expected),
                                    );
                                }

                                if depth < 4 {
                                    let mut advanced = state.clone();
                                    if advanced.commit_bytes(bytes) {
                                        let mut next_path = path.clone();
                                        next_path.push(token_id);
                                        next.push((advanced, next_path));
                                    }
                                }
                            }
                        }
                        frontier = next;
                    }
                }
            }
        }
    }

    #[test]
    fn rejected_public_commits_enter_fail_state() {
        let vocab = Vocab::from_entries_unchecked(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a";
                nt start = A;
            "#,
            &vocab,
        )
        .unwrap();

        let assert_failed = |state: &ConstraintState| {
            assert!(state.state.is_empty());
            assert!(state.mask().iter().all(|&word| word == 0));
        };

        let mut state = constraint.start();
        assert!(!state.commit_token(1));
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_timed_ns(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_profiled(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_per_advance(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(!state.commit_bytes(b"b"));
        assert_failed(&state);
    }

    fn assert_fast_and_general_queue_match(
        constraint: &Constraint,
        fast_state: &ConstraintState,
        token_id: u32,
        bytes: &[u8],
        context: &str,
    ) -> Option<ConstraintState> {
        let mut fast = fast_state.clone();
        let mut profiled = fast_state.clone();
        let mut general = fast_state.clone();

        let fast_result = commit_bytes_impl(
            constraint,
            &mut fast.state,
            bytes,
            &mut fast.buffers,
        );
        let profiled_result = commit_bytes_impl_profiled(
            constraint,
            &mut profiled.state,
            bytes,
            &mut profiled.buffers,
            None,
            true,
        );
        let general_result = commit_bytes_impl_profiled(
            constraint,
            &mut general.state,
            bytes,
            &mut general.buffers,
            None,
            false,
        );

        assert_eq!(
            fast_result.is_ok(),
            general_result.is_ok(),
            "commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast={:?}\ngeneral={:?}",
            fast.state,
            general.state,
        );
        assert_eq!(
            profiled_result.is_ok(),
            general_result.is_ok(),
            "profiled commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled={:?}\ngeneral={:?}",
            profiled.state,
            general.state,
        );
        if fast_result.is_err() {
            return None;
        }
        assert_eq!(
            canonical_commit_state(&fast.state),
            canonical_commit_state(&general.state),
            "successful commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast_stacks={:#?}\ngeneral_stacks={:#?}",
            fast.state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
            general
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            canonical_commit_state(&profiled.state),
            canonical_commit_state(&general.state),
            "successful profiled commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled_stacks={:#?}\ngeneral_stacks={:#?}",
            profiled
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
            general
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit")))
                .collect::<Vec<_>>(),
        );

        Some(fast)
    }

    #[test]
    fn flat_frontier_accepts_input_past_the_inline_state_capacity() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec())]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a";
                nt start = A;
            "#,
            &vocab,
        )
        .expect("single-terminal grammar should compile");
        let start = constraint.start();
        let (tokenizer_state, parser_gss) = start.state.entries[0].clone();
        let mut state = ParserStateMap::default();
        for _ in 0..=INLINE_PARSER_STATE_CAPACITY {
            state.insert_flat_alternative(tokenizer_state, parser_gss.clone());
        }
        assert_eq!(state.len(), INLINE_PARSER_STATE_CAPACITY + 1);

        let mut actual = start.clone();
        actual.state = state;
        assert!(actual.commit_bytes(b"a"));
        let mut expected = start;
        assert!(expected.commit_bytes(b"a"));
        assert_eq!(canonical_commit_state(&actual.state), canonical_commit_state(&expected.state));
        assert_eq!(actual.mask(), expected.mask());
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
    }

    #[test]
    fn flat_frontier_preserves_all_stacks_in_a_live_lexer_continuation_group() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"ba".to_vec()),
        ]);
        let grammar = r#"
            start start;
            t A = "a";
            t B = "b" | "ab";
            nt item = A | B;
            nt start = item item? item?;
        "#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let start = constraint.start();
        let after_ab = assert_fast_and_general_queue_match(
            &constraint,
            &start,
            2,
            b"ab",
            "grouped lexer continuation regression after ab",
        )
        .unwrap();
        let after_ba = assert_fast_and_general_queue_match(
            &constraint,
            &after_ab,
            3,
            b"ba",
            "grouped lexer continuation regression after ab, ba",
        )
        .unwrap();

        let stacks = canonical_commit_state(&after_ba.state);
        let tokenizer_one = stacks
            .iter()
            .find(|(tokenizer_state, _)| *tokenizer_state == 1)
            .expect("lexer continuation state must remain live");
        assert_eq!(
            tokenizer_one.1,
            vec![
                (vec![0, 3, 2], Vec::new()),
                (vec![0, 3, 5, 2], Vec::new()),
            ],
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~11s); run with --features slow-tests"
    )]
    fn monolithic_commit_fast_paths_match_general_queue_on_small_language_space() {
        const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
        let vocab = Vocab::from_entries_unchecked(
            WORDS
                .iter()
                .enumerate()
                .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
                .collect());
        let languages = (1u32..1u32 << WORDS.len())
            .filter(|mask| mask.count_ones() <= 2)
            .collect::<Vec<_>>();

        let rule = |name: &str, mask: u32| {
            let rhs = WORDS
                .iter()
                .enumerate()
                .filter(|&(index, word)| mask & (1 << index) != 0).map(|(index, word)| format!("\"{word}\""))
                .collect::<Vec<_>>()
                .join(" | ");
            format!("t {name} = {rhs};\n")
        };

        for &a in &languages {
            for &b in &languages {
                let grammar = format!(
                    "start start;\n{}{}nt item = A | B;\nnt start = item item? item?;\n",
                    rule("A", a),
                    rule("B", b),
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];

                for depth in 0..3 {
                    let mut next = Vec::new();
                    for (state, path) in frontier {
                        let mask = state.mask();
                        for (&token_id, bytes) in vocab.entries_map().iter() {
                            let context = format!(
                                "A_mask={a:#06b} B_mask={b:#06b} depth={depth} path={path:?}\ngrammar:\n{grammar}"
                            );
                            let next_state = assert_fast_and_general_queue_match(
                                &constraint,
                                &state,
                                token_id,
                                bytes,
                                &context,
                            );
                            let token_in_mask = mask
                                .get(token_id as usize / 32)
                                .is_some_and(|word| {
                                    word & (1u32 << (token_id % 32)) != 0
                                });
                            assert_eq!(
                                token_in_mask,
                                next_state.is_some(),
                                "mask/commit mismatch: {context} token_id={token_id} bytes={bytes:?}"
                            );
                            if let Some(next_state) = next_state {
                                let mut next_path = path.clone();
                                next_path.push(token_id);
                                next.push((next_state, next_path));
                            }
                        }
                    }
                    frontier = next;
                }
            }
        }
    }

    #[test]
    fn residual_bc_fast_path_matches_general_queue() {
        let vocab = Vocab::from_entries_unchecked(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"ba".to_vec()),
                (5, b"bc".to_vec()),
                (6, b"abc".to_vec()),
            ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A = "a" | "ab";
                t B = "bc";
                nt item = A | B;
                nt start = item item? item?;
            "#,
            &vocab,
        )
        .unwrap();

        let mut fast = constraint.start();
        let mut slow = constraint.start();
        let fast_result = commit_bytes_impl(
            &constraint,
            &mut fast.state,
            vocab.entries_map().get(&0).unwrap(),
            &mut fast.buffers,
        );
        fast.generation += 1;
        assert!(fast_result.is_ok());
        let slow_result = commit_bytes_impl_profiled(
            &constraint,
            &mut slow.state,
            vocab.entries_map().get(&0).unwrap(),
            &mut slow.buffers,
            None,
            false,
        );
        slow.generation += 1;
        assert!(slow_result.is_ok());
        assert_eq!(fast.state, slow.state, "state mismatch after token a");

        let mut next_fast = fast.clone();
        let mut next_slow = slow.clone();
        let fast_result = commit_bytes_impl(
            &constraint,
            &mut next_fast.state,
            vocab.entries_map().get(&5).unwrap(),
            &mut next_fast.buffers,
        );
        let slow_result = commit_bytes_impl_profiled(
            &constraint,
            &mut next_slow.state,
            vocab.entries_map().get(&5).unwrap(),
            &mut next_slow.buffers,
            None,
            false,
        );
        assert_eq!(
            fast_result.is_ok(),
            slow_result.is_ok(),
            "fast={:?}\nslow={:?}",
            next_fast.state,
            next_slow.state,
        );
        if fast_result.is_ok() {
            assert_eq!(next_fast.state, next_slow.state);
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~15s); run with --features slow-tests"
    )]
    fn epsilon_commit_fast_paths_match_no_fast_path_reference() {
        let vocab = Vocab::from_entries_unchecked(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"aa".to_vec()),
                (4, b"ab".to_vec()),
                (5, b" ".to_vec()),
                (6, b" a".to_vec()),
                (7, b"a ".to_vec()),
                (8, b" a ".to_vec()),
                (9, b"abc".to_vec()),
                (10, b"aab".to_vec()),
            ]);
        let grammar = crate::grammar::glrm::from_glrm(
            r#"
                start start;
                ignore WS;
                pragma glrmask {
                pragma glrmask {
                    lexer group ws = WS;
                    lexer group a = A;
                    lexer group b = B;
                    lexer group c = C;
                }
                }
                t WS = " "+;
                t A = "a"+;
                t B = "b";
                t C = "c";
                nt item = A | B | C;
                nt start = item item? item?;
            "#,
        )
        .unwrap();
        let grammar = crate::grammar::ast::lower(&grammar).unwrap();
        let constraint = crate::compiler::pipeline::compile_owned_partitioned_for_test(
            grammar,
            &vocab,
        );
        assert!(constraint.tokenizer_has_epsilon_transitions);

        let mut frontier = vec![(
            constraint.start(),
            constraint.start(),
            constraint.start(),
            Vec::<u32>::new(),
        )];
        for depth in 0..=4 {
            let mut next = Vec::new();
            for (fast, profiled, general, path) in frontier {
                assert_eq!(
                    fast.mask(),
                    general.mask(),
                    "epsilon mask mismatch after path {path:?}\nfast={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&fast.state),
                    canonical_commit_state(&general.state),
                );
                assert_eq!(
                    profiled.mask(),
                    general.mask(),
                    "epsilon profiled-mask mismatch after path {path:?}\nprofiled={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&profiled.state),
                    canonical_commit_state(&general.state),
                );
                assert_eq!(
                    fast.is_accepting(),
                    general.is_accepting(),
                    "epsilon completion mismatch after path {path:?}",
                );
                assert_eq!(
                    profiled.is_accepting(),
                    general.is_accepting(),
                    "epsilon profiled completion mismatch after path {path:?}",
                );
                if depth == 4 {
                    continue;
                }

                for (&token_id, bytes) in vocab.entries_map().iter() {
                    let mut next_fast = fast.clone();
                    let mut next_profiled = profiled.clone();
                    let mut next_general = general.clone();
                    let fast_result = commit_bytes_impl(
                        &constraint,
                        &mut next_fast.state,
                        bytes,
                        &mut next_fast.buffers,
                    );
                    let profiled_result = commit_bytes_impl_profiled(
                        &constraint,
                        &mut next_profiled.state,
                        bytes,
                        &mut next_profiled.buffers,
                        None,
                        true,
                    );
                    let general_result = commit_bytes_impl_profiled(
                        &constraint,
                        &mut next_general.state,
                        bytes,
                        &mut next_general.buffers,
                        None,
                        false,
                    );
                    assert_eq!(
                        fast_result.is_ok(),
                        general_result.is_ok(),
                        "epsilon commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nfast={:#?}\ngeneral={:#?}",
                        canonical_commit_state(&next_fast.state),
                        canonical_commit_state(&next_general.state),
                    );
                    assert_eq!(
                        profiled_result.is_ok(),
                        general_result.is_ok(),
                        "epsilon profiled commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nprofiled={:#?}\ngeneral={:#?}",
                        canonical_commit_state(&next_profiled.state),
                        canonical_commit_state(&next_general.state),
                    );
                    if fast_result.is_ok() {
                        let mut next_path = path.clone();
                        next_path.push(token_id);
                        next.push((next_fast, next_profiled, next_general, next_path));
                    }
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn epsilon_full_width_terminal_with_empty_accumulators_profiles_advance() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let grammar = crate::grammar::glrm::from_glrm(
            r#"
                start start;
                pragma glrmask {
                    lexer group left = A;
                    lexer group right = B;
                }
                t A = "a";
                t B = "b";
                nt start = A | B;
            "#,
        )
        .unwrap();
        let grammar = crate::grammar::ast::lower(&grammar).unwrap();
        let constraint = crate::compiler::pipeline::compile_owned_partitioned_for_test(
            grammar,
            &vocab,
        );

        let mut state = constraint.start();
        let profile = state.commit_token_profiled(0).unwrap();
        assert!(profile.n_queue_entries > 0, "profile={profile:?}");
        assert_eq!(profile.n_advances, 1, "profile={profile:?}");
    }
}

#[cfg(test)]
mod commit_bucket_tests {
    use super::*;

    #[test]
    fn bucket_promotion_preserves_correlated_parser_languages() {
        let mut bucket = CommitBucket::default();
        for key in 0..12 {
            bucket.merge(key, ParserGSS::from_single_stack(vec![key, 100], TerminalsDisallowed::new()));
            bucket.merge(key, ParserGSS::from_single_stack(vec![key, 200], TerminalsDisallowed::new()));
        }
        assert!(matches!(bucket, CommitBucket::Hashed(_)));
        let rows = bucket.take_entries();
        assert_eq!(rows.len(), 12);
        for (key, gss) in rows {
            let mut stacks = parser_stacks_only(&gss);
            stacks.sort();
            assert_eq!(stacks, vec![vec![key, 100], vec![key, 200]]);
        }
        assert!(bucket.take_entries().is_empty());
    }

    #[test]
    fn read_only_buffers_leave_worklist_unallocated() {
        let mut buffers = CommitBuffers::for_mask_only_shadow();
        assert_eq!(buffers.processing_queue.capacity(), 0);
        buffers.clear_all();
        buffers.reset_all();
        assert_eq!(buffers.processing_queue.capacity(), 0);
        assert!(std::mem::size_of::<CommitBuffers>() < 8 * 1024);
    }
}


