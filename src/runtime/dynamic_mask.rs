//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{
    TokenizerExecResult, TokenizerMatch, TokenizerStateSet,
};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    advance_stacks, stack_may_advance_on, stack_may_advance_on_any, ParserGSS,
};
use crate::ds::leveled_gss::LeveledGSS;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

use super::artifact::{Constraint, DynamicMaskStateKey, DynamicMaskTrie};
use super::state::ConstraintState;

type ParserStacks = LeveledGSS<u32, ()>;

#[derive(Default)]
struct DynamicTraversalCache {
    boundary: FxHashMap<(u32, usize), (ParserStacks, bool)>,
    lexer_relevant: FxHashMap<(u32, usize), (ParserStacks, bool)>,
    parser_children: FxHashMap<(usize, TerminalID), (ParserStacks, Option<ParserStacks>)>,
}

#[derive(Clone)]
struct TraverseWork {
    trie_index: usize,
    node: u32,
    tokenizer_state: u32,
    gss: ParserStacks,
    initial_prune_guard: InitialPruneGuard,
    continuation_filter: Option<ContinuationFilter>,
}

#[derive(Clone, Copy)]
enum ContinuationFilter {
    Narrow {
        partition_index: usize,
        required_groups: u64,
    },
    AlreadyMarked,
}

#[derive(Clone)]
enum InitialPruneGuard {
    Passed,
    Pending {
        blocked: Arc<BTreeSet<TerminalID>>,
        lexer_states: TokenizerStateSet,
        actionable_states: Arc<[u32]>,
        has_actionable_match: bool,
    },
}

#[inline]
fn set_mask_bit(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    if let Some(slot) = buf.get_mut(word) {
        *slot |= 1u32 << bit;
    }
}

#[inline]
fn or_mask(buf: &mut [u32], mask: &[u32]) {
    for (target, source) in buf.iter_mut().zip(mask) {
        *target |= *source;
    }
}

const DYNAMIC_NFA_CONFIG_UNKNOWN: u32 = u32::MAX;
const DYNAMIC_NFA_CONFIG_DEAD: u32 = u32::MAX - 1;

struct DynamicNfaScanCache<'a> {
    constraint: &'a Constraint,
    deadline: Option<Instant>,
    max_collection_items: Option<usize>,
    epsilon_closures: Arc<[Box<[u32]>]>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    raw_start_config: Vec<u32>,
}

impl<'a> DynamicNfaScanCache<'a> {
    fn new(constraint: &'a Constraint, deadline: Option<Instant>) -> Self {
        Self {
            constraint,
            deadline,
            max_collection_items: deadline.map(|_| 5_000_000),
            epsilon_closures: constraint.tokenizer.all_singleton_epsilon_closures(),
            config_ids: FxHashMap::default(),
            configs: Vec::new(),
            transitions: Vec::new(),
            raw_start_config: vec![
                DYNAMIC_NFA_CONFIG_UNKNOWN;
                constraint.tokenizer.num_states() as usize
            ],
        }
    }

    fn check_growth(&self, current: usize, additional: usize) -> Result<(), String> {
        if self.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err("glrmask_dynamic mask generation timed out".to_owned());
        }
        if self.max_collection_items.is_some_and(|limit| {
            current
                .checked_add(additional)
                .is_none_or(|next| next > limit)
        }) {
            return Err("glrmask_dynamic mask generation exceeded its work ceiling".to_owned());
        }
        Ok(())
    }

    fn intern_config(&mut self, mut states: Vec<u32>) -> Result<u32, String> {
        self.check_growth(0, states.len())?;
        states.sort_unstable();
        states.dedup();
        if let Some(&id) = self.config_ids.get(states.as_slice()) {
            return Ok(id);
        }
        self.check_growth(self.configs.len(), 1)?;
        let id = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.transitions.push(None);
        Ok(id)
    }

    fn config_for_raw_start(&mut self, state: u32) -> Result<u32, String> {
        let slot = state as usize;
        let cached = self.raw_start_config[slot];
        if cached != DYNAMIC_NFA_CONFIG_UNKNOWN {
            return Ok(cached);
        }
        let config = self.intern_config(self.epsilon_closures[slot].to_vec())?;
        self.raw_start_config[slot] = config;
        Ok(config)
    }

    fn step_config(&mut self, config: u32, byte: u8) -> Result<Option<u32>, String> {
        let config_index = config as usize;
        if let Some(row) = self.transitions[config_index].as_ref() {
            let cached = row[byte as usize];
            if cached != DYNAMIC_NFA_CONFIG_UNKNOWN {
                return Ok((cached != DYNAMIC_NFA_CONFIG_DEAD).then_some(cached));
            }
        }

        let closed_targets = {
            let mut targets = Vec::<u32>::new();
            for &state in self.configs[config_index].iter() {
                let target = self.constraint.tokenizer_fast_transitions[state as usize]
                    [byte as usize];
                if target != u32::MAX {
                    self.check_growth(targets.len(), self.epsilon_closures[target as usize].len())?;
                    targets.extend_from_slice(&self.epsilon_closures[target as usize]);
                }
            }
            targets
        };
        let target = if closed_targets.is_empty() {
            DYNAMIC_NFA_CONFIG_DEAD
        } else {
            self.intern_config(closed_targets)?
        };
        let row = self.transitions[config_index]
            .get_or_insert_with(|| Box::new([DYNAMIC_NFA_CONFIG_UNKNOWN; 256]));
        row[byte as usize] = target;
        Ok((target != DYNAMIC_NFA_CONFIG_DEAD).then_some(target))
    }

    fn execute_from_state_all_widths(
        &mut self,
        input: &[u8],
        start: u32,
    ) -> Result<TokenizerExecResult, String> {
        let mut config = self.config_for_raw_start(start)?;
        let mut matches = Vec::new();
        for (index, &byte) in input.iter().enumerate() {
            let Some(next_config) = self.step_config(config, byte)? else {
                return Ok(TokenizerExecResult {
                    end_state: TokenizerStateSet::new(),
                    matches,
                });
            };
            config = next_config;
            let width = index + 1;
            for &state in self.configs[config as usize].iter() {
                for id in self.constraint.tokenizer.matched_terminals_iter(state) {
                    self.check_growth(matches.len(), 1)?;
                    matches.push(TokenizerMatch {
                        id,
                        width,
                        end_state: state,
                    });
                }
            }
        }
        let mut end_state = TokenizerStateSet::new();
        end_state.extend(
            self.configs[config as usize]
                .iter()
                .copied()
                .filter(|&state| !self.constraint.tokenizer.is_end(state)),
        );
        Ok(TokenizerExecResult { end_state, matches })
    }
}

#[inline]
fn mask_bit_is_set(buf: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    buf.get(word)
        .is_some_and(|slot| *slot & (1u32 << bit) != 0)
}

fn canonical_token_is_marked(
    vocab: &super::artifact::DynamicMaskVocab,
    canonical_token_id: u32,
    buf: &[u32],
) -> bool {
    vocab
        .token_ids(canonical_token_id)
        .is_some_and(|token_ids| token_ids.iter().all(|&token_id| mask_bit_is_set(buf, token_id)))
}

fn subtree_is_fully_marked(
    vocab: &super::artifact::DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    node: u32,
    buf: &[u32],
) -> bool {
    trie.subtree_tokens(node)
        .iter()
        .all(|&token_id| canonical_token_is_marked(vocab, token_id, buf))
}

fn should_lazy_build_continuation_partition(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    traversal_cache: &mut DynamicTraversalCache,
) -> bool {
    let tokenizer = &constraint.tokenizer;
    if tokenizer_state == tokenizer.initial_state()
        || tokenizer.transitions_from(tokenizer_state).count() < 128
        || tokenizer.self_loop_bytes(tokenizer_state).len() >= 24
    {
        return false;
    }

    // The expensive residual shape is only worth partitioning when a broad
    // immediate successor is already admissible for the current parser stack.
    // This cheaply rejects lookalike lexer states whose partition would prove
    // only a handful of tokens.
    let mut target_widths = FxHashMap::<u32, usize>::default();
    for (_, target) in tokenizer.transitions_from(tokenizer_state) {
        *target_widths.entry(target).or_default() += 1;
    }
    target_widths.into_iter().any(|(target, width)| {
        width >= 24
            && token_boundary_allowed_cached(constraint, target, stacks, traversal_cache)
    })
}

fn update_eos_mask(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let Some(token_id) = state.constraint.eos_token_id else {
        return;
    };
    if state.constraint.has_special_token_id(token_id) {
        return;
    }
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    let Some(slot) = buf.get_mut(word) else {
        return;
    };
    *slot &= !(1u32 << bit);
    if state.is_complete() {
        *slot |= 1u32 << bit;
    }
}

fn update_special_token_mask(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let mut previous_token_id = None;
    for special in &state.constraint.special_token_terminals {
        if previous_token_id == Some(special.token_id) {
            continue;
        }
        previous_token_id = Some(special.token_id);
        if super::commit::advance_special_token_paths(
            state.constraint,
            &state.state,
            special.token_id,
        )
        .is_some_and(|gss| !gss.is_empty())
        {
            set_mask_bit(buf, special.token_id);
        }
    }
}

/// Dynamic masking keeps terminal restrictions outside the parser GSS. The
/// parser table routines still use `ParserGSS`, so give their stack operations
/// an otherwise-unused empty accumulator.
fn with_empty_accumulators(stacks: &ParserStacks) -> ParserGSS {
    stacks.apply(|_| TerminalsDisallowed::new())
}

fn terminal_is_actionable_from_states(
    constraint: &Constraint,
    parser_states: &[u32],
    terminal: TerminalID,
) -> bool {
    parser_states
        .iter()
        .any(|&parser_state| constraint.table.advance_row_allows(parser_state, terminal))
}

impl InitialPruneGuard {
    /// Build the token-start pruning state for one correlated tokenizer/GSS
    /// branch. This is the incremental form of
    /// `prune_single_initial_state_for_exec`: only restrictions attached to the
    /// active tokenizer state participate, and parser actionability is frozen at
    /// the token boundary before any in-token parser advance.
    fn new(
        constraint: &Constraint,
        tokenizer_state: u32,
        stacks: &ParserStacks,
        terminals_disallowed: &TerminalsDisallowed,
    ) -> Self {
        let Some(blocked) = terminals_disallowed.get(&tokenizer_state) else {
            return Self::Passed;
        };
        if blocked.is_empty() {
            return Self::Passed;
        }

        let actionable_states: Vec<u32> = if let Some(parser_state) = stacks.single_top_value() {
            vec![parser_state]
        } else {
            stacks.peek_values().into_vec()
        };
        if !blocked.iter().any(|&terminal| {
            terminal_is_actionable_from_states(constraint, &actionable_states, terminal)
        }) {
            return Self::Passed;
        }

        Self::Pending {
            blocked: Arc::new(blocked.clone()),
            lexer_states: smallvec::smallvec![tokenizer_state],
            actionable_states: actionable_states.into(),
            has_actionable_match: false,
        }
    }

    #[inline]
    fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// At a vocabulary-token leaf, commit keeps the seed branch if it saw no
    /// actionable terminal at all, or if any actionable match was unblocked.
    /// `Pending` can only represent the first case or the all-blocked case;
    /// unblocked matches transition permanently to `Passed`.
    #[inline]
    fn allows_token_boundary(&self) -> bool {
        match self {
            Self::Passed => true,
            Self::Pending {
                has_actionable_match,
                ..
            } => !*has_actionable_match,
        }
    }

    /// Advance the original token-start lexer branch through a trie segment.
    /// Parser resets caused by terminal matches elsewhere in the dynamic walk
    /// deliberately do not affect this guard: commit evaluates its initial
    /// pruning predicate once, over the whole candidate token, before advancing
    /// the parser.
    fn advance(&self, constraint: &Constraint, segment: &[u8]) -> Option<Self> {
        let Self::Pending {
            blocked,
            lexer_states,
            actionable_states,
            has_actionable_match,
        } = self
        else {
            return Some(Self::Passed);
        };

        let mut next_states = TokenizerStateSet::new();
        let mut saw_actionable = *has_actionable_match;
        for &tokenizer_state in lexer_states {
            let execution = constraint
                .tokenizer
                .execute_from_state_all_widths(segment, tokenizer_state);
            for matched in &execution.matches {
                if Some(matched.id) == constraint.ignore_terminal
                    || !terminal_is_actionable_from_states(
                        constraint,
                        actionable_states,
                        matched.id,
                    )
                {
                    continue;
                }
                saw_actionable = true;
                if !blocked.contains(&matched.id) {
                    return Some(Self::Passed);
                }
            }
            for end_state in execution.end_state {
                if !next_states.contains(&end_state) {
                    next_states.push(end_state);
                }
            }
        }

        if next_states.is_empty() {
            return (!saw_actionable).then_some(Self::Passed);
        }

        Some(Self::Pending {
            blocked: Arc::clone(blocked),
            lexer_states: next_states,
            actionable_states: Arc::clone(actionable_states),
            has_actionable_match: saw_actionable,
        })
    }
}

fn parser_child(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
) -> Option<ParserStacks> {
    // Ignore terminals reset the lexer but deliberately leave the parser alone.
    if Some(terminal) == constraint.ignore_terminal {
        return Some(stacks.clone());
    }
    let parser_gss = with_empty_accumulators(stacks);
    if !stack_may_advance_on(&constraint.table, &parser_gss, terminal) {
        return None;
    }
    let advanced = advance_stacks(&constraint.table, &parser_gss, terminal).apply(|_| ());
    (!advanced.is_empty()).then_some(advanced)
}

fn parser_child_cached(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
    cache: &mut DynamicTraversalCache,
) -> Option<ParserStacks> {
    let key = (stacks.ptr_key(), terminal);
    if let Some((cached_stacks, result)) = cache.parser_children.get(&key) {
        debug_assert!(cached_stacks.ptr_eq(stacks));
        return result.clone();
    }
    let result = parser_child(constraint, stacks, terminal);
    cache
        .parser_children
        .insert(key, (stacks.clone(), result.clone()));
    result
}

fn token_boundary_allowed(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
) -> bool {
    let parser_gss = with_empty_accumulators(stacks);
    constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state)
        .iter()
        .any(|terminal| {
            let terminal = terminal as TerminalID;
            Some(terminal) == constraint.ignore_terminal
                || stack_may_advance_on(&constraint.table, &parser_gss, terminal)
        })
}

fn token_boundary_allowed_cached(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    let key = (tokenizer_state, stacks.ptr_key());
    if let Some((cached_stacks, result)) = cache.boundary.get(&key) {
        debug_assert!(cached_stacks.ptr_eq(stacks));
        return *result;
    }
    let result = token_boundary_allowed(constraint, tokenizer_state, stacks);
    cache
        .boundary
        .insert(key, (stacks.clone(), result));
    result
}

fn lexer_state_relevant_cached(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    let key = (tokenizer_state, stacks.ptr_key());
    if let Some((cached_stacks, result)) = cache.lexer_relevant.get(&key) {
        debug_assert!(cached_stacks.ptr_eq(stacks));
        return *result;
    }

    let accessible = constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state);
    let matched = constraint.tokenizer.matched_terminal_bitset(tokenizer_state);
    let ignore_relevant = constraint.ignore_terminal.is_some_and(|terminal| {
        accessible.contains(terminal as usize) || matched.contains(terminal as usize)
    });
    let result = if ignore_relevant {
        true
    } else {
        let parser_gss = with_empty_accumulators(stacks);
        stack_may_advance_on_any(&constraint.table, &parser_gss, accessible)
            || stack_may_advance_on_any(&constraint.table, &parser_gss, matched)
    };
    cache
        .lexer_relevant
        .insert(key, (stacks.clone(), result));
    result
}

#[inline]
fn mark_subtree_tokens(
    constraint: &Constraint,
    trie: &DynamicMaskTrie,
    node: u32,
    buf: &mut [u32],
) {
    for &canonical_token_id in trie.subtree_tokens(node) {
        let token_ids = constraint
            .dynamic_mask_vocab
            .token_ids(canonical_token_id)
            .expect("dynamic vocabulary trie node lacks token ids");
        for &token_id in token_ids {
            set_mask_bit(buf, token_id);
        }
    }
}

enum RawSelfLoopSubtree {
    CannotSkip,
    MarkAllTokens,
}

#[inline]
fn cached_self_loop_bytes(
    constraint: &Constraint,
    tokenizer_state: u32,
    cache: &mut FxHashMap<u32, U8Set>,
) -> U8Set {
    *cache
        .entry(tokenizer_state)
        .or_insert_with(|| constraint.tokenizer.self_loop_bytes(tokenizer_state))
}

/// A raw tokenizer-state self-loop is a particularly strong residual-language
/// certificate: consuming the byte leaves every lexer possibility exactly
/// unchanged. If every remaining byte below a vocabulary-trie node has that
/// property, the no-finalization continuation witnesses every token in the
/// subtree without any per-token lexer or parser work.
///
/// A pending token-start prune guard cannot use the shortcut because a later
/// byte may still supply the unblocked actionable match that rescues the whole
/// candidate token.
fn raw_self_loop_subtree(
    constraint: &Constraint,
    trie: &DynamicMaskTrie,
    node: u32,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    initial_prune_guard: &InitialPruneGuard,
    initial_tsid: u32,
    self_loop_cache: &mut FxHashMap<u32, U8Set>,
    traversal_cache: &mut DynamicTraversalCache,
) -> RawSelfLoopSubtree {
    if !initial_prune_guard.is_passed() {
        return RawSelfLoopSubtree::CannotSkip;
    }

    // Work at the initial state may represent either an untouched lexer or a
    // lexer reset after an in-token terminal match. The current work item does
    // not distinguish those cases, so keep this optimization conservative.
    if tokenizer_state == initial_tsid {
        return RawSelfLoopSubtree::CannotSkip;
    }

    let subtree_bytes = U8Set::from_words(trie.subtree_bytes(node));
    let loop_bytes = cached_self_loop_bytes(constraint, tokenizer_state, self_loop_cache);
    if !subtree_bytes.is_subset(&loop_bytes)
        || !token_boundary_allowed_cached(constraint, tokenizer_state, stacks, traversal_cache)
    {
        return RawSelfLoopSubtree::CannotSkip;
    }

    RawSelfLoopSubtree::MarkAllTokens
}

const DYNAMIC_MASK_CACHE_MAX_STACKS: usize = 4_096;
const DYNAMIC_MASK_CACHE_MAX_DEPTH: u32 = 256;

fn dynamic_mask_state_key(state: &ConstraintState<'_>) -> Option<DynamicMaskStateKey> {
    let mut remaining = DYNAMIC_MASK_CACHE_MAX_STACKS;
    let mut key = Vec::with_capacity(state.state.len());
    for (&tokenizer_state, gss) in &state.state {
        if gss.max_depth() > DYNAMIC_MASK_CACHE_MAX_DEPTH {
            return None;
        }
        let stacks = gss.to_stacks(remaining)?;
        remaining = remaining.checked_sub(stacks.len())?;
        let mut paths = stacks
            .into_iter()
            .map(|(stack, exclusions)| {
                let exclusion_entries = exclusions
                    .0
                    .iter()
                    .map(|(&excluded_state, terminals)| {
                        (excluded_state, terminals.iter().copied().collect::<Vec<_>>())
                    })
                    .collect::<Vec<_>>();
                (stack, exclusion_entries)
            })
            .collect::<Vec<_>>();
        paths.sort_unstable();
        key.push((tokenizer_state, paths));
    }
    Some(key)
}

pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    fill_mask_dynamic_impl(state, buf, None)
        .expect("unbounded dynamic mask generation cannot time out");
}

pub(crate) fn fill_mask_dynamic_bounded(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    timeout_ms: u64,
) -> Result<(), String> {
    fill_mask_dynamic_impl(
        state,
        buf,
        Some(Instant::now() + Duration::from_millis(timeout_ms)),
    )
}

fn fill_mask_dynamic_impl(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    deadline: Option<Instant>,
) -> Result<(), String> {
    let check_deadline = || {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            Err("glrmask_dynamic mask generation timed out".to_owned())
        } else {
            Ok(())
        }
    };
    let vocab = &state.constraint.dynamic_mask_vocab;
    let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
    let total_started_at = profile.then(std::time::Instant::now);
    let key_started_at = profile.then(std::time::Instant::now);
    let cache_key = dynamic_mask_state_key(state);
    let key_ms = key_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if cache_key
        .as_ref()
        .is_some_and(|cache_key| vocab.copy_cached_mask(cache_key, buf))
    {
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][dynamic_mask] generation={} cache_hit=true key_ms={:.3} total_ms={:.3}",
                state.generation,
                key_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(());
    }

    buf.fill(0);
    let initial_tsid = state.constraint.tokenizer.initial_state();
    let mut traversal = Vec::<TraverseWork>::with_capacity(4096);
    let mut segment_stack = Vec::<(usize, u32, ParserStacks)>::with_capacity(8);
    let mut self_loop_cache = FxHashMap::<u32, U8Set>::default();
    let mut traversal_cache = DynamicTraversalCache::default();
    let mut lexer_scan_cache = DynamicNfaScanCache::new(state.constraint, deadline);
    let tries = [vocab.trie.clone()];
    let mut continuation_partitions = Vec::new();
    let mut work_items = 0usize;
    let mut trie_edges = 0usize;
    let mut lexer_executions = 0usize;
    let mut subtree_marks = 0usize;
    let mut subtree_mark_tokens = 0usize;
    let mut continuation_groups_admitted = 0usize;
    let mut continuation_groups_traversed = 0usize;
    let mut lazy_continuation_builds_remaining = 1usize;
    if profile {
        eprintln!(
            "[glrmask/profile][dynamic_mask_config] tokenizer_states={} epsilon={} fast_transition_rows={}",
            state.constraint.tokenizer.num_states(),
            state.constraint.tokenizer.has_epsilon_transitions(),
            state.constraint.tokenizer_fast_transitions.len(),
        );
    }

    for (&tokenizer_state, gss) in &state.state {
        check_deadline()?;
        for (stacks, terminals_disallowed) in gss.partition_by_accumulator() {
            check_deadline()?;
            let initial_prune_guard = InitialPruneGuard::new(
                state.constraint,
                tokenizer_state,
                &stacks,
                &terminals_disallowed,
            );
            if profile {
                let loop_bytes = cached_self_loop_bytes(
                    state.constraint,
                    tokenizer_state,
                    &mut self_loop_cache,
                );
                eprintln!(
                    "[glrmask/profile][dynamic_seed] generation={} tokenizer_state={} initial={} stack_paths={} exclusions={} transitions={} matched={} futures={} loop_bytes={} boundary_allowed={}",
                    state.generation,
                    tokenizer_state,
                    tokenizer_state == initial_tsid,
                    stacks.path_count_at_most(1_000_000),
                    terminals_disallowed
                        .0
                        .values()
                        .map(BTreeSet::len)
                        .sum::<usize>(),
                    state.constraint.tokenizer.transitions_from(tokenizer_state).count(),
                    state
                        .constraint
                        .tokenizer
                        .matched_terminals_iter(tokenizer_state)
                        .count(),
                    state
                        .constraint
                        .tokenizer
                        .possible_future_terminals_iter(tokenizer_state)
                        .count(),
                    loop_bytes.len(),
                    token_boundary_allowed_cached(
                        state.constraint,
                        tokenizer_state,
                        &stacks,
                        &mut traversal_cache,
                    ),
                );
            }
            if initial_prune_guard.is_passed() {
                let mut partition = vocab.cached_continuation_partition(tokenizer_state);
                if partition.is_none()
                    && deadline.is_none()
                    && lazy_continuation_builds_remaining != 0
                    && should_lazy_build_continuation_partition(
                        state.constraint,
                        tokenizer_state,
                        &stacks,
                        &mut traversal_cache,
                    )
                {
                    lazy_continuation_builds_remaining -= 1;
                    partition = vocab.cached_or_build_continuation_partition(
                        &state.constraint.tokenizer,
                        tokenizer_state,
                        buf.len(),
                    );
                }
                if let Some(partition) = partition {
                    let mut admitted_groups = vec![false; partition.groups.len()];
                    let mut admitted_tokens = 0usize;
                    for (group_id, group) in partition.groups.iter().enumerate() {
                        check_deadline()?;
                        let residual_admitted = group.end_states.iter().any(|&end_state| {
                            token_boundary_allowed_cached(
                                state.constraint,
                                end_state,
                                &stacks,
                                &mut traversal_cache,
                            )
                        });
                        let final_terminal_admitted = !residual_admitted
                            && group.final_terminals.iter().any(|&terminal| {
                                let Some(advanced) = parser_child_cached(
                                    state.constraint,
                                    &stacks,
                                    terminal,
                                    &mut traversal_cache,
                                ) else {
                                    return false;
                                };
                                token_boundary_allowed_cached(
                                    state.constraint,
                                    initial_tsid,
                                    &advanced,
                                    &mut traversal_cache,
                                )
                            });
                        let admitted = residual_admitted || final_terminal_admitted;
                        if admitted {
                            admitted_groups[group_id] = true;
                            admitted_tokens += group.token_count;
                        }
                    }
                    if admitted_tokens != 0 {
                        for (group_id, group) in partition.groups.iter().enumerate() {
                            if admitted_groups[group_id] {
                                or_mask(buf, &group.mask);
                                continuation_groups_admitted += 1;
                            }
                        }
                        if profile {
                            eprintln!(
                                "[glrmask/profile][dynamic_continuation_use] source={} admitted_tokens={}",
                                tokenizer_state,
                                admitted_tokens,
                            );
                        }
                        let required_group_count = admitted_groups
                            .iter()
                            .filter(|&&admitted| !admitted)
                            .count();
                        if required_group_count != 0 {
                            let continuation_filter = if partition.has_narrow_group_set() {
                            let required_groups = admitted_groups
                                .iter()
                                .enumerate()
                                .fold(0u64, |groups, (group_id, &admitted)| {
                                    groups | ((!admitted) as u64) << group_id
                                });
                            let partition_index = continuation_partitions.len();
                            continuation_partitions.push(partition);
                            ContinuationFilter::Narrow {
                                partition_index,
                                required_groups,
                            }
                            } else {
                                ContinuationFilter::AlreadyMarked
                            };
                            traversal.push(TraverseWork {
                                trie_index: 0,
                                node: 0,
                                tokenizer_state,
                                gss: stacks.clone(),
                                initial_prune_guard: initial_prune_guard.clone(),
                                continuation_filter: Some(continuation_filter),
                            });
                            continuation_groups_traversed += required_group_count;
                        }
                        continue;
                    }
                }
            }

            traversal.push(TraverseWork {
                trie_index: 0,
                node: 0,
                tokenizer_state,
                gss: stacks,
                initial_prune_guard,
                continuation_filter: None,
            });
        }
    }

    // Continuation partitions prove tokens for the union of all dynamic seeds,
    // not just for the seed that selected the partition. Once any partition has
    // filled part of the output mask, an otherwise-unfiltered seed can skip
    // leaves and complete subtrees that are already globally admitted.
    if continuation_groups_admitted != 0 {
        for work in &mut traversal {
            if work.continuation_filter.is_none() {
                work.continuation_filter = Some(ContinuationFilter::AlreadyMarked);
            }
        }
    }

    while let Some(current) = traversal.pop() {
        check_deadline()?;
        work_items += 1;
        let trie = &tries[current.trie_index];
        let node = trie.node(current.node);
        match current.continuation_filter {
            Some(ContinuationFilter::Narrow {
                partition_index,
                required_groups,
            }) if continuation_partitions[partition_index].subtree_groups(current.node)
                & required_groups
                == 0 =>
            {
                continue;
            }
            Some(ContinuationFilter::AlreadyMarked)
                if subtree_is_fully_marked(vocab, trie, current.node, buf) =>
            {
                continue;
            }
            _ => {}
        }
        let subtree_action = raw_self_loop_subtree(
            state.constraint,
            trie,
            current.node,
            current.tokenizer_state,
            &current.gss,
            &current.initial_prune_guard,
            initial_tsid,
            &mut self_loop_cache,
            &mut traversal_cache,
        );
        if matches!(subtree_action, RawSelfLoopSubtree::MarkAllTokens) {
            subtree_marks += 1;
            subtree_mark_tokens += trie.subtree_tokens(current.node).len();
            mark_subtree_tokens(state.constraint, trie, current.node, buf);
            continue;
        }

        let token_is_required = match current.continuation_filter {
            None => true,
            Some(ContinuationFilter::Narrow {
                partition_index,
                required_groups,
            }) => {
                node.token_id
                    .and_then(|token_id| {
                        continuation_partitions[partition_index].token_group(token_id)
                    })
                    .is_some_and(|group| required_groups & (1u64 << group) != 0)
            }
            Some(ContinuationFilter::AlreadyMarked) => node
                .token_id
                .is_some_and(|token_id| !canonical_token_is_marked(vocab, token_id, buf)),
        };
        if token_is_required
            && node.token_id.is_some()
            && current.initial_prune_guard.allows_token_boundary()
            && (current.tokenizer_state == initial_tsid
                || token_boundary_allowed_cached(
                    state.constraint,
                    current.tokenizer_state,
                    &current.gss,
                    &mut traversal_cache,
                ))
        {
            let canonical_token_id = node.token_id.expect("token leaf checked");
            let token_ids = vocab
                .token_ids(canonical_token_id)
                .expect("dynamic vocabulary trie node lacks token ids");
            for &token_id in token_ids {
                set_mask_bit(buf, token_id);
            }
        }

        for edge in trie.children(current.node) {
            check_deadline()?;
            if let Some(ContinuationFilter::Narrow {
                partition_index,
                required_groups,
            }) = current.continuation_filter
            {
                if continuation_partitions[partition_index].subtree_groups(edge.child)
                    & required_groups
                    == 0
                {
                    continue;
                }
            }
            trie_edges += 1;
            let segment = trie.edge_bytes(edge);
            let Some(segment_prune_guard) = current
                .initial_prune_guard
                .advance(state.constraint, segment)
            else {
                continue;
            };

            segment_stack.clear();
            segment_stack.push((0usize, current.tokenizer_state, current.gss.clone()));

            while let Some((position, tokenizer_state, gss)) = segment_stack.pop() {
                check_deadline()?;
                lexer_executions += 1;
                let execution = lexer_scan_cache
                    .execute_from_state_all_widths(&segment[position..], tokenizer_state)?;
                check_deadline()?;

                for matched in &execution.matches {
                    debug_assert!(matched.width > 0);
                    let Some(advanced_parser) = parser_child_cached(
                        state.constraint,
                        &gss,
                        matched.id,
                        &mut traversal_cache,
                    )
                    else {
                        continue;
                    };

                    let next_position = position + matched.width;
                    if next_position == segment.len() {
                        traversal.push(TraverseWork {
                            trie_index: current.trie_index,
                            node: edge.child,
                            tokenizer_state: initial_tsid,
                            gss: advanced_parser,
                            initial_prune_guard: segment_prune_guard.clone(),
                            continuation_filter: current.continuation_filter,
                        });
                    } else {
                        segment_stack.push((next_position, initial_tsid, advanced_parser));
                    }
                }

                for &end_state in &execution.end_state {
                    if !lexer_state_relevant_cached(
                        state.constraint,
                        end_state,
                        &gss,
                        &mut traversal_cache,
                    ) {
                        continue;
                    }
                    traversal.push(TraverseWork {
                        trie_index: current.trie_index,
                        node: edge.child,
                        tokenizer_state: end_state,
                        gss: gss.clone(),
                        initial_prune_guard: segment_prune_guard.clone(),
                        continuation_filter: current.continuation_filter,
                    });
                }
            }
        }
    }

    update_special_token_mask(state, buf);
    update_eos_mask(state, buf);
    if let Some(cache_key) = cache_key {
        vocab.cache_mask(cache_key, buf);
    }
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][dynamic_mask] generation={} cache_hit=false key_ms={:.3} work_items={} trie_edges={} lexer_execs={} subtree_marks={} subtree_tokens={} continuation_admitted={} continuation_traversed={} boundary_cache={} relevant_cache={} child_cache={} total_ms={:.3}",
            state.generation,
            key_ms,
            work_items,
            trie_edges,
            lexer_executions,
            subtree_marks,
            subtree_mark_tokens,
            continuation_groups_admitted,
            continuation_groups_traversed,
            traversal_cache.boundary.len(),
            traversal_cache.lexer_relevant.len(),
            traversal_cache.parser_children.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraint, Vocab};
    use std::collections::BTreeSet;

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word).is_some_and(|word| word & (1u32 << bit) != 0)
    }

    fn direct_mask(state: &ConstraintState<'_>) -> Vec<u32> {
        let mut mask = vec![0u32; state.constraint.mask_len()];
        state.fill_mask_dynamic(&mut mask);
        mask
    }

    fn assert_dynamic_parity(state: &ConstraintState<'_>) {
        assert_eq!(state.mask(), direct_mask(state));
    }

    fn assert_dynamic_parity_on_reachable_states(
        constraint: &Constraint,
        max_depth: usize,
        context: &str,
    ) {
        let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];
        let mut seen = BTreeSet::new();

        for depth in 0..=max_depth {
            let mut next = Vec::new();
            for (state, path) in frontier {
                if let Some(key) = dynamic_mask_state_key(&state)
                    && !seen.insert(key)
                {
                    continue;
                }

                let static_mask = state.mask();
                let dynamic_mask = direct_mask(&state);
                assert_eq!(
                    static_mask, dynamic_mask,
                    "dynamic/static mask mismatch: {context} depth={depth} path={path:?}"
                );
                if depth == max_depth {
                    continue;
                }

                for (&token_id, bytes) in constraint.token_bytes.iter() {
                    let expected = token_allowed(&static_mask, token_id);
                    let mut advanced = state.clone();
                    let accepted = advanced.commit_bytes(bytes).is_ok();
                    assert_eq!(
                        accepted, expected,
                        "static mask/commit mismatch during dynamic sweep: {context} depth={depth} path={path:?} token={token_id}"
                    );
                    if accepted {
                        let mut next_path = path.clone();
                        next_path.push(token_id);
                        next.push((advanced, next_path));
                    }
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn dynamic_mask_matches_normal_for_repeat_and_cross_terminal_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"aab".to_vec()),
                (5, b"aaa".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));

        state.commit_token(2).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_trie_is_rebuilt_after_load() {
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec())],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert_dynamic_parity(&loaded.start());
    }

    #[test]
    fn dynamic_mask_keeps_duplicate_byte_token_aliases() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (7, b"a".to_vec()),
                (12, b"ab".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        let mask = direct_mask(&state);
        assert!(token_allowed(&mask, 0));
        assert!(token_allowed(&mask, 7));
        assert!(token_allowed(&mask, 12));

        state.commit_token(7).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
    }

    #[test]
    fn dynamic_mask_matches_normal_across_an_ignore_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b" b".to_vec()),
                (4, b"  b".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(3).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_preserves_repeated_terminal_after_ignore_reset_inside_token() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"aa".to_vec()),
                (4, b"bb".to_vec()),
                (5, b"cc".to_vec()),
                (6, b"ab".to_vec()),
                (7, b"ac".to_vec()),
                (8, b"ba".to_vec()),
                (9, b"bc".to_vec()),
                (10, b"abc".to_vec()),
                (11, b"aab".to_vec()),
                (12, b"abb".to_vec()),
                (13, b"acc".to_vec()),
                (14, b" ".to_vec()),
                (15, b"  ".to_vec()),
                (16, b" a".to_vec()),
                (17, b"a ".to_vec()),
                (18, b" a ".to_vec()),
                (19, b"ab c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
lexer group ws ::= WS;
lexer group a ::= A;
lexer group b ::= B;
lexer group c ::= C;
t WS ::= " "+;
t A ::= "a"+;
t B ::= "b";
t C ::= "c";
nt item ::= A | B | C;
nt start ::= item item? item?;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        state.commit_token(16).unwrap();

        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 0));
        assert!(token_allowed(&direct_mask(&state), 3));
        assert!(token_allowed(&direct_mask(&state), 17));
    }

    #[test]
    fn masks_preserve_overlap_continuation_after_ignore_reset() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
                (3, b" ".to_vec()),
                (4, b" a".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
t WS ::= " "+;
t A ::= "ab";
t B ::= "a" | "ab";
nt item ::= A | B;
nt start ::= item item? item?;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        state.commit_token(4).unwrap();

        let static_mask = state.mask();
        let dynamic_mask = direct_mask(&state);

        let mut probe = state.clone();
        assert!(probe.commit_bytes(b"b").is_ok());
        assert!(
            token_allowed(&static_mask, 1),
            "static mask must admit b because a-ab = B WS A"
        );
        assert!(
            token_allowed(&dynamic_mask, 1),
            "dynamic mask must admit b because a-ab = B WS A"
        );
    }

    #[test]
    fn dynamic_mask_generated_small_language_sweep() {
        const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
        let vocab = Vocab::new(
            [
                "a", "b", "ab", "ba", " ", " a", "a ", " b", "b ", " a ", " b ",
            ]
            .into_iter()
            .enumerate()
            .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
            .collect(),
            None,
        );
        let languages = (1u32..1u32 << WORDS.len())
            .filter(|mask| mask.count_ones() <= 2)
            .collect::<Vec<_>>();
        let rule = |name: &str, mask: u32| {
            let rhs = WORDS
                .iter()
                .enumerate()
                .filter_map(|(index, word)| {
                    (mask & (1 << index) != 0).then(|| format!("\"{word}\""))
                })
                .collect::<Vec<_>>()
                .join(" | ");
            format!("t {name} ::= {rhs};\n")
        };

        for grouped in [false, true] {
            for ignored in [false, true] {
                let grouping = if grouped {
                    if ignored {
                        "lexer group ws ::= WS;\nlexer group a ::= A;\nlexer group b ::= B;\n"
                    } else {
                        "lexer group a ::= A;\nlexer group b ::= B;\n"
                    }
                } else {
                    ""
                };
                let ignore = if ignored {
                    "ignore WS;\nt WS ::= \" \"+;\n"
                } else {
                    ""
                };

                for &a in &languages {
                    for &b in &languages {
                        if grouped && a == b {
                            continue;
                        }
                        for start_rule in [
                            "nt item ::= A | B;\nnt start ::= item item? item?;",
                            "nt start ::= A A | B B;",
                            "nt start ::= A B | B A;",
                        ] {
                            let grammar = format!(
                                "start start;\n{ignore}{grouping}{}{}{start_rule}\n",
                                rule("A", a),
                                rule("B", b),
                            );
                            let constraint =
                                Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                            let context = format!(
                                "finite grouped={grouped} ignored={ignored} A={a:#06b} B={b:#06b}\ngrammar:\n{grammar}"
                            );
                            assert_dynamic_parity_on_reachable_states(&constraint, 3, &context);
                        }
                    }
                }

                let grammar = format!(
                    "start start;\n{ignore}{grouping}t A ::= \"a\"+;\nt B ::= \"b\"+;\nnt item ::= A | B;\nnt start ::= item item? item?;\n"
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let context = format!(
                    "repeat grouped={grouped} ignored={ignored}\ngrammar:\n{grammar}"
                );
                assert_dynamic_parity_on_reachable_states(&constraint, 4, &context);

                let grammar = format!(
                    "start start;\n{ignore}{grouping}t A ::= \"a\"+ \"b\";\nt B ::= \"a\"+;\nnt item ::= A | B;\nnt start ::= item item? item?;\n"
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let context = format!(
                    "delayed-overlap grouped={grouped} ignored={ignored}\ngrammar:\n{grammar}"
                );
                assert_dynamic_parity_on_reachable_states(&constraint, 4, &context);
            }
        }
    }

    #[test]
    fn dynamic_mask_matches_normal_at_every_reachable_small_state() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bb".to_vec()),
                (4, b"c".to_vec()),
                (5, b"ab".to_vec()),
                (6, b"ba".to_vec()),
                (7, b"a c".to_vec()),
                (8, b"b c".to_vec()),
                (9, b" aa".to_vec()),
                (10, b" bb".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b'+;
t C ::= 'c';
nt start ::= A B C | B A C | A C | B C;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..11u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_when_one_repeated_terminal_crosses_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"aaa".to_vec()),
                (3, b"aaaa".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..4u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_for_a_partial_json_string() {
        let vocab = Vocab::new(
            vec![
                (0, b"\"".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
                (3, b"\\\"".to_vec()),
                (4, b"\"a".to_vec()),
                (5, b"a\"".to_vec()),
            ],
            None,
        );
        let constraint =
            Constraint::from_json_schema(r#"{"type":"string"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_monolithic_json_number() {
        let vocab = Vocab::new(
            vec![
                (0, b"-".to_vec()),
                (1, b"0".to_vec()),
                (2, b"1".to_vec()),
                (3, b"2".to_vec()),
                (4, b"3".to_vec()),
                (5, b".".to_vec()),
                (6, b"e".to_vec()),
                (7, b"+".to_vec()),
            ],
            None,
        );
        let constraint =
            Constraint::from_json_schema(r#"{"type":"number"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
            state.commit_bytes(bytes).unwrap();
            assert_dynamic_parity(&state);
        }
        assert!(state.is_complete());
    }

    #[test]
    fn dynamic_mask_keeps_other_gss_paths_when_one_path_excludes_a_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'a';
t C ::= 'c';
t D ::= 'b';
nt start ::= A C | B D;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();

        let paths = state
            .state
            .values()
            .flat_map(|gss| gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"))
            .collect::<Vec<_>>();
        assert!(paths.iter().any(|(_, exclusions)| exclusions.is_empty()));
        assert!(paths.iter().any(|(_, exclusions)| !exclusions.is_empty()));

        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
        state.commit_token(1).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_overlapping_live_terminal_paths() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'b' | 'bc';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        state.commit_token(3).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_a_live_cross_terminal_prefix() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"abc".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'abc';
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        assert!(!token_allowed(&state.mask(), 4));
        state.commit_token(3).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

}
