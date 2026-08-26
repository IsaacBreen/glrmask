//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::hash::{Hash, Hasher};

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{
    TokenizerExecResult, TokenizerMatch, TokenizerStateSet,
};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    advance_stacks, stack_admissible_terminals, stack_may_advance_on_any, ParserGSS,
};
use crate::compiler::glr::table::Action;
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::LeveledGSS;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

use super::artifact::{
    Constraint, DynamicMaskStateKey, DynamicMaskTrie, DynamicMaskVocab,
    DynamicSelfLoopProjection,
};
use super::state::ConstraintState;

type ParserStacks = LeveledGSS<u32, ()>;

#[derive(Default)]
struct DynamicTraversalCache {
    admissible_terminals: FxHashMap<usize, (ParserStacks, BitSet)>,
    terminal_admissible: FxHashMap<(usize, TerminalID), bool>,
    lexer_relevant: FxHashMap<(u32, usize), bool>,
    parser_children: FxHashMap<(usize, TerminalID), (ParserStacks, Option<ParserStacks>)>,
    profile_interaction_hash: Option<u64>,
    profile_interaction_events: usize,
    profile_parser_action_counts: [usize; 10],
    profile_parser_child_terminals: SmallVec<[TerminalID; 32]>,
}

impl DynamicTraversalCache {
    #[inline]
    fn profile_event(&mut self, tag: u64, left: u64, right: u64) {
        let Some(hash) = self.profile_interaction_hash.as_mut() else {
            return;
        };
        *hash = (*hash ^ tag).wrapping_mul(0x100000001b3);
        *hash = (*hash ^ left).wrapping_mul(0x100000001b3);
        *hash = (*hash ^ right).wrapping_mul(0x100000001b3);
        self.profile_interaction_events += 1;
    }
}

#[inline]
fn parser_stacks_cache_key(stacks: &ParserStacks) -> usize {
    stacks
        .single_interface_lower_id()
        .unwrap_or_else(|| stacks.ptr_key())
}


const DYNAMIC_NO_COMPONENT: u32 = u32::MAX;

#[inline]
fn overlay_terminal_component(constraint: &Constraint, terminal: TerminalID) -> u32 {
    let Some(metadata) = constraint.static_dynamic_overlay.as_ref() else {
        return DYNAMIC_NO_COMPONENT;
    };
    metadata
        .terminal_offsets
        .partition_point(|&offset| offset <= terminal)
        .saturating_sub(1) as u32
}

#[inline]
fn overlay_tokenizer_component(constraint: &Constraint, tokenizer_state: u32) -> u32 {
    if tokenizer_state == 0 {
        return DYNAMIC_NO_COMPONENT;
    }
    let Some(metadata) = constraint.static_dynamic_overlay.as_ref() else {
        return DYNAMIC_NO_COMPONENT;
    };
    metadata
        .tokenizer_state_offsets
        .partition_point(|&offset| offset <= tokenizer_state)
        .saturating_sub(1) as u32
}

#[inline]
fn overlay_advance_repair(
    constraint: &Constraint,
    branch: &DynamicBranch,
    terminal: TerminalID,
) -> (u32, bool) {
    let next_component = overlay_terminal_component(constraint, terminal);
    let switched = branch.last_component != DYNAMIC_NO_COMPONENT
        && next_component != DYNAMIC_NO_COMPONENT
        && branch.last_component != next_component;
    let terminal_repairs = constraint
        .static_dynamic_overlay
        .as_ref()
        .and_then(|metadata| metadata.repair_terminals.get(terminal as usize))
        .copied()
        .unwrap_or(false);
    (
        next_component,
        branch.repair_used || switched || terminal_repairs,
    )
}

#[inline]
fn lazy_repair_parser_enabled(constraint: &Constraint, branch: &DynamicBranch) -> bool {
    // Compact composition makes zero-width CALL/RETURN part of the live parser
    // semantics. Deferring parser execution would also defer the only exact
    // observation of whether this model token has crossed a component boundary.
    if constraint.uses_compact_segmented_parser_runtime() {
        return false;
    }
    !branch.repair_used
        && constraint.static_dynamic_overlay.is_some()
        && std::env::var_os("GLRMASK_EXPERIMENT_LAZY_REPAIR_PARSER").is_some()
}

#[inline]
fn compact_parser_scope_changed(
    constraint: &Constraint,
    before: &ParserStacks,
    after: &ParserStacks,
) -> bool {
    if !constraint.uses_compact_segmented_parser_runtime() {
        return false;
    }
    fn scopes(
        constraint: &Constraint,
        stacks: &ParserStacks,
    ) -> Option<SmallVec<[usize; 4]>> {
        let mut result = SmallVec::<[usize; 4]>::new();
        for top in stacks.peek_values() {
            let (component, _) = constraint.compact_segmented_parser_component(top)?;
            if !result.contains(&component) {
                result.push(component);
            }
        }
        result.sort_unstable();
        Some(result)
    }
    match (scopes(constraint, before), scopes(constraint, after)) {
        (Some(before), Some(after)) => before != after,
        // Unknown scoped state is not a reason to suppress B. A conservative
        // repair mark can only duplicate a token already admitted by A.
        _ => true,
    }
}

fn replay_pending_terminals(
    constraint: &Constraint,
    gss: &ParserStacks,
    pending: &[TerminalID],
    cache: &mut DynamicTraversalCache,
) -> Option<ParserStacks> {
    let mut current = gss.clone();
    for &terminal in pending {
        current = parser_child_cached(constraint, &current, terminal, cache)?;
    }
    Some(current)
}

fn replay_pending_with_terminal(
    constraint: &Constraint,
    branch: &DynamicBranch,
    terminal: TerminalID,
    cache: &mut DynamicTraversalCache,
) -> Option<ParserStacks> {
    let current = replay_pending_terminals(constraint, &branch.gss, &branch.pending_terminals, cache)?;
    parser_child_cached(constraint, &current, terminal, cache)
}



#[derive(Clone)]
struct DynamicBranch {
    tokenizer_config: u32,
    gss: ParserStacks,
    initial_prune_guard: InitialPruneGuard,
    /// Component containing the most recently committed lexical terminal.
    /// `u32::MAX` means the merged reset dispatcher has not selected one yet.
    last_component: u32,
    /// True once this token path has used behavior absent from the transported
    /// component parser artifacts: either a cross-component terminal switch or
    /// a terminal whose composed template requires additive repair.
    repair_used: bool,
    /// Parser terminals matched since the last materialized parser advance.
    /// Additive repair masking may defer these while `repair_used == false`:
    /// such a branch cannot contribute a token until a repair event occurs.
    pending_terminals: SmallVec<[TerminalID; 4]>,
    /// The lexer was reset by a terminal match on the most recently consumed
    /// byte. At a compressed-edge boundary that fresh initial state is already
    /// a valid continuation and must not be stripped as though it were an
    /// unmatched residual configuration.
    fresh_reset: bool,
}

type DynamicBranches = SmallVec<[DynamicBranch; 4]>;

#[derive(Clone, PartialEq, Eq, Hash)]
enum InitialPruneGuard {
    Passed,
    Pending {
        memories: Arc<[(u32, TerminalID)]>,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DynamicBranchKey {
    tokenizer_config: u32,
    gss_ptr: usize,
    initial_prune_guard: InitialPruneGuard,
    last_component: u32,
    repair_used: bool,
    pending_terminals: SmallVec<[TerminalID; 4]>,
    fresh_reset: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DynamicSimpleBranchKey {
    tokenizer_config: u32,
    gss_ptr: usize,
    last_component: u32,
    repair_used: bool,
}

const DYNAMIC_RECOGNIZER_UNKNOWN: u32 = u32::MAX;
const DYNAMIC_RECOGNIZER_DEAD: u32 = u32::MAX - 1;

struct DynamicRecognizerStateCache {
    state_ids: FxHashMap<Box<[DynamicBranchKey]>, u32>,
    simple_state_ids: FxHashMap<DynamicSimpleBranchKey, u32>,
    branches: Vec<DynamicBranches>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    normalized: Vec<u32>,
    metadata: Vec<Option<DynamicRecognizerStateMetadata>>,
    transition_misses: usize,
}

struct DynamicRecognizerStateMetadata {
    token_boundary_allowed: bool,
    subtree_loop_bytes: SmallVec<[U8Set; 4]>,
    repair_token_boundary_allowed: bool,
    repair_subtree_loop_bytes: SmallVec<[U8Set; 4]>,
}

impl DynamicRecognizerStateCache {
    fn new() -> Self {
        Self {
            state_ids: FxHashMap::default(),
            simple_state_ids: FxHashMap::default(),
            branches: Vec::new(),
            transitions: Vec::new(),
            normalized: Vec::new(),
            metadata: Vec::new(),
            transition_misses: 0,
        }
    }

    fn branch_key(branch: &DynamicBranch) -> DynamicBranchKey {
        DynamicBranchKey {
            tokenizer_config: branch.tokenizer_config,
            gss_ptr: branch.gss.ptr_key(),
            initial_prune_guard: branch.initial_prune_guard.clone(),
            last_component: branch.last_component,
            repair_used: branch.repair_used,
            pending_terminals: branch.pending_terminals.clone(),
            fresh_reset: branch.fresh_reset,
        }
    }

    #[inline]
    fn simple_branch_key(branch: &DynamicBranch) -> Option<DynamicSimpleBranchKey> {
        (branch.initial_prune_guard.is_passed()
            && branch.pending_terminals.is_empty()
            && !branch.fresh_reset)
            .then_some(DynamicSimpleBranchKey {
                tokenizer_config: branch.tokenizer_config,
                gss_ptr: branch.gss.ptr_key(),
                last_component: branch.last_component,
                repair_used: branch.repair_used,
            })
    }

    fn intern(&mut self, branches: DynamicBranches) -> u32 {
        if let [branch] = branches.as_slice()
            && let Some(key) = Self::simple_branch_key(branch)
        {
            if let Some(&state_id) = self.simple_state_ids.get(&key) {
                return state_id;
            }
            let state_id = self.branches.len() as u32;
            debug_assert!(state_id < DYNAMIC_RECOGNIZER_DEAD);
            self.simple_state_ids.insert(key, state_id);
            self.branches.push(branches);
            self.transitions.push(None);
            self.normalized.push(DYNAMIC_RECOGNIZER_UNKNOWN);
            self.metadata.push(None);
            return state_id;
        }
        let key = branches
            .iter()
            .map(Self::branch_key)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        if let Some(&state_id) = self.state_ids.get(key.as_ref()) {
            return state_id;
        }
        let state_id = self.branches.len() as u32;
        debug_assert!(state_id < DYNAMIC_RECOGNIZER_DEAD);
        self.state_ids.insert(key, state_id);
        self.branches.push(branches);
        self.transitions.push(None);
        self.normalized.push(DYNAMIC_RECOGNIZER_UNKNOWN);
        self.metadata.push(None);
        state_id
    }

    #[inline]
    fn intern_simple_successor(&mut self, source_state_id: u32, tokenizer_config: u32) -> u32 {
        let key = {
            let source = &self.branches[source_state_id as usize][0];
            debug_assert!(Self::simple_branch_key(source).is_some());
            DynamicSimpleBranchKey {
                tokenizer_config,
                gss_ptr: source.gss.ptr_key(),
                last_component: source.last_component,
                repair_used: source.repair_used,
            }
        };
        if let Some(&state_id) = self.simple_state_ids.get(&key) {
            return state_id;
        }
        let branch = {
            let source = &self.branches[source_state_id as usize][0];
            DynamicBranch {
                tokenizer_config,
                gss: source.gss.clone(),
                initial_prune_guard: InitialPruneGuard::Passed,
                last_component: source.last_component,
                repair_used: source.repair_used,
                pending_terminals: SmallVec::new(),
                fresh_reset: false,
            }
        };
        let state_id = self.branches.len() as u32;
        debug_assert!(state_id < DYNAMIC_RECOGNIZER_DEAD);
        self.simple_state_ids.insert(key, state_id);
        let mut branches = DynamicBranches::new();
        branches.push(branch);
        self.branches.push(branches);
        self.transitions.push(None);
        self.normalized.push(DYNAMIC_RECOGNIZER_UNKNOWN);
        self.metadata.push(None);
        state_id
    }

    #[inline]
    fn branches(&self, state_id: u32) -> &DynamicBranches {
        &self.branches[state_id as usize]
    }

    fn profile_shape_into(&self, stats: &mut DynamicWalkStats) {
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_none() {
            return;
        }
        for branches in &self.branches {
            match branches.as_slice() {
                [branch] if Self::simple_branch_key(branch).is_some() => {
                    stats.recognizer_simple_one_states += 1;
                }
                [left, right]
                    if Self::simple_branch_key(left).is_some()
                        && Self::simple_branch_key(right).is_some() =>
                {
                    stats.recognizer_simple_two_states += 1;
                }
                _ => stats.recognizer_other_states += 1,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn metadata(
        &mut self,
        state_id: u32,
        constraint: &Constraint,
        initial_config: u32,
        lexer_scan_cache: &DynamicNfaScanCache<'_>,
        raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
        config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
        traversal_cache: &mut DynamicTraversalCache,
    ) -> &DynamicRecognizerStateMetadata {
        let state_index = state_id as usize;
        if self.metadata[state_index].is_none() {
            self.initialize_metadata(
                state_index,
                constraint,
                initial_config,
                lexer_scan_cache,
                raw_self_loop_cache,
                config_self_loop_cache,
                traversal_cache,
            );
        }
        // `initialize_metadata` always fills exactly this slot.
        unsafe {
            self.metadata
                .get_unchecked(state_index)
                .as_ref()
                .unwrap_unchecked()
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    fn initialize_metadata(
        &mut self,
        state_index: usize,
        constraint: &Constraint,
        initial_config: u32,
        lexer_scan_cache: &DynamicNfaScanCache<'_>,
        raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
        config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
        traversal_cache: &mut DynamicTraversalCache,
    ) {
        let mut token_boundary_allowed = false;
        let mut subtree_loop_bytes = SmallVec::<[U8Set; 4]>::new();
        let mut repair_token_boundary_allowed = false;
        let mut repair_subtree_loop_bytes = SmallVec::<[U8Set; 4]>::new();
        let collect_subtree_loops = self.branches[state_index].len() == 1
            || dynamic_multi_branch_subtree_loop_min_tokens() != usize::MAX;
        for branch in &self.branches[state_index] {
            if !branch.initial_prune_guard.allows_token_boundary() {
                continue;
            }

            let lazy_pending = lazy_repair_parser_enabled(constraint, branch)
                && !branch.pending_terminals.is_empty();
            let boundary_allowed = if lazy_pending {
                false
            } else {
                branch.tokenizer_config == initial_config
                    || config_token_boundary_allowed_cached(
                        constraint,
                        lexer_scan_cache,
                        branch.tokenizer_config,
                        &branch.gss,
                        traversal_cache,
                    )
            };
            token_boundary_allowed |= boundary_allowed;
            // B is the language of this *complete model token* that actually
            // used composed-only behavior. A prefix that could cross a linker
            // after more bytes must keep its trie descendants live, but it is
            // not itself a B token. In particular this prevents a scoped
            // ignore from being accepted merely because some longer token
            // beginning with the same bytes later enters another component.
            let repair_boundary_allowed = branch.repair_used && boundary_allowed;
            repair_token_boundary_allowed |= repair_boundary_allowed;

            if collect_subtree_loops
                && boundary_allowed
                && branch.initial_prune_guard.is_passed()
                && branch.tokenizer_config != initial_config
            {
                let loop_bytes = cached_config_self_loop_bytes(
                    constraint,
                    lexer_scan_cache,
                    branch.tokenizer_config,
                    raw_self_loop_cache,
                    config_self_loop_cache,
                );
                if !subtree_loop_bytes.contains(&loop_bytes) {
                    subtree_loop_bytes.push(loop_bytes);
                }
                if branch.repair_used && !repair_subtree_loop_bytes.contains(&loop_bytes) {
                    repair_subtree_loop_bytes.push(loop_bytes);
                }
            }
        }
        self.metadata[state_index] = Some(DynamicRecognizerStateMetadata {
            token_boundary_allowed,
            subtree_loop_bytes,
            repair_token_boundary_allowed,
            repair_subtree_loop_bytes,
        });
    }


    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        state_id: u32,
        byte: u8,
        constraint: &Constraint,
        initial_config: u32,
        lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
        traversal_cache: &mut DynamicTraversalCache,
        stats: &mut DynamicWalkStats,
    ) -> Result<Option<u32>, String> {
        stats.branch_steps += self.branches(state_id).len();
        if let Some(row) = self.transitions[state_id as usize].as_ref() {
            let cached = row[byte as usize];
            if cached != DYNAMIC_RECOGNIZER_UNKNOWN {
                return Ok((cached != DYNAMIC_RECOGNIZER_DEAD).then_some(cached));
            }
        }

        self.transition_misses += 1;
        // The dominant ordinary trie path has one parser/lexer branch and no
        // pending token-start guard. If consuming this byte produces no lexer
        // terminal match, the entire recognizer transition is just a change of
        // tokenizer config: parser stacks and all other branch metadata are
        // identical. Avoid cloning/keying the generic branch vector for that
        // case. Any actual terminal match falls back to the exact generic path
        // below.
        if self.branches(state_id).len() == 1
            && Self::simple_branch_key(&self.branches[state_id as usize][0]).is_some()
        {
            let source_config = self.branches[state_id as usize][0].tokenizer_config;
            let fast_target = lexer_scan_cache.step_config(source_config, byte)?;
            if let Some(next_config) = fast_target {
                let has_match = (0..lexer_scan_cache.config_len(next_config)).any(|index| {
                    let raw_state = lexer_scan_cache.config_state(next_config, index);
                    constraint
                        .tokenizer
                        .matched_terminals_iter(raw_state)
                        .next()
                        .is_some()
                });
                if !has_match {
                    let target = self.intern_simple_successor(state_id, next_config);
                    self.transitions[state_id as usize]
                        .get_or_insert_with(|| Box::new([DYNAMIC_RECOGNIZER_UNKNOWN; 256]))
                        [byte as usize] = target;
                    return Ok(Some(target));
                }
            } else {
                self.transitions[state_id as usize]
                    .get_or_insert_with(|| Box::new([DYNAMIC_RECOGNIZER_UNKNOWN; 256]))
                    [byte as usize] = DYNAMIC_RECOGNIZER_DEAD;
                return Ok(None);
            }
        }
        let mut next = DynamicBranches::new();
        for branch_index in 0..self.branches(state_id).len() {
            let branch = self.branches(state_id)[branch_index].clone();
            let Some(next_config) = lexer_scan_cache.step_config(branch.tokenizer_config, byte)?
            else {
                continue;
            };
            let Some(advanced_prune_guard) = branch
                .initial_prune_guard
                .advance(constraint, std::slice::from_ref(&byte))
            else {
                continue;
            };

            for config_index in 0..lexer_scan_cache.config_len(next_config) {
                let matched_state = lexer_scan_cache.config_state(next_config, config_index);
                for terminal in constraint.tokenizer.matched_terminals_iter(matched_state) {
                    let matched_prune_guard = if Some(terminal) == constraint.ignore_terminal {
                        advanced_prune_guard.clone()
                    } else {
                        advanced_prune_guard.remember_terminal_match(
                            constraint,
                            matched_state,
                            terminal,
                        )
                    };
                    let (last_component, repair_used) =
                        overlay_advance_repair(constraint, &branch, terminal);
                    if lazy_repair_parser_enabled(constraint, &branch)
                        && !repair_used
                        && branch.pending_terminals.is_empty()
                    {
                        let mut pending_terminals = branch.pending_terminals.clone();
                        pending_terminals.push(terminal);
                        next.push(DynamicBranch {
                            tokenizer_config: initial_config,
                            gss: branch.gss.clone(),
                            initial_prune_guard: matched_prune_guard,
                            last_component,
                            repair_used: false,
                            pending_terminals,
                            fresh_reset: true,
                        });
                    } else {
                        let advanced_parser = if lazy_repair_parser_enabled(constraint, &branch) {
                            replay_pending_with_terminal(
                                constraint,
                                &branch,
                                terminal,
                                traversal_cache,
                            )
                        } else {
                            parser_child_cached(
                                constraint,
                                &branch.gss,
                                terminal,
                                traversal_cache,
                            )
                        };
                        let Some(advanced_parser) = advanced_parser else {
                            continue;
                        };
                        let repair_used = repair_used
                            || compact_parser_scope_changed(
                                constraint,
                                &branch.gss,
                                &advanced_parser,
                            );
                        next.push(DynamicBranch {
                            tokenizer_config: initial_config,
                            gss: advanced_parser,
                            initial_prune_guard: matched_prune_guard,
                            last_component,
                            repair_used,
                            pending_terminals: SmallVec::new(),
                            fresh_reset: true,
                        });
                    }
                }
            }

            next.push(DynamicBranch {
                tokenizer_config: next_config,
                gss: branch.gss,
                initial_prune_guard: advanced_prune_guard,
                last_component: branch.last_component,
                repair_used: branch.repair_used,
                pending_terminals: branch.pending_terminals.clone(),
                fresh_reset: false,
            });
        }

        let target = if next.is_empty() {
            DYNAMIC_RECOGNIZER_DEAD
        } else {
            stats.max_branches = stats.max_branches.max(next.len());
            self.intern(next)
        };
        self.transitions[state_id as usize]
            .get_or_insert_with(|| Box::new([DYNAMIC_RECOGNIZER_UNKNOWN; 256]))[byte as usize] =
            target;
        Ok((target != DYNAMIC_RECOGNIZER_DEAD).then_some(target))
    }

    fn normalize(
        &mut self,
        state_id: u32,
        constraint: &Constraint,
        lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
        traversal_cache: &mut DynamicTraversalCache,
        stats: &mut DynamicWalkStats,
    ) -> Result<Option<u32>, String> {
        let cached = self.normalized[state_id as usize];
        if cached != DYNAMIC_RECOGNIZER_UNKNOWN {
            return Ok((cached != DYNAMIC_RECOGNIZER_DEAD).then_some(cached));
        }

        // For the ordinary scalar path normalization is very often the
        // identity: there was no fresh reset, the residual lexer config is the
        // same non-final state, and at least one future lexer terminal remains
        // relevant to the unchanged parser stack. Avoid cloning the branch and
        // re-entering the recognizer interner in that case. If residualization
        // changes the config (or the branch is not the simple form), preserve
        // the generic path below exactly.
        if self.branches(state_id).len() == 1
            && Self::simple_branch_key(&self.branches[state_id as usize][0]).is_some()
        {
            let tokenizer_config = self.branches[state_id as usize][0].tokenizer_config;
            if let Some(end_config) = lexer_scan_cache.residual_config(tokenizer_config)?
                && end_config == tokenizer_config
            {
                let relevant = {
                    let branch = &self.branches[state_id as usize][0];
                    lexer_config_relevant_cached(
                        constraint,
                        lexer_scan_cache,
                        tokenizer_config,
                        &branch.gss,
                        traversal_cache,
                    )
                };
                let target = if relevant {
                    state_id
                } else {
                    DYNAMIC_RECOGNIZER_DEAD
                };
                self.normalized[state_id as usize] = target;
                return Ok((target != DYNAMIC_RECOGNIZER_DEAD).then_some(target));
            }
        }

        let mut normalized = DynamicBranches::new();
        for branch_index in 0..self.branches(state_id).len() {
            let mut branch = self.branches(state_id)[branch_index].clone();
            let was_fresh_reset = branch.fresh_reset;
            let end_config = if was_fresh_reset {
                branch.fresh_reset = false;
                branch.tokenizer_config
            } else {
                let Some(end_config) =
                    lexer_scan_cache.residual_config(branch.tokenizer_config)?
                else {
                    continue;
                };
                end_config
            };
            if !was_fresh_reset
                && !(lazy_repair_parser_enabled(constraint, &branch)
                    && !branch.pending_terminals.is_empty())
                && !lexer_config_relevant_cached(
                    constraint,
                    lexer_scan_cache,
                    end_config,
                    &branch.gss,
                    traversal_cache,
                )
            {
                continue;
            }
            branch.tokenizer_config = end_config;
            normalized.push(branch);
        }

        let target = if normalized.is_empty() {
            DYNAMIC_RECOGNIZER_DEAD
        } else {
            stats.max_branches = stats.max_branches.max(normalized.len());
            self.intern(normalized)
        };
        self.normalized[state_id as usize] = target;
        Ok((target != DYNAMIC_RECOGNIZER_DEAD).then_some(target))
    }
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
fn set_mask_bit_known_in_range(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    debug_assert!(word < buf.len());
    // Dynamic vocabulary ids come from the same Vocab used to size the mask.
    // Avoid a bounds branch for every accepted token in large subtree marks.
    unsafe {
        *buf.get_unchecked_mut(word) |= 1u32 << bit;
    }
}

#[derive(Debug)]
struct DynamicBaselineCoverage {
    /// Prefix count over `DynamicMaskTrie::all_subtree_tokens()`: one unit for
    /// each canonical token with at least one original-token alias not already
    /// present in the static baseline mask. Every trie subtree occupies a
    /// contiguous interval in that token order, making `subtree_fully_covered`
    /// O(1) while preserving exact alias semantics.
    uncovered_prefix: Vec<u32>,
}

impl DynamicBaselineCoverage {
    fn new(vocab: &DynamicMaskVocab, trie: &DynamicMaskTrie, baseline: &[u32]) -> Self {
        let ordered = trie.all_subtree_tokens();
        let mut uncovered_prefix = Vec::with_capacity(ordered.len() + 1);
        uncovered_prefix.push(0);
        let mut missing = 0u32;
        for &canonical_token in ordered {
            let covered = vocab.token_ids(canonical_token).is_some_and(|aliases| {
                !aliases.is_empty()
                    && aliases.iter().all(|&token_id| {
                        let word = token_id as usize / 32;
                        let bit = token_id % 32;
                        baseline
                            .get(word)
                            .is_some_and(|bits| (bits & (1u32 << bit)) != 0)
                    })
            });
            missing += u32::from(!covered);
            uncovered_prefix.push(missing);
        }
        Self { uncovered_prefix }
    }

    #[inline(always)]
    fn subtree_fully_covered(&self, trie: &DynamicMaskTrie, node: u32) -> bool {
        let range = trie.subtree_token_index_range(node);
        unsafe {
            *self.uncovered_prefix.get_unchecked(range.start)
                == *self.uncovered_prefix.get_unchecked(range.end)
        }
    }
}

#[derive(Debug)]
enum DynamicCandidateCoverage {
    /// Prefix count over `DynamicMaskTrie::all_subtree_tokens()`: one unit for
    /// each canonical token with at least one original-token alias present in
    /// the candidate mask. This lets the exact dynamic recognizer skip every
    /// trie subtree that cannot possibly contribute a requested candidate.
    Prefix(Vec<u32>),
    /// Build-time materialized equivalent for a fixed candidate set. One bit
    /// per trie node says whether that complete subtree contains any candidate.
    SubtreeBits(Arc<[u64]>),
}

impl DynamicCandidateCoverage {
    fn new(vocab: &DynamicMaskVocab, trie: &DynamicMaskTrie, candidates: &[u32]) -> Self {
        let ordered = trie.all_subtree_tokens();
        let mut candidate_prefix = Vec::with_capacity(ordered.len() + 1);
        candidate_prefix.push(0);
        let mut count = 0u32;
        for &canonical_token in ordered {
            let is_candidate = vocab.token_ids(canonical_token).is_some_and(|aliases| {
                aliases.iter().any(|&token_id| {
                    let word = token_id as usize / 32;
                    let bit = token_id % 32;
                    candidates
                        .get(word)
                        .is_some_and(|bits| (bits & (1u32 << bit)) != 0)
                })
            });
            count += u32::from(is_candidate);
            candidate_prefix.push(count);
        }
        Self::Prefix(candidate_prefix)
    }

    #[inline]
    fn from_subtree_bits(bits: Arc<[u64]>) -> Self {
        Self::SubtreeBits(bits)
    }

    #[inline(always)]
    fn subtree_has_candidate(&self, trie: &DynamicMaskTrie, node: u32) -> bool {
        match self {
            Self::Prefix(candidate_prefix) => {
                let range = trie.subtree_token_index_range(node);
                unsafe {
                    *candidate_prefix.get_unchecked(range.start)
                        != *candidate_prefix.get_unchecked(range.end)
                }
            }
            Self::SubtreeBits(bits) => {
                let word = node as usize >> 6;
                let bit = node & 63;
                bits.get(word)
                    .is_some_and(|bits| bits & (1u64 << bit) != 0)
            }
        }
    }
}

const DYNAMIC_TOKEN_MARKER_FALLBACK: u64 = 1u64 << 63;

#[inline(always)]
fn mark_dynamic_token_marker(vocab: &DynamicMaskVocab, marker: u64, buf: &mut [u32]) {
    debug_assert_ne!(marker, 0);
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        let word = (marker >> 32) as usize;
        let bits = marker as u32;
        debug_assert_ne!(bits, 0);
        debug_assert!(word < buf.len());
        unsafe {
            *buf.get_unchecked_mut(word) |= bits;
        }
        return;
    }

    let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
    let token_ids = vocab
        .token_ids(canonical_token)
        .expect("dynamic vocabulary trie node lacks token ids");
    for &token_id in token_ids {
        set_mask_bit_known_in_range(buf, token_id);
    }
}

const DYNAMIC_NFA_CONFIG_UNKNOWN: u32 = u32::MAX;
const DYNAMIC_NFA_CONFIG_DEAD: u32 = u32::MAX - 1;
// In a tokenizer that contains epsilon edges somewhere, most runtime states can
// still be ordinary scalar DFA states with no outgoing epsilon transition. Do
// not force those states through the generic NFA-config interner. The high bit
// tags a raw tokenizer state; interned multi-state configs remain dense low
// integers. Tokenizer state counts are already u32-sized and in practice many
// orders of magnitude below this boundary; fail closed if that ever changes.
const DYNAMIC_NFA_RAW_CONFIG_TAG: u32 = 1 << 31;

#[derive(Clone)]
struct DynamicNfaScanCache<'a> {
    constraint: &'a Constraint,
    deterministic: bool,
    deadline: Option<Instant>,
    max_collection_items: Option<usize>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    residual_configs: Vec<u32>,
    raw_start_config: FxHashMap<u32, u32>,
}

struct DynamicConfigExecResult {
    end_config: Option<u32>,
    matches: Vec<TokenizerMatch>,
}

impl<'a> DynamicNfaScanCache<'a> {
    #[inline]
    fn raw_config(state: u32) -> u32 {
        assert!(
            state < DYNAMIC_NFA_RAW_CONFIG_TAG,
            "dynamic lexer state exceeds raw-config tag coordinate"
        );
        DYNAMIC_NFA_RAW_CONFIG_TAG | state
    }

    #[inline]
    fn raw_config_state(config: u32) -> Option<u32> {
        (config & DYNAMIC_NFA_RAW_CONFIG_TAG != 0
            && config != DYNAMIC_NFA_CONFIG_UNKNOWN
            && config != DYNAMIC_NFA_CONFIG_DEAD)
            .then_some(config & !DYNAMIC_NFA_RAW_CONFIG_TAG)
    }

    fn new(constraint: &'a Constraint, deadline: Option<Instant>) -> Self {
        Self {
            constraint,
            deterministic: !constraint.tokenizer_has_epsilon_transitions,
            deadline,
            max_collection_items: deadline.map(|_| 5_000_000),
            config_ids: FxHashMap::default(),
            configs: Vec::new(),
            transitions: Vec::new(),
            residual_configs: Vec::new(),
            raw_start_config: FxHashMap::default(),
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
        if let [state] = states.as_slice()
            && !self.constraint.tokenizer.state_has_epsilon_transitions(*state)
        {
            return Ok(Self::raw_config(*state));
        }
        if let Some(&id) = self.config_ids.get(states.as_slice()) {
            return Ok(id);
        }
        self.check_growth(self.configs.len(), 1)?;
        let id = self.configs.len() as u32;
        debug_assert!(id < DYNAMIC_NFA_RAW_CONFIG_TAG);
        self.config_ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.transitions.push(None);
        self.residual_configs.push(DYNAMIC_NFA_CONFIG_UNKNOWN);
        Ok(id)
    }

    fn config_for_raw_start(&mut self, state: u32) -> Result<u32, String> {
        if self.deterministic {
            self.raw_start_config.entry(state).or_insert(state);
            return Ok(state);
        }
        if !self.constraint.tokenizer.state_has_epsilon_transitions(state) {
            return Ok(Self::raw_config(state));
        }
        if let Some(&cached) = self.raw_start_config.get(&state) {
            return Ok(cached);
        }
        let closure = self
            .constraint
            .tokenizer
            .singleton_epsilon_closure(state)
            .into_vec();
        let config = self.intern_config(closure)?;
        self.raw_start_config.insert(state, config);
        Ok(config)
    }

    fn config_for_raw_start_restricted(
        &mut self,
        state: u32,
        admitted: &BitSet,
        ignore_terminal: Option<TerminalID>,
    ) -> Result<Option<u32>, String> {
        let relevant = |tokenizer_state: u32| {
            let matched = self
                .constraint
                .tokenizer
                .matched_terminal_bitset(tokenizer_state);
            let future = self
                .constraint
                .tokenizer
                .possible_future_terminals(tokenizer_state);
            !admitted.is_disjoint(matched)
                || !admitted.is_disjoint(future)
                || ignore_terminal.is_some_and(|terminal| {
                    matched.contains(terminal as usize) || future.contains(terminal as usize)
                })
        };

        if self.deterministic {
            return Ok(relevant(state).then_some(state));
        }
        if !self.constraint.tokenizer.state_has_epsilon_transitions(state) {
            return Ok(relevant(state).then_some(Self::raw_config(state)));
        }
        let states = self
            .constraint
            .tokenizer
            .singleton_epsilon_closure(state)
            .iter()
            .copied()
            .filter(|&candidate| relevant(candidate))
            .collect::<Vec<_>>();
        if states.is_empty() {
            return Ok(None);
        }
        self.intern_config(states).map(Some)
    }

    fn step_config(&mut self, config: u32, byte: u8) -> Result<Option<u32>, String> {
        if self.deterministic {
            let target = self.constraint.tokenizer_fast_transitions.transition(
                &self.constraint.tokenizer,
                config,
                byte,
            );
            return Ok((target != u32::MAX).then_some(target));
        }
        if let Some(state) = Self::raw_config_state(config) {
            let target = self.constraint.tokenizer_fast_transitions.transition(
                &self.constraint.tokenizer,
                state,
                byte,
            );
            return if target == u32::MAX {
                Ok(None)
            } else {
                self.config_for_raw_start(target).map(Some)
            };
        }
        let config_index = config as usize;
        if let Some(row) = self.transitions[config_index].as_ref() {
            let cached = row[byte as usize];
            if cached != DYNAMIC_NFA_CONFIG_UNKNOWN {
                return Ok((cached != DYNAMIC_NFA_CONFIG_DEAD).then_some(cached));
            }
        }

        let closed_targets = {
            let mut targets = Vec::<u32>::new();
            let config_len = self.configs[config_index].len();
            for state_index in 0..config_len {
                let state = self.configs[config_index][state_index];
                let target = self.constraint.tokenizer_fast_transitions.transition(
                    &self.constraint.tokenizer,
                    state,
                    byte,
                );
                if target != u32::MAX {
                    let target_config = self.config_for_raw_start(target)?;
                    if let Some(target_state) = Self::raw_config_state(target_config) {
                        self.check_growth(targets.len(), 1)?;
                        targets.push(target_state);
                    } else {
                        let target_states = &self.configs[target_config as usize];
                        self.check_growth(targets.len(), target_states.len())?;
                        targets.extend_from_slice(target_states);
                    }
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

    fn residual_config(&mut self, config: u32) -> Result<Option<u32>, String> {
        if self.deterministic {
            return Ok((!self.constraint.tokenizer.is_end(config)).then_some(config));
        }
        if let Some(state) = Self::raw_config_state(config) {
            return Ok((!self.constraint.tokenizer.is_end(state)).then_some(config));
        }
        let config_index = config as usize;
        let cached = self.residual_configs[config_index];
        if cached != DYNAMIC_NFA_CONFIG_UNKNOWN {
            return Ok((cached != DYNAMIC_NFA_CONFIG_DEAD).then_some(cached));
        }

        let residual_states = self.configs[config_index]
            .iter()
            .copied()
            .filter(|&state| !self.constraint.tokenizer.is_end(state))
            .collect::<Vec<_>>();
        let residual = if residual_states.is_empty() {
            DYNAMIC_NFA_CONFIG_DEAD
        } else if residual_states.len() == self.configs[config_index].len() {
            config
        } else {
            self.intern_config(residual_states)?
        };
        self.residual_configs[config_index] = residual;
        Ok((residual != DYNAMIC_NFA_CONFIG_DEAD).then_some(residual))
    }

    fn execute_from_config_all_widths(
        &mut self,
        input: &[u8],
        start_config: u32,
    ) -> Result<DynamicConfigExecResult, String> {
        let mut config = start_config;
        let mut matches = Vec::new();
        for (index, &byte) in input.iter().enumerate() {
            let Some(next_config) = self.step_config(config, byte)? else {
                return Ok(DynamicConfigExecResult {
                    end_config: None,
                    matches,
                });
            };
            config = next_config;
            let width = index + 1;
            for state_index in 0..self.config_len(config) {
                let state = self.config_state(config, state_index);
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
        Ok(DynamicConfigExecResult {
            end_config: self.residual_config(config)?,
            matches,
        })
    }

    fn execute_from_state_all_widths(
        &mut self,
        input: &[u8],
        start: u32,
    ) -> Result<TokenizerExecResult, String> {
        let start_config = self.config_for_raw_start(start)?;
        let execution = self.execute_from_config_all_widths(input, start_config)?;
        let mut end_state = TokenizerStateSet::new();
        if let Some(end_config) = execution.end_config {
            for state_index in 0..self.config_len(end_config) {
                end_state.push(self.config_state(end_config, state_index));
            }
        }
        Ok(TokenizerExecResult {
            end_state,
            matches: execution.matches,
        })
    }

    #[inline]
    fn config_len(&self, config: u32) -> usize {
        if self.deterministic {
            1
        } else if Self::raw_config_state(config).is_some() {
            1
        } else {
            self.configs[config as usize].len()
        }
    }

    #[inline]
    fn config_state(&self, config: u32, index: usize) -> u32 {
        if self.deterministic {
            debug_assert_eq!(index, 0);
            config
        } else if let Some(state) = Self::raw_config_state(config) {
            debug_assert_eq!(index, 0);
            state
        } else {
            self.configs[config as usize][index]
        }
    }

    fn config_next_bytes(&self, config: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for state_index in 0..self.config_len(config) {
            let state = self.config_state(config, state_index);
            for (byte, _) in self.constraint.tokenizer.transitions_from(state) {
                bytes.insert(byte);
            }
        }
        bytes
    }

}

fn for_each_token_matching_terminal_from_state(
    constraint: &Constraint,
    start_state: u32,
    terminal: TerminalID,
    mut visit_token: impl FnMut(u32),
) -> Result<(), String> {
    if constraint.visit_possible_match_original_tokens(start_state, terminal, &mut visit_token) {
        return Ok(());
    }

    let vocab = constraint.dynamic_mask_vocab_for_runtime();
    let trie = vocab.trie.as_ref();
    let mut scan_cache = DynamicNfaScanCache::new(constraint, None);
    let start_config = scan_cache.config_for_raw_start(start_state)?;
    let mut work = vec![(0u32, start_config)];

    while let Some((node, config)) = work.pop() {
        for edge in trie.children(node) {
            let mut config = config;
            let mut alive = true;
            let mut matched = false;
            for &byte in trie.edge_bytes(edge) {
                let Some(next_config) = scan_cache.step_config(config, byte)? else {
                    alive = false;
                    break;
                };
                config = next_config;
                matched = (0..scan_cache.config_len(config)).any(|state_index| {
                    let state = scan_cache.config_state(config, state_index);
                    constraint
                        .tokenizer
                        .matched_terminals_iter(state)
                        .any(|matched_terminal| matched_terminal == terminal)
                });
                if matched {
                    break;
                }
            }

            if matched {
                for &canonical_token in trie.subtree_tokens(edge.child) {
                    if let Some(token_ids) = vocab.token_ids(canonical_token) {
                        for &token_id in token_ids {
                            visit_token(token_id);
                        }
                    }
                }
            } else if alive {
                work.push((edge.child, config));
            }
        }
    }

    Ok(())
}

pub(crate) fn or_blocked_internal_tokens_for_exclusions(
    constraint: &Constraint,
    exclusions: &TerminalsDisallowed,
    dense: &mut [u64],
) -> Result<(), String> {
    for (&lexer_state, terminals) in exclusions.iter() {
        for &terminal in terminals.iter() {
            for_each_token_matching_terminal_from_state(
                constraint,
                lexer_state,
                terminal,
                |token_id| {
                    if let Some(internal_token) =
                        constraint.final_internal_token_for_original(token_id)
                    {
                        let word = internal_token as usize / 64;
                        let bit = internal_token % 64;
                        if let Some(slot) = dense.get_mut(word) {
                            *slot |= 1u64 << bit;
                        }
                    }
                },
            )?;
        }
    }
    Ok(())
}

fn update_special_token_mask(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let mut previous_token_id = None;
    for special in &state.constraint.special_token_terminals {
        if state
            .constraint
            .is_late_grammar_placeholder_terminal(special.terminal_id)
        {
            continue;
        }
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

impl InitialPruneGuard {
    /// Build the token-start pruning state for one correlated tokenizer/GSS
    /// branch. Every remembered `(lexer state, terminal)` pair is an independent
    /// condition: a later match of that same terminal invalidates the
    /// provisional boundary that created this parser path.
    fn new(
        _constraint: &Constraint,
        _tokenizer_state: u32,
        _stacks: &ParserStacks,
        terminals_disallowed: &TerminalsDisallowed,
    ) -> Self {
        let mut memories = Vec::new();
        for (&lexer_state, terminals) in terminals_disallowed.iter() {
            for &terminal in terminals.iter() {
                memories.push((lexer_state, terminal));
            }
        }
        if memories.is_empty() {
            return Self::Passed;
        }
        memories.sort_unstable();
        memories.dedup();
        Self::Pending { memories: memories.into() }
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
        true
    }

    fn allows_token_bytes(&self, constraint: &Constraint, bytes: &[u8]) -> bool {
        self.advance(constraint, bytes).is_some()
    }

    fn blocked_output_mask(
        &self,
        constraint: &Constraint,
        mask_words: usize,
    ) -> Result<Option<Vec<u32>>, String> {
        let Self::Pending { memories } = self else {
            return Ok(None);
        };
        let mut blocked = vec![0u32; mask_words];
        for &(lexer_state, terminal) in memories.iter() {
            for_each_token_matching_terminal_from_state(
                constraint,
                lexer_state,
                terminal,
                |token_id| {
                    let word = token_id as usize / 32;
                    let bit = token_id % 32;
                    if let Some(slot) = blocked.get_mut(word) {
                        *slot |= 1u32 << bit;
                    }
                },
            )?;
        }
        Ok(Some(blocked))
    }

    fn remember_terminal_match(
        &self,
        constraint: &Constraint,
        lexer_state: u32,
        terminal: TerminalID,
    ) -> Self {
        if !constraint
            .tokenizer
            .possible_future_terminals(lexer_state)
            .contains(terminal as usize)
        {
            return self.clone();
        }

        let mut memories = match self {
            Self::Passed => Vec::new(),
            Self::Pending { memories } => memories.to_vec(),
        };
        memories.push((lexer_state, terminal));
        memories.sort_unstable();
        memories.dedup();
        Self::Pending {
            memories: memories.into(),
        }
    }

    /// Advance the original token-start lexer branch through a trie segment.
    /// Parser resets caused by terminal matches elsewhere in the dynamic walk
    /// deliberately do not affect this guard: commit evaluates its initial
    /// pruning predicate once, over the whole candidate token, before advancing
    /// the parser.
    fn advance(&self, constraint: &Constraint, segment: &[u8]) -> Option<Self> {
        let Self::Pending { memories } = self else {
            return Some(Self::Passed);
        };

        let mut next_memories = Vec::new();
        let mut index = 0usize;
        while index < memories.len() {
            let tokenizer_state = memories[index].0;
            let start = index;
            while index < memories.len() && memories[index].0 == tokenizer_state {
                index += 1;
            }
            let blocked = &memories[start..index];
            let execution = constraint
                .tokenizer
                .execute_from_state_all_widths(segment, tokenizer_state);
            for matched in &execution.matches {
                if blocked.iter().any(|&(_, terminal)| terminal == matched.id) {
                    return None;
                }
            }
            for end_state in execution.end_state {
                let future = constraint.tokenizer.possible_future_terminals(end_state);
                for &(_, terminal) in blocked {
                    if future.contains(terminal as usize) {
                        next_memories.push((end_state, terminal));
                    }
                }
            }
        }

        if next_memories.is_empty() {
            return Some(Self::Passed);
        }
        next_memories.sort_unstable();
        next_memories.dedup();
        Some(Self::Pending { memories: next_memories.into() })
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
    // The actual structural advance is already the definitive admissibility
    // test. Running exact admission first would duplicate reduction simulation
    // for every terminal branch explored by the dynamic traversal.
    let advanced = if let Some(advanced) =
        constraint.advance_compact_segmented_parser(&parser_gss, terminal)
    {
        advanced
    } else {
        constraint
            .direct_regular_cached_advance(&parser_gss, terminal)
            .or_else(|| super::commit::advance_stacks_template_dfa(constraint, &parser_gss, terminal))
            .unwrap_or_else(|| advance_stacks(&constraint.table, &parser_gss, terminal))
    }
    .apply(|_| ());
    (!advanced.is_empty()).then_some(advanced)
}

fn parser_child_cached(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
    cache: &mut DynamicTraversalCache,
) -> Option<ParserStacks> {
    let key = (parser_stacks_cache_key(stacks), terminal);
    if let Some((cached_stacks, result)) = cache.parser_children.get(&key) {
        debug_assert!(same_parser_stack_language(cached_stacks, stacks));
        let result = result.clone();
        cache.profile_event(1, u64::from(terminal), u64::from(result.is_some()));
        return result;
    }
    if cache.profile_interaction_hash.is_some()
        && !constraint.uses_compact_segmented_parser_runtime()
        && let Some(top) = stacks.single_top_value()
    {
        if !cache.profile_parser_child_terminals.contains(&terminal) {
            cache.profile_parser_child_terminals.push(terminal);
        }
        let bucket = match constraint.table.action(top, terminal) {
            None => 0,
            Some(Action::Shift(_, false)) => 1,
            Some(Action::Shift(_, true)) => 2,
            Some(Action::ReplaceShifts(_)) => 3,
            Some(Action::StackShifts(shifts)) if shifts.len() == 1 => 4,
            Some(Action::StackShifts(_)) => 5,
            Some(Action::GuardedStackShifts(_)) => 6,
            Some(Action::Reduce(..)) => 7,
            Some(Action::Split { .. }) => 8,
            Some(Action::Accept | Action::Skip) => 9,
        };
        cache.profile_parser_action_counts[bucket] += 1;
    }
    let result = parser_child(constraint, stacks, terminal);
    cache
        .parser_children
        .insert(key, (stacks.clone(), result.clone()));
    cache.profile_event(1, u64::from(terminal), u64::from(result.is_some()));
    result
}

fn token_boundary_allowed(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
) -> bool {
    let accessible = constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state);
    if constraint
        .ignore_terminal
        .is_some_and(|terminal| accessible.contains(terminal as usize))
    {
        return true;
    }
    let parser_gss = with_empty_accumulators(stacks);
    constraint
        .compact_segmented_parser_may_advance_on_any(&parser_gss, accessible)
        .or_else(|| constraint.direct_regular_may_advance_on_any(&parser_gss, accessible))
        .unwrap_or_else(|| stack_may_advance_on_any(&constraint.table, &parser_gss, accessible))
}

fn admissible_terminals_cached<'a>(
    constraint: &Constraint,
    stacks: &ParserStacks,
    cache: &'a mut DynamicTraversalCache,
) -> &'a BitSet {
    let key = parser_stacks_cache_key(stacks);
    if !cache.admissible_terminals.contains_key(&key) {
        let parser_gss = with_empty_accumulators(stacks);
        let admitted = if constraint.uses_compact_segmented_parser_runtime() {
            let mut admitted = BitSet::new(constraint.table.num_terminals as usize);
            for terminal in 0..constraint.table.num_terminals {
                if constraint
                    .compact_segmented_parser_may_advance_on(&parser_gss, terminal)
                    .unwrap_or(false)
                {
                    admitted.set(terminal as usize);
                }
            }
            admitted
        } else {
            constraint
                .direct_regular_admissible_terminals(&parser_gss)
                .unwrap_or_else(|| {
                    let candidates = BitSet::all(constraint.table.num_terminals as usize);
                    stack_admissible_terminals(&constraint.table, &parser_gss, &candidates)
                })
        };
        cache
            .admissible_terminals
            .insert(key, (stacks.clone(), admitted));
    }
    let (cached_stacks, admitted) = cache
        .admissible_terminals
        .get(&key)
        .expect("admissible terminal cache insertion must be visible");
    debug_assert!(same_parser_stack_language(cached_stacks, stacks));
    admitted
}

#[inline]
fn parser_terminal_admissible_cached(
    constraint: &Constraint,
    terminal: TerminalID,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    if Some(terminal) == constraint.ignore_terminal {
        cache.profile_event(2, u64::from(terminal), 1);
        return true;
    }
    let stack_key = parser_stacks_cache_key(stacks);
    let key = (stack_key, terminal);
    if let Some(&result) = cache.terminal_admissible.get(&key) {
        cache.profile_event(2, u64::from(terminal), u64::from(result));
        return result;
    }
    if let Some((cached_stacks, admitted)) = cache.admissible_terminals.get(&stack_key) {
        debug_assert!(same_parser_stack_language(cached_stacks, stacks));
        let result = admitted.contains(terminal as usize);
        cache.terminal_admissible.insert(key, result);
        cache.profile_event(2, u64::from(terminal), u64::from(result));
        return result;
    }
    let parser_gss = with_empty_accumulators(stacks);
    let result = constraint
        .compact_segmented_parser_may_advance_on(&parser_gss, terminal)
        .or_else(|| constraint.direct_regular_may_advance_on(&parser_gss, terminal))
        .unwrap_or_else(|| {
            admissible_terminals_cached(constraint, stacks, cache).contains(terminal as usize)
        });
    cache.terminal_admissible.insert(key, result);
    cache.profile_event(2, u64::from(terminal), u64::from(result));
    result
}

fn token_boundary_allowed_cached(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    let accessible = constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state);
    let ignore_relevant = constraint
        .ignore_terminal
        .is_some_and(|terminal| accessible.contains(terminal as usize));
    let result = if ignore_relevant {
        true
    } else if accessible.count_ones() <= 8 {
        accessible
            .iter()
            .any(|terminal| parser_terminal_admissible_cached(
                constraint,
                terminal as TerminalID,
                stacks,
                cache,
            ))
    } else {
        !admissible_terminals_cached(constraint, stacks, cache).is_disjoint(accessible)
    };
    cache.profile_event(4, u64::from(tokenizer_state), u64::from(result));
    result
}

fn config_token_boundary_allowed_cached(
    constraint: &Constraint,
    scan_cache: &DynamicNfaScanCache<'_>,
    tokenizer_config: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    (0..scan_cache.config_len(tokenizer_config)).any(|state_index| {
        let tokenizer_state = scan_cache.config_state(tokenizer_config, state_index);
            token_boundary_allowed_cached(constraint, tokenizer_state, stacks, cache)
    })
}

fn lexer_state_relevant_cached(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    let stack_key = parser_stacks_cache_key(stacks);
    let key = (tokenizer_state, stack_key);
    if let Some(&result) = cache.lexer_relevant.get(&key) {
        cache.profile_event(3, u64::from(tokenizer_state), u64::from(result));
        return result;
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
        let mut candidates = accessible.clone();
        candidates.union_with(matched);
        if candidates.count_ones() <= 8 {
            candidates.iter().any(|terminal| {
                parser_terminal_admissible_cached(
                    constraint,
                    terminal as TerminalID,
                    stacks,
                    cache,
                )
            })
        } else {
            !admissible_terminals_cached(constraint, stacks, cache).is_disjoint(&candidates)
        }
    };
    cache.lexer_relevant.insert(key, result);
    cache.profile_event(3, u64::from(tokenizer_state), u64::from(result));
    result
}

fn lexer_config_relevant_cached(
    constraint: &Constraint,
    scan_cache: &DynamicNfaScanCache<'_>,
    tokenizer_config: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    (0..scan_cache.config_len(tokenizer_config)).any(|state_index| {
        let tokenizer_state = scan_cache.config_state(tokenizer_config, state_index);
            lexer_state_relevant_cached(constraint, tokenizer_state, stacks, cache)
    })
}

#[inline]
fn mark_subtree_tokens(
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    node: u32,
    buf: &mut [u32],
) {
    for &token_id in vocab.subtree_original_tokens(node) {
        set_mask_bit_known_in_range(buf, token_id);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DynamicOracleNodeClass {
    Disallowed,
    Allowed,
    Mixed,
}

#[inline]
fn dynamic_oracle_token_allowed(vocab: &DynamicMaskVocab, node: u32, mask: &[u32]) -> Option<bool> {
    let marker = vocab.node_token_marker(node);
    if marker == 0 {
        return None;
    }
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        let word = (marker >> 32) as usize;
        let bits = marker as u32;
        let present = mask.get(word).copied().unwrap_or(0) & bits;
        debug_assert!(present == 0 || present == bits);
        return Some(present == bits);
    }

    let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
    let token_ids = vocab
        .token_ids(canonical_token)
        .expect("dynamic vocabulary trie node lacks token ids");
    let mut first = None;
    for &token_id in token_ids {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        let allowed = mask
            .get(word)
            .is_some_and(|word| word & (1u32 << bit) != 0);
        debug_assert!(first.is_none_or(|value| value == allowed));
        first = Some(allowed);
    }
    first
}

/// Diagnostic lower bound for trie-based dynamic masking.  Given the already
/// computed exact mask, classify every vocabulary-trie node as wholly allowed,
/// wholly disallowed, or mixed.  Then time rebuilding the mask from those
/// precomputed oracle labels.  Classification is intentionally outside the
/// timed region: this answers how cheap runtime could be with perfect static
/// subtree knowledge, not how to obtain that knowledge.
fn profile_dynamic_oracle_cover(
    generation: u64,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    exact_mask: &[u32],
) {
    let node_count = trie.node_count();
    if node_count == 0 {
        return;
    }

    let mut classes = vec![DynamicOracleNodeClass::Disallowed; node_count];
    let mut token_allowed = vec![0u8; node_count];
    for node_index in (0..node_count).rev() {
        let node = node_index as u32;
        let token = dynamic_oracle_token_allowed(vocab, node, exact_mask);
        token_allowed[node_index] = u8::from(token == Some(true));

        let mut class = token.map(|allowed| {
            if allowed {
                DynamicOracleNodeClass::Allowed
            } else {
                DynamicOracleNodeClass::Disallowed
            }
        });
        for edge in trie.children(node) {
            let child = classes[edge.child as usize];
            class = Some(match class {
                None => child,
                Some(current) if current == child => current,
                Some(_) => DynamicOracleNodeClass::Mixed,
            });
            if class == Some(DynamicOracleNodeClass::Mixed) {
                // The node can never become homogeneous again once two local
                // outcomes differ, so there is no need to inspect more child
                // labels for classification.
                break;
            }
        }
        classes[node_index] = class.unwrap_or(DynamicOracleNodeClass::Disallowed);
    }

    let mut rebuilt = vec![0u32; exact_mask.len()];
    let mut stack = Vec::<u32>::with_capacity(128);
    stack.push(0);
    let started = Instant::now();
    let mut visited = 0usize;
    let mut allowed_subtrees = 0usize;
    let mut allowed_subtree_tokens = 0usize;
    let mut mixed_nodes = 0usize;
    while let Some(node) = stack.pop() {
        visited += 1;
        match classes[node as usize] {
            DynamicOracleNodeClass::Allowed => {
                allowed_subtrees += 1;
                allowed_subtree_tokens += trie.subtree_tokens(node).len();
                mark_subtree_tokens(vocab, trie, node, &mut rebuilt);
            }
            DynamicOracleNodeClass::Disallowed => {}
            DynamicOracleNodeClass::Mixed => {
                mixed_nodes += 1;
                if token_allowed[node as usize] != 0 {
                    let marker = vocab.node_token_marker(node);
                    debug_assert_ne!(marker, 0);
                    mark_dynamic_token_marker(vocab, marker, &mut rebuilt);
                }
                for edge in trie.children(node).iter().rev() {
                    stack.push(edge.child);
                }
            }
        }
    }
    let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    debug_assert_eq!(rebuilt, exact_mask);
    eprintln!(
        "[glrmask/profile][dynamic_oracle_cover] generation={} ns={} visited_nodes={} mixed_nodes={} allowed_subtrees={} allowed_subtree_tokens={} trie_nodes={}",
        generation,
        elapsed_ns,
        visited,
        mixed_nodes,
        allowed_subtrees,
        allowed_subtree_tokens,
        node_count,
    );
}

enum RawSelfLoopSubtree {
    CannotSkip,
    MarkAllTokens,
}

fn dynamic_subtree_loop_min_tokens() -> usize {
    static MIN_TOKENS: OnceLock<usize> = OnceLock::new();
    *MIN_TOKENS.get_or_init(|| {
        std::env::var("GLRMASK_DYNAMIC_SUBTREE_MIN_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value >= 2)
            .unwrap_or(2)
    })
}

fn dynamic_multi_branch_subtree_loop_min_tokens() -> usize {
    static MIN_TOKENS: OnceLock<usize> = OnceLock::new();
    *MIN_TOKENS.get_or_init(|| {
        std::env::var("GLRMASK_DYNAMIC_MULTI_BRANCH_SUBTREE_MIN_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value >= 2)
            .unwrap_or(usize::MAX)
    })
}

fn dynamic_bounded_subtree_min_tokens() -> usize {
    static MIN_TOKENS: OnceLock<usize> = OnceLock::new();
    *MIN_TOKENS.get_or_init(|| {
        std::env::var("GLRMASK_DYNAMIC_BOUNDED_SUBTREE_MIN_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value >= 2)
            .unwrap_or(1_024)
    })
}

fn dynamic_projection_reentry_min_tokens() -> usize {
    static MIN_TOKENS: OnceLock<usize> = OnceLock::new();
    *MIN_TOKENS.get_or_init(|| {
        std::env::var("GLRMASK_DYNAMIC_PROJECTION_REENTRY_MIN_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value >= 2)
            .unwrap_or(8)
    })
}

struct DynamicDeadlinePoll {
    deadline: Option<Instant>,
    remaining: u16,
}

#[derive(Default)]
struct DynamicWalkStats {
    work_items: usize,
    trie_edges: usize,
    branch_steps: usize,
    duplicate_branches: usize,
    max_branches: usize,
    subtree_marks: usize,
    subtree_mark_tokens: usize,
    bounded_subtree_attempts: usize,
    bounded_subtree_marks: usize,
    projection_reentry_marks: usize,
    projection_reentry_tokens: usize,
    config_projection_marks: usize,
    config_projection_tokens: usize,
    config_projection_node_hits: usize,
    config_projection_candidate_edges: usize,
    config_projection_step_dead: usize,
    config_projection_normalize_dead: usize,
    recognizer_simple_one_states: usize,
    recognizer_simple_two_states: usize,
    recognizer_other_states: usize,
    config_projection_guard_rejects: usize,
    config_projection_config_matches: usize,
    config_projection_parser_rejects: usize,
    recognizer_states: usize,
    recognizer_transition_misses: usize,
}


impl DynamicWalkStats {
    fn merge_from(&mut self, other: &Self) {
        self.work_items += other.work_items;
        self.trie_edges += other.trie_edges;
        self.branch_steps += other.branch_steps;
        self.duplicate_branches += other.duplicate_branches;
        self.max_branches = self.max_branches.max(other.max_branches);
        self.subtree_marks += other.subtree_marks;
        self.subtree_mark_tokens += other.subtree_mark_tokens;
        self.bounded_subtree_attempts += other.bounded_subtree_attempts;
        self.bounded_subtree_marks += other.bounded_subtree_marks;
        self.projection_reentry_marks += other.projection_reentry_marks;
        self.projection_reentry_tokens += other.projection_reentry_tokens;
        self.config_projection_marks += other.config_projection_marks;
        self.config_projection_tokens += other.config_projection_tokens;
        self.config_projection_node_hits += other.config_projection_node_hits;
        self.config_projection_candidate_edges += other.config_projection_candidate_edges;
        self.config_projection_step_dead += other.config_projection_step_dead;
        self.config_projection_normalize_dead += other.config_projection_normalize_dead;
        self.recognizer_simple_one_states += other.recognizer_simple_one_states;
        self.recognizer_simple_two_states += other.recognizer_simple_two_states;
        self.recognizer_other_states += other.recognizer_other_states;
        self.config_projection_guard_rejects += other.config_projection_guard_rejects;
        self.config_projection_config_matches += other.config_projection_config_matches;
        self.config_projection_parser_rejects += other.config_projection_parser_rejects;
        self.recognizer_states += other.recognizer_states;
        self.recognizer_transition_misses += other.recognizer_transition_misses;
    }
}

impl DynamicDeadlinePoll {
    fn new(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            remaining: 0,
        }
    }

    #[inline]
    fn check(&mut self) -> Result<(), String> {
        let Some(deadline) = self.deadline else {
            return Ok(());
        };
        if self.remaining != 0 {
            self.remaining -= 1;
            return Ok(());
        }
        self.remaining = 1_023;
        if Instant::now() >= deadline {
            Err("glrmask_dynamic mask generation timed out".to_owned())
        } else {
            Ok(())
        }
    }
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

fn cached_config_self_loop_bytes(
    constraint: &Constraint,
    scan_cache: &DynamicNfaScanCache<'_>,
    tokenizer_config: u32,
    raw_cache: &mut FxHashMap<u32, U8Set>,
    config_cache: &mut FxHashMap<u32, U8Set>,
) -> U8Set {
    if let Some(&cached) = config_cache.get(&tokenizer_config) {
        return cached;
    }
    let mut bytes = U8Set::all();
    for state_index in 0..scan_cache.config_len(tokenizer_config) {
        let tokenizer_state = scan_cache.config_state(tokenizer_config, state_index);
        bytes &= cached_self_loop_bytes(constraint, tokenizer_state, raw_cache);
    }
    config_cache.insert(tokenizer_config, bytes);
    bytes
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
    scan_cache: &DynamicNfaScanCache<'_>,
    tokenizer_config: u32,
    stacks: &ParserStacks,
    initial_prune_guard: &InitialPruneGuard,
    initial_config: u32,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    traversal_cache: &mut DynamicTraversalCache,
) -> RawSelfLoopSubtree {
    if !initial_prune_guard.is_passed() {
        return RawSelfLoopSubtree::CannotSkip;
    }

    // A one-token subtree has no traversal work to eliminate. Running the
    // self-loop certificate there merely replaces the ordinary boundary check
    // with multiple hash lookups and a 256-bit subset test. Large vocabularies
    // contain tens of thousands of such leaves, so reserve this shortcut for
    // actual multi-token subtrees.
    if trie.subtree_tokens(node).len() < dynamic_subtree_loop_min_tokens() {
        return RawSelfLoopSubtree::CannotSkip;
    }

    // Work at the initial state may represent either an untouched lexer or a
    // lexer reset after an in-token terminal match. The current work item does
    // not distinguish those cases, so keep this optimization conservative.
    if tokenizer_config == initial_config {
        return RawSelfLoopSubtree::CannotSkip;
    }

    let subtree_bytes = U8Set::from_words(trie.subtree_bytes(node));
    let loop_bytes = cached_config_self_loop_bytes(
        constraint,
        scan_cache,
        tokenizer_config,
        raw_self_loop_cache,
        config_self_loop_cache,
    );
    if !subtree_bytes.is_subset(&loop_bytes)
        || !config_token_boundary_allowed_cached(
            constraint,
            scan_cache,
            tokenizer_config,
            stacks,
            traversal_cache,
        )
    {
        return RawSelfLoopSubtree::CannotSkip;
    }

    RawSelfLoopSubtree::MarkAllTokens
}

#[allow(clippy::too_many_arguments)]
fn process_dynamic_trie_node(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    node_id: u32,
    branches: &DynamicBranches,
    initial_config: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    traversal_cache: &mut DynamicTraversalCache,
    buf: &mut [u32],
    subtree_marks: &mut usize,
    subtree_mark_tokens: &mut usize,
) -> bool {
    for branch in branches {
        if matches!(
            raw_self_loop_subtree(
                state.constraint,
                trie,
                node_id,
                lexer_scan_cache,
                branch.tokenizer_config,
                &branch.gss,
                &branch.initial_prune_guard,
                initial_config,
                raw_self_loop_cache,
                config_self_loop_cache,
                traversal_cache,
            ),
            RawSelfLoopSubtree::MarkAllTokens
        ) {
            *subtree_marks += 1;
            *subtree_mark_tokens += trie.subtree_tokens(node_id).len();
            mark_subtree_tokens(vocab, trie, node_id, buf);
            return true;
        }
    }

    let token_marker = vocab.node_token_marker(node_id);
    if token_marker == 0 {
        return false;
    }
    let allowed = branches.iter().any(|branch| {
        branch.initial_prune_guard.allows_token_boundary()
            && (branch.tokenizer_config == initial_config
                || config_token_boundary_allowed_cached(
                    state.constraint,
                    lexer_scan_cache,
                    branch.tokenizer_config,
                    &branch.gss,
                    traversal_cache,
                ))
    });
    if allowed {
        mark_dynamic_token_marker(vocab, token_marker, buf);
    }
    false
}

/// Finite-horizon generalization of the literal raw-state self-loop check.
///
/// Constraint finalization precomputes, for every deterministic tokenizer
/// state, one canonical byte alphabet safe for 16 bytes and one safe for 64
/// bytes.  If every byte below a vocabulary subtree is in the appropriate set,
/// every mixed byte string up to the subtree's maximum remaining length keeps
/// the complete lexer finalizer/future observation unchanged.
fn bounded_observation_branch_certificate(
    vocab: &DynamicMaskVocab,
    constraint: &Constraint,
    branch: &DynamicBranch,
    initial_config: u32,
    node_id: u32,
    subtree_token_count: usize,
    subtree_bytes: U8Set,
    horizon: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
) -> bool {
    let detail = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_BOUNDED_DETAIL").is_some();
    let byte_count = subtree_bytes.iter().count();
    macro_rules! reject {
        ($reason:expr) => {{
            if detail {
                eprintln!(
                    "[glrmask/profile][dynamic_bounded_detail] node={} tokens={} bytes={} horizon={} result=reject reason={}",
                    node_id, subtree_token_count, byte_count, horizon, $reason
                );
            }
            return false;
        }};
    }
    if branch.tokenizer_config == initial_config
    {
        reject!("initial_config");
    }
    if branch.fresh_reset {
        reject!("fresh_reset");
    }
    if !branch.pending_terminals.is_empty() {
        reject!("pending_terminals");
    }
    if !branch.initial_prune_guard.is_passed() {
        reject!("prune_guard");
    }
    if subtree_bytes.is_empty() {
        reject!("empty_bytes");
    }
    if horizon == 0 {
        reject!("zero_horizon");
    }
    if horizon > 64 {
        reject!("horizon_gt_64");
    }
    if lexer_scan_cache.config_len(branch.tokenizer_config) != 1 {
        reject!("config_not_scalar");
    }

    if !config_token_boundary_allowed_cached(
        constraint,
        lexer_scan_cache,
        branch.tokenizer_config,
        &branch.gss,
        traversal_cache,
    ) {
        reject!("token_boundary");
    }

    let source = lexer_scan_cache.config_state(branch.tokenizer_config, 0);
    if constraint.tokenizer.state_has_epsilon_transitions(source) {
        reject!("source_epsilon");
    }

    let Some(safe_bytes) = vocab.bounded_observation_safe_bytes(source, horizon) else {
        reject!("no_precomputed_horizon");
    };
    let certified = subtree_bytes.is_subset(&safe_bytes);
    if detail {
        let exact_full_horizon = (!certified
            && std::env::var_os("GLRMASK_PROFILE_DYNAMIC_BOUNDED_EXACT").is_some()
            && subtree_token_count >= 10_000)
            .then(|| {
                let all_terminals = BitSet::all(constraint.tokenizer.num_terminals() as usize);
                constraint
                    .tokenizer
                    .bounded_observation_safe_horizon_from_state(
                        source,
                        subtree_bytes,
                        &all_terminals,
                        64,
                    )
            });
        let missing = subtree_bytes
            .iter()
            .filter(|&byte| !safe_bytes.contains(byte))
            .collect::<Vec<_>>();
        eprintln!(
            "[glrmask/profile][dynamic_bounded_detail] node={} tokens={} bytes={} horizon={} source={} precomputed_safe_bytes={} exact_full_horizon={:?} missing={:?} result={}",
            node_id,
            subtree_token_count,
            byte_count,
            horizon,
            source,
            safe_bytes.len(),
            exact_full_horizon,
            missing,
            if certified { "accept" } else { "reject" }
        );
    }
    certified
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn process_interned_dynamic_trie_node(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    node_id: u32,
    recognizer: &mut DynamicRecognizerStateCache,
    recognizer_state: u32,
    initial_config: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    traversal_cache: &mut DynamicTraversalCache,
    buf: &mut [u32],
    stats: &mut DynamicWalkStats,
    require_repair_used: bool,
) -> bool {
    let branch_count = recognizer.branches(recognizer_state).len();
    let subtree_tokens = trie.subtree_tokens(node_id);

    // A precomputed trie-aware projection can be reused after the walk has
    // entered its source state, provided projection construction reached this
    // same trie node in that source state too. This handles structured byte
    // sequences (notably UTF-8) that a byte-union certificate must reject.
    // `source_reentry_safe_subtrees` is deliberately stricter than the root
    // projection's ordinary `safe_subtrees`: it makes the suffix-language
    // reuse independent of how the runtime reached this node.
    if !require_repair_used && subtree_tokens.len() >= dynamic_projection_reentry_min_tokens() {
        for branch in recognizer.branches(recognizer_state) {
        if !branch.fresh_reset
            && branch.pending_terminals.is_empty()
            && branch.initial_prune_guard.is_passed()
            && lexer_scan_cache.config_len(branch.tokenizer_config) == 1
        {
            let raw_state = lexer_scan_cache.config_state(branch.tokenizer_config, 0);
            if let Some(projection) = vocab.self_loop_projection(raw_state)
                && projection.subtree_is_safe_from_source(node_id)
                && projection.future_terminals.iter().copied().any(|terminal| {
                    parser_terminal_admissible_cached(
                        state.constraint,
                        terminal,
                        &branch.gss,
                        traversal_cache,
                    )
                })
            {
                stats.subtree_marks += 1;
                stats.subtree_mark_tokens += subtree_tokens.len();
                stats.projection_reentry_marks += 1;
                stats.projection_reentry_tokens += subtree_tokens.len();
                mark_subtree_tokens(vocab, trie, node_id, buf);
                return true;
            }
        }
        }
    }

    let bounded_min_tokens = dynamic_bounded_subtree_min_tokens();
    // Projection reuse is attempted immediately above and returns on success.
    // The bounded check is now only a precomputed set lookup/subset test, so a
    // projection that does not cover this node must not globally suppress it.
    // (The older lazy bounded proof was expensive enough that the two paths had
    // to be mutually exclusive.)
    let bounded_branches = (subtree_tokens.len() >= bounded_min_tokens)
        .then(|| recognizer.branches(recognizer_state).clone());

    // Bounded observation equivalence strictly generalizes literal raw-state
    // self-loops for finite vocabulary subtrees. Try it first on large normal
    // subtrees even when some exact loop bytes exist: a bounded-repeat lexer
    // commonly has many literal loops *and* a counter-advancing family of
    // observation-equivalent transitions, and the latter is what lets us skip
    // the whole partition.
    if !require_repair_used {
        if let Some(bounded_branches) = bounded_branches.as_ref() {
            let subtree_bytes = U8Set::from_words(trie.subtree_bytes(node_id));
            let horizon = trie.subtree_max_byte_len(node_id);
            stats.bounded_subtree_attempts += 1;
            let certified = bounded_branches.iter().any(|branch| {
                bounded_observation_branch_certificate(
                    vocab,
                    state.constraint,
                    branch,
                    initial_config,
                    node_id,
                    subtree_tokens.len(),
                    subtree_bytes,
                    horizon,
                    lexer_scan_cache,
                    traversal_cache,
                )
            });
            if certified {
                stats.subtree_marks += 1;
                stats.subtree_mark_tokens += subtree_tokens.len();
                stats.bounded_subtree_marks += 1;
                mark_subtree_tokens(vocab, trie, node_id, buf);
                return true;
            }
        }
    }

    let metadata = recognizer.metadata(
        recognizer_state,
        state.constraint,
        initial_config,
        lexer_scan_cache,
        raw_self_loop_cache,
        config_self_loop_cache,
        traversal_cache,
    );

    if require_repair_used {
        let subtree_tokens = trie.subtree_tokens(node_id);
        let min_subtree_tokens = if branch_count == 1 {
            dynamic_subtree_loop_min_tokens()
        } else {
            dynamic_multi_branch_subtree_loop_min_tokens()
        };
        if subtree_tokens.len() >= min_subtree_tokens
            && !metadata.repair_subtree_loop_bytes.is_empty()
        {
            let subtree_bytes = U8Set::from_words(trie.subtree_bytes(node_id));
            if metadata
                .repair_subtree_loop_bytes
                .iter()
                .any(|loop_bytes| subtree_bytes.is_subset(loop_bytes))
            {
                stats.subtree_marks += 1;
                stats.subtree_mark_tokens += subtree_tokens.len();
                mark_subtree_tokens(vocab, trie, node_id, buf);
                return true;
            }
        }
        if metadata.repair_token_boundary_allowed {
            let token_marker = vocab.node_token_marker(node_id);
            if token_marker != 0 {
                mark_dynamic_token_marker(vocab, token_marker, buf);
            }
        }
        return false;
    }

    // Most recognizer states have no exact self-loop certificate. Keep that
    // overwhelmingly common node path independent of subtree metadata: only a
    // token-bearing node can change the mask, and non-token radix nodes need
    // no work at all.
    if metadata.subtree_loop_bytes.is_empty() {
        if metadata.token_boundary_allowed {
            let token_marker = vocab.node_token_marker(node_id);
            if token_marker != 0 {
                mark_dynamic_token_marker(vocab, token_marker, buf);
            }
        }
        return false;
    }

    process_interned_dynamic_trie_node_with_loops(
        state,
        vocab,
        trie,
        node_id,
        branch_count,
        metadata,
        buf,
        stats,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn process_interned_dynamic_trie_node_with_loops(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    node_id: u32,
    branch_count: usize,
    metadata: &DynamicRecognizerStateMetadata,
    buf: &mut [u32],
    stats: &mut DynamicWalkStats,
) -> bool {
    let subtree_tokens = trie.subtree_tokens(node_id);
    let min_subtree_tokens = if branch_count == 1 {
        dynamic_subtree_loop_min_tokens()
    } else {
        dynamic_multi_branch_subtree_loop_min_tokens()
    };
    if subtree_tokens.len() >= min_subtree_tokens {
        let subtree_bytes = U8Set::from_words(trie.subtree_bytes(node_id));
        if metadata
            .subtree_loop_bytes
            .iter()
            .any(|loop_bytes| subtree_bytes.is_subset(loop_bytes))
        {
            stats.subtree_marks += 1;
            stats.subtree_mark_tokens += subtree_tokens.len();
            mark_subtree_tokens(vocab, trie, node_id, buf);
            return true;
        }
    }

    if metadata.token_boundary_allowed {
        let token_marker = vocab.node_token_marker(node_id);
        if token_marker != 0 {
            mark_dynamic_token_marker(vocab, token_marker, buf);
        }
    }
    false
}

const DYNAMIC_MASK_CACHE_MAX_STACKS: usize = 4_096;
const DYNAMIC_MASK_CACHE_MAX_DEPTH: u32 = 256;

fn dynamic_mask_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("GLRMASK_DISABLE_DYNAMIC_MASK_CACHE").is_none())
}

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
                    .iter()
                    .map(|(excluded_state, terminals)| {
                        (*excluded_state, terminals.iter().copied().collect::<Vec<_>>())
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
    fill_mask_dynamic_impl(state, buf, None, false, None)
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
        false,
        None,
    )
}


/// OR only the dynamic language contribution not already covered by a static
/// mask. The baseline is used solely as a trie-pruning certificate; all
/// surviving leaves are still validated by the ordinary exact dynamic walker.
pub(crate) fn or_mask_dynamic_additions(state: &ConstraintState<'_>, buf: &mut [u32]) {
    fill_mask_dynamic_impl(state, buf, None, true, None)
        .expect("unbounded additive dynamic mask generation cannot time out");
}

/// Validate only the requested additive candidates with the exact dynamic
/// recognizer. `buf` is both the static baseline and the output; additions not
/// present in `candidate_mask` are discarded even if an internal subtree
/// certificate marked them while validating a candidate-bearing subtree.
pub(crate) fn or_mask_dynamic_candidate_additions(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    candidate_mask: &[u32],
) {
    fill_mask_dynamic_impl(state, buf, None, true, Some(candidate_mask))
        .expect("unbounded candidate dynamic mask generation cannot time out");
}

#[inline]
fn same_parser_stack_language(left: &ParserStacks, right: &ParserStacks) -> bool {
    left.ptr_eq(right)
        || left
            .single_interface_lower_id()
            .zip(right.single_interface_lower_id())
            .is_some_and(|(left, right)| left == right)
}

fn push_unique_dynamic_branch(
    branches: &mut DynamicBranches,
    branch: DynamicBranch,
) -> bool {
    branches.push(branch);
    true
}

fn try_advance_scalar_branch_over_segment(
    constraint: &Constraint,
    input: &DynamicBranches,
    segment: &[u8],
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
    branch_steps: &mut usize,
) -> Result<Option<DynamicBranches>, String> {
    let [branch] = input.as_slice() else {
        return Ok(None);
    };
    if branch.fresh_reset
        || !matches!(branch.initial_prune_guard, InitialPruneGuard::Passed)
    {
        return Ok(None);
    }

    let mut tokenizer_config = branch.tokenizer_config;
    for &byte in segment {
        *branch_steps += 1;
        let Some(next_config) = lexer_scan_cache.step_config(tokenizer_config, byte)? else {
            return Ok(Some(DynamicBranches::new()));
        };
        for state_index in 0..lexer_scan_cache.config_len(next_config) {
            let state = lexer_scan_cache.config_state(next_config, state_index);
            for terminal in constraint.tokenizer.matched_terminals_iter(state) {
                if parser_terminal_admissible_cached(
                    constraint,
                    terminal,
                    &branch.gss,
                    traversal_cache,
                ) {
                    // A parser-actionable match forks a reset branch from this
                    // byte. Re-run the uncommon edge through the exact general
                    // branch engine rather than burdening the scalar loop.
                    return Ok(None);
                }
            }
        }
        tokenizer_config = next_config;
    }

    let Some(tokenizer_config) = lexer_scan_cache.residual_config(tokenizer_config)? else {
        return Ok(Some(DynamicBranches::new()));
    };
    if !lexer_config_relevant_cached(
        constraint,
        lexer_scan_cache,
        tokenizer_config,
        &branch.gss,
        traversal_cache,
    ) {
        return Ok(Some(DynamicBranches::new()));
    }

    let mut result = DynamicBranches::new();
    result.push(DynamicBranch {
        tokenizer_config,
        gss: branch.gss.clone(),
        initial_prune_guard: InitialPruneGuard::Passed,
        last_component: branch.last_component,
        repair_used: branch.repair_used,
        pending_terminals: branch.pending_terminals.clone(),
        fresh_reset: false,
    });
    Ok(Some(result))
}

fn advance_dynamic_branches_over_segment(
    constraint: &Constraint,
    input: &DynamicBranches,
    segment: &[u8],
    initial_config: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
    branch_steps: &mut usize,
    duplicate_branches: &mut usize,
    max_branches: &mut usize,
) -> Result<DynamicBranches, String> {
    if let Some(result) = try_advance_scalar_branch_over_segment(
        constraint,
        input,
        segment,
        lexer_scan_cache,
        traversal_cache,
        branch_steps,
    )? {
        *max_branches = (*max_branches).max(result.len());
        return Ok(result);
    }

    let mut current = input.clone();
    let mut next = DynamicBranches::new();

    for &byte in segment {
        next.clear();
        for branch in current.drain(..) {
            *branch_steps += 1;
            let Some(next_config) = lexer_scan_cache.step_config(branch.tokenizer_config, byte)?
            else {
                continue;
            };
            let Some(advanced_prune_guard) = branch
                .initial_prune_guard
                .advance(constraint, std::slice::from_ref(&byte))
            else {
                continue;
            };

            let config_len = lexer_scan_cache.config_len(next_config);
            for state_index in 0..config_len {
                let matched_state = lexer_scan_cache.config_state(next_config, state_index);
                for terminal in constraint.tokenizer.matched_terminals_iter(matched_state) {
                    let matched_prune_guard = if Some(terminal) == constraint.ignore_terminal {
                        advanced_prune_guard.clone()
                    } else {
                        advanced_prune_guard.remember_terminal_match(
                            constraint,
                            matched_state,
                            terminal,
                        )
                    };
                    let (last_component, repair_used) =
                        overlay_advance_repair(constraint, &branch, terminal);
                    let child = if lazy_repair_parser_enabled(constraint, &branch)
                        && !repair_used
                        && branch.pending_terminals.is_empty()
                    {
                        let mut pending_terminals = branch.pending_terminals.clone();
                        pending_terminals.push(terminal);
                        Some(DynamicBranch {
                            tokenizer_config: initial_config,
                            gss: branch.gss.clone(),
                            initial_prune_guard: matched_prune_guard,
                            last_component,
                            repair_used: false,
                            pending_terminals,
                            fresh_reset: true,
                        })
                    } else {
                        let advanced_parser = if lazy_repair_parser_enabled(constraint, &branch) {
                            replay_pending_with_terminal(
                                constraint,
                                &branch,
                                terminal,
                                traversal_cache,
                            )
                        } else {
                            parser_child_cached(
                                constraint,
                                &branch.gss,
                                terminal,
                                traversal_cache,
                            )
                        };
                        advanced_parser.map(|advanced_parser| {
                            let repair_used = repair_used
                                || compact_parser_scope_changed(
                                    constraint,
                                    &branch.gss,
                                    &advanced_parser,
                                );
                            DynamicBranch {
                                tokenizer_config: initial_config,
                                gss: advanced_parser,
                                initial_prune_guard: matched_prune_guard,
                                last_component,
                                repair_used,
                                pending_terminals: SmallVec::new(),
                                fresh_reset: true,
                            }
                        })
                    };
                    if let Some(child) = child
                        && !push_unique_dynamic_branch(&mut next, child)
                    {
                        *duplicate_branches += 1;
                    }
                }
            }

            if !push_unique_dynamic_branch(
                &mut next,
                DynamicBranch {
                    tokenizer_config: next_config,
                    gss: branch.gss,
                    initial_prune_guard: advanced_prune_guard,
                    last_component: branch.last_component,
                    repair_used: branch.repair_used,
                    pending_terminals: branch.pending_terminals.clone(),
                    fresh_reset: false,
                },
            ) {
                *duplicate_branches += 1;
            }
        }
        std::mem::swap(&mut current, &mut next);
        *max_branches = (*max_branches).max(current.len());
        if current.is_empty() {
            return Ok(current);
        }
    }

    next.clear();
    for mut branch in current.drain(..) {
        let was_fresh_reset = branch.fresh_reset;
        let end_config = if was_fresh_reset {
            branch.fresh_reset = false;
            branch.tokenizer_config
        } else {
            let Some(end_config) = lexer_scan_cache.residual_config(branch.tokenizer_config)? else {
                continue;
            };
            end_config
        };
        if !was_fresh_reset
            && !lexer_config_relevant_cached(
            constraint,
            lexer_scan_cache,
            end_config,
            &branch.gss,
            traversal_cache,
        )
        {
            continue;
        }
        if !push_unique_dynamic_branch(
            &mut next,
            DynamicBranch {
                tokenizer_config: end_config,
                gss: branch.gss,
                initial_prune_guard: branch.initial_prune_guard,
                last_component: branch.last_component,
                repair_used: branch.repair_used,
                pending_terminals: branch.pending_terminals,
                fresh_reset: false,
            },
        ) {
            *duplicate_branches += 1;
        }
    }
    *max_branches = (*max_branches).max(next.len());
    Ok(next)
}

#[allow(clippy::too_many_arguments)]
fn process_scalar_dynamic_trie_node(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    node_id: u32,
    tokenizer_config: u32,
    stacks: &ParserStacks,
    initial_config: u32,
    lexer_scan_cache: &DynamicNfaScanCache<'_>,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    traversal_cache: &mut DynamicTraversalCache,
    buf: &mut [u32],
    stats: &mut DynamicWalkStats,
) -> bool {
    if matches!(
        raw_self_loop_subtree(
            state.constraint,
            trie,
            node_id,
            lexer_scan_cache,
            tokenizer_config,
            stacks,
            &InitialPruneGuard::Passed,
            initial_config,
            raw_self_loop_cache,
            config_self_loop_cache,
            traversal_cache,
        ),
        RawSelfLoopSubtree::MarkAllTokens
    ) {
        stats.subtree_marks += 1;
        stats.subtree_mark_tokens += trie.subtree_tokens(node_id).len();
        mark_subtree_tokens(vocab, trie, node_id, buf);
        return true;
    }

    let token_marker = vocab.node_token_marker(node_id);
    if token_marker == 0 {
        return false;
    }
    if tokenizer_config == initial_config
        || config_token_boundary_allowed_cached(
            state.constraint,
            lexer_scan_cache,
            tokenizer_config,
            stacks,
            traversal_cache,
    )
    {
        mark_dynamic_token_marker(vocab, token_marker, buf);
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn walk_scalar_dynamic_subtree(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_depth: u16,
    walk_start: usize,
    walk_end: usize,
    root_config: u32,
    stacks: &ParserStacks,
    initial_config: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    deadline_poll: &mut DynamicDeadlinePoll,
    buf: &mut [u32],
    stats: &mut DynamicWalkStats,
) -> Result<(), String> {
    let mut config_stack = Vec::<u32>::with_capacity(64);
    config_stack.push(root_config);
    let walk_edges = trie.walk_edges();
    let mut walk_index = walk_start;

    while walk_index < walk_end {
        deadline_poll.check()?;
        let edge = walk_edges[walk_index];
        let parent_index = usize::from(
            edge.parent_depth
                .checked_sub(root_depth)
                .expect("scalar trie walk escaped its subtree root"),
        );
        debug_assert!(parent_index < config_stack.len());
        config_stack.truncate(parent_index + 1);
        let parent_config = config_stack[parent_index];
        stats.trie_edges += 1;

        let mut tokenizer_config = parent_config;
        let mut needs_general = false;
        let mut dead = false;
        for &byte in trie.walk_edge_bytes(&edge) {
            stats.branch_steps += 1;
            let Some(next_config) = lexer_scan_cache.step_config(tokenizer_config, byte)? else {
                dead = true;
                break;
            };
            for state_index in 0..lexer_scan_cache.config_len(next_config) {
                let matched_state = lexer_scan_cache.config_state(next_config, state_index);
                if state
                    .constraint
                    .tokenizer
                    .matched_terminals_iter(matched_state)
                    .any(|terminal| {
                        parser_terminal_admissible_cached(
                            state.constraint,
                            terminal,
                            stacks,
                            traversal_cache,
                        )
                    })
                {
                    needs_general = true;
                    break;
                }
            }
            if needs_general {
                break;
            }
            tokenizer_config = next_config;
        }

        if dead {
            walk_index = edge.subtree_end as usize;
            continue;
        }

        if needs_general {
            let mut parent_branches = DynamicBranches::new();
            parent_branches.push(DynamicBranch {
                tokenizer_config: parent_config,
                gss: stacks.clone(),
                initial_prune_guard: InitialPruneGuard::Passed,
                last_component: DYNAMIC_NO_COMPONENT,
                repair_used: true,
                pending_terminals: SmallVec::new(),
                fresh_reset: false,
            });
            let child_branches = advance_dynamic_branches_over_segment(
                state.constraint,
                &parent_branches,
                trie.walk_edge_bytes(&edge),
                initial_config,
                lexer_scan_cache,
                traversal_cache,
                &mut stats.branch_steps,
                &mut stats.duplicate_branches,
                &mut stats.max_branches,
            )?;
            if !child_branches.is_empty() {
                walk_dynamic_subtree(
                    state,
                    vocab,
                    trie,
                    edge.child,
                    edge.parent_depth + 1,
                    walk_index + 1,
                    edge.subtree_end as usize,
                    child_branches,
                    initial_config,
                    lexer_scan_cache,
                    traversal_cache,
                    raw_self_loop_cache,
                    config_self_loop_cache,
                    deadline_poll,
                    buf,
                    stats,
                )?;
            }
            walk_index = edge.subtree_end as usize;
            continue;
        }

        let Some(tokenizer_config) = lexer_scan_cache.residual_config(tokenizer_config)? else {
            walk_index = edge.subtree_end as usize;
            continue;
        };
        if !lexer_config_relevant_cached(
            state.constraint,
            lexer_scan_cache,
            tokenizer_config,
            stacks,
            traversal_cache,
        ) {
            walk_index = edge.subtree_end as usize;
            continue;
        }

        stats.work_items += 1;
        if process_scalar_dynamic_trie_node(
            state,
            vocab,
            trie,
            edge.child,
            tokenizer_config,
            stacks,
            initial_config,
            lexer_scan_cache,
            raw_self_loop_cache,
            config_self_loop_cache,
            traversal_cache,
            buf,
            stats,
        ) {
            walk_index = edge.subtree_end as usize;
            continue;
        }

        config_stack.push(tokenizer_config);
        walk_index += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn walk_dynamic_subtree(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_node: u32,
    root_depth: u16,
    walk_start: usize,
    walk_end: usize,
    root_branches: DynamicBranches,
    initial_config: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    deadline_poll: &mut DynamicDeadlinePoll,
    buf: &mut [u32],
    stats: &mut DynamicWalkStats,
) -> Result<(), String> {
    stats.work_items += 1;
    stats.max_branches = stats.max_branches.max(root_branches.len());
    if process_dynamic_trie_node(
        state,
        vocab,
        trie,
        root_node,
        &root_branches,
        initial_config,
        lexer_scan_cache,
        raw_self_loop_cache,
        config_self_loop_cache,
        traversal_cache,
        buf,
        &mut stats.subtree_marks,
        &mut stats.subtree_mark_tokens,
    ) {
        return Ok(());
    }

    if let [branch] = root_branches.as_slice()
        && !branch.fresh_reset
        && matches!(branch.initial_prune_guard, InitialPruneGuard::Passed)
    {
        return walk_scalar_dynamic_subtree(
            state,
            vocab,
            trie,
            root_depth,
            walk_start,
            walk_end,
            branch.tokenizer_config,
            &branch.gss,
            initial_config,
            lexer_scan_cache,
            traversal_cache,
            raw_self_loop_cache,
            config_self_loop_cache,
            deadline_poll,
            buf,
            stats,
        );
    }

    let mut branch_stack = Vec::<DynamicBranches>::with_capacity(64);
    branch_stack.push(root_branches);
    let walk_edges = trie.walk_edges();
    let mut walk_index = walk_start;
    while walk_index < walk_end {
        deadline_poll.check()?;
        let edge = walk_edges[walk_index];
        let parent_index = usize::from(
            edge.parent_depth
                .checked_sub(root_depth)
                .expect("dynamic trie walk escaped its subtree root"),
        );
        debug_assert!(parent_index < branch_stack.len());
        branch_stack.truncate(parent_index + 1);
        stats.trie_edges += 1;
        let child_branches = advance_dynamic_branches_over_segment(
            state.constraint,
            &branch_stack[parent_index],
            trie.walk_edge_bytes(&edge),
            initial_config,
            lexer_scan_cache,
            traversal_cache,
            &mut stats.branch_steps,
            &mut stats.duplicate_branches,
            &mut stats.max_branches,
        )?;
        if child_branches.is_empty() {
            walk_index = edge.subtree_end as usize;
            continue;
        }

        stats.work_items += 1;
        if process_dynamic_trie_node(
            state,
            vocab,
            trie,
            edge.child,
            &child_branches,
            initial_config,
            lexer_scan_cache,
            raw_self_loop_cache,
            config_self_loop_cache,
            traversal_cache,
            buf,
            &mut stats.subtree_marks,
            &mut stats.subtree_mark_tokens,
        ) {
            walk_index = edge.subtree_end as usize;
            continue;
        }

        if let [branch] = child_branches.as_slice()
            && !branch.fresh_reset
            && matches!(branch.initial_prune_guard, InitialPruneGuard::Passed)
        {
            walk_scalar_dynamic_subtree(
                state,
                vocab,
                trie,
                edge.parent_depth + 1,
                walk_index + 1,
                edge.subtree_end as usize,
                branch.tokenizer_config,
                &branch.gss,
                initial_config,
                lexer_scan_cache,
                traversal_cache,
                raw_self_loop_cache,
                config_self_loop_cache,
                deadline_poll,
                buf,
                stats,
            )?;
            walk_index = edge.subtree_end as usize;
            continue;
        }

        branch_stack.push(child_branches);
        walk_index += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn walk_interned_dynamic_trie_range(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    baseline_coverage: Option<&DynamicBaselineCoverage>,
    candidate_coverage: Option<&DynamicCandidateCoverage>,
    require_repair_used: bool,
    root_branches: DynamicBranches,
    initial_config: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    walk_start: usize,
    walk_end: usize,
    buf: &mut [u32],
    stats: &mut DynamicWalkStats,
) -> Result<(), String> {
    let mut recognizer = DynamicRecognizerStateCache::new();
    let root_state = recognizer.intern(root_branches.clone());
    stats.max_branches = stats.max_branches.max(recognizer.branches(root_state).len());
    let mut state_stack = Vec::<u32>::with_capacity(64);
    state_stack.push(root_state);
    let walk_edges = trie.walk_edges();
    let mut walk_index = walk_start;
    while walk_index < walk_end {
        let edge = walk_edges[walk_index];
        debug_assert!((edge.subtree_end as usize) <= walk_end);
        if baseline_coverage
            .is_some_and(|coverage| coverage.subtree_fully_covered(trie, edge.child))
            || candidate_coverage
                .is_some_and(|coverage| !coverage.subtree_has_candidate(trie, edge.child))
        {
            walk_index = edge.subtree_end as usize;
            continue;
        }
        let parent_depth = edge.parent_depth as usize;
        debug_assert!(parent_depth < state_stack.len());
        state_stack.truncate(parent_depth + 1);
        let mut recognizer_state = state_stack[parent_depth];
        stats.trie_edges += 1;
        let mut alive = true;
        for &byte in trie.walk_edge_bytes(&edge) {
            let Some(next_state) = recognizer.step(
                recognizer_state,
                byte,
                state.constraint,
                initial_config,
                lexer_scan_cache,
                traversal_cache,
                stats,
            )? else {
                alive = false;
                break;
            };
            recognizer_state = next_state;
        }
        if !alive {
            walk_index = edge.subtree_end as usize;
            continue;
        }
        let normalized = recognizer.normalize(
            recognizer_state,
            state.constraint,
            lexer_scan_cache,
            traversal_cache,
            stats,
        )?;
        let Some(recognizer_state) = normalized else {
            walk_index = edge.subtree_end as usize;
            continue;
        };
        stats.work_items += 1;
        let processed = process_interned_dynamic_trie_node(
            state,
            vocab,
            trie,
            edge.child,
            &mut recognizer,
            recognizer_state,
            initial_config,
            lexer_scan_cache,
            raw_self_loop_cache,
            config_self_loop_cache,
            traversal_cache,
            buf,
            stats,
            require_repair_used,
        );
        if processed {
            walk_index = edge.subtree_end as usize;
            continue;
        }
        state_stack.push(recognizer_state);
        walk_index += 1;
    }
    stats.recognizer_states += recognizer.branches.len();
    stats.recognizer_transition_misses += recognizer.transition_misses;
    Ok(())
}

#[inline]
fn dynamic_config_matches_projection(
    vocab: &DynamicMaskVocab,
    lexer_scan_cache: &DynamicNfaScanCache<'_>,
    tokenizer_config: u32,
    expected: &[u32],
) -> bool {
    if lexer_scan_cache.config_len(tokenizer_config) > 16 || expected.len() > 16 {
        return false;
    }
    let mut projected = SmallVec::<[u32; 16]>::new();
    for index in 0..lexer_scan_cache.config_len(tokenizer_config) {
        projected.push(vocab.mask_projection_state(
            lexer_scan_cache.config_state(tokenizer_config, index),
        ));
    }
    projected.sort_unstable();
    projected.dedup();
    projected.as_slice() == expected
}

fn dynamic_config_projection_certifies_subtree(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    projection: &DynamicSelfLoopProjection,
    node: u32,
    recognizer: &DynamicRecognizerStateCache,
    recognizer_state: u32,
    lexer_scan_cache: &DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
    stats: &mut DynamicWalkStats,
) -> bool {
    let certificates = projection.config_subtree_certificates_for_node(node);
    if certificates.is_empty() {
        return false;
    }
    stats.config_projection_node_hits += 1;
    for certificate in certificates {
        for branch in recognizer.branches(recognizer_state) {
            // The certificate skips all remaining bytes below this node, so
            // token-start guard state must already be discharged.  Pending
            // lazy parser terminals likewise require byte-by-byte replay.
            if branch.fresh_reset
                || !branch.pending_terminals.is_empty()
                || !branch.initial_prune_guard.is_passed()
            {
                stats.config_projection_guard_rejects += 1;
                continue;
            }
            if !dynamic_config_matches_projection(
                    vocab,
                    lexer_scan_cache,
                    branch.tokenizer_config,
                    certificate.projected_config.as_ref(),
                ) {
                continue;
            }
            stats.config_projection_config_matches += 1;
            if certificate
                .common_future_terminals
                .iter()
                .copied()
                .any(|terminal| {
                    parser_terminal_admissible_cached(
                        state.constraint,
                        terminal,
                        &branch.gss,
                        traversal_cache,
                    )
                })
            {
                return true;
            }
            stats.config_projection_parser_rejects += 1;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn walk_interned_dynamic_trie(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    projections: &[(&DynamicSelfLoopProjection, u64)],
    alias_projections_vocab: &[(&DynamicSelfLoopProjection, u64)],
    alias_projections_h64: &[(&DynamicSelfLoopProjection, u64)],
    baseline_coverage: Option<&DynamicBaselineCoverage>,
    candidate_coverage: Option<&DynamicCandidateCoverage>,
    require_repair_used: bool,
    root_branches: DynamicBranches,
    initial_config: u32,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    traversal_cache: &mut DynamicTraversalCache,
    raw_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    config_self_loop_cache: &mut FxHashMap<u32, U8Set>,
    deadline_poll: &mut DynamicDeadlinePoll,
    buf: &mut [u32],
    stats: &mut DynamicWalkStats,
) -> Result<(), String> {
    let mut recognizer = DynamicRecognizerStateCache::new();
    let root_state = recognizer.intern(root_branches.clone());
    let pre_match_projection = if let [branch] = root_branches.as_slice()
        && !branch.fresh_reset
        && branch.pending_terminals.is_empty()
        && branch.initial_prune_guard.is_passed()
        && lexer_scan_cache.config_len(branch.tokenizer_config) == 1
    {
        let raw_state = lexer_scan_cache.config_state(branch.tokenizer_config, 0);
        vocab
            .self_loop_projection(raw_state)
            .filter(|projection| projection.has_pre_match_dead_subtrees())
    } else {
        None
    };
    let root_next_bytes = root_branches
        .iter()
        .all(|branch| {
            !branch.fresh_reset
                && branch.pending_terminals.is_empty()
                && branch.initial_prune_guard.is_passed()
        })
        .then(|| {
            root_branches.iter().fold(U8Set::empty(), |bytes, branch| {
                bytes.union(&lexer_scan_cache.config_next_bytes(branch.tokenizer_config))
            })
        });
    stats.max_branches = stats.max_branches.max(recognizer.branches(root_state).len());
    stats.work_items += 1;
    if baseline_coverage.is_some_and(|coverage| coverage.subtree_fully_covered(trie, 0))
        || candidate_coverage.is_some_and(|coverage| !coverage.subtree_has_candidate(trie, 0))
    {
        recognizer.profile_shape_into(stats);
        stats.recognizer_states = recognizer.branches.len();
        stats.recognizer_transition_misses = recognizer.transition_misses;
        return Ok(());
    }
    if projections.iter().any(|(projection, admissible_future_mask)| {
        projection.subtree_common_future_mask(0) & *admissible_future_mask != 0
    }) || alias_projections_vocab
        .iter()
        .any(|(projection, admissible_future_mask)| {
            projection.subtree_common_future_mask(0) & *admissible_future_mask != 0
        })
    {
        stats.subtree_marks += 1;
        stats.subtree_mark_tokens += trie.subtree_tokens(0).len();
        mark_subtree_tokens(vocab, trie, 0, buf);
        recognizer.profile_shape_into(stats);
        stats.recognizer_states = recognizer.branches.len();
        stats.recognizer_transition_misses = recognizer.transition_misses;
        return Ok(());
    }
    let root_processed = process_interned_dynamic_trie_node(
        state,
        vocab,
        trie,
        0,
        &mut recognizer,
        root_state,
        initial_config,
        lexer_scan_cache,
        raw_self_loop_cache,
        config_self_loop_cache,
        traversal_cache,
        buf,
        stats,
        require_repair_used,
    );
    if root_processed {
        recognizer.profile_shape_into(stats);
        stats.recognizer_states = recognizer.branches.len();
        stats.recognizer_transition_misses = recognizer.transition_misses;
        return Ok(());
    }

    let use_parallel_overlay = require_repair_used
        && std::env::var_os("GLRMASK_EXPERIMENT_PARALLEL_DYNAMIC_OVERLAY").is_some()
        && rayon::current_num_threads() > 1
        && trie.walk_edges().len() >= 1_024;
    if use_parallel_overlay {
        let walk_edges = trie.walk_edges();
        let mut top_ranges = Vec::<(usize, usize)>::new();
        let mut index = 0usize;
        while index < walk_edges.len() {
            let edge = walk_edges[index];
            debug_assert_eq!(edge.parent_depth, 0);
            let end = edge.subtree_end as usize;
            debug_assert!(index < end && end <= walk_edges.len());
            if !baseline_coverage
                .is_some_and(|coverage| coverage.subtree_fully_covered(trie, edge.child))
                && !candidate_coverage
                    .is_some_and(|coverage| !coverage.subtree_has_candidate(trie, edge.child))
            {
                top_ranges.push((index, end));
            }
            index = end;
        }
        if top_ranges.len() > 1 {
            let base_scan_cache = lexer_scan_cache.clone();
            let results = top_ranges
                .par_iter()
                .map(|&(walk_start, walk_end)| -> Result<(Vec<u32>, DynamicWalkStats), String> {
                    let mut local_scan_cache = base_scan_cache.clone();
                    let mut local_traversal_cache = DynamicTraversalCache::default();
                    let mut local_raw_self_loop_cache = FxHashMap::<u32, U8Set>::default();
                    let mut local_config_self_loop_cache = FxHashMap::<u32, U8Set>::default();
                    let mut local_buf = vec![0u32; buf.len()];
                    let mut local_stats = DynamicWalkStats::default();
                    walk_interned_dynamic_trie_range(
                        state,
                        vocab,
                        trie,
                        baseline_coverage,
                        candidate_coverage,
                        require_repair_used,
                        root_branches.clone(),
                        initial_config,
                        &mut local_scan_cache,
                        &mut local_traversal_cache,
                        &mut local_raw_self_loop_cache,
                        &mut local_config_self_loop_cache,
                        walk_start,
                        walk_end,
                        &mut local_buf,
                        &mut local_stats,
                    )?;
                    Ok((local_buf, local_stats))
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (local_buf, local_stats) in results {
                for (dst, src) in buf.iter_mut().zip(local_buf) {
                    *dst |= src;
                }
                stats.merge_from(&local_stats);
            }
            stats.recognizer_states += recognizer.branches.len();
            stats.recognizer_transition_misses += recognizer.transition_misses;
            return Ok(());
        }
    }

    let mut state_stack = Vec::<u32>::with_capacity(64);
    state_stack.push(root_state);
    let mut pre_match_stack = Vec::<bool>::with_capacity(64);
    pre_match_stack.push(pre_match_projection.is_some());
    let walk_edges = trie.walk_edges();
    let mut walk_index = 0usize;
    while walk_index < walk_edges.len() {
        deadline_poll.check()?;
        let edge = walk_edges[walk_index];
        let config_projection_candidate = pre_match_projection.is_some_and(|projection| {
            !projection
                .config_subtree_certificates_for_node(edge.child)
                .is_empty()
        });
        if config_projection_candidate {
            stats.config_projection_candidate_edges += 1;
        }
        if baseline_coverage
            .is_some_and(|coverage| coverage.subtree_fully_covered(trie, edge.child))
            || candidate_coverage
                .is_some_and(|coverage| !coverage.subtree_has_candidate(trie, edge.child))
        {
            walk_index = edge.subtree_end as usize;
            continue;
        }
        // The partitioned runtime trie has zero-byte layout edges at depth
        // zero.  If none of the current lexer branches can consume the first
        // byte of any token in such a class, reject the entire class without
        // entering its radix subtree.  The same test also handles ordinary
        // non-empty root edges.  Restrict this to normalized root branches so
        // pending-finalization/reset semantics cannot introduce a different
        // next-byte source before input is consumed.
        if edge.parent_depth == 0
            && let Some(next_bytes) = root_next_bytes
        {
            let impossible = if edge.byte_len == 0 {
                next_bytes.is_disjoint(&U8Set::from_words(
                    trie.subtree_first_bytes(edge.child),
                ))
            } else {
                !next_bytes.contains(trie.walk_edge_bytes(&edge)[0])
            };
            if impossible {
                walk_index = edge.subtree_end as usize;
                continue;
            }
        }
        if projections
            .iter()
            .any(|(projection, _)| projection.subtree_is_safe(edge.child))
        {
            stats.subtree_marks += 1;
            stats.subtree_mark_tokens += trie.subtree_tokens(edge.child).len();
            walk_index = edge.subtree_end as usize;
            continue;
        }
        // Vocabulary-relative aliases were proven by exact equality of the
        // selected terminal's common-future result over this concrete vocab
        // trie. They are therefore valid for every token length, but only for
        // the common-future certificate (not source-specific safe/re-entry
        // fields from the canonical projection).
        if alias_projections_vocab
            .iter()
            .any(|(projection, admissible_future_mask)| {
                projection.subtree_common_future_mask(edge.child)
                    & *admissible_future_mask
                    != 0
            })
        {
            stats.subtree_marks += 1;
            stats.subtree_mark_tokens += trie.subtree_tokens(edge.child).len();
            mark_subtree_tokens(vocab, trie, edge.child, buf);
            walk_index = edge.subtree_end as usize;
            continue;
        }
        // H64 aliases only prove equivalence of the selected continuing
        // terminal from the *root source state*.  Reuse only the projection's
        // common-future certificate, and only when every complete vocabulary
        // token below this node fits inside the 64-byte proof horizon.
        if trie.subtree_max_total_byte_len(edge.child) <= 64
            && alias_projections_h64
                .iter()
                .any(|(projection, admissible_future_mask)| {
                    projection.subtree_common_future_mask(edge.child)
                        & *admissible_future_mask
                        != 0
                })
        {
            stats.subtree_marks += 1;
            stats.subtree_mark_tokens += trie.subtree_tokens(edge.child).len();
            mark_subtree_tokens(vocab, trie, edge.child, buf);
            walk_index = edge.subtree_end as usize;
            continue;
        }
        if projections.iter().any(|(projection, admissible_future_mask)| {
            projection.subtree_common_future_mask(edge.child) & *admissible_future_mask != 0
        }) {
            stats.subtree_marks += 1;
            stats.subtree_mark_tokens += trie.subtree_tokens(edge.child).len();
            mark_subtree_tokens(vocab, trie, edge.child, buf);
            walk_index = edge.subtree_end as usize;
            continue;
        }
        let parent_depth = edge.parent_depth as usize;
        debug_assert!(parent_depth < state_stack.len());
        state_stack.truncate(parent_depth + 1);
        pre_match_stack.truncate(parent_depth + 1);
        let parent_pre_match = pre_match_stack[parent_depth];
        let child_pre_match = if parent_pre_match {
            let projection = pre_match_projection
                .expect("pre-match stack can only be live with a projection");
            if projection.pre_match_subtree_is_dead(edge.child) {
                walk_index = edge.subtree_end as usize;
                continue;
            }
            !projection.pre_match_subtree_is_frontier(edge.child)
        } else {
            false
        };
        let mut recognizer_state = state_stack[parent_depth];
        stats.trie_edges += 1;
        let mut alive = true;
        for &byte in trie.walk_edge_bytes(&edge) {
            let Some(next_state) = recognizer.step(
                recognizer_state,
                byte,
                state.constraint,
                initial_config,
                lexer_scan_cache,
                traversal_cache,
                stats,
            )? else {
                alive = false;
                break;
            };
            recognizer_state = next_state;
        }
        if !alive {
            if config_projection_candidate {
                stats.config_projection_step_dead += 1;
            }
            walk_index = edge.subtree_end as usize;
            continue;
        }

        let normalized = recognizer.normalize(
            recognizer_state,
            state.constraint,
            lexer_scan_cache,
            traversal_cache,
            stats,
        )?;
        let Some(recognizer_state) = normalized else {
            if config_projection_candidate {
                stats.config_projection_normalize_dead += 1;
            }
            walk_index = edge.subtree_end as usize;
            continue;
        };

        if !require_repair_used
            && pre_match_projection.is_some_and(|projection| {
                dynamic_config_projection_certifies_subtree(
                    state,
                    vocab,
                    projection,
                    edge.child,
                    &recognizer,
                    recognizer_state,
                    lexer_scan_cache,
                    traversal_cache,
                    stats,
                )
            })
        {
            let tokens = trie.subtree_tokens(edge.child).len();
            stats.subtree_marks += 1;
            stats.subtree_mark_tokens += tokens;
            stats.config_projection_marks += 1;
            stats.config_projection_tokens += tokens;
            mark_subtree_tokens(vocab, trie, edge.child, buf);
            walk_index = edge.subtree_end as usize;
            continue;
        }

        stats.work_items += 1;
        let processed = process_interned_dynamic_trie_node(
            state,
            vocab,
            trie,
            edge.child,
            &mut recognizer,
            recognizer_state,
            initial_config,
            lexer_scan_cache,
            raw_self_loop_cache,
            config_self_loop_cache,
            traversal_cache,
            buf,
            stats,
            require_repair_used,
        );
        if processed {
            walk_index = edge.subtree_end as usize;
            continue;
        }

        state_stack.push(recognizer_state);
        pre_match_stack.push(child_pre_match);
        walk_index += 1;
    }

    recognizer.profile_shape_into(stats);
    stats.recognizer_states = recognizer.branches.len();
    stats.recognizer_transition_misses = recognizer.transition_misses;
    Ok(())
}

fn fill_mask_dynamic_impl(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    deadline: Option<Instant>,
    additive_static_baseline: bool,
    candidate_mask: Option<&[u32]>,
) -> Result<(), String> {
    let required = state.constraint.mask_len();
    assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
    let (buf, tail) = buf.split_at_mut(required);
    tail.fill(0);
    let mut deadline_poll = DynamicDeadlinePoll::new(deadline);
    let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
    let total_started_at = profile.then(std::time::Instant::now);
    let key_started_at = profile.then(std::time::Instant::now);
    let cache_key = (!additive_static_baseline && dynamic_mask_cache_enabled())
        .then(|| dynamic_mask_state_key(state))
        .flatten();
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

    let mut baseline_snapshot = candidate_mask.map(|_| buf.to_vec());
    let baseline_coverage = additive_static_baseline
        .then(|| DynamicBaselineCoverage::new(vocab, vocab.trie.as_ref(), buf));
    let mut candidate_coverage = candidate_mask
        .map(|candidates| DynamicCandidateCoverage::new(vocab, vocab.trie.as_ref(), candidates));
    let mut one_step_candidate_mask = None::<Vec<u32>>;
    if !additive_static_baseline {
        buf.fill(0);
    }
    let initial_tsid = state.constraint.tokenizer.initial_state();
    let mut root_branches = DynamicBranches::new();
    let mut sole_root_source_state = None::<u32>;
    let mut raw_self_loop_cache = FxHashMap::<u32, U8Set>::default();
    let mut config_self_loop_cache = FxHashMap::<u32, U8Set>::default();
    let mut traversal_cache = DynamicTraversalCache::default();
    if profile {
        traversal_cache.profile_interaction_hash = Some(0xcbf29ce484222325u64);
    }
    let mut lexer_scan_cache = DynamicNfaScanCache::new(state.constraint, deadline);
    let initial_config = lexer_scan_cache.config_for_raw_start(initial_tsid)?;
    let trie = vocab.trie.as_ref();
    let mut stats = DynamicWalkStats::default();
    if profile {
        eprintln!(
            "[glrmask/profile][dynamic_mask_config] tokenizer_states={} epsilon={} fast_transition_rows={}",
            state.constraint.tokenizer.num_states(),
            state.constraint.tokenizer_has_epsilon_transitions,
            state.constraint.tokenizer_fast_transitions.len(),
        );
        if let Ok(value) = std::env::var("GLRMASK_PROFILE_DYNAMIC_RELEVANT_TERMINALS") {
            let terminals = value
                .split(',')
                .filter_map(|value| value.trim().parse::<TerminalID>().ok())
                .map(|terminal| {
                    (
                        terminal,
                        state
                            .constraint
                            .terminal_display_name(terminal)
                            .unwrap_or("<unnamed>")
                            .to_string(),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][dynamic_relevant_terminal_names] {:?}",
                terminals
            );
        }
    }

    for (&tokenizer_state, gss) in &state.state {
        deadline_poll.check()?;
        for (stacks, terminals_disallowed) in gss.partition_by_accumulator() {
            deadline_poll.check()?;
            let initial_prune_guard = InitialPruneGuard::new(
                state.constraint,
                tokenizer_state,
                &stacks,
                &terminals_disallowed,
            );
            if profile {
                // Keep diagnostic parser-signature queries out of the live
                // traversal cache.  They must not prime or otherwise alter
                // the mask walk whose interaction transcript is measured
                // below.
                let mut signature_cache = DynamicTraversalCache::default();
                let bitset_fingerprint = |bits: &BitSet| {
                    bits.words().iter().fold(0xcbf29ce484222325u64, |hash, &word| {
                        (hash ^ word).wrapping_mul(0x100000001b3)
                    })
                };
                let root_admissible = admissible_terminals_cached(
                    state.constraint,
                    &stacks,
                    &mut signature_cache,
                );
                let root_admissible_fingerprint = bitset_fingerprint(root_admissible);
                let root_admissible_count = root_admissible.count_ones();
                let diagnostic_terminals = std::env::var(
                    "GLRMASK_PROFILE_DYNAMIC_RELEVANT_TERMINALS",
                )
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|value| value.trim().parse::<TerminalID>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
                let relevant_signature = |admitted: &BitSet| {
                    diagnostic_terminals
                        .iter()
                        .copied()
                        .enumerate()
                        .fold(0u64, |mask, (index, terminal)| {
                            if index < 64 && admitted.contains(terminal as usize) {
                                mask | (1u64 << index)
                            } else {
                                mask
                            }
                        })
                };
                let root_relevant_signature = relevant_signature(root_admissible);
                let top_action_fingerprint = (!state
                    .constraint
                    .uses_compact_segmented_parser_runtime())
                .then(|| stacks.single_top_value())
                .flatten()
                .map(|top| {
                    let mut hasher = rustc_hash::FxHasher::default();
                    top.hash(&mut hasher);
                    for terminal in 0..state.constraint.table.num_terminals {
                        terminal.hash(&mut hasher);
                        state
                            .constraint
                            .table
                            .action(top, terminal)
                            .hash(&mut hasher);
                    }
                    hasher.finish()
                });
                let mut futures = state
                    .constraint
                    .tokenizer
                    .possible_future_terminals_iter(tokenizer_state);
                let sole_future = futures.next().filter(|_| futures.next().is_none());
                let sole_future_action = (!state
                    .constraint
                    .uses_compact_segmented_parser_runtime())
                .then_some(sole_future)
                .flatten()
                .and_then(|terminal| {
                    stacks.single_top_value().map(|top| {
                        (
                            terminal,
                            top,
                            state.constraint.table.action(top, terminal).cloned(),
                        )
                    })
                });
                let post_first = sole_future.and_then(|terminal| {
                    parser_child(state.constraint, &stacks, terminal).map(|child| {
                        let admitted = admissible_terminals_cached(
                            state.constraint,
                            &child,
                            &mut signature_cache,
                        );
                        (
                            terminal,
                            bitset_fingerprint(admitted),
                            admitted.count_ones(),
                            relevant_signature(admitted),
                            child.single_top_value(),
                            (admitted.count_ones() <= 16).then(|| {
                                admitted
                                    .iter_ones()
                                    .map(|terminal| terminal as TerminalID)
                                    .collect::<Vec<_>>()
                            }),
                        )
                    })
                });
                eprintln!(
                    "[glrmask/profile][dynamic_seed_parser_signature] generation={} tokenizer_state={} root_admissible={:016x} root_admissible_count={} relevant_signature={:016x} top_action={:?} sole_future_action={:?} post_first={:?}",
                    state.generation,
                    tokenizer_state,
                    root_admissible_fingerprint,
                    root_admissible_count,
                    root_relevant_signature,
                    top_action_fingerprint,
                    sole_future_action,
                    post_first,
                );
                if let Some((stack, _)) = stacks.try_single_stack_bounded(128) {
                    eprintln!(
                        "[glrmask/profile][dynamic_seed_stack] tokenizer_state={} depth={} bottom_first={:?}",
                        tokenizer_state,
                        stack.len(),
                        stack,
                    );
                }
                let loop_bytes = cached_self_loop_bytes(
                    state.constraint,
                    tokenizer_state,
                    &mut raw_self_loop_cache,
                );
                eprintln!(
                    "[glrmask/profile][dynamic_seed] generation={} tokenizer_state={} initial={} stack_paths={} exclusions={} transitions={} matched={} futures={} loop_bytes={} boundary_allowed={}",
                    state.generation,
                    tokenizer_state,
                    tokenizer_state == initial_tsid,
                    stacks.path_count_at_most(1_000_000),
                    terminals_disallowed
                        .iter()
                        .map(|(_, terminals)| terminals.len())
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
            let tokenizer_config = lexer_scan_cache.config_for_raw_start(tokenizer_state)?;
            if !push_unique_dynamic_branch(
                &mut root_branches,
                DynamicBranch {
                    tokenizer_config,
                    gss: stacks,
                    initial_prune_guard,
                    last_component: if additive_static_baseline {
                        overlay_tokenizer_component(state.constraint, tokenizer_state)
                    } else {
                        DYNAMIC_NO_COMPONENT
                    },
                    repair_used: !additive_static_baseline,
                    pending_terminals: SmallVec::new(),
                    fresh_reset: false,
                },
            ) {
                stats.duplicate_branches += 1;
            }
            sole_root_source_state = match sole_root_source_state {
                None => Some(tokenizer_state),
                Some(existing) if existing == tokenizer_state => Some(existing),
                Some(_) => Some(u32::MAX),
            };
        }
    }

    // General root lexical-effect program.  This is the multi-future analogue
    // of the single-first-match path below: the lexer work for every ordinary
    // vocabulary token was compiled at constraint construction time, while
    // parser advances remain live and exact here.  Any depth-limited token is
    // left in the candidate mask for the ordinary dynamic walker.
    if !additive_static_baseline
        && candidate_mask.is_none()
        && state.constraint.static_dynamic_overlay.is_none()
        && let [branch] = root_branches.as_slice()
        && !branch.fresh_reset
        && branch.pending_terminals.is_empty()
        && branch.initial_prune_guard.is_passed()
        && let Some(source_state) = sole_root_source_state.filter(|&state| state != u32::MAX)
    {
        if let Some(projection) = vocab.self_loop_projection(source_state)
            && projection.has_root_effect_from(source_state)
        {
            #[inline]
            fn root_top_may_have_action(
                constraint: &Constraint,
                gss: &ParserStacks,
                terminal: TerminalID,
            ) -> bool {
                let mut found = false;
                gss.for_each_top_value(|top| {
                    if found {
                        return;
                    }
                    found = constraint
                        .table
                        .advance
                        .get(top as usize)
                        .is_some_and(|row| row.contains(terminal as usize));
                    if !found
                        && constraint.table.advance.len() != constraint.table.num_states as usize
                    {
                        found = constraint.table.action(top, terminal).is_some();
                    }
                });
                found
            }

            fn root_post_row_admissible(
                constraint: &Constraint,
                row: &crate::runtime::artifact::DynamicFirstMatchPostRow,
                gss: &ParserStacks,
                traversal_cache: &mut DynamicTraversalCache,
            ) -> bool {
                for &terminal in row.terminals.iter() {
                    if root_top_may_have_action(constraint, gss, terminal)
                        && parser_child_cached(constraint, gss, terminal, traversal_cache).is_some()
                    {
                        return true;
                    }
                }
                false
            }

            #[inline]
            fn mark_post_row(
                row: &crate::runtime::artifact::DynamicFirstMatchPostRow,
                buf: &mut [u32],
            ) {
                if !row.dense_mask.is_empty() {
                    for (dst, &src) in buf.iter_mut().zip(row.dense_mask.iter()) {
                        *dst |= src;
                    }
                } else {
                    for &token_id in row.tokens.iter() {
                        set_mask_bit_known_in_range(buf, token_id);
                    }
                }
            }

            fn apply_root_effect_rows(
                constraint: &Constraint,
                rows: &[crate::runtime::artifact::DynamicFirstMatchSecondRow],
                gss: &ParserStacks,
                traversal_cache: &mut DynamicTraversalCache,
                buf: &mut [u32],
            ) {
                for effect_row in rows {
                    if !root_top_may_have_action(constraint, gss, effect_row.terminal) {
                        continue;
                    }
                    let Some(child_gss) = parser_child_cached(
                        constraint,
                        gss,
                        effect_row.terminal,
                        traversal_cache,
                    ) else {
                        continue;
                    };
                    for &token_id in effect_row.exact_end_tokens.iter() {
                        set_mask_bit_known_in_range(buf, token_id);
                    }
                    for row in effect_row.post_rows.iter() {
                        if root_post_row_admissible(
                            constraint,
                            row,
                            &child_gss,
                            traversal_cache,
                        ) {
                            mark_post_row(row, buf);
                        }
                    }
                    if !effect_row.next_rows.is_empty() {
                        apply_root_effect_rows(
                            constraint,
                            effect_row.next_rows.as_ref(),
                            &child_gss,
                            traversal_cache,
                            buf,
                        );
                    }
                }
            }

            for row in projection.root_effect_post_rows.iter() {
                if root_post_row_admissible(
                    state.constraint,
                    row,
                    &branch.gss,
                    &mut traversal_cache,
                ) {
                    mark_post_row(row, buf);
                }
            }
            apply_root_effect_rows(
                state.constraint,
                projection.root_effect_rows.as_ref(),
                &branch.gss,
                &mut traversal_cache,
                buf,
            );

            if projection.root_effect_unknown_tokens.is_empty() {
                // The lexical-effect program is complete for every ordinary
                // vocabulary token.  Special tokens are handled independently
                // by `update_special_token_mask` below, so there is no reason
                // to allocate a candidate mask or enter the byte recognizer.
                root_branches.clear();
            } else {
                let mut unknown_mask = vec![0u32; required];
                for &token_id in projection.root_effect_unknown_tokens.iter() {
                    set_mask_bit_known_in_range(&mut unknown_mask, token_id);
                }
                for special in &state.constraint.special_token_terminals {
                    set_mask_bit_known_in_range(&mut unknown_mask, special.token_id);
                }
                baseline_snapshot = Some(buf.to_vec());
                candidate_coverage = Some(DynamicCandidateCoverage::from_subtree_bits(
                    Arc::clone(&projection.root_effect_unknown_subtrees),
                ));
                one_step_candidate_mask = Some(unknown_mask);
            }

            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_root_effect_runtime] source_state={} post_rows={} root_rows={} baseline_tokens={} unknown={}",
                    source_state,
                    projection.root_effect_post_rows.len(),
                    projection.root_effect_rows.len(),
                    buf.iter().map(|word| word.count_ones() as usize).sum::<usize>(),
                    projection.root_effect_unknown_tokens.len(),
                );
            }
        }
    }

    // Exact vocabulary-relative one-finalization projection.  This path is
    // intentionally independent of the ordinary dynamic-mask cache: all
    // lexical classification was performed at constraint construction time.
    // Runtime does one real parser advance for the sole first terminal, ORs
    // the precomputed residual-language rows admitted by that child parser
    // state, and restricts the exact walker to the statically-unknown tokens.
    //
    // Keep the pilot conservative: one correlated root branch, no additive
    // composition overlay, no pending token-start exclusions, and one scalar
    // lexer state.  Everything else takes the existing exact path unchanged.
    if !additive_static_baseline
        && candidate_mask.is_none()
        && one_step_candidate_mask.is_none()
        && state.constraint.static_dynamic_overlay.is_none()
        && let [branch] = root_branches.as_slice()
        && !branch.fresh_reset
        && branch.pending_terminals.is_empty()
        && branch.initial_prune_guard.is_passed()
        && lexer_scan_cache.config_len(branch.tokenizer_config) == 1
    {
        let source_state = lexer_scan_cache.config_state(branch.tokenizer_config, 0);
        if let Some(projection) = vocab.self_loop_projection(source_state)
            && projection.has_first_match_step_from(source_state)
            && let [terminal] = projection.future_terminals.as_ref()
            && let Some(post_first_gss) = parser_child_cached(
                state.constraint,
                &branch.gss,
                *terminal,
                &mut traversal_cache,
            )
        {
            #[inline]
            fn mark_ids(ids: &[u32], buf: &mut [u32]) {
                for &token_id in ids {
                    set_mask_bit_known_in_range(buf, token_id);
                }
            }

            #[inline]
            fn top_may_have_terminal_action(
                constraint: &Constraint,
                gss: &ParserStacks,
                terminal: TerminalID,
            ) -> bool {
                if constraint.uses_compact_segmented_parser_runtime() {
                    let parser_gss = with_empty_accumulators(gss);
                    return constraint
                        .compact_segmented_parser_may_advance_on(&parser_gss, terminal)
                        .unwrap_or(false);
                }
                let mut found = false;
                gss.for_each_top_value(|top| {
                    if found {
                        return;
                    }
                    found = constraint
                        .table
                        .advance
                        .get(top as usize)
                        .is_some_and(|row| row.contains(terminal as usize));
                    if !found && constraint.table.advance.len() != constraint.table.num_states as usize
                    {
                        found = constraint.table.action(top, terminal).is_some();
                    }
                });
                found
            }

            #[inline]
            fn post_row_is_admissible(
                constraint: &Constraint,
                row: &crate::runtime::artifact::DynamicFirstMatchPostRow,
                gss: &ParserStacks,
                traversal_cache: &mut DynamicTraversalCache,
            ) -> bool {
                for &terminal in row.terminals.iter() {
                    if top_may_have_terminal_action(constraint, gss, terminal)
                        && parser_child_cached(constraint, gss, terminal, traversal_cache).is_some()
                    {
                        return true;
                    }
                }
                false
            }

            mark_ids(projection.first_match_step_root_live_tokens.as_ref(), buf);
            mark_ids(projection.first_match_step_exact_end_tokens.as_ref(), buf);

            let mut matched_rows = 0usize;
            for row in projection.first_match_step_post_rows.iter() {
                if post_row_is_admissible(
                    state.constraint,
                    row,
                    &post_first_gss,
                    &mut traversal_cache,
                ) {
                    matched_rows += 1;
                    mark_ids(row.tokens.as_ref(), buf);
                }
            }

            fn apply_effect_rows(
                constraint: &Constraint,
                rows: &[crate::runtime::artifact::DynamicFirstMatchSecondRow],
                gss: &ParserStacks,
                traversal_cache: &mut DynamicTraversalCache,
                buf: &mut [u32],
                matched_effect_rows: &mut usize,
                matched_effect_post_rows: &mut usize,
            ) {
                for effect_row in rows {
                    if !top_may_have_terminal_action(constraint, gss, effect_row.terminal) {
                        continue;
                    }
                    let Some(child_gss) = parser_child_cached(
                        constraint,
                        gss,
                        effect_row.terminal,
                        traversal_cache,
                    ) else {
                        continue;
                    };
                    *matched_effect_rows += 1;
                    for &token_id in effect_row.exact_end_tokens.iter() {
                        set_mask_bit_known_in_range(buf, token_id);
                    }
                    if !effect_row.post_rows.is_empty() {
                        for row in effect_row.post_rows.iter() {
                            if post_row_is_admissible(
                                constraint,
                                row,
                                &child_gss,
                                traversal_cache,
                            ) {
                                *matched_effect_post_rows += 1;
                                for &token_id in row.tokens.iter() {
                                    set_mask_bit_known_in_range(buf, token_id);
                                }
                            }
                        }
                    }
                    if !effect_row.next_rows.is_empty() {
                        apply_effect_rows(
                            constraint,
                            effect_row.next_rows.as_ref(),
                            &child_gss,
                            traversal_cache,
                            buf,
                            matched_effect_rows,
                            matched_effect_post_rows,
                        );
                    }
                }
            }

            let mut matched_second_rows = 0usize;
            let mut matched_second_post_rows = 0usize;
            apply_effect_rows(
                state.constraint,
                projection.first_match_step_second_rows.as_ref(),
                &post_first_gss,
                &mut traversal_cache,
                buf,
                &mut matched_second_rows,
                &mut matched_second_post_rows,
            );

            if projection.first_match_step_unknown_tokens.is_empty() {
                root_branches.clear();
            } else {
                let mut unknown_mask = vec![0u32; required];
                mark_ids(
                    projection.first_match_step_unknown_tokens.as_ref(),
                    &mut unknown_mask,
                );
                // Special tokens do not live in the ordinary byte trie.  Keep
                // them in the final candidate filter so `update_special_token_mask`
                // can still add them after the restricted walk.
                for special in &state.constraint.special_token_terminals {
                    set_mask_bit_known_in_range(&mut unknown_mask, special.token_id);
                }

                baseline_snapshot = Some(buf.to_vec());
                candidate_coverage = Some(DynamicCandidateCoverage::from_subtree_bits(
                    Arc::clone(&projection.first_match_step_unknown_subtrees),
                ));
                one_step_candidate_mask = Some(unknown_mask);
            }

            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_first_match_one_step_runtime] source_state={} terminal={} root_live={} exact_end={} post_rows={} matched_rows={} second_rows={} matched_second_rows={} matched_second_post_rows={} baseline_tokens={} unknown={}",
                    source_state,
                    terminal,
                    projection.first_match_step_root_live_tokens.len(),
                    projection.first_match_step_exact_end_tokens.len(),
                    projection.first_match_step_post_rows.len(),
                    matched_rows,
                    projection.first_match_step_second_rows.len(),
                    matched_second_rows,
                    matched_second_post_rows,
                    buf.iter().map(|word| word.count_ones() as usize).sum::<usize>(),
                    projection.first_match_step_unknown_tokens.len(),
                );
            }
        }
    }
    let effective_candidate_mask = candidate_mask.or(one_step_candidate_mask.as_deref());

    let mut projections = SmallVec::<[(&DynamicSelfLoopProjection, u64); 4]>::new();
    let mut alias_projections_vocab =
        SmallVec::<[(&DynamicSelfLoopProjection, u64); 4]>::new();
    let mut alias_projections_h64 =
        SmallVec::<[(&DynamicSelfLoopProjection, u64); 4]>::new();
    if !additive_static_baseline {
        // These projections certify the complete dynamic language, not the
        // additive repair sublanguage. Static masking already covers their
        // ordinary local paths, so do not use them for overlay marking.
        for branch in &root_branches {
            if branch.fresh_reset
                || !branch.pending_terminals.is_empty()
                || !branch.initial_prune_guard.is_passed()
                || lexer_scan_cache.config_len(branch.tokenizer_config) != 1
            {
                continue;
            }
            let source_state = lexer_scan_cache.config_state(branch.tokenizer_config, 0);
            let exact_projection = vocab.self_loop_projection(source_state);
            let vocab_alias_projection = vocab.self_loop_projection_alias_vocab(source_state);
            let h64_alias_projection = vocab.self_loop_projection_alias_h64(source_state);
            let Some(projection) = exact_projection
                .or(vocab_alias_projection)
                .or(h64_alias_projection)
            else {
                continue;
            };
            let admissible_mask = projection
                .future_terminals
                .iter()
                .copied()
                .enumerate()
                .fold(0u64, |mask, (index, terminal)| {
                    if index < 64
                        && parser_terminal_admissible_cached(
                            state.constraint,
                            terminal,
                            &branch.gss,
                            &mut traversal_cache,
                        )
                    {
                        mask | (1u64 << index)
                    } else {
                        mask
                    }
            });
            if admissible_mask != 0 {
                let target = if exact_projection.is_some() {
                    &mut projections
                } else if vocab_alias_projection.is_some() {
                    &mut alias_projections_vocab
                } else {
                    &mut alias_projections_h64
                };
                if !target.iter().any(|(existing, mask)| {
                    std::ptr::eq(*existing, projection) && *mask == admissible_mask
                }) {
                    target.push((projection, admissible_mask));
                }
            }
        }
    }
    for (projection, _) in &projections {
        for (target, source) in buf.iter_mut().zip(projection.safe_no_match_mask.iter()) {
            *target |= *source;
        }
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_self_loop_projection] source_state={} futures={:?} safe_tokens={}",
                projection.source_state,
                projection.future_terminals,
                projection
                    .safe_no_match_mask
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>(),
            );
        }
    }
    if !root_branches.is_empty() {
        walk_interned_dynamic_trie(
            state,
            vocab,
            trie,
            &projections,
            &alias_projections_vocab,
            &alias_projections_h64,
            baseline_coverage.as_ref(),
            candidate_coverage.as_ref(),
            additive_static_baseline,
            root_branches,
            initial_config,
            &mut lexer_scan_cache,
            &mut traversal_cache,
            &mut raw_self_loop_cache,
            &mut config_self_loop_cache,
            &mut deadline_poll,
            buf,
            &mut stats,
        )?;
    }

    update_special_token_mask(state, buf);
    if let (Some(baseline), Some(candidates)) =
        (baseline_snapshot.as_deref(), effective_candidate_mask)
    {
        for (word_index, dst) in buf.iter_mut().enumerate() {
            let base = baseline.get(word_index).copied().unwrap_or(0);
            let candidate = candidates.get(word_index).copied().unwrap_or(0);
            *dst = base | ((*dst & !base) & candidate);
        }
    }
    state.clear_late_grammar_placeholder_mask(buf);
    if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_ORACLE_COVER").is_some() {
        profile_dynamic_oracle_cover(state.generation, vocab, trie, buf);
    }
    if let Some(cache_key) = cache_key {
        vocab.cache_mask(cache_key, buf);
    }
    if let Some(total_started_at) = total_started_at {
        let mask_fingerprint = buf.iter().fold(0xcbf29ce484222325u64, |hash, &word| {
            (hash ^ u64::from(word)).wrapping_mul(0x100000001b3)
        });
        let mask_popcount = buf.iter().map(|word| word.count_ones() as usize).sum::<usize>();
        if mask_popcount <= 256 {
            let mut small_tokens = Vec::<(u32, String)>::with_capacity(mask_popcount);
            for (word_index, &word) in buf.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let token_id = (word_index * 32 + bit) as u32;
                    let bytes = state
                        .constraint
                        .token_bytes
                        .get(&token_id)
                        .map(|bytes| String::from_utf8_lossy(bytes).escape_debug().to_string())
                        .unwrap_or_else(|| "<special>".to_string());
                    small_tokens.push((token_id, bytes));
                    bits &= bits - 1;
                }
            }
            eprintln!(
                "[glrmask/profile][dynamic_mask_small_tokens] generation={} tokenizer_state={:?} fingerprint={:016x} tokens={:?}",
                state.generation,
                state.state.keys().copied().collect::<SmallVec<[u32; 4]>>(),
                mask_fingerprint,
                small_tokens,
            );
        }
        let parser_interaction_hash = traversal_cache
            .profile_interaction_hash
            .unwrap_or(0);
        eprintln!(
            "[glrmask/profile][dynamic_mask] generation={} cache_hit=false key_ms={:.3} mask_fingerprint={:016x} mask_popcount={} parser_interaction_hash={:016x} parser_interaction_events={} parser_action_counts={:?} parser_child_terminals={:?} work_items={} trie_edges={} branch_steps={} duplicate_branches={} max_branches={} subtree_marks={} subtree_tokens={} bounded_attempts={} bounded_marks={} projection_reentry_marks={} projection_reentry_tokens={} config_projection_marks={} config_projection_tokens={} config_projection_candidate_edges={} config_projection_step_dead={} config_projection_normalize_dead={} config_projection_node_hits={} config_projection_guard_rejects={} config_projection_config_matches={} config_projection_parser_rejects={} recognizer_states={} recognizer_simple_one_states={} recognizer_simple_two_states={} recognizer_other_states={} recognizer_transition_misses={} boundary_cache={} relevant_cache={} child_cache={} total_ms={:.3}",
            state.generation,
            key_ms,
            mask_fingerprint,
            mask_popcount,
            parser_interaction_hash,
            traversal_cache.profile_interaction_events,
            traversal_cache.profile_parser_action_counts,
            traversal_cache.profile_parser_child_terminals,
            stats.work_items,
            stats.trie_edges,
            stats.branch_steps,
            stats.duplicate_branches,
            stats.max_branches,
            stats.subtree_marks,
            stats.subtree_mark_tokens,
            stats.bounded_subtree_attempts,
            stats.bounded_subtree_marks,
            stats.projection_reentry_marks,
            stats.projection_reentry_tokens,
            stats.config_projection_marks,
            stats.config_projection_tokens,
            stats.config_projection_candidate_edges,
            stats.config_projection_step_dead,
            stats.config_projection_normalize_dead,
            stats.config_projection_node_hits,
            stats.config_projection_guard_rejects,
            stats.config_projection_config_matches,
            stats.config_projection_parser_rejects,
            stats.recognizer_states,
            stats.recognizer_simple_one_states,
            stats.recognizer_simple_two_states,
            stats.recognizer_other_states,
            stats.recognizer_transition_misses,
            traversal_cache.admissible_terminals.len(),
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
    use crate::{DynamicConstraint, Constraint as Constraint, Vocab};
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

                for (token_id, bytes) in constraint.token_bytes_iter() {
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
    fn dynamic_nfa_cache_only_indexes_touched_raw_states() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut cache = DynamicNfaScanCache::new(&constraint, None);
        assert!(cache.raw_start_config.is_empty());

        let initial = constraint.tokenizer.initial_state();
        let config = cache.config_for_raw_start(initial).unwrap();
        assert_eq!(cache.raw_start_config.len(), 1);
        let _ = cache.step_config(config, b'a').unwrap();
        assert!(cache.raw_start_config.len() < constraint.tokenizer.num_states() as usize);
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
            ]);
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
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_trie_is_rebuilt_after_load() {
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec())]);
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
            ]);
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
            ]);
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
        assert!(state.is_accepting());
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
            ]);
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
            ]);
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
            .collect());
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
            ]);
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
            ]);
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
            ]);
        let constraint =
            Constraint::from_json_schema(r#"{"type":"string"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_matches_certified_long_terminal_run() {
        let vocab = Vocab::new(
            vec![
                (0, b"++++++++a".to_vec()),
                (1, b"++++".to_vec()),
                (2, b"a".to_vec()),
            ]);
        let grammar = r#"
start start;
t U ::= '+';
nt start ::= U* 'a';
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();

        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 0));
        assert!(token_allowed(&state.mask(), 1));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));
        state.commit_token(2).unwrap();
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
            ]);
        let constraint =
            Constraint::from_json_schema(r#"{"type":"number"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
            state.commit_bytes(bytes).unwrap();
            assert_dynamic_parity(&state);
        }
        assert!(state.is_accepting());
    }

    #[test]
    fn dynamic_mask_keeps_other_gss_paths_when_one_path_excludes_a_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
            ]);
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
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn static_and_dynamic_direct_regular_compilation_preserve_backend_contract() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
            (3, b"x".to_vec()),
            (4, b"bx".to_vec()),
        ]);
        let mut grammar = String::from(
            "start start;
t A ::= 'a' | 'ab';
t X ::= 'x';
nt start ::= A r0;
",
        );
        for index in 0..39 {
            grammar.push_str(&format!("nt r{index} ::= X r{};
", index + 1));
        }
        grammar.push_str("nt r39 ::= X;
");

        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        assert!(!constraint.possible_matches.is_empty());

        let dynamic = DynamicConstraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        assert!(dynamic.inner.uses_dynamic_runtime());
        assert!(!dynamic.inner.possible_matches_complete);

        fn delayed_query(constraint: &Constraint) -> (u32, TerminalID) {
            let execution = constraint.tokenizer.execute_from_state_all_widths(
                b"a",
                constraint.tokenizer.initial_state(),
            );
            execution
                .matches
                .iter()
                .find_map(|matched| {
                    constraint
                        .tokenizer
                        .possible_future_terminals(matched.end_state)
                        .contains(matched.id as usize)
                        .then_some((matched.end_state, matched.id))
                })
                .expect("A=a|ab must create one delayed-terminal query state")
        }

        let (compiled_state, compiled_terminal) = delayed_query(&constraint);
        let (fallback_state, fallback_terminal) = delayed_query(&dynamic.inner);
        assert_eq!(compiled_terminal, fallback_terminal);

        let mut direct_table_tokens = Vec::new();
        assert!(constraint.visit_possible_match_original_tokens(
            compiled_state,
            compiled_terminal,
            |token| direct_table_tokens.push(token),
        ));
        direct_table_tokens.sort_unstable();
        direct_table_tokens.dedup();

        let mut helper_table_tokens = Vec::new();
        for_each_token_matching_terminal_from_state(
            &constraint,
            compiled_state,
            compiled_terminal,
            |token| helper_table_tokens.push(token),
        )
        .unwrap();
        helper_table_tokens.sort_unstable();
        helper_table_tokens.dedup();

        let mut fallback_tokens = Vec::new();
        for_each_token_matching_terminal_from_state(
            &dynamic.inner,
            fallback_state,
            fallback_terminal,
            |token| fallback_tokens.push(token),
        )
        .unwrap();
        fallback_tokens.sort_unstable();
        fallback_tokens.dedup();

        assert_eq!(helper_table_tokens, direct_table_tokens);
        assert_eq!(direct_table_tokens, fallback_tokens);
        assert_eq!(direct_table_tokens, vec![2, 4]);
    }

    #[test]
    fn derived_single_use_terminal_possible_matches_match_dynamic_fallback() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"b".to_vec()),
            (5, b"ba".to_vec()),
        ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        assert_eq!(constraint.possible_matches.len(), 1);

        // Committing one accepting prefix leaves the same terminal live, so
        // the next mask exercises the delayed-terminal exclusion table that is
        // derived from the sole global-L1 transition.
        let mut after_a = constraint.start();
        after_a.commit_token(0).unwrap();
        assert!(after_a.is_accepting());
        assert!(after_a.state.values().any(|gss| {
            !gss.all_accs_satisfy(|excluded: &TerminalsDisallowed| excluded.is_empty())
        }));
        assert_dynamic_parity(&after_a);

        assert_dynamic_parity_on_reachable_states(
            &constraint,
            3,
            "derived single-use terminal possible matches",
        );
        assert!(
            constraint
                .seed_terminal_dense_fallback
                .lock()
                .expect("fallback cache poisoned")
                .is_empty(),
            "complete derived possible matches must not invoke runtime fallback",
        );
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
            ]);
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
        assert!(state.is_accepting());
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
            ]);
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
        assert!(state.is_accepting());
        assert_dynamic_parity(&state);
    }

}
