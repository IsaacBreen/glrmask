//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::hash::{Hash, Hasher};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{
    Tokenizer, TokenizerExecResult, TokenizerMatch, TokenizerStateSet,
};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{advance_stacks, stack_admissible_terminals, ParserGSS};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::LeveledGSS;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

use super::artifact::{
    Constraint, DynamicDenseSubset16, DynamicLazyUnion16Cache, DynamicLazyUnion16Metadata, DynamicMaskLexerStateKey,
    DynamicMaskStateKey, DynamicMaskTrie, DynamicMaskTrieFullWalkOp, DynamicMaskVocab,
    FastTokenizerTransitions,
};
use super::state::ConstraintState;

mod full_walk_dense;

type ParserStacks = LeveledGSS<u32, ()>;

#[cfg(test)]
thread_local! {
    static TEST_FULL_WALK_USES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_CONFIG_FULL_WALK_USES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_PARTITION_SLICER_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_PARTITION_SLICER_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
fn dynamic_partition_slicer_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var_os("GLRMASK_DISABLE_DYNAMIC_PARTITION_SLICER").is_some()
    })
}

// The bounded-code proof is not free. Tiny vocabulary partitions are cheaper
// to walk directly, especially on the first mask before the residual oracle's
// internal caches are warm. Keep the slicer for partitions large enough that
// skipping the subtree can plausibly repay the proof work.
const DYNAMIC_PARTITION_SLICER_MIN_SUBTREE_TOKENS: usize = 128;

trait FullWalkTransitionTable {
    type Cell: Copy;

    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell;

    fn cell_is_dead(cell: Self::Cell) -> bool;

    fn cell_has_finalizer(cell: Self::Cell) -> bool;

    fn cell_target(cell: Self::Cell) -> u32;

    #[inline(always)]
    fn transition(&mut self, state: u32, byte: u8) -> u32 {
        let cell = self.cell(state, byte);
        if Self::cell_is_dead(cell) {
            u32::MAX
        } else {
            Self::cell_target(cell)
        }
    }

    fn root_state(&mut self, state: u32) -> Result<u32, String>;

    fn finalizer_code(&self, state: u32) -> u32;

    fn single_finalizer_continues(&mut self, state: u32) -> bool;

    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]>;

    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool;

    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool;

    /// Merge several lexer coordinates that carry the same parser object. This
    /// is only an optimization: returning `None` keeps the branches separate.
    fn merge_states(&mut self, states: &[u32]) -> Option<u32>;

    /// Dense deterministic coordinates use a vector cache for token-boundary
    /// checks. Lazy NFA-subset coordinates use the sparse cache instead.
    fn dense_state_count(&self) -> Option<usize>;

    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool;

    /// Optional direct proof for one parser-transparent vocabulary partition.
    /// Dense finite lexer coordinates use the precomputed certificate table;
    /// the lazy config backend overrides this only for exact bounded-code
    /// virtual residual states.
    fn virtual_residual_partition_is_transparent(
        &mut self,
        _state: u32,
        _bytes: U8Set,
        _max_horizon: u32,
    ) -> Option<bool> {
        None
    }
}

#[derive(Clone, Copy)]
struct FullWalkFlat16<'a> {
    transitions: &'a [u16],
    finalizer_code: &'a [u32],
    single_finalizer_continues: &'a [u8],
    tokenizer: &'a Tokenizer,
    vocab: &'a DynamicMaskVocab,
}

impl FullWalkTransitionTable for FullWalkFlat16<'_> {
    type Cell = u16;

    #[inline(always)]
    fn cell(&mut self, state: u32, byte: u8) -> u16 {
        unsafe {
            *self
                .transitions
                .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u16) -> bool {
        cell == u16::MAX
    }

    #[inline(always)]
    fn cell_has_finalizer(cell: u16) -> bool {
        cell & 0x8000 != 0
    }

    #[inline(always)]
    fn cell_target(cell: u16) -> u32 {
        u32::from(cell & 0x7fff)
    }

    #[inline(always)]
    fn root_state(&mut self, state: u32) -> Result<u32, String> { Ok(state) }

    #[inline(always)]
    fn finalizer_code(&self, state: u32) -> u32 {
        unsafe { *self.finalizer_code.get_unchecked(state as usize) }
    }

    #[inline(always)]
    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        unsafe { *self.single_finalizer_continues.get_unchecked(state as usize) != 0 }
    }

    #[inline]
    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.tokenizer.matched_terminals_slice(state).iter().copied().collect()
    }

    #[inline(always)]
    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        self.tokenizer.possible_future_terminals(state).contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        !terminals.is_disjoint(self.tokenizer.possible_future_terminals(state))
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        self.vocab.mask_projection_state_for_projection_states(states)
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { Some(self.tokenizer.num_states() as usize) }

    #[inline(always)]
    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        parser_cache.token_boundary_allowed_dense(
            constraint,
            self.tokenizer,
            initial_lexer_state,
            lexer_state,
            parser_node,
        )
    }

}

#[derive(Clone, Copy)]
struct FullWalkFlat32<'a> {
    transitions: &'a [u32],
    finalizer_code: &'a [u32],
    single_finalizer_continues: &'a [u8],
    tokenizer: &'a Tokenizer,
    vocab: &'a DynamicMaskVocab,
}

impl FullWalkTransitionTable for FullWalkFlat32<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&mut self, state: u32, byte: u8) -> u32 {
        unsafe {
            *self
                .transitions
                .get_unchecked((state as usize).wrapping_mul(256) + byte as usize)
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: u32) -> bool {
        cell == u32::MAX
    }

    #[inline(always)]
    fn cell_has_finalizer(cell: u32) -> bool {
        cell & 0x8000_0000 != 0
    }

    #[inline(always)]
    fn cell_target(cell: u32) -> u32 {
        cell & 0x7fff_ffff
    }

    #[inline(always)]
    fn root_state(&mut self, state: u32) -> Result<u32, String> { Ok(state) }

    #[inline(always)]
    fn finalizer_code(&self, state: u32) -> u32 {
        unsafe { *self.finalizer_code.get_unchecked(state as usize) }
    }

    #[inline(always)]
    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        unsafe { *self.single_finalizer_continues.get_unchecked(state as usize) != 0 }
    }

    #[inline]
    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.tokenizer.matched_terminals_slice(state).iter().copied().collect()
    }

    #[inline(always)]
    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        self.tokenizer.possible_future_terminals(state).contains(terminal as usize)
    }

    #[inline(always)]
    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        !terminals.is_disjoint(self.tokenizer.possible_future_terminals(state))
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        self.vocab.mask_projection_state_for_projection_states(states)
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { Some(self.tokenizer.num_states() as usize) }

    #[inline(always)]
    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        parser_cache.token_boundary_allowed_dense(
            constraint,
            self.tokenizer,
            initial_lexer_state,
            lexer_state,
            parser_node,
        )
    }

}

#[derive(Clone, Copy)]
struct FullWalkConfigCell {
    target: u32,
    has_finalizer: bool,
}

/// Strict-walk transition backend for the lexer representation selected by the
/// ordinary runtime. In a deterministic tokenizer the config id is simply the
/// raw state id. In an epsilon-NFA tokenizer it is a `DynamicNfaScanCache`
/// handle for an epsilon-closed subset. Subsets are interned lazily only when
/// the complete vocabulary walk actually reaches them.
struct FullWalkConfigTransitions<'a, 'b> {
    cache: &'a mut DynamicNfaScanCache<'b>,
    error: Option<String>,
}

impl FullWalkConfigTransitions<'_, '_> {
    fn finish(self) -> Result<(), String> {
        self.error.map_or(Ok(()), Err)
    }

    fn config_future_contains_exact(&mut self, state: u32, terminal: TerminalID) -> bool {
        match self.cache.config_future_contains_exact(state, terminal) {
            Ok(value) => value,
            Err(error) => {
                if self.error.is_none() { self.error = Some(error); }
                false
            }
        }
    }

    fn config_future_intersects_exact(&mut self, state: u32, terminals: &BitSet) -> bool {
        match self.cache.config_future_intersects_exact(state, terminals) {
            Ok(value) => value,
            Err(error) => {
                if self.error.is_none() { self.error = Some(error); }
                false
            }
        }
    }
}

impl FullWalkTransitionTable for FullWalkConfigTransitions<'_, '_> {
    type Cell = FullWalkConfigCell;

    #[inline]
    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell {
        if self.error.is_some() {
            return FullWalkConfigCell { target: u32::MAX, has_finalizer: false };
        }
        match self.cache.step_config(state, byte) {
            Ok(Some(target)) => FullWalkConfigCell {
                target,
                has_finalizer: self.cache.config_has_finalizer(target),
            },
            Ok(None) => FullWalkConfigCell { target: u32::MAX, has_finalizer: false },
            Err(error) => {
                self.error = Some(error);
                FullWalkConfigCell { target: u32::MAX, has_finalizer: false }
            }
        }
    }

    #[inline(always)]
    fn cell_is_dead(cell: Self::Cell) -> bool { cell.target == u32::MAX }

    #[inline(always)]
    fn cell_has_finalizer(cell: Self::Cell) -> bool { cell.has_finalizer }

    #[inline(always)]
    fn cell_target(cell: Self::Cell) -> u32 { cell.target }

    fn root_state(&mut self, state: u32) -> Result<u32, String> {
        self.cache.config_for_raw_start(state)
    }

    fn finalizer_code(&self, state: u32) -> u32 {
        self.cache.config_finalizer_code(state)
    }

    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        let code = self.cache.config_finalizer_code(state);
        code != u32::MAX
            && code != u32::MAX - 1
            && self.config_future_contains_exact(state, code)
    }

    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        self.cache.config_matched_terminals(state)
    }

    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        self.config_future_contains_exact(state, terminal)
    }

    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        self.config_future_intersects_exact(state, terminals)
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        match self.cache.union_configs(states) {
            Ok(state) => state,
            Err(error) => {
                if self.error.is_none() { self.error = Some(error); }
                None
            }
        }
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { None }

    #[inline(always)]
    fn token_boundary_allowed(
        &mut self,
        parser_cache: &mut FullWalkParserCache,
        constraint: &Constraint,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        if lexer_state == initial_lexer_state {
            return true;
        }
        parser_cache.token_boundary_allowed_sparse(constraint, self, lexer_state, parser_node)
    }

    #[inline]
    fn virtual_residual_partition_is_transparent(
        &mut self,
        state: u32,
        bytes: U8Set,
        max_horizon: u32,
    ) -> Option<bool> {
        let raw_state = self.cache.raw_state_for_config(state)?;
        self.cache
            .tokenizer()
            .virtual_residual_parser_transparent_byte_family(raw_state, bytes, max_horizon)
    }
}

#[derive(Clone, PartialEq, Eq)]
enum FullWalkPruneGuard {
    Passed,
    Pending(SmallVec<[(u32, TerminalID); 2]>),
}

impl FullWalkPruneGuard {
    fn from_initial<T: FullWalkTransitionTable>(
        guard: &InitialPruneGuard,
        transitions: &mut T,
    ) -> Result<Self, String> {
        match guard {
            InitialPruneGuard::Passed => Ok(Self::Passed),
            InitialPruneGuard::Pending { memories } => {
                let mut projected = SmallVec::<[(u32, TerminalID); 2]>::new();
                for &(state, terminal) in memories.iter() {
                    projected.push((
                        // `InitialPruneGuard::new` already stores lexer states
                        // in the mask-runtime tokenizer coordinate. Do not
                        // project them a second time here: virtual projection
                        // state IDs are not exact source-runtime state IDs.
                        transitions.root_state(state)?,
                        terminal,
                    ));
                }
                Ok(Self::Pending(projected))
            }
        }
    }

    #[inline(always)]
    fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// Advance the maximal-munch guard in the same deterministic lexer
    /// coordinate as the direct full walk. This is only exercised by the slow
    /// side branch; the dominant scalar path always has `Passed`.
    fn advance<T: FullWalkTransitionTable>(
        &self,
        transitions: &mut T,
        byte: u8,
    ) -> Option<Self> {
        let Self::Pending(memories) = self else {
            return Some(Self::Passed);
        };
        let mut next = SmallVec::<[(u32, TerminalID); 2]>::new();
        for &(lexer_state, terminal) in memories {
            let target = transitions.transition(lexer_state, byte);
            if target == u32::MAX {
                continue;
            }
            if transitions.matched_terminals(target).contains(&terminal) {
                return None;
            }
            if transitions.future_contains(target, terminal)
                && !next.contains(&(target, terminal))
            {
                next.push((target, terminal));
            }
        }
        if next.is_empty() {
            Some(Self::Passed)
        } else {
            Some(Self::Pending(next))
        }
    }

    fn remember_terminal_match<T: FullWalkTransitionTable>(
        &self,
        transitions: &mut T,
        lexer_state: u32,
        terminal: TerminalID,
    ) -> Self {
        if !transitions.future_contains(lexer_state, terminal) {
            return self.clone();
        }
        let mut memories = match self {
            Self::Passed => SmallVec::new(),
            Self::Pending(memories) => memories.clone(),
        };
        if !memories.contains(&(lexer_state, terminal)) {
            memories.push((lexer_state, terminal));
        }
        Self::Pending(memories)
    }
}

#[derive(Clone, PartialEq, Eq)]
struct FullWalkBranch {
    lexer_state: u32,
    parser_node: u32,
    prune_guard: FullWalkPruneGuard,
}

type FullWalkBranches = SmallVec<[FullWalkBranch; 4]>;

#[derive(Clone)]
enum FullWalkManyState {
    Branches(FullWalkBranches),
    ThreeSameParser {
        lexers: (u32, u32, u32),
        parser_node: u32,
    },
}

struct FullWalkParserNode {
    gss: ParserStacks,
    admitted: Option<BitSet>,
    token_boundary_allowed: Vec<u8>,
    children: SmallVec<[(TerminalID, u32); 16]>,
    last_child_terminal: TerminalID,
    last_child_target: u32,
}

struct FullWalkParserCache {
    nodes: Vec<FullWalkParserNode>,
    dense_lexer_state_count: Option<usize>,
    sparse_token_boundary_allowed: Vec<FxHashMap<u32, u8>>,
}

impl FullWalkParserCache {
    const DEAD: u32 = u32::MAX;

    fn from_roots(
        root_branches: &DynamicBranches,
        dense_lexer_state_count: Option<usize>,
    ) -> (Self, SmallVec<[u32; 4]>) {
        let mut nodes = Vec::<FullWalkParserNode>::new();
        let mut sparse_token_boundary_allowed = Vec::<FxHashMap<u32, u8>>::new();
        let mut root_nodes = SmallVec::<[u32; 4]>::new();
        for branch in root_branches {
            if let Some((index, _)) = nodes
                .iter()
                .enumerate()
                .find(|(_, node)| node.gss.ptr_eq(&branch.gss))
            {
                root_nodes.push(index as u32);
                continue;
            }
            let id = nodes.len() as u32;
            nodes.push(FullWalkParserNode {
                gss: branch.gss.clone(),
                admitted: None,
                token_boundary_allowed: dense_lexer_state_count
                    .map_or_else(Vec::new, |count| vec![0; count]),
                children: SmallVec::new(),
                last_child_terminal: TerminalID::MAX,
                last_child_target: Self::DEAD,
            });
            sparse_token_boundary_allowed.push(FxHashMap::default());
            root_nodes.push(id);
        }
        (
            Self {
                nodes,
                dense_lexer_state_count,
                sparse_token_boundary_allowed,
            },
            root_nodes,
        )
    }

    #[inline(always)]
    fn advance(
        &mut self,
        constraint: &Constraint,
        node: u32,
        terminal: TerminalID,
    ) -> Option<u32> {
        let node_index = node as usize;
        let (last_child_terminal, last_child_target) = unsafe {
            let cached_node = self.nodes.get_unchecked(node_index);
            (cached_node.last_child_terminal, cached_node.last_child_target)
        };
        if last_child_terminal == terminal {
            let cached = last_child_target;
            return (cached != Self::DEAD).then_some(cached);
        }
        if Some(terminal) == constraint.ignore_terminal {
            return Some(node);
        }
        if let Some(&(_, cached)) = self.nodes[node_index]
            .children
            .iter()
            .find(|&&(candidate, _)| candidate == terminal)
        {
            self.nodes[node_index].last_child_terminal = terminal;
            self.nodes[node_index].last_child_target = cached;
            return (cached != Self::DEAD).then_some(cached);
        }
        // With no zero-width control terminals, a single-top parser frontier
        // whose LR row has no action for this terminal cannot advance. This is
        // exactly the first branch that the generic GLR advance would reject;
        // avoid constructing an empty-accumulator GSS and entering the GLR
        // engine for that overwhelmingly common negative lookup.
        if Some(terminal) != constraint.ignore_terminal
            && !constraint.uses_sparse_direct_regular_runtime()
            && !constraint.uses_compact_segmented_parser_runtime()
            && constraint.table.control_terminals.is_empty()
            && self.nodes[node_index]
                .gss
                .single_top_value()
                .is_some_and(|top| constraint.table.action(top, terminal).is_none())
        {
            self.nodes[node_index].children.push((terminal, Self::DEAD));
            self.nodes[node_index].last_child_terminal = terminal;
            self.nodes[node_index].last_child_target = Self::DEAD;
            return None;
        }
        let next = parser_child(constraint, &self.nodes[node_index].gss, terminal);
        let target = if let Some(gss) = next {
            let id = self.nodes.len() as u32;
            self.nodes.push(FullWalkParserNode {
                gss,
                admitted: None,
                token_boundary_allowed: self
                    .dense_lexer_state_count
                    .map_or_else(Vec::new, |count| vec![0; count]),
                children: SmallVec::new(),
                last_child_terminal: TerminalID::MAX,
                last_child_target: Self::DEAD,
            });
            self.sparse_token_boundary_allowed.push(FxHashMap::default());
            id
        } else {
            Self::DEAD
        };
        self.nodes[node_index].children.push((terminal, target));
        self.nodes[node_index].last_child_terminal = terminal;
        self.nodes[node_index].last_child_target = target;
        (target != Self::DEAD).then_some(target)
    }

    fn admitted(&mut self, constraint: &Constraint, node: u32) -> &BitSet {
        let index = node as usize;
        if self.nodes[index].admitted.is_none() {
            let parser_gss = with_empty_accumulators(&self.nodes[index].gss);
            let admitted = constraint
                .direct_regular_admissible_terminals(&parser_gss)
                .unwrap_or_else(|| {
                    let candidates = BitSet::all(constraint.table.num_terminals as usize);
                    super::commit::exact_admitted_terminals_for_candidates(
                        constraint,
                        &parser_gss,
                        &candidates,
                    )
                });
            self.nodes[index].admitted = Some(admitted);
        }
        self.nodes[index].admitted.as_ref().unwrap()
    }

    #[inline(always)]
    fn physical_token_boundary_allowed_dense(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        parser_node: u32,
        lexer_state: u32,
    ) -> bool {
        let node = parser_node as usize;
        let lexer = lexer_state as usize;
        let cached = unsafe {
            *self
                .nodes
                .get_unchecked(node)
                .token_boundary_allowed
                .get_unchecked(lexer)
        };
        if cached != 0 {
            return cached == 2;
        }
        let future = tokenizer.possible_future_terminals(lexer_state);
        let allowed = constraint
            .ignore_terminal
            .is_some_and(|terminal| future.contains(terminal as usize))
            || !self.admitted(constraint, parser_node).is_disjoint(future);
        unsafe {
            *self
                .nodes
                .get_unchecked_mut(node)
                .token_boundary_allowed
                .get_unchecked_mut(lexer) = if allowed { 2 } else { 1 };
        }
        allowed
    }

    #[inline(always)]
    fn token_boundary_allowed_dense(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        lexer_state == initial_lexer_state
            || self.physical_token_boundary_allowed_dense(
                constraint,
                tokenizer,
                parser_node,
                lexer_state,
            )
    }

    #[inline(always)]
    fn token_boundary_allowed_sparse<T: FullWalkTransitionTable>(
        &mut self,
        constraint: &Constraint,
        transitions: &mut T,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        let node = parser_node as usize;
        let cached = unsafe { self.sparse_token_boundary_allowed.get_unchecked(node) }
            .get(&lexer_state)
            .copied()
            .unwrap_or(0);
        if cached != 0 {
            return cached == 2;
        }
        let allowed = constraint.ignore_terminal.is_some_and(|terminal| {
            transitions.future_contains(lexer_state, terminal)
        }) || transitions.future_intersects(
            lexer_state,
            self.admitted(constraint, parser_node),
        );
        unsafe { self.sparse_token_boundary_allowed.get_unchecked_mut(node) }
            .insert(lexer_state, if allowed { 2 } else { 1 });
        allowed
    }

}

#[inline]
fn full_walk_push_unique(
    branches: &mut FullWalkBranches,
    branch: FullWalkBranch,
) {
    if !branches.contains(&branch) {
        branches.push(branch);
    }
}

enum FullWalkScalarFinalizerOutcome {
    Scalar(FullWalkBranch),
    Two(FullWalkBranch, FullWalkBranch),
    Many(FullWalkBranches),
}

enum FullWalkTwoStepOutcome {
    Dead,
    One((u32, u32)),
    Two((u32, u32), (u32, u32)),
    Many(FullWalkBranches),
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_step_two<T: FullWalkTransitionTable>(
    branches: ((u32, u32), (u32, u32)),
    byte: u8,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkTwoStepOutcome {
    let first_cell = transitions.cell(branches.0.0, byte);
    let second_cell = transitions.cell(branches.1.0, byte);

    // Dominant correlated-parser case: both deterministic lexer branches stay
    // alive without finalizing, while their exact parser identities differ.
    // Keep the correlation tuple intact and skip the general option/collapse
    // classification below.
    if branches.0.1 != branches.1.1
        && !T::cell_is_dead(first_cell)
        && !T::cell_is_dead(second_cell)
        && !T::cell_has_finalizer(first_cell)
        && !T::cell_has_finalizer(second_cell)
    {
        return FullWalkTwoStepOutcome::Two(
            (T::cell_target(first_cell), branches.0.1),
            (T::cell_target(second_cell), branches.1.1),
        );
    }

    // Dominant two-branch case: neither branch finalizes. Keep the whole byte
    // step as two table loads plus a tiny collapse; only the rare finalizer
    // case below constructs general branch values.
    if !T::cell_has_finalizer(first_cell) && !T::cell_has_finalizer(second_cell) {
        let first = (!T::cell_is_dead(first_cell))
            .then_some((T::cell_target(first_cell), branches.0.1));
        let second = (!T::cell_is_dead(second_cell))
            .then_some((T::cell_target(second_cell), branches.1.1));
        return match (first, second) {
            (None, None) => FullWalkTwoStepOutcome::Dead,
            (Some(branch), None) | (None, Some(branch)) => FullWalkTwoStepOutcome::One(branch),
            (Some(first), Some(second)) if first == second => FullWalkTwoStepOutcome::One(first),
            (Some(first), Some(second)) => FullWalkTwoStepOutcome::Two(first, second),
        };
    }

    full_walk_step_two_finalizing::<T>(
        branches,
        first_cell,
        second_cell,
        initial_lexer_state,
        transitions,
        parser_cache,
        constraint,
    )
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn full_walk_step_two_finalizing<T: FullWalkTransitionTable>(
    branches: ((u32, u32), (u32, u32)),
    first_cell: T::Cell,
    second_cell: T::Cell,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkTwoStepOutcome {
    let mut next = FullWalkBranches::new();
    for (cell, (source_lexer, parser_node)) in
        [(first_cell, branches.0), (second_cell, branches.1)]
    {
        if T::cell_is_dead(cell) {
            continue;
        }
        let target = T::cell_target(cell);
        if !T::cell_has_finalizer(cell) {
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: target,
                    parser_node,
                    prune_guard: FullWalkPruneGuard::Passed,
                },
            );
            continue;
        }
        let _ = source_lexer;
        match full_walk_scalar_finalizer(
            target,
            parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        ) {
            FullWalkScalarFinalizerOutcome::Scalar(branch) => {
                full_walk_push_unique(&mut next, branch);
            }
            FullWalkScalarFinalizerOutcome::Two(first, second) => {
                full_walk_push_unique(&mut next, first);
                full_walk_push_unique(&mut next, second);
            }
            FullWalkScalarFinalizerOutcome::Many(branches) => {
                for branch in branches {
                    full_walk_push_unique(&mut next, branch);
                }
            }
        }
    }
    match next.len() {
        0 => FullWalkTwoStepOutcome::Dead,
        1 if next[0].prune_guard.is_passed() => {
            let branch = next.pop().expect("one full-walk branch disappeared");
            FullWalkTwoStepOutcome::One((branch.lexer_state, branch.parser_node))
        }
        2 if next.iter().all(|branch| branch.prune_guard.is_passed()) => {
            let second = next.pop().expect("second full-walk branch disappeared");
            let first = next.pop().expect("first full-walk branch disappeared");
            FullWalkTwoStepOutcome::Two(
                (first.lexer_state, first.parser_node),
                (second.lexer_state, second.parser_node),
            )
        }
        _ => FullWalkTwoStepOutcome::Many(next),
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn full_walk_scalar_finalizer(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut impl FullWalkTransitionTable,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target);
    if code != MULTI {
        if let Some(next_parser) = parser_cache.advance(constraint, parser_node, code) {
            let reset = FullWalkBranch {
                lexer_state: initial_lexer_state,
                parser_node: next_parser,
                prune_guard: if Some(code) == constraint.ignore_terminal {
                    FullWalkPruneGuard::Passed
                } else if transitions.single_finalizer_continues(target) {
                    FullWalkPruneGuard::Pending(smallvec::smallvec![(target, code)])
                } else {
                    FullWalkPruneGuard::Passed
                },
            };
            let continuing = FullWalkBranch {
                lexer_state: target,
                parser_node,
                prune_guard: FullWalkPruneGuard::Passed,
            };
            if reset == continuing {
                return FullWalkScalarFinalizerOutcome::Scalar(continuing);
            }
            return FullWalkScalarFinalizerOutcome::Two(reset, continuing);
        }
        return FullWalkScalarFinalizerOutcome::Scalar(FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        });
    }

    let mut next = FullWalkBranches::new();
    for terminal in transitions.matched_terminals(target) {
        if let Some(next_parser) = parser_cache.advance(constraint, parser_node, terminal) {
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: initial_lexer_state,
                    parser_node: next_parser,
                    prune_guard: if Some(terminal) == constraint.ignore_terminal {
                        FullWalkPruneGuard::Passed
                    } else {
                        FullWalkPruneGuard::Passed.remember_terminal_match(
                            transitions, target, terminal,
                        )
                    },
                },
            );
        }
    }

    if next.is_empty() {
        return FullWalkScalarFinalizerOutcome::Scalar(FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        });
    }
    full_walk_push_unique(
        &mut next,
        FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        },
    );
    if next.len() == 1 && next[0].prune_guard.is_passed() {
        FullWalkScalarFinalizerOutcome::Scalar(
            next.pop().expect("one full-walk branch disappeared"),
        )
    } else {
        FullWalkScalarFinalizerOutcome::Many(next)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_scalar_finalizer_hot_single(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut impl FullWalkTransitionTable,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target);
    if code == MULTI {
        return full_walk_scalar_finalizer(
            target,
            parser_node,
            initial_lexer_state,
            transitions,
            parser_cache,
            constraint,
        );
    }
    if let Some(next_parser) = parser_cache.advance(constraint, parser_node, code) {
        let reset = FullWalkBranch {
            lexer_state: initial_lexer_state,
            parser_node: next_parser,
            prune_guard: if Some(code) == constraint.ignore_terminal {
                FullWalkPruneGuard::Passed
            } else if transitions.single_finalizer_continues(target) {
                FullWalkPruneGuard::Pending(smallvec::smallvec![(target, code)])
            } else {
                FullWalkPruneGuard::Passed
            },
        };
        let continuing = FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        };
        if reset == continuing {
            FullWalkScalarFinalizerOutcome::Scalar(continuing)
        } else {
            FullWalkScalarFinalizerOutcome::Two(reset, continuing)
        }
    } else {
        FullWalkScalarFinalizerOutcome::Scalar(FullWalkBranch {
            lexer_state: target,
            parser_node,
            prune_guard: FullWalkPruneGuard::Passed,
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn full_walk_try_apply_plain_single_finalizer(
    target: u32,
    parser_node: u32,
    initial_lexer_state: u32,
    transitions: &mut impl FullWalkTransitionTable,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    two_distinct_marker: u32,
    scalar_lexer: &mut u32,
    scalar_parser: &mut u32,
    current_two: &mut ((u32, u32), (u32, u32)),
) -> bool {
    const MULTI: u32 = u32::MAX - 1;
    let code = transitions.finalizer_code(target);
    if code == MULTI
        || (Some(code) != constraint.ignore_terminal
            && transitions.single_finalizer_continues(target))
    {
        return false;
    }

    let Some(next_parser) = parser_cache.advance(constraint, parser_node, code) else {
        *scalar_lexer = target;
        *scalar_parser = parser_node;
        return true;
    };

    if next_parser != parser_node {
        *scalar_lexer = two_distinct_marker;
        *current_two = (
            (initial_lexer_state, next_parser),
            (target, parser_node),
        );
        return true;
    }

    // If both exact coordinates are identical there is only one branch. When
    // the parser is the same but lexer coordinates differ, leave the rare case
    // to the existing exact union logic below.
    if initial_lexer_state == target {
        *scalar_lexer = target;
        *scalar_parser = parser_node;
        return true;
    }
    false
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn full_walk_step_many<T: FullWalkTransitionTable>(
    branches: &FullWalkBranches,
    byte: u8,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkBranches {
    const NONE: u32 = u32::MAX;
    const MULTI: u32 = u32::MAX - 1;
    let mut next = FullWalkBranches::new();
    for branch in branches {
        let Some(advanced_guard) = branch
            .prune_guard
            .advance(transitions, byte)
        else {
            continue;
        };
        let target = transitions.transition(branch.lexer_state, byte);
        if target == u32::MAX {
            continue;
        }
        let code = transitions.finalizer_code(target);
        if code == MULTI {
            for terminal in transitions.matched_terminals(target) {
                if let Some(parser_node) = parser_cache.advance(
                    constraint, branch.parser_node, terminal,
                ) {
                    let matched_guard = if Some(terminal) == constraint.ignore_terminal {
                        advanced_guard.clone()
                    } else {
                        advanced_guard.remember_terminal_match(transitions, target, terminal)
                    };
                    full_walk_push_unique(
                        &mut next,
                        FullWalkBranch {
                            lexer_state: initial_lexer_state,
                            parser_node,
                            prune_guard: matched_guard,
                        },
                    );
                }
            }
        } else if code != NONE
            && let Some(parser_node) = parser_cache.advance(
                constraint, branch.parser_node, code,
            )
        {
            let matched_guard = if Some(code) == constraint.ignore_terminal {
                advanced_guard.clone()
            } else {
                advanced_guard.remember_terminal_match(transitions, target, code)
            };
            full_walk_push_unique(
                &mut next,
                FullWalkBranch {
                    lexer_state: initial_lexer_state,
                    parser_node,
                    prune_guard: matched_guard,
                },
            );
        }
        full_walk_push_unique(
            &mut next,
            FullWalkBranch {
                lexer_state: target,
                parser_node: branch.parser_node,
                prune_guard: advanced_guard,
            },
        );
    }
    next
}

#[inline]
fn full_walk_projection_union_two<T: FullWalkTransitionTable>(
    transitions: &mut T,
    cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    first: u32,
    second: u32,
) -> Option<u32> {
    let key = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    if let Some(&cached) = cache.get(&key) {
        return cached;
    }
    let result = transitions.merge_states(&[key.0, key.1]);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_projection_union_three<T: FullWalkTransitionTable>(
    transitions: &mut T,
    cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    first: u32,
    second: u32,
    third: u32,
) -> Option<u32> {
    let mut states = [first, second, third];
    states.sort_unstable();
    let key = (states[0], states[1], states[2]);
    if let Some(&cached) = cache.get(&key) {
        return cached;
    }
    let result = transitions.merge_states(&states);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_merge_two_same_parser<T: FullWalkTransitionTable>(
    transitions: &mut T,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    first: (u32, u32),
    second: (u32, u32),
) -> Option<(u32, u32)> {
    if first.1 != second.1 {
        return None;
    }
    full_walk_projection_union_two(transitions, pair_union_cache, first.0, second.0)
        .map(|lexer_state| (lexer_state, first.1))
}

#[inline]
fn full_walk_merge_three_same_parser<T: FullWalkTransitionTable>(
    transitions: &mut T,
    triple_union_cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    lexers: (u32, u32, u32),
    parser_node: u32,
) -> Option<(u32, u32)> {
    full_walk_projection_union_three(
        transitions,
        triple_union_cache,
        lexers.0,
        lexers.1,
        lexers.2,
    )
    .map(|lexer_state| (lexer_state, parser_node))
}

#[inline]
fn full_walk_merge_branches_same_parser<T: FullWalkTransitionTable>(
    transitions: &mut T,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    triple_union_cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    branches: &FullWalkBranches,
) -> Option<(u32, u32)> {
    let first = branches.first()?;
    if branches.len() < 2
        || !first.prune_guard.is_passed()
        || branches.iter().skip(1).any(|branch| {
            !branch.prune_guard.is_passed() || branch.parser_node != first.parser_node
        })
    {
        return None;
    }
    let lexer_state = match branches.as_slice() {
        [first, second] => full_walk_projection_union_two(
            transitions,
            pair_union_cache,
            first.lexer_state,
            second.lexer_state,
        ),
        [first, second, third] => full_walk_projection_union_three(
            transitions,
            triple_union_cache,
            first.lexer_state,
            second.lexer_state,
            third.lexer_state,
        ),
        _ => {
            let lexers = branches
                .iter()
                .map(|branch| branch.lexer_state)
                .collect::<SmallVec<[u32; 4]>>();
            transitions.merge_states(&lexers)
        }
    }?;
    Some((lexer_state, first.parser_node))
}

#[inline]
fn full_walk_many_state_from_branches(branches: FullWalkBranches) -> FullWalkManyState {
    if let [first, second, third] = branches.as_slice()
        && first.prune_guard.is_passed()
        && second.prune_guard.is_passed()
        && third.prune_guard.is_passed()
        && first.parser_node == second.parser_node
        && first.parser_node == third.parser_node
    {
        return FullWalkManyState::ThreeSameParser {
            lexers: (first.lexer_state, second.lexer_state, third.lexer_state),
            parser_node: first.parser_node,
        };
    }
    FullWalkManyState::Branches(branches)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn full_walk_step_many_state<T: FullWalkTransitionTable>(
    state: &FullWalkManyState,
    byte: u8,
    initial_lexer_state: u32,
    transitions: &mut T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkManyState {
    match state {
        FullWalkManyState::Branches(branches) => full_walk_many_state_from_branches(
            full_walk_step_many(
                branches,
                byte,
                initial_lexer_state,
                transitions,
                parser_cache,
                constraint,
            ),
        ),
        FullWalkManyState::ThreeSameParser {
            lexers,
            parser_node,
        } => {
            let first = transitions.cell(lexers.0, byte);
            let second = transitions.cell(lexers.1, byte);
            let third = transitions.cell(lexers.2, byte);
            if !T::cell_has_finalizer(first)
                && !T::cell_has_finalizer(second)
                && !T::cell_has_finalizer(third)
                && !T::cell_is_dead(first)
                && !T::cell_is_dead(second)
                && !T::cell_is_dead(third)
            {
                let next = (
                    T::cell_target(first),
                    T::cell_target(second),
                    T::cell_target(third),
                );
                if next.0 != next.1 && next.0 != next.2 && next.1 != next.2 {
                    return FullWalkManyState::ThreeSameParser {
                        lexers: next,
                        parser_node: *parser_node,
                    };
                }
            }

            let mut branches = FullWalkBranches::new();
            for lexer_state in [lexers.0, lexers.1, lexers.2] {
                branches.push(FullWalkBranch {
                    lexer_state,
                    parser_node: *parser_node,
                    prune_guard: FullWalkPruneGuard::Passed,
                });
            }
            full_walk_many_state_from_branches(full_walk_step_many(
                &branches,
                byte,
                initial_lexer_state,
                transitions,
                parser_cache,
                constraint,
            ))
        }
    }
}



/// Exact direct dynamic-mask path for the complete vocabulary.
///
/// This deliberately performs the complete vocabulary walk. It does not use
/// subtree certificates, segment-effect caches, recognizer-state interning, or
/// any other mechanism that can omit vocabulary edges. Deterministic lexers use
/// dense Flat16/Flat32 rows when available. Epsilon-NFA and oversized/sparse
/// coordinates use the same walk over lazily interned runtime configurations.
/// The same complete walk is also used for composed constraints. Composition
/// callers that already hold a static A baseline compute the complete exact
/// dynamic language into scratch and combine it after the walk; the walker
/// itself therefore does not need the legacy `repair_used`/component tracking
/// that existed solely to traverse B-minus-A.
fn try_full_walk_mask(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
) -> Result<bool, String> {
    if root_branches.is_empty() {
        buf.fill(0);
        update_special_token_mask(state, buf);
        state.clear_late_grammar_placeholder_mask(buf);
        #[cfg(test)]
        TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
        return Ok(true);
    }
    let tokenizer = lexer_scan_cache.tokenizer();
    match vocab.mask_projection_fast_transitions() {
        Some(FastTokenizerTransitions::Flat16 {
            transitions,
            finalizer_code,
            single_finalizer_continues,
        }) if lexer_scan_cache.deterministic => {
            if trie.full_walk_max_parent_depth() < 255 {
                let result = if root_branches.len() == 1 {
                    full_walk_dense::try_flat16::<true>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                } else {
                    full_walk_dense::try_flat16::<false>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                };
                #[cfg(test)]
                if matches!(result, Ok(true)) {
                    TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
                }
                return result;
            }
            let mut table = FullWalkFlat16 {
                transitions: transitions.as_ref(),
                finalizer_code: finalizer_code.as_ref(),
                single_finalizer_continues: single_finalizer_continues.as_ref(),
                tokenizer,
                vocab,
            };
            if root_branches.len() == 1 {
                try_full_walk_mask_with_table::<_, true>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            } else {
                try_full_walk_mask_with_table::<_, false>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            }
        }
        Some(FastTokenizerTransitions::Flat32 {
            transitions,
            finalizer_code,
            single_finalizer_continues,
        }) if lexer_scan_cache.deterministic => {
            if trie.full_walk_max_parent_depth() < 255 {
                let result = if root_branches.len() == 1 {
                    full_walk_dense::try_flat32::<true>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                } else {
                    full_walk_dense::try_flat32::<false>(
                        state,
                        vocab,
                        trie,
                        root_branches,
                        lexer_scan_cache,
                        buf,
                        transitions.as_ref(),
                        finalizer_code.as_ref(),
                        single_finalizer_continues.as_ref(),
                    )
                };
                #[cfg(test)]
                if matches!(result, Ok(true)) {
                    TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
                }
                return result;
            }
            let mut table = FullWalkFlat32 {
                transitions: transitions.as_ref(),
                finalizer_code: finalizer_code.as_ref(),
                single_finalizer_continues: single_finalizer_continues.as_ref(),
                tokenizer,
                vocab,
            };
            if root_branches.len() == 1 {
                try_full_walk_mask_with_table::<_, true>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            } else {
                try_full_walk_mask_with_table::<_, false>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            }
        }
        _ => {
            let mut table = FullWalkConfigTransitions {
                cache: lexer_scan_cache,
                error: None,
            };
            let result = if root_branches.len() == 1 {
                try_full_walk_mask_with_table::<_, true>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            } else {
                try_full_walk_mask_with_table::<_, false>(
                    state,
                    vocab,
                    trie,
                    root_branches,
                    buf,
                    &mut table,
                )
            };
            match result {
                Ok(used) => {
                    table.finish()?;
                    #[cfg(test)]
                    if used {
                        TEST_CONFIG_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
                    }
                    Ok(used)
                }
                Err(error) => Err(error),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn try_full_walk_mask_with_table<T: FullWalkTransitionTable, const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    buf: &mut [u32],
    transitions: &mut T,
) -> Result<bool, String> {

    let initial_lexer_state = transitions
        .root_state(vocab.mask_runtime_state(state.constraint.tokenizer.initial_state()))?;

    let all_words = vocab.all_original_token_words();
    let copy_len = buf.len().min(all_words.len());
    buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
    if copy_len < buf.len() { buf[copy_len..].fill(0); }

    let (mut parser_cache, root_parser_nodes) = FullWalkParserCache::from_roots(
        root_branches,
        transitions.dense_state_count(),
    );
    // Scalar is overwhelmingly dominant. Encode dead/multi directly in the
    // lexer-state coordinate so the common DFS path needs no separate kind
    // load/store. Full-walk lexer states are bounded far below these u32
    // sentinels by the dense-transition memory budget.
    const FULL_WALK_LEXER_TWO_DISTINCT: u32 = u32::MAX - 3;
    const FULL_WALK_LEXER_TWO: u32 = u32::MAX - 2;
    const FULL_WALK_LEXER_MULTI: u32 = u32::MAX - 1;
    const FULL_WALK_LEXER_DEAD: u32 = u32::MAX;
    let stack_len = usize::from(trie.full_walk_max_parent_depth()).saturating_add(2);
    // Keep the overwhelmingly common <=256-depth vocabulary on the original
    // fixed-array storage.  `SmallVec` here looks attractive, but every
    // `get_unchecked` still has to resolve inline-vs-spilled storage and that
    // showed up measurably across a complete 128k-token walk.  Deep tries pay
    // one allocation up front and then use the same plain-slice inner loop.
    let mut inline_lexer = [FULL_WALK_LEXER_DEAD; 256];
    let mut heap_lexer = Vec::<u32>::new();
    let stack_lexer: &mut [u32] = if stack_len <= inline_lexer.len() {
        &mut inline_lexer[..stack_len]
    } else {
        heap_lexer.resize(stack_len, FULL_WALK_LEXER_DEAD);
        heap_lexer.as_mut_slice()
    };
    let mut inline_parser = [0u32; 256];
    let mut heap_parser = Vec::<u32>::new();
    let stack_parser: &mut [u32] = if stack_len <= inline_parser.len() {
        &mut inline_parser[..stack_len]
    } else {
        heap_parser.resize(stack_len, 0);
        heap_parser.as_mut_slice()
    };
    let mut inline_two = [((0u32, 0u32), (0u32, 0u32)); 256];
    let mut heap_two = Vec::<((u32, u32), (u32, u32))>::new();
    let stack_two: &mut [((u32, u32), (u32, u32))] = if stack_len <= inline_two.len() {
        &mut inline_two[..stack_len]
    } else {
        heap_two.resize(stack_len, ((0, 0), (0, 0)));
        heap_two.as_mut_slice()
    };
    // Multi-branch stack states are uncommon, and `FullWalkManyState` embeds a
    // SmallVec. Do not eagerly construct/drop 256 empty SmallVec values on
    // every complete vocabulary walk; materialize only the depths that
    // actually carry a multi state.
    let mut inline_many: [Option<FullWalkManyState>; 256] = std::array::from_fn(|_| None);
    let mut heap_many = Vec::<Option<FullWalkManyState>>::new();
    let stack_many: &mut [Option<FullWalkManyState>] = if stack_len <= inline_many.len() {
        &mut inline_many[..stack_len]
    } else {
        heap_many.resize_with(stack_len, || None);
        heap_many.as_mut_slice()
    };
    let mut pair_union_cache = FxHashMap::<(u32, u32), Option<u32>>::default();
    let mut triple_union_cache = FxHashMap::<(u32, u32, u32), Option<u32>>::default();
    if root_branches.len() == 1 && root_branches[0].initial_prune_guard.is_passed() {
        stack_lexer[0] = root_branches[0].tokenizer_config;
        stack_parser[0] = root_parser_nodes[0];
    } else if root_branches.len() == 2
        && root_branches.iter().all(|root| root.initial_prune_guard.is_passed())
    {
        let first = (root_branches[0].tokenizer_config, root_parser_nodes[0]);
        let second = (root_branches[1].tokenizer_config, root_parser_nodes[1]);
        if let Some((lexer_state, parser_node)) =
            full_walk_merge_two_same_parser(transitions, &mut pair_union_cache, first, second)
        {
            stack_lexer[0] = lexer_state;
            stack_parser[0] = parser_node;
        } else {
            stack_lexer[0] = if first.1 != second.1 {
                FULL_WALK_LEXER_TWO_DISTINCT
            } else {
                FULL_WALK_LEXER_TWO
            };
            stack_two[0] = (first, second);
        }
    } else {
        stack_lexer[0] = FULL_WALK_LEXER_MULTI;
        let mut roots = FullWalkBranches::new();
        for (root_index, root) in root_branches.iter().enumerate() {
            full_walk_push_unique(
                &mut roots,
                FullWalkBranch {
                    lexer_state: root.tokenizer_config,
                    parser_node: root_parser_nodes[root_index],
                    prune_guard: FullWalkPruneGuard::from_initial(
                        &root.initial_prune_guard,
                        transitions,
                    )?,
                },
            );
        }
        if let Some((lexer_state, parser_node)) = full_walk_merge_branches_same_parser(
            transitions,
            &mut pair_union_cache,
            &mut triple_union_cache,
            &roots,
        )
        {
            stack_lexer[0] = lexer_state;
            stack_parser[0] = parser_node;
        } else {
            stack_many[0] = Some(full_walk_many_state_from_branches(roots));
        }
    }

    let walk_ops = trie.full_walk_ops();
    let token_markers = vocab.full_walk_token_markers();
    let mut token_marker_index = 0usize;
    let mut scalar_lexer = FULL_WALK_LEXER_DEAD;
    let mut scalar_parser = 0u32;
    let mut current_two = ((0u32, 0u32), (0u32, 0u32));
    let mut current_many = FullWalkManyState::Branches(FullWalkBranches::new());
    let profile_partition_slicer =
        std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PARTITION_SLICER").is_some();
    let mut partition_slicer_hits = 0usize;
    let mut partition_slicer_tokens = 0usize;
    let mut partition_root_slot = 0usize;
    let root_edges = trie.children(0);
    #[cfg(test)]
    let partition_slicer_disabled = dynamic_partition_slicer_disabled()
        || TEST_PARTITION_SLICER_DISABLED.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let partition_slicer_disabled = dynamic_partition_slicer_disabled();
    let mut remaining_ops = walk_ops.iter();

    while let Some(&op) = remaining_ops.next() {
        let parent_depth = op.parent_depth() as usize;
        if op.starts_edge() {
            scalar_lexer = unsafe { *stack_lexer.get_unchecked(parent_depth) };
            if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                scalar_parser = unsafe { *stack_parser.get_unchecked(parent_depth) };
            } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                current_two = unsafe { *stack_two.get_unchecked(parent_depth) };
            } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                current_many.clone_from(unsafe {
                    stack_many
                        .get_unchecked(parent_depth)
                        .as_ref()
                        .unwrap_unchecked()
                });
            }

            if !partition_slicer_disabled && parent_depth == 0 && !op.consumes_byte() {
                let root_slot = partition_root_slot;
                partition_root_slot += 1;
                if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    let directly_certified = if let Some(edge) = root_edges.get(root_slot) {
                        trie.subtree_tokens(edge.child).len()
                            >= DYNAMIC_PARTITION_SLICER_MIN_SUBTREE_TOKENS
                            && transitions
                                .virtual_residual_partition_is_transparent(
                                    scalar_lexer,
                                    U8Set::from_words(trie.subtree_bytes(edge.child)),
                                    trie.subtree_max_byte_len(edge.child),
                                )
                                .unwrap_or(false)
                    } else {
                        false
                    };
                    if directly_certified
                        && parser_cache.token_boundary_allowed_sparse(
                            state.constraint,
                            transitions,
                            scalar_lexer,
                            scalar_parser,
                        )
                    {
                        partition_slicer_hits += 1;
                        #[cfg(test)]
                        TEST_PARTITION_SLICER_HITS.with(|hits| hits.set(hits.get() + 1));
                        partition_slicer_tokens += full_walk_skip_admitted_subtree_generic(
                            trie,
                            walk_ops,
                            &mut remaining_ops,
                            &mut token_marker_index,
                        );
                        continue;
                    }
                }
            }
        }

        if op.consumes_byte() {
            let byte = op.byte();
            if scalar_lexer == FULL_WALK_LEXER_DEAD {
            } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                let cell = transitions.cell(scalar_lexer, byte);
                if T::cell_is_dead(cell) {
                    scalar_lexer = FULL_WALK_LEXER_DEAD;
                } else {
                    let target = T::cell_target(cell);
                    if !T::cell_has_finalizer(cell) {
                        scalar_lexer = target;
                    } else {
                        let direct_applied = HOT_SINGLE_ROOT
                            && full_walk_try_apply_plain_single_finalizer(
                                target,
                                scalar_parser,
                                initial_lexer_state,
                                transitions,
                                &mut parser_cache,
                                state.constraint,
                                FULL_WALK_LEXER_TWO_DISTINCT,
                                &mut scalar_lexer,
                                &mut scalar_parser,
                                &mut current_two,
                            );
                        if !direct_applied {
                            let outcome = if HOT_SINGLE_ROOT {
                                full_walk_scalar_finalizer_hot_single(
                                    target,
                                    scalar_parser,
                                    initial_lexer_state,
                                    transitions,
                                    &mut parser_cache,
                                    state.constraint,
                                )
                            } else {
                                full_walk_scalar_finalizer(
                                    target,
                                    scalar_parser,
                                    initial_lexer_state,
                                    transitions,
                                    &mut parser_cache,
                                    state.constraint,
                                )
                            };
                            match outcome {
                            FullWalkScalarFinalizerOutcome::Scalar(branch) => {
                                scalar_lexer = branch.lexer_state;
                                scalar_parser = branch.parser_node;
                            }
                            FullWalkScalarFinalizerOutcome::Two(first, second) => {
                                if first.prune_guard.is_passed() && second.prune_guard.is_passed() {
                                    if let Some((lexer_state, parser_node)) =
                                        full_walk_merge_two_same_parser(
                                            transitions,
                                            &mut pair_union_cache,
                                            (first.lexer_state, first.parser_node),
                                            (second.lexer_state, second.parser_node),
                                        )
                                    {
                                        scalar_lexer = lexer_state;
                                        scalar_parser = parser_node;
                                    } else {
                                        scalar_lexer = if first.parser_node != second.parser_node {
                                            FULL_WALK_LEXER_TWO_DISTINCT
                                        } else {
                                            FULL_WALK_LEXER_TWO
                                        };
                                        current_two = (
                                            (first.lexer_state, first.parser_node),
                                            (second.lexer_state, second.parser_node),
                                        );
                                    }
                                } else {
                                    let mut next = FullWalkBranches::new();
                                    next.push(first);
                                    next.push(second);
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = full_walk_many_state_from_branches(next);
                                }
                            }
                            FullWalkScalarFinalizerOutcome::Many(next) => {
                                if let [first, second] = next.as_slice() {
                                    if first.prune_guard.is_passed() && second.prune_guard.is_passed() {
                                        if let Some((lexer_state, parser_node)) =
                                            full_walk_merge_two_same_parser(
                                                transitions,
                                                &mut pair_union_cache,
                                                (first.lexer_state, first.parser_node),
                                                (second.lexer_state, second.parser_node),
                                            )
                                        {
                                            scalar_lexer = lexer_state;
                                            scalar_parser = parser_node;
                                        } else {
                                            scalar_lexer = if first.parser_node != second.parser_node {
                                                FULL_WALK_LEXER_TWO_DISTINCT
                                            } else {
                                                FULL_WALK_LEXER_TWO
                                            };
                                            current_two = (
                                                (first.lexer_state, first.parser_node),
                                                (second.lexer_state, second.parser_node),
                                            );
                                        }
                                    } else {
                                        scalar_lexer = FULL_WALK_LEXER_MULTI;
                                        current_many = full_walk_many_state_from_branches(next);
                                    }
                                } else if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_branches_same_parser(
                                        transitions,
                                        &mut pair_union_cache,
                                        &mut triple_union_cache,
                                        &next,
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = full_walk_many_state_from_branches(next);
                                }
                            }
                            }
                        }
                    }
                }
            } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT {
                let first_cell = transitions.cell(current_two.0.0, byte);
                let second_cell = transitions.cell(current_two.1.0, byte);
                if !T::cell_is_dead(first_cell)
                    && !T::cell_is_dead(second_cell)
                    && !T::cell_has_finalizer(first_cell)
                    && !T::cell_has_finalizer(second_cell)
                {
                    current_two.0.0 = T::cell_target(first_cell);
                    current_two.1.0 = T::cell_target(second_cell);
                } else {
                    match full_walk_step_two::<T>(
                        current_two,
                        byte,
                        initial_lexer_state,
                        transitions,
                        &mut parser_cache,
                        state.constraint,
                    ) {
                        FullWalkTwoStepOutcome::Dead => scalar_lexer = FULL_WALK_LEXER_DEAD,
                        FullWalkTwoStepOutcome::One((lexer, parser)) => {
                            scalar_lexer = lexer;
                            scalar_parser = parser;
                        }
                        FullWalkTwoStepOutcome::Two(first, second) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_two_same_parser(
                                    transitions,
                                    &mut pair_union_cache,
                                    first,
                                    second,
                                )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = if first.1 != second.1 {
                                    FULL_WALK_LEXER_TWO_DISTINCT
                                } else {
                                    FULL_WALK_LEXER_TWO
                                };
                                current_two = (first, second);
                            }
                        }
                        FullWalkTwoStepOutcome::Many(next) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_branches_same_parser(
                                            transitions,
                                            &mut pair_union_cache,
                                            &mut triple_union_cache,
                                            &next,
                                        )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = FULL_WALK_LEXER_MULTI;
                                current_many = full_walk_many_state_from_branches(next);
                            }
                        }
                    }
                }
            } else if scalar_lexer == FULL_WALK_LEXER_TWO {
                match full_walk_step_two::<T>(
                        current_two,
                        byte,
                        initial_lexer_state,
                        transitions,
                        &mut parser_cache,
                        state.constraint,
                    ) {
                        FullWalkTwoStepOutcome::Dead => scalar_lexer = FULL_WALK_LEXER_DEAD,
                        FullWalkTwoStepOutcome::One((lexer, parser)) => {
                            scalar_lexer = lexer;
                            scalar_parser = parser;
                        }
                        FullWalkTwoStepOutcome::Two(first, second) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_two_same_parser(
                                    transitions,
                                    &mut pair_union_cache,
                                    first,
                                    second,
                                )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = if first.1 != second.1 {
                                    FULL_WALK_LEXER_TWO_DISTINCT
                                } else {
                                    FULL_WALK_LEXER_TWO
                                };
                                current_two = (first, second);
                            }
                        }
                        FullWalkTwoStepOutcome::Many(next) => {
                            if let Some((lexer_state, parser_node)) =
                                full_walk_merge_branches_same_parser(
                                            transitions,
                                            &mut pair_union_cache,
                                            &mut triple_union_cache,
                                            &next,
                                        )
                            {
                                scalar_lexer = lexer_state;
                                scalar_parser = parser_node;
                            } else {
                                scalar_lexer = FULL_WALK_LEXER_MULTI;
                                current_many = full_walk_many_state_from_branches(next);
                            }
                        }
                    }
            } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                let next = full_walk_step_many_state(
                    &current_many,
                    byte,
                    initial_lexer_state,
                    transitions,
                    &mut parser_cache,
                    state.constraint,
                );
                match next {
                    FullWalkManyState::Branches(next) => {
                        match next.as_slice() {
                            [] => scalar_lexer = FULL_WALK_LEXER_DEAD,
                            [branch] if branch.prune_guard.is_passed() => {
                                scalar_lexer = branch.lexer_state;
                                scalar_parser = branch.parser_node;
                            }
                            [first, second]
                                if first.prune_guard.is_passed() && second.prune_guard.is_passed() =>
                            {
                                if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_two_same_parser(
                                        transitions,
                                        &mut pair_union_cache,
                                        (first.lexer_state, first.parser_node),
                                        (second.lexer_state, second.parser_node),
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    scalar_lexer = if first.parser_node != second.parser_node {
                                        FULL_WALK_LEXER_TWO_DISTINCT
                                    } else {
                                        FULL_WALK_LEXER_TWO
                                    };
                                    current_two = (
                                        (first.lexer_state, first.parser_node),
                                        (second.lexer_state, second.parser_node),
                                    );
                                }
                            }
                            _ => {
                                if let Some((lexer_state, parser_node)) =
                                    full_walk_merge_branches_same_parser(
                                        transitions,
                                        &mut pair_union_cache,
                                        &mut triple_union_cache,
                                        &next,
                                    )
                                {
                                    scalar_lexer = lexer_state;
                                    scalar_parser = parser_node;
                                } else {
                                    scalar_lexer = FULL_WALK_LEXER_MULTI;
                                    current_many = FullWalkManyState::Branches(next);
                                }
                            }
                        }
                    }
                    next @ FullWalkManyState::ThreeSameParser { lexers, parser_node } => {
                        if let Some((lexer_state, parser_node)) =
                            full_walk_merge_three_same_parser(
                                transitions,
                                &mut triple_union_cache,
                                lexers,
                                parser_node,
                            )
                        {
                            scalar_lexer = lexer_state;
                            scalar_parser = parser_node;
                        } else {
                            scalar_lexer = FULL_WALK_LEXER_MULTI;
                            current_many = next;
                        }
                    }
                }
            }
        }

        if op.ends_edge() {
            if op.child_is_token() {
                let token_marker = unsafe { *token_markers.get_unchecked(token_marker_index) };
                token_marker_index += 1;
                let allowed = if scalar_lexer == FULL_WALK_LEXER_DEAD {
                    false
                } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    transitions.token_boundary_allowed(
                        &mut parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        scalar_lexer,
                        scalar_parser,
                    )
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                    transitions.token_boundary_allowed(
                        &mut parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        current_two.0.0,
                        current_two.0.1,
                    ) || transitions.token_boundary_allowed(
                        &mut parser_cache,
                        state.constraint,
                        initial_lexer_state,
                        current_two.1.0,
                        current_two.1.1,
                    )
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    match &current_many {
                        FullWalkManyState::Branches(branches) => branches.iter().any(|branch| {
                            transitions.token_boundary_allowed(
                                &mut parser_cache,
                                state.constraint,
                                initial_lexer_state,
                                branch.lexer_state,
                                branch.parser_node,
                            )
                        }),
                        FullWalkManyState::ThreeSameParser {
                            lexers,
                            parser_node,
                        } => transitions.token_boundary_allowed(
                            &mut parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            lexers.0,
                            *parser_node,
                        ) || transitions.token_boundary_allowed(
                            &mut parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            lexers.1,
                            *parser_node,
                        ) || transitions.token_boundary_allowed(
                            &mut parser_cache,
                            state.constraint,
                            initial_lexer_state,
                            lexers.2,
                            *parser_node,
                        ),
                    }
                } else {
                    false
                };
                if !allowed {
                    clear_dynamic_token_marker(vocab, token_marker, buf);
                }
            }

            unsafe {
                *stack_lexer.get_unchecked_mut(parent_depth + 1) = scalar_lexer;
                if scalar_lexer == FULL_WALK_LEXER_DEAD {
                } else if scalar_lexer < FULL_WALK_LEXER_TWO_DISTINCT {
                    *stack_parser.get_unchecked_mut(parent_depth + 1) = scalar_parser;
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                    *stack_two.get_unchecked_mut(parent_depth + 1) = current_two;
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    let slot = stack_many.get_unchecked_mut(parent_depth + 1);
                    if let Some(existing) = slot.as_mut() {
                        existing.clone_from(&current_many);
                    } else {
                        *slot = Some(current_many.clone());
                    }
                }
            }
        }
    }
    // Ordinary vocabulary bytes and exact special-token-ID paths are a union.
    // The strict walk above computes the byte-language contribution for every
    // model token; the existing special-token routine then ORs in token-ID-only
    // paths. This also handles a token ID that is valid through both routes.
    update_special_token_mask(state, buf);
    state.clear_late_grammar_placeholder_mask(buf);
    if profile_partition_slicer {
        eprintln!(
            "[glrmask/profile][dynamic_partition_slicer] hits={} skipped_tokens={}",
            partition_slicer_hits, partition_slicer_tokens,
        );
    }
    #[cfg(test)]
    TEST_FULL_WALK_USES.with(|count| count.set(count.get() + 1));
    Ok(true)
}

#[inline(always)]
fn full_walk_skip_admitted_subtree_generic<'a>(
    trie: &DynamicMaskTrie,
    walk_ops: &'a [DynamicMaskTrieFullWalkOp],
    remaining_ops: &mut std::slice::Iter<'a, DynamicMaskTrieFullWalkOp>,
    token_marker_index: &mut usize,
) -> usize {
    let op_index = walk_ops.len() - remaining_ops.as_slice().len() - 1;
    let (child, subtree_end_op) = trie.full_walk_dead_subtree(op_index);
    let token_count = trie.subtree_tokens(child).len();
    let root_token_offset = usize::from(trie.node(0).token_id.is_some());
    let token_end = trie
        .subtree_token_index_range(child)
        .end
        .saturating_sub(root_token_offset);
    debug_assert!(*token_marker_index <= token_end);
    *token_marker_index = token_end;
    *remaining_ops = walk_ops[subtree_end_op as usize..].iter();
    token_count
}

#[derive(Clone)]
struct DynamicBranch {
    tokenizer_config: u32,
    gss: ParserStacks,
    initial_prune_guard: InitialPruneGuard,
}

type DynamicBranches = SmallVec<[DynamicBranch; 4]>;

#[derive(Clone, PartialEq, Eq, Hash)]
enum InitialPruneGuard {
    Passed,
    Pending {
        memories: Arc<[(u32, TerminalID)]>,
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

#[inline(always)]
fn clear_mask_bit_known_in_range(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    debug_assert!(word < buf.len());
    unsafe { *buf.get_unchecked_mut(word) &= !(1u32 << bit); }
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

#[inline(always)]
fn clear_dynamic_token_marker(vocab: &DynamicMaskVocab, marker: u64, buf: &mut [u32]) {
    debug_assert_ne!(marker, 0);
    if marker & DYNAMIC_TOKEN_MARKER_FALLBACK == 0 {
        let word = (marker >> 32) as usize;
        let bits = marker as u32;
        debug_assert_ne!(bits, 0);
        debug_assert!(word < buf.len());
        unsafe { *buf.get_unchecked_mut(word) &= !bits; }
        return;
    }
    let canonical_token = ((marker & !DYNAMIC_TOKEN_MARKER_FALLBACK) - 1) as u32;
    let token_ids = vocab.token_ids(canonical_token).expect("dynamic vocabulary trie node lacks token ids");
    for &token_id in token_ids { clear_mask_bit_known_in_range(buf, token_id); }
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
// The strict full walker uses the four top u32 values as branch-kind/dead
// sentinels. Tagged raw NFA configs share that high half of the u32 namespace,
// so keep those four values permanently outside the raw-config domain. This is
// also stricter than the old cache contract, which already could not represent
// raw states whose tagged IDs became UNKNOWN or DEAD.
const DYNAMIC_NFA_RAW_STATE_LIMIT: u32 = DYNAMIC_NFA_RAW_CONFIG_TAG - 4;

#[derive(Clone)]
struct DynamicNfaScanCache<'a> {
    constraint: &'a Constraint,
    tokenizer: &'a Tokenizer,
    use_constraint_fast_transitions: bool,
    deterministic: bool,
    subset_union_requested: bool,
    deadline: Option<Instant>,
    max_collection_items: Option<usize>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    residual_configs: Vec<u32>,
    // Canonical union metadata for interned multi-state configs. These make a
    // cached lazy-subset transition genuinely O(1): the strict vocabulary walk
    // must not rescan every member merely to discover finalizer/future bits.
    config_matched: Vec<BitSet>,
    config_futures: Vec<BitSet>,
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
            state < DYNAMIC_NFA_RAW_STATE_LIMIT,
            "dynamic lexer state exceeds representable raw-config coordinate"
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
        Self::new_with_tokenizer(constraint, &constraint.tokenizer, deadline, true)
    }

    fn new_for_mask(
        constraint: &'a Constraint,
        vocab: &'a DynamicMaskVocab,
        deadline: Option<Instant>,
    ) -> Self {
        let tokenizer = vocab
            .mask_runtime_tokenizer()
            .unwrap_or(&constraint.tokenizer);
        let use_constraint_fast_transitions = std::ptr::eq(tokenizer, &constraint.tokenizer);
        Self::new_with_tokenizer(
            constraint,
            tokenizer,
            deadline,
            use_constraint_fast_transitions,
        )
    }

    fn new_with_tokenizer(
        constraint: &'a Constraint,
        tokenizer: &'a Tokenizer,
        deadline: Option<Instant>,
        use_constraint_fast_transitions: bool,
    ) -> Self {
        Self {
            constraint,
            tokenizer,
            use_constraint_fast_transitions,
            deterministic: !tokenizer.has_epsilon_transitions(),
            subset_union_requested: false,
            deadline,
            max_collection_items: deadline.map(|_| 5_000_000),
            config_ids: FxHashMap::default(),
            configs: Vec::new(),
            transitions: Vec::new(),
            residual_configs: Vec::new(),
            config_matched: Vec::new(),
            config_futures: Vec::new(),
            raw_start_config: FxHashMap::default(),
        }
    }

    #[inline]
    fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer
    }

    #[inline]
    fn raw_state_for_config(&self, config: u32) -> Option<u32> {
        Self::raw_config_state(config)
    }

    #[inline]
    fn config_index(&self, config: u32) -> Option<usize> {
        Self::raw_config_state(config).is_none().then_some(config as usize)
    }

    #[inline]
    fn encode_raw_state(&self, state: u32) -> u32 {
        Self::raw_config(state)
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        if self.use_constraint_fast_transitions {
            self.constraint.tokenizer_fast_transitions.transition(
                &self.constraint.tokenizer,
                state,
                byte,
            )
        } else {
            self.tokenizer.get_transition(state, byte)
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
            && !self.tokenizer.state_has_epsilon_transitions(*state)
        {
            return Ok(self.encode_raw_state(*state));
        }
        if let Some(&id) = self.config_ids.get(states.as_slice()) {
            return Ok(id);
        }
        self.check_growth(self.configs.len(), 1)?;
        if self.configs.len() >= DYNAMIC_NFA_RAW_CONFIG_TAG as usize {
            return Err(
                "dynamic lexer configuration-id namespace exhausted below raw-state tag"
                    .to_owned(),
            );
        }
        let local_id = u32::try_from(self.configs.len())
            .map_err(|_| "dynamic lexer configuration-id overflow".to_owned())?;
        let id = local_id;
        let mut matched = BitSet::new(self.tokenizer.num_terminals() as usize);
        let mut futures = BitSet::new(self.tokenizer.num_terminals() as usize);
        for &state in &states {
            matched.union_with(self.tokenizer.matched_terminal_bitset(state));
            futures.union_with(self.tokenizer.possible_future_terminals(state));
        }
        self.config_ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.transitions.push(None);
        self.residual_configs.push(DYNAMIC_NFA_CONFIG_UNKNOWN);
        self.config_matched.push(matched);
        self.config_futures.push(futures);
        Ok(id)
    }

    fn config_for_raw_start(&mut self, state: u32) -> Result<u32, String> {
        if self.deterministic {
            self.raw_start_config.entry(state).or_insert(state);
            return Ok(state);
        }
        if !self.tokenizer.state_has_epsilon_transitions(state) {
            return Ok(self.encode_raw_state(state));
        }
        if let Some(&cached) = self.raw_start_config.get(&state) {
            return Ok(cached);
        }
        let closure = self.tokenizer.singleton_epsilon_closure(state).into_vec();
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
            let matched = self.tokenizer.matched_terminal_bitset(tokenizer_state);
            let future = self.tokenizer.possible_future_terminals(tokenizer_state);
            !admitted.is_disjoint(matched)
                || !admitted.is_disjoint(future)
                || ignore_terminal.is_some_and(|terminal| {
                    matched.contains(terminal as usize) || future.contains(terminal as usize)
                })
        };

        if self.deterministic {
            return Ok(relevant(state).then_some(state));
        }
        if !self.tokenizer.state_has_epsilon_transitions(state) {
            return Ok(relevant(state).then_some(self.encode_raw_state(state)));
        }
        let states = self
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
            let target = self.transition(config, byte);
            return Ok((target != u32::MAX).then_some(target));
        }
        if let Some(state) = self.raw_state_for_config(config) {
            let target = self.transition(state, byte);
            return if target == u32::MAX {
                Ok(None)
            } else {
                self.config_for_raw_start(target).map(Some)
            };
        }
        let config_index = self.config_index(config).ok_or_else(|| "unknown dynamic lexer config".to_owned())?;
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
                let target = self.transition(state, byte);
                if target != u32::MAX {
                    let target_config = self.config_for_raw_start(target)?;
                    if let Some(target_state) = self.raw_state_for_config(target_config) {
                        self.check_growth(targets.len(), 1)?;
                        targets.push(target_state);
                    } else {
                        let target_index = self.config_index(target_config).ok_or_else(|| "unknown dynamic target config".to_owned())?;
                        let target_states = &self.configs[target_index];
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
            return Ok(self
                .tokenizer
                .exact_dynamic_state_has_future(config)?
                .then_some(config));
        }
        if let Some(state) = self.raw_state_for_config(config) {
            return Ok(self
                .tokenizer
                .exact_dynamic_state_has_future(state)?
                .then_some(config));
        }
        let config_index = self.config_index(config).ok_or_else(|| "unknown dynamic residual config".to_owned())?;
        let cached = self.residual_configs[config_index];
        if cached != DYNAMIC_NFA_CONFIG_UNKNOWN {
            return Ok((cached != DYNAMIC_NFA_CONFIG_DEAD).then_some(cached));
        }

        let mut residual_states = Vec::new();
        for &state in self.configs[config_index].iter() {
            if self.tokenizer.exact_dynamic_state_has_future(state)? {
                residual_states.push(state);
            }
        }
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
                for id in self.tokenizer.matched_terminals_iter(state) {
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
        } else if self.raw_state_for_config(config).is_some() {
            1
        } else {
            self.configs[self.config_index(config).expect("known config")].len()
        }
    }

    #[inline]
    fn config_state(&self, config: u32, index: usize) -> u32 {
        if self.deterministic {
            debug_assert_eq!(index, 0);
            config
        } else if let Some(state) = self.raw_state_for_config(config) {
            debug_assert_eq!(index, 0);
            state
        } else {
            self.configs[self.config_index(config).expect("known config")][index]
        }
    }

    #[inline]
    fn config_has_finalizer(&self, config: u32) -> bool {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return !self.config_matched[self.config_index(config).expect("known config")].is_empty();
        }
        !self
            .tokenizer
            .matched_terminal_bitset(self.config_state(config, 0))
            .is_empty()
    }

    fn config_matched_terminals(&self, config: u32) -> SmallVec<[TerminalID; 4]> {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return self.config_matched[self.config_index(config).expect("known config")]
                .iter_ones()
                .map(|terminal| terminal as TerminalID)
                .collect();
        }
        self.tokenizer
            .matched_terminals_iter(self.config_state(config, 0))
            .collect()
    }

    #[inline]
    fn config_finalizer_code(&self, config: u32) -> u32 {
        const NONE: u32 = u32::MAX;
        const MULTI: u32 = u32::MAX - 1;
        let terminals = self.config_matched_terminals(config);
        match terminals.as_slice() {
            [] => NONE,
            [terminal] => *terminal,
            _ => MULTI,
        }
    }

    #[inline]
    fn config_future_contains(&self, config: u32, terminal: TerminalID) -> bool {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return self.config_futures[self.config_index(config).expect("known config")].contains(terminal as usize);
        }
        self.tokenizer
            .possible_future_terminals(self.config_state(config, 0))
            .contains(terminal as usize)
    }

    fn config_future_contains_exact(
        &self,
        config: u32,
        terminal: TerminalID,
    ) -> Result<bool, String> {
        for index in 0..self.config_len(config) {
            let state = self.config_state(config, index);
            if self.tokenizer.exact_dynamic_state_has_future(state)?
                && self
                    .tokenizer
                    .possible_future_terminals(state)
                    .contains(terminal as usize)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    #[inline]
    fn config_future_intersects(&self, config: u32, terminals: &BitSet) -> bool {
        if !self.deterministic && self.raw_state_for_config(config).is_none() {
            return !terminals.is_disjoint(&self.config_futures[self.config_index(config).expect("known config")]);
        }
        !terminals.is_disjoint(
            self.tokenizer
                .possible_future_terminals(self.config_state(config, 0)),
        )
    }

    fn config_future_intersects_exact(
        &self,
        config: u32,
        terminals: &BitSet,
    ) -> Result<bool, String> {
        for index in 0..self.config_len(config) {
            let state = self.config_state(config, index);
            if self.tokenizer.exact_dynamic_state_has_future(state)?
                && !terminals.is_disjoint(self.tokenizer.possible_future_terminals(state))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Exact union of already-interned runtime configurations. This is the NFA
    /// analogue of merging deterministic mask states that carry the same parser
    /// object. Deterministic raw coordinates have no subset namespace to merge
    /// into, so only identical states collapse in that mode.
    fn union_configs(&mut self, configs: &[u32]) -> Result<Option<u32>, String> {
        let Some((&first, rest)) = configs.split_first() else {
            return Ok(None);
        };
        if self.deterministic {
            return Ok(rest.iter().all(|&state| state == first).then_some(first));
        }
        let mut states = Vec::<u32>::new();
        for &config in configs {
            if let Some(state) = self.raw_state_for_config(config) {
                self.check_growth(states.len(), 1)?;
                states.push(state);
            } else {
                let index = self.config_index(config).ok_or_else(|| "dynamic config union referenced an unknown config".to_owned())?;
                let members = self
                    .configs
                    .get(index)
                    .ok_or_else(|| "dynamic NFA config union referenced an unknown config".to_owned())?;
                self.check_growth(states.len(), members.len())?;
                states.extend_from_slice(members);
            }
        }
        if states.is_empty() {
            return Ok(None);
        }
        self.intern_config(states).map(Some)
    }

    fn config_next_bytes(&self, config: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for state_index in 0..self.config_len(config) {
            let state = self.config_state(config, state_index);
            for (byte, _) in self.tokenizer.transitions_from(state) {
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
        vocab: &DynamicMaskVocab,
        terminals_disallowed: &TerminalsDisallowed,
    ) -> Self {
        let mut memories = Vec::new();
        for (&lexer_state, terminals) in terminals_disallowed.iter() {
            let lexer_state = vocab.mask_runtime_state(lexer_state);
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

    fn allows_token_bytes(&self, tokenizer: &Tokenizer, bytes: &[u8]) -> bool {
        self.advance(tokenizer, bytes).is_some()
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
        tokenizer: &Tokenizer,
        lexer_state: u32,
        terminal: TerminalID,
    ) -> Self {
        if !tokenizer
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
    fn advance(&self, tokenizer: &Tokenizer, segment: &[u8]) -> Option<Self> {
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
            let execution = tokenizer.execute_from_state_all_widths(segment, tokenizer_state);
            for matched in &execution.matches {
                if blocked.iter().any(|&(_, terminal)| terminal == matched.id) {
                    return None;
                }
            }
            for end_state in execution.end_state {
                let future = tokenizer.possible_future_terminals(end_state);
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

struct DynamicDeadlinePoll {
    deadline: Option<Instant>,
    remaining: u16,
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

const DYNAMIC_MASK_CACHE_MAX_STACKS: usize = 4_096;
const DYNAMIC_MASK_CACHE_MAX_DEPTH: u32 = 256;

fn dynamic_mask_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("GLRMASK_DISABLE_DYNAMIC_MASK_CACHE").is_none())
}

fn dynamic_mask_state_key(state: &ConstraintState<'_>) -> Option<DynamicMaskStateKey> {
    let mut remaining = DYNAMIC_MASK_CACHE_MAX_STACKS;
    let mut key = Vec::with_capacity(state.state.len());
    let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    let observation_cache_enabled =
        std::env::var_os("GLRMASK_DISABLE_DYNAMIC_TERMINAL_OBSERVATION_CACHE").is_none()
            && vocab.has_terminal_observation_classes()
            && !state.constraint.tokenizer.has_any_virtual_runtime()
            // Static/dynamic composition can defer parser terminals and repair
            // component switches. Those observations are not represented by
            // the ordinary singleton parser-admission proof below.
            && state.constraint.static_dynamic_overlay.is_none();
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
        let exclusions_empty = paths.iter().all(|(_, exclusions)| exclusions.is_empty());

        // Exact parser-relative lexer quotient. When this parser frontier admits
        // exactly one terminal, every lexer event before the first successful
        // parser advance is observable only through that terminal's
        // `(matched, possible-future)` pair. Equal precomputed exact quotient
        // classes therefore have the same next-token mask; after a finalization
        // both executions enter the same parser child and common lexer reset.
        let lexer_key = if observation_cache_enabled
            && exclusions_empty
            && state.constraint.ignore_terminal.is_none_or(|ignore| {
                let ignore = ignore as usize;
                !state
                    .constraint
                    .tokenizer
                    .matched_terminal_bitset(tokenizer_state)
                    .contains(ignore)
                    && !state
                        .constraint
                        .tokenizer
                        .possible_future_terminals(tokenizer_state)
                        .contains(ignore)
            })
        {
            let admitted = state
                .constraint
                .direct_regular_admissible_terminals(gss)
                .unwrap_or_else(|| {
                    let candidates = BitSet::all(state.constraint.table.num_terminals as usize);
                    stack_admissible_terminals(&state.constraint.table, gss, &candidates)
                });
            let mut terminals = admitted.iter_ones();
            terminals
                .next()
                .filter(|_| terminals.next().is_none())
                .and_then(|terminal| {
                    let terminal = terminal as TerminalID;
                    vocab
                        .terminal_observation_class(terminal, tokenizer_state)
                        .map(|class| DynamicMaskLexerStateKey::TerminalObservation {
                            terminal,
                            class,
                            initial: tokenizer_state == state.constraint.tokenizer.initial_state(),
                        })
                })
                .unwrap_or(DynamicMaskLexerStateKey::Exact(tokenizer_state))
        } else {
            DynamicMaskLexerStateKey::Exact(tokenizer_state)
        };
        key.push((lexer_key, paths));
    }
    // Raw parser-state entries are sorted by tokenizer id. Observation classes
    // deliberately identify different raw ids, so canonicalize ordering after
    // replacing that coordinate and collapse redundant equivalent branches.
    key.sort_unstable();
    key.dedup();
    Some(key)
}

pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    assert!(
        !state.constraint.uses_compact_segmented_parser_runtime(),
        "unified dynamic walker cannot consume recursive provider coordinates",
    );
    fill_mask_dynamic_impl(state, buf, None, false, None)
        .expect("unbounded dynamic mask generation cannot time out");
}

pub(crate) fn fill_mask_dynamic_bounded(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    timeout_ms: u64,
) -> Result<(), String> {
    if state.constraint.uses_compact_segmented_parser_runtime() {
        return Err(
            "bounded unified dynamic walker cannot consume recursive provider coordinates"
                .to_owned(),
        );
    }
    fill_mask_dynamic_impl(
        state,
        buf,
        Some(Instant::now() + Duration::from_millis(timeout_ms)),
        false,
        None,
    )
}


/// OR the complete exact dynamic language into an existing static/component
/// baseline. The baseline does not prune traversal: the strict walker still
/// visits the complete vocabulary and combines the exact result afterward.
pub(crate) fn or_mask_dynamic_additions(state: &ConstraintState<'_>, buf: &mut [u32]) {
    assert!(
        !state.constraint.uses_compact_segmented_parser_runtime(),
        "unified dynamic additions cannot consume recursive provider coordinates",
    );
    fill_mask_dynamic_impl(state, buf, None, true, None)
        .expect("unbounded additive dynamic mask generation cannot time out");
}

/// OR only requested candidate bits from the complete exact dynamic result
/// into `buf`. `candidate_mask` filters the result after traversal; it never
/// restricts which vocabulary prefixes/endpoints the strict walker visits.
pub(crate) fn or_mask_dynamic_candidate_additions(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    candidate_mask: &[u32],
) {
    assert!(
        !state.constraint.uses_compact_segmented_parser_runtime(),
        "unified dynamic candidate additions cannot consume recursive provider coordinates",
    );
    fill_mask_dynamic_impl(state, buf, None, true, Some(candidate_mask))
        .expect("unbounded candidate dynamic mask generation cannot time out");
}

#[inline]
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

    let exact_initial_tsid = state.constraint.tokenizer.initial_state();
    let initial_tsid = vocab.mask_runtime_state(exact_initial_tsid);
    let mut root_branches = DynamicBranches::new();
    let mut lexer_scan_cache = DynamicNfaScanCache::new_for_mask(state.constraint, vocab, deadline);
    let trie = vocab.trie.as_ref();
    if profile {
        eprintln!(
            "[glrmask/profile][dynamic_mask_config] tokenizer_states={} epsilon={} exact_tokenizer_states={} projected={}",
            lexer_scan_cache.tokenizer().num_states(),
            lexer_scan_cache.tokenizer().has_epsilon_transitions(),
            state.constraint.tokenizer.num_states(),
            !std::ptr::eq(lexer_scan_cache.tokenizer(), &state.constraint.tokenizer),
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

    let mut seed_entries = SmallVec::<[(u32, u32, &ParserGSS); 16]>::new();
    // The deterministic mask projection contains only subsets reachable from
    // its own lexical start states. Parser filtering can later expose an exact
    // union of projection states that was never reachable lexically on its own.
    // In that case use the existing lazy config interner as an outer subset
    // coordinate for this mask call. Its non-deterministic mode is also exact
    // for an epsilon-free tokenizer: singleton states are tagged raw configs,
    // arbitrary unions are flattened/canonicalized, and successor rows are
    // memoized lazily per (subset, byte).
    let mut saw_missing_subset = false;
    if vocab.has_mask_subset_provenance() {
        let mut groups = SmallVec::<[(usize, &ParserGSS, SmallVec<[u32; 8]>); 8]>::new();
        for (&tokenizer_state, gss) in &state.state {
            let key = gss.ptr_key();
            if let Some((_, _, states)) = groups.iter_mut().find(|(candidate, _, _)| *candidate == key) {
                states.push(tokenizer_state);
            } else {
                groups.push((key, gss, SmallVec::from_slice(&[tokenizer_state])));
            }
        }
        for (_, gss, mut states) in groups {
            states.sort_unstable();
            states.dedup();
            if states.len() > 1 {
                if let Some(projected) = vocab.mask_runtime_state_for_source_states(
                    &state.constraint.tokenizer,
                    &states,
                ) {
                    seed_entries.push((states[0], projected, gss));
                    continue;
                }
                // Same parser object, but the parser-filtered lexer subset was
                // not materialized by compile-time determinization. The Flat16
                // walker can represent this exact frontier as one runtime
                // subset state.
                saw_missing_subset = true;
            }
            for raw_state in states {
                seed_entries.push((raw_state, vocab.mask_runtime_state(raw_state), gss));
            }
        }
    } else {
        for (&tokenizer_state, gss) in &state.state {
            seed_entries.push((
                tokenizer_state,
                vocab.mask_runtime_state(tokenizer_state),
                gss,
            ));
        }
    }

    if saw_missing_subset && lexer_scan_cache.deterministic {
        // The deterministic projection contains only subsets reachable from
        // its lexical start states. Tell the Flat16 walker that parser
        // filtering exposed an exact same-parser union missing from that
        // projection. It uses the runtime subset representation for this call.
        if matches!(vocab.mask_projection_fast_transitions(), Some(FastTokenizerTransitions::Flat16 { .. })) {
            lexer_scan_cache.subset_union_requested = true;
        }
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_mask_config] subset_union_mode=true"
            );
        }
    }

    // Preserve exact parser identity across lexer-only fanout. `state.state`
    // can contain several tokenizer states carrying the same ParserGSS object.
    // Partitioning that object independently for every lexer state would create
    // distinct transformed ParserStacks objects and erase that provenance.
    // Partition once, then clone the resulting ParserStacks for each lexer
    // seed; those clones retain pointer identity by construction.
    let mut seed_groups =
        SmallVec::<[(usize, &ParserGSS, SmallVec<[(u32, u32); 8]>); 8]>::new();
    for (tokenizer_state, projected_tokenizer_state, gss) in seed_entries {
        let key = gss.ptr_key();
        if let Some((_, _, seeds)) = seed_groups
            .iter_mut()
            .find(|(candidate, _, _)| *candidate == key)
        {
            seeds.push((tokenizer_state, projected_tokenizer_state));
        } else {
            seed_groups.push((
                key,
                gss,
                SmallVec::from_slice(&[(tokenizer_state, projected_tokenizer_state)]),
            ));
        }
    }
    let mut partitioned_seed_entries =
        SmallVec::<[(u32, u32, ParserStacks, TerminalsDisallowed); 16]>::new();
    for (_, gss, seeds) in seed_groups {
        deadline_poll.check()?;
        for (stacks, terminals_disallowed) in gss.partition_by_accumulator() {
            for &(tokenizer_state, projected_tokenizer_state) in &seeds {
                partitioned_seed_entries.push((
                    tokenizer_state,
                    projected_tokenizer_state,
                    stacks.clone(),
                    terminals_disallowed.clone(),
                ));
            }
        }
    }

    for (tokenizer_state, projected_tokenizer_state, stacks, terminals_disallowed) in
        partitioned_seed_entries
    {
        deadline_poll.check()?;
        let initial_prune_guard = InitialPruneGuard::new(vocab, &terminals_disallowed);
            if profile {
                // Keep diagnostic parser-signature queries out of the live
                // full-walk parser cache. They must not prime or otherwise
                // alter the mask walk whose interaction transcript is measured.
                let admitted_for = |stacks: &ParserStacks| {
                    let parser_gss = with_empty_accumulators(stacks);
                    state
                        .constraint
                        .direct_regular_admissible_terminals(&parser_gss)
                        .unwrap_or_else(|| {
                            let candidates =
                                BitSet::all(state.constraint.table.num_terminals as usize);
                            super::commit::exact_admitted_terminals_for_candidates(
                                state.constraint,
                                &parser_gss,
                                &candidates,
                            )
                        })
                };
                let bitset_fingerprint = |bits: &BitSet| {
                    bits.words().iter().fold(0xcbf29ce484222325u64, |hash, &word| {
                        (hash ^ word).wrapping_mul(0x100000001b3)
                    })
                };
                let root_admissible = admitted_for(&stacks);
                let root_admissible_fingerprint = bitset_fingerprint(&root_admissible);
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
                let root_relevant_signature = relevant_signature(&root_admissible);
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
                        let admitted = admitted_for(&child);
                        (
                            terminal,
                            bitset_fingerprint(&admitted),
                            admitted.count_ones(),
                            relevant_signature(&admitted),
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
                eprintln!(
                    "[glrmask/profile][dynamic_seed] generation={} tokenizer_state={} initial={} stack_paths={} exclusions={} transitions={} matched={} futures={}",
                    state.generation,
                    tokenizer_state,
                    tokenizer_state == exact_initial_tsid,
                    stacks.path_count_at_most(1_000_000),
                    terminals_disallowed
                        .iter()
                        .map(|(_, terminals)| terminals.len())
                        .sum::<usize>(),
                    lexer_scan_cache
                        .tokenizer()
                        .transitions_from(projected_tokenizer_state)
                        .count(),
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
                );
            }
            let tokenizer_config =
                lexer_scan_cache.config_for_raw_start(projected_tokenizer_state)?;
            root_branches.push(DynamicBranch {
                tokenizer_config,
                gss: stacks,
                initial_prune_guard,
            });
    }

    let full_walk_started_at = profile.then(std::time::Instant::now);
    // Additive composition used to carry `repair_used` through a second trie
    // recognizer so it could enumerate only B-minus-A. The strict walker is
    // already an exact complete-vocabulary evaluator of the composed parser
    // and lexer, so compute that language directly and combine it with the
    // caller's static baseline. Candidate-only callers likewise filter the
    // complete exact result after traversal rather than pruning vocabulary
    // subtrees during traversal.
    let mut full_walk_scratch = Vec::<u32>::new();
    let full_walk_output = if additive_static_baseline || candidate_mask.is_some() {
        full_walk_scratch.resize(required, 0);
        full_walk_scratch.as_mut_slice()
    } else {
        &mut *buf
    };
    let full_walk_used = try_full_walk_mask(
        state,
        vocab,
        trie,
        &root_branches,
        &mut lexer_scan_cache,
        full_walk_output,
    )?;
    if full_walk_used {
        if !full_walk_scratch.is_empty() {
            for (word_index, dst) in buf.iter_mut().enumerate() {
                let exact = full_walk_scratch[word_index];
                let allowed = candidate_mask
                    .and_then(|candidate| candidate.get(word_index))
                    .copied()
                    .unwrap_or(u32::MAX);
                *dst |= exact & allowed;
            }
        }
        if profile {
            let full_walk_layout = match vocab.mask_projection_fast_transitions() {
                Some(FastTokenizerTransitions::Flat16 { .. }) => "flat16",
                Some(FastTokenizerTransitions::Flat32 { .. }) => "flat32",
                _ => "unknown",
            };
            eprintln!(
                "[glrmask/profile][dynamic_mask] generation={} full_walk=true layout={} tokenizer_states={} full_walk_ms={:.3}",
                state.generation,
                full_walk_layout,
                lexer_scan_cache.tokenizer().num_states(),
                full_walk_started_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        if let Some(cache_key) = cache_key.as_ref() {
            vocab.cache_mask(cache_key.clone(), buf);
        }
        return Ok(());
    }

    Err("strict full vocabulary walk unexpectedly declined".to_owned())
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
    fn strict_full_walk_executes_native_epsilon_nfa_configs() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"ba".to_vec()),
            (5, b"bb".to_vec()),
        ]);
        // Only the parser table is used from this ordinary constraint. The
        // dynamic constraint below substitutes a deliberately retained
        // epsilon-NFA tokenizer with the same two terminal IDs.
        let parser_source = Constraint::from_glrm_grammar(
            r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A | B | A A | A B | B A | B B;
"#,
            &vocab,
        )
        .unwrap();
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        assert!(tokenizer.has_epsilon_transitions());
        let mut dynamic = DynamicConstraint::from_parts(
            parser_source.table.clone(),
            parser_source.terminal_display_names.clone(),
            tokenizer,
            None,
            None,
            Vec::new(),
            &vocab,
        );
        assert!(dynamic.inner.tokenizer.has_epsilon_transitions());
        // Production may prepare a dense deterministic execution coordinate
        // when it fits. Disable that representation here so this regression
        // exercises the native epsilon-NFA backend itself.
        dynamic
            .inner
            .dynamic_mask_vocab
            .disable_prepared_mask_execution_for_test();
        assert!(
            dynamic.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_none(),
            "test setup should retain the source epsilon-NFA coordinate",
        );
        assert!(
            dynamic.inner.dynamic_mask_vocab.mask_projection_fast_transitions().is_none(),
            "epsilon-NFA strict walk must not depend on a dense DFA table",
        );

        TEST_FULL_WALK_USES.with(|count| count.set(0));
        TEST_CONFIG_FULL_WALK_USES.with(|count| count.set(0));
        let state = dynamic.inner.start();
        let mask = state.mask();
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        TEST_CONFIG_FULL_WALK_USES.with(|count| assert!(count.get() > 0));

        for (token_id, _) in dynamic.inner.token_bytes_iter() {
            let expected = token_allowed(&mask, token_id);
            let mut advanced = state.clone();
            let accepted = advanced.commit_token(token_id).is_ok();
            assert_eq!(accepted, expected, "token={token_id}");
        }
    }

    #[test]
    fn additive_and_candidate_modes_use_complete_strict_full_walk() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"ba".to_vec()),
            (4, b"aa".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | B A;
"#,
            &vocab,
        )
        .unwrap();
        let state = dynamic.inner.start();
        let mut exact = vec![0u32; dynamic.inner.mask_len()];
        fill_mask_dynamic(&state, &mut exact);

        let mut additive = vec![0u32; exact.len()];
        set_mask_bit(&mut additive, 4);
        let mut expected_additive = additive.clone();
        for (dst, &word) in expected_additive.iter_mut().zip(&exact) {
            *dst |= word;
        }
        TEST_FULL_WALK_USES.with(|count| count.set(0));
        or_mask_dynamic_additions(&state, &mut additive);
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        assert_eq!(additive, expected_additive);

        let mut candidates = vec![0u32; exact.len()];
        set_mask_bit(&mut candidates, 0);
        set_mask_bit(&mut candidates, 3);
        let mut filtered = vec![0u32; exact.len()];
        set_mask_bit(&mut filtered, 4);
        let mut expected_filtered = filtered.clone();
        for ((dst, &word), &candidate) in expected_filtered
            .iter_mut()
            .zip(&exact)
            .zip(&candidates)
        {
            *dst |= word & candidate;
        }
        TEST_FULL_WALK_USES.with(|count| count.set(0));
        or_mask_dynamic_candidate_additions(&state, &mut filtered, &candidates);
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        assert_eq!(filtered, expected_filtered);
    }

    #[test]
    fn strict_full_walk_spills_for_vocab_depth_beyond_255_edges() {
        let mut entries = Vec::new();
        for depth in 1..=300u32 {
            let mut token = vec![b'a'; depth as usize];
            token.push(b'b');
            entries.push((depth - 1, token));
        }
        entries.push((300, vec![b'a'; 301]));
        let vocab = Vocab::new(entries);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
start start;
t A ::= /a+b?/;
nt start ::= A;
"#,
            &vocab,
        )
        .unwrap();
        assert!(
            dynamic
                .inner
                .dynamic_mask_vocab_for_runtime()
                .trie
                .full_walk_max_parent_depth()
                >= 255,
            "test vocabulary did not create the intended deep radix tree",
        );

        TEST_FULL_WALK_USES.with(|count| count.set(0));
        let state = dynamic.start();
        let mask = state.mask();
        TEST_FULL_WALK_USES.with(|count| assert!(count.get() > 0));
        for (token_id, _) in dynamic.inner.token_bytes_iter() {
            let expected = token_allowed(&mask, token_id);
            let mut advanced = state.clone();
            assert_eq!(advanced.commit_token(token_id).is_ok(), expected, "token={token_id}");
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
    fn dynamic_partition_slicer_matches_uncertified_bounded_string_mask() {
        // Keep several distinct structural root classes, including quote/control
        // families which are invalid inside a JSON string. This catches a
        // root-slot/DFS-index mixup as well as the bounded advancing-state proof.
        let mut tokens = vec![
            (0, b"alpha".to_vec()),
            (1, b" beta".to_vec()),
            (2, b"123".to_vec()),
            (3, b"!!!".to_vec()),
            (4, b"_name".to_vec()),
            (5, b"longword".to_vec()),
            (6, b" quote".to_vec()),
            (7, b"\"".to_vec()),
            (8, b" \"\n".to_vec()),
            (9, b"\\n".to_vec()),
            (10, "é".as_bytes().to_vec()),
            (11, b"a-b".to_vec()),
            (12, b"{".to_vec()),
            (13, b"}".to_vec()),
            (14, b":".to_vec()),
            (15, b"x".to_vec()),
            // Keep the finite mask-only bounded-repeat horizon well beyond H64.
            (16, vec![b'z'; 128]),
        ];
        // Make one ordinary alphabetic layout partition large enough that the
        // profitability gate exercises the direct virtual-residual slicer.
        for index in 0..160u32 {
            tokens.push((17 + index, format!("extra{index:04}").into_bytes()));
        }
        let vocab = Vocab::new(tokens);
        let schema = r#"{"type":"string","maxLength":1000000000000}"#;
        let accelerated = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            accelerated
                .inner
                .dynamic_mask_vocab
                .mask_projection_fast_transitions()
                .is_none(),
            "test must exercise the virtual-residual path without a finite mask projection",
        );

        let mut accelerated_state = accelerated.inner.start();
        let prefix = b"\"inside a bounded string ";
        accelerated_state.commit_bytes(prefix).unwrap();

        TEST_PARTITION_SLICER_HITS.with(|hits| hits.set(0));
        let accelerated_mask = direct_mask(&accelerated_state);
        TEST_PARTITION_SLICER_HITS.with(|hits| {
            assert!(hits.get() > 0, "test state did not exercise the partition slicer")
        });
        TEST_PARTITION_SLICER_DISABLED.with(|disabled| disabled.set(true));
        let reference_mask = direct_mask(&accelerated_state);
        TEST_PARTITION_SLICER_DISABLED.with(|disabled| disabled.set(false));
        assert_eq!(accelerated_mask, reference_mask);
        for token in [0, 1, 2, 3, 4, 5, 6, 10, 11, 15] {
            assert!(token_allowed(&accelerated_mask, token), "token {token} should remain inside the string");
        }
        assert!(!token_allowed(&accelerated_mask, 8));
    }

    #[test]
    fn dynamic_partition_slicer_declines_tiny_vocab_partitions() {
        let vocab = Vocab::new(vec![
            (0, b"alpha".to_vec()),
            (1, b" beta".to_vec()),
            (2, b"123".to_vec()),
            (3, b"!!!".to_vec()),
            (4, b"_name".to_vec()),
            (5, b"longword".to_vec()),
            (6, b" quote".to_vec()),
            (7, b"\"".to_vec()),
            (8, b" \"\n".to_vec()),
            (9, b"\\n".to_vec()),
            (10, "é".as_bytes().to_vec()),
            (11, b"a-b".to_vec()),
            (12, b"{".to_vec()),
            (13, b"}".to_vec()),
            (14, b":".to_vec()),
            (15, b"x".to_vec()),
            (16, vec![b'z'; 128]),
        ]);
        let schema = r#"{"type":"string","maxLength":1000000000000}"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        let mut state = dynamic.inner.start();
        state.commit_bytes(b"\"inside a bounded string ").unwrap();

        TEST_PARTITION_SLICER_HITS.with(|hits| hits.set(0));
        let _ = direct_mask(&state);
        TEST_PARTITION_SLICER_HITS.with(|hits| {
            assert_eq!(hits.get(), 0, "tiny partitions should use the exact trie walk")
        });
    }

    #[test]
    fn dynamic_full_walk_accepts_long_compressed_vocab_edge() {
        let vocab = Vocab::new(vec![
            (0, vec![b'a'; 300]),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
        ]);
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
        let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mask_vocab = dynamic.inner.dynamic_mask_vocab_for_runtime();
        assert!(mask_vocab.trie.subtree_max_total_byte_len(0) >= 300);
        assert!(
            mask_vocab.trie.full_walk_max_parent_depth() < 255,
            "a long compressed edge must not consume one DFS stack slot per byte",
        );
        assert!(mask_vocab.mask_projection_fast_transitions().is_some());

        let mask = dynamic.start().mask();
        assert!(token_allowed(&mask, 0));
        assert!(token_allowed(&mask, 1));
        assert!(!token_allowed(&mask, 2));
    }

    #[test]
    fn dynamic_virtual_unit_repeat_uses_static_mask_projection_end_to_end() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"aaaa".to_vec()),
            (4, b"aaaaa".to_vec()),
            (5, b"aaaaaa".to_vec()),
            (6, b"b".to_vec()),
        ]);

        let billion_grammar = r#"
start start;
t A ::= /a{0,1000000000}/;
nt start ::= A;
"#;
        let billion = DynamicConstraint::from_glrm_grammar(billion_grammar, &vocab).unwrap();
        assert_eq!(
            billion.inner.tokenizer.num_states(),
            1,
            "the exact billion-bound lexer must remain an arithmetic runtime state",
        );
        let mask_tokenizer = billion
            .inner
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .expect("virtual exact lexer must install a static mask lexer");
        assert_eq!(
            mask_tokenizer.num_states(),
            vocab.max_token_byte_len() as u32 + 3,
            "mask lexer size must depend on vocabulary horizon, not the repeat bound",
        );
        let billion_mask = billion.inner.start().mask();
        for token in 0..=5 {
            assert!(token_allowed(&billion_mask, token), "a-only token {token} was rejected");
        }
        assert!(!token_allowed(&billion_mask, 6));

        // For a small bound, compare the virtual dynamic implementation with
        // the ordinary materialized static implementation through the exact
        // upper-bound transition. This checks both re-projection after commit
        // and rejection of a vocabulary token which crosses the bound.
        let boundary_grammar = r#"
start start;
t A ::= /a{0,5}/;
nt start ::= A;
"#;
        let dynamic = DynamicConstraint::from_glrm_grammar(boundary_grammar, &vocab).unwrap();
        let ordinary = Constraint::from_glrm_grammar(boundary_grammar, &vocab).unwrap();
        let mut dynamic_state = dynamic.inner.start();
        let mut ordinary_state = ordinary.start();

        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(!token_allowed(&dynamic_state.mask(), 5));
        dynamic_state.commit_token(2).unwrap(); // consume three a's
        ordinary_state.commit_token(2).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(token_allowed(&dynamic_state.mask(), 0));
        assert!(token_allowed(&dynamic_state.mask(), 1));
        assert!(!token_allowed(&dynamic_state.mask(), 2));

        dynamic_state.commit_token(1).unwrap(); // reach the exact upper bound
        ordinary_state.commit_token(1).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(dynamic_state.is_accepting());
        assert!(!token_allowed(&dynamic_state.mask(), 0));
        assert!(dynamic_state.clone().commit_token(0).is_err());
    }

    #[test]
    fn dynamic_hybrid_virtual_repeat_coexists_with_static_terminals() {
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaa".to_vec()),
            (4, b"baaa".to_vec()),
            (5, b"baaaaa".to_vec()),
            (6, b"baaaaaa".to_vec()),
            (7, b"x".to_vec()),
        ]);
        let billion_grammar = r#"
start start;
t A ::= /a{0,1000000000}/;
t B ::= 'b';
nt start ::= B A;
"#;
        let hybrid = DynamicConstraint::from_glrm_grammar(billion_grammar, &vocab).unwrap();
        assert!(
            hybrid
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "the pathological terminal should be the arithmetic component",
        );
        assert!(
            hybrid.inner.tokenizer.num_states() < 64,
            "physical hybrid lexer unexpectedly scales with the billion bound",
        );
        let start_mask = hybrid.start().mask();
        assert!(token_allowed(&start_mask, 4));
        assert!(token_allowed(&start_mask, 5));
        assert!(token_allowed(&start_mask, 6));
        assert!(!token_allowed(&start_mask, 7));

        let small_grammar = r#"
start start;
t A ::= /a{0,5}/;
t B ::= 'b';
nt start ::= B A;
"#;
        // The hybrid threshold intentionally leaves this small grammar on the
        // ordinary path; it is therefore an independent materialized oracle.
        let dynamic_small = DynamicConstraint::from_glrm_grammar(small_grammar, &vocab).unwrap();
        let ordinary_small = Constraint::from_glrm_grammar(small_grammar, &vocab).unwrap();
        let mut dynamic_state = dynamic_small.start();
        let mut ordinary_state = ordinary_small.start();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(token_allowed(&dynamic_state.mask(), 5));
        assert!(!token_allowed(&dynamic_state.mask(), 6));

        dynamic_state.commit_token(0).unwrap();
        ordinary_state.commit_token(0).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(3).unwrap();
        ordinary_state.commit_token(3).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
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
