use super::*;

trait FullWalkTransitionTable {
    type Cell: Copy;

    fn cell(&self, state: u32, byte: u8) -> Self::Cell;

    fn cell_is_dead(cell: Self::Cell) -> bool;

    fn cell_has_finalizer(cell: Self::Cell) -> bool;

    fn cell_target(cell: Self::Cell) -> u32;

    #[inline(always)]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        let cell = self.cell(state, byte);
        if Self::cell_is_dead(cell) {
            u32::MAX
        } else {
            Self::cell_target(cell)
        }
    }
}

#[derive(Clone, Copy)]
struct FullWalkFlat16<'a> {
    transitions: &'a [u16],
}

impl FullWalkTransitionTable for FullWalkFlat16<'_> {
    type Cell = u16;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u16 {
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

}

#[derive(Clone, Copy)]
struct FullWalkFlat32<'a> {
    transitions: &'a [u32],
}

impl FullWalkTransitionTable for FullWalkFlat32<'_> {
    type Cell = u32;

    #[inline(always)]
    fn cell(&self, state: u32, byte: u8) -> u32 {
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

}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_flat16<const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: &[u16],
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        root_branches,
        lexer_scan_cache,
        buf,
        FullWalkFlat16 { transitions },
        finalizer_code,
        single_finalizer_continues,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_flat32<const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: &[u32],
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    try_full_walk_mask_with_table::<_, HOT_SINGLE_ROOT>(
        state,
        vocab,
        trie,
        root_branches,
        lexer_scan_cache,
        buf,
        FullWalkFlat32 { transitions },
        finalizer_code,
        single_finalizer_continues,
    )
}

#[derive(Clone, PartialEq, Eq)]
enum FullWalkPruneGuard {
    Passed,
    Pending(SmallVec<[(u32, TerminalID); 2]>),
}

impl FullWalkPruneGuard {
    fn from_initial(guard: &InitialPruneGuard, vocab: &DynamicMaskVocab) -> Self {
        match guard {
            InitialPruneGuard::Passed => Self::Passed,
            InitialPruneGuard::Pending { memories } => {
                Self::Pending(
                    memories
                        .iter()
                        .map(|&(state, terminal)| (state, terminal))
                        .collect(),
                )
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
        tokenizer: &Tokenizer,
        transitions: &T,
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
            if tokenizer
                .matched_terminals_slice(target)
                .contains(&terminal)
            {
                return None;
            }
            if tokenizer
                .possible_future_terminals(target)
                .contains(terminal as usize)
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
    lexer_state_count: usize,
}

impl FullWalkParserCache {
    const DEAD: u32 = u32::MAX;

    fn from_roots(
        root_branches: &DynamicBranches,
        lexer_state_count: usize,
    ) -> (Self, SmallVec<[u32; 4]>) {
        let mut nodes = Vec::<FullWalkParserNode>::new();
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
                token_boundary_allowed: vec![0; lexer_state_count],
                children: SmallVec::new(),
                last_child_terminal: TerminalID::MAX,
                last_child_target: Self::DEAD,
            });
            root_nodes.push(id);
        }
        (
            Self {
                nodes,
                lexer_state_count,
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
                token_boundary_allowed: vec![0; self.lexer_state_count],
                children: SmallVec::new(),
                last_child_terminal: TerminalID::MAX,
                last_child_target: Self::DEAD,
            });
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
                    super::super::commit::exact_admitted_terminals_for_candidates(
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
    fn physical_token_boundary_allowed(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        parser_node: u32,
        lexer_state: u32,
    ) -> bool {
        let node = parser_node as usize;
        let lexer = lexer_state as usize;
        let cached = unsafe {
            *self.nodes
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
            *self.nodes
                .get_unchecked_mut(node)
                .token_boundary_allowed
                .get_unchecked_mut(lexer) = if allowed { 2 } else { 1 };
        }
        allowed
    }

    #[inline(always)]
    fn token_boundary_allowed_raw(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        initial_lexer_state: u32,
        lexer_state: u32,
        parser_node: u32,
    ) -> bool {
        lexer_state == initial_lexer_state
            || self.physical_token_boundary_allowed(
                constraint,
                tokenizer,
                parser_node,
                lexer_state,
            )
    }

    #[inline(always)]
    fn token_boundary_allowed(
        &mut self,
        constraint: &Constraint,
        tokenizer: &Tokenizer,
        initial_lexer_state: u32,
        branch: &FullWalkBranch,
    ) -> bool {
        self.token_boundary_allowed_raw(
            constraint,
            tokenizer,
            initial_lexer_state,
            branch.lexer_state,
            branch.parser_node,
        )
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
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    transitions: &T,
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
        finalizer_code,
        single_finalizer_continues,
        tokenizer,
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
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
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
            finalizer_code,
            single_finalizer_continues,
            tokenizer,
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
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    const MULTI: u32 = u32::MAX - 1;
    let code = unsafe { *finalizer_code.get_unchecked(target as usize) };
    if code != MULTI {
        if let Some(next_parser) = parser_cache.advance(constraint, parser_node, code) {
            let reset = FullWalkBranch {
                lexer_state: initial_lexer_state,
                parser_node: next_parser,
                prune_guard: if Some(code) == constraint.ignore_terminal {
                    FullWalkPruneGuard::Passed
                } else if unsafe { *single_finalizer_continues.get_unchecked(target as usize) } != 0 {
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
    for &terminal in tokenizer.matched_terminals_slice(target) {
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
                            tokenizer, target, terminal,
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
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    tokenizer: &Tokenizer,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkScalarFinalizerOutcome {
    const MULTI: u32 = u32::MAX - 1;
    let code = unsafe { *finalizer_code.get_unchecked(target as usize) };
    if code == MULTI {
        return full_walk_scalar_finalizer(
            target,
            parser_node,
            initial_lexer_state,
            finalizer_code,
            single_finalizer_continues,
            tokenizer,
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
            } else if unsafe { *single_finalizer_continues.get_unchecked(target as usize) } != 0 {
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
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
    two_distinct_marker: u32,
    scalar_lexer: &mut u32,
    scalar_parser: &mut u32,
    current_two: &mut ((u32, u32), (u32, u32)),
) -> bool {
    const MULTI: u32 = u32::MAX - 1;
    let code = unsafe { *finalizer_code.get_unchecked(target as usize) };
    if code == MULTI
        || (Some(code) != constraint.ignore_terminal
            && unsafe { *single_finalizer_continues.get_unchecked(target as usize) } != 0)
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
    finalizer_code: &[u32],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkBranches {
    const NONE: u32 = u32::MAX;
    const MULTI: u32 = u32::MAX - 1;
    let mut next = FullWalkBranches::new();
    for branch in branches {
        let Some(advanced_guard) = branch
            .prune_guard
            .advance(tokenizer, transitions, byte)
        else {
            continue;
        };
        let target = transitions.transition(branch.lexer_state, byte);
        if target == u32::MAX {
            continue;
        }
        let code = unsafe { *finalizer_code.get_unchecked(target as usize) };
        if code == MULTI {
            for &terminal in tokenizer.matched_terminals_slice(target) {
                if let Some(parser_node) = parser_cache.advance(
                    constraint, branch.parser_node, terminal,
                ) {
                    let matched_guard = if Some(terminal) == constraint.ignore_terminal {
                        advanced_guard.clone()
                    } else {
                        advanced_guard.remember_terminal_match(tokenizer, target, terminal)
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
                advanced_guard.remember_terminal_match(tokenizer, target, code)
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
fn full_walk_projection_union_two(
    vocab: &DynamicMaskVocab,
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
    let result = vocab.mask_projection_state_for_projection_states(&[key.0, key.1]);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_projection_union_three(
    vocab: &DynamicMaskVocab,
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
    let result = vocab.mask_projection_state_for_projection_states(&states);
    cache.insert(key, result);
    result
}

#[inline]
fn full_walk_merge_two_same_parser(
    vocab: &DynamicMaskVocab,
    pair_union_cache: &mut FxHashMap<(u32, u32), Option<u32>>,
    first: (u32, u32),
    second: (u32, u32),
) -> Option<(u32, u32)> {
    if first.1 != second.1 {
        return None;
    }
    full_walk_projection_union_two(vocab, pair_union_cache, first.0, second.0)
        .map(|lexer_state| (lexer_state, first.1))
}

#[inline]
fn full_walk_merge_three_same_parser(
    vocab: &DynamicMaskVocab,
    triple_union_cache: &mut FxHashMap<(u32, u32, u32), Option<u32>>,
    lexers: (u32, u32, u32),
    parser_node: u32,
) -> Option<(u32, u32)> {
    full_walk_projection_union_three(
        vocab,
        triple_union_cache,
        lexers.0,
        lexers.1,
        lexers.2,
    )
    .map(|lexer_state| (lexer_state, parser_node))
}

#[inline]
fn full_walk_merge_branches_same_parser(
    vocab: &DynamicMaskVocab,
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
            vocab,
            pair_union_cache,
            first.lexer_state,
            second.lexer_state,
        ),
        [first, second, third] => full_walk_projection_union_three(
            vocab,
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
            vocab.mask_projection_state_for_projection_states(&lexers)
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
    finalizer_code: &[u32],
    tokenizer: &Tokenizer,
    transitions: &T,
    parser_cache: &mut FullWalkParserCache,
    constraint: &Constraint,
) -> FullWalkManyState {
    match state {
        FullWalkManyState::Branches(branches) => full_walk_many_state_from_branches(
            full_walk_step_many(
                branches,
                byte,
                initial_lexer_state,
                finalizer_code,
                tokenizer,
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
                finalizer_code,
                tokenizer,
                transitions,
                parser_cache,
                constraint,
            ))
        }
    }
}



/// Exact direct dynamic-mask path for bounded deterministic lexer coordinates.
///
/// This deliberately performs the complete vocabulary walk. It does not use
/// subtree certificates, segment-effect caches, recognizer-state interning, or
/// any other mechanism that can omit vocabulary edges. Unsupported lexer or
/// composition shapes return `false` and use the existing exact fallback.

#[inline(never)]
fn try_full_walk_mask_with_table<T: FullWalkTransitionTable, const HOT_SINGLE_ROOT: bool>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    trie: &DynamicMaskTrie,
    root_branches: &DynamicBranches,
    lexer_scan_cache: &mut DynamicNfaScanCache<'_>,
    buf: &mut [u32],
    transitions: T,
    finalizer_code: &[u32],
    single_finalizer_continues: &[u8],
) -> Result<bool, String> {
    if root_branches.iter().any(|root| root.fresh_reset) {
        return Ok(false);
    }
    debug_assert!(trie.full_walk_max_parent_depth() < 255);

    let initial_lexer_state = lexer_scan_cache
        .config_for_raw_start(vocab.mask_runtime_state(state.constraint.tokenizer.initial_state()))?;

    let all_words = vocab.all_original_token_words();
    let copy_len = buf.len().min(all_words.len());
    buf[..copy_len].copy_from_slice(&all_words[..copy_len]);
    if copy_len < buf.len() { buf[copy_len..].fill(0); }

    let (mut parser_cache, root_parser_nodes) = FullWalkParserCache::from_roots(
        root_branches,
        lexer_scan_cache.tokenizer().num_states() as usize,
    );
    // Scalar is overwhelmingly dominant. Encode dead/multi directly in the
    // lexer-state coordinate so the common DFS path needs no separate kind
    // load/store. Full-walk lexer states are bounded far below these u32
    // sentinels by the dense-transition memory budget.
    const FULL_WALK_LEXER_TWO_DISTINCT: u32 = u32::MAX - 3;
    const FULL_WALK_LEXER_TWO: u32 = u32::MAX - 2;
    const FULL_WALK_LEXER_MULTI: u32 = u32::MAX - 1;
    const FULL_WALK_LEXER_DEAD: u32 = u32::MAX;
    let mut stack_lexer = [FULL_WALK_LEXER_DEAD; 256];
    let mut stack_parser = [0u32; 256];
    let mut stack_two = [((0u32, 0u32), (0u32, 0u32)); 256];
    // Multi-branch stack states are uncommon, and `FullWalkManyState` embeds a
    // SmallVec. Do not eagerly construct/drop 256 empty SmallVec values on
    // every complete vocabulary walk; materialize only the depths that
    // actually carry a multi state.
    let mut stack_many: [Option<FullWalkManyState>; 256] = std::array::from_fn(|_| None);
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
            full_walk_merge_two_same_parser(vocab, &mut pair_union_cache, first, second)
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
                    prune_guard: FullWalkPruneGuard::from_initial(&root.initial_prune_guard, vocab),
                },
            );
        }
        if let Some((lexer_state, parser_node)) = full_walk_merge_branches_same_parser(
            vocab,
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
    let tokenizer = lexer_scan_cache.tokenizer();

    let mut scalar_lexer = FULL_WALK_LEXER_DEAD;
    let mut scalar_parser = 0u32;
    let mut current_two = ((0u32, 0u32), (0u32, 0u32));
    let mut current_many = FullWalkManyState::Branches(FullWalkBranches::new());

    for &op in walk_ops {
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
                                finalizer_code,
                                single_finalizer_continues,
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
                                    finalizer_code,
                                    single_finalizer_continues,
                                    tokenizer,
                                    &mut parser_cache,
                                    state.constraint,
                                )
                            } else {
                                full_walk_scalar_finalizer(
                                    target,
                                    scalar_parser,
                                    initial_lexer_state,
                                    finalizer_code,
                                    single_finalizer_continues,
                                    tokenizer,
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
                                            vocab,
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
                                                vocab,
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
                                        vocab,
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
                        finalizer_code,
                        single_finalizer_continues,
                        tokenizer,
                        &transitions,
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
                                    vocab,
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
                                            vocab,
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
                        finalizer_code,
                        single_finalizer_continues,
                        tokenizer,
                        &transitions,
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
                                    vocab,
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
                                            vocab,
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
                    finalizer_code,
                    tokenizer,
                    &transitions,
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
                                        vocab,
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
                                        vocab,
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
                                vocab,
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
                    parser_cache.token_boundary_allowed_raw(
                        state.constraint,
                        tokenizer,
                        initial_lexer_state,
                        scalar_lexer,
                        scalar_parser,
                    )
                } else if scalar_lexer == FULL_WALK_LEXER_TWO_DISTINCT || scalar_lexer == FULL_WALK_LEXER_TWO {
                    parser_cache.token_boundary_allowed_raw(
                        state.constraint,
                        tokenizer,
                        initial_lexer_state,
                        current_two.0.0,
                        current_two.0.1,
                    ) || parser_cache.token_boundary_allowed_raw(
                        state.constraint,
                        tokenizer,
                        initial_lexer_state,
                        current_two.1.0,
                        current_two.1.1,
                    )
                } else if scalar_lexer == FULL_WALK_LEXER_MULTI {
                    match &current_many {
                        FullWalkManyState::Branches(branches) => branches.iter().any(|branch| {
                            parser_cache.token_boundary_allowed(
                                state.constraint,
                                tokenizer,
                                initial_lexer_state,
                                branch,
                            )
                        }),
                        FullWalkManyState::ThreeSameParser {
                            lexers,
                            parser_node,
                        } => parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            initial_lexer_state,
                            lexers.0,
                            *parser_node,
                        ) || parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
                            initial_lexer_state,
                            lexers.1,
                            *parser_node,
                        ) || parser_cache.token_boundary_allowed_raw(
                            state.constraint,
                            tokenizer,
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
    Ok(true)
}
