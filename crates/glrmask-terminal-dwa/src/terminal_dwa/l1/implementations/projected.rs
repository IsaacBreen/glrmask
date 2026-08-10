//! Exact projected-residual L1 compiler.
//!
//! A state represents one terminal's residual language from one lexer
//! configuration. Every non-dead state is accepting: surviving to a model-token
//! boundary is exactly `finalizer || possible_future`. All `(raw state, terminal)`
//! residuals are roots; minimization therefore never prunes by reachability from
//! one distinguished start state.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::terminal_dwa::l1::implementations::support::{DEAD, Scanner};
use crate::Vocab;

const UNKNOWN: u32 = u32::MAX - 1;

enum ConfigStates {
    One(u32),
    Many(Arc<[u32]>),
}

impl ConfigStates {
    fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Many(states) => states.len(),
        }
    }
}

struct Projected<'a> {
    input: BuildInput<'a>,
    active: Vec<Vec<u32>>,
    active_by_group: Vec<Box<[u64]>>,
    terminals: Vec<u32>,
    terminal_groups: Vec<usize>,
    group_for_terminal: Vec<usize>,
    configs: Vec<(u32, ConfigStates)>,
    singleton_ids: FxHashMap<u64, u32>,
    ids: FxHashMap<(u32, Arc<[u32]>), u32>,
    transitions: Vec<Vec<(u8, u32)>>,
    root_closure_states: usize,
    root_memberships: usize,
}

impl<'a> Projected<'a> {
    fn new(input: BuildInput<'a>) -> Self {
        let active = (0..input.tokenizer.num_states())
            .map(|state| {
                super::super::collect_active_terminal_signature(
                    input.tokenizer,
                    state,
                    input.active_terminals,
                )
            })
            .collect::<Vec<_>>();
        let terminals = input
            .active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
            .collect::<Vec<_>>();
        let mut terminal_index = vec![usize::MAX; input.active_terminals.len()];
        for (index, &terminal) in terminals.iter().enumerate() {
            terminal_index[terminal as usize] = index;
        }
        let words = (input.tokenizer.num_states() as usize).div_ceil(64);
        let mut active_by_terminal = (0..terminals.len())
            .map(|_| vec![0u64; words].into_boxed_slice())
            .collect::<Vec<_>>();
        for (state, signature) in active.iter().enumerate() {
            for &terminal in signature {
                let index = terminal_index[terminal as usize];
                active_by_terminal[index][state / 64] |= 1u64 << (state % 64);
            }
        }

        // Terminal identity is irrelevant to the residual language once two
        // terminals have exactly the same active-state predicate. Quotienting
        // those predicates before residual construction prevents duplicated
        // `(terminal, configuration)` subautomata while retaining every
        // terminal in the emitted signature.
        let mut group_ids = FxHashMap::<Vec<u64>, usize>::default();
        let mut active_by_group = Vec::<Box<[u64]>>::new();
        let mut group_for_terminal = vec![usize::MAX; input.active_terminals.len()];
        let mut terminal_groups = Vec::with_capacity(terminals.len());
        for (&terminal, active_states) in terminals.iter().zip(active_by_terminal) {
            let group = if let Some(&group) = group_ids.get(active_states.as_ref()) {
                group
            } else {
                let group = active_by_group.len();
                group_ids.insert(active_states.to_vec(), group);
                active_by_group.push(active_states);
                group
            };
            group_for_terminal[terminal as usize] = group;
            terminal_groups.push(group);
        }
        Self {
            input,
            active,
            active_by_group,
            terminals,
            terminal_groups,
            group_for_terminal,
            configs: Vec::new(),
            singleton_ids: FxHashMap::default(),
            ids: FxHashMap::default(),
            transitions: Vec::new(),
            root_closure_states: 0,
            root_memberships: 0,
        }
    }

    fn state_active(&self, state: u32, group: u32) -> bool {
        self.active_by_group[group as usize][state as usize / 64]
            & (1u64 << (state as usize % 64))
            != 0
    }

    fn intern_singleton(&mut self, group: u32, state: u32) -> u32 {
        let key = (u64::from(group) << 32) | u64::from(state);
        if let Some(&id) = self.singleton_ids.get(&key) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.singleton_ids.insert(key, id);
        self.configs.push((group, ConfigStates::One(state)));
        self.transitions.push(Vec::new());
        id
    }

    fn intern_filtered(&mut self, group: u32, states: Vec<u32>) -> u32 {
        if states.is_empty() {
            return DEAD;
        }
        debug_assert!(states.windows(2).all(|pair| pair[0] < pair[1]));
        if states.len() == 1 {
            return self.intern_singleton(group, states[0]);
        }
        let key = (group, Arc::<[u32]>::from(states));
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.ids.insert(key.clone(), id);
        self.configs
            .push((group, ConfigStates::Many(Arc::clone(&key.1))));
        self.transitions.push(Vec::new());
        id
    }

    fn intern(&mut self, group: u32, mut states: Vec<u32>) -> u32 {
        states.retain(|&state| self.state_active(state, group));
        self.intern_filtered(group, states)
    }

    /// Project one epsilon closure onto all active terminals at once.
    fn root_row(&mut self, raw: u32) -> Vec<u32> {
        let closure = self
            .input
            .tokenizer
            .execute_from_state_end_only(&[], raw)
            .to_vec();
        self.root_closure_states += closure.len();
        if closure.len() == 1 {
            let state = closure[0];
            let mut row = vec![DEAD; self.active_by_group.len()];
            let terminals = self.active[state as usize].clone();
            for terminal in terminals {
                let group = self.group_for_terminal[terminal as usize];
                debug_assert_ne!(group, usize::MAX);
                if row[group] == DEAD {
                    row[group] = self.intern_singleton(group as u32, state);
                    self.root_memberships += 1;
                }
            }
            return row;
        }
        let mut grouped = (0..self.active_by_group.len())
            .map(|_| Vec::<u32>::new())
            .collect::<Vec<_>>();
        for state in closure {
            for &terminal in &self.active[state as usize] {
                let group = self.group_for_terminal[terminal as usize];
                debug_assert_ne!(group, usize::MAX);
                if grouped[group].last().copied() != Some(state) {
                    grouped[group].push(state);
                    self.root_memberships += 1;
                }
            }
        }
        let mut row = Vec::with_capacity(self.active_by_group.len());
        for (group, states) in grouped.iter_mut().enumerate() {
            row.push(self.intern_filtered(group as u32, std::mem::take(states)));
        }
        row
    }

    fn step(&mut self, state: u32, byte: u8, roots: &[Vec<u32>]) -> u32 {
        let group = self.configs[state as usize].0;
        match &self.configs[state as usize].1 {
            ConfigStates::One(source) => {
                let target = self.input.flat_trans[*source as usize * 256 + byte as usize];
                if target == DEAD {
                    DEAD
                } else {
                    roots[target as usize][group as usize]
                }
            }
            ConfigStates::Many(config) => {
                let config = Arc::clone(config);
                self.intern(
                    group,
                    self.input.tokenizer.step_all(config.as_ref(), byte).to_vec(),
                )
            }
        }
    }
}

struct Minimized {
    classes: Vec<u32>,
    /// Exact byte-equivalence columns, stored target-contiguously for reverse
    /// preimage scans.
    columns: Vec<Box<[u32]>>,
    byte_class: [u8; 256],
    state_count: usize,
    rounds: usize,
}

fn minimize(transitions: &[Vec<(u8, u32)>], alphabet: &[u8]) -> Minimized {
    minimize_seeded(transitions, alphabet, None)
}

fn minimize_seeded(
    transitions: &[Vec<(u8, u32)>],
    alphabet: &[u8],
    initial_classes: Option<&[u32]>,
) -> Minimized {
    // The input alphabet is already quotiented by exact raw tokenizer columns.
    // That equality is preserved by epsilon closure and terminal projection,
    // so no second projected-column quotient is necessary.
    let mut byte_class = [0u8; 256];
    for (symbol, &byte) in alphabet.iter().enumerate() {
        byte_class[byte as usize] = symbol as u8;
    }

    // Every represented state is accepting; DEAD is the one implicit rejecting
    // sink. Keep the large projected automaton sparse. Its literal-heavy rows
    // contain very few live edges, and materializing `states × alphabet` here
    // would dominate both time and memory before minimization collapses it.
    let live = transitions.len();
    let mut class = initial_classes.map_or_else(|| vec![0u32; live], <[u32]>::to_vec);
    debug_assert_eq!(class.len(), live);
    let block_count = class.iter().copied().max().map_or(0usize, |value| value as usize + 1);
    let mut blocks = vec![Vec::<u32>::new(); block_count.max(usize::from(live != 0))];
    if initial_classes.is_none() && live != 0 {
        blocks[0].reserve(live);
    }
    for (state, &block) in class.iter().enumerate() {
        blocks[block as usize].push(state as u32);
    }
    debug_assert!(blocks.iter().all(|block| !block.is_empty()));

    let mut incoming_counts = vec![0u32; live];
    let mut block_incoming = vec![vec![0u32; alphabet.len()].into_boxed_slice(); blocks.len()];
    for row in transitions {
        for &(symbol, target) in row {
            incoming_counts[target as usize] += 1;
            block_incoming[class[target as usize] as usize][symbol as usize] += 1;
        }
    }
    let mut incoming_offsets = vec![0u32; live + 1];
    for target in 0..live {
        incoming_offsets[target + 1] = incoming_offsets[target] + incoming_counts[target];
    }
    let mut next = incoming_offsets[..live].to_vec();
    let mut incoming = vec![(0u8, 0u32); incoming_offsets[live] as usize];
    for (source, row) in transitions.iter().enumerate() {
        for &(symbol, target) in row {
            let slot = &mut next[target as usize];
            incoming[*slot as usize] = (symbol, source as u32);
            *slot += 1;
        }
    }
    for target in 0..live {
        let start = incoming_offsets[target] as usize;
        let end = incoming_offsets[target + 1] as usize;
        incoming[start..end].sort_unstable();
    }

    // Hopcroft over all residual roots. With no seed, one live block is
    // sufficient because the implicit rejecting sink's predecessor set is the
    // complement of the represented live predecessors. With a seed, refinement
    // computes the coarsest right congruence that respects those initial blocks.
    let mut queue = VecDeque::<(u32, usize)>::new();
    let mut queued = vec![vec![false; alphabet.len()]; blocks.len()];
    for block in 0..blocks.len() {
        for symbol in 0..alphabet.len() {
            if block_incoming[block][symbol] != 0 {
                queue.push_back((block as u32, symbol));
                queued[block][symbol] = true;
            }
        }
    }
    let mut marked = vec![false; live];
    let mut affected_members = vec![Vec::<u32>::new(); blocks.len()];
    let mut affected_blocks = Vec::<u32>::new();
    let mut pops = 0usize;

    while let Some((splitter, symbol)) = queue.pop_front() {
        queued[splitter as usize][symbol] = false;
        pops += 1;
        let symbol = symbol as u8;
        for &target in &blocks[splitter as usize] {
            let start = incoming_offsets[target as usize] as usize;
            let end = incoming_offsets[target as usize + 1] as usize;
            let edges = &incoming[start..end];
            let first = edges.partition_point(|&(edge_symbol, _)| edge_symbol < symbol);
            let last = edges.partition_point(|&(edge_symbol, _)| edge_symbol <= symbol);
            for &(_, source) in &edges[first..last] {
                let block = class[source as usize];
                let members = &mut affected_members[block as usize];
                if members.is_empty() {
                    affected_blocks.push(block);
                }
                members.push(source);
            }
        }
        for block_id in affected_blocks.drain(..) {
            let block_id = block_id as usize;
            let sources = std::mem::take(&mut affected_members[block_id]);
            if sources.len() == blocks[block_id].len() {
                continue;
            }
            for &source in &sources {
                marked[source as usize] = true;
            }
            let old = std::mem::take(&mut blocks[block_id]);
            let mut inside = Vec::with_capacity(sources.len());
            let mut outside = Vec::with_capacity(old.len() - sources.len());
            for state in old {
                if marked[state as usize] {
                    inside.push(state);
                } else {
                    outside.push(state);
                }
            }
            for &source in &sources {
                marked[source as usize] = false;
            }
            blocks[block_id] = outside;
            let new_id = blocks.len();
            for &state in &inside {
                class[state as usize] = new_id as u32;
            }
            blocks.push(inside);

            let mut inside_incoming = vec![0u32; alphabet.len()];
            for &state in &blocks[new_id] {
                let start = incoming_offsets[state as usize] as usize;
                let end = incoming_offsets[state as usize + 1] as usize;
                for &(incoming_symbol, _) in &incoming[start..end] {
                    inside_incoming[incoming_symbol as usize] += 1;
                }
            }
            let mut outside_incoming = std::mem::take(&mut block_incoming[block_id]).into_vec();
            for symbol in 0..alphabet.len() {
                outside_incoming[symbol] -= inside_incoming[symbol];
            }
            block_incoming[block_id] = outside_incoming.into_boxed_slice();
            block_incoming.push(inside_incoming.into_boxed_slice());
            queued.push(vec![false; alphabet.len()]);
            affected_members.push(Vec::new());

            for other_symbol in 0..alphabet.len() {
                if queued[block_id][other_symbol] {
                    if block_incoming[new_id][other_symbol] != 0 {
                        queue.push_back((new_id as u32, other_symbol));
                        queued[new_id][other_symbol] = true;
                    }
                } else {
                    let smaller = if blocks[block_id].len() <= blocks[new_id].len() {
                        block_id
                    } else {
                        new_id
                    };
                    if block_incoming[smaller][other_symbol] != 0
                        && !queued[smaller][other_symbol]
                    {
                        queue.push_back((smaller as u32, other_symbol));
                        queued[smaller][other_symbol] = true;
                    }
                }
            }
        }
    }

    let representatives = blocks
        .iter()
        .map(|members| *members.iter().min().expect("non-empty DFA block") as usize)
        .collect::<Vec<_>>();
    let classes = class;
    let mut columns = vec![vec![DEAD; representatives.len()]; alphabet.len()];
    for (new_state, &representative) in representatives.iter().enumerate() {
        for &(symbol, target) in &transitions[representative] {
            columns[symbol as usize][new_state] = classes[target as usize];
        }
    }
    Minimized {
        classes,
        columns: columns.into_iter().map(Vec::into_boxed_slice).collect(),
        byte_class,
        state_count: representatives.len(),
        rounds: pops,
    }
}


#[derive(Default)]
struct GroupedMinimizeStats {
    local_states: usize,
    local_rounds: usize,
    local_ms: f64,
    global_ms: f64,
    dag_groups: usize,
    dag_states: usize,
    hopcroft_groups: usize,
    hopcroft_states: usize,
}

/// Exact two-level minimization for the projected residual automaton.
/// Transitions never change terminal group, so language equivalence can first
/// be computed independently inside each disconnected group. Quotienting those
/// components cannot remove any cross-group equivalence; a second minimization
/// of their much smaller disjoint union performs exactly those remaining
/// cross-group merges.

/// Exact congruence for a deterministic component whose only cycles are
/// literal self-loops.  Removing self-loops yields a DAG, so child behavior is
/// already known in reverse topological order.  Equal structural signatures
/// are safe to merge.  This may intentionally over-distinguish states (for
/// example, a self-loop versus an edge to a language-equivalent descendant);
/// the final global minimizer recovers any such missed equivalences.
fn minimize_dag_with_self_loops(
    transitions: &[Vec<(u8, u32)>],
    states: &[u32],
    local_of_global: &[u32],
) -> Option<(Vec<u32>, Vec<u32>)> {
    let n = states.len();
    let mut indegree = vec![0u32; n];
    let mut outgoing = vec![Vec::<u32>::new(); n];
    for (local, &global) in states.iter().enumerate() {
        for &(_, target) in &transitions[global as usize] {
            if target == global {
                continue;
            }
            let target_local = local_of_global[target as usize];
            debug_assert_ne!(target_local, u32::MAX);
            outgoing[local].push(target_local);
            indegree[target_local as usize] += 1;
        }
    }
    let mut queue = VecDeque::<u32>::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::<u32>::with_capacity(n);
    while let Some(state) = queue.pop_front() {
        topo.push(state);
        for &target in &outgoing[state as usize] {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    if topo.len() != n {
        return None;
    }

    // Non-self targets are later in topological order, hence already assigned
    // when processing the order backwards.  u32::MAX denotes SELF and cannot
    // collide with an ordinary local class ID.
    let mut class = vec![u32::MAX; n];
    let mut class_ids = FxHashMap::<Vec<(u8, u32)>, u32>::default();
    let mut representatives = Vec::<u32>::new();
    for &local in topo.iter().rev() {
        let global = states[local as usize];
        let mut signature = Vec::with_capacity(transitions[global as usize].len());
        for &(symbol, target) in &transitions[global as usize] {
            let target_class = if target == global {
                u32::MAX
            } else {
                let target_local = local_of_global[target as usize];
                let target_class = class[target_local as usize];
                debug_assert_ne!(target_class, u32::MAX);
                target_class
            };
            signature.push((symbol, target_class));
        }
        let next = representatives.len() as u32;
        let state_class = *class_ids.entry(signature).or_insert_with(|| {
            representatives.push(global);
            next
        });
        class[local as usize] = state_class;
    }
    Some((class, representatives))
}


/// Exact congruence reduction for a deterministic component by its SCC DAG.
/// Most projected terminal residuals are a large DAG feeding a tiny cyclic core.
/// Singleton SCCs are structurally hashed bottom-up. Nontrivial SCCs are refined
/// only against themselves, treating already-reduced successor SCC classes as
/// fixed observations. The result may over-distinguish across SCC boundaries;
/// the final cross-group minimizer recovers any missed language equivalences.
struct FlatLocalTransitions {
    offsets: Vec<u32>,
    edges: Vec<(u8, u32)>,
}

impl FlatLocalTransitions {
    #[inline]
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    fn row(&self, state: usize) -> &[(u8, u32)] {
        let start = self.offsets[state] as usize;
        let end = self.offsets[state + 1] as usize;
        &self.edges[start..end]
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[inline]
fn transition_symbol_mask(row: &[(u8, u32)]) -> [u64; 4] {
    let mut mask = [0u64; 4];
    for &(symbol, _) in row {
        mask[symbol as usize >> 6] |= 1u64 << (symbol & 63);
    }
    mask
}

fn minimize_scc_dag(transitions: &FlatLocalTransitions) -> (Vec<u32>, Vec<u32>, usize, usize) {
    let n = transitions.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0, 0);
    }
    let profiling = std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some();
    let phase_started = profiling.then(Instant::now);

    // Kosaraju, ignoring literal self-loops (they do not affect SCC membership).
    let mut reverse = vec![Vec::<u32>::new(); n];
    for source in 0..n {
        for &(_, target) in transitions.row(source) {
            if target as usize != source {
                reverse[target as usize].push(source as u32);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut order = Vec::<u32>::with_capacity(n);
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root as u32, 0usize)];
        while let Some((state, edge_index)) = stack.pop() {
            let row = transitions.row(state as usize);
            let mut next_index = edge_index;
            let mut descended = false;
            while next_index < row.len() {
                let target = row[next_index].1;
                next_index += 1;
                if target == state || seen[target as usize] {
                    continue;
                }
                stack.push((state, next_index));
                seen[target as usize] = true;
                stack.push((target, 0));
                descended = true;
                break;
            }
            if !descended {
                order.push(state);
            }
        }
    }

    seen.fill(false);
    let mut component_of = vec![u32::MAX; n];
    let mut components = Vec::<Vec<u32>>::new();
    for &root in order.iter().rev() {
        if seen[root as usize] {
            continue;
        }
        let component = components.len() as u32;
        seen[root as usize] = true;
        let mut members = Vec::<u32>::new();
        let mut stack = vec![root];
        while let Some(state) = stack.pop() {
            component_of[state as usize] = component;
            members.push(state);
            for &source in &reverse[state as usize] {
                if !seen[source as usize] {
                    seen[source as usize] = true;
                    stack.push(source);
                }
            }
        }
        components.push(members);
    }

    let decompose_ms = phase_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let condensation_started = profiling.then(Instant::now);

    // Process sink SCCs first so every edge leaving the current SCC already has
    // a final reduced class.
    let mut successors = vec![Vec::<u32>::new(); components.len()];
    let mut predecessors = vec![Vec::<u32>::new(); components.len()];
    for source in 0..n {
        let source_component = component_of[source];
        for &(_, target) in transitions.row(source) {
            let target_component = component_of[target as usize];
            if target_component != source_component {
                successors[source_component as usize].push(target_component);
            }
        }
    }
    for edges in &mut successors {
        edges.sort_unstable();
        edges.dedup();
    }
    for (source, edges) in successors.iter().enumerate() {
        for &target in edges {
            predecessors[target as usize].push(source as u32);
        }
    }
    let mut remaining_successors = successors.iter().map(Vec::len).collect::<Vec<_>>();
    let mut ready = VecDeque::<u32>::new();
    for (component, &remaining) in remaining_successors.iter().enumerate() {
        if remaining == 0 {
            ready.push_back(component as u32);
        }
    }

    let condensation_ms = condensation_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let solve_started = profiling.then(Instant::now);
    let mut candidate_tests = 0usize;
    let mut candidate_states = 0usize;
    let mut candidate_matches = 0usize;
    let mut max_candidate_bucket = 0usize;

    let mut class = vec![u32::MAX; n];
    let mut representatives = Vec::<u32>::new();
    // Canonical quotient transition row for every solved language class.
    // Exact row lookup handles acyclic states without physical self-loops.
    // For a state with a physical self-loop on symbol b, any equivalent class C
    // must itself transition on b to C; index those candidates by b so the
    // fixed-point test only scans a tiny exact candidate set.
    let mut class_rows = Vec::<Vec<(u8, u32)>>::new();
    let mut row_ids = FxHashMap::<Vec<(u8, u32)>, u32>::default();
    let mut self_classes_by_shape =
        FxHashMap::<(u8, [u64; 4]), Vec<u32>>::default();
    let mut cyclic_components = 0usize;
    let mut cyclic_states = 0usize;
    let mut processed_components = 0usize;

    while let Some(component) = ready.pop_front() {
        processed_components += 1;
        let members = &components[component as usize];
        if members.len() == 1 {
            let state = members[0];
            let source_row = transitions.row(state as usize);

            // A singleton SCC may be language-equivalent to an already-solved
            // descendant. Without a physical self-loop its quotient row is fully
            // known, so exact row interning suffices. With a self-loop, candidate
            // C must semantically self-loop on that same symbol; test only those
            // indexed candidates and interpret physical SELF as C.
            let symbol_mask = transition_symbol_mask(source_row);
            let best_self_symbol = source_row
                .iter()
                .filter_map(|&(symbol, target)| (target == state).then_some(symbol))
                .min_by_key(|&symbol| {
                    self_classes_by_shape
                        .get(&(symbol, symbol_mask))
                        .map_or(0, Vec::len)
                });
            let state_class = if let Some(self_symbol) = best_self_symbol {
                let mut matched = None;
                let candidates = self_classes_by_shape
                    .get(&(self_symbol, symbol_mask))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                candidate_states += 1;
                candidate_tests += candidates.len();
                max_candidate_bucket = max_candidate_bucket.max(candidates.len());
                'candidate: for &candidate in candidates {
                    let candidate_row = &class_rows[candidate as usize];
                    if candidate_row.len() != source_row.len() {
                        continue;
                    }
                    for (&(source_symbol, source_target), &(candidate_symbol, candidate_target)) in
                        source_row.iter().zip(candidate_row)
                    {
                        if source_symbol != candidate_symbol {
                            continue 'candidate;
                        }
                        let source_target_class = if source_target == state {
                            candidate
                        } else {
                            let target_class = class[source_target as usize];
                            debug_assert_ne!(target_class, u32::MAX);
                            target_class
                        };
                        if source_target_class != candidate_target {
                            continue 'candidate;
                        }
                    }
                    matched = Some(candidate);
                    break;
                }
                if let Some(candidate) = matched {
                    candidate_matches += 1;
                    candidate
                } else {
                    let new_class = class_rows.len() as u32;
                    let row = source_row
                        .iter()
                        .map(|&(symbol, target)| {
                            let target_class = if target == state {
                                new_class
                            } else {
                                let target_class = class[target as usize];
                                debug_assert_ne!(target_class, u32::MAX);
                                target_class
                            };
                            (symbol, target_class)
                        })
                        .collect::<Vec<_>>();
                    let row_mask = transition_symbol_mask(&row);
                    for &(symbol, target_class) in &row {
                        if target_class == new_class {
                            self_classes_by_shape
                                .entry((symbol, row_mask))
                                .or_default()
                                .push(new_class);
                        }
                    }
                    row_ids.entry(row.clone()).or_insert(new_class);
                    class_rows.push(row);
                    representatives.push(state);
                    new_class
                }
            } else {
                let row = source_row
                    .iter()
                    .map(|&(symbol, target)| {
                        let target_class = class[target as usize];
                        debug_assert_ne!(target_class, u32::MAX);
                        (symbol, target_class)
                    })
                    .collect::<Vec<_>>();
                if let Some(&existing) = row_ids.get(&row) {
                    existing
                } else {
                    let new_class = class_rows.len() as u32;
                    row_ids.insert(row.clone(), new_class);
                    class_rows.push(row);
                    representatives.push(state);
                    new_class
                }
            };
            class[state as usize] = state_class;
        } else {
            cyclic_components += 1;
            cyclic_states += members.len();
            let mut member_index = vec![u32::MAX; n];
            for (index, &state) in members.iter().enumerate() {
                member_index[state as usize] = index as u32;
            }

            // Standard deterministic partition refinement only inside this tiny
            // SCC. External successor classes are fixed observations.
            let mut local_classes = vec![0u32; members.len()];
            loop {
                let mut ids = FxHashMap::<Vec<(u8, u64)>, u32>::default();
                let mut next_classes = vec![0u32; members.len()];
                for (local, &state) in members.iter().enumerate() {
                    let mut signature =
                        Vec::<(u8, u64)>::with_capacity(transitions.row(state as usize).len());
                    for &(symbol, target) in transitions.row(state as usize) {
                        let target_local = member_index[target as usize];
                        let behavior = if target_local != u32::MAX {
                            (1u64 << 63) | u64::from(local_classes[target_local as usize])
                        } else {
                            let target_class = class[target as usize];
                            debug_assert_ne!(target_class, u32::MAX);
                            u64::from(target_class)
                        };
                        signature.push((symbol, behavior));
                    }
                    let next = ids.len() as u32;
                    next_classes[local] = *ids.entry(signature).or_insert(next);
                }
                if next_classes == local_classes {
                    break;
                }
                local_classes = next_classes;
            }
            let local_count = local_classes
                .iter()
                .copied()
                .max()
                .map_or(0usize, |value| value as usize + 1);
            let base = class_rows.len() as u32;
            let mut local_representatives = vec![u32::MAX; local_count];
            for (local, &state) in members.iter().enumerate() {
                let local_class = local_classes[local] as usize;
                class[state as usize] = base + local_class as u32;
                if local_representatives[local_class] == u32::MAX {
                    local_representatives[local_class] = state;
                }
            }
            for (local_class, &representative) in local_representatives.iter().enumerate() {
                let global_class = base + local_class as u32;
                let row = transitions
                    .row(representative as usize)
                    .iter()
                    .map(|&(symbol, target)| {
                        let target_class = class[target as usize];
                        debug_assert_ne!(target_class, u32::MAX);
                        (symbol, target_class)
                    })
                    .collect::<Vec<_>>();
                let row_mask = transition_symbol_mask(&row);
                for &(symbol, target_class) in &row {
                    if target_class == global_class {
                        self_classes_by_shape
                            .entry((symbol, row_mask))
                            .or_default()
                            .push(global_class);
                    }
                }
                row_ids.entry(row.clone()).or_insert(global_class);
                class_rows.push(row);
                representatives.push(representative);
            }
        }

        for &predecessor in &predecessors[component as usize] {
            let remaining = &mut remaining_successors[predecessor as usize];
            *remaining -= 1;
            if *remaining == 0 {
                ready.push_back(predecessor);
            }
        }
    }
    debug_assert_eq!(processed_components, components.len());
    debug_assert!(class.iter().all(|&value| value != u32::MAX));
    debug_assert_eq!(class_rows.len(), representatives.len());
    if profiling {
        let solve_ms = solve_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        eprintln!(
            "[glrmask/profile][l1_projected_scc_reduce] states={} edges={} classes={} sccs={} cyclic_sccs={} cyclic_states={} decompose_ms={:.3} condensation_ms={:.3} solve_ms={:.3} candidate_states={} candidate_tests={} candidate_matches={} max_candidate_bucket={}",
            n,
            transitions.edge_count(),
            representatives.len(),
            components.len(),
            cyclic_components,
            cyclic_states,
            decompose_ms,
            condensation_ms,
            solve_ms,
            candidate_states,
            candidate_tests,
            candidate_matches,
            max_candidate_bucket,
        );
    }
    (class, representatives, cyclic_components, cyclic_states)
}

fn minimize_grouped_local(
    transitions: &[Vec<(u8, u32)>],
    groups: &[u32],
    alphabet: &[u8],
) -> (Minimized, GroupedMinimizeStats) {
    debug_assert_eq!(transitions.len(), groups.len());
    let group_count = groups
        .iter()
        .copied()
        .max()
        .map_or(0usize, |group| group as usize + 1);
    let mut states_by_group = vec![Vec::<u32>::new(); group_count];
    for (state, &group) in groups.iter().enumerate() {
        states_by_group[group as usize].push(state as u32);
    }

    let local_started = Instant::now();
    let mut reduced_of_state = vec![u32::MAX; transitions.len()];
    let mut local_of_global = vec![u32::MAX; transitions.len()];
    let mut reduced_representatives = Vec::<u32>::new();
    let mut local_rounds = 0usize;
    let mut dag_groups = 0usize;
    let mut dag_states = 0usize;
    let mut hopcroft_groups = 0usize;
    let mut hopcroft_states = 0usize;
    let dag_enabled = std::env::var("GLRMASK_L1_PROJECTED_DAG_MINIMIZE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let projected_edge_count = transitions.iter().map(Vec::len).sum::<usize>();
    let use_scc = std::env::var("GLRMASK_L1_PROJECTED_SCC_MINIMIZE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or_else(|_| {
            // Dense projected residual graphs are the regime where the local
            // components are large DAGs feeding tiny SCC cores.  SCC-DAG
            // reduction avoids repeated global refinement there; on sparse
            // residual graphs ordinary grouped minimization has lower overhead.
            transitions.len() >= 5_000
                && projected_edge_count >= transitions.len().saturating_mul(10)
        });

    for (group_id, states) in states_by_group
        .iter()
        .enumerate()
        .filter(|(_, states)| !states.is_empty())
    {
        for (local, &global) in states.iter().enumerate() {
            local_of_global[global as usize] = local as u32;
        }
        // Large projected terminal components in the p90 shapes are almost
        // DAGs but contain one tiny SCC. Probing them with a full DAG walk and
        // then rebuilding the same graph for SCC reduction doubles the edge
        // traffic. Both reducers are exact, so route large components directly
        // to SCC when that kernel is enabled; retain the cheaper DAG probe for
        // small components.
        let direct_scc = use_scc && states.len() >= 128;
        let dag_result = (!direct_scc && dag_enabled)
            .then(|| minimize_dag_with_self_loops(transitions, states, &local_of_global))
            .flatten();
        let (local_classes, representatives) = if let Some(result) = dag_result {
            dag_groups += 1;
            dag_states += states.len();
            result
        } else {
            hopcroft_groups += 1;
            hopcroft_states += states.len();
            let local_classes = if use_scc {
                let edge_capacity = states
                    .iter()
                    .map(|&global| transitions[global as usize].len())
                    .sum::<usize>();
                let mut offsets = Vec::with_capacity(states.len() + 1);
                let mut edges = Vec::with_capacity(edge_capacity);
                offsets.push(0);
                for &global in states {
                    for &(symbol, target) in &transitions[global as usize] {
                        debug_assert_eq!(groups[target as usize], groups[global as usize]);
                        let local_target = local_of_global[target as usize];
                        debug_assert_ne!(local_target, u32::MAX);
                        edges.push((symbol, local_target));
                    }
                    offsets.push(edges.len() as u32);
                }
                let local_transitions = FlatLocalTransitions { offsets, edges };
                let (classes, _representatives, cyclic_components, cyclic_states) =
                    minimize_scc_dag(&local_transitions);
                if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                    eprintln!(
                        "[glrmask/profile][l1_projected_group_scc] group={} states={} edges={} cyclic_sccs={} cyclic_states={} mode=scc_dag",
                        group_id,
                        states.len(),
                        local_transitions.edge_count(),
                        cyclic_components,
                        cyclic_states,
                    );
                }
                classes
            } else {
                let mut local_transitions = Vec::with_capacity(states.len());
                for &global in states {
                    let mut row = Vec::with_capacity(transitions[global as usize].len());
                    for &(symbol, target) in &transitions[global as usize] {
                        debug_assert_eq!(groups[target as usize], groups[global as usize]);
                        let local_target = local_of_global[target as usize];
                        debug_assert_ne!(local_target, u32::MAX);
                        row.push((symbol, local_target));
                    }
                    local_transitions.push(row);
                }
                let local_minimized = minimize(&local_transitions, alphabet);
                local_rounds += local_minimized.rounds;
                local_minimized.classes
            };
            let state_count = local_classes
                .iter()
                .copied()
                .max()
                .map_or(0usize, |class| class as usize + 1);
            let mut representatives = vec![u32::MAX; state_count];
            for (local, &global) in states.iter().enumerate() {
                let class = local_classes[local] as usize;
                if representatives[class] == u32::MAX {
                    representatives[class] = global;
                }
            }
            (local_classes, representatives)
        };

        let base = reduced_representatives.len() as u32;
        for (local, &global) in states.iter().enumerate() {
            reduced_of_state[global as usize] = base + local_classes[local];
        }
        reduced_representatives.extend(representatives);
        for &global in states {
            local_of_global[global as usize] = u32::MAX;
        }
    }
    let local_ms = local_started.elapsed().as_secs_f64() * 1000.0;

    let global_started = Instant::now();
    let mut reduced_transitions = Vec::with_capacity(reduced_representatives.len());
    for &representative in &reduced_representatives {
        reduced_transitions.push(
            transitions[representative as usize]
                .iter()
                .map(|&(symbol, target)| (symbol, reduced_of_state[target as usize]))
                .collect::<Vec<_>>(),
        );
    }
    let skip_global = std::env::var("GLRMASK_L1_PROJECTED_SKIP_GLOBAL_MINIMIZE")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        });
    let (classes, columns, byte_class, state_count, global_rounds) = if skip_global {
        let state_count = reduced_transitions.len();
        let mut byte_class = [0u8; 256];
        for (symbol, &byte) in alphabet.iter().enumerate() {
            byte_class[byte as usize] = symbol as u8;
        }
        let mut columns = vec![vec![DEAD; state_count]; alphabet.len()];
        for (source, row) in reduced_transitions.iter().enumerate() {
            for &(symbol, target) in row {
                columns[symbol as usize][source] = target;
            }
        }
        (
            reduced_of_state,
            columns.into_iter().map(Vec::into_boxed_slice).collect::<Vec<_>>(),
            byte_class,
            state_count,
            0usize,
        )
    } else {
        let reduced = minimize(&reduced_transitions, alphabet);
        let classes = reduced_of_state
            .into_iter()
            .map(|local_class| reduced.classes[local_class as usize])
            .collect::<Vec<_>>();
        (
            classes,
            reduced.columns,
            reduced.byte_class,
            reduced.state_count,
            reduced.rounds,
        )
    };
    let global_ms = global_started.elapsed().as_secs_f64() * 1000.0;
    (
        Minimized {
            classes,
            columns,
            byte_class,
            state_count,
            rounds: local_rounds + global_rounds,
        },
        GroupedMinimizeStats {
            local_states: reduced_representatives.len(),
            local_rounds,
            local_ms,
            global_ms,
            dag_groups,
            dag_states,
            hopcroft_groups,
            hopcroft_states,
        },
    )
}

fn minimize_grouped(
    transitions: &[Vec<(u8, u32)>],
    groups: &[u32],
    alphabet: &[u8],
) -> (Minimized, GroupedMinimizeStats) {
    let seeded = std::env::var("GLRMASK_L1_PROJECTED_GROUPED_SEEDED")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        });
    if seeded {
        minimize_grouped_seeded(transitions, groups, alphabet)
    } else {
        minimize_grouped_local(transitions, groups, alphabet)
    }
}

fn minimize_grouped_seeded(
    transitions: &[Vec<(u8, u32)>],
    groups: &[u32],
    alphabet: &[u8],
) -> (Minimized, GroupedMinimizeStats) {
    debug_assert_eq!(transitions.len(), groups.len());
    let local_started = Instant::now();
    let local = minimize_seeded(transitions, alphabet, Some(groups));
    let local_ms = local_started.elapsed().as_secs_f64() * 1000.0;

    let mut representatives = vec![u32::MAX; local.state_count];
    for (state, &class) in local.classes.iter().enumerate() {
        if representatives[class as usize] == u32::MAX {
            representatives[class as usize] = state as u32;
        }
    }
    debug_assert!(representatives.iter().all(|&state| state != u32::MAX));

    let global_started = Instant::now();
    let mut reduced_transitions = Vec::with_capacity(representatives.len());
    for &representative in &representatives {
        reduced_transitions.push(
            transitions[representative as usize]
                .iter()
                .map(|&(symbol, target)| (symbol, local.classes[target as usize]))
                .collect::<Vec<_>>(),
        );
    }
    let reduced = minimize(&reduced_transitions, alphabet);
    let global_ms = global_started.elapsed().as_secs_f64() * 1000.0;
    let classes = local
        .classes
        .iter()
        .map(|&local_class| reduced.classes[local_class as usize])
        .collect::<Vec<_>>();
    (
        Minimized {
            classes,
            columns: reduced.columns,
            byte_class: reduced.byte_class,
            state_count: reduced.state_count,
            rounds: local.rounds + reduced.rounds,
        },
        GroupedMinimizeStats {
            local_states: representatives.len(),
            local_rounds: local.rounds,
            local_ms,
            global_ms,
            dag_groups: 0,
            dag_states: 0,
            hopcroft_groups: 0,
            hopcroft_states: 0,
        },
    )
}

fn unique_vocab(input: BuildInput<'_>) -> (Vec<Vec<u32>>, Vec<Arc<[u8]>>) {
    let mut tokens = Vec::<Arc<[u8]>>::with_capacity(input.vocab.len());
    let mut aliases = Vec::<Vec<u32>>::with_capacity(input.vocab.len());

    let mut push = |id: u32, bytes: &Arc<[u8]>| {
        if tokens
            .last()
            .is_some_and(|token| token.as_ref() == bytes.as_ref())
        {
            aliases
                .last_mut()
                .expect("duplicate token has predecessor")
                .push(id);
        } else {
            tokens.push(Arc::clone(bytes));
            aliases.push(vec![id]);
        }
    };

    if let Some(parent) = input.subset_parent_order {
        // Split-L1 is a subset of an already sorted parent vocabulary. Filtering
        // that order is sufficient; do not construct another L1IdentityVocabOrder
        // (and especially do not build its packed-bucket metadata).
        let mut included = vec![false; parent.original_to_internal.len()];
        for (id, _) in input.vocab.iter() {
            included[id as usize] = true;
        }
        for (id, bytes) in parent.token_entries_sorted.iter() {
            if included[*id as usize] {
                push(*id, bytes);
            }
        }
    } else {
        let order = super::super::prepared_l1_identity_vocab_order(input.vocab);
        for (id, bytes) in order.token_entries_sorted.iter() {
            push(*id, bytes);
        }
    }
    debug_assert_eq!(
        aliases.iter().map(Vec::len).sum::<usize>(),
        input.vocab.len()
    );
    (aliases, tokens)
}

/// Quotient vocabulary bytes by their exact raw tokenizer transition column.
/// Equality here is stronger than needed but cheap to prove: identical direct
/// transitions from every raw state imply identical `step_all` results from
/// every epsilon-closed configuration.
fn quotient_input_bytes(input: BuildInput<'_>, bytes: &[u8]) -> (Vec<u8>, [u8; 256]) {
    let states = input.tokenizer.num_states() as usize;
    let mut column_ids = FxHashMap::<Vec<u32>, u8>::default();
    let mut representatives = Vec::<u8>::new();
    let mut representative = [0u8; 256];
    for &byte in bytes {
        let column = if let Some(by_byte) = input.transitions_by_byte {
            by_byte[byte as usize * states..(byte as usize + 1) * states].to_vec()
        } else {
            (0..states)
                .map(|state| input.flat_trans[state * 256 + byte as usize])
                .collect::<Vec<_>>()
        };
        let class = if let Some(&class) = column_ids.get(&column) {
            class
        } else {
            let class = representatives.len() as u8;
            column_ids.insert(column, class);
            representatives.push(byte);
            class
        };
        representative[byte as usize] = representatives[class as usize];
    }
    (representatives, representative)
}

/// Exact finite-vocabulary base case for one-byte partitions.
///
/// This is the same single-group semantics without constructing the closure of
/// the residual language beyond the only byte each token can consume. It is a
/// base case of the algorithm, not a fallback to the production builder.
fn build_one_byte(
    input: BuildInput<'_>,
    aliases: &[Vec<u32>],
    tokens: &[Arc<[u8]>],
    total: Instant,
) -> Option<LocalIdMapTerminalDwa> {
    let scan = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut raw_to_start = Vec::with_capacity(input.tokenizer.num_states() as usize);
    for raw in 0..input.tokenizer.num_states() {
        raw_to_start.push(scanner.start(raw));
    }
    let mut raw_by_start = vec![Vec::<u32>::new(); scanner.configs.len()];
    for (raw, &start) in raw_to_start.iter().enumerate() {
        raw_by_start[start as usize].push(raw as u32);
    }

    let token_bytes = tokens.iter().map(|token| token[0]).collect::<Vec<_>>();
    let (byte_representatives, byte_representative) = quotient_input_bytes(input, &token_bytes);
    let mut representative_index = [usize::MAX; 256];
    for (index, &byte) in byte_representatives.iter().enumerate() {
        representative_index[byte as usize] = index;
    }

    let mut state_class = vec![0u32; input.tokenizer.num_states() as usize];
    let mut row_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut rows = Vec::<Vec<u32>>::new();
    for (start, raw_states) in raw_by_start.iter().enumerate() {
        if raw_states.is_empty() {
            continue;
        }
        let singleton = (scanner.configs[start].len() == 1).then_some(scanner.configs[start][0]);
        let representative_signatures = byte_representatives
            .iter()
            .map(|&byte| {
                if let Some(state) = singleton {
                    let target = input.flat_trans[state as usize * 256 + byte as usize];
                    if target == DEAD {
                        0
                    } else {
                        scanner.signature(raw_to_start[target as usize])
                    }
                } else {
                    let target = scanner.step(start as u32, byte);
                    scanner.signature(target)
                }
            })
            .collect::<Vec<_>>();
        let row = token_bytes
            .iter()
            .map(|&byte| {
                let representative = byte_representative[byte as usize];
                representative_signatures[representative_index[representative as usize]]
            })
            .collect::<Vec<_>>();
        let next = rows.len() as u32;
        let class = *row_ids.entry(row.clone()).or_insert_with(|| {
            rows.push(row);
            next
        });
        for &raw in raw_states {
            state_class[raw as usize] = class;
        }
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.0;
    let finished = common::finish(
        input,
        aliases,
        &scanner.signatures,
        state_class,
        rows,
        scan_ms,
        || total.elapsed().as_secs_f64() * 1000.0,
    )?;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_single_one_byte] partition={} raw_states={} tokens={} byte_classes={} configs={} signatures={} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            tokens.len(),
            byte_representatives.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            finished.state_classes,
            finished.token_classes,
            scan_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}

/// Deterministic reverse subset construction for the all-live DFA.
///
/// For a byte string `x`, a subset state denotes
/// `A_x = { q | delta(q, x) is live }`. The empty suffix starts at every live
/// state, and prepending `b` is the exact preimage
/// `A_{bx} = { q | delta(q, b) in A_x }`. Walking a token backwards therefore
/// yields its complete acceptance column over every minimized residual state.
struct ReverseColumn {
    offsets: Box<[u32]>,
    sources: Box<[u32]>,
    live_sources: Box<[u64]>,
}

struct ReverseSubsets<'a> {
    columns: Vec<ReverseColumn>,
    byte_class: &'a [u8; 256],
    state_count: usize,
    sets: Vec<Box<[u64]>>,
    ids: FxHashMap<Vec<u64>, u32>,
    cache: Vec<Box<[u32]>>,
    computed_transitions: usize,
    target_visits: usize,
    predecessor_visits: usize,
}

impl<'a> ReverseSubsets<'a> {
    fn new(columns: &'a [Box<[u32]>], byte_class: &'a [u8; 256], state_count: usize) -> Self {
        let words = state_count.div_ceil(64);
        let mut all = vec![u64::MAX; words];
        if let Some(last) = all.last_mut() {
            let remainder = state_count % 64;
            if remainder != 0 {
                *last = (1u64 << remainder) - 1;
            }
        }
        let reverse_columns = columns
            .iter()
            .map(|column| {
                let mut counts = vec![0u32; state_count];
                let mut live_sources = vec![0u64; words];
                for (source, &target) in column.iter().enumerate() {
                    if target != DEAD {
                        counts[target as usize] += 1;
                        live_sources[source / 64] |= 1u64 << (source % 64);
                    }
                }
                let mut offsets = vec![0u32; state_count + 1];
                for target in 0..state_count {
                    offsets[target + 1] = offsets[target] + counts[target];
                }
                let mut next = offsets[..state_count].to_vec();
                let mut sources = vec![0u32; offsets[state_count] as usize];
                for (source, &target) in column.iter().enumerate() {
                    if target != DEAD {
                        let slot = &mut next[target as usize];
                        sources[*slot as usize] = source as u32;
                        *slot += 1;
                    }
                }
                ReverseColumn {
                    offsets: offsets.into_boxed_slice(),
                    sources: sources.into_boxed_slice(),
                    live_sources: live_sources.into_boxed_slice(),
                }
            })
            .collect();
        let mut result = Self {
            columns: reverse_columns,
            byte_class,
            state_count,
            sets: Vec::new(),
            ids: FxHashMap::default(),
            cache: Vec::new(),
            computed_transitions: 0,
            target_visits: 0,
            predecessor_visits: 0,
        };
        let start = result.intern(all);
        debug_assert_eq!(start, 0);
        result
    }

    fn intern(&mut self, set: Vec<u64>) -> u32 {
        if let Some(&id) = self.ids.get(&set) {
            return id;
        }
        let id = self.sets.len() as u32;
        self.ids.insert(set.clone(), id);
        self.sets.push(set.into_boxed_slice());
        self.cache
            .push(vec![UNKNOWN; self.columns.len()].into_boxed_slice());
        id
    }

    #[inline]
    fn contains(set: &[u64], state: u32) -> bool {
        set[state as usize / 64] & (1u64 << (state as usize % 64)) != 0
    }

    fn visit_targets(set: &[u64], state_count: usize, included: bool, mut visit: impl FnMut(usize)) {
        for (word_index, &set_word) in set.iter().enumerate() {
            let mut word = if included { set_word } else { !set_word };
            if word_index + 1 == set.len() && !state_count.is_multiple_of(64) {
                word &= (1u64 << (state_count % 64)) - 1;
            }
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                visit(word_index * 64 + bit);
                word &= word - 1;
            }
        }
    }

    fn prepend(&mut self, suffix: u32, byte: u8) -> u32 {
        let symbol = self.byte_class[byte as usize] as usize;
        let cached = self.cache[suffix as usize][symbol];
        if cached != UNKNOWN {
            return cached;
        }
        let suffix_set = &self.sets[suffix as usize];
        let column = &self.columns[symbol];
        let included_targets = suffix_set.iter().map(|word| word.count_ones() as usize).sum::<usize>();
        // Choosing by target cardinality avoids a separate degree-summing pass.
        // In these deterministic columns total predecessor work is linear in
        // sources, while the target scan was the dominant cost.
        let use_included = included_targets <= self.state_count - included_targets;
        let mut predecessor = if use_included {
            vec![0u64; suffix_set.len()]
        } else {
            column.live_sources.to_vec()
        };
        let mut target_visits = 0usize;
        let mut predecessor_visits = 0usize;
        Self::visit_targets(suffix_set, self.state_count, use_included, |target| {
            target_visits += 1;
            let start = column.offsets[target] as usize;
            let end = column.offsets[target + 1] as usize;
            predecessor_visits += end - start;
            for &source in &column.sources[start..end] {
                let word = &mut predecessor[source as usize / 64];
                let bit = 1u64 << (source as usize % 64);
                if use_included {
                    *word |= bit;
                } else {
                    *word &= !bit;
                }
            }
        });
        self.computed_transitions += 1;
        self.target_visits += target_visits;
        self.predecessor_visits += predecessor_visits;
        let target = self.intern(predecessor);
        self.cache[suffix as usize][symbol] = target;
        target
    }

    fn token(&mut self, token: &[u8]) -> u32 {
        token
            .iter()
            .rev()
            .fold(0, |suffix, &byte| self.prepend(suffix, byte))
    }
}

const PROJECTED_CELL_BUDGET: usize = 4_000_000;
const PROJECTED_SUBSET_CELL_BUDGET: usize = 1_500_000;

fn projected_cell_budget(input: BuildInput<'_>) -> usize {
    let (env_name, default) = if input.subset_parent_order.is_some() {
        (
            "GLRMASK_L1_SINGLE_PROJECTED_SUBSET_CELL_BUDGET",
            PROJECTED_SUBSET_CELL_BUDGET,
        )
    } else {
        (
            "GLRMASK_L1_SINGLE_PROJECTED_CELL_BUDGET",
            PROJECTED_CELL_BUDGET,
        )
    };
    std::env::var(env_name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn projected_shape(input: BuildInput<'_>) -> (usize, usize) {
    let mut relevant = [false; 256];
    for (_, token) in input.vocab.iter() {
        for &byte in token {
            relevant[byte as usize] = true;
        }
    }
    let vocab_bytes = relevant.iter().filter(|&&used| used).count();
    let memberships = (0..input.tokenizer.num_states())
        .map(|state| {
            super::super::collect_active_terminal_signature(
                input.tokenizer,
                state,
                input.active_terminals,
            )
            .len()
        })
        .sum();
    (memberships, vocab_bytes)
}

/// Return whether the legacy experimental auto-selector would choose projected.
/// Production does not call this: projected is the unconditional default.
pub(super) fn should_use_projected(input: BuildInput<'_>) -> bool {
    let generic_epsilon = input.tokenizer.has_epsilon_transitions()
        && !input.tokenizer.has_scalar_deterministic_dispatch();
    if !generic_epsilon {
        return true;
    }
    let (memberships, vocab_bytes) = projected_shape(input);
    memberships.saturating_mul(vocab_bytes) <= projected_cell_budget(input)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectedKernel {
    Finite,
    Residual,
}

fn projected_kernel(input: BuildInput<'_>) -> ProjectedKernel {
    if let Ok(value) = std::env::var("GLRMASK_L1_PROJECTED_KERNEL") {
        return match value.trim().to_ascii_lowercase().as_str() {
            "finite" | "vocab" | "trie" => ProjectedKernel::Finite,
            "residual" | "dfa" => ProjectedKernel::Residual,
            "" | "auto" => projected_kernel_auto(input),
            other => panic!(
                "unknown GLRMASK_L1_PROJECTED_KERNEL={other:?}; expected auto, finite, or residual"
            ),
        };
    }
    projected_kernel_auto(input)
}


fn profile_vector_projection_shape(input: BuildInput<'_>) {
    let Ok(scope) = std::env::var("GLRMASK_L1_PROJECTED_VECTOR_PROFILE") else {
        return;
    };
    let scope = scope.trim();
    if !scope.is_empty() && scope != "1" && scope != input.partition_label {
        return;
    }
    let started = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut raw_roots = Vec::with_capacity(input.tokenizer.num_states() as usize);
    for raw in 0..input.tokenizer.num_states() {
        raw_roots.push(scanner.start(raw));
    }
    let mut vocab_bytes = input.vocab.relevant_bytes().iter().copied().collect::<Vec<_>>();
    vocab_bytes.sort_unstable();
    let (bytes, _) = quotient_input_bytes(input, &vocab_bytes);
    let mut transitions = Vec::<Vec<(u8, u32)>>::new();
    let mut state = 0usize;
    while state < scanner.configs.len() {
        let mut row = Vec::new();
        for (symbol, &byte) in bytes.iter().enumerate() {
            let target = scanner.step(state as u32, byte);
            if target != DEAD {
                row.push((symbol as u8, target));
            }
        }
        transitions.push(row);
        state += 1;
    }
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    let outputs = (0..scanner.configs.len())
        .map(|state| scanner.signature(state as u32))
        .collect::<Vec<_>>();
    let minimize_started = Instant::now();
    let minimized = minimize_seeded(&transitions, &bytes, Some(&outputs));
    let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
    let mut root_classes = raw_roots
        .iter()
        .map(|&root| minimized.classes[root as usize])
        .collect::<Vec<_>>();
    root_classes.sort_unstable();
    root_classes.dedup();
    eprintln!(
        "[glrmask/profile][l1_projected_vector] partition={} configs={} edges={} output_signatures={} minimized_states={} raw_root_classes={} build_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
        input.partition_label,
        transitions.len(),
        transitions.iter().map(Vec::len).sum::<usize>(),
        scanner.signatures.len(),
        minimized.state_count,
        root_classes.len(),
        build_ms,
        minimize_ms,
        started.elapsed().as_secs_f64() * 1000.0,
    );
}

fn projected_kernel_auto(input: BuildInput<'_>) -> ProjectedKernel {
    let active_terminals = input.active_terminals.iter().filter(|&&active| active).count();
    let vocab_tokens = input.vocab.len();
    let vocab_bytes = input.vocab.relevant_bytes().len();

    // Both kernels are exact projections of the same finite-vocabulary L1
    // relation. Residual projection is strongest for small terminal families,
    // tiny vocabularies, and very wide vocabularies where enumerating trie
    // behavior creates a large state×token matrix. Finite projection wins when
    // many terminal residuals share behavior over a moderate vocabulary, most
    // notably split-L1 subsets.
    if active_terminals <= 64
        || vocab_tokens <= 32
        || vocab_tokens > 18_000
        || (active_terminals >= 120 && vocab_tokens > 3_000 && vocab_bytes > 128)
    {
        ProjectedKernel::Residual
    } else {
        ProjectedKernel::Finite
    }
}

/// Build L1 only through projected representations. `quotient` remains an
/// explicit diagnostic/reference implementation but is never used here.
pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    profile_vector_projection_shape(input);
    let kernel = projected_kernel(input);
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_projected_plan] partition={} active_terminals={} vocab_tokens={} vocab_bytes={} kernel={:?}",
            input.partition_label,
            input.active_terminals.iter().filter(|&&active| active).count(),
            input.vocab.len(),
            input.vocab.relevant_bytes().len(),
            kernel,
        );
    }
    match kernel {
        ProjectedKernel::Finite => build_finite_projected(input),
        ProjectedKernel::Residual => build_binary(input),
    }
}


#[derive(Default)]
struct FiniteTrieNode {
    token: Option<usize>,
    edge_token: usize,
    edge_start: u32,
    edge_end: u32,
    subtree_start: u32,
    subtree_end: u32,
    subtree_bytes: [u64; 4],
    children: Vec<u32>,
}

struct FiniteTrie {
    nodes: Vec<FiniteTrieNode>,
}

struct FiniteVocabProjection {
    aliases: Vec<Vec<u32>>,
    tokens: Vec<Arc<[u8]>>,
    trie: FiniteTrie,
}

impl crate::vocab::VocabDerivedArtifact for FiniteVocabProjection {}

fn build_finite_vocab_projection(vocab: &Vocab) -> Arc<FiniteVocabProjection> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<FiniteVocabProjection>() {
        return cached;
    }
    let order = super::super::prepared_l1_identity_vocab_order(vocab);
    let mut tokens = Vec::<Arc<[u8]>>::with_capacity(vocab.len());
    let mut aliases = Vec::<Vec<u32>>::with_capacity(vocab.len());
    for (id, bytes) in order.token_entries_sorted.iter() {
        if tokens
            .last()
            .is_some_and(|token| token.as_ref() == bytes.as_ref())
        {
            aliases
                .last_mut()
                .expect("duplicate token has predecessor")
                .push(*id);
        } else {
            tokens.push(Arc::clone(bytes));
            aliases.push(vec![*id]);
        }
    }
    debug_assert_eq!(aliases.iter().map(Vec::len).sum::<usize>(), vocab.len());
    let trie = FiniteTrie::build(&tokens);
    let projection = Arc::new(FiniteVocabProjection {
        aliases,
        tokens,
        trie,
    });
    vocab.vocab_derived_cache_set(Arc::clone(&projection));
    projection
}

pub(crate) fn prepare_finite_vocab_projection(vocab: &Vocab) {
    if vocab.len() >= 50_000 {
        let _ = build_finite_vocab_projection(vocab);
    }
}

fn finite_vocab_projection(input: BuildInput<'_>) -> (Arc<FiniteVocabProjection>, bool, f64) {
    let started = Instant::now();
    if input.subset_parent_order.is_none() {
        let cached = input
            .vocab
            .vocab_derived_cache_get::<FiniteVocabProjection>();
        if let Some(cached) = cached {
            return (cached, true, started.elapsed().as_secs_f64() * 1000.0);
        }
        let projection = build_finite_vocab_projection(input.vocab);
        return (
            projection,
            false,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let (aliases, tokens) = unique_vocab(input);
    let trie = FiniteTrie::build(&tokens);
    (
        Arc::new(FiniteVocabProjection {
            aliases,
            tokens,
            trie,
        }),
        false,
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

impl FiniteTrie {
    /// Build a compressed radix trie directly from the byte-sorted unique
    /// vocabulary. Edge labels are slices of one vocabulary token, so the trie
    /// allocates no copied edge bytes and carries no unrelated subtree metadata.
    fn build(tokens: &[Arc<[u8]>]) -> Self {
        fn lcp(first: &[u8], last: &[u8], from: usize) -> usize {
            let mut index = from;
            let end = first.len().min(last.len());
            while index < end && first[index] == last[index] {
                index += 1;
            }
            index
        }

        fn add_range(
            tokens: &[Arc<[u8]>],
            begin: usize,
            end: usize,
            parent_prefix_len: usize,
            nodes: &mut Vec<FiniteTrieNode>,
        ) -> u32 {
            debug_assert!(begin < end);
            let first = tokens[begin].as_ref();
            let last = tokens[end - 1].as_ref();
            let prefix_len = lcp(first, last, parent_prefix_len);
            let token = (first.len() == prefix_len).then_some(begin);
            let node = nodes.len() as u32;
            nodes.push(FiniteTrieNode {
                token,
                edge_token: begin,
                edge_start: parent_prefix_len as u32,
                edge_end: prefix_len as u32,
                subtree_start: begin as u32,
                subtree_end: end as u32,
                subtree_bytes: [0; 4],
                children: Vec::new(),
            });
            let mut cursor = begin + usize::from(token.is_some());
            while cursor < end {
                let byte = tokens[cursor][prefix_len];
                let child_begin = cursor;
                cursor += 1;
                while cursor < end && tokens[cursor][prefix_len] == byte {
                    cursor += 1;
                }
                let child = add_range(tokens, child_begin, cursor, prefix_len, nodes);
                nodes[node as usize].children.push(child);
            }
            let children = nodes[node as usize].children.clone();
            let mut subtree_bytes = [0u64; 4];
            for child in children {
                let child_node = &nodes[child as usize];
                let edge = &tokens[child_node.edge_token]
                    [child_node.edge_start as usize..child_node.edge_end as usize];
                for &byte in edge {
                    subtree_bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
                }
                for (dst, &src) in subtree_bytes.iter_mut().zip(child_node.subtree_bytes.iter()) {
                    *dst |= src;
                }
            }
            nodes[node as usize].subtree_bytes = subtree_bytes;
            node
        }

        if tokens.is_empty() {
            return Self { nodes: vec![FiniteTrieNode::default()] };
        }
        let mut nodes = Vec::new();
        let root = add_range(tokens, 0, tokens.len(), 0, &mut nodes);
        debug_assert_eq!(root, 0);
        // If all tokens have a non-empty common prefix, `add_range` puts that
        // prefix on node zero. Split it into an empty root so the first cached
        // transition is still one common vocabulary bucket.
        if nodes[0].edge_end != 0 {
            let old = std::mem::take(&mut nodes);
            let mut shifted = Vec::with_capacity(old.len() + 1);
            shifted.push(FiniteTrieNode {
                children: vec![1],
                ..FiniteTrieNode::default()
            });
            shifted.extend(old.into_iter().map(|mut node| {
                for child in &mut node.children {
                    *child += 1;
                }
                node
            }));
            nodes = shifted;
        }
        Self { nodes }
    }

    #[inline]
    fn edge<'a>(&self, node: u32, tokens: &'a [Arc<[u8]>]) -> &'a [u8] {
        let node = &self.nodes[node as usize];
        &tokens[node.edge_token][node.edge_start as usize..node.edge_end as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProfileRun {
    start: u32,
    end: u32,
    signature: u32,
}

#[inline]
fn push_profile_run(out: &mut Vec<ProfileRun>, start: u32, end: u32, signature: u32) {
    if signature == 0 || start == end {
        return;
    }
    if let Some(last) = out.last_mut()
        && last.end == start
        && last.signature == signature
    {
        last.end = end;
        return;
    }
    out.push(ProfileRun { start, end, signature });
}

fn collect_finite_profile(
    scanner: &mut Scanner<'_>,
    trie: &FiniteTrie,
    tokens: &[Arc<[u8]>],
    self_loops: &[crate::ds::u8set::U8Set],
    node: u32,
    config: u32,
    out: &mut Vec<ProfileRun>,
    pair_visits: &mut usize,
    uniform_subtrees: &mut usize,
    uniform_tokens: &mut usize,
) {
    *pair_visits += 1;
    let current = &trie.nodes[node as usize];
    if let Some(state) = scanner.singleton_state(config) {
        let loops = self_loops[state as usize];
        let covered = crate::ds::u8set::U8Set::from_words(current.subtree_bytes).is_subset(&loops);
        if covered {
            let signature = scanner.signature(config);
            push_profile_run(
                out,
                current.subtree_start,
                current.subtree_end,
                signature,
            );
            *uniform_subtrees += 1;
            *uniform_tokens += (current.subtree_end - current.subtree_start) as usize;
            return;
        }
    }
    if let Some(token) = current.token {
        let signature = scanner.signature(config);
        push_profile_run(out, token as u32, token as u32 + 1, signature);
    }
    for &child in &current.children {
        let target = scanner.step_bytes(config, trie.edge(child, tokens));
        if target != DEAD {
            collect_finite_profile(
                scanner, trie, tokens, self_loops, child, target, out, pair_visits,
                uniform_subtrees, uniform_tokens,
            );
        }
    }
}

#[derive(Clone, Copy)]
struct FiniteRowEvent {
    position: u32,
    row: u32,
    signature: u32,
}

/// Collision-free canonical ID for a mutable vector of finite-row signatures.
/// Leaves and internal pairs are hash-consed, so the root ID is equal iff the
/// entire vector is equal. One row update costs O(log rows) and never copies the
/// full state-class vector.
struct CanonicalSignatureVector {
    base: usize,
    tree: Vec<u32>,
    leaf_ids: FxHashMap<u32, u32>,
    pair_ids: FxHashMap<(u32, u32), u32>,
    next_id: u32,
}

impl CanonicalSignatureVector {
    fn new(width: usize) -> Self {
        let base = width.max(1).next_power_of_two();
        let mut this = Self {
            base,
            tree: vec![0; base * 2],
            leaf_ids: FxHashMap::default(),
            pair_ids: FxHashMap::default(),
            next_id: 0,
        };
        let zero = this.intern_leaf(0);
        this.tree[base..].fill(zero);
        for node in (1..base).rev() {
            let left = this.tree[node * 2];
            let right = this.tree[node * 2 + 1];
            this.tree[node] = this.intern_pair(left, right);
        }
        this
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn intern_leaf(&mut self, value: u32) -> u32 {
        if let Some(&id) = self.leaf_ids.get(&value) {
            return id;
        }
        let id = self.alloc_id();
        self.leaf_ids.insert(value, id);
        id
    }

    fn intern_pair(&mut self, left: u32, right: u32) -> u32 {
        if let Some(&id) = self.pair_ids.get(&(left, right)) {
            return id;
        }
        let id = self.alloc_id();
        self.pair_ids.insert((left, right), id);
        id
    }

    fn set(&mut self, index: usize, value: u32) {
        debug_assert!(index < self.base);
        let mut node = self.base + index;
        let leaf = self.intern_leaf(value);
        if self.tree[node] == leaf {
            return;
        }
        self.tree[node] = leaf;
        node /= 2;
        while node != 0 {
            let left = self.tree[node * 2];
            let right = self.tree[node * 2 + 1];
            let id = self.intern_pair(left, right);
            if self.tree[node] == id {
                break;
            }
            self.tree[node] = id;
            node /= 2;
        }
    }

    fn root(&self) -> u32 {
        self.tree[1]
    }
}

fn finite_compact_runs(
    root_token: Option<usize>,
    class_fingerprints: &[Vec<u32>],
    profiles: &[Arc<[ProfileRun]>],
    token_count: usize,
) -> (Vec<u32>, Vec<usize>, Vec<Vec<u32>>, usize, usize) {
    let mut rows = Vec::<Vec<ProfileRun>>::with_capacity(class_fingerprints.len());
    let mut events = Vec::<FiniteRowEvent>::new();
    let mut referenced_runs = 0usize;
    for (row_index, fingerprint) in class_fingerprints.iter().enumerate() {
        let mut row = Vec::<ProfileRun>::new();
        let mut offset = 0usize;
        if let Some(token) = root_token {
            let signature = fingerprint[0];
            push_profile_run(&mut row, token as u32, token as u32 + 1, signature);
            offset = 1;
        }
        for &profile in &fingerprint[offset..] {
            for run in profiles[profile as usize].iter() {
                push_profile_run(&mut row, run.start, run.end, run.signature);
            }
        }
        referenced_runs += row.len();
        for run in &row {
            events.push(FiniteRowEvent {
                position: run.start,
                row: row_index as u32,
                signature: run.signature,
            });
            events.push(FiniteRowEvent {
                position: run.end,
                row: row_index as u32,
                signature: 0,
            });
        }
        rows.push(row);
    }

    // At an adjacent run boundary the old run must end before the new one
    // starts. Sorting zero-signature events first makes the final value exact.
    events.sort_unstable_by_key(|event| {
        (event.position, event.row, u8::from(event.signature != 0))
    });
    let mut vector = CanonicalSignatureVector::new(class_fingerprints.len());
    let mut class_by_root = FxHashMap::<u32, u32>::default();
    let mut token_class = vec![0u32; token_count];
    let mut token_reps = Vec::<usize>::new();
    let mut position = 0usize;
    let mut event_index = 0usize;
    while position < token_count {
        while event_index < events.len() && events[event_index].position as usize == position {
            let event = events[event_index];
            vector.set(event.row as usize, event.signature);
            event_index += 1;
        }
        let next_position = events
            .get(event_index)
            .map_or(token_count, |event| event.position as usize)
            .min(token_count);
        debug_assert!(next_position > position || event_index < events.len());
        if next_position > position {
            let root = vector.root();
            let class = if let Some(&class) = class_by_root.get(&root) {
                class
            } else {
                let class = token_reps.len() as u32;
                class_by_root.insert(root, class);
                token_reps.push(position);
                class
            };
            token_class[position..next_position].fill(class);
            position = next_position;
        }
    }

    let mut reps_sorted = token_reps
        .iter()
        .enumerate()
        .map(|(class, &token)| (token, class))
        .collect::<Vec<_>>();
    reps_sorted.sort_unstable();
    let mut compact_rows = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut compact = vec![0u32; token_reps.len()];
        let mut run_index = 0usize;
        for &(token, class) in &reps_sorted {
            while run_index < row.len() && row[run_index].end as usize <= token {
                run_index += 1;
            }
            if let Some(run) = row.get(run_index)
                && run.start as usize <= token
                && token < run.end as usize
            {
                compact[class] = run.signature;
            }
        }
        compact_rows.push(compact);
    }
    (token_class, token_reps, compact_rows, events.len(), referenced_runs)
}

/// Exact finite-vocabulary L1 projection.
///
/// For one first radix bucket, a raw lexer state matters only through the lexer
/// configuration reached after that bucket's common prefix. Equal targets share
/// the complete computation over the remaining finite suffix set. The resulting
/// per-bucket profile IDs form an exact short fingerprint for the raw state;
/// only fingerprints that survive as state classes are materialized into rows.
fn build_finite_projected(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
    let (finite_vocab, finite_vocab_cache_hit, prep_ms) = finite_vocab_projection(input);
    let aliases = finite_vocab.aliases.as_slice();
    let tokens = finite_vocab.tokens.as_slice();
    let trie = &finite_vocab.trie;

    let scan_started = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut starts = BTreeMap::<u32, Vec<u32>>::new();
    for raw in 0..input.tokenizer.num_states() {
        starts.entry(scanner.start(raw)).or_default().push(raw);
    }

    let root_token = trie.nodes[0].token;
    let root_children = trie.nodes[0].children.clone();
    let mut state_class = vec![0u32; input.tokenizer.num_states() as usize];
    let mut class_fingerprints = Vec::<Vec<u32>>::new();
    let mut class_ids = FxHashMap::<Vec<u32>, u32>::default();

    let mut profiles = vec![Arc::<[ProfileRun]>::from([])];
    let mut profile_ids = FxHashMap::<Arc<[ProfileRun]>, u32>::default();
    let mut bucket_cache = FxHashMap::<(u32, u32), u32>::default();
    let mut cache_hits = 0usize;
    let mut pair_visits = 0usize;
    let mut uniform_subtrees = 0usize;
    let mut uniform_tokens = 0usize;
    let self_loops = input.tokenizer.all_self_loop_bytes();

    for (start, raw_states) in starts {
        let mut fingerprint = Vec::with_capacity(root_children.len() + usize::from(root_token.is_some()));
        if root_token.is_some() {
            fingerprint.push(scanner.signature(start));
        }
        for &child in &root_children {
            let target = scanner.step_bytes(start, trie.edge(child, tokens));
            if target == DEAD {
                fingerprint.push(0);
                continue;
            }
            let key = (child, target);
            let profile = if let Some(&profile) = bucket_cache.get(&key) {
                cache_hits += 1;
                profile
            } else {
                let mut values = Vec::new();
                collect_finite_profile(
                    &mut scanner,
                    trie,
                    tokens,
                    self_loops.as_ref(),
                    child,
                    target,
                    &mut values,
                    &mut pair_visits,
                    &mut uniform_subtrees,
                    &mut uniform_tokens,
                );
                let values: Arc<[ProfileRun]> = Arc::from(values);
                let profile = if values.is_empty() {
                    0
                } else if let Some(&profile) = profile_ids.get(&values) {
                    profile
                } else {
                    let profile = profiles.len() as u32;
                    profile_ids.insert(Arc::clone(&values), profile);
                    profiles.push(values);
                    profile
                };
                bucket_cache.insert(key, profile);
                profile
            };
            fingerprint.push(profile);
        }

        let next = class_fingerprints.len() as u32;
        let class = *class_ids.entry(fingerprint.clone()).or_insert_with(|| {
            class_fingerprints.push(fingerprint);
            next
        });
        for raw in raw_states {
            state_class[raw as usize] = class;
        }
    }
    let scan_ms = scan_started.elapsed().as_secs_f64() * 1000.0;

    let materialize_started = Instant::now();
    let use_run_sweep = std::env::var_os("GLRMASK_DISABLE_L1_FINITE_RUN_SWEEP").is_none();
    let (finished, run_sweep_events, referenced_runs) = if use_run_sweep {
        let compact_started = Instant::now();
        let (token_class, _token_reps, compact_rows, events, referenced_runs) =
            finite_compact_runs(
                root_token,
                &class_fingerprints,
                &profiles,
                aliases.len(),
            );
        let compact_ms = compact_started.elapsed().as_secs_f64() * 1000.0;
        let finished = common::finish_compacted(
            input,
            aliases,
            &scanner.signatures,
            state_class,
            compact_rows,
            token_class,
            prep_ms + scan_ms,
            compact_ms,
            || total.elapsed().as_secs_f64() * 1000.0,
        )?;
        (finished, events, referenced_runs)
    } else {
        let mut rows = Vec::with_capacity(class_fingerprints.len());
        for fingerprint in &class_fingerprints {
            let mut row = vec![0u32; aliases.len()];
            let mut offset = 0usize;
            if let Some(token) = root_token {
                row[token] = fingerprint[0];
                offset = 1;
            }
            for &profile in &fingerprint[offset..] {
                for run in profiles[profile as usize].iter() {
                    row[run.start as usize..run.end as usize].fill(run.signature);
                }
            }
            rows.push(row);
        }
        let materialize_ms = materialize_started.elapsed().as_secs_f64() * 1000.0;
        let finished = common::finish(
            input,
            aliases,
            &scanner.signatures,
            state_class,
            rows,
            prep_ms + scan_ms + materialize_ms,
            || total.elapsed().as_secs_f64() * 1000.0,
        )?;
        (finished, 0, 0)
    };
    let materialize_ms = materialize_started.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_projected] partition={} raw_states={} tokens={} trie_nodes={} configs={} signatures={} bucket_cache={} cache_hits={} pair_visits={} uniform_subtrees={} uniform_tokens={} profile_runs={} profiles={} state_classes={} token_classes={} run_sweep={} run_events={} referenced_runs={} vocab_cache_hit={} prep_ms={:.3} scan_ms={:.3} materialize_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            aliases.len(),
            trie.nodes.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            bucket_cache.len(),
            cache_hits,
            pair_visits,
            uniform_subtrees,
            uniform_tokens,
            profiles.iter().map(|profile| profile.len()).sum::<usize>(),
            profiles.len(),
            finished.state_classes,
            finished.token_classes,
            use_run_sweep,
            run_sweep_events,
            referenced_runs,
            finite_vocab_cache_hit,
            prep_ms,
            scan_ms,
            materialize_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}

fn projected_limit_exceeded(input: BuildInput<'_>, states: usize, limit: usize) -> bool {
    if states <= limit {
        return false;
    }
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_single_projected_abort] partition={} projected_states={} limit={} action=reject",
            input.partition_label, states, limit,
        );
    }
    true
}

fn projected_root_membership_estimate(input: BuildInput<'_>) -> usize {
    if let Some(state_map) = input.initial_state_map {
        state_map
            .representative_original_ids
            .iter()
            .filter(|&&state| state != u32::MAX)
            .map(|&state| {
                super::super::collect_active_terminal_signature(
                    input.tokenizer,
                    state,
                    input.active_terminals,
                )
                .len()
            })
            .sum()
    } else {
        (0..input.tokenizer.num_states())
            .map(|state| {
                super::super::collect_active_terminal_signature(
                    input.tokenizer,
                    state,
                    input.active_terminals,
                )
                .len()
            })
            .sum()
    }
}

fn residual_finite_switch_states(input: BuildInput<'_>) -> usize {
    if input.subset_parent_order.is_some() || input.vocab.len() < 50_000 {
        return usize::MAX;
    }
    std::env::var("GLRMASK_L1_RESIDUAL_FINITE_SWITCH_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20_000)
}

fn build_binary(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
    let finite_switch_states = residual_finite_switch_states(input);
    let use_finite_precheck = std::env::var("GLRMASK_L1_RESIDUAL_FINITE_PRECHECK")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true")
        })
        .unwrap_or(true);
    if use_finite_precheck && finite_switch_states != usize::MAX {
        let precheck_started = Instant::now();
        let memberships = projected_root_membership_estimate(input);
        let precheck_ms = precheck_started.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
            eprintln!(
                "[glrmask/profile][l1_residual_finite_precheck] partition={} memberships={} threshold={} selected={} precheck_ms={:.3}",
                input.partition_label,
                memberships,
                finite_switch_states,
                memberships > finite_switch_states,
                precheck_ms,
            );
        }
        if memberships > finite_switch_states {
            return build_finite_projected(input);
        }
    }
    let (aliases, tokens) = unique_vocab(input);
    if tokens.iter().all(|token| token.len() == 1) {
        return build_one_byte(input, &aliases, &tokens, total);
    }
    let mut relevant = [false; 256];
    for token in &tokens {
        for &byte in token.iter() {
            relevant[byte as usize] = true;
        }
    }
    let vocab_bytes = relevant
        .iter()
        .enumerate()
        .filter_map(|(byte, &used)| used.then_some(byte as u8))
        .collect::<Vec<_>>();
    let (bytes, input_byte_representative) = quotient_input_bytes(input, &vocab_bytes);

    let projected_started = Instant::now();
    let mut projected = Projected::new(input);
    // `initial_state_map` is already an exact certified quotient for this L1
    // branch.  Preserve a raw-state-indexed root table for O(1) transition
    // lookup, but construct residual roots only for quotient representatives.
    // Every transition that lands on a raw state therefore enters the same
    // projected residual as its certified representative instead of rebuilding
    // duplicate `(terminal, raw-state)` subautomata.
    let mut roots = if let Some(state_map) = input.initial_state_map {
        debug_assert_eq!(
            state_map.original_to_internal.len(),
            input.tokenizer.num_states() as usize
        );
        let representative_rows = state_map
            .representative_original_ids
            .iter()
            .map(|&raw| {
                assert_ne!(raw, u32::MAX, "L1 state quotient has an unmapped representative");
                projected.root_row(raw)
            })
            .collect::<Vec<_>>();
        state_map
            .original_to_internal
            .iter()
            .enumerate()
            .map(|(raw, &class)| {
                if class == u32::MAX {
                    projected.root_row(raw as u32)
                } else {
                    representative_rows[class as usize].clone()
                }
            })
            .collect::<Vec<_>>()
    } else {
        (0..input.tokenizer.num_states())
            .map(|raw| projected.root_row(raw))
            .collect::<Vec<_>>()
    };
    if projected.configs.len() > finite_switch_states {
        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
            eprintln!(
                "[glrmask/profile][l1_residual_finite_switch] partition={} projected_states={} threshold={} phase=roots action=finite",
                input.partition_label,
                projected.configs.len(),
                finite_switch_states,
            );
        }
        return build_finite_projected(input);
    }
    let limit = std::env::var("GLRMASK_L1_SINGLE_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000_000usize);
    assert!(
        !projected_limit_exceeded(input, projected.configs.len(), limit),
        "projected L1 exceeded GLRMASK_L1_SINGLE_MAX_STATES; raise the projected-state limit for this diagnostic guard"
    );
    let mut queue = VecDeque::from_iter(0..projected.configs.len() as u32);
    let mut expanded = 0usize;
    while let Some(state) = queue.pop_front() {
        let mut row = Vec::new();
        for (symbol, &byte) in bytes.iter().enumerate() {
            let before = projected.configs.len();
            let target = projected.step(state, byte, &roots);
            if target != DEAD {
                row.push((symbol as u8, target));
            }
            if projected.configs.len() > before {
                queue.extend(before as u32..projected.configs.len() as u32);
                if projected.configs.len() > finite_switch_states {
                    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                        eprintln!(
                            "[glrmask/profile][l1_residual_finite_switch] partition={} projected_states={} threshold={} action=finite",
                            input.partition_label,
                            projected.configs.len(),
                            finite_switch_states,
                        );
                    }
                    return build_finite_projected(input);
                }
                if projected_limit_exceeded(input, projected.configs.len(), limit) {
                    return build_finite_projected(input);
                }
            }
        }
        projected.transitions[state as usize] = row;
        expanded += 1;
    }
    let projected_ms = projected_started.elapsed().as_secs_f64() * 1000.0;

    let minimize_started = Instant::now();
    let groups = projected
        .configs
        .iter()
        .map(|(group, _)| *group)
        .collect::<Vec<_>>();
    let use_grouped_minimize = std::env::var("GLRMASK_L1_PROJECTED_GROUPED_MINIMIZE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let (mut minimized, grouped_minimize) = if use_grouped_minimize {
        let (minimized, stats) = minimize_grouped(&projected.transitions, &groups, &bytes);
        (minimized, Some(stats))
    } else {
        (minimize(&projected.transitions, &bytes), None)
    };
    for &byte in &vocab_bytes {
        minimized.byte_class[byte as usize] =
            minimized.byte_class[input_byte_representative[byte as usize] as usize];
    }
    let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
    for row in &mut roots {
        for state in row {
            if *state != DEAD {
                *state = minimized.classes[*state as usize];
            }
        }
    }

    let traverse_started = Instant::now();
    let root_vectors_started = Instant::now();
    // Full residual-root vectors exactly preclassify raw lexer states.
    let mut root_vector_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut root_vectors = Vec::<Vec<u32>>::new();
    let mut raw_root_class = vec![0u32; roots.len()];
    for (raw, root_row) in roots.iter().enumerate() {
        let next = root_vectors.len() as u32;
        raw_root_class[raw] = *root_vector_ids.entry(root_row.clone()).or_insert_with(|| {
            root_vectors.push(root_row.clone());
            next
        });
    }

    let mut used_classes = roots
        .iter()
        .flatten()
        .copied()
        .filter(|&state| state != DEAD)
        .collect::<Vec<_>>();
    used_classes.sort_unstable();
    used_classes.dedup();
    let mut used_index = vec![usize::MAX; minimized.state_count];
    for (index, &state) in used_classes.iter().enumerate() {
        used_index[state as usize] = index;
    }
    let root_vectors_ms = root_vectors_started.elapsed().as_secs_f64() * 1000.0;

    let reverse_started = Instant::now();
    let mut reverse = ReverseSubsets::new(
        &minimized.columns,
        &minimized.byte_class,
        minimized.state_count,
    );
    let mut final_subset_to_class = FxHashMap::<u32, u32>::default();
    let mut class_subsets = Vec::<u32>::new();
    let mut token_class = vec![0u32; aliases.len()];
    let mut token_bytes = 0usize;
    for (token_index, token) in tokens.iter().enumerate() {
        token_bytes += token.len();
        let subset = reverse.token(token);
        let next = class_subsets.len() as u32;
        token_class[token_index] = *final_subset_to_class.entry(subset).or_insert_with(|| {
            class_subsets.push(subset);
            next
        });
    }
    let reverse_ms = reverse_started.elapsed().as_secs_f64() * 1000.0;
    let incidence_started = Instant::now();
    let mut live_token_classes = vec![Vec::<u32>::new(); used_classes.len()];
    for (token_class, &subset) in class_subsets.iter().enumerate() {
        let set = &reverse.sets[subset as usize];
        for (used, &state) in used_classes.iter().enumerate() {
            if ReverseSubsets::contains(set, state) {
                live_token_classes[used].push(token_class as u32);
            }
        }
    }
    let incidence_ms = incidence_started.elapsed().as_secs_f64() * 1000.0;

    let signature_started = Instant::now();
    // A root row is exactly the sparse relation
    //     (terminal identity, residual-token class)
    // induced by its residual states.  Canonicalize that relation *before*
    // constructing dense token rows or interned terminal-set signatures.  This
    // is an exact representation: two roots have equal L1 rows iff these pair
    // lists are equal.  In the p90 p2 shapes the dense row can be 1k+ token
    // classes while the sparse relation is typically only tens of pairs.
    let mut sparse_row_ids = FxHashMap::<Vec<(u32, u32)>, u32>::default();
    let mut sparse_rows = Vec::<Vec<(u32, u32)>>::new();
    let mut root_to_state_class = vec![0u32; root_vectors.len()];
    let mut sparse_pair_visits = 0usize;
    for (root_class, root_row) in root_vectors.iter().enumerate() {
        let mut sparse = Vec::<(u32, u32)>::new();
        for (terminal_index, &group) in projected.terminal_groups.iter().enumerate() {
            let state = root_row[group];
            if state == DEAD {
                continue;
            }
            for &token in &live_token_classes[used_index[state as usize]] {
                // terminal_index-major, then token-major, is canonical because
                // both source lists are stable and strictly ordered.
                sparse.push((projected.terminals[terminal_index], token));
            }
        }
        sparse_pair_visits += sparse.len();
        let next = sparse_rows.len() as u32;
        root_to_state_class[root_class] = match sparse_row_ids.entry(sparse) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                sparse_rows.push(entry.key().clone());
                entry.insert(next);
                next
            }
        };
    }

    // The sparse relation is already the final semantic row. No dense
    // token-class matrix or terminal-set signature interning is necessary.
    let signature_updates = 0usize;
    let state_class = raw_root_class
        .into_iter()
        .map(|class| root_to_state_class[class as usize])
        .collect::<Vec<_>>();
    let sparse_row_count = sparse_rows.len();
    let signature_count = 0usize;
    let signature_ms = signature_started.elapsed().as_secs_f64() * 1000.0;
    let traverse_ms = traverse_started.elapsed().as_secs_f64() * 1000.0;
    let finished = common::finish_sparse_terminal_rows(
        input,
        &aliases,
        state_class,
        sparse_rows,
        token_class,
        projected_ms + minimize_ms + traverse_ms,
        0.0,
        || total.elapsed().as_secs_f64() * 1000.0,
    )?;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        let live_edges = projected.transitions.iter().map(Vec::len).sum::<usize>();
        let max_live_edges = projected.transitions.iter().map(Vec::len).max().unwrap_or(0);
        let singleton_configs = projected
            .configs
            .iter()
            .filter(|(_, states)| states.len() == 1)
            .count();
        let total_config_states = projected
            .configs
            .iter()
            .map(|(_, states)| states.len())
            .sum::<usize>();
        let max_config_states = projected
            .configs
            .iter()
            .map(|(_, states)| states.len())
            .max()
            .unwrap_or(0);
        eprintln!(
            "[glrmask/profile][l1_single] partition={} raw_states={} terminals={} terminal_groups={} vocab_bytes={} input_byte_classes={} minimized_bytes={} minimize_rounds={} grouped_local_states={} grouped_local_rounds={} grouped_local_ms={:.3} grouped_global_ms={:.3} dag_groups={} dag_states={} hopcroft_groups={} hopcroft_states={} projected_states={} live_edges={} max_live_edges={} singleton_configs={} total_config_states={} max_config_states={} expanded={} minimized_states={} root_closure_states={} root_memberships={} root_classes={} root_vectors={} residual_token_classes={} sparse_rows={} sparse_pairs={} signature_updates={} reverse_states={} reverse_transitions={} reverse_target_visits={} reverse_predecessor_visits={} token_bytes={} signatures={} state_classes={} token_classes={} project_ms={:.3} minimize_ms={:.3} root_vectors_ms={:.3} reverse_ms={:.3} incidence_ms={:.3} signature_ms={:.3} traverse_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            projected.terminals.len(),
            projected.active_by_group.len(),
            vocab_bytes.len(),
            bytes.len(),
            minimized.columns.len(),
            minimized.rounds,
            grouped_minimize.as_ref().map_or(0, |stats| stats.local_states),
            grouped_minimize.as_ref().map_or(0, |stats| stats.local_rounds),
            grouped_minimize.as_ref().map_or(0.0, |stats| stats.local_ms),
            grouped_minimize.as_ref().map_or(0.0, |stats| stats.global_ms),
            grouped_minimize.as_ref().map_or(0, |stats| stats.dag_groups),
            grouped_minimize.as_ref().map_or(0, |stats| stats.dag_states),
            grouped_minimize.as_ref().map_or(0, |stats| stats.hopcroft_groups),
            grouped_minimize.as_ref().map_or(0, |stats| stats.hopcroft_states),
            projected.configs.len(),
            live_edges,
            max_live_edges,
            singleton_configs,
            total_config_states,
            max_config_states,
            expanded,
            minimized.state_count,
            projected.root_closure_states,
            projected.root_memberships,
            used_classes.len(),
            root_vectors.len(),
            class_subsets.len(),
            sparse_row_count,
            sparse_pair_visits,
            signature_updates,
            reverse.sets.len(),
            reverse.computed_transitions,
            reverse.target_visits,
            reverse.predecessor_visits,
            token_bytes,
            signature_count,
            finished.state_classes,
            finished.token_classes,
            projected_ms,
            minimize_ms,
            root_vectors_ms,
            reverse_ms,
            incidence_ms,
            signature_ms,
            traverse_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}

#[cfg(test)]
mod finite_run_sweep_tests {
    use super::*;

    #[test]
    fn run_sweep_matches_dense_token_vector_equivalence() {
        const ROWS: usize = 17;
        const TOKENS: usize = 137;
        let mut dense = vec![vec![0u32; TOKENS]; ROWS];
        for (row, values) in dense.iter_mut().enumerate() {
            for (token, value) in values.iter_mut().enumerate() {
                // Piecewise structure with many adjacent boundaries and repeated
                // vectors across non-adjacent token intervals.
                let block = token / (2 + row % 7);
                *value = if (block + row) % 5 == 0 {
                    0
                } else {
                    1 + ((block * 3 + row * 11) % 9) as u32
                };
            }
        }

        let mut profiles = vec![Arc::<[ProfileRun]>::from([])];
        let mut fingerprints = Vec::with_capacity(ROWS);
        for values in &dense {
            let mut runs = Vec::new();
            let mut start = 0usize;
            while start < TOKENS {
                let signature = values[start];
                let mut end = start + 1;
                while end < TOKENS && values[end] == signature {
                    end += 1;
                }
                push_profile_run(&mut runs, start as u32, end as u32, signature);
                start = end;
            }
            let profile = profiles.len() as u32;
            profiles.push(Arc::from(runs));
            fingerprints.push(vec![profile]);
        }

        let (classes, reps, compact_rows, events, referenced_runs) =
            finite_compact_runs(None, &fingerprints, &profiles, TOKENS);
        assert!(events > 0);
        assert!(referenced_runs > 0);
        assert_eq!(compact_rows.len(), ROWS);
        assert_eq!(compact_rows[0].len(), reps.len());

        for left in 0..TOKENS {
            for right in 0..TOKENS {
                let dense_equal = dense.iter().all(|row| row[left] == row[right]);
                assert_eq!(
                    classes[left] == classes[right],
                    dense_equal,
                    "token equivalence mismatch for {left} and {right}",
                );
            }
        }
        for (class, &representative) in reps.iter().enumerate() {
            for row in 0..ROWS {
                assert_eq!(compact_rows[row][class], dense[row][representative]);
            }
        }
    }
}
