//! Default L1 compiler with an exact projected-residual fast path.
//!
//! A state represents one terminal's residual language from one lexer
//! configuration. Every non-dead state is accepting: surviving to a model-token
//! boundary is exactly `finalizer || possible_future`. All `(raw state, terminal)`
//! residuals are roots; minimization therefore never prunes by reachability from
//! one distinguished start state.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::terminal_dwa::l1::implementations::support::{DEAD, Scanner};

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
    let mut incoming_counts = vec![0u32; live];
    let mut initial_incoming = vec![0u32; alphabet.len()];
    for row in transitions {
        for &(symbol, target) in row {
            incoming_counts[target as usize] += 1;
            initial_incoming[symbol as usize] += 1;
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

    // Hopcroft over all residual roots. Starting with the live block is enough:
    // for any byte, the rejecting-sink predecessor set is the complement of the
    // live predecessor set over represented sources and therefore induces the
    // same split.
    let mut blocks = vec![(0..live as u32).collect::<Vec<_>>()];
    let mut class = vec![0u32; live];
    let mut block_incoming = vec![initial_incoming.into_boxed_slice()];
    let mut queue = VecDeque::<(u32, usize)>::new();
    let mut queued = vec![vec![false; alphabet.len()]; blocks.len()];
    for symbol in 0..alphabet.len() {
        if block_incoming[0][symbol] != 0 {
            queue.push_back((0, symbol));
            queued[0][symbol] = true;
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

fn ordered_vocab(input: BuildInput<'_>) -> Arc<super::super::L1IdentityVocabOrder> {
    input
        .subset_parent_order
        .map(|parent| super::super::derive_l1_identity_vocab_order_from_parent(parent, input.vocab))
        .unwrap_or_else(|| super::super::prepared_l1_identity_vocab_order(input.vocab))
}

fn unique_vocab(input: BuildInput<'_>) -> (Vec<Vec<u32>>, Vec<Arc<[u8]>>) {
    let order = ordered_vocab(input);
    let mut tokens = Vec::<Arc<[u8]>>::new();
    let mut aliases = Vec::<Vec<u32>>::new();
    for (id, bytes) in order.token_entries_sorted.iter() {
        if tokens.last().is_some_and(|token| token.as_ref() == bytes.as_ref()) {
            aliases.last_mut().unwrap().push(*id);
        } else {
            tokens.push(Arc::clone(bytes));
            aliases.push(vec![*id]);
        }
    }
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

fn projected_cell_budget() -> usize {
    std::env::var("GLRMASK_L1_SINGLE_PROJECTED_CELL_BUDGET")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(PROJECTED_CELL_BUDGET)
}

fn projected_shape(input: BuildInput<'_>) -> (usize, usize) {
    let order = ordered_vocab(input);
    let mut relevant = [false; 256];
    for (_, token) in order.token_entries_sorted.iter() {
        for &byte in token.iter() {
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

/// One exact L1 compiler with two build-time representations of the same
/// finite-vocabulary relation. Sparse terminal products use the projected
/// residual DFA; denser products use the established exact quotient. The
/// choice depends only on lexer/vocabulary shape, never corpus identity, and
/// neither representation adds runtime mask-generation work.
pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    let generic_epsilon = input.tokenizer.has_epsilon_transitions()
        && !input.tokenizer.has_scalar_deterministic_dispatch();
    if !generic_epsilon {
        super::production::build(input)
    } else {
        let (memberships, vocab_bytes) = projected_shape(input);
        let projected_cells = memberships.saturating_mul(vocab_bytes);
        let projected_cell_budget = projected_cell_budget();
        let use_established = projected_cells > projected_cell_budget;
        if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
            eprintln!(
                "[glrmask/profile][l1_single_plan] partition={} memberships={} vocab_bytes={} projected_cell_estimate={} budget={} representation={}",
                input.partition_label,
                memberships,
                vocab_bytes,
                projected_cells,
                projected_cell_budget,
                if use_established { "established" } else { "projected" },
            );
        }
        if use_established {
            super::production::build(input)
        } else {
            build_binary(input)
        }
    }
}

fn projected_limit_exceeded(input: BuildInput<'_>, states: usize, limit: usize) -> bool {
    if states <= limit {
        return false;
    }
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_single_projected_abort] partition={} projected_states={} limit={} fallback=established",
            input.partition_label, states, limit,
        );
    }
    true
}

fn build_binary(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
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
    let mut roots = Vec::with_capacity(input.tokenizer.num_states() as usize);
    for raw in 0..input.tokenizer.num_states() {
        roots.push(projected.root_row(raw));
    }
    let limit = std::env::var("GLRMASK_L1_SINGLE_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000_000usize);
    if projected_limit_exceeded(input, projected.configs.len(), limit) {
        return super::production::build(input);
    }
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
                if projected_limit_exceeded(input, projected.configs.len(), limit) {
                    return super::production::build(input);
                }
            }
        }
        projected.transitions[state as usize] = row;
        expanded += 1;
    }
    let projected_ms = projected_started.elapsed().as_secs_f64() * 1000.0;

    let minimize_started = Instant::now();
    let mut minimized = minimize(&projected.transitions, &bytes);
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
    let mut signatures = vec![Vec::<u32>::new()];
    let mut signature_extensions = vec![Vec::<u32>::new(); projected.terminals.len()];
    let mut row_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut rows = Vec::<Vec<u32>>::new();
    let mut root_to_state_class = vec![0u32; root_vectors.len()];
    let mut signature_updates = 0usize;
    for (root_class, root_row) in root_vectors.iter().enumerate() {
        let mut row = vec![0u32; class_subsets.len()];
        for (terminal_index, (&terminal, &group)) in projected
            .terminals
            .iter()
            .zip(&projected.terminal_groups)
            .enumerate()
        {
            let state = root_row[group];
            if state == DEAD {
                continue;
            }
            for &token in &live_token_classes[used_index[state as usize]] {
                let previous = row[token as usize];
                let extensions = &mut signature_extensions[terminal_index];
                if extensions.len() <= previous as usize {
                    extensions.resize(previous as usize + 1, UNKNOWN);
                }
                let cached = extensions[previous as usize];
                row[token as usize] = if cached != UNKNOWN {
                    cached
                } else {
                    let next = signatures.len() as u32;
                    let mut signature = signatures[previous as usize].clone();
                    signature.push(terminal);
                    signatures.push(signature);
                    extensions[previous as usize] = next;
                    next
                };
                signature_updates += 1;
            }
        }
        let next = rows.len() as u32;
        root_to_state_class[root_class] = match row_ids.entry(row) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                rows.push(entry.key().clone());
                entry.insert(next);
                next
            }
        };
    }
    let state_class = raw_root_class
        .into_iter()
        .map(|class| root_to_state_class[class as usize])
        .collect::<Vec<_>>();
    let signature_ms = signature_started.elapsed().as_secs_f64() * 1000.0;
    let traverse_ms = traverse_started.elapsed().as_secs_f64() * 1000.0;
    let finished = common::finish_compacted(
        input,
        &aliases,
        &signatures,
        state_class,
        rows,
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
            "[glrmask/profile][l1_single] partition={} raw_states={} terminals={} terminal_groups={} vocab_bytes={} input_byte_classes={} minimized_bytes={} minimize_rounds={} projected_states={} live_edges={} max_live_edges={} singleton_configs={} total_config_states={} max_config_states={} expanded={} minimized_states={} root_closure_states={} root_memberships={} root_classes={} root_vectors={} residual_token_classes={} signature_updates={} reverse_states={} reverse_transitions={} reverse_target_visits={} reverse_predecessor_visits={} token_bytes={} signatures={} state_classes={} token_classes={} project_ms={:.3} minimize_ms={:.3} root_vectors_ms={:.3} reverse_ms={:.3} incidence_ms={:.3} signature_ms={:.3} traverse_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            projected.terminals.len(),
            projected.active_by_group.len(),
            vocab_bytes.len(),
            bytes.len(),
            minimized.columns.len(),
            minimized.rounds,
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
            signature_updates,
            reverse.sets.len(),
            reverse.computed_transitions,
            reverse.target_visits,
            reverse.predecessor_visits,
            token_bytes,
            signatures.len(),
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
