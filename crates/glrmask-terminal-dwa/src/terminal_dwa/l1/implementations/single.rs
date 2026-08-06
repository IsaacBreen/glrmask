//! Single-group prefix DFA experiment.
//!
//! A state represents one terminal's residual language from one lexer
//! configuration. Every non-dead state is accepting: surviving to a model-token
//! boundary is exactly `finalizer || possible_future`. All `(raw state, terminal)`
//! residuals are roots; minimization therefore never prunes by reachability from
//! one distinguished start state.

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::terminal_dwa::l1::implementations::support::{DEAD, FlatNode, flatten, vocab};

const UNKNOWN: u32 = u32::MAX - 1;

type ByteMask = [u64; 4];

fn set(mask: &mut ByteMask, byte: u8) {
    mask[byte as usize / 64] |= 1u64 << (byte as usize % 64);
}

fn subset(left: &ByteMask, right: &ByteMask) -> bool {
    left.iter().zip(right).all(|(&left, &right)| left & !right == 0)
}

struct Projected<'a> {
    input: BuildInput<'a>,
    active: Vec<Vec<u32>>,
    terminals: Vec<u32>,
    configs: Vec<(u32, Box<[u32]>)>,
    ids: FxHashMap<(u32, Vec<u32>), u32>,
    transitions: Vec<[u32; 256]>,
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
            .collect();
        let terminals = input
            .active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
            .collect();
        Self {
            input,
            active,
            terminals,
            configs: Vec::new(),
            ids: FxHashMap::default(),
            transitions: Vec::new(),
        }
    }

    fn state_active(&self, state: u32, terminal: u32) -> bool {
        self.active[state as usize].binary_search(&terminal).is_ok()
    }

    fn intern(&mut self, terminal: u32, mut states: Vec<u32>) -> u32 {
        states.retain(|&state| self.state_active(state, terminal));
        states.sort_unstable();
        states.dedup();
        if states.is_empty() {
            return DEAD;
        }
        let key = (terminal, states);
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.ids.insert(key.clone(), id);
        self.configs.push((terminal, key.1.into_boxed_slice()));
        self.transitions.push([UNKNOWN; 256]);
        id
    }

    fn root(&mut self, raw: u32, terminal: u32) -> u32 {
        self.intern(
            terminal,
            self.input
                .tokenizer
                .execute_from_state_end_only(&[], raw)
                .to_vec(),
        )
    }

    fn step(&mut self, state: u32, byte: u8) -> u32 {
        let cached = self.transitions[state as usize][byte as usize];
        if cached != UNKNOWN {
            return cached;
        }
        let (terminal, config) = &self.configs[state as usize];
        let terminal = *terminal;
        let target = self.intern(
            terminal,
            self.input.tokenizer.step_all(config, byte).to_vec(),
        );
        self.transitions[state as usize][byte as usize] = target;
        target
    }
}

fn minimize(transitions: &[[u32; 256]], bytes: &[u8]) -> (Vec<u32>, Vec<[u32; 256]>) {
    let mut classes = vec![0u32; transitions.len()];
    loop {
        let mut ids = FxHashMap::<Vec<u32>, u32>::default();
        let next = transitions
            .iter()
            .map(|row| {
                let key = bytes
                    .iter()
                    .map(|&byte| {
                        let target = row[byte as usize];
                        if target == DEAD { 0 } else { classes[target as usize] + 1 }
                    })
                    .collect::<Vec<_>>();
                let next = ids.len() as u32;
                *ids.entry(key).or_insert(next)
            })
            .collect::<Vec<_>>();
        if next == classes {
            break;
        }
        classes = next;
    }
    let count = classes.iter().copied().max().map_or(0, |class| class + 1) as usize;
    let mut representatives = vec![usize::MAX; count];
    for (state, &class) in classes.iter().enumerate() {
        representatives[class as usize] = representatives[class as usize].min(state);
    }
    let mut compact = vec![[DEAD; 256]; count];
    for (class, &state) in representatives.iter().enumerate() {
        for &byte in bytes {
            let target = transitions[state][byte as usize];
            if target != DEAD {
                compact[class][byte as usize] = classes[target as usize];
            }
        }
    }
    (classes, compact)
}

#[derive(Clone, Copy, Default)]
struct NodeInfo {
    bytes: ByteMask,
    first_token: usize,
    token_count: usize,
}

fn node_info(node: usize, trie: &[FlatNode], info: &mut [NodeInfo]) -> NodeInfo {
    let mut result = NodeInfo {
        bytes: [0; 4],
        first_token: trie[node].token.unwrap_or(usize::MAX),
        token_count: usize::from(trie[node].token.is_some()),
    };
    for (edge, child) in &trie[node].edges {
        for &byte in edge.iter() {
            set(&mut result.bytes, byte);
        }
        let child_info = node_info(*child, trie, info);
        for part in 0..4 {
            result.bytes[part] |= child_info.bytes[part];
        }
        result.first_token = result.first_token.min(child_info.first_token);
        result.token_count += child_info.token_count;
    }
    info[node] = result;
    result
}

fn live_tokens(
    node: usize,
    state: u32,
    trie: &[FlatNode],
    info: &[NodeInfo],
    transitions: &[[u32; 256]],
    loops: &[ByteMask],
    out: &mut Vec<usize>,
    subtree_skips: &mut usize,
) {
    if info[node].token_count != 0 && subset(&info[node].bytes, &loops[state as usize]) {
        let first = info[node].first_token;
        out.extend(first..first + info[node].token_count);
        *subtree_skips += 1;
        return;
    }
    if let Some(token) = trie[node].token {
        out.push(token);
    }
    for (edge, child) in &trie[node].edges {
        let mut target = state;
        for &byte in edge.iter() {
            target = transitions[target as usize][byte as usize];
            if target == DEAD {
                break;
            }
        }
        if target != DEAD {
            live_tokens(*child, target, trie, info, transitions, loops, out, subtree_skips);
        }
    }
}

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
    let (aliases, tree) = vocab(input);
    let trie = flatten(&tree);
    let mut relevant = [false; 256];
    for (_, token) in input.vocab.iter() {
        for &byte in token {
            relevant[byte as usize] = true;
        }
    }
    let bytes = relevant
        .iter()
        .enumerate()
        .filter_map(|(byte, &used)| used.then_some(byte as u8))
        .collect::<Vec<_>>();

    let projected_started = Instant::now();
    let mut projected = Projected::new(input);
    let mut roots = vec![vec![DEAD; projected.terminals.len()]; input.tokenizer.num_states() as usize];
    for raw in 0..input.tokenizer.num_states() {
        for index in 0..projected.terminals.len() {
            let terminal = projected.terminals[index];
            roots[raw as usize][index] = projected.root(raw, terminal);
        }
    }
    let mut queue = VecDeque::from_iter(0..projected.configs.len() as u32);
    let mut expanded = 0usize;
    let limit = std::env::var("GLRMASK_L1_SINGLE_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000_000usize);
    while let Some(state) = queue.pop_front() {
        for &byte in &bytes {
            let before = projected.configs.len();
            projected.step(state, byte);
            if projected.configs.len() > before {
                queue.extend(before as u32..projected.configs.len() as u32);
                assert!(projected.configs.len() <= limit, "single-group DFA exceeded {limit} states");
            }
        }
        expanded += 1;
    }
    for row in &mut projected.transitions {
        for &byte in &bytes {
            if row[byte as usize] == UNKNOWN {
                row[byte as usize] = DEAD;
            }
        }
    }
    let projected_ms = projected_started.elapsed().as_secs_f64() * 1000.0;

    let minimize_started = Instant::now();
    let (classes, transitions) = minimize(&projected.transitions, &bytes);
    let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
    let mut loops = vec![[0u64; 4]; transitions.len()];
    for (state, row) in transitions.iter().enumerate() {
        for &byte in &bytes {
            if row[byte as usize] == state as u32 {
                set(&mut loops[state], byte);
            }
        }
    }
    for row in &mut roots {
        for state in row {
            if *state != DEAD {
                *state = classes[*state as usize];
            }
        }
    }

    let traverse_started = Instant::now();
    let mut info = vec![NodeInfo::default(); trie.len()];
    node_info(0, &trie, &mut info);
    let mut used_classes = roots
        .iter()
        .flatten()
        .copied()
        .filter(|&state| state != DEAD)
        .collect::<Vec<_>>();
    used_classes.sort_unstable();
    used_classes.dedup();
    let mut token_sets = FxHashMap::<u32, Vec<usize>>::default();
    let mut subtree_skips = 0usize;
    for state in used_classes {
        let mut tokens = Vec::new();
        live_tokens(0, state, &trie, &info, &transitions, &loops, &mut tokens, &mut subtree_skips);
        token_sets.insert(state, tokens);
    }
    // Full residual-root vectors are an exact preclassification of raw lexer
    // states. Build one expensive vocabulary row per distinct vector, not per
    // raw state, then retain the later vocabulary-specific row merge.
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

    let mut signature_ids = FxHashMap::<(u32, u32), u32>::default();
    let mut signatures = vec![Vec::<u32>::new()];
    let mut row_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut rows = Vec::<Vec<u32>>::new();
    let mut root_to_state_class = vec![0u32; root_vectors.len()];
    let mut signature_updates = 0usize;
    for (root_class, root_row) in root_vectors.iter().enumerate() {
        let mut row = vec![0u32; aliases.len()];
        for (&terminal, &state) in projected.terminals.iter().zip(root_row) {
            if state == DEAD {
                continue;
            }
            for &token in &token_sets[&state] {
                let previous = row[token];
                let next = signatures.len() as u32;
                row[token] = *signature_ids.entry((previous, terminal)).or_insert_with(|| {
                    let mut signature = signatures[previous as usize].clone();
                    signature.push(terminal);
                    signatures.push(signature);
                    next
                });
                signature_updates += 1;
            }
        }
        let next = rows.len() as u32;
        root_to_state_class[root_class] = *row_ids.entry(row.clone()).or_insert_with(|| {
            rows.push(row);
            next
        });
    }
    let state_class = raw_root_class
        .into_iter()
        .map(|class| root_to_state_class[class as usize])
        .collect::<Vec<_>>();
    let traverse_ms = traverse_started.elapsed().as_secs_f64() * 1000.0;
    let finished = common::finish(
        input,
        &aliases,
        &signatures,
        state_class,
        rows,
        projected_ms + minimize_ms + traverse_ms,
        || total.elapsed().as_secs_f64() * 1000.0,
    )?;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_single] partition={} raw_states={} terminals={} bytes={} projected_states={} expanded={} minimized_states={} root_classes={} root_vectors={} signature_updates={} subtree_skips={} signatures={} state_classes={} token_classes={} project_ms={:.3} minimize_ms={:.3} traverse_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            projected.terminals.len(),
            bytes.len(),
            projected.configs.len(),
            expanded,
            transitions.len(),
            token_sets.len(),
            root_vectors.len(),
            signature_updates,
            subtree_skips,
            signatures.len(),
            finished.state_classes,
            finished.token_classes,
            projected_ms,
            minimize_ms,
            traverse_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}
