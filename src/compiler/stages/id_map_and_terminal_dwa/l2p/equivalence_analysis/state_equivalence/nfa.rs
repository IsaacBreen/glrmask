use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::super::compat::{FlatDfa, FlatDfaState, TokenizerView};

#[derive(Debug, Clone, Copy)]
pub(crate) enum RefinementDepth {
    Stable,
    Bounded(usize),
}

pub(crate) struct BoundedAnalysisView {
    pub(crate) tokenizer_view: TokenizerView,
    raw_start_to_view: Vec<u32>,
}

pub(crate) struct RelevantPowersetView {
    pub(crate) dfa: FlatDfa,
    pub(crate) raw_start_to_view: Arc<[u32]>,
    pub(crate) configurations: Arc<[Box<[u32]>]>,
}

impl BoundedAnalysisView {
    pub(crate) fn view_state_for_raw_start(&self, raw_state: usize) -> usize {
        let state = self.raw_start_to_view[raw_state];
        assert_ne!(state, u32::MAX, "raw state was not seeded into bounded NFA analysis");
        state as usize
    }
}

pub(crate) fn build_relevant_powerset_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
) -> RelevantPowersetView {
    let raw_state_count = tokenizer.num_states() as usize;
    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let mut raw_start_to_view = vec![u32::MAX; raw_state_count];
    let mut worklist = VecDeque::<u32>::new();
    let mut queued = Vec::<bool>::new();

    for raw_state in 0..raw_state_count {
        let closure = tokenizer
            .execute_from_state_end_only(&[], raw_state as u32)
            .to_vec();
        let state = intern_config(closure, &mut config_ids, &mut configs);
        raw_start_to_view[raw_state] = state;
        if queued.len() < configs.len() {
            queued.resize(configs.len(), false);
        }
        if !queued[state as usize] {
            queued[state as usize] = true;
            worklist.push_back(state);
        }
    }

    let start_state = raw_start_to_view[tokenizer.initial_state_id() as usize] as usize;
    let mut transitions = vec![u32::MAX; configs.len() * 256];
    while let Some(state) = worklist.pop_front() {
        let config = configs[state as usize].clone();
        for (byte, &relevant) in relevant_bytes.iter().enumerate() {
            if !relevant {
                continue;
            }
            let targets = tokenizer.step_all(&config, byte as u8);
            if targets.is_empty() {
                continue;
            }
            let target = intern_config(targets.to_vec(), &mut config_ids, &mut configs);
            if transitions.len() < configs.len() * 256 {
                transitions.resize(configs.len() * 256, u32::MAX);
                queued.resize(configs.len(), false);
            }
            transitions[state as usize * 256 + byte] = target;
            if !queued[target as usize] {
                queued[target as usize] = true;
                worklist.push_back(target);
            }
        }
    }

    let states = configs
        .iter()
        .map(|config| FlatDfaState {
            finalizers: filtered_config_groups(tokenizer, config, active_groups, true),
            possible_future_group_ids: filtered_config_groups(
                tokenizer,
                config,
                active_groups,
                false,
            ),
        })
        .collect();
    RelevantPowersetView {
        dfa: FlatDfa {
            states,
            start_state,
            transitions: Arc::from(transitions),
        },
        raw_start_to_view: Arc::from(raw_start_to_view),
        configurations: Arc::from(configs),
    }
}

#[derive(Default)]
struct ByteTrieNode {
    children: Vec<(u8, usize)>,
}

fn build_byte_trie<'a>(sequences: impl IntoIterator<Item = &'a [u8]>) -> Vec<ByteTrieNode> {
    let mut nodes = vec![ByteTrieNode::default()];
    for sequence in sequences {
        let mut node = 0usize;
        for &byte in sequence {
            let child = if let Some((_, child)) = nodes[node]
                .children
                .iter()
                .find(|(candidate, _)| *candidate == byte)
            {
                *child
            } else {
                let child = nodes.len();
                nodes.push(ByteTrieNode::default());
                nodes[node].children.push((byte, child));
                child
            };
            node = child;
        }
    }
    nodes
}

fn filtered_config_groups(
    tokenizer: &Tokenizer,
    config: &[u32],
    active_groups: Option<&[bool]>,
    finalizers: bool,
) -> Vec<usize> {
    let mut groups = Vec::<usize>::new();
    for &state in config {
        if finalizers {
            groups.extend(tokenizer.matched_terminals_iter(state).map(|group| group as usize));
        } else {
            groups.extend(
                tokenizer
                    .possible_future_terminals_iter(state)
                    .map(|group| group as usize),
            );
        }
    }
    groups.retain(|&group| {
        active_groups.is_none_or(|active| active.get(group).copied().unwrap_or(false))
    });
    groups.sort_unstable();
    groups.dedup();
    groups
}

fn ensure_config_transition(
    tokenizer: &Tokenizer,
    state: u32,
    byte: u8,
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
) -> u32 {
    let slot = state as usize * 256 + byte as usize;
    if known_transitions[slot] != 0 {
        return transitions[slot];
    }
    known_transitions[slot] = 1;
    let targets = tokenizer.step_all(&configs[state as usize], byte);
    if targets.is_empty() {
        return u32::MAX;
    }
    let target = intern_config(targets.to_vec(), config_ids, configs);
    if transitions.len() < configs.len() * 256 {
        transitions.resize(configs.len() * 256, u32::MAX);
        known_transitions.resize(configs.len() * 256, 0);
    }
    transitions[slot] = target;
    target
}

fn expand_trie_from_config(
    tokenizer: &Tokenizer,
    start_state: u32,
    trie: &[ByteTrieNode],
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    visited: &mut rustc_hash::FxHashSet<(u32, usize)>,
) {
    let mut stack = vec![(start_state, 0usize)];
    while let Some((state, node)) = stack.pop() {
        if !visited.insert((state, node)) {
            continue;
        }
        for &(byte, child) in &trie[node].children {
            let target = ensure_config_transition(
                tokenizer,
                state,
                byte,
                configs,
                config_ids,
                transitions,
                known_transitions,
            );
            if target != u32::MAX {
                stack.push((target, child));
            }
        }
    }
}

pub(crate) fn build_bounded_analysis_view(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    let raw_state_count = tokenizer.num_states() as usize;
    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let mut raw_start_to_view = vec![u32::MAX; raw_state_count];

    let start_closure = tokenizer
        .execute_from_state_end_only(&[], tokenizer.initial_state_id())
        .to_vec();
    let start_state = intern_config(start_closure, &mut config_ids, &mut configs);
    for &raw_state in raw_start_states {
        assert!(raw_state < raw_state_count, "invalid raw NFA analysis seed");
        let closure = tokenizer
            .execute_from_state_end_only(&[], raw_state as u32)
            .to_vec();
        let state = intern_config(closure, &mut config_ids, &mut configs);
        raw_start_to_view[raw_state] = state;
    }

    let mut transitions = vec![u32::MAX; configs.len() * 256];
    let mut known_transitions = vec![0u8; transitions.len()];
    let token_trie = build_byte_trie(tokens.iter().copied());
    let suffix_trie = build_byte_trie(
        tokens
            .iter()
            .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..])),
    );
    let mut seeded_configs = raw_start_states
        .iter()
        .map(|&raw| raw_start_to_view[raw])
        .collect::<Vec<_>>();
    seeded_configs.sort_unstable();
    seeded_configs.dedup();
    let mut token_visited = rustc_hash::FxHashSet::<(u32, usize)>::default();
    for state in seeded_configs {
        expand_trie_from_config(
            tokenizer,
            state,
            &token_trie,
            &mut configs,
            &mut config_ids,
            &mut transitions,
            &mut known_transitions,
            &mut token_visited,
        );
    }
    let mut suffix_visited = rustc_hash::FxHashSet::<(u32, usize)>::default();
    expand_trie_from_config(
        tokenizer,
        start_state,
        &suffix_trie,
        &mut configs,
        &mut config_ids,
        &mut transitions,
        &mut known_transitions,
        &mut suffix_visited,
    );

    let states = configs
        .iter()
        .map(|config| FlatDfaState {
            finalizers: filtered_config_groups(tokenizer, config, active_groups, true),
            possible_future_group_ids: filtered_config_groups(
                tokenizer,
                config,
                active_groups,
                false,
            ),
        })
        .collect();
    BoundedAnalysisView {
        tokenizer_view: TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state: start_state as usize,
                transitions: Arc::from(transitions),
            },
        },
        raw_start_to_view,
    }
}

fn intern_config(
    states: Vec<u32>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    configs: &mut Vec<Box<[u32]>>,
) -> u32 {
    if let Some(&id) = config_ids.get(&states) {
        return id;
    }
    let id = configs.len() as u32;
    config_ids.insert(states.clone(), id);
    configs.push(states.into_boxed_slice());
    id
}

fn inherited_classes(
    num_states: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> Vec<u32> {
    let mut classes = vec![u32::MAX; num_states];
    let mut next = initial_state_map.map_or(0, |map| map.num_internal_ids());
    if let Some(map) = initial_state_map {
        for (state, &class) in map.original_to_internal.iter().enumerate().take(num_states) {
            if class != u32::MAX {
                classes[state] = class;
            }
        }
    }
    for class in &mut classes {
        if *class == u32::MAX {
            *class = next;
            next += 1;
        }
    }
    classes
}

fn observation_words(
    tokenizer: &Tokenizer,
    states: &[u32],
    active_groups: Option<&[bool]>,
) -> Vec<u64> {
    let terminal_count = tokenizer.num_terminals() as usize;
    let word_count = terminal_count.div_ceil(64);
    let mut matched = vec![0u64; word_count];
    let mut future = vec![0u64; word_count];
    for &state in states {
        for terminal in tokenizer.matched_terminals_iter(state) {
            let terminal = terminal as usize;
            if active_groups.is_none_or(|active| active.get(terminal).copied().unwrap_or(false)) {
                matched[terminal >> 6] |= 1u64 << (terminal & 63);
            }
        }
        for terminal in tokenizer.possible_future_terminals_iter(state) {
            let terminal = terminal as usize;
            if active_groups.is_none_or(|active| active.get(terminal).copied().unwrap_or(false)) {
                future[terminal >> 6] |= 1u64 << (terminal & 63);
            }
        }
    }
    matched.extend(future);
    matched
}

fn same_partition(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut left_to_right = FxHashMap::<u32, u32>::default();
    let mut right_to_left = FxHashMap::<u32, u32>::default();
    for (&left, &right) in left.iter().zip(right) {
        if left_to_right.insert(left, right).is_some_and(|mapped| mapped != right) {
            return false;
        }
        if right_to_left.insert(right, left).is_some_and(|mapped| mapped != left) {
            return false;
        }
    }
    true
}

fn build_state_map(classes: &[u32]) -> ManyToOneIdMap {
    let mut class_to_internal = FxHashMap::<u32, u32>::default();
    let mut original_to_internal = vec![u32::MAX; classes.len()];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();
    for (state, &class) in classes.iter().enumerate() {
        let internal = if let Some(&internal) = class_to_internal.get(&class) {
            internal
        } else {
            let internal = internal_to_originals.len() as u32;
            class_to_internal.insert(class, internal);
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(state as u32);
            internal
        };
        original_to_internal[state] = internal;
        internal_to_originals[internal as usize].push(state as u32);
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

pub(crate) fn compute_state_map(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    depth: RefinementDepth,
) -> ManyToOneIdMap {
    let num_states = tokenizer.num_states() as usize;
    if num_states == 0 {
        return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(Vec::new(), 0);
    }
    let active_bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &active)| active.then_some(byte as u8))
        .collect::<Vec<_>>();
    let inherited = inherited_classes(num_states, initial_state_map);

    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let start_configs = (0..tokenizer.num_states())
        .map(|state| {
            intern_config(
                tokenizer
                    .execute_from_state_end_only(&[], state)
                    .to_vec(),
                &mut config_ids,
                &mut configs,
            )
        })
        .collect::<Vec<_>>();

    let observations = start_configs
        .iter()
        .map(|&config| observation_words(tokenizer, &configs[config as usize], active_groups))
        .collect::<Vec<_>>();
    let mut initial_keys = FxHashMap::<(u32, Vec<u64>), u32>::default();
    let mut classes = vec![0u32; num_states];
    for state in 0..num_states {
        let key = (inherited[state], observations[state].clone());
        let next = initial_keys.len() as u32;
        classes[state] = *initial_keys.entry(key).or_insert(next);
    }

    let mut target_configs = vec![u32::MAX; num_states * active_bytes.len()];
    for state in 0..num_states {
        let source = configs[start_configs[state] as usize].to_vec();
        for (slot, &byte) in active_bytes.iter().enumerate() {
            let target = tokenizer.step_all(&source, byte);
            if !target.is_empty() {
                target_configs[state * active_bytes.len() + slot] =
                    intern_config(target.to_vec(), &mut config_ids, &mut configs);
            }
        }
    }

    let round_limit = match depth {
        RefinementDepth::Stable => num_states,
        RefinementDepth::Bounded(rounds) => rounds,
    };
    for _ in 0..round_limit {
        let mut signatures = FxHashMap::<Vec<u32>, u32>::default();
        let mut next_classes = vec![0u32; num_states];
        for state in 0..num_states {
            let mut signature = Vec::<u32>::with_capacity(2 + active_bytes.len() * 2);
            signature.push(inherited[state]);
            signature.push(classes[state]);
            for slot in 0..active_bytes.len() {
                let config = target_configs[state * active_bytes.len() + slot];
                if config == u32::MAX {
                    signature.push(0);
                    continue;
                }
                let mut target_classes = configs[config as usize]
                    .iter()
                    .map(|&target| classes[target as usize] + 1)
                    .collect::<Vec<_>>();
                target_classes.sort_unstable();
                target_classes.dedup();
                signature.push(target_classes.len() as u32 + 1);
                signature.extend(target_classes);
            }
            let next = signatures.len() as u32;
            next_classes[state] = *signatures.entry(signature).or_insert(next);
        }
        let stable = same_partition(&classes, &next_classes);
        classes = next_classes;
        if stable {
            break;
        }
    }

    build_state_map(&classes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

    #[test]
    fn set_valued_refinement_distinguishes_epsilon_successor_class_sets() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let map = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            None,
            RefinementDepth::Stable,
        );
        assert_ne!(map.original_to_internal[2], map.original_to_internal[4]);
        assert_ne!(map.original_to_internal[1], map.original_to_internal[2]);
    }
}
