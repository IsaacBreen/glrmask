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
    pub(crate) states: Vec<FlatDfaState>,
    pub(crate) start_state: usize,
    pub(crate) bytes: Vec<u8>,
    pub(crate) edge_offsets: Vec<u32>,
    pub(crate) edges: Vec<(u8, u32)>,
    pub(crate) raw_start_to_view: Arc<[u32]>,
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
    state_map: Option<&ManyToOneIdMap>,
) -> RelevantPowersetView {
    let raw_state_count = tokenizer.num_states() as usize;
    if let Some(state_map) = state_map {
        assert_eq!(state_map.original_to_internal.len(), raw_state_count);
        assert_eq!(
            state_map.internal_to_originals.len(),
            state_map.representative_original_ids.len(),
        );
        assert!(state_map.original_to_internal.iter().all(|&class| {
            class != u32::MAX
                && (class as usize) < state_map.representative_original_ids.len()
        }));
    }
    let (mut config_ids, mut configs, raw_start_to_view, mut worklist, mut queued) =
        if let Some(state_map) = state_map {
            let class_count = state_map.representative_original_ids.len();
            let configs = (0..class_count as u32)
                .map(|class| vec![class].into_boxed_slice())
                .collect::<Vec<_>>();
            (
                FxHashMap::<Vec<u32>, u32>::default(),
                configs,
                state_map.original_to_internal.clone(),
                (0..class_count as u32).collect::<VecDeque<_>>(),
                vec![true; class_count],
            )
        } else {
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
            (
                config_ids,
                configs,
                raw_start_to_view,
                worklist,
                queued,
            )
        };

    let start_state = raw_start_to_view[tokenizer.initial_state_id() as usize] as usize;
    let bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &relevant)| relevant.then_some(byte as u8))
        .collect::<Vec<_>>();
    let mut edge_offsets = Vec::<u32>::with_capacity(configs.len() + 1);
    let mut edges = Vec::<(u8, u32)>::new();
    edge_offsets.push(0);
    if let Some(state_map) = state_map {
        while let Some(state) = worklist.pop_front() {
            assert_eq!(
                state as usize + 1,
                edge_offsets.len(),
                "powerset states must be processed in interning order",
            );
            let config = configs[state as usize].clone();
            let source_states = config
                .iter()
                .map(|&class| state_map.representative_original_ids[class as usize])
                .collect::<Vec<_>>();
            for &byte in &bytes {
                let targets = tokenizer.step_all(&source_states, byte);
                if targets.is_empty() {
                    continue;
                }
                let mut target_config = targets
                    .iter()
                    .map(|&raw_state| state_map.original_to_internal[raw_state as usize])
                    .collect::<Vec<_>>();
                target_config.sort_unstable();
                target_config.dedup();
                let target = if target_config.len() == 1 {
                    target_config[0]
                } else {
                    intern_config(target_config, &mut config_ids, &mut configs)
                };
                if queued.len() < configs.len() {
                    queued.resize(configs.len(), false);
                }
                edges.push((byte, target));
                if !queued[target as usize] {
                    queued[target as usize] = true;
                    worklist.push_back(target);
                }
            }
            edge_offsets.push(edges.len() as u32);
        }
    } else {
        while let Some(state) = worklist.pop_front() {
            assert_eq!(
                state as usize + 1,
                edge_offsets.len(),
                "powerset states must be processed in interning order",
            );
            let config = configs[state as usize].clone();
            for &byte in &bytes {
                let targets = tokenizer.step_all(&config, byte);
                if targets.is_empty() {
                    continue;
                }
                let target = intern_config(targets.to_vec(), &mut config_ids, &mut configs);
                if queued.len() < configs.len() {
                    queued.resize(configs.len(), false);
                }
                edges.push((byte, target));
                if !queued[target as usize] {
                    queued[target as usize] = true;
                    worklist.push_back(target);
                }
            }
            edge_offsets.push(edges.len() as u32);
        }
    }

    let states = if let Some(state_map) = state_map {
        let class_states = state_map
            .representative_original_ids
            .iter()
            .map(|&representative| {
                let closure = tokenizer.execute_from_state_end_only(&[], representative);
                FlatDfaState {
                    finalizers: filtered_config_groups(
                        tokenizer,
                        &closure,
                        active_groups,
                        true,
                    ),
                    possible_future_group_ids: filtered_config_groups(
                        tokenizer,
                        &closure,
                        active_groups,
                        false,
                    ),
                }
            })
            .collect::<Vec<_>>();
        configs
            .iter()
            .map(|config| {
                if config.len() == 1 {
                    return class_states[config[0] as usize].clone();
                }
                let mut finalizers = Vec::<usize>::new();
                let mut possible_future_group_ids = Vec::<usize>::new();
                for &class in config.iter() {
                    let class_state = &class_states[class as usize];
                    finalizers.extend(class_state.finalizers.iter().copied());
                    possible_future_group_ids
                        .extend(class_state.possible_future_group_ids.iter().copied());
                }
                finalizers.sort_unstable();
                finalizers.dedup();
                possible_future_group_ids.sort_unstable();
                possible_future_group_ids.dedup();
                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect()
    } else {
        configs
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
            .collect()
    };
    RelevantPowersetView {
        states,
        start_state,
        bytes,
        edge_offsets,
        edges,
        raw_start_to_view: Arc::from(raw_start_to_view),
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

fn candidate_partition(
    num_states: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let mut members = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<usize>::new();
    let mut raw_to_candidate = vec![usize::MAX; num_states];

    if let Some(map) = initial_state_map {
        for originals in &map.internal_to_originals {
            let mut candidate_members = Vec::with_capacity(originals.len());
            for &raw in originals {
                let raw = raw as usize;
                if raw < num_states && raw_to_candidate[raw] == usize::MAX {
                    candidate_members.push(raw as u32);
                }
            }
            if candidate_members.is_empty() {
                continue;
            }
            let candidate = members.len();
            let representative = candidate_members[0] as usize;
            for &raw in &candidate_members {
                raw_to_candidate[raw as usize] = candidate;
            }
            representatives.push(representative);
            members.push(candidate_members);
        }
    }

    for raw in 0..num_states {
        if raw_to_candidate[raw] != usize::MAX {
            continue;
        }
        let candidate = members.len();
        raw_to_candidate[raw] = candidate;
        representatives.push(raw);
        members.push(vec![raw as u32]);
    }

    (members, representatives, raw_to_candidate)
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

fn build_state_map(
    candidate_members: &[Vec<u32>],
    candidate_representatives: &[usize],
    candidate_classes: &[u32],
    num_states: usize,
) -> ManyToOneIdMap {
    let num_classes = candidate_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class + 1);
    let mut original_to_internal = vec![u32::MAX; num_states];
    let mut internal_to_originals = vec![Vec::new(); num_classes as usize];
    let mut representative_original_ids = vec![u32::MAX; num_classes as usize];

    for ((members, &representative), &class) in candidate_members
        .iter()
        .zip(candidate_representatives)
        .zip(candidate_classes)
    {
        let bucket = &mut internal_to_originals[class as usize];
        if bucket.is_empty() {
            representative_original_ids[class as usize] = representative as u32;
        }
        for &raw in members {
            original_to_internal[raw as usize] = class;
            bucket.push(raw);
        }
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
    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let num_candidates = candidate_representatives.len();

    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let start_configs = candidate_representatives
        .iter()
        .map(|&state| {
            intern_config(
                tokenizer
                    .execute_from_state_end_only(&[], state as u32)
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
    let mut initial_keys = FxHashMap::<Vec<u64>, u32>::default();
    let mut classes = vec![0u32; num_candidates];
    for candidate in 0..num_candidates {
        let key = observations[candidate].clone();
        let next = initial_keys.len() as u32;
        classes[candidate] = *initial_keys.entry(key).or_insert(next);
    }

    let mut target_configs = vec![u32::MAX; num_candidates * active_bytes.len()];
    for candidate in 0..num_candidates {
        let source = configs[start_configs[candidate] as usize].to_vec();
        for (slot, &byte) in active_bytes.iter().enumerate() {
            let target = tokenizer.step_all(&source, byte);
            if !target.is_empty() {
                target_configs[candidate * active_bytes.len() + slot] =
                    intern_config(target.to_vec(), &mut config_ids, &mut configs);
            }
        }
    }

    let round_limit = match depth {
        RefinementDepth::Stable => num_candidates,
        RefinementDepth::Bounded(rounds) => rounds,
    };
    for _ in 0..round_limit {
        let mut signatures = FxHashMap::<Vec<u32>, u32>::default();
        let mut next_classes = vec![0u32; num_candidates];
        for candidate in 0..num_candidates {
            let mut signature = Vec::<u32>::with_capacity(1 + active_bytes.len() * 2);
            signature.push(classes[candidate]);
            for slot in 0..active_bytes.len() {
                let config = target_configs[candidate * active_bytes.len() + slot];
                if config == u32::MAX {
                    signature.push(0);
                    continue;
                }
                let mut target_classes = configs[config as usize]
                    .iter()
                    .map(|&target| classes[raw_to_candidate[target as usize]] + 1)
                    .collect::<Vec<_>>();
                target_classes.sort_unstable();
                target_classes.dedup();
                signature.push(target_classes.len() as u32 + 1);
                signature.extend(target_classes);
            }
            let next = signatures.len() as u32;
            next_classes[candidate] = *signatures.entry(signature).or_insert(next);
        }
        let stable = same_partition(&classes, &next_classes);
        classes = next_classes;
        if stable {
            break;
        }
    }

    build_state_map(
        &candidate_members,
        &candidate_representatives,
        &classes,
        num_states,
    )
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

    #[test]
    fn identity_input_map_does_not_block_nfa_equivalence_merges() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let relevant = [true; 256];
        let identity = super::super::identity_state_map(tokenizer.num_states() as usize);
        let direct = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            None,
            RefinementDepth::Stable,
        );
        let from_identity = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            Some(&identity),
            RefinementDepth::Stable,
        );

        assert!(direct.num_internal_ids() < tokenizer.num_states());
        assert!(same_partition(
            &direct.original_to_internal,
            &from_identity.original_to_internal,
        ));
    }
}
