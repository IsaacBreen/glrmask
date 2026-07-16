use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

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

enum RepresentativeClosure {
    Singleton(u32),
    Multi(Box<[u32]>),
}

impl RepresentativeClosure {
    #[inline]
    fn singleton(&self) -> Option<u32> {
        match self {
            Self::Singleton(state) => Some(*state),
            Self::Multi(_) => None,
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Singleton(state) => std::slice::from_ref(state),
            Self::Multi(states) => states,
        }
    }
}

fn intern_mapped_target_config(
    targets: &[u32],
    state_map: &ManyToOneIdMap,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    configs: &mut Vec<Box<[u32]>>,
) -> u32 {
    debug_assert!(!targets.is_empty());
    let mut target_config = targets
        .iter()
        .map(|&raw_state| state_map.original_to_internal[raw_state as usize])
        .collect::<Vec<_>>();
    target_config.sort_unstable();
    target_config.dedup();
    if target_config.len() == 1 {
        target_config[0]
    } else {
        intern_config(target_config, config_ids, configs)
    }
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
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let total_started_at = profile_timing.then(std::time::Instant::now);
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
    let seed_started_at = profile_timing.then(std::time::Instant::now);
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
    let seed_ms = seed_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let start_state = raw_start_to_view[tokenizer.initial_state_id() as usize] as usize;
    let bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &relevant)| relevant.then_some(byte as u8))
        .collect::<Vec<_>>();
    let closure_started_at = profile_timing.then(std::time::Instant::now);
    let closure_by_class = state_map.map(|state_map| {
        state_map
            .representative_original_ids
            .iter()
            .map(|&representative| {
                let closure = tokenizer.execute_from_state_end_only(&[], representative);
                if closure.len() == 1 {
                    RepresentativeClosure::Singleton(closure[0])
                } else {
                    RepresentativeClosure::Multi(closure.to_vec().into_boxed_slice())
                }
            })
            .collect::<Vec<_>>()
    });
    let closure_ms = closure_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let singleton_closures = closure_by_class.as_ref().map_or(0, |closures| {
        closures.iter().filter(|closure| closure.singleton().is_some()).count()
    });
    let mut edge_offsets = Vec::<u32>::with_capacity(configs.len() + 1);
    let mut edges = Vec::<(u8, u32)>::new();
    edge_offsets.push(0);
    let walk_started_at = profile_timing.then(std::time::Instant::now);
    let mut singleton_fast_states = 0usize;
    let mut generic_states = 0usize;
    let mut generic_byte_steps = 0usize;
    let mut generic_source_states = 0usize;
    let mut generic_target_states = 0usize;
    if let Some(state_map) = state_map {
        let closure_by_class = closure_by_class
            .as_ref()
            .expect("mapped powerset must retain representative closures");
        while let Some(state) = worklist.pop_front() {
            assert_eq!(
                state as usize + 1,
                edge_offsets.len(),
                "powerset states must be processed in interning order",
            );
            let config = configs[state as usize].clone();
            if config.len() == 1
                && let Some(raw_source) = closure_by_class[config[0] as usize].singleton()
            {
                singleton_fast_states += 1;
                for (byte, raw_target) in tokenizer.transitions_from(raw_source) {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let targets = tokenizer.execute_from_state_end_only(&[], raw_target);
                    debug_assert!(!targets.is_empty());
                    let target = intern_mapped_target_config(
                        &targets,
                        state_map,
                        &mut config_ids,
                        &mut configs,
                    );
                    if queued.len() < configs.len() {
                        queued.resize(configs.len(), false);
                    }
                    edges.push((byte, target));
                    if !queued[target as usize] {
                        queued[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
            } else {
                generic_states += 1;
                let source_states = config
                    .iter()
                    .map(|&class| state_map.representative_original_ids[class as usize])
                    .collect::<Vec<_>>();
                generic_source_states += source_states.len();
                for &byte in &bytes {
                    generic_byte_steps += 1;
                    let targets = tokenizer.step_all(&source_states, byte);
                    if targets.is_empty() {
                        continue;
                    }
                    generic_target_states += targets.len();
                    let target = intern_mapped_target_config(
                        &targets,
                        state_map,
                        &mut config_ids,
                        &mut configs,
                    );
                    if queued.len() < configs.len() {
                        queued.resize(configs.len(), false);
                    }
                    edges.push((byte, target));
                    if !queued[target as usize] {
                        queued[target as usize] = true;
                        worklist.push_back(target);
                    }
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
    let walk_ms = walk_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let states_started_at = profile_timing.then(std::time::Instant::now);
    let states = if let Some(state_map) = state_map {
        let closure_by_class = closure_by_class
            .as_ref()
            .expect("mapped powerset must retain representative closures");
        let class_states = closure_by_class
            .iter()
            .map(|closure| {
                FlatDfaState {
                    finalizers: filtered_config_groups(
                        tokenizer,
                        closure.as_slice(),
                        active_groups,
                        true,
                    ),
                    possible_future_group_ids: filtered_config_groups(
                        tokenizer,
                        closure.as_slice(),
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
    let states_ms = states_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    if profile_timing {
        eprintln!(
            "[glrmask/profile][relevant_powerset_view] mapped={} raw_states={} seed_configs={} selected_bytes={} singleton_closures={} final_configs={} edges={} singleton_fast_states={} generic_states={} generic_byte_steps={} generic_source_states={} generic_target_states={} seed_ms={:.3} closure_ms={:.3} walk_ms={:.3} states_ms={:.3} total_ms={:.3}",
            state_map.is_some(),
            raw_state_count,
            state_map.map_or(raw_start_to_view.len(), |map| map.representative_original_ids.len()),
            bytes.len(),
            singleton_closures,
            configs.len(),
            edges.len(),
            singleton_fast_states,
            generic_states,
            generic_byte_steps,
            generic_source_states,
            generic_target_states,
            seed_ms,
            closure_ms,
            walk_ms,
            states_ms,
            total_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
        );
    }
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
    build_bounded_analysis_view_impl(tokenizer, raw_start_states, tokens, active_groups, true)
}

fn build_bounded_analysis_view_impl(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    factor_common_first_byte: bool,
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
    let common_first_byte = factor_common_first_byte
        .then(|| {
            let first = tokens.first()?.first().copied()?;
            tokens
                .iter()
                .all(|token| token.len() > 1 && token.first().copied() == Some(first))
                .then_some(first)
        })
        .flatten();
    let token_trie = if common_first_byte.is_some() {
        build_byte_trie(tokens.iter().map(|token| &token[1..]))
    } else {
        build_byte_trie(tokens.iter().copied())
    };
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
    if let Some(first_byte) = common_first_byte {
        seeded_configs = seeded_configs
            .into_iter()
            .filter_map(|state| {
                let target = ensure_config_transition(
                    tokenizer,
                    state,
                    first_byte,
                    &mut configs,
                    &mut config_ids,
                    &mut transitions,
                    &mut known_transitions,
                );
                (target != u32::MAX).then_some(target)
            })
            .collect();
        seeded_configs.sort_unstable();
        seeded_configs.dedup();
    }
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

fn blocks_from_classes(classes: &[u32], class_count: usize) -> Vec<Vec<u32>> {
    let mut blocks = vec![Vec::new(); class_count];
    for (state, &class) in classes.iter().enumerate() {
        blocks[class as usize].push(state as u32);
    }
    blocks
}

/// Coarsest strong-bisimulation refinement of an observation partition over a
/// labelled transition relation.
///
/// A candidate's transition on one byte is a set of successor candidates.
/// Therefore two candidates are equivalent exactly when their observations
/// agree and, for every byte, they reach the same set of equivalence classes.
/// Splitting by `Pre_byte(block)` for every current block computes that fixed
/// point directly. Unlike DFA Hopcroft refinement, both halves of every split
/// are re-queued: an NFA source may reach both halves, so predecessor-to-one-
/// half does not determine predecessor-to-the-other.
fn refine_relation_stable(
    initial_classes: &[u32],
    inverse: &[Vec<(u8, u32)>],
) -> Vec<u32> {
    let num_states = initial_classes.len();
    let mut partition = initial_classes.to_vec();
    let initial_count = partition
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut blocks = blocks_from_classes(&partition, initial_count);

    let mut worklist: VecDeque<u32> = (0..blocks.len() as u32).collect();
    let mut in_worklist = vec![true; blocks.len()];
    let mut source_set = vec![false; num_states];
    let mut sources_to_clear = Vec::<u32>::with_capacity(num_states.min(10_000));
    let mut touched_blocks = Vec::<u32>::with_capacity(1024);
    let mut block_touched = vec![false; blocks.len()];
    let mut block_sources = vec![Vec::<u32>::new(); blocks.len()];
    let mut input_sources: [Vec<u32>; 256] = std::array::from_fn(|_| Vec::new());
    let mut touched_inputs = Vec::<u8>::with_capacity(64);

    while let Some(splitter_block) = worklist.pop_front() {
        let splitter = splitter_block as usize;
        if splitter >= blocks.len() || blocks[splitter].is_empty() {
            continue;
        }
        in_worklist[splitter] = false;
        touched_inputs.clear();
        for &target in &blocks[splitter] {
            for &(byte, source) in &inverse[target as usize] {
                let sources = &mut input_sources[byte as usize];
                if sources.is_empty() {
                    touched_inputs.push(byte);
                }
                sources.push(source);
            }
        }

        for &byte in &touched_inputs {
            sources_to_clear.clear();
            let sources = &mut input_sources[byte as usize];
            for &source in sources.iter() {
                if !source_set[source as usize] {
                    source_set[source as usize] = true;
                    sources_to_clear.push(source);
                    let block = partition[source as usize] as usize;
                    if !block_touched[block] {
                        block_touched[block] = true;
                        touched_blocks.push(block as u32);
                    }
                    block_sources[block].push(source);
                }
            }
            sources.clear();

            for &block_id in &touched_blocks {
                let block = block_id as usize;
                let block_len = blocks[block].len();
                let source_count = block_sources[block].len();
                if block_len <= 1 || source_count == 0 || source_count == block_len {
                    continue;
                }

                let new_block = blocks.len();
                let old_members = std::mem::take(&mut blocks[block]);
                let mut remaining = Vec::with_capacity(block_len - source_count);
                let mut moved = Vec::with_capacity(source_count);
                for state in old_members {
                    if source_set[state as usize] {
                        moved.push(state);
                    } else {
                        remaining.push(state);
                    }
                }
                for &state in &moved {
                    partition[state as usize] = new_block as u32;
                }
                blocks[block] = remaining;
                blocks.push(moved);
                in_worklist.push(false);
                block_touched.push(false);
                block_sources.push(Vec::new());

                // This relation is nondeterministic: a source may have byte
                // edges into both split halves. Re-queue both halves rather
                // than applying DFA Hopcroft's smaller-half shortcut.
                if !in_worklist[block] {
                    in_worklist[block] = true;
                    worklist.push_back(block as u32);
                }
                if !in_worklist[new_block] {
                    in_worklist[new_block] = true;
                    worklist.push_back(new_block as u32);
                }
            }

            for &source in &sources_to_clear {
                source_set[source as usize] = false;
            }
            for &block_id in &touched_blocks {
                let block = block_id as usize;
                block_touched[block] = false;
                block_sources[block].clear();
            }
            touched_blocks.clear();
        }
    }
    partition
}

fn build_sparse_inverse_relation(
    num_candidates: usize,
    target_offsets: &[u32],
    target_bytes: &[u8],
    sparse_target_configs: &[u32],
    config_candidates: &[Vec<usize>],
) -> Vec<Vec<(u8, u32)>> {
    let mut inverse = vec![Vec::<(u8, u32)>::new(); num_candidates];
    for source in 0..num_candidates {
        let start = target_offsets[source] as usize;
        let end = target_offsets[source + 1] as usize;
        for edge in start..end {
            let byte = target_bytes[edge];
            let config = sparse_target_configs[edge] as usize;
            for &target in &config_candidates[config] {
                inverse[target].push((byte, source as u32));
            }
        }
    }
    inverse
}

pub(crate) fn compute_state_map(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    depth: RefinementDepth,
) -> ManyToOneIdMap {
    let profile = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let num_states = tokenizer.num_states() as usize;
    if num_states == 0 {
        return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(Vec::new(), 0);
    }
    let setup_started_at = profile.then(Instant::now);
    let active_bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &active)| active.then_some(byte as u8))
        .collect::<Vec<_>>();
    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let num_candidates = candidate_representatives.len();
    let setup_ms = setup_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let closures_started_at = profile.then(Instant::now);
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
    let closures_ms = closures_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let start_config_count = configs.len();

    let observations_started_at = profile.then(Instant::now);
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
    let observations_ms = observations_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let initial_class_count = initial_keys.len();

    let targets_started_at = profile.then(Instant::now);
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
    let targets_ms = targets_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let all_config_count = configs.len();

    // The NFA transition relation is extremely sparse. Encode only live
    // candidate/byte cells once, and project each distinct successor
    // configuration through the current partition once per refinement round.
    // The previous dense loop revisited every absent byte cell and repeatedly
    // sorted the same successor configuration for every source that reached
    // it. That becomes catastrophic on deep 40k-state epsilon lexers.
    let mut target_offsets = Vec::with_capacity(num_candidates + 1);
    let mut target_bytes = Vec::<u8>::new();
    let mut sparse_target_configs = Vec::<u32>::new();
    let mut referenced_config = vec![false; configs.len()];
    target_offsets.push(0u32);
    for candidate in 0..num_candidates {
        for (slot, &byte) in active_bytes.iter().enumerate() {
            let config = target_configs[candidate * active_bytes.len() + slot];
            if config == u32::MAX {
                continue;
            }
            target_bytes.push(byte);
            sparse_target_configs.push(config);
            referenced_config[config as usize] = true;
        }
        target_offsets.push(target_bytes.len() as u32);
    }
    let referenced_configs = referenced_config
        .iter()
        .enumerate()
        .filter_map(|(config, &referenced)| referenced.then_some(config))
        .collect::<Vec<_>>();
    let mut config_candidates = vec![Vec::<usize>::new(); configs.len()];
    for &config in &referenced_configs {
        let candidates = &mut config_candidates[config];
        candidates.extend(
            configs[config]
                .iter()
                .map(|&target| raw_to_candidate[target as usize]),
        );
        candidates.sort_unstable();
        candidates.dedup();
    }
    let nondead_target_cells = sparse_target_configs.len();
    let unique_target_configs = referenced_configs.len();

    let refine_started_at = profile.then(Instant::now);
    let mut rounds = 0usize;
    let mut final_class_count = initial_class_count;
    match depth {
        RefinementDepth::Stable => {
            let inverse = build_sparse_inverse_relation(
                num_candidates,
                &target_offsets,
                &target_bytes,
                &sparse_target_configs,
                &config_candidates,
            );
            classes = refine_relation_stable(&classes, &inverse);
            final_class_count = classes
                .iter()
                .copied()
                .max()
                .map_or(0usize, |class| class as usize + 1);
        }
        RefinementDepth::Bounded(round_limit) => {
            let mut projected_classes = vec![Vec::<u32>::new(); configs.len()];
            let mut current_class_count = initial_class_count;
            for _ in 0..round_limit {
                rounds += 1;
                for &config in &referenced_configs {
                    let projected = &mut projected_classes[config];
                    projected.clear();
                    projected.extend(
                        config_candidates[config]
                            .iter()
                            .map(|&target| classes[target] + 1),
                    );
                    projected.sort_unstable();
                    projected.dedup();
                }

                let mut signatures = FxHashMap::<Vec<u32>, u32>::default();
                let mut next_classes = vec![0u32; num_candidates];
                let mut signature = Vec::<u32>::new();
                for candidate in 0..num_candidates {
                    signature.clear();
                    signature.push(classes[candidate]);
                    let start = target_offsets[candidate] as usize;
                    let end = target_offsets[candidate + 1] as usize;
                    for edge in start..end {
                        let config = sparse_target_configs[edge] as usize;
                        let target_classes = &projected_classes[config];
                        // The byte and target-set length make absent bytes and
                        // the variable-width class set explicit here.
                        signature.push(target_bytes[edge] as u32 + 1);
                        signature.push(target_classes.len() as u32 + 1);
                        signature.extend_from_slice(target_classes);
                    }
                    next_classes[candidate] =
                        if let Some(&class) = signatures.get(&signature) {
                            class
                        } else {
                            let class = signatures.len() as u32;
                            signatures.insert(signature.clone(), class);
                            class
                        };
                }
                final_class_count = signatures.len();
                // Every signature includes the source's current class, so
                // each round is a pure refinement. Equal class count means no
                // class was split and the partition is stable.
                let stable = final_class_count == current_class_count;
                classes = next_classes;
                current_class_count = final_class_count;
                if stable {
                    break;
                }
            }
        }
    }
    let refine_ms = refine_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    if profile {
        eprintln!(
            "[glrmask/profile][nfa_state_equiv] states={} candidates={} active_bytes={} nondead_target_cells={} unique_target_configs={} start_configs={} all_configs={} initial_classes={} final_classes={} rounds={} setup_ms={:.3} closures_ms={:.3} observations_ms={:.3} targets_ms={:.3} refine_ms={:.3}",
            num_states,
            num_candidates,
            active_bytes.len(),
            nondead_target_cells,
            unique_target_configs,
            start_config_count,
            all_config_count,
            initial_class_count,
            final_class_count,
            rounds,
            setup_ms,
            closures_ms,
            observations_ms,
            targets_ms,
            refine_ms,
        );
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

    fn bounded_view_trace(
        view: &BoundedAnalysisView,
        start_state: usize,
        token: &[u8],
    ) -> Vec<(Vec<usize>, Vec<usize>, bool)> {
        let dfa = view.tokenizer_view.dfa();
        let mut state = start_state;
        let mut trace = vec![(
            dfa.states[state].finalizers.clone(),
            dfa.states[state].possible_future_group_ids.clone(),
            false,
        )];
        for &byte in token {
            let target = dfa.trans(state, byte as usize);
            if target == u32::MAX {
                trace.push((Vec::new(), Vec::new(), true));
                break;
            }
            state = target as usize;
            trace.push((
                dfa.states[state].finalizers.clone(),
                dfa.states[state].possible_future_group_ids.clone(),
                false,
            ));
        }
        trace
    }

    #[test]
    fn bounded_nfa_common_first_factorization_preserves_observed_token_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [b"aa".as_slice(), b"ab".as_slice(), b"aab".as_slice()];
        let factored =
            build_bounded_analysis_view_impl(&tokenizer, &raw_start_states, &tokens, None, true);
        let reference =
            build_bounded_analysis_view_impl(&tokenizer, &raw_start_states, &tokens, None, false);

        for &raw_state in &raw_start_states {
            let factored_start = factored.view_state_for_raw_start(raw_state);
            let reference_start = reference.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    bounded_view_trace(&factored, factored_start, token),
                    bounded_view_trace(&reference, reference_start, token),
                );
            }
        }

        for token in tokens {
            for offset in 0..token.len() {
                assert_eq!(
                    bounded_view_trace(
                        &factored,
                        factored.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                    bounded_view_trace(
                        &reference,
                        reference.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                );
            }
        }
    }

    fn compute_state_map_dense_reference(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        active_groups: Option<&[bool]>,
        initial_state_map: Option<&ManyToOneIdMap>,
        depth: RefinementDepth,
    ) -> ManyToOneIdMap {
        let num_states = tokenizer.num_states() as usize;
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

    #[test]
    fn sparse_nfa_refinement_matches_dense_reference() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let identity = super::super::identity_state_map(tokenizer.num_states() as usize);
        let mut relevant_ab = [false; 256];
        relevant_ab[b'a' as usize] = true;
        relevant_ab[b'b' as usize] = true;
        let relevant_all = [true; 256];

        for relevant in [&relevant_ab, &relevant_all] {
            for depth in [
                RefinementDepth::Bounded(0),
                RefinementDepth::Bounded(1),
                RefinementDepth::Bounded(2),
                RefinementDepth::Bounded(5),
                RefinementDepth::Stable,
            ] {
                for initial in [None, Some(&identity)] {
                    let expected = compute_state_map_dense_reference(
                        &tokenizer,
                        relevant,
                        None,
                        initial,
                        depth,
                    );
                    let actual = compute_state_map(
                        &tokenizer,
                        relevant,
                        None,
                        initial,
                        depth,
                    );
                    assert!(
                        same_partition(
                            &expected.original_to_internal,
                            &actual.original_to_internal,
                        ),
                        "depth={depth:?} initial_map={} relevant_all={}",
                        initial.is_some(),
                        std::ptr::eq(relevant, &relevant_all),
                    );
                }
            }
        }
    }

    #[test]
    fn stable_relation_refinement_matches_signature_fixed_point_on_random_relations() {
        fn next_u32(seed: &mut u64) -> u32 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*seed >> 32) as u32
        }

        for case in 0..256u64 {
            let mut seed = 0x9e37_79b9_7f4a_7c15 ^ case;
            let num_states = 1 + next_u32(&mut seed) as usize % 20;
            let num_bytes = 1 + next_u32(&mut seed) as usize % 5;
            let num_labels = 1 + next_u32(&mut seed) as usize % 4;
            let initial_classes = (0..num_states)
                .map(|_| next_u32(&mut seed) % num_labels as u32)
                .collect::<Vec<_>>();
            let mut relation = vec![vec![Vec::<usize>::new(); num_bytes]; num_states];
            let mut inverse = vec![Vec::<(u8, u32)>::new(); num_states];
            for source in 0..num_states {
                for byte in 0..num_bytes {
                    for target in 0..num_states {
                        if next_u32(&mut seed) % 7 == 0 {
                            relation[source][byte].push(target);
                            inverse[target].push((byte as u8, source as u32));
                        }
                    }
                }
            }

            let mut expected = initial_classes.clone();
            for _ in 0..num_states {
                let mut signatures = FxHashMap::<Vec<u32>, u32>::default();
                let mut next = vec![0u32; num_states];
                for source in 0..num_states {
                    let mut signature = vec![expected[source]];
                    for byte in 0..num_bytes {
                        let mut target_classes = relation[source][byte]
                            .iter()
                            .map(|&target| expected[target] + 1)
                            .collect::<Vec<_>>();
                        target_classes.sort_unstable();
                        target_classes.dedup();
                        signature.push(target_classes.len() as u32 + 1);
                        signature.extend(target_classes);
                    }
                    let class = signatures.len() as u32;
                    next[source] = *signatures.entry(signature).or_insert(class);
                }
                if same_partition(&expected, &next) {
                    expected = next;
                    break;
                }
                expected = next;
            }

            let actual = refine_relation_stable(&initial_classes, &inverse);
            assert!(
                same_partition(&expected, &actual),
                "case={case} states={num_states} bytes={num_bytes}",
            );
        }
    }
}
