use rustc_hash::FxHashMap;
use smallvec::SmallVec;
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
    pub(crate) configurations: Arc<[Box<[u32]>]>,
}

pub(crate) struct PrebuiltSparsePowersetRefinement<'a> {
    pub(crate) raw_start_to_view: &'a [u32],
    pub(crate) configurations: &'a [Box<[u32]>],
    pub(crate) output_class_by_config: &'a [u32],
    pub(crate) edge_offsets: &'a [u32],
    pub(crate) edges: &'a [(u8, u32)],
}

impl PrebuiltSparsePowersetRefinement<'_> {
    pub(crate) fn compute_state_map(
        &self,
        tokenizer: &Tokenizer,
        initial_state_map: Option<&ManyToOneIdMap>,
        depth: RefinementDepth,
    ) -> ManyToOneIdMap {
        compute_state_map_from_prebuilt_sparse_powerset(
            tokenizer,
            initial_state_map,
            depth,
            self.raw_start_to_view,
            self.configurations,
            self.output_class_by_config,
            None,
            self.edge_offsets,
            self.edges,
        )
    }
}

pub(crate) fn powerset_output_class_ids(view: &RelevantPowersetView) -> Vec<u32> {
    let mut output_ids = FxHashMap::<(Vec<usize>, Vec<usize>), u32>::default();
    view.states
        .iter()
        .map(|state| {
            let key = (
                state.finalizers.clone(),
                state.possible_future_group_ids.clone(),
            );
            let next = output_ids.len() as u32;
            *output_ids.entry(key).or_insert(next)
        })
        .collect()
}

impl RelevantPowersetView {
    pub(crate) fn into_tokenizer_view(self) -> TokenizerView {
        let mut transitions = vec![u32::MAX; self.states.len() * 256];
        for state in 0..self.states.len() {
            let start = self.edge_offsets[state] as usize;
            let end = self.edge_offsets[state + 1] as usize;
            for &(byte, target) in &self.edges[start..end] {
                transitions[state * 256 + byte as usize] = target;
            }
        }
        TokenizerView {
            flat_dfa: FlatDfa {
                states: self.states,
                start_state: self.start_state,
                transitions: Arc::from(transitions),
            },
        }
    }
}

pub(crate) fn build_relevant_powerset_analysis_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    let view = build_relevant_powerset_view(
        tokenizer,
        relevant_bytes,
        Some(active_groups),
        None,
    );
    let raw_start_to_view = view.raw_start_to_view.to_vec();
    BoundedAnalysisView {
        tokenizer_view: view.into_tokenizer_view(),
        raw_start_to_view,
    }
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

pub(crate) fn raw_active_language_states(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
) -> Option<Vec<bool>> {
    let active_groups = active_groups?;
    Some(
        (0..tokenizer.num_states())
            .map(|state| {
                tokenizer
                    .matched_terminals_iter(state)
                    .chain(tokenizer.possible_future_terminals_iter(state))
                    .any(|group| {
                        active_groups
                            .get(group as usize)
                            .copied()
                            .unwrap_or(false)
                    })
            })
            .collect(),
    )
}

fn project_raw_config(mut states: Vec<u32>, active_language: Option<&[bool]>) -> Vec<u32> {
    if let Some(active_language) = active_language {
        states.retain(|&state| active_language[state as usize]);
    }
    states.sort_unstable();
    states.dedup();
    states
}

fn mapped_class_active_language(
    state_map: &ManyToOneIdMap,
    raw_active_language: Option<&[bool]>,
) -> Option<Vec<bool>> {
    raw_active_language.map(|raw_active_language| {
        state_map
            .internal_to_originals
            .iter()
            .map(|members| {
                let active = members
                    .iter()
                    .any(|&raw| raw_active_language[raw as usize]);
                debug_assert!(
                    members
                        .iter()
                        .all(|&raw| raw_active_language[raw as usize] == active),
                    "mapped powerset seed mixed active-live and active-dead scanner states",
                );
                active
            })
            .collect()
    })
}

fn intern_mapped_target_config(
    targets: &[u32],
    state_map: &ManyToOneIdMap,
    class_active_language: Option<&[bool]>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    configs: &mut Vec<Box<[u32]>>,
) -> u32 {
    let mut target_config = targets
        .iter()
        .map(|&raw_state| state_map.original_to_internal[raw_state as usize])
        .filter(|&class| {
            class_active_language.is_none_or(|active| active[class as usize])
        })
        .collect::<Vec<_>>();
    target_config.sort_unstable();
    target_config.dedup();
    intern_config(target_config, config_ids, configs)
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

    // Project epsilon configurations themselves, not only their visible output
    // metadata. A raw component with no active finalizer and no active future
    // terminal has empty active language, so retaining it in a mixed powerset
    // configuration would let inactive-only topology distinguish successors.
    let raw_active_language = raw_active_language_states(tokenizer, active_groups);
    let class_active_language = state_map.and_then(|state_map| {
        mapped_class_active_language(state_map, raw_active_language.as_deref())
    });

    let (mut config_ids, mut configs, raw_start_to_view, mut worklist, mut queued) =
        if let Some(state_map) = state_map {
            let class_count = state_map.representative_original_ids.len();
            let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
            let mut configs = Vec::<Box<[u32]>>::new();
            let mut class_to_view = vec![u32::MAX; class_count];
            let mut worklist = VecDeque::<u32>::new();
            let mut queued = Vec::<bool>::new();
            for class in 0..class_count as u32 {
                let config = if class_active_language
                    .as_deref()
                    .is_none_or(|active| active[class as usize])
                {
                    vec![class]
                } else {
                    Vec::new()
                };
                let state = intern_config(config, &mut config_ids, &mut configs);
                class_to_view[class as usize] = state;
                if queued.len() < configs.len() {
                    queued.resize(configs.len(), false);
                }
                if !queued[state as usize] {
                    queued[state as usize] = true;
                    worklist.push_back(state);
                }
            }
            let raw_start_to_view = state_map
                .original_to_internal
                .iter()
                .map(|&class| class_to_view[class as usize])
                .collect::<Vec<_>>();
            (
                config_ids,
                configs,
                raw_start_to_view,
                worklist,
                queued,
            )
        } else {
            let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
            let mut configs = Vec::<Box<[u32]>>::new();
            let mut raw_start_to_view = vec![u32::MAX; raw_state_count];
            let mut worklist = VecDeque::<u32>::new();
            let mut queued = Vec::<bool>::new();
            for raw_state in 0..raw_state_count {
                let closure = project_raw_config(
                    tokenizer
                        .execute_from_state_end_only(&[], raw_state as u32)
                        .to_vec(),
                    raw_active_language.as_deref(),
                );
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
    let mut edge_offsets = Vec::<u32>::with_capacity(configs.len() + 1);
    let mut edges = Vec::<(u8, u32)>::new();
    edge_offsets.push(0);
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
                for (byte, raw_target) in tokenizer.transitions_from(raw_source) {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let targets = tokenizer.execute_from_state_end_only(&[], raw_target);
                    debug_assert!(!targets.is_empty());
                    let target = intern_mapped_target_config(
                        &targets,
                        state_map,
                        class_active_language.as_deref(),
                        &mut config_ids,
                        &mut configs,
                    );
                    if queued.len() < configs.len() {
                        queued.resize(configs.len(), false);
                    }
                    // The canonical empty projected configuration is the
                    // implicit dead target, exactly like an absent edge.
                    if configs[target as usize].is_empty() {
                        continue;
                    }
                    edges.push((byte, target));
                    if !queued[target as usize] {
                        queued[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
            } else {
                let source_states = config
                    .iter()
                    .map(|&class| state_map.representative_original_ids[class as usize])
                    .collect::<Vec<_>>();
                for &byte in &bytes {
                    let targets = tokenizer.step_all(&source_states, byte);
                    if targets.is_empty() {
                        continue;
                    }
                    let target = intern_mapped_target_config(
                        &targets,
                        state_map,
                        class_active_language.as_deref(),
                        &mut config_ids,
                        &mut configs,
                    );
                    if queued.len() < configs.len() {
                        queued.resize(configs.len(), false);
                    }
                    if configs[target as usize].is_empty() {
                        continue;
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
        let mut byte_marks = [0u32; 256];
        let mut byte_epoch = 0u32;
        let mut candidate_bytes = Vec::<u8>::new();
        while let Some(state) = worklist.pop_front() {
            assert_eq!(
                state as usize + 1,
                edge_offsets.len(),
                "powerset states must be processed in interning order",
            );
            let config = configs[state as usize].clone();
            if let [source] = config.as_ref() {
                // Raw starts were all seeded before traversal, so the exact
                // projected epsilon closure of every direct byte target is
                // already interned in `raw_start_to_view`.
                for (byte, raw_target) in tokenizer.transitions_from(*source) {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let target = raw_start_to_view[raw_target as usize];
                    debug_assert_ne!(target, u32::MAX);
                    if configs[target as usize].is_empty() {
                        continue;
                    }
                    edges.push((byte, target));
                    if !queued[target as usize] {
                        queued[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
                edge_offsets.push(edges.len() as u32);
                continue;
            }

            byte_epoch = byte_epoch.wrapping_add(1);
            if byte_epoch == 0 {
                byte_marks.fill(0);
                byte_epoch = 1;
            }
            candidate_bytes.clear();
            for &source in config.iter() {
                for (byte, _) in tokenizer.transitions_from(source) {
                    let byte_index = byte as usize;
                    if relevant_bytes[byte_index] && byte_marks[byte_index] != byte_epoch {
                        byte_marks[byte_index] = byte_epoch;
                        candidate_bytes.push(byte);
                    }
                }
            }
            candidate_bytes.sort_unstable();
            for &byte in &candidate_bytes {
                let targets = tokenizer.step_all(&config, byte);
                if targets.is_empty() {
                    continue;
                }
                let projected = project_raw_config(
                    targets.to_vec(),
                    raw_active_language.as_deref(),
                );
                if projected.is_empty() {
                    continue;
                }
                let target = intern_config(projected, &mut config_ids, &mut configs);
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
    RelevantPowersetView {
        states,
        start_state,
        bytes,
        edge_offsets,
        edges,
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
    let mut edges = FxHashMap::<u64, usize>::default();
    for sequence in sequences {
        let mut node = 0usize;
        for &byte in sequence {
            let key = ((node as u64) << 8) | byte as u64;
            let child = if let Some(&child) = edges.get(&key) {
                child
            } else {
                let child = nodes.len();
                nodes.push(ByteTrieNode::default());
                nodes[node].children.push((byte, child));
                edges.insert(key, child);
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TokenBoundedAnalysisWork {
    pub(crate) configurations: usize,
    pub(crate) trie_visits: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenBoundedAnalysisWorkBudget {
    pub(crate) max_configurations: usize,
    pub(crate) max_trie_visits: usize,
}

fn ensure_config_transition(
    tokenizer: &Tokenizer,
    state: u32,
    byte: u8,
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    preseeded_raw_closures: Option<&[u32]>,
    active_language: Option<&[bool]>,
) -> u32 {
    let slot = state as usize * 256 + byte as usize;
    if known_transitions[slot] != 0 {
        return transitions[slot];
    }
    known_transitions[slot] = 1;
    if let Some(raw_start_to_view) = preseeded_raw_closures
        && let [source] = configs[state as usize].as_ref()
        && let Some(raw_target) = tokenizer.step(*source, byte)
    {
        let target = raw_start_to_view[raw_target as usize];
        if target != u32::MAX {
            transitions[slot] = target;
            return target;
        }
    }
    let targets = tokenizer.step_all(&configs[state as usize], byte);
    let targets = if let Some(active_language) = active_language {
        targets
            .iter()
            .copied()
            .filter(|&target| active_language[target as usize])
            .collect::<Vec<_>>()
    } else {
        targets.to_vec()
    };
    if targets.is_empty() {
        return u32::MAX;
    }
    let target = intern_config(targets, config_ids, configs);
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
    preseeded_raw_closures: Option<&[u32]>,
    active_language: Option<&[bool]>,
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
                preseeded_raw_closures,
                active_language,
            );
            if target != u32::MAX {
                stack.push((target, child));
            }
        }
    }
}

fn expand_trie_from_config_budgeted(
    tokenizer: &Tokenizer,
    start_state: u32,
    trie: &[ByteTrieNode],
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    visited: &mut rustc_hash::FxHashSet<(u32, usize)>,
    preseeded_raw_closures: Option<&[u32]>,
    active_language: Option<&[bool]>,
    budget: TokenBoundedAnalysisWorkBudget,
    prior_trie_visits: usize,
) -> Result<(), TokenBoundedAnalysisWork> {
    const CONFIG_CHECK_INTERVAL: usize = 4096;
    debug_assert!(CONFIG_CHECK_INTERVAL.is_power_of_two());
    let first_disallowed_trie_visit = budget.max_trie_visits.saturating_add(1);
    let mut stack = vec![(start_state, 0usize)];
    while let Some((state, node)) = stack.pop() {
        if !visited.insert((state, node)) {
            continue;
        }
        let trie_visits = prior_trie_visits + visited.len();
        if trie_visits == first_disallowed_trie_visit
            || (trie_visits & (CONFIG_CHECK_INTERVAL - 1) == 0
                && configs.len() > budget.max_configurations)
        {
            return Err(TokenBoundedAnalysisWork {
                configurations: configs.len(),
                trie_visits,
            });
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
                preseeded_raw_closures,
                active_language,
            );
            if target != u32::MAX {
                stack.push((target, child));
            }
        }
    }
    if configs.len() > budget.max_configurations {
        return Err(TokenBoundedAnalysisWork {
            configurations: configs.len(),
            trie_visits: prior_trie_visits + visited.len(),
        });
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct TokenBoundedAnalysisTopology {
    configurations: Arc<[Box<[u32]>]>,
    start_state: usize,
    transitions: Arc<[u32]>,
    raw_start_to_view: Arc<[u32]>,
}

fn materialize_bounded_analysis_view(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    configurations: &[Box<[u32]>],
    start_state: usize,
    transitions: Arc<[u32]>,
    raw_start_to_view: Vec<u32>,
) -> BoundedAnalysisView {
    let raw_state_count = tokenizer.num_states() as usize;
    let raw_state_outputs = (0..raw_state_count)
        .map(|state| FlatDfaState {
            finalizers: filtered_config_groups(
                tokenizer,
                std::slice::from_ref(&(state as u32)),
                active_groups,
                true,
            ),
            possible_future_group_ids: filtered_config_groups(
                tokenizer,
                std::slice::from_ref(&(state as u32)),
                active_groups,
                false,
            ),
        })
        .collect::<Vec<_>>();
    let states = configurations
        .iter()
        .map(|config| {
            if let [state] = config.as_ref() {
                return raw_state_outputs[*state as usize].clone();
            }
            let mut finalizers = Vec::<usize>::new();
            let mut possible_future_group_ids = Vec::<usize>::new();
            for &state in config.iter() {
                let output = &raw_state_outputs[state as usize];
                finalizers.extend_from_slice(&output.finalizers);
                possible_future_group_ids.extend_from_slice(&output.possible_future_group_ids);
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
        .collect();
    BoundedAnalysisView {
        tokenizer_view: TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state,
                transitions,
            },
        },
        raw_start_to_view,
    }
}

impl TokenBoundedAnalysisTopology {
    pub(crate) fn state_count(&self) -> usize {
        self.configurations.len()
    }

    /// Materialize a topology that was already constructed with the same
    /// active-language projection. Re-projecting such a topology would be
    /// exact but wastefully scan and rebuild its full dense transition table.
    fn materialize_already_projected(
        &self,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> BoundedAnalysisView {
        materialize_bounded_analysis_view(
            tokenizer,
            Some(active_groups),
            &self.configurations,
            self.start_state,
            Arc::clone(&self.transitions),
            self.raw_start_to_view.to_vec(),
        )
    }

    pub(crate) fn materialize(
        &self,
        tokenizer: &Tokenizer,
        active_groups: Option<&[bool]>,
    ) -> BoundedAnalysisView {
        let Some(active_language) = raw_active_language_states(tokenizer, active_groups) else {
            return materialize_bounded_analysis_view(
                tokenizer,
                active_groups,
                &self.configurations,
                self.start_state,
                Arc::clone(&self.transitions),
                self.raw_start_to_view.to_vec(),
            );
        };

        // The token topology is intentionally built once for a superset of
        // terminal masks. Project raw NFA members when the topology is consumed
        // so inactive-only lexer structure cannot distinguish active-language
        // states. Since an inactive-language state has no active-language
        // successor, projecting after each prebuilt transition is equivalent to
        // projecting its source before taking that transition.
        let mut projected_ids = FxHashMap::<Vec<u32>, u32>::default();
        let mut projected_configs = Vec::<Box<[u32]>>::new();
        let projected_state_by_topology_state = self
            .configurations
            .iter()
            .map(|config| {
                let projected = config
                    .iter()
                    .copied()
                    .filter(|&raw| active_language[raw as usize])
                    .collect::<Vec<_>>();
                intern_config(projected, &mut projected_ids, &mut projected_configs)
            })
            .collect::<Vec<_>>();

        let mut projected_transitions =
            vec![u32::MAX; projected_configs.len().saturating_mul(256)];
        for topology_source in 0..self.configurations.len() {
            let source = projected_state_by_topology_state[topology_source] as usize;
            if projected_configs[source].is_empty() {
                continue;
            }
            let topology_row = topology_source * 256;
            for byte in 0..256 {
                let topology_target = self.transitions[topology_row + byte];
                if topology_target == u32::MAX {
                    continue;
                }
                let target = projected_state_by_topology_state[topology_target as usize];
                if projected_configs[target as usize].is_empty() {
                    continue;
                }
                let slot = source * 256 + byte;
                let previous = projected_transitions[slot];
                debug_assert!(
                    previous == u32::MAX || previous == target,
                    "equal projected NFA configs had different projected byte successors",
                );
                projected_transitions[slot] = target;
            }
        }

        let raw_start_to_view = self
            .raw_start_to_view
            .iter()
            .map(|&state| {
                (state != u32::MAX)
                    .then(|| projected_state_by_topology_state[state as usize])
                    .unwrap_or(u32::MAX)
            })
            .collect::<Vec<_>>();
        let start_state = projected_state_by_topology_state[self.start_state] as usize;
        materialize_bounded_analysis_view(
            tokenizer,
            active_groups,
            &projected_configs,
            start_state,
            Arc::from(projected_transitions),
            raw_start_to_view,
        )
    }
}

fn build_bounded_analysis_topology_impl(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    combine_start_states: bool,
    factor_common_first_byte: bool,
    include_reset_suffixes: bool,
    active_groups: Option<&[bool]>,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
) -> Result<(TokenBoundedAnalysisTopology, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let raw_state_count = tokenizer.num_states() as usize;
    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let mut raw_start_to_view = vec![u32::MAX; raw_state_count];
    let singleton_closures = tokenizer.all_singleton_epsilon_closures();
    let active_language = raw_active_language_states(tokenizer, active_groups);
    let project_closure = |closure: &[u32]| {
        if let Some(active_language) = active_language.as_deref() {
            closure
                .iter()
                .copied()
                .filter(|&state| active_language[state as usize])
                .collect::<Vec<_>>()
        } else {
            closure.to_vec()
        }
    };

    let start_closure = project_closure(&singleton_closures[tokenizer.initial_state_id() as usize]);
    let start_state = intern_config(start_closure, &mut config_ids, &mut configs);
    if combine_start_states {
        let mut closure = Vec::<u32>::new();
        for &raw_state in raw_start_states {
            assert!(raw_state < raw_state_count, "invalid raw NFA analysis seed");
            if let Some(active_language) = active_language.as_deref() {
                closure.extend(
                    singleton_closures[raw_state]
                        .iter()
                        .copied()
                        .filter(|&state| active_language[state as usize]),
                );
            } else {
                closure.extend_from_slice(&singleton_closures[raw_state]);
            }
        }
        closure.sort_unstable();
        closure.dedup();
        let state = intern_config(closure, &mut config_ids, &mut configs);
        for &raw_state in raw_start_states {
            raw_start_to_view[raw_state] = state;
        }
    } else {
        for &raw_state in raw_start_states {
            assert!(raw_state < raw_state_count, "invalid raw NFA analysis seed");
            let closure = project_closure(&singleton_closures[raw_state]);
            let state = intern_config(closure, &mut config_ids, &mut configs);
            raw_start_to_view[raw_state] = state;
        }
    }

    if budget.is_some_and(|budget| configs.len() > budget.max_configurations) {
        return Err(TokenBoundedAnalysisWork {
            configurations: configs.len(),
            trie_visits: 0,
        });
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
    let suffix_trie = include_reset_suffixes.then(|| {
        build_byte_trie(
            tokens
                .iter()
                .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..])),
        )
    });
    let mut seeded_configs = raw_start_states
        .iter()
        .map(|&raw| raw_start_to_view[raw])
        .collect::<Vec<_>>();
    seeded_configs.sort_unstable();
    seeded_configs.dedup();
    if let Some(first_byte) = common_first_byte {
        let mut advanced_seeded_configs = Vec::with_capacity(seeded_configs.len());
        for state in seeded_configs {
            let target = ensure_config_transition(
                tokenizer,
                state,
                first_byte,
                &mut configs,
                &mut config_ids,
                &mut transitions,
                &mut known_transitions,
                (!combine_start_states).then_some(raw_start_to_view.as_slice()),
                active_language.as_deref(),
            );
            if target != u32::MAX {
                advanced_seeded_configs.push(target);
            }
        }
        advanced_seeded_configs.sort_unstable();
        advanced_seeded_configs.dedup();
        seeded_configs = advanced_seeded_configs;
        if budget.is_some_and(|budget| configs.len() > budget.max_configurations) {
            return Err(TokenBoundedAnalysisWork {
                configurations: configs.len(),
                trie_visits: 0,
            });
        }
    }
    let preseeded_raw_closures = (!combine_start_states).then_some(raw_start_to_view.as_slice());
    let mut token_visited = rustc_hash::FxHashSet::<(u32, usize)>::default();
    for state in seeded_configs {
        if let Some(budget) = budget {
            expand_trie_from_config_budgeted(
                tokenizer,
                state,
                &token_trie,
                &mut configs,
                &mut config_ids,
                &mut transitions,
                &mut known_transitions,
                &mut token_visited,
                preseeded_raw_closures,
                active_language.as_deref(),
                budget,
                0,
            )?;
        } else {
            expand_trie_from_config(
                tokenizer,
                state,
                &token_trie,
                &mut configs,
                &mut config_ids,
                &mut transitions,
                &mut known_transitions,
                &mut token_visited,
                preseeded_raw_closures,
                active_language.as_deref(),
            );
        }
    }
    let mut suffix_visited = rustc_hash::FxHashSet::<(u32, usize)>::default();
    if let Some(suffix_trie) = suffix_trie.as_ref() {
        if let Some(budget) = budget {
            expand_trie_from_config_budgeted(
                tokenizer,
                start_state,
                suffix_trie,
                &mut configs,
                &mut config_ids,
                &mut transitions,
                &mut known_transitions,
                &mut suffix_visited,
                None,
                active_language.as_deref(),
                budget,
                token_visited.len(),
            )?;
        } else {
            expand_trie_from_config(
                tokenizer,
                start_state,
                suffix_trie,
                &mut configs,
                &mut config_ids,
                &mut transitions,
                &mut known_transitions,
                &mut suffix_visited,
                None,
                active_language.as_deref(),
            );
        }
    }

    let work = TokenBoundedAnalysisWork {
        configurations: configs.len(),
        trie_visits: token_visited.len() + suffix_visited.len(),
    };
    Ok((
        TokenBoundedAnalysisTopology {
            configurations: Arc::from(configs),
            start_state: start_state as usize,
            transitions: Arc::from(transitions),
            raw_start_to_view: Arc::from(raw_start_to_view),
        },
        work,
    ))
}

fn build_bounded_analysis_view_inner(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    combine_start_states: bool,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        raw_start_states,
        tokens,
        combine_start_states,
        true,
        true,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

fn build_bounded_analysis_view_impl(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    combine_start_states: bool,
    factor_common_first_byte: bool,
    include_reset_suffixes: bool,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        raw_start_states,
        tokens,
        combine_start_states,
        factor_common_first_byte,
        include_reset_suffixes,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

pub(crate) fn build_bounded_analysis_view(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_view_inner(
        tokenizer,
        raw_start_states,
        tokens,
        active_groups,
        false,
    )
}

pub(crate) fn build_bounded_analysis_view_from_combined_starts(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_view_inner(
        tokenizer,
        raw_start_states,
        tokens,
        active_groups,
        true,
    )
}

pub(crate) fn build_token_bounded_analysis_topology(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
) -> TokenBoundedAnalysisTopology {
    build_bounded_analysis_topology_impl(
        tokenizer,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
}

pub(crate) fn build_token_bounded_analysis_view_projected(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        Some(active_groups),
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize_already_projected(tokenizer, active_groups)
}

pub(crate) fn try_build_token_bounded_analysis_view_projected(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let (topology, work) = build_bounded_analysis_topology_impl(
        tokenizer,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        Some(active_groups),
        Some(budget),
    )?;
    Ok((
        topology.materialize_already_projected(tokenizer, active_groups),
        work,
    ))
}

pub(crate) fn build_token_bounded_analysis_view(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_token_bounded_analysis_topology(tokenizer, raw_start_states, tokens)
        .materialize(tokenizer, active_groups)
}

pub(crate) fn build_token_bounded_analysis_view_from_combined_starts(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        raw_start_states,
        tokens,
        true,
        true,
        false,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
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

/// Refine raw scanner starts through a prebuilt sparse powerset topology.
/// When `raw_active_language` is supplied, inactive-only members are erased
/// from powerset class sets and edges into projected-empty configurations are
/// treated as the common implicit missing target. This permits a topology
/// built for a larger terminal mask to be reused after the mask shrinks.
pub(crate) fn compute_state_map_from_prebuilt_sparse_powerset(
    tokenizer: &Tokenizer,
    initial_state_map: Option<&ManyToOneIdMap>,
    depth: RefinementDepth,
    raw_start_to_view: &[u32],
    configurations: &[Box<[u32]>],
    output_class_by_config: &[u32],
    raw_active_language: Option<&[bool]>,
    edge_offsets: &[u32],
    edges: &[(u8, u32)],
) -> ManyToOneIdMap {
    let num_states = tokenizer.num_states() as usize;
    assert_eq!(raw_start_to_view.len(), num_states);
    assert!(output_class_by_config.len() >= configurations.len());
    assert!(edge_offsets.len() > configurations.len());
    if let Some(active_language) = raw_active_language {
        assert_eq!(active_language.len(), num_states);
    }
    let projected_empty = configurations
        .iter()
        .map(|config| {
            raw_active_language.is_some_and(|active| {
                config.iter().all(|&raw| !active[raw as usize])
            })
        })
        .collect::<Vec<_>>();

    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let num_candidates = candidate_representatives.len();
    let start_configs = candidate_representatives
        .iter()
        .map(|&state| raw_start_to_view[state])
        .collect::<Vec<_>>();
    let mut classes = start_configs
        .iter()
        .map(|&config| output_class_by_config[config as usize])
        .collect::<Vec<_>>();

    let round_limit = match depth {
        RefinementDepth::Stable => num_candidates,
        RefinementDepth::Bounded(rounds) => rounds,
    };
    let mut class_set_by_config = vec![0u32; configurations.len()];
    for _ in 0..round_limit {
        // Preserve singleton class-set IDs directly. Nearly every powerset
        // configuration in large epsilon lexers is a singleton closure, so
        // allocating, sorting, and hashing a SmallVec for each one is pure
        // overhead. Composite configurations can themselves collapse to a
        // singleton class set after refinement; map those to the same direct ID
        // so equality remains exactly identical to generic set interning.
        let singleton_id_limit = classes.iter().copied().max().map_or(0, |class| class + 1);
        let mut composite_class_set_ids =
            FxHashMap::<SmallVec<[u32; 4]>, u32>::default();
        for (config_id, config) in configurations.iter().enumerate() {
            if let [raw] = config.as_ref() {
                class_set_by_config[config_id] = classes[raw_to_candidate[*raw as usize]];
                continue;
            }

            let mut class_set = SmallVec::<[u32; 4]>::new();
            if let Some(active_language) = raw_active_language {
                class_set.extend(
                    config
                        .iter()
                        .filter(|&&raw| active_language[raw as usize])
                        .map(|&raw| classes[raw_to_candidate[raw as usize]]),
                );
            } else {
                class_set.extend(
                    config
                        .iter()
                        .map(|&raw| classes[raw_to_candidate[raw as usize]]),
                );
            }
            class_set.sort_unstable();
            class_set.dedup();
            class_set_by_config[config_id] = if let [class] = class_set.as_slice() {
                *class
            } else {
                let next = singleton_id_limit + composite_class_set_ids.len() as u32;
                *composite_class_set_ids.entry(class_set).or_insert(next)
            };
        }
        let mut zero_edge_signatures = FxHashMap::<u32, u32>::default();
        let mut one_edge_signatures = FxHashMap::<(u32, u8, u32), u32>::default();
        let mut larger_signatures = FxHashMap::<SmallVec<[u32; 8]>, u32>::default();
        let mut next_class = 0u32;
        let mut next_classes = vec![0u32; num_candidates];
        for candidate in 0..num_candidates {
            let config = start_configs[candidate] as usize;
            let edge_start = edge_offsets[config] as usize;
            let edge_end = edge_offsets[config + 1] as usize;
            let candidate_class = classes[candidate];
            next_classes[candidate] = if raw_active_language.is_none() {
                match &edges[edge_start..edge_end] {
                    [] => *zero_edge_signatures.entry(candidate_class).or_insert_with(|| {
                        let class = next_class;
                        next_class += 1;
                        class
                    }),
                    [(byte, target)] => {
                        let key = (
                            candidate_class,
                            *byte,
                            class_set_by_config[*target as usize],
                        );
                        *one_edge_signatures.entry(key).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                    direct_edges => {
                        let mut signature = SmallVec::<[u32; 8]>::new();
                        signature.push(candidate_class);
                        for &(byte, target) in direct_edges {
                            signature.push(byte as u32 + 1);
                            signature.push(class_set_by_config[target as usize] + 1);
                        }
                        *larger_signatures.entry(signature).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                }
            } else {
                let mut projected_edges = SmallVec::<[(u8, u32); 8]>::new();
                if !projected_empty[config] {
                    projected_edges.extend(
                        edges[edge_start..edge_end]
                            .iter()
                            .copied()
                            .filter(|&(_, target)| !projected_empty[target as usize]),
                    );
                }
                match projected_edges.as_slice() {
                    [] => *zero_edge_signatures.entry(candidate_class).or_insert_with(|| {
                        let class = next_class;
                        next_class += 1;
                        class
                    }),
                    [(byte, target)] => {
                        let key = (
                            candidate_class,
                            *byte,
                            class_set_by_config[*target as usize],
                        );
                        *one_edge_signatures.entry(key).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                    projected_edges => {
                        let mut signature = SmallVec::<[u32; 8]>::new();
                        signature.push(candidate_class);
                        for &(byte, target) in projected_edges {
                            signature.push(byte as u32 + 1);
                            signature.push(class_set_by_config[target as usize] + 1);
                        }
                        *larger_signatures.entry(signature).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                }
            };
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
    let singleton_closures = tokenizer.all_singleton_epsilon_closures();
    let raw_active_language = raw_active_language_states(tokenizer, active_groups);

    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let start_configs = candidate_representatives
        .iter()
        .map(|&state| {
            let config = project_raw_config(
                singleton_closures[state].to_vec(),
                raw_active_language.as_deref(),
            );
            intern_config(config, &mut config_ids, &mut configs)
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
    let mut target_marks = vec![0u32; num_states];
    let mut target_generation = 0u32;
    let mut target_config = Vec::<u32>::new();
    for candidate in 0..num_candidates {
        let source = configs[start_configs[candidate] as usize].to_vec();
        for (slot, &byte) in active_bytes.iter().enumerate() {
            target_generation = target_generation.wrapping_add(1);
            if target_generation == 0 {
                target_marks.fill(0);
                target_generation = 1;
            }
            target_config.clear();
            // Every source config is already epsilon-closed.  Consume the byte
            // from each member and union cached singleton closures of the raw
            // targets.  This is exactly `step_all` without re-closing the
            // source, allocating a per-call seen vector, or sorting an
            // intermediate direct-target list.
            for &source_state in &source {
                let raw_target = tokenizer.get_transition(source_state, byte);
                if raw_target == u32::MAX {
                    continue;
                }
                for &reachable in singleton_closures[raw_target as usize].iter() {
                    if raw_active_language
                        .as_deref()
                        .is_some_and(|active| !active[reachable as usize])
                    {
                        continue;
                    }
                    let mark = &mut target_marks[reachable as usize];
                    if *mark != target_generation {
                        *mark = target_generation;
                        target_config.push(reachable);
                    }
                }
            }
            if !target_config.is_empty() {
                target_config.sort_unstable();
                target_configs[candidate * active_bytes.len() + slot] =
                    intern_config(target_config.clone(), &mut config_ids, &mut configs);
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

    fn view_trace(
        view: &TokenizerView,
        start_state: usize,
        token: &[u8],
    ) -> Vec<(Vec<usize>, Vec<usize>, bool)> {
        let dfa = view.dfa();
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
    fn active_filtered_powerset_drops_inactive_members_before_refinement() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_partitioned_with_adaptive;

        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let view = build_relevant_powerset_view(
            &tokenizer,
            &[true; 256],
            Some(&[true, false, false]),
            None,
        );
        let start_config = &view.configurations[view.start_state];
        assert!(!start_config.is_empty());
        assert!(start_config.iter().all(|&state| {
            tokenizer.matched_terminals_iter(state).any(|terminal| terminal == 0)
                || tokenizer
                    .possible_future_terminals_iter(state)
                    .any(|terminal| terminal == 0)
        }));
        let edge_start = view.edge_offsets[view.start_state] as usize;
        let edge_end = view.edge_offsets[view.start_state + 1] as usize;
        let start_edges = &view.edges[edge_start..edge_end];
        assert!(!start_edges.iter().any(|&(byte, _)| byte == b'x' || byte == b'y'));
    }

    #[test]
    fn token_bounded_topology_materialization_drops_inactive_members() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_partitioned_with_adaptive;

        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"a".as_slice(),
            b"ab".as_slice(),
            b"x".as_slice(),
            b"xq".as_slice(),
            b"y".as_slice(),
            b"yrr".as_slice(),
        ];
        let topology =
            build_token_bounded_analysis_topology(&tokenizer, &raw_start_states, &tokens);
        let projected = topology.materialize(&tokenizer, Some(&[true, false, false]));
        let dfa = projected.tokenizer_view.dfa();

        assert_ne!(dfa.trans(dfa.start_state, b'a' as usize), u32::MAX);
        assert_eq!(dfa.trans(dfa.start_state, b'x' as usize), u32::MAX);
        assert_eq!(dfa.trans(dfa.start_state, b'y' as usize), u32::MAX);
    }

    #[test]
    fn already_projected_token_topology_materialization_matches_reprojection() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"a".as_slice(),
            b"aa".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"bb".as_slice(),
        ];
        let active_groups = [true, false];
        let topology = build_bounded_analysis_topology_impl(
            &tokenizer,
            &raw_start_states,
            &tokens,
            false,
            true,
            false,
            Some(&active_groups),
            None,
        )
        .expect("projected topology build")
        .0;
        let direct = topology.materialize_already_projected(&tokenizer, &active_groups);
        let reprojected = topology.materialize(&tokenizer, Some(&active_groups));

        for &raw_state in &raw_start_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let reprojected_start = reprojected.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&direct.tokenizer_view, direct_start, token),
                    view_trace(&reprojected.tokenizer_view, reprojected_start, token),
                );
            }
        }
    }

    #[test]
    fn token_bounded_projected_build_aborts_at_actual_work_budget() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"a".as_slice(),
            b"aa".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"bb".as_slice(),
        ];
        let active_groups = [true, true];
        let tiny_budget = TokenBoundedAnalysisWorkBudget {
            max_configurations: usize::MAX,
            max_trie_visits: 1,
        };

        let aborted = try_build_token_bounded_analysis_view_projected(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
            tiny_budget,
        );
        let work = match aborted {
            Ok(_) => panic!("tiny actual-work budget unexpectedly completed"),
            Err(work) => work,
        };
        assert_eq!(work.trie_visits, tiny_budget.max_trie_visits + 1);

        let generous_budget = TokenBoundedAnalysisWorkBudget {
            max_configurations: 1_000,
            max_trie_visits: 10_000,
        };
        let (budgeted, _) = try_build_token_bounded_analysis_view_projected(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
            generous_budget,
        )
        .unwrap_or_else(|work| panic!("generous budget aborted unexpectedly: {work:?}"));
        let reference = build_token_bounded_analysis_view_projected(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
        );

        for &raw_state in &raw_start_states {
            let budgeted_start = budgeted.view_state_for_raw_start(raw_state);
            let reference_start = reference.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&budgeted.tokenizer_view, budgeted_start, token),
                    view_trace(&reference.tokenizer_view, reference_start, token),
                );
            }
        }
    }

    #[test]
    fn bounded_nfa_common_first_factorization_preserves_observed_token_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [b"aa".as_slice(), b"ab".as_slice(), b"aab".as_slice()];
        let factored = build_bounded_analysis_view_impl(
            &tokenizer,
            &raw_start_states,
            &tokens,
            None,
            false,
            true,
            true,
        );
        let reference = build_bounded_analysis_view_impl(
            &tokenizer,
            &raw_start_states,
            &tokens,
            None,
            false,
            false,
            true,
        );

        for &raw_state in &raw_start_states {
            let factored_start = factored.view_state_for_raw_start(raw_state);
            let reference_start = reference.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&factored.tokenizer_view, factored_start, token),
                    view_trace(&reference.tokenizer_view, reference_start, token),
                );
            }
        }

        for token in tokens {
            for offset in 0..token.len() {
                assert_eq!(
                    view_trace(
                        &factored.tokenizer_view,
                        factored.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                    view_trace(
                        &reference.tokenizer_view,
                        reference.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                );
            }
        }
    }

    #[test]
    fn relevant_powerset_view_preserves_bounded_token_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"".as_slice(),
            b"a".as_slice(),
            b"b".as_slice(),
            b"aa".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"aab".as_slice(),
        ];
        let active_groups = [true, true];
        let bounded = build_bounded_analysis_view(
            &tokenizer,
            &raw_start_states,
            &tokens,
            Some(&active_groups),
        );
        let mut relevant_bytes = [false; 256];
        for token in tokens {
            for &byte in token {
                relevant_bytes[byte as usize] = true;
            }
        }
        let powerset = build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            None,
        );
        let raw_start_to_powerset = Arc::clone(&powerset.raw_start_to_view);
        let powerset = powerset.into_tokenizer_view();

        for &raw_state in &raw_start_states {
            let bounded_start = bounded.view_state_for_raw_start(raw_state);
            let powerset_start = raw_start_to_powerset[raw_state] as usize;
            for token in tokens {
                assert_eq!(
                    view_trace(&bounded.tokenizer_view, bounded_start, token),
                    view_trace(&powerset, powerset_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
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
    fn prebuilt_sparse_powerset_refinement_matches_fresh_nfa_refinement() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let direct = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            None,
            RefinementDepth::Stable,
        );
        let view = build_relevant_powerset_view(&tokenizer, &relevant, None, None);
        let mut output_ids = FxHashMap::<Vec<u64>, u32>::default();
        let output_class_by_config = view
            .configurations
            .iter()
            .map(|config| {
                let observation = observation_words(&tokenizer, config, None);
                let next = output_ids.len() as u32;
                *output_ids.entry(observation).or_insert(next)
            })
            .collect::<Vec<_>>();
        let reused = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            None,
            RefinementDepth::Stable,
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            None,
            &view.edge_offsets,
            &view.edges,
        );

        assert!(same_partition(
            &direct.original_to_internal,
            &reused.original_to_internal,
        ));
    }
}
