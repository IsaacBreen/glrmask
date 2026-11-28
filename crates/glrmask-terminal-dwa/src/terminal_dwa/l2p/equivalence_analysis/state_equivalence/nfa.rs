use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};
use crate::vocab::vocab_prefix_tree::VocabPrefixTree;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::observation::{ConfigInterner, EpsilonImageScratch};
use crate::automata::lexer::tokenizer::{SingletonEpsilonClosures, Tokenizer};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::super::compat::{ByteRows, FlatDfa, FlatDfaState, conservative_byte_classes};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawPowersetTargetMode {
    Scalar,
    Direct,
}

#[derive(Debug, Clone, Copy)]
pub enum RefinementDepth {
    Stable,
    Bounded(usize),
}

pub struct BoundedAnalysisView {
    pub tokenizer_view: FlatDfa,
    raw_start_to_view: Vec<u32>,
}

pub struct RelevantPowersetView {
    pub states: Vec<FlatDfaState>,
    pub start_state: usize,
    pub bytes: Vec<u8>,
    pub edge_offsets: Vec<u32>,
    pub edges: Vec<(u8, u32)>,
    pub raw_start_to_view: Arc<[u32]>,
    pub configurations: Arc<[Box<[u32]>]>,
}

#[derive(Clone, Copy, Debug)]
pub struct RelevantPowersetWorkBudget {
    pub max_configurations: usize,
    pub max_edges: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelevantPowersetWork {
    pub configurations: usize,
    pub edges: usize,
}

pub struct PrebuiltSparsePowersetRefinement<'a> {
    pub raw_start_to_view: &'a [u32],
    pub configurations: &'a [Box<[u32]>],
    pub output_class_by_config: &'a [u32],
    pub edge_offsets: &'a [u32],
    pub edges: &'a [(u8, u32)],
}

impl PrebuiltSparsePowersetRefinement<'_> {
    pub fn compute_state_map(
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

pub fn powerset_output_class_ids(view: &RelevantPowersetView) -> Vec<u32> {
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
    /// Return the powerset-view state corresponding to a raw lexer start.
    ///
    /// Unlike token-bounded analysis, a relevant-powerset view is total over
    /// the raw lexer-state domain. Keep that invariant behind a checked API so
    /// callers cannot accidentally cast the unmapped sentinel into a view ID.
    pub fn view_state_for_raw_start(&self, raw_state: usize) -> usize {
        let state = self
            .raw_start_to_view
            .get(raw_state)
            .copied()
            .expect("raw state is outside relevant-powerset domain");
        assert_ne!(
            state,
            u32::MAX,
            "relevant-powerset view must map every raw lexer state",
        );
        let state = state as usize;
        assert!(
            state < self.states.len(),
            "relevant-powerset raw-start map produced an invalid view state",
        );
        state
    }

    fn byte_rows(&self) -> ByteRows {
        let mut relevant = [false; 256];
        for &byte in &self.bytes {
            relevant[byte as usize] = true;
        }
        let classes = conservative_byte_classes(&relevant)
            .unwrap_or_else(|| std::array::from_fn(|byte| byte as u8));
        let width = classes.iter().copied().max().unwrap() as usize + 1;
        let mut transitions = vec![u32::MAX; self.states.len() * width];
        for state in 0..self.states.len() {
            let start = self.edge_offsets[state] as usize;
            let end = self.edge_offsets[state + 1] as usize;
            for &(byte, target) in &self.edges[start..end] {
                debug_assert!(relevant[byte as usize]);
                transitions[state * width + classes[byte as usize] as usize] = target;
            }
        }
        ByteRows::new(Arc::from(transitions), classes)
    }

    pub fn into_tokenizer_view(self) -> FlatDfa {
        let transitions = self.byte_rows();
        FlatDfa {
            states: self.states,
            start_state: self.start_state,
            transitions,
        }
    }
}

struct PowersetTrajectory<'a> {
    view: &'a RelevantPowersetView,
    rows: &'a ByteRows,
    selected: &'a mut [bool],
    observed_bytes: &'a mut [[u64; 4]],
}

impl PowersetTrajectory<'_> {
    fn record(&mut self, source: u32, byte: u8, target: u32) {
        self.observed_bytes[source as usize][byte as usize / 64] |= 1u64 << (byte as usize % 64);
        self.selected[target as usize] = true;
    }

    fn mark(&mut self, starts: &[u32], trie: &[ByteTrieNode]) {
        for &state in starts {
            self.selected[state as usize] = true;
        }
        traverse_frontiers(self, starts, PrefixWalk::Bytes(trie), None, 0)
            .expect("unbounded powerset trajectory");
    }
}

impl FrontierMachine for PowersetTrajectory<'_> {
    fn state_count(&self) -> usize { self.view.states.len() }
    fn sparse_root(&self) -> bool { true }

    fn step(&mut self, source: u32, byte: u8) -> u32 {
        let target = self.rows.trans(source as usize, byte as usize);
        if target != u32::MAX {
            self.record(source, byte, target);
        }
        target
    }

    fn route_root(
        &mut self, source: u32, _children: &[u8],
        slots: &[usize; 256], frontiers: &mut [Vec<u32>],
    ) {
        let start = self.view.edge_offsets[source as usize] as usize;
        let end = self.view.edge_offsets[source as usize + 1] as usize;
        for edge in start..end {
            let (byte, target) = self.view.edges[edge];
            let slot = slots[byte as usize];
            if slot != usize::MAX {
                self.record(source, byte, target);
                frontiers[slot].push(target);
            }
        }
    }
}

/// Restrict an exact relevant-byte powerset to precisely the whole-token and
/// reset-suffix trajectories needed by the bounded vocabulary analysis.
/// This preserves the bounded view's observation domain while reusing the
/// powerset configurations and transitions already constructed for the NFA
/// refinement prepass.
pub fn build_bounded_analysis_view_from_relevant_powerset(
    view: &RelevantPowersetView,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
) -> BoundedAnalysisView {
    let rows = view.byte_rows();
    let token_trie = build_byte_trie(tokens.iter().copied());
    let suffix_trie = build_byte_trie(
        tokens
            .iter()
            .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..])),
    );
    let mut selected = vec![false; view.states.len()];
    let mut observed_bytes = vec![[0u64; 4]; view.states.len()];
    let mut raw_seed_states = raw_start_states
        .iter()
        .map(|&raw| view.view_state_for_raw_start(raw) as u32)
        .collect::<Vec<_>>();
    raw_seed_states.sort_unstable();
    raw_seed_states.dedup();
    {
        let mut trajectory = PowersetTrajectory {
            view, rows: &rows, selected: &mut selected, observed_bytes: &mut observed_bytes,
        };
        trajectory.mark(&raw_seed_states, &token_trie);
        trajectory.mark(std::slice::from_ref(&(view.start_state as u32)), &suffix_trie);
    }
    selected[view.start_state] = true;

    let mut old_to_new = vec![u32::MAX; view.states.len()];
    let mut states = Vec::<FlatDfaState>::new();
    for (old, &is_selected) in selected.iter().enumerate() {
        if is_selected {
            old_to_new[old] = states.len() as u32;
            states.push(view.states[old].clone());
        }
    }
    let mut transitions = vec![u32::MAX; states.len().saturating_mul(rows.width)];
    for (old_source, &new_source) in old_to_new.iter().enumerate() {
        if new_source == u32::MAX {
            continue;
        }
        let edge_start = view.edge_offsets[old_source] as usize;
        let edge_end = view.edge_offsets[old_source + 1] as usize;
        for &(byte, old_target) in &view.edges[edge_start..edge_end] {
            let word = byte as usize / 64;
            if observed_bytes[old_source][word] & (1u64 << (byte as usize % 64)) == 0 {
                continue;
            }
            let new_target = old_to_new[old_target as usize];
            debug_assert_ne!(new_target, u32::MAX);
            transitions[new_source as usize * rows.width + rows.byte_to_column[byte as usize] as usize] = new_target;
        }
    }

    let mut raw_start_to_view = vec![u32::MAX; view.raw_start_to_view.len()];
    for &raw in raw_start_states {
        let old = view.view_state_for_raw_start(raw);
        let new = old_to_new[old];
        assert_ne!(new, u32::MAX, "raw analysis seed was not retained");
        raw_start_to_view[raw] = new;
    }
    let start_state = old_to_new[view.start_state];
    assert_ne!(start_state, u32::MAX);
    BoundedAnalysisView {
        tokenizer_view: FlatDfa {
            states,
            start_state: start_state as usize,
            transitions: ByteRows::new(Arc::from(transitions), rows.byte_to_column),
        },
        raw_start_to_view,
    }
}

pub fn build_relevant_powerset_analysis_view(
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

pub fn raw_active_language_states(
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

impl BoundedAnalysisView {
    pub fn view_state_for_raw_start(&self, raw_state: usize) -> usize {
        let state = self
            .raw_start_to_view
            .get(raw_state)
            .copied()
            .expect("raw state is outside bounded NFA analysis domain");
        assert_ne!(
            state,
            u32::MAX,
            "raw state was not seeded into bounded NFA analysis",
        );
        let state = state as usize;
        assert!(
            state < self.tokenizer_view.states.len(),
            "bounded raw-start map produced an invalid view state",
        );
        state
    }
}

pub fn build_relevant_powerset_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    state_map: Option<&ManyToOneIdMap>,
) -> RelevantPowersetView {
    try_build_relevant_powerset_view(
        tokenizer,
        relevant_bytes,
        active_groups,
        state_map,
        None,
        RawPowersetTargetMode::Direct,
    )
    .expect("unbounded relevant-powerset construction cannot abort")
}

pub fn build_relevant_powerset_view_budgeted(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    state_map: Option<&ManyToOneIdMap>,
    budget: RelevantPowersetWorkBudget,
) -> Result<RelevantPowersetView, RelevantPowersetWork> {
    try_build_relevant_powerset_view(
        tokenizer,
        relevant_bytes,
        active_groups,
        state_map,
        Some(budget),
        RawPowersetTargetMode::Direct,
    )
}

fn try_build_relevant_powerset_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    state_map: Option<&ManyToOneIdMap>,
    budget: Option<RelevantPowersetWorkBudget>,
    raw_target_mode: RawPowersetTargetMode,
) -> Result<RelevantPowersetView, RelevantPowersetWork> {
    #[inline]
    fn check_budget(
        configurations: usize,
        edges: usize,
        budget: Option<RelevantPowersetWorkBudget>,
    ) -> Result<(), RelevantPowersetWork> {
        if budget.is_some_and(|budget| {
            configurations > budget.max_configurations || edges > budget.max_edges
        }) {
            return Err(RelevantPowersetWork {
                configurations,
                edges,
            });
        }
        Ok(())
    }

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

    // All singleton starts are seeded before expansion. Every new configuration
    // is appended in discovery order, so the interner itself is the queue.
    let mut configs = ConfigInterner::default();
    let class_to_view = state_map.map(|map| {
        (0..map.representative_original_ids.len() as u32)
            .map(|class| {
                if class_active_language.as_deref().is_none_or(|a| a[class as usize]) {
                    configs.intern_slice(&[class])
                } else {
                    configs.intern_slice(&[])
                }
            })
            .collect::<Vec<_>>()
    });
    let raw_start_to_view = if let (Some(map), Some(class_to_view)) = (state_map, &class_to_view) {
        map.original_to_internal.iter().map(|&c| class_to_view[c as usize]).collect::<Vec<_>>()
    } else {
        (0..raw_state_count)
            .map(|state| {
                let closure = tokenizer.execute_from_state_end_only(&[], state as u32);
                configs.intern(project_raw_config(closure.to_vec(), raw_active_language.as_deref()))
            })
            .collect::<Vec<_>>()
    };
    check_budget(configs.len(), 0, budget)?;

    let start_state = raw_start_to_view[tokenizer.initial_state_id() as usize] as usize;
    let bytes = relevant_bytes.iter().enumerate()
        .filter_map(|(byte, &relevant)| relevant.then_some(byte as u8))
        .collect::<Vec<_>>();
    let closure_by_class = state_map.map(|map| {
        map.representative_original_ids.iter()
            .map(|&r| tokenizer.execute_from_state_end_only(&[], r))
            .collect::<Vec<_>>()
    });

    // A mapped singleton is a representative's epsilon image. Once its
    // sparse row exists, all larger mapped subsets are unions of those
    // already-built rows. No separate class x byte table is necessary.
    // Raw closed subsets instead union the preseeded raw target closures.
    // Large raw domains keep the low-memory two-pass scan.
    let one_pass = state_map.is_some()
        || (raw_target_mode == RawPowersetTargetMode::Direct && raw_state_count <= 65_536);
    let mut buckets = one_pass.then(|| std::array::from_fn::<_, 256, _>(|_| Vec::<u32>::new()));
    let mut candidate_bytes = Vec::<u8>::new();
    let mut byte_seen = [false; 256];
    let target_domain = state_map.map_or(raw_state_count, |map| map.representative_original_ids.len());
    let mut target_marks = (raw_target_mode == RawPowersetTargetMode::Direct || state_map.is_some())
        .then(|| vec![0u32; target_domain]);
    let mut target_epoch = 0u32;
    let mut target_scratch = Vec::<u32>::new();
    let mut edge_offsets = Vec::<u32>::with_capacity(configs.len() + 1);
    let mut edges = Vec::<(u8, u32)>::new();
    edge_offsets.push(0);
    while edge_offsets.len() <= configs.len() {
        let state = edge_offsets.len() - 1;
        // Singleton raw states have exactly the original sparse byte row;
        // direct targets already have a canonical projected closure ID.
        if state_map.is_none() && let [source] = configs[state].as_ref() {
            let source = *source;
            for (byte, raw_target) in tokenizer.transitions_from(source) {
                if !relevant_bytes[byte as usize] { continue; }
                let target = raw_start_to_view[raw_target as usize];
                if configs[target as usize].is_empty() { continue; }
                edges.push((byte, target));
                check_budget(configs.len(), edges.len(), budget)?;
            }
            edge_offsets.push(edges.len() as u32);
            continue;
        }

        candidate_bytes.clear();
        byte_seen.fill(false);
        let mapped_singleton = state_map.is_some() && configs[state].len() == 1;
        if let (Some(map), Some(closures), Some(singletons)) =
            (state_map, &closure_by_class, &class_to_view)
        {
            let buckets = buckets.as_mut().expect("mapped image scratch");
            if let [class] = configs[state].as_ref() {
                let closure = &closures[*class as usize];
                // The common singleton-closure case can publish each edge
                // directly, without constructing any per-byte buckets.
                if let [source] = closure.as_slice() {
                    for (byte, raw_target) in tokenizer.transitions_from(*source) {
                        if !relevant_bytes[byte as usize] { continue; }
                        let raw_targets = tokenizer.execute_from_state_end_only(&[], raw_target);
                        target_scratch.clear();
                        target_scratch.extend(raw_targets.iter()
                            .map(|&raw| map.original_to_internal[raw as usize])
                            .filter(|&c| class_active_language.as_deref().is_none_or(|a| a[c as usize])));
                        target_scratch.sort_unstable();
                        target_scratch.dedup();
                        let target = match target_scratch.as_slice() {
                            [] => continue,
                            [class] => singletons[*class as usize],
                            classes => configs.intern_slice(classes),
                        };
                        edges.push((byte, target));
                        check_budget(configs.len(), edges.len(), budget)?;
                    }
                    edge_offsets.push(edges.len() as u32);
                    continue;
                }
                for &source in closure {
                    for (byte, raw_target) in tokenizer.transitions_from(source) {
                        let slot = byte as usize;
                        if !relevant_bytes[slot] { continue; }
                        if !byte_seen[slot] {
                            byte_seen[slot] = true;
                            candidate_bytes.push(byte);
                        }
                        buckets[slot].push(raw_target);
                    }
                }
            } else {
                for &class in configs[state].iter() {
                    let source = singletons[class as usize] as usize;
                    // All singleton class IDs were interned before the first
                    // non-singleton target could be discovered.
                    debug_assert!(source < state);
                    for &(byte, target) in &edges[edge_offsets[source] as usize..edge_offsets[source+1] as usize] {
                        let slot = byte as usize;
                        if !byte_seen[slot] {
                            byte_seen[slot] = true;
                            candidate_bytes.push(byte);
                        }
                        buckets[slot].push(target);
                    }
                }
            }
        } else {
            for &source in configs[state].iter() {
                for (byte, raw_target) in tokenizer.transitions_from(source) {
                    let slot = byte as usize;
                    if !relevant_bytes[slot] { continue; }
                    if !byte_seen[slot] {
                        byte_seen[slot] = true;
                        candidate_bytes.push(byte);
                    }
                    if let Some(buckets) = &mut buckets {
                        buckets[slot].push(raw_start_to_view[raw_target as usize]);
                    }
                }
            }
        }

        candidate_bytes.sort_unstable();
        for &byte in &candidate_bytes {
            if let Some(buckets) = &mut buckets {
                let bucket = &mut buckets[byte as usize];
                bucket.sort_unstable();
                bucket.dedup();
                if !mapped_singleton && let [target] = bucket.as_slice() {
                    let target = *target;
                    if !configs[target as usize].is_empty() {
                        edges.push((byte, target));
                        check_budget(configs.len(), edges.len(), budget)?;
                    }
                    bucket.clear();
                    continue;
                }
            }
            let projected = if let Some(marks) = &mut target_marks {
                target_epoch = target_epoch.wrapping_add(1);
                if target_epoch == 0 {
                    marks.fill(0);
                    target_epoch = 1;
                }
                target_scratch.clear();
                let mut insert = |target: u32| {
                    let mark = &mut marks[target as usize];
                    if *mark != target_epoch {
                        *mark = target_epoch;
                        target_scratch.push(target);
                    }
                };
                if let Some(buckets) = &mut buckets {
                    let bucket = &mut buckets[byte as usize];
                    if mapped_singleton {
                        let map = state_map.expect("mapped singleton coordinate");
                        for &raw_target in bucket.iter() {
                            let closure = tokenizer.execute_from_state_end_only(&[], raw_target);
                            for &raw in &closure {
                                let class = map.original_to_internal[raw as usize];
                                if class_active_language.as_deref().is_none_or(|a| a[class as usize]) {
                                    insert(class);
                                }
                            }
                        }
                    } else {
                        for &view in bucket.iter() {
                            for &target in configs[view as usize].iter() {
                                insert(target);
                            }
                        }
                    }
                    bucket.clear();
                } else {
                    for &source in configs[state].iter() {
                        let Some(raw_target) = tokenizer.step(source, byte) else { continue; };
                        for &target in configs[raw_start_to_view[raw_target as usize] as usize].iter() {
                            insert(target);
                        }
                    }
                }
                target_scratch.sort_unstable();
                target_scratch.as_slice()
            } else {
                // Independent scalar closure oracle.
                target_scratch = project_raw_config(
                    tokenizer.step_all(&configs[state], byte).to_vec(),
                    raw_active_language.as_deref(),
                );
                target_scratch.as_slice()
            };
            if !projected.is_empty() {
                let target = configs.intern_slice(projected);
                edges.push((byte, target));
                check_budget(configs.len(), edges.len(), budget)?;
            }
        }
        edge_offsets.push(edges.len() as u32);
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
    Ok(RelevantPowersetView {
        states,
        start_state,
        bytes,
        edge_offsets,
        edges,
        raw_start_to_view: Arc::from(raw_start_to_view),
        configurations: Arc::from(configs.into_configs()),
    })
}

#[derive(Debug, Default, PartialEq, Eq)]
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

/// A virtual position in either the transient byte trie (reset suffixes and
/// subset walks) or the shared lexical radix. Inside a compressed edge, a
/// position advances one byte at a time: no lexical event may be skipped.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PrefixPosition {
    node: usize,
    depth: usize,
}

#[derive(Clone, Copy)]
enum PrefixWalk<'a> {
    Bytes(&'a [ByteTrieNode]),
    Radix(&'a VocabPrefixTree),
}

impl PrefixWalk<'_> {
    fn root(self) -> PrefixPosition { PrefixPosition { node: 0, depth: 0 } }

    fn child_at(self, position: PrefixPosition, slot: usize) -> Option<(u8, PrefixPosition)> {
        match self {
            Self::Bytes(nodes) => nodes[position.node].children.get(slot)
                .map(|&(byte, node)| (byte, PrefixPosition { node, depth: 0 })),
            Self::Radix(tree) => {
                let node = tree.node(position.node as u32);
                if position.depth < node.prefix_length() {
                    return (slot == 0).then(|| (node.prefix()[position.depth],
                        PrefixPosition { node: position.node, depth: position.depth + 1 }));
                }
                let child = tree.node(*node.children().get(slot)?);
                Some((child.prefix()[position.depth],
                    PrefixPosition { node: child.index() as usize, depth: position.depth + 1 }))
            }
        }
    }

    fn mark_relevant(self, relevant: &mut [bool; 256]) {
        match self {
            Self::Bytes(nodes) => {
                for node in nodes {
                    for &(byte, _) in &node.children { relevant[byte as usize] = true; }
                }
            }
            Self::Radix(tree) => {
                let words = tree.root().subtree_bytes();
                for byte in 0..256 {
                    relevant[byte] |= words[byte / 64] & (1u64 << (byte % 64)) != 0;
                }
            }
        }
    }
}

#[derive(Debug)]
struct PrefixEventDag {
    children: Vec<Vec<(u8, usize)>>,
    root: usize,
}

/// A vocabulary-pinned prefix walk and its lazily shared suffix quotient.
/// The radix owns spelling and child topology; expanded positions are not
/// retained. The suffix quotient is purely a vocabulary observation, so it
/// is reusable across lexer event predicates and active terminal scopes.
#[derive(Debug)]
pub struct TokenBoundedAnalysisTrie {
    tree: Arc<VocabPrefixTree>,
    byte_positions: usize,
    event_dag: OnceLock<PrefixEventDag>,
}

impl TokenBoundedAnalysisTrie {
    pub fn from_radix(tree: Arc<VocabPrefixTree>) -> Self {
        let mut byte_positions = 1usize;
        for index in 0..tree.len() {
            for (edge, _) in tree.node(index as u32).iter_children() {
                byte_positions = byte_positions.saturating_add(edge.len());
            }
        }
        Self { tree, byte_positions, event_dag: OnceLock::new() }
    }

    pub fn len(&self) -> usize {
        self.byte_positions
    }

    pub fn is_reasonable_superset_for(&self, token_count: usize) -> bool {
        const MAX_NODE_RATIO: usize = 8;
        self.len() <= token_count.saturating_mul(MAX_NODE_RATIO).max(256)
    }

    fn event_dag(&self) -> &PrefixEventDag {
        self.event_dag.get_or_init(|| {
            let mut classes = vec![0usize; self.tree.len()];
            let mut dag = Vec::<Vec<(u8, usize)>>::new();
            let mut intern = FxHashMap::<Vec<(u8, usize)>, usize>::default();
            let mut intern_row = |signature: Vec<(u8, usize)>| {
                if let Some(&id) = intern.get(&signature) { return id; }
                let id = dag.len();
                dag.push(signature.clone());
                intern.insert(signature, id);
                id
            };
            for index in (0..self.tree.len()).rev() {
                let mut signature = Vec::new();
                for (edge, child) in self.tree.node(index as u32).iter_children() {
                    let mut target = classes[child.index() as usize];
                    for &byte in edge[1..].iter().rev() {
                        target = intern_row(vec![(byte, target)]);
                    }
                    signature.push((edge[0], target));
                }
                // Radix children are in byte order, exactly the canonical
                // signature order of the uncompressed suffix quotient.
                classes[index] = intern_row(signature);
            }
            PrefixEventDag { children: dag, root: classes[0] }
        })
    }

    /// Raw states from which an event can be reached on a vocabulary prefix.
    ///
    /// All trie positions (including the root and token endpoints) may witness
    /// an event. The result is exact for this query; a caller may use a broad
    /// event predicate to obtain a conservative semantic support filter.
    /// `initial_events` optionally substitutes the event predicate only at the
    /// zero-byte root; every positive prefix still uses `events`.
    /// `epsilon_closures` supplies complete reflexive/transitive closures for
    /// nontrivial epsilon sources; omitted sources have the singleton closure.
    ///
    /// For trie node p, E_p = eps_pre(events OR union_b pre_b(E_child(p,b))).
    /// Induction on remaining trie height proves the recurrence. Nodes with
    /// identical outgoing byte/child signatures have identical E_p and can
    /// share one row. This quotients the *observation trie*, never lexer states,
    /// terminal labels, parser coordinates, or token identities.
    ///
    /// A bounded analysis may decline without changing the caller's domain.
    /// Unknown/virtual transitions also decline rather than become dead edges.
    pub fn prefix_event_sources(
        &self,
        flat_trans: &[u32],
        events: &[bool],
        epsilon_closures: &[(usize, Box<[u32]>)],
        initial_events: Option<&[bool]>,
    ) -> Option<(Vec<bool>, usize)> {
        let states = events.len();
        if flat_trans.len() != states.checked_mul(256)?
            || initial_events.is_some_and(|events| events.len() != states)
        {
            return None;
        }
        for (source, closure) in epsilon_closures {
            if *source >= states || closure.iter().any(|&q| q as usize >= states) {
                return None;
            }
        }

        let event_dag = self.event_dag();
        let dag = &event_dag.children;
        let edges = dag.iter().map(Vec::len).sum::<usize>()
            + initial_events.map_or(0, |_| dag[event_dag.root].len());
        let epsilon_volume = epsilon_closures.iter().map(|(_, closure)| closure.len()).sum::<usize>();
        let words = states.div_ceil(64);
        if states.checked_mul(edges)?.checked_add(epsilon_volume.checked_mul(dag.len())?)? > 120_000_000
            || words.checked_mul(dag.len())?.checked_mul(16)? > 64 * 1024 * 1024
        {
            return None;
        }
        let mut used_bytes = [false; 256];
        for &(byte, _) in dag.iter().flatten() { used_bytes[byte as usize] = true; }
        let byte_count = used_bytes.iter().filter(|&&used| used).count();
        if states.checked_mul(byte_count)?.checked_mul(4)? > 64 * 1024 * 1024 {
            return None;
        }
        let used_bytes=used_bytes.iter().enumerate().filter_map(|(b,&yes)|yes.then_some(b)).collect::<Vec<_>>();
        let mut columns=vec![Vec::<(u32,u32)>::new();256];
        let mut stored_pairs=0usize;
        // Scan each contiguous raw transition row once, not one strided full
        // lexer pass for every byte. This changes storage order only; each
        // byte column remains sorted by original source-state identity.
        for (source,row) in flat_trans.as_chunks::<256>().0.iter().enumerate(){
            for &byte in &used_bytes{
                let target=row[byte];
                if target==u32::MAX{continue;}
                if target as usize>=states{return None;}
                stored_pairs=stored_pairs.checked_add(1)?;
                if stored_pairs>8_000_000{return None;}
                columns[byte].push((source as u32,target));
            }
        }
        let mut event_words = vec![0u64; words];
        for (state, &event) in events.iter().enumerate() {
            if event { event_words[state / 64] |= 1u64 << (state % 64); }
        }
        let mut rows = Vec::<Vec<u64>>::with_capacity(dag.len());
        let mut row_ids=Vec::<usize>::with_capacity(dag.len());
        let mut interned_rows=FxHashMap::<Vec<u64>,usize>::default();
        let mut preimages=FxHashMap::<(u8,usize),Vec<(usize,u64)>>::default();
        let mut cache_words=0usize;
        for children in dag {
            let mut live = event_words.clone();
            for &(byte, child) in children {
                let target_live = &rows[child];
                let key=(byte,row_ids[child]);
                if let Some(cached)=preimages.get(&key){
                    for &(word,bits) in cached{live[word]|=bits;}
                }else{
                    let mut observed=vec![0u64;words];
                    for &(source,target)in &columns[byte as usize]{
                        if target_live[target as usize/64]&(1u64<<(target%64))!=0{
                            observed[source as usize/64]|=1u64<<(source%64);
                        }
                    }
                    let nonzero=observed.into_iter().enumerate().filter(|(_,bits)|*bits!=0).collect::<Vec<_>>();
                    for &(word,bits)in &nonzero{live[word]|=bits;}
                    // Stop caching rather than approximating if the auxiliary
                    // index reaches its memory budget. This is only memoization.
                    if cache_words+nonzero.len()<=2_000_000 && preimages.len()<65_536{
                        cache_words+=nonzero.len();preimages.insert(key,nonzero);
                    }
                }
            }
            // Full closures make this order-independent, including cycles:
            // any newly observed member's closure is already contained in the
            // current source's transitive closure.
            for (source, closure) in epsilon_closures {
                if closure.iter().any(|&q| live[q as usize / 64] & (1u64 << (q % 64)) != 0) {
                    live[source / 64] |= 1u64 << (source % 64);
                }
            }
            let next_id=interned_rows.len();
            let id=*interned_rows.entry(live.clone()).or_insert(next_id);
            row_ids.push(id);
            rows.push(live);
        }
        // Some observers distinguish the token's zero-byte entry from later
        // positions (e.g. terminal completion requires consuming a byte, while
        // an already-visible foreign future can matter at entry). This changes
        // only the root event set; every child still uses ordinary events.
        let initial_row;
        let root = if let Some(initial_events) = initial_events {
            let mut live = vec![0u64; words];
            for (q, &event) in initial_events.iter().enumerate() {
                if event { live[q / 64] |= 1u64 << (q % 64); }
            }
            for &(byte, child) in &dag[event_dag.root] {
                for &(source, target) in &columns[byte as usize] {
                    let source=source as usize;
                    if rows[child][target as usize / 64] & (1u64 << (target % 64)) != 0
                    { live[source / 64] |= 1u64 << (source % 64); }
                }
            }
            for (source, closure) in epsilon_closures {
                if closure.iter().any(|&q| live[q as usize / 64] & (1u64 << (q % 64)) != 0) {
                    live[source / 64] |= 1u64 << (source % 64);
                }
            }
            initial_row = live;
            &initial_row
        } else { &rows[event_dag.root] };
        Some(((0..states).map(|q| root[q / 64] & (1u64 << (q % 64)) != 0).collect(), dag.len()))
    }
}

#[cfg(test)]
#[test]
fn prefix_event_sources_match_independent_forward_execution() {
    // Generated byte DFAs with nondeterministic epsilon closures, cycles,
    // duplicate spellings, empty tokens, and event-free graphs. Compare the
    // backward DAG recurrence with explicit forward execution of every token
    // from every raw source (no trie or quotient in that oracle).
    let mut rng = 7u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as usize
    };
    for case in 0..512 {
        let n = 2 + next() % 13;
        let mut flat = vec![u32::MAX; n * 256];
        let mut eps = vec![Vec::<usize>::new(); n];
        for q in 0..n {
            for b in 0..3 {
                let target = next() % (n + 2);
                if target < n { flat[q * 256 + b] = target as u32; }
            }
            if next() % 3 == 0 { eps[q].push(next() % n); }
        }
        let mut closures = Vec::new();
        let mut full_closures = Vec::new();
        for q in 0..n {
            let mut seen = vec![false; n];
            let mut pending = vec![q];
            while let Some(s) = pending.pop() {
                if seen[s] { continue; }
                seen[s] = true;
                pending.extend(eps[s].iter().copied());
            }
            let closure = (0..n).filter(|&s| seen[s]).map(|s| s as u32).collect::<Vec<_>>();
            if closure.len() > 1 { closures.push((q, closure.clone().into_boxed_slice())); }
            full_closures.push(closure);
        }
        let events = (0..n).map(|_| next() % 4 == 0).collect::<Vec<_>>();
        let mut tokens = (0..12).map(|_| {
            let length = next() % 7;
            (0..length).map(|_| (next() % 3) as u8).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        tokens.sort();
        let borrowed = tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let trie = build_token_bounded_analysis_trie_sorted(&borrowed);
        let initial_events = (0..n).map(|_| next() % 5 == 0).collect::<Vec<_>>();
        for initial in [None, Some(initial_events.as_slice())] {
        let (actual, _) = trie.prefix_event_sources(&flat, &events, &closures, initial).unwrap();
        let expected = (0..n).map(|source| tokens.iter().any(|token| {
            let mut current = full_closures[source].clone();
            if current.iter().any(|&q| initial.unwrap_or(&events)[q as usize]) { return true; }
            for &byte in token {
                let mut target = Vec::new();
                for &q in &current {
                    let q = flat[q as usize * 256 + byte as usize];
                    if q != u32::MAX { target.extend(full_closures[q as usize].iter().copied()); }
                }
                target.sort_unstable(); target.dedup(); current = target;
                if current.iter().any(|&q| events[q as usize]) { return true; }
            }
            false
        })).collect::<Vec<_>>();
        assert_eq!(actual, expected, "generated prefix event case {case}");
        }
    }
}

#[cfg(test)]
#[test]
fn prefix_event_sources_declines_unknown_targets_and_bad_domains() {
    let trie = build_token_bounded_analysis_trie_sorted(&[b"ab"]);
    assert!(trie.prefix_event_sources(&[], &[false], &[], None).is_none());
    let mut flat = vec![u32::MAX; 256];
    flat[b'a' as usize] = 3;
    assert!(trie.prefix_event_sources(&flat, &[false], &[], None).is_none());
    flat[b'a' as usize] = 0;
    assert!(trie.prefix_event_sources(&flat, &[false], &[(0, vec![1].into_boxed_slice())], None).is_none());
    assert_eq!(trie.prefix_event_sources(&flat, &[false], &[], None).unwrap().0, vec![false]);
    assert_eq!(trie.prefix_event_sources(&flat, &[true], &[], None).unwrap().0, vec![true]);
    flat[b'a' as usize] = u32::MAX;
    assert_eq!(trie.prefix_event_sources(&flat, &[true], &[], Some(&[false])).unwrap().0, vec![false]);
    assert!(trie.prefix_event_sources(&flat, &[true], &[], Some(&[])).is_none());
}

pub fn build_token_bounded_analysis_trie_sorted(
    sequences: &[&[u8]],
) -> TokenBoundedAnalysisTrie {
    debug_assert!(sequences.windows(2).all(|pair| pair[0] <= pair[1]));
    let mut words = sequences.to_vec();
    words.dedup();
    TokenBoundedAnalysisTrie::from_radix(Arc::new(
        VocabPrefixTree::from_ordered_tokens(Arc::new(
            crate::vocab::TokenBytes::pack(words)
        ))
    ))
}

fn build_byte_trie_sorted(sequences: &[&[u8]]) -> Vec<ByteTrieNode> {
    debug_assert!(sequences.windows(2).all(|pair| pair[0] <= pair[1]));

    let mut nodes = vec![ByteTrieNode::default()];
    let mut path = vec![0usize];
    let mut previous: &[u8] = &[];
    for &sequence in sequences {
        let lcp = previous
            .iter()
            .zip(sequence.iter())
            .take_while(|(left, right)| left == right)
            .count();
        path.truncate(lcp + 1);
        let mut node = *path.last().expect("root path must remain present");
        for &byte in &sequence[lcp..] {
            let child = nodes.len();
            nodes.push(ByteTrieNode::default());
            nodes[node].children.push((byte, child));
            path.push(child);
            node = child;
        }
        previous = sequence;
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
pub struct TokenBoundedAnalysisWork {
    pub configurations: usize,
    pub trie_visits: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TokenBoundedAnalysisWorkBudget {
    pub max_configurations: usize,
    pub max_trie_visits: usize,
}

fn ensure_config_transition(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    state: u32,
    byte: u8,
    configs: &mut ConfigInterner,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    byte_to_column: &[u8; 256],
    width: usize,
    preseeded_raw_closures: Option<&[u32]>,
    singleton_closures: &SingletonEpsilonClosures,
    active_language: Option<&[bool]>,
    image: &mut EpsilonImageScratch,
) -> u32 {
    let slot = state as usize * width + byte_to_column[byte as usize] as usize;
    if known_transitions[slot] != 0 {
        return transitions[slot];
    }
    known_transitions[slot] = 1;
    if let Some(raw_start_to_view) = preseeded_raw_closures
        && let [source] = configs[state as usize].as_ref()
    {
        let raw_target = raw_transitions.map_or_else(
            || tokenizer.get_transition(*source, byte),
            |transitions| transitions[*source as usize * 256 + byte as usize],
        );
        if raw_target == u32::MAX {
            return u32::MAX;
        }
        let target = raw_start_to_view[raw_target as usize];
        if target != u32::MAX {
            transitions[slot] = target;
            return target;
        }
    }
    let targets = image.step(
        &configs[state as usize], byte, true, singleton_closures,
        active_language, |source, byte| raw_transitions.map_or_else(
            || tokenizer.get_transition(source, byte),
            |transitions| transitions[source as usize * 256 + byte as usize],
        ),
    );
    if targets.is_empty() {
        return u32::MAX;
    }
    let target = configs.intern_slice(targets);
    if transitions.len() < configs.len() * width {
        transitions.resize(configs.len() * width, u32::MAX);
        known_transitions.resize(configs.len() * width, 0);
    }
    transitions[slot] = target;
    target
}

/// Mutable transition interning state shared by prefix and reset-suffix walks.
/// A cached transition describes an exact epsilon-closed configuration step,
/// independently of the trie position that first requested it.
struct BoundedExpansion<'a> {
    tokenizer: &'a Tokenizer,
    raw_transitions: Option<&'a [u32]>,
    configs: &'a mut ConfigInterner,
    transitions: &'a mut Vec<u32>,
    known_transitions: &'a mut Vec<u8>,
    byte_to_column: &'a [u8; 256],
    width: usize,
    singleton_closures: &'a SingletonEpsilonClosures,
    active_language: Option<&'a [bool]>,
    image: &'a mut EpsilonImageScratch,
}

/// The exact image recurrence on a byte trie is independent of how its
/// transition relation is obtained: a closed prebuilt powerset or lazy
/// configuration interning. Only the current root-to-leaf frontier is live.
trait FrontierMachine {
    fn state_count(&self) -> usize;
    fn step(&mut self, state: u32, byte: u8) -> u32;
    fn sparse_root(&self) -> bool;
    fn route_root(
        &mut self, state: u32, children: &[u8],
        slots: &[usize; 256], frontiers: &mut [Vec<u32>],
    );
}

struct FrontierFrame {
    node: PrefixPosition,
    next_child: usize,
    states: Vec<u32>,
}

fn charge_frontier(
    configurations: usize,
    visits: &mut usize,
    count: usize,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
) -> Result<(), TokenBoundedAnalysisWork> {
    if let Some(budget) = budget {
        if visits.saturating_add(count) > budget.max_trie_visits {
            return Err(TokenBoundedAnalysisWork {
                configurations,
                trie_visits: budget.max_trie_visits.saturating_add(1),
            });
        }
        if configurations > budget.max_configurations {
            return Err(TokenBoundedAnalysisWork { configurations, trie_visits: *visits });
        }
    }
    *visits = visits.saturating_add(count);
    Ok(())
}

struct FrontierScratch {
    visits: usize,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
    marks: Vec<u32>,
    generation: u32,
    spare: Vec<Vec<u32>>,
}

impl FrontierScratch {
    fn walk(
        &mut self, machine: &mut impl FrontierMachine,
        node: PrefixPosition, states: Vec<u32>, trie: PrefixWalk<'_>,
    ) -> Result<(), TokenBoundedAnalysisWork> {
        if states.is_empty() {
            self.spare.push(states);
            return Ok(());
        }
        charge_frontier(machine.state_count(), &mut self.visits, states.len(), self.budget)?;
        let mut stack = vec![FrontierFrame { node, next_child: 0, states }];
        while let Some(frame) = stack.last_mut() {
            let Some((byte, child)) = trie.child_at(frame.node, frame.next_child) else {
                let mut finished = stack.pop().expect("current frontier frame").states;
                finished.clear();
                self.spare.push(finished);
                continue;
            };
            frame.next_child += 1;
            self.generation = self.generation.wrapping_add(1);
            if self.generation == 0 {
                self.marks.fill(0);
                self.generation = 1;
            }
            let mut targets = self.spare.pop().unwrap_or_default();
            targets.clear();
            for &state in &frame.states {
                let target = machine.step(state, byte);
                if target == u32::MAX {
                    continue;
                }
                if self.marks.len() < machine.state_count() {
                    self.marks.resize(machine.state_count(), 0);
                }
                if self.marks[target as usize] != self.generation {
                    self.marks[target as usize] = self.generation;
                    targets.push(target);
                }
            }
            charge_frontier(machine.state_count(), &mut self.visits, targets.len(), self.budget)?;
            if targets.is_empty() {
                self.spare.push(targets);
            } else {
                stack.push(FrontierFrame { node: child, next_child: 0, states: targets });
            }
        }
        Ok(())
    }
}

fn traverse_frontiers(
    machine: &mut impl FrontierMachine, starts: &[u32], trie: PrefixWalk<'_>,
    budget: Option<TokenBoundedAnalysisWorkBudget>, prior_visits: usize,
) -> Result<usize, TokenBoundedAnalysisWork> {
    if starts.is_empty() {
        return Ok(0);
    }
    let mut scratch = FrontierScratch {
        visits: prior_visits, budget, marks: vec![0; machine.state_count()],
        generation: 0, spare: Vec::new(),
    };
    let root = trie.root();
    if machine.sparse_root() && trie.child_at(root, 1).is_some() {
        // Route broad sparse roots once per source, retaining byte positions
        // in a compressed edge rather than expanding its topology.
        charge_frontier(machine.state_count(), &mut scratch.visits, starts.len(), budget)?;
        let children = (0..256).map_while(|slot| trie.child_at(root, slot)).collect::<Vec<_>>();
        let bytes = children.iter().map(|&(byte, _)| byte).collect::<Vec<_>>();
        let mut slots = [usize::MAX; 256];
        for (slot, &byte) in bytes.iter().enumerate() {
            slots[byte as usize] = slot;
        }
        let mut frontiers = vec![Vec::new(); children.len()];
        for &state in starts {
            machine.route_root(state, &bytes, &slots, &mut frontiers);
        }
        for ((_, child), mut frontier) in children.into_iter().zip(frontiers) {
            frontier.sort_unstable();
            frontier.dedup();
            scratch.walk(machine, child, frontier, trie)?;
        }
    } else {
        scratch.walk(machine, root, starts.to_vec(), trie)?;
    }
    charge_frontier(machine.state_count(), &mut scratch.visits, 0, budget)?;
    Ok(scratch.visits - prior_visits)
}

struct LazyTrajectory<'a, 'b> {
    expansion: &'a mut BoundedExpansion<'b>,
    preseeded: Option<&'a [u32]>,
}

impl FrontierMachine for LazyTrajectory<'_, '_> {
    fn state_count(&self) -> usize { self.expansion.configs.len() }
    fn sparse_root(&self) -> bool { self.preseeded.is_some() }

    fn step(&mut self, state: u32, byte: u8) -> u32 {
        self.expansion.step(state, byte, self.preseeded)
    }

    fn route_root(
        &mut self, state: u32, children: &[u8],
        slots: &[usize; 256], frontiers: &mut [Vec<u32>],
    ) {
        let tokenizer = self.expansion.tokenizer;
        if let [source] = self.expansion.configs[state as usize].as_ref() {
            let source = *source;
            for (byte, _) in tokenizer.transitions_from(source) {
                let slot = slots[byte as usize];
                if slot != usize::MAX {
                    let target = self.step(state, byte);
                    if target != u32::MAX {
                        frontiers[slot].push(target);
                    }
                }
            }
        } else {
            for (slot, &byte) in children.iter().enumerate() {
                let target = self.step(state, byte);
                if target != u32::MAX {
                    frontiers[slot].push(target);
                }
            }
        }
    }
}

impl BoundedExpansion<'_> {
    #[inline]
    fn step(&mut self, state: u32, byte: u8, preseeded: Option<&[u32]>) -> u32 {
        ensure_config_transition(
            self.tokenizer, self.raw_transitions, state, byte,
            self.configs, self.transitions, self.known_transitions,
            self.byte_to_column, self.width,
            preseeded, self.singleton_closures, self.active_language,
            self.image,
        )
    }

    fn charge(
        &self, visits: &mut usize, count: usize, budget: Option<TokenBoundedAnalysisWorkBudget>,
    ) -> Result<(), TokenBoundedAnalysisWork> {
        charge_frontier(self.configs.len(), visits, count, budget)
    }

    fn expand(
        &mut self, start_states: &[u32], trie: PrefixWalk<'_>, preseeded: Option<&[u32]>,
        budget: Option<TokenBoundedAnalysisWorkBudget>, prior_visits: usize,
    ) -> Result<usize, TokenBoundedAnalysisWork> {
        traverse_frontiers(&mut LazyTrajectory { expansion: self, preseeded },
            start_states, trie, budget, prior_visits)
    }

    /// Independent literal product traversal, deliberately not implemented
    /// through the frontier recurrence. Used only for semantic comparisons.
    #[cfg(test)]
    fn expand_reference(
        &mut self,
        start_states: &[u32],
        trie: PrefixWalk<'_>,
        preseeded: Option<&[u32]>,
        budget: Option<TokenBoundedAnalysisWorkBudget>,
        prior_visits: usize,
    ) -> Result<usize, TokenBoundedAnalysisWork> {
        let mut visited = rustc_hash::FxHashSet::default();
        let mut visits = prior_visits;
        let mut stack = Vec::new();
        for &start in start_states {
            stack.push((start, trie.root()));
            while let Some((state, node)) = stack.pop() {
                if !visited.insert((state, node)) {
                    continue;
                }
                self.charge(&mut visits, 1, budget)?;
                for (byte, child) in (0..256).map_while(|slot| trie.child_at(node, slot)) {
                    let target = self.step(state, byte, preseeded);
                    if target != u32::MAX {
                        stack.push((target, child));
                    }
                }
            }
        }
        self.charge(&mut visits, 0, budget)?;
        Ok(visits - prior_visits)
    }

    fn expand_selected(
        &mut self,
        start_states: &[u32],
        trie: PrefixWalk<'_>,
        preseeded: Option<&[u32]>,
        budget: Option<TokenBoundedAnalysisWorkBudget>,
        prior_visits: usize,
        use_frontier: bool,
    ) -> Result<usize, TokenBoundedAnalysisWork> {
        #[cfg(test)]
        if !use_frontier {
            return self.expand_reference(start_states, trie, preseeded, budget, prior_visits);
        }
        let _ = use_frontier;
        self.expand(start_states, trie, preseeded, budget, prior_visits)
    }
}

#[derive(Clone)]
pub struct TokenBoundedAnalysisTopology {
    configurations: Arc<[Box<[u32]>]>,
    start_state: usize,
    transitions: ByteRows,
    raw_start_to_view: Arc<[u32]>,
}

fn materialize_bounded_analysis_view(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    configurations: &[Box<[u32]>],
    start_state: usize,
    transitions: ByteRows,
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
        tokenizer_view: FlatDfa {
            states,
            start_state,
            transitions,
        },
        raw_start_to_view,
    }
}

impl TokenBoundedAnalysisTopology {
    pub fn state_count(&self) -> usize {
        self.configurations.len()
    }

    /// Materialize a topology that was already constructed with the same
    /// active-language projection. Re-projecting such a topology would be
    /// exact but wastefully scan and rebuild its transition rows.
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
            self.transitions.clone(),
            self.raw_start_to_view.to_vec(),
        )
    }

    pub fn materialize(
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
                self.transitions.clone(),
                self.raw_start_to_view.to_vec(),
            );
        };

        // The token topology is intentionally built once for a superset of
        // terminal masks. Project raw NFA members when the topology is consumed
        // so inactive-only lexer structure cannot distinguish active-language
        // states. Since an inactive-language state has no active-language
        // successor, projecting after each prebuilt transition is equivalent to
        // projecting its source before taking that transition.
        let mut projected_configs = ConfigInterner::default();
        let projected_state_by_topology_state = self
            .configurations
            .iter()
            .map(|config| {
                let projected = config
                    .iter()
                    .copied()
                    .filter(|&raw| active_language[raw as usize])
                    .collect::<Vec<_>>();
                projected_configs.intern(projected)
            })
            .collect::<Vec<_>>();

        let mut projected_transitions =
            vec![u32::MAX; projected_configs.len().saturating_mul(self.transitions.width)];
        for topology_source in 0..self.configurations.len() {
            let source = projected_state_by_topology_state[topology_source] as usize;
            if projected_configs[source].is_empty() {
                continue;
            }
            for (column, &topology_target) in self.transitions.row(topology_source).iter().enumerate() {
                if topology_target == u32::MAX {
                    continue;
                }
                let target = projected_state_by_topology_state[topology_target as usize];
                if projected_configs[target as usize].is_empty() {
                    continue;
                }
                let slot = source * self.transitions.width + column;
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
                if state != u32::MAX { projected_state_by_topology_state[state as usize] } else { u32::MAX }
            })
            .collect::<Vec<_>>();
        let start_state = projected_state_by_topology_state[self.start_state] as usize;
        materialize_bounded_analysis_view(
            tokenizer,
            active_groups,
            &projected_configs,
            start_state,
            ByteRows::new(Arc::from(projected_transitions), self.transitions.byte_to_column),
            raw_start_to_view,
        )
    }
}

fn build_bounded_analysis_topology_impl(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    combine_start_states: bool,
    factor_common_first_byte: bool,
    include_reset_suffixes: bool,
    tokens_are_sorted: bool,
    active_groups: Option<&[bool]>,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> Result<(TokenBoundedAnalysisTopology, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let use_frontier_expansion = true;
    build_bounded_analysis_topology_impl_with_expansion(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        combine_start_states,
        factor_common_first_byte,
        include_reset_suffixes,
        tokens_are_sorted,
        active_groups,
        budget,
        prebuilt_token_trie,
        use_frontier_expansion,
    )
}

fn build_bounded_analysis_topology_impl_with_expansion(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    combine_start_states: bool,
    factor_common_first_byte: bool,
    include_reset_suffixes: bool,
    tokens_are_sorted: bool,
    active_groups: Option<&[bool]>,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
    use_frontier_expansion: bool,
) -> Result<(TokenBoundedAnalysisTopology, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let total_started_at = profile_timing.then(std::time::Instant::now);
    let setup_started_at = profile_timing.then(std::time::Instant::now);
    let raw_state_count = tokenizer.num_states() as usize;
    let mut configs = ConfigInterner::default();
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
    let start_state = configs.intern(start_closure);
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
        let state = configs.intern(closure);
        for &raw_state in raw_start_states {
            raw_start_to_view[raw_state] = state;
        }
    } else {
        for &raw_state in raw_start_states {
            assert!(raw_state < raw_state_count, "invalid raw NFA analysis seed");
            let closure = project_closure(&singleton_closures[raw_state]);
            let state = configs.intern(closure);
            raw_start_to_view[raw_state] = state;
        }
    }
    if budget.is_some_and(|budget| configs.len() > budget.max_configurations) {
        return Err(TokenBoundedAnalysisWork {
            configurations: configs.len(),
            trie_visits: 0,
        });
    }
    let setup_ms = setup_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let trie_started_at = profile_timing.then(std::time::Instant::now);
    let common_first_byte = factor_common_first_byte
        .then(|| {
            let first = tokens.first()?.first().copied()?;
            tokens
                .iter()
                .all(|token| token.len() > 1 && token.first().copied() == Some(first))
                .then_some(first)
        })
        .flatten();
    let owned_token_trie;
    let (token_trie, token_trie_len) = if common_first_byte.is_none()
        && let Some(prebuilt) = prebuilt_token_trie
    {
        (PrefixWalk::Radix(&prebuilt.tree), prebuilt.len())
    } else {
        owned_token_trie = if common_first_byte.is_some() {
            if tokens_are_sorted {
                let suffixes = tokens.iter().map(|token| &token[1..]).collect::<Vec<_>>();
                build_byte_trie_sorted(&suffixes)
            } else {
                build_byte_trie(tokens.iter().map(|token| &token[1..]))
            }
        } else if tokens_are_sorted {
            build_byte_trie_sorted(tokens)
        } else {
            build_byte_trie(tokens.iter().copied())
        };
        (PrefixWalk::Bytes(&owned_token_trie), owned_token_trie.len())
    };
    let suffix_trie = include_reset_suffixes.then(|| {
        build_byte_trie(
            tokens
                .iter()
                .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..])),
        )
    });
    let trie_ms = trie_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    // The shared prefix trie may be a superset of the requested tokens.
    // The alphabet is that of the actual walks, not just the local token list.
    let mut relevant = [false; 256];
    token_trie.mark_relevant(&mut relevant);
    if let Some(suffix) = &suffix_trie {
        PrefixWalk::Bytes(suffix).mark_relevant(&mut relevant);
    }
    if let Some(byte) = common_first_byte {
        relevant[byte as usize] = true;
    }
    let classes = conservative_byte_classes(&relevant)
        .unwrap_or_else(|| std::array::from_fn(|byte| byte as u8));
    let width = classes.iter().copied().max().unwrap() as usize + 1;
    let mut transitions = vec![u32::MAX; configs.len() * width];
    let mut known_transitions = vec![0u8; transitions.len()];
    let expansion_started_at = profile_timing.then(std::time::Instant::now);
    let mut image = EpsilonImageScratch::new(raw_state_count);
    let mut seeded_configs = raw_start_states
        .iter()
        .map(|&raw| raw_start_to_view[raw])
        .collect::<Vec<_>>();
    seeded_configs.sort_unstable();
    seeded_configs.dedup();
    let preseeded_raw_closures = (!combine_start_states).then_some(raw_start_to_view.as_slice());
    let mut expansion = BoundedExpansion {
        tokenizer,
        raw_transitions,
        configs: &mut configs,
        transitions: &mut transitions,
        known_transitions: &mut known_transitions,
        byte_to_column: &classes,
        width,
        singleton_closures: singleton_closures.as_ref(),
        active_language: active_language.as_deref(),
        image: &mut image,
    };
    if let Some(first_byte) = common_first_byte {
        seeded_configs = seeded_configs.into_iter()
            .map(|state| expansion.step(state, first_byte, preseeded_raw_closures))
            .filter(|&state| state != u32::MAX)
            .collect();
        seeded_configs.sort_unstable();
        seeded_configs.dedup();
        let mut visits = 0;
        expansion.charge(&mut visits, 0, budget)?;
    }
    let token_trie_visits = expansion.expand_selected(
        &seeded_configs, token_trie, preseeded_raw_closures,
        budget, 0, use_frontier_expansion,
    )?;
    let suffix_trie_visits = if let Some(suffix_trie) = suffix_trie.as_ref() {
        expansion.expand_selected(
            std::slice::from_ref(&start_state), PrefixWalk::Bytes(suffix_trie), None,
            budget, token_trie_visits, use_frontier_expansion,
        )?
    } else {
        0
    };

    let expansion_ms = expansion_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let work = TokenBoundedAnalysisWork {
        configurations: configs.len(),
        trie_visits: token_trie_visits + suffix_trie_visits,
    };
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][token_bounded_topology] raw_states={} tokens={} active_projection={} combine_starts={} common_first={} trie_nodes={} configs={} trie_visits={} setup_ms={:.3} trie_ms={:.3} expansion_ms={:.3} total_ms={:.3}",
            raw_state_count,
            tokens.len(),
            active_groups.is_some(),
            combine_start_states,
            common_first_byte.is_some(),
            token_trie_len,
            work.configurations,
            work.trie_visits,
            setup_ms,
            trie_ms,
            expansion_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((
        TokenBoundedAnalysisTopology {
            configurations: Arc::from(configs.into_configs()),
            start_state: start_state as usize,
            transitions: ByteRows::new(Arc::from(transitions), classes),
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
        None,
        raw_start_states,
        tokens,
        combine_start_states,
        true,
        true,
        false,
        None,
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
        None,
        raw_start_states,
        tokens,
        combine_start_states,
        factor_common_first_byte,
        include_reset_suffixes,
        false,
        None,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

pub fn build_bounded_analysis_view(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_view_with_trie(
        tokenizer,
        raw_start_states,
        tokens,
        active_groups,
        None,
    )
}

pub fn build_bounded_analysis_view_with_trie(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    // Invariant: analysis views used by vocabulary equivalence must materialize
    // all token suffix trajectories from start_state (whose epsilon closure covers
    // all component reset roots) so that restart/suffix evaluations encounter valid
    // transitions.
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        false,
        true,
        true, // include_reset_suffixes
        false,
        None,
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

pub fn build_bounded_analysis_view_from_combined_starts(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_view_from_combined_starts_with_trie(
        tokenizer,
        raw_start_states,
        tokens,
        active_groups,
        None,
    )
}

pub fn build_bounded_analysis_view_from_combined_starts_with_trie(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    // Invariant: analysis views used by vocabulary equivalence must materialize
    // all token suffix trajectories from start_state (whose epsilon closure covers
    // all component reset roots) so that restart/suffix evaluations encounter valid
    // transitions.
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        true,
        true,
        true, // include_reset_suffixes
        false,
        None,
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

/// Build the combined-start bounded view needed for exact L2P prefix scans.
/// Unlike the general bounded view, this does not materialize reset-state
/// transitions for every token suffix; callers must evaluate reset suffixes
/// separately.
pub fn build_bounded_analysis_view_from_combined_starts_prefix_only_with_trie(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        true,
        true,
        false,
        false,
        None,
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded prefix-only token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

pub fn build_token_bounded_analysis_topology(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
) -> TokenBoundedAnalysisTopology {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        false,
        None,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
}

pub fn build_token_bounded_analysis_view_projected(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        false,
        None,
    )
}

pub fn build_token_bounded_analysis_view_projected_sorted(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        true,
        None,
    )
}

pub fn build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        active_groups,
        None,
    )
}

pub fn build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        Some(raw_transitions),
        raw_start_states,
        tokens,
        active_groups,
        true,
        prebuilt_token_trie,
    )
}

fn build_token_bounded_analysis_view_projected_with_order(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    tokens_are_sorted: bool,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        tokens_are_sorted,
        Some(active_groups),
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize_already_projected(tokenizer, active_groups)
}

pub fn try_build_token_bounded_analysis_view_projected(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        budget,
        false,
        None,
    )
}

pub fn try_build_token_bounded_analysis_view_projected_sorted(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        budget,
        true,
        None,
    )
}

pub fn try_build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        active_groups,
        budget,
        None,
    )
}

pub fn try_build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        Some(raw_transitions),
        raw_start_states,
        tokens,
        active_groups,
        budget,
        true,
        prebuilt_token_trie,
    )
}

fn try_build_token_bounded_analysis_view_projected_with_order(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
    tokens_are_sorted: bool,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let (topology, work) = build_bounded_analysis_topology_impl(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        tokens_are_sorted,
        Some(active_groups),
        Some(budget),
        prebuilt_token_trie,
    )?;
    Ok((
        topology.materialize_already_projected(tokenizer, active_groups),
        work,
    ))
}

pub fn build_token_bounded_analysis_view(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_token_bounded_analysis_topology(tokenizer, raw_start_states, tokens)
        .materialize(tokenizer, active_groups)
}

pub fn build_token_bounded_analysis_view_from_combined_starts(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        true,
        true,
        false,
        false,
        None,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
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
    let class_label_count = candidate_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    let mut class_to_internal = vec![u32::MAX; class_label_count];
    let mut original_to_internal = vec![u32::MAX; num_states];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for ((members, &representative), &class) in candidate_members
        .iter()
        .zip(candidate_representatives)
        .zip(candidate_classes)
    {
        let class = class as usize;
        let internal = if class_to_internal[class] == u32::MAX {
            let internal = internal_to_originals.len() as u32;
            class_to_internal[class] = internal;
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(representative as u32);
            internal
        } else {
            class_to_internal[class]
        };
        let bucket = &mut internal_to_originals[internal as usize];
        for &raw in members {
            original_to_internal[raw as usize] = internal;
            bucket.push(raw);
        }
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn prebuilt_candidate_target_sets(
    configurations: &[Box<[u32]>],
    raw_to_candidate: &[usize],
    raw_active_language: Option<&[bool]>,
) -> Vec<SmallVec<[u32; 4]>> {
    configurations
        .iter()
        .map(|config| {
            let mut candidates = SmallVec::<[u32; 4]>::new();
            candidates.extend(config.iter().filter_map(|&raw| {
                if raw_active_language.is_some_and(|active| !active[raw as usize]) {
                    None
                } else {
                    Some(raw_to_candidate[raw as usize] as u32)
                }
            }));
            candidates.sort_unstable();
            candidates.dedup();
            candidates
        })
        .collect()
}

fn prebuilt_candidate_signature(
    candidate: usize,
    class_for_candidate: &[u32],
    start_configs: &[u32],
    candidate_sets_by_config: &[SmallVec<[u32; 4]>],
    edge_offsets: &[u32],
    edges: &[(u8, u32)],
) -> SmallVec<[u32; 64]> {
    let config = start_configs[candidate] as usize;
    if candidate_sets_by_config[config].is_empty() {
        return SmallVec::new();
    }
    let edge_start = edge_offsets[config] as usize;
    let edge_end = edge_offsets[config + 1] as usize;
    let mut signature = SmallVec::<[u32; 64]>::new();
    let mut target_classes = SmallVec::<[u32; 8]>::new();
    for &(byte, target_config) in &edges[edge_start..edge_end] {
        let target_candidates = &candidate_sets_by_config[target_config as usize];
        if target_candidates.is_empty() {
            continue;
        }
        target_classes.clear();
        target_classes.extend(
            target_candidates
                .iter()
                .map(|&target| class_for_candidate[target as usize]),
        );
        target_classes.sort_unstable();
        target_classes.dedup();
        signature.push(byte as u32 + 1);
        signature.push(target_classes.len() as u32);
        signature.extend(target_classes.iter().map(|&class| class + 1));
    }
    signature
}

/// Compute the same stable set-valued NFA partition as the synchronous Moore
/// recurrence, but only revisit source blocks whose signatures can change.
///
/// A block split changes the class id of only the moved target candidates.
/// Therefore only source candidates with an edge to one of those moved targets
/// can acquire a different transition-class-set signature. Repartitioning just
/// those predecessor blocks reaches the same coarsest stable refinement while
/// avoiding one whole-graph pass per byte of a long lexer chain.
fn refine_prebuilt_sparse_powerset_worklist(
    initial_classes: &[u32],
    start_configs: &[u32],
    configurations: &[Box<[u32]>],
    raw_to_candidate: &[usize],
    raw_active_language: Option<&[bool]>,
    edge_offsets: &[u32],
    edges: &[(u8, u32)],
) -> Vec<u32> {
    let num_candidates = initial_classes.len();
    if num_candidates <= 1 {
        return initial_classes.to_vec();
    }

    let candidate_sets_by_config = prebuilt_candidate_target_sets(
        configurations,
        raw_to_candidate,
        raw_active_language,
    );

    let initial_class_count = initial_classes
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut class_for_candidate = initial_classes.to_vec();
    let mut members_by_class = vec![Vec::<u32>::new(); initial_class_count];
    for (candidate, &class) in class_for_candidate.iter().enumerate() {
        members_by_class[class as usize].push(candidate as u32);
    }

    // Reverse dependency graph: target candidate -> source candidates whose
    // transition signature mentions that target. Byte labels are unnecessary
    // here because a queued source block recomputes its complete exact
    // signature. Duplicate source entries are harmless and are suppressed by
    // the per-block queued bit.
    let mut reverse_predecessors = vec![Vec::<u32>::new(); num_candidates];
    for source in 0..num_candidates {
        let config = start_configs[source] as usize;
        if candidate_sets_by_config[config].is_empty() {
            continue;
        }
        let edge_start = edge_offsets[config] as usize;
        let edge_end = edge_offsets[config + 1] as usize;
        for &(_, target_config) in &edges[edge_start..edge_end] {
            for &target in &candidate_sets_by_config[target_config as usize] {
                reverse_predecessors[target as usize].push(source as u32);
            }
        }
    }
    for predecessors in &mut reverse_predecessors {
        predecessors.sort_unstable();
        predecessors.dedup();
    }

    // Keep exact signatures for established blocks. Initially every output
    // block must be classified once. After that, a target split dirties only
    // predecessor *candidates*, and only those candidates are re-signatured.
    // This avoids repeatedly scanning a large mostly-unchanged source block.
    let mut block_signature = vec![None::<SmallVec<[u32; 64]>>; members_by_class.len()];
    let mut position_in_class = vec![0usize; num_candidates];
    for members in &members_by_class {
        for (position, &candidate) in members.iter().enumerate() {
            position_in_class[candidate as usize] = position;
        }
    }
    let mut dirty_by_class = vec![Vec::<u32>::new(); members_by_class.len()];
    let mut candidate_dirty = vec![false; num_candidates];
    let mut queue = VecDeque::<usize>::new();
    let mut queued = vec![false; members_by_class.len()];
    for class in 0..members_by_class.len() {
        if members_by_class[class].len() > 1 {
            queued[class] = true;
            queue.push_back(class);
        }
    }

    let mut moved_candidates = Vec::<u32>::new();

    while let Some(class) = queue.pop_front() {
        queued[class] = false;
        if members_by_class[class].len() <= 1 {
            dirty_by_class[class].clear();
            continue;
        }
        moved_candidates.clear();

        if block_signature[class].is_none() {
            // First exact classification of this output block. New blocks are
            // born with a known exact signature and thereafter use the dirty
            // candidate path below.
            let old_members = std::mem::take(&mut members_by_class[class]);
            let mut groups = Vec::<(SmallVec<[u32; 64]>, Vec<u32>)>::new();
            let mut group_by_signature =
                FxHashMap::<SmallVec<[u32; 64]>, usize>::default();
            for candidate in old_members {
                let signature = prebuilt_candidate_signature(
                    candidate as usize,
                    &class_for_candidate,
                    start_configs,
                    &candidate_sets_by_config,
                    edge_offsets,
                    edges,
                );
                let next_group = groups.len();
                let group = *group_by_signature.entry(signature.clone()).or_insert_with(|| {
                    groups.push((signature, Vec::new()));
                    next_group
                });
                groups[group].1.push(candidate);
            }

            let retained = groups
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, members))| members.len())
                .map(|(index, _)| index)
                .expect("nonempty block has a signature group");
            groups.swap(0, retained);
            let (retained_signature, retained_members) = groups.remove(0);
            block_signature[class] = Some(retained_signature);
            members_by_class[class] = retained_members;
            for (position, &candidate) in members_by_class[class].iter().enumerate() {
                class_for_candidate[candidate as usize] = class as u32;
                position_in_class[candidate as usize] = position;
            }

            for (signature, group) in groups {
                let new_class = members_by_class.len();
                for (position, &candidate) in group.iter().enumerate() {
                    class_for_candidate[candidate as usize] = new_class as u32;
                    position_in_class[candidate as usize] = position;
                    moved_candidates.push(candidate);
                }
                members_by_class.push(group);
                block_signature.push(Some(signature));
                dirty_by_class.push(Vec::new());
                queued.push(false);
            }
        } else {
            // Only candidates explicitly invalidated by a target-class change
            // can differ from this block's stored exact signature.
            let old_signature = block_signature[class]
                .as_ref()
                .expect("initialized block has a signature")
                .clone();
            let dirty = std::mem::take(&mut dirty_by_class[class]);
            let mut changed_groups =
                FxHashMap::<SmallVec<[u32; 64]>, Vec<u32>>::default();
            for candidate in dirty {
                let candidate = candidate as usize;
                if !candidate_dirty[candidate]
                    || class_for_candidate[candidate] as usize != class
                {
                    continue;
                }
                candidate_dirty[candidate] = false;
                let signature = prebuilt_candidate_signature(
                    candidate,
                    &class_for_candidate,
                    start_configs,
                    &candidate_sets_by_config,
                    edge_offsets,
                    edges,
                );
                if signature != old_signature {
                    changed_groups
                        .entry(signature)
                        .or_default()
                        .push(candidate as u32);
                }
            }

            if !changed_groups.is_empty() {
                let changed_count = changed_groups.values().map(Vec::len).sum::<usize>();
                let old_signature_members = members_by_class[class].len() - changed_count;

                // If every member changed, no candidate still represents the
                // old block signature. Retain the largest new group under the
                // existing class id; only candidates moved to other ids need
                // predecessor invalidation.
                let retained_changed_signature = if old_signature_members == 0 {
                    changed_groups
                        .iter()
                        .max_by_key(|(_, members)| members.len())
                        .map(|(signature, _)| signature.clone())
                } else {
                    None
                };

                let retained_changed_group = retained_changed_signature.is_some();
                if let Some(retained_signature) = retained_changed_signature {
                    let retained = changed_groups
                        .remove(&retained_signature)
                        .expect("retained changed group exists");
                    block_signature[class] = Some(retained_signature);
                    members_by_class[class].clear();
                    members_by_class[class].extend_from_slice(&retained);
                    for (position, &candidate) in retained.iter().enumerate() {
                        class_for_candidate[candidate as usize] = class as u32;
                        position_in_class[candidate as usize] = position;
                    }
                }

                for (signature, group) in changed_groups {
                    let new_class = members_by_class.len();
                    // Remove only changed candidates. Unaffected members stay
                    // in place and keep the old exact block signature.
                    if !retained_changed_group {
                        for &candidate in &group {
                            let candidate_index = candidate as usize;
                            let position = position_in_class[candidate_index];
                            let removed = members_by_class[class].swap_remove(position);
                            debug_assert_eq!(removed, candidate);
                            if position < members_by_class[class].len() {
                                let swapped = members_by_class[class][position] as usize;
                                position_in_class[swapped] = position;
                            }
                        }
                    }
                    for (position, &candidate) in group.iter().enumerate() {
                        class_for_candidate[candidate as usize] = new_class as u32;
                        position_in_class[candidate as usize] = position;
                        moved_candidates.push(candidate);
                    }
                    members_by_class.push(group);
                    block_signature.push(Some(signature));
                    dirty_by_class.push(Vec::new());
                    queued.push(false);
                }
            }
        }

        // A moved target changes only the signatures of source candidates that
        // mention it. Mark those candidates, not their whole blocks. Blocks
        // still awaiting their first exact classification need no dirty mark:
        // that full classification will observe the current target classes.
        for &moved in &moved_candidates {
            for &source in &reverse_predecessors[moved as usize] {
                let source = source as usize;
                let source_class = class_for_candidate[source] as usize;
                if members_by_class[source_class].len() <= 1
                    || block_signature[source_class].is_none()
                    || candidate_dirty[source]
                {
                    continue;
                }
                candidate_dirty[source] = true;
                dirty_by_class[source_class].push(source as u32);
                if !queued[source_class] {
                    queued[source_class] = true;
                    queue.push_back(source_class);
                }
            }
        }
    }

    class_for_candidate
}

/// Refine raw scanner starts through a prebuilt sparse powerset topology.
/// When `raw_active_language` is supplied, inactive-only members are erased
/// from powerset class sets and edges into projected-empty configurations are
/// treated as the common implicit missing target. This permits a topology
/// built for a larger terminal mask to be reused after the mask shrinks.
pub fn compute_state_map_from_prebuilt_sparse_powerset(
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

    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let num_candidates = candidate_representatives.len();
    let start_configs = candidate_representatives
        .iter()
        .map(|&state| {
            let config = raw_start_to_view[state];
            assert_ne!(
                config,
                u32::MAX,
                "prebuilt powerset omitted candidate raw representative {state}",
            );
            assert!(
                (config as usize) < configurations.len(),
                "prebuilt powerset mapped raw representative {state} to invalid configuration {config}",
            );
            config
        })
        .collect::<Vec<_>>();
    let mut classes = start_configs
        .iter()
        .map(|&config| output_class_by_config[config as usize])
        .collect::<Vec<_>>();

    if matches!(depth, RefinementDepth::Stable) {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = std::time::Instant::now();
        let refined = refine_prebuilt_sparse_powerset_worklist(
            &classes,
            &start_configs,
            configurations,
            &raw_to_candidate,
            raw_active_language,
            edge_offsets,
            edges,
        );
        if profile_timing {
            let class_count = refined
                .iter()
                .copied()
                .collect::<rustc_hash::FxHashSet<_>>()
                .len();
            eprintln!(
                "[glrmask/profile][nfa_restricted_worklist] states={} candidates={} configs={} edges={} classes={} total_ms={:.3}",
                num_states,
                num_candidates,
                configurations.len(),
                edges.len(),
                class_count,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return build_state_map(
            &candidate_members,
            &candidate_representatives,
            &refined,
            num_states,
        );
    }

    let projected_empty = configurations
        .iter()
        .map(|config| {
            raw_active_language.is_some_and(|active| {
                config.iter().all(|&raw| !active[raw as usize])
            })
        })
        .collect::<Vec<_>>();
    let round_limit = match depth {
        RefinementDepth::Stable => unreachable!("stable refinement returned above"),
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

pub fn compute_state_map(
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

    let mut configs = ConfigInterner::default();
    let start_configs = candidate_representatives
        .iter()
        .map(|&state| {
            let config = project_raw_config(
                singleton_closures[state].to_vec(),
                raw_active_language.as_deref(),
            );
            configs.intern(config)
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
    let mut image = EpsilonImageScratch::new(num_states);
    for candidate in 0..num_candidates {
        for (slot, &byte) in active_bytes.iter().enumerate() {
            let targets = image.step(
                &configs[start_configs[candidate] as usize], byte, true,
                singleton_closures.as_ref(), raw_active_language.as_deref(),
                |source, byte| tokenizer.get_transition(source, byte),
            );
            if !targets.is_empty() {
                target_configs[candidate * active_bytes.len() + slot] =
                    configs.intern_slice(targets);
            }
        }
    }

    if matches!(depth, RefinementDepth::Stable) {
        // The direct NFA analysis has already materialized the exact one-byte
        // epsilon-closed successor configuration for every candidate and
        // relevant byte.  Reuse the same predecessor-driven stable refinement
        // as the sparse powerset path instead of rescanning every candidate in
        // synchronous Moore rounds.  Long deterministic chains can require one
        // logical refinement propagation per byte of distinguishing depth; the
        // worklist follows only affected predecessors rather than performing a
        // whole-graph pass for each propagation step.
        //
        // Multiple candidates can share the same epsilon-closed start
        // configuration. Their one-byte successor configurations are then
        // identical, so one candidate row is sufficient to materialize that
        // configuration's sparse outgoing edges.
        let mut source_candidate_by_config = vec![usize::MAX; configs.len()];
        for (candidate, &config) in start_configs.iter().enumerate() {
            let slot = &mut source_candidate_by_config[config as usize];
            if *slot == usize::MAX {
                *slot = candidate;
            } else {
                debug_assert!((0..active_bytes.len()).all(|byte_slot| {
                    target_configs[*slot * active_bytes.len() + byte_slot]
                        == target_configs[candidate * active_bytes.len() + byte_slot]
                }));
            }
        }

        let mut edge_offsets = Vec::<u32>::with_capacity(configs.len() + 1);
        let mut edges = Vec::<(u8, u32)>::new();
        edge_offsets.push(0);
        for source_candidate in source_candidate_by_config {
            if source_candidate != usize::MAX {
                let row_start = source_candidate * active_bytes.len();
                for (byte_slot, &byte) in active_bytes.iter().enumerate() {
                    let target = target_configs[row_start + byte_slot];
                    if target != u32::MAX {
                        edges.push((byte, target));
                    }
                }
            }
            edge_offsets.push(edges.len() as u32);
        }

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = std::time::Instant::now();
        let refined = refine_prebuilt_sparse_powerset_worklist(
            &classes,
            &start_configs,
            &configs,
            &raw_to_candidate,
            None,
            &edge_offsets,
            &edges,
        );
        if profile_timing {
            let class_count = refined
                .iter()
                .copied()
                .collect::<rustc_hash::FxHashSet<_>>()
                .len();
            eprintln!(
                "[glrmask/profile][nfa_restricted_direct_worklist] states={} candidates={} configs={} edges={} classes={} total_ms={:.3}",
                num_states,
                num_candidates,
                configs.len(),
                edges.len(),
                class_count,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return build_state_map(
            &candidate_members,
            &candidate_representatives,
            &refined,
            num_states,
        );
    }

    let round_limit = match depth {
        RefinementDepth::Stable => unreachable!("stable refinement returned above"),
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

    #[test]
    fn state_map_compacts_sparse_worklist_class_labels() {
        let map = build_state_map(
            &[vec![0], vec![1], vec![2]],
            &[0, 1, 2],
            &[0, 2, 2],
            3,
        );

        assert_eq!(map.original_to_internal, vec![0, 1, 1]);
        assert_eq!(map.internal_to_originals, vec![vec![0], vec![1, 2]]);
        assert_eq!(map.representative_original_ids, vec![0, 1]);
    }

    #[test]
    fn prebuilt_superset_token_trie_matches_subset_analysis() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let flat_trans = crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(&tokenizer);
        let parent_tokens: Vec<&[u8]> = vec![b"", b"a", b"aa", b"ab", b"b", b"ba", b"x"];
        let subset_tokens: Vec<&[u8]> = vec![b"", b"a", b"aa", b"ab", b"b", b"ba"];
        let active = [true, true];
        let parent_trie = build_token_bounded_analysis_trie_sorted(&parent_tokens);
        let direct = build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
            &tokenizer,
            &flat_trans,
            &raw_states,
            &subset_tokens,
            &active,
        );
        let reused =
            build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
                &tokenizer,
                &flat_trans,
                &raw_states,
                &subset_tokens,
                &active,
                Some(&parent_trie),
            );
        for &raw_state in &raw_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let reused_start = reused.view_state_for_raw_start(raw_state);
            for &token in &subset_tokens {
                assert_eq!(
                    view_trace(&direct.tokenizer_view, direct_start, token),
                    view_trace(&reused.tokenizer_view, reused_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    #[test]
    fn prebuilt_superset_token_trie_matches_bounded_reset_suffix_analysis() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let parent_tokens: Vec<&[u8]> =
            vec![b"a", b"aa", b"ab", b"aba", b"b", b"ba", b"x"];
        let subset_tokens: Vec<&[u8]> = vec![b"aa", b"aba", b"ba"];
        let parent_trie = build_token_bounded_analysis_trie_sorted(&parent_tokens);
        let direct = build_bounded_analysis_view(
            &tokenizer,
            &raw_states,
            &subset_tokens,
            Some(&[true, true]),
        );
        let reused = build_bounded_analysis_view_with_trie(
            &tokenizer,
            &raw_states,
            &subset_tokens,
            Some(&[true, true]),
            Some(&parent_trie),
        );
        let observed = subset_tokens
            .iter()
            .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..]))
            .collect::<Vec<_>>();
        for &raw_state in &raw_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let reused_start = reused.view_state_for_raw_start(raw_state);
            for &bytes in &observed {
                assert_eq!(
                    view_trace(&direct.tokenizer_view, direct_start, bytes),
                    view_trace(&reused.tokenizer_view, reused_start, bytes),
                    "raw_state={raw_state} bytes={bytes:?}",
                );
            }
        }
    }

    #[test]
    fn radix_virtual_positions_preserve_every_literal_prefix() {
        let words: Vec<&[u8]> = vec![b"", b"abc", b"abcd", b"abef", b"z", b"zz"];
        let radix = build_token_bounded_analysis_trie_sorted(&words);
        let bytes = build_byte_trie_sorted(&words);
        let walk = PrefixWalk::Radix(&radix.tree);
        let mut pending = vec![(walk.root(), 0usize)];
        let mut count = 0;
        while let Some((position, literal)) = pending.pop() {
            count += 1;
            for (slot, &(byte, child)) in bytes[literal].children.iter().enumerate() {
                let (observed, next) = walk.child_at(position, slot).unwrap();
                assert_eq!(byte, observed);
                pending.push((next, child));
            }
            assert!(walk.child_at(position, bytes[literal].children.len()).is_none());
        }
        assert_eq!(count, radix.len());
        assert_eq!(count, bytes.len());
    }

    #[test]
    fn sorted_byte_trie_matches_generic_builder() {
        let sequences: Vec<&[u8]> = vec![
            b"",
            b"a",
            b"a",
            b"aa",
            b"ab",
            b"aba",
            b"b",
            b"ba",
            b"z",
            b"\xff",
        ];
        assert_eq!(
            build_byte_trie(sequences.iter().copied()),
            build_byte_trie_sorted(&sequences),
        );
    }
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

    #[test]
    fn cached_closed_config_step_matches_scalar_step_all() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let singleton_closures = tokenizer.all_singleton_epsilon_closures();
        let mut raw_transitions = vec![u32::MAX; tokenizer.num_states() as usize * 256];
        for state in 0..tokenizer.num_states() {
            for (byte, target) in tokenizer.transitions_from(state) {
                raw_transitions[state as usize * 256 + byte as usize] = target;
            }
        }
        let active_group_masks: [Option<&[bool]>; 4] = [
            None,
            Some(&[true, false]),
            Some(&[false, true]),
            Some(&[true, true]),
        ];

        for active_groups in active_group_masks {
            let active_language = raw_active_language_states(&tokenizer, active_groups);
            let mut image = EpsilonImageScratch::new(tokenizer.num_states() as usize);
            for raw_state in 0..tokenizer.num_states() as usize {
                let config = singleton_closures[raw_state]
                    .iter()
                    .copied()
                    .filter(|&state| {
                        active_language
                            .as_deref()
                            .is_none_or(|active| active[state as usize])
                    })
                    .collect::<Vec<_>>();
                for byte in 0u8..=u8::MAX {
                    let scalar = tokenizer
                        .step_all(&config, byte)
                        .iter()
                        .copied()
                        .filter(|&state| {
                            active_language
                                .as_deref()
                                .is_none_or(|active| active[state as usize])
                        })
                        .collect::<Vec<_>>();
                    let cached = image.step(
                        &config, byte, true, singleton_closures.as_ref(),
                        active_language.as_deref(),
                        |source, byte| tokenizer.get_transition(source, byte),
                    );
                    assert_eq!(cached, scalar.as_slice(), "raw_state={raw_state} byte={byte}");
                    let cached_from_raw = image.step(
                        &config, byte, true, singleton_closures.as_ref(),
                        active_language.as_deref(),
                        |source, byte| raw_transitions[source as usize * 256 + byte as usize],
                    );
                    assert_eq!(
                        cached_from_raw, scalar.as_slice(),
                        "raw transition table raw_state={raw_state} byte={byte}",
                    );
                }
            }
        }
    }

    #[test]
    fn projected_sorted_raw_transition_view_matches_tokenizer_step_view() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let active_groups = [true, false];
        let tokens: [&[u8]; 7] = [b"a", b"aa", b"ab", b"aba", b"b", b"ba", b"xyz"];
        let mut raw_transitions = vec![u32::MAX; tokenizer.num_states() as usize * 256];
        for state in 0..tokenizer.num_states() {
            for (byte, target) in tokenizer.transitions_from(state) {
                raw_transitions[state as usize * 256 + byte as usize] = target;
            }
        }

        let scalar = build_token_bounded_analysis_view_projected_sorted(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
        );
        let dense = build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
            &tokenizer,
            &raw_transitions,
            &raw_start_states,
            &tokens,
            &active_groups,
        );

        assert_eq!(
            dense.tokenizer_view.states.len(),
            scalar.tokenizer_view.states.len(),
        );
        for &raw_state in &raw_start_states {
            let scalar_start = scalar.view_state_for_raw_start(raw_state);
            let dense_start = dense.view_state_for_raw_start(raw_state);
            for &token in &tokens {
                assert_eq!(
                    view_trace(&dense.tokenizer_view, dense_start, token),
                    view_trace(&scalar.tokenizer_view, scalar_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    fn view_trace(
        view: &FlatDfa,
        start_state: usize,
        token: &[u8],
    ) -> Vec<(Vec<usize>, Vec<usize>, bool)> {
        let dfa = &view;
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
    fn trie_frontier_expansion_matches_pair_hash_dfs() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let active_groups = [true, false];
        let token_sets: [&[&[u8]]; 2] = [
            &[b"a", b"aa", b"ab", b"aba"],
            &[b"a", b"ab", b"b", b"ba", b"x", b"xyz"],
        ];

        for tokens in token_sets {
            let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
                &tokenizer,
                None,
                &raw_start_states,
                tokens,
                false,
                true,
                false,
                true,
                Some(&active_groups),
                None,
                None,
                true,
            )
            .expect("frontier topology build");
            let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
                &tokenizer,
                None,
                &raw_start_states,
                tokens,
                false,
                true,
                false,
                true,
                Some(&active_groups),
                None,
                None,
                false,
            )
            .expect("DFS topology build");
            assert_eq!(frontier_work, dfs_work);

            let frontier = frontier.materialize_already_projected(&tokenizer, &active_groups);
            let dfs = dfs.materialize_already_projected(&tokenizer, &active_groups);
            for &raw_state in &raw_start_states {
                let frontier_start = frontier.view_state_for_raw_start(raw_state);
                let dfs_start = dfs.view_state_for_raw_start(raw_state);
                for &token in tokens {
                    assert_eq!(
                        view_trace(&frontier.tokenizer_view, frontier_start, token),
                        view_trace(&dfs.tokenizer_view, dfs_start, token),
                        "raw_state={raw_state} token={token:?}",
                    );
                }
            }
        }

        // A sparse-root edge can reach a raw state that was not itself one of
        // the preseeded start states. The sparse path must fall back to exact
        // configuration construction rather than treating the missing
        // raw_start_to_view entry as DEAD.
        let subset_start_states = vec![2usize, 4usize];
        let subset_tokens: &[&[u8]] = &[b"a", b"b"];
        let subset_active_groups = [true, true];
        let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &subset_start_states,
            subset_tokens,
            false,
            true,
            false,
            true,
            Some(&subset_active_groups),
            None,
            None,
            true,
        )
        .expect("frontier topology build with unpreseeded targets");
        let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &subset_start_states,
            subset_tokens,
            false,
            true,
            false,
            true,
            Some(&subset_active_groups),
            None,
            None,
            false,
        )
        .expect("DFS topology build with unpreseeded targets");
        assert_eq!(frontier_work, dfs_work);
        let frontier = frontier.materialize_already_projected(&tokenizer, &subset_active_groups);
        let dfs = dfs.materialize_already_projected(&tokenizer, &subset_active_groups);
        for &raw_state in &subset_start_states {
            let frontier_start = frontier.view_state_for_raw_start(raw_state);
            let dfs_start = dfs.view_state_for_raw_start(raw_state);
            for &token in subset_tokens {
                assert_eq!(
                    view_trace(&frontier.tokenizer_view, frontier_start, token),
                    view_trace(&dfs.tokenizer_view, dfs_start, token),
                    "unpreseeded target raw_state={raw_state} token={token:?}",
                );
            }
        }

        // Also cover the generic unsorted-token path, combined starts, and the
        // reset-suffix trie used by non-L1 callers.
        let unsorted_tokens: &[&[u8]] = &[b"xyz", b"a", b"ba", b"ab", b"x"];
        let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &raw_start_states,
            unsorted_tokens,
            true,
            false,
            true,
            false,
            None,
            None,
            None,
            true,
        )
        .expect("frontier topology build with reset suffixes");
        let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &raw_start_states,
            unsorted_tokens,
            true,
            false,
            true,
            false,
            None,
            None,
            None,
            false,
        )
        .expect("DFS topology build with reset suffixes");
        assert_eq!(frontier_work, dfs_work);
        let frontier = frontier.materialize(&tokenizer, None);
        let dfs = dfs.materialize(&tokenizer, None);
        for &raw_state in &raw_start_states {
            let frontier_start = frontier.view_state_for_raw_start(raw_state);
            let dfs_start = dfs.view_state_for_raw_start(raw_state);
            for &token in unsorted_tokens {
                assert_eq!(
                    view_trace(&frontier.tokenizer_view, frontier_start, token),
                    view_trace(&dfs.tokenizer_view, dfs_start, token),
                    "combined raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    #[test]
    fn active_filtered_powerset_drops_inactive_members_before_refinement() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_partitioned;

        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ];
        let tokenizer = build_regex_partitioned(
            &expressions,
            &[0, 1, 2])
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
        use crate::automata::lexer::compile::build_regex_partitioned;

        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ];
        let tokenizer = build_regex_partitioned(
            &expressions,
            &[0, 1, 2])
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
        let dfa = &projected.tokenizer_view;

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
            None,
            &raw_start_states,
            &tokens,
            false,
            true,
            false,
            false,
            Some(&active_groups),
            None,
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
                        factored.tokenizer_view.start_state,
                        &token[offset..],
                    ),
                    view_trace(
                        &reference.tokenizer_view,
                        reference.tokenizer_view.start_state,
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
    fn raw_relevant_powerset_projected_targets_match_scalar_exactly() {
        fn assert_same(left: &RelevantPowersetView, right: &RelevantPowersetView) {
            assert_eq!(left.start_state, right.start_state);
            assert_eq!(left.bytes, right.bytes);
            assert_eq!(left.edge_offsets, right.edge_offsets);
            assert_eq!(left.edges, right.edges);
            assert_eq!(left.raw_start_to_view, right.raw_start_to_view);
            assert_eq!(left.configurations, right.configurations);
            assert_eq!(left.states.len(), right.states.len());
            for (left, right) in left.states.iter().zip(&right.states) {
                assert_eq!(left.finalizers, right.finalizers);
                assert_eq!(
                    left.possible_future_group_ids,
                    right.possible_future_group_ids,
                );
            }
        }

        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant_bytes = [false; 256];
        relevant_bytes[b'a' as usize] = true;
        relevant_bytes[b'b' as usize] = true;

        for active_groups in [None, Some([true, true]), Some([true, false]), Some([false, true])] {
            let active_groups = active_groups.as_ref().map(<[_; 2]>::as_slice);
            let scalar = try_build_relevant_powerset_view(
                &tokenizer,
                &relevant_bytes,
                active_groups,
                None,
                None,
                RawPowersetTargetMode::Scalar,
            )
            .expect("unbounded scalar powerset construction");
            let direct = try_build_relevant_powerset_view(
                &tokenizer,
                &relevant_bytes,
                active_groups,
                None,
                None,
                RawPowersetTargetMode::Direct,
            )
            .expect("unbounded direct powerset construction");
            assert_same(&scalar, &direct);
        }
    }

    #[test]
    fn powerset_restricted_bounded_view_matches_direct_bounded_view() {
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
        let direct = build_bounded_analysis_view(
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
        let restricted = build_bounded_analysis_view_from_relevant_powerset(
            &powerset,
            &raw_start_states,
            &tokens,
        );

        for &raw_state in &raw_start_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let restricted_start = restricted.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&restricted.tokenizer_view, restricted_start, token),
                    view_trace(&direct.tokenizer_view, direct_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
        for token in tokens {
            for offset in 0..token.len() {
                assert_eq!(
                    view_trace(
                        &restricted.tokenizer_view,
                        restricted.tokenizer_view.start_state,
                        &token[offset..],
                    ),
                    view_trace(
                        &direct.tokenizer_view,
                        direct.tokenizer_view.start_state,
                        &token[offset..],
                    ),
                    "reset suffix={:?}",
                    &token[offset..],
                );
            }
        }
    }

    #[test]
    fn powerset_restricted_bounded_view_matches_direct_across_projections() {
        fn binary_inputs(max_len: usize) -> Vec<Vec<u8>> {
            let mut inputs = vec![Vec::new()];
            for len in 1..=max_len {
                let start = inputs.len();
                for bits in 0..(1usize << len) {
                    inputs.push(
                        (0..len)
                            .rev()
                            .map(|shift| if bits & (1 << shift) == 0 { b'a' } else { b'b' })
                            .collect(),
                    );
                }
                debug_assert_eq!(inputs.len() - start, 1usize << len);
            }
            inputs
        }

        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let exhaustive = binary_inputs(4);
        let sparse_cases = vec![
            vec![Vec::new(), b"ab".to_vec(), b"baa".to_vec(), b"bbbb".to_vec()],
            vec![b"a".to_vec(), b"bb".to_vec(), b"aaba".to_vec()],
        ];
        let mut token_cases = vec![exhaustive];
        token_cases.extend(sparse_cases);

        for active_groups in [[true, true], [true, false], [false, true]] {
            for owned_tokens in &token_cases {
                let tokens = owned_tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let direct = build_bounded_analysis_view(
                    &tokenizer,
                    &raw_start_states,
                    &tokens,
                    Some(&active_groups),
                );
                let mut relevant_bytes = [false; 256];
                for token in &tokens {
                    for &byte in *token {
                        relevant_bytes[byte as usize] = true;
                    }
                }
                let powerset = build_relevant_powerset_view(
                    &tokenizer,
                    &relevant_bytes,
                    Some(&active_groups),
                    None,
                );
                let restricted = build_bounded_analysis_view_from_relevant_powerset(
                    &powerset,
                    &raw_start_states,
                    &tokens,
                );

                for &raw_state in &raw_start_states {
                    let direct_start = direct.view_state_for_raw_start(raw_state);
                    let restricted_start = restricted.view_state_for_raw_start(raw_state);
                    for token in &tokens {
                        assert_eq!(
                            view_trace(&restricted.tokenizer_view, restricted_start, token),
                            view_trace(&direct.tokenizer_view, direct_start, token),
                            "active_groups={active_groups:?} raw_state={raw_state} token={token:?}",
                        );
                    }
                }
                for token in &tokens {
                    for offset in 0..token.len() {
                        assert_eq!(
                            view_trace(
                                &restricted.tokenizer_view,
                                restricted.tokenizer_view.start_state,
                                &token[offset..],
                            ),
                            view_trace(
                                &direct.tokenizer_view,
                                direct.tokenizer_view.start_state,
                                &token[offset..],
                            ),
                            "active_groups={active_groups:?} reset suffix={:?}",
                            &token[offset..],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stable_restricted_observation_quotient_preserves_relevant_powerset_traces() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let active_groups = [true, true];
        let mut relevant_bytes = [false; 256];
        relevant_bytes[b'a' as usize] = true;
        relevant_bytes[b'b' as usize] = true;
        let stable_map = compute_state_map(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            None,
            RefinementDepth::Stable,
        );
        let raw_powerset = build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            None,
        );
        let quotient_powerset = build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            Some(&stable_map),
        );
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let owned_tokens = [
            Vec::new(),
            b"a".to_vec(),
            b"b".to_vec(),
            b"ab".to_vec(),
            b"ba".to_vec(),
            b"aab".to_vec(),
            b"bba".to_vec(),
            b"abba".to_vec(),
        ];
        let tokens = owned_tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let raw_bounded = build_bounded_analysis_view_from_relevant_powerset(
            &raw_powerset,
            &raw_start_states,
            &tokens,
        );
        let quotient_bounded = build_bounded_analysis_view_from_relevant_powerset(
            &quotient_powerset,
            &raw_start_states,
            &tokens,
        );

        for &raw_state in &raw_start_states {
            let raw_start = raw_bounded.view_state_for_raw_start(raw_state);
            let quotient_start = quotient_bounded.view_state_for_raw_start(raw_state);
            for token in &tokens {
                assert_eq!(
                    view_trace(&raw_bounded.tokenizer_view, raw_start, token),
                    view_trace(&quotient_bounded.tokenizer_view, quotient_start, token),
                    "raw_state={raw_state} token={token:?}"
                );
            }
        }
        for token in &tokens {
            for offset in 0..token.len() {
                assert_eq!(
                    view_trace(
                        &raw_bounded.tokenizer_view,
                        raw_bounded.tokenizer_view.start_state,
                        &token[offset..],
                    ),
                    view_trace(
                        &quotient_bounded.tokenizer_view,
                        quotient_bounded.tokenizer_view.start_state,
                        &token[offset..],
                    ),
                    "reset suffix={:?}",
                    &token[offset..],
                );
            }
        }
    }

    #[test]
    fn relevant_powerset_handles_fully_filtered_empty_configurations() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let relevant = [true; 256];
        let inactive_groups = [false, false];
        let view = build_relevant_powerset_view(
            &tokenizer,
            &relevant,
            Some(&inactive_groups),
            None,
        );

        assert!(view.configurations.iter().all(|config| config.is_empty()));
        assert!(view.edges.is_empty());
        assert!(view.raw_start_to_view.iter().all(|&state| state == view.start_state as u32));
    }

    #[test]
    fn relevant_powerset_budget_aborts_without_changing_successful_construction() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let relevant = [true; 256];
        let reference = build_relevant_powerset_view(&tokenizer, &relevant, None, None);

        let generous = build_relevant_powerset_view_budgeted(
            &tokenizer,
            &relevant,
            None,
            None,
            RelevantPowersetWorkBudget {
                max_configurations: reference.configurations.len(),
                max_edges: reference.edges.len(),
            },
        )
        .expect("budget equal to realized exact work must succeed");
        assert_eq!(generous.configurations.as_ref(), reference.configurations.as_ref());
        assert_eq!(generous.edge_offsets, reference.edge_offsets);
        assert_eq!(generous.edges, reference.edges);

        let too_small = build_relevant_powerset_view_budgeted(
            &tokenizer,
            &relevant,
            None,
            None,
            RelevantPowersetWorkBudget {
                max_configurations: reference.configurations.len(),
                max_edges: reference.edges.len().saturating_sub(1),
            },
        );
        let Err(work) = too_small else {
            panic!("undersized edge budget must abort powerset construction");
        };
        assert!(work.edges > reference.edges.len().saturating_sub(1));
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
    fn direct_nfa_worklist_matches_synchronous_refinement() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant_ab = [false; 256];
        relevant_ab[b'a' as usize] = true;
        relevant_ab[b'b' as usize] = true;
        let relevant_all = [true; 256];
        let active_left = [true, false];
        let active_right = [false, true];
        let active_both = [true, true];

        for relevant in [&relevant_ab, &relevant_all] {
            for active_groups in [
                None,
                Some(active_left.as_slice()),
                Some(active_right.as_slice()),
                Some(active_both.as_slice()),
            ] {
                let synchronous = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    None,
                    RefinementDepth::Bounded(tokenizer.num_states() as usize),
                );
                let worklist = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    None,
                    RefinementDepth::Stable,
                );
                assert!(same_partition(
                    &synchronous.original_to_internal,
                    &worklist.original_to_internal,
                ));

                let seed = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    None,
                    RefinementDepth::Bounded(1),
                );
                let seeded_synchronous = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    Some(&seed),
                    RefinementDepth::Bounded(tokenizer.num_states() as usize),
                );
                let seeded_worklist = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    Some(&seed),
                    RefinementDepth::Stable,
                );
                assert!(same_partition(
                    &seeded_synchronous.original_to_internal,
                    &seeded_worklist.original_to_internal,
                ));
            }
        }
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
        let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            None,
            RefinementDepth::Bounded(tokenizer.num_states() as usize),
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
        assert!(same_partition(
            &synchronous.original_to_internal,
            &reused.original_to_internal,
        ));
    }

    #[test]
    fn prebuilt_sparse_worklist_matches_projected_synchronous_refinement() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let active_groups = [true, false];
        let view = build_relevant_powerset_view(
            &tokenizer,
            &relevant,
            Some(&active_groups),
            None,
        );
        let output_class_by_config = powerset_output_class_ids(&view);
        let active_language = raw_active_language_states(&tokenizer, Some(&active_groups))
            .expect("active group projection");

        let seed = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            None,
            RefinementDepth::Bounded(1),
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            Some(&active_language),
            &view.edge_offsets,
            &view.edges,
        );
        let worklist = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            Some(&seed),
            RefinementDepth::Stable,
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            Some(&active_language),
            &view.edge_offsets,
            &view.edges,
        );
        let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            Some(&seed),
            RefinementDepth::Bounded(tokenizer.num_states() as usize),
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            Some(&active_language),
            &view.edge_offsets,
            &view.edges,
        );

        assert!(same_partition(
            &synchronous.original_to_internal,
            &worklist.original_to_internal,
        ));
    }

    #[test]
    fn prebuilt_sparse_worklist_matches_synchronous_on_random_topologies() {
        fn next_u32(state: &mut u64) -> u32 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 32) as u32
        }

        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let num_states = tokenizer.num_states() as usize;
        for seed in 0..128u64 {
            let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
            let extra_configs = num_states * 3 + 1;
            let mut configurations = (0..num_states)
                .map(|state| vec![state as u32].into_boxed_slice())
                .collect::<Vec<_>>();
            for _ in 0..extra_configs {
                let mut config = Vec::<u32>::new();
                for raw in 0..num_states {
                    if next_u32(&mut rng).is_multiple_of(3) {
                        config.push(raw as u32);
                    }
                }
                if config.is_empty() && next_u32(&mut rng) & 1 != 0 {
                    config.push((next_u32(&mut rng) as usize % num_states) as u32);
                }
                configurations.push(config.into_boxed_slice());
            }

            let raw_start_to_view = (0..num_states as u32).collect::<Vec<_>>();
            let output_class_by_config = (0..configurations.len())
                .map(|_| next_u32(&mut rng) % 5)
                .collect::<Vec<_>>();
            let mut edge_offsets = Vec::<u32>::with_capacity(configurations.len() + 1);
            let mut edges = Vec::<(u8, u32)>::new();
            edge_offsets.push(0);
            for _ in 0..configurations.len() {
                for byte in 0..5u8 {
                    if next_u32(&mut rng) & 1 != 0 {
                        edges.push((
                            byte,
                            next_u32(&mut rng) % configurations.len() as u32,
                        ));
                    }
                }
                edge_offsets.push(edges.len() as u32);
            }
            let mut active_language = (0..num_states)
                .map(|_| next_u32(&mut rng) & 1 != 0)
                .collect::<Vec<_>>();
            if !active_language.iter().any(|&active| active) {
                active_language[next_u32(&mut rng) as usize % num_states] = true;
            }

            for active_language in [None, Some(active_language.as_slice())] {
                let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    None,
                    RefinementDepth::Bounded(num_states),
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                let worklist = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    None,
                    RefinementDepth::Stable,
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                assert!(
                    same_partition(
                        &synchronous.original_to_internal,
                        &worklist.original_to_internal,
                    ),
                    "unseeded mismatch at seed={seed} active_projection={}",
                    active_language.is_some(),
                );

                let initial = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    None,
                    RefinementDepth::Bounded(1),
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                let seeded_synchronous = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    Some(&initial),
                    RefinementDepth::Bounded(num_states),
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                let seeded_worklist = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    Some(&initial),
                    RefinementDepth::Stable,
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                assert!(
                    same_partition(
                        &seeded_synchronous.original_to_internal,
                        &seeded_worklist.original_to_internal,
                    ),
                    "seeded mismatch at seed={seed} active_projection={}",
                    active_language.is_some(),
                );
            }
        }
    }

    #[test]
    fn bounded_analysis_view_materializes_reset_suffix_transitions_from_start_state() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens: Vec<&[u8]> = vec![b"xa", b"xaa", b"xb"];
        let bounded = build_bounded_analysis_view_with_trie(
            &tokenizer,
            &raw_states,
            &tokens,
            None,
            None,
        );
        let dfa = &bounded.tokenizer_view;
        let start_state = dfa.start_state;
        let trans_a = dfa.trans(start_state, b'a' as usize);
        assert_ne!(
            trans_a,
            u32::MAX,
            "bounded view must materialize suffix transitions from start_state for byte 'a'",
        );
    }
}
