//! Flattened tokenizer-DFA views for the equivalence-analysis passes.

use crate::automata::lexer::Lexer;
use crate::automata::lexer::compile::compile_terminal_expr_dfa;
use crate::automata::lexer::DFA;
use crate::ds::u8set::U8Set;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

fn build_transition_table(
    transitions: impl Iterator<Item = (u8, u32)>,
) -> [u32; 256] {
    let mut table = [u32::MAX; 256];
    for (byte, target) in transitions {
        table[byte as usize] = target;
    }
    table
}

fn normalize_group_ids(mut groups: Vec<usize>) -> Vec<usize> {
    groups.sort_unstable();
    groups.dedup();
    groups
}

fn collect_group_ids(groups: impl Iterator<Item = u32>) -> Vec<usize> {
    normalize_group_ids(groups.map(|group| group as usize).collect())
}

fn collect_filtered_group_ids(
    groups: impl Iterator<Item = u32>,
    active_groups: &[bool],
) -> Vec<usize> {
    normalize_group_ids(
        groups
            .map(|group| group as usize)
            .filter(|&group| group < active_groups.len() && active_groups[group])
            .collect(),
    )
}

/// Per-state metadata: finalizers and reachable groups.
/// Transitions are stored separately in `FlatDfa::transitions` for sharing.
#[derive(Debug, Clone)]
pub struct FlatDfaState {
    /// Sorted list of group IDs that finalize at this state.
    pub finalizers: Vec<usize>,
    /// Sorted list of group IDs reachable from this state.
    pub possible_future_group_ids: Vec<usize>,
}

/// Pre-extracted DFA in the format used by equivalence analysis.
/// Transitions are stored contiguously in a flat table (`transitions[state * 256 + byte]`),
/// separated from per-state metadata to enable zero-copy sharing across partitions via `Arc`.
#[derive(Debug, Clone)]
pub struct FlatDfa {
    pub states: Vec<FlatDfaState>,
    pub start_state: usize,
    /// Flat transition table: `transitions[state * 256 + byte] = target_state`.
    /// Shared via `Arc` to avoid 35MB duplication per partition.
    pub transitions: Arc<[u32]>,
}

const BYTE_COLUMN_HASH_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

/// Transition-only data needed to construct the shared equivalence base.
///
/// This is created lazily only when an unsimplified L2P partition is built.
/// The dense transition table already exists for terminal construction; we
/// derive byte-column hashes and self-loop masks from the tokenizer's sparse
/// edge rows, then retain exact full-column checks for hash collisions.
pub(crate) struct FlatTransitionCache {
    pub(crate) transitions: Arc<[u32]>,
    pub(crate) byte_to_class: [u8; 256],
    pub(crate) self_loop_bytes: Arc<[U8Set]>,
}

fn byte_classes_from_column_hashes(
    transitions: &[u32],
    num_states: usize,
    column_hashes: &[u64; 256],
) -> [u8; 256] {
    let mut sorted_indices: [u8; 256] = std::array::from_fn(|i| i as u8);
    sorted_indices.sort_unstable_by_key(|&byte| column_hashes[byte as usize]);

    let mut byte_to_class = [0u8; 256];
    let mut next_class = 0u8;
    byte_to_class[sorted_indices[0] as usize] = 0;
    for i in 1..256 {
        let current = sorted_indices[i];
        let hash = column_hashes[current as usize];
        if hash != column_hashes[sorted_indices[i - 1] as usize] {
            next_class += 1;
            byte_to_class[current as usize] = next_class;
            continue;
        }

        // A hash match is only a candidate. The final relation remains the
        // exact equality of the 256-byte transition columns.
        let mut assigned = false;
        for j in (0..i).rev() {
            let previous = sorted_indices[j];
            if column_hashes[previous as usize] != hash {
                break;
            }
            let same = (0..num_states).all(|state| {
                let base = state * 256;
                transitions[base + current as usize] == transitions[base + previous as usize]
            });
            if same {
                byte_to_class[current as usize] = byte_to_class[previous as usize];
                assigned = true;
                break;
            }
        }
        if !assigned {
            next_class += 1;
            byte_to_class[current as usize] = next_class;
        }
    }
    byte_to_class
}

pub(crate) fn derive_flat_transition_cache(
    tokenizer: &Tokenizer,
    transitions: Arc<[u32]>,
) -> FlatTransitionCache {
    let num_states = tokenizer.num_states() as usize;
    let dead = u32::MAX;
    assert_eq!(transitions.len(), num_states * 256);

    // h(v_0..v_n) = Σ v_i M^(n-1-i). Starting from the all-dead
    // column, each sparse edge changes just one term in that sum.
    let mut row_weight = vec![0u64; num_states];
    let mut power = 1u64;
    for state in (0..num_states).rev() {
        row_weight[state] = power;
        power = power.wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER);
    }
    let mut all_dead_hash = 0u64;
    for _ in 0..num_states {
        all_dead_hash = all_dead_hash
            .wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
            .wrapping_add(dead as u64);
    }
    let mut column_hashes = [all_dead_hash; 256];
    let mut self_loop_bytes = Vec::with_capacity(num_states);

    for state in 0..num_states {
        let mut self_loops = U8Set::empty();
        let base = state * 256;
        for (byte, target) in tokenizer.transitions_from(state as u32) {
            let actual = transitions[base + byte as usize];
            debug_assert_eq!(actual, target);
            let delta = (target as u64).wrapping_sub(dead as u64);
            column_hashes[byte as usize] = column_hashes[byte as usize]
                .wrapping_add(delta.wrapping_mul(row_weight[state]));
            if target == state as u32 {
                self_loops.insert(byte);
            }
        }
        self_loop_bytes.push(self_loops);
    }

    FlatTransitionCache {
        byte_to_class: byte_classes_from_column_hashes(&transitions, num_states, &column_hashes),
        transitions,
        self_loop_bytes: Arc::from(self_loop_bytes),
    }
}


pub(crate) fn compute_byte_classes(dfa: &FlatDfa) -> [u8; 256] {
    let mut column_hashes = [0u64; 256];
    for row in dfa.transitions.chunks_exact(256) {
        for (hash, &target) in column_hashes.iter_mut().zip(row) {
            *hash = hash
                .wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
                .wrapping_add(target as u64);
        }
    }
    byte_classes_from_column_hashes(&dfa.transitions, dfa.states.len(), &column_hashes)
}


/// Exact byte congruence of the selected terminal languages.
///
/// The shared lexer can retain inactive product/component coordinates even
/// after its output labels are filtered. Recompile the active expressions as
/// independent deterministic components and compare byte transition columns
/// across all of their residual states. Two bytes in one class therefore have
/// identical action from every residual of every active terminal language.
/// Replacing one by the other preserves active-terminal match positions and
/// residuals; grammar follow constraints only filter terminal labels later.
fn compute_active_terminal_language_byte_classes_combined(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
) -> Option<[u8; 256]> {
    let active_exprs = active_groups
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
        .map(|terminal| tokenizer.terminal_expr(terminal).cloned())
        .collect::<Option<Vec<_>>>()?;
    if active_exprs.is_empty() {
        return Some([0u8; 256]);
    }
    let partitions = (0..active_exprs.len() as u32).collect::<Vec<_>>();
    let regex = crate::automata::lexer::compile::build_regex_partitioned_with_adaptive(
        &active_exprs,
        &partitions,
        false,
    );
    let terminal_count = active_exprs.len() as u32;
    let active_tokenizer = regex.into_tokenizer(
        terminal_count,
        Some(Arc::from(active_exprs.into_boxed_slice())),
    );
    let active_view = TokenizerView::new(&active_tokenizer);
    Some(compute_byte_classes(active_view.dfa()))
}

fn compute_byte_classes_for_independent_dfas(dfas: &[DFA]) -> [u8; 256] {
    let num_states = dfas.iter().map(DFA::num_states).sum::<usize>();
    let mut transitions = Vec::with_capacity(num_states.saturating_mul(256));
    let mut column_hashes = [0u64; 256];
    for dfa in dfas {
        for state in 0..dfa.num_states() {
            let row = build_transition_table(dfa.transitions(state as u32));
            for (hash, &target) in column_hashes.iter_mut().zip(row.iter()) {
                *hash = hash
                    .wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
                    .wrapping_add(target as u64);
            }
            transitions.extend_from_slice(&row);
        }
    }
    byte_classes_from_column_hashes(&transitions, num_states, &column_hashes)
}

fn byte_class_relations_match(left: &[u8; 256], right: &[u8; 256]) -> bool {
    (0..256).all(|first| {
        (0..256).all(|second| {
            (left[first] == left[second]) == (right[first] == right[second])
        })
    })
}

fn active_terminal_exprs(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
) -> Option<Vec<crate::automata::regex::Expr>> {
    active_groups
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
        .map(|terminal| tokenizer.terminal_expr(terminal).cloned())
        .collect::<Option<Vec<_>>>()
}

fn compute_active_terminal_language_byte_classes_independent(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
) -> Option<[u8; 256]> {
    let active_exprs = active_terminal_exprs(tokenizer, active_groups)?;
    if active_exprs.is_empty() {
        return Some([0u8; 256]);
    }
    let dfas = active_exprs
        .par_iter()
        .map(compile_terminal_expr_dfa)
        .collect::<Vec<_>>();
    Some(compute_byte_classes_for_independent_dfas(&dfas))
}

pub(crate) fn compute_active_terminal_language_byte_classes(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
) -> Option<[u8; 256]> {
    let use_independent = std::env::var("GLRMASK_ACTIVE_LANGUAGE_BYTE_CLASSES_INDEPENDENT")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false);
    if !use_independent {
        return compute_active_terminal_language_byte_classes_combined(
            tokenizer,
            active_groups,
        );
    }

    let independent =
        compute_active_terminal_language_byte_classes_independent(tokenizer, active_groups)?;
    if std::env::var_os("GLRMASK_ACTIVE_LANGUAGE_BYTE_CLASSES_STRICT_REFERENCE").is_some() {
        let reference =
            compute_active_terminal_language_byte_classes_combined(tokenizer, active_groups)?;
        assert!(
            byte_class_relations_match(&independent, &reference),
            "independent active-terminal byte congruence differs from combined-tokenizer reference",
        );
    }
    Some(independent)
}


#[inline]
fn state_has_active_language(state: &FlatDfaState) -> bool {
    !state.finalizers.is_empty() || !state.possible_future_group_ids.is_empty()
}

/// Erase topology whose target language is empty under already-filtered state
/// metadata. Missing edges and edges into an active-dead state are the same
/// transition in the active-terminal language. Rows of active-dead sources are
/// empty for the same reason.
fn project_filtered_transition_topology(
    states: &[FlatDfaState],
    transitions: &[u32],
) -> Arc<[u32]> {
    assert_eq!(transitions.len(), states.len() * 256);
    let active_language = states
        .iter()
        .map(state_has_active_language)
        .collect::<Vec<_>>();
    let mut projected = transitions.to_vec();
    for source in 0..states.len() {
        let row = &mut projected[source * 256..(source + 1) * 256];
        if !active_language[source] {
            row.fill(u32::MAX);
            continue;
        }
        for target in row {
            if *target != u32::MAX && !active_language[*target as usize] {
                *target = u32::MAX;
            }
        }
    }
    Arc::from(projected)
}

impl FlatDfa {
    /// Get the transition target for a given state and byte.
    #[inline]
    pub fn trans(&self, state: usize, byte: usize) -> u32 {
        self.transitions[state * 256 + byte]
    }

    /// Get the 256-entry transition slice for a given state.
    #[inline]
    pub fn transitions_for(&self, state: usize) -> &[u32] {
        let base = state * 256;
        &self.transitions[base..base + 256]
    }
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let start_state = tokenizer.start_state() as usize;
        let num_states = tokenizer.num_states() as usize;
        let mut transitions = vec![u32::MAX; num_states * 256];
        let states: Vec<FlatDfaState> = (0..num_states)
            .map(|i| {
                let base = i * 256;
                for (byte, target) in tokenizer.transitions_from(i as u32) {
                    transitions[base + byte as usize] = target;
                }
                let finalizers = collect_group_ids(tokenizer.matched_terminals_iter(i as u32));
                let possible_future_group_ids =
                    collect_group_ids(tokenizer.possible_future_terminals_iter(i as u32));

                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();

        FlatDfa {
            states,
            start_state,
            transitions: Arc::from(transitions),
        }
    }

    /// Build a DFA view of the active-terminal language. Besides filtering
    /// finalizers/futures, transitions from or into states with empty active
    /// language are canonicalized to dead.
    pub fn from_tokenizer_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        let start_state = tokenizer.start_state() as usize;
        let num_states = tokenizer.num_states() as usize;
        let mut transitions = vec![u32::MAX; num_states * 256];
        let states: Vec<FlatDfaState> = (0..num_states)
            .map(|i| {
                let base = i * 256;
                for (byte, target) in tokenizer.transitions_from(i as u32) {
                    transitions[base + byte as usize] = target;
                }
                let finalizers = collect_filtered_group_ids(
                    tokenizer.matched_terminals_iter(i as u32),
                    active_groups,
                );
                let possible_future_group_ids = collect_filtered_group_ids(
                    tokenizer.possible_future_terminals_iter(i as u32),
                    active_groups,
                );

                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();

        let transitions = project_filtered_transition_topology(&states, &transitions);
        FlatDfa {
            states,
            start_state,
            transitions,
        }
    }

    /// Build a FlatDfa using a pre-built flat transition table, sharing the
    /// transition data via Arc. Copies all finalizers/futures without filtering.
    pub fn from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
    ) -> Self {
        let start_state = tokenizer.start_state() as usize;
        let states: Vec<FlatDfaState> = (0..tokenizer.num_states() as usize)
            .map(|i| {
                let finalizers = collect_group_ids(tokenizer.matched_terminals_iter(i as u32));
                let possible_future_group_ids =
                    collect_group_ids(tokenizer.possible_future_terminals_iter(i as u32));
                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();
        FlatDfa { states, start_state, transitions: Arc::clone(flat_trans) }
    }

    /// Build an active-terminal DFA view from a shared raw transition table.
    /// The raw table is projected to dead for active-dead sources and targets.
    pub fn from_flat_trans_filtered(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> Self {
        let start_state = tokenizer.start_state() as usize;
        let states: Vec<FlatDfaState> = (0..tokenizer.num_states() as usize)
            .map(|i| {
                let finalizers = collect_filtered_group_ids(
                    tokenizer.matched_terminals_iter(i as u32),
                    active_groups,
                );
                let possible_future_group_ids = collect_filtered_group_ids(
                    tokenizer.possible_future_terminals_iter(i as u32),
                    active_groups,
                );
                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();
        let transitions = project_filtered_transition_topology(&states, flat_trans);
        FlatDfa {
            states,
            start_state,
            transitions,
        }
    }

}

/// A thin wrapper around glrmask's `Tokenizer` that exposes the flattened DFA.
///
/// The equivalence-analysis code calls `dfa()` and accesses `.states` and
/// `.start_state` directly.
/// This wrapper pre-extracts all data into `FlatDfa` on construction.
pub struct TokenizerView {
    pub flat_dfa: FlatDfa,
}

impl TokenizerView {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_tokenizer(tokenizer),
        }
    }

    /// Build a view that filters finalizers and futures to only active groups.
    pub fn new_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_tokenizer_filtered(tokenizer, active_groups),
        }
    }

    /// Build a view using a pre-built shared flat transition table (no group filtering).
    /// Shares transition data via Arc — zero-copy.
    pub fn new_from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
    ) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_flat_trans(flat_trans, tokenizer),
        }
    }

    /// Build an active-terminal view using a pre-built shared raw transition
    /// table. The projected transition table is owned by this view.
    pub fn new_filtered_from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_flat_trans_filtered(flat_trans, tokenizer, active_groups),
        }
    }

    pub fn dfa(&self) -> &FlatDfa {
        &self.flat_dfa
    }

    pub fn initial_state_id(&self) -> usize {
        self.flat_dfa.start_state
    }

    /// Build a filtered quotient directly from the shared raw transition table.
    /// This is used only after a raw-coordinate congruence certificate, so one
    /// representative row per state class preserves every vocabulary-byte walk.
    pub(crate) fn new_filtered_quotient_from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
        state_map: &ManyToOneIdMap,
    ) -> (Self, f64) {
        let raw_states = tokenizer.num_states() as usize;
        assert_eq!(flat_trans.len(), raw_states * 256, "invalid raw transition table");
        assert_eq!(state_map.original_to_internal.len(), raw_states, "invalid state map");
        let quotient_states = state_map.internal_to_originals.len();
        let mut transitions = vec![u32::MAX; quotient_states * 256];
        let mut states = Vec::with_capacity(quotient_states);
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let filter_started_at = profile_timing.then(Instant::now);
        for (internal, &representative) in state_map.representative_original_ids.iter().enumerate() {
            let representative = representative as usize;
            assert!(representative < raw_states, "invalid quotient representative");
            let finalizers = collect_filtered_group_ids(
                tokenizer.matched_terminals_iter(representative as u32),
                active_groups,
            );
            let possible_future_group_ids = collect_filtered_group_ids(
                tokenizer.possible_future_terminals_iter(representative as u32),
                active_groups,
            );
            states.push(FlatDfaState {
                finalizers,
                possible_future_group_ids,
            });
            let raw_base = representative * 256;
            let quotient_base = internal * 256;
            for byte in 0..256usize {
                let target = flat_trans[raw_base + byte];
                transitions[quotient_base + byte] = if target == u32::MAX {
                    u32::MAX
                } else {
                    let mapped = state_map.original_to_internal[target as usize];
                    assert_ne!(mapped, u32::MAX, "quotient target must be mapped");
                    mapped
                };
            }
        }
        let transitions = project_filtered_transition_topology(&states, &transitions);
        let start_state = state_map.original_to_internal[tokenizer.start_state() as usize];
        assert_ne!(start_state, u32::MAX, "quotient start state must be mapped");
        (
            Self {
                flat_dfa: FlatDfa {
                    states,
                    start_state: start_state as usize,
                    transitions,
                },
            },
            filter_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
        )
    }

    pub(crate) fn new_filtered_quotient_from_flat_trans_with_observation_cache(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
        state_map: &ManyToOneIdMap,
        raw_observation_ids: &[u32],
        observation_representatives: &[u32],
    ) -> (Self, f64) {
        let raw_states = tokenizer.num_states() as usize;
        assert_eq!(flat_trans.len(), raw_states * 256, "invalid raw transition table");
        assert_eq!(state_map.original_to_internal.len(), raw_states, "invalid state map");
        assert_eq!(raw_observation_ids.len(), raw_states, "invalid observation IDs");

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let observation_states = observation_representatives
            .iter()
            .map(|&representative| {
                let representative = representative as usize;
                assert!(representative < raw_states, "invalid observation representative");
                FlatDfaState {
                    finalizers: collect_filtered_group_ids(
                        tokenizer.matched_terminals_iter(representative as u32),
                        active_groups,
                    ),
                    possible_future_group_ids: collect_filtered_group_ids(
                        tokenizer.possible_future_terminals_iter(representative as u32),
                        active_groups,
                    ),
                }
            })
            .collect::<Vec<_>>();

        let quotient_states = state_map.internal_to_originals.len();
        let mut transitions = vec![u32::MAX; quotient_states * 256];
        let mut states = Vec::with_capacity(quotient_states);
        for (internal, &representative) in state_map.representative_original_ids.iter().enumerate() {
            let representative = representative as usize;
            let observation = raw_observation_ids[representative] as usize;
            states.push(
                observation_states
                    .get(observation)
                    .expect("invalid observation class")
                    .clone(),
            );
            let raw_base = representative * 256;
            let quotient_base = internal * 256;
            for byte in 0..256usize {
                let target = flat_trans[raw_base + byte];
                transitions[quotient_base + byte] = if target == u32::MAX {
                    u32::MAX
                } else {
                    state_map.original_to_internal[target as usize]
                };
            }
        }
        let transitions = project_filtered_transition_topology(&states, &transitions);
        let start_state = state_map.original_to_internal[tokenizer.start_state() as usize];
        assert_ne!(start_state, u32::MAX, "quotient start state must be mapped");
        (
            Self {
                flat_dfa: FlatDfa {
                    states,
                    start_state: start_state as usize,
                    transitions,
                },
            },
            started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
        )
    }

    /// Build the filtered quotient metadata together with a precompressed
    /// finite-vocabulary transition base. The returned view deliberately has
    /// no dense 256-column transition table; callers must use the returned
    /// compatible base for all token walks.
    pub(crate) fn new_filtered_quotient_from_flat_trans_with_observation_cache_and_relevant_base(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
        state_map: &ManyToOneIdMap,
        raw_observation_ids: &[u32],
        observation_representatives: &[u32],
        relevant_bytes: &[bool; 256],
    ) -> Option<(Self, super::vocab::fast::SharedVocabDfaBase, f64)> {
        let raw_states = tokenizer.num_states() as usize;
        if flat_trans.len() != raw_states * 256
            || state_map.original_to_internal.len() != raw_states
            || raw_observation_ids.len() != raw_states
        {
            return None;
        }
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let observation_states = observation_representatives
            .iter()
            .map(|&representative| {
                let representative = representative as usize;
                if representative >= raw_states {
                    return None;
                }
                Some(FlatDfaState {
                    finalizers: collect_filtered_group_ids(
                        tokenizer.matched_terminals_iter(representative as u32),
                        active_groups,
                    ),
                    possible_future_group_ids: collect_filtered_group_ids(
                        tokenizer.possible_future_terminals_iter(representative as u32),
                        active_groups,
                    ),
                })
            })
            .collect::<Option<Vec<_>>>()?;

        let mut states = Vec::with_capacity(state_map.internal_to_originals.len());
        for &representative in &state_map.representative_original_ids {
            let representative = representative as usize;
            if representative >= raw_states {
                return None;
            }
            let observation = raw_observation_ids[representative] as usize;
            states.push(observation_states.get(observation)?.clone());
        }
        let start_state = state_map.original_to_internal[tokenizer.start_state() as usize];
        if start_state == u32::MAX {
            return None;
        }
        let active_dead_classes = states
            .iter()
            .map(|state| !state_has_active_language(state))
            .collect::<Vec<_>>();
        let base = super::vocab::fast::SharedVocabDfaBase::build_from_raw_quotient_for_relevant_bytes(
            flat_trans,
            state_map,
            relevant_bytes,
            Some(&active_dead_classes),
        )?;
        Some((
            Self {
                flat_dfa: FlatDfa {
                    states,
                    start_state: start_state as usize,
                    transitions: Arc::from(Vec::<u32>::new()),
                },
            },
            base,
            started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
        ))
    }

    /// Verify that `state_map` is an output-labelled right congruence for every
    /// byte that may occur in the active vocabulary. This is the exact condition
    /// needed to evaluate token paths in the quotient DFA.
    pub(crate) fn is_relevant_byte_congruent(
        &self,
        state_map: &ManyToOneIdMap,
        relevant_bytes: &[bool; 256],
    ) -> bool {
        let dfa = self.dfa();
        if state_map.original_to_internal.len() != dfa.states.len() {
            return false;
        }
        for (internal, members) in state_map.internal_to_originals.iter().enumerate() {
            let Some(&representative) = state_map.representative_original_ids.get(internal) else {
                return false;
            };
            let representative = representative as usize;
            if representative >= dfa.states.len()
                || state_map.original_to_internal[representative] != internal as u32
            {
                return false;
            }
            let representative_state = &dfa.states[representative];
            for &raw_state in members {
                let raw_state = raw_state as usize;
                if raw_state >= dfa.states.len()
                    || state_map.original_to_internal[raw_state] != internal as u32
                {
                    return false;
                }
                let state = &dfa.states[raw_state];
                if state.finalizers != representative_state.finalizers
                    || state.possible_future_group_ids
                        != representative_state.possible_future_group_ids
                {
                    return false;
                }
                for (byte, &relevant) in relevant_bytes.iter().enumerate() {
                    if !relevant {
                        continue;
                    }
                    let representative_target = dfa.trans(representative, byte);
                    let target = dfa.trans(raw_state, byte);
                    let mapped_representative = if representative_target == u32::MAX {
                        u32::MAX
                    } else {
                        state_map.original_to_internal[representative_target as usize]
                    };
                    let mapped_target = if target == u32::MAX {
                        u32::MAX
                    } else {
                        state_map.original_to_internal[target as usize]
                    };
                    if mapped_target != mapped_representative {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Materialize the exact active-byte quotient after a congruence check. The
    /// returned view owns only one row and observation pair per state class.
    pub(crate) fn quotient_by_state_map(&self, state_map: &ManyToOneIdMap) -> Self {
        let dfa = self.dfa();
        let quotient_states = state_map.internal_to_originals.len();
        let mut transitions = vec![u32::MAX; quotient_states * 256];
        let mut states = Vec::with_capacity(quotient_states);
        for (internal, &representative) in state_map.representative_original_ids.iter().enumerate() {
            let representative = representative as usize;
            assert!(representative < dfa.states.len(), "invalid quotient representative");
            let source = &dfa.states[representative];
            states.push(FlatDfaState {
                finalizers: source.finalizers.clone(),
                possible_future_group_ids: source.possible_future_group_ids.clone(),
            });
            let base = internal * 256;
            for byte in 0..256usize {
                let target = dfa.trans(representative, byte);
                transitions[base + byte] = if target == u32::MAX {
                    u32::MAX
                } else {
                    let mapped = state_map.original_to_internal[target as usize];
                    assert_ne!(mapped, u32::MAX, "quotient target must be mapped");
                    mapped
                };
            }
        }
        let transitions = project_filtered_transition_topology(&states, &transitions);
        let start_state = state_map.original_to_internal[dfa.start_state];
        assert_ne!(start_state, u32::MAX, "quotient start state must be mapped");
        Self {
            flat_dfa: FlatDfa {
                states,
                start_state: start_state as usize,
                transitions,
            },
        }
    }

}


#[cfg(test)]
mod sparse_transition_cache_tests {
    use super::*;

    #[test]
    fn filtered_byte_classes_ignore_inactive_only_transition_topology() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_monolithic as build_regex;

        fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
            let terminal_count = expressions.len() as u32;
            build_regex(&expressions).into_tokenizer(
                terminal_count,
                Some(Arc::from(expressions.into_boxed_slice())),
            )
        }

        fn same_partition(left: &[u8; 256], right: &[u8; 256]) -> bool {
            (0..256).all(|a| {
                (0..256).all(|b| (left[a] == left[b]) == (right[a] == right[b]))
            })
        }

        let with_inactive = tokenizer(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ]);
        let active_only = tokenizer(vec![Expr::U8Seq(b"ab".to_vec())]);

        let filtered = TokenizerView::new_filtered(&with_inactive, &[true, false, false]);
        let reference = TokenizerView::new(&active_only);
        let filtered_classes = compute_byte_classes(filtered.dfa());
        let reference_classes = compute_byte_classes(reference.dfa());

        assert_eq!(filtered.dfa().trans(filtered.dfa().start_state, b'x' as usize), u32::MAX);
        assert_eq!(filtered.dfa().trans(filtered.dfa().start_state, b'y' as usize), u32::MAX);
        assert_eq!(filtered_classes[b'x' as usize], filtered_classes[b'y' as usize]);
        assert!(
            same_partition(&filtered_classes, &reference_classes),
            "inactive-only lexer topology changed the active-language byte congruence",
        );
    }

    #[test]
    fn sparse_column_hashes_preserve_exact_byte_classes() {
        let num_states = 4usize;
        let dead = u32::MAX;
        let mut transitions = vec![dead; num_states * 256];
        let rows = [
            vec![(b'a', 1), (b'b', 1), (b'c', 2)],
            vec![(b'a', 2), (b'b', 2), (b'd', 3)],
            vec![(b'a', 3), (b'b', 3), (b'c', 1)],
            vec![(b'a', 3), (b'b', 3), (b'd', 1)],
        ];
        let mut row_weight = vec![0u64; num_states];
        let mut power = 1u64;
        for state in (0..num_states).rev() {
            row_weight[state] = power;
            power = power.wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER);
        }
        let mut all_dead_hash = 0u64;
        for _ in 0..num_states {
            all_dead_hash = all_dead_hash
                .wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
                .wrapping_add(dead as u64);
        }
        let mut sparse_hashes = [all_dead_hash; 256];
        for (state, row) in rows.iter().enumerate() {
            for &(byte, target) in row {
                transitions[state * 256 + byte as usize] = target;
                sparse_hashes[byte as usize] = sparse_hashes[byte as usize].wrapping_add(
                    (target as u64)
                        .wrapping_sub(dead as u64)
                        .wrapping_mul(row_weight[state]),
                );
            }
        }
        let dfa = FlatDfa {
            states: (0..num_states)
                .map(|_| FlatDfaState {
                    finalizers: Vec::new(),
                    possible_future_group_ids: Vec::new(),
                })
                .collect(),
            start_state: 0,
            transitions: Arc::from(transitions),
        };
        assert_eq!(
            byte_classes_from_column_hashes(&dfa.transitions, num_states, &sparse_hashes),
            compute_byte_classes(&dfa),
        );
    }
}
