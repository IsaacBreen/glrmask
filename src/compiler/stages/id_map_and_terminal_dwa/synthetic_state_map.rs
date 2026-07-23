//! Exact full-lexer to synthesized-lexer state certification.
//!
//! The parser DWA observes at most one vocabulary token at a time. Put the two
//! independently built lexers in one disjoint epsilon union, run the existing
//! exact K-bounded residual observer, and require every full residual state to
//! share a class with at least one synthesized residual state.

use std::sync::Arc;

use crate::Vocab;
use crate::automata::lexer::compile::{
    VocabularyRepeatHorizonCache, structural_pair_component_count,
};
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::l2p::equivalence_analysis::compat::TokenizerView;
use super::l2p::equivalence_analysis::state::fast::find_state_equivalence_classes_with_state_resets;
use super::l2p::equivalence_analysis::state_equivalence::max_length::{
    self, MaxLengthMode,
};

#[derive(Debug, Clone)]
pub(crate) struct CertifiedFullToSynthesizedStateMap {
    /// Full raw lexer state -> synthesized raw lexer state.
    pub(crate) full_to_synthesized: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct CertifiedVocabularyExactStateCandidates {
    primary: Vec<u32>,
    candidates_by_full_state: Vec<Arc<[u32]>>,
}

impl CertifiedVocabularyExactStateCandidates {
    pub(crate) fn primary(&self) -> &[u32] {
        &self.primary
    }

    pub(crate) fn visit_candidates(
        &self,
        full_state: u32,
        mut visit: impl FnMut(u32) -> bool,
    ) -> bool {
        self.candidates_by_full_state[full_state as usize]
            .iter()
            .copied()
            .any(&mut visit)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SynthesizedTerminalExpressions {
    pub(crate) expressions: Vec<Expr>,
    pub(crate) changed_terminals: Vec<u32>,
}

pub(crate) struct MaterializedActiveTokenizer {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) full_to_active: CertifiedFullToSynthesizedStateMap,
    pub(crate) build_ms: f64,
}

/// Turn an exact active-language state quotient into the tokenizer actually
/// consumed by an L1/L2P branch. Previously the quotient was only threaded as
/// an initial ID map, leaving downstream token replay and profile construction
/// on the full raw tokenizer. Materialization removes that hidden raw-state
/// cost while retaining an explicit lift back to the source coordinate.
pub(crate) fn materialize_active_tokenizer(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    active_terminals: &[bool],
    mut state_map: ManyToOneIdMap,
) -> Option<MaterializedActiveTokenizer> {
    if state_map.original_to_internal.len() != tokenizer.num_states() as usize
        || state_map.num_internal_ids() >= tokenizer.num_states()
    {
        return None;
    }
    let quotient_states = state_map.num_internal_ids() as usize;
    let started_at = std::time::Instant::now();
    let statistic = max_length::cached_statistic(vocab);

    let (compact, full_to_synthesized) = if tokenizer.has_epsilon_transitions() {
        // NFA state equivalence is defined over epsilon-closed configurations,
        // not raw physical edges. Materialize that exact powerset coordinate,
        // seeded by the already-computed active-language quotient, instead of
        // attempting an invalid raw-edge quotient.
        let view = super::l2p::equivalence_analysis::state_equivalence::nfa::build_relevant_powerset_view(
            tokenizer,
            statistic.relevant_bytes(),
            Some(active_terminals),
            Some(&state_map),
        );
        let finalizers = view
            .states
            .iter()
            .map(|state| state.finalizers.clone())
            .collect::<Vec<_>>();
        let futures = view
            .states
            .iter()
            .map(|state| state.possible_future_group_ids.clone())
            .collect::<Vec<_>>();
        let (compact, old_to_new) = tokenizer.materialize_deterministic_view(
            view.start_state,
            &finalizers,
            &futures,
            &view.edge_offsets,
            &view.edges,
            active_terminals,
        )?;
        let full_to_synthesized = view
            .raw_start_to_view
            .iter()
            .map(|&view_state| {
                old_to_new
                    .get(view_state as usize)
                    .copied()
                    .filter(|&state| state != u32::MAX)
            })
            .collect::<Option<Vec<_>>>()?;
        (compact, full_to_synthesized)
    } else {
        let start = tokenizer.start_state();
        state_map.isolate_original(start);
        state_map.reorder_internal_by_representative_key(|representative| {
            (representative != start, representative)
        });
        if state_map.original_to_internal[start as usize] != 0 {
            return None;
        }
        let compact = tokenizer.materialize_active_quotient(
            &state_map.original_to_internal,
            &state_map.representative_original_ids,
            active_terminals,
            statistic.relevant_bytes(),
        )?;
        (compact, state_map.original_to_internal)
    };
    let source_states = tokenizer.num_states() as usize;
    let compact_states = compact.num_states() as usize;
    if compact_states >= source_states
        || source_states.saturating_sub(compact_states) < 1_024
        || compact_states.saturating_mul(10) > source_states.saturating_mul(9)
        // Epsilon-view materialization may expand an exact quotient back into
        // many powerset states. Downstream equivalence can consume the quotient
        // directly, so a large expansion is negative work: it costs a build
        // here and increases the later state-token product. Retain near-size
        // materializations, which remove raw replay overhead, but fall back to
        // the exact initial state map when the deterministic coordinate grows
        // by more than 25 percent.
        || compact_states.saturating_mul(4) > quotient_states.saturating_mul(5)
    {
        return None;
    }
    let build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    Some(MaterializedActiveTokenizer {
        tokenizer: compact,
        full_to_active: CertifiedFullToSynthesizedStateMap {
            full_to_synthesized,
        },
        build_ms,
    })
}

pub(crate) fn profile_dispatch_component_activity(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    branch_label: &str,
) {
    if std::env::var_os("GLRMASK_PROFILE_INACTIVE_COMPONENTS").is_none() {
        return;
    }
    let Some(components) = tokenizer.disjoint_dispatch_components() else {
        eprintln!(
            "[glrmask/profile][dispatch_component_activity] branch={} available=false",
            branch_label,
        );
        return;
    };
    let active_terminal_ids = active_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| active.then_some(terminal as u32))
        .collect::<Vec<_>>();
    eprintln!(
        "[glrmask/profile][dispatch_component_activity] branch={} available=true components={} active_terminal_count={} active_terminal_ids={:?}",
        branch_label,
        components.len(),
        active_terminal_ids.len(),
        active_terminal_ids,
    );
    for (component, states) in components.iter().enumerate() {
        let mut observed = std::collections::BTreeSet::<u32>::new();
        for &state in states {
            observed.extend(tokenizer.matched_terminals_iter(state));
            observed.extend(tokenizer.possible_future_terminals_iter(state));
        }
        let observed = observed.into_iter().collect::<Vec<_>>();
        let active_observed = observed
            .iter()
            .copied()
            .filter(|&terminal| {
                active_terminals
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        eprintln!(
            "[glrmask/profile][dispatch_component_activity_component] branch={} component={} states={} observed_terminals={:?} active_observed_terminals={:?} active={}",
            branch_label,
            component,
            states.len(),
            observed,
            active_observed,
            !active_observed.is_empty(),
        );
    }
}

/// Collapse every closed dispatch component that can never observe an active
/// terminal. Active components remain state-for-state identity classes.
///
/// Once a scanner residual is inside an inactive component, no future byte
/// string can produce an active match or active continuation because byte and
/// epsilon edges do not leave the component. All such residuals therefore
/// have the same active-language behavior: the empty observation.
pub(crate) fn inactive_dispatch_component_state_map(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
) -> Option<ManyToOneIdMap> {
    if std::env::var_os("GLRMASK_DISABLE_INACTIVE_COMPONENT_STATE_MAP").is_some() {
        return None;
    }
    let components = tokenizer.disjoint_dispatch_components()?;
    let num_states = tokenizer.num_states() as usize;
    let start = tokenizer.start_state() as usize;
    let mut original_to_internal = vec![u32::MAX; num_states];
    let mut representatives = Vec::new();
    let add_class = |original_to_internal: &mut [u32],
                     representatives: &mut Vec<u32>,
                     states: &[u32]| {
        let internal = representatives.len() as u32;
        representatives.push(states[0]);
        for &state in states {
            original_to_internal[state as usize] = internal;
        }
    };

    add_class(
        &mut original_to_internal,
        &mut representatives,
        &[start as u32],
    );
    let mut inactive_states = Vec::new();
    for states in components {
        let active = states.iter().any(|&state| {
            let (matched, future) = filtered_state_label(tokenizer, state, Some(active_terminals));
            !matched.is_empty() || !future.is_empty()
        });
        if active {
            for state in states {
                add_class(
                    &mut original_to_internal,
                    &mut representatives,
                    &[state],
                );
            }
        } else {
            inactive_states.extend(states);
        }
    }
    if !inactive_states.is_empty() {
        add_class(
            &mut original_to_internal,
            &mut representatives,
            &inactive_states,
        );
    }
    // Nullable-start isolation and structural augmentation can retain states
    // outside the live dispatcher components. Keep them singleton rather than
    // extending the component proof beyond its verified domain.
    for state in 0..num_states {
        if original_to_internal[state] == u32::MAX {
            add_class(
                &mut original_to_internal,
                &mut representatives,
                &[state as u32],
            );
        }
    }

    Some(ManyToOneIdMap::from_original_to_internal_with_representatives(
        original_to_internal,
        representatives.len() as u32,
        representatives,
    ))
}

const BYTE_COLUMN_HASH_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

fn byte_columns_equal(
    full: &Tokenizer,
    synthesized: &Tokenizer,
    left: u8,
    right: u8,
) -> bool {
    (0..synthesized.num_states())
        .all(|state| synthesized.step(state, left) == synthesized.step(state, right))
        && (0..full.num_states())
            .all(|state| full.step(state, left) == full.step(state, right))
}

/// Return one exact representative for every transition-column class among
/// vocabulary-relevant bytes across both deterministic lexers.
///
/// Hashes only identify candidate classes. Every hash match is verified by a
/// complete row-wise comparison, so the resulting alphabet quotient is exact.
fn exact_combined_byte_representatives(
    full: &Tokenizer,
    synthesized: &Tokenizer,
    relevant_bytes: &[u8],
) -> Vec<u8> {
    let mut relevant = relevant_bytes.to_vec();
    relevant.sort_unstable();
    relevant.dedup();
    if relevant.len() <= 1 {
        return relevant;
    }

    let total_states = full.num_states() as usize + synthesized.num_states() as usize;
    let mut row_weights = vec![0u64; total_states];
    let mut power = 1u64;
    for row in (0..total_states).rev() {
        row_weights[row] = power;
        power = power.wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER);
    }
    let dead = u32::MAX as u64;
    let mut all_dead_hash = 0u64;
    for _ in 0..total_states {
        all_dead_hash = all_dead_hash
            .wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
            .wrapping_add(dead);
    }
    let mut hashes = [all_dead_hash; 256];
    let mut row = 0usize;
    for tokenizer in [synthesized, full] {
        for state in 0..tokenizer.num_states() {
            let weight = row_weights[row];
            for (byte, target) in tokenizer.transitions_from(state) {
                let delta = (target as u64).wrapping_sub(dead);
                hashes[byte as usize] =
                    hashes[byte as usize].wrapping_add(delta.wrapping_mul(weight));
            }
            row += 1;
        }
    }

    relevant.sort_unstable_by_key(|&byte| (hashes[byte as usize], byte));
    let mut representatives = Vec::new();
    let mut group_start = 0usize;
    while group_start < relevant.len() {
        let hash = hashes[relevant[group_start] as usize];
        let mut group_end = group_start + 1;
        while group_end < relevant.len() && hashes[relevant[group_end] as usize] == hash {
            group_end += 1;
        }

        let mut group_representatives = Vec::new();
        for &byte in &relevant[group_start..group_end] {
            if !group_representatives
                .iter()
                .any(|&representative| byte_columns_equal(full, synthesized, representative, byte))
            {
                group_representatives.push(byte);
            }
        }
        representatives.extend(group_representatives);
        group_start = group_end;
    }
    representatives.sort_unstable();
    representatives
}

fn collect_component_states(
    tokenizer: &Tokenizer,
    root: u32,
    claimed: &mut [bool],
) -> Option<Vec<u32>> {
    let mut states = Vec::new();
    let mut stack = vec![root];
    while let Some(state) = stack.pop() {
        let slot = claimed.get_mut(state as usize)?;
        if *slot {
            continue;
        }
        if tokenizer.state_has_epsilon_transitions(state) {
            return None;
        }
        *slot = true;
        states.push(state);
        stack.extend(tokenizer.transitions_from(state).map(|(_, target)| target));
    }
    states.sort_unstable();
    Some(states)
}

fn filtered_state_label(
    tokenizer: &Tokenizer,
    state: u32,
    active_terminals: Option<&[bool]>,
) -> (Vec<u32>, Vec<u32>) {
    let active = |terminal: u32| {
        active_terminals.is_none_or(|active| {
            active
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
        })
    };
    (
        tokenizer
            .matched_terminals_iter(state)
            .filter(|&terminal| active(terminal))
            .collect(),
        tokenizer
            .possible_future_terminals_iter(state)
            .filter(|&terminal| active(terminal))
            .collect(),
    )
}

fn map_deterministic_component_states(
    full: &Tokenizer,
    synthesized: &Tokenizer,
    full_states: &[u32],
    synthesized_states: &[u32],
    depth: usize,
    relevant_bytes: &[u8],
    active_terminals: Option<&[bool]>,
) -> Option<Vec<u32>> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(std::time::Instant::now);
    let mut label_intern = FxHashMap::<(Vec<u32>, Vec<u32>), u32>::default();
    let synthesized_labels = synthesized_states
        .iter()
        .map(|&state| {
            let label = filtered_state_label(synthesized, state, active_terminals);
            if let Some(&class) = label_intern.get(&label) {
                class
            } else {
                let class = label_intern.len() as u32;
                label_intern.insert(label, class);
                class
            }
        })
        .collect::<Vec<_>>();
    let full_labels = full_states
        .par_iter()
        .map(|&state| {
            let label = filtered_state_label(full, state, active_terminals);
            label_intern.get(&label).copied()
        })
        .collect::<Option<Vec<_>>>();
    let Some(full_labels) = full_labels else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] deterministic_certification_rejected stage=labels full_states={} synthesized_states={} depth={} relevant_bytes={} elapsed_ms={:.3}",
                full_states.len(),
                synthesized_states.len(),
                depth,
                relevant_bytes.len(),
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        return None;
    };

    let mut synthesized_position = vec![usize::MAX; synthesized.num_states() as usize];
    for (position, &state) in synthesized_states.iter().enumerate() {
        synthesized_position[state as usize] = position;
    }
    let mut full_position = vec![usize::MAX; full.num_states() as usize];
    for (position, &state) in full_states.iter().enumerate() {
        full_position[state as usize] = position;
    }

    let mut classes_synthesized = synthesized_labels.clone();
    let mut classes_full = full_labels.clone();
    let mut relevant_bytes = relevant_bytes.to_vec();
    relevant_bytes.sort_unstable();
    relevant_bytes.dedup();

    for current_depth in 0..depth {
        let mut intern = FxHashMap::<Vec<u32>, u32>::default();
        let mut signature = vec![0u32; 1 + relevant_bytes.len()];
        let mut next_synthesized = vec![0u32; synthesized_states.len()];
        for (position, &state) in synthesized_states.iter().enumerate() {
            signature[0] = synthesized_labels[position];
            for (byte_index, &byte) in relevant_bytes.iter().enumerate() {
                signature[byte_index + 1] = match synthesized.step(state, byte) {
                    Some(target) => {
                        let target_position = *synthesized_position.get(target as usize)?;
                        if target_position == usize::MAX {
                            return None;
                        }
                        classes_synthesized[target_position]
                    }
                    None => u32::MAX,
                };
            }
            next_synthesized[position] = if let Some(&class) = intern.get(&signature) {
                class
            } else {
                let class = intern.len() as u32;
                intern.insert(signature.clone(), class);
                class
            };
        }
        let next_full = full_states
            .par_iter()
            .enumerate()
            .map_init(
                || vec![0u32; 1 + relevant_bytes.len()],
                |signature, (position, &state)| {
                    signature[0] = full_labels[position];
                    for (byte_index, &byte) in relevant_bytes.iter().enumerate() {
                        signature[byte_index + 1] = match full.step(state, byte) {
                            Some(target) => {
                                let target_position = full_position[target as usize];
                                if target_position == usize::MAX {
                                    return None;
                                }
                                classes_full[target_position]
                            }
                            None => u32::MAX,
                        };
                    }
                    intern.get(signature).copied()
                },
            )
            .collect::<Option<Vec<_>>>();
        let Some(next_full) = next_full else {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] deterministic_certification_rejected stage=refinement current_depth={} requested_depth={} full_states={} synthesized_states={} relevant_bytes={} synthesized_classes={} elapsed_ms={:.3}",
                    current_depth + 1,
                    depth,
                    full_states.len(),
                    synthesized_states.len(),
                    relevant_bytes.len(),
                    intern.len(),
                    started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
            return None;
        };
        classes_synthesized = next_synthesized;
        classes_full = next_full;
    }

    let class_count = classes_synthesized
        .iter()
        .chain(&classes_full)
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut synthesized_for_class = vec![u32::MAX; class_count];
    for (position, &state) in synthesized_states.iter().enumerate() {
        synthesized_for_class[classes_synthesized[position] as usize] = state;
    }
    let mut mapping = Vec::with_capacity(full_states.len());
    for &class in &classes_full {
        let synthesized_state = synthesized_for_class[class as usize];
        if synthesized_state == u32::MAX {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] deterministic_certification_rejected stage=final_class full_states={} synthesized_states={} depth={} relevant_bytes={} elapsed_ms={:.3}",
                    full_states.len(),
                    synthesized_states.len(),
                    depth,
                    relevant_bytes.len(),
                    started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
            return None;
        }
        mapping.push(synthesized_state);
    }
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] deterministic_certification_accepted full_states={} synthesized_states={} depth={} relevant_bytes={} elapsed_ms={:.3}",
            full_states.len(),
            synthesized_states.len(),
            depth,
            relevant_bytes.len(),
            started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some(mapping)
}

fn certify_deterministic_dispatch_state_map(
    full: &Tokenizer,
    synthesized: &Tokenizer,
    vocab: &Vocab,
    active_terminals: Option<&[bool]>,
) -> Option<CertifiedFullToSynthesizedStateMap> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let max_token_len = vocab.max_token_byte_len();
    let vocabulary_bytes = vocab.relevant_bytes();
    let raw_relevant_byte_count = vocabulary_bytes.len();
    let byte_quotient_started_at = profile.then(std::time::Instant::now);
    let relevant_bytes =
        exact_combined_byte_representatives(full, synthesized, &vocabulary_bytes);
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] deterministic_certification_byte_quotient raw_bytes={} representative_bytes={} elapsed_ms={:.3}",
            raw_relevant_byte_count,
            relevant_bytes.len(),
            byte_quotient_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }

    if !full.has_epsilon_transitions() && !synthesized.has_epsilon_transitions() {
        let full_states = (0..full.num_states()).collect::<Vec<_>>();
        let synthesized_states = (0..synthesized.num_states()).collect::<Vec<_>>();
        let full_to_synthesized = map_deterministic_component_states(
            full,
            synthesized,
            &full_states,
            &synthesized_states,
            max_token_len,
            &relevant_bytes,
            active_terminals,
        )?;
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] deterministic_certification_shape shape=whole_dfa full_states={} synthesized_states={} depth={} relevant_bytes={}",
                full.num_states(),
                synthesized.num_states(),
                max_token_len,
                relevant_bytes.len(),
            );
        }
        return Some(CertifiedFullToSynthesizedStateMap {
            full_to_synthesized,
        });
    }

    let full_roots = match full.deterministic_dispatch_roots() {
        Some(roots) => roots,
        None => {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] deterministic_dispatch_certification_unavailable side=full epsilon={} initial_branches={}",
                    full.has_epsilon_transitions(),
                    full.initial_epsilon_branch_count(),
                );
            }
            return None;
        }
    };
    let synthesized_roots = match synthesized.deterministic_dispatch_roots() {
        Some(roots) => roots,
        None => {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] deterministic_dispatch_certification_unavailable side=synthesized epsilon={} initial_branches={}",
                    synthesized.has_epsilon_transitions(),
                    synthesized.initial_epsilon_branch_count(),
                );
            }
            return None;
        }
    };
    if full_roots.len() != synthesized_roots.len() {
        return None;
    }

    let mut full_claimed = vec![false; full.num_states() as usize];
    let mut synthesized_claimed = vec![false; synthesized.num_states() as usize];
    full_claimed[full.start_state() as usize] = true;
    synthesized_claimed[synthesized.start_state() as usize] = true;
    let mut full_to_synthesized = vec![u32::MAX; full.num_states() as usize];
    full_to_synthesized[full.start_state() as usize] = synthesized.start_state();

    for (&full_root, &synthesized_root) in full_roots.iter().zip(synthesized_roots) {
        let full_states = collect_component_states(full, full_root, &mut full_claimed)?;
        let synthesized_states =
            collect_component_states(synthesized, synthesized_root, &mut synthesized_claimed)?;
        let component_map = map_deterministic_component_states(
            full,
            synthesized,
            &full_states,
            &synthesized_states,
            max_token_len,
            &relevant_bytes,
            active_terminals,
        )?;
        for (&full_state, &synthesized_state) in full_states.iter().zip(&component_map) {
            full_to_synthesized[full_state as usize] = synthesized_state;
        }
    }

    let unmapped_full = full_to_synthesized
        .iter()
        .filter(|&&state| state == u32::MAX)
        .count();
    if unmapped_full != 0 {
        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][tokenizer] deterministic_dispatch_certification_fallback unmapped_full={} full_states={} synthesized_states={}",
                unmapped_full,
                full.num_states(),
                synthesized.num_states(),
            );
        }
        return None;
    }

    Some(CertifiedFullToSynthesizedStateMap {
        full_to_synthesized,
    })
}

fn minimum_consumed_bytes(expr: &Expr) -> usize {
    match expr {
        Expr::U8Seq(bytes) => bytes.len(),
        Expr::U8Class(_) => 1,
        Expr::Dfa(dfa) => dfa.min_match_byte_len().unwrap_or(0),
        Expr::Intersect { expr, intersect } => {
            minimum_consumed_bytes(expr).max(minimum_consumed_bytes(intersect))
        }
        Expr::Seq(parts) => parts
            .iter()
            .fold(0usize, |total, part| total.saturating_add(minimum_consumed_bytes(part))),
        Expr::Choice(options) => options
            .iter()
            .map(minimum_consumed_bytes)
            .min()
            .unwrap_or(0),
        Expr::Exclude { expr, .. } => minimum_consumed_bytes(expr),
        Expr::Repeat { expr, min, .. } => minimum_consumed_bytes(expr).saturating_mul(*min),
        Expr::Shared(expr) => minimum_consumed_bytes(expr),
        Expr::Epsilon => 0,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ReducibleRepeatStats {
    largest_max: usize,
    max_product: u128,
    count: usize,
}

const STENCIL_TOKEN_NEIGHBORHOODS: usize = 1;

impl ReducibleRepeatStats {
    fn include(&mut self, max: usize) {
        self.largest_max = self.largest_max.max(max);
        self.max_product = if self.count == 0 {
            max as u128
        } else {
            self.max_product.saturating_mul(max as u128)
        };
        self.count += 1;
    }

    fn merge(&mut self, other: Self) {
        self.largest_max = self.largest_max.max(other.largest_max);
        if other.count != 0 {
            self.max_product = if self.count == 0 {
                other.max_product
            } else {
                self.max_product.saturating_mul(other.max_product)
            };
            self.count += other.count;
        }
    }

    fn is_pathological_candidate(self) -> bool {
        self.largest_max >= 256 || self.max_product > 4_096
    }
}

fn reducible_repeat_analysis_inner(
    expr: &Expr,
    max_token_len: usize,
    shared_cache: &mut FxHashMap<usize, (usize, ReducibleRepeatStats)>,
) -> (usize, ReducibleRepeatStats) {
    match expr {
        Expr::U8Seq(bytes) => (bytes.len(), ReducibleRepeatStats::default()),
        Expr::U8Class(_) => (1, ReducibleRepeatStats::default()),
        Expr::Dfa(dfa) => (
            dfa.min_match_byte_len().unwrap_or(0),
            ReducibleRepeatStats::default(),
        ),
        Expr::Epsilon => (0, ReducibleRepeatStats::default()),
        Expr::Shared(inner) => {
            let key = Arc::as_ptr(inner) as usize;
            if let Some(&cached) = shared_cache.get(&key) {
                return cached;
            }
            let result = reducible_repeat_analysis_inner(inner, max_token_len, shared_cache);
            shared_cache.insert(key, result);
            result
        }
        Expr::Intersect { expr, intersect } => {
            let (left_min, mut left_stats) =
                reducible_repeat_analysis_inner(expr, max_token_len, shared_cache);
            let (right_min, right_stats) =
                reducible_repeat_analysis_inner(intersect, max_token_len, shared_cache);
            left_stats.merge(right_stats);
            (left_min.max(right_min), left_stats)
        }
        Expr::Exclude { expr, exclude } => {
            let (base_min, mut base_stats) =
                reducible_repeat_analysis_inner(expr, max_token_len, shared_cache);
            let (_, excluded_stats) =
                reducible_repeat_analysis_inner(exclude, max_token_len, shared_cache);
            base_stats.merge(excluded_stats);
            (base_min, base_stats)
        }
        Expr::Seq(parts) => {
            let mut minimum = 0usize;
            let mut stats = ReducibleRepeatStats::default();
            for part in parts {
                let (part_min, part_stats) =
                    reducible_repeat_analysis_inner(part, max_token_len, shared_cache);
                minimum = minimum.saturating_add(part_min);
                stats.merge(part_stats);
            }
            (minimum, stats)
        }
        Expr::Choice(parts) => {
            let mut minimum = None;
            let mut stats = ReducibleRepeatStats::default();
            for part in parts {
                let (part_min, part_stats) =
                    reducible_repeat_analysis_inner(part, max_token_len, shared_cache);
                minimum = Some(minimum.map_or(part_min, |current: usize| current.min(part_min)));
                stats.merge(part_stats);
            }
            (minimum.unwrap_or(0), stats)
        }
        Expr::Repeat { expr, min, max } => {
            let (child_min, mut stats) =
                reducible_repeat_analysis_inner(expr, max_token_len, shared_cache);
            if let Some(max) = *max
                && child_min != 0
            {
                let crossed_per_token = max_token_len.div_ceil(child_min).saturating_add(1);
                let stencil_max = min
                    .saturating_add(
                        crossed_per_token.saturating_mul(STENCIL_TOKEN_NEIGHBORHOODS),
                    )
                    .saturating_add(1);
                if max > stencil_max {
                    stats.include(max);
                }
            }
            (child_min.saturating_mul(*min), stats)
        }
    }
}

fn reducible_repeat_analysis(
    expr: &Expr,
    max_token_len: usize,
) -> (usize, ReducibleRepeatStats) {
    reducible_repeat_analysis_inner(expr, max_token_len, &mut FxHashMap::default())
}

fn reducible_repeat_stats(expr: &Expr, max_token_len: usize) -> ReducibleRepeatStats {
    reducible_repeat_analysis(expr, max_token_len).1
}

const MIN_VOCAB_ONLY_PROBE_FULL_ESTIMATE: u128 = 16_000_000;
const MIN_VOCAB_ONLY_PROBE_SAVING: u128 = 8_000_000;

fn vocabulary_only_probe_is_worthwhile(expr: &Expr, full_estimate: u128) -> bool {
    if full_estimate < MIN_VOCAB_ONLY_PROBE_FULL_ESTIMATE {
        return false;
    }

    // A zero-byte conservative horizon is an optimistic lower bound on the
    // representative expression attainable from any real vocabulary. Only pay
    // for exact vocabulary analysis when even that optimistic candidate removes
    // a large absolute amount of work and at least half of the estimated state
    // product. Compute that lower bound without cloning or rewriting the
    // expression; only inspect the more expensive structural compile plan after
    // the arithmetic gate passes. The exact planner and certifier remain
    // authoritative.
    let optimistic = optimistic_synthesis_estimate(expr);
    if !optimistic.changed {
        return false;
    }
    if full_estimate.saturating_sub(optimistic.synthesized_volume)
        < MIN_VOCAB_ONLY_PROBE_SAVING
        || optimistic.synthesized_volume.saturating_mul(2) > full_estimate
    {
        return false;
    }
    structural_pair_component_count(expr, expr).is_some_and(|count| count >= 2)
}

/// Cheap prefilter for the production synthesis lane.
///
/// The vocabulary's maximum token length is already cached, so use the exact
/// conservative repeat analysis here instead of a token-length-independent
/// approximation. This prevents grammars with large but fully token-observable
/// repeats from entering expensive synthesis only to reject every candidate.
pub(crate) struct BoundedTerminalCandidateScanner {
    analysis: BoundedTerminalAnalysisCache,
}

impl BoundedTerminalCandidateScanner {
    pub(crate) fn new(max_token_len: usize) -> Self {
        Self {
            analysis: BoundedTerminalAnalysisCache::new(max_token_len),
        }
    }

    pub(crate) fn is_candidate(&mut self, expression: &Expr) -> bool {
        if self.analysis.is_pathological_candidate(expression) {
            return true;
        }
        let full_estimate = estimated_expression_state_volume(expression);
        vocabulary_only_probe_is_worthwhile(expression, full_estimate)
    }
}

pub(crate) struct BoundedTerminalAnalysisCache {
    max_token_len: usize,
    shared_cache: FxHashMap<usize, (usize, ReducibleRepeatStats)>,
}

impl BoundedTerminalAnalysisCache {
    pub(crate) fn new(max_token_len: usize) -> Self {
        Self {
            max_token_len,
            shared_cache: FxHashMap::default(),
        }
    }

    fn stats(&mut self, expr: &Expr) -> ReducibleRepeatStats {
        reducible_repeat_analysis_inner(expr, self.max_token_len, &mut self.shared_cache).1
    }

    pub(crate) fn is_pathological_candidate(&mut self, expr: &Expr) -> bool {
        self.stats(expr).is_pathological_candidate()
    }

    pub(crate) fn has_reducible_repeat(&mut self, expr: &Expr) -> bool {
        self.stats(expr).count != 0
    }
}

pub(crate) fn debug_reducible_repeat_stats(
    expr: &Expr,
    max_token_len: usize,
) -> (usize, u128, usize, bool) {
    let stats = reducible_repeat_stats(expr, max_token_len);
    (
        stats.largest_max,
        stats.max_product,
        stats.count,
        stats.is_pathological_candidate(),
    )
}

pub(crate) fn is_pathological_bounded_terminal_candidate(
    expr: &Expr,
    max_token_len: usize,
) -> bool {
    reducible_repeat_stats(expr, max_token_len).is_pathological_candidate()
}

pub(crate) fn has_reducible_bounded_repeat(expr: &Expr, max_token_len: usize) -> bool {
    reducible_repeat_stats(expr, max_token_len).count != 0
}

fn synthesize_expression(expr: &Expr, max_token_len: usize) -> (Expr, bool) {
    match expr {
        Expr::Repeat { expr, min, max } => {
            let (child, child_changed) = synthesize_expression(expr, max_token_len);
            let child_min = minimum_consumed_bytes(&child);
            let mut changed = child_changed;
            let synthesized_max = max.map(|max| {
                if child_min == 0 {
                    return max;
                }
                // A token can enter part-way through one repetition, cross at
                // most ceil(K / child_min) complete repetition boundaries, and
                // leave part-way through another. Keep token-width
                // neighbourhoods around both boundaries plus one full
                // token-width synchronized interior. The generic certifier
                // remains authoritative.
                let crossed_per_token = max_token_len.div_ceil(child_min).saturating_add(1);
                let stencil_max = min
                    .saturating_add(
                        crossed_per_token.saturating_mul(STENCIL_TOKEN_NEIGHBORHOODS),
                    )
                    .saturating_add(1);
                if max > stencil_max
                    && estimated_repeat_reduction_is_material(
                        expr,
                        &child,
                        max,
                        stencil_max,
                    )
                {
                    changed = true;
                    stencil_max
                } else {
                    max
                }
            });
            (
                Expr::Repeat {
                    expr: Box::new(child),
                    min: *min,
                    max: synthesized_max,
                },
                changed,
            )
        }
        Expr::Intersect { expr, intersect } => {
            let (expr, left_changed) = synthesize_expression(expr, max_token_len);
            let (intersect, right_changed) = synthesize_expression(intersect, max_token_len);
            (
                Expr::Intersect {
                    expr: Box::new(expr),
                    intersect: Box::new(intersect),
                },
                left_changed || right_changed,
            )
        }
        Expr::Seq(parts) => {
            let mut changed = false;
            let parts = parts
                .iter()
                .map(|part| {
                    let (part, part_changed) = synthesize_expression(part, max_token_len);
                    changed |= part_changed;
                    part
                })
                .collect();
            (Expr::Seq(parts), changed)
        }
        Expr::Choice(options) => {
            let mut changed = false;
            let options = options
                .iter()
                .map(|option| {
                    let (option, option_changed) = synthesize_expression(option, max_token_len);
                    changed |= option_changed;
                    option
                })
                .collect();
            (Expr::Choice(options), changed)
        }
        Expr::Exclude { expr, exclude } => {
            let (expr, left_changed) = synthesize_expression(expr, max_token_len);
            let (exclude, right_changed) = synthesize_expression(exclude, max_token_len);
            (
                Expr::Exclude {
                    expr: Box::new(expr),
                    exclude: Box::new(exclude),
                },
                left_changed || right_changed,
            )
        }
        Expr::Shared(inner) => {
            let (inner, changed) = synthesize_expression(inner, max_token_len);
            (Expr::Shared(std::sync::Arc::new(inner)), changed)
        }
        leaf => (leaf.clone(), false),
    }
}

/// Vocabulary-relative counterpart of `synthesize_expression`.
///
/// For each bounded repeat, use the exact maximum number of repeat boundaries
/// observable within a suffix of one actual vocabulary token. The traditional
/// byte-length/minimum-width estimate remains the fail-closed fallback when the
/// translation automaton exceeds its proof budget.
fn synthesize_expression_for_vocab(
    expr: &Expr,
    max_token_len: usize,
    vocab: &Vocab,
    horizons: &VocabularyRepeatHorizonCache,
) -> (Expr, bool, bool) {
    const MAX_VOCAB_HORIZON_BODY_ESTIMATE: u128 = 4_096;
    const MIN_HORIZON_IMPROVEMENT_FACTOR: usize = 4;

    match expr {
        Expr::Repeat { expr: body, min, max } => {
            let (child, child_changed, child_used_vocab) =
                synthesize_expression_for_vocab(body, max_token_len, vocab, horizons);
            let child_min = minimum_consumed_bytes(&child);
            let mut changed = child_changed;
            let mut used_vocab = child_used_vocab;
            let synthesized_max = max.map(|max| {
                if child_min == 0 {
                    return max;
                }
                let fallback = max_token_len.div_ceil(child_min).saturating_add(1);
                let conservative_stencil_max = min
                    .saturating_add(
                        fallback.saturating_mul(STENCIL_TOKEN_NEIGHBORHOODS),
                    )
                    .saturating_add(1);
                let conservative_reduction = max > conservative_stencil_max
                    && estimated_repeat_reduction_is_material(
                        body,
                        &child,
                        max,
                        conservative_stencil_max,
                    );
                let mut selected_max = if conservative_reduction {
                    changed = true;
                    conservative_stencil_max
                } else {
                    max
                };

                // The finite-vocabulary proof is useful only when it removes a
                // genuinely large amount of conservatism. Running it for a
                // one-layer improvement creates brittle planning cliffs, while
                // compiling a large repeat-body DFA can cost more than the
                // tokenizer optimization can save. The ordinary byte-length
                // proof remains the fail-closed baseline in both cases.
                // Even a zero-boundary vocabulary horizon yields a stencil of
                // `min + 1`. Do not compile the repeat-body DFA or scan the
                // vocabulary when that best possible result cannot shorten the
                // repeat. This is the common case for small bounded helpers
                // nested inside otherwise large terminal expressions.
                let best_possible_vocab_stencil = min.saturating_add(1);
                if max > best_possible_vocab_stencil
                    && selected_max > best_possible_vocab_stencil
                    && estimated_expression_state_volume(body)
                        <= MAX_VOCAB_HORIZON_BODY_ESTIMATE
                    && let Some(vocab_horizon) = horizons.horizon_for_expr(body, vocab)
                    && vocab_horizon
                        .saturating_mul(MIN_HORIZON_IMPROVEMENT_FACTOR)
                        <= fallback
                {
                    let vocab_stencil_max = min
                        .saturating_add(
                            vocab_horizon.saturating_mul(STENCIL_TOKEN_NEIGHBORHOODS),
                        )
                        .saturating_add(1);
                    if vocab_stencil_max < selected_max
                        && max > vocab_stencil_max
                        && estimated_repeat_reduction_is_material(
                            body,
                            &child,
                            max,
                            vocab_stencil_max,
                        )
                    {
                        selected_max = vocab_stencil_max;
                        changed = true;
                        used_vocab = true;
                    }
                }
                selected_max
            });
            (
                Expr::Repeat {
                    expr: Box::new(child),
                    min: *min,
                    max: synthesized_max,
                },
                changed,
                used_vocab,
            )
        }
        Expr::Intersect { expr, intersect } => {
            let (expr, left_changed, left_used_vocab) =
                synthesize_expression_for_vocab(expr, max_token_len, vocab, horizons);
            let (intersect, right_changed, right_used_vocab) =
                synthesize_expression_for_vocab(intersect, max_token_len, vocab, horizons);
            (
                Expr::Intersect {
                    expr: Box::new(expr),
                    intersect: Box::new(intersect),
                },
                left_changed || right_changed,
                left_used_vocab || right_used_vocab,
            )
        }
        Expr::Seq(parts) => {
            let mut changed = false;
            let mut used_vocab = false;
            let parts = parts
                .iter()
                .map(|part| {
                    let (part, part_changed, part_used_vocab) =
                        synthesize_expression_for_vocab(part, max_token_len, vocab, horizons);
                    changed |= part_changed;
                    used_vocab |= part_used_vocab;
                    part
                })
                .collect();
            (Expr::Seq(parts), changed, used_vocab)
        }
        Expr::Choice(options) => {
            let mut changed = false;
            let mut used_vocab = false;
            let options = options
                .iter()
                .map(|option| {
                    let (option, option_changed, option_used_vocab) =
                        synthesize_expression_for_vocab(option, max_token_len, vocab, horizons);
                    changed |= option_changed;
                    used_vocab |= option_used_vocab;
                    option
                })
                .collect();
            (Expr::Choice(options), changed, used_vocab)
        }
        Expr::Exclude { expr, exclude } => {
            let (expr, left_changed, left_used_vocab) =
                synthesize_expression_for_vocab(expr, max_token_len, vocab, horizons);
            let (exclude, right_changed, right_used_vocab) =
                synthesize_expression_for_vocab(exclude, max_token_len, vocab, horizons);
            (
                Expr::Exclude {
                    expr: Box::new(expr),
                    exclude: Box::new(exclude),
                },
                left_changed || right_changed,
                left_used_vocab || right_used_vocab,
            )
        }
        Expr::Shared(inner) => {
            let (inner, changed, used_vocab) =
                synthesize_expression_for_vocab(inner, max_token_len, vocab, horizons);
            (
                Expr::Shared(std::sync::Arc::new(inner)),
                changed,
                used_vocab,
            )
        }
        leaf => (leaf.clone(), false, false),
    }
}

fn estimated_expression_state_volume_inner(
    expr: &Expr,
    shared_cache: &mut FxHashMap<usize, u128>,
) -> u128 {
    match expr {
        Expr::U8Seq(bytes) => bytes.len().saturating_add(1) as u128,
        Expr::U8Class(_) => 2,
        Expr::Dfa(dfa) => dfa.num_states().max(1) as u128,
        Expr::Epsilon => 1,
        Expr::Shared(inner) => {
            let key = Arc::as_ptr(inner) as usize;
            if let Some(&cached) = shared_cache.get(&key) {
                return cached;
            }
            let estimate = estimated_expression_state_volume_inner(inner, shared_cache);
            shared_cache.insert(key, estimate);
            estimate
        }
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().fold(1u128, |total, part| {
            total.saturating_add(estimated_expression_state_volume_inner(part, shared_cache))
        }),
        Expr::Repeat { expr, max, .. } => {
            let body = estimated_expression_state_volume_inner(expr, shared_cache);
            let copies = max.map_or(2u128, |max| max.saturating_add(1) as u128);
            1u128.saturating_add(body.saturating_mul(copies))
        }
        Expr::Intersect { expr, intersect } => {
            let left = estimated_expression_state_volume_inner(expr, shared_cache);
            let right = estimated_expression_state_volume_inner(intersect, shared_cache);
            left.saturating_mul(right)
        }
        Expr::Exclude { expr, exclude } => {
            let base = estimated_expression_state_volume_inner(expr, shared_cache);
            let excluded = estimated_expression_state_volume_inner(exclude, shared_cache);
            base.saturating_mul(excluded)
        }
    }
}

fn estimated_expression_state_volume(expr: &Expr) -> u128 {
    estimated_expression_state_volume_inner(expr, &mut FxHashMap::default())
}

#[derive(Clone, Copy)]
struct OptimisticSynthesisEstimate {
    minimum_bytes: usize,
    full_volume: u128,
    synthesized_volume: u128,
    changed: bool,
}

fn optimistic_synthesis_estimate_inner(
    expr: &Expr,
    shared_cache: &mut FxHashMap<usize, OptimisticSynthesisEstimate>,
) -> OptimisticSynthesisEstimate {
    match expr {
        Expr::U8Seq(bytes) => {
            let volume = bytes.len().saturating_add(1) as u128;
            OptimisticSynthesisEstimate {
                minimum_bytes: bytes.len(),
                full_volume: volume,
                synthesized_volume: volume,
                changed: false,
            }
        }
        Expr::U8Class(_) => OptimisticSynthesisEstimate {
            minimum_bytes: 1,
            full_volume: 2,
            synthesized_volume: 2,
            changed: false,
        },
        Expr::Dfa(dfa) => {
            let volume = dfa.num_states().max(1) as u128;
            OptimisticSynthesisEstimate {
                minimum_bytes: dfa.min_match_byte_len().unwrap_or(0),
                full_volume: volume,
                synthesized_volume: volume,
                changed: false,
            }
        }
        Expr::Epsilon => OptimisticSynthesisEstimate {
            minimum_bytes: 0,
            full_volume: 1,
            synthesized_volume: 1,
            changed: false,
        },
        Expr::Shared(inner) => {
            let key = Arc::as_ptr(inner) as usize;
            if let Some(&cached) = shared_cache.get(&key) {
                return cached;
            }
            let estimate = optimistic_synthesis_estimate_inner(inner, shared_cache);
            shared_cache.insert(key, estimate);
            estimate
        }
        Expr::Seq(parts) => {
            let mut estimate = OptimisticSynthesisEstimate {
                minimum_bytes: 0,
                full_volume: 1,
                synthesized_volume: 1,
                changed: false,
            };
            for part in parts {
                let part = optimistic_synthesis_estimate_inner(part, shared_cache);
                estimate.minimum_bytes = estimate
                    .minimum_bytes
                    .saturating_add(part.minimum_bytes);
                estimate.full_volume = estimate.full_volume.saturating_add(part.full_volume);
                estimate.synthesized_volume = estimate
                    .synthesized_volume
                    .saturating_add(part.synthesized_volume);
                estimate.changed |= part.changed;
            }
            estimate
        }
        Expr::Choice(parts) => {
            let mut estimate = OptimisticSynthesisEstimate {
                minimum_bytes: usize::MAX,
                full_volume: 1,
                synthesized_volume: 1,
                changed: false,
            };
            for part in parts {
                let part = optimistic_synthesis_estimate_inner(part, shared_cache);
                estimate.minimum_bytes = estimate.minimum_bytes.min(part.minimum_bytes);
                estimate.full_volume = estimate.full_volume.saturating_add(part.full_volume);
                estimate.synthesized_volume = estimate
                    .synthesized_volume
                    .saturating_add(part.synthesized_volume);
                estimate.changed |= part.changed;
            }
            if parts.is_empty() {
                estimate.minimum_bytes = 0;
            }
            estimate
        }
        Expr::Intersect { expr, intersect } => {
            let left = optimistic_synthesis_estimate_inner(expr, shared_cache);
            let right = optimistic_synthesis_estimate_inner(intersect, shared_cache);
            OptimisticSynthesisEstimate {
                minimum_bytes: left.minimum_bytes.max(right.minimum_bytes),
                full_volume: left.full_volume.saturating_mul(right.full_volume),
                synthesized_volume: left
                    .synthesized_volume
                    .saturating_mul(right.synthesized_volume),
                changed: left.changed || right.changed,
            }
        }
        Expr::Exclude { expr, exclude } => {
            let base = optimistic_synthesis_estimate_inner(expr, shared_cache);
            let excluded = optimistic_synthesis_estimate_inner(exclude, shared_cache);
            OptimisticSynthesisEstimate {
                minimum_bytes: base.minimum_bytes,
                full_volume: base.full_volume.saturating_mul(excluded.full_volume),
                synthesized_volume: base
                    .synthesized_volume
                    .saturating_mul(excluded.synthesized_volume),
                changed: base.changed || excluded.changed,
            }
        }
        Expr::Repeat { expr, min, max } => {
            let child = optimistic_synthesis_estimate_inner(expr, shared_cache);
            let full_max = max.unwrap_or(1);
            let mut synthesized_max = full_max;
            let mut changed = child.changed;
            if max.is_some() && child.minimum_bytes != 0 {
                let stencil_max = min
                    .saturating_add(STENCIL_TOKEN_NEIGHBORHOODS)
                    .saturating_add(1);
                let full_repeat = child
                    .full_volume
                    .saturating_mul(full_max.saturating_add(1) as u128);
                let synthesized_repeat = child
                    .synthesized_volume
                    .saturating_mul(stencil_max.saturating_add(1) as u128);
                if full_max > stencil_max
                    && full_repeat.saturating_sub(synthesized_repeat) >= 64
                    && synthesized_repeat.saturating_mul(4)
                        <= full_repeat.saturating_mul(3)
                {
                    synthesized_max = stencil_max;
                    changed = true;
                }
            }
            OptimisticSynthesisEstimate {
                minimum_bytes: child.minimum_bytes.saturating_mul(*min),
                full_volume: 1u128.saturating_add(
                    child
                        .full_volume
                        .saturating_mul(full_max.saturating_add(1) as u128),
                ),
                synthesized_volume: 1u128.saturating_add(
                    child
                        .synthesized_volume
                        .saturating_mul(synthesized_max.saturating_add(1) as u128),
                ),
                changed,
            }
        }
    }
}

fn optimistic_synthesis_estimate(expr: &Expr) -> OptimisticSynthesisEstimate {
    optimistic_synthesis_estimate_inner(expr, &mut FxHashMap::default())
}

pub(crate) fn estimated_synthesis_state_volume(expr: &Expr) -> u128 {
    estimated_expression_state_volume(expr)
}

fn estimated_repeat_reduction_is_material(
    full_body: &Expr,
    synthesized_body: &Expr,
    full_max: usize,
    synthesized_max: usize,
) -> bool {
    const MIN_LOCAL_STATE_SAVING: u128 = 64;
    const MAX_SYNTHESIZED_RATIO_NUMERATOR: u128 = 3;
    const MAX_SYNTHESIZED_RATIO_DENOMINATOR: u128 = 4;

    let full = estimated_expression_state_volume(full_body)
        .saturating_mul(full_max.saturating_add(1) as u128);
    let synthesized = estimated_expression_state_volume(synthesized_body)
        .saturating_mul(synthesized_max.saturating_add(1) as u128);
    full.saturating_sub(synthesized) >= MIN_LOCAL_STATE_SAVING
        && synthesized.saturating_mul(MAX_SYNTHESIZED_RATIO_DENOMINATOR)
            <= full.saturating_mul(MAX_SYNTHESIZED_RATIO_NUMERATOR)
}

fn estimated_synthesis_reduction_is_profitable(full: &Expr, synthesized: &Expr) -> bool {
    const MIN_ESTIMATED_FULL_STATES: u128 = 8_192;
    const MIN_ESTIMATED_STATE_SAVING: u128 = 4_096;
    const MAX_SYNTHESIZED_RATIO_NUMERATOR: u128 = 3;
    const MAX_SYNTHESIZED_RATIO_DENOMINATOR: u128 = 4;

    let full_estimate = estimated_expression_state_volume(full);
    let synthesized_estimate = estimated_expression_state_volume(synthesized);
    full_estimate >= MIN_ESTIMATED_FULL_STATES
        && full_estimate.saturating_sub(synthesized_estimate) >= MIN_ESTIMATED_STATE_SAVING
        && synthesized_estimate.saturating_mul(MAX_SYNTHESIZED_RATIO_DENOMINATOR)
            <= full_estimate.saturating_mul(MAX_SYNTHESIZED_RATIO_NUMERATOR)
}

/// Build a finite-token-horizon stencil without applying the global
/// pathological-candidate threshold. This is used for partition-local
/// analysis lexers: a terminal whose exact bound is observable by the longest
/// token in the full vocabulary can still be safely shortened for a vocabulary
/// partition whose tokens are all substantially shorter.
pub(crate) fn synthesize_terminal_expressions_for_horizon(
    expressions: &[Expr],
    max_token_len: usize,
) -> SynthesizedTerminalExpressions {
    let mut changed_terminals = Vec::new();
    let expressions = expressions
        .iter()
        .enumerate()
        .map(|(terminal, expression)| {
            let (synthesized, changed) = synthesize_expression(expression, max_token_len);
            if changed {
                changed_terminals.push(terminal as u32);
            }
            synthesized.optimize()
        })
        .collect();
    SynthesizedTerminalExpressions {
        expressions,
        changed_terminals,
    }
}

pub(crate) fn synthesize_bounded_terminal_expressions(
    expressions: &[Expr],
    vocab: &Vocab,
    horizons: &VocabularyRepeatHorizonCache,
) -> SynthesizedTerminalExpressions {
    let max_token_len = vocab.max_token_byte_len();
    let allow_vocab_only_candidates =
        std::env::var_os("GLRMASK_SYNTHETIC_VOCAB_ONLY_CANDIDATES").is_some();
    let mut analysis = BoundedTerminalAnalysisCache::new(max_token_len);
    let candidates = expressions
        .iter()
        .enumerate()
        .filter_map(|(terminal, expression)| {
            let full_estimate = estimated_expression_state_volume(expression);
            if full_estimate < 8_192 {
                return None;
            }
            let conservative_pathological = analysis.is_pathological_candidate(expression);
            let guarded_large_candidate = !conservative_pathological
                && vocabulary_only_probe_is_worthwhile(expression, full_estimate);
            // Vocabulary-only candidate discovery is materially more expensive
            // than the conservative repeat analysis because it compiles repeat
            // bodies and scans the full vocabulary. Keep that speculative lane
            // out of the production planner until a caller explicitly requests
            // it. Conservative-pathological terminals still use vocabulary
            // horizons to tighten an already justified synthesis attempt.
            if !conservative_pathological
                && !allow_vocab_only_candidates
                && !guarded_large_candidate
            {
                return None;
            }
            // New vocabulary-only reductions currently require an explicit
            // multi-component structural product. Single-component candidates
            // fall back to generic finite-horizon equivalence and have shown
            // large, input-sensitive planning costs. Check the full expression
            // shape before compiling any repeat-body DFA.
            if !conservative_pathological
                && !structural_pair_component_count(expression, expression)
                    .is_some_and(|components| components >= 2)
            {
                return None;
            }
            Some((
                terminal,
                full_estimate,
                conservative_pathological,
                guarded_large_candidate,
            ))
        })
        .collect::<Vec<_>>();

    // Candidate discovery is deliberately scalar and cache-sharing: most
    // grammars have zero or one candidate, and walking every terminal in the
    // Rayon pool made one pathological terminal force unrelated ordinary
    // helpers through vocabulary analysis. Only the small, already-qualified
    // candidate set performs independent exact horizon proofs in parallel.
    let synthesized_candidates = candidates
        .par_iter()
        .map(
            |&(terminal, full_estimate, conservative_pathological, guarded_large_candidate)| {
            let expression = &expressions[terminal];
            let (candidate, changed, used_vocab) =
                synthesize_expression_for_vocab(expression, max_token_len, vocab, horizons);
            let vocabulary_shape_supported = !used_vocab
                || conservative_pathological
                || structural_pair_component_count(expression, &candidate)
                    .is_some_and(|components| components >= 2);
            let profitable = changed
                && (conservative_pathological || used_vocab || guarded_large_candidate)
                && vocabulary_shape_supported
                && estimated_synthesis_reduction_is_profitable(expression, &candidate);
            if std::env::var_os("GLRMASK_PROFILE_SYNTHETIC_PLAN").is_some() {
                eprintln!(
                    "[glrmask/profile][synthetic_candidate] terminal={} changed={} selected={} conservative_pathological={} guarded_large_candidate={} used_vocab={} vocabulary_shape_supported={} full_estimate={} synthesized_estimate={}",
                    terminal,
                    changed,
                    profitable,
                    conservative_pathological,
                    guarded_large_candidate,
                    used_vocab,
                    vocabulary_shape_supported,
                    full_estimate,
                    estimated_expression_state_volume(&candidate),
                );
            }
            if profitable {
                (terminal, candidate, true)
            } else {
                (terminal, expression.clone(), false)
            }
        },
        )
        .collect::<Vec<_>>();

    let mut changed_terminals = Vec::new();
    let mut synthesized_expressions = expressions.to_vec();
    for (terminal, expression, changed) in synthesized_candidates {
        if changed {
            changed_terminals.push(terminal as u32);
            synthesized_expressions[terminal] = expression;
        }
    }
    SynthesizedTerminalExpressions {
        expressions: synthesized_expressions,
        changed_terminals,
    }
}

impl CertifiedFullToSynthesizedStateMap {
    /// Replace the synthesized raw-state domain of a finished parser-DWA id map
    /// with the certified full raw-state domain. Internal TSID numbers remain
    /// unchanged, so parser-DWA and possible-match weights need no rewriting.
    pub(crate) fn lift_internal_tsid_map(
        &self,
        synthesized_state_map: &ManyToOneIdMap,
    ) -> Option<ManyToOneIdMap> {
        let mut full_to_internal = Vec::with_capacity(self.full_to_synthesized.len());
        for &synthesized_state in &self.full_to_synthesized {
            let internal = *synthesized_state_map
                .original_to_internal
                .get(synthesized_state as usize)?;
            if internal == u32::MAX
                || internal as usize >= synthesized_state_map.internal_to_originals.len()
            {
                return None;
            }
            full_to_internal.push(internal);
        }
        Some(ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            full_to_internal,
            synthesized_state_map.num_internal_ids(),
        ))
    }
}

pub(crate) fn certify_vocabulary_exact_state_candidates(
    full: &Tokenizer,
    synthesized: &Tokenizer,
    vocab: &Vocab,
    active_terminals: Option<&[bool]>,
) -> Option<CertifiedVocabularyExactStateCandidates> {
    if full.has_epsilon_transitions() || synthesized.has_epsilon_transitions() {
        return None;
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(std::time::Instant::now);
    let union = Tokenizer::disjoint_union_for_analysis(full, synthesized);
    let active = active_terminals
        .map(<[bool]>::to_vec)
        .unwrap_or_else(|| vec![true; full.num_terminals() as usize]);
    let view = TokenizerView::new_filtered(&union.tokenizer, &active);
    let tokens = vocab
        .entries
        .values()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let disallowed_follows = (0..full.num_terminals() as usize)
        .map(|_| BitSet::new(full.num_terminals() as usize))
        .collect::<Vec<_>>();

    // Put synthesized states first. The exact equivalence engine chooses the
    // first state in each class as its representative, so every class that has
    // a synthesized member is represented in the synthesized coordinate.
    let synthesized_reset = union.right_offset + synthesized.start_state();
    let full_reset = union.left_offset + full.start_state();
    let mut states = Vec::with_capacity(
        synthesized.num_states() as usize + full.num_states() as usize,
    );
    let mut reset_states = Vec::with_capacity(states.capacity());
    for state in 0..synthesized.num_states() {
        states.push((union.right_offset + state) as usize);
        reset_states.push(synthesized_reset as usize);
    }
    let full_position_start = states.len();
    for state in 0..full.num_states() {
        states.push((union.left_offset + state) as usize);
        reset_states.push(full_reset as usize);
    }

    let representatives = find_state_equivalence_classes_with_state_resets(
        &view,
        &tokens,
        &states,
        &disallowed_follows,
        &reset_states,
        true,
    );
    if representatives.len() != states.len() {
        return None;
    }

    let synthesized_start = union.right_offset as usize;
    let synthesized_end = synthesized_start + synthesized.num_states() as usize;
    let mut synthesized_candidates_by_representative =
        FxHashMap::<usize, Vec<u32>>::default();
    for (synthesized_state, &representative) in representatives[..full_position_start]
        .iter()
        .enumerate()
    {
        synthesized_candidates_by_representative
            .entry(representative)
            .or_default()
            .push(synthesized_state as u32);
    }
    let synthesized_candidates_by_representative = synthesized_candidates_by_representative
        .into_iter()
        .map(|(representative, candidates)| {
            (representative, Arc::<[u32]>::from(candidates.into_boxed_slice()))
        })
        .collect::<FxHashMap<_, _>>();

    let mut primary = Vec::with_capacity(full.num_states() as usize);
    let mut candidates_by_full_state = Vec::with_capacity(full.num_states() as usize);
    for &representative in &representatives[full_position_start..] {
        if representative < synthesized_start || representative >= synthesized_end {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] vocabulary_certification_rejected full_states={} synthesized_states={} tokens={} elapsed_ms={:.3}",
                    full.num_states(),
                    synthesized.num_states(),
                    tokens.len(),
                    started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
            return None;
        }
        let candidates = Arc::clone(synthesized_candidates_by_representative.get(&representative)?);
        primary.push(*candidates.first()?);
        candidates_by_full_state.push(candidates);
    }

    if profile {
        let flexible_states = candidates_by_full_state
            .iter()
            .filter(|candidates| candidates.len() > 1)
            .count();
        let max_candidates = candidates_by_full_state
            .iter()
            .map(|candidates| candidates.len())
            .max()
            .unwrap_or(0);
        eprintln!(
            "[glrmask/profile][tokenizer] vocabulary_certification_accepted full_states={} synthesized_states={} tokens={} flexible_states={} max_candidates={} elapsed_ms={:.3}",
            full.num_states(),
            synthesized.num_states(),
            tokens.len(),
            flexible_states,
            max_candidates,
            started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some(CertifiedVocabularyExactStateCandidates {
        primary,
        candidates_by_full_state,
    })
}

pub(crate) fn certify_vocabulary_exact_state_map(
    full: &Tokenizer,
    synthesized: &Tokenizer,
    vocab: &Vocab,
    active_terminals: Option<&[bool]>,
) -> Option<CertifiedFullToSynthesizedStateMap> {
    let certified = certify_vocabulary_exact_state_candidates(
        full,
        synthesized,
        vocab,
        active_terminals,
    )?;
    Some(CertifiedFullToSynthesizedStateMap {
        full_to_synthesized: certified.primary,
    })
}

pub(crate) fn certify_full_to_synthesized_state_map(
    full: &Tokenizer,
    synthesized: &Tokenizer,
    vocab: &Vocab,
    active_terminals: Option<&[bool]>,
) -> Option<CertifiedFullToSynthesizedStateMap> {
    if full.num_terminals() != synthesized.num_terminals() {
        return None;
    }
    if active_terminals.is_some_and(|active| active.len() < full.num_terminals() as usize) {
        return None;
    }

    let full_is_deterministic = !full.has_epsilon_transitions();
    let synthesized_is_deterministic = !synthesized.has_epsilon_transitions();
    if full_is_deterministic && synthesized_is_deterministic {
        let certified =
            certify_vocabulary_exact_state_map(full, synthesized, vocab, active_terminals);
        if certified.is_some()
            && std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        {
            eprintln!(
                "[glrmask/profile][tokenizer] synthetic_certification_path path=vocabulary_exact full_states={} synthesized_states={}",
                full.num_states(),
                synthesized.num_states(),
            );
        }
        return certified;
    }
    if let Some(certified) =
        certify_deterministic_dispatch_state_map(full, synthesized, vocab, active_terminals)
    {
        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][tokenizer] synthetic_certification_path path=deterministic_dispatch full_states={} synthesized_states={}",
                full.num_states(),
                synthesized.num_states(),
            );
        }
        return Some(certified);
    }
    let union = Tokenizer::disjoint_union_for_analysis(full, synthesized);
    let statistic = max_length::compute_statistic(vocab);
    let state_map = max_length::compute_state_map(
        &union.tokenizer,
        &statistic,
        None,
        active_terminals,
        MaxLengthMode::KBoundedByteRestricted,
        None,
        None,
    );

    let synthesized_start = union.right_offset as usize;
    let synthesized_end = synthesized_start + synthesized.num_states() as usize;
    let mut synthesized_for_class = vec![u32::MAX; state_map.internal_to_originals.len()];
    for (class, members) in state_map.internal_to_originals.iter().enumerate() {
        if let Some(&combined_state) = members
            .iter()
            .find(|&&state| (state as usize) >= synthesized_start && (state as usize) < synthesized_end)
        {
            synthesized_for_class[class] = combined_state - union.right_offset;
        }
    }

    let mut full_to_synthesized = Vec::with_capacity(full.num_states() as usize);
    for full_state in 0..full.num_states() {
        let combined_state = union.left_offset + full_state;
        let class = *state_map
            .original_to_internal
            .get(combined_state as usize)?;
        if class == u32::MAX {
            return None;
        }
        let synthesized_state = *synthesized_for_class.get(class as usize)?;
        if synthesized_state == u32::MAX {
            return None;
        }
        full_to_synthesized.push(synthesized_state);
    }

    Some(CertifiedFullToSynthesizedStateMap {
        full_to_synthesized,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::compile::{
        build_regex,
        build_regex_partitioned_with_adaptive,
    };
    use crate::automata::regex::Expr;
    use crate::{Constraint, DynamicConstraint};

    fn repeat_a(max: usize) -> Tokenizer {
        let expressions = vec![Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
            min: 1,
            max: Some(max),
        }];
        build_regex(&expressions).into_tokenizer(
            1,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    #[test]
    fn inactive_dispatch_components_collapse_without_touching_active_components() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(4),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: Some(5),
            },
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
            .into_tokenizer(2, Some(Arc::from(expressions.into_boxed_slice())));
        let components = tokenizer
            .disjoint_dispatch_components()
            .expect("partitioned lexer dispatch components");
        assert_eq!(components.len(), 2);

        let map = inactive_dispatch_component_state_map(&tokenizer, &[true, false])
            .expect("inactive component quotient");
        let active_component = components
            .iter()
            .find(|states| {
                states.iter().any(|&state| {
                    tokenizer.matched_terminals_iter(state).any(|terminal| terminal == 0)
                        || tokenizer
                            .possible_future_terminals_iter(state)
                            .any(|terminal| terminal == 0)
                })
            })
            .expect("active component");
        let inactive_component = components
            .iter()
            .find(|states| !std::ptr::eq(*states, active_component))
            .expect("inactive component");

        let inactive_class = map.original_to_internal[inactive_component[0] as usize];
        assert!(inactive_component
            .iter()
            .all(|&state| map.original_to_internal[state as usize] == inactive_class));
        let mut active_classes = active_component
            .iter()
            .map(|&state| map.original_to_internal[state as usize])
            .collect::<Vec<_>>();
        active_classes.sort_unstable();
        active_classes.dedup();
        assert_eq!(active_classes.len(), active_component.len());
        assert!(!active_classes.contains(&inactive_class));
        let start_class = map.original_to_internal[tokenizer.start_state() as usize];
        assert_ne!(start_class, inactive_class);
        assert!(!active_classes.contains(&start_class));
    }

    #[test]
    fn kbounded_certification_maps_large_repeat_to_small_repeat() {
        let full = repeat_a(64);
        let synthesized = repeat_a(10);
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"aaaa".to_vec()),
            (4, b"x".to_vec()),
        ]);

        let certified = certify_full_to_synthesized_state_map(
            &full,
            &synthesized,
            &vocab,
            Some(&[true]),
        )
        .expect("a ten-byte stencil must represent every four-byte-local residual of a 64-byte repeat");

        assert_eq!(
            certified.full_to_synthesized.len(),
            full.num_states() as usize,
        );
        assert!(
            certified
                .full_to_synthesized
                .iter()
                .all(|&state| state < synthesized.num_states()),
        );
        assert!(
            certified.full_to_synthesized[8..full.num_states() as usize - 5]
                .windows(2)
                .any(|pair| pair[0] == pair[1]),
            "the deep full interior should collapse onto a synthesized residual",
        );
    }

    #[test]
    fn kbounded_certification_rejects_too_small_repeat() {
        let full = repeat_a(64);
        let synthesized = repeat_a(3);
        let vocab = Vocab::new(vec![(0, b"aaaa".to_vec())]);

        assert!(
            certify_full_to_synthesized_state_map(
                &full,
                &synthesized,
                &vocab,
                Some(&[true]),
            )
            .is_none(),
            "a stencil shorter than one vocabulary token cannot represent the deep interior",
        );
    }

    #[test]
    fn certified_map_lifts_a_finished_synthesized_tsid_map_without_relabeling() {
        let full = repeat_a(64);
        let synthesized = repeat_a(10);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aaaa".to_vec())]);
        let certified = certify_full_to_synthesized_state_map(
            &full,
            &synthesized,
            &vocab,
            Some(&[true]),
        )
        .expect("certification");

        let synthesized_state_map = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            (0..synthesized.num_states()).map(|state| state % 3).collect(),
            3,
        );
        let lifted = certified
            .lift_internal_tsid_map(&synthesized_state_map)
            .expect("lifted state map");

        assert_eq!(
            lifted.original_to_internal.len(),
            full.num_states() as usize,
        );
        assert_eq!(lifted.num_internal_ids(), 3);
        for (full_state, &synthesized_state) in
            certified.full_to_synthesized.iter().enumerate()
        {
            assert_eq!(
                lifted.original_to_internal[full_state],
                synthesized_state_map.original_to_internal[synthesized_state as usize],
            );
        }
    }

    #[test]
    fn expression_synthesis_uses_vocab_displacement_and_preserves_small_bounds() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"word ".to_vec())),
                min: 0,
                max: Some(1_000_000),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                min: 0,
                max: Some(8),
            },
        ];
        let vocab = Vocab::new(vec![(0, b"0123456789".to_vec())]);
        let horizons = VocabularyRepeatHorizonCache::new();
        let synthesized =
            synthesize_bounded_terminal_expressions(&expressions, &vocab, &horizons);

        assert_eq!(synthesized.changed_terminals, vec![0]);
        let Expr::Repeat { max, .. } = &synthesized.expressions[0] else {
            panic!("expected repeat");
        };
        // No vocabulary-token suffix can complete `"word "`, so the repeat
        // counter has zero observable displacement. Retain one canonical
        // interior layer; cross-component synchronization is handled by
        // structural tuple materialization.
        assert_eq!(*max, Some(1));
        assert_eq!(synthesized.expressions[1], expressions[1]);
    }

    #[test]
    fn static_synthesized_pipeline_matches_exact_dynamic_runtime_through_full_bound() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaaa".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^a{1,80}$",
            "minLength": 1,
            "maxLength": 80
        }"#;
        let constraint = Constraint::from_json_schema(schema, &vocab).expect("static constraint");
        let dynamic =
            DynamicConstraint::from_json_schema(schema, &vocab).expect("dynamic constraint");
        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();

        assert_eq!(static_state.mask(), dynamic_state.mask());
        static_state.commit_token(0).expect("opening quote");
        dynamic_state.commit_token(0).expect("opening quote");
        for chunk in 0..20 {
            assert_eq!(
                static_state.mask(),
                dynamic_state.mask(),
                "mask mismatch before four-byte chunk {chunk}",
            );
            static_state.commit_token(3).expect("four a bytes");
            dynamic_state.commit_token(3).expect("four a bytes");
        }
        assert_eq!(static_state.mask(), dynamic_state.mask());
        let mask = static_state.mask();
        assert_ne!(mask[0] & (1 << 0), 0, "closing quote must be allowed");
        assert_eq!(mask[0] & (1 << 1), 0, "65th a must be rejected");
        assert_eq!(mask[0] & (1 << 2), 0, "66th a must be rejected");
        assert_eq!(mask[0] & (1 << 3), 0, "68th a must be rejected");
        static_state.commit_token(0).expect("closing quote");
        dynamic_state.commit_token(0).expect("closing quote");
        assert_eq!(static_state.is_finished(), dynamic_state.is_finished());
        assert_eq!(static_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn independently_synthesized_identical_terminals_preserve_different_full_lifetimes() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aaaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let schema = r#"{
            "anyOf": [
                {"type":"string","pattern":"^a{1,80}$","maxLength":80},
                {"type":"string","pattern":"^a{1,160}$","maxLength":160}
            ]
        }"#;
        let constraint = Constraint::from_json_schema(schema, &vocab).expect("static constraint");
        let dynamic =
            DynamicConstraint::from_json_schema(schema, &vocab).expect("dynamic constraint");
        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();

        static_state.commit_token(0).expect("opening quote");
        dynamic_state.commit_token(0).expect("opening quote");
        for chunk in 0..20 {
            assert_eq!(static_state.mask(), dynamic_state.mask(), "chunk {chunk}");
            static_state.commit_token(2).expect("four a bytes");
            dynamic_state.commit_token(2).expect("four a bytes");
        }

        let at_short_limit = static_state.mask();
        assert_eq!(at_short_limit, dynamic_state.mask());
        assert_ne!(at_short_limit[0] & (1 << 0), 0, "short terminal may close");
        assert_ne!(
            at_short_limit[0] & (1 << 2),
            0,
            "long terminal must remain alive after the short terminal expires",
        );

        for chunk in 20..40 {
            assert_eq!(static_state.mask(), dynamic_state.mask(), "chunk {chunk}");
            static_state.commit_token(2).expect("long terminal continuation");
            dynamic_state.commit_token(2).expect("long terminal continuation");
        }
        let at_long_limit = static_state.mask();
        assert_eq!(at_long_limit, dynamic_state.mask());
        assert_ne!(at_long_limit[0] & (1 << 0), 0, "long terminal may close");
        assert_eq!(
            at_long_limit[0] & (1 << 2),
            0,
            "long terminal must expire at its exact full bound",
        );
    }

    #[test]
    #[ignore = "profiling probe for the pathological nested-repeat/max-length product"]
    fn profile_pathological_nested_repeat_max_length() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"aabb".to_vec()),
            (5, b"aaaa".to_vec()),
            (6, b"bbbb".to_vec()),
        ]);
        let schema = r#"{
            "type":"string",
            "pattern":"^(?:a+b+){0,100}a+$",
            "minLength":2,
            "maxLength":500
        }"#;
        std::hint::black_box(
            Constraint::from_json_schema(schema, &vocab).expect("pathological exact constraint"),
        );
    }
}
