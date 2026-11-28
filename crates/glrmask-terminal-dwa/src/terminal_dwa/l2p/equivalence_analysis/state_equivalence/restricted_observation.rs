//! Exact state quotient for L2P's restricted observation model.
//!
//! This pass does not modify the lexer DFA.  It computes a `ManyToOneIdMap`
//! over its original state IDs.  Its observation alphabet consists of the
//! bytes present in the current vocabulary partition, and its terminal labels
//! are restricted to the active L2P terminals (TI representatives when TI is
//! enabled).  Future-terminal labels are read from the original DFA and never
//! recomputed after restricting bytes.


use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use rustc_hash::FxHashMap;
use crate::terminal_dwa::congruence::refine_partition_rows;
use std::time::Instant;

use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::super::compat::TokenizerView;
use super::max_length::active_byte_representatives;

const NO_CANDIDATE: usize = usize::MAX;

pub struct RawRestrictedObservationResult {
    pub state_map: ManyToOneIdMap,
    pub raw_observation_ids: Vec<u32>,
    pub observation_representatives: Vec<u32>,
    pub label_ms: f64,
    pub refine_ms: f64,
    pub certificate_ms: f64,
    pub rounds: usize,
}

struct RawObservationIds {
    ids: Vec<u32>,
    representatives: Vec<u32>,
    active_language: Vec<bool>,
}

/// Intern visible terminal observations already normalized in the shared
/// analysis view. This avoids a second raw-tokenizer scan and active-mask pass.
fn target_label_ids(tokenizer: &TokenizerView) -> Vec<u32> {
    let dfa = tokenizer.dfa();
    let mut ids = vec![0u32; dfa.states.len()];
    let mut order: Vec<usize> = (0..dfa.states.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        dfa.states[left]
            .finalizers
            .cmp(&dfa.states[right].finalizers)
            .then_with(|| {
                dfa.states[left]
                    .possible_future_group_ids
                    .cmp(&dfa.states[right].possible_future_group_ids)
            })
            .then_with(|| left.cmp(&right))
    });

    let mut next_id = 0u32;
    let mut previous: Option<usize> = None;
    for state in order {
        let starts_new = previous.is_none_or(|prev| {
            dfa.states[prev].finalizers != dfa.states[state].finalizers
                || dfa.states[prev].possible_future_group_ids
                    != dfa.states[state].possible_future_group_ids
        });
        if starts_new {
            next_id += 1;
        }
        ids[state] = next_id - 1;
        previous = Some(state);
    }
    ids
}

/// Build the candidate-state partition inherited from an earlier pass.
///
/// Every raw lexer state remains represented.  A prior map contributes one
/// candidate for each of its classes; raw states absent from that map are kept
/// as singleton candidates instead of being silently dropped.
fn candidate_partition(
    num_states: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let mut members = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<usize>::new();
    let mut raw_to_candidate = vec![NO_CANDIDATE; num_states];

    if let Some(map) = initial_state_map {
        for originals in &map.internal_to_originals {
            let mut candidate_members = Vec::with_capacity(originals.len());
            for &raw in originals {
                let raw = raw as usize;
                if raw < num_states && raw_to_candidate[raw] == NO_CANDIDATE {
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
        if raw_to_candidate[raw] != NO_CANDIDATE {
            continue;
        }
        let candidate = members.len();
        raw_to_candidate[raw] = candidate;
        representatives.push(raw);
        members.push(vec![raw as u32]);
    }

    (members, representatives, raw_to_candidate)
}

fn candidate_partition_by_source_observation(
    num_states: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
    target_labels: &[u32],
    tokenizer: &TokenizerView,
) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let dfa = tokenizer.dfa();
    let mut members = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<usize>::new();
    let mut raw_to_candidate = vec![NO_CANDIDATE; num_states];
    let mut candidates_by_key = FxHashMap::<(u32, u32, bool), usize>::default();
    for raw in 0..num_states {
        let inherited = initial_state_map.map(|map| map.original_to_internal[raw]).unwrap_or(raw as u32);
        if inherited == u32::MAX { continue; }
        let has_any_transition = dfa.transitions_for(raw).iter().any(|&target| target != u32::MAX);
        let key = (inherited, target_labels[raw], has_any_transition);
        let candidate = if let Some(&candidate) = candidates_by_key.get(&key) {
            candidate
        } else {
            let candidate = members.len();
            candidates_by_key.insert(key, candidate);
            representatives.push(raw);
            members.push(Vec::new());
            candidate
        };
        raw_to_candidate[raw] = candidate;
        members[candidate].push(raw as u32);
    }
    for raw in 0..num_states {
        if raw_to_candidate[raw] != NO_CANDIDATE { continue; }
        let candidate = members.len();
        raw_to_candidate[raw] = candidate;
        representatives.push(raw);
        members.push(vec![raw as u32]);
    }
    (members, representatives, raw_to_candidate)
}

fn map_from_candidate_classes(
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

/// Compute the coarsest fixed point of the restricted observation recurrence:
///
/// `s ↦ ((b, class(dst(s,b)), F(dst(s,b)), U(dst(s,b))) for b in bytes)`.
///
/// `F` and `U` are the original DFA's finalizer and future-finalizer sets,
/// filtered only by `active_groups`.  They are intentionally not recomputed
/// from the byte-restricted transition relation.
pub fn compute_state_map(
    tokenizer: &TokenizerView,
    relevant_bytes: &[bool; 256],
    initial_state_map: Option<&ManyToOneIdMap>,
    byte_to_class: Option<&[u8; 256]>,
    include_source_observation: bool,
) -> ManyToOneIdMap {
    let dfa = tokenizer.dfa();
    let num_states = dfa.states.len();
    if num_states == 0 {
        return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(Vec::new(), 0);
    }

    let active_bytes = active_byte_representatives(Some(relevant_bytes), byte_to_class);
    let target_labels = target_label_ids(tokenizer);
    let (candidate_members, candidate_representatives, raw_to_candidate) =
        if include_source_observation {
            candidate_partition_by_source_observation(
                num_states,
                initial_state_map,
                &target_labels,
                tokenizer,
            )
        } else {
            candidate_partition(num_states, initial_state_map)
        };
    let num_candidates = candidate_representatives.len();

    // Destination observations are fixed independently of successor classes.
    // Seed them once instead of re-hashing them in every refinement round.
    // A missing edge has code zero; a present edge includes its raw target
    // label even if the inherited candidate also contains other raw labels.
    let mut seeds = FxHashMap::<Vec<u32>, u32>::default();
    let mut signature = Vec::with_capacity(active_bytes.len() + 2);
    let mut initial = Vec::with_capacity(num_candidates);
    for &state in &candidate_representatives {
        signature.clear();
        if include_source_observation {
            signature.push(target_labels[state]);
            signature.push(u32::from(dfa.transitions_for(state).iter().any(|&t| t != u32::MAX)));
        }
        for &byte in &active_bytes {
            let target = dfa.trans(state, byte as usize);
            signature.push(if target == u32::MAX { 0 } else { target_labels[target as usize] + 1 });
        }
        let class = if let Some(&class) = seeds.get(signature.as_slice()) {
            class
        } else {
            let class = seeds.len() as u32;
            seeds.insert(signature.clone(), class);
            class
        };
        initial.push(class);
    }
    drop(seeds);
    let (classes, _, _) = refine_partition_rows(
        initial,
        |candidate| {
            let state = candidate_representatives[candidate];
            let raw_to_candidate = &raw_to_candidate;
            active_bytes.iter().filter_map(move |&byte| {
                let target = dfa.trans(state, byte as usize);
                (target != u32::MAX).then(|| (byte, raw_to_candidate[target as usize] as u32))
            })
        },
        None,
    );
    map_from_candidate_classes(
        &candidate_members, &candidate_representatives, &classes, num_states,
    )
}

fn active_mask_words(active_groups: &[bool]) -> Vec<u64> {
    let mut words = vec![0u64; active_groups.len().div_ceil(64)];
    for (terminal, &active) in active_groups.iter().enumerate() {
        if active {
            words[terminal / 64] |= 1u64 << (terminal % 64);
        }
    }
    words
}

#[inline]
fn mix_observation_hash(mut hash: u64, word: u64) -> u64 {
    hash ^= word.wrapping_add(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^ (hash >> 29)
}

/// Build an exact compact key for a masked finalizer/future observation.
///
/// The first cell is the number of `(word_index, value)` pairs belonging to
/// finalizers; remaining pairs belong to frozen future observations. Zero
/// words are omitted, while their positions are retained, so equality of keys
/// is equality of the original active-terminal bitsets.
fn visible_observation_key(
    tokenizer: &Tokenizer,
    state: u32,
    active_words: &[u64],
    key: &mut Vec<u64>,
) {
    key.clear();
    key.push(0);
    let final_start = key.len();
    for (word_index, (&word, &active)) in tokenizer
        .matched_terminal_bitset(state)
        .words()
        .iter()
        .zip(active_words)
        .enumerate()
    {
        let visible = word & active;
        if visible != 0 {
            key.push(word_index as u64);
            key.push(visible);
        }
    }
    key[0] = ((key.len() - final_start) / 2) as u64;
    for (word_index, (&word, &active)) in tokenizer
        .possible_future_terminals(state)
        .words()
        .iter()
        .zip(active_words)
        .enumerate()
    {
        let visible = word & active;
        if visible != 0 {
            key.push(word_index as u64);
            key.push(visible);
        }
    }
}

#[inline]
fn hash_visible_observation_key(key: &[u64]) -> u64 {
    let mut hash = 0x243f_6a88_85a3_08d3u64;
    for &value in key {
        hash = mix_observation_hash(hash, value);
    }
    hash
}

fn raw_target_label_ids(tokenizer: &Tokenizer, active_groups: &[bool]) -> RawObservationIds {
    let active_words = active_mask_words(active_groups);
    let mut ids = vec![0u32; tokenizer.num_states() as usize];
    let mut buckets = FxHashMap::<u64, Vec<(Vec<u64>, u32)>>::default();
    let mut representatives = Vec::<u32>::new();
    let mut active_language = Vec::with_capacity(tokenizer.num_states() as usize);
    let mut key = Vec::with_capacity(16);
    let mut next_id = 0u32;
    for state in 0..tokenizer.num_states() {
        visible_observation_key(tokenizer, state, &active_words, &mut key);
        active_language.push(key.len() > 1);
        let hash = hash_visible_observation_key(&key);
        let bucket = buckets.entry(hash).or_default();
        if let Some((_, id)) = bucket.iter().find(|(representative, _)| representative == &key) {
            ids[state as usize] = *id;
        } else {
            bucket.push((key.clone(), next_id));
            ids[state as usize] = next_id;
            representatives.push(state);
            next_id += 1;
        }
    }
    if std::env::var_os("GLRMASK_PROFILE_L2P_DETAIL").is_some() {
        let mut final_bits = 0u64;
        let mut future_bits = 0u64;
        let mut nonempty_final_states = 0usize;
        let mut nonempty_future_states = 0usize;
        for state in 0..tokenizer.num_states() {
            let mut state_final_bits = 0u32;
            let mut state_future_bits = 0u32;
            for (&word, &active) in tokenizer
                .matched_terminal_bitset(state)
                .words()
                .iter()
                .zip(&active_words)
            {
                state_final_bits += (word & active).count_ones();
            }
            for (&word, &active) in tokenizer
                .possible_future_terminals(state)
                .words()
                .iter()
                .zip(&active_words)
            {
                state_future_bits += (word & active).count_ones();
            }
            final_bits += state_final_bits as u64;
            future_bits += state_future_bits as u64;
            nonempty_final_states += usize::from(state_final_bits != 0);
            nonempty_future_states += usize::from(state_future_bits != 0);
        }
        eprintln!(
            "[glrmask/profile][raw_visible_observations] active_groups={} active_words={} labels={} final_bits={} future_bits={} nonempty_final_states={} nonempty_future_states={}",
            active_groups.iter().filter(|&&active| active).count(),
            active_words.iter().filter(|&&word| word != 0).count(),
            next_id,
            final_bits,
            future_bits,
            nonempty_final_states,
            nonempty_future_states,
        );
    }
    RawObservationIds {
        ids,
        representatives,
        active_language,
    }
}

fn raw_active_language_states(tokenizer: &Tokenizer, active_groups: &[bool]) -> Vec<bool> {
    (0..tokenizer.num_states())
        .map(|state| {
            tokenizer
                .matched_terminals_iter(state)
                .chain(tokenizer.possible_future_terminals_iter(state))
                .any(|terminal| {
                    active_groups
                        .get(terminal as usize)
                        .copied()
                        .unwrap_or(false)
                })
        })
        .collect()
}

fn map_from_raw_classes(classes: &[u32]) -> ManyToOneIdMap {
    let class_count = classes.iter().copied().max().map_or(0usize, |id| id as usize + 1);
    let mut internal_to_originals = vec![Vec::new(); class_count];
    let mut representative_original_ids = vec![u32::MAX; class_count];
    for (state, &class) in classes.iter().enumerate() {
        let bucket = &mut internal_to_originals[class as usize];
        if bucket.is_empty() {
            representative_original_ids[class as usize] = state as u32;
        }
        bucket.push(state as u32);
    }
    ManyToOneIdMap {
        original_to_internal: classes.to_vec(),
        internal_to_originals,
        representative_original_ids,
    }
}

fn raw_map_is_relevant_byte_congruent(
    state_map: &ManyToOneIdMap,
    target_labels: &[u32],
    transitions: &[u32],
    active_bytes: &[u8],
    active_language: &[bool],
) -> bool {
    let projected_class = |source: usize, byte: u8| {
        if !active_language[source] {
            return u32::MAX;
        }
        let target = transitions[source * 256 + byte as usize];
        if target == u32::MAX || !active_language[target as usize] {
            u32::MAX
        } else {
            state_map.original_to_internal[target as usize]
        }
    };
    for (internal, members) in state_map.internal_to_originals.iter().enumerate() {
        let representative = state_map.representative_original_ids[internal] as usize;
        if representative >= target_labels.len()
            || state_map.original_to_internal[representative] != internal as u32
        {
            return false;
        }
        for &raw in members {
            let raw = raw as usize;
            if target_labels[raw] != target_labels[representative]
                || state_map.original_to_internal[raw] != internal as u32
            {
                return false;
            }
            for &byte in active_bytes {
                if projected_class(representative, byte) != projected_class(raw, byte) {
                    return false;
                }
            }
        }
    }
    true
}

/// Exact finite-token state map for vocabularies whose tokens all begin with
/// one of a small number of first bytes and have a nonempty suffix.
///
/// First minimize the raw scanner under the union of all suffix bytes. The
/// resulting classes retain frozen visible observations and are a right
/// congruence for every suffix continuation. A starting state is then fully
/// characterized by the class reached by each possible first byte. This is
/// exact for complete local-token walks: after consuming a first byte, every
/// remaining byte is in the minimized suffix alphabet.
pub fn compute_state_map_raw(
    tokenizer: &Tokenizer,
    transitions: &[u32],
    active_groups: &[bool],
    relevant_bytes: &[bool; 256],
    precomputed_observations: Option<(&[u32], &[u32])>,
) -> Option<RawRestrictedObservationResult> {
    let num_states = tokenizer.num_states() as usize;
    if transitions.len() != num_states * 256 || num_states == 0 {
        return None;
    }
    let active_bytes = active_byte_representatives(Some(relevant_bytes), None);
    let labels_started_at = Instant::now();
    let observations = match precomputed_observations {
        Some((ids, representatives)) if ids.len() == num_states => RawObservationIds {
            ids: ids.to_vec(),
            representatives: representatives.to_vec(),
            active_language: raw_active_language_states(tokenizer, active_groups),
        },
        _ => raw_target_label_ids(tokenizer, active_groups),
    };
    let target_labels = &observations.ids;
    let label_ms = labels_started_at.elapsed().as_secs_f64() * 1000.0;
    let refine_started_at = Instant::now();
    let active_language = &observations.active_language;
    let (classes, _, _) = refine_partition_rows(
        target_labels.clone(),
        |source| {
            active_bytes.iter().filter_map(move |&byte| {
                if !active_language[source] { return None; }
                let target = transitions[source * 256 + byte as usize];
                (target != u32::MAX && active_language[target as usize]).then_some((byte, target))
            })
        },
        None,
    );
    let refine_ms = refine_started_at.elapsed().as_secs_f64() * 1000.0;
    let state_map = map_from_raw_classes(&classes);
    debug_assert!(raw_map_is_relevant_byte_congruent(
        &state_map, target_labels, transitions, &active_bytes, active_language,
    ));
    Some(RawRestrictedObservationResult {
        state_map,
        raw_observation_ids: observations.ids,
        observation_representatives: observations.representatives,
        label_ms,
        refine_ms,
        certificate_ms: 0.0,
        rounds: 0,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::automata::lexer::Lexer;
    use crate::automata::lexer::compile::build_regex_monolithic as build_regex;

    fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
        let terminal_count = expressions.len() as u32;
        build_regex(&expressions).into_tokenizer(
            terminal_count,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    fn class_of(map: &ManyToOneIdMap, state: u32) -> u32 {
        map.original_to_internal[state as usize]
    }

    #[test]
    fn frozen_future_labels_survive_byte_restriction_and_obey_active_mask() {
        // `x` and `y` are deliberately absent from the restricted byte set.
        // The states after `a` and `b` nevertheless remain distinguishable
        // through their `c` successors' *original* future-terminal labels.
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"acx".to_vec()),
            Expr::U8Seq(b"bcy".to_vec()),
        ]);
        let start = tokenizer.initial_state_id();
        let after_a = tokenizer.get_transition(start, b'a');
        let after_b = tokenizer.get_transition(start, b'b');
        let after_ac = tokenizer.get_transition(after_a, b'c');
        let after_bc = tokenizer.get_transition(after_b, b'c');
        assert_ne!(after_a, u32::MAX);
        assert_ne!(after_b, u32::MAX);
        assert_ne!(after_ac, u32::MAX);
        assert_ne!(after_bc, u32::MAX);
        assert!(tokenizer.possible_future_terminals_iter(after_ac).any(|t| t == 0));
        assert!(tokenizer.possible_future_terminals_iter(after_bc).any(|t| t == 1));

        let mut only_c = [false; 256];
        only_c[b'c' as usize] = true;

        let all_active_view = TokenizerView::new_filtered(&tokenizer, &[true, true]);
        let all_active = compute_state_map(&all_active_view, &only_c, None, None, false);
        assert_ne!(
            class_of(&all_active, after_a),
            class_of(&all_active, after_b),
            "frozen future-terminal labels after `c` must remain observable even though x/y are restricted out",
        );

        let none_active_view = TokenizerView::new_filtered(&tokenizer, &[false, false]);
        let none_active = compute_state_map(&none_active_view, &only_c, None, None, false);
        assert_eq!(
            class_of(&none_active, after_a),
            class_of(&none_active, after_b),
            "active-terminal filtering must remove the only remaining observation",
        );

        let no_bytes = [false; 256];
        let no_byte_observation = compute_state_map(&all_active_view, &no_bytes, None, None, false);
        assert_eq!(
            class_of(&no_byte_observation, after_a),
            class_of(&no_byte_observation, after_b),
            "without c in the byte set the future labels are not reached by the recurrence",
        );
    }
}
