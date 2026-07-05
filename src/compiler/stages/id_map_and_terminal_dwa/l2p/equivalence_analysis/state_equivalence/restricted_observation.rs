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
use std::collections::VecDeque;
use std::time::Instant;

use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::super::compat::TokenizerView;
use super::max_length::active_byte_representatives;

const NO_CANDIDATE: usize = usize::MAX;

pub(crate) struct RawRestrictedObservationResult {
    pub(crate) state_map: ManyToOneIdMap,
    pub(crate) label_ms: f64,
    pub(crate) refine_ms: f64,
    pub(crate) certificate_ms: f64,
    pub(crate) rounds: usize,
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

fn same_candidate_partition(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut left_to_right = FxHashMap::<u32, u32>::default();
    let mut right_to_left = FxHashMap::<u32, u32>::default();
    for (&left_class, &right_class) in left.iter().zip(right) {
        match left_to_right.get(&left_class) {
            Some(&mapped) if mapped != right_class => return false,
            Some(_) => {}
            None => {
                left_to_right.insert(left_class, right_class);
            }
        }
        match right_to_left.get(&right_class) {
            Some(&mapped) if mapped != left_class => return false,
            Some(_) => {}
            None => {
                right_to_left.insert(right_class, left_class);
            }
        }
    }
    true
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
pub(crate) fn compute_state_map(
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

    // At depth zero every candidate has the same recursive characterization.
    let (mut current_classes, mut current_class_count) = if include_source_observation {
        let mut classes_by_source = FxHashMap::<(u32, bool), u32>::default();
        let mut classes = vec![0u32; num_candidates];
        for (candidate, &state) in candidate_representatives.iter().enumerate() {
            let has_any_transition = dfa.transitions_for(state).iter().any(|&target| target != u32::MAX);
            let next = classes_by_source.len() as u32;
            classes[candidate] = *classes_by_source
                .entry((target_labels[state], has_any_transition))
                .or_insert(next);
        }
        (classes, classes_by_source.len())
    } else {
        (vec![0u32; num_candidates], usize::from(num_candidates != 0))
    };
    let mut signature = vec![0u64; 1 + active_bytes.len()];

    for _ in 0..num_candidates {
        let mut next_classes = vec![0u32; num_candidates];
        let mut classes_by_signature = FxHashMap::<Vec<u64>, u32>::default();

        for (candidate, &state) in candidate_representatives.iter().enumerate() {
            signature[0] = current_classes[candidate] as u64;
            for (slot, &byte) in active_bytes.iter().enumerate() {
                let target = dfa.trans(state, byte as usize);
                signature[slot + 1] = if target == u32::MAX {
                    0
                } else {
                    let target_candidate = raw_to_candidate[target as usize];
                    debug_assert_ne!(target_candidate, NO_CANDIDATE);
                    let target_class = current_classes[target_candidate] as u64 + 1;
                    let labels = target_labels[target as usize] as u64 + 1;
                    (labels << 32) | target_class
                };
            }

            let next_class = classes_by_signature.len() as u32;
            let class = *classes_by_signature
                .entry(signature.clone())
                .or_insert(next_class);
            next_classes[candidate] = class;
        }

        let next_class_count = classes_by_signature.len();
        if next_class_count == current_class_count
            && same_candidate_partition(&current_classes, &next_classes)
        {
            return map_from_candidate_classes(
                &candidate_members,
                &candidate_representatives,
                &next_classes,
                num_states,
            );
        }
        current_classes = next_classes;
        current_class_count = next_class_count;
    }

    unreachable!("restricted-observation partition refinement did not stabilize");
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

fn raw_target_label_ids(tokenizer: &Tokenizer, active_groups: &[bool]) -> Vec<u32> {
    let active_words = active_mask_words(active_groups);
    let mut ids = vec![0u32; tokenizer.num_states() as usize];
    let mut buckets = FxHashMap::<u64, Vec<(Vec<u64>, u32)>>::default();
    let mut key = Vec::with_capacity(16);
    let mut next_id = 0u32;
    for state in 0..tokenizer.num_states() {
        visible_observation_key(tokenizer, state, &active_words, &mut key);
        let hash = hash_visible_observation_key(&key);
        let bucket = buckets.entry(hash).or_default();
        if let Some((_, id)) = bucket.iter().find(|(representative, _)| representative == &key) {
            ids[state as usize] = *id;
        } else {
            bucket.push((key.clone(), next_id));
            ids[state as usize] = next_id;
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
    ids
}

enum ExactSignatureBucket {
    Single { representative: usize, class: u32 },
    Collisions(Vec<(usize, u32)>),
}

/// Canonical sparse rows for the active vocabulary-byte alphabet. Dead edges
/// are omitted; their shared absence is implicit in every signature.
struct RawRelevantEdges {
    offsets: Vec<u32>,
    bytes: Vec<u8>,
    targets: Vec<u32>,
}

fn build_raw_relevant_edges(
    transitions: &[u32],
    num_states: usize,
    active_bytes: &[u8],
) -> RawRelevantEdges {
    let mut offsets = Vec::with_capacity(num_states + 1);
    let mut bytes = Vec::new();
    let mut targets = Vec::new();
    offsets.push(0);
    for state in 0..num_states {
        let base = state * 256;
        for &byte in active_bytes {
            let target = transitions[base + byte as usize];
            if target != u32::MAX {
                bytes.push(byte);
                targets.push(target);
            }
        }
        offsets.push(bytes.len() as u32);
    }
    RawRelevantEdges {
        offsets,
        bytes,
        targets,
    }
}

#[inline]
fn sparse_signature_hash(
    state: usize,
    current_classes: &[u32],
    edges: &RawRelevantEdges,
) -> u64 {
    let mut hash = mix_observation_hash(0x1319_8a2e_0370_7344, current_classes[state] as u64);
    let start = edges.offsets[state] as usize;
    let end = edges.offsets[state + 1] as usize;
    for index in start..end {
        let edge = ((edges.bytes[index] as u64) << 32)
            | (current_classes[edges.targets[index] as usize] as u64 + 1);
        hash = mix_observation_hash(hash, edge);
    }
    hash
}

fn same_sparse_signature(
    left: usize,
    right: usize,
    current_classes: &[u32],
    edges: &RawRelevantEdges,
) -> bool {
    if current_classes[left] != current_classes[right] {
        return false;
    }
    let left_start = edges.offsets[left] as usize;
    let left_end = edges.offsets[left + 1] as usize;
    let right_start = edges.offsets[right] as usize;
    let right_end = edges.offsets[right + 1] as usize;
    if left_end - left_start != right_end - right_start {
        return false;
    }
    (0..left_end - left_start).all(|offset| {
        edges.bytes[left_start + offset] == edges.bytes[right_start + offset]
            && current_classes[edges.targets[left_start + offset] as usize]
                == current_classes[edges.targets[right_start + offset] as usize]
    })
}

fn blocks_from_classes(classes: &[u32], class_count: usize) -> Vec<Vec<u32>> {
    let mut blocks = vec![Vec::new(); class_count];
    for (state, &class) in classes.iter().enumerate() {
        blocks[class as usize].push(state as u32);
    }
    blocks
}

fn build_sparse_inverse_edges(edges: &RawRelevantEdges, num_states: usize) -> Vec<Vec<(u8, u32)>> {
    let mut inverse = vec![Vec::new(); num_states];
    for source in 0..num_states {
        let start = edges.offsets[source] as usize;
        let end = edges.offsets[source + 1] as usize;
        for index in start..end {
            inverse[edges.targets[index] as usize].push((edges.bytes[index], source as u32));
        }
    }
    inverse
}

/// Exact Hopcroft refinement for the partial DFA induced by active vocabulary
/// bytes. An absent edge is one common implicit missing target, so predecessors
/// of every materialized block are sufficient to split all observable states.
fn hopcroft_refine_sparse(initial_classes: &[u32], edges: &RawRelevantEdges) -> Vec<u32> {
    let num_states = initial_classes.len();
    let mut partition = initial_classes.to_vec();
    let initial_count = partition
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut blocks = blocks_from_classes(&partition, initial_count);
    let inverse = build_sparse_inverse_edges(edges, num_states);

    let mut worklist: VecDeque<u32> = (0..blocks.len() as u32).collect();
    let mut in_worklist = vec![true; blocks.len()];
    let mut source_set = vec![false; num_states];
    let mut sources_to_clear = Vec::<u32>::with_capacity(num_states.min(10_000));
    let mut touched_blocks = Vec::<u32>::with_capacity(1024);
    let mut block_touched = vec![false; blocks.len()];
    let mut block_sources = vec![Vec::<u32>::new(); blocks.len()];
    let mut input_sources = vec![Vec::<u32>::new(); 256];
    let mut touched_inputs = Vec::<u8>::with_capacity(64);

    while let Some(splitter_block) = worklist.pop_front() {
        let splitter = splitter_block as usize;
        if splitter >= blocks.len() || blocks[splitter].is_empty() {
            continue;
        }
        in_worklist[splitter] = false;
        touched_inputs.clear();
        {
            let splitter_states = &blocks[splitter];
            for &target in splitter_states {
                for &(byte, source) in &inverse[target as usize] {
                    let sources = &mut input_sources[byte as usize];
                    if sources.is_empty() {
                        touched_inputs.push(byte);
                    }
                    sources.push(source);
                }
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
                let move_sources = source_count <= block_len - source_count;
                let old_members = std::mem::take(&mut blocks[block]);
                let (remaining, moved) = if move_sources {
                    let mut remaining = Vec::with_capacity(block_len - source_count);
                    for state in old_members {
                        if !source_set[state as usize] {
                            remaining.push(state);
                        }
                    }
                    (remaining, std::mem::take(&mut block_sources[block]))
                } else {
                    let mut moved = Vec::with_capacity(block_len - source_count);
                    for state in old_members {
                        if !source_set[state as usize] {
                            moved.push(state);
                        }
                    }
                    (std::mem::take(&mut block_sources[block]), moved)
                };
                for &state in &moved {
                    partition[state as usize] = new_block as u32;
                }
                blocks[block] = remaining;
                blocks.push(moved);
                in_worklist.push(false);
                block_touched.push(false);
                block_sources.push(Vec::new());
                if in_worklist[block] {
                    in_worklist[new_block] = true;
                    worklist.push_back(new_block as u32);
                } else if blocks[block].len() <= blocks[new_block].len() {
                    in_worklist[block] = true;
                    worklist.push_back(block as u32);
                } else {
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
) -> bool {
    for (internal, members) in state_map.internal_to_originals.iter().enumerate() {
        let representative = state_map.representative_original_ids[internal] as usize;
        if representative >= target_labels.len()
            || state_map.original_to_internal[representative] != internal as u32
        {
            return false;
        }
        let representative_base = representative * 256;
        for &raw in members {
            let raw = raw as usize;
            if target_labels[raw] != target_labels[representative]
                || state_map.original_to_internal[raw] != internal as u32
            {
                return false;
            }
            let raw_base = raw * 256;
            for &byte in active_bytes {
                let representative_target = transitions[representative_base + byte as usize];
                let raw_target = transitions[raw_base + byte as usize];
                let representative_class = if representative_target == u32::MAX { u32::MAX } else { state_map.original_to_internal[representative_target as usize] };
                let raw_class = if raw_target == u32::MAX { u32::MAX } else { state_map.original_to_internal[raw_target as usize] };
                if representative_class != raw_class {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) fn compute_state_map_raw(
    tokenizer: &Tokenizer,
    transitions: &[u32],
    active_groups: &[bool],
    relevant_bytes: &[bool; 256],
) -> Option<RawRestrictedObservationResult> {
    let num_states = tokenizer.num_states() as usize;
    if transitions.len() != num_states * 256 || num_states == 0 {
        return None;
    }
    let active_bytes = active_byte_representatives(Some(relevant_bytes), None);
    let labels_started_at = Instant::now();
    let target_labels = raw_target_label_ids(tokenizer, active_groups);
    let label_ms = labels_started_at.elapsed().as_secs_f64() * 1000.0;
    let refine_started_at = Instant::now();
    let mut current_classes = target_labels.clone();
    let mut current_class_count = current_classes
        .iter()
        .copied()
        .max()
        .map_or(0usize, |id| id as usize + 1);
    let edges = build_raw_relevant_edges(transitions, num_states, &active_bytes);
    // Hopcroft is the default for the large raw-coordinate case. The
    // signature fixed point remains available as a diagnostic reference.
    if std::env::var_os("GLRMASK_RAW_RESTRICTED_SIGNATURE_REFINE").is_none() {
        let classes = hopcroft_refine_sparse(&target_labels, &edges);
        let refine_ms = refine_started_at.elapsed().as_secs_f64() * 1000.0;
        let state_map = map_from_raw_classes(&classes);
        debug_assert!(raw_map_is_relevant_byte_congruent(
            &state_map,
            &target_labels,
            transitions,
            &active_bytes,
        ));
        return Some(RawRestrictedObservationResult {
            state_map,
            label_ms,
            refine_ms,
            certificate_ms: 0.0,
            rounds: 0,
        });
    }
    for round in 0..num_states {
        let mut next_classes = vec![0u32; num_states];
        let mut buckets = FxHashMap::<u64, ExactSignatureBucket>::default();
        let mut next_class_count = 0u32;
        for state in 0..num_states {
            let hash = sparse_signature_hash(state, &current_classes, &edges);
            let class = match buckets.get_mut(&hash) {
                None => {
                    let class = next_class_count;
                    next_class_count += 1;
                    buckets.insert(hash, ExactSignatureBucket::Single {
                        representative: state,
                        class,
                    });
                    class
                }
                Some(ExactSignatureBucket::Single {
                    representative,
                    class,
                }) => {
                    if same_sparse_signature(*representative, state, &current_classes, &edges) {
                        *class
                    } else {
                        let previous = (*representative, *class);
                        let class = next_class_count;
                        next_class_count += 1;
                        *buckets.get_mut(&hash).expect("bucket exists") =
                            ExactSignatureBucket::Collisions(vec![previous, (state, class)]);
                        class
                    }
                }
                Some(ExactSignatureBucket::Collisions(entries)) => {
                    if let Some((_, class)) = entries.iter().find(|&&(representative, _)| {
                        same_sparse_signature(representative, state, &current_classes, &edges)
                    }) {
                        *class
                    } else {
                        let class = next_class_count;
                        next_class_count += 1;
                        entries.push((state, class));
                        class
                    }
                }
            };
            next_classes[state] = class;
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][raw_restricted_round] round={} classes={}",
                round + 1,
                next_class_count,
            );
        }
        if next_class_count as usize == current_class_count
            && same_candidate_partition(&current_classes, &next_classes)
        {
            let refine_ms = refine_started_at.elapsed().as_secs_f64() * 1000.0;
            let state_map = map_from_raw_classes(&next_classes);
            debug_assert!(raw_map_is_relevant_byte_congruent(
                &state_map,
                &target_labels,
                transitions,
                &active_bytes,
            ));
            return Some(RawRestrictedObservationResult {
                state_map,
                label_ms,
                refine_ms,
                certificate_ms: 0.0,
                rounds: round + 1,
            });
        }
        current_classes = next_classes;
        current_class_count = next_class_count as usize;
    }
    None


}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::automata::lexer::Lexer;
    use crate::automata::lexer::compile::build_regex;

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
