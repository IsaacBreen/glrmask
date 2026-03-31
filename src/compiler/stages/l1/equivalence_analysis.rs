//! L1 equivalence analysis: lightweight id_map for terminals with max path length ≤ 1.
//!
//! Uses ONLY max_length state analysis and byte-class token deduplication.
//! No vocab equivalence, no further state refinement.
//! This produces a coarser partition (more state/token classes) but is very fast.

use crate::compiler::stages::equivalence_analysis::compat::TokenizerView;
use crate::compiler::stages::equivalence_analysis::combined_equivalence_analysis::hash_byte_class_seq;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::Vocab;

fn compile_profile_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_COMPILE")
        .or_else(|_| std::env::var("GLRMASK_PROFILE_COMPILE_SUMMARY"))
        .map(|v| {
            let t = v.trim();
            !t.is_empty() && t != "0" && !t.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn elapsed_ms(started_at: std::time::Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

/// Build an L1-only id_map using max_length state analysis + byte-class token dedup.
///
/// This is much faster than full vocab equivalence but produces coarser classes.
/// Suitable for terminals with max path length ≤ 1.
pub fn build_l1_id_map(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> InternalIdMap {
    let profile = compile_profile_enabled();
    let total_started_at = std::time::Instant::now();

    let tokenizer_view = TokenizerView::new(tokenizer);
    let dfa = tokenizer_view.dfa();
    let num_states = tokenizer.num_states() as usize;

    // --- Byte-class token deduplication ---
    let dedup_started_at = std::time::Instant::now();
    let byte_to_class = crate::compiler::stages::equivalence_analysis::compat::compute_byte_classes(dfa);

    let max_token_id = vocab.max_token_id();
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    let mut token_bytes_list: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in &vocab.entries {
        token_ids.push(tid);
        token_bytes_list.push(bytes.as_slice());
    }

    // Dedup tokens by byte-class sequence.
    let mut bc_hash_to_repr: hashbrown::HashMap<u128, usize> = hashbrown::HashMap::with_capacity(token_bytes_list.len() / 2);
    let mut repr_indices: Vec<usize> = Vec::new();
    let mut original_to_repr: Vec<usize> = Vec::with_capacity(token_bytes_list.len());
    for (idx, &bytes) in token_bytes_list.iter().enumerate() {
        let h = hash_byte_class_seq(bytes, &byte_to_class);
        let repr = *bc_hash_to_repr.entry(h).or_insert_with(|| {
            let r = repr_indices.len();
            repr_indices.push(idx);
            r
        });
        original_to_repr.push(repr);
    }
    let num_repr = repr_indices.len();
    let repr_token_bytes: Vec<&[u8]> = repr_indices.iter().map(|&idx| token_bytes_list[idx]).collect();
    let dedup_ms = elapsed_ms(dedup_started_at);

    // --- Max-length state equivalence ---
    let max_length_started_at = std::time::Instant::now();
    let initial_states: Vec<usize> = (0..num_states).collect();
    let state_reps = crate::compiler::stages::equivalence_analysis::state::max_length::find_state_equivalence_classes(
        &tokenizer_view,
        &repr_token_bytes,
        &initial_states,
    );
    let max_length_ms = elapsed_ms(max_length_started_at);

    // Build state map.
    let mut state_class_map: hashbrown::HashMap<usize, u32> = hashbrown::HashMap::new();
    let mut state_original_to_internal = vec![0u32; num_states];
    let mut state_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut state_representative_ids: Vec<u32> = Vec::new();

    for (state_idx, &rep) in state_reps.iter().enumerate() {
        let internal = *state_class_map.entry(rep).or_insert_with(|| {
            let id = state_internal_to_originals.len() as u32;
            state_internal_to_originals.push(Vec::new());
            state_representative_ids.push(rep as u32);
            id
        });
        state_original_to_internal[state_idx] = internal;
        state_internal_to_originals[internal as usize].push(state_idx as u32);
    }
    let num_state_classes = state_internal_to_originals.len();

    // --- Token classes: each dedup representative is its own class ---
    let mut token_original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut token_internal_to_originals: Vec<Vec<u32>> = vec![Vec::new(); num_repr];
    let mut token_representative_ids: Vec<u32> = vec![u32::MAX; num_repr];

    for (orig_idx, &tid) in token_ids.iter().enumerate() {
        let repr = original_to_repr[orig_idx];
        let class = repr as u32;
        token_original_to_internal[tid as usize] = class;
        token_internal_to_originals[class as usize].push(tid);
        if tid < token_representative_ids[class as usize] {
            token_representative_ids[class as usize] = tid;
        }
    }
    let num_token_classes = num_repr;

    if profile {
        eprintln!(
            "[glrmask/profile][l1_id_map] dedup_ms={:.3} tokens={}->{} max_length_ms={:.3} states={}->{} token_classes={} total_ms={:.3}",
            dedup_ms,
            vocab.len(),
            num_repr,
            max_length_ms,
            num_states,
            num_state_classes,
            num_token_classes,
            elapsed_ms(total_started_at),
        );
    }

    InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_original_to_internal,
            internal_to_originals: state_internal_to_originals,
            representative_original_ids: state_representative_ids,
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: token_original_to_internal,
            internal_to_originals: token_internal_to_originals,
            representative_original_ids: token_representative_ids,
        },
    }
}
