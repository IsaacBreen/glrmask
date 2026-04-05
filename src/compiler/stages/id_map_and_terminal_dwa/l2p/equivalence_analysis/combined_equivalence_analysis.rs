//! Combined state and vocab equivalence analysis.
//!
//! State representatives are computed first, then vocab equivalence runs only
//! on the surviving representative set.

use std::collections::{BTreeMap, BTreeSet};

use hashbrown::HashMap;

use super::compat::TokenizerView;
use crate::ds::bitset::BitSet;

use super::state::fast::{self as state_equivalence_analysis, StateEquivalenceResult};
use super::vocab::fast::{self as vocab_equivalence_analysis, VocabEquivalenceResult};
use super::vocab::slow::partition_is_at_least_as_fine;

// Identity hasher for pre-hashed u128 keys: avoids redundant hashing in HashMap.
// Only valid for keys that are already well-distributed.
struct PreHashedU128Hasher(u64);

impl std::hash::Hasher for PreHashedU128Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("PreHashedU128Hasher only supports write_u128");
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.0 = i as u64;
    }
}

impl Default for PreHashedU128Hasher {
    fn default() -> Self {
        PreHashedU128Hasher(0)
    }
}

type PreHashedU128BuildHasher = std::hash::BuildHasherDefault<PreHashedU128Hasher>;

const REFERENCE_EQUIV_VERIFICATION_ENV: &str = "REFERENCE_EQUIV_VERIFICATION";
const USE_REFERENCE_EQUIV_ENV: &str = "GLRMASK_USE_REFERENCE_EQUIV";
const SKIP_MAX_LENGTH_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_MAX_LENGTH_STATE_EQUIV";
const SKIP_TOKEN_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_TOKEN_STATE_EQUIV";
const USE_SLOW_VOCAB_EQUIV_ENV: &str = "GLRMASK_USE_SLOW_VOCAB_EQUIV";
const FORCE_PRE_VOCAB_STATE_REDUCTION_ENV: &str = "GLRMASK_FORCE_PRE_VOCAB_STATE_REDUCTION";
const DISABLE_PRE_VOCAB_STATE_REDUCTION_ENV: &str = "GLRMASK_DISABLE_PRE_VOCAB_STATE_REDUCTION";
const SKIP_MAX_LENGTH_SMALL_STATE_THRESHOLD: usize = 128;
const PRE_VOCAB_STATE_REDUCTION_MIN_STATES: usize = 200;
const PRE_VOCAB_STATE_REDUCTION_MAX_GROUPS: usize = 64;
/// Only run pre-vocab state reduction when the deduped token count is high
/// enough that the vocab signature pass is expensive. With few tokens, the
/// vocab pass is already cheap and pre-reduction adds overhead.
const PRE_VOCAB_STATE_REDUCTION_MIN_TOKENS: usize = 5000;
/// When the deduped token count exceeds this, limit state reduction to a single
/// batch (5000 tokens) to avoid the cost of processing the full token set.
const PRE_VOCAB_STATE_REDUCTION_MAX_FULL_TOKENS: usize = 5000;

/// Result of combined equivalence analysis.
pub struct CombinedEquivalenceResult {
    /// Vocab equivalence classes: sets of token indices that behave identically.
    pub vocab_classes: VocabEquivalenceResult,

    /// State equivalence classes: sets of state IDs that behave identically.
    pub state_classes: StateEquivalenceResult,
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn compile_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE") || env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
}

fn debug_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_DEBUG_PROFILE")
}

fn elapsed_ms(started_at: std::time::Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn verify_state_partition_reference(
    fast_state_classes: &StateEquivalenceResult,
    reference_state_classes: &StateEquivalenceResult,
) {
    let fast_state_classes: BTreeSet<Vec<_>> = fast_state_classes
        .iter()
        .map(|class| class.iter().copied().collect())
        .collect();
    let reference_state_classes: BTreeSet<Vec<_>> = reference_state_classes
        .iter()
        .map(|class| class.iter().copied().collect())
        .collect();
    assert!(
        partition_is_at_least_as_fine(&fast_state_classes, &reference_state_classes),
        "Fast state equivalence merged states that reference kept separate!\n\
         Fast classes: {}\n\
         Reference classes: {}",
        fast_state_classes.len(),
        reference_state_classes.len(),
    );
}

fn verify_vocab_partition_reference_with_tokens<S: AsRef<[u8]>>(
    fast_vocab_classes: &VocabEquivalenceResult,
    reference_vocab_classes: &VocabEquivalenceResult,
    tokens: &[S],
    original_to_repr: &[usize],
) {
    if partition_is_at_least_as_fine(fast_vocab_classes, reference_vocab_classes) {
        return;
    }

    // Find which fast classes incorrectly merge tokens from different reference classes
    // Build reverse map: token -> reference class index
    let mut token_to_ref_class: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (ref_class_idx, ref_class) in reference_vocab_classes.iter().enumerate() {
        for &token_idx in ref_class {
            token_to_ref_class.insert(token_idx, ref_class_idx);
        }
    }

    let mut bad_classes_count = 0;
    for (fast_class_idx, fast_class) in fast_vocab_classes.iter().enumerate() {
        let ref_classes_in_fast: BTreeSet<usize> = fast_class
            .iter()
            .filter_map(|t| token_to_ref_class.get(t).copied())
            .collect();
        if ref_classes_in_fast.len() > 1 {
            bad_classes_count += 1;
            if bad_classes_count <= 5 {
                eprintln!(
                    "[verify_vocab] Fast class {} (size={}) spans {} reference classes: {:?}",
                    fast_class_idx,
                    fast_class.len(),
                    ref_classes_in_fast.len(),
                    ref_classes_in_fast,
                );
                // Show first few tokens from each reference class
                for &ref_class_idx in &ref_classes_in_fast {
                    let tokens_in_this_ref: Vec<usize> = fast_class
                        .iter()
                        .filter(|t| token_to_ref_class.get(t) == Some(&ref_class_idx))
                        .copied()
                        .collect();
                    eprintln!(
                        "  ref_class {}: {} tokens, first 5: {:?}",
                        ref_class_idx,
                        tokens_in_this_ref.len(),
                        &tokens_in_this_ref[..tokens_in_this_ref.len().min(5)],
                    );
                    // Show token bytes and dedup representatives for diagnosis
                    for &tok_idx in &tokens_in_this_ref[..tokens_in_this_ref.len().min(5)] {
                        let bytes = tokens[tok_idx].as_ref();
                        let repr_idx = original_to_repr[tok_idx];
                        eprintln!(
                            "    token[{}]: bytes={:?} repr_idx={} str={:?}",
                            tok_idx,
                            bytes,
                            repr_idx,
                            String::from_utf8_lossy(bytes),
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "[verify_vocab] Total bad fast classes: {} / {}",
        bad_classes_count,
        fast_vocab_classes.len(),
    );

    panic!(
        "Fast vocab equivalence merged tokens that reference kept separate!\n\
         Fast classes: {}\n\
         Reference classes: {}",
        fast_vocab_classes.len(),
        reference_vocab_classes.len(),
    );
}

fn collect_representative_states(states: &[usize]) -> Vec<usize> {
    states.iter().copied().collect::<BTreeSet<_>>().into_iter().collect()
}

/// Compute which groups can be skipped in state equivalence hashing.
///
/// A group is "universally disallowed" if it appears in the disallowed set
/// of EVERY other group, meaning it can never follow any match.
fn compute_skip_groups(num_groups: usize, disallowed_follows: &BTreeMap<u32, BitSet>) -> Vec<bool> {
    let mut skip = vec![false; num_groups];
    for gid in 0..num_groups {
        let is_disallowed_by_all = (0..num_groups).all(|other| {
            disallowed_follows
                .get(&(other as u32))
                .map_or(false, |bs| bs.contains(gid))
        });
        if is_disallowed_by_all {
            skip[gid] = true;
        }
    }
    skip
}

fn tokenizer_group_count(tokenizer: &TokenizerView) -> usize {
    tokenizer
        .dfa()
        .states
        .iter()
        .flat_map(|state| {
            state
                .finalizers
                .iter()
                .copied()
                .chain(state.possible_future_group_ids.iter().copied())
        })
        .max()
        .map_or(0, |max_group| max_group + 1)
}

/// Compute byte equivalence classes from the tokenizer DFA.
///
/// Bytes with identical transitions across all DFA states are merged into
/// the same class. This is used to deduplicate tokens before equivalence
/// analysis: tokens whose byte-class sequences are identical will always
/// produce the same DFA behavior from any starting state.
/// Token deduplication result.
struct TokenDedup<'a> {
    /// Byte slices for representative tokens (references into the original array).
    representative_token_bytes: Vec<&'a [u8]>,
    /// For each original token index, the index of its representative.
    original_to_repr: Vec<usize>,
}

/// Hash a token's byte-class sequence into a u128 for dedup.
/// Collision probability is ~n²/2^128 ≈ 0 for any practical n.
#[inline]
pub(crate) fn hash_byte_class_seq(bytes: &[u8], byte_to_class: &[u8; 256]) -> u128 {
    // Length-prefixed hash with a good mixing function.
    let mut h: u128 = 0xFF51_AFD7_ED55_8CCD;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53).wrapping_add(bytes.len() as u128);
    for &b in bytes {
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(byte_to_class[b as usize] as u128);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 29;
    h
}

/// Deduplicate tokens by their byte-class sequence.
///
/// Tokens whose bytes map to the same sequence of byte classes under the
/// tokenizer DFA will always produce identical DFA trajectories from any
/// starting state. We only need to analyze one representative per group.
fn deduplicate_tokens_by_byte_class<'a, S: AsRef<[u8]>>(
    tokens: &'a [S],
    byte_to_class: &[u8; 256],
) -> TokenDedup<'a> {
    let mut hash_to_repr: HashMap<u128, usize, PreHashedU128BuildHasher> =
        HashMap::with_capacity_and_hasher(tokens.len() / 2, PreHashedU128BuildHasher::default());
    let mut representative_token_bytes: Vec<&'a [u8]> = Vec::new();
    let mut original_to_repr: Vec<usize> = Vec::with_capacity(tokens.len());

    for token in tokens {
        let bytes = token.as_ref();
        let h = hash_byte_class_seq(bytes, byte_to_class);
        let repr_idx = *hash_to_repr.entry(h).or_insert_with(|| {
            let idx = representative_token_bytes.len();
            representative_token_bytes.push(bytes);
            idx
        });
        original_to_repr.push(repr_idx);
    }

    TokenDedup {
        representative_token_bytes,
        original_to_repr,
    }
}

/// Expand vocab equivalence classes from representative indices back to
/// original token indices.
fn expand_vocab_classes(
    dedup_classes: VocabEquivalenceResult,
    original_to_repr: &[usize],
    num_representatives: usize,
) -> VocabEquivalenceResult {
    let mut repr_to_class = vec![usize::MAX; num_representatives];
    let mut original_classes: Vec<Vec<usize>> = Vec::with_capacity(dedup_classes.len());

    for (class_idx, dedup_class) in dedup_classes.iter().enumerate() {
        for &dedup_idx in dedup_class {
            repr_to_class[dedup_idx] = class_idx;
        }
        original_classes.push(Vec::new());
    }

    for (original_idx, &repr_idx) in original_to_repr.iter().enumerate() {
        original_classes[repr_to_class[repr_idx]].push(original_idx);
    }

    original_classes.into_iter().collect()
}

fn representative_tokens_for_vocab_classes<'a>(
    dedup_vocab_classes: &VocabEquivalenceResult,
    representative_token_bytes: &'a [&'a [u8]],
) -> Vec<&'a [u8]> {
    dedup_vocab_classes
        .iter()
        .map(|dedup_class| representative_token_bytes[dedup_class[0]])
        .collect()
}

/// Compute combined state and vocab equivalence analysis.
///
/// This function:
/// 1. Computes state equivalence classes to find representative states
/// 2. Runs vocab equivalence analysis only on representative states
///
/// # Arguments
/// * `tokenizer` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to analyze
/// * `initial_states` - Initial tokenizer state IDs to consider
///
/// # Returns
/// Combined result containing vocab classes and state classes.
pub fn compute_combined_equivalence<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
) -> CombinedEquivalenceResult {
    compute_combined_equivalence_with_group_filter(
        tokenizer, tokens, initial_states, disallowed_follows, ignore_terminal, None, None,
    )
}

pub fn compute_combined_equivalence_with_group_filter<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&vocab_equivalence_analysis::SharedVocabDfaCache>,
) -> CombinedEquivalenceResult {
    let skip_max_length = env_flag_enabled(SKIP_MAX_LENGTH_STATE_EQUIV_ENV)
        || initial_states.len() <= SKIP_MAX_LENGTH_SMALL_STATE_THRESHOLD;
    let skip_token_state = env_flag_enabled(SKIP_TOKEN_STATE_EQUIV_ENV);
    let profile_compile = compile_profile_enabled();
    let debug_profile = debug_profile_enabled();
    let combined_started_at = std::time::Instant::now();

    // Eagerly initialize the shared DFA cache (if provided) so that both
    // the dedup and build_dfa steps can reuse byte_to_class, trans_by_class,
    // and self_loop_bytes. Since filter_for_terminals preserves transitions,
    // the first partition's cache is valid for all.
    if let Some(cache) = shared_vocab_dfa_cache {
        cache.get_or_init(|| vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa(tokenizer.dfa()));
    }

    // Deduplicate tokens by byte-class sequence. Tokens whose bytes map
    // to the same DFA byte-class sequence behave identically from every
    // starting state, so we only need to analyze one representative.
    let dedup_started_at = std::time::Instant::now();
    let byte_to_class = shared_vocab_dfa_cache
        .and_then(|cache| cache.get())
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer.dfa()));
    let dedup = deduplicate_tokens_by_byte_class(tokens, &byte_to_class);
    let dedup_ms = elapsed_ms(dedup_started_at);

    let max_length_started_at = std::time::Instant::now();
    let pre_state_reps = if skip_max_length {
        initial_states.to_vec()
    } else {
        super::state::max_length::find_state_equivalence_classes_byte_restricted(
            tokenizer,
            &dedup.representative_token_bytes,
            initial_states,
            Some(&byte_to_class),
        )
    };
    let max_length_ms = elapsed_ms(max_length_started_at);

    let pre_reduced_states = collect_representative_states(&pre_state_reps);
    let tokenizer_num_groups = tokenizer_group_count(tokenizer);
    let force_pre_vocab_state_reduction = env_flag_enabled(FORCE_PRE_VOCAB_STATE_REDUCTION_ENV);
    let disable_pre_vocab_state_reduction = env_flag_enabled(DISABLE_PRE_VOCAB_STATE_REDUCTION_ENV);
    let use_pre_vocab_state_reduction = if force_pre_vocab_state_reduction {
        true
    } else if disable_pre_vocab_state_reduction {
        false
    } else {
        !skip_token_state
            && pre_reduced_states.len() >= PRE_VOCAB_STATE_REDUCTION_MIN_STATES
            && tokenizer_num_groups <= PRE_VOCAB_STATE_REDUCTION_MAX_GROUPS
            && dedup.representative_token_bytes.len() >= PRE_VOCAB_STATE_REDUCTION_MIN_TOKENS
    };
    let vocab_states = if use_pre_vocab_state_reduction {
        let pre_vocab_state_started_at = std::time::Instant::now();

        // NOTE: disallowed_follows cannot be used to skip groups in the state
        // equivalence hash because "universally disallowed" groups can still be
        // the FIRST match in a sequence. Skipping them would incorrectly merge
        // states that differ in first-match behavior. Context-dependent filtering
        // (per-parent-edge) would be correct but prohibitively expensive.
        let reduced_state_reps = state_equivalence_analysis::find_state_equivalence_classes_ex(
            tokenizer,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            &[], // skip_groups
            None, // no batch limit — process until convergence
            None, // default batch_size
            Some(true), // early_stop: halt once classes stabilize for 2 batches
        );
        let vocab_states = collect_representative_states(&reduced_state_reps);
        if profile_compile {
            eprintln!(
                "[glrmask/profile][pre_vocab_state_reduction] input_states={} reduced_states={} tokens={} num_groups={} ms={:.3}",
                pre_reduced_states.len(),
                vocab_states.len(),
                dedup.representative_token_bytes.len(),
                tokenizer_num_groups,
                elapsed_ms(pre_vocab_state_started_at),
            );
        }
        vocab_states
    } else {
        pre_reduced_states.clone()
    };

    let use_slow_vocab = env_flag_enabled(USE_SLOW_VOCAB_EQUIV_ENV);
    let vocab_started_at = std::time::Instant::now();
    let dedup_vocab_classes = if use_slow_vocab {
        super::vocab::slow::find_vocab_equivalence_classes_with_follow(
            tokenizer,
            &dedup.representative_token_bytes,
            &vocab_states,
            disallowed_follows,
        )
    } else {
        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            tokenizer,
            &dedup.representative_token_bytes,
            &vocab_states,
            disallowed_follows,
            Some(&byte_to_class),
            active_groups,
            shared_vocab_dfa_cache,
        )
    };
    let vocab_ms = elapsed_ms(vocab_started_at);

    // Running vocab first shrinks the token set before token_state refinement.
    // Tokens in the same vocab class are behaviorally identical across the
    // surviving states, so one representative token per class is sufficient
    // for the state refinement pass.
    let token_state_started_at = std::time::Instant::now();
    let representative_states = if skip_token_state {
        pre_state_reps.clone()
    } else {
        let vocab_representative_tokens = representative_tokens_for_vocab_classes(
            &dedup_vocab_classes,
            &dedup.representative_token_bytes,
        );
        let reduced_state_reps = state_equivalence_analysis::find_state_equivalence_classes(
            tokenizer,
            &vocab_representative_tokens,
            &pre_reduced_states,
        );
        let rep_to_final: BTreeMap<usize, usize> = pre_reduced_states
            .iter()
            .copied()
            .zip(reduced_state_reps)
            .collect();
        pre_state_reps
            .iter()
            .map(|pre_rep| rep_to_final[pre_rep])
            .collect()
    };
    let token_state_ms = elapsed_ms(token_state_started_at);

    // Expand dedup vocab classes back to original token indices.
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes.clone(),
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );

    let reduced_states = collect_representative_states(&representative_states);
    let state_classes = state_equivalence_analysis::mapping_to_equivalence_classes(
        initial_states,
        &representative_states,
    );

    if env_flag_enabled(USE_REFERENCE_EQUIV_ENV) {
        let reference = super::reference::find_equivalence_classes(
            tokenizer,
            tokens,
            initial_states,
            disallowed_follows,
            ignore_terminal.map(|terminal| terminal as usize),
        );
        return CombinedEquivalenceResult {
            vocab_classes: reference.vocab_classes,
            state_classes: reference.state_classes,
        };
    }

    if env_flag_enabled(REFERENCE_EQUIV_VERIFICATION_ENV) {
        let reference = super::reference::find_equivalence_classes(
            tokenizer,
            tokens,
            initial_states,
            disallowed_follows,
            ignore_terminal.map(|terminal| terminal as usize),
        );
        verify_state_partition_reference(&state_classes, &reference.state_classes);

        // Before checking expanded classes, diagnose at dedup level
        // Build reference on original tokens, then check if dedup expansion introduces errors
        let ref_token_to_class: std::collections::HashMap<usize, usize> = reference.vocab_classes
            .iter()
            .enumerate()
            .flat_map(|(ci, class)| class.iter().map(move |&ti| (ti, ci)))
            .collect();

        // Check for dedup hash collisions: tokens mapped to the same repr but in different ref classes
        let mut repr_ref_classes: std::collections::HashMap<usize, BTreeSet<usize>> = std::collections::HashMap::new();
        for (orig_idx, &repr_idx) in dedup.original_to_repr.iter().enumerate() {
            if let Some(&ref_class) = ref_token_to_class.get(&orig_idx) {
                repr_ref_classes.entry(repr_idx).or_default().insert(ref_class);
            }
        }
        let collision_reprs: Vec<(usize, BTreeSet<usize>)> = repr_ref_classes.iter()
            .filter(|(_, ref_classes)| ref_classes.len() > 1)
            .map(|(&k, v)| (k, v.clone()))
            .collect();
        if !collision_reprs.is_empty() {
            eprintln!("[verify_dedup] DEDUP HASH COLLISION: {} representatives span multiple reference classes!", collision_reprs.len());
            for (repr_idx, ref_classes) in &collision_reprs {
                let repr_bytes = dedup.representative_token_bytes[*repr_idx];
                eprintln!("  repr[{}] bytes={:?} str={:?} spans ref_classes {:?}", repr_idx, repr_bytes, String::from_utf8_lossy(repr_bytes), ref_classes);
                // Show all original tokens mapped to this repr
                for (orig_idx, &r) in dedup.original_to_repr.iter().enumerate() {
                    if r == *repr_idx {
                        let orig_bytes = tokens[orig_idx].as_ref();
                        let orig_ref = ref_token_to_class.get(&orig_idx);
                        let orig_hash = hash_byte_class_seq(orig_bytes, &byte_to_class);
                        let repr_hash = hash_byte_class_seq(repr_bytes, &byte_to_class);
                        eprintln!("    orig[{}] bytes={:?} str={:?} ref_class={:?} hash={:#x} repr_hash={:#x} hash_match={}",
                            orig_idx, orig_bytes, String::from_utf8_lossy(orig_bytes), orig_ref, orig_hash, repr_hash, orig_hash == repr_hash);
                    }
                }
            }
        } else {
            eprintln!("[verify_dedup] No dedup hash collisions found — bug is in vocab signature");
            
            // Diagnose: find two representative tokens that are in the same fast dedup class
            // but different reference classes
            let ref_token_to_ref_class: std::collections::HashMap<usize, usize> = reference.vocab_classes.iter()
                .enumerate()
                .flat_map(|(ci, class)| class.iter().map(move |&ti| (ti, ci)))
                .collect();
            
            // Build repr→dedup_class mapping
            let mut repr_to_dedup_class: Vec<usize> = vec![usize::MAX; dedup.representative_token_bytes.len()];
            for (class_idx, dedup_class) in dedup_vocab_classes.iter().enumerate() {
                for &dedup_idx in dedup_class {
                    repr_to_dedup_class[dedup_idx] = class_idx;
                }
            }
            
            // Find two reprs in same dedup class but with original tokens in different ref classes
            'outer: for dedup_class in &dedup_vocab_classes {
                if dedup_class.len() < 2 { continue; }
                // Map each repr to the set of ref classes it represents (through original tokens)
                let mut repr_ref_map: Vec<(usize, BTreeSet<usize>)> = Vec::new();
                for &repr_idx in dedup_class {
                    let mut ref_classes = BTreeSet::new();
                    for (orig_idx, &r) in dedup.original_to_repr.iter().enumerate() {
                        if r == repr_idx {
                            if let Some(&rc) = ref_token_to_ref_class.get(&orig_idx) {
                                ref_classes.insert(rc);
                            }
                        }
                    }
                    repr_ref_map.push((repr_idx, ref_classes));
                }
                // Check if the dedup class spans multiple ref classes
                let all_ref_classes: BTreeSet<usize> = repr_ref_map.iter().flat_map(|(_, rcs)| rcs.iter().copied()).collect();
                if all_ref_classes.len() > 1 {
                    // Found it! Pick one repr from each ref class and run detailed comparison
                    let first_ref_class = *all_ref_classes.iter().next().unwrap();
                    let second_ref_class = *all_ref_classes.iter().nth(1).unwrap();
                    let repr_a = repr_ref_map.iter().find(|(_, rcs)| rcs.contains(&first_ref_class)).unwrap().0;
                    let repr_b = repr_ref_map.iter().find(|(_, rcs)| rcs.contains(&second_ref_class)).unwrap().0;
                    let bytes_a = dedup.representative_token_bytes[repr_a];
                    let bytes_b = dedup.representative_token_bytes[repr_b];
                    eprintln!("[verify_dedup] Running debug_compare for repr {} vs {} in dedup_class",
                        repr_a, repr_b);
                    vocab_equivalence_analysis::debug_compare_token_signatures(
                        tokenizer,
                        bytes_a,
                        bytes_b,
                        &vocab_states,
                        disallowed_follows,
                        active_groups,
                        shared_vocab_dfa_cache,
                    );
                    break 'outer;
                }
            }
        }

        verify_vocab_partition_reference_with_tokens(&vocab_classes, &reference.vocab_classes, tokens, &dedup.original_to_repr);
    }

    if profile_compile {
        eprintln!(
            "[glrmask/profile][equiv] dedup_ms={:.3} tokens={}->{} max_length_ms={:.3} pre_states={} pre_reduced_states={} token_state_ms={:.3} reduced_states={} vocab_ms={:.3} state_classes={} vocab_classes={} total_ms={:.3}",
            dedup_ms,
            tokens.len(),
            dedup.representative_token_bytes.len(),
            max_length_ms,
            initial_states.len(),
            pre_reduced_states.len(),
            token_state_ms,
            reduced_states.len(),
            vocab_ms,
            state_classes.len(),
            vocab_classes.len(),
            elapsed_ms(combined_started_at),
        );
    }

    if debug_profile {
        eprintln!(
            "[glrmask/debug][equiv] raw_vocab={} dedup_vocab={} vocab_classes={} raw_states={} pre_reduced_states={} state_classes={} total_ms={:.3}",
            tokens.len(),
            dedup.representative_token_bytes.len(),
            vocab_classes.len(),
            initial_states.len(),
            pre_reduced_states.len(),
            state_classes.len(),
            elapsed_ms(combined_started_at),
        );
    }

    CombinedEquivalenceResult {
        vocab_classes,
        state_classes,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::automata::lexer::ast::{bytes, star};
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use super::super::compat::TokenizerView;
    use super::super::reference::find_equivalence_classes;
    use super::super::state::fast as fast_state_equivalence;
    use crate::ds::bitset::BitSet;

    use super::verify_state_partition_reference;

    #[test]
    fn unrestricted_state_partition_refines_disallowed_follow_reference() {
        let exprs = [bytes(b"a"), star(bytes(b"b")), bytes(b"c")];
        let tokenizer = build_tokenizer_from_exprs(&exprs);
        let tokenizer_view = TokenizerView::new(&tokenizer);

        let tokens: Vec<Vec<u8>> = vec![
            b"c".to_vec(),
            b"ca".to_vec(),
            b"cba".to_vec(),
            b"bb".to_vec(),
        ];
        let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

        let mut disallowed = BTreeMap::new();
        let mut bits = BitSet::new(3);
        bits.set(1);
        disallowed.insert(2u32, bits);

        let fast_mapping =
            fast_state_equivalence::find_state_equivalence_classes(&tokenizer_view, &tokens, &states);
        let fast_classes =
            fast_state_equivalence::mapping_to_equivalence_classes(&states, &fast_mapping);
        let reference = find_equivalence_classes(&tokenizer_view, &tokens, &states, &disallowed, None);

        verify_state_partition_reference(&fast_classes, &reference.state_classes);
    }
}
