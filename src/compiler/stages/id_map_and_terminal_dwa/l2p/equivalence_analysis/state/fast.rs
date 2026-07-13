//! State equivalence analysis.
//!
//! Performs full token-based refinement over the supplied tokenizer states.
//! Any coarse max-length reduction happens in combined equivalence analysis.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use rayon::prelude::*;

use super::super::compat::TokenizerView;
use super::super::vocab::fast::SharedVocabDfaBase;
use crate::ds::bitset::BitSet;

/// The result of state equivalence analysis: sets of state IDs that behave identically.
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

#[derive(Clone, Copy)]
struct WalkFrame {
    state: u32,
    dead_at_depth: Option<usize>,
    changes_len: usize,
}

struct StateBatchScratch {
    walk_frames: Vec<WalkFrame>,
    positions: Vec<i32>,
    active_bits: Vec<u64>,
    changes: Vec<(usize, i32)>,
}

impl StateBatchScratch {
    fn new(num_groups: usize) -> Self {
        Self {
            walk_frames: Vec::new(),
            positions: vec![-1; num_groups],
            active_bits: vec![0u64; bit_words(num_groups)],
            changes: Vec::new(),
        }
    }
}

#[derive(Default)]
struct StateBatchScratchPool {
    available: Mutex<Vec<StateBatchScratch>>,
}

impl StateBatchScratchPool {
    fn checkout(&self, num_groups: usize) -> StateBatchScratchLease<'_> {
        let scratch = self
            .available
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| StateBatchScratch::new(num_groups));
        debug_assert_eq!(scratch.positions.len(), num_groups);
        StateBatchScratchLease {
            pool: self,
            scratch: Some(scratch),
        }
    }
}

struct StateBatchScratchLease<'a> {
    pool: &'a StateBatchScratchPool,
    scratch: Option<StateBatchScratch>,
}

impl StateBatchScratchLease<'_> {
    fn scratch_mut(&mut self) -> &mut StateBatchScratch {
        self.scratch.as_mut().expect("state scratch lease must be populated")
    }
}

impl Drop for StateBatchScratchLease<'_> {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            self.pool.available.lock().unwrap().push(scratch);
        }
    }
}

#[inline(always)]
fn bit_words(num_bits: usize) -> usize {
    num_bits.div_ceil(64)
}

#[inline(always)]
fn bitset_set(bits: &mut [u64], idx: usize) {
    bits[idx >> 6] |= 1u64 << (idx & 63);
}

#[inline(always)]
fn bitset_clear(bits: &mut [u64], idx: usize) {
    bits[idx >> 6] &= !(1u64 << (idx & 63));
}

#[inline(always)]
fn clear_active_positions(positions: &mut [i32], active_bits: &mut [u64]) {
    for (word_idx, word) in active_bits.iter_mut().enumerate() {
        let mut bits = *word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            positions[word_idx * 64 + bit] = -1;
            bits &= bits - 1;
        }
        *word = 0;
    }
}

#[inline(always)]
fn mix_u128(mut x: u128) -> u128 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[inline(always)]
fn mix_tagged(hash: u128, tag: u128, value: u128) -> u128 {
    mix_u128(hash ^ tag.wrapping_add(value.rotate_left(17)))
}

fn hash_future_groups(future_groups: &[usize]) -> u128 {
    let mut hash = mix_u128(0xF0C7_F0C7_F0C7_F0C7 ^ future_groups.len() as u128);
    for &gid in future_groups {
        hash = mix_tagged(hash, 0x9E37_79B9_7F4A_7C15, gid as u128);
    }
    hash
}

fn hash_future_groups_filtered(future_groups: &[usize], disallowed: &BitSet) -> u128 {
    let allowed_count = future_groups
        .iter()
        .filter(|&&gid| !disallowed.contains(gid))
        .count();
    let mut hash = mix_u128(0xF0C7_F0C7_F0C7_F0C7 ^ allowed_count as u128);
    for &gid in future_groups {
        if !disallowed.contains(gid) {
            hash = mix_tagged(hash, 0x9E37_79B9_7F4A_7C15, gid as u128);
        }
    }
    hash
}

#[derive(Clone, Copy)]
enum FollowRows<'a> {
    Dense(Option<&'a [BitSet]>),
    Borrowed(&'a [Option<&'a BitSet>]),
    Sparse(&'a BTreeMap<u32, BitSet>),
}

impl FollowRows<'_> {
    fn num_groups_hint(self) -> usize {
        match self {
            Self::Dense(rows) => rows.map_or(0, <[BitSet]>::len),
            Self::Borrowed(rows) => rows.len(),
            Self::Sparse(rows) => rows.iter().fold(0usize, |max_group, (&source, row)| {
                let mut max_group = max_group.max(source as usize + 1);
                for target in row.iter() {
                    max_group = max_group.max(target + 1);
                }
                max_group
            }),
        }
    }
}

#[derive(Clone)]
struct FollowContextTable<'a> {
    gid_to_context: Vec<usize>,
    disallowed_by_context: Vec<Option<&'a BitSet>>,
}

impl<'a> FollowContextTable<'a> {
    fn new(num_groups: usize, rows: FollowRows<'a>) -> Self {
        let row_for_gid = |gid: usize| match rows {
            FollowRows::Dense(disallowed_follows) => disallowed_follows
                .and_then(|rows| rows.get(gid))
                .filter(|bits| !bits.is_zero()),
            FollowRows::Borrowed(disallowed_follows) => disallowed_follows
                .get(gid)
                .and_then(|row| *row)
                .filter(|bits| !bits.is_zero()),
            FollowRows::Sparse(disallowed_follows) => disallowed_follows
                .get(&(gid as u32))
                .filter(|bits| !bits.is_zero()),
        };

        let mut gid_to_context = vec![0; num_groups];
        let mut disallowed_by_context = vec![None];
        let mut seen: std::collections::HashMap<&BitSet, usize> = std::collections::HashMap::new();

        for gid in 0..num_groups {
            let ctx = match row_for_gid(gid) {
                None => 0,
                Some(bits) => match seen.get(&bits) {
                    Some(&ctx) => ctx,
                    None => {
                        let ctx = disallowed_by_context.len();
                        seen.insert(bits, ctx);
                        disallowed_by_context.push(Some(bits));
                        ctx
                    }
                },
            };
            gid_to_context[gid] = ctx;
        }

        Self {
            gid_to_context,
            disallowed_by_context,
        }
    }

    #[inline(always)]
    fn num_contexts(&self) -> usize {
        self.disallowed_by_context.len()
    }

    #[inline(always)]
    fn context_for_gid(&self, gid: usize) -> usize {
        self.gid_to_context.get(gid).copied().unwrap_or(0)
    }

    #[inline(always)]
    fn allows_follow(&self, context: usize, gid: usize) -> bool {
        !self.disallowed_by_context[context].is_some_and(|row| row.contains(gid))
    }
}

#[derive(Clone, Default)]
struct SuffixNode {
    end_state: Option<usize>,
    edges: Vec<(usize, usize)>,
}

struct TokenSuffixHashes {
    len: usize,
    num_contexts: usize,
    hashes: Vec<u128>,
}

impl TokenSuffixHashes {
    #[inline(always)]
    fn get(&self, context: usize, pos: usize) -> u128 {
        self.hashes[context * self.len + pos]
    }
}

fn build_future_group_hashes_by_context(
    dfa_future_groups: &[&[usize]],
    follow_contexts: &FollowContextTable<'_>,
) -> Vec<Vec<u128>> {
    (0..follow_contexts.num_contexts())
        .map(|context| {
            let disallowed = follow_contexts.disallowed_by_context[context];
            dfa_future_groups
                .iter()
                .map(|future_groups| {
                    disallowed.map_or_else(
                        || hash_future_groups(future_groups),
                        |disallowed| hash_future_groups_filtered(future_groups, disallowed),
                    )
                })
                .collect()
        })
        .collect()
}

fn hash_suffix_node(
    context: usize,
    pos: usize,
    nodes: &[SuffixNode],
    token_len: usize,
    follow_contexts: &FollowContextTable<'_>,
    future_group_hashes_by_context: &[Vec<u128>],
    memo: &mut [u128],
    ready: &mut [bool],
) -> u128 {
    const DEAD_NODE_TAG: u128 = 0xDEAD_DEAD_DEAD_DEAD;
    const ACCEPT_SINK_HASH: u128 = 0xA11C_EA5E_A11C_EA5E;
    const EDGE_COUNT_TAG: u128 = 0xEDEC_EDEC_EDEC_EDEC;
    const EDGE_GID_TAG: u128 = 0xE001_E001_E001_E001;
    const EDGE_POS_TAG: u128 = 0xE002_E002_E002_E002;
    const EDGE_CHILD_TAG: u128 = 0xE003_E003_E003_E003;

    let idx = context * token_len + pos;
    if ready[idx] {
        return memo[idx];
    }

    let node = &nodes[pos];
    let mut edge_count = 0usize;
    let mut hash = match node.end_state {
        Some(state) => mix_tagged(
            0x51A7_E000_0000_0001,
            0xF070_F070_F070_F070,
            future_group_hashes_by_context[context][state],
        ),
        None => mix_u128(DEAD_NODE_TAG),
    };

    for &(gid, target_pos) in &node.edges {
        if !follow_contexts.allows_follow(context, gid) {
            continue;
        }
        edge_count += 1;
        let child_hash = if target_pos >= token_len {
            ACCEPT_SINK_HASH
        } else {
            let child_context = follow_contexts.context_for_gid(gid);
            hash_suffix_node(
                child_context,
                target_pos,
                nodes,
                token_len,
                follow_contexts,
                future_group_hashes_by_context,
                memo,
                ready,
            )
        };
        hash = mix_tagged(hash, EDGE_GID_TAG, gid as u128);
        hash = mix_tagged(hash, EDGE_POS_TAG, target_pos as u128);
        hash = mix_tagged(hash, EDGE_CHILD_TAG, child_hash);
    }

    let result = mix_tagged(hash, EDGE_COUNT_TAG, edge_count as u128);
    ready[idx] = true;
    memo[idx] = result;
    result
}

fn build_token_suffix_hashes(
    nodes: Vec<SuffixNode>,
    follow_contexts: &FollowContextTable<'_>,
    future_group_hashes_by_context: &[Vec<u128>],
) -> TokenSuffixHashes {
    let len = nodes.len();
    let num_contexts = follow_contexts.num_contexts();
    let mut hashes = vec![0u128; len * num_contexts];
    let mut ready = vec![false; len * num_contexts];

    for context in 0..num_contexts {
        for pos in 0..len {
            let _ = hash_suffix_node(
                context,
                pos,
                &nodes,
                len,
                follow_contexts,
                future_group_hashes_by_context,
                &mut hashes,
                &mut ready,
            );
        }
    }

    TokenSuffixHashes {
        len,
        num_contexts,
        hashes,
    }
}

fn hash_trellis_node_from_positions(
    end_state: Option<usize>,
    positions: &[i32],
    active_bits: &[u64],
    token_len: usize,
    future_group_hashes: &[u128],
    follow_contexts: &FollowContextTable<'_>,
    suffix_hashes: Option<&TokenSuffixHashes>,
) -> u128 {
    const DEAD_NODE_TAG: u128 = 0xDEAD_DEAD_DEAD_DEAD;
    const ACCEPT_SINK_HASH: u128 = 0xA11C_EA5E_A11C_EA5E;
    const EDGE_COUNT_TAG: u128 = 0xEDEC_EDEC_EDEC_EDEC;
    const EDGE_GID_TAG: u128 = 0xE001_E001_E001_E001;
    const EDGE_POS_TAG: u128 = 0xE002_E002_E002_E002;
    const EDGE_CHILD_TAG: u128 = 0xE003_E003_E003_E003;

    let mut edge_count = 0usize;
    let mut hash = match end_state {
        Some(state) => mix_tagged(
            0x51A7_E000_0000_0001,
            0xF070_F070_F070_F070,
            future_group_hashes[state],
        ),
        None => mix_u128(DEAD_NODE_TAG),
    };

    for (word_idx, &word) in active_bits.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let gid = word_idx * 64 + bit;
            bits &= bits - 1;

            let target_pos = positions[gid] as usize;
            edge_count += 1;
            let child_hash = if target_pos >= token_len {
                ACCEPT_SINK_HASH
            } else {
                let suffix_hashes = suffix_hashes.expect("child suffix hashes required for live edge");
                let child_context = follow_contexts.context_for_gid(gid);
                suffix_hashes.get(child_context, target_pos)
            };
            hash = mix_tagged(hash, EDGE_GID_TAG, gid as u128);
            hash = mix_tagged(hash, EDGE_POS_TAG, target_pos as u128);
            hash = mix_tagged(hash, EDGE_CHILD_TAG, child_hash);
        }
    }

    mix_tagged(hash, EDGE_COUNT_TAG, edge_count as u128)
}

fn build_contiguous_batches(total_tokens: usize, target_batch_size: usize) -> Vec<Vec<usize>> {
    if total_tokens == 0 {
        return Vec::new();
    }

    let batch_size = target_batch_size.max(1);
    (0..total_tokens)
        .step_by(batch_size)
        .map(|start| (start..(start + batch_size).min(total_tokens)).collect())
        .collect()
}

fn build_start_state_suffix_nodes(
    token: &[u8],
    tokenizer_start: usize,
    dfa_transitions: &[u32],
    byte_to_class: &[u8; 256],
    num_bc: usize,
    dfa_finalizers: &[&[usize]],
    state_has_future: &[bool],
    skip_groups: &[bool],
    positions: &mut [i32],
    active_bits: &mut [u64],
) -> Vec<SuffixNode> {
    let len = token.len();
    let num_groups = positions.len();
    let mut suffix_nodes = vec![SuffixNode::default(); len];
    let skip_groups_enabled = !skip_groups.is_empty();

    clear_active_positions(positions, active_bits);

    for pos in (0..len).rev() {
        let mut current = tokenizer_start;
        let mut current_ct_base = current * num_bc;
        let mut done = !state_has_future[current];

        for (offset, &byte) in token[pos..].iter().enumerate() {
            if done {
                break;
            }
            let next = dfa_transitions[current_ct_base + byte_to_class[byte as usize] as usize];
            if next == u32::MAX {
                done = true;
                break;
            }
            current = next as usize;
            current_ct_base = current * num_bc;
            let absolute_pos = (pos + offset + 1) as i32;
            for &gid in dfa_finalizers[current] {
                if gid >= num_groups || (skip_groups_enabled && skip_groups[gid]) {
                    continue;
                }
                if positions[gid] < 0 {
                    bitset_set(active_bits, gid);
                }
                positions[gid] = absolute_pos;
            }
            if !state_has_future[current] {
                done = true;
            }
        }

        let mut edges = Vec::new();
        for (word_idx, &word) in active_bits.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let gid = word_idx * 64 + bit;
                bits &= bits - 1;
                edges.push((gid, positions[gid] as usize));
            }
        }

        suffix_nodes[pos] = SuffixNode {
            end_state: (!done).then_some(current),
            edges,
        };

        clear_active_positions(positions, active_bits);
    }
    suffix_nodes
}

pub fn find_state_equivalence_classes_with_disallowed<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[BitSet],
) -> Vec<usize> {
    find_state_equivalence_classes_with_disallowed_and_shared_base(
        tokenizer,
        tokens,
        states,
        disallowed_follows,
        None,
    )
}

pub fn find_state_equivalence_classes_with_disallowed_and_shared_base<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[BitSet],
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Dense(Some(disallowed_follows)),
        None,
        None,
        None,
        false,
        shared_base,
        false,
        false,
    )
}

/// Exact common-prefix sibling of the ordinary shared-base entry point. The
/// caller has already consumed one byte before each supplied start state, so
/// finalizers on that state are matches at byte position zero for the remaining
/// token suffix.
pub(crate) fn find_state_equivalence_classes_with_disallowed_and_shared_base_with_initial_finalizers<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[BitSet],
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Dense(Some(disallowed_follows)),
        None,
        None,
        None,
        false,
        shared_base,
        false,
        true,
    )
}

/// Exact sibling of the dense follow-table entry point. It borrows only the
/// non-empty grammar rows and derives the same follow-row contexts by bitset
/// equality, avoiding a dense terminal-square normalization for tiny vocab
/// partitions with large active alphabets.
pub(crate) fn find_state_equivalence_classes_with_sparse_disallowed_and_shared_base<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Sparse(disallowed_follows),
        None,
        None,
        None,
        false,
        shared_base,
        false,
        false,
    )
}

/// Dense-indexed but borrowed follow rows. This is equivalent to normalized
/// dense rows while avoiding a clone of every grammar-terminal bitset.
pub(crate) fn find_state_equivalence_classes_with_borrowed_disallowed_and_shared_base<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[Option<&BitSet>],
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Borrowed(disallowed_follows),
        None,
        None,
        None,
        false,
        shared_base,
        false,
        false,
    )
}

/// Exact sparse-row entry point for a deliberately small set of source
/// states. It bypasses whole-DFA byte-class materialization and walks the raw
/// transition table directly.
pub(crate) fn find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Sparse(disallowed_follows),
        None,
        None,
        None,
        false,
        None,
        true,
        false,
    )
}

/// Exact raw-table entry point that treats finalizers on each supplied start
/// state as matches at byte position zero. This models a factored common
/// prefix that has already been consumed before the remaining token suffix.
pub(crate) fn find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions_with_initial_finalizers<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Sparse(disallowed_follows),
        None,
        None,
        None,
        false,
        None,
        true,
        true,
    )
}

pub fn find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[BitSet],
    max_batches: Option<usize>,
    batch_size: Option<usize>,
    early_stop_override: Option<bool>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
        tokenizer,
        tokens,
        states,
        disallowed_follows,
        max_batches,
        batch_size,
        early_stop_override,
        None,
    )
}

pub fn find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[BitSet],
    max_batches: Option<usize>,
    batch_size: Option<usize>,
    early_stop_override: Option<bool>,
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Dense(Some(disallowed_follows)),
        max_batches,
        batch_size,
        early_stop_override,
        true,
        shared_base,
        false,
        false,
    )
}

/// Exact common-prefix sibling of the shared-base entry point. The caller has
/// already consumed one byte before each supplied start state, so finalizers on
/// that state are matches at byte position zero for the remaining token suffix.
pub(crate) fn find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base_with_initial_finalizers<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[BitSet],
    max_batches: Option<usize>,
    batch_size: Option<usize>,
    early_stop_override: Option<bool>,
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Dense(Some(disallowed_follows)),
        max_batches,
        batch_size,
        early_stop_override,
        true,
        shared_base,
        false,
        true,
    )
}

pub(crate) fn find_state_equivalence_classes_ex_with_rep_confirmation_and_sparse_disallowed_and_shared_base<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    max_batches: Option<usize>,
    batch_size: Option<usize>,
    early_stop_override: Option<bool>,
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Sparse(disallowed_follows),
        max_batches,
        batch_size,
        early_stop_override,
        true,
        shared_base,
        false,
        false,
    )
}

pub(crate) fn find_state_equivalence_classes_ex_with_rep_confirmation_and_borrowed_disallowed_and_shared_base<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    disallowed_follows: &[Option<&BitSet>],
    max_batches: Option<usize>,
    batch_size: Option<usize>,
    early_stop_override: Option<bool>,
    shared_base: Option<&SharedVocabDfaBase>,
) -> Vec<usize> {
    find_state_equivalence_classes_ex_inner(
        tokenizer,
        tokens,
        states,
        &[],
        FollowRows::Borrowed(disallowed_follows),
        max_batches,
        batch_size,
        early_stop_override,
        true,
        shared_base,
        false,
        false,
    )
}

fn find_state_equivalence_classes_ex_inner<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    skip_groups: &[bool],
    follow_rows: FollowRows<'_>,
    max_batches: Option<usize>,
    batch_size: Option<usize>,
    early_stop_override: Option<bool>,
    rep_only_confirmation: bool,
    shared_base: Option<&SharedVocabDfaBase>,
    force_raw_transitions: bool,
    seed_initial_finalizers: bool,
) -> Vec<usize> {
    if states.is_empty() {
        return Vec::new();
    }

    find_state_equivalence_classes_token_based(
        tokenizer,
        tokens,
        states,
        skip_groups,
        follow_rows,
        max_batches,
        batch_size,
        early_stop_override,
        rep_only_confirmation,
        shared_base,
        force_raw_transitions,
        seed_initial_finalizers,
    )
}

fn find_state_equivalence_classes_token_based<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    states: &[usize],
    skip_groups: &[bool],
    follow_rows: FollowRows<'_>,
    max_batches: Option<usize>,
    custom_batch_size: Option<usize>,
    early_stop_override: Option<bool>,
    rep_only_confirmation: bool,
    shared_base: Option<&SharedVocabDfaBase>,
    force_raw_transitions: bool,
    seed_initial_finalizers: bool,
) -> Vec<usize> {
    use std::collections::{hash_map::Entry, HashMap};

    let dfa = tokenizer.dfa();

    const NONE_STATE: u32 = u32::MAX;

    // Build byte-class compressed transition table for cache efficiency when the DFA
    // is large enough. With 54K+ states, the raw table (state*256*4 bytes) can be 56MB,
    // causing severe cache thrashing. Byte-class compression typically reduces 256
    // columns to ~60, fitting the table in L2/L3 cache. For smaller DFAs (<16K states,
    // <16MB table), the raw table already fits in cache and compression overhead
    // outweighs the benefit.
    const COMPACT_THRESHOLD_STATES: usize = 16_000;
    let num_dfa_states = dfa.states.len();
    let identity_byte_class: [u8; 256] = std::array::from_fn(|i| i as u8);
    let compatible_shared_base = shared_base.filter(|base| base.is_compatible_with_dfa(dfa));
    let use_compact = !force_raw_transitions
        && (num_dfa_states >= COMPACT_THRESHOLD_STATES || compatible_shared_base.is_some());
    let computed_byte_class = (use_compact && compatible_shared_base.is_none())
        .then(|| super::super::compat::compute_byte_classes(dfa));
    let byte_to_class: &[u8; 256] = compatible_shared_base
        .map(SharedVocabDfaBase::byte_to_class_ref)
        .or_else(|| computed_byte_class.as_ref())
        .unwrap_or(&identity_byte_class);
    let num_bc = compatible_shared_base
        .map(|base| base.num_classes)
        .or_else(|| {
            computed_byte_class
                .as_ref()
                .map(|classes| classes.iter().copied().max().map_or(0usize, |max| max as usize + 1))
        })
        .unwrap_or(256);
    let compact_transitions: std::borrow::Cow<'_, [u32]> = if let Some(base) = compatible_shared_base {
        std::borrow::Cow::Borrowed(base.transitions_by_state_class())
    } else if let Some(classes) = computed_byte_class.as_ref() {
        let mut transitions = vec![NONE_STATE; num_dfa_states * num_bc];
        let raw = &dfa.transitions;
        for state in 0..num_dfa_states {
            let raw_base = state * 256;
            let compact_base = state * num_bc;
            for byte in 0..256u16 {
                let class = classes[byte as usize] as usize;
                transitions[compact_base + class] = raw[raw_base + byte as usize];
            }
        }
        std::borrow::Cow::Owned(transitions)
    } else {
        std::borrow::Cow::Borrowed(&dfa.transitions)
    };

    let dfa_finalizers: Vec<&[usize]> = dfa
        .states
        .iter()
        .map(|state| state.finalizers.as_slice())
        .collect();
    let dfa_future_groups: Vec<&[usize]> = dfa
        .states
        .iter()
        .map(|state| state.possible_future_group_ids.as_slice())
        .collect();
    let state_has_future: Vec<bool> = dfa_future_groups
        .iter()
        .map(|future_groups| !future_groups.is_empty())
        .collect();
    let mut max_gid = dfa_finalizers
        .iter()
        .chain(dfa_future_groups.iter())
        .flat_map(|groups| groups.iter().copied())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    max_gid = max_gid.max(follow_rows.num_groups_hint());
    let num_groups = max_gid;
    let follow_contexts = FollowContextTable::new(num_groups, follow_rows);
    let future_group_hashes_by_context =
        build_future_group_hashes_by_context(&dfa_future_groups, &follow_contexts);
    let mut sorted_indices: Vec<usize> = (0..tokens.len()).collect();
    sorted_indices.par_sort_unstable_by(|&a, &b| tokens[a].as_ref().cmp(tokens[b].as_ref()));

    let mut sorted_tokens: Vec<&[u8]> = Vec::with_capacity(tokens.len());
    let mut sorted_weights: Vec<u128> = Vec::with_capacity(tokens.len());
    for &idx in &sorted_indices {
        sorted_tokens.push(tokens[idx].as_ref());
        sorted_weights.push(mix_u128((idx + 1) as u128));
    }

    let total_tokens = sorted_tokens.len();

    let early_stop = early_stop_override.unwrap_or_else(|| {
        std::env::var("STATE_EQUIV_EARLY_STOP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    let batch_size = custom_batch_size.unwrap_or(250);
    let batches = build_contiguous_batches(total_tokens, batch_size);

    let needed_token_flags = if let Some(max) = max_batches {
        let mut flags = vec![false; total_tokens];
        let mut used_batches = 0usize;
        for batch_indices in &batches {
            if used_batches >= max {
                break;
            }
            if batch_indices.is_empty() {
                continue;
            }
            used_batches += 1;
            for &token_idx in batch_indices {
                flags[token_idx] = true;
            }
        }
        flags
    } else {
        vec![true; total_tokens]
    };

    let tokenizer_start = tokenizer.initial_state_id();
    let suffix_hashes_by_token: Vec<Option<TokenSuffixHashes>> = sorted_tokens
        .par_iter()
        .enumerate()
        .map_init(
            || {
                (
                    vec![-1i32; num_groups],
                    vec![0u64; bit_words(num_groups)],
                )
            },
            |(positions, active_bits), (token_idx, token)| {
                if !needed_token_flags[token_idx] {
                    None
                } else {
                    let nodes = build_start_state_suffix_nodes(
                        token,
                        tokenizer_start,
                        compact_transitions.as_ref(),
                        byte_to_class,
                        num_bc,
                        &dfa_finalizers,
                        &state_has_future,
                        skip_groups,
                        positions,
                        active_bits,
                    );
                    Some(build_token_suffix_hashes(
                        nodes,
                        &follow_contexts,
                        &future_group_hashes_by_context,
                    ))
                }
            },
        )
        .collect();

    let common_prefix_len = |a: &[u8], b: &[u8]| -> usize {
        let len = a.len().min(b.len());
        let mut i = 0usize;
        while i < len && a[i] == b[i] {
            i += 1;
        }
        i
    };

    let dead_positions = vec![-1i32; num_groups];
    let dead_active_bits = vec![0u64; bit_words(num_groups)];
    let fully_dead_token_hash = hash_trellis_node_from_positions(
        None,
        &dead_positions,
        &dead_active_bits,
        0,
        &future_group_hashes_by_context[0],
        &follow_contexts,
        None,
    );
    // Most state/token observations never see a terminal edge. In that case the
    // observation depends only on the end state (or dead state), so precompute
    // the hash once per DFA state instead of rebuilding the same empty-edge node
    // for every token.
    let no_edge_state_hashes: Vec<u128> = (0..num_dfa_states)
        .map(|state| {
            hash_trellis_node_from_positions(
                Some(state),
                &dead_positions,
                &dead_active_bits,
                0,
                &future_group_hashes_by_context[0],
                &follow_contexts,
                None,
            )
        })
        .collect();

    let mut group_ids: Vec<usize> = vec![0usize; states.len()];
    let mut group_sizes: Vec<usize> = vec![states.len()];
    let mut active_indices: Vec<usize> = (0..states.len()).collect();
    let mut touched_group_flags: Vec<bool> = vec![false; group_sizes.len()];
    let mut reused_group_flags: Vec<bool> = vec![false; group_sizes.len()];
    let mut prev_groups = 1usize;
    let mut stable_batches = 0usize;
    let mut tokens_tested = 0usize;
    let mut batches_processed = 0usize;
    let batch_scratch_pool = StateBatchScratchPool::default();
    for batch_indices in &batches {
        if active_indices.is_empty() {
            break;
        }
        if let Some(max) = max_batches {
            if batches_processed >= max {
                break;
            }
        }

        let batch_len = batch_indices.len();
        if batch_len == 0 {
            continue;
        }
        tokens_tested += batch_len;

        let mut batch_tokens: Vec<&[u8]> = Vec::with_capacity(batch_len);
        let mut batch_lcp_with_prev = Vec::with_capacity(batch_len);
        let mut batch_weight_prefix = vec![0u128; batch_len + 1];
        let mut prev_token: Option<&[u8]> = None;

        for (local_idx, &token_idx) in batch_indices.iter().enumerate() {
            let token = sorted_tokens[token_idx];
            let lcp = prev_token.map_or(0, |prev| common_prefix_len(prev, token));
            batch_tokens.push(token);
            batch_lcp_with_prev.push(lcp);
            batch_weight_prefix[local_idx + 1] =
                batch_weight_prefix[local_idx].wrapping_add(sorted_weights[token_idx]);
            prev_token = Some(token);
        }

        let batch_empty_end = batch_tokens
            .iter()
            .take_while(|token| token.is_empty())
            .count();
        let batch_empty_range = (0usize, batch_empty_end);

        let mut batch_first_byte_ranges = [(0usize, 0usize); 256];
        let mut batch_nonempty_first_bytes: Vec<usize> = Vec::new();
        let mut batch_pos = batch_empty_end;
        while batch_pos < batch_len {
            let byte = batch_tokens[batch_pos][0] as usize;
            let start = batch_pos;
            batch_pos += 1;
            while batch_pos < batch_len
                && !batch_tokens[batch_pos].is_empty()
                && batch_tokens[batch_pos][0] as usize == byte
            {
                batch_pos += 1;
            }
            batch_first_byte_ranges[byte] = (start, batch_pos);
            batch_nonempty_first_bytes.push(byte);
        }

        let mut batch_hashes: Vec<(usize, u128)> = active_indices
            .par_iter()
            .map_init(
                || batch_scratch_pool.checkout(num_groups),
                |lease, &state_idx| {
                    let state = states[state_idx] as u32;
                    let mut hash_delta: u128 = 0;
                    let state_ct_base = (state as usize) * num_bc;

                    let mut live_ranges: Vec<(usize, usize)> = Vec::new();

                    if batch_empty_range.0 < batch_empty_range.1 {
                        live_ranges.push(batch_empty_range);
                    }

                    for &byte in &batch_nonempty_first_bytes {
                        let (range_start, range_end) = batch_first_byte_ranges[byte];
                        if range_start >= range_end {
                            continue;
                        }

                        if compact_transitions[state_ct_base + byte_to_class[byte] as usize] == NONE_STATE {
                            let weight_sum =
                                batch_weight_prefix[range_end].wrapping_sub(batch_weight_prefix[range_start]);
                            hash_delta = hash_delta
                                .wrapping_add(fully_dead_token_hash.wrapping_mul(weight_sum));
                        } else {
                            live_ranges.push((range_start, range_end));
                        }
                    }
                    let scratch = lease.scratch_mut();
                    let walk_frames = &mut scratch.walk_frames;
                    let positions = &mut scratch.positions;
                    let active_bits = &mut scratch.active_bits;
                    let changes = &mut scratch.changes;

                    for (range_start, range_end) in live_ranges {
                        if range_start >= range_end {
                            continue;
                        }

                        walk_frames.clear();
                        clear_active_positions(positions, active_bits);
                        changes.clear();
                        if seed_initial_finalizers && num_groups > 0 {
                            for &gid in dfa_finalizers[state as usize] {
                                if gid >= num_groups
                                    || (!skip_groups.is_empty() && skip_groups[gid])
                                {
                                    continue;
                                }
                                positions[gid] = 0;
                                bitset_set(active_bits, gid);
                            }
                        }
                        walk_frames.push(WalkFrame {
                            state,
                            dead_at_depth: None,
                            changes_len: 0,
                        });

                        for token_idx in range_start..range_end {
                            let global_token_idx = batch_indices[token_idx];
                            let token = batch_tokens[token_idx];
                            let mut prefix_len = if token_idx == range_start {
                                0
                            } else {
                                batch_lcp_with_prev[token_idx]
                            };
                            let max_prefix = walk_frames.len().saturating_sub(1);
                            if prefix_len > max_prefix {
                                prefix_len = max_prefix;
                            }

                            if walk_frames.len() > prefix_len + 1 {
                                let target_mark = walk_frames[prefix_len].changes_len;
                                while changes.len() > target_mark {
                                    let (gid, prev_pos) = changes.pop().unwrap();
                                    if prev_pos < 0 {
                                        positions[gid] = -1;
                                        bitset_clear(active_bits, gid);
                                    } else {
                                        positions[gid] = prev_pos;
                                        bitset_set(active_bits, gid);
                                    }
                                }

                                walk_frames.truncate(prefix_len + 1);
                            }

                            let mut dead_at_depth = walk_frames[prefix_len].dead_at_depth;

                            if dead_at_depth.is_none() {
                                let mut current = walk_frames.last().unwrap().state;
                                for (offset, &byte) in token[prefix_len..].iter().enumerate() {
                                    if current == NONE_STATE {
                                        dead_at_depth = Some(prefix_len + offset);
                                        break;
                                    }
                                    let next = compact_transitions[current as usize * num_bc + byte_to_class[byte as usize] as usize];
                                    if next == NONE_STATE {
                                        dead_at_depth = Some(prefix_len + offset + 1);
                                        walk_frames.push(WalkFrame {
                                            state: NONE_STATE,
                                            dead_at_depth,
                                            changes_len: changes.len(),
                                        });
                                        break;
                                    }
                                    current = next;
                                    let position = prefix_len + offset + 1;

                                    if num_groups > 0 {
                                        for &gid in dfa_finalizers[current as usize] {
                                            if gid >= num_groups {
                                                continue;
                                            }
                                            if !skip_groups.is_empty() && skip_groups[gid] {
                                                continue;
                                            }
                                            let pos_i32 = position as i32;
                                            let prev = positions[gid];
                                            if prev != pos_i32 {
                                                if prev < 0 {
                                                    bitset_set(active_bits, gid);
                                                }
                                                changes.push((gid, prev));
                                                positions[gid] = pos_i32;
                                            }
                                        }
                                    }

                                    walk_frames.push(WalkFrame {
                                        state: current,
                                        dead_at_depth,
                                        changes_len: changes.len(),
                                    });
                                }
                            }

                            let has_active_edges = active_bits.iter().any(|&word| word != 0);
                            let token_hash = if !has_active_edges {
                                if dead_at_depth.is_some() {
                                    fully_dead_token_hash
                                } else {
                                    let current = walk_frames.last().unwrap().state;
                                    no_edge_state_hashes[current as usize]
                                }
                            } else if dead_at_depth.is_some() {
                                hash_trellis_node_from_positions(
                                    None,
                                    positions,
                                    active_bits,
                                    token.len(),
                                    &future_group_hashes_by_context[0],
                                    &follow_contexts,
                                    suffix_hashes_by_token[global_token_idx].as_ref(),
                                )
                            } else {
                                let current = walk_frames.last().unwrap().state;
                                hash_trellis_node_from_positions(
                                    Some(current as usize),
                                    positions,
                                    active_bits,
                                    token.len(),
                                    &future_group_hashes_by_context[0],
                                    &follow_contexts,
                                    suffix_hashes_by_token[global_token_idx].as_ref(),
                                )
                            };
                            hash_delta = hash_delta.wrapping_add(
                                token_hash.wrapping_mul(sorted_weights[global_token_idx]),
                            );
                        }
                    }

                    (state_idx, hash_delta)
                },
            )
            .collect();

        batches_processed += 1;
        let previous_active_indices = std::mem::take(&mut active_indices);
        let all_active = previous_active_indices.len() == states.len();

        if all_active {
            let mut key_to_group: HashMap<(usize, u128), usize> =
                HashMap::with_capacity(states.len());
            group_sizes.clear();

            for (state_idx, hash) in batch_hashes.drain(..) {
                let key = (group_ids[state_idx], hash);
                let gid = *key_to_group.entry(key).or_insert_with(|| {
                    let id = group_sizes.len();
                    group_sizes.push(0);
                    id
                });
                group_ids[state_idx] = gid;
                group_sizes[gid] += 1;
            }

            touched_group_flags.clear();
            touched_group_flags.resize(group_sizes.len(), false);
            reused_group_flags.clear();
            reused_group_flags.resize(group_sizes.len(), false);
        } else {
            let mut key_to_group: HashMap<(usize, u128), usize> =
                HashMap::with_capacity(previous_active_indices.len());
            let mut touched_groups: Vec<usize> = Vec::new();

            for &state_idx in &previous_active_indices {
                let gid = group_ids[state_idx];
                if !touched_group_flags[gid] {
                    touched_group_flags[gid] = true;
                    reused_group_flags[gid] = false;
                    touched_groups.push(gid);
                    group_sizes[gid] = 0;
                }
            }

            for (state_idx, hash) in batch_hashes.drain(..) {
                let old_gid = group_ids[state_idx];
                let key = (old_gid, hash);
                let gid = match key_to_group.entry(key) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let gid = if !reused_group_flags[old_gid] {
                            reused_group_flags[old_gid] = true;
                            old_gid
                        } else {
                            let new_gid = group_sizes.len();
                            group_sizes.push(0);
                            touched_group_flags.push(false);
                            reused_group_flags.push(false);
                            new_gid
                        };
                        *entry.insert(gid)
                    }
                };
                group_ids[state_idx] = gid;
                group_sizes[gid] += 1;
            }

            for gid in touched_groups {
                touched_group_flags[gid] = false;
                reused_group_flags[gid] = false;
            }
        }

        let num_groups = group_sizes.len();
        active_indices.reserve(previous_active_indices.len());
        for state_idx in previous_active_indices {
            if group_sizes[group_ids[state_idx]] > 1 {
                active_indices.push(state_idx);
            }
        }

        // All tokens must be processed before early-stop convergence is trusted.
        let min_tokens_met = tokens_tested >= total_tokens;
        if early_stop && min_tokens_met {
            if num_groups == prev_groups {
                stable_batches += 1;
            } else {
                stable_batches = 0;
            }

            // Rep-only confirmation: after the first stable batch, collapse
            // active_indices to one representative per group. The next batch
            // walks only these ~N reps instead of all ~K active states. Since
            // each rep is in a distinct group, the refinement trivially
            // confirms stability, giving stable_batches=2 cheaply.
            //
            // This saves one full batch of K×batch_size walks (e.g. 2462×5000
            // = 12.3M walks) at the cost of a single cheap N×batch_size batch
            // (e.g. 19×5000 = 95K walks).
            if rep_only_confirmation && stable_batches == 1 {
                let mut seen_groups: Vec<bool> = vec![false; group_sizes.len()];
                let mut rep_indices: Vec<usize> = Vec::with_capacity(num_groups);
                for &state_idx in &active_indices {
                    let gid = group_ids[state_idx];
                    if !seen_groups[gid] {
                        seen_groups[gid] = true;
                        rep_indices.push(state_idx);
                    }
                }
                active_indices = rep_indices;
            }

            if stable_batches >= 2 {
                break;
            }
        }

        prev_groups = num_groups;
    }

    let num_groups = group_ids.iter().copied().max().map(|v| v + 1).unwrap_or(0);
    let mut rep_for_group: Vec<usize> = vec![usize::MAX; num_groups];
    for (idx, &gid) in group_ids.iter().enumerate() {
        if rep_for_group[gid] == usize::MAX {
            rep_for_group[gid] = states[idx];
        }
    }

    let mut mapping = vec![0usize; states.len()];
    for (idx, &gid) in group_ids.iter().enumerate() {
        mapping[idx] = rep_for_group[gid];
    }

    mapping
}

/// Convert a state-to-representative mapping to `StateEquivalenceResult` format.
pub fn mapping_to_equivalence_classes(
    states: &[usize],
    mapping: &[usize],
) -> StateEquivalenceResult {
    let mut rep_to_class: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();

    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }

    rep_to_class.into_values().collect()
}


#[cfg(test)]
mod state_batch_scratch_pool_tests {
    use super::*;

    #[test]
    fn recycled_scratch_can_be_scrubbed_before_next_walk() {
        let pool = StateBatchScratchPool::default();
        let positions_ptr;
        {
            let mut lease = pool.checkout(65);
            let scratch = lease.scratch_mut();
            positions_ptr = scratch.positions.as_ptr();
            scratch.positions[0] = 7;
            scratch.positions[64] = 11;
            bitset_set(&mut scratch.active_bits, 0);
            bitset_set(&mut scratch.active_bits, 64);
        }

        {
            let mut lease = pool.checkout(65);
            let scratch = lease.scratch_mut();
            assert_eq!(scratch.positions.as_ptr(), positions_ptr);
            clear_active_positions(&mut scratch.positions, &mut scratch.active_bits);
            assert!(scratch.positions.iter().all(|&position| position == -1));
            assert!(scratch.active_bits.iter().all(|&word| word == 0));
        }
    }
}
