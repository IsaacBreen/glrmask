use crate::runtime::state::ConstraintState;
use crate::runtime::actions::commit::token_bytes_may_commit_from_state_map;
use crate::compiler::glr::labels::{encode_positive_label, DEFAULT_LABEL};
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

type DenseTokenMaskCache = FxHashMap<usize, Box<[u64]>>;
type MaskQueue = BTreeMap<u32, FxHashMap<u32, DenseMaskGSS>>;

// DenseMaskAcc keeps mask traversal in dense bitmaps.

/// Dense bitmap accumulator for the mask BFS. Stores the set of allowed internal
/// tokens as a fixed-size u64 bitmap, enabling O(1)-per-word intersection (AND),
/// union (OR), and equality checks instead of O(k) RangeSetBlaze operations.
///
/// Uses `Arc<[u64]>` for cheap cloning (refcount bump instead of heap alloc),
/// which is critical since `apply_and_prune` clones accumulators for memoization.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DenseMaskAcc(BTreeMap<u32, Arc<[u64]>>);

impl DenseMaskAcc {
    fn from_internal_tokens(
        start: u32,
        end: u32,
        tokens: &RangeSetBlaze<u32>,
        dense_words: usize,
    ) -> Self {
        if tokens.is_empty() || start > end {
            return Self(BTreeMap::new());
        }
        let mut dense = vec![0u64; dense_words];
        for range in tokens.ranges() {
            let lo = *range.start() as usize;
            let hi = *range.end() as usize;
            let word_lo = lo / 64;
            let word_hi = hi / 64;
            if word_lo == word_hi {
                if let Some(w) = dense.get_mut(word_lo) {
                    let mask = if hi % 64 == 63 { !0u64 } else { (1u64 << (hi % 64 + 1)) - 1 };
                    let mask = mask & !((1u64 << (lo % 64)) - 1);
                    *w |= mask;
                }
            } else {
                if let Some(w) = dense.get_mut(word_lo) {
                    *w |= !((1u64 << (lo % 64)) - 1);
                }
                for wi in (word_lo + 1)..word_hi {
                    if let Some(w) = dense.get_mut(wi) {
                        *w = !0u64;
                    }
                }
                if let Some(w) = dense.get_mut(word_hi) {
                    let mask = if hi % 64 == 63 { !0u64 } else { (1u64 << (hi % 64 + 1)) - 1 };
                    *w |= mask;
                }
            }
        }
        let dense: Arc<[u64]> = dense.into();
        let mut map = BTreeMap::new();
        for tsid in start..=end {
            map.insert(tsid, Arc::clone(&dense));
        }
        Self(map)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn dense_len(&self) -> usize {
        self.0.values().next().map(|dense| dense.len()).unwrap_or(0)
    }

    fn dense_to_tokens(dense: &[u64]) -> RangeSetBlaze<u32> {
        let mut tokens = RangeSetBlaze::new();
        for (wi, &w) in dense.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let b = bits.trailing_zeros() as u32;
                tokens.insert((wi as u32) * 64 + b);
                bits &= bits - 1;
            }
        }
        tokens
    }

    /// Intersect this accumulator with a DWA weight using precomputed dense masks.
    /// Returns None if the result is empty.
    fn intersect_with_weight(
        &self,
        weight: &Weight,
        precomputed: &rustc_hash::FxHashMap<usize, Box<[u64]>>,
    ) -> Option<Self> {
        if self.is_empty() {
            return None;
        }

        let mut result = BTreeMap::new();

        for (&tsid, dense) in &self.0 {
            let Some(token_set) = weight.0.get(tsid) else {
                continue;
            };
            let key = Arc::as_ptr(token_set) as usize;
            if let Some(other_dense) = precomputed.get(&key) {
                if !dense.iter().zip(other_dense.iter()).any(|(&s, &o)| s & o != 0) {
                    continue;
                }
                let result_dense: Arc<[u64]> = dense
                    .iter()
                    .zip(other_dense.iter())
                    .map(|(&s, &o)| s & o)
                    .collect();
                result.insert(tsid, result_dense);
            } else {
                return self.intersect_with_weight_fallback(weight);
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(Self(result))
        }
    }

    fn intersect_with_weight_fallback(&self, weight: &Weight) -> Option<Self> {
        let mut result = BTreeMap::new();
        let dense_words = self.dense_len();

        for (&tsid, dense) in &self.0 {
            let tokens = Self::dense_to_tokens(dense);
            if tokens.is_empty() {
                continue;
            }
            let result_weight = weight.intersect_single_parts(tsid, tsid, &Arc::new(tokens));
            if result_weight.is_empty() {
                continue;
            }
            if let Some((_, _, result_tokens)) = result_weight.single_compact_entry_parts() {
                let result_dense = Self::from_internal_tokens(tsid, tsid, &result_tokens, dense_words);
                if let Some(dense) = result_dense.0.get(&tsid) {
                    result.insert(tsid, Arc::clone(dense));
                }
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(Self(result))
        }
    }

    /// OR this accumulator's internal tokens into a merged dense bitmap.
    fn or_into_merged(&self, merged: &mut [u64]) {
        for dense in self.0.values() {
            let n = dense.len().min(merged.len());
            for i in 0..n {
                merged[i] |= dense[i];
            }
        }
    }

    /// OR (accumulator ∩ final_weight) into a merged dense bitmap.
    fn or_intersection_into_merged(
        &self,
        final_weight: &Weight,
        precomputed: &rustc_hash::FxHashMap<usize, Box<[u64]>>,
        merged: &mut [u64],
    ) {
        for (&tsid, dense) in &self.0 {
            let Some(token_set) = final_weight.0.get(tsid) else {
                continue;
            };
            let key = Arc::as_ptr(token_set) as usize;
            if let Some(other_dense) = precomputed.get(&key) {
                let n = dense.len().min(other_dense.len()).min(merged.len());
                for i in 0..n {
                    merged[i] |= dense[i] & other_dense[i];
                }
            } else {
                // Fallback: compute intersection via Weight API.
                let tokens = Self::dense_to_tokens(dense);
                let result_weight = final_weight.intersect_single_parts(tsid, tsid, &Arc::new(tokens));
                if let Some((_, _, result_tokens)) = result_weight.single_compact_entry_parts() {
                    for t in result_tokens.iter() {
                        let w = t as usize / 64;
                        let b = t as usize % 64;
                        if w < merged.len() {
                            merged[w] |= 1u64 << b;
                        }
                    }
                }
            }
        }
    }
}

impl Merge for DenseMaskAcc {
    fn merge(&self, other: &Self) -> Self {
        let mut merged = self.0.clone();
        for (tsid, other_dense) in &other.0 {
            merged
                .entry(*tsid)
                .and_modify(|dense| {
                    let len = dense.len().max(other_dense.len());
                    let mut combined = vec![0u64; len];
                    for i in 0..len {
                        combined[i] = dense.get(i).copied().unwrap_or(0)
                            | other_dense.get(i).copied().unwrap_or(0);
                    }
                    *dense = combined.into();
                })
                .or_insert_with(|| other_dense.clone());
        }
        Self(merged)
    }
}

type DenseMaskGSS = LeveledGSS<u32, DenseMaskAcc>;

fn enqueue_gss(
    queue: &mut MaskQueue,
    depth: u32,
    target: u32,
    gss: DenseMaskGSS,
) {
    queue
        .entry(depth)
        .or_default()
        .entry(target)
        .and_modify(|existing| *existing = existing.merge(&gss))
        .or_insert(gss);
}

fn transition_labels(parser_state: u32) -> [i32; 2] {
    [encode_positive_label(parser_state), DEFAULT_LABEL]
}

fn enqueue_weighted_transition(
    queue: &mut MaskQueue,
    popped: &DenseMaskGSS,
    target: u32,
    weight: &Weight,
    precomputed: &DenseTokenMaskCache,
) {
    if weight.is_full() {
        enqueue_gss(queue, popped.max_depth(), target, popped.clone());
        return;
    }
    let pruned = popped.apply_and_prune_no_promote(|allowed| {
        allowed.intersect_with_weight(weight, precomputed)
    });
    if pruned.is_empty() {
        return;
    }
    enqueue_gss(queue, pruned.max_depth(), target, pruned);
}

fn enqueue_parser_state_transitions(
    queue: &mut MaskQueue,
    fast_trans: &FxHashMap<i32, (u32, Weight)>,
    parser_state: u32,
    popped: &DenseMaskGSS,
    precomputed: &DenseTokenMaskCache,
) {
    for label in transition_labels(parser_state) {
        let Some((target, weight)) = fast_trans.get(&label) else {
            continue;
        };
        enqueue_weighted_transition(queue, popped, *target, weight, precomputed);
    }
}

fn update_eos_mask(buf: &mut [u32], eos_token_id: Option<u32>, is_complete: bool) {
    let Some(eos_token_id) = eos_token_id else {
        return;
    };
    let word = eos_token_id as usize / 32;
    let bit = eos_token_id as usize % 32;
    let Some(slot) = buf.get_mut(word) else {
        return;
    };

    *slot &= !(1u32 << bit);
    if is_complete {
        *slot |= 1u32 << bit;
    }
}

impl<'a> ConstraintState<'a> {
    fn gss_accumulators_empty(gss: &crate::compiler::glr::parser::ParserGSS) -> bool {
        let mut all_empty = true;
        gss.for_each_acc(|acc| {
            if !acc.is_empty() {
                all_empty = false;
            }
        });
        all_empty
    }

    fn internal_token_bytes(&self, internal_token: u32) -> Option<&[u8]> {
        self.constraint
            .internal_token_bytes
            .get(&internal_token)
            .map(Vec::as_slice)
    }

    fn supplement_empty_mask_with_exact_commit(&self, merged: &mut [u64]) {
        if std::env::var_os("GLRMASK_DISABLE_EMPTY_MASK_SUPPLEMENT").is_some() {
            return;
        }

        if merged.iter().any(|&word| word != 0) {
            return;
        }

        let mut commit_buffers = crate::runtime::state::CommitBuffers::default();
        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() || !Self::gss_accumulators_empty(gss) {
                continue;
            }

            let Some(possible_matches) = self
                .constraint
                .possible_matches_for_state_internal(tokenizer_state)
            else {
                continue;
            };

            let candidate_tokens: BTreeSet<u32> = possible_matches
                .values()
                .flat_map(|token_ids| token_ids.iter())
                .collect();

            let probe_state = BTreeMap::from([(tokenizer_state, gss.clone())]);

            for internal_token in candidate_tokens {
                let Some(token_bytes) = self.internal_token_bytes(internal_token) else {
                    continue;
                };

                if !token_bytes_may_commit_from_state_map(
                    self.constraint,
                    &probe_state,
                    token_bytes,
                    &mut commit_buffers,
                ) {
                    continue;
                }

                let internal_token = internal_token as usize;
                let word = internal_token / 64;
                let bit = internal_token % 64;
                if let Some(slot) = merged.get_mut(word) {
                    *slot |= 1u64 << bit;
                }
            }
        }
    }

    /// Merge final_weight contribution into internal token bitmap instead of buf.
    fn merge_final_weight_to_internal(
        &self,
        final_weight: &Weight,
        acc: &DenseMaskAcc,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        if final_weight.is_full() {
            acc.or_into_merged(merged);
        } else {
            acc.or_intersection_into_merged(final_weight, precomputed, merged);
        }
    }

    fn merge_final_weight_for_accs(
        &self,
        final_weight: &Weight,
        accs: &[DenseMaskAcc],
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        for acc in accs {
            self.merge_final_weight_to_internal(final_weight, acc, precomputed, merged);
        }
    }

    fn merge_final_weight_for_gss(
        &self,
        final_weight: &Weight,
        gss: &DenseMaskGSS,
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
    ) {
        gss.for_each_acc(|acc| {
            self.merge_final_weight_to_internal(final_weight, acc, precomputed, merged);
        });
    }

    /// Transform a TerminalsDisallowed into a DenseMaskAcc.
    fn terminals_disallowed_to_dense_acc(
        &self,
        terminals_disallowed: &crate::compiler::glr::accumulator::TerminalsDisallowed,
        internal_tsid: u32,
    ) -> Option<DenseMaskAcc> {
        let universe = &self.constraint.seed_universe_dense;
        let terminal_masks = &self.constraint.seed_terminal_dense;

        if terminals_disallowed.is_empty()
            || terminals_disallowed.values().all(|disallowed| disallowed.is_empty())
        {
            if universe.iter().all(|&word| word == 0) {
                return None;
            }
            let dense: Arc<[u64]> = Arc::from(&**universe);
            return Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense)])));
        }

        let mut dense: Vec<u64> = vec![0u64; universe.len()];
        for (&orig_tokenizer_state, disallowed_in_state) in terminals_disallowed.iter() {
            let tsid = self.constraint.internal_tsid_for_state(orig_tokenizer_state);
            let mut allowed_for_state = universe.to_vec();
            for &terminal_id in disallowed_in_state {
                if let Some(mask) = terminal_masks.get(&(tsid, terminal_id)) {
                    for (allowed_word, mask_word) in allowed_for_state.iter_mut().zip(mask.iter()) {
                        *allowed_word &= !mask_word;
                    }
                }
            }
            for (dense_word, allowed_word) in dense.iter_mut().zip(allowed_for_state.iter()) {
                *dense_word |= *allowed_word;
            }
        }

        if dense.iter().all(|&word| word == 0) {
            None
        } else {
            Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense.into())])))
        }
    }

    /// Seed the mask queue, merging FW contributions into internal token bitmap.
    fn seed_mask_queue_merged(
        &self,
        start_final_weight: Option<&Weight>,
        start_fast_trans: &FxHashMap<i32, (u32, Weight)>,
        precomputed: &DenseTokenMaskCache,
        queue: &mut MaskQueue,
        merged: &mut [u64],
    ) {
        let parser_dwa = self.constraint.parser_dwa();

        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }

            let internal_tsid = self.constraint.internal_tsid_for_state(tokenizer_state);

            // Chain fast path: extract chain from parser GSS and walk DWA inline,
            // avoiding intermediate DenseMaskGSS allocations and BFS queue round-trips.
            if std::env::var_os("GLRMASK_DISABLE_MASK_CHAIN_FAST_PATH").is_none()
                && let Some((chain_states, parser_acc, tail)) = gss.extract_chain_and_tail()
            {
                let Some(mut acc) = self.terminals_disallowed_to_dense_acc(parser_acc, internal_tsid) else {
                    continue;
                };
                let mut chain_merged = vec![0u64; merged.len()];

                // Merge start FW contribution
                if let Some(final_weight) = start_final_weight {
                    self.merge_final_weight_to_internal(final_weight, &acc, precomputed, &mut chain_merged);
                }

                // Walk chain: first state goes through start_fast_trans,
                // subsequent states through the target DWA state's transitions.
                let mut cur_wa_state = parser_dwa.start_state;
                let mut alive = true;

                for (index, parser_state) in chain_states.iter().enumerate() {
                    if index == 0
                        && cur_wa_state == parser_dwa.start_state
                        && *parser_state == 0
                    {
                        continue;
                    }
                    let cur_fast_trans = &self.constraint.dwa_fast_transitions[cur_wa_state as usize];
                    let labels = transition_labels(*parser_state);
                    let mut found_target = None;
                    for label in labels {
                        if let Some((target, weight)) = cur_fast_trans.get(&label) {
                            if weight.is_full() {
                                found_target = Some(*target);
                            } else {
                                match acc.intersect_with_weight(weight, precomputed) {
                                    Some(new_acc) => {
                                        acc = new_acc;
                                        found_target = Some(*target);
                                    }
                                    None => break,
                                }
                            }
                            break;
                        }
                    }

                    let Some(target) = found_target else {
                        alive = false;
                        break;
                    };

                    cur_wa_state = target;
                    let next_dwa_state = &parser_dwa.states[cur_wa_state as usize];

                    // Merge FW at the new DWA state
                    if let Some(final_weight) = &next_dwa_state.final_weight {
                        self.merge_final_weight_to_internal(final_weight, &acc, precomputed, &mut chain_merged);
                    }
                }

                // Enqueue tail if chain walk completed successfully
                if alive && !acc.is_empty() {
                    for (merged_word, chain_word) in merged.iter_mut().zip(chain_merged.iter()) {
                        *merged_word |= *chain_word;
                    }
                    let cur_fast_trans = &self.constraint.dwa_fast_transitions[cur_wa_state as usize];
                    let tail_gss = DenseMaskGSS::from_chain_tail_and_acc(tail, acc);
                    if !tail_gss.is_empty() {
                        tail_gss.for_each_decomposed(|ps, popped| {
                            enqueue_parser_state_transitions(
                                queue,
                                cur_fast_trans,
                                ps,
                                &popped,
                                precomputed,
                            );
                        });
                    }

                    continue;
                }
            }

            // General path: apply_transform_and_decompose for non-chain GSS.
            let (decomposed, root_accs) = gss.apply_transform_and_decompose(|terminals_disallowed| {
                self.terminals_disallowed_to_dense_acc(terminals_disallowed, internal_tsid)
            });

            if decomposed.is_empty() && root_accs.is_empty() {
                continue;
            }

            if let Some(final_weight) = start_final_weight {
                self.merge_final_weight_for_accs(final_weight, &root_accs, precomputed, merged);
                for (_, sub_gss) in &decomposed {
                    self.merge_final_weight_for_gss(final_weight, sub_gss, precomputed, merged);
                }
            }

            for (parser_state, popped) in &decomposed {
                enqueue_parser_state_transitions(
                    queue,
                    start_fast_trans,
                    *parser_state,
                    popped,
                    precomputed,
                );
            }
        }
    }

    pub fn mask(&self) -> Vec<u32> {
        let mut buf = vec![0u32; self.constraint.mask_len()];
        self.fill_mask(&mut buf);
        buf
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        // Check mask cache: if generation matches, state hasn't changed since last fill_mask.
        {
            let cache = self.mask_cache.lock().unwrap();
            if let Some(ref cache_data) = *cache {
                if cache_data.generation == self.generation {
                    buf.copy_from_slice(&cache_data.mask);
                    return;
                }
            }
        }

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            buf.fill(0);
            *self.mask_cache.lock().unwrap() = Some(crate::runtime::state::MaskCacheData {
                generation: self.generation,
                mask: buf.to_vec(),
                merged_dense: Vec::new(),
            });
            return;
        }

        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;

        // Merged internal token bitmap: collect all FW contributions here,
        // then convert to buf once at the end to avoid redundant complement-path passes.
        let mut merged = vec![0u64; dense_words];

        let mut queue = MaskQueue::new();

        let start_state = parser_dwa.start_state;
        let start_dwa_state = &parser_dwa.states[start_state as usize];
        let start_fast_trans = &self.constraint.dwa_fast_transitions[start_state as usize];

        self.seed_mask_queue_merged(
            start_dwa_state.final_weight.as_ref(),
            start_fast_trans,
            precomputed,
            &mut queue,
            &mut merged,
        );

        // Process DWA states depth-first.
        while let Some((_depth, states_at_depth)) = queue.pop_last() {
            for (wa_state, gss) in states_at_depth {
                let dwa_state = &parser_dwa.states[wa_state as usize];
                let fast_trans = &self.constraint.dwa_fast_transitions[wa_state as usize];

                // Merge final weight contribution into internal token bitmap.
                if let Some(final_weight) = &dwa_state.final_weight {
                    self.merge_final_weight_for_gss(final_weight, &gss, precomputed, &mut merged);
                }

                // Chain optimization: if the GSS has a long chain of single-path
                // levels, walk through them directly instead of decomposing one
                // level at a time. This avoids intermediate GSS allocations.
                if std::env::var_os("GLRMASK_DISABLE_MASK_CHAIN_FAST_PATH").is_none()
                    && let Some((chain_states, chain_acc, tail_lower)) = gss.extract_chain_and_tail()
                {
                    let mut acc = chain_acc.clone();
                    let mut cur_wa_state = wa_state;
                    let mut alive = true;
                    let mut chain_merged = vec![0u64; merged.len()];

                    for parser_state in chain_states.iter() {
                        let cur_fast_trans = &self.constraint.dwa_fast_transitions[cur_wa_state as usize];
                        // Look up transition for this parser state
                        let labels = transition_labels(*parser_state);
                        let mut found_target = None;
                        for label in labels {
                            if let Some((target, weight)) = cur_fast_trans.get(&label) {
                                if weight.is_full() {
                                    found_target = Some(*target);
                                } else {
                                    match acc.intersect_with_weight(weight, precomputed) {
                                        Some(new_acc) => {
                                            acc = new_acc;
                                            found_target = Some(*target);
                                        }
                                        None => break,
                                    }
                                }
                                break;
                            }
                        }

                        let Some(target) = found_target else {
                            alive = false;
                            break;
                        };

                        cur_wa_state = target;
                        let next_dwa_state = &parser_dwa.states[cur_wa_state as usize];

                        // Merge final weight at the new DWA state
                        if let Some(final_weight) = &next_dwa_state.final_weight {
                            self.merge_final_weight_to_internal(final_weight, &acc, precomputed, &mut chain_merged);
                        }
                    }

                    // Enqueue the tail if the chain walk completed successfully
                    if alive && !acc.is_empty() {
                        for (merged_word, chain_word) in merged.iter_mut().zip(chain_merged.iter()) {
                            *merged_word |= *chain_word;
                        }
                        let cur_fast_trans = &self.constraint.dwa_fast_transitions[cur_wa_state as usize];
                        let tail_gss = DenseMaskGSS::from_chain_tail_and_acc(tail_lower, acc);
                        if !tail_gss.is_empty() {
                            tail_gss.for_each_decomposed(|ps, popped| {
                                enqueue_parser_state_transitions(
                                    &mut queue,
                                    cur_fast_trans,
                                    ps,
                                    &popped,
                                    precomputed,
                                );
                            });
                        }

                        continue;
                    }
                }

                // Standard path: decompose one level at a time.
                gss.for_each_decomposed(|parser_state, popped| {
                    enqueue_parser_state_transitions(
                        &mut queue,
                        fast_trans,
                        parser_state,
                        &popped,
                        precomputed,
                    );
                });
            }
        }

        self.supplement_empty_mask_with_exact_commit(&mut merged);

        // Convert merged internal token bitmap to output buffer.
        // Try incremental update from cached mask if delta is small.
        let did_incremental = {
            let cache = self.mask_cache.lock().unwrap();
            if let Some(ref cache_data) = *cache {
                if !cache_data.merged_dense.is_empty()
                    && cache_data.merged_dense.len() == merged.len()
                {
                    let offsets = &self.constraint.internal_token_buf_offsets;
                    let flat = &self.constraint.internal_token_buf_flat;
                    let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

                    // Compute delta cost (total entries for changed tokens).
                    let mut delta_cost: usize = 0;
                    let mut delta_tokens: usize = 0;
                    for (wi, (&old_w, &new_w)) in cache_data
                        .merged_dense
                        .iter()
                        .zip(merged.iter())
                        .enumerate()
                    {
                        let delta = old_w ^ new_w;
                        if delta == 0 {
                            continue;
                        }
                        delta_tokens += delta.count_ones() as usize;
                        let mut bits = delta;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            let internal_token = wi * 64 + bit;
                            if internal_token < n_internal {
                                delta_cost += (offsets[internal_token + 1] - offsets[internal_token]) as usize;
                            }
                            bits &= bits - 1;
                        }
                    }

                    if delta_tokens > 0 && delta_cost < buf.len() {
                        // Incremental: copy cached mask, apply delta.
                        buf.copy_from_slice(&cache_data.mask);

                        // Apply added/removed tokens.
                        for (wi, (&old_w, &new_w)) in cache_data
                            .merged_dense
                            .iter()
                            .zip(merged.iter())
                            .enumerate()
                        {
                            let delta = old_w ^ new_w;
                            if delta == 0 {
                                continue;
                            }
                            // Removed tokens (was set, now not): AND-NOT
                            let removed = old_w & delta;
                            let mut bits = removed;
                            while bits != 0 {
                                let bit = bits.trailing_zeros() as usize;
                                let internal_token = wi * 64 + bit;
                                if internal_token < n_internal {
                                    let start = offsets[internal_token] as usize;
                                    let end = offsets[internal_token + 1] as usize;
                                    for &(buf_word, mask) in &flat[start..end] {
                                        buf[buf_word as usize] &= !mask;
                                    }
                                }
                                bits &= bits - 1;
                            }
                            // Added tokens (wasn't set, now is): OR
                            let added = new_w & delta;
                            let mut bits = added;
                            while bits != 0 {
                                let bit = bits.trailing_zeros() as usize;
                                let internal_token = wi * 64 + bit;
                                if internal_token < n_internal {
                                    let start = offsets[internal_token] as usize;
                                    let end = offsets[internal_token + 1] as usize;
                                    for &(buf_word, mask) in &flat[start..end] {
                                        buf[buf_word as usize] |= mask;
                                    }
                                }
                                bits &= bits - 1;
                            }
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };

        if !did_incremental {
            buf.fill(0);
            self.constraint.or_internal_dense_to_buf(&merged, buf);
        }

        update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());

        // Update mask cache with current state + computed mask + merged bitvec.
        *self.mask_cache.lock().unwrap() = Some(crate::runtime::state::MaskCacheData {
            generation: self.generation,
            mask: buf.to_vec(),
            merged_dense: merged,
        });
    }
}
