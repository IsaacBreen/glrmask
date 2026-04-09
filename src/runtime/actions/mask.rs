use crate::runtime::state::ConstraintState;
use crate::compiler::glr::labels::{encode_positive_label, DEFAULT_LABEL};
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
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

    /// OR this accumulator's tokens (intersected with `final_weight`) into the output buffer.
    fn or_intersection_to_buf(
        &self,
        constraint: &crate::runtime::constraint::Constraint,
        final_weight: &Weight,
        precomputed: &rustc_hash::FxHashMap<usize, Box<[u64]>>,
        buf: &mut [u32],
    ) {
        for (&tsid, dense) in &self.0 {
            let Some(token_set) = final_weight.0.get(tsid) else {
                continue;
            };
            let key = Arc::as_ptr(token_set) as usize;
            if let Some(other_dense) = precomputed.get(&key) {
                constraint.or_dense_intersection_to_buf(dense, other_dense, buf);
            } else {
                let tokens = Self::dense_to_tokens(dense);
                constraint.or_single_weight_intersection_to_buf(
                    tsid,
                    tsid,
                    &Arc::new(tokens),
                    final_weight,
                    buf,
                );
            }
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

    /// OR all tokens in this accumulator into the output buffer.
    fn or_to_buf(
        &self,
        constraint: &crate::runtime::constraint::Constraint,
        buf: &mut [u32],
    ) {
        let word_groups = &constraint.word_group_buf_masks;
        let all_mask = &constraint.all_tokens_buf_mask;
        let internal_masks = &constraint.internal_token_buf_masks;
        for dense in self.0.values() {
            // Super-fast path: check if entire dense matches full universe.
            let n_full_groups = word_groups.len().saturating_sub(1);
            if !all_mask.is_empty()
                && dense.len() == word_groups.len()
                && dense[..n_full_groups].iter().all(|&w| w == !0u64)
            {
                let last_expected = if internal_masks.len() % 64 == 0 {
                    !0u64
                } else {
                    (1u64 << (internal_masks.len() % 64)) - 1
                };
                if dense[n_full_groups] == last_expected {
                    for (i, &mask) in all_mask.iter().enumerate() {
                        buf[i] |= mask;
                    }
                    continue;
                }
            }

            // Complement path: when most words are full, start from the
            // all-tokens mask and subtract the few missing tokens.
            // This turns O(n_full_groups × buf_len) into O(n_missing × buf_len).
            let n_full = dense.iter().filter(|&&w| w == !0u64).count();
            let n_non_full = dense.len() - n_full;
            if !all_mask.is_empty()
                && dense.len() == word_groups.len()
                && n_full > 0
                && n_non_full < n_full
            {
                // Build complement: all_mask with missing tokens removed.
                let mut temp: Vec<u32> = all_mask.to_vec();
                for (wi, &w) in dense.iter().enumerate() {
                    if w == !0u64 {
                        continue;
                    }
                    if w == 0 {
                        // Remove entire group from temp.
                        if let Some(group) = word_groups.get(wi) {
                            for (i, &mask) in group.iter().enumerate() {
                                temp[i] &= !mask;
                            }
                        }
                    } else {
                        // Remove only the missing bits.
                        let missing = !w;
                        let n_internal = internal_masks.len();
                        let mut bits = missing;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            let internal_token = wi * 64 + bit;
                            if internal_token < n_internal {
                                for &(buf_word, mask) in &internal_masks[internal_token] {
                                    temp[buf_word as usize] &= !mask;
                                }
                            }
                            bits &= bits - 1;
                        }
                    }
                }
                for (i, &mask) in temp.iter().enumerate() {
                    buf[i] |= mask;
                }
                continue;
            }

            for (wi, &w) in dense.iter().enumerate() {
                if w == 0 {
                    continue;
                }
                // Fast path: all 64 internal tokens in this word are set.
                if w == !0u64 {
                    if let Some(group) = word_groups.get(wi) {
                        for (i, &mask) in group.iter().enumerate() {
                            buf[i] |= mask;
                        }
                        continue;
                    }
                }
                // Slow path: iterate individual bits.
                let mut bits = w;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    let masks = &internal_masks[internal_token];
                    for &(buf_word, mask) in masks {
                        buf[buf_word as usize] |= mask;
                    }
                    bits &= bits - 1;
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
    fn or_final_weight_to_buf(
        &self,
        final_weight: &Weight,
        acc: &DenseMaskAcc,
        precomputed: &DenseTokenMaskCache,
        buf: &mut [u32],
    ) {
        if final_weight.is_full() {
            acc.or_to_buf(&self.constraint, buf);
        } else {
            acc.or_intersection_to_buf(&self.constraint, final_weight, precomputed, buf);
        }
    }

    fn or_final_weight_for_accs(
        &self,
        final_weight: &Weight,
        accs: &[DenseMaskAcc],
        precomputed: &DenseTokenMaskCache,
        buf: &mut [u32],
    ) {
        for acc in accs {
            self.or_final_weight_to_buf(final_weight, acc, precomputed, buf);
        }
    }

    fn or_final_weight_for_gss(
        &self,
        final_weight: &Weight,
        gss: &DenseMaskGSS,
        precomputed: &DenseTokenMaskCache,
        buf: &mut [u32],
    ) {
        gss.for_each_acc(|acc| {
            self.or_final_weight_to_buf(final_weight, acc, precomputed, buf);
        });
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

    /// Seed the mask queue, merging FW contributions into internal token bitmap.
    fn seed_mask_queue_merged(
        &self,
        start_final_weight: Option<&Weight>,
        start_fast_trans: &FxHashMap<i32, (u32, Weight)>,
        precomputed: &DenseTokenMaskCache,
        queue: &mut MaskQueue,
        merged: &mut [u64],
    ) {
        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }

            let internal_tsid = self.constraint.internal_tsid_for_state(tokenizer_state);
            let universe = &self.constraint.seed_universe_dense;
            let terminal_masks = &self.constraint.seed_terminal_dense;

            let (decomposed, root_accs) = gss.apply_transform_and_decompose(|terminals_disallowed| {
                if terminals_disallowed.is_empty()
                    || terminals_disallowed.values().all(|disallowed| disallowed.is_empty())
                {
                    if universe.iter().all(|&word| word == 0) {
                        return None;
                    }
                    let dense: Arc<[u64]> = Arc::from(&**universe);
                    return Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense)])));
                }

                let mut dense: Vec<u64> = universe.to_vec();
                for (&orig_tokenizer_state, disallowed_in_state) in terminals_disallowed.iter() {
                    let internal_tsid = self.constraint.internal_tsid_for_state(orig_tokenizer_state);
                    for &terminal_id in disallowed_in_state {
                        if let Some(mask) = terminal_masks.get(&(internal_tsid, terminal_id)) {
                            for (dense_word, mask_word) in dense.iter_mut().zip(mask.iter()) {
                                *dense_word &= !mask_word;
                            }
                        }
                    }
                }

                if dense.iter().all(|&word| word == 0) {
                    None
                } else {
                    Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense.into())])))
                }
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

    fn seed_mask_queue(
        &self,
        start_final_weight: Option<&Weight>,
        start_fast_trans: &FxHashMap<i32, (u32, Weight)>,
        precomputed: &DenseTokenMaskCache,
        queue: &mut MaskQueue,
        buf: &mut [u32],
    ) {
        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }

            let internal_tsid = self.constraint.internal_tsid_for_state(tokenizer_state);
            let universe = &self.constraint.seed_universe_dense;
            let terminal_masks = &self.constraint.seed_terminal_dense;

            let (decomposed, root_accs) = gss.apply_transform_and_decompose(|terminals_disallowed| {
                if terminals_disallowed.is_empty()
                    || terminals_disallowed.values().all(|disallowed| disallowed.is_empty())
                {
                    if universe.iter().all(|&word| word == 0) {
                        return None;
                    }
                    let dense: Arc<[u64]> = Arc::from(&**universe);
                    return Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense)])));
                }

                let mut dense: Vec<u64> = universe.to_vec();
                for (&orig_tokenizer_state, disallowed_in_state) in terminals_disallowed.iter() {
                    let internal_tsid = self.constraint.internal_tsid_for_state(orig_tokenizer_state);
                    for &terminal_id in disallowed_in_state {
                        if let Some(mask) = terminal_masks.get(&(internal_tsid, terminal_id)) {
                            for (dense_word, mask_word) in dense.iter_mut().zip(mask.iter()) {
                                *dense_word &= !mask_word;
                            }
                        }
                    }
                }

                if dense.iter().all(|&word| word == 0) {
                    None
                } else {
                    Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense.into())])))
                }
            });

            if decomposed.is_empty() && root_accs.is_empty() {
                continue;
            }

            if let Some(final_weight) = start_final_weight {
                self.or_final_weight_for_accs(final_weight, &root_accs, precomputed, buf);
                for (_, sub_gss) in &decomposed {
                    self.or_final_weight_for_gss(final_weight, sub_gss, precomputed, buf);
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
        // Check mask cache: if parser state matches cached snapshot, reuse mask.
        {
            let cache = self.mask_cache.lock().unwrap();
            if let Some(ref cache_data) = *cache {
                if cache_data.state_snapshot == self.state {
                    buf.copy_from_slice(&cache_data.mask);
                    return;
                }
            }
        }

        buf.fill(0);

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            *self.mask_cache.lock().unwrap() = Some(crate::runtime::state::MaskCacheData {
                state_snapshot: self.state.clone(),
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
                if let Some((chain_states, chain_acc, tail_lower)) = gss.extract_chain_and_tail() {
                    let mut acc = chain_acc.clone();
                    let mut cur_wa_state = wa_state;
                    let mut alive = true;

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
                            self.merge_final_weight_to_internal(final_weight, &acc, precomputed, &mut merged);
                        }
                    }

                    // Enqueue the tail if the chain walk completed successfully
                    if alive && !acc.is_empty() {
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
                    }
                } else {
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
        }

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

                    if delta_tokens > 0 && delta_cost < buf.len() * 2 {
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
            self.constraint.or_internal_dense_to_buf(&merged, buf);
        }

        update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());

        // Update mask cache with current state + computed mask + merged bitvec.
        *self.mask_cache.lock().unwrap() = Some(crate::runtime::state::MaskCacheData {
            state_snapshot: self.state.clone(),
            mask: buf.to_vec(),
            merged_dense: merged,
        });
    }

    /// Like `fill_mask` (merged path, no cache) but returns detailed timing breakdown.
    /// Returns: (total_ns, seed_ns, bfs_decompose_ns, bfs_enqueue_ns, bfs_fw_merge_ns,
    ///           convert_ns, cache_update_ns, n_dwa_visits, n_decompose_ops, n_fw_ops, n_enqueue_calls)
    pub fn fill_mask_timed(&self, buf: &mut [u32]) -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
        use std::time::Instant;
        let t_total = Instant::now();

        buf.fill(0);

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            let total = t_total.elapsed().as_nanos() as u64;
            return (total, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;
        let mut merged = vec![0u64; dense_words];
        let mut queue = MaskQueue::new();

        let start_state = parser_dwa.start_state;
        let start_dwa_state = &parser_dwa.states[start_state as usize];
        let start_fast_trans = &self.constraint.dwa_fast_transitions[start_state as usize];

        let t_seed = Instant::now();
        self.seed_mask_queue_merged(
            start_dwa_state.final_weight.as_ref(),
            start_fast_trans,
            precomputed,
            &mut queue,
            &mut merged,
        );
        let seed_ns = t_seed.elapsed().as_nanos() as u64;

        let mut decompose_ns: u64 = 0;
        let mut enqueue_ns: u64 = 0;
        let mut fw_merge_ns: u64 = 0;
        let mut n_dwa_visits: u64 = 0;
        let mut n_decompose_ops: u64 = 0;
        let mut n_fw_ops: u64 = 0;
        let mut n_enqueue_calls: u64 = 0;

        while let Some((_depth, states_at_depth)) = queue.pop_last() {
            for (wa_state, gss) in states_at_depth {
                n_dwa_visits += 1;
                let dwa_state = &parser_dwa.states[wa_state as usize];
                let fast_trans = &self.constraint.dwa_fast_transitions[wa_state as usize];

                if let Some(final_weight) = &dwa_state.final_weight {
                    n_fw_ops += 1;
                    let t = Instant::now();
                    self.merge_final_weight_for_gss(final_weight, &gss, precomputed, &mut merged);
                    fw_merge_ns += t.elapsed().as_nanos() as u64;
                }

                let t = Instant::now();
                let mut local_decompose = 0u64;
                let mut local_enqueue = 0u64;

                if let Some((chain_states, chain_acc, tail_lower)) = gss.extract_chain_and_tail() {
                    let mut acc = chain_acc.clone();
                    let mut cur_wa_state = wa_state;
                    let mut alive = true;

                    for parser_state in chain_states.iter() {
                        local_decompose += 1;
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
                        n_enqueue_calls += 1;
                        local_enqueue += 1;
                        let next_dwa_state = &parser_dwa.states[cur_wa_state as usize];

                        if let Some(final_weight) = &next_dwa_state.final_weight {
                            n_fw_ops += 1;
                            let t_fw = Instant::now();
                            self.merge_final_weight_to_internal(final_weight, &acc, precomputed, &mut merged);
                            fw_merge_ns += t_fw.elapsed().as_nanos() as u64;
                        }
                    }

                    if alive && !acc.is_empty() {
                        let cur_fast_trans = &self.constraint.dwa_fast_transitions[cur_wa_state as usize];
                        let tail_gss = DenseMaskGSS::from_chain_tail_and_acc(tail_lower, acc);
                        if !tail_gss.is_empty() {
                            tail_gss.for_each_decomposed(|ps, popped| {
                                local_decompose += 1;
                                n_enqueue_calls += 1;
                                local_enqueue += 1;
                                enqueue_parser_state_transitions(
                                    &mut queue,
                                    cur_fast_trans,
                                    ps,
                                    &popped,
                                    precomputed,
                                );
                            });
                        }
                    }
                } else {
                    gss.for_each_decomposed(|parser_state, popped| {
                        local_decompose += 1;
                        n_enqueue_calls += 1;
                        enqueue_parser_state_transitions(
                            &mut queue,
                            fast_trans,
                            parser_state,
                            &popped,
                            precomputed,
                        );
                        local_enqueue += 1;
                    });
                }
                let elapsed = t.elapsed().as_nanos() as u64;
                n_decompose_ops += local_decompose;
                // Split time proportionally between decompose and enqueue.
                // Since for_each_decomposed interleaves them, attribute half to each.
                decompose_ns += elapsed / 2;
                enqueue_ns += elapsed - elapsed / 2;
            }
        }

        let t_convert = Instant::now();
        self.constraint.or_internal_dense_to_buf(&merged, buf);
        update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());
        let convert_ns = t_convert.elapsed().as_nanos() as u64;

        let t_cache = Instant::now();
        *self.mask_cache.lock().unwrap() = Some(crate::runtime::state::MaskCacheData {
            state_snapshot: self.state.clone(),
            mask: buf.to_vec(),
            merged_dense: merged,
        });
        let cache_update_ns = t_cache.elapsed().as_nanos() as u64;

        let total_ns = t_total.elapsed().as_nanos() as u64;
        (total_ns, seed_ns, decompose_ns, enqueue_ns, fw_merge_ns,
         convert_ns, cache_update_ns, n_dwa_visits, n_decompose_ops, n_fw_ops, n_enqueue_calls)
    }

    /// Like `fill_mask` but returns detailed profiling stats.
    /// Returns a tuple of 10 u64 values:
    /// (total_ns, seed_ns, bfs_ns, final_weight_ns, decompose_ns, enqueue_ns,
    ///  n_depth_buckets, n_dwa_visits, n_decompose_ops, n_final_weight_ops)
    pub fn fill_mask_profiled(&self, buf: &mut [u32]) -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
        use std::time::Instant;
        let t_total = Instant::now();

        buf.fill(0);

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            return (t_total.elapsed().as_nanos() as u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        let precomputed = &self.constraint.weight_token_dense_masks;
        let mut queue = MaskQueue::new();

        let start_state = parser_dwa.start_state;
        let start_dwa_state = &parser_dwa.states[start_state as usize];
        let start_fast_trans = &self.constraint.dwa_fast_transitions[start_state as usize];

        let t_seed = Instant::now();
        self.seed_mask_queue(
            start_dwa_state.final_weight.as_ref(),
            start_fast_trans,
            precomputed,
            &mut queue,
            buf,
        );
        let seed_ns = t_seed.elapsed().as_nanos() as u64;

        let t_bfs = Instant::now();
        let mut n_depth_buckets: u64 = 0;
        let mut n_dwa_visits: u64 = 0;
        let mut n_decompose_ops: u64 = 0;
        let mut n_final_weight_ops: u64 = 0;
        let mut final_weight_ns: u64 = 0;
        let mut decompose_ns: u64 = 0;
        let mut enqueue_ns: u64 = 0;

        while let Some((_depth, states_at_depth)) = queue.pop_last() {
            n_depth_buckets += 1;
            for (wa_state, gss) in states_at_depth {
                n_dwa_visits += 1;
                let dwa_state = &parser_dwa.states[wa_state as usize];
                let fast_trans = &self.constraint.dwa_fast_transitions[wa_state as usize];

                if let Some(final_weight) = &dwa_state.final_weight {
                    n_final_weight_ops += 1;
                    let t = Instant::now();
                    self.or_final_weight_for_gss(final_weight, &gss, precomputed, buf);
                    final_weight_ns += t.elapsed().as_nanos() as u64;
                }

                let t = Instant::now();
                let mut local_decompose = 0u64;
                gss.for_each_decomposed(|parser_state, popped| {
                    local_decompose += 1;
                    enqueue_parser_state_transitions(
                        &mut queue,
                        fast_trans,
                        parser_state,
                        &popped,
                        precomputed,
                    );
                });
                let elapsed = t.elapsed().as_nanos() as u64;
                n_decompose_ops += local_decompose;
                decompose_ns += elapsed / 2;
                enqueue_ns += elapsed - elapsed / 2;
            }
        }
        let bfs_ns = t_bfs.elapsed().as_nanos() as u64;

        let t_eoscheck = Instant::now();
        update_eos_mask(buf, self.constraint.eos_token_id, self.is_complete());
        let is_complete_ns = t_eoscheck.elapsed().as_nanos() as u64;

        let total_ns = t_total.elapsed().as_nanos() as u64;
        (total_ns, seed_ns, bfs_ns, final_weight_ns, decompose_ns, enqueue_ns,
         is_complete_ns, n_depth_buckets, n_dwa_visits, n_decompose_ops, n_final_weight_ops)
    }
}
