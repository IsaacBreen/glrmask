use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;
use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// DenseMaskAcc — compact accumulator for mask traversal using dense bitmaps
// ---------------------------------------------------------------------------

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

    /// OR all tokens in this accumulator into the output buffer.
    fn or_to_buf(
        &self,
        constraint: &crate::runtime::constraint::Constraint,
        buf: &mut [u32],
    ) {
        for dense in self.0.values() {
            for (wi, &w) in dense.iter().enumerate() {
                let mut bits = w;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    let masks = &constraint.internal_token_buf_masks[internal_token];
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
    queue: &mut BTreeMap<u32, FxHashMap<u32, DenseMaskGSS>>,
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

impl<'a> ConstraintState<'a> {
    pub fn mask(&self) -> Vec<u32> {
        let mut buf = vec![0u32; self.constraint.mask_len()];
        self.fill_mask(&mut buf);
        buf
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        buf.fill(0);

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            return;
        }

        let precomputed = &self.constraint.weight_token_dense_masks;

        // Depth buckets let us pop the deepest frontier without rescanning or
        // linearly searching for matching (depth, state) entries on enqueue.
        let mut queue: BTreeMap<u32, FxHashMap<u32, DenseMaskGSS>> = BTreeMap::new();

        let start_state = parser_dwa.start_state;
        let start_dwa_state = &parser_dwa.states[start_state as usize];
        let start_fast_trans = &self.constraint.dwa_fast_transitions[start_state as usize];

        // Seed: decompose parser GSS and produce DenseMaskGSS sub-trees directly,
        // skipping the construction of the root-level Branch node.
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
                    if universe.iter().all(|&w| w == 0) {
                        return None;
                    }
                    let dense: Arc<[u64]> = Arc::from(&**universe);
                    return Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense)])));
                }
                let mut dense: Vec<u64> = universe.to_vec();
                for (&orig_tokenizer_state, disallowed_in_state) in terminals_disallowed {
                    for &terminal_id in disallowed_in_state {
                        if let Some(mask) = terminal_masks.get(&(orig_tokenizer_state, terminal_id)) {
                            for (d, m) in dense.iter_mut().zip(mask.iter()) {
                                *d &= !m;
                            }
                        }
                    }
                }
                if dense.iter().all(|&w| w == 0) {
                    None
                } else {
                    Some(DenseMaskAcc(BTreeMap::from([(internal_tsid, dense.into())])))
                }
            });

            if decomposed.is_empty() && root_accs.is_empty() {
                continue;
            }

            // Apply start_state's final_weight to the seed accumulators.
            if let Some(final_weight) = &start_dwa_state.final_weight {
                if final_weight.is_full() {
                    for acc in &root_accs {
                        acc.or_to_buf(&self.constraint, buf);
                    }
                    for (_, sub_gss) in &decomposed {
                        sub_gss.for_each_acc(|acc| {
                            acc.or_to_buf(&self.constraint, buf);
                        });
                    }
                } else {
                    for acc in &root_accs {
                        acc.or_intersection_to_buf(
                            &self.constraint, final_weight, precomputed, buf,
                        );
                    }
                    for (_, sub_gss) in &decomposed {
                        sub_gss.for_each_acc(|acc| {
                            acc.or_intersection_to_buf(
                                &self.constraint, final_weight, precomputed, buf,
                            );
                        });
                    }
                }
            }

            // Apply start_state transitions to each decomposed sub-GSS.
            for (parser_state, popped) in &decomposed {
                let labels = [
                    (crate::compiler::glr::labels::encode_positive_label(*parser_state), false),
                    (crate::compiler::glr::labels::DEFAULT_LABEL, true),
                ];
                for (label, _is_default) in labels {
                    let Some((target, weight)) = start_fast_trans.get(&label) else {
                        continue;
                    };
                    let pruned = popped.apply_and_prune_no_promote(|allowed| {
                        allowed.intersect_with_weight(weight, precomputed)
                    });
                    if pruned.is_empty() {
                        continue;
                    }
                    enqueue_gss(&mut queue, pruned.max_depth(), *target, pruned);
                }
            }
        }

        // Process DWA states depth-first.
        while let Some((_depth, states_at_depth)) = queue.pop_last() {
            let items: Vec<(u32, DenseMaskGSS)> = states_at_depth.into_iter().collect();
            for (wa_state, gss) in items {
                let dwa_state = &parser_dwa.states[wa_state as usize];
                let fast_trans = &self.constraint.dwa_fast_transitions[wa_state as usize];

                // Final weight → OR allowed tokens into buf.
                if let Some(final_weight) = &dwa_state.final_weight {
                    if final_weight.is_full() {
                        gss.for_each_acc(|acc| {
                            acc.or_to_buf(&self.constraint, buf);
                        });
                    } else {
                        gss.for_each_acc(|acc| {
                            acc.or_intersection_to_buf(
                                &self.constraint, final_weight, precomputed, buf,
                            );
                        });
                    }
                }

                // Advance through DWA transitions for each parser state.
                let decomposed = gss.decompose_and_pop();
                for (parser_state, popped) in &decomposed {
                    let labels = [
                        crate::compiler::glr::labels::encode_positive_label(*parser_state),
                        crate::compiler::glr::labels::DEFAULT_LABEL,
                    ];
                    for label in labels {
                        let Some((target, weight)) = fast_trans.get(&label) else {
                            continue;
                        };
                        let pruned = popped.apply_and_prune_no_promote(|allowed| {
                            allowed.intersect_with_weight(weight, precomputed)
                        });
                        if pruned.is_empty() {
                            continue;
                        }
                        enqueue_gss(&mut queue, pruned.max_depth(), *target, pruned);
                    }
                }
            }
        }

        // EOS token: clear unconditionally, then re-set if constraint is complete.
        if let Some(eos_token_id) = self.constraint.eos_token_id {
            let word = eos_token_id as usize / 32;
            let bit = eos_token_id as usize % 32;
            if let Some(slot) = buf.get_mut(word) {
                *slot &= !(1u32 << bit);
            }
            if self.is_complete() {
                if let Some(slot) = buf.get_mut(word) {
                    *slot |= 1u32 << bit;
                }
            }
        }
    }
}
