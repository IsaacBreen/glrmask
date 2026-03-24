use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::{Weight, WeightStats, reset_weight_stats, snapshot_weight_stats};
use crate::runtime::state::ConstraintStateSummary;
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaskMetrics {
    pub state_summary: ConstraintStateSummary,
    pub weight_ops: WeightStats,
    pub mask_words: usize,
    pub allowed_token_count: usize,
    pub seeded_entries: usize,
    pub seeded_empty_after_weight: usize,
    pub queue_depth_buckets_processed: usize,
    pub queue_items_processed: usize,
    pub final_weight_checks: usize,
    pub final_weight_full_hits: usize,
    pub final_weight_intersection_hits: usize,
    pub parser_states_peeked: usize,
    pub transitions_considered: usize,
    pub transitions_hit: usize,
    pub transitions_missing: usize,
    pub transitions_popped_empty: usize,
    pub transitions_pruned_empty: usize,
    pub transitions_enqueued: usize,
    pub max_queue_items: usize,
    pub max_weighted_gss_top_values: usize,
    pub max_weighted_gss_unique_nodes: usize,
    pub max_weighted_gss_total_edges: usize,
    pub max_weighted_gss_depth: u32,
    pub max_depth_bucket_processed: u32,
    pub min_depth_bucket_processed: u32,
    pub max_items_in_depth_bucket: usize,
    pub positive_transitions_hit: usize,
    pub positive_transitions_enqueued: usize,
    pub default_transitions_hit: usize,
    pub default_transitions_enqueued: usize,
    /// Timing breakdown (nanoseconds), only populated by `mask_metrics`.
    pub seed_ns: u64,
    pub final_weight_ns: u64,
    pub transition_gss_ns: u64,
    pub transition_intersect_ns: u64,
    pub transition_enqueue_ns: u64,
    pub queue_pop_ns: u64,
    pub bfs_loop_ns: u64,
    pub total_ns: u64,
    pub internal_token_dense_words: usize,
}

impl<'a> ConstraintState<'a> {
    pub fn mask(&self) -> Vec<u32> {
        let mut buf = vec![0u32; self.constraint.mask_len()];
        self.fill_mask(&mut buf);
        buf
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        self.fill_mask_impl(buf, None);
    }

    pub fn mask_metrics(&self) -> MaskMetrics {
        let mut metrics = MaskMetrics {
            state_summary: self.summary(),
            mask_words: self.constraint.mask_len(),
            ..MaskMetrics::default()
        };
        let mut buf = vec![0u32; self.constraint.mask_len()];
        reset_weight_stats();
        self.fill_mask_impl(&mut buf, Some(&mut metrics));
        metrics.allowed_token_count = buf.iter().map(|word| word.count_ones() as usize).sum();
        metrics.weight_ops = snapshot_weight_stats();
        metrics.internal_token_dense_words = self.constraint.internal_token_dense_words;
        metrics
    }

    fn fill_mask_impl(&self, buf: &mut [u32], mut metrics: Option<&mut MaskMetrics>) {
        buf.fill(0);

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            return;
        }

        let precomputed = &self.constraint.weight_token_dense_masks;
        let timed = metrics.is_some();

        let t_total = if timed { Some(std::time::Instant::now()) } else { None };
        let t_seed_start = if timed { Some(std::time::Instant::now()) } else { None };

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
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.seeded_empty_after_weight += 1;
                }
                continue;
            }
            if let Some(metrics) = metrics.as_deref_mut() {
                metrics.seeded_entries += 1;
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
                    let new_depth = pruned.max_depth();
                    queue
                        .entry(new_depth)
                        .or_default()
                        .entry(*target)
                        .and_modify(|existing| *existing = existing.merge(&pruned))
                        .or_insert(pruned);
                }
            }
        }

        // Process DWA states depth-first.
        if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_seed_start) {
            metrics.seed_ns = t.elapsed().as_nanos() as u64;
        }
        let t_bfs_loop = if timed { Some(std::time::Instant::now()) } else { None };
        while let Some((max_depth, states_at_depth)) = queue.pop_last() {
            let t_pop = if timed { Some(std::time::Instant::now()) } else { None };
            let items: Vec<(u32, DenseMaskGSS)> = states_at_depth.into_iter().collect();
            if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_pop) {
                metrics.queue_pop_ns += t.elapsed().as_nanos() as u64;
                metrics.queue_depth_buckets_processed += 1;
                if metrics.queue_depth_buckets_processed == 1 {
                    metrics.max_depth_bucket_processed = max_depth;
                }
                metrics.min_depth_bucket_processed = max_depth;
                metrics.max_items_in_depth_bucket = metrics.max_items_in_depth_bucket.max(items.len());
            }
            for (wa_state, gss) in items {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.queue_items_processed += 1;
                }
                let dwa_state = &parser_dwa.states[wa_state as usize];
                let fast_trans = &self.constraint.dwa_fast_transitions[wa_state as usize];

                // Final weight → OR allowed tokens into buf.
                let t_fw = if timed { Some(std::time::Instant::now()) } else { None };
                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(metrics) = metrics.as_deref_mut() {
                        metrics.final_weight_checks += 1;
                    }
                    if final_weight.is_full() {
                        let mut hit = false;
                        gss.for_each_acc(|acc| {
                            acc.or_to_buf(&self.constraint, buf);
                            hit = true;
                        });
                        if hit {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.final_weight_full_hits += 1;
                            }
                        }
                    } else {
                        let mut hit = false;
                        gss.for_each_acc(|acc| {
                            acc.or_intersection_to_buf(
                                &self.constraint, final_weight, precomputed, buf,
                            );
                            hit = true;
                        });
                        if hit {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.final_weight_intersection_hits += 1;
                            }
                        }
                    }
                }

                // Advance through DWA transitions for each parser state.
                if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_fw) {
                    metrics.final_weight_ns += t.elapsed().as_nanos() as u64;
                }
                let t_gss = if timed { Some(std::time::Instant::now()) } else { None };
                let decomposed = gss.decompose_and_pop();
                if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_gss) {
                    metrics.parser_states_peeked += decomposed.len();
                    metrics.transition_gss_ns += t.elapsed().as_nanos() as u64;
                }
                for (parser_state, popped) in &decomposed {
                    let labels = [
                        (crate::compiler::glr::labels::encode_positive_label(*parser_state), false),
                        (crate::compiler::glr::labels::DEFAULT_LABEL, true),
                    ];
                    for (label, is_default) in labels {
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.transitions_considered += 1;
                        }
                        let Some((target, weight)) = fast_trans.get(&label) else {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.transitions_missing += 1;
                            }
                            continue;
                        };
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.transitions_hit += 1;
                            if is_default {
                                metrics.default_transitions_hit += 1;
                            } else {
                                metrics.positive_transitions_hit += 1;
                            }
                        }
                        let t_int = if timed { Some(std::time::Instant::now()) } else { None };
                        let pruned = popped.apply_and_prune_no_promote(|allowed| {
                            allowed.intersect_with_weight(weight, precomputed)
                        });
                        if pruned.is_empty() {
                            if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_int) {
                                metrics.transitions_pruned_empty += 1;
                                metrics.transition_intersect_ns += t.elapsed().as_nanos() as u64;
                            }
                            continue;
                        }
                        let t_enq = if timed { Some(std::time::Instant::now()) } else { None };
                        if let (Some(metrics), Some(te), Some(ti)) = (metrics.as_deref_mut(), t_enq, t_int) {
                            metrics.transition_intersect_ns += te.duration_since(ti).as_nanos() as u64;
                        }
                        let new_depth = pruned.max_depth();
                        queue
                            .entry(new_depth)
                            .or_default()
                            .entry(*target)
                            .and_modify(|existing| *existing = existing.merge(&pruned))
                            .or_insert(pruned);
                        if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_enq) {
                            metrics.transitions_enqueued += 1;
                            if is_default {
                                metrics.default_transitions_enqueued += 1;
                            } else {
                                metrics.positive_transitions_enqueued += 1;
                            }
                            metrics.transition_enqueue_ns += t.elapsed().as_nanos() as u64;
                        }
                    }
                }
            }
        }
        if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_bfs_loop) {
            metrics.bfs_loop_ns = t.elapsed().as_nanos() as u64;
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

        if let (Some(metrics), Some(t)) = (metrics.as_deref_mut(), t_total) {
            metrics.total_ns = t.elapsed().as_nanos() as u64;
        }
    }

}
