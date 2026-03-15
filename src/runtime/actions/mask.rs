use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::{Weight, WeightDebugStats, reset_weight_debug_stats, snapshot_weight_debug_stats};
use crate::runtime::state::ConstraintStateSummary;
use range_set_blaze::RangeSetBlaze;
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
#[derive(Clone)]
struct DenseMaskAcc {
    start: u32,
    end: u32,
    dense: Arc<[u64]>,
}

impl PartialEq for DenseMaskAcc {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start
            && self.end == other.end
            && (Arc::ptr_eq(&self.dense, &other.dense) || self.dense == other.dense)
    }
}
impl Eq for DenseMaskAcc {}

impl std::hash::Hash for DenseMaskAcc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.end.hash(state);
        self.dense.hash(state);
    }
}

impl DenseMaskAcc {
    fn from_internal_tokens(start: u32, end: u32, tokens: &RangeSetBlaze<u32>, dense_words: usize) -> Self {
        let mut dense = vec![0u64; dense_words];
        for t in tokens.iter() {
            let idx = t as usize / 64;
            let bit = t as usize % 64;
            if let Some(w) = dense.get_mut(idx) {
                *w |= 1u64 << bit;
            }
        }
        Self { start, end, dense: dense.into() }
    }

    fn is_empty(&self) -> bool {
        self.dense.iter().all(|&w| w == 0)
    }

    /// Intersect this accumulator with a DWA weight using precomputed dense masks.
    /// Returns None if the result is empty.
    fn intersect_with_weight(
        &self,
        weight: &Weight,
        precomputed: &rustc_hash::FxHashMap<usize, Box<[u64]>>,
    ) -> Option<Self> {
        let mut result = vec![0u64; self.dense.len()];
        let mut any_nonzero = false;
        let mut fallback_needed = false;

        for (range, other_tokens) in weight.0.range_values() {
            if self.end < *range.start() || *range.end() < self.start {
                continue;
            }
            let key = Arc::as_ptr(other_tokens) as usize;
            if let Some(other_dense) = precomputed.get(&key) {
                for ((r, &s), &o) in result.iter_mut().zip(self.dense.iter()).zip(other_dense.iter()) {
                    let v = s & o;
                    *r |= v;
                    if v != 0 { any_nonzero = true; }
                }
            } else {
                fallback_needed = true;
                break;
            }
        }

        if fallback_needed {
            // Rare case: DWA weight entry not precomputed. Fall back to
            // full intersection then convert.
            return self.intersect_with_weight_fallback(weight);
        }

        if any_nonzero {
            Some(Self { start: self.start, end: self.end, dense: result.into() })
        } else {
            None
        }
    }

    fn intersect_with_weight_fallback(&self, weight: &Weight) -> Option<Self> {
        // Reconstruct a RangeSetBlaze from the dense bitmap, then do the intersection
        // using the original path. This is slow but correct.
        let mut tokens = RangeSetBlaze::new();
        for (wi, &w) in self.dense.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let b = bits.trailing_zeros() as u32;
                tokens.insert((wi as u32) * 64 + b);
                bits &= bits - 1;
            }
        }
        let result = weight.intersect_single_parts(self.start, self.end, &Arc::new(tokens));
        if result.is_empty() {
            return None;
        }
        // Convert result back to dense
        if let Some((start, end, result_tokens)) = result.single_compact_entry_parts() {
            let dense_words = self.dense.len();
            Some(Self::from_internal_tokens(start, end, &result_tokens, dense_words))
        } else {
            // Multi-entry result — shouldn't happen in practice, but handle it
            let dense_words = self.dense.len();
            let mut dense = vec![0u64; dense_words];
            for token_set in result.unique_token_sets() {
                for t in token_set.iter() {
                    let idx = t as usize / 64;
                    let bit = t as usize % 64;
                    if let Some(w) = dense.get_mut(idx) {
                        *w |= 1u64 << bit;
                    }
                }
            }
            Some(Self { start: self.start, end: self.end, dense: dense.into() })
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
        for (range, other_tokens) in final_weight.0.range_values() {
            if self.end < *range.start() || *range.end() < self.start {
                continue;
            }
            let key = Arc::as_ptr(other_tokens) as usize;
            if let Some(other_dense) = precomputed.get(&key) {
                for (wi, (&sd, &od)) in self.dense.iter().zip(other_dense.iter()).enumerate() {
                    let mut overlap = sd & od;
                    while overlap != 0 {
                        let bit = overlap.trailing_zeros() as usize;
                        let internal_token = wi * 64 + bit;
                        if let Some(masks) = constraint.internal_token_buf_masks.get(internal_token) {
                            for &(buf_word, mask) in masks {
                                if let Some(slot) = buf.get_mut(buf_word as usize) {
                                    *slot |= mask;
                                }
                            }
                        }
                        overlap &= overlap - 1;
                    }
                }
            } else {
                // Fallback
                let mut tokens = RangeSetBlaze::new();
                for (wi, &w) in self.dense.iter().enumerate() {
                    let mut bits = w;
                    while bits != 0 {
                        let b = bits.trailing_zeros() as u32;
                        tokens.insert((wi as u32) * 64 + b);
                        bits &= bits - 1;
                    }
                }
                constraint.or_single_weight_intersection_to_buf(
                    self.start, self.end, &Arc::new(tokens), final_weight, buf,
                );
                return;
            }
        }
    }

    /// OR all tokens in this accumulator into the output buffer.
    fn or_to_buf(
        &self,
        constraint: &crate::runtime::constraint::Constraint,
        buf: &mut [u32],
    ) {
        for (wi, &w) in self.dense.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let internal_token = wi * 64 + bit;
                if let Some(masks) = constraint.internal_token_buf_masks.get(internal_token) {
                    for &(buf_word, mask) in masks {
                        if let Some(slot) = buf.get_mut(buf_word as usize) {
                            *slot |= mask;
                        }
                    }
                }
                bits &= bits - 1;
            }
        }
    }
}

impl Merge for DenseMaskAcc {
    fn merge(&self, other: &Self) -> Self {
        let start = self.start.min(other.start);
        let end = self.end.max(other.end);
        let dense: Arc<[u64]> = self.dense.iter()
            .zip(other.dense.iter())
            .map(|(&a, &b)| a | b)
            .collect::<Vec<_>>()
            .into();
        Self { start, end, dense }
    }
}

type DenseMaskGSS = LeveledGSS<u32, DenseMaskAcc>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaskDebugMetrics {
    pub state_summary: ConstraintStateSummary,
    pub weight_ops: WeightDebugStats,
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
    pub max_weighted_gss_depth: isize,
    pub max_depth_bucket_processed: isize,
    pub min_depth_bucket_processed: isize,
    pub max_items_in_depth_bucket: usize,
    pub positive_transitions_hit: usize,
    pub positive_transitions_enqueued: usize,
    pub default_transitions_hit: usize,
    pub default_transitions_enqueued: usize,
    /// Timing breakdown (nanoseconds), only populated by debug_mask_metrics.
    pub seed_ns: u64,
    pub final_weight_ns: u64,
    pub transition_gss_ns: u64,
    pub transition_intersect_ns: u64,
    pub transition_enqueue_ns: u64,
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

    pub fn debug_mask_metrics(&self) -> MaskDebugMetrics {
        let mut metrics = MaskDebugMetrics {
            state_summary: self.summary(),
            mask_words: self.constraint.mask_len(),
            ..MaskDebugMetrics::default()
        };
        let mut buf = vec![0u32; self.constraint.mask_len()];
        reset_weight_debug_stats();
        self.fill_mask_impl(&mut buf, Some(&mut metrics));
        metrics.allowed_token_count = buf.iter().map(|word| word.count_ones() as usize).sum();
        metrics.weight_ops = snapshot_weight_debug_stats();
        metrics.internal_token_dense_words = self.constraint.internal_token_dense_words;
        metrics
    }

    fn fill_mask_impl(&self, buf: &mut [u32], mut metrics: Option<&mut MaskDebugMetrics>) {
        buf.fill(0);

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            return;
        }

        let dense_words = self.constraint.internal_token_dense_words;
        let precomputed = &self.constraint.weight_token_dense_masks;

        let t_total = std::time::Instant::now();
        let t_seed_start = std::time::Instant::now();

        // Queue keyed by GSS depth (process deepest first).
        let mut queue = std::collections::BTreeMap::<
            isize,
            std::collections::BTreeMap<u32, DenseMaskGSS>,
        >::new();

        // Seed: build initial dense GSS from each tokenizer state.
        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }
            let internal_tsid = self.constraint.internal_tsid_for_state(tokenizer_state);
            let seeded = self.seed_weight_dense(internal_tsid, gss, dense_words);
            if seeded.is_empty() {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.seeded_empty_after_weight += 1;
                }
                continue;
            }
            queue
                .entry(seeded.max_depth())
                .or_default()
                .entry(parser_dwa.start_state)
                .and_modify(|existing| *existing = existing.merge(&seeded))
                .or_insert(seeded);
            if let Some(metrics) = metrics.as_deref_mut() {
                metrics.seeded_entries += 1;
            }
        }

        // Process DWA states depth-first.
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.seed_ns = t_seed_start.elapsed().as_nanos() as u64;
        }
        while let Some((depth_key, items)) = queue.pop_last() {
            if let Some(metrics) = metrics.as_deref_mut() {
                metrics.queue_depth_buckets_processed += 1;
                if metrics.queue_depth_buckets_processed == 1 {
                    metrics.max_depth_bucket_processed = depth_key;
                }
                metrics.min_depth_bucket_processed = depth_key;
                metrics.max_items_in_depth_bucket = metrics.max_items_in_depth_bucket.max(items.len());
            }
            for (wa_state, gss) in items {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.queue_items_processed += 1;
                }
                let dwa_state = &parser_dwa.states[wa_state as usize];

                // Final weight → OR allowed tokens into buf.
                let t_fw = std::time::Instant::now();
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
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.final_weight_ns += t_fw.elapsed().as_nanos() as u64;
                }
                let t_gss = std::time::Instant::now();
                let decomposed = gss.decompose_and_pop();
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.parser_states_peeked += decomposed.len();
                    metrics.transition_gss_ns += t_gss.elapsed().as_nanos() as u64;
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
                        let Some((target, weight)) = dwa_state.transitions.get(&label) else {
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
                        let t_int = std::time::Instant::now();
                        let pruned = popped.apply_and_prune(|allowed| {
                            allowed.intersect_with_weight(weight, precomputed)
                        });
                        if pruned.is_empty() {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.transitions_pruned_empty += 1;
                                metrics.transition_intersect_ns += t_int.elapsed().as_nanos() as u64;
                            }
                            continue;
                        }
                        let t_enq = std::time::Instant::now();
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.transition_intersect_ns += t_enq.duration_since(t_int).as_nanos() as u64;
                        }
                        queue
                            .entry(pruned.max_depth())
                            .or_default()
                            .entry(*target)
                            .and_modify(|existing| *existing = existing.merge(&pruned))
                            .or_insert(pruned);
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.transitions_enqueued += 1;
                            if is_default {
                                metrics.default_transitions_enqueued += 1;
                            } else {
                                metrics.positive_transitions_enqueued += 1;
                            }
                            metrics.transition_enqueue_ns += t_enq.elapsed().as_nanos() as u64;
                        }
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

        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.total_ns = t_total.elapsed().as_nanos() as u64;
        }
    }

    fn seed_weight_dense(
        &self,
        internal_tsid: u32,
        gss: &crate::compiler::glr::parser::ParserGSS,
        dense_words: usize,
    ) -> DenseMaskGSS {
        gss.apply_and_prune(|terminals_disallowed| {
            let mut allowed = self.constraint.internal_token_universe();
            if terminals_disallowed.is_empty()
                || terminals_disallowed.values().all(|disallowed| disallowed.is_empty())
            {
                if allowed.is_empty() {
                    return None;
                }
                return Some(
                    DenseMaskAcc::from_internal_tokens(internal_tsid, internal_tsid, &allowed, dense_words),
                );
            }

            for (&orig_tokenizer_state, disallowed_in_state) in terminals_disallowed {
                if disallowed_in_state.is_empty() {
                    continue;
                }

                let state_matches = self.constraint.possible_matches_for_state_internal(orig_tokenizer_state);
                if !state_matches.is_empty() {
                    for (terminal_id, llm_tokens) in state_matches {
                        if disallowed_in_state.contains(&terminal_id) {
                            allowed = allowed - llm_tokens;
                        }
                    }
                }
            }

            if allowed.is_empty() {
                None
            } else {
                Some(
                    DenseMaskAcc::from_internal_tokens(internal_tsid, internal_tsid, &allowed, dense_words),
                )
            }
        })
    }

}
