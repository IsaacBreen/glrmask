use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, LeveledGSSSummary, Merge};
use crate::ds::weight::{Weight, WeightDebugStats, reset_weight_debug_stats, snapshot_weight_debug_stats};
use crate::runtime::state::ConstraintStateSummary;
use range_set_blaze::RangeSetBlaze;
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, Hash)]
enum RuntimeWeight {
    Single {
        start: u32,
        end: u32,
        tokens: Arc<RangeSetBlaze<u32>>,
    },
    Multi(Weight),
}

impl RuntimeWeight {
    fn from_weight(weight: Weight) -> Self {
        if let Some((start, end, tokens)) = weight.single_compact_entry_parts() {
            Self::Single { start, end, tokens }
        } else {
            Self::Multi(weight)
        }
    }

    fn from_token_set_for_tsid(tsid: u32, tokens: RangeSetBlaze<u32>) -> Self {
        Self::Single {
            start: tsid,
            end: tsid,
            tokens: Arc::new(tokens),
        }
    }

    fn to_weight(&self) -> Weight {
        match self {
            Self::Single { start, end, tokens } => {
                Weight::from_compact_ranges(std::iter::once((*start..=*end, tokens.ranges())))
            }
            Self::Multi(weight) => weight.clone(),
        }
    }

    fn intersect_with_weight(&self, other: &Weight) -> Option<Self> {
        let next = match self {
            Self::Single { start, end, tokens } => other.intersect_single_parts(*start, *end, tokens),
            Self::Multi(weight) => weight.intersection(other),
        };
        (!next.is_empty()).then(|| Self::from_weight(next))
    }

    fn or_to_buf(&self, state: &ConstraintState<'_>, buf: &mut [u32]) {
        match self {
            Self::Single { tokens, .. } => state.constraint.or_internal_token_set_to_buf(tokens, buf),
            Self::Multi(weight) => state.constraint.or_weight_to_buf(weight, buf),
        }
    }

    fn or_intersection_to_buf(
        &self,
        state: &ConstraintState<'_>,
        other: &Weight,
        buf: &mut [u32],
    ) {
        match self {
            Self::Single { start, end, tokens } => state
                .constraint
                .or_single_weight_intersection_to_buf(*start, *end, tokens, other, buf),
            Self::Multi(weight) => {
                let allowed = weight.intersection(other);
                state.constraint.or_weight_to_buf(&allowed, buf);
            }
        }
    }
}

impl Merge for RuntimeWeight {
    fn merge(&self, other: &Self) -> Self {
        match (self, other) {
            (
                Self::Single {
                    start: left_start,
                    end: left_end,
                    tokens: left_tokens,
                },
                Self::Single {
                    start: right_start,
                    end: right_end,
                    tokens: right_tokens,
                },
            ) if left_tokens == right_tokens
                && (left_end.saturating_add(1) >= *right_start || right_end.saturating_add(1) >= *left_start) =>
            {
                Self::Single {
                    start: (*left_start).min(*right_start),
                    end: (*left_end).max(*right_end),
                    tokens: Arc::clone(left_tokens),
                }
            }
            _ => Self::from_weight(self.to_weight().union(&other.to_weight())),
        }
    }
}

type WeightedParserGSS = LeveledGSS<u32, RuntimeWeight>;

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
}

fn queue_item_count(
    queue: &std::collections::BTreeMap<
        isize,
        std::collections::BTreeMap<u32, WeightedParserGSS>,
    >,
) -> usize {
    queue.values().map(|items| items.len()).sum()
}

fn update_weighted_gss_metrics(metrics: &mut MaskDebugMetrics, gss: &WeightedParserGSS) {
    let summary: LeveledGSSSummary = gss.summary();
    metrics.max_weighted_gss_top_values = metrics
        .max_weighted_gss_top_values
        .max(summary.top_values_count);
    metrics.max_weighted_gss_unique_nodes = metrics
        .max_weighted_gss_unique_nodes
        .max(summary.total_unique_nodes);
    metrics.max_weighted_gss_total_edges = metrics
        .max_weighted_gss_total_edges
        .max(summary.total_edges);
    metrics.max_weighted_gss_depth = metrics.max_weighted_gss_depth.max(summary.max_depth);
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
        metrics
    }

    fn fill_mask_impl(&self, buf: &mut [u32], mut metrics: Option<&mut MaskDebugMetrics>) {
        buf.fill(0);

        let parser_dwa = self.constraint.parser_dwa();
        if self.state.is_empty() || parser_dwa.states.is_empty() {
            return;
        }

        // Queue keyed by GSS depth (process deepest first).
        let mut queue = std::collections::BTreeMap::<
            isize,
            std::collections::BTreeMap<u32, WeightedParserGSS>,
        >::new();

        // Seed: build initial weighted GSS from each tokenizer state.
        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }
            let internal_tsid = self.constraint.internal_tsid_for_state(tokenizer_state);
            let seeded = self.seed_weight(internal_tsid, gss);
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
                if let Some(seed_bucket) = queue.get(&queue.keys().next_back().copied().unwrap_or_default()) {
                    if let Some(seed_gss) = seed_bucket.get(&parser_dwa.start_state) {
                        update_weighted_gss_metrics(metrics, seed_gss);
                    }
                }
                metrics.max_queue_items = metrics.max_queue_items.max(queue_item_count(&queue));
            }
        }

        // Process DWA states depth-first.
        while let Some((_, items)) = queue.pop_last() {
            if let Some(metrics) = metrics.as_deref_mut() {
                metrics.queue_depth_buckets_processed += 1;
            }
            for (wa_state, gss) in items {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.queue_items_processed += 1;
                    update_weighted_gss_metrics(metrics, &gss);
                }
                let dwa_state = &parser_dwa.states[wa_state as usize];

                // Final weight → OR allowed tokens into buf.
                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(metrics) = metrics.as_deref_mut() {
                        metrics.final_weight_checks += 1;
                    }
                    if let Some(reduced_acc) = gss.reduce_acc() {
                        if final_weight.is_full() {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.final_weight_full_hits += 1;
                            }
                            reduced_acc.or_to_buf(self, buf);
                        } else if let RuntimeWeight::Single { start, end, tokens } = &reduced_acc {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.final_weight_intersection_hits += 1;
                            }
                            self.constraint
                                .or_single_weight_intersection_to_buf(*start, *end, tokens, final_weight, buf);
                        } else {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.final_weight_intersection_hits += 1;
                            }
                            reduced_acc.or_intersection_to_buf(self, final_weight, buf);
                        }
                    }
                }

                // Advance through DWA transitions for each parser state.
                let parser_states = gss.peek_values();
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.parser_states_peeked += parser_states.len();
                }
                for parser_state in parser_states {
                    let mut advance = |label: i32, current: &WeightedParserGSS, metrics: &mut Option<&mut MaskDebugMetrics>| {
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.transitions_considered += 1;
                        }
                        let Some((target, weight)) = dwa_state.transitions.get(&label) else {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.transitions_missing += 1;
                            }
                            return;
                        };
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.transitions_hit += 1;
                        }
                        let isolated = current.isolate(Some(parser_state));
                        let popped = isolated.pop();
                        if popped.is_empty() {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.transitions_popped_empty += 1;
                            }
                            return;
                        }
                        let pruned = popped.apply_and_prune(|allowed| allowed.intersect_with_weight(weight));
                        if pruned.is_empty() {
                            if let Some(metrics) = metrics.as_deref_mut() {
                                metrics.transitions_pruned_empty += 1;
                            }
                            return;
                        }
                        queue
                            .entry(pruned.max_depth())
                            .or_default()
                            .entry(*target)
                            .and_modify(|existing| *existing = existing.merge(&pruned))
                            .or_insert(pruned);
                        if let Some(metrics) = metrics.as_deref_mut() {
                            metrics.transitions_enqueued += 1;
                            if let Some(enqueued) = queue
                                .get(&queue.keys().next_back().copied().unwrap_or_default())
                                .and_then(|bucket| bucket.get(target))
                            {
                                update_weighted_gss_metrics(metrics, enqueued);
                            }
                            metrics.max_queue_items = metrics.max_queue_items.max(queue_item_count(&queue));
                        }
                    };

                    advance(crate::compiler::glr::labels::encode_positive_label(parser_state), &gss, &mut metrics);
                    advance(crate::compiler::glr::labels::DEFAULT_LABEL, &gss, &mut metrics);
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

    fn seed_weight(
        &self,
        internal_tsid: u32,
        gss: &crate::compiler::glr::parser::ParserGSS,
    ) -> WeightedParserGSS {
        gss.apply_and_prune(|terminals_disallowed| {
            let mut allowed = self.constraint.internal_token_universe();
            if terminals_disallowed.is_empty()
                || terminals_disallowed.values().all(|disallowed| disallowed.is_empty())
            {
                if allowed.is_empty() {
                    return None;
                }
                return Some(
                    RuntimeWeight::from_token_set_for_tsid(internal_tsid, allowed),
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
                    RuntimeWeight::from_token_set_for_tsid(internal_tsid, allowed),
                )
            }
        })
    }

}
