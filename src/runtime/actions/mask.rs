use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;
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

    fn num_ranges(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::Multi(weight) => weight.num_ranges(),
        }
    }

    fn single_compact_entry_parts(&self) -> Option<(u32, u32, Arc<RangeSetBlaze<u32>>)> {
        match self {
            Self::Single { start, end, tokens } => Some((*start, *end, Arc::clone(tokens))),
            Self::Multi(weight) => weight.single_compact_entry_parts(),
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

        #[cfg(feature = "profile-mask")]
        let t_start = std::time::Instant::now();
        #[cfg(feature = "profile-mask")]
        #[allow(unused_assignments)]
        let mut t_seed = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut t_reduce = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut t_intersect_final = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut t_or_buf = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut t_advance = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut t_advance_isolate_pop = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut t_advance_intersection = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut t_advance_queue = std::time::Duration::ZERO;
        #[cfg(feature = "profile-mask")]
        let mut n_or_buf = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut n_advance = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut n_iters = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut final_reduced_ranges_total = 0usize;
        #[cfg(feature = "profile-mask")]
        let mut final_weight_ranges_total = 0usize;
        #[cfg(feature = "profile-mask")]
        let mut final_single_side_hits = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut advance_weight_ranges_total = 0usize;
        #[cfg(feature = "profile-mask")]
        let mut advance_single_weight_hits = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut advance_allowed_ranges_total = 0usize;
        #[cfg(feature = "profile-mask")]
        let mut advance_allowed_single_hits = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut advance_allowed_count = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut advance_result_ranges_total = 0usize;
        #[cfg(feature = "profile-mask")]
        let mut advance_result_single_hits = 0u32;
        #[cfg(feature = "profile-mask")]
        let mut advance_result_count = 0u32;

        let mut queue = std::collections::BTreeMap::<
            isize,
            std::collections::BTreeMap<u32, WeightedParserGSS>,
        >::new();

        #[cfg(feature = "profile-mask")]
        let t0 = std::time::Instant::now();

        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }

            let internal_tsid = self.constraint.internal_tsid_for_state(tokenizer_state);
            let seeded = self.seed_weight(internal_tsid, gss);
            if seeded.is_empty() {
                continue;
            }

            queue
                .entry(seeded.max_depth())
                .or_default()
                .entry(parser_dwa.start_state)
                .and_modify(|existing| *existing = existing.merge(&seeded))
                .or_insert(seeded);
        }

        #[cfg(feature = "profile-mask")]
        { t_seed = t0.elapsed(); }

        while let Some((_, items)) = queue.pop_last() {
            for (wa_state, gss) in items {
                #[cfg(feature = "profile-mask")]
                { n_iters += 1; }

                let dwa_state = &parser_dwa.states[wa_state as usize];

                if let Some(final_weight) = &dwa_state.final_weight {
                    #[cfg(feature = "profile-mask")]
                    let t0 = std::time::Instant::now();

                    if let Some(reduced_acc) = gss.reduce_acc() {
                        #[cfg(feature = "profile-mask")]
                        {
                            t_reduce += t0.elapsed();
                            final_reduced_ranges_total += reduced_acc.num_ranges();
                            final_weight_ranges_total += final_weight.num_ranges();
                            if reduced_acc.num_ranges() == 1 || final_weight.num_ranges() == 1 {
                                final_single_side_hits += 1;
                            }
                        }

                        #[cfg(feature = "profile-mask")]
                        let t1 = std::time::Instant::now();
                        if final_weight.is_full() {
                            reduced_acc.or_to_buf(self, buf);
                        } else {
                            reduced_acc.or_intersection_to_buf(self, final_weight, buf);
                        }
                        #[cfg(feature = "profile-mask")]
                        { t_intersect_final += t1.elapsed(); }

                        #[cfg(feature = "profile-mask")]
                        { n_or_buf += 1; }
                    }
                }

                for parser_state in gss.peek() {
                    #[cfg(feature = "profile-mask")]
                    let t0 = std::time::Instant::now();

                    let mut advance = |label: i32, current: &WeightedParserGSS| {
                        let Some((target, weight)) = dwa_state.transitions.get(&label) else {
                            return;
                        };
                        #[cfg(feature = "profile-mask")]
                        {
                            advance_weight_ranges_total += weight.num_ranges();
                            if weight.num_ranges() == 1 {
                                advance_single_weight_hits += 1;
                            }
                        }
                        #[cfg(feature = "profile-mask")]
                        let t_isolate = std::time::Instant::now();
                        let isolated = current.isolate(Some(parser_state));
                        let popped = isolated.pop();
                        #[cfg(feature = "profile-mask")]
                        { t_advance_isolate_pop += t_isolate.elapsed(); }
                        if popped.is_empty() {
                            return;
                        }
                        #[cfg(feature = "profile-mask")]
                        let t_intersection = std::time::Instant::now();
                        #[cfg(feature = "profile-mask")]
                        let pruned = popped.apply_and_prune(|allowed| {
                            advance_allowed_ranges_total += allowed.num_ranges();
                            advance_allowed_count += 1;
                            if allowed.num_ranges() == 1 {
                                advance_allowed_single_hits += 1;
                            }
                            let next = allowed.intersect_with_weight(weight);
                            if let Some(ref next_weight) = next {
                                advance_result_ranges_total += next_weight.num_ranges();
                                advance_result_count += 1;
                                if next_weight.num_ranges() == 1 {
                                    advance_result_single_hits += 1;
                                }
                            }
                            next
                        });
                        #[cfg(not(feature = "profile-mask"))]
                        let pruned = popped.apply_and_prune(|allowed| allowed.intersect_with_weight(weight));
                        #[cfg(feature = "profile-mask")]
                        { t_advance_intersection += t_intersection.elapsed(); }
                        if pruned.is_empty() {
                            return;
                        }
                        #[cfg(feature = "profile-mask")]
                        let t_queue = std::time::Instant::now();
                        queue
                            .entry(pruned.max_depth())
                            .or_default()
                            .entry(*target)
                            .and_modify(|existing| *existing = existing.merge(&pruned))
                            .or_insert(pruned);
                        #[cfg(feature = "profile-mask")]
                        { t_advance_queue += t_queue.elapsed(); }
                    };

                            advance(crate::compiler::glr::labels::encode_positive_label(parser_state), &gss);
                            advance(crate::compiler::glr::labels::DEFAULT_LABEL, &gss);

                    #[cfg(feature = "profile-mask")]
                    { t_advance += t0.elapsed(); n_advance += 1; }
                }
            }
        }

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

        #[cfg(feature = "profile-mask")]
        {
            let total = t_start.elapsed();
            eprintln!(
                "[glrmask/profile][mask] total={}us seed={}us reduce={}us ifinal={}us or_buf={}us(n={}) advance={}us(n={}; isolate_pop={}us intersection={}us queue={}us) iters={} final_ranges(avg_reduced={:.1},avg_weight={:.1},single_side={}/{}) advance_weight_ranges(avg={:.1},single={}/{}) advance_allowed_ranges(avg={:.1},single={}/{}) advance_result_ranges(avg={:.1},single={}/{})",
                total.as_micros(),
                t_seed.as_micros(),
                t_reduce.as_micros(),
                t_intersect_final.as_micros(),
                t_or_buf.as_micros(),
                n_or_buf,
                t_advance.as_micros(),
                n_advance,
                t_advance_isolate_pop.as_micros(),
                t_advance_intersection.as_micros(),
                t_advance_queue.as_micros(),
                n_iters,
                if n_or_buf == 0 { 0.0 } else { final_reduced_ranges_total as f64 / n_or_buf as f64 },
                if n_or_buf == 0 { 0.0 } else { final_weight_ranges_total as f64 / n_or_buf as f64 },
                final_single_side_hits,
                n_or_buf,
                if n_advance == 0 { 0.0 } else { advance_weight_ranges_total as f64 / (n_advance as f64 * 2.0) },
                advance_single_weight_hits,
                n_advance * 2,
                if advance_allowed_count == 0 { 0.0 } else { advance_allowed_ranges_total as f64 / advance_allowed_count as f64 },
                advance_allowed_single_hits,
                advance_allowed_count,
                if advance_result_count == 0 { 0.0 } else { advance_result_ranges_total as f64 / advance_result_count as f64 },
                advance_result_single_hits,
                advance_result_count,
            );
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

            for (&tsid, disallowed_in_state) in terminals_disallowed {
                if disallowed_in_state.is_empty() {
                    continue;
                }

                let state_matches = self.constraint.possible_matches_for_state_internal(tsid);
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

    fn intersect_weight(
        &self,
        gss: &WeightedParserGSS,
        weight: &Weight,
    ) -> WeightedParserGSS {
        gss.apply_and_prune(|allowed| allowed.intersect_with_weight(weight))
    }
}
