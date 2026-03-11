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
                continue;
            }
            queue
                .entry(seeded.max_depth())
                .or_default()
                .entry(parser_dwa.start_state)
                .and_modify(|existing| *existing = existing.merge(&seeded))
                .or_insert(seeded);
        }

        // Process DWA states depth-first.
        while let Some((_, items)) = queue.pop_last() {
            for (wa_state, gss) in items {
                let dwa_state = &parser_dwa.states[wa_state as usize];

                // Final weight → OR allowed tokens into buf.
                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(reduced_acc) = gss.reduce_acc() {
                        if final_weight.is_full() {
                            reduced_acc.or_to_buf(self, buf);
                        } else if let RuntimeWeight::Single { start, end, tokens } = &reduced_acc {
                            self.constraint
                                .or_single_weight_intersection_to_buf(*start, *end, tokens, final_weight, buf);
                        } else {
                            reduced_acc.or_intersection_to_buf(self, final_weight, buf);
                        }
                    }
                }

                // Advance through DWA transitions for each parser state.
                for parser_state in gss.peek() {
                    let mut advance = |label: i32, current: &WeightedParserGSS| {
                        let Some((target, weight)) = dwa_state.transitions.get(&label) else {
                            return;
                        };
                        let isolated = current.isolate(Some(parser_state));
                        let popped = isolated.pop();
                        if popped.is_empty() {
                            return;
                        }
                        let pruned = popped.apply_and_prune(|allowed| allowed.intersect_with_weight(weight));
                        if pruned.is_empty() {
                            return;
                        }
                        queue
                            .entry(pruned.max_depth())
                            .or_default()
                            .entry(*target)
                            .and_modify(|existing| *existing = existing.merge(&pruned))
                            .or_insert(pruned);
                    };

                    advance(crate::compiler::glr::labels::encode_positive_label(parser_state), &gss);
                    advance(crate::compiler::glr::labels::DEFAULT_LABEL, &gss);
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

}
