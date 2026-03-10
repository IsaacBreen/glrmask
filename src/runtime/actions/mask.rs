use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;
use range_set_blaze::RangeSetBlaze;

type WeightedParserGSS = LeveledGSS<u32, Weight>;

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

        let mut queue = std::collections::BTreeMap::<
            isize,
            std::collections::BTreeMap<u32, WeightedParserGSS>,
        >::new();

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

        while let Some((_, items)) = queue.pop_last() {
            for (wa_state, gss) in items {
                let dwa_state = &parser_dwa.states[wa_state as usize];

                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(reduced_acc) = gss.reduce_acc() {
                        let allowed = if final_weight.is_full() {
                            reduced_acc.clone()
                        } else {
                            reduced_acc.intersection(final_weight)
                        };
                        for token_id in self.collapse_weight_tokens(&allowed).iter() {
                            let word = token_id as usize / 32;
                            let bit = token_id as usize % 32;
                            if let Some(slot) = buf.get_mut(word) {
                                *slot |= 1u32 << bit;
                            }
                        }
                    }
                }

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
                        let pruned = self.intersect_weight(&popped, weight);
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
                    Weight::from_token_set_for_tsid(internal_tsid, allowed),
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
                    Weight::from_token_set_for_tsid(internal_tsid, allowed),
                )
            }
        })
    }

    fn intersect_weight(
        &self,
        gss: &WeightedParserGSS,
        weight: &Weight,
    ) -> WeightedParserGSS {
        gss.apply_and_prune(|allowed| {
            let next = allowed.intersection(weight);
            if next.is_empty() {
                None
            } else {
                Some(next)
            }
        })
    }

    fn collapse_weight_tokens(
        &self,
        weight: &Weight,
    ) -> RangeSetBlaze<u32> {
        let mut all = RangeSetBlaze::new();
        for (internal_tsid, _) in self.constraint.internal_tsid_to_states.iter().enumerate() {
            let internal_token_ids = weight.tokens_for_tsid(internal_tsid as u32);
            if !internal_token_ids.is_empty() {
                let expanded = self.constraint.expand_internal_token_set(&internal_token_ids);
                all = all | expanded;
            }
        }
        all
    }
}

