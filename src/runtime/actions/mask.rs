use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::ds::weight::Weight;

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
                            self.constraint.or_weight_to_buf(&reduced_acc, buf);
                        } else if let Some((start, end, tokens)) = reduced_acc.single_compact_entry_parts() {
                            self.constraint
                                .or_single_weight_intersection_to_buf(start, end, &tokens, final_weight, buf);
                        } else {
                            let allowed = reduced_acc.intersection(final_weight);
                            self.constraint.or_weight_to_buf(&allowed, buf);
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
                "[glrmask/profile][mask] total={}us seed={}us reduce={}us ifinal={}us or_buf={}us(n={}) advance={}us(n={}) iters={} final_ranges(avg_reduced={:.1},avg_weight={:.1},single_side={}/{}) advance_weight_ranges(avg={:.1},single={}/{})",
                total.as_micros(),
                t_seed.as_micros(),
                t_reduce.as_micros(),
                t_intersect_final.as_micros(),
                t_or_buf.as_micros(),
                n_or_buf,
                t_advance.as_micros(),
                n_advance,
                n_iters,
                if n_or_buf == 0 { 0.0 } else { final_reduced_ranges_total as f64 / n_or_buf as f64 },
                if n_or_buf == 0 { 0.0 } else { final_weight_ranges_total as f64 / n_or_buf as f64 },
                final_single_side_hits,
                n_or_buf,
                if n_advance == 0 { 0.0 } else { advance_weight_ranges_total as f64 / (n_advance as f64 * 2.0) },
                advance_single_weight_hits,
                n_advance * 2,
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
}
