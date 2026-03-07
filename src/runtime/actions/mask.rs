


#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;
use range_set_blaze::RangeSetBlaze;

// SEP1_MAP: this file is the glrmask split of sep1 mask generation from
// `grammars2024/src/constraint_fns.rs::{compute_internal_mask,get_mask,fill_mask_i32}`.
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
            std::collections::BTreeMap<(u32, u32), crate::compiler::glr::parser::ParserGSS>,
        >::new();

        for (&tokenizer_state, gss) in &self.state {
            if gss.is_empty() {
                continue;
            }

            let tsid_label = self.constraint.table.num_terminals as i32 + tokenizer_state as i32;
            let Some((target_state, seed_weight)) = parser_dwa.states[parser_dwa.start_state as usize]
                .transitions
                .get(&tsid_label)
            else {
                continue;
            };

            let seeded = self.prune_by_weight(tokenizer_state, gss, seed_weight);
            if seeded.is_empty() {
                continue;
            }

            queue
                .entry(seeded.max_depth())
                .or_default()
                .entry((tokenizer_state, *target_state))
                .and_modify(|existing| *existing = existing.merge(&seeded))
                .or_insert(seeded);
        }

        while let Some((_, items)) = queue.pop_last() {
            for ((tokenizer_state, wa_state), gss) in items {
                let dwa_state = &parser_dwa.states[wa_state as usize];

                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(reduced_acc) = gss.reduce_acc() {
                        let allowed = if final_weight.is_full() {
                            self.filter_weight_tokens(
                                tokenizer_state,
                                self.all_tokens_for_state(tokenizer_state),
                                &reduced_acc,
                            )
                        } else {
                            self.filter_weight_tokens(
                                tokenizer_state,
                                final_weight.tokens_for_tsid(tokenizer_state),
                                &reduced_acc,
                            )
                        };
                        for token_id in allowed.iter() {
                            let word = token_id as usize / 32;
                            let bit = token_id as usize % 32;
                            if let Some(slot) = buf.get_mut(word) {
                                *slot |= 1u32 << bit;
                            }
                        }
                    }
                }

                for parser_state in gss.peek() {
                    let mut advance = |label: i32, current: &crate::compiler::glr::parser::ParserGSS| {
                        let Some((target, weight)) = dwa_state.transitions.get(&label) else {
                            return;
                        };
                        let isolated = current.isolate(Some(parser_state));
                        let popped = isolated.pop();
                        if popped.is_empty() {
                            return;
                        }
                        let pruned = self.prune_by_weight(tokenizer_state, &popped, weight);
                        if pruned.is_empty() {
                            return;
                        }
                        queue
                            .entry(pruned.max_depth())
                            .or_default()
                            .entry((tokenizer_state, *target))
                            .and_modify(|existing| *existing = existing.merge(&pruned))
                            .or_insert(pruned);
                    };

                    advance(crate::compiler::glr::labels::encode_positive_label(parser_state), &gss);
                    advance(crate::compiler::glr::labels::DEFAULT_LABEL, &gss);
                }
            }
        }

        if self.is_finished() {
            if let Some(eos_token_id) = self.constraint.eos_token_id {
                let word = eos_token_id as usize / 32;
                let bit = eos_token_id as usize % 32;
                if let Some(slot) = buf.get_mut(word) {
                    *slot |= 1u32 << bit;
                }
            }
        }
    }

    fn prune_by_weight(
        &self,
        tokenizer_state: u32,
        gss: &crate::compiler::glr::parser::ParserGSS,
        weight: &crate::ds::weight::Weight,
    ) -> crate::compiler::glr::parser::ParserGSS {
        let tokens = weight.tokens_for_tsid(tokenizer_state);
        if tokens.is_empty() && !weight.is_full() {
            return crate::compiler::glr::parser::ParserGSS::empty();
        }
        gss.apply_and_prune(|terminals_disallowed| {
            let allowed = if weight.is_full() {
                self.filter_weight_tokens(
                    tokenizer_state,
                    self.all_tokens_for_state(tokenizer_state),
                    terminals_disallowed,
                )
            } else {
                self.filter_weight_tokens(tokenizer_state, tokens.clone(), terminals_disallowed)
            };
            if allowed.is_empty() {
                None
            } else {
                Some(terminals_disallowed.clone())
            }
        })
    }

    fn all_tokens_for_state(&self, tokenizer_state: u32) -> RangeSetBlaze<u32> {
        let mut all = RangeSetBlaze::new();
        for token_ids in self.constraint.possible_matches_for_state(tokenizer_state).values() {
            all = all | token_ids.clone();
        }
        all
    }

    fn filter_weight_tokens(
        &self,
        tokenizer_state: u32,
        tokens: RangeSetBlaze<u32>,
        terminals_disallowed: &crate::compiler::glr::parser::TerminalsDisallowed,
    ) -> RangeSetBlaze<u32> {
        let mut allowed = tokens;
        let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) else {
            return allowed;
        };
        let possible_matches = self.constraint.possible_matches_for_state(tokenizer_state);
        for terminal in disallowed {
            if let Some(token_ids) = possible_matches.get(terminal) {
                allowed = allowed - token_ids.clone();
            }
        }
        allowed
    }
}
