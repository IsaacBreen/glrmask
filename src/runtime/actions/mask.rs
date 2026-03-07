


#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::runtime::state::ConstraintState;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use range_set_blaze::RangeSetBlaze;

#[derive(Clone, Debug)]
struct AllowedTokens(RangeSetBlaze<u32>);

impl PartialEq for AllowedTokens {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for AllowedTokens {}

impl std::hash::Hash for AllowedTokens {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for range in self.0.ranges() {
            range.start().hash(state);
            range.end().hash(state);
        }
    }
}

impl Merge for AllowedTokens {
    fn merge(&self, other: &Self) -> Self {
        Self(&self.0 | &other.0)
    }
}

type WeightedParserGSS = LeveledGSS<u32, AllowedTokens>;

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
            std::collections::BTreeMap<(u32, u32), WeightedParserGSS>,
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

            let seeded = self.seed_by_weight(tokenizer_state, gss, seed_weight);
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
                            reduced_acc.0.clone()
                        } else {
                            &reduced_acc.0 & &final_weight.tokens_for_tsid(tokenizer_state)
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
                    let mut advance = |label: i32, current: &WeightedParserGSS| {
                        let Some((target, weight)) = dwa_state.transitions.get(&label) else {
                            return;
                        };
                        let isolated = current.isolate(Some(parser_state));
                        let popped = isolated.pop();
                        if popped.is_empty() {
                            return;
                        }
                        let pruned = self.intersect_weight_tokens(tokenizer_state, &popped, weight);
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

    fn seed_by_weight(
        &self,
        tokenizer_state: u32,
        gss: &crate::compiler::glr::parser::ParserGSS,
        weight: &crate::ds::weight::Weight,
    ) -> WeightedParserGSS {
        let tokens = self.tokens_for_weight(tokenizer_state, weight);
        if tokens.is_empty() {
            return WeightedParserGSS::empty();
        }
        gss.apply_and_prune(|terminals_disallowed| {
            let allowed = self.filter_weight_tokens(
                tokenizer_state,
                tokens.clone(),
                terminals_disallowed,
            );
            if allowed.is_empty() {
                None
            } else {
                Some(AllowedTokens(allowed))
            }
        })
    }

    fn intersect_weight_tokens(
        &self,
        tokenizer_state: u32,
        gss: &WeightedParserGSS,
        weight: &crate::ds::weight::Weight,
    ) -> WeightedParserGSS {
        let tokens = self.tokens_for_weight(tokenizer_state, weight);
        if tokens.is_empty() {
            return WeightedParserGSS::empty();
        }
        gss.apply_and_prune(|allowed| {
            let next = &allowed.0 & &tokens;
            if next.is_empty() {
                None
            } else {
                Some(AllowedTokens(next))
            }
        })
    }

    fn tokens_for_weight(
        &self,
        tokenizer_state: u32,
        weight: &crate::ds::weight::Weight,
    ) -> RangeSetBlaze<u32> {
        if weight.is_full() {
            self.all_tokens_for_state(tokenizer_state)
        } else {
            weight.tokens_for_tsid(tokenizer_state)
        }
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
        if terminals_disallowed.is_empty()
            || terminals_disallowed.values().all(|disallowed| disallowed.is_empty())
        {
            return allowed;
        }

        for (&tsid, disallowed) in terminals_disallowed {
            if disallowed.is_empty() {
                continue;
            }
            let possible_matches = self.constraint.possible_matches_for_state(tsid);
            for terminal in disallowed {
                if let Some(token_ids) = possible_matches.get(terminal) {
                    allowed = allowed - token_ids.clone();
                }
            }
        }
        allowed
    }
}
