use crate::constraint::{GrammarConstraintState, TerminalAllowanceCheckMode};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::leveled_gss::LeveledGSS;
use crate::glr::parser::{GLRParserState, ParseStateEdgeContent};
use crate::glr::table::TerminalID;
use crate::precompute4::weighted_automata::common::StateID as WAStateID;
use crate::tokenizer::TokenizerStateID;
use profiler_macro::time_it;
use range_set_blaze::RangeSetBlaze;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::BitOrAssign;
use crate::datastructures::gss_acc::Acc;

type ParserGSS = LeveledGSS<ParseStateEdgeContent, Acc>;

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask4(&self) -> HybridBitset {
        let final_mask_internal = RefCell::new(HybridBitset::zeros());
        if self.state.is_empty() {
            return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let mut queue: BTreeMap<isize, BTreeMap<WAStateID, LeveledGSS<ParseStateEdgeContent, RangeSetBlaze<usize>>>> = BTreeMap::new();
        let dwa = &self.parent.precomputed4;
        let dwa_start_state = &dwa.states[dwa.body.start_state];

        // 1. Seed initial states
        for (&tokenizer_state_id, glr_state) in &self.state {
            if glr_state.stack.is_empty() {
                continue;
            }

            // Prune GSS based on disallowed terminals before starting.
            let mut gss = glr_state.stack.clone();
            let possible_matches = &self.parent.possible_matches;
            gss = gss.apply_and_prune(|acc| {
                if acc.terminals_union.is_empty() {
                    return Some(acc.clone());
                }
                let mut forbidden_llm_tokens = HybridBitset::zeros();
                for (&tokenizer_state_id, disallowed_in_state) in &acc.terminals_union {
                    if disallowed_in_state.is_empty() { continue; }
                    if let Some(state_matches) = possible_matches.get(&TokenizerStateID(tokenizer_state_id)) {
                        for (terminal_id, llm_tokens) in state_matches {
                            if disallowed_in_state.contains(terminal_id.0) {
                                forbidden_llm_tokens |= llm_tokens;
                            }
                        }
                    }
                }

                if forbidden_llm_tokens.is_empty() {
                    return Some(acc.clone());
                }
                let mut new_acc = acc.clone();
                new_acc.llm_tokens_union -= &forbidden_llm_tokens;
                if new_acc.llm_tokens_union.is_empty() { None } else { Some(new_acc) }
            });

            if gss.is_empty() {
                continue;
            }

            if let Some((target_wa_state_id, weight)) = dwa_start_state.get_transition(tokenizer_state_id.0 as i16) {
                let f = |acc: &Acc| {
                    let new_rsb = acc.llm_tokens_union.inner.as_ref() & &weight.rsb;
                    if new_rsb.is_empty() { None } else { Some(new_rsb) }
                };
                let weighted_gss = gss.apply_and_prune(f);

                if !weighted_gss.is_empty() {
                    queue
                        .entry(weighted_gss.max_depth())
                        .or_default()
                        .entry(target_wa_state_id)
                        .and_modify(|existing| *existing = existing.merge(&weighted_gss))
                        .or_insert(weighted_gss);
                }
            }
        }

        // 2. Main worklist loop
        while let Some((_depth, states_at_depth)) = queue.pop_last() {
            for (current_wa_state_id, mut gss) in states_at_depth {
                let dwa_state = &dwa.states[current_wa_state_id];

                // Apply state weight
                if let Some(state_weight) = &dwa_state.state_weight {
                    let f = |rsb: &RangeSetBlaze<usize>| {
                        let new_rsb = rsb & &state_weight.rsb;
                        if new_rsb.is_empty() { None } else { Some(new_rsb) }
                    };
                    gss = gss.apply_and_prune(f);
                    if gss.is_empty() {
                        continue;
                    }
                }

                // Check for final state
                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(reduced_acc) = gss.reduce_acc() {
                        let final_tokens = &reduced_acc & &final_weight.rsb;
                        if !final_tokens.is_empty() {
                            *final_mask_internal.borrow_mut() |= HybridBitset::from(final_tokens);
                        }
                    }
                }

                // Process transitions
                for peeked_edge in gss.peek() {
                    let parser_state_id = peeked_edge.state_id.0 as i16;
                    if let Some((target_wa_state_id, trans_weight)) = dwa_state.get_transition(parser_state_id) {
                        let isolated_gss = gss.isolate(Some(peeked_edge));
                        let popped_gss = isolated_gss.pop();
                        if popped_gss.is_empty() { continue; }

                        let f = |rsb: &RangeSetBlaze<usize>| {
                            let new_rsb = rsb & &trans_weight.rsb;
                            if new_rsb.is_empty() { None } else { Some(new_rsb) }
                        };
                        let final_gss = popped_gss.apply_and_prune(f);

                        if !final_gss.is_empty() {
                            queue
                                .entry(final_gss.max_depth())
                                .or_default()
                                .entry(target_wa_state_id)
                                .and_modify(|existing| *existing = existing.merge(&final_gss))
                                .or_insert(final_gss);
                        }
                    }

                    if let Some((target_wa_state_id, trans_weight)) = dwa_state.get_transition(crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL) {
                        let isolated_gss = gss.isolate(Some(peeked_edge));
                        let popped_gss = isolated_gss.pop();
                        if popped_gss.is_empty() { continue; }

                        let f = |rsb: &RangeSetBlaze<usize>| {
                            let new_rsb = rsb & &trans_weight.rsb;
                            if new_rsb.is_empty() { None } else { Some(new_rsb) }
                        };
                        let final_gss = popped_gss.apply_and_prune(f);

                        if !final_gss.is_empty() {
                            queue
                                .entry(final_gss.max_depth())
                                .or_default()
                                .entry(target_wa_state_id)
                                .and_modify(|existing| *existing = existing.merge(&final_gss))
                                .or_insert(final_gss);
                        }
                    }
                }
            }
        }

        self.parent.internal_bv_to_original(&final_mask_internal.into_inner())
    }

    #[time_it]
    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) {
        if llm_token_bytes.is_empty() {
            return;
        }
        crate::debug!(3, "Committing bytes: {:?}", String::from_utf8_lossy(llm_token_bytes));

        let (state_map, terminals_map) = self.compute_commit_maps(llm_token_bytes);

        // Prune stacks based on matched terminals and remap tokenizer state constraints.
        for glr_state in self.state.values_mut() {
            let mut gss = glr_state.stack.clone();
            // Prune based on matched terminals
            gss = gss.apply_and_prune(|acc| {
                for (sid, matched_terminals) in &terminals_map {
                    if let Some(disallowed) = acc.terminals_union.get(&sid.0) {
                        if matched_terminals.intersects(disallowed) {
                            return None;
                        }
                    }
                }
                Some(acc.clone())
            });
            // Remap tokenizer states
            gss = gss.apply(|acc| {
                let mut new_terminals_union: BTreeMap<usize, HybridBitset> = BTreeMap::new();
                for (old, new) in &state_map {
                    if let Some(bv) = acc.terminals_union.get(&old.0) {
                        new_terminals_union.entry(new.0).or_default().bitor_assign(bv);
                    }
                }
                let mut new_acc = acc.clone();
                new_acc.terminals_union = new_terminals_union;
                new_acc
            });
            glr_state.stack = gss;
        }
        self.state.retain(|_, s| !s.stack.is_empty());

        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();
        let mut processing_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, ParserGSS>> = BTreeMap::new();
        
        let initial_states: BTreeMap<_,_> = self.state.iter().map(|(sid, s)| (*sid, s.stack.clone())).collect();
        processing_queue.insert(0, initial_states);

        while let Some((offset, states_to_process)) = processing_queue.pop_first() {
            for (tokenizer_s_id_at_offset, gss_at_offset) in states_to_process {
                let exec_result = self.parent.tokenizer.execute_from_state(&llm_token_bytes[offset..], tokenizer_s_id_at_offset);

                for match_info in &exec_result.matches {
                    let mut gss = gss_at_offset.clone();
                    let terminal_id = TerminalID(match_info.id);

                    if let Some(dummy_id) = self.parent.original_to_dummy_map.get(&terminal_id) {
                        gss = self.parent.parser.process_token_gss(&gss, *dummy_id);
                    }
                    gss = self.parent.parser.process_token_gss(&gss, terminal_id);

                    if !gss.is_empty() {
                        if let Some(end_state_id) = exec_result.end_state {
                            if self.parent.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_id)).contains(&terminal_id) {
                                let terminal_to_disallow = match_info.id;
                                gss = gss.apply(|acc| {
                                    let mut na = acc.clone();
                                    na.terminals_union.entry(end_state_id).or_default().insert(terminal_to_disallow);
                                    na
                                });
                            }
                        }

                        if !gss.is_empty() {
                            let new_offset = offset + match_info.width;
                            let next_tsid = self.parent.tokenizer.initial_state_id();
                            if new_offset == llm_token_bytes.len() {
                                new_overall_state.entry(next_tsid).and_modify(|s| s.stack = s.stack.merge(&gss)).or_insert_with(|| GLRParserState { parser: &self.parent.parser, stack: gss });
                            } else {
                                processing_queue.entry(new_offset).or_default().entry(next_tsid).and_modify(|s| *s = s.merge(&gss)).or_insert(gss);
                            }
                        }
                    }
                }

                if let Some(end_state_id) = exec_result.end_state {
                    let final_tsid = TokenizerStateID(end_state_id);
                    new_overall_state.entry(final_tsid).and_modify(|s| s.stack = s.stack.merge(&gss_at_offset)).or_insert_with(|| GLRParserState { parser: &self.parent.parser, stack: gss_at_offset });
                }
            }
        }

        self.state = new_overall_state;

        for glr_state in self.state.values_mut() {
            glr_state.stack = glr_state.stack.apply(|acc| {
                let mut new_acc = acc.clone();
                new_acc.llm_tokens_union = HybridBitset::max_ones();
                new_acc
            });
            glr_state.stack = glr_state.stack.fuse(Some(1));
        }
        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        if self.parent.post_commit_allow_check_mode != TerminalAllowanceCheckMode::None {
            // The simplified parser does not currently support these checks.
            // To match Python, this logic is disabled.
        }

        crate::debug!(4, "Active tokenizer states after committing text (bytes {:?}): {:?}", llm_token_bytes, self.state.keys().map(|k| k.0).collect::<Vec<_>>());
    }
}
