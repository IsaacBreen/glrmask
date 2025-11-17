use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use profiler_macro::time_it;
use range_set_blaze::RangeSetBlaze;
use crate::constraint::{GrammarConstraintState, LLMTokenBV, StateIDBV, TerminalAllowanceCheckMode, TerminalBV};
use crate::datastructures::gss_leveled_adapter::{disallow_terminals_and_prune_arc, fuse_predecessors_recursive, gather_gss_stats, map_allowed_terminals_tokenizer_states, prune_disallowed_terminals, prune_llm_tokens_by_disallowed_terminals, reset_llm_tokens, Acc};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::datastructures::leveled_gss::LeveledGSS;
use crate::datastructures::trie::Trie;
use crate::glr::parser::{GLRParser, GLRParserState};
use crate::glr::table::{StateID, TerminalID};
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::precompute4::weighted_automata::common::StateID as WAStateID;

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask4(&self) -> LLMTokenBV {
        let final_mask_internal = RefCell::new(HybridBitset::zeros());
        if self.state.is_empty() {
            return self
                .parent
                .internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let mut queue: BTreeMap<usize, BTreeMap<WAStateID, LeveledGSS<_, _>>> =
            BTreeMap::new();
        let dwa = &self.parent.precomputed4;
        let dwa_start_state = &dwa.states[dwa.body.start_state];

        // 1. Seed initial states
        for (&tokenizer_state_id, glr_state) in &self.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            let mut glr_state = glr_state.clone();
            prune_llm_tokens_by_disallowed_terminals(
                &mut glr_state.active_state.stack,
                &self.parent.possible_matches,
                &mut HashMap::new(),
            );

            if !glr_state.is_ok() {
                continue;
            }

            if let Some((target_wa_state_id, weight)) =
                dwa_start_state.get_transition(tokenizer_state_id.0 as i16)
            {
                let f = |acc: &Acc| {
                    let new_rsb = acc.llm_tokens_union.inner.as_ref() & &weight.rsb;
                    if new_rsb.is_empty() { None } else { Some(new_rsb) }
                };
                let gss = glr_state.active_state.stack.inner.apply_and_prune(f);

                if !gss.is_empty() {
                    queue
                        .entry(gss.max_depth() as usize)
                        .or_default()
                        .entry(target_wa_state_id)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert(gss);
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
                    if let Some((target_wa_state_id, trans_weight)) =
                        dwa_state.get_transition(parser_state_id)
                    {
                        let isolated_gss = gss.isolate(Some(peeked_edge));
                        let popped_gss = isolated_gss.pop();

                        if popped_gss.is_empty() {
                            continue;
                        }

                        let f = |rsb: &RangeSetBlaze<usize>| {
                            let new_rsb = rsb & &trans_weight.rsb;
                            if new_rsb.is_empty() { None } else { Some(new_rsb) }
                        };
                        let final_gss = popped_gss.apply_and_prune(f);

                        if !final_gss.is_empty() {
                            queue
                                .entry(final_gss.max_depth() as usize)
                                .or_default()
                                .entry(target_wa_state_id)
                                .and_modify(|existing| *existing = existing.merge(&final_gss))
                                .or_insert(final_gss);
                        }
                    }

                    if let Some((target_wa_state_id, trans_weight)) = dwa_state
                        .get_transition(crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL)
                    {
                        let isolated_gss = gss.isolate(Some(peeked_edge));
                        let popped_gss = isolated_gss.pop();

                        if popped_gss.is_empty() {
                            continue;
                        }

                        let f = |rsb: &RangeSetBlaze<usize>| {
                            let new_rsb = rsb & &trans_weight.rsb;
                            if new_rsb.is_empty() { None } else { Some(new_rsb) }
                        };
                        let final_gss = popped_gss.apply_and_prune(f);

                        if !final_gss.is_empty() {
                            queue
                                .entry(final_gss.max_depth() as usize)
                                .or_default()
                                .entry(target_wa_state_id)
                                .and_modify(|existing| *existing = existing.merge(&final_gss))
                                .or_insert(final_gss);
                        }
                    }
                }
            }
        }

        let final_mask_mapped = self
            .parent
            .internal_bv_to_original(&final_mask_internal.into_inner());

        final_mask_mapped
    }

    #[time_it]
    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) {
        if llm_token_bytes.is_empty() {
            return;
        }

        crate::debug!(
            3,
            "Committing bytes: {:?}",
            String::from_utf8_lossy(llm_token_bytes)
        );

        self.transform_gss_stacks(|stack, memo| reset_llm_tokens(stack, memo));

        let (state_map, terminals_map) = self.compute_commit_maps(llm_token_bytes);

        let gss_stats_before_pruning = gather_gss_stats(
            &self
                .state
                .values()
                .map(|s| s.active_state.stack.as_ref())
                .collect::<Vec<_>>(),
        );
        crate::debug!(5, "Terminals map: {:?}", terminals_map);
        self.transform_gss_stacks(|stack, memo| {
            prune_disallowed_terminals(stack, &terminals_map, memo)
        });
        let gss_stats_after_pruning = gather_gss_stats(
            &self
                .state
                .values()
                .map(|s| s.active_state.stack.as_ref())
                .collect::<Vec<_>>(),
        );
        crate::debug!(
            4,
            "GSS stats before pruning disallowed terminals: {:#?}",
            gss_stats_before_pruning
        );
        if gss_stats_after_pruning != gss_stats_before_pruning {
            crate::debug!(
                4,
                "GSS stats after pruning disallowed terminals: {:#?}",
                gss_stats_after_pruning
            );
            crate::debug!(
                4,
                "GSS stats changed after pruning disallowed terminals."
            );
        } else {
            crate::debug!(
                4,
                "GSS stats did not change after pruning disallowed terminals."
            );
        }

        self.transform_gss_stacks(|stack, memo| {
            map_allowed_terminals_tokenizer_states(stack, &state_map, memo)
        });

        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> =
            BTreeMap::new();

        let mut processing_queue: BTreeMap<
            usize,
            BTreeMap<TokenizerStateID, GLRParserState<'a>>,
        > = BTreeMap::new();
        processing_queue.insert(0, std::mem::take(&mut self.state));

        while let Some((offset, states_to_process)) = processing_queue.pop_first()
        {
            crate::debug!(
                3,
                "Processing offset {} with states {:?}.",
                offset,
                states_to_process.keys().map(|k| k.0).collect::<Vec<_>>()
            );
            for (tokenizer_s_id_at_offset, glr_s_at_offset) in states_to_process
            {
                assert!(offset < llm_token_bytes.len());

                let exec_result = self.parent.tokenizer.execute_from_state(
                    &llm_token_bytes[offset..],
                    tokenizer_s_id_at_offset,
                );

                for match_info in &exec_result.matches {
                    let mut cloned_glr_s = glr_s_at_offset.clone();
                    let terminal_id = TerminalID(match_info.id);

                    if let Some(dummy_id) =
                        self.parent.original_to_dummy_map.get(&terminal_id)
                    {
                        crate::debug!(5, "Processing dummy token {:?}", dummy_id);
                        cloned_glr_s.process_token(*dummy_id);
                    }

                    crate::debug!(5, "Processing terminal token {:?}", terminal_id);
                    cloned_glr_s.process_token(terminal_id);

                    if cloned_glr_s.is_ok() {
                        let new_offset = offset + match_info.width;
                        let next_tokenizer_id_for_segment =
                            self.parent.tokenizer.initial_state_id();

                        if let Some(end_state_id) = exec_result.end_state {
                            let terminals_accessible_from_end_state = self
                                .parent
                                .tokenizer
                                .tokens_accessible_from_state(
                                    TokenizerStateID(end_state_id),
                                );
                            if terminals_accessible_from_end_state
                                .contains(&TerminalID(match_info.id))
                            {
                                let mut disallowed_terminals =
                                    HybridL2Bitset::new();
                                let mut disallowed_terminals_for_end_state =
                                    TerminalBV::zeros();
                                disallowed_terminals_for_end_state
                                    .insert(match_info.id);
                                disallowed_terminals.insert_l2_bitset(
                                    end_state_id,
                                    disallowed_terminals_for_end_state,
                                );
                                disallow_terminals_and_prune_arc(
                                    &mut cloned_glr_s.active_state.stack,
                                    &disallowed_terminals,
                                    &mut HashMap::new(),
                                );
                            }
                        }

                        if new_offset == llm_token_bytes.len() {
                            new_overall_state
                                .entry(next_tokenizer_id_for_segment)
                                .and_modify(|existing| {
                                    existing.merge_with(cloned_glr_s.clone())
                                })
                                .or_insert(cloned_glr_s);
                        } else {
                            processing_queue
                                .entry(new_offset)
                                .or_default()
                                .entry(next_tokenizer_id_for_segment)
                                .and_modify(|existing| {
                                    existing.merge_with(cloned_glr_s.clone())
                                })
                                .or_insert(cloned_glr_s);
                        }
                    }
                }

                if let Some(final_tokenizer_s_id_for_llm_token_segment) =
                    exec_result.end_state
                {
                    let final_tokenizer_state =
                        TokenizerStateID(final_tokenizer_s_id_for_llm_token_segment);
                    new_overall_state
                        .entry(final_tokenizer_state)
                        .and_modify(|existing| {
                            existing.merge_with(glr_s_at_offset.clone())
                        })
                        .or_insert(glr_s_at_offset.clone());
                }
            }
        }

        self.state = new_overall_state.clone();

        self.transform_gss_stacks(|stack, memo| reset_llm_tokens(stack, memo));
        self.map_gss_stacks(|stack, memo| {
            fuse_predecessors_recursive(stack, 1, memo)
        });
        self.state
            .retain(|_, glr_parser_state| glr_parser_state.is_ok());

        match self.parent.post_commit_allow_check_mode {
            TerminalAllowanceCheckMode::None => {}
            TerminalAllowanceCheckMode::ImmediateSets => {
                self.state.retain(|tokenizer_state_id, glr_state| {
                    let accessible = self
                        .parent
                        .tokenizer
                        .tokens_accessible_from_state(*tokenizer_state_id);
                    if accessible.len() >= self.parent.tokenizer.num_groups() {
                        return true;
                    }

                    let mut union = glr_state.immediate_shift_terminals();
                    union.extend(glr_state.immediate_reduce_terminals());
                    !union.is_disjoint(&accessible)
                });
            }
            TerminalAllowanceCheckMode::ImmediateProbe => {
                self.state.retain(|tokenizer_state_id, glr_state| {
                    let accessible = self
                        .parent
                        .tokenizer
                        .tokens_accessible_from_state(*tokenizer_state_id);
                    if accessible.len() >= self.parent.tokenizer.num_groups() {
                        return true;
                    }
                    for tid in &accessible {
                        if glr_state
                            .has_immediate_action_for_terminal(*tid)
                            .unwrap_or(false)
                        {
                            return true;
                        }
                    }
                    false
                });
            }
            TerminalAllowanceCheckMode::StepProbe => {
                self.state.retain(|tokenizer_state_id, glr_state| {
                    let accessible = self
                        .parent
                        .tokenizer
                        .tokens_accessible_from_state(*tokenizer_state_id);
                    if accessible.len() >= self.parent.tokenizer.num_groups() {
                        return true;
                    }
                    for tid in &accessible {
                        let mut glr_state = glr_state.clone();
                        if let Some(dummy_id) =
                            self.parent.original_to_dummy_map.get(tid)
                        {
                            crate::debug!(5, "Processing dummy token {:?}", dummy_id);
                            glr_state.process_token(*dummy_id);
                        }

                        if glr_state.allows_terminal(*tid) {
                            return true;
                        }
                    }
                    false
                });
            }
        }

        crate::debug!(
            4,
            "Active tokenizer states after committing text (bytes {:?}): {:?}",
            llm_token_bytes,
            self.state.keys().map(|k| k.0).collect::<Vec<_>>()
        );
    }
}