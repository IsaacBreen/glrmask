use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::glr::parser::{
    ParserGSS,
    TerminalsDisallowed,
    advance_stacks,
    stack_may_advance_on,
    stack_may_advance_on_any,
};
use crate::runtime::constraint::Constraint;
use crate::runtime::state::ConstraintState;
use rustc_hash::{FxHashMap, FxHashSet};

fn token_bytes_for_id(constraint: &Constraint, token_id: u32) -> Option<&[u8]> {
    constraint
        .token_bytes_dense
        .get(token_id as usize)
        .and_then(|bytes| bytes.as_deref())
        .or_else(|| constraint.token_bytes.get(&token_id).map(Vec::as_slice))
}

enum ActionableTerminals {
    SingleState(u32),
    Many(FxHashSet<u32>),
}

impl ActionableTerminals {
    fn from_gss(constraint: &Constraint, gss: &ParserGSS) -> Option<Self> {
        if let Some(state_id) = gss.single_top_value() {
            return Some(Self::SingleState(state_id));
        }

        let mut terminals = FxHashSet::default();
        for state_id in gss.peek_values() {
            if let Some(by_terminal) = constraint.table.action.get(state_id as usize) {
                terminals.extend(by_terminal.keys().copied());
            }
        }

        if terminals.is_empty() {
            None
        } else {
            Some(Self::Many(terminals))
        }
    }

    fn contains(&self, constraint: &Constraint, terminal: u32) -> bool {
        match self {
            Self::SingleState(state_id) => constraint.table.action(*state_id, terminal).is_some(),
            Self::Many(terminals) => terminals.contains(&terminal),
        }
    }
}

fn merge_state(states: &mut FxHashMap<u32, ParserGSS>, tokenizer_state: u32, gss: ParserGSS) {
    states
        .entry(tokenizer_state)
        .and_modify(|existing| *existing = existing.merge(&gss))
        .or_insert(gss);
}

fn merge_cloned_state(states: &mut FxHashMap<u32, ParserGSS>, tokenizer_state: u32, gss: &ParserGSS) {
    states
        .entry(tokenizer_state)
        .and_modify(|existing| *existing = existing.merge(gss))
        .or_insert_with(|| gss.clone());
}

fn commit_bytes_impl(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }

    let ignore_terminal = constraint.ignore_terminal;
    let mut initial_exec_results = FxHashMap::default();
    let mut remapped_tokenizer_states = FxHashMap::default();
    let mut accepted_terminals = FxHashMap::<u32, FxHashSet<u32>>::default();

    for (&tokenizer_state, parser_gss) in state.iter() {
        let actionable_terminals = ActionableTerminals::from_gss(constraint, parser_gss);
        let exec_result = constraint.tokenizer.execute_from_state(bytes, tokenizer_state);

        if let Some(end_state) = exec_result.end_state {
            remapped_tokenizer_states.insert(tokenizer_state, end_state);
        }

        for matched in &exec_result.matches {
            let ignored = Some(matched.id) == ignore_terminal;
            let actionable = !ignored
                && !actionable_terminals
                    .as_ref()
                    .is_some_and(|actionable| !actionable.contains(constraint, matched.id));

            if ignored || !actionable {
                continue;
            }

            accepted_terminals
                .entry(tokenizer_state)
                .or_default()
                .insert(matched.id);
        }

        initial_exec_results.insert(tokenizer_state, exec_result);
    }

    for parser_state in state.values_mut() {
        *parser_state = parser_state.apply_and_prune_no_promote(
            |terminals_disallowed: &TerminalsDisallowed| {
                for (state_id, matched_terminals) in &accepted_terminals {
                    if let Some(disallowed) = terminals_disallowed.get(state_id) {
                        if !matched_terminals.is_empty()
                            && matched_terminals
                                .iter()
                                .all(|terminal| disallowed.contains(terminal))
                        {
                            return None;
                        }
                    }
                }

                let mut remapped = BTreeMap::new();
                for (old_state, new_state) in &remapped_tokenizer_states {
                    if let Some(disallowed) = terminals_disallowed.get(old_state) {
                        remapped
                            .entry(*new_state)
                            .or_insert_with(BTreeSet::new)
                            .extend(disallowed.iter().copied());
                    }
                }
                Some(remapped)
            },
        );
    }

    state.retain(|_, parser_state| !parser_state.is_empty());

    let mut pending_state = FxHashMap::<u32, ParserGSS>::default();
    let mut advance_result_cache = FxHashMap::<(usize, u32), ParserGSS>::default();
    let mut processing_queue: Vec<FxHashMap<u32, ParserGSS>> =
        (0..=bytes.len()).map(|_| FxHashMap::default()).collect();
    processing_queue[0] = std::mem::take(state).into_iter().collect();

    let mut offset = 0usize;
    while offset < processing_queue.len() {
        if processing_queue[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut processing_queue[offset]);
        for (tokenizer_state, gss_at_offset) in states_to_process {
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            let exec_result = if offset == 0 {
                initial_exec_results.remove(&tokenizer_state).unwrap_or_else(|| {
                    constraint
                        .tokenizer
                        .execute_from_state(&bytes[offset..], tokenizer_state)
                })
            } else {
                constraint
                    .tokenizer
                    .execute_from_state(&bytes[offset..], tokenizer_state)
            };

            let mut seen_matches = FxHashSet::default();
            let mut terminal_result_cache = FxHashMap::<u32, ParserGSS>::default();

            for matched in &exec_result.matches {
                let new_offset = offset + matched.width;
                let ignored = Some(matched.id) == ignore_terminal;
                let actionable = !ignored
                    && !actionable_terminals
                        .as_ref()
                        .is_some_and(|actionable| !actionable.contains(constraint, matched.id));

                if !ignored && !actionable {
                    continue;
                }
                if !seen_matches.insert((matched.width, matched.id)) {
                    continue;
                }

                if ignored {
                    let next_tsid = constraint.tokenizer.initial_state();
                    if new_offset == bytes.len() {
                        merge_cloned_state(&mut pending_state, next_tsid, &gss_at_offset);
                    } else {
                        merge_cloned_state(&mut processing_queue[new_offset], next_tsid, &gss_at_offset);
                    }
                    continue;
                }

                let gss = if let Some(cached) = terminal_result_cache.get(&matched.id) {
                    cached.clone()
                } else {
                    let advance_cache_key = (gss_at_offset.ptr_key(), matched.id);
                    let mut gss = if let Some(cached) = advance_result_cache.get(&advance_cache_key)
                    {
                        cached.clone()
                    } else {
                        if !stack_may_advance_on(&constraint.table, &gss_at_offset, matched.id) {
                            let empty = ParserGSS::empty();
                            advance_result_cache.insert(advance_cache_key, empty.clone());
                            terminal_result_cache.insert(matched.id, empty);
                            continue;
                        }

                        let gss = advance_stacks(&constraint.table, &gss_at_offset, matched.id);
                        advance_result_cache.insert(advance_cache_key, gss.clone());
                        gss
                    };

                    if !gss.is_empty() {
                        if let Some(end_state) = exec_result.end_state {
                            if constraint
                                .tokenizer
                                .dfa
                                .possible_future_group_ids(end_state)
                                .contains(matched.id as usize)
                            {
                                gss = gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
                                    let mut updated = terminals_disallowed.clone();
                                    updated.entry(end_state).or_default().insert(matched.id);
                                    updated
                                });
                            }
                        }
                    }

                    terminal_result_cache.insert(matched.id, gss.clone());
                    gss
                };

                if gss.is_empty() {
                    continue;
                }

                let next_tsid = constraint.tokenizer.initial_state();
                if new_offset == bytes.len() {
                    merge_state(&mut pending_state, next_tsid, gss);
                } else {
                    merge_state(&mut processing_queue[new_offset], next_tsid, gss);
                }
            }

            if let Some(end_state) = exec_result.end_state {
                let future_terminals = constraint.tokenizer.possible_future_terminals(end_state);
                if !stack_may_advance_on_any(&constraint.table, &gss_at_offset, future_terminals)
                {
                    continue;
                }

                merge_state(&mut pending_state, end_state, gss_at_offset);
            }
        }
    }

    let mut new_state: BTreeMap<u32, ParserGSS> = pending_state.into_iter().collect();
    for parser_state in new_state.values_mut() {
        *parser_state = parser_state.fuse(Some(1));
    }
    new_state.retain(|_, parser_state| !parser_state.is_empty());

    *state = new_state;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    Ok(())
}

impl<'a> ConstraintState<'a> {
    /// Commit a sampled token, advancing the constraint state.
    ///
    /// `token_id` must be a token that exists in the vocabulary the constraint
    /// was built with.  Committing a token that is grammatically invalid (not
    /// in the current mask) drives the constraint into a fail state — this is
    /// normal and observable via an all-zero mask.
    ///
    /// # Errors
    ///
    /// Returns an error if `token_id` is not present in the vocabulary at all.
    pub fn commit_token(
        &mut self,
        token_id: u32,
    ) -> Result<(), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        commit_bytes_impl(constraint, &mut self.state, bytes)
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        commit_bytes_impl(self.constraint, &mut self.state, bytes)
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token(token)?;
        }
        Ok(())
    }
}
