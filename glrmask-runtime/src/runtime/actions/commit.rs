use std::collections::{BTreeMap, BTreeSet};
use crate::runtime::state::ConstraintState;
use crate::compiler::glr::parser::{advance_stacks, ParserGSS, TerminalsDisallowed};
use crate::runtime::constraint::Constraint;

fn commit_bytes_impl(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) {
    if bytes.is_empty() {
        return;
    }

    let ignore_terminal = constraint.ignore_terminal;
    let mut initial_exec_results = BTreeMap::new();
    let mut state_map = BTreeMap::new();
    let mut terminals_map = BTreeMap::<u32, Vec<u32>>::new();
    for &tokenizer_state in state.keys() {
        let exec = constraint.tokenizer.execute_from_state(bytes, tokenizer_state);
        if let Some(end_state) = exec.end_state {
            state_map.insert(tokenizer_state, end_state);
        }
        for matched in &exec.matches {
            if Some(matched.id) == ignore_terminal {
                continue;
            }
            // TODO: expand via mutually_greedy_group() once greedy groups
            // are wired into glrmask (see sep1 compute_commit_maps).
            terminals_map
                .entry(tokenizer_state)
                .or_default()
                .push(matched.id);
        }
        initial_exec_results.insert(tokenizer_state, exec);
    }

    for parser_state in state.values_mut() {
        let mut gss = parser_state.apply_and_prune(|terminals_disallowed: &TerminalsDisallowed| {
            for (state_id, matched_terminals) in &terminals_map {
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
            Some(terminals_disallowed.clone())
        });
        gss = gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
            let mut remapped = BTreeMap::new();
            for (old_state, new_state) in &state_map {
                if let Some(disallowed) = terminals_disallowed.get(old_state) {
                    remapped
                        .entry(*new_state)
                        .or_insert_with(BTreeSet::new)
                        .extend(disallowed.iter().copied());
                }
            }
            remapped
        });
        *parser_state = gss;
    }

    state.retain(|_, parser_state| !parser_state.is_empty());

    let mut new_overall_state: BTreeMap<u32, ParserGSS> = BTreeMap::new();
    let mut processing_queue: BTreeMap<usize, BTreeMap<u32, ParserGSS>> = BTreeMap::new();

    // Take ownership instead of cloning — state will be fully replaced below.
    processing_queue.insert(0, std::mem::take(state));

    while let Some((offset, states_to_process)) = processing_queue.pop_first() {
        for (tokenizer_state, gss_at_offset) in states_to_process {
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

            for matched in &exec_result.matches {
                let new_offset = offset + matched.width;

                if Some(matched.id) == ignore_terminal {
                    let next_tsid = constraint.tokenizer.initial_state();
                    if new_offset == bytes.len() {
                        new_overall_state
                            .entry(next_tsid)
                            .and_modify(|existing| *existing = existing.merge(&gss_at_offset))
                            .or_insert_with(|| gss_at_offset.clone());
                    } else {
                        processing_queue
                            .entry(new_offset)
                            .or_default()
                            .entry(next_tsid)
                            .and_modify(|existing| *existing = existing.merge(&gss_at_offset))
                            .or_insert_with(|| gss_at_offset.clone());
                    }
                    continue;
                }

                let mut gss = advance_stacks(&constraint.table, &gss_at_offset, matched.id);
                if gss.is_empty() {
                    continue;
                }

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

                if gss.is_empty() {
                    continue;
                }

                let next_tsid = constraint.tokenizer.initial_state();
                if new_offset == bytes.len() {
                    new_overall_state
                        .entry(next_tsid)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert(gss);
                } else {
                    processing_queue
                        .entry(new_offset)
                        .or_default()
                        .entry(next_tsid)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert(gss);
                }
            }

            if let Some(end_state) = exec_result.end_state {
                new_overall_state
                    .entry(end_state)
                    .and_modify(|existing| *existing = existing.merge(&gss_at_offset))
                    .or_insert(gss_at_offset);
            }
        }
    }

    for parser_state in new_overall_state.values_mut() {
        *parser_state = parser_state.fuse(Some(1));
    }
    new_overall_state.retain(|_, parser_state| !parser_state.is_empty());
    *state = new_overall_state;
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
        let bytes = self.constraint.token_bytes
            .get(&token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        commit_bytes_impl(self.constraint, &mut self.state, bytes);
        Ok(())
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) {
        commit_bytes_impl(self.constraint, &mut self.state, bytes);
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token(token)?;
        }
        Ok(())
    }

    pub(crate) fn process_bytes_raw(&mut self, bytes: &[u8]) {
        self.commit_bytes(bytes)
    }
}
