//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

impl InitialCommitScan {
    pub(super) fn collect(
        constraint: &Constraint,
        state: &BTreeMap<u32, ParserGSS>,
        bytes: &[u8],
    ) -> Self {
        let ignore_terminal = constraint.ignore_terminal;
        let mut exec_results = FxHashMap::default();
        let mut remapped_tokenizer_states = FxHashMap::default();
        let mut accepted_terminals = FxHashMap::<u32, FxHashSet<u32>>::default();

        for (&tokenizer_state, parser_gss) in state {
            let actionable_terminals = ActionableTerminals::from_gss(constraint, parser_gss);
            let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);

            if let Some(end_state) = exec_result.end_state {
                remapped_tokenizer_states.insert(tokenizer_state, end_state);
            }

            for matched in &exec_result.matches {
                if is_ignored_terminal(ignore_terminal, matched.id)
                    || !is_actionable_terminal(
                        actionable_terminals.as_ref(),
                        constraint,
                        matched.id,
                    )
                {
                    continue;
                }

                accepted_terminals
                    .entry(tokenizer_state)
                    .or_default()
                    .insert(matched.id);
            }

            exec_results.insert(tokenizer_state, exec_result);
        }

        Self {
            exec_results,
            remapped_tokenizer_states,
            accepted_terminals,
        }
    }

    pub(super) fn take_exec_result(&mut self, tokenizer_state: u32) -> Option<TokenizerExecResult> {
        self.exec_results.remove(&tokenizer_state)
    }
}

