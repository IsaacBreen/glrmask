//! Read-only observations of a runtime state.
//!
//! These methods inspect the frontier without performing either Mask or Commit.
//! They are therefore safe to use as diagnostics and as tests of parser
//! ambiguity/completeness.

use crate::parser::glr::advance::stacks_finished;

use super::ConstraintState;

impl<'a> ConstraintState<'a> {
    /// Return whether the current state accepts the generated prefix as complete.
    pub fn is_complete(&self) -> bool {
        let initial_tsid = self.constraint.tokenizer.initial_state();
        let Some(stack) = self.state.get(&initial_tsid) else {
            return false;
        };
        !stack.is_empty() && stacks_finished(&self.constraint.table, stack)
    }

    /// Compatibility synonym for [`ConstraintState::is_complete`].
    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }

    /// Return the number of active top parser-stack values across tokenizer states.
    pub fn parser_root_count(&self) -> usize {
        self.state.values().map(|gss| gss.peek_values().len()).sum()
    }

    /// Count active parser-stack paths, saturating at `limit`.
    pub fn parser_path_count(&self, limit: usize) -> usize {
        self.state.values().map(|gss| gss.path_count_at_most(limit)).sum::<usize>().min(limit)
    }

    /// Return whether more than one parser-stack path is currently active.
    pub fn has_parser_ambiguity(&self) -> bool {
        self.parser_path_count(2) > 1
    }

    /// Return all flattened parser stacks for debugging.
    /// Each entry is (tokenizer_state, Vec<(stack_of_parser_states, disallowed_terminals)>).
    pub fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
        self.state.iter().map(|(&ts, gss)| {
            let stacks = gss.to_stacks();
            let formatted: Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)> = stacks.into_iter().map(|(stack, acc)| {
                let disallowed: Vec<(u32, Vec<u32>)> = acc.0.iter().map(|(&k, v)| {
                    (k, v.iter().copied().collect())
                }).collect();
                (stack, disallowed)
            }).collect();
            (ts, formatted)
        }).collect()
    }
}
