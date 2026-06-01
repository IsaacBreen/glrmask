//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

impl<'a> ConstraintState<'a> {
    /// Commit one original vocabulary token and advance this state.
    ///
    /// This is the runtime **Commit** operation from the paper.  The token id is
    /// resolved to its byte string, the bytes are scanned by the tokenizer, every
    /// completed terminal sequence is advanced through the parser frontier, and
    /// the state is replaced by the resulting frontier.
    ///
    /// A token outside the current mask is not an API error: it advances to the
    /// empty/failing frontier, which will be observable as an all-zero mask.
    ///
    /// # Errors
    ///
    /// Returns an error if `token_id` is not present in the vocabulary at all.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut state = constraint.start();
    /// let mut mask = vec![0; constraint.mask_len()];
    /// state.fill_mask(&mut mask);
    /// state.commit_token(next_token_id)?;
    /// ```
    pub fn commit_token(
        &mut self,
        token_id: u32,
    ) -> Result<(), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let result = commit_bytes_impl(constraint, &mut self.state, bytes, &mut self.buffers);
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    /// Commit one token and return elapsed wall-clock time in nanoseconds.
    ///
    /// This is a diagnostics helper; normal generation should use
    /// [`ConstraintState::commit_token`].
    pub fn commit_token_timed_ns(&mut self, token_id: u32) -> Result<u64, String> {
        use std::time::Instant;

        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| format!("commit_token: token_id {token_id} not in vocabulary"))?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let start = Instant::now();
        let result = commit_bytes_impl(constraint, &mut self.state, bytes, &mut self.buffers);
        let total_ns = start.elapsed().as_nanos() as u64;
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result.map(|()| total_ns)
    }

    /// Commit one token and return a phase-level Commit profile.
    ///
    /// The profile fields separate scanner work, parser advance work, queueing,
    /// fast paths, template-DFA execution, and GSS maintenance.
    pub fn commit_token_profiled(&mut self, token_id: u32) -> Result<CommitProfile, String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| format!("commit_token: token_id {token_id} not in vocabulary"))?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let result = commit_bytes_impl_profiled(
            constraint,
            &mut self.state,
            bytes,
            &mut self.buffers,
            None,
        );
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    /// Commit one token and return per-parser-advance diagnostics.
    ///
    /// This is intentionally lower-level than [`ConstraintState::commit_token`].
    pub fn commit_token_per_advance(
        &mut self,
        token_id: u32,
    ) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| format!("commit_token: token_id {token_id} not in vocabulary"))?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let mut advances = Vec::new();
        let result = commit_bytes_impl_profiled(
            constraint,
            &mut self.state,
            bytes,
            &mut self.buffers,
            Some(&mut advances),
        )
        .map(|profile| (advances, final_stacks(&self.state), profile));
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    /// Commit raw bytes and advance this state.
    ///
    /// This is useful for byte-oriented tests and for integrations that do not
    /// operate through vocabulary token ids.
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let result = commit_bytes_impl(self.constraint, &mut self.state, bytes, &mut self.buffers);
        self.generation += 1;
        result
    }

    /// Commit a sequence of original vocabulary token ids in order.
    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token(token)?;
        }
        Ok(())
    }
}

