// Brute-force grammar constraint verifier
//
// This module provides a slow but correct reference implementation
// for validating token masks. It works by:
// 1. For each token in the vocabulary
// 2. Clone the current state
// 3. Try committing the token's bytes
// 4. Check if the resulting state is valid
//
// This is O(vocab_size * commit_time) per mask query, which is slow
// but provides a ground truth for correctness testing.

use std::collections::BTreeMap;

use crate::constraint::{GrammarConstraint, GrammarConstraintState};
use crate::glr::parser::GLRParserState;
use crate::tokenizer::{TokenizerStateID, LLMTokenID};

/// A brute-force grammar constraint state that provides guaranteed-correct
/// mask computation by trying every token in the vocabulary.
#[derive(Clone)]
pub struct BruteforceConstraintState<'a> {
    pub parent: &'a GrammarConstraint,
    pub state: BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

impl<'a> BruteforceConstraintState<'a> {
    /// Create a new brute-force constraint state from a GrammarConstraint.
    pub fn new(constraint: &'a GrammarConstraint) -> Self {
        let mut state = BTreeMap::new();
        state.insert(
            constraint.tokenizer.initial_state_id(),
            constraint.parser.init_glr_parser(None),
        );
        BruteforceConstraintState {
            parent: constraint,
            state,
        }
    }

    /// Create from an existing GrammarConstraintState (for comparison testing).
    pub fn from_constraint_state(gcs: &GrammarConstraintState<'a>) -> Self {
        BruteforceConstraintState {
            parent: gcs.parent,
            state: gcs.state.clone(),
        }
    }

    /// Check if the current state represents a valid prefix.
    /// A prefix is valid if there exists some continuation that leads to acceptance.
    pub fn is_valid_prefix(&self) -> bool {
        if self.state.is_empty() {
            return false;
        }
        
        // If we're at the initial tokenizer state, we might be able to complete
        if self.state.contains_key(&self.parent.tokenizer.initial_state_id()) {
            return true;
        }
        
        // Check if any active state can reach a valid completion
        for (tid, glr_state) in self.state.iter() {
            for gtid in self
                .parent
                .tokenizer
                .tokens_accessible_from_state(TokenizerStateID(tid.0))
            {
                let mut glr_state = glr_state.clone();
                glr_state.step(gtid);
                if glr_state.is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Commit bytes to advance the state (same as GrammarConstraintState::commit_bytes).
    /// This is a copy of the implementation from constraint_fns.rs.
    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) {
        // Create a temporary GrammarConstraintState and use its commit_bytes
        let mut temp_state = GrammarConstraintState {
            parent: self.parent,
            state: self.state.clone(),
        };
        temp_state.commit_bytes(llm_token_bytes);
        self.state = temp_state.state;
    }

    /// Check if committing the given bytes would result in a valid prefix.
    /// This clones self, commits the bytes, and checks is_valid_prefix.
    pub fn is_valid_continuation(&self, bytes: &[u8]) -> bool {
        let mut cloned = self.clone();
        cloned.commit_bytes(bytes);
        cloned.is_valid_prefix()
    }

    /// Get the mask by brute-force: for each token, check if it's a valid continuation.
    /// Returns a vector of booleans where mask[i] is true if token i is valid.
    pub fn get_mask_bruteforce(&self) -> Vec<bool> {
        let vocab_size = self.parent.vocab_trie.max_token_id + 1;
        let mut mask = vec![false; vocab_size];
        
        for token_id in 0..vocab_size {
            if let Some(token_bytes) = self.parent.vocab_trie.token_bytes(LLMTokenID(token_id)) {
                mask[token_id] = self.is_valid_continuation(token_bytes);
            }
        }
        
        mask
    }

    /// Get the count of valid tokens (useful for quick comparison).
    pub fn valid_token_count(&self) -> usize {
        self.get_mask_bruteforce().iter().filter(|&&v| v).count()
    }

    /// Get list of valid token IDs.
    pub fn valid_token_ids(&self) -> Vec<usize> {
        self.get_mask_bruteforce()
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests will be added after Python binding integration
}
