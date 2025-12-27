// Brute-force grammar constraint verifier
//
// This module provides a slow but correct reference implementation
// for validating token masks. It works by:
// 1. For each token in the vocabulary
// 2. Clone the current state
// 3. Try committing the token's bytes
// 4. Check if the resulting state is valid
//
// IMPORTANT: This implementation MUST be independent of GrammarConstraintState
// to serve as a true validator. We only reference the read-only data structures
// (tokenizer, parser tables) but implement our own state management.

use std::collections::BTreeMap;

use crate::constraint::GrammarConstraint;
use crate::glr::parser::{GLRParser, GLRParserState, ParserGSS};
use crate::glr::table::TerminalID;
use crate::tokenizer::{TokenizerStateID, LLMTokenID};

/// A brute-force grammar constraint state that provides guaranteed-correct
/// mask computation by trying every token in the vocabulary.
/// 
/// This is an INDEPENDENT implementation that doesn't use GrammarConstraintState's
/// commit_bytes. It directly uses the tokenizer and parser for state transitions.
#[derive(Clone)]
pub struct BruteforceConstraintState<'a> {
    pub parent: &'a GrammarConstraint,
    /// Map from tokenizer state -> GLR parser state
    /// This is our own independent tracking, not borrowed from GrammarConstraintState
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

    /// Check if the current state represents a valid prefix.
    /// A prefix is valid if there exists some continuation that leads to acceptance.
    pub fn is_valid_prefix(&self) -> bool {
        if self.state.is_empty() {
            return false;
        }
        
        // If we're at the initial tokenizer state with a non-empty parser state,
        // we can potentially continue
        if self.state.contains_key(&self.parent.tokenizer.initial_state_id()) {
            // Check if the parser state is valid (not in error)
            if let Some(parser_state) = self.state.get(&self.parent.tokenizer.initial_state_id()) {
                if parser_state.is_ok() {
                    return true;
                }
            }
        }
        
        // Check if any active state can reach a valid completion
        for (tid, glr_state) in self.state.iter() {
            // Get all terminals that could match from this tokenizer state
            for term_id in self
                .parent
                .tokenizer
                .tokens_accessible_from_state(TokenizerStateID(tid.0))
            {
                let mut test_state = glr_state.clone();
                test_state.step(term_id);
                if test_state.is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Commit bytes to advance the state.
    /// 
    /// This is an INDEPENDENT implementation that doesn't use GrammarConstraintState.
    /// We directly run the tokenizer and parser ourselves.
    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) {
        if llm_token_bytes.is_empty() {
            return;
        }

        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();
        
        // Process from each current state
        // We use a queue to handle multi-byte tokens where we process part of the input
        // and need to continue from the new tokenizer state
        let mut processing_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, ParserGSS>> = BTreeMap::new();
        
        // Initialize queue with current states at offset 0
        let initial_states: BTreeMap<_, _> = self.state.iter()
            .map(|(sid, s)| (*sid, s.stack.clone()))
            .collect();
        processing_queue.insert(0, initial_states);
        
        while let Some((offset, states_to_process)) = processing_queue.pop_first() {
            for (tokenizer_state_id, gss) in states_to_process {
                // Run the tokenizer on the remaining bytes
                let remaining_bytes = &llm_token_bytes[offset..];
                let exec_result = self.parent.tokenizer.execute_from_state(remaining_bytes, tokenizer_state_id);
                
                // For each matched terminal, apply the parser action
                for match_info in &exec_result.matches {
                    let terminal_id = TerminalID(match_info.id);
                    
                    // Apply parser step
                    let new_gss = self.parent.parser.process_token_gss(&gss, terminal_id);
                    
                    if !new_gss.is_empty() {
                        let new_offset = offset + match_info.width;
                        let next_tsid = self.parent.tokenizer.initial_state_id();
                        
                        if new_offset == llm_token_bytes.len() {
                            // Finished processing all bytes
                            new_overall_state.entry(next_tsid)
                                .and_modify(|s| s.stack = s.stack.merge(&new_gss))
                                .or_insert_with(|| GLRParserState { 
                                    parser: &self.parent.parser, 
                                    stack: new_gss.clone() 
                                });
                        } else {
                            // More bytes to process - add to queue
                            processing_queue.entry(new_offset)
                                .or_default()
                                .entry(next_tsid)
                                .and_modify(|s| *s = s.merge(&new_gss))
                                .or_insert(new_gss);
                        }
                    }
                }
                
                // Also track partial matches (tokenizer ended in non-initial state)
                if let Some(end_state_id) = exec_result.end_state {
                    let final_tsid = TokenizerStateID(end_state_id);
                    new_overall_state.entry(final_tsid)
                        .and_modify(|s| s.stack = s.stack.merge(&gss))
                        .or_insert_with(|| GLRParserState {
                            parser: &self.parent.parser,
                            stack: gss.clone()
                        });
                }
            }
        }
        
        // Filter out invalid states
        new_overall_state.retain(|_, s| s.is_ok());
        self.state = new_overall_state;
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
