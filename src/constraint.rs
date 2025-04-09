//! Each node has
//!
//! 1. a map from LLM token ID to final tokeniser state ID
//! 2. a map from possible final grammar token ID to LLM token ID set
//! 3. a map from edge key to edge value and child node
//!
//! ### Get LLM token mask
//!
//! Define a mutable bitset of valid next LLM tokens.
//!
//! Start with a queue of (node, ([active parse state], active LLM token bitset)) pairs.
//!
//! Traverse all edges. Do a bitwise AND between the active LLM token bitset and the edge's LLM token gate bitset. If the result is non-empty, run the parser from active parse states. Otherwise, use an empty list of parse states. Use these two results (the AND operation result and the new parse states or empty list) as the new value for the destination node.
//!
//! At each node (in the process function), loop over each (possible final grammar token ID, LLM token ID set) pair in map 2. Run the parser on the possible final grammar token ID from the current GLR parser states. If there are any active states, or any dormant states that indicate a complete parse (successful termination), take the bitwise AND of the corresponiding LLM token ID set and the active LLM token bitset, and use this on the RHS of an in-place bitwise OR with the mutable bitset of valid next LLM tokens.
//!
//! ### Committing a token
//!
//! Start with a queue of (node, [parse state]) pairs.
//!
//! Traverse all edges for which the edge value (the LLM token gate bitset) contains the LLM token we're committing.
//!
//! At each node (in the process function) look at map 1 to get the final tokeniser state ID. If it exists, add the current GLR states to the new list of GLR states. Associate them with the tokeniser state somehow.
//!
//! If the list of currrent GLR states is empty, return false to halt at this node.
//!
//! #### Redundant grammar token parsing
//!
//! Actually this might not be a huge concern.
//!
//! I was thinking that, if we end a commit (when a node has an entry in map 1 for the LLM token we're committing) and push the same parse state (cloned) with many different tokeniser states, and if on the next call to commit or get mask those tokeniser states yield the same grammar token, then we've essentially split one GLR state into many and then run the same grammar token through it. And it could then immediately merge again. So that'd be inefficient.
//!
//! But it's unlikely to be an issue. At the end of a commit, there's no more than one final tokeniser state that we can go to.
//!
//! ### More efficient representations for maps keyed by LLM tokens
//!
//! More generally, how can we efficiently map from a large set of u32 to a small set of u32?
//!
//! We want to reduce memory usage but keep operations fast.
//!
use crate::glr::parser::{GLRParser, GLRParserState, InsertWith, ParseState, ParseStateKey};
use crate::glr::table::{StateID, TerminalID};
use crate::{precompute, debug};
use crate::precompute::{LLMTokenID, TokenID, Tokenizer};
use bitvec::prelude::*;
use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use crate::trie::TrieNode;

type LLMToken = Vec<u8>;
type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone)]
pub struct GrammarConstraint<T: Tokenizer> {
    pub(crate) tokenizer: T,
    pub(crate) parser: GLRParser,
    pub precomputed: BTreeMap<StateID, TrieNode<BitVec, TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>,
    pub(crate) max_llm_token_id: usize,
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<T: Tokenizer> {
    parent: GrammarConstraint<T>,
    states: Vec<(ParseState, BTreeSet<StateID>)>,
}

impl<T: Tokenizer> GrammarConstraint<T> {
    pub fn new(
        tokenizer: T, 
        parser: GLRParser, 
        llm_tokens: LLMTokenMap, 
        eof_llm_token_id: usize, 
        max_llm_token_id: usize
    ) -> Self {
        let mut precomputed = precompute::precompute(&tokenizer, &llm_tokens, LLMTokenID(eof_llm_token_id), max_llm_token_id);

        Self {
            tokenizer,
            parser,
            precomputed,
            max_llm_token_id,
        }
    }

    pub fn init(self) -> GrammarConstraintState<T> {
        let parser_initial_state = self.parser.init_parse_state();
        let tokenizer_initial_state_id = StateID(self.tokenizer.initial_state_id());

        GrammarConstraintState {
            parent: self,
            states: vec![(parser_initial_state, BTreeSet::from([tokenizer_initial_state_id]))],
        }
    }
}

impl<'a, T: Tokenizer> GrammarConstraintState<T> {
    pub fn get_mask(&self) -> BitVec {
        let mut result = bitvec![0; self.parent.max_llm_token_id + 1];

        let mut initial_nodes_and_values = Vec::new();

        for (parse_state, tokenizer_state_ids) in &self.states {
            for tokenizer_state in tokenizer_state_ids {
                let token_sequence_map = &self.parent.precomputed[tokenizer_state];
                let active_tokens = bitvec![1; self.parent.max_llm_token_id + 1];
                initial_nodes_and_values.push((Arc::new(Mutex::new(token_sequence_map.clone())), (vec![parse_state.clone()], active_tokens)));
            }
        }

        TrieNode::special_map(
            initial_nodes_and_values,
            |(current_parse_states, active_tokens), token_id, token_gate, _dst_node| {
                let mut glr_parse_state = self.parent.parser.init_glr_parser_from_parse_states(current_parse_states.clone());
                glr_parse_state.step(TerminalID(*token_id));
                (glr_parse_state.active_states, active_tokens.clone() & token_gate)
            },
            |values| {
                let mut all_parse_states = Vec::new();
                let mut all_active_tokens = bitvec![0; self.parent.max_llm_token_id + 1];
                for (parse_states, active_tokens) in values {
                    all_parse_states.extend(parse_states);
                    all_active_tokens |= active_tokens;
                }
                let mut new_glr_parse_state = self.parent.parser.init_glr_parser_from_parse_states(all_parse_states);
                new_glr_parse_state.merge_active_states();
                (new_glr_parse_state.active_states, all_active_tokens)

            },
            |(_, bitsets, maybe_clean_end_bitset), (current_parse_states, active_tokens)| {
                let mut glr_parse_state = self.parent.parser.init_glr_parser_from_parse_states(current_parse_states.clone());
                if glr_parse_state.is_ok() {
                    for (possible_next_grammar_token, bitset) in bitsets {
                        let mut new_glr_parse_state = glr_parse_state.clone();
                        new_glr_parse_state.step(TerminalID(*possible_next_grammar_token));

                        if new_glr_parse_state.is_ok() {
                            result |= bitset;
                        }
                    }
                    if let Some(bitset) = maybe_clean_end_bitset {
                        result |= bitset;
                    }
                };
                !active_tokens.is_empty()
            },
        );
        result
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let mut new_states: BTreeMap<(ParseStateKey, BTreeSet<StateID>), ParseState> = BTreeMap::new();
        let mut initial_nodes_and_values = Vec::new();
        for (parse_state, tokenizer_state_ids) in &self.states {
            for tokenizer_state_id in tokenizer_state_ids {
                let token_sequence_map = &self.parent.precomputed[tokenizer_state_id];
                let active_tokens = bitvec![1; self.parent.max_llm_token_id + 1];
                initial_nodes_and_values.push((Arc::new(Mutex::new(token_sequence_map.clone())), (vec![parse_state.clone()], active_tokens)));
            }
        }

        // todo: should be able to remove active tokens from this, not relevant.

        // todo: should be able to do the below loop more efficiently by optimising the precomputed
        //  stuff for earlier llm token lookup
        TrieNode::special_map(
            // todo: it's messy that we need to access the value in dst_node here.
            initial_nodes_and_values,
            |(current_parse_states, active_tokens), token_id, token_gate, _dst_node| {
                // todo: this is introducing redundancy... ?
                let mut glr_parse_state = self.parent.parser.init_glr_parser_from_parse_states(current_parse_states.clone());
                glr_parse_state.step(TerminalID(*token_id));
                (glr_parse_state.active_states, active_tokens.clone() & token_gate)
            },
            |values| {
                let mut all_parse_states = Vec::new();
                let mut all_active_tokens = bitvec![0; self.parent.max_llm_token_id + 1];
                for (parse_states, active_tokens) in values {
                    all_parse_states.extend(parse_states);
                    all_active_tokens |= active_tokens;
                }
                let mut new_glr_parse_state = self.parent.parser.init_glr_parser_from_parse_states(all_parse_states);
                new_glr_parse_state.merge_active_states();
                (new_glr_parse_state.active_states, all_active_tokens)
            },
            |(llm_token_id_to_state_id, _, _), (current_parse_states, active_tokens)| {
                let mut new_glr_parse_state = self.parent.parser.init_glr_parser_from_parse_states(current_parse_states.clone());
                if let Some(info) = llm_token_id_to_state_id.get(&llm_token_id) {
                    for active_parse_state in new_glr_parse_state.active_states {
                        new_states.insert_with(
                            (active_parse_state.key(), BTreeSet::from([info.unwrap_or(StateID(0))])),
                            active_parse_state,
                            |old, new| {
                                old.merge(new);
                            },
                        );
                    }
                };
                true
            },
        );
        
        self.states = new_states.into_iter().map(|((_key, tokenizer_state_ids), parse_state)| {
            (parse_state, tokenizer_state_ids)
        }).collect();
    }

    pub fn commit_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id);
        }
    }

    pub fn get_precomputed(&self) -> &BTreeMap<StateID, TrieNode<BitVec, TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>> {
        &self.parent.precomputed
    }
}