use crate::finite_automata::{GroupID, Regex};
use crate::glr;
use crate::glr::table::StateID;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use bitvec::prelude::BitVec;
use kdam::tqdm;
use crate::trie::{TrieNode};
use bimap::BiBTreeMap;
use bitvec::bitvec;
use crate::debug;

use crate::glr::parser::{GLRParser, GLRParserState, InsertWith, ParseState, ParseStateKey};
use crate::glr::table::{TerminalID};
use crate::constraint::create;
use bitvec::prelude::*;


pub type TokenID = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenID(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Token {
    pub id: GroupID,
    pub width: usize,
}

pub struct ExecuteResult {
    pub matches: Vec<Token>,
    pub new_state: Option<usize>,
}

// TODO: get rid of this trait. Just implement it directly on the Tokenizer struct.
/// Trait defining the tokenizer behavior.
pub trait Tokenizer: Sized {
    /// Returns the initial state ID.
    fn initial_state_id(&self) -> usize;

    /// Executes the tokenizer on the given text starting from the specified state.
    /// Returns all possible next tokens (**not** a sequence of tokens).
    fn execute_from_state(&self, text: &[u8], state: usize) -> ExecuteResult;

    /// Returns the list of tokens accessible from the given state.
    fn tokens_accessible_from_state(&self, state: usize) -> Vec<TokenID>;

    /// Returns the maximum state ID in the DFA.
    fn max_state(&self) -> usize;
}

/// Precomputes a map from state -> token sequence -> LLM token -> state.
pub fn precompute<'a>(
    tokenizer: &impl Tokenizer,
    llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    eof_llm_token_id: LLMTokenID,
    max_llm_token_id: usize,
) -> BTreeMap<StateID, TrieNode<(), TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>> {
    todo!()
}

impl Tokenizer for Regex {
    fn initial_state_id(&self) -> usize {
        0
    }

    fn execute_from_state(&self, text: &[u8], state: usize) -> ExecuteResult {
        let mut regex_state = self.init_to_state(state);
        regex_state.execute(text);

        let matches: Vec<_> = regex_state.matches.iter().map(|(&id, &width)| Token { id, width })
            // Filter out zero-width tokens
            .filter(|token| token.width != 0).collect();

        ExecuteResult {
            matches,
            new_state: if regex_state.done { None } else { Some(regex_state.current_state) },
        }
    }

    fn tokens_accessible_from_state(&self, state: usize) -> Vec<TokenID> {
        let regex_state = self.init_to_state(state);
        regex_state.possible_group_ids().iter().cloned().collect()
    }

    fn max_state(&self) -> usize {
        self.dfa.states.len()
    }
}

pub fn print_precomputed(precomputed: &BTreeMap<StateID, TrieNode<(), TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charmap::TrieMap;
    use crate::finite_automata::{eat_u8, DFAState, Regex, DFA};
    use crate::u8set::U8Set;
    use crate::{groups, seq};
    use std::collections::{BTreeMap, BTreeSet};
    use bimap::BiBTreeMap;

    #[test]
    fn test_precompute() {
        let _tokenizer = groups![
            eat_u8(b'a'), // Token 0: 'a'
            eat_u8(b'b'), // Token 1: 'b'
            seq![eat_u8(b'a'), eat_u8(b'b')], // Token 2: 'ab'
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'c')], // Token 3: 'abc'
        ].build();

        let tokenizer = Regex {
            dfa: DFA {
                states: vec![
                    DFAState {
                        transitions: TrieMap::from_iter(vec![(b'a', 1), (b'b', 2)]),
                        finalizers: BTreeSet::new(),
                        possible_group_ids: BTreeSet::from([0, 1, 2, 3]),
                        group_id_to_u8set: BTreeMap::from([
                            (0, U8Set::from_bytes(b"a")),
                            (1, U8Set::from_bytes(b"b")),
                            (2, U8Set::from_bytes(b"a")),
                            (3, U8Set::from_bytes(b"a")),
                        ]),
                    },
                    DFAState {
                        transitions: TrieMap::from_iter(vec![(b'b', 3)]),
                        finalizers: BTreeSet::from([0]),
                        possible_group_ids: BTreeSet::from([0, 2, 3]),
                        group_id_to_u8set: BTreeMap::from([
                            (2, U8Set::from_bytes(b"b")),
                            (3, U8Set::from_bytes(b"b")),
                        ]),
                    },
                    DFAState {
                        transitions: TrieMap::new(),
                        finalizers: BTreeSet::from([1]),
                        possible_group_ids: BTreeSet::from([1]),
                        group_id_to_u8set: BTreeMap::new(),
                    },
                    DFAState {
                        transitions: TrieMap::from_iter(vec![(b'c', 4)]),
                        finalizers: BTreeSet::from([2]),
                        possible_group_ids: BTreeSet::from([2, 3]),
                        group_id_to_u8set: BTreeMap::from([(3, U8Set::from_bytes(b"c"))]),
                    },
                    DFAState {
                        transitions: TrieMap::new(),
                        finalizers: BTreeSet::from([3]),
                        possible_group_ids: BTreeSet::from([3]),
                        group_id_to_u8set: BTreeMap::new(),
                    },
                ],
                start_state: 0,
                non_greedy_finalizers: BTreeSet::new(),
            },
        };
        assert_eq!(_tokenizer, tokenizer);

        // Define the LLM tokens
        let llm_tokens: &[&[u8]] = &[b"a", b"b", b"c", b"ab", b"bc", b"abc"];
        let llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID> = llm_tokens.iter().enumerate().map(|(i, token)| (token.to_vec(), LLMTokenID(i))).collect();

        // Run precompute
        let max_llm_token_id = llm_tokens.len() + 1;
        let result = precompute(&tokenizer, &llm_token_map, LLMTokenID(max_llm_token_id), max_llm_token_id);
    }
}


type LLMToken = Vec<u8>;
type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone)]
pub struct GrammarConstraint<T: Tokenizer> {
    pub(crate) tokenizer: T,
    pub(crate) parser: GLRParser,
    pub precomputed: BTreeMap<StateID, TrieNode<(), TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>,
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
        let mut precomputed = create::precompute(&tokenizer, &llm_tokens, LLMTokenID(eof_llm_token_id), max_llm_token_id);

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
                (glr_parse_state.active_states, active_tokens.clone())
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
                (glr_parse_state.active_states, active_tokens.clone())
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

    pub fn get_precomputed(&self) -> &BTreeMap<StateID, TrieNode<(), TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>> {
        &self.parent.precomputed
    }
}