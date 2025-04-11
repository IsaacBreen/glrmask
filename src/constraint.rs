use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, ParseState, ParseStateKey};
use crate::tokenizer::{GrammarTokenID, LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use keyed_priority_queue::KeyedPriorityQueue;

type Precomputed = BTreeMap<
    TokenizerStateID,
    Trie<
        GrammarTokenID,
        (
            BTreeMap<LLMTokenID, Option<TokenizerStateID>>,
            BTreeMap<GrammarTokenID, BitVec>,
            Option<BitVec>
        )
    >,
>;

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub(crate) tokenizer: Regex,
    pub(crate) parser: GLRParser,
    pub(crate) precomputed: Precomputed,
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub(crate) max_llm_token_id: usize,
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState {
    pub(crate) parent: GrammarConstraint,
    pub(crate) states: Vec<(ParseState, BTreeSet<TokenizerStateID>)>,
}

impl GrammarConstraint {
    pub fn new(
        tokenizer: Regex,
        parser: GLRParser,
        llm_tokens: LLMTokenMap,
        max_llm_token_id: usize
    ) -> Self {
        todo!()
    }

    pub fn init(self) -> GrammarConstraintState {
        let parser_initial_state = self.parser.init_parse_state();
        let tokenizer_initial_state_id = self.tokenizer.initial_state_id();

        GrammarConstraintState {
            parent: self,
            states: vec![(parser_initial_state, BTreeSet::from([tokenizer_initial_state_id]))],
        }
    }
}

/// Precomputes a map from state -> token sequence -> LLM token -> state.
pub fn precompute<'a>(
    tokenizer: &Regex,
    llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    max_llm_token_id: usize,
) -> Precomputed {
    todo!()
}

impl<'a> GrammarConstraintState {
    pub fn get_mask(&self) -> BitVec {
        todo!()
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        fn pop_and_merge(queue: &mut KeyedPriorityQueue<(usize, ParseStateKey), (ParseState, BTreeSet<TokenizerStateID>)>) -> Option<((usize, ParseStateKey), (ParseState, BTreeSet<TokenizerStateID>))> {
            let ((position, parse_state_key), (parse_state, tokenizer_states)) = queue.pop()?;
            let mut combined_parse_state = parse_state;
            let mut combined_tokenizer_states = tokenizer_states;
            // Pop while the priority is the same
            loop {
                if let Some(((next_position, next_parse_state_key), _)) = queue.peek() {
                    if (next_position, next_parse_state_key) == (&position, &parse_state_key) {
                        let (_, (next_parse_state, next_tokenizer_states)) = queue.pop().unwrap();
                        combined_parse_state.merge(next_parse_state);
                        combined_tokenizer_states.extend(next_tokenizer_states);
                        continue;
                    }
                }
                break;
            }
            Some(((position, parse_state_key), (combined_parse_state, combined_tokenizer_states)))
        };

        let mut queue = KeyedPriorityQueue::new();
        for (parse_state, tokenizer_states) in std::mem::take(&mut self.states) {
            let parse_state_key = parse_state.key();
            queue.push((0, parse_state_key), (parse_state, tokenizer_states));
        }
        let mut new_states: Vec<(ParseState, BTreeSet<TokenizerStateID>)> = Vec::new();
        let mut token = self.parent.llm_token_map.get_by_right(&llm_token_id).unwrap().clone(); // Get the LLM token contents
        while let Some(((position, _parse_state_key), (mut parse_state, tokenizer_states))) = pop_and_merge(&mut queue) {
            // Get matches
            let mut tokens = BTreeSet::new();
            let mut new_tokenizer_states = BTreeSet::new();
            for tokenizer_state in tokenizer_states {
                let match_results = self.parent.tokenizer.execute_from_state(&token[position..], tokenizer_state);
                tokens.extend(match_results.matches);
                if let Some(new_state) = match_results.new_state {
                    new_tokenizer_states.insert(new_state);
                }
            }
            // Feed the new tokens into the parse state and add to queue
            for token in tokens {
                let new_parse_state = parse_state.clone();
                // new_parse_state.step(token);
                // Ah fuck, we need to construct GLRParseState to call step.
                // Probably should make this simpler. Lean into the glr module more to do all this complex shit.
                
            }
        }
        self.states = new_states;
    }

    pub fn commit_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id);
        }
    }
}
