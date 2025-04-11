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
        todo!()
    }

    pub fn commit_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id);
        }
    }
}
