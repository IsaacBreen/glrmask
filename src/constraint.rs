use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use keyed_priority_queue::KeyedPriorityQueue;
use crate::types::{TerminalID as GrammarTokenID};

pub type LLMTokenBV = BitVec;
pub type GrammarTokenBV = BitVec;

type Precomputed = BTreeMap<
    TokenizerStateID,
    Trie<
        GrammarTokenID,
        Vec<(GrammarTokenBV, LLMTokenBV, TokenizerStateID)>,
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
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    pub(crate) state: GLRParserState<'a>,
}

impl GrammarConstraint {
    pub fn new(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        max_llm_token_id: usize
    ) -> Self {
        let precomputed = GrammarConstraint::precompute(&tokenizer, &llm_token_map, max_llm_token_id);
        Self {
            tokenizer,
            parser,
            precomputed,
            llm_token_map,
            max_llm_token_id,
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


    pub fn init(&self) -> GrammarConstraintState<'_> {
        let glr_parser_initial_state = self.parser.init_glr_parser();
        let tokenizer_initial_state_id = self.tokenizer.initial_state_id();

        GrammarConstraintState {
            parent: self,
            state: glr_parser_initial_state,
        }
    }
}

impl GrammarConstraintState<'_> {
    pub fn get_mask(&self) -> LLMTokenBV {
        // let initial_nodes_and_values = Vec::new();

        // for state in self.state.active_states.iter() {
        //     let node = &self.parent.precomputed[state.];
        // }

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
