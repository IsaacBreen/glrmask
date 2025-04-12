use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use keyed_priority_queue::KeyedPriorityQueue;
use crate::managed_glr_parser::ManagedGLRParserState;
use crate::types::{TerminalID as GrammarTokenID};

pub type LLMTokenBV = BitVec;
pub type GrammarTokenBV = BitVec;

#[derive(Debug, Clone)]
pub struct PrecomputedFinalizer {
    pub(crate) compatible_grammar_tokens: GrammarTokenBV,
    pub(crate) compatible_llm_tokens: LLMTokenBV,
    pub(crate) tokenizer_state_id: TokenizerStateID,
}

#[derive(Debug, Clone)]
pub struct PrecomputedNodeContents {
    pub(crate) finalizers: Vec<PrecomputedFinalizer>,
    pub(crate) relevant_llm_token_ids: LLMTokenBV,
}

type Precomputed = BTreeMap<TokenizerStateID, Trie<GrammarTokenID, PrecomputedNodeContents>>;

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
    pub(crate) state: ManagedGLRParserState<'a>,
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
        let glr_parser_initial_state = self.parser.init_managed_glr_parser();
        let tokenizer_initial_state_id = self.tokenizer.initial_state_id();

        GrammarConstraintState {
            parent: self,
            state: glr_parser_initial_state,
        }
    }
}

impl GrammarConstraintState<'_> {
    pub fn get_mask(&self) -> LLMTokenBV {
        todo!()
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let mut grammar_token_trie_roots: BTreeMap<TokenizerStateID, Trie<GrammarTokenID, BTreeSet<TokenizerStateID>>> = BTreeMap::new();

    }

    pub fn commit_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id);
        }
    }
}
