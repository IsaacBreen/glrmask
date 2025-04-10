use crate::finite_automata::{GroupID, Regex};
use crate::glr::parser::{GLRParser, InsertWith, ParseState};
use crate::glr::table::StateID;
use crate::datastructures::trie::Trie;
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};


type LLMToken = Vec<u8>;
type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GrammarTokenID(pub usize);
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

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub(crate) tokenizer: Regex,
    pub(crate) parser: GLRParser,
    pub(crate) precomputed: BTreeMap<StateID, Trie<GrammarTokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<GrammarTokenID, BitVec>, Option<BitVec>)>>,
    pub(crate) max_llm_token_id: usize,
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState {
    pub(crate) parent: GrammarConstraint,
    pub(crate) states: Vec<(ParseState, BTreeSet<StateID>)>,
}

impl Regex {
    fn initial_state_id(&self) -> usize {
        0
    }

    fn execute_from_state(&self, text: &[u8], state: usize) -> ExecuteResult {
        let mut regex_state = self.init_to_state(state);
        regex_state.execute(text);

        let matches: Vec<_> = regex_state.matches.iter().map(|(&id, &width)| Token { id, width })
            // Filter out zero-width tokens
            .filter(|token| token.width != 0).collect();

        let new_state = if regex_state.done { None } else { Some(regex_state.current_state) };

        ExecuteResult { matches, new_state }
    }

    fn tokens_accessible_from_state(&self, state: usize) -> Vec<GrammarTokenID> {
        let regex_state = self.init_to_state(state);
        regex_state.possible_group_ids().iter().cloned().map(|id| GrammarTokenID(id)).collect()
    }

    fn max_state(&self) -> usize {
        self.dfa.states.len()
    }
}

/// Precomputes a map from state -> token sequence -> LLM token -> state.
pub fn precompute<'a>(
    tokenizer: &Regex,
    llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    max_llm_token_id: usize,
) -> BTreeMap<StateID, Trie<GrammarTokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<GrammarTokenID, BitVec>, Option<BitVec>)>> {
    todo!()
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
        let tokenizer_initial_state_id = StateID(self.tokenizer.initial_state_id());

        GrammarConstraintState {
            parent: self,
            states: vec![(parser_initial_state, BTreeSet::from([tokenizer_initial_state_id]))],
        }
    }
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
