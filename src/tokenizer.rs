use crate::finite_automata::{GroupID, Regex};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::types::TerminalID as GrammarTokenID;
use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
// Added
use std::collections::{BTreeMap as StdMap, BTreeSet, BTreeMap};
// Added for derive macro pattern, aliased to avoid conflict

pub type LLMToken = Vec<u8>;
// Changed from BiBTreeMap to BTreeMap - we never use bidirectional lookup for this map
pub type LLMTokenMap = BTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct LLMTokenID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct TokenizerStateID(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, JSONConvertible)] // Added Ord for potential use in BTreeSet/Map
pub struct Token {
    pub id: GroupID, // GroupID is usize
    pub width: usize,
}

#[derive(Debug, JSONConvertible)]
pub struct ExecuteResult {
    pub matches: Vec<Token>,
    pub end_state: Option<usize>,
}


impl Regex {
    pub fn initial_state_id(&self) -> TokenizerStateID {
        TokenizerStateID(0)
    }

    pub fn execute_from_state(&self, text: &[u8], state: TokenizerStateID) -> ExecuteResult {
        let mut regex_state = self.init_to_state(state.0);
        regex_state.execute(text);

        // dbg!(&regex_state.matches);
        // println!("Executed from state {} with text {:?}. Matches: {:?}", state.0, text, regex_state.matches);

        let matches: Vec<_> = regex_state.matches.iter().map(|(&id, &width)| Token { id, width })
            // Filter out zero-width tokens
            .filter(|token| token.width != 0).collect();

        let new_state = if regex_state.done { None } else { Some(regex_state.current_state) };

        ExecuteResult { matches, end_state: new_state }
    }

    pub fn tokens_accessible_from_state(&self, state: TokenizerStateID) -> BTreeSet<GrammarTokenID> {
        let regex_state = self.init_to_state(state.0);
        regex_state.possible_future_group_ids().iter().cloned().map(|id| GrammarTokenID(id)).collect()
    }

    pub fn max_state(&self) -> usize {
        self.dfa.states.len()
    }

    pub(crate) fn iter_states(&self) -> impl Iterator<Item=TokenizerStateID> {
        (0..self.max_state()).map(|id| TokenizerStateID(id))
    }
}

