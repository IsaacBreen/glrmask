use serde::{Serialize, Deserialize};
use crate::finite_automata::{GroupID, Regex};
use crate::types::{TerminalID as GrammarTokenID};
use bimap::BiBTreeMap;

pub type LLMToken = Vec<u8>;
pub type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LLMTokenID(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenizerStateID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Token {
    pub id: GroupID,
    pub width: usize,
}

#[derive(Debug)]
pub struct ExecuteResult {
    pub matches: Vec<Token>,
    pub end_state: Option<usize>,
}

impl Regex {
    pub(crate) fn initial_state_id(&self) -> TokenizerStateID {
        TokenizerStateID(0)
    }

    pub(crate) fn execute_from_state(&self, text: &[u8], state: TokenizerStateID) -> ExecuteResult {
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

    pub(crate) fn tokens_accessible_from_state(&self, state: TokenizerStateID) -> Vec<GrammarTokenID> {
        let regex_state = self.init_to_state(state.0);
        regex_state.possible_future_group_ids().iter().cloned().map(|id| GrammarTokenID(id)).collect()
    }

    pub(crate) fn max_state(&self) -> usize {
        self.dfa.states.len()
    }
}

