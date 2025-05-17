use crate::finite_automata::{GroupID, Regex};
use crate::types::{TerminalID as GrammarTokenID};
use bimap::BiBTreeMap;

pub type LLMToken = Vec<u8>;
pub type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenID(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenizerStateID(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Token {
    pub id: GroupID,
    pub width: usize,
}

#[derive(Debug)]
pub struct ExecuteResult {
    pub matches: Vec<Token>,
    pub end_state: Option<usize>,
}

use crate::json_serialization::{JSONNode, JSONConvertible};

impl JSONConvertible for LLMTokenID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(LLMTokenID)
    }
}

impl JSONConvertible for TokenizerStateID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(TokenizerStateID)
    }
}

impl JSONConvertible for Token {
    fn to_json(&self) -> JSONNode {
        crate::json_serialization::struct_to_json_object(vec![
            ("id", self.id.to_json()),
            ("width", self.width.to_json()),
        ])
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        let map = crate::json_serialization::json_object_to_btreemap(node)?;
        Ok(Token {
            id: map.get("id").ok_or_else(|| "Missing field 'id'".to_string()).and_then(GroupID::from_json)?,
            width: map.get("width").ok_or_else(|| "Missing field 'width'".to_string()).and_then(usize::from_json)?,
        })
    }
}

impl JSONConvertible for ExecuteResult {
    fn to_json(&self) -> JSONNode {
        crate::json_serialization::struct_to_json_object(vec![
            ("matches", self.matches.to_json()),
            ("end_state", self.end_state.to_json()),
        ])
    }
    fn from_json(node: &JSONNode) -> Result<Self, String> {
        let map = crate::json_serialization::json_object_to_btreemap(node)?;
        Ok(ExecuteResult {
            matches: map.get("matches").ok_or_else(|| "Missing field 'matches'".to_string()).and_then(Vec::<Token>::from_json)?,
            end_state: map.get("end_state").ok_or_else(|| "Missing field 'end_state'".to_string()).and_then(Option::<usize>::from_json)?,
        })
    }
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
