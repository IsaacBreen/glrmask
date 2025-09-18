use crate::finite_automata::{GroupID, Regex};
use crate::types::{TerminalID as GrammarTokenID};
use bimap::BiBTreeMap;
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::{BTreeMap as StdMap, BTreeSet}; // Added for derive macro pattern, aliased to avoid conflict

pub type LLMToken = Vec<u8>;
pub type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenID(pub usize);

// Manual impl for LLMTokenID (could be derived)
impl JSONConvertible for LLMTokenID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(LLMTokenID)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenizerStateID(pub usize);

// Manual impl for TokenizerStateID (could be derived)
impl JSONConvertible for TokenizerStateID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(TokenizerStateID)
    }
}


#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)] // Added Ord for potential use in BTreeSet/Map
pub struct Token {
    pub id: GroupID, // GroupID is usize
    pub width: usize,
}

// Manual impl for Token (could be derived)
impl JSONConvertible for Token {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("id".to_string(), self.id.to_json());
        obj.insert("width".to_string(), self.width.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let id = obj.remove("id").ok_or_else(|| "Missing field id for Token".to_string())
                                 .and_then(GroupID::from_json)?;
                let width = obj.remove("width").ok_or_else(|| "Missing field width for Token".to_string())
                                 .and_then(usize::from_json)?;
                Ok(Token { id, width })
            }
            _ => Err("Expected JSONNode::Object for Token".to_string()),
        }
    }
}


#[derive(Debug)]
pub struct ExecuteResult {
    pub matches: Vec<Token>,
    pub end_state: Option<usize>,
}

// Manual impl for ExecuteResult (could be derived)
impl JSONConvertible for ExecuteResult {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("matches".to_string(), self.matches.to_json());
        obj.insert("end_state".to_string(), self.end_state.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let matches = obj.remove("matches").ok_or_else(|| "Missing field matches for ExecuteResult".to_string())
                                 .and_then(Vec::<Token>::from_json)?;
                let end_state = obj.remove("end_state").ok_or_else(|| "Missing field end_state for ExecuteResult".to_string())
                                 .and_then(Option::<usize>::from_json)?;
                Ok(ExecuteResult { matches, end_state })
            }
            _ => Err("Expected JSONNode::Object for ExecuteResult".to_string()),
        }
    }
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

