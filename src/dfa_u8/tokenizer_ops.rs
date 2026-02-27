use crate::dfa_u8::dfa::{GroupID, Regex, DFA};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::types::TerminalID as GrammarTokenID;
use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
// Added
use std::collections::{BTreeSet, BTreeMap};
// Added for derive macro pattern, aliased to avoid conflict

pub type LLMToken = Vec<u8>;
// Use BTreeMap for compatibility with serialization
pub type LLMTokenMap = BTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible, serde::Serialize, serde::Deserialize)]
pub struct LLMTokenID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible, serde::Serialize, serde::Deserialize)]
pub struct TokenizerStateID(pub usize);

// TODO: Just use Match and ExecutionResult in finite_automata.rs
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

/// A tokenizer that lexes byte sequences into terminal tokens.
/// 
/// This is a newtype wrapper around `Regex` that provides a more semantic
/// interface for tokenization operations. While `Regex` is focused on pattern
/// matching, `Tokenizer` represents the lexer component that produces terminal
/// tokens for the GLR parser.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Tokenizer {
    inner: Regex,
}

impl Tokenizer {
    /// Create a new Tokenizer from a Regex.
    pub fn new(regex: Regex) -> Self {
        Self { inner: regex }
    }
    
    /// Get the initial state ID.
    pub fn initial_state_id(&self) -> TokenizerStateID {
        TokenizerStateID(self.inner.dfa.start_state)
    }
    
    /// Execute the tokenizer from a given state on a byte slice.
    /// Returns the matched tokens and the end state (None if no more matches possible).
    pub fn execute_from_state(&self, text: &[u8], state: TokenizerStateID) -> ExecuteResult {
        self.inner.execute_from_state(text, state)
    }
    
    /// Get the set of terminal IDs that could be matched from a given state.
    pub fn tokens_accessible_from_state(&self, state: TokenizerStateID) -> BTreeSet<GrammarTokenID> {
        self.inner.tokens_accessible_from_state(state)
    }
    
    /// Get the maximum state ID (number of states).
    pub fn max_state(&self) -> usize {
        self.inner.max_state()
    }
    
    /// Iterate over all valid state IDs.
    pub fn iter_states(&self) -> impl Iterator<Item=TokenizerStateID> {
        (0..self.max_state()).map(|id| TokenizerStateID(id))
    }
    
    /// Access the underlying Regex.
    pub fn as_regex(&self) -> &Regex {
        &self.inner
    }
    
    /// Access the underlying DFA.
    pub fn dfa(&self) -> &DFA {
        &self.inner.dfa
    }

    /// Reorder DFA states according to an old->new mapping.
    /// The mapping must be a permutation and should keep the start state at 0.
    pub fn reorder_states(&mut self, old_to_new: &[usize]) {
        self.inner.dfa.reorder_states(old_to_new);
    }
    
    /// Get the total number of groups (terminal types) in the tokenizer.
    pub fn num_groups(&self) -> usize {
        self.inner.num_groups()
    }
    
    /// Execute the tokenizer from a given state, filtering out zero-width matches.
    pub fn execute_from_state_nonzero(&self, text: &[u8], state: usize) -> crate::dfa_u8::dfa::ExecutionResult {
        self.inner.execute_from_state_nonzero(text, state)
    }
    
    /// Initialize to a specific state (for advanced usage).
    pub fn init_to_state(&self, state: usize) -> crate::dfa_u8::dfa::RegexState<'_> {
        self.inner.init_to_state(state)
    }
    
    /// Initialize the tokenizer.
    pub fn init(&self) -> crate::dfa_u8::dfa::RegexState<'_> {
        self.inner.init()
    }
}

impl From<Regex> for Tokenizer {
    fn from(regex: Regex) -> Self {
        Self::new(regex)
    }
}

impl std::fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl JSONConvertible for Tokenizer {
    fn to_json(&self) -> JSONNode {
        self.inner.to_json()
    }
    
    fn from_json(node: JSONNode) -> Result<Self, String> {
        Regex::from_json(node).map(Self::new)
    }
}

impl Regex {
    pub fn initial_state_id(&self) -> TokenizerStateID {
        TokenizerStateID(self.dfa.start_state)
    }

    pub fn execute_from_state(&self, text: &[u8], state: TokenizerStateID) -> ExecuteResult {
        // TODO: redundant -- see finite_automata.rs
        let mut regex_state = self.init_to_state(state.0);
        regex_state.execute(text);

        // dbg!(&regex_state.matches);
        // println!("Executed from state {} with text {:?}. Matches: {:?}", state.0, text, regex_state.matches);

        let matches: Vec<_> = regex_state.matches.iter().map(|(&id, &width)| Token { id, width })
            // Filter out zero-width tokens. Zero-width tokens would indicate a nullable
            // terminal (can match epsilon). Nullable terminals are handled at the GLR parser
            // level by transforming them into optional non-terminals with epsilon alternatives.
            // Keeping zero-width matches here would interfere with that handling.
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

    pub fn iter_states(&self) -> impl Iterator<Item=TokenizerStateID> {
        (0..self.max_state()).map(|id| TokenizerStateID(id))
    }
}

