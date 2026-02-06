use crate::dfa_u8::dfa::{GroupID, Regex, DFA};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::types::TerminalID as GrammarTokenID;
use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
// Added
use std::collections::{BTreeMap as StdMap, BTreeSet, BTreeMap};
// Added for derive macro pattern, aliased to avoid conflict

pub type LLMToken = Vec<u8>;
// Use BTreeMap for compatibility with serialization
pub type LLMTokenMap = BTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct LLMTokenID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct TokenizerStateID(pub usize);

// TODO: Just use Match and ExecutionResult in finite_automata.rs
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, JSONConvertible)] // Added Ord for potential use in BTreeSet/Map
pub struct Token {
    pub id: GroupID, // GroupID is usize
    pub width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerLenConstraint {
    pub min: usize,
    pub max: Option<usize>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenizer {
    inner: Regex,
    len_constraints: Option<BTreeMap<GroupID, TokenizerLenConstraint>>,
}

impl Tokenizer {
    /// Create a new Tokenizer from a Regex.
    pub fn new(regex: Regex) -> Self {
        Self {
            inner: regex,
            len_constraints: None,
        }
    }

    pub fn new_with_len_constraints(
        regex: Regex,
        len_constraints: BTreeMap<GroupID, TokenizerLenConstraint>,
    ) -> Self {
        Self {
            inner: regex,
            len_constraints: if len_constraints.is_empty() { None } else { Some(len_constraints) },
        }
    }
    
    /// Get the initial state ID.
    pub fn initial_state_id(&self) -> TokenizerStateID {
        TokenizerStateID(0)
    }
    
    /// Execute the tokenizer from a given state on a byte slice.
    /// Returns the matched tokens and the end state (None if no more matches possible).
    pub fn execute_from_state(&self, text: &[u8], state: TokenizerStateID) -> ExecuteResult {
        let mut result = self.inner.execute_from_state(text, state);
        if let Some(constraints) = &self.len_constraints {
            result.matches.retain(|token| {
                if let Some(constraint) = constraints.get(&token.id) {
                    let slice = if token.width <= text.len() {
                        &text[..token.width]
                    } else {
                        return false;
                    };
                    if let Some(count) = count_json_string_units(slice) {
                        if count < constraint.min {
                            return false;
                        }
                        if let Some(max) = constraint.max {
                            if count > max {
                                return false;
                            }
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            });
        }
        result
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

fn count_json_string_units(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 2 || bytes.first().copied() != Some(b'"') || bytes.last().copied() != Some(b'"') {
        return None;
    }

    let mut i = 1usize;
    let end = bytes.len() - 1;
    let mut count = 0usize;

    while i < end {
        let b = bytes[i];
        if b == b'\\' {
            if i + 1 >= end {
                return None;
            }
            let esc = bytes[i + 1];
            match esc {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    i += 2;
                    count += 1;
                }
                b'u' => {
                    if i + 5 >= end {
                        return None;
                    }
                    if !bytes[i + 2..i + 6].iter().all(|c| c.is_ascii_hexdigit()) {
                        return None;
                    }
                    i += 6;
                    count += 1;
                }
                _ => return None,
            }
        } else {
            i += 1;
            count += 1;
        }
    }

    Some(count)
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
        TokenizerStateID(0)
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

