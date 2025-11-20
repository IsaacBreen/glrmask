use std::collections::BTreeMap;
use std::collections::BTreeMap as StdMap;

use crate::constraint_vocab::LLMTokenBV;
use crate::datastructures::trie::{God, GodWrapper, Trie, Trie2Index};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::tokenizer::TokenizerStateID;
use crate::types::TerminalID as GrammarTokenID;
use crate::precompute4::weighted_automata::bitset::SimpleBitset;

// ---------------------------------------------------------------------------
// Trie node contents
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents {
    pub(crate) end: bool,
    pub(crate) live_tokens: LLMTokenBV,
}

impl PrecomputedNodeContents {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self {
            end: false,
            live_tokens: LLMTokenBV::ones(internal_max_llm_token_id + 1),
        }
    }

    pub(crate) fn internal() -> Self {
        profiler_macro::timeit!("PrecomputedNodeContents::internal", {});
        Self {
            end: false,
            live_tokens: LLMTokenBV::zeros(),
        }
    }

    pub(crate) fn leaf() -> Self {
        Self {
            end: true,
            live_tokens: LLMTokenBV::zeros(),
        }
    }
}

impl JSONConvertible for PrecomputedNodeContents {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clean_end".to_string(), self.end.to_json());
        obj.insert("live_tokens".to_string(), self.live_tokens.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let end = obj
                    .remove("clean_end")
                    .ok_or_else(|| "Missing field clean_end for PrecomputedNodeContents".to_string())
                    .and_then(bool::from_json)?;
                let live_tokens = obj
                    .remove("live_tokens")
                    .ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents".to_string())
                    .and_then(LLMTokenBV::from_json)?;
                Ok(PrecomputedNodeContents { end, live_tokens })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents".to_string()),
        }
    }
}

// Minimal node contents for the (now empty) Trie0 type, kept only because
// optimize_trie1_size still takes a Trie0GodWrapper in its signature.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents0 {
    pub(crate) live_tokens: LLMTokenBV,
    pub(crate) final_tokenizer_state: Option<TokenizerStateID>,
}

impl PrecomputedNodeContents0 {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self {
            live_tokens: LLMTokenBV::ones(internal_max_llm_token_id + 1),
            final_tokenizer_state: None,
        }
    }

    pub(crate) fn internal() -> Self {
        Self {
            live_tokens: LLMTokenBV::zeros(),
            final_tokenizer_state: None,
        }
    }

    pub(crate) fn leaf(final_sid: TokenizerStateID) -> Self {
        Self {
            live_tokens: LLMTokenBV::zeros(),
            final_tokenizer_state: Some(final_sid),
        }
    }
}

impl JSONConvertible for PrecomputedNodeContents0 {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert(
            "clean_end".to_string(),
            self.final_tokenizer_state.is_some().to_json(),
        );
        obj.insert("live_tokens".to_string(), self.live_tokens.to_json());
        obj.insert(
            "final_tokenizer_state".to_string(),
            self.final_tokenizer_state.to_json(),
        );
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let live_tokens = obj
                    .remove("live_tokens")
                    .ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents0".to_string())
                    .and_then(LLMTokenBV::from_json)?;
                let final_tokenizer_state = obj
                    .remove("final_tokenizer_state")
                    .ok_or_else(|| {
                        "Missing field final_tokenizer_state for PrecomputedNodeContents0".to_string()
                    })
                    .and_then(Option::<TokenizerStateID>::from_json)?;
                Ok(PrecomputedNodeContents0 {
                    live_tokens,
                    final_tokenizer_state,
                })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents0".to_string()),
        }
    }
}

// Final precompute1 types
pub type PrecomputeNode1 =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode1Index = Trie2Index;
pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode1Index>;

// Trie0 is now only a type alias passed into optimize_trie1_size; we never build it.
pub type PrecomputeNode0 = Trie<
    Option<(GrammarTokenID, Option<TokenizerStateID>)>,
    LLMTokenBV,
    PrecomputedNodeContents0,
>;
pub type PrecomputeNode0Index = Trie2Index;

// God wrappers
pub type Trie0GodWrapper = GodWrapper<
    Option<(GrammarTokenID, Option<TokenizerStateID>)>,
    LLMTokenBV,
    PrecomputedNodeContents0,
>;
pub type Trie0God = God<
    Option<(GrammarTokenID, Option<TokenizerStateID>)>,
    LLMTokenBV,
    PrecomputedNodeContents0,
>;
pub type Trie1GodWrapper =
    GodWrapper<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type Trie1God = God<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
