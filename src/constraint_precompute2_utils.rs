use crate::constraint::{PrecomputeNode2Index, Trie2GodWrapper};
use crate::tokenizer::TokenizerStateID;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Trie2Config {
    pub enabled: bool,
}

impl Default for Trie2Config {
    fn default() -> Self {
        Self {
            enabled: true,
        }
    }
}

impl Trie2Config {
    pub fn off() -> Self {
        Self {
            enabled: false,
        }
    }
}

pub fn optimize_trie2_size(
    _roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode2Index>,
    _trie2_god: &Trie2GodWrapper,
    _config: &Trie2Config,
) {
    // All optimizations have been removed.
}
