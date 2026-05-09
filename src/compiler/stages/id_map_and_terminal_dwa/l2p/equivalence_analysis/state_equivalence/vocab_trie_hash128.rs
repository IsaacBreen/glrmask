use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::identity_state_map;
use super::pass::StateEquivalencePass;

#[derive(Debug, Clone, Default)]
pub(crate) struct VocabTrieNode {
    pub children: BTreeMap<u8, usize>,
    pub terminal_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct VocabTrie {
    pub nodes: Vec<VocabTrieNode>,
}

impl VocabTrie {
    fn new() -> Self {
        Self {
            nodes: vec![VocabTrieNode::default()],
        }
    }

    fn insert(&mut self, token_id: u32, token: &[u8]) {
        let mut node = 0usize;
        for &byte in token {
            let next = if let Some(&next) = self.nodes[node].children.get(&byte) {
                next
            } else {
                let next = self.nodes.len();
                self.nodes.push(VocabTrieNode::default());
                self.nodes[node].children.insert(byte, next);
                next
            };
            node = next;
        }
        self.nodes[node].terminal_token_ids.push(token_id);
    }

    fn from_vocab(vocab: &Vocab) -> Self {
        let mut trie = Self::new();
        let mut entries: Vec<(u32, &[u8])> = vocab
            .entries
            .iter()
            .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
            .collect();
        entries.sort_unstable_by_key(|(token_id, _)| *token_id);
        for (token_id, bytes) in entries {
            trie.insert(token_id, bytes);
        }
        trie
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VocabTrieHash128Pass;

impl StateEquivalencePass for VocabTrieHash128Pass {
    type Statistic = VocabTrie;

    fn name(&self) -> &'static str {
        "vocab_trie_hash128"
    }

    fn compute_statistic(&self, vocab: &Vocab) -> Self::Statistic {
        VocabTrie::from_vocab(vocab)
    }

    fn compute_state_map(
        &self,
        tokenizer: &Tokenizer,
        _statistic: &Self::Statistic,
        initial_state_map: Option<&ManyToOneIdMap>,
        _active_groups: Option<&[bool]>,
    ) -> ManyToOneIdMap {
        // TODO: Replace this identity implementation with the real 128-bit trie/hash pass.
        // The future version must be extremely fast for the full llama3 vocab, tokenizers
        // with roughly 78k states, and the `JSON_STRING_CHAR{0,256}` +
        // `^(?:\\S+\\s+){0,49}\\S+$` problem shape. It should use the exact
        // `f(tsid, string)` semantics from `f_signature.rs`.
        initial_state_map
            .cloned()
            .unwrap_or_else(|| identity_state_map(tokenizer.num_states() as usize))
    }
}