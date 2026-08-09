//! Shared scanner and vocabulary-trie support for experimental L1 builders.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::BuildInput;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::SingletonEpsilonClosures;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

pub(super) const UNKNOWN: u32 = u32::MAX - 1;
pub(super) const DEAD: u32 = u32::MAX;

pub(super) struct Scanner<'a> {
    input: BuildInput<'a>,
    pub configs: Vec<Box<[u32]>>,
    ids: FxHashMap<Vec<u32>, u32>,
    transitions: Vec<[u32; 256]>,
    pub signatures: Vec<Vec<u32>>,
    signature_ids: FxHashMap<Vec<u32>, u32>,
    config_signature: Vec<u32>,
    singleton_closures: Arc<SingletonEpsilonClosures>,
}

impl<'a> Scanner<'a> {
    pub fn new(input: BuildInput<'a>) -> Self {
        Self {
            input,
            configs: Vec::new(),
            ids: FxHashMap::default(),
            transitions: Vec::new(),
            signatures: vec![Vec::new()],
            signature_ids: FxHashMap::from_iter([(Vec::new(), 0)]),
            config_signature: Vec::new(),
            singleton_closures: input.tokenizer.all_singleton_epsilon_closures(),
        }
    }

    fn intern(&mut self, mut states: Vec<u32>) -> u32 {
        if states.is_empty() {
            return DEAD;
        }
        states.sort_unstable();
        states.dedup();
        if let Some(&id) = self.ids.get(&states) {
            return id;
        }
        let mut signature = states
            .iter()
            .flat_map(|&state| {
                super::super::collect_active_terminal_signature(
                    self.input.tokenizer,
                    state,
                    self.input.active_terminals,
                )
            })
            .collect::<Vec<_>>();
        signature.sort_unstable();
        signature.dedup();
        let next_signature = self.signatures.len() as u32;
        let signature_id = *self.signature_ids.entry(signature.clone()).or_insert_with(|| {
            self.signatures.push(signature);
            next_signature
        });
        let id = self.configs.len() as u32;
        self.ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.transitions.push([UNKNOWN; 256]);
        self.config_signature.push(signature_id);
        id
    }

    pub fn start(&mut self, state: u32) -> u32 {
        self.intern(self.singleton_closures[state as usize].to_vec())
    }

    #[inline]
    pub fn step(&mut self, config: u32, byte: u8) -> u32 {
        if config == DEAD {
            return DEAD;
        }
        let cached = self.transitions[config as usize][byte as usize];
        if cached != UNKNOWN {
            return cached;
        }
        let target = if self.configs[config as usize].len() == 1 {
            let state = self.configs[config as usize][0];
            match self.input.tokenizer.step(state, byte) {
                Some(raw_target) => self.intern(self.singleton_closures[raw_target as usize].to_vec()),
                None => DEAD,
            }
        } else {
            self.intern(
                self.input
                    .tokenizer
                    .step_all(&self.configs[config as usize], byte)
                    .to_vec(),
            )
        };
        self.transitions[config as usize][byte as usize] = target;
        target
    }

    pub fn step_bytes(&mut self, mut config: u32, bytes: &[u8]) -> u32 {
        for &byte in bytes {
            config = self.step(config, byte);
            if config == DEAD {
                break;
            }
        }
        config
    }

    #[inline]
    pub fn singleton_state(&self, config: u32) -> Option<u32> {
        let states = self.configs.get(config as usize)?;
        (states.len() == 1).then_some(states[0])
    }

    #[inline]
    pub fn signature(&self, config: u32) -> u32 {
        if config == DEAD { 0 } else { self.config_signature[config as usize] }
    }
}

pub(super) fn vocab(input: BuildInput<'_>) -> (Vec<Vec<u32>>, VocabPrefixTree) {
    // L1 already prepares one byte-sorted vocabulary order per partition. Reuse
    // it here instead of cloning every token and sorting the same vocabulary a
    // second time. Split-L1 derives its order from the parent in linear time.
    let order = input
        .subset_parent_order
        .map(|parent| super::super::derive_l1_identity_vocab_order_from_parent(parent, input.vocab))
        .unwrap_or_else(|| super::super::prepared_l1_identity_vocab_order(input.vocab));
    let mut aliases = Vec::<Vec<u32>>::new();
    let mut unique = Vec::<&[u8]>::new();
    for &(id, ref bytes) in order.token_entries_sorted.iter() {
        if unique.last().is_some_and(|previous| *previous == bytes.as_ref()) {
            aliases.last_mut().expect("duplicate token has predecessor").push(id);
        } else {
            unique.push(bytes.as_ref());
            aliases.push(vec![id]);
        }
    }
    let refs = unique
        .iter()
        .enumerate()
        .map(|(id, &bytes)| (id, bytes))
        .collect::<Vec<_>>();
    (aliases, VocabPrefixTree::build_presorted(&refs))
}

#[derive(Default)]
pub(super) struct FlatNode {
    pub token: Option<usize>,
    pub edges: Vec<(Box<[u8]>, usize)>,
}

pub(super) fn flatten(tree: &VocabPrefixTree) -> Vec<FlatNode> {
    fn add(node: &VocabPrefixTreeNode, out: &mut Vec<FlatNode>) -> usize {
        let index = out.len();
        out.push(FlatNode::default());
        let token = node.has_token().then(|| node.token_id());
        let edges = node
            .iter_children()
            .map(|(edge, child)| (edge.to_vec().into_boxed_slice(), add(child, out)))
            .collect();
        out[index] = FlatNode { token, edges };
        index
    }
    let mut nodes = Vec::new();
    add(&tree.root, &mut nodes);
    nodes
}
