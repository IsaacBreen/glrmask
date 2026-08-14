//! Shared scanner and vocabulary-trie support for experimental L1 builders.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::BuildInput;
use crate::automata::lexer::tokenizer::SingletonEpsilonClosures;
use crate::automata::lexer::Lexer;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

pub(super) const UNKNOWN: u32 = u32::MAX - 1;
pub(super) const DEAD: u32 = u32::MAX;

pub(super) struct Scanner<'a> {
    input: BuildInput<'a>,
    pub configs: Vec<Box<[u32]>>,
    ids: FxHashMap<Vec<u32>, u32>,
    singleton_ids: Vec<u32>,
    transitions: Vec<u32>,
    byte_slot: [u16; 256],
    transition_width: usize,
    pub signatures: Vec<Vec<u32>>,
    signature_ids: FxHashMap<Vec<u32>, u32>,
    config_signature: Vec<u32>,
    singleton_closures: Arc<SingletonEpsilonClosures>,
}

impl<'a> Scanner<'a> {
    pub fn new(input: BuildInput<'a>) -> Self {
        let relevant_bytes = input.vocab.relevant_bytes();
        let mut byte_slot = [u16::MAX; 256];
        for (slot, &byte) in relevant_bytes.iter().enumerate() {
            byte_slot[byte as usize] = slot as u16;
        }
        let transition_width = relevant_bytes.len();
        let transition_capacity = (input.tokenizer.num_states() as usize)
            .saturating_mul(transition_width);
        Self {
            input,
            configs: Vec::new(),
            ids: FxHashMap::default(),
            singleton_ids: vec![UNKNOWN; input.tokenizer.num_states() as usize],
            transitions: Vec::with_capacity(transition_capacity),
            byte_slot,
            transition_width,
            signatures: vec![Vec::new()],
            signature_ids: FxHashMap::from_iter([(Vec::new(), 0)]),
            config_signature: Vec::new(),
            singleton_closures: input.tokenizer.all_singleton_epsilon_closures(),
        }
    }

    #[inline]
    fn push_transition_row(&mut self) {
        self.transitions
            .extend(std::iter::repeat_n(UNKNOWN, self.transition_width));
    }

    fn intern_signature(&mut self, signature: Vec<u32>) -> u32 {
        let next_signature = self.signatures.len() as u32;
        *self.signature_ids.entry(signature.clone()).or_insert_with(|| {
            self.signatures.push(signature);
            next_signature
        })
    }

    fn intern_singleton(&mut self, state: u32) -> u32 {
        let existing = self.singleton_ids[state as usize];
        if existing != UNKNOWN {
            return existing;
        }
        let signature = super::super::collect_active_terminal_signature(
            self.input.tokenizer,
            state,
            self.input.active_terminals,
        );
        let signature_id = self.intern_signature(signature);
        let id = self.configs.len() as u32;
        self.singleton_ids[state as usize] = id;
        self.configs.push(Box::new([state]));
        self.push_transition_row();
        self.config_signature.push(signature_id);
        id
    }

    fn intern(&mut self, mut states: Vec<u32>) -> u32 {
        if states.is_empty() {
            return DEAD;
        }
        if states.len() == 1 {
            return self.intern_singleton(states[0]);
        }
        states.sort_unstable();
        states.dedup();
        if states.len() == 1 {
            return self.intern_singleton(states[0]);
        }
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
        let signature_id = self.intern_signature(signature);
        let id = self.configs.len() as u32;
        self.ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.push_transition_row();
        self.config_signature.push(signature_id);
        id
    }

    pub fn start(&mut self, state: u32) -> u32 {
        let closure = &self.singleton_closures[state as usize];
        if closure.len() == 1 {
            let singleton = closure[0];
            self.intern_singleton(singleton)
        } else {
            let states = closure.to_vec();
            self.intern(states)
        }
    }

    #[inline]
    pub fn step(&mut self, config: u32, byte: u8) -> u32 {
        if config == DEAD {
            return DEAD;
        }
        let slot = self.byte_slot[byte as usize];
        let cache_index = if slot == u16::MAX {
            None
        } else {
            Some(config as usize * self.transition_width + slot as usize)
        };
        if let Some(cache_index) = cache_index {
            let cached = self.transitions[cache_index];
            if cached != UNKNOWN {
                return cached;
            }
        }
        let target = if self.configs[config as usize].len() == 1 {
            let state = self.configs[config as usize][0];
            // L1 callers already carry the tokenizer's exact dense transition
            // table.  The finite projected scanner used to rediscover the same
            // edge through `Tokenizer::step` for every uncached singleton
            // `(config, byte)` pair.  Large p2 vocabularies generate hundreds
            // of thousands of these probes; use the O(1) dense row directly.
            let raw_target = self.input.flat_trans[state as usize * 256 + byte as usize];
            if raw_target == u32::MAX {
                DEAD
            } else {
                let closure = &self.singleton_closures[raw_target as usize];
                if closure.len() == 1 {
                    let singleton = closure[0];
                    self.intern_singleton(singleton)
                } else {
                    let states = closure.to_vec();
                    self.intern(states)
                }
            }
        } else {
            // `configs` are already epsilon-closed.  Calling `Tokenizer::step_all`
            // here closes the source set again, performs sparse transition
            // lookup, then closes the targets.  Reuse the dense transition table
            // and the precomputed singleton closures instead; unioning those
            // closures is exactly the same target configuration.
            let sources = &self.configs[config as usize];
            let mut states = Vec::with_capacity(sources.len().saturating_mul(2));
            for &state in sources.iter() {
                let raw_target = self.input.flat_trans[state as usize * 256 + byte as usize];
                if raw_target != u32::MAX {
                    states.extend_from_slice(&self.singleton_closures[raw_target as usize]);
                }
            }
            self.intern(states)
        };
        if let Some(cache_index) = cache_index {
            self.transitions[cache_index] = target;
        }
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
