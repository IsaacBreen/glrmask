//! Exact scanner for projected L1 construction.

use std::sync::Arc;
use rustc_hash::FxHashMap;

use super::BuildInput;
use crate::automata::lexer::observation::{ConfigActionCache, ConfigInterner, EpsilonImageScratch};
use crate::automata::lexer::tokenizer::SingletonEpsilonClosures;
use crate::automata::lexer::Lexer;

pub(super) const DEAD: u32 = u32::MAX;

pub(super) struct Scanner<'a> {
    input: BuildInput<'a>,
    action: ConfigActionCache,
    pub signatures: Vec<Vec<u32>>,
    signature_ids: FxHashMap<Vec<u32>, u32>,
    config_signature: Vec<u32>,
    singleton_closures: Arc<SingletonEpsilonClosures>,
    image: Option<EpsilonImageScratch>,
}

impl<'a> Scanner<'a> {
    pub fn new(input: BuildInput<'a>) -> Self {
        let relevant_bytes = input.vocab.relevant_bytes();
        let state_count = input.tokenizer.num_states() as usize;
        Self {
            input,
            action: ConfigActionCache::new(
                relevant_bytes.iter().copied(),
                |byte| input.transitions_by_byte.map_or(byte as usize, |index| index.class(byte)),
                Some(state_count),
            ),
            signatures: vec![Vec::new()],
            signature_ids: FxHashMap::from_iter([(Vec::new(), 0)]),
            config_signature: Vec::new(),
            singleton_closures: input.tokenizer.all_singleton_epsilon_closures(),
            image: None,
        }
    }

    #[inline]
    pub fn configs(&self) -> &ConfigInterner { self.action.configs() }

    fn register_signature(&mut self, id: u32, inserted: bool) -> u32 {
        if !inserted { return id; }
        let signature = if let [state] = self.action.configs()[id as usize].as_ref() {
            super::super::collect_active_terminal_signature(
                self.input.tokenizer, *state, self.input.active_terminals,
            )
        } else {
            let mut signature = self.action.configs()[id as usize]
                .iter()
                .flat_map(|&state| {
                    super::super::collect_active_terminal_signature(
                        self.input.tokenizer, state, self.input.active_terminals,
                    )
                })
                .collect::<Vec<_>>();
            signature.sort_unstable();
            signature.dedup();
            signature
        };
        let next = self.signatures.len() as u32;
        let signature_id = *self.signature_ids.entry(signature.clone()).or_insert_with(|| {
            self.signatures.push(signature);
            next
        });
        self.config_signature.push(signature_id);
        id
    }

    pub fn start(&mut self, state: u32) -> u32 {
        let (id, inserted) = self.action.intern_canonical_slice(&self.singleton_closures[state as usize]);
        self.register_signature(id, inserted)
    }

    #[inline]
    pub fn step(&mut self, config: u32, byte: u8) -> u32 {
        if config == DEAD { return DEAD; }
        if let Some(target) = self.action.get(config, byte) {
            return target;
        }
        let (id, inserted) = if let [state] = self.action.configs()[config as usize].as_ref() {
            // Direct singleton lookup retains the inexpensive closure fast path.
            let raw_target = self.input.flat_trans[*state as usize * 256 + byte as usize];
            if raw_target == DEAD {
                (DEAD, false)
            } else {
                self.action.intern_canonical_slice(&self.singleton_closures[raw_target as usize])
            }
        } else {
            // Unrestricted L1 configurations are already epsilon-closed.
            // Filter the terminal signature, not the source/target state set.
            let flat = self.input.flat_trans;
            let image = self.image.get_or_insert_with(|| EpsilonImageScratch::new(flat.len() / 256));
            let states = image.step(
                self.action.configs()[config as usize].as_ref(), byte, true,
                self.singleton_closures.as_ref(), None,
                |state, byte| flat[state as usize * 256 + byte as usize],
            );
            self.action.intern_canonical_slice(states)
        };
        let target = self.register_signature(id, inserted);
        self.action.set(config, byte, target);
        target
    }

    pub fn step_bytes(&mut self, mut config: u32, bytes: &[u8]) -> u32 {
        for &byte in bytes {
            config = self.step(config, byte);
            if config == DEAD { break; }
        }
        config
    }

    #[inline]
    pub fn singleton_state(&self, config: u32) -> Option<u32> {
        let states = self.action.configs().get(config as usize)?;
        (states.len() == 1).then_some(states[0])
    }

    #[inline]
    pub fn signature(&self, config: u32) -> u32 {
        if config == DEAD { 0 } else { self.config_signature[config as usize] }
    }
}
