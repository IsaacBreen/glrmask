//! Possible-match tables for tokenizer states and vocab-prefix subtrees.

use std::rc::Rc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::grammar::flat::TerminalID;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;

type PossibleMatchMap = FxHashMap<TerminalID, RangeSetBlaze<u32>>;

fn reachable_u32(node: &VocabPrefixTreeNode) -> RangeSetBlaze<u32> {
    let mut out = RangeSetBlaze::new();
    for range in node.reachable_token_ids().ranges() {
        out.ranges_insert(*range.start() as u32..=*range.end() as u32);
    }
    out
}

fn merge_token_ids(into: &mut RangeSetBlaze<u32>, other: &RangeSetBlaze<u32>) {
    *into |= other;
}

fn merge_possible_match_maps(into: &mut PossibleMatchMap, other: &PossibleMatchMap) {
    for (terminal, token_ids) in other {
        let existing = into.entry(*terminal).or_default();
        merge_token_ids(existing, token_ids);
    }
}

pub(crate) struct PossibleMatchesComputer<'a> {
    tokenizer: &'a Tokenizer,
    canonical_state: Option<&'a [u32]>,
    cache: FxHashMap<(usize, u32), Rc<PossibleMatchMap>>,
    reachable_cache: FxHashMap<usize, Rc<RangeSetBlaze<u32>>>,
    self_loop_bytes: FxHashMap<u32, U8Set>,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
}

impl<'a> PossibleMatchesComputer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer) -> Self {
        Self::new_with_canonical_state(tokenizer, None)
    }

    pub(crate) fn new_with_canonical_state(
        tokenizer: &'a Tokenizer,
        canonical_state: Option<&'a [u32]>,
    ) -> Self {
        Self {
            tokenizer,
            canonical_state,
            cache: FxHashMap::default(),
            reachable_cache: FxHashMap::default(),
            self_loop_bytes: FxHashMap::default(),
            flat_transitions: vec![None; tokenizer.num_states() as usize],
        }
    }

    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            self.flat_transitions[state_idx] = Some(self.tokenizer.transition_row(state));
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        if next == u32::MAX { None } else { Some(next) }
    }

    fn reachable_for_node(&mut self, node: &VocabPrefixTreeNode) -> Rc<RangeSetBlaze<u32>> {
        let cache_key = node as *const VocabPrefixTreeNode as usize;
        if let Some(cached) = self.reachable_cache.get(&cache_key) {
            Rc::clone(cached)
        } else {
            let reachable = Rc::new(reachable_u32(node));
            self.reachable_cache.insert(cache_key, Rc::clone(&reachable));
            reachable
        }
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: u32,
    ) -> bool {
        let self_loop_bytes = self
            .self_loop_bytes
            .entry(tokenizer_state)
            .or_insert_with(|| self.tokenizer.self_loop_bytes(tokenizer_state));
        U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes)
    }

    pub(crate) fn possible_matches_for_node(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: u32,
    ) -> Rc<PossibleMatchMap> {
        let cache_key = (node as *const VocabPrefixTreeNode as usize, tokenizer_state);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Rc::clone(cached);
        }

        let mut result = PossibleMatchMap::default();

        // This intentionally includes the token ending exactly at `node` before
        // recursing into child segments, so the recursive part only adds longer
        // continuations.
        if node.has_token() {
            let token_id = node.token_id() as u32;
            for terminal in self.tokenizer.matched_terminals_iter(tokenizer_state) {
                result.entry(terminal).or_default().insert(token_id);
            }
        }

        for (segment_bytes, child) in node.iter_children() {
            let mut current_state = tokenizer_state;
            let mut segment_blocked = false;
            let reachable = self.reachable_for_node(child);

            for &byte in segment_bytes {
                let Some(next_state) = self.fast_step(current_state, byte) else {
                    segment_blocked = true;
                    break;
                };
                current_state = next_state;
                for terminal in self.tokenizer.matched_terminals_iter(current_state) {
                    let existing = result.entry(terminal).or_default();
                    merge_token_ids(existing, reachable.as_ref());
                }
            }

            if !segment_blocked && !self.tokenizer.is_end(current_state) {
                let descend_state = self
                    .canonical_state
                    .and_then(|map| map.get(current_state as usize).copied())
                    .unwrap_or(current_state);
                if self.can_skip_self_loop_subtree(child, descend_state) {
                    continue;
                }
                let child_matches = self.possible_matches_for_node(child, descend_state);
                merge_possible_match_maps(&mut result, child_matches.as_ref());
            }
        }

        let result = Rc::new(result);
        self.cache.insert(cache_key, Rc::clone(&result));
        result
    }
}
