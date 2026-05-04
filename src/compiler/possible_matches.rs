//! Possible-match tables for tokenizer states and vocab-prefix subtrees.

use std::collections::BTreeMap;
use std::rc::Rc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::grammar::flat::TerminalID;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

pub(crate) type PossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;
type PossibleMatchMap = FxHashMap<TerminalID, RangeSetBlaze<u32>>;

fn clone_token_entries(token_entries: &[(u32, Vec<u8>)]) -> Vec<(u32, Vec<u8>)> {
    token_entries
        .iter()
        .map(|(token_id, bytes)| (*token_id, bytes.clone()))
        .collect()
}

fn ordered_possible_matches(matches_for_state: Rc<PossibleMatchMap>) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
    match Rc::try_unwrap(matches_for_state) {
        Ok(map) => map.into_iter().collect(),
        Err(shared) => shared
            .iter()
            .map(|(&terminal, token_ids)| (terminal, token_ids.clone()))
            .collect(),
    }
}

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
    cache: FxHashMap<(usize, u32), Rc<PossibleMatchMap>>,
    reachable_cache: FxHashMap<usize, Rc<RangeSetBlaze<u32>>>,
    self_loop_bytes: FxHashMap<u32, U8Set>,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
}

impl<'a> PossibleMatchesComputer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
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
            let dfa_state = &self.tokenizer.dfa.states()[state_idx];
            let mut flat = Box::new([u32::MAX; 256]);
            for (b, &target) in dfa_state.transitions.iter() {
                flat[b as usize] = target;
            }
            self.flat_transitions[state_idx] = Some(flat);
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
        let self_loop_bytes = self.self_loop_bytes.entry(tokenizer_state).or_insert_with(|| {
            let state = &self.tokenizer.dfa.states()[tokenizer_state as usize];
            let mut bytes = U8Set::empty();
            for (byte, &target) in state.transitions.iter() {
                if target == tokenizer_state {
                    bytes.insert(byte);
                }
            }
            bytes
        });
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
                if self.can_skip_self_loop_subtree(child, current_state) {
                    continue;
                }
                let child_matches = self.possible_matches_for_node(child, current_state);
                merge_possible_match_maps(&mut result, child_matches.as_ref());
            }
        }

        let result = Rc::new(result);
        self.cache.insert(cache_key, Rc::clone(&result));
        result
    }
}

pub(crate) fn build_possible_matches_from_owned_token_entries(
    tokenizer: &Tokenizer,
    token_entries: Vec<(u32, Vec<u8>)>,
) -> PossibleMatchesByState {
    let trie = VocabPrefixTree::build_owned(
        token_entries
            .into_iter()
            .map(|(token_id, bytes)| (token_id as usize, bytes))
            .collect(),
    );

    let mut computer = PossibleMatchesComputer::new(tokenizer);
    collect_possible_matches_by_state(tokenizer, &trie.root, &mut computer)
}

pub(crate) fn collect_possible_matches_by_state(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    computer: &mut PossibleMatchesComputer<'_>,
) -> PossibleMatchesByState {
    collect_possible_matches_by_keys(
        tokenizer,
        root,
        computer,
        (0..tokenizer.num_states()).map(|state| (state, state)),
        tokenizer.num_states(),
    )
}

fn collect_possible_matches_by_keys(
    _tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    computer: &mut PossibleMatchesComputer<'_>,
    keyed_states: impl Iterator<Item = (u32, u32)>,
    total_keys: u32,
) -> PossibleMatchesByState {
    let mut possible_matches_by_state = BTreeMap::new();
    let root_key = root as *const VocabPrefixTreeNode as usize;
    for (index, (result_state_id, representative_state)) in keyed_states.enumerate() {
        let cache_key = (root_key, representative_state);
        let _ = computer.possible_matches_for_node(root, representative_state);
        let matches_for_state = computer
            .cache
            .remove(&cache_key)
            .expect("root possible-match map should be cached");
        possible_matches_by_state.insert(result_state_id, ordered_possible_matches(matches_for_state));
    }

    possible_matches_by_state
}
