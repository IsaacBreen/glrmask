use std::collections::BTreeMap;
use std::rc::Rc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::grammar::model::TerminalID;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

pub(crate) type PossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;
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
    cache: FxHashMap<(usize, u32), Rc<PossibleMatchMap>>,
    reachable_cache: FxHashMap<usize, Rc<RangeSetBlaze<u32>>>,
}

impl<'a> PossibleMatchesComputer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            cache: FxHashMap::default(),
            reachable_cache: FxHashMap::default(),
        }
    }

    fn reachable_for_node(&mut self, node: &VocabPrefixTreeNode) -> Rc<RangeSetBlaze<u32>> {
        let cache_key = node as *const VocabPrefixTreeNode as usize;
        if let Some(cached) = self.reachable_cache.get(&cache_key) {
            return Rc::clone(cached);
        }

        let reachable = Rc::new(reachable_u32(node));
        self.reachable_cache.insert(cache_key, Rc::clone(&reachable));
        reachable
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
            let mut state = tokenizer_state;
            let mut blocked = false;
            let reachable = self.reachable_for_node(child);

            for &byte in segment_bytes {
                let Some(next_state) = self.tokenizer.step(state, byte) else {
                    blocked = true;
                    break;
                };
                state = next_state;
                for matched in self.tokenizer.matched_terminals_iter(state) {
                    let existing = result.entry(matched).or_default();
                    merge_token_ids(existing, reachable.as_ref());
                }
            }

            if !blocked && !self.tokenizer.is_end(state) {
                let child_matches = self.possible_matches_for_node(child, state);
                merge_possible_match_maps(&mut result, child_matches.as_ref());
            }
        }

        let result = Rc::new(result);
        self.cache.insert(cache_key, Rc::clone(&result));
        result
    }
}

pub(crate) fn build_possible_matches_by_state(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> PossibleMatchesByState {
    let token_entries: Vec<(u32, Vec<u8>)> = token_bytes
        .iter()
        .map(|(token_id, bytes)| (*token_id, bytes.clone()))
        .collect();
    build_possible_matches_from_token_entries(tokenizer, &token_entries)
}

pub(crate) fn build_possible_matches_from_token_bytes(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> PossibleMatchesByState {
    let token_entries: Vec<(u32, Vec<u8>)> = token_bytes
        .iter()
        .map(|(token_id, bytes)| (*token_id, bytes.clone()))
        .collect();
    build_possible_matches_from_token_entries(tokenizer, &token_entries)
}

pub(crate) fn build_possible_matches_from_token_entries(
    tokenizer: &Tokenizer,
    token_entries: &[(u32, Vec<u8>)],
) -> PossibleMatchesByState {
    build_possible_matches_from_owned_token_entries(
        tokenizer,
        token_entries.iter().map(|(token_id, bytes)| (*token_id, bytes.clone())).collect(),
    )
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
    let mut possible_matches_by_state = BTreeMap::new();
    let root_key = root as *const VocabPrefixTreeNode as usize;
    for tokenizer_state in 0..tokenizer.num_states() {
        let cache_key = (root_key, tokenizer_state);
        let _ = computer.possible_matches_for_node(root, tokenizer_state);
        let matches_for_state = computer
            .cache
            .remove(&cache_key)
            .expect("root possible-match map should be cached");
        let ordered_matches: BTreeMap<TerminalID, RangeSetBlaze<u32>> =
            match Rc::try_unwrap(matches_for_state) {
                Ok(map) => map.into_iter().collect(),
                Err(shared) => shared
                    .iter()
                    .map(|(&terminal, token_ids)| (terminal, token_ids.clone()))
                    .collect(),
            };
        possible_matches_by_state.insert(
            tokenizer_state,
            ordered_matches,
        );
    }

    possible_matches_by_state
}

pub(crate) fn permute_possible_matches_in_place(
    possible_matches_by_state: &mut PossibleMatchesByState,
    token_perm: &[u32],
) {
    for matches_by_terminal in possible_matches_by_state.values_mut() {
        for token_ids in matches_by_terminal.values_mut() {
            let mut mapped: Vec<u32> = token_ids
                .iter()
                .filter_map(|token_id| token_perm.get(token_id as usize).copied())
                .collect();
            mapped.sort_unstable();
            mapped.dedup();
            *token_ids = RangeSetBlaze::from_iter(mapped.into_iter().map(|token_id| token_id..=token_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::bytes;
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn test_possible_matches_supports_distinct_bytes_for_same_internal_token() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b")]);
        let token_entries = vec![(0u32, b"a".to_vec()), (0u32, b"b".to_vec())];

        let possible_matches =
            build_possible_matches_from_token_entries(&tokenizer, &token_entries);
        let start_matches = possible_matches
            .get(&tokenizer.initial_state())
            .expect("start state should have possible matches");

        assert_eq!(
            start_matches.get(&0),
            Some(&RangeSetBlaze::from_iter([0u32..=0u32]))
        );
        assert_eq!(
            start_matches.get(&1),
            Some(&RangeSetBlaze::from_iter([0u32..=0u32]))
        );
    }
}
