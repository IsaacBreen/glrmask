#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::grammar::model::{GrammarDef, TerminalID};
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

pub(crate) type PossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;
type PossibleMatchMap = BTreeMap<TerminalID, RangeSetBlaze<u32>>;

fn token_range(token_id: u32) -> RangeSetBlaze<u32> {
    RangeSetBlaze::from_iter([token_id..=token_id])
}

fn reachable_u32(node: &VocabPrefixTreeNode) -> RangeSetBlaze<u32> {
    let mut out = RangeSetBlaze::new();
    for token_id in node.reachable_token_ids().iter() {
        out.insert(token_id as u32);
    }
    out
}

fn merge_possible_match_maps(
    into: &mut PossibleMatchMap,
    other: &PossibleMatchMap,
) {
    for (terminal, token_ids) in other {
        into.entry(*terminal)
            .and_modify(|existing| *existing = existing.clone() | token_ids.clone())
            .or_insert_with(|| token_ids.clone());
    }
}

pub(crate) struct PossibleMatchesComputer<'a> {
    tokenizer: &'a Tokenizer,
    cache: HashMap<(usize, u32), Rc<PossibleMatchMap>>,
}

impl<'a> PossibleMatchesComputer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            cache: HashMap::new(),
        }
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

        let mut result = PossibleMatchMap::new();

        // This intentionally includes the token ending exactly at `node`.
        // sep1's `possible_matches(node, state)` does the same before recursing
        // into child segments, so the recursive part only adds longer continuations.
        if node.has_token() {
            let token_id = node.token_id() as u32;
            let token_range = token_range(token_id);
            for terminal in self.tokenizer.all_matched_terminals(tokenizer_state) {
                result
                    .entry(terminal)
                    .and_modify(|existing: &mut RangeSetBlaze<u32>| {
                        *existing = existing.clone() | token_range.clone()
                    })
                    .or_insert_with(|| token_range.clone());
            }
        }

        for (segment_bytes, child) in node.iter_children() {
            let exec = self.tokenizer.execute_from_state(segment_bytes, tokenizer_state);
            let reachable = reachable_u32(child);

            for matched in &exec.matches {
                result
                    .entry(matched.id)
                    .and_modify(|existing| *existing = existing.clone() | reachable.clone())
                    .or_insert_with(|| reachable.clone());
            }

            if let Some(end_state) = exec.end_state {
                let child_matches = self.possible_matches_for_node(child, end_state);
                merge_possible_match_maps(&mut result, child_matches.as_ref());
            }
        }

        let result = Rc::new(result);
        self.cache.insert(cache_key, Rc::clone(&result));
        result
    }
}

pub(crate) fn build_possible_matches_by_state(
    grammar: &GrammarDef,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> PossibleMatchesByState {
    let _ = grammar;
    build_possible_matches_from_token_bytes(
        tokenizer,
        &vocab.entries,
    )
}

pub(crate) fn build_possible_matches_from_token_bytes(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> PossibleMatchesByState {
    let trie = VocabPrefixTree::build(
        &token_bytes
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect::<Vec<_>>(),
    );

    let mut computer = PossibleMatchesComputer::new(tokenizer);
    let mut possible_matches_by_state = BTreeMap::new();
    for tokenizer_state in 0..tokenizer.num_states() {
        possible_matches_by_state.insert(
            tokenizer_state,
            computer
                .possible_matches_for_node(&trie.root, tokenizer_state)
                .as_ref()
                .clone(),
        );
    }
    possible_matches_by_state
}
