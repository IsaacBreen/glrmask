#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::grammar::model::{GrammarDef, TerminalID};

pub(crate) type PossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;

#[derive(Debug, Default)]
struct VocabTrieNode {
    token_id: Option<u32>,
    reachable_token_ids: RangeSetBlaze<u32>,
    children: BTreeMap<u8, VocabTrieNode>,
}

impl VocabTrieNode {
    fn insert(&mut self, token_id: u32, bytes: &[u8]) {
        self.reachable_token_ids |= RangeSetBlaze::from_iter([token_id..=token_id]);
        if bytes.is_empty() {
            self.token_id = Some(token_id);
            return;
        }

        self.children
            .entry(bytes[0])
            .or_default()
            .insert(token_id, &bytes[1..]);
    }
}

fn merge_possible_match_maps(
    into: &mut BTreeMap<TerminalID, RangeSetBlaze<u32>>,
    other: BTreeMap<TerminalID, RangeSetBlaze<u32>>,
) {
    for (terminal, token_ids) in other {
        into.entry(terminal)
            .and_modify(|existing| *existing = existing.clone() | token_ids.clone())
            .or_insert(token_ids);
    }
}

fn possible_matches_for_node(
    node: &VocabTrieNode,
    tokenizer: &Tokenizer,
    tokenizer_state: u32,
    cache: &mut HashMap<(usize, u32), BTreeMap<TerminalID, RangeSetBlaze<u32>>>,
) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
    let cache_key = (node as *const VocabTrieNode as usize, tokenizer_state);
    if let Some(cached) = cache.get(&cache_key) {
        return cached.clone();
    }

    let mut result = BTreeMap::new();

    if let Some(token_id) = node.token_id {
        for terminal in tokenizer.all_matched_terminals(tokenizer_state) {
            result
                .entry(terminal)
                .and_modify(|existing: &mut RangeSetBlaze<u32>| {
                    *existing = existing.clone() | RangeSetBlaze::from_iter([token_id..=token_id])
                })
                .or_insert_with(|| RangeSetBlaze::from_iter([token_id..=token_id]));
        }
    }

    for (&byte, child) in &node.children {
        let exec = tokenizer.execute_from_state(&[byte], tokenizer_state);

        for matched in &exec.matches {
            result
                .entry(matched.id)
                .and_modify(|existing| *existing = existing.clone() | child.reachable_token_ids.clone())
                .or_insert_with(|| child.reachable_token_ids.clone());
        }

        if let Some(end_state) = exec.end_state {
            let accessible: BTreeSet<_> = tokenizer
                .tokens_accessible_from_state(end_state)
                .into_iter()
                .collect();
            let matches_here: BTreeSet<_> = exec.matches.iter().map(|matched| matched.id).collect();
            let possible_new_matches = &accessible - &matches_here;
            if !possible_new_matches.is_empty() {
                merge_possible_match_maps(
                    &mut result,
                    possible_matches_for_node(child, tokenizer, end_state, cache),
                );
            }
        }
    }

    cache.insert(cache_key, result.clone());
    result
}

pub(crate) fn build_possible_matches_by_state(
    grammar: &GrammarDef,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> PossibleMatchesByState {
    let _ = grammar;
    let mut trie = VocabTrieNode::default();
    for (token_id, bytes) in &vocab.entries {
        trie.insert(*token_id, bytes);
    }

    let mut cache = HashMap::new();
    let mut possible_matches_by_state = BTreeMap::new();
    for tokenizer_state in 0..tokenizer.num_states() {
        possible_matches_by_state.insert(
            tokenizer_state,
            possible_matches_for_node(&trie, tokenizer, tokenizer_state, &mut cache),
        );
    }
    possible_matches_by_state
}
