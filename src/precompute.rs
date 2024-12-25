use crate::finite_automata::{GroupID, Regex};
use crate::glr;
use crate::glr::table::StateID;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use bitvec::prelude::BitVec;
use kdam::tqdm;
use crate::trie::{TrieNode};
use bimap::BiBTreeMap;
use bitvec::bitvec;
use crate::debug;

pub type TokenID = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenID(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Token {
    pub id: GroupID,
    pub width: usize,
}

pub struct ExecuteResult {
    pub matches: Vec<Token>,
    pub new_state: Option<usize>,
}

// TODO: get rid of this trait. Just implement it directly on the Tokenizer struct.
/// Trait defining the tokenizer behavior.
pub trait Tokenizer: Sized {
    /// Returns the initial state ID.
    fn initial_state_id(&self) -> usize;

    /// Executes the tokenizer on the given text starting from the specified state.
    /// Returns all possible next tokens (**not** a sequence of tokens).
    fn execute_from_state(&self, text: &[u8], state: usize) -> ExecuteResult;

    /// Returns the list of tokens accessible from the given state.
    fn tokens_accessible_from_state(&self, state: usize) -> Vec<TokenID>;

    /// Returns the maximum state ID in the DFA.
    fn max_state(&self) -> usize;

    /// Executes the tokenizer on the entire string and returns all possible token sequences and final states.
    fn execute_all_from_state(
        &self,
        text: &[u8],
        state: usize,
        state_map_root_arc: Arc<Mutex<TrieNode<GroupID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>>,
        llm_token_id: LLMTokenID,
        max_llm_token_id: usize,
    ) {
        let mut queue: BTreeMap<(usize, Option<usize>), BTreeMap<_, _>> = BTreeMap::new();
        // TODO: Should this be a bimap?
        let mut queue_positions: BTreeMap<*const Mutex<TrieNode<GroupID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>, (usize, Option<usize>)> = BTreeMap::new();
        let mut new_nodes_for_positions: BTreeMap<(usize, Option<usize>), Arc<Mutex<TrieNode<GroupID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>>> = BTreeMap::new();

        let root = state_map_root_arc.clone();

        // Initialize the queue with the starting state
        queue.insert((0, Some(state)), BTreeMap::from([(Arc::as_ptr(&root), root.clone())]));
        queue_positions.insert(Arc::as_ptr(&root), (0, Some(state)));

        while let Some(((position, maybe_state), nodes)) = queue.pop_first() {
            for (_, node) in nodes {
                if position == text.len() {
                    let mut node_guard = node.try_lock().unwrap();
                    assert!(!node_guard.value.0.contains_key(&llm_token_id));
                    node_guard.value.0.insert(llm_token_id, maybe_state.map(StateID));
                    if let Some(state) = maybe_state {
                        for possible_grammar_token_id in &self.tokens_accessible_from_state(state) {
                            node_guard.value.1.entry(*possible_grammar_token_id).or_insert_with(|| {
                                let mut bitset = BitVec::new();
                                bitset.resize(max_llm_token_id + 1, false);
                                bitset
                            }).set(llm_token_id.0, true);
                        }
                    } else {
                        node_guard.value.2.get_or_insert_with(|| {
                            let mut bitset = BitVec::new();
                            bitset.resize(max_llm_token_id + 1, false);
                            bitset
                        }).set(llm_token_id.0, true);
                    }
                    continue;
                }

                let remaining_text = &text[position..];
                let execute_result = self.execute_from_state(remaining_text, maybe_state.unwrap_or(0));

                // Process all matches
                for token in &execute_result.matches {
                    let new_position = position + token.width;
                    assert_ne!(token.width, 0);
                    assert!(new_position <= text.len());
                    let new_state: Option<usize> = None;
                    let mut node_guard = node.try_lock().unwrap();
                    if let Some(child) = node_guard.get(&token.id) {
                        let child_ptr = Arc::as_ptr(&child);
                        if let Some(&(child_position, child_state)) = queue_positions.get(&child_ptr) {
                            if (child_position, child_state) != (new_position, new_state) {
                                // Child exists and is already queued with different position or state
                                // Need to replace the child with a clone
                                let new_child = node_guard.replace_child_with_clone(&token.id);
                                queue_positions.insert(Arc::as_ptr(&new_child), (new_position, new_state));
                                queue.entry((new_position, new_state)).or_default().insert(Arc::as_ptr(&new_child), new_child.clone());
                            }
                        } else {
                            // Child exists but is not already queued
                            // Need to add it to the queue
                            queue_positions.insert(child_ptr, (new_position, new_state));
                            queue.entry((new_position, new_state)).or_default().insert(child_ptr, child.clone());
                        }
                    } else {
                        if let Some(new_node) = new_nodes_for_positions.get(&(new_position, new_state)) {
                            // A new node already exists for this position and state
                            // Add an edge from the current node to the new node
                            node_guard.insert(token.id, new_node.clone());
                        } else {
                            // A new node does not exist for this position and state
                            // Create a new node and add an edge from the current node to the new node
                            let new_node = Arc::new(Mutex::new(TrieNode::new((BTreeMap::new(), BTreeMap::new(), None))));
                            new_nodes_for_positions.insert((new_position, new_state), new_node.clone());
                            node_guard.insert(token.id, new_node.clone());
                            queue_positions.insert(Arc::as_ptr(&new_node), (new_position, new_state));
                            queue.entry((new_position, new_state)).or_default().insert(Arc::as_ptr(&new_node), new_node.clone());
                        }
                    }
                }
            }
        }
    }
}

/// Precomputes a map from state -> token sequence -> LLM token -> state.
pub fn precompute<'a>(
    tokenizer: &impl Tokenizer,
    llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    eof_llm_token_id: LLMTokenID,
    max_llm_token_id: usize,
) -> BTreeMap<StateID, TrieNode<TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>> {
    let mut result: BTreeMap<StateID, TrieNode<GroupID, _>> = BTreeMap::new();

    // Ensure the tokenizer doesn't match on empty strings
    debug!(2, "Ensuring tokenizer doesn't match on empty strings");
    let execute_result = tokenizer.execute_from_state(&[], 0);
    if !execute_result.matches.is_empty() {
        panic!("Tokenizer should not match on empty string. If it did, there would be infinitely many possible token sequences for any LLM token.");
    }

    // TODO: remove this
    let mut max_depths: BTreeMap<LLMTokenID, usize> = BTreeMap::new();

    debug!(2, "Precomputing in precompute");
    for state_id in tqdm!(0..tokenizer.max_state()) {
        let mut state_map_root_arc: Arc<Mutex<TrieNode<GroupID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>> = Arc::new(Mutex::new(TrieNode::new((BTreeMap::new(), BTreeMap::new(), None))));

        for (i, (llm_token, llm_token_id)) in llm_token_map.iter().enumerate() {
            debug!(1, "Precomputing for token {:?} ({:?}) ({})", llm_token_id, llm_token, i);
            tokenizer.execute_all_from_state(
                llm_token,
                state_id,
                state_map_root_arc.clone(),
                *llm_token_id,
                max_llm_token_id,
            );

            // TOOD: remove this
            // Compute maximum depth for this state and LLM token
            // let mut temp_root_arc = Arc::new(Mutex::new(TrieNode::new((BTreeMap::new(), BTreeMap::new(), None))));
            // tokenizer.execute_all_from_state(
            //     llm_token,
            //     state_id,
            //     temp_root_arc.clone(),
            //     *llm_token_id,
            //     max_llm_token_id,
            // );
            // let max_depth = temp_root_arc.try_lock().unwrap().max_depth();
            // if let Some(existing_depth) = max_depths.get_mut(llm_token_id) {
            //     *existing_depth = std::cmp::max(*existing_depth, max_depth);
            // } else {
            //     max_depths.insert(*llm_token_id, max_depth);
            // }
        }

        let state_map_root = state_map_root_arc.try_lock().unwrap().clone();
        result.insert(glr::table::StateID(state_id), state_map_root);
    }

    let mut max_depth_frequency: BTreeMap<usize, usize> = BTreeMap::new();
    for (llm_token_id, max_depth) in max_depths {
        *max_depth_frequency.entry(max_depth).or_default() += 1;
    }
    for (max_depth, frequency) in max_depth_frequency {
        println!("Max depth {} occurred {} times", max_depth, frequency);
    }
    result
}

pub(crate) fn precompute_add_eof(
    precomputed: &mut BTreeMap<StateID, TrieNode<TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>,
    eof_llm_token_id: LLMTokenID,
    eof_grammar_token_id: TokenID,
    max_llm_token_id: usize,
) {
    let mut bitset = bitvec![0; max_llm_token_id + 1];
    bitset.set(eof_llm_token_id.0, true);
    let node = precomputed.get_mut(&StateID(0)).expect("State 0 should exist");
    node.value.1.insert(eof_grammar_token_id, bitset);
}

impl Tokenizer for Regex {
    fn initial_state_id(&self) -> usize {
        0
    }

    fn execute_from_state(&self, text: &[u8], state: usize) -> ExecuteResult {
        let mut regex_state = self.init_to_state(state);
        regex_state.execute(text);

        let matches: Vec<_> = regex_state.matches.iter().map(|(&id, &width)| Token { id, width })
            // Filter out zero-width tokens
            .filter(|token| token.width != 0).collect();

        ExecuteResult {
            matches,
            new_state: if regex_state.done { None } else { Some(regex_state.current_state) },
        }
    }

    fn tokens_accessible_from_state(&self, state: usize) -> Vec<TokenID> {
        let regex_state = self.init_to_state(state);
        regex_state.possible_group_ids().iter().cloned().collect()
    }

    fn max_state(&self) -> usize {
        self.dfa.states.len()
    }
}

pub fn print_precomputed(precomputed: &BTreeMap<StateID, TrieNode<TokenID, (BTreeMap<LLMTokenID, Option<StateID>>, BTreeMap<TokenID, BitVec>, Option<BitVec>)>>) {
    println!("Precomputed:");
    for (tokenizer_state, root) in precomputed {
        println!("  Tokenizer state: {}", tokenizer_state.0);
        for node in TrieNode::all_nodes(Arc::new(Mutex::new(root.clone()))) {
            println!("    Node address: {:p}, value: {:?}", Arc::as_ptr(&node), node.try_lock().unwrap().value);
            // print edge values and destination addresses
            for (edge, dest) in node.try_lock().unwrap().children() {
                println!("      Edge value: {:?}, destination address: {:p}", edge, Arc::as_ptr(&dest));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charmap::TrieMap;
    use crate::finite_automata::{eat_u8, DFAState, Regex, DFA};
    use crate::u8set::U8Set;
    use crate::{groups, seq};
    use std::collections::{BTreeMap, BTreeSet};
    use bimap::BiBTreeMap;

    #[test]
    fn test_precompute() {
        let _tokenizer = groups![
            eat_u8(b'a'), // Token 0: 'a'
            eat_u8(b'b'), // Token 1: 'b'
            seq![eat_u8(b'a'), eat_u8(b'b')], // Token 2: 'ab'
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'c')], // Token 3: 'abc'
        ].build();

        let tokenizer = Regex {
            dfa: DFA {
                states: vec![
                    DFAState {
                        transitions: TrieMap::from_iter(vec![(b'a', 1), (b'b', 2)]),
                        finalizers: BTreeSet::new(),
                        possible_group_ids: BTreeSet::from([0, 1, 2, 3]),
                        group_id_to_u8set: BTreeMap::from([
                            (0, U8Set::from_bytes(b"a")),
                            (1, U8Set::from_bytes(b"b")),
                            (2, U8Set::from_bytes(b"a")),
                            (3, U8Set::from_bytes(b"a")),
                        ]),
                    },
                    DFAState {
                        transitions: TrieMap::from_iter(vec![(b'b', 3)]),
                        finalizers: BTreeSet::from([0]),
                        possible_group_ids: BTreeSet::from([0, 2, 3]),
                        group_id_to_u8set: BTreeMap::from([
                            (2, U8Set::from_bytes(b"b")),
                            (3, U8Set::from_bytes(b"b")),
                        ]),
                    },
                    DFAState {
                        transitions: TrieMap::new(),
                        finalizers: BTreeSet::from([1]),
                        possible_group_ids: BTreeSet::from([1]),
                        group_id_to_u8set: BTreeMap::new(),
                    },
                    DFAState {
                        transitions: TrieMap::from_iter(vec![(b'c', 4)]),
                        finalizers: BTreeSet::from([2]),
                        possible_group_ids: BTreeSet::from([2, 3]),
                        group_id_to_u8set: BTreeMap::from([(3, U8Set::from_bytes(b"c"))]),
                    },
                    DFAState {
                        transitions: TrieMap::new(),
                        finalizers: BTreeSet::from([3]),
                        possible_group_ids: BTreeSet::from([3]),
                        group_id_to_u8set: BTreeMap::new(),
                    },
                ],
                start_state: 0,
                non_greedy_finalizers: BTreeSet::new(),
            },
        };
        assert_eq!(_tokenizer, tokenizer);

        // Define the LLM tokens
        let llm_tokens: &[&[u8]] = &[b"a", b"b", b"c", b"ab", b"bc", b"abc"];
        let llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID> = llm_tokens.iter().enumerate().map(|(i, token)| (token.to_vec(), LLMTokenID(i))).collect();

        // Run precompute
        let max_llm_token_id = llm_tokens.len() + 1;
        let result = precompute(&tokenizer, &llm_token_map, LLMTokenID(max_llm_token_id), max_llm_token_id);
    }
}