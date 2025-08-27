// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::mem;
use crate::datastructures::gss::{disallow_llm_tokens_and_prune_arc, get_roots, print_gss_forest, reset_terminals};
use crate::datastructures::gss::{map_allowed_terminals_tokenizer_states, prune_disallowed_terminals};
use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::{BitOrAssign};
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;
use std::cell::RefCell;

use bimap::BiBTreeMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, calculate_final_stats2, dump_precompute_trie_recursive, print_precompute_stats, print_precompute_stats2, PrecomputeStats};
use crate::datastructures::arena::NodeId;
use crate::datastructures::gss::{allow_only_llm_tokens_and_prune_arc, disallow_terminals_and_prune_arc, gather_gss_stats, merge_trie2_nodes_if_needed, reset_llm_tokens, GSSNode, GSSPrintConfig, LLMTokenBV, PrecomputedNodeContents, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::glr::parser::{BelowBottomReductionMode, GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, ProcessDefaultReductionsAdvancedConfig, ProcessTokenAdvancedConfig};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use kdam::{tqdm, BarExt};
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};
use crate::datastructures::gss::Acc;
use crate::glr::table::StateID;
use crate::glr::analyze::compute_terminal_follow_sets;
use crate::glr::grammar::Terminal;
use std::ops::{BitAnd};
use crate::glr::items::{LRMode, LR_MODE};
use crate::interface::CompiledGrammar;
use crate::profiler::{GSS_LOGGING_ENABLED, PROGRESS_BAR_ENABLED};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use serde_json::Value as SerdeValue;

const MERGE_THRESHOLD: usize = 20;

pub type PrecomputeGraph =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type Precomputed = BTreeMap<TokenizerStateID, NodeId>;

pub type PrecomputeGraph2 = Trie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;
pub type Precomputed2 = BTreeMap<TokenizerStateID, NodeId>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMVocab {
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub(crate) max_original_llm_token_id: usize,
    pub(crate) original_to_internal_id_bimap: BiBTreeMap<usize, usize>,
    pub(crate) internal_max_llm_token: usize
}

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub(crate) tokenizer:        Regex,
    pub(crate) parser:           GLRParser,
    pub(crate) precomputed_arena: PrecomputeGraph,
    pub(crate) precomputed:      Precomputed,
    pub(crate) precomputed2_arena: PrecomputeGraph2,
    pub(crate) precomputed2:     Precomputed2,
    pub(crate) llm_vocab:        Arc<LLMVocab>,
    pub(crate) token_name_map:   BiBTreeMap<Terminal, usize>,
    pub(crate) possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        assert_eq!(self.precomputed.len(), other.precomputed.len());
        for ((sid1, root1), (sid2, root2)) in self.precomputed.iter().zip(other.precomputed.iter()) {
            assert_eq!(sid1, sid2);
            // This requires Trie to implement PartialEq, which compares from the roots.
            // assert!(self.precomputed_arena.compare_from_roots(*root1, &other.precomputed_arena, *root2));
            todo!();
        }
        assert_eq!(self.precomputed2.len(), other.precomputed2.len());
        for ((sid1, root1), (sid2, root2)) in self.precomputed2.iter().zip(other.precomputed2.iter()) {
            assert_eq!(sid1, sid2);
            // assert!(self.precomputed2_arena.compare_from_roots(*root1, &other.precomputed2_arena, *root2));
            todo!();
        }
        assert_eq!(self.llm_vocab.llm_token_map, other.llm_vocab.llm_token_map);
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.llm_vocab.max_original_llm_token_id, other.llm_vocab.max_original_llm_token_id);
        assert_eq!(self.llm_vocab.original_to_internal_id_bimap, other.llm_vocab.original_to_internal_id_bimap);
        assert_eq!(self.llm_vocab.internal_max_llm_token, other.llm_vocab.internal_max_llm_token);
        assert_eq!(self.possible_matches, other.possible_matches);
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("precomputed_arena".to_string(), self.precomputed_arena.to_json());
        obj.insert("precomputed".to_string(), self.precomputed.to_json());
        obj.insert("precomputed2_arena".to_string(), self.precomputed2_arena.to_json());
        obj.insert("precomputed2".to_string(), self.precomputed2.to_json());
        obj.insert("llm_token_map".to_string(), self.llm_vocab.llm_token_map.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("max_original_llm_token_id".to_string(), self.llm_vocab.max_original_llm_token_id.to_json());
        obj.insert("original_to_internal_id_bimap".to_string(), self.llm_vocab.original_to_internal_id_bimap.to_json());
        obj.insert("internal_max_llm_token".to_string(), self.llm_vocab.internal_max_llm_token.to_json());
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let tokenizer = obj.remove("tokenizer").ok_or_else(|| "Missing field tokenizer".to_string())
                                   .and_then(Regex::from_json)?;
                let parser = obj.remove("parser").ok_or_else(|| "Missing field parser".to_string())
                                .and_then(GLRParser::from_json)?;
                let precomputed_arena = obj.remove("precomputed_arena").ok_or_else(|| "Missing field precomputed_arena".to_string())
                                     .and_then(PrecomputeGraph::from_json)?;
                let precomputed = obj.remove("precomputed").ok_or_else(|| "Missing field precomputed".to_string())
                                     .and_then(Precomputed::from_json)?;
                let precomputed2_arena = obj.remove("precomputed2_arena").ok_or_else(|| "Missing field precomputed2_arena".to_string())
                                     .and_then(PrecomputeGraph2::from_json)?;
                let precomputed2 = obj.remove("precomputed2").ok_or_else(|| "Missing field precomputed2".to_string())
                                     .and_then(Precomputed2::from_json)?;

                let llm_token_map = obj.remove("llm_token_map").ok_or_else(|| "Missing field llm_token_map".to_string())
                                       .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let token_name_map = obj.remove("token_name_map").ok_or_else(|| "Missing field token_name_map".to_string())
                                        .and_then(|n| BiBTreeMap::<Terminal, usize>::from_json(n))?;
                let max_original_llm_token_id = obj.remove("max_original_llm_token_id").ok_or_else(|| "Missing field max_original_llm_token_id".to_string())
                                                   .and_then(usize::from_json)?;
                let original_to_internal_id_bimap = obj.remove("original_to_internal_id_bimap").ok_or_else(|| "Missing field original_to_internal_id_bimap".to_string())
                                                       .and_then(|n| BiBTreeMap::<usize, usize>::from_json(n))?;
                let internal_max_llm_token = obj.remove("internal_max_llm_token").ok_or_else(|| "Missing field internal_max_llm_token".to_string())
                                                  .and_then(usize::from_json)?;
                let possible_matches = obj.remove("possible_matches").ok_or_else(|| "Missing field possible_matches".to_string())
                                          .and_then(|n| BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::from_json(n))?;
                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed_arena,
                    precomputed,
                    precomputed2_arena,
                    precomputed2,
                    llm_vocab: Arc::new(LLMVocab { llm_token_map, max_original_llm_token_id, original_to_internal_id_bimap, internal_max_llm_token }),
                    token_name_map,
                    possible_matches,
                })
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}

type NormalizedPath = Vec<(usize, StateID)>;

/// Samples a single normalized path by performing a random walk from the root.
fn sample_normalized_path(
    trie: &PrecomputeGraph2,
    root_id: NodeId,
    rng: &mut impl Rng,
    max_len: usize,
) -> Option<NormalizedPath> {
    let mut current_node_id = root_id;
    let mut path = NormalizedPath::new();
    let mut current_k = 0;
    let mut bv = trie.get_node(root_id).value.live_tokens.clone();

    while path.len() < max_len {
        let current_node = trie.get_node(current_node_id);
        let can_terminate = current_node.value.end;
        let can_continue = !current_node.children.is_empty();

        if !can_continue {
            return if can_terminate { Some(path) } else { None };
        }

        if can_terminate && rng.gen_bool(0.2) { // 20% chance to terminate at an end node
            return Some(path);
        }

        let all_outgoing_edges: Vec<_> = current_node
            .children
            .iter()
            .flat_map(|(ek, dest_map)| {
                dest_map.iter().map(move |(&dest_id, edge_bv)| (ek.clone(), dest_id, edge_bv.clone()))
            })
            .collect();

        if all_outgoing_edges.is_empty() {
            return if current_node.value.end { Some(path) } else { None };
        }

        let (ek, dest_id, edge_bv) = all_outgoing_edges.choose(rng)?;

        bv &= edge_bv;
        if bv.is_empty() {
            return None; // Path became invalid
        }

        let (k, sid_opt) = ek;
        current_k += k;
        if let Some(sid) = sid_opt {
            path.push((current_k, *sid));
            current_k = 0;
        }

        current_node_id = *dest_id;
    }

    Some(path)
}

/// For a given normalized path, computes the union of LLM token bitvectors for all
/// possible ways to traverse that path in the trie.
fn get_bv_for_normalized_path(
    trie: &PrecomputeGraph2,
    root_id: NodeId,
    path: &NormalizedPath,
) -> LLMTokenBV {
    // State: (current_node_id, path_segment_index, accumulated_k, current_bv)
    let mut q: VecDeque<(NodeId, usize, usize, LLMTokenBV)> = VecDeque::new();
    let mut final_bv = LLMTokenBV::zeros();

    let initial_bv = trie.get_node(root_id).value.live_tokens.clone();
    q.push_back((root_id, 0, 0, initial_bv.clone()));

    // To handle cycles and redundant exploration
    let mut visited: HashMap<(NodeId, usize, usize), LLMTokenBV> = HashMap::new();
    visited.insert((root_id, 0, 0), initial_bv);

    while let Some((node_id, path_idx, k_so_far, bv)) = q.pop_front() {
        // Check if we've completed the path
        if path_idx == path.len() {
            // We have successfully traversed the path. Now we need to reach an `end` node from here
            // with only `(k, None)` edges.
            let end_bv = find_end_bv_from_node_via_none_edges(trie, node_id, bv);
            final_bv |= &end_bv;
            continue;
        }

        let (target_k, target_sid) = path[path_idx];

        // Explore children
        let node = trie.get_node(node_id);
        for (ek, dest_map) in &node.children {
            for (&dest_id, edge_bv) in dest_map {
                let new_bv = &bv & edge_bv;
                if new_bv.is_empty() { continue; }

                let (k, sid_opt) = ek;
                let new_k = k_so_far + *k;

                if let Some(sid) = sid_opt {
                    if new_k == target_k && sid == &target_sid {
                        // Matched a path segment. Advance.
                        let visited_key = (dest_id, path_idx + 1, 0);
                        if let Some(existing_bv) = visited.get_mut(&visited_key) {
                            let diff = &new_bv - &*existing_bv;
                            if !diff.is_empty() {
                                *existing_bv |= &diff;
                                q.push_back((dest_id, path_idx + 1, 0, diff));
                            }
                        } else {
                            visited.insert(visited_key, new_bv.clone());
                            q.push_back((dest_id, path_idx + 1, 0, new_bv));
                        }
                    }
                } else { // sid_opt is None
                    if new_k <= target_k {
                        // Continue accumulating k
                        let visited_key = (dest_id, path_idx, new_k);
                        if let Some(existing_bv) = visited.get_mut(&visited_key) {
                            let diff = &new_bv - &*existing_bv;
                            if !diff.is_empty() {
                                *existing_bv |= &diff;
                                q.push_back((dest_id, path_idx, new_k, diff));
                            }
                        } else {
                            visited.insert(visited_key, new_bv.clone());
                            q.push_back((dest_id, path_idx, new_k, new_bv));
                        }
                    }
                }
            }
        }
    }
    final_bv
}

/// Helper to find the union of BVs for all paths from a start node to any `end` node
/// that consist solely of `(k, None)` edges.
fn find_end_bv_from_node_via_none_edges(
    trie: &PrecomputeGraph2,
    start_node_id: NodeId,
    initial_bv: LLMTokenBV,
) -> LLMTokenBV {
    let mut end_bv = LLMTokenBV::zeros();
    let mut q = VecDeque::new();
    q.push_back((start_node_id, initial_bv));
    let mut visited: HashMap<NodeId, LLMTokenBV> = HashMap::new();

    while let Some((node_id, bv)) = q.pop_front() {
        let node = trie.get_node(node_id);
        if node.value.end {
            end_bv |= &bv;
        }

        for (ek, dest_map) in &node.children {
            let (_k, sid_opt) = ek;
            if sid_opt.is_none() { // Only (k, None) edges
                for (&dest_id, edge_bv) in dest_map {
                    let new_bv = &bv & edge_bv;
                    if new_bv.is_empty() { continue; }

                    if let Some(existing_bv) = visited.get_mut(&dest_id) {
                        let diff = &new_bv - &*existing_bv;
                        if !diff.is_empty() {
                            *existing_bv |= &diff;
                            q.push_back((dest_id, diff));
                        }
                    } else {
                        visited.insert(dest_id, new_bv.clone());
                        q.push_back((dest_id, new_bv));
                    }
                }
            }
        }
    }
    end_bv
}

/// Checks for semantic equivalence between two `precompute2` trees.
pub fn are_precompute2_trees_equivalent(
    trie_a: &PrecomputeGraph2, root_a: NodeId,
    trie_b: &PrecomputeGraph2, root_b: NodeId
) -> bool {
    // Stochastic version
    // if trie_a.get_node(root_a) == trie_b.get_node(root_b) && trie_a.compare_from_roots(root_a, trie_b, root_b) {
    if trie_a.get_node(root_a) == trie_b.get_node(root_b) && todo!() {
        return true;
    }

    const NUM_SAMPLES: usize = 100;
    const MAX_PATH_LEN: usize = 32;
    let mut rng = rand::thread_rng();

    // Sample from A, check in B
    for _ in 0..NUM_SAMPLES {
        if let Some(path) = sample_normalized_path(trie_a, root_a, &mut rng, MAX_PATH_LEN) {
            let bv_a = get_bv_for_normalized_path(trie_a, root_a, &path);
            if bv_a.is_empty() { continue; } // Skip trivial paths
            let bv_b = get_bv_for_normalized_path(trie_b, root_b, &path);
            if bv_a != bv_b {
                println!("\n--- Precompute2 Equivalence Mismatch ---");
                println!("Path sampled from Tree A:");
                println!("  Path: {:?}", path);
                println!("  BV from A: {:?}", bv_a);
                println!("  BV from B: {:?}", bv_b);
                println!("  Difference (A ^ B): {:?}", bv_a.symmetric_difference(&bv_b));
                return false;
            }
        }
    }

    // Sample from B, check in A
    for _ in 0..NUM_SAMPLES {
        if let Some(path) = sample_normalized_path(trie_b, root_b, &mut rng, MAX_PATH_LEN) {
            let bv_b = get_bv_for_normalized_path(trie_b, root_b, &path);
            if bv_b.is_empty() { continue; } // Skip trivial paths
            let bv_a = get_bv_for_normalized_path(trie_a, root_a, &path);
            if bv_a != bv_b {
                println!("\n--- Precompute2 Equivalence Mismatch ---");
                println!("Path sampled from Tree B:");
                println!("  Path: {:?}", path);
                println!("  BV from A: {:?}", bv_a);
                println!("  BV from B: {:?}", bv_b);
                println!("  Difference (A ^ B): {:?}", bv_a.symmetric_difference(&bv_b));
                return false;
            }
        }
    }

    true
}


pub fn precompute(
    tokenizer: &Regex,
    parser: Option<&GLRParser>,
    llm_vocab: Option<Arc<LLMVocab>>,
    internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    token_name_map: Option<&BiBTreeMap<Terminal, usize>>,
    internal_max_llm_token: usize,
    terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
) -> (PrecomputeGraph, Precomputed) {
    todo!()
}

pub fn precompute2(
    precomputed_arena: &PrecomputeGraph,
    precomputed: &Precomputed,
    tokenizer: &Regex,
    parser: Option<&GLRParser>,
    llm_vocab: Option<Arc<LLMVocab>>,
    internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
) -> (PrecomputeGraph2, Precomputed2) {
    todo!()
}


impl GrammarConstraint {
    pub fn from_compiled_grammar(
        compiled_grammar: CompiledGrammar,
        llm_token_map: LLMTokenMap,
        _eof_token_id: LLMTokenID,
        max_original_llm_token_id: usize,
    ) -> Self {
        let token_name_map = compiled_grammar.definition.terminal_to_group_id().clone();

        Self::new(
            compiled_grammar.tokenizer,
            compiled_grammar.glr_parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
        )
    }

    pub(crate) fn setup_llm_token_mappings(
        original_llm_token_map: &LLMTokenMap,
    ) -> BiBTreeMap<usize, usize>
    {
        let mut sorted_tokens_with_original_ids: Vec<(Vec<u8>, LLMTokenID)> = original_llm_token_map
            .iter()
            .map(|(bytes, original_id)| (bytes.clone(), *original_id))
            .collect();
        sorted_tokens_with_original_ids.sort_by(|(bytes_a, _), (bytes_b, _)| bytes_a.cmp(bytes_b));

        let mut original_to_internal_id_bimap = BiBTreeMap::new();
        let mut internal_id_counter = 0;

        for (_bytes, original_llm_id) in sorted_tokens_with_original_ids {
            let internal_llm_id_val = internal_id_counter;
            original_to_internal_id_bimap.insert(original_llm_id.0, internal_llm_id_val);
            internal_id_counter += 1;
        }

        original_to_internal_id_bimap
    }

    pub fn new(
        tokenizer:        Regex,
        parser:           GLRParser,
        llm_token_map:    LLMTokenMap,
        token_name_map:   BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
    ) -> Self {
        let epsilon_terminal_group_ids: BTreeSet<_> = tokenizer.execute_from_state(&[], tokenizer.initial_state_id()).matches.iter().map(|token| token.id).collect();
        let epsilon_terminals: BTreeSet<&Terminal> = epsilon_terminal_group_ids.iter().map(|id| token_name_map.get_by_right(id).unwrap()).collect();
        assert!(epsilon_terminals.is_empty(), "Epsilon tokens (tokens that can match an empty string) are not supported by the grammar constraint. Got: {:?}", epsilon_terminals);
        let original_to_internal_id_bimap = Self::setup_llm_token_mappings(&llm_token_map);

        let internal_max_llm_token = original_to_internal_id_bimap.iter().map(|(_, id)| *id).max().unwrap_or(0);

        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_id_bimap.get_by_left(&original_id.0) {
                internal_llm_token_map_for_precompute.insert(bytes.clone(), LLMTokenID(*internal_id_val));
            }
        }

        let internal_tokens_for_vocab: Vec<(usize, Vec<u8>)> = internal_llm_token_map_for_precompute
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        crate::debug!(2, "Building vocab prefix tree for possible_matches computation");
        let vocab_for_possible_matches = VocabPrefixTree::build(&internal_tokens_for_vocab);
        crate::debug!(2, "Done building vocab prefix tree for possible_matches computation");

        let mut computed_possible_matches = BTreeMap::new();
        let mut pm_cache: HashMap<(*const VocabPrefixTreeNode, TokenizerStateID), BTreeMap<GrammarTokenID, LLMTokenBV>> = HashMap::new();

        crate::debug!(2, "Computing possible_matches for all {} tokenizer states", tokenizer.iter_states().count());
        for sid in tokenizer.iter_states() {
            let matches_for_sid = Self::compute_possible_matches_for_vocab_node(
                &tokenizer,
                &vocab_for_possible_matches.root,
                sid,
                &mut pm_cache,
            );
            computed_possible_matches.insert(sid, matches_for_sid);
        }
        crate::debug!(2, "Finished computing possible_matches");

        let grammar_productions = &parser.productions;
        let grammar_term_map = &parser.terminal_map;

        let terminal_follow_sets_named = compute_terminal_follow_sets(grammar_productions);
        crate::debug!(5, "terminal_follow_sets_named:");
        for (terminal, following_terminals) in &terminal_follow_sets_named {
            crate::debug!(4, "{} -> {}", terminal, following_terminals.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "));
        }

        let mut terminal_follow_map: BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>> = BTreeMap::new();
        for (terminal1, following_terminals) in terminal_follow_sets_named {
            let t1_id = *grammar_term_map.get_by_left(&terminal1).unwrap();
            let mut following_ids = BTreeSet::new();
            for t2 in following_terminals {
                let t2_id = *grammar_term_map.get_by_left(&t2).unwrap();
                following_ids.insert(t2_id);
            }
            if !following_ids.is_empty() {
                terminal_follow_map.insert(t1_id, following_ids);
            }
        }

        crate::debug!(2, "Computed terminal_follow_map_ids with {} entries.", terminal_follow_map.len());

        let llm_vocab = Arc::new(LLMVocab {
            llm_token_map,
            max_original_llm_token_id,
            original_to_internal_id_bimap,
            internal_max_llm_token,
        });

        let (precomputed_arena, precomputed) = precompute(
            &tokenizer, Some(&parser), Some(llm_vocab.clone()), &internal_llm_token_map_for_precompute, Some(&token_name_map), internal_max_llm_token, &terminal_follow_map, parser.ignore_terminal_id, &mut computed_possible_matches,
        );
        Self::_dump_precomputed(
            &precomputed_arena,
            &precomputed,
            &llm_vocab.original_to_internal_id_bimap,
            &token_name_map,
            &llm_vocab.llm_token_map,
        );

        let (precomputed2_arena, precomputed2) = precompute2(
            &precomputed_arena, &precomputed, &tokenizer, Some(&parser), Some(llm_vocab.clone()), &internal_llm_token_map_for_precompute, internal_max_llm_token, &terminal_follow_map, parser.ignore_terminal_id, &mut computed_possible_matches,
        );

        let mut stats2 = PrecomputeStats::default();
        calculate_final_stats2(&precomputed2_arena, &precomputed2, &mut stats2);
        print_precompute_stats2(&stats2);

        Self::_dump_precomputed2(
            &precomputed2_arena,
            &precomputed2,
            &llm_vocab.original_to_internal_id_bimap,
            &llm_vocab.llm_token_map,
        );

        Self {
            tokenizer,
            parser,
            precomputed_arena,
            precomputed,
            precomputed2_arena,
            precomputed2,
            llm_vocab,
            token_name_map,
            possible_matches: computed_possible_matches,
        }
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            self.parser.init_glr_parser(Some(self.llm_vocab.clone())),
        );

        GrammarConstraintState { parent: self, state }
    }

    #[inline]
    pub(crate) fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.llm_vocab.original_to_internal_id_bimap.get_by_left(&original_id.0).map(|internal_val| LLMTokenID(*internal_val))
    }

    #[inline]
    pub(crate) fn internal_id_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.llm_vocab.original_to_internal_id_bimap.get_by_right(&internal_id.0).map(|original_val| LLMTokenID(*original_val))
    }

    #[allow(dead_code)]
    pub(crate) fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
        for original_id_val in original_bv.iter() {
            let internal_id_val = self.llm_vocab.original_to_internal_id_bimap.get_by_left(&(original_id_val as usize)).expect(format!("Original ID {} not found in original_to_internal_id_bimap", original_id_val).as_str());
            internal_bv.insert(*internal_id_val as usize);
        }
        internal_bv
    }

    #[time_it]
    fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        let internal_bv = internal_bv & &LLMTokenBV::max_ones();
        let mut original_bv = HybridBitset::zeros();
        for i in 0..=self.llm_vocab.internal_max_llm_token {
            if internal_bv.contains(i) {
                if let Some(original_id_val) = self.llm_vocab.original_to_internal_id_bimap.get_by_right(&i) {
                    original_bv.insert(*original_id_val);
                }
            }
        }
        original_bv
    }

    pub(crate) fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        HybridBitset::max_ones()
    }

    fn compute_possible_matches_for_vocab_node(
        tokenizer: &Regex,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
        cache: &mut HashMap<(*const VocabPrefixTreeNode, TokenizerStateID), BTreeMap<GrammarTokenID, LLMTokenBV>>,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key = (vocab_node as *const VocabPrefixTreeNode, tokenizer_state_id);
        if let Some(cached_result) = cache.get(&cache_key) {
            return cached_result.clone();
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let child_vocab_node_ref = child_vocab_arc; // Get &VocabPrefixTreeNode
            let exec_result = tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);

            for token_match in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token_match.id);
                // LLM tokens reachable under child_vocab_node_ref are those that start with segment_bytes
                let applicable_tokens = child_vocab_node_ref.reachable_token_ids();
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros) |= applicable_tokens;
            }

            if let Some(final_state_val) = exec_result.end_state {
                let final_tokenizer_state_id = TokenizerStateID(final_state_val);

                let matches_possible_from_new_tokenizer_state: BTreeSet<_> = tokenizer
                    .tokens_accessible_from_state(final_tokenizer_state_id)
                    .into_iter()
                    .collect();

                let matches_from_current_segment: BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();

                let new_grammar_tokens_to_look_for = &matches_possible_from_new_tokenizer_state - &matches_from_current_segment;

                if !new_grammar_tokens_to_look_for.is_empty() {
                    let next_results = Self::compute_possible_matches_for_vocab_node(
                        tokenizer,
                        child_vocab_node_ref, // Recurse with the child node
                        final_tokenizer_state_id,
                        cache,
                    );
                    for (token, bv) in next_results {
                        *result_map.entry(token).or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }
        cache.insert(cache_key, result_map.clone());
        result_map
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
}

impl ParseState {
    pub fn merge(&mut self, other: ParseState) {
        Arc::make_mut(&mut self.stack).merge_with_depth(usize::MAX, &other.stack);
        Arc::make_mut(&mut self.accepted_state).merge_with_depth(usize::MAX, &other.accepted_state);
    }
}

pub trait InsertWith<K, V> {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F);
}

impl<K, V> InsertWith<K, V> for BTreeMap<K, V> where K: Eq + Ord {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F) {
        match self.entry(k) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                let value = occupied.get_mut();
                combine(value, v);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(v);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    pub(crate) state:  BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

impl<'a> PartialEq for GrammarConstraintState<'a> {
    fn eq(&self, other: &Self) -> bool {
        // Compare parent by pointer to ensure they originate from the same constraint object.
        std::ptr::eq(self.parent, other.parent) && self.state == other.state
    }
}

impl<'a> Eq for GrammarConstraintState<'a> {}

impl<'a> Display for GrammarConstraintState<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "GrammarConstraintState ({} active tokenizer states):", self.state.len())?;
        if self.state.is_empty() {
            return Ok(());
        }

        let mut gss_roots = Vec::new();
        let mut tokenizer_state_info = Vec::new();

        for (tokenizer_state_id, glr_state) in &self.state {
            if !glr_state.active_state.stack.is_empty() {
                gss_roots.push(glr_state.active_state.stack.clone());
                tokenizer_state_info.push(format!(
                    "  - Tokenizer State {:>3}: GSS Root ({} predecessors)",
                    tokenizer_state_id.0,
                    glr_state.active_state.stack.num_predecessors()
                ));
            } else {
                tokenizer_state_info.push(format!(
                    "  - Tokenizer State {:>3}: (Empty GSS)",
                    tokenizer_state_id.0
                ));
            }
        }

        for info in tokenizer_state_info {
            writeln!(f, "{}", info)?;
        }

        if !gss_roots.is_empty() {
            writeln!(f, "\nCombined GSS Forest (showing up to 50 nodes):")?;
            let config = GSSPrintConfig {
                labels: None,
                max_edges: 50,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            let (gss_str, _) =
                crate::datastructures::gss::print_gss_forest(&gss_roots, &self.parent.parser.terminal_map, &config);
            write!(f, "{}", gss_str)?;
        }

        Ok(())
    }
}

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&self) -> LLMTokenBV {
        self.get_mask2()
    }

    #[time_it]
    pub fn get_mask1(&self) -> LLMTokenBV {
        let t0 = std::time::Instant::now();
        crate::debug!(2, "Getting mask {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        let stats = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats: {:#?}", stats);
        let roots = self.state.values().map(|s| s.active_state.stack.clone()).collect::<Vec<_>>();
        if GSS_LOGGING_ENABLED {
            let (s, state_ids) = print_gss_forest(&roots, &self.parent.parser.terminal_map, &GSSPrintConfig::default());
            println!("{}", s);
            println!("\n\n--- GSS State Explanations ---\n");
            for state_id in state_ids {
                let mut explanation = String::new();
                println!("\n--- State {} ---", state_id.0);
                self.parent.parser.format_state_details(&mut explanation, state_id, "  ").unwrap();
                println!("{}", explanation);
            }

            println!("\n\n--- Begin GSS Graphviz ---");
            let labels: Vec<String> = self.state.keys().map(|k| format!("State {}", k.0)).collect();
            let roots_with_labels: Vec<(&str, &GSSNode)> = labels.iter()
                .map(|s| s.as_str())
                .zip(self.state.values().map(|s| s.active_state.stack.as_ref()))
                .collect();
            println!("{}", self.parent.parser.gss_forest_to_dot(
                &roots_with_labels,
                Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                Some(&self.parent.llm_vocab.llm_token_map),
            ));
            println!("\n\n--- End GSS Graphviz ---");
        }

        for (state_id, state) in self.state.iter() {
            crate::debug!(3, "State {}:", state_id.0);
        }

        let final_mask_internal = RefCell::new(HybridBitset::zeros());

        if self.state.is_empty() {
            return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        #[derive(Default, Clone, Copy, Debug)]
        struct StepCount {
            total: usize,
            successful: usize,
        }

        let step_counts = Arc::new(std::sync::RwLock::new(BTreeMap::<TerminalID, StepCount>::new()));

        let mut initial_values_for_map: Vec<(NodeId, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_id) = self.parent.precomputed.get(tokenizer_state_id) {
                let mut forbidden_llm_tokens = LLMTokenBV::zeros();
                let disallowed_terminals_l2 = glr_state.active_state.stack.disallowed_terminals();

                for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
                    if disallowed_terminals_for_range.is_empty() {
                        continue;
                    }
                    let relevant_possible_matches = self.parent.possible_matches.range(TokenizerStateID(*tokenizer_state_range.start())..=TokenizerStateID(*tokenizer_state_range.end()));
                    for (_tokenizer_state_id, possible_matches_for_state) in relevant_possible_matches {
                        for (terminal_id, llm_tokens_that_match_this_terminal) in possible_matches_for_state {
                            if disallowed_terminals_for_range.contains(terminal_id.0) {
                                forbidden_llm_tokens |= llm_tokens_that_match_this_terminal;
                            }
                        }
                    }
                }
                let mut glr_state = glr_state.clone();
                if !forbidden_llm_tokens.is_empty() {
                    disallow_llm_tokens_and_prune_arc(&mut glr_state.active_state.stack, &forbidden_llm_tokens, &mut HashMap::new());
                }
                initial_values_for_map.push((*precomputed_trie_root_id, glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));

        let step_counts_clone1 = Arc::clone(&step_counts);

        crate::profiler::reset();

        let mut precomputed_arena = self.parent.precomputed_arena.clone(); // Clone to get a mutable copy for traversal
        precomputed_arena.special_map_grouped(
            initial_values_for_map,
            // step_fn
            |glr_s, grammar_token_opt, dest_map| {
                if true {
                    timeit!("get_mask try to avoid step for no additional llm tokens", {
                    let mut all_edge_llm_tokens = HybridBitset::zeros();
                    for edge_llm_tokens_bv in dest_map.values() {
                        all_edge_llm_tokens |= edge_llm_tokens_bv;
                    }
                    let glr_s_llm_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                    let potential_additional_llm_tokens = &glr_s_llm_tokens & &all_edge_llm_tokens;
                    if potential_additional_llm_tokens.is_subset(&final_mask_internal.borrow()) {
                        crate::debug!(4, "Skipping step for grammar token {:?} as all edge LLM tokens are already in final mask.", grammar_token_opt);
                        return Vec::new();
                    }
                    });
                }

                let mut num_end = 0;
                let mut num_non_end = 0;
                for &child_node_id in dest_map.keys() {
                    if precomputed_arena.get_node(child_node_id).value.end {
                        num_end += 1;
                    } else {
                        num_non_end += 1;
                    }
                }
                timeit!(format!("get_mask step_fn - end only? {}", num_end > 0 && num_non_end == 0), {
                    if num_non_end == 0 {
                        if let Some(gtid) = grammar_token_opt {
                            match glr_s.has_action_for(*gtid) {
                                Some(glr_s_llm_tokens) => {
                                    timeit!(format!("get_mask step_fn - has_action_for"), {
                                        crate::debug!(4, "Step with grammar token {:?} ({}) has action, but all children are end nodes, so we can skip stepping and update final mask directly.", gtid, self.parent.parser.terminal_map.get_by_right(gtid).map_or("UNKNOWN_TERMINAL".to_string(), |s| s.to_string()));
                                        let mut edge_llm_tokens = HybridBitset::zeros();
                                        for edge_llm_tokens_bv in dest_map.values() {
                                            edge_llm_tokens |= edge_llm_tokens_bv;
                                        }
                                        let llm_tokens = &glr_s_llm_tokens & &edge_llm_tokens;
                                        crate::debug!(4, "Adding active tokens {:?} to final mask", llm_tokens);
                                        *final_mask_internal.borrow_mut() |= llm_tokens;
                                        crate::debug!(4, "Final mask after adding tokens: {:?}", final_mask_internal.borrow());
                                        return Vec::new();
                                    });
                                },
                                None => {
                                    timeit!(format!("get_mask step_fn - has_action_for - inconclusive"), {
                                        crate::debug!(4, "Inconclusive step for grammar token {:?}, no action found.", gtid);
                                    });
                                },
                            }
                        }
                    }

                    let mut glr_s = glr_s.clone();

                    if let Some(gtid) = grammar_token_opt {
                        let mut counts_guard = step_counts_clone1.write().unwrap();
                        let entry = counts_guard.entry(*gtid).or_default();
                        entry.total += 1;
                        glr_s.process_token(*gtid);
                        if glr_s.is_ok() {
                            entry.successful += 1;
                        } else {
                            return Vec::new();
                        }
                    }

                    let mut results = Vec::new();
                    crate::debug!(4, "Processing edge: {:?}", grammar_token_opt);
                    for (&child_node_id, edge_llm_tokens_bv) in dest_map.iter() {
                        let mut glr_s = glr_s.clone();
                        allow_only_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &edge_llm_tokens_bv, &mut HashMap::new());
                        if !glr_s.is_ok() {
                            continue;
                        }
                        if precomputed_arena.get_node(child_node_id).value.end {
                            let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                            *final_mask_internal.borrow_mut() |= glr_active_tokens;
                        }
                        results.push((child_node_id, glr_s));
                    }
                    results
                })
            },
            // merge_fn
            |glr_s1, glr_s2| {
                timeit!("get_mask merge_fn", {
                    glr_s1.merge_with(glr_s2);
                })
            },
            // process_fn
            |trie, precomputed_node_data, glr_s| {
                timeit!("get_mask process_fn", {
                    if precomputed_node_data.value.end {
                        let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                        *final_mask_internal.borrow_mut() |= glr_active_tokens;
                        false
                    } else {
                        let mut num_outgoing_edges_that_lead_to_non_end_nodes = 0;
                        for (edge_terminal_opt, dest_map) in precomputed_node_data.children.iter() {
                            if edge_terminal_opt.is_none() {
                                num_outgoing_edges_that_lead_to_non_end_nodes += 1
                            } else {
                                for &child_node_id in dest_map.keys() {
                                    if !trie.get_node(child_node_id).value.end {
                                        num_outgoing_edges_that_lead_to_non_end_nodes += 1;
                                        break;
                                    }
                                }
                            }
                            if num_outgoing_edges_that_lead_to_non_end_nodes >= 2 {
                                break;
                            }
                        }
                        disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());
                        Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                        let mut do_phase3 = false;
                        do_phase3 |= num_outgoing_edges_that_lead_to_non_end_nodes >= 2;
                        do_phase3 |= match LR_MODE {
                            LRMode::LR1 | LRMode::LALR_EX_SHIFT_STATES => false,
                            LRMode::LALR => true,
                        };
                        if do_phase3 {
                            let mut allowed_terminals = TerminalBV::zeros();
                            for gtid_opt in precomputed_node_data.children.keys() {
                                if let Some(gtid) = gtid_opt {
                                    allowed_terminals.insert(gtid.0);
                                }
                            }
                            let disallowed_terminals_bv = allowed_terminals.inverted();
                            if !disallowed_terminals_bv.is_empty() {
                                let disallowed_l2 = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::from_iter(
                                    std::iter::once((0..=usize::MAX, disallowed_terminals_bv))
                                );
                                disallow_terminals_and_prune_arc(&mut glr_s.active_state.stack, &disallowed_l2, &mut HashMap::new());
                            }
                            glr_s.process_default_reductions();
                            Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                        }
                        !glr_s.active_state.stack.is_empty()
                    }
                })
            },
        );

        let t_after_special_map = std::time::Instant::now();
        println!("after special_map: {:>15?}", t_after_special_map.duration_since(t0));

        crate::profiler::print_summary_flat();

        let counts = step_counts.read().unwrap();
        if !counts.is_empty() {
            let mut sorted_counts: Vec<_> = counts.iter().collect();
            sorted_counts.sort_by_key(|&(_, count)| std::cmp::Reverse(count.total));
            let mut log_msg = String::from("get_mask step() counts:");
            for (terminal_id, count) in sorted_counts {
                let terminal_name = self.parent.parser.terminal_map.get_by_right(terminal_id)
                    .map(|s| s.to_string())
                    .unwrap_or("UNKNOWN_TERMINAL".to_string());
                log_msg.push_str(&format!("\n  - '{}': {}/{} successful", terminal_name, count.successful, count.total));
            }
            crate::debug!(3, "{}", log_msg);
        }

        crate::profiler::print_summary();
        crate::profiler::reset();

        if GSS_LOGGING_ENABLED {
            crate::debug!(3, "Final GSS states after get_mask:");
            let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
            let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();
            let config = GSSPrintConfig {
                labels: Some(&labels),
                max_edges: 300,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);
        }

        let final_mask_mapped = self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        let t_end = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t_end.duration_since(t0));
        final_mask_mapped
    }

    pub fn run_precompute2_traversal(&mut self, initial_values_for_map: &mut Vec<(NodeId, GLRParserState)>) {
        todo!()
    }

    pub fn get_mask2(&self) -> LLMTokenBV {
        let t0 = std::time::Instant::now();
        crate::debug!(2, "Getting mask {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        let stats = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats: {:#?}", stats);
        let roots = self.state.values().map(|s| s.active_state.stack.clone()).collect::<Vec<_>>();
        if GSS_LOGGING_ENABLED {
            let (s, state_ids) = print_gss_forest(&roots, &self.parent.parser.terminal_map, &GSSPrintConfig::default());
            println!("{}", s);
            println!("\n\n--- GSS State Explanations ---\n");
            for state_id in state_ids {
                let mut explanation = String::new();
                println!("\n--- State {} ---", state_id.0);
                self.parent.parser.format_state_details(&mut explanation, state_id, "  ").unwrap();
                println!("{}", explanation);
            }

            println!("\n\n--- Begin GSS Graphviz ---");
            let labels: Vec<String> = self.state.keys().map(|k| format!("State {}", k.0)).collect();
            let roots_with_labels: Vec<(&str, &GSSNode)> = labels.iter()
                .map(|s| s.as_str())
                .zip(self.state.values().map(|s| s.active_state.stack.as_ref()))
                .collect();
            println!("{}", self.parent.parser.gss_forest_to_dot(
                &roots_with_labels,
                Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                Some(&self.parent.llm_vocab.llm_token_map),
            ));
            println!("\n\n--- End GSS Graphviz ---");
        }

        for (state_id, state) in self.state.iter() {
            crate::debug!(3, "State {}:", state_id.0);
        }

        let final_mask_internal = RefCell::new(HybridBitset::zeros());

        if self.state.is_empty() {
            return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let mut initial_values_for_map: Vec<(NodeId, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_id) = self.parent.precomputed2.get(tokenizer_state_id) {
                let mut forbidden_llm_tokens = LLMTokenBV::zeros();
                let disallowed_terminals_l2 = glr_state.active_state.stack.disallowed_terminals();

                for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
                    if disallowed_terminals_for_range.is_empty() {
                        continue;
                    }
                    let relevant_possible_matches = self.parent.possible_matches.range(TokenizerStateID(*tokenizer_state_range.start())..=TokenizerStateID(*tokenizer_state_range.end()));
                    for (_tokenizer_state_id, possible_matches_for_state) in relevant_possible_matches {
                        for (terminal_id, llm_tokens_that_match_this_terminal) in possible_matches_for_state {
                            if disallowed_terminals_for_range.contains(terminal_id.0) {
                                forbidden_llm_tokens |= llm_tokens_that_match_this_terminal;
                            }
                        }
                    }
                }
                let mut glr_state = glr_state.clone();
                if !forbidden_llm_tokens.is_empty() {
                    disallow_llm_tokens_and_prune_arc(&mut glr_state.active_state.stack, &forbidden_llm_tokens, &mut HashMap::new());
                }
                initial_values_for_map.push((*precomputed_trie_root_id, glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));

        crate::profiler::reset();

        self.run_precompute2_traversal(initial_values_for_map, &final_mask_internal);

        let t_after_special_map = std::time::Instant::now();
        println!("after traversal: {:>15?}", t_after_special_map.duration_since(t0));

        crate::profiler::print_summary_flat();
        crate::profiler::print_summary();
        crate::profiler::reset();

        if GSS_LOGGING_ENABLED {
            crate::debug!(3, "Final GSS states after get_mask:");
            let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
            let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();
            let config = GSSPrintConfig {
                labels: Some(&labels),
                max_edges: 300,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);
        }

        crate::debug!(4, "Final mask internal: {:?}", final_mask_internal.borrow());
        let final_mask_mapped = self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        crate::debug!(4, "Final mask mapped: {:?}", final_mask_mapped);

        let t_end = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t_end.duration_since(t0));

        final_mask_mapped
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) { // llm_token_id is original
        let llm_token_bytes = self.parent.llm_vocab.llm_token_map.get_by_right(&llm_token_id).unwrap();
        self.commit_bytes(llm_token_bytes);
    }

    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) { // llm_token_id is original
        if llm_token_bytes.is_empty() {
            return;
        }

        crate::debug!(2, "Committing bytes: {:?}", String::from_utf8_lossy(llm_token_bytes));

        let mut gss_transformation_memo = HashMap::new();

        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        let mut state_map: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
        let mut terminals_map: BTreeMap<TokenizerStateID, TerminalBV> = BTreeMap::new();
        for (tokenizer_state_id, _state) in self.state.iter() {
            let exec_result = self.parent.tokenizer.execute_from_state(
                &llm_token_bytes,
                *tokenizer_state_id,
            );
            if let Some(new_state) = exec_result.end_state {
                state_map.insert(*tokenizer_state_id, TokenizerStateID(new_state));
            }
            let mut terminals = TerminalBV::zeros();
            for token in exec_result.matches {
                terminals.insert(token.id);
            }
            terminals_map.insert(*tokenizer_state_id, terminals);
        }

        let gss_stats_before_pruning = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats before pruning disallowed terminals: {:#?}", gss_stats_before_pruning);
        for state in self.state.values_mut() {
            prune_disallowed_terminals(&mut state.active_state.stack, &terminals_map, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();
        let gss_stats_after_pruning = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats after pruning disallowed terminals: {:#?}", gss_stats_after_pruning);
        if gss_stats_after_pruning != gss_stats_before_pruning {
            crate::debug!(3, "GSS stats changed after pruning disallowed terminals.");
        } else {
            crate::debug!(3, "GSS stats did not change after pruning disallowed terminals.");
        }

        for state in self.state.values_mut() {
            map_allowed_terminals_tokenizer_states(&mut state.active_state.stack, &state_map, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();

        let mut processing_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, GLRParserState<'a>>> = BTreeMap::new();
        processing_queue.insert(0, std::mem::take(&mut self.state));

        while let Some((offset, states_to_process)) = processing_queue.pop_first() {
            crate::debug!(3, "Processing offset {} with states {:?}.", offset, states_to_process.keys().map(|k| k.0).collect::<Vec<_>>());
            for (tokenizer_s_id_at_offset, glr_s_at_offset) in states_to_process {
                assert!(offset < llm_token_bytes.len());

                let exec_result = self.parent.tokenizer.execute_from_state(
                    &llm_token_bytes[offset..],
                    tokenizer_s_id_at_offset,
                );

                for match_info in &exec_result.matches {
                    let mut cloned_glr_s = glr_s_at_offset.clone();

                    cloned_glr_s.step(TerminalID(match_info.id));

                    if cloned_glr_s.is_ok() {
                        let new_offset = offset + match_info.width;
                        let next_tokenizer_id_for_segment = self.parent.tokenizer.initial_state_id();

                        let mut disallowed_terminals = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::new();
                        if let Some(end_state_id) = exec_result.end_state {
                            let mut disallowed_terminals_for_end_state = TerminalBV::zeros();
                            disallowed_terminals_for_end_state.insert(match_info.id);
                            disallowed_terminals.insert_l2_bitset(end_state_id, disallowed_terminals_for_end_state);
                        }
                        disallow_terminals_and_prune_arc(&mut cloned_glr_s.active_state.stack, &disallowed_terminals, &mut HashMap::new());

                        if new_offset == llm_token_bytes.len() {
                            new_overall_state.entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        } else {
                            processing_queue.entry(new_offset).or_default().entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        }
                    }
                }

                if let Some(final_tokenizer_s_id_for_llm_token_segment) = exec_result.end_state {
                    let final_tokenizer_state = TokenizerStateID(final_tokenizer_s_id_for_llm_token_segment);
                    new_overall_state.entry(final_tokenizer_state).and_modify(|existing| existing.merge_with(glr_s_at_offset.clone())).or_insert(glr_s_at_offset.clone());
                }
            }
        }

        self.state = new_overall_state.clone();

        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        crate::debug!(2, "Active tokenizer states after committing text (bytes {:?}): {:?}", llm_token_bytes, self.state.keys().map(|k|k.0).collect::<Vec<_>>());
    }

    pub fn is_active_or_accepted(&self) -> bool {
        !self.state.is_empty() && self.state.values().any(|s| !s.active_state.stack.is_empty() || s.has_accepted())
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}