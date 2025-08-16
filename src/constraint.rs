// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::sync::RwLock;
use std::mem;
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::gss::{disallow_llm_tokens_and_prune_arc, fuse_predecessors_recursive};
use crate::datastructures::gss::{map_allowed_terminals_tokenizer_states, prune_disallowed_terminals};
use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc};
use std::cell::RefCell;

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, dump_precompute_trie_recursive, print_precompute_stats, PrecomputeStats};
use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
use crate::datastructures::gss::{print_gss_forest, GSSNode, allow_only_llm_tokens_and_prune_arc, gather_gss_stats, reset_llm_tokens, disallow_terminals_and_prune_arc, GSSPrintConfig, LLMTokenBV, TerminalBV, PrecomputedNodeContents, PrecomputeNode2};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    GLRParser, GLRParserState, ParseState, ParseStateEdgeContent,
};
use crate::tokenizer::{LLMToken, LLMTokenID, LLMTokenMap, Token, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use kdam::{tqdm, BarBuilder};
use profiler_macro::{time_it, timeit};
use crate::datastructures::arc_wrapper::{NodePtr, WeakPtrWrapper};
use crate::datastructures::gss::Acc;
use crate::glr::table::StateID;
use crate::glr::analyze::compute_terminal_follow_sets;
use crate::glr::grammar::Terminal;
use crate::glr::items::{LRMode, LR_MODE};
use crate::interface::CompiledGrammar;
use crate::profiler::{print_summary, print_summary_flat, reset, GSS_LOGGING_ENABLED, PROGRESS_BAR_ENABLED};

const MERGE_THRESHOLD: usize = 20;

pub type PrecomputeNode =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

pub type Precomputed = BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode>>>;
pub type Precomputed2 = BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>;

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
    pub(crate) precomputed:      Precomputed,
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
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed.iter().zip(other.precomputed.iter()) {
            assert_eq!(sid1, sid2);
            let node1 = arc1.read().unwrap();
            let node2 = arc2.read().unwrap();
            assert_eq!(*node1, *node2);
        }
        assert_eq!(self.precomputed2.len(), other.precomputed2.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed2.iter().zip(other.precomputed2.iter()) {
            assert_eq!(sid1, sid2);
            let node1 = arc1.read().unwrap();
            let node2 = arc2.read().unwrap();
            assert_eq!(*node1, *node2);
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
        obj.insert("precomputed".to_string(), self.precomputed.to_json());
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
                let precomputed = obj.remove("precomputed").ok_or_else(|| "Missing field precomputed".to_string())
                                     .and_then(|n| Precomputed::from_json(n))?;
                let precomputed2 = obj.remove("precomputed2").ok_or_else(|| "Missing field precomputed2".to_string())
                                     .and_then(|n| Precomputed2::from_json(n))?;

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
                    precomputed,
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
        // // TODO: delete this
        // let mut original_to_internal_id_bimap = BiBTreeMap::new();
        // for (_, id) in original_llm_token_map.iter() {
        //     original_to_internal_id_bimap.insert(id.0, id.0);
        // }
        // return original_to_internal_id_bimap;

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

        // Build VocabPrefixTree for internal LLM tokens (needed for possible_matches computation)
        let internal_tokens_for_vocab: Vec<(usize, Vec<u8>)> = internal_llm_token_map_for_precompute
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();
        // Note: The tokenizer parameter to `new` is shadowed here by the struct field.
        // We need to use the parameter `tokenizer` for the computation.
        // Let's rename the parameter to avoid confusion, or be careful.
        // Assuming `tokenizer` in `Self { tokenizer, ... }` refers to the parameter, it's fine.
        
        crate::debug!(2, "Building vocab prefix tree for possible_matches computation");
        let vocab_for_possible_matches = VocabPrefixTree::build(&internal_tokens_for_vocab);
        crate::debug!(2, "Done building vocab prefix tree for possible_matches computation");

        let mut computed_possible_matches = BTreeMap::new();
        // Cache for the possible_matches computation
        let mut pm_cache: HashMap<(*const VocabPrefixTreeNode, TokenizerStateID), BTreeMap<GrammarTokenID, LLMTokenBV>> = HashMap::new();

        crate::debug!(2, "Computing possible_matches for all {} tokenizer states", tokenizer.iter_states().count());
        for sid in tokenizer.iter_states() { // Use the `tokenizer` parameter passed to `new`
            let matches_for_sid = Self::compute_possible_matches_for_vocab_node(
                &tokenizer, // Pass the tokenizer parameter from `new`
                &vocab_for_possible_matches.root,
                sid,
                &mut pm_cache,
            );
            computed_possible_matches.insert(sid, matches_for_sid);
        }
        crate::debug!(2, "Finished computing possible_matches");
        // pm_cache is dropped here as it's no longer needed.

        let grammar_productions = &parser.productions; // Assuming parser is the GLRParser instance
        let grammar_term_map = &parser.terminal_map;

        // These might be computed elsewhere or need to be computed here.
        // Assuming compute_first_sets is available from grammar module.

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

        let precomputed = Self::precompute(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            internal_max_llm_token,
            &terminal_follow_map, // Pass the new map
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
        );

        let precomputed2 = Self::precompute2(
            &precomputed,
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            internal_max_llm_token,
            &terminal_follow_map, // Pass the new map
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
        );

        // Promote weak edges in the second precomputed trie, which may have been
        // created by the GLR parser logic to break cycles.
        let roots2: Vec<_> = precomputed2.values().cloned().collect();
        let promotions2 = Trie::promote_weak_edges_to_strong(&roots2);
        crate::debug!(2, "Promoted {} weak edges to strong in precomputed trie 2.", promotions2);

        let mut gc = Self {
            tokenizer, // This is the tokenizer parameter being moved into the struct
            parser,
            precomputed,
            precomputed2,
            llm_vocab,
            token_name_map,
            possible_matches: computed_possible_matches, // Add this line
        };

        gc
    }

    pub fn precompute(
        tokenizer:        &Regex,
        parser:           Option<&GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode>>> {
        let mut helper = Precomputer::new(
            tokenizer,
            parser,
            llm_vocab,
            internal_llm_token_map,
            internal_max_llm_token,
            MERGE_THRESHOLD,
            terminal_follow_map, // Pass to Precomputer::new
            ignore_terminal_id,
        );

        helper.run_dfs();
        // helper.optimize_precomputed_via_substring_parser();
        helper.replace_ignore_token_edges_with_none_edges();
        helper.simplify_none_edges(); // Simplify out None-edges by shortcutting predecessors to successors
        helper.refine_tokens_forward();
        helper.prune_dead_paths();
        helper.prune_on_no_terminal_follow();
        helper.prune_dead_paths();
        helper.factor_common_destinations();
        helper.merge_nodes();
        // helper.merge_nodes_basic();
        helper.finish(token_name_map, possible_matches, internal_max_llm_token)
    }

    /// Build the "Trie 2" precomputation.
    pub fn precompute2(
        precomputed: &BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode>>>,
        tokenizer:        &Regex,
        parser:           Option<&GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> Precomputed2 {
        crate::debug!(2, "Precomputing Trie 2...");
        let mut precomputed2 = BTreeMap::new();
        let mut memo: HashMap<ArcPtrWrapper<RwLock<PrecomputeNode>>, Arc<RwLock<_>>> = HashMap::new();

        let mut initial_values_for_map: Vec<(Arc<RwLock<PrecomputeNode>>, GLRParserState)> =
            Vec::new();
        let parser = parser.unwrap();
        // for (tokenizer_state_id, trie1_root) in tqdm!(precomputed.iter(), desc = "Precomputing Trie 2", disable = !PROGRESS_BAR_ENABLED) {
        for (tokenizer_state_id, trie1_root) in precomputed.iter() {
            if let Some(trie2_root) = memo.get(&ArcPtrWrapper::new(trie1_root.clone())) {
                precomputed2.insert(*tokenizer_state_id, trie2_root.clone());
                continue;
            }
            let trie2_root = Arc::new(RwLock::new(PrecomputeNode2::new(
                PrecomputedNodeContents::no_end(),
            )));
            precomputed2.insert(*tokenizer_state_id, trie2_root.clone());

            let mut gss_nodes_to_merge = Vec::new();

            for state_id in parser.table.keys() {
                let new_trie2_node = Arc::new(RwLock::new(PrecomputeNode2::new(
                    PrecomputedNodeContents::no_end(),
                )));

                let mut inserter = EdgeInserter::new(
                    trie2_root.clone(),
                    (0, Some(*state_id)),
                    LLMTokenBV::ones(internal_max_llm_token + 1),
                    |e, n| *e |= n,
                );
                inserter = inserter.try_destination(new_trie2_node.clone());
                inserter.expect("Failed to insert initial edge into Trie2 root");

                let mut acc = Acc::new_fresh();
                acc.trie2_nodes
                    .insert(ArcPtrWrapper::new(new_trie2_node));
                let gss_root = GSSNode::new(acc);
                let gss_node =
                    gss_root.push(ParseStateEdgeContent { state_id: *state_id });
                gss_nodes_to_merge.push(Arc::new(gss_node));
            }

            let merged_gss = GSSNode::merge_many_with_depth(usize::MAX, gss_nodes_to_merge);
            let parse_state = ParseState { stack: merged_gss };
            let glr_state = parser.init_glr_parser_from_parse_state(parse_state);

            memo.insert(ArcPtrWrapper::new(trie1_root.clone()), trie2_root.clone());

            initial_values_for_map.push((trie1_root.clone(), glr_state));

        }

        let trie2_end = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::end(), )));

        crate::debug!(2, "Running special_map_grouped for Trie 2 precomputation");
        Trie::special_map_grouped(
            initial_values_for_map,
            // step_fn: (current_glr_state, edge_grammar_token_opt, destinations_map)
            |current_glr_state, edge_grammar_token_opt, destinations_map| {
                crate::debug!(3, "Trie2: Processing GLR state with {} destinations for edge grammar token: {:?}", destinations_map.len(), edge_grammar_token_opt);
                let mut glr_s = current_glr_state.clone();
                if let Some(gt) = edge_grammar_token_opt {
                    glr_s.process_token(*gt);
                        print_summary_flat();
                        print_summary();
                        reset();
                }

                let mut out = Vec::new();
                for (dst_node_wrapper, edge_bv) in destinations_map.iter() {
                    let mut glr_s_copy = glr_s.clone();
                    // Restrict the GLR state to the LLM tokens allowed on this edge.
                    crate::debug!(3, "Trie2: Restricting GLR state to edge bitset: {:?}", edge_bv);
                    allow_only_llm_tokens_and_prune_arc(
                        &mut glr_s_copy.active_state.stack,
                        edge_bv,
                        &mut HashMap::new(),
                    );
                    glr_s_copy.log_gss(
                        "Trie2: After restricting GLR state to edge bitset",
                        TerminalID(0),
                        false,
                        false,
                    );
                    out.push((
                        dst_node_wrapper.clone(),
                        glr_s_copy,
                    ));
                }
                out
            },
            |glr_s1, glr_s2| {
                crate::debug!(4, "Trie2: Merging GLR states");
                glr_s1.log_gss("Before merge...", TerminalID(0), false, false);
                glr_s2.log_gss("...with", TerminalID(0), false, false);
                glr_s1.merge_with(glr_s2);
                glr_s1.log_gss("After merge", TerminalID(0), false, false);
            },
            // process_fn
            |precomputed_node_data, glr_s| {
                // Dump precomputed2
                // pub fn _dump_precomputed2(precomputed2: &BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>, original_to_internal_id_bimap: &BiBTreeMap<usize, usize>, llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>) {
                // GrammarConstraint::_dump_precomputed2(&precomputed2, &llm_vocab.as_ref().unwrap().original_to_internal_id_bimap, &llm_vocab.as_ref().unwrap().llm_token_map);

                crate::datastructures::gss::merge_trie2_nodes_if_needed(
                    &mut glr_s.active_state.stack,
                    MERGE_THRESHOLD,
                    &mut HashMap::new(),
                );
                let active_llm_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                let keep_going = !active_llm_tokens.is_empty();
                if precomputed_node_data.value.end {
                    crate::debug!(3, "Trie2: Found end state for GLR state");
                    glr_s.log_gss(
                        "Trie2: Found end state for GLR state",
                        TerminalID(0),
                        false,
                        false,
                    );
                    for gss_root in glr_s.active_state.stack.get_roots() {
                        let gss_root_acc: Arc<Acc> = gss_root.resolved_acc();
                        let active_llm_tokens_for_root = gss_root_acc.union_llm_tokens();
                        crate::debug!(3, "Trie2: Inserting end edge into Trie2 node with active LLM tokens: {:?} into Trie2 nodes: {:?}", active_llm_tokens_for_root, gss_root_acc.trie2_nodes);
                        for trie2_node in gss_root_acc.trie2_nodes.iter() {
                            let mut inserter = EdgeInserter::new(
                                trie2_node.as_arc().clone(),
                                (0, None),
                                active_llm_tokens_for_root.clone(),
                                |e, n| *e |= n,
                            );
                            inserter = inserter.try_destination(trie2_end.clone());
                            inserter.expect("Failed to insert end edge into Trie2 node");
                        }
                    }
                }
                keep_going
            },
        );

        crate::debug!(2, "Finished precomputing Trie 2");

        // Optimizations for Trie 2
        let all_llm_tokens = HybridBitset::max_ones();
        refine_tokens_forward_pass_generic(&mut precomputed2, &all_llm_tokens);
        prune_dead_paths_generic(&mut precomputed2, &all_llm_tokens);
        merge_nodes_generic(&mut precomputed2);

        precomputed2
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
        // for internal_id_val in internal_bv.iter() {
        //     let original_id_val = self.llm_vocab.original_to_internal_id_bimap.get_by_right(&(internal_id_val as usize)).expect(format!("Internal ID {} not found in original_to_internal_id_bimap while converting to original BV from internal BV: {:?}", internal_id_val, internal_bv).as_str());
        //     original_bv.insert(*original_id_val as usize);
        // }
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

fn refine_tokens_forward_pass_generic<EK>(
    roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>>,
    all_llm_tokens: &LLMTokenBV,
)
where
    EK: Ord + Clone + Send + Sync + 'static,
    Trie<EK, LLMTokenBV, PrecomputedNodeContents>: Send + Sync,
{
    crate::debug!(2, "Refining tokens with forward pass...");

    let mut reachable_tokens: HashMap<*const RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>, LLMTokenBV> = HashMap::new();
    let mut worklist: VecDeque<Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>> = VecDeque::new();
    let mut in_worklist: HashSet<*const RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>> = HashSet::new();

    // Initialize roots
    for root_arc in roots.values() {
        let root_ptr = Arc::as_ptr(root_arc);
        reachable_tokens.insert(root_ptr, all_llm_tokens.clone());
        if in_worklist.insert(root_ptr) {
            worklist.push_back(root_arc.clone());
        }
    }

    // Propagate tokens forward until a fixed point is reached.
    while let Some(src_arc) = worklist.pop_front() {
        let src_ptr = Arc::as_ptr(&src_arc);
        in_worklist.remove(&src_ptr);

        let tokens_reaching_src = reachable_tokens.get(&src_ptr).cloned().unwrap_or_else(LLMTokenBV::zeros);

        let children_to_process: Vec<(NodePtr<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>, LLMTokenBV)> = {
            let src_guard = src_arc.read().unwrap();
            src_guard.children().values().flat_map(|dest_map| {
                dest_map.iter().map(|(k, v)| (k.clone(), v.clone()))
            }).collect()
        };

        for (child_wrapper, edge_bv) in children_to_process {
            if let Some(child_arc) = child_wrapper.upgrade() {
                let child_ptr = Arc::as_ptr(&child_arc);
                let tokens_for_child = &tokens_reaching_src & &edge_bv;

                if tokens_for_child.is_empty() {
                    continue;
                }

                let child_entry = reachable_tokens.entry(child_ptr).or_insert_with(LLMTokenBV::zeros);
                let old_tokens = child_entry.clone();
                *child_entry |= &tokens_for_child;

                if *child_entry != old_tokens {
                    if in_worklist.insert(child_ptr) {
                        worklist.push_back(child_arc);
                    }
                }
            }
        }
    }

    // Now, refine the edge bitsets based on the computed reachable tokens for each source node.
    let all_nodes: Vec<_> = roots.values().flat_map(|r| Trie::all_nodes(r.clone())).collect();
    let mut seen = HashSet::new();

    for src_arc in all_nodes {
        let src_ptr = Arc::as_ptr(&src_arc);
        if !seen.insert(src_ptr) { continue; }

        if let Some(tokens_reaching_src) = reachable_tokens.get(&src_ptr) {
            let mut src_guard = src_arc.write().unwrap();
            src_guard.children_mut().retain(|_ek, dest_map| {
                dest_map.retain(|_child_wrapper, edge_bv| {
                    *edge_bv &= tokens_reaching_src;
                    !edge_bv.is_empty()
                });
                !dest_map.is_empty()
            });
        } else {
            // This node is unreachable by any token path from a root. Prune all its outgoing edges.
            let mut src_guard = src_arc.write().unwrap();
            src_guard.children_mut().clear();
        }
    }
    crate::debug!(2, "Finished refining tokens with forward pass.");
}

fn prune_dead_paths_generic<EK>(
    roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>>,
    all_llm_tokens: &LLMTokenBV,
) where
    EK: Ord + Clone + Send + Sync + 'static,
    Trie<EK, LLMTokenBV, PrecomputedNodeContents>: Send + Sync,
{
    crate::debug!(2, "Pruning dead paths from precomputed trie.");

    // 1. Collect all nodes and build predecessor map.
    let mut all_nodes = Vec::new();
    let mut seen = HashSet::new();
    for root in roots.values() {
        for node in Trie::all_nodes(root.clone()) {
            if seen.insert(Arc::as_ptr(&node)) {
                all_nodes.push(node);
            }
        }
    }

    let mut predecessors: HashMap<*const RwLock<_>, Vec<*const RwLock<_>>> = HashMap::new();
    let mut node_map: HashMap<*const RwLock<_>, Arc<RwLock<_>>> = HashMap::new();

    for src_arc in &all_nodes {
        let src_ptr = Arc::as_ptr(src_arc);
        node_map.insert(src_ptr, src_arc.clone());
        let guard = src_arc.read().unwrap();
        for dest_map in guard.children().values() {
            for child_wrapper in dest_map.keys() {
                if let Some(child_arc) = child_wrapper.upgrade() {
                    let child_ptr = Arc::as_ptr(&child_arc);
                    predecessors.entry(child_ptr).or_default().push(src_ptr);
                }
            }
        }
    }

    // 2. Initialize live_tokens and worklist.
    let mut live_tokens: HashMap<*const RwLock<_>, LLMTokenBV> = HashMap::new();
    let mut worklist: VecDeque<*const RwLock<_>> = VecDeque::new();
    let mut in_worklist: HashSet<*const RwLock<_>> = HashSet::new();

    for node_arc in &all_nodes {
        let node_ptr = Arc::as_ptr(node_arc);
        let guard = node_arc.read().unwrap();
        if guard.value.end {
            live_tokens.insert(node_ptr, all_llm_tokens.clone());
            if in_worklist.insert(node_ptr) {
                worklist.push_back(node_ptr);
            }
        } else {
            live_tokens.insert(node_ptr, LLMTokenBV::zeros());
        }
    }

    // 3. Fixed-point iteration to compute live tokens.
    while let Some(node_ptr) = worklist.pop_front() {
        in_worklist.remove(&node_ptr);
        let node_live_tokens = live_tokens.get(&node_ptr).unwrap().clone();
        if node_live_tokens.is_empty() { continue; }

        if let Some(preds) = predecessors.get(&node_ptr) {
            for &pred_ptr in preds {
                let pred_arc = node_map.get(&pred_ptr).unwrap();
                let mut changed = false;

                let pred_guard = pred_arc.read().unwrap();
                let mut total_contrib = LLMTokenBV::zeros();
                for dest_map in pred_guard.children().values() {
                    for (child_wrapper, edge_bv) in dest_map.iter() {
                        if let Some(child_arc) = child_wrapper.upgrade() {
                            if Arc::as_ptr(&child_arc) == node_ptr {
                                total_contrib |= &(&node_live_tokens & edge_bv);
                            }
                        }
                    }
                }
                drop(pred_guard);

                if !total_contrib.is_empty() {
                    let pred_live_tokens = live_tokens.get_mut(&pred_ptr).unwrap();
                    let old_len = pred_live_tokens.len();
                    *pred_live_tokens |= &total_contrib;
                    if pred_live_tokens.len() != old_len {
                        changed = true;
                    }
                }

                if changed {
                    if in_worklist.insert(pred_ptr) {
                        worklist.push_back(pred_ptr);
                    }
                }
            }
        }
    }

    // 4. Prune edges based on computed live_tokens.
    for node_arc in &all_nodes {
        let mut guard = node_arc.write().unwrap();
        guard.children_mut().retain(|_ek, dest_map| {
            dest_map.retain(|child_wrapper, edge_bv| {
                if let Some(child_arc) = child_wrapper.upgrade() {
                    let child_ptr = Arc::as_ptr(&child_arc);
                    if let Some(child_live_tokens) = live_tokens.get(&child_ptr) {
                        *edge_bv &= child_live_tokens;
                        !edge_bv.is_empty()
                    } else {
                        false
                    }
                } else {
                    false // Weak pointer expired, prune.
                }
            });
            !dest_map.is_empty()
        });
    }

    // 5. Prune roots if they are no longer live.
    roots.retain(|_sid, root_arc| {
        let root_ptr = Arc::as_ptr(root_arc);
        live_tokens.get(&root_ptr).map_or(false, |bv| !bv.is_empty())
    });
}

fn merge_nodes_generic<EK>(
    roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>>
)
where 
    EK: Ord + Clone + Hash + Send + Sync + 'static,
    Trie<EK, LLMTokenBV, PrecomputedNodeContents>: Eq + Hash + Clone + Send + Sync,
{
    crate::debug!(2, "Merging identical subtrees in precomputed trie.");
    let mut canonical_nodes: HashMap<Trie<EK, LLMTokenBV, PrecomputedNodeContents>, Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>> = HashMap::new();
    let mut visited: HashMap<*const RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>, Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>> = HashMap::new();

    let mut new_roots = BTreeMap::new();
    for (sid, root_arc) in roots.iter() {
        let canonical_root = deduplicate_recursive_generic(root_arc.clone(), &mut canonical_nodes, &mut visited);
        new_roots.insert(*sid, canonical_root);
    }
    *roots = new_roots;
    crate::debug!(2, "Finished merging subtrees. Canonical nodes: {}", canonical_nodes.len());
}

fn deduplicate_recursive_generic<EK>(
    node_arc: Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>,
    canonical_nodes: &mut HashMap<Trie<EK, LLMTokenBV, PrecomputedNodeContents>, Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>>,
    visited: &mut HashMap<*const RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>, Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>>,
) -> Arc<RwLock<Trie<EK, LLMTokenBV, PrecomputedNodeContents>>>
where 
    EK: Ord + Clone + Hash + Send + Sync + 'static,
    Trie<EK, LLMTokenBV, PrecomputedNodeContents>: Eq + Hash + Clone + Send + Sync,
{
    let node_ptr = Arc::as_ptr(&node_arc);
    if let Some(canonical_arc) = visited.get(&node_ptr) {
        return canonical_arc.clone();
    }

    // Post-order traversal: first, canonicalize all children.
    let mut new_children_map = BTreeMap::new();
    let mut children_changed = false;

    {
        let node_guard = node_arc.read().unwrap();
        for (edge_key, dest_map) in node_guard.children() {
            let mut new_dest_map = OrderedHashMap::new();
            for (node_ptr_wrapper, edge_val) in dest_map.iter() {
                if let Some(child_arc) = node_ptr_wrapper.upgrade() {
                    let canonical_child_arc = deduplicate_recursive_generic(child_arc.clone(), canonical_nodes, visited);
                    if !Arc::ptr_eq(&child_arc, &canonical_child_arc) {
                        children_changed = true;
                    }
                    let new_node_ptr_wrapper = if node_ptr_wrapper.is_strong() {
                        NodePtr::Strong(ArcPtrWrapper::new(canonical_child_arc))
                    } else {
                        NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(&canonical_child_arc)))
                    };
                    new_dest_map.insert(new_node_ptr_wrapper, edge_val.clone());
                }
            }
            if !new_dest_map.is_empty() {
                new_children_map.insert(edge_key.clone(), new_dest_map);
            }
        }
    }

    if children_changed {
        let mut node_guard = node_arc.write().unwrap();
        *node_guard.children_mut() = new_children_map;
    }

    let canonical_arc = {
        let node_guard = node_arc.read().unwrap();
        let node_content = (*node_guard).clone();
        canonical_nodes.entry(node_content).or_insert_with(|| node_arc.clone()).clone()
    };

    visited.insert(node_ptr, canonical_arc.clone());
    canonical_arc
}

struct Precomputer<'r> {
    tokenizer:        &'r Regex,
    parser:           Option<&'r GLRParser>,
    llm_vocab:        Option<Arc<LLMVocab>>,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode>>>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats:            PrecomputeStats,
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    // Map each precompute node to the set of LLM tokens that can pass through it.
    tags:             RefCell<HashMap<NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV>>,
    end_node:         ArcPtrWrapper<RwLock<PrecomputeNode>>,
}

impl<'r> Precomputer<'r> {
    fn new(
        tokenizer:        &'r Regex,
        parser:           Option<&'r GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,                       
        merge_threshold:  usize,
        terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>, // New parameter
        ignore_terminal_id: Option<TerminalID>,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map 
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone())) 
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut roots = BTreeMap::new();
        for sid in tokenizer.iter_states() {
            roots.insert(
                sid,
                Arc::new(RwLock::new(PrecomputeNode::new(
                    PrecomputedNodeContents::no_end(),
                ))),
            );
        }

        let total_nodes = count_vocab_nodes(&vocab.root);
        let pb = ProgressBar::new(total_nodes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] \
                           [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        Self {
            tokenizer,
            parser,
            llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            all_llm_tokens: HybridBitset::max_ones(),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
            terminal_follow_map, // Store the map
            ignore_terminal_id,
            tags: RefCell::new(HashMap::new()),
            end_node: ArcPtrWrapper::new(Arc::new(RwLock::new(PrecomputeNode::new(PrecomputedNodeContents::end())))),
        }
    }

    fn possible_matches(&self, vocab_node: &VocabPrefixTreeNode, tokenizer_state_id: TokenizerStateID) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key_ptr = vocab_node as *const VocabPrefixTreeNode;

        if let Some(cached_for_vocab_node) = self.possible_matches.borrow().get(&cache_key_ptr) {
            if let Some(cached_result) = cached_for_vocab_node.get(&tokenizer_state_id) {
                return cached_result.clone();
            }
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let exec_result = self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_arc.reachable_token_ids();
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros) |= applicable_tokens;
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: BTreeSet<_> = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(final_state_val)).into_iter().collect();
                let matches_here: BTreeSet<_> = exec_result.matches.iter().map(|m| GrammarTokenID(m.id)).collect();
                let possible_new_matches = &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(child_vocab_arc, TokenizerStateID(final_state_val));
                    for (token, bv) in next_results {
                        *result_map.entry(token).or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }

        self.possible_matches.borrow_mut().entry(cache_key_ptr).or_default().insert(tokenizer_state_id, result_map.clone());

        result_map
    }

    fn run_dfs(&mut self) {
        let mut assoc: BTreeMap<
            TokenizerStateID,
            OrderedHashSet<NodePtr<RwLock<PrecomputeNode>>>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(NodePtr::Strong(ArcPtrWrapper::new(arc.clone())));
        }

        crate::debug!(2, "Starting precompute DFS");
        crate::debug!(3, "Roots for each tokenizer state:");
        for (sid, root) in &self.roots {
            crate::debug!(6, "  {}: {:p}", sid.0, Arc::as_ptr(root));
        }
        self.dfs(&self.vocab.root, assoc, HashMap::new());
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish_with_message("Precomputation complete");
        crate::debug!(2, "Precomputation complete");
    }

    fn refine_tokens_forward(&mut self) {
        refine_tokens_forward_pass_generic(&mut self.roots, &self.all_llm_tokens);
    }

    fn optimize_precomputed_via_substring_parser(&mut self) {
        if self.parser.is_none() {
            crate::debug!(2, "Skipping optimization via substring parser as parser is None");
            return;
        }
        crate::debug!(2, "Optimizing precomputed trie via substring parser");

        // Collect all unique nodes from all roots to process each node only once.
        let mut all_nodes: Vec<Arc<RwLock<PrecomputeNode>>> = Vec::new();
        let mut seen_nodes = HashSet::new();
        for root_arc in self.roots.values() {
            for node in Trie::all_nodes(root_arc.clone()) {
                if seen_nodes.insert(Arc::as_ptr(&node)) {
                    all_nodes.push(node);
                }
            }
        }

        let pb = ProgressBar::new(all_nodes.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [optimizing precomputed trie] [{elapsed_precise}] \
                           [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        // Try to find an existing end node in the whole graph to reuse.
        let mut existing_end_node: Option<Arc<RwLock<PrecomputeNode>>> = None;
        for n in &all_nodes {
            if n.read().unwrap().value.end {
                existing_end_node = Some(n.clone());
                break;
            }
        }

        // Process each node independently.
        for node_arc in &all_nodes {
            pb.inc(1);
            // Snapshot the outgoing edges so we can run analyses without holding the lock.
            let edges: Vec<(Option<GrammarTokenID>, OrderedHashMap<NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV>)> = {
                let guard = node_arc.read().expect("poison");
                guard.children()
                    .iter()
                    .map(|(ek, dst_map)| (ek.clone(), dst_map.clone()))
                    .collect()
            };

            // For each grammar edge (Some(gt)), compute can/must masks and mutate the trie.
            for (edge_key_opt, dest_map_snapshot) in edges {
                let gtid = match edge_key_opt {
                    Some(gt) => gt,
                    None => continue, // Skip None-edges here.
                };

                if dest_map_snapshot.is_empty() {
                    continue;
                }

                // Build initial values: for each child under this edge, create a GLR
                // substring state, step with `gtid`, restrict tokens by child's BV.
                let mut initial_values: Vec<(Arc<RwLock<PrecomputeNode>>, GLRParserState)> = Vec::new();
                for (child_wrapper, edge_bv) in dest_map_snapshot.iter() {
                    let mut glr = self.parser.unwrap().init_glr_substring_parser(self.llm_vocab.clone());
                    glr.process_token(gtid); // GrammarTokenID is the same underlying type
                    // Restrict this path to the LLM tokens permitted on this edge.
                    {
                        let mut memo = HashMap::new();
                        allow_only_llm_tokens_and_prune_arc(&mut glr.active_state.stack, edge_bv, &mut memo);
                    }
                    if glr.is_ok() {
                        initial_values.push((child_wrapper.upgrade().unwrap().clone(), glr));
                    }
                }

                if initial_values.is_empty() {
                    // All paths died immediately; we will remove this edge entirely below.
                    // Continue to clean-up phase.
                }

                // Pass 1: tokens that CAN reach an end (using normal merges).
                let can_mask = self.walk_subtree_and_collect_mask(&initial_values, /*merge_intersection=*/false);

                // Pass 2: tokens that MUST reach an end (using intersection-based merges).
                // let must_mask = self.walk_subtree_and_collect_mask(&initial_values, /*merge_intersection=*/true);
                let must_mask = LLMTokenBV::zeros();

                // Mutate this node's children: intersect start-edge LLM tokens with `can_mask`.
                // Also, if `must_mask` is non-empty and the start-edge does not already have
                // an end-node destination, add a None-edge to an end node with `must_mask`,
                // and subtract `must_mask` from the start-edge tokens.
                {
                    let mut node_guard = node_arc.write().expect("poison");

                    // Get the mutable destination map for this grammar edge; it may have been
                    // removed by earlier iterations, so recheck existence.
                    let dest_map_mut = match node_guard.children_mut().get_mut(&Some(gtid)) {
                        Some(dm) => dm,
                        None => continue,
                    };

                    // 1) Intersect each child's BV with `can_mask`. Collect empties to remove.
                    let mut to_remove: Vec<NodePtr<RwLock<PrecomputeNode>>> = Vec::new();
                    for (child_w, bv) in dest_map_mut.iter_mut() {
                        *bv &= &can_mask;
                        if bv.is_empty() {
                            to_remove.push(child_w.clone());
                        }
                    }
                    for k in to_remove {
                        dest_map_mut.remove(&k);
                    }

                    // If the edge is completely dead now, remove it.
                    if dest_map_mut.is_empty() {
                        node_guard.children_mut().remove(&Some(gtid));
                        continue;
                    }

                    // 2) If this edge already goes to an end node, do not add a None shortcut.
                    let mut has_end_dest = false;
                    for (child_w, _bv) in dest_map_mut.iter() {
                        if child_w.upgrade().unwrap().read().unwrap().value.end {
                            has_end_dest = true;
                            break;
                        }
                    }

                    if !has_end_dest && !must_mask.is_empty() {
                        // Subtract must_mask from the start-edge BVs.
                        let mut to_remove2: Vec<NodePtr<RwLock<PrecomputeNode>>> = Vec::new();
                        for (child_w, bv) in dest_map_mut.iter_mut() {
                            *bv -= &must_mask;
                            if bv.is_empty() {
                                to_remove2.push(child_w.clone());
                            }
                        }
                        for k in to_remove2 {
                            dest_map_mut.remove(&k);
                        }
                        if dest_map_mut.is_empty() {
                            node_guard.children_mut().remove(&Some(gtid));
                        }

                        // Add (None; must_mask) -> end node
                        // Reuse an existing end node if possible; otherwise, create one.
                        let end_arc: Arc<RwLock<PrecomputeNode>> = if let Some(ref existing) = existing_end_node {
                            existing.clone()
                        } else {
                            let new_end = Arc::new(RwLock::new(PrecomputeNode::new(PrecomputedNodeContents::end())));
                            existing_end_node = Some(new_end.clone());
                            new_end
                        };

                        let dest_none = node_guard.children_mut().entry(None).or_default();
                        let end_key = NodePtr::Strong(ArcPtrWrapper::new(end_arc));
                        if let Some(existing_bv) = dest_none.get_mut(&end_key) {
                            *existing_bv |= &must_mask;
                        } else {
                            dest_none.insert(end_key, must_mask.clone());
                        }
                    }
                }
            }
        }
        pb.finish_with_message("Optimization complete");
        crate::debug!(2, "Finished optimizing precomputed trie via substring parser");
    }

    // Walk a trie subtree starting from the provided initial (node, GLRState) pairs.
    // When an end node is reached, gather the allowed LLM tokens of the GLR state
    // into a final mask:
    //   - merge_intersection = false: OR tokens from all ends (tokens that CAN succeed).
    //   - merge_intersection = true:  AND tokens across all ends (tokens that MUST succeed).
    fn walk_subtree_and_collect_mask(
        &self,
        initial: &[(Arc<RwLock<PrecomputeNode>>, GLRParserState)],
        merge_intersection: bool,
    ) -> LLMTokenBV {
        use crate::datastructures::trie::Trie;

        // Build fresh vector (special_map_grouped consumes values).
        let mut initials: Vec<(Arc<RwLock<PrecomputeNode>>, GLRParserState)> = Vec::with_capacity(initial.len());
        for (n, s) in initial {
            initials.push((n.clone(), s.clone()));
        }

        // Accumulators
        let final_mask_internal = std::cell::RefCell::new(LLMTokenBV::zeros());
        let first_end_seen = std::cell::RefCell::new(false);

        Trie::special_map_grouped(
            initials,
            // step: given current GLR state, edge key (Option<GrammarTokenID>), and grouped destinations,
            // produce next (child, GLR state) items.
            |glr_s, grammar_token_opt, dest_map| {
                let mut out: Vec<(NodePtr<RwLock<PrecomputeNode>>, GLRParserState)> = Vec::new();

                match grammar_token_opt {
                    Some(gtid) => {
                        // Step once with this grammar token.
                        let mut stepped = glr_s.clone();
                        stepped.process_token(*gtid);
                        if !stepped.is_ok() {
                            return out;
                        }
                        // For each child destination, restrict tokens and enqueue.
                        for (child_w, edge_bv) in dest_map.iter() {
                            let mut child_state = stepped.clone();
                            let mut memo = HashMap::new();
                            allow_only_llm_tokens_and_prune_arc(&mut child_state.active_state.stack, edge_bv, &mut memo);
                            if child_state.is_ok() {
                                out.push((child_w.clone(), child_state));
                            }
                        }
                    }
                    None => {
                        // No grammar token; just restrict per-child and forward the state.
                        for (child_w, edge_bv) in dest_map.iter() {
                            let mut child_state = glr_s.clone();
                            let mut memo = HashMap::new();
                            allow_only_llm_tokens_and_prune_arc(&mut child_state.active_state.stack, edge_bv, &mut memo);
                            if child_state.is_ok() {
                                out.push((child_w.clone(), child_state));
                            }
                        }
                    }
                }
                out
            },
            // merge: unify two GLR states that reached the same node.
            |glr_a, glr_b| {
                if merge_intersection {
                    // Intersection-based merging:
                    let a_tokens = glr_a.active_state.stack.allowed_llm_tokens();
                    let b_tokens = glr_b.active_state.stack.allowed_llm_tokens();
                    let common = &a_tokens & &b_tokens;
                    let mut memo = HashMap::new();
                    allow_only_llm_tokens_and_prune_arc(&mut glr_a.active_state.stack, &common, &mut memo);
                    let mut glr_b_mut = glr_b.clone();
                    let mut memo2 = HashMap::new();
                    allow_only_llm_tokens_and_prune_arc(&mut glr_b_mut.active_state.stack, &common, &mut memo2);
                    glr_a.merge_with(glr_b_mut);
                } else {
                    // Standard merging
                    glr_a.merge_with(glr_b.clone());
                }
            },
            // process: when reaching a node, update masks if it's an end node.
            |node, glr_s| {
                if node.value.end {
                    let toks = glr_s.active_state.stack.allowed_llm_tokens();
                    if merge_intersection {
                        if !*first_end_seen.borrow() {
                            *final_mask_internal.borrow_mut() = toks.clone();
                            *first_end_seen.borrow_mut() = true;
                        } else {
                            let mut tmp = final_mask_internal.borrow().clone();
                            tmp &= &toks;
                            *final_mask_internal.borrow_mut() = tmp;
                        }
                    } else {
                        *final_mask_internal.borrow_mut() |= toks;
                    }
                    // Stop descending beyond an end node (it’s a bound).
                    false
                } else {
                    true
                }
            },
        );

        final_mask_internal.into_inner()
    }

    fn replace_ignore_token_edges_with_none_edges(&mut self) {
        let ignore_tid = if let Some(id) = self.ignore_terminal_id {
            id
        } else {
            return; // No ignore token, nothing to do.
        };

        crate::debug!(2, "Replacing ignore token edges with None edges...");

        // 1. Collect all unique nodes.
        let mut seen: HashSet<*const RwLock<PrecomputeNode>> = HashSet::new();
        let mut all_nodes: Vec<Arc<RwLock<PrecomputeNode>>> = Vec::new();
        for root in self.roots.values() {
            for arc in Trie::all_nodes(root.clone()) {
                let ptr = Arc::as_ptr(&arc);
                if seen.insert(ptr) {
                    all_nodes.push(arc);
                }
            }
        }

        // 2. Iterate over each node and modify its children map.
        for node_arc in all_nodes {
            let mut node_guard = node_arc.write().expect("poison");
            
            // Check if there are any edges with the ignore token key.
            let ignore_key = Some(ignore_tid);
            if let Some(dest_map_for_ignore_token) = node_guard.children_mut().remove(&ignore_key) {
                // Get or create the destination map for None edges.
                let dest_map_for_none = node_guard.children_mut().entry(None).or_default();

                // Move each destination from the ignore token map to the None map.
                for (dest_wrapper, edge_bv) in dest_map_for_ignore_token {
                    // If an edge to this destination already exists under None, merge the bitvectors.
                    if let Some(existing_bv) = dest_map_for_none.get_mut(&dest_wrapper) {
                        *existing_bv |= &edge_bv;
                    } else {
                        dest_map_for_none.insert(dest_wrapper, edge_bv);
                    }
                }
            }
        }

        crate::debug!(2, "Done replacing ignore token edges.");
    }

    /// Simplify out `None` edges by shortcutting predecessors to successors.
    ///
    /// For every `B -(None; bv2)-> C`, and for every incoming edge `A -(x; bv1)-> B`,
    /// we:
    ///   - add/merge an edge `A -(x; bv1 ∩ bv2)-> C`
    ///   - remove the moved tokens `bv1 ∩ bv2` from `A -(x; ...)-> B`
    /// After processing all incoming edges to B, we remove all `None` edges from B.
    ///
    /// This transformation preserves behavior while eliminating `None` edges and
    /// allows subsequent pruning and merging passes to operate on a simpler graph.
    fn simplify_none_edges(&mut self) {
        crate::debug!(2, "Simplifying None edges (shortcut predecessors to successors)...");

        let root_node_ptrs: HashSet<*const RwLock<PrecomputeNode>> = self.roots.values().map(|arc| Arc::as_ptr(arc)).collect();

        // 1) Collect all unique nodes reachable from any root
        let mut seen: HashSet<*const RwLock<PrecomputeNode>> = HashSet::new();
        let mut all_nodes: Vec<Arc<RwLock<PrecomputeNode>>> = Vec::new();
        for root in self.roots.values() {
            for arc in Trie::all_nodes(root.clone()) {
                let ptr = Arc::as_ptr(&arc);
                if seen.insert(ptr) {
                    all_nodes.push(arc);
                }
            }
        }
        // Map pointer -> Arc for quick retrieval
        let mut arc_by_ptr: HashMap<*const RwLock<PrecomputeNode>, Arc<RwLock<PrecomputeNode>>> = HashMap::new();
        for n in &all_nodes {
            arc_by_ptr.insert(Arc::as_ptr(n), n.clone());
        }

        // 2) Build:
        //    - incoming[B] = vec of (A, key_x, bv1) for edges A -(x; bv1)-> B
        //    - none_edges_from[B] = vec of (C, bv2) for edges B -(None; bv2)-> C
        //    - none_union[B] = union of all bv2 for None edges from B
        let mut incoming: HashMap<
            *const RwLock<PrecomputeNode>,
            Vec<(Arc<RwLock<PrecomputeNode>>, Option<GrammarTokenID>, LLMTokenBV)>
        > = HashMap::new();
        let mut none_edges_from: HashMap<
            *const RwLock<PrecomputeNode>,
            Vec<(Arc<RwLock<PrecomputeNode>>, LLMTokenBV)>
        > = HashMap::new();
        let mut none_union: HashMap<*const RwLock<PrecomputeNode>, LLMTokenBV> = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = Arc::as_ptr(src_arc);
            let guard = src_arc.read().expect("poison");
            // Record all outgoing edges for incoming map
            for (ek, dest_map) in guard.children().iter() {
                for (child_wrap, ev_bv) in dest_map.iter() {
                    let child_arc = child_wrap.upgrade().unwrap().clone();
                    let child_ptr = Arc::as_ptr(&child_arc);
                    incoming.entry(child_ptr)
                        .or_default()
                        .push((src_arc.clone(), ek.clone(), ev_bv.clone()));
                }
            }
            // Record None edges out of src_arc (B -> C)
            if let Some(dest_map_none) = guard.children().get(&None) {
                let list = none_edges_from.entry(src_ptr).or_default();
                for (child_wrap, ev_bv) in dest_map_none.iter() {
                    list.push((child_wrap.upgrade().unwrap().clone(), ev_bv.clone()));
                    let entry = none_union.entry(src_ptr).or_insert_with(LLMTokenBV::zeros);
                    *entry |= ev_bv;
                }
            }
        }

        // 3) For every node B that has None edges to children, rewrite predecessors.
        for (b_ptr, none_edges) in none_edges_from.into_iter() {
            let union_mask = match none_union.get(&b_ptr) {
                Some(bv) if !bv.is_empty() => bv.clone(),
                _ => continue,
            };
            // If no predecessors, still remove None edges later (could help pruning)
            let in_edges = match incoming.get(&b_ptr) {
                Some(v) if !v.is_empty() => v.clone(),
                _ => {
                    // No predecessors.
                    // If B is a root node, we must not remove its None edges, as there are no
                    // predecessors to shortcut from.
                    if root_node_ptrs.contains(&b_ptr) {
                        continue; // It's a root, leave its None edges.
                    }

                    // Not a root and no predecessors means it's an unreachable internal node.
                    // It's safe to remove its outgoing None edges.
                    if let Some(b_arc) = arc_by_ptr.get(&b_ptr).cloned() {
                        let mut b_guard = b_arc.write().expect("poison");
                        b_guard.children_mut().remove(&None);
                    }
                    continue;
                }
            };

            let b_arc = match arc_by_ptr.get(&b_ptr) {
                Some(a) => a.clone(),
                None => continue,
            };
            let b_key = NodePtr::Strong(ArcPtrWrapper::new(b_arc.clone()));

            // For each incoming edge A -(x; bv1)-> B, split tokens:
            //   move:    to C with mask (bv1 ∩ bv2)
            //   leftover on A->B: bv1 - union_over_C(bv1 ∩ bv2) = bv1 ∩ (!union_mask)
            for (a_arc, edge_key, bv1_original) in in_edges.into_iter() {
                let mut total_to_move = bv1_original.clone();
                total_to_move &= &union_mask; // total tokens to redirect to all C via None edges
                if total_to_move.is_empty() {
                    continue;
                }

                let mut a_guard = a_arc.write().expect("poison");
                let dest_map = a_guard.children_mut().entry(edge_key.clone()).or_default();

                // Add/merge edges to each C with per-child mask
                for (c_arc, bv2) in &none_edges {
                    let mut to_move_for_c = bv1_original.clone();
                    to_move_for_c &= bv2;
                    if to_move_for_c.is_empty() {
                        continue;
                    }
                    let c_key = NodePtr::Strong(ArcPtrWrapper::new(c_arc.clone()));
                    if let Some(existing_ev) = dest_map.get_mut(&c_key) {
                        *existing_ev |= &to_move_for_c;
                    } else {
                        dest_map.insert(c_key, to_move_for_c);
                    }
                }

                // Reduce/remove the A -> B edge for the moved tokens
                let mut remove_b_edge = false;
                if let Some(ev_ab) = dest_map.get_mut(&b_key) {
                    *ev_ab -= &total_to_move;
                    remove_b_edge = ev_ab.is_empty();
                }
                if remove_b_edge {
                    dest_map.remove(&b_key);
                }
            }

            // Finally, remove all None edges out of B
            {
                let mut b_guard = b_arc.write().expect("poison");
                b_guard.children_mut().remove(&None);
            }
        }

        crate::debug!(2, "Done simplifying None edges.");
    }
    
    fn prune_on_no_terminal_follow(&mut self) {
        crate::debug!(2, "Pruning based on terminal follow sets.");

        let terminal_follow_map = self.terminal_follow_map;
        let ignore_terminal_id = self.ignore_terminal_id;

        let initial_nodes_and_values: Vec<_> = self.roots.values()
            .map(|root_arc| (root_arc.clone(), None))
            .collect();

        Trie::special_map(
            initial_nodes_and_values,
            // step: Propagate predecessor terminals.
            |predecessors, edge_terminal_opt, _edge_bv, _child_node| {
                match edge_terminal_opt {
                    Some(t) if Some(*t) == ignore_terminal_id => Some(predecessors.clone()),
                    Some(t) => Some(Some(BTreeSet::from([*t]))),
                    None => Some(predecessors.clone()),
                }
            },
            // merge: Union of predecessor sets from different paths.
            |existing_set, new_set| {
                match (existing_set, new_set) {
                    (None, _) => {},
                    (existing_set @ _, None) => *existing_set = None,
                    (Some(existing), Some(new)) => existing.extend(new),
                }
            },
            // process: Prune outgoing edges based on allowed follows.
            move |node, maybe_all_immediate_predecessors| {
                // If there are no preceding terminals (e.g., root or only None-edges path from root),
                // all outgoing terminals are considered valid.
                if maybe_all_immediate_predecessors.is_none() {
                    return true; // Continue traversal
                }

                // Compute the set of all allowed terminals that can follow any of the immediate predecessors.
                let mut allowed_follow_terminals = BTreeSet::new();
                if let Some(all_immediate_predecessors) = &*maybe_all_immediate_predecessors {
                    for preceding_terminal in all_immediate_predecessors {
                        if let Some(follow_set) = terminal_follow_map.get(preceding_terminal) {
                            allowed_follow_terminals.extend(follow_set.iter().cloned());
                        }
                    }
                }

                // Prune children of the current node.
                node.children_mut().retain(|edge_terminal_opt, _dest_map| {
                    match edge_terminal_opt {
                        // Keep edges with terminals that are in the allowed follow set (or ignore edges).
                        Some(edge_terminal) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
                        // Always keep `None` edges, as they don't represent grammar terminals.
                        None => true,
                    }
                });
    
                true // Continue traversal
            },
        );
        crate::debug!(2, "Finished pruning based on terminal follow sets.");
    }

    fn prune_dead_paths(&mut self) {
        prune_dead_paths_generic(&mut self.roots, &self.all_llm_tokens);
    }

    fn factor_common_destinations(&mut self) {
        crate::debug!(2, "Factoring out common destinations to reduce non-None edges.");

        const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3; // Configurable threshold

        // 1. Collect all nodes in the graph.
        let mut all_nodes = Vec::new();
        let mut seen = HashSet::new();
        for root in self.roots.values() {
            for node in Trie::all_nodes(root.clone()) {
                if seen.insert(Arc::as_ptr(&node)) {
                    all_nodes.push(node);
                }
            }
        }
        let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (Arc::as_ptr(n), n.clone())).collect();

        // 2. Build an incoming edge map for every node.
        // incoming_map: D_ptr -> (gtid -> Vec<(S_ptr, bv)>)
        let mut incoming_map: HashMap<
            *const RwLock<PrecomputeNode>, // Dst node ptr
            HashMap<
                GrammarTokenID, // Edge key 'gtid'
                Vec<(*const RwLock<PrecomputeNode>, LLMTokenBV)>, // List of (Src node ptr, edge bv)
            >,
        > = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = Arc::as_ptr(src_arc);
            let guard = src_arc.read().expect("poison");
            for (ek_opt, dest_map) in guard.children() {
                if let Some(gtid) = ek_opt { // Only consider non-None edges
                    for (dest_wrapper, bv) in dest_map {
                        if let Some(dest_arc) = dest_wrapper.upgrade() {
                            let dest_ptr = Arc::as_ptr(&dest_arc);
                            incoming_map
                                .entry(dest_ptr)
                                .or_default()
                                .entry(*gtid)
                                .or_default()
                                .push((src_ptr, bv.clone()));
                        }
                    }
                }
            }
        }

        // 3. Iterate through the map and find factoring opportunities.
        for (dest_ptr, edges_by_key) in incoming_map {
            for (gtid, sources) in edges_by_key {
                if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                    // Opportunity found!
                    let dest_arc = arc_map.get(&dest_ptr).unwrap().clone();

                    // a. Create a new intermediate node `I`.
                    let intermediate_node = Arc::new(RwLock::new(PrecomputeNode::new(
                        PrecomputedNodeContents::no_end(),
                    )));

                    // b. Add edge I --(gtid)--> D
                    let mut union_bv = LLMTokenBV::zeros();
                    for (_, bv) in &sources {
                        union_bv |= bv;
                    }

                    {
                        let mut intermediate_guard = intermediate_node.write().expect("poison");
                        let mut edge_val_opt = Some(union_bv);
                        // No cycle possible since I is new. Use unchecked for speed.
                        // Depth will be propagated to D.
                        intermediate_guard.try_insert_unchecked(Some(gtid), &mut edge_val_opt, dest_arc.clone())
                            .expect("Cycle detected when adding factored edge; this should not happen.");
                    }

                    // c. For each source, remove old edge and add new `None` edge to `I`.
                    for (src_ptr, bv) in &sources {
                        let src_arc = arc_map.get(src_ptr).unwrap();
                        let mut src_guard = src_arc.write().expect("poison");

                        // Remove S --(gtid)--> D
                        if let Some(dest_map_for_gtid) = src_guard.children_mut().get_mut(&Some(gtid)) {
                            dest_map_for_gtid.remove(&NodePtr::Strong(ArcPtrWrapper::new(dest_arc.clone())));
                            if dest_map_for_gtid.is_empty() {
                                src_guard.children_mut().remove(&Some(gtid));
                            }
                        }

                        // Add S --(None)--> I
                        let mut edge_val_opt = Some(bv.clone());
                        src_guard.try_insert_unchecked(None, &mut edge_val_opt, intermediate_node.clone())
                            .expect("Cycle detected when adding None edge to intermediate node; this should not happen.");
                    }
                }
            }
        }
        crate::debug!(2, "Finished factoring common destinations.");
    }

    fn merge_nodes_basic(&mut self) {
        crate::debug!(2, "Merging nodes: first collecting unique roots and their canonical Arcs");
        let mut content_to_canonical_arc_map: HashMap<PrecomputeNode, Arc<RwLock<PrecomputeNode>>> = HashMap::new();

        for (_tokenizer_state_id, root_arc_ref) in &self.roots {
            crate::debug!(3, "Merging nodes: first collecting unique roots and their canonical Arcs: Root {:p}", root_arc_ref);
            let node_content = root_arc_ref.read().unwrap().clone();
            crate::debug!(3, "Merging nodes: first collecting unique roots and their canonical Arcs: Root {:p} lock acquired, content: {:?}", root_arc_ref, node_content);
            // This will associate node_content with root_arc_ref.clone().
            // If node_content was already in the map, its associated Arc gets updated to root_arc_ref.clone().
            // This implements a "last one wins" policy for which Arc becomes canonical for a given content.
            content_to_canonical_arc_map.insert(node_content, root_arc_ref.clone());
        }

        crate::debug!(2, "Merging nodes: second pass, rewriting roots in self.roots to point to canonical Arcs");
        for (_tokenizer_state_id, root_arc_in_self_roots_mut) in &mut self.roots {
            crate::debug!(3, "Merging nodes: second pass, rewriting roots in self.roots to point to canonical Arcs: Root {:p}", root_arc_in_self_roots_mut);
            let current_content = root_arc_in_self_roots_mut.read().unwrap().clone();
            if let Some(canonical_arc) = content_to_canonical_arc_map.get(&current_content) {
                crate::debug!(3, "Merging nodes: canonical Arc found for {:?}, updating root to {:p}", current_content, canonical_arc);
                *root_arc_in_self_roots_mut = canonical_arc.clone();
            } else {
                // This should not happen if content_to_canonical_arc_map was built correctly from all roots
                // and PrecomputeNode's Ord/Eq implementations are consistent.
                panic!(
                    "Error in merge_nodes: content of a root from self.roots (tokenizer_state_id: {:?}) \
                    was not found in the canonical map. This indicates a potential issue with \
                    PrecomputeNode's Ord/Eq implementation or the merge_nodes logic itself.",
                    _tokenizer_state_id);
            };
        }
    }

    fn merge_nodes(&mut self) {
        merge_nodes_generic(&mut self.roots);
    }

    fn finish(
        mut self,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        internal_max_llm_token: usize,
    ) -> BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode>>> {

        calculate_final_stats(&self.roots, &mut self.stats);
        print_precompute_stats(&self.stats, token_name_map);

        self.roots
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<
            TokenizerStateID,
            OrderedHashSet<NodePtr<RwLock<PrecomputeNode>>>,
        >,
        no_go: HashMap<NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV>,

    ) {
        self.pb.inc(1);

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, OrderedHashSet<NodePtr<RwLock<PrecomputeNode>>>>> = BTreeMap::new();
            work_queue.insert(0, assoc_by_state.clone());

            let mut next_level_assoc: BTreeMap<_, OrderedHashSet<_>> = BTreeMap::new();

            while let Some((pos, states_at_pos)) = work_queue.pop_first() {
                if pos == segment_bytes.len() {
                    for (tokenizer_state_id, nodes) in states_at_pos {
                        next_level_assoc.entry(tokenizer_state_id).or_default().extend(nodes);
                    }
                    continue;
                }

                for (tokenizer_state_id, precompute_nodes) in states_at_pos {
                    let exec_result = self.tokenizer.execute_from_state(&segment_bytes[pos..], tokenizer_state_id);

                    let possible_matches_at_end = if let Some(end_state_val) = exec_result.end_state {
                        self.possible_matches(child_vocab_node, TokenizerStateID(end_state_val))
                    } else {
                        BTreeMap::new()
                    };

                    for match_info in &exec_result.matches {
                        let terminal_id = GrammarTokenID(match_info.id);
                        let next_pos = pos + match_info.width;

                        // TODO: could make this so much faster by moving loop down...
                        for src_node_wrapper in &precompute_nodes {
                            if next_pos == segment_bytes.len() {
                                // TODO: should be some way of avoiding ignored terminal here.
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let mut inserter = EdgeInserter::new(
                                    src_node_wrapper.upgrade().unwrap().clone(),
                                    Some(terminal_id),
                                    edge_bv,
                                    |e, n| *e |= n,
                                );
                                // Print the source node.
                                // dump_precompute_trie_recursive(src_node_wrapper, String::new(), &mut HashSet::new(), None);
                                inserter.try_destination(self.end_node.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
                            }

                            let mut edge_bv = child_vocab_node.reachable_token_ids().clone();
                            if next_pos == segment_bytes.len() {
                                edge_bv.set(child_vocab_node.token_id(), false);
                            }
                            if let Some(matches_for_terminal) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv -= matches_for_terminal;
                            }

                            if edge_bv.is_empty() { continue; }

                            let mut inserter = EdgeInserter::new(
                                src_node_wrapper.upgrade().unwrap().clone(),
                                Some(terminal_id),
                                edge_bv.clone(),
                                |e, n| *e |= n,
                            );

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos).or_default().entry(next_tokenizer_state).or_default();

                            inserter = inserter.try_destinations_iter(dest_nodes_in_queue.iter().filter_map(|w| w.upgrade()).filter(|w| !w.read().unwrap().value.end));

                            if true {
                                let children_of_src: Vec<_> = src_node_wrapper.upgrade().unwrap().read().unwrap().children().values().flat_map(|m| m.keys().cloned()).collect();
                                let tags = self.tags.borrow();
                                let eligible_children = children_of_src.iter().filter_map(|child_node_ptr| {
                                    if let Some(child_arc) = child_node_ptr.upgrade() {
                                        if tags.get(child_node_ptr).map_or(true, |tag| (tag & &edge_bv).is_empty()) && !child_arc.read().unwrap().value.end {
                                            Some(child_arc)
                                        } else { None }
                                    } else { None }
                                });
                                inserter = inserter.try_destinations_iter(eligible_children);
                                drop(tags);
                            }

                            let result_node = inserter.else_create_destination_with_value(PrecomputedNodeContents::no_end()).unwrap();
                            let result_node_ptr = NodePtr::Strong(ArcPtrWrapper::new(result_node.clone()));
                            dest_nodes_in_queue.insert(result_node_ptr.clone());
                            *self.tags.borrow_mut().entry(result_node_ptr).or_insert_with(HybridBitset::zeros) |= &edge_bv;
                        }
                    }

                    if let Some(end_state_val) = exec_result.end_state {
                        let possible_final_tokens = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_val));
                        for terminal_id in possible_final_tokens {
                            for src_node_wrapper in &precompute_nodes {
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let mut inserter = EdgeInserter::new(
                                    src_node_wrapper.upgrade().unwrap().clone(),
                                    Some(terminal_id),
                                    edge_bv,
                                    |e, n| *e |= n,
                                );
                                // Print the source node.
                                // dump_precompute_trie_recursive(src_node_wrapper, String::new(), &mut HashSet::new(), None);
                                inserter.try_destination(self.end_node.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
                            }
                        }
                        next_level_assoc.entry(TokenizerStateID(end_state_val)).or_default().extend(precompute_nodes.iter().cloned());
                    }
                }
            }



            if !next_level_assoc.is_empty() {
                self.dfs(child_vocab_node, next_level_assoc, no_go.clone());
            }
        }
    }

    fn merge_handles(
        &self,
        set: &OrderedHashSet<NodePtr<RwLock<PrecomputeNode>>>,
    ) -> OrderedHashSet<NodePtr<RwLock<PrecomputeNode>>> {
        if set.len() <= self.merge_threshold {
            return set.clone();
        }

        let merged_node_arc = Arc::new(RwLock::new(PrecomputeNode::new( 
            PrecomputedNodeContents::no_end(),
        )));

        for child_wrapper in set { 
            let edge_tokens_for_merge = self.all_llm_tokens.clone(); // This seems wrong.
            let mut inserter = EdgeInserter::new(
                child_wrapper.upgrade().unwrap().clone(),
                None::<GrammarTokenID>,   
                edge_tokens_for_merge.clone(), 
                |existing_edge_data: &mut HybridBitset, new_edge_data: HybridBitset| *existing_edge_data |= new_edge_data,
            );

            inserter = inserter.try_children();
            inserter = inserter.try_destination(merged_node_arc.clone());
        }

        let mut out = OrderedHashSet::new();
        out.insert(NodePtr::Strong(ArcPtrWrapper::new(merged_node_arc))); 
        out
    }
}

fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
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
                max_nodes: 50,
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
        // self.get_mask1()
        self.get_mask2()
    }

    #[time_it]
    pub fn get_mask1(&self) -> LLMTokenBV {
        let t0 = std::time::Instant::now();
        crate::debug!(2, "Computing mask with {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
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

        let step_counts = Arc::new(RwLock::new(BTreeMap::<TerminalID, StepCount>::new()));

        let mut initial_values_for_map: Vec<(Arc<RwLock<PrecomputeNode>>, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            // crate::debug!(4, "Initializing GSS for state {}", tokenizer_state_id.0);
            // Ensure the GLR state's GSS stack is not empty before proceeding
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_arc) = self.parent.precomputed.get(tokenizer_state_id) {
                let mut forbidden_llm_tokens = LLMTokenBV::zeros();
                let disallowed_terminals_l2 = glr_state.active_state.stack.disallowed_terminals();

                // Iterate over ranges of tokenizer states that have the same set of disallowed terminals.
                for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
                    if disallowed_terminals_for_range.is_empty() {
                        continue;
                    }

                    // Get a sub-view of possible_matches that covers this range of tokenizer states.
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
                    // glr_state.log_gss(format!("Subtracting forbidden LLM tokens: {:?}", forbidden_llm_tokens).as_str(), TerminalID(0));
                    disallow_llm_tokens_and_prune_arc(&mut glr_state.active_state.stack, &forbidden_llm_tokens, &mut HashMap::new());
                    // glr_state.log_gss("Done subtracting forbidden LLM tokens.", TerminalID(0));
                }
                initial_values_for_map.push((precomputed_trie_root_arc.clone(), glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             // This can happen if all GLR states had empty GSS stacks or no corresponding precomputed tries.
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));

        let step_counts_clone1 = Arc::clone(&step_counts);
        let step_counts_clone2 = Arc::clone(&step_counts);

        crate::profiler::reset();

        Trie::special_map_grouped(
            initial_values_for_map,
            // step_fn: (current_glr_state, edge_grammar_token_opt, destinations_map)
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
                        // If the potential additional tokens are already in the final mask, skip stepping.
                        crate::debug!(4, "Skipping step for grammar token {:?} as all edge LLM tokens are already in final mask.", grammar_token_opt);
                        return Vec::new();
                    }
                    });
                }

                // let mut glr_s = glr_s.clone();
                // disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());

                // Count num end nodes vs non end nodes
                let mut num_end = 0;
                let mut num_non_end = 0;
                for child_node_trie_data in dest_map.keys() {
                    if child_node_trie_data.upgrade().unwrap().read().unwrap().value.end {
                        num_end += 1;
                    } else {
                        num_non_end += 1;
                    }
                }
                timeit!(format!("get_mask step_fn - end only? {}", num_end > 0 && num_non_end == 0), {
                    if num_non_end == 0 {
                        if let Some(gtid) = grammar_token_opt {
                            // let stats = gather_gss_stats(&[glr_s.active_state.stack.as_ref()]);
                            // crate::debug!(3, "Step for grammar token {:?} with only end nodes, GSS stats: {:#?}", gtid, stats);
                            // Perhaps we can avoid stepping by calling `has_action_for`
                            match glr_s.has_action_for(*gtid) {
                                Some(glr_s_llm_tokens) => {
                                    timeit!(format!("get_mask step_fn - has_action_for"), {
                                        // This token will succeed
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
                                        // Inconclusive
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

                        let terminal_name = self.parent.parser.terminal_map.get_by_right(gtid)
                            .map(|s| s.to_string())
                            .unwrap_or("UNKNOWN_TERMINAL".to_string());
                        // timeit!(format!("get_mask step for terminal '{}'", terminal_name), {
                        glr_s.process_token(*gtid);
                        // });

                        crate::debug!(4, "glr_s.is_ok()_after_process_token: {}", glr_s.is_ok());

                        if glr_s.is_ok() {
                            entry.successful += 1;
                        } else {
                            return Vec::new();

                        }
                    }

                    // glr_s.log_gss("After stepping", grammar_token_opt.unwrap_or(TerminalID(0)));
                    // disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());

                    let mut results = Vec::new();

                    crate::debug!(4, "Processing edge: {:?}", grammar_token_opt);
                    for (child_node_trie_data, edge_llm_tokens_bv) in dest_map.iter() {
                        let mut glr_s = glr_s.clone();
                        allow_only_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &edge_llm_tokens_bv, &mut HashMap::new());
                        crate::debug!(4, "Stepping with grammar_token_opt: {:?}", grammar_token_opt);
                        glr_s.log_gss("Stepping with grammar_token_opt", grammar_token_opt.unwrap_or(TerminalID(0)), false, false);
                        crate::debug!(4, "Active LLM tokens: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                        crate::debug!(4, "Edge LLM tokens: {:?}", edge_llm_tokens_bv);
                        // crate::debug!(4, "Intersecting with edge_llm_tokens_bv: {:?}", edge_llm_tokens_bv);
                        // subtract_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());
                        // glr_s.log_gss("After intersecting", grammar_token_opt.unwrap_or(TerminalID(0)));

                        if !glr_s.is_ok() {
                            crate::debug!(4, "GLR state is not alive after step, skipping.");
                            continue;
                        }

                        if child_node_trie_data.upgrade().unwrap().read().unwrap().value.end {
                            let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                            crate::debug!(4, "Adding active tokens {:?} to final mask", glr_active_tokens);
                            // timeit!("get_mask final_mask update", {
                            *final_mask_internal.borrow_mut() |= glr_active_tokens;
                            // });
                            crate::debug!(4, "Final mask after adding end node tokens: {:?}", final_mask_internal.borrow());
                        }

                        results.push((child_node_trie_data.clone(), glr_s));
                    }
                    crate::debug!(4, "Step function results len: {}", results.len());
                    results
                })
            },
            // merge_fn
            |glr_s1, glr_s2| {
                timeit!("get_mask merge_fn", {
                    crate::debug!(4, "Active LLM tokens in glr_s1 before merge: {:?}", glr_s1.active_state.stack.allowed_llm_tokens());
                    crate::debug!(4, "Active LLM tokens in glr_s2 before merge: {:?}", glr_s2.active_state.stack.allowed_llm_tokens());
                    glr_s1.merge_with(glr_s2);
                    crate::debug!(4, "Active LLM tokens in glr_s1 after merge: {:?}", glr_s1.active_state.stack.allowed_llm_tokens());
                })
            },
            // process_fn: (precomputed_node_data, final_glr_s_for_this_path)
            |precomputed_node_data, glr_s| {
                timeit!("get_mask process_fn", {
                    crate::debug!(4, "Processing precomputed node data: {:?}", precomputed_node_data);
                    if precomputed_node_data.value.end {
                        let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                        crate::debug!(4, "Precomputed node data is an end node, adding active tokens {:?} to final mask", glr_active_tokens);
                        *final_mask_internal.borrow_mut() |= glr_active_tokens;
                        crate::debug!(4, "Final mask after adding end node tokens: {:?}", final_mask_internal.borrow());
                        false
                    } else {
                        let mut num_outgoing_edges_that_lead_to_non_end_nodes = 0;
                        for (edge_terminal_opt, dest_map) in precomputed_node_data.children().iter() {
                            if edge_terminal_opt.is_none() {
                                num_outgoing_edges_that_lead_to_non_end_nodes += 1
                            } else {
                                for (child_node_trie_data, _edge_llm_tokens_bv) in dest_map.iter() {
                                    if !child_node_trie_data.upgrade().unwrap().read().unwrap().value.end {
                                        num_outgoing_edges_that_lead_to_non_end_nodes += 1;
                                        break;
                                    }
                                }
                            }
                            if num_outgoing_edges_that_lead_to_non_end_nodes >= 2 {
                                break; // No need to check further, we have at least two non-end nodes.
                            }
                        }
                        // Print GSS stats
                        disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());
                        Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                        let stats = gather_gss_stats(&[glr_s.active_state.stack.as_ref()]);
                        // crate::debug!(3, "GSS stats for precomputed node data: {:#?}", stats);
                        let mut do_phase3 = false;
                        do_phase3 |= num_outgoing_edges_that_lead_to_non_end_nodes >= 2;
                        do_phase3 |= match LR_MODE {
                            LRMode::LR1 | LRMode::LALR_EX_SHIFT_STATES => false,
                            LRMode::LALR => true,
                        };
                        // do_phase3 |= true;
                        if do_phase3 {
                            // There will be a split.
                            // Let's do some work ahead of time to avoid redundant computations due to the upcoming split.
                            crate::debug!(4, "Processing non-end precomputed node data");
                            crate::debug!(4, "Active LLM tokens before phase 3: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                            glr_s.process_default_reductions();
                            crate::debug!(4, "After phase 3, active stack.stack.is_empty(): {}", glr_s.active_state.stack.is_empty());
                            Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                            crate::debug!(4, "Active LLM tokens after phase 3: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                            crate::debug!(4, "Disallowing LLM tokens and pruning arc for precomputed node data: {:?}", final_mask_internal.borrow());
                            Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                        }
                        crate::debug!(4, "After processing precomputed node data, active stack.stack.is_empty(): {}", glr_s.active_state.stack.is_empty());
                        crate::debug!(4, "Final active LLM tokens: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                        !glr_s.active_state.stack.is_empty()
                    }
                })
            },
        );

        crate::debug!(2, "Done main part of get_mask");
        let t1 = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t1.duration_since(t0));

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

        // Log the GSSs
        if GSS_LOGGING_ENABLED {
            crate::debug!(3, "Final GSS states after get_mask:");
            let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
            let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();
            let config = GSSPrintConfig {
                labels: Some(&labels),
                max_nodes: 300,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);
        }

        let final_mask_mapped = self.parent.internal_bv_to_original(&final_mask_internal.into_inner());

        crate::debug!(2, "Done computing mask");
        let t1 = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t1.duration_since(t0));

        final_mask_mapped
    }

    pub fn get_mask2(&self) -> LLMTokenBV {
        let t0 = std::time::Instant::now();
        crate::debug!(2, "Computing mask with {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
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

        let step_counts = Arc::new(RwLock::new(BTreeMap::<TerminalID, StepCount>::new()));

        let mut initial_values_for_map: Vec<(Arc<RwLock<PrecomputeNode2>>, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            // crate::debug!(4, "Initializing GSS for state {}", tokenizer_state_id.0);
            // Ensure the GLR state's GSS stack is not empty before proceeding
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_arc) = self.parent.precomputed2.get(tokenizer_state_id) {
                let mut forbidden_llm_tokens = LLMTokenBV::zeros();
                let disallowed_terminals_l2 = glr_state.active_state.stack.disallowed_terminals();

                // Iterate over ranges of tokenizer states that have the same set of disallowed terminals.
                for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
                    if disallowed_terminals_for_range.is_empty() {
                        continue;
                    }

                    // Get a sub-view of possible_matches that covers this range of tokenizer states.
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
                    // glr_state.log_gss(format!("Subtracting forbidden LLM tokens: {:?}", forbidden_llm_tokens).as_str(), TerminalID(0));
                    disallow_llm_tokens_and_prune_arc(&mut glr_state.active_state.stack, &forbidden_llm_tokens, &mut HashMap::new());
                    // glr_state.log_gss("Done subtracting forbidden LLM tokens.", TerminalID(0));
                }
                initial_values_for_map.push((precomputed_trie_root_arc.clone(), glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             // This can happen if all GLR states had empty GSS stacks or no corresponding precomputed tries.
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));

        let step_counts_clone1 = Arc::clone(&step_counts);
        let step_counts_clone2 = Arc::clone(&step_counts);

        crate::profiler::reset();

        Trie::special_map_grouped(
            initial_values_for_map,
            // step_fn: (current_glr_state, (k, option state ID), destinations_map)
            |glr_s, (k, expected_state_id_opt ), dest_map| {
                crate::debug!(4, "Processing step for k: {:?}, expected_state_id_opt: {:?}", k, expected_state_id_opt);
                let mut out_gsss = Vec::new();
                let popped = glr_s.active_state.stack.popn(*k);
                for popper_item in popped.iter() {
                    for peek in popper_item.peek_iter() {
                        let ok = if let Some(expected_state_id) = expected_state_id_opt {
                            expected_state_id == &peek.edge_value().state_id
                        } else {
                            true
                        };
                        if ok {
                            out_gsss.push(peek.isolated_parent());
                        }
                    }
                }
                let out_gss = GSSNode::merge_many_with_depth(1, out_gsss);
                let mut out = Vec::new();
                for (dst_node_wrapper, edge_bv) in dest_map.iter() {
                    let mut out_gss_filtered = out_gss.clone();
                    allow_only_llm_tokens_and_prune_arc(&mut out_gss_filtered, edge_bv, &mut HashMap::new());
                    let mut out_glr_s = glr_s.clone();
                    out_glr_s.active_state.stack = out_gss_filtered;
                    if out_glr_s.is_ok() {
                        out.push((dst_node_wrapper.clone(), out_glr_s));
                    }
                }
                out
            },
            // merge_fn
            |glr_s1, glr_s2| {
                crate::debug!(4, "Merging two GLR states");
                glr_s1.merge_with(glr_s2);
            },
            // process_fn: (precomputed_node_data, final_glr_s_for_this_path)
            |precomputed_node_data, glr_s| {
                let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                let keep_going = !glr_active_tokens.is_empty();
                if precomputed_node_data.value.end {
                    crate::debug!(4, "Precomputed node data is an end node, adding active tokens {:?} to final mask", glr_active_tokens);
                    *final_mask_internal.borrow_mut() |= glr_active_tokens;
                } else {
                    crate::debug!(4, "Precomputed node data is not an end node, active tokens: {:?}", glr_active_tokens);
                }
                keep_going
            },
        );

        crate::debug!(2, "Done main part of get_mask");
        let t1 = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t1.duration_since(t0));

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

        // Log the GSSs
        if GSS_LOGGING_ENABLED {
            crate::debug!(3, "Final GSS states after get_mask:");
            let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
            let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();
            let config = GSSPrintConfig {
                labels: Some(&labels),
                max_nodes: 300,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);
        }

        crate::debug!(4, "Final mask internal: {:?}", final_mask_internal.borrow());
        let final_mask_mapped = self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        crate::debug!(4, "Final mask mapped: {:?}", final_mask_mapped);

        crate::debug!(2, "Done computing mask");
        let t1 = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t1.duration_since(t0));

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

        // Handle allowed terminals
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
        // println!("State after preparation: {}", self);

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
                    // cloned_glr_s.do_phase3();

                    if cloned_glr_s.is_ok() {
                        let new_offset = offset + match_info.width;
                        // After a grammar token is consumed, the tokenizer resets for the next segment of the LLM token.
                        let next_tokenizer_id_for_segment = self.parent.tokenizer.initial_state_id();

                        let mut disallowed_terminals = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::new();
                        if let Some(end_state_id) = exec_result.end_state {
                            let mut disallowed_terminals_for_end_state = TerminalBV::zeros();
                            // Disallow this token from being matched again immediately.
                            disallowed_terminals_for_end_state.insert(match_info.id);
                            disallowed_terminals.insert_l2_bitset(end_state_id, disallowed_terminals_for_end_state);
                        }
                        disallow_terminals_and_prune_arc(&mut cloned_glr_s.active_state.stack, &disallowed_terminals, &mut HashMap::new());

                        if new_offset == llm_token_bytes.len() {
                            // reset_allowed_terminals(&mut cloned_glr_s.active_state.stack);
                            new_overall_state.entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        } else {
                            processing_queue.entry(new_offset).or_default().entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        }
                    }
                }

                if let Some(final_tokenizer_s_id_for_llm_token_segment) = exec_result.end_state {
                    // The rest of llm_token_bytes (from offset) was consumed, tokenizer ended in this state.
                    // The glr_s_at_offset is carried over. This is a state *after* the current LLM token.
                    let final_tokenizer_state = TokenizerStateID(final_tokenizer_s_id_for_llm_token_segment);
                    new_overall_state.entry(final_tokenizer_state).and_modify(|existing| existing.merge_with(glr_s_at_offset.clone())).or_insert(glr_s_at_offset.clone());
                }
            }
        }

        self.state = new_overall_state.clone();

        for glr_parser_state in self.state.values_mut() {
            glr_parser_state.process_default_reductions();
        }

        // TODO: this shouldn't be necessary, but due to some order-dependent LLM token BV weirdness in GSS, it is necessary to ensure commit order invariance.
        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        let mut fuse_memo = HashMap::new();
        for state in self.state.values_mut() {
            state.active_state.stack = fuse_predecessors_recursive(&mut state.active_state.stack, 3, &mut fuse_memo);
        }
        fuse_memo.clear();

        // let mut roots_to_simplify_arcs = Vec::new();
        // for glr_parser_state in self.state.values_mut() {
        //     if !glr_parser_state.active_state.stack.is_empty() {
        //         roots_to_simplify_arcs.push(&mut glr_parser_state.active_state.stack);
        //     }
        // }
        //
        // if !roots_to_simplify_arcs.is_empty() {
        //     GSSNode::simplify_together(&mut roots_to_simplify_arcs);
        // }

        crate::debug!(2, "Active tokenizer states after committing text (bytes {:?}): {:?}", llm_token_bytes, self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        for (tokenizer_id, glr_state) in &self.state {
            if !glr_state.active_state.stack.is_empty() { // Log only for non-empty GSS
                // glr_state.log_gss(
                //     &format!("GSS for tokenizer state {} after commit of text", tokenizer_id.0),
                //     TerminalID(0)
                // );
            }
        }
    }

    pub fn is_active(&self) -> bool {
        !self.state.is_empty() && self.state.values().any(|s| !s.active_state.stack.is_empty())
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}
