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
use crate::glr::parser::{BelowBottomReductionMode, GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, ProcessTokenAdvancedConfig};
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
        Self::_dump_precomputed(
            &precomputed,
            &llm_vocab.original_to_internal_id_bimap,
            &token_name_map,
            &llm_vocab.llm_token_map,
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

        let mut stats2 = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats2(&precomputed2, &mut stats2);
        crate::constraint_extra::print_precompute_stats2(&stats2);

        Self::_dump_precomputed2(
            &precomputed2,
            &llm_vocab.original_to_internal_id_bimap,
            &llm_vocab.llm_token_map,
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
        helper.simplify_none_edges(); // This can invalidate max_depth.

        // Recompute all max_depth values after major graph surgery.
        let roots_for_recompute: Vec<_> = helper.roots.values().cloned().collect();
        Trie::recompute_all_max_depths(&roots_for_recompute);

        helper.prune_dead_paths();
        helper.prune_on_no_terminal_follow();
        helper.prune_dead_paths();
        // New: prune using substring parser in "everything state" mode
        // helper.prune_with_substring_everything_state();
        helper.prune_dead_paths(); // Clean up after GLR-based pruning
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
        const BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING: bool = true;
        const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
            BelowBottomReductionMode::ContinueFromEverything
        } else {
            BelowBottomReductionMode::ContinueFromAll
        };

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

            let glr_state;

            if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
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
                glr_state = parser.init_glr_parser_from_parse_state(parse_state);
            } else {
                let new_trie2_node = Arc::new(RwLock::new(PrecomputeNode2::new(
                    PrecomputedNodeContents::no_end(),
                )));
                let mut inserter = EdgeInserter::new(
                    trie2_root.clone(),
                    (0, None),
                    LLMTokenBV::ones(internal_max_llm_token + 1),
                    |e, n| *e |= n,
                );
                inserter = inserter.try_destination(new_trie2_node.clone());
                inserter.expect("Failed to insert initial edge into Trie2 root");
                let mut acc = Acc::new_fresh();
                acc.trie2_nodes.insert(ArcPtrWrapper::new(new_trie2_node));
                let gss_root = GSSNode::new(acc);
                let gss_node = gss_root.push(ParseStateEdgeContent { state_id: parser.everything_state_id });
                gss_nodes_to_merge.push(Arc::new(gss_node));
                glr_state = parser.init_glr_parser_from_parse_state(
                    ParseState {
                        stack: GSSNode::merge_many_with_depth(usize::MAX, gss_nodes_to_merge),
                    },
                );
            }

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
                    glr_s.process_token_advanced(*gt, &ProcessTokenAdvancedConfig { below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE });
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

        let roots2: Vec<_> = precomputed2.values().cloned().collect();
        let mut all_nodes = Vec::new();
        let mut visited_ptrs = HashSet::new();
        for root_arc in precomputed2.values() {
            for node in Trie::all_nodes(&[root_arc.clone()]) {
                if visited_ptrs.insert(Arc::as_ptr(&node)) {
                    all_nodes.push(node);
                }
            }
        }
        prune_dead_paths_trie2(&mut precomputed2);
        merge_nodes_trie2(&mut precomputed2);
        let promotions2 = Trie::promote_weak_edges_to_strong(&roots2);
        crate::debug!(2, "Promoted {} weak edges to strong in precomputed trie 2.", promotions2);

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

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result = tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);

            for token_match in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token_match.id);
                // LLM tokens reachable under child_vocab_node_ref are those that start with segment_bytes
                let applicable_tokens = child_vocab_node.reachable_token_ids();
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
                        child_vocab_node, // Recurse with the child node
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

fn prune_dead_paths_trie2(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 2.");

    // Collect all unique nodes into a Vec to hold strong references to them.
    // This prevents any node from being dropped while we are modifying the graph structure,
    // which could otherwise cause a `Weak::upgrade` to fail unexpectedly.
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let _all_nodes = Trie::all_nodes(&roots_vec);

    let mut live_tokens_cache: HashMap<NodePtr<RwLock<PrecomputeNode2>>, LLMTokenBV> = HashMap::new();

    let all_llm_tokens = HybridBitset::max_ones();

    let sids_to_remove: Vec<_> = roots
        .iter()
        .filter_map(|(sid, root_arc)| {
            let root_wrapper = NodePtr::Strong(ArcPtrWrapper::new(root_arc.clone()));
            if get_live_tokens_and_prune_trie2(root_wrapper, &mut live_tokens_cache, &all_llm_tokens)
                .is_empty()
            {
                Some(*sid)
            } else {
                None
            }
        })
        .collect();

    for sid in sids_to_remove {
        roots.remove(&sid);
    }
    crate::debug!(2, "Finished pruning dead paths from trie 2.");
}

fn get_live_tokens_and_prune_trie2(
    node_wrapper: NodePtr<RwLock<PrecomputeNode2>>,
    live_tokens_cache: &mut HashMap<NodePtr<RwLock<PrecomputeNode2>>, LLMTokenBV>,
    all_llm_tokens: &LLMTokenBV,
) -> LLMTokenBV {
    if let Some(cached_bv) = live_tokens_cache.get(&node_wrapper) {
        return cached_bv.clone();
    }
    // Insert a temporary empty BV to break cycles. If we revisit this node during this
    // recursion, it will return an empty set, which is correct as no new live paths
    // have been found through it yet.
    live_tokens_cache.insert(node_wrapper.clone(), LLMTokenBV::zeros());

    let node_arc = node_wrapper.upgrade().unwrap();

    let children_to_check: Vec<NodePtr<RwLock<PrecomputeNode2>>> = {
        let node_guard = node_arc.read().unwrap();
        node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
    };

    for child_wrapper in children_to_check {
        get_live_tokens_and_prune_trie2(child_wrapper, live_tokens_cache, all_llm_tokens);
    }

    let mut live_tokens_for_this_node = LLMTokenBV::zeros();
    {
        let mut node_guard = node_arc.write().unwrap();

        if node_guard.value.end {
            live_tokens_for_this_node = all_llm_tokens.clone();
        }

        node_guard.children_mut().retain(|_edge_key, dest_map| {
            dest_map.retain(|child_wrapper, edge_value_bv| {
                let live_tokens_from_child = live_tokens_cache
                    .get(child_wrapper)
                    .expect("Child not found in live_tokens_cache. Logic error in post-order traversal.");

                let live_tokens_for_this_edge = &*edge_value_bv & live_tokens_from_child;

                if live_tokens_for_this_edge.is_empty() {
                    false
                } else {
                    *edge_value_bv = live_tokens_for_this_edge;
                    true
                }
            });
            !dest_map.is_empty()
        });

        for dest_map in node_guard.children().values() {
            for edge_bv in dest_map.values() {
                live_tokens_for_this_node |= edge_bv;
            }
        }
        node_guard.value.live_tokens = live_tokens_for_this_node.clone();
    }

    live_tokens_cache.insert(node_wrapper, live_tokens_for_this_node.clone());
    live_tokens_for_this_node
}

fn merge_nodes_trie2(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 2.");

    // Collect all unique nodes to keep them alive during the merging process.
    // This prevents weak pointers from dangling if a node's strong count temporarily drops to zero.
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let _all_nodes = Trie::all_nodes(&roots_vec);

    let mut canonical_nodes: HashMap<PrecomputeNode2, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();
    let mut visited: HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();

    let mut new_roots = BTreeMap::new();
    for (sid, root_arc) in roots.iter() {
        let canonical_root = deduplicate_recursive_trie2(root_arc.clone(), &mut canonical_nodes, &mut visited);
        new_roots.insert(*sid, canonical_root);
    }
    *roots = new_roots;
    crate::debug!(2, "Finished merging subtrees in trie 2. Canonical nodes: {}", canonical_nodes.len());
}

fn deduplicate_recursive_trie2(
    node_arc: Arc<RwLock<PrecomputeNode2>>,
    canonical_nodes: &mut HashMap<PrecomputeNode2, Arc<RwLock<PrecomputeNode2>>>,
    visited: &mut HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
) -> Arc<RwLock<PrecomputeNode2>> {
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
                    let canonical_child_arc = deduplicate_recursive_trie2(child_arc.clone(), canonical_nodes, visited);
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
        node_guard.recompute_max_depth();
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

    /// Prune the Precompute-1 trie using a substring GLR parser initialized in
    /// "everything state" mode.
    ///
    /// Forward pass:
    ///   - Traverse the trie with a single GLR state (cloned per path),
    ///     stepping the parser for token edges (Some(GrammarTokenID)).
    ///   - For every visited node, record the union of "active" LLM tokens
    ///     allowed by the GLR state that reaches that node.
    ///
    /// Backward pass:
    ///   - For every edge, keep only those tokens which:
    ///       edge_ev ∧ forward_mark(child) ∧ reachable_to_end(child)
    ///   - If a node is an "end" node, its reachable_to_end is its forward mark.
    ///   - Remove edges with empty bitsets. Subsequent prune_dead_paths will
    ///     remove now-unreachable nodes.
    fn prune_with_substring_everything_state(&mut self) {
        let parser = match self.parser {
            Some(p) => p,
            None => {
                crate::debug!(2, "Skipping GLR-based pruning: parser is None");
                return;
            }
        };

        use crate::datastructures::trie::Trie;
        type NodeDataPtr = *const Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

        // Forward mark: for each node, the union of GLR-allowed LLM tokens
        // after stepping along the path to it.
        let forward_mark: std::sync::Arc<RwLock<HashMap<NodeDataPtr, LLMTokenBV>>> =
            std::sync::Arc::new(RwLock::new(HashMap::new()));

        // Optional: Edge mark (not strictly necessary for pruning, but useful for debugging).
        #[allow(dead_code)]
        let _edge_mark: std::sync::Arc<RwLock<HashMap<(NodeDataPtr, Option<GrammarTokenID>, NodeDataPtr), LLMTokenBV>>> =
            std::sync::Arc::new(RwLock::new(HashMap::new()));

        #[derive(Clone)]
        struct ForwardCtx<'a> {
            glr: GLRParserState<'a>,
            parent_ptr: NodeDataPtr, // pointer to the current node's Trie data (child will use this as its "parent")
        }

        // Initialize a single "everything state" substring parser.
        let glr0 = parser.init_glr_substring_parser_with_everything_state(None);

        // Seed special_map with (root, glr0.clone()) for all roots.
        let mut initial: Vec<(Arc<RwLock<PrecomputeNode>>, ForwardCtx)> = Vec::new();
        for root in self.roots.values() {
            // Acquire a stable data pointer for this root.
            let root_ptr = {
                let guard = root.read().expect("poison");
                &*guard as *const _
            };
            initial.push((
                root.clone(),
                ForwardCtx {
                    glr: glr0.clone(),
                    parent_ptr: root_ptr,
                },
            ));
        }

        // Forward pass using special_map
        Trie::special_map(
            initial,
            {
                let forward_mark = forward_mark.clone();
                let _edge_mark = _edge_mark.clone();
                move |ctx: &ForwardCtx, ek: &Option<GrammarTokenID>, ev: &LLMTokenBV, child_node: &Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>| -> Option<ForwardCtx> {
                    // Step the GLR parser if this is a token edge.
                    let mut glr2 = ctx.glr.clone();
                    if let Some(tok) = ek {
                        glr2.process_token_advanced(*tok, &ProcessTokenAdvancedConfig {
                            below_bottom_mode: BelowBottomReductionMode::ContinueFromEverything,
                        });
                    }
                    if !glr2.is_ok() {
                        return None; // No viable continuation along this edge
                    }

                    // Mark the destination (child) node with the active LLM tokens.
                    let active_tokens = glr2.active_state.stack.allowed_llm_tokens();
                    let child_ptr: NodeDataPtr = child_node as *const _;

                    {
                        let mut fm = forward_mark.write().expect("poison");
                        fm.entry(child_ptr)
                            .and_modify(|bv| *bv |= &active_tokens)
                            .or_insert(active_tokens.clone());
                    }

                    // Optionally record edge marks (GLR-allowed ∧ edge constraint).
                    #[allow(unused_mut)]
                    let mut _edge_tokens = active_tokens & ev;
                    #[allow(unused_variables)]
                    if false {
                        let mut em = _edge_mark.write().expect("poison");
                        let key = (ctx.parent_ptr, *ek, child_ptr);
                        em.entry(key)
                            .and_modify(|bv| *bv |= &_edge_tokens)
                            .or_insert(_edge_tokens.clone());
                    }

                    // Pass context to child: it becomes the new parent for its outgoing edges.
                    Some(ForwardCtx {
                        glr: glr2,
                        parent_ptr: child_ptr,
                    })
                }
            },
            // Merge GLR states when multiple parents flow into the same node.
            |ctx_accum: &mut ForwardCtx, ctx_new: ForwardCtx| {
                let mut other = ctx_new;
                ctx_accum.glr.merge_with(other.glr);
                // parent_ptr should already match (the node being processed).
            },
            // Process: nothing to mutate on nodes here; keep going.
            |_node_data, _ctx| true,
        );

        // Backward pass: compute reachable-to-end tokens and trim edges in-place.
        fn node_data_ptr(
            arc: &Arc<RwLock<PrecomputeNode>>,
        ) -> NodeDataPtr {
            let guard = arc.read().expect("poison");
            &*guard as *const _
        }

        fn backward_prune_dfs(
            node_arc: Arc<RwLock<PrecomputeNode>>,
            forward_mark: &HashMap<NodeDataPtr, LLMTokenBV>,
            memo: &mut HashMap<NodeDataPtr, LLMTokenBV>,
            all_llm_tokens: &LLMTokenBV,
        ) -> LLMTokenBV {
            let this_ptr = node_data_ptr(&node_arc);
            if let Some(cached) = memo.get(&this_ptr) {
                return cached.clone();
            }
            // Mark as visited with an empty set to handle cycles.
            memo.insert(this_ptr, LLMTokenBV::zeros());

            // Snapshot children to avoid holding the lock during recursion.
            let (is_end, children_snapshot): (bool, Vec<(Option<GrammarTokenID>, Vec<(NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV)>)>) = {
                let guard = node_arc.read().expect("poison");
                let is_end = guard.value.end;
                let mut out = Vec::new();
                for (ek, dest_map) in guard.children() {
                    let mut edges = Vec::new();
                    for (np, bv) in dest_map.iter() {
                        edges.push((np.clone(), bv.clone()));
                    }
                    out.push((ek.clone(), edges));
                }
                (is_end, out)
            };

            // Compute reachable tokens from children.
            let mut reachable = LLMTokenBV::zeros();
            // For updating edges later, collect the post-trim values.
            let mut trimmed: BTreeMap<Option<GrammarTokenID>, OrderedHashMap<NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV>> = BTreeMap::new();

            for (ek, edges) in &children_snapshot {
                let mut new_map = OrderedHashMap::new();
                for (child_ptr, edge_bv) in edges {
                    if let Some(child_arc) = child_ptr.upgrade() {
                        // Recurse
                        let child_reach = backward_prune_dfs(child_arc.clone(), forward_mark, memo, all_llm_tokens);
                        // Child's forward mark (GLR constraint at child)
                        let child_ptr_data = {
                            let guard = child_arc.read().expect("poison");
                            &*guard as *const _
                        };
                        let child_forward = forward_mark.get(&child_ptr_data).cloned().unwrap_or_else(LLMTokenBV::zeros);

                        // Tokens to keep on this edge
                        let mut keep = edge_bv.clone();
                        keep &= &child_forward;
                        keep &= &child_reach;

                        if !keep.is_empty() {
                            reachable |= &keep;
                            new_map.insert(child_ptr.clone(), keep);
                        }
                    }
                }
                if !new_map.is_empty() {
                    trimmed.insert(ek.clone(), new_map);
                }
            }

            // If this node is an end node, also allow tokens recorded by forward mark at this node.
            if is_end {
                if let Some(fwd) = forward_mark.get(&this_ptr) {
                    reachable |= fwd;
                } else {
                    // Be conservative: end node with no forward mark contributes nothing extra.
                    // Another option would be "all_llm_tokens", but forward marks encode GLR acceptance.
                    let _ = all_llm_tokens; // silence unused var warning under cfgs
                }
            }

            // Apply the trimming to this node's outgoing edges.
            {
                let mut guard = node_arc.write().expect("poison");
                // Replace each edge-key's map with the trimmed version (if any), else remove it.
                let ek_list: Vec<_> = guard.children().keys().cloned().collect();
                for ek in ek_list {
                    if let Some(new_map) = trimmed.get(&ek) {
                        // Rebuild with the existing order where possible.
                        let dest_map = guard.children_mut().get_mut(&ek).expect("must exist");
                        let mut old = std::mem::take(dest_map);
                        let mut rebuilt = OrderedHashMap::new();
                        for (k, _v_old) in old.into_iter() {
                            if let Some(v_new) = new_map.get(&k) {
                                rebuilt.insert(k, v_new.clone());
                            }
                        }
                        *dest_map = rebuilt;
                    } else {
                        // Remove this edge key entirely.
                        guard.children_mut().remove(&ek);
                    }
                }
            }

            memo.insert(this_ptr, reachable.clone());
            reachable
        }

        // Collect all unique nodes reachable from all roots to drive the backward pass.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let unique_nodes = Trie::all_nodes(&roots_vec);

        // Backward pass with memoization.
        let mut memo: HashMap<NodeDataPtr, LLMTokenBV> = HashMap::new();
        let all_llm_tokens = self.all_llm_tokens.clone();
        for node_arc in &unique_nodes {
            let _ = backward_prune_dfs(node_arc.clone(), &forward_mark.read().expect("poison"), &mut memo, &all_llm_tokens);
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

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result = self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_node.reachable_token_ids();
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros) |= applicable_tokens;
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: BTreeSet<_> = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(final_state_val)).into_iter().collect();
                let matches_here: BTreeSet<_> = exec_result.matches.iter().map(|m| GrammarTokenID(m.id)).collect();
                let possible_new_matches = &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(child_vocab_node, TokenizerStateID(final_state_val));
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

    fn replace_ignore_token_edges_with_none_edges(&mut self) {
        let ignore_tid = if let Some(id) = self.ignore_terminal_id {
            id
        } else {
            return; // No ignore token, nothing to do.
        };

        crate::debug!(2, "Replacing ignore token edges with None edges...");

        // 1. Collect all unique nodes.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&roots_vec);
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
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&roots_vec);
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
            *const RwLock<PrecomputeNode>, // Dst node ptr
            HashMap<
                GrammarTokenID, // Edge key 'gtid'
                Vec<(*const RwLock<PrecomputeNode>, LLMTokenBV)>, // List of (Src node ptr, edge bv)
            >,
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

        let initial_values_for_map: Vec<_> = self.roots.values()
            .map(|root_arc| (root_arc.clone(), None))
            .collect();

        type NodePtr = *const PrecomputeNode;
        let edges_to_keep = Arc::new(RwLock::new(HashMap::<NodePtr, BTreeSet<Option<GrammarTokenID>>>::new()));

        Trie::special_map(
            initial_values_for_map,
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
            // process: Collect information about which edges to prune.
            {
                let edges_to_keep = edges_to_keep.clone();
                move |node, maybe_all_immediate_predecessors| {
                    // If there are no preceding terminals (e.g., root or only None-edges path from root),
                    // all outgoing terminals are considered valid.
                    if maybe_all_immediate_predecessors.is_none() {
                        return true; // Continue traversal, no pruning needed for this node.
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

                    let keys_to_keep: BTreeSet<_> = node.children().keys().filter(|edge_terminal_opt| {
                        match edge_terminal_opt {
                            // Keep edges with terminals that are in the allowed follow set (or ignore edges).
                            Some(edge_terminal) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
                            // Always keep `None` edges, as they don't represent grammar terminals.
                            None => true,
                        }
                    }).cloned().collect();

                    let node_ptr: NodePtr = node;
                    edges_to_keep.write().unwrap().insert(node_ptr, keys_to_keep);
        
                    true // Continue traversal
                }
            }
        );

        // Now, apply the pruning.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&roots_vec);
        let edges_to_keep = edges_to_keep.read().unwrap();
        for node_arc in all_nodes {
            let node_ptr: NodePtr = &*node_arc.read().unwrap();
            if let Some(keys_to_keep) = edges_to_keep.get(&node_ptr) {
                let mut node_guard = node_arc.write().unwrap();
                node_guard.children_mut().retain(|k, _| keys_to_keep.contains(k));
            }
        }


        crate::debug!(2, "Finished pruning based on terminal follow sets.");
    }

    fn prune_dead_paths(&mut self) {
        crate::debug!(2, "Pruning dead paths from precomputed trie.");

        // A cache of nodes to the set of "live" LLM tokens reachable from them.
        let mut live_tokens_cache: HashMap<NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV> = HashMap::new();

        // A node is "live" if it can reach a node with `value.end == true`. We do a post-order
        // traversal (DFS) from each root. `is_live_and_prune` recursively determines if a node
        // is live and prunes its dead children.
        //
        // We can't use `BTreeMap::retain` directly because its closure would borrow `self`
        // immutably (to call `get_live_tokens_and_prune`) while `retain` itself holds a mutable borrow
        // on `self.roots`. Instead, we collect the keys of roots to remove and then remove them.
        let sids_to_remove: Vec<_> = self.roots.iter().filter_map(|(sid, root_arc)| {
            let root_wrapper = NodePtr::Strong(ArcPtrWrapper::new(root_arc.clone()));
            // A root is dead if no live tokens are reachable from it.
            if self.get_live_tokens_and_prune(root_wrapper, &mut live_tokens_cache).is_empty() {
                Some(*sid) // This root is dead, mark for removal.
            } else {
                None // This root is live, keep it.
            }
        }).collect();

        for sid in sids_to_remove {
            self.roots.remove(&sid);
        }

        crate::debug!(2, "Finished pruning dead paths.");
    }

    /// Recursively computes the set of "live" LLM tokens reachable from a node
    /// and prunes its children that are not live or have dead token paths.
    /// This is a post-order traversal.
    ///
    /// - `node_wrapper`: The node to check.
    /// - `live_tokens_cache`: A cache of nodes to their live token bitvectors.
    ///
    /// Returns a `LLMTokenBV` of all live tokens reachable from `node_wrapper`.
    fn get_live_tokens_and_prune(
        &self,
        node_wrapper: NodePtr<RwLock<PrecomputeNode>>,
        live_tokens_cache: &mut HashMap<NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV>,
    ) -> LLMTokenBV {
        // If we've already computed the live tokens for this node, return the cached result.
        if let Some(cached_bv) = live_tokens_cache.get(&node_wrapper) {
            return cached_bv.clone();
        }
        // Insert a temporary empty BV to break cycles. If we revisit this node during this
        // recursion, it will return an empty set, which is correct as no new live paths
        // have been found through it yet.
        live_tokens_cache.insert(node_wrapper.clone(), LLMTokenBV::zeros());

        let node_arc = node_wrapper.upgrade().unwrap();

        // We must collect children before recursing to avoid holding the lock.
        let children_to_check: Vec<NodePtr<RwLock<PrecomputeNode>>> = {
            let node_guard = node_arc.read().unwrap();
            node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
        };

        // Recursively call on all unique children to populate the cache for them.
        for child_wrapper in children_to_check {
            self.get_live_tokens_and_prune(child_wrapper, live_tokens_cache);
        }

        // Now that the cache is populated for all children, we can prune the current node.
        let mut live_tokens_for_this_node = LLMTokenBV::zeros();
        {
            let mut node_guard = node_arc.write().unwrap();

            // A node is live if it's an end node itself. The tokens that end here are
            // on the edges pointing to this node.
            if node_guard.value.end {
                // This is the special "end node". It doesn't represent tokens itself,
                // but it is the source of "liveness". The tokens are on the edges leading *to* it.
                // When we calculate the live tokens for a parent, the edge BV leading to this
                // end node will be considered fully live. For the end node itself, we can
                // consider it to represent "all possible tokens" for the purpose of intersection,
                // so that any edge leading to it is kept.
                live_tokens_for_this_node = self.all_llm_tokens.clone();
            }

            node_guard.children_mut().retain(|_edge_key, dest_map| {
                dest_map.retain(|child_wrapper, edge_value_bv| {
                    // Get the live tokens reachable from the child node. This must be in the cache.
                    let live_tokens_from_child = live_tokens_cache.get(child_wrapper)
                        .expect("Child not found in live_tokens_cache. Logic error in post-order traversal.");

                    // The tokens on this edge that are actually live are the intersection
                    // of the edge's original tokens and the live tokens from the child.
                    let live_tokens_for_this_edge = &*edge_value_bv & live_tokens_from_child;

                    if live_tokens_for_this_edge.is_empty() {
                        false // Prune this destination, as no live paths go through it.
                    } else {
                        *edge_value_bv = live_tokens_for_this_edge; // Narrow the edge's BV.
                        true // Keep this destination.
                    }
                });
                // Keep the edge key only if it still has destinations.
                !dest_map.is_empty()
            });

            // The total live tokens for the current node are the union of all its (now narrowed) outgoing edge BVs.
            for dest_map in node_guard.children().values() {
                for edge_bv in dest_map.values() {
                live_tokens_for_this_node |= edge_bv;
            }
        }
        node_guard.value.live_tokens = live_tokens_for_this_node.clone();
    }

    // Update the cache with the final computed live tokens for this node.
        live_tokens_cache.insert(node_wrapper, live_tokens_for_this_node.clone());

        live_tokens_for_this_node
    }

    fn factor_common_destinations(&mut self) {
        crate::debug!(2, "Factoring out common destinations to reduce non-None edges.");

        const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3; // Configurable threshold

        // 1. Collect all nodes in the graph.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&roots_vec);
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

    fn merge_nodes(&mut self) {
        crate::debug!(2, "Merging identical subtrees in precomputed trie.");
        // A map from a node's content to its canonical Arc.
        let mut canonical_nodes: HashMap<PrecomputeNode, Arc<RwLock<PrecomputeNode>>> = HashMap::new();
        // A map from a node's pointer to its canonicalized Arc, to avoid re-processing.
        let mut visited: HashMap<*const RwLock<PrecomputeNode>, Arc<RwLock<PrecomputeNode>>> = HashMap::new();

        // We need to process all roots.
        let mut new_roots = BTreeMap::new();
        for (sid, root_arc) in self.roots.iter() {
            let canonical_root = self.deduplicate_recursive(root_arc.clone(), &mut canonical_nodes, &mut visited);
            new_roots.insert(*sid, canonical_root);
        }
        self.roots = new_roots;
        crate::debug!(2, "Finished merging subtrees. Canonical nodes: {}", canonical_nodes.len());
    }

    fn deduplicate_recursive(
        &self,
        node_arc: Arc<RwLock<PrecomputeNode>>,
        canonical_nodes: &mut HashMap<PrecomputeNode, Arc<RwLock<PrecomputeNode>>>,
        visited: &mut HashMap<*const RwLock<PrecomputeNode>, Arc<RwLock<PrecomputeNode>>>,
    ) -> Arc<RwLock<PrecomputeNode>> {
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
                        let canonical_child_arc = self.deduplicate_recursive(child_arc.clone(), canonical_nodes, visited);
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
        node_guard.recompute_max_depth();
    }

    let canonical_arc = {
            let node_guard = node_arc.read().unwrap();
            let node_content = (*node_guard).clone();
            canonical_nodes.entry(node_content).or_insert_with(|| node_arc.clone()).clone()
        };

        visited.insert(node_ptr, canonical_arc.clone());
        canonical_arc
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
}

fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
}

impl ParseState { // No longer generic
    pub fn merge(&mut self, mut other: ParseState) {
        // if self.stack.max_depth() > other.stack.max_depth() {
        //     std::mem::swap(self, &mut other);
        // }
        // Arc::make_mut(&mut self.stack).merge_with_depth(1, &other.stack);
        // Arc::make_mut(&mut self.stack).merge_with_depth(2, &other.stack);
        // Arc::make_mut(&mut self.stack).merge_with_depth(3, &other.stack);
        Arc::make_mut(&mut self.stack).merge_with_depth(usize::MAX, &other.stack);
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
 
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLRParserState<'a> { // No longer generic
    pub parser: &'a GLRParser,
    pub active_state: ParseState,
    accepted: bool,                // <-- NEW
    phase: ParserPhase,
    below_bottom_cache: std::collections::HashMap<BelowBottomCacheKey, ArcPtrWrapper<RwLock<PrecomputeNode2>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BelowBottomCacheKey {
    nonterminal_id: NonTerminalID,
    source_state_id: StateID,
    // k: usize,
    // Important: this Acc must have trie2_nodes cleared before being placed here.
    acc: Acc,
}

impl Display for GLRParserState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO: this is bad. make this better
        // Display the stack
        self.log_gss("    ", TerminalID(0), false, false);
        Ok(())
    }
}

// Key is (depth, state_id) to process stacks in a specific order.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct WorkMapKey(usize, StateID);

impl PartialOrd for WorkMapKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkMapKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This ordering is chosen for performance. It processes deeper stacks first.
        // The idea is that deeper stacks are more constrained and processing them
        // first might lead to quicker pruning of invalid paths.
        // Sorting by depth descending, then by state_id ascending.
        other.0.cmp(&self.0).then_with(|| self.1.cmp(&other.1))
    }
}

type WorkMap = BTreeMap<WorkMapKey, ParseState>;

impl<'a> GLRParserState<'a> { // No longer generic
    fn enqueue(work_map: &mut WorkMap, state: ParseState) {
        // Peel off the top edges of the GSS in the given state,
        // and group the resulting isolated paths by their (depth, state_id) key.
        // This merges paths that are in the same logical state, reducing redundant processing.
        for peek in GSSNode::peek_iter(&state.stack) {
            let isolated_state = ParseState { stack: peek.isolated_parent() };
            let depth = isolated_state.stack.max_depth();
            let state_id = peek.edge_value().state_id;
            work_map.entry(WorkMapKey(depth, state_id))
                .and_modify(|s| s.merge(isolated_state.clone()))
                .or_insert(isolated_state);
        }
    }

    fn push_state(
        &self,
        peek: &GSSPeek,
        new_content: ParseStateEdgeContent,
    ) -> ParseState {
        crate::debug!(4, "Pushing new state with content: {:?}", new_content);
        let new_gss_node_instance = peek.push_on_parent(new_content);
        ParseState { stack: Arc::new(new_gss_node_instance) }
    }

    /// Shared inner loop for phase 1 and phase 2.
    /// `action_selector` chooses between the phase-1 or phase-2 action map.
    fn process_action_queue<F>(
        &mut self,
        token_id: TerminalID,
        work_map: &mut WorkMap,
        mut reduce_map: Option<&mut WorkMap>,
        shifted_states_todo: &mut VecDeque<ParseState>,
        action_selector: F,
        use_full_action_map: bool,
        config: &ProcessTokenAdvancedConfig,
    ) where
        F: Fn(&Row) -> &BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>,
    {
        while let Some((WorkMapKey(_depth, state_id), state)) = work_map.pop_first() {
            let row = &self.parser.table[&state_id];
            if let Some(action) = action_selector(row).get(&token_id) {
                for peek in GSSNode::peek_iter(&state.stack) {
                    match action {
                        Stage7ShiftsAndReducesLookaheadValue::Shift(to) => {
                            crate::debug!(5, "Action: Shift to state {}", to.0);
                            let new_parse_state =
                                self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                            shifted_states_todo.push_back(new_parse_state);
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Reduce {
                            nonterminal_id: nt,
                            len,
                            ..
                        } => {
                            crate::debug!(5, "Action: Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), len);
                            let s_new_arc = self.reduce_and_goto(&peek, *nt, *len, token_id, &action_selector, config);
                            if !s_new_arc.is_empty() {
                                let new_parse_state = ParseState { stack: s_new_arc };
                                if let Some(ref mut r_map) = reduce_map {
                                    Self::enqueue(r_map, new_parse_state);
                                } else {
                                    Self::enqueue(work_map, new_parse_state);
                                }
                            }
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                            crate::debug!(5, "Action: Split with shift and reduces");
                            if let Some(to) = shift {
                                crate::debug!(5, "Action (Split): Shift to state {}", to.0);
                                let new_parse_state =
                                    self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                                shifted_states_todo.push_back(new_parse_state);
                            }
                            for (len, nts) in reduces {
                                for (nt, _prod_ids) in nts {
                                    crate::debug!(5, "Action (Split): Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), *len);
                                    let s_new_arc = self.reduce_and_goto(&peek, *nt, *len, token_id, &action_selector, config);
                                    if !s_new_arc.is_empty() {
                                        let new_parse_state = ParseState { stack: s_new_arc };
                                        if let Some(ref mut r_map) = reduce_map {
                                            Self::enqueue(r_map, new_parse_state);
                                        } else {
                                            Self::enqueue(work_map, new_parse_state);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                crate::debug!(5, "No action found for token '{}' in state {}", self.parser.terminal_map.get_by_right(&token_id).unwrap(), state_id.0);
            }
        }
        self.below_bottom_cache.clear();
    }

    fn _do_actions_without_default(&mut self, token_id: TerminalID, phase1_todo: &mut WorkMap, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig) {
        let token_display = self.parser.terminal_map.get_by_right(&token_id).unwrap();
        crate::debug!(4, "Phase 1: Processing token '{}'", token_display);
        timeit!("GLRParserState::step::phase1", {
            self.process_action_queue(
                token_id,
                phase1_todo,
                Some(phase2_todo),
                shifted_states_todo,
                |row| &row.shifts_and_reduces_without_default_reduce,
                false, // Not using full action map
                config,
            );
        });
    }

    fn _do_actions_with_default(&mut self, token_id: TerminalID, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig) {
        crate::debug!(4, "Phase 1 completed, proceeding to Phase 2 with {} shifted states", shifted_states_todo.len());
        timeit!("GLRParserState::step::phase2", {
            // Reduces are pushed back onto the same queue (`None`).
            self.process_action_queue(
                token_id,
                phase2_todo,
                None,
                shifted_states_todo,
                |row| &row.shifts_and_reduces_full,
                true, // Using full action map
                config,
            );
            self.phase = ParserPhase::ReadyForDefaultReductions;
        });
    }

    #[time_it("GLRParserState::reduce_and_goto")]
    fn reduce_and_goto<F>(
        &mut self,
        peek: &GSSPeek,
        nt: NonTerminalID,
        len: usize,
        token_id: TerminalID,
        action_selector: &F,
        config: &ProcessTokenAdvancedConfig,
    ) -> Arc<GSSNode>
    where
        F: Fn(&Row) -> &BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>,
    {
        let popper: GSSPopper = timeit!(peek.popn(len));
        crate::debug!(4, "Reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
        crate::debug!(4, "Popped with {} results...", popper.num_predecessors());
        let mut any_below_bottom = !popper.below_bottom.is_empty();
        // timeit!(format!("GLRParserState::reduce_and_goto reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len), {});
        // timeit!(format!("GLRParserState::reduce_and_goto reducing with len {}", len), {});

        let mut out = Vec::new();
        for popper_item in popper.iter() {
            for peek2 in popper_item.peek_iter() {
                let predecessor_state_id = peek2.edge_value().state_id;
                let mut current_nt = nt;

                // Fast loop for unit reduction chains based on the current lookahead token.
                let mut i = 0;
                loop {
                    i += 1;
                    let goto = self.parser.table.get(&predecessor_state_id).and_then(|row| row.gotos.get(&current_nt)).expect_else(|| {
                        format!("Goto not found for NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id)
                    });

                    if goto.accept {
                        crate::debug!(4, "Accepting with NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id);
                        self.accepted = true;
                    }

                    if let Some(goto_state_id) = goto.state_id {
                        let next_row = &self.parser.table[&goto_state_id];
                        // Check if the action in the new state for the current token is a len-1 reduce.
                        if let Some(Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: next_nt, len: 1, .. }) = action_selector(next_row).get(&token_id) {
                            // It is. Continue the chain by updating the non-terminal and looping.
                            current_nt = *next_nt;
                            continue; // Continue the fast loop.
                        } else {
                            // It's not a len-1 reduce. This is our final state for this chain.
                            let new_gss_node = peek2.push_on_parent(ParseStateEdgeContent { state_id: goto_state_id });
                            out.push(Arc::new(new_gss_node));
                            // timeit!(format!("Exiting fast loop. Reason: Found incompatible action: {:?}", action_selector(next_row).get(&token_id)), {});
                            break; // Exit the fast loop for this path
                        }
                    } else {
                        // No further state to go to. This path terminates here.
                        timeit!(format!("Exloring path. Reason: No goto state found for NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id), {});
                        break; // Exit the fast loop for this path
                    }
                }
                // Round to nearest power of 2
                let i_rounded_to_nearest_pow = if i == 0 {
                    1
                } else {
                    1 << (32 - (i as u32 - 1).leading_zeros())
                };
 
                timeit!(format!("GLRParserState::step::phase2::goto::number of loops (rounded to nearest pow of 2): {}", i_rounded_to_nearest_pow), {});
            }
        }
 
        // Handle “popped below bottom” cases:
        //
        // If the reduction pops below the bottom, we have recognized only the
        // suffix β of a rule A ::= α β. Per substring parsing semantics,
        // α lies before the substring start and must be considered unknown (but derivable),
        // so we continue in every state that has a GOTO on A. We also merge the Acc
        // accumulated along these paths to create a new virtual root to push onto.
        // timeit!(format!("GLRParserState::reduce_and_goto: Handling popped below bottom cases for NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len), {
            timeit!("GLRParserState::reduce_and_goto: Handling popped below bottom cases", {
            if any_below_bottom {
                match config.below_bottom_mode {
                    BelowBottomReductionMode::ContinueFromAll => {
                        crate::debug!(5, "Handling popped below bottom cases for NT '{}' and len {} with ContinueFromAll", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
                        timeit!(format!("GLRParserState::reduce_and_goto: Popped below bottom cases for NT '{}' and len {}, number of imagined reduces: {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len, self.parser.substring_gotos.get(&nt).unwrap().len()), {});
                        let mut below_zero = Vec::new();

                        if let Some(gotos_for_nt) = self.parser.substring_gotos.get(&nt) {
                            crate::debug!(6, "States to push after reduction (precomputed): {:?}", gotos_for_nt);
                            let mut trie2_dst_nodes = HashMap::new();
                            for (k, acc_arc) in popper.below_bottom {
                                let mut acc: Acc = acc_arc.as_ref().clone();
                                let active_llm_tokens = acc.union_llm_tokens();
                                let trie2_nodes = std::mem::take(&mut acc.trie2_nodes);
                                for goto_info in gotos_for_nt {
                                    // Key that ignores trie2_nodes (they are already cleared from 'acc' by std::mem::take above)
                                    let cache_key = BelowBottomCacheKey {
                                        nonterminal_id: nt,
                                        source_state_id: goto_info.source_state_id,
                                        // k,
                                        acc: acc.clone(),
                                    };

                                    // If we have seen this exact situation before, reuse the cached Trie-2 node
                                    if let Some(cached_trie2_node) = self.below_bottom_cache.get(&cache_key) {
                                        timeit!("GLRParserState::reduce_and_goto: Using cached Trie-2 node", {
                                        for existing_trie2_node in &trie2_nodes {
                                            timeit!("GLRParserState::reduce_and_goto: Inserting cached Trie-2 node (loop iteration)", {});
                                            // Use auto-insert to degrade to a WEAK edge if a strong cycle would be formed.
                                            let inserter = EdgeInserter::new(
                                                existing_trie2_node.as_arc().clone(),
                                                (k, Some(goto_info.source_state_id)),
                                                active_llm_tokens.clone(),
                                                |e, n| *e |= n,
                                            ).to_destination_weakly(cached_trie2_node.as_arc().clone());
                                            inserter.expect("GLRParserState::reduce_and_goto: cached insert failed");
                                        }
                                        });

                                        if goto_info.accept {
                                            self.accepted = true;
                                        }

                                        // IMPORTANT: No need to push a new GSS node here.
                                        // It would be equivalent to the one created when this key was first seen.
                                        continue;
                                    }
                                    if let Some(goto_state_id) = goto_info.goto_state_id {
                                        // Create and cache the new Trie-2 node under this key (before wiring or GSS building).
                                        let new_trie2_node = trie2_dst_nodes
                                            .entry(goto_info.source_state_id)
                                            .or_insert_with(|| Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::no_end()))))
                                            .clone();
                                        self.below_bottom_cache.insert(cache_key, ArcPtrWrapper::new(new_trie2_node.clone()));

                                        timeit!("GLRParserState::reduce_and_goto: Inserting new Trie-2 node", {
                                        for existing_trie2_node in &trie2_nodes {
                                            // Allow cycles to be represented as WEAK edges if they occur.
                                            timeit!("GLRParserState::reduce_and_goto: Inserting new Trie-2 node (loop iteration)", {});
                                            let inserter = EdgeInserter::new(
                                                existing_trie2_node.as_arc().clone(),
                                                (k, Some(goto_info.source_state_id)),
                                                active_llm_tokens.clone(),
                                                |e, n| *e |= n,
                                            ).try_destination_auto(new_trie2_node.clone());
                                            inserter.expect("GLRParserState::reduce_and_goto: EdgeInserter failed");
                                        }
                                        });

                                        let mut acc2 = acc.clone();
                                        acc2.trie2_nodes = vec![ArcPtrWrapper::new(new_trie2_node.clone())].into_iter().collect();
                                        let new_gss0 = GSSNode::new(acc2);
                                        let new_gss1 = new_gss0.push(ParseStateEdgeContent { state_id: goto_info.source_state_id });
                                        let new_gss2 = new_gss1.push(ParseStateEdgeContent { state_id: goto_state_id });
                                        below_zero.push(Arc::new(new_gss2));
                                    }

                                    if goto_info.accept {
                                        self.accepted = true;
                                    }
                                }
                            }
                        }
                        let merged = timeit!("GLRParserState::reduce_and_goto: Merging below-zero nodes", {
                            timeit!(format!("GLRParserState::reduce_and_goto: Merging {} below-zero nodes", below_zero.len()), {
                                GSSNode::merge_many_with_depth(usize::MAX, below_zero)
                            })
                        });
                        out.push(merged);
                    }
                    BelowBottomReductionMode::ContinueFromEverything => {
                        crate::debug!(5, "Handling popped below bottom cases for NT '{}' and len {} with ContinueFromEverything", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
                        let mut below_zero = Vec::new();

                        // Find the single goto from the 'everything' state for this non-terminal.
                        let everything_state_id = self.parser.everything_state_id;
                        if let Some(goto) = self.parser.table.get(&everything_state_id).and_then(|row| row.gotos.get(&nt)) {
                            let goto_info = SubstringGoto {
                                source_state_id: everything_state_id,
                                goto_state_id: goto.state_id,
                                accept: goto.accept,
                            };

                            // Now, the logic is very similar to the loop in ContinueFromAll, but just for this one goto_info.
                            let mut trie2_dst_nodes = HashMap::new();
                            for (k, acc_arc) in popper.below_bottom {
                                let mut acc: Acc = acc_arc.as_ref().clone();
                                let active_llm_tokens = acc.union_llm_tokens();
                                let trie2_nodes = std::mem::take(&mut acc.trie2_nodes);

                                // Key that ignores trie2_nodes
                                let cache_key = BelowBottomCacheKey {
                                    nonterminal_id: nt,
                                    source_state_id: goto_info.source_state_id,
                                    // k,
                                    acc: acc.clone(),
                                };

                                // Cache check
                                if let Some(cached_trie2_node) = self.below_bottom_cache.get(&cache_key) {
                                    for existing_trie2_node in &trie2_nodes {
                                        let inserter = EdgeInserter::new(
                                            existing_trie2_node.as_arc().clone(),
                                            (k, Some(goto_info.source_state_id)),
                                            active_llm_tokens.clone(),
                                            |e, n| *e |= n,
                                        ).to_destination_weakly(cached_trie2_node.as_arc().clone());
                                        inserter.expect("GLRParserState::reduce_and_goto: cached insert failed");
                                    }
                                    if goto_info.accept {
                                        self.accepted = true;
                                    }
                                    continue;
                                }

                                // If there's a state to go to, build the GSS path.
                                if let Some(goto_state_id) = goto_info.goto_state_id {
                                    let new_trie2_node = trie2_dst_nodes
                                        .entry(goto_info.source_state_id)
                                        .or_insert_with(|| Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::no_end()))))
                                        .clone();
                                    self.below_bottom_cache.insert(cache_key, ArcPtrWrapper::new(new_trie2_node.clone()));

                                    for existing_trie2_node in &trie2_nodes {
                                        let inserter = EdgeInserter::new(
                                            existing_trie2_node.as_arc().clone(),
                                            (k, Some(goto_info.source_state_id)),
                                            active_llm_tokens.clone(),
                                            |e, n| *e |= n,
                                        ).try_destination_auto(new_trie2_node.clone());
                                        inserter.expect("GLRParserState::reduce_and_goto: EdgeInserter failed");
                                    }

                                    let mut acc2 = acc.clone();
                                    acc2.trie2_nodes = vec![ArcPtrWrapper::new(new_trie2_node.clone())].into_iter().collect();
                                    let new_gss0 = GSSNode::new(acc2);
                                    let new_gss1 = new_gss0.push(ParseStateEdgeContent { state_id: goto_info.source_state_id });
                                    let new_gss2 = new_gss1.push(ParseStateEdgeContent { state_id: goto_state_id });
                                    below_zero.push(Arc::new(new_gss2));
                                }

                                if goto_info.accept {
                                    self.accepted = true;
                                }
                            }
                        }
                        let merged = GSSNode::merge_many_with_depth(usize::MAX, below_zero);
                        out.push(merged);
                    }
                    BelowBottomReductionMode::Fail => {
                        crate::debug!(5, "Popped below bottom, failing these parse paths.");
                        // Do nothing, paths are dropped.
                    }
                    BelowBottomReductionMode::Panic => {
                        panic!("A reduction popped below the bottom of the stack, and BelowBottomReductionMode was set to Panic.");
                    }
                }
            }
            });
 
        timeit!("GLRParserState::reduce_and_goto", {
        timeit!(format!("GLRParserState::reduce_and_goto: Merging {} nodes", out.len()), {
            GSSNode::merge_many_with_depth(usize::MAX, out)
        })
        })
    }

    pub fn process_token(&mut self, token_id: TerminalID) {
        self.process_token_advanced(token_id, &ProcessTokenAdvancedConfig::default())
    }

    #[time_it("GLRParserState::process_token_advanced")]
    pub fn process_token_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
        // Reset acceptance flag for the new token
        self.accepted = false;

        if Some(token_id) == self.parser.ignore_terminal_id {
            crate::debug!(4, "Ignoring token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());
            self.phase = ParserPhase::ReadyForDefaultReductions; // Skip phase 1 and 2, go straight to phase 3
            return;
        }

        self.log_gss("Phase1/2-start", token_id, false, false);

        let mut phase2_todo: WorkMap = WorkMap::new();
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();

        if self.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: WorkMap = WorkMap::new();
            Self::enqueue(&mut phase1_todo, self.active_state.clone());
            self._do_actions_without_default(token_id, &mut phase1_todo, &mut phase2_todo, &mut shifted_states_todo, config);
        } else { // ParserPhase::ReadyForDefaultReductions
            Self::enqueue(&mut phase2_todo, self.active_state.clone());
        }

        // --- Phase 2 ---
        self._do_actions_with_default(token_id, &mut phase2_todo, &mut shifted_states_todo, config);

        // Consolidate all shifted states into the new active_state for phase 3
        crate::debug!(4, "Phase 2 completed, consolidating {} shifted states into active state", shifted_states_todo.len());
        let mut next_active = ParseState::new();
        for state in shifted_states_todo {
            next_active.merge(state);
        }
        self.active_state = next_active;
        self.log_gss("Phase1/2-end", token_id, false, false);
    }

    #[time_it("GLRParserState::process_default_reductions")]
    pub fn process_default_reductions(&mut self) {
        return;
        self.log_gss("Phase3-start", TerminalID(0), false, false); // Log with dummy token ID
        if self.phase == ParserPhase::ReadyForToken {
            crate::debug!(4, "Phase 3 skipped, parser is ready for Phase 1");
            return;
        }
        assert_eq!(self.phase, ParserPhase::ReadyForDefaultReductions);

        let enqueue_local = |work_map: &mut WorkMap, isolated_state: &ParseState, peek: &GSSPeek| {
            let depth = isolated_state.stack.max_depth();
            let state_id = peek.edge_value().state_id;
            work_map.entry(WorkMapKey(depth, state_id))
                .and_modify(|s| s.merge(isolated_state.clone()))
                .or_insert(isolated_state.clone());
        };
        let mut work_map: WorkMap = BTreeMap::new();

        // Peel off the top edges to populate the initial work map.
        for peek in GSSNode::peek_iter(&self.active_state.stack) {
            let isolated_state = ParseState { stack: peek.isolated_parent() };
            enqueue_local(&mut work_map, &isolated_state, &peek);
        }

        let mut next_active_state = ParseState::new();

        let stats = gather_gss_stats(&[self.active_state.stack.as_ref()]);
        crate::debug!(5, "GLRParserState::process_default_reductions: Stats: {:?}", stats);

        crate::debug!(4, "Phase 3: Processing {} states", work_map.len());
        timeit!(format!("GLRParserState::step::phase3 - unique_nodes: {}", stats.unique_nodes), {
        // timeit!("GLRParserState::step::phase3", {
            while let Some((WorkMapKey(_depth, state_id), state)) = work_map.pop_first() {
                // let stats = gather_gss_stats(&[&state.stack]);
                // if stats.unique_nodes > stats.structurally_unique_nodes { crate::debug!(3, "Expected unique_nodes <= structurally_unique_nodes. Got unique_nodes: {}, structurally_unique_nodes: {}", stats.unique_nodes, stats.structurally_unique_nodes); }

                let row = &self.parser.table[&state_id];

                if let Some(ref r) = row.default_reduce.reduce {
                    crate::debug!(5, "Action (Phase 3): Default Reduce by NT '{}' (len {}) in state {}, num_predecessors: {}",
                                  self.parser.non_terminal_map.get_by_right(&r.nonterminal_id).unwrap(),
                                  r.len, state_id.0, state.stack.num_predecessors());
                    timeit!(format!("GLRParserState::step::phase3::reduce NT '{}' (len {}) in state {}",
                                    self.parser.non_terminal_map.get_by_right(&r.nonterminal_id).unwrap(),
                                    r.len, state_id.0), {
                    // timeit!(format!("GLRParserState::step::phase3::reduce NT (len {})", r.len), {
                        // For each peek in the current state, reduce and goto.
                        // This is the core of phase 3: reducing all stacks with the same state_id.
                        // We will merge the results into a new stack part.
                    let mut reduced_stack = GSSNode::new_fresh();
                    for peek in GSSNode::peek_iter(&state.stack) {
                        // println!("GLRParserState::do_phase3: Reducing with state_id: {}, len: {}, nonterminal: {}, production_ids: {:?}",
                        //          state_id.0, r.len, self.parser.non_terminal_map.get_by_right(&r.nonterminal_id).unwrap(), r.production_ids);


                        let len = r.len;
                        let nt = r.nonterminal_id;
                        let popper = timeit!(peek.popn(len));
                        crate::debug!(4, "Reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
                        crate::debug!(4, "Popped with {} results...", popper.num_predecessors());

                        let mut out_nodes_for_this_peek = Vec::new();

                        // --- Handle paths that remained on the stack ---
                        for popper_item in popper.iter() {
                            for peek2 in popper_item.peek_iter() {
                                let predecessor_state_id = peek2.edge_value().state_id;
                                let goto_state_ids = default_reduce_chain(self.parser, predecessor_state_id, nt);
                                let new_gss_node = peek2.isolated_parent().push_many(goto_state_ids.into_iter().map(|sid| ParseStateEdgeContent { state_id: sid }).collect());
                                out_nodes_for_this_peek.push(new_gss_node);
                            }
                        }

                        // --- Handle paths that popped below the bottom (substring parsing) ---
                        if !popper.below_bottom.is_empty() {
                            let mut merged_acc_opt: Option<Acc> = None;
                            for acc_arc in popper.below_bottom.values() {
                                merged_acc_opt = Some(match merged_acc_opt.take() {
                                    None => (**acc_arc).clone(),
                                    Some(prev) => Acc::merge(&prev, acc_arc),
                                });
                            }

                            if let Some(merged_acc) = merged_acc_opt {
                                let mut states_to_push = BTreeSet::new();
                                // For substring parsing, when popping below bottom, we can transition
                                // from *any* state in the automaton that has a GOTO on `nt`.
                                for (source_state_id, _source_row) in &self.parser.table {
                                    let goto_ids = default_reduce_chain(self.parser, *source_state_id, nt);
                                    if !goto_ids.is_empty() {
                                        states_to_push.insert(*source_state_id);
                                        states_to_push.extend(goto_ids);
                                    }
                                }
                                if !states_to_push.is_empty() {
                                    let base = GSSNode::new(merged_acc);
                                    let new_gss_node = base.push_many(states_to_push.into_iter().map(|sid| ParseStateEdgeContent { state_id: sid }).collect());
                                    out_nodes_for_this_peek.push(new_gss_node);
                                }
                            }
                        }

                        // --- Merge results for this peek ---
                        if !out_nodes_for_this_peek.is_empty() {
                            let mut iter = out_nodes_for_this_peek.into_iter();
                            let mut merged = iter.next().unwrap();
                            for next in iter {
                                merged.merge_with_depth(usize::MAX, &next);
                            }
                            reduced_stack.merge_with_depth(usize::MAX, &merged);
                        }
                    }

                    if !reduced_stack.is_empty() {
                        // Deconstruct the result and put it back into the work map.
                        for new_peek in GSSNode::peek_iter(&Arc::new(reduced_stack)) {
                            let isolated = ParseState { stack: new_peek.isolated_parent() };
                            enqueue_local(&mut work_map, &isolated, &new_peek);
                        }
                    }
                    });
                }
 
                if row.default_reduce.clone_and_merge {
                    // println!("next_active_state.stack: {}", print_gss_forest(&[next_active_state.stack.clone()], &Default::default(), &GSSPrintConfig { verbose: true, ..Default::default() }).0);
                    // println!("state.stack: {}", print_gss_forest(&[state.stack.clone()], &Default::default(), &GSSPrintConfig { verbose: true, ..Default::default() }).0);
                    next_active_state.merge(state);
                    // println!("next_active_state.stack after merge: {}", print_gss_forest(&[next_active_state.stack.clone()], &Default::default(), &GSSPrintConfig { verbose: true, ..Default::default() }).0);
                    // let stats = gather_gss_stats(&[&next_active_state.stack]);
                    // if stats.unique_nodes > stats.structurally_unique_nodes { crate::debug!(3, "Expected unique_nodes <= structurally_unique_nodes. Got unique_nodes: {}, structurally_unique_nodes: {}", stats.unique_nodes, stats.structurally_unique_nodes); }
                }
            }
        });

        crate::debug!(4, "Phase 3 completed, merging {} states into next active state", next_active_state.stack.num_predecessors());
        self.active_state = next_active_state;
        self.phase = ParserPhase::ReadyForToken;
        self.log_gss("Phase3-end", TerminalID(0), false, false); // Log with dummy token ID
    }

    pub fn has_action_for(&self, token_id: TerminalID) -> Option<LLMTokenBV> {
        match LR_MODE {
            LRMode::LR1 | LRMode::LALR_EX_SHIFT_STATES => {
                if Some(token_id) == self.parser.ignore_terminal_id {
                    timeit!("GLRParserState::has_action_for::ignore_token", {
                        crate::debug!(4, "Ignoring token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());
                        // return Some(self.active_state.stack.allowed_llm_tokens());
                        return Some(LLMTokenBV::max_ones());
                    });
                }
                // let mut hasher = DeterministicHasher::new(DefaultHasher::new());
                // self.active_state.hash(&mut hasher);
                // let self_hash = hasher.finish();
                // println!("GLRParserState::has_action_for: {:?}", self_hash);
                self.log_gss("has_action_for-start", token_id, false, false);
                let mut llm_tokens = LLMTokenBV::zeros();
                for peek in GSSNode::peek_iter(&self.active_state.stack) {
                    let row = &self.parser.table[&peek.edge_value().state_id];
                    let shifts_and_reduces = match self.phase {
                        ParserPhase::ReadyForToken => &row.shifts_and_reduces_without_default_reduce,
                        ParserPhase::ReadyForDefaultReductions => &row.shifts_and_reduces_full,
                    };
                    if let Some(action) = shifts_and_reduces.get(&token_id) {
                        crate::debug!(4, "Found action for token '{}' in state {}: {:?}. LLM tokens: {:?}",
                                      self.parser.terminal_map.get_by_right(&token_id).unwrap(),
                                      peek.edge_value().state_id.0, action, peek.resolved_llm_tokens_union());
                        // That's it! Since this is a LR(1) parser, it's enough to know that there's *any* action.
                        timeit!("GLRParserState::has_action_for::action_found::add_llm_tokens", {
                            let peek_llm_tokens = timeit!(peek.resolved_llm_tokens_union());
                            timeit!(llm_tokens |= peek_llm_tokens);
                        });
                    } else {
                        timeit!("GLRParserState::has_action_for::no_action_found", {
                            crate::debug!(4, "No action for token '{}' in state {}", self.parser.terminal_map.get_by_right(&token_id).unwrap(), peek.edge_value().state_id.0);
                        });
                    }
                }
                Some(llm_tokens)
            }
            LRMode::LALR => None,
        }
    }

    pub fn step(&mut self, token_id: TerminalID) {
        self.process_token(token_id);
    }

    pub fn parse(&mut self, input: &[TerminalID]) {
        self.parse_part(input);
    }

    pub fn parse_part(&mut self, input: &[TerminalID]) {
        for &token_id in input {
            self.step(token_id);
        }
    }

    pub fn step_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
        self.process_token_advanced(token_id, config);
    }

    pub fn parse_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        self.parse_part_advanced(input, config);
    }

    pub fn parse_part_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        for &token_id in input {
            self.step_advanced(token_id, config);
        }
    }

    pub fn and_step(mut self, token_id: TerminalID) -> Self {
        self.step(token_id);
        self
    }

    pub fn and_parse(mut self, input: &[TerminalID]) -> Self {
        self.parse(input);
        self
    }

    pub fn merge_active_states(&mut self) {
        // No longer strictly necessary due to BTreeMap merge-on-insert, but GSS merge is explicit.
        // This method could be used if multiple GLRParserStates are combined.
    }

    pub fn merge_with(&mut self, mut other: GLRParserState) { // No longer generic
        assert!(std::ptr::eq(self.parser, other.parser));
        match (self.phase, other.phase) {
            (ParserPhase::ReadyForToken, ParserPhase::ReadyForDefaultReductions) => self.process_default_reductions(),
            (ParserPhase::ReadyForDefaultReductions, ParserPhase::ReadyForToken) => other.process_default_reductions(),
            _ => {},
        }
        self.active_state.merge(other.active_state);
        self.accepted |= other.accepted;
    }

    pub fn is_ok(&self) -> bool {
        self.accepted || (!self.active_state.stack.is_empty() && self.active_state.stack.is_alive())
    }

    /// Returns true if the previous step lead to an `accept` action.
    pub fn has_accepted(&self) -> bool {
        self.accepted
    }

    // #[time_it("GLRParserState::log_gss")]
    pub fn log_gss(&self, phase: &str, token: TerminalID, explain_states: bool, generate_dot: bool) {
        if !GSS_LOGGING_ENABLED {
            return;
        }
        // crate::debug!(3, "{} - token {} ({:?}) - nodes", phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0));
        const MAX: usize = 30;
        const PANIC_THRESHOLD: usize = 1_000_000;

        let roots: Vec<_> = vec![self.active_state.stack.clone()];
        let stats = gather_gss_stats(
            &roots.iter().map(|r| r.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "{} ({:?}) - accepted: {} - token '{}' ({}) - nodes: {:?}",
                      phase, self.phase, self.accepted, self.parser.terminal_map.get_by_right(&token).unwrap(), token.0, stats);

        let (gss_string, state_ids) = {
            let print_full_forest = stats.unique_nodes <= MAX;
            let max_nodes_to_print = if print_full_forest { usize::MAX } else { MAX };
            let config = GSSPrintConfig {
                labels: None,
                max_nodes: max_nodes_to_print,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            let (gss_string, state_ids) = print_gss_forest(&roots, &self.parent.parser.terminal_map, &config);
            let final_string = if print_full_forest {
                format!("GSS ({} nodes):\n{}", stats.unique_nodes, gss_string)
            } else {
                match find_longest_path(&self.active_state.stack) {
                    Some(p) => format!("GSS too big ({} nodes). Longest path ({}): {}",
                                       stats.unique_nodes,
                                       p.len(),
                                       p.iter().map(|(ec, _n)| ec.state_id.0) // n is Arc<GSSNode>
                                            .map(|id| id.to_string())
                                            .collect::<Vec<_>>()
                                        .join(" → ")),
                    None => format!("GSS too big ({} nodes) – path not found", stats.unique_nodes),
                }
            };
            (final_string, state_ids)
        };

        let mut final_string = gss_string;
        if explain_states && !state_ids.is_empty() {
            final_string.push_str("\n\n--- GSS State Explanations ---\n");
                for state_id in state_ids {
                    let mut explanation = String::new();
                    writeln!(&mut explanation, "\n--- State {} ---", state_id.0).unwrap();
                    self.parser.format_state_details(&mut explanation, state_id, "  ").unwrap();
                    final_string.push_str(&explanation);
                }
        }

        if stats.unique_nodes > PANIC_THRESHOLD {
            panic!("GSS too big ({} nodes). {}", stats.unique_nodes, final_string);
        }

        debug!(3, "{}", final_string);

        if generate_dot {
            let dot_string = self.gss_to_dot();
            // Log the DOT string. It can be copied into a .dot file and rendered with Graphviz.
            // e.g., `dot -Tpng -o gss.png gss.dot`
            crate::debug!(1, "GSS DOT graph:\n{}", dot_string);
        }
    }

    /// Generates a Graphviz DOT representation of the GSS state graph.
    pub fn gss_to_dot(&self) -> String {
        self.parser.gss_to_dot(&self.active_state.stack, None, None)
    }
}

impl GLRParser {
    /// Generates a Graphviz DOT representation of the state transitions present in a GSS forest.
    /// This visualizes the portion of the state machine explored by the parser.
    pub fn gss_forest_to_dot(
        &self,
        roots: &[(&str, &GSSNode)],
        original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
        llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    ) -> String {
        let mut dot = String::new();
        writeln!(&mut dot, "digraph GSS_Forest {{").unwrap();
        writeln!(&mut dot, "  rankdir=LR;").unwrap();
        writeln!(&mut dot, "  node [shape=box, fontname=\"Courier New\", style=rounded];").unwrap();
        writeln!(&mut dot, "  edge [arrowhead=vee];").unwrap();

        let mut visited_nodes = HashSet::new();
        let mut node_ids = HashMap::new();
        let mut edge_node_ids = HashMap::new();
        let mut next_id_counter = 0;

        let mut queue: VecDeque<Arc<GSSNode>> = roots.iter().map(|(_, n)| Arc::new((*n).clone())).collect();

        // Define root labels and connect them
        for (i, (label, root)) in roots.iter().enumerate() {
            let root_ptr = *root as *const GSSNode;
            let root_id = *node_ids.entry(root_ptr).or_insert_with(|| {
                let id = next_id_counter;
                next_id_counter += 1;
                id
            });

            writeln!(&mut dot, "  subgraph cluster_{} {{", i).unwrap();
            writeln!(&mut dot, "    label=\"{}\";", label).unwrap();
            writeln!(&mut dot, "    style=filled;").unwrap();
            writeln!(&mut dot, "    color=lightgrey;").unwrap();
            writeln!(&mut dot, "    node [style=filled,color=white];").unwrap();
            let root_node_name = format!("Root_{}", i);
            writeln!(&mut dot, "    {} [label=\"{}\", shape=ellipse];", root_node_name, label).unwrap();
            writeln!(&mut dot, "  }}").unwrap();
            writeln!(&mut dot, "  {} -> N{};", root_node_name, root_id).unwrap();
        }

        // Traverse and define all nodes and edges
        while let Some(node_arc) = queue.pop_front() {
            let node_ptr = Arc::as_ptr(&node_arc);
            if visited_nodes.contains(&node_ptr) {
                continue;
            }
            
            let parent_id = *node_ids.entry(node_ptr).or_insert_with(|| {
                let id = next_id_counter;
                next_id_counter += 1;
                id
            });

            // Define the GSS node if it hasn't been visited yet
            if visited_nodes.insert(node_ptr) {
                let acc_str = crate::datastructures::gss::format_acc(
                    &node_arc,
                    &self.terminal_map,
                    original_internal_bimap,
                    llm_token_map,
                );
                let escaped_acc = acc_str
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\l")
                    .replace('{', "\\{")
                    .replace('}', "\\}")
                    .replace('<', "\\<")
                    .replace('>', "\\>")
                    .replace('\'', "\\'"); // Escape single quotes for DOT

                writeln!(&mut dot, "  N{} [label=\"Node {}\\lDepth: {}\\l{}\"];", parent_id, parent_id, node_arc.max_depth(), escaped_acc).unwrap();
            }

            for (edge_val, preds_by_depth) in &node_arc.predecessors {
                let state_id = edge_val.state_id;
                let edge_key = (node_ptr, edge_val.clone());
                
                let edge_node_id = *edge_node_ids.entry(edge_key).or_insert_with(|| {
                    let id = next_id_counter;
                    next_id_counter += 1;
                    
                    // Define the edge node
                    let mut explanation = String::new();
                    self.format_state_details(&mut explanation, state_id, "").unwrap();
                    let escaped_explanation = explanation
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\l")
                        .replace('{', "\\{")
                        .replace('}', "\\}")
                        .replace('<', "\\<")
                        .replace('>', "\\>")
                        .replace('\'', "\\'"); // Escape single quotes for DOT
                    
                    writeln!(&mut dot, "  E{} [label=\"State {}\\l{}\", shape=plaintext, fontname=\"Courier New\"];", id, state_id.0, escaped_explanation).unwrap();
                    id
                });

                // Connect parent to edge node
                writeln!(&mut dot, "  N{} -> E{};", parent_id, edge_node_id).unwrap();

                for pred_vec in preds_by_depth.values() {
                    for pred_arc in pred_vec {
                        let pred_ptr = Arc::as_ptr(pred_arc);
                        let pred_id = *node_ids.entry(pred_ptr).or_insert_with(|| {
                            let id = next_id_counter;
                            next_id_counter += 1;
                            id
                        });
                        
                        // Connect edge node to predecessor
                        writeln!(&mut dot, "  E{} -> N{} [arrowhead=none];", edge_node_id, pred_id).unwrap();
                        queue.push_back(pred_arc.clone());
                    }
                }
            }
        }

        writeln!(&mut dot, "}}").unwrap();
        dot
    }

    /// Generates a Graphviz DOT representation of the state transitions present in a GSS.
    /// This visualizes the portion of the state machine explored by the parser.
    pub fn gss_to_dot(&self, root: &GSSNode, original_internal_bimap: Option<&BiBTreeMap<usize, usize>>, llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>) -> String {
        self.gss_forest_to_dot(&[("Root", root)], original_internal_bimap, llm_token_map)
    }
}

// Helper for default reductions' fast unit-reduction chain
fn default_reduce_chain(
    parser: &GLRParser,
    start_state_id: StateID, // The state *before* the GOTO
    initial_nt: NonTerminalID,
) -> BTreeSet<StateID> {
    let mut final_goto_state_ids = BTreeSet::new();
    let mut current_nt = initial_nt;
    // The state for GOTO lookups is always the one before the reduction sequence.
    let goto_source_state_id = start_state_id;

    loop {
        if let Some(goto) = parser.table.get(&goto_source_state_id).and_then(|row| row.gotos.get(&current_nt)) {
            if let Some(goto_state_id) = goto.state_id {
                let next_row = &parser.table[&goto_state_id];
                if let Some(next_reduce) = &next_row.default_reduce.reduce {
                    if next_reduce.len == 1 {
                        // This is a unit reduction. Continue the chain with the new non-terminal.
                        current_nt = next_reduce.nonterminal_id;
                        continue; // Continue the loop
                    }
                }
                // Not a unit reduction, or no default reduce. This is the end of the chain.
                final_goto_state_ids.insert(goto_state_id);
                break;
            } else {
                // No goto state. End of chain.
                break;
            }
        } else {
            // No goto for current_nt from goto_source_state_id. End of chain.
            break;
        }
    }
    final_goto_state_ids
}

