// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::sync::RwLock;
use std::mem;
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::gss::{disallow_llm_tokens_and_prune_arc, fuse_predecessors_recursive, get_roots, print_gss_forest, reset_terminals};
use crate::datastructures::gss::{map_allowed_terminals_tokenizer_states, prune_disallowed_terminals};
use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::{BitOr, BitOrAssign};
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc};
use std::cell::RefCell;

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, dump_precompute_trie_recursive, print_precompute_stats, PrecomputeStats};
use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
use crate::datastructures::gss::{gather_gss_stats, GSSNode, allow_only_llm_tokens_and_prune_arc, reset_llm_tokens, disallow_terminals_and_prune_arc, GSSPrintConfig, LLMTokenBV, TerminalBV, PrecomputedNodeContents, PrecomputeNode2};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::arc_wrapper::{ArcPtrWrapper};
use crate::finite_automata::Regex;
use crate::glr::parser::{BelowBottomReductionMode, GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, ProcessTokenAdvancedConfig, ProcessDefaultReductionsAdvancedConfig};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use kdam::{tqdm, BarBuilder};
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};
use crate::datastructures::arc_wrapper::{NodePtr, WeakPtrWrapper};
use crate::datastructures::gss::Acc;
use crate::glr::table::StateID;
use crate::glr::analyze::compute_terminal_follow_sets;
use crate::glr::grammar::Terminal;
use crate::glr::items::{Item, LRMode, LR_MODE};
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
            &terminal_follow_map,
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
            &terminal_follow_map,
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

        let mut gc = Self {
            tokenizer,
            parser,
            precomputed,
            precomputed2,
            llm_vocab,
            token_name_map,
            possible_matches: computed_possible_matches,
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
            terminal_follow_map,
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
        const BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING: bool = false;
        const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
            BelowBottomReductionMode::ContinueFromEverything
        } else {
            BelowBottomReductionMode::ContinueFromAll
        };

        let mut precomputed2 = BTreeMap::new();
        // let mut memo: HashMap<ArcPtrWrapper<RwLock<PrecomputeNode>>, Arc<RwLock<_>>> = HashMap::new(); // Old memo, removed

        let mut initial_values_for_map: Vec<(Arc<RwLock<PrecomputeNode>>, GLRParserState)> =
            Vec::new();
        let parser = parser.unwrap();
 
        // 1) Build a single base Trie2 root.
        let base_trie2_root = Arc::new(RwLock::new(PrecomputeNode2::new(
            PrecomputedNodeContents::root(internal_max_llm_token),
        )));
        let base_trie2_root_wr = ArcPtrWrapper::new(base_trie2_root.clone());

        let mut base_gss_nodes: Vec<Arc<GSSNode>> = Vec::new();

        if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
            let mut acc = Acc::new_fresh();
            acc.trie2_nodes.insert(base_trie2_root_wr.clone());
            let gss_leaf = Arc::new(GSSNode::new(acc));
            base_gss_nodes.push(Arc::new(
                gss_leaf.push(ParseStateEdgeContent { state_id: parser.everything_state_id })
            ));
        } else {
            for state_id in parser.table.keys() {
                let mut acc = Acc::new_fresh();
                acc.trie2_nodes.insert(base_trie2_root_wr.clone());
                let gss_leaf = Arc::new(GSSNode::new(acc));
                base_gss_nodes.push(Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: *state_id })));
            }
        }

        // Merge the base per-state initial nodes into one GSS and build a GLR state from it.
        let base_gss_merged = GSSNode::merge_many_with_depth(usize::MAX, base_gss_nodes);
        let mut base_glr_state = parser.init_glr_parser_from_parse_state(ParseState::with_stack(base_gss_merged));

        // Optional: pre-warm once with default reductions (your idea)
        // base_glr_state.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig {
        //     fuel: None,
        //     below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE,
        // });

        #[cfg(not(rustrover))]
        let it = tqdm!(precomputed.iter(), desc = "Precomputing Trie 2", disable = !PROGRESS_BAR_ENABLED, leave=false);
        #[cfg(rustrover)]
        let it = precomputed.iter();
        for (tokenizer_state_id, trie1_root) in it {
            // Deep clone Trie2
            let (cloned_trie2_root, trie2_map) = clone_trie2_graph(&base_trie2_root);

            // Deep clone the base GSS, remapping trie2_nodes
            let cloned_gss = crate::datastructures::gss::deep_clone_gss_with_trie2_map(
                &base_glr_state.active_state.stack,
                &trie2_map,
            );
            let glr_state_for_sid = parser.init_glr_parser_from_parse_state(ParseState::with_stack(cloned_gss));

            // Record per tokenizer state
            precomputed2.insert(*tokenizer_state_id, cloned_trie2_root);
            initial_values_for_map.push((trie1_root.clone(), glr_state_for_sid));
        }

        let trie2_end = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::leaf(), )));

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
                crate::debug!(3, "Trie2: At precomputed node {:p}, processing GLR state", precomputed_node_data);
                // Dump precomputed2
                // pub fn _dump_precomputed2(precomputed2: &BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>, original_to_internal_id_bimap: &BiBTreeMap<usize, usize>, llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>) {
                // GrammarConstraint::_dump_precomputed2(&precomputed2, &llm_vocab.as_ref().unwrap().original_to_internal_id_bimap, &llm_vocab.as_ref().unwrap().llm_token_map);

                crate::datastructures::gss::merge_trie2_nodes_if_needed(
                    &mut glr_s.active_state.stack,
                    1,
                    &mut HashMap::new(),
                );
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
                    crate::debug!(3, "Trie2: Found end state for GLR state");
                    glr_s.log_gss(
                        "Trie2: Found end state for GLR state",
                        TerminalID(0),
                        false,
                        false,
                    );
                    let mut end_dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> = BTreeMap::new();
                    let end_wr = ArcPtrWrapper::new(trie2_end.clone());

                    let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> = BTreeMap::new();

                    // for gss_root in get_roots([glr_s.active_state.stack.as_ref(), glr_s.active_state.accepted_state.as_ref()]) {
                    for (last_edge, gss_root_accs) in get_roots([glr_s.active_state.stack.as_ref()]) {
                        for gss_root_acc in gss_root_accs {
                            let active_llm_tokens_for_root = gss_root_acc.union_llm_tokens();
                            crate::debug!(4, "Trie2: For GSS root with edge {:?}, active LLM tokens: {:?}", last_edge, active_llm_tokens_for_root);

                            for src_wr in gss_root_acc.trie2_nodes.iter() {
                                let src_arc = src_wr.as_arc().clone();
                                let src_live = { src_arc.read().expect("poison").value.live_tokens.clone() };
                                let tokens_to_push = &active_llm_tokens_for_root & &src_live;
                                if tokens_to_push.is_empty() {
                                    crate::debug!(4, "Trie2: No tokens to push from this source node");
                                    continue;
                                }
                                {
                                    // Mark the source node as live for these tokens so the backward pass can see them.
                                    let mut src_w = src_arc.write().expect("poison");
                                    src_w.value.live_tokens |= tokens_to_push.clone();
                                }
                                crate::debug!(4, "Trie2: Pushing tokens {:?} from source node {:p}", tokens_to_push, src_arc);
 
                                let edge_key = (0, Some(last_edge.state_id));

                                let mut inserter = EdgeInserter::new(
                                    src_arc.clone(),
                                    edge_key,
                                    tokens_to_push.clone(),
                                    |e, n| *e |= n,
                                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                                    |ev, t| *ev &= &t.live_tokens,
                                );

                                inserter = inserter.try_destination(trie2_end.clone());

                                let final_dest_arc = inserter.clone_into_option().expect("Failed to insert end edge into Trie2 node");
                                let final_dest_wr = ArcPtrWrapper::new(final_dest_arc.clone());
                                dest_agg.entry(final_dest_wr.clone()).and_modify(|bv| *bv |= &tokens_to_push).or_insert(tokens_to_push.clone());
                            }
                        }
                    }
                    for (dst_wr, added) in &dest_agg {
                        let mut g = dst_wr.as_arc().write().expect("poison");
                        g.value.live_tokens |= added.clone();
                    }
                }

                if false {
                    let mut allowed_terminals = TerminalBV::zeros();
                    for gtid_opt in precomputed_node_data.children().keys() {
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
                    // glr_s.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig { fuel: None, per_state_fuel: Some(2), below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE });
                    glr_s.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig { fuel: None, per_state_fuel: None, below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE });
                    reset_terminals(&mut glr_s.active_state.stack, &mut HashMap::new());
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
        merge_consecutive_edges_trie2(&mut precomputed2);
        merge_nodes_trie2(&mut precomputed2);
        let promotions2 = Trie::promote_weak_edges_to_strong(&roots2);
        crate::debug!(2, "Promoted {} weak edges to strong in precomputed trie 2.", promotions2);

        // Recompute depths again after promotions, as they can change the graph structure.
        let roots2_final: Vec<_> = precomputed2.values().cloned().collect();
        Trie::recompute_all_max_depths(&roots2_final);

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
        //     let original_id_val = self.llm_vocab.original_to_internal_id_bimap.get_by_right(&(internal_id_val as usize)).expect(format!("Internal ID {} not found in original_to_internal_id_bimap while converting to original BV: {:?}", internal_id_val, internal_bv).as_str());
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

fn prune_dead_paths_trie2(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 2.");

    // Use a worklist algorithm to propagate "liveness" backwards from end nodes.
    // This correctly handles cycles, iterating until a fixed point is reached.
    let all_nodes = Trie::all_nodes(&roots.values().cloned().collect::<Vec<_>>());
    let mut predecessors: HashMap<*const RwLock<PrecomputeNode2>, Vec<(*const RwLock<PrecomputeNode2>, LLMTokenBV)>> = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<*const RwLock<PrecomputeNode2>, LLMTokenBV> = HashMap::new();

    // 1. Initialize live sets and build predecessor map.
    for node_arc in &all_nodes {
        let node_ptr = Arc::as_ptr(node_arc);
        live.insert(node_ptr, LLMTokenBV::zeros());

        let guard = node_arc.read().unwrap();
        if guard.value.end {
            let initial_live = guard.value.live_tokens.clone();
            if !initial_live.is_empty() {
                live.insert(node_ptr, initial_live);
                worklist.push_back(node_ptr);
            }
        }

        for dest_map in guard.children().values() {
            for (child_wrap, edge_bv) in dest_map {
                if let Some(child_arc) = child_wrap.upgrade() {
                    let child_ptr = Arc::as_ptr(&child_arc);
                    predecessors.entry(child_ptr).or_default().push((node_ptr, edge_bv.clone()));
                }
            }
        }
    }

    // 2. Propagate liveness until a fixed point is reached.
    while let Some(node_ptr) = worklist.pop_front() {
        let live_at_node = live.get(&node_ptr).unwrap().clone();
        if let Some(preds) = predecessors.get(&node_ptr) {
            for (pred_ptr, edge_bv) in preds {
                let live_from_edge = &live_at_node & edge_bv;
                if live_from_edge.is_empty() {
                    continue;
                }

                let pred_live = live.get_mut(pred_ptr).unwrap();
                let old_len = pred_live.len();
                *pred_live |= &live_from_edge;
                if pred_live.len() > old_len {
                    worklist.push_back(*pred_ptr);
                }
            }
        }
    }

    // 3. Prune the graph based on the computed live sets.
    for node_arc in &all_nodes {
        let mut guard = node_arc.write().unwrap();
        guard.children_mut().retain(|_edge_key, dest_map| {
            dest_map.retain(|child_wrapper, edge_value_bv| {
                if let Some(child_arc) = child_wrapper.upgrade() {
                    let child_ptr = Arc::as_ptr(&child_arc);
                    let live_from_child = live.get(&child_ptr).unwrap();
                    let live_on_edge = &*edge_value_bv & live_from_child;
                    if live_on_edge.is_empty() {
                        false
                    } else {
                        *edge_value_bv = live_on_edge;
                        true
                    }
                } else {
                    false // Dangling weak pointer, prune it.
                }
            });
            !dest_map.is_empty()
        });
        // Update the node's own live_tokens field with the final computed value.
        let node_ptr = Arc::as_ptr(node_arc);
        guard.value.live_tokens = live.get(&node_ptr).unwrap().clone();
    }
    crate::debug!(2, "Finished pruning dead paths from trie 2.");
}

fn merge_consecutive_edges_trie2(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Merging consecutive edges in precomputed trie 2.");

    let mut changed_in_pass = true;
    let mut pass_num = 0;
    while changed_in_pass {
        pass_num += 1;
        crate::debug!(3, "Running merge pass #{}", pass_num);
        changed_in_pass = false;

        let roots_vec: Vec<_> = roots.values().cloned().collect();
        if roots_vec.is_empty() {
            break;
        }
        let all_nodes = Trie::all_nodes(&roots_vec);
        let root_ptrs: HashSet<_> = roots_vec.iter().map(Arc::as_ptr).collect();

        let mut predecessors: HashMap<*const RwLock<PrecomputeNode2>, Vec<(*const RwLock<PrecomputeNode2>, (usize, Option<StateID>), LLMTokenBV, NodePtr<RwLock<PrecomputeNode2>>)>> = HashMap::new();
        let mut arc_map: HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();

        for node_arc in &all_nodes {
            arc_map.insert(Arc::as_ptr(node_arc), node_arc.clone());
            let node_ptr = Arc::as_ptr(node_arc);
            let guard = node_arc.read().unwrap();
            for (edge_key, dest_map) in guard.children() {
                for (child_wrapper, edge_bv) in dest_map {
                    if let Some(child_arc) = child_wrapper.upgrade() {
                        let child_ptr = Arc::as_ptr(&child_arc);
                        predecessors.entry(child_ptr).or_default().push((node_ptr, edge_key.clone(), edge_bv.clone(), child_wrapper.clone()));
                    }
                }
            }
        }

        // Find a candidate node to merge away.
        let mut candidate_to_merge = None;

        for b_arc in &all_nodes {
            let b_ptr = Arc::as_ptr(b_arc);
            let b_guard = b_arc.read().unwrap();

            // Condition 1: Not a root and not an end node.
            if root_ptrs.contains(&b_ptr) || b_guard.value.end {
                continue;
            }

            // Condition 2: Exactly one outgoing edge to a single child C.
            let mut outgoing_edge_info = None;
            if b_guard.children().len() == 1 {
                if let Some((ek_b, dest_map_b)) = b_guard.children().iter().next() {
                    if dest_map_b.len() == 1 {
                        if let Some((c_wrapper, bv_b)) = dest_map_b.iter().next() {
                            if let Some(c_arc) = c_wrapper.upgrade() {
                                outgoing_edge_info = Some((ek_b.clone(), c_arc, bv_b.clone(), c_wrapper.is_strong()));
                            }
                        }
                    }
                }
            }

            if outgoing_edge_info.is_none() {
                continue;
            }
            let (ek_b, _c_arc, _bv_b, _is_strong_bc) = outgoing_edge_info.as_ref().unwrap();
            let (_k_b, s_b_opt) = ek_b;

            // Condition 3: Has at least one predecessor.
            let preds = if let Some(p) = predecessors.get(&b_ptr) {
                if p.is_empty() { continue; }
                p
            } else {
                continue;
            };

            // Condition 4: All incoming edges are mergeable.
            let all_preds_mergeable = preds.iter().all(|(_, ek_a, _, _)| {
                let (_k_a, s_a_opt) = ek_a;
                s_a_opt.is_none() || s_b_opt.is_none()
            });

            if all_preds_mergeable {
                candidate_to_merge = Some((b_arc.clone(), outgoing_edge_info.unwrap()));
                break; // Found one, let's process it and restart the pass.
            }
        }

        if let Some((b_arc, (ek_b, c_arc, bv_b, is_strong_bc))) = candidate_to_merge {
            changed_in_pass = true;
            let b_ptr = Arc::as_ptr(&b_arc);
            let preds = predecessors.remove(&b_ptr).unwrap_or_default();
            let (k_b, s_b_opt) = ek_b;

            for (a_ptr, ek_a, bv_a, b_node_ptr_from_a) in preds {
                let a_arc = arc_map.get(&a_ptr).unwrap();
                let mut a_guard = a_arc.write().unwrap();

                // 1. Remove edge A -> B
                if let Some(dest_map_a) = a_guard.children_mut().get_mut(&ek_a) {
                    dest_map_a.remove(&b_node_ptr_from_a);
                    if dest_map_a.is_empty() {
                        a_guard.children_mut().remove(&ek_a);
                    }
                }

                // 2. Add/merge edge A -> C
                let (k_a, s_a_opt) = ek_a;
                let k_new = k_a + k_b;
                let s_new_opt = s_a_opt.or(s_b_opt);
                let ek_new = (k_new, s_new_opt);
                let bv_new = bv_a & &bv_b;

                if !bv_new.is_empty() {
                    let dest_map_new = a_guard.children_mut().entry(ek_new).or_default();
                    let is_strong_ab = b_node_ptr_from_a.is_strong();
                    let is_strong_ac = is_strong_ab && is_strong_bc;

                    let c_key = if is_strong_ac {
                        NodePtr::Strong(ArcPtrWrapper::new(c_arc.clone()))
                    } else {
                        NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(&c_arc)))
                    };
                    
                    if let Some(existing_bv) = dest_map_new.get_mut(&c_key) {
                        *existing_bv |= &bv_new;
                    } else {
                        dest_map_new.insert(c_key, bv_new);
                    }
                }
            }
        }
    }
    if pass_num > 1 {
        crate::debug!(2, "Finished merging consecutive edges after {} passes.", pass_num - 1);
    } else {
        crate::debug!(2, "No consecutive edges to merge.");
    }
}

fn trie2_shape_hash(
    arc: &Arc<RwLock<PrecomputeNode2>>,
    memo: &mut HashMap<*const RwLock<PrecomputeNode2>, u64>,
) -> u64 {
    let ptr = Arc::as_ptr(arc);
    if let Some(&h) = memo.get(&ptr) {
        return h;
    }

    // Insert a placeholder to break cycles. A fixed value like 0 is fine.
    memo.insert(ptr, 0);

    let node_guard = arc.read().unwrap();
    let mut hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());

    // Hash shape-defining value fields
    node_guard.value.end.hash(&mut hasher);

    // Hash children structure
    let mut edge_hashes = Vec::new();
    for (ek, dest_map) in node_guard.children() {
        for (np, ev) in dest_map {
            let child = np.upgrade().expect("Dangling weak pointer in trie2_shape_hash");
            let child_h = trie2_shape_hash(&child, memo);
            let mut pair_hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
            ek.hash(&mut pair_hasher);
            ev.hash(&mut pair_hasher);
            np.is_strong().hash(&mut pair_hasher);
            child_h.hash(&mut pair_hasher);
            edge_hashes.push(pair_hasher.finish());
        }
    }

    edge_hashes.sort_unstable();
    for h in edge_hashes {
        h.hash(&mut hasher);
    }

    let final_hash = hasher.finish();
    // Update the memo with the real hash.
    memo.insert(ptr, final_hash);
    final_hash
}

fn trie2_shape_eq(
    a: &Arc<RwLock<PrecomputeNode2>>,
    b: &Arc<RwLock<PrecomputeNode2>>,
    cache: &mut HashMap<(*const RwLock<PrecomputeNode2>, *const RwLock<PrecomputeNode2>), bool>,
) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }

    let (p1, p2) = if Arc::as_ptr(a) < Arc::as_ptr(b) {
        (Arc::as_ptr(a), Arc::as_ptr(b))
    } else {
        (Arc::as_ptr(b), Arc::as_ptr(a))
    };

    if let Some(&res) = cache.get(&(p1, p2)) {
        return res;
    }

    cache.insert((p1, p2), true); // Optimistic insertion for cycles

    let guard_a = a.read().unwrap();
    let guard_b = b.read().unwrap();

    // Compare shape-defining value fields
    if guard_a.value.end != guard_b.value.end {
        cache.insert((p1, p2), false);
        return false;
    }

    // Compare children
    if guard_a.children().len() != guard_b.children().len() {
        cache.insert((p1, p2), false);
        return false;
    }

    for (ek, dest_map_a) in guard_a.children() {
        if let Some(dest_map_b) = guard_b.children().get(ek) {
            if dest_map_a.len() != dest_map_b.len() {
                cache.insert((p1, p2), false);
                return false;
            }

            let mut pairs_b: Vec<_> = dest_map_b.iter().map(|(np, ev)| (np.is_strong(), ev, np.upgrade().expect("Dangling weak pointer in trie2_shape_eq (b)"))).collect();

            for (np_a, ev_a) in dest_map_a.iter() {
                let arc_a = np_a.upgrade().expect("Dangling weak pointer in trie2_shape_eq (a)");
                let is_strong_a = np_a.is_strong();
                let mut found_match = false;
                for i in 0..pairs_b.len() {
                    let (is_strong_b, ev_b, ref arc_b) = pairs_b[i];
                    if is_strong_a == is_strong_b && ev_a == ev_b {
                        if trie2_shape_eq(&arc_a, arc_b, cache) {
                            pairs_b.remove(i);
                            found_match = true;
                            break;
                        }
                    }
                }
                if !found_match {
                    cache.insert((p1, p2), false);
                    return false;
                }
            }
        } else {
            cache.insert((p1, p2), false);
            return false;
        }
    }

    true
}

fn merge_nodes_trie2(roots: &mut BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 2.");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(&roots_vec);

    let pb = ProgressBar::new(all_nodes.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
            .expect("progress-bar"),
    );
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut canonical_nodes: HashMap<u64, Vec<Arc<RwLock<PrecomputeNode2>>>> = HashMap::new();
    let mut visited: HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();
    let mut shape_hash_memo: HashMap<*const RwLock<PrecomputeNode2>, u64> = HashMap::new();
    let mut shape_eq_cache: HashMap<(*const RwLock<PrecomputeNode2>, *const RwLock<PrecomputeNode2>), bool> = HashMap::new();

    let mut new_roots = BTreeMap::new();
    for (sid, root_arc) in roots.iter() {
        let canonical_root = deduplicate_recursive_trie2(
            root_arc.clone(),
            &mut canonical_nodes,
            &mut visited,
            &mut shape_hash_memo,
            &mut shape_eq_cache,
            &pb,
        );
        new_roots.insert(*sid, canonical_root);
    }
    *roots = new_roots;

    // Recompute depths after structural changes from merging
    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(&final_roots_vec);

    pb.finish_with_message("Finished merging Trie 2 nodes");
    crate::debug!(2, "Finished merging subtrees in trie 2. Canonical nodes: {}", canonical_nodes.values().map(|v| v.len()).sum::<usize>());
}

fn deduplicate_recursive_trie2(
    node_arc: Arc<RwLock<PrecomputeNode2>>,
    canonical_nodes: &mut HashMap<u64, Vec<Arc<RwLock<PrecomputeNode2>>>>,
    visited: &mut HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
    shape_hash_memo: &mut HashMap<*const RwLock<PrecomputeNode2>, u64>,
    shape_eq_cache: &mut HashMap<(*const RwLock<PrecomputeNode2>, *const RwLock<PrecomputeNode2>), bool>,
    pb: &ProgressBar,
) -> Arc<RwLock<PrecomputeNode2>> {
    let node_ptr = Arc::as_ptr(&node_arc);
    if let Some(cached_node) = visited.get(&node_ptr) {
        return cached_node.clone();
    }

    // Pre-emptively insert to break cycles.
    // We will update this later if we find a different canonical node.
    visited.insert(node_ptr, node_arc.clone());

    pb.inc(1);

    // Post-order: canonicalize children first
    let mut new_children_map = BTreeMap::new();
    let mut children_changed = false;

    {
        let node_guard = node_arc.read().unwrap();
        for (edge_key, dest_map) in node_guard.children() {
            let mut new_dest_map = OrderedHashMap::new();
            for (node_ptr_wrapper, edge_val) in dest_map.iter() {
                let child_arc = node_ptr_wrapper.upgrade().expect("Dangling weak pointer in deduplicate_recursive_trie2");
                let canonical_child_arc = deduplicate_recursive_trie2(
                    child_arc.clone(),
                    canonical_nodes,
                    visited,
                    shape_hash_memo,
                    shape_eq_cache,
                    pb,
                );
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
            if !new_dest_map.is_empty() {
                new_children_map.insert(edge_key.clone(), new_dest_map);
            }
        }
    }

    if children_changed {
        let mut node_guard = node_arc.write().unwrap();
        *node_guard.children_mut() = new_children_map;
        // max_depth will be recomputed globally at the end
    }

    // Now find a canonical representative for the current node
    let fp = trie2_shape_hash(&node_arc, shape_hash_memo);
    let bucket = canonical_nodes.entry(fp).or_default();

    for candidate_arc in bucket.iter() {
        if trie2_shape_eq(&node_arc, candidate_arc, shape_eq_cache) {
            // Found a match. Merge live_tokens and return the canonical version.
            let node_live_tokens = { node_arc.read().unwrap().value.live_tokens.clone() };
            if !node_live_tokens.is_empty() {
                let mut candidate_guard = candidate_arc.write().unwrap();
                candidate_guard.value.live_tokens |= node_live_tokens;
            }
            // Update visited map with the true canonical node.
            visited.insert(node_ptr, candidate_arc.clone());
            return candidate_arc.clone();
        }
    }

    // No match found. This node becomes a new canonical representative.
    bucket.push(node_arc.clone());
    // The visited map already contains (node_ptr, node_arc), which is correct in this case.
    node_arc
}

fn clone_trie2_graph(
    root: &Arc<RwLock<PrecomputeNode2>>,
) -> (
    Arc<RwLock<PrecomputeNode2>>,
    HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
) {
    // old_ptr -> new arc
    let mut map: HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();
    let mut q: VecDeque<Arc<RwLock<PrecomputeNode2>>> = VecDeque::new();

    let root_ptr = Arc::as_ptr(root);
    let root_value = { root.read().expect("poison").value.clone() };
    let new_root = Arc::new(RwLock::new(PrecomputeNode2::new(root_value)));
    map.insert(root_ptr, new_root.clone());
    q.push_back(root.clone());

    while let Some(old_arc) = q.pop_front() {
        let old_ptr = Arc::as_ptr(&old_arc);
        let new_arc = map.get(&old_ptr).expect("parent must be created").clone();

        // Snapshot children outside of lock to avoid recursive lock explosion.
        let children_snapshot: Vec<( (usize, Option<StateID>), Vec<(NodePtr<RwLock<PrecomputeNode2>>, LLMTokenBV)> )> = {
            let g = old_arc.read().expect("poison");
            g.children()
                .iter()
                .map(|(ek, dest_map)| {
                    let entries = dest_map
                        .iter()
                        .map(|(node_ptr, ev)| {
                            let _ = node_ptr.upgrade().expect("Dangling weak pointer in clone_trie2_graph (snapshot)");
                            (node_ptr.clone(), ev.clone())
                        })
                        .collect::<Vec<_>>();
                    (ek.clone(), entries)
                })
                .collect()
        };

        // For each child, ensure it exists in map (create a blank new node with same value).
        for (_ek, entries) in &children_snapshot {
            for (node_ptr, _ev) in entries {
                let child_arc_old = node_ptr.upgrade().expect("Dangling weak pointer in clone_trie2_graph (map population)");
                let child_ptr_old = Arc::as_ptr(&child_arc_old);
                if !map.contains_key(&child_ptr_old) {
                    let child_value = { child_arc_old.read().expect("poison").value.clone() };
                    let child_arc_new = Arc::new(RwLock::new(PrecomputeNode2::new(child_value)));
                    map.insert(child_ptr_old, child_arc_new);
                    q.push_back(child_arc_old);
                }
            }
        }

        // Now wire edges on new_arc
        {
            let mut new_g = new_arc.write().expect("poison");
            for (ek, entries) in children_snapshot {
                let dest_map = new_g.children_mut().entry(ek).or_default();
                for (old_node_ptr, ev) in entries {
                    let child_arc_old = old_node_ptr.upgrade().expect("Dangling weak pointer in clone_trie2_graph (wiring)");
                    let child_ptr_old = Arc::as_ptr(&child_arc_old);
                    let child_arc_new = map.get(&child_ptr_old).expect("must exist").clone();
                    // Preserve strong/weak kind of the original key
                    let new_key = if old_node_ptr.is_strong() {
                        NodePtr::Strong(ArcPtrWrapper::new(child_arc_new))
                    } else {
                        NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(&child_arc_new)))
                    };
                    dest_map.insert(new_key, ev);
                }
            }
        }
    }

    // Recompute max_depths in the clone to keep invariants consistent.
    Trie::recompute_all_max_depths(&[new_root.clone()]);
    (new_root, map)
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
    // tags:             RefCell<HashMap<NodePtr<RwLock<PrecomputeNode>>, LLMTokenBV>>, // Removed
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
        terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
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
                    PrecomputedNodeContents::root(internal_max_llm_token)
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
            terminal_follow_map,
            ignore_terminal_id,
            // tags: RefCell::new(HashMap::new()), // Removed
            end_node: ArcPtrWrapper::new(Arc::new(RwLock::new(PrecomputeNode::new(PrecomputedNodeContents::leaf())))),
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
                let node_guard = node_arc.read().unwrap();
                let is_end = node_guard.value.end;
                let mut out = Vec::new();
                for (ek, dest_map) in node_guard.children() {
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
                let child_arc = child_ptr.upgrade().expect("Dangling weak pointer in prune_with_substring_everything_state");
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
                let mut guard = node_arc.write().unwrap();
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
        self.dfs(&self.vocab.root, assoc);
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

        type NodePtr = *const PrecomputeNode;
        let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<GrammarTokenID>>> = HashMap::new();

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
            // process: Collect information about which edges to prune.
            |node, maybe_all_immediate_predecessors| {
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
                edges_to_keep.insert(node_ptr, keys_to_keep);
    
                true // Continue traversal
            },
        );

        // Now, apply the pruning.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&roots_vec);
        for node_arc in all_nodes {
            let node_ptr: NodePtr = {
                let guard = node_arc.read().expect("poison");
                &*guard as *const _
            };
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

        // For each root, run the pruning process. This will modify the trie in-place.
        // We do not remove the root from the map even if it becomes "dead" (has no live paths).
        // This ensures that every tokenizer state ID that started with a trie root still has one,
        // preventing panics in later stages that expect a complete map.
        for root_arc in self.roots.values() {
            let root_wrapper = NodePtr::Strong(ArcPtrWrapper::new(root_arc.clone()));
            self.get_live_tokens_and_prune(root_wrapper, &mut live_tokens_cache);
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
            // Update the node's own live_tokens field
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
                        PrecomputedNodeContents::internal(),
                    )));

                    // b. Add edge I --(gtid)--> D
                    let mut union_bv = LLMTokenBV::zeros();
                    for (_, bv) in &sources {
                        union_bv |= bv;
                    }

                    {
                        let mut intermediate_guard = intermediate_node.write().expect("poison");
                        let mut edge_val_opt = Some(union_bv.clone());
                        // No cycle possible since I is new. Use unchecked for speed.
                        // Depth will be propagated to D.
                        intermediate_guard.try_insert_unchecked(Some(gtid), &mut edge_val_opt, dest_arc.clone())
                            .expect("Cycle detected when adding factored edge; this should not happen.");
                        intermediate_guard.value.live_tokens |= &union_bv; // Update live_tokens for intermediate node
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
                        src_guard.value.live_tokens |= bv; // Update live_tokens for source node
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

        // Pre-emptively insert to break cycles.
        visited.insert(node_ptr, node_arc.clone());

        // Post-order traversal: first, canonicalize all children.
        let mut new_children_map = BTreeMap::new();
        let mut children_changed = false;

        {
            let node_guard = node_arc.read().unwrap();
        for (edge_key, dest_map) in node_guard.children() {
            let mut new_dest_map = OrderedHashMap::new();
            for (node_ptr_wrapper, edge_val) in dest_map.iter() {
                let child_arc = node_ptr_wrapper.upgrade().expect("Dangling weak pointer in deduplicate_recursive");
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
            if !new_dest_map.is_empty() {
                new_children_map.insert(edge_key.clone(), new_dest_map);
                }
            }
        }

    if children_changed {
        let mut node_guard = node_arc.write().unwrap();
        *node_guard.children_mut() = new_children_map;
        node_guard.recompute_max_depth();
        // The live_tokens field will be recomputed by prune_dead_paths after merging.
    }

    let canonical_arc = {
            let node_guard = node_arc.read().unwrap();
            let node_content = (*node_guard).clone();
            canonical_nodes.entry(node_content).or_insert_with(|| node_arc.clone()).clone()
        };

        // Update with the final canonical arc.
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
                                    |node_value, edge_value| {
                                        crate::debug!(7, "Before updating live tokens {:?} |= {:?}", node_value.live_tokens, edge_value);
                                        node_value.live_tokens |= edge_value;
                                        crate::debug!(7, "After updating live tokens: {:?}", node_value.live_tokens);
                                    },
                                    |ev, t| *ev &= &t.live_tokens,
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
                                |node_value, edge_value| node_value.live_tokens |= edge_value,
                                |ev, t| *ev &= &t.live_tokens,
                            );

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos).or_default().entry(next_tokenizer_state).or_default();

                            inserter = inserter.try_destinations_iter(dest_nodes_in_queue.iter().filter_map(|w| w.upgrade()).filter(|w| !w.read().unwrap().value.end));

                            if true {
                                let children_of_src: Vec<_> = src_node_wrapper.upgrade().unwrap().read().unwrap().children().values().flat_map(|m| m.keys().cloned()).collect();
                                // let tags = self.tags.borrow(); // Removed
                                let eligible_children = children_of_src.iter().map(|child_node_ptr| {
                                    child_node_ptr.upgrade().expect("Dangling weak pointer in Precomputer::dfs")
                                }).filter(|child_arc| {
                                    (child_arc.read().unwrap().value.live_tokens.clone() & &edge_bv).is_empty() && !child_arc.read().unwrap().value.end
                                });
                                inserter = inserter.try_destinations_iter(eligible_children);
                                // drop(tags); // Removed
                            }

                            let result_node = inserter.else_create_destination_with_value(PrecomputedNodeContents::internal()).unwrap();
                            let result_node_ptr = NodePtr::Strong(ArcPtrWrapper::new(result_node.clone()));
                            dest_nodes_in_queue.insert(result_node_ptr.clone());
                            // *self.tags.borrow_mut().entry(result_node_ptr).or_insert_with(HybridBitset::zeros) |= &edge_bv; // Removed
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
                                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                                    |ev, t| *ev &= &t.live_tokens,
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
                self.dfs(child_vocab_node, next_level_assoc);
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
        // self.get_mask1()
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

                            let mut allowed_terminals = TerminalBV::zeros();
                            for gtid_opt in precomputed_node_data.children().keys() {
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

        // Log the GSSs
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
                // if !glr_s.is_ok() {
                //     crate::debug!(4, "GLR state is not alive before popping, skipping.");
                //     return Vec::new();
                // }
                crate::debug!(4, "Processing step for k: {:?}, expected_state_id_opt: {:?}", k, expected_state_id_opt);
                // glr_s.log_gss("Before popping", TerminalID(0), false, false);
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
                if out_gsss.is_empty() {
                    crate::debug!(4, "No valid GSS nodes after popping, skipping.");
                    return Vec::new();
                }
                let out_gss = GSSNode::merge_many_with_depth(1, out_gsss);
                crate::debug!(4, "After popping {} from GSS: {}", k, print_gss_forest(&[out_gss.clone()], &self.parent.parser.terminal_map, &GSSPrintConfig::default()).0);
                // if !out_gss.is_alive() {
                //     crate::debug!(4, "GLR state is not alive after popping, skipping.");
                //     return Vec::new();
                // }
                let mut out = Vec::new();
                for (dst_node_wrapper, edge_bv) in dest_map.iter() {
                    let mut out_gss_filtered = out_gss.clone();
                    crate::debug!(5, "Filtering GSS for edge LLM tokens: {:?}", edge_bv);
                    allow_only_llm_tokens_and_prune_arc(&mut out_gss_filtered, edge_bv, &mut HashMap::new());
                    let mut out_glr_s = glr_s.clone();
                    out_glr_s.active_state.stack = out_gss_filtered;
                    crate::debug!(4, "Allowed LLM tokens in out_gss_filtered: {:?}", out_glr_s.active_state.stack.allowed_llm_tokens());
                    // out_glr_s.log_gss("After filtering for edge LLM tokens", TerminalID(0), false, false);
                    // if out_glr_s.is_ok() {
                        out.push((dst_node_wrapper.clone(), out_glr_s));
                    }
                // }
                out
            },
            // merge_fn
            |glr_s1, glr_s2| {
                crate::debug!(4, "Merging two GLR states");
                glr_s1.merge_with(glr_s2);
            },
            // process_fn: (precomputed_node_data, final_glr_s_for_this_path)
            |precomputed_node_data, glr_s| {
                crate::debug!(4, "Processing node {:p}", precomputed_node_data);
                // glr_s.log_gss("At process_fn", TerminalID(0), false, false);
                let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                // let keep_going = !glr_active_tokens.is_empty();
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
                    crate::debug!(4, "Precomputed node data is an end node, adding active tokens {:?} to final mask", glr_active_tokens);
                    *final_mask_internal.borrow_mut() |= glr_active_tokens;
                } else {
                    crate::debug!(4, "Precomputed node data is not an end node, active tokens: {:?}", glr_active_tokens);
                }
                keep_going
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

        // Log the GSSs
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

        // for (state_id, state) in &self.state {
        //     crate::debug!(3, "State {} before commit:", state_id.0);
        //     state.log_gss("Before commit", TerminalID(0), false, false);
        // }

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
                        // cloned_glr_s.log_gss(format!("Before disallowing terminals {:?} after committing bytes {:?}", &disallowed_terminals, &llm_token_bytes[offset..new_offset]).as_str(), TerminalID(match_info.id), false, false);
                        disallow_terminals_and_prune_arc(&mut cloned_glr_s.active_state.stack, &disallowed_terminals, &mut HashMap::new());
                        // cloned_glr_s.log_gss(format!("After disallowing terminals {:?} after committing bytes {:?}", &disallowed_terminals, &llm_token_bytes[offset..new_offset]).as_str(), TerminalID(match_info.id), false, false);

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
            // glr_parser_state.process_default_reductions();
        }

        // TODO: this shouldn't be necessary, but due to some order-dependent LLM token BV weirdness in GSS, it is necessary to ensure commit order invariance.
        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        // let mut fuse_memo = HashMap::new();
        // for state in self.state.values_mut() {
            // state.active_state.stack = fuse_predecessors_recursive(&mut state.active_state.stack, 3, &mut fuse_memo);
        // }
        // fuse_memo.clear();

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
                // glr_state.log_gss("After commit", TerminalID(0), false, false);
            }
        }
    }

    pub fn is_active_or_accepted(&self) -> bool {
        !self.state.is_empty() && self.state.values().any(|s| !s.active_state.stack.is_empty() || s.has_accepted())
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}
