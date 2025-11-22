use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ops::BitOrAssign;
use std::sync::Arc;

use bimap::BiBTreeMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use ordered_hash_map::OrderedHashMap;
use range_set_blaze::RangeSetBlaze;

use crate::constraint_extra::PrecomputeStats;
use crate::constraint_precompute1_utils;
use crate::constraint_trie::{
    PrecomputeNode1, PrecomputeNode1Index, Precomputed, PrecomputedNodeContents, Trie0GodWrapper,
    Trie1GodWrapper, Trie2Index,
};
use crate::constraint_vocab::{LLMTokenBV, LLMVocab, StageVocab};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::Trie;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::glr::grammar::Terminal;
use crate::glr::parser::GLRParser;
use crate::precompute4::weighted_automata::bitset::SimpleBitset;
use crate::precompute4::weighted_automata::{NWA, NWAStateID, Weight};
use crate::profiler::{self, PROGRESS_BAR_ENABLED};
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::constraint::GrammarConstraintConfig;
use crate::precompute4::weighted_automata::common::Label;

// ---------------------------------------------------------------------------
// Precomputer1
// ---------------------------------------------------------------------------

pub(crate) struct Precomputer1<'r> {
    pub(crate) tokenizer: &'r Regex,
    pub(crate) parser: Option<&'r GLRParser>,
    pub(crate) llm_vocab: Option<Arc<LLMVocab>>,
    pub(crate) vocab: VocabPrefixTree,
    pub(crate) roots: BTreeMap<TokenizerStateID, NWAStateID>,
    pub(crate) possible_matches: RefCell<
        BTreeMap<
            *const VocabPrefixTreeNode,
            BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>,
        >,
    >,
    pub(crate) all_llm_tokens: RangeSetBlaze<usize>,
    pub(crate) pb: ProgressBar,
    pub(crate) stats: PrecomputeStats,
    pub(crate) leaf_state: NWAStateID,
    pub(crate) nwa: NWA,
    pub(crate) original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
}

impl<'r> Precomputer1<'r> {
    fn new(
        tokenizer: &'r Regex,
        parser: Option<&'r GLRParser>,
        llm_vocab: Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
        active_states: Vec<TokenizerStateID>,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        crate::debug!(5, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(5, "Done building vocab prefix tree");

        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Clear default start state

        let mut roots = BTreeMap::new();
        for sid in active_states {
            let root_state = nwa.add_state();
            roots.insert(sid, root_state);
        }
        crate::debug!(4, "Created trie1 roots ({} states)", roots.len());

        crate::debug!(5, "Counting vocab nodes for progress bar...");
        let total_nodes = count_vocab_nodes(&vocab.root);
        crate::debug!(5, "Counted {} vocab nodes", total_nodes);
        let pb = ProgressBar::new(total_nodes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] \
                     [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})",
                )
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let leaf_state = nwa.add_state();
        nwa.states[leaf_state].final_weight = Some(Weight::all());
        crate::debug!(5, "Created trie1 leaf state");

        Self {
            tokenizer,
            parser,
            llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            all_llm_tokens: RangeSetBlaze::from_iter(0..=internal_max_llm_token),
            pb,
            stats: PrecomputeStats::default(),
            leaf_state,
            nwa,
            original_to_dummy_map,
        }
    }

    fn get_leaf_node(&self) -> NWAStateID {
        self.leaf_state
    }

    fn finish(mut self) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper)
    {
        // TODO: make this simpler.
        let new_start_state = self.nwa.add_state();
        for (tsid, state) in &self.roots {
            self.nwa.add_transition(new_start_state, tsid.0 as Label, *state, Weight::all()).unwrap();
        }
        self.nwa.body.start_state = new_start_state;
        crate::debug!(3, "Simplifying NWA with {} states...", self.nwa.states.len());
        self.nwa.simplify();
        crate::debug!(3, "Determinizing NWA with {} states...", self.nwa.states.len());
        let mut dwa = self.nwa.determinize();
        crate::debug!(3, "Simplifying DWA with {} states...", dwa.states.len());
        self.nwa.simplify();
        crate::debug!(3, "Unrolling DWA with {} states...", dwa.states.len());
        dwa = dwa.unroll_cycles();
        let sink_state = dwa.add_state();
        for (tsid, state) in &mut self.roots {
            let new_state = *dwa.states[dwa.body.start_state].transitions.get(&(tsid.0 as Label)).unwrap_or(&sink_state);
            *state = new_state;
        }
        crate::debug!(3, "Converting DWA to NWA with {} states...", dwa.states.len());
        self.nwa = NWA::from_dwa(&dwa);
        crate::debug!(4, "Done converting DWA to NWA with {} states...", dwa.states.len());

        let final_trie1_god = Trie1GodWrapper::new();
        let mut final_roots = BTreeMap::new();

        (final_roots, final_trie1_god)
    }

    fn possible_matches(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key_ptr = vocab_node as *const VocabPrefixTreeNode;

        if let Some(cached_for_vocab_node) =
            self.possible_matches.borrow().get(&cache_key_ptr)
        {
            if let Some(cached_result) =
                cached_for_vocab_node.get(&tokenizer_state_id)
            {
                return cached_result.clone();
            }
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result =
                self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_node.reachable_token_ids();
                *result_map
                    .entry(grammar_token_id)
                    .or_insert_with(LLMTokenBV::zeros) |=
                    HybridBitset::from(applicable_tokens);
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: std::collections::BTreeSet<_> = self
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(final_state_val))
                    .into_iter()
                    .collect();
                let matches_here: std::collections::BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();
                let possible_new_matches =
                    &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(
                        child_vocab_node,
                        TokenizerStateID(final_state_val),
                    );
                    for (token, bv) in next_results {
                        *result_map
                            .entry(token)
                            .or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }

        self.possible_matches
            .borrow_mut()
            .entry(cache_key_ptr)
            .or_default()
            .insert(tokenizer_state_id, result_map.clone());

        result_map
    }

    fn run_dfs(&mut self) {
        let assoc = self.roots.clone();
        crate::debug!(3, "Starting precompute DFS for {} tokenizer states", self.roots.len());
        for (sid, root) in &self.roots {
            crate::debug!(6, "  {}: {}", sid.0, root);
        }
        profiler::reset();
        let vocab = std::mem::replace(&mut self.vocab, VocabPrefixTree::new());
        self.dfs(&vocab.root, assoc);
        self.vocab = vocab;
        self.pb.finish();
        profiler::print_summary();
        crate::debug!(3, "Precomputation complete");
    }

    fn dfs(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, NWAStateID>,
    ) {
        self.pb.inc(1);
        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut next_level_assoc: BTreeMap<TokenizerStateID, NWAStateID> =
                BTreeMap::new();

            // Queue: pos -> TokenizerState -> (NWAState -> ContextTokens)
            let mut pending: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, NWAStateID>,
            > = BTreeMap::new();
            pending.insert(0, assoc_by_state.clone());

            let mut pending_edges = Vec::new();

            let child_reachable = child_vocab_node.reachable_token_ids();
            let child_token_id = child_vocab_node.token_id();

            // Caches possible matches for end states to prune edge_bv
            let mut possible_matches_at_end_cache: HashMap<
                TokenizerStateID,
                BTreeMap<GrammarTokenID, LLMTokenBV>,
            > = HashMap::new();

            while let Some((pos, states_at_pos)) = pending.pop_first() {
                // If we reached the end of the segment, these states are ready for the next vocab node
                if pos == segment_bytes.len() {
                    for (tokenizer_state_id, node) in states_at_pos {
                        let next = *next_level_assoc.entry(tokenizer_state_id).or_insert_with(|| self.nwa.add_state());
                        self.nwa.add_epsilon(node, next, SimpleBitset::all());
                    }
                    continue;
                }

                for (tokenizer_state_id, src_node) in states_at_pos {
                    let exec_result = self
                        .tokenizer
                        .execute_from_state(&segment_bytes[pos..], tokenizer_state_id);

                    let possible_matches_at_end = if let Some(end_val) = exec_result.end_state {
                        let ts = TokenizerStateID(end_val);
                        possible_matches_at_end_cache
                            .entry(ts)
                            .or_insert_with(|| self.possible_matches(child_vocab_node, ts))
                    } else {
                        // Dummy empty map
                        possible_matches_at_end_cache
                            .entry(TokenizerStateID(usize::MAX)) // Arbitrary key that won't be hit
                            .or_default()
                    };

                    // 1. Handle Matches -> Transitions to Initial State
                    for match_info in &exec_result.matches {
                        let terminal_id = GrammarTokenID(match_info.id);
                        let next_pos = pos + match_info.width;

                        // Leaf check: if match consumes remainder of segment
                        if next_pos == segment_bytes.len() {
                            let mut edge_bv = RangeSetBlaze::new();
                            edge_bv.insert(child_token_id);
                            let final_bv = edge_bv;
                            if !final_bv.is_empty() {
                                let leaf = self.get_leaf_node();
                                pending_edges.push((
                                    src_node,
                                    leaf,
                                    Some(terminal_id),
                                    final_bv.clone(),
                                ));
                            }
                        }

                        // Continuation logic
                        let mut edge_bv = child_reachable.clone();
                        if next_pos == segment_bytes.len() {
                            edge_bv.remove(child_token_id);
                            if let Some(pm) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv = &edge_bv - pm.inner.as_ref();
                            }
                        }

                        let final_bv = edge_bv;
                        if final_bv.is_empty() {
                            continue;
                        }

                        let dest_map = pending.entry(next_pos).or_default();

                        let initial_tsid = self.tokenizer.initial_state_id();
                        let target = *dest_map
                            .entry(initial_tsid)
                            .or_insert_with(|| self.nwa.add_state());

                        pending_edges.push((src_node, target, Some(terminal_id), final_bv));
                    }

                    // 2. Handle End State -> Continuation
                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state = TokenizerStateID(end_state_val);
                        let accessible_terminals = self
                            .tokenizer
                            .tokens_accessible_from_state(final_tokenizer_state);

                        let mut edge_bv = RangeSetBlaze::new();
                        edge_bv.insert(child_token_id);
                        let final_edge_bv = edge_bv;

                        if !final_edge_bv.is_empty() {
                            let end_idx = self.get_leaf_node();
                            for terminal_id in &accessible_terminals {
                                pending_edges.push((
                                    src_node,
                                    end_idx,
                                    Some(*terminal_id),
                                    final_edge_bv.clone(),
                                ));
                            }
                        }

                        let next = *next_level_assoc.entry(final_tokenizer_state).or_insert_with(|| self.nwa.add_state());
                        self.nwa.add_epsilon(src_node, next, Weight::all());
                    }
                }
            }

            // Apply all batched writes
            for (src, dst, key, bv) in pending_edges {
                if let Some(k) = key {
                    let weight = SimpleBitset::from_rsb(bv);
                    let _ = self.nwa.add_transition(src, k.0 as Label, dst, weight);
                }
            }

            if !next_level_assoc.is_empty() {
                self.dfs(child_vocab_node, next_level_assoc);
            }
        }
    }
}

pub(crate) fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
}

// Public entry point wrapper
pub fn run_precompute1(
    tokenizer: &Regex,
    parser: Option<&GLRParser>,
    llm_vocab: Option<Arc<LLMVocab>>,
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    token_name_map: &BiBTreeMap<Terminal, usize>,
    stage_vocab: &mut StageVocab,
    terminal_follow_map: &BTreeMap<GrammarTokenID, std::collections::BTreeSet<GrammarTokenID>>,
    config: &GrammarConstraintConfig,
    original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper) {
    let mut dummy_terminal_penalties: BTreeMap<TerminalID, usize> = BTreeMap::new();
    if !config.dummy_terminal_penalties.is_empty() {
        if let Some(p) = parser {
            for (dummy_name, penalty) in &config.dummy_terminal_penalties {
                let dummy_term = Terminal::regex_name(dummy_name);
                if let Some(&dummy_id) = p.terminal_map.get_by_left(&dummy_term) {
                    dummy_terminal_penalties.insert(dummy_id, *penalty);
                }
            }
        }
    } else {
        for dummy_tid in original_to_dummy_map.values() {
            *dummy_terminal_penalties.entry(*dummy_tid).or_default() += 1;
        }
    }

    // Reduce internal_llm_token_map to representatives to speed up precomputation
    let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut seen_internal_ids = std::collections::HashSet::new();

    for (bytes, id) in internal_llm_token_map {
        if seen_internal_ids.insert(id.0) {
            representative_llm_token_map.insert(bytes.clone(), *id);
        }
    }

    let representative_states: Vec<TokenizerStateID> = tokenizer.iter_states().collect();

    let mut helper = Precomputer1::new(
        tokenizer,
        parser,
        llm_vocab,
        &representative_llm_token_map,
        stage_vocab.internal_max_llm_token,
        original_to_dummy_map,
        representative_states,
    );

    helper.run_dfs();

    let (mut precomputed1, trie1_god) = helper.finish();

    // Trie1 optimization (size, vocab compression)
    constraint_precompute1_utils::optimize_trie1_size(
        &mut precomputed1,
        &trie1_god,
        // Dummy values for Trie0-dependent params (we no longer build Trie0).
        &Trie0GodWrapper::new(),
        &HashMap::new(),
        parser.and_then(|p| p.ignore_terminal_id),
        stage_vocab.internal_max_llm_token,
        terminal_follow_map,
        &config.trie1,
        stage_vocab,
        token_name_map,
        &dummy_terminal_penalties,
    );

    (precomputed1, trie1_god)
}
