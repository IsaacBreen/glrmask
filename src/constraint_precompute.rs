use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::BitOrAssign;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use range_set_blaze::RangeSetBlaze;

use crate::constraint_vocab::{LLMTokenBV, LLMVocab};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::glr::parser::GLRParser;
use crate::precompute4::weighted_automata::bitset::SimpleBitset;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, Weight};
use crate::profiler::{self, PROGRESS_BAR_ENABLED};
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID as GrammarTokenID;
use crate::precompute4::weighted_automata::common::Label;

// ---------------------------------------------------------------------------
// Precomputer1
// ---------------------------------------------------------------------------

pub(crate) struct Precomputer1<'r> {
    pub(crate) tokenizer: &'r Regex,
    pub(crate) parser: Option<&'r GLRParser>,
    pub(crate) original_llm_vocab: Option<Arc<LLMVocab>>,
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
    pub(crate) leaf_state: NWAStateID,
    pub(crate) nwa: NWA,
    pub(crate) terminals_count: usize,
    pub(crate) pending_transitions: HashMap<NWAStateID, HashMap<Label, HashMap<NWAStateID, Weight>>>,
    pub(crate) pending_epsilons: HashMap<NWAStateID, HashMap<NWAStateID, Weight>>,
    pub(crate) live_tokens: HashMap<NWAStateID, Weight>,
}

impl<'r> Precomputer1<'r> {
    fn new(
        tokenizer: &'r Regex,
        parser: Option<&'r GLRParser>,
        original_llm_vocab: Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        terminals_count: usize,
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
            original_llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            all_llm_tokens: RangeSetBlaze::from_iter(0..=internal_max_llm_token),
            pb,
            leaf_state,
            nwa,
            terminals_count,
            pending_transitions: HashMap::new(),
            pending_epsilons: HashMap::new(),
            live_tokens: HashMap::new(),
        }
    }

    fn finish(mut self) -> DWA {
        // Flush pending transitions and epsilons into the NWA
        for (src, labels) in std::mem::take(&mut self.pending_transitions) {
            for (label, dsts) in labels {
                for (dst, weight) in dsts {
                    self.nwa.add_transition(src, label, dst, weight).unwrap();
                }
            }
        }
        for (src, dsts) in std::mem::take(&mut self.pending_epsilons) {
            for (dst, weight) in dsts {
                self.nwa.add_epsilon(src, dst, weight);
            }
        }

        let new_start_state = self.nwa.add_state();
        for (tsid, state) in &self.roots {
            let label = (tsid.0 + self.terminals_count) as Label;
            self.nwa.add_transition(new_start_state, label, *state, Weight::all()).unwrap();
        }
        self.nwa.body.start_states = vec![new_start_state];

        // Stats
        // Find cases where there's multiple instances of same transition - incl symbol/epsilon transition - from one state to another, regardless of weight.
        let mut duplicate_transitions = 0;
        for state in &self.nwa.states.0 {
            let mut dst_counts = HashMap::new();
            for (dst, _) in &state.epsilons {
                *dst_counts.entry(*dst).or_insert(0) += 1;
            }
            for count in dst_counts.values() {
                if *count > 1 {
                    duplicate_transitions += count - 1;
                }
            }

            for targets in state.transitions.values() {
                let mut dst_counts = HashMap::new();
                for (dst, _) in targets {
                    *dst_counts.entry(*dst).or_insert(0) += 1;
                }
                for count in dst_counts.values() {
                    if *count > 1 {
                        duplicate_transitions += count - 1;
                    }
                }
            }
        }
        if duplicate_transitions > 0 {
            crate::debug!(4, "NWA: Found {} duplicate transitions (same src, dst, label)", duplicate_transitions);
        }

        // Find cases where there's multiple instances of same transition - regardless of symbol/epsilon transition - from one state to another, regardless of weight.
        let mut parallel_connections = 0;
        for state in &self.nwa.states.0 {
            let mut dst_counts = HashMap::new();
            for (dst, _) in &state.epsilons {
                *dst_counts.entry(*dst).or_insert(0) += 1;
            }
            for targets in state.transitions.values() {
                for (dst, _) in targets {
                    *dst_counts.entry(*dst).or_insert(0) += 1;
                }
            }

            for count in dst_counts.values() {
                if *count > 1 {
                    parallel_connections += 1;
                }
            }
        }
        if parallel_connections > 0 {
            crate::debug!(4, "NWA: Found {} pairs of states connected by multiple transitions", parallel_connections);
        }

        crate::debug!(3, "{} states and {} transitions", self.nwa.states.len(), self.nwa.states.num_transitions());
        
        // OPTIMIZATION: Use lightweight operations instead of full simplify()
        // This skeleton DWA is only used as input to precompute4, so expensive minimization
        // provides little benefit. Just do basic cleanup.
        self.nwa.compress_transitions();
        crate::debug!(3, "Compressed NWA with {} states and {} transitions", self.nwa.states.len(), self.nwa.states.num_transitions());
        
        let dwa = self.nwa.determinize_and_simplify("Precompute1");
        crate::debug!(3, "Simplified DWA with {} states and {} transitions", dwa.states.len(), dwa.states.num_transitions());

        dwa
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

    fn get_or_create_next_state(
        &mut self,
        src_node: NWAStateID,
        tokenizer_state: TokenizerStateID,
        next_level_assoc: &mut BTreeMap<TokenizerStateID, NWAStateID>,
    ) -> NWAStateID {
        match next_level_assoc.entry(tokenizer_state) {
            std::collections::btree_map::Entry::Occupied(o) => *o.get(),
            std::collections::btree_map::Entry::Vacant(v) => {
                let mut reuse = None;
                if let Some(dsts) = self.pending_epsilons.get(&src_node) {
                    for (dst, _) in dsts {
                        if self.live_tokens.get(dst).map_or(true, |live| live.is_disjoint(&Weight::all())) {
                            reuse = Some(*dst);
                            break;
                        }
                    }
                }
                let t = reuse.unwrap_or_else(|| self.nwa.add_state());
                v.insert(t);
                t
            }
        }
    }

    fn add_pending_transition(&mut self, src: NWAStateID, label: Label, dst: NWAStateID, weight: Weight) {
        self.pending_transitions
            .entry(src)
            .or_default()
            .entry(label)
            .or_default()
            .entry(dst)
            .and_modify(|w| *w |= &weight)
            .or_insert(weight.clone());
        *self.live_tokens.entry(dst).or_insert_with(Weight::zeros) |= &weight;
    }

    fn add_pending_epsilon(&mut self, src: NWAStateID, dst: NWAStateID, weight: Weight) {
        self.pending_epsilons
            .entry(src)
            .or_default()
            .entry(dst)
            .and_modify(|w| *w |= &weight)
            .or_insert(weight.clone());
        *self.live_tokens.entry(dst).or_insert_with(Weight::zeros) |= &weight;
    }

    fn run_dfs(&mut self) {
        let assoc = self.roots.clone();
        crate::debug!(3, "Starting precompute DFS for {} tokenizer states", self.roots.len());
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
                        let next = self.get_or_create_next_state(node, tokenizer_state_id, &mut next_level_assoc);
                        self.add_pending_epsilon(node, next, Weight::all());
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
                                let leaf = self.leaf_state;
                                let weight = SimpleBitset::from_rsb(final_bv);
                                self.add_pending_transition(src_node, terminal_id.0 as Label, leaf, weight);
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
                        let weight = SimpleBitset::from_rsb(final_bv);

                        let target_entry = dest_map.entry(initial_tsid);
                        let target = match target_entry {
                            std::collections::btree_map::Entry::Occupied(o) => *o.get(),
                            std::collections::btree_map::Entry::Vacant(v) => {
                                let t = self.nwa.add_state();
                                v.insert(t);
                                t
                            }
                        };

                        self.add_pending_transition(src_node, terminal_id.0 as Label, target, weight);
                    }

                    // 2. Handle End State -> Continuation
                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state = TokenizerStateID(end_state_val);
                        let accessible_terminals = self.tokenizer.tokens_accessible_from_state(final_tokenizer_state);

                        let mut edge_bv = RangeSetBlaze::new();
                        edge_bv.insert(child_token_id);
                        let final_edge_bv = edge_bv;

                        if !final_edge_bv.is_empty() {
                            let end_idx = self.leaf_state;
                            for terminal_id in &accessible_terminals {
                                let weight = SimpleBitset::from_rsb(final_edge_bv.clone());
                                self.add_pending_transition(
                                        src_node,
                                        terminal_id.0 as Label,
                                        end_idx,
                                        weight,
                                    );
                            }
                        }

                        let next = self.get_or_create_next_state(src_node, final_tokenizer_state, &mut next_level_assoc);
                        self.add_pending_epsilon(src_node, next, Weight::all());
                    }
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
    original_llm_vocab: Option<Arc<LLMVocab>>,
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminals_count: usize,
    active_states: Vec<TokenizerStateID>,
) -> DWA {
    // Reduce internal_llm_token_map to representatives to speed up precomputation
    let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut seen_internal_ids = std::collections::HashSet::new();

    for (bytes, id) in internal_llm_token_map {
        if seen_internal_ids.insert(id.0) {
            representative_llm_token_map.insert(bytes.clone(), *id);
        }
    }

    let mut helper = Precomputer1::new(
        tokenizer,
        parser,
        original_llm_vocab,
        &representative_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        active_states,
    );


    helper.run_dfs();

    helper.finish()
}
