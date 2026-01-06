use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::BitOrAssign;
use std::sync::Arc;


use range_set_blaze::RangeSetBlaze;

use crate::constraint_vocab::LLMTokenBV;
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::glr::parser::GLRParser;
use crate::precompute4::weighted_automata::rangeset::RangeSet as WARangeSet;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, Weight};
use crate::precompute4::weighted_automata::weight_expansion::{expand_rsb, create_tsid_set_mask};
use crate::profiler::{self};

use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID as GrammarTokenID;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

// No-op progress bar replacement
struct NoOpPb;
impl NoOpPb {
    fn inc(&self, _: u64) {}
    fn finish(&self) {}
}

// ---------------------------------------------------------------------------
// Precomputer1
// ---------------------------------------------------------------------------

pub(crate) struct Precomputer1<'r> {
    pub(crate) tokenizer: &'r Regex,
    pub(crate) vocab: VocabPrefixTree,
    pub(crate) roots: BTreeMap<TokenizerStateID, NWAStateID>,
    pub(crate) state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
    pub(crate) possible_matches: RefCell<
        BTreeMap<
            *const VocabPrefixTreeNode,
            BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>,
        >,
    >,
    pub(crate) all_llm_tokens: RangeSetBlaze<usize>,
    pub(crate) pb: NoOpPb,
    pub(crate) leaf_state: NWAStateID,
    pub(crate) nwa: NWA,
    pub(crate) terminals_count: usize,
    pub(crate) pending_transitions: HashMap<NWAStateID, HashMap<Label, HashMap<NWAStateID, Weight>>>,
    pub(crate) pending_epsilons: HashMap<NWAStateID, HashMap<NWAStateID, Weight>>,
    pub(crate) live_tokens: HashMap<NWAStateID, Weight>,
    // Cache for tokens_accessible_from_state - only 389 unique states but called 700k+ times
    accessible_terminals_cache: HashMap<TokenizerStateID, std::rc::Rc<Vec<GrammarTokenID>>>,
    // Weight-heavy mode: number of tokenizer states
    pub(crate) num_tsids: usize,
    // Max LLM token ID for creating tsid masks
    pub(crate) internal_max_llm_token: usize,
}

impl<'r> Precomputer1<'r> {
    fn new(
        tokenizer: &'r Regex,
        internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        terminals_count: usize,
        state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
        num_tsids: usize,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        crate::debug!(6, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(6, "Done building vocab prefix tree");

        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Clear default start state

        let mut roots = BTreeMap::new();
        for &rep_sid in state_to_rep.values() {
            if !roots.contains_key(&rep_sid) {
                let root_state = nwa.add_state();
                roots.insert(rep_sid, root_state);
            }
        }
        crate::debug!(5, "Created trie1 roots ({} states for {} total tsids)", roots.len(), state_to_rep.len());

        let pb = NoOpPb;

        let leaf_state = nwa.add_state();
        // In weight-heavy mode, final weight should also be expanded
        // IMPORTANT: Use [0..=...] to create from ONE range, not iterate over all integers!
        nwa.states[leaf_state].final_weight = Some(Weight::from_rsb(
            expand_rsb(&RangeSetBlaze::from_iter([0..=internal_max_llm_token]), num_tsids)
        ));
        crate::debug!(6, "Created trie1 leaf state with expanded final weight");

        Self {
            tokenizer,
            vocab,
            roots,
            state_to_rep,
            possible_matches: RefCell::new(BTreeMap::new()),
            // IMPORTANT: Use [0..=...] to create from ONE range, not iterate over all integers!
            all_llm_tokens: RangeSetBlaze::from_iter([0..=internal_max_llm_token]),
            pb,
            leaf_state,
            nwa,
            terminals_count,
            pending_transitions: HashMap::new(),
            pending_epsilons: HashMap::new(),
            live_tokens: HashMap::new(),
            accessible_terminals_cache: HashMap::new(),
            num_tsids,
            internal_max_llm_token,
        }
    }

    fn finish(mut self) -> DWA {
        // Debug: print all states and transitions before processing
        crate::debug!(7, "=== NWA before flush (leaf_state={}, roots={:?}) ===", self.leaf_state, self.roots);
        for (i, state) in self.nwa.states.0.iter().enumerate() {
            let trans_count = state.transitions.values().map(|v| v.len()).sum::<usize>();
            let eps_count = state.epsilons.len();
            let is_final = state.final_weight.is_some();
            crate::debug!(7, "State {}: {} transitions, {} epsilons, final={}", i, trans_count, eps_count, is_final);
        }
        crate::debug!(7, "Pending transitions:");
        for (src, labels) in &self.pending_transitions {
            for (label, dsts) in labels {
                for (dst, weight) in dsts {
                    crate::debug!(7, "  {} --{}--> {} (weight: {:?})", src, label, dst, weight);
                }
            }
        }
        crate::debug!(7, "Pending epsilons:");
        for (src, dsts) in &self.pending_epsilons {
            for (dst, weight) in dsts {
                crate::debug!(7, "  {} --eps--> {} (weight: {:?})", src, dst, weight);
            }
        }
        
        // Debug: Count transitions
        let mut total_transitions = 0;
        let mut transitions_to_leaf = 0;
        for (src, labels) in &self.pending_transitions {
            for (label, dsts) in labels {
                for (dst, weight) in dsts {
                    total_transitions += 1;
                    if *dst == self.leaf_state {
                        transitions_to_leaf += 1;
                        // Check if token 6 and 31 are in the same weight
                        if weight.contains(6) && weight.contains(31) {
                            // Good - merged
                        } else if weight.contains(6) || weight.contains(31) {
                            // crate::debug!(7, "SEPARATE: transition from {} on label {} has weight with 6={} 31={}",
                            //     src, label, weight.contains(6), weight.contains(31));
                        }
                    }
                }
            }
        }
        // crate::debug!(5, "Pending transitions: {} total, {} to leaf", total_transitions, transitions_to_leaf);
        
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

        // Create start state with labeled tsid transitions for weight-heavy mode
        // Each tsid gets a transition labeled (tsid + terminals_count) with a tsid-masked weight
        // This preserves compatibility with build_parser_dwa which expects labeled tsid transitions
        let new_start_state = self.nwa.add_state();
        
        // Group tsids by their representative to call create_tsid_set_mask once per group
        let mut rep_to_tsids: BTreeMap<TokenizerStateID, Vec<usize>> = BTreeMap::new();
        for (tsid, rep_tsid) in &self.state_to_rep {
            rep_to_tsids.entry(*rep_tsid).or_default().push(tsid.0);
        }
        
        // Create one epsilon transition per representative with combined tsid mask
        for (rep_tsid, tsids) in rep_to_tsids {
            if let Some(&state) = self.roots.get(&rep_tsid) {
                // Create combined tsid mask for all tsids that map to this representative
                let tsid_mask = create_tsid_set_mask(tsids, self.num_tsids, self.internal_max_llm_token);
                self.nwa.add_epsilon(new_start_state, state, tsid_mask);
            }
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
            crate::debug!(6, "NWA: Found {} duplicate transitions (same src, dst, label)", duplicate_transitions);
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
            crate::debug!(5, "NWA: Found {} pairs of states connected by multiple transitions", parallel_connections);
        }

        crate::debug!(5, "{} states and {} transitions", self.nwa.states.len(), self.nwa.states.num_transitions());

        if std::env::var("DWA_DUMP_NWA").map(|v| v == "1").unwrap_or(false) {
            crate::debug!(5, "Dumping NWA to nwa_dump.json");
            let json = serde_json::to_string(&self.nwa).unwrap();
            std::fs::write("nwa_dump.json", json).unwrap();
        }
        
        // // OPTIMIZATION: Use lightweight operations instead of full minimize()
        // // This terminal DWA is only used as input to precompute4, so expensive minimization
        // // provides little benefit. Just do basic cleanup.
        // self.nwa.compress_transitions();
        // crate::debug!(5, "Compressed NWA with {} states and {} transitions", self.nwa.states.len(), self.nwa.states.num_transitions());
        //
        // // let dwa = self.nwa.determinize_and_minimize("Precompute1");
        // let mut dwa = self.nwa.determinize();
        // dwa.minimize();
        // crate::debug!(5, "Minimized DWA with {} states and {} transitions", dwa.states.len(), dwa.states.num_transitions());

        crate::debug!(5, "Starting RustFST-based minimization and determinization");
        self.nwa.minimize_with_rustfst_full();
        crate::debug!(5, "Minimized NWA with {} states and {} transitions", self.nwa.states.len(), self.nwa.states.num_transitions());
        self.nwa.compress_transitions();
        crate::debug!(5, "Compressed NWA with {} states and {} transitions", self.nwa.states.len(), self.nwa.states.num_transitions());
        // self.nwa.minimize();
        crate::debug!(5, "Minimized NWA with {} states and {} transitions", self.nwa.states.len(), self.nwa.states.num_transitions());
        let mut dwa = if std::env::var("DWA_USE_RUSTFST_DETERMINIZE").map(|v| v == "1").unwrap_or(false) {
            crate::debug!(5, "Using RustFST-based determinization");
            self.nwa.determinize_to_dwa_with_rustfst()
        } else {
            crate::debug!(5, "Using built-in determinization");
            self.nwa.determinize()
        };
        crate::debug!(5, "Determinized DWA with {} states and {} transitions", dwa.states.len(), dwa.states.num_transitions());
        
        // === Debug: compare weights before and after minimization ===
        let states_before_minimize = dwa.states.len();
        let unique_weights_before: std::collections::HashSet<_> = dwa.states.0.iter()
            .flat_map(|s| s.trans_weights.values().chain(s.final_weight.iter()))
            .cloned()
            .collect();
        crate::debug!(5, "Before minimize_with_rustfst_full: {} states, {} unique weights", 
                      states_before_minimize, unique_weights_before.len());
        
        dwa.minimize_with_rustfst_full();
        
        let unique_weights_after: std::collections::HashSet<_> = dwa.states.0.iter()
            .flat_map(|s| s.trans_weights.values().chain(s.final_weight.iter()))
            .cloned()
            .collect();
        crate::debug!(5, "After minimize_with_rustfst_full: {} states, {} unique weights", 
                      dwa.states.len(), unique_weights_after.len());
        
        crate::debug!(5, "Minimized DWA with {} states and {} transitions", dwa.states.len(), dwa.states.num_transitions());
        dwa.minimize();
        crate::debug!(5, "Final minimized DWA with {} states and {} transitions", dwa.states.len(), dwa.states.num_transitions());
        
        let mut dwa2 = self.nwa.determinize_to_dwa_with_rustfst();
        let unique_weights_dwa2: std::collections::HashSet<_> = dwa2.states.0.iter()
            .flat_map(|s| s.trans_weights.values().chain(s.final_weight.iter()))
            .cloned()
            .collect();
        crate::debug!(5, "RustFST DWA before minimize: {} states, {} unique weights", 
                      dwa2.states.len(), unique_weights_dwa2.len());
        dwa2.minimize_with_rustfst_full();
        let unique_weights_dwa2_after: std::collections::HashSet<_> = dwa2.states.0.iter()
            .flat_map(|s| s.trans_weights.values().chain(s.final_weight.iter()))
            .cloned()
            .collect();
        crate::debug!(5, "RustFST DWA after minimize: {} states, {} unique weights", 
                      dwa2.states.len(), unique_weights_dwa2_after.len());
        
        crate::debug!(5, "dwa is cyclic? {}", if dwa.is_cyclic() { "yes" } else { "no" });
        crate::debug!(5, "dwa2 is cyclic? {}", if dwa2.is_cyclic() { "yes" } else { "no" });
        stochastic_equivalence_test(dwa.clone(), dwa2);

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
                    RangeSet::from(applicable_tokens);
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
        _src_node: NWAStateID,
        tokenizer_state: TokenizerStateID,
        next_level_assoc: &mut BTreeMap<TokenizerStateID, NWAStateID>,
    ) -> NWAStateID {
        match next_level_assoc.entry(tokenizer_state) {
            std::collections::btree_map::Entry::Occupied(o) => *o.get(),
            std::collections::btree_map::Entry::Vacant(v) => {
                // NOTE: The previous state reuse optimization was removed because it
                // iterated through all pending_epsilons (~84 items on avg) but NEVER
                // found a reusable state (0 reuses in 500k+ calls, 42M+ loop iterations).
                // The check `live.is_disjoint(&Weight::all())` can only be true if the
                // live_tokens entry is empty, which almost never happens.
                let t = self.nwa.add_state();
                v.insert(t);
                t
            }
        }
    }

    /// Create an expanded weight from a single token ID.
    /// Expands from N-space to N×M-space where M = num_tsids.
    #[inline]
    fn expanded_weight_from_item(&self, token_id: usize) -> Weight {
        // A single token ID in N-space becomes a range in N×M-space
        // Token i becomes positions [i*M, i*M + M - 1]
        let start = token_id * self.num_tsids;
        let end = start + self.num_tsids - 1;
        // IMPORTANT: Use [start..=end] to create from ONE range, not iterate over all integers!
        Weight::from_rsb(RangeSetBlaze::from_iter([start..=end]))
    }

    /// Create an expanded weight from a RangeSetBlaze of token IDs.
    #[inline]
    fn expanded_weight_from_rsb(&self, rsb: RangeSetBlaze<usize>) -> Weight {
        Weight::from_rsb(expand_rsb(&rsb, self.num_tsids))
    }

    /// Create an expanded "all" weight (all tokens for all tsids).
    #[inline]
    fn expanded_weight_all(&self) -> Weight {
        // All tokens in N×M space
        let max_pos = self.internal_max_llm_token * self.num_tsids + self.num_tsids - 1;
        // IMPORTANT: Use [0..=max_pos] to create from ONE range, not iterate over all integers!
        Weight::from_rsb(RangeSetBlaze::from_iter([0..=max_pos]))
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
        crate::debug!(5, "Starting precompute DFS for {} tokenizer states", self.roots.len());
        profiler::reset();
        let vocab = std::mem::replace(&mut self.vocab, VocabPrefixTree::new());
        
        // Count vocab nodes for progress tracking
        let vocab_node_count = count_vocab_nodes(&vocab.root);
        crate::debug!(5, "Vocab tree has {} nodes", vocab_node_count);
        
        self.dfs(&vocab.root, assoc);
        self.vocab = vocab;
        self.pb.finish();
        profiler::print_summary();
        crate::debug!(5, "Precomputation complete");
    }

    fn dfs(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, NWAStateID>,
    ) {
        self.pb.inc(1);
        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            crate::debug!(7, "=== Processing vocab segment: {:?} (token_id={}) ===",
                String::from_utf8_lossy(segment_bytes), child_vocab_node.token_id());
            crate::debug!(7, "Initial assoc_by_state: {:?}", assoc_by_state);
            
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
                crate::debug!(7, "--- Position {} (segment len={}) ---", pos, segment_bytes.len());
                crate::debug!(7, "States at pos: {:?}", states_at_pos);
                
                // If we reached the end of the segment, these states are ready for the next vocab node
                if pos == segment_bytes.len() {
                    crate::debug!(7, "  -> End of segment, adding epsilons to next level");
                    for (tokenizer_state_id, node) in states_at_pos {
                        let next = self.get_or_create_next_state(node, tokenizer_state_id, &mut next_level_assoc);
                        crate::debug!(7, "     State {} (tsid={:?}) -> epsilon to state {}", node, tokenizer_state_id, next);
                        // Use expanded "all" weight
                        self.add_pending_epsilon(node, next, self.expanded_weight_all());
                    }
                    continue;
                }

                for (tokenizer_state_id, src_node) in states_at_pos {
                    let slice = &segment_bytes[pos..];
                    let exec_result = self
                        .tokenizer
                        .execute_from_state(slice, tokenizer_state_id);
                    
                    crate::debug!(7, "  Tokenizer on {:?} from state {:?} (src_node={}): matches={:?}, end_state={:?}",
                        String::from_utf8_lossy(slice), tokenizer_state_id, src_node, exec_result.matches, exec_result.end_state);

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
                        crate::debug!(7, "    Match: terminal_id={}, width={}, next_pos={}", terminal_id.0, match_info.width, next_pos);

                        // Leaf check: if match consumes remainder of segment
                        if next_pos == segment_bytes.len() {
                            let leaf = self.leaf_state;
                            // Use expanded weight from single token
                            let weight = self.expanded_weight_from_item(child_token_id);
                            crate::debug!(7, "      -> LEAF transition: {} --{}--> {} (leaf_state), weight={:?}", 
                                src_node, terminal_id.0, leaf, weight);
                            self.add_pending_transition(src_node, terminal_id.0 as Label, leaf, weight);
                        }

                        // Continuation logic
                        // Avoid cloning if we don't need to modify the bitset
                        let final_bv: std::borrow::Cow<RangeSetBlaze<usize>> = if next_pos == segment_bytes.len() {
                            let mut edge_bv = child_reachable.clone();
                            edge_bv.remove(child_token_id);
                            if let Some(pm) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv = &edge_bv - pm.inner.as_ref();
                            }
                            crate::debug!(7, "      Continuation at end of segment: edge_bv={:?} (removed child_token_id={}, pm={:?})",
                                edge_bv.iter().collect::<Vec<_>>(), child_token_id, possible_matches_at_end.get(&terminal_id).map(|pm| &pm.inner));
                            std::borrow::Cow::Owned(edge_bv)
                        } else {
                            crate::debug!(7, "      Continuation (not end): using child_reachable={:?}", child_reachable.iter().collect::<Vec<_>>());
                            std::borrow::Cow::Borrowed(child_reachable)
                        };

                        if final_bv.is_empty() {
                            crate::debug!(7, "      -> Skip continuation (empty edge_bv)");
                            continue;
                        }

                        let dest_map = pending.entry(next_pos).or_default();

                        let initial_tsid = self.tokenizer.initial_state_id();
                        // Use expanded weight from rsb
                        let weight = self.expanded_weight_from_rsb(final_bv.into_owned());

                        let target_entry = dest_map.entry(initial_tsid);
                        let target = match target_entry {
                            std::collections::btree_map::Entry::Occupied(o) => {
                                crate::debug!(7, "      -> Continuation to existing state: target={}", *o.get());
                                *o.get()
                            }
                            std::collections::btree_map::Entry::Vacant(v) => {
                                let t = self.nwa.add_state();
                                crate::debug!(7, "      -> Created new continuation state: target={}", t);
                                v.insert(t);
                                t
                            }
                        };

                        crate::debug!(7, "      -> CONT transition: {} --{}--> {}, weight={:?}", 
                            src_node, terminal_id.0, target, weight);
                        self.add_pending_transition(src_node, terminal_id.0 as Label, target, weight);
                    }

                    // 2. Handle End State -> Continuation
                    crate::debug!(7, "  End state handling: end_state={:?}", exec_result.end_state);
                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state = TokenizerStateID(end_state_val);
                        
                        // Use cached accessible terminals (389 unique states, but called 700k+ times)
                        let accessible_terminals: std::rc::Rc<Vec<GrammarTokenID>> = if let Some(cached) = self.accessible_terminals_cache.get(&final_tokenizer_state) {
                            cached.clone() // Rc clone is cheap
                        } else {
                            let result = std::rc::Rc::new(self.tokenizer.tokens_accessible_from_state(final_tokenizer_state)
                                .into_iter().collect::<Vec<_>>());
                            self.accessible_terminals_cache.insert(final_tokenizer_state, result.clone());
                            result
                        };
                        
                        crate::debug!(7, "    accessible_terminals={:?}", accessible_terminals.as_slice());

                        // Create expanded weight once, it's just a single token expanded to N×M space
                        let single_token_weight = self.expanded_weight_from_item(child_token_id);

                        let end_idx = self.leaf_state;
                        
                        for terminal_id in accessible_terminals.iter() {
                            crate::debug!(7, "    -> END_STATE transition: {} --{}--> {} (leaf_state), weight={:?}",
                                src_node, terminal_id.0, end_idx, single_token_weight);
                            self.add_pending_transition(
                                    src_node,
                                    terminal_id.0 as Label,
                                    end_idx,
                                    single_token_weight.clone(),
                                );
                        }

                        let next = self.get_or_create_next_state(src_node, final_tokenizer_state, &mut next_level_assoc);
                        crate::debug!(7, "    -> END_STATE epsilon: {} --eps--> {}", src_node, next);
                        // Use expanded "all" weight
                        self.add_pending_epsilon(src_node, next, self.expanded_weight_all());
                    }
                }
            }

            crate::debug!(7, "=== Done processing segment {:?}, next_level_assoc={:?} ===",
                String::from_utf8_lossy(segment_bytes), next_level_assoc);

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
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminals_count: usize,
    state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
) -> DWA {
    // Compute num_tsids from tokenizer
    let num_tsids = tokenizer.dfa.states.len();
    
    let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut seen_internal_ids = std::collections::HashSet::new();

    for (bytes, id) in internal_llm_token_map {
        if seen_internal_ids.insert(id.0) {
            representative_llm_token_map.insert(bytes.clone(), *id);
        }
    }

    let mut helper = Precomputer1::new(
        tokenizer,
        &representative_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        state_to_rep,
        num_tsids,
    );

    helper.run_dfs();
    helper.finish()
}
