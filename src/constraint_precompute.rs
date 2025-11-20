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

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Clear default start state

        let mut roots = BTreeMap::new();
        for sid in active_states {
            let root_state = nwa.add_state();
            roots.insert(sid, root_state);
        }
        crate::debug!(
            2,
            "Created trie1 roots for {} representative tokenizer states",
            roots.len()
        );

        crate::debug!(2, "Counting vocab nodes for progress bar...");
        let total_nodes = count_vocab_nodes(&vocab.root);
        crate::debug!(2, "Counted {} vocab nodes", total_nodes);
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
        crate::debug!(2, "Created trie1 leaf state");

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

    fn finish(self) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper)
    {
        let final_trie1_god = Trie1GodWrapper::new();
        let mut final_roots = BTreeMap::new();
        let mut node_map: HashMap<
            NWAStateID,
            PrecomputeNode1Index,
        > = HashMap::new();

        for (sid, temp_root) in &self.roots {
            let final_root = self.convert_nwa_to_trie(
                *temp_root,
                &final_trie1_god,
                &mut node_map,
            );
            final_roots.insert(*sid, final_root);
        }

        // Compute live tokens as union of all edge masks
        for node in Trie::all_nodes(&final_trie1_god, &final_roots.values().cloned().collect::<Vec<_>>()) {
            let mut lt = LLMTokenBV::zeros();
            let mut node = node.write(&final_trie1_god).unwrap();
            for (_, edges) in node.children() {
                for bv in edges.values() {
                    lt |= bv;
                }
            }
            node.value.live_tokens = lt;
        }

        (final_roots, final_trie1_god)
    }

    fn convert_nwa_to_trie(
        &self,
        state_id: NWAStateID,
        final_god: &Trie1GodWrapper,
        node_map: &mut HashMap<NWAStateID, PrecomputeNode1Index>,
    ) -> PrecomputeNode1Index {
        if let Some(final_idx) = node_map.get(&state_id) {
            return *final_idx;
        }

        let live = RangeSetBlaze::new();
        let is_end = self.nwa.states[state_id].final_weight.as_ref().map_or(false, |w| !w.is_empty());
        
        let final_node_contents = PrecomputedNodeContents {
            end: is_end,
            live_tokens: HybridBitset::from(live),
        };
        let new_node = PrecomputeNode1::new(final_node_contents);
        let final_idx = PrecomputeNode1Index::new(final_god.insert(new_node));
        node_map.insert(state_id, final_idx);

        // Group transitions by label
        let mut children_to_copy: BTreeMap<Option<GrammarTokenID>, Vec<(NWAStateID, RangeSetBlaze<usize>)>> = BTreeMap::new();
        for (label, targets) in &self.nwa.states[state_id].transitions {
            let grammar_token_id = GrammarTokenID(*label as usize);
            for (target, weight) in targets {
                // Convert SimpleBitset weight back to RangeSetBlaze
                let rsb = weight.rsb.clone();
                children_to_copy.entry(Some(grammar_token_id)).or_default().push((*target, rsb));
            }
        }

        for (target, weight) in &self.nwa.states[state_id].epsilons {
            let rsb = weight.rsb.clone();
            children_to_copy.entry(None).or_default().push((*target, rsb));
        }

        if self.original_to_dummy_map.is_empty() {
            for (ek, dest_map) in children_to_copy {
                for (child_state_id, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_nwa_to_trie(
                        child_state_id,
                        final_god,
                        node_map,
                    );
                    let hybrid_bitset = HybridBitset::from(rs_blaze);
                    final_god.insert_edge_simple(
                        final_idx,
                        final_child_idx,
                        ek.clone(),
                        hybrid_bitset,
                    );
                }
            }
        } else {
            let mut direct_edges = Vec::new();
            let mut injected_edges_by_dummy: BTreeMap<
                TerminalID,
                Vec<(
                    Option<TerminalID>,
                    OrderedHashMap<PrecomputeNode1Index, RangeSetBlaze<usize>>,
                )>,
            > = BTreeMap::new();

            for (ek, dest_map) in children_to_copy {
                if let Some(tid) = ek {
                    if let Some(dummy_tid) =
                        self.original_to_dummy_map.get(&tid)
                    {
                        injected_edges_by_dummy
                            .entry(*dummy_tid)
                            .or_default()
                            .push((Some(tid), dest_map.into_iter().map(|(s, w)| (Trie2Index::from(s), w)).collect()));
                        continue;
                    }
                }
                direct_edges.push((ek, dest_map));
            }

            for (ek, dest_map) in direct_edges {
                for (child_state_id, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_nwa_to_trie(
                        child_state_id,
                        final_god,
                        node_map,
                    );
                    let hybrid_bitset = HybridBitset::from(rs_blaze);
                    final_god.insert_edge_simple(
                        final_idx,
                        final_child_idx,
                        ek.clone(),
                        hybrid_bitset,
                    );
                }
            }

            for (dummy_tid, edges) in injected_edges_by_dummy {
                let inter_node =
                    PrecomputeNode1::new(PrecomputedNodeContents::internal());
                let inter_idx =
                    PrecomputeNode1Index::new(final_god.insert(inter_node));
                let mut total_inter_bitset = HybridBitset::zeros();

                for (original_ek, dest_map) in edges {
                    for (child_state_id, rs_blaze) in dest_map {
                        let final_child_idx = self.convert_nwa_to_trie(
                            child_state_id.as_usize(),
                            final_god,
                            node_map,
                        );
                        let hybrid_bitset = HybridBitset::from(rs_blaze);
                        total_inter_bitset |= &hybrid_bitset;
                        final_god.insert_edge_simple(
                            inter_idx,
                            final_child_idx,
                            original_ek,
                            hybrid_bitset,
                        );
                    }
                }
                final_god.insert_edge_simple(
                    final_idx,
                    inter_idx,
                    Some(dummy_tid),
                    total_inter_bitset,
                );
            }
        }

        final_idx
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
        let mut assoc: BTreeMap<
            TokenizerStateID,
            Vec<NWAStateID>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .push(*arc);
        }

        crate::debug!(2, "Starting precompute DFS for {} tokenizer states", self.roots.len());
        crate::debug!(6, "Roots for each tokenizer state:");
        for (sid, root) in &self.roots {
            crate::debug!(6, "  {}: {}", sid.0, root);
        }
        profiler::reset();
        let vocab = std::mem::replace(&mut self.vocab, VocabPrefixTree::new());
        self.dfs(&vocab.root, assoc);
        self.vocab = vocab;
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish();
        profiler::print_summary();
        crate::debug!(2, "Precomputation complete");
    }

    fn dfs(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<
            TokenizerStateID,
            Vec<NWAStateID>,
        >,
    ) {
        self.pb.inc(1);
        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut next_level_assoc: BTreeMap<
                TokenizerStateID,
                Vec<NWAStateID>,
            > = BTreeMap::new();

            // Queue: pos -> TokenizerState -> (NWAState -> ContextTokens)
            let mut pending: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, Vec<NWAStateID>>,
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

            while let Some((pos, mut states_at_pos)) = pending.pop_first() {
                for (_, nodes) in states_at_pos.iter_mut() {
                    let merged_state = self.nwa.add_state();
                    let weight = SimpleBitset::all();;
                    for node in &*nodes {
                        self.nwa.add_epsilon(*node, merged_state, weight.clone());
                    }
                    nodes.clear();
                    nodes.push(merged_state);
                }

                // If we reached the end of the segment, these states are ready for the next vocab node
                if pos == segment_bytes.len() {
                    for (tokenizer_state_id, nodes) in states_at_pos {
                        let entry = next_level_assoc.entry(tokenizer_state_id).or_default();
                        entry.extend(nodes);
                    }
                    continue;
                }

                for (tokenizer_state_id, nodes) in states_at_pos {
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

                        for src_node in &nodes {
                            // Leaf check: if match consumes remainder of segment
                            if next_pos == segment_bytes.len() {
                                let mut edge_bv = RangeSetBlaze::new();
                                edge_bv.insert(child_token_id);
                                let final_bv = edge_bv;
                                if !final_bv.is_empty() {
                                    let leaf = self.get_leaf_node();
                                    pending_edges.push((
                                        *src_node,
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

                            let dest_map = pending
                                .entry(next_pos)
                                .or_default()
                                .entry(self.tokenizer.initial_state_id())
                                .or_default();

                            let target = {
                                let n = self.nwa.add_state();
                                n
                            };

                            dest_map.push(target);

                            pending_edges.push((
                                *src_node,
                                target,
                                Some(terminal_id),
                                final_bv,
                            ));
                        }
                    }

                    // 2. Handle End State -> Continuation
                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state = TokenizerStateID(end_state_val);
                        let accessible_terminals = self
                            .tokenizer
                            .tokens_accessible_from_state(final_tokenizer_state);

                        for src_node in &nodes {
                            let mut edge_bv = RangeSetBlaze::new();
                            edge_bv.insert(child_token_id);
                            let final_edge_bv = edge_bv;

                            if !final_edge_bv.is_empty() {
                                let end_idx = self.get_leaf_node();
                                for terminal_id in &accessible_terminals {
                                    pending_edges.push((
                                        *src_node,
                                        end_idx,
                                        Some(*terminal_id),
                                        final_edge_bv.clone(),
                                    ));
                                }
                            }

                            next_level_assoc
                                .entry(final_tokenizer_state)
                                .or_default()
                                .push(*src_node);
                        }
                    }
                }
            }

            // Apply all batched writes
            for (src, dst, key, bv) in pending_edges {
                if let Some(k) = key {
                    let weight = SimpleBitset::from_rsb(bv);
                    let _ = self.nwa.add_transition(src, k.0 as i16, dst, weight);
                }
            }

            for nodes in next_level_assoc.values_mut() {
                nodes.sort_unstable();
                nodes.dedup();
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

pub(crate) fn has_llm_compatible_cycle(
    arena: &Trie1GodWrapper,
    roots: &[PrecomputeNode1Index],
    internal_max_llm_token: usize,
) {
    let mut visited: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();
    let initial_tokens = LLMTokenBV::ones(internal_max_llm_token + 1);

    for &root in roots {
        if let Some((cycle_path, llm_token_id)) = detect_cycle_recursive(
            root,
            None,
            initial_tokens.clone(),
            arena,
            &mut HashMap::new(),
            &mut visited,
            &mut Vec::new(),
        ) {
            let mut report = format!(
                "LLM-compatible cycle detected in precompute1 trie for internal LLM token ID \
                 {}.\nCycle path:\n",
                llm_token_id.0
            );
            for i in 0..cycle_path.len() {
                let (node_idx, _) = cycle_path[i];
                let next_i = (i + 1) % cycle_path.len();
                let (next_node_idx, edge_to_next_opt) = &cycle_path[next_i];
                let edge_str = edge_to_next_opt.as_ref().map_or_else(
                    || " (root edge)".to_string(),
                    |ek| format!("{:?}", ek),
                );
                report.push_str(&format!(
                    "  {} --[{}]--> {}\n",
                    node_idx, edge_str, next_node_idx
                ));
            }
            panic!("{}", report);
        }
    }
}

pub(crate) fn detect_cycle_recursive(
    node_idx: PrecomputeNode1Index,
    edge_key_opt: Option<Option<GrammarTokenID>>,
    current_tokens: LLMTokenBV,
    arena: &Trie1GodWrapper,
    recursion_stack: &mut HashMap<PrecomputeNode1Index, (LLMTokenBV, usize)>,
    visited: &mut HashMap<PrecomputeNode1Index, LLMTokenBV>,
    path: &mut Vec<(PrecomputeNode1Index, Option<Option<GrammarTokenID>>)>,
) -> Option<(Vec<(PrecomputeNode1Index, Option<Option<GrammarTokenID>>)>, LLMTokenID)>
{
    path.push((node_idx, edge_key_opt));

    if let Some((tokens_on_stack, path_start_idx)) = recursion_stack.get(&node_idx) {
        let intersection = &current_tokens & tokens_on_stack;
        if !intersection.is_empty() {
            let cycle_llm_token = intersection.iter_up_to(usize::MAX).next().unwrap();
            let cycle_path = path[*path_start_idx..].to_vec();
            path.pop();
            return Some((cycle_path, LLMTokenID(cycle_llm_token)));
        }
    }

    let new_tokens_to_process = match visited.entry(node_idx) {
        std::collections::hash_map::Entry::Occupied(mut entry) => {
            let previously_visited_tokens = entry.get_mut();
            let new_unseen_tokens = &current_tokens - &*previously_visited_tokens;
            if new_unseen_tokens.is_empty() {
                path.pop();
                return None;
            }
            *previously_visited_tokens |= &current_tokens;
            new_unseen_tokens
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(current_tokens.clone());
            current_tokens.clone()
        }
    };

    recursion_stack.insert(node_idx, (current_tokens, path.len() - 1));

    let children_to_visit = if let Some(guard) = node_idx.read(arena) {
        guard.children().clone()
    } else {
        recursion_stack.remove(&node_idx);
        path.pop();
        return None;
    };

    for (edge_key, dest_map) in children_to_visit.iter() {
        for (child_idx, edge_tokens) in dest_map.iter() {
            let next_tokens = &new_tokens_to_process & edge_tokens;
            if !next_tokens.is_empty() {
                if let Some(report) = detect_cycle_recursive(
                    *child_idx,
                    Some(edge_key.clone()),
                    next_tokens,
                    arena,
                    recursion_stack,
                    visited,
                    path,
                ) {
                    return Some(report);
                }
            }
        }
    }

    recursion_stack.remove(&node_idx);
    path.pop();
    None
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
    let roots_after: Vec<_> = precomputed1.values().cloned().collect();

    has_llm_compatible_cycle(
        &trie1_god,
        &roots_after,
        stage_vocab.internal_max_llm_token,
    );

    let mut stats = PrecomputeStats::default();
    crate::constraint_extra::calculate_final_stats1(
        &precomputed1,
        &mut stats,
        &trie1_god,
    );
    crate::constraint_extra::print_precompute_stats1(
        &stats,
        token_name_map,
        &trie1_god,
    );

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
