use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use bimap::BiBTreeMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use crate::constraint::{PrecomputeNode0, PrecomputeNode0Index, PrecomputedNodeContents0, Trie0GodWrapper};
use crate::datastructures::gss::LLMTokenBV;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::glr::grammar::Terminal;
use crate::glr::parser::GLRParser;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::constraint::LLMVocab;
use crate::constraint_extra::{calculate_final_stats0, print_precompute_stats0, PrecomputeStats};
use crate::datastructures::ordered_hash_map::Retain;

const MERGE_THRESHOLD: usize = 20;

pub(crate) fn do_precompute0(
    tokenizer:        &Regex,
    parser:           Option<&GLRParser>,
    llm_vocab:        Option<Arc<LLMVocab>>,
    internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    token_name_map:   &BiBTreeMap<Terminal, usize>,
    internal_max_llm_token: usize,
    terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    _possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
) -> (BTreeMap<TokenizerStateID, PrecomputeNode0Index>, Trie0GodWrapper) {
    let mut helper = Precomputer0::new(
        tokenizer,
        parser,
        llm_vocab,
        internal_llm_token_map,
        internal_max_llm_token,
        MERGE_THRESHOLD,
        terminal_follow_map,
        ignore_terminal_id,
        token_name_map,
    );

    helper.run_dfs();
    helper.optimize();
    helper.finish()
}


struct Precomputer0<'r> {
    tokenizer:        &'r Regex,
    parser:           Option<&'r GLRParser>,
    llm_vocab:        Option<Arc<LLMVocab>>,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats:            PrecomputeStats,
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    token_name_map:   &'r BiBTreeMap<Terminal, usize>,
    // Map each precompute node to the set of LLM tokens that can pass through it.
    // tags:             RefCell<HashMap<PrecomputeNodeIndex, LLMTokenBV>>, // Removed
    // One end node per final tokenizer state.
    end_nodes:        BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
    trie0_god:        Trie0GodWrapper,
}

impl<'r> Precomputer0<'r> {
    fn new(
        tokenizer:        &'r Regex,
        parser:           Option<&'r GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        merge_threshold:  usize,
        terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        token_name_map: &'r BiBTreeMap<Terminal, usize>,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut roots = BTreeMap::new();
        let trie0_god = Trie0GodWrapper::new();
        for sid in tokenizer.iter_states() {
            roots.insert(
                sid,
                PrecomputeNode0Index::new(trie0_god.insert(PrecomputeNode0::new(PrecomputedNodeContents0::root(internal_max_llm_token)))),
            );
        }
        crate::debug!(2, "Created trie0 roots for {} tokenizer states", tokenizer.iter_states().count());

        crate::debug!(2, "Counting vocab nodes for progress bar...");
        let total_nodes = count_vocab_nodes(&vocab.root);
        crate::debug!(2, "Counted {} vocab nodes", total_nodes);
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

        let end_nodes = tokenizer.iter_states()
            .map(|tsid| (tsid, PrecomputeNode0Index::new(trie0_god.insert(PrecomputeNode0::new(PrecomputedNodeContents0::leaf(tsid))))))
            .collect();
        crate::debug!(2, "Created trie0 end nodes for {} tokenizer states", tokenizer.iter_states().count());

        Self {
            tokenizer,
            parser,
            llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            all_llm_tokens: HybridBitset::ones(internal_max_llm_token + 1),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
            terminal_follow_map,
            ignore_terminal_id,
            token_name_map,
            // tags: RefCell::new(HashMap::new()), // Removed
            end_nodes,
            trie0_god,
        }
    }

    fn optimize(&mut self) {
        crate::debug!(2, "Initial Trie0 stats:");
        let mut stats = PrecomputeStats::default();
        calculate_final_stats0(&self.roots, &mut stats, &self.trie0_god);
        print_precompute_stats0(&stats, self.token_name_map, &self.trie0_god);

        self.replace_ignore_token_edges_with_none_edges();
        self.simplify_none_edges(); // This can invalidate max_depth.

        // Recompute all max_depth values after major graph surgery.
        Trie::recompute_all_max_depths(&self.trie0_god, &self.roots.values().cloned().collect::<Vec<_>>());

        self.prune_dead_paths();
        self.prune_on_no_terminal_follow();
        self.prune_dead_paths();
        // New: prune using substring parser in "everything state" mode
        // self.prune_with_substring_everything_state();
        self.prune_dead_paths(); // Clean up after GLR-based pruning
        self.factor_common_destinations();
        self.merge_nodes();
        // self.merge_nodes_basic();
        self.gc();
        Trie::recompute_all_max_depths(&self.trie0_god, &self.roots.values().cloned().collect::<Vec<_>>());

        crate::debug!(2, "Final Trie0 stats:");
        let mut stats = PrecomputeStats::default();
        calculate_final_stats0(&self.roots, &mut stats, &self.trie0_god);
        print_precompute_stats0(&stats, self.token_name_map, &self.trie0_god);
    }

    fn get_end_node(&self, final_sid: TokenizerStateID) -> PrecomputeNode0Index {
        self.end_nodes[&final_sid].clone()
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
            OrderedHashSet<PrecomputeNode0Index>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(arc.clone());
        }

        crate::debug!(2, "Starting precompute DFS");
        crate::debug!(6, "Roots for each tokenizer state:");
        for (sid, root) in &self.roots {
            crate::debug!(6, "  {}: {}", sid.0, root);
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
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        // 2. Iterate over each node and modify its children map.
        for node_arc in all_nodes {
            let mut node_guard = node_arc.write(&self.trie0_god).expect("poison");
            let mut edges_to_move = Vec::new();

            for (key, dest_map) in node_guard.children() {
                if let Some((gtid, tokenizer_state_id_opt)) = key {
                    if *gtid == ignore_tid && tokenizer_state_id_opt.is_none() {
                        edges_to_move.push((key.clone(), dest_map.clone()));
                    }
                }
            }

            for (old_key, dest_map_to_move) in edges_to_move {
                node_guard.children_mut().remove(&old_key);
                let dest_map_for_new_key = node_guard.children_mut().entry(None).or_default();
                for (dest_wrapper, edge_bv) in dest_map_to_move {
                    // If an edge to this destination already exists under None, merge the bitvectors.
                    if let Some(existing_bv) = dest_map_for_new_key.get_mut(&dest_wrapper) {
                        *existing_bv |= &edge_bv;
                    } else {
                        dest_map_for_new_key.insert(dest_wrapper, edge_bv);
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

        let root_node_ptrs: HashSet<PrecomputeNode0Index> = self.roots.values().cloned().collect();

        // 1) Collect all unique nodes reachable from any root
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        // Map pointer -> Arc for quick retrieval
        let mut arc_by_ptr: HashMap<PrecomputeNode0Index, PrecomputeNode0Index> = HashMap::new();
        for n in &all_nodes {
            arc_by_ptr.insert(*n, n.clone());
        }

        // 2) Build:
        //    - incoming[B] = vec of (A, key_x, bv1) for edges A -(x; bv1)-> B
        //    - none_edges_from[B] = vec of (C, bv2) for edges B -(None; bv2)-> C
        //    - none_union[B] = union of all bv2 for None edges from B
        let mut incoming: HashMap<
            PrecomputeNode0Index,
            Vec<(PrecomputeNode0Index, Option<(GrammarTokenID, Option<TokenizerStateID>)>, LLMTokenBV)>
        > = HashMap::new();
        let mut none_edges_from: HashMap<
            PrecomputeNode0Index,
            Vec<(PrecomputeNode0Index, LLMTokenBV)>
        > = HashMap::new();
        let mut none_union: HashMap<PrecomputeNode0Index, LLMTokenBV> = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(&self.trie0_god).expect("poison");
            // Record all outgoing edges for incoming map
            for (ek, dest_map) in guard.children().iter() {
                for (child_wrap, ev_bv) in dest_map.iter() {
                    let child_arc = child_wrap.as_arc().clone();
                    let child_ptr = child_arc;
                    incoming.entry(child_ptr)
                        .or_default()
                        .push((src_arc.clone(), ek.clone(), ev_bv.clone()));
                }
            }
            // Record None edges out of src_arc (B -> C)
            for (ek, dest_map) in guard.children().iter() {
                if ek.is_none() {
                    let list = none_edges_from.entry(*src_ptr).or_default();
                    for (child_wrap, ev_bv) in dest_map.iter() {
                        list.push((child_wrap.as_arc().clone(), ev_bv.clone()));
                        let entry = none_union.entry(*src_ptr).or_insert_with(LLMTokenBV::zeros);
                        *entry |= ev_bv;
                    }
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
                        let mut b_guard = b_arc.write(&self.trie0_god).expect("poison");
                        b_guard.children_mut().retain(|k, _| k.is_some());
                    }
                    continue;
                }
            };

            let b_arc = match arc_by_ptr.get(&b_ptr) {
                Some(a) => a.clone(),
                None => continue,
            };
            let b_key = b_arc.clone();

            // For each incoming edge A -(x; bv1)-> B, split tokens:
            //   move:    to C with mask (bv1 ∩ bv2)
            //   leftover on A->B: bv1 - union_over_C(bv1 ∩ bv2) = bv1 ∩ (!union_mask)
            for (a_arc, edge_key, bv1_original) in in_edges.into_iter() {
                let mut total_to_move = bv1_original.clone();
                total_to_move &= &union_mask; // total tokens to redirect to all C via None edges
                if total_to_move.is_empty() {
                    continue;
                }

                let mut a_guard = a_arc.write(&self.trie0_god).expect("poison");
                let dest_map = a_guard.children_mut().entry(edge_key.clone()).or_default();

                // Add/merge edges to each C with per-child mask
                for (c_arc, bv2) in &none_edges {
                    let mut to_move_for_c = bv1_original.clone();
                    to_move_for_c &= bv2;
                    if to_move_for_c.is_empty() {
                        continue;
                    }
                    let c_key = c_arc.clone();
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
                let mut b_guard = b_arc.write(&self.trie0_god).expect("poison");
                b_guard.children_mut().retain(|k, _| k.is_some());
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

        type NodePtr = *const PrecomputeNode0; let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<(GrammarTokenID, Option<TokenizerStateID>)>>> = HashMap::new();

        Trie::special_map(
            &self.trie0_god,
            initial_nodes_and_values,
            |predecessors: &Option<BTreeSet<GrammarTokenID>>, edge_key: &Option<(GrammarTokenID, Option<TokenizerStateID>)>, _edge_bv, _child_node| {
                match edge_key {
                    Some((t, _)) if Some(*t) == ignore_terminal_id => Some(predecessors.clone()),
                    Some((t, _)) => Some(Some(BTreeSet::from([*t]))),
                    None => Some(predecessors.clone()),
                }
            },
            |existing_set, new_set| {
                match (existing_set, new_set) {
                    (None, _) => {},
                    (existing_set @ _, None) => *existing_set = None,
                    (Some(existing), Some(new)) => existing.extend(new),
                }
            },
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

                let keys_to_keep: BTreeSet<_> = node.children().keys().filter(|edge_key| {
                    match edge_key {
                        // Keep edges with terminals that are in the allowed follow set (or ignore edges).
                        Some((edge_terminal, _)) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
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
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        for node_arc in all_nodes {
            let node_ptr: NodePtr = {
                let guard = node_arc.read(&self.trie0_god).expect("poison");
                &*guard as *const _
            };
            if let Some(keys_to_keep) = edges_to_keep.get(&node_ptr) {
                let mut node_guard = node_arc.write(&self.trie0_god).unwrap();
                node_guard.children_mut().retain(|k, _| keys_to_keep.contains(k));
            }
        }

        crate::debug!(2, "Finished pruning based on terminal follow sets.");
    }

    fn prune_dead_paths(&mut self) {
        crate::debug!(2, "Pruning dead paths from precomputed trie.");

        // A cache of nodes to the set of "live" LLM tokens reachable from them.
        let mut live_tokens_cache: HashMap<PrecomputeNode0Index, LLMTokenBV> = HashMap::new();

        // For each root, run the pruning process. This will modify the trie in-place.
        // We do not remove the root from the map even if it becomes "dead" (has no live paths).
        // This ensures that every tokenizer state ID that started with a trie root still has one,
        // preventing panics in later stages that expect a complete map.
        for root_arc in self.roots.values() {
            let root_wrapper = root_arc.clone();
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
        node_wrapper: PrecomputeNode0Index,
        live_tokens_cache: &mut HashMap<PrecomputeNode0Index, LLMTokenBV>,
    ) -> LLMTokenBV {
        // If we've already computed the live tokens for this node, return the cached result.
        if let Some(cached_bv) = live_tokens_cache.get(&node_wrapper) {
            return cached_bv.clone();
        }
        // Insert a temporary empty BV to break cycles. If we revisit this node during this
        // recursion, it will return an empty set, which is correct as no new live paths
        // have been found through it yet.
        live_tokens_cache.insert(node_wrapper.clone(), LLMTokenBV::zeros());

        let node_arc = node_wrapper.as_arc().clone();

        // We must collect children before recursing to avoid holding the lock.
        let children_to_check: Vec<PrecomputeNode0Index> = {
            let node_guard = node_arc.read(&self.trie0_god).unwrap();
            node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
        };

        // Recursively call on all unique children to populate the cache for them.
        for child_wrapper in children_to_check {
            self.get_live_tokens_and_prune(child_wrapper, live_tokens_cache);
        }

        // Now that the cache is populated for all children, we can prune the current node.
        let mut live_tokens_for_this_node = LLMTokenBV::zeros();
        {
            let mut node_guard = node_arc.write(&self.trie0_god).unwrap();

            // A node is live if it's an end node itself. The tokens that end here are
            // on the edges pointing to this node.
            if node_guard.value.final_tokenizer_state.is_some() {
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
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (n, n.clone())).collect();

        // 2. Build an incoming edge map for every node.
        // incoming_map: D_ptr -> (gtid -> Vec<(S_ptr, bv)>)
        let mut incoming_map: HashMap<
            PrecomputeNode0Index, // Dst node ptr
            HashMap<
                Option<(GrammarTokenID, Option<TokenizerStateID>)>, // Full edge key
                Vec<(PrecomputeNode0Index, LLMTokenBV)>, // List of (Src node ptr, edge bv)
            >,
        > = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(&self.trie0_god).expect("poison");
            for (edge_key, dest_map) in guard.children() {
                if edge_key.is_some() { // Only consider non-None edges
                    for (dest_wrapper, bv) in dest_map {
                        let dest_arc = dest_wrapper.as_arc();
                        let dest_ptr = dest_arc;
                        incoming_map.entry(*dest_ptr).or_default().entry(edge_key.clone()).or_default().push((*src_ptr, bv.clone()));
                    }
                }
            }
        }

        // 3. Iterate through the map and find factoring opportunities.
        for (dest_ptr, edges_by_key) in incoming_map {
            for (edge_key, sources) in edges_by_key {
                if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                    // Opportunity found!
                    let dest_arc = arc_map.get(&dest_ptr).unwrap().clone();

                    // a. Create a new intermediate node `I`.
                    let intermediate_node = PrecomputeNode0Index::new(self.trie0_god.insert(PrecomputeNode0::new(PrecomputedNodeContents0::internal())));

                    // b. Add edge I --(edge_key)--> D
                    let mut union_bv = LLMTokenBV::zeros();
                    for (_, bv) in &sources {
                        union_bv |= bv;
                    }

                    {
                        let mut intermediate_guard = intermediate_node.write(&self.trie0_god).expect("poison");
                        let mut edge_val_opt = Some(union_bv.clone());
                        // No cycle possible since I is new. Use unchecked for speed.
                        // Depth will be propagated to D.
                        intermediate_guard.try_insert_unchecked(edge_key.clone(), &mut edge_val_opt, dest_arc.clone());
                        intermediate_guard.value.live_tokens |= &union_bv; // Update live_tokens for intermediate node
                    }

                    // c. For each source, remove old edge and add new `None` edge to `I`.
                    for (src_ptr, bv) in &sources {
                        let src_arc = arc_map.get(src_ptr).unwrap();
                        let mut src_guard = src_arc.write(&self.trie0_god).expect("poison");

                        // Remove S --(edge_key)--> D
                        if let Some(dest_map_for_key) = src_guard.children_mut().get_mut(&edge_key) {
                            dest_map_for_key.remove(&dest_arc.clone());
                            if dest_map_for_key.is_empty() {
                                src_guard.children_mut().remove(&edge_key);
                            }
                        }

                        // Add S --(None)--> I
                        let mut edge_val_opt = Some(bv.clone());
                        src_guard.try_insert_unchecked(None, &mut edge_val_opt, intermediate_node.clone());
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
        let mut canonical_nodes: HashMap<PrecomputeNode0, PrecomputeNode0Index> = HashMap::new();
        // A map from a node's pointer to its canonicalized Arc, to avoid re-processing.
        let mut visited: HashMap<PrecomputeNode0Index, PrecomputeNode0Index> = HashMap::new();

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
        node_arc: PrecomputeNode0Index,
        canonical_nodes: &mut HashMap<PrecomputeNode0, PrecomputeNode0Index>,
        visited: &mut HashMap<PrecomputeNode0Index, PrecomputeNode0Index>,
    ) -> PrecomputeNode0Index {
        let node_ptr = node_arc;
        if let Some(canonical_arc) = visited.get(&node_ptr) {
            return canonical_arc.clone();
        }

        // Pre-emptively insert to break cycles.
        visited.insert(node_ptr, node_arc.clone());

        // Post-order traversal: first, canonicalize all children.
        let mut new_children_map = BTreeMap::new();
        let mut children_changed = false;

        {
            let node_guard = node_arc.read(&self.trie0_god).unwrap();
        for (edge_key, dest_map) in node_guard.children() {
            let mut new_dest_map = OrderedHashMap::new();
            for (node_ptr_wrapper, edge_val) in dest_map.iter() {
                let child_arc = node_ptr_wrapper.as_arc().clone();
                let canonical_child_arc = self.deduplicate_recursive(child_arc.clone(), canonical_nodes, visited);
                if &child_arc != &canonical_child_arc {
                    children_changed = true;
                }
                let new_node_ptr_wrapper = canonical_child_arc;
                new_dest_map.insert(new_node_ptr_wrapper, edge_val.clone());
            }
            if !new_dest_map.is_empty() {
                new_children_map.insert(edge_key.clone(), new_dest_map);
                }
            }
        }

    if children_changed {
        let mut node_guard = node_arc.write(&self.trie0_god).unwrap();
        *node_guard.children_mut() = new_children_map;
        node_guard.recompute_max_depth(&self.trie0_god);
        // The live_tokens field will be recomputed by prune_dead_paths after merging.
    }

    let canonical_arc = {
            let node_guard = node_arc.read(&self.trie0_god).unwrap();
            let node_content = (*node_guard).clone();
            canonical_nodes.entry(node_content).or_insert_with(|| node_arc.clone()).clone()
        };

        // Update with the final canonical arc.
        visited.insert(node_ptr, canonical_arc.clone());
        canonical_arc
    }

    pub fn gc(&mut self) {
        crate::debug!(2, "Running garbage collection on precomputed trie.");
        let roots: Vec<_> = self.roots.values().cloned().collect();
        Trie::gc(&self.trie0_god, &roots);
    }

    fn finish(
        self,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode0Index>, Trie0GodWrapper) {
        (self.roots, self.trie0_god)
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNode0Index>>,
    ) {
        self.pb.inc(1);

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNode0Index>>,
            > = BTreeMap::new();
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

                        let mut disallowed_tokenizer_state_info = None;
                        // if let Some(end_state_val) = exec_result.end_state {
                        //     let end_tokenizer_state_id = TokenizerStateID(end_state_val);
                        //     let terminals_accessible = self.tokenizer.tokens_accessible_from_state(end_tokenizer_state_id);
                        //     if terminals_accessible.contains(&terminal_id) {
                        //         disallowed_tokenizer_state_info = Some(end_tokenizer_state_id);
                        //     }
                        // }

                        for src_node_wrapper in &precompute_nodes {
                            if next_pos == segment_bytes.len() {
                                // Exact end-of-segment terminal match: finishing LLM token here goes to tokenizer initial state.
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let edge_key = Some((terminal_id, disallowed_tokenizer_state_info));
                                let mut inserter = EdgeInserter::new(
                                    &self.trie0_god,
                                    src_node_wrapper.as_arc().clone(),
                                    edge_key,
                                    edge_bv,
                                    |e, n| *e |= n,
                                    |node_value, edge_value| {
                                        node_value.live_tokens |= edge_value;
                                    },
                                    |ev, t| *ev &= &t.live_tokens,
                                );
                                let end_idx = {
                                    let s0 = self.tokenizer.initial_state_id();
                                    self.get_end_node(s0)
                                };
                                inserter.try_destination(end_idx.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
                            }

                            let mut edge_bv = child_vocab_node.reachable_token_ids().clone();
                            if next_pos == segment_bytes.len() {
                                edge_bv.set(child_vocab_node.token_id(), false);
                            }
                            if let Some(matches_for_terminal) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv -= matches_for_terminal;
                            }

                            if edge_bv.is_empty() { continue; }

                            // NOTE: It is likely wrong to just use disallowed_tokenizer_state_info as-is here.
                            //  The actual disallowed state can vary by LLM token. But we don't capture that here.
                            //  We're doing it in a way that's local to the segment. This is wrong.
                            let edge_key = Some((terminal_id, None));
                            let mut inserter = EdgeInserter::new(
                                &self.trie0_god,
                                src_node_wrapper.as_arc().clone(),
                                edge_key,
                                edge_bv.clone(),
                                |e, n| *e |= n,
                                |node_value, edge_value| node_value.live_tokens |= edge_value,
                                |ev, t| *ev &= &t.live_tokens,
                            );

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos).or_default().entry(next_tokenizer_state).or_default();

                            inserter = inserter.try_destinations_iter(dest_nodes_in_queue.iter().map(|w| w.as_arc().clone()).filter(|w| w.read(&self.trie0_god).unwrap().value.final_tokenizer_state.is_none()));

                            let children_of_src: Vec<_> = src_node_wrapper.as_arc().read(&self.trie0_god).unwrap().children().values().flat_map(|m| m.keys().cloned()).collect();
                            let eligible_children = children_of_src.iter().map(|child_node_ptr| {
                                child_node_ptr.as_arc().clone()
                            }).filter(|child_arc| {
                                (child_arc.read(&self.trie0_god).unwrap().value.live_tokens.clone() & &edge_bv).is_empty() && child_arc.read(&self.trie0_god).unwrap().value.final_tokenizer_state.is_none()
                            });
                            inserter = inserter.try_destinations_iter(eligible_children);

                            let result_node = inserter.else_create_destination_with_value(PrecomputedNodeContents0::internal()).unwrap();
                            let result_node_ptr = result_node.clone();
                            dest_nodes_in_queue.insert(result_node_ptr.clone());
                        }
                    }

                    if let Some(end_state_val) = exec_result.end_state {
                        for src_node_wrapper in &precompute_nodes {
                            let llm_token_id = child_vocab_node.token_id();
                            let mut edge_bv = HybridBitset::zeros();
                            edge_bv.insert(llm_token_id);
                            let edge_key = None;
                            let mut inserter = EdgeInserter::new(
                                &self.trie0_god,
                                src_node_wrapper.as_arc().clone(),
                                edge_key,
                                edge_bv,
                                |e, n| *e |= n,
                                |node_value, edge_value| node_value.live_tokens |= edge_value,
                                |ev, t| *ev &= &t.live_tokens,
                            );
                            let end_idx = self.get_end_node(TokenizerStateID(end_state_val));
                            inserter.try_destination(end_idx.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
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
