use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use ordered_hash_map::OrderedHashMap;
use crate::constraint::{PrecomputeNode0, PrecomputeNode0Index, PrecomputedNodeContents0, Precomputer0, Trie0GodWrapper};
use crate::constraint::LLMTokenBV;
use crate::datastructures::trie::Trie;
use crate::tokenizer::TokenizerStateID;
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::constraint_extra::{calculate_final_stats0, print_precompute_stats0, PrecomputeStats};
use crate::datastructures::ordered_hash_map::Retain;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use crate::profiler::PROGRESS_BAR_ENABLED;
use kdam::tqdm;

#[derive(Debug, Clone)]
pub struct Trie0Config {
    pub enabled: bool,
    pub simplify_none_edges: bool,
    pub prune_dead_paths: bool,
    pub prune_on_no_terminal_follow: bool,
    pub merge_nodes: bool,
    pub gc: bool,
    pub factor_common_destinations: bool,
}

impl Default for Trie0Config {
    fn default() -> Self {
        Self {
            enabled: true,
            simplify_none_edges: false, // was commented out, seems risky
            prune_dead_paths: true,
            prune_on_no_terminal_follow: true,
            merge_nodes: true,
            gc: true,
            factor_common_destinations: false, // was commented out
        }
    }
}

impl Trie0Config {
    pub fn off() -> Self {
        Self {
            enabled: false,
            simplify_none_edges: false,
            prune_dead_paths: false,
            prune_on_no_terminal_follow: false,
            merge_nodes: false,
            gc: false,
            factor_common_destinations: false,
        }
    }
}

impl<'r> Precomputer0<'r> {
    pub(crate) fn optimize(&mut self, config: &Trie0Config) {
        crate::debug!(2, "Initial Trie0 stats:");
        let mut stats = PrecomputeStats::default();
        calculate_final_stats0(&self.roots, &mut stats, &self.trie0_god);
        print_precompute_stats0(&stats, self.token_name_map, &self.trie0_god);

        self.replace_ignore_token_edges_with_none_edges();
        if config.simplify_none_edges {
            self.simplify_none_edges(); // This can invalidate max_depth.
        }

        // Recompute all max_depth values after major graph surgery.
        Trie::recompute_all_max_depths(&self.trie0_god, &self.roots.values().cloned().collect::<Vec<_>>());

        if config.prune_dead_paths { self.prune_dead_paths(); }
        if config.prune_on_no_terminal_follow { self.prune_on_no_terminal_follow(); }
        if config.prune_dead_paths { self.prune_dead_paths(); }

        if config.factor_common_destinations {
            self.factor_common_destinations();
        }
        if config.merge_nodes {
            self.merge_nodes();
        }

        self.break_structural_cycles();
        self.assert_no_cycles();
        if config.prune_dead_paths { self.prune_dead_paths(); }

        if config.gc {
            self.gc();
        }
        Trie::recompute_all_max_depths(&self.trie0_god, &self.roots.values().cloned().collect::<Vec<_>>());

        crate::debug!(2, "Final Trie0 stats:");
        let mut stats = PrecomputeStats::default();
        calculate_final_stats0(&self.roots, &mut stats, &self.trie0_god);
        print_precompute_stats0(&stats, self.token_name_map, &self.trie0_god);
    }

    pub(crate) fn break_structural_cycles(&mut self) {
        crate::debug!(2, "Breaking structural cycles...");
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .expect("progress bar style"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }
        pb.set_message("Breaking structural cycles...");

        let mut clones: HashMap<PrecomputeNode0Index, PrecomputeNode0Index> = HashMap::new();

        for i in 0.. {
            let back_edges = self.find_back_edges();
            if back_edges.is_empty() {
                pb.finish_with_message(format!("Broke structural cycles after {} iterations.", i));
                crate::debug!(2, "No more structural cycles found after {} iterations.", i);
                break;
            }
            if i > 100 { // Sanity check
                panic!("Cycle breaking seems to be in an infinite loop.");
            }
            pb.set_message(format!("Breaking structural cycles (iter {}, {} back-edges)...", i, back_edges.len()));
            crate::debug!(3, "Found {} back edges in iteration {}.", back_edges.len(), i);

            let mut changes = Vec::new();
            for (src_idx, key, old_dest_idx) in back_edges {
                let new_dest_idx = clones.entry(old_dest_idx.clone()).or_insert_with(|| {
                    // When breaking a cycle, we clone the destination node but without its children.
                    // This turns the back-edge into an edge to a new "leaf" node (in the context
                    // of the cycle), effectively breaking the cycle instead of just displacing it.
                    // A full clone would copy the children, which would just recreate the cycle
                    // with the cloned node, leading to an infinite loop.
                    let new_node_content = PrecomputeNode0::new(old_dest_idx.read(&self.trie0_god).unwrap().value.clone());
                    let new_node_arc = self.trie0_god.insert(new_node_content);
                    PrecomputeNode0Index::new(new_node_arc)
                }).clone();
                changes.push((src_idx, key, old_dest_idx, new_dest_idx));
            }

            for (src_idx, key, old_dest_idx, new_dest_idx) in changes {
                let mut src_guard = src_idx.write(&self.trie0_god).unwrap();
                if let Some(dest_map) = src_guard.children_mut().get_mut(&key) {
                    if let Some(bv) = dest_map.remove(&old_dest_idx) {
                        dest_map.insert(new_dest_idx, bv);
                    }
                }
            }
        }
        crate::debug!(2, "Finished breaking structural cycles.");
    }

    fn find_back_edges(&self) -> Vec<(PrecomputeNode0Index, Option<(GrammarTokenID, Option<TokenizerStateID>)>, PrecomputeNode0Index)> {
        let mut back_edges = Vec::new();
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();
        let roots: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots);

        for node in all_nodes {
            // The visited check is important here to avoid re-starting DFS from nodes
            // that have already been fully explored as part of another node's traversal.
            if !visited.contains(&node) {
                self.find_back_edges_dfs(node, &mut visited, &mut recursion_stack, &mut back_edges);
            }
        }

        back_edges
    }

    fn find_back_edges_dfs(
        &self,
        node_idx: PrecomputeNode0Index,
        visited: &mut HashSet<PrecomputeNode0Index>,
        recursion_stack: &mut HashSet<PrecomputeNode0Index>,
        back_edges: &mut Vec<(PrecomputeNode0Index, Option<(GrammarTokenID, Option<TokenizerStateID>)>, PrecomputeNode0Index)>,
    ) {
        if visited.contains(&node_idx) {
            return;
        }

        recursion_stack.insert(node_idx.clone());

        let children_to_visit = node_idx.read(&self.trie0_god).unwrap().children().clone();

        for (edge_key, dest_map) in children_to_visit.iter() {
            for (child_idx, _edge_val) in dest_map.iter() {
                if recursion_stack.contains(child_idx) {
                    back_edges.push((node_idx.clone(), edge_key.clone(), child_idx.clone()));
                } else {
                    self.find_back_edges_dfs(child_idx.clone(), visited, recursion_stack, back_edges);
                }
            }
        }

        recursion_stack.remove(&node_idx);
        visited.insert(node_idx.clone());
    }

    pub(crate) fn assert_no_cycles(&self) {
        let roots: Vec<_> = self.roots.values().cloned().collect();
        let has_cycle = Trie::has_cycle(&self.trie0_god, roots);
        if has_cycle {
            panic!("Structural cycles detected after attempting to break them.");
        }
        crate::debug!(2, "Assertion passed: no structural cycles found.");
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

        type NodePtr = *const PrecomputeNode0;
        let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<(GrammarTokenID, Option<TokenizerStateID>)>>> = HashMap::new();

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
        crate::debug!(2, "Pruning dead paths from precomputed trie (fixpoint).");

        // 1) Gather all nodes reachable from roots.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);

        // 2) Snapshot graph topology (outgoing and incoming) and final flags without holding locks across the whole pass.
        //    outgoing[src] = vec of (edge_key, dst, bv)
        //    incoming[dst] = vec of (src, bv)  [edge_key is irrelevant to liveness math]
        let mut is_final: HashMap<PrecomputeNode0Index, bool> = HashMap::new();
        let mut outgoing: HashMap<
            PrecomputeNode0Index,
            Vec<(Option<(GrammarTokenID, Option<TokenizerStateID>)>, PrecomputeNode0Index, LLMTokenBV)>
        > = HashMap::new();
        let mut incoming: HashMap<
            PrecomputeNode0Index,
            Vec<(PrecomputeNode0Index, LLMTokenBV)>
        > = HashMap::new();

        for src in &all_nodes {
            let guard = src.read(&self.trie0_god).unwrap();
            is_final.insert(*src, guard.value.final_tokenizer_state.is_some());
            for (ek, dest_map) in guard.children().iter() {
                for (dst_wrap, bv) in dest_map.iter() {
                    let dst = dst_wrap.as_arc().clone();
                    outgoing.entry(*src)
                        .or_default()
                        .push((ek.clone(), dst.clone(), bv.clone()));
                    incoming.entry(dst)
                        .or_default()
                        .push((*src, bv.clone()));
                }
            }
        }

        // 3) Backward liveness fixpoint:
        //    L[n] = Universe if final(n) else 0
        //    L[p] |= (bv(p->c) ∧ L[c])
        let mut live: HashMap<PrecomputeNode0Index, LLMTokenBV> = HashMap::new();
        let mut q: VecDeque<PrecomputeNode0Index> = VecDeque::new();
        for n in &all_nodes {
            if *is_final.get(n).unwrap_or(&false) {
                live.insert(*n, self.all_llm_tokens.clone());
                q.push_back(*n);
            } else {
                live.insert(*n, LLMTokenBV::zeros());
            }
        }

        while let Some(dst) = q.pop_front() {
            let l_dst = live.get(&dst).cloned().unwrap_or_else(LLMTokenBV::zeros);
            if let Some(preds) = incoming.get(&dst) {
                for (src, edge_bv) in preds {
                    // contribution = edge_bv ∧ L[dst]
                    let contribution = &*edge_bv & &l_dst;
                    if contribution.is_empty() {
                        continue;
                    }
                    let entry = live.get_mut(src).unwrap();
                    let before = entry.clone();
                    *entry |= &contribution;
                    if *entry != before {
                        q.push_back(*src);
                    }
                }
            }
        }

        // 4) Apply pruning and update per-node live_tokens.
        for src in &all_nodes {
            // Rebuild children: keep only edges with (bv ∧ L[dst]) != ∅
            let mut new_children: BTreeMap<
                Option<(GrammarTokenID, Option<TokenizerStateID>)>,
                OrderedHashMap<PrecomputeNode0Index, LLMTokenBV>
            > = BTreeMap::new();
            let mut live_for_src = if *is_final.get(src).unwrap_or(&false) {
                self.all_llm_tokens.clone()
            } else {
                LLMTokenBV::zeros()
            };

            if let Some(outs) = outgoing.get(src) {
                for (ek, dst, bv) in outs {
                    let l_dst = live.get(dst).cloned().unwrap_or_else(LLMTokenBV::zeros);
                    let new_bv = &*bv & &l_dst;
                    if new_bv.is_empty() {
                        continue;
                    }
                    live_for_src |= &new_bv;
                    let dest_map = new_children.entry(ek.clone()).or_default();
                    if let Some(existing) = dest_map.get_mut(dst) {
                        *existing |= &new_bv;
                    } else {
                        dest_map.insert(dst.clone(), new_bv);
                    }
                }
            }

            let mut guard = src.write(&self.trie0_god).unwrap();
            *guard.children_mut() = new_children;
            guard.value.live_tokens = live_for_src;
        }

        crate::debug!(2, "Finished pruning dead paths (fixpoint).");
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
        #[cfg(not(rustrover))]
        let it = tqdm!(self.roots.iter(), desc="Merging subtrees", unit="root");
        #[cfg(rustrover)]
        let it = self.roots.iter();
        for (sid, root_arc) in it {
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
        // By collecting children first, we avoid holding a lock on `node_arc` during recursion,
        // which prevents deadlocks.
        let children_to_process = {
            node_arc.read(&self.trie0_god).unwrap().children().clone()
        };

        let mut new_children_map = BTreeMap::new();
        let mut children_changed = false;

        for (edge_key, dest_map) in &children_to_process {
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

        if children_changed {
            // Update children under a short write lock. Do NOT recompute max_depth here.
            // Calling recompute_max_depth while holding a write lock can deadlock on
            // self-loops, because it may attempt to acquire a read lock on the same node.
            // We recompute max depths globally after merging (see optimize()), which is safe.
            let mut node_guard = node_arc.write(&self.trie0_god).unwrap();
            *node_guard.children_mut() = new_children_map;
            // IMPORTANT: No recompute_max_depth here to avoid deadlocks on self-loops.
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
}