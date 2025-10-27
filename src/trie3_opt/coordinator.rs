use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use ordered_hash_map::OrderedHashMap;

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::metrics::run_all_metrics;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::{
    CanonicalizeEndNodesPass, CompressEdgesPass, EliminatePop0ExceptRootsPass, MergeStructuralPass,
    OptimizationPass, PruneDeadPathsPass,
};

use crate::constraint::{
    PrecomputeNode3Index, PrecomputedNodeContents, Trie3GodWrapper, LLMTokenBV, StateIDBV,
};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::Trie;
use crate::tokenizer::TokenizerStateID;

/// Configuration for the coordinator: which passes to run and their key parameters.
#[derive(Clone, Debug)]
pub struct CoordinatorConfig {
    pub enable_prune_dead_paths: bool,
    pub enable_compress_edges: bool,
    pub enable_merge_structural: bool,
    pub merge_structural_max_iters: usize,
    pub enable_eliminate_pop0_except_roots: bool,
    pub enable_canonicalize_end_nodes: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            enable_prune_dead_paths: true,
            enable_compress_edges: true,
            enable_merge_structural: true,
            merge_structural_max_iters: 4,
            enable_eliminate_pop0_except_roots: true,
            enable_canonicalize_end_nodes: true,
        }
    }
}

impl CoordinatorConfig {
    pub fn off() -> Self {
        Self {
            enable_prune_dead_paths: false,
            enable_compress_edges: false,
            enable_merge_structural: false,
            merge_structural_max_iters: 0,
            enable_eliminate_pop0_except_roots: false,
            enable_canonicalize_end_nodes: false,
        }
    }
}

/// Build a default sequence of passes from config.
fn build_pipeline(config: &CoordinatorConfig) -> Vec<Box<dyn OptimizationPass>> {
    let mut pipeline: Vec<Box<dyn OptimizationPass>> = Vec::new();

    if config.enable_prune_dead_paths {
        pipeline.push(Box::new(PruneDeadPathsPass));
    }
    if config.enable_canonicalize_end_nodes {
        pipeline.push(Box::new(CanonicalizeEndNodesPass));
    }
    if config.enable_compress_edges {
        pipeline.push(Box::new(CompressEdgesPass));
    }
    if config.enable_merge_structural {
        pipeline.push(Box::new(MergeStructuralPass::new(
            config.merge_structural_max_iters,
        )));
    }
    if config.enable_eliminate_pop0_except_roots {
        pipeline.push(Box::new(EliminatePop0ExceptRootsPass));
    }
    // Final compression after potential rewires
    if config.enable_compress_edges {
        pipeline.push(Box::new(CompressEdgesPass));
    }

    pipeline
}

/// Convert the given precompute3 trie into a MiniTrie along with a stable list of root keys.
pub(crate) fn export_to_mini(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
) -> (MiniTrie, Vec<(TokenizerStateID, NodeId)>, HashMap<PrecomputeNode3Index, NodeId>) {
    let mut mini = MiniTrie::new();

    // BFS from roots to collect reachable nodes and assign NodeId
    let mut map_old_to_new: HashMap<PrecomputeNode3Index, NodeId> = HashMap::new();
    let mut q: VecDeque<PrecomputeNode3Index> = VecDeque::new();

    for (_k, r) in roots {
        if !map_old_to_new.contains_key(r) {
            let end = r.read(trie3_god).map(|g| g.value.end).unwrap_or(false);
            let id = mini.add_node(end);
            map_old_to_new.insert(*r, id);
            q.push_back(*r);
            mini.add_root(id);
        }
    }

    while let Some(old_idx) = q.pop_front() {
        let new_id = *map_old_to_new.get(&old_idx).unwrap();
        let r = if let Some(g) = old_idx.read(trie3_god) { g } else { continue };

        for (ek, dm) in r.children() {
            // Build token set
            let mut toks = SortedSet::new();
            if ek.1.is_all() {
                toks = SortedSet::from_iter(0..=max_llm_token_id);
            } else {
                for t in ek.1.iter() {
                    toks.insert(t);
                }
            }
            if toks.is_empty() {
                continue;
            }
            let key = crate::trie3_opt::core::EdgeKey::new(ek.0, toks.clone());
            for (dst, sids) in dm {
                // Ensure dst is mapped
                let dst_id = if let Some(id) = map_old_to_new.get(dst) {
                    *id
                } else {
                    let end = dst.read(trie3_god).map(|g| g.value.end).unwrap_or(false);
                    let id = mini.add_node(end);
                    map_old_to_new.insert(*dst, id);
                    q.push_back(*dst);
                    id
                };
                // Build state set
                let mut st = SortedSet::new();
                if sids.is_all() {
                    st = SortedSet::from_iter(0..=max_state_id);
                } else {
                    for s in sids.iter() {
                        st.insert(s);
                    }
                }
                if !st.is_empty() {
                    mini.add_edge(new_id, key.clone(), dst_id, st);
                }
            }
        }
    }

    // Stable root list preserving the original key ordering
    let mut root_pairs: Vec<(TokenizerStateID, NodeId)> = Vec::new();
    for (k, r) in roots {
        if let Some(id) = map_old_to_new.get(r).cloned() {
            root_pairs.push((*k, id));
        }
    }

    (mini, root_pairs, map_old_to_new)
}

/// Import a MiniTrie back into the precompute3 arena by constructing a fresh graph.
/// - All new nodes are inserted; roots remapped to these.
/// - Old nodes become unreachable and are GC'd.
/// - live_tokens are recomputed from outgoing edges.
/// - end flags are set from the mini-trie.
fn import_from_mini(
    mini: &MiniTrie,
    root_pairs: &[(TokenizerStateID, NodeId)],
    trie3_god: &Trie3GodWrapper,
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    ctx: &OptimizationContext,
) {
    // Create all nodes first
    let mut new_nodes: Vec<PrecomputeNode3Index> = Vec::with_capacity(mini.nodes.len());
    for _ in 0..mini.nodes.len() {
        let idx = PrecomputeNode3Index::new(trie3_god.insert(Trie::new(PrecomputedNodeContents::internal())));
        new_nodes.push(idx);
    }

    // Fill end flags and edges
    for node in &mini.nodes {
        let idx = new_nodes[node.id as usize];
        if let Some(mut w) = idx.write(trie3_god) {
            w.value.end = node.end;

            let mut children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<PrecomputeNode3Index, StateIDBV>> = BTreeMap::new();

            for (ek, dm) in node.children.iter() {
                // Build LLMTokenBV from tokens set
                let mut bv = LLMTokenBV::zeros();
                for t in ek.tokens.iter() {
                    if t <= ctx.max_llm_token_id {
                        bv.insert(t);
                    }
                }
                if bv.is_empty() {
                    continue;
                }

                let entry = children.entry((ek.pop, bv.clone())).or_insert_with(OrderedHashMap::new);
                for (dst, sset) in dm {
                    let mut sbv = StateIDBV::zeros();
                    for s in sset.iter() {
                        if s <= ctx.max_state_id {
                            sbv.insert(s);
                        }
                    }
                    if sbv.is_empty() {
                        continue;
                    }
                    let dst_idx = new_nodes[*dst as usize];
                    entry.entry(dst_idx)
                        .and_modify(|e| *e |= &sbv)
                        .or_insert(sbv);
                }
            }

            // Recompute live tokens from children
            let mut live = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in &children {
                live |= llm_bv;
            }
            w.value.live_tokens = live;
            *w.children_mut() = children;
        }
    }

    // Remap roots to new nodes
    roots.clear();
    for (key, nid) in root_pairs {
        let idx = new_nodes[*nid as usize];
        roots.insert(*key, idx);
    }

    // GC and recompute depths
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie3_god, &roots_vec);
    Trie::recompute_all_max_depths(trie3_god, &roots_vec);
}

/// High-level entry point: export -> run passes on mini -> import.
/// The coordinator isolates optimization authors from the full system types: they only need
/// to implement passes over MiniTrie and add them to the pipeline here.
pub fn run_pipeline_on_precompute3(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
    config: CoordinatorConfig,
) {
    // Export the current graph into a minimal structure
    let (mut mini, root_pairs, _old_mapping) =
        export_to_mini(roots, trie3_god, max_llm_token_id, max_state_id);

    // Build a fresh pass pipeline and context
    let mut ctx = OptimizationContext::new(max_llm_token_id, max_state_id);
    ctx.debug_level = 1;

    // Run initial metrics
    ctx.metrics_before = run_all_metrics(&mini);
    if ctx.debug_level > 0 {
        crate::debug!(
            1,
            "[Trie3 Opt] Metrics before optimization: {:?}",
            ctx.metrics_before
        );
    }

    let pipeline = build_pipeline(&config);

    // Run passes
    for pass in pipeline.iter() {
        if ctx.debug_level > 0 {
            crate::debug!(1, "[Trie3 Opt] Running pass: {}", pass.name());
        }
        pass.run(&mut mini, &mut ctx);
        if ctx.debug_level > 0 {
            let metrics = run_all_metrics(&mini);
            crate::debug!(
                1,
                "[Trie3 Opt] Metrics after '{}': {:?}",
                pass.name(),
                metrics
            );
        }
    }

    // Run final metrics
    ctx.metrics_after = run_all_metrics(&mini);
    if ctx.debug_level > 0 {
        crate::debug!(1, "[Trie3 Opt] Metrics after optimization: {:?}", ctx.metrics_after);
    }

    // Import the result back and finalize
    import_from_mini(&mini, &root_pairs, trie3_god, roots, &ctx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{LLMTokenBV, StateIDBV};
    use crate::datastructures::trie::Trie;
    use ordered_hash_map::OrderedHashMap;

    // A very basic roundtrip sanity test on a trivial graph.
    #[test]
    fn roundtrip_basic() {
        // Build a tiny God/arena with 2 nodes and one edge
        let arena = Trie3GodWrapper::new();

        let a_idx = PrecomputeNode3Index::new(arena.insert(Trie::new(PrecomputedNodeContents::internal())));
        let b_idx = PrecomputeNode3Index::new(arena.insert(Trie::new(PrecomputedNodeContents::internal())));

        {
            let mut aw = a_idx.write(&arena).unwrap();
            aw.value.end = false;

            let mut bv = LLMTokenBV::zeros();
            bv.insert(1);
            bv.insert(3);

            let mut sids = StateIDBV::zeros();
            sids.insert(2);
            sids.insert(4);

            let mut dm = OrderedHashMap::new();
            dm.insert(b_idx, sids);

            aw.children_mut().insert((0isize, bv), dm);

            // live tokens is derived but we'll set for completeness
            let mut live = LLMTokenBV::zeros();
            for ((_, l), _) in aw.children() {
                live |= l;
            }
            aw.value.live_tokens = live;
        }
        {
            let mut bw = b_idx.write(&arena).unwrap();
            bw.value.end = true;
        }

        let mut roots = BTreeMap::new();
        roots.insert(TokenizerStateID(0), a_idx);

        let cfg = CoordinatorConfig::default();

        run_pipeline_on_precompute3(
            &mut roots,
            &arena,
            10, // max token id
            10, // max state id
            cfg,
        );

        // After running, ensure roots still exist and points to some node.
        assert!(roots.get(&TokenizerStateID(0)).is_some());
    }
}
