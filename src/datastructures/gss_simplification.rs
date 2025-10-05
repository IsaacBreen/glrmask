use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use crate::datastructures::gss::{process_predecessors, DestKey, GSSNode, NodeMap, NodeSet};
use crate::glr::parser::ParseStateEdgeContent;
use crate::tokenizer::TokenizerStateID;

pub fn fuse_predecessors_recursive(
    node_arc: &Arc<GSSNode>,
    levels: usize,
    memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>,
) -> Arc<GSSNode> {
    if levels == 0 {
        return node_arc.clone();
    }
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(fused_arc) = memo.get(&node_ptr) {
        return fused_arc.clone();
    }

    // 1. Recursively fuse the predecessors first (post-order traversal).
    let mut recursively_fused_predecessors = Vec::new();
    for (edge_val, preds_by_depth) in node_arc.predecessors() {
        for pred_vec in preds_by_depth.values() {
            for pred_arc in pred_vec {
                let fused_pred_arc = fuse_predecessors_recursive(pred_arc, levels - 1, memo);
                recursively_fused_predecessors.push((edge_val.clone(), fused_pred_arc));
            }
        }
    }

    // 2. Group the now-fused predecessors by their edge value.
    let mut grouped_by_edge = BTreeMap::<ParseStateEdgeContent, Vec<Arc<GSSNode>>>::new();
    for (edge_val, pred_arc) in recursively_fused_predecessors {
        grouped_by_edge.entry(edge_val).or_default().push(pred_arc);
    }

    // 3. For each edge value, merge all predecessors associated with it into a single node.
    let mut new_predecessors_set = NodeSet::new();
    for (edge_val, pred_arcs_to_merge) in grouped_by_edge {
        if pred_arcs_to_merge.is_empty() { continue; }

        let mut iter = pred_arcs_to_merge.into_iter();
        let first = iter.next().unwrap();

        let final_pred_arc = if iter.len() == 0 {
            first
        } else {
            let mut merged_node = (*first).clone();
            for other_arc in iter {
                merged_node.merge_with_depth(1, &other_arc);
            }
            Arc::new(merged_node)
        };
        new_predecessors_set.insert((final_pred_arc, edge_val));
    }

    // 4. Rebuild the current node with the new, fused set of predecessors.
    let new_predecessors_map = process_predecessors(&new_predecessors_set);
    let local = node_arc.local_acc();
    let fused_node = GSSNode::new_with_map(local, new_predecessors_map);

    let result_arc = Arc::new(fused_node);
    memo.insert(node_ptr, result_arc.clone());
    result_arc
}

// --- Deep canonicalization & interning ---

#[derive(Default)]
struct GSSInternPool {
    // Intern by full structural equality. Using BTreeMap keeps the process deterministic.
    by_node: BTreeMap<GSSNode, Arc<GSSNode>>,
}

impl GSSInternPool {
    fn new() -> Self { Self { by_node: BTreeMap::new() } }

    fn intern(&mut self, node: GSSNode) -> Arc<GSSNode> {
        if let Some(existing) = self.by_node.get(&node) {
            return existing.clone();
        }
        // Store an Arc of the created node as the canonical representative.
        let arc = Arc::new(node);
        // Key is a value-clone of the node; that's fine because we want structural keying.
        self.by_node.insert((*arc).clone(), arc.clone());
        arc
    }
}

// Canonicalize a single GSS node (bottom-up) using a global intern pool and a per-walk memo.
fn canonicalize_node(
    node_arc: &Arc<GSSNode>,
    pool: &mut GSSInternPool,
    memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>,
) -> Arc<GSSNode> {
    let ptr = Arc::as_ptr(node_arc);
    if let Some(cached) = memo.get(&ptr) {
        return cached.clone();
    }

    let out = match node_arc.as_ref() {
        GSSNode::Root(root) => {
            // Intern roots by their Acc value. This makes identical roots share a single Arc.
            let acc = (*root.acc).clone();
            pool.intern(GSSNode::new(acc))
        }
        GSSNode::Internal(internal) => {
            // 1) Recursively canonicalize all predecessors.
            // 2) Deduplicate siblings that are structurally equal under the same (edge, DestKey).
            // 3) Build a new NodeMap and intern it.

            let mut new_map: NodeMap = BTreeMap::new();

            for (edge_val, preds_by_depth) in &internal.predecessors {
                let mut new_by_depth: BTreeMap<DestKey, Vec<Arc<GSSNode>>> = BTreeMap::new();

                for (dest_key, pred_vec) in preds_by_depth {
                    // Deduplicate by structure: multiple identical preds collapse into one.
                    let mut unify: BTreeMap<GSSNode, Arc<GSSNode>> = BTreeMap::new();

                    for pred_arc in pred_vec {
                        let canon_pred = canonicalize_node(pred_arc, pool, memo);
                        // Use value-based keying to ensure structural dedup, not pointer-only.
                        let key = (*canon_pred).clone();
                        // Keep a single canonical Arc per structural node.
                        unify.entry(key).or_insert(canon_pred);
                    }

                    let mut uniq_vec: Vec<Arc<GSSNode>> = unify.into_values().collect();
                    // Deterministic ordering: sort by structural value.
                    uniq_vec.sort_by(|a, b| (**a).cmp(&**b));
                    new_by_depth.insert(*dest_key, uniq_vec);
                }

                if !new_by_depth.is_empty() {
                    new_map.insert(edge_val.clone(), new_by_depth);
                }
            }

            // Important: pass a neutral Acc when rebuilding internal nodes, so we keep structure only.
            // Now: we must preserve local acc as well.
            pool.intern(GSSNode::new_with_map(internal.acc.clone(), new_map))
        }
    };

    memo.insert(ptr, out.clone());
    out
}

// Convenience: canonicalize a list of roots in-place with a single pool for maximal sharing.
pub(crate) fn simplify_roots_in_place(roots: &mut [Arc<GSSNode>]) {
    if roots.is_empty() { return; }
    let mut pool = GSSInternPool::new();
    let mut memo: HashMap<*const GSSNode, Arc<GSSNode>> = HashMap::new();

    for r in roots.iter_mut() {
        let canon = canonicalize_node(r, &mut pool, &mut memo);
        *r = canon;
    }
}

pub fn simplify(states: &mut BTreeMap<TokenizerStateID, Arc<GSSNode>>) {
    // We want cross-state sharing, so we collect all roots, canonicalize with a single pool, then write back.
    // 1) Collect all roots from all states.
    let mut all_roots: Vec<Arc<GSSNode>> =
        states.values().map(|s| s.clone()).collect();

    if all_roots.is_empty() { return; }

    // 2) Canonicalize them in place using one shared pool.
    simplify_roots_in_place(&mut all_roots);

    // 3) Write the canonicalized roots back.
    let mut canonical_roots_iter = all_roots.into_iter();
    for state in states.values_mut() {
        // This assumes a 1-to-1 mapping and preserved order, which BTreeMap provides.
        *state = canonical_roots_iter.next().unwrap();
    }
}
