// src/constraint_precompute3_intermediate_utils.rs
use crate::{
    constraint::{
        IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
        IntermediateTrie3GodWrapper, LLMTokenBV,
    },
    datastructures::trie::Trie,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
pub(crate) fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
    let mut combined_llm_bv = LLMTokenBV::max_ones();
    let mut has_llm_check = false;

    let mut other_ops: Vec<_> = path
        .into_iter()
        .filter_map(|ek| match ek {
            IntermediateTrie3EdgeKey::CheckLLM(bv) => {
                combined_llm_bv &= bv;
                has_llm_check = true;
                None
            }
            IntermediateTrie3EdgeKey::NoOp => None,
            IntermediateTrie3EdgeKey::Push(_) | IntermediateTrie3EdgeKey::Pop(_, _) => Some(ek),
        })
        .collect();

    if has_llm_check {
        other_ops.insert(0, IntermediateTrie3EdgeKey::CheckLLM(combined_llm_bv));
    }

    other_ops
}

/// Compares two Intermediate Trie3 graphs for equivalence by comparing their sets of normalized paths.
/// This is a strong equivalence check, suitable for testing optimization passes.
pub fn are_intermediate_trie3_graphs_equal<F>(
    roots_a: &[IntermediatePrecomputeNode3Index],
    god_a: &IntermediateTrie3GodWrapper,
    roots_b: &[IntermediatePrecomputeNode3Index],
    god_b: &IntermediateTrie3GodWrapper,
    is_end: &F,
    max_path_length: usize,
) -> bool
where
    F: Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
{
    // Only Pop and Push operations count towards path length for cycle detection.
    let is_path_edge: fn(&IntermediateTrie3EdgeKey, &(), IntermediatePrecomputeNode3Index) -> bool =
        |ek, _, _| {
            !matches!(ek, IntermediateTrie3EdgeKey::NoOp)
        };

    let get_normalized_paths = |god, roots| {
        Trie::get_all_paths_with_cycles(god, roots, is_end, is_path_edge, max_path_length)
            .into_iter()
            .map(|(_, path)| normalize_path(path.into_iter().map(|(ek, ..)| ek).collect()))
            .collect::<HashSet<_>>()
    };

    get_normalized_paths(god_a, roots_a) == get_normalized_paths(god_b, roots_b)
}

#[derive(Clone, Debug)]
struct NodeData {
    end: bool,
    // EdgeKey -> Vec<child_index>
    edges: BTreeMap<IntermediateTrie3EdgeKey, Vec<IntermediatePrecomputeNode3Index>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ClassSignature {
    end: bool,
    // Sorted by edge key; for each key we store a sorted, de-duplicated list of child class IDs
    edges: Vec<(IntermediateTrie3EdgeKey, Vec<usize>)>,
}

fn count_edges_for_roots(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
) -> usize {
    let nodes = Trie::all_nodes(god, roots);
    let mut edges = 0usize;
    for idx in nodes {
        if let Some(read) = idx.read(god) {
            for (_ek, dsts) in read.children() {
                edges += dsts.len();
            }
        }
    }
    edges
}

pub fn optimize_intermediate_trie3(
    roots: &[IntermediatePrecomputeNode3Index],
    god: &IntermediateTrie3GodWrapper,
    is_end: impl Fn(IntermediatePrecomputeNode3Index, &IntermediatePrecomputeNode3) -> bool,
) -> BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> {
    let t0 = Instant::now();
    eprintln!("[optimize_intermediate_trie3] Starting optimization for {} root(s)...", roots.len());

    // Snapshot original arena and roots for equivalence checking
    let original_god = god.deep_clone();
    let original_roots: Vec<_> = roots.to_vec();

    // Collect all reachable nodes from the provided roots
    let reachable = Trie::all_nodes(god, roots);
    let num_nodes_before = reachable.len();
    let num_edges_before = count_edges_for_roots(roots, god);
    eprintln!(
        "[optimize_intermediate_trie3] Reachable: {} nodes, {} edges (collect {:.2?})",
        num_nodes_before,
        num_edges_before,
        t0.elapsed()
    );

    // Early exit if trivial
    if num_nodes_before <= 1 {
        eprintln!("[optimize_intermediate_trie3] Nothing to optimize.");
        // Sanity check (no-op)
        let node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
            reachable.iter().map(|&i| (i, i)).collect();
        let new_roots: Vec<_> = original_roots
            .iter()
            .map(|r| *node_map.get(r).unwrap_or(r))
            .collect();
        assert!(
            are_intermediate_trie3_graphs_equal(
                &original_roots,
                &original_god,
                &new_roots,
                god,
                &is_end,
                25
            ),
            "Optimization failed to preserve graph equivalence for all roots (trivial case)"
        );
        return node_map;
    }

    // Build immutable snapshot of node data (end flag and outgoing edges)
    let mut data: BTreeMap<IntermediatePrecomputeNode3Index, NodeData> = BTreeMap::new();
    {
        for &idx in &reachable {
            let Some(read) = idx.read(god) else { continue };
            let end = read.value.end;
            let mut edges: BTreeMap<IntermediateTrie3EdgeKey, Vec<IntermediatePrecomputeNode3Index>> =
                BTreeMap::new();
            for (ek, dsts) in read.children() {
                let mut vs: Vec<_> = dsts.iter().map(|(d, _)| *d).collect();
                // Children are unique by construction (keys in OrderedHashMap), but sort for determinism
                vs.sort_by_key(|i| i.as_usize());
                edges.insert(ek.clone(), vs);
            }
            data.insert(idx, NodeData { end, edges });
        }
    }

    // Index nodes 0..N for array-based storage
    let mut indices: Vec<_> = data.keys().cloned().collect();
    indices.sort_by_key(|i| i.as_usize());
    let mut pos_of: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::with_capacity(indices.len());
    for (pos, idx) in indices.iter().enumerate() {
        pos_of.insert(*idx, pos);
    }

    // Initial partition: by (end flag, set of present edge keys)
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct InitialSignature {
        end: bool,
        keys: Vec<IntermediateTrie3EdgeKey>,
    }
    let mut class_ids: Vec<usize> = vec![0; indices.len()];
    {
        let mut interner: HashMap<InitialSignature, usize> = HashMap::new();
        for (pos, idx) in indices.iter().enumerate() {
            let nd = data.get(idx).unwrap();
            let mut keys: Vec<_> = nd.edges.keys().cloned().collect();
            keys.sort();
            let sig = InitialSignature { end: nd.end, keys };
            let cid = *interner.entry(sig).or_insert_with(|| interner.len());
            class_ids[pos] = cid;
        }
    }

    // Iterative refinement using child class sets
    let mut iterations = 0usize;
    let refine_start = Instant::now();
    loop {
        iterations += 1;
        let mut interner: HashMap<ClassSignature, usize> = HashMap::new();
        let mut new_class_ids = vec![0usize; indices.len()];
        let mut changed = 0usize;

        for (pos, idx) in indices.iter().enumerate() {
            let nd = data.get(idx).unwrap();
            let mut edges_sig: Vec<(IntermediateTrie3EdgeKey, Vec<usize>)> =
                Vec::with_capacity(nd.edges.len());
            for (ek, dsts) in nd.edges.iter() {
                // Map child indices to their current classes
                let mut child_classes: Vec<usize> = dsts
                    .iter()
                    .filter_map(|d| pos_of.get(d).map(|p| class_ids[*p]))
                    .collect();
                child_classes.sort_unstable();
                child_classes.dedup(); // collapse multiple children in the same class
                edges_sig.push((ek.clone(), child_classes));
            }
            edges_sig.sort_by(|a, b| a.0.cmp(&b.0));

            let sig = ClassSignature {
                end: nd.end,
                edges: edges_sig,
            };
            let cid = *interner.entry(sig).or_insert_with(|| interner.len());
            new_class_ids[pos] = cid;
            if new_class_ids[pos] != class_ids[pos] {
                changed += 1;
            }
        }

        eprintln!(
            "[optimize_intermediate_trie3] Refinement iter {}: classes = {} (changed {} of {})",
            iterations,
            new_class_ids.iter().max().map(|m| m + 1).unwrap_or(0),
            changed,
            indices.len()
        );

        class_ids = new_class_ids;
        if changed == 0 {
            break;
        }
        if iterations >= 100 {
            eprintln!(
                "[optimize_intermediate_trie3] WARNING: reached max iterations; proceeding with current partition"
            );
            break;
        }
    }
    eprintln!(
        "[optimize_intermediate_trie3] Partition stabilized after {} iteration(s) ({:.2?})",
        iterations,
        refine_start.elapsed()
    );

    let num_classes = class_ids.iter().max().map(|m| m + 1).unwrap_or(0);
    eprintln!(
        "[optimize_intermediate_trie3] Classes: {} (compression ratio {:.3}x)",
        num_classes,
        if num_classes > 0 {
            (num_nodes_before as f64) / (num_classes as f64)
        } else {
            1.0
        }
    );

    // Build per-class info from any representative node within the class.
    #[derive(Clone, Debug)]
    struct ClassInfo {
        end: bool,
        edges: BTreeMap<IntermediateTrie3EdgeKey, Vec<usize>>, // for each key, unique sorted child classes
    }
    let mut class_representative_pos: Vec<usize> = vec![usize::MAX; num_classes];
    for (pos, &cid) in class_ids.iter().enumerate() {
        if class_representative_pos[cid] == usize::MAX {
            class_representative_pos[cid] = pos;
        }
    }
    let mut classes: Vec<ClassInfo> = Vec::with_capacity(num_classes);
    classes.resize_with(num_classes, || ClassInfo {
        end: false,
        edges: BTreeMap::new(),
    });
    for cid in 0..num_classes {
        let pos = class_representative_pos[cid];
        let idx = indices[pos];
        let nd = data.get(&idx).unwrap();
        let mut edges: BTreeMap<IntermediateTrie3EdgeKey, Vec<usize>> = BTreeMap::new();
        for (ek, dsts) in nd.edges.iter() {
            let mut child_classes: Vec<usize> = dsts
                .iter()
                .filter_map(|d| pos_of.get(d).map(|p| class_ids[*p]))
                .collect();
            child_classes.sort_unstable();
            child_classes.dedup(); // collapse duplicates to a set
            edges.insert(ek.clone(), child_classes);
        }
        classes[cid] = ClassInfo {
            end: nd.end,
            edges,
        };
    }

    // Create canonical nodes for each class in the same arena (god)
    let build_start = Instant::now();
    eprintln!(
        "[optimize_intermediate_trie3] Building canonical graph: {} class node(s)...",
        num_classes
    );
    let mut class_to_new_index: Vec<IntermediatePrecomputeNode3Index> = Vec::with_capacity(num_classes);
    class_to_new_index.resize_with(num_classes, || IntermediatePrecomputeNode3Index::from_usize(usize::MAX));
    for cid in 0..num_classes {
        let value = if classes[cid].end {
            // Preserve "end" status
            // Prefer leaf() for end nodes; otherwise internal()
            IntermediatePrecomputeNode3::value_type::leaf()
        } else {
            IntermediatePrecomputeNode3::value_type::internal()
        };
        // The above uses associated constructors on the value; since IntermediatePrecomputeNode3 is a Trie<T>,
        // we access the value constructors directly.
        // But IntermediatePrecomputeNode3::value_type is not an alias; instead call via type name:
        let value = if classes[cid].end {
            crate::constraint::IntermediatePrecomputedNodeContents3::leaf()
        } else {
            crate::constraint::IntermediatePrecomputedNodeContents3::internal()
        };
        let new_idx = god.insert(Trie::new(value));
        class_to_new_index[cid] = new_idx;
    }

    // Now connect edges between class nodes
    for cid in 0..num_classes {
        let src_idx = class_to_new_index[cid];
        let edges = &classes[cid].edges;
        god.with_mut(src_idx, |node| {
            for (ek, child_classes) in edges {
                for &cc in child_classes {
                    let dst_idx = class_to_new_index[cc];
                    node.force_insert_to_node(ek.clone(), (), dst_idx);
                }
            }
        });
    }
    eprintln!(
        "[optimize_intermediate_trie3] Built canonical nodes and edges ({:.2?})",
        build_start.elapsed()
    );

    // Map original nodes to their canonical class node indices
    let mut node_map: BTreeMap<IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        BTreeMap::new();
    for (pos, &idx) in indices.iter().enumerate() {
        let cid = class_ids[pos];
        node_map.insert(idx, class_to_new_index[cid]);
    }

    // Prepare new roots (mapped to canonical nodes)
    let new_roots: Vec<_> = original_roots
        .iter()
        .map(|r| *node_map.get(r).unwrap_or(r))
        .collect();

    // Print before/after counts (before GC)
    let num_nodes_after = Trie::all_nodes(god, &new_roots).len();
    let num_edges_after = count_edges_for_roots(&new_roots, god);
    eprintln!(
        "[optimize_intermediate_trie3] Before: {} nodes, {} edges | After (pre-GC): {} nodes, {} edges",
        num_nodes_before, num_edges_before, num_nodes_after, num_edges_after
    );

    // GC unreachable old nodes; keep only the optimized component
    let gc_start = Instant::now();
    Trie::gc(god, &new_roots);
    let num_nodes_post_gc = Trie::all_nodes(god, &new_roots).len();
    let num_edges_post_gc = count_edges_for_roots(&new_roots, god);
    eprintln!(
        "[optimize_intermediate_trie3] GC complete: reachable now {} nodes, {} edges ({:.2?})",
        num_nodes_post_gc, num_edges_post_gc, gc_start.elapsed()
    );

    eprintln!(
        "[optimize_intermediate_trie3] Total optimization time: {:.2?}",
        t0.elapsed()
    );

    // Check equivalence after optimization
    assert!(
        are_intermediate_trie3_graphs_equal(&original_roots, &original_god, &new_roots, god, &is_end, 25),
        "Optimization failed to preserve graph equivalence for all roots"
    );

    node_map
}

