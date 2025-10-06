use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Write};
use std::sync::Arc;

use bimap::BiBTreeMap;
use profiler_macro::time_it;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use crate::datastructures::gss::{Acc, DestKey, GSSNode, MaxDepth};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::glr::grammar::Terminal;
use crate::glr::parser::ParseStateEdgeContent;
use crate::glr::table::StateID;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;

// --- Analysis and Debugging ---
#[derive(Debug, Clone, Eq, Hash)]
#[allow(dead_code)] pub(crate) struct RootItem<'a> {
    node: &'a GSSNode,
    path_acc: Arc<Acc>,
}

impl<'a> PartialEq for RootItem<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.path_acc == other.path_acc
    }
}

impl<'a> PartialOrd for RootItem<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for RootItem<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.cmp(other.node)
            .then_with(|| self.path_acc.cmp(&other.path_acc))
    }
}

impl<'a> RootItem<'a> {
    #[allow(dead_code)] pub(crate) fn resolved_acc(&self) -> Arc<Acc> {
        Arc::new(Acc::narrow(&self.path_acc, &self.node.acc()))
    }
}

/// Traverses the GSS graph from the given nodes and returns all unique root nodes (nodes with no predecessors).
pub(crate) fn get_roots<'a>(nodes: impl IntoIterator<Item = &'a GSSNode>) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
    // We carry the "last edge" used to reach the next node; when we finally hit a root,
    // that last edge is the key used for the result map.
    let mut queue: BTreeMap<
        MaxDepth,
        BTreeMap<(*const GSSNode, Option<ParseStateEdgeContent>), BTreeSet<Arc<Acc>>>
    > = BTreeMap::new();

    let mut results: BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> = BTreeMap::new();

    for node in nodes {
        let node_ptr = node as *const GSSNode;
        let depth = node.max_depth();
        queue
            .entry(depth)
            .or_default()
            .entry((node_ptr, None))
            .or_insert_with(|| {
                // Initialize path_acc with the starting node's local acc.
                BTreeSet::from([Arc::new((*node.local_acc()).clone())])
            });
    }

    while let Some((_depth, nodes_at_depth)) = queue.pop_last() {
        for ((node_ptr, last_edge_opt), path_acc_set) in nodes_at_depth {
            let current_node = unsafe { &*node_ptr };

            if current_node.is_root() {
                if let Some(edge) = last_edge_opt {
                    results
                        .entry(edge)
                        .or_default()
                        .extend(path_acc_set);
                }
            } else {
                for (edge_val, preds_by_depth) in current_node.predecessors().iter() {
                    for pred_arc in preds_by_depth.values().flatten() {
                        // For each incoming path acc, create a new outgoing path acc
                        let mut per_child_acc_set = BTreeSet::new();
                        for path_acc in &path_acc_set {
                            let per_child_acc = Arc::new(Acc::narrow(path_acc, &pred_arc.local_acc()));
                            per_child_acc_set.insert(per_child_acc);
                        }

                        let pred_ptr = pred_arc.as_ref() as *const GSSNode;
                        let pred_depth = pred_arc.max_depth();
                        queue
                            .entry(pred_depth)
                            .or_default()
                            .entry((pred_ptr, Some(edge_val.clone())))
                            .and_modify(|e| e.extend(per_child_acc_set.clone()))
                            .or_insert_with(|| per_child_acc_set);
                    }
                }
            }
        }
    }

    results
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GSSStats {
    pub(crate) num_roots: usize,
    pub(crate) num_root_predecessors: usize,
    pub(crate) num_unique_root_predecessor_keys: usize,
    pub(crate) total_edges: usize,
    pub(crate) unique_nodes: usize,
    pub(crate) num_leaves: usize,
    pub(crate) structurally_unique_nodes: usize,
    pub(crate) structural_redundancy: f64,
    pub(crate) num_redundant_nodes: usize,
    pub(crate) max_depth: usize,
    pub(crate) average_depth: f64,
    pub(crate) merge_points: usize,
    pub(crate) max_predecessors_with_values: usize,
    pub(crate) average_predecessors_with_values: f64,
}

/// Gathers statistics about the structure and complexity of a GSS forest.
#[time_it]
pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut total_depth = 0u64;
    let mut total_preds = 0u64;

    let mut root_predecessor_dest_keys = HashSet::new();
    for root_node in roots {
        stats.num_root_predecessors += root_node.num_predecessors();
        for edge_value in root_node.predecessors().keys() {
            root_predecessor_dest_keys.insert(edge_value.clone());
        }

        let node_ptr = *root_node as *const GSSNode;
        if visited.insert(node_ptr) {
            queue.push_back((*root_node, 0));
        }
    }
    stats.num_unique_root_predecessor_keys = root_predecessor_dest_keys.len();

    // Reset visited for the main traversal to correctly process all nodes.
    visited.clear();

    while let Some((node, depth)) = queue.pop_front() {
        let node_ptr = node as *const GSSNode;
        if !visited.insert(node_ptr) {
            continue;
        }

        if node.is_root() {
            stats.num_leaves += 1;
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        total_depth += depth as u64;

        let num_preds = node.num_predecessors();
        stats.max_predecessors_with_values = stats.max_predecessors_with_values.max(num_preds);
        total_preds += num_preds as u64;

        let unique_pred_arcs: HashSet<_> = node
            .predecessors()
            .values()
            .flat_map(|v| v.values())
            .flat_map(|v| v.iter())
            .map(Arc::as_ptr)
            .collect();
        if unique_pred_arcs.len() > 1 {
            stats.merge_points += 1;
        }

        for pred_arc in node
            .predecessors()
            .values()
            .flat_map(|v| v.values())
            .flat_map(|v| v.iter()) {
            queue.push_back((pred_arc.as_ref(), depth + 1));
        }
    }

    stats.total_edges = total_preds as usize;

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds as f64 / stats.unique_nodes as f64;
    }

    // Calculate structural uniqueness
    let mut structural_memo = HashMap::new();
    let mut structural_cache: BTreeMap<BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<usize>>>, usize> = BTreeMap::new();
    for root_node in roots {
        get_structural_id(root_node, &mut structural_memo, &mut structural_cache);
    }
    stats.structurally_unique_nodes = structural_cache.len();
    if stats.unique_nodes > 0 {
        stats.structural_redundancy = 1.0 - (stats.structurally_unique_nodes as f64 / stats.unique_nodes as f64);
    }
    stats.num_redundant_nodes = stats.unique_nodes - stats.structurally_unique_nodes;
    stats
}

/// Helper for `gather_gss_stats` to compute a unique ID for a node's structure.
fn get_structural_id(
    node: &GSSNode,
    memo: &mut HashMap<*const GSSNode, usize>,
    structural_cache: &mut BTreeMap<BTreeMap<ParseStateEdgeContent, BTreeMap<DestKey, Vec<usize>>>, usize>,
) -> usize {
    let node_ptr = node as *const GSSNode;
    if let Some(id) = memo.get(&node_ptr) {
        return *id;
    }

    let mut pred_structural_ids = BTreeMap::new();
    for (edge_val, preds_by_depth) in node.predecessors() {
        let mut ids_by_depth = BTreeMap::new();
        for (dest_key, pred_vec) in preds_by_depth {
            let mut ids_vec = Vec::new();
            for pred_arc in pred_vec {
                let pred_id = get_structural_id(pred_arc.as_ref(), memo, structural_cache);
                ids_vec.push(pred_id);
            }
            ids_vec.sort();
            ids_by_depth.insert(*dest_key, ids_vec);
        }
        pred_structural_ids.insert(edge_val.clone(), ids_by_depth);
    }

    let next_id = structural_cache.len();
    let id = *structural_cache.entry(pred_structural_ids).or_insert(next_id);
    memo.insert(node_ptr, id);
    id
}

/// Finds the longest path from any leaf to the given root node.
/// Returns `None` if the node has no predecessors.
pub(crate) fn find_longest_path(root_node: &Arc<GSSNode>) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    if root_node.predecessors().is_empty() {
        return None;
    }

    fn find_longest_recursive(
        node_arc: &Arc<GSSNode>,
        memo: &mut HashMap<*const GSSNode, Vec<(ParseStateEdgeContent, Arc<GSSNode>)>>,
    ) -> Vec<(ParseStateEdgeContent, Arc<GSSNode>)> {
        let node_ptr = Arc::as_ptr(node_arc);
        if let Some(cached) = memo.get(&node_ptr) {
            return cached.clone();
        }

        if node_arc.predecessors().is_empty() {
            return Vec::new();
        }

        let mut longest_path = Vec::new();
        for (edge_val, preds_by_depth) in node_arc.predecessors().iter() {
            for pred_vec in preds_by_depth.values() {
                for pred_arc in pred_vec {
                    let mut path_from_pred = find_longest_recursive(pred_arc, memo);
                    path_from_pred.push((edge_val.clone(), node_arc.clone()));
                    if path_from_pred.len() > longest_path.len() {
                        longest_path = path_from_pred;
                    }
                }
            }
        }

        memo.insert(node_ptr, longest_path.clone());
        longest_path
    }

    let mut memo = HashMap::new();
    let path = find_longest_recursive(root_node, &mut memo);
    if path.is_empty() { None } else { Some(path) }
}

/// Randomly samples a single path from a GSS forest.
#[allow(dead_code)] pub(crate) fn sample_path(roots: &[&GSSNode], seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    if roots.is_empty() {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let root_index = rng.gen_range(0..roots.len());
    let mut current_node_arc = Arc::new(roots[root_index].clone());

    let mut path = Vec::new();

    loop {
        if current_node_arc.is_empty() {
            break;
        }

        let predecessors: Vec<_> = GSSNode::peek_iter(&current_node_arc).collect();
        if predecessors.is_empty() {
            break;
        }

        let chosen_index = rng.gen_range(0..predecessors.len());
        let chosen_peek = &predecessors[chosen_index];

        path.push(chosen_peek.edge_value().clone());

        current_node_arc = chosen_peek.predecessor_node().clone();
    }

    Some(path)
}

pub(crate) struct GSSPrintConfig<'a> {
    pub(crate) labels: Option<&'a [String]>,
    pub(crate) max_edges: usize,
    pub(crate) original_internal_bimap: Option<&'a BTreeMap<usize, usize>>,
    pub(crate) llm_token_map: Option<&'a BiBTreeMap<Vec<u8>, LLMTokenID>>,
    pub(crate) verbose: bool,
}

impl<'a> Default for GSSPrintConfig<'a> {
    fn default() -> Self {
        Self {
            labels: None,
            max_edges: usize::MAX,
            original_internal_bimap: None,
            llm_token_map: None,
            verbose: false,
        }
    }
}

/// Pretty-prints a GSS forest for debugging.
pub fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    config: &GSSPrintConfig,
) -> (String, Vec<StateID>) {
    fn print_predecessors_recursive(
        node_arc: &Arc<GSSNode>,
        node_ids: &mut HashMap<*const GSSNode, usize>,
        visited_nodes: &mut HashSet<*const GSSNode>,
        prefix: &str,
        node_count: &mut usize,
        output: &mut String,
        terminal_map: &BiBTreeMap<Terminal, TerminalID>,
        state_ids_in_order: &mut Vec<StateID>,
        seen_state_ids: &mut HashSet<StateID>,
        config: &GSSPrintConfig,
    ) -> Result<(), std::fmt::Error> {
        let node_ptr = Arc::as_ptr(node_arc);
        if visited_nodes.contains(&node_ptr) {
            return Ok(());
        }
        visited_nodes.insert(node_ptr);

        let predecessors: Vec<_> = node_arc.predecessors()
            .iter()
            .flat_map(|(edge_val, preds_by_depth)| {
                preds_by_depth.values().flat_map(move |pred_vec| {
                    pred_vec.iter().map(move |pred_arc| (edge_val, pred_arc))
                })
            })
            .collect();

        for (i, (edge_val, pred_arc)) in predecessors.iter().enumerate() {
            if *node_count >= config.max_edges {
                writeln!(output, "{}... (Truncated)", prefix)?;
                return Ok(());
            }

            let is_last = i == predecessors.len() - 1;
            let connector = if is_last { "└──" } else { "├──" };
            let new_prefix = if is_last {
                format!("{}  ", prefix)
            } else {
                format!("{}│ ", prefix)
            };

            let pred_ptr = Arc::as_ptr(pred_arc);
            let node_ids_len = node_ids.len();
            let pred_id = *node_ids.entry(pred_ptr).or_insert(node_ids_len);

            // Collect state ID for explanation
            if seen_state_ids.insert(edge_val.state_id) {
                state_ids_in_order.push(edge_val.state_id);
            }

            let acc_child = format_acc(
                pred_arc.as_ref(),
                terminal_map,
                config.original_internal_bimap,
                config.llm_token_map,
                config,
            );
            if config.verbose {
                if acc_child.is_empty() {
                    writeln!(
                        output,
                        "{}{} edge {} -> Node {} (ptr: {:p}, hash: {:x})",
                        prefix, connector, edge_val.state_id.0, pred_id, pred_ptr, pred_arc.hash_code(),
                    )?;
                } else {
                    writeln!(
                        output,
                        "{}{} edge {} -> Node {} (ptr: {:p}, hash: {:x}) {}",
                        prefix, connector, edge_val.state_id.0, pred_id, pred_ptr, pred_arc.hash_code(), acc_child,
                    )?;
                }
            } else if acc_child.is_empty() {
                writeln!(
                    output,
                    "{}{} edge {} -> Node {}",
                    prefix, connector, edge_val.state_id.0, pred_id,
                )?;
            } else {
                writeln!(
                    output,
                    "{}{} edge {} -> Node {} {}",
                    prefix, connector, edge_val.state_id.0, pred_id, acc_child,
                )?;
            }
            *node_count += 1;

            print_predecessors_recursive(
                pred_arc, node_ids, visited_nodes, &new_prefix, node_count,
                output, terminal_map, state_ids_in_order, seen_state_ids, config,
            )?;
        }
        Ok(())
    }

    let mut node_ids = HashMap::new();
    let mut visited_nodes = HashSet::new();
    let mut count = 0;
    let mut out_str = String::new();
    let mut state_ids_in_order = Vec::new();
    let mut seen_state_ids = HashSet::new();

    if roots.is_empty() { return ("GSS Forest: (No roots)".to_string(), state_ids_in_order); }
    writeln!(&mut out_str, "GSS Forest (Max Edges: {}):", config.max_edges).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        if count >= config.max_edges {
            writeln!(&mut out_str, "... (Truncated)").unwrap();
            break;
        }

        let root_ptr = Arc::as_ptr(root_arc);
        let node_ids_len = node_ids.len();
        let root_id = *node_ids.entry(root_ptr).or_insert(node_ids_len);

        let acc_str = format_acc(
            root_arc.as_ref(),
            terminal_map,
            config.original_internal_bimap,
            config.llm_token_map,
            config,
        );
        let root_label = config.labels.map_or_else(|| format!("Root {}", i), |l| l[i].clone());

        if config.verbose {
            if acc_str.is_empty() {
                writeln!(
                    &mut out_str,
                    "{}: Node {} (ptr: {:p}, hash: {:x})",
                    root_label, root_id, root_ptr, root_arc.hash_code()
                ).unwrap();
            } else {
                writeln!(
                    &mut out_str,
                    "{}: Node {} (ptr: {:p}, hash: {:x}) {}",
                    root_label, root_id, root_ptr, root_arc.hash_code(), acc_str
                ).unwrap();
            }
        } else if acc_str.is_empty() {
            writeln!(&mut out_str, "{}: Node {}", root_label, root_id).unwrap();
        } else {
            writeln!(&mut out_str, "{}: Node {} {}", root_label, root_id, acc_str).unwrap();
        }
        count += 1;

        let _ = print_predecessors_recursive(
            root_arc, &mut node_ids, &mut visited_nodes, "  ", &mut count,
            &mut out_str, terminal_map, &mut state_ids_in_order, &mut seen_state_ids, config,
        );
    }

    (out_str, state_ids_in_order)
}

/// Formats an accumulator for concise display in the GSS printout.
pub(crate) fn format_acc(
    node: &GSSNode,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    config: &GSSPrintConfig,
) -> String {
    let _ = (original_internal_bimap, llm_token_map);

    let acc = node.local_acc();

    if config.verbose {
        // In verbose mode, print the full debug representation of the Acc.
        return format!("[acc: {:?}]", acc);
    }

    let summarize_llm = |bv: &HybridBitset, label: &str| -> Option<String> {
        if *bv == HybridBitset::max_ones() {
            return None;
        }
        if bv.is_empty() {
            return Some(format!("{}=∅", label));
        }
        let total = bv.len();
        const MAX_SHOW: usize = 8;
        let sample: Vec<String> = bv.iter().take(MAX_SHOW).map(|id| id.to_string()).collect();
        if total > MAX_SHOW {
            Some(format!("{}({}): [{} …]", label, total, sample.join(", ")))
        } else {
            Some(format!("{}({}): [{}]", label, total, sample.join(", ")))
        }
    };

    let summarize_disallowed_terminals = |allowed_terminals: &HybridL2Bitset, label: &str| -> Option<String> {
        let mut any_disallowed = false;
        let mut parts = Vec::new();
        const MAX_RANGES_TO_SHOW: usize = 3;
        for (range, allowed_bv) in allowed_terminals.range_values() {
            let disallowed_bv = HybridBitset::max_ones() - allowed_bv;
            if disallowed_bv.is_empty() {
                continue;
            }
            any_disallowed = true;
            if parts.len() >= MAX_RANGES_TO_SHOW {
                break;
            }
            let range_str = if range.start() == range.end() {
                format!("{}", range.start())
            } else {
                format!("{}..={}", range.start(), range.end())
            };

            if disallowed_bv == HybridBitset::max_ones() {
                parts.push(format!("state(s) {}: all", range_str));
                continue;
            }

            const MAX_NAMES_TO_SHOW: usize = 5;
            let num_disallowed = disallowed_bv.len();
            let names: Vec<_> = disallowed_bv.iter().take(MAX_NAMES_TO_SHOW)
                .map(|tid_val| terminal_map.get_by_right(&TerminalID(tid_val))
                    .map_or_else(|| format!("<ID:{}>", tid_val), |t| t.to_string()))
                .collect();
            let names_str = names.join(", ");

            if num_disallowed > MAX_NAMES_TO_SHOW {
                parts.push(format!("state(s) {} ({}): [{}, …]", range_str, num_disallowed, names_str));
            } else {
                parts.push(format!("state(s) {}: [{}]", range_str, names_str));
            }
        }
        if !any_disallowed {
            None
        } else if parts.is_empty() {
            Some(format!("Disallowed {}(…)", label))
        } else {
            Some(format!("Disallowed {}({})", label, parts.join("; ")))
        }
    };

    let union_llm_opt = summarize_llm(&acc.llm_tokens_union, "LLM(U)");
    let union_terminals_opt = summarize_disallowed_terminals(&acc.terminals_union, "Term(U)");

    let stored_trie_nodes_str = {
        const MAX_PTRS_TO_SHOW: usize = 5;
        let n = acc.stored_trie_nodes().len();
        if n == 0 {
            None
        } else if n <= MAX_PTRS_TO_SHOW {
            let ptrs: Vec<String> = acc
                .stored_trie_nodes()
                .iter()
                .map(|wrapper| format!("{}", wrapper.as_arc()))
                .collect();
            Some(format!("Trie(n={}, [{}])", n, ptrs.join(", ")))
        } else {
            let ptrs_sample: Vec<String> = acc
                .stored_trie_nodes()
                .iter()
                .take(MAX_PTRS_TO_SHOW)
                .map(|wrapper| format!("{}", wrapper.as_arc()))
                .collect();
            let remaining = n - MAX_PTRS_TO_SHOW;
            Some(format!("Trie(n={}, first {}: {}, …; +{} more)", n, MAX_PTRS_TO_SHOW, ptrs_sample.join(", "), remaining))
        }
    };

    let mut parts: Vec<String> = Vec::new();
    if let Some(s) = union_llm_opt { parts.push(s); }
    if let Some(s) = union_terminals_opt { parts.push(s); }
    if let Some(s) = stored_trie_nodes_str { parts.push(s); }

    if parts.is_empty() {
        String::new()
    } else {
        format!("[{}]", parts.join(", "))
    }
}

/// Checks if a GSS has a simple structure that can be cached.
/// The simple structures are:
/// `Internal(state_id) -> Internal(hallucinated_id) -> Root(leaf)`
/// Returns `Some((state_id, acc))` if it matches.
#[time_it]
pub(crate) fn is_simple_gss(
    node: &Arc<GSSNode>,
    hallucinated_state_id: StateID,
) -> Option<(StateID, Arc<Acc>)> {
    if let GSSNode::Internal(internal) = node.as_ref() {
        // Must have exactly one predecessor edge group.
        if internal.predecessors().len() == 1 {
            let (edge_content, preds_by_depth) = internal.predecessors().iter().next().unwrap();
            // Must have predecessors at exactly one depth.
            if preds_by_depth.len() == 1 {
                let (_depth, pred_vec) = preds_by_depth.iter().next().unwrap();
                // Must have exactly one predecessor node.
                if pred_vec.len() == 1 {
                    let predecessor = &pred_vec[0];
                    let state_id_x = edge_content.state_id;

                    // Now check the predecessor. It must be an internal node.
                    if let GSSNode::Internal(pred_internal) = predecessor.as_ref() {
                        // This must be the node with the hallucinated state ID edge.
                        if pred_internal.predecessors().len() == 1 {
                            let (halluc_edge, halluc_preds_by_depth) = pred_internal.predecessors().iter().next().unwrap();
                            if halluc_edge.state_id == hallucinated_state_id && halluc_preds_by_depth.len() == 1 {
                                let (_depth, halluc_pred_vec) = halluc_preds_by_depth.iter().next().unwrap();
                                if halluc_pred_vec.len() == 1 {
                                    let leaf = &halluc_pred_vec[0];
                                    if let GSSNode::Root(leaf_root) = leaf.as_ref() {
                                        // This is the valid pattern.
                                        if !leaf_root.acc().stored_trie_nodes().is_empty() {
                                            // The returned Acc must be the result of narrowing down the path.
                                            let path_acc1 = Acc::narrow(&internal.acc(), &pred_internal.acc());
                                            let final_acc = Acc::narrow(&path_acc1, &leaf_root.acc());
                                            return Some((state_id_x, Arc::new(final_acc)));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}
