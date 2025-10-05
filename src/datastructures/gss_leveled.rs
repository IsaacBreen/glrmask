// src/datastructures/gss_leveled.rs

//! A GSS implementation based on the purely functional, trie-based LeveledGSS.
//! This module provides a wrapper `GSSNode` around `LeveledGSS` and re-implements
//! the necessary analysis, pruning, and utility functions to be used by the parser
//! and constraint logic.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use bimap::BiBTreeMap;
use profiler_macro::time_it;

use crate::constraint::{LLMTokenBV, StateIDBV, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::glr::grammar::Terminal;
use crate::glr::parser::ParseStateEdgeContent;
use crate::glr::table::StateID;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;
use crate::datastructures::leveled_gss::{LeveledGSS, Merge};
use crate::datastructures::gss::{StoredPrecomputeNodeIndex, StoredTrieGodWrapper};

// --- Acc (Accumulator) ---

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    pub stored_trie_nodes: BTreeSet<StoredPrecomputeNodeIndex>,
}

impl Merge for Acc {
    fn merge(&self, other: &Self) -> Self {
        Acc {
            llm_tokens_union: &self.llm_tokens_union | &other.llm_tokens_union,
            terminals_union: &self.terminals_union | &other.terminals_union,
            stored_trie_nodes: &self.stored_trie_nodes | &other.stored_trie_nodes,
        }
    }
}

impl Acc {
    pub(crate) fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    pub(crate) fn new_dead() -> Self {
        Self {
            llm_tokens_union: HybridBitset::zeros(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    pub(crate) fn stored_trie_nodes(&self) -> &BTreeSet<StoredPrecomputeNodeIndex> { &self.stored_trie_nodes }
    pub(crate) fn stored_trie_nodes_mut(&mut self) -> &mut BTreeSet<StoredPrecomputeNodeIndex> { &mut self.stored_trie_nodes }
    pub(crate) fn union_llm_tokens(&self) -> LLMTokenBV { self.llm_tokens_union.clone() }
}

// --- GSSNode Wrapper ---

#[derive(Clone)]
pub struct GSSNode(pub LeveledGSS<ParseStateEdgeContent, Acc>);

impl Debug for GSSNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // For debugging, it's useful to see the stacks. This can be very verbose.
        write!(f, "GSSNode({} stacks)", self.0.to_stacks().len())
    }
}

impl PartialEq for GSSNode {
    fn eq(&self, other: &Self) -> bool {
        // LeveledGSS doesn't implement Eq, so we compare by stacks. This can be slow.
        self.0.to_stacks() == other.0.to_stacks()
    }
}
impl Eq for GSSNode {}


// --- GSSPopper and related items ---

#[derive(Debug, Clone)]
pub struct GSSPopper {
    pub paths: BTreeMap<ParseStateEdgeContent, Arc<GSSNode>>,
    pub below_bottom: BTreeMap<usize, Acc>,
}

pub struct GSSPopperItem<'a> {
    pub edge_content: &'a ParseStateEdgeContent,
    pub predecessors: &'a GSSNode,
}

impl GSSNode {
    pub fn new_fresh() -> Self {
        Self(LeveledGSS::from_stacks(&[(vec![], Acc::new_fresh())]))
    }

    pub fn new_dead() -> Self {
        Self(LeveledGSS::empty())
    }
    
    pub fn new(acc: Acc) -> Self {
        Self(LeveledGSS::from_stacks(&[(vec![], acc)]))
    }

    pub fn from_stacks(stacks: &[(Vec<ParseStateEdgeContent>, Acc)]) -> Self {
        Self(LeveledGSS::from_stacks(stacks))
    }

    pub fn push(&self, edge_value: ParseStateEdgeContent) -> Self {
        Self(self.0.push(edge_value))
    }
    
    pub fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Self {
        if edge_values.is_empty() {
            return Self::new_dead();
        }
        let mut merged = self.push(edge_values[0].clone());
        for edge in edge_values.iter().skip(1) {
            let pushed = self.push(edge.clone());
            merged = merged.merge(&pushed);
        }
        merged
    }

    pub fn popn(&self, n: usize) -> GSSPopper {
        let popped_gss = self.0.popn(n as isize);
        
        let mut paths = BTreeMap::new();
        let top_items = popped_gss.peek();

        for item in top_items {
            let predecessors_gss = popped_gss.isolate(Some(item.clone())).pop();
            paths.insert(item, Arc::new(GSSNode(predecessors_gss)));
        }

        // TODO: LeveledGSS does not support tracking pop distance below bottom.
        // This means substring parsing will not work correctly.
        let below_bottom = BTreeMap::new();

        GSSPopper { paths, below_bottom }
    }

    pub fn merge(&self, other: &Self) -> Self {
        Self(self.0.merge(&other.0))
    }

    pub fn merge_many_with_depth(_depth: usize, nodes: impl IntoIterator<Item = Arc<GSSNode>>) -> Arc<GSSNode> {
        let mut iter = nodes.into_iter();
        if let Some(first) = iter.next() {
            let mut merged = (*first).clone();
            for other in iter {
                merged = merged.merge(&other);
            }
            Arc::new(merged)
        } else {
            Arc::new(GSSNode::new_fresh())
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn allowed_llm_tokens(&self) -> LLMTokenBV {
        self.0.reduce_acc().map_or(LLMTokenBV::zeros(), |acc| acc.llm_tokens_union)
    }

    pub fn is_alive(&self) -> bool {
        !self.allowed_llm_tokens().is_empty()
    }

    pub fn num_predecessors(&self) -> usize {
        self.0.peek().len()
    }

    pub fn max_depth(&self) -> usize {
        self.0.to_stacks().iter().map(|(s, _)| s.len()).max().unwrap_or(0)
    }
}


// --- Analysis and Debugging ---
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GSSStats {
    pub num_stacks: usize,
    pub max_depth: usize,
    pub average_depth: f64,
    // The following fields from the old GSS are not easily computed with LeveledGSS
    // and are set to default values.
    pub num_roots: usize,
    pub num_root_predecessors: usize,
    pub num_unique_root_predecessor_keys: usize,
    pub total_edges: usize,
    pub unique_nodes: usize,
    pub num_leaves: usize,
    pub structurally_unique_nodes: usize,
    pub structural_redundancy: f64,
    pub num_redundant_nodes: usize,
    pub merge_points: usize,
    pub max_predecessors_with_values: usize,
    pub average_predecessors_with_values: f64,
}

pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut all_stacks = Vec::new();
    for root in roots {
        all_stacks.extend(root.0.to_stacks());
    }

    if all_stacks.is_empty() {
        return GSSStats::default();
    }

    let num_stacks = all_stacks.len();
    let max_depth = all_stacks.iter().map(|(s, _)| s.len()).max().unwrap_or(0);
    let total_depth: usize = all_stacks.iter().map(|(s, _)| s.len()).sum();
    let average_depth = if num_stacks > 0 { total_depth as f64 / num_stacks as f64 } else { 0.0 };

    GSSStats {
        num_stacks,
        max_depth,
        average_depth,
        unique_nodes: num_stacks, // Approximation
        ..Default::default()
    }
}

pub fn get_roots(_nodes: impl IntoIterator<Item = &GSSNode>) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
    // This function's semantics are tied to the graph model and don't map well.
    // Returning an empty map as a placeholder.
    BTreeMap::new()
}

pub fn find_longest_path(root_node: &Arc<GSSNode>) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    let stacks = root_node.0.to_stacks();
    let longest_stack = stacks.into_iter().max_by_key(|(s, _)| s.len())?;
    
    if longest_stack.0.is_empty() {
        return None;
    }

    // This is a rough approximation. The original returned nodes at each step.
    let path_edges: Vec<_> = longest_stack.0.into_iter().map(|edge| (edge, root_node.clone())).collect();
    Some(path_edges)
}

pub fn sample_path(roots: &[&GSSNode], _seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    let mut all_stacks = Vec::new();
    for root in roots {
        all_stacks.extend(root.0.to_stacks());
    }
    if all_stacks.is_empty() {
        return None;
    }
    // Just return the first stack for simplicity. A real random sample would be better.
    Some(all_stacks.into_iter().next().unwrap().0)
}

#[derive(Default)]
pub struct GSSPrintConfig<'a> {
    pub labels: Option<&'a [String]>,
    pub max_edges: usize, // Interpreted as max stacks to print
    pub original_internal_bimap: Option<&'a BTreeMap<usize, usize>>,
    pub llm_token_map: Option<&'a BiBTreeMap<Vec<u8>, LLMTokenID>>,
    pub verbose: bool,
}

pub fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    _terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    config: &GSSPrintConfig,
) -> (String, Vec<StateID>) {
    let mut out_str = String::new();
    let mut state_ids_in_order = Vec::new();

    writeln!(&mut out_str, "GSS Forest (LeveledGSS):").unwrap();

    for (i, root) in roots.iter().enumerate() {
        let label = config.labels.map_or_else(|| format!("Root {}", i), |l| l[i].clone());
        writeln!(&mut out_str, "{}:", label).unwrap();
        
        let stacks = root.0.to_stacks();
        for (j, (stack, acc)) in stacks.iter().take(config.max_edges).enumerate() {
            let stack_str: Vec<_> = stack.iter().map(|e| e.state_id.0.to_string()).collect();
            for edge in stack {
                state_ids_in_order.push(edge.state_id);
            }
            // Simplified format_acc for LeveledGSS
            let acc_str = format!("[LLM({}), Trie({})]", acc.llm_tokens_union.len(), acc.stored_trie_nodes.len());
            writeln!(&mut out_str, "  Stack {}: [{}] -> {}", j, stack_str.join(", "), acc_str).unwrap();
        }
        if stacks.len() > config.max_edges {
            writeln!(&mut out_str, "  ... ({} more stacks truncated)", stacks.len() - config.max_edges).unwrap();
        }
    }
    state_ids_in_order.sort();
    state_ids_in_order.dedup();
    (out_str, state_ids_in_order)
}

pub fn is_simple_gss(
    node: &Arc<GSSNode>,
    hallucinated_state_id: StateID,
) -> Option<(StateID, Arc<Acc>)> {
    let stacks = node.0.to_stacks();
    if stacks.len() == 1 {
        let (stack, acc) = &stacks[0];
        if stack.len() == 2 {
            // Stacks in LeveledGSS are top-to-bottom, so [top, bottom]
            let (outer, inner) = (&stack[0], &stack[1]);
            if inner.state_id == hallucinated_state_id {
                return Some((outer.state_id, Arc::new(acc.clone())));
            }
        }
    }
    None
}

// --- Pruning and Transformation ---

pub type PruneAndTransformRecursiveMemo = HashMap<usize, Arc<GSSNode>>; // Not really used with LeveledGSS

pub fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let new_inner = root_arc.0.apply_and_prune(|acc| {
        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union &= allowed_tokens;
        if new_acc.llm_tokens_union.is_empty() {
            None
        } else {
            Some(new_acc)
        }
    });
    *root_arc = Arc::new(GSSNode(new_inner));
}

pub fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let allowed_mask = HybridBitset::max_ones() - tokens_to_disallow.clone();
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed_mask, memo);
}

pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let new_inner = root_arc.0.apply(|acc| {
        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        new_acc
    });
    *root_arc = Arc::new(GSSNode(new_inner));
}

pub fn reset_terminals(
    root_arc: &mut Arc<GSSNode>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let new_inner = root_arc.0.apply(|acc| {
        let mut new_acc = acc.clone();
        new_acc.terminals_union = HybridL2Bitset::all();
        new_acc
    });
    *root_arc = Arc::new(GSSNode(new_inner));
}

pub fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let new_inner = root_arc.0.apply_and_prune(|acc| {
        let mut new_acc = acc.clone();
        new_acc.terminals_union -= disallowed_terminals;
        // Note: LeveledGSS doesn't have an easy way to check if a path is "dead"
        // due to terminal constraints without a full token walk. We don't prune here.
        Some(new_acc)
    });
    *root_arc = Arc::new(GSSNode(new_inner));
}

pub fn prune_llm_tokens_by_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let new_inner = root_arc.0.apply_and_prune(|acc| {
        if acc.terminals_union == HybridL2Bitset::all() {
            return Some(acc.clone());
        }

        let mut forbidden_llm_tokens = LLMTokenBV::zeros();
        let disallowed_terminals_l2 = acc.terminals_union.complement();

        for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
            if disallowed_terminals_for_range.is_empty() { continue; }
            let relevant_possible_matches = possible_matches.range(TokenizerStateID(*tokenizer_state_range.start())..=TokenizerStateID(*tokenizer_state_range.end()));
            for (_tokenizer_state_id, possible_matches_for_state) in relevant_possible_matches {
                for (terminal_id, llm_tokens_that_match_this_terminal) in possible_matches_for_state {
                    if disallowed_terminals_for_range.contains(terminal_id.0) {
                        forbidden_llm_tokens |= llm_tokens_that_match_this_terminal;
                    }
                }
            }
        }

        if forbidden_llm_tokens.is_empty() {
            return Some(acc.clone());
        }

        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union -= &forbidden_llm_tokens;

        if new_acc.llm_tokens_union.is_empty() { None } else { Some(new_acc) }
    });
    *root_arc = Arc::new(GSSNode(new_inner));
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let new_inner = root_arc.0.prune(|acc| {
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_union = acc.terminals_union.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_union) {
                return false; // Prune this path
            }
        }
        true
    });
    *root_arc = Arc::new(GSSNode(new_inner));
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let new_inner = root_arc.0.apply(|acc| {
        let mut new_terminals_btreemap = BTreeMap::new();
        for (old_state_id, new_state_id) in map {
            let bv_source = acc.terminals_union.get_l2_bitset(old_state_id.0).unwrap();
            new_terminals_btreemap.entry(*new_state_id)
                .and_modify(|bv| *bv |= bv_source)
                .or_insert_with(|| bv_source.clone());
        }

        let mut new_terminals_l2_bitset = HybridL2Bitset::all();
        for (state_id, bv) in new_terminals_btreemap {
            new_terminals_l2_bitset.insert_l2_bitset(state_id.0, bv);
        }

        let mut new_acc = acc.clone();
        new_acc.terminals_union = new_terminals_l2_bitset;
        new_acc
    });
    *root_arc = Arc::new(GSSNode(new_inner));
}


// --- Trie Utils ---

pub fn merge_stored_trie_nodes(
    _root_arc: &mut Arc<GSSNode>,
    _memo: &mut PruneAndTransformRecursiveMemo,
    _stored_trie_god: &StoredTrieGodWrapper,
) {
    // TODO: LeveledGSS - This function is complex and relies on graph traversal.
    // A full reimplementation is required. For now, it's a no-op.
}

#[time_it]
pub(crate) fn deep_add_precompute_trie_edges(
    _root_arc: &mut Arc<GSSNode>,
    _god: &StoredTrieGodWrapper,
    _edge_key: &(usize, LLMTokenBV),
    _edge_value: &StateIDBV,
    _tokens_for_update: &LLMTokenBV,
    _destination_provider: &mut impl FnMut() -> StoredPrecomputeNodeIndex,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    // TODO: LeveledGSS - This function is complex and relies on graph traversal.
    // A full reimplementation is required. For now, it's a no-op.
}
