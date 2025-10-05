use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use bimap::BiBTreeMap;
use crate::glr::grammar::Terminal;
use crate::glr::parser::ParseStateEdgeContent;
use crate::glr::table::StateID;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;
use crate::datastructures::leveled_gss::{LeveledGSS, Merge};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, TerminalBV};

// --- Accumulator Definition (adapted from original gss.rs) ---

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    pub stored_trie_nodes: BTreeSet<PrecomputeNode3Index>,
}

impl Acc {
    pub fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }

    pub fn new_dead() -> Self {
        Self {
            llm_tokens_union: HybridBitset::zeros(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }
    
    pub fn merge(lhs: &Self, rhs: &Self) -> Self {
        Acc {
            llm_tokens_union: &lhs.llm_tokens_union | &rhs.llm_tokens_union,
            terminals_union: &lhs.terminals_union | &rhs.terminals_union,
            stored_trie_nodes: &lhs.stored_trie_nodes | &rhs.stored_trie_nodes,
        }
    }

    pub fn union_llm_tokens(&self) -> HybridBitset { self.llm_tokens_union.clone() }
    pub fn stored_trie_nodes(&self) -> &BTreeSet<PrecomputeNode3Index> { &self.stored_trie_nodes }
    pub fn stored_trie_nodes_mut(&mut self) -> &mut BTreeSet<PrecomputeNode3Index> { &mut self.stored_trie_nodes }
}

impl Merge for Arc<Acc> {
    fn merge(&self, other: &Self) -> Self {
        if Arc::ptr_eq(self, other) {
            return self.clone();
        }
        Arc::new(Acc::merge(self, other))
    }
}

// --- GSSNode Wrapper ---

#[derive(Clone, Debug)]
pub struct GSSNode {
    pub inner: LeveledGSS<ParseStateEdgeContent, Arc<Acc>>,
}

pub type PruneAndTransformRecursiveMemo = (); // Memoization is handled inside LeveledGSS apply/prune

impl GSSNode {
    pub fn new_fresh() -> Self {
        Self { inner: LeveledGSS::empty() }
    }

    pub fn new(acc: Acc) -> Self {
        let stacks = vec![(vec![], Arc::new(acc))];
        Self { inner: LeveledGSS::from_stacks(&stacks) }
    }
    
    pub fn new_dead() -> Self {
        Self::new(Acc::new_dead())
    }

    pub fn push(&self, edge_value: ParseStateEdgeContent) -> Self {
        Self { inner: self.inner.push(edge_value) }
    }
    
    pub fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Self {
        if edge_values.is_empty() {
            return Self::new_fresh();
        }
        let mut merged = self.push(edge_values[0].clone());
        for edge in edge_values.iter().skip(1) {
            merged = merged.merge(&self.push(edge.clone()));
        }
        merged
    }

    pub fn popn(&self, n: usize) -> Self {
        Self { inner: self.inner.popn(n as isize) }
    }
    
    pub fn merge(&self, other: &Self) -> Self {
        Self { inner: self.inner.merge(&other.inner) }
    }
    
    pub fn merge_many_with_depth(_depth: usize, nodes: impl IntoIterator<Item=Arc<GSSNode>>) -> Arc<GSSNode> {
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
        self.inner.is_empty()
    }
    
    pub fn is_alive(&self) -> bool {
        if self.is_empty() { return false; }
        self.allowed_llm_tokens().is_any()
    }
    
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV {
        self.inner.reduce_acc().map_or(LLMTokenBV::zeros(), |acc| acc.llm_tokens_union.clone())
    }
    
    pub fn acc(&self) -> Arc<Acc> {
        self.inner.reduce_acc().unwrap_or_else(|| Arc::new(Acc::new_dead()))
    }

    pub fn to_stacks(&self) -> Vec<(Vec<ParseStateEdgeContent>, Arc<Acc>)> {
        self.inner.to_stacks()
    }
}

// --- Helper Functions Re-implemented for LeveledGSS ---

pub fn allow_only_llm_tokens_and_prune_arc(root_arc: &mut Arc<GSSNode>, allowed_tokens: &LLMTokenBV, _memo: &mut PruneAndTransformRecursiveMemo) {
    let mutator = |acc: &Arc<Acc>| {
        let mut new_acc = (**acc).clone();
        new_acc.llm_tokens_union &= allowed_tokens;
        if new_acc.llm_tokens_union.is_empty() {
            None
        } else {
            Some(Arc::new(new_acc))
        }
    };
    let new_inner = root_arc.inner.apply_and_prune(mutator);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub fn disallow_terminals_and_prune_arc(root_arc: &mut Arc<GSSNode>, disallowed_terminals: &HybridL2Bitset, _memo: &mut PruneAndTransformRecursiveMemo) {
    let mutator = |acc: &Arc<Acc>| {
        let mut new_acc = (**acc).clone();
        new_acc.terminals_union -= disallowed_terminals;
        Some(Arc::new(new_acc))
    };
    let new_inner = root_arc.inner.apply(mutator);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub fn reset_llm_tokens(root_arc: &mut Arc<GSSNode>, _memo: &mut PruneAndTransformRecursiveMemo) {
    let mutator = |acc: &Arc<Acc>| {
        let mut new_acc = (**acc).clone();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        Arc::new(new_acc)
    };
    let new_inner = root_arc.inner.apply(mutator);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub fn reset_terminals(root_arc: &mut Arc<GSSNode>, _memo: &mut PruneAndTransformRecursiveMemo) {
    let mutator = |acc: &Arc<Acc>| {
        let mut new_acc = (**acc).clone();
        new_acc.terminals_union = HybridL2Bitset::all();
        Arc::new(new_acc)
    };
    let new_inner = root_arc.inner.apply(mutator);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub fn prune_disallowed_terminals(root_arc: &mut Arc<GSSNode>, matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>, _memo: &mut PruneAndTransformRecursiveMemo) {
    let predicate = |acc: &Arc<Acc>| -> bool {
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_union = acc.terminals_union.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_union) {
                return false; // Prune
            }
        }
        true // Keep
    };
    let new_inner = root_arc.inner.prune(predicate);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub fn map_allowed_terminals_tokenizer_states(root_arc: &mut Arc<GSSNode>, map: &BTreeMap<TokenizerStateID, TokenizerStateID>, _memo: &mut PruneAndTransformRecursiveMemo) {
    let map_one = |terminals: &HybridL2Bitset| -> HybridL2Bitset {
        let mut new_terminals_btreemap = BTreeMap::new();
        for (old_state_id, new_state_id) in map {
            let bv_source = terminals.get_l2_bitset(old_state_id.0).unwrap();
            new_terminals_btreemap.entry(*new_state_id)
                .and_modify(|bv| *bv |= bv_source)
                .or_insert_with(|| bv_source.clone());
        }
        let mut new_terminals_l2_bitset = HybridL2Bitset::all();
        for (state_id, bv) in new_terminals_btreemap {
            new_terminals_l2_bitset.insert_l2_bitset(state_id.0, bv);
        }
        new_terminals_l2_bitset
    };

    let mutator = |acc: &Arc<Acc>| {
        let mut new_acc = (**acc).clone();
        new_acc.terminals_union = map_one(&acc.terminals_union);
        Arc::new(new_acc)
    };
    let new_inner = root_arc.inner.apply(mutator);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub fn prune_llm_tokens_by_disallowed_terminals(root_arc: &mut Arc<GSSNode>, possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>, _memo: &mut PruneAndTransformRecursiveMemo) {
    let mutator = |acc: &Arc<Acc>| -> Option<Arc<Acc>> {
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
        let mut new_acc = (**acc).clone();
        new_acc.llm_tokens_union -= &forbidden_llm_tokens;
        if new_acc.llm_tokens_union.is_empty() { None } else { Some(Arc::new(new_acc)) }
    };
    let new_inner = root_arc.inner.apply_and_prune(mutator);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

// --- Analysis and Debugging ---
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GSSStats {
    pub num_stacks: usize,
    pub max_depth: usize,
    pub average_depth: f64,
}

pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut total_depth = 0;
    let mut max_depth = 0;
    let mut num_stacks = 0;

    for root in roots {
        let stacks = root.to_stacks();
        num_stacks += stacks.len();
        for (stack, _) in stacks {
            total_depth += stack.len();
            if stack.len() > max_depth {
                max_depth = stack.len();
            }
        }
    }
    
    GSSStats {
        num_stacks,
        max_depth,
        average_depth: if num_stacks > 0 { total_depth as f64 / num_stacks as f64 } else { 0.0 },
    }
}

pub use crate::datastructures::gss::GSSPrintConfig;
pub fn print_gss_forest(roots: &[Arc<GSSNode>], _terminal_map: &BiBTreeMap<Terminal, TerminalID>, config: &GSSPrintConfig) -> (String, Vec<StateID>) {
    let mut out = String::new();
    let mut all_state_ids = BTreeSet::new();
    for (i, root) in roots.iter().enumerate() {
        let root_label = config.labels.map_or_else(|| format!("Root {}", i), |l| l[i].clone());
        out.push_str(&format!("{}:\n", root_label));
        let stacks = root.to_stacks();
        for (j, (stack, acc)) in stacks.iter().take(10).enumerate() {
            let stack_str: Vec<String> = stack.iter().map(|s| s.state_id.0.to_string()).collect();
            for s in stack { all_state_ids.insert(s.state_id); }
            out.push_str(&format!("  Stack {}: [{}] -> {:?}\n", j, stack_str.join(", "), acc.llm_tokens_union.len()));
        }
        if stacks.len() > 10 {
            out.push_str(&format!("  ... ({} more stacks)\n", stacks.len() - 10));
        }
    }
    (out, all_state_ids.into_iter().collect())
}

pub fn get_roots(nodes: impl IntoIterator<Item = &GSSNode>) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
    let mut results = BTreeMap::new();
    for node in nodes {
        for (mut stack, acc) in node.to_stacks() {
            if let Some(edge) = stack.pop() {
                results.entry(edge).or_default().insert(acc);
            }
        }
    }
    results
}

pub fn simplify(states: &mut BTreeMap<TokenizerStateID, Arc<GSSNode>>) {
    let mut all_stacks: Vec<(TokenizerStateID, Vec<ParseStateEdgeContent>, Arc<Acc>)> = Vec::new();
    for (sid, gss) in states.iter() {
        for (stack, acc) in gss.to_stacks() {
            all_stacks.push((*sid, stack, acc));
        }
    }

    let mut stacks_by_sid: BTreeMap<TokenizerStateID, Vec<(Vec<ParseStateEdgeContent>, Arc<Acc>)>> = BTreeMap::new();
    for (sid, stack, acc) in all_stacks {
        stacks_by_sid.entry(sid).or_default().push((stack, acc));
    }
    
    for (sid, stacks) in stacks_by_sid {
        states.insert(sid, Arc::new(GSSNode { inner: LeveledGSS::from_stacks(&stacks) }));
    }
}

pub fn fuse_predecessors_recursive(node_arc: &Arc<GSSNode>, _levels: usize, _memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>) -> Arc<GSSNode> {
    // LeveledGSS is already maximally fused/merged. This is a no-op.
    node_arc.clone()
}

pub fn sample_path<'a>(roots: &[&'a GSSNode], seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let all_stacks: Vec<_> = roots.iter().flat_map(|r| r.to_stacks()).collect();
    if all_stacks.is_empty() {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let (stack, _) = all_stacks[rng.gen_range(0..all_stacks.len())].clone();
    Some(stack)
}
