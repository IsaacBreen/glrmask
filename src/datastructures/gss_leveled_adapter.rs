use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bimap::BiBTreeMap;
use profiler_macro::time_it;
use crate::constraint::StateIDBV;
use crate::constraint::{LLMTokenBV, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::datastructures::leveled_gss::{LeveledGSS, Merge as LGMerge};
use crate::datastructures::trie::Trie2Index;
use crate::glr::grammar::Terminal;
use crate::glr::parser::{GLRParser, ParseStateEdgeContent};
use crate::glr::table::{StateID, TerminalID};
use crate::hit;
use crate::tokenizer::LLMTokenID;

// Adapter aliases for precompute-trie types (referencing constraint.rs)
pub type StoredPrecomputeNodeIndex = crate::constraint::PrecomputeNode3Index;
pub type StoredTrieGodWrapper = crate::constraint::Trie3GodWrapper;

// --- Acc type compatible with LeveledGSS A ---
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    pub stored_trie_nodes: BTreeSet<StoredPrecomputeNodeIndex>,
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
    pub fn is_default(&self) -> bool {
        self.llm_tokens_union == HybridBitset::max_ones()
            && self.terminals_union == HybridL2Bitset::all()
            && self.stored_trie_nodes.is_empty()
    }
    pub fn union_llm_tokens(&self) -> HybridBitset {
        self.llm_tokens_union.clone()
    }

    pub fn stored_trie_nodes(&self) -> &BTreeSet<StoredPrecomputeNodeIndex> {
        &self.stored_trie_nodes
    }

    pub fn stored_trie_nodes_mut(&mut self) -> &mut BTreeSet<StoredPrecomputeNodeIndex> {
        &mut self.stored_trie_nodes
    }
}

impl LGMerge for Acc {
    fn merge(&self, other: &Self) -> Self {
        hit!("LGMerge for Acc");
        Acc {
            llm_tokens_union: &self.llm_tokens_union | &other.llm_tokens_union,
            terminals_union: &self.terminals_union | &other.terminals_union,
            stored_trie_nodes: &self.stored_trie_nodes | &other.stored_trie_nodes,
        }
    }
}

// --- Minimal GSSStats for logging/debug ---
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

#[time_it]
pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();
    let mut total_depth: usize = 0;
    let mut paths: usize = 0;
    let mut max_depth = 0;
    let mut total_edges = 0;
    let mut num_root_predecessors = 0usize;
    let mut unique_root_keys = BTreeSet::new();
    for r in roots {
        let info = r.inner.paths_info();
        paths += info.num_paths;
        total_depth += info.total_depth;
        total_edges += info.total_depth;
        max_depth = max_depth.max(r.inner.max_depth() as usize);

        if let Some(peek) = r.inner.peek().into_iter().next() {
            unique_root_keys.insert(peek);
        }
        num_root_predecessors += r.inner.peek().len();
    }
    stats.max_depth = max_depth;
    stats.total_edges = total_edges;
    stats.unique_nodes = paths;
    stats.num_leaves = paths;
    stats.num_root_predecessors = num_root_predecessors;
    stats.num_unique_root_predecessor_keys = unique_root_keys.len();
    if paths > 0 {
        stats.average_depth = total_depth as f64 / paths as f64;
        stats.average_predecessors_with_values = (num_root_predecessors as f64) / (roots.len().max(1) as f64);
    }
    stats
}

// --- GSS printer config & helpers ---
pub struct GSSPrintConfig<'a> {
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

pub fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    terminal_map: &BiBTreeMap<crate::glr::grammar::Terminal, TerminalID>,
    config: &GSSPrintConfig,
) -> (String, Vec<StateID>) {
    let mut out = String::new();
    let mut state_ids_in_order = Vec::new();
    if roots.is_empty() {
        return ("GSS Forest: (No roots)".to_string(), state_ids_in_order);
    }
    writeln!(&mut out, "GSS Forest (leveled adapter):").unwrap();
    for (i, r) in roots.iter().enumerate() {
        // Use label if available
        if let Some(labels) = config.labels {
            assert!(labels.len() == roots.len());
            writeln!(&mut out, "{}", labels[i]).unwrap();
        } else {
            writeln!(&mut out, " Root {}:", i).unwrap();
        }
        let stacks = r.inner.to_stacks();
        for (path, acc) in stacks {
            let sids: Vec<_> = path.iter().map(|e| e.state_id).collect();
            for sid in &sids {
                if !state_ids_in_order.contains(sid) {
                    state_ids_in_order.push(*sid);
                }
            }
            let s: Vec<_> = sids.iter().map(|s| s.0.to_string()).collect();
            let acc_str = format_acc(
                &acc,
                terminal_map,
                config.original_internal_bimap,
                config.llm_token_map,
                config,
            );
            if acc_str.is_empty() {
                writeln!(&mut out, "  - [{}]", s.join(" ")).unwrap();
            } else {
                writeln!(&mut out, "  - [{}] {}", s.join(" "), acc_str).unwrap();
            }
        }
    }
    (out, state_ids_in_order)
}

pub fn find_longest_path(root: &Arc<GSSNode>) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    root.inner
        .get_longest_path()
        .map(|(p, _a)| p.into_iter().map(|e| (e, root.clone())).collect())
}

pub fn sample_path<'a>(roots: &[&'a GSSNode], _seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    roots
        .get(0)
        .map(|r| r.inner.get_first_path().map(|(p, _)| p).unwrap_or_default())
}

// --- GSS wrapper ---
#[derive(Clone)]
pub struct GSSNode {
    pub(crate) inner: LeveledGSS<ParseStateEdgeContent, Acc>,
}

impl Debug for GSSNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GSSNode(len_paths={})", self.inner.num_paths())
    }
}

impl PartialEq for GSSNode {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}
impl Eq for GSSNode {}

impl GSSNode {
    pub fn new(acc: Acc) -> Self {
        GSSNode {
            inner: LeveledGSS::from_stacks(&vec![(vec![], acc)]),
        }
    }
    pub fn new_fresh() -> Self {
        GSSNode {
            inner: LeveledGSS::from_stacks(&vec![(vec![], Acc::new_fresh())]),
        }
    }
    pub fn new_dead() -> Self {
        // Represent dead as empty; allowed_llm_tokens() on empty returns zeros.
        GSSNode {
            inner: LeveledGSS::empty(),
        }
    }

    // Helper: append a value to all stacks; if empty, create singleton stack.
    fn push_all(
        inner: &LeveledGSS<ParseStateEdgeContent, Acc>,
        edge: ParseStateEdgeContent,
    ) -> LeveledGSS<ParseStateEdgeContent, Acc> {
        if inner.is_empty() {
            LeveledGSS::from_stacks(&[(vec![edge], Acc::new_fresh())])
        } else {
            inner.push(edge)
        }
    }

    pub fn push(&self, edge_value: ParseStateEdgeContent) -> Self {
        GSSNode {
            inner: Self::push_all(&self.inner, edge_value),
        }
    }
    pub fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Self {
        if edge_values.is_empty() {
            return self.clone();
        }
        let mut merged = LeveledGSS::empty();
        for e in edge_values {
            let next = Self::push_all(&self.inner, e);
            merged = merged.merge(&next);
        }
        GSSNode { inner: merged }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty() || self.inner.max_depth() == 0
    }
    pub fn is_alive(&self) -> bool {
        !self.is_empty() && !self.allowed_llm_tokens().is_empty()
    }
    pub fn is_ok(&self) -> bool {
        self.is_alive()
    }
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV {
        self.inner
            .reduce_acc()
            .map_or(HybridBitset::zeros(), |acc| acc.llm_tokens_union)
    }
    pub fn disallowed_terminals(&self) -> HybridL2Bitset {
        self.inner
            .reduce_acc()
            .map(|acc| acc.terminals_union.complement())
            .unwrap_or_else(|| HybridL2Bitset::all().complement())
    }
    #[time_it]
    pub fn stored_trie_nodes(&self) -> BTreeSet<StoredPrecomputeNodeIndex> {
        self.inner
            .reduce_acc()
            .map_or_else(BTreeSet::new, |acc| acc.stored_trie_nodes.clone())
    }
    pub fn max_depth(&self) -> usize {
        self.inner.max_depth() as usize
    }
    pub fn flatten(&self) -> Vec<(Vec<ParseStateEdgeContent>, Acc)> {
        self.inner.to_stacks()
    }

    pub fn print(&self) -> String {
        let (s, _sid) = print_gss_forest(&[Arc::new(self.clone())], &BiBTreeMap::new(), &GSSPrintConfig::default());
        s
    }

    pub fn merge_with_depth(&mut self, _merge_depth: usize, other: &Self) {
        self.inner = self.inner.merge(&other.inner);
    }
    pub fn merged(mut self, other: Self, depth: usize) -> Self {
        self.merge_with_depth(depth, &other);
        self
    }

    pub fn merge_many_with_depth(merge_depth: usize, nodes: impl IntoIterator<Item = Arc<GSSNode>>) -> Arc<GSSNode> {
        let mut it = nodes.into_iter();
        if let Some(first) = it.next() {
            let mut acc = (*first).clone();
            for n in it {
                acc.merge_with_depth(merge_depth, &n);
            }
            Arc::new(acc)
        } else {
            Arc::new(GSSNode::new_dead())
        }
    }

    pub fn predecessors(&self) -> BTreeMap<ParseStateEdgeContent, BTreeMap<isize, Vec<Arc<GSSNode>>>> {
        self.inner
            .predecessors()
            .into_iter()
            .map(|(edge_val, preds_by_depth)| {
                let new_preds_by_depth = preds_by_depth
                    .into_iter()
                    .map(|(depth, gss_vec)| {
                        let new_gss_vec = gss_vec.into_iter().map(|gss| Arc::new(GSSNode { inner: gss })).collect();
                        (depth, new_gss_vec)
                    })
                    .collect();
                (edge_val, new_preds_by_depth)
            })
            .collect()
    }

    pub fn num_predecessors(&self) -> usize {
        self.inner.peek().len()
    }

    pub fn popn(&self, n: usize) -> GSSPopper {
        let mut popper = GSSPopper {
            inner: self.inner.clone(),
            below_bottom: BTreeMap::new(),
        };
        popper.popn(n);
        popper
    }

    pub(crate) fn peek_iter(parent_arc: &Arc<GSSNode>) -> impl Iterator<Item = GSSPeek<'_>> {
        let keys: Vec<_> = parent_arc.inner.peek().into_iter().collect();
        GSSPeekIter {
            parent: &parent_arc.inner,
            keys,
            idx: 0,
        }
    }

    #[time_it]
    pub fn fuse_predecessors(&mut self, levels: usize) {
        if levels == 0 {
            return;
        }
        let self_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let fused_arc = fuse_predecessors_recursive(&self_arc, levels, &mut memo);
        if !Arc::ptr_eq(&self_arc, &fused_arc) {
            *self = (*fused_arc).clone();
        }
    }

    pub fn fuse_predecessors_recursive(&self, levels: usize, _memo: &mut PruneAndTransformRecursiveMemo) -> Arc<GSSNode> {
        let self_arc = Arc::new(self.clone());
        let mut dummy_memo = HashMap::new();
        fuse_predecessors_recursive(&self_arc, levels, &mut dummy_memo)
    }
}

// --- GSSPeek & iterator (updated to borrow LeveledGSS directly) ---
pub struct GSSPeek<'a> {
    parent_arc: &'a LeveledGSS<ParseStateEdgeContent, Acc>,
    edge_value: ParseStateEdgeContent,
}
struct GSSPeekIter<'a> {
    parent: &'a LeveledGSS<ParseStateEdgeContent, Acc>,
    keys: Vec<ParseStateEdgeContent>,
    idx: usize,
}
impl<'a> Iterator for GSSPeekIter<'a> {
    type Item = GSSPeek<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.keys.len() {
            None
        } else {
            let ev = self.keys[self.idx].clone();
            self.idx += 1;
            Some(GSSPeek {
                parent_arc: self.parent,
                edge_value: ev,
            })
        }
    }
}
impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &ParseStateEdgeContent {
        &self.edge_value
    }
    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        let iso = self.parent_arc.isolate(Some(self.edge_value.clone()));
        Arc::new(GSSNode { inner: iso })
    }
    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let iso = self.isolated_parent();
        iso.as_ref().push(edge_value)
    }
    pub fn popn(&self, n: usize) -> GSSPopper {
        self.isolated_parent().popn(n)
    }
    pub fn resolved_llm_tokens_union(&self) -> LLMTokenBV {
        self.isolated_parent().allowed_llm_tokens()
    }
}

// --- Popper ---
pub struct GSSPopper {
    inner: LeveledGSS<ParseStateEdgeContent, Acc>,
    below_bottom: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Acc>>,
}
pub struct GSSPopperItem {
    inner: LeveledGSS<ParseStateEdgeContent, Acc>,
}
pub struct GSSPopperItemPeek<'a> {
    parent_arc: &'a LeveledGSS<ParseStateEdgeContent, Acc>,
    edge_value: &'a ParseStateEdgeContent,
}

impl GSSPopper {
    pub fn new_from_node(node: Arc<GSSNode>, _acc: Arc<Acc>) -> Self {
        GSSPopper { inner: node.inner.clone(), below_bottom: BTreeMap::new() }
    }
    pub fn iter(&self) -> impl Iterator<Item = GSSPopperItem> {
        let keys: Vec<_> = self.inner.peek().into_iter().collect();
        let parent = self.inner.clone();
        keys.into_iter().map(move |edge| {
            let iso = parent.isolate(Some(edge));
            GSSPopperItem { inner: iso }
        })
    }
    pub fn below_bottom(&self) -> &BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Acc>> {
        &self.below_bottom
    }
    pub fn num_predecessors(&self) -> usize {
        self.inner.peek().len()
    }
    pub fn popn(&mut self, n: usize) {
        for _ in 0..n {
            self.pop();
        }
    }
    pub fn pop(&mut self) {
        let mut inner = self.inner.clone();
        let mut belows: BTreeMap<_, _> = self.below_bottom.iter().map(|(k, v)| (*k + 1, v.clone())).collect();
        let new_below_slice = inner.filter_by_length(Some(1), Some(1));
        if !new_below_slice.is_empty() {
            let mut new_below_map = BTreeMap::new();
            for edge in new_below_slice.peek() {
                let isolated = new_below_slice.isolate(Some(edge.clone()));
                if let Some(acc) = isolated.reduce_acc() {
                    new_below_map.insert(edge, acc);
                }
            }
            if !new_below_map.is_empty() {
                belows.insert(1, new_below_map);
            }
        }
        self.below_bottom = belows;
        inner = inner.pop();
        inner = inner.filter_by_length(Some(1), None);
        self.inner = inner;
    }
}

impl GSSPopperItem {
    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPopperItemPeek<'_>> {
        let keys: Vec<_> = self.inner.peek().into_iter().collect();
        GSSPopperItemPeekIter {
            parent: &self.inner,
            keys,
            idx: 0,
        }
    }
    pub(crate) fn resolved_acc(&self) -> Acc {
        self.inner.reduce_acc().unwrap_or_else(Acc::new_dead)
    }
}
struct GSSPopperItemPeekIter<'a> {
    parent: &'a LeveledGSS<ParseStateEdgeContent, Acc>,
    keys: Vec<ParseStateEdgeContent>,
    idx: usize,
}
impl<'a> Iterator for GSSPopperItemPeekIter<'a> {
    type Item = GSSPopperItemPeek<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.keys.len() {
            None
        } else {
            let ev = &self.keys[self.idx];
            self.idx += 1;
            Some(GSSPopperItemPeek {
                parent_arc: self.parent,
                edge_value: unsafe { &*(ev as *const ParseStateEdgeContent) },
            })
        }
    }
}
impl<'a> GSSPopperItemPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent {
        self.edge_value
    }
    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        let iso = self.parent_arc.isolate(Some(self.edge_value.clone()));
        Arc::new(GSSNode { inner: iso })
    }
    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let iso = self.isolated_parent();
        iso.as_ref().push(edge_value)
    }
}

// --- Roots & helpers ---
#[time_it]
pub fn get_roots<'a>(nodes: impl IntoIterator<Item = &'a GSSNode>) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
    let mut out: BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> = BTreeMap::new();
    for n in nodes {
        for edge in n.inner.peek() {
            let iso = n.inner.isolate(Some(edge.clone()));
            if let Some(acc) = iso.reduce_acc() {
                out.entry(edge).or_default().insert(Arc::new(acc));
            }
        }
    }
    out
}

// --- Transformations (simplified) ---
fn transform_all(root_arc: &mut Arc<GSSNode>, f: impl FnMut(&Acc) -> Option<Acc>) {
    let new_inner = root_arc.inner.apply_and_prune(f);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub type PruneAndTransformRecursiveMemo = HashMap<*const GSSNode, Option<Arc<GSSNode>>>;

pub fn allow_only_llm_tokens_and_prune(root_arc: &mut Arc<GSSNode>, allowed_tokens: &LLMTokenBV) {
    let mut memo = HashMap::new();
    allow_only_llm_tokens_and_prune_arc(root_arc, allowed_tokens, &mut memo);
}

pub(crate) fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        na.llm_tokens_union &= allowed_tokens;
        if na.llm_tokens_union.is_empty() {
            None
        } else {
            Some(na)
        }
    });
}

pub(crate) fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let allowed = HybridBitset::max_ones() - tokens_to_disallow.clone();
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed, memo);
}

pub fn reset_llm_tokens(root_arc: &mut Arc<GSSNode>, _memo: &mut PruneAndTransformRecursiveMemo) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        na.llm_tokens_union = HybridBitset::max_ones();
        Some(na)
    });
}

pub(crate) fn reset_terminals(root_arc: &mut Arc<GSSNode>, _memo: &mut PruneAndTransformRecursiveMemo) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        na.terminals_union = HybridL2Bitset::all();
        Some(na)
    });
}

pub(crate) fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        println!("Disallowing terminals: {:?}", disallowed_terminals);
        println!("Before: {:?}", na.terminals_union);
        na.terminals_union -= disallowed_terminals;
        println!("After: {:?}", na.terminals_union);
        Some(na)
    });
}

pub fn prune_llm_tokens_by_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    possible_matches: &BTreeMap<crate::tokenizer::TokenizerStateID, BTreeMap<crate::types::TerminalID, LLMTokenBV>>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    // Very simplified: keep as-is (no change). Implement exact logic later.
    let _ = possible_matches;
    let _ = root_arc;
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<crate::tokenizer::TokenizerStateID, TerminalBV>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let matched = matched_terminals.clone();
    let all = HybridBitset::max_ones();
    transform_all(root_arc, |a| {
        for (sid, bv) in &matched {
            let allowed = a.terminals_union.get_l2_bitset(sid.0).unwrap_or(&all);
            if !bv.is_subset(allowed) {
                return None;
            }
        }
        Some(a.clone())
    });
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<crate::tokenizer::TokenizerStateID, crate::tokenizer::TokenizerStateID>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mapping = map.clone();
    transform_all(root_arc, |a| {
        let mut new_map = BTreeMap::new();
        for (old, new) in &mapping {
            if let Some(bv) = a.terminals_union.get_l2_bitset(old.0) {
                new_map
                    .entry(new.0)
                    .and_modify(|b: &mut HybridBitset| *b |= bv.clone())
                    .or_insert_with(|| bv.clone());
            }
        }
        let mut out = HybridL2Bitset::all();
        for (sid, bv) in new_map {
            out.insert_l2_bitset(sid, bv);
        }
        let mut na = a.clone();
        na.terminals_union = out;
        Some(na)
    });
}

pub(crate) fn allow_only_llm_tokens_on_stored_trie_nodes_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    _memo: &mut PruneAndTransformRecursiveMemo,
    stored_trie_god: &StoredTrieGodWrapper,
) {
    // --- Global Analysis: Find common subsets (macros) ---
    let all_accs: Vec<Acc> = root_arc.inner.to_stacks().into_iter().map(|(_, acc)| acc).collect();
    let mut all_needs_edge_sets = Vec::new();
    for acc in &all_accs {
        let needs_edge: BTreeSet<StoredPrecomputeNodeIndex> = acc
            .stored_trie_nodes()
            .iter()
            .filter(|node| {
                !node
                    .as_arc()
                    .read(stored_trie_god)
                    .expect("poison")
                    .value
                    .live_tokens
                    .is_subset(allowed_tokens)
            })
            .cloned()
            .collect();
        if !needs_edge.is_empty() {
            all_needs_edge_sets.push(needs_edge);
        }
    }

    let mut pair_counts: BTreeMap<BTreeSet<StoredPrecomputeNodeIndex>, usize> = BTreeMap::new();
    for needs_edge_set in &all_needs_edge_sets {
        let nodes: Vec<_> = needs_edge_set.iter().cloned().collect();
        if nodes.len() >= 2 {
            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    let mut pair = BTreeSet::new();
                    pair.insert(nodes[i].clone());
                    pair.insert(nodes[j].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }
        }
    }

    let mut macros: BTreeMap<BTreeSet<StoredPrecomputeNodeIndex>, StoredPrecomputeNodeIndex> = BTreeMap::new();
    for (pair, count) in pair_counts {
        if count > 1 { // Only create macros for pairs that are reused.
            let dest_node = StoredPrecomputeNodeIndex::new(
                stored_trie_god.insert(crate::constraint::PrecomputeNode3::new(
                    crate::constraint::PrecomputedNodeContents::internal(),
                )),
            );
            macros.insert(pair, dest_node);
        }
    }
    let sorted_macros: Vec<_> = macros.into_iter().collect();

    // --- Transformation ---
    let mut new_destinations_for_leftovers = BTreeMap::new();
    transform_all(root_arc, |acc| {
        let mut na = acc.clone();
        let original_nodes = na.stored_trie_nodes.clone();
        let mut needs_edge: BTreeSet<_> = original_nodes
            .iter()
            .filter(|node| {
                !node
                .as_arc()
                .read(stored_trie_god)
                .expect("poison")
                .value
                .live_tokens
                    .is_subset(allowed_tokens)
            })
            .cloned()
            .collect();

        if needs_edge.is_empty() {
            return Some(na);
        }

        let mut final_nodes = original_nodes.difference(&needs_edge).cloned().collect::<BTreeSet<_>>();

        // Greedily apply discovered macros
        for (macro_nodes, macro_dest) in &sorted_macros {
            if macro_nodes.is_subset(&needs_edge) {
                let edge_key = (0, allowed_tokens.clone());
                let edge_value = StateIDBV::max_ones();
                for src in macro_nodes {
                    let inserter = crate::datastructures::trie::EdgeInserter::new(
                        stored_trie_god,
                        src.as_arc().clone(),
                        edge_key.clone(),
                        edge_value.clone(),
                        |e, n| *e |= n,
                        |node_value, _| node_value.live_tokens |= allowed_tokens,
                        |_, _| {},
                    );
                    inserter.try_destination(macro_dest.clone()).expect("Cycle detected");
                }
                macro_dest.write(stored_trie_god).expect("poison").value.live_tokens |= allowed_tokens;
                final_nodes.insert(macro_dest.clone());
                needs_edge = needs_edge.difference(macro_nodes).cloned().collect();
            }
        }

        // Handle leftovers using the original "macro for the whole set" logic
        if !needs_edge.is_empty() {
            let new_dest = new_destinations_for_leftovers
                .entry(needs_edge.clone())
                .or_insert_with(|| {
                    StoredPrecomputeNodeIndex::new(
                        stored_trie_god.insert(crate::constraint::PrecomputeNode3::new(
                            crate::constraint::PrecomputedNodeContents::internal(),
                        )),
                    )
                })
                .clone();

            let edge_key = (0, allowed_tokens.clone());
            let edge_value = StateIDBV::max_ones();
            for src in &needs_edge {
                let inserter = crate::datastructures::trie::EdgeInserter::new(
                    stored_trie_god,
                    src.as_arc().clone(),
                    edge_key.clone(),
                    edge_value.clone(),
                    |e, n| *e |= n,
                    |node_value, _| node_value.live_tokens |= allowed_tokens,
                    |_, _| {},
                );
                inserter.try_destination(new_dest.clone()).expect("Cycle detected");
            }
            new_dest.write(stored_trie_god).expect("poison").value.live_tokens |= allowed_tokens;
            final_nodes.insert(new_dest);
        }

        na.stored_trie_nodes = final_nodes;
        Some(na)
    });
}

pub fn simplify(_states: &mut BTreeMap<crate::tokenizer::TokenizerStateID, Arc<GSSNode>>) {}
pub(crate) fn simplify_roots_in_place(_roots: &mut [Arc<GSSNode>]) {}
pub fn fuse_predecessors_recursive(node_arc: &Arc<GSSNode>, levels: usize, _memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>) -> Arc<GSSNode> {
    let fused_inner = node_arc.inner.fuse(Some(levels as isize));
    if fused_inner.ptr_eq(&node_arc.inner) {
        node_arc.clone()
    } else {
        Arc::new(GSSNode { inner: fused_inner })
    }
}

impl GSSNode {
    pub fn reset_llm_tokens(&mut self) {
        let mut memo = PruneAndTransformRecursiveMemo::new();
        let mut arc = Arc::new(self.clone());
        reset_llm_tokens(&mut arc, &mut memo);
        *self = (*arc).clone();
    }
}

// --- Trie-utils stubs (no-ops) ---
#[time_it]
pub(crate) fn deep_add_precompute_trie_edges(
    root_arc: &mut Arc<GSSNode>,
    god: &StoredTrieGodWrapper,
    edge_key: &(usize, LLMTokenBV),
    edge_value: &StateIDBV,
    tokens_for_update: &LLMTokenBV,
    destination_provider: &mut impl FnMut() -> StoredPrecomputeNodeIndex,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    transform_all(root_arc, |acc| {
        if acc.stored_trie_nodes().is_empty() {
            return Some(acc.clone());
        }

        let destination = destination_provider();

        for source_wrapper in acc.stored_trie_nodes() {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = crate::datastructures::trie::EdgeInserter::new(
                god,
                source_arc,
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= tokens_for_update,
                |_, _| {}, // Unconditional insertion
            );
            inserter.try_destination(destination.clone()).expect("Cycle detected when adding precompute trie edges");
        }

        // destination.write(god).expect("poison").value.live_tokens |= tokens_for_update;

        let mut new_acc = acc.clone();
        *new_acc.stored_trie_nodes_mut() = BTreeSet::from([destination]);
        Some(new_acc)
    });
}

#[time_it]
pub(crate) fn merge_stored_trie_nodes(
    root_arc: &mut Arc<GSSNode>,
    _memo: &mut PruneAndTransformRecursiveMemo,
    stored_trie_god: &StoredTrieGodWrapper,
) {
    let mut new_destinations = BTreeMap::new();

    transform_all(root_arc, |acc| {
        if !acc.stored_trie_nodes().iter().any(
            |n| n.as_arc().read(stored_trie_god).expect("poison").value.live_tokens != acc.llm_tokens_union
        ) {
            return Some(acc.clone());
        }

        let mut new_acc = acc.clone();
        // Create a single new destination for this merge operation.
        let new_destination = new_destinations.entry((new_acc.stored_trie_nodes().clone(), acc.llm_tokens_union.clone()))
            .or_insert_with(|| StoredPrecomputeNodeIndex::new(stored_trie_god.insert(crate::constraint::PrecomputeNode3::new(crate::constraint::PrecomputedNodeContents::internal()))))
            .clone();

        let edge_key = (0, new_acc.llm_tokens_union.clone());
        let edge_value = crate::constraint::StateIDBV::max_ones();
        let tokens_for_edge = new_acc.llm_tokens_union.clone();

        for source_wrapper in new_acc.stored_trie_nodes() {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = crate::datastructures::trie::EdgeInserter::new(
                stored_trie_god,
                source_arc,
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= &tokens_for_edge,
                |_, _| {}, // Unconditional insertion
            );
            inserter.try_destination(new_destination.clone()).expect("Cycle detected when merging stored_trie nodes; this should be impossible.");
        }

        new_destination.write(stored_trie_god).expect("poison").value.live_tokens |= &tokens_for_edge;

        new_acc.stored_trie_nodes = BTreeSet::from([new_destination]);
        Some(new_acc)
    });
}

pub(crate) fn is_simple_gss(
    node: &Arc<GSSNode>,
    hallucinated_state_id: StateID,
) -> Option<(StateID, Arc<Acc>)> {
    if let Some((path, acc)) = node.inner.as_single_path() {
        // Path must have length 2. The path is from leaf to root.
        if path.len() == 2 {
            let edge_to_leaf = &path[0];
            let edge_to_middle = &path[1];

            // The edge into the leaf must be the hallucinated one.
            if edge_to_leaf.state_id == hallucinated_state_id {
                let state_id_x = edge_to_middle.state_id;
                // The leaf must have stored_trie_nodes.
                if !acc.stored_trie_nodes().is_empty() {
                    return Some((state_id_x, Arc::new(acc)));
                }
            }
        }
    }
    None
}

// helper used by parser logging
pub fn popn_collect_isolated_parents(node_arc: &Arc<GSSNode>, n: usize) -> Vec<(StateID, Arc<GSSNode>)> {
    let popped = node_arc.inner.popn(n as isize);
    let mut out = Vec::new();
    for edge in popped.peek() {
        let iso_inner = popped.isolate(Some(edge.clone()));
        let num_paths = iso_inner.num_paths();
        if num_paths > 0 {
            let gss_node = Arc::new(GSSNode { inner: iso_inner });
            for _ in 0..num_paths {
                out.push((edge.state_id, gss_node.clone()));
            }
        }
    }
    out
}

// Compatibility for formatting acc (used by printer)
pub(crate) fn format_acc(
    acc: &Acc,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    config: &GSSPrintConfig,
) -> String {
    let _ = (original_internal_bimap, llm_token_map);

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