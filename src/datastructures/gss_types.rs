use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;
use std::sync::Arc;
use bimap::BiBTreeMap;
use crate::constraint::{LLMTokenBV, PrecomputeNode3, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV, TerminalBV, TerminalInfo, Trie3God, Trie3GodWrapper};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::datastructures::leveled_gss::LeveledGSS;
use crate::datastructures::trie::{EdgeInserter, Trie2Index};
use crate::glr::grammar::Terminal;
use crate::glr::parser::ParseStateEdgeContent;
use crate::glr::table::StateID;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;

pub use crate::datastructures::trie::Trie2Index as StoredPrecomputeNodeIndex;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    pub stored_trie_nodes: BTreeSet<StoredPrecomputeNodeIndex>,
}

impl crate::datastructures::leveled_gss::Merge for Acc {
    fn merge(lhs: &Self, rhs: &Self) -> Self {
        Self {
            llm_tokens_union: &lhs.llm_tokens_union | &rhs.llm_tokens_union,
            terminals_union: &lhs.terminals_union | &rhs.terminals_union,
            stored_trie_nodes: &lhs.stored_trie_nodes | &rhs.stored_trie_nodes,
        }
    }
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

    pub fn narrow(from: &Self, to: &Self) -> Self {
        Acc {
            llm_tokens_union: &from.llm_tokens_union & &to.llm_tokens_union,
            terminals_union: &from.terminals_union & &to.terminals_union,
            stored_trie_nodes: to.stored_trie_nodes.clone(),
        }
    }

    pub fn stored_trie_nodes_mut(&mut self) -> &mut BTreeSet<StoredPrecomputeNodeIndex> {
        &mut self.stored_trie_nodes
    }
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

pub(crate) struct GSSPrintConfig<'a> {
    pub(crate) labels: Option<Vec<String>>,
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

pub(crate) fn format_acc(
    acc: &Acc,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    config: &GSSPrintConfig,
) -> String {
    let _ = (original_internal_bimap, llm_token_map);

    if config.verbose {
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
        let n = acc.stored_trie_nodes.len();
        if n == 0 {
            None
        } else if n <= MAX_PTRS_TO_SHOW {
            let ptrs: Vec<String> = acc
                .stored_trie_nodes
                .iter()
                .map(|wrapper| format!("{}", wrapper))
                .collect();
            Some(format!("Trie(n={}, [{}])", n, ptrs.join(", ")))
        } else {
            let ptrs_sample: Vec<String> = acc
                .stored_trie_nodes
                .iter()
                .take(MAX_PTRS_TO_SHOW)
                .map(|wrapper| format!("{}", wrapper))
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

pub fn reset_llm_tokens(gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>) {
    gss.map_accs(|acc| {
        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union = HybridBitset::max_ones();
        new_acc
    });
}

pub fn reset_terminals(gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>) {
    gss.map_accs(|acc| {
        let mut new_acc = acc.clone();
        new_acc.terminals_union = HybridL2Bitset::all();
        new_acc
    });
}

pub fn allow_only_llm_tokens(gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>, allowed_tokens: &LLMTokenBV) {
    gss.map_accs(|acc| {
        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union &= allowed_tokens;
        new_acc
    });
    gss.prune(|acc| !acc.llm_tokens_union.is_empty());
}

pub fn disallow_llm_tokens(gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>, tokens_to_disallow: &LLMTokenBV) {
    let allowed_mask = HybridBitset::max_ones() - tokens_to_disallow.clone();
    allow_only_llm_tokens(gss, &allowed_mask);
}

pub fn disallow_terminals(gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>, disallowed_terminals: &HybridL2Bitset) {
    gss.map_accs(|acc| {
        let mut new_acc = acc.clone();
        new_acc.terminals_union -= disallowed_terminals;
        new_acc
    });
}

pub fn prune_llm_tokens_by_disallowed_terminals(
    gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>,
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
) {
    gss.map_accs(|acc| {
        if acc.terminals_union == HybridL2Bitset::all() {
            return acc.clone();
        }

        let mut forbidden_llm_tokens = LLMTokenBV::zeros();
        let disallowed_terminals_l2 = acc.terminals_union.complement();

        for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
            if disallowed_terminals_for_range.is_empty() {
                continue;
            }

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
            return acc.clone();
        }

        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union -= &forbidden_llm_tokens;
        new_acc
    });
    gss.prune(|acc| !acc.llm_tokens_union.is_empty());
}

pub fn prune_disallowed_terminals(
    gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>,
    matched_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
) {
    gss.prune(|acc| {
        for (state_id, matched_bv) in matched_terminals {
            let allowed_terminals_union = acc.terminals_union.get_l2_bitset(state_id.0).unwrap();
            if !matched_bv.is_subset(allowed_terminals_union) {
                return false;
            }
        }
        true
    });
}

pub fn map_allowed_terminals_tokenizer_states(
    gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
) {
    gss.map_accs(|acc| {
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
}

pub fn merge_stored_trie_nodes(
    gss: &mut LeveledGSS<ParseStateEdgeContent, Acc>,
    stored_trie_god: &Trie3GodWrapper,
) {
    let mut new_destinations = BTreeMap::new();

    gss.map_accs(|acc| {
        if !acc.stored_trie_nodes.iter().any(
            |n| n.read(stored_trie_god).expect("poison").value.live_tokens != acc.llm_tokens_union
        ) {
            return acc.clone();
        }
        let mut new_acc = acc.clone();
        let new_destination = new_destinations.entry((new_acc.stored_trie_nodes.clone(), acc.llm_tokens_union.clone()))
            .or_insert_with(|| StoredPrecomputeNodeIndex::new(stored_trie_god.insert(PrecomputeNode3::new(PrecomputedNodeContents::internal()))))
            .clone();
        let edge_key = (0, new_acc.llm_tokens_union.clone());
        let edge_value = StateIDBV::max_ones();
        let tokens_for_edge = new_acc.llm_tokens_union.clone();

        for source_wrapper in &new_acc.stored_trie_nodes {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = EdgeInserter::new(
                &stored_trie_god,
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
        new_acc
    });
}

pub fn is_simple_gss(
    gss: &LeveledGSS<ParseStateEdgeContent, Acc>,
    hallucinated_state_id: StateID,
) -> Option<(StateID, Acc)> {
    let mut stacks = Vec::new();
    gss.stacks_for_each(|stack, acc| {
        stacks.push((stack.to_vec(), acc.clone()));
    });

    if stacks.len() == 1 {
        let (path, acc) = &stacks[0];
        // Path is from leaf to root, so we check in reverse order of structure
        if path.len() == 2 {
            let first_edge = &path[0]; // Corresponds to hallucinated_id edge
            let second_edge = &path[1]; // Corresponds to state_id edge

            if first_edge.state_id == hallucinated_state_id {
                if !acc.stored_trie_nodes.is_empty() {
                    return Some((second_edge.state_id, acc.clone()));
                }
            }
        }
    }
    None
}
