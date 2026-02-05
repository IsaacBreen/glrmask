//! Terminal DWA construction.
//!
//! This module builds the Terminal DWA from the tokenizer and LLM vocabulary.
//!
//! The Terminal DWA encodes which LLM tokens can be generated in each tokenizer state.
//! It's called "Terminal" because it handles the terminal symbols of the grammar -
//! specifically, how LLM tokens map to grammar terminals via the tokenizer.
//!
//! This is distinct from "Template DFAs" (in precompute4/template_dfa.rs) which encode
//! how each terminal type interacts with the parser stack.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::BitOrAssign;
use std::sync::Arc;
use bimap::BiBTreeMap;
use range_set_blaze::RangeSetBlaze;
use profiler_macro::{time_it, timeit};

use crate::constraint_vocab::LLMTokenBV;
use crate::datastructures::abstract_weight::{BackendChoice, current_backend_choice};
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::datastructures::rangemap_weight::{RangeMapWeight, intern_rangemap};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::dfa_u8::{Tokenizer, Regex};
use crate::glr::approximate_dfa::LazyApproximateDFA;
use crate::glr::grammar::Terminal;
use crate::glr::table::TerminalID;
use crate::glr::parser::GLRParser;
use crate::dwa_i32::rangeset::RangeSet as WARangeSet;
use crate::dwa_i32::{DeterminizeAndMinimizeProfile, DWA, NWA, NWAStateID, Weight};
use crate::dwa_i32::weight_expansion::{expand_rsb, create_tsid_set_mask_with_offset_map};
use crate::profiler::{self};
use crate::interface::{GrammarDefinition, prune_dwa_with_suffix_grammar, prune_nwa_with_suffix_grammar};

use crate::dfa_u8::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID as GrammarTokenID;
use crate::dwa_i32::common::Label;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ChainCollapseStats {
    pub(crate) collapsed_states: usize,
    pub(crate) rewired_transitions: usize,
    pub(crate) blocked_repeat_transitions: usize,
    pub(crate) pruned_states: usize,
    pub(crate) iterations: usize,
}

pub(crate) fn collapse_self_extending_chains(
    dwa: &mut DWA,
    self_extending_labels: &HashSet<Label>,
) -> ChainCollapseStats {
    let mut stats = ChainCollapseStats::default();

    loop {
        let num_states = dwa.states.len();
        if num_states == 0 {
            break;
        }

        let mut incoming_total = vec![0usize; num_states];
        let mut incoming_by_label: HashMap<(usize, Label), usize> = HashMap::new();
        let mut incoming_sources: HashMap<(usize, Label), Vec<usize>> = HashMap::new();

        for (src, state) in dwa.states.0.iter().enumerate() {
            for (&label, &dst) in &state.transitions {
                incoming_total[dst] += 1;
                *incoming_by_label.entry((dst, label)).or_insert(0) += 1;
                incoming_sources.entry((dst, label)).or_default().push(src);
            }
        }

        let mut changed = false;
        for b in 0..num_states {
            if b == dwa.body.start_state {
                continue;
            }
            let state_b = &dwa.states[b];
            if state_b.transitions.len() != 1 {
                continue;
            }
            let (&label, &c) = match state_b.transitions.iter().next() {
                Some(entry) => entry,
                None => continue,
            };
            if !self_extending_labels.contains(&label) {
                continue;
            }
            if c == b {
                continue; // self-loop
            }
            let incoming_label = incoming_by_label.get(&(b, label)).copied().unwrap_or(0);
            if incoming_label == 0 || incoming_total[b] != incoming_label {
                continue; // other incoming labels exist
            }

            let preds = incoming_sources.get(&(b, label)).cloned().unwrap_or_default();
            if preds.is_empty() {
                continue;
            }

            let w_out = state_b
                .trans_weights
                .get(&label)
                .cloned()
                .unwrap_or_else(Weight::zeros);
            let final_weight_b = state_b.final_weight.clone();

            for p in preds {
                let w_in = dwa.states[p]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::zeros);
                let mut new_w = w_in;
                new_w |= &w_out;
                dwa.states[p].trans_weights.insert(label, new_w);
                dwa.states[p].transitions.insert(label, c);
                stats.rewired_transitions += 1;
            }

            if let Some(fw_b) = final_weight_b {
                let fw_c = dwa.states[c]
                    .final_weight
                    .get_or_insert_with(Weight::zeros);
                *fw_c |= &fw_b;
            }

            let state_b_mut = &mut dwa.states[b];
            state_b_mut.transitions.clear();
            state_b_mut.trans_weights.clear();
            state_b_mut.final_weight = None;

            stats.collapsed_states += 1;
            changed = true;
        }

        stats.iterations += 1;
        if !changed {
            break;
        }
    }

    if !self_extending_labels.is_empty() && !dwa.states.0.is_empty() {
        let mut incoming_label: HashSet<(usize, Label)> = HashSet::new();
        for state in dwa.states.0.iter() {
            for (&label, &dst) in &state.transitions {
                if self_extending_labels.contains(&label) {
                    incoming_label.insert((dst, label));
                }
            }
        }

        for (dst, label) in incoming_label {
            if dst == dwa.body.start_state {
                continue;
            }
            if let Some(state) = dwa.states.0.get_mut(dst) {
                if state.transitions.remove(&label).is_some() {
                    state.trans_weights.remove(&label);
                    stats.blocked_repeat_transitions += 1;
                }
            }
        }
    }

    let before = dwa.states.len();
    if dwa.prune_unreachable() {
        let after = dwa.states.len();
        if after < before {
            stats.pruned_states += before - after;
        }
    }

    stats
}

// No-op progress bar replacement
struct NoOpPb;
impl NoOpPb {
    fn inc(&self, _: u64) {}
    fn finish(&self) {}
}

#[derive(Default, Clone)]
struct DfsProfile {
    exec_calls: u64,
    exec_time_us: u64,
    possible_matches_calls: u64,
    possible_matches_time_us: u64,
    tokens_accessible_calls: u64,
    tokens_accessible_time_us: u64,
    expanded_item_calls: u64,
    expanded_item_time_us: u64,
    expanded_rsb_calls: u64,
    expanded_rsb_time_us: u64,
    expanded_all_calls: u64,
    expanded_all_time_us: u64,
    add_transition_calls: u64,
    add_transition_time_us: u64,
    add_epsilon_calls: u64,
    add_epsilon_time_us: u64,
}

const EXPANDED_RSB_CACHE_MAX_ENTRIES: usize = 100_000;
const EXPANDED_RSB_VALUE_CACHE_MAX_ENTRIES: usize = 50_000;

impl DfsProfile {
    fn print(&self) {
        let ms = |us: u64| us as f64 / 1000.0;
        crate::debug!(5, "precompute1 dfs profile: exec={} calls, {:.2}ms", self.exec_calls, ms(self.exec_time_us));
        crate::debug!(5, "precompute1 dfs profile: possible_matches={} calls, {:.2}ms", self.possible_matches_calls, ms(self.possible_matches_time_us));
        crate::debug!(5, "precompute1 dfs profile: tokens_accessible={} calls, {:.2}ms", self.tokens_accessible_calls, ms(self.tokens_accessible_time_us));
        crate::debug!(5, "precompute1 dfs profile: expanded_item={} calls, {:.2}ms", self.expanded_item_calls, ms(self.expanded_item_time_us));
        crate::debug!(5, "precompute1 dfs profile: expanded_rsb={} calls, {:.2}ms", self.expanded_rsb_calls, ms(self.expanded_rsb_time_us));
        crate::debug!(5, "precompute1 dfs profile: expanded_all={} calls, {:.2}ms", self.expanded_all_calls, ms(self.expanded_all_time_us));
        crate::debug!(5, "precompute1 dfs profile: add_transition={} calls, {:.2}ms", self.add_transition_calls, ms(self.add_transition_time_us));
        crate::debug!(5, "precompute1 dfs profile: add_epsilon={} calls, {:.2}ms", self.add_epsilon_calls, ms(self.add_epsilon_time_us));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DfsKey {
    tokenizer_state: TokenizerStateID,
    approx_state: usize,
}

impl DfsKey {
    #[inline]
    fn new(tokenizer_state: TokenizerStateID, approx_state: usize) -> Self {
        Self { tokenizer_state, approx_state }
    }
}

#[derive(Clone)]
pub struct ApproximateDfaPruner {
    pub dfa: LazyApproximateDFA,
    pub orig_to_suffix_tid: Vec<Option<crate::types::TerminalID>>,
    pub ignored_terminals: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Precomputer1
// ---------------------------------------------------------------------------

pub(crate) struct Precomputer1<'r> {
    pub(crate) tokenizer: &'r Tokenizer,
    pub(crate) vocab: VocabPrefixTree,
    pub(crate) roots: BTreeMap<DfsKey, NWAStateID>,
    pub(crate) state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
    pub(crate) possible_matches: RefCell<
        BTreeMap<
            *const VocabPrefixTreeNode,
            BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>,
        >,
    >,
    pub(crate) all_llm_tokens: RangeSetBlaze<usize>,
    pub(crate) pb: NoOpPb,
    pub(crate) leaf_state: NWAStateID,
    pub(crate) nwa: NWA,
    pub(crate) terminals_count: usize,
    pub(crate) pending_transitions: HashMap<NWAStateID, HashMap<Label, HashMap<NWAStateID, Weight>>>,
    pub(crate) pending_epsilons: HashMap<NWAStateID, HashMap<NWAStateID, Weight>>,
    pub(crate) live_tokens: HashMap<NWAStateID, Weight>,
    // Cache for tokens_accessible_from_state - only 389 unique states but called 700k+ times
    accessible_terminals_cache: HashMap<TokenizerStateID, std::rc::Rc<Vec<GrammarTokenID>>>,
    // Cache for expanded single-token weights (indexed by token id)
    expanded_item_cache: Vec<Option<Weight>>,
    // Cache for expanded RangeSetBlaze weights (pointer-keyed, stable sets only)
    expanded_rsb_cache: HashMap<usize, Weight>,
    expanded_rsb_value_cache: HashMap<RangeSetBlaze<usize>, Weight>,
    // Weight-heavy mode: number of tokenizer states
    pub(crate) num_tsids: usize,
    // Max LLM token ID for creating tsid masks
    pub(crate) internal_max_llm_token: usize,
    /// Optional tsid->offset mapping for weight-heavy encoding (empty = identity).
    pub(crate) tsid_offset_map: Vec<usize>,
    expanded_all_weight: Weight,
    dfs_profile_enabled: bool,
    dfs_profile: DfsProfile,
    approx_dfa: Option<ApproximateDfaPruner>,
    approx_start_state: usize,
    direct_insert: bool,
    suffix_prune_grammar: Option<Arc<GrammarDefinition>>,
    suffix_prune_terminal_map: Option<BiBTreeMap<Terminal, TerminalID>>,
    self_extending_labels_for_collapse: Option<Arc<HashSet<Label>>>,
}

impl<'r> Precomputer1<'r> {
    fn new(
        tokenizer: &'r Tokenizer,
        internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        terminals_count: usize,
        state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
        num_tsids: usize,
        tsid_offset_map: Vec<usize>,
        approx_dfa: Option<ApproximateDfaPruner>,
        suffix_prune_grammar: Option<Arc<GrammarDefinition>>,
        suffix_prune_terminal_map: Option<BiBTreeMap<Terminal, TerminalID>>,
        self_extending_labels_for_collapse: Option<Arc<HashSet<Label>>>,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!(
                "Precompute1 tokens: internal_llm_token_map entries={}, internal_max_llm_token={}, num_tsids={}",
                internal_llm_token_map.len(),
                internal_max_llm_token,
                num_tsids,
            );
        }

        crate::debug!(6, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(6, "Done building vocab prefix tree");

        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Clear default start state

        let approx_start_state = approx_dfa.as_ref().map(|dfa| dfa.dfa.start_state).unwrap_or(0);

        let mut roots = BTreeMap::new();
        for &rep_sid in state_to_rep.values() {
            let key = DfsKey::new(rep_sid, approx_start_state);
            if !roots.contains_key(&key) {
                let root_state = nwa.add_state();
                roots.insert(key, root_state);
            }
        }
        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!(
                "Created trie1 roots ({} states for {} total tsids)",
                roots.len(),
                state_to_rep.len()
            );
        }

        let pb = NoOpPb;

        let leaf_state = nwa.add_state();
        // Final weight - expanded in weight-heavy mode, simple in symbol-heavy mode
        // IMPORTANT: Use [0..=...] to create from ONE range, not iterate over all integers!
        let final_weight = if num_tsids == 0 {
            // Symbol-heavy mode: all tokens in N-space
            Weight::from_rsb(RangeSetBlaze::from_iter([0..=internal_max_llm_token]))
        } else {
            // Weight-heavy mode: all tokens in N×M-space
            Weight::from_rsb(expand_rsb(&RangeSetBlaze::from_iter([0..=internal_max_llm_token]), num_tsids))
        };
        nwa.states[leaf_state].final_weight = Some(final_weight);
        crate::debug!(6, "Created trie1 leaf state with final weight (num_tsids={})", num_tsids);

        let expanded_all_weight = if num_tsids == 0 {
            // Symbol-heavy mode: all tokens in N-space
            Weight::from_rsb(RangeSetBlaze::from_iter([0..=internal_max_llm_token]))
        } else {
            // Weight-heavy mode: All tokens in N×M space
            let max_pos = internal_max_llm_token * num_tsids + num_tsids - 1;
            // IMPORTANT: Use [0..=max_pos] to create from ONE range, not iterate over all integers!
            Weight::from_rsb(RangeSetBlaze::from_iter([0..=max_pos]))
        };

        let direct_insert = std::env::var("PRECOMPUTE1_DIRECT_INSERT")
            .map(|v| v == "1")
            .unwrap_or(false);

        Self {
            tokenizer,
            vocab,
            roots,
            state_to_rep,
            possible_matches: RefCell::new(BTreeMap::new()),
            // IMPORTANT: Use [0..=...] to create from ONE range, not iterate over all integers!
            all_llm_tokens: RangeSetBlaze::from_iter([0..=internal_max_llm_token]),
            pb,
            leaf_state,
            nwa,
            terminals_count,
            pending_transitions: HashMap::new(),
            pending_epsilons: HashMap::new(),
            live_tokens: HashMap::new(),
            accessible_terminals_cache: HashMap::new(),
            expanded_item_cache: vec![None; internal_max_llm_token.saturating_add(1)],
            expanded_rsb_cache: HashMap::new(),
            expanded_rsb_value_cache: HashMap::new(),
            num_tsids,
            internal_max_llm_token,
            tsid_offset_map,
            expanded_all_weight,
            dfs_profile_enabled: std::env::var("PROFILE_PRECOMPUTE1_DFS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            dfs_profile: DfsProfile::default(),
            approx_dfa,
            approx_start_state,
            direct_insert,
            suffix_prune_grammar,
            suffix_prune_terminal_map,
            self_extending_labels_for_collapse,
        }
    }

    #[time_it("Precompute1::finish")]
    fn finish(mut self) -> DWA {
        let run_debug_scan = std::env::var("PRECOMPUTE1_DEBUG_SCAN")
            .map(|v| v == "1")
            .unwrap_or(false)
            || crate::r#macro::is_debug_level_enabled(7);
        if run_debug_scan && !self.direct_insert {
            timeit!("precompute1::debug_scan", {
                let debug_scan_start = std::time::Instant::now();
                // Debug: print all states and transitions before processing
                crate::debug!(7, "=== NWA before flush (leaf_state={}, roots={:?}) ===", self.leaf_state, self.roots);
                for (i, state) in self.nwa.states.0.iter().enumerate() {
                    let trans_count = state.transitions.values().map(|v| v.len()).sum::<usize>();
                    let eps_count = state.epsilons.len();
                    let is_final = state.final_weight.is_some();
                    crate::debug!(7, "State {}: {} transitions, {} epsilons, final={}", i, trans_count, eps_count, is_final);
                }
                crate::debug!(7, "Pending transitions:");
                for (src, labels) in &self.pending_transitions {
                    for (label, dsts) in labels {
                        for (dst, weight) in dsts {
                            crate::debug!(7, "  {} --{}--> {} (weight: {:?})", src, label, dst, weight);
                        }
                    }
                }
                crate::debug!(7, "Pending epsilons:");
                for (src, dsts) in &self.pending_epsilons {
                    for (dst, weight) in dsts {
                        crate::debug!(7, "  {} --eps--> {} (weight: {:?})", src, dst, weight);
                    }
                }
                
                // Debug: Count transitions
                let mut total_transitions = 0;
                let mut transitions_to_leaf = 0;
                for (src, labels) in &self.pending_transitions {
                    for (label, dsts) in labels {
                        for (dst, weight) in dsts {
                            total_transitions += 1;
                            if *dst == self.leaf_state {
                                transitions_to_leaf += 1;
                                // Check if token 6 and 31 are in the same weight
                                if weight.contains(6) && weight.contains(31) {
                                    // Good - merged
                                } else if weight.contains(6) || weight.contains(31) {
                                    // crate::debug!(7, "SEPARATE: transition from {} on label {} has weight with 6={} 31={}",
                                    //     src, label, weight.contains(6), weight.contains(31));
                                }
                            }
                        }
                    }
                }
                // crate::debug!(5, "Pending transitions: {} total, {} to leaf", total_transitions, transitions_to_leaf);

                crate::debug!(5, "Precompute1 finish: debug scans in {:?}", debug_scan_start.elapsed());
            });
        }
        
        // Flush pending transitions and epsilons into the NWA
        if !self.direct_insert {
            timeit!("precompute1::flush_pending", {
                let flush_start = std::time::Instant::now();
                for (src, labels) in std::mem::take(&mut self.pending_transitions) {
                    let state = &mut self.nwa.states[src];
                    for (label, dsts) in labels {
                        let targets = state.transitions.entry(label).or_default();
                        targets.reserve(dsts.len());
                        targets.extend(dsts.into_iter());
                    }
                }
                for (src, dsts) in std::mem::take(&mut self.pending_epsilons) {
                    let state = &mut self.nwa.states[src];
                    state.epsilons.reserve(dsts.len());
                    state.epsilons.extend(dsts.into_iter());
                }
                crate::debug!(4, "Precompute1 finish: flushed pending transitions/epsilons in {:?}", flush_start.elapsed());
            });
        }

        if let Ok(token_str) = std::env::var("DEBUG_PRECOMPUTE1_NWA_TOKEN") {
            if let Ok(token_id) = token_str.parse::<usize>() {
                let token_len = std::env::var("DEBUG_PRECOMPUTE1_NWA_TOKEN_LEN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);

                let weight_contains_token = |weight: &Weight| -> bool {
                    if self.num_tsids == 0 {
                        weight.contains(token_id)
                    } else {
                        let start = token_id.saturating_mul(self.num_tsids);
                        let end = start.saturating_add(self.num_tsids.saturating_sub(1));
                        for range in weight.ranges() {
                            let r_start = *range.start();
                            let r_end = *range.end();
                            if r_start > end {
                                break;
                            }
                            if r_end >= start {
                                return true;
                            }
                        }
                        false
                    }
                };

                let mut best_seen: HashMap<NWAStateID, usize> = HashMap::new();
                let mut found_path: Option<Vec<(NWAStateID, Option<Label>, NWAStateID)>> = None;

                fn dfs_nwa(
                    nwa: &NWA,
                    terminals_count: usize,
                    state: NWAStateID,
                    term_len: usize,
                    token_len: usize,
                    weight_contains_token: &impl Fn(&Weight) -> bool,
                    best_seen: &mut HashMap<NWAStateID, usize>,
                    path: &mut Vec<(NWAStateID, Option<Label>, NWAStateID)>,
                    found_path: &mut Option<Vec<(NWAStateID, Option<Label>, NWAStateID)>>,
                ) {
                    if found_path.is_some() {
                        return;
                    }
                    if term_len > token_len {
                        *found_path = Some(path.clone());
                        return;
                    }
                    if let Some(best) = best_seen.get(&state) {
                        if *best >= term_len {
                            return;
                        }
                    }
                    best_seen.insert(state, term_len);

                    // Epsilon transitions
                    for (next_state, w) in &nwa.states[state].epsilons {
                        if !weight_contains_token(w) {
                            continue;
                        }
                        path.push((state, None, *next_state));
                        dfs_nwa(nwa, terminals_count, *next_state, term_len, token_len, weight_contains_token, best_seen, path, found_path);
                        path.pop();
                        if found_path.is_some() {
                            return;
                        }
                    }

                    // Terminal transitions
                    for (&label, targets) in &nwa.states[state].transitions {
                        for (next_state, w) in targets {
                            if !weight_contains_token(w) {
                                continue;
                            }
                            let label_usize = label as usize;
                            let add = if label_usize < terminals_count { 1 } else { 0 };
                            path.push((state, Some(label), *next_state));
                            dfs_nwa(nwa, terminals_count, *next_state, term_len + add, token_len, weight_contains_token, best_seen, path, found_path);
                            path.pop();
                            if found_path.is_some() {
                                return;
                            }
                        }
                    }
                }

                for &start in &self.nwa.body.start_states {
                    let mut path = Vec::new();
                    dfs_nwa(
                        &self.nwa,
                        self.terminals_count,
                        start,
                        0,
                        token_len,
                        &weight_contains_token,
                        &mut best_seen,
                        &mut path,
                        &mut found_path,
                    );
                    if found_path.is_some() {
                        break;
                    }
                }

                if let Some(path) = found_path {
                    eprintln!("DEBUG_NWA_TOKEN_PATH token_id={} token_len={} steps={}", token_id, token_len, path.len());
                    for (idx, (src, label, dst)) in path.iter().enumerate() {
                        if let Some(label) = label {
                            let label_usize = *label as usize;
                            let kind = if label_usize < self.terminals_count { "terminal" } else { "tsid" };
                            eprintln!("DEBUG_NWA[{}] {} --{} {}--> {}", idx, src, kind, label, dst);
                        } else {
                            eprintln!("DEBUG_NWA[{}] {} --eps--> {}", idx, src, dst);
                        }
                    }
                } else {
                    eprintln!("DEBUG_NWA_TOKEN_PATH token_id={} token_len={} not found", token_id, token_len);
                }
            }
        }

        // Create start state with transitions to root states
        let new_start_state = timeit!("precompute1::start_state", {
            let start_state_start = std::time::Instant::now();
            let new_start_state = self.nwa.add_state();
            
            if self.num_tsids == 0 {
                // Symbol-heavy mode: create labeled transitions with Weight::all()
                // Label = tsid + terminals_count
                // Important: We need to create labels for ALL tsids (not just representatives),
                // because at runtime we'll look up by the raw tokenizer state ID.
                // All tsids that map to the same representative get their own label but point
                // to the same root state.
                let mut transitions_added = 0;
                let mut add_transition_time = std::time::Duration::ZERO;
                let mut unique_targets = std::collections::HashSet::new();
                for (tsid, rep_tsid) in &self.state_to_rep {
                    let root_key = DfsKey::new(*rep_tsid, self.approx_start_state);
                    if let Some(&state) = self.roots.get(&root_key) {
                        let label = (tsid.0 + self.terminals_count) as Label;
                        let weight = Weight::from_rsb(RangeSetBlaze::from_iter([0..=self.internal_max_llm_token]));
                        let add_start = std::time::Instant::now();
                        self.nwa.add_transition(new_start_state, label, state, weight).unwrap();
                        add_transition_time += add_start.elapsed();
                        transitions_added += 1;
                        unique_targets.insert(state);
                    }
                }
                crate::debug!(4, "Precompute1 start-state breakdown (symbol-heavy): add_transition={:?}", add_transition_time);
                crate::debug!(3, "Symbol-heavy mode: added {} tsid transitions to {} unique root states", 
                    transitions_added, unique_targets.len());
            } else {
                // Weight-heavy mode: create epsilon transitions with tsid-masked weights
                // Group tsids by their representative to call create_tsid_set_mask once per group
                let group_start = std::time::Instant::now();
                let mut rep_to_tsids: BTreeMap<TokenizerStateID, Vec<usize>> = BTreeMap::new();
                for (tsid, rep_tsid) in &self.state_to_rep {
                    rep_to_tsids.entry(*rep_tsid).or_default().push(tsid.0);
                }
                let group_time = group_start.elapsed();

                let mut mask_time = std::time::Duration::ZERO;
                let mut add_eps_time = std::time::Duration::ZERO;
                let mut group_count = 0usize;
                let mut tsid_count = 0usize;

                let tsid_offset_map = if self.tsid_offset_map.is_empty() {
                    None
                } else {
                    Some(self.tsid_offset_map.as_slice())
                };

                // Create one epsilon transition per representative with combined tsid mask
                for (rep_tsid, tsids) in rep_to_tsids {
                    debug_assert!(tsids.contains(&rep_tsid.0));
                    let root_key = DfsKey::new(rep_tsid, self.approx_start_state);
                    if let Some(&state) = self.roots.get(&root_key) {
                        group_count += 1;
                        tsid_count += tsids.len();
                        // Create combined tsid mask for all tsids that map to this representative.
                        // If we have a tsid->offset map, build the mask in the permuted offset space
                        // (this can substantially reduce RangeSet fragmentation when representative
                        // groups are scattered across the original tsid numbering).
                        let mask_start = std::time::Instant::now();
                        let tsid_mask = create_tsid_set_mask_with_offset_map(
                            tsids,
                            self.num_tsids,
                            self.internal_max_llm_token,
                            tsid_offset_map,
                        );
                        mask_time += mask_start.elapsed();
                        let add_eps_start = std::time::Instant::now();
                        self.nwa.add_epsilon(new_start_state, state, tsid_mask);
                        add_eps_time += add_eps_start.elapsed();
                    }
                }
                crate::debug!(
                    4,
                    "Precompute1 start-state breakdown: group_build={:?}, mask_build={:?}, add_epsilon={:?}, groups={}, tsids={}",
                    group_time,
                    mask_time,
                    add_eps_time,
                    group_count,
                    tsid_count,
                );
            }
            crate::debug!(4, "Precompute1 finish: added start state transitions in {:?}", start_state_start.elapsed());
            new_start_state
        });
        self.nwa.body.start_states = vec![new_start_state];

        // Stats
        // Find cases where there's multiple instances of same transition - incl symbol/epsilon transition - from one state to another, regardless of weight.
        let run_duplicate_scan = std::env::var("PRECOMPUTE1_DUPLICATE_SCAN")
            .map(|v| v == "1")
            .unwrap_or(false)
            || crate::r#macro::is_debug_level_enabled(6);
        if run_duplicate_scan {
            timeit!("precompute1::duplicate_scan", {
                let mut duplicate_transitions = 0;
                let duplicate_start = std::time::Instant::now();
                for state in &self.nwa.states.0 {
                    let mut dst_counts = HashMap::new();
                    for (dst, _) in &state.epsilons {
                        *dst_counts.entry(*dst).or_insert(0) += 1;
                    }
                    for count in dst_counts.values() {
                        if *count > 1 {
                            duplicate_transitions += count - 1;
                        }
                    }

                    for targets in state.transitions.values() {
                        let mut dst_counts = HashMap::new();
                        for (dst, _) in targets {
                            *dst_counts.entry(*dst).or_insert(0) += 1;
                        }
                        for count in dst_counts.values() {
                            if *count > 1 {
                                duplicate_transitions += count - 1;
                            }
                        }
                    }
                }
                if duplicate_transitions > 0 {
                    crate::debug!(6, "NWA: Found {} duplicate transitions (same src, dst, label)", duplicate_transitions);
                }
                crate::debug!(4, "Precompute1 finish: duplicate transition scan in {:?}", duplicate_start.elapsed());
            });
        }

        // Find cases where there's multiple instances of same transition - regardless of symbol/epsilon transition - from one state to another, regardless of weight.
        let run_parallel_scan = std::env::var("PRECOMPUTE1_PARALLEL_SCAN")
            .map(|v| v == "1")
            .unwrap_or(false)
            || crate::r#macro::is_debug_level_enabled(6);
        if run_parallel_scan {
            timeit!("precompute1::parallel_scan", {
                let mut parallel_connections = 0;
                let parallel_start = std::time::Instant::now();
                for state in &self.nwa.states.0 {
                    let mut dst_counts = HashMap::new();
                    for (dst, _) in &state.epsilons {
                        *dst_counts.entry(*dst).or_insert(0) += 1;
                    }
                    for targets in state.transitions.values() {
                        for (dst, _) in targets {
                            *dst_counts.entry(*dst).or_insert(0) += 1;
                        }
                    }

                    for count in dst_counts.values() {
                        if *count > 1 {
                            parallel_connections += 1;
                        }
                    }
                }
                if parallel_connections > 0 {
                    crate::debug!(5, "NWA: Found {} pairs of states connected by multiple transitions", parallel_connections);
                }
                crate::debug!(4, "Precompute1 finish: parallel transition scan in {:?}", parallel_start.elapsed());
            });
        }

        crate::debug!(3, "Terminal NWA: {}, num_tsids={}", 
                  self.nwa.stats(), self.num_tsids);

        if std::env::var("DWA_DUMP_NWA").map(|v| v == "1").unwrap_or(false) {
            crate::debug!(5, "Dumping NWA to nwa_dump.json");
            let json = serde_json::to_string(&self.nwa).unwrap();
            std::fs::write("nwa_dump.json", json).unwrap();
        }

        let do_nwa_suffix_prune = std::env::var("NWA_SUFFIX_PRUNE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        if do_nwa_suffix_prune {
            match (&self.suffix_prune_grammar, &self.suffix_prune_terminal_map) {
                (Some(grammar_def), Some(terminal_map)) => {
                    crate::debug!(4, "Terminal NWA (before suffix pruning): {}", self.nwa.stats());
                    let prune_start = std::time::Instant::now();
                    let (kept, pruned) = prune_nwa_with_suffix_grammar(
                        &mut self.nwa,
                        grammar_def,
                        terminal_map,
                        self.terminals_count,
                    );
                    crate::debug!(4, "Terminal NWA suffix pruning complete. Kept={}, pruned={}", kept, pruned);
                    crate::debug!(4, "Terminal NWA (after suffix pruning): {}", self.nwa.stats());
                    eprintln!("TIMING: terminal_nwa_suffix_prune {:?}", prune_start.elapsed());
                }
                _ => {
                    crate::debug!(4, "NWA_SUFFIX_PRUNE set but missing grammar definition or terminal map; skipping");
                }
            }
        }

        let do_dwa_suffix_prune = std::env::var("DWA_SUFFIX_PRUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let pre_dwa_suffix_prune = if do_dwa_suffix_prune {
            let suffix_prune_grammar = self.suffix_prune_grammar.clone();
            let suffix_prune_terminal_map = self.suffix_prune_terminal_map.clone();
            let terminals_count = self.terminals_count;
            Some(move |dwa: &mut DWA| {
                match (&suffix_prune_grammar, &suffix_prune_terminal_map) {
                    (Some(grammar_def), Some(terminal_map)) => {
                        crate::debug!(4, "Terminal DWA (before pre-min suffix pruning): {}", dwa.stats());
                        let prune_start = std::time::Instant::now();
                        let (kept, pruned) = prune_dwa_with_suffix_grammar(
                            dwa,
                            grammar_def,
                            terminal_map,
                            terminals_count,
                        );
                        crate::debug!(4, "Terminal DWA pre-min suffix pruning complete. Kept={}, pruned={}", kept, pruned);
                        crate::debug!(4, "Terminal DWA (after pre-min suffix pruning): {}", dwa.stats());
                        eprintln!("TIMING: terminal_dwa_suffix_prune_pre_min {:?}", prune_start.elapsed());
                    }
                    _ => {
                        crate::debug!(4, "DWA_SUFFIX_PRUNE set but missing grammar definition or terminal map; skipping");
                    }
                }
            })
        } else {
            None
        };

        let collapse_labels = self.self_extending_labels_for_collapse.clone();
        let pre_dwa_hook = match (pre_dwa_suffix_prune, collapse_labels) {
            (None, None) => None,
            (prune_opt, collapse_opt) => Some(move |dwa: &mut DWA| {
                if let Some(labels) = collapse_opt.as_ref() {
                    let before_stats = dwa.stats();
                    let collapse_start = std::time::Instant::now();
                    let stats = collapse_self_extending_chains(dwa, labels);
                    eprintln!(
                        "TIMING: terminal_dwa_self_ext_chain_collapse {:?}",
                        collapse_start.elapsed()
                    );
                    crate::debug!(
                        4,
                        "Terminal DWA self-ext chain collapse: {} -> {} (collapsed_states={}, rewired={}, blocked_repeats={}, pruned={}, iterations={})",
                        before_stats,
                        dwa.stats(),
                        stats.collapsed_states,
                        stats.rewired_transitions,
                        stats.blocked_repeat_transitions,
                        stats.pruned_states,
                        stats.iterations,
                    );
                }
                if let Some(prune) = prune_opt {
                    prune(dwa);
                }
            }),
        };

        // Use unified determinize_and_minimize with "Terminal" profile
        // Pipeline: NWA minimize → compress → rm_epsilon → determinize → DWA minimize
        // Expected results: 14647 → 5904 → 5904 → 889 → 189 states
        let profile_minimize_only = std::env::var("PROFILE_FACTORIZED_WEIGHT_MINIMIZE_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if profile_minimize_only {
            crate::datastructures::factorized_weight::set_factorized_weight_profile_active(true);
            crate::datastructures::factorized_weight::reset_factorized_weight_profile();
        }
        crate::debug!(5, "precompute1::determinize_and_minimize start");
        let dwa = timeit!("precompute1::determinize_and_minimize", {
            self.nwa.determinize_and_minimize_with_hook(
                DeterminizeAndMinimizeProfile::Terminal,
                pre_dwa_hook,
            )
        });
        crate::debug!(5, "precompute1::determinize_and_minimize end");
        if profile_minimize_only {
            crate::datastructures::factorized_weight::flush_factorized_weight_profile("terminal_dwa_minimize");
            crate::datastructures::factorized_weight::set_factorized_weight_profile_active(false);
        }
        
        // NOTE: Stats are printed AFTER suffix grammar pruning in constraint.rs
        // This includes path counts, average path lengths, and sample paths.
        crate::debug!(4, "Terminal DWA (before suffix pruning): {}", 
                  dwa.stats());

        dwa
    }

    fn possible_matches(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key_ptr = vocab_node as *const VocabPrefixTreeNode;

        if let Some(cached_for_vocab_node) =
            self.possible_matches.borrow().get(&cache_key_ptr)
        {
            if let Some(cached_result) =
                cached_for_vocab_node.get(&tokenizer_state_id)
            {
                return cached_result.clone();
            }
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result =
                self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_node.reachable_token_ids();
                *result_map
                    .entry(grammar_token_id)
                    .or_insert_with(LLMTokenBV::zeros) |=
                    RangeSet::from(applicable_tokens);
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: std::collections::BTreeSet<_> = self
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(final_state_val))
                    .into_iter()
                    .collect();
                let matches_here: std::collections::BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();
                let possible_new_matches =
                    &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(
                        child_vocab_node,
                        TokenizerStateID(final_state_val),
                    );
                    for (token, bv) in next_results {
                        *result_map
                            .entry(token)
                            .or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }

        self.possible_matches
            .borrow_mut()
            .entry(cache_key_ptr)
            .or_default()
            .insert(tokenizer_state_id, result_map.clone());

        result_map
    }

    #[inline]
    fn approx_step(&mut self, approx_state: usize, terminal_id: GrammarTokenID) -> Option<usize> {
        let Some(approx_dfa) = self.approx_dfa.as_mut() else {
            return Some(approx_state);
        };

        let term_idx = terminal_id.0;
        if approx_dfa
            .ignored_terminals
            .get(term_idx)
            .copied()
            .unwrap_or(false)
        {
            return Some(approx_state);
        }

        let suffix_tid = approx_dfa.orig_to_suffix_tid.get(term_idx).copied().flatten();
        let Some(suffix_tid) = suffix_tid else {
            return Some(approx_state);
        };

        approx_dfa.dfa.step(approx_state, suffix_tid)
    }

    fn get_or_create_next_state(
        &mut self,
        _src_node: NWAStateID,
        tokenizer_state: TokenizerStateID,
        approx_state: usize,
        next_level_assoc: &mut BTreeMap<DfsKey, NWAStateID>,
    ) -> NWAStateID {
        match next_level_assoc.entry(DfsKey::new(tokenizer_state, approx_state)) {
            std::collections::btree_map::Entry::Occupied(o) => *o.get(),
            std::collections::btree_map::Entry::Vacant(v) => {
                // NOTE: The previous state reuse optimization was removed because it
                // iterated through all pending_epsilons (~84 items on avg) but NEVER
                // found a reusable state (0 reuses in 500k+ calls, 42M+ loop iterations).
                // The check `live.is_disjoint(&Weight::all())` can only be true if the
                // live_tokens entry is empty, which almost never happens.
                let t = self.nwa.add_state();
                v.insert(t);
                t
            }
        }
    }

    /// Create an expanded weight from a single token ID.
    /// Expands from N-space to N×M-space where M = num_tsids.
    /// If num_tsids == 0 (symbol-heavy mode), returns the token ID directly in N-space.
    #[inline]
    fn expanded_weight_from_item(&mut self, token_id: usize) -> Weight {
        let start = self.dfs_profile_enabled.then(std::time::Instant::now);
        if let Some(Some(cached)) = self.expanded_item_cache.get(token_id) {
            if let Some(start) = start {
                self.dfs_profile.expanded_item_calls += 1;
                self.dfs_profile.expanded_item_time_us += start.elapsed().as_micros() as u64;
            }
            return cached.clone();
        }

        let weight = if self.num_tsids == 0 {
            // Symbol-heavy mode: just use the token ID directly
            Weight::from_rsb(RangeSetBlaze::from_iter([token_id..=token_id]))
        } else if matches!(current_backend_choice(), BackendChoice::RangeMap) {
            // RangeMap backend: avoid expand->from_rsb per-token overhead
            Weight::RangeMap(intern_rangemap(
                RangeMapWeight::from_token_range_full_tsids(token_id, token_id, self.num_tsids),
            ))
        } else {
            // Weight-heavy mode: A single token ID in N-space becomes a range in N×M-space
            // Token i becomes positions [i*M, i*M + M - 1]
            let start = token_id * self.num_tsids;
            let end = start + self.num_tsids - 1;
            // IMPORTANT: Use [start..=end] to create from ONE range, not iterate over all integers!
            Weight::from_rsb(RangeSetBlaze::from_iter([start..=end]))
        };

        if let Some(slot) = self.expanded_item_cache.get_mut(token_id) {
            *slot = Some(weight.clone());
        }
        if let Some(start) = start {
            self.dfs_profile.expanded_item_calls += 1;
            self.dfs_profile.expanded_item_time_us += start.elapsed().as_micros() as u64;
        }
        weight
    }

    /// Create an expanded weight from a RangeSetBlaze of token IDs.
    /// If num_tsids <= 1 (symbol-heavy or degenerate single-tsid mode), returns the rsb directly.
    #[inline]
    fn expanded_weight_from_rsb(&mut self, rsb: &RangeSetBlaze<usize>, cache_key: Option<usize>) -> Weight {
        let start = self.dfs_profile_enabled.then(std::time::Instant::now);
        if rsb.is_empty() {
            if let Some(start) = start {
                self.dfs_profile.expanded_rsb_calls += 1;
                self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
            }
            return Weight::zeros();
        }
        if std::ptr::eq(rsb, &self.all_llm_tokens) {
            if let Some(start) = start {
                self.dfs_profile.expanded_rsb_calls += 1;
                self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
            }
            return self.expanded_all_weight.clone();
        }
        if let Some(key) = cache_key {
            if let Some(cached) = self.expanded_rsb_cache.get(&key) {
                if let Some(start) = start {
                    self.dfs_profile.expanded_rsb_calls += 1;
                    self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
                }
                return cached.clone();
            }
        }

        let weight = if self.num_tsids <= 1 {
            // Symbol-heavy or single-tsid mode: use rsb directly
            Weight::from_rsb(rsb.clone())
        } else if matches!(current_backend_choice(), BackendChoice::RangeMap) {
            // RangeMap backend: create full-tsid token ranges directly
            Weight::RangeMap(intern_rangemap(
                RangeMapWeight::from_token_ranges_full_tsids(rsb, self.num_tsids),
            ))
        } else {
            // Weight-heavy mode: expand to N×M space
            Weight::from_rsb(expand_rsb(rsb, self.num_tsids))
        };
        if let Some(key) = cache_key {
            if self.expanded_rsb_cache.len() < EXPANDED_RSB_CACHE_MAX_ENTRIES {
                self.expanded_rsb_cache.insert(key, weight.clone());
            }
        }
        if let Some(start) = start {
            self.dfs_profile.expanded_rsb_calls += 1;
            self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
        }
        weight
    }

    /// Create an expanded weight from an owned RangeSetBlaze of token IDs.
    /// Uses a value cache keyed by the full RangeSetBlaze (avoids pointer-only caching).
    #[inline]
    fn expanded_weight_from_rsb_owned(&mut self, rsb: RangeSetBlaze<usize>) -> Weight {
        let start = self.dfs_profile_enabled.then(std::time::Instant::now);
        if rsb.is_empty() {
            if let Some(start) = start {
                self.dfs_profile.expanded_rsb_calls += 1;
                self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
            }
            return Weight::zeros();
        }
        if rsb == self.all_llm_tokens {
            if let Some(start) = start {
                self.dfs_profile.expanded_rsb_calls += 1;
                self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
            }
            return self.expanded_all_weight.clone();
        }

        if let Some(cached) = self.expanded_rsb_value_cache.get(&rsb) {
            if let Some(start) = start {
                self.dfs_profile.expanded_rsb_calls += 1;
                self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
            }
            return cached.clone();
        }

        let weight = if self.num_tsids <= 1 {
            Weight::from_rsb(rsb.clone())
        } else if matches!(current_backend_choice(), BackendChoice::RangeMap) {
            Weight::RangeMap(intern_rangemap(
                RangeMapWeight::from_token_ranges_full_tsids(&rsb, self.num_tsids),
            ))
        } else {
            Weight::from_rsb(expand_rsb(&rsb, self.num_tsids))
        };

        if self.expanded_rsb_value_cache.len() < EXPANDED_RSB_VALUE_CACHE_MAX_ENTRIES {
            self.expanded_rsb_value_cache.insert(rsb, weight.clone());
        }
        if let Some(start) = start {
            self.dfs_profile.expanded_rsb_calls += 1;
            self.dfs_profile.expanded_rsb_time_us += start.elapsed().as_micros() as u64;
        }
        weight
    }

    /// Create an expanded "all" weight (all tokens for all tsids).
    /// If num_tsids == 0 (symbol-heavy mode), returns Weight::all().
    #[inline]
    fn expanded_weight_all(&mut self) -> Weight {
        let start = self.dfs_profile_enabled.then(std::time::Instant::now);
        let weight = self.expanded_all_weight.clone();
        if let Some(start) = start {
            self.dfs_profile.expanded_all_calls += 1;
            self.dfs_profile.expanded_all_time_us += start.elapsed().as_micros() as u64;
        }
        weight
    }

    fn add_pending_transition(&mut self, src: NWAStateID, label: Label, dst: NWAStateID, weight: Weight) {
        let start = self.dfs_profile_enabled.then(std::time::Instant::now);
        if self.direct_insert {
            let state = &mut self.nwa.states[src];
            *self.live_tokens.entry(dst).or_insert_with(Weight::zeros) |= &weight;
            state.transitions.entry(label).or_default().push((dst, weight));
            if let Some(start) = start {
                self.dfs_profile.add_transition_calls += 1;
                self.dfs_profile.add_transition_time_us += start.elapsed().as_micros() as u64;
            }
            return;
        }
        self.pending_transitions
            .entry(src)
            .or_default()
            .entry(label)
            .or_default()
            .entry(dst)
            .and_modify(|w| *w |= &weight)
            .or_insert(weight.clone());
        *self.live_tokens.entry(dst).or_insert_with(Weight::zeros) |= &weight;
        if let Some(start) = start {
            self.dfs_profile.add_transition_calls += 1;
            self.dfs_profile.add_transition_time_us += start.elapsed().as_micros() as u64;
        }
    }

    fn add_pending_epsilon(&mut self, src: NWAStateID, dst: NWAStateID, weight: Weight) {
        let start = self.dfs_profile_enabled.then(std::time::Instant::now);
        if self.direct_insert {
            let state = &mut self.nwa.states[src];
            *self.live_tokens.entry(dst).or_insert_with(Weight::zeros) |= &weight;
            state.epsilons.push((dst, weight));
            if let Some(start) = start {
                self.dfs_profile.add_epsilon_calls += 1;
                self.dfs_profile.add_epsilon_time_us += start.elapsed().as_micros() as u64;
            }
            return;
        }
        self.pending_epsilons
            .entry(src)
            .or_default()
            .entry(dst)
            .and_modify(|w| *w |= &weight)
            .or_insert(weight.clone());
        *self.live_tokens.entry(dst).or_insert_with(Weight::zeros) |= &weight;
        if let Some(start) = start {
            self.dfs_profile.add_epsilon_calls += 1;
            self.dfs_profile.add_epsilon_time_us += start.elapsed().as_micros() as u64;
        }
    }

    fn run_dfs(&mut self) {
        let assoc = self.roots.clone();
        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!("Starting precompute DFS for {} tokenizer states", self.roots.len());
        }
        let vocab = std::mem::replace(&mut self.vocab, VocabPrefixTree::new());
        
        // Count vocab nodes for progress tracking
        let vocab_node_count = count_vocab_nodes(&vocab.root);
        if crate::r#macro::is_debug_level_enabled(3) {
            eprintln!("Vocab tree has {} nodes", vocab_node_count);
        }
        
        self.dfs(&vocab.root, assoc);
        self.vocab = vocab;
        self.pb.finish();
        if self.dfs_profile_enabled {
            self.dfs_profile.print();
        }
        crate::debug!(5, "Precomputation complete");
    }

    fn dfs(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<DfsKey, NWAStateID>,
    ) {
        self.pb.inc(1);
        let mut total_pending_iters = 0usize;
        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            crate::debug!(7, "=== Processing vocab segment: {:?} (token_id={}) ===",
                String::from_utf8_lossy(segment_bytes), child_vocab_node.token_id());
            crate::debug!(7, "Initial assoc_by_state: {:?}", assoc_by_state);
            
            let mut next_level_assoc: BTreeMap<DfsKey, NWAStateID> =
                BTreeMap::new();

            // Queue: pos -> TokenizerState -> (NWAState -> ContextTokens)
            let mut pending: BTreeMap<usize, BTreeMap<DfsKey, NWAStateID>> = BTreeMap::new();
            pending.insert(0, assoc_by_state.clone());

            let child_reachable = child_vocab_node.reachable_token_ids();
            let child_token_id = child_vocab_node.token_id();

            // Caches possible matches for end states to prune edge_bv
            let mut possible_matches_at_end_cache: HashMap<
                TokenizerStateID,
                BTreeMap<GrammarTokenID, LLMTokenBV>,
            > = HashMap::new();

            let mut segment_pending_iters = 0usize;
            while let Some((pos, states_at_pos)) = pending.pop_first() {
                segment_pending_iters += 1;
                total_pending_iters += 1;
                crate::debug!(7, "--- Position {} (segment len={}) ---", pos, segment_bytes.len());
                crate::debug!(7, "States at pos: {:?}", states_at_pos);
                
                // If we reached the end of the segment, these states are ready for the next vocab node
                if pos == segment_bytes.len() {
                    crate::debug!(7, "  -> End of segment, adding epsilons to next level");
                    for (state_key, node) in states_at_pos {
                        let next = self.get_or_create_next_state(
                            node,
                            state_key.tokenizer_state,
                            state_key.approx_state,
                            &mut next_level_assoc,
                        );
                        crate::debug!(7, "     State {} (tsid={:?}) -> epsilon to state {}", node, state_key.tokenizer_state, next);
                        // Use expanded "all" weight
                        let weight_all = self.expanded_weight_all();
                        self.add_pending_epsilon(node, next, weight_all);
                    }
                    continue;
                }

                for (state_key, src_node) in states_at_pos {
                    let tokenizer_state_id = state_key.tokenizer_state;
                    let approx_state = state_key.approx_state;
                    let slice = &segment_bytes[pos..];
                    let exec_start = self.dfs_profile_enabled.then(std::time::Instant::now);
                    let exec_result = self
                        .tokenizer
                        .execute_from_state(slice, tokenizer_state_id);
                    if let Some(start) = exec_start {
                        self.dfs_profile.exec_calls += 1;
                        self.dfs_profile.exec_time_us += start.elapsed().as_micros() as u64;
                    }
                    
                    crate::debug!(7, "  Tokenizer on {:?} from state {:?} (src_node={}): matches={:?}, end_state={:?}",
                        String::from_utf8_lossy(slice), tokenizer_state_id, src_node, exec_result.matches, exec_result.end_state);

                    let possible_matches_at_end = if let Some(end_val) = exec_result.end_state {
                        let ts = TokenizerStateID(end_val);
                        possible_matches_at_end_cache
                            .entry(ts)
                            .or_insert_with(|| {
                                let start = self.dfs_profile_enabled.then(std::time::Instant::now);
                                let result = self.possible_matches(child_vocab_node, ts);
                                if let Some(start) = start {
                                    self.dfs_profile.possible_matches_calls += 1;
                                    self.dfs_profile.possible_matches_time_us += start.elapsed().as_micros() as u64;
                                }
                                result
                            })
                    } else {
                        // Dummy empty map
                        possible_matches_at_end_cache
                            .entry(TokenizerStateID(usize::MAX)) // Arbitrary key that won't be hit
                            .or_default()
                    };

                    // 1. Handle Matches -> Transitions to Initial State
                    for match_info in &exec_result.matches {
                        let terminal_id = GrammarTokenID(match_info.id);
                        let Some(next_approx_state) = self.approx_step(approx_state, terminal_id) else {
                            crate::debug!(7, "      -> Skip match (no approx DFA transition for terminal {})", terminal_id.0);
                            continue;
                        };
                        let next_pos = pos + match_info.width;
                        crate::debug!(7, "    Match: terminal_id={}, width={}, next_pos={}", terminal_id.0, match_info.width, next_pos);

                        // Leaf check: if match consumes remainder of segment
                        if next_pos == segment_bytes.len() {
                            let leaf = self.leaf_state;
                            // Use expanded weight from single token
                            let weight = self.expanded_weight_from_item(child_token_id);
                            crate::debug!(7, "      -> LEAF transition: {} --{}--> {} (leaf_state), weight={:?}", 
                                src_node, terminal_id.0, leaf, weight);
                            self.add_pending_transition(src_node, terminal_id.0 as Label, leaf, weight);
                        }

                        // Continuation logic
                        // Avoid cloning if we don't need to modify the bitset
                        let final_bv: std::borrow::Cow<RangeSetBlaze<usize>> = if next_pos == segment_bytes.len() {
                            let mut edge_bv = child_reachable.clone();
                            edge_bv.remove(child_token_id);
                            if let Some(pm) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv = &edge_bv - pm.inner.as_ref();
                            }
                            crate::debug!(7, "      Continuation at end of segment: edge_bv={:?} (removed child_token_id={}, pm={:?})",
                                edge_bv.iter().collect::<Vec<_>>(), child_token_id, possible_matches_at_end.get(&terminal_id).map(|pm| &pm.inner));
                            std::borrow::Cow::Owned(edge_bv)
                        } else {
                            crate::debug!(7, "      Continuation (not end): using child_reachable={:?}", child_reachable.iter().collect::<Vec<_>>());
                            std::borrow::Cow::Borrowed(child_reachable)
                        };

                        if final_bv.is_empty() {
                            crate::debug!(7, "      -> Skip continuation (empty edge_bv)");
                            continue;
                        }

                        let dest_map = pending.entry(next_pos).or_default();

                        let initial_tsid = self.tokenizer.initial_state_id();
                        // Use expanded weight from rsb
                        let weight = match final_bv {
                            std::borrow::Cow::Borrowed(rsb) => {
                                self.expanded_weight_from_rsb(rsb, Some(rsb as *const _ as usize))
                            }
                            std::borrow::Cow::Owned(rsb) => self.expanded_weight_from_rsb_owned(rsb),
                        };

                        let target_entry = dest_map.entry(DfsKey::new(initial_tsid, next_approx_state));
                        let target = match target_entry {
                            std::collections::btree_map::Entry::Occupied(o) => {
                                crate::debug!(7, "      -> Continuation to existing state: target={}", *o.get());
                                *o.get()
                            }
                            std::collections::btree_map::Entry::Vacant(v) => {
                                let t = self.nwa.add_state();
                                crate::debug!(7, "      -> Created new continuation state: target={}", t);
                                v.insert(t);
                                t
                            }
                        };

                        crate::debug!(7, "      -> CONT transition: {} --{}--> {}, weight={:?}", 
                            src_node, terminal_id.0, target, weight);
                        self.add_pending_transition(src_node, terminal_id.0 as Label, target, weight);
                    }

                    // 2. Handle End State -> Continuation
                    crate::debug!(7, "  End state handling: end_state={:?}", exec_result.end_state);
                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state = TokenizerStateID(end_state_val);
                        
                        // Use cached accessible terminals (389 unique states, but called 700k+ times)
                        let accessible_terminals: std::rc::Rc<Vec<GrammarTokenID>> = if let Some(cached) = self.accessible_terminals_cache.get(&final_tokenizer_state) {
                            cached.clone() // Rc clone is cheap
                        } else {
                            let start = self.dfs_profile_enabled.then(std::time::Instant::now);
                            let result = std::rc::Rc::new(self.tokenizer.tokens_accessible_from_state(final_tokenizer_state)
                                .into_iter().collect::<Vec<_>>());
                            if let Some(start) = start {
                                self.dfs_profile.tokens_accessible_calls += 1;
                                self.dfs_profile.tokens_accessible_time_us += start.elapsed().as_micros() as u64;
                            }
                            self.accessible_terminals_cache.insert(final_tokenizer_state, result.clone());
                            result
                        };
                        
                        crate::debug!(7, "    accessible_terminals={:?}", accessible_terminals.as_slice());

                        // Create expanded weight once, it's just a single token expanded to N×M space
                        let single_token_weight = self.expanded_weight_from_item(child_token_id);

                        let end_idx = self.leaf_state;
                        
                        for terminal_id in accessible_terminals.iter() {
                            let Some(_next_approx_state) = self.approx_step(approx_state, *terminal_id) else {
                                crate::debug!(7, "    -> Skip END_STATE terminal {} (no approx DFA transition)", terminal_id.0);
                                continue;
                            };
                            crate::debug!(7, "    -> END_STATE transition: {} --{}--> {} (leaf_state), weight={:?}",
                                src_node, terminal_id.0, end_idx, single_token_weight);
                            self.add_pending_transition(
                                    src_node,
                                    terminal_id.0 as Label,
                                    end_idx,
                                    single_token_weight.clone(),
                                );
                        }

                        let next = self.get_or_create_next_state(
                            src_node,
                            final_tokenizer_state,
                            approx_state,
                            &mut next_level_assoc,
                        );
                        crate::debug!(7, "    -> END_STATE epsilon: {} --eps--> {}", src_node, next);
                        // Use expanded "all" weight
                        let weight_all = self.expanded_weight_all();
                        self.add_pending_epsilon(src_node, next, weight_all);
                    }
                }
            }

            if crate::r#macro::is_debug_level_enabled(6)
                && std::env::var("SUPPRESS_PRECOMPUTE1_DFS_LOGS").is_err()
            {
                eprintln!(
                    "DFS segment done: segment_len={}, pending_iters={}, next_level_assoc={}",
                    segment_bytes.len(),
                    segment_pending_iters,
                    next_level_assoc.len()
                );
            }

            crate::debug!(7, "=== Done processing segment {:?}, next_level_assoc={:?} ===",
                String::from_utf8_lossy(segment_bytes), next_level_assoc);

            if !next_level_assoc.is_empty() {
                self.dfs(child_vocab_node, next_level_assoc);
            }
        }

        if crate::r#macro::is_debug_level_enabled(6)
            && std::env::var("SUPPRESS_PRECOMPUTE1_DFS_LOGS").is_err()
        {
            eprintln!("DFS total pending iterations: {}", total_pending_iters);
        }
    }
}

pub(crate) fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
}

/// Check if weight-heavy mode is enabled via environment variable.
/// Returns true (weight-heavy enabled) unless DISABLE_WEIGHT_HEAVY=1 is set.
pub fn is_weight_heavy_enabled() -> bool {
    std::env::var("DISABLE_WEIGHT_HEAVY").map(|v| v != "1").unwrap_or(true)
}

// Public entry point wrapper
#[time_it("run_precompute1")]
pub fn run_precompute1(
    tokenizer: &Tokenizer,
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminals_count: usize,
    state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
    tsid_offset_map: Vec<usize>,
    approx_dfa: Option<ApproximateDfaPruner>,
    suffix_prune_grammar: Option<Arc<GrammarDefinition>>,
    suffix_prune_terminal_map: Option<BiBTreeMap<Terminal, TerminalID>>,
    self_extending_labels_for_collapse: Option<Arc<HashSet<Label>>>,
) -> DWA {
    // Compute num_tsids from tokenizer - 0 means symbol-heavy mode
    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    // Ensure global dimensions are set when run_precompute1 is called directly (e.g., tests).
    crate::datastructures::set_global_dims_all_threads(
        internal_max_llm_token,
        if num_tsids > 0 { num_tsids } else { 1 },
    );

    let profile_minimize_only = std::env::var("PROFILE_FACTORIZED_WEIGHT_MINIMIZE_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if profile_minimize_only {
        crate::datastructures::factorized_weight::set_factorized_weight_profile_active(false);
    }
    
    let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut seen_internal_ids = std::collections::HashSet::new();

    for (bytes, id) in internal_llm_token_map {
        if seen_internal_ids.insert(id.0) {
            representative_llm_token_map.insert(bytes.clone(), *id);
        }
    }

    let mut helper = timeit!("precompute1::setup", {
        Precomputer1::new(
            tokenizer,
            &representative_llm_token_map,
            internal_max_llm_token,
            terminals_count,
            state_to_rep,
            num_tsids,
            tsid_offset_map,
            approx_dfa,
            suffix_prune_grammar,
            suffix_prune_terminal_map,
            self_extending_labels_for_collapse,
        )
    });

    timeit!("precompute1::dfs", {
        helper.run_dfs();
    });

    timeit!("precompute1::finish", {
        helper.finish()
    })
}

#[cfg(test)]
pub(crate) fn run_precompute1_nwa_for_tests(
    tokenizer: &Tokenizer,
    internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
    internal_max_llm_token: usize,
    terminals_count: usize,
    state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID>,
    tsid_offset_map: Vec<usize>,
    approx_dfa: Option<ApproximateDfaPruner>,
) -> NWA {
    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    crate::datastructures::set_global_dims_all_threads(
        internal_max_llm_token,
        if num_tsids > 0 { num_tsids } else { 1 },
    );

    let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut seen_internal_ids = std::collections::HashSet::new();

    for (bytes, id) in internal_llm_token_map {
        if seen_internal_ids.insert(id.0) {
            representative_llm_token_map.insert(bytes.clone(), *id);
        }
    }

    let mut helper = Precomputer1::new(
        tokenizer,
        &representative_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        state_to_rep,
        num_tsids,
        tsid_offset_map,
        approx_dfa,
        None,
        None,
        None,
    );

    helper.run_dfs();

    if !helper.direct_insert {
        for (src, labels) in std::mem::take(&mut helper.pending_transitions) {
            let state = &mut helper.nwa.states[src];
            for (label, dsts) in labels {
                let targets = state.transitions.entry(label).or_default();
                targets.reserve(dsts.len());
                targets.extend(dsts.into_iter());
            }
        }
        for (src, dsts) in std::mem::take(&mut helper.pending_epsilons) {
            let state = &mut helper.nwa.states[src];
            state.epsilons.reserve(dsts.len());
            state.epsilons.extend(dsts.into_iter());
        }
    }

    helper.nwa
}
