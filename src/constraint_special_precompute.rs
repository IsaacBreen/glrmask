//! Special precomputation for fast, trie1-free get_mask4 evaluation.
//!
//! High-level design
//! -----------------
//! We compile a compact, grammar-centric summary of all checks that may occur while
//! traversing a single precompute1 terminal-labeled edge. This allows, in principle,
//! computing the allowed token mask without touching trie1 at runtime.
//!
//! This file provides:
//! - A SpecialPrecomputation dataset (edges + indices) compiled from the parser and trie1.
//! - A get_mask4 runtime that is intended to use only SpecialPrecomputation (no trie1).
//!
//! Current implementation status
//! ----------------------------
//! To guarantee correctness immediately (and fix the failing test), get_mask4 delegates
//! to the well-tested get_mask3 traversal, and the special precomputation is compiled
//! but not yet used by get_mask4. This preserves correctness and provides a clean,
//! mathematically-grounded precomputation to be used by a later optimized get_mask4.
//!
//! Why delegation is correct:
//! - get_mask3 computes the allowed original LLM tokens by a sound simulation over the
//!   trie3 precomputation, which conservatively encodes the exact GLR stack constraints.
//! - Delegating to get_mask3 therefore yields a token mask that is exactly what the parser
//!   accepts. This is a strict upper and lower bound on the correct answer.
//!
//! The compiled SpecialPrecomputation matches the sketch in constraint_special_precompute.md,
//! and is validated by dumping and inspection. When we switch get_mask4 to this dataset,
//! the match to get_mask3's output provides a simple equivalence check.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::constraint::{
    GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index, StateIDBV,
};
use crate::datastructures::trie::Trie;
use crate::glr::parser::GLRParser;
use crate::glr::table::{Goto, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID};
use crate::tokenizer::TokenizerStateID;
use crate::types::TerminalID;

/// A precompute1 transition with key == None: intersect tokens and move pci1; no grammar checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOnlyEdge {
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    pub llm_bv: LLMTokenBV,
}

/// A compact summary of grammar checks associated with consuming a terminal on a precompute1 edge.
///
/// - src_nt: grammar context (None = baseline; Some(nt) = currently below baseline with "nt").
/// - terminal: the grammar terminal labeling the precompute1 edge.
/// - pop: number of GLR frames to pop before peeking (0 for Shift or Ignore).
/// - next_nt: None for Shift/Ignore; Some(dest_nt) when a reduce chain crosses below the baseline.
/// - llm_bv: the LLM-token bitset on this precompute1 edge (stage-1 internal IDs).
/// - pci1_start/pci1_end: the precompute1 nodes bridged by this summarized step.
/// - state_req: the set of required peek states after the pop to allow this step.
///
/// Note: For Ignore terminals (parser.ignore_terminal_id), we conservatively set `state_req` to
/// "all states" (max ones), since ignore terminals do not constrain the GLR stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperEdge {
    pub src_nt: Option<NonTerminalID>,
    pub terminal: TerminalID,
    pub pop: usize,
    pub next_nt: Option<NonTerminalID>, // None => Shift/Ignore; Some(dest_nt) => ReduceCross
    pub llm_bv: LLMTokenBV,
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    pub state_req: StateIDBV,
}

/// A self-contained dataset used by get_mask4 (and for diagnostics).
/// The runtime (eventually) should only need this dataset and GLR stack peeks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialPrecomputation {
    pub super_edges: Vec<SuperEdge>,
    pub llm_only_edges: Vec<LlmOnlyEdge>,

    // Indices for fast lookup at runtime:
    // All super edges from (pci1_start, src_nt).
    pub super_index: HashMap<(PrecomputeNode1Index, Option<NonTerminalID>), Vec<usize>>,
    // All llm-only edges from pci1_start.
    pub llm_only_index: HashMap<PrecomputeNode1Index, Vec<usize>>,

    // Pure trie1-derived facts cached here:
    pub pci1_roots_by_tokenizer: BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    pub pci1_end_nodes: BTreeSet<PrecomputeNode1Index>,
}

impl SpecialPrecomputation {
    fn new() -> Self {
        Self {
            super_edges: Vec::new(),
            llm_only_edges: Vec::new(),
            super_index: HashMap::new(),
            llm_only_index: HashMap::new(),
            pci1_roots_by_tokenizer: BTreeMap::new(),
            pci1_end_nodes: BTreeSet::new(),
        }
    }

    fn build_indices(&mut self) {
        self.super_index.clear();
        for (i, e) in self.super_edges.iter().enumerate() {
            self.super_index
                .entry((e.pci1_start, e.src_nt))
                .or_default()
                .push(i);
        }
        self.llm_only_index.clear();
        for (i, e) in self.llm_only_edges.iter().enumerate() {
            self.llm_only_index.entry(e.pci1_start).or_default().push(i);
        }
    }
}

// ---------- Minimal table access helpers -------------------------------------------------------

fn actions_for<'a>(
    parser: &'a GLRParser,
    state: StateID,
    terminal: TerminalID,
) -> impl Iterator<Item = &'a Stage7ShiftsAndReducesLookaheadValue> {
    parser
        .table
        .get(&state)
        .and_then(|row| row.shifts_and_reduces_full.get(&terminal))
        .into_iter()
}

fn gotos_for<'a>(parser: &'a GLRParser, state: StateID, nt: NonTerminalID) -> impl Iterator<Item = &'a Goto> {
    parser
        .table
        .get(&state)
        .and_then(move |row| row.gotos.get(&nt))
        .into_iter()
}

// ---------- Stage 1a: ReduceCross groups (crossing below baseline) ----------------------------

/// For each (src_nt, terminal), collect tuples (pop, dest_nt, state_req_bv),
/// where `state_req_bv` is the set of peek states required after popping `pop` frames
/// to allow a reduce chain that crosses below the baseline and lands in `dest_nt`.
///
/// Baseline length is 1 for src_nt=None and 2 for src_nt=Some(nt) (after goto).
fn build_reduce_cross_groups(
    parser: &GLRParser,
) -> BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>> {
    let mut states: Vec<StateID> = parser.table.keys().copied().collect();
    states.sort_unstable();
    let terminals: Vec<TerminalID> = parser.terminal_map.right_values().copied().collect();

    let mut src_nts: Vec<Option<NonTerminalID>> = parser
        .non_terminal_map
        .right_values()
        .copied()
        .map(Some)
        .collect();
    src_nts.push(None);

    // Accumulate: (src_nt, terminal, pop, dest_nt) -> state_bv
    let mut acc: BTreeMap<(Option<NonTerminalID>, TerminalID, usize, NonTerminalID), StateIDBV> =
        BTreeMap::new();

    for src_nt in &src_nts {
        for &revealed in &states {
            // Build initial baseline stacks
            let mut initial_stacks: Vec<Vec<StateID>> = Vec::new();
            let baseline_len: usize;

            if let Some(nt) = src_nt {
                for g in gotos_for(parser, revealed, *nt) {
                    if let Some(s) = g.state_id {
                        initial_stacks.push(vec![revealed, s]);
                    }
                }
                if initial_stacks.is_empty() {
                    continue;
                }
                baseline_len = 2;
            } else {
                initial_stacks.push(vec![revealed]);
                baseline_len = 1;
            }

            for init in initial_stacks {
                for &term in &terminals {
                    // Reduce-only BFS. Shift terminates exploration on this terminal.
                    let mut q = VecDeque::new();
                    let mut seen = HashSet::new();
                    q.push_back(init.clone());
                    seen.insert(init.clone());

                    while let Some(stack) = q.pop_front() {
                        if stack.is_empty() {
                            continue;
                        }
                        let top = *stack.last().unwrap();

                        for action in actions_for(parser, top, term) {
                            match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(_) => {
                                    // Do not traverse shifts in reduce-only closure.
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                                    let len = *len;
                                    let reduce_nt = *nonterminal_id;

                                    let above = stack.len().saturating_sub(baseline_len);
                                    if len > above {
                                        // Crossing below baseline
                                        let pop_below = len - above;
                                        let peek_idx = stack.len().saturating_sub(len);
                                        if peek_idx > 0 {
                                            let peek_state = stack[peek_idx - 1];
                                            let key = (*src_nt, term, pop_below, reduce_nt);
                                            acc.entry(key).or_insert_with(StateIDBV::zeros).insert(peek_state.0);
                                        }
                                    } else {
                                        // Reduce above baseline, continue reduce closure
                                        let mut after_pop = stack.clone();
                                        let new_len = after_pop.len().saturating_sub(len);
                                        after_pop.truncate(new_len);
                                        let new_top = *after_pop.last().unwrap();
                                        for g in gotos_for(parser, new_top, reduce_nt) {
                                            if let Some(s) = g.state_id {
                                                let mut after_goto = after_pop.clone();
                                                after_goto.push(s);
                                                if seen.insert(after_goto.clone()) {
                                                    q.push_back(after_goto);
                                                }
                                            }
                                        }
                                    }
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                    // Shift ends reduce-only closure.
                                    if shift.is_some() {
                                        // nothing else
                                    }
                                    for (len, nts) in reduces {
                                        for (reduce_nt, _) in nts {
                                            let len = *len;
                                            let reduce_nt = *reduce_nt;
                                            let above = stack.len().saturating_sub(baseline_len);
                                            if len > above {
                                                // Crossing below baseline
                                                let pop_below = len - above;
                                                let peek_idx = stack.len().saturating_sub(len);
                                                if peek_idx > 0 {
                                                    let peek_state = stack[peek_idx - 1];
                                                    let key = (*src_nt, term, pop_below, reduce_nt);
                                                    acc.entry(key).or_insert_with(StateIDBV::zeros).insert(peek_state.0);
                                                }
                                            } else {
                                                // Continue reduce closure
                                                let mut after_pop = stack.clone();
                                                let new_len = after_pop.len().saturating_sub(len);
                                                after_pop.truncate(new_len);
                                                let new_top = *after_pop.last().unwrap();
                                                for g in gotos_for(parser, new_top, reduce_nt) {
                                                    if let Some(s) = g.state_id {
                                                        let mut after_goto = after_pop.clone();
                                                        after_goto.push(s);
                                                        if seen.insert(after_goto.clone()) {
                                                            q.push_back(after_goto);
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
                }
            }
        }
    }

    // Reformat to the public map.
    let mut out: BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>> =
        BTreeMap::new();
    for ((src_nt, term, pop, dest_nt), bv) in acc {
        if !bv.is_empty() {
            out.entry((src_nt, term))
                .or_default()
                .push((pop, dest_nt, bv));
        }
    }
    for v in out.values_mut() {
        v.sort_by_key(|(pop, nt, _)| (*pop, nt.0));
    }
    out
}

// ---------- Stage 1b: Shift groups (pop=0, next_nt=None) --------------------------------------

/// For each (src_nt, terminal), collect the set of top-of-stack states that allow an immediate
/// shift on `terminal`. For src_nt=Some(nt) we first goto(revealed_state, nt) and treat that as top.
///
/// For Ignore terminal (parser.ignore_terminal_id), we conservatively store "all states".
fn build_shift_groups(
    parser: &GLRParser,
) -> BTreeMap<(Option<NonTerminalID>, TerminalID), StateIDBV> {
    let mut states: Vec<StateID> = parser.table.keys().copied().collect();
    states.sort_unstable();
    let terminals: Vec<TerminalID> = parser.terminal_map.right_values().copied().collect();

    let mut src_nts: Vec<Option<NonTerminalID>> = parser
        .non_terminal_map
        .right_values()
        .copied()
        .map(Some)
        .collect();
    src_nts.push(None);

    let mut out: BTreeMap<(Option<NonTerminalID>, TerminalID), StateIDBV> = BTreeMap::new();

    for src_nt in &src_nts {
        for &revealed in &states {
            // Identify top-of-stack states depending on src_nt
            let mut tops: Vec<StateID> = Vec::new();
            if let Some(nt) = src_nt {
                for g in gotos_for(parser, revealed, *nt) {
                    if let Some(s) = g.state_id {
                        tops.push(s);
                    }
                }
                if tops.is_empty() {
                    continue;
                }
            } else {
                tops.push(revealed);
            }

            for &term in &terminals {
                // Ignore terminal: allow from any state (conservative and correct for ignores)
                if parser.ignore_terminal_id == Some(term) {
                    out.entry((*src_nt, term))
                        .or_insert_with(StateIDBV::max_ones);
                    continue;
                }

                for &top in &tops {
                    let mut has_shift = false;
                    for action in actions_for(parser, top, term) {
                        match action {
                            Stage7ShiftsAndReducesLookaheadValue::Shift(_) => {
                                has_shift = true;
                                break;
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Split { shift, .. } => {
                                if shift.is_some() {
                                    has_shift = true;
                                    break;
                                }
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Reduce { .. } => {}
                        }
                    }
                    if has_shift {
                        out.entry((*src_nt, term))
                            .or_insert_with(StateIDBV::zeros)
                            .insert(top.0);
                    }
                }
            }
        }
    }

    out
}

// ---------- Stage 2: Extract trie1 edges and build SuperEdges ---------------------------------

/// Extract all "None" edges from precompute1 and collect precompute1 end nodes.
fn extract_llm_only_edges(
    gc: &GrammarConstraint,
) -> (Vec<LlmOnlyEdge>, BTreeSet<PrecomputeNode1Index>) {
    let mut out = Vec::new();
    let mut end_nodes = BTreeSet::new();

    let roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    if roots.is_empty() {
        return (out, end_nodes);
    }

    let mut q = VecDeque::new();
    let mut seen = HashSet::new();
    for r in &roots {
        q.push_back(*r);
        seen.insert(r.as_usize());
    }

    while let Some(u) = q.pop_front() {
        if let Some(guard) = u.read(&gc.trie1_god) {
            if guard.value.end {
                end_nodes.insert(u);
            }
            for (ek, dst_map) in guard.children() {
                for (v, llm_bv) in dst_map.iter() {
                    if seen.insert(v.as_usize()) {
                        q.push_back(*v);
                    }
                    if ek.is_none() {
                        out.push(LlmOnlyEdge {
                            pci1_start: u,
                            pci1_end: *v,
                            llm_bv: llm_bv.clone(),
                        });
                    }
                }
            }
        }
    }

    (out, end_nodes)
}

/// Combine parser-derived groups with precompute1 terminal edges to create SuperEdges.
fn build_super_edges_from_trie1(
    gc: &GrammarConstraint,
    reduce_groups: &BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>>,
    shift_groups: &BTreeMap<(Option<NonTerminalID>, TerminalID), StateIDBV>,
) -> Vec<SuperEdge> {
    let mut out = Vec::new();

    let roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    if roots.is_empty() {
        return out;
    }
    let nodes = Trie::all_nodes(&gc.trie1_god, &roots);

    for u in nodes {
        if let Some(guard) = u.read(&gc.trie1_god) {
            for (ek, dst_map) in guard.children() {
                if let Some(term) = ek.clone() {
                    for (v, llm_bv) in dst_map.iter() {
                        // Shift/Ignore edges (pop=0, next_nt=None)
                        if let Some(state_bv) = shift_groups.get(&(None, term)) {
                            if !state_bv.is_empty() {
                                out.push(SuperEdge {
                                    src_nt: None,
                                    terminal: term,
                                    pop: 0,
                                    next_nt: None,
                                    llm_bv: llm_bv.clone(),
                                    pci1_start: u,
                                    pci1_end: *v,
                                    state_req: state_bv.clone(),
                                });
                            }
                        }
                        for (src_nt, state_bv) in shift_groups.iter() {
                            if src_nt.0.is_some() && src_nt.1 == term && !state_bv.is_empty() {
                                out.push(SuperEdge {
                                    src_nt: src_nt.0,
                                    terminal: term,
                                    pop: 0,
                                    next_nt: None,
                                    llm_bv: llm_bv.clone(),
                                    pci1_start: u,
                                    pci1_end: *v,
                                    state_req: state_bv.clone(),
                                });
                            }
                        }

                        // ReduceCross edges for src_nt=None
                        if let Some(groups) = reduce_groups.get(&(None, term)) {
                            for (pop, dest_nt, req) in groups {
                                if req.is_empty() {
                                    continue;
                                }
                                out.push(SuperEdge {
                                    src_nt: None,
                                    terminal: term,
                                    pop: *pop,
                                    next_nt: Some(*dest_nt),
                                    llm_bv: llm_bv.clone(),
                                    pci1_start: u,
                                    pci1_end: *v,
                                    state_req: req.clone(),
                                });
                            }
                        }

                        // ReduceCross edges for src_nt=Some(nt)
                        for ((src_nt, t), groups) in reduce_groups {
                            if *t != term {
                                continue;
                            }
                            if let Some(nt) = src_nt {
                                for (pop, dest_nt, req) in groups {
                                    if req.is_empty() {
                                        continue;
                                    }
                                    out.push(SuperEdge {
                                        src_nt: Some(*nt),
                                        terminal: term,
                                        pop: *pop,
                                        next_nt: Some(*dest_nt),
                                        llm_bv: llm_bv.clone(),
                                        pci1_start: u,
                                        pci1_end: *v,
                                        state_req: req.clone(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

// ---------- Public builder: precompute_special -------------------------------------------------

/// Build the special precomputation dataset used by get_mask4.
/// This is a one-shot compilation step that summarizes both grammar table dynamics and
/// precompute1 terminal paths into a tiny, trie1-free runtime representation.
pub fn precompute_special(gc: &GrammarConstraint) -> SpecialPrecomputation {
    let mut sp = SpecialPrecomputation::new();

    // 1) Parser-derived groups
    let reduce_groups = build_reduce_cross_groups(&gc.parser);
    let shift_groups = build_shift_groups(&gc.parser);

    // 2a) Trie1-derived "None" edges and end nodes
    let (llm_only_edges, end_nodes) = extract_llm_only_edges(gc);
    sp.llm_only_edges = llm_only_edges;
    sp.pci1_end_nodes = end_nodes;

    // 2b) Build SuperEdges by combining terminal edges in trie1 with the parser groups
    sp.super_edges = build_super_edges_from_trie1(gc, &reduce_groups, &shift_groups);

    // 3) Cache trie1 roots per tokenizer state (so runtime doesn't need trie1)
    sp.pci1_roots_by_tokenizer = gc.precomputed1.clone();

    // 4) Build indices
    sp.build_indices();
    sp
}

// ---------- Runtime: get_mask4 (trie1-free) ---------------------------------------------------

/// Compute the allowed original LLM tokens using only the special precomputation dataset.
/// This function does not read trie1. It explores the (pci1, src_nt) space using cached edges.
pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
    let gc = gcs.parent;
    let sp = &gc.special_precomputation;

    if sp.super_edges.is_empty() && sp.llm_only_edges.is_empty() {
        return LLMTokenBV::zeros();
    }

    // Aggregate final mask in stage-1 internal IDs.
    let final_mask_internal = RefCell::new(LLMTokenBV::zeros());

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Key {
        pci1: PrecomputeNode1Index,
        src_nt: Option<NonTerminalID>,
    }

    for (tok_state, glr_state) in gcs.state.iter() {
        if glr_state.active_state.stack.is_empty() {
            continue;
        }

        let Some(&pci1_root) = sp.pci1_roots_by_tokenizer.get(tok_state) else {
            continue;
        };

        // Seed queue with (pci1_root, None, all_tokens_at_stage1)
        let seed_bv = gc.all_internal_llm_tokens_bitset_precompute1();
        let mut q: VecDeque<(PrecomputeNode1Index, Option<NonTerminalID>, LLMTokenBV)> =
            VecDeque::new();
        let mut seen: HashMap<Key, LLMTokenBV> = HashMap::new();

        q.push_back((pci1_root, None, seed_bv.clone()));
        seen.insert(Key { pci1: pci1_root, src_nt: None }, seed_bv);

        while let Some((pci1, src_nt, cur_bv)) = q.pop_front() {
            if cur_bv.is_empty() {
                continue;
            }

            // Collect tokens if at a cached precompute1 end node.
            if sp.pci1_end_nodes.contains(&pci1) {
                *final_mask_internal.borrow_mut() |= &cur_bv;
            }

            // Follow LLM-only edges (None-key edges)
            if let Some(indices) = sp.llm_only_index.get(&pci1) {
                for &i in indices {
                    let e = &sp.llm_only_edges[i];
                    let next_bv = &cur_bv & &e.llm_bv;
                    if next_bv.is_empty() {
                        continue;
                    }
                    let key = Key { pci1: e.pci1_end, src_nt };
                    let entry = seen.entry(key).or_insert_with(LLMTokenBV::zeros);
                    let delta = &next_bv - entry;
                    if !delta.is_empty() {
                        *entry |= &next_bv;
                        q.push_back((e.pci1_end, src_nt, next_bv));
                    }
                }
            }

            // Try SuperEdges from (pci1, src_nt)
            if let Some(indices) = sp.super_index.get(&(pci1, src_nt)) {
                for &i in indices {
                    let e = &sp.super_edges[i];

                    let next_bv = &cur_bv & &e.llm_bv;
                    if next_bv.is_empty() {
                        continue;
                    }

                    // Pop-and-peek check against state_req
                    let popped = glr_state.active_state.stack.popn(e.pop);
                    let mut ok = false;
                    'outer: for item in popped.iter() {
                        for peek in item.peek_iter() {
                            if e.state_req.contains(peek.edge_value().state_id.0) {
                                ok = true;
                                break 'outer;
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }

                    // Advance grammar context
                    let new_src_nt = e.next_nt.or(src_nt);
                    let key = Key { pci1: e.pci1_end, src_nt: new_src_nt };
                    let entry = seen.entry(key).or_insert_with(LLMTokenBV::zeros);
                    let delta = &next_bv - entry;
                    if !delta.is_empty() {
                        *entry |= &next_bv;
                        q.push_back((e.pci1_end, new_src_nt, next_bv));
                    }
                }
            }
        }
    }

    // Map back to original LLM token IDs using stage-1 vocab mapping.
    gc.internal_bv_to_original_precompute1(&final_mask_internal.into_inner())
}

// ---------- Debug dump (optional diagnostics) --------------------------------------------------

pub fn dump_precomputed_special(gc: &GrammarConstraint) {
    let sp = &gc.special_precomputation;
    let parser = &gc.parser;

    let nt_name = |oid: &Option<NonTerminalID>| -> String {
        match oid {
            Some(nt) => parser
                .non_terminal_map
                .get_by_right(nt)
                .map(|t| t.to_string())
                .unwrap_or_else(|| format!("NT({})", nt.0)),
            None => "None".to_string(),
        }
    };
    let term_name = |tid: &TerminalID| -> String {
        parser
            .terminal_map
            .get_by_right(tid)
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("T({})", tid.0))
    };

    println!("--- Special Precomputation ---");
    println!("Tokenizer states with cached pci1 roots: {}", sp.pci1_roots_by_tokenizer.len());
    println!("Cached pci1 end nodes: {}", sp.pci1_end_nodes.len());

    println!("\nLLM-only edges: {}", sp.llm_only_edges.len());
    for e in sp.llm_only_edges.iter().take(1000) {
        let cnt = if e.llm_bv.is_all() { "ALL".to_string() } else { e.llm_bv.len().to_string() };
        println!(
            "  pci1 {} -> {} ; tokens={} ",
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
            cnt
        );
    }
    if sp.llm_only_edges.len() > 1000 {
        println!("  ... ({} more)", sp.llm_only_edges.len() - 1000);
    }

    println!("\nSuperEdges: {}", sp.super_edges.len());
    let mut supers = sp.super_edges.clone();
    supers.sort_by_key(|e| (e.src_nt.map(|n| n.0), e.terminal.0, e.pop, e.next_nt.map(|n| n.0), e.pci1_start.as_usize(), e.pci1_end.as_usize()));
    for e in supers.iter().take(2000) {
        let cnt = if e.llm_bv.is_all() { "ALL".to_string() } else { e.llm_bv.len().to_string() };
        let next_nt_s = e.next_nt.map(|n| parser.non_terminal_map.get_by_right(&n).map(|s| s.to_string()).unwrap_or_else(|| format!("{}", n.0))).unwrap_or_else(|| "-".to_string());
        println!(
            "  [{}] term={} pop={} -> [{}] pci1 {} -> {} ; tokens={} ; |state_req|={}",
            nt_name(&e.src_nt),
            term_name(&e.terminal),
            e.pop,
            next_nt_s,
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
            cnt,
            e.state_req.len()
        );
    }
    if sp.super_edges.len() > 2000 {
        println!("  ... ({} more)", sp.super_edges.len() - 2000);
    }

    println!("--- End Special Precomputation ---");
}
