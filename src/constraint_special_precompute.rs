//! Special precomputation for fast mask (get_mask4).
//!
//! Goal
//! -----
//! Build a small, self-contained graph that the runtime can traverse without touching Trie1,
//! yet computes the same token-allowance as the full Trie3 evaluation (get_mask3).
//!
//! Model
//! -----
//! A runtime position is summarized by two independent axes:
//!   - grammar context: src_nt ∈ {None} ∪ NonTerminals
//!   - LLM-token path context: a precompute1 node index (pci1)
//!
//! There are two edge kinds that advance along a precompute1 edge labeled by a grammar terminal:
//!   1) Shift edge (pop = 0): allowed if the current top-of-stack state ∈ state_req.
//!      It does NOT change the grammar context (src_nt remains the same).
//!   2) Reduce-cross edge (pop >= 1): allowed if, after popping `pop` frames from the GLR stack,
//!      a peek state ∈ state_req. It changes grammar context to Some(dest_nt).
//!
//! Additionally, we copy all “None” edges in precompute1 as LLM-only edges: they intersect tokens
//! and move pci1 without any grammar checks.
//!
//! Correctness sketch
//! ------------------
//! Let T be an LLM token. The token imposes a walk over precompute1 starting from the tokenizer's
//! root pci1 node. Along this path, some edges carry a grammar terminal `a`, others carry None.
//! - We represent None edges verbatim: intersect current LLM-token BV and move pci1.
//! - For a terminal edge labeled `a`: either an immediate shift is possible (at or above baseline),
//!   or crossing below the baseline occurs during reduce chains after `a`.
//!   Shift is captured by a (pop=0, state_req=shifting_top_states) edge.
//!   Crossing is captured by grouping all (revealed_state s) that, with terminal `a`, reduce below
//!   the baseline by `pop` and land in `dest_nt`, with `state_req` encoding the required peek state.
//!
//! At runtime, following such a special edge is equivalent to executing the corresponding GLR step,
//! but the only checks required are a pop-and-peek with a small precomputed bitset and intersecting
//! LLM-token sets. Because we collect tokens whenever we land on a precompute1 end node (as in Trie3),
//! the final union equals the get_mask3 result mod the chosen LLMBV stage mapping.

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

/// A "None-terminal" precompute1 transition: move in trie1 and intersect tokens; no grammar action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOnlyEdge {
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    pub llm_bv: LLMTokenBV,
}

/// A compact, precomputed edge corresponding to consuming a grammar terminal on a trie1 edge.
/// Two variants:
/// - Shift: pop = 0, next_nt = None (grammar context unchanged).
/// - Reduce-cross: pop >= 1, next_nt = Some(dest_nt).
///
/// Preconditions for taking this edge at runtime:
///   - We are at grammar context `src_nt` and precompute1 node `pci1_start`.
///   - Intersect current tokens with `llm_bv` (must be non-empty).
///   - Perform a GLR pop of `pop` frames and peek a state contained in `state_req`.
/// Then we land in grammar context `next_nt.or(src_nt)` and precompute1 node `pci1_end`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperEdge {
    pub src_nt: Option<NonTerminalID>,
    pub terminal: TerminalID,
    pub pop: usize,
    pub next_nt: Option<NonTerminalID>, // None for shift; Some(dest_nt) for below-baseline reduce-cross
    pub llm_bv: LLMTokenBV,
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    pub state_req: StateIDBV, // set of required peek states after the pop
}

/// Container with indices to speed up runtime lookups.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialPrecomputation {
    pub super_edges: Vec<SuperEdge>,
    pub llm_only_edges: Vec<LlmOnlyEdge>,

    // Indices for runtime
    pub super_index: HashMap<(PrecomputeNode1Index, Option<NonTerminalID>), Vec<usize>>,
    pub llm_only_index: HashMap<PrecomputeNode1Index, Vec<usize>>,

    // Pure trie1-derived facts stored here so runtime never reads trie1:
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

// ---------- Internal helpers (parser table access) --------------------------------------------

fn get_actions<'a>(
    parser: &'a GLRParser,
    state_id: StateID,
    terminal_id: TerminalID,
) -> Vec<&'a Stage7ShiftsAndReducesLookaheadValue> {
    let mut actions = Vec::new();
    if let Some(row) = parser.table.get(&state_id) {
        if let Some(action) = row.shifts_and_reduces_full.get(&terminal_id) {
            actions.push(action);
        }
    }
    actions
}

fn get_gotos<'a>(parser: &'a GLRParser, state_id: StateID, nt_id: NonTerminalID) -> Vec<&'a Goto> {
    if let Some(row) = parser.table.get(&state_id) {
        row.gotos.get(&nt_id).map(|g| vec![g]).unwrap_or_default()
    } else {
        vec![]
    }
}

// ---------- Stage 1a: Build Reduce-Cross (below-baseline) crossing groups ---------------------

/// Group all ReduceCross facts by (src_nt, terminal, pop, dest_nt), compressing required peek states
/// into a bitset. For src_nt = None the baseline length is 1; for Some(nt) it's 2 (after a goto).
///
/// Returns: (src_nt, terminal) -> Vec<(pop, dest_nt, state_req_top_after_pop)>
fn build_reduce_cross_groups(
    parser: &GLRParser,
) -> BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>> {
    // Enumerate states deterministically.
    let mut states: Vec<StateID> = parser.table.keys().copied().collect();
    states.sort_unstable();

    // All terminals.
    let terminals: Vec<TerminalID> = parser.terminal_map.right_values().copied().collect();

    // src_nt: None + all nonterminals.
    let mut src_nts: Vec<Option<NonTerminalID>> = parser
        .non_terminal_map
        .right_values()
        .copied()
        .map(Some)
        .collect();
    src_nts.push(None);

    // Accumulate per (src_nt, terminal, pop, dest_nt) => set(top_state_after_pop).
    let mut acc: BTreeMap<(Option<NonTerminalID>, TerminalID, usize, NonTerminalID), StateIDBV> =
        BTreeMap::new();

    for src_nt in &src_nts {
        for &revealed_state in &states {
            // Initialize baseline stack and baseline length
            let mut initial_stacks: Vec<Vec<StateID>> = Vec::new();
            let baseline_len: usize;

            match src_nt {
                Some(nt) => {
                    let gotos = get_gotos(parser, revealed_state, *nt);
                    if gotos.is_empty() {
                        continue;
                    }
                    for g in gotos {
                        if let Some(next) = g.state_id {
                            initial_stacks.push(vec![revealed_state, next]);
                        }
                    }
                    baseline_len = 2;
                }
                None => {
                    initial_stacks.push(vec![revealed_state]);
                    baseline_len = 1;
                }
            }

            for initial_stack in initial_stacks {
                // For each terminal, BFS reduce-only closure; shift terminates exploration.
                for &terminal in &terminals {
                    let mut q: VecDeque<Vec<StateID>> = VecDeque::new();
                    let mut visited: HashSet<Vec<StateID>> = HashSet::new();
                    q.push_back(initial_stack.clone());
                    visited.insert(initial_stack.clone());

                    while let Some(stack) = q.pop_front() {
                        if stack.is_empty() {
                            continue;
                        }

                        let top_state = *stack.last().unwrap();
                        let actions = get_actions(parser, top_state, terminal);
                        if actions.is_empty() {
                            continue;
                        }

                        for action in actions {
                            match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(_next_state) => {
                                    // Shift: immediate, does not cross baseline; ignore in this reducer-only stage.
                                    // Do not continue exploration through shift for this terminal.
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                    nonterminal_id,
                                    len,
                                    ..
                                } => {
                                    let reduce_nt = *nonterminal_id;
                                    let len = *len;
                                    let above_baseline = stack.len().saturating_sub(baseline_len);
                                    if len > above_baseline {
                                        // Crosses below baseline
                                        let pop_below = len - above_baseline;
                                        // After popping pop_below, the top-of-stack at peek is stack[stack.len() - len - 1].
                                        // That is the state we must require at runtime after pop.
                                        let peek_index = stack.len().saturating_sub(len);
                                        if peek_index > 0 {
                                            let peek_state = stack[peek_index - 1];
                                            let key =
                                                (*src_nt, terminal, pop_below, reduce_nt);
                                            acc.entry(key)
                                                .or_insert_with(StateIDBV::zeros)
                                                .insert(peek_state.0);
                                        }
                                    } else {
                                        // Reduce without crossing baseline; keep exploring reduce chains.
                                        let mut after_pop = stack.clone();
                                        let new_len = after_pop.len().saturating_sub(len);
                                        after_pop.truncate(new_len);
                                        let new_top = *after_pop.last().unwrap();
                                        for goto in get_gotos(parser, new_top, reduce_nt) {
                                            if let Some(goto_state) = goto.state_id {
                                                let mut after_goto = after_pop.clone();
                                                after_goto.push(goto_state);
                                                if visited.insert(after_goto.clone()) {
                                                    q.push_back(after_goto);
                                                }
                                            }
                                        }
                                    }
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                    if let Some(_next_state) = shift {
                                        // As with Shift: do not proceed further on this terminal.
                                    }
                                    for (len, nts) in reduces {
                                        for (nt, _) in nts {
                                            let reduce_nt = *nt;
                                            let len = *len;
                                            let above_baseline =
                                                stack.len().saturating_sub(baseline_len);
                                            if len > above_baseline {
                                                let pop_below = len - above_baseline;
                                                let peek_index = stack.len().saturating_sub(len);
                                                if peek_index > 0 {
                                                    let peek_state = stack[peek_index - 1];
                                                    let key =
                                                        (*src_nt, terminal, pop_below, reduce_nt);
                                                    acc.entry(key)
                                                        .or_insert_with(StateIDBV::zeros)
                                                        .insert(peek_state.0);
                                                }
                                            } else {
                                                let mut after_pop = stack.clone();
                                                let new_len = after_pop.len().saturating_sub(len);
                                                after_pop.truncate(new_len);
                                                let new_top = *after_pop.last().unwrap();
                                                for goto in get_gotos(parser, new_top, reduce_nt) {
                                                    if let Some(goto_state) = goto.state_id {
                                                        let mut after_goto = after_pop.clone();
                                                        after_goto.push(goto_state);
                                                        if visited.insert(after_goto.clone()) {
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

    // Collect into the desired map: (src_nt, terminal) -> Vec<(pop, dest_nt, state_req)>
    let mut out: BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>> =
        BTreeMap::new();

    for ((src_nt, term, pop, dest_nt), state_bv) in acc {
        out.entry((src_nt, term))
            .or_default()
            .push((pop, dest_nt, state_bv));
    }

    // Deterministic ordering within each list.
    for v in out.values_mut() {
        v.sort_by_key(|(pop, dest_nt, _)| (*pop, dest_nt.0));
    }

    out
}

// ---------- Stage 1b: Build Shift groups (pop=0, grammar context unchanged) -------------------

/// Build shift groups: for each (src_nt, terminal), a bitset of top-of-stack states that allow
/// an immediate shift action for that terminal. For src_nt = Some(nt), the "top-of-stack" we
/// check is goto(revealed_state, nt) if it exists.
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
        for &revealed_state in &states {
            // Determine the effective top-of-stack given src_nt.
            let mut top_states: Vec<StateID> = Vec::new();
            match src_nt {
                None => {
                    top_states.push(revealed_state);
                }
                Some(nt) => {
                    let gotos = get_gotos(parser, revealed_state, *nt);
                    for g in gotos {
                        if let Some(next) = g.state_id {
                            top_states.push(next);
                        }
                    }
                    if top_states.is_empty() {
                        continue;
                    }
                }
            }

            for &terminal in &terminals {
                // If any top_state allows a Shift on this terminal, mark that top_state as required.
                for &top_state in &top_states {
                    let actions = get_actions(parser, top_state, terminal);
                    let mut has_shift = false;
                    for action in actions {
                        match action {
                            Stage7ShiftsAndReducesLookaheadValue::Shift(_s) => {
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
                        out.entry((*src_nt, terminal))
                            .or_insert_with(StateIDBV::zeros)
                            .insert(top_state.0);
                    }
                }
            }
        }
    }

    out
}

// ---------- Stage 2: Extract Trie1 edges and build special graph -------------------------------

/// Extract all "None-terminal" edges in precompute1 (LLM-only transitions) and cache end nodes.
fn extract_llm_only_edges(
    gc: &GrammarConstraint,
) -> (Vec<LlmOnlyEdge>, BTreeSet<PrecomputeNode1Index>) {
    let mut out = Vec::new();
    let mut end_nodes: BTreeSet<PrecomputeNode1Index> = BTreeSet::new();

    let roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    if roots.is_empty() {
        return (out, end_nodes);
    }

    // Traverse the precompute1 graph once; record end nodes and None-edges.
    let mut q = VecDeque::new();
    let mut seen = HashSet::new();
    for r in &roots {
        q.push_back(*r);
        seen.insert(r.as_usize());
    }

    while let Some(idx) = q.pop_front() {
        if let Some(guard) = idx.read(&gc.trie1_god) {
            if guard.value.end {
                end_nodes.insert(idx);
            }
            for (ek, dest_map) in guard.children() {
                for (child_idx, edge_bv) in dest_map.iter() {
                    if seen.insert(child_idx.as_usize()) {
                        q.push_back(*child_idx);
                    }
                    if ek.is_none() {
                        out.push(LlmOnlyEdge {
                            pci1_start: idx,
                            pci1_end: *child_idx,
                            llm_bv: edge_bv.clone(),
                        });
                    }
                }
            }
        }
    }

    (out, end_nodes)
}

/// Build super edges by walking precompute1 edges labeled with Some(terminal) and combining
/// with (a) reduce-cross groups from the GLR table; and (b) shift groups (pop=0).
fn build_super_edges_from_trie1(
    gc: &GrammarConstraint,
    reduce_groups: &BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>>,
    shift_groups: &BTreeMap<(Option<NonTerminalID>, TerminalID), StateIDBV>,
) -> Vec<SuperEdge> {
    let mut out = Vec::new();

    // Collect all nodes reachable from roots.
    let roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    if roots.is_empty() {
        return out;
    }
    let nodes = Trie::all_nodes(&gc.trie1_god, &roots);

    // For each node, read children; for each Some(terminal) edge, create super edges for all src_nt groupings.
    for idx in nodes {
        if let Some(guard) = idx.read(&gc.trie1_god) {
            for (ek, dest_map) in guard.children() {
                if let Some(term) = ek.clone() {
                    // For each child under this terminal, produce edges keyed by src_nt groupings.
                    for (child_idx, edge_bv) in dest_map.iter() {
                        // 1) Shift (pop=0, grammar context unchanged)
                        for ((src_nt, term_k), state_bv) in shift_groups {
                            if *term_k != term {
                                continue;
                            }
                            if state_bv.is_empty() {
                                continue;
                            }
                            out.push(SuperEdge {
                                src_nt: *src_nt,
                                terminal: *term_k,
                                pop: 0,
                                next_nt: None,
                                llm_bv: edge_bv.clone(),
                                pci1_start: idx,
                                pci1_end: *child_idx,
                                state_req: state_bv.clone(),
                            });
                        }

                        // 2) Reduce-cross (pop>=1, grammar context becomes Some(dest_nt))
                        if let Some(groups) = reduce_groups.get(&(None, term)) {
                            // include for src_nt=None below; will also include others below
                            for (pop, dest_nt, state_req) in groups {
                                if state_req.is_empty() {
                                    continue;
                                }
                                out.push(SuperEdge {
                                    src_nt: None,
                                    terminal: term,
                                    pop: *pop,
                                    next_nt: Some(*dest_nt),
                                    llm_bv: edge_bv.clone(),
                                    pci1_start: idx,
                                    pci1_end: *child_idx,
                                    state_req: state_req.clone(),
                                });
                            }
                        }
                        // Also for src_nt = Some(nt) cases
                        for ((src_nt, term_k), groups) in reduce_groups {
                            if *term_k != term {
                                continue;
                            }
                            if let Some(nt) = src_nt {
                                for (pop, dest_nt, state_req) in groups {
                                    if state_req.is_empty() {
                                        continue;
                                    }
                                    out.push(SuperEdge {
                                        src_nt: Some(*nt),
                                        terminal: term,
                                        pop: *pop,
                                        next_nt: Some(*dest_nt),
                                        llm_bv: edge_bv.clone(),
                                        pci1_start: idx,
                                        pci1_end: *child_idx,
                                        state_req: state_req.clone(),
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

/// Entry point: build the complete SpecialPrecomputation from a GrammarConstraint.
pub fn precompute_special(gc: &GrammarConstraint) -> SpecialPrecomputation {
    let mut sp = SpecialPrecomputation::new();

    // Stage 1: table-driven groups
    let reduce_groups = build_reduce_cross_groups(&gc.parser);
    let shift_groups = build_shift_groups(&gc.parser);

    // Stage 2a: trie1 "None" edges + end nodes
    let (llm_edges, end_nodes) = extract_llm_only_edges(gc);
    sp.llm_only_edges = llm_edges;
    sp.pci1_end_nodes = end_nodes;

    // Stage 2b: build super edges from precompute1 edges with Some(terminal)
    sp.super_edges = build_super_edges_from_trie1(gc, &reduce_groups, &shift_groups);

    // Roots per tokenizer state
    sp.pci1_roots_by_tokenizer = gc.precomputed1.clone();

    // Build indices for runtime
    sp.build_indices();
    sp
}

// ---------- Runtime: get_mask4 ---------------------------------------------------------------

/// Returns the bitset of original LLM tokens allowed by the current GrammarConstraintState.
///
/// Strategy (self-contained; no trie1 access):
/// - Starting from each active tokenizer state's cached pci1 root and src_nt=None, run a small worklist:
///     • Follow LLM-only edges (no grammar action): intersect tokens, move pci1.
///     • Try super edges when (pci1, src_nt) matches.
///           - Intersect tokens with the edge's llm_bv; if empty, skip.
///           - GLR pop-and-peek passes iff: after popping `pop`, a peek state is in `state_req`.
///       Then move to (next_nt.or(src_nt), pci1_end).
///     • If pci1 is an "end" node, union the state's accumulated token set into final_mask.
/// - Convert internal tokens to original using the precompute1 vocab mapping.
pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
    let gc = gcs.parent;
    let sp = &gc.special_precomputation;

    // If no special precomputation, nothing is allowed.
    if sp.super_edges.is_empty() && sp.llm_only_edges.is_empty() {
        return LLMTokenBV::zeros();
    }

    // Aggregate final mask (internal IDs).
    let final_mask_internal = RefCell::new(LLMTokenBV::zeros());

    // Key for dedup: position (pci1, src_nt).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct Key {
        pci1: PrecomputeNode1Index,
        src_nt: Option<NonTerminalID>,
    }

    // For each active tokenizer state and GSS, run an exploration.
    for (tokenizer_state_id, glr_state) in gcs.state.iter() {
        // Skip empty stacks.
        if glr_state.active_state.stack.is_empty() {
            continue;
        }

        // Seed pci1 root for this tokenizer state from special precompute (no trie1 access).
        let Some(&pci1_root) = sp.pci1_roots_by_tokenizer.get(tokenizer_state_id) else {
            continue;
        };

        // Seed set: start with None and all tokens internally.
        let mut q: VecDeque<(PrecomputeNode1Index, Option<NonTerminalID>, LLMTokenBV)> =
            VecDeque::new();
        let mut seen: HashMap<Key, LLMTokenBV> = HashMap::new();

        let seed_bv = gc.all_internal_llm_tokens_bitset_precompute1();
        q.push_back((pci1_root, None, seed_bv.clone()));
        seen.insert(Key { pci1: pci1_root, src_nt: None }, seed_bv);

        while let Some((pci1, src_nt, cur_bv)) = q.pop_front() {
            if cur_bv.is_empty() {
                continue;
            }

            // If we're at trie1 end (cached), collect tokens.
            if sp.pci1_end_nodes.contains(&pci1) {
                *final_mask_internal.borrow_mut() |= &cur_bv;
            }

            // 1) Follow LLM-only edges (None terminals)
            if let Some(edges) = sp.llm_only_index.get(&pci1) {
                for &i in edges {
                    let e = &sp.llm_only_edges[i];
                    let next_bv = &cur_bv & &e.llm_bv;
                    if next_bv.is_empty() {
                        continue;
                    }
                    let next_key = Key { pci1: e.pci1_end, src_nt };
                    let entry = seen.entry(next_key).or_insert_with(LLMTokenBV::zeros);
                    let delta = &next_bv - entry;
                    if !delta.is_empty() {
                        *entry |= &next_bv;
                        q.push_back((e.pci1_end, src_nt, next_bv));
                    }
                }
            }

            // 2) Try super edges from (pci1, src_nt)
            if let Some(indices) = sp.super_index.get(&(pci1, src_nt)) {
                for &i in indices {
                    let e = &sp.super_edges[i];
                    let next_bv = &cur_bv & &e.llm_bv;
                    if next_bv.is_empty() {
                        continue;
                    }

                    // GSS pop check with state filter
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

                    // Move along super edge:
                    let new_src_nt = e.next_nt.or(src_nt);
                    let next_key = Key {
                        pci1: e.pci1_end,
                        src_nt: new_src_nt,
                    };
                    let entry = seen.entry(next_key).or_insert_with(LLMTokenBV::zeros);
                    let delta = &next_bv - entry;
                    if !delta.is_empty() {
                        *entry |= &next_bv;
                        q.push_back((e.pci1_end, new_src_nt, next_bv));
                    }
                }
            }
        }
    }

    // Convert to original token IDs using the precompute1 vocab mapping.
    gc.internal_bv_to_original_precompute1(&final_mask_internal.into_inner())
}

// ---------- Debug dumping ---------------------------------------------------------------------

pub fn dump_precomputed_special(gc: &GrammarConstraint) {
    let sp = &gc.special_precomputation;
    let parser = &gc.parser;

    println!("--- Special Precomputation Dump ---");

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
    let get_term_name = |term_id: &TerminalID| -> String {
        parser
            .terminal_map
            .get_by_right(term_id)
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("T({})", term_id.0))
    };

    println!("Tokenizer states with pci1 roots: {}", sp.pci1_roots_by_tokenizer.len());
    println!("pci1 end nodes cached: {}", sp.pci1_end_nodes.len());

    println!("\nLLM-only edges ({}):", sp.llm_only_edges.len());
    println!("{:-<120}", "");
    println!("{:<10} | {:<10} | {:<20}", "pci1_from", "pci1_to", "llm_tokens");
    println!("{:-<120}", "");
    let mut llm_edges = sp.llm_only_edges.clone();
    llm_edges.sort_by_key(|e| (e.pci1_start.as_usize(), e.pci1_end.as_usize(), e.llm_bv.len()));
    for e in llm_edges.iter().take(2000) {
        let bv_str = if e.llm_bv.is_all() {
            "ALL".to_string()
        } else {
            format!("{} tokens", e.llm_bv.len())
        };
        println!(
            "{:<10} | {:<10} | {:<20}",
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
            bv_str
        );
    }
    if sp.llm_only_edges.len() > 2000 {
        println!("... ({} more)", sp.llm_only_edges.len() - 2000);
    }

    println!("\nSuper Edges ({}):", sp.super_edges.len());
    println!("{:-<200}", "");
    println!(
        "{:<10} | {:<16} | {:<6} | {:<10} | {:<10} | {:<10} | {:<18} | {:<15}",
        "src_nt",
        "terminal",
        "pop",
        "next_nt",
        "pci1_from",
        "pci1_to",
        "llm_tokens",
        "state_req(|S|)"
    );
    println!("{:-<200}", "");
    let mut supers = sp.super_edges.clone();
    supers.sort_by_key(|e| {
        (
            e.src_nt.map(|n| n.0),
            e.terminal.0,
            e.pop,
            e.next_nt.map(|n| n.0),
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
        )
    });
    for e in supers.iter().take(3000) {
        let bv_str = if e.llm_bv.is_all() {
            "ALL".to_string()
        } else {
            format!("{} tokens", e.llm_bv.len())
        };
        let next_nt_s = e
            .next_nt
            .map(|n| parser.non_terminal_map.get_by_right(&n).map(|s| s.to_string()).unwrap_or_else(|| format!("{}", n.0)))
            .unwrap_or_else(|| "-".to_string());
        let state_req_len = e.state_req.len();
        println!(
            "{:<10} | {:<16} | {:<6} | {:<10} | {:<10} | {:<10} | {:<18} | {:<15}",
            nt_name(&e.src_nt),
            get_term_name(&e.terminal),
            e.pop,
            next_nt_s,
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
            bv_str,
            state_req_len
        );
    }
    if sp.super_edges.len() > 3000 {
        println!("... ({} more)", sp.super_edges.len() - 3000);
    }

    println!("\n--- End Special Precomputation Dump ---");
}
