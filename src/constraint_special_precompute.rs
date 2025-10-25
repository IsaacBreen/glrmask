//! Special precomputation for fast mask (get_mask4).
//!
//! What we build
//! -------------
//! We construct a compact, self-contained graph whose nodes summarize two axes of context:
//!   - grammar context: src_nt ∈ {None} ∪ NonTerminals
//!   - LLM-token path context: a precompute1 node index (pci1)
//!
//! Edges in this graph are of two kinds:
//!   1) LLM-only edges: (src_nt, pci1_a) --[llm_bv]--> (src_nt, pci1_b)
//!      - They come from precompute1 "None" edges; they don't change src_nt.
//!   2) Super edges: (src_nt, pci1_a) --[terminal, pop, dest_nt, state_req, llm_bv]--> (Some(dest_nt), pci1_b)
//!      - They summarize "reductions below baseline" that become possible upon consuming `terminal`.
//!      - pop is the number of frames that must be popped below baseline (>= 1).
//!      - state_req is a bitset of grammar states that must be present at the peek point after popping `pop` frames.
//!      - They are produced by combining:
//!          • precompute1 edges labeled with Some(terminal), and
//!          • a table-derived summary of “reduce-cross” events grouped by (src_nt, terminal).
//!
//! At runtime, get_mask4 explores only this special graph. It never consults the trie1 arena.
//! It intersects the running LLMTokenBV as it follows edges and performs GSS pop-and-peek checks
//! only on super edges. Whenever the current pci1 index is an "end" node (captured ahead of time),
//! it unions the running LLMTokenBV into the final mask.
//!
//! Correctness sketch
//! ------------------
//! Let T be an LLM token, whose internal sub-parts (segments) induce a walk over precompute1.
//! The precompute1 "None" edges correspond exactly to moving across T's bytes where no grammar
//! token is consumed; we capture those via LLM-only edges and intersect the running LLM-token set.
//!
//! When a precompute1 edge consumes a grammar terminal `a` (Some(a)), either it shifts (no below-baseline
//! reduction) or reduces. All “reduce-cross” outcomes for (src_nt, revealed_state, a) that pass below
//! baseline are summarized by a finite set of (pop, dest_nt) groups, one for each (src_nt, a); each group
//! accumulates the set of revealed_state values (compressed as a bitset) that allow that crossing.
//! This grouping is a lossless compression of all per-state facts. Because the “below-baseline” pop count
//! is defined relative to the current baseline, it is invariant to any number of preceding above-baseline
//! shifts (including those along earlier terminals); only the revealed_state and the terminal matter.
//! At runtime, a single pop-and-peek check against `state_req` is sufficient and necessary for validity.
//!
//! Therefore, any path over precompute1 that leads to a below-baseline reduction is represented in our
//! special graph by exactly one super edge (with the same pci1 start/end, LLMBV on the precompute1 edge,
//! and the (pop, dest_nt, state_req) appropriate to the terminal and src_nt). LLM-only edges preserve
//! the LLMBV constraints between terminal-consumption sites. Because we union the running LLMBV into the
//! final mask whenever we reach an end pci1, the set of tokens accepted by get_mask4 equals the set accepted
//! by the original trie3-based evaluation (get_mask3), assuming identical GLR table and tokenizer.
//!
//! Implementation notes
//! --------------------
//! - We precompute a "crossing groups" map: for each (src_nt, terminal), a small list of (pop, dest_nt, state_req)
//!   where state_req is the set of revealed_state values that allow that crossing. This is built by a bounded
//!   BFS over reduce-only closures per revealed_state and terminal.
//! - We extract all precompute1 “None” edges as LLM-only edges once.
//! - For every precompute1 edge labeled Some(terminal), we emit one super edge per crossing group (src_nt, terminal).
//!   We do not need to explicitly pre-propagate shift “escapes”: their effect is subsumed by the revealed_state groupings.
//! - We also cache:
//!     • which pci1 nodes are “end” nodes,
//!     • the starting pci1 root per tokenizer state,
//!   so that get_mask4 relies solely on this precomputed structure.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::constraint::{
    GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index, StateIDBV,
};
use crate::datastructures::trie::Trie;
use crate::glr::parser::GLRParser;
use crate::glr::table::{Goto, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID};
use crate::types::{TerminalID, TokenizerStateID};

/// Escape or ReduceCross resulting from (src_nt, revealed_state, terminal) on the parser table.
///
/// We only keep ReduceCross in the final graph; Escapes are implicitly accounted for by grouping
/// below-baseline events by revealed_state and terminal (see correctness sketch above).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalEdgeKind {
    /// We can shift this terminal before crossing below baseline; pushes the listed states.
    /// Not used in the final graph; kept here for completeness and debug.
    Escape { push_states: Vec<StateID> },
    /// We reduce across baseline by `pop` frames (below baseline), landing in `dest_nt`.
    ReduceCross { pop: usize, dest_nt: NonTerminalID },
}

/// Raw per-(src_nt, revealed_state, terminal) fact (for debug/inspection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalEdge {
    pub src_nt: Option<NonTerminalID>,
    pub revealed_state: StateID,
    pub terminal: TerminalID,
    pub kind: NormalEdgeKind,
}

/// A "None-terminal" precompute1 transition: move in trie1 and intersect tokens; no grammar action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOnlyEdge {
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    pub llm_bv: LLMTokenBV,
}

/// A compact, precomputed "super edge".
/// Preconditions for taking this edge at runtime:
///   - We are at grammar context `src_nt` and precompute1 node `pci1_start`.
///   - Intersect current tokens with `llm_bv` (must be non-empty).
///   - Pop `pop` frames from GSS baseline and peek a state contained in `state_req`.
/// Then we land in grammar context Some(dest_nt) and precompute1 node `pci1_end`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperEdge {
    pub src_nt: Option<NonTerminalID>,
    pub terminal: TerminalID,
    pub pop: usize,
    pub dest_nt: NonTerminalID,
    pub llm_bv: LLMTokenBV,
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    pub state_req: StateIDBV,
}

/// Container with indices to speed up runtime lookups.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialPrecomputation {
    // Stage1 (debug only)
    pub normal_edges: Vec<NormalEdge>,

    // Stage2 results
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
            normal_edges: Vec::new(),
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

// ---------- Stage 1: Build Normal Edges + Crossing Groups -------------------------------------

/// Enumerate per-(src_nt, revealed_state, terminal) facts.
fn build_normal_edges(parser: &GLRParser) -> Vec<NormalEdge> {
    let mut out = Vec::new();

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

    for src_nt in &src_nts {
        for &revealed_state in &states {
            for &terminal in &terminals {
                // Establish initial baseline and starting stacks:
                // - If src_nt is Some(nt), we do an initial goto (if any). Baseline length is 2.
                // - If None, baseline is a single frame [revealed_state].
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
                    // BFS over reduce-only transformations for this terminal.
                    // Shift terminates exploration on this terminal and produces an Escape edge.
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
                                Stage7ShiftsAndReducesLookaheadValue::Shift(next_state) => {
                                    // Shift consumes terminal; record Escape with the tail beyond baseline.
                                    let mut shifted = stack.clone();
                                    shifted.push(*next_state);
                                    let tail: Vec<StateID> = shifted[baseline_len..].to_vec();
                                    out.push(NormalEdge {
                                        src_nt: *src_nt,
                                        revealed_state,
                                        terminal,
                                        kind: NormalEdgeKind::Escape { push_states: tail },
                                    });
                                    // No continuation after shift for this terminal.
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
                                        out.push(NormalEdge {
                                            src_nt: *src_nt,
                                            revealed_state,
                                            terminal,
                                            kind: NormalEdgeKind::ReduceCross {
                                                pop: pop_below,
                                                dest_nt: reduce_nt,
                                            },
                                        });
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
                                    if let Some(next_state) = shift {
                                        let mut shifted = stack.clone();
                                        shifted.push(*next_state);
                                        let tail: Vec<StateID> = shifted[baseline_len..].to_vec();
                                        out.push(NormalEdge {
                                            src_nt: *src_nt,
                                            revealed_state,
                                            terminal,
                                            kind: NormalEdgeKind::Escape { push_states: tail },
                                        });
                                    }
                                    for (len, nts) in reduces {
                                        for (nt, _) in nts {
                                            let reduce_nt = *nt;
                                            let len = *len;
                                            let above_baseline = stack.len().saturating_sub(baseline_len);
                                            if len > above_baseline {
                                                let pop_below = len - above_baseline;
                                                out.push(NormalEdge {
                                                    src_nt: *src_nt,
                                                    revealed_state,
                                                    terminal,
                                                    kind: NormalEdgeKind::ReduceCross {
                                                        pop: pop_below,
                                                        dest_nt: reduce_nt,
                                                    },
                                                });
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

    out
}

/// Group all ReduceCross facts by (src_nt, terminal, pop, dest_nt), compressing revealed_state into a bitset.
fn build_crossing_groups(
    parser: &GLRParser,
    normal_edges: &[NormalEdge],
) -> BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>> {
    // Accumulate state sets per (src_nt, terminal, pop, dest_nt).
    let mut acc: BTreeMap<(Option<NonTerminalID>, TerminalID, usize, NonTerminalID), StateIDBV> =
        BTreeMap::new();

    for e in normal_edges {
        if let NormalEdgeKind::ReduceCross { pop, dest_nt } = e.kind {
            let key = (e.src_nt, e.terminal, pop, dest_nt);
            let entry = acc.entry(key).or_insert_with(StateIDBV::zeros);
            entry.insert(e.revealed_state.0);
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

// ---------- Stage 2: Build special graph from precompute1 + crossing groups -------------------

/// Extract all "None-terminal" edges in precompute1 (LLM-only transitions).
fn extract_llm_only_edges(gc: &GrammarConstraint) -> (Vec<LlmOnlyEdge>, BTreeSet<PrecomputeNode1Index>) {
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
/// with crossing groups from the GLR table.
fn build_super_edges_from_trie1(
    gc: &GrammarConstraint,
    crossing_groups: &BTreeMap<(Option<NonTerminalID>, TerminalID), Vec<(usize, NonTerminalID, StateIDBV)>>,
) -> Vec<SuperEdge> {
    let mut out = Vec::new();

    // Collect all nodes reachable from roots.
    let roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    if roots.is_empty() {
        return out;
    }
    let nodes = Trie::all_nodes(&gc.trie1_god, &roots);

    // For each node, read children; for each Some(terminal) edge, create super edges for all src_nt groups.
    for idx in nodes {
        if let Some(guard) = idx.read(&gc.trie1_god) {
            for (ek, dest_map) in guard.children() {
                let terminal_opt = ek.clone();
                if let Some(term) = terminal_opt {
                    // For each child under this terminal, produce edges keyed by src_nt groupings.
                    for (child_idx, edge_bv) in dest_map.iter() {
                        // For src_nt in {None} ∪ NonTerminals that have non-empty crossing groups for this terminal:
                        // This automatically prunes src_nt that do not allow any crossing on this terminal.
                        // We enumerate keys in crossing_groups and match on `term`.
                        for ((src_nt, term_k), groups) in crossing_groups {
                            if *term_k != term {
                                continue;
                            }
                            for (pop, dest_nt, state_req) in groups {
                                // pop >= 1 by construction (crossing below baseline).
                                // Create a single super edge summarizing this possibility.
                                out.push(SuperEdge {
                                    src_nt: *src_nt,
                                    terminal: *term_k,
                                    pop: *pop,
                                    dest_nt: *dest_nt,
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

    out
}

/// Entry point: build the complete SpecialPrecomputation from a GrammarConstraint.
pub fn precompute_special(gc: &GrammarConstraint) -> SpecialPrecomputation {
    let mut sp = SpecialPrecomputation::new();

    // Stage 1: table-driven normal edges and grouped crossings
    sp.normal_edges = build_normal_edges(&gc.parser);
    let crossing_groups = build_crossing_groups(&gc.parser, &sp.normal_edges);

    // Stage 2a: trie1 "None" edges + end nodes
    let (llm_edges, end_nodes) = extract_llm_only_edges(gc);
    sp.llm_only_edges = llm_edges;
    sp.pci1_end_nodes = end_nodes;

    // Stage 2b: build super edges from precompute1 edges with Some(terminal)
    sp.super_edges = build_super_edges_from_trie1(gc, &crossing_groups);

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
///     • Try super edges when (pci1, src_nt) matches. A super edge passes iff:
///           - intersected tokens non-empty, and
///           - GLR pop-and-peek passes: after popping `pop`, a peek state is in `state_req`.
///       Then move to (Some(dest_nt), pci1_end).
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
                    let next_key = Key {
                        pci1: e.pci1_end,
                        src_nt: Some(e.dest_nt),
                    };
                    let entry = seen.entry(next_key).or_insert_with(LLMTokenBV::zeros);
                    let delta = &next_bv - entry;
                    if !delta.is_empty() {
                        *entry |= &next_bv;
                        q.push_back((e.pci1_end, Some(e.dest_nt), next_bv));
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

    println!("\nNormal Edges ({}):", sp.normal_edges.len());
    println!("{:-<120}", "");
    println!(
        "{:<10} | {:<8} | {:<20} | {:<75}",
        "src_nt", "state", "terminal", "kind"
    );
    println!("{:-<120}", "");
    let mut normals = sp.normal_edges.clone();
    normals.sort_by_key(|e| (e.src_nt.map(|n| n.0), e.revealed_state.0, e.terminal.0));
    for e in normals.iter().take(2000) { // cap output
        let kind_s = match e.kind {
            NormalEdgeKind::Escape { ref push_states } => {
                let s: Vec<_> = push_states.iter().map(|s| s.0.to_string()).collect();
                format!("Escape(push=[{}])", s.join(","))
            }
            NormalEdgeKind::ReduceCross { pop, dest_nt } => {
                format!("ReduceCross(pop={}, dest_nt={})", pop, dest_nt.0)
            }
        };
        println!(
            "{:<10} | {:<8} | {:<20} | {}",
            nt_name(&e.src_nt),
            e.revealed_state.0,
            get_term_name(&e.terminal),
            kind_s
        );
    }
    if sp.normal_edges.len() > 2000 {
        println!("... ({} more)", sp.normal_edges.len() - 2000);
    }

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
    println!("{:-<180}", "");
    println!(
        "{:<10} | {:<16} | {:<10} | {:<10} | {:<10} | {:<10} | {:<18} | {:<15}",
        "src_nt",
        "terminal",
        "pop",
        "dest_nt",
        "pci1_from",
        "pci1_to",
        "llm_tokens",
        "state_req(|S|)"
    );
    println!("{:-<180}", "");
    let mut supers = sp.super_edges.clone();
    supers.sort_by_key(|e| {
        (
            e.src_nt.map(|n| n.0),
            e.terminal.0,
            e.pop,
            e.dest_nt.0,
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
        )
    });
    for e in supers.iter().take(2000) {
        let bv_str = if e.llm_bv.is_all() {
            "ALL".to_string()
        } else {
            format!("{} tokens", e.llm_bv.len())
        };
        let state_req_len = e.state_req.len();
        println!(
            "{:<10} | {:<16} | {:<10} | {:<10} | {:<10} | {:<10} | {:<18} | {:<15}",
            nt_name(&e.src_nt),
            get_term_name(&e.terminal),
            e.pop,
            e.dest_nt.0,
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
            bv_str,
            state_req_len
        );
    }
    if sp.super_edges.len() > 2000 {
        println!("... ({} more)", sp.super_edges.len() - 2000);
    }

    println!("\n--- End Special Precomputation Dump ---");
}
