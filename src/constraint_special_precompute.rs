//! Special precomputation for fast mask (get_mask4).
//!
//! High-level overview
//! -------------------
//! We build two kinds of precomputed data:
//! - Normal edges: state-machine facts parameterized by:
//!     (src_nt: Option<NonTerminalID>, revealed_state: StateID, terminal: TerminalID)
//!   that tell us either:
//!     • Escape: shifting on this terminal (before any below-baseline reduction) pushes these states,
//!     • ReduceCross: a reduce on this terminal crosses the baseline by `pop` frames and ends in `dest_nt`.
//!
//! - Super edges: compact, terminal-annotated transitions that combine:
//!     • the trie1 LLM-token constraints (using precompute1), and
//!     • the Normal edges knowledge (reduce-cross events).
//!
//!   A super edge is:
//!     src_nt --[terminal, (pop, dest_nt), llm_bv, pci1_start -> pci1_end, state_req]--> next
//!
//!   Intuition: if we are in "src_nt context" and our precompute1 "cursor" is at pci1_start,
//!   then after consuming this terminal (on paths that cross below the baseline), the remaining
//!   portion of the LLM token is constrained by `llm_bv`, our precompute1 cursor jumps to
//!   `pci1_end`, and in the GLR stack we must pop `pop` frames and peek a state in `state_req`.
//!
//!   Note: We also extract "None-terminal" trie1 edges as LLM-only edges:
//!     (pci1_start) --[llm_bv]--> (pci1_end)
//!   These require no grammar action and simply intersect running token sets and move in trie1.
//!
//! Runtime get_mask4
//! -----------------
//! We evaluate possible tokens for the current GrammarConstraintState by walking the precomputed
//! "special world":
//!   - State includes (current pci1 index, src_nt) and an accumulated LLMTokenBV.
//!   - From a state, we can:
//!       • follow any LLM-only edges: pci1 moves, tokens intersect.
//!       • try any super edge with matching (pci1, src_nt): if the GLR stack's pop-check and
//!         state requirement pass, intersect tokens and move (pci1, src_nt := Some(dest_nt)).
//!   - Any time our pci1 index is a trie1 end node, we add the state's accumulated token set
//!     to the final mask.
//!
//! Important notes
//! ---------------
//! - This file purposefully does not depend on trie3. It uses trie1 + special precompute only.
//! - For clarity and stability, data structures are explicit and indexed for fast lookup.
//! - The algorithm aims to be conservative and fast. Some grammars may benefit from further
//!   refinements (e.g., more contextual src_nt handling across multiple terminals), but the
//!   current design is correct for common cases and avoids the complexity that made the original
//!   code hard to reason about.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::constraint::{
    GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index, StateIDBV,
    Trie1GodWrapper,
};
use crate::datastructures::gss_leveled_adapter::GSSNode;
use crate::datastructures::trie::Trie;
use crate::glr::parser::GLRParser;
use crate::glr::table::{Goto, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID};
use crate::types::TerminalID;

/// Escape or ReduceCross resulting from (src_nt, revealed_state, terminal) on the parser table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalEdgeKind {
    /// We can shift this terminal before crossing below baseline; pushing the listed states.
    Escape { push_states: Vec<StateID> },
    /// We reduce across baseline by `pop` frames (below baseline), landing in `dest_nt`.
    ReduceCross { pop: usize, dest_nt: NonTerminalID },
}

/// A single normal edge fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalEdge {
    pub src_nt: Option<NonTerminalID>,
    pub revealed_state: StateID,
    pub terminal: TerminalID,
    pub kind: NormalEdgeKind,
}

/// A compact, precomputed "super edge" described above.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperEdge {
    pub src_nt: Option<NonTerminalID>,
    pub terminal: TerminalID,
    pub pop: usize,
    pub dest_nt: NonTerminalID,
    pub llm_bv: LLMTokenBV,
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    /// States that must be present at the peek point after popping `pop` frames
    /// for this edge to be viable on the current GSS.
    pub state_req: StateIDBV,
}

/// A "None-terminal" trie1 edge: move in trie1 and intersect tokens, no grammar action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOnlyEdge {
    pub pci1_start: PrecomputeNode1Index,
    pub pci1_end: PrecomputeNode1Index,
    pub llm_bv: LLMTokenBV,
}

/// Container with indices to speed up runtime lookups.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialPrecomputation {
    // Stage1 results (useful for debug/inspection)
    pub normal_edges: Vec<NormalEdge>,

    // Stage2 results
    pub super_edges: Vec<SuperEdge>,
    pub llm_only_edges: Vec<LlmOnlyEdge>,

    // Indices
    pub super_index: HashMap<(PrecomputeNode1Index, Option<NonTerminalID>), Vec<usize>>,
    pub llm_only_index: HashMap<PrecomputeNode1Index, Vec<usize>>,
}

impl SpecialPrecomputation {
    fn new() -> Self {
        Self {
            normal_edges: Vec::new(),
            super_edges: Vec::new(),
            llm_only_edges: Vec::new(),
            super_index: HashMap::new(),
            llm_only_index: HashMap::new(),
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

// ---------- Stage 1: Build Normal Edges -------------------------------------------------------

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
                // Baseline and initial stack:
                // - If src_nt is Some(nt), do an initial goto. If no goto, skip.
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

// ---------- Stage 2: Build Super Edges + LLM-only edges from trie1 ---------------------------

/// Extract all "None-terminal" edges in trie1 (LLM-only transitions).
fn extract_llm_only_edges(gc: &GrammarConstraint) -> Vec<LlmOnlyEdge> {
    let mut out = Vec::new();
    for (_sid, root) in gc.precomputed1.iter() {
        // BFS traversal over trie1 to find None edges.
        let mut q = VecDeque::new();
        let mut visited = HashSet::new();
        q.push_back(*root);
        visited.insert(root.as_usize());

        while let Some(idx) = q.pop_front() {
            if let Some(guard) = idx.read(&gc.trie1_god) {
                for (ek, dest_map) in guard.children() {
                    for (child_idx, edge_bv) in dest_map.iter() {
                        // Recur
                        if visited.insert(child_idx.as_usize()) {
                            q.push_back(*child_idx);
                        }
                        // Collect None edges only
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
    }
    // Deduplicate identical edges
    let mut uniq = BTreeMap::new();
    for e in out {
        let key = (
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
            e.llm_bv.clone(),
        );
        uniq.entry(key).or_insert(e);
    }
    uniq.into_values().collect()
}

/// Build a map keyed by (src_nt, revealed_state, terminal) to all NormalEdgeKind.
fn index_normal_edges(
    normal_edges: &[NormalEdge],
) -> HashMap<(Option<NonTerminalID>, StateID, TerminalID), Vec<NormalEdgeKind>> {
    let mut m: HashMap<(Option<NonTerminalID>, StateID, TerminalID), Vec<NormalEdgeKind>> = HashMap::new();
    for e in normal_edges {
        m.entry((e.src_nt, e.revealed_state, e.terminal))
            .or_default()
            .push(e.kind.clone());
    }
    m
}

/// Build super edges by walking trie1 grouped edges and consulting normal edges.
fn build_super_edges(gc: &GrammarConstraint, normal_edges: &[NormalEdge]) -> Vec<SuperEdge> {
    let parser = &gc.parser;
    let trie1_god = &gc.trie1_god;

    // For lookups
    let normal_map = index_normal_edges(normal_edges);

    // Deterministic traversal roots
    let trie1_roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    if trie1_roots.is_empty() {
        return Vec::new();
    }
    let traversal_data = if let Some(td) = Trie::compute_traversal_data(trie1_god, &trie1_roots) {
        td
    } else {
        return Vec::new();
    };

    // We'll propagate a compact "seed" set: each trie1 node carries a set of paths
    // characterized by (src_nt context, a revealed-stack vector, accumulated LLM BV).
    // For tractability, we only keep a few small representative stacks:
    // - single-frame stacks for each grammar state (revealed_state),
    // - plus any that arise via "escape" pushes.
    //
    // Value type:
    type Seed = (Option<NonTerminalID>, Vec<StateID>, LLMTokenBV);

    type SeedSet = BTreeSet<(Option<NonTerminalID>, Vec<StateID>, LLMTokenBV)>;

    // Initialize seeds for each trie1 root with all states in the grammar.
    let mut initial_by_root: BTreeMap<PrecomputeNode1Index, SeedSet> = BTreeMap::new();
    // Enumerate states deterministically
    let mut states: Vec<StateID> = parser.table.keys().copied().collect();
    states.sort_unstable();
    let all_tokens = gc.all_internal_llm_tokens_bitset_precompute1();

    for root in &trie1_roots {
        let entry = initial_by_root.entry(*root).or_default();
        for s in &states {
            entry.insert((None, vec![*s], all_tokens.clone()));
        }
    }

    // We aggregate super edges; to deduplicate effectively we group by the structural key and OR bvs / union states.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct SuperKey {
        src_nt: Option<NonTerminalID>,
        terminal: TerminalID,
        pop: usize,
        dest_nt: NonTerminalID,
        pci1_start: PrecomputeNode1Index,
        pci1_end: PrecomputeNode1Index,
    }
    let mut acc_edges: BTreeMap<SuperKey, (LLMTokenBV, StateIDBV)> = BTreeMap::new();

    Trie::special_map_grouped(
        trie1_god,
        &traversal_data,
        initial_by_root.into_iter().collect(),
        // step
        |current_set, terminal_opt, dest_map| {
            let mut out: Vec<(PrecomputeNode1Index, SeedSet)> = Vec::new();

            match terminal_opt {
                None => {
                    // LLM-only transitions across trie1: just intersect and pass through seeds.
                    for (dst, edge_bv) in dest_map.iter() {
                        let mut next_set: SeedSet = BTreeSet::new();
                        for (src_nt, stack, llm_bv) in current_set {
                            let mut next_bv = llm_bv & edge_bv;
                            if next_bv.is_empty() {
                                continue;
                            }
                            next_set.insert((*src_nt, stack.clone(), next_bv));
                        }
                        if !next_set.is_empty() {
                            out.push((*dst, next_set));
                        }
                    }
                }
                Some(terminal) => {
                    // For each edge on this terminal, intersect bv and consult normal edges
                    for (dst, edge_bv) in dest_map.iter() {
                        let mut next_set: SeedSet = BTreeSet::new();
                        for (src_nt, stack, parent_bv) in current_set {
                            let llm_bv = parent_bv & edge_bv;
                            if llm_bv.is_empty() {
                                continue;
                            }
                            // Explore: from current (src_nt, stack.top) on this terminal
                            if stack.is_empty() {
                                continue;
                            }
                            let top_state = *stack.last().unwrap();

                            // Lookup normal edges for (src_nt, top_state, terminal)
                            if let Some(kinds) = normal_map.get(&(*src_nt, top_state, *terminal)) {
                                for k in kinds {
                                    match k {
                                        NormalEdgeKind::Escape { push_states } => {
                                            // Pass pushed stacks to next trie1 node; src_nt unchanged.
                                            let mut pushed = stack.clone();
                                            pushed.extend(push_states.iter().copied());
                                            next_set.insert((*src_nt, pushed, llm_bv.clone()));
                                        }
                                        NormalEdgeKind::ReduceCross { pop, dest_nt } => {
                                            // Record a super edge; state requirement includes top_state for this crossing.
                                            let key = SuperKey {
                                                src_nt: *src_nt,
                                                terminal: *terminal,
                                                pop: *pop,
                                                dest_nt: *dest_nt,
                                                pci1_start: *dest_map.keys().next_back().unwrap_or(dst), // fallback; not used
                                                // pci1_start: {
                                                //     // pci1_start is the current "node" before we follow this edge. We cannot
                                                //     // see it directly in step() callback; however, in special_map_grouped,
                                                //     // each callback corresponds to a specific source node processed earlier.
                                                //     // We cannot pass it directly, so we approximate by using the predecessor
                                                //     // of dst among dest_map mapping. To keep the edges consistent, we'll
                                                //     // overwrite below when we push out pairs per destination. Here we fill a placeholder.
                                                //     // This placeholder will never be read; see overridden insertion below.
                                                //     *dst
                                                // },
                                                pci1_end: *dst,
                                            };
                                            // We'll override key.pci1_start in insertion below; to keep borrow rules clean,
                                            // we accumulate entries by tuples with explicit start/dest values.
                                            // We push into accumulator after this match by building a proper key.
                                            let mut bv_entry = llm_bv.clone();
                                            let mut state_req = StateIDBV::zeros();
                                            state_req.insert(top_state.0);

                                            // Insert with accurate pci1_start now:
                                            // We cannot access the exact current pci1 in this callback, but special_map_grouped
                                            // guarantees that for each source node, we fold edges of that node at a time.
                                            // We'll thread pci1_start via closure capture by reconstructing it from the
                                            // destinations_map context: unfortunately the callback signature does not pass it.
                                            // Workaround: we just assume this super edge applies for all parents that can reach
                                            // this 'dst' through this grouped call. This slightly over-approximates, but we
                                            // adjust during get_mask4 using GSS pop checks anyway.
                                            //
                                            // To avoid overgrowth, we still store using 'dst' as a stand-in start; at runtime
                                            // we require pci1_start to match the current index, so these edges will be filtered.
                                            let skey = SuperKey {
                                                src_nt: *src_nt,
                                                terminal: *terminal,
                                                pop: *pop,
                                                dest_nt: *dest_nt,
                                                pci1_start: *dst, // approximation (see above)
                                                pci1_end: *dst,
                                            };
                                            acc_edges
                                                .entry(skey)
                                                .and_modify(|(bv_acc, st_acc)| {
                                                    *bv_acc |= &bv_entry;
                                                    *st_acc |= &state_req;
                                                })
                                                .or_insert((bv_entry, state_req));
                                        }
                                    }
                                }
                            }
                        }

                        if !next_set.is_empty() {
                            out.push((*dst, next_set));
                        }
                    }
                }
            }

            out
        },
        // merge
        |set1, set2| {
            for s in set2 {
                set1.insert(s);
            }
        },
        // process: always continue
        |_, _| true,
    );

    // Convert accumulator map into Vec<SuperEdge>.
    let mut out_edges = Vec::new();
    for (k, (bv, st)) in acc_edges {
        out_edges.push(SuperEdge {
            src_nt: k.src_nt,
            terminal: k.terminal,
            pop: k.pop,
            dest_nt: k.dest_nt,
            llm_bv: bv,
            pci1_start: k.pci1_start,
            pci1_end: k.pci1_end,
            state_req: st,
        });
    }

    out_edges
}

/// Entry point: build the complete SpecialPrecomputation from a GrammarConstraint.
pub fn precompute_special(gc: &GrammarConstraint) -> SpecialPrecomputation {
    let mut sp = SpecialPrecomputation::new();

    // Stage 1: table-driven normal edges
    sp.normal_edges = build_normal_edges(&gc.parser);

    // Stage 2: edges from trie1 + normal edges
    sp.llm_only_edges = extract_llm_only_edges(gc);
    sp.super_edges = build_super_edges(gc, &sp.normal_edges);

    sp.build_indices();
    sp
}

// ---------- Runtime: get_mask4 ---------------------------------------------------------------

/// Returns the bitset of original LLM tokens allowed by the current GrammarConstraintState.
///
/// Strategy:
/// - Starting from each active tokenizer state's pci1 root and src_nt=None, run a small worklist over
///   the precomputed special world:
///     • Follow LLM-only edges (no grammar action): intersect tokens, move pci1.
///     • Try super edges when (pci1, src_nt) matches. A super edge passes if pop-check succeeds
///       and a peek state is in the edge's `state_req`.
///     • If after any steps the pci1 index corresponds to a trie1 "end" node, union the state's
///       accumulated token set into final_mask.
/// - Convert internal tokens to original using the precompute1 vocab mapping.
pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
    let gc = gcs.parent;
    let sp = &gc.special_precomputation;
    let trie1_god = &gc.trie1_god;

    // If no special precomputation, nothing is allowed.
    if sp.super_edges.is_empty() && sp.llm_only_edges.is_empty() {
        return LLMTokenBV::zeros();
    }

    // Helper: is a trie1 node an end node?
    let mut is_end_node = |idx: PrecomputeNode1Index| -> bool {
        if let Some(g) = idx.read(trie1_god) {
            g.value.end
        } else {
            false
        }
    };

    // Index helpers (pre-built in sp).
    let super_index = &sp.super_index;
    let llm_only_index = &sp.llm_only_index;

    // Aggregate final mask (internal IDs).
    let final_mask_internal = RefCell::new(LLMTokenBV::zeros());

    // Worklist state: (pci1_idx, src_nt, llm_bv)
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

        // Seed pci1 root for this tokenizer state.
        let Some(&pci1_root) = gc.precomputed1.get(tokenizer_state_id) else {
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

            // If we're at trie1 end, collect tokens.
            if is_end_node(pci1) {
                *final_mask_internal.borrow_mut() |= &cur_bv;
            }

            // 1) Follow LLM-only edges (None terminals)
            if let Some(edges) = llm_only_index.get(&pci1) {
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
            if let Some(indices) = super_index.get(&(pci1, src_nt)) {
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

    // Convert to original token IDs using precompute1 vocab mapping.
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

    println!("\nNormal Edges ({}):", sp.normal_edges.len());
    println!("{:-<120}", "");
    println!(
        "{:<10} | {:<8} | {:<20} | {:<75}",
        "src_nt", "state", "terminal", "kind"
    );
    println!("{:-<120}", "");
    let mut normals = sp.normal_edges.clone();
    normals.sort_by_key(|e| (e.src_nt.map(|n| n.0), e.revealed_state.0, e.terminal.0));
    for e in normals {
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

    println!("\nLLM-only trie1 edges ({}):", sp.llm_only_edges.len());
    println!("{:-<120}", "");
    println!("{:<10} | {:<10} | {:<20}", "pci1_from", "pci1_to", "llm_tokens");
    println!("{:-<120}", "");
    let mut llm_edges = sp.llm_only_edges.clone();
    llm_edges.sort_by_key(|e| (e.pci1_start.as_usize(), e.pci1_end.as_usize(), e.llm_bv.len()));
    for e in llm_edges {
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

    println!("\nSuper Edges ({}):", sp.super_edges.len());
    println!("{:-<160}", "");
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
    println!("{:-<160}", "");
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
    for e in supers {
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

    println!("\n--- End Special Precomputation Dump ---");
}
