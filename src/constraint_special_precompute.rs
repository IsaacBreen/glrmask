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

// -----------------------------------------------------------------------------------------------
// Overview (new spec, superseding prior design)
// -----------------------------------------------------------------------------------------------
//
// One node per nonterminal Some(nt), plus one start node None.
// Three edge types:
//
// Reduce edges (grammar-local):
//   (Option<NonTerminalID>, TerminalID, StateID, usize, NonTerminalID)
//   From this (optional) nt, using this terminal, when the current top-of-stack state is `state`,
//   if we perform a reduce chain that crosses below the current baseline, we pop `usize` frames
//   below the baseline and land into nonterminal `NonTerminalID`.
//
// Shift edges (grammar-local):
//   (Option<NonTerminalID>, TerminalID, StateID, Vec<StateID>)
//   From this (optional) nt, using this terminal, when the current top-of-stack state is `state`,
//   we can immediately shift, and the vector lists the GLR states to push above the baseline.
//   The vector length is either 1 (starting from None baseline) or 2 (starting from Some(nt)
//   baseline + immediate shift).
//
// Super edges (precompute1-aware):
//   (Option<NonTerminalID>, TerminalID, PrecomputeNodeIndex1, usize, NonTerminalID, LLMTokenBV, PrecomputeNodeIndex1)
//   Edge from this (optional) ntid, labeled by a terminal and guarded by the starting precompute1
//   node, that pops `usize` frames below the baseline, reaches `NonTerminalID` as the new grammar
//   node, filters tokens by LLM bitset, and moves to the destination precompute1 node.
//
// Stage 1 computes Reduce and Shift edges using a direct queue-of-stacks exploration per
// (src_nt, terminal, revealed_state). No cleverness: copy stacks on split, process until either
// we encounter a shift (record a Shift edge) or cross below the baseline (record a Reduce edge).
//
// Stage 2 runs a traversal over precompute1 (Trie::special_map_grouped). It carries as values
// a bag of (Vec<StateID>, LLMTokenBV, PrecomputeNode1Index) and, per terminal edge in the trie,
// it:
//   - intersects LLM masks,
//   - explores only Reduce edges (pop from the vector), queuing more reduce steps if frames remain,
//   - when the vector becomes empty, it emits a Super edge,
//   - explores Shift edges by extending vectors, adding those to the output value bag.
// This produces all Super edges. We do not keep LLM-only edges anymore.
//
// Runtime (get_mask4) does not touch trie1. It processes only Super edges and Reduce edges:
//   - we track (pci1, src_nt, llm_bv) per tokenizer/GLR-state,
//   - for each candidate Super edge from (pci1, src_nt), we check feasibility against the
//     concrete GSS by popping `pop` frames and examining peek states; feasibility is true iff
//     some peek state s satisfies a Stage-1 Reduce-edge (src_nt, terminal, s, pop, dest_nt).
//   - upon feasibility, we move to (pci1_end, Some(dest_nt)) and intersect llm_bv.
//   - if pci1 is an end node, we accumulate llm_bv into the final mask.
// No Shift edges are traversed at runtime.
//
// Additionally, a simple GraphViz dump is provided for diagnostics of the Super-edge graph.
//
// -----------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceEdge {
    pub src_nt: Option<NonTerminalID>,
    pub terminal: TerminalID,
    pub state: StateID,
    pub pop: usize,
    pub dest_nt: NonTerminalID,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShiftEdge {
    pub src_nt: Option<NonTerminalID>,
    pub terminal: TerminalID,
    pub state: StateID,
    pub push: Vec<StateID>, // states to push above the baseline bottom
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperEdge {
    pub src_nt: Option<NonTerminalID>,
    pub terminal: TerminalID,
    pub pci1_start: PrecomputeNode1Index,
    pub pop: usize,                    // amount to pop below baseline at runtime
    pub dest_nt: NonTerminalID,        // new "node" in the special graph
    pub llm_bv: LLMTokenBV,            // token filter
    pub pci1_end: PrecomputeNode1Index,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialPrecomputation {
    // Stage-1, grammar-only edges:
    pub reduce_edges: Vec<ReduceEdge>,
    pub shift_edges: Vec<ShiftEdge>,

    // Stage-2, trie-aware summary edges:
    pub super_edges: Vec<SuperEdge>,

    // Indices for runtime:
    //   - Super edges available from (pci1_start, src_nt)
    pub super_index: HashMap<(PrecomputeNode1Index, Option<NonTerminalID>), Vec<usize>>,
    //   - Reduce-edge feasibility map: (src_nt, terminal, pop, dest_nt) -> allowed peek states
    pub reduce_feasibility: HashMap<(Option<NonTerminalID>, TerminalID, usize, NonTerminalID), StateIDBV>,

    // Trie1-derived facts for runtime:
    pub pci1_roots_by_tokenizer: BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    pub pci1_end_nodes: BTreeSet<PrecomputeNode1Index>,
}

impl SpecialPrecomputation {
    fn new() -> Self {
        Self {
            reduce_edges: Vec::new(),
            shift_edges: Vec::new(),
            super_edges: Vec::new(),
            super_index: HashMap::new(),
            reduce_feasibility: HashMap::new(),
            pci1_roots_by_tokenizer: BTreeMap::new(),
            pci1_end_nodes: BTreeSet::new(),
        }
    }

    fn build_indices(&mut self) {
        // Super edges
        self.super_index.clear();
        for (i, e) in self.super_edges.iter().enumerate() {
            self.super_index
                .entry((e.pci1_start, e.src_nt))
                .or_default()
                .push(i);
        }

        // Reduce feasibility: group reduce edges by (src_nt, terminal, pop, dest_nt),
        // and collect the top-of-stack states that permit them.
        self.reduce_feasibility.clear();
        for r in &self.reduce_edges {
            let key = (r.src_nt, r.terminal, r.pop, r.dest_nt);
            self.reduce_feasibility
                .entry(key)
                .or_insert_with(StateIDBV::zeros)
                .insert(r.state.0);
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

fn get_goto(parser: &GLRParser, state: StateID, nt: NonTerminalID) -> Option<StateID> {
    parser
        .table
        .get(&state)
        .and_then(|row| row.gotos.get(&nt))
        .and_then(|g| g.state_id)
}

// ---------- Stage 1: Build grammar-local Reduce and Shift edges -------------------------------

fn build_stage1_edges(parser: &GLRParser) -> (Vec<ReduceEdge>, Vec<ShiftEdge>) {
    let mut reduce_edges = Vec::new();
    let mut shift_edges = Vec::new();

    let mut all_states: Vec<StateID> = parser.table.keys().copied().collect();
    all_states.sort_unstable();

    let all_terminals: Vec<TerminalID> = parser.terminal_map.right_values().copied().collect();

    let mut all_src_nts: Vec<Option<NonTerminalID>> = parser
        .non_terminal_map
        .right_values()
        .copied()
        .map(Some)
        .collect();
    all_src_nts.push(None);

    // For each (src_nt, terminal, revealed_state), explore:
    // - Initialize baseline stack (either [revealed] or [revealed, goto(revealed, nt)]).
    // - Continue parsing with only this terminal:
    //     * On Reduce above baseline: perform goto and continue.
    //     * On crossing below baseline: record Reduce edge.
    //     * On Shift: record Shift edge (vector is everything above the baseline bottom after shift).
    for src_nt in all_src_nts.iter().copied() {
        for &revealed in &all_states {
            // Build initial baseline stacks
            let mut initial_stacks: Vec<Vec<StateID>> = Vec::new();
            let baseline_len: usize;

            if let Some(nt) = src_nt {
                if let Some(g) = get_goto(parser, revealed, nt) {
                    initial_stacks.push(vec![revealed, g]);
                    baseline_len = 2;
                } else {
                    continue;
                }
            } else {
                initial_stacks.push(vec![revealed]);
                baseline_len = 1;
            }

            for init in initial_stacks {
                for &term in &all_terminals {
                    let mut q: VecDeque<Vec<StateID>> = VecDeque::new();
                    let mut seen: HashSet<Vec<StateID>> = HashSet::new();
                    q.push_back(init.clone());
                    seen.insert(init.clone());

                    let mut recorded_shift = false;

                    while let Some(stack) = q.pop_front() {
                        if stack.is_empty() {
                            continue;
                        }
                        let top = *stack.last().unwrap();

                        for action in actions_for(parser, top, term) {
                            match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(shift_to) => {
                                    // Immediate shift => record shift edge and stop exploring this (src_nt, term, revealed)
                                    // The vector to push excludes the baseline bottom frame (revealed).
                                    let mut after = stack.clone();
                                    after.push(*shift_to);
                                    let push_vec = after[1..].to_vec(); // items above the bottom
                                    shift_edges.push(ShiftEdge {
                                        src_nt,
                                        terminal: term,
                                        state: revealed, // condition checked at runtime/2nd stage
                                        push: push_vec,
                                    });
                                    recorded_shift = true;
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                                    let len = *len;
                                    let reduce_nt = *nonterminal_id;

                                    let above = stack.len().saturating_sub(baseline_len);
                                    if len > above {
                                        // Crossing below the baseline => record reduce edge
                                        let pop_below = len - above;
                                        reduce_edges.push(ReduceEdge {
                                            src_nt,
                                            terminal: term,
                                            state: top, // record the current top-of-stack state
                                            pop: pop_below,
                                            dest_nt: reduce_nt,
                                        });
                                    } else {
                                        // Reduce above baseline: pop len and goto(reduce_nt)
                                        let mut after_pop = stack.clone();
                                        let new_len = after_pop.len().saturating_sub(len);
                                        after_pop.truncate(new_len);
                                        if let Some(&new_top) = after_pop.last() {
                                            if let Some(g) = get_goto(parser, new_top, reduce_nt) {
                                                let mut after_goto = after_pop;
                                                after_goto.push(g);
                                                if seen.insert(after_goto.clone()) {
                                                    q.push_back(after_goto);
                                                }
                                            }
                                        }
                                    }
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                    if let Some(shift_to) = shift {
                                        let mut after = stack.clone();
                                        after.push(*shift_to);
                                        let push_vec = after[1..].to_vec();
                                        shift_edges.push(ShiftEdge {
                                            src_nt,
                                            terminal: term,
                                            state: revealed,
                                            push: push_vec,
                                        });
                                        recorded_shift = true;
                                    }
                                    for (len, nts) in reduces {
                                        for (reduce_nt, _) in nts {
                                            let len = *len;
                                            let reduce_nt = *reduce_nt;
                                            let above = stack.len().saturating_sub(baseline_len);
                                            if len > above {
                                                let pop_below = len - above;
                                                reduce_edges.push(ReduceEdge {
                                                    src_nt,
                                                    terminal: term,
                                                    state: top,
                                                    pop: pop_below,
                                                    dest_nt: reduce_nt,
                                                });
                                            } else {
                                                let mut after_pop = stack.clone();
                                                let new_len = after_pop.len().saturating_sub(len);
                                                after_pop.truncate(new_len);
                                                if let Some(&new_top) = after_pop.last() {
                                                    if let Some(g) = get_goto(parser, new_top, reduce_nt) {
                                                        let mut after_goto = after_pop;
                                                        after_goto.push(g);
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
                        // If we've recorded a shift for this (src_nt, term, revealed), do not explore further.
                        if recorded_shift {
                            break;
                        }
                    }
                }
            }
        }
    }

    (reduce_edges, shift_edges)
}

// ---------- Stage 2: Build Super edges using trie1 special_map_grouped ------------------------

#[derive(Clone)]
struct ValueItem {
    stack: Vec<StateID>,               // above-baseline frames
    llm_bv: LLMTokenBV,                // current token mask
    pci1_start: PrecomputeNode1Index,  // where this value started (for Super-edge pci1_start)
}

type ValueBag = Vec<ValueItem>;

fn collect_trie1_roots_and_end_nodes(gc: &GrammarConstraint) -> (Vec<PrecomputeNode1Index>, BTreeSet<PrecomputeNode1Index>) {
    let roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    let mut end_nodes = BTreeSet::new();

    if roots.is_empty() {
        return (roots, end_nodes);
    }
    let nodes = Trie::all_nodes(&gc.trie1_god, &roots);
    for n in nodes {
        if let Some(g) = n.read(&gc.trie1_god) {
            if g.value.end {
                end_nodes.insert(n);
            }
        }
    }
    (roots, end_nodes)
}

fn build_super_edges_via_trie(
    gc: &GrammarConstraint,
    reduce_edges: &[ReduceEdge],
    shift_edges: &[ShiftEdge],
) -> (Vec<SuperEdge>, BTreeSet<PrecomputeNode1Index>) {
    // Index reduce edges by (src_nt, terminal, state) -> Vec<(pop, dest_nt)>
    let mut reduce_by_key: HashMap<(Option<NonTerminalID>, TerminalID, StateID), Vec<(usize, NonTerminalID)>> =
        HashMap::new();
    for r in reduce_edges {
        reduce_by_key
            .entry((r.src_nt, r.terminal, r.state))
            .or_default()
            .push((r.pop, r.dest_nt));
    }

    // Index shift edges by (src_nt, terminal, state) -> Vec<Vec<StateID>> (push vectors)
    let mut shift_by_key: HashMap<(Option<NonTerminalID>, TerminalID, StateID), Vec<Vec<StateID>>> =
        HashMap::new();
    for s in shift_edges {
        shift_by_key
            .entry((s.src_nt, s.terminal, s.state))
            .or_default()
            .push(s.push.clone());
    }

    // Precompute roots and end nodes
    let (roots, end_nodes) = collect_trie1_roots_and_end_nodes(gc);
    if roots.is_empty() {
        return (Vec::new(), end_nodes);
    }

    // Initial values: at each root, every state as a singleton vector with all tokens.
    let all_states: Vec<StateID> = {
        let mut v: Vec<StateID> = gc.parser.table.keys().copied().collect();
        v.sort_unstable();
        v
    };
    let seed_bv = gc.all_internal_llm_tokens_bitset_precompute1();

    let mut initial_nodes_and_values: Vec<(PrecomputeNode1Index, ValueBag)> = Vec::new();
    for r in &roots {
        let mut bag = Vec::with_capacity(all_states.len());
        for &s in &all_states {
            bag.push(ValueItem {
                stack: vec![s],
                llm_bv: seed_bv.clone(),
                pci1_start: *r,
            });
        }
        initial_nodes_and_values.push((*r, bag));
    }

    let traversal = Trie::compute_traversal_data(&gc.trie1_god, &roots);
    if traversal.is_none() {
        return (Vec::new(), end_nodes);
    }
    let traversal = traversal.unwrap();

    let super_edges_out: RefCell<Vec<SuperEdge>> = RefCell::new(Vec::new());

    // Dedup super-edges to avoid combinatorial blow-up when values coalesce.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct SuperKey {
        src_nt: Option<NonTerminalID>,
        term: TerminalID,
        pci1_start: usize,
        pop: usize,
        dest_nt: NonTerminalID,
        pci1_end: usize,
        // We do NOT include llm_bv in the key; different llm_bv will be accumulated as edges may repeat.
    }
    let super_seen: RefCell<HashSet<SuperKey>> = RefCell::new(HashSet::new());

    Trie::special_map_grouped::<ValueBag, _, _>(
        &gc.trie1_god,
        &traversal,
        initial_nodes_and_values,
        // step
        |values, ek, dsts| {
            let mut out: Vec<(PrecomputeNode1Index, ValueBag)> = Vec::new();

            let term = match ek {
                Some(t) => *t,
                None => {
                    // New spec: no "LLM-only" edges; ignore None-edges if any exist accidentally.
                    return out;
                }
            };

            for (v, llm_bv_edge) in dsts.iter() {
                // Intersect in-place each value's LLM mask with the edge's mask.
                let mut next_bag: ValueBag = Vec::new();
                next_bag.reserve(values.len());

                for item in values.iter() {
                    let next_bv = &item.llm_bv & llm_bv_edge;
                    if next_bv.is_empty() {
                        continue;
                    }

                    // Explore grammar reduce/shift dynamics starting at start node None.
                    // Queue entries: (src_nt, stack_vec, llm_bv)
                    let mut q: VecDeque<(Option<NonTerminalID>, Vec<StateID>, LLMTokenBV)> = VecDeque::new();
                    q.push_back((None, item.stack.clone(), next_bv.clone()));

                    while let Some((src_nt, stack, cur_bv)) = q.pop_front() {
                        if stack.is_empty() {
                            continue;
                        }

                        // 1) REDUCE steps (pop from stack using matching reduce edges)
                        if let Some((&top, rest)) = stack.split_last() {
                            if let Some(reduces) = reduce_by_key.get(&(src_nt, term, top)) {
                                for &(pop, dest_nt) in reduces {
                                    let len = stack.len();
                                    let trimmed_len = len.saturating_sub(pop);
                                    let mut remainder_stack = stack.clone();
                                    remainder_stack.truncate(trimmed_len);

                                    // SPECIAL CASE: If destination precompute1 node is an end node, clear the stack.
                                    let is_end = end_nodes.contains(v);
                                    if is_end {
                                        remainder_stack.clear();
                                    }

                                    if remainder_stack.is_empty() {
                                        // Pop crossed (or reached) the baseline: emit a Super edge
                                        // The pop carried with the Super edge is the "remainder below baseline".
                                        let remainder_below = pop.saturating_sub(len);
                                        let key = SuperKey {
                                            src_nt,
                                            term,
                                            pci1_start: item.pci1_start.as_usize(),
                                            pop: remainder_below,
                                            dest_nt,
                                            pci1_end: v.as_usize(),
                                        };
                                        let mut seen = super_seen.borrow_mut();
                                        if !seen.contains(&key) {
                                            super_edges_out.borrow_mut().push(SuperEdge {
                                                src_nt,
                                                terminal: term,
                                                pci1_start: item.pci1_start,
                                                pop: remainder_below,
                                                dest_nt,
                                                llm_bv: cur_bv.clone(),
                                                pci1_end: *v,
                                            });
                                            seen.insert(key);
                                        } else {
                                            // If a duplicate key is seen with a different LLM BV, merge by OR-ing tokens
                                            if let Some(last) = super_edges_out.borrow_mut().last_mut() {
                                                if last.src_nt == src_nt
                                                    && last.terminal == term
                                                    && last.pci1_start == item.pci1_start
                                                    && last.pop == remainder_below
                                                    && last.dest_nt == dest_nt
                                                    && last.pci1_end == *v
                                                {
                                                    last.llm_bv |= &cur_bv;
                                                }
                                            }
                                        }
                                    } else {
                                        // Keep reducing with dest_nt as the new src_nt
                                        q.push_back((Some(dest_nt), remainder_stack, cur_bv.clone()));
                                    }
                                }
                            }
                        }

                        // 2) SHIFT steps: extend the vector and keep pci1 at the start node.
                        if let Some((&top, _)) = stack.split_last() {
                            if let Some(shifts) = shift_by_key.get(&(src_nt, term, top)) {
                                for push_vec in shifts {
                                    let mut extended = stack.clone();
                                    extended.extend(push_vec.iter().copied());
                                    next_bag.push(ValueItem {
                                        stack: extended,
                                        llm_bv: cur_bv.clone(),
                                        pci1_start: item.pci1_start,
                                    });
                                }
                            }
                        }
                    }
                }

                out.push((*v, next_bag));
            }

            out
        },
        // merge
        |acc, mut other| {
            // Just append; duplicates are harmless and culled naturally in subsequent steps.
            acc.append(&mut other);
        },
        // process (unused)
        |_node, _idx, _v| false,
    );

    (super_edges_out.into_inner(), end_nodes)
}

// ---------- Public builder: precompute_special -------------------------------------------------

pub fn precompute_special(gc: &GrammarConstraint) -> SpecialPrecomputation {
    let (reduce_edges, shift_edges) = build_stage1_edges(&gc.parser);
    let (super_edges, end_nodes) = build_super_edges_via_trie(gc, &reduce_edges, &shift_edges);

    let mut sp = SpecialPrecomputation::new();
    sp.reduce_edges = reduce_edges;
    sp.shift_edges = shift_edges;
    sp.super_edges = super_edges;

    // Trie1 roots per tokenizer state; used to seed get_mask4
    sp.pci1_roots_by_tokenizer = gc.precomputed1.clone();
    sp.pci1_end_nodes = end_nodes;

    // Build runtime indices
    sp.build_indices();

    sp
}

// ---------- Runtime: get_mask4 (trie1-free, reduce+super only) --------------------------------

pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
    let gc = gcs.parent;
    let sp = &gc.special_precomputation;

    if sp.super_edges.is_empty() {
        return LLMTokenBV::zeros();
    }

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

        // Seed queue with (pci1_root, None, all tokens at stage-1)
        let seed_bv = gc.all_internal_llm_tokens_bitset_precompute1();
        let mut q: VecDeque<(PrecomputeNode1Index, Option<NonTerminalID>, LLMTokenBV)> = VecDeque::new();
        let mut seen: HashMap<Key, LLMTokenBV> = HashMap::new();

        q.push_back((pci1_root, None, seed_bv.clone()));
        seen.insert(Key { pci1: pci1_root, src_nt: None }, seed_bv);

        while let Some((pci1, src_nt, cur_bv)) = q.pop_front() {
            if cur_bv.is_empty() {
                continue;
            }

            // If at an end node, accumulate tokens
            if sp.pci1_end_nodes.contains(&pci1) {
                *final_mask_internal.borrow_mut() |= &cur_bv;
            }

            // Try all Super edges from (pci1, src_nt)
            if let Some(indices) = sp.super_index.get(&(pci1, src_nt)) {
                'super_loop: for &i in indices {
                    let e = &sp.super_edges[i];

                    // Intersect token mask
                    let next_bv = &cur_bv & &e.llm_bv;
                    if next_bv.is_empty() {
                        continue;
                    }

                    // Check feasibility against the actual GSS using Reduce-edge feasibility map.
                    // We need (src_nt, e.terminal, e.pop, e.dest_nt) and confirm that some peek
                    // state is in the allowed set.
                    let feas_key = (src_nt, e.terminal, e.pop, e.dest_nt);
                    let Some(state_req) = sp.reduce_feasibility.get(&feas_key) else {
                        continue 'super_loop;
                    };

                    // Pop-and-peek across the active GSS
                    let popped = glr_state.active_state.stack.popn(e.pop);
                    let mut ok = false;
                    'outer: for item in popped.iter() {
                        for peek in item.peek_iter() {
                            if state_req.contains(peek.edge_value().state_id.0) {
                                ok = true;
                                break 'outer;
                            }
                        }
                    }
                    if !ok {
                        continue 'super_loop;
                    }

                    // Move to next (pci1, src_nt) with filtered tokens
                    let new_src_nt = Some(e.dest_nt);
                    let key = Key { pci1: e.pci1_end, src_nt: new_src_nt };
                    let entry = seen.entry(key).or_insert_with(LLMTokenBV::zeros);
                    let delta = &next_bv - entry;
                    if !delta.is_empty() {
                        *entry |= &next_bv;
                        q.push_back((e.pci1_end, new_src_nt, next_bv.clone()));
                    }
                }
            }
        }
    }

    // Map back to original LLM token IDs using stage-1 -> original mapping
    gc.internal_bv_to_original_precompute1(&final_mask_internal.into_inner())
}

// ---------- GraphViz visualization -------------------------------------------------------------

pub fn graphviz_special(gc: &GrammarConstraint) -> String {
    let sp = &gc.special_precomputation;
    let parser = &gc.parser;

    let nt_name = |oid: &Option<NonTerminalID>| -> String {
        match oid {
            Some(nt) => parser
                .non_terminal_map
                .get_by_right(nt)
                .map(|t| t.to_string())
                .unwrap_or_else(|| format!("NT({})", nt.0)),
            None => "Start".to_string(),
        }
    };
    let term_name = |tid: &TerminalID| -> String {
        parser
            .terminal_map
            .get_by_right(tid)
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("T({})", tid.0))
    };

    let mut out = String::new();
    out.push_str("digraph SpecialPrecompute {\n");
    out.push_str("  rankdir=LR;\n");
    out.push_str("  node [shape=circle, fontsize=10];\n");

    // Create per-(pci1) clusters with nodes per (src_nt)
    // Gather all pci1 that are referenced in super edges
    let mut by_pci1: BTreeMap<PrecomputeNode1Index, Vec<&SuperEdge>> = BTreeMap::new();
    for e in &sp.super_edges {
        by_pci1.entry(e.pci1_start).or_default().push(e);
    }

    for (pci1, edges) in by_pci1.iter() {
        out.push_str(&format!("  subgraph cluster_{} {{\n", pci1.as_usize()));
        out.push_str(&format!("    label = \"pci1:{}{}\";\n",
            pci1.as_usize(),
            if sp.pci1_end_nodes.contains(pci1) { " (end)" } else { "" }
        ));
        out.push_str("    style=rounded;\n");

        // Nodes in this cluster: all src_nt that appear as sources
        let mut nts = BTreeSet::<Option<NonTerminalID>>::new();
        for e in edges {
            nts.insert(e.src_nt);
        }
        for nt in nts {
            let node_id = format!("n_{}_{}", pci1.as_usize(), nt.map(|n| n.0.to_string()).unwrap_or_else(|| "none".to_string()));
            let shape = if sp.pci1_end_nodes.contains(pci1) { "doublecircle" } else { "circle" };
            out.push_str(&format!("    {} [label=\"{}\", shape={}];\n", node_id, nt_name(&nt), shape));
        }
        out.push_str("  }\n");
    }

    // Edges (across clusters)
    // From (pci1_start, src_nt) -> (pci1_end, Some(dest_nt))
    for e in &sp.super_edges {
        let from = format!("n_{}_{}", e.pci1_start.as_usize(), e.src_nt.map(|n| n.0.to_string()).unwrap_or_else(|| "none".to_string()));
        let to = format!("n_{}_{}", e.pci1_end.as_usize(), e.dest_nt.0);
        let label = format!("{} / pop={} / tokens={}", term_name(&e.terminal), e.pop, if e.llm_bv.is_all() { "ALL".to_string() } else { e.llm_bv.len().to_string() });
        out.push_str(&format!("  {} -> {} [label=\"{}\", fontsize=9];\n", from, to, label));
    }

    out.push_str("}\n");
    out
}

// ---------- Debug dump (concise) ---------------------------------------------------------------

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
            None => "Start".to_string(),
        }
    };
    let term_name = |tid: &TerminalID| -> String {
        parser
            .terminal_map
            .get_by_right(tid)
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("T({})", tid.0))
    };

    println!("--- Special Precomputation (new spec) ---");
    println!("Tokenizer roots cached: {}", sp.pci1_roots_by_tokenizer.len());
    println!("pci1 end nodes: {}", sp.pci1_end_nodes.len());
    println!("Reduce edges: {}", sp.reduce_edges.len());
    println!("Shift edges: {}", sp.shift_edges.len());
    println!("Super edges: {}", sp.super_edges.len());

    // Print a small sample of super edges
    let mut supers = sp.super_edges.clone();
    supers.sort_by_key(|e| (
        e.pci1_start.as_usize(),
        e.src_nt.map(|n| n.0),
        e.terminal.0,
        e.pop,
        e.dest_nt.0,
        e.pci1_end.as_usize(),
    ));

    for e in supers.iter().take(1000) {
        let cnt = if e.llm_bv.is_all() { "ALL".to_string() } else { e.llm_bv.len().to_string() };
        println!(
            "  [{}] -- {} / pop={} / {} --> [{}]  pci1 {} -> {}  tokens={}",
            nt_name(&e.src_nt),
            term_name(&e.terminal),
            e.pop,
            parser.non_terminal_map.get_by_right(&e.dest_nt).map(|s| s.to_string()).unwrap_or_else(|| format!("{}", e.dest_nt.0)),
            nt_name(&Some(e.dest_nt)),
            e.pci1_start.as_usize(),
            e.pci1_end.as_usize(),
            cnt
        );
    }
    if sp.super_edges.len() > 1000 {
        println!("  ... ({} more)", sp.super_edges.len() - 1000);
    }

    println!("--- End Special Precomputation ---");
}

// ------------------------------------------------------------------------------------------------
// Notes:
// - No LLM-only edges are generated or used (per updated spec).
// - get_mask4 explores only Super edges + Reduce-edge feasibility. Shift edges are not traversed
//   at runtime; they are only used during Stage 2 to build Super edges.
// - The precompute trie is not accessed at runtime.
// ------------------------------------------------------------------------------------------------
