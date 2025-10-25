use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use crate::datastructures::trie::Trie;
use std::sync::Arc;

use crate::constraint::{
    GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index, Trie1GodWrapper,
};
use crate::glr::parser::{GLRParser, GLRParserState, ParseStateEdgeContent};
use crate::glr::table::{Goto, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID};
use crate::datastructures::gss_leveled_adapter::{allow_only_llm_tokens_and_prune_arc, GSSNode};
use crate::types::TerminalID;

// Types for special precomputation
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpecialPrecomputeDest {
    Reduce { pop: usize, dest_nt: NonTerminalID },
    Escape { push_states: Vec<StateID> },
}

// (Option<NonTerminalID>, StateID, TerminalID, SpecialPrecomputeDest)
pub type SpecialPrecomputeNormalEdge = (
    Option<NonTerminalID>,
    StateID,
    TerminalID,
    SpecialPrecomputeDest,
);

// (Option<NonTerminalID>, TerminalID, (usize, NonTerminalID), LLMTokenBV, PrecomputeNode1Index, PrecomputeNode1Index)
pub type SpecialPrecomputeSuperEdge = (
    Option<NonTerminalID>,
    TerminalID,
    (usize, NonTerminalID),
    LLMTokenBV,
    PrecomputeNode1Index,
    PrecomputeNode1Index,
);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialPrecomputation {
    pub normal_edges: HashSet<SpecialPrecomputeNormalEdge>,
    pub super_edges: HashSet<SpecialPrecomputeSuperEdge>,
}

// Helper to get actions for a state and terminal
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

// Helper to get gotos for a state and non-terminal
fn get_gotos<'a>(parser: &'a GLRParser, state_id: StateID, nt_id: NonTerminalID) -> Vec<&'a Goto> {
    if let Some(row) = parser.table.get(&state_id) {
        row.gotos.get(&nt_id).map(|g| vec![g]).unwrap_or_default()
    } else {
        vec![]
    }
}

pub fn precompute_special(gc: &GrammarConstraint) -> SpecialPrecomputation {
    let mut normal_edges = HashSet::new();
    let super_edges = RefCell::new(HashSet::new());
    let parser = &gc.parser;

    let mut non_terminals: Vec<Option<NonTerminalID>> = parser
        .non_terminal_map
        .right_values()
        .copied()
        .map(Some)
        .collect();
    non_terminals.push(None);

    let terminals: Vec<TerminalID> = parser.terminal_map.right_values().copied().collect();

    let mut states: Vec<StateID> = parser.table.keys().copied().collect();
    states.sort();
    states.dedup();

    // Stage 1 (rewritten): compute normal_edges via baseline-aware BFS.
    //
    // For each (src_nt, revealed_state, terminal):
    // - Build an initial stack and define its baseline length:
    //   * src_nt = None       -> initial stack = [revealed_state], baseline_len = 1
    //   * src_nt = Some(nt)   -> initial stack = [revealed_state, goto(revealed_state, nt)] if such goto exists
    // - BFS over stacks for that terminal. On:
    //   * Shift: record Escape edge with push_states = new_stack[baseline_len..]
    //   * Reduce: if it crosses the baseline, record Reduce edge with pop = amount below baseline;
    //             otherwise perform goto and continue BFS.
    //   * Split: do both (Shift and Reduce).
    for src_nt in &non_terminals {
        for &revealed_state in &states {
            for &terminal in &terminals {
                // Build initial stack with baseline
                let mut initial_stack_opt: Option<Vec<StateID>> = None;
                match src_nt {
                    Some(nt) => {
                        // Internal node: must first goto on the provided nonterminal
                        for goto in get_gotos(parser, revealed_state, *nt) {
                            if let Some(goto_state) = goto.state_id {
                                initial_stack_opt = Some(vec![revealed_state, goto_state]);
                            }
                        }
                    }
                    None => {
                        // Start node
                        initial_stack_opt = Some(vec![revealed_state]);
                    }
                }

                let Some(initial_stack) = initial_stack_opt else {
                    // No goto for this (revealed_state, nt); nothing to explore
                    continue;
                };
                let baseline_len = initial_stack.len();

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
                                // Lookahead is consumed here. Record the escape edge with the
                                // suffix needed to push beyond the baseline.
                                let mut new_stack = stack.clone();
                                new_stack.push(*next_state);
                                let to_push = new_stack[baseline_len..].to_vec();
                                let dest = SpecialPrecomputeDest::Escape { push_states: to_push };
                                normal_edges.insert((*src_nt, revealed_state, terminal, dest));
                                // Do not continue after a shift.
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                nonterminal_id,
                                len,
                                ..
                            } => {
                                // Inlined handle_reduce
                                let len = *len;
                                let reduce_nt = *nonterminal_id;
                                let above_baseline = stack.len() - baseline_len;
                                if len > above_baseline {
                                    let pop_below_baseline = len - above_baseline;
                                    let dest = SpecialPrecomputeDest::Reduce {
                                        pop: pop_below_baseline,
                                        dest_nt: reduce_nt,
                                    };
                                    normal_edges.insert((*src_nt, revealed_state, terminal, dest));
                                } else {
                                    let mut after_pop = stack.clone();
                                    after_pop.truncate(after_pop.len() - len);
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
                                    let mut new_stack = stack.clone();
                                    new_stack.push(*next_state);
                                    let to_push = new_stack[baseline_len..].to_vec();
                                    let dest =
                                        SpecialPrecomputeDest::Escape { push_states: to_push };
                                    normal_edges.insert((*src_nt, revealed_state, terminal, dest));
                                }
                                for (len, nts) in reduces {
                                    for (nt, _) in nts {
                                        // Inlined handle_reduce
                                        let len = *len;
                                        let reduce_nt = *nt;
                                        let above_baseline = stack.len() - baseline_len;
                                        if len > above_baseline {
                                            let pop_below_baseline = len - above_baseline;
                                            let dest = SpecialPrecomputeDest::Reduce {
                                                pop: pop_below_baseline,
                                                dest_nt: reduce_nt,
                                            };
                                            normal_edges.insert((*src_nt, revealed_state, terminal, dest));
                                        } else {
                                            let mut after_pop = stack.clone();
                                            after_pop.truncate(after_pop.len() - len);
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

    // Stage 2: Super Edges
    let trie1_god = &gc.trie1_god;
    let trie1_roots: Vec<_> = gc.precomputed1.values().cloned().collect();
    if trie1_roots.is_empty() {
        return SpecialPrecomputation {
            normal_edges,
            super_edges: super_edges.into_inner(),
        };
    }

    // The value type for special_map
    type SpecialMapValue = HashSet<(Vec<StateID>, LLMTokenBV, PrecomputeNode1Index)>;

    // Initial values for the traversal
    let mut initial_values_map: BTreeMap<PrecomputeNode1Index, SpecialMapValue> = BTreeMap::new();
    for pci1_root in trie1_roots.iter().cloned() {
        let initial_set = initial_values_map.entry(pci1_root).or_default();
        initial_set.insert((
            vec![parser.start_state_id],
            gc.all_internal_llm_tokens_bitset_precompute1(),
            pci1_root,
        ));
    }
    let initial_values: Vec<_> = initial_values_map.into_iter().collect();

    // We need to group normal_edges for efficient lookup.
    let mut normal_edges_map: HashMap<
        (Option<NonTerminalID>, StateID, TerminalID),
        Vec<&SpecialPrecomputeDest>,
    > = HashMap::new();
    for edge in &normal_edges {
        normal_edges_map
            .entry((edge.0, edge.1, edge.2))
            .or_default()
            .push(&edge.3);
    }

    let traversal_data = Trie::compute_traversal_data(trie1_god, &trie1_roots).unwrap();

    Trie::special_map_grouped(
        trie1_god,
        &traversal_data,
        initial_values,
        // step
        |current_set, terminal_opt, dest_map| {
            let mut results_for_dests: HashMap<PrecomputeNode1Index, SpecialMapValue> =
                HashMap::new();
            let terminal = match terminal_opt {
                Some(t) => *t,
                None => return vec![], // No grammar token on edge, no grammar actions to take.
            };

            for (stack, llm_bv, pci1_start) in current_set {
                for (pci1_dest, edge_llm_bv) in dest_map.iter() {
                    let intersected_bv = llm_bv & edge_llm_bv;
                    if intersected_bv.is_empty() {
                        continue;
                    }

                    let mut q: VecDeque<(Option<NonTerminalID>, Vec<StateID>, LLMTokenBV)> =
                        VecDeque::new();
                    q.push_back((None, stack.clone(), intersected_bv.clone()));

                    let mut visited_q_states = HashSet::new();

                    while let Some((src_nt, current_stack, current_bv)) = q.pop_front() {
                        if !visited_q_states.insert((src_nt, current_stack.clone())) {
                            continue;
                        }

                        if current_stack.is_empty() {
                            continue;
                        }
                        let top_state = *current_stack.last().unwrap();

                        if let Some(dests) = normal_edges_map.get(&(src_nt, top_state, terminal))
                        {
                            for dest in dests {
                                match dest {
                                    SpecialPrecomputeDest::Reduce { pop, dest_nt } => {
                                        if current_stack.len() <= *pop {
                                            let pop_remainder = pop - current_stack.len();
                                            let super_edge = (
                                                src_nt,
                                                terminal,
                                                (pop_remainder, *dest_nt),
                                                current_bv.clone(),
                                                *pci1_start,
                                                *pci1_dest,
                                            );
                                            super_edges.borrow_mut().insert(super_edge);
                                        } else {
                                            let mut new_stack = current_stack.clone();
                                            new_stack.truncate(new_stack.len() - *pop);
                                            q.push_back((
                                                Some(*dest_nt),
                                                new_stack,
                                                current_bv.clone(),
                                            ));
                                        }
                                    }
                                    SpecialPrecomputeDest::Escape { push_states } => {
                                        let mut new_stack = current_stack.clone();
                                        new_stack.extend(push_states);
                                        results_for_dests
                                            .entry(*pci1_dest)
                                            .or_default()
                                            .insert((
                                                new_stack,
                                                current_bv.clone(),
                                                *pci1_start,
                                            ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            results_for_dests.into_iter().collect::<Vec<_>>()
        },
        // merge
        |set1, set2| {
            set1.extend(set2.into_iter());
        },
        // process
        |_, _| true,
    );

    SpecialPrecomputation {
        normal_edges,
        super_edges: super_edges.into_inner(),
    }
}

pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
    let sp = &gcs.parent.special_precomputation;
    let final_mask = RefCell::new(LLMTokenBV::zeros());

    let mut q: VecDeque<(GLRParserState, PrecomputeNode1Index, Option<NonTerminalID>)> =
        VecDeque::new();
    let mut visited: HashSet<(
        Arc<GSSNode>,
        PrecomputeNode1Index,
        Option<NonTerminalID>,
    )> = HashSet::new();

    // Initial states
    for (tokenizer_id, glr_state) in &gcs.state {
        if let Some(pci1_root) = gcs.parent.precomputed1.get(tokenizer_id) {
            let key = (glr_state.active_state.stack.clone(), *pci1_root, None);
            if visited.insert(key) {
                q.push_back((glr_state.clone(), *pci1_root, None));
            }
        }
    }

    while let Some((glr_state, pci1_idx, src_nt)) = q.pop_front() {
        if !glr_state.is_ok() {
            continue;
        }

        let guard = pci1_idx.read(&gcs.parent.trie1_god).unwrap();
        if guard.value.end {
            *final_mask.borrow_mut() |= &glr_state.active_state.stack.allowed_llm_tokens();
        }

        // Follow super edges
        // ... (This part is complex and requires careful GSS manipulation, omitted for now)

        // Follow normal precompute1 edges using normal_edges (Escape)
        // ... (This part is also complex, omitted for now)
    }

    gcs.parent.internal_bv_to_original_precompute1(&final_mask.into_inner())
}

pub fn dump_precomputed_special(gc: &GrammarConstraint) {
    let sp = &gc.special_precomputation;
    let parser = &gc.parser;

    println!("--- Special Precomputation Dump ---");

    // For resolving NT IDs to names
    let get_nt_name = |nt_id: &NonTerminalID| -> String {
        parser
            .non_terminal_map
            .get_by_right(nt_id)
            .map(|nt| nt.to_string())
            .unwrap_or_else(|| format!("NT({})", nt_id.0))
    };
    let get_opt_nt_name = |opt_nt_id: &Option<NonTerminalID>| -> String {
        opt_nt_id
            .as_ref()
            .map(get_nt_name)
            .unwrap_or_else(|| "None".to_string())
    };

    // For resolving Terminal IDs to names
    let get_term_name = |term_id: &TerminalID| -> String {
        parser
            .terminal_map
            .get_by_right(term_id)
            .map(|t| t.to_string())
            .unwrap_or_else(|| format!("T({})", term_id.0))
    };

    println!("\nNormal Edges ({}):", sp.normal_edges.len());
    println!("{:-<125}", "");
    println!(
        "{:<20} | {:<15} | {:<20} | {:<60}",
        "Source NT", "Initial State", "Terminal", "Destination"
    );
    println!("{:-<125}", "");

    let mut sorted_normal_edges: Vec<_> = sp.normal_edges.iter().collect();
    sorted_normal_edges.sort_unstable();

    for (src_nt, initial_state, terminal, dest) in sorted_normal_edges {
        let dest_str = match dest {
            SpecialPrecomputeDest::Reduce { pop, dest_nt } => {
                format!("Reduce(pop={}, dest_nt={})", pop, get_nt_name(dest_nt))
            }
            SpecialPrecomputeDest::Escape { push_states } => {
                let states_str: Vec<_> = push_states.iter().map(|s| s.0.to_string()).collect();
                format!("Escape(push=[{}])", states_str.join(", "))
            }
        };

        println!(
            "{:<20} | S{:<14} | {:<20} | {}",
            get_opt_nt_name(src_nt),
            initial_state.0,
            get_term_name(terminal),
            dest_str
        );
    }

    println!("\nSuper Edges ({}):", sp.super_edges.len());
    println!("{:-<150}", "");
    println!(
        "{:<20} | {:<20} | {:<30} | {:<15} | {:<15} | {:<20}",
        "Source NT", "Terminal", "Destination", "PCI1 Start", "PCI1 End", "LLM Tokens"
    );
    println!("{:-<150}", "");

    let mut sorted_super_edges: Vec<_> = sp.super_edges.iter().collect();
    sorted_super_edges.sort_unstable();

    for (src_nt, terminal, (pop, dest_nt), edge_bv, pci1_start, pci1_end) in sorted_super_edges {
        let dest_str = format!("Reduce(pop={}, dest_nt={})", pop, get_nt_name(dest_nt));
        let bv_str = if edge_bv.is_all() {
            "ALL".to_string()
        } else {
            format!("{} tokens", edge_bv.len())
        };

        println!(
            "{:<20} | {:<20} | {:<30} | {:<15} | {:<15} | {:<20}",
            get_opt_nt_name(src_nt),
            get_term_name(terminal),
            dest_str,
            pci1_start.as_usize(),
            pci1_end.as_usize(),
            bv_str
        );
    }

    println!("\n--- End Special Precomputation Dump ---");
}
