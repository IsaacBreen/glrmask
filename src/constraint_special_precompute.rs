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

    for src_nt in &non_terminals {
        for &initial_state in &states {
            for &terminal in &terminals {
                let mut q: VecDeque<(Vec<StateID>, Vec<StateID>)> = VecDeque::new(); // (stack, pushed_path)
                let mut visited_stacks = HashSet::new();

                let initial_stacks = if let Some(nt) = src_nt {
                    let gotos = get_gotos(parser, initial_state, *nt);
                    let mut stacks = vec![];
                    for goto in gotos {
                        if let Some(goto_state) = goto.state_id {
                            stacks.push((vec![initial_state, goto_state], vec![goto_state]));
                        }
                    }
                    stacks
                } else {
                    vec![(vec![initial_state], vec![])]
                };

                for (stack, path) in initial_stacks {
                    if visited_stacks.insert(stack.clone()) {
                        q.push_back((stack, path));
                    }
                }

                while let Some((stack, pushed_path)) = q.pop_front() {
                    let top_state = *stack.last().unwrap();
                    let actions = get_actions(parser, top_state, terminal);

                    for action in actions {
                        let mut handle_reduce =
                            |normal_edges: &mut HashSet<SpecialPrecomputeNormalEdge>,
                             len: usize,
                             reduce_nt: NonTerminalID| {
                                if stack.len() <= len {
                                    let pop_below = len - stack.len();
                                    let dest = SpecialPrecomputeDest::Reduce {
                                        pop: pop_below,
                                        dest_nt: reduce_nt,
                                    };
                                    normal_edges.insert((*src_nt, initial_state, terminal, dest));
                                } else {
                                    let mut new_stack = stack.clone();
                                    new_stack.truncate(new_stack.len() - len);
                                    let new_top_state = *new_stack.last().unwrap();
                                    let gotos = get_gotos(parser, new_top_state, reduce_nt);
                                    for goto in gotos {
                                        if let Some(goto_state) = goto.state_id {
                                            let mut stack_after_goto = new_stack.clone();
                                            stack_after_goto.push(goto_state);
                                            let mut path_after_goto = pushed_path.clone();
                                            path_after_goto.push(goto_state);
                                            if visited_stacks.insert(stack_after_goto.clone()) {
                                                q.push_back((stack_after_goto, path_after_goto));
                                            }
                                        }
                                    }
                                }
                            };

                        match action {
                            Stage7ShiftsAndReducesLookaheadValue::Shift(next_state) => {
                                let mut new_pushed = pushed_path.clone();
                                new_pushed.push(*next_state);
                                let dest = SpecialPrecomputeDest::Escape {
                                    push_states: new_pushed,
                                };
                                normal_edges.insert((*src_nt, initial_state, terminal, dest));
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                nonterminal_id,
                                len,
                                ..
                            } => {
                                handle_reduce(&mut normal_edges, *len, *nonterminal_id);
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                if let Some(next_state) = shift {
                                    let mut new_pushed = pushed_path.clone();
                                    new_pushed.push(*next_state);
                                    let dest = SpecialPrecomputeDest::Escape {
                                        push_states: new_pushed,
                                    };
                                    normal_edges.insert((*src_nt, initial_state, terminal, dest));
                                }
                                for (len, nts) in reduces {
                                    for (nt, _) in nts {
                                        handle_reduce(&mut normal_edges, *len, *nt);
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
    let precomputed1 = &gc.precomputed1;
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();

    if trie1_roots.is_empty() {
        return SpecialPrecomputation {
            normal_edges,
            super_edges: super_edges.into_inner(),
        };
    }

    let traversal_data = Trie::compute_traversal_data(trie1_god, &trie1_roots)
        .expect("Failed to compute traversal data for trie1 in special precomputation");

    let mut initial_special_state = BTreeSet::new();
    for &state_id in &states {
        initial_special_state.insert((None, state_id));
    }

    let initial_nodes_and_values: Vec<_> = precomputed1
        .values()
        .map(|root_idx| (*root_idx, initial_special_state.clone()))
        .collect();

    Trie::special_map_grouped(
        trie1_god,
        &traversal_data,
        initial_nodes_and_values,
        // step
        |current_special_states, edge_terminal_opt, destinations_map| {
            let mut results = Vec::new();
            let terminal = match edge_terminal_opt {
                Some(t) => *t,
                None => {
                    // Pass through
                    for (dest_pci1, _edge_bv) in destinations_map {
                        results.push((*dest_pci1, current_special_states.clone()));
                    }
                    return results;
                }
            };

            let mut next_special_states_for_escape = BTreeSet::new();
            for (src_nt, initial_state) in current_special_states {
                // Find matching normal edges
                for (ne_src_nt, ne_initial_state, ne_terminal, ne_dest) in &normal_edges {
                    if ne_src_nt == src_nt
                        && *ne_initial_state == *initial_state
                        && *ne_terminal == terminal
                    {
                        match ne_dest {
                            SpecialPrecomputeDest::Reduce { pop, dest_nt } => {
                                for (dest_pci1, edge_bv) in destinations_map {
                                    if !edge_bv.is_empty() {
                                        super_edges.borrow_mut().insert((
                                            *src_nt,
                                            terminal,
                                            (*pop, *dest_nt),
                                            edge_bv.clone(),
                                            pci1_idx,
                                            *dest_pci1,
                                        ));
                                    }
                                }
                            }
                            SpecialPrecomputeDest::Escape { push_states } => {
                                if let Some(new_top_state) = push_states.last() {
                                    next_special_states_for_escape.insert((*src_nt, *new_top_state));
                                }
                            }
                        }
                    }
                }
            }

            if !next_special_states_for_escape.is_empty() {
                for (dest_pci1, _edge_bv) in destinations_map {
                    results.push((*dest_pci1, next_special_states_for_escape.clone()));
                }
            }
            results
        },
        // merge
        |set1, set2| {
            set1.extend(set2);
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
            "{:<20} | S{:<13} | {:<20} | {}",
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
