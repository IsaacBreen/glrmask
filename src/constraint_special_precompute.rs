use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index, Trie1GodWrapper};
use crate::datastructures::gss_leveled_adapter::{
    allow_only_llm_tokens_and_prune_arc, prune_llm_tokens_by_disallowed_terminals, GSSNode,
};
use crate::datastructures::trie::Trie;
use crate::glr::parser::{GLRParser, GLRParserState};
use crate::glr::table::{Goto, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID};
use crate::types::TerminalID;

/// A destination reached by a normal (per-terminal) exploration step:
/// - Reduce: the path reduces below the baseline by `pop` and transitions to `dest_nt`.
/// - Escape: the path shifts; the new states to push relative to baseline are `push_states`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpecialPrecomputeDest {
    Reduce { pop: usize, dest_nt: NonTerminalID },
    Escape { push_states: Vec<StateID> },
}

/// A "normal" edge precomputation entry.
/// (src_node_nonterminal, revealed_state, terminal, dest)
pub type SpecialPrecomputeNormalEdge = (
    Option<NonTerminalID>,
    StateID,
    TerminalID,
    SpecialPrecomputeDest,
);

/// A "super edge" summarizes a compound reduction step along a Trie1 edge carrying a token filter:
/// (src_node_nonterminal, terminal, (pop, dest_nt), llm_token_filter, trie1_start, trie1_end)
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

/// Get actions (shift/reduce/split) for the given state and terminal.
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

/// Get goto transitions for a given state and non-terminal.
fn get_gotos<'a>(parser: &'a GLRParser, state_id: StateID, nt_id: NonTerminalID) -> Vec<&'a Goto> {
    if let Some(row) = parser.table.get(&state_id) {
        row.gotos.get(&nt_id).map(|g| vec![g]).unwrap_or_default()
    } else {
        vec![]
    }
}

/// Build a compact set of "normal" edges by exploring, per (src_nt, revealed_state, terminal):
/// - Start stack baseline:
///   * src_nt=None       => stack=[revealed_state], baseline_len=1
///   * src_nt=Some(nt)   => stack=[revealed_state, goto(revealed_state, nt)] if exists (else skip)
/// - BFS transitions:
///   * Shift(next_state) => Escape edge with push = suffix beyond baseline
///   * Reduce(len, nt)   => If it crosses baseline, emit Reduce with pop below baseline; else apply goto and continue
///   * Split             => Do both (shift and reduces)
fn precompute_normal_edges(parser: &GLRParser) -> HashSet<SpecialPrecomputeNormalEdge> {
    let mut normal_edges = HashSet::new();

    // Enumerate nonterminals including the synthetic "start/end" node represented as None.
    let mut src_nts: Vec<Option<NonTerminalID>> =
        parser.non_terminal_map.right_values().copied().map(Some).collect();
    src_nts.push(None);

    let mut terminals: Vec<TerminalID> = parser.terminal_map.right_values().copied().collect();
    terminals.sort_by_key(|t| t.0);

    let mut states: Vec<StateID> = parser.table.keys().copied().collect();
    states.sort_by_key(|s| s.0);

    for src_nt in &src_nts {
        for &revealed_state in &states {
            for &terminal in &terminals {
                // Build initial stack with baseline
                let mut initial_stack_opt: Option<Vec<StateID>> = None;
                match src_nt {
                    Some(nt) => {
                        for goto in get_gotos(parser, revealed_state, *nt) {
                            if let Some(goto_state) = goto.state_id {
                                initial_stack_opt = Some(vec![revealed_state, goto_state]);
                            }
                        }
                    }
                    None => {
                        initial_stack_opt = Some(vec![revealed_state]);
                    }
                }

                let Some(initial_stack) = initial_stack_opt else {
                    continue;
                };

                let baseline_len = initial_stack.len();
                let mut q: VecDeque<Vec<StateID>> = VecDeque::new();
                let mut seen: HashSet<Vec<StateID>> = HashSet::new();
                q.push_back(initial_stack.clone());
                seen.insert(initial_stack);

                while let Some(stack) = q.pop_front() {
                    if stack.is_empty() {
                        continue;
                    }

                    let top = *stack.last().unwrap();
                    let actions = get_actions(parser, top, terminal);
                    if actions.is_empty() {
                        continue;
                    }

                    for action in actions {
                        match action {
                            Stage7ShiftsAndReducesLookaheadValue::Shift(next_state) => {
                                let mut next_stack = stack.clone();
                                next_stack.push(*next_state);
                                // Push suffix beyond baseline
                                let suffix = next_stack[baseline_len..].to_vec();
                                normal_edges.insert((
                                    *src_nt,
                                    revealed_state,
                                    terminal,
                                    SpecialPrecomputeDest::Escape { push_states: suffix },
                                ));
                                // Do not continue after the first consuming shift; baseline modeling accounts for this edge already.
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                nonterminal_id,
                                len,
                                ..
                            } => {
                                let len = *len;
                                let red_nt = *nonterminal_id;
                                let above_baseline = stack.len().saturating_sub(baseline_len);
                                if len > above_baseline {
                                    // Crosses baseline: below-baseline pop remainder
                                    let pop_below_baseline = len - above_baseline;
                                    normal_edges.insert((
                                        *src_nt,
                                        revealed_state,
                                        terminal,
                                        SpecialPrecomputeDest::Reduce {
                                            pop: pop_below_baseline,
                                            dest_nt: red_nt,
                                        },
                                    ));
                                } else {
                                    // Still above or at baseline: apply pop + goto and continue BFS
                                    let mut after_pop = stack.clone();
                                    after_pop.truncate(after_pop.len().saturating_sub(len));
                                    if let Some(&new_top) = after_pop.last() {
                                        for goto in get_gotos(parser, new_top, red_nt) {
                                            if let Some(goto_state) = goto.state_id {
                                                let mut after_goto = after_pop.clone();
                                                after_goto.push(goto_state);
                                                if seen.insert(after_goto.clone()) {
                                                    q.push_back(after_goto);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                if let Some(next_state) = shift {
                                    let mut next_stack = stack.clone();
                                    next_stack.push(*next_state);
                                    let suffix = next_stack[baseline_len..].to_vec();
                                    normal_edges.insert((
                                        *src_nt,
                                        revealed_state,
                                        terminal,
                                        SpecialPrecomputeDest::Escape { push_states: suffix },
                                    ));
                                }
                                for (len, nts) in reduces {
                                    for (nt, _) in nts {
                                        let len = *len;
                                        let red_nt = *nt;
                                        let above_baseline = stack.len().saturating_sub(baseline_len);
                                        if len > above_baseline {
                                            let pop_below_baseline = len - above_baseline;
                                            normal_edges.insert((
                                                *src_nt,
                                                revealed_state,
                                                terminal,
                                                SpecialPrecomputeDest::Reduce {
                                                    pop: pop_below_baseline,
                                                    dest_nt: red_nt,
                                                },
                                            ));
                                        } else {
                                            let mut after_pop = stack.clone();
                                            after_pop.truncate(after_pop.len().saturating_sub(len));
                                            if let Some(&new_top) = after_pop.last() {
                                                for goto in get_gotos(parser, new_top, red_nt) {
                                                    if let Some(goto_state) = goto.state_id {
                                                        let mut after_goto = after_pop.clone();
                                                        after_goto.push(goto_state);
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

    normal_edges
}

/// Public entry: compute special precomputation artifacts.
/// This rewrite focuses on correctness and determinism:
/// - Build compact "normal edges" via a baseline-aware BFS.
/// - "Super edges" are optional summaries across Trie1; for now we keep them empty as
///   the runtime algorithm below no longer requires them to compute masks.
pub fn precompute_special(gc: &GrammarConstraint) -> SpecialPrecomputation {
    let parser = &gc.parser;
    let normal_edges = precompute_normal_edges(parser);

    // This implementation of get_mask4 no longer depends on super edges.
    // We keep the set for debugging/compatibility and leave it empty.
    SpecialPrecomputation {
        normal_edges,
        super_edges: HashSet::new(),
    }
}

/// Compute the current-commit mask (original token space) using Trie1 + GLR simulation.
/// This version avoids the previous push/pop template machinery and instead:
/// - Initializes (Trie1-root, GLRState) pairs for each active tokenizer state.
/// - Traverses Trie1 using special_map_grouped with SCC-aware scheduling:
///   * For edge Some(terminal): process_token(terminal) on a clone of the GLR state,
///     then prune the GSS by the edge LLMTokenBV.
///   * For edge None: only prune by the edge LLMTokenBV.
/// - Whenever a Trie1 node with end=true is reached, union the GSS's allowed LLM tokens
///   into the final internal mask.
/// - Finally, map the internal mask (stage 1) to original-space IDs.
pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
    // If there are no active states, return empty mask.
    if gcs.state.is_empty() {
        return LLMTokenBV::zeros();
    }

    // Seed initial (Trie1-root -> GLRState) map, pruning disallowed terminals first.
    let mut initial_values_by_trie_node: BTreeMap<PrecomputeNode1Index, GLRParserState<'_>> = BTreeMap::new();
    for (&tokenizer_state_id, glr_state) in &gcs.state {
        if glr_state.active_state.stack.is_empty() {
            continue;
        }
        let mut glr_state_cloned = glr_state.clone();
        prune_llm_tokens_by_disallowed_terminals(
            &mut glr_state_cloned.active_state.stack,
            &gcs.parent.possible_matches,
            &mut HashMap::new(),
        );

        if let Some(&trie1_root) = gcs.parent.precomputed1.get(&tokenizer_state_id) {
            initial_values_by_trie_node
                .entry(trie1_root)
                .and_modify(|existing_glr| {
                    existing_glr.merge_with(glr_state_cloned.clone());
                })
                .or_insert(glr_state_cloned);
        }
    }

    if initial_values_by_trie_node.is_empty() {
        return LLMTokenBV::zeros();
    }

    let trie1_god: &Trie1GodWrapper = &gcs.parent.trie1_god;

    let roots_for_traversal: Vec<_> = initial_values_by_trie_node.keys().cloned().collect();
    let traversal_data = match Trie::compute_traversal_data(trie1_god, &roots_for_traversal) {
        Some(d) => d,
        None => {
            // No reachable structure; return empty
            return LLMTokenBV::zeros();
        }
    };

    // Prepare final-mask accumulator in internal (stage 1) domain.
    let final_mask_internal = RefCell::new(LLMTokenBV::zeros());

    // Prepare initial value vector for the grouped traversal
    let initial_values_vec: Vec<_> = initial_values_by_trie_node.into_iter().collect();

    // Traverse Trie1 with SCC-aware scheduling.
    Trie::special_map_grouped(
        trie1_god,
        &traversal_data,
        initial_values_vec,
        // step: (glr_s, edge_key, destinations_map) -> Vec<(child_idx, new_glr_state)>
        |glr_s, edge_key_opt, destinations_map: &OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>| {
            let mut out: Vec<(PrecomputeNode1Index, GLRParserState<'_>)> = Vec::new();

            match edge_key_opt {
                Some(tid) => {
                    // Consume grammar terminal along this edge.
                    for (child_idx, edge_bv) in destinations_map.iter() {
                        let mut next_state = glr_s.clone();

                        // Apply grammar token
                        next_state.process_token(*tid);

                        // If parse state OK, prune LLM tokens by the edge's bitset
                        if next_state.is_ok() {
                            allow_only_llm_tokens_and_prune_arc(
                                &mut next_state.active_state.stack,
                                edge_bv,
                                &mut HashMap::new(),
                            );
                            if next_state.is_ok() {
                                out.push((*child_idx, next_state));
                            }
                        }
                    }
                }
                None => {
                    // No grammar token: just filter by LLM token bitset and keep going.
                    for (child_idx, edge_bv) in destinations_map.iter() {
                        let mut next_state = glr_s.clone();
                        allow_only_llm_tokens_and_prune_arc(
                            &mut next_state.active_state.stack,
                            edge_bv,
                            &mut HashMap::new(),
                        );
                        if next_state.is_ok() {
                            out.push((*child_idx, next_state));
                        }
                    }
                }
            }

            out
        },
        // merge: combine GLR parser states
        |s1, s2| s1.merge_with(s2),
        // process: when at an end node, accumulate GSS-allowed tokens into final mask.
        |precomputed_node_data, glr_s| {
            if !glr_s.is_ok() {
                return false;
            }
            if precomputed_node_data.value.end {
                let allowed = glr_s.active_state.stack.allowed_llm_tokens();
                if !allowed.is_empty() {
                    *final_mask_internal.borrow_mut() |= &allowed;
                }
            }
            true
        },
    );

    // Map internal (stage 1) mask to original token IDs.
    gcs.parent.internal_bv_to_original_precompute1(&final_mask_internal.into_inner())
}

/// Debug/inspection helper: dump both normal and super edges if present.
pub fn dump_precomputed_special(gc: &GrammarConstraint) {
    let sp = &gc.special_precomputation;
    let parser = &gc.parser;

    println!("--- Special Precomputation Dump ---");

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
