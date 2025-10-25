use std::collections::{BTreeSet, HashSet, VecDeque};

use crate::constraint::{
    GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index
};
use crate::glr::parser::GLRParser;
use crate::glr::table::{Goto, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID};
use crate::types::TerminalID;

// Types for special precomputation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum SpecialPrecomputeDest {
        Reduce { pop: usize, dest_nt: NonTerminalID },
        Escape { push_states: Vec<StateID> },
    }

    // (Option<NonTerminalID>, StateID, TerminalID, SpecialPrecomputeDest)
    pub type SpecialPrecomputeNormalEdge =
        (Option<NonTerminalID>, StateID, TerminalID, SpecialPrecomputeDest);

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
        let parser = &gc.parser;

        let mut non_terminals: Vec<Option<NonTerminalID>> =
            parser.non_terminal_map.right_values().copied().map(Some).collect();
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
                            let mut handle_reduce = |
                                normal_edges: &mut HashSet<SpecialPrecomputeNormalEdge>,
                                len: usize,
                                reduce_nt: NonTerminalID,
                            | {
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

        SpecialPrecomputation {
            normal_edges,
            super_edges: HashSet::new(),
        }
    }

    pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
        todo!()
    }