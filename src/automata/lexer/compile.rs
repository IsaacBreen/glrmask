
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This regex-to-automata build surface is closest to sep1's regex and tokenizer compilation flow spread across `finite_automata.rs` and `dfa_u8/dfa.rs`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ds::u8set::U8Set;

use super::ast::{Expr, ExprGroups};
use super::dfa::DFA;
use super::nfa::NFA;

fn expr_accepts_empty(expr: &Expr) -> bool {
    match expr {
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::U8Class(_) => false,
        Expr::Seq(parts) => parts.iter().all(expr_accepts_empty),
        Expr::Choice(options) => options.iter().any(expr_accepts_empty),
        Expr::Repeat { expr: _, min, .. } => *min == 0,
        Expr::Shared(inner) => expr_accepts_empty(inner),
        Expr::Epsilon => true,
    }
}

fn expr_u8set(expr: &Expr) -> U8Set {
    match expr {
        Expr::U8Seq(bytes) => U8Set::from_bytes(bytes),
        Expr::U8Class(set) => *set,
        Expr::Seq(parts) | Expr::Choice(parts) => parts
            .iter()
            .fold(U8Set::empty(), |acc, part| acc | expr_u8set(part)),
        Expr::Repeat { expr, .. } => expr_u8set(expr),
        Expr::Shared(inner) => expr_u8set(inner),
        Expr::Epsilon => U8Set::empty(),
    }
}

fn compile_expr(expr: &Expr, nfa: &mut NFA, start: u32, end: u32) {
    match expr {
        Expr::U8Seq(bytes) => {
            let mut state = start;
            for (index, &byte) in bytes.iter().enumerate() {
                let next = if index + 1 == bytes.len() {
                    end
                } else {
                    nfa.add_state()
                };
                nfa.add_transition(state, byte, next);
                state = next;
            }
            if bytes.is_empty() {
                nfa.add_epsilon(start, end);
            }
        }
        Expr::U8Class(set) => {
            nfa.add_u8set_transition(start, *set, end);
        }
        Expr::Seq(parts) => {
            let mut state = start;
            for (index, part) in parts.iter().enumerate() {
                let next = if index + 1 == parts.len() {
                    end
                } else {
                    nfa.add_state()
                };
                compile_expr(part, nfa, state, next);
                state = next;
            }
            if parts.is_empty() {
                nfa.add_epsilon(start, end);
            }
        }
        Expr::Choice(options) => {
            if options.is_empty() {
                nfa.add_epsilon(start, end);
            }
            for option in options {
                compile_expr(option, nfa, start, end);
            }
        }
        Expr::Repeat { expr, min, max } => {
            let mut current = start;
            for _ in 0..*min {
                let next = nfa.add_state();
                compile_expr(expr, nfa, current, next);
                current = next;
            }

            match max {
                Some(max) => {
                    let optional = max.saturating_sub(*min);
                    for _ in 0..optional {
                        nfa.add_epsilon(current, end);
                        let next = nfa.add_state();
                        compile_expr(expr, nfa, current, next);
                        current = next;
                    }
                    nfa.add_epsilon(current, end);
                }
                None => {
                    nfa.add_epsilon(current, end);
                    let loop_state = nfa.add_state();
                    compile_expr(expr, nfa, current, loop_state);
                    nfa.add_epsilon(loop_state, current);
                    if expr_accepts_empty(expr) {
                        nfa.add_epsilon(loop_state, end);
                    }
                }
            }
        }
        Expr::Shared(inner) => compile_expr(inner, nfa, start, end),
        Expr::Epsilon => nfa.add_epsilon(start, end),
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regex {
    
    pub dfa: DFA,
}

impl Regex {
    
    pub fn num_states(&self) -> usize {
        self.dfa.num_states()
    }

    
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    
    pub fn get_u8set(&self, state: u32) -> U8Set {
        self.dfa.get_u8set(state)
    }
}

impl Expr {
    
    pub fn build(self) -> Regex {
        ExprGroups {
            groups: vec![super::ast::ExprGroup {
                expr: self,
                is_non_greedy: false,
            }],
        }
        .build()
    }

}

impl ExprGroups {
    
    pub fn build(self) -> Regex {
        let group_sets: Vec<U8Set> = self
            .groups
            .iter()
            .map(|group| expr_u8set(&group.expr))
            .collect();
        let mut dfa = self.build_nfa().to_dfa();
        dfa.ensure_group_capacity(group_sets.len());
        for (group_id, set) in group_sets.into_iter().enumerate() {
            dfa.set_group_u8set(group_id as u32, set);
        }
        Regex { dfa }
    }

    
    pub fn build_nfa(self) -> NFA {
        let mut nfa = NFA::new(1);
        for (group_id, group) in self.groups.into_iter().enumerate() {
            let accept = nfa.add_state();
            compile_expr(&group.expr, &mut nfa, 0, accept);
            nfa.add_finalizer(accept, group_id as u32);
            if group.is_non_greedy {
                nfa.add_non_greedy_finalizer(accept, group_id as u32);
            }
        }
        nfa
    }
}
