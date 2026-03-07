
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


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regex {
    
    pub dfa: DFA,
}

impl Regex {
    
    pub fn num_states(&self) -> usize {
        unimplemented!()
    }

    
    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        unimplemented!()
    }

    
    pub fn get_u8set(&self, state: u32) -> U8Set {
        unimplemented!()
    }
}

impl Expr {
    
    pub fn build(self) -> Regex {
        unimplemented!()
    }

}

impl ExprGroups {
    
    pub fn build(self) -> Regex {
        unimplemented!()
    }

    
    pub fn build_nfa(self) -> NFA {
        unimplemented!()
    }
}
