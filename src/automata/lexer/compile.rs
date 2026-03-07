
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

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
