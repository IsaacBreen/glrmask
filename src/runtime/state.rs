#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::compiler::glr::parser::{ParserGSS, stacks_finished};

use super::constraint::Constraint;



#[derive(Debug, Clone)]
pub struct ConstraintState<'a> {
    
    pub(crate) constraint: &'a Constraint,
    
    pub(crate) state: BTreeMap<u32, ParserGSS>,
}

impl<'a> ConstraintState<'a> {
    pub fn is_complete(&self) -> bool {
        let initial_tsid = self.constraint.tokenizer.initial_state();
        let Some(stack) = self.state.get(&initial_tsid) else {
            return false;
        };
        !stack.is_empty() && stacks_finished(&self.constraint.table, stack)
    }

    
    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }
}
