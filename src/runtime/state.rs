




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::compiler::glr::parser::ParserGSS;

use super::constraint::Constraint;








#[derive(Debug, Clone)]
pub struct ConstraintState<'a> {
    
    pub(crate) constraint: &'a Constraint,
    
    pub(crate) state: BTreeMap<u32, ParserGSS>,
}

impl<'a> ConstraintState<'a> {
    
    pub fn is_finished(&self) -> bool {
        unimplemented!()
    }
}
