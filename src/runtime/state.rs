




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::compiler::glr::parser::{ParserGSS, stacks_finished};

use super::constraint::Constraint;

// SEP1_MAP: this file maps to sep1 `GrammarConstraintState` in
// `grammars2024/src/constraint.rs`, with execution/mask helpers split into
// sibling files instead of living on one large impl block.







#[derive(Debug, Clone)]
// SEP1_MAP: `ConstraintState` is the direct analogue of sep1
// `GrammarConstraintState`, but glrmask stores a simpler
// `BTreeMap<u32, ParserGSS>` instead of sep1's `GLRParserState` wrapper.
pub struct ConstraintState<'a> {
    
    pub(crate) constraint: &'a Constraint,
    
    pub(crate) state: BTreeMap<u32, ParserGSS>,
}

impl<'a> ConstraintState<'a> {
    // SEP1_MAP: nearest sep1 analogue is `GrammarConstraintState::is_valid()`
    // plus sep1's completion checks inside `get_mask()`; there is no clean
    // one-method equivalent for glrmask `is_finished()`.
    
    pub fn is_finished(&self) -> bool {
        self.state
            .values()
            .any(|stack| stacks_finished(&self.constraint.table, stack))
    }
}
