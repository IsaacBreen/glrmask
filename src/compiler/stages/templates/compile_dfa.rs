//! NOTE: template DFA compilation is intentionally deferred.
//! Keep only the terminal → template-DFA mapping surface in this cleanup pass.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;

#[derive(Debug, Clone, Default)]
pub struct Templates {
    pub by_terminal: BTreeMap<TerminalID, UnweightedDfa>,
}

impl Templates {
    pub(crate) fn from_characterizations(
        _characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> Self {
        todo!("template DFA compilation is intentionally left as a placeholder in this cleanup pass")
    }
}