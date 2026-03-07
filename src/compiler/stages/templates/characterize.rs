//! NOTE: terminal characterization is intentionally deferred.
//! Keep only the minimal data shape and entrypoint for this cleanup pass.
// SEP1_MAP: This placeholder file corresponds directly to sep1's `precompute4/characterize.rs` terminal-characterization pass.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::{NonterminalID, TerminalID};

type InitialShift = (u32, u32);

type InitialReduce = (u32, usize, NonterminalID);

type NtEscape = (NonterminalID, u32, u32, u32);

type NtRereduce = (NonterminalID, u32, usize, NonterminalID);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub shifts: Vec<InitialShift>,
    pub reduces: Vec<InitialReduce>,
    pub nt_escapes: Vec<NtEscape>,
    pub nt_rereduces: Vec<NtRereduce>,
    pub all_nts: BTreeSet<NonterminalID>,
}

pub(crate) fn characterize_terminals(
    _table: &GLRTable,
    _grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, TerminalCharacterization> {
    todo!("template characterization is intentionally left as a placeholder in this cleanup pass")
}
