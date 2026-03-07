//! Parser-side terminal characterization.
//!
//! The real characterization algorithm is intentionally deferred. This module
//! keeps only the minimal data shape and entrypoint needed for the current
//! compile-clean tree.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::{NonterminalID, TerminalID};

/// Shift: from parser state `from`, terminal T shifts to state `to`.
type InitialShift = (u32, u32);

/// Reduce: from parser state `from`, terminal T reduces rule with
/// `pop_count` states and LHS nonterminal `nt`.
type InitialReduce = (u32, usize, NonterminalID);

/// After reducing to nonterminal `nt_from`, if the revealed state is `revealed`,
/// then goto(revealed, nt_from) = `goto_state`, and from `goto_state`, terminal T
/// can shift to `shift_state`.
type NtEscape = (NonterminalID, u32, u32, u32);

/// After reducing to nonterminal `nt_from`, if the revealed state is `revealed`,
/// goto(revealed, nt_from) = `goto_state`, and from `goto_state`, terminal T
/// reduces again.
type NtRereduce = (NonterminalID, u32, usize, NonterminalID);

/// Stack pattern characterization for a single terminal.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub shifts: Vec<InitialShift>,
    pub reduces: Vec<InitialReduce>,
    pub nt_escapes: Vec<NtEscape>,
    pub nt_rereduces: Vec<NtRereduce>,
    /// All nonterminals involved in reduce cascades.
    pub all_nts: BTreeSet<NonterminalID>,
}

/// Characterize terminals: find all parser-stack patterns that allow them.
pub(crate) fn characterize_terminals(
    _table: &GLRTable,
    _grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, TerminalCharacterization> {
    todo!("template characterization is intentionally left as a placeholder in this cleanup pass")
}
