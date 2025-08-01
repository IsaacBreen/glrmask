use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::items::{LRMode, LR_MODE};
use crate::glr::table::{Stage7ShiftsAndReducesLookaheadValue, Table, StateID};
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Row, ShiftsAndReducesFull, TerminalID};
use crate::glr::table::{ShiftsAndReducesWithoutDefaultReduce, DefaultReduce};


/// Implements Pager's algorithm to eliminate unit productions from a Stage 6 parse table.
pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &mut Vec<Production>,
    start_production_id: usize,
) {
    todo!()
}
