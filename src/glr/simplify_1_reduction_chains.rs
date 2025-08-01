use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::items::Item;
use crate::glr::table::{ProductionID, Stage6Table};

pub fn simplify_1_reduction_chains(
    stage_6_table: &mut Stage6Table,
    productions: &[Production],
    start_production_id: usize,
) {
    todo!()
}
