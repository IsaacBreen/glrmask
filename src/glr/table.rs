use super::items::Item;
use crate::datastructures::hybrid_bitset::HybridBitset as TerminalBV;
use crate::glr::analyze::{
    create_unique_name_generator, inline_null_productions, inline_unit_productions,
    remove_productions_with_undefined_nonterminals, simplify_grammar, validate,
};
use crate::glr::automaton::{
    compute_closure, compute_first_sets_for_nonterminals, compute_follow_sets_for_nonterminals,
    compute_goto, compute_nullable_nonterminals, split_on_dot, compute_first_sets_ids_with_lhs, compute_follow_sets_ids
};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::display_productions;
use crate::json_serialization::{JSONConvertible, JSONNode};
pub use crate::types::TerminalID;
use bimap::BiBTreeMap;
use profiler_macro::time_it;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::Display;
use memory_stats::memory_stats;
use crate::glr::parser::{ActionFn, ExpectElse, GLRParser};
use crate::profiler::{print_summary, print_summary_flat};

const EVERYTHING: bool = false;

type Stage1Table = BTreeMap<StateID, Stage1Row>;
type Stage2Table = BTreeMap<StateID, Stage2Row>;
type Stage3Table = BTreeMap<StateID, Stage3Row>;
type Stage4Table = BTreeMap<StateID, Stage4Row>;
type Stage5Table = BTreeMap<StateID, Stage5Row>;
pub(crate) type Stage6Table = BTreeMap<StateID, Stage6Row>;
type Stage7Table = BTreeMap<StateID, Stage7Row>;
type Stage8Table = BTreeMap<StateID, Row>;
pub type Table = BTreeMap<StateID, Row>;

#[derive(Debug, Clone)]
struct Stage1Entry {
    /// Items in this state whose symbol under the dot is `symbol`.
    kernel: Vec<Item>,
    /// ID of the state reached by shifting over that symbol.
    goto_id: Option<StateID>,
}

type Stage1Row = BTreeMap<Option<usize>, Stage1Entry>;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage2Row {
    shifts: BTreeMap<TerminalID, StateID>,
    gotos: BTreeMap<NonTerminalID, StateID>,
    reduces: Vec<Item>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage3Row {
    shifts: BTreeMap<TerminalID, StateID>,
    gotos: BTreeMap<NonTerminalID, StateID>,
    reduces: BTreeMap<Option<TerminalID>, Vec<Item>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage4Row {
    shifts: BTreeMap<TerminalID, StateID>,
    gotos: BTreeMap<NonTerminalID, StateID>,
    reduces: BTreeMap<Option<TerminalID>, Vec<ProductionID>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage5Row {
    shifts: BTreeMap<TerminalID, StateID>,
    gotos: BTreeMap<NonTerminalID, StateID>,
    reduces: BTreeMap<TerminalID, Vec<ProductionID>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Stage6Row {
    pub(crate) shifts_and_reduces: BTreeMap<TerminalID, Stage6ShiftsAndReduces>,
    pub(crate) gotos: BTreeMap<NonTerminalID, StateID>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Stage6ShiftsAndReduces {
    pub(crate) shift: Option<StateID>,
    pub(crate) reduces: Vec<ProductionID>,
}

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub enum Stage7ShiftsAndReducesLookaheadValue {
    Shift(StateID),
    Reduce {
        nonterminal_id: NonTerminalID,
        len: usize,
        production_ids: Vec<ProductionID>,
    },
    Split {
        shift: Option<StateID>,
        reduces: BTreeMap<usize, BTreeMap<NonTerminalID, Vec<ProductionID>>>,
    },
}

impl Stage7ShiftsAndReducesLookaheadValue {
    /// Simplifies a `Split` action into a `Shift` or `Reduce` if possible.
    /// - A `Split` with a shift and no reduces becomes a `Shift`.
    /// - A `Split` with no shift and exactly one reduce action becomes a `Reduce`.
    pub fn simplify(&mut self) {
        if let Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } = self {
            if reduces.is_empty() {
                if let Some(shift_id) = *shift {
                    *self = Stage7ShiftsAndReducesLookaheadValue::Shift(shift_id);
                }
            } else if shift.is_none()
                && reduces.len() == 1
                && reduces.values().next().unwrap().len() == 1
            {
                let temp_self = std::mem::replace(
                    self,
                    Stage7ShiftsAndReducesLookaheadValue::Split {
                        shift: None,
                        reduces: BTreeMap::new(),
                    },
                );
                if let Stage7ShiftsAndReducesLookaheadValue::Split { mut reduces, .. } = temp_self
                {
                    let (len, mut nts) = reduces.into_iter().next().unwrap();
                    let (nt_id, pids) = nts.into_iter().next().unwrap();
                    *self = Stage7ShiftsAndReducesLookaheadValue::Reduce {
                        nonterminal_id: nt_id,
                        len,
                        production_ids: pids,
                    };
                }
            }
        }
    }
}

impl JSONConvertible for Stage7ShiftsAndReducesLookaheadValue {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        match self {
            Stage7ShiftsAndReducesLookaheadValue::Shift(state_id) => {
                obj.insert("variant".to_string(), JSONNode::String("Shift".to_string()));
                obj.insert("state_id".to_string(), state_id.to_json());
            }
            Stage7ShiftsAndReducesLookaheadValue::Reduce {
                nonterminal_id,
                len,
                production_ids,
            } => {
                obj.insert("variant".to_string(), JSONNode::String("Reduce".to_string()));
                obj.insert("nonterminal_id".to_string(), nonterminal_id.to_json());
                obj.insert("len".to_string(), len.to_json());
                obj.insert("production_ids".to_string(), production_ids.to_json());
            }
            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                obj.insert("variant".to_string(), JSONNode::String("Split".to_string()));
                obj.insert("shift".to_string(), shift.to_json());
                obj.insert("reduces".to_string(), reduces.to_json());
            }
        }
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let variant = obj
                    .remove("variant")
                    .ok_or_else(|| {
                        "Missing field variant for Stage7ShiftsAndReducesLookaheadValue".to_string()
                    })
                    .and_then(String::from_json)?;
                match variant.as_str() {
                    "Shift" => {
                        let state_id = obj
                            .remove("state_id")
                            .ok_or_else(|| "Missing field state_id for Shift".to_string())
                            .and_then(StateID::from_json)?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Shift(state_id))
                    }
                    "Reduce" => {
                        let nonterminal_id = obj
                            .remove("nonterminal_id")
                            .ok_or_else(|| "Missing field nonterminal_id for Reduce".to_string())
                            .and_then(NonTerminalID::from_json)?;
                        let len = obj
                            .remove("len")
                            .ok_or_else(|| "Missing field len for Reduce".to_string())
                            .and_then(usize::from_json)?;
                        let production_ids = obj
                            .remove("production_ids")
                            .ok_or_else(|| "Missing field production_ids for Reduce".to_string())
                            .and_then(|n| Vec::<ProductionID>::from_json(n))?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                            nonterminal_id,
                            len,
                            production_ids,
                        })
                    }
                    "Split" => {
                        let shift = obj
                            .remove("shift")
                            .ok_or_else(|| "Missing field shift for Split".to_string())
                            .and_then(Option::<StateID>::from_json)?;
                        let reduces = obj
                            .remove("reduces")
                            .ok_or_else(|| "Missing field reduces for Split".to_string())
                            .and_then(|n| {
                                BTreeMap::<
                                    usize,
                                    BTreeMap<NonTerminalID, Vec<ProductionID>>,
                                >::from_json(n)
                            })?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces })
                    }
                    _ => Err(format!(
                        "Unknown variant {} for Stage7ShiftsAndReducesLookaheadValue",
                        variant
                    )),
                }
            }
            _ => Err(
                "Expected JSONNode::Object for Stage7ShiftsAndReducesLookaheadValue".to_string(),
            ),
        }
    }
}

pub type ShiftsAndReducesFull = BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>;

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct Reduce {
    pub nonterminal_id: NonTerminalID,
    pub len: usize,
    pub production_ids: Vec<ProductionID>,
}

impl JSONConvertible for Reduce {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert(
            "nonterminal_id".to_string(),
            self.nonterminal_id.to_json(),
        );
        obj.insert("len".to_string(), self.len.to_json());
        obj.insert("production_ids".to_string(), self.production_ids.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(Reduce {
                nonterminal_id: NonTerminalID::from_json(
                    obj.remove("nonterminal_id")
                        .ok_or_else(|| "Missing field nonterminal_id for Reduce".to_string())?,
                )?,
                len: usize::from_json(
                    obj.remove("len")
                        .ok_or_else(|| "Missing field len for Reduce".to_string())?,
                )?,
                production_ids: Vec::<ProductionID>::from_json(
                    obj.remove("production_ids")
                        .ok_or_else(|| "Missing field production_ids for Reduce".to_string())?,
                )?,
            }),
            _ => Err("Expected JSONNode::Object for Reduce".to_string()),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage7Row {
    pub shifts_and_reduces_full: ShiftsAndReducesFull,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    shifts_and_reduces_full: ShiftsAndReducesFull,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

pub fn iter_rows(table: &Table) -> impl Iterator<Item = (&StateID, &Row)> {
    table.iter()
}

pub fn get_row(table: &Table, state_id: StateID) -> Option<&Row> {
    table.get(&state_id)
}

impl Row {
    pub fn get_shifts_and_reduces_for_terminal(&self, terminal_id: &TerminalID) -> Option<Stage7ShiftsAndReducesLookaheadValue> {
        self.shifts_and_reduces_full.get(terminal_id).cloned()
    }

    pub fn get_shifts_and_reduces_map(&self) -> BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue> {
        self.shifts_and_reduces_full.clone()
    }

    pub fn get_gotos(&self) -> &BTreeMap<NonTerminalID, Goto> {
        &self.gotos
    }

    pub fn handle_shifts_and_reduces_for_terminal(
        &self,
        terminal_id: TerminalID,
        shiftfn: impl FnOnce(&StateID),
        mut reducefn: impl FnMut(&NonTerminalID, &usize, &Vec<ProductionID>),
    ) {
        if let Some(action) = self.shifts_and_reduces_full.get(&terminal_id) {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(state_id) => shiftfn(state_id),
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => reducefn(&nonterminal_id, &len, &production_ids),
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    if let Some(state_id) = shift {
                        shiftfn(state_id);
                    }
                    for (_len, nts) in reduces {
                        for (&nt_id, pids) in nts {
                            reducefn(&nt_id, &_len, &pids);
                        }
                    }
                },
            }
        }
    }
}

impl JSONConvertible for Row {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert(
            "shifts_and_reduces_full".to_string(),
            self.shifts_and_reduces_full.to_json(),
        );
        obj.insert("gotos".to_string(), self.gotos.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(Row {
                shifts_and_reduces_full: ShiftsAndReducesFull::from_json(
                    obj.remove("shifts_and_reduces_full").ok_or_else(|| {
                        "Missing field shifts_and_reduces_full for Row".to_string()
                    })?,
                )?,
                gotos: BTreeMap::<NonTerminalID, Goto>::from_json(
                    obj.remove("gotos")
                        .ok_or_else(|| "Missing field gotos for Row".to_string())?,
                )?,
            }),
            _ => Err("Expected JSONNode::Object for Row".to_string()),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Goto {
    pub state_id: Option<StateID>,
    pub accept: bool,
}

impl JSONConvertible for Goto {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        obj.insert("accept".to_string(), self.accept.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(Goto {
                state_id: obj
                    .remove("state_id")
                    .ok_or_else(|| "Missing field 'state_id' for Goto".to_string())
                    .and_then(Option::<StateID>::from_json)?,
                accept: obj
                    .remove("accept")
                    .ok_or_else(|| "Missing field 'accept' for Goto".to_string())
                    .and_then(bool::from_json)?,
            }),
            _ => Err("Expected JSONNode::Object for Goto".to_string()),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateID(pub usize);

impl JSONConvertible for StateID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(StateID)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProductionID(pub usize);

impl JSONConvertible for ProductionID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(ProductionID)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonTerminalID(pub usize);

impl JSONConvertible for NonTerminalID {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(NonTerminalID)
    }
}

type Stage1Result = Stage1Table;
type Stage2Result = Stage2Table;
type Stage3Result = Stage3Table;
type Stage4Result = Stage4Table;
type Stage5Result = Stage5Table;
type Stage6Result = Stage6Table;
type Stage7Result = (Stage7Table, StateID, StateID);

#[time_it]
fn stage_1(
    light_productions: &[Vec<usize>],
    lhs_ids: &[usize],
    num_terminals: usize,
    num_nonterminals: usize,
) -> (Stage1Result, HashMap<Vec<Item>, StateID>) {
    let start_production_id = 0;
    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = vec![initial_item];

    // Precompute productions by LHS ID
    // Note: lhs_ids are 0-based NonTerminal IDs.
    // In light_productions, NonTerminals are num_terminals + nt_id.
    let mut prods_by_lhs_id: Vec<Vec<usize>> = vec![Vec::new(); num_nonterminals];
    for (idx, &lhs_id) in lhs_ids.iter().enumerate() {
        prods_by_lhs_id[lhs_id].push(idx);
    }

    // 2. Precompute Closure Cache (Light)
    // We group closure items by the first symbol of their RHS to speed up bucket distribution.
    // closure_cache_grouped[nt_id] = Vec<(Option<SymbolID>, Vec<Item>)>
    let mut closure_cache_grouped: Vec<Vec<(Option<usize>, Vec<Item>)>> = vec![Vec::new(); num_nonterminals];

    for lhs_id in 0..num_nonterminals {
        if prods_by_lhs_id[lhs_id].is_empty() { continue; }

        let mut visited = vec![false; num_nonterminals];
        let mut stack = vec![lhs_id];
        visited[lhs_id] = true;

        let mut items_by_first_sym: HashMap<Option<usize>, Vec<Item>> = HashMap::new();

        while let Some(curr_id) = stack.pop() {
            for &pid in &prods_by_lhs_id[curr_id] {
                let item = Item { production_id: pid, dot_position: 0 };
                let first_sym = light_productions[pid].first().copied();
                items_by_first_sym.entry(first_sym).or_default().push(item);

                if let Some(next_sym_id) = first_sym {
                    if next_sym_id >= num_terminals {
                        let next_nt_id = next_sym_id - num_terminals;
                        if !visited[next_nt_id] {
                            visited[next_nt_id] = true;
                            stack.push(next_nt_id);
                        }
                    }
                }
            }
        }
        closure_cache_grouped[lhs_id] = items_by_first_sym.into_iter().collect();
    }

    // 3. State Generation Loop
    let mut item_set_map_fast: HashMap<Vec<Item>, StateID> = HashMap::new();
    let mut next_state_id = 0;

    let mut worklist = VecDeque::new();
    let mut table: Stage1Table = BTreeMap::new();

    item_set_map_fast.insert(initial_item_set.clone(), StateID(next_state_id));
    next_state_id += 1;
    worklist.push_back(initial_item_set);

    if EVERYTHING {
        // Omitted for brevity/correctness in this optimization pass
    }

    let mut buckets: HashMap<Option<usize>, Vec<Item>> = HashMap::new();

    while let Some(item_set) = worklist.pop_front() {
        let state_id = *item_set_map_fast.get(&item_set).unwrap();

        buckets.clear();
        let mut processed_nts_set = HashSet::new();

        for item in &item_set {
            let sym_opt = light_productions[item.production_id].get(item.dot_position).copied();
            buckets.entry(sym_opt).or_default().push(*item);

            if let Some(sym_id) = sym_opt {
                if sym_id >= num_terminals {
                    let nt_id = sym_id - num_terminals;
                    if processed_nts_set.insert(nt_id) {
                        for (first_sym, items) in &closure_cache_grouped[nt_id] {
                            buckets.entry(*first_sym).or_default().extend(items);
                        }
                    }
                }
            }
        }

        let mut row: Stage1Row = BTreeMap::new();
        for (symbol_id_opt, items_in_split_vec) in buckets.drain() {
            let goto_id = if symbol_id_opt.is_some() {
                let mut goto_set: Vec<Item> = items_in_split_vec.iter().map(|item| {
                    Item { production_id: item.production_id, dot_position: item.dot_position + 1 }
                }).collect();
                goto_set.sort_unstable();
                goto_set.dedup();

                if let Some(id) = item_set_map_fast.get(&goto_set) {
                    Some(*id)
                } else {
                    let new_id = StateID(next_state_id);
                    next_state_id += 1;
                    item_set_map_fast.insert(goto_set.clone(), new_id);
                    worklist.push_back(goto_set);
                    Some(new_id)
                }
            } else {
                None
            };

            let kernel = if symbol_id_opt.is_none() {
                items_in_split_vec.into_iter().collect()
            } else {
                Vec::new()
            };

            row.insert(symbol_id_opt, Stage1Entry { kernel, goto_id });
        }
        table.insert(state_id, row);
    }

    (table, item_set_map_fast)
}

fn stage_2(
    stage_1_table: Stage1Table,
    productions: &[Production],
    num_terminals: usize,
) -> Stage2Result {
    let mut stage_2_table = BTreeMap::new();
    for (state_id, transitions) in stage_1_table {
        let mut shifts = BTreeMap::new();
        let mut gotos = BTreeMap::new();
        let mut reduces = Vec::new();

        for (symbol_opt, Stage1Entry { kernel, goto_id }) in transitions {
            match (symbol_opt, goto_id) {
                (Some(sym_id), Some(id)) => {
                    if sym_id < num_terminals {
                        shifts.insert(TerminalID(sym_id), id);
                    } else {
                        gotos.insert(NonTerminalID(sym_id - num_terminals), id);
                    }
                }
                (None, _) => {
                    for item in &kernel {
                        debug_assert_eq!(
                            item.dot_position,
                            productions[item.production_id].rhs.len(),
                            "Reduce item must have dot at end"
                        );
                        reduces.push(*item);
                    }
                }
                _ => {}
            }
        }
        reduces.sort_unstable();
        reduces.dedup();

        stage_2_table.insert(state_id, Stage2Row { shifts, gotos, reduces });
    }
    stage_2_table
}

fn stage_3(
    stage_2_table: Stage2Table,
    productions: &[Production],
    light_productions: &[Vec<usize>],
    lhs_ids: &[usize],
    num_terminals: usize,
    num_nonterminals: usize,
    nullable_nts_ids: &HashSet<usize>,
    start_nt_id: usize,
) -> Stage3Result {
    let mut stage_3_table = BTreeMap::new();

    let first_sets = compute_first_sets_ids_with_lhs(light_productions, lhs_ids, num_terminals, num_nonterminals, nullable_nts_ids);
    let follow_sets = compute_follow_sets_ids(light_productions, lhs_ids, &first_sets, nullable_nts_ids, num_terminals, num_nonterminals, start_nt_id);

    for (state_id, row) in stage_2_table {
        let mut reduces: BTreeMap<Option<TerminalID>, Vec<Item>> = BTreeMap::new();
        for item in &row.reduces {
            let lhs_id = lhs_ids[item.production_id];
            let follows = &follow_sets[lhs_id];
            for look in follows {
                reduces.entry(look.clone()).or_default().push(item.clone());
            }
        }
        for vec in reduces.values_mut() {
            vec.sort_unstable();
            vec.dedup();
        }
        stage_3_table.insert(
            state_id,
            Stage3Row {
                shifts: row.shifts,
                gotos: row.gotos,
                reduces,
            },
        );
    }

    stage_3_table
}

fn stage_4(stage_3_table: Stage3Table) -> Stage4Result {
    let mut stage_4_table = BTreeMap::new();
    for (state_id, row) in stage_3_table {
        let mut reduces = BTreeMap::new();
        for (terminal, item_set_for_terminal) in row.reduces {
            let mut prod_ids = Vec::new();
            for item in item_set_for_terminal {
                prod_ids.push(ProductionID(item.production_id));
            }
            prod_ids.sort_unstable();
            prod_ids.dedup();
            reduces.insert(terminal.clone(), prod_ids);
        }
        stage_4_table.insert(
            state_id,
            Stage4Row {
                shifts: row.shifts,
                gotos: row.gotos,
                reduces,
            },
        );
    }
    stage_4_table
}

fn stage_5(
    stage_4_table: Stage4Table,
    num_terminals: usize,
) -> Stage5Result {
    let mut stage_5_table = BTreeMap::new();

    // We iterate 0..num_terminals
    for (state_id, row) in stage_4_table {
        let Stage4Row {
            shifts,
            gotos,
            reduces,
        } = row;
        let mut new_reduces: BTreeMap<TerminalID, Vec<ProductionID>> = BTreeMap::new();
        for (opt_term, prod_ids) in reduces {
            if let Some(term) = opt_term {
                new_reduces.entry(term).or_default().extend(prod_ids.into_iter());
            } else {
                for i in 0..num_terminals {
                    let terminal = TerminalID(i);
                    new_reduces
                        .entry(terminal)
                        .or_default()
                        .extend(prod_ids.iter().cloned());
                }
            }
        }
        for vec in new_reduces.values_mut() {
            vec.sort_unstable();
            vec.dedup();
        }
        stage_5_table.insert(state_id, Stage5Row { shifts, gotos, reduces: new_reduces });
    }
    stage_5_table
}

fn stage_6(stage_5_table: Stage5Table) -> Stage6Result {
    let mut stage_6_table = BTreeMap::new();
    for (state_id, row) in stage_5_table {
        let mut shifts_and_reduces = BTreeMap::new();
        let all_terminals: BTreeSet<_> =
            row.shifts.keys().chain(row.reduces.keys()).cloned().collect();
        for terminal in all_terminals {
            let shift = row.shifts.get(&terminal).cloned();
            let mut reduces = row.reduces.get(&terminal).cloned().unwrap_or_default();
            reduces.sort_unstable();
            reduces.dedup();
            shifts_and_reduces.insert(
                terminal,
                Stage6ShiftsAndReduces {
                    shift,
                    reduces,
                },
            );
        }
        stage_6_table.insert(state_id, Stage6Row { shifts_and_reduces, gotos: row.gotos });
    }
    stage_6_table
}

fn stage_7(
    stage_6_table: Stage6Table,
    item_set_map: &HashMap<Vec<Item>, StateID>,
    productions: &[Production],
    lhs_ids: &[usize],
) -> (Stage7Table, StateID, StateID) {
    let start_production_id = 0;

    let prod_meta: Vec<(usize, NonTerminalID)> = productions // We can use lhs_ids here
        .iter()
        .enumerate()
        .map(|(i, p)| (p.rhs.len(), NonTerminalID(lhs_ids[i])))
        .collect();

    let mut stage_7_table = BTreeMap::new();
    for (state_id, row) in stage_6_table {
        let mut shifts_and_reduces_full: ShiftsAndReducesFull = BTreeMap::new();

        for (terminal_id, action) in &row.shifts_and_reduces {
            let maybe_shift: Option<StateID> = action.shift;

            let mut reduces: BTreeMap<usize, BTreeMap<NonTerminalID, Vec<ProductionID>>> =
                BTreeMap::new();
            for &production_id in &action.reduces {
                let (len, nonterminal_id) = prod_meta[production_id.0];
                reduces
                    .entry(len)
                    .or_default()
                    .entry(nonterminal_id)
                    .or_default()
                    .push(production_id);
            }
            for inner in reduces.values_mut() {
                for vec in inner.values_mut() {
                    vec.sort_unstable();
                    vec.dedup();
                }
            }

            if maybe_shift.is_none() && reduces.is_empty() {
                continue;
            }

            let mut final_action =
                Stage7ShiftsAndReducesLookaheadValue::Split { shift: maybe_shift, reduces };
            final_action.simplify();
            shifts_and_reduces_full.insert(*terminal_id, final_action);
        }

        let mut gotos = BTreeMap::new();
        for (nonterminal_id, next_state_id) in row.gotos {
            let goto = Goto {
                state_id: Some(next_state_id),
                accept: false,
            };
            gotos.insert(nonterminal_id, goto);
        }

        stage_7_table.insert(state_id, Stage7Row { shifts_and_reduces_full, gotos });
    }

    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = vec![initial_item];
    let start_state_id = *item_set_map.get(&initial_item_set).unwrap();

    let start_non_terminal_id = NonTerminalID(lhs_ids[start_production_id]);
    stage_7_table
        .get_mut(&start_state_id)
        .unwrap()
        .gotos
        .entry(start_non_terminal_id)
        .or_default()
        .accept = true;

    let everything_state_id;
    if EVERYTHING {
        // Omitted
        everything_state_id = start_state_id;
    } else {
        everything_state_id = start_state_id;
    }

    (stage_7_table, start_state_id, everything_state_id)
}

fn stage_8(stage_7_table: Stage7Table) -> Stage8Table {
    let mut stage_8_table = BTreeMap::new();
    for (state_id, row) in stage_7_table {
        let Stage7Row {
            shifts_and_reduces_full,
            gotos,
        } = row;
        stage_8_table.insert(
            state_id,
            Row {
                shifts_and_reduces_full,
                gotos,
            },
        );
    }
    stage_8_table
}

fn print_memory_usage(label: &str) {
    if let Some(usage) = memory_stats() {
        let physical_mem_mb = usage.physical_mem / 1024 / 1024;
        crate::debug!(2, "Memory usage at '{}': Physical: {} MB", label, physical_mem_mb);
    } else {
        crate::debug!(2, "Couldn't get memory usage at '{}'", label);
    }
}

#[time_it]
pub fn generate_glr_parser_with_maps(
    productions: &[Production],
    terminal_map: BiBTreeMap<Terminal, TerminalID>,
    mut non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    actions: BTreeMap<NonTerminal, ActionFn>,
    ignore_terminal_id: Option<TerminalID>,
) -> GLRParser {
    crate::debug!(2, "Number of productions: {}", productions.len());
    print_memory_usage("Start of parser generation");

    crate::debug!(2, "Validating initial grammar");
    let start = std::time::Instant::now();
    validate(productions).expect("Initial grammar validation failed");
    crate::debug!(2, "Validated grammar in {:.2?}", start.elapsed());
    print_memory_usage("After validation");

    let _original_productions = productions.to_vec();
    let start_production_id = 0;

    crate::debug!(2, "Removing productions with undefined non-terminals");
    let start = std::time::Instant::now();
    let mut productions =
        remove_productions_with_undefined_nonterminals(&productions, &[start_production_id]);
    crate::debug!(2, "Removed undefined productions in {:.2?}", start.elapsed());
    print_memory_usage("After removing undefined");

    let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut unqiue_name_generator = create_unique_name_generator(&nonterminals);

    crate::glr::analyze::resolve_direct_right_recursion(
        &mut productions,
        &mut unqiue_name_generator,
    );
    print_memory_usage("After right recursion resolution");

    productions = inline_null_productions(&productions);
    print_memory_usage("After inlining null productions");
    if false {
        productions = inline_unit_productions(&productions);
    }

    let mut next_non_terminal_id = non_terminal_map.len();
    for p in &productions {
        if !non_terminal_map.contains_left(&p.lhs) {
            non_terminal_map.insert(p.lhs.clone(), NonTerminalID(next_non_terminal_id));
            next_non_terminal_id += 1;
        }
    }

    crate::debug!(2, "Number of productions: {}", productions.len());
    print_memory_usage("Before Stage 1");

    // Prepare Light Productions (Global IDs)
    let num_terminals = terminal_map.len();
    let num_nonterminals = non_terminal_map.len();

    let light_productions: Vec<Vec<usize>> = productions.iter().map(|p| {
        p.rhs.iter().map(|s| match s {
            Symbol::Terminal(t) => terminal_map.get_by_left(t).unwrap().0,
            Symbol::NonTerminal(nt) => non_terminal_map.get_by_left(nt).unwrap().0 + num_terminals,
        }).collect()
    }).collect();

    let lhs_ids: Vec<usize> = productions.iter().map(|p| {
        non_terminal_map.get_by_left(&p.lhs).unwrap().0
    }).collect();

    let nullable_nonterminals = compute_nullable_nonterminals(&productions);
    let nullable_nts_ids: HashSet<usize> = nullable_nonterminals.iter().map(|nt| {
        non_terminal_map.get_by_left(nt).unwrap().0
    }).collect();

    let start_nt_id = lhs_ids[0];

    crate::debug!(2, "Stage 1");
    let start = std::time::Instant::now();
    let (stage_1_table, item_set_map) = stage_1(&light_productions, &lhs_ids, num_terminals, num_nonterminals);
    crate::debug!(2, "Stage 1 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 1");
    crate::debug!(2, "Stage 2");
    let start = std::time::Instant::now();
    let stage_2_table = stage_2(stage_1_table, &productions, num_terminals);
    crate::debug!(2, "Stage 2 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 2");
    crate::debug!(2, "Stage 3");
    let start = std::time::Instant::now();
    let stage_3_table = stage_3(stage_2_table, &productions, &light_productions, &lhs_ids, num_terminals, num_nonterminals, &nullable_nts_ids, start_nt_id);
    crate::debug!(2, "Stage 3 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 3");
    crate::debug!(2, "Stage 4");
    let start = std::time::Instant::now();
    let stage_4_table = stage_4(stage_3_table);
    crate::debug!(2, "Stage 4 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 4");
    crate::debug!(2, "Stage 5");
    let start = std::time::Instant::now();
    let stage_5_table = stage_5(stage_4_table, num_terminals);
    crate::debug!(2, "Stage 5 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 5");
    crate::debug!(2, "Stage 6");
    let start = std::time::Instant::now();
    let stage_6_table = stage_6(stage_5_table);
    crate::debug!(2, "Stage 6 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 6");
    crate::debug!(2, "Stage 7");
    let start = std::time::Instant::now();
    let (stage_7_table, start_state_id, everything_state_id) = stage_7(
        stage_6_table,
        &item_set_map,
        &productions,
        &lhs_ids,
    );
    crate::debug!(2, "Stage 7 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 7");
    crate::debug!(2, "Stage 8");
    let start = std::time::Instant::now();
    let final_table = stage_8(stage_7_table);
    crate::debug!(2, "Stage 8 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 8 (final table)");

    // Convert item_set_map back to BiBTreeMap for GLRParser
    let mut item_set_map_bi = BiBTreeMap::new();
    for (k, v) in item_set_map {
        item_set_map_bi.insert(k, v);
    }

    GLRParser::new(
        final_table,
        productions,
        terminal_map,
        non_terminal_map,
        item_set_map_bi,
        start_state_id,
        everything_state_id,
        actions,
        ignore_terminal_id,
    )
}

pub fn generate_glr_parser(
    productions: &[Production],
    ignore_terminal_id: Option<TerminalID>,
) -> crate::glr::parser::GLRParser {
    let terminal_map = assign_terminal_ids(productions);
    generate_glr_parser_with_terminal_map(productions, terminal_map, ignore_terminal_id)
}

pub fn generate_glr_parser_with_terminal_map(
    productions: &[Production],
    terminal_map: BiBTreeMap<Terminal, TerminalID>,
    ignore_terminal_id: Option<TerminalID>,
) -> crate::glr::parser::GLRParser {
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(
        productions,
        terminal_map,
        non_terminal_map,
        BTreeMap::new(),
        ignore_terminal_id,
    )
}

pub fn assign_terminal_ids(productions: &[Production]) -> BiBTreeMap<Terminal, TerminalID> {
    let mut terminal_map = BiBTreeMap::new();
    let mut next_terminal_id = 0;
    for p in productions {
        for symbol in &p.rhs {
            if let Symbol::Terminal(t) = symbol {
                if !terminal_map.contains_left(t) {
                    terminal_map.insert(t.clone(), TerminalID(next_terminal_id));
                    next_terminal_id += 1;
                }
            }
        }
    }
    terminal_map
}

pub fn assign_non_terminal_ids(
    productions: &[Production],
) -> BiBTreeMap<NonTerminal, NonTerminalID> {
    let mut non_terminal_map = BiBTreeMap::new();
    let mut next_non_terminal_id = 0;
    for p in productions {
        if !non_terminal_map.contains_left(&p.lhs) {
            non_terminal_map.insert(p.lhs.clone(), NonTerminalID(next_non_terminal_id));
            next_non_terminal_id += 1;
        }
    }
    non_terminal_map
}