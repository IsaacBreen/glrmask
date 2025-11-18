use super::items::Item;
use crate::datastructures::hybrid_bitset::HybridBitset as TerminalBV;
use crate::glr::analyze::{
    create_unique_name_generator, inline_null_productions, inline_unit_productions,
    remove_productions_with_undefined_nonterminals, simplify_grammar, validate,
};
use crate::glr::automaton::{
    compute_closure, compute_first_sets_for_nonterminals, compute_follow_sets_for_nonterminals,
    compute_goto, compute_nullable_nonterminals, split_on_dot,
};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::display_productions;
use crate::json_serialization::{JSONConvertible, JSONNode};
pub use crate::types::TerminalID;
use bimap::BiBTreeMap;
use profiler_macro::time_it;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Display;
use crate::constraint::StateIDBV;
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
    kernel: BTreeSet<Item>,
    /// ID of the state reached by shifting over that symbol.
    goto_id: Option<StateID>,
}

type Stage1Row = BTreeMap<Option<Symbol>, Stage1Entry>;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage2Row {
    shifts: BTreeMap<Terminal, StateID>,
    gotos: BTreeMap<NonTerminal, StateID>,
    reduces: BTreeSet<Item>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage3Row {
    shifts: BTreeMap<Terminal, StateID>,
    gotos: BTreeMap<NonTerminal, StateID>,
    reduces: BTreeMap<Option<Terminal>, BTreeSet<Item>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage4Row {
    shifts: BTreeMap<Terminal, StateID>,
    gotos: BTreeMap<NonTerminal, StateID>,
    reduces: BTreeMap<Option<Terminal>, BTreeSet<ProductionID>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage5Row {
    shifts: BTreeMap<Terminal, StateID>,
    gotos: BTreeMap<NonTerminal, StateID>,
    reduces: BTreeMap<Terminal, BTreeSet<ProductionID>>,
    default_reduces: BTreeSet<ProductionID>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Stage6Row {
    pub(crate) shifts_and_reduces: BTreeMap<Terminal, Stage6ShiftsAndReduces>,
    pub(crate) gotos: BTreeMap<NonTerminal, StateID>,
    pub(crate) default_reduces: BTreeSet<ProductionID>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Stage6ShiftsAndReduces {
    pub(crate) shift: Option<StateID>,
    pub(crate) reduces: BTreeSet<ProductionID>,
}

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub enum Stage7ShiftsAndReducesLookaheadValue {
    Shift(StateID),
    Reduce {
        nonterminal_id: NonTerminalID,
        len: usize,
        production_ids: BTreeSet<ProductionID>,
    },
    Split {
        shift: Option<StateID>,
        reduces: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>,
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
                            .and_then(|n| BTreeSet::<ProductionID>::from_json(n))?;
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
                                    BTreeMap<NonTerminalID, BTreeSet<ProductionID>>,
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

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct SubstringGoto {
    pub accepting_sources: BTreeSet<StateID>,
    pub gotos: BTreeMap<StateID, BTreeSet<StateID>>,
}

impl JSONConvertible for SubstringGoto {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert(
            "accepting_sources".to_string(),
            self.accepting_sources.to_json(),
        );
        obj.insert("gotos".to_string(), self.gotos.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(SubstringGoto {
                accepting_sources: BTreeSet::<StateID>::from_json(
                    obj.remove("accepting_sources").ok_or_else(|| {
                        "Missing field accepting_sources for SubstringGoto".to_string()
                    })?,
                )?,
                gotos: BTreeMap::<StateID, BTreeSet<StateID>>::from_json(
                    obj.remove("gotos")
                        .ok_or_else(|| "Missing field gotos for SubstringGoto".to_string())?,
                )?,
            }),
            _ => Err("Expected JSONNode::Object for SubstringGoto".to_string()),
        }
    }
}

pub type ShiftsAndReducesFull = BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>;

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct Reduce {
    pub nonterminal_id: NonTerminalID,
    pub len: usize,
    pub production_ids: BTreeSet<ProductionID>,
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
                production_ids: BTreeSet::<ProductionID>::from_json(
                    obj.remove("production_ids").ok_or_else(|| {
                        "Missing field production_ids for Reduce".to_string()
                    })?,
                )?,
            }),
            _ => Err("Expected JSONNode::Object for Reduce".to_string()),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage7Row {
    pub shifts_and_reduces_full: ShiftsAndReducesFull,
    pub default_action: Option<Stage7ShiftsAndReducesLookaheadValue>,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub shifts_and_reduces_full: ShiftsAndReducesFull,
    pub default_action: Option<Stage7ShiftsAndReducesLookaheadValue>,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

impl JSONConvertible for Row {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert(
            "shifts_and_reduces_full".to_string(),
            self.shifts_and_reduces_full.to_json(),
        );
        obj.insert("default_action".to_string(), self.default_action.to_json());
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
                default_action: Option::<Stage7ShiftsAndReducesLookaheadValue>::from_json(
                    obj.remove("default_action")
                        .ok_or_else(|| "Missing field default_action for Row".to_string())?,
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
fn stage_1(productions: &[Production]) -> (Stage1Result, BiBTreeMap<BTreeSet<Item>, StateID>) {
    let start_production_id = 0;
    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = BTreeSet::from([initial_item]);

    // Map each non-terminal to the indices of its productions.
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<usize>> = BTreeMap::new();
    for (idx, p) in productions.iter().enumerate() {
        prods_by_lhs.entry(p.lhs.clone()).or_default().push(idx);
    }

    let mut item_set_map = BiBTreeMap::new();
    let mut next_state_id = 0;

    let mut worklist = VecDeque::new();
    let mut table: Stage1Table = BTreeMap::new();

    item_set_map.insert(initial_item_set.clone(), StateID(next_state_id));
    next_state_id += 1;
    worklist.push_back(initial_item_set);
    if EVERYTHING {
        let mut everything_item_set = BTreeSet::new();
        for (prod_idx, prod) in productions.iter().enumerate() {
            for dot_position in 0..=prod.rhs.len() {
                let item = Item {
                    production_id: prod_idx,
                    dot_position,
                };
                everything_item_set.insert(item);
            }
        }
        if !item_set_map.contains_left(&everything_item_set) {
            item_set_map.insert(everything_item_set.clone(), StateID(next_state_id));
            next_state_id += 1;
            worklist.push_back(everything_item_set);
        }
    }

    while let Some(item_set) = worklist.pop_front() {
        let state_id = *item_set_map.get_by_left(&item_set).unwrap();
        let closure = compute_closure(&item_set, &prods_by_lhs, productions);
        let splits = split_on_dot(&closure, productions);

        let mut row: Stage1Row = BTreeMap::new();
        for (symbol, items_in_split) in splits {
            let goto_id = if symbol.is_some() {
                let goto_set = compute_goto(&items_in_split, productions);
                if let Some(id) = item_set_map.get_by_left(&goto_set) {
                    Some(*id)
                } else {
                    let new_id = StateID(next_state_id);
                    next_state_id += 1;
                    item_set_map.insert(goto_set.clone(), new_id);
                    worklist.push_back(goto_set);
                    Some(new_id)
                }
            } else {
                None
            };
            row.insert(symbol, Stage1Entry { kernel: items_in_split, goto_id });
        }
        table.insert(state_id, row);
    }

    (table, item_set_map)
}

fn stage_2(stage_1_table: Stage1Table, productions: &[Production]) -> Stage2Result {
    let mut stage_2_table = BTreeMap::new();
    for (state_id, transitions) in stage_1_table {
        let mut shifts = BTreeMap::new();
        let mut gotos = BTreeMap::new();
        let mut reduces = BTreeSet::new();

        for (symbol_opt, Stage1Entry { kernel, goto_id }) in transitions {
            match (symbol_opt, goto_id) {
                (Some(Symbol::Terminal(t)), Some(id)) => {
                    shifts.insert(t, id);
                }
                (Some(Symbol::NonTerminal(nt)), Some(id)) => {
                    gotos.insert(nt, id);
                }
                (None, _) => {
                    for item in &kernel {
                        debug_assert_eq!(
                            item.dot_position,
                            productions[item.production_id].rhs.len(),
                            "Reduce item must have dot at end"
                        );
                        reduces.insert(*item);
                    }
                }
                _ => {}
            }
        }

        stage_2_table.insert(state_id, Stage2Row { shifts, gotos, reduces });
    }
    stage_2_table
}

fn stage_3(stage_2_table: Stage2Table, productions: &[Production]) -> Stage3Result {
    let mut stage_3_table = BTreeMap::new();

    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let first_sets = compute_first_sets_for_nonterminals(productions, &nullable_nonterminals);
    let follow_sets =
        compute_follow_sets_for_nonterminals(productions, &first_sets, &nullable_nonterminals);

    for (state_id, row) in stage_2_table {
        let mut reduces: BTreeMap<Option<Terminal>, BTreeSet<Item>> = BTreeMap::new();
        for item in &row.reduces {
            let lhs = &productions[item.production_id].lhs;
            if let Some(follows) = follow_sets.get(lhs) {
                for look in follows {
                    reduces.entry(look.clone()).or_default().insert(item.clone());
                }
            }
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
            let mut prod_ids = BTreeSet::new();
            for item in item_set_for_terminal {
                prod_ids.insert(ProductionID(item.production_id));
            }
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

fn stage_5(stage_4_table: Stage4Table) -> Stage5Result {
    let mut stage_5_table = BTreeMap::new();
    for (state_id, row) in stage_4_table {
        let Stage4Row {
            shifts,
            gotos,
            reduces,
        } = row;
        let mut new_reduces: BTreeMap<Terminal, BTreeSet<ProductionID>> = BTreeMap::new();
        let mut default_reduces = BTreeSet::new();

        for (opt_term, prod_ids) in reduces {
            if let Some(term) = opt_term {
                new_reduces.entry(term).or_default().extend(prod_ids);
            } else {
                default_reduces.extend(prod_ids);
            }
        }
        stage_5_table.insert(
            state_id,
            Stage5Row {
                shifts,
                gotos,
                reduces: new_reduces,
                default_reduces,
            },
        );
    }
    stage_5_table
}

fn stage_6(stage_5_table: Stage5Table) -> Stage6Result {
    let mut stage_6_table = BTreeMap::new();
    for (state_id, row) in stage_5_table {
        let mut shifts_and_reduces = BTreeMap::new();
        let specific_terminals: BTreeSet<_> =
            row.shifts.keys().chain(row.reduces.keys()).cloned().collect();

        for terminal in specific_terminals {
            let shift = row.shifts.get(&terminal).cloned();
            let reduces = row.reduces.get(&terminal).cloned().unwrap_or_default();
            if shift.is_some() || !reduces.is_empty() {
                shifts_and_reduces.insert(
                    terminal,
                    Stage6ShiftsAndReduces {
                        shift,
                        reduces,
                    },
                );
            }
        }
        stage_6_table.insert(
            state_id,
            Stage6Row {
                shifts_and_reduces,
                gotos: row.gotos,
                default_reduces: row.default_reduces,
            },
        );
    }
    stage_6_table
}

fn stage_7(
    stage_6_table: Stage6Table,
    item_set_map: &BiBTreeMap<BTreeSet<Item>, StateID>,
    productions: &[Production],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    non_terminal_map: &BiBTreeMap<NonTerminal, NonTerminalID>,
) -> (Stage7Table, StateID, StateID) {
    let start_production_id = 0;

    let mut stage_7_table = BTreeMap::new();
    for (state_id, row) in stage_6_table {
        let mut shifts_and_reduces_full: ShiftsAndReducesFull = BTreeMap::new();

        for (terminal, action) in &row.shifts_and_reduces {
            let terminal_id = *terminal_map
                .get_by_left(terminal)
                .expect_else(|| format!("Terminal {} not found in terminal map. Terminals: {:?}", terminal, terminal_map.left_values()));
            let maybe_shift: Option<StateID> = action.shift;

            let mut reduces: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>> =
                BTreeMap::new();
            for &production_id in &action.reduces {
                let production = &productions[production_id.0];
                let len = production.rhs.len();
                let nonterminal_id = *non_terminal_map.get_by_left(&production.lhs).unwrap();
                reduces
                    .entry(len)
                    .or_default()
                    .entry(nonterminal_id)
                    .or_default()
                    .insert(production_id);
            }

            if maybe_shift.is_none() && reduces.is_empty() {
                continue;
            }

            let mut final_action =
                Stage7ShiftsAndReducesLookaheadValue::Split { shift: maybe_shift, reduces };
            final_action.simplify();
            shifts_and_reduces_full.insert(terminal_id, final_action);
        }

        let mut default_reduces_grouped: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>> =
            BTreeMap::new();
        for &production_id in &row.default_reduces {
            let production = &productions[production_id.0];
            let len = production.rhs.len();
            let nonterminal_id = *non_terminal_map.get_by_left(&production.lhs).unwrap();
            default_reduces_grouped
                .entry(len)
                .or_default()
                .entry(nonterminal_id)
                .or_default()
                .insert(production_id);
        }

        let default_action = if default_reduces_grouped.is_empty() {
            None
        } else {
            let mut action = Stage7ShiftsAndReducesLookaheadValue::Split {
                shift: None,
                reduces: default_reduces_grouped,
            };
            action.simplify();
            Some(action)
        };

        let mut gotos = BTreeMap::new();
        for (nonterminal, next_state_id) in row.gotos {
            let non_terminal_id = *non_terminal_map
                .get_by_left(&nonterminal)
                .expect(&format!("Non-terminal '{}' not found in map", nonterminal));
            let goto = Goto {
                state_id: Some(next_state_id),
                accept: false,
            };
            gotos.insert(non_terminal_id, goto);
        }

        stage_7_table.insert(
            state_id,
            Stage7Row {
                shifts_and_reduces_full,
                default_action,
                gotos,
            },
        );
    }

    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = BTreeSet::from([initial_item]);
    let start_state_id = *item_set_map.get_by_left(&initial_item_set).unwrap();

    let start_non_terminal_id =
        *non_terminal_map.get_by_left(&productions[start_production_id].lhs).unwrap();
    stage_7_table
        .get_mut(&start_state_id)
        .unwrap()
        .gotos
        .entry(start_non_terminal_id)
        .or_default()
        .accept = true;

    let everything_state_id;
    if EVERYTHING {
        let mut everything_item_set = BTreeSet::new();
        for (prod_idx, prod) in productions.iter().enumerate() {
            for dot_position in 0..=prod.rhs.len() {
                let item = Item {
                    production_id: prod_idx,
                    dot_position,
                };
                everything_item_set.insert(item);
            }
        }
        everything_state_id = *item_set_map
            .get_by_left(&everything_item_set)
            .expect("Everything item set not found in state map");
        stage_7_table
            .get_mut(&everything_state_id)
            .unwrap()
            .gotos
            .entry(start_non_terminal_id)
            .or_default()
            .accept = true;
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
            default_action,
            gotos,
        } = row;
        stage_8_table.insert(
            state_id,
            Row {
                shifts_and_reduces_full,
                default_action,
                gotos,
            },
        );
    }
    stage_8_table
}

/// Pre-compute complex GOTO relations used by substring parsing.
pub fn stage_9(
    table: &Table,
    non_terminal_map: &BiBTreeMap<NonTerminal, NonTerminalID>,
) -> BTreeMap<NonTerminalID, SubstringGoto> {
    let mut substring_gotos = BTreeMap::new();
    let all_nt_ids: Vec<_> = non_terminal_map.right_values().copied().collect();

    for &nt_id in &all_nt_ids {
        let mut accepting_sources = BTreeSet::new();
        let mut gotos: BTreeMap<StateID, BTreeSet<StateID>> = BTreeMap::new();

        for (&source_state_id, row) in table {
            if let Some(goto) = row.gotos.get(&nt_id) {
                if goto.accept {
                    accepting_sources.insert(source_state_id);
                }
                if let Some(goto_state_id) = goto.state_id {
                    gotos.entry(goto_state_id).or_default().insert(source_state_id);
                }
            }
        }

        if !accepting_sources.is_empty() || !gotos.is_empty() {
            substring_gotos.insert(
                nt_id,
                SubstringGoto {
                    accepting_sources,
                    gotos,
                },
            );
        }
    }

    substring_gotos
}

/// Inverted GOTO index: (reduce non-terminal, goto state) -> bitvector of source states.
pub fn stage_10(table: &Table) -> BTreeMap<NonTerminalID, BTreeMap<StateID, StateIDBV>> {
    let mut reduce_goto_map: BTreeMap<NonTerminalID, BTreeMap<StateID, StateIDBV>> =
        BTreeMap::new();

    for (&source_state_id, row) in table {
        for (&nt_id, goto) in &row.gotos {
            if let Some(goto_state_id) = goto.state_id {
                reduce_goto_map
                    .entry(nt_id)
                    .or_default()
                    .entry(goto_state_id)
                    .or_default()
                    .insert(source_state_id.0);
            }
        }
    }
    reduce_goto_map
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

    crate::debug!(2, "Validating initial grammar");
    validate(productions).expect("Initial grammar validation failed");

    let _original_productions = productions.to_vec();
    let start_production_id = 0;

    crate::debug!(2, "Removing productions with undefined non-terminals");
    let mut productions =
        remove_productions_with_undefined_nonterminals(&productions, &[start_production_id]);

    let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut unqiue_name_generator = create_unique_name_generator(&nonterminals);

    crate::glr::analyze::resolve_direct_right_recursion(
        &mut productions,
        &mut unqiue_name_generator,
    );

    productions = inline_null_productions(&productions);
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

    crate::debug!(2, "Stage 1");
    let (stage_1_table, item_set_map) = stage_1(&productions);
    crate::debug!(2, "Stage 2");
    let stage_2_table = stage_2(stage_1_table, &productions);
    crate::debug!(2, "Stage 3");
    let stage_3_table = stage_3(stage_2_table, &productions);
    crate::debug!(2, "Stage 4");
    let stage_4_table = stage_4(stage_3_table);
    crate::debug!(2, "Stage 5");
    let stage_5_table = stage_5(stage_4_table);
    crate::debug!(2, "Stage 6");
    let stage_6_table = stage_6(stage_5_table);
    crate::debug!(2, "Stage 7");
    let (stage_7_table, start_state_id, everything_state_id) = stage_7(
        stage_6_table,
        &item_set_map,
        &productions,
        &terminal_map,
        &non_terminal_map,
    );
    crate::debug!(2, "Stage 8");
    let final_table = stage_8(stage_7_table);

    crate::debug!(2, "Stage 9: Precomputing substring gotos");
    let substring_gotos = stage_9(&final_table, &non_terminal_map);

    crate::debug!(2, "Stage 10: Precomputing reduce goto map");
    let reduce_goto_map = stage_10(&final_table);

    crate::debug!(2, "Done generating GLR parser");
    print_summary();
    print_summary_flat();

    GLRParser::new(
        final_table,
        productions,
        terminal_map,
        non_terminal_map,
        item_set_map,
        start_state_id,
        everything_state_id,
        actions,
        ignore_terminal_id,
        substring_gotos,
        reduce_goto_map,
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
