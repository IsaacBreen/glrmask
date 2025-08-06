use super::items::{Item, LRMode, LR_MODE};
use crate::glr::automaton::{compute_closure, compute_first_sets_for_nonterminals, compute_follow_sets_for_nonterminals, compute_goto, compute_nullable_nonterminals, split_on_dot};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Display;
use crate::glr::analyze::{create_unique_name_generator, remove_productions_with_undefined_nonterminals, simplify_grammar, validate, inline_unit_productions, inline_null_productions};
pub use crate::types::{TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use crate::interface::display_productions;
// Added for derive macro pattern


type Stage1Table = BTreeMap<BTreeSet<Item>, Stage1Row>;
type Stage2Table = BTreeMap<BTreeSet<Item>, Stage2Row>;
type Stage3Table = BTreeMap<BTreeSet<Item>, Stage3Row>;
type Stage4Table = BTreeMap<BTreeSet<Item>, Stage4Row>;
type Stage5Table = BTreeMap<BTreeSet<Item>, Stage5Row>;
pub(crate) type Stage6Table = BTreeMap<BTreeSet<Item>, Stage6Row>;
type Stage7Table = BTreeMap<StateID, Stage7Row>;
type Stage8Table = BTreeMap<StateID, Row>;
pub type Table = BTreeMap<StateID, Row>;


type Stage1Row = BTreeMap<Option<Symbol>, BTreeSet<Item>>;
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage2Row {
    shifts: BTreeMap<Terminal, BTreeSet<Item>>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
    reduces: BTreeSet<Item>,
}
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage3Row {
    shifts: BTreeMap<Terminal, BTreeSet<Item>>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
    reduces: BTreeMap<Option<Terminal>, BTreeSet<Item>>,
}
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage4Row {
    shifts: BTreeMap<Terminal, BTreeSet<Item>>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
    reduces: BTreeMap<Option<Terminal>, BTreeSet<ProductionID>>,
}
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage5Row {
    shifts: BTreeMap<Terminal, BTreeSet<Item>>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
    reduces: BTreeMap<Terminal, BTreeSet<ProductionID>>,
}
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Stage6Row {
    pub(crate) shifts_and_reduces: BTreeMap<Terminal, Stage6ShiftsAndReduces>,
    pub(crate) gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Stage6ShiftsAndReduces {
    pub(crate) shift: Option<BTreeSet<Item>>,
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
            } else if shift.is_none() && reduces.len() == 1 && reduces.values().next().unwrap().len() == 1 {
                // No shift, and only one kind of reduce (one len, one nt).
                let temp_self = std::mem::replace(self, Stage7ShiftsAndReducesLookaheadValue::Split { shift: None, reduces: BTreeMap::new() }); // dummy
                if let Stage7ShiftsAndReducesLookaheadValue::Split { mut reduces, .. } = temp_self {
                    let (len, mut nts) = reduces.into_iter().next().unwrap();
                    let (nt_id, pids) = nts.into_iter().next().unwrap();
                    *self = Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nt_id, len, production_ids: pids };
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
            Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
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
                let variant = obj.remove("variant").ok_or_else(|| "Missing field variant for Stage7ShiftsAndReducesLookaheadValue".to_string())
                                   .and_then(String::from_json)?;
                match variant.as_str() {
                    "Shift" => {
                        let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for Shift".to_string())
                                          .and_then(StateID::from_json)?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Shift(state_id))
                    }
                    "Reduce" => {
                        let nonterminal_id = obj.remove("nonterminal_id").ok_or_else(|| "Missing field nonterminal_id for Reduce".to_string())
                                                .and_then(NonTerminalID::from_json)?;
                        let len = obj.remove("len").ok_or_else(|| "Missing field len for Reduce".to_string())
                                     .and_then(usize::from_json)?;
                        let production_ids = obj.remove("production_ids").ok_or_else(|| "Missing field production_ids for Reduce".to_string())
                                                .and_then(|n| BTreeSet::<ProductionID>::from_json(n))?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids })
                    }
                    "Split" => {
                        let shift = obj.remove("shift").ok_or_else(|| "Missing field shift for Split".to_string())
                                       .and_then(Option::<StateID>::from_json)?;
                        let reduces = obj.remove("reduces").ok_or_else(|| "Missing field reduces for Split".to_string())
                                         .and_then(|n| BTreeMap::<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>::from_json(n))?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces })
                    }
                    _ => Err(format!("Unknown variant {} for Stage7ShiftsAndReducesLookaheadValue", variant)),
                }
            }
            _ => Err("Expected JSONNode::Object for Stage7ShiftsAndReducesLookaheadValue".to_string()),
        }
    }
}

pub type ShiftsAndReducesWithoutDefaultReduce = BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>;
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
        obj.insert("nonterminal_id".to_string(), self.nonterminal_id.to_json());
        obj.insert("len".to_string(), self.len.to_json());
        obj.insert("production_ids".to_string(), self.production_ids.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(Reduce {
                nonterminal_id: NonTerminalID::from_json(obj.remove("nonterminal_id").ok_or_else(|| "Missing field nonterminal_id for Reduce".to_string())?)?,
                len: usize::from_json(obj.remove("len").ok_or_else(|| "Missing field len for Reduce".to_string())?)?,
                production_ids: BTreeSet::<ProductionID>::from_json(obj.remove("production_ids").ok_or_else(|| "Missing field production_ids for Reduce".to_string())?)?,
            }),
            _ => Err("Expected JSONNode::Object for Reduce".to_string()),
        }
    }
}

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct DefaultReduce {
    pub clone_and_merge: bool, // Indicates that there are phase 1 actions to be performed here.
    pub reduce: Option<Reduce>,
}

impl JSONConvertible for DefaultReduce {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clone_and_merge".to_string(), self.clone_and_merge.to_json());
        obj.insert("reduce".to_string(), self.reduce.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(DefaultReduce {
                clone_and_merge: bool::from_json(obj.remove("clone_and_merge").ok_or_else(|| "Missing field clone_and_merge for DefaultReduce".to_string())?)?,
                reduce: Option::<Reduce>::from_json(obj.remove("reduce").ok_or_else(|| "Missing field reduce for DefaultReduce".to_string())?)?,
            }),
            _ => Err("Expected JSONNode::Object for DefaultReduce".to_string()),
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
    pub shifts_and_reduces_without_default_reduce: ShiftsAndReducesWithoutDefaultReduce,
    pub shifts_and_reduces_full: ShiftsAndReducesFull,
    pub default_reduce: DefaultReduce,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

// Manual impl for Row (could be derived)
impl JSONConvertible for Row {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("shifts_and_reduces_without_default_reduce".to_string(), self.shifts_and_reduces_without_default_reduce.to_json());
        obj.insert("shifts_and_reduces_full".to_string(), self.shifts_and_reduces_full.to_json());
        obj.insert("default_reduce".to_string(), self.default_reduce.to_json());
        obj.insert("gotos".to_string(), self.gotos.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(Row {
                shifts_and_reduces_without_default_reduce: ShiftsAndReducesWithoutDefaultReduce::from_json(obj.remove("shifts_and_reduces_without_default_reduce").ok_or_else(|| "Missing field shifts_and_reduces_without_default_reduce for Row".to_string())?)?,
                shifts_and_reduces_full: ShiftsAndReducesFull::from_json(obj.remove("shifts_and_reduces_full").ok_or_else(|| "Missing field shifts_and_reduces_full for Row".to_string())?)?,
                default_reduce: DefaultReduce::from_json(obj.remove("default_reduce").ok_or_else(|| "Missing field default_reduce for Row".to_string())?)?,
                gotos: BTreeMap::<NonTerminalID, Goto>::from_json(obj.remove("gotos").ok_or_else(|| "Missing field gotos for Row".to_string())?)?,
            }),
            _ => Err("Expected JSONNode::Object for Row".to_string()),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateID(pub usize);

impl JSONConvertible for StateID {
    fn to_json(&self) -> JSONNode { self.0.to_json() }
    fn from_json(node: JSONNode) -> Result<Self, String> { usize::from_json(node).map(StateID) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProductionID(pub usize);

impl JSONConvertible for ProductionID {
    fn to_json(&self) -> JSONNode { self.0.to_json() }
    fn from_json(node: JSONNode) -> Result<Self, String> { usize::from_json(node).map(ProductionID) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonTerminalID(pub usize);

impl JSONConvertible for NonTerminalID {
    fn to_json(&self) -> JSONNode { self.0.to_json() }
    fn from_json(node: JSONNode) -> Result<Self, String> { usize::from_json(node).map(NonTerminalID) }
}


type Stage1Result = Stage1Table;
type Stage2Result = Stage2Table;
type Stage3Result = Stage3Table;
type Stage4Result = Stage4Table;
type Stage5Result = Stage5Table;
type Stage6Result = Stage6Table;
type Stage7Result = (
    Stage7Table,
    BiBTreeMap<BTreeSet<Item>, StateID>,
    StateID,
);

fn stage_1(productions: &[Production]) -> Stage1Result {
    let start_production_id = 0;
    let initial_item = Item {
        production: productions[start_production_id].clone(),
        dot_position: 0,
        lookahead: None,
    };
    let initial_item_set = BTreeSet::from([initial_item.clone()]); // Clone initial_item here

    let first_sets = compute_first_sets_for_nonterminals(productions);
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let follow_sets = compute_follow_sets_for_nonterminals(productions, &first_sets, &nullable_nonterminals);

    let mut worklist = VecDeque::from([initial_item_set.clone()]); // Use initial_item_set here

    let mut transitions: BTreeMap<BTreeSet<Item>, BTreeMap<Option<Symbol>, BTreeSet<Item>>> = BTreeMap::new();

    while let Some(item_set) = worklist.pop_front() {
        let closure = compute_closure(&item_set, productions, &first_sets, &nullable_nonterminals, &follow_sets);
        let splits = split_on_dot(&closure);
        let mut row = BTreeMap::new();

        for (symbol, item_set) in &splits {
            row.insert(symbol.clone(), item_set.clone());
            if symbol.is_some() {
                let goto_set = compute_goto(item_set);
                if transitions.contains_key(&goto_set) {
                    worklist.push_back(goto_set);
                }
            }
        }

        transitions.insert(item_set.clone(), row);
    }

    transitions
}

fn stage_2(stage_1_table: Stage1Table, productions: &[Production]) -> Stage2Result {
    let mut stage_2_table = BTreeMap::new();
    for (item_set, transitions) in stage_1_table {
        let mut shifts = BTreeMap::new();
        let mut gotos = BTreeMap::new();
        let mut reduces = BTreeSet::new();

        for (symbol_opt, item_set) in &transitions {
            match symbol_opt {
                Some(Symbol::Terminal(t)) => {
                    shifts.insert(t.clone(), compute_goto(item_set));
                }
                Some(Symbol::NonTerminal(nt)) => {
                    gotos.insert(nt.clone(), compute_goto(item_set));
                }
                None => {
                    for item in item_set {
                        assert_eq!(item.dot_position, item.production.rhs.len());
                        reduces.insert(item.clone());
                    }
                }
            }
        }

        stage_2_table.insert(
            item_set,
            Stage2Row {
                shifts,
                gotos,
                reduces,
            },
        );
    }
    stage_2_table
}

fn stage_3(stage_2_table: Stage2Table, productions: &[Production]) -> Stage3Result {
    let mut stage_3_table = BTreeMap::new();

    for (item_set, row) in stage_2_table {
        let mut reduces: BTreeMap<Option<Terminal>, BTreeSet<Item>> = BTreeMap::new();

        for item in &row.reduces {
            reduces
                .entry(item.lookahead.clone())
                .or_default()
                .insert(item.clone());
        }

        stage_3_table.insert(
            item_set,
            Stage3Row {
                shifts: row.shifts,
                gotos: row.gotos,
                reduces,
            },
        );
    }

    stage_3_table
}

fn stage_4(stage_3_table: Stage3Table, productions: &[Production]) -> Stage4Result {
    let production_ids: BTreeMap<Production, ProductionID> = productions
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), ProductionID(i)))
        .collect();

    let mut stage_4_table = BTreeMap::new();

    for (item_set, row) in stage_3_table {
        let mut reduces = BTreeMap::new();

        for (terminal, item_set) in row.reduces {
            let mut prod_ids = BTreeSet::new();
            for item in item_set {
                let prod_id = production_ids.get(&item.production).unwrap();
                prod_ids.insert(*prod_id);
            }
            reduces.insert(terminal.clone(), prod_ids);
        }

        stage_4_table.insert(
            item_set,
            Stage4Row {
                shifts: row.shifts,
                gotos: row.gotos,
                reduces,
            },
        );
    }

    stage_4_table
}

fn stage_5(stage_4_table: Stage4Table, productions: &[Production], terminal_map: &BiBTreeMap<Terminal, TerminalID>) -> Stage5Result {
    // Stage 5 turns
    //     reduces: BTreeMap<Option<Terminal>, BTreeSet<ProductionID>>,
    // into
    //     reduces: BTreeMap<Terminal, BTreeSet<ProductionID>>,
    // ie it removes the None entries, which represent EOF.
    // It does this by copying the values for None entries across to all other possible terminals (determined by the terminal_map),
    // merging with any existing production ID sets in the reduces map.
    let mut stage_5_table = BTreeMap::new();
    let all_terminals: BTreeSet<Terminal> = terminal_map.left_values().cloned().collect();

    for (item_set, row) in stage_4_table {
        let Stage4Row { shifts, gotos, reduces } = row;

        // 2. Start building the new reduces map keyed by concrete terminals.
        let mut new_reduces: BTreeMap<Terminal, BTreeSet<ProductionID>> = BTreeMap::new();


        // 2a. Copy over entries that already have a concrete terminal key.
        for (opt_term, prod_ids) in reduces {
            if let Some(term) = opt_term {
                new_reduces
                    .entry(term)
                    .or_default()
                    .extend(prod_ids.into_iter());
            } else {
                // 2b. For None entries, copy the production IDs to all terminals.
                for terminal in &all_terminals {
                    new_reduces
                        .entry(terminal.clone())
                        .or_default()
                        .extend(prod_ids.iter().cloned());
                }
            }
        }

        stage_5_table.insert(
            item_set,
            Stage5Row {
                shifts,
                gotos,
                reduces: new_reduces,
            },
        );
    }

    stage_5_table
}

fn stage_6(stage_5_table: Stage5Table) -> Stage6Result {
    let mut stage_6_table = BTreeMap::new();

    for (item_set, row) in stage_5_table {
        let mut shifts_and_reduces = BTreeMap::new();

        // Get all terminals that appear in either shifts or reduces
        let all_terminals: BTreeSet<_> = row.shifts.keys()
            .chain(row.reduces.keys())
            .cloned()
            .collect();

        for terminal in all_terminals {
            let shift = row.shifts.get(&terminal).cloned();
            let reduces = row.reduces.get(&terminal).cloned().unwrap_or_default();

            let action = Stage6ShiftsAndReduces {
                shift,
                reduces,
            };
            shifts_and_reduces.insert(terminal, action);
        }

        stage_6_table.insert(
            item_set,
            Stage6Row {
                shifts_and_reduces,
                gotos: row.gotos,
            },
        );
    }

    stage_6_table
}

fn stage_7(stage_6_table: Stage6Table, productions: &[Production], terminal_map: &BiBTreeMap<Terminal, TerminalID>, non_terminal_map: &BiBTreeMap<NonTerminal, NonTerminalID>) -> Stage7Result {
    let start_production_id = 0;
    let mut item_set_map = BiBTreeMap::new();
    let mut next_state_id = 0;

    for item_set in stage_6_table.keys() {
        item_set_map.insert(item_set.clone(), StateID(next_state_id));
        next_state_id += 1;
    }

    let mut stage_7_table = BTreeMap::new();

    for (item_set, row) in stage_6_table {
        let state_id = *item_set_map.get_by_left(&item_set).unwrap();

        // Phase 2 map contains all possible actions for a state.
        let mut shifts_and_reduces_full: ShiftsAndReducesFull = BTreeMap::new();

        for (terminal, action) in &row.shifts_and_reduces {
            let terminal_id = *terminal_map.get_by_left(terminal).unwrap();

            // 1. Get the potential shift action.
            let maybe_shift: Option<StateID> = action.shift.as_ref().map(|shift_item_set| {
                *item_set_map.get_by_left(shift_item_set).unwrap()
            });

            // 2. Group all reduce actions.
            let mut reduces: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>> = BTreeMap::new();
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

            // 3. Create a combined action and simplify it.
            if maybe_shift.is_none() && reduces.is_empty() {
                // This should not happen because we iterate over terminals that have actions from stage 6.
                // panic!("Action without shift or reduce for terminal {:?}", terminal);
                continue;
            }

            let mut final_action = Stage7ShiftsAndReducesLookaheadValue::Split {
                shift: maybe_shift,
                reduces,
            };
            final_action.simplify();
            shifts_and_reduces_full.insert(terminal_id, final_action);
        }

        let mut gotos = BTreeMap::new();
        for (nonterminal, next_item_set) in row.gotos {
            let non_terminal_id = *non_terminal_map.get_by_left(&nonterminal).expect(&format!("Non-terminal '{}' not found in map", nonterminal));
            let goto = Goto {
                state_id: Some(*item_set_map.get_by_left(&next_item_set).unwrap()),
                accept: false,
            };
            gotos.insert(non_terminal_id, goto);
        }

        stage_7_table.insert(state_id, Stage7Row {
            shifts_and_reduces_full,
            gotos,
        });
    }

    let initial_item = Item {
        production: productions[start_production_id].clone(),
        dot_position: 0,
        lookahead: None,
    };
    let initial_item_set = BTreeSet::from([initial_item]);
    let start_state_id = *item_set_map.get_by_left(&initial_item_set).unwrap();

    // Goto for initial production
    let start_non_terminal_id = *non_terminal_map.get_by_left(&productions[start_production_id].lhs).unwrap();
    stage_7_table.get_mut(&start_state_id).unwrap().gotos.entry(start_non_terminal_id).or_default().accept = true;

    (stage_7_table, item_set_map, start_state_id)
}

fn stage_8(stage_7_table: Stage7Table) -> Stage8Table {
    let mut stage_8_table = BTreeMap::new();

    for (state_id, row) in stage_7_table {
        let Stage7Row { shifts_and_reduces_full, gotos } = row;

        // --- Promotion Logic ---
        let mut reduce_counts: BTreeMap<(NonTerminalID, usize), (usize, BTreeSet<ProductionID>)> = BTreeMap::new();
        for action in shifts_and_reduces_full.values() {
            let mut process_reduces = |reduces: &BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>| {
                for (&len, nts) in reduces {
                    for (&nt_id, pids) in nts {
                        let entry = reduce_counts.entry((nt_id, len)).or_default();
                        entry.0 += 1; // Count how many terminals trigger this kind of reduce
                        entry.1.extend(pids.iter().cloned());
                    }
                }
            };
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                    let entry = reduce_counts.entry((*nonterminal_id, *len)).or_default();
                    entry.0 += 1;
                    entry.1.extend(production_ids.iter().cloned());
                },
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => process_reduces(reduces),
                _ => {}
            }
        }

        let promoted_reduce_key = reduce_counts.iter().max_by_key(|(_, (count, _))| *count).map(|(key, _)| *key);

        let (shifts_and_reduces_without_default_reduce, default_reduce) = if let Some((nonterminal_id, len)) = promoted_reduce_key {
            let (_, production_ids) = reduce_counts.remove(&(nonterminal_id, len)).unwrap();
            
            let shifts_and_reduces_without_default = shifts_and_reduces_full.iter().filter_map(|(&tid, action)| {
                let mut new_action = action.clone();
                let mut was_modified = false;

                match &mut new_action {
                    Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: action_nt_id, len: action_len, .. }
                        if *action_nt_id == nonterminal_id && *action_len == len => {
                        // This whole action is the promoted one, so we remove it from phase1.
                        return None;
                    }
                    Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                        // Check if the promoted reduce is part of this split.
                        if let Some(nts) = reduces.get_mut(&len) {
                            if nts.remove(&nonterminal_id).is_some() {
                                was_modified = true;
                                // If this was the last NT for this length, remove the length entry.
                                if nts.is_empty() {
                                    reduces.remove(&len);
                                }
                            }
                        }
                    }
                    _ => {} // Shift or non-matching Reduce, keep as is.
                }

                if was_modified {
                    // We modified a Split, now simplify it.
                    new_action.simplify();
                }
                Some((tid, new_action))
            }).collect::<ShiftsAndReducesWithoutDefaultReduce>();

            let default_reduce = DefaultReduce {
                clone_and_merge: !shifts_and_reduces_without_default.is_empty(),
                reduce: Some(Reduce { nonterminal_id, len, production_ids }),
            };
            (shifts_and_reduces_without_default, default_reduce)
        } else {
            let shifts_and_reduces_without_default = shifts_and_reduces_full.clone();
            let default_reduce = DefaultReduce {
                clone_and_merge: true,
                reduce: None,
            };
            (shifts_and_reduces_without_default, default_reduce)
        };

        stage_8_table.insert(state_id, Row {
            shifts_and_reduces_without_default_reduce,
            shifts_and_reduces_full,
            default_reduce,
            gotos,
        });
    }
    stage_8_table
}

/// Merges compatible states in a parse table to reduce its size (LALR(1) optimization).
///
/// This function takes a table and a list of compatible state pairs. It merges these
/// pairs and transitively merges states. For example, if (s1, s2) and (s2, s3) are
/// compatible, all three states will be merged into one.
///
/// The process involves:
/// 1.  Finding the canonical representative for each state using a disjoint-set union (DSU) approach.
/// 2.  Creating a new, smaller parse table where each row corresponds to a representative state.
/// 3.  Merging the goto maps of the merged states.
/// 4.  Updating all StateID references (in shifts and gotos) throughout the new table to point
///     to the new representative state IDs.
///
/// # Arguments
/// * `table` - The `Stage7Table` to be compacted.
/// * `item_set_map` - The mapping from item sets to their `StateID`s.
/// * `start_state_id` - The initial state ID of the parser.
/// * `compatible_pairs` - A list of state pairs that have been identified as compatible for merging.
///
/// # Returns
/// A tuple containing:
/// * The new, compacted `Stage7Table`.
/// * The new `BiBTreeMap` mapping merged item sets to the new `StateID`s.
/// * The new start state ID.
fn merge_compatible_states(
    table: &Table,
    item_set_map: &BiBTreeMap<BTreeSet<Item>, StateID>,
    start_state_id: StateID,
    compatible_pairs: &[(StateID, StateID)],
) -> (Table, BiBTreeMap<BTreeSet<Item>, StateID>, StateID) {
    if compatible_pairs.is_empty() {
        return (table.clone(), item_set_map.clone(), start_state_id);
    }

    crate::debug!(2, "Merging {} compatible state pairs.", compatible_pairs.len());

    // 1. DSU structure to manage state merging.
    let mut parent: BTreeMap<StateID, StateID> = table.keys().map(|&id| (id, id)).collect();
    fn find_set(parent: &mut BTreeMap<StateID, StateID>, i: StateID) -> StateID {
        if parent[&i] == i {
            i
        } else {
            let root = find_set(parent, parent[&i]);
            parent.insert(i, root); // Path compression
            root
        }
    }
    fn unite_sets(parent: &mut BTreeMap<StateID, StateID>, mut a: StateID, mut b: StateID) {
        a = find_set(parent, a);
        b = find_set(parent, b);
        if a != b {
            // A simple union, could be optimized by rank/size
            parent.insert(b, a);
        }
    }

    for &(s1, s2) in compatible_pairs {
        unite_sets(&mut parent, s1, s2);
    }

    // Create final mapping from old state ID to its representative.
    let mut state_map: BTreeMap<StateID, StateID> = BTreeMap::new();
    for &old_id in table.keys() {
        state_map.insert(old_id, find_set(&mut parent, old_id));
    }

    // 2. Create the new merged table.
    let mut new_table: Table = BTreeMap::new();
    let mut new_item_sets: BTreeMap<StateID, BTreeSet<Item>> = BTreeMap::new();

    let state_to_items: BTreeMap<StateID, &BTreeSet<Item>> = item_set_map.iter().map(|(k, v)| (*v, k)).collect();

    for (&old_id, row) in table.iter() {
        let new_id = state_map[&old_id];
        new_item_sets.entry(new_id).or_default().extend(state_to_items[&old_id].iter().cloned());

        let new_row = new_table.entry(new_id).or_insert_with(|| Row {
            shifts_and_reduces_without_default_reduce: row.shifts_and_reduces_without_default_reduce.clone(),
            shifts_and_reduces_full: row.shifts_and_reduces_full.clone(),
            default_reduce: row.default_reduce.clone(),
            gotos: BTreeMap::new(),
        });

        for (nt_id, goto) in &row.gotos {
            new_row.gotos.insert(*nt_id, *goto);
        }
    }

    fn remap_action_state_ids(action: &mut Stage7ShiftsAndReducesLookaheadValue, map: &BTreeMap<StateID, StateID>) {
        match action {
            Stage7ShiftsAndReducesLookaheadValue::Shift(ref mut sid) => *sid = map[sid],
            Stage7ShiftsAndReducesLookaheadValue::Split { ref mut shift, .. } => {
                if let Some(ref mut sid) = shift { *sid = map[sid]; }
            }
            _ => {}
        }
    }

    for row in new_table.values_mut() {
        row.shifts_and_reduces_without_default_reduce.values_mut().for_each(|a| remap_action_state_ids(a, &state_map));
        row.shifts_and_reduces_full.values_mut().for_each(|a| remap_action_state_ids(a, &state_map));
        row.gotos.values_mut().for_each(|g| { if let Some(ref mut sid) = g.state_id { *sid = state_map[sid]; } });
    }

    let new_item_set_map: BiBTreeMap<BTreeSet<Item>, StateID> = new_item_sets.into_iter().map(|(state_id, item_set)| (item_set, state_id)).collect();
    let new_start_state_id = state_map[&start_state_id];

    crate::debug!(2, "Merged states. Original: {}, New: {}", table.len(), new_table.len());

    (new_table, new_item_set_map, new_start_state_id)
}

pub fn generate_glr_parser_with_maps(productions: &[Production], terminal_map: BiBTreeMap<Terminal, TerminalID>, mut non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>, actions: BTreeMap<NonTerminal, crate::glr::parser::ActionFn>, ignore_terminal_id: Option<TerminalID>) -> crate::glr::parser::GLRParser {
    let original_productions = productions.to_vec();
    let start_production_id = 0;

    crate::debug!(2, "Removing productions with undefined non-terminals");
    println!("Before removing undefined non-terminals:\n{}", display_productions(&productions));
    let mut productions = remove_productions_with_undefined_nonterminals(&productions, &[start_production_id]);
    // productions = simplify_grammar(&mut productions);

    // Resolve right-recursion
    let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut unqiue_name_generator = create_unique_name_generator(&nonterminals);
    println!("Before recursion resolution:\n{}", display_productions(&productions));
    // crate::glr::analyze::resolve_right_recursion(&mut productions, &mut unqiue_name_generator);
    crate::glr::analyze::resolve_direct_right_recursion(&mut productions, &mut unqiue_name_generator);
    println!("After direct right recursion:\n{}", display_productions(&productions));

    if true {
        // println!("Before inlining nullable productions:\n{}", display_productions(&productions));
        println!("Before inlining nullable productions: Number of productions: {}", productions.len());
        productions = inline_null_productions(&productions);
        // println!("After inlining nullable productions:\n{}", display_productions(&productions));
        println!("After inlining nullable productions: Number of productions: {}", productions.len());
    }
    if false {
        println!("Before inlining unit productions:\n{}", display_productions(&productions));
        productions = inline_unit_productions(&productions);
        println!("After inlining unit productions:\n{}", display_productions(&productions));
    }

    // After recursion resolution, new non-terminals may have been added.
    // We need to update the non_terminal_map.
    let mut next_non_terminal_id = non_terminal_map.len();
    for p in &productions {
        if !non_terminal_map.contains_left(&p.lhs) {
            non_terminal_map.insert(p.lhs.clone(), NonTerminalID(next_non_terminal_id));
            next_non_terminal_id += 1;
        }
    }

    crate::debug!(2, "Validating");
    validate(&productions).expect("Validation error");

    crate::debug!(2, "Stage 1");
    let stage_1_table = stage_1(&productions);
    crate::debug!(6, &stage_1_table);
    crate::debug!(2, "Stage 2");
    let stage_2_table = stage_2(stage_1_table, &productions);
    crate::debug!(6, &stage_2_table);
    crate::debug!(2, "Stage 3");
    let stage_3_table = stage_3(stage_2_table, &productions);
    crate::debug!(6, &stage_3_table);
    crate::debug!(2, "Stage 4");
    let stage_4_table = stage_4(stage_3_table, &productions);
    crate::debug!(6, &stage_4_table);
    crate::debug!(2, "Stage 5");
    let stage_5_table = stage_5(stage_4_table, &productions, &terminal_map);
    crate::debug!(6, &stage_5_table);
    crate::debug!(2, "Stage 6");
    let stage_6_table = stage_6(stage_5_table);
    crate::debug!(6, &stage_6_table);
    crate::debug!(2, "Stage 7");
    let (mut stage_7_table, mut item_set_map, mut start_state_id) = stage_7(stage_6_table, &productions, &terminal_map, &non_terminal_map);
    crate::debug!(6, &stage_7_table);
    crate::debug!(2, "Stage 8");
    let stage_8_table = stage_8(stage_7_table);
    crate::debug!(6, &stage_8_table);
    crate::debug!(2, "Finalizing table");
    let final_table = stage_8_table;

    crate::debug!(2, "Done generating GLR parser");
    // crate::debug!(6, "Number of states: {}", final_table.len());
    // panic!("GLR parser generation complete. Number of states: {}", final_table.len());

    crate::glr::parser::GLRParser::new(final_table, productions, terminal_map, non_terminal_map, item_set_map, start_state_id, actions, ignore_terminal_id)
}

pub fn generate_glr_parser(productions: &[Production], ignore_terminal_id: Option<TerminalID>) -> crate::glr::parser::GLRParser {
    let terminal_map = assign_terminal_ids(productions);
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, terminal_map, non_terminal_map, BTreeMap::new(), ignore_terminal_id)
}

pub fn generate_glr_parser_with_terminal_map(productions: &[Production], terminal_map: BiBTreeMap<Terminal, TerminalID>, ignore_terminal_id: Option<TerminalID>) -> crate::glr::parser::GLRParser {
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, terminal_map, non_terminal_map, BTreeMap::new(), ignore_terminal_id)
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

pub fn assign_non_terminal_ids(productions: &[Production]) -> BiBTreeMap<NonTerminal, NonTerminalID> {
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
use crate::glr::parser::{GLRParser, ActionFn};
