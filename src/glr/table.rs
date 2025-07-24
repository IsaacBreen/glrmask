use super::items::{compute_closure, compute_goto, split_on_dot, Item};
use crate::glr::grammar::{compute_epsilon_nonterminals, compute_first_sets_for_nonterminals, NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::{GLRParser, ActionFn};
use bimap::BiBTreeMap;
use std::collections::{HashMap, VecDeque};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use crate::glr::analyze::{create_unique_name_generator, drop_dead, remove_productions_with_undefined_nonterminals, simplify_grammar, validate, validate_start_production_ends_with_terminal};
pub use crate::types::{TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap;
use crate::interface::display_productions;
// Added for derive macro pattern


type Stage1Table = BTreeMap<BTreeSet<Item>, Stage1Row>;
type Stage2Table = BTreeMap<BTreeSet<Item>, Stage2Row>;
type Stage3Table = BTreeMap<BTreeSet<Item>, Stage3Row>;
type Stage4Table = BTreeMap<BTreeSet<Item>, Stage4Row>;
type Stage5Table = BTreeMap<BTreeSet<Item>, Stage5Row>;
type Stage6Table = BTreeMap<BTreeSet<Item>, Stage6Row>;
pub type Stage7Table = BTreeMap<StateID, Stage7Row>;


type Stage1Row = BTreeMap<Option<Symbol>, BTreeSet<Item>>;
#[derive(Debug)]
struct Stage2Row {
    shifts: BTreeMap<Terminal, BTreeSet<Item>>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
    reduces: BTreeSet<Item>,
}
#[derive(Debug)]
struct Stage3Row {
    shifts: BTreeMap<Terminal, BTreeSet<Item>>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
    reduces: BTreeMap<Terminal, BTreeSet<Item>>,
}
#[derive(Debug)]
struct Stage4Row {
    shifts: BTreeMap<Terminal, BTreeSet<Item>>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
    reduces: BTreeMap<Terminal, BTreeSet<ProductionID>>,
}
type Stage5Row = Stage4Row;
#[derive(Debug)]
struct Stage6Row {
    shifts_and_reduces: BTreeMap<Terminal, Stage6ShiftsAndReduces>,
    gotos: BTreeMap<NonTerminal, BTreeSet<Item>>,
}

#[derive(Debug)]
struct Stage6ShiftsAndReduces {
    shift: Option<BTreeSet<Item>>,
    reduces: BTreeSet<ProductionID>,
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

pub type Stage7Phase1ShiftsAndReduces = BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>;
pub type Stage7Phase2ShiftsAndReduces = BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>;

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
pub struct Stage7Phase3DefaultReduce {
    pub clone_and_merge: bool,
    pub reduce: Option<Reduce>,
}

impl JSONConvertible for Stage7Phase3DefaultReduce {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clone_and_merge".to_string(), self.clone_and_merge.to_json());
        obj.insert("reduce".to_string(), self.reduce.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(Stage7Phase3DefaultReduce {
                clone_and_merge: bool::from_json(obj.remove("clone_and_merge").ok_or_else(|| "Missing field clone_and_merge for Stage7Phase3DefaultReduce".to_string())?)?,
                reduce: Option::<Reduce>::from_json(obj.remove("reduce").ok_or_else(|| "Missing field reduce for Stage7Phase3DefaultReduce".to_string())?)?,
            }),
            _ => Err("Expected JSONNode::Object for Stage7Phase3DefaultReduce".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage7Row {
    pub phase1_shifts_and_reduces: Stage7Phase1ShiftsAndReduces,
    pub phase2_shifts_and_reduces: Stage7Phase2ShiftsAndReduces,
    pub phase3_default_reduce: Stage7Phase3DefaultReduce,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

// Manual impl for Stage7Row (could be derived)
impl JSONConvertible for Stage7Row {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("phase1_shifts_and_reduces".to_string(), self.phase1_shifts_and_reduces.to_json());
        obj.insert("phase2_shifts_and_reduces".to_string(), self.phase2_shifts_and_reduces.to_json());
        obj.insert("phase3_default_reduce".to_string(), self.phase3_default_reduce.to_json());
        obj.insert("gotos".to_string(), self.gotos.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(Stage7Row {
                phase1_shifts_and_reduces: Stage7Phase1ShiftsAndReduces::from_json(obj.remove("phase1_shifts_and_reduces").ok_or_else(|| "Missing field phase1_shifts_and_reduces for Stage7Row".to_string())?)?,
                phase2_shifts_and_reduces: Stage7Phase2ShiftsAndReduces::from_json(obj.remove("phase2_shifts_and_reduces").ok_or_else(|| "Missing field phase2_shifts_and_reduces for Stage7Row".to_string())?)?,
                phase3_default_reduce: Stage7Phase3DefaultReduce::from_json(obj.remove("phase3_default_reduce").ok_or_else(|| "Missing field phase3_default_reduce for Stage7Row".to_string())?)?,
                gotos: BTreeMap::<NonTerminalID, Goto>::from_json(obj.remove("gotos").ok_or_else(|| "Missing field gotos for Stage7Row".to_string())?)?,
            }),
            _ => Err("Expected JSONNode::Object for Stage7Row".to_string()),
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Goto {
    State(StateID),
    Accept,
}

impl JSONConvertible for Goto {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        match self {
            Goto::State(state_id) => {
                obj.insert("variant".to_string(), JSONNode::String("State".to_string()));
                obj.insert("value".to_string(), state_id.to_json());
            }
            Goto::Accept => {
                obj.insert("variant".to_string(), JSONNode::String("Accept".to_string()));
                // No value needed for Accept
            }
        }
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let variant = obj.remove("variant")
                    .ok_or_else(|| "Missing field 'variant' for Goto".to_string())
                    .and_then(String::from_json)?;
                match variant.as_str() {
                    "State" => {
                        let value_node = obj.remove("value").ok_or_else(|| "Missing field 'value' for Goto::State".to_string())?;
                        StateID::from_json(value_node).map(Goto::State)
                    }
                    "Accept" => Ok(Goto::Accept),
                    _ => Err(format!("Unknown variant '{}' for Goto", variant)),
                }
            }
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

fn stage_1(productions: &[Production], start_production_id: usize) -> Stage1Result {
    let initial_item = Item {
        production: productions[start_production_id].clone(),
        dot_position: 0,
        lookahead: Terminal::EOF,
    };
    let initial_closure = BTreeSet::from([initial_item]);
    let mut worklist = VecDeque::from([initial_closure.clone()]);

    let mut transitions: BTreeMap<BTreeSet<Item>, BTreeMap<Option<Symbol>, BTreeSet<Item>>> = BTreeMap::new();

    while let Some(items) = worklist.pop_front() {
        if transitions.contains_key(&items) {
            continue;
        }

        let closure = compute_closure(&items, productions);
        let splits = split_on_dot(&closure);
        let mut row = BTreeMap::new();

        for (symbol, items) in splits {
            if symbol.is_none() {
                continue;
            }
            let goto_set = compute_goto(&items);
            row.insert(symbol.clone(), goto_set.clone());
            worklist.push_back(goto_set);
        }

        transitions.insert(items.clone(), row);
    }

    transitions
}

fn stage_2(stage_1_table: Stage1Table, productions: &[Production]) -> Stage2Result {
    let mut stage_2_table = BTreeMap::new();
    for (item_set, transitions) in stage_1_table {
        let mut shifts = BTreeMap::new();
        let mut gotos = BTreeMap::new();
        let mut reduces = BTreeSet::new();

        let closure = compute_closure(&item_set, productions);
        for item in &closure { // Check the full closure for reductions
            if item.dot_position == item.production.rhs.len() {
                reduces.insert(item.clone());
            }
        }

        for (symbol_opt, next_item_set) in &transitions {
            if let Some(symbol) = symbol_opt {
                match symbol {
                    Symbol::Terminal(t) => {
                        shifts.insert(t.clone(), next_item_set.clone());
                    }
                    Symbol::NonTerminal(nt) => {
                        gotos.insert(nt.clone(), next_item_set.clone());
                    }
                }
            }
        }

        for item in &closure {
            // e.g. start rules
            if item.dot_position == 0 && !gotos.contains_key(&item.production.lhs) {
                gotos.insert(item.production.lhs.clone(), BTreeSet::new());
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
        let mut reduces: BTreeMap<Terminal, BTreeSet<Item>> = BTreeMap::new();

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

        for (terminal, items) in row.reduces {
            let mut prod_ids = BTreeSet::new();
            for item in items {
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

fn stage_5(stage_4_table: Stage4Table, productions: &[Production]) -> Stage5Result {
    // todo: remove this
    stage_4_table
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

fn stage_7(stage_6_table: Stage6Table, productions: &[Production], start_production_id: usize, terminal_map: &BiBTreeMap<Terminal, TerminalID>, non_terminal_map: &BiBTreeMap<NonTerminal, NonTerminalID>) -> Stage7Result {
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
        let mut phase2_shifts_and_reduces: Stage7Phase2ShiftsAndReduces = BTreeMap::new();

        for (terminal, action) in &row.shifts_and_reduces {
            if *terminal == Terminal::EOF {
                continue; // Skip EOF terminal
            }
            let terminal_id = *terminal_map.get_by_left(terminal).unwrap();
            let shift_state_id = action.shift.as_ref().map(|set| *item_set_map.get_by_left(set).unwrap());

            let mut reduces_by_len_and_nt: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>> = BTreeMap::new();
            for &production_id in &action.reduces {
                let production = &productions[production_id.0];
                let len = production.rhs.len();
                let nonterminal_id = *non_terminal_map.get_by_left(&production.lhs).unwrap();
                reduces_by_len_and_nt.entry(len).or_default().entry(nonterminal_id).or_default().insert(production_id);
            }

            let converted_action = if let Some(shift_id) = shift_state_id {
                if reduces_by_len_and_nt.is_empty() {
                    Stage7ShiftsAndReducesLookaheadValue::Shift(shift_id)
                } else {
                    Stage7ShiftsAndReducesLookaheadValue::Split { shift: Some(shift_id), reduces: reduces_by_len_and_nt }
                }
            } else {
                if reduces_by_len_and_nt.is_empty() {
                    panic!("Action without shift or reduce for terminal {:?}", terminal);
                } else if reduces_by_len_and_nt.len() == 1 {
                    let (len, nts) = reduces_by_len_and_nt.into_iter().next().unwrap();
                    if nts.len() == 1 {
                        let (nonterminal_id, production_ids) = nts.into_iter().next().unwrap();
                        Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids }
                    } else {
                        Stage7ShiftsAndReducesLookaheadValue::Split { shift: None, reduces: BTreeMap::from([(len, nts)]) }
                    }
                } else {
                    Stage7ShiftsAndReducesLookaheadValue::Split { shift: None, reduces: reduces_by_len_and_nt }
                }
            };
            phase2_shifts_and_reduces.insert(terminal_id, converted_action);
        }

        // --- Promotion Logic ---
        let mut reduce_counts: BTreeMap<(NonTerminalID, usize), (usize, BTreeSet<ProductionID>)> = BTreeMap::new();
        for action in phase2_shifts_and_reduces.values() {
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

        let (phase1_shifts_and_reduces, phase3_default_reduce) = if let Some((nonterminal_id, len)) = promoted_reduce_key {
            let (_, production_ids) = reduce_counts.remove(&(nonterminal_id, len)).unwrap();
            
            let phase1 = phase2_shifts_and_reduces.iter().filter_map(|(&tid, action)| {
                match action {
                    Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: action_nt_id, len: action_len, .. }
                        if *action_nt_id == nonterminal_id && *action_len == len => None,
                    Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces, .. } => {
                        let mut reduces = reduces.clone();
                        use std::collections::btree_map::Entry;
                        match reduces.entry(len) {
                            Entry::Occupied(mut entry) => {
                                if entry.get_mut().remove(&nonterminal_id).is_some() {
                                    if entry.get().is_empty() {
                                        entry.remove();
                                    }
                                }
                                match (shift, reduces.iter().map(|(_, nts)| nts.len()).sum::<usize>()) {
                                    (None, 0) => None,
                                    (None, 1) => {
                                        let (len, nonterminal_id_to_production_ids) = reduces.into_iter().next().unwrap();
                                        let (nonterminal_id, production_ids) = nonterminal_id_to_production_ids.into_iter().next().unwrap();
                                        Some((tid, Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids }))
                                    },
                                    (Some(shift_id), 0) => Some((tid, Stage7ShiftsAndReducesLookaheadValue::Shift(*shift_id))),
                                    (&shift, 2..) => Some((tid, Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces })),
                                    (&shift @ Some(_), 1..) => Some((tid, Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces })),
                                }
                            },
                            Entry::Vacant(_) => Some((tid, action.clone())),
                        }
                    },
                    _ => Some((tid, action.clone()))
                }
            }).collect::<Stage7Phase1ShiftsAndReduces>();

            let phase3 = Stage7Phase3DefaultReduce {
                clone_and_merge: !phase1.is_empty(),
                reduce: Some(Reduce { nonterminal_id, len, production_ids }),
            };
            (phase1, phase3)
        } else {
            let phase1 = phase2_shifts_and_reduces.clone();
            let phase3 = Stage7Phase3DefaultReduce {
                clone_and_merge: true,
                reduce: None,
            };
            (phase1, phase3)
        };

        let mut gotos = BTreeMap::new();
        for (nonterminal, next_item_set) in row.gotos {
            let non_terminal_id = *non_terminal_map.get_by_left(&nonterminal).unwrap();
            let goto = item_set_map.get_by_left(&next_item_set).map_or(Goto::Accept, |&id| Goto::State(id));
            if goto == Goto::Accept { assert!(next_item_set.is_empty()); }
            gotos.insert(non_terminal_id, goto);
        }

        stage_7_table.insert(state_id, Stage7Row {
            phase1_shifts_and_reduces,
            phase2_shifts_and_reduces,
            phase3_default_reduce,
            gotos,
        });
    }

    let initial_item = Item {
        production: productions[start_production_id].clone(),
        dot_position: 0,
        lookahead: Terminal::EOF,
    };
    let initial_item_set = BTreeSet::from([initial_item]);
    let start_state_id = *item_set_map.get_by_left(&initial_item_set).unwrap();

    // Ensure at least one row has more than one item
    println!("There are {} states in the GLR parser.", stage_7_table.len());
    if !stage_7_table.values().any(|row| row.phase2_shifts_and_reduces.len() > 1) {
        panic!("Generated GLR parser has no state with multiple items. This is likely an error in the grammar or the parser generation logic.");
    }

    (stage_7_table, item_set_map, start_state_id)
}

pub fn generate_glr_parser_with_maps(productions: &[Production], start_production_id: usize, terminal_map: BiBTreeMap<Terminal, TerminalID>, mut non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>, actions: BTreeMap<NonTerminal, ActionFn>, ignore_terminal_id: Option<TerminalID>) -> GLRParser {
    let original_productions = productions.to_vec();

    crate::debug!(2, "Removing productions with undefined non-terminals");
    println!("Before removing undefined non-terminals:\n{}", display_productions(&productions));
    let productions = remove_productions_with_undefined_nonterminals(&productions, &[start_production_id]);
    // (productions, start_production_id) = simplify_grammar(&mut productions, start_production_id);

    // Resolve right-recursion
    let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut unqiue_name_generator = create_unique_name_generator(&nonterminals);
    let mut productions = productions.to_vec();
    println!("Before recursion resolution:\n{}", display_productions(&productions));
    // crate::glr::analyze::resolve_right_recursion(&mut productions, &mut unqiue_name_generator);
    crate::glr::analyze::resolve_direct_right_recursion(&mut productions, &mut unqiue_name_generator);
    println!("After direct right recursion:\n{}", display_productions(&productions));

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
    validate_start_production_ends_with_terminal(&productions, start_production_id)
        .expect("Start production does not end with a terminal");

    crate::debug!(2, "Stage 1");
    let stage_1_table = stage_1(&productions, start_production_id);
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
    let stage_5_table = stage_5(stage_4_table, &productions);
    crate::debug!(6, &stage_5_table);
    crate::debug!(2, "Stage 6");
    let stage_6_table = stage_6(stage_5_table);
    crate::debug!(6, &stage_6_table);
    crate::debug!(2, "Stage 7");
    let (mut stage_7_table, item_set_map, start_state_id) = stage_7(stage_6_table, &productions, start_production_id, &terminal_map, &non_terminal_map);
    crate::debug!(6, &stage_7_table);

    crate::debug!(2, "Done generating GLR parser");

    GLRParser::new(stage_7_table, productions, terminal_map, non_terminal_map, item_set_map, start_state_id, actions, ignore_terminal_id)
}

pub fn generate_glr_parser(productions: &[Production], start_production_id: usize, ignore_terminal_id: Option<TerminalID>) -> GLRParser {
    let terminal_map = assign_terminal_ids(productions);
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, start_production_id, terminal_map, non_terminal_map, BTreeMap::new(), ignore_terminal_id)
}

pub fn generate_glr_parser_with_terminal_map(productions: &[Production], start_production_id: usize, terminal_map: BiBTreeMap<Terminal, TerminalID>, ignore_terminal_id: Option<TerminalID>) -> GLRParser {
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, start_production_id, terminal_map, non_terminal_map, BTreeMap::new(), ignore_terminal_id)
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
