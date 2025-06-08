use super::items::{compute_closure, compute_goto, split_on_dot, Item};
use crate::glr::grammar::{compute_first_sets, compute_follow_sets, NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::{GLRParser, ActionFn};
use bimap::BiBTreeMap;
use std::collections::{HashMap, VecDeque};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use crate::glr::analyze::{create_unique_name_generator, drop_dead, remove_productions_with_undefined_nonterminals, simplify_grammar, validate};
pub use crate::types::{TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap; // Added for derive macro pattern


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
enum Stage6ShiftsAndReduces {
    Shift(BTreeSet<Item>),
    Reduce(ProductionID),
    Split {
        shift: Option<BTreeSet<Item>>,
        reduces: BTreeSet<ProductionID>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage7ShiftsAndReduces {
    Shift(StateID),
    Reduce { production_id: ProductionID, nonterminal_id: NonTerminalID, len: usize },
    Split {
        shift: Option<StateID>,
        reduces: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>,
    },
}

impl JSONConvertible for Stage7ShiftsAndReduces {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        match self {
            Stage7ShiftsAndReduces::Shift(state_id) => {
                obj.insert("variant".to_string(), JSONNode::String("Shift".to_string()));
                obj.insert("state_id".to_string(), state_id.to_json());
            }
            Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id, len } => {
                obj.insert("variant".to_string(), JSONNode::String("Reduce".to_string()));
                obj.insert("production_id".to_string(), production_id.to_json());
                obj.insert("nonterminal_id".to_string(), nonterminal_id.to_json());
                obj.insert("len".to_string(), len.to_json());
            }
            Stage7ShiftsAndReduces::Split { shift, reduces } => {
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
                let variant = obj.remove("variant").ok_or_else(|| "Missing field variant for Stage7ShiftsAndReduces".to_string())
                                   .and_then(String::from_json)?;
                match variant.as_str() {
                    "Shift" => {
                        let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for Shift".to_string())
                                          .and_then(StateID::from_json)?;
                        Ok(Stage7ShiftsAndReduces::Shift(state_id))
                    }
                    "Reduce" => {
                        let production_id = obj.remove("production_id").ok_or_else(|| "Missing field production_id for Reduce".to_string())
                                               .and_then(ProductionID::from_json)?;
                        let nonterminal_id = obj.remove("nonterminal_id").ok_or_else(|| "Missing field nonterminal_id for Reduce".to_string())
                                                .and_then(NonTerminalID::from_json)?;
                        let len = obj.remove("len").ok_or_else(|| "Missing field len for Reduce".to_string())
                                     .and_then(usize::from_json)?;
                        Ok(Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id, len })
                    }
                    "Split" => {
                        let shift = obj.remove("shift").ok_or_else(|| "Missing field shift for Split".to_string())
                                       .and_then(Option::<StateID>::from_json)?;
                        let reduces = obj.remove("reduces").ok_or_else(|| "Missing field reduces for Split".to_string())
                                         .and_then(|n| BTreeMap::<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>::from_json(n))?;
                        Ok(Stage7ShiftsAndReduces::Split { shift, reduces })
                    }
                    _ => Err(format!("Unknown variant {} for Stage7ShiftsAndReduces", variant)),
                }
            }
            _ => Err("Expected JSONNode::Object for Stage7ShiftsAndReduces".to_string()),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage7Row {
    pub shifts_and_reduces: BTreeMap<TerminalID, Stage7ShiftsAndReduces>,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

// Manual impl for Stage7Row (could be derived)
impl JSONConvertible for Stage7Row {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("shifts_and_reduces".to_string(), self.shifts_and_reduces.to_json());
        obj.insert("gotos".to_string(), self.gotos.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let shifts_and_reduces = obj.remove("shifts_and_reduces").ok_or_else(|| "Missing field shifts_and_reduces for Stage7Row".to_string())
                                            .and_then(|n| BTreeMap::<TerminalID, Stage7ShiftsAndReduces>::from_json(n))?;
                let gotos = obj.remove("gotos").ok_or_else(|| "Missing field gotos for Stage7Row".to_string())
                               .and_then(|n| BTreeMap::<NonTerminalID, Goto>::from_json(n))?;
                Ok(Stage7Row { shifts_and_reduces, gotos })
            }
            _ => Err("Expected JSONNode::Object for Stage7Row".to_string()),
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
    let follow_sets = compute_follow_sets(productions);
    crate::debug!(3, "Follow sets:");
    for (nt, follow_set) in &follow_sets {
        crate::debug!(3, "  {}: {}", nt.0, follow_set.iter().map(|t| t.0.to_string()).collect::<Vec<_>>().join(", "));
    }

    let mut stage_3_table = BTreeMap::new();

    for (item_set, row) in stage_2_table {
        let mut reduces: BTreeMap<Terminal, BTreeSet<Item>> = BTreeMap::new();

        for item in &row.reduces {
            let lhs = &item.production.lhs;
            let lookaheads = follow_sets.get(lhs).cloned().unwrap_or_default(); // Handle if NT not in follow_sets

            for terminal in lookaheads {
                reduces
                    .entry(terminal.clone())
                    .or_default()
                    .insert(item.clone());
            }
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

        for (terminal, next_item_set) in row.shifts {
            shifts_and_reduces.insert(terminal, Stage6ShiftsAndReduces::Shift(next_item_set));
        }

        for (terminal, mut production_ids) in row.reduces {
            if let Some(mut existing) = shifts_and_reduces.remove(&terminal) {
                match existing {
                    Stage6ShiftsAndReduces::Shift(shift_set) => {
                        shifts_and_reduces.insert(terminal, Stage6ShiftsAndReduces::Split {
                            shift: Some(shift_set.clone()),
                            reduces: production_ids.clone(),
                        });
                    }
                    Stage6ShiftsAndReduces::Reduce(existing_production_id) => {
                        production_ids.insert(existing_production_id);
                        shifts_and_reduces.insert(terminal, Stage6ShiftsAndReduces::Split {
                            shift: None,
                            reduces: production_ids,
                        });
                    }
                    Stage6ShiftsAndReduces::Split { shift, mut reduces } => {
                        reduces.extend(production_ids.into_iter());
                        shifts_and_reduces.insert(terminal, Stage6ShiftsAndReduces::Split { shift, reduces });
                    }
                }
            } else {
                // If there's only one production ID, we can optimize by storing it directly
                if production_ids.len() == 1 {
                    shifts_and_reduces.insert(terminal, Stage6ShiftsAndReduces::Reduce(production_ids.iter().next().unwrap().clone()));
                } else {
                    shifts_and_reduces.insert(terminal, Stage6ShiftsAndReduces::Split { shift: None, reduces: production_ids });
                }
            }
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

    let mut terminals = BTreeSet::new();
    let mut non_terminals = BTreeSet::new();

    for (item_set, row) in &stage_6_table {
        item_set_map.insert(item_set.clone(), StateID(next_state_id));
        next_state_id += 1;

        for t in row.shifts_and_reduces.keys() {
            terminals.insert(t.clone());
        }

        for nt in row.gotos.keys() {
            non_terminals.insert(nt.clone());
        }
    }

    let mut stage_7_table = BTreeMap::new();

    for (item_set, row) in stage_6_table {
        let state_id = *item_set_map.get_by_left(&item_set).unwrap();
        let mut shifts_and_reduces = BTreeMap::new();
        let mut gotos = BTreeMap::new();

        for (terminal, action) in row.shifts_and_reduces {
            let terminal_id = *terminal_map.get_by_left(&terminal).expect(format!("{:?} not found in terminal map {:?}", terminal, terminal_map.left_values().map(|t| t.0.clone()).collect::<Vec<String>>()).as_str());
            let converted_action = match action {
                Stage6ShiftsAndReduces::Shift(next_item_set) => {
                    let next_state_id = *item_set_map.get_by_left(&next_item_set).unwrap();
                    Stage7ShiftsAndReduces::Shift(next_state_id)
                }
                Stage6ShiftsAndReduces::Reduce(production_id) => {
                    let production = productions.get(production_id.0).unwrap();
                    let nonterminal_id = *non_terminal_map.get_by_left(&production.lhs).unwrap();
                    let len = production.rhs.len();
                    Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id, len }
                }
                Stage6ShiftsAndReduces::Split { shift, reduces } => {
                    let shift_state_id = shift.as_ref().map(|set| *item_set_map.get_by_left(set).unwrap());
                    let mut len_to_nt_to_production_id: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>> = BTreeMap::new();
                    for production_id in reduces {
                        let production = productions.get(production_id.0).unwrap();
                        let nonterminal_id = *non_terminal_map.get_by_left(&production.lhs).unwrap();
                        let len = production.rhs.len();
                        len_to_nt_to_production_id.entry(len).or_default().entry(nonterminal_id).or_default().insert(production_id);
                    }
                    Stage7ShiftsAndReduces::Split { shift: shift_state_id, reduces: len_to_nt_to_production_id }
                }
            };
            shifts_and_reduces.insert(terminal_id, converted_action);
        }

        for (nonterminal, next_item_set) in row.gotos {
            let non_terminal_id = *non_terminal_map.get_by_left(&nonterminal).unwrap();
            let goto = item_set_map.get_by_left(&next_item_set).map_or(Goto::Accept, |&next_state_id| Goto::State(next_state_id));
            if goto == Goto::Accept { assert!(next_item_set.is_empty()); }
            gotos.insert(non_terminal_id, goto);
        }

        stage_7_table.insert(state_id, Stage7Row { shifts_and_reduces, gotos });
    }

    let start_item = Item {
        production: productions[start_production_id].clone(),
        dot_position: 0,
    };
    let start_state_id = *item_set_map.get_by_left(&BTreeSet::from([start_item])).unwrap();

    (stage_7_table, item_set_map, start_state_id)
}

pub fn generate_glr_parser_with_maps(productions: &[Production], start_production_id: usize, terminal_map: BiBTreeMap<Terminal, TerminalID>, non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>, actions: BTreeMap<NonTerminal, ActionFn>) -> GLRParser {
    let original_productions = productions.to_vec();

    crate::debug!(2, "Removing productions with undefined non-terminals");
    let productions = remove_productions_with_undefined_nonterminals(&productions, &[start_production_id]);
    // (productions, start_production_id) = simplify_grammar(&mut productions, start_production_id);

    // Resolve right-recursion
    let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut unqiue_name_generator = create_unique_name_generator(&nonterminals);
    let mut productions = productions.to_vec();
    crate::glr::analyze::remove_direct_right_recursion(&mut productions, &mut unqiue_name_generator);
    // dbg!(&productions);

    // crate::debug!(2, "Validating");
    // validate(&productions).expect("Validation error");

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
    let (stage_7_table, item_set_map, start_state_id) = stage_7(stage_6_table, &productions, start_production_id, &terminal_map, &non_terminal_map);
    crate::debug!(6, &stage_7_table);
    crate::debug!(2, "Done generating GLR parser");

    GLRParser::new(stage_7_table, original_productions, terminal_map, non_terminal_map, item_set_map, start_state_id, actions)
}

pub fn generate_glr_parser(productions: &[Production], start_production_id: usize) -> GLRParser {
    let terminal_map = assign_terminal_ids(productions);
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, start_production_id, terminal_map, non_terminal_map, BTreeMap::new())
}

pub fn generate_glr_parser_with_terminal_map(productions: &[Production], start_production_id: usize, terminal_map: BiBTreeMap<Terminal, TerminalID>) -> GLRParser {
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, start_production_id, terminal_map, non_terminal_map, BTreeMap::new())
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
