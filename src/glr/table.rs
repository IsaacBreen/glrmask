use super::items::{compute_closure, compute_goto, split_on_dot, Item};
use crate::glr::grammar::{compute_first_sets, compute_follow_sets, NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::GLRParser;
use bimap::BiBTreeMap;
use std::collections::{HashMap, VecDeque};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use crate::glr::analyze::{drop_dead, validate};
pub use crate::types::{TerminalID};

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

#[derive(Debug, Clone)]
pub enum Stage7ShiftsAndReduces {
    Shift(StateID),
    Reduce { production_id: ProductionID, nonterminal_id: NonTerminalID, len: usize },
    Split {
        shift: Option<StateID>,
        reduces: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>,
    },
}

#[derive(Debug, Clone)]
pub struct Stage7Row {
    pub shifts_and_reduces: BTreeMap<TerminalID, Stage7ShiftsAndReduces>,
    pub gotos: BTreeMap<NonTerminalID, StateID>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateID(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProductionID(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonTerminalID(pub usize);

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

        for item in &item_set {
            if item.dot_position >= item.production.rhs.len() {
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

    let mut stage_3_table = BTreeMap::new();

    for (item_set, row) in stage_2_table {
        let mut reduces: BTreeMap<Terminal, BTreeSet<Item>> = BTreeMap::new();

        for item in &row.reduces {
            let lhs = &item.production.lhs;
            let lookaheads = follow_sets.get(lhs).unwrap();

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

        // Determine if this state is an accepting state
        let is_accepting_state = item_set.iter().any(|item| {
            item.production == productions[start_production_id] && item.dot_position == item.production.rhs.len()
        });



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
            let next_state_id = *item_set_map.get_by_left(&next_item_set).unwrap();
            gotos.insert(non_terminal_id, next_state_id);
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

pub fn generate_glr_parser_with_maps(productions: &[Production], start_production_id: usize, mut terminal_map: BiBTreeMap<Terminal, TerminalID>, non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>) -> GLRParser {
    crate::debug!(2, "Validating");
    validate(productions).expect("Validation error");

    crate::debug!(2, "Stage 1");
    let stage_1_table = stage_1(productions, start_production_id);
    crate::debug!(6, &stage_1_table);
    crate::debug!(2, "Stage 2");
    let stage_2_table = stage_2(stage_1_table, productions);
    crate::debug!(6, &stage_2_table);
    crate::debug!(2, "Stage 3");
    let stage_3_table = stage_3(stage_2_table, productions);
    crate::debug!(6, &stage_3_table);
    crate::debug!(2, "Stage 4");
    let stage_4_table = stage_4(stage_3_table, productions);
    crate::debug!(6, &stage_4_table);
    crate::debug!(2, "Stage 5");
    let stage_5_table = stage_5(stage_4_table, productions);
    crate::debug!(6, &stage_5_table);
    crate::debug!(2, "Stage 6");
    let stage_6_table = stage_6(stage_5_table);
    crate::debug!(6, &stage_6_table);
    crate::debug!(2, "Stage 7");
    let (stage_7_table, item_set_map, start_state_id) = stage_7(stage_6_table, productions, start_production_id, &terminal_map, &non_terminal_map);
    crate::debug!(6, &stage_7_table);
    crate::debug!(2, "Done generating GLR parser");

    GLRParser::new(stage_7_table, productions.to_vec(), terminal_map, non_terminal_map, item_set_map, start_state_id)
}

pub fn generate_glr_parser(productions: &[Production], start_production_id: usize) -> GLRParser {
    let terminal_map = assign_terminal_ids(productions);
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, start_production_id, terminal_map, non_terminal_map)
}

pub fn generate_glr_parser_with_terminal_map(productions: &[Production], start_production_id: usize, terminal_map: BiBTreeMap<Terminal, TerminalID>) -> GLRParser {
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(productions, start_production_id, terminal_map, non_terminal_map)
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

