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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::Display;
use memory_stats::memory_stats;
use crate::glr::parser::{ActionFn, ExpectElse, GLRParser};
use crate::profiler::{print_summary, print_summary_flat};

const EVERYTHING: bool = false;

type Stage1Table = Vec<Stage1Row>;
type Stage2Table = Vec<Stage2Row>;
type Stage3Table = Vec<Stage3Row>;
type Stage4Table = Vec<Stage4Row>;
type Stage5Table = Vec<Stage5Row>;
pub(crate) type Stage6Table = Vec<Stage6Row>;
type Stage7Table = Vec<Stage7Row>;
type Stage8Table = Vec<Row>;
pub type Table = Vec<Row>;

#[derive(Debug, Clone)]
struct Stage1Entry {
    /// Items in this state whose symbol under the dot is `symbol`.
    kernel: Vec<Item>,
    /// ID of the state reached by shifting over that symbol.
    goto_id: Option<StateID>,
}

type Stage1Row = Vec<(Option<SymbolID>, Stage1Entry)>;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage2Row {
    shifts: Vec<(TerminalID, StateID)>,
    gotos: Vec<(NonTerminalID, StateID)>,
    reduces: Vec<Item>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage3Row {
    shifts: Vec<(TerminalID, StateID)>,
    gotos: Vec<(NonTerminalID, StateID)>,
    reduces: Vec<(Option<TerminalID>, Vec<Item>)>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage4Row {
    shifts: Vec<(TerminalID, StateID)>,
    gotos: Vec<(NonTerminalID, StateID)>,
    reduces: Vec<(Option<TerminalID>, Vec<ProductionID>)>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage5Row {
    shifts: Vec<(TerminalID, StateID)>,
    gotos: Vec<(NonTerminalID, StateID)>,
    reduces: Vec<(TerminalID, Vec<ProductionID>)>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Stage6Row {
    pub(crate) shifts_and_reduces: Vec<(TerminalID, Stage6ShiftsAndReduces)>,
    pub(crate) gotos: Vec<(NonTerminalID, StateID)>,
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
        reduces: Vec<(usize, Vec<(NonTerminalID, Vec<ProductionID>)>)>,
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
                    obj.remove("production_ids").ok_or_else(|| {
                        "Missing field production_ids for Reduce".to_string()
                    })?,
                )?,
            }),
            _ => Err("Expected JSONNode::Object for Reduce".to_string()),
        }
    }
}

type ShiftsAndReducesFull = Vec<(TerminalID, Stage7ShiftsAndReducesLookaheadValue)>;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Stage7Row {
    pub shifts_and_reduces_full: ShiftsAndReducesFull,
    pub gotos: Vec<(NonTerminalID, Goto)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    shifts_and_reduces_full: ShiftsAndReducesFull,
    pub gotos: Vec<(NonTerminalID, Goto)>,
}

pub fn iter_rows(table: &Table) -> impl Iterator<Item = (StateID, &Row)> {
    table.iter().enumerate().map(|(i, r)| (StateID(i), r))
}

pub fn get_row(table: &Table, state_id: StateID) -> Option<&Row> {
    table.get(state_id.0)
}

impl Row {
    pub fn get_shifts_and_reduces_for_terminal(&self, terminal_id: &TerminalID) -> Option<Stage7ShiftsAndReducesLookaheadValue> {
        self.shifts_and_reduces_full.binary_search_by_key(terminal_id, |(k, _)| *k)
            .ok()
            .map(|idx| self.shifts_and_reduces_full[idx].1.clone())
    }

    pub fn get_shifts_and_reduces_map(&self) -> BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue> {
        self.shifts_and_reduces_full.iter().cloned().collect()
    }

    pub fn get_gotos(&self) -> BTreeMap<NonTerminalID, Goto> {
        self.gotos.iter().cloned().collect()
    }

    pub fn handle_shifts_and_reduces_for_terminal(
        &self,
        terminal_id: TerminalID,
        shiftfn: impl FnOnce(&StateID),
        mut reducefn: impl FnMut(&NonTerminalID, &usize, &Vec<ProductionID>),
    ) {
        if let Some(action) = self.get_shifts_and_reduces_for_terminal(&terminal_id) {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(state_id) => shiftfn(&state_id),
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => reducefn(&nonterminal_id, &len, &production_ids),
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    if let Some(state_id) = shift {
                        shiftfn(&state_id);
                    }
                    for (_len, nts) in reduces {
                        for (nt_id, pids) in nts {
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
                gotos: Vec::<(NonTerminalID, Goto)>::from_json(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymbolID {
    Terminal(TerminalID),
    NonTerminal(NonTerminalID),
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
    productions: &[Production],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    non_terminal_map: &BiBTreeMap<NonTerminal, NonTerminalID>,
) -> (Stage1Result, BiBTreeMap<Vec<Item>, StateID>) {
    let start_production_id = 0;
    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = vec![initial_item];

    // 1. Intern Symbols
    let mut symbol_to_id: HashMap<Symbol, usize> = HashMap::new();
    let mut id_to_symbol_id: Vec<SymbolID> = Vec::new();
    
    let mut get_id = |sym: &Symbol| -> usize {
        if let Some(&id) = symbol_to_id.get(sym) {
            id
        } else {
            let id = id_to_symbol_id.len();
            let symbol_id = match sym {
                Symbol::Terminal(t) => SymbolID::Terminal(*terminal_map.get_by_left(t).unwrap()),
                Symbol::NonTerminal(nt) => SymbolID::NonTerminal(*non_terminal_map.get_by_left(nt).unwrap()),
            };
            symbol_to_id.insert(sym.clone(), id);
            id_to_symbol_id.push(symbol_id);
            id
        }
    };

    let rhs_light: Vec<Vec<usize>> = productions.iter().map(|p| {
        p.rhs.iter().map(|s| get_id(s)).collect()
    }).collect();
    
    let lhs_light: Vec<usize> = productions.iter().map(|p| {
        get_id(&Symbol::NonTerminal(p.lhs.clone()))
    }).collect();

    let num_symbols = id_to_symbol_id.len();
    let mut prods_by_lhs_id: Vec<Vec<usize>> = vec![Vec::new(); num_symbols];
    for (idx, &lhs_id) in lhs_light.iter().enumerate() {
        prods_by_lhs_id[lhs_id].push(idx);
    }

    // 2. Precompute Closure Cache (Light)
    let mut closure_cache: Vec<Vec<Item>> = vec![Vec::new(); num_symbols];
    
    for (lhs_id, indices) in prods_by_lhs_id.iter().enumerate() {
        if indices.is_empty() { continue; }
        
        let mut visited = vec![false; num_symbols];
        let mut stack = vec![lhs_id];
        visited[lhs_id] = true;
        
        let mut items = Vec::new();
        
        while let Some(curr_id) = stack.pop() {
            for &pid in &prods_by_lhs_id[curr_id] {
                items.push(Item { production_id: pid, dot_position: 0 });
                if let Some(&next_sym_id) = rhs_light[pid].first() {
                    if !prods_by_lhs_id[next_sym_id].is_empty() {
                        if !visited[next_sym_id] {
                            visited[next_sym_id] = true;
                            stack.push(next_sym_id);
                        }
                    }
                }
            }
        }
        closure_cache[lhs_id] = items;
    }

    // 3. State Generation Loop
    let mut item_set_map_fast: HashMap<Vec<Item>, StateID> = HashMap::new();
    let mut next_state_id = 0;

    let mut worklist = VecDeque::new();
    let mut table: Stage1Table = Vec::new();

    item_set_map_fast.insert(initial_item_set.clone(), StateID(next_state_id));
    next_state_id += 1;
    worklist.push_back(initial_item_set);
    
    if EVERYTHING {
        let mut everything_item_set = Vec::new();
        for (prod_idx, prod) in productions.iter().enumerate() {
            for dot_position in 0..=prod.rhs.len() {
                let item = Item {
                    production_id: prod_idx,
                    dot_position,
                };
                everything_item_set.push(item);
            }
        }
        everything_item_set.sort_unstable();
        everything_item_set.dedup();
        if !item_set_map_fast.contains_key(&everything_item_set) {
            item_set_map_fast.insert(everything_item_set.clone(), StateID(next_state_id));
            next_state_id += 1;
            worklist.push_back(everything_item_set);
        }
    }

    let mut buckets: HashMap<Option<usize>, Vec<Item>> = HashMap::new();

    while let Some(item_set) = worklist.pop_front() {
        let state_id = *item_set_map_fast.get(&item_set).unwrap();
        
        buckets.clear();
        let mut processed_nts_set = HashSet::new();

        for item in &item_set {
            let sym_opt = rhs_light[item.production_id].get(item.dot_position).copied();
            buckets.entry(sym_opt).or_default().push(*item);

            if let Some(sym_id) = sym_opt {
                if !prods_by_lhs_id[sym_id].is_empty() {
                    if processed_nts_set.insert(sym_id) {
                        for &cached_item in &closure_cache[sym_id] {
                            let c_sym_opt = rhs_light[cached_item.production_id].get(0).copied();
                            buckets.entry(c_sym_opt).or_default().push(cached_item);
                        }
                    }
                }
            }
        }

        let mut row: Stage1Row = Vec::new();
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
            
            let symbol = symbol_id_opt.map(|id| id_to_symbol_id[id]);
            let mut kernel = if symbol.is_none() {
                items_in_split_vec
            } else {
                Vec::new()
            };
            kernel.sort_unstable();
            kernel.dedup();
            
            row.push((symbol, Stage1Entry { kernel, goto_id }));
        }
        row.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        // Verify state_id matches table length
        debug_assert_eq!(state_id.0, table.len());
        table.push(row);
    }

    let mut item_set_map = BiBTreeMap::new();
    for (k, v) in item_set_map_fast {
        item_set_map.insert(k, v);
    }

    (table, item_set_map)
}

fn print_stage_1_stats(table: &Stage1Table) {
    let num_states = table.len();
    let mut total_kernel_size = 0;
    let mut max_kernel_size = 0;
    for row in table {
        for (_, entry) in row {
            total_kernel_size += entry.kernel.len();
            max_kernel_size = max_kernel_size.max(entry.kernel.len());
        }
    }
    println!("Stage 1 Stats:");
    println!("  Num States: {}", num_states);
    println!("  Avg Kernel Size: {:.2}", total_kernel_size as f64 / num_states as f64); // This is rough, kernel is per transition?
    // Actually kernel is per entry in row.
    // Let's count transitions too.
    let mut total_transitions = 0;
    for row in table {
        total_transitions += row.len();
    }
    println!("  Num Transitions: {}", total_transitions);
    println!("  Max Kernel Size: {}", max_kernel_size);
}

fn stage_2(stage_1_table: Stage1Table, productions: &[Production]) -> Stage2Table {
    let mut stage_2_table = Vec::new();
    for (state_id_usize, transitions) in stage_1_table.into_iter().enumerate() {
        let state_id = StateID(state_id_usize);
        let mut shifts = Vec::new();
        let mut gotos = Vec::new();
        let mut reduces = Vec::new();

        for (symbol_opt, Stage1Entry { kernel, goto_id }) in transitions {
            match (symbol_opt, goto_id) {
                (Some(SymbolID::Terminal(t_id)), Some(id)) => {
                    shifts.push((t_id, id));
                }
                (Some(SymbolID::NonTerminal(nt_id)), Some(id)) => {
                    gotos.push((nt_id, id));
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

        shifts.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        gotos.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        reduces.sort_unstable();
        reduces.dedup();
        stage_2_table.push(Stage2Row { shifts, gotos, reduces });
    }
    stage_2_table
}

fn stage_3(
    stage_2_table: Stage2Table,
    productions: &[Production],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
) -> Stage3Table {
    let mut stage_3_table = Vec::new();

    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let first_sets = compute_first_sets_for_nonterminals(productions, &nullable_nonterminals);
    let follow_sets =
        compute_follow_sets_for_nonterminals(productions, &first_sets, &nullable_nonterminals);

    for (state_id_usize, row) in stage_2_table.into_iter().enumerate() {
        let state_id = StateID(state_id_usize);
        let mut reduces_map: BTreeMap<Option<TerminalID>, Vec<Item>> = BTreeMap::new();
        for item in &row.reduces {
            let lhs = &productions[item.production_id].lhs;
            if let Some(follows) = follow_sets.get(lhs) {
                for look in follows {
                    let look_id = match look {
                        Some(t) => Some(*terminal_map.get_by_left(t).unwrap()),
                        None => None,
                    };
                    reduces_map.entry(look_id).or_default().push(item.clone());
                }
            }
        }
        for vec in reduces_map.values_mut() {
            vec.sort_unstable();
            vec.dedup();
        }
        let mut reduces: Vec<(Option<TerminalID>, Vec<Item>)> = reduces_map.into_iter().collect();
        // BTreeMap iter is sorted by key, so reduces is sorted.
        
        stage_3_table.push(
            Stage3Row {
                shifts: row.shifts,
                gotos: row.gotos,
                reduces,
            },
        );
    }

    stage_3_table
}

fn stage_4(stage_3_table: Stage3Table) -> Stage4Table {
    let mut stage_4_table = Vec::new();
    for row in stage_3_table {
        let mut reduces = Vec::new();
        for (terminal, item_set_for_terminal) in row.reduces {
            let mut prod_ids = Vec::new();
            for item in item_set_for_terminal {
                prod_ids.push(ProductionID(item.production_id));
            }
            prod_ids.sort_unstable();
            prod_ids.dedup();
            reduces.push((terminal, prod_ids));
        }
        stage_4_table.push(
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
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
) -> Stage5Table {
    let mut stage_5_table = Vec::new();

    let all_terminals: Vec<TerminalID> = terminal_map.right_values().cloned().collect();
    for row in stage_4_table {
        let Stage4Row {
            shifts,
            gotos,
            reduces,
        } = row;
        let mut new_reduces_map: BTreeMap<TerminalID, Vec<ProductionID>> = BTreeMap::new();
        for (opt_term, prod_ids) in reduces {
            if let Some(term) = opt_term {
                new_reduces_map.entry(term).or_default().extend(prod_ids.into_iter());
            } else {
                for terminal in &all_terminals {
                    new_reduces_map
                        .entry(terminal.clone())
                        .or_default()
                        .extend(prod_ids.iter().cloned());
                }
            }
        }
        for vec in new_reduces_map.values_mut() {
            vec.sort_unstable();
            vec.dedup();
        }
        let new_reduces: Vec<(TerminalID, Vec<ProductionID>)> = new_reduces_map.into_iter().collect();
        stage_5_table.push(Stage5Row { shifts, gotos, reduces: new_reduces });
    }
    stage_5_table
}

fn stage_6(stage_5_table: Stage5Table) -> Stage6Table {
    let mut stage_6_table = Vec::new();
    for row in stage_5_table {
        let mut shifts_and_reduces_map = BTreeMap::new();
        
        for (terminal, state_id) in &row.shifts {
            shifts_and_reduces_map.entry(*terminal).or_insert(Stage6ShiftsAndReduces {
                shift: None,
                reduces: Vec::new(),
            }).shift = Some(*state_id);
        }
        
        for (terminal, prod_ids) in row.reduces {
            shifts_and_reduces_map.entry(terminal).or_insert(Stage6ShiftsAndReduces {
                shift: None,
                reduces: Vec::new(),
            }).reduces = prod_ids;
        }

        let shifts_and_reduces: Vec<(TerminalID, Stage6ShiftsAndReduces)> = shifts_and_reduces_map.into_iter().collect();
        stage_6_table.push(Stage6Row { shifts_and_reduces, gotos: row.gotos });
    }
    stage_6_table
}

fn stage_7(
    stage_6_table: Stage6Table,
    item_set_map: &BiBTreeMap<Vec<Item>, StateID>,
    productions: &[Production],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    non_terminal_map: &BiBTreeMap<NonTerminal, NonTerminalID>,
) -> (Stage7Table, StateID, StateID) {
    let mut stage_7_table = Vec::new();
    for row in stage_6_table {
        let mut shifts_and_reduces_full = Vec::new();
        for (terminal_id, shifts_and_reduces) in row.shifts_and_reduces {
            let lookahead_value = if shifts_and_reduces.reduces.is_empty() {
                if let Some(shift) = shifts_and_reduces.shift {
                    Stage7ShiftsAndReducesLookaheadValue::Shift(shift)
                } else {
                    continue;
                }
            } else if shifts_and_reduces.shift.is_none() && shifts_and_reduces.reduces.len() == 1 {
                let production_id = shifts_and_reduces.reduces[0];
                let production = &productions[production_id.0];
                Stage7ShiftsAndReducesLookaheadValue::Reduce {
                    nonterminal_id: production.lhs_id,
                    len: production.rhs.len(),
                    production_ids: vec![production_id],
                }
            } else {
                // Split
                let mut reduces_map: BTreeMap<usize, BTreeMap<NonTerminalID, Vec<ProductionID>>> =
                    BTreeMap::new();
                for production_id in shifts_and_reduces.reduces {
                    let production = &productions[production_id.0];
                    reduces_map
                        .entry(production.rhs.len())
                        .or_default()
                        .entry(production.lhs_id)
                        .or_default()
                        .push(production_id);
                }

                // Flatten reduces_map
                let mut reduces_vec = Vec::new();
                for (len, inner_map) in reduces_map {
                    let mut inner_vec = Vec::new();
                    for (nt_id, pids) in inner_map {
                        inner_vec.push((nt_id, pids));
                    }
                    // inner_map iteration is sorted by key (NonTerminalID)
                    reduces_vec.push((len, inner_vec));
                }
                // reduces_map iteration is sorted by key (len)

                Stage7ShiftsAndReducesLookaheadValue::Split {
                    shift: shifts_and_reduces.shift,
                    reduces: reduces_vec,
                }
            };
            shifts_and_reduces_full.push((terminal_id, lookahead_value));
        }
        
        // shifts_and_reduces_full is already sorted by terminal_id because stage_6 output was sorted?
        // stage_6 used BTreeMap -> Vec, so yes.
        
        let mut gotos_vec = Vec::new();
        for (nonterminal_id, state_id) in row.gotos {
            gotos_vec.push((nonterminal_id, Goto { state_id }));
        }
        // row.gotos was sorted (Vec from BTreeMap)

        stage_7_table.push(Stage7Row {
            shifts_and_reduces_full,
            gotos: gotos_vec,
        });
    }

    let start_production_id = 0;
    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = vec![initial_item];
    let start_state_id = *item_set_map.get_by_left(&initial_item_set).unwrap();

    let start_non_terminal_id =
        *non_terminal_map.get_by_left(&productions[start_production_id].lhs).unwrap();
    
    // Mark accept in start state
    if let Some(row) = stage_7_table.get_mut(start_state_id.0) {
        if let Ok(idx) = row.gotos.binary_search_by_key(&start_non_terminal_id, |(k, _)| *k) {
            row.gotos[idx].1.accept = true;
        } else {
             // It's possible there is no goto for start symbol if it's epsilon?
             // But usually there is.
             // If not found, we might need to insert it?
             // Original code: .entry().or_default().accept = true;
             // If it wasn't there, it adds it.
             // So we should insert if not present.
             row.gotos.push((start_non_terminal_id, Goto { state_id: None, accept: true }));
             row.gotos.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }
    }

    let everything_state_id;
    if EVERYTHING {
        let mut everything_item_set = Vec::new();
        for (prod_idx, prod) in productions.iter().enumerate() {
            for dot_position in 0..=prod.rhs.len() {
                let item = Item {
                    production_id: prod_idx,
                    dot_position,
                };
                everything_item_set.push(item);
            }
        }
        everything_item_set.sort_unstable();
        everything_item_set.dedup();
        everything_state_id = *item_set_map
            .get_by_left(&everything_item_set)
            .expect("Everything item set not found in state map");
        
        if let Some(row) = stage_7_table.get_mut(everything_state_id.0) {
             if let Ok(idx) = row.gotos.binary_search_by_key(&start_non_terminal_id, |(k, _)| *k) {
                row.gotos[idx].1.accept = true;
            } else {
                 row.gotos.push((start_non_terminal_id, Goto { state_id: None, accept: true }));
                 row.gotos.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            }
        }
    } else {
        everything_state_id = start_state_id;
    }

    (stage_7_table, start_state_id, everything_state_id)
}

fn stage_8(stage_7_table: Stage7Table) -> Stage8Table {
    let mut stage_8_table = Vec::new();
    for row in stage_7_table {
        stage_8_table.push(Row {
            shifts_and_reduces_full: row.shifts_and_reduces_full,
            gotos: row.gotos,
        });
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

    crate::debug!(2, "Stage 1");
    let start = std::time::Instant::now();
    let (stage_1_table, item_set_map) = stage_1(&productions, &terminal_map, &non_terminal_map);
    crate::debug!(2, "Stage 1 done in {:.2?}", start.elapsed());
    print_stage_1_stats(&stage_1_table);
    print_memory_usage("After Stage 1");
    crate::debug!(2, "Stage 2");
    let start = std::time::Instant::now();
    let stage_2_table = stage_2(stage_1_table, &productions);
    crate::debug!(2, "Stage 2 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 2");
    crate::debug!(2, "Stage 3");
    let start = std::time::Instant::now();
    let stage_3_table = stage_3(stage_2_table, &productions, &terminal_map);
    crate::debug!(2, "Stage 3 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 3");
    crate::debug!(2, "Stage 4");
    let start = std::time::Instant::now();
    let stage_4_table = stage_4(stage_3_table);
    crate::debug!(2, "Stage 4 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 4");
    crate::debug!(2, "Stage 5");
    let start = std::time::Instant::now();
    let stage_5_table = stage_5(stage_4_table, &terminal_map);
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
        &terminal_map,
        &non_terminal_map,
    );
    crate::debug!(2, "Stage 7 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 7");
    crate::debug!(2, "Stage 8");
    let start = std::time::Instant::now();
    let final_table = stage_8(stage_7_table);
    crate::debug!(2, "Stage 8 done in {:.2?}", start.elapsed());
    print_memory_usage("After Stage 8 (final table)");

    crate::debug!(2, "Done generating GLR parser");
    print_summary();
    print_summary_flat();
    std::process::exit(0);

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
