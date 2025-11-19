use super::items::Item;
use crate::glr::analyze::{
    create_unique_name_generator, inline_null_productions,
    remove_productions_with_undefined_nonterminals, validate,
};
use crate::glr::automaton::{
    compute_first_sets_ids_with_lhs, compute_follow_sets_ids, compute_nullable_nonterminals,
};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::{ActionFn, GLRParser};
use crate::json_serialization::{JSONConvertible, JSONNode};
pub use crate::types::TerminalID;
use bimap::BiBTreeMap;
use memory_stats::memory_stats;
use profiler_macro::time_it;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::Display;

const EVERYTHING: bool = false;

// Intermediate table from Stage 1
type Stage1Table = Vec<Stage1Row>;
type Stage1Row = BTreeMap<Option<usize>, Stage1Entry>;

#[derive(Debug, Clone)]
struct Stage1Entry {
    /// Items in this state whose symbol under the dot is `symbol`.
    kernel: Vec<Item>,
    /// ID of the state reached by shifting over that symbol.
    goto_id: Option<StateID>,
}

// Final Table Types
pub type Table = BTreeMap<StateID, Row>;

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

#[time_it]
fn stage_1(
    light_productions: &[Vec<usize>],
    lhs_ids: &[usize],
    num_terminals: usize,
    num_nonterminals: usize,
) -> (Stage1Table, HashMap<Vec<Item>, StateID>) {
    let start_production_id = 0;
    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = vec![initial_item];

    // Precompute productions by LHS ID
    let mut prods_by_lhs_id: Vec<Vec<usize>> = vec![Vec::new(); num_nonterminals];
    for (idx, &lhs_id) in lhs_ids.iter().enumerate() {
        prods_by_lhs_id[lhs_id].push(idx);
    }

    // Precompute Closure Cache (Light)
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

    // State Generation Loop
    let mut item_set_map_fast: HashMap<Vec<Item>, StateID> = HashMap::new();
    let mut next_state_id = 0;

    let mut worklist = VecDeque::new();
    let mut table: Stage1Table = Vec::new();

    item_set_map_fast.insert(initial_item_set.clone(), StateID(next_state_id));
    next_state_id += 1;
    worklist.push_back(initial_item_set);

    let total_symbols = num_terminals + num_nonterminals;
    let mut buckets: Vec<Vec<Item>> = vec![Vec::new(); total_symbols];
    let mut bucket_none: Vec<Item> = Vec::new();
    let mut active_symbols: Vec<usize> = Vec::with_capacity(128);
    let mut processed_nts: Vec<bool> = vec![false; num_nonterminals];
    let mut touched_nts: Vec<usize> = Vec::with_capacity(128);

    // BitSet for deduplication
    // Calculate max bits needed for Item index
    let max_prod_id = light_productions.len();
    let max_rhs_len = light_productions.iter().map(|rhs| rhs.len()).max().unwrap_or(0);
    // We need enough bits for dot_position.
    let dot_bits = (usize::BITS - (max_rhs_len + 1).leading_zeros()) as usize;
    let item_index_shift = dot_bits;
    let max_item_index = (max_prod_id << item_index_shift) | (max_rhs_len + 1);
    let bitset_size_u64 = (max_item_index + 63) / 64;
    let mut seen_items_bitset: Vec<u64> = vec![0; bitset_size_u64];

    while let Some(item_set) = worklist.pop_front() {
        let state_id = *item_set_map_fast.get(&item_set).unwrap();
        
        if state_id.0 >= table.len() {
            table.resize(state_id.0 + 1, BTreeMap::new());
        }

        // Clear buckets and tracking structures
        for &sym in &active_symbols {
            buckets[sym].clear();
        }
        active_symbols.clear();
        bucket_none.clear();
        
        for &nt in &touched_nts {
            processed_nts[nt] = false;
        }
        touched_nts.clear();

        for item in &item_set {
            let sym_opt = light_productions[item.production_id].get(item.dot_position).copied();
            
            match sym_opt {
                Some(sym_id) => {
                    if buckets[sym_id].is_empty() {
                        active_symbols.push(sym_id);
                    }
                    buckets[sym_id].push(*item);

                    if sym_id >= num_terminals {
                        let nt_id = sym_id - num_terminals;
                        if !processed_nts[nt_id] {
                            processed_nts[nt_id] = true;
                            touched_nts.push(nt_id);
                            
                            for (first_sym, items) in &closure_cache_grouped[nt_id] {
                                match first_sym {
                                    Some(fs) => {
                                        if buckets[*fs].is_empty() {
                                            active_symbols.push(*fs);
                                        }
                                        buckets[*fs].extend(items);
                                    }
                                    None => {
                                        bucket_none.extend(items);
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    bucket_none.push(*item);
                }
            }
        }

        let mut row: Stage1Row = BTreeMap::new();
        
        // Process active buckets
        for &sym_id in &active_symbols {
            let items_in_split_vec = &buckets[sym_id];
            
            // Optimized goto_set construction with BitSet deduplication
            let mut goto_set: Vec<Item> = Vec::with_capacity(items_in_split_vec.len());
            
            // Clear relevant parts of bitset? No, we must clear all used bits or use a generation counter.
            // Since we don't track which bits are set efficiently without iterating, 
            // and iterating the whole bitset is slow if large, we track modified indices?
            // Actually, just clearing the specific bits we set is faster if the set is sparse.
            // But we don't know which bits we set unless we store them.
            // Let's store indices to clear.
            let mut indices_to_clear: Vec<usize> = Vec::with_capacity(items_in_split_vec.len());

            for item in items_in_split_vec {
                let next_dot = item.dot_position + 1;
                let index = (item.production_id << item_index_shift) | next_dot;
                let word_idx = index / 64;
                let bit_idx = index % 64;
                
                if (seen_items_bitset[word_idx] & (1 << bit_idx)) == 0 {
                    seen_items_bitset[word_idx] |= 1 << bit_idx;
                    indices_to_clear.push(index);
                    goto_set.push(Item { production_id: item.production_id, dot_position: next_dot });
                }
            }

            // Cleanup bitset
            for &index in &indices_to_clear {
                let word_idx = index / 64;
                let bit_idx = index % 64;
                seen_items_bitset[word_idx] &= !(1 << bit_idx);
            }

            // Sort for canonical key
            goto_set.sort_unstable();
            // No dedup needed!

            let goto_id = if let Some(id) = item_set_map_fast.get(&goto_set) {
                *id
            } else {
                let new_id = StateID(next_state_id);
                next_state_id += 1;
                item_set_map_fast.insert(goto_set.clone(), new_id);
                worklist.push_back(goto_set);
                new_id
            };

            row.insert(Some(sym_id), Stage1Entry { kernel: Vec::new(), goto_id: Some(goto_id) });
        }

        if !bucket_none.is_empty() {
            row.insert(None, Stage1Entry { kernel: bucket_none.clone(), goto_id: None });
        }

        table[state_id.0] = row;
    }

    (table, item_set_map_fast)
}

#[time_it]
fn compute_final_table(
    stage_1_table: Stage1Table,
    item_set_map: &HashMap<Vec<Item>, StateID>,
    productions: &[Production],
    lhs_ids: &[usize],
    follow_sets: &[BTreeSet<Option<TerminalID>>],
    num_terminals: usize,
) -> (Table, StateID, StateID) {
    let start_production_id = 0;
    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_item_set = vec![initial_item];
    let start_state_id = *item_set_map.get(&initial_item_set).unwrap();
    let everything_state_id = start_state_id; // Placeholder if needed

    let mut final_table_map: Table = BTreeMap::new();

    // Precompute production metadata
    let prod_meta: Vec<(usize, NonTerminalID)> = productions
        .iter()
        .enumerate()
        .map(|(i, p)| (p.rhs.len(), NonTerminalID(lhs_ids[i])))
        .collect();

    for (state_idx, row) in stage_1_table.into_iter().enumerate() {
        let mut shifts: BTreeMap<TerminalID, StateID> = BTreeMap::new();
        let mut gotos: BTreeMap<NonTerminalID, Goto> = BTreeMap::new();
        
        // Reduces grouped by terminal. 
        // Using BTreeMap to ensure deterministic iteration order for the final table construction if needed,
        // though for intermediate accumulation HashMap might be faster. 
        // Given we iterate 0..num_terminals later, BTreeMap is fine or we can use a dense Vec.
        // Since num_terminals can be large, sparse map is better.
        let mut reduces_map: BTreeMap<TerminalID, Vec<ProductionID>> = BTreeMap::new();
        let mut eof_reduces: Vec<ProductionID> = Vec::new();

        // 1. Extract info from Stage 1 Row
        for (key, entry) in row {
            if let Some(goto_id) = entry.goto_id {
                // Shift or Goto
                if let Some(sym_id) = key {
                    if sym_id < num_terminals {
                        shifts.insert(TerminalID(sym_id), goto_id);
                    } else {
                        let nt_id = NonTerminalID(sym_id - num_terminals);
                        gotos.insert(nt_id, Goto { state_id: Some(goto_id), accept: false });
                    }
                }
            } else {
                // Reduce (Kernel items)
                for item in entry.kernel {
                    let prod_id = item.production_id;
                    let lhs = lhs_ids[prod_id];
                    let follow = &follow_sets[lhs];
                    
                    for lookahead in follow {
                        match lookahead {
                            Some(t) => reduces_map.entry(*t).or_default().push(ProductionID(prod_id)),
                            None => eof_reduces.push(ProductionID(prod_id)),
                        }
                    }
                }
            }
        }

        // 2. Apply EOF expansion (Stage 5 behavior: None in Follow -> Reduce on all terminals)
        if !eof_reduces.is_empty() {
            eof_reduces.sort_unstable();
            eof_reduces.dedup();
            // This is the expensive part if num_terminals is large.
            // However, we can optimize by not expanding yet, but handling it during the merge below.
            // But `shifts_and_reduces_full` requires explicit keys.
            // We iterate active terminals + all terminals if eof_reduces exists?
            // No, if eof_reduces exists, it affects ALL terminals.
            // So we must iterate 0..num_terminals.
            for i in 0..num_terminals {
                reduces_map.entry(TerminalID(i)).or_default().extend(eof_reduces.iter().cloned());
            }
        }

        // 3. Construct Final Row
        let mut shifts_and_reduces_full: ShiftsAndReducesFull = BTreeMap::new();
        
        // The set of terminals to process is the union of shifts and reduces.
        // If eof_reduces was not empty, this is all terminals.
        // Otherwise, it's sparse.
        let active_terminals: BTreeSet<TerminalID> = shifts.keys().chain(reduces_map.keys()).cloned().collect();

        for t in active_terminals {
            let shift_opt = shifts.get(&t).cloned();
            let mut reduc_list = reduces_map.remove(&t).unwrap_or_default();
            
            if shift_opt.is_none() && reduc_list.is_empty() {
                continue;
            }

            if !reduc_list.is_empty() {
                reduc_list.sort_unstable();
                reduc_list.dedup();
            }

            // Group reduces by len and NT (Stage 7 logic)
            let mut reduces_grouped: BTreeMap<usize, BTreeMap<NonTerminalID, Vec<ProductionID>>> = BTreeMap::new();
            for pid in reduc_list {
                let (len, nt) = prod_meta[pid.0];
                reduces_grouped.entry(len).or_default().entry(nt).or_default().push(pid);
            }

            let mut val = Stage7ShiftsAndReducesLookaheadValue::Split { 
                shift: shift_opt, 
                reduces: reduces_grouped 
            };
            val.simplify();
            shifts_and_reduces_full.insert(t, val);
        }

        // Handle Accept state
        if StateID(state_idx) == start_state_id {
             let start_nt_id = NonTerminalID(lhs_ids[start_production_id]);
             gotos.entry(start_nt_id).or_default().accept = true;
        }

        final_table_map.insert(StateID(state_idx), Row { shifts_and_reduces_full, gotos });
    }

    (final_table_map, start_state_id, everything_state_id)
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
    validate(productions).expect("Initial grammar validation failed");
    print_memory_usage("After validation");

    let start_production_id = 0;

    crate::debug!(2, "Removing productions with undefined non-terminals");
    let mut productions =
        remove_productions_with_undefined_nonterminals(&productions, &[start_production_id]);
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

    crate::debug!(2, "Stage 1 (LR(0) Automaton)");
    let (stage_1_table, item_set_map) = stage_1(&light_productions, &lhs_ids, num_terminals, num_nonterminals);
    print_memory_usage("After Stage 1");

    crate::debug!(2, "Computing First/Follow Sets");
    let first_sets = compute_first_sets_ids_with_lhs(&light_productions, &lhs_ids, num_terminals, num_nonterminals, &nullable_nts_ids);
    let follow_sets = compute_follow_sets_ids(&light_productions, &lhs_ids, &first_sets, &nullable_nts_ids, num_terminals, num_nonterminals, start_nt_id);
    print_memory_usage("After First/Follow");

    crate::debug!(2, "Computing Final Table (Merging Stages 2-8)");
    let (final_table_map, start_state_id, everything_state_id) = compute_final_table(
        stage_1_table,
        &item_set_map,
        &productions,
        &lhs_ids,
        &follow_sets,
        num_terminals
    );
    print_memory_usage("After Final Table");

    // Convert item_set_map back to BiBTreeMap for GLRParser
    let mut item_set_map_bi = BiBTreeMap::new();
    for (k, v) in item_set_map {
        item_set_map_bi.insert(k, v);
    }

    GLRParser::new(
        final_table_map,
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