use super::items::Item;
use crate::glr::analyze::{
    create_unique_name_generator, inline_null_productions, inline_unit_productions,
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
use std::hash::{BuildHasherDefault, Hasher};

const EVERYTHING: bool = false;

pub type Table = BTreeMap<StateID, Row>;

// --- Fast Hasher (FxHash variant) ---
pub struct FxHasher {
    hash: u64,
}

impl Default for FxHasher {
    #[inline]
    fn default() -> FxHasher {
        FxHasher { hash: 0 }
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.hash;
        let mut i = 0;
        while i + 8 <= bytes.len() {
            let mut k = 0u64;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(i), &mut k as *mut _ as *mut u8, 8);
            }
            hash = (hash.rotate_left(5) ^ k).wrapping_mul(0x517cc1b727220a95);
            i += 8;
        }
        if i < bytes.len() {
            let mut k = 0u64;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(i), &mut k as *mut _ as *mut u8, bytes.len() - i);
            }
            hash = (hash.rotate_left(5) ^ k).wrapping_mul(0x517cc1b727220a95);
        }
        self.hash = hash;
    }
    
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.hash = (self.hash.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
}

type FxBuildHasher = BuildHasherDefault<FxHasher>;

// --- Intermediate Structures ---

/// A compact representation of a state in the LR(0) automaton.
struct LR0State {
    /// Transitions to other states. Key is symbol ID (terminal < num_terms, non-terminal >= num_terms).
    transitions: Vec<(usize, StateID)>,
    /// Reduction items present in this state (dot at end).
    reductions: Vec<Item>,
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
                let variant = obj.remove("variant").ok_or_else(|| "Missing variant".to_string())?.as_str().to_string();
                 match variant.as_str() {
                    "Shift" => {
                         let state_id = StateID::from_json(obj.remove("state_id").unwrap())?;
                         Ok(Self::Shift(state_id))
                    },
                    "Reduce" => {
                        let nonterminal_id = NonTerminalID::from_json(obj.remove("nonterminal_id").unwrap())?;
                        let len = usize::from_json(obj.remove("len").unwrap())?;
                        let production_ids = Vec::from_json(obj.remove("production_ids").unwrap())?;
                        Ok(Self::Reduce { nonterminal_id, len, production_ids })
                    },
                    "Split" => {
                         let shift = Option::from_json(obj.remove("shift").unwrap())?;
                         let reduces = BTreeMap::from_json(obj.remove("reduces").unwrap())?;
                         Ok(Self::Split { shift, reduces })
                    },
                    _ => Err("Unknown variant".to_string())
                 }
            }
            _ => Err("Expected Object".to_string())
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
    pub fn handle_shifts_and_reduces_for_terminal(
        &self,
        terminal_id: TerminalID,
        shiftfn: impl FnOnce(&StateID),
        mut reducefn: impl FnMut(&NonTerminalID, &usize, &Vec<ProductionID>),
    ) {
        if let Some(action) = self.shifts_and_reduces_full.get(&terminal_id) {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(state_id) => shiftfn(state_id),
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => reducefn(nonterminal_id, len, production_ids),
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    if let Some(state_id) = shift {
                        shiftfn(state_id);
                    }
                    for (_len, nts) in reduces {
                        for (nt_id, pids) in nts {
                            reducefn(nt_id, _len, pids);
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
        obj.insert("shifts_and_reduces_full".to_string(), self.shifts_and_reduces_full.to_json());
        obj.insert("gotos".to_string(), self.gotos.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
         match node {
            JSONNode::Object(mut obj) => Ok(Row {
                shifts_and_reduces_full: ShiftsAndReducesFull::from_json(obj.remove("shifts_and_reduces_full").unwrap())?,
                gotos: BTreeMap::from_json(obj.remove("gotos").unwrap())?,
            }),
            _ => Err("Expected Object".to_string())
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
                state_id: Option::from_json(obj.remove("state_id").unwrap())?,
                accept: bool::from_json(obj.remove("accept").unwrap())?,
            }),
            _ => Err("Expected Object".to_string())
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[time_it]
fn stage_1(
    light_productions: &[Vec<usize>],
    lhs_ids: &[usize],
    num_terminals: usize,
    num_nonterminals: usize,
) -> (Vec<LR0State>, HashMap<Vec<Item>, StateID, FxBuildHasher>) {
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

    // 2. Precompute Closure Cache (Light)
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
    let mut item_set_map_fast: HashMap<Vec<Item>, StateID, FxBuildHasher> = HashMap::with_hasher(FxBuildHasher::default());
    let mut next_state_id = 0;

    let mut worklist = VecDeque::new();
    let mut table: Vec<LR0State> = Vec::new();

    item_set_map_fast.insert(initial_item_set.clone(), StateID(next_state_id));
    next_state_id += 1;
    worklist.push_back(initial_item_set);

    let total_symbols = num_terminals + num_nonterminals;
    let mut buckets: Vec<Vec<Item>> = vec![Vec::new(); total_symbols];
    let mut bucket_none: Vec<Item> = Vec::new();
    let mut active_symbols: Vec<usize> = Vec::with_capacity(128);
    let mut processed_nts: Vec<bool> = vec![false; num_nonterminals];
    let mut touched_nts: Vec<usize> = Vec::with_capacity(128);
    
    // Scratch buffer for constructing goto sets to avoid reallocating
    let mut goto_set_buffer: Vec<Item> = Vec::with_capacity(1024);

    while let Some(item_set) = worklist.pop_front() {
        // Clear tracking structures
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

        let mut transitions: Vec<(usize, StateID)> = Vec::with_capacity(active_symbols.len());
        
        for &sym_id in &active_symbols {
            let items_in_bucket = &buckets[sym_id];
            
            // Construct goto_set reusing buffer
            goto_set_buffer.clear();
            goto_set_buffer.extend(items_in_bucket.iter().map(|item| {
                 Item { production_id: item.production_id, dot_position: item.dot_position + 1 }
            }));
            goto_set_buffer.sort_unstable();
            goto_set_buffer.dedup();

            let goto_id = if let Some(id) = item_set_map_fast.get(&goto_set_buffer) {
                *id
            } else {
                let new_id = StateID(next_state_id);
                next_state_id += 1;
                // Must clone to insert key
                item_set_map_fast.insert(goto_set_buffer.clone(), new_id);
                worklist.push_back(goto_set_buffer.clone());
                new_id
            };

            transitions.push((sym_id, goto_id));
        }

        // Store
        table.push(LR0State {
            transitions,
            reductions: bucket_none.clone(),
        });
    }

    (table, item_set_map_fast)
}

#[time_it]
fn finalize_table(
    lr0_table: Vec<LR0State>,
    follow_sets: &[BTreeSet<Option<TerminalID>>],
    item_set_map: &HashMap<Vec<Item>, StateID, FxBuildHasher>,
    lhs_ids: &[usize],
    num_terminals: usize,
    start_production_id: usize,
    prod_rhs_lens: &[usize],
) -> (Table, StateID, StateID) {
    let mut final_table = BTreeMap::new();
    
    let initial_item = Item { production_id: start_production_id, dot_position: 0 };
    let start_state_id = *item_set_map.get(&vec![initial_item]).unwrap();
    let start_non_terminal_id = NonTerminalID(lhs_ids[start_production_id]);

    for (idx, state) in lr0_table.into_iter().enumerate() {
        let state_id = StateID(idx);
        let mut shifts_and_reduces_full: ShiftsAndReducesFull = BTreeMap::new();
        let mut gotos: BTreeMap<NonTerminalID, Goto> = BTreeMap::new();

        // Process Transitions (Shifts and Gotos)
        for (sym_id, target_state) in state.transitions {
            if sym_id < num_terminals {
                let term = TerminalID(sym_id);
                // Insert Shift. In case of existing shift, it's a duplicate logic or error, but LR0 should be unique per symbol
                shifts_and_reduces_full.insert(term, Stage7ShiftsAndReducesLookaheadValue::Shift(target_state));
            } else {
                let nt = NonTerminalID(sym_id - num_terminals);
                gotos.insert(nt, Goto { state_id: Some(target_state), accept: false });
            }
        }

        // Process Reductions (SLR Logic)
        for item in state.reductions {
            let lhs = lhs_ids[item.production_id];
            let len = prod_rhs_lens[item.production_id];
            let pid = ProductionID(item.production_id);
            let nt_id = NonTerminalID(lhs);

            // Get Follow(LHS)
            for lookahead in &follow_sets[lhs] {
                let targets = if let Some(t) = lookahead {
                    vec![*t]
                } else {
                    (0..num_terminals).map(TerminalID).collect()
                };

                for term in targets {
                    shifts_and_reduces_full.entry(term)
                        .and_modify(|e| {
                            match e {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(sid) => {
                                    // Shift-Reduce conflict -> Split
                                    let mut map = BTreeMap::new();
                                    map.entry(len).or_default().entry(nt_id).or_default().push(pid);
                                    *e = Stage7ShiftsAndReducesLookaheadValue::Split {
                                        shift: Some(*sid),
                                        reduces: map,
                                    };
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len: r_len, production_ids } => {
                                    // Reduce-Reduce conflict -> Split
                                    let mut map = BTreeMap::new();
                                    map.entry(*r_len).or_default().entry(*nonterminal_id).or_default().extend(production_ids.iter().cloned());
                                    map.entry(len).or_default().entry(nt_id).or_default().push(pid);
                                    
                                    *e = Stage7ShiftsAndReducesLookaheadValue::Split {
                                        shift: None,
                                        reduces: map,
                                    };
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                                    reduces.entry(len).or_default().entry(nt_id).or_default().push(pid);
                                }
                            }
                        })
                        .or_insert_with(|| {
                            Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                nonterminal_id: nt_id,
                                len,
                                production_ids: vec![pid],
                            }
                        });
                }
            }
        }
        
        // Simplify Splits
        for val in shifts_and_reduces_full.values_mut() {
             if let Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } = val {
                 for inner in reduces.values_mut() {
                     for vec in inner.values_mut() {
                         vec.sort_unstable();
                         vec.dedup();
                     }
                 }
             }
             val.simplify();
        }

        // Handle Accept state special case
        if state_id == start_state_id {
             gotos.entry(start_non_terminal_id).and_modify(|g| g.accept = true).or_insert(Goto { state_id: None, accept: true });
        }

        final_table.insert(state_id, Row { shifts_and_reduces_full, gotos });
    }

    (final_table, start_state_id, start_state_id) // Assuming everything_state_id == start_state_id for now
}

fn print_memory_usage(label: &str) {
    if let Some(usage) = memory_stats() {
        let physical_mem_mb = usage.physical_mem / 1024 / 1024;
        crate::debug!(2, "Memory usage at '{}': Physical: {} MB", label, physical_mem_mb);
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

    // Grammar prep
    validate(productions).expect("Initial grammar validation failed");
    let mut productions = remove_productions_with_undefined_nonterminals(&productions, &[0]);
    let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut unqiue_name_generator = create_unique_name_generator(&nonterminals);
    crate::glr::analyze::resolve_direct_right_recursion(&mut productions, &mut unqiue_name_generator);
    productions = inline_null_productions(&productions);

    let mut next_non_terminal_id = non_terminal_map.len();
    for p in &productions {
        if !non_terminal_map.contains_left(&p.lhs) {
            non_terminal_map.insert(p.lhs.clone(), NonTerminalID(next_non_terminal_id));
            next_non_terminal_id += 1;
        }
    }

    // ID Mapping
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
    let prod_rhs_lens: Vec<usize> = productions.iter().map(|p| p.rhs.len()).collect();

    // Compute Lookaheads First
    crate::debug!(2, "Computing First/Follow sets");
    let nullable_nonterminals = compute_nullable_nonterminals(&productions);
    let nullable_nts_ids: HashSet<usize> = nullable_nonterminals.iter().map(|nt| {
        non_terminal_map.get_by_left(nt).unwrap().0
    }).collect();
    let first_sets = compute_first_sets_ids_with_lhs(&light_productions, &lhs_ids, num_terminals, num_nonterminals, &nullable_nts_ids);
    let follow_sets = compute_follow_sets_ids(&light_productions, &lhs_ids, &first_sets, &nullable_nts_ids, num_terminals, num_nonterminals, lhs_ids[0]);

    // Stage 1
    crate::debug!(2, "Stage 1: Generating Automaton");
    let (lr0_table, item_set_map) = stage_1(&light_productions, &lhs_ids, num_terminals, num_nonterminals);
    print_memory_usage("After Stage 1");

    // Finalize
    crate::debug!(2, "Finalizing Table (merging Stages 2-8)");
    let (final_table, start_state_id, everything_state_id) = finalize_table(
        lr0_table,
        &follow_sets,
        &item_set_map,
        &lhs_ids,
        num_terminals,
        0, // start_production_id
        &prod_rhs_lens,
    );
    print_memory_usage("After Finalize");

    // BiMap Conversion
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
