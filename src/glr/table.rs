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
use std::hash::{BuildHasherDefault, Hasher};

const EVERYTHING: bool = false;

// --- Fast Hasher & BitSet ---

pub struct FxHasher {
    hash: usize,
}

impl Default for FxHasher {
    fn default() -> Self {
        Self { hash: 0 }
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        // Simple FxHash-like implementation
        for &byte in bytes {
            self.hash = (self.hash.rotate_left(5) ^ (byte as usize))
                .wrapping_mul(0x517cc1b727220a95);
        }
    }

    fn write_usize(&mut self, i: usize) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(0x517cc1b727220a95);
    }

    fn finish(&self) -> u64 {
        self.hash as u64
    }
}

type FxBuildHasher = BuildHasherDefault<FxHasher>;

#[derive(Clone)]
struct BitSet {
    data: Vec<u64>,
    size: usize,
}

impl BitSet {
    fn new(size: usize) -> Self {
        let words = (size + 63) / 64;
        Self {
            data: vec![0; words],
            size,
        }
    }

    #[inline]
    fn insert(&mut self, idx: usize) -> bool {
        if idx >= self.size {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        let mask = 1 << bit;
        if (self.data[word] & mask) == 0 {
            self.data[word] |= mask;
            true
        } else {
            false
        }
    }

    #[inline]
    fn contains(&self, idx: usize) -> bool {
        if idx >= self.size {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        (self.data[word] & (1 << bit)) != 0
    }

    fn clear(&mut self) {
        self.data.fill(0);
    }

    fn is_empty(&self) -> bool {
        self.data.iter().all(|&x| x == 0)
    }

    fn union_with(&mut self, other: &BitSet) {
        for (w1, w2) in self.data.iter_mut().zip(&other.data) {
            *w1 |= *w2;
        }
    }

    fn iter(&self) -> BitSetIter {
        BitSetIter {
            data: &self.data,
            word_idx: 0,
            current_word: if self.data.is_empty() { 0 } else { self.data[0] },
        }
    }
}

struct BitSetIter<'a> {
    data: &'a [u64],
    word_idx: usize,
    current_word: u64,
}

impl<'a> Iterator for BitSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let trailing = self.current_word.trailing_zeros();
                self.current_word &= !(1 << trailing);
                return Some(self.word_idx * 64 + trailing as usize);
            }

            self.word_idx += 1;
            if self.word_idx >= self.data.len() {
                return None;
            }
            self.current_word = self.data[self.word_idx];
        }
    }
}

// --- Table Structures ---

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
                if let Stage7ShiftsAndReducesLookaheadValue::Split { mut reduces, .. } = temp_self {
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
                        "Missing field variant for Stage7ShiftsAndReducesLookaheadValue"
                            .to_string()
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
                            .ok_or_else(|| {
                                "Missing field nonterminal_id for Reduce".to_string()
                            })
                            .and_then(NonTerminalID::from_json)?;
                        let len = obj
                            .remove("len")
                            .ok_or_else(|| "Missing field len for Reduce".to_string())
                            .and_then(usize::from_json)?;
                        let production_ids = obj
                            .remove("production_ids")
                            .ok_or_else(|| {
                                "Missing field production_ids for Reduce".to_string()
                            })
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
    pub default_reduce: Option<Stage7ShiftsAndReducesLookaheadValue>,
    pub gotos: BTreeMap<NonTerminalID, Goto>,
}

pub fn iter_rows(table: &Table) -> impl Iterator<Item = (&StateID, &Row)> {
    table.iter()
}

pub fn get_row(table: &Table, state_id: StateID) -> Option<&Row> {
    table.get(&state_id)
}

impl Row {
    pub fn get_shifts_and_reduces_for_terminal(
        &self,
        terminal_id: &TerminalID,
    ) -> Option<Stage7ShiftsAndReducesLookaheadValue> {
        self.shifts_and_reduces_full
            .get(terminal_id)
            .cloned()
            .or_else(|| self.default_reduce.clone())
    }

    pub fn get_shifts_and_reduces_map(
        &self,
    ) -> BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue> {
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
        let action = self
            .shifts_and_reduces_full
            .get(&terminal_id)
            .or(self.default_reduce.as_ref());

        if let Some(action) = action {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(state_id) => shiftfn(state_id),
                Stage7ShiftsAndReducesLookaheadValue::Reduce {
                    nonterminal_id,
                    len,
                    production_ids,
                } => reducefn(nonterminal_id, len, production_ids),
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    if let Some(state_id) = shift {
                        shiftfn(state_id);
                    }
                    for (len, nts) in reduces {
                        for (&nt_id, pids) in nts {
                            reducefn(&nt_id, len, pids);
                        }
                    }
                }
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
        obj.insert("default_reduce".to_string(), self.default_reduce.to_json());
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
                default_reduce: Option::<Stage7ShiftsAndReducesLookaheadValue>::from_json(
                    obj.remove("default_reduce").unwrap_or(JSONNode::Null),
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
) -> (Stage1Table, HashMap<Vec<Item>, StateID, FxBuildHasher>) {
    let start_production_id = 0;
    let initial_item = Item {
        production_id: start_production_id,
        dot_position: 0,
    };
    let initial_kernel = vec![initial_item];

    // Build index: for each non-terminal ID, which productions have it on the LHS.
    let mut prods_by_lhs: Vec<Vec<usize>> = vec![Vec::new(); num_nonterminals];
    for (pid, &lhs) in lhs_ids.iter().enumerate() {
        prods_by_lhs[lhs].push(pid);
    }

    // Precompute a dense index for every possible LR(0) item so we can store sets of items
    // in a reusable BitSet during closure computation.
    let mut item_base_offsets: Vec<usize> = Vec::with_capacity(light_productions.len());
    let mut total_items = 0usize;
    for rhs in light_productions {
        item_base_offsets.push(total_items);
        // For a production with RHS length L, there are L+1 LR(0) items (dot at 0..=L).
        total_items += rhs.len() + 1;
    }
    let mut used_items = BitSet::new(total_items);

    let mut item_set_map_fast: HashMap<Vec<Item>, StateID, FxBuildHasher> =
        HashMap::with_hasher(FxBuildHasher::default());
    let mut next_state_id = 0usize;

    let mut worklist: VecDeque<Vec<Item>> = VecDeque::new();
    let mut table: Stage1Table = Vec::new();

    item_set_map_fast.insert(initial_kernel.clone(), StateID(next_state_id));
    next_state_id += 1;
    worklist.push_back(initial_kernel);

    while let Some(kernel_items) = worklist.pop_front() {
        let state_id = *item_set_map_fast.get(&kernel_items).unwrap();

        if state_id.0 >= table.len() {
            table.resize(state_id.0 + 1, BTreeMap::new());
        }

        // --- Compute LR(0) closure for this kernel ---
        used_items.clear();
        let mut closure: Vec<Item> = Vec::new();
        let mut queue: VecDeque<Item> = VecDeque::new();

        // Seed closure with the kernel items.
        for &item in &kernel_items {
            let idx = item_base_offsets[item.production_id] + item.dot_position;
            if used_items.insert(idx) {
                closure.push(item);
                queue.push_back(item);
            }
        }

        // Standard LR(0) closure:
        // whenever the symbol after the dot is a non-terminal B,
        // add all items B -> .γ and repeat.
        while let Some(item) = queue.pop_front() {
            let rhs = &light_productions[item.production_id];
            if let Some(&sym) = rhs.get(item.dot_position) {
                if sym >= num_terminals {
                    let nt_id = sym - num_terminals;
                    for &prod_idx in &prods_by_lhs[nt_id] {
                        let new_item = Item {
                            production_id: prod_idx,
                            dot_position: 0,
                        };
                        let idx =
                            item_base_offsets[new_item.production_id] + new_item.dot_position;
                        if used_items.insert(idx) {
                            closure.push(new_item);
                            queue.push_back(new_item);
                        }
                    }
                }
            }
        }

        // --- Split closure items by the symbol under the dot ---
        let mut transitions: BTreeMap<Option<usize>, Vec<Item>> = BTreeMap::new();
        for item in &closure {
            let rhs = &light_productions[item.production_id];
            if let Some(&sym) = rhs.get(item.dot_position) {
                // Shift/goto on `sym`: kernel of the successor state has dot moved past `sym`.
                transitions
                    .entry(Some(sym))
                    .or_default()
                    .push(Item {
                        production_id: item.production_id,
                        dot_position: item.dot_position + 1,
                    });
            } else {
                // Dot at end => potential reduction in this state.
                transitions.entry(None).or_default().push(*item);
            }
        }

        // --- Build Stage 1 row for this state ---
        let mut row: Stage1Row = BTreeMap::new();

        for (sym_opt, mut items_vec) in transitions {
            if let Some(sym) = sym_opt {
                // Successor kernel for symbol `sym`.
                items_vec.sort_unstable(); // make kernel canonical for hashing
                // No dedup is necessary: each (prod_id, dot) pair is unique and shifting is injective.

                let goto_id = if let Some(&existing) = item_set_map_fast.get(&items_vec) {
                    existing
                } else {
                    let new_id = StateID(next_state_id);
                    next_state_id += 1;
                    item_set_map_fast.insert(items_vec.clone(), new_id);
                    worklist.push_back(items_vec.clone());
                    new_id
                };

                row.insert(
                    Some(sym),
                    Stage1Entry {
                        kernel: Vec::new(),
                        goto_id: Some(goto_id),
                    },
                );
            } else {
                // Reductions (items with dot at the end).
                items_vec.sort_unstable();
                items_vec.dedup();
                if !items_vec.is_empty() {
                    row.insert(
                        None,
                        Stage1Entry {
                            kernel: items_vec,
                            goto_id: None,
                        },
                    );
                }
            }
        }

        table[state_id.0] = row;
    }

    (table, item_set_map_fast)
}

#[derive(Clone)]
struct EntryBuilder {
    shift: Option<StateID>,
    reduces: Vec<ProductionID>,
}

#[time_it]
fn compute_final_table(
    stage_1_table: Stage1Table,
    item_set_map: &HashMap<Vec<Item>, StateID, FxBuildHasher>,
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
    let everything_state_id = start_state_id;

    let mut final_table_map: Table = BTreeMap::new();

    // Precompute production metadata: (RHS length, LHS non-terminal ID).
    let prod_meta: Vec<(usize, NonTerminalID)> = productions
        .iter()
        .enumerate()
        .map(|(i, p)| (p.rhs.len(), NonTerminalID(lhs_ids[i])))
        .collect();

    // Precompute FOLLOW terminals and EOF flags per non-terminal.
    let mut follow_terminals: Vec<Vec<usize>> = Vec::with_capacity(follow_sets.len());
    let mut follow_eof: Vec<bool> = Vec::with_capacity(follow_sets.len());

    for set in follow_sets {
        let mut terms = Vec::new();
        let mut has_eof = false;
        for opt in set {
            match opt {
                Some(t) => terms.push(t.0),
                None => has_eof = true,
            }
        }
        follow_terminals.push(terms);
        follow_eof.push(has_eof);
    }

    // Reusable structures
    let mut row_builder: Vec<EntryBuilder> =
        vec![EntryBuilder { shift: None, reduces: Vec::new() }; num_terminals];
    let mut dirty_terminals: Vec<usize> = Vec::with_capacity(num_terminals);
    let mut eof_reduces: Vec<ProductionID> = Vec::new();

    for (state_idx, row) in stage_1_table.into_iter().enumerate() {
        // Clear reusable structures for this row.
        for &t_idx in &dirty_terminals {
            row_builder[t_idx].shift = None;
            row_builder[t_idx].reduces.clear();
        }
        dirty_terminals.clear();
        eof_reduces.clear();

        let mut gotos: BTreeMap<NonTerminalID, Goto> = BTreeMap::new();

        // 1. Extract info from Stage 1 Row
        for (key, entry) in row {
            if let Some(goto_id) = entry.goto_id {
                if let Some(sym_id) = key {
                    if sym_id < num_terminals {
                        // Shift on terminal
                        let rb = &mut row_builder[sym_id];
                        if rb.shift.is_none() && rb.reduces.is_empty() {
                            dirty_terminals.push(sym_id);
                        }
                        rb.shift = Some(goto_id);
                    } else {
                        // Goto on non-terminal
                        let nt_id = NonTerminalID(sym_id - num_terminals);
                        gotos.insert(
                            nt_id,
                            Goto {
                                state_id: Some(goto_id),
                                accept: false,
                            },
                        );
                    }
                }
            } else {
                // Reduce (kernel items with dot at end)
                for item in entry.kernel {
                    let prod_id = item.production_id;
                    let lhs = lhs_ids[prod_id];

                    // Scatter reduces to terminals in FOLLOW(lhs).
                    for &t_idx in &follow_terminals[lhs] {
                        let rb = &mut row_builder[t_idx];
                        if rb.shift.is_none() && rb.reduces.is_empty() {
                            dirty_terminals.push(t_idx);
                        }
                        rb.reduces.push(ProductionID(prod_id));
                    }

                    if follow_eof[lhs] {
                        eof_reduces.push(ProductionID(prod_id));
                    }
                }
            }
        }

        // 2. Calculate default reduce (EOF) and flatten its production IDs once.
        let (default_reduce, default_pids) = if !eof_reduces.is_empty() {
            eof_reduces.sort_unstable();
            eof_reduces.dedup();

            let mut reduces_grouped: BTreeMap<
                usize,
                BTreeMap<NonTerminalID, Vec<ProductionID>>,
            > = BTreeMap::new();
            for pid in &eof_reduces {
                let (len, nt) = prod_meta[pid.0];
                reduces_grouped
                    .entry(len)
                    .or_default()
                    .entry(nt)
                    .or_default()
                    .push(*pid);
            }

            let mut val = Stage7ShiftsAndReducesLookaheadValue::Split {
                shift: None,
                reduces: reduces_grouped,
            };
            val.simplify();

            // `eof_reduces` is already sorted and deduped; reuse as flattened list.
            (Some(val), eof_reduces.clone())
        } else {
            (None, Vec::new())
        };

        // 3. Construct Final Row
        let mut shifts_and_reduces_full: ShiftsAndReducesFull = BTreeMap::new();

        for &t_idx in &dirty_terminals {
            let entry = &mut row_builder[t_idx];
            let t = TerminalID(t_idx);

            // Merge default reduce PIDs if needed.
            if !default_pids.is_empty() {
                entry.reduces.extend_from_slice(&default_pids);
            }

            if entry.reduces.is_empty() && entry.shift.is_none() {
                continue;
            }

            entry.reduces.sort_unstable();
            entry.reduces.dedup();

            // Group reduces.
            let mut reduces_grouped: BTreeMap<
                usize,
                BTreeMap<NonTerminalID, Vec<ProductionID>>,
            > = BTreeMap::new();
            for pid in entry.reduces.iter() {
                let (len, nt) = prod_meta[pid.0];
                reduces_grouped
                    .entry(len)
                    .or_default()
                    .entry(nt)
                    .or_default()
                    .push(*pid);
            }

            let mut val = Stage7ShiftsAndReducesLookaheadValue::Split {
                shift: entry.shift,
                reduces: reduces_grouped,
            };
            val.simplify();
            shifts_and_reduces_full.insert(t, val);
        }

        if StateID(state_idx) == start_state_id {
            let start_nt_id = NonTerminalID(lhs_ids[start_production_id]);
            gotos.entry(start_nt_id).or_default().accept = true;
        }

        final_table_map.insert(
            StateID(state_idx),
            Row {
                shifts_and_reduces_full,
                default_reduce,
                gotos,
            },
        );
    }

    (final_table_map, start_state_id, everything_state_id)
}

fn print_memory_usage(label: &str) {
    if let Some(usage) = memory_stats() {
        let physical_mem_mb = usage.physical_mem / 1024 / 1024;
        crate::debug!(4, "Memory usage at '{}': Physical: {} MB", label, physical_mem_mb);
    } else {
        crate::debug!(4, "Couldn't get memory usage at '{}'", label);
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
    crate::debug!(3, "Number of productions: {}", productions.len());
    print_memory_usage("Start of parser generation");

    crate::debug!(3, "Validating initial grammar");
    validate(productions).expect("Initial grammar validation failed");
    print_memory_usage("After validation");

    let start_production_id = 0;

    crate::debug!(3, "Removing productions with undefined non-terminals");
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

    crate::debug!(3, "Number of productions: {}", productions.len());
    print_memory_usage("Before Stage 1");

    // Prepare Light Productions (Global IDs)
    let num_terminals = terminal_map.len();
    let num_nonterminals = non_terminal_map.len();

    let light_productions: Vec<Vec<usize>> = productions
        .iter()
        .map(|p| {
            p.rhs
                .iter()
                .map(|s| match s {
                    Symbol::Terminal(t) => terminal_map.get_by_left(t).unwrap().0,
                    Symbol::NonTerminal(nt) => {
                        non_terminal_map.get_by_left(nt).unwrap().0 + num_terminals
                    }
                })
                .collect()
        })
        .collect();

    let lhs_ids: Vec<usize> = productions
        .iter()
        .map(|p| non_terminal_map.get_by_left(&p.lhs).unwrap().0)
        .collect();

    let nullable_nonterminals = compute_nullable_nonterminals(&productions);
    let nullable_nts_ids: HashSet<usize> = nullable_nonterminals
        .iter()
        .map(|nt| non_terminal_map.get_by_left(nt).unwrap().0)
        .collect();

    let start_nt_id = lhs_ids[0];

    crate::debug!(3, "Stage 1 (LR(0) Automaton)");
    let (stage_1_table, item_set_map) =
        stage_1(&light_productions, &lhs_ids, num_terminals, num_nonterminals);
    print_memory_usage("After Stage 1");

    crate::debug!(3, "Computing First/Follow Sets");
    let first_sets = compute_first_sets_ids_with_lhs(
        &light_productions,
        &lhs_ids,
        num_terminals,
        num_nonterminals,
        &nullable_nts_ids,
    );
    let follow_sets = compute_follow_sets_ids(
        &light_productions,
        &lhs_ids,
        &first_sets,
        &nullable_nts_ids,
        num_terminals,
        num_nonterminals,
        start_nt_id,
    );
    print_memory_usage("After First/Follow");

    crate::debug!(3, "Computing Final Table (Merging Stages 2-8)");
    let (final_table_map, start_state_id, everything_state_id) = compute_final_table(
        stage_1_table,
        &item_set_map,
        &productions,
        &lhs_ids,
        &follow_sets,
        num_terminals,
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
