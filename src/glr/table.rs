use super::items::Item;
use crate::glr::analyze::{
    create_unique_name_generator, inline_null_productions,
    remove_productions_with_undefined_nonterminals, validate,
};
use crate::glr::minimizer::{substitute_single_productions_and_report, remove_productions_for_nts};
use crate::glr::automaton::{
    compute_first_sets_ids_with_lhs, compute_follow_sets_ids, compute_nullable_nonterminals,
};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::{ActionFn, GLRParser};
use crate::json_serialization::{JSONConvertible, JSONNode};
pub use crate::types::TerminalID;
use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
use memory_stats::memory_stats;
use profiler_macro::time_it;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};

/// Transform nullable terminals into optional non-terminals.
/// 
/// For each terminal in `nullable_terminals`, this creates a new non-terminal
/// `{TerminalName}Opt` with two productions:
/// - `{TerminalName}Opt -> terminal`
/// - `{TerminalName}Opt -> ε`
/// 
/// Then replaces all occurrences of the terminal in productions with the new non-terminal.
/// 
/// This is required because the tokenizer can produce zero-width matches for nullable
/// terminals, and the GLR parser needs to handle these as optional non-terminals to
/// correctly parse inputs where the nullable terminal matches empty string.
fn transform_nullable_terminals(
    productions: &[Production],
    nullable_terminals: &HashSet<Terminal>,
    existing_nonterminals: &BTreeSet<NonTerminal>,
) -> (Vec<Production>, HashMap<Terminal, NonTerminal>) {
    if nullable_terminals.is_empty() {
        return (productions.to_vec(), HashMap::new());
    }
    
    // Generate unique names for optional non-terminals
    let mut all_names: HashSet<String> = existing_nonterminals.iter().map(|nt| nt.0.clone()).collect();
    let mut terminal_to_opt_nt: HashMap<Terminal, NonTerminal> = HashMap::new();
    let mut new_productions: Vec<Production> = Vec::new();
    
    for terminal in nullable_terminals {
        // Generate a unique name for this terminal's optional wrapper
        let base_name = match terminal {
            Terminal::RegexName(name) => name.trim_matches('"').to_string(),
            Terminal::Literal(bytes) => {
                // For literals, create a readable name
                if bytes.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_') {
                    String::from_utf8_lossy(bytes).to_string()
                } else {
                    format!("Lit{:x}", bytes.iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64)))
                }
            }
        };
        
        let mut opt_name = format!("{}Opt", base_name);
        let mut counter = 0;
        while all_names.contains(&opt_name) {
            counter += 1;
            opt_name = format!("{}Opt{}", base_name, counter);
        }
        all_names.insert(opt_name.clone());
        
        let opt_nt = NonTerminal(opt_name);
        terminal_to_opt_nt.insert(terminal.clone(), opt_nt.clone());
        
        // Add productions for the optional non-terminal:
        // OptNT -> terminal
        new_productions.push(Production {
            lhs: opt_nt.clone(),
            rhs: vec![Symbol::Terminal(terminal.clone())],
        });
        // OptNT -> ε
        new_productions.push(Production {
            lhs: opt_nt,
            rhs: vec![],
        });
        
        crate::debug!(5, "  {} -> {}", terminal, terminal_to_opt_nt.get(terminal).unwrap().0);
    }
    
    // Transform existing productions: replace nullable terminals with their optional non-terminals
    let transformed_productions: Vec<Production> = productions.iter().map(|prod| {
        let transformed_rhs: Vec<Symbol> = prod.rhs.iter().map(|sym| {
            if let Symbol::Terminal(t) = sym {
                if let Some(opt_nt) = terminal_to_opt_nt.get(t) {
                    return Symbol::NonTerminal(opt_nt.clone());
                }
            }
            sym.clone()
        }).collect();
        
        Production {
            lhs: prod.lhs.clone(),
            rhs: transformed_rhs,
        }
    }).collect();
    
    // Combine transformed productions with new optional productions
    let mut all_productions = transformed_productions;
    all_productions.extend(new_productions);
    
    (all_productions, terminal_to_opt_nt)
}

/// Detect terminals that are "whitespace-like" (always optional).
/// 
/// A terminal T is whitespace-like if for every production `A → α T β`,
/// there exists another production `A → α β`. In other words, T can always
/// be skipped without changing the grammar's accepted language.
/// 
/// These terminals can be treated as "ignore" terminals by the parser,
/// allowing them to appear anywhere without affecting parsing.
/// 
/// Returns the set of terminal IDs that are whitespace-like.
/// 
/// NOTE: This function is currently disabled (returns empty set) because
/// the simple heuristic of "always optional" is too broad - it incorrectly
/// classifies meaningful optional content (like in `A*` patterns) as whitespace.
/// 
/// A better heuristic would need to distinguish between:
/// 1. True whitespace (appears ubiquitously, has no semantic meaning)
/// 2. Optional content (meaningful but happens to be optional in some positions)
/// 
/// For now, ignore terminals must be explicitly specified via the grammar
/// (e.g., `#![ignore(WS)]` directive) rather than auto-detected.
#[allow(unused_variables)]
pub fn detect_whitespace_like_terminals(
    productions: &[Production],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
) -> HashSet<TerminalID> {
    // Auto-detection disabled - see NOTE above
    HashSet::new()
}

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

type Stage1Table = Vec<Stage1Row>;
type Stage1Row = BTreeMap<Option<usize>, Stage1Entry>;

#[derive(Debug, Clone)]
struct Stage1Entry {
    /// Items in this state whose symbol under the dot is `symbol`.
    kernel: Vec<Item>,
    /// ID of the state reached by shifting over that symbol.
    goto_id: Option<StateID>,
}

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
        match self {
            Stage7ShiftsAndReducesLookaheadValue::Shift(state_id) => {
                // ["S", state_id]
                JSONNode::Array(vec![
                    JSONNode::String("S".to_string()),
                    state_id.to_json(),
                ])
            }
            Stage7ShiftsAndReducesLookaheadValue::Reduce {
                nonterminal_id,
                len,
                production_ids: _, // Drop production_ids from serialization
            } => {
                // ["R", nonterminal_id, len]
                JSONNode::Array(vec![
                    JSONNode::String("R".to_string()),
                    nonterminal_id.to_json(),
                    len.to_json(),
                ])
            }
            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                // ["X", shift_opt, [[nt_id, len], ...]]
                // Flatten reduces map into array of [nt_id, len] pairs
                let mut reduce_pairs = Vec::new();
                for (len, nts) in reduces {
                    for (nt_id, _pids) in nts {
                        reduce_pairs.push(JSONNode::Array(vec![
                            nt_id.to_json(),
                            len.to_json(),
                        ]));
                    }
                }
                JSONNode::Array(vec![
                    JSONNode::String("X".to_string()),
                    shift.to_json(),
                    JSONNode::Array(reduce_pairs),
                ])
            }
        }
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(mut arr) if !arr.is_empty() => {
                let tag = String::from_json(arr.remove(0))?;
                match tag.as_str() {
                    "S" => {
                        if arr.len() != 1 {
                            return Err(format!(
                                "Expected [\"S\", state_id], got {} elements",
                                arr.len() + 1
                            ));
                        }
                        let state_id = StateID::from_json(arr.remove(0))?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Shift(state_id))
                    }
                    "R" => {
                        if arr.len() != 2 {
                            return Err(format!(
                                "Expected [\"R\", nt_id, len], got {} elements",
                                arr.len() + 1
                            ));
                        }
                        let nonterminal_id = NonTerminalID::from_json(arr.remove(0))?;
                        let len = usize::from_json(arr.remove(0))?;
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                            nonterminal_id,
                            len,
                            production_ids: Vec::new(), // Reconstructed as empty
                        })
                    }
                    "X" => {
                        if arr.len() != 2 {
                            return Err(format!(
                                "Expected [\"X\", shift_opt, reduces], got {} elements",
                                arr.len() + 1
                            ));
                        }
                        let shift = Option::<StateID>::from_json(arr.remove(0))?;
                        let reduce_array = arr.remove(0);
                        
                        // Parse [[nt_id, len], ...] back into nested BTreeMap
                        let mut reduces: BTreeMap<usize, BTreeMap<NonTerminalID, Vec<ProductionID>>> = BTreeMap::new();
                        match reduce_array {
                            JSONNode::Array(pairs) => {
                                for pair in pairs {
                                    match pair {
                                        JSONNode::Array(mut p) if p.len() == 2 => {
                                            let nt_id = NonTerminalID::from_json(p.remove(0))?;
                                            let len = usize::from_json(p.remove(0))?;
                                            reduces.entry(len).or_default().insert(nt_id, Vec::new());
                                        }
                                        _ => return Err("Expected [nt_id, len] pair in Split reduces".to_string()),
                                    }
                                }
                            }
                            _ => return Err("Expected array of reduces in Split".to_string()),
                        }
                        
                        Ok(Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces })
                    }
                    _ => Err(format!(
                        "Unknown variant tag '{}' for Stage7ShiftsAndReducesLookaheadValue",
                        tag
                    )),
                }
            }
            _ => Err("Expected JSONNode::Array for Stage7ShiftsAndReducesLookaheadValue".to_string()),
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
        // [[[term_id, Action], ...], [[nt_id, Goto], ...], default_reduce]
        let shifts_pairs: Vec<JSONNode> = self
            .shifts_and_reduces_full
            .iter()
            .map(|(tid, action)| JSONNode::Array(vec![tid.to_json(), action.to_json()]))
            .collect();
        let gotos_pairs: Vec<JSONNode> = self
            .gotos
            .iter()
            .map(|(ntid, goto)| JSONNode::Array(vec![ntid.to_json(), goto.to_json()]))
            .collect();
        JSONNode::Array(vec![
            JSONNode::Array(shifts_pairs),
            JSONNode::Array(gotos_pairs),
            self.default_reduce.to_json(),
        ])
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(mut arr) if arr.len() == 3 => {
                // Parse shifts_and_reduces_full
                let mut shifts_and_reduces_full = BTreeMap::new();
                match arr.remove(0) {
                    JSONNode::Array(pairs) => {
                        for pair in pairs {
                            match pair {
                                JSONNode::Array(mut p) if p.len() == 2 => {
                                    let tid = TerminalID::from_json(p.remove(0))?;
                                    let action = Stage7ShiftsAndReducesLookaheadValue::from_json(p.remove(0))?;
                                    shifts_and_reduces_full.insert(tid, action);
                                }
                                _ => return Err("Expected [term_id, action] pair in Row shifts".to_string()),
                            }
                        }
                    }
                    _ => return Err("Expected array of shift pairs in Row".to_string()),
                }
                
                // Parse gotos
                let mut gotos = BTreeMap::new();
                match arr.remove(0) {
                    JSONNode::Array(pairs) => {
                        for pair in pairs {
                            match pair {
                                JSONNode::Array(mut p) if p.len() == 2 => {
                                    let ntid = NonTerminalID::from_json(p.remove(0))?;
                                    let goto = Goto::from_json(p.remove(0))?;
                                    gotos.insert(ntid, goto);
                                }
                                _ => return Err("Expected [nt_id, goto] pair in Row gotos".to_string()),
                            }
                        }
                    }
                    _ => return Err("Expected array of goto pairs in Row".to_string()),
                }
                
                // Parse default_reduce
                let default_reduce = Option::<Stage7ShiftsAndReducesLookaheadValue>::from_json(arr.remove(0))?;
                
                Ok(Row {
                    shifts_and_reduces_full,
                    default_reduce,
                    gotos,
                })
            }
            _ => Err("Expected JSONNode::Array of length 3 for Row".to_string()),
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
        // [state_id_opt, accept_bool]
        JSONNode::Array(vec![
            self.state_id.to_json(),
            self.accept.to_json(),
        ])
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(mut arr) if arr.len() == 2 => {
                let state_id = Option::<StateID>::from_json(arr.remove(0))?;
                let accept = bool::from_json(arr.remove(0))?;
                Ok(Goto { state_id, accept })
            }
            _ => Err("Expected JSONNode::Array of length 2 for Goto".to_string()),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct StateID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct ProductionID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct NonTerminalID(pub usize);

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

    let mut prods_by_lhs: Vec<Vec<usize>> = vec![Vec::new(); num_nonterminals];
    for (pid, &lhs) in lhs_ids.iter().enumerate() {
        prods_by_lhs[lhs].push(pid);
    }

    let mut item_base_offsets: Vec<usize> = Vec::with_capacity(light_productions.len());
    let mut total_items = 0usize;
    for rhs in light_productions {
        item_base_offsets.push(total_items);
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

        used_items.clear();
        let mut closure: Vec<Item> = Vec::new();
        let mut queue: VecDeque<Item> = VecDeque::new();

        for &item in &kernel_items {
            let idx = item_base_offsets[item.production_id] + item.dot_position;
            if used_items.insert(idx) {
                closure.push(item);
                queue.push_back(item);
            }
        }

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

        let mut transitions: BTreeMap<Option<usize>, Vec<Item>> = BTreeMap::new();
        for item in &closure {
            let rhs = &light_productions[item.production_id];
            if let Some(&sym) = rhs.get(item.dot_position) {
                transitions
                    .entry(Some(sym))
                    .or_default()
                    .push(Item {
                        production_id: item.production_id,
                        dot_position: item.dot_position + 1,
                    });
            } else {
                transitions.entry(None).or_default().push(*item);
            }
        }

        let mut row: Stage1Row = BTreeMap::new();

        for (sym_opt, mut items_vec) in transitions {
            if let Some(sym) = sym_opt {
                items_vec.sort_unstable();
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
    let substring_state_id = start_state_id;

    let mut final_table_map: Table = BTreeMap::new();

    let prod_meta: Vec<(usize, NonTerminalID)> = productions
        .iter()
        .enumerate()
        .map(|(i, p)| (p.rhs.len(), NonTerminalID(lhs_ids[i])))
        .collect();

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

    let mut row_builder: Vec<EntryBuilder> =
        vec![EntryBuilder { shift: None, reduces: Vec::new() }; num_terminals];
    let mut dirty_terminals: Vec<usize> = Vec::with_capacity(num_terminals);
    let mut eof_reduces: Vec<ProductionID> = Vec::new();

    for (state_idx, row) in stage_1_table.into_iter().enumerate() {
        for &t_idx in &dirty_terminals {
            row_builder[t_idx].shift = None;
            row_builder[t_idx].reduces.clear();
        }
        dirty_terminals.clear();
        eof_reduces.clear();

        let mut gotos: BTreeMap<NonTerminalID, Goto> = BTreeMap::new();

        for (key, entry) in row {
            if let Some(goto_id) = entry.goto_id {
                if let Some(sym_id) = key {
                    if sym_id < num_terminals {
                        let rb = &mut row_builder[sym_id];
                        if rb.shift.is_none() && rb.reduces.is_empty() {
                            dirty_terminals.push(sym_id);
                        }
                        rb.shift = Some(goto_id);
                    } else {
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
                for item in entry.kernel {
                    let prod_id = item.production_id;
                    let lhs = lhs_ids[prod_id];

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

            (Some(val), eof_reduces.clone())
        } else {
            (None, Vec::new())
        };

        let mut shifts_and_reduces_full: ShiftsAndReducesFull = BTreeMap::new();

        for &t_idx in &dirty_terminals {
            let entry = &mut row_builder[t_idx];
            let t = TerminalID(t_idx);

            if !default_pids.is_empty() {
                entry.reduces.extend_from_slice(&default_pids);
            }

            if entry.reduces.is_empty() && entry.shift.is_none() {
                continue;
            }

            entry.reduces.sort_unstable();
            entry.reduces.dedup();

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

    (final_table_map, start_state_id, substring_state_id)
}

fn print_memory_usage(label: &str) {
    if let Some(usage) = memory_stats() {
        let physical_mem_mb = usage.physical_mem / 1024 / 1024;
        crate::debug!(5, "Mem: {} MB ({})", physical_mem_mb, label);
    }
}

#[time_it]
fn generate_glr_parser_with_maps(
    productions: &[Production],
    terminal_map: BiBTreeMap<Terminal, TerminalID>,
    mut non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    actions: BTreeMap<NonTerminal, ActionFn>,
    ignore_terminal_ids: HashSet<TerminalID>,
) -> GLRParser {
    crate::debug!(5, "Number of productions: {}", productions.len());
    print_memory_usage("Start of parser generation");

    crate::debug!(4, "Validating initial grammar");
    validate(productions).expect("Initial grammar validation failed");
    print_memory_usage("After validation");

    let start_production_id = 0;

    crate::debug!(4, "Removing productions with undefined non-terminals");
    let mut productions =
        remove_productions_with_undefined_nonterminals(&productions, &[start_production_id]);
    print_memory_usage("After removing undefined");

    // Inline null productions FIRST to expose hidden right recursion.
    // Hidden right recursion (A -> α A β where β is nullable) becomes
    // direct right recursion (A -> α A) after this step.
    crate::debug!(4, "Inlining null productions (pass 1: expose hidden right recursion)");
    productions = inline_null_productions(&productions);
    print_memory_usage("After inlining null productions (pass 1)");

    // Now resolve direct right recursion (including formerly hidden ones).
    // This transforms A -> α A into A -> A' α, A' -> ε | A' α,
    // converting right recursion to left recursion.
    // Per the theorem from "Even Faster Generalized LR Parsing",
    // eliminating right recursion (along with hidden left recursion,
    // which is checked during validation) guarantees bounded reductions.
    let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut unique_name_generator = create_unique_name_generator(&nonterminals);

    // Iterate right recursion elimination until no more right recursion exists.
    // This is needed because productions like A -> α A β A require multiple passes:
    // Pass 1: A -> α A β A becomes A -> A'(α A β), A' -> A'(α A β) | ε
    // But A'(α A β) still contains A!
    // After inline_null_productions, we need to re-check for right recursion.
    const MAX_RIGHT_RECURSION_PASSES: usize = 10;
    for pass in 0..MAX_RIGHT_RECURSION_PASSES {
        crate::debug!(4, "Right recursion elimination pass {}", pass + 1);
        
        // Check if any right recursion remains
        let right_recursion_errors = crate::glr::analyze::check_for_right_recursion(&productions);
        if right_recursion_errors.is_empty() {
            crate::debug!(4, "No right recursion detected, done after {} passes", pass + 1);
            break;
        }
        crate::debug!(4, "Found {} right recursion patterns, continuing...", right_recursion_errors.len());
        for err in &right_recursion_errors {
            crate::debug!(5, "  {}", err);
        }
        
        // First resolve indirect right recursion by inlining
        crate::debug!(5, "Resolving indirect right recursion (pass {})", pass + 1);
        crate::glr::analyze::resolve_indirect_right_recursion(
            &mut productions,
            &mut unique_name_generator,
        );

        // Then resolve direct right recursion
        crate::debug!(5, "Resolving direct right recursion (pass {})", pass + 1);
        crate::glr::analyze::resolve_direct_right_recursion(
            &mut productions,
            &mut unique_name_generator,
        );

        // Inline null productions because right recursion resolution
        // may have introduced new nullable non-terminals (like A' -> ε).
        crate::debug!(5, "Inlining null productions (pass {})", pass + 1);
        productions = inline_null_productions(&productions);
        
        if pass == MAX_RIGHT_RECURSION_PASSES - 1 {
            crate::log_warn!("Right recursion elimination did not converge after {} passes", MAX_RIGHT_RECURSION_PASSES);
        }
    }
    print_memory_usage("After right recursion elimination");

    // Eliminate hidden left recursion
    // This is required for the bounded reductions theorem (Aycock et al., 1999)
    crate::debug!(4, "Eliminating hidden left recursion");
    crate::glr::analyze::eliminate_hidden_left_recursion(&mut productions);
    
    // Re-inline null productions after hidden left recursion elimination
    productions = inline_null_productions(&productions);
    print_memory_usage("After hidden left recursion elimination");

    // Re-validate after transformations to catch any newly introduced issues
    crate::debug!(4, "Validating grammar after transformations");
    let post_transform_errors = crate::glr::analyze::check_for_length_1_recursion(&productions);
    let left_recursion_errors = crate::glr::analyze::check_for_left_nullable_left_recursion(&productions);
    let indirect_errors = crate::glr::analyze::check_for_indirect_hidden_left_recursion(&productions);
    let right_recursion_errors = crate::glr::analyze::check_for_right_recursion(&productions);
    
    // Count total warnings
    let total_warnings = post_transform_errors.len() + left_recursion_errors.len() + 
                         indirect_errors.len() + right_recursion_errors.len();
    
    if total_warnings > 0 {
        // At level 1-4: just show summary counts
        if crate::r#macro::get_macro_debug_level() <= 4 {
            if !indirect_errors.is_empty() {
                crate::log_warn!("Grammar has {} hidden left recursion(s) (non-fatal)", indirect_errors.len());
            }
            if !left_recursion_errors.is_empty() {
                crate::log_warn!("Grammar has {} left-nullable recursion(s)", left_recursion_errors.len());
            }
        } else {
            // At level 5+: show full details
            for err in &post_transform_errors {
                crate::log_warn!("Validation error: {}", err);
            }
            for err in &left_recursion_errors {
                crate::log_warn!("Left-nullable recursion: {}", err);
            }
            for err in &indirect_errors {
                crate::log_warn!("Hidden left recursion: {}", err);
            }
            for err in &right_recursion_errors {
                crate::log_warn!("Right recursion: {}", err);
            }
        }
    }
    
    // If there are any critical errors, we should panic
    if !post_transform_errors.is_empty() || !right_recursion_errors.is_empty() || !indirect_errors.is_empty() {
        panic!(
            "Grammar transformations failed to eliminate problematic patterns:\n  Length-1: {:?}\n  Right recursion: {:?}\n  Hidden left recursion: {:?}",
            post_transform_errors, right_recursion_errors, indirect_errors
        );
    }

    let mut next_non_terminal_id = non_terminal_map.len();
    for p in &productions {
        if !non_terminal_map.contains_left(&p.lhs) {
            non_terminal_map.insert(p.lhs.clone(), NonTerminalID(next_non_terminal_id));
            next_non_terminal_id += 1;
        }
    }

    crate::debug!(5, "Number of productions: {}", productions.len());
    print_memory_usage("Before Stage 1");

    let num_terminals = terminal_map.len();
    let num_nonterminals = non_terminal_map.len();

    let light_productions: Vec<Vec<usize>> = productions
        .iter()
        .map(|p| {
            p.rhs
                .iter()
                .map(|s| match s {
                    Symbol::Terminal(t) => terminal_map
                        .get_by_left(t)
                        .unwrap_or_else(|| panic!("Terminal not found in map: {:?}", t))
                        .0,
                    Symbol::NonTerminal(nt) => non_terminal_map
                        .get_by_left(nt)
                        .unwrap_or_else(|| panic!("NonTerminal not found in map: {:?}", nt))
                        .0
                        + num_terminals,
                })
                .collect()
        })
        .collect();

    let lhs_ids: Vec<usize> = productions
        .iter()
        .map(|p| {
            non_terminal_map
                .get_by_left(&p.lhs)
                .unwrap_or_else(|| panic!("LHS NonTerminal not found in map: {:?}", p.lhs))
                .0
        })
        .collect();

    let nullable_nonterminals = compute_nullable_nonterminals(&productions);
    let nullable_nts_ids: HashSet<usize> = nullable_nonterminals
        .iter()
        .map(|nt| {
            non_terminal_map
                .get_by_left(nt)
                .unwrap_or_else(|| panic!("Nullable NonTerminal not found in map: {:?}", nt))
                .0
        })
        .collect();

    let start_nt_id = lhs_ids[0];

    crate::debug!(4, "Stage 1 (LR(0) Automaton)");
    let (stage_1_table, item_set_map) =
        stage_1(&light_productions, &lhs_ids, num_terminals, num_nonterminals);
    print_memory_usage("After Stage 1");

    crate::debug!(4, "Computing First/Follow Sets");
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

    crate::debug!(4, "Computing Final Table (Merging Stages 2-8)");
    let (final_table_map, start_state_id, substring_state_id) = compute_final_table(
        stage_1_table,
        &item_set_map,
        &productions,
        &lhs_ids,
        &follow_sets,
        num_terminals,
    );
    print_memory_usage("After Final Table");

    let mut item_set_map_bi = BiBTreeMap::new();
    for (k, v) in item_set_map {
        item_set_map_bi.insert(k, v);
    }

    crate::debug!(4, "GLR Parser generation complete. {} states.", final_table_map.len());

    GLRParser::new(
        final_table_map,
        productions,
        terminal_map,
        non_terminal_map,
        item_set_map_bi,
        start_state_id,
        substring_state_id,
        actions,
        ignore_terminal_ids,
    )
}

/// Generate a GLR parser from productions, with automatic detection of ignore terminals.
/// 
/// This function auto-detects whitespace-like terminals (terminals that are always optional)
/// and adds them to the ignore set. Additional explicit ignore terminals can be provided.
pub fn generate_glr_parser(
    productions: &[Production],
    nullable_terminals: &HashSet<Terminal>,
    explicit_ignore_terminal_ids: HashSet<TerminalID>,
) -> crate::glr::parser::GLRParser {
    let terminal_map = assign_terminal_ids(productions);
    generate_glr_parser_with_terminal_map(productions, terminal_map, nullable_terminals, explicit_ignore_terminal_ids)
}

pub fn generate_glr_parser_with_terminal_map(
    productions: &[Production],
    terminal_map: BiBTreeMap<Terminal, TerminalID>,
    nullable_terminals: &HashSet<Terminal>,
    explicit_ignore_terminal_ids: HashSet<TerminalID>,
) -> crate::glr::parser::GLRParser {
    // Transform nullable terminals into optional non-terminals BEFORE generating the parser
    let existing_nonterminals: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();
    let (transformed_productions, _terminal_to_opt_nt) = transform_nullable_terminals(
        productions,
        nullable_terminals,
        &existing_nonterminals,
    );
    
    // Auto-detect whitespace-like terminals and combine with explicit ignore terminals
    let detected_ignore = detect_whitespace_like_terminals(&transformed_productions, &terminal_map);
    let explicit_count = explicit_ignore_terminal_ids.len();
    let mut all_ignore_ids = explicit_ignore_terminal_ids;
    all_ignore_ids.extend(detected_ignore.iter());
    
    if !all_ignore_ids.is_empty() {
        crate::debug!(4, "Using {} ignore terminals ({} explicit, {} auto-detected)",
            all_ignore_ids.len(), 
            explicit_count,
            detected_ignore.len());
        for t in &detected_ignore {
            crate::debug!(5, "  {}", terminal_map.get_by_right(t).unwrap());
        }
    }
    
    // Simplify grammar by eliminating unit productions (A → B → X becomes A → X)
    // This reduces parser construction time for grammars with many trivial chains.
    let start_nt = &transformed_productions.get(0).map(|p| p.lhs.clone()).unwrap_or(NonTerminal("start".to_string()));
    const MAX_SUBSTITUTION_RHS_LEN: usize = 1;
    let (simplified_with_defs, substituted_nts) = substitute_single_productions_and_report(
        &transformed_productions,
        start_nt,
        MAX_SUBSTITUTION_RHS_LEN,
    );
    let simplified_productions = remove_productions_for_nts(&simplified_with_defs, &substituted_nts);
    
    if simplified_productions.len() < transformed_productions.len() {
        crate::debug!(4, "Grammar simplification: {} → {} productions (eliminated {} unit productions)",
            transformed_productions.len(),
            simplified_productions.len(),
            transformed_productions.len() - simplified_productions.len());
    }
    
    let non_terminal_map = assign_non_terminal_ids(&simplified_productions);
    generate_glr_parser_with_maps(
        &simplified_productions,
        terminal_map,
        non_terminal_map,
        BTreeMap::new(),
        all_ignore_ids,
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
