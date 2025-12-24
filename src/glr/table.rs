use super::items::Item;
use crate::glr::analyze::{
    create_unique_name_generator, inline_null_productions,
    remove_productions_with_undefined_nonterminals, validate,
};
use crate::glr::minimizer::{substitute_single_productions_and_report, remove_productions_for_nts, left_factor_grammar};
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
pub fn detect_whitespace_like_terminals(
    productions: &[Production],
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
) -> HashSet<TerminalID> {
    // Collect all terminals that appear in the grammar
    let mut all_terminals: HashSet<Terminal> = HashSet::new();
    for prod in productions {
        for sym in &prod.rhs {
            if let Symbol::Terminal(t) = sym {
                all_terminals.insert(t.clone());
            }
        }
    }
    
    // For each terminal, check if it's always optional
    // A terminal T is whitespace-like if for every occurrence of T in a production,
    // there exists a corresponding production without T at that position
    let mut whitespace_like: HashSet<TerminalID> = HashSet::new();
    
    // Build a set of all (LHS, RHS) pairs for quick lookup
    let production_set: HashSet<(NonTerminal, Vec<Symbol>)> = productions
        .iter()
        .map(|p| (p.lhs.clone(), p.rhs.clone()))
        .collect();
    
    'terminal_loop: for terminal in &all_terminals {
        let terminal_sym = Symbol::Terminal(terminal.clone());
        
        // Check every production to see if this terminal appears
        for prod in productions {
            // Find all positions where this terminal appears in this production
            for (pos, sym) in prod.rhs.iter().enumerate() {
                if *sym == terminal_sym {
                    // This terminal appears at position `pos` in production `prod`
                    // Check if there's a corresponding production without this terminal
                    let mut rhs_without_terminal = prod.rhs.clone();
                    rhs_without_terminal.remove(pos);
                    
                    // Check if the production A → α β (without T at this position) exists
                    if !production_set.contains(&(prod.lhs.clone(), rhs_without_terminal)) {
                        // This terminal is NOT always optional - skip to next terminal
                        continue 'terminal_loop;
                    }
                }
            }
        }
        
        // If we get here, every occurrence of this terminal has a corresponding
        // production without it - this terminal is whitespace-like
        if let Some(tid) = terminal_map.get_by_left(terminal) {
            whitespace_like.insert(*tid);
            crate::debug!(5, "Detected whitespace-like terminal: {:?} (ID {})", terminal, tid.0);
        }
    }
    
    whitespace_like
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

/// Grammar Optimization Pipeline
/// ==============================
///
/// This function implements a multi-pass grammar optimization pipeline that transforms
/// the input grammar into a form suitable for bounded GLR parsing. The optimizations
/// are organized into two categories:
///
/// ## ESSENTIAL Optimizations (Required for correctness)
///
/// These optimizations MUST complete fully for the parser to work correctly:
///
/// 1. **Nullable Terminal Transformation** (Phase 1)
///    - Transforms terminals that can match empty string into optional non-terminals
///    - Required because the tokenizer produces zero-width matches for nullable terminals
///    - Must happen BEFORE any null production inlining
///
/// 2. **Null Production Inlining** (Phase 2)
///    - Expands productions with nullable non-terminals to explicit alternatives
///    - Exposes hidden recursion patterns (e.g., A → α A β where β is nullable)
///    - Required for proper FIRST/FOLLOW set computation
///
/// 3. **Right Recursion Elimination** (Phase 3)
///    - Transforms A → α A into left recursion A → A' α, A' → ε | A' α
///    - Per "Even Faster Generalized LR Parsing" (Aycock & Horspool, 1999),
///      eliminating right recursion guarantees bounded reductions
///    - May require multiple passes when productions have multiple recursive references
///
/// 4. **Hidden Left Recursion Elimination** (Phase 4)
///    - Detects A → β B where B →* A α and β is nullable
///    - Required for the bounded reductions theorem
///
/// ## DECORATIVE Optimizations (Performance/Clarity)
///
/// These optimizations improve performance or grammar clarity but aren't strictly required:
///
/// 1. **Unit Production Elimination**
///    - Simplifies A → B → X into A → X
///    - Reduces parser state count but doesn't affect correctness
///
/// 2. **Whitespace Terminal Detection** (currently disabled)
///    - Auto-detects terminals that are always optional
///    - Could be used to mark implicit whitespace
///
/// ## Optimization Loop Structure
///
/// ```text
/// ┌─────────────────────────────────┐
/// │ Phase 1: Nullable Terminals     │  (before loop)
/// │ (transform to optional NTs)     │
/// └─────────────────────────────────┘
///                  │
///                  ▼
///        ╔═══════════════════════════════════════╗
///        ║     FIXED-POINT LOOP (Phases 2-4)     ║
///        ║  ┌─────────────────────────────────┐  ║
///   ┌───►║  │ Phase 2: Inline Null Productions│  ║
///   │    ║  │ (expose hidden patterns)        │  ║
///   │    ║  └─────────────────────────────────┘  ║
///   │    ║                   │                   ║
///   │    ║                   ▼                   ║
///   │    ║  ┌─────────────────────────────────┐  ║
///   │    ║  │ Phase 3: Right Recursion        │  ║
///   │    ║  │ (eliminate direct & indirect)   │  ║
///   │    ║  └─────────────────────────────────┘  ║
///   │    ║                   │                   ║
///   │    ║                   ▼                   ║
///   │    ║  ┌─────────────────────────────────┐  ║
///   │    ║  │ Phase 4: Hidden Left Recursion  │  ║
///   │    ║  │ (best effort elimination)       │  ║
///   │    ║  └─────────────────────────────────┘  ║
///   │    ║                   │                   ║
///   │    ║          ┌────────┴────────┐          ║
///   │    ║          │ changes made?   │          ║
///   │    ║          └────────┬────────┘          ║
///   │    ║             yes ╱   ╲ no              ║
///   │    ║                ╱     ╲                ║
///   └────╫───────────────┘       │               ║
///        ╚═══════════════════════│═══════════════╝
///                                ▼
///                 ┌─────────────────────────────────┐
///                 │ Phase 5: Whitespace Detection   │
///                 │ (DECORATIVE - currently off)    │
///                 └─────────────────────────────────┘
///                                │
///                                ▼
///                 ┌─────────────────────────────────┐
///                 │ Phase 6: Unit Production Elim.  │
///                 │ (DECORATIVE - simplify grammar) │
///                 └─────────────────────────────────┘
/// ```
///
#[time_it]
fn generate_glr_parser_with_maps(
    productions: &[Production],
    terminal_map: BiBTreeMap<Terminal, TerminalID>,
    mut non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    actions: BTreeMap<NonTerminal, ActionFn>,
    nullable_terminals: HashSet<Terminal>,
    explicit_ignore_terminal_ids: HashSet<TerminalID>,
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

    // ============================================================
    // Phase 0: Start symbol isolation
    // ============================================================
    // If the first production's LHS nonterminal appears in the RHS of ANY
    // production (including itself), we need to create a new initial production
    // with a fresh nonterminal to isolate the start symbol.
    if !productions.is_empty() {
        let original_start_nt = productions[0].lhs.clone();
        let start_appears_in_rhs = productions.iter().any(|p| {
            p.rhs.iter().any(|sym| {
                if let Symbol::NonTerminal(nt) = sym {
                    *nt == original_start_nt
                } else {
                    false
                }
            })
        });

        if start_appears_in_rhs {
            crate::debug!(4, "Start symbol '{}' appears in RHS, creating isolated start", original_start_nt.0);
            // Generate a unique name for the new start nonterminal
            let existing_nonterminals: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();
            let mut new_start_name = format!("{}_Start", original_start_nt.0);
            let mut counter = 0;
            while existing_nonterminals.iter().any(|nt| nt.0 == new_start_name) {
                counter += 1;
                new_start_name = format!("{}_Start_{}", original_start_nt.0, counter);
            }
            let new_start_nt = NonTerminal(new_start_name);

            // Create new initial production: NewStart → OriginalStartRHS
            let new_initial_production = Production {
                lhs: new_start_nt,
                rhs: productions[0].rhs.clone(),
            };

            // Insert the new production at the beginning
            productions.insert(0, new_initial_production);
            crate::debug!(5, "Created new start production: {} → {:?}", productions[0].lhs.0, productions[0].rhs);
        }
    }

    // Note the current start nonterminal (LHS of the first production)
    let start_nonterminal = if productions.is_empty() {
        NonTerminal("__empty__".to_string())
    } else {
        productions[0].lhs.clone()
    };
    crate::debug!(5, "Start nonterminal: {}", start_nonterminal.0);

    // Determine if the grammar as a whole is nullable (start symbol can derive ε)
    let initial_nullable_nonterminals = crate::glr::automaton::compute_nullable_nonterminals(&productions);
    let grammar_is_nullable = initial_nullable_nonterminals.contains(&start_nonterminal);
    crate::debug!(4, "Grammar is nullable: {}", grammar_is_nullable);

    // ============================================================
    // Phase 1: ESSENTIAL - Transform nullable terminals
    // ============================================================
    // This must happen EARLY before any null production inlining.
    // Nullable terminals (terminals that can match empty string) create
    // zero-width matches in the tokenizer. We transform them into optional
    // non-terminals: T → OptT, OptT → T | ε
    if !nullable_terminals.is_empty() {
        crate::debug!(4, "Phase 1: Transforming {} nullable terminals into optional non-terminals",
            nullable_terminals.len());
        let existing_nonterminals: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();
        let (transformed, _terminal_map) = transform_nullable_terminals(
            &productions,
            &nullable_terminals,
            &existing_nonterminals,
        );
        productions = transformed;
        print_memory_usage("After nullable terminal transformation");
    }

    // ============================================================
    // Phase 2-4: ESSENTIAL - Grammar normalization loop
    // ============================================================
    // These transformations are interdependent:
    // - Null inlining exposes hidden recursion
    // - Right recursion elimination may introduce new nullable NTs
    // - Hidden left recursion elimination may need re-inlining
    //
    // We loop until no more changes occur (fixed point).

    let mut detected_ignore = HashSet::new();

    const MAX_OPTIMIZATION_PASSES: usize = usize::MAX;
    for pass in 0..MAX_OPTIMIZATION_PASSES {
        crate::debug!(4, "Grammar optimization pass {}", pass + 1);
        let initial_productions = productions.clone();
        let initial_production_count = productions.len(); // Keep for logging if needed, or remove

        // Phase 2: ESSENTIAL - Inline null productions
        // This exposes hidden right recursion: A → α A β where β is nullable
        // becomes A → α A | A → α A β
        crate::debug!(5, "  Phase 2: Inlining null productions");
        productions = inline_null_productions(&productions);

        // Phase 3: ESSENTIAL - Right recursion elimination
        // Transforms A → α A into A → A' α, A' → ε | A' α
        // Per "Even Faster Generalized LR Parsing", this guarantees bounded reductions
        let right_recursion_errors = crate::glr::analyze::check_for_right_recursion(&productions);
        if !right_recursion_errors.is_empty() {
            crate::debug!(5, "  Phase 3: Eliminating {} right recursion patterns", right_recursion_errors.len());
            for err in &right_recursion_errors {
                crate::debug!(6, "    {}", err);
            }

            let nonterminals: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
            let mut unique_name_generator = create_unique_name_generator(&nonterminals);

            // Resolve indirect right recursion first (by inlining)
            crate::glr::analyze::resolve_indirect_right_recursion(
                &mut productions,
                &mut unique_name_generator,
            );

            // Then resolve direct right recursion
            crate::glr::analyze::resolve_direct_right_recursion(
                &mut productions,
                &mut unique_name_generator,
            );
        }

        // Phase 4: Hidden left recursion elimination (best effort)
        // Detects A → β B where B →* A α and β is nullable
        // Note: Some grammars (like ambiguous E → E + E | E * E) have inherent
        // hidden left recursion that cannot be eliminated. We do our best but
        // don't require it to be fully eliminated.
        crate::debug!(5, "  Phase 4: Checking for hidden left recursion");
        crate::glr::analyze::eliminate_hidden_left_recursion(&mut productions);

        // Phase 5: DECORATIVE - Whitespace detection and removal
        // ============================================================
        // Auto-detect whitespace-like terminals and remove productions using them.
        // We run this in the loop so it stabilizes with the grammar.
        let newly_detected = detect_whitespace_like_terminals(&productions, &terminal_map);
        detected_ignore.extend(newly_detected);

        let all_ignored: HashSet<_> = explicit_ignore_terminal_ids.iter()
            .chain(detected_ignore.iter())
            .cloned()
            .collect();

        // Remove productions that use any ignored terminal
        let pre_filter_count = productions.len();
        productions.retain(|p| {
            !p.rhs.iter().any(|s| {
                if let Symbol::Terminal(t) = s {
                    if let Some(tid) = terminal_map.get_by_left(t) {
                        all_ignored.contains(tid)
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
        });

        if productions.len() < pre_filter_count {
            crate::debug!(4, "Phase 5: Removed {} productions containing ignored terminals", pre_filter_count - productions.len());
        }

        // Check if we've reached a fixed point
        // Note: We only require right_recursion to be empty (essential for bounded reductions).
        // Hidden left recursion may persist in some grammars (it's non-fatal).
        let final_production_count = productions.len();
        let right_recursion_remaining = crate::glr::analyze::check_for_right_recursion(&productions);

        if right_recursion_remaining.is_empty() && productions == initial_productions {
            crate::debug!(4, "Grammar optimization converged after {} passes", pass + 1);
            break;
        }

        if pass == MAX_OPTIMIZATION_PASSES - 1 {
            crate::log_warn!("Grammar optimization did not converge after {} passes", MAX_OPTIMIZATION_PASSES);
        }
    }
    print_memory_usage("After grammar normalization loop");

    // ============================================================
    // Final epsilon elimination
    // ============================================================
    // The optimization loop may have created new epsilon productions
    // (e.g., from right recursion elimination: A' → ε). We need to
    // inline these so that no epsilon productions remain.
    // inline_null_productions now guarantees elimination of ALL
    // epsilon productions by inlining into the start production as well.
    crate::debug!(4, "Final epsilon production elimination");
    productions = inline_null_productions(&productions);
    print_memory_usage("After final epsilon elimination");
    
    // Assertion: Ensure NO explicitly ignored terminals appear in final productions
    for (i, p) in productions.iter().enumerate() {
        for s in &p.rhs {
            if let Symbol::Terminal(t) = s {
                if let Some(tid) = terminal_map.get_by_left(t) {
                    assert!(
                        !explicit_ignore_terminal_ids.contains(tid),
                        "CRITICAL INVARIANT VIOLATION: Explicitly ignored terminal '{}' found in production {}: {}",
                        t, i, p
                    );
                }
            }
        }
    }

    let explicit_count = explicit_ignore_terminal_ids.len();
    let mut ignore_terminal_ids = explicit_ignore_terminal_ids;
    ignore_terminal_ids.extend(detected_ignore.iter());

    if !ignore_terminal_ids.is_empty() {
        crate::debug!(4, "Using {} ignore terminals ({} explicit, {} auto-detected)",
            ignore_terminal_ids.len(),
            explicit_count,
            detected_ignore.len());
        for t in &detected_ignore {
            crate::debug!(5, "  Auto-detected: {}", terminal_map.get_by_right(t).unwrap());
        }
    }

    // ============================================================
    // Phase 6: DECORATIVE - Left factoring
    // ============================================================
    // Extract common prefixes from productions with the same LHS.
    // If A → α β and A → α γ, create A → α A' and A' → β | γ.
    // This preserves the language but reduces LR states.
    let nonterminals_for_naming: BTreeSet<_> = productions.iter().map(|p| p.lhs.clone()).collect();
    let mut left_factor_name_gen = create_unique_name_generator(&nonterminals_for_naming);
    let (factored_productions, new_nts) = left_factor_grammar(&productions, &mut left_factor_name_gen);
    
    if !new_nts.is_empty() {
        crate::debug!(4, "Phase 6: Left factoring created {} new nonterminals ({} → {} productions)",
            new_nts.len(),
            productions.len(),
            factored_productions.len());
    }
    productions = factored_productions;
    print_memory_usage("After left factoring");

    // ============================================================
    // Phase 7: DECORATIVE - Unit production elimination
    // ============================================================
    // Simplify grammar by eliminating unit productions (A → B → X becomes A → X).
    // This reduces parser construction time but doesn't affect correctness.
    let start_nt = &productions.get(0).map(|p| p.lhs.clone()).unwrap_or(NonTerminal("start".to_string()));
    const MAX_SUBSTITUTION_RHS_LEN: usize = 1;
    let (simplified_with_defs, substituted_nts) = substitute_single_productions_and_report(
        &productions,
        start_nt,
        MAX_SUBSTITUTION_RHS_LEN,
    );
    let simplified_productions = remove_productions_for_nts(&simplified_with_defs, &substituted_nts);

    if simplified_productions.len() < productions.len() {
        crate::debug!(4, "Phase 7: Eliminated {} unit productions ({} → {})",
            productions.len() - simplified_productions.len(),
            productions.len(),
            simplified_productions.len());
    }
    productions = simplified_productions;
    print_memory_usage("After unit production elimination");

    // ============================================================
    // Epsilon Production Assertion
    // ============================================================
    // After all optimizations, there should be NO epsilon productions.
    // They should all have been eliminated by inline_null_productions.
    let epsilon_productions: Vec<_> = productions.iter()
        .filter(|p| p.rhs.is_empty())
        .collect();
    assert!(
        epsilon_productions.is_empty(),
        "Epsilon productions should have been eliminated, but found: {:?}",
        epsilon_productions.iter().map(|p| &p.lhs.0).collect::<Vec<_>>()
    );

    // ============================================================
    // Phase 8: Restore grammar nullability / Finalize Structure
    // ============================================================
    // Handle edge case: grammar was nullable but all productions were eliminated
    // (e.g., grammar was just "S → ε"). We need to restore a minimal nullable structure.
    if productions.is_empty() && grammar_is_nullable {
        crate::debug!(4, "Phase 8: Restoring nullable grammar from empty productions");
        // Create: Start_nullable → ε (this is the only production)
        // Then: Start_final → Start_nullable
        let null_nt = NonTerminal(format!("{}_nullable", start_nonterminal.0));
        let final_nt = NonTerminal(format!("{}_start", start_nonterminal.0));
        
        // Create the nullable wrapper's epsilon production
        let null_prod_eps = Production { lhs: null_nt.clone(), rhs: vec![] };
        
        // Create the final start production
        let final_prod = Production { lhs: final_nt.clone(), rhs: vec![Symbol::NonTerminal(null_nt)] };
        
        productions.push(final_prod);
        productions.push(null_prod_eps);
    } else if !productions.is_empty() {
        let current_start_nt = start_nonterminal.clone();
        let existing_nonterminals: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();
        // Simple unique name generator
        let mut name_gen = |base: &str| -> NonTerminal {
             let mut name = base.to_string();
             let mut i = 0;
             while existing_nonterminals.contains(&NonTerminal(name.clone())) {
                 i += 1;
                 name = format!("{}_{}", base, i);
             }
             NonTerminal(name)
        };

        // Step 1: Handle Nullability restoration
        let mut working_start = current_start_nt.clone();
        
        if grammar_is_nullable {
             crate::debug!(4, "Phase 8: Re-introducing nullability");
             let null_wrapper_nt = name_gen(&format!("{}_nullable", start_nonterminal.0));
             
             // Create StartNullable -> CurrentStart
             let prod_val = Production { lhs: null_wrapper_nt.clone(), rhs: vec![Symbol::NonTerminal(working_start.clone())] };
             // Create StartNullable -> epsilon
             let prod_eps = Production { lhs: null_wrapper_nt.clone(), rhs: vec![] };
             
             productions.push(prod_val);
             productions.push(prod_eps);
             
             working_start = null_wrapper_nt;
        }

        // Step 2: Ensure single unique start production at index 0
        // Condition: (LHS of prod[0] != working_start) OR (More than 1 production for working_start)
        let first_prod_lhs = &productions[0].lhs;
        let start_prod_count = productions.iter().filter(|p| p.lhs == working_start).count();
        
        if first_prod_lhs != &working_start || start_prod_count > 1 {
             crate::debug!(4, "Phase 8: Creating final unique start production");
             
             let final_start_nt = name_gen(&format!("{}_start", start_nonterminal.0));
             
             let final_prod = Production { lhs: final_start_nt.clone(), rhs: vec![Symbol::NonTerminal(working_start)] };
             
             productions.insert(0, final_prod);
        }
    }


    // ============================================================
    // CRITICAL GRAMMAR INVARIANTS VALIDATION
    // ============================================================
    // ╔═══════════════════════════════════════════════════════════════════════════╗
    // ║  WARNING: This validation function MUST NEVER be weakened or removed!     ║
    // ║                                                                           ║
    // ║  These invariants are essential for correct GLR parser operation.         ║
    // ║  Violating them will cause subtle parsing bugs that are hard to debug.    ║
    // ║                                                                           ║
    // ║  If a grammar transformation breaks these invariants, FIX THE             ║
    // ║  TRANSFORMATION, do not weaken the validation!                            ║
    // ╚═══════════════════════════════════════════════════════════════════════════╝
    fn validate_critical_grammar_invariants(productions: &[Production]) {
        if productions.is_empty() {
            return;
        }

        let start_prod = &productions[0];
        let start_nt = &start_prod.lhs;

        // INVARIANT 1: Production at index 0 is the start production.
        // (This is implicit by the position, but we document it here.)

        // INVARIANT 2: The start nonterminal must NOT appear in any production
        // other than the start production itself - neither on LHS nor RHS.
        for (idx, prod) in productions.iter().enumerate().skip(1) {
            // Check LHS
            assert!(
                prod.lhs != *start_nt,
                "CRITICAL INVARIANT VIOLATION: Start nonterminal '{}' appears as LHS in production {} (should only be in production 0). \
                 Production: {} -> {:?}",
                start_nt.0, idx, prod.lhs.0, prod.rhs
            );
            
            // Check RHS
            for sym in &prod.rhs {
                if let Symbol::NonTerminal(nt) = sym {
                    assert!(
                        nt != start_nt,
                        "CRITICAL INVARIANT VIOLATION: Start nonterminal '{}' appears in RHS of production {} (should not appear anywhere except as LHS of production 0). \
                         Production: {} -> {:?}",
                        start_nt.0, idx, prod.lhs.0, prod.rhs
                    );
                }
            }
        }

        // INVARIANT 3: No epsilon productions, with ONE exception:
        // If the start production has exactly 1 symbol and that symbol is a nonterminal A,
        // then A is allowed to have an epsilon production (this is the "nullable wrapper" pattern).
        let allowed_epsilon_nt: Option<&NonTerminal> = if start_prod.rhs.len() == 1 {
            if let Symbol::NonTerminal(nt) = &start_prod.rhs[0] {
                Some(nt)
            } else {
                None
            }
        } else {
            None
        };

        for (idx, prod) in productions.iter().enumerate() {
            if prod.rhs.is_empty() {
                // This is an epsilon production
                if let Some(allowed_nt) = allowed_epsilon_nt {
                    if prod.lhs == *allowed_nt {
                        // This epsilon is allowed (nullable wrapper pattern)
                        continue;
                    }
                }
                
                panic!(
                    "CRITICAL INVARIANT VIOLATION: Epsilon production found at index {} for nonterminal '{}'. \
                     Epsilon productions are only allowed for the nullable wrapper nonterminal. \
                     Start production: {} -> {:?}",
                    idx, prod.lhs.0, start_nt.0, start_prod.rhs
                );
            }
        }
    }

    // Run the critical invariant validation
    crate::debug!(4, "Validating critical grammar invariants");
    validate_critical_grammar_invariants(&productions);

    // ============================================================
    // Additional Validation - Check for remaining grammar issues
    // ============================================================
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
    // Note: indirect_errors (hidden left recursion) is non-fatal - it may cause
    // performance issues but the parser will still produce correct results.
    if !post_transform_errors.is_empty() || !right_recursion_errors.is_empty() {
        panic!(
            "Grammar transformations failed to eliminate problematic patterns:\n  Length-1: {:?}\n  Right recursion: {:?}",
            post_transform_errors, right_recursion_errors
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
    let non_terminal_map = assign_non_terminal_ids(productions);
    generate_glr_parser_with_maps(
        productions,
        terminal_map,
        non_terminal_map,
        BTreeMap::new(),
        nullable_terminals.clone(),
        explicit_ignore_terminal_ids,
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
