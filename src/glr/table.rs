use super::items::Item;
use crate::glr::analyze::{
    create_unique_name_generator, inline_null_productions,
    remove_productions_with_undefined_nonterminals, validate, merge_identical_nonterminals,
};
use crate::glr::null_inline::{NullableInliningStrategy, make_null_inline_name_gen, run_null_inline};
use crate::glr::minimizer::{substitute_single_productions_and_report, remove_productions_for_nts, left_factor_grammar, eliminate_unreachable_productions};
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

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
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
    pub fn minimize(&mut self) {
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Row {
    shifts_and_reduces_full: ShiftsAndReducesFull,
    pub default_reduce: Option<Stage7ShiftsAndReducesLookaheadValue>,
    pub default_reduce_lookaheads: Option<BTreeSet<TerminalID>>,
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
        if let Some(action) = self.shifts_and_reduces_full.get(terminal_id) {
            return Some(action.clone());
        }

        // Check if default reduce applies to this specific terminal
        if let Some(default) = &self.default_reduce {
            if let Some(lookaheads) = &self.default_reduce_lookaheads {
                if lookaheads.contains(terminal_id) {
                    return Some(default.clone());
                }
            } else {
                // Legacy/Wildcard behavior: applies to all unseen terminals
                return Some(default.clone());
            }
        }
        
        None
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
        // Use the centralized lookup logic
        let action = self.get_shifts_and_reduces_for_terminal(&terminal_id);

        if let Some(action) = action {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(state_id) => shiftfn(&state_id),
                Stage7ShiftsAndReducesLookaheadValue::Reduce {
                    nonterminal_id,
                    len,
                    production_ids,
                } => reducefn(&nonterminal_id, &len, &production_ids),
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    if let Some(state_id) = shift {
                        shiftfn(&state_id);
                    }
                    for (len, nts) in reduces {
                        for (nt_id, pids) in nts {
                            reducefn(&nt_id, &len, &pids);
                        }
                    }

                }
            }
        }
    }
}

impl JSONConvertible for Row {
    fn to_json(&self) -> JSONNode {
        // [[[term_id, Action], ...], [[nt_id, Goto], ...], default_reduce, default_reduce_lookaheads?]
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
            
        let mut fields = vec![
            JSONNode::Array(shifts_pairs),
            JSONNode::Array(gotos_pairs),
            self.default_reduce.to_json(),
        ];
        
        // Add optional 4th field if present
        if let Some(lookaheads) = &self.default_reduce_lookaheads {
            let lookahead_nodes: Vec<JSONNode> = lookaheads.iter().map(|tid| tid.to_json()).collect();
            fields.push(JSONNode::Array(lookahead_nodes));
        }

        JSONNode::Array(fields)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(mut arr) if arr.len() >= 3 && arr.len() <= 4 => {
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

                let default_reduce = Option::<Stage7ShiftsAndReducesLookaheadValue>::from_json(arr.remove(0))?;
                
                let default_reduce_lookaheads = if !arr.is_empty() {
                    match arr.remove(0) {
                        JSONNode::Array(items) => {
                            let mut lookaheads = BTreeSet::new();
                            for item in items {
                                lookaheads.insert(TerminalID::from_json(item)?);
                            }
                            Some(lookaheads)
                        }
                        _ => return Err("Expected array of terminal IDs for default_reduce_lookaheads".to_string())
                    }
                } else {
                    None
                };

                Ok(Row {
                    shifts_and_reduces_full,
                    gotos,
                    default_reduce,
                    default_reduce_lookaheads,
                })
            }
            _ => Err("Expected [shifts, gotos, default_reduce, (optional) lookaheads] array for Row".to_string()),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible, serde::Serialize, serde::Deserialize)]
pub struct StateID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible, serde::Serialize, serde::Deserialize)]
pub struct ProductionID(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible, serde::Serialize, serde::Deserialize)]
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
            val.minimize();

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
            val.minimize();
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
                default_reduce_lookaheads: None,
                gotos,
            }
,
        );
    }

    (final_table_map, start_state_id, substring_state_id)
}

// ============================================================
// Table State Reduction Optimizations
// ============================================================
// These optimizations run AFTER the table is constructed to reduce
// the number of states through merging and elimination.

/// Optimize the LR table by eliminating unreachable states and merging identical rows.
/// Returns the optimized table and updated start/substring state IDs.
#[time_it]
fn optimize_table(
    mut table: Table,
    mut start_state_id: StateID,
    mut substring_state_id: StateID,
) -> (Table, StateID, StateID) {
    let initial_count = table.len();
    
    // Step 1: Eliminate unreachable states
    let (t, s, sub) = eliminate_unreachable_states(table, start_state_id, substring_state_id);
    table = t;
    start_state_id = s;
    substring_state_id = sub;
    
    let after_unreachable = table.len();
    
    // Step 2: Merge identical rows (LALR-style state merging)
    // Loop until convergence to handle cascading merges
    loop {
        let before_merge_count = table.len();
        let (t, s, sub) = merge_identical_rows(table, start_state_id, substring_state_id);
        table = t;
        start_state_id = s;
        substring_state_id = sub;
        
        if table.len() == before_merge_count {
            break;
        }
    }
    let after_merge = table.len();
    
    // Step 3: Promote most frequent reduce to default_reduce
    // This allows us to remove even more explicit entries in Step 4
    promote_common_reduces(&mut table);
    
    // Step 4: Compact default reduces (remove redundant entries)
    let table = compact_default_reduces(table);
    
    if initial_count != after_merge {
        crate::debug!(4, "Table optimization: {} → {} states ({} unreachable, {} merged)",
            initial_count, after_merge, 
            initial_count - after_unreachable,
            after_unreachable - after_merge);
    }
    
    (table, start_state_id, substring_state_id)
}

/// For each row, identify the most frequent Reduce action and promote it to default_reduce.
/// This significantly reduces the size of the table map.
fn promote_common_reduces(table: &mut Table) {
    for row in table.values_mut() {
        // If we already have a default, we might still want to check if another reduce is MORE common
        // combined with the existing default coverage? 
        // For simplicity, let's count all reduces in the map.
        
        let mut reduce_counts: BTreeMap<Stage7ShiftsAndReducesLookaheadValue, usize> = BTreeMap::new();
        
        for action in row.shifts_and_reduces_full.values() {
            if let Stage7ShiftsAndReducesLookaheadValue::Reduce { .. } = action {
                *reduce_counts.entry(action.clone()).or_default() += 1;
            }
        }
        
        if let Some(current_default) = &row.default_reduce {
            // Count virtual entries covered by default? 
            // It's hard to know how many "None" entries effectively became default.
            // But usually default_reduce is used for "everything else".
            // If we switch default, we must ensure correctness.
            // 
            // Strategy: Only promote if NO default exists yet, OR if we are overriding.
            // Actually, safe strategy:
            // 1. Find most frequent reduce in `shifts_and_reduces_full`.
            // 2. If its count > threshold, make it default.
            // 3. BUT: What about "Error" entries?
            //    If we set default_reduce, then ANY lookahead not in the map becomes that reduce.
            //    This effectively eliminates syntax errors for that state! 
            //    GLR parsers CAN strictly rely on the table for errors.
            //    
            //    CRITICAL: setting default_reduce implies that for ALL terminals not in the map,
            //    we perform this reduction. This changes behavior if those terminals should be errors.
            //    
            //    HOWEVER: In standard LR parsing, "Default Reduce" is valid if the reduction 
            //    action doesn't consume lookahead (it doesn't). The parser will reduce, 
            //    pop stack, and retry the same lookahead in the new state.
            //    Eventually it shifts or errors.
            //    So "default reduce" is generally semantics-preserving for LR parsers
            //    because it just delays the error detection to the next state (after reduction).
            //    
            //    So YES, we can promote the most frequent reduce to default.
        }
        
        // Find the reduce with max count
        let best_reduce = reduce_counts.into_iter().max_by_key(|(_, count)| *count);
        
        if let Some((reduce_action, count)) = best_reduce {
            // Threshold: only worth it if we save entries.
            // Let's say count > 1.
            if count > 1 {
                // Determine if we should replace existing default
                // If we replace, the old default (if any) is lost -> potentially risky?
                // Actually, if we have a default, it applies to "holes".
                // If we change it, "holes" change behavior.
                //
                // SAFE APPROACH: Only set default if it is currently None.
                // Existing logic in `compute_final_table` sets default based on EOF often.
                // Overriding it might handle EOF incorrectly if EOF relied on default?
                // No, explicit EOF entries are usually in the map.
                
                if row.default_reduce.is_none() {
                     row.default_reduce = Some(reduce_action);
                }
            }
        }
    }
}

/// Eliminate states that are not reachable from the start state.
fn eliminate_unreachable_states(
    table: Table,
    start_state_id: StateID,
    substring_state_id: StateID,
) -> (Table, StateID, StateID) {
    // Find all reachable states via BFS from start
    let mut reachable: HashSet<StateID> = HashSet::new();
    let mut worklist: VecDeque<StateID> = VecDeque::new();
    
    reachable.insert(start_state_id);
    worklist.push_back(start_state_id);
    
    // Also include substring_state_id as a root if different
    if substring_state_id != start_state_id {
        reachable.insert(substring_state_id);
        worklist.push_back(substring_state_id);
    }
    
    while let Some(state_id) = worklist.pop_front() {
        if let Some(row) = table.get(&state_id) {
            // Follow shift transitions
            for (_terminal, action) in &row.shifts_and_reduces_full {
                collect_target_states(action, &mut reachable, &mut worklist);
            }
            if let Some(ref default) = row.default_reduce {
                collect_target_states(default, &mut reachable, &mut worklist);
            }
            // Follow goto transitions
            for (_nt, goto) in &row.gotos {
                if let Some(target) = goto.state_id {
                    if reachable.insert(target) {
                        worklist.push_back(target);
                    }
                }
            }
        }
    }
    
    // Filter to only reachable states
    let filtered: Table = table.into_iter()
        .filter(|(state_id, _)| reachable.contains(state_id))
        .collect();
    
    (filtered, start_state_id, substring_state_id)
}

/// Helper to collect target states from an action
fn collect_target_states(
    action: &Stage7ShiftsAndReducesLookaheadValue,
    reachable: &mut HashSet<StateID>,
    worklist: &mut VecDeque<StateID>,
) {
    match action {
        Stage7ShiftsAndReducesLookaheadValue::Shift(target) => {
            if reachable.insert(*target) {
                worklist.push_back(*target);
            }
        }
        Stage7ShiftsAndReducesLookaheadValue::Split { shift, .. } => {
            if let Some(target) = shift {
                if reachable.insert(*target) {
                    worklist.push_back(*target);
                }
            }
        }
        Stage7ShiftsAndReducesLookaheadValue::Reduce { .. } => {
            // Reduces don't have direct state targets
        }
    }
}

/// Merge states that have identical action and goto tables.
/// This is similar to LALR state merging but operates on the final table.
fn merge_identical_rows(
    table: Table,
    start_state_id: StateID,
    substring_state_id: StateID,
) -> (Table, StateID, StateID) {
    // Group states by their row content (excluding the state ID key)
    // We need a way to hash/compare Row content
    let mut canonical: BTreeMap<RowSignature, StateID> = BTreeMap::new();
    let mut state_mapping: BTreeMap<StateID, StateID> = BTreeMap::new();
    
    // First pass: identify canonical representatives for each unique row
    for (state_id, row) in &table {
        let sig = compute_row_signature(row);
        
        if let Some(&canonical_id) = canonical.get(&sig) {
            // This row is identical to an existing one
            state_mapping.insert(*state_id, canonical_id);
        } else {
            // This is a new unique row
            canonical.insert(sig, *state_id);
            state_mapping.insert(*state_id, *state_id);
        }
    }
    
    // Check if any merging happened
    let unique_states: HashSet<StateID> = state_mapping.values().cloned().collect();
    if unique_states.len() == table.len() {
        // No merging possible
        return (table, start_state_id, substring_state_id);
    }
    
    // Second pass: create new table with merged states and remapped references
    let mut new_table: Table = BTreeMap::new();
    
    for (state_id, row) in table {
        let canonical_id = state_mapping[&state_id];
        
        // Only include canonical representatives
        if canonical_id != state_id {
            continue;
        }
        
        // Remap all state references in the row
        let remapped_row = remap_row_states(&row, &state_mapping);
        new_table.insert(canonical_id, remapped_row);
    }
    
    // Remap start and substring state IDs
    let new_start = state_mapping.get(&start_state_id).copied().unwrap_or(start_state_id);
    let new_substring = state_mapping.get(&substring_state_id).copied().unwrap_or(substring_state_id);
    
    (new_table, new_start, new_substring)
}

/// A signature that uniquely identifies a row's behavior (for merging identical rows).
/// We use a serialized representation that can be compared/ordered.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RowSignature {
    // We serialize to a comparable format
    // The key insight: two rows are "mergeable" if they have the same actions
    // for all terminals and same gotos for all non-terminals
    shifts_and_reduces: Vec<(TerminalID, ActionSignature)>,
    default_reduce: Option<ActionSignature>,
    gotos: Vec<(NonTerminalID, GotoSignature)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ActionSignature {
    Shift(StateID),
    Reduce { nonterminal_id: NonTerminalID, len: usize, production_ids: Vec<ProductionID> },
    Split { 
        shift: Option<StateID>, 
        reduces: Vec<(usize, Vec<(NonTerminalID, Vec<ProductionID>)>)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GotoSignature {
    state_id: Option<StateID>,
    accept: bool,
}

fn compute_row_signature(row: &Row) -> RowSignature {
    let shifts_and_reduces: Vec<_> = row.shifts_and_reduces_full
        .iter()
        .map(|(tid, action)| (*tid, compute_action_signature(action)))
        .collect();
    
    let default_reduce = row.default_reduce
        .as_ref()
        .map(compute_action_signature);
    
    let gotos: Vec<_> = row.gotos
        .iter()
        .map(|(ntid, goto)| (*ntid, GotoSignature { 
            state_id: goto.state_id, 
            accept: goto.accept 
        }))
        .collect();
    
    RowSignature { shifts_and_reduces, default_reduce, gotos }
}

fn compute_action_signature(action: &Stage7ShiftsAndReducesLookaheadValue) -> ActionSignature {
    match action {
        Stage7ShiftsAndReducesLookaheadValue::Shift(s) => ActionSignature::Shift(*s),
        Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
            ActionSignature::Reduce { 
                nonterminal_id: *nonterminal_id, 
                len: *len, 
                production_ids: production_ids.clone() 
            }
        }
        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
            let reduces_vec: Vec<_> = reduces
                .iter()
                .map(|(len, nt_map)| {
                    let nt_vec: Vec<_> = nt_map
                        .iter()
                        .map(|(nt, pids)| (*nt, pids.clone()))
                        .collect();
                    (*len, nt_vec)
                })
                .collect();
            ActionSignature::Split { shift: *shift, reduces: reduces_vec }
        }
    }
}

fn remap_row_states(row: &Row, mapping: &BTreeMap<StateID, StateID>) -> Row {
    let shifts_and_reduces_full: ShiftsAndReducesFull = row.shifts_and_reduces_full
        .iter()
        .map(|(tid, action)| (*tid, remap_action_states(action, mapping)))
        .collect();
    
    let default_reduce = row.default_reduce
        .as_ref()
        .map(|action| remap_action_states(action, mapping));
    let default_reduce_lookaheads = row.default_reduce_lookaheads.clone();
    
    let gotos: BTreeMap<NonTerminalID, Goto> = row.gotos
        .iter()
        .map(|(ntid, goto)| {
            let new_state_id = goto.state_id.map(|s| mapping.get(&s).copied().unwrap_or(s));
            (*ntid, Goto { state_id: new_state_id, accept: goto.accept })
        })
        .collect();
    
    Row { shifts_and_reduces_full, default_reduce, default_reduce_lookaheads, gotos }
}


fn remap_action_states(
    action: &Stage7ShiftsAndReducesLookaheadValue, 
    mapping: &BTreeMap<StateID, StateID>
) -> Stage7ShiftsAndReducesLookaheadValue {
    match action {
        Stage7ShiftsAndReducesLookaheadValue::Shift(s) => {
            Stage7ShiftsAndReducesLookaheadValue::Shift(mapping.get(s).copied().unwrap_or(*s))
        }
        Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
            Stage7ShiftsAndReducesLookaheadValue::Reduce { 
                nonterminal_id: *nonterminal_id, 
                len: *len, 
                production_ids: production_ids.clone() 
            }
        }
        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
            Stage7ShiftsAndReducesLookaheadValue::Split { 
                shift: shift.map(|s| mapping.get(&s).copied().unwrap_or(s)), 
                reduces: reduces.clone() 
            }
        }
    }
}

/// Compact default reduces by removing entries from shifts_and_reduces_full
/// that are identical to the default_reduce.
fn compact_default_reduces(mut table: Table) -> Table {
    for (_state_id, row) in table.iter_mut() {
        if let Some(ref default) = row.default_reduce {
            // Collect terminals that will be removed (handled by default reduce)
            let mut lookaheads = BTreeSet::new();
            
            // Remove entries that are identical to the default, but track which terminals they were!
            // This allows us to restrict the default reduce to ONLY these terminals later.
            row.shifts_and_reduces_full.retain(|tid, action| {
                if action == default {
                    lookaheads.insert(*tid);
                    false // Remove
                } else {
                    true // Keep
                }
            });
            
            if lookaheads.is_empty() {
                // No explicit entries matched the default reduce.
                // Keep default_reduce_lookaheads as None so the default reduce
                // continues to apply to all unseen terminals (wildcard behavior).
                // Setting Some(empty_set) would incorrectly disable the reduce.
            } else {
                row.default_reduce_lookaheads = Some(lookaheads);
            }
        }
    }
    table
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
/// 2. **Explicit Ignore Terminals**
///    - Removes productions containing terminals declared via `%ignore`
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
///                 │ Phase 5: Explicit Ignore Filter │
///                 └─────────────────────────────────┘
///                                │
///                                ▼
///                 ┌─────────────────────────────────┐
///                 │ Phase 6: Unit Production Elim.  │
///                 │ (DECORATIVE - minimize grammar) │
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
    // IMPORTANT: Only transform terminals that are used in "real" productions,
    // not in wrapper definitions created by optimization.rs.
    // We use a STRUCTURAL check:
    // 1. Identify "nullable-capable" non-terminals (those that have an epsilon production N -> ε)
    // 2. A terminal T is "already wrapped" if ALL its usages are in productions N -> T
    //    where N is nullable-capable.
    
    // First, find nullable-capable non-terminals (have explicit epsilon production)
    let mut nullable_capable_nts: HashSet<NonTerminal> = HashSet::new();
    for p in productions.iter() {
        if p.rhs.is_empty() {
            nullable_capable_nts.insert(p.lhs.clone());
        }
    }
    
    // Collect all usages of each terminal
    // We Map Terminal -> List of (LHS, is_sole_rhs_symbol)
    let mut terminal_usages: HashMap<Terminal, Vec<(NonTerminal, bool)>> = HashMap::new();
    
    for p in productions.iter() {
        for sym in &p.rhs {
            if let Symbol::Terminal(t) = sym {
                let is_sole = p.rhs.len() == 1;
                terminal_usages.entry(t.clone())
                    .or_default()
                    .push((p.lhs.clone(), is_sole));
            }
        }
    }
    
    // Filter nullable terminals: Keep only those that have at least one "unwrapped" usage
    let actual_nullable: HashSet<Terminal> = nullable_terminals.iter()
        .filter(|t| {
            if let Some(usages) = terminal_usages.get(t) {
                // Check if ALL usages are wrapped
                // A usage is wrapped if:
                // 1. It is the sole symbol in RHS (N -> T)
                // 2. The LHS (N) is nullable-capable (N -> ε exists)
                let all_wrapped = usages.iter().all(|(lhs, is_sole)| {
                    *is_sole && nullable_capable_nts.contains(lhs)
                });
                !all_wrapped // Keep if NOT all wrapped (i.e., has exposed usage)
            } else {
                false // Unused terminals don't need wrapping (removed later)
            }
        })
        .cloned()
        .collect();
    
    if !actual_nullable.is_empty() {
        crate::debug!(4, "Phase 1: Transforming {} nullable terminals into optional non-terminals (of {} passed)",
            actual_nullable.len(), nullable_terminals.len());
        let existing_nonterminals: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();
        let (transformed, _terminal_map) = transform_nullable_terminals(
            &productions,
            &actual_nullable,
            &existing_nonterminals,
        );
        productions = transformed;
        print_memory_usage("After nullable terminal transformation");
    }
    
    // ============================================================
    // Phase 1.5: ESSENTIAL - Minimize redundant nullable wrappers
    // ============================================================
    // Nullable wrapper productions (T_Opt -> T | ε) can be created by:
    //   - Phase 1 above (transform_nullable_terminals)
    //   - optimization.rs (handle_nullable_terminals)
    //
    // If T is a nullable terminal (matches empty string), then T | ε = T.
    // So we can minimize T_Opt -> T | ε to just T_Opt -> T.
    // This prevents combinatorial explosion in inline_null_productions later.
    //
    // Detection: For each NT with exactly 2 productions:
    //   - One production with RHS = [Terminal(T)] where T is nullable
    //   - One production with RHS = [] (epsilon)
    // Action: Remove the epsilon production.
    // [DISABLE] Phase 1.5: ESSENTIAL - Minimize redundant nullable wrappers
    //
    // This optimization is currently DISABLED because it is incorrect for our use case.
    // Explanation:
    // If T is a nullable terminal (e.g. `__opt_x[1]__`), it might match an empty string.
    // However, our tokenizer/parser interaction implies that even for an "empty match",
    // we might need distinct handling or the wrapper non-terminal adds necessary structure
    // (creating a shift action vs just reducing).
    // Specifically, when we have T_Opt -> T | ε, and T -> ε, they are NOT equivalent in the
    // generated parser state machine if the tokenizer behavior for T depends on lookahead or
    // if T simply doesn't produce a token when empty, whereas the epsilon production allows
    // a reduction without consuming tokens.
    //
    // By merging them, we force the parser to expect the terminal T, effectively removing the
    // "pure epsilon" path that doesn't involve the terminal symbol at all.
    /*
    let nullable_terminal_ids: HashSet<TerminalID> = nullable_terminals.iter()
        .filter_map(|t| terminal_map.get_by_left(t).copied())
        .collect();
    
    // Group productions by LHS
    let mut by_lhs: BTreeMap<NonTerminal, Vec<usize>> = BTreeMap::new();
    for (i, p) in productions.iter().enumerate() {
        by_lhs.entry(p.lhs.clone()).or_default().push(i);
    }
    
    let mut to_remove: HashSet<usize> = HashSet::new();
    for (_nt, indices) in &by_lhs {
        if indices.len() != 2 {
            continue;
        }
        let p0 = &productions[indices[0]];
        let p1 = &productions[indices[1]];
        
        // Check if one is epsilon (empty RHS) and one is single nullable terminal
        let (epsilon_idx, term_idx) = if p0.rhs.is_empty() && p1.rhs.len() == 1 {
            (indices[0], indices[1])
        } else if p1.rhs.is_empty() && p0.rhs.len() == 1 {
            (indices[1], indices[0])
        } else {
            continue;
        };
        
        // Check if the single-symbol production is a nullable terminal
        let term_prod = &productions[term_idx];
        if let Some(Symbol::Terminal(t)) = term_prod.rhs.first() {
            if let Some(tid) = terminal_map.get_by_left(t) {
                if nullable_terminal_ids.contains(tid) {
                    // T is nullable, so T_Opt -> T | ε simplifies to T_Opt -> T
                    to_remove.insert(epsilon_idx);
                }
            }
        }
    }
    
    if !to_remove.is_empty() {
        crate::debug!(4, "Phase 1.5: Minimized {} redundant epsilon productions from nullable wrappers",
            to_remove.len());
        productions = productions.into_iter()
            .enumerate()
            .filter(|(i, _)| !to_remove.contains(i))
            .map(|(_, p)| p)
            .collect();
    }
    */
    print_memory_usage("After nullable wrapper minimization");


    // ============================================================
    // Phase 2-4: ESSENTIAL - Grammar normalization loop
    // ============================================================
    // These transformations are interdependent:
    // - Null inlining exposes hidden recursion
    // - Right recursion elimination may introduce new nullable NTs
    // - Hidden left recursion elimination may need re-inlining
    //
    // We loop until no more changes occur (fixed point).
    const MAX_OPTIMIZATION_PASSES: usize = usize::MAX;
    let normalization_start = std::time::Instant::now();

    // Read the nullable inlining strategy once from the environment.
    let null_inline_strategy = NullableInliningStrategy::from_env();
    crate::debug!(4, "Nullable inlining strategy: {}", null_inline_strategy.name());

    for pass in 0..MAX_OPTIMIZATION_PASSES {
        crate::debug!(4, "Grammar optimization pass {}", pass + 1);
        let initial_productions = productions.clone();
        let initial_production_count = productions.len(); // Keep for logging if needed, or remove

        // Phase 2: ESSENTIAL - Inline null productions
        // This exposes hidden right recursion: A → α A β where β is nullable
        // becomes A → α A | A → α A β
        crate::debug!(5, "  Phase 2: Inlining null productions (strategy: {})", null_inline_strategy.name());
        {
            let existing_nts: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();
            let mut null_name_gen = make_null_inline_name_gen(&existing_nts);
            productions = run_null_inline(&productions, &null_inline_strategy, &mut null_name_gen);
        }

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

        // Phase 5: Remove productions with explicit ignore terminals
        let pre_filter_count = productions.len();
        productions.retain(|p| {
            !p.rhs.iter().any(|s| {
                if let Symbol::Terminal(t) = s {
                    if let Some(tid) = terminal_map.get_by_left(t) {
                        explicit_ignore_terminal_ids.contains(tid)
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
            crate::timing!(
                "TIMING: glr_normalization_loop {:?} ({} passes, {} prods)",
                normalization_start.elapsed(),
                pass + 1,
                productions.len()
            );
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
    crate::debug!(4, "Final epsilon production elimination (strategy: {})", null_inline_strategy.name());
    let t_eps = std::time::Instant::now();
    {
        let existing_nts: BTreeSet<NonTerminal> = productions.iter().map(|p| p.lhs.clone()).collect();
        let mut null_name_gen = make_null_inline_name_gen(&existing_nts);
        productions = run_null_inline(&productions, &null_inline_strategy, &mut null_name_gen);
    }
    crate::timing!("TIMING: glr_final_epsilon {:?}", t_eps.elapsed());
    print_memory_usage("After final epsilon elimination");
    
    // ============================================================
    // Unreachable Production Elimination
    // ============================================================
    // After grammar transformations (especially right recursion elimination),
    // some non-terminals may become unreachable from the start symbol.
    // Remove these to minimize the grammar and reduce parser states.
    let start_nt_for_unreachable = productions.get(0).map(|p| p.lhs.clone()).unwrap_or(NonTerminal("start".to_string()));
    let before_unreachable = productions.len();
    productions = eliminate_unreachable_productions(&productions, &start_nt_for_unreachable);
    if productions.len() < before_unreachable {
        crate::debug!(4, "Eliminated {} unreachable productions ({} → {})",
            before_unreachable - productions.len(),
            before_unreachable,
            productions.len());
    }
    print_memory_usage("After unreachable elimination");
    
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

    let ignore_terminal_ids = explicit_ignore_terminal_ids;

    if !ignore_terminal_ids.is_empty() {
        crate::debug!(4, "Using {} explicit ignore terminals", ignore_terminal_ids.len());
    }

    // ============================================================
    // Phase 6: DECORATIVE - Left factoring
    // ============================================================
    // Extract common prefixes from productions with the same LHS.
    // If A → α β and A → α γ, create A → α A' and A' → β | γ.
    // This preserves the language but reduces LR states.
    let t_left_factor = std::time::Instant::now();
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
    crate::timing!("TIMING: glr_left_factoring {:?}", t_left_factor.elapsed());
    print_memory_usage("After left factoring");

    // ============================================================
    // Phase 7: DECORATIVE - Merge identical nonterminals
    // ============================================================
    // If two nonterminals have exactly the same set of productions,
    // merge them into one to reduce grammar size and parser states.
    let t_merge = std::time::Instant::now();
    let start_nt_for_merge = productions.get(0).map(|p| p.lhs.clone()).unwrap_or(NonTerminal("start".to_string()));
    let before_merge = productions.len();
    productions = merge_identical_nonterminals(&productions, &start_nt_for_merge);
    crate::timing!("TIMING: glr_merge_identical {:?}", t_merge.elapsed());
    if productions.len() < before_merge {
        crate::debug!(4, "Phase 7: Merged identical nonterminals ({} → {} productions)",
            before_merge,
            productions.len());
    }
    print_memory_usage("After merge identical nonterminals");

    // ============================================================
    // Phase 8: DECORATIVE - Unit production elimination
    // ============================================================
    // Minimize grammar by eliminating unit productions (A → B → X becomes A → X).
    // This reduces parser construction time but doesn't affect correctness.
    let t_unit = std::time::Instant::now();
    let start_nt = &productions.get(0).map(|p| p.lhs.clone()).unwrap_or(NonTerminal("start".to_string()));
    const MAX_SUBSTITUTION_RHS_LEN: usize = 1;
    let (minimized_with_defs, substituted_nts) = substitute_single_productions_and_report(
        &productions,
        start_nt,
        MAX_SUBSTITUTION_RHS_LEN,
    );
    let minimized_productions = remove_productions_for_nts(&minimized_with_defs, &substituted_nts);

    if minimized_productions.len() < productions.len() {
        crate::debug!(4, "Phase 8: Eliminated {} unit productions ({} → {})",
            productions.len() - minimized_productions.len(),
            productions.len(),
            minimized_productions.len());
    }
    productions = minimized_productions;
    crate::timing!("TIMING: glr_unit_elim {:?} ({} prods)", t_unit.elapsed(), productions.len());
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
    // Phase 9: Restore grammar nullability / Finalize Structure
    // ============================================================
    // Handle edge case: grammar was nullable but all productions were eliminated
    // (e.g., grammar was just "S → ε"). We need to restore a minimal nullable structure.
    if productions.is_empty() && grammar_is_nullable {
        crate::debug!(4, "Phase 9: Restoring nullable grammar from empty productions");
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
             crate::debug!(4, "Phase 9: Re-introducing nullability");
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
             crate::debug!(4, "Phase 9: Creating final unique start production");
             
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
    let t_stage1 = std::time::Instant::now();
    let (stage_1_table, item_set_map) =
        stage_1(&light_productions, &lhs_ids, num_terminals, num_nonterminals);
    crate::timing!(
        "TIMING: glr_stage1_lr0 {:?} ({} states)",
        t_stage1.elapsed(),
        stage_1_table.len()
    );
    print_memory_usage("After Stage 1");

    crate::debug!(4, "Computing First/Follow Sets");
    let t_ff = std::time::Instant::now();
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
    crate::timing!("TIMING: glr_first_follow {:?}", t_ff.elapsed());
    print_memory_usage("After First/Follow");

    crate::debug!(4, "Computing Final Table (Merging Stages 2-8)");
    let t_final = std::time::Instant::now();
    let (final_table_map, start_state_id, substring_state_id) = compute_final_table(
        stage_1_table,
        &item_set_map,
        &productions,
        &lhs_ids,
        &follow_sets,
        num_terminals,
    );
    crate::timing!(
        "TIMING: glr_final_table {:?} ({} states)",
        t_final.elapsed(),
        final_table_map.len()
    );
    print_memory_usage("After Final Table");

    // Post-construction table optimization
    crate::debug!(4, "Optimizing Table (state reduction)");
    let t_opt = std::time::Instant::now();
    let (final_table_map, start_state_id, substring_state_id) = optimize_table(
        final_table_map,
        start_state_id,
        substring_state_id,
    );
    crate::timing!(
        "TIMING: glr_table_opt {:?} ({} states)",
        t_opt.elapsed(),
        final_table_map.len()
    );
    print_memory_usage("After Table Optimization");

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

/// Generate a GLR parser from productions, using explicit ignore terminals.
///
/// Terminals declared via `%ignore` are treated as ignorable and removed from productions.
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
