use std::any::Any;
use std::cmp::Ordering;
use crate::datastructures::gss::{print_gss_forest};
use crate::datastructures::gss::{gather_gss_stats, find_longest_path, GSSNode, GSSStats, GSSPeek};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Stage7ShiftsAndReducesLookaheadValue, Stage7Table, StateID, TerminalID};
use crate::constraint::{LLMTokenBV, LLMVocab}; // Import LLMTokenInfo

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::{Debug, Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::debug;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use profiler_macro::{time_it, timeit};
use crate::datastructures::gss::Acc;
use crate::glr::items::{compute_closure, Item};

pub trait DynEq {
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}
pub trait DynOrd {
    fn dyn_cmp(&self, other: &dyn Any) -> std::cmp::Ordering;
}
pub trait DynHash {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
}
impl DynEq for () {
    fn dyn_eq(&self, _other: &dyn Any) -> bool { true }
}
impl DynOrd for () {
    fn dyn_cmp(&self, _other: &dyn Any) -> std::cmp::Ordering { std::cmp::Ordering::Equal }
}
impl DynHash for () {
    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) { }
}

pub trait UserDataTrait: Any + Send + Sync + Debug + DynEq + DynOrd + DynHash {}
impl UserDataTrait for () {}

pub type ActionFn = Arc<dyn Fn(&mut Arc<dyn UserDataTrait>) -> bool + Send + Sync>;


#[derive(Debug, Clone)]
pub struct ParseStateEdgeContent { 
    pub state_id: StateID,
}
impl PartialEq for ParseStateEdgeContent {
    fn eq(&self, other: &Self) -> bool {
        self.state_id == other.state_id
    }
}
impl Eq for ParseStateEdgeContent {}
impl PartialOrd for ParseStateEdgeContent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.state_id.partial_cmp(&other.state_id) {
            other_ord => other_ord,
        }
    }
}
impl Ord for ParseStateEdgeContent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.state_id.cmp(&other.state_id)
    }
}
impl Hash for ParseStateEdgeContent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state_id.hash(state);
    }
}

// JSONConvertible for ParseStateEdgeContent
impl JSONConvertible for ParseStateEdgeContent {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        // Handle user_data serialization:
        // Option 1: Panic if not default.
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for ParseStateEdgeContent".to_string()) // Corrected struct name
                                  .and_then(StateID::from_json)?;
                Ok(ParseStateEdgeContent { state_id })
            }
            _ => Err("Expected JSONNode::Object for ParseStateEdgeContent".to_string()), // Corrected struct name
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState { // No longer generic
    pub stack: Arc<GSSNode>, // GSSNode is now concrete
}

impl ParseState {
    pub fn new(llm_vocab: Option<Arc<LLMVocab>>) -> Self {
        ParseState { stack: Arc::new(GSSNode::new(Acc::new_fresh(llm_vocab))) }
    }
    pub fn from_existing(existing: Self) -> Self {
        ParseState::new(existing.stack.llm_tokens().llm_vocab().clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StopReason {
    ActionNotFound,
    GotoNotFound,
}
impl JSONConvertible for StopReason {
    fn to_json(&self) -> JSONNode {
        let variant_name = match self {
            StopReason::ActionNotFound => "ActionNotFound",
            StopReason::GotoNotFound => "GotoNotFound",
        };
        JSONNode::String(variant_name.to_string())
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::String(s) => match s.as_str() {
                "ActionNotFound" => Ok(StopReason::ActionNotFound),
                "GotoNotFound" => Ok(StopReason::GotoNotFound),
                _ => Err(format!("Unknown variant {} for StopReason", s)),
            },
            _ => Err("Expected JSONNode::String for StopReason".to_string()),
        }
    }
}


#[derive(Clone)]
pub struct GLRParser {
    pub stage_7_table: Stage7Table,
    pub productions: Vec<Production>,
    pub terminal_map: BiBTreeMap<Terminal, TerminalID>,
    pub non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    pub item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
    pub start_state_id: StateID,
    pub ignore_terminal_id: Option<TerminalID>,
}

impl JSONConvertible for GLRParser {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("stage_7_table".to_string(), self.stage_7_table.to_json());
        obj.insert("productions".to_string(), self.productions.to_json());
        obj.insert("terminal_map".to_string(), self.terminal_map.to_json());
        obj.insert("non_terminal_map".to_string(), self.non_terminal_map.to_json());
        obj.insert("item_set_map".to_string(), self.item_set_map.to_json());
        obj.insert("start_state_id".to_string(), self.start_state_id.to_json());
        obj.insert("ignore_terminal_id".to_string(), self.ignore_terminal_id.to_json());
        // Do not serialize self.actions
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let stage_7_table = obj.remove("stage_7_table").ok_or_else(|| "Missing field stage_7_table".to_string())
                                       .and_then(Stage7Table::from_json)?;
                let productions = obj.remove("productions").ok_or_else(|| "Missing field productions".to_string())
                                     .and_then(Vec::<Production>::from_json)?;
                let terminal_map = obj.remove("terminal_map").ok_or_else(|| "Missing field terminal_map".to_string())
                                      .and_then(|n| BiBTreeMap::<Terminal, TerminalID>::from_json(n))?;
                let non_terminal_map = obj.remove("non_terminal_map").ok_or_else(|| "Missing field non_terminal_map".to_string())
                                          .and_then(|n| BiBTreeMap::<NonTerminal, NonTerminalID>::from_json(n))?;
                let item_set_map = obj.remove("item_set_map").ok_or_else(|| "Missing field item_set_map".to_string())
                                      .and_then(|n| BiBTreeMap::<BTreeSet<Item>, StateID>::from_json(n))?;
                let start_state_id = obj.remove("start_state_id").ok_or_else(|| "Missing field start_state_id".to_string())
                                        .and_then(StateID::from_json)?;
                let ignore_terminal_id = obj.remove("ignore_terminal_id")
                    .ok_or_else(|| "Missing field ignore_terminal_id for GLRParser".to_string())
                    .and_then(Option::<TerminalID>::from_json)?;
                Ok(GLRParser {
                    stage_7_table,
                    productions,
                    terminal_map,
                    non_terminal_map,
                    item_set_map,
                    start_state_id,
                    ignore_terminal_id,
                })
            }
            _ => Err("Expected JSONNode::Object for GLRParser".to_string()),
        }
    }
}

impl Debug for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLRParser")
            .field("stage_7_table", &self.stage_7_table)
            .field("productions", &self.productions)
            .field("terminal_map", &self.terminal_map)
            .field("non_terminal_map", &self.non_terminal_map)
            .field("item_set_map", &self.item_set_map)
            .field("start_state_id", &self.start_state_id)
            .field("ignore_terminal_id", &self.ignore_terminal_id)
            .finish()
    }
}

impl PartialEq for GLRParser {
    fn eq(&self, other: &Self) -> bool {
        self.stage_7_table == other.stage_7_table &&
        self.productions == other.productions &&
        self.terminal_map == other.terminal_map &&
        self.non_terminal_map == other.non_terminal_map &&
        self.item_set_map == other.item_set_map &&
        self.start_state_id == other.start_state_id &&
        self.ignore_terminal_id == other.ignore_terminal_id
    }
}

impl Eq for GLRParser {}

impl GLRParser {
    pub fn new(
        stage_7_table: Stage7Table,
        productions: Vec<Production>,
        terminal_map: BiBTreeMap<Terminal, TerminalID>,
        non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
        item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
        start_state_id: StateID,
        actions: BTreeMap<NonTerminal, ActionFn>, // Parameter type
        ignore_terminal_id: Option<TerminalID>,
    ) -> Self {
        let converted_actions: BTreeMap<NonTerminalID, ActionFn> = actions
            .into_iter()
            .map(|(nt, func)| {
                let nt_id = non_terminal_map.get_by_left(&nt)
                    .unwrap_or_else(|| panic!("NonTerminal {:?} not found in non_terminal_map during GLRParser construction", nt));
                (*nt_id, func)
            })
            .collect();

        Self {
            stage_7_table,
            productions,
            terminal_map,
            non_terminal_map,
            item_set_map,
            start_state_id,
            ignore_terminal_id,
        }
    }

    pub fn init_glr_parser(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        self.init_glr_parser_with_acc(Acc::new_fresh(llm_vocab))
    }

    pub fn init_glr_parser_null(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: ParseState::new(llm_vocab.clone()),
            action_not_found_states: ParseState::new(None),
            cycled_states: ParseState::new(None),
        }
    }

    pub fn init_glr_parser_with_acc(&self, initial_acc: Acc) -> GLRParserState { // No longer generic
        let initial_parse_state = self.init_parse_state_with_acc(initial_acc);
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            action_not_found_states: ParseState::new(None),
            cycled_states: ParseState::new(None),
        }
    }
    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: parse_state,
            action_not_found_states: ParseState::new(None),
            cycled_states: ParseState::new(None),
        }
    }

    pub fn init_parse_state(&self, llm_vocab: Option<Arc<LLMVocab>>) -> ParseState { // No longer generic
        self.init_parse_state_with_acc(Acc::new_fresh(llm_vocab))
    }

    pub fn init_parse_state_with_acc(&self, initial_acc: Acc) -> ParseState { // No longer generic
        let initial_user_data: Arc<dyn UserDataTrait> = Arc::new(());
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
        };
        let llm_vocab = initial_acc.llm_tokens().llm_vocab().clone();
        let root = Arc::new(GSSNode::new(Acc::new_fresh(llm_vocab))); // root has empty acc
        let stack = Arc::new(root.as_ref().push(initial_content, initial_acc)); // pushed node has initial_acc
        ParseState { stack }
    }

    pub fn parse(&self, input: &[TerminalID], llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        let mut state = self.init_glr_parser(llm_vocab);
        state.parse(input);
        state
    }

    pub fn explain_stack(&self, stack: &[StateID]) -> String {
        let mut result = String::new();
        writeln!(&mut result, "--- Explaining Parse Stack: {:?} ---", stack.iter().map(|s| s.0).collect::<Vec<_>>()).unwrap();

        for &state_id in stack {
            writeln!(&mut result, "\nState {}:", state_id.0).unwrap();

            // Get and print items
            if let Some(core_items) = self.item_set_map.get_by_right(&state_id) {
                let full_closure = compute_closure(core_items, &self.productions);
                let closure_only_items: BTreeSet<_> = full_closure.difference(core_items).cloned().collect();

                writeln!(&mut result, "  Core Items:").unwrap();
                if core_items.is_empty() {
                    writeln!(&mut result, "    (None)").unwrap();
                } else {
                    for item in core_items {
                        writeln!(&mut result, "    - {}", item).unwrap();
                    }
                }

                if !closure_only_items.is_empty() {
                    writeln!(&mut result, "  Closure-only Items:").unwrap();
                    for item in &closure_only_items {
                        writeln!(&mut result, "    - {}", item).unwrap();
                    }
                }
            } else {
                writeln!(&mut result, "  (State ID not found in item set map)").unwrap();
            }

            // Get and print actions
            if let Some(row) = self.stage_7_table.get(&state_id) {
                writeln!(&mut result, "  Actions:").unwrap();
                let actions = &row.phase2_shifts_and_reduces;
                if actions.is_empty() {
                    writeln!(&mut result, "    (No shift/reduce actions)").unwrap();
                } else {
                    // Sort by terminal name for consistent output
                    let mut sorted_actions: Vec<_> = actions.iter().collect();
                    sorted_actions.sort_by_key(|(tid, _)| self.terminal_map.get_by_right(tid).unwrap());

                    for (terminal_id, action) in sorted_actions {
                        let terminal = &self.terminal_map.get_by_right(terminal_id).unwrap();
                        write!(&mut result, "    - On '{}': ", terminal).unwrap();
                        match action {
                            Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
                                writeln!(&mut result, "Shift to State {}", next_state_id.0).unwrap();
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Reduce { production_ids, .. } => {
                                if production_ids.len() == 1 {
                                    let prod_id = production_ids.iter().next().unwrap();
                                    let prod = &self.productions[prod_id.0];
                                    writeln!(&mut result, "Reduce by rule #{} ({})", prod_id.0, prod).unwrap();
                                } else {
                                    let pids: Vec<String> = production_ids.iter().map(|p| format!("#{}", p.0)).collect();
                                    writeln!(&mut result, "Reduce by rules {}", pids.join(", ")).unwrap();
                                }
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                writeln!(&mut result, "Conflict:").unwrap();
                                if let Some(shift_state) = shift {
                                    writeln!(&mut result, "      - Shift to State {}", shift_state.0).unwrap();
                                }
                                for (_len, nts) in reduces {
                                    for (_nt_id, prod_ids) in nts {
                                        for prod_id in prod_ids {
                                            let prod = &self.productions[prod_id.0];
                                            writeln!(&mut result, "      - Reduce by rule #{} ({})", prod_id.0, prod).unwrap();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(reduce) = &row.phase3_default_reduces.reduce {
                    let nt_name = self.non_terminal_map.get_by_right(&reduce.nonterminal_id).unwrap();
                    let pids: Vec<String> = reduce.production_ids.iter().map(|p| p.0.to_string()).collect();
                    writeln!(&mut result, "    - Default Reduce {} (len {}) via rules [{}]", nt_name.0, reduce.len, pids.join(", ")).unwrap();
                }
 
                 writeln!(&mut result, "  Gotos:").unwrap();
                 if row.gotos.is_empty() {
                     writeln!(&mut result, "    (No goto actions)").unwrap();
                 } else {
                     // Sort by non-terminal name
                     let mut sorted_gotos: Vec<_> = row.gotos.iter().collect();
                     sorted_gotos.sort_by_key(|(ntid, _)| self.non_terminal_map.get_by_right(ntid).unwrap());
 
                     for (non_terminal_id, goto) in sorted_gotos {
                         let non_terminal_name = &self.non_terminal_map.get_by_right(non_terminal_id).unwrap().0;
                         write!(&mut result, "    - On '{}': ", non_terminal_name).unwrap();
                         match goto {
                             Goto::State(next_state_id) => writeln!(&mut result, "Goto State {}", next_state_id.0).unwrap(),
                             Goto::Accept => writeln!(&mut result, "Accept").unwrap(),
                         }
                     }
                 }
 
             } else {
                 writeln!(&mut result, "  (State ID not found in parse table)").unwrap();
             }
             writeln!(&mut result, "---").unwrap();
 
         result
     }
 }
 
 impl Display for GLRParser {
     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
         let stage_7_table = &self.stage_7_table;
         let terminal_map = &self.terminal_map;
         let non_terminal_map = &self.non_terminal_map;
         let item_set_map = &self.item_set_map;
 
         use crate::glr::items::{compute_closure, Item};
         use std::collections::BTreeSet;
 
         writeln!(f, "Parse Table:")?;
         writeln!(f, "  Start State: {}", self.start_state_id.0)?;
         for (&state_id, row) in stage_7_table.iter().collect::<BTreeMap<_, _>>() {
             writeln!(f, "  State {}:", state_id.0)?;
 
             let core_item_set = item_set_map.get_by_right(&state_id).unwrap();
             let full_closure = compute_closure(core_item_set, &self.productions);
 
             writeln!(f, "    Core Items:")?;
             for item in core_item_set {
                 write!(f, "      - {} ->", item.production.lhs.0)?;
                 for (i, symbol) in item.production.rhs.iter().enumerate() {
                     if i == item.dot_position {
                         write!(f, " •")?;
                     }
                     match symbol {
                         Symbol::Terminal(terminal) => write!(f, " {}", terminal),
                         Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0),
                     }?;
                 }
                 if item.dot_position == item.production.rhs.len() {
                     write!(f, " •")?;
                 }
                 writeln!(f)?;
             }
 
             let closure_only_items: BTreeSet<_> = full_closure.difference(core_item_set).cloned().collect();
             if !closure_only_items.is_empty() {
                 writeln!(f, "    Closure Items:")?;
                 for item in &closure_only_items {
                     write!(f, "      - {} ->", item.production.lhs.0)?;
                     for (i, symbol) in item.production.rhs.iter().enumerate() {
                         if i == item.dot_position {
                             write!(f, " •")?;
                         }
                         match symbol {
                             Symbol::Terminal(terminal) => write!(f, " {}", terminal),
                             Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0),
                         }?;
                     }
                     if item.dot_position == item.production.rhs.len() {
                         write!(f, " •")?;
                     }
                     writeln!(f)?;
                 }
             }
 
             writeln!(f, "    Actions:")?;
-            match &row.shifts_and_reduces {
-                Stage7ShiftsAndReduces::Lookahead(actions) => {
-                    for (&terminal_id, action) in actions {
-                        let terminal = terminal_map.get_by_right(&terminal_id).unwrap();
-                        match action {
-                            Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
-                                writeln!(f, "      - {} -> Shift {}", terminal, next_state_id.0)?;
-                            }
-                            Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nonterminal, len, production_ids } => {
-                                let nt_name = non_terminal_map.get_by_right(nonterminal).unwrap();
-                                let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
-                                writeln!(f, "      - {} -> Reduce {} (len {}) via rules [{}]", terminal, nt_name.0, len, pids.join(", "))?;
-                            }
-                            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
-                                writeln!(f, "      - {} -> Conflict:", terminal)?;
-                                if let Some(shift_state) = shift {
-                                    writeln!(f, "        - Shift {}", shift_state.0)?;
-                                }
-                                for (len, nts) in reduces {
-                                    writeln!(f, "        - Reduce (len {}):", len)?;
-                                    for (nt_id, prod_ids) in nts {
-                                        let nt = non_terminal_map.get_by_right(nt_id).unwrap();
-                                        for prod_id_val in prod_ids {
-                                            let prod = self.productions.get(prod_id_val.0).expect(format!("Production ID {} not found in productions", prod_id_val.0).as_str());
-                                            writeln!(f, "          - {} -> {}", nt.0, prod.lhs.0)?;
-                                        }
-                                    }
-                                }
-                            }
-                        }
-                    }
-                }
-                Stage7ShiftsAndReduces::DefaultReduce { nonterminal_id, len, production_ids } => {
-                    let nt_name = non_terminal_map.get_by_right(nonterminal_id).unwrap();
-                    let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
-                    writeln!(f, "      - Default Reduce {} (len {}) via rules [{}]", nt_name.0, len, pids.join(", "))?;
-                }
+            for (&terminal_id, action) in &row.phase2_shifts_and_reduces {
+                let terminal = terminal_map.get_by_right(&terminal_id).unwrap();
+                match action {
+                    Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
+                        writeln!(f, "      - {} -> Shift {}", terminal, next_state_id.0)?;
+                    }
+                    Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nonterminal, len, production_ids } => {
+                        let nt_name = non_terminal_map.get_by_right(nonterminal).unwrap();
+                        let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
+                        writeln!(f, "      - {} -> Reduce {} (len {}) via rules [{}]", terminal, nt_name.0, len, pids.join(", "))?;
+                    }
+                    Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
+                        writeln!(f, "      - {} -> Conflict:", terminal)?;
+                        if let Some(shift_state) = shift {
+                            writeln!(f, "        - Shift {}", shift_state.0)?;
+                        }
+                        for (len, nts) in reduces {
+                            writeln!(f, "        - Reduce (len {}):", len)?;
+                            for (nt_id, prod_ids) in nts {
+                                let nt = non_terminal_map.get_by_right(nt_id).unwrap();
+                                for prod_id_val in prod_ids {
+                                    let prod = self.productions.get(prod_id_val.0).expect(format!("Production ID {} not found in productions", prod_id_val.0).as_str());
+                                    writeln!(f, "          - {} -> {}", nt.0, prod.lhs.0)?;
+                                }
+                            }
+                        }
+                    }
+                }
+            }
+            if let Some(reduce) = &row.phase3_default_reduces.reduce {
+                let nt_name = non_terminal_map.get_by_right(&reduce.nonterminal_id).unwrap();
+                let pids: Vec<String> = reduce.production_ids.iter().map(|p| p.0.to_string()).collect();
+                writeln!(f, "      - Default Reduce {} (len {}) via rules [{}]", nt_name.0, len, pids.join(", "))?;
             }
 
             writeln!(f, "    Gotos:")?;
