use std::any::Any;
use std::cmp::Ordering;
use crate::datastructures::gss::{print_gss_forest, Acc};
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
use crate::glr::items::{compute_closure, Item};
use crate::glr::table::{Stage7Phase1ShiftsAndReduces, Stage7Phase2ShiftsAndReduces, Stage7Phase3DefaultReduce, Reduce};

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
    pub fn new_without_vocab() -> Self {
        ParseState { stack: Arc::new(GSSNode::new(Acc::new_fresh_without_vocab())) }
    }
    pub fn from_existing(existing: &Self) -> Self {
        ParseState { stack: Arc::new(GSSNode::fresh_from_existing(&existing.stack)) }
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
        self.init_glr_parser_with_acc(Acc::new_fresh_without_vocab())
    }

    pub fn init_glr_parser_null(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: ParseState::new_without_vocab(),
            action_not_found_states: ParseState::new_without_vocab(),
            cycled_states: ParseState::new_without_vocab(),
        }
    }

    pub fn init_glr_parser_with_acc(&self, initial_acc: Acc) -> GLRParserState { // No longer generic
        let initial_parse_state = self.init_parse_state_with_acc(initial_acc);
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            action_not_found_states: ParseState::new_without_vocab(),
            cycled_states: ParseState::new_without_vocab(),
        };
        // Mini phase 3 on initialization
        let mut default_reduce_todo: VecDeque<ParseState> = VecDeque::new();
        default_reduce_todo.push_back(parser_state.active_state.clone());
        let mut next = ParseState::from_existing(&parser_state.active_state);

        timeit!("GLRParserState::init_glr_parser::mini_phase3", {
        while let Some(state) = default_reduce_todo.pop_front() {
            for peek in state.stack.peek_iter() {
                let row = &parser_state.parser.stage_7_table[&peek.edge_value().state_id];
                if row.phase3_default_reduce.clone_and_merge {
                    timeit!("GLRParser::init::mini_phase3::merge", {
                        next.merge(ParseState { stack: peek.to_arc_node() });
                    });
                }
                if let Some(ref r) = row.phase3_default_reduce.reduce {
                    timeit!("GLRParser::init::mini_phase3::reduce", {
                        let new_stack = parser_state.reduce_and_goto(&peek, r.nonterminal_id, r.len);
                        if !new_stack.is_empty() {
                            default_reduce_todo.push_back(ParseState { stack: new_stack });
                        }
                    });
                }
            }
        }
        });
        parser_state.active_state = next;
        parser_state
    }
    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState { // No longer generic
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: parse_state,
            action_not_found_states: ParseState::new_without_vocab(),
            cycled_states: ParseState::new_without_vocab(),
        };
        // Mini phase 3 on initialization
        let mut default_reduce_todo: VecDeque<ParseState> = VecDeque::new();
        default_reduce_todo.push_back(parser_state.active_state.clone());
        let mut next = ParseState::from_existing(&parser_state.active_state);

        timeit!("GLRParserState::init_glr_parser_from_parse_state::mini_phase3", {
        while let Some(state) = default_reduce_todo.pop_front() {
            for peek in state.stack.peek_iter() {
                let row = &parser_state.parser.stage_7_table[&peek.edge_value().state_id];
                if row.phase3_default_reduce.clone_and_merge {
                    timeit!("GLRParser::init_from_state::mini_phase3::merge", {
                        next.merge(ParseState { stack: peek.to_arc_node() });
                    });
                }
                if let Some(ref r) = row.phase3_default_reduce.reduce {
                    timeit!("GLRParser::init_from_state::mini_phase3::reduce", {
                        let new_stack = parser_state.reduce_and_goto(&peek, r.nonterminal_id, r.len);
                        if !new_stack.is_empty() {
                            default_reduce_todo.push_back(ParseState { stack: new_stack });
                        }
                    });
                }
            }
        }
        });
        parser_state.active_state = next;
        parser_state
    }

    pub fn init_parse_state(&self, llm_vocab: Option<Arc<LLMVocab>>) -> ParseState { // No longer generic
        self.init_parse_state_with_acc(Acc::new_fresh_without_vocab())
    }

    pub fn init_parse_state_with_acc(&self, initial_acc: Acc) -> ParseState { // No longer generic
        let initial_user_data: Arc<dyn UserDataTrait> = Arc::new(());
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
        };
        let root = Arc::new(GSSNode::new(Acc::new_fresh_from_existing_stack(&initial_acc))); // root has empty acc
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
                writeln!(&mut result, "  Phase 1 Actions (Full Lookahead):").unwrap();
                let actions = &row.phase1_shifts_and_reduces;
                if actions.is_empty() {
                    writeln!(&mut result, "    (No lookahead actions)").unwrap();
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

                writeln!(&mut result, "  Phase 2 Actions (Full Lookahead):").unwrap();
                let actions = &row.phase2_shifts_and_reduces;
                if actions.is_empty() {
                    writeln!(&mut result, "    (No lookahead actions)").unwrap();
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

                writeln!(&mut result, "  Phase 3 Default Action:").unwrap();
                if let Some(reduce_action) = &row.phase3_default_reduce.reduce {
                    let nt_name = self.non_terminal_map.get_by_right(&reduce_action.nonterminal_id).unwrap();
                    let pids: Vec<String> = reduce_action.production_ids.iter().map(|p| p.0.to_string()).collect();
                    writeln!(&mut result, "    - Default Reduce {} (len {}) via rules [{}]", nt_name.0, reduce_action.len, pids.join(", ")).unwrap();
                } else {
                    writeln!(&mut result, "    (None - state will be merged after shift)").unwrap();
                }
                if row.phase3_default_reduce.clone_and_merge {
                    writeln!(&mut result, "    (State will be merged after shift)").unwrap();
                } else {
                    writeln!(&mut result, "    (State will not be merged after shift)").unwrap();
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
        }

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

            writeln!(f, "    Actions (Phase 1):")?;
            let actions = &row.phase1_shifts_and_reduces;
            for (&terminal_id, action) in actions {
                let terminal = terminal_map.get_by_right(&terminal_id).unwrap();
                match action {
                    Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
                        writeln!(f, "      - {} -> Shift {}", terminal, next_state_id.0)?;
                    }
                    Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nonterminal, len, production_ids } => {
                        let nt_name = non_terminal_map.get_by_right(nonterminal).unwrap();
                        let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
                        writeln!(f, "      - {} -> Reduce {} (len {}) via rules [{}]", terminal, nt_name.0, len, pids.join(", "))?;
                    }
                    Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                        writeln!(f, "      - {} -> Conflict:", terminal)?;
                        if let Some(shift_state) = shift {
                            writeln!(f, "        - Shift {}", shift_state.0)?;
                        }
                        for (len, nts) in reduces {
                            writeln!(f, "        - Reduce (len {}):", len)?;
                            for (nt_id, prod_ids) in nts {
                                let nt = non_terminal_map.get_by_right(nt_id).unwrap();
                                for prod_id_val in prod_ids {
                                    let prod = self.productions.get(prod_id_val.0).expect(format!("Production ID {} not found in productions", prod_id_val.0).as_str());
                                    writeln!(f, "          - {} -> {}", nt.0, prod.lhs.0)?;
                                }
                            }
                        }
                    }
                }
            }

            writeln!(f, "    Actions (Phase 2):")?;
            let actions = &row.phase2_shifts_and_reduces;
            for (&terminal_id, action) in actions {
                let terminal = terminal_map.get_by_right(&terminal_id).unwrap();
                match action {
                    Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
                        writeln!(f, "      - {} -> Shift {}", terminal, next_state_id.0)?;
                    }
                    Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nonterminal, len, production_ids } => {
                        let nt_name = non_terminal_map.get_by_right(nonterminal).unwrap();
                        let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
                        writeln!(f, "      - {} -> Reduce {} (len {}) via rules [{}]", terminal, nt_name.0, len, pids.join(", "))?;
                    }
                    Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                        writeln!(f, "      - {} -> Conflict:", terminal)?;
                        if let Some(shift_state) = shift {
                            writeln!(f, "        - Shift {}", shift_state.0)?;
                        }
                        for (len, nts) in reduces {
                            writeln!(f, "        - Reduce (len {}):", len)?;
                            for (nt_id, prod_ids) in nts {
                                let nt = non_terminal_map.get_by_right(nt_id).unwrap();
                                for prod_id_val in prod_ids {
                                    let prod = self.productions.get(prod_id_val.0).expect(format!("Production ID {} not found in productions", prod_id_val.0).as_str());
                                    writeln!(f, "          - {} -> {}", nt.0, prod.lhs.0)?;
                                }
                            }
                        }
                    }
                }
            }

            writeln!(f, "    Default Action (Phase 3):")?;
            if let Some(reduce) = &row.phase3_default_reduce.reduce {
                let nt_name = non_terminal_map.get_by_right(&reduce.nonterminal_id).unwrap();
                let pids: Vec<String> = reduce.production_ids.iter().map(|p| p.0.to_string()).collect();
                writeln!(f, "      - Default Reduce {} (len {}) via rules [{}]", nt_name.0, reduce.len, pids.join(", "))?;
            } else {
                writeln!(f, "      - None (Merge state)")?;
            }
            if row.phase3_default_reduce.clone_and_merge {
                writeln!(f, "      - Clone and merge after shift")?;
            } else {
                writeln!(f, "      - No clone and merge")?;
            }

            writeln!(f, "    Gotos:")?;
            for (&non_terminal_id, &next_state_id) in &row.gotos {
                let non_terminal = non_terminal_map.get_by_right(&non_terminal_id).unwrap();
                let goto_str = match &next_state_id {
                    Goto::State(state_id_val) => format!("{}", state_id_val.0), // Renamed state_id
                    Goto::Accept => "accept".to_string(),
                };
                writeln!(f, "      - {} -> {}", non_terminal.0, goto_str)?;
            }
        }

        writeln!(f, "\nTerminal Map (name to terminal ID):")?;
        for (terminal, terminal_id) in terminal_map {
            writeln!(f, "  {} -> {}", terminal, terminal_id.0)?;
        }

        writeln!(f, "\nNon-Terminal Map:")?;
        for (non_terminal, non_terminal_id) in non_terminal_map {
            writeln!(f, "  {} -> {}", non_terminal.0, non_terminal_id.0)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLRParserState<'a> { // No longer generic
    pub parser: &'a GLRParser,
    pub active_state: ParseState,
    pub action_not_found_states: ParseState,
    pub cycled_states: ParseState,
}

impl<'a> GLRParserState<'a> { // No longer generic
    fn push_state(
        &self,
        stack: &Arc<GSSNode>, 
        new_content: ParseStateEdgeContent,
    ) -> ParseState {
        crate::debug!(4, "Pushing new state with content: {:?}", new_content);
        let new_gss_node_instance = stack.as_ref().push(new_content, Acc::new_fresh_from_existing_stack2(stack));
        ParseState { stack: Arc::new(new_gss_node_instance) }
    }

    #[time_it("GLRParserState::reduce_and_goto")]
    fn reduce_and_goto(
        &self,
        peek: &GSSPeek,
        nt: NonTerminalID,
        len: usize,
    ) -> Arc<GSSNode> {
        let popped = timeit!(peek.popn(len));
        crate::debug!(4, "Popped with {} results...", popped.num_predecessors());
        crate::debug!(6, "Reducing by {} with parent node: {}", len, print_gss_forest(&[Arc::new(peek.parent_node.clone())], None, 30, &self.parser.terminal_map, None, None));
        crate::debug!(6, "...and predecessor node: {}", print_gss_forest(&[peek.predecessor_node.clone()], None, 30, &self.parser.terminal_map, None, None));
        crate::debug!(6, "...and popped peek node: {}", print_gss_forest(&[popped.clone()], None, 30, &self.parser.terminal_map, None, None));
        // let mut out = GSSNode::new(Acc::new_for_merging()); // Start with a default acc
        let mut out = Vec::new();
        timeit!("GLRParserState::reduce_and_goto::process_peeks", {
        for popped_peek in popped.peek_iter() { // Renamed predecessor to predecessor_arc
            let goto = self.parser.stage_7_table.get(&popped_peek.edge_value().state_id).map_or_else(|| Err(format!("State {} not found in stage_7_table", popped_peek.edge_value().state_id.0)), |row| row.gotos.get(&nt).map_or_else(|| Err(format!("Non-terminal {} not found in gotos for {:?} (processing predecessor ??)", nt.0, popped_peek.edge_value().state_id)), |state_id| Ok(*state_id))).unwrap();
            match goto {
                Goto::State(goto_state_id) => {
                    timeit!("GLRParserState::reduce_and_goto::process_peaks::push_with_existing_acc", {
                    // crate::debug!(4, " ...and edge value {:?}, predecessor {:p}, goto state ID {}", edge_value.state_id, Arc::as_ptr(&predecessor_arc), goto_state_id.0);
                    crate::debug!(6, "Popped peek parent node: {}", print_gss_forest(&[Arc::new(popped_peek.parent_node.clone())], None, 30, &self.parser.terminal_map, None, None));
                    crate::debug!(6, "Popped peek predecessor node: {}", print_gss_forest(&[popped_peek.predecessor_node.clone()], None, 30, &self.parser.terminal_map, None, None));
                    let popped_peek_node = popped_peek.to_node();
                    let new_gss_node = popped_peek_node.push_with_existing_acc(ParseStateEdgeContent { state_id: goto_state_id });
                    crate::debug!(6, "Popped peek node to_node: {}", print_gss_forest(&[Arc::new(popped_peek.to_node())], None, 30, &self.parser.terminal_map, None, None));
                    crate::debug!(6, "New GSS node after reduction: {}", print_gss_forest(&[Arc::new(new_gss_node.clone())], None, 30, &self.parser.terminal_map, None, None));
                    out.push(new_gss_node);
                    });
                }
                Goto::Accept => {
                    // No action needed for Accept
                }
            }
        }
        });
        timeit!("GLRParserState::reduce_and_goto::merge_results", {
        if out.is_empty() {
            Arc::new(GSSNode::new(Acc::new_fresh_from_existing_stack2(&peek.predecessor_node)))
        } else if out.len() == 1 {
            Arc::new(out.into_iter().next().unwrap())
        } else {
            let mut out_iter = out.into_iter();
            let mut out_node = out_iter.next().unwrap();
            for next_node in out_iter {
                out_node.merge(&next_node);
            }
            Arc::new(out_node)
        }
        })
    }

    #[time_it("GLRParserState::step")]
    pub fn step(&mut self, token_id: TerminalID) {
        if Some(token_id) == self.parser.ignore_terminal_id {
            crate::debug!(4, "Ignoring token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());
            return;
        }

        let token_display = self.parser.terminal_map.get_by_right(&token_id).unwrap();
        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        self.log_gss("Step-start", token_id);

        let mut phase1_todo: VecDeque<ParseState> = VecDeque::new();
        phase1_todo.push_back(ParseState { stack: self.active_state.stack.clone() });

        let mut phase2_todo: VecDeque<ParseState> = VecDeque::new();

        let mut phase3_todo: VecDeque<ParseState> = VecDeque::new();
        let mut next = ParseState::from_existing(&self.active_state);

        // --- Phase 1: Process lookahead-based actions excluding reductions promoted to default ---
        crate::debug!(4, "--- Phase 1: Processing initial states ({} items in todo) ---", phase1_todo.len());
        timeit!("GLRParserState::step::phase1", {
        while let Some(state) = phase1_todo.pop_front() {
            for peek in state.stack.peek_iter() {
                let row = &self.parser.stage_7_table[&peek.edge_value().state_id];
                crate::debug!(5, "Phase 1: Peeking state {}, looking for action on token '{}'", peek.edge_value().state_id.0, token_display);
                // We use phase1 actions here, which exclude the default reduce action,
                // as that should have been handled at the end of the previous step.
                if let Some(action) = row.phase1_shifts_and_reduces.get(&token_id) {
                    crate::debug!(4, "Phase 1: Found action {:?} for state {} on token '{}'", action, peek.edge_value().state_id.0, token_display);
                    match action {
                        Stage7ShiftsAndReducesLookaheadValue::Shift(to) => {
                            timeit!("GLRParserState::step::phase1::shift", {
                            let stack_for_push = peek.to_arc_node();
                            let new_content = ParseStateEdgeContent { state_id: *to };
                            let new_parse_state = self.push_state(&stack_for_push, new_content);
                            phase3_todo.push_back(new_parse_state);
                            });
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nt, len, .. } => {
                            timeit!("GLRParserState::step::phase1::reduce", {
                            let s_new_arc = self.reduce_and_goto(&peek, *nt, *len);
                            if !s_new_arc.is_empty() {
                                phase2_todo.push_back(ParseState { stack: s_new_arc });
                            }
                            });
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                            timeit!("GLRParserState::step::phase1::split", {
                            if let Some(to) = shift {
                                timeit!("GLRParserState::step::phase1::split::shift", {
                                let stack_for_push = peek.to_arc_node();
                                let new_content = ParseStateEdgeContent { state_id: *to };
                                let new_parse_state = self.push_state(&stack_for_push, new_content);
                                phase3_todo.push_back(new_parse_state);
                                });
                            }
                            for (len, nts) in reduces {
                                for (nt, _prod_ids) in nts {
                                    timeit!("GLRParserState::step::phase1::split::reduce", {
                                    let s_new_arc = self.reduce_and_goto(&peek, *nt, *len);
                                    if !s_new_arc.is_empty() {
                                        phase2_todo.push_back(ParseState { stack: s_new_arc });
                                    }
                                    });
                                }
                            }
                            });
                        }
                    }
                } else {
                    crate::debug!(5, "Phase 1: No specific action for state {}, token '{}'. Path will be pruned if no default reduce applies later.", peek.edge_value().state_id.0, token_display);
                }
            }
        }
        });

        // // Merge before Phase 2
        // if !phase2_todo.is_empty() {
        //     crate::debug!(4, "Merging phase2_todo before Phase 2");
        //     let mut merged_phase2 = phase2_todo.pop_front().unwrap();
        //     for state in std::mem::take(&mut phase2_todo) {
        //         merged_phase2.merge(state);
        //     }
        //     phase2_todo.push_back(merged_phase2);
        // }

        crate::debug!(4, "--- Phase 2: Processing states from reductions ({} items in todo) ---", phase2_todo.len());
        // --- Phase 2: Process lookahead-based actions ---
        timeit!("GLRParserState::step::phase2", {
        while let Some(state) = phase2_todo.pop_front() {
            for peek in state.stack.peek_iter() {
                let row = &self.parser.stage_7_table[&peek.edge_value().state_id];
                crate::debug!(5, "Phase 2: Peeking state {}, looking for action on token '{}'", peek.edge_value().state_id.0, token_display);
                // We use phase2 actions here, which include all lookahead-specific actions.
                if let Some(action) = row.phase2_shifts_and_reduces.get(&token_id) {
                    crate::debug!(4, "Phase 2: Found action {:?} for state {} on token '{}'", action, peek.edge_value().state_id.0, token_display);
                    match action {
                        Stage7ShiftsAndReducesLookaheadValue::Shift(to) => {
                            timeit!("GLRParserState::step::phase2::shift", {
                                let stack_for_push = peek.to_arc_node();
                                let new_content = ParseStateEdgeContent { state_id: *to };
                                let new_parse_state = self.push_state(&stack_for_push, new_content);
                                phase3_todo.push_back(new_parse_state);
                            });
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nt, len, .. } => {
                            timeit!("GLRParserState::step::phase2::reduce", {
                                let s_new_arc = self.reduce_and_goto(&peek, *nt, *len);
                                if !s_new_arc.is_empty() {
                                    phase2_todo.push_back(ParseState { stack: s_new_arc });
                                }
                            });
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                            timeit!("GLRParserState::step::phase2::split", {
                                if let Some(to) = shift {
                                    timeit!("GLRParserState::step::phase2::split::shift", {
                                        let stack_for_push = peek.to_arc_node();
                                        let new_content = ParseStateEdgeContent { state_id: *to };
                                        let new_parse_state = self.push_state(&stack_for_push, new_content);
                                        phase3_todo.push_back(new_parse_state);
                                    });
                                }
                                for (len, nts) in reduces {
                                    for (nt, _prod_ids) in nts {
                                        timeit!("GLRParserState::step::phase2::split::reduce", {
                                            let s_new_arc = self.reduce_and_goto(&peek, *nt, *len);
                                            if !s_new_arc.is_empty() {
                                                phase2_todo.push_back(ParseState { stack: s_new_arc });
                                            }
                                        });
                                    }
                                }
                            });
                        }
                    }
                } else {
                    crate::debug!(5, "Phase 2: No specific action for state {}, token '{}'. Path will be pruned if no default reduce applies later.", peek.edge_value().state_id.0, token_display);
                }
            }
        }
        });

        // // Merge before Phase 3
        // if !phase3_todo.is_empty() {
        //     crate::debug!(4, "Merging phase3_todo before Phase 3");
        //     // Merge all states in phase3_todo into next
        //     let mut merged_phase3 = phase3_todo.pop_front().unwrap();
        //     for state in std::mem::take(&mut phase3_todo) {
        //         merged_phase3.merge(state);
        //     }
        //     phase3_todo.push_back(merged_phase3);
        // }

        crate::debug!(4, "--- Phase 3: Processing states from shifts ({} items in todo) ---", phase3_todo.len());
        // --- Phase 3: Process default reductions on post-shift states ---
        timeit!("GLRParserState::step::phase3", {
        while let Some(state) = phase3_todo.pop_front() {
            for peek in state.stack.peek_iter() {
                let row = &self.parser.stage_7_table[&peek.edge_value().state_id];
                crate::debug!(5, "Phase 3: Peeking state {}", peek.edge_value().state_id.0);
                if let Some(ref r) = row.phase3_default_reduce.reduce {
                    timeit!("GLRParserState::step::phase3::reduce", {
                        crate::debug!(4, "Phase 3: Found default reduce {:?} for state {}", r, peek.edge_value().state_id.0);
                        // This peek has a default reduce.
                        // The result of the reduction goes back on the todo list for this phase.
                        let new_stack = self.reduce_and_goto(&peek, r.nonterminal_id, r.len);
                        if !new_stack.is_empty() {
                            crate::debug!(5, "Phase 3: Pushing result of default reduce to phase3_todo");
                            phase3_todo.push_back(ParseState { stack: new_stack });
                        }
                    });
                }
                if row.phase3_default_reduce.clone_and_merge {
                    timeit!("GLRParserState::step::phase3::merge", {
                        crate::debug!(4, "Phase 3: Merging state {} into next active states", peek.edge_value().state_id.0);
                        next.merge(ParseState { stack: peek.to_arc_node() });
                    });
                }
            }
        }
        });

        self.active_state = next;

        if !self.active_state.stack.is_empty() {
            self.log_gss("Step-end", token_id);
        }
        crate::debug!(4, "----------------------------------------------------------------");
    }

    pub fn parse(&mut self, input: &[TerminalID]) {
        self.parse_part(input);
    }

    pub fn parse_part(&mut self, input: &[TerminalID]) {
        for &token_id in input {
            self.step(token_id);
        }
    }

    pub fn and_step(mut self, token_id: TerminalID) -> Self {
        self.step(token_id);
        self
    }

    pub fn and_parse(mut self, input: &[TerminalID]) -> Self {
        self.parse(input);
        self
    }

    pub fn merge_active_states(&mut self) {
        // No longer strictly necessary due to BTreeMap merge-on-insert, but GSS merge is explicit.
        // This method could be used if multiple GLRParserStates are combined.
    }

    pub fn merge_with(&mut self, other: GLRParserState) { // No longer generic
        assert!(std::ptr::eq(self.parser, other.parser));
        self.active_state.merge(other.active_state);
        // self.action_not_found_states.merge(other.action_not_found_states);
        // self.cycled_states.merge(other.cycled_states);
    }

    pub fn is_ok(&self) -> bool {
        !self.active_state.stack.is_empty() && self.active_state.stack.is_alive()
    }

    // #[time_it("GLRParserState::log_gss")]
    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
        // crate::debug!(3, "{} - token {} ({:?}) - nodes", phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0));
        // return;
        const MAX: usize = 30;
        const PANIC_THRESHOLD: usize = 10000;

        let roots: Vec<_> = vec![self.active_state.stack.clone()];
        let stats = gather_gss_stats(&roots.iter().map(|r| r.as_ref()).collect::<Vec<_>>());
        crate::debug!(4, "{} - token '{}' ({}) - nodes: {:?}",
                      phase, self.parser.terminal_map.get_by_right(&token).unwrap(), token.0, stats);

        let make_msg = |print_full_forest, max_nodes_to_print| {
            if print_full_forest {
                format!("GSS ({} nodes):\n{}", stats.unique_nodes,
                        print_gss_forest(&roots, None, max_nodes_to_print, &self.parser.terminal_map, None, None))
            } else {
                match find_longest_path(&self.active_state.stack) {
                    Some(p) => format!("GSS too big ({} nodes). Longest path ({}): {}",
                                       stats.unique_nodes,
                                       p.len(),
                                       p.iter().map(|(ec, _n)| ec.state_id.0) // n is Arc<GSSNode>
                                            .map(|id| id.to_string())
                                            .collect::<Vec<_>>()
                                            .join(" → ")),
                    None => format!("GSS too big ({} nodes) – path not found", stats.unique_nodes),
                }
            }
        };

        if stats.unique_nodes > PANIC_THRESHOLD {
            let msg = make_msg(true, usize::MAX);
            panic!("GSS too big ({} nodes). {}", stats.unique_nodes, msg);
        }

        debug!(5, "{}", make_msg(stats.unique_nodes <= MAX, MAX));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
}

impl ParseState { // No longer generic
    pub fn merge(&mut self, other: ParseState) {
        Arc::make_mut(&mut self.stack).merge(&other.stack);
    }
}

pub trait InsertWith<K, V> {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F);
}

impl<K, V> InsertWith<K, V> for BTreeMap<K, V> where K: Eq + Ord {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F) {
        match self.entry(k) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                let value = occupied.get_mut();
                combine(value, v);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(v);
            }
        }
    }
}
