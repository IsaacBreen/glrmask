use std::any::Any;
use std::cmp::Ordering;
use crate::datastructures::gss::print_gss_forest;
use crate::datastructures::gss::{gather_gss_stats, find_longest_path, PathAccumulator, GSSNode, GSSTrait, GSSStats};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID};
use crate::constraint::{LLMTokenBV, LLMTokenInfo}; // Import LLMTokenInfo

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::debug;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;

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
    pub user_data: Arc<dyn UserDataTrait>, // Changed to Arc<dyn UserDataTrait>
}
impl PartialEq for ParseStateEdgeContent {
    fn eq(&self, other: &Self) -> bool {
        self.state_id == other.state_id &&
        self.user_data.dyn_eq(&other.user_data)
    }
}
impl Eq for ParseStateEdgeContent {}
impl PartialOrd for ParseStateEdgeContent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.state_id.partial_cmp(&other.state_id) {
            Some(Ordering::Equal) => {
                Some(self.user_data.dyn_cmp(&other.user_data))
            }
            other_ord => other_ord,
        }
    }
}
impl Ord for ParseStateEdgeContent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.state_id.cmp(&other.state_id)
            .then_with(|| self.user_data.dyn_cmp(&other.user_data))
    }
}
impl Hash for ParseStateEdgeContent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state_id.hash(state);
        self.user_data.dyn_hash(state);
    }
}

// JSONConvertible for ParseStateEdgeContent
impl JSONConvertible for ParseStateEdgeContent {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        // Handle user_data serialization:
        // Option 1: Panic if not default.
        if self.user_data.type_id() == std::any::TypeId::of::<()>() {
             // Optionally, add a marker: obj.insert("user_data_type".to_string(), JSONNode::String("default_unit".to_string()));
        } else {
            panic!("Cannot serialize ParseStateEdgeContent with non-default user_data of type {:?}", self.user_data.type_id());
        }
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for ParseStateEdgeContent".to_string()) // Corrected struct name
                                  .and_then(StateID::from_json)?;
                // Always deserialize user_data as default Arc::new(())
                let user_data: Arc<dyn UserDataTrait> = Arc::new(());
                Ok(ParseStateEdgeContent { state_id, user_data })
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
    pub fn new() -> Self {
        ParseState { stack: Arc::new(GSSNode::new(LLMTokenInfo::default())) }
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
    pub actions: BTreeMap<NonTerminalID, ActionFn>, // Changed field type
}

impl JSONConvertible for GLRParser {
    fn to_json(&self) -> JSONNode {
        if !self.actions.is_empty() {
            panic!("Cannot serialize GLRParser with actions defined. Actions are not JSON serializable.");
        }
        let mut obj = StdMap::new();
        obj.insert("stage_7_table".to_string(), self.stage_7_table.to_json());
        obj.insert("productions".to_string(), self.productions.to_json());
        obj.insert("terminal_map".to_string(), self.terminal_map.to_json());
        obj.insert("non_terminal_map".to_string(), self.non_terminal_map.to_json());
        obj.insert("item_set_map".to_string(), self.item_set_map.to_json());
        obj.insert("start_state_id".to_string(), self.start_state_id.to_json());
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
                Ok(GLRParser {
                    stage_7_table,
                    productions,
                    terminal_map,
                    non_terminal_map,
                    item_set_map,
                    start_state_id,
                    actions: BTreeMap::new(), // Initialize actions as empty
                })
            }
            _ => Err("Expected JSONNode::Object for GLRParser".to_string()),
        }
    }
}

impl Debug for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let actions_str = if self.actions.is_empty() {
            "None".to_string()
        } else {
            // ActionFn is not Debug, so we can't print it directly.
            // We can list the NonTerminalIDs that have actions.
            let nt_ids_with_actions: Vec<usize> = self.actions.keys().map(|id| id.0).collect();
            format!("[Actions defined for NT IDs: {:?}]", nt_ids_with_actions)
        };
        f.debug_struct("GLRParser")
            .field("stage_7_table", &self.stage_7_table)
            .field("productions", &self.productions)
            .field("terminal_map", &self.terminal_map)
            .field("non_terminal_map", &self.non_terminal_map)
            .field("item_set_map", &self.item_set_map)
            .field("start_state_id", &self.start_state_id)
            .field("actions", &actions_str)
            .finish()
    }
}

impl PartialEq for GLRParser {
    fn eq(&self, other: &Self) -> bool {
        let actions_are_equal = self.actions.len() == other.actions.len() &&
                               self.actions.iter().all(|(key, val_self)| {
                                   other.actions.get(key).map_or(false, |val_other| Arc::ptr_eq(val_self, val_other))
                               });
        self.stage_7_table == other.stage_7_table &&
        self.productions == other.productions &&
        self.terminal_map == other.terminal_map &&
        self.non_terminal_map == other.non_terminal_map &&
        self.item_set_map == other.item_set_map &&
        self.start_state_id == other.start_state_id &&
        actions_are_equal
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
            actions: converted_actions, // Assign converted actions
        }
    }

    pub fn init_glr_parser(&self) -> GLRParserState { // No longer generic
        self.init_glr_parser_with_acc(LLMTokenInfo::default())
    }

    pub fn init_glr_parser_null(&self) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: ParseState::new(),
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }

    pub fn init_glr_parser_with_acc(&self, initial_acc: LLMTokenInfo) -> GLRParserState { // No longer generic
        let initial_parse_state = self.init_parse_state_with_acc(initial_acc);
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }
    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: parse_state,
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }

    pub fn init_parse_state(&self) -> ParseState { // No longer generic
        self.init_parse_state_with_acc(LLMTokenInfo::default())
    }

    pub fn init_parse_state_with_acc(&self, initial_acc: LLMTokenInfo) -> ParseState { // No longer generic
        let initial_user_data: Arc<dyn UserDataTrait> = Arc::new(());
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
            user_data: initial_user_data, // Add default user_data
        };
        let root = Arc::new(GSSNode::new(initial_acc.clone())); // initial_acc for the root
        // Push creates a new node. Its acc should be derived from the parent (root in this case).
        let stack = Arc::new(root.push(initial_content, initial_acc)); 
        ParseState { stack }
    }

    pub fn parse(&self, input: &[TerminalID]) -> GLRParserState { // No longer generic
        let mut state = self.init_glr_parser();
        state.parse(input);
        state
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
                        Symbol::Terminal(terminal) => write!(f, " {}", terminal.0),
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
                            Symbol::Terminal(terminal) => write!(f, " {}", terminal.0),
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
            for (&terminal_id, action) in &row.shifts_and_reduces {
                let terminal = terminal_map.get_by_right(&terminal_id).unwrap();
                match action {
                    Stage7ShiftsAndReduces::Shift(next_state_id) => {
                        writeln!(f, "      - {} -> Shift {}", terminal.0, next_state_id.0)?;
                    }
                    Stage7ShiftsAndReduces::Reduce { production_id: _ , nonterminal_id: nonterminal, len } => { // production_id ignored
                        let nt_name = non_terminal_map.get_by_right(nonterminal).unwrap();
                        writeln!(f, "      - {} -> Reduce {} (len {})", terminal.0, nt_name.0, len)?;
                    }
                    Stage7ShiftsAndReduces::Split { shift, reduces } => {
                        writeln!(f, "      - {} -> Conflict:", terminal.0)?;
                        if let Some(shift_state) = shift {
                            writeln!(f, "        - Shift {}", shift_state.0)?;
                        }
                        for (len, nts) in reduces {
                            writeln!(f, "        - Reduce (len {}):", len)?;
                            for (nt_id, prod_ids) in nts {
                                let nt = non_terminal_map.get_by_right(nt_id).unwrap();
                                for prod_id_val in prod_ids {
                                    let prod = self.productions.get(prod_id_val.0).unwrap();
                                    writeln!(f, "          - {} -> {}", nt.0, prod.lhs.0)?;
                                }
                            }

                        }
                    }
                }
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
            writeln!(f, "  {} -> {}", terminal.0, terminal_id.0)?;
        }

        writeln!(f, "\nNon-Terminal Map:")?;
        for (non_terminal, non_terminal_id) in non_terminal_map {
            writeln!(f, "  {} -> {}", non_terminal.0, non_terminal_id.0)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
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
        acc_for_new_node: LLMTokenInfo,
    ) -> ParseState {
        let new_gss_node_instance = stack.push(new_content, acc_for_new_node);
        ParseState { stack: Arc::new(new_gss_node_instance) }
    }

    fn pop_and_goto(
        &self,
        stack: &Arc<GSSNode>, 
        edge_content: &ParseStateEdgeContent,
        edge_src: &Arc<GSSNode>, 
        len: usize,
        nt: NonTerminalID,
    ) -> Arc<GSSNode> { 
        let cur_acc_from_reducible_node = stack.acc().clone(); // Clone before potential modification

        let parent_gss_node = if len == 0 { // Renamed parent to parent_gss_node
            Arc::new(edge_src.push(edge_content.clone(), edge_src.acc().clone())) // Provide acc for push
        } else {
            Arc::new(edge_src.popn(len - 1))
        };
        let mut out = GSSNode::new(Some(LLMTokenBV::new())); // Start with a default acc
        crate::debug!(4, "Popped with {} predecessors...", parent_gss_node.num_predecessors());

        for (predecessor_arc, edge_value) in parent_gss_node.pop_iter() { // Renamed predecessor to predecessor_arc
            let goto = self.parser.stage_7_table.get(&edge_value.state_id).map_or_else(|| Err(format!("State {} not found in stage_7_table", edge_value.state_id.0)), |row| row.gotos.get(&nt).map_or_else(|| Err(format!("Non-terminal {} not found in gotos for {:?} (processing predecessor {:p})", nt.0, edge_value.state_id, Arc::as_ptr(&predecessor_arc))), |state_id| Ok(*state_id))).unwrap();
            match goto {
                Goto::State(goto_state_id) => {
                    crate::debug!(4, " ...and edge value {:?}, predecessor {:p}, goto state ID {}", edge_value.state_id, Arc::as_ptr(&predecessor_arc), goto_state_id.0);

                    let new_acc_for_goto_child = parent_gss_node.acc().clone().intersect(cur_acc_from_reducible_node.clone());
                    let goto_node_content = ParseStateEdgeContent { state_id: goto_state_id, user_data: edge_content.user_data.clone() };

                    let isolated_parent_arc = Arc::new(predecessor_arc.push(edge_value, new_acc_for_goto_child.clone()));
                    let new_gss_node = isolated_parent_arc.push(goto_node_content, new_acc_for_goto_child);
                    out.merge(&Arc::new(new_gss_node));
                }
                Goto::Accept => {
                    // No action needed for Accept
                }
            }
        }
        Arc::new(out)
    }

    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
        crate::debug!(3, "{} - token {} ({:?}) - nodes", phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0));
        return;
        const MAX: usize = 30;
        const PANIC_THRESHOLD: usize = 10000;

        let roots: Vec<_> = vec![self.active_state.stack.clone()];
        let stats = gather_gss_stats(&roots.iter().map(|r| r.as_ref()).collect::<Vec<_>>());
        crate::debug!(3, "{} - token {} ({:?}) - nodes: {:?}",
                      phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0), stats);

        let make_msg = |print_full_forest, max_nodes_to_print| {
            if print_full_forest {
                format!("GSS ({} nodes):\n{}", stats.unique_nodes,
                        print_gss_forest(&roots, max_nodes_to_print))
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

        debug!(4, "{}", make_msg(stats.unique_nodes <= MAX, MAX));
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

    pub fn step(&mut self, token_id: TerminalID) {
        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        self.log_gss("Step-start", token_id);
        self.cycled_states = ParseState::new();

        let mut todo: Vec<(ParseState, BTreeSet<Arc<GSSNode>>)> = Vec::new();
        todo.push((ParseState { stack: self.active_state.stack.clone() }, BTreeSet::new()));

        let mut next = ParseState::new();
        let mut not_found = ParseState::new();

        while let Some((state, visited_on_this_path)) = todo.pop() { 
            if visited_on_this_path.contains(&state.stack) {
                crate::debug!(2, "Cycle detected: GSSNode at {:p} encountered again in reduction path while processing token {:?}.", Arc::as_ptr(&state.stack), token_id);
                print_gss_forest(&[state.stack.clone()], usize::MAX);
                panic!("Cycle detected: GSSNode at {:p} encountered again in reduction path.", Arc::as_ptr(&state.stack));
            }

            let mut next_visited_on_this_path = visited_on_this_path; 
            // next_visited_on_this_path.insert(state.stack.clone());

            let stack_arc_for_operations = &state.stack; 
            for (parent_arc, mut top_edge_content) in state.stack.pop_iter() { // Renamed top to top_edge_content
                let current_path_acc = state.stack.acc().clone().intersect(parent_arc.acc().clone());
                let temp_idk = Arc::new(parent_arc.push(top_edge_content.clone(), current_path_acc.clone())); // Acc for push

                let row = &self.parser.stage_7_table[&top_edge_content.state_id];

                match row.shifts_and_reduces.get(&token_id) {
                    Some(Stage7ShiftsAndReduces::Shift(to)) => {
                        crate::debug!(4, "Shift from state {} via token {} to state {}", top_edge_content.state_id.0, token_id.0, to.0);
                        let new_content = ParseStateEdgeContent { state_id: *to, user_data: top_edge_content.user_data.clone() };
                        let new_parse_state = self.push_state(&temp_idk, new_content, stack_arc_for_operations.acc().clone());
                        next.merge(new_parse_state);
                    }

                    Some(Stage7ShiftsAndReduces::Reduce {
                             nonterminal_id: nt,
                             len, ..
                         }) => {
                        crate::debug!(4, "Reduce from state {} via token {} to nonterminal {} of length {}", top_edge_content.state_id.0, token_id.0, nt.0, len);
                        if let Some(action_fn) = self.parser.actions.get(&nt) {
                            let is_ok = action_fn(&mut top_edge_content.user_data);
                            if !is_ok {
                                crate::debug!(4, "Action for non-terminal {} returned false, pruning this reduction path.", nt.0);
                                continue;
                            }
                        }
                        let s_new_arc = self.pop_and_goto(&temp_idk, &top_edge_content, &parent_arc, *len, *nt);
                        if !s_new_arc.is_empty() { // Only add to todo if the reduction leads to valid states
                           todo.push((ParseState { stack: s_new_arc }, next_visited_on_this_path.clone()));
                        }
                    }

                    Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                        crate::debug!(4, "Split from state {} via token {}", top_edge_content.state_id.0, token_id.0);
                        if let Some(to) = shift {
                            crate::debug!(4, " Shift from state {} via token {} to state {}", top_edge_content.state_id.0, token_id.0, to.0);
                            let new_content = ParseStateEdgeContent { state_id: *to, user_data: top_edge_content.user_data.clone() };
                            let new_parse_state = self.push_state(&temp_idk, new_content, stack_arc_for_operations.acc().clone());
                            next.merge(new_parse_state);
                        }
                        for (len, nts) in reduces {
                            crate::debug!(4, " Reduce from state {} via token {} to nonterminals {:?}", top_edge_content.state_id.0, token_id.0, nts);
                            for (nt, _prod_ids) in nts {
                                crate::debug!(4, "  Reducing via nonterminal {} of length {}", nt.0, len);
                                if let Some(action_fn) = self.parser.actions.get(&nt) {
                                    let is_ok = action_fn(&mut top_edge_content.user_data);
                                    if !is_ok {
                                        crate::debug!(4, "Action for non-terminal {} returned false, pruning this reduction path.", nt.0);
                                        continue;
                                    }
                                }
                                let s_new_arc = self.pop_and_goto(&temp_idk, &top_edge_content, &parent_arc, *len, *nt);
                                if !s_new_arc.is_empty() {
                                    todo.push((ParseState { stack: s_new_arc }, next_visited_on_this_path.clone()));
                                }
                            }
                        }
                    }

                    None => {
                        crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, top_edge_content.state_id.0);
                        not_found.merge(state.clone());
                    },
                }
            }
        }

        self.active_state = next;
        self.action_not_found_states = not_found; // Retain for potential inspection, though current design drops them.
        
        // Simplify the active GSS forest at the end of the step
        if !self.active_state.stack.is_empty() {
            Arc::make_mut(&mut self.active_state.stack).simplify();
        }

        self.log_gss("Step-end", token_id);
        // self.action_not_found_states = ParseState::new(); // Reset if not needed beyond the step

        crate::debug!(4, "----------------------------------------------------------------");
    }

    pub fn merge_active_states(&mut self) {
        // No longer strictly necessary due to BTreeMap merge-on-insert, but GSS merge is explicit.
        // This method could be used if multiple GLRParserStates are combined.
    }

    pub fn merge_with(&mut self, other: GLRParserState) { // No longer generic
        assert!(std::ptr::eq(self.parser, other.parser));
        self.active_state.merge(other.active_state);
        self.action_not_found_states.merge(other.action_not_found_states);
        self.cycled_states.merge(other.cycled_states);
    }

    pub fn is_ok(&self) -> bool {
        !self.active_state.stack.is_empty()
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
