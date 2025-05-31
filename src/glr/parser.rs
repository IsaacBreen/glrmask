use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::any::Any;
use std::collections::BTreeMap as StdMap;

use crate::datastructures::gss::print_gss_forest;
use crate::datastructures::gss::{gather_gss_stats, find_longest_path, PathAccumulator, GSSNode, GSSTrait, GSSStats};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID};
use crate::constraint::{LLMTokenBV, LLMTokenInfo};

use bimap::BiBTreeMap;
use crate::debug;
use crate::json_serialization::{JSONConvertible, JSONNode};

// UserData type alias
pub type UserData = Arc<dyn Any + Send + Sync>;

// Action trait
pub trait Action: Send + Sync + Debug {
    /// Executes the action.
    /// `lhs_user_data`: Mutable reference to the UserData for the LHS non-terminal being created.
    /// `rhs_user_data`: Slice of UserData from the RHS symbols.
    /// Returns `true` if the action is valid and parsing should continue for this path, `false` otherwise.
    fn execute(&self, lhs_user_data: &mut UserData, rhs_user_data: &[UserData]) -> bool;

    /// Provides a unique name for the action, used for error messages if serialization is attempted.
    fn name(&self) -> String;
}

// ActionContainer to hold an Arc<dyn Action>
#[derive(Clone)]
pub struct ActionContainer(pub Arc<dyn Action>);

impl Debug for ActionContainer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ActionContainer")
         .field(&self.0.name())
         .finish()
    }
}

// PartialEq for ActionContainer based on pointer equality of the Arc
// This is a simplification; true equality of dyn Action is complex.
// For table generation, distinct Arc<dyn Action> instances are treated as distinct.
impl PartialEq for ActionContainer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for ActionContainer {}

// PartialOrd and Ord for ActionContainer (needed for BTreeMap keys if actions were part of keys)
// For now, actions are not directly in BTreeMap keys that require Ord in a meaningful way beyond pointer comparison.
// We'll use pointer comparison for ordering.
impl PartialOrd for ActionContainer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ActionContainer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let ptr_self = Arc::as_ptr(&self.0) as usize;
        let ptr_other = Arc::as_ptr(&other.0) as usize;
        ptr_self.cmp(&ptr_other)
    }
}

// Hash for ActionContainer (needed if actions were part of Hash keys)
// Using pointer hash.
impl Hash for ActionContainer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
    }
}


#[derive(Debug, Clone)]
pub struct ParseStateEdgeContent {
    pub state_id: StateID,
    pub user_data: UserData,
}

// Custom implementations for Hash, PartialEq, Eq, PartialOrd, Ord
impl Hash for ParseStateEdgeContent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state_id.hash(state);
    }
}

impl PartialEq for ParseStateEdgeContent {
    fn eq(&self, other: &Self) -> bool {
        self.state_id == other.state_id
        // user_data is not compared for equality in this context
    }
}

impl Eq for ParseStateEdgeContent {}

impl PartialOrd for ParseStateEdgeContent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.state_id.partial_cmp(&other.state_id)
    }
}

impl Ord for ParseStateEdgeContent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.state_id.cmp(&other.state_id)
    }
}

// JSONConvertible for ParseStateEdgeContent
impl JSONConvertible for ParseStateEdgeContent {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        // Error if user_data is not the default Arc::new(())
        let is_default_type = self.user_data.is::<()>();
        if !is_default_type {
            panic!("Serialization of custom UserData in ParseStateEdgeContent is not supported.");
        }
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for ParseStateEdgeContent".to_string())
                                  .and_then(StateID::from_json)?;
                // Always deserialize user_data to default
                let user_data: UserData = Arc::new(());
                Ok(ParseStateEdgeContent { state_id, user_data })
            }
            _ => Err("Expected JSONNode::Object for ParseStateEdgeContent".to_string()),
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState {
    pub stack: Arc<GSSNode>,
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


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLRParser {
    pub stage_7_table: Stage7Table,
    pub productions: Vec<Production>,
    pub terminal_map: BiBTreeMap<Terminal, TerminalID>,
    pub non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    pub item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
    pub start_state_id: StateID,
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
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let definition_node = obj.remove("stage_7_table")
                    .ok_or_else(|| "Missing field stage_7_table".to_string())?;
                let stage_7_table = Stage7Table::from_json(definition_node)?;

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
                })
            }
            _ => Err("Expected JSONNode::Object for GLRParser".to_string()),
        }
    }
}


impl GLRParser {
    pub fn new(
        stage_7_table: Stage7Table,
        productions: Vec<Production>,
        terminal_map: BiBTreeMap<Terminal, TerminalID>,
        non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
        item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
        start_state_id: StateID,
    ) -> Self {
        Self {
            stage_7_table,
            productions,
            terminal_map,
            non_terminal_map,
            item_set_map,
            start_state_id,
        }
    }

    pub fn init_glr_parser(&self) -> GLRParserState {
        self.init_glr_parser_with_acc(LLMTokenInfo::default())
    }

    pub fn init_glr_parser_null(&self) -> GLRParserState {
        GLRParserState {
            parser: self,
            active_state: ParseState::new(),
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }

    pub fn init_glr_parser_with_acc(&self, initial_acc: LLMTokenInfo) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_with_acc(initial_acc);
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }
    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState {
        GLRParserState {
            parser: self,
            active_state: parse_state,
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }

    pub fn init_parse_state(&self) -> ParseState {
        self.init_parse_state_with_acc(LLMTokenInfo::default())
    }

    pub fn init_parse_state_with_acc(&self, initial_acc: LLMTokenInfo) -> ParseState {
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
            user_data: Arc::new(()),
        };
        let root = Arc::new(GSSNode::new(initial_acc.clone()));
        let stack = Arc::new(root.push(initial_content, initial_acc));
        ParseState { stack }
    }

    pub fn parse(&self, input: &[TerminalID]) -> GLRParserState {
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
                    Stage7ShiftsAndReduces::Reduce { production_id: _ , nonterminal_id: nonterminal, len, action: prod_action } => {
                        let nt_name = non_terminal_map.get_by_right(nonterminal).unwrap();
                        let action_str = prod_action.as_ref().map_or("".to_string(), |ac| format!(" (Action: {})", ac.0.name()));
                        writeln!(f, "      - {} -> Reduce {}{} (len {})", terminal.0, nt_name.0, action_str, len)?;
                    }
                    Stage7ShiftsAndReduces::Split { shift, reduces } => {
                        writeln!(f, "      - {} -> Conflict:", terminal.0)?;
                        if let Some(shift_state) = shift {
                            writeln!(f, "        - Shift {}", shift_state.0)?;
                        }
                        for (len, nts) in reduces {
                            writeln!(f, "        - Reduce (len {}):", len)?;
                            for (nt_id, (prods_no_action, prods_with_action)) in nts {
                                let nt = non_terminal_map.get_by_right(nt_id).unwrap();
                                for prod_id_val in prods_no_action {
                                    let prod = self.productions.get(prod_id_val.0).unwrap();
                                    writeln!(f, "          - {}: {} -> {}", nt.0, prod.lhs.0, prod.rhs.iter().map(|s| format!("{:?}", s)).collect::<Vec<_>>().join(" "))?;
                                }
                                for (prod_id_val, action_container) in prods_with_action {
                                    let prod = self.productions.get(prod_id_val.0).unwrap();
                                    writeln!(f, "          - {} (Action: {}): {} -> {}", nt.0, action_container.0.name(), prod.lhs.0, prod.rhs.iter().map(|s| format!("{:?}", s)).collect::<Vec<_>>().join(" "))?;
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
                    Goto::State(state_id_val) => format!("{}", state_id_val.0),
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
pub struct GLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub active_state: ParseState,
    pub action_not_found_states: ParseState,
    pub cycled_states: ParseState,
}

impl<'a> GLRParserState<'a> {
    fn push_state(
        &self,
        stack: &Arc<GSSNode>,
        next_state_id: StateID,
        acc_for_new_node: LLMTokenInfo,
        user_data_for_edge: UserData,
    ) -> ParseState {
        let new_content = ParseStateEdgeContent {
            state_id: next_state_id,
            user_data: user_data_for_edge,
        };
        let new_gss_node_instance = stack.push(new_content, acc_for_new_node);
        ParseState { stack: Arc::new(new_gss_node_instance) }
    }

    // Helper to collect RHS UserData from a linear path in GSS
    fn collect_rhs_user_data(mut gss_node_after_rhs: Arc<GSSNode>, len: usize) -> Vec<UserData> {
        if len == 0 {
            return Vec::new();
        }
        let mut rhs_data = Vec::with_capacity(len);
        let mut current_node = gss_node_after_rhs.clone();

        for _i in 0..len {
            // A linear path assumed for `temp_idk` in step().
            // This means current_node should have exactly one predecessor.
            let (edge_content, pred_node) = if let Some((ec, pn)) = current_node.predecessors().iter().next() {
                (ec.clone(), pn.clone())
            } else {
                // This scenario indicates a broken assumption or an empty GSS node where a path is expected.
                // For robustness, could return an error or fill with defaults.
                // Given the context (it's called from temp_idk's reduction path), it implies `len` valid predecessors existed.
                // If it happens, it's a critical logic error or an unexpected GSS state.
                crate::debug!(0, "Error: Expected a predecessor for RHS UserData collection, but found none. Node: {:p}, len: {}, collected: {}", Arc::as_ptr(&current_node), len, rhs_data.len());
                return vec![Arc::new(()) as UserData; len - rhs_data.len()]; // Fill remaining with defaults
            };
            rhs_data.push(edge_content.user_data.clone());
            current_node = pred_node;
        }
        rhs_data.reverse(); // Collected in reverse order (from last RHS symbol to first)
        rhs_data
    }

    fn pop_and_goto(
        &self,
        stack_after_rhs: &Arc<GSSNode>,
        len: usize,
        nt_id: NonTerminalID,
        action: Option<&ActionContainer>,
    ) -> Vec<(Arc<GSSNode>, UserData)> {
        let mut resulting_goto_paths = Vec::new();

        // 1. Collect UserData for RHS symbols
        let rhs_user_data_vec = Self::collect_rhs_user_data(stack_after_rhs.clone(), len);

        // 2. Prepare LHS UserData and execute action
        let mut lhs_user_data: UserData = Arc::new(());
        let action_is_valid = if let Some(act_container) = action {
            act_container.0.execute(&mut lhs_user_data, &rhs_user_data_vec)
        } else {
            true
        };

        if !action_is_valid {
            crate::debug!(4, "Action for NT {} (len {}) deemed reduction invalid.", nt_id.0, len);
            return resulting_goto_paths;
        }

        // 3. Determine the node from which GOTO originates
        let node_before_rhs_start = Arc::new(stack_after_rhs.popn(len));

        // Handle the special case where len is 0 (epsilon production)
        // In this case, node_before_rhs_start is the same as stack_after_rhs.
        // The GOTO should originate from the state *at the top* of the stack.
        // The pop_iter() of node_before_rhs_start would typically yield nothing or the GSS root's original edge.
        // Let's refine the source of GOTO based on len.
        let goto_source_state_edges_and_nodes: Vec<(ParseStateEdgeContent, Arc<GSSNode>)> = if len == 0 {
            // For epsilon productions, the GOTO happens from the state that *just processed* the epsilon.
            // This is equivalent to the state *before* the epsilon production.
            // So, for (GSSNode A --edge--> B), if epsilon is applied at B, GOTO comes from A.
            // The GSS structure stores `(edge_val, pred_arc)`.
            // So, `stack_after_rhs` is `B`. We need A and its edge.
            stack_after_rhs.pop_iter().map(|(pred_arc, edge_val)| (edge_val.clone(), pred_arc.clone())).collect()
        } else {
            // For non-epsilon reductions, GOTO originates from the node `len` steps back.
            // We need to iterate the predecessors of `node_before_rhs_start` to find the states from which it was reached.
            node_before_rhs_start.pop_iter().map(|(pred_arc, edge_val)| (edge_val.clone(), pred_arc.clone())).collect()
        };

        if goto_source_state_edges_and_nodes.is_empty() {
            // This means we are trying to GOTO from the very root of the GSS (initial conceptual state).
            // This applies to the start symbol reduction, e.g., S' -> S.
            // The GOTO is from the parser's start_state_id.
            let goto_source_state_id = self.parser.start_state_id;
            let goto_action = self.parser.stage_7_table.get(&goto_source_state_id)
                .and_then(|row| row.gotos.get(&nt_id));

            if let Some(Goto::State(goto_target_state_id)) = goto_action {
                let goto_edge_content = ParseStateEdgeContent {
                    state_id: *goto_target_state_id,
                    user_data: lhs_user_data.clone(),
                };
                let acc_for_goto_node = stack_after_rhs.acc().clone();
                let new_gss_path_top = Arc::new(node_before_rhs_start.push(goto_edge_content, acc_for_goto_node));
                resulting_goto_paths.push((new_gss_path_top, lhs_user_data));
            } else if let Some(Goto::Accept) = goto_action {
                resulting_goto_paths.push((stack_after_rhs.clone(), lhs_user_data));
            }
            return resulting_goto_paths;
        }

        // Iterate over distinct paths leading to `node_before_rhs_start`
        for (edge_into_source_node, source_node_for_goto) in goto_source_state_edges_and_nodes {
            let goto_source_state_id = edge_into_source_node.state_id;
            let goto_action = self.parser.stage_7_table.get(&goto_source_state_id)
                .and_then(|row| row.gotos.get(&nt_id));

            match goto_action {
                Some(Goto::State(goto_target_state_id)) => {
                    crate::debug!(4, " GOTO from state {} via NT {} to state {}", goto_source_state_id.0, nt_id.0, goto_target_state_id.0);

                    // The acc for the new node after GOTO.
                    // It should be the accumulator of the reduced path segment (`stack_after_rhs.acc()`)
                    // intersected with the accumulator of the GSS node from which GOTO originates (`source_node_for_goto.acc()`).
                    let path_acc = stack_after_rhs.acc().clone();
                    let acc_for_goto_node = path_acc.intersect(source_node_for_goto.acc().clone());

                    let goto_edge_content = ParseStateEdgeContent {
                        state_id: *goto_target_state_id,
                        user_data: lhs_user_data.clone(),
                    };

                    // Construct the new GSS path segment for this GOTO
                    // It's `source_node_for_goto` with a new edge and new state.
                    let new_gss_path_top = Arc::new(source_node_for_goto.push(goto_edge_content, acc_for_goto_node));
                    resulting_goto_paths.push((new_gss_path_top, lhs_user_data.clone()));
                }
                Some(Goto::Accept) => {
                    // Path is accepted. The GSS node representing this is `stack_after_rhs` (the node *after* the RHS).
                    resulting_goto_paths.push((stack_after_rhs.clone(), lhs_user_data.clone()));
                    crate::debug!(4, " GOTO from state {} via NT {} to ACCEPT", goto_source_state_id.0, nt_id.0);
                }
                None => {
                    crate::debug!(4, " GOTO not found for state {} via NT {}", goto_source_state_id.0, nt_id.0);
                }
            }
        }
        resulting_goto_paths
    }

    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
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
                                       p.iter().map(|(ec, _n)| ec.state_id.0)
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

        // todo stores GSS stack tops to be processed for reductions.
        let mut reduction_q: VecDeque<(Arc<GSSNode>, BTreeSet<Arc<GSSNode>>)> = VecDeque::new();
        reduction_q.push_back((self.active_state.stack.clone(), BTreeSet::new()));

        // next_shift_states accumulates GSS stack tops resulting from shifts.
        let mut next_shift_states = ParseState::new();
        // action_not_found_for_token accumulates states where no action (shift/reduce) was found for the current token.
        let mut action_not_found_for_token = ParseState::new();


        while let Some((current_stack_top, visited_on_this_reduction_path)) = reduction_q.pop_front() {
            if visited_on_this_reduction_path.contains(&current_stack_top) {
                crate::debug!(2, "Cycle detected during reduction: GSSNode at {:p} encountered again while processing token {:?}.", Arc::as_ptr(&current_stack_top), token_id);
                continue;
            }

            let mut next_visited_for_path = visited_on_this_reduction_path.clone();
            next_visited_for_path.insert(current_stack_top.clone());

            for (predecessor_node, edge_to_predecessor) in current_stack_top.pop_iter() {
                let current_path_acc = current_stack_top.acc().clone().intersect(predecessor_node.acc().clone());
                let temp_idk = Arc::new(predecessor_node.push(edge_to_predecessor.clone(), current_path_acc.clone()));

                let current_state_id = edge_to_predecessor.state_id;
                let row = match self.parser.stage_7_table.get(&current_state_id) {
                    Some(r) => r,
                    None => {
                        action_not_found_for_token.merge(ParseState { stack: temp_idk.clone() });
                        continue;
                    }
                };

                match row.shifts_and_reduces.get(&token_id) {
                    Some(Stage7ShiftsAndReduces::Shift(to_state_id)) => {
                        crate::debug!(4, "Shift from state {} via token {} to state {}", current_state_id.0, token_id.0, to_state_id.0);
                        let new_parse_state = self.push_state(&temp_idk, *to_state_id, temp_idk.acc().clone(), Arc::new(()));
                        next_shift_states.merge(new_parse_state);
                    }
                    Some(Stage7ShiftsAndReduces::Reduce { nonterminal_id, len, action, .. }) => {
                        crate::debug!(4, "Reduce from state {} via token {} to NT {} (len {}), action: {:?}", current_state_id.0, token_id.0, nonterminal_id.0, len, action.is_some());
                        let goto_results = self.pop_and_goto(&temp_idk, *len, *nonterminal_id, action.as_ref());
                        for (goto_stack_top, _lhs_user_data) in goto_results {
                            if !goto_stack_top.is_empty() {
                                reduction_q.push_back((goto_stack_top, next_visited_for_path.clone()));
                            }
                        }
                    }
                    Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                        crate::debug!(4, "Split from state {} via token {}", current_state_id.0, token_id.0);
                        if let Some(to_state_id) = shift {
                            crate::debug!(4, " Shift part of split: to state {}", to_state_id.0);
                            let new_parse_state = self.push_state(&temp_idk, *to_state_id, temp_idk.acc().clone(), Arc::new(()));
                            next_shift_states.merge(new_parse_state);
                        }
                        for (len, nt_map) in reduces {
                            for (nt_id, (prods_no_action, prods_with_action)) in nt_map {
                                crate::debug!(4, " Reduce part of split: NT {}, len {}", nt_id.0, len);
                                for _prod_id in prods_no_action {
                                    let goto_results = self.pop_and_goto(&temp_idk, *len, *nt_id, None);
                                    for (goto_stack_top, _) in goto_results {
                                        if !goto_stack_top.is_empty() {
                                            reduction_q.push_back((goto_stack_top, next_visited_for_path.clone()));
                                        }
                                    }
                                }
                                for (_prod_id, action) in prods_with_action {
                                    let goto_results = self.pop_and_goto(&temp_idk, *len, *nt_id, Some(action));
                                    for (goto_stack_top, _) in goto_results {
                                        if !goto_stack_top.is_empty() {
                                            reduction_q.push_back((goto_stack_top, next_visited_for_path.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, current_state_id.0);
                        action_not_found_for_token.merge(ParseState { stack: temp_idk.clone() });
                    }
                }
            }
        }

        self.active_state = next_shift_states;
        self.action_not_found_states = action_not_found_for_token;

        if !self.active_state.stack.is_empty() {
            Arc::make_mut(&mut self.active_state.stack).simplify();
        }

        self.log_gss("Step-end", token_id);
        crate::debug!(4, "----------------------------------------------------------------");
    }

    pub fn merge_active_states(&mut self) {
    }

    pub fn merge_with(&mut self, other: GLRParserState) {
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

impl ParseState {
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
