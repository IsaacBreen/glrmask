use crate::datastructures::gss::print_gss_forest;
use crate::datastructures::gss::{gather_gss_stats, find_longest_path, PathAccumulator, prune_and_transform_recursive}; // Import PathAccumulator and prune_and_transform_recursive
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID};
use crate::datastructures::gss::{GSSNode, GSSTrait, GSSStats};

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;
use crate::debug;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;


// Remove MergeAndIntersect trait definition

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateEdgeContent { // No longer generic
    pub state_id: StateID,
    // Removed pub t: T,
}
// JSONConvertible for ParseStateNodeContent (now concrete type)
impl JSONConvertible for ParseStateEdgeContent {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for ParseStateNodeContent".to_string())
                                  .and_then(StateID::from_json)?;
                Ok(ParseStateEdgeContent { state_id })
            }
            _ => Err("Expected JSONNode::Object for ParseStateNodeContent".to_string()),
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState<A: PathAccumulator> { // Generic over Accumulator A
    pub stack: Arc<GSSNode<ParseStateEdgeContent, A>>,
}
// No JSONConvertible for ParseState<A> directly (depends on GSSNode).

impl<A: PathAccumulator> ParseState<A> {
    pub fn new() -> Self {
        ParseState { stack: Arc::new(GSSNode::new(A::default())) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StopReason {
    ActionNotFound,
    GotoNotFound,
}
// Manual impl for StopReason (enum) - unchanged.
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



// TODO: should this *really* derive `Clone`? Users probably shouldn't clone this, should they?
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

    pub fn init_glr_parser<A: PathAccumulator>(&self) -> GLRParserState<A> {
        self.init_glr_parser_with_acc(A::default())
    }

    pub fn init_glr_parser_with_acc<A: PathAccumulator>(&self, initial_acc: A) -> GLRParserState<A> {
        let initial_parse_state = self.init_parse_state_with_acc(initial_acc);
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }
    pub fn init_glr_parser_from_parse_state<A: PathAccumulator>(&self, parse_state: ParseState<A>) -> GLRParserState<A> {
        GLRParserState {
            parser: self,
            active_state: parse_state,
            action_not_found_states: ParseState::new(),
            cycled_states: ParseState::new(),
        }
    }

    pub fn init_parse_state<A: PathAccumulator>(&self) -> ParseState<A> {
        self.init_parse_state_with_acc(A::default())
    }

    pub fn init_parse_state_with_acc<A: PathAccumulator>(&self, initial_acc: A) -> ParseState<A> {
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
        };
        let root = Arc::new(GSSNode::new(initial_acc));
        let stack = Arc::new(root.push(initial_content));
        ParseState { stack }
    }

    pub fn parse<A: PathAccumulator + Default>(&self, input: &[TerminalID]) -> GLRParserState<A> {
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

        // Import necessary items for closure computation
        use crate::glr::items::{compute_closure, Item};
        use std::collections::BTreeSet;

        writeln!(f, "Parse Table:")?;
        writeln!(f, "  Start State: {}", self.start_state_id.0)?;
        for (&state_id, row) in stage_7_table.iter().collect::<BTreeMap<_, _>>() {
            writeln!(f, "  State {}:", state_id.0)?;

            // Get the core items that define this state
            let core_item_set = item_set_map.get_by_right(&state_id).unwrap();
            // Compute the full closure based on the core items
            let full_closure = compute_closure(core_item_set, &self.productions);

            // Print Core Items
            writeln!(f, "    Core Items:")?;
            for item in core_item_set {
                write!(f, "      - {} ->", item.production.lhs.0)?;
                for (i, symbol) in item.production.rhs.iter().enumerate() {
                    if i == item.dot_position {
                        write!(f, " •")?;
                    }
                    match symbol {
                        Symbol::Terminal(terminal) => write!(f, " {:?}", terminal.0),
                        Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0),
                    }?;
                }
                if item.dot_position == item.production.rhs.len() {
                    write!(f, " •")?;
                }
                writeln!(f)?;
            }

            // Print Closure Items (items in full_closure but not in core_item_set)
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

            // --- Rest of the state information ---
            writeln!(f, "    Actions:")?;
            for (&terminal_id, action) in &row.shifts_and_reduces {
                let terminal = terminal_map.get_by_right(&terminal_id).unwrap();
                match action {
                    Stage7ShiftsAndReduces::Shift(next_state_id) => {
                        writeln!(f, "      - {} -> Shift {}", terminal.0, next_state_id.0)?;
                    }
                    Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id: nonterminal, len } => {
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
                    Goto::State(state_id) => format!("{}", state_id.0),
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
pub struct GLRParserState<'a, A: PathAccumulator> { // Generic over Accumulator A
    pub parser: &'a GLRParser,
    pub active_state: ParseState<A>,
    pub action_not_found_states: ParseState<A>,
    pub cycled_states: ParseState<A>,
}

impl<'a, A: PathAccumulator> GLRParserState<'a, A> {
    /* -------------------------------------------------
     * Helper utilities to make `step` compact and clear
     * ------------------------------------------------- */

    /// Push a new state on `stack` and wrap it in a `ParseState`.
    fn push_state(
        &self,
        stack: &Arc<GSSNode<ParseStateEdgeContent, A>>, // This is the parent stack Arc
        next_state_id: StateID,
        // The new node's acc will be inherited from stack.acc by stack.push()
    ) -> ParseState<A> {
        let new_content = ParseStateEdgeContent { state_id: next_state_id };
        // stack.push() is GSSTrait push for Arc<GSSNode<...>>
        // It creates a new GSSNode instance whose `acc` is cloned from `stack.acc`.
        let new_gss_node_instance = stack.push(new_content);
        ParseState { stack: Arc::new(new_gss_node_instance) }
    }

    /// Pop `len` nodes, follow the goto on `nt`, and return the resulting stacks.
    fn pop_and_goto(
        &self,
        stack: &Arc<GSSNode<ParseStateEdgeContent, A>>, // Node being reduced from
        edge_content: &ParseStateEdgeContent,
        edge_src: &Arc<GSSNode<ParseStateEdgeContent, A>>, // Parent node
        len: usize,
        nt: NonTerminalID,
        // cur_t: &T was &LLMTokenInfo, now it's stack.acc
        // So, pass stack.acc as cur_acc_from_reducible_node
    ) -> Arc<GSSNode<ParseStateEdgeContent, A>> { // Returns list of new stack tops
        let cur_acc_from_reducible_node = &stack.acc(); // Get it from the stack being reduced

        let parent = Arc::new(if len == 0 {
            edge_src.push(edge_content.clone())
        } else {
            edge_src.popn(len - 1)
        });
        // println!("Parent: {}", print_gss_forest(&[parent.clone()], usize::MAX));
        let mut out = GSSNode::new_default();
        crate::debug!(4, "Popped with {} predecessors...", parent.predecessors_with_values().len());

        for (predecessor, edge_value) in parent.predecessors_with_values() { // parent_arc is Arc<GSSNode<ParseStateNodeContent, A>>
            // This is ParseStateNodeContent { state_id }
            // let goto = self.parser.stage_7_table[&edge_value.state_id].gotos[&nt];
            // let goto = *self.parser.stage_7_table.get(&edge_value.state_id).expect(format!("State {} not found in stage_7_table", top_of_parent_value.state_id.0).as_str()).gotos.get(&nt).expect(format!("Non-terminal {} not found in gotos", nt.0).as_str());
            let goto = self.parser.stage_7_table.get(&edge_value.state_id).map_or_else(|| Err(format!("State {} not found in stage_7_table", edge_value.state_id.0)), |row| row.gotos.get(&nt).map_or_else(|| Err(format!("Non-terminal {} not found in gotos for {:?} (processing predecessor {:p})", nt.0, edge_value.state_id, Arc::as_ptr(&predecessor))), |state_id| Ok(*state_id))).unwrap();
            match goto {
                Goto::State(goto_state_id) => {
                    crate::debug!(4, " ...and edge value {:?}, predecessor {:p}, goto state ID {}", edge_value.state_id, Arc::as_ptr(&predecessor), goto_state_id.0);

                    // Calculate acc for the new GOTO state's GSS node
                    // It's the parent's acc intersected with the accumulator from the node being reduced.
                    let new_acc_for_goto_child = parent.acc().pop(cur_acc_from_reducible_node); // Use parent_arc.acc()

                    let goto_node_content = ParseStateEdgeContent { state_id: goto_state_id };

                    // TODO: what the heck
                    let new_parent_idk = predecessor.push(edge_value.clone());

                    let mut new_gss_node_arc = new_parent_idk.push(goto_node_content);
                    // Now, explicitly set its acc to the computed intersection
                    *new_gss_node_arc.acc_mut() = new_acc_for_goto_child;

                    out.merge(&new_gss_node_arc);
                }
                Goto::Accept => {
                    // No action needed for Accept
                }
            }
        }
        // println!("{}", print_gss_forest(&[Arc::new(out.clone())], usize::MAX));
        Arc::new(out)
    }

    /// Debug helper so the main `step` body stays short.
    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
        const MAX: usize = 30;
        const PANIC_THRESHOLD: usize = 10000;

        let roots: Vec<_> = vec![self.active_state.stack.clone()];
        let stats = gather_gss_stats(&roots);
        crate::debug!(3, "{} - token {} ({:?}) - nodes: {:?}",
                      phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0), stats);

        let make_msg = |print_full_forest, max_nodes_to_print| {
            if print_full_forest {
                format!("GSS ({} nodes):\n{}", stats.unique_nodes,
                        print_gss_forest(&roots, max_nodes_to_print))
            } else {
                // fall back to longest path printing
                match find_longest_path(&self.active_state.stack) {
                    Some(p) => format!("GSS too big ({} nodes). Longest path ({}): {}",
                                       stats.unique_nodes,
                                       p.len(),
                                       p.iter().map(|(ec, n)| ec.state_id.0)
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
        /* ---------- logging & preparation ---------- */
        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

        Arc::make_mut(&mut self.active_state.stack).simplify();

        self.log_gss("Step-start", token_id);

        // Clear cycled_states at the beginning of each step, as cycle detection is per-step.
        self.cycled_states = ParseState::new();

        // Change the type of `todo` to include a BTreeSet for visited nodes in the current reduction path.
        let mut todo: Vec<(ParseState<A>, BTreeSet<Arc<GSSNode<ParseStateEdgeContent, A>>>)> = Vec::new();

        // Initial population of todo:
        // States from active_states are roots of new reduction chains. Their visited set is initially empty.
        // self.log_gss("Simplified GSS after initial step", token_id);
        todo.push((ParseState { stack: self.active_state.stack.clone() }, BTreeSet::new()));

        let mut next = ParseState::new();
        let mut not_found = ParseState::new();

        /* ---------- core loop ---------- */
        // Modify the while loop condition and variable extraction:
        while let Some((state, visited_on_this_path)) = todo.pop() { // Process states from the worklist
            // `state` is the current ParseState. `state.stack` is the Arc<GSSNode> for its stack top.
            // Check for cycle: if state.stack is already in visited_on_this_path for this reduction chain.
            if visited_on_this_path.contains(&state.stack) {
                crate::debug!(2, "Cycle detected: GSSNode at {:p} encountered again in reduction path while processing token {:?}.", Arc::as_ptr(&state.stack), token_id);
                // The `state` (which includes state.stack, the Arc that forms the cycle point) is moved into cycled_states.
                // self.cycled_states.insert_with(state.key(), state, |existing, new_s| existing.merge(new_s));
                // continue; // Don't process this state further down this cyclic path.
                // Print the tree.
                print_gss_forest(&[state.stack.clone()], usize::MAX);
                // Panic
                panic!("Cycle detected: GSSNode at {:p} encountered again in reduction path.", Arc::as_ptr(&state.stack));
            }

            // Add state.stack to the history for paths stemming from this node.
            // This new set will be passed to children generated by reductions.
            let mut next_visited_on_this_path = visited_on_this_path; // Takes ownership
            next_visited_on_this_path.insert(state.stack.clone());

            // Use state.stack for operations.
            let stack_arc_for_operations = &state.stack; // This is &Arc<GSSNode<...>>
            for (parent_arc, top) in state.stack.predecessors_with_values() {
                let temp_idk = Arc::new(parent_arc.push(top.clone()));
                let row = &self.parser.stage_7_table[&top.state_id];

                match row.shifts_and_reduces.get(&token_id) {
                    /* ------ 1. plain shift ------ */
                    Some(Stage7ShiftsAndReduces::Shift(to)) => {
                        crate::debug!(4, "Shift from state {} via token {} to state {}", top.state_id.0, token_id.0, to.0);
                        // Use stack_arc_for_operations
                        let new_parse_state = self.push_state(&temp_idk, *to);
                        next.merge(new_parse_state);
                    }

                    /* ------ 2. single reduce ------ */
                    Some(Stage7ShiftsAndReduces::Reduce {
                             nonterminal_id: nt,
                             len, ..
                         }) => {
                        crate::debug!(4, "Reduce from state {} via token {} to nonterminal {} of length {}", top.state_id.0, token_id.0, nt.0, len);
                        // Use stack_arc_for_operations
                        let s_new_arc = self.pop_and_goto(&stack_arc_for_operations, top, parent_arc, *len, *nt);
                        // Add to worklist for current step, passing the cloned updated visited set.
                        todo.push((ParseState { stack: s_new_arc }, next_visited_on_this_path.clone()));
                    }

                    /* ------ 3. shift / reduce split ------ */
                    Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                        crate::debug!(4, "Split from state {} via token {}", top.state_id.0, token_id.0);
                        // optional shift part
                        if let Some(to) = shift {
                            crate::debug!(4, " Shift from state {} via token {} to state {}", top.state_id.0, token_id.0, to.0);
                            // Use stack_arc_for_operations
                            let new_parse_state = self.push_state(&temp_idk, *to);
                            next.merge(new_parse_state);
                        }
                        // every reduce alternative
                        for (len, nts) in reduces {
                            crate::debug!(4, " Reduce from state {} via token {} to nonterminals {:?}", top.state_id.0, token_id.0, nts);
                            for (nt, _prod_ids) in nts {
                                // Use stack_arc_for_operations
                                crate::debug!(4, "  Reducing via nonterminal {} of length {}", nt.0, len);
                                let s_new_arc = self.pop_and_goto(&stack_arc_for_operations, top, parent_arc, *len, *nt);
                                // Add to worklist for current step, passing the cloned updated visited set.
                                todo.push((ParseState { stack: s_new_arc }, next_visited_on_this_path.clone()));
                            }
                        }
                    }

                    /* ------ 4. no action ------ */
                    None => {
                        crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, top.state_id.0);
                        // The `state` is moved into not_found.
                        // not_found.insert_with(state.key(), state, |existing, new_s| existing.merge(new_s));
                        not_found.merge(state.clone());
                    },
                }
            }
        }

        /* ---------- finish up ---------- */
        self.active_state = next;
        self.action_not_found_states = not_found;

        self.log_gss("Step-end", token_id);
        self.action_not_found_states = ParseState::new();        // current design: we drop them

        crate::debug!(4, "----------------------------------------------------------------");
    }

    /// Merging is handled implicitly when states are added to `next` in the `step` method via `BTreeMap::insert_with`.
    /// The `ParseState::merge` method performs the actual merge logic on the GSS stacks.
    pub fn merge_active_states(&mut self) {
        // This method is no longer necessary as merging is done on insertion.
        // crate::debug!(3, "merge_active_states called (now a no-op due to BTreeMap usage)");
    }

    pub fn merge_with(&mut self, other: GLRParserState<A>) {
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
    // Removed action_stack
}

impl<A: PathAccumulator> ParseState<A> { // Generic over Accumulator A
    /// Merges `other` into `self`. Assumes `self.key() == other.key()`.
    /// Merges the GSS structures and combines the `acc` value at the top node using `PathAccumulator::union`.
    pub fn merge(&mut self, other: ParseState<A>) {
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
