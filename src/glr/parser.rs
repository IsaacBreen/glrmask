use crate::datastructures::gss::{print_gss_forest, BulkMerge, gather_gss_stats, find_longest_path};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{
    NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID,
};
use crate::datastructures::gss::{GSSNode, GSSTrait, GSSStats};

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;
use crate::debug;

pub trait MergeAndIntersect: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash {
    /// Merges the information represented by `self` and `other`.
    fn merge(&self, other: &Self) -> Self;
    /// Intersects the information represented by `self` and `other`.
    fn intersect(&self, other: &Self) -> Self;
}

impl MergeAndIntersect for () {
    fn merge(&self, _: &Self) -> Self { () }
    fn intersect(&self, _: &Self) -> Self { () }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateNodeContent<T: MergeAndIntersect> {
    pub state_id: StateID,
    pub t: T,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState<T: MergeAndIntersect> {
    pub stack: Arc<GSSNode<ParseStateNodeContent<T>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StopReason {
    ActionNotFound,
    GotoNotFound,
}


// TODO: should this *really* derive `Clone`? Users probably shouldn't clone this, should they?
#[derive(Clone)]
pub struct GLRParser {
    pub stage_7_table: Stage7Table,
    pub productions: Vec<Production>,
    pub terminal_map: BiBTreeMap<Terminal, TerminalID>,
    pub non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    pub item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
    pub start_state_id: StateID,
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

    pub fn init_glr_parser<T: MergeAndIntersect + Default>(&self) -> GLRParserState<T> {
        self.init_glr_parser_with_t(T::default())
    }

    pub fn init_glr_parser_with_t<T: MergeAndIntersect>(&self, t: T) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            active_states: vec![self.init_parse_state_with_t(t)],
            action_not_found_states: Vec::new(),
        }
    }
    pub fn init_glr_parser_from_parse_state<T: MergeAndIntersect>(&self, parse_state: ParseState<T>) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            active_states: vec![parse_state],
            action_not_found_states: Vec::new(),
        }
    }

    pub fn init_glr_parser_from_parse_states<T: MergeAndIntersect>(
        &self,
        parse_states: Vec<ParseState<T>>,
    ) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            active_states: parse_states,
            action_not_found_states: Vec::new(),
        }
    }

    pub fn init_parse_state<T: MergeAndIntersect + Default>(&self) -> ParseState<T> {
        self.init_parse_state_with_t(T::default())
    }

    pub fn init_parse_state_with_t<T: MergeAndIntersect>(&self, t: T) -> ParseState<T> {
        let initial_content = ParseStateNodeContent {
            state_id: self.start_state_id,
            t,
        };
        ParseState {
            stack: Arc::new(GSSNode::new(initial_content)),
        }
    }

    pub fn parse<T: MergeAndIntersect + Default>(&self, input: &[TerminalID]) -> GLRParserState<T> {
        let mut state = self.init_glr_parser();
        state.parse(input);
        state
    }
}

impl Debug for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
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
                            Symbol::Terminal(terminal) => write!(f, " {:?}", terminal.0),
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
                        writeln!(f, "      - {:?} -> Shift {}", terminal.0, next_state_id.0)?;
                    }
                    Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id: nonterminal, len } => {
                        let nt_name = non_terminal_map.get_by_right(nonterminal).unwrap();
                        writeln!(f, "      - {:?} -> Reduce {} (len {})", terminal.0, nt_name.0, len)?;
                    }
                    Stage7ShiftsAndReduces::Split { shift, reduces } => {
                        writeln!(f, "      - {:?} -> Conflict:", terminal.0)?;
                        if let Some(shift_state) = shift {
                            writeln!(f, "        - Shift {}", shift_state.0)?;
                        }
                        for (len, nt_id_to_prod_ids) in reduces {
                            writeln!(f, "        - Reduce (len {}):", len)?;
                            for (nt_id, prod_ids) in nt_id_to_prod_ids {
                                let nt = non_terminal_map.get_by_right(nt_id).unwrap();
                                for prod_id in prod_ids {
                                    let prod = self.productions.get(prod_id.0).unwrap();
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
                writeln!(f, "      - {} -> {}", non_terminal.0, next_state_id.0)?;
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
pub struct GLRParserState<'a, T: MergeAndIntersect> {
    pub parser: &'a GLRParser,
    pub active_states: Vec<ParseState<T>>,
    pub action_not_found_states: Vec<ParseState<T>>,
}

impl<'a, T: MergeAndIntersect + Debug> GLRParserState<'a, T> {
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

    /// Performs the reduction operation for a given stack, length, and non-terminal.
    /// Handles popping, GSS node merging, GOTO lookup, and T value intersection.
    /// Returns a vector of the new stack tops (as Arcs) generated by the reduction paths.
    /// These results still need to be merged later if multiple reductions produce the same state.
    fn perform_reduction(
        &self,
        stack: &Arc<GSSNode<ParseStateNodeContent<T>>>,
        len: usize,
        nonterminal_id: NonTerminalID,
        current_t: &T, // T value from the state *before* popping
    ) -> Vec<Arc<GSSNode<ParseStateNodeContent<T>>>> {
        let nt_name = self.parser.non_terminal_map.get_by_right(&nonterminal_id).unwrap(); // For logging
        crate::debug!(6, "Performing reduction for NT {} (len {}) on stack node {:p}", nt_name.0, len, Arc::as_ptr(stack));

        // 1. Pop 'len' nodes, potentially revealing multiple paths
        let mut popped_stack_nodes = stack.popn(len);
        let initial_pop_count = popped_stack_nodes.len();
        if initial_pop_count > 1 { crate::debug!(5, "Popped {} times to reveal {} stack nodes", len, initial_pop_count); }

        // 2. Merge the revealed stack nodes that are identical
        popped_stack_nodes.bulk_merge();
        if initial_pop_count > 1 { crate::debug!(5, "Merged into {} stack nodes after pop", popped_stack_nodes.len()); }

        let mut new_stacks = Vec::new();
        for stack_node in popped_stack_nodes { // stack_node is Arc<GSSNode<...>> representing a state *before* the reduction's GOTO shift
            let revealed_content = stack_node.peek();
            let revealed_state_id = revealed_content.state_id;
            let revealed_t = &revealed_content.t;

            // 3. Perform GOTO lookup from the revealed state
            let goto_state_id = match self.parser.stage_7_table.get(&revealed_state_id) {
                Some(row) => match row.gotos.get(&nonterminal_id) {
                    Some(goto_id) => *goto_id,
                    None => {
                        // This indicates an error in the table generation or grammar.
                        // A valid parse should always find a GOTO after a reduction from a valid state.
                        crate::debug!(0, "CRITICAL ERROR: GOTO not found for state {} and non-terminal {} ({}) during reduction. Skipping path.", revealed_state_id.0, nt_name.0, nonterminal_id.0);
                        // Consider panicking here in production if the table should be guaranteed correct.
                        continue; // Skip this invalid path
                    }
                },
                None => {
                     // This indicates an error in the table generation or grammar.
                     crate::debug!(0, "CRITICAL ERROR: State {} not found in table during GOTO lookup for reduction. Skipping path.", revealed_state_id.0);
                     continue; // Skip this invalid path
                }
            };

            let node_ptr = Arc::as_ptr(&stack_node); // For logging
            crate::debug!(6, "  Reduced path via node {:p}: Revealed state {}, going to state {} for NT {}", node_ptr, revealed_state_id.0, goto_state_id.0, nt_name.0);

            // 4. Combine T values using intersection
            let combined_t = revealed_t.intersect(current_t);

            // 5. Create the new stack top by pushing the GOTO state
            let new_content = ParseStateNodeContent { state_id: goto_state_id, t: combined_t };
            let new_stack = stack_node.push(new_content); // push returns GSSNode
            new_stacks.push(Arc::new(new_stack)); // Wrap in Arc for GSS management
        }
        // Note: Merging of these new_stacks happens *after* collecting results
        // from all reductions triggered by a single state in the main loop.
        new_stacks
    }

    pub fn step(&mut self, token_id: TerminalID) {
        // --- Logging setup ---
        let root_nodes: Vec<_> = self.active_states.iter().map(|s| s.stack.clone()).collect();
        let stats = gather_gss_stats(&root_nodes);
        crate::debug!(3, "Step Start (Token {:?}): Active States: {}, GSS Stats: {:?}", token_id, self.active_states.len(), stats);
        const MAX_NODES_TO_PRINT: usize = 30;
        debug!(4, "{}", { // Use a closure to avoid potentially expensive calculations if debug level is lower
            let final_root_nodes: Vec<_> = self.active_states.iter().map(|s| s.stack.clone()).collect();
            let final_stats = gather_gss_stats(&final_root_nodes);
            if final_stats.unique_nodes <= MAX_NODES_TO_PRINT {
                format!("GSS Structure ({} nodes):\n{}", final_stats.unique_nodes, print_gss_forest(&final_root_nodes, MAX_NODES_TO_PRINT))
            } else {
                // Find and print the longest path instead
                if let Some(longest_path) = find_longest_path(&final_root_nodes) {
                    let path_str = longest_path.iter()
                        .map(|node| format!("{}", node.value.state_id.0))
                        .collect::<Vec<_>>()
                        .join(" -> ");
                    format!("GSS Structure too large ({} nodes > {}). Longest path ({} nodes): {}",
                            final_stats.unique_nodes, MAX_NODES_TO_PRINT, longest_path.len(), path_str)
                } else {
                    format!("GSS Structure too large ({} nodes > {}), and no path found.",
                            final_stats.unique_nodes, MAX_NODES_TO_PRINT)
                }
            }
        });


        // --- Initialization ---
        // Stores results of SHIFT actions, becomes the active_states for the *next* step.
        let mut next_active_states = Vec::new();
        // Stores states where the current token leads to no action.
        let mut current_action_not_found_states = Vec::new();
        // Worklist for the current step, initialized with current active states.
        // Reduction results will be added back to this list during processing.
        let mut worklist = std::mem::take(&mut self.active_states);

        let mut fuel = 1_000; // Safety break

        // --- Main Processing Loop ---
        // Process initial states and states generated by reductions within this step.
        while let Some(state) = worklist.pop() {
            // --- Fuel check ---
            if fuel == 0 {
                // Dump info and panic
                worklist.push(state); // Put the state back for debugging info
                let current_root_nodes: Vec<_> = worklist.iter().map(|s| s.stack.clone()).collect();
                let current_stats = gather_gss_stats(&current_root_nodes);
                crate::debug!(0, "Ran out of fuel! Current Worklist Size: {}, GSS Stats: {:?}", worklist.len(), current_stats);
                debug!(0, "{}", { // Use a closure to avoid potentially expensive calculations if debug level is lower
                    let final_root_nodes: Vec<_> = worklist.iter().map(|s| s.stack.clone()).collect();
                    let final_stats = gather_gss_stats(&final_root_nodes);
                    if final_stats.unique_nodes <= MAX_NODES_TO_PRINT {
                        format!("GSS Structure ({} nodes):\n{}", final_stats.unique_nodes, print_gss_forest(&final_root_nodes, MAX_NODES_TO_PRINT))
                    } else {
                        // Find and print the longest path instead
                        if let Some(longest_path) = find_longest_path(&final_root_nodes) {
                            let path_str = longest_path.iter()
                                .map(|node| format!("{}", node.value.state_id.0))
                                .collect::<Vec<_>>()
                                .join(" -> ");
                            format!("GSS Structure too large ({} nodes > {}). Longest path ({} nodes): {}",
                                    final_stats.unique_nodes, MAX_NODES_TO_PRINT, longest_path.len(), path_str)
                        } else {
                            format!("GSS Structure too large ({} nodes > {}), and no path found.",
                                    final_stats.unique_nodes, MAX_NODES_TO_PRINT)
                        }
                    }
                });
                panic!("Ran out of fuel during GLR step processing");
            }
            fuel -= 1;

            // --- Get current state info ---
            let current_content = state.stack.peek();
            let current_state_id = current_content.state_id;
            let current_t = &current_content.t; // Reference to T in the current top node

            // --- Action Lookup ---
            let row = match self.parser.stage_7_table.get(&current_state_id) {
                 Some(r) => r,
                 None => {
                     crate::debug!(0, "CRITICAL ERROR: State {} not found in parse table!", current_state_id.0);
                     current_action_not_found_states.push(state); // Treat as action not found
                     continue;
                 }
            };

            // Stores results from reductions triggered *by this specific state* before merging and adding back to worklist.
            let mut reduction_results_for_this_state = Vec::new();

            if let Some(action) = row.shifts_and_reduces.get(&token_id) {
                match action {
                    Stage7ShiftsAndReduces::Shift(next_state_id) => {
                        crate::debug!(5, "State {} -> {}: Shifting", current_state_id.0, next_state_id.0);
                        let new_content = ParseStateNodeContent { state_id: *next_state_id, t: current_t.clone() };
                        let new_stack = state.stack.push(new_content); // push returns GSSNode
                        // Add shift results directly to the list for the *next* step.
                        next_active_states.push(ParseState { stack: Arc::new(new_stack) });
                    }
                    Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id, len } => {
                        let nt_name = self.parser.non_terminal_map.get_by_right(nonterminal_id).unwrap();
                        crate::debug!(5, "State {}: Reducing by production {} ({}) len {}", current_state_id.0, production_id.0, nt_name.0, len);

                        // Perform the reduction, get resulting stacks (unmerged)
                        let new_stacks = self.perform_reduction(&state.stack, *len, *nonterminal_id, current_t);
                        // Add these new stacks to the collection for this state's reductions
                        reduction_results_for_this_state.extend(new_stacks);
                    }
                    Stage7ShiftsAndReduces::Split { shift, reduces } => {
                        crate::debug!(4, "State {}: Split action", current_state_id.0);
                        // Handle Shift part
                        if let Some(shift_state_id) = shift {
                            crate::debug!(5, "  Split -> Shift to {}", shift_state_id.0);
                            let new_content = ParseStateNodeContent { state_id: *shift_state_id, t: current_t.clone() };
                            let new_stack = state.stack.push(new_content);
                            // Add shift results directly to the list for the *next* step.
                            next_active_states.push(ParseState { stack: Arc::new(new_stack) });
                        }

                        // Handle Reduce part
                        crate::debug!(5, "  Split -> Reduces ({} lengths)", reduces.len());
                        for (len, nt_ids_map) in reduces {
                            crate::debug!(6, "    Reducing with len {}", len);
                            for (nt_id, prod_ids) in nt_ids_map { // prod_ids not used here, but available
                                 let nt_name = self.parser.non_terminal_map.get_by_right(nt_id).unwrap();
                                 crate::debug!(6, "      Reducing for NT {} ({} productions)", nt_name.0, prod_ids.len());
                                 // Perform the reduction for this specific NT and len
                                 let new_stacks = self.perform_reduction(&state.stack, *len, *nt_id, current_t);
                                 // Add results to the collection for this state's reductions
                                 reduction_results_for_this_state.extend(new_stacks);
                            }
                        }
                    }
                }
            } else {
                // No action found for this token in this state
                crate::debug!(4, "State {}: No action found for token {:?}", current_state_id.0, token_id);
                current_action_not_found_states.push(state);
            }

            // If reductions occurred for this state, merge the results and add them back to the worklist
            if !reduction_results_for_this_state.is_empty() {
                 crate::debug!(5, "Merging {} reduction results generated by state {}", reduction_results_for_this_state.len(), current_state_id.0);
                 reduction_results_for_this_state.bulk_merge();
                 crate::debug!(5, "Merged into {} states, adding back to worklist", reduction_results_for_this_state.len());
                 // Add merged reduction results back to the worklist for processing *within* this step
                 worklist.extend(reduction_results_for_this_state.into_iter().map(|stack_arc| ParseState { stack: stack_arc }));
            }

        } // End while worklist.pop()

        // --- Finalization ---
        // Merge the shift results collected for the next step
        crate::debug!(4, "Merging {} shift results for next step", next_active_states.len());
        // Convert Vec<ParseState> to Vec<Arc<GSSNode>> for bulk_merge
        let mut shift_stacks: Vec<_> = next_active_states.into_iter().map(|ps| ps.stack).collect();
        shift_stacks.bulk_merge();
        crate::debug!(4, "Merged into {} final active states for next step", shift_stacks.len());

        // Update active_states for the *next* step
        self.active_states = shift_stacks.into_iter().map(|stack_arc| ParseState { stack: stack_arc }).collect();

        // Update action_not_found_states
        self.action_not_found_states = current_action_not_found_states;

        // --- Logging End ---
        let end_root_nodes: Vec<_> = self.active_states.iter().map(|s| s.stack.clone()).collect();
        let end_stats = gather_gss_stats(&end_root_nodes);
        crate::debug!(3, "Step End (Token {:?}): Active States: {}, Action Not Found: {}, GSS Stats: {:?}", token_id, self.active_states.len(), self.action_not_found_states.len(), end_stats);

        // Clear action_not_found_states
        // TODO: Decide if keeping these across steps is ever useful. Currently cleared.
        self.action_not_found_states.clear();
    }

    // TODO: Review merge logic, especially interaction with GSSNode::merge and ParseState::merge
    pub fn merge_active_states(&mut self) {
        let mut active_state_map: BTreeMap<ParseStateKey, ParseState<T>> = BTreeMap::new();
        let num_active_states = self.active_states.len();

        for state in std::mem::take(&mut self.active_states) {
            let key = state.key();
            active_state_map.insert_with(key, state, |existing, new_state| {
                existing.merge(new_state);
            });
        }

        crate::debug!(3, "Merged {} active states into {} active states", num_active_states, active_state_map.len());
        self.active_states = active_state_map.into_values().collect();
    }

    pub fn merge_with(&mut self, other: GLRParserState<T>) {
        assert!(std::ptr::eq(self.parser, other.parser));
        self.active_states.extend(other.active_states);
        self.action_not_found_states.extend(other.action_not_found_states);
        // Consider merging active states here if performance becomes an issue
        // self.merge_active_states();
    }

    pub fn is_ok(&self) -> bool {
        !self.active_states.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
    // Removed action_stack
}

impl<T: MergeAndIntersect> ParseState<T> {
    pub fn key(&self) -> ParseStateKey {
        ParseStateKey {
            stack_state_id: self.stack.peek().state_id,
        }
    }

    /// Merges `other` into `self`. Assumes `self.key() == other.key()`.
    /// Merges the GSS structures and combines the `t` value at the top node using `MergeAndIntersect::merge`.
    pub fn merge(&mut self, other: ParseState<T>) {
        assert_eq!(self.key(), other.key());

        // Combine 't' values at the top node using 'or'
        let self_content = self.stack.peek();
        let other_content = other.stack.peek();
        let combined_t = self_content.t.merge(&other_content.t);

        // Get mutable access to self.stack, potentially cloning if shared (Arc > 1)
        let mut mutable_stack = Arc::make_mut(&mut self.stack);

        // Update the 't' value in the mutable top node's content
        mutable_stack.value.t = combined_t;

        // Merge the parent structures using GSSNode's merge
        mutable_stack.merge_unchecked(Arc::unwrap_or_clone(other.stack));
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
