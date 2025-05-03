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

/// Helper function to handle the reduction logic for a given state and production.
///
/// Pops the required number of states from the GSS, merges paths,
/// finds the GOTO state for each revealed path, intersects T values,
/// pushes the new state onto the GSS, and returns the resulting new stack tops.
fn handle_reduce<T: MergeAndIntersect + Debug>(
    parser: &GLRParser,
    stack: Arc<GSSNode<ParseStateNodeContent<T>>>,
    production_id: ProductionID, // For logging/debugging
    nonterminal_id: NonTerminalID,
    len: usize,
    current_t: &T, // T value from the state *before* reduction
) -> Vec<Arc<GSSNode<ParseStateNodeContent<T>>>> {
    let nt_name = parser.non_terminal_map.get_by_right(&nonterminal_id).unwrap(); // Assume valid ID
    let node_ptr = Arc::as_ptr(&stack);
    let current_state_id = stack.peek().state_id; // For logging
    debug!(5, "State {}, Node {:?}: Reducing by production {} ({}) with len {}", current_state_id.0, node_ptr, production_id.0, nt_name.0, len);

    // 1. Pop 'len' items from the stack. This returns multiple potential parent nodes.
    let mut popped_stack_nodes = stack.popn(len);
    let initial_popped_count = popped_stack_nodes.len(); // For debug logging

    // 2. Merge identical parent nodes revealed by popping.
    popped_stack_nodes.bulk_merge();
    if initial_popped_count > 1 { // Log only if merging could have occurred
         crate::debug!(4, "Popped {} times, merged {} revealed stack nodes into {}", len, initial_popped_count, popped_stack_nodes.len());
    }


    let mut resulting_stacks = Vec::new();
    // 3. For each unique revealed parent node:
    for stack_node in popped_stack_nodes { // stack_node is Arc<GSSNode<...>> representing a state *before* the reduction's RHS
        let revealed_content = stack_node.peek();
        let revealed_state_id = revealed_content.state_id;
        let revealed_t = &revealed_content.t;

        // 4. Look up the GOTO state based on the revealed state and the reduction's non-terminal.
        let goto_state_id = match parser.stage_7_table.get(&revealed_state_id) {
            Some(row) => match row.gotos.get(&nonterminal_id) {
                Some(goto) => *goto,
                None => {
                    // Error: Parse table is incomplete or inconsistent.
                    crate::error!("GOTO transition not found: state={}, nonterminal={}", revealed_state_id.0, nt_name.0);
                    // Skip this path; alternative could be panic or collecting errors.
                    continue;
                }
            },
            None => {
                // Error: Invalid state ID encountered.
                crate::error!("State {} not found in parse table during GOTO lookup", revealed_state_id.0);
                continue;
            }
        };

        let node_ptr = Arc::as_ptr(&stack_node); // For logging
        debug!(5, "  Node {:?}: Revealed state {}, GOTO state {} for NonTerminal {}", node_ptr, revealed_state_id.0, goto_state_id.0, nt_name.0);

        // 5. Combine T values using intersect.
        let combined_t = revealed_t.intersect(current_t);
        let new_content = ParseStateNodeContent { state_id: goto_state_id, t: combined_t };

        // 6. Push the new state (GOTO state + combined T) onto the revealed stack node.
        let new_stack = stack_node.push(new_content);
        resulting_stacks.push(Arc::new(new_stack));
    }

    // 7. Merge any identical stacks resulting from different reduction paths.
    resulting_stacks.bulk_merge();
    if initial_popped_count > 1 { // Log only if merging could have occurred
        crate::debug!(4, "Reduction produced {} resulting stacks after merge", resulting_stacks.len());
    }

    resulting_stacks
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

    pub fn step(&mut self, token_id: TerminalID) {
        // --- Logging Setup ---
        let start_root_nodes: Vec<_> = self.active_states.iter().map(|s| s.stack.clone()).collect();
        let start_stats = gather_gss_stats(&start_root_nodes);
        crate::debug!(3, "Step Start (Token {:?}): Active States: {}, GSS Stats: {:?}", token_id, self.active_states.len(), start_stats);

        const MAX_NODES_TO_PRINT: usize = 30;
        debug!(4, "{}", { // Lazy GSS structure logging
            if start_stats.unique_nodes <= MAX_NODES_TO_PRINT {
                format!("Initial GSS Structure ({} nodes):\n{}", start_stats.unique_nodes, print_gss_forest(&start_root_nodes, MAX_NODES_TO_PRINT))
            } else if let Some(longest_path) = find_longest_path(&start_root_nodes) {
                let path_str = longest_path.iter().map(|node| format!("{}", node.value.state_id.0)).collect::<Vec<_>>().join(" -> ");
                format!("Initial GSS Structure too large ({} nodes > {}). Longest path ({} nodes): {}", start_stats.unique_nodes, MAX_NODES_TO_PRINT, longest_path.len(), path_str)
            } else {
                format!("Initial GSS Structure too large ({} nodes > {}), and no path found.", start_stats.unique_nodes, MAX_NODES_TO_PRINT)
            }
        });

        // --- State Processing Setup ---
        // Worklist contains states to process for the current token. Initially, it's the active states from the previous step.
        let mut worklist = std::mem::take(&mut self.active_states);
        // Stores states resulting from SHIFT actions, ready for the *next* token. Merged by key.
        let mut next_active_states_map: BTreeMap<ParseStateKey, ParseState<T>> = BTreeMap::new();
        // Stores states where no action was found for the current token.
        let mut current_action_not_found_states = Vec::new();
        // Tracks stack nodes already processed in this step to prevent redundant work/loops caused by reductions.
        let mut processed_for_token: BTreeSet<Arc<GSSNode<ParseStateNodeContent<T>>>> = BTreeSet::new();


        // --- Main Processing Loop ---
        // Process states until the worklist is empty. Reductions add states back to the worklist.
        while let Some(current_state) = worklist.pop() {
            // Avoid reprocessing the same stack state within this step.
            if !processed_for_token.insert(current_state.stack.clone()) {
                continue; // Already processed this stack top for this token
            }

            let stack = &current_state.stack;
            let current_content = stack.peek();
            let current_state_id = current_content.state_id;
            let current_t = &current_content.t; // Reference to T

            // Find the action for the current state and input token.
            let row = match self.parser.stage_7_table.get(&current_state_id) {
                 Some(r) => r,
                 None => {
                    crate::error!("State {} not found in parse table during step", current_state_id.0);
                    current_action_not_found_states.push(current_state); // Treat as error/no action
                    continue;
                 }
            };

            match row.shifts_and_reduces.get(&token_id) {
                // --- Shift Action ---
                Some(Stage7ShiftsAndReduces::Shift(next_state_id)) => {
                    debug!(5, "State {}: Shifting to state {}", current_state_id.0, next_state_id.0);
                    let new_content = ParseStateNodeContent {
                        state_id: *next_state_id,
                        t: current_t.clone(), // T value carries over on shift
                    };
                    let new_stack = stack.push(new_content);
                    let shifted_state = ParseState { stack: Arc::new(new_stack) };

                    // Add to map for the *next* step, merging if key exists.
                    next_active_states_map.insert_with(
                        shifted_state.key(),
                        shifted_state,
                        |existing, new| existing.merge(new), // Use ParseState::merge
                    );
                }
                // --- Reduce Action ---
                Some(Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id, len }) => {
                    // Delegate to the helper function.
                    let reduced_stacks = handle_reduce(
                        self.parser,
                        stack.clone(), // Clone Arc for the helper
                        *production_id,
                        *nonterminal_id,
                        *len,
                        current_t, // Pass reference to current T
                    );

                    // Add resulting states back to the worklist for processing *in this step*.
                    for reduced_stack in reduced_stacks {
                        worklist.push(ParseState { stack: reduced_stack });
                    }
                }
                // --- Split (Shift/Reduce or Reduce/Reduce Conflict) ---
                Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                    debug!(4, "State {}: Split action for token {:?}", current_state_id.0, token_id);

                    // Handle Shift part (if it exists)
                    if let Some(shift_state_id) = shift {
                        debug!(5, "  Split: Shifting to state {}", shift_state_id.0);
                        let shift_content = ParseStateNodeContent {
                            state_id: *shift_state_id,
                            t: current_t.clone(),
                        };
                        let shifted_stack = stack.push(shift_content);
                        let shifted_state = ParseState { stack: Arc::new(shifted_stack) };
                        next_active_states_map.insert_with(
                            shifted_state.key(),
                            shifted_state,
                            |existing, new| existing.merge(new),
                        );
                    }

                    // Handle Reduce parts
                    for (len, nt_id_to_prod_ids) in reduces {
                         debug!(5, "  Split: Reducing with len {}", len);
                         for (nt_id, prod_ids) in nt_id_to_prod_ids {
                            // Pick one production ID for logging purposes within handle_reduce.
                            let representative_prod_id = prod_ids.iter().next().cloned().unwrap_or(ProductionID(usize::MAX));

                            let reduced_stacks = handle_reduce(
                                self.parser,
                                stack.clone(), // Clone Arc
                                representative_prod_id,
                                *nt_id,
                                *len,
                                current_t,
                            );
                            // Add resulting states back to the worklist.
                            for reduced_stack in reduced_stacks {
                                worklist.push(ParseState { stack: reduced_stack });
                            }
                        }
                    }
                }
                // --- No Action ---
                None => {
                    debug!(4, "State {}: No action found for token {:?}", current_state_id.0, token_id);
                    current_action_not_found_states.push(current_state);
                }
            } // End match action
        } // End while worklist not empty


        // --- Update Parser State for Next Step ---
        self.active_states = next_active_states_map.into_values().collect();
        self.action_not_found_states = current_action_not_found_states;

        // --- Logging at Step End ---
        let end_root_nodes: Vec<_> = self.active_states.iter().map(|s| s.stack.clone()).collect();
        let end_stats = gather_gss_stats(&end_root_nodes);
        crate::debug!(3, "Step End (Token {:?}): Active States: {}, Action Not Found: {}, GSS Stats: {:?}", token_id, self.active_states.len(), self.action_not_found_states.len(), end_stats);

        debug!(4, "{}", { // Lazy GSS structure logging
             if end_stats.unique_nodes <= MAX_NODES_TO_PRINT {
                 format!("Final GSS Structure ({} nodes):\n{}", end_stats.unique_nodes, print_gss_forest(&end_root_nodes, MAX_NODES_TO_PRINT))
             } else if let Some(longest_path) = find_longest_path(&end_root_nodes) {
                 let path_str = longest_path.iter().map(|node| format!("{}", node.value.state_id.0)).collect::<Vec<_>>().join(" -> ");
                 format!("Final GSS Structure too large ({} nodes > {}). Longest path ({} nodes): {}", end_stats.unique_nodes, MAX_NODES_TO_PRINT, longest_path.len(), path_str)
             } else {
                 format!("Final GSS Structure too large ({} nodes > {}), and no path found.", end_stats.unique_nodes, MAX_NODES_TO_PRINT)
             }
        });

        // Optional: Clear action_not_found_states if they are not used after the step.
        self.action_not_found_states.clear();
    }

    pub fn merge_with(&mut self, other: GLRParserState<T>) {
        assert!(std::ptr::eq(self.parser, other.parser));
        self.active_states.extend(other.active_states);
        self.action_not_found_states.extend(other.action_not_found_states);
        // Consider merging active states here if performance becomes an issue
        // self.merge_active_states(); // Note: merge_active_states method is now removed
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
