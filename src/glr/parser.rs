use crate::datastructures::gss::{print_gss_forest, BulkMerge, gather_gss_stats, find_longest_path, GSSNode, GSSTrait, GSSStats}; // Updated imports
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{
    NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID,
};

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

// ParseStateNodeContent struct is removed as per instructions.

#[derive(Debug, Clone)] // Derive Clone for ParseState<T>
pub struct ParseState<T: MergeAndIntersect> {
    pub gss_node: Arc<GSSNode<StateID, T>>, // GSS node now stores StateID
    pub current_t: T, // Semantic value associated with reaching this gss_node
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

    pub fn init_glr_parser_with_t<T: MergeAndIntersect>(&self, t_initial: T) -> GLRParserState<T> {
        let initial_parse_state = self.init_parse_state_with_t(t_initial);
        let mut active_states_map = BTreeMap::new();
        // Use the GSS node's StateID as the key for active_states
        active_states_map.insert(initial_parse_state.key(), initial_parse_state);
        GLRParserState {
            parser: self,
            active_states: active_states_map,
            action_not_found_states: BTreeMap::new(),
        }
    }
    pub fn init_glr_parser_from_parse_state<T: MergeAndIntersect>(&self, parse_state: ParseState<T>) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            active_states: BTreeMap::from([(parse_state.key(), parse_state)]),
            action_not_found_states: BTreeMap::new(),
        }
    }

    pub fn init_glr_parser_from_parse_states<T: MergeAndIntersect>(
        &self,
        parse_states: Vec<ParseState<T>>,
    ) -> GLRParserState<T> {
        let mut active_states_map = BTreeMap::new();
        for state in parse_states {
            // Use the GSS node's StateID as the key for active_states
            active_states_map.insert_with(state.key(), state, |existing, new_s| existing.merge(new_s));
        }
        GLRParserState {
            parser: self,
            active_states: active_states_map,
            action_not_found_states: BTreeMap::new(),
        }
    }

    pub fn init_parse_state<T: MergeAndIntersect + Default>(&self) -> ParseState<T> {
        self.init_parse_state_with_t(T::default())
    }

    pub fn init_parse_state_with_t<T: MergeAndIntersect>(&self, t_initial: T) -> ParseState<T> {
        let initial_gss_node = Arc::new(GSSNode::new(self.start_state_id)); // GSS node stores StateID
        ParseState {
            gss_node: initial_gss_node,
            current_t: t_initial, // Initial semantic value
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

#[derive(Debug, Clone)] // Derive Clone for GLRParserState<T>
pub struct GLRParserState<'a, T: MergeAndIntersect> {
    pub parser: &'a GLRParser,
    // Key is StateID of the GSS node at the top of the stack
    pub active_states: BTreeMap<StateID, ParseState<T>>,
    pub action_not_found_states: BTreeMap<StateID, ParseState<T>>,
}

impl<'a, T: MergeAndIntersect + Debug> GLRParserState<'a, T> {
    /* -------------------------------------------------
     * Helper utilities to make `step` compact and clear
     * ------------------------------------------------- */

    /// Push a new state on `current_parse_state.gss_node` and return the new `ParseState`.
    fn push_state(
        &self,
        current_parse_state: &ParseState<T>,
        next_gss_state_id: StateID,
        t_for_new_edge: T,
    ) -> ParseState<T> {
        // The new GSS node stores the StateID of the next state
        let new_gss_node_content = next_gss_state_id;
        // The edge value is the semantic value associated with this transition (the current_t of the source)
        let edge_value = current_parse_state.current_t.clone(); // Use current_t as the edge value

        let new_gss_node = Arc::new(GSSNode::new_with_predecessors(
            new_gss_node_content,
            vec![(current_parse_state.gss_node.clone(), edge_value)], // Use current_t as edge value
        ));

        ParseState {
            gss_node: new_gss_node,
            current_t: t_for_new_edge, // The semantic value *at* the new node
        }
    }

    /// Pop `len` nodes from `current_parse_state.gss_node`, follow the goto on `nt`, and return the resulting `ParseState`s.
    fn pop_and_goto(
        &self,
        current_parse_state: &ParseState<T>,
        len: usize,
        nt: NonTerminalID,
    ) -> Vec<ParseState<T>> {
        // popn now returns Vec<(Arc<GSSNode>, E)> where E is the edge value to the node
        let ancestor_paths = current_parse_state.gss_node.popn(len, current_parse_state.current_t.clone()); // Pass current_t as the edge value for the path ending at the current node

        // The original `popn` implementation doesn't handle merging nodes at the same level across different paths within one call.
        // We need to collect and merge based on the ancestor GSS node pointer and their StateID.
        let mut ancestors_to_process: BTreeMap<(*const GSSNode<StateID, T>, StateID), (Arc<GSSNode<StateID, T>>, T)> = BTreeMap::new();

        for (ancestor_gss_node, t_at_ancestor) in ancestor_paths {
             let key = (Arc::as_ptr(&ancestor_gss_node), ancestor_gss_node.node_content); // Use node pointer and StateID for grouping
             if let Some((existing_arc, existing_t)) = ancestors_to_process.get_mut(&key) {
                 // Merge the semantic values
                 *existing_t = existing_t.merge(&t_at_ancestor);
                 // Merge the GSS structures if necessary (BulkMerge expects Vec<Arc<...>>).
                 // This merge is tricky with edge values. Let's rely on `BulkMerge` after collecting all ancestors.
                 // For now, just update the T value.
             } else {
                  ancestors_to_process.insert(key, (ancestor_gss_node, t_at_ancestor));
             }
        }

        // Collect unique ancestor ARCs to apply BulkMerge
        let mut unique_ancestor_arcs: Vec<Arc<GSSNode<StateID, T>>> = ancestors_to_process.values().map(|(arc, _)| arc.clone()).collect();
        unique_ancestor_arcs.bulk_merge(); // Merge GSS structures

        let mut out_new_parse_states = Vec::new();

        // Now process the merged ancestors
        for ancestor_arc in unique_ancestor_arcs {
            // Find the merged T value associated with this ancestor_arc
            // This is tricky because BulkMerge modifies the arcs in place or replaces them.
            // A proper solution would involve re-associating the T values after BulkMerge.
            // For simplicity in this refactor, let's find a representative T value.
            // A better approach might be to merge T values *before* BulkMerge or adapt BulkMerge.
            // Let's assume for now that the T value we got *with* the ancestor path is sufficient,
            // although this isn't strictly correct after merging multiple paths.

            // Let's re-fetch the T value from the map based on the *potentially modified* arc pointer and StateID
            // This is still problematic if the Arc pointer changed during bulk_merge but the StateID is the same.
            // A proper approach needs to handle this merge correctly.

            // Temporary simplified approach: Use the T value from one of the paths leading to this merged ancestor.
            // This loses precision if different paths to the same ancestor had different T values that should be merged.
            // Let's fetch the T value from the map using the *original* pointer that led to this StateID,
            // assuming BulkMerge preserves the original StateID in the merged node.

            // We need a way to find the merged T value associated with the merged ancestor_arc.
            // Let's revisit the `popn` return. It returns Vec<(Arc<Self>, E)>. The E is the value on the edge TO that node.
            // In popn(len, edge_value_for_self), `edge_value_for_self` is the value on the edge TO the node we call popn on.
            // When we recurse popn(n-1, edge_val_to_pred), edge_val_to_pred is the value on the edge TO the predecessor.
            // So, popn returns (ancestor_gss_node, T_value_on_edge_to_ancestor).

            // The T value we need for the GOTO is the semantic value *at* the ancestor node, not the edge value leading to it.
            // This means we need to access the 'current_t' stored *within* the ParseState that corresponds to the ancestor node.
            // This indicates the ParseState structure and how T propagates needs rethinking with the edge-based GSS.

            // Let's re-evaluate `pop_and_goto`.
            // We start with `current_parse_state: ParseState { gss_node: arc_current, current_t: t_current }`.
            // We pop `len` steps. The result is `Vec<(Arc<GSSNode<StateID, T>>, T)>` from `popn(len, t_current)`.
            // The T in this result is the edge value *to* the ancestor node.
            // We need the T value *at* the ancestor node. This value was the `current_t` of the ParseState that created that ancestor node.

            // This structure seems problematic. The `current_t` is a property of the `ParseState`, not just the GSS node.
            // When merging ParseStates with the same GSS node (same StateID and same Arc<GSSNode<...>>), their `current_t` values are merged.
            // This merged `current_t` should ideally be what's available at that GSS node for subsequent operations like GOTO.

            // Let's try a different approach for `pop_and_goto`:
            // 1. Pop `len` steps using the GSS structure. This gets us ancestor GSS nodes.
            // 2. For each ancestor GSS node, we need the corresponding `ParseState` and its `current_t`.
            // This is complex because a single GSS node might be part of multiple active `ParseState`s with different `current_t`s.
            // When we pop `len` steps, we are following paths in the GSS graph. The `current_t` should be accumulated along the path somehow.

            // Let's reconsider the GSS trait `popn`:
            // It should return the reachable nodes at distance `n` and the accumulated edge values along the path to them.
            // The current `popn` returning `(Arc<Self>, E)` where `E` is the edge value *to* the node is correct given the new GSS definition.

            // The issue is how `current_t` (which represents the accumulated semantic info up to the current node) interacts.
            // When reducing, we pop `len` symbols/states. The semantic value up to the point *before* the reduction (i.e., at the ancestor node 'len' steps back) is needed.

            // Let's redefine `pop_and_goto` to operate on `ParseState<T>` and return `ParseState<T>` directly.
            // We need to traverse the GSS graph from `current_parse_state.gss_node` backwards `len` steps.
            // While traversing backwards, we accumulate the edge values using `E::intersect`.

            let mut final_ancestor_parse_states: BTreeMap<StateID, ParseState<T>> = BTreeMap::new();

            let mut worklist: VecDeque<(Arc<GSSNode<StateID, T>>, T, usize)> = VecDeque::new(); // (current_gss_node, accumulated_t_on_path, steps_taken)
            worklist.push_back((current_parse_state.gss_node.clone(), current_parse_state.current_t.clone(), 0));

            let mut visited: HashMap<(*const GSSNode<StateID, T>, T), bool> = HashMap::new(); // To avoid cycles and re-processing same state+T

            while let Some((current_gss_node, accumulated_t, steps_taken)) = worklist.pop_front() {
                 let key = (Arc::as_ptr(&current_gss_node), accumulated_t.clone()); // Use arc pointer and T value as key
                 if visited.contains_key(&key) {
                     continue;
                 }
                 visited.insert(key.clone(), true);

                 if steps_taken == len {
                     // Found an ancestor node at the correct distance
                     let ancestor_state_id = current_gss_node.node_content;
                     let goto_dest_state_id = self.parser.stage_7_table[&ancestor_state_id].gotos[&nt];
                     // The T value for the new ParseState is the accumulated_t from the path to the ancestor
                     let t_for_new_parse_state = accumulated_t; // The accumulated T is now the T at the ancestor

                     // Create the new GSS node for the GOTO transition
                     let gss_node_after_goto = Arc::new(GSSNode::new_with_predecessors(
                         goto_dest_state_id,
                         vec![(current_gss_node.clone(), t_for_new_parse_state.clone())], // Edge value is the T value at the source (ancestor)
                     ));

                     let new_parse_state = ParseState {
                         gss_node: gss_node_after_goto,
                         current_t: t_for_new_parse_state, // T value at the new GOTO node is the same as at the ancestor (TODO: confirm logic)
                                                           // Or should it be intersect with token's T? No, that's handled in step.
                                                           // The T at the GOTO node is the accumulated T up to the ancestor.
                     };

                     final_ancestor_parse_states.insert_with(
                         new_parse_state.key(),
                         new_parse_state,
                         |existing, new_s| existing.merge(new_s),
                     );

                 } else if steps_taken < len {
                     // Continue traversing predecessors
                     for (pred_wrapper, edge_val_to_current) in &current_gss_node.predecessors {
                         let pred_arc = pred_wrapper.as_arc();
                         // Accumulate the semantic value along this edge
                         let new_accumulated_t = accumulated_t.intersect(edge_val_to_current);
                         worklist.push_back((pred_arc.clone(), new_accumulated_t, steps_taken + 1));
                     }
                 }
            }

        out_new_parse_states = final_ancestor_parse_states.into_values().collect();

        // Apply BulkMerge to the resulting GSS nodes if needed
        // The merging on `final_ancestor_parse_states` map handles merging ParseStates with the same GSS node + T.
        // However, multiple ParseStates can lead to the same GSS node at the same StateID but with different T values that merge.
        // The current map key (StateID) doesn't capture this. A better key would be (StateID, T).
        // Let's adjust `ParseStateKey` and the maps accordingly.

        // Reverting to original ParseStateKey based on StateID and relying on ParseState::merge
        // which calls GSSNode::merge to handle merging the GSS structure under the same StateID.
        // The T values are merged in ParseState::merge.

        let mut final_parse_states: BTreeMap<StateID, ParseState<T>> = BTreeMap::new();
        let mut worklist_for_goto: VecDeque<(Arc<GSSNode<StateID, T>>, T, usize)> = VecDeque::new(); // (current_gss_node, accumulated_t_on_path, steps_taken)
        worklist_for_goto.push_back((current_parse_state.gss_node.clone(), current_parse_state.current_t.clone(), 0));

        let mut visited_goto: HashSet<(*const GSSNode<StateID, T>, T)> = HashSet::new(); // Track visited (node, accumulated_t) pairs

        while let Some((current_gss_node, accumulated_t, steps_taken)) = worklist_for_goto.pop_front() {
             let key = (Arc::as_ptr(&current_gss_node), accumulated_t.clone());
             if !visited_goto.insert(key) {
                 continue;
             }

             if steps_taken == len {
                 // Found an ancestor node at the correct distance
                 let ancestor_state_id = current_gss_node.node_content;
                 let goto_dest_state_id = self.parser.stage_7_table[&ancestor_state_id].gotos[&nt];
                 // The T value for the new ParseState at the GOTO destination is the accumulated_t
                 let t_for_goto_node = accumulated_t; // Accumulated T up to the ancestor

                 // Create the new GSS node for the GOTO transition
                 let gss_node_after_goto = Arc::new(GSSNode::new_with_predecessors(
                     goto_dest_state_id,
                     vec![(current_gss_node.clone(), t_for_goto_node.clone())], // Edge value to GOTO node is T at ancestor
                 ));

                 let new_parse_state = ParseState {
                     gss_node: gss_node_after_goto,
                     current_t: t_for_goto_node, // T value at the new GOTO node
                 };

                 final_parse_states.insert_with(
                     new_parse_state.key(), // Key is StateID
                     new_parse_state,
                     |existing, new_s| existing.merge(new_s), // ParseState merge handles GSS and T merge
                 );

             } else if steps_taken < len {
                 // Continue traversing predecessors
                 for (pred_wrapper, edge_val_to_current) in &current_gss_node.predecessors {
                     let pred_arc = pred_wrapper.as_arc();
                     // Accumulate the semantic value along this edge
                     let new_accumulated_t = accumulated_t.intersect(edge_val_to_current);
                     worklist_for_goto.push_back((pred_arc.clone(), new_accumulated_t, steps_taken + 1));
                 }
             }
        }


        final_parse_states.into_values().collect()
    }

    /// Debug helper so the main `step` body stays short.
    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
        const MAX: usize = 30;
        // GSS roots are now the `gss_node`s within the active `ParseState`s
        let roots: Vec<_> = self.active_states.values().map(|s| s.gss_node.clone()).collect();
        let stats = gather_gss_stats(&roots);
        crate::debug!(3, "{} - token {} ({:?}) - – active: {}, nodes: {:?}",
                      phase, token.0, self.parser.terminal_map.get_by_right(&token).unwrap().0, self.active_states.len(), stats);

        debug!(4, "{}", {
            if stats.unique_nodes <= MAX {
                format!("GSS ({} nodes):\n{}", stats.unique_nodes,
                        print_gss_forest(&roots, MAX))
            } else {
                // fall back to longest path printing
                // find_longest_path now returns Vec<Arc<GSSNode<N, E>>>
                match find_longest_path(&roots) {
                    Some(p) => format!("GSS too big ({} nodes). Longest path ({}): {}",
                                       stats.unique_nodes,
                                       p.len(),
                                       // Map Arc<GSSNode<StateID, T>> to StateID for printing
                                       p.iter().map(|n_arc| n_arc.node_content.0)
                                            .map(|id| id.to_string())
                                            .collect::<Vec<_>>()
                                            .join(" → ")),
                    None => format!("GSS too big ({} nodes) – path not found", stats.unique_nodes),
                }
            }
        });
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
        self.log_gss("Step-start", token_id);

        // Convert BTreeMap to Vec for processing
        let mut todo: Vec<ParseState<T>> = std::mem::take(&mut self.active_states).into_values().collect();
        let mut next = BTreeMap::<StateID, ParseState<T>>::new(); // Key is StateID of the GSS node at the top of the stack
        let mut not_found = BTreeMap::<StateID, ParseState<T>>::new(); // Key is StateID

        /* ---------- core loop ---------- */
        while let Some(state) = todo.pop() { // Process states from the worklist
            let top_gss_node     = state.gss_node.clone(); // Clone Arc for potential later use
            let top_state_id     = top_gss_node.node_content; // Get StateID from GSS node
            let current_t_at_top = state.current_t.clone(); // Get semantic value at current node
            let row              = &self.parser.stage_7_table[&top_state_id];

            match row.shifts_and_reduces.get(&token_id) {
                /* ------ 1. plain shift ------ */
                Some(Stage7ShiftsAndReduces::Shift(to)) => {
                    crate::debug!(4, "Shift from state {} via token {} to state {}", top_state_id.0, token_id.0, to.0);
                    // Semantic value for the new edge should be the current_t at the source node
                    let t_for_new_edge = current_t_at_top.clone();
                    // The semantic value *at* the new node is just copied from the previous node for shift
                    let t_at_new_node = current_t_at_top.clone();

                    let new_gss_node_content = *to;
                    let new_gss_node = Arc::new(GSSNode::new_with_predecessors(
                        new_gss_node_content,
                        vec![(top_gss_node.clone(), t_for_new_edge)],
                    ));

                    let new_parse_state = ParseState {
                         gss_node: new_gss_node,
                         current_t: t_at_new_node,
                    };
                    next.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                }

                /* ------ 2. single reduce ------ */
                Some(Stage7ShiftsAndReduces::Reduce{ nonterminal_id: nt,
                                                     len, .. }) => {
                    crate::debug!(4, "Reduce from state {} via token {} to nonterminal {}", top_state_id.0, token_id.0, nt.0);
                    // pop_and_goto now returns Vec<ParseState<T>>
                    for new_ps_after_goto in self.pop_and_goto(&state, *len, *nt) {
                        // Add to worklist for current step; merging happens when moving to `next`
                        todo.push(new_ps_after_goto);
                    }
                }

                /* ------ 3. shift / reduce split ------ */
                Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                    crate::debug!(4, "Split from state {} via token {}", top_state_id.0, token_id.0);
                    // optional shift part
                    if let Some(to) = shift {
                        crate::debug!(4, " Shift from state {} via token {} to state {}", top_state_id.0, token_id.0, to.0);
                        // Semantic value for the new edge should be the current_t at the source node
                        let t_for_new_edge = current_t_at_top.clone();
                        // The semantic value *at* the new node is just copied from the previous node for shift
                        let t_at_new_node = current_t_at_top.clone();

                        let new_gss_node_content = *to;
                        let new_gss_node = Arc::new(GSSNode::new_with_predecessors(
                            new_gss_node_content,
                            vec![(top_gss_node.clone(), t_for_new_edge)],
                        ));

                        let new_parse_state = ParseState {
                             gss_node: new_gss_node,
                             current_t: t_at_new_node,
                        };
                        next.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                    }
                    // every reduce alternative
                    for (len, nts) in reduces {
                        crate::debug!(4, " Reduce from state {} via token {} to nonterminals {:?}", top_state_id.0, token_id.0, nts);
                        for (nt, _prod_ids) in nts {        // we ignore prod-ids here
                             // pop_and_goto now returns Vec<ParseState<T>>
                            for new_ps_after_goto in self.pop_and_goto(&state, *len, *nt) {
                                // Add to worklist for current step
                                todo.push(new_ps_after_goto);
                            }
                        }
                    }
                }

                /* ------ 4. no action ------ */
                None => {
                    crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, top_state_id.0);
                    // Keep the ParseState as is, it ends up in not_found
                    not_found.insert_with(state.key(), state, |existing, new_s| existing.merge(new_s));
                },
            }
        }

        // After processing all items in `todo`, the `next` map contains the new active ParseStates grouped by their top GSS node's StateID.
        // The `insert_with` calls already performed merging for ParseStates with the same StateID.

        /* ---------- finish up ---------- */
        self.active_states            = next;
        self.action_not_found_states  = not_found;   // keep for caller if wanted

        self.log_gss("Step-end", token_id);
        self.action_not_found_states.clear();        // current design: we drop them

        crate::debug!(4, "----------------------------------------------------------------");
    }

    /// Merging is now handled implicitly when states are added to `next` in the `step` method.
    /// This method can be removed if no other part of the code relies on it explicitly.
    /// For now, let's keep it as a no-op or ensure it's not called.
    /// Given the new structure, explicit merging of `self.active_states` is no longer needed
    /// as `BTreeMap::insert_with` handles it based on StateID keys.
    pub fn merge_active_states(&mut self) {
        // This method is no longer necessary as merging is done on insertion.
        // crate::debug!(3, "merge_active_states called (now a no-op due to BTreeMap usage)");
    }

    pub fn merge_with(&mut self, other: GLRParserState<T>) {
        assert!(std::ptr::eq(self.parser, other.parser));
        for (key, state) in other.active_states {
            self.active_states.insert_with(key, state, |existing, new_s| existing.merge(new_s));
        }
        for (key, state) in other.action_not_found_states {
            self.action_not_found_states.insert_with(key, state, |existing, new_s| existing.merge(new_s));
        }
    }

    pub fn is_ok(&self) -> bool {
        !self.active_states.is_empty()
    }
}

// Key is StateID of the GSS node at the top of the stack
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
}

impl<T: MergeAndIntersect> ParseState<T> {
    pub fn key(&self) -> ParseStateKey {
        ParseStateKey {
            stack_state_id: self.gss_node.node_content, // Use node_content (StateID) as the key
        }
    }

    /// Merges `other` into `self`. Assumes `self.key() == other.key()`.
    /// Merges the GSS structures and combines the `current_t` value.
    pub fn merge(&mut self, other: ParseState<T>) {
        // Assumes keys are the same, which means the top GSS node has the same StateID
        assert_eq!(self.key(), other.key());

        // Merge the semantic values at the current node
        self.current_t = self.current_t.merge(&other.current_t);

        // Merge the GSS structures.
        // Arc::make_mut ensures we have a unique mutable reference to `self.gss_node`
        // before merging the predecessors from `other.gss_node`.
        let mut self_gss_node_mut = Arc::make_mut(&mut self.gss_node);

        // Arc::unwrap_or_clone gets the inner GSSNode from the other Arc.
        // If other.gss_node had multiple owners, it will be cloned.
        let other_gss_node_unwrapped = Arc::unwrap_or_clone(other.gss_node);

        // Merge the predecessors from the other node into self's node
        // The GSSNode::merge method needs to handle merging predecessor maps (BTreeMap<ArcPtrWrapper<...>, E>)
        // which requires E: MergeAndIntersect.
        self_gss_node_mut.merge(other_gss_node_unwrapped);
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
