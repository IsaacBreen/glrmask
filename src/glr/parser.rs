use crate::datastructures::gss::{print_gss_forest, BulkMerge, gather_gss_stats, find_longest_path, GSSTrait, GSSNode};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{
    NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID,
};
use crate::datastructures::gss::GSSStats;

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;
use crate::debug;

pub trait MergeAndIntersect: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash + Default {
    /// Merges the information represented by `self` and `other`.
    fn merge(&self, other: &Self) -> Self;
    /// Intersects the information represented by `self` and `other`.
    fn intersect(&self, other: &Self) -> Self;
}

impl MergeAndIntersect for () {
    fn merge(&self, _: &Self) -> Self { () }
    fn intersect(&self, _: &Self) -> Self { () }
    fn default() -> Self { () }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateNodeContent<T: MergeAndIntersect> {
    pub state_id: StateID,
    pub t: T,
}

// Use type alias for the stack head
type ParseStack<T> = Arc<GSSNode<ParseStateNodeContent<T>>>;


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
        let initial_stack = self.init_parse_state_with_t(t);
        GLRParserState {
            parser: self,
            head: initial_stack,
            not_found: Arc::new(GSSNode::new()), // Initialize not_found as an empty node
        }
    }

    pub fn init_glr_parser_from_parse_state<T: MergeAndIntersect>(&self, parse_state_stack: ParseStack<T>) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            head: parse_state_stack,
            not_found: Arc::new(GSSNode::new()), // Initialize not_found as an empty node
        }
    }

    // init_glr_parser_from_parse_states is removed as we now have a single head.
    // Merging multiple starting states happens outside this function now, before creating the GLRParserState.


    pub fn init_parse_state<T: MergeAndIntersect + Default>(&self) -> ParseStack<T> {
        self.init_parse_state_with_t(T::default())
    }

    pub fn init_parse_state_with_t<T: MergeAndIntersect>(&self, t: T) -> ParseStack<T> {
        let initial_content = ParseStateNodeContent {
            state_id: self.start_state_id,
            t,
        };
        // Create a root node (no predecessors) and push the initial state content onto it.
        let root_node = Arc::new(GSSNode::new());
        root_node.push(initial_content)
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
    pub head: ParseStack<T>,
    not_found: ParseStack<T>, // Keep for action_not_found states
}

impl<'a, T: MergeAndIntersect + Debug> GLRParserState<'a, T> {
    /* -------------------------------------------------
     * Helper utilities to make `step` compact and clear
     * ------------------------------------------------- */

    /// Push a new state on `stack` and return the resulting new stack head.
    fn push_state(
        &self,
        stack: ParseStack<T>,
        next_state: StateID,
        t: T, // label for the new edge
    ) -> ParseStack<T> {
        let new_content = ParseStateNodeContent { state_id: next_state, t };
        stack.push(new_content)
    }

    /// Pop `len` nodes, follow the goto on `nt`, and return the resulting stacks.
    /// Returns a vector of potential new head nodes after reduction.
    fn pop_and_goto(
        &self,
        stack: ParseStack<T>,
        len: usize,
        nt: NonTerminalID,
        cur_t: &T, // t value from the node *before* the production match
    ) -> Vec<ParseStack<T>>
    where T: Clone + Eq + Hash // Bounds needed for popn and bulk_merge
    {
        let mut parents = stack.popn(len); // Returns Vec<(Arc<GSSNode<T>>, T)> where T is the label on edge into the parent
        parents.bulk_merge(); // Merges parents that are the same GSSNode instance, merging their incoming labels

        let mut next_stacks: Vec<ParseStack<T>> = Vec::new();

        // parents is now Vec<(Arc<GSSNode<T>>, T)>
        // Each tuple is (parent_node_arc, label_on_edge_to_parent)
        for (parent_node_arc, parent_label_content) in parents {
            let parent_state_id = parent_label_content.state_id; // State ID of the parent node
            let merged_t = parent_label_content.t.intersect(cur_t); // Intersect parent's T with the current state's T

            let goto = self.parser.stage_7_table[&parent_state_id].gotos[&nt];
            crate::debug!(4, "  Goto from state {} via NT {} to state {}", parent_state_id.0, nt.0, goto.0);

            let new_label = ParseStateNodeContent {
                state_id: goto,
                t: merged_t,
            };

            // Push the new state onto the parent stack node
            next_stacks.push(parent_node_arc.push(new_label));
        }
        next_stacks
    }


    /// Debug helper so the main `step` body stays short.
    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
        const MAX: usize = 30;
        // Log the single head node structure
        let roots = vec![self.head.clone()]; // GSS stats/print expects a vector of roots
        let stats = gather_gss_stats(&roots);
        crate::debug!(3, "{} - token {} ({:?}) - – head exists: {}, nodes: {:?}",
                      phase, token.0, self.parser.terminal_map.get_by_right(&token).unwrap().0, self.head.peek().next().is_some(), stats); // Check if head node has any incoming edges (is not an empty root)

        debug!(4, "{}", {
            if stats.unique_nodes <= MAX {
                format!("GSS ({} nodes):\n{}", stats.unique_nodes,
                        print_gss_forest(&roots, MAX))
            } else {
                // fall back to longest path printing
                match find_longest_path(&roots) {
                    Some(p) => format!("GSS too big ({} nodes). Longest path ({}): {}",
                                       stats.unique_nodes,
                                       p.len(),
                                       p.iter().map(|n| {
                                            // Need to show the state ID on the edge leading to this node.
                                            // This requires traversing back one step and getting the edge label.
                                            // This is complex with the current print_gss_forest and find_longest_path which just return nodes.
                                            // For now, just print the node address.
                                            // Or, ideally, the path should be (NodePtr, EdgeLabel, NodePtr, EdgeLabel, ...)
                                            // Let's modify find_longest_path to return Vec<(Arc<GSSNode<T>>, Option<T>)>, where Option<T> is the label on the edge LEADING to this node in the path.
                                            // For the root of the path, the label is None.

                                            // Temporary: Just print node address
                                            format!("{:p}", Arc::as_ptr(n))

                                            // Ideal (requires modified find_longest_path):
                                            // if let Some((node_arc, Some(label))) = find_longest_path returns this structure
                                            // format!("{:p}[{:?}]", Arc::as_ptr(node_arc), label.state_id)
                                       })
                                            .collect::<Vec<_>>()
                                            .join(" → ")) , // Join node addresses or states
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
            // After each step, check if the head is empty, meaning the parse failed.
            if self.head.peek().next().is_none() {
                 // If the head node has no incoming edges, the parse has failed for this path.
                 // In GLR, if there are other parser states (which there aren't with a single head),
                 // we'd check them. With a single head, if it becomes empty, the entire parse fails.
                 // However, the GLR step should handle failures by producing an empty `next_heads` list.
                 // The `self.head = next_heads` will make the head empty if all paths failed.
                 break; // Stop processing input if the head is empty
            }
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

    pub fn step(&mut self, token_id: TerminalID)
    where T: Clone + Eq + Hash // Bounds needed for popn and bulk_merge
    {
        /* ---------- logging & preparation ---------- */
        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        self.log_gss("Step-start", token_id);

        // Worklist now contains the current head node(s) that need processing.
        // With a single head, the worklist starts with its predecessors.
        let mut worklist: Vec<ParseStack<T>> = self.head.pop().into_iter().map(|(p, _)| p).collect(); // Process nodes one step down from the head
        let mut next_heads: Vec<ParseStack<T>> = Vec::new();
        let mut not_found_heads: Vec<ParseStack<T>> = Vec::new();

        // Clear the current head to build the new one from the results
        let old_head = std::mem::replace(&mut self.head, Arc::new(GSSNode::new()));


        /* ---------- core loop ---------- */
        // worklist contains nodes one step below the current head.
        // We are processing actions based on the state *of these nodes*.
        while let Some(current_node_arc) = worklist.pop() { // Process nodes from the worklist

            // The state and T value are on the incoming edge label *to* this current_node_arc.
            // We need to get the label on the edge from the parent *to* `current_node_arc`.
            // This means `current_node_arc` must have `old_head` as a predecessor, and the label is on that edge.
            // This suggests the worklist should be tuples of (node, label) representing edges leading to the current head.

            // Let's rethink the step loop with a single head.
            // The head node represents the *current set of possible parser states*.
            // Each edge leading *into* the head node represents a specific state reached.
            // So, we need to iterate over the edges leading *into* the current `self.head`.
            // For each edge `(pred_node, label)`, `pred_node` is the node *before* the current state, and `label` contains the `state_id` and `t` value for the current state.

            let mut processing_edges: Vec<(Arc<GSSNode<T>>, ParseStateNodeContent<T>)> = old_head.pop().into_iter().map(|(p, l_ref)| (p, l_ref.clone())).collect();
            let mut next_states_edges: Vec<(ParseStack<T>, ParseStateNodeContent<T>)> = Vec::new();
            let mut not_found_states_edges: Vec<(ParseStack<T>, ParseStateNodeContent<T>)> = Vec::new();


            while let Some((pred_node_arc, current_label)) = processing_edges.pop() { // Process (parent_node, current_state_label) tuples

                 let current_state_id = current_label.state_id;
                 let current_t = &current_label.t; // T value associated with the current state

                 let row     = &self.parser.stage_7_table[&current_state_id];

                 match row.shifts_and_reduces.get(&token_id) {
                     /* ------ 1. plain shift ------ */
                     Some(Stage7ShiftsAndReduces::Shift(to)) => {
                         crate::debug!(4, "Shift from state {} via token {} to state {}", current_state_id.0, token_id.0, to.0);
                         // Push new state onto the predecessor node (pred_node_arc)
                         let new_label_content = ParseStateNodeContent { state_id: *to, t: current_t.clone() };
                         let new_stack_head = pred_node_arc.push(new_label_content);
                         next_heads.push(new_stack_head); // Collect new head nodes
                     }

                     /* ------ 2. single reduce ------ */
                     Some(Stage7ShiftsAndReduces::Reduce{ nonterminal_id: nt,
                                                          len, .. }) => {
                         crate::debug!(4, "Reduce from state {} via token {} to nonterminal {}", current_state_id.0, token_id.0, nt.0);
                         // pop len, follow goto on nt, results are new potential head nodes
                         // The popn call should start from the predecessor node (pred_node_arc)
                         // popn(len) on `pred_node_arc` finds nodes `len` steps back from `pred_node_arc`.
                         // These are nodes `len + 1` steps back from the original `self.head`.
                         // No, popn(len) on the current node finds nodes `len` steps back.
                         // The state `current_state_id` was reached from `pred_node_arc` with label `current_label`.
                         // A reduction rule `A -> beta` where `|beta| = len` applies at `current_state_id`.
                         // We need to pop `len` symbols and their associated states/nodes.
                         // If `len == 0`, we apply the reduction directly from the current node's predecessor.
                         // If `len > 0`, we pop `len` nodes from the GSS stack starting from `pred_node_arc`.

                         let nodes_after_pop = pred_node_arc.popn(*len); // Returns Vec<(Arc<GSSNode<T>>, T)>
                         nodes_after_pop.bulk_merge(); // Merges paths that arrive at the same node N steps back

                         // For each node N steps back (after reduction), apply the goto.
                         // The goto applies from the state of the node N steps back.
                         // The label `T` in the `nodes_after_pop` tuple is the label on the edge into that node.
                         // This label contains the state_id and T value of the node N steps back.

                         for (node_len_steps_back_arc, label_len_steps_back) in nodes_after_pop {
                             let goto_from_state_id = label_len_steps_back.state_id;
                             let merged_t_for_goto = label_len_steps_back.t.intersect(current_t); // Intersect T from node N steps back with current state's T

                             let goto = self.parser.stage_7_table[&goto_from_state_id].gotos[&nt];
                             crate::debug!(4, "  Goto after reduce from state {} via NT {} to state {}", goto_from_state_id.0, nt.0, goto.0);

                             let new_label = ParseStateNodeContent { state_id: goto, t: merged_t_for_goto };
                             let new_stack_head = node_len_steps_back_arc.push(new_label);

                             // Add results to the worklist for potential further reductions in this step
                             // Or add to next_heads directly? GLR step should handle multiple actions at a state.
                             // The `todo` list in the original code was for processing states.
                             // Here, we are processing incoming edges to the head.
                             // If a reduce creates new potential heads, those should be processed in the *current* step if they can trigger further actions (like more reductions).
                             // This means adding the resulting stack heads back into `worklist` or a similar structure that feeds the main processing loop.

                             // Let's collect all results (shifts and reductions) into a temporary list
                             // and then iterate that list to build the `next_heads`.
                             // If a reduction leads to a state that can immediately reduce again, that needs to be handled.
                             // This suggests a recursive approach or a dedicated queue for reduction results.

                             // Simpler: collect all results (shift-results and reduce-results) into a single list `step_results`.
                             // Then, iterate `step_results`. If an item can reduce, add its reduction results to `step_results`. If it shifts, add to `next_heads`.

                             let mut step_results: Vec<ParseStack<T>> = Vec::new();
                             step_results.push(new_stack_head); // Add the result of this reduction

                             while let Some(result_stack) = step_results.pop() {
                                  // Check if this newly formed stack can perform immediate actions (reductions)
                                  if let Some(result_label_ref) = result_stack.peek().next() {
                                       let result_state_id = result_label_ref.state_id;
                                       let result_t = &result_label_ref.t;
                                       let result_row = &self.parser.stage_7_table[&result_state_id];

                                       // Check for actions on the current token from the new state
                                       match result_row.shifts_and_reduces.get(&token_id) {
                                            Some(Stage7ShiftsAndReduces::Reduce{ nonterminal_id: inner_nt, len: inner_len, .. }) => {
                                                 // Immediate reduction possible from this new state
                                                 crate::debug!(4, "  Immediate Reduce from state {} via token {} to nonterminal {}", result_state_id.0, token_id.0, inner_nt.0);
                                                 // Need to pop `inner_len` nodes from `result_stack` (which has result_state_id at its head)
                                                 let inner_nodes_after_pop = result_stack.popn(*inner_len);
                                                 inner_nodes_after_pop.bulk_merge();

                                                 for (inner_parent_node_arc, inner_parent_label) in inner_nodes_after_pop {
                                                     let inner_goto_from_state_id = inner_parent_label.state_id;
                                                     let inner_merged_t = inner_parent_label.t.intersect(result_t);

                                                     let inner_goto = self.parser.stage_7_table[&inner_goto_from_state_id].gotos[&inner_nt];
                                                     let inner_new_label = ParseStateNodeContent { state_id: inner_goto, t: inner_merged_t };
                                                     let inner_new_stack_head = inner_parent_node_arc.push(inner_new_label);

                                                     step_results.push(inner_new_stack_head); // Add result of inner reduction back to worklist
                                                 }
                                            }
                                            Some(Stage7ShiftsAndReduces::Split { reduces: inner_reduces, .. }) => {
                                                // Handle immediate reductions within a split
                                                crate::debug!(4, "  Immediate Split (reduces) from state {} via token {}", result_state_id.0, token_id.0);
                                                 for (inner_len, inner_nts) in inner_reduces {
                                                      for (inner_nt, _prod_ids) in inner_nts {
                                                           let inner_nodes_after_pop = result_stack.popn(*inner_len);
                                                           inner_nodes_after_pop.bulk_merge();
                                                            for (inner_parent_node_arc, inner_parent_label) in inner_nodes_after_pop {
                                                                let inner_goto_from_state_id = inner_parent_label.state_id;
                                                                let inner_merged_t = inner_parent_label.t.intersect(result_t);

                                                                let inner_goto = self.parser.stage_7_table[&inner_goto_from_state_id].gotos[&inner_nt];
                                                                let inner_new_label = ParseStateNodeContent { state_id: inner_goto, t: inner_merged_t };
                                                                let inner_new_stack_head = inner_parent_node_arc.push(inner_new_label);

                                                                step_results.push(inner_new_stack_head); // Add result of inner reduction back to worklist
                                                            }
                                                      }
                                                 }
                                            }
                                            Some(Stage7ShiftsAndReduces::Shift(_)) => {
                                                 // Cannot immediately shift after a reduction with the same token. This should be handled by the grammar/table construction.
                                                 // If it happens, maybe it's an error? Or should this stack head be added to next_heads?
                                                 // In standard GLR, a state processes all its actions for a token before moving to the next token.
                                                 // Immediate reductions happen within the same token step.
                                                 // Shifts (and reduce results that *don't* immediately reduce further) contribute to the states for the *next* token.
                                                 // So, if a state can shift OR reduce to a state that cannot immediately reduce, the resulting stack head goes to `next_heads`.
                                                 next_heads.push(result_stack); // Add to next_heads for the next token
                                            }
                                            None => {
                                                 // No immediate action from this new state with the current token.
                                                 // This means this path terminates for this token unless it was a shift result.
                                                 // Since it came from a reduce, it should be a state ready for the next token.
                                                 next_heads.push(result_stack); // Add to next_heads for the next token
                                            }
                                       }
                                  } else {
                                       // Should not happen for a valid stack head returned by push.
                                       crate::debug!(4, "Warning: Reduced to an empty stack head?");
                                  }
                             }
                         }
                     }

                     /* ------ 3. shift / reduce split ------ */
                     Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                         crate::debug!(4, "Split from state {} via token {}", current_state_id.0, token_id.0);
                         // Optional shift part
                         if let Some(to) = shift {
                             crate::debug!(4, " Shift from state {} via token {} to state {}", current_state_id.0, token_id.0, to.0);
                             let new_label_content = ParseStateNodeContent { state_id: *to, t: current_t.clone() };
                             let new_stack_head = pred_node_arc.push(new_label_content);
                             next_heads.push(new_stack_head); // Shift results go to next_heads
                         }
                         // Every reduce alternative
                         for (len, nts) in reduces {
                             crate::debug!(4, " Reduce from state {} via token {} (len {})", current_state_id.0, token_id.0, len);
                             let nodes_after_pop = pred_node_arc.popn(*len);
                             nodes_after_pop.bulk_merge();

                             for (node_len_steps_back_arc, label_len_steps_back) in nodes_after_pop {
                                 let goto_from_state_id = label_len_steps_back.state_id;
                                 let merged_t_for_goto = label_len_steps_back.t.intersect(current_t);

                                 for (nt, _prod_ids) in nts { // we ignore prod-ids here
                                     let goto = self.parser.stage_7_table[&goto_from_state_id].gotos[&nt];
                                     crate::debug!(4, "  Goto after split/reduce from state {} via NT {} to state {}", goto_from_state_id.0, nt.0, goto.0);
                                     let new_label = ParseStateNodeContent { state_id: goto, t: merged_t_for_goto.clone() }; // Clone merged_t if multiple NTs from same base
                                     let new_stack_head = node_len_steps_back_arc.push(new_label);

                                     // Add reduce results to a temporary list for immediate processing
                                     let mut step_results: Vec<ParseStack<T>> = Vec::new();
                                     step_results.push(new_stack_head); // Add result of this reduction

                                     while let Some(result_stack) = step_results.pop() {
                                          // Check for immediate reductions from this new state
                                          if let Some(result_label_ref) = result_stack.peek().next() {
                                              let result_state_id = result_label_ref.state_id;
                                              let result_t = &result_label_ref.t;
                                              let result_row = &self.parser.stage_7_table[&result_state_id];

                                              match result_row.shifts_and_reduces.get(&token_id) {
                                                  Some(Stage7ShiftsAndReduces::Reduce{ nonterminal_id: inner_nt, len: inner_len, .. }) => {
                                                      // Immediate reduction possible
                                                      crate::debug!(4, "  Immediate Reduce (from split) from state {} via token {} to nonterminal {}", result_state_id.0, token_id.0, inner_nt.0);
                                                      let inner_nodes_after_pop = result_stack.popn(*inner_len);
                                                      inner_nodes_after_pop.bulk_merge();
                                                       for (inner_parent_node_arc, inner_parent_label) in inner_nodes_after_pop {
                                                           let inner_goto_from_state_id = inner_parent_label.state_id;
                                                           let inner_merged_t = inner_parent_label.t.intersect(result_t);

                                                           let inner_goto = self.parser.stage_7_table[&inner_goto_from_state_id].gotos[&inner_nt];
                                                           let inner_new_label = ParseStateNodeContent { state_id: inner_goto, t: inner_merged_t };
                                                           let inner_new_stack_head = inner_parent_node_arc.push(inner_new_label);

                                                           step_results.push(inner_new_stack_head); // Add result of inner reduction back to worklist
                                                       }
                                                  }
                                                  Some(Stage7ShiftsAndReduces::Split { reduces: inner_reduces, .. }) => {
                                                      // Handle immediate reductions within a nested split
                                                      crate::debug!(4, "  Immediate Nested Split (reduces) from state {} via token {}", result_state_id.0, token_id.0);
                                                       for (inner_len, inner_nts) in inner_reduces {
                                                            for (inner_nt, _prod_ids) in inner_nts {
                                                                 let inner_nodes_after_pop = result_stack.popn(*inner_len);
                                                                 inner_nodes_after_pop.bulk_merge();
                                                                  for (inner_parent_node_arc, inner_parent_label) in inner_nodes_after_pop {
                                                                      let inner_goto_from_state_id = inner_parent_label.state_id;
                                                                      let inner_merged_t = inner_parent_label.t.intersect(result_t);

                                                                      let inner_goto = self.parser.stage_7_table[&inner_goto_from_state_id].gotos[&inner_nt];
                                                                      let inner_new_label = ParseStateNodeContent { state_id: inner_goto, t: inner_merged_t };
                                                                      let inner_new_stack_head = inner_parent_node_arc.push(inner_new_label);

                                                                      step_results.push(inner_new_stack_head); // Add result of inner reduction back to worklist
                                                                  }
                                                            }
                                                       }
                                                  }
                                                  Some(Stage7ShiftsAndReduces::Shift(_)) => {
                                                      // Shift result from a reduction -> add to next_heads
                                                      next_heads.push(result_stack);
                                                  }
                                                  None => {
                                                      // No immediate action -> add to next_heads
                                                      next_heads.push(result_stack);
                                                  }
                                              }
                                          }
                                     }
                                 }
                             }
                         }
                     }

                     /* ------ 4. no action ------ */
                     None => {
                         crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, current_state_id.0);
                         // This path (represented by pred_node_arc + current_label) terminates for this token.
                         // We could collect these into `not_found`, but the current design drops them.
                         // not_found_heads.push(pred_node_arc.push(current_label)); // Keep the state that failed
                     },
                 }
            }


        /* ---------- finish up ---------- */
        // Merge the collected next_heads into a single new head node.
        // Create a new root node to push results onto.
        let new_root = Arc::new(GSSNode::new());
        for stack_head in next_heads {
             // Each stack_head is an Arc<GSSNode<T>> representing a potential head.
             // We want the edges leading into these potential heads.
             // We'll merge these edges into the `new_root`.
             // This is conceptually wrong. The new head should represent the union of states reached.
             // Each stack_head from `next_heads` represents a distinct valid path reaching a state for the next token.
             // The new head should be a node whose predecessors are the union of the `stack_head` arcs, with appropriate labels.

             // If next_heads is [H1, H2, H3], the new head N should have H1, H2, H3 as predecessors.
             // N { predecessors: { Edge{pred: H1, label: ?}, Edge{pred: H2, label: ?}, Edge{pred: H3, label: ?} } }
             // This seems incorrect. The head node should be the *single* node representing the set of states.
             // Each incoming edge *to* the head represents one specific state.
             // So, the new head node `self.head` will have edges representing the states reached after processing the token.

             // Let's collect the edges that should lead *into* the *new* head.
             // If a shift goes from state S to S', the new state is S'. The previous node was `pred_node_arc`.
             // The new head should have an edge from `pred_node_arc` with label `ParseStateNodeContent{state_id: S', t: merged_t}`.

             // Let's reconstruct `next_heads` as a list of edges for the *new* head node.
             // `next_states_edges` collected tuples `(stack_head_arc, label)`.
             // The `stack_head_arc` was the node *before* the state represented by `label`.
             // The `label` is the state reached.
             // So, an edge `stack_head_arc --label--> new_head` is formed.

             // Collect edges for the new head:
             let mut edges_for_new_head: BTreeSet<GSSEdge<ParseStateNodeContent<T>>> = BTreeSet::new();

             // Re-process `next_heads` to get the edges
             for stack_head in next_heads {
                  // Each stack_head was the result of a push: `parent_arc.push(label_content)`
                  // We need the parent_arc and the label_content from this push.
                  // `stack_head` is an Arc<GSSNode<T>>. Its predecessors are the parent_arc.
                  // And the label on the edge is the label_content.
                  for edge in &stack_head.predecessors {
                       // edge.pred is the parent_arc, edge.label is the label_content
                       edges_for_new_head.insert(edge.clone()); // Clone the edge (ArcPtrWrapper and Label)
                  }
             }

             // Create the new head node with these edges
             let new_head_node = Arc::new(GSSNode { predecessors: edges_for_new_head });
             self.head = new_head_node;


        // The not_found states were paths that terminated. We are dropping them.
        // self.not_found = not_found_heads; // This would collect them into a single not_found head node?

        crate::debug!(4, "----------------------------------------------------------------");
        self.log_gss("Step-end", token_id);

        // The current design is to drop not_found states per step.
        // self.not_found.clear(); // How to clear a GSS node? Replace with empty root?
        // self.not_found = Arc::new(GSSNode::new()); // Replace with an empty node

    }

    /// Merging is now handled implicitly when edges are added to the `new_head` in the `step` method
    /// and by `BulkMerge` during reduction processing.
    /// This method is likely no longer needed.
    pub fn merge_active_states(&mut self) {
        // This method is no longer necessary. Merging is done during step processing.
    }

    /// Merge another GLRParserState into this one.
    /// This means merging the head nodes.
    pub fn merge_with(&mut self, other: GLRParserState<T>)
    where T: Clone + Ord // Bounds needed for BTreeSet merge
    {
        assert!(std::ptr::eq(self.parser, other.parser));

        // Merge the predecessors of `other.head` into `self.head`.
        // Create a new head node that has the union of predecessors from self.head and other.head.
        let mut merged_predecessors = self.head.predecessors.clone();
        merged_predecessors.extend(other.head.predecessors.clone());

        // Create the new merged head node
        self.head = Arc::new(GSSNode { predecessors: merged_predecessors });

        // Merge not_found nodes similarly if they are kept
        let mut merged_not_found_predecessors = self.not_found.predecessors.clone();
        merged_not_found_predecessors.extend(other.not_found.predecessors.clone());
        self.not_found = Arc::new(GSSNode { predecessors: merged_not_found_predecessors });
    }

    pub fn is_ok(&self) -> bool {
        // A parser state is OK if its head node has at least one incoming edge,
        // representing at least one active parser state path.
        !self.head.predecessors.is_empty()
    }
}

// ParseStateKey and ParseState struct are removed.
// InsertWith trait and impl for BTreeMap are removed.

