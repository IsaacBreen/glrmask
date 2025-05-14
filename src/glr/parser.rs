use crate::datastructures::gss::{print_gss_forest, BulkMerge, gather_gss_stats, find_longest_path, GSSEdge};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{
    NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID,
};
use crate::datastructures::gss::{GSSNode, GSSTrait, GSSStats}; // GSSNode will be GSSNode<V>

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

// ParseStateNodeContent is removed as StateID is in GSSNode and T is on the edge / in ParseState.

// Represents an active state in the GLR parser.
// It consists of the top GSS node of a stack and the 'T' value accumulated for the path ending at this node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState<T: MergeAndIntersect> {
    // The GSS node that is at the "top" of this conceptual stack.
    pub gss_node: Arc<GSSNode<T>>,
    // The 'T' value associated with the path ending at gss_node.
    // This is the value from the edge leading to gss_node, or an initial value for a root.
    pub t_value: T,
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
        let initial_parse_state = self.init_parse_state_with_t(t);
        let mut active_states_map = BTreeMap::new();
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

    pub fn init_parse_state_with_t<T: MergeAndIntersect>(&self, t: T) -> ParseState<T> {
        ParseState {
            gss_node: Arc::new(GSSNode::new(self.start_state_id)),
            t_value: t,
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
                    Stage7ShiftsAndReduces::Reduce { production_id: _production_id, nonterminal_id: nonterminal, len } => {
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
    // The `head` field change is deferred. For now, active_states uses the new ParseState<T>.
    pub active_states: BTreeMap<ParseStateKey, ParseState<T>>,
    pub action_not_found_states: BTreeMap<ParseStateKey, ParseState<T>>,
}

impl<'a, T: MergeAndIntersect + Debug> GLRParserState<'a, T> {
    /* -------------------------------------------------
     * Helper utilities to make `step` compact and clear
     * ------------------------------------------------- */

    /// Push a new state onto a GSS path.
    /// `current_gss_node` is the top node of the current path.
    /// `new_state_id` is the StateID for the new GSS node.
    /// `new_edge_t_value` is the T value for the edge leading to the new GSS node.
    fn push_state(
        &self,
        current_gss_node: &Arc<GSSNode<T>>,
        new_node_state_id: StateID,
        new_edge_t_value: T,
    ) -> ParseState<T> {
        let new_gss_node = Arc::new(current_gss_node.push(new_node_state_id, new_edge_t_value.clone()));
        ParseState {
            gss_node: new_gss_node,
            t_value: new_edge_t_value,
        }
    }

    /// Pop `len` edges/nodes, follow the goto on `nt`, and return the resulting ParseStates.
    /// `current_gss_node` is the node from which popping occurs.
    /// `current_t_value` is the T value associated with the path ending at `current_gss_node`.
    fn pop_and_goto(
        &self,
        current_gss_node: &Arc<GSSNode<T>>,
        _current_t_value: &T, // current_t_value is associated with the node, not directly used for pop's T value logic
        len: usize,
        nt: NonTerminalID,
    ) -> Vec<ParseState<T>> {
        // popn returns Vec<(Arc<GSSNode<T>>, V)> where V is the value of the edge *leading to* that GSSNode
        let mut parent_paths = current_gss_node.popn(len);
        parent_paths.bulk_merge(); // BulkMerge needs to be adapted for Vec<(Arc<GSSNode<T>>, T)>

        let mut out_parse_states = Vec::new();

        for (parent_node, _parent_edge_t_value) in parent_paths { // parent_edge_t_value is for edge to parent_node
            let goto_target_state_id = self.parser.stage_7_table[&parent_node.state_id].gotos[&nt];
            
            // The T value for the new edge (parent_node -> goto_node) needs to be determined.
            // This involves intersecting the T value of the path ending at parent_node
            // with the T value of the path that just got reduced (current_t_value).
            // However, parent_edge_t_value is for the edge *to* parent_node.
            // We need the t_value of the *path ending at parent_node*.
            // This requires popn to return not just (node, edge_val_to_node) but (node, path_val_at_node).
            // This is a deeper GSS change. For now, let's assume a simplification:
            // The new edge's T value is based on the T value of the path being reduced.
            // This is a common simplification but might lose precision if path T values differ.
            // A more accurate approach would be to use parent_edge_t_value if len=0,
            // or the T from the reduction if len > 0.
            // The original code used top.t (which is current_t_value here) and intersected it.
            // The `parent_node.t` (from old code `top.t`) is `parent_edge_t_value` here.
            // So, we need to intersect `parent_edge_t_value` with `_current_t_value`.
            // This implies `popn` must return the T value *for the path ending at that node*.

            // Let's assume popn returns Vec<ParseState<T>> directly for now, to simplify.
            // This means GSSNode.popn itself would need to track the T value for the path.
            // This is not how GSS is structured with T on edge.

            // Correct approach:
            // popn returns Vec<(Arc<GSSNode<T>>, T_edge_val_to_that_node)>
            // The T value for the path ending at `parent_node` is `parent_edge_t_value`.
            // The T value for the reduction is `_current_t_value`.
            // The new edge from `parent_node` to `goto_node` should have a T value derived from these.
            // The original code: merged_t = top.t.intersect(cur_t);
            // top.t was the T value of the node *before* the goto (parent_node here).
            // So, it's parent_edge_t_value.intersect(_current_t_value).

            // This means popn needs to return (Arc<GSSNode<T>>, T_path_value_at_node)
            // For now, let's assume `popn` gives us `Vec<ParseState<T>>` representing the states after popping.
            // This is a temporary simplification to get the parser structure right.
            // The GSS `popn` will need careful implementation.

            // Re-simplifying: popn returns Vec<Arc<GSSNode<T>>> (nodes after pop)
            // The T value for the edge leading to *these* popped nodes is what GSS.popn would give.
            // Let's assume GSS.popn returns Vec<(Arc<GSSNode<T>>, T_val_for_edge_to_it)>
            
            // Let's stick to the current GSS popn signature: Vec<Arc<GSSEdge<T>>>
            // This means each item is (value_on_edge_to_pred, pred_node_itself)
            // If len=1, popn returns edges to current_gss_node's direct predecessors.
            // Each edge has a .value (T) and .predecessor_node (Arc<GSSNode<T>>)
            // The `parents` from `popn(len)` are the nodes *after* popping `len` items.
            // The `t` value associated with these `parents` is the `t` from the edge that led to them.

            let new_edge_t = _current_t_value.clone(); // Simplified: T of reduced path becomes T of new edge.
                                                       // A more complex merge might be needed if paths carry distinct T values.
                                                       // The original code did `top.t.intersect(cur_t)`.
                                                       // `top.t` is the T value of the node that `nt` goes from.
                                                       // This is `parent_node.t_value`.
                                                       // So, if popn returns `Vec<ParseState<T>>`, then for each `p_state` in that:
                                                       // `merged_t = p_state.t_value.intersect(_current_t_value)`.

            // Assuming popn returns Vec<ParseState<T>> where t_value is path t_value at that node.
            // This is a placeholder for now.
            let popped_states: Vec<ParseState<T>> = current_gss_node.popn_to_parse_states(len);


            for p_state in popped_states {
                let parent_node_for_goto = p_state.gss_node;
                let t_at_parent_node_for_goto = p_state.t_value;

                if let Some(goto_state_id) = self.parser.stage_7_table[&parent_node_for_goto.state_id].gotos.get(&nt) {
                    let merged_t = t_at_parent_node_for_goto.intersect(_current_t_value);
                    crate::debug!(4, "  Goto from state {} to state {}", parent_node_for_goto.state_id.0, goto_state_id.0);
                    
                    let new_gss_node_after_goto = Arc::new(parent_node_for_goto.push(*goto_state_id, merged_t.clone()));
                    out_parse_states.push(ParseState {
                        gss_node: new_gss_node_after_goto,
                        t_value: merged_t,
                    });
                } else {
                    // Handle GotoNotFound if necessary, though table generation should ensure this doesn't happen for valid reductions.
                }
            }
        }
        out_parse_states
    }


    /// Debug helper so the main `step` body stays short.
    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
        const MAX: usize = 30;
        // We need to collect GSS roots and their T values to pass to print_gss_forest if its signature changes.
        // For now, assuming print_gss_forest takes Vec<Arc<GSSNode<T>>>.
        let roots: Vec<_> = self.active_states.values().map(|s| s.gss_node.clone()).collect();
        let stats = gather_gss_stats(&roots); // gather_gss_stats needs to be GSSNode<V> aware
        crate::debug!(3, "{} - token {} ({:?}) - – active: {}, nodes: {:?}",
                      phase, token.0, self.parser.terminal_map.get_by_right(&token).unwrap().0, self.active_states.len(), stats);

        debug!(4, "{}", {
            if stats.unique_nodes <= MAX {
                // print_gss_forest needs to be GSSNode<V> aware
                format!("GSS ({} nodes):\n{}", stats.unique_nodes,
                        print_gss_forest(&roots, MAX)) 
            } else {
                // find_longest_path needs to be GSSNode<V> aware
                match find_longest_path(&roots) {
                    Some(p) => format!("GSS too big ({} nodes). Longest path ({}): {}",
                                       stats.unique_nodes,
                                       p.len(),
                                       p.iter().map(|n_arc| n_arc.state_id.0) // n_arc is Arc<GSSNode<T>>
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

        let mut todo = std::mem::take(&mut self.active_states).into_values().collect::<Vec<_>>();
        let mut next_active_states = BTreeMap::<ParseStateKey, ParseState<T>>::new();
        let mut not_found_states = BTreeMap::<ParseStateKey, ParseState<T>>::new();

        /* ---------- core loop ---------- */
        while let Some(current_parse_state) = todo.pop() { // Process states from the worklist
            let current_gss_node = &current_parse_state.gss_node;
            let current_t_value  = &current_parse_state.t_value;
            let top_state_id     = current_gss_node.state_id; // Assuming GSSNode has state_id directly
            let row              = &self.parser.stage_7_table[&top_state_id];

            match row.shifts_and_reduces.get(&token_id) {
                /* ------ 1. plain shift ------ */
                Some(Stage7ShiftsAndReduces::Shift(to_state_id)) => {
                    crate::debug!(4, "Shift from state {} via token {} to state {}", top_state_id.0, token_id.0, to_state_id.0);
                    // For a shift, the T value of the new edge is typically the T value of the current path.
                    let new_parse_state = self.push_state(current_gss_node, *to_state_id, current_t_value.clone());
                    next_active_states.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                }

                /* ------ 2. single reduce ------ */
                Some(Stage7ShiftsAndReduces::Reduce{ nonterminal_id: nt, len, .. }) => {
                    crate::debug!(4, "Reduce from state {} via token {} to nonterminal {}", top_state_id.0, token_id.0, nt.0);
                    for new_parse_state_after_goto in self.pop_and_goto(current_gss_node, current_t_value, *len, *nt) {
                        // Add to worklist for current step; merging happens when moving to `next_active_states`
                        // Reductions can lead to states that need further processing in the same step (epsilon transitions essentially)
                        todo.push(new_parse_state_after_goto); 
                    }
                }

                /* ------ 3. shift / reduce split ------ */
                Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                    crate::debug!(4, "Split from state {} via token {}", top_state_id.0, token_id.0);
                    // optional shift part
                    if let Some(to_state_id) = shift {
                        crate::debug!(4, " Shift from state {} via token {} to state {}", top_state_id.0, token_id.0, to_state_id.0);
                        let new_parse_state = self.push_state(current_gss_node, *to_state_id, current_t_value.clone());
                        next_active_states.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                    }
                    // every reduce alternative
                    for (len, nts) in reduces {
                        crate::debug!(4, " Reduce from state {} via token {} to nonterminals {:?}", top_state_id.0, token_id.0, nts);
                        for (nt, _prod_ids) in nts {        // we ignore prod-ids here
                            for new_parse_state_after_goto in self.pop_and_goto(current_gss_node, current_t_value, *len, *nt) {
                                todo.push(new_parse_state_after_goto);
                            }
                        }
                    }
                }

                /* ------ 4. no action ------ */
                None => {
                    crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, top_state_id.0);
                    // current_parse_state is the state that had no action.
                    not_found_states.insert_with(current_parse_state.key(), current_parse_state, |existing, new_s| existing.merge(new_s));
                },
            }
        }

        /* ---------- finish up ---------- */
        self.active_states            = next_active_states;
        self.action_not_found_states  = not_found_states;   // keep for caller if wanted

        self.log_gss("Step-end", token_id);
        self.action_not_found_states.clear();        // current design: we drop them

        crate::debug!(4, "----------------------------------------------------------------");
    }
    
    pub fn merge_active_states(&mut self) {
        // This method is no longer necessary as merging is done on insertion into BTreeMap.
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    // The StateID of the GSS node at the top of the stack.
    // If T were part of the key, it would need to be Hashable. T is already Hash.
    // For now, keying only by StateID. Merging of T values happens in ParseState::merge.
    stack_node_state_id: StateID,
}

impl<T: MergeAndIntersect> ParseState<T> {
    pub fn key(&self) -> ParseStateKey {
        ParseStateKey {
            stack_node_state_id: self.gss_node.state_id, // Assuming GSSNode has state_id
        }
    }

    /// Merges `other` into `self`. Assumes `self.key() == other.key()`.
    /// Merges the GSS structures and combines the `t_value` using `MergeAndIntersect::merge`.
    pub fn merge(&mut self, other: ParseState<T>) {
        assert_eq!(self.key(), other.key());
        assert_eq!(self.gss_node.state_id, other.gss_node.state_id);

        // Combine 't_value'
        self.t_value = self.t_value.merge(&other.t_value);

        // Merge GSS nodes. This requires GSSNode to have a merge method.
        // If self.gss_node and other.gss_node are different Arcs but point to logically
        // equivalent nodes (same state_id), we need to merge their predecessor sets.
        // Arc::make_mut will clone if shared.
        if Arc::ptr_eq(&self.gss_node, &other.gss_node) {
            // Same GSS node Arc, t_value already merged. Nothing more to do for GSS.
            return;
        }
        
        // Different Arcs, potentially different predecessor sets.
        // We need a way to merge the GSS structures pointed to by self.gss_node and other.gss_node.
        // This typically involves making one mutable and absorbing the predecessors of the other.
        // This is complex if they are deep structures.
        // For now, we assume that if keys are same, the GSS nodes should be merged.
        // GSSNode::merge_into(&mut self.gss_node, other.gss_node) would be ideal.
        // Or, GSSNode::merge_predecessors_from(Arc::make_mut(&mut self.gss_node), other.gss_node);
        
        // Simplified: If state_ids are the same, we assume the GSS nodes should be merged.
        // The BTreeMap insert_with handles merging ParseState objects.
        // If two ParseState objects have the same key, this merge function is called.
        // We need to ensure their GSS representations are also merged.
        // This means GSSNode needs a way to merge `other.gss_node` into `self.gss_node`.
        // This is effectively what GSSNode::merge_unchecked did, but on GSSNode itself.
        // Now it needs to happen on Arc<GSSNode<T>>.

        // Let current GSSNode::merge take other: Arc<GSSNode<T>>
        // Arc::make_mut(&mut self.gss_node).merge(other.gss_node); // This would consume other.gss_node
        
        // For now, this merge focuses on t_value. GSS merging is handled by bulk_merge on pop results
        // and by the fact that new nodes are created via push, naturally sharing common history.
        // If two distinct ParseStates arrive at the same StateID via different immediate histories
        // but should be considered the "same" GSS head, their GSSNode Arcs might be different.
        // The BTreeMap merging based on ParseStateKey(StateID) means we pick one GSSNode Arc
        // and merge T values. If the GSSNode Arcs are truly different but represent the same logical
        // state, this might lead to one path's history being preferred/kept.
        // This is a classic GSS problem: ensuring maximal sharing.
        // The current GSS `popn().bulk_merge()` helps with this for common ancestors.
        // For heads, if multiple paths lead to the same StateID with different GSSNode Arcs,
        // `insert_with` will call this merge. We need to merge the GSS structures.
        // This requires `GSSNode` to have a method like `absorb_predecessors_from(other_node_arc)`.
        let mut self_node_mut = Arc::make_mut(&mut self.gss_node);
        self_node_mut.absorb_predecessors_from(&other.gss_node);

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

