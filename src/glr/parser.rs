use crate::datastructures::gss::{print_gss_forest, BulkMerge, gather_gss_stats, find_longest_path, GSSNode, PredecessorLink, make_successor_node}; // Import GSSNode, PredecessorLink, make_successor_node
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{
    NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID,
};
use crate::datastructures::gss::{GSSStats}; // GSSTrait likely needs removing or redesign, keep for now but expect issues

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;
use crate::debug;
use crate::datastructures::ArcPtrWrapper;

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

// ParseStateNodeContent is removed as value is now on the edge

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState<T: MergeAndIntersect + Ord + Clone + Debug + Hash> { // T needs bounds for GSSNode and PredecessorLink
    pub stack_top_node: Arc<GSSNode<T>>, // Pointer to the top node of this stack path
    pub top_edge_value: T,             // The value on the edge leading TO stack_top_node
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StopReason {
    ActionNotFound,
    GotoNotFound,
}

// Define a dummy state ID for the head node of the GSS forest representing active states
pub(crate) const DUMMY_HEAD_STATE_ID: StateID = StateID(usize::MAX);

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

    pub fn init_glr_parser<T: MergeAndIntersect + Default + Ord + Clone + Debug + Hash>(&self) -> GLRParserState<T> {
        self.init_glr_parser_with_t(T::default())
    }

    pub fn init_glr_parser_with_t<T: MergeAndIntersect + Ord + Clone + Debug + Hash>(&self, t: T) -> GLRParserState<T> {
        let initial_parse_state = self.init_parse_state_with_t(t);
        let initial_link = PredecessorLink {
            node: ArcPtrWrapper::new(initial_parse_state.stack_top_node),
            edge_value: initial_parse_state.top_edge_value,
        };
        let head_node = GSSNode::new_with_predecessors(DUMMY_HEAD_STATE_ID, vec![initial_link]);

        GLRParserState {
            parser: self,
            head: Arc::new(head_node),
            action_not_found_states: BTreeMap::new(), // Remains BTreeMap for now
        }
    }

    pub fn init_glr_parser_from_parse_state<T: MergeAndIntersect + Ord + Clone + Debug + Hash>(&self, parse_state: ParseState<T>) -> GLRParserState<T> {
        let initial_link = PredecessorLink {
            node: ArcPtrWrapper::new(parse_state.stack_top_node),
            edge_value: parse_state.top_edge_value,
        };
        let head_node = GSSNode::new_with_predecessors(DUMMY_HEAD_STATE_ID, vec![initial_link]);

        GLRParserState {
            parser: self,
            head: Arc::new(head_node),
            action_not_found_states: BTreeMap::new(),
        }
    }

    pub fn init_glr_parser_from_parse_states<T: MergeAndIntersect + Ord + Clone + Debug + Hash>(
        &self,
        parse_states: Vec<ParseState<T>>,
    ) -> GLRParserState<T> {
        let mut active_states_map = BTreeMap::new();
        for state in parse_states {
             active_states_map.insert_with(state.key(), state, |existing, new_s| existing.merge(new_s));
        }

        let mut head_links = BTreeSet::new();
        for (_key, parse_state) in active_states_map {
            head_links.insert(PredecessorLink {
                node: ArcPtrWrapper::new(parse_state.stack_top_node),
                edge_value: parse_state.top_edge_value,
            });
        }
        let head_node = GSSNode::new_with_predecessors(DUMMY_HEAD_STATE_ID, head_links.into_iter().collect());

        GLRParserState {
            parser: self,
            head: Arc::new(head_node),
            action_not_found_states: BTreeMap::new(),
        }
    }


    pub fn init_parse_state<T: MergeAndIntersect + Default + Ord + Clone + Debug + Hash>(&self) -> ParseState<T> {
        self.init_parse_state_with_t(T::default())
    }

    pub fn init_parse_state_with_t<T: MergeAndIntersect + Ord + Clone + Debug + Hash>(&self, t: T) -> ParseState<T> {
        let initial_stack_top_node = Arc::new(GSSNode::new(self.start_state_id));
        ParseState {
            stack_top_node: initial_stack_top_node,
            top_edge_value: t,
        }
    }

    pub fn parse<T: MergeAndIntersect + Default + Ord + Clone + Debug + Hash>(&self, input: &[TerminalID]) -> GLRParserState<T> {
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
pub struct GLRParserState<'a, T: MergeAndIntersect + Ord + Clone + Debug + Hash> { // T needs bounds
    pub parser: &'a GLRParser,
    // The head node's predecessors represent the set of active ParseStates.
    // Each PredecessorLink contains the stack_top_node and the top_edge_value.
    pub head: Arc<GSSNode<T>>,
    // Keep action_not_found_states as BTreeMap for now.
    pub action_not_found_states: BTreeMap<ParseStateKey, ParseState<T>>,
}

impl<'a, T: MergeAndIntersect + Debug + Ord + Clone + Hash> GLRParserState<'a, T> {
    /* -------------------------------------------------
     * Helper utilities to make `step` compact and clear
     * ------------------------------------------------- */

    /// Push a new state on `stack_top_node` and return the new `ParseState`.
    fn push_state(
        &self,
        current_top_node: &Arc<GSSNode<T>>,
        new_node_state_id: StateID,
        new_edge_value: T,
    ) -> ParseState<T> {
        let new_gss_node_obj = GSSNode::make_successor_node(current_top_node.clone(), new_node_state_id, new_edge_value.clone());
        ParseState {
            stack_top_node: Arc::new(new_gss_node_obj),
            top_edge_value: new_edge_value,
        }
    }

    /// Pop `len` nodes from `stack_top_node`, follow the goto on `nt`, and return the resulting `ParseState`s.
    fn pop_and_goto(
        &self,
        stack_top_node: &Arc<GSSNode<T>>,
        len: usize,
        nt: NonTerminalID,
        current_top_edge_value: &T,
    ) -> Vec<ParseState<T>> {
        let mut parents_and_edges = stack_top_node.popn(len); // Returns Vec<(Arc<GSSNode<T>>, T_edge_to_it)>
        parents_and_edges.bulk_merge(); // BulkMerge on Vec<(Arc<GSSNode<T>>, T)>

        let mut out_parse_states = Vec::new();

        for (parent_node, edge_val_to_parent) in parents_and_edges {
            let goto_target_state_id = self.parser.stage_7_table[&parent_node.state_id].gotos[&nt];

            // The value on the new edge is the intersection of the edge value to the parent node
            // and the edge value that led to the current node being reduced.
            let new_edge_val_for_goto = edge_val_to_parent.intersect(current_top_edge_value);

            crate::debug!(4, "  Goto from state {} to state {}", parent_node.state_id.0, goto_target_state_id.0);

            let new_successor_obj = GSSNode::make_successor_node(parent_node, goto_target_state_id, new_edge_val_for_goto.clone());
            out_parse_states.push(ParseState {
                stack_top_node: Arc::new(new_successor_obj),
                top_edge_value: new_edge_val_for_goto,
            });
        }
        out_parse_states
    }

    /// Debug helper so the main `step` body stays short.
    pub(crate) fn log_gss(&self, phase: &str, token: TerminalID) {
        const MAX: usize = 30;
        // Get the stack top nodes from the predecessors of the head node
        let roots: Vec<_> = self.head.predecessors.iter()
            .map(|link| link.node.as_arc().clone())
            .collect();

        let stats = gather_gss_stats(&roots);
        crate::debug!(3, "{} - token {} ({:?}) - – active: {}, nodes: {:?}",
                      phase, token.0, self.parser.terminal_map.get_by_right(&token).unwrap().0, self.head.predecessors.len(), stats); // Use head.predecessors.len() for active states

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
                                       // Use state_id directly
                                       p.iter().map(|n| n.state_id.0)
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

        // Extract ParseStates from the head node's predecessors
        let current_head_node = std::mem::replace(
            &mut self.head,
            Arc::new(GSSNode::new(DUMMY_HEAD_STATE_ID)), // Replace with a temporary empty head
        );
        let mut todo: Vec<ParseState<T>> = current_head_node.predecessors.iter().map(|link| {
            ParseState {
                stack_top_node: link.node.as_arc().clone(),
                top_edge_value: link.edge_value.clone(),
            }
        }).collect();

        let mut next = BTreeMap::<ParseStateKey, ParseState<T>>::new();
        let mut not_found = BTreeMap::<ParseStateKey, ParseState<T>>::new();

        /* ---------- core loop ---------- */
        while let Some(state) = todo.pop() { // Process states from the worklist
            let stack_top_node = state.stack_top_node;
            let top_state_id   = stack_top_node.state_id;
            let top_edge_value = &state.top_edge_value;

            let row = &self.parser.stage_7_table[&top_state_id];

            match row.shifts_and_reduces.get(&token_id) {
                /* ------ 1. plain shift ------ */
                Some(Stage7ShiftsAndReduces::Shift(to)) => {
                    crate::debug!(4, "Shift from state {} via token {} to state {}", top_state_id.0, token_id.0, to.0);
                    let new_parse_state = self.push_state(&stack_top_node, *to, top_edge_value.clone());
                    next.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                }

                /* ------ 2. single reduce ------ */
                Some(Stage7ShiftsAndReduces::Reduce{ nonterminal_id: nt,
                                                     len, .. }) => {
                    crate::debug!(4, "Reduce from state {} via token {} to nonterminal {}", top_state_id.0, token_id.0, nt.0);
                    for s in self.pop_and_goto(&stack_top_node, *len, *nt, top_edge_value) {
                        // Add to worklist for current step; merging happens when moving to `next`
                        todo.push(s);
                    }
                }

                /* ------ 3. shift / reduce split ------ */
                Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                    crate::debug!(4, "Split from state {} via token {}", top_state_id.0, token_id.0);
                    // optional shift part
                    if let Some(to) = shift {
                        crate::debug!(4, " Shift from state {} via token {} to state {}", top_state_id.0, token_id.0, to.0);
                        let new_parse_state = self.push_state(&stack_top_node, *to, top_edge_value.clone());
                        next.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                    }
                    // every reduce alternative
                    for (len, nts) in reduces {
                        crate::debug!(4, " Reduce from state {} via token {} to nonterminals {:?}", top_state_id.0, token_id.0, nts);
                        for (nt, _prod_ids) in nts {        // we ignore prod-ids here
                            for s in self.pop_and_goto(&stack_top_node, *len, *nt, top_edge_value) {
                                // Add to worklist for current step
                                todo.push(s);
                            }
                        }
                    }
                }

                /* ------ 4. no action ------ */
                None => {
                    crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, top_state_id.0);
                    // The ParseState 'state' is already the one that had no action
                    not_found.insert_with(state.key(), state, |existing, new_s| existing.merge(new_s));
                },
            }
        }

        /* ---------- finish up ---------- */
        // Rebuild the head node from the `next` map
        let mut new_head_links = BTreeSet::new();
        for (_key, parse_state) in next {
            new_head_links.insert(PredecessorLink {
                node: ArcPtrWrapper::new(parse_state.stack_top_node),
                edge_value: parse_state.top_edge_value,
            });
        }
        self.head = Arc::new(GSSNode::new_with_predecessors(DUMMY_HEAD_STATE_ID, new_head_links.into_iter().collect()));


        self.action_not_found_states  = not_found;   // keep for caller if wanted

        self.log_gss("Step-end", token_id);
        self.action_not_found_states.clear();        // current design: we drop them

        crate::debug!(4, "----------------------------------------------------------------");
    }

    /// Merging is now handled implicitly when states are added to `next` in the `step` method
    /// (via BTreeMap::insert_with) and when the head node is rebuilt.
    /// This method is now a no-op.
    pub fn merge_active_states(&mut self) {
        // This method is no longer necessary as merging is done on insertion and head rebuilding.
        // crate::debug!(3, "merge_active_states called (now a no-op due to BTreeMap usage)");
    }

    pub fn merge_with(&mut self, other: GLRParserState<T>) {
        assert!(std::ptr::eq(self.parser, other.parser));

        // Collect all parse states from both heads into a single map for merging
        let mut merged_states_map = BTreeMap::new();

        // Take links from self's head (if it's uniquely owned, it drains, otherwise it clones)
        let self_links_iter: Box<dyn Iterator<Item = PredecessorLink<T>>> = if let Ok(mut_head) = Arc::try_unwrap(self.head.clone()) {
             Box::new(mut_head.predecessors.into_iter())
        } else {
             Box::new(self.head.predecessors.iter().cloned()) // Clone if shared
        };

        for link in self_links_iter {
            let ps = ParseState { stack_top_node: link.node.as_arc().clone(), top_edge_value: link.edge_value.clone() };
            merged_states_map.insert_with(ps.key(), ps, |existing, new_s| existing.merge(new_s));
        }
        self.head = Arc::new(GSSNode::new(DUMMY_HEAD_STATE_ID)); // Reset self.head temporarily

        // Add states from other's head
        for link in other.head.predecessors.iter() { // Iterate to avoid consuming other.head if it's shared
            let ps = ParseState { stack_top_node: link.node.as_arc().clone(), top_edge_value: link.edge_value.clone() };
            merged_states_map.insert_with(ps.key(), ps, |existing, new_s| existing.merge(new_s));
        }

        // Rebuild self.head from the merged map
        let mut new_head_links = BTreeSet::new();
        for (_key, parse_state) in merged_states_map {
             new_head_links.insert(PredecessorLink {
                 node: ArcPtrWrapper::new(parse_state.stack_top_node),
                 edge_value: parse_state.top_edge_value,
             });
        }
        self.head = Arc::new(GSSNode::new_with_predecessors(DUMMY_HEAD_STATE_ID, new_head_links.into_iter().collect()));


        // Merge action_not_found_states (remains BTreeMap)
        for (key, state) in other.action_not_found_states {
            self.action_not_found_states.insert_with(key, state, |existing, new_s| existing.merge(new_s));
        }
    }

    pub fn is_ok(&self) -> bool {
        !self.head.predecessors.is_empty() // Check if the head node has any predecessors
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
    // Removed action_stack
}

impl<T: MergeAndIntersect + Ord + Clone + Debug + Hash> ParseState<T> {
    pub fn key(&self) -> ParseStateKey {
        ParseStateKey {
            stack_state_id: self.stack_top_node.state_id, // Use state_id from the stack_top_node
        }
    }

    /// Merges `other` into `self`. Assumes `self.key() == other.key()`.
    /// Merges the `top_edge_value` and the GSS predecessors of `stack_top_node`.
    pub fn merge(&mut self, other: ParseState<T>) {
        assert_eq!(self.key(), other.key());

        // Merge the top edge values using MergeAndIntersect::merge
        let combined_t = self.top_edge_value.merge(&other.top_edge_value);
        self.top_edge_value = combined_t;

        // Merge the GSS node predecessors
        // Get mutable access to self.stack_top_node, potentially cloning if shared (Arc > 1)
        let mutable_top_node = Arc::make_mut(&mut self.stack_top_node);

        // Merge the predecessor structures using GSSNode's merge_unchecked (state_id is already checked by ParseStateKey)
        mutable_top_node.merge_unchecked(Arc::unwrap_or_clone(other.stack_top_node));
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

