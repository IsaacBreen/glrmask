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
    pub active_states: BTreeMap<ParseStateKey, ParseState<T>>,
    pub action_not_found_states: BTreeMap<ParseStateKey, ParseState<T>>,
}

impl<'a, T: MergeAndIntersect + Debug> GLRParserState<'a, T> {
    /* -------------------------------------------------
     * Helper utilities to make `step` compact and clear
     * ------------------------------------------------- */

    /// Push a new state on `stack` and wrap it in a `ParseState`.
    fn push_state(
        &self,
        stack: &Arc<GSSNode<ParseStateNodeContent<T>>>,
        next_state: StateID,
        t: T,
    ) -> ParseState<T> {
        let new_content = ParseStateNodeContent { state_id: next_state, t };
        ParseState { stack: Arc::new(stack.push(new_content)) }
    }

    /// Pop `len` nodes, follow the goto on `nt`, and return the resulting stacks.
    fn pop_and_goto(
        &self,
        stack: &Arc<GSSNode<ParseStateNodeContent<T>>>,
        len: usize,
        nt: NonTerminalID,
        cur_t: &T,
    ) -> Vec<Arc<GSSNode<ParseStateNodeContent<T>>>> {
        let mut parents = stack.popn(len);        // 1. pop
        let mut out = Vec::new();

        for parent in parents {
            let top = parent.peek();
            let goto = self.parser.stage_7_table[&top.state_id].gotos[&nt];
            let merged_t = top.t.intersect(cur_t);
            crate::debug!(4, "  Goto from state {} to state {}", top.state_id.0, goto.0);
            out.push(Arc::new(parent.push(ParseStateNodeContent {
                state_id: goto,
                t: merged_t,
            })));
        }
        out
    }

    /// Debug helper so the main `step` body stays short.
    fn log_gss(&self, phase: &str, token: TerminalID) {
        const MAX: usize = 30;
        let roots: Vec<_> = self.active_states.values().map(|s| s.stack.clone()).collect();
        let stats = gather_gss_stats(&roots);
        crate::debug!(3, "{} - token {} ({:?}) - – active: {}, nodes: {:?}",
                      phase, token.0, self.parser.terminal_map.get_by_right(&token).unwrap().0, self.active_states.len(), stats);

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
                                       p.iter().map(|n| n.value.state_id.0)
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
        self.log_gss("Step-start", token_id);

        let mut todo = std::mem::take(&mut self.active_states).into_values().collect::<Vec<_>>();
        let mut next = BTreeMap::<ParseStateKey, ParseState<T>>::new();
        let mut not_found = BTreeMap::<ParseStateKey, ParseState<T>>::new();

        /* ---------- core loop ---------- */
        while let Some(state) = todo.pop() { // Process states from the worklist
            let stack   = state.stack;
            let top     = stack.peek();
            let row     = &self.parser.stage_7_table[&top.state_id];

            match row.shifts_and_reduces.get(&token_id) {
                /* ------ 1. plain shift ------ */
                Some(Stage7ShiftsAndReduces::Shift(to)) => {
                    crate::debug!(4, "Shift from state {} via token {} to state {}", top.state_id.0, token_id.0, to.0);
                    let new_parse_state = self.push_state(&stack, *to, top.t.clone());
                    next.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                }

                /* ------ 2. single reduce ------ */
                Some(Stage7ShiftsAndReduces::Reduce{ nonterminal_id: nt,
                                                     len, .. }) => {
                    crate::debug!(4, "Reduce from state {} via token {} to nonterminal {}", top.state_id.0, token_id.0, nt.0);
                    for s in self.pop_and_goto(&stack, *len, *nt, &top.t) {
                        // Add to worklist for current step; merging happens when moving to `next`
                        todo.push(ParseState { stack: s }); 
                    }
                }

                /* ------ 3. shift / reduce split ------ */
                Some(Stage7ShiftsAndReduces::Split { shift, reduces }) => {
                    crate::debug!(4, "Split from state {} via token {}", top.state_id.0, token_id.0);
                    // optional shift part
                    if let Some(to) = shift {
                        crate::debug!(4, " Shift from state {} via token {} to state {}", top.state_id.0, token_id.0, to.0);
                        let new_parse_state = self.push_state(&stack, *to, top.t.clone());
                        next.insert_with(new_parse_state.key(), new_parse_state, |existing, new_s| existing.merge(new_s));
                    }
                    // every reduce alternative
                    for (len, nts) in reduces {
                        crate::debug!(4, " Reduce from state {} via token {} to nonterminals {:?}", top.state_id.0, token_id.0, nts);
                        for (nt, _prod_ids) in nts {        // we ignore prod-ids here
                            for s in self.pop_and_goto(&stack, *len, *nt, &top.t) {
                                // Add to worklist for current step
                                todo.push(ParseState { stack: s });
                            }
                        }
                    }
                }

                /* ------ 4. no action ------ */
                None => {
                    crate::debug!(4, "No action found for token {:?} in state {}", token_id.0, top.state_id.0);
                    let not_found_state = ParseState { stack };
                    not_found.insert_with(not_found_state.key(), not_found_state, |existing, new_s| existing.merge(new_s));
                },
            }
        }

        /* ---------- finish up ---------- */
        self.active_states            = next;
        self.action_not_found_states  = not_found;   // keep for caller if wanted

        self.log_gss("Step-end", token_id);
        self.action_not_found_states.clear();        // current design: we drop them
    }

    /// Merging is now handled implicitly when states are added to `next` in the `step` method.
    /// This method can be removed if no other part of the code relies on it explicitly.
    /// For now, let's keep it as a no-op or ensure it's not called.
    /// Given the new structure, explicit merging of `self.active_states` is no longer needed
    /// as `BTreeMap::insert_with` handles it.
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
