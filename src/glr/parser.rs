use crate::datastructures::gss::BulkMerge;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{
    NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID,
};
use crate::datastructures::gss::{GSSNode, GSSTrait};

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;
use crate::debug;

pub trait AndAndOr: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash {
    fn and(&self, other: &Self) -> Self;
    fn or(&self, other: &Self) -> Self;
}

impl AndAndOr for () {
    fn and(&self, _: &Self) -> Self { () }
    fn or(&self, _: &Self) -> Self { () }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateNodeContent<T: AndAndOr> {
    pub state_id: StateID,
    pub t: T,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState<T: AndAndOr> {
    pub stack: Arc<GSSNode<ParseStateNodeContent<T>>>,
    pub status: ParseStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    Shift(TerminalID),
    Reduce { production_id: ProductionID, len: usize, nonterminal_id: NonTerminalID },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParseStatus {
    Active,
    Inactive(StopReason),
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

    pub fn init_glr_parser<T: AndAndOr + Default>(&self) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            active_states: vec![self.init_parse_state()],
            inactive_states: Vec::new(),
        }
    }

    pub fn init_glr_parser_from_parse_state<T: AndAndOr + Default>(&self, parse_state: ParseState<T>) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            active_states: vec![parse_state],
            inactive_states: Vec::new(),
        }
    }

    pub fn init_glr_parser_from_parse_states<T: AndAndOr + Default>(
        &self,
        parse_states: Vec<ParseState<T>>,
    ) -> GLRParserState<T> {
        GLRParserState {
            parser: self,
            active_states: parse_states,
            inactive_states: Vec::new(),
        }
    }

    pub fn init_parse_state<T: AndAndOr + Default>(&self) -> ParseState<T> {
        let initial_content = ParseStateNodeContent {
            state_id: self.start_state_id,
            t: T::default(),
        };
        ParseState {
            stack: Arc::new(GSSNode::new(initial_content)),
            status: ParseStatus::Active,
        }
    }
    pub fn parse<T: AndAndOr + Default>(&self, input: &[TerminalID]) -> GLRParserState<T> {
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

        writeln!(f, "Parse Table:")?;
        writeln!(f, "  Start State: {}", self.start_state_id.0)?;
        for (&state_id, row) in stage_7_table.iter().collect::<BTreeMap<_, _>>() {
            writeln!(f, "  State {}:", state_id.0)?;

            writeln!(f, "    Items:")?;
            let item_set = item_set_map.get_by_right(&state_id).unwrap();
            for item in item_set {
                write!(f, "      - {} ->", item.production.lhs.0)?;
                for (i, symbol) in item.production.rhs.iter().enumerate() {
                    if i == item.dot_position {
                        write!(f, " •")?;
                    }
                    match symbol {
                        Symbol::Terminal(terminal) => write!(f, " {:?}", terminal.0)?,
                        Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0)?,
                    }
                }
                if item.dot_position == item.production.rhs.len() {
                    write!(f, " •")?;
                }
                writeln!(f)?;
            }

            writeln!(f, "    Actions:")?;
            for (&terminal_id, action) in &row.shifts_and_reduces {
                let terminal = terminal_map.get_by_right(&terminal_id).unwrap();
                match action {
                    Stage7ShiftsAndReduces::Shift(next_state_id) => {
                        writeln!(f, "      - {:?} -> Shift {}", terminal.0, next_state_id.0)?;
                    }
                    Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id: nonterminal, len } => {
                        let nt = non_terminal_map.get_by_right(nonterminal).unwrap();
                        writeln!(f, "      - {:?} -> Reduce {} (len {})", terminal.0, nt.0, len)?;
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
pub struct GLRParserState<'a, T: AndAndOr + Default> {
    pub parser: &'a GLRParser,
    pub active_states: Vec<ParseState<T>>,
    pub inactive_states: Vec<ParseState<T>>,
}

impl<'a, T: AndAndOr + Default> GLRParserState<'a, T> {
    pub fn parse(&mut self, input: &[TerminalID]) {
        self.parse_part(input);
    }

    pub fn parse_part(&mut self, input: &[TerminalID]) {
        for &token_id in input {
            self.step(token_id);
        }
    }

    pub fn with_step(mut self, token_id: TerminalID) -> Self {
        let mut parser = self;
        parser.step(token_id);
        parser
    }

    pub fn step(&mut self, token_id: TerminalID) {
        let mut next_active_states = Vec::new();
        let mut inactive_states = Vec::new();

        while let Some(state) = self.active_states.pop() {
            let stack = state.stack; // Arc<GSSNode<ParseStateNodeContent<T>>>
            let current_content = stack.peek(); // &ParseStateNodeContent<T>
            let current_state_id = current_content.state_id;
            let current_t = &current_content.t;

            let row = self.parser.stage_7_table.get(&current_state_id).unwrap();

            if let Some(action) = row.shifts_and_reduces.get(&token_id) {
                match action {
                    Stage7ShiftsAndReduces::Shift(next_state_id) => {
                        debug!(3, "Shifting");
                        let new_content = ParseStateNodeContent { state_id: *next_state_id, t: current_t.clone() };
                        let new_stack = stack.push(new_content);
                        next_active_states.push(ParseState {
                            // stack: Arc::new(new_stack), // GSSNode::push now returns Arc
                            stack: Arc::new(new_stack),
                            status: ParseStatus::Active,
                        });
                    }
                    Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id: nonterminal, len } => {
                        debug!(3, "Reducing by production {:?} with len {}", production_id, len);
                        let mut popped_stack_nodes = stack.popn(*len);
                        popped_stack_nodes.bulk_merge();
                        for stack_node in popped_stack_nodes {
                            // stack_node is Arc<GSSNode<ParseStateNodeContent<T>>>
                            let revealed_content = stack_node.peek(); // &ParseStateNodeContent<T>
                            let revealed_state_id = revealed_content.state_id;
                            let revealed_t = &revealed_content.t;
                            let goto_row = self.parser.stage_7_table.get(&revealed_state_id).unwrap();

                            if let Some(&goto_state) = goto_row.gotos.get(nonterminal) {
                                debug!(3, "Going to state {:?}", goto_state);
                                // Apply 'and' operation
                                let combined_t = revealed_t.and(current_t);
                                let new_content = ParseStateNodeContent { state_id: goto_state, t: combined_t };
                                let new_stack = stack_node.push(new_content);
                                self.active_states.push(ParseState {
                                    stack: Arc::new(new_stack),
                                    status: ParseStatus::Active,
                                });
                            } else {
                                inactive_states.push(ParseState {
                                    stack: stack_node,
                                    status: ParseStatus::Inactive(StopReason::GotoNotFound),
                                });
                            }
                        }
                    }
                    Stage7ShiftsAndReduces::Split { shift, reduces } => {
                        debug!(3, "Split");
                        if let Some(shift_state) = shift {
                            // Shift part (same as above)
                            let new_content = ParseStateNodeContent { state_id: *shift_state, t: current_t.clone() };
                            let new_stack = stack.push(new_content);
                            next_active_states.push(ParseState {
                                stack: Arc::new(new_stack),
                                status: ParseStatus::Active,
                            });
                        }

                        for (len, nt_ids) in reduces {
                            let mut popped_stack_nodes = stack.popn(*len);
                            popped_stack_nodes.bulk_merge();
                            for (nt_id, prod_ids) in nt_ids {
                                for stack_node in &popped_stack_nodes {
                                    // Reduce part (same as above)
                                    let revealed_content = stack_node.peek();
                                    let revealed_state_id = revealed_content.state_id;
                                    let revealed_t = &revealed_content.t;
                                    let goto_row = self.parser.stage_7_table.get(&revealed_state_id).unwrap();

                                    if let Some(&goto_state) = goto_row.gotos.get(nt_id) {
                                        // Apply 'and' operation
                                        let combined_t = revealed_t.and(current_t);
                                        let new_content = ParseStateNodeContent { state_id: goto_state, t: combined_t };
                                        let new_stack = Arc::new(stack_node.push(new_content));
                                        for prod_id in prod_ids {
                                            self.active_states.push(ParseState {
                                                stack: new_stack.clone(),
                                                status: ParseStatus::Active,
                                            });
                                        }
                                    } else {
                                        inactive_states.push(ParseState {
                                            stack: stack_node.clone(),
                                            status: ParseStatus::Inactive(StopReason::GotoNotFound),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                inactive_states.push(ParseState {
                    stack,
                    status: ParseStatus::Inactive(StopReason::ActionNotFound),
                });
            }
        }
        self.active_states = next_active_states;
        self.inactive_states.extend(inactive_states);
    }

    // TODO: Review merge logic, especially interaction with GSSNode::merge and ParseState::merge
    pub fn merge_active_states(&mut self) {
        let mut active_state_map: BTreeMap<ParseStateKey, ParseState<T>> = BTreeMap::new();

        for state in std::mem::take(&mut self.active_states) {
            let key = state.key();
            active_state_map.insert_with(key, state, |existing, new_state| {
                existing.merge(new_state);
            });
        }

        self.active_states = active_state_map.into_values().collect();
    }

    pub fn merge_with(&mut self, other: GLRParserState<T>) {
        assert!(std::ptr::eq(&self.parser, &other.parser));
        self.active_states.extend(other.active_states);
        self.inactive_states.extend(other.inactive_states);
    }

    pub fn fully_matches(&self) -> bool {
        !self.fully_matching_states().is_empty()
    }

    pub fn fully_matching_states(&self) -> Vec<&ParseState<T>> {
        self.inactive_states.iter().filter(|state| state.status == ParseStatus::Inactive(StopReason::GotoNotFound)).collect()
    }

    pub fn can_match(&self) -> bool {
        !self.active_states.is_empty()
    }

    pub fn matches_or_can_match(&self) -> bool {
        self.can_match() || self.fully_matches()
    }

    pub fn is_ok(&self) -> bool {
        !self.active_states.is_empty() || self.fully_matches()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
    // Removed action_stack
}

impl<T: AndAndOr + Default> ParseState<T> {
    pub fn key(&self) -> ParseStateKey {
        ParseStateKey {
            stack_state_id: self.stack.peek().state_id,
        }
    }

    /// Merges `other` into `self`. Assumes `self.key() == other.key()`.
    /// Merges the GSS structures and combines the `t` value at the top node using `AndAndOr::or`.
    pub fn merge(&mut self, other: ParseState<T>) {
        assert_eq!(self.key(), other.key());

        // Combine 't' values at the top node using 'or'
        let self_content = self.stack.peek();
        let other_content = other.stack.peek();
        let combined_t = self_content.t.or(&other_content.t);

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