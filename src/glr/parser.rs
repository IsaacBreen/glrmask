use crate::datastructures::gss::BulkMerge;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID};
use crate::datastructures::gss::{GSSNode, GSSTrait};

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet, HashMap}; // Added HashMap
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;
use crate::debug;

// Represents the state of a single path in the GLR parse forest.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState {
    pub stack: Arc<GSSNode<StateID>>,
    // action_stack might be useful for reconstructing parse trees, but isn't strictly
    // necessary for determining valid next tokens. Keep it optional for now.
    pub action_stack: Option<Arc<GSSNode<Action>>>,
    pub status: ParseStatus,
}

// Custom Debug implementation to avoid overly verbose GSSNode output
impl Debug for ParseState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParseState")
         .field("stack_top", &self.stack.peek())
         // Optionally add more fields like stack depth or action stack top if needed
         // .field("action_stack_top", &self.action_stack.as_ref().map(|a| a.peek()))
         .field("status", &self.status)
         .finish()
    }
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
    ActionNotFound, // No shift/reduce action defined for the current state and lookahead token
    GotoNotFound,   // After a reduction, no GOTO state defined for the revealed state and non-terminal
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

    pub fn init_glr_parser(&self) -> GLRParserState {
        GLRParserState {
            parser: self,
            active_states: vec![self.init_parse_state()],
            inactive_states: Vec::new(),
        }
    }

    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState {
        let mut active_states = Vec::new();
        let mut inactive_states = Vec::new();
        if parse_state.status == ParseStatus::Active {
            active_states.push(parse_state);
        } else {
            inactive_states.push(parse_state);
        }
        GLRParserState {
            parser: self,
            active_states,
            inactive_states,
        }
    }

    pub fn init_glr_parser_from_parse_states(&self, parse_states: Vec<ParseState>) -> GLRParserState {
         let mut active_states = Vec::new();
         let mut inactive_states = Vec::new();
         for state in parse_states {
             if state.status == ParseStatus::Active {
                 active_states.push(state);
             } else {
                 inactive_states.push(state);
             }
         }
        GLRParserState {
            parser: self,
            active_states,
            inactive_states,
        }
    }

    pub fn init_parse_state(&self) -> ParseState {
        ParseState {
            stack: Arc::new(GSSNode::new(self.start_state_id)),
            action_stack: None, // Initialize action stack as None
            status: ParseStatus::Active,
        }
    }

    pub fn parse(&self, input: &[TerminalID]) -> GLRParserState {
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

// Represents the collection of all active and inactive parse states at a point in time.
#[derive(Debug, Clone)]
pub struct GLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub active_states: Vec<ParseState>,
    pub inactive_states: Vec<ParseState>,
}

impl<'a> GLRParserState<'a> {
    pub fn parse(&mut self, input: &[TerminalID]) {
        self.parse_part(input);
    }

    pub fn parse_part(&mut self, input: &[TerminalID]) {
        for &token_id in input {
            self.step(token_id);
            // Optional: Merge after each token if performance becomes an issue
            // self.merge_active_states();
        }
        // Final merge after processing all tokens in the part
        self.merge_active_states();
    }

    pub fn with_step(mut self, token_id: TerminalID) -> Self {
        self.step(token_id);
        self.merge_active_states(); // Merge after the step
        self
    }

    // Processes one input token (TerminalID) and updates the active/inactive states.
    pub fn step(&mut self, token_id: TerminalID) {
        let mut next_active_states = Vec::new();
        let mut current_inactive_states = std::mem::take(&mut self.inactive_states); // Keep previous inactive states

        // Use a temporary vector for states generated during reductions to avoid modifying
        // self.active_states while iterating.
        let mut reduction_generated_states = Vec::new();

        let mut current_active = std::mem::take(&mut self.active_states);

        while !current_active.is_empty() || !reduction_generated_states.is_empty() {
            // Process states generated by reductions first if any exist
            let state = if !reduction_generated_states.is_empty() {
                reduction_generated_states.pop().unwrap()
            } else {
                current_active.pop().unwrap()
            };

            // Ensure we only process active states
            if state.status != ParseStatus::Active {
                 current_inactive_states.push(state);
                 continue;
            }

            let stack = state.stack;
            let action_stack = state.action_stack; // Keep track of action stack
            let state_id = *stack.peek();

            // Lookup actions for the current state and input token
            let row = self.parser.stage_7_table.get(&state_id)
                .expect(&format!("State ID {:?} not found in parse table", state_id)); // More informative panic

            if let Some(action) = row.shifts_and_reduces.get(&token_id) {
                match action {
                    Stage7ShiftsAndReduces::Shift(next_state_id) => {
                        debug!(3, "Shifting to state {:?}", next_state_id);
                        let new_stack = stack.push(*next_state_id);
                        // Push Shift action onto the action stack
                        let new_actions = action_stack.push_or_init(Action::Shift(token_id));
                        next_active_states.push(ParseState {
                            stack: Arc::new(new_stack),
                            action_stack: Some(Arc::new(new_actions)),
                            status: ParseStatus::Active,
                        });
                    }
                    Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id, len } => {
                        debug!(3, "Reducing by production {:?} (len {})", production_id, len);
                        // Pop states from the GSS stack
                        let popped_stack_nodes = stack.popn(*len);
                        // popped_stack_nodes.bulk_merge(); // Merging happens later

                        for stack_node in popped_stack_nodes {
                            let revealed_state = *stack_node.peek();
                            let goto_row = self.parser.stage_7_table.get(&revealed_state)
                                .expect(&format!("State ID {:?} not found in parse table during GOTO lookup", revealed_state));

                            if let Some(&goto_state) = goto_row.gotos.get(nonterminal_id) {
                                debug!(3, "GOTO state {:?}", goto_state);
                                let new_stack = stack_node.push(goto_state);
                                // Push Reduce action onto the action stack
                                let new_actions = action_stack.clone().push_or_init(Action::Reduce { production_id: *production_id, len: *len, nonterminal_id: *nonterminal_id });
                                // Add to reduction_generated_states to re-process in the same step
                                reduction_generated_states.push(ParseState {
                                    stack: Arc::new(new_stack),
                                    action_stack: Some(Arc::new(new_actions)),
                                    status: ParseStatus::Active,
                                });
                            } else {
                                // No GOTO defined: This path becomes inactive
                                debug!(3, "No GOTO found for state {:?} and non-terminal {:?}", revealed_state, nonterminal_id);
                                current_inactive_states.push(ParseState {
                                    stack: stack_node, // Keep the stack state before GOTO failure
                                    action_stack: action_stack.clone(), // Keep action stack up to this point
                                    status: ParseStatus::Inactive(StopReason::GotoNotFound),
                                });
                            }
                        }
                    }
                    Stage7ShiftsAndReduces::Split { shift, reduces } => {
                        debug!(3, "Split action");
                        // Handle Shift part of the split
                        if let Some(shift_state) = shift {
                            debug!(3, "Split: Shifting to state {:?}", shift_state);
                            let new_stack = stack.push(*shift_state);
                            let new_actions = action_stack.clone().push_or_init(Action::Shift(token_id));
                            next_active_states.push(ParseState {
                                stack: Arc::new(new_stack),
                                action_stack: Some(Arc::new(new_actions)),
                                status: ParseStatus::Active,
                            });
                        }

                        // Handle Reduce parts of the split
                        for (len, nt_ids_to_prod_ids) in reduces {
                            debug!(3, "Split: Reducing with len {}", len);
                            let popped_stack_nodes = stack.popn(*len);
                            // popped_stack_nodes.bulk_merge(); // Merging happens later

                            for (nt_id, prod_ids) in nt_ids_to_prod_ids {
                                for stack_node in &popped_stack_nodes {
                                    let revealed_state = *stack_node.peek();
                                    let goto_row = self.parser.stage_7_table.get(&revealed_state)
                                        .expect(&format!("State ID {:?} not found in parse table during GOTO lookup", revealed_state));

                                    if let Some(&goto_state) = goto_row.gotos.get(nt_id) {
                                        debug!(3, "Split: GOTO state {:?}", goto_state);
                                        let new_stack_base = stack_node.push(goto_state);
                                        let new_stack_arc = Arc::new(new_stack_base); // Create Arc once

                                        for prod_id in prod_ids {
                                            let new_actions = action_stack.clone().push_or_init(Action::Reduce { production_id: *prod_id, len: *len, nonterminal_id: *nt_id });
                                            // Add to reduction_generated_states
                                            reduction_generated_states.push(ParseState {
                                                stack: new_stack_arc.clone(), // Clone Arc
                                                action_stack: Some(Arc::new(new_actions)),
                                                status: ParseStatus::Active,
                                            });
                                        }
                                    } else {
                                        debug!(3, "Split: No GOTO found for state {:?} and non-terminal {:?}", revealed_state, nt_id);
                                        current_inactive_states.push(ParseState {
                                            stack: stack_node.clone(),
                                            action_stack: action_stack.clone(),
                                            status: ParseStatus::Inactive(StopReason::GotoNotFound),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // No action defined for this state and token: This path becomes inactive
                 debug!(3, "No action found for state {:?} and token {:?}", state_id, token_id);
                current_inactive_states.push(ParseState {
                    stack, // Keep the stack state where the action failed
                    action_stack, // Keep action stack up to this point
                    status: ParseStatus::Inactive(StopReason::ActionNotFound),
                });
            }
        } // end while loop processing active/reduction states

        self.active_states = next_active_states;
        self.inactive_states = current_inactive_states;
        // Optional: Merge active states here if not done per token or after full parse
        // self.merge_active_states();
    }


    // Merges active states that have the same key (top stack state, top action).
    pub fn merge_active_states(&mut self) {
        if self.active_states.len() <= 1 { return; } // No need to merge 0 or 1 state

        let mut active_state_map: HashMap<ParseStateKey, ParseState> = HashMap::with_capacity(self.active_states.len());

        for state in std::mem::take(&mut self.active_states) {
             let key = state.key();
             match active_state_map.entry(key) {
                 std::collections::hash_map::Entry::Occupied(mut entry) => {
                     // Merge the incoming state into the existing one
                     entry.get_mut().merge(state);
                 }
                 std::collections::hash_map::Entry::Vacant(entry) => {
                     // Insert the new state
                     entry.insert(state);
                 }
             }
        }
        self.active_states = active_state_map.into_values().collect();
    }

    // Merges another GLRParserState into this one.
    pub fn merge_with(&mut self, other: GLRParserState) {
        // Ensure parsers are compatible (optional but recommended)
        assert!(std::ptr::eq(self.parser, other.parser), "Cannot merge states from different GLRParser instances");

        self.active_states.extend(other.active_states);
        self.inactive_states.extend(other.inactive_states);
        // Merge the combined active states
        self.merge_active_states();
    }

    // Checks if any inactive state represents a fully successful parse (stopped due to GotoNotFound, potentially at the end state).
    // Note: This is a basic check. A more robust check might involve verifying if the stack contains only the start symbol after reduction.
    pub fn fully_matches(&self) -> bool {
        !self.fully_matching_states().is_empty()
    }

    // Returns references to inactive states that stopped due to GotoNotFound.
    pub fn fully_matching_states(&self) -> Vec<&ParseState> {
        self.inactive_states.iter().filter(|state|
            state.status == ParseStatus::Inactive(StopReason::GotoNotFound)
            // Add more conditions here if needed, e.g., check stack content
        ).collect()
    }

    // Checks if there are any active states remaining.
    pub fn can_match(&self) -> bool {
        !self.active_states.is_empty()
    }

    // Checks if the parser is in a state where it has either already fully matched
    // or could potentially match with further input.
    pub fn matches_or_can_match(&self) -> bool {
        self.can_match() || self.fully_matches()
    }

    // Checks if the parser is in an "ok" state (can continue parsing or has successfully parsed).
    // This is often used to detect errors (when is_ok() is false).
    pub fn is_ok(&self) -> bool {
        !self.active_states.is_empty() || self.fully_matches()
    }
}

// Key used for merging ParseStates. Based on the top state ID on the stack
// and optionally the last action performed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)] // Use Hash for HashMap key
pub struct ParseStateKey {
    stack_top_state: StateID,
    // Including action_stack_top makes merging less aggressive but might be necessary
    // if the action history affects future parsing possibilities differently.
    // For constraint checking, often only the stack_top_state matters.
    // action_stack_top: Option<Action>,
}

impl ParseState {
    // Generates the key for this ParseState used for merging.
    pub fn key(&self) -> ParseStateKey {
        ParseStateKey {
            stack_top_state: *self.stack.peek(),
            // action_stack_top: self.action_stack.peek().cloned(),
        }
    }

    // Merges the 'other' ParseState into 'self'. Assumes keys are identical.
    pub fn merge(&mut self, other: ParseState) {
        // Basic assertion: Ensure keys match before merging.
        // If action_stack_top is included in the key, this assertion is sufficient.
        // If only stack_top_state is used, further checks might be needed if action stacks differ.
        assert_eq!(self.key(), other.key(), "Attempting to merge ParseStates with different keys");

        // Merge the GSS stacks. This handles the core graph merging.
        // `Arc::make_mut` ensures we don't modify shared Arcs unnecessarily.
        Arc::make_mut(&mut self.stack).merge(Arc::unwrap_or_clone(other.stack));

        // Merge action stacks if they exist.
        match (&mut self.action_stack, other.action_stack) {
            (Some(self_actions), Some(other_actions)) => {
                Arc::make_mut(self_actions).merge(Arc::unwrap_or_clone(other_actions));
            }
            (None, None) => { /* Both None, nothing to do */ }
            // If one is Some and the other is None, this indicates an inconsistency
            // if the keys (which might include action_stack_top) were supposed to match.
            // If keys *don't* include action_stack_top, decide on merging strategy:
            // - Keep self's action stack?
            // - Keep other's action stack?
            // - Set to None?
            // Current implementation assumes keys match fully, so this case is unreachable.
            _ => unreachable!("Mismatched action stack presence during merge with matching keys"),
        }
        // Status should be the same if keys match, no need to merge status.
    }
}

// Helper trait/impl for GSSNode to simplify pushing actions
trait PushOrInit<T: Clone + Ord + Hash + Debug> {
    fn push_or_init(&self, value: T) -> GSSNode<T>;
}

impl<T: Clone + Ord + Hash + Debug> PushOrInit<T> for Option<Arc<GSSNode<T>>> {
    fn push_or_init(&self, value: T) -> GSSNode<T> {
        match self {
            Some(arc_node) => arc_node.push(value),
            None => GSSNode::new(value),
        }
    }
}


// Helper trait for inserting/merging into BTreeMap (already present, kept for reference)
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
