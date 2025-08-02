use std::any::Any;
use std::cmp::Ordering;
use crate::datastructures::gss::{print_gss_forest, Acc, GSSPopper, GSSPopperItem};
use crate::datastructures::gss::{gather_gss_stats, find_longest_path, GSSNode, GSSStats, GSSPeek};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Row, Stage7ShiftsAndReducesLookaheadValue, Table, StateID, TerminalID};
use crate::constraint::{LLMTokenBV, LLMVocab}; // Import LLMTokenInfo

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::{Debug, Display, Formatter, Write};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use crate::debug;
use crate::profiler::GSS_LOGGING_ENABLED;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};
use crate::glr::automaton::compute_closure;
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::glr::table::{Reduce, ShiftsAndReducesWithoutDefaultReduce, ShiftsAndReducesFull, DefaultReduce};

/// A trait to provide a lazily-evaluated `expect`.
pub trait ExpectElse<T> {
    /// Unwraps an option, panicking with a message from a closure on `None`.
    fn expect_else<F>(self, f: F) -> T
    where
        F: FnOnce() -> String;
}

impl<T> ExpectElse<T> for Option<T> {
    #[inline]
    #[track_caller]
    fn expect_else<F>(self, f: F) -> T
    where
        F: FnOnce() -> String,
    {
        match self { Some(v) => v, None => panic!("{}", f()) }
    }
}

/// Helper enum that tells `process_action_queue` where the *new* states that
/// originate from a **reduce** should be put.
///
/// • `SameAsWork`   – push the produced states back onto the current
///                    `work_queue` (this is the classic LR(1) behaviour that
///                    Phase-2 needs).
/// • `Separate(_)` – push the states onto **another** queue that the caller
///                   supplies (Phase-1 needs this so that the reduce results
///                   are processed later during Phase-2).
enum ReduceQueue<'a> {
    SameAsWork,
    Separate(&'a mut VecDeque<ParseState>),
}

pub trait DynEq {
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}
pub trait DynOrd {
    fn dyn_cmp(&self, other: &dyn Any) -> std::cmp::Ordering;
}
pub trait DynHash {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
}
impl DynEq for () {
    fn dyn_eq(&self, _other: &dyn Any) -> bool { true }
}
impl DynOrd for () {
    fn dyn_cmp(&self, _other: &dyn Any) -> std::cmp::Ordering { std::cmp::Ordering::Equal }
}
impl DynHash for () {
    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) { }
}

pub trait UserDataTrait: Any + Send + Sync + Debug + DynEq + DynOrd + DynHash {}
impl UserDataTrait for () {}

pub type ActionFn = Arc<dyn Fn(&mut Arc<dyn UserDataTrait>) -> bool + Send + Sync>;


#[derive(Debug, Clone)]
pub struct ParseStateEdgeContent { 
    pub state_id: StateID,
}
impl PartialEq for ParseStateEdgeContent {
    fn eq(&self, other: &Self) -> bool {
        self.state_id == other.state_id
    }
}
impl Eq for ParseStateEdgeContent {}
impl PartialOrd for ParseStateEdgeContent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.state_id.partial_cmp(&other.state_id) {
            other_ord => other_ord,
        }
    }
}
impl Ord for ParseStateEdgeContent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.state_id.cmp(&other.state_id)
    }
}
impl Hash for ParseStateEdgeContent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state_id.hash(state);
    }
}

// JSONConvertible for ParseStateEdgeContent
impl JSONConvertible for ParseStateEdgeContent {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        // Handle user_data serialization:
        // Option 1: Panic if not default.
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for ParseStateEdgeContent".to_string()) // Corrected struct name
                                  .and_then(StateID::from_json)?;
                Ok(ParseStateEdgeContent { state_id })
            }
            _ => Err("Expected JSONNode::Object for ParseStateEdgeContent".to_string()), // Corrected struct name
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseState { // No longer generic
    pub stack: Arc<GSSNode>, // GSSNode is now concrete
}

impl ParseState {
    pub fn new() -> Self {
        ParseState { stack: Arc::new(GSSNode::new_fresh()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StopReason {
    ActionNotFound,
    GotoNotFound,
}
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserPhase {
    /// The parser has completed all reductions for the current state and is ready for a new token.
    ReadyForToken,
    /// The parser has processed a token (shifts and lookahead-reduces) and is ready for default reductions.
    ReadyForDefaultReductions,
}

#[derive(Clone)]
pub struct GLRParser {
    pub table: Table,
    pub productions: Vec<Production>,
    pub terminal_map: BiBTreeMap<Terminal, TerminalID>,
    pub non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    pub item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
    pub start_state_id: StateID,
    pub ignore_terminal_id: Option<TerminalID>,
}

impl JSONConvertible for GLRParser {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("stage_7_table".to_string(), self.table.to_json());
        obj.insert("productions".to_string(), self.productions.to_json());
        // obj.insert("start_production_id".to_string(), self.start_production_id.to_json()); // Implicitly 0
        obj.insert("terminal_map".to_string(), self.terminal_map.to_json());
        obj.insert("non_terminal_map".to_string(), self.non_terminal_map.to_json());
        obj.insert("item_set_map".to_string(), self.item_set_map.to_json());
        obj.insert("start_state_id".to_string(), self.start_state_id.to_json());
        obj.insert("ignore_terminal_id".to_string(), self.ignore_terminal_id.to_json());
        // Do not serialize self.actions
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let stage_7_table = obj.remove("stage_7_table").ok_or_else(|| "Missing field stage_7_table".to_string())
                                       .and_then(Table::from_json)?;
                let productions = obj.remove("productions").ok_or_else(|| "Missing field productions".to_string())
                                     .and_then(Vec::<Production>::from_json)?;
                // For backwards compatibility, we can read and ignore it.
                let _start_production_id = obj.remove("start_production_id").and_then(|n| usize::from_json(n).ok());
                let terminal_map = obj.remove("terminal_map").ok_or_else(|| "Missing field terminal_map".to_string())
                                      .and_then(|n| BiBTreeMap::<Terminal, TerminalID>::from_json(n))?;
                let non_terminal_map = obj.remove("non_terminal_map").ok_or_else(|| "Missing field non_terminal_map".to_string())
                                          .and_then(|n| BiBTreeMap::<NonTerminal, NonTerminalID>::from_json(n))?;
                let item_set_map = obj.remove("item_set_map").ok_or_else(|| "Missing field item_set_map".to_string())
                                      .and_then(|n| BiBTreeMap::<BTreeSet<Item>, StateID>::from_json(n))?;
                let start_state_id = obj.remove("start_state_id").ok_or_else(|| "Missing field start_state_id".to_string())
                                        .and_then(StateID::from_json)?;
                let ignore_terminal_id = obj.remove("ignore_terminal_id")
                    .ok_or_else(|| "Missing field ignore_terminal_id for GLRParser".to_string())
                    .and_then(Option::<TerminalID>::from_json)?;
                Ok(GLRParser {
                    table: stage_7_table,
                    productions,
                    terminal_map,
                    non_terminal_map,
                    item_set_map,
                    start_state_id,
                    ignore_terminal_id,
                })
            }
            _ => Err("Expected JSONNode::Object for GLRParser".to_string()),
        }
    }
}

impl Debug for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLRParser")
            .field("stage_7_table", &self.table)
            .field("productions", &self.productions)
            .field("terminal_map", &self.terminal_map)
            .field("non_terminal_map", &self.non_terminal_map)
            .field("item_set_map", &self.item_set_map)
            .field("start_state_id", &self.start_state_id)
            .field("ignore_terminal_id", &self.ignore_terminal_id)
            .finish()
    }
}

impl PartialEq for GLRParser {
    fn eq(&self, other: &Self) -> bool {
        self.table == other.table &&
        self.productions == other.productions &&
        self.terminal_map == other.terminal_map &&
        self.non_terminal_map == other.non_terminal_map &&
        self.item_set_map == other.item_set_map &&
        self.start_state_id == other.start_state_id &&
        self.ignore_terminal_id == other.ignore_terminal_id
    }
}

impl Eq for GLRParser {}

impl GLRParser {
    pub fn new(
        stage_7_table: Table,
        productions: Vec<Production>,
        terminal_map: BiBTreeMap<Terminal, TerminalID>,
        non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
        item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
        start_state_id: StateID,
        actions: BTreeMap<NonTerminal, ActionFn>, // Parameter type
        ignore_terminal_id: Option<TerminalID>,
    ) -> Self {
        let converted_actions: BTreeMap<NonTerminalID, ActionFn> = actions
            .into_iter()
            .map(|(nt, func)| {
                let nt_id = non_terminal_map.get_by_left(&nt)
                    .unwrap_or_else(|| panic!("NonTerminal {:?} not found in non_terminal_map during GLRParser construction", nt));
                (*nt_id, func)
            })
            .collect();

        Self {
            table: stage_7_table,
            productions,
            terminal_map,
            non_terminal_map,
            item_set_map,
            start_state_id,
            ignore_terminal_id,
        }
    }

    pub fn init_glr_parser(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        self.init_glr_parser_with_acc()
    }

    pub fn init_glr_parser_null(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: ParseState::new(),
            accepted: false,
            phase: ParserPhase::ReadyForToken, // It's empty, so it's ready for a token.
        }
    }

    pub fn init_glr_parser_with_acc(&self) -> GLRParserState { // No longer generic
        let initial_parse_state = self.init_parse_state_with_acc();
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions, // An initial state might have default reductions.
        };
        parser_state
    }
    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState { // No longer generic
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: parse_state,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
        };
        parser_state
    }

    pub fn init_parse_state(&self, llm_vocab: Option<Arc<LLMVocab>>) -> ParseState { // No longer generic
        self.init_parse_state_with_acc()
    }

    pub fn init_parse_state_with_acc(&self) -> ParseState { // No longer generic
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
        };
        let stack = Arc::new(GSSNode::new_fresh().push(initial_content)); // pushed node has initial_acc
        ParseState { stack }
    }

    pub fn parse(&self, input: &[TerminalID], llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        let mut state = self.init_glr_parser(llm_vocab);
        state.parse(input);
        state
    }

    pub fn explain_stack(&self, stack: &[StateID]) -> String {
        let mut result = String::new();
        writeln!(&mut result, "--- Explaining Parse Stack: {:?} ---", stack.iter().map(|s| s.0).collect::<Vec<_>>()).unwrap();

        for &state_id in stack {
            writeln!(&mut result, "\nState {}:", state_id.0).unwrap();
            self.format_state_details(&mut result, state_id, "  ").unwrap();
            writeln!(&mut result, "---").unwrap();
        }

        result
    }

    pub fn format_state_details<W: std::fmt::Write>(
        &self,
        f: &mut W,
        state_id: StateID,
        indent: &str,
    ) -> std::fmt::Result {
        let sub_indent = format!("{}  ", indent);

        // --- Items ---
        if let Some(items) = self.item_set_map.get_by_right(&state_id) {
            writeln!(f, "{}Items:", indent)?;
            if items.is_empty() {
                writeln!(f, "{}  (None)", indent)?;
            } else {
                let mut grouped_items: BTreeMap<(&Production, usize), BTreeSet<Option<Terminal>>> = BTreeMap::new();
                for item in items {
                    grouped_items
                        .entry((&item.production, item.dot_position))
                        .or_default()
                        .insert(item.lookahead.clone());
                }

                for ((production, dot_pos), lookaheads) in grouped_items {
                    write!(f, "{}- [{} ->", sub_indent, production.lhs.0)?;
                    for (i, symbol) in production.rhs.iter().enumerate() {
                        if i == dot_pos {
                            write!(f, " •")?;
                        }
                        match symbol {
                            Symbol::Terminal(terminal) => write!(f, " {}", terminal)?,
                            Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0)?,
                        }
                    }
                    if dot_pos == production.rhs.len() {
                        write!(f, " •")?;
                    }
                    write!(f, ", {{")?;
                    let mut lookahead_strs: Vec<String> = lookaheads.iter().map(|l| if let Some(t) = l { t.to_string() } else { "ε".to_string() }).collect();
                    lookahead_strs.sort();
                    const MAX_LOOKAHEADS_TO_SHOW: usize = 5;
                    if lookahead_strs.len() > MAX_LOOKAHEADS_TO_SHOW {
                        let truncated: Vec<_> = lookahead_strs.iter().take(MAX_LOOKAHEADS_TO_SHOW).cloned().collect();
                        write!(f, "{}... ({} total)", truncated.join(", "), lookahead_strs.len())?;
                    } else {
                        write!(f, "{}", lookahead_strs.join(", "))?;
                    }
                    writeln!(f, "}}]")?;
                }
            }
        } else {
            writeln!(f, "{}Items: (State ID not found in item set map)", indent)?;
        }

        // --- Actions & Gotos ---
        if let Some(row) = self.table.get(&state_id) {
            writeln!(f, "{}Actions (without default reduce):", indent)?;
            format_actions(f, &row.shifts_and_reduces_without_default_reduce, &self.terminal_map, &self.non_terminal_map, &self.productions, &sub_indent)?;

            writeln!(f, "{}Actions (full):", indent)?;
            format_actions(f, &row.shifts_and_reduces_full, &self.terminal_map, &self.non_terminal_map, &self.productions, &sub_indent)?;

            writeln!(f, "{}Default Action:", indent)?;
            if let Some(reduce_action) = &row.default_reduce.reduce {
                let nt_name = self.non_terminal_map.get_by_right(&reduce_action.nonterminal_id).unwrap();
                let pids: Vec<String> = reduce_action.production_ids.iter().map(|p| p.0.to_string()).collect();
                writeln!(f, "{}  - Default Reduce {} (len {}) via rules [{}]", indent, nt_name.0, reduce_action.len, pids.join(", "))?;
            } else {
                writeln!(f, "{}  - No default reduce", indent)?;
            }
            if row.default_reduce.clone_and_merge {
                writeln!(f, "{}  - Clone and merge", indent)?;
            } else {
                writeln!(f, "{}  - No clone and merge", indent)?;
            }

            writeln!(f, "{}Gotos:", indent)?;
            if row.gotos.is_empty() {
                writeln!(f, "{}  (No goto actions)", indent)?;
            } else {
                let mut sorted_gotos: Vec<_> = row.gotos.iter().collect();
                sorted_gotos.sort_by_key(|(ntid, _)| self.non_terminal_map.get_by_right(ntid).unwrap());

                for (non_terminal_id, goto) in sorted_gotos {
                    let non_terminal = self.non_terminal_map.get_by_right(non_terminal_id).unwrap();
                    let goto_str = if let Some(state_id_val) = goto.state_id {
                        if goto.accept {
                            format!("{} or accept", state_id_val.0)
                        } else {
                            format!("{}", state_id_val.0)
                        }
                    } else if goto.accept {
                        "accept".to_string()
                    } else {
                        "no-op".to_string()
                    };
                    writeln!(f, "{}  - {} -> {}", indent, non_terminal.0, goto_str)?;
                }
            }
        } else {
            writeln!(f, "{}Actions & Gotos: (State ID not found in parse table)", indent)?;
        }
        Ok(())
    }
}

fn format_actions<W: std::fmt::Write>(
    f: &mut W,
    actions: &BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    non_terminal_map: &BiBTreeMap<NonTerminal, NonTerminalID>,
    productions: &[Production],
    indent: &str,
) -> std::fmt::Result {
    if actions.is_empty() {
        return writeln!(f, "{} (none)", indent);
    }

    // Group terminals by action
    let mut grouped_actions: BTreeMap<&Stage7ShiftsAndReducesLookaheadValue, BTreeSet<&Terminal>> = BTreeMap::new();
    for (tid, action) in actions {
        let terminal = terminal_map.get_by_right(tid).unwrap();
        grouped_actions.entry(action).or_default().insert(terminal);
    }

    for (action, terminals) in grouped_actions {
        // Format action
        let action_str = match action {
            Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
                format!("Shift {}", next_state_id.0)
            }
            Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                let nt_name = non_terminal_map.get_by_right(nonterminal_id).unwrap();
                let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
                format!("Reduce {} (len {}) via rules [{}]", nt_name.0, len, pids.join(", "))
            }
            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                let mut s = "Conflict:".to_string();
                let inner_indent = format!("\n{}        ", indent); // indent + "        "
                if let Some(shift_state) = shift {
                    let _ = write!(s, "{}  - Shift {}", inner_indent, shift_state.0);
                }
                for (_len, nts) in reduces {
                    for (_nt_id, prod_ids) in nts {
                        for prod_id_val in prod_ids {
                            let prod = productions.get(prod_id_val.0).unwrap();
                            let _ = write!(s, "{}  - Reduce by rule #{} ({})", inner_indent, prod_id_val.0, prod);
                        }
                    }
                }
                s
            }
        };

        // Format terminals
        let mut terminal_names: Vec<_> = terminals.iter().map(|t| t.to_string()).collect();
        terminal_names.sort();
        const MAX_TERMINALS_TO_SHOW: usize = 5;
        let terms_str = if terminal_names.len() > MAX_TERMINALS_TO_SHOW {
            let truncated: Vec<_> = terminal_names.iter().take(MAX_TERMINALS_TO_SHOW).map(|s| s.as_str()).collect();
            format!("{}... ({} total)", truncated.join(", "), terminal_names.len())
        } else {
            terminal_names.join(", ")
        };

        // Put action on its own line, then terminals on the next, to avoid extremely long lines.
        writeln!(f, "{}- {} on {{ {} }}", indent, action_str, terms_str)?;
    }
    Ok(())
}

impl Display for GLRParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stage_7_table = &self.table;
        let terminal_map = &self.terminal_map;
        let non_terminal_map = &self.non_terminal_map;
        let item_set_map = &self.item_set_map;

        use crate::glr::grammar::{Production, Symbol, Terminal};

        writeln!(f, "Parse Table:")?;
        writeln!(f, "  Start State: {}", self.start_state_id.0)?;
        for (&state_id, row) in stage_7_table.iter().collect::<BTreeMap<_, _>>() {
            writeln!(f, "  State {}:", state_id.0)?;
            self.format_state_details(f, state_id, "    ")?;
        }

        writeln!(f, "\nTerminal Map (name to terminal ID):")?;
        for (terminal, terminal_id) in terminal_map {
            writeln!(f, "  {} -> {}", terminal, terminal_id.0)?;
        }

        writeln!(f, "\nNon-Terminal Map:")?;
        for (non_terminal, non_terminal_id) in non_terminal_map {
            writeln!(f, "  {} -> {}", non_terminal.0, non_terminal_id.0)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLRParserState<'a> { // No longer generic
    pub parser: &'a GLRParser,
    pub active_state: ParseState,
    accepted: bool,                // <-- NEW
    phase: ParserPhase,
}

impl Display for GLRParserState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO: this is bad. make this better
        // Display the stack
        self.log_gss("    ", TerminalID(0), false, false);
        Ok(())
    }
}

impl<'a> GLRParserState<'a> { // No longer generic
    fn push_state(
        &self,
        peek: &GSSPeek,
        new_content: ParseStateEdgeContent,
    ) -> ParseState {
        crate::debug!(4, "Pushing new state with content: {:?}", new_content);
        let new_gss_node_instance = peek.push_on_parent(new_content);
        ParseState { stack: Arc::new(new_gss_node_instance) }
    }

    /// Shared inner loop for phase 1 and phase 2.
    ///
    /// `reduce_queue`
    ///   • `ReduceQueue::Separate(queue)` – reduces are pushed onto `queue`
    ///   • `ReduceQueue::SameAsWork`      – reduces are pushed back onto `work_queue`
    ///
    /// `action_selector` chooses between the phase-1 or phase-2 action map.
    fn process_action_queue<F>(
        &mut self,
        token_id: TerminalID,
        work_queue: &mut VecDeque<ParseState>,
        mut reduce_queue: ReduceQueue,
        shifted_states_todo: &mut VecDeque<ParseState>,
        action_selector: F,
    ) where
        F: Fn(&Row)
            -> &BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>,
    {
        while let Some(state) = work_queue.pop_front() {
            for peek in GSSNode::peek_iter(&state.stack) {
                let row = &self.parser.table[&peek.edge_value().state_id];
                if let Some(action) = action_selector(row).get(&token_id) {
                    match action {
                        Stage7ShiftsAndReducesLookaheadValue::Shift(to) => {
                            crate::debug!(5, "Action: Shift to state {}", to.0);
                            let new_parse_state =
                                self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                            shifted_states_todo.push_back(new_parse_state);
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Reduce {
                            nonterminal_id: nt,
                            len,
                            ..
                        } => {
                            crate::debug!(5, "Action: Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), len);
                            let s_new_arc = self.reduce_and_goto(&peek, *nt, *len);
                            if !s_new_arc.is_empty() {
                                let new_parse_state = ParseState { stack: s_new_arc };
                                match &mut reduce_queue {
                                    ReduceQueue::SameAsWork         => work_queue.push_back(new_parse_state),
                                    ReduceQueue::Separate(queue) => queue.push_back(new_parse_state),
                                }
                            }
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                            crate::debug!(5, "Action: Split with shift and reduces");
                            if let Some(to) = shift {
                                crate::debug!(5, "Action (Split): Shift to state {}", to.0);
                                let new_parse_state =
                                    self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                                shifted_states_todo.push_back(new_parse_state);
                            }
                            for (len, nts) in reduces {
                                for (nt, _prod_ids) in nts {
                                    crate::debug!(5, "Action (Split): Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), *len);
                                    let s_new_arc = self.reduce_and_goto(&peek, *nt, *len);
                                    if !s_new_arc.is_empty() {
                                        let new_parse_state = ParseState { stack: s_new_arc };
                                        match &mut reduce_queue {
                                            ReduceQueue::SameAsWork         => work_queue.push_back(new_parse_state),
                                            ReduceQueue::Separate(queue) => queue.push_back(new_parse_state),
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    crate::debug!(5, "No action found for token '{}' in state {}", self.parser.terminal_map.get_by_right(&token_id).unwrap(), peek.edge_value().state_id.0);
                }
            }
        }
    }

    fn _do_actions_without_default(&mut self, token_id: TerminalID, phase1_todo: &mut VecDeque<ParseState>, phase2_todo: &mut VecDeque<ParseState>, shifted_states_todo: &mut VecDeque<ParseState>) {
        let token_display = self.parser.terminal_map.get_by_right(&token_id).unwrap();
        crate::debug!(4, "Phase 1: Processing token '{}'", token_display);
        timeit!("GLRParserState::step::phase1", {
            self.process_action_queue(
                token_id,
                phase1_todo,
                ReduceQueue::Separate(phase2_todo),
                shifted_states_todo,
                |row| &row.shifts_and_reduces_without_default_reduce,
            );
        });
    }

    fn _do_actions_with_default(&mut self, token_id: TerminalID, phase2_todo: &mut VecDeque<ParseState>, shifted_states_todo: &mut VecDeque<ParseState>) {
        crate::debug!(4, "Phase 1 completed, proceeding to Phase 2 with {} shifted states", shifted_states_todo.len());
        timeit!("GLRParserState::step::phase2", {
            // Reduces are pushed back onto the same queue (`None`).
            self.process_action_queue(
                token_id,
                phase2_todo,
                ReduceQueue::SameAsWork,
                shifted_states_todo,
                |row| &row.shifts_and_reduces_full,
            );
            self.phase = ParserPhase::ReadyForDefaultReductions;
        });
    }

    #[time_it("GLRParserState::reduce_and_goto")]
    fn reduce_and_goto(
        &mut self,
        peek: &GSSPeek,
        nt: NonTerminalID,
        len: usize,
    ) -> Arc<GSSNode> {
        let popped = timeit!(peek.popn(len));
        crate::debug!(4, "Reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
        crate::debug!(4, "Popped with {} results...", popped.num_predecessors());

        let mut out = Vec::new();
        for popper_item in popped.iter() {
            for peek2 in popper_item.peek_iter() {
                let state_id = peek2.edge_value().state_id;
                let goto = self.parser.table.get(&state_id).and_then(|row| row.gotos.get(&nt)).expect_else(|| {
                    format!("Goto not found for NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), state_id)
                });

                if goto.accept {
                    crate::debug!(4, "Accepting with NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), state_id);
                    self.accepted = true;
                }

                if let Some(goto_state_id) = goto.state_id {
                    crate::debug!(4, "Goto found for NT '{}' in state {:?}: Goto State {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), state_id, goto_state_id.0);
                    let new_gss_node = peek2.push_on_parent(ParseStateEdgeContent { state_id: goto_state_id });
                        out.push(new_gss_node);
                }
            }
        }

        if out.is_empty() {
            Arc::new(GSSNode::new_fresh())
        } else if out.len() == 1 {
            Arc::new(out.into_iter().next().unwrap())
        } else {
            let mut out_iter = out.into_iter();
            let mut out_node = out_iter.next().unwrap();
            for next_node in out_iter {
                out_node.merge_with_depth(2, &next_node);
            }
            Arc::new(out_node)
        }
    }

    #[time_it("GLRParserState::process_token")]
    pub fn process_token(&mut self, token_id: TerminalID) {
        // Reset acceptance flag for the new token
        self.accepted = false;

        if Some(token_id) == self.parser.ignore_terminal_id {
            crate::debug!(4, "Ignoring token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());
            self.phase = ParserPhase::ReadyForDefaultReductions; // Skip phase 1 and 2, go straight to phase 3
            return;
        }

        self.log_gss("Phase1/2-start", token_id, false, false);

        let mut phase2_todo: VecDeque<ParseState> = VecDeque::new();
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();

        if self.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: VecDeque<ParseState> = VecDeque::new();
            phase1_todo.push_back(self.active_state.clone());
            self._do_actions_without_default(token_id, &mut phase1_todo, &mut phase2_todo, &mut shifted_states_todo);
        } else { // ParserPhase::ReadyForDefaultReductions
            // If we are ready for phase 3, it means we have a set of shifted states.
            // Instead of performing default reductions (phase 3), we can process the next token.
            // The user suggests skipping phase 1 and going straight to phase 2.
            // This means we take the current active states and process them with the full action table.
            phase2_todo.push_back(self.active_state.clone());
        }

        // --- Phase 2 ---
        self._do_actions_with_default(token_id, &mut phase2_todo, &mut shifted_states_todo);

        // Consolidate all shifted states into the new active_state for phase 3
        crate::debug!(4, "Phase 2 completed, consolidating {} shifted states into active state", shifted_states_todo.len());
        let mut next_active = ParseState::new();
        for state in shifted_states_todo {
            next_active.merge(state);
        }
        self.active_state = next_active;
        self.log_gss("Phase1/2-end", token_id, false, false);
    }

    #[time_it("GLRParserState::process_default_reductions")]
    pub fn process_default_reductions(&mut self) {
        self.log_gss("Phase3-start", TerminalID(0), false, false); // Log with dummy token ID
        if self.phase == ParserPhase::ReadyForToken {
            crate::debug!(4, "Phase 3 skipped, parser is ready for Phase 1");
            return;
        }
        assert_eq!(self.phase, ParserPhase::ReadyForDefaultReductions);

        // Key is (depth, state_id) to process shortest stacks first.
        let mut work_map: BTreeMap<(usize, StateID), ParseState> = BTreeMap::new();

        // let get_depth = |peek: &GSSNode| peek.max_depth();
        let get_depth = |peek: &GSSNode| 0;

        // Peel off the top edges to populate the initial work map.
        for peek in GSSNode::peek_iter(&self.active_state.stack) {
            let isolated_state = ParseState { stack: peek.isolated_parent() };
            let depth = get_depth(&isolated_state.stack);
            let state_id = peek.edge_value().state_id;
            work_map.entry((depth, state_id))
                .and_modify(|s| s.merge(isolated_state.clone()))
                .or_insert(isolated_state);
        }

        let mut next_active_state = ParseState::new();

        let stats = gather_gss_stats(&[self.active_state.stack.as_ref()]);
        crate::debug!(5, "GLRParserState::process_default_reductions: Stats: {:?}", stats);

        crate::debug!(4, "Phase 3: Processing {} states", work_map.len());
        timeit!(format!("GLRParserState::step::phase3 - unique_nodes: {}", stats.unique_nodes), {
        // timeit!("GLRParserState::step::phase3", {
            while let Some(((_depth, state_id), state)) = work_map.pop_first() {
                let row = &self.parser.table[&state_id];

                if let Some(ref r) = row.default_reduce.reduce {
                    crate::debug!(5, "Action (Phase 3): Default Reduce by NT '{}' (len {}) in state {}, num_predecessors: {}",
                                  self.parser.non_terminal_map.get_by_right(&r.nonterminal_id).unwrap(),
                                  r.len, state_id.0, state.stack.num_predecessors());
                    timeit!(format!("GLRParserState::step::phase3::reduce NT '{}' (len {}) in state {}",
                                    self.parser.non_terminal_map.get_by_right(&r.nonterminal_id).unwrap(),
                                    r.len, state_id.0), {
                    // timeit!(format!("GLRParserState::step::phase3::reduce NT (len {})", r.len), {
                        // For each peek in the current state, reduce and goto.
                        // This is the core of phase 3: reducing all stacks with the same state_id.
                        // We will merge the results into a new stack part.
                    let mut reduced_stack = GSSNode::new_fresh();
                    for peek in GSSNode::peek_iter(&state.stack) {
                        // println!("GLRParserState::do_phase3: Reducing with state_id: {}, len: {}, nonterminal: {}, production_ids: {:?}",
                        //          state_id.0, r.len, self.parser.non_terminal_map.get_by_right(&r.nonterminal_id).unwrap(), r.production_ids);


                        let len = r.len;
                        let nt = r.nonterminal_id;
                        let popped = timeit!(peek.popn(len));
                        crate::debug!(4, "Reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
                        crate::debug!(4, "Popped with {} results...", popped.num_predecessors());

                        let mut out = Vec::new();
                        for popper_item in popped.iter() {
                            for peek2 in popper_item.peek_iter() {
                                timeit!(format!("GLRParserState::step::phase3::reduce::fast NT (len {})", len), {});
                                let mut goto_state_ids = BTreeSet::new();
                                let state_id = peek2.edge_value().state_id;
                                let mut current_nt = nt;
                                // println!("loop start");
                                let mut i = 0;
                                loop {
                                    i += 1;
                                    // println!("loop current_nt: {:?}", current_nt);
                                    let goto = self.parser.table.get(&state_id).and_then(|row| row.gotos.get(&current_nt)).expect_else(|| {
                                        format!("Goto not found for NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), state_id)
                                    });

                                    if goto.accept {
                                        crate::debug!(4, "Accepting with NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), state_id);
                                        self.accepted = true;
                                    }

                                    if let Some(goto_state_id) = goto.state_id {
                                        timeit!(format!("GLRParserState::step::phase3::reduce::fast::loop NT (len {})", len), {});
                                        let DefaultReduce { clone_and_merge, reduce } = &self.parser.table[&goto_state_id].default_reduce;
                                        let continue_fast_reduce = matches!(reduce, Some(r) if r.len == 1);
                                        // let continue_fast_reduce = false;
                                        if continue_fast_reduce {
                                            if *clone_and_merge {
                                                goto_state_ids.insert(goto_state_id);
                                            }
                                            current_nt = reduce.clone().unwrap().nonterminal_id;
                                        } else {
                                            goto_state_ids.insert(goto_state_id);
                                            // println!("break reason: continue_fast_reduce is false");
                                            break;
                                        }
                                    } else {
                                        // println!("break reason: goto.state_id is None");
                                        break;
                                    }
                                }
                                // if i > 1 {
                                //     println!("number of loops: {:5}, number of goto_state_ids: {}", i, goto_state_ids.len());

                                    // Round to nearest power of 2
                                    let i_rounded_to_nearest_pow = if i == 0 {
                                        1
                                    } else {
                                        1 << (32 - (i as u32 - 1).leading_zeros())
                                    };

                                    let goto_state_ids_rounded_to_nearest_pow = if goto_state_ids.len() == 0 {
                                        1
                                    } else {
                                        1 << (32 - (goto_state_ids.len() as u32 - 1).leading_zeros())
                                    };

                                    timeit!(format!("GLRParserState::step::phase2::goto::number of loops (rounded to nearest pow of 2), number of goto_state_ids (rounded to nearest pow): {:5}, {:5}", i_rounded_to_nearest_pow, goto_state_ids_rounded_to_nearest_pow), {});
                                // }

                                let new_gss_node = peek2.isolated_parent().push_many(goto_state_ids.into_iter().map(|state_id| ParseStateEdgeContent { state_id }).collect());
                                out.push(new_gss_node);
                            }
                        }

                        let new_stack_part = if out.is_empty() {
                            Arc::new(GSSNode::new_fresh())
                        } else if out.len() == 1 {
                            Arc::new(out.into_iter().next().unwrap())
                        } else {
                            let mut out_iter = out.into_iter();
                            let mut out_node = out_iter.next().unwrap();
                            for next_node in out_iter {
                                out_node.merge_with_depth(3, &next_node);
                            }
                            Arc::new(out_node)
                        };


                        // let new_stack_part = self.reduce_and_goto(&peek, r.nonterminal_id, r.len);
                        if !new_stack_part.is_empty() {
                            reduced_stack.merge_with_depth(1, &new_stack_part);
                        }
                    }

                    if !reduced_stack.is_empty() {
                        // Deconstruct the result and put it back into the work map.
                        for new_peek in GSSNode::peek_iter(&Arc::new(reduced_stack)) {
                            let isolated = ParseState { stack: new_peek.isolated_parent() };
                            let new_depth = get_depth(&isolated.stack);
                            let new_state_id = new_peek.edge_value().state_id;
                            work_map.entry((new_depth, new_state_id))
                                .and_modify(|s| s.merge(isolated.clone()))
                                .or_insert(isolated);
                        }
                    }
                    });
                }

                if row.default_reduce.clone_and_merge {
                    next_active_state.merge(state);
                }
            }
        });

        crate::debug!(4, "Phase 3 completed, merging {} states into next active state", next_active_state.stack.num_predecessors());
        self.active_state = next_active_state;
        self.phase = ParserPhase::ReadyForToken;
        self.log_gss("Phase3-end", TerminalID(0), false, false); // Log with dummy token ID
    }

    pub fn has_action_for(&self, token_id: TerminalID) -> Option<LLMTokenBV> {
        if LR_MODE != LRMode::LR1 {
            return None;
        }
        if Some(token_id) == self.parser.ignore_terminal_id {
            timeit!("GLRParserState::has_action_for::ignore_token", {
                crate::debug!(4, "Ignoring token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());
                // return Some(self.active_state.stack.allowed_llm_tokens());
                return Some(LLMTokenBV::max_ones());
            });
        }
        // let mut hasher = DeterministicHasher::new(DefaultHasher::new());
        // self.active_state.hash(&mut hasher);
        // let self_hash = hasher.finish();
        // println!("GLRParserState::has_action_for: {:?}", self_hash);
        self.log_gss("has_action_for-start", token_id, false, false);
        let mut llm_tokens = LLMTokenBV::zeros();
        for peek in GSSNode::peek_iter(&self.active_state.stack) {
            let row = &self.parser.table[&peek.edge_value().state_id];
            let shifts_and_reduces = match self.phase {
                ParserPhase::ReadyForToken => &row.shifts_and_reduces_without_default_reduce,
                ParserPhase::ReadyForDefaultReductions => &row.shifts_and_reduces_full,
            };
            if let Some(action) = shifts_and_reduces.get(&token_id) {
                crate::debug!(4, "Found action for token '{}' in state {}: {:?}",
                              self.parser.terminal_map.get_by_right(&token_id).unwrap(),
                              peek.edge_value().state_id.0, action);
                // That's it! Since this is a LR(1) parser, it's enough to know that there's *any* action.
                timeit!("GLRParserState::has_action_for::action_found::add_llm_tokens", {
                    let peek_llm_tokens = timeit!(peek.resolved_llm_tokens_union());
                    timeit!(llm_tokens |= peek_llm_tokens);
                });
            } else {
                timeit!("GLRParserState::has_action_for::no_action_found", {
                    crate::debug!(4, "No action for token '{}' in state {}", self.parser.terminal_map.get_by_right(&token_id).unwrap(), peek.edge_value().state_id.0);
                });
            }
        }
        Some(llm_tokens)
    }

    #[time_it("GLRParserState::step")]
    pub fn step(&mut self, token_id: TerminalID) {
        self.process_token(token_id);
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

    pub fn merge_active_states(&mut self) {
        // No longer strictly necessary due to BTreeMap merge-on-insert, but GSS merge is explicit.
        // This method could be used if multiple GLRParserStates are combined.
    }

    pub fn merge_with(&mut self, mut other: GLRParserState) { // No longer generic
        assert!(std::ptr::eq(self.parser, other.parser));
        match (self.phase, other.phase) {
            (ParserPhase::ReadyForToken, ParserPhase::ReadyForDefaultReductions) => self.process_default_reductions(),
            (ParserPhase::ReadyForDefaultReductions, ParserPhase::ReadyForToken) => other.process_default_reductions(),
            _ => {},
        }
        self.active_state.merge(other.active_state);
        self.accepted |= other.accepted;
    }

    pub fn is_ok(&self) -> bool {
        self.accepted || (!self.active_state.stack.is_empty() && self.active_state.stack.is_alive())
    }

    /// Returns true if the previous step lead to an `accept` action.
    pub fn has_accepted(&self) -> bool {
        self.accepted
    }

    // #[time_it("GLRParserState::log_gss")]
    pub fn log_gss(&self, phase: &str, token: TerminalID, explain_states: bool, generate_dot: bool) {
        if !GSS_LOGGING_ENABLED {
            return;
        }
        // crate::debug!(3, "{} - token {} ({:?}) - nodes", phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0));
        const MAX: usize = 150;
        const PANIC_THRESHOLD: usize = 10000;

        let roots: Vec<_> = vec![self.active_state.stack.clone()];
        let stats = gather_gss_stats(&roots.iter().map(|r| r.as_ref()).collect::<Vec<_>>());
        crate::debug!(4, "{} ({:?}) - accepted: {} - token '{}' ({}) - nodes: {:?}",
                      phase, self.phase, self.accepted, self.parser.terminal_map.get_by_right(&token).unwrap(), token.0, stats);

        let make_msg = |print_full_forest, max_nodes_to_print| {
            let (gss_string, state_ids) = print_gss_forest(&roots, None, max_nodes_to_print, &self.parser.terminal_map, None, None);
            let mut final_string = if print_full_forest {
                format!("GSS ({} nodes):\n{}", stats.unique_nodes, gss_string)
            } else {
                match find_longest_path(&self.active_state.stack) {
                    Some(p) => format!("GSS too big ({} nodes). Longest path ({}): {}",
                                       stats.unique_nodes,
                                       p.len(),
                                       p.iter().map(|(ec, _n)| ec.state_id.0) // n is Arc<GSSNode>
                                            .map(|id| id.to_string())
                                            .collect::<Vec<_>>()
                                        .join(" → ")),
                None => format!("GSS too big ({} nodes) – path not found", stats.unique_nodes),
            }
        };

        if explain_states && !state_ids.is_empty() {
            final_string.push_str("\n\n--- GSS State Explanations ---\n");
                for state_id in state_ids {
                    let mut explanation = String::new();
                    writeln!(&mut explanation, "\n--- State {} ---", state_id.0).unwrap();
                    self.parser.format_state_details(&mut explanation, state_id, "  ").unwrap();
                    final_string.push_str(&explanation);
                }
            }
            final_string
        };

        if stats.unique_nodes > PANIC_THRESHOLD {
            let msg = make_msg(true, usize::MAX);
            panic!("GSS too big ({} nodes). {}", stats.unique_nodes, msg);
        }

        debug!(4, "{}", make_msg(stats.unique_nodes <= MAX, MAX));

        if generate_dot {
            let dot_string = self.gss_to_dot();
            // Log the DOT string. It can be copied into a .dot file and rendered with Graphviz.
            // e.g., `dot -Tpng -o gss.png gss.dot`
            crate::debug!(1, "GSS DOT graph:\n{}", dot_string);
        }
    }

    /// Generates a Graphviz DOT representation of the GSS state graph.
    pub fn gss_to_dot(&self) -> String {
        self.parser.gss_to_dot(&self.active_state.stack)
    }
}

impl GLRParser {
    /// Generates a Graphviz DOT representation of the state transitions present in a GSS forest.
    /// This visualizes the portion of the state machine explored by the parser.
    pub fn gss_forest_to_dot(&self, roots: &[(&str, &GSSNode)]) -> String {
        let mut dot = String::new();
        writeln!(&mut dot, "digraph GSS_Forest_States {{").unwrap();
        writeln!(&mut dot, "  rankdir=TB;").unwrap();
        writeln!(&mut dot, "  node [shape=box, fontname=\"Courier New\", style=rounded];").unwrap();

        let mut visited_nodes = HashSet::new();
        let mut defined_states = BTreeSet::new();
        let mut edges = BTreeSet::new();
        let mut root_labels: BTreeMap<StateID, Vec<String>> = BTreeMap::new();

        let mut queue = VecDeque::new();
        for (label, root) in roots {
            if !root.is_empty() {
                let root_arc = Arc::new((*root).clone());
                // Label the root states
                for (edge_val, _) in &root_arc.predecessors {
                    root_labels.entry(edge_val.state_id).or_default().push(label.to_string());
                }
                queue.push_back(root_arc);
            }
        }

        while let Some(node_arc) = queue.pop_front() {
            let node_ptr = Arc::as_ptr(&node_arc);
            if !visited_nodes.insert(node_ptr) {
                continue;
            }

            for (edge_val, preds_by_depth) in &node_arc.predecessors {
                let parent_state_id = edge_val.state_id;
                defined_states.insert(parent_state_id);

                for pred_vec in preds_by_depth.values() {
                    for pred_arc in pred_vec {
                        if pred_arc.predecessors.is_empty() {
                            edges.insert((Some(parent_state_id), None));
                        } else {
                            for (pred_edge_val, _) in &pred_arc.predecessors {
                                let child_state_id = pred_edge_val.state_id;
                                edges.insert((Some(parent_state_id), Some(child_state_id)));
                            }
                        }
                        queue.push_back(pred_arc.clone());
                    }
                }
            }
        }

        for state_id in &defined_states {
            let mut label = String::new();
            if let Some(labels) = root_labels.get(state_id) {
                let mut unique_labels: Vec<_> = labels.iter().cloned().collect();
                unique_labels.sort();
                unique_labels.dedup();
                writeln!(&mut label, "ROOTS: {}", unique_labels.join(", ")).unwrap();
            }
            self.format_state_details(&mut label, *state_id, "").unwrap();
            let escaped_label = label.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\l");
            writeln!(&mut dot, "  S{} [label=\"{}\"];", state_id.0, escaped_label).unwrap();
        }

        if edges.iter().any(|(_, to)| to.is_none()) {
            writeln!(&mut dot, "  START [shape=doublecircle, label=\"START\"];").unwrap();
        }

        for (from_opt, to_opt) in edges {
            match (from_opt, to_opt) {
                (Some(from), Some(to)) => writeln!(&mut dot, "  S{} -> S{};", from.0, to.0).unwrap(),
                (Some(from), None) => writeln!(&mut dot, "  S{} -> START;", from.0).unwrap(),
                _ => {}
            }
        }

        writeln!(&mut dot, "}}").unwrap();
        dot
    }

    /// Generates a Graphviz DOT representation of the state transitions present in a GSS.
    /// This visualizes the portion of the state machine explored by the parser.
    pub fn gss_to_dot(&self, root: &GSSNode) -> String {
        self.gss_forest_to_dot(&[("Root", root)])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
}

impl ParseState { // No longer generic
    pub fn merge(&mut self, mut other: ParseState) {
        // if self.stack.max_depth() > other.stack.max_depth() {
        //     std::mem::swap(self, &mut other);
        // }
        Arc::make_mut(&mut self.stack).merge_with_depth(1, &other.stack);
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
