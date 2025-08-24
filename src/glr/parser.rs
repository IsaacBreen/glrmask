use std::sync::{Mutex, RwLock};
use crate::datastructures::ArcPtrWrapper;
use std::any::Any;
use std::cmp::Ordering;
use crate::datastructures::gss::{print_gss_forest, Acc, GSSPopper, GSSPopperItem, GSSPrintConfig, PrecomputeNode2, PrecomputedNodeContents};
use crate::tokenizer::LLMTokenID;
use crate::datastructures::gss::{gather_gss_stats, find_longest_path, GSSNode, GSSStats, GSSPeek, LLMTokenBV};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Row, Stage7ShiftsAndReducesLookaheadValue, Table, StateID, TerminalID, SubstringGoto};
use crate::constraint::LLMVocab; // Import LLMTokenInfo

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use crate::debug;
use crate::profiler::GSS_LOGGING_ENABLED;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};
use crate::glr::automaton::compute_closure;
use std::collections::HashMap;
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::glr::table::{Reduce, ShiftsAndReducesWithoutDefaultReduce, ShiftsAndReducesFull, DefaultReduce, stage_9};
use crate::datastructures::trie::EdgeInserter;

// A single combined action for a given (state,row) and token:
// - Normal(...) is a concrete per-token action from the row's action map
// - Default(...) is the row's default reduction (token-independent)
#[derive(Debug)]
enum Action<'a> {
    Normal(&'a Stage7ShiftsAndReducesLookaheadValue),
    Default(&'a DefaultReduce),
}

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
pub struct ParseState {
    pub stack: Arc<GSSNode>,
    pub accepted_state: Arc<GSSNode>,
}

impl ParseState {
    pub fn new() -> Self {
        ParseState {
            stack: Arc::new(GSSNode::new_fresh()),
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
    }

    pub fn with_stack(stack: Arc<GSSNode>) -> Self {
        ParseState {
            stack,
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BelowBottomReductionMode {
    ContinueFromAll,
    ContinueFromEverything,
    Fail,
    #[default]
    Panic,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessTokenAdvancedConfig {
    pub below_bottom_mode: BelowBottomReductionMode,
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessDefaultReductionsAdvancedConfig {
    pub fuel: Option<usize>,
    pub per_state_fuel: Option<usize>,
    pub below_bottom_mode: BelowBottomReductionMode,
}

impl Default for ProcessDefaultReductionsAdvancedConfig {
    fn default() -> Self {
        Self {
            fuel: None,
            per_state_fuel: None,
            below_bottom_mode: BelowBottomReductionMode::default(),
        }
    }
}

#[derive(Clone)]
pub struct GLRParser {
    pub table: Table,
    pub productions: Vec<Production>,
    pub terminal_map: BiBTreeMap<Terminal, TerminalID>,
    pub non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    pub item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
    pub start_state_id: StateID,
    pub everything_state_id: StateID,
    pub ignore_terminal_id: Option<TerminalID>,
    // Precomputed tables for substring parsing reductions.
    pub(crate) substring_gotos: BTreeMap<NonTerminalID, SubstringGoto>,
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
        obj.insert("everything_state_id".to_string(), self.everything_state_id.to_json());
        obj.insert("ignore_terminal_id".to_string(), self.ignore_terminal_id.to_json());
        // Do not serialize precomputed substring gotos; they will be re-derived from the table.
        // Do not serialize self.actions
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let table = obj.remove("stage_7_table").ok_or_else(|| "Missing field stage_7_table".to_string())
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
                let everything_state_id = obj.remove("everything_state_id").ok_or_else(|| "Missing field everything_state_id".to_string())
                                        .and_then(StateID::from_json)?;
                let ignore_terminal_id = obj.remove("ignore_terminal_id")
                    .ok_or_else(|| "Missing field ignore_terminal_id for GLRParser".to_string())
                    .and_then(Option::<TerminalID>::from_json)?;

                let substring_gotos = stage_9(&table, &non_terminal_map);

                Ok(GLRParser {
                    table,
                    productions,
                    terminal_map,
                    non_terminal_map,
                    item_set_map,
                    start_state_id,
                    everything_state_id,
                    ignore_terminal_id,
                    substring_gotos,
                })
            }
            _ => Err("Expected JSONNode::Object for GLRParser".to_string()),
        }
    }
}

impl Debug for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLRParser")
            .field("table", &self.table)
            .field("productions", &self.productions)
            .field("terminal_map", &self.terminal_map)
            .field("non_terminal_map", &self.non_terminal_map)
            .field("item_set_map", &self.item_set_map)
            .field("start_state_id", &self.start_state_id)
            .field("everything_state_id", &self.everything_state_id)
            .field("ignore_terminal_id", &self.ignore_terminal_id)
            .field("substring_gotos_size", &self.substring_gotos.len())
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
        self.everything_state_id == other.everything_state_id &&
        self.ignore_terminal_id == other.ignore_terminal_id &&
        self.substring_gotos == other.substring_gotos
    }
}

impl Eq for GLRParser {}

impl GLRParser {
    pub fn new(
        table: Table,
        productions: Vec<Production>,
        terminal_map: BiBTreeMap<Terminal, TerminalID>,
        non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
        item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
        start_state_id: StateID,
        everything_state_id: StateID,
        actions: BTreeMap<NonTerminal, ActionFn>, // Parameter type
        ignore_terminal_id: Option<TerminalID>,
        substring_gotos: BTreeMap<NonTerminalID, SubstringGoto>,
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
            table,
            productions,
            terminal_map,
            non_terminal_map,
            item_set_map,
            start_state_id,
            everything_state_id,
            ignore_terminal_id,
            substring_gotos,
        }
    }

    pub fn init_glr_parser(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        self.init_glr_parser_with_acc()
    }

    pub fn init_glr_parser_with_stack(&self, stack: ParseState) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: stack,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }

    pub fn init_glr_parser_null(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: ParseState::new(),
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }

    pub fn init_glr_parser_with_acc(&self) -> GLRParserState { // No longer generic
        let initial_parse_state = self.init_parse_state_with_acc();
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        };
        parser_state
    }

    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState { // No longer generic
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: parse_state,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        };
        parser_state
    }

    /// Initializes a substring parser state. This seeds the active GSS with
    /// every state in the LR automaton as the top-of-stack, enabling parsing
    /// to begin at any context simultaneously (substring recognition).
    ///
    /// This corresponds to starting “for each state directly reachable”
    /// simultaneously (as per Rekers/Koorn, 4.3), but since we do not know
    /// the first token a priori, we simply include all table states here.
    pub fn init_glr_substring_parser_with_all_states(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_substring_with_all_states();
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            accepted: false,
        }
    }

    /// Builds a parse state whose stack top has a predecessor edge for each table state.
    /// The effect is “parser is in all states at once” at depth 1.
    pub fn init_parse_state_substring_with_all_states(&self) -> ParseState {
        let all_edges: Vec<ParseStateEdgeContent> = self.table
            .keys()
            .map(|sid| ParseStateEdgeContent { state_id: *sid })
            .collect();
        let stack_top = GSSNode::new_fresh().push_many(all_edges);
        ParseState {
            stack: Arc::new(stack_top),
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
    }

    pub fn init_glr_substring_parser_with_everything_state(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_substring_with_everything_state();
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            accepted: false,
        }
    }

    /// Builds a parse state whose stack top has a single predecessor edge for the 'everything' state.
    /// The effect is “parser is in all states at once” at depth 1.
    pub fn init_parse_state_substring_with_everything_state(&self) -> ParseState {
        let initial_content = ParseStateEdgeContent {
            state_id: self.everything_state_id,
        };
        let stack = Arc::new(GSSNode::new_fresh().push(initial_content));
        ParseState {
            stack,
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
    }

    pub fn init_parse_state(&self, llm_vocab: Option<Arc<LLMVocab>>) -> ParseState { // No longer generic
        self.init_parse_state_with_acc()
    }

    pub fn init_parse_state_with_acc(&self) -> ParseState { // No longer generic
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
        };
        ParseState {
            stack: Arc::new(GSSNode::new_fresh().push(initial_content)), // pushed node has initial_acc
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
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
                    write!(f, ", ")?;
                    // Display the lookahead
                    if lookaheads.len() == 1 {
                        if let Some(lookahead) = lookaheads.iter().next().unwrap() {
                            write!(f, "{}", lookahead)?;
                        } else {
                            write!(f, "ε")?; // Epsilon for no lookahead
                        }
                    } else {
                        write!(f, "{{")?;
                        let mut lookahead_strs: Vec<String> = lookaheads.iter().map(|l| if let Some(t) = l { t.to_string() } else { "ε".to_string() }).collect();
                        lookahead_strs.sort();
                        const MAX_LOOKAHEADS_TO_SHOW: usize = 5;
                        if lookahead_strs.len() > MAX_LOOKAHEADS_TO_SHOW {
                            let truncated: Vec<_> = lookahead_strs.iter().take(MAX_LOOKAHEADS_TO_SHOW).cloned().collect();
                            write!(f, "{}... ({} total)", truncated.join(", "), lookahead_strs.len())?;
                        } else {
                            write!(f, "{}", lookahead_strs.join(", "))?;
                        }
                        writeln!(f, "}}")?;
                    }
                    writeln!(f, "]")?;
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
                let nt_name = self.non_terminal_map.get_by_right(&reduce_action.0.nonterminal_id).unwrap();
                let pids: Vec<String> = reduce_action.0.production_ids.iter().map(|p| p.0.to_string()).collect();
                writeln!(f, "{}  - Default Reduce {} (len {}) via rules [{}]", indent, nt_name.0, reduce_action.0.len, pids.join(", "))?;
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

    // Sort by terminal name for deterministic output
    let mut sorted_actions: Vec<_> = actions.iter().collect();
    sorted_actions.sort_by_key(|(tid, _)| terminal_map.get_by_right(tid).unwrap());

    for (tid, action) in sorted_actions {
        let terminal = terminal_map.get_by_right(tid).unwrap();

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
                let has_shift = shift.is_some();
                let num_reduces: usize = reduces.values().map(|nts| nts.values().map(|pids| pids.len()).sum::<usize>()).sum();
                let conflict_type = if has_shift && num_reduces > 0 {
                    "Shift-Reduce Conflict"
                } else if !has_shift && num_reduces > 1 {
                    "Reduce-Reduce Conflict"
                } else {
                    "Conflict" // Should be simplified away
                };

                let mut s = format!("{}:", conflict_type);
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
        writeln!(f, "{}- On {}: {}", indent, terminal, action_str)?;
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

        writeln!(f, "\nSubstring Gotos ({} entries):", self.substring_gotos.len())?;
        if !self.substring_gotos.is_empty() {
            // Sort by NT name for deterministic output
            let mut sorted_substring_gotos: Vec<_> = self.substring_gotos.iter().collect();
            sorted_substring_gotos.sort_by_key(|(nt_id, _)| self.non_terminal_map.get_by_right(nt_id).unwrap());

            for (nt_id, gotos) in sorted_substring_gotos {
                let nt = self.non_terminal_map.get_by_right(nt_id).unwrap();
                writeln!(f, "  - For NT '{}' (ID {}):", nt.0, nt_id.0)?;
                if !gotos.accepting_sources.is_empty() {
                    let sources: Vec<String> = gotos.accepting_sources.iter().map(|s| s.0.to_string()).collect();
                    writeln!(f, "    - accepting sources: [{}]", sources.join(", "))?;
                }
                let mut sorted_gotos_by_dest: Vec<_> = gotos.gotos.iter().collect();
                sorted_gotos_by_dest.sort_by_key(|(k, _)| *k);

                for (goto_id, source_ids) in sorted_gotos_by_dest {
                    let sources: Vec<String> = source_ids.iter().map(|s| s.0.to_string()).collect();
                    writeln!(
                        f,
                        "    - goto: {:<4} from sources: [{}]",
                        goto_id.0,
                        sources.join(", ")
                    )?;
                }
            }
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
    below_bottom_cache: HashMap<BelowBottomCacheKey, HashMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BelowBottomCacheKey {
    nonterminal_id: NonTerminalID,
    source_state_id: StateID,
    goto_state_id: StateID,
    k: usize,
    // Important: this Acc must have trie2_nodes cleared before being placed here.
    // acc: Acc,
}

impl Display for GLRParserState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO: this is bad. make this better
        // Display the stack
        self.log_gss("    ", TerminalID(0), false, false);
        Ok(())
    }
}

// Key is (depth, state_id) to process stacks in a specific order.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct WorkMapKey(usize, StateID);

impl PartialOrd for WorkMapKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkMapKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This ordering is chosen for performance. It processes deeper stacks first.
        // The idea is that deeper stacks are more constrained and processing them
        // first might lead to quicker pruning of invalid paths.
        // Sorting by depth descending, then by state_id ascending.
        let WorkMapKey(self_depth, self_state_id) = self;
        let WorkMapKey(other_depth, other_state_id) = other;
        other_depth.cmp(&self_depth).then_with(|| self_state_id.cmp(&other_state_id))
        // self_depth.cmp(&other_depth).then_with(|| self_state_id.cmp(&other_state_id))
        // self_state_id.cmp(&other_state_id).then_with(|| other_depth.cmp(&self_depth))
        // self_state_id.cmp(&other_state_id).then_with(|| self_depth.cmp(&other_depth))
    }
}

type WorkMap = BTreeMap<WorkMapKey, (ParseState, Option<usize>)>;

impl<'a> GLRParserState<'a> { // No longer generic
    fn enqueue(work_map: &mut WorkMap, state: ParseState, fuel: Option<usize>) {
        // Peel off the top edges of the GSS in the given state,
        // and group the resulting isolated paths by their (depth, state_id) key.
        // This merges paths that are in the same logical state, reducing redundant processing.
        for peek in GSSNode::peek_iter(&state.stack) {
            let isolated_state = ParseState {
                stack: peek.isolated_parent(),
                accepted_state: state.accepted_state.clone(),
            };
            let depth = isolated_state.stack.max_depth();
            let state_id = peek.edge_value().state_id;
            work_map
                .entry(WorkMapKey(depth, state_id))
                .and_modify(|(s, existing_fuel)| {
                    s.merge(isolated_state.clone());
                    *existing_fuel = std::cmp::max(*existing_fuel, fuel);
                })
                .or_insert((isolated_state, fuel));
        }
    }

    fn push_state(
        &self,
        peek: &GSSPeek,
        new_content: ParseStateEdgeContent,
    ) -> ParseState {
        crate::debug!(4, "Pushing new state with content: {:?}", new_content);
        let new_gss_node_instance = peek.push_on_parent(new_content);
        ParseState {
            stack: Arc::new(new_gss_node_instance),
            accepted_state: self.active_state.accepted_state.clone(),
        }
    }

    /// Shared inner loop for phase 1 and phase 2.
    /// `action_selector` chooses between the phase-1 or phase-2 action map.
    fn process_action_queue<F>(
        &mut self,
        work_map: &mut WorkMap,
        mut reduce_map: Option<&mut WorkMap>,
        shifted_states_todo: &mut VecDeque<ParseState>,
        accepted_states_todo: &mut VecDeque<ParseState>,
        action_selector: F,
        config: &ProcessTokenAdvancedConfig,
        fuel: &mut Option<usize>,
    )
    where
        F: for<'r> Fn(&'r Row) -> Option<Action<'r>>,
    {
        assert!(fuel.is_none(), "Fuel is not supported in process_action_queue yet");
        for (state, per_state_fuel) in work_map.values() {
            assert!(per_state_fuel.is_none(), "Per-state fuel is not supported in process_action_queue yet");
        }
        while let Some(entry) = work_map.pop_first() {
            let (key, (state, per_state_fuel)) = entry;
            if let Some(f) = fuel {
                if *f == 0 {
                    // Out of fuel. Put the state back and return.
                    work_map.insert(key, (state, per_state_fuel));
                    return;
                }
                *f -= 1;
            }
            let WorkMapKey(_depth, state_id) = key;
            let row = &self.parser.table[&state_id];
            let action_opt = action_selector(row);
            if let Some(action) = action_opt {
                for peek in GSSNode::peek_iter(&state.stack) {
                    match action {
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Shift(to)) => {
                            crate::debug!(5, "Action: Shift to state {}", to.0);
                            let new_parse_state =
                                self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                            shifted_states_todo.push_back(new_parse_state);
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                            nonterminal_id: nt,
                            len,
                            ..
                        }) => {
                            if per_state_fuel == Some(0) { continue; }
                            let new_per_state_fuel = per_state_fuel.map(|f| f - 1);

                            crate::debug!(5, "Action: Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), len);
                            let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(&peek, *nt, *len, &action_selector, config);
                            if !s_new_arc.is_empty() {
                                let new_parse_state = ParseState {
                                    stack: s_new_arc,
                                    accepted_state: state.accepted_state.clone(),
                                };
                                if let Some(ref mut r_map) = reduce_map {
                                    Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                } else {
                                    Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                }
                            }
                            if !accepted_s_new_arc.is_empty() {
                                self.accepted = true;
                                let accepted_parse_state = ParseState {
                                    stack: Arc::new(GSSNode::new_fresh()),
                                    accepted_state: accepted_s_new_arc,
                                };
                                accepted_states_todo.push_back(accepted_parse_state);
                            }
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces }) => {
                            crate::debug!(5, "Action: Split with shift and reduces");
                            if let Some(to) = shift {
                                crate::debug!(5, "Action (Split): Shift to state {}", to.0);
                                let new_parse_state =
                                    self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                                shifted_states_todo.push_back(new_parse_state);
                            }
                            if per_state_fuel != Some(0) {
                                let new_per_state_fuel = per_state_fuel.map(|f| f - 1);
                                for (len, nts) in reduces {
                                    for (nt, _prod_ids) in nts {
                                        crate::debug!(5, "Action (Split): Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), *len);
                                        let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(&peek, *nt, *len, &action_selector, config);
                                        if !s_new_arc.is_empty() {
                                            let new_parse_state = ParseState {
                                                stack: s_new_arc,
                                                accepted_state: state.accepted_state.clone(),
                                            };
                                            if let Some(ref mut r_map) = reduce_map {
                                                Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                            } else {
                                                Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                            }
                                        }
                                        if !accepted_s_new_arc.is_empty() {
                                            self.accepted = true;
                                            let accepted_parse_state = ParseState {
                                                stack: Arc::new(GSSNode::new_fresh()),
                                                accepted_state: accepted_s_new_arc,
                                            };
                                            accepted_states_todo.push_back(accepted_parse_state);
                                        }
                                    }
                                }
                            }
                        }
                        Action::Default(default_reduce) => {
                            // 1) If clone_and_merge is set, add the "current stuff" (not the reduce result) to the shifted queue.
                            if default_reduce.clone_and_merge {
                                shifted_states_todo.push_back(state.clone());
                            }

                            // 2) If there's a reduction in the default, do it like a normal reduce.
                            if let Some((reduce, allowed_terminals)) = &default_reduce.reduce {
                                if per_state_fuel != Some(0) {
                                    let new_per_state_fuel = per_state_fuel.map(|f| f - 1);
                                    let mut constrained_state = state.clone();
                                    let can_proceed = constrained_state.stack.is_alive();

                                    if can_proceed {
                                        let disallowed_terminals_bv = allowed_terminals.inverted();
                                        if !disallowed_terminals_bv.is_empty() {
                                            let disallowed_l2 = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::from_iter(
                                                std::iter::once((0..=usize::MAX, disallowed_terminals_bv))
                                            );

                                            crate::datastructures::gss::disallow_terminals_and_prune_arc(
                                                &mut constrained_state.stack,
                                                &disallowed_l2,
                                                &mut HashMap::new(),
                                            );
                                        }

                                        if !constrained_state.stack.is_empty() {
                                            for peek in GSSNode::peek_iter(&constrained_state.stack) {
                                                let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(
                                                    &peek,
                                                    reduce.nonterminal_id,
                                                    reduce.len,
                                                    &action_selector,
                                                    config,
                                                );
                                                if !s_new_arc.is_empty() {
                                                    let new_parse_state = ParseState {
                                                        stack: s_new_arc,
                                                        accepted_state: state.accepted_state.clone(),
                                                    };
                                                    if let Some(ref mut r_map) = reduce_map {
                                                        Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                                    } else {
                                                        Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                                    }
                                                }
                                                if !accepted_s_new_arc.is_empty() {
                                                    self.accepted = true;
                                                    let accepted_parse_state = ParseState {
                                                        stack: Arc::new(GSSNode::new_fresh()),
                                                        accepted_state: accepted_s_new_arc,
                                                    };
                                                    accepted_states_todo.push_back(accepted_parse_state);
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                // No reduction in default: we already handled clone_and_merge above.
                                // Nothing else to do for this action.
                            }
                        }
                    }
                }
            } else {
                crate::debug!(5, "No action found in state {}", state_id.0);
            }
        }
    }

    fn _do_actions_without_default(&mut self, token_id: TerminalID, phase1_todo: &mut WorkMap, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, accepted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig) {
        let token_display = self.parser.terminal_map.get_by_right(&token_id).unwrap();
        crate::debug!(4, "Phase 1: Processing token '{}'", token_display);
        timeit!("GLRParserState::step::phase1", {
            let tid = token_id;
            self.process_action_queue(
                phase1_todo,
                Some(phase2_todo),
                shifted_states_todo,
                accepted_states_todo,
                move |row| {
                    row.shifts_and_reduces_without_default_reduce
                        .get(&tid)
                        .map(Action::Normal)
                },
                config,
                &mut None,
            );
        });
    }

    fn _do_actions_with_default(&mut self, token_id: TerminalID, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, accepted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig) {
        crate::debug!(4, "Phase 1 completed, proceeding to Phase 2 with {} shifted states", shifted_states_todo.len());
        timeit!("GLRParserState::step::phase2", {
            // Reduces are pushed back onto the same queue (`None`).
            let tid = token_id;
            self.process_action_queue(
                phase2_todo,
                None,
                shifted_states_todo,
                accepted_states_todo,
                move |row| {
                    // Prefer a concrete token action; otherwise use the default reduce.
                    row.shifts_and_reduces_full
                        .get(&tid)
                        .map(Action::Normal)
                },
                config,
                &mut None,
            );
            self.phase = ParserPhase::ReadyForDefaultReductions;
        });
    }

    #[time_it("GLRParserState::reduce_and_goto")]
    fn reduce_and_goto<G>(
        &mut self,
        peek: &GSSPeek,
        nt: NonTerminalID,
        len: usize,
        action_selector: &G,
        config: &ProcessTokenAdvancedConfig,
    ) -> (Arc<GSSNode>, Arc<GSSNode>)
    where
        G: for<'r> Fn(&'r Row) -> Option<Action<'r>>,
    {
        let popper: GSSPopper = timeit!(peek.popn(len));
        crate::debug!(4, "Reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
        crate::debug!(4, "Popped with {} results...", popper.num_predecessors());
        let mut any_below_bottom = !popper.below_bottom().is_empty();
        // timeit!(format!("GLRParserState::reduce_and_goto reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len), {});
        // timeit!(format!("GLRParserState::reduce_and_goto reducing with len {}", len), {});

        let mut out: Vec<Arc<GSSNode>> = Vec::new();
        let mut accepted_out: Vec<Arc<GSSNode>> = Vec::new();
        for popper_item in popper.iter() {
            for peek2 in popper_item.peek_iter() {
                let predecessor_state_id = peek2.edge_value().state_id;
                let mut current_nt = nt;

                // Fast loop for unit reduction chains based on the current lookahead token.
                let mut i = 0;
                loop {
                    i += 1;
                    let goto = self.parser.table.get(&predecessor_state_id).and_then(|row| row.gotos.get(&current_nt)).expect_else(|| {
                        format!("Goto not found for NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id)
                    });

                    if goto.accept {
                        crate::debug!(4, "Accepting with NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id);

                        // Add the stack with the reduced state (predecessor_state_id) at the top to the accepted_state accumulator.
                        let accepted_stack_instance = peek2.isolated_parent();
                        accepted_out.push(accepted_stack_instance);
                    }

                    if let Some(goto_state_id) = goto.state_id {
                        let next_row = &self.parser.table[&goto_state_id];
                        match action_selector(next_row) {
                            Some(Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: next_nt, len: 1, .. })) => {
                                // Token-based unit reduction: continue the chain.
                                current_nt = *next_nt;
                                continue;
                            }
                            Some(Action::Default(def)) => {
                                // Default reduction handling.
                                // If clone_and_merge or reduce.len != 1 is set, we "submit" the current goto result now,
                                // as if we broke the chain here, but we may still continue chaining if allowed.
                                if def.clone_and_merge || def.reduce.as_ref().map_or(false, |r| r.0.len != 1) {
                                    out.push(Arc::new(peek2.push_on_parent(ParseStateEdgeContent { state_id: goto_state_id })));
                                }

                                match &def.reduce {
                                    Some(reduce) if reduce.0.len == 1 => {
                                        current_nt = reduce.0.nonterminal_id;
                                        continue;
                                    }
                                    _ => break,
                                }
                            }
                            _ => {
                                // Not a unit reduction (could be shift, split, non-matching reduce, or no action):
                                // finalize at this GOTO and stop chaining.
                                out.push(Arc::new(peek2.push_on_parent(ParseStateEdgeContent { state_id: goto_state_id })));
                                break;
                            }
                        }
                    } else {
                        // No further state to go to. This path terminates here.
                        // timeit!(format!("Exloring path. Reason: No goto state found for NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id), {});
                        break; // Exit the fast loop for this path
                    }
                }
                // Round to nearest power of 2
                let i_rounded_to_nearest_pow = if i == 0 {
                    1
                } else {
                    1 << (32 - (i as u32 - 1).leading_zeros())
                };
 
                // timeit!(format!("GLRParserState::step::phase2::goto::number of loops (rounded to nearest pow of 2): {}", i_rounded_to_nearest_pow), {});
            }
        }
 
        // Handle “popped below bottom” cases:
        //
        // If the reduction pops below the bottom, we have recognized only the
        // suffix β of a rule A ::= α β. Per substring parsing semantics,
        // α lies before the substring start and must be considered unknown (but derivable),
        // so we continue in every state that has a GOTO on A. We also merge the Acc
        // accumulated along these paths to create a new virtual root to push onto.
        // timeit!(format!("GLRParserState::reduce_and_goto: Handling popped below bottom cases for NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len), {
        timeit!("GLRParserState::reduce_and_goto: Handling popped below bottom cases", {
        let mut gotos_for_nt_storage = SubstringGoto::default();
        let empty_substring_goto = SubstringGoto::default();
        if any_below_bottom {
            let mut below_bottom_integrated: BTreeMap<usize, Acc> = BTreeMap::new();
            for (k, accs_by_edge) in popper.below_bottom().iter() {
                for (last_edge_content, acc_arc) in accs_by_edge {
                    let acc = acc_arc.as_ref();
                    let state_id_edge_key = (0, Some(last_edge_content.state_id));

                    let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> = BTreeMap::new();
                    let new_trie2_node = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));

                    for existing_trie2_node in &acc.trie2_nodes {
                        let source_arc = existing_trie2_node.as_arc().clone();
                        let source_live = { source_arc.read().expect("poison").value.live_tokens.clone() };
                        let tokens_to_push = &source_live & &acc.llm_tokens_union;
                        if tokens_to_push.is_empty() { continue; }

                        let mut inserter = EdgeInserter::new(
                            source_arc.clone(),
                            state_id_edge_key,
                            tokens_to_push.clone(),
                            |e, n| *e |= n,
                            |node_value, edge_value| {},
                            |ev, t| *ev &= &t.live_tokens,
                        );
                        inserter = inserter.try_destination_auto(new_trie2_node.clone());
                        let final_dest_arc = inserter.clone_into_option().expect("GLRParserState::reduce_and_goto: EdgeInserter failed");
                        let final_dest_wr = ArcPtrWrapper::new(final_dest_arc.clone());
                        dest_agg.entry(final_dest_wr.clone()).and_modify(|bv| *bv |= &tokens_to_push).or_insert(tokens_to_push.clone());
                    }

                    for (dst_wr, added) in &dest_agg {
                        let mut dg = dst_wr.as_arc().write().expect("poison");
                        dg.value.live_tokens |= added.clone();
                    }

                    let mut integrated_acc = acc.clone();
                    integrated_acc.trie2_nodes = dest_agg.keys().cloned().collect();
                    below_bottom_integrated.entry(*k).and_modify(|existing| *existing = Acc::merge(existing, &integrated_acc)).or_insert(integrated_acc);
                }
            }

                let gotos_for_nt: &SubstringGoto = match config.below_bottom_mode {
                    BelowBottomReductionMode::ContinueFromAll => {
                        crate::debug!(5, "Handling popped below bottom cases for NT '{}' and len {} with ContinueFromAll", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
                        self.parser.substring_gotos.get(&nt).unwrap_or(&empty_substring_goto)
                    }
                    BelowBottomReductionMode::ContinueFromEverything => {
                        crate::debug!(5, "Handling popped below bottom cases for NT '{}' and len {} with ContinueFromEverything", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
                        let everything_state_id = self.parser.everything_state_id;
                        if let Some(goto) = self.parser.table.get(&everything_state_id).and_then(|row| row.gotos.get(&nt)) {
                            let mut compacted = SubstringGoto::default();
                            if goto.accept {
                                compacted.accepting_sources.insert(everything_state_id);
                            }
                            if let Some(goto_state_id) = goto.state_id {
                                compacted.gotos.insert(goto_state_id, BTreeSet::from([everything_state_id]));
                            }
                            gotos_for_nt_storage = compacted;
                            &gotos_for_nt_storage
                        } else {
                            &empty_substring_goto
                        }
                    }
                    BelowBottomReductionMode::Fail => {
                        crate::debug!(5, "Popped below bottom, failing these parse paths.");
                        &empty_substring_goto
                    }
                    BelowBottomReductionMode::Panic => {
                        panic!("A reduction popped below the bottom of the stack, and BelowBottomReductionMode was set to Panic.");
                    }
                };

                let num_total_gotos = gotos_for_nt.gotos.values().map(|s| s.len()).sum::<usize>();
                let num_accepting_sources = gotos_for_nt.accepting_sources.len();
                let num_unique_destinations = gotos_for_nt.gotos.len();
                let num_shared_destinations = gotos_for_nt.gotos.values().filter(|s| s.len() > 1).count();

                let num_with_action = gotos_for_nt.gotos.keys().filter(|sid| {
                    self.parser.table.get(sid).map_or(false, |row| action_selector(row).is_some())
                }).count();

                // println!(
                //     "Popped below bottom: NT '{}', len {}. Substring GOTO stats (total gotos {}, accepting sources {}): unique_dests={}, shared_dests={}, with_action={}",
                //     self.parser.non_terminal_map.get_by_right(&nt).unwrap(),
                //     len,
                //     num_total_gotos,
                //     num_accepting_sources,
                //     num_unique_destinations,
                //     num_shared_destinations,
                //     num_with_action
                // );
                if !gotos_for_nt.accepting_sources.is_empty() || !gotos_for_nt.gotos.is_empty() {
                    timeit!("GLRParserState::reduce_and_goto: Processing accepting gotos", {
                    let accepting_sources = &gotos_for_nt.accepting_sources;
                    if !accepting_sources.is_empty() {
                        let mut accepted_stacks = Vec::new();
                        for (k, mut acc) in below_bottom_integrated.iter().map(|(k, acc)| (k, acc.clone())) {
                            let trie2_nodes = std::mem::take(&mut acc.trie2_nodes);
                            let new_trie2_node = Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal())));
                            let active_llm_tokens = acc.union_llm_tokens();
                            for source_state_id in accepting_sources {
                                let edge_key = (*k, Some(*source_state_id));
                                // let edge_key = (*k, None);
                                let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> = BTreeMap::new();
                                let mut used_dests = BTreeSet::new();

                                for existing_trie2_node in &trie2_nodes {
                                    let source_arc = existing_trie2_node.as_arc().clone();
                                    let source_live = { source_arc.read().expect("poison").value.live_tokens.clone() };
                                    let tokens_to_push = &source_live & &active_llm_tokens;
                                    if tokens_to_push.is_empty() { continue; }

                                    // Build an iterator of all eligible strong children under edge_key
                                    let eligible_iter_builder = || {
                                        let g = source_arc.read().expect("poison");
                                        let mut v = Vec::new();
                                        if let Some(dest_map) = g.children().get(&edge_key) {
                                            for (node_ptr, _ev) in dest_map.iter() {
                                                if !node_ptr.is_strong() { continue; }
                                                if let Some(dest_arc) = node_ptr.upgrade() {
                                                    let dl = dest_arc.read().expect("poison").value.live_tokens.clone();
                                                    if (&dl & &tokens_to_push).is_empty() && !dest_arc.read().expect("poison").value.end {
                                                        v.push(dest_arc.clone());
                                                    }
                                                }
                                            }
                                        }
                                        v.into_iter()
                                    };

                                    let mut inserter = EdgeInserter::new(
                                        source_arc.clone(),
                                        edge_key,
                                        tokens_to_push.clone(),
                                        |e, n| *e |= n,
                                        |node_value, edge_value| {},
                                        |ev, t| *ev &= &t.live_tokens,
                                    ).try_destinations_iter_with(eligible_iter_builder);

                                    inserter = inserter.try_destination_auto(new_trie2_node.clone());

                                    let final_dest_arc = inserter.clone_into_option().expect("GLRParserState::reduce_and_goto: EdgeInserter failed");
                                    let final_dest_wr = ArcPtrWrapper::new(final_dest_arc.clone());
                                    dest_agg.entry(final_dest_wr.clone()).and_modify(|bv| *bv |= &tokens_to_push).or_insert(tokens_to_push.clone());
                                    used_dests.insert(final_dest_wr);
                                }

                                for (dst_wr, added) in &dest_agg {
                                    let mut dg = dst_wr.as_arc().write().expect("poison");
                                    dg.value.live_tokens |= added.clone();
                                }

                                let mut acc2 = acc.clone();
                                acc2.trie2_nodes = used_dests.clone();
                                let new_gss0 = GSSNode::new(acc2);
                                let new_gss1 = new_gss0.push(ParseStateEdgeContent { state_id: *source_state_id });
                                accepted_stacks.push(Arc::new(new_gss1));
                            }
                        }
                        let merged_accepted = GSSNode::merge_many_with_depth(usize::MAX, accepted_stacks);
                        accepted_out.push(merged_accepted);
                    }
                    });

                    // THIS is where the program spends almost all its compute time
                    timeit!("GLRParserState::reduce_and_goto: Processing non-accepting gotos", {
                    // timeit!(format!("GLRParserState::reduce_and_goto: Popped below bottom cases for NT '{}' and len {}, number of imagined reduces: {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len, gotos_for_nt.len()), {});
                    let mut below_zero = Vec::new();

                    let mut trie2_dst_nodes = HashMap::new();
                    for (k, mut acc) in below_bottom_integrated {
                        let trie2_nodes = std::mem::take(&mut acc.trie2_nodes);
                        timeit!(format!("GLRParserState::reduce_and_goto: Processing pop below"), {});
                        let edge_key = (k, None);
                        for (goto_state_id, source_state_ids) in &gotos_for_nt.gotos {
                            for source_state_id in source_state_ids {
                                // Key that ignores trie2_nodes (they are already cleared from 'acc' by std::mem::take above)
                                let cache_key = BelowBottomCacheKey {
                                    // nonterminal_id: nt,
                                    nonterminal_id: NonTerminalID(0),
                                    source_state_id: *source_state_id,
                                    // source_state_id: StateID(0),
                                    // goto_state_id: *goto_state_id,
                                    goto_state_id: StateID(0),
                                    k: 0,
                                    // k,
                                    // acc: acc.clone(),
                                };

                                let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> = BTreeMap::new();
                                let mut used_dests: BTreeSet<ArcPtrWrapper<RwLock<PrecomputeNode2>>> = BTreeSet::new();

                                timeit!("GLRParserState::reduce_and_goto::BLOCK_1: Below-bottom reduction goto processing", {
                                let new_trie2_node = trie2_dst_nodes
                                    .entry(*source_state_id)
                                    .or_insert_with(|| Arc::new(RwLock::new(PrecomputeNode2::new(PrecomputedNodeContents::internal()))))
                                    .clone();

                                for existing_trie2_node in &trie2_nodes {
                                    let mut inserter;
                                    let tokens_to_push;
                                    let source_arc;

                                    timeit!("GLRParserState::reduce_and_goto::BLOCK_1::BLOCK_1: Below-bottom reduction goto processing", {
                                    source_arc = existing_trie2_node.as_arc().clone();
                                    let source_live = { source_arc.read().expect("poison").value.live_tokens.clone() };
                                    tokens_to_push = &source_live & &acc.llm_tokens_union;
                                    if tokens_to_push.is_empty() { continue; }

                                    inserter = EdgeInserter::new(
                                        source_arc.clone(),
                                        edge_key,
                                        tokens_to_push.clone(),
                                        |e, n| *e |= n,
                                        |node_value, edge_value| {},
                                        |ev, t| *ev &= &t.live_tokens,
                                    );
                                    });

                                    timeit!("GLRParserState::reduce_and_goto::BLOCK_1::BLOCK_1.3: Below-bottom reduction goto processing", {
                                    if let Some(cached_entries) = self.below_bottom_cache.get(&cache_key) {
                                        let eligible_cached_destinations: Vec<_> = cached_entries.iter().filter_map(|(wrapper, cached_tokens)| {
                                            let dest_arc = wrapper.as_arc();
                                            let guard = dest_arc.read().expect("poison");
                                            let temp = &guard.value.live_tokens - &cached_tokens;
                                            if (&temp & &tokens_to_push).is_empty() && !guard.value.end {
                                                // crate::debug!(6, "Using cached destination in below-bottom reduction for NT '{}' and len {}: {:?}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len, wrapper);
                                                timeit!("GLRParserState::reduce_and_goto::BLOCK_1::BLOCK_1.3::Using cached destination", {});
                                                Some(dest_arc.clone())
                                            } else {
                                                // crate::debug!(6, "Skipping cached destination in below-bottom reduction for NT '{}' and len {}: {:?}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len, wrapper);
                                                None
                                            }
                                        }).collect();
                                        inserter = inserter.to_destinations_weakly_iter(eligible_cached_destinations.into_iter());
                                    }
                                    });
                                    let eligible_iter_builder;
                                    timeit!("GLRParserState::reduce_and_goto::BLOCK_1::BLOCK_3: Below-bottom reduction goto processing", {

                                    eligible_iter_builder = || {
                                        let g = source_arc.read().expect("poison");
                                        let mut v = Vec::new();
                                        if let Some(dest_map) = g.children().get(&edge_key) {
                                            for (node_ptr, _ev) in dest_map.iter() {
                                                if !node_ptr.is_strong() { continue; }
                                                if let Some(dest_arc) = node_ptr.upgrade() {
                                                    let dest_guard = dest_arc.read().expect("poison");
                                                    if (&dest_guard.value.live_tokens & &tokens_to_push).is_empty() && !dest_guard.value.end {
                                                        v.push(dest_arc.clone());
                                                    }
                                                }
                                            }
                                        }
                                        v.into_iter()
                                    };
                                    });
                                    timeit!("GLRParserState::reduce_and_goto::BLOCK_1::BLOCK_4: Below-bottom reduction goto processing", {

                                    inserter = inserter.try_destinations_iter_with(eligible_iter_builder);
                                    inserter = inserter.try_destination_auto(new_trie2_node.clone());

                                    });
                                    timeit!("GLRParserState::reduce_and_goto::BLOCK_1::BLOCK_5: Below-bottom reduction goto processing", {

                                    let final_dest_arc = inserter.clone_into_option().expect("GLRParserState::reduce_and_goto: EdgeInserter failed");
                                    let final_dest_wr = ArcPtrWrapper::new(final_dest_arc.clone());
                                    dest_agg.entry(final_dest_wr.clone()).and_modify(|bv| *bv |= &tokens_to_push).or_insert(tokens_to_push.clone());
                                    });
                                }
                                });
                                timeit!("GLRParserState::reduce_and_goto::BLOCK_2: Below-bottom reduction goto processing", {

                                // Update the cache and populate used_dests
                                let cache_entry = self.below_bottom_cache.entry(cache_key).or_default();
                                for (dest_wrapper, new_tokens) in &dest_agg {
                                    if let Some(existing_tokens) = cache_entry.get(dest_wrapper) {
                                        if !new_tokens.is_subset(existing_tokens) {
                                            crate::debug!(6, "Updating cache for below-bottom reduction for NT '{}' and len {}: {:?}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len, dest_wrapper);
                                            used_dests.insert(dest_wrapper.clone());
                                        } else {
                                            crate::debug!(6, "Not updating cache for below-bottom reduction for NT '{}' and len {}: {:?}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len, dest_wrapper);
                                        }
                                    } else {
                                        crate::debug!(6, "Adding to cache for below-bottom reduction for NT '{}' and len {}: {:?}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len, dest_wrapper);
                                        used_dests.insert(dest_wrapper.clone());
                                    }
                                    cache_entry.entry(dest_wrapper.clone()).and_modify(|bv| *bv |= new_tokens).or_insert(new_tokens.clone());
                                }

                                // Update live tokens on all destinations that were used
                                for (dst_wr, added) in &dest_agg {
                                    let mut dg = dst_wr.as_arc().write().expect("poison");
                                    dg.value.live_tokens |= added.clone();
                                }
                                });
                                timeit!("GLRParserState::reduce_and_goto::BLOCK_3: Below-bottom reduction goto processing", {
                                if !used_dests.is_empty() {
                                    let mut acc2 = acc.clone();
                                    acc2.trie2_nodes = used_dests.clone();
                                    let new_gss0 = GSSNode::new(acc2);
                                    let new_gss1 = new_gss0.push(ParseStateEdgeContent { state_id: *source_state_id });
                                    let new_gss2 = new_gss1.push(ParseStateEdgeContent { state_id: *goto_state_id });
                                    below_zero.push(Arc::new(new_gss2));
                                }
                                });
                            }
                        }
                    }
                    let merged = timeit!("GLRParserState::reduce_and_goto: Merging below-zero nodes", {
                        // timeit!(format!("GLRParserState::reduce_and_goto: Merging {} below-zero nodes", below_zero.len()), {
                        GSSNode::merge_many_with_depth(usize::MAX, below_zero)
                    // })
                    });
                    out.push(merged);});
                }
            }
            });

        timeit!("GLRParserState::reduce_and_goto", {
        // timeit!(format!("GLRParserState::reduce_and_goto: Merging {} nodes", out.len()), {
            (GSSNode::merge_many_with_depth(usize::MAX, out), GSSNode::merge_many_with_depth(usize::MAX, accepted_out))
        // })
        })
    }

    pub fn process_token(&mut self, token_id: TerminalID) {
        self.process_token_advanced(token_id, &ProcessTokenAdvancedConfig::default())
    }

    #[time_it("GLRParserState::process_token_advanced")]
    pub fn process_token_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
        // Reset acceptance flag for the new token
        self.accepted = false;
        self.active_state.accepted_state = Arc::new(GSSNode::new_fresh());

        self.below_bottom_cache.clear();

        if Some(token_id) == self.parser.ignore_terminal_id {
            crate::debug!(4, "Ignoring token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());
            self.phase = ParserPhase::ReadyForDefaultReductions; // Skip phase 1 and 2, go straight to phase 3
            return;
        }

        self.log_gss("Phase1/2-start", token_id, false, false);

        let mut phase2_todo: WorkMap = WorkMap::new();
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let mut accepted_states_todo: VecDeque<ParseState> = VecDeque::new();

        if self.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: WorkMap = WorkMap::new();
            Self::enqueue(&mut phase1_todo, self.active_state.clone(), None);
            self._do_actions_without_default(token_id, &mut phase1_todo, &mut phase2_todo, &mut shifted_states_todo, &mut accepted_states_todo, config);
        } else { // ParserPhase::ReadyForDefaultReductions
            Self::enqueue(&mut phase2_todo, self.active_state.clone(), None);
        }

        // --- Phase 2 ---
        self._do_actions_with_default(token_id, &mut phase2_todo, &mut shifted_states_todo, &mut accepted_states_todo, config);

        // Consolidate all shifted states into the new active_state for phase 3
        crate::debug!(4, "Phase 2 completed, consolidating {} shifted states into active state", shifted_states_todo.len());
        let mut next_active = ParseState::new();
        for state in shifted_states_todo {
            next_active.merge(state);
        }
        for state in accepted_states_todo {
            next_active.merge(state);
        }
        self.active_state = next_active;
        self.log_gss("Phase1/2-end", token_id, false, false);
        self.below_bottom_cache.clear();
    }

    pub fn process_default_reductions(&mut self) {
        self.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig::default());
    }

    #[time_it("GLRParserState::process_default_reductions_advanced")]
    pub fn process_default_reductions_advanced(&mut self, config: &ProcessDefaultReductionsAdvancedConfig) {
        self.log_gss("Phase3-start", TerminalID(0), false, false); // Log with dummy token ID
        if self.phase == ParserPhase::ReadyForToken {
            crate::debug!(4, "Phase 3 skipped, parser is ready for Phase 1");
            return;
        }
        assert_eq!(self.phase, ParserPhase::ReadyForDefaultReductions);

        // Phase 3: apply default reductions until fixpoint (no token involved).
        let mut work_map: WorkMap = WorkMap::new();
        // Seed the queue with the current active state.
        Self::enqueue(&mut work_map, self.active_state.clone(), config.per_state_fuel);

        // Collect survivors (clone-and-merge states and reduction results that finalize).
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let mut accepted_states_todo: VecDeque<ParseState> = VecDeque::new();

        let mut fuel = config.fuel;
        let token_config = ProcessTokenAdvancedConfig { below_bottom_mode: config.below_bottom_mode };

        // Run the generic action-processing loop with a Default-only selector.
        // - reduce_map = None to keep enqueuing reductions back to the same queue until closure.
        // - action_selector returns the Default action for each row (no token actions here).
        self.process_action_queue(
            &mut work_map,
            None,
            &mut shifted_states_todo,
            &mut accepted_states_todo,
            |row| Some(Action::Default(&row.default_reduce)),
            &token_config,
            &mut fuel,
        );

        // Consolidate all survivors into the new active state.
        let mut next_active = ParseState::new();
        for state in shifted_states_todo {
            next_active.merge(state);
        }
        for state in accepted_states_todo {
            next_active.merge(state);
        }
        for (_, (state, _fuel)) in work_map {
            next_active.merge(state);
        }
        self.active_state = next_active;

        // After Phase 3, we’re ready for the next token.
        self.phase = ParserPhase::ReadyForToken;
        self.log_gss("Phase3-end", TerminalID(0), false, false);
    }

    pub fn has_action_for(&self, token_id: TerminalID) -> Option<LLMTokenBV> {
        match LR_MODE {
            LRMode::LR1 | LRMode::LALR_EX_SHIFT_STATES => {
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
                    let action_opt = match self.phase {
                        ParserPhase::ReadyForToken => row.shifts_and_reduces_without_default_reduce.get(&token_id).map(Action::Normal),
                        ParserPhase::ReadyForDefaultReductions => row.shifts_and_reduces_full.get(&token_id).map(Action::Normal).or_else(|| Some(Action::Default(&row.default_reduce))),
                    };
                    if let Some(action) = action_opt {
                        crate::debug!(4, "Found action for token '{}' in state {}: {:?}. LLM tokens: {:?}",
                                      self.parser.terminal_map.get_by_right(&token_id).unwrap(),
                                      peek.edge_value().state_id.0, action, peek.resolved_llm_tokens_union());
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
            LRMode::LALR => None,
        }
    }

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

    pub fn step_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
        self.process_token_advanced(token_id, config);
    }

    pub fn parse_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        self.parse_part_advanced(input, config);
    }

    pub fn parse_part_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        for &token_id in input {
            self.step_advanced(token_id, config);
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
        const MAX: usize = 100;
        const PANIC_THRESHOLD: usize = 1_000_000;

        let mut roots_to_log: Vec<(&str, Arc<GSSNode>)> = vec![("Active", self.active_state.stack.clone())];
        if !self.active_state.accepted_state.is_empty() {
            roots_to_log.push(("Accepted", self.active_state.accepted_state.clone()));
        }

        let stats_breakdown = roots_to_log.iter().map(|(name, root)| {
            let stats = gather_gss_stats(&[root.as_ref()]);
            format!("{}_nodes: {:?}", name.to_lowercase(), stats)
        }).collect::<Vec<_>>().join(" ");

        crate::debug!(3, "{} ({:?}) - accepted: {} - token '{}' ({}) - {}",
                      phase, self.phase, self.accepted, self.parser.terminal_map.get_by_right(&token).unwrap(), token.0, stats_breakdown);

        let mut gss_strings = vec![];
        let mut all_state_ids = BTreeSet::new();
        let mut total_nodes = 0;

        for (name, root) in &roots_to_log {
            let stats = gather_gss_stats(&[root.as_ref()]);
            total_nodes += stats.unique_nodes;

            let (current_gss_string, current_state_ids) = {
                let print_full_forest = stats.total_edges <= MAX;
                let max_edges_to_print = if print_full_forest { usize::MAX } else { MAX };
                let config = GSSPrintConfig {
                    max_edges: max_edges_to_print,
                    ..Default::default()
                };
                let (gss_string, state_ids) = print_gss_forest(&[root.clone()], &self.parser.terminal_map, &config);
                let final_string = if print_full_forest {
                    format!("{} GSS ({} nodes, {} edges):\n{}", name, stats.unique_nodes, stats.total_edges, gss_string)
                } else {
                    match find_longest_path(root) {
                        Some(p) => format!("{} GSS too big ({} nodes, {} edges). Longest path ({}): {}",
                                           name,
                                           stats.unique_nodes,
                                           stats.total_edges,
                                           p.len(),
                                           p.iter().map(|(ec, _n)| ec.state_id.0) // n is Arc<GSSNode>
                                                .map(|id| id.to_string())
                                                .collect::<Vec<_>>()
                                            .join(" → ")),
                        None => format!("{} GSS too big ({} nodes, {} edges) – path not found", name, stats.unique_nodes, stats.total_edges),
                    }
                };
                (final_string, state_ids)
            };
            gss_strings.push(current_gss_string);
            all_state_ids.extend(current_state_ids);
        }

        let mut final_string = gss_strings.join("\n\n");
        if explain_states && !all_state_ids.is_empty() {
            final_string.push_str("\n\n--- GSS State Explanations ---\n");
                for state_id in all_state_ids {
                    let mut explanation = String::new();
                    writeln!(&mut explanation, "\n--- State {} ---", state_id.0).unwrap();
                    self.parser.format_state_details(&mut explanation, state_id, "  ").unwrap();
                    final_string.push_str(&explanation);
                }
        }

        if total_nodes > PANIC_THRESHOLD {
            panic!("GSS too big ({} nodes). {}", total_nodes, final_string);
        }

        debug!(3, "{}", final_string);

        if generate_dot {
            let dot_string = self.gss_to_dot();
            // Log the DOT string. It can be copied into a .dot file and rendered with Graphviz.
            // e.g., `dot -Tpng -o gss.png gss.dot`
            crate::debug!(1, "GSS DOT graph:\n{}", dot_string);
        }
    }

    /// Generates a Graphviz DOT representation of the GSS state graph.
    pub fn gss_to_dot(&self) -> String {
        let mut roots: Vec<(&str, &GSSNode)> = vec![("Active", &self.active_state.stack)];
        if !self.active_state.accepted_state.is_empty() {
            roots.push(("Accepted", &self.active_state.accepted_state));
        }
        self.parser.gss_forest_to_dot(&roots, None, None)
    }
}

impl GLRParser {
    /// Generates a Graphviz DOT representation of the state transitions present in a GSS forest.
    /// This visualizes the portion of the state machine explored by the parser.
    pub fn gss_forest_to_dot(
        &self,
        roots: &[(&str, &GSSNode)],
        original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
        llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    ) -> String {
        let mut dot = String::new();
        writeln!(&mut dot, "digraph GSS_Forest {{").unwrap();
        writeln!(&mut dot, "  rankdir=LR;").unwrap();
        writeln!(&mut dot, "  node [shape=box, fontname=\"Courier New\", style=rounded];").unwrap();
        writeln!(&mut dot, "  edge [arrowhead=vee];").unwrap();

        let mut visited_nodes = HashSet::new();
        let mut node_ids = HashMap::new();
        let mut edge_node_ids = HashMap::new();
        let mut next_id_counter = 0;

        let mut queue: VecDeque<Arc<GSSNode>> = roots.iter().map(|(_, n)| Arc::new((*n).clone())).collect();

        // Define root labels and connect them
        for (i, (label, root)) in roots.iter().enumerate() {
            let root_ptr = *root as *const GSSNode;
            let root_id = *node_ids.entry(root_ptr).or_insert_with(|| {
                let id = next_id_counter;
                next_id_counter += 1;
                id
            });

            writeln!(&mut dot, "  subgraph cluster_{} {{", i).unwrap();
            writeln!(&mut dot, "    label=\"{}\";", label).unwrap();
            writeln!(&mut dot, "    style=filled;").unwrap();
            writeln!(&mut dot, "    color=lightgrey;").unwrap();
            writeln!(&mut dot, "    node [style=filled,color=white];").unwrap();
            let root_node_name = format!("Root_{}", i);
            writeln!(&mut dot, "    {} [label=\"{}\", shape=ellipse];", root_node_name, root_id).unwrap();
            writeln!(&mut dot, "  }}").unwrap();
            writeln!(&mut dot, "  {} -> N{};", root_node_name, root_id).unwrap();
        }

        // Traverse and define all nodes and edges
        while let Some(node_arc) = queue.pop_front() {
            let node_ptr = Arc::as_ptr(&node_arc);
            if visited_nodes.contains(&node_ptr) {
                continue;
            }
            
            let parent_id = *node_ids.entry(node_ptr).or_insert_with(|| {
                let id = next_id_counter;
                next_id_counter += 1;
                id
            });

            // Define the GSS node if it hasn't been visited yet
            if visited_nodes.insert(node_ptr) {
                let acc_str = crate::datastructures::gss::format_acc(
                    &node_arc,
                    &self.terminal_map,
                    original_internal_bimap,
                    llm_token_map,
                );
                let escaped_acc = acc_str
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\l")
                    .replace('{', "\\{")
                    .replace('}', "\\}")
                    .replace('<', "\\<")
                    .replace('>', "\\>")
                    .replace('\'', "\\'"); // Escape single quotes for DOT

                writeln!(&mut dot, "  N{} [label=\"Node {}\\lDepth: {}\\l{}\"];", parent_id, parent_id, node_arc.max_depth(), escaped_acc).unwrap();
            }

            for (edge_val, preds_by_depth) in node_arc.predecessors() {
                let state_id = edge_val.state_id;
                let edge_key = (node_ptr, edge_val.clone());
                
                let edge_node_id = *edge_node_ids.entry(edge_key).or_insert_with(|| {
                    let id = next_id_counter;
                    next_id_counter += 1;
                    
                    // Define the edge node
                    let mut explanation = String::new();
                    self.format_state_details(&mut explanation, state_id, "").unwrap();
                    let escaped_explanation = explanation
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\l")
                        .replace('{', "\\{")
                        .replace('}', "\\}")
                        .replace('<', "\\<")
                        .replace('>', "\\>")
                        .replace('\'', "\\'"); // Escape single quotes for DOT
                    
                    writeln!(&mut dot, "  E{} [label=\"State {}\\l{}\", shape=plaintext, fontname=\"Courier New\"];", id, state_id.0, escaped_explanation).unwrap();
                    id
                });

                // Connect parent to edge node
                writeln!(&mut dot, "  N{} -> E{};", parent_id, edge_node_id).unwrap();

                for pred_vec in preds_by_depth.values() {
                    for pred_arc in pred_vec {
                        let pred_ptr = Arc::as_ptr(pred_arc);
                        let pred_id = *node_ids.entry(pred_ptr).or_insert_with(|| {
                            let id = next_id_counter;
                            next_id_counter += 1;
                            id
                        });
                        
                        // Connect edge node to predecessor
                        writeln!(&mut dot, "  E{} -> N{} [arrowhead=none];", edge_node_id, pred_id).unwrap();
                        queue.push_back(pred_arc.clone());
                    }
                }
            }
        }

        writeln!(&mut dot, "}}").unwrap();
        dot
    }

    /// Generates a Graphviz DOT representation of the state transitions present in a GSS.
    /// This visualizes the portion of the state machine explored by the parser.
    pub fn gss_to_dot(&self, root: &GSSNode, original_internal_bimap: Option<&BiBTreeMap<usize, usize>>, llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>) -> String {
        self.gss_forest_to_dot(&[("Root", root)], original_internal_bimap, llm_token_map)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
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
 
// Helper for default reductions' fast unit-reduction chain
fn default_reduce_chain(
    parser: &GLRParser,
    start_state_id: StateID, // The state *before* the GOTO
    initial_nt: NonTerminalID,
) -> BTreeSet<StateID> {
    let mut final_goto_state_ids = BTreeSet::new();
    let mut current_nt = initial_nt;
    // The state for GOTO lookups is always the one before the reduction sequence.
    let goto_source_state_id = start_state_id;

    loop {
        if let Some(goto) = parser.table.get(&goto_source_state_id).and_then(|row| row.gotos.get(&current_nt)) {
            if let Some(goto_state_id) = goto.state_id {
                let next_row = &parser.table[&goto_state_id];
                if let Some(next_reduce) = &next_row.default_reduce.reduce {
                    if next_reduce.0.len == 1 {
                        // This is a unit reduction. Continue the chain with the new non-terminal.
                        current_nt = next_reduce.0.nonterminal_id;
                        continue; // Continue the loop
                    }
                }
                // Not a unit reduction, or no default reduce. This is the end of the chain.
                final_goto_state_ids.insert(goto_state_id);
                break;
            } else {
                // No goto state. End of chain.
                break;
            }
        } else {
            // No goto for current_nt from goto_source_state_id. End of chain.
            break;
        }
    }
    final_goto_state_ids
}

