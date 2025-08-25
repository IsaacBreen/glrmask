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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
                        write!(f, "}}")?;
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

