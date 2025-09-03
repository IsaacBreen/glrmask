use crate::constraint::{LLMVocab, PrecomputeNode3, PrecomputeNode3Index, StateIDBV, Trie2GodWrapper, Trie3God, Trie3GodWrapper};
use crate::datastructures::gss::{find_longest_path, gather_gss_stats, GSSNode, GSSPeek, GSSStats, LLMTokenBV, PrecomputedNodeContents};
use crate::datastructures::gss::{print_gss_forest, Acc, GSSPopper, GSSPopperItem, GSSPrintConfig};
use crate::datastructures::ArcPtrWrapper;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Row, Stage7ShiftsAndReducesLookaheadValue, StateID, SubstringGoto, Table, TerminalID};
use crate::tokenizer::LLMTokenID;
use std::any::Any;
use std::cmp::Ordering;
use std::sync::{Mutex, RwLock};
// Import LLMTokenInfo

use crate::datastructures::trie::EdgeInserter;
use crate::{debug, hit};
use crate::glr::automaton::compute_closure;
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::glr::table::{stage_9, DefaultReduce, Reduce, ShiftsAndReducesFull, ShiftsAndReducesWithoutDefaultReduce};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::GSS_LOGGING_ENABLED;
use bimap::BiBTreeMap;
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};
use std::collections::BTreeMap as StdMap;
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use crate::datastructures::trie::{God, GodWrapper};

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
    pub accepted_state: Option<Arc<GSSNode>>,
    pub prev_accepted_state: Arc<GSSNode>,
    pub stored_trie_god: Option<Trie3GodWrapper>,
}

impl ParseState {
    pub fn new() -> Self {
        ParseState {
            stack: Arc::new(GSSNode::new_fresh()),
            accepted_state: None,
            prev_accepted_state: Arc::new(GSSNode::new_fresh()),
            stored_trie_god: None,
        }
    }

    pub(crate) fn with_stack(stack: Arc<GSSNode>) -> Self {
        ParseState {
            stack,
            accepted_state: None,
            prev_accepted_state: Arc::new(GSSNode::new_fresh()),
            stored_trie_god: None,
        }
    }

    pub(crate) fn with_god(mut self, trie2_god: Trie3GodWrapper) -> Self {
        self.stored_trie_god = Some(trie2_god);
        self
    }

    pub(crate) fn with_maybe_god(mut self, maybe_god: Option<Trie3GodWrapper>) -> Self {
        self.stored_trie_god = maybe_god;
        self
    }

    #[time_it]
    pub fn merge(&mut self, mut other: ParseState) {
        Arc::make_mut(&mut self.stack).merge_with_depth(usize::MAX, &other.stack);
        if let Some(other_accepted) = other.accepted_state {
            if let Some(self_accepted) = self.accepted_state.as_mut() {
                Arc::make_mut(self_accepted).merge_with_depth(usize::MAX, &other_accepted);
            } else {
                self.accepted_state = Some(other_accepted);
            }
        }
        Arc::make_mut(&mut self.prev_accepted_state).merge_with_depth(usize::MAX, &other.prev_accepted_state);
        // assert_eq!(self.stored_trie_god.is_none(), other.stored_trie_god.is_none());
        if self.stored_trie_god.is_some() && other.stored_trie_god.is_some() {
            assert_eq!(self.stored_trie_god.as_ref().unwrap(), other.stored_trie_god.as_ref().unwrap());
        } else if other.stored_trie_god.is_some() {
            self.stored_trie_god = other.stored_trie_god;
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
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }

    pub fn init_glr_parser_null(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: ParseState::new(),
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }

    pub fn init_glr_parser_with_acc(&self) -> GLRParserState { // No longer generic
        let initial_parse_state = self.init_parse_state_with_acc();
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        };
        parser_state
    }

    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState { // No longer generic
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        };
        parser_state
    }

    pub fn init_glr_parser_from_stack(&self, stack: Arc<GSSNode>) -> GLRParserState {
        self.init_glr_parser_from_parse_state(ParseState::with_stack(stack))
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
            accepted_state: None,
            prev_accepted_state: Arc::new(GSSNode::new_fresh()),
            stored_trie_god: None,
        }
    }

    pub fn init_glr_substring_parser_with_everything_state(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_substring_with_everything_state();
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
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
            accepted_state: None,
            prev_accepted_state: Arc::new(GSSNode::new_fresh()),
            stored_trie_god: None,
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
            accepted_state: None,
            prev_accepted_state: Arc::new(GSSNode::new_fresh()),
            stored_trie_god: None,
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
                for item in items {
                    write!(f, "{}- [{} ->", sub_indent, item.production.lhs.0)?;
                    for (i, symbol) in item.production.rhs.iter().enumerate() {
                        if i == item.dot_position {
                            write!(f, " •")?;
                        }
                        match symbol {
                            Symbol::Terminal(terminal) => write!(f, " {}", terminal)?,
                            Symbol::NonTerminal(non_terminal) => write!(f, " {}", non_terminal.0)?,
                        }
                    }
                    if item.dot_position == item.production.rhs.len() {
                        write!(f, " •")?;
                    }
                    writeln!(f, "]")?;
                }
            }
        } else {
            writeln!(f, "{}Items: (State ID not found in item set map)", indent)?;
        }

        // --- Actions & Gotos ---
        if let Some(row) = self.table.get(&state_id) {
            // writeln!(f, "{}Actions (without default reduce):", indent)?;
            // format_actions(f, &row.shifts_and_reduces_without_default_reduce, &self.terminal_map, &self.non_terminal_map, &self.productions, &sub_indent)?;

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
    phase: ParserPhase,
    below_bottom_cache: HashMap<BelowBottomCacheKey, PrecomputeNode3Index>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BelowBottomCacheKey {
    nonterminal_id: NonTerminalID,
    source_state_id: StateID,
    goto_state_id: StateID,
    k: usize,
    // Important: this Acc must have stored_trie_nodes cleared before being placed here.
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
    pub fn with_god(mut self, stored_trie_god: Trie3GodWrapper) -> GLRParserState<'a> {
        self.active_state.stored_trie_god = Some(stored_trie_god);
        self
    }

    fn enqueue(work_map: &mut WorkMap, state: ParseState, fuel: Option<usize>) {
        // Peel off the top edges of the GSS in the given state,
        // and group the resulting isolated paths by their (depth, state_id) key.
        // This merges paths that are in the same logical state, reducing redundant processing.
        for peek in GSSNode::peek_iter(&state.stack) {
            let isolated_state = ParseState {
                stack: peek.isolated_parent(),
                accepted_state: state.accepted_state.clone(),
                prev_accepted_state: state.prev_accepted_state.clone(),
                stored_trie_god: state.stored_trie_god.clone(),
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
            prev_accepted_state: self.active_state.prev_accepted_state.clone(),
            stored_trie_god: None,
        }
    }

    /// Shared inner loop for phase 1 and phase 2.
    /// `action_selector` chooses between the phase-1 or phase-2 action map.
    #[time_it("GLRParserState::process_action_queue")]
    fn process_action_queue<F>(
        &mut self,
        work_map: &mut WorkMap,
        mut reduce_map: Option<&mut WorkMap>,
        shifted_states_todo: &mut VecDeque<ParseState>,
        accepted_states_todo: &mut VecDeque<ParseState>,
        action_selector: F,
        config: &ProcessTokenAdvancedConfig,
        fuel: &mut Option<usize>,
        early_exit_on_shift: bool,
    ) -> bool
    where
        F: for<'r> Fn(&'r Row) -> Option<Action<'r>>,
    {
        let mut found_shift = false;
        assert!(fuel.is_none(), "Fuel is not supported in process_action_queue yet");
        for (state, per_state_fuel) in work_map.values() {
            assert!(per_state_fuel.is_none(), "Per-state fuel is not supported in process_action_queue yet");
        }
        while let Some(entry) = work_map.pop_first() {
            hit!("GLRParserState::process_action_queue::WhileLet");
            let (key, (state, per_state_fuel)) = entry;
            if let Some(f) = fuel {
                if *f == 0 {
                    // Out of fuel. Put the state back and return.
                    work_map.insert(key, (state, per_state_fuel));
                    return found_shift;
                }
                *f -= 1;
            }
            let WorkMapKey(_depth, state_id) = key;
            let row = &self.parser.table[&state_id];
            let action_opt = action_selector(row);
            if let Some(action) = action_opt {
                for peek in GSSNode::peek_iter(&state.stack) {
                    hit!("GLRParserState::process_action_queue::ForEachPeek");
                    match action {
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Shift(to)) => {
                            hit!("GLRParserState::process_action_queue::Shift");
                            crate::debug!(5, "Action: Shift to state {}", to.0);
                            let new_parse_state =
                                self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                            shifted_states_todo.push_back(new_parse_state);
                            found_shift = true;
                            if early_exit_on_shift {
                                return true;
                            }
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                            nonterminal_id: nt,
                            len,
                            ..
                        }) => {
                            hit!("GLRParserState::process_action_queue::Reduce");
                            if per_state_fuel == Some(0) { continue; }
                            let new_per_state_fuel = per_state_fuel.map(|f| f - 1);

                            crate::debug!(5, "Action: Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), len);
                            let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(&peek, *nt, *len, &action_selector, config);
                            if !s_new_arc.is_empty() {
                                let new_parse_state = ParseState {
                                    stack: s_new_arc,
                                    accepted_state: state.accepted_state.clone(),
                                    prev_accepted_state: state.prev_accepted_state.clone(),
                                    stored_trie_god: state.stored_trie_god.clone(),
                                };
                                if let Some(ref mut r_map) = reduce_map {
                                    Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                } else {
                                    Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                }
                            }
                            if !accepted_s_new_arc.is_empty() {
                                let accepted_parse_state = ParseState {
                                    stack: Arc::new(GSSNode::new_fresh()),
                                    accepted_state: Some(accepted_s_new_arc),
                                    prev_accepted_state: state.prev_accepted_state.clone(),
                                    stored_trie_god: state.stored_trie_god.clone(),
                                };
                                accepted_states_todo.push_back(accepted_parse_state);
                            }
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces }) => {
                            crate::debug!(5, "Action: Split with shift and reduces");
                            if let Some(to) = shift {
                                hit!("GLRParserState::process_action_queue::Split::Shift");
                                crate::debug!(5, "Action (Split): Shift to state {}", to.0);
                                let new_parse_state =
                                    self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                                shifted_states_todo.push_back(new_parse_state);
                                found_shift = true;
                                if early_exit_on_shift {
                                    return true;
                                }
                            }
                            if per_state_fuel != Some(0) {
                                let new_per_state_fuel = per_state_fuel.map(|f| f - 1);
                                for (len, nts) in reduces {
                                    for (nt, _prod_ids) in nts {
                                        hit!("GLRParserState::process_action_queue::Split::Reduce");
                                        crate::debug!(5, "Action (Split): Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), *len);
                                        let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(&peek, *nt, *len, &action_selector, config);
                                        if !s_new_arc.is_empty() {
                                            let new_parse_state = ParseState {
                                                stack: s_new_arc,
                                                accepted_state: state.accepted_state.clone(),
                                                prev_accepted_state: state.prev_accepted_state.clone(),
                                                stored_trie_god: state.stored_trie_god.clone(),
                                            };
                                            if let Some(ref mut r_map) = reduce_map {
                                                Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                            } else {
                                                Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                            }
                                        }
                                        if !accepted_s_new_arc.is_empty() {
                                            let accepted_parse_state = ParseState {
                                                    stack: Arc::new(GSSNode::new_fresh()),
                                                    accepted_state: Some(accepted_s_new_arc),
                                                    prev_accepted_state: state.prev_accepted_state.clone(),
                                                    stored_trie_god: state.stored_trie_god.clone(),
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
                                                        prev_accepted_state: state.prev_accepted_state.clone(),
                                                        stored_trie_god: state.stored_trie_god.clone(),
                                                    };
                                                    if let Some(ref mut r_map) = reduce_map {
                                                        Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                                    } else {
                                                        Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                                    }
                                                }
                                                if !accepted_s_new_arc.is_empty() {
                                                    let accepted_parse_state = ParseState {
                                                        stack: Arc::new(GSSNode::new_fresh()),
                                                        accepted_state: Some(accepted_s_new_arc),
                                                        prev_accepted_state: state.prev_accepted_state.clone(),
                                                        stored_trie_god: state.stored_trie_god.clone(),
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
        return found_shift;
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
                false,
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
                false,
            );
            self.phase = ParserPhase::ReadyForDefaultReductions;
        });
    }

    // ----------------------------------------------------------------------
    // Refactored helpers to make reduce_and_goto clearer
    // ----------------------------------------------------------------------

    #[inline]
    fn substring_gotos_for<'b>(
        &self,
        nt: NonTerminalID,
        config: &ProcessTokenAdvancedConfig,
        storage: &'b mut SubstringGoto,
    ) -> &'b SubstringGoto where 'a: 'b {
        match config.below_bottom_mode {
            BelowBottomReductionMode::ContinueFromAll => {
                self.parser.substring_gotos.get(&nt).unwrap_or(storage)
            }
            BelowBottomReductionMode::ContinueFromEverything => {
                // Build a compact SubstringGoto from the synthetic "everything" state.
                let everything = self.parser.everything_state_id;
                if let Some(goto) = self
                    .parser
                    .table
                    .get(&everything)
                    .and_then(|row| row.gotos.get(&nt))
                {
                    storage.accepting_sources.clear();
                    storage.gotos.clear();
                    if goto.accept {
                        storage.accepting_sources.insert(everything);
                    }
                    if let Some(goto_state_id) = goto.state_id {
                        storage.gotos.insert(goto_state_id, BTreeSet::from([everything]));
                    }
                }
                storage
            }
            BelowBottomReductionMode::Fail => storage,
            BelowBottomReductionMode::Panic => {
                // Handled by caller if a below-bottom pop happens; not used here.
                storage
            }
        }
    }

    fn build_below_bottom_accs(&self, popper: &GSSPopper) -> BTreeMap<usize, Acc> {
        let god = self.active_state.stored_trie_god.as_ref().expect("Trie3 god missing");
        let mut result: BTreeMap<usize, Acc> = BTreeMap::new();

        for (k, accs_by_edge) in popper.below_bottom() {
            // Union of Acc over all last-edge entries for this k
            let mut acc_union: Option<Acc> = None;
            // New set of stored trie nodes created/used by these insertions (for this k)
            let mut new_stored: BTreeSet<PrecomputeNode3Index> = BTreeSet::new();

            for (last_edge, acc_arc) in accs_by_edge {
                let acc = acc_arc.as_ref();
                let edge_key = (0, LLMTokenBV::max_ones());
                let mut edge_bv = StateIDBV::zeros();
                edge_bv.insert(last_edge.state_id.0);

                // Union this acc into the accumulator for k
                acc_union = Some(match acc_union.take() {
                    None => acc.clone(),
                    Some(prev) => Acc::merge(&prev, acc),
                });

                // For each existing stored trie node, wire a strong edge to a fresh destination,
                // and make sure the destination accumulates max-ones (same as original behavior).
                for existing in acc.stored_trie_nodes() {
                    let source = existing.as_arc().clone();
                    let fallback = PrecomputeNode3Index::new(
                        god.insert(PrecomputeNode3::new(crate::constraint::PrecomputedNode3Contents::internal())),
                    );

                    let dst = EdgeInserter::new(
                            god,
                            source,
                            edge_key.clone(),
                            edge_bv.clone(),
                            |e, n| *e |= n,                                 // merge edge bitset
                            |_, _| {}, // propagate to node
                            |_, _| {},                  // edge_value &= source.live_tokens
                        )
                        .try_destination(fallback)
                        .expect("build_below_bottom_accs: insert failed");

                    // Ensure destination accumulates max-ones (matches original OR with dest_agg).
                    new_stored.insert(dst);
                }
            }

            // Build final Acc for this k: union of all accs (same as before) with new stored_trie set.
            let mut final_acc = acc_union.unwrap_or_else(Acc::new_fresh);
            *final_acc.stored_trie_nodes_mut() = new_stored;

            result
                .entry(*k)
                .and_modify(|existing| *existing = Acc::merge(existing, &final_acc))
                .or_insert(final_acc);
        }

        result
    }

    fn handle_below_bottom_accepts(
        &mut self,
        nt: NonTerminalID,
        below: &BTreeMap<usize, Acc>,
        gotos: &SubstringGoto,
    ) -> Option<Arc<GSSNode>> {
        if gotos.accepting_sources.is_empty() {
            return None;
        }

        let god = self
            .active_state
            .stored_trie_god
            .as_ref()
            .expect("Trie3 god missing");

        // Single cached destination for all accept contributions (same sentinel keying as before)
        let accept_cache_key = BelowBottomCacheKey {
            nonterminal_id: nt,
            source_state_id: StateID(usize::MAX), // sentinel
            goto_state_id: StateID(usize::MAX),   // sentinel
            k: 0,                                 // sentinel
        };
        let (dst_arc, _is_new) = if let Some(dst) = self.below_bottom_cache.get(&accept_cache_key) {
            (dst.clone(), false)
        } else {
            let dst = PrecomputeNode3Index::new(
                god.insert(PrecomputeNode3::new(crate::constraint::PrecomputedNode3Contents::internal())),
            );
            self.below_bottom_cache.insert(accept_cache_key, dst.clone());
            (dst, true)
        };

        let edge_bv = StateIDBV::max_ones();
        let mut accepted_stacks: Vec<Arc<GSSNode>> = Vec::new();

        // For each k and its Acc, add edges for every accepting source state,
        // then build the accepted GSS node that starts from that source.
        for (k, acc) in below {
            let stored = acc.stored_trie_nodes().clone();
            for source_state_id in &gotos.accepting_sources {
                let mut edge_val = StateIDBV::zeros();
                edge_val.insert(source_state_id.0);

                for existing in &stored {
                    let _ = EdgeInserter::new(
                        god,
                        existing.as_arc().clone(),
                        (*k, LLMTokenBV::max_ones()),
                        edge_bv.clone(),
                        |e, n| *e |= n,
                        |_, _| {},
                        |_, _| {},
                    )
                    .try_destination(dst_arc.clone())
                    .expect("Cycle in below-bottom accept wiring");
                }

                // Create the accepted stack node with the updated Acc (points only to the cached dst).
                let mut acc_for_gss = acc.clone();
                acc_for_gss.stored_trie_nodes_mut().clear();
                acc_for_gss.stored_trie_nodes_mut().insert(dst_arc.clone());
                let gss0 = GSSNode::new(acc_for_gss);
                let gss1 = gss0.push(ParseStateEdgeContent { state_id: *source_state_id });
                accepted_stacks.push(Arc::new(gss1));
            }
        }

        if accepted_stacks.is_empty() {
            None
        } else {
            Some(GSSNode::merge_many_with_depth(usize::MAX, accepted_stacks))
        }
    }
    
    fn handle_below_bottom_gotos(
        &mut self,
        nt: NonTerminalID,
        below: BTreeMap<usize, Acc>,
        gotos: &SubstringGoto,
    ) -> Arc<GSSNode> {
        if gotos.gotos.is_empty() {
            return Arc::new(GSSNode::new_fresh());
        }

        let god = self
            .active_state
            .stored_trie_god
            .as_ref()
            .expect("Trie3 god missing");

        // Cache key (same as original)
        let cache_key = BelowBottomCacheKey {
            nonterminal_id: nt,
            source_state_id: StateID(0),
            goto_state_id: StateID(0),
            k: 0,
        };

        // Merge all k-accs (then clear stored nodes; we’ll point to the cached dest below)
        let mut merged_acc = {
            let mut it = below.values();
            match it.next() {
                None => Acc::new_fresh(),
                Some(first) => it.fold(first.clone(), |acc, nxt| Acc::merge(&acc, nxt)),
            }
        };
        merged_acc.stored_trie_nodes_mut().clear();

        // Obtain or create a shared destination trie node.
        let (dest_node, enqueue_gss) = if let Some(dst) = self.below_bottom_cache.get(&cache_key) {
            (dst.clone(), false)
        } else {
            let dst = PrecomputeNode3Index::new(
                god.insert(PrecomputeNode3::new(crate::constraint::PrecomputedNode3Contents::internal())),
            );
            self.below_bottom_cache.insert(cache_key, dst.clone());
            (dst, true)
        };

        // Insert strong edges from all source trie nodes to the cached destination, keyed by (k, None).
        let edge_bv = StateIDBV::max_ones();
        for (k, acc) in &below {
            for existing in acc.stored_trie_nodes() {
                let _ = EdgeInserter::new(
                    god,
                    existing.as_arc().clone(),
                    (*k, LLMTokenBV::max_ones()),
                    edge_bv.clone(),
                    |e, n| *e |= n,
                    |_, _| {},
                    |_, _| {}, // no per-source restriction here
                )
                .try_destination(dest_node.clone());
            }
        }

        // Only build the GSS result when we first created the cached destination.
        if enqueue_gss {
            merged_acc.stored_trie_nodes_mut().insert(dest_node);
            let mut out: Vec<Arc<GSSNode>> = Vec::new();

            for (goto_state_id, source_state_ids) in &gotos.gotos {
                let edge_contents = source_state_ids
                    .iter()
                    .map(|sid| ParseStateEdgeContent { state_id: *sid })
                    .collect::<Vec<_>>();

                let gss0 = GSSNode::new(merged_acc.clone());
                let gss1 = gss0.push_many(edge_contents);
                let gss2 = gss1.push(ParseStateEdgeContent { state_id: *goto_state_id });
                out.push(Arc::new(gss2));
            }

            GSSNode::merge_many_with_depth(usize::MAX, out)
        } else {
            Arc::new(GSSNode::new_fresh())
        }
    }

    /// Reduce by non-terminal `nt` of length `len`, and perform the corresponding gotos.
    /// Returns (new_active_stack, new_accepted_stack).
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
        // 1) Pop len
        let popper: GSSPopper = timeit!(peek.popn(len));
        crate::debug!(4, "Reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
        crate::debug!(4, "Popped with {} results...", popper.num_predecessors());

        let mut out: Vec<Arc<GSSNode>> = Vec::new();
        let mut accepted_out: Vec<Arc<GSSNode>> = Vec::new();

        // 2) Standard reductions along in-graph paths
        for popper_item in popper.iter() {
            for peek2 in popper_item.peek_iter() {
                // Follow unit-reduction chains quickly on the goto side
                let predecessor_state_id = peek2.edge_value().state_id;
                let mut current_nt = nt;

                loop {
                    // GOTO lookup from predecessor_state_id
                    let goto = self
                        .parser
                        .table
                        .get(&predecessor_state_id)
                        .and_then(|row| row.gotos.get(&current_nt))
                        .expect_else(|| {
                            format!(
                                "Goto not found for NT '{}' in state {:?}",
                                self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(),
                                predecessor_state_id
                            )
                        });

                    // Accept contribution (store isolated parent)
                    if goto.accept {
                        accepted_out.push(peek2.isolated_parent());
                    }

                    if let Some(goto_state_id) = goto.state_id {
                        let next_row = &self.parser.table[&goto_state_id];
                        match action_selector(next_row) {
                            Some(Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                nonterminal_id: next_nt,
                                len: 1,
                                ..
                            })) => {
                                // Unit reduce chain: continue
                                current_nt = *next_nt;
                                continue;
                            }
                            Some(Action::Default(def)) => {
                                // If the default reduce isn't a unit reduce, we must commit the current goto result.
                                if def.clone_and_merge
                                    || def
                                        .reduce
                                        .as_ref()
                                        .map_or(false, |r| r.0.len != 1)
                                {
                                    out.push(Arc::new(peek2.push_on_parent(
                                        ParseStateEdgeContent {
                                            state_id: goto_state_id,
                                        },
                                    )));
                                }
                                // If it's a unit reduction, continue chaining.
                                if let Some(reduce) = &def.reduce {
                                    if reduce.0.len == 1 {
                                        current_nt = reduce.0.nonterminal_id;
                                        continue;
                                    }
                                }
                                // Otherwise, end chain
                                break;
                            }
                            _ => {
                                // Not a unit reduction path anymore -> emit a single push to goto_state
                                out.push(Arc::new(peek2.push_on_parent(ParseStateEdgeContent {
                                    state_id: goto_state_id,
                                })));
                                break;
                            }
                        }
                    } else {
                        // No goto target -> we're done.
                        break;
                    }
                }
            }
        }

        // 3) Handle "below bottom" (substring parsing continuation)
        if !popper.below_bottom().is_empty() {
            match config.below_bottom_mode {
                BelowBottomReductionMode::Fail => {
                    crate::debug!(5, "Popped below bottom, failing these parse paths.");
                }
                BelowBottomReductionMode::Panic => {
                    panic!("A reduction popped below the bottom of the stack, and BelowBottomReductionMode was set to Panic.");
                }
                _ => {
                    // Build Accs aggregated by k, then continue from either all states or the everything state.
                    let below_accs = self.build_below_bottom_accs(&popper);

                    let mut storage = SubstringGoto::default();
                    let gotos_for_nt =
                        self.substring_gotos_for(nt, config, &mut storage);

                    // Accepting sources (if any)
                    if let Some(accepted_merged) =
                        self.handle_below_bottom_accepts(nt, &below_accs, gotos_for_nt)
                    {
                        accepted_out.push(accepted_merged);
                    }

                    // Non-accepting gotos
                    let merged_below = self.handle_below_bottom_gotos(nt, below_accs, gotos_for_nt);
                    out.push(merged_below);
                }
            }
        }

        // 4) Merge results and return
        let new_active = GSSNode::merge_many_with_depth(usize::MAX, out);
        let new_accepted = GSSNode::merge_many_with_depth(usize::MAX, accepted_out);
        (new_active, new_accepted)
    }

    pub fn process_token(&mut self, token_id: TerminalID) {
        self.process_token_advanced(token_id, &ProcessTokenAdvancedConfig::default())
    }

    #[time_it("GLRParserState::process_token_advanced")]
    pub fn process_token_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
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
        let mut next_active = ParseState {
            stack: Arc::new(GSSNode::new_fresh()),
            accepted_state: None,
            prev_accepted_state: self.active_state.prev_accepted_state.clone(),
            stored_trie_god: self.active_state.stored_trie_god.clone(),
        };
        for state in shifted_states_todo {
            next_active.merge(state);
        }
        for state in accepted_states_todo {
            next_active.merge(state);
        }
        self.active_state = next_active;

        // Move current accepted state to previous, and reset current.
        self.active_state.prev_accepted_state = self.active_state.accepted_state.take().unwrap_or_else(|| Arc::new(GSSNode::new_fresh()));
        self.active_state.accepted_state = None;

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
            false,
        );

        // Consolidate all survivors into the new active state.
        let mut next_active = ParseState::new().with_maybe_god(self.active_state.stored_trie_god.clone());
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

    /// Returns true iff simulating a single-step with `token_id` would perform any SHIFT.
    /// This clones the parser state and early-exits on the first SHIFT.
    pub fn allows_terminal(&self, token_id: TerminalID) -> bool {
        // Treat the ignore token as always "allowed" (it doesn't shift but is always consumable).
        if Some(token_id) == self.parser.ignore_terminal_id {
            return true;
        }

        let mut s = self.clone();
        s.below_bottom_cache.clear();

        let mut phase2_todo: WorkMap = WorkMap::new();
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let mut accepted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let cfg = ProcessTokenAdvancedConfig::default();

        if s.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: WorkMap = WorkMap::new();
            Self::enqueue(&mut phase1_todo, s.active_state.clone(), None);
            if s.process_action_queue(
                &mut phase1_todo,
                Some(&mut phase2_todo),
                &mut shifted_states_todo,
                &mut accepted_states_todo,
                |row| row
                    .shifts_and_reduces_without_default_reduce
                    .get(&token_id)
                    .map(Action::Normal),
                &cfg,
                &mut None,
                true, // early_exit_on_shift
            ) {
                return true;
            }
        } else {
            Self::enqueue(&mut phase2_todo, s.active_state.clone(), None);
        }

        s.process_action_queue(
            &mut phase2_todo,
            None,
            &mut shifted_states_todo,
            &mut accepted_states_todo,
            |row| row
                .shifts_and_reduces_full
                .get(&token_id)
                .map(Action::Normal),
            &cfg,
            &mut None,
            true, // early_exit_on_shift
        )
    }

    /// Returns Some(true) if at least one top-of-stack state has an action for `token_id`,
    /// Some(false) otherwise. (Uses the row action map immediately, does not simulate.)
    pub fn has_immediate_action_for_terminal(&self, token_id: TerminalID) -> Option<bool> {
        let mut any = false;
        for peek in GSSNode::peek_iter(&self.active_state.stack) {
            let row = &self.parser.table[&peek.edge_value().state_id];
            let has = if self.phase == ParserPhase::ReadyForToken {
                row.shifts_and_reduces_without_default_reduce.contains_key(&token_id)
            } else {
                row.shifts_and_reduces_full.contains_key(&token_id)
            };
            if has {
                any = true;
                break;
            }
        }
        Some(any)
    }

    /// Returns the set of terminals that cause a SHIFT from at least one top-of-stack state.
    pub fn immediate_shift_terminals(&self) -> BTreeSet<TerminalID> {
        let mut out = BTreeSet::new();
        for peek in GSSNode::peek_iter(&self.active_state.stack) {
            let row = &self.parser.table[&peek.edge_value().state_id];
            let actions = if self.phase == ParserPhase::ReadyForToken {
                &row.shifts_and_reduces_without_default_reduce
            } else {
                &row.shifts_and_reduces_full
            };
            for (tid, action) in actions {
                match action {
                    Stage7ShiftsAndReducesLookaheadValue::Shift(_) => {
                        out.insert(*tid);
                    }
                    Stage7ShiftsAndReducesLookaheadValue::Split { shift, .. } => {
                        if shift.is_some() {
                            out.insert(*tid);
                        }
                    }
                    _ => {}
                }
            }
        }
        out
    }

    /// Returns the set of terminals that cause a REDUCE from at least one top-of-stack state.
    pub fn immediate_reduce_terminals(&self) -> BTreeSet<TerminalID> {
        let mut out = BTreeSet::new();
        for peek in GSSNode::peek_iter(&self.active_state.stack) {
            let row = &self.parser.table[&peek.edge_value().state_id];
            let actions = if self.phase == ParserPhase::ReadyForToken {
                &row.shifts_and_reduces_without_default_reduce
            } else {
                &row.shifts_and_reduces_full
            };
            for (tid, action) in actions {
                match action {
                    Stage7ShiftsAndReducesLookaheadValue::Reduce { .. } => {
                        out.insert(*tid);
                    }
                    Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                        if !reduces.is_empty() {
                            out.insert(*tid);
                        }
                    }
                    _ => {}
                }
            }
        }
        out
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

    pub fn and_step(mut self, token_id: TerminalID) -> GLRParserState<'a> {
        self.step(token_id);
        self
    }

    pub fn and_parse(mut self, input: &[TerminalID]) -> GLRParserState<'a > {
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
    }

    pub fn is_ok(&self) -> bool {
        !self.active_state.stack.is_empty() && self.active_state.stack.is_alive()
    }

    /// Returns true if the token processed two steps ago lead to an `accept` action.
    pub fn has_accepted_prev(&self) -> bool {
        !self.active_state.prev_accepted_state.is_empty()
    }

    /// Returns true if the most recently processed token could lead to an `accept` action,
    /// possibly after subsequent default reductions. This may mutate the state by running
    /// default reductions if they haven't been run yet for the current token.
    pub fn has_accepted(&mut self) -> bool {
        if self.phase == ParserPhase::ReadyForDefaultReductions {
            self.process_default_reductions();
        }
        self.active_state.accepted_state.as_ref().map_or(false, |s| !s.is_empty())
    }

    pub fn log_gss(&self, phase: &str, token: TerminalID, explain_states: bool, generate_dot: bool) {
        if !GSS_LOGGING_ENABLED {
            return;
        }
        // crate::debug!(3, "{} - token {} ({:?}) - nodes", phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0));
        const MAX: usize = 100;
        const PANIC_THRESHOLD: usize = 1_000_000;

        let mut roots_to_log: Vec<(&str, Arc<GSSNode>)> = vec![("Active", self.active_state.stack.clone())];
        if let Some(accepted_state) = &self.active_state.accepted_state {
            if !accepted_state.is_empty() {
                roots_to_log.push(("Accepted", accepted_state.clone()));
            }
        }
        if !self.active_state.prev_accepted_state.is_empty() {
            roots_to_log.push(("PrevAccepted", self.active_state.prev_accepted_state.clone()));
        }

        let stats_breakdown = roots_to_log.iter().map(|(name, root)| {
            let stats = gather_gss_stats(&[root.as_ref()]);
            format!("{}_nodes: {:?}", name.to_lowercase(), stats)
        }).collect::<Vec<_>>().join(" ");

        let accepted_now = self.active_state.accepted_state.is_some();
        let accepted_prev = !self.active_state.prev_accepted_state.is_empty();
        crate::debug!(3, "{} ({:?}) - accepted: now={}, prev={} - token '{}' ({}) - {}",
                      phase, self.phase, accepted_now, accepted_prev, self.parser.terminal_map.get_by_right(&token).expect_else(|| format!("Token {} not found in terminal map: {:?}", token.0, self.parser.terminal_map)), token.0, stats_breakdown);

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
        if let Some(accepted_state) = &self.active_state.accepted_state {
            if !accepted_state.is_empty() {
                roots.push(("Accepted", accepted_state));
            }
        }
        if !self.active_state.prev_accepted_state.is_empty() {
            roots.push(("PrevAccepted", &self.active_state.prev_accepted_state));
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