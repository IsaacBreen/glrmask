//! GLR parser core.
//!
//! This module implements a practical Generalized LR (GLR) parser with three clear stages
//! for each token:
//!   - Phase 1: token-specific actions without default reductions.
//!   - Phase 2: token-specific actions with default reductions.
//!   - Phase 3: default-reduction closure (no token).
//!
//! It is designed to be:
//!   - Readable: broken into small helpers with clear responsibilities.
//!   - Correct: logic is kept equivalent to the previous implementation.
//!   - Compatible: all public types, functions, and behavior remain the same.
//!
//! Key structures:
//!   - GLRParser: the parser itself with tables, maps and initialization helpers.
//!   - GLRParserState<'a>: a running parsing state with an active GSS.
//!   - ParseState: a pair of GSS roots (active and accepted).
//!
//! Notes:
//!   - The GSS (Graph-Structured Stack) is handled in crate::datastructures::gss.
//!   - Substring parsing (continuation below bottom) is supported via "everything/all states" modes.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, RwLock};

use bimap::BiBTreeMap;
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};

use crate::constraint::LLMVocab;
use crate::datastructures::arc_wrapper::ArcPtrWrapper;
use crate::datastructures::gss::{
    find_longest_path, format_acc, gather_gss_stats, print_gss_forest, Acc, GSSNode, GSSPeek,
    GSSPopper, GSSPopperItem, GSSPrintConfig, LLMTokenBV, PrecomputeNode2, PrecomputedNodeContents,
};
use crate::datastructures::trie::EdgeInserter;
use crate::glr::automaton::compute_closure;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::glr::table::{
    stage_9, DefaultReduce, Goto, NonTerminalID, ProductionID, Row, ShiftsAndReducesFull,
    ShiftsAndReducesWithoutDefaultReduce, Stage7ShiftsAndReducesLookaheadValue, StateID,
    SubstringGoto, Table, TerminalID,
};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::GSS_LOGGING_ENABLED;
use crate::tokenizer::LLMTokenID;

/// A single combined action for a given (state,row) and token:
/// - Normal(...) is a concrete per-token action from the row's action map
/// - Default(...) is the row's default reduction (token-independent)
#[derive(Debug)]
enum Action<'a> {
    Normal(&'a Stage7ShiftsAndReducesLookaheadValue),
    Default(&'a DefaultReduce),
}

/// A trait with a lazily-evaluated expect (useful for readable error messages).
pub trait ExpectElse<T> {
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
        match self {
            Some(v) => v,
            None => panic!("{}", f()),
        }
    }
}

/// Erased user-data hooks (kept for API completeness).
pub trait DynEq {
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}
pub trait DynOrd {
    fn dyn_cmp(&self, other: &dyn Any) -> Ordering;
}
pub trait DynHash {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
}
impl DynEq for () {
    fn dyn_eq(&self, _other: &dyn Any) -> bool {
        true
    }
}
impl DynOrd for () {
    fn dyn_cmp(&self, _other: &dyn Any) -> Ordering {
        Ordering::Equal
    }
}
impl DynHash for () {
    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) {}
}

pub trait UserDataTrait: Any + Send + Sync + Debug + DynEq + DynOrd + DynHash {}
impl UserDataTrait for () {}

pub type ActionFn = Arc<dyn Fn(&mut Arc<dyn UserDataTrait>) -> bool + Send + Sync>;

/// Edge payload in the GSS — simply the LR state id.
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
        self.state_id.partial_cmp(&other.state_id)
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
impl JSONConvertible for ParseStateEdgeContent {
    fn to_json(&self) -> JSONNode {
        let mut obj = std::collections::BTreeMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let state_id = obj
                    .remove("state_id")
                    .ok_or_else(|| "Missing field state_id for ParseStateEdgeContent".to_string())
                    .and_then(StateID::from_json)?;
                Ok(ParseStateEdgeContent { state_id })
            }
            _ => Err("Expected JSONNode::Object for ParseStateEdgeContent".to_string()),
        }
    }
}

/// Parse state contains two roots:
///   - stack: the active graph-structured stack
///   - accepted_state: the graph of accepting continuations accumulated during a token
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Why we stopped in a single-step parse attempt (internal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    ActionNotFound,
    GotoNotFound,
}
impl JSONConvertible for StopReason {
    fn to_json(&self) -> JSONNode {
        let s = match self {
            StopReason::ActionNotFound => "ActionNotFound",
            StopReason::GotoNotFound => "GotoNotFound",
        };
        JSONNode::String(s.to_string())
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

/// A small phase indicator used in processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserPhase {
    ReadyForToken,
    ReadyForDefaultReductions,
}
impl Default for ParserPhase {
    fn default() -> Self {
        ParserPhase::ReadyForDefaultReductions
    }
}

/// How substring parsing behaves when reductions pop below the root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BelowBottomReductionMode {
    ContinueFromAll,
    ContinueFromEverything,
    Fail,
    #[default]
    Panic,
}

/// Advanced token-processing configuration (mostly for substring parsing).
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessTokenAdvancedConfig {
    pub below_bottom_mode: BelowBottomReductionMode,
}

/// Advanced default-reduction configuration (fuel for closure, etc.).
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

/// The parser with its static data and helpers. This is lightweight to clone.
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
        let mut obj = std::collections::BTreeMap::new();
        obj.insert("stage_7_table".to_string(), self.table.to_json());
        obj.insert("productions".to_string(), self.productions.to_json());
        obj.insert("terminal_map".to_string(), self.terminal_map.to_json());
        obj.insert("non_terminal_map".to_string(), self.non_terminal_map.to_json());
        obj.insert("item_set_map".to_string(), self.item_set_map.to_json());
        obj.insert("start_state_id".to_string(), self.start_state_id.to_json());
        obj.insert(
            "everything_state_id".to_string(),
            self.everything_state_id.to_json(),
        );
        obj.insert(
            "ignore_terminal_id".to_string(),
            self.ignore_terminal_id.to_json(),
        );
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let table = obj
                    .remove("stage_7_table")
                    .ok_or_else(|| "Missing field stage_7_table".to_string())
                    .and_then(Table::from_json)?;
                let productions = obj
                    .remove("productions")
                    .ok_or_else(|| "Missing field productions".to_string())
                    .and_then(Vec::<Production>::from_json)?;
                // Back compat with potential old field:
                let _ = obj.remove("start_production_id");

                let terminal_map = obj
                    .remove("terminal_map")
                    .ok_or_else(|| "Missing field terminal_map".to_string())
                    .and_then(|n| BiBTreeMap::<Terminal, TerminalID>::from_json(n))?;
                let non_terminal_map = obj
                    .remove("non_terminal_map")
                    .ok_or_else(|| "Missing field non_terminal_map".to_string())
                    .and_then(|n| BiBTreeMap::<NonTerminal, NonTerminalID>::from_json(n))?;
                let item_set_map = obj
                    .remove("item_set_map")
                    .ok_or_else(|| "Missing field item_set_map".to_string())
                    .and_then(|n| BiBTreeMap::<BTreeSet<Item>, StateID>::from_json(n))?;
                let start_state_id = obj
                    .remove("start_state_id")
                    .ok_or_else(|| "Missing field start_state_id".to_string())
                    .and_then(StateID::from_json)?;
                let everything_state_id = obj
                    .remove("everything_state_id")
                    .ok_or_else(|| "Missing field everything_state_id".to_string())
                    .and_then(StateID::from_json)?;
                let ignore_terminal_id = obj
                    .remove("ignore_terminal_id")
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
        self.table == other.table
            && self.productions == other.productions
            && self.terminal_map == other.terminal_map
            && self.non_terminal_map == other.non_terminal_map
            && self.item_set_map == other.item_set_map
            && self.start_state_id == other.start_state_id
            && self.everything_state_id == other.everything_state_id
            && self.ignore_terminal_id == other.ignore_terminal_id
            && self.substring_gotos == other.substring_gotos
    }
}
impl Eq for GLRParser {}

impl GLRParser {
    /// Constructs a parser. The `actions` argument is accepted to preserve compatibility.
    /// (It is not stored; user hooks are handled elsewhere.)
    pub fn new(
        table: Table,
        productions: Vec<Production>,
        terminal_map: BiBTreeMap<Terminal, TerminalID>,
        non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
        item_set_map: BiBTreeMap<BTreeSet<Item>, StateID>,
        start_state_id: StateID,
        everything_state_id: StateID,
        _actions: BTreeMap<NonTerminal, ActionFn>,
        ignore_terminal_id: Option<TerminalID>,
        substring_gotos: BTreeMap<NonTerminalID, SubstringGoto>,
    ) -> Self {
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

    // --- Initialization helpers ------------------------------------------------

    pub fn init_glr_parser(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        self.init_glr_parser_with_acc()
    }
    pub fn init_glr_parser_with_stack(&self, stack: ParseState) -> GLRParserState {
        GLRParserState {
            parser: self,
            active_state: stack,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }
    pub fn init_glr_parser_null(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        GLRParserState {
            parser: self,
            active_state: ParseState::new(),
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }
    pub fn init_glr_parser_with_acc(&self) -> GLRParserState {
        let initial = self.init_parse_state_with_acc();
        GLRParserState {
            parser: self,
            active_state: initial,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }
    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState {
        GLRParserState {
            parser: self,
            active_state: parse_state,
            accepted: false,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
        }
    }

    /// Substring parser mode: seed with all states as possible top-of-stack.
    pub fn init_glr_substring_parser_with_all_states(
        &self,
        _llm_vocab: Option<Arc<LLMVocab>>,
    ) -> GLRParserState {
        let initial = self.init_parse_state_substring_with_all_states();
        GLRParserState {
            parser: self,
            active_state: initial,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            accepted: false,
        }
    }
    pub fn init_parse_state_substring_with_all_states(&self) -> ParseState {
        let edges: Vec<ParseStateEdgeContent> = self
            .table
            .keys()
            .map(|sid| ParseStateEdgeContent { state_id: *sid })
            .collect();
        let stack_top = GSSNode::new_fresh().push_many(edges);
        ParseState {
            stack: Arc::new(stack_top),
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
    }

    /// Substring parser mode: seed with a special "everything" state.
    pub fn init_glr_substring_parser_with_everything_state(
        &self,
        _llm_vocab: Option<Arc<LLMVocab>>,
    ) -> GLRParserState {
        let initial = self.init_parse_state_substring_with_everything_state();
        GLRParserState {
            parser: self,
            active_state: initial,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            accepted: false,
        }
    }
    pub fn init_parse_state_substring_with_everything_state(&self) -> ParseState {
        let top = GSSNode::new_fresh().push(ParseStateEdgeContent {
            state_id: self.everything_state_id,
        });
        ParseState {
            stack: Arc::new(top),
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
    }

    pub fn init_parse_state(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> ParseState {
        self.init_parse_state_with_acc()
    }
    pub fn init_parse_state_with_acc(&self) -> ParseState {
        let top =
            GSSNode::new_fresh().push(ParseStateEdgeContent { state_id: self.start_state_id });
        ParseState {
            stack: Arc::new(top),
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
    }

    // --- Batch helpers ---------------------------------------------------------

    pub fn parse(&self, input: &[TerminalID], llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let mut s = self.init_glr_parser(llm_vocab);
        s.parse(input);
        s
    }

    pub fn explain_stack(&self, stack: &[StateID]) -> String {
        let mut result = String::new();
        writeln!(
            &mut result,
            "--- Explaining Parse Stack: {:?} ---",
            stack.iter().map(|s| s.0).collect::<Vec<_>>()
        )
        .unwrap();

        for &state_id in stack {
            writeln!(&mut result, "\nState {}:", state_id.0).unwrap();
            self.format_state_details(&mut result, state_id, "  ").unwrap();
            writeln!(&mut result, "---").unwrap();
        }

        result
    }

    /// A compact, human-readable view of a single row: items, actions, gotos.
    pub fn format_state_details<W: std::fmt::Write>(
        &self,
        f: &mut W,
        state_id: StateID,
        indent: &str,
    ) -> std::fmt::Result {
        let sub = format!("{}  ", indent);

        // Items
        if let Some(items) = self.item_set_map.get_by_right(&state_id) {
            writeln!(f, "{}Items:", indent)?;
            if items.is_empty() {
                writeln!(f, "{}  (None)", indent)?;
            } else {
                let mut grouped: BTreeMap<(&Production, usize), BTreeSet<Option<Terminal>>> =
                    BTreeMap::new();
                for item in items {
                    grouped
                        .entry((&item.production, item.dot_position))
                        .or_default()
                        .insert(item.lookahead.clone());
                }
                for ((production, dot_pos), lookaheads) in grouped {
                    write!(f, "{}- [{} ->", sub, production.lhs.0)?;
                    for (i, symbol) in production.rhs.iter().enumerate() {
                        if i == dot_pos {
                            write!(f, " •")?;
                        }
                        match symbol {
                            Symbol::Terminal(t) => write!(f, " {}", t)?,
                            Symbol::NonTerminal(nt) => write!(f, " {}", nt.0)?,
                        }
                    }
                    if dot_pos == production.rhs.len() {
                        write!(f, " •")?;
                    }
                    write!(f, ", ")?;
                    if lookaheads.len() == 1 {
                        if let Some(look) = lookaheads.iter().next().unwrap() {
                            write!(f, "{}", look)?;
                        } else {
                            write!(f, "ε")?;
                        }
                    } else {
                        write!(f, "{{")?;
                        let mut strs: Vec<String> = lookaheads
                            .iter()
                            .map(|l| {
                                if let Some(t) = l {
                                    t.to_string()
                                } else {
                                    "ε".to_string()
                                }
                            })
                            .collect();
                        strs.sort();
                        const MAX: usize = 5;
                        if strs.len() > MAX {
                            let truncated: Vec<_> = strs.iter().take(MAX).cloned().collect();
                            write!(f, "{}... ({} total)", truncated.join(", "), strs.len())?;
                        } else {
                            write!(f, "{}", strs.join(", "))?;
                        }
                        write!(f, "}}")?;
                    }
                    writeln!(f, "]")?;
                }
            }
        } else {
            writeln!(f, "{}Items: (State ID not found in item set map)", indent)?;
        }

        // Actions & Gotos
        if let Some(row) = self.table.get(&state_id) {
            writeln!(f, "{}Actions (without default reduce):", indent)?;
            format_actions(
                f,
                &row.shifts_and_reduces_without_default_reduce,
                &self.terminal_map,
                &self.non_terminal_map,
                &self.productions,
                &sub,
            )?;

            writeln!(f, "{}Actions (full):", indent)?;
            format_actions(
                f,
                &row.shifts_and_reduces_full,
                &self.terminal_map,
                &self.non_terminal_map,
                &self.productions,
                &sub,
            )?;

            writeln!(f, "{}Default Action:", indent)?;
            if let Some(reduce) = &row.default_reduce.reduce {
                let nt_name = self.non_terminal_map.get_by_right(&reduce.0.nonterminal_id).unwrap();
                let pids: Vec<String> = reduce
                    .0
                    .production_ids
                    .iter()
                    .map(|p| p.0.to_string())
                    .collect();
                writeln!(
                    f,
                    "{}  - Default Reduce {} (len {}) via rules [{}]",
                    indent, nt_name.0, reduce.0.len, pids.join(", ")
                )?;
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
                let mut sorted: Vec<_> = row.gotos.iter().collect();
                sorted.sort_by_key(|(ntid, _)| self.non_terminal_map.get_by_right(ntid).unwrap());
                for (nt_id, goto) in sorted {
                    let nt = self.non_terminal_map.get_by_right(nt_id).unwrap();
                    let goto_str = if let Some(to) = goto.state_id {
                        if goto.accept {
                            format!("{} or accept", to.0)
                        } else {
                            format!("{}", to.0)
                        }
                    } else if goto.accept {
                        "accept".to_string()
                    } else {
                        "no-op".to_string()
                    };
                    writeln!(f, "{}  - {} -> {}", indent, nt.0, goto_str)?;
                }
            }
        } else {
            writeln!(
                f,
                "{}Actions & Gotos: (State ID not found in parse table)",
                indent
            )?;
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

    let mut sorted: Vec<_> = actions.iter().collect();
    sorted.sort_by_key(|(tid, _)| terminal_map.get_by_right(tid).unwrap());

    for (tid, action) in sorted {
        let terminal = terminal_map.get_by_right(tid).unwrap();
        let action_str = match action {
            Stage7ShiftsAndReducesLookaheadValue::Shift(next) => {
                format!("Shift {}", next.0)
            }
            Stage7ShiftsAndReducesLookaheadValue::Reduce {
                nonterminal_id,
                len,
                production_ids,
            } => {
                let nt = non_terminal_map.get_by_right(nonterminal_id).unwrap();
                let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
                format!("Reduce {} (len {}) via rules [{}]", nt.0, len, pids.join(", "))
            }
            Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                let has_shift = shift.is_some();
                let num_reduces: usize =
                    reduces.values().map(|nts| nts.values().map(|v| v.len()).sum::<usize>()).sum();
                let conflict_type = if has_shift && num_reduces > 0 {
                    "Shift-Reduce Conflict"
                } else if !has_shift && num_reduces > 1 {
                    "Reduce-Reduce Conflict"
                } else {
                    "Conflict"
                };

                let mut s = format!("{}:", conflict_type);
                let inner = format!("\n{}        ", indent);
                if let Some(shift_state) = shift {
                    let _ = write!(s, "{}  - Shift {}", inner, shift_state.0);
                }
                for (_len, nts) in reduces {
                    for (_nt_id, prod_ids) in nts {
                        for pid in prod_ids {
                            let prod = productions.get(pid.0).unwrap();
                            let _ = write!(s, "{}  - Reduce by rule #{} ({})", inner, pid.0, prod);
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Parse Table:")?;
        writeln!(f, "  Start State: {}", self.start_state_id.0)?;
        for (&state_id, row) in self.table.iter().collect::<BTreeMap<_, _>>() {
            writeln!(f, "  State {}:", state_id.0)?;
            self.format_state_details(f, state_id, "    ")?;
        }

        writeln!(f, "\nTerminal Map (name to terminal ID):")?;
        for (terminal, terminal_id) in &self.terminal_map {
            writeln!(f, "  {} -> {}", terminal, terminal_id.0)?;
        }

        writeln!(f, "\nNon-Terminal Map:")?;
        for (non_terminal, non_terminal_id) in &self.non_terminal_map {
            writeln!(f, "  {} -> {}", non_terminal.0, non_terminal_id.0)?;
        }

        writeln!(f, "\nSubstring Gotos ({} entries):", self.substring_gotos.len())?;
        if !self.substring_gotos.is_empty() {
            let mut sorted: Vec<_> = self.substring_gotos.iter().collect();
            sorted.sort_by_key(|(ntid, _)| self.non_terminal_map.get_by_right(ntid).unwrap());
            for (nt_id, gotos) in sorted {
                let nt = self.non_terminal_map.get_by_right(nt_id).unwrap();
                writeln!(f, "  - For NT '{}' (ID {}):", nt.0, nt_id.0)?;
                if !gotos.accepting_sources.is_empty() {
                    let sources: Vec<String> =
                        gotos.accepting_sources.iter().map(|s| s.0.to_string()).collect();
                    writeln!(f, "    - accepting sources: [{}]", sources.join(", "))?;
                }
                let mut by_dest: Vec<_> = gotos.gotos.iter().collect();
                by_dest.sort_by_key(|(k, _)| *k);
                for (goto_id, sources) in by_dest {
                    let s: Vec<String> = sources.iter().map(|sid| sid.0.to_string()).collect();
                    writeln!(
                        f,
                        "    - goto: {:<4} from sources: [{}]",
                        goto_id.0,
                        s.join(", ")
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// A compact key for queueing work: deeper stacks first (heuristic).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct WorkMapKey(usize, StateID);
impl PartialOrd for WorkMapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for WorkMapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Deeper first, then state id
        other.0.cmp(&self.0).then_with(|| self.1.cmp(&other.1))
    }
}
type WorkMap = BTreeMap<WorkMapKey, (ParseState, Option<usize>)>;

/// Running parser state. Lightweight to clone, holds references into GLRParser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub active_state: ParseState,
    accepted: bool,
    phase: ParserPhase,
    // A very coarse memo used by substring continuation
    below_bottom_cache:
        HashMap<BelowBottomCacheKey, HashMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BelowBottomCacheKey {
    nonterminal_id: NonTerminalID,
    source_state_id: StateID,
    goto_state_id: StateID,
    k: usize,
}

impl Display for GLRParserState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.log_gss("    ", TerminalID(0), false, false);
        Ok(())
    }
}

impl<'a> GLRParserState<'a> {
    // --- Queue helpers ---------------------------------------------------------

    /// Enqueue a ParseState by splitting it across top-of-stack edges and grouping by depth/state.
    fn enqueue(work: &mut WorkMap, state: ParseState, fuel: Option<usize>) {
        for peek in GSSNode::peek_iter(&state.stack) {
            let isolated = ParseState {
                stack: peek.isolated_parent(),
                accepted_state: state.accepted_state.clone(),
            };
            let depth = isolated.stack.max_depth();
            let sid = peek.edge_value().state_id;
            work.entry(WorkMapKey(depth, sid))
                .and_modify(|(s, existing_fuel)| {
                    s.merge(isolated.clone());
                    *existing_fuel = std::cmp::max(*existing_fuel, fuel);
                })
                .or_insert((isolated, fuel));
        }
    }

    /// Push a new edge onto the parent of `peek`.
    fn push_state(&self, peek: &GSSPeek, content: ParseStateEdgeContent) -> ParseState {
        let gss = peek.push_on_parent(content);
        ParseState {
            stack: Arc::new(gss),
            accepted_state: self.active_state.accepted_state.clone(),
        }
    }

    // --- Generic action-processing inner loop ---------------------------------

    /// Applies either token actions (phase 1/2) or default reduce (phase 3) to a work queue.
    ///
    /// - work/reduce_map: the primary queue and, optionally, a side queue for reductions.
    /// - shifted_states_todo: states that result from Shift or clone-and-merge (defaults).
    /// - accepted_states_todo: states which reached accept during this action processing.
    /// - action_selector(row) -> Some(Action) selects either a specific token action, or
    ///   the row default action (for phase 3).
    fn process_action_queue<F>(
        &mut self,
        work: &mut WorkMap,
        mut reduce_map: Option<&mut WorkMap>,
        shifted_states_todo: &mut VecDeque<ParseState>,
        accepted_states_todo: &mut VecDeque<ParseState>,
        action_selector: F,
        config: &ProcessTokenAdvancedConfig,
        fuel: &mut Option<usize>,
    ) where
        F: for<'r> Fn(&'r Row) -> Option<Action<'r>>,
    {
        // Current implementation doesn't use fuel (kept for compatibility).
        assert!(fuel.is_none(), "Fuel is not supported in process_action_queue yet");
        for (_, (_, per_state_fuel)) in work.iter() {
            assert!(
                per_state_fuel.is_none(),
                "Per-state fuel is not supported in process_action_queue yet"
            );
        }

        while let Some(entry) = work.pop_first() {
            let (key, (state, per_state_fuel)) = entry;

            let row = match self.parser.table.get(&key.1) {
                Some(r) => r,
                None => continue,
            };
            if let Some(action) = action_selector(row) {
                for peek in GSSNode::peek_iter(&state.stack) {
                    match action {
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Shift(to)) => {
                            let pushed =
                                self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                            shifted_states_todo.push_back(pushed);
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                            nonterminal_id: nt,
                            len,
                            ..
                        }) => {
                            if per_state_fuel == Some(0) {
                                continue;
                            }
                            let (s_new, accepted_new) =
                                self.reduce_and_goto(&peek, *nt, *len, &action_selector, config);
                            if !s_new.is_empty() {
                                let new_state = ParseState {
                                    stack: s_new,
                                    accepted_state: state.accepted_state.clone(),
                                };
                                if let Some(ref mut rmap) = reduce_map {
                                    Self::enqueue(rmap, new_state, per_state_fuel.map(|f| f - 1));
                                } else {
                                    Self::enqueue(work, new_state, per_state_fuel.map(|f| f - 1));
                                }
                            }
                            if !accepted_new.is_empty() {
                                self.accepted = true;
                                let accepted_parse_state = ParseState {
                                    stack: Arc::new(GSSNode::new_fresh()),
                                    accepted_state: accepted_new,
                                };
                                accepted_states_todo.push_back(accepted_parse_state);
                            }
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Split {
                            shift,
                            reduces,
                        }) => {
                            if let Some(to) = shift {
                                let pushed =
                                    self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                                shifted_states_todo.push_back(pushed);
                            }
                            if per_state_fuel != Some(0) {
                                for (len, nts) in reduces {
                                    for (nt, _prod_ids) in nts {
                                        let (s_new, accepted_new) = self.reduce_and_goto(
                                            &peek,
                                            *nt,
                                            *len,
                                            &action_selector,
                                            config,
                                        );
                                        if !s_new.is_empty() {
                                            let new_state = ParseState {
                                                stack: s_new,
                                                accepted_state: state.accepted_state.clone(),
                                            };
                                            if let Some(ref mut rmap) = reduce_map {
                                                Self::enqueue(
                                                    rmap,
                                                    new_state,
                                                    per_state_fuel.map(|f| f - 1),
                                                );
                                            } else {
                                                Self::enqueue(
                                                    work,
                                                    new_state,
                                                    per_state_fuel.map(|f| f - 1),
                                                );
                                            }
                                        }
                                        if !accepted_new.is_empty() {
                                            self.accepted = true;
                                            let accepted_parse_state = ParseState {
                                                stack: Arc::new(GSSNode::new_fresh()),
                                                accepted_state: accepted_new,
                                            };
                                            accepted_states_todo.push_back(accepted_parse_state);
                                        }
                                    }
                                }
                            }
                        }
                        Action::Default(default_reduce) => {
                            // Clone-and-merge: keep "current" as a survivor.
                            if default_reduce.clone_and_merge {
                                shifted_states_todo.push_back(state.clone());
                            }
                            if let Some((reduce, allowed_terminals)) = &default_reduce.reduce {
                                if per_state_fuel != Some(0) {
                                    // Constrain by allowed terminals.
                                    let mut constrained = state.clone();
                                    if constrained.stack.is_alive() {
                                        let disallowed = allowed_terminals.inverted();
                                        if !disallowed.is_empty() {
                                            let disallowed_l2 = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::from_iter(
                                                std::iter::once((0..=usize::MAX, disallowed))
                                            );
                                            crate::datastructures::gss::disallow_terminals_and_prune_arc(
                                                &mut constrained.stack,
                                                &disallowed_l2,
                                                &mut HashMap::new(),
                                            );
                                        }

                                        if !constrained.stack.is_empty() {
                                            for peek2 in GSSNode::peek_iter(&constrained.stack) {
                                                let (s_new, accepted_new) = self.reduce_and_goto(
                                                    &peek2,
                                                    reduce.nonterminal_id,
                                                    reduce.len,
                                                    &action_selector,
                                                    config,
                                                );
                                                if !s_new.is_empty() {
                                                    let new_state = ParseState {
                                                        stack: s_new,
                                                        accepted_state: state.accepted_state.clone(),
                                                    };
                                                    if let Some(ref mut rmap) = reduce_map {
                                                        Self::enqueue(
                                                            rmap,
                                                            new_state,
                                                            per_state_fuel.map(|f| f - 1),
                                                        );
                                                    } else {
                                                        Self::enqueue(
                                                            work,
                                                            new_state,
                                                            per_state_fuel.map(|f| f - 1),
                                                        );
                                                    }
                                                }
                                                if !accepted_new.is_empty() {
                                                    self.accepted = true;
                                                    let accepted_parse_state = ParseState {
                                                        stack: Arc::new(GSSNode::new_fresh()),
                                                        accepted_state: accepted_new,
                                                    };
                                                    accepted_states_todo.push_back(
                                                        accepted_parse_state,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Phase 1 & 2 wrappers --------------------------------------------------

    fn phase1_without_default(
        &mut self,
        token_id: TerminalID,
        phase1_todo: &mut WorkMap,
        phase2_todo: &mut WorkMap,
        shifted: &mut VecDeque<ParseState>,
        accepted: &mut VecDeque<ParseState>,
        config: &ProcessTokenAdvancedConfig,
    ) {
        let tid = token_id;
        self.process_action_queue(
            phase1_todo,
            Some(phase2_todo),
            shifted,
            accepted,
            move |row| row
                .shifts_and_reduces_without_default_reduce
                .get(&tid)
                .map(Action::Normal),
            config,
            &mut None,
        );
    }

    fn phase2_with_default(
        &mut self,
        token_id: TerminalID,
        phase2_todo: &mut WorkMap,
        shifted: &mut VecDeque<ParseState>,
        accepted: &mut VecDeque<ParseState>,
        config: &ProcessTokenAdvancedConfig,
    ) {
        let tid = token_id;
        self.process_action_queue(
            phase2_todo,
            None,
            shifted,
            accepted,
            move |row| {
                row.shifts_and_reduces_full
                    .get(&tid)
                    .map(Action::Normal)
                    .or_else(|| Some(Action::Default(&row.default_reduce)))
            },
            config,
            &mut None,
        );
        self.phase = ParserPhase::ReadyForDefaultReductions;
    }

    // --- Reduce + Goto ---------------------------------------------------------

    /// Reduce by non-terminal `nt` of length `len`, then perform gotos.
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
        let popper: GSSPopper = timeit!(peek.popn(len));
        let mut active_out: Vec<Arc<GSSNode>> = Vec::new();
        let mut accepted_out: Vec<Arc<GSSNode>> = Vec::new();

        // 1) Standard in-graph reductions
        for pop_item in popper.iter() {
            for peek2 in pop_item.peek_iter() {
                // Follow a chain of unit reductions quickly on the goto side
                let predecessor_state_id = peek2.edge_value().state_id;
                let mut current_nt = nt;

                loop {
                    let goto = self
                        .parser
                        .table
                        .get(&predecessor_state_id)
                        .and_then(|row| row.gotos.get(&current_nt))
                        .expect_else(|| {
                            format!(
                                "Goto not found for NT '{}' in state {:?}",
                                self.parser
                                    .non_terminal_map
                                    .get_by_right(&current_nt)
                                    .unwrap(),
                                predecessor_state_id
                            )
                        });

                    // Accept graph contribution
                    if goto.accept {
                        accepted_out.push(peek2.isolated_parent());
                    }

                    if let Some(to) = goto.state_id {
                        let next_row = &self.parser.table[&to];
                        match action_selector(next_row) {
                            Some(Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                nonterminal_id: next_nt,
                                len: 1,
                                ..
                            })) => {
                                // Continue chaining unit reductions
                                current_nt = *next_nt;
                                continue;
                            }
                            Some(Action::Default(def)) => {
                                // If default isn't a unit reduce, we must materialize the goto edge.
                                if def.clone_and_merge
                                    || def
                                        .reduce
                                        .as_ref()
                                        .map_or(false, |r| r.0.len != 1)
                                {
                                    active_out.push(Arc::new(peek2.push_on_parent(
                                        ParseStateEdgeContent { state_id: to },
                                    )));
                                }
                                // Unit-reduction default? chain again.
                                if let Some(reduce) = &def.reduce {
                                    if reduce.0.len == 1 {
                                        current_nt = reduce.0.nonterminal_id;
                                        continue;
                                    }
                                }
                                // Otherwise done here.
                                break;
                            }
                            _ => {
                                // Not a unit reduce path -> emit a single push-to-goto-state.
                                active_out.push(Arc::new(peek2.push_on_parent(
                                    ParseStateEdgeContent { state_id: to },
                                )));
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

        // 2) Below-bottom handling (substring continuation)
        if !popper.below_bottom().is_empty() {
            match config.below_bottom_mode {
                BelowBottomReductionMode::Fail => {
                    // Do nothing for below-bottom paths.
                }
                BelowBottomReductionMode::Panic => {
                    panic!(
                        "A reduction popped below the bottom of the stack, and BelowBottomReductionMode::Panic is set."
                    );
                }
                _ => {
                    // Build Accs aggregated by k, then continue from either all states or the everything state.
                    let below_accs = self.build_below_bottom_accs(&popper);

                    // Determine gotos for `nt` depending on mode.
                    let mut storage = SubstringGoto::default();
                    let gotos_for_nt = self.substring_gotos_for(nt, config, &mut storage);

                    // Accept contributions (if any)
                    if let Some(accepted_merged) =
                        self.handle_below_bottom_accepts(nt, &below_accs, gotos_for_nt)
                    {
                        accepted_out.push(accepted_merged);
                    }

                    // Non-accepting gotos
                    let merged_below = self.handle_below_bottom_gotos(nt, below_accs, gotos_for_nt);
                    active_out.push(merged_below);
                }
            }
        }

        (
            GSSNode::merge_many_with_depth(usize::MAX, active_out),
            GSSNode::merge_many_with_depth(usize::MAX, accepted_out),
        )
    }

    // --- Token processing ------------------------------------------------------

    pub fn process_token(&mut self, token_id: TerminalID) {
        self.process_token_advanced(token_id, &ProcessTokenAdvancedConfig::default())
    }

    #[time_it("GLRParserState::process_token_advanced")]
    pub fn process_token_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
        // Reset per-token acceptance and accepted graph.
        self.accepted = false;
        self.active_state.accepted_state = Arc::new(GSSNode::new_fresh());
        self.below_bottom_cache.clear();

        // Ignore token path (e.g., whitespace).
        if Some(token_id) == self.parser.ignore_terminal_id {
            self.phase = ParserPhase::ReadyForDefaultReductions;
            return;
        }

        self.log_gss("Phase1/2-start", token_id, false, false);

        // Phase 1 (if needed) -> fill phase 2; Phase 2 -> do with default.
        let mut phase2_todo: WorkMap = WorkMap::new();
        let mut shifted_todo: VecDeque<ParseState> = VecDeque::new();
        let mut accepted_todo: VecDeque<ParseState> = VecDeque::new();

        if self.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: WorkMap = WorkMap::new();
            Self::enqueue(&mut phase1_todo, self.active_state.clone(), None);
            self.phase1_without_default(
                token_id,
                &mut phase1_todo,
                &mut phase2_todo,
                &mut shifted_todo,
                &mut accepted_todo,
                config,
            );
        } else {
            // Already allowed to use defaults; enqueue directly for phase 2.
            Self::enqueue(&mut phase2_todo, self.active_state.clone(), None);
        }

        // Phase 2
        self.phase2_with_default(
            token_id,
            &mut phase2_todo,
            &mut shifted_todo,
            &mut accepted_todo,
            config,
        );

        // Consolidate survivors and accepted contributions.
        let mut next_active = ParseState::new();
        for state in shifted_todo {
            next_active.merge(state);
        }
        for state in accepted_todo {
            next_active.merge(state);
        }
        self.active_state = next_active;
        self.log_gss("Phase1/2-end", token_id, false, false);

        self.below_bottom_cache.clear();
    }

    pub fn process_default_reductions(&mut self) {
        self.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig::default());
    }

    /// Phase 3: Apply default reductions (no token) until a fixpoint.
    #[time_it("GLRParserState::process_default_reductions_advanced")]
    pub fn process_default_reductions_advanced(&mut self, config: &ProcessDefaultReductionsAdvancedConfig) {
        self.log_gss("Phase3-start", TerminalID(0), false, false);
        if self.phase == ParserPhase::ReadyForToken {
            // Already closed.
            return;
        }
        assert_eq!(self.phase, ParserPhase::ReadyForDefaultReductions);

        let mut work: WorkMap = WorkMap::new();
        Self::enqueue(&mut work, self.active_state.clone(), config.per_state_fuel);

        let mut shifted: VecDeque<ParseState> = VecDeque::new();
        let mut accepted: VecDeque<ParseState> = VecDeque::new();
        let mut fuel = config.fuel;
        let token_cfg = ProcessTokenAdvancedConfig {
            below_bottom_mode: config.below_bottom_mode,
        };

        self.process_action_queue(
            &mut work,
            None,
            &mut shifted,
            &mut accepted,
            |row| Some(Action::Default(&row.default_reduce)),
            &token_cfg,
            &mut fuel,
        );

        let mut next_active = ParseState::new();
        for s in shifted {
            next_active.merge(s);
        }
        for s in accepted {
            next_active.merge(s);
        }
        for (_, (s, _)) in work {
            next_active.merge(s);
        }
        self.active_state = next_active;

        self.phase = ParserPhase::ReadyForToken;
        self.log_gss("Phase3-end", TerminalID(0), false, false);
    }

    /// Lightweight check: if any action exists for `token_id`, return the union of
    /// allowed LLM tokens across the active GSS peeks (LR(1) and similar modes).
    pub fn has_action_for(&self, token_id: TerminalID) -> Option<LLMTokenBV> {
        match LR_MODE {
            LRMode::LR1 | LRMode::LALR_EX_SHIFT_STATES => {
                if Some(token_id) == self.parser.ignore_terminal_id {
                    return Some(LLMTokenBV::max_ones());
                }
                self.log_gss("has_action_for-start", token_id, false, false);
                let mut llm_tokens = LLMTokenBV::zeros();
                for peek in GSSNode::peek_iter(&self.active_state.stack) {
                    let row = &self.parser.table[&peek.edge_value().state_id];
                    let action_opt = match self.phase {
                        ParserPhase::ReadyForToken => row
                            .shifts_and_reduces_without_default_reduce
                            .get(&token_id)
                            .map(Action::Normal),
                        ParserPhase::ReadyForDefaultReductions => row
                            .shifts_and_reduces_full
                            .get(&token_id)
                            .map(Action::Normal)
                            .or_else(|| Some(Action::Default(&row.default_reduce))),
                    };
                    if action_opt.is_some() {
                        let peek_tokens = peek.resolved_llm_tokens_union();
                        llm_tokens |= peek_tokens;
                    }
                }
                Some(llm_tokens)
            }
            LRMode::LALR => None,
        }
    }

    // --- Batch utilities -------------------------------------------------------

    pub fn step(&mut self, token_id: TerminalID) {
        self.process_token(token_id);
    }
    pub fn parse(&mut self, input: &[TerminalID]) {
        self.parse_part(input);
    }
    pub fn parse_part(&mut self, input: &[TerminalID]) {
        for &tid in input {
            self.step(tid);
        }
    }
    pub fn step_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
        self.process_token_advanced(token_id, config);
    }
    pub fn parse_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        self.parse_part_advanced(input, config);
    }
    pub fn parse_part_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        for &tid in input {
            self.step_advanced(tid, config);
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
        // Kept for compatibility; merges happen in enqueue/merge paths.
    }

    /// Merge two GLR states (must use the same parser).
    pub fn merge_with(&mut self, mut other: GLRParserState) {
        assert!(std::ptr::eq(self.parser, other.parser));
        match (self.phase, other.phase) {
            (ParserPhase::ReadyForToken, ParserPhase::ReadyForDefaultReductions) => {
                self.process_default_reductions()
            }
            (ParserPhase::ReadyForDefaultReductions, ParserPhase::ReadyForToken) => {
                other.process_default_reductions()
            }
            _ => {}
        }
        self.active_state.merge(other.active_state);
        self.accepted |= other.accepted;
    }

    pub fn is_ok(&self) -> bool {
        self.accepted || (!self.active_state.stack.is_empty() && self.active_state.stack.is_alive())
    }
    pub fn has_accepted(&self) -> bool {
        self.accepted
    }

    // --- Logging ---------------------------------------------------------------

    pub fn log_gss(&self, phase: &str, token: TerminalID, explain_states: bool, generate_dot: bool) {
        if !GSS_LOGGING_ENABLED {
            return;
        }
        const MAX_EDGES_TO_PRINT: usize = 100;
        const PANIC_THRESHOLD: usize = 1_000_000;

        let mut roots_to_log: Vec<(&str, Arc<GSSNode>)> =
            vec![("Active", self.active_state.stack.clone())];
        if !self.active_state.accepted_state.is_empty() {
            roots_to_log.push(("Accepted", self.active_state.accepted_state.clone()));
        }

        let stats_breakdown = roots_to_log
            .iter()
            .map(|(name, root)| {
                let stats = gather_gss_stats(&[root.as_ref()]);
                format!("{}_nodes: {:?}", name.to_lowercase(), stats)
            })
            .collect::<Vec<_>>()
            .join(" ");

        crate::debug!(
            3,
            "{} ({:?}) - accepted: {} - token '{}' ({}) - {}",
            phase,
            self.phase,
            self.accepted,
            self.parser.terminal_map.get_by_right(&token).unwrap(),
            token.0,
            stats_breakdown
        );

        let mut gss_strings = vec![];
        let mut all_state_ids = BTreeSet::new();
        let mut total_nodes = 0;

        for (name, root) in &roots_to_log {
            let stats = gather_gss_stats(&[root.as_ref()]);
            total_nodes += stats.unique_nodes;

            let (string, ids) = {
                let print_full = stats.total_edges <= MAX_EDGES_TO_PRINT;
                let max_edges = if print_full { usize::MAX } else { MAX_EDGES_TO_PRINT };
                let config = GSSPrintConfig {
                    max_edges,
                    ..Default::default()
                };
                let (s, state_ids) = print_gss_forest(&[root.clone()], &self.parser.terminal_map, &config);
                let summary = if print_full {
                    format!(
                        "{} GSS ({} nodes, {} edges):\n{}",
                        name, stats.unique_nodes, stats.total_edges, s
                    )
                } else {
                    match find_longest_path(root) {
                        Some(p) => format!(
                            "{} GSS too big ({} nodes, {} edges). Longest path ({}): {}",
                            name,
                            stats.unique_nodes,
                            stats.total_edges,
                            p.len(),
                            p.iter()
                                .map(|(ec, _n)| ec.state_id.0.to_string())
                                .collect::<Vec<_>>()
                                .join(" → ")
                        ),
                        None => format!(
                            "{} GSS too big ({} nodes, {} edges) – path not found",
                            name, stats.unique_nodes, stats.total_edges
                        ),
                    }
                };
                (summary, state_ids)
            };
            gss_strings.push(string);
            all_state_ids.extend(ids);
        }

        let mut final_string = gss_strings.join("\n\n");
        if explain_states && !all_state_ids.is_empty() {
            final_string.push_str("\n\n--- GSS State Explanations ---\n");
            for state_id in all_state_ids {
                let mut explanation = String::new();
                writeln!(&mut explanation, "\n--- State {} ---", state_id.0).unwrap();
                self.parser
                    .format_state_details(&mut explanation, state_id, "  ")
                    .unwrap();
                final_string.push_str(&explanation);
            }
        }

        if total_nodes > PANIC_THRESHOLD {
            panic!("GSS too big ({} nodes). {}", total_nodes, final_string);
        }
        crate::debug!(3, "{}", final_string);

        if generate_dot {
            let dot = self.gss_to_dot();
            crate::debug!(1, "GSS DOT graph:\n{}", dot);
        }
    }

    /// Graphviz DOT for the current GSS forest (active + accepted if present).
    pub fn gss_to_dot(&self) -> String {
        let mut roots: Vec<(&str, &GSSNode)> = vec![("Active", &self.active_state.stack)];
        if !self.active_state.accepted_state.is_empty() {
            roots.push(("Accepted", &self.active_state.accepted_state));
        }
        self.parser.gss_forest_to_dot(&roots, None, None)
    }

    // --- Substring Helpers (below-bottom) -------------------------------------

    #[inline]
    fn substring_gotos_for<'b>(
        &self,
        nt: NonTerminalID,
        config: &ProcessTokenAdvancedConfig,
        storage: &'b mut SubstringGoto,
    ) -> &'b SubstringGoto
    where
        'a: 'b,
    {
        match config.below_bottom_mode {
            BelowBottomReductionMode::ContinueFromAll => {
                self.parser.substring_gotos.get(&nt).unwrap_or(storage)
            }
            BelowBottomReductionMode::ContinueFromEverything => {
                // Compose a tiny SubstringGoto from the synthetic "everything" state.
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
                    if let Some(to) = goto.state_id {
                        storage.gotos.insert(to, BTreeSet::from([everything]));
                    }
                }
                storage
            }
            BelowBottomReductionMode::Fail => storage,
            BelowBottomReductionMode::Panic => storage,
        }
    }

    fn build_below_bottom_accs(&self, popper: &GSSPopper) -> BTreeMap<usize, Acc> {
        let mut result: BTreeMap<usize, Acc> = BTreeMap::new();
        for (k, accs_by_edge) in popper.below_bottom() {
            for (last_edge_content, acc_arc) in accs_by_edge {
                let acc = acc_arc.as_ref();

                let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> =
                    BTreeMap::new();
                let edge_key = (0, Some(last_edge_content.state_id));
                let mut new_acc = acc.clone();

                let mut used = BTreeSet::new();
                for existing in &acc.trie2_nodes {
                    let source_arc = existing.as_arc().clone();
                    let source_live = { source_arc.read().expect("poison").value.live_tokens.clone() };
                    let tokens_to_push = &source_live & &acc.llm_tokens_union;
                    if tokens_to_push.is_empty() {
                        continue;
                    }

                    let mut inserter = EdgeInserter::new(
                        source_arc.clone(),
                        edge_key,
                        tokens_to_push.clone(),
                        |e, n| *e |= n,
                        |_, _| {},
                        |ev, t| *ev &= &t.live_tokens,
                    );

                    // Prefer eligible strong children
                    let eligible_iter = || {
                        let g = source_arc.read().expect("poison");
                        let mut v = Vec::new();
                        if let Some(dest_map) = g.children().get(&edge_key) {
                            for (node_ptr, _ev) in dest_map.iter() {
                                if !node_ptr.is_strong() {
                                    continue;
                                }
                                if let Some(dest_arc) = node_ptr.upgrade() {
                                    let dl = dest_arc.read().expect("poison").value.live_tokens.clone();
                                    if (&dl & &tokens_to_push).is_empty()
                                        && !dest_arc.read().expect("poison").value.end
                                    {
                                        v.push(dest_arc.clone());
                                    }
                                }
                            }
                        }
                        v.into_iter()
                    };

                    let fallback = Arc::new(RwLock::new(PrecomputeNode2::new(
                        PrecomputedNodeContents::internal(),
                    )));
                    inserter = inserter
                        .try_destinations_iter_with(eligible_iter)
                        .try_destination_auto(fallback);

                    let final_dest_arc =
                        inserter.clone_into_option().expect("below-bottom Acc construction failed");
                    let final_wr = ArcPtrWrapper::new(final_dest_arc.clone());
                    dest_agg
                        .entry(final_wr.clone())
                        .and_modify(|bv| *bv |= &tokens_to_push)
                        .or_insert(tokens_to_push.clone());
                    used.insert(final_wr);
                }

                for (dst_wr, added) in &dest_agg {
                    let mut g = dst_wr.as_arc().write().expect("poison");
                    g.value.live_tokens |= added.clone();
                }

                new_acc.trie2_nodes = used;
                result
                    .entry(*k)
                    .and_modify(|existing| *existing = Acc::merge(existing, &new_acc))
                    .or_insert(new_acc);
            }
        }
        result
    }

    fn handle_below_bottom_accepts(
        &self,
        _nt: NonTerminalID,
        below: &BTreeMap<usize, Acc>,
        gotos: &SubstringGoto,
    ) -> Option<Arc<GSSNode>> {
        if gotos.accepting_sources.is_empty() {
            return None;
        }
        let mut accepted_stacks: Vec<Arc<GSSNode>> = Vec::new();

        for (k, acc) in below {
            let mut acc = acc.clone();
            let trie2_nodes = std::mem::take(&mut acc.trie2_nodes);
            let active_tokens = acc.union_llm_tokens();

            for source_state_id in &gotos.accepting_sources {
                let edge_key = (*k, Some(*source_state_id));
                let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> =
                    BTreeMap::new();
                let mut used = BTreeSet::new();

                let new_trie2_node = Arc::new(RwLock::new(PrecomputeNode2::new(
                    PrecomputedNodeContents::internal(),
                )));

                for existing in &trie2_nodes {
                    let source_arc = existing.as_arc().clone();
                    let source_live = { source_arc.read().expect("poison").value.live_tokens.clone() };
                    let tokens_to_push = &source_live & &active_tokens;
                    if tokens_to_push.is_empty() {
                        continue;
                    }

                    let mut inserter = EdgeInserter::new(
                        source_arc.clone(),
                        edge_key,
                        tokens_to_push.clone(),
                        |e, n| *e |= n,
                        |_, _| {},
                        |ev, t| *ev &= &t.live_tokens,
                    );

                    let eligible_iter = || {
                        let g = source_arc.read().expect("poison");
                        let mut v = Vec::new();
                        if let Some(dest_map) = g.children().get(&edge_key) {
                            for (node_ptr, _ev) in dest_map.iter() {
                                if !node_ptr.is_strong() {
                                    continue;
                                }
                                if let Some(dest_arc) = node_ptr.upgrade() {
                                    let dl = dest_arc.read().expect("poison").value.live_tokens.clone();
                                    if (&dl & &tokens_to_push).is_empty()
                                        && !dest_arc.read().expect("poison").value.end
                                    {
                                        v.push(dest_arc.clone());
                                    }
                                }
                            }
                        }
                        v.into_iter()
                    };
                    inserter = inserter
                        .try_destinations_iter_with(eligible_iter)
                        .try_destination_auto(new_trie2_node.clone());

                    let final_dest_arc = inserter
                        .clone_into_option()
                        .expect("below-bottom accepting: EdgeInserter returned no destination");
                    let final_wr = ArcPtrWrapper::new(final_dest_arc.clone());
                    dest_agg
                        .entry(final_wr.clone())
                        .and_modify(|bv| *bv |= &tokens_to_push)
                        .or_insert(tokens_to_push.clone());
                    used.insert(final_wr);
                }

                for (dst_wr, added) in &dest_agg {
                    let mut dg = dst_wr.as_arc().write().expect("poison");
                    dg.value.live_tokens |= added.clone();
                }

                let mut acc2 = acc.clone();
                acc2.trie2_nodes = used;
                let gss0 = GSSNode::new(acc2);
                let gss1 = gss0.push(ParseStateEdgeContent {
                    state_id: *source_state_id,
                });
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
        _nt: NonTerminalID,
        below: BTreeMap<usize, Acc>,
        gotos: &SubstringGoto,
    ) -> Arc<GSSNode> {
        if gotos.gotos.is_empty() {
            return Arc::new(GSSNode::new_fresh());
        }

        let mut below_zero: Vec<Arc<GSSNode>> = Vec::new();
        let mut trie2_dst_nodes: HashMap<StateID, Arc<RwLock<PrecomputeNode2>>> = HashMap::new();

        for (k, acc) in below {
            let mut acc = acc.clone();
            let trie2_nodes = std::mem::take(&mut acc.trie2_nodes);
            let edge_key = (k, None);

            for (goto_state_id, source_state_ids) in &gotos.gotos {
                for source_state_id in source_state_ids {
                    // Coarse cache key (as before).
                    let cache_key = BelowBottomCacheKey {
                        nonterminal_id: NonTerminalID(0),
                        source_state_id: *source_state_id,
                        goto_state_id: *goto_state_id,
                        k: 0,
                    };

                    let mut dest_agg: BTreeMap<ArcPtrWrapper<RwLock<PrecomputeNode2>>, LLMTokenBV> =
                        BTreeMap::new();
                    let mut used_dests: BTreeSet<ArcPtrWrapper<RwLock<PrecomputeNode2>>> =
                        BTreeSet::new();

                    let new_trie2_node = trie2_dst_nodes
                        .entry(*source_state_id)
                        .or_insert_with(|| {
                            Arc::new(RwLock::new(PrecomputeNode2::new(
                                PrecomputedNodeContents::internal(),
                            )))
                        })
                        .clone();

                    for existing in &trie2_nodes {
                        let source_arc = existing.as_arc().clone();
                        let source_live = { source_arc.read().expect("poison").value.live_tokens.clone() };
                        let tokens_to_push = &source_live & &acc.llm_tokens_union;
                        if tokens_to_push.is_empty() {
                            continue;
                        }

                        let mut inserter = EdgeInserter::new(
                            source_arc.clone(),
                            edge_key,
                            tokens_to_push.clone(),
                            |e, n| *e |= n,
                            |_, _| {},
                            |ev, t| *ev &= &t.live_tokens,
                        );

                        // Try cached destinations (weak edges) if compatible
                        if let Some(cached_entries) = self.below_bottom_cache.get(&cache_key) {
                            let eligible_cached_iter =
                                cached_entries.iter().filter_map(|(wrapper, cached_tokens)| {
                                    let dest_arc = wrapper.as_arc();
                                    let guard = dest_arc.read().expect("poison");
                                    let temp = &guard.value.live_tokens - &cached_tokens;
                                    if (&temp & &tokens_to_push).is_empty() && !guard.value.end {
                                        Some(dest_arc.clone())
                                    } else {
                                        None
                                    }
                                });
                            inserter = inserter.to_destinations_weakly_iter(eligible_cached_iter);
                        }

                        // Try strong children next
                        let eligible_iter = || {
                            let g = source_arc.read().expect("poison");
                            let mut v = Vec::new();
                            if let Some(dest_map) = g.children().get(&edge_key) {
                                for (node_ptr, _ev) in dest_map.iter() {
                                    if !node_ptr.is_strong() {
                                        continue;
                                    }
                                    if let Some(dest_arc) = node_ptr.upgrade() {
                                        let guard = dest_arc.read().expect("poison");
                                        if (&guard.value.live_tokens & &tokens_to_push).is_empty()
                                            && !guard.value.end
                                        {
                                            v.push(dest_arc.clone());
                                        }
                                    }
                                }
                            }
                            v.into_iter()
                        };
                        inserter = inserter
                            .try_destinations_iter_with(eligible_iter)
                            .try_destination_auto(new_trie2_node.clone());

                        let final_dest_arc = inserter
                            .clone_into_option()
                            .expect("below-bottom goto: EdgeInserter failed");
                        let final_wr = ArcPtrWrapper::new(final_dest_arc.clone());
                        dest_agg
                            .entry(final_wr.clone())
                            .and_modify(|bv| *bv |= &tokens_to_push)
                            .or_insert(tokens_to_push.clone());
                    }

                    // Update cache and live tokens
                    let cache_entry = self.below_bottom_cache.entry(cache_key).or_default();
                    for (dest_wrapper, new_tokens) in &dest_agg {
                        if let Some(existing) = cache_entry.get(dest_wrapper) {
                            if !new_tokens.is_subset(existing) {
                                used_dests.insert(dest_wrapper.clone());
                            }
                        } else {
                            used_dests.insert(dest_wrapper.clone());
                        }
                        cache_entry
                            .entry(dest_wrapper.clone())
                            .and_modify(|bv| *bv |= new_tokens.clone())
                            .or_insert(new_tokens.clone());
                    }
                    for (dst_wr, added) in &dest_agg {
                        let mut guard = dst_wr.as_arc().write().expect("poison");
                        guard.value.live_tokens |= added.clone();
                    }

                    if !used_dests.is_empty() {
                        let mut acc2 = acc.clone();
                        acc2.trie2_nodes = used_dests;
                        let g0 = GSSNode::new(acc2);
                        let g1 = g0.push(ParseStateEdgeContent {
                            state_id: *source_state_id,
                        });
                        let g2 = g1.push(ParseStateEdgeContent {
                            state_id: *goto_state_id,
                        });
                        below_zero.push(Arc::new(g2));
                    }
                }
            }
        }

        GSSNode::merge_many_with_depth(usize::MAX, below_zero)
    }
}

impl GLRParser {
    /// Render a Graphviz DOT representation for a GSS forest.
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

        let mut visited = HashSet::new();
        let mut node_ids = HashMap::new();
        let mut edge_node_ids = HashMap::new();
        let mut next_id = 0;

        let mut queue: VecDeque<Arc<GSSNode>> =
            roots.iter().map(|(_, n)| Arc::new((*n).clone())).collect();

        // Root clusters and edges to actual nodes
        for (i, (label, root)) in roots.iter().enumerate() {
            let root_ptr = *root as *const GSSNode;
            let root_id = *node_ids.entry(root_ptr).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            writeln!(&mut dot, "  subgraph cluster_{} {{", i).unwrap();
            writeln!(&mut dot, "    label=\"{}\";", label).unwrap();
            writeln!(&mut dot, "    style=filled;").unwrap();
            writeln!(&mut dot, "    color=lightgrey;").unwrap();
            writeln!(&mut dot, "    node [style=filled,color=white];").unwrap();
            let root_node_name = format!("Root_{}", i);
            writeln!(
                &mut dot,
                "    {} [label=\"{}\", shape=ellipse];",
                root_node_name, root_id
            )
            .unwrap();
            writeln!(&mut dot, "  }}").unwrap();
            writeln!(&mut dot, "  {} -> N{};", root_node_name, root_id).unwrap();
        }

        while let Some(node_arc) = queue.pop_front() {
            let node_ptr = Arc::as_ptr(&node_arc);
            if visited.contains(&node_ptr) {
                continue;
            }

            let parent_id = *node_ids.entry(node_ptr).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            if visited.insert(node_ptr) {
                let acc_str = format_acc(
                    &node_arc,
                    &self.terminal_map,
                    original_internal_bimap,
                    llm_token_map,
                );
                let escaped = acc_str
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\l")
                    .replace('{', "\\{")
                    .replace('}', "\\}")
                    .replace('<', "\\<")
                    .replace('>', "\\>")
                    .replace('\'', "\\'");
                writeln!(
                    &mut dot,
                    "  N{} [label=\"Node {}\\lDepth: {}\\l{}\"];",
                    parent_id,
                    parent_id,
                    node_arc.max_depth(),
                    escaped
                )
                .unwrap();
            }

            for (edge_val, preds_by_depth) in node_arc.predecessors() {
                let state_id = edge_val.state_id;
                let edge_key = (node_ptr, edge_val.clone());
                let edge_node_id = *edge_node_ids.entry(edge_key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;

                    // A plaintext node for the edge with the state's details
                    let mut explanation = String::new();
                    self.format_state_details(&mut explanation, state_id, "").unwrap();
                    let esc = explanation
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\l")
                        .replace('{', "\\{")
                        .replace('}', "\\}")
                        .replace('<', "\\<")
                        .replace('>', "\\>")
                        .replace('\'', "\\'");
                    writeln!(
                        &mut dot,
                        "  E{} [label=\"State {}\\l{}\", shape=plaintext, fontname=\"Courier New\"];",
                        id, state_id.0, esc
                    )
                    .unwrap();
                    id
                });

                // Connect parent to edge-node
                writeln!(&mut dot, "  N{} -> E{};", parent_id, edge_node_id).unwrap();

                for pred_vec in preds_by_depth.values() {
                    for pred_arc in pred_vec {
                        let pred_ptr = Arc::as_ptr(pred_arc);
                        let pred_id = *node_ids.entry(pred_ptr).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            id
                        });
                        writeln!(&mut dot, "  E{} -> N{} [arrowhead=none];", edge_node_id, pred_id)
                            .unwrap();
                        queue.push_back(pred_arc.clone());
                    }
                }
            }
        }

        writeln!(&mut dot, "}}").unwrap();
        dot
    }

    /// Graphviz DOT for a single-root view.
    pub fn gss_to_dot(
        &self,
        root: &GSSNode,
        original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
        llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    ) -> String {
        self.gss_forest_to_dot(&[("Root", root)], original_internal_bimap, llm_token_map)
    }
}

// --- Misc compatibility helpers -----------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
}

pub trait InsertWith<K, V> {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F);
}
impl<K, V> InsertWith<K, V> for BTreeMap<K, V>
where
    K: Eq + Ord,
{
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F) {
        match self.entry(k) {
            std::collections::btree_map::Entry::Occupied(mut o) => {
                let value = o.get_mut();
                combine(value, v);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(v);
            }
        }
    }
}

/// Fast default-reduction chain helper — kept for compatibility and potential debugging.
fn default_reduce_chain(
    parser: &GLRParser,
    start_state_id: StateID,
    initial_nt: NonTerminalID,
) -> BTreeSet<StateID> {
    let mut final_goto_state_ids = BTreeSet::new();
    let mut current_nt = initial_nt;
    let goto_source_state_id = start_state_id;

    loop {
        if let Some(goto) = parser
            .table
            .get(&goto_source_state_id)
            .and_then(|row| row.gotos.get(&current_nt))
        {
            if let Some(goto_state_id) = goto.state_id {
                let next_row = &parser.table[&goto_state_id];
                if let Some(next_reduce) = &next_row.default_reduce.reduce {
                    if next_reduce.0.len == 1 {
                        current_nt = next_reduce.0.nonterminal_id;
                        continue;
                    }
                }
                final_goto_state_ids.insert(goto_state_id);
                break;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    final_goto_state_ids
}
