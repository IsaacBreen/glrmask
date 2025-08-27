use std::any::Any;
use std::cmp::Ordering;
use crate::datastructures::gss::{print_gss_forest, Acc, GSSPopper, GSSPrintConfig, PrecomputedNodeContents};
use crate::datastructures::gss::{gather_gss_stats, find_longest_path, GSSNode, GSSStats, GSSPeek, LLMTokenBV};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{Goto, NonTerminalID, ProductionID, Row, Stage7ShiftsAndReducesLookaheadValue, Table, StateID, TerminalID, SubstringGoto};
use crate::constraint::LLMVocab;

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::debug;
use crate::profiler::GSS_LOGGING_ENABLED;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::glr::table::{DefaultReduce, stage_9};
use std::collections::HashMap;
use crate::constraint::PrecomputeGraph2;
use crate::datastructures::arena::NodeId;
use crate::tokenizer::LLMTokenID;

#[derive(Debug)]
enum Action<'a> {
    Normal(&'a Stage7ShiftsAndReducesLookaheadValue),
    Default(&'a DefaultReduce),
}

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
        Some(self.cmp(other))
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
        let mut obj = StdMap::new();
        obj.insert("state_id".to_string(), self.state_id.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let state_id = obj.remove("state_id").ok_or_else(|| "Missing field state_id for ParseStateEdgeContent".to_string())
                                  .and_then(StateID::from_json)?;
                Ok(ParseStateEdgeContent { state_id })
            }
            _ => Err("Expected JSONNode::Object for ParseStateEdgeContent".to_string()),
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
    ReadyForToken,
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
    pub(crate) substring_gotos: BTreeMap<NonTerminalID, SubstringGoto>,
}

impl JSONConvertible for GLRParser {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("stage_7_table".to_string(), self.table.to_json());
        obj.insert("productions".to_string(), self.productions.to_json());
        obj.insert("terminal_map".to_string(), self.terminal_map.to_json());
        obj.insert("non_terminal_map".to_string(), self.non_terminal_map.to_json());
        obj.insert("item_set_map".to_string(), self.item_set_map.to_json());
        obj.insert("start_state_id".to_string(), self.start_state_id.to_json());
        obj.insert("everything_state_id".to_string(), self.everything_state_id.to_json());
        obj.insert("ignore_terminal_id".to_string(), self.ignore_terminal_id.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let table = obj.remove("stage_7_table").ok_or_else(|| "Missing field stage_7_table".to_string())
                                 .and_then(Table::from_json)?;
                let productions = obj.remove("productions").ok_or_else(|| "Missing field productions".to_string())
                                     .and_then(Vec::<Production>::from_json)?;
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
        let initial_parse_state = self.init_parse_state_with_acc();
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
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

    pub fn init_parse_state(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> ParseState {
        self.init_parse_state_with_acc()
    }

    pub fn init_parse_state_with_acc(&self) -> ParseState {
        let initial_content = ParseStateEdgeContent {
            state_id: self.start_state_id,
        };
        ParseState {
            stack: Arc::new(GSSNode::new_fresh().push(initial_content)),
            accepted_state: Arc::new(GSSNode::new_fresh()),
        }
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
                    if lookaheads.len() == 1 {
                        if let Some(lookahead) = lookaheads.iter().next().unwrap() {
                            write!(f, "{}", lookahead)?;
                        } else {
                            write!(f, "ε")?;
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

    let mut sorted_actions: Vec<_> = actions.iter().collect();
    sorted_actions.sort_by_key(|(tid, _)| terminal_map.get_by_right(tid).unwrap());

    for (tid, action) in sorted_actions {
        let terminal = terminal_map.get_by_right(tid).unwrap();

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
                    "Conflict"
                };

                let mut s = format!("{}:", conflict_type);
                let inner_indent = format!("\n{}        ", indent);
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
        writeln!(f, "Parse Table:")?;
        writeln!(f, "  Start State: {}", self.start_state_id.0)?;
        for (&state_id, _row) in self.table.iter().collect::<BTreeMap<_, _>>() {
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
pub struct GLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub active_state: ParseState,
    accepted: bool,
    phase: ParserPhase,
    below_bottom_cache: HashMap<BelowBottomCacheKey, (NodeId, LLMTokenBV)>,
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct WorkMapKey(usize, StateID);

impl PartialOrd for WorkMapKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkMapKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let WorkMapKey(self_depth, self_state_id) = self;
        let WorkMapKey(other_depth, other_state_id) = other;
        other_depth.cmp(&self_depth).then_with(|| self_state_id.cmp(&other_state_id))
    }
}

type WorkMap = BTreeMap<WorkMapKey, (ParseState, Option<usize>)>;

impl<'a> GLRParserState<'a> {
    fn enqueue(work_map: &mut WorkMap, state: ParseState, fuel: Option<usize>) {
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

    fn process_action_queue<F>(
        &mut self,
        work_map: &mut WorkMap,
        mut reduce_map: Option<&mut WorkMap>,
        shifted_states_todo: &mut VecDeque<ParseState>,
        accepted_states_todo: &mut VecDeque<ParseState>,
        action_selector: F,
        config: &ProcessTokenAdvancedConfig,
        fuel: &mut Option<usize>,
        trie2_arena: &mut PrecomputeGraph2,
    )
    where
        F: for<'r> Fn(&'r Row) -> Option<Action<'r>>,
    {
        assert!(fuel.is_none(), "Fuel is not supported in process_action_queue yet");
        for (_key, (_state, per_state_fuel)) in work_map.iter() {
            assert!(per_state_fuel.is_none(), "Per-state fuel is not supported in process_action_queue yet");
        }
        while let Some(entry) = work_map.pop_first() {
            let (key, (state, per_state_fuel)) = entry;
            let WorkMapKey(_depth, state_id) = key;
            let row = &self.parser.table[&state_id];
            let action_opt = action_selector(row);
            if let Some(action) = action_opt {
                for peek in GSSNode::peek_iter(&state.stack) {
                    match action {
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Shift(to)) => {
                            let new_parse_state = self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                            shifted_states_todo.push_back(new_parse_state);
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nt, len, .. }) => {
                            let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(&peek, *nt, *len, &action_selector, config, trie2_arena);
                            if !s_new_arc.is_empty() {
                                let new_parse_state = ParseState { stack: s_new_arc, accepted_state: state.accepted_state.clone() };
                                if let Some(ref mut r_map) = reduce_map { Self::enqueue(r_map, new_parse_state, per_state_fuel); } else { Self::enqueue(work_map, new_parse_state, per_state_fuel); }
                            }
                            if !accepted_s_new_arc.is_empty() {
                                self.accepted = true;
                                let accepted_parse_state = ParseState { stack: Arc::new(GSSNode::new_fresh()), accepted_state: accepted_s_new_arc };
                                accepted_states_todo.push_back(accepted_parse_state);
                            }
                        }
                        Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces }) => {
                            if let Some(to) = shift {
                                let new_parse_state = self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                                shifted_states_todo.push_back(new_parse_state);
                            }
                            for (len, nts) in reduces {
                                for (nt, _prod_ids) in nts {
                                    let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(&peek, *nt, *len, &action_selector, config, trie2_arena);
                                    if !s_new_arc.is_empty() {
                                        let new_parse_state = ParseState { stack: s_new_arc, accepted_state: state.accepted_state.clone() };
                                        if let Some(ref mut r_map) = reduce_map { Self::enqueue(r_map, new_parse_state, per_state_fuel); } else { Self::enqueue(work_map, new_parse_state, per_state_fuel); }
                                    }
                                    if !accepted_s_new_arc.is_empty() {
                                        self.accepted = true;
                                        let accepted_parse_state = ParseState { stack: Arc::new(GSSNode::new_fresh()), accepted_state: accepted_s_new_arc };
                                        accepted_states_todo.push_back(accepted_parse_state);
                                    }
                                }
                            }
                        }
                        Action::Default(default_reduce) => {
                            if default_reduce.clone_and_merge {
                                shifted_states_todo.push_back(state.clone());
                            }
                            if let Some((reduce, allowed_terminals)) = &default_reduce.reduce {
                                let mut constrained_state = state.clone();
                                if constrained_state.stack.is_alive() {
                                    let disallowed_terminals_bv = allowed_terminals.inverted();
                                    if !disallowed_terminals_bv.is_empty() {
                                        let disallowed_l2 = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::from_iter(std::iter::once((0..=usize::MAX, disallowed_terminals_bv)));
                                        crate::datastructures::gss::disallow_terminals_and_prune_arc(&mut constrained_state.stack, &disallowed_l2, &mut HashMap::new());
                                    }
                                    if !constrained_state.stack.is_empty() {
                                        for peek in GSSNode::peek_iter(&constrained_state.stack) {
                                            let (s_new_arc, accepted_s_new_arc) = self.reduce_and_goto(&peek, reduce.nonterminal_id, reduce.len, &action_selector, config, trie2_arena);
                                            if !s_new_arc.is_empty() {
                                                let new_parse_state = ParseState { stack: s_new_arc, accepted_state: state.accepted_state.clone() };
                                                if let Some(ref mut r_map) = reduce_map { Self::enqueue(r_map, new_parse_state, per_state_fuel); } else { Self::enqueue(work_map, new_parse_state, per_state_fuel); }
                                            }
                                            if !accepted_s_new_arc.is_empty() {
                                                self.accepted = true;
                                                let accepted_parse_state = ParseState { stack: Arc::new(GSSNode::new_fresh()), accepted_state: accepted_s_new_arc };
                                                accepted_states_todo.push_back(accepted_parse_state);
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

    fn _do_actions_without_default(&mut self, token_id: TerminalID, phase1_todo: &mut WorkMap, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, accepted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig, trie2_arena: &mut PrecomputeGraph2) {
        let token_display = self.parser.terminal_map.get_by_right(&token_id).unwrap();
        crate::debug!(4, "Phase 1: Processing token '{}'", token_display);
        timeit!("GLRParserState::step::phase1", {
            let tid = token_id;
            self.process_action_queue(
                phase1_todo, Some(phase2_todo), shifted_states_todo, accepted_states_todo,
                move |row| row.shifts_and_reduces_without_default_reduce.get(&tid).map(Action::Normal),
                config, &mut None, trie2_arena,
            );
        });
    }

    fn _do_actions_with_default(&mut self, token_id: TerminalID, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, accepted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig, trie2_arena: &mut PrecomputeGraph2) {
        crate::debug!(4, "Phase 1 completed, proceeding to Phase 2 with {} shifted states", shifted_states_todo.len());
        timeit!("GLRParserState::step::phase2", {
            let tid = token_id;
            self.process_action_queue(
                phase2_todo, None, shifted_states_todo, accepted_states_todo,
                move |row| row.shifts_and_reduces_full.get(&tid).map(Action::Normal),
                config, &mut None, trie2_arena,
            );
            self.phase = ParserPhase::ReadyForDefaultReductions;
        });
    }

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
                let everything = self.parser.everything_state_id;
                if let Some(goto) = self.parser.table.get(&everything).and_then(|row| row.gotos.get(&nt)) {
                    storage.accepting_sources.clear();
                    storage.gotos.clear();
                    if goto.accept { storage.accepting_sources.insert(everything); }
                    if let Some(goto_state_id) = goto.state_id { storage.gotos.insert(goto_state_id, BTreeSet::from([everything])); }
                }
                storage
            }
            BelowBottomReductionMode::Fail => storage,
            BelowBottomReductionMode::Panic => storage,
        }
    }

    fn build_below_bottom_accs(&self, popper: &GSSPopper, trie2_arena: &mut PrecomputeGraph2) -> BTreeMap<usize, Acc> {
        let mut result: BTreeMap<usize, Acc> = BTreeMap::new();
        for (k, accs_by_edge) in popper.below_bottom() {
            for (last_edge_content, acc_arc) in accs_by_edge {
                let acc = acc_arc.as_ref();
                let edge_key = (0, Some(last_edge_content.state_id));
                let edge_bv = LLMTokenBV::max_ones();

                let mut used_dest_ids = BTreeSet::new();
                for &source_id in &acc.trie2_nodes {
                    let dest_id = trie2_arena.create_node(PrecomputedNodeContents::internal());
                    trie2_arena.force_insert(source_id, edge_key, edge_bv.clone(), dest_id);
                    used_dest_ids.insert(dest_id);
                }

                let mut new_acc = acc.clone();
                new_acc.trie2_nodes = used_dest_ids;
                result.entry(*k).and_modify(|existing| *existing = Acc::merge(existing, &new_acc)).or_insert(new_acc);
            }
        }
        result
    }

    fn handle_below_bottom_accepts(
        &mut self,
        nt: NonTerminalID,
        below: &BTreeMap<usize, Acc>,
        gotos: &SubstringGoto,
        trie2_arena: &mut PrecomputeGraph2,
    ) -> Option<Arc<GSSNode>> {
        if gotos.accepting_sources.is_empty() { return None; }
        let mut accepted_stacks: Vec<Arc<GSSNode>> = Vec::new();
        for (k, acc) in below {
            let mut acc = acc.clone();
            let trie2_nodes = std::mem::take(&mut acc.trie2_nodes);
            for source_state_id in &gotos.accepting_sources {
                let accept_cache_key = BelowBottomCacheKey { nonterminal_id: nt, source_state_id: StateID(usize::MAX), goto_state_id: StateID(usize::MAX), k: 0 };
                let (dst_id, _is_new) = self.below_bottom_cache.entry(accept_cache_key).or_insert_with(|| {
                    let new_id = trie2_arena.create_node(PrecomputedNodeContents::internal());
                    (new_id, LLMTokenBV::max_ones())
                });
                let edge_bv = LLMTokenBV::max_ones();
                let edge_key = (*k, Some(*source_state_id));
                for &source_id in &trie2_nodes {
                    trie2_arena.force_insert(source_id, edge_key, edge_bv.clone(), *dst_id);
                }
                let mut acc_for_gss = acc.clone();
                acc_for_gss.trie2_nodes.insert(*dst_id);
                let gss0 = GSSNode::new(acc_for_gss);
                let gss1 = gss0.push(ParseStateEdgeContent { state_id: *source_state_id });
                accepted_stacks.push(Arc::new(gss1));
            }
        }
        if accepted_stacks.is_empty() { None } else { Some(GSSNode::merge_many_with_depth(usize::MAX, accepted_stacks)) }
    }

    fn handle_below_bottom_gotos(
        &mut self,
        nt: NonTerminalID,
        below: BTreeMap<usize, Acc>,
        gotos: &SubstringGoto,
        trie2_arena: &mut PrecomputeGraph2,
    ) -> Arc<GSSNode> {
        if gotos.gotos.is_empty() { return Arc::new(GSSNode::new_fresh()); }
        let cache_key = BelowBottomCacheKey { nonterminal_id: nt, source_state_id: StateID(0), goto_state_id: StateID(0), k: 0 };
        let mut merged_acc = {
            let mut below_it = below.iter();
            let first = below_it.next().unwrap().1.clone();
            below_it.fold(first, |acc, (_k, acc2)| Acc::merge(&acc, acc2))
        };
        merged_acc.trie2_nodes.clear();

        if let Some((node_id, llm_tokens)) = self.below_bottom_cache.get_mut(&cache_key) {
            for (k, acc) in below {
                let trie2_nodes = &acc.trie2_nodes;
                let edge_key = (k, None);
                let edge_bv = LLMTokenBV::max_ones();
                for &source_id in trie2_nodes {
                    trie2_arena.force_insert(source_id, edge_key, edge_bv.clone(), *node_id);
                }
            }
            if !merged_acc.llm_tokens_union.is_subset(llm_tokens) {
                *llm_tokens |= &merged_acc.llm_tokens_union;
                let mut out = Vec::new();
                for (goto_state_id, source_state_ids) in &gotos.gotos {
                    let edge_contents = source_state_ids.iter().map(|sid| ParseStateEdgeContent { state_id: *sid }).collect();
                    let gss0 = GSSNode::new(merged_acc.clone());
                    let gss1 = gss0.push_many(edge_contents);
                    let gss2 = gss1.push(ParseStateEdgeContent { state_id: *goto_state_id });
                    out.push(Arc::new(gss2));
                }
                GSSNode::merge_many_with_depth(usize::MAX, out)
            } else { Arc::new(GSSNode::new_fresh()) }
        } else {
            let new_trie2_node_id = trie2_arena.create_node(PrecomputedNodeContents::internal());
            self.below_bottom_cache.insert(cache_key, (new_trie2_node_id, LLMTokenBV::max_ones()));
            let mut out = Vec::new();
            for (k, acc) in below {
                let trie2_nodes = &acc.trie2_nodes;
                let edge_key = (k, None);
                let edge_bv = LLMTokenBV::max_ones();
                for &source_id in trie2_nodes {
                    trie2_arena.force_insert(source_id, edge_key, edge_bv.clone(), new_trie2_node_id);
                }
            }
            merged_acc.trie2_nodes.insert(new_trie2_node_id);
            for (goto_state_id, source_state_ids) in &gotos.gotos {
                let edge_contents = source_state_ids.iter().map(|sid| ParseStateEdgeContent { state_id: *sid }).collect();
                let gss0 = GSSNode::new(merged_acc.clone());
                let gss1 = gss0.push_many(edge_contents);
                let gss2 = gss1.push(ParseStateEdgeContent { state_id: *goto_state_id });
                out.push(Arc::new(gss2));
            }
            GSSNode::merge_many_with_depth(usize::MAX, out)
        }
    }

    #[time_it("GLRParserState::reduce_and_goto")]
    fn reduce_and_goto<G>(
        &mut self,
        peek: &GSSPeek,
        nt: NonTerminalID,
        len: usize,
        action_selector: &G,
        config: &ProcessTokenAdvancedConfig,
        trie2_arena: &mut PrecomputeGraph2,
    ) -> (Arc<GSSNode>, Arc<GSSNode>)
    where
        G: for<'r> Fn(&'r Row) -> Option<Action<'r>>,
    {
        let popper: GSSPopper = timeit!(peek.popn(len));
        let mut out: Vec<Arc<GSSNode>> = Vec::new();
        let mut accepted_out: Vec<Arc<GSSNode>> = Vec::new();

        for popper_item in popper.iter() {
            for peek2 in popper_item.peek_iter() {
                let predecessor_state_id = peek2.edge_value().state_id;
                let mut current_nt = nt;
                loop {
                    let goto = self.parser.table.get(&predecessor_state_id).and_then(|row| row.gotos.get(&current_nt)).expect_else(|| format!("Goto not found for NT '{}' in state {:?}", self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id));
                    if goto.accept { accepted_out.push(peek2.isolated_parent()); }
                    if let Some(goto_state_id) = goto.state_id {
                        let next_row = &self.parser.table[&goto_state_id];
                        match action_selector(next_row) {
                            Some(Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: next_nt, len: 1, .. })) => { current_nt = *next_nt; continue; }
                            Some(Action::Default(def)) => {
                                if def.clone_and_merge || def.reduce.as_ref().map_or(false, |r| r.0.len != 1) { out.push(Arc::new(peek2.push_on_parent(ParseStateEdgeContent { state_id: goto_state_id }))); }
                                if let Some(reduce) = &def.reduce { if reduce.0.len == 1 { current_nt = reduce.0.nonterminal_id; continue; } }
                                break;
                            }
                            _ => { out.push(Arc::new(peek2.push_on_parent(ParseStateEdgeContent { state_id: goto_state_id }))); break; }
                        }
                    } else { break; }
                }
            }
        }

        if !popper.below_bottom().is_empty() {
            match config.below_bottom_mode {
                BelowBottomReductionMode::Fail => {}
                BelowBottomReductionMode::Panic => panic!("A reduction popped below the bottom of the stack, and BelowBottomReductionMode was set to Panic."),
                _ => {
                    let below_accs = self.build_below_bottom_accs(&popper, trie2_arena);
                    let mut storage = SubstringGoto::default();
                    let gotos_for_nt = self.substring_gotos_for(nt, config, &mut storage);
                    if let Some(accepted_merged) = self.handle_below_bottom_accepts(nt, &below_accs, gotos_for_nt, trie2_arena) { accepted_out.push(accepted_merged); }
                    let merged_below = self.handle_below_bottom_gotos(nt, below_accs, gotos_for_nt, trie2_arena);
                    out.push(merged_below);
                }
            }
        }

        let new_active = GSSNode::merge_many_with_depth(usize::MAX, out);
        let new_accepted = GSSNode::merge_many_with_depth(usize::MAX, accepted_out);
        (new_active, new_accepted)
    }

    #[time_it("GLRParserState::process_token_advanced")]
    pub fn process_token_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig, trie2_arena: &mut PrecomputeGraph2) {
        self.accepted = false;
        self.active_state.accepted_state = Arc::new(GSSNode::new_fresh());
        self.below_bottom_cache.clear();

        if Some(token_id) == self.parser.ignore_terminal_id {
            self.phase = ParserPhase::ReadyForDefaultReductions;
            return;
        }

        self.log_gss("Phase1/2-start", token_id, false, false);
        let mut phase2_todo: WorkMap = WorkMap::new();
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let mut accepted_states_todo: VecDeque<ParseState> = VecDeque::new();

        if self.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: WorkMap = WorkMap::new();
            Self::enqueue(&mut phase1_todo, self.active_state.clone(), None);
            self._do_actions_without_default(token_id, &mut phase1_todo, &mut phase2_todo, &mut shifted_states_todo, &mut accepted_states_todo, config, trie2_arena);
        } else {
            Self::enqueue(&mut phase2_todo, self.active_state.clone(), None);
        }

        self._do_actions_with_default(token_id, &mut phase2_todo, &mut shifted_states_todo, &mut accepted_states_todo, config, trie2_arena);

        let mut next_active = ParseState::new();
        for state in shifted_states_todo { next_active.merge(state); }
        for state in accepted_states_todo { next_active.merge(state); }
        self.active_state = next_active;
        self.log_gss("Phase1/2-end", token_id, false, false);
        self.below_bottom_cache.clear();
    }

    pub fn process_default_reductions(&mut self) {
        // This method is now problematic as it can't get the trie2_arena.
        // For now, we assume it's not needed in a context where below-bottom reductions happen.
        // A proper fix would be to require the arena here too.
        // Let's create a dummy arena, which will panic if used.
        let mut dummy_arena = PrecomputeGraph2::new(PrecomputedNodeContents::internal());
        self.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig::default(), &mut dummy_arena);
    }

    #[time_it("GLRParserState::process_default_reductions_advanced")]
    pub fn process_default_reductions_advanced(&mut self, config: &ProcessDefaultReductionsAdvancedConfig, trie2_arena: &mut PrecomputeGraph2) {
        self.log_gss("Phase3-start", TerminalID(0), false, false);
        if self.phase == ParserPhase::ReadyForToken {
            return;
        }
        assert_eq!(self.phase, ParserPhase::ReadyForDefaultReductions);

        let mut work_map: WorkMap = WorkMap::new();
        Self::enqueue(&mut work_map, self.active_state.clone(), config.per_state_fuel);
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let mut accepted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let mut fuel = config.fuel;
        let token_config = ProcessTokenAdvancedConfig { below_bottom_mode: config.below_bottom_mode };

        self.process_action_queue(
            &mut work_map, None, &mut shifted_states_todo, &mut accepted_states_todo,
            |row| Some(Action::Default(&row.default_reduce)),
            &token_config, &mut fuel, trie2_arena,
        );

        let mut next_active = ParseState::new();
        for state in shifted_states_todo { next_active.merge(state); }
        for state in accepted_states_todo { next_active.merge(state); }
        for (_, (state, _fuel)) in work_map { next_active.merge(state); }
        self.active_state = next_active;
        self.phase = ParserPhase::ReadyForToken;
        self.log_gss("Phase3-end", TerminalID(0), false, false);
    }

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
                        ParserPhase::ReadyForToken => row.shifts_and_reduces_without_default_reduce.get(&token_id).map(Action::Normal),
                        ParserPhase::ReadyForDefaultReductions => row.shifts_and_reduces_full.get(&token_id).map(Action::Normal).or_else(|| Some(Action::Default(&row.default_reduce))),
                    };
                    if let Some(action) = action_opt {
                        timeit!("GLRParserState::has_action_for::action_found::add_llm_tokens", {
                            let peek_llm_tokens = timeit!(peek.resolved_llm_tokens_union());
                            timeit!(llm_tokens |= peek_llm_tokens);
                        });
                    }
                }
                Some(llm_tokens)
            }
            LRMode::LALR => None,
        }
    }

    pub fn step(&mut self, token_id: TerminalID) {
        self.process_token_advanced(token_id, &ProcessTokenAdvancedConfig::default(), &mut PrecomputeGraph2::new(PrecomputedNodeContents::internal()));
    }

    pub fn parse(&mut self, input: &[TerminalID]) {
        self.parse_part(input, &mut PrecomputeGraph2::new(PrecomputedNodeContents::internal()));
    }

    pub fn parse_part(&mut self, input: &[TerminalID], trie2_arena: &mut PrecomputeGraph2) {
        for &terminal_id in input {
            self.step(terminal_id);
        }
    }

    pub fn parse_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        self.parse_part_advanced(input, config);
    }

    pub fn parse_part_advanced(&mut self, input: &[TerminalID], config: &ProcessTokenAdvancedConfig) {
        for &terminal_id in input {
            self.process_token_advanced(terminal_id, config, &mut PrecomputeGraph2::new(PrecomputedNodeContents::internal()));
        }
    }

    pub fn process_token(&mut self, token_id: TerminalID) {
        self.process_token_advanced(token_id, &ProcessTokenAdvancedConfig::default(), &mut PrecomputeGraph2::new(PrecomputedNodeContents::internal()));
    }

    pub fn merge_with(&mut self, other: GLRParserState) {
        assert!(std::ptr::eq(self.parser, other.parser));
        // The caller is responsible for ensuring phases are compatible or running default reductions.
        self.active_state.merge(other.active_state);
        self.accepted |= other.accepted;
    }

    pub fn is_ok(&self) -> bool {
        self.accepted || (!self.active_state.stack.is_empty() && self.active_state.stack.is_alive())
    }

    pub fn has_accepted(&self) -> bool {
        self.accepted
    }

    pub fn log_gss(&self, phase: &str, token: TerminalID, explain_states: bool, generate_dot: bool) {
        if !GSS_LOGGING_ENABLED { return; }
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
                let config = GSSPrintConfig { max_edges: max_edges_to_print, ..Default::default() };
                let (gss_string, state_ids) = print_gss_forest(&[root.clone()], &self.parser.terminal_map, &config);
                let final_string = if print_full_forest {
                    format!("{} GSS ({} nodes, {} edges):\n{}", name, stats.unique_nodes, stats.total_edges, gss_string)
                } else {
                    match find_longest_path(root) {
                        Some(p) => format!("{} GSS too big ({} nodes, {} edges). Longest path ({}): {}", name, stats.unique_nodes, stats.total_edges, p.len(), p.iter().map(|(ec, _n)| ec.state_id.0).map(|id| id.to_string()).collect::<Vec<_>>().join(" → ")),
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
            crate::debug!(1, "GSS DOT graph:\n{}", dot_string);
        }
    }

    pub fn gss_to_dot(&self) -> String {
        let mut roots: Vec<(&str, &GSSNode)> = vec![("Active", &self.active_state.stack)];
        if !self.active_state.accepted_state.is_empty() {
            roots.push(("Accepted", &self.active_state.accepted_state));
        }
        self.parser.gss_forest_to_dot(&roots, None, None)
    }
}

impl GLRParser {
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

        for (i, (label, root)) in roots.iter().enumerate() {
            let root_ptr = *root as *const GSSNode;
            let root_id = *node_ids.entry(root_ptr).or_insert_with(|| { let id = next_id_counter; next_id_counter += 1; id });
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

        while let Some(node_arc) = queue.pop_front() {
            let node_ptr = Arc::as_ptr(&node_arc);
            if visited_nodes.contains(&node_ptr) { continue; }
            let parent_id = *node_ids.entry(node_ptr).or_insert_with(|| { let id = next_id_counter; next_id_counter += 1; id });

            if visited_nodes.insert(node_ptr) {
                let acc_str = crate::datastructures::gss::format_acc(&node_arc, &self.terminal_map, original_internal_bimap, llm_token_map);
                let escaped_acc = acc_str.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\l").replace('{', "\\{").replace('}', "\\}").replace('<', "\\<").replace('>', "\\>").replace('\'', "\\'");
                writeln!(&mut dot, "  N{} [label=\"Node {}\\lDepth: {}\\l{}\"];", parent_id, parent_id, node_arc.max_depth(), escaped_acc).unwrap();
            }

            for (edge_val, preds_by_depth) in node_arc.predecessors() {
                let state_id = edge_val.state_id;
                let edge_key = (node_ptr, edge_val.clone());
                let edge_node_id = *edge_node_ids.entry(edge_key).or_insert_with(|| {
                    let id = next_id_counter;
                    next_id_counter += 1;
                    let mut explanation = String::new();
                    self.format_state_details(&mut explanation, state_id, "").unwrap();
                    let escaped_explanation = explanation.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\l").replace('{', "\\{").replace('}', "\\}").replace('<', "\\<").replace('>', "\\>").replace('\'', "\\'");
                    writeln!(&mut dot, "  E{} [label=\"State {}\\l{}\", shape=plaintext, fontname=\"Courier New\"];", id, state_id.0, escaped_explanation).unwrap();
                    id
                });
                writeln!(&mut dot, "  N{} -> E{};", parent_id, edge_node_id).unwrap();
                for pred_vec in preds_by_depth.values() {
                    for pred_arc in pred_vec {
                        let pred_ptr = Arc::as_ptr(pred_arc);
                        let pred_id = *node_ids.entry(pred_ptr).or_insert_with(|| { let id = next_id_counter; next_id_counter += 1; id });
                        writeln!(&mut dot, "  E{} -> N{} [arrowhead=none];", edge_node_id, pred_id).unwrap();
                        queue.push_back(pred_arc.clone());
                    }
                }
            }
        }

        writeln!(&mut dot, "}}").unwrap();
        dot
    }

    pub fn gss_to_dot(&self, root: &GSSNode, original_internal_bimap: Option<&BiBTreeMap<usize, usize>>, llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>) -> String {
        self.gss_forest_to_dot(&[("Root", root)], original_internal_bimap, llm_token_map)
    }
}