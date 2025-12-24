use crate::constraint::LLMVocab;
use crate::datastructures::gss_acc::Acc;
use crate::datastructures::leveled_gss::{LeveledGSS, LeveledGSSStats};
use crate::glr::grammar::{NonTerminal, Production};
use crate::glr::items::Item;
use crate::glr::table::{get_row, NonTerminalID, StateID, Table, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::GSS_LOGGING_ENABLED;
use crate::debug;
use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
use profiler_macro::time_it;
use std::any::Any;
use std::cmp::Ordering;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;

/// A trait to provide a lazily-evaluated `expect`.
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct ParseStateEdgeContent {
    pub state_id: StateID,
}

pub type ParserGSS = LeveledGSS<ParseStateEdgeContent, Acc>;
pub type GSSStats = LeveledGSSStats<ParseStateEdgeContent, Acc>;

#[derive(Clone)]
pub struct ParseState {
    pub stack: ParserGSS,
}

impl Debug for ParseState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParseState").finish()
    }
}

impl PartialEq for ParseState {
    fn eq(&self, other: &Self) -> bool {
        self.stack.to_stacks() == other.stack.to_stacks()
    }
}

impl Eq for ParseState {}

impl ParseState {
    pub fn new() -> Self {
        ParseState {
            stack: LeveledGSS::empty(),
        }
    }

    pub(crate) fn with_stack(stack: ParserGSS) -> Self {
        ParseState { stack }
    }
}

#[derive(Clone)]
pub struct GLRParser {
    pub table: Table,
    pub productions: Vec<Production>,
    pub terminal_map: BiBTreeMap<crate::glr::grammar::Terminal, TerminalID>,
    pub non_terminal_map: BiBTreeMap<NonTerminal, crate::glr::table::NonTerminalID>,
    pub item_set_map: BiBTreeMap<Vec<Item>, StateID>,
    pub start_state_id: StateID,
    pub substring_state_id: StateID,
    /// Set of terminal IDs to ignore (skip without consuming).
    /// These are typically whitespace-like terminals that are always optional.
    pub ignore_terminal_ids: HashSet<TerminalID>,
    pub actions: BTreeMap<crate::glr::table::NonTerminalID, ActionFn>,
}

/// Intermediate type for GLRParser JSON serialization
#[derive(JSONConvertible)]
struct GLRParserJSON {
    stage_7_table: Table,
    productions: Vec<Production>,
    terminal_map: BiBTreeMap<crate::glr::grammar::Terminal, TerminalID>,
    non_terminal_map: BiBTreeMap<NonTerminal, crate::glr::table::NonTerminalID>,
    item_set_map: BiBTreeMap<Vec<Item>, StateID>,
    start_state_id: StateID,
    substring_state_id: StateID,
    ignore_terminal_ids: HashSet<TerminalID>,
}

impl GLRParserJSON {
    fn from_parser(p: &GLRParser) -> Self {
        GLRParserJSON {
            stage_7_table: p.table.clone(),
            productions: p.productions.clone(),
            terminal_map: p.terminal_map.clone(),
            non_terminal_map: p.non_terminal_map.clone(),
            item_set_map: p.item_set_map.clone(),
            start_state_id: p.start_state_id,
            substring_state_id: p.substring_state_id,
            ignore_terminal_ids: p.ignore_terminal_ids.clone(),
        }
    }

    fn to_parser(self) -> GLRParser {
        GLRParser::new(
            self.stage_7_table,
            self.productions,
            self.terminal_map,
            self.non_terminal_map,
            self.item_set_map,
            self.start_state_id,
            self.substring_state_id,
            BTreeMap::new(), // actions provided at runtime
            self.ignore_terminal_ids,
        )
    }
}

impl JSONConvertible for GLRParser {
    fn to_json(&self) -> JSONNode {
        GLRParserJSON::from_parser(self).to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        GLRParserJSON::from_json(node).map(|p| p.to_parser())
    }
}

impl Display for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
        
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f, "  GLR PARSER")?;
        writeln!(f, "  {} states  •  {} productions  •  {} terminals  •  {} non-terminals", 
            self.table.len(), self.productions.len(), self.terminal_map.len(), self.non_terminal_map.len())?;
        writeln!(f, "  Start: state {}", self.start_state_id.0)?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f)?;
        
        // Print Grammar Productions
        writeln!(f, "GRAMMAR:")?;
        for (i, prod) in self.productions.iter().enumerate() {
            writeln!(f, "  {:2}. {}", i, prod)?;
        }
        writeln!(f)?;
        
        // Helper to get terminal name (clean, without Debug wrapper)
        let get_terminal_name = |tid: &TerminalID| -> String {
            self.terminal_map
                .get_by_right(tid)
                .map(|t| format!("{}", t))
                .unwrap_or_else(|| format!("T{}", tid.0))
        };
        
        // Helper to get non-terminal name (clean, without Debug wrapper)
        let get_nonterminal_name = |ntid: &NonTerminalID| -> String {
            self.non_terminal_map
                .get_by_right(ntid)
                .map(|nt| format!("{}", nt))
                .unwrap_or_else(|| format!("NT{}", ntid.0))
        };
        
        // Helper to format action
        let format_action = |action: &Stage7ShiftsAndReducesLookaheadValue| -> String {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(sid) => {
                    format!("shift {}", sid.0)
                }
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                    let prod_info = if production_ids.len() == 1 {
                        format!(" [#{}]", production_ids[0].0)
                    } else if production_ids.len() > 1 {
                        format!(" [{} prods]", production_ids.len())
                    } else {
                        String::new()
                    };
                    format!(
                        "reduce {} (pop {}){}",
                        get_nonterminal_name(nonterminal_id),
                        len,
                        prod_info
                    )
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    let mut parts = Vec::new();
                    if let Some(sid) = shift {
                        parts.push(format!("shift {}", sid.0));
                    }
                    for (len, nts) in reduces {
                        for (ntid, pids) in nts {
                            let prod_info = if pids.len() == 1 {
                                format!(" [#{}]", pids[0].0)
                            } else if pids.len() > 1 {
                                format!(" [{} prods]", pids.len())
                            } else {
                                String::new()
                            };
                            parts.push(format!(
                                "reduce {} (pop {}){}",
                                get_nonterminal_name(ntid),
                                len,
                                prod_info
                            ));
                        }
                    }
                    parts.join(" / ")
                }
            }
        };
        
        // Print Parse Table
        writeln!(f, "PARSE TABLE:")?;
        for (state_id, row) in &self.table {
            writeln!(f)?;
            writeln!(f, "  State {}:", state_id.0)?;
            
            // Collect all actions for better formatting
            let mut actions = Vec::new();
            
            // Collect shifts and reduces
            let shifts_map = row.get_shifts_and_reduces_map();
            for (tid, action) in &shifts_map {
                actions.push((get_terminal_name(tid), format_action(action), false));
            }
            
            // Add default reduce if present
            if let Some(default) = &row.default_reduce {
                actions.push(("<default>".to_string(), format_action(default), false));
            }
            
            // Collect gotos
            let gotos = row.get_gotos();
            for (ntid, goto) in gotos {
                if let Some(sid) = goto.state_id {
                    let action_str = if goto.accept {
                        format!("goto {} (ACCEPT)", sid.0)
                    } else {
                        format!("goto {}", sid.0)
                    };
                    actions.push((get_nonterminal_name(ntid), action_str, true));
                } else if goto.accept {
                    actions.push((get_nonterminal_name(ntid), "ACCEPT".to_string(), true));
                }
            }
            
            // Find max symbol width for alignment
            let max_symbol_width = actions.iter()
                .map(|(sym, _, _)| sym.len())
                .max()
                .unwrap_or(0);
            
            // Print all actions with alignment
            for (symbol, action, is_goto) in actions {
                writeln!(f, "    {:width$}  →  {}", symbol, action, width = max_symbol_width)?;
            }
        }
        
        Ok(())
    }
}

impl Debug for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLRParser")
            .field("table_len", &self.table.len())
            .field("productions_len", &self.productions.len())
            .field("start_state_id", &self.start_state_id)
            .field("substring_state_id", &self.substring_state_id)
            .field("ignore_terminal_ids", &self.ignore_terminal_ids)
            .field("actions_size", &self.actions.len())
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
            && self.substring_state_id == other.substring_state_id
            && self.ignore_terminal_ids == other.ignore_terminal_ids
    }
}
impl Eq for GLRParser {}

impl GLRParser {
    pub fn new(
        table: Table,
        productions: Vec<Production>,
        terminal_map: BiBTreeMap<crate::glr::grammar::Terminal, TerminalID>,
        non_terminal_map: BiBTreeMap<NonTerminal, crate::glr::table::NonTerminalID>,
        item_set_map: BiBTreeMap<Vec<Item>, StateID>,
        start_state_id: StateID,
        substring_state_id: StateID,
        actions: BTreeMap<NonTerminal, ActionFn>,
        ignore_terminal_ids: HashSet<TerminalID>,
    ) -> Self {
        let converted_actions: BTreeMap<crate::glr::table::NonTerminalID, ActionFn> = actions
            .into_iter()
            .map(|(nt, func)| {
                let nt_id = non_terminal_map.get_by_left(&nt).unwrap_or_else(|| {
                    panic!(
                        "NonTerminal {:?} not found in non_terminal_map during GLRParser construction",
                        nt
                    )
                });
                (*nt_id, func)
            })
            .collect();

        GLRParser {
            table,
            productions,
            terminal_map,
            non_terminal_map,
            item_set_map,
            start_state_id,
            substring_state_id,
            ignore_terminal_ids,
            actions: converted_actions,
        }
    }

    pub fn init_glr_parser(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        self.init_glr_parser_with_acc()
    }

    pub fn init_glr_parser_with_acc(&self) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_with_acc();
        GLRParserState {
            parser: self,
            stack: initial_parse_state.stack,
        }
    }

    pub fn init_parse_state_with_acc(&self) -> ParseState {
        let initial_edge = ParseStateEdgeContent {
            state_id: self.start_state_id,
        };
        let acc = Acc::new_fresh();
        let gss = LeveledGSS::from_stacks(&[(vec![], acc)]).push(initial_edge);
        ParseState::with_stack(gss)
    }

    pub fn init_parse_state_with_gss(&self, gss: ParserGSS) -> GLRParserState {
        GLRParserState { parser: self, stack: gss }
    }

    pub fn init_glr_parser_null(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        self.init_glr_parser_with_acc()
    }

    pub fn parse(&self, input: &[TerminalID], original_llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let mut state = self.init_glr_parser(original_llm_vocab);
        state.parse(input);
        state
    }

    #[time_it]
    pub fn process_token_gss(&self, gss: &ParserGSS, token: TerminalID) -> ParserGSS {
        // Skip tokens that are in the ignore set
        if self.ignore_terminal_ids.contains(&token) {
            return gss.clone();
        }
        if gss.is_empty() {
            return gss.clone();
        }

        let mut heads_by_state: BTreeMap<StateID, ParserGSS> = BTreeMap::new();
        for edge in gss.peek() {
            let sid = edge.state_id;
            let iso = gss.isolate(Some(edge));
            heads_by_state
                .entry(sid)
                .and_modify(|acc| *acc = acc.merge(&iso))
                .or_insert(iso);
        }

        let mut shifted: Vec<ParserGSS> = Vec::new();

        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 1000;
        while let Some((state_id, state_gss)) = heads_by_state.pop_first() {
            iteration += 1;
            if iteration > MAX_ITERATIONS {
                eprintln!("DEBUG: process_token_gss hit {} iterations, breaking!", MAX_ITERATIONS);
                break;
            }
            if iteration <= 20 || iteration % 100 == 0 {
                eprintln!("DEBUG: iter={}, state_id={:?}, heads_by_state.len()={}", iteration, state_id, heads_by_state.len());
            }
            if let Some(row) = get_row(&self.table, state_id) {
                row.handle_shifts_and_reduces_for_terminal(
                    token,
                    |to| shifted.push(state_gss.push(ParseStateEdgeContent { state_id: *to })),
                    |nt_id, len, _pids| {
                        eprintln!("DEBUG: reduce nt_id={:?}, len={}", nt_id, len);
                        self.apply_reduces(&state_gss, *len, *nt_id, &mut heads_by_state);
                    },
                );
            }
        }

        if shifted.is_empty() {
            return LeveledGSS::empty();
        }
        let mut it = shifted.into_iter();
        let first = it.next().unwrap();
        it.fold(first, |acc, g| acc.merge(&g))
    }

    fn apply_reduces(
        &self,
        state_gss: &ParserGSS,
        len: usize,
        nt: NonTerminalID,
        heads_by_state: &mut BTreeMap<StateID, ParserGSS>,
    ) {
        let popped = state_gss.popn(len as isize);
        if popped.is_empty() {
            return;
        }

        for edge in popped.peek() {
            let from_id = edge.state_id;
            if let Some(row) = get_row(&self.table, from_id) {
                if let Some(goto) = row.gotos.get(&nt) {
                    if let Some(next_id) = goto.state_id {
                        let iso = popped.isolate(Some(edge));
                        let pushed = iso.push(ParseStateEdgeContent { state_id: next_id });
                        heads_by_state
                            .entry(next_id)
                            .and_modify(|acc| *acc = acc.merge(&pushed))
                            .or_insert(pushed);
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct GLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub stack: ParserGSS,
}

impl Debug for GLRParserState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLRParserState")
            .field("parser", &self.parser)
            .finish()
    }
}

impl PartialEq for GLRParserState<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.stack.to_stacks() == other.stack.to_stacks()
    }
}

impl Eq for GLRParserState<'_> {}

impl<'a> GLRParserState<'a> {
    pub fn step(&mut self, token_id: TerminalID) {
        self.stack = self.parser.process_token_gss(&self.stack, token_id);
    }

    pub fn parse(&mut self, input: &[TerminalID]) {
        for &t in input {
            self.step(t);
        }
    }

    pub fn is_ok(&self) -> bool {
        !self.stack.is_empty()
    }

    pub fn stats(&self) -> GSSStats {
        self.stack.stats()
    }

    pub fn log_gss(&self, phase: &str, token: TerminalID) {
        if !GSS_LOGGING_ENABLED {
            return;
        }
        let stats = self.stats();
        debug!(2, "{} - token {} - GSS stats: {:?}", phase, token.0, stats);
        if let Some((path, _acc)) = self.stack.get_first_path() {
            let ids: Vec<_> = path.into_iter().map(|e| e.state_id.0).collect();
            debug!(5, "Sample path: {:?}", ids);
        }
    }

    pub fn merge_with(&mut self, other: Self) {
        self.stack = self.stack.merge(&other.stack);
    }
}
