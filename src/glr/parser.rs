use crate::constraint::{LLMVocab, StateIDBV};
use crate::datastructures::leveled_gss::{LeveledGSS, LeveledGSSStats};
use crate::datastructures::hybrid_bitset::HybridBitset as TerminalBV;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::{Item};
use crate::glr::table::{
    Goto, NonTerminalID, Row, Stage7ShiftsAndReducesLookaheadValue, StateID, SubstringGoto, Table,
    TerminalID,
};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::GSS_LOGGING_ENABLED;
use crate::{debug, hit};
use bimap::BiBTreeMap;
use profiler_macro::time_it;
use std::any::Any;
use std::cmp::Ordering;
use std::collections::BTreeMap as StdMap;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::datastructures::gss_acc::Acc;

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

#[derive(Debug, Copy, Clone)]
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
        let mut obj = StdMap::new();
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

pub type ParserGSS = LeveledGSS<ParseStateEdgeContent, Acc>;
pub type GSSStats = LeveledGSSStats<ParseStateEdgeContent, Acc>;

#[derive(Clone)]
pub struct ParseState {
    pub stack: ParserGSS,
}

impl Debug for ParseState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParseState")
            // .field("stack", &self.stack)
            .finish()
    }
}

impl PartialEq for ParseState {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl Eq for ParseState {}

impl ParseState {
    pub fn new() -> Self {
        ParseState { stack: LeveledGSS::empty() }
    }

    pub(crate) fn with_stack(stack: ParserGSS) -> Self {
        ParseState { stack }
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
    pub substring_gotos: BTreeMap<NonTerminalID, SubstringGoto>,
    pub reduce_goto_map: BTreeMap<NonTerminalID, BTreeMap<StateID, StateIDBV>>,
    pub actions: BTreeMap<NonTerminalID, ActionFn>,
}

impl Display for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "GLRParser:")?;
        eprintln!("TODO:");
        return Ok(());
    }
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
        // substring_gotos, reduce_goto_map, actions are recomputed or provided at runtime.
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
                let _start_production_id =
                    obj.remove("start_production_id").and_then(|n| usize::from_json(n).ok());
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

                let substring_gotos = crate::glr::table::stage_9(&table, &non_terminal_map);
                let reduce_goto_map = crate::glr::table::stage_10(&table);
                Ok(GLRParser::new(
                    table,
                    productions,
                    terminal_map,
                    non_terminal_map,
                    item_set_map,
                    start_state_id,
                    everything_state_id,
                    BTreeMap::new(),
                    ignore_terminal_id,
                    substring_gotos,
                    reduce_goto_map,
                ))
            }
            _ => Err("Expected JSONNode::Object for GLRParser".to_string()),
        }
    }
}

impl Debug for GLRParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLRParser")
            .field("table_len", &self.table.len())
            .field("productions_len", &self.productions.len())
            .field("start_state_id", &self.start_state_id)
            .field("everything_state_id", &self.everything_state_id)
            .field("ignore_terminal_id", &self.ignore_terminal_id)
            .field("substring_gotos_size", &self.substring_gotos.len())
            .field("reduce_goto_map_size", &self.reduce_goto_map.len())
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
            && self.everything_state_id == other.everything_state_id
            && self.ignore_terminal_id == other.ignore_terminal_id
            && self.substring_gotos == other.substring_gotos
            && self.reduce_goto_map == other.reduce_goto_map
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
        actions: BTreeMap<NonTerminal, ActionFn>,
        ignore_terminal_id: Option<TerminalID>,
        substring_gotos: BTreeMap<NonTerminalID, SubstringGoto>,
        reduce_goto_map: BTreeMap<NonTerminalID, BTreeMap<StateID, StateIDBV>>,
    ) -> Self {
        let converted_actions: BTreeMap<NonTerminalID, ActionFn> = actions
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
            everything_state_id,
            ignore_terminal_id,
            substring_gotos,
            reduce_goto_map,
            actions: converted_actions,
        }
    }

    pub fn init_glr_parser(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        self.init_glr_parser_with_acc()
    }

    pub fn init_glr_parser_with_acc(&self) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_with_acc();
        GLRParserState { parser: self, stack: initial_parse_state.stack }
    }

    pub fn init_parse_state_with_acc(&self) -> ParseState {
        let initial_edge = ParseStateEdgeContent { state_id: self.start_state_id };
        let acc = Acc::new_fresh();
        let gss = LeveledGSS::from_stacks(&[(vec![], acc)]).push(initial_edge);
        ParseState::with_stack(gss)
    }

    pub fn init_parse_state_with_gss(&self, gss: LeveledGSS<ParseStateEdgeContent, Acc>) -> GLRParserState {
        GLRParserState { parser: self, stack: gss }
    }

    pub fn init_glr_parser_null(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_with_acc();
        GLRParserState { parser: self, stack: initial_parse_state.stack }
    }

    pub fn parse(&self, input: &[TerminalID], llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let mut state = self.init_glr_parser(llm_vocab);
        state.parse(input);
        state
    }

    #[time_it]
    pub fn process_token_gss(&self, gss: &ParserGSS, token: TerminalID) -> ParserGSS {
        if Some(token) == self.ignore_terminal_id {
            return gss.clone();
        }
        if gss.is_empty() {
            return gss.clone();
        }

        let mut heads_by_state: BTreeMap<StateID, ParserGSS> = BTreeMap::new();
        for edge in gss.peek() {
            let sid = edge.state_id;
            let iso = gss.isolate(Some(edge));
            heads_by_state.entry(sid).and_modify(|acc| *acc = acc.merge(&iso)).or_insert(iso);
        }

        let mut shifted: Vec<ParserGSS> = Vec::new();

        while let Some((state_id, state_gss)) = heads_by_state.pop_first() {
            let row = match self.table.get(&state_id) {
                Some(r) => r,
                None => continue,
            };
            
            let shift = row.shifts.get(&token).cloned();
            let mut reduces_map: BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<crate::glr::table::ProductionID>>> = BTreeMap::new();

            for (bv, reduce) in &row.reduces {
                if bv.contains(token.0) {
                    reduces_map.entry(reduce.len)
                        .or_default()
                        .entry(reduce.nonterminal_id)
                        .or_default()
                        .extend(reduce.production_ids.iter().cloned());
                }
            }

            if shift.is_none() && reduces_map.is_empty() {
                continue;
            }

            // Construct the action on the fly
            let action = if shift.is_some() && reduces_map.is_empty() {
                Stage7ShiftsAndReducesLookaheadValue::Shift(shift.unwrap())
            } else if shift.is_none() && reduces_map.len() == 1 && reduces_map.values().next().unwrap().len() == 1 {
                let (len, nts) = reduces_map.into_iter().next().unwrap();
                let (nt_id, pids) = nts.into_iter().next().unwrap();
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nt_id, len, production_ids: pids }
            } else {
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces: reduces_map }
            };

            match action {
                Stage7ShiftsAndReducesLookaheadValue::Shift(to) => {
                    let edge = ParseStateEdgeContent { state_id: to };
                    shifted.push(state_gss.push(edge));
                }
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                    self.apply_reduces(&state_gss, len, nonterminal_id, &mut heads_by_state);
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                    if let Some(to) = shift {
                        let edge = ParseStateEdgeContent { state_id: to };
                        shifted.push(state_gss.push(edge));
                    }
                    for (len, nts) in reduces {
                        for (&nt_id, _pids) in nts.iter() {
                            self.apply_reduces(&state_gss, *len, nt_id, &mut heads_by_state);
                        }
                    }
                }
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
            if let Some(goto) = self.table[&from_id].gotos.get(&nt) {
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

#[derive(Clone)]
pub struct GLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub stack: ParserGSS,
}

impl Debug for GLRParserState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLRParserState")
            .field("parser", &self.parser)
            // .field("stack", &self.stack)
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
            debug!(3, "Sample path: {:?}", ids);
        }
    }

    pub fn merge_with(&mut self, other: Self) {
        self.stack = self.stack.merge(&other.stack);
    }
}
