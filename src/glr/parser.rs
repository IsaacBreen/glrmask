use crate::constraint::{IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediatePrecomputedNodeContents3, IntermediateTrie3EdgeKey, LLMTokenBV, LLMVocab, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV, Trie3God, IntermediateTrie3GodWrapper};
use crate::datastructures::gss_leveled_adapter::{find_longest_path, gather_gss_stats, GSSNode, GSSPeek, GSSStats, StoredPrecomputeNodeIndex, StoredTrieGodWrapper};
use crate::datastructures::gss_leveled_adapter::{print_gss_forest, Acc, GSSPopper, GSSPopperItem, GSSPrintConfig, deep_add_precompute_trie_edges};
use crate::datastructures::ArcPtrWrapper;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::datastructures::gss_leveled_adapter::map_trie3_node_ids;
use crate::glr::table::{CombinedRow, Goto, HallucinatedRow, NonTerminalID, ProductionID, Row, Stage7ShiftsAndReducesLookaheadValue, StateID, SubstringGoto, Table, TerminalID};
use crate::tokenizer::LLMTokenID;
use std::any::Any;
use crate::constraint_stored_cache_utils::optimize_stored_cache;
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
use crate::datastructures::gss_leveled_adapter::{is_simple_gss, PruneAndTransformRecursiveMemo};
use crate::datastructures::leveled_gss::Merge;
use crate::datastructures::trie::{Trie2Index, TrieStats};


// const MAX_MERGE_DEPTH: usize = usize::MAX;
// const MAX_MERGE_DEPTH: usize = 2;
const MAX_MERGE_DEPTH: usize = 1;
// const MAX_MERGE_DEPTH: usize = 0;


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


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseState {
    pub stack: Arc<GSSNode>,
    pub accepted_state: Option<Arc<GSSNode>>,
    pub prev_accepted_state: Arc<GSSNode>,
    pub trie2_god: Option<IntermediateTrie3GodWrapper>,
}

impl ParseState {
    pub fn new() -> Self {
        ParseState {
            stack: Arc::new(GSSNode::new_dead()),
            accepted_state: None,
            prev_accepted_state: Arc::new(GSSNode::new_dead()),
            trie2_god: None,
        }
    }

    pub(crate) fn with_stack(stack: Arc<GSSNode>) -> Self {
        ParseState {
            stack,
            accepted_state: None,
            prev_accepted_state: Arc::new(GSSNode::new_dead()),
            trie2_god: None, // TODO: rename
        }
    }

    pub(crate) fn with_god(mut self, trie2_god: IntermediateTrie3GodWrapper) -> Self {
        self.trie2_god = Some(trie2_god);
        self
    } 
    
    pub(crate) fn with_maybe_god(mut self, maybe_god: Option<IntermediateTrie3GodWrapper>) -> Self {
        self.trie2_god = maybe_god;
        self
    }

    #[time_it]
    pub fn merge(&mut self, mut other: ParseState) {
        timeit!("ParseState::merge::merge main stacks", {
        Arc::make_mut(&mut self.stack).merge_with_depth(MAX_MERGE_DEPTH, &other.stack);
        // Arc::make_mut(&mut self.stack).inner = self.stack.inner.normalize();
        });
        timeit!("ParseState::merge::merge accepted states", {
        if let Some(other_accepted) = other.accepted_state {
            if let Some(self_accepted) = self.accepted_state.as_mut() {
                Arc::make_mut(self_accepted).merge_with_depth(MAX_MERGE_DEPTH, &other_accepted);
            } else {
                self.accepted_state = Some(other_accepted);
            }
        }
        });
        timeit!("ParseState::merge::merge prev accepted states", {
        Arc::make_mut(&mut self.prev_accepted_state).merge_with_depth(MAX_MERGE_DEPTH, &other.prev_accepted_state);
        // assert_eq!(self.trie2_god.is_none(), other.trie2_god.is_none());
        if self.trie2_god.is_some() && other.trie2_god.is_some() {
            assert_eq!(self.trie2_god.as_ref().unwrap(), other.trie2_god.as_ref().unwrap());
        } else if other.trie2_god.is_some() {
            self.trie2_god = other.trie2_god;
        }
        });
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
    ContinueFromHallucinateState,
    Fail,
    #[default]
    Panic,
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessTokenAdvancedConfig {
    pub below_bottom_mode: BelowBottomReductionMode,
    // When set (during a token step), allows reuse of stored precomputations
    // for (nonterminal, terminal) pairs.
    pub current_token: Option<TerminalID>,
    pub reset_cache: bool,
}

impl Default for ProcessTokenAdvancedConfig {
    fn default() -> Self {
        Self {
            below_bottom_mode: BelowBottomReductionMode::default(),
            current_token: None,
            reset_cache: true,
        }
    }
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
    pub reduce_goto_map: BTreeMap<NonTerminalID, BTreeMap<StateID, StateIDBV>>,
    pub hallucinated_row: HallucinatedRow,
    pub hallucinated_state_id: StateID,
    // New: support multiple combined states (including the "combined start" which replaces hallucinated semantics)
    pub combined_rows: BTreeMap<StateID, CombinedRow>,
    pub combined_start_state_id: StateID,
    pub combined_gss: Arc<GSSNode>,
    pub hallucinated_gss: Arc<GSSNode>,
    // New: a dedicated god for stored precomputations and the stored cache itself.
    pub stored_trie_god: IntermediateTrie3GodWrapper,
    pub stored_below_bottom_cache: HashMap<(NonTerminalID, TerminalID), (PrecomputeNode3Index, Arc<GSSNode>)>,
    // New: synthetic reduce rows (one synthetic state per NonTerminal).
    pub synthetic_reduce_rows: BTreeMap<StateID, BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue>>,
    pub synthetic_reduce_state_for_nt: BTreeMap<NonTerminalID, StateID>,
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
        // Do not serialize reduce_goto_map
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
                let reduce_goto_map = crate::glr::table::stage_10(&table);
                let hallucinated_row = crate::glr::table::stage_11_create_hallucinated_row(&table);
                let hallucinated_state_id = StateID(usize::MAX);
                // Build combined states (start combined replaces hallucinated behavior)
                let (combined_rows, combined_start_state_id) = crate::glr::table::stage_12_build_combined_states(&table);

                let combined_gss = {
                    let gss_leaf = GSSNode::new(Acc::new_fresh());
                    let gss = Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: combined_start_state_id }));
                    gss
                };

                let hallucinated_gss = {
                    let gss_leaf = GSSNode::new(Acc::new_fresh());
                    let gss = Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: hallucinated_state_id }));
                    gss
                };

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
                    reduce_goto_map,
                    hallucinated_row,
                    hallucinated_state_id,
                    combined_rows,
                    combined_start_state_id,
                    combined_gss,
                    hallucinated_gss,
                    // Initialize new runtime fields; they will be populated below.
                    stored_trie_god: IntermediateTrie3GodWrapper::new(),
                    stored_below_bottom_cache: HashMap::new(),
                    synthetic_reduce_rows: BTreeMap::new(),
                    synthetic_reduce_state_for_nt: BTreeMap::new(),
                }.initialize_synthetic_and_stored_cache())
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
            .field("combined_rows_size", &self.combined_rows.len())
            .field("combined_start_state_id", &self.combined_start_state_id)
            .field("non_terminal_map", &self.non_terminal_map)
            .field("item_set_map", &self.item_set_map)
            .field("start_state_id", &self.start_state_id)
            .field("everything_state_id", &self.everything_state_id)
            .field("ignore_terminal_id", &self.ignore_terminal_id)
            .field("substring_gotos_size", &self.substring_gotos.len())
            .field("reduce_goto_map_size", &self.reduce_goto_map.len())
            .field("stored_below_bottom_cache_size", &self.stored_below_bottom_cache.len())
            .field("synthetic_reduce_rows_size", &self.synthetic_reduce_rows.len())
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
        self.substring_gotos == other.substring_gotos &&
        self.reduce_goto_map == other.reduce_goto_map &&
        self.hallucinated_row == other.hallucinated_row &&
        self.hallucinated_state_id == other.hallucinated_state_id &&
        self.combined_rows == other.combined_rows &&
        self.combined_start_state_id == other.combined_start_state_id
        // Note: stored_trie_god, stored_below_bottom_cache, synthetic_reduce_rows,
        // and synthetic_reduce_state_for_nt are runtime caches and structures and
        // are intentionally not part of structural equality.
        // Comparing arenas and caches across runs is not meaningful here.
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
        reduce_goto_map: BTreeMap<NonTerminalID, BTreeMap<StateID, StateIDBV>>,
        hallucinated_row: HallucinatedRow,
        hallucinated_state_id: StateID,
        combined_rows: BTreeMap<StateID, CombinedRow>,
        combined_start_state_id: StateID,
    ) -> Self {
        let converted_actions: BTreeMap<NonTerminalID, ActionFn> = actions
            .into_iter()
            .map(|(nt, func)| {
                let nt_id = non_terminal_map.get_by_left(&nt)
                    .unwrap_or_else(|| panic!("NonTerminal {:?} not found in non_terminal_map during GLRParser construction", nt));
                (*nt_id, func)
            })
            .collect();

        let combined_gss = {
            let gss_leaf = GSSNode::new(Acc::new_fresh());
            let gss = Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: combined_start_state_id }));
            gss
        };

        let hallucinated_gss = {
            let gss_leaf = GSSNode::new(Acc::new_fresh());
            let gss = Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: hallucinated_state_id }));
            gss
        };

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
            reduce_goto_map,
            hallucinated_row,
            hallucinated_state_id,
            combined_rows,
            combined_start_state_id,
            combined_gss,
            hallucinated_gss,
            stored_trie_god: IntermediateTrie3GodWrapper::new(),
            stored_below_bottom_cache: HashMap::new(),
            synthetic_reduce_rows: BTreeMap::new(),
            synthetic_reduce_state_for_nt: BTreeMap::new(),
        }
        .initialize_synthetic_and_stored_cache()
    }

    fn initialize_synthetic_and_stored_cache(mut self) -> Self {
        // Reset caches
        self.stored_trie_god = IntermediateTrie3GodWrapper::new();
        self.stored_below_bottom_cache.clear();
        self.synthetic_reduce_rows.clear();
        self.synthetic_reduce_state_for_nt.clear();

        // Determine a base state ID for synthetic states (exclude hallucinated_state_id)
        let max_table_sid = self.table.keys().filter(|sid| **sid != self.hallucinated_state_id).map(|sid| sid.0).max().unwrap_or(0);
        let max_combined_sid = self.combined_rows.keys().map(|sid| sid.0).max().unwrap_or(0);
        let mut next_sid_val = std::cmp::max(max_table_sid, max_combined_sid) + 1;

        // Collect deterministic orderings
        let mut nt_ids: Vec<NonTerminalID> = self.non_terminal_map.iter().map(|(_l, r)| *r).collect();
        nt_ids.sort_by_key(|nid| nid.0);
        let mut term_ids: Vec<TerminalID> = self.terminal_map.iter().map(|(_l, r)| *r).collect();
        term_ids.sort_by_key(|tid| tid.0);

        // Build synthetic reduce rows: one state per NT, reduce len=2 for every terminal.
        for nt in nt_ids.iter().cloned() {
            let sid = StateID(next_sid_val);
            next_sid_val += 1;
            let mut per_token: BTreeMap<TerminalID, Stage7ShiftsAndReducesLookaheadValue> = BTreeMap::new();
            for tid in term_ids.iter().cloned() {
                per_token.insert(tid, Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id: nt, len: 1, production_ids: BTreeSet::new() });
            }
            self.synthetic_reduce_rows.insert(sid, per_token);
            self.synthetic_reduce_state_for_nt.insert(nt, sid);
        }

        // Precompute stored cache for all (NT, Terminal) pairs.
        let mut stored_below_bottom_cache: HashMap<(NonTerminalID, TerminalID), (PrecomputeNode3Index, Arc<GSSNode>)> = HashMap::new(); // TEMP
        for nt in nt_ids.iter().cloned() {
            let synthetic_sid = *self.synthetic_reduce_state_for_nt.get(&nt).expect("synthetic state for NT missing");
            for tid in term_ids.iter().cloned() {
                // Create a dedicated precompute root in the stored god
                let root = PrecomputeNode3Index::new(
                    self.stored_trie_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal()))
                );
                // Seed acc with this root
                let mut acc = Acc::new_fresh();
                acc.stored_trie_nodes_mut().insert(root.clone());

                // Build initial state: combined + acc + stored god
                let mut s = self.init_parser_state_combined_with_acc(acc).with_god(self.stored_trie_god.clone());
                // Push the synthetic state corresponding to this NT
                let pushed = s.active_state.stack.as_ref().clone().push(ParseStateEdgeContent { state_id: synthetic_sid });
                s.active_state.stack = Arc::new(pushed);
                // Run a single token step with the configured current_token (so downstream caching can reference it)
                let cfg = ProcessTokenAdvancedConfig { below_bottom_mode: BelowBottomReductionMode::ContinueFromAll, current_token: Some(tid), reset_cache: false, ..Default::default() };
                s.process_token_advanced(tid, &cfg);
                // Store the result in the stored cache
                // self.stored_below_bottom_cache.insert((nt, tid), (root, s.active_state.stack.clone()));
                stored_below_bottom_cache.insert((nt, tid), (root, s.active_state.stack.clone()));  // TEMP
            }
        }
        self.stored_below_bottom_cache = stored_below_bottom_cache;
        
        self.optimize_stored_cache();
        
        self
    }

    /// Optimizes the stored cache by merging structurally equivalent trie nodes.
    pub fn optimize_stored_cache(&mut self) {
        optimize_stored_cache(&mut self.stored_below_bottom_cache, &self.stored_trie_god, 40);
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
            runtime_below_bottom_cache: Default::default(),
        }
    }

    pub fn init_glr_parser_null(&self, llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState { // No longer generic
        GLRParserState {
            parser: self,
            active_state: ParseState::new(),
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            runtime_below_bottom_cache: Default::default(),
        }
    }

    pub fn init_glr_parser_with_acc(&self) -> GLRParserState { // No longer generic
        let initial_parse_state = self.init_parse_state_with_acc();
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            runtime_below_bottom_cache: Default::default(),
        };
        parser_state
    }

    pub fn init_glr_parser_from_parse_state(&self, parse_state: ParseState) -> GLRParserState { // No longer generic
        let mut parser_state = GLRParserState {
            parser: self,
            active_state: parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            runtime_below_bottom_cache: Default::default(),
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
            runtime_below_bottom_cache: Default::default(),
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
            prev_accepted_state: Arc::new(GSSNode::new_dead()),
            trie2_god: None,
        }
    }

    pub fn init_glr_substring_parser_with_everything_state(&self, _llm_vocab: Option<Arc<LLMVocab>>) -> GLRParserState {
        let initial_parse_state = self.init_parse_state_substring_with_everything_state();
        GLRParserState {
            parser: self,
            active_state: initial_parse_state,
            phase: ParserPhase::ReadyForDefaultReductions,
            below_bottom_cache: Default::default(),
            runtime_below_bottom_cache: Default::default(),
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
            prev_accepted_state: Arc::new(GSSNode::new_dead()),
            trie2_god: None,
        }
    }

    pub fn init_parser_state_hallucinated(&self) -> GLRParserState {
        let gss = self.get_hallucinated_gss();
        self.init_glr_parser_from_stack(gss)
    }

    pub fn init_parser_state_hallucinated_with_acc(&self, acc: Acc) -> GLRParserState {
        let gss = self.get_hallucinated_gss_with_acc(acc);
        self.init_glr_parser_from_stack(gss)
    }

    pub fn init_parser_state_combined(&self) -> GLRParserState {
        let gss = self.get_combined_gss();
        self.init_glr_parser_from_stack(gss)
    }

    pub fn init_parser_state_combined_with_acc(&self, acc: Acc) -> GLRParserState {
        let gss = self.get_combined_gss_with_acc(acc);
        self.init_glr_parser_from_stack(gss)
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
            prev_accepted_state: Arc::new(GSSNode::new_dead()),
            trie2_god: None,
        }
    }

    pub fn get_combined_gss(&self) -> Arc<GSSNode> {
        self.combined_gss.clone()
    }

    pub fn get_combined_gss_with_acc(&self, acc: Acc) -> Arc<GSSNode> {
        let mut gss = (*self.combined_gss).clone();
        gss.inner = gss.inner.apply(|_| acc.clone());
        Arc::new(gss)
    }

    pub fn get_hallucinated_gss(&self) -> Arc<GSSNode> {
        self.hallucinated_gss.clone()
    }

    pub fn get_hallucinated_gss_with_acc(&self, acc: Acc) -> Arc<GSSNode> {
        let mut gss = (*self.hallucinated_gss).clone();
        gss.inner = gss.inner.apply(|_| acc.clone());
        Arc::new(gss)
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
        } else if state_id == self.hallucinated_state_id {
            writeln!(f, "{}Items: (Hallucinated state)", indent)?;
        } else {
            writeln!(f, "{}Items: (State ID not found in item set map)", indent)?;
        }

        // --- Actions & Gotos ---
        if let Some(row) = self.table.get(&state_id) {
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
        } else if state_id == self.hallucinated_state_id {
            writeln!(f, "{}Actions (hallucinated):", indent)?;
            if self.hallucinated_row.shifts_and_reduces.is_empty() {
                writeln!(f, "{}  (none)", indent)?;
            } else {
                // Sort by terminal for deterministic output
                let mut keys: Vec<_> = self.hallucinated_row.shifts_and_reduces.keys().cloned().collect();
                keys.sort_by_key(|tid| self.terminal_map.get_by_right(tid).unwrap());
                for tid in keys {
                    let terminal = self.terminal_map.get_by_right(&tid).unwrap();
                    let actions = &self.hallucinated_row.shifts_and_reduces[&tid];
                    for (action, bv) in actions {
                        let action_str = match action {
                            Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
                                format!("Shift {} [states: {:?}]", next_state_id.0, bv)
                            }
                            Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                                let nt_name = self.non_terminal_map.get_by_right(nonterminal_id).unwrap();
                                let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
                                format!("Reduce {} (len {}) via rules [{}] [states: {:?}]", nt_name.0, len, pids.join(", "), bv)
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
                                let mut s = format!("{} [states: {:?}]:", conflict_type, bv);
                                let inner_indent = format!("\n{}        ", indent);
                                if let Some(shift_state) = shift {
                                    let _ = write!(s, "{}  - Shift {}", inner_indent, shift_state.0);
                                }
                                for (_len, nts) in reduces {
                                    for (_nt_id, prod_ids) in nts {
                                        for prod_id_val in prod_ids {
                                            let prod = self.productions.get(prod_id_val.0).unwrap();
                                            let _ = write!(s, "{}  - Reduce by rule #{} ({})", inner_indent, prod_id_val.0, prod);
                                        }
                                    }
                                }
                                s
                            }
                        };
                        writeln!(f, "{}- On {}: {}", indent, terminal, action_str)?;
                    }
                }
            }

            writeln!(f, "{}Default Action:", indent)?;
            let def = &self.hallucinated_row.default_reduce;
            if let Some(reduce_action) = &def.reduce {
                let nt_name = self.non_terminal_map.get_by_right(&reduce_action.0.nonterminal_id).unwrap();
                let pids: Vec<String> = reduce_action.0.production_ids.iter().map(|p| p.0.to_string()).collect();
                writeln!(f, "{}  - Default Reduce {} (len {}) via rules [{}]", indent, nt_name.0, reduce_action.0.len, pids.join(", "))?;
            } else {
                writeln!(f, "{}  - No default reduce", indent)?;
            }
            if def.clone_and_merge {
                writeln!(f, "{}  - Clone and merge", indent)?;
            }

            writeln!(f, "{}Gotos (hallucinated):", indent)?;
            if self.hallucinated_row.gotos.is_empty() {
                writeln!(f, "{}  (No goto actions)", indent)?;
            } else {
                let mut gotos_sorted: Vec<_> = self.hallucinated_row.gotos.keys().cloned().collect();
                gotos_sorted.sort_by_key(|ntid| self.non_terminal_map.get_by_right(ntid).unwrap());
                for ntid in gotos_sorted {
                    let non_terminal = self.non_terminal_map.get_by_right(&ntid).unwrap();
                    let entries = &self.hallucinated_row.gotos[&ntid];
                    for (goto, bv) in entries {
                        let goto_str = if let Some(state_id_val) = goto.state_id {
                            if goto.accept {
                                format!("{} or accept [states: {:?}]", state_id_val.0, bv)
                            } else {
                                format!("{} [states: {:?}]", state_id_val.0, bv)
                            }
                        } else if goto.accept {
                            format!("accept [states: {:?}]", bv)
                        } else {
                            format!("no-op [states: {:?}]", bv)
                        };
                        writeln!(f, "{}  - {} -> {}", indent, non_terminal.0, goto_str)?;
                    }
                }
            }
        } else {
            // Combined state?
            if self.is_combined_state(state_id) {
                writeln!(f, "{}Items: (Combined state)", indent)?;

                writeln!(f, "{}Actions (combined):", indent)?;
                let row = self.combined_rows.get(&state_id).unwrap();
                if row.shifts_and_reduces.is_empty() {
                    writeln!(f, "{}  (none)", indent)?;
                } else {
                    let mut keys: Vec<_> = row.shifts_and_reduces.keys().cloned().collect();
                    keys.sort_by_key(|tid| self.terminal_map.get_by_right(tid).unwrap());
                    for tid in keys {
                        let terminal = self.terminal_map.get_by_right(&tid).unwrap();
                        let actions = &row.shifts_and_reduces[&tid];
                        for (action, bv) in actions {
                            let action_str = match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(next_state_id) => {
                                    format!("Shift {} [states: {:?}]", next_state_id.0, bv)
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                                    let nt_name = self.non_terminal_map.get_by_right(nonterminal_id).unwrap();
                                    let pids: Vec<String> = production_ids.iter().map(|p| p.0.to_string()).collect();
                                    format!("Reduce {} (len {}) via rules [{}] [states: {:?}]", nt_name.0, len, pids.join(", "), bv)
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
                                    let mut s = format!("{} [states: {:?}]:", conflict_type, bv);
                                    let inner_indent = format!("\n{}        ", indent);
                                    if let Some(shift_state) = shift {
                                        let _ = write!(s, "{}  - Shift {}", inner_indent, shift_state.0);
                                    }
                                    for (_len, nts) in reduces {
                                        for (_nt_id, prod_ids) in nts {
                                            for prod_id_val in prod_ids {
                                                let prod = self.productions.get(prod_id_val.0).unwrap();
                                                let _ = write!(s, "{}  - Reduce by rule #{} ({})", inner_indent, prod_id_val.0, prod);
                                            }
                                        }
                                    }
                                    s
                                }
                            };
                            writeln!(f, "{}- On {}: {}", indent, terminal, action_str)?;
                        }
                    }
                }

                writeln!(f, "{}Default Action:", indent)?;
                let def = &row.default_reduce;
                if let Some(reduce_action) = &def.reduce {
                    let nt_name = self.non_terminal_map.get_by_right(&reduce_action.0.nonterminal_id).unwrap();
                    let pids: Vec<String> = reduce_action.0.production_ids.iter().map(|p| p.0.to_string()).collect();
                    writeln!(f, "{}  - Default Reduce {} (len {}) via rules [{}]", indent, nt_name.0, reduce_action.0.len, pids.join(", "))?;
                } else {
                    writeln!(f, "{}  - No default reduce", indent)?;
                }
                if def.clone_and_merge {
                    writeln!(f, "{}  - Clone and merge", indent)?;
                }

                writeln!(f, "{}Gotos (combined):", indent)?;
                if row.gotos.is_empty() {
                    writeln!(f, "{}  (No goto actions)", indent)?;
                } else {
                    let mut gotos_sorted: Vec<_> = row.gotos.keys().cloned().collect();
                    gotos_sorted.sort_by_key(|ntid| self.non_terminal_map.get_by_right(ntid).unwrap());
                    for ntid in gotos_sorted {
                        let non_terminal = self.non_terminal_map.get_by_right(&ntid).unwrap();
                        let entries = &row.gotos[&ntid];
                        for (goto, bv) in entries {
                            let goto_str = if let Some(state_id_val) = goto.state_id {
                                if goto.accept {
                                    format!("{} or accept [states: {:?}]", state_id_val.0, bv)
                                } else {
                                    format!("{} [states: {:?}]", state_id_val.0, bv)
                                }
                            } else if goto.accept {
                                format!("accept [states: {:?}]", bv)
                            } else {
                                format!("no-op [states: {:?}]", bv)
                            };
                            writeln!(f, "{}  - {} -> {}", indent, non_terminal.0, goto_str)?;
                        }
                    }
                }
            } else {
                writeln!(f, "{}Actions & Gotos: (State ID not found in parse table)", indent)?;
            }
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

        // Print all combined states, including the combined start.
        for (sid, _) in &self.combined_rows {
            writeln!(f, "  State {} (Combined):", sid.0)?;
            self.format_state_details(f, *sid, "    ")?;
        }

        // Print the hallucinated state.
        writeln!(f, "  State {} (Hallucinated):", self.hallucinated_state_id.0)?;
        self.format_state_details(f, self.hallucinated_state_id, "    ")?;

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
    phase: ParserPhase,
    below_bottom_cache: HashMap<BelowBottomCacheKey, PrecomputeNode3Index>,
    runtime_below_bottom_cache: HashMap<(NonTerminalID, TerminalID), (PrecomputeNode3Index, Arc<GSSNode>)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct BelowBottomCacheKey {
    nonterminal_id: NonTerminalID,
    terminal_id: TerminalID,
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
        self._log_gss("    ", TerminalID(0), false, false);
        Ok(())
    }
}

// Key is (depth, state_id) to process stacks in a specific order.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct WorkMapKey(usize, StateID);

impl WorkMapKey {
    fn new(depth: usize, state_id: StateID) -> Self {
        WorkMapKey(depth, state_id)
    }
}

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

// New alias: an action coupled with an optional StateID bitvector filter (used by hallucinated state).
type FilteredAction<'a> = (Action<'a>, Option<StateIDBV>);

impl<'a> GLRParserState<'a> { // No longer generic
    pub fn with_god(mut self, trie2_god: IntermediateTrie3GodWrapper) -> GLRParserState<'a> {
        self.active_state.trie2_god = Some(trie2_god);
        self
    } 
    
    pub fn set_runtime_cache(&mut self, cache: HashMap<(NonTerminalID, TerminalID), (PrecomputeNode3Index, Arc<GSSNode>)>) {
        self.runtime_below_bottom_cache = cache;
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
                trie2_god: state.trie2_god.clone(),
            };
            let depth = isolated_state.stack.max_depth();
            let state_id = peek.edge_value().state_id;
            work_map
                .entry(WorkMapKey::new(depth, state_id))
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
        timeit!("GLRParserState::push_state::push_on_parent GSS PUSH", {});
        let new_gss_node_instance = peek.push_on_parent(new_content);
        ParseState {
            stack: Arc::new(new_gss_node_instance),
            accepted_state: self.active_state.accepted_state.clone(),
            prev_accepted_state: self.active_state.prev_accepted_state.clone(),
            trie2_god: None,
        }
    }

    fn handle_action<F>(
        &mut self,
        action: &Action<'a>,
        filter: Option<&StateIDBV>,
        state_id: StateID,
        state: &ParseState,
        per_state_fuel: &Option<usize>,
        work_map: &mut WorkMap,
        reduce_map: &mut Option<&mut WorkMap>,
        shifted_states_todo: &mut VecDeque<ParseState>,
        accepted_states_todo: &mut VecDeque<ParseState>,
        action_selector: &F,
        config: &ProcessTokenAdvancedConfig,
        early_exit_on_shift: bool,
    ) -> (bool, bool) // (found_shift, should_early_exit)
    where
        F: Fn(StateID) -> Vec<FilteredAction<'a>>,
    {
        let mut found_shift = false;

        // If we have a filter (hallucinated action), apply it by adding a precompute3-trie edge
        // across the entire state's GSS before processing the action.
        // This produces a constrained copy of the state.
        let constrained_state_opt = if let Some(bv) = filter {
            timeit!("GLRParserState::handle_action::apply_filter", {
                let mut constrained = state.clone();
                if let Some(god) = constrained.trie2_god.as_ref() { // TODO: rename
                    let key = IntermediateTrie3EdgeKey::Pop(0, bv.clone());
                    deep_add_precompute_trie_edges(
                        &mut constrained.stack,
                        god,
                        &key,
                        &mut || StoredPrecomputeNodeIndex::new(god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3 { end: false }))),
                        &mut PruneAndTransformRecursiveMemo::default(),
                    );
                }
                Some(constrained)
            })
        } else {
            None
        };
        let state = constrained_state_opt.as_ref().unwrap_or(state);

        crate::debug!(5, "Handling action for state ID {}. Action: {:?}, Filter: {:?}", state_id.0, action, filter);
        for peek in GSSNode::peek_iter(&state.stack) {
            assert_eq!(peek.edge_value().state_id, state_id);
            hit!("GLRParserState::handle_action::ForEachPeek");
            match action {
                Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Shift(to)) => {
                    hit!("GLRParserState::handle_action::Shift");
                    crate::debug!(5, "Action: Shift to state {}", to.0);
                    let new_parse_state =
                        self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                    shifted_states_todo.push_back(new_parse_state);
                    found_shift = true;
                    if early_exit_on_shift {
                        return (found_shift, true);
                    }
                }
                Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                    nonterminal_id: nt,
                    len,
                    ..
                }) => {
                    hit!("GLRParserState::handle_action::Reduce");
                    if per_state_fuel == &Some(0) { continue; }
                    let new_per_state_fuel = per_state_fuel.map(|f| f - 1);

                    crate::debug!(5, "Action: Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), len);
                    let (s_new_arc, accepted_s_new_arc, s_new_shifted_arc) = self.reduce_and_goto(&peek, *nt, *len, action_selector, config);
                    if !s_new_arc.is_empty() {
                        let new_parse_state = ParseState {
                            stack: s_new_arc,
                            accepted_state: state.accepted_state.clone(),
                            prev_accepted_state: state.prev_accepted_state.clone(),
                            trie2_god: state.trie2_god.clone(),
                        };
                        if let Some(ref mut r_map) = reduce_map {
                            Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                        } else {
                            Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                        }
                    }
                    if !accepted_s_new_arc.is_empty() {
                        let accepted_parse_state = ParseState {
                            stack: Arc::new(GSSNode::new_dead()),
                            accepted_state: Some(accepted_s_new_arc),
                            prev_accepted_state: state.prev_accepted_state.clone(),
                            trie2_god: state.trie2_god.clone(),
                        };
                        accepted_states_todo.push_back(accepted_parse_state);
                    }
                    if !s_new_shifted_arc.is_empty() {
                        let shifted_parse_state = ParseState {
                            stack: s_new_shifted_arc,
                            accepted_state: state.accepted_state.clone(),
                            prev_accepted_state: state.prev_accepted_state.clone(),
                            trie2_god: state.trie2_god.clone(),
                        };
                        shifted_states_todo.push_back(shifted_parse_state);
                        found_shift = true;
                        if early_exit_on_shift {
                            return (found_shift, true);
                        }
                    }
                }
                Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces }) => {
                    crate::debug!(5, "Action: Split with shift and reduces");
                    if let Some(to) = shift {
                        hit!("GLRParserState::handle_action::Split::Shift");
                        crate::debug!(5, "Action (Split): Shift to state {}", to.0);
                        let new_parse_state =
                            self.push_state(&peek, ParseStateEdgeContent { state_id: *to });
                        shifted_states_todo.push_back(new_parse_state);
                        found_shift = true;
                        if early_exit_on_shift {
                            return (found_shift, true);
                        }
                    }
                    if per_state_fuel != &Some(0) {
                        let new_per_state_fuel = per_state_fuel.map(|f| f - 1);
                        for (len, nts) in reduces {
                            for (nt, _prod_ids) in nts {
                                hit!("GLRParserState::handle_action::Split::Reduce");
                                crate::debug!(5, "Action (Split): Reduce by NT '{}' (len {})", self.parser.non_terminal_map.get_by_right(nt).unwrap(), *len);
                                let (s_new_arc, accepted_s_new_arc, s_new_shifted_arc) = self.reduce_and_goto(&peek, *nt, *len, action_selector, config);
                                if !s_new_arc.is_empty() {
                                    let new_parse_state = ParseState {
                                        stack: s_new_arc,
                                        accepted_state: state.accepted_state.clone(),
                                        prev_accepted_state: state.prev_accepted_state.clone(),
                                        trie2_god: state.trie2_god.clone(),
                                    };
                                    if let Some(ref mut r_map) = reduce_map {
                                        Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                    } else {
                                        Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                    }
                                }
                                if !accepted_s_new_arc.is_empty() {
                                    let accepted_parse_state = ParseState {
                                        stack: Arc::new(GSSNode::new_dead()),
                                        accepted_state: Some(accepted_s_new_arc),
                                        prev_accepted_state: state.prev_accepted_state.clone(),
                                        trie2_god: state.trie2_god.clone(),
                                    };
                                    accepted_states_todo.push_back(accepted_parse_state);
                                }
                                if !s_new_shifted_arc.is_empty() {
                                    let shifted_parse_state = ParseState {
                                        stack: s_new_shifted_arc,
                                        accepted_state: state.accepted_state.clone(),
                                        prev_accepted_state: state.prev_accepted_state.clone(),
                                        trie2_god: state.trie2_god.clone(),
                                    };
                                    shifted_states_todo.push_back(shifted_parse_state);
                                    found_shift = true;
                                    if early_exit_on_shift {
                                        return (found_shift, true);
                                    }
                                }
                            }
                        }
                    }
                }
                Action::Default(default_reduce) => {
                    // This logic is directly moved from process_action_queue.
                    // It operates on `state` within a loop over `peek`, which might be inefficient
                    // but preserves the original behavior.
                    self.handle_default_action(default_reduce, state, per_state_fuel, work_map, reduce_map, shifted_states_todo, accepted_states_todo, action_selector, config);
                }
            }
        }
        (found_shift, false)
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
        F: Fn(StateID) -> Vec<FilteredAction<'a>>,
    {
        let mut found_shift = false;
        assert!(fuel.is_none(), "Fuel is not supported in process_action_queue yet");
        for (state, per_state_fuel) in work_map.values() {
            assert!(per_state_fuel.is_none(), "Per-state fuel is not supported in process_action_queue yet");
        }
        while let Some(entry) = timeit!("GLRParserState::process_action_queue::pop_first", work_map.pop_first()) {
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
            let actions = timeit!("GLRParserState::process_action_queue::action_selector", action_selector(state_id));
            if actions.is_empty() {
                crate::debug!(5, "No action found in state {}", state_id.0);
            } else {
                timeit!("GLRParserState::process_action_queue::handle_actions_loop", {
                    for (action, filter_opt) in actions {
                        let (new_found_shift, early_exit) = self.handle_action(
                            &action,
                            filter_opt.as_ref(),
                            state_id,
                            &state,
                            &per_state_fuel,
                            work_map,
                            &mut reduce_map,
                            shifted_states_todo,
                            accepted_states_todo,
                            &action_selector,
                            config,
                            early_exit_on_shift,
                        );
                        found_shift |= new_found_shift;
                        if early_exit {
                            return found_shift;
                        }
                    }
                });
            }
        }
        return found_shift;
    }

    fn handle_default_action<F>(
        &mut self,
        default_reduce: &DefaultReduce,
        state: &ParseState,
        per_state_fuel: &Option<usize>,
        work_map: &mut WorkMap,
        reduce_map: &mut Option<&mut WorkMap>,
        shifted_states_todo: &mut VecDeque<ParseState>,
        accepted_states_todo: &mut VecDeque<ParseState>,
        action_selector: &F,
        config: &ProcessTokenAdvancedConfig,
    ) where
        F: Fn(StateID) -> Vec<FilteredAction<'a>>,
    {
        // 1) If clone_and_merge is set, add the "current stuff" (not the reduce result) to the shifted queue.
        if default_reduce.clone_and_merge {
            shifted_states_todo.push_back(state.clone());
        }

        // 2) If there's a reduction in the default, do it like a normal reduce.
        if let Some((reduce, allowed_terminals)) = &default_reduce.reduce {
            if per_state_fuel != &Some(0) {
                let new_per_state_fuel = per_state_fuel.map(|f| f - 1);
                let mut constrained_state = state.clone();
                let can_proceed = constrained_state.stack.is_alive();

                if can_proceed {
                    let disallowed_terminals_bv = allowed_terminals.inverted();
                    if !disallowed_terminals_bv.is_empty() {
                        let disallowed_l2 = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::from_iter(
                            std::iter::once((0..=usize::MAX, disallowed_terminals_bv))
                        );

                        crate::datastructures::gss_leveled_adapter::disallow_terminals_and_prune_arc(
                            &mut constrained_state.stack,
                            &disallowed_l2,
                            &mut HashMap::new(),
                        );
                    }

                    if !constrained_state.stack.is_empty() {
                        for peek in GSSNode::peek_iter(&constrained_state.stack) {
                            let (s_new_arc, accepted_s_new_arc, s_new_shifted_arc) = self.reduce_and_goto(
                                &peek,
                                reduce.nonterminal_id,
                                reduce.len,
                                action_selector,
                                config,
                            );
                            if !s_new_arc.is_empty() {
                                let new_parse_state = ParseState {
                                    stack: s_new_arc,
                                    accepted_state: state.accepted_state.clone(),
                                    prev_accepted_state: state.prev_accepted_state.clone(),
                                    trie2_god: state.trie2_god.clone(),
                                };
                                if let Some(ref mut r_map) = reduce_map {
                                    Self::enqueue(r_map, new_parse_state, new_per_state_fuel);
                                } else {
                                    Self::enqueue(work_map, new_parse_state, new_per_state_fuel);
                                }
                            }
                            if !accepted_s_new_arc.is_empty() {
                                let accepted_parse_state = ParseState {
                                    stack: Arc::new(GSSNode::new_dead()),
                                    accepted_state: Some(accepted_s_new_arc),
                                    prev_accepted_state: state.prev_accepted_state.clone(),
                                    trie2_god: state.trie2_god.clone(),
                                };
                                accepted_states_todo.push_back(accepted_parse_state);
                            }
                            if !s_new_shifted_arc.is_empty() {
                                let shifted_parse_state = ParseState {
                                    stack: s_new_shifted_arc,
                                    accepted_state: state.accepted_state.clone(),
                                    prev_accepted_state: state.prev_accepted_state.clone(),
                                    trie2_god: state.trie2_god.clone(),
                                };
                                shifted_states_todo.push_back(shifted_parse_state);
                            }
                        }
                    }
                }
            }
        }
    }

    fn _do_actions_without_default(&mut self, token_id: TerminalID, phase1_todo: &mut WorkMap, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, accepted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig) {
        let token_display = self.parser.terminal_map.get_by_right(&token_id).unwrap();
        crate::debug!(4, "Phase 1: Processing token '{}'", token_display);
        let parser = self.parser;
        timeit!("GLRParserState::step::phase1", {
            self.process_action_queue(
                phase1_todo,
                Some(phase2_todo),
                shifted_states_todo,
                accepted_states_todo,
                |state_id| {
                    if let Some(per_tok) = parser.synthetic_reduce_rows.get(&state_id) {
                        per_tok.get(&token_id)
                            .map(|a| vec![(Action::Normal(a), None)])
                            .unwrap_or_else(|| Vec::new())
                    } else if parser.is_hallucinated_state(state_id) {
                        parser
                            .hallucinated_row
                            .shifts_and_reduces
                            .get(&token_id)
                            .map(|v| {
                                v.iter()
                                    .map(|(a, bv)| (Action::Normal(a), Some(bv.clone())))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_else(|| Vec::new())
                    } else if parser.is_combined_state(state_id) {
                        parser
                            .combined_rows
                            .get(&state_id)
                            .and_then(|r| r.shifts_and_reduces.get(&token_id))
                            .map(|v| {
                                // Note: previous panic removed; behavior now mirrors combined state handling.
                                v.iter()
                                    .map(|(a, bv)| (Action::Normal(a), Some(bv.clone())))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_else(|| Vec::new())
                    } else {
                        parser.table[&state_id].shifts_and_reduces_without_default_reduce
                            .get(&token_id)
                            .map(|a| vec![(Action::Normal(a), None)])
                            .unwrap_or_default()
                    }
                },
                config,
                &mut None,
                false,
            );
        });
    }

    fn _do_actions_with_default(&mut self, token_id: TerminalID, phase2_todo: &mut WorkMap, shifted_states_todo: &mut VecDeque<ParseState>, accepted_states_todo: &mut VecDeque<ParseState>, config: &ProcessTokenAdvancedConfig) {
        crate::debug!(4, "Phase 1 completed, proceeding to Phase 2 with {} shifted states", shifted_states_todo.len());
        let parser = self.parser;
        timeit!("GLRParserState::step::phase2", {
            // Reduces are pushed back onto the same queue (`None`).
            self.process_action_queue(
                phase2_todo,
                None,
                shifted_states_todo,
                accepted_states_todo,
                |state_id| {
                    if let Some(per_tok) = parser.synthetic_reduce_rows.get(&state_id) {
                        per_tok.get(&token_id)
                            .map(|a| vec![(Action::Normal(a), None)])
                            .unwrap_or_else(|| Vec::new())
                    } else if parser.is_hallucinated_state(state_id) {
                        let row = &parser.hallucinated_row;
                        if let Some(token_actions) = row.shifts_and_reduces.get(&token_id) {
                            token_actions.iter().map(|(a, bv)| (Action::Normal(a), Some(bv.clone()))).collect()
                        } else {
                            vec![(Action::Default(&row.default_reduce), None)]
                        }
                    } else if parser.is_combined_state(state_id) {
                        // Prefer concrete actions; no default reductions during Phase 2.
                        parser
                            .combined_rows
                            .get(&state_id)
                            .and_then(|r| r.shifts_and_reduces.get(&token_id))
                            .map(|v| {
                                v.iter()
                                    .map(|(a, bv)| (Action::Normal(a), Some(bv.clone())))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_else(|| Vec::new())
                    } else {
                        parser.table.get(&state_id).expect_else(|| format!("State ID {} not found in parse table during Phase 2", state_id.0)).shifts_and_reduces_full
                            .get(&token_id)
                            .map(|a| vec![(Action::Normal(a), None)])
                            .unwrap_or_default()
                    }
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

    #[time_it]
    fn build_below_bottom_accs(&self, popper: &GSSPopper) -> BTreeMap<usize, Acc> {
        // New simplified version: do NOT push state ID edges here anymore.
        // Just merge Accs by k and return them; edge additions will be handled later.
        let mut result: BTreeMap<usize, Acc> = BTreeMap::new();

        for (k, accs_by_edge) in popper.below_bottom() {
            let final_acc = accs_by_edge.values().map(|arc| arc).fold(Acc::new_fresh(), |a, b| Acc::merge(&a, b));
            // Do not mutate stored_trie_nodes here; handled later.
            result.insert(*k, final_acc);
        }
        result
    }
    #[time_it]
    fn handle_below_bottom(
        &self,
        nt: NonTerminalID,
        below: BTreeMap<usize, Acc>,
        _config: &ProcessTokenAdvancedConfig,
    ) -> Vec<(StateID, Arc<GSSNode>)> {
        // New strategy:
        // - For each k-group, create a GSS root with the merged Acc.
        // - Add a precompute3-trie edge (k, LLMTokenBV::max_ones()) with StateIDBV::max_ones() across this GSS (shallowly via deep helper).
        // - Push a single hallucinated state edge on top.
        // - Return a single (hallucinated_state_id, gss) todo per k-group.
        if below.is_empty() {
            return Vec::new();
        }

        let god = self.active_state.trie2_god.as_ref().expect("Trie2 god missing"); // TODO: rename

        let mut merged_acc_opt: Option<Acc> = None;
        let mut any_sources = false;
        // Eagerly create a destination node. It will be used for any source nodes found.
        let dest = StoredPrecomputeNodeIndex::new(god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3 { end: false })));

        for (k, acc) in below {
            // Add the "k" edge info for popped-below-bottom to precompute trie across this GSS
            let all_states = StateIDBV::max_ones();
            let key = IntermediateTrie3EdgeKey::Pop(k, all_states);

            if !acc.stored_trie_nodes().is_empty() {
                any_sources = true;
                deep_add_precompute_trie_edges(&mut Arc::new(GSSNode::new(acc.clone())), god, &key, &mut || dest.clone(), &mut PruneAndTransformRecursiveMemo::default());
            }

            merged_acc_opt = Some(match merged_acc_opt {
                None => acc,
                Some(existing) => Acc::merge(&existing, &acc),
            });
        }

        let mut new_acc = merged_acc_opt.expect("No Acc built for below-bottom handling");
        // Only rewrite stored_trie_nodes if there were any source nodes to connect to the destination.
        if any_sources {
            new_acc.stored_trie_nodes_mut().clear();
            new_acc.stored_trie_nodes_mut().insert(dest);
        } else {
            new_acc.stored_trie_nodes_mut().clear();
        }

        let (new_gss, start_sid) = match _config.below_bottom_mode {
            BelowBottomReductionMode::ContinueFromHallucinateState => (
                self.parser.get_hallucinated_gss_with_acc(new_acc),
                self.parser.hallucinated_state_id,
            ),
            _ => (
                self.parser.get_combined_gss_with_acc(new_acc),
                self.parser.combined_start_state_id,
            ),
        };
        let new_todo_items = vec![(start_sid, new_gss)];

        new_todo_items
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
    ) -> (Arc<GSSNode>, Arc<GSSNode>, Arc<GSSNode>)
    where
        G: Fn(StateID) -> Vec<FilteredAction<'a>>,
    {
        timeit!({
            let stats = gather_gss_stats(&[peek.isolated_parent().as_ref()]);
            let num_nodes = stats.unique_nodes();
            // format!("GLRParserState::reduce_and_goto::PoppedGSSStats: {} unique nodes, {} edges. len {}", stats.unique_nodes(), stats.total_edges(), len)
            "GLRParserState::reduce_and_goto"
        }, {
        // hit!(&format!("GLRParserState::reduce_and_goto popped nt '{}', len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len));
        // 1) Pop len
        let popper: GSSPopper = timeit!(peek.popn(len));
        crate::debug!(4, "Reducing with NT '{}' and len {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), len);
        crate::debug!(4, "Popped with {} results...", popper.num_predecessors());

        let mut out: Vec<Arc<GSSNode>> = Vec::new();
        let mut accepted_out: Vec<Arc<GSSNode>> = Vec::new();

        // Shared constants and caches for this call
        let noop_key = IntermediateTrie3EdgeKey::NoOp;

        // Memoize deep_add across identical filters and reuse a single destination per filter for this call.
        // This drastically reduces trie insertions and GSS rewrites.
        let god_opt = self.active_state.trie2_god.as_ref();
        let mut filter_ctxs: HashMap<StateIDBV, (StoredPrecomputeNodeIndex, PruneAndTransformRecursiveMemo)> = HashMap::new();

        // Also memoize deep_add calls performed during the "simple GSS" caching for each cached destination.
        let mut cached_dest_memos: BTreeMap<PrecomputeNode3Index, PruneAndTransformRecursiveMemo> = BTreeMap::new();

        // Collect todo pairs and deduplicate by (predecessor_state_id, isolated_parent pointer).
        let mut todo_map: BTreeMap<StateID, BTreeMap<*const GSSNode, Arc<GSSNode>>> = BTreeMap::new();

        // Handle "below bottom" (substring parsing continuation) first, adding to the todo list.
        timeit!("GLRParserState::reduce_and_goto::HandleBelowBottom", {
            if !popper.below_bottom().is_empty() {
                match config.below_bottom_mode {
                    BelowBottomReductionMode::Fail => {
                        crate::debug!(5, "Popped below bottom, failing these parse paths.");
                    }
                    BelowBottomReductionMode::Panic => {
                        panic!("A reduction popped below the bottom of the stack, and BelowBottomReductionMode was set to Panic.");
                    }
                    _ => {
                        let below_accs = self.build_below_bottom_accs(&popper);
                        let below_todo = self.handle_below_bottom(nt, below_accs, config);
                        crate::debug!(5, "Popped below bottom, hallucinating {} new parse paths.", below_todo.len());
                        for (predecessor_state_id, isolated_parent) in below_todo {
                            let pred_ptr = Arc::as_ptr(&isolated_parent);
                            todo_map.entry(predecessor_state_id)
                                .or_default()
                                .entry(pred_ptr)
                                .or_insert(isolated_parent);
                        }
                    }
                }
            }
        });

        // Standard reductions along in-graph paths
        timeit!("GLRParserState::reduce_and_goto::BuildTodoMap", {
            for popper_item in popper.iter() {
                for peek2 in popper_item.peek_iter() {
                    let predecessor_state_id = peek2.edge_value().state_id;
                    let isolated_parent = peek2.isolated_parent();
                    // if predecessor_state_id == self.parser.combined_start_state_id {
                    //     println!("peek: {}", peek._parent().inner.to_graph_string(false));
                    //     println!("popper: {}", popper._inner().to_graph_string(false));
                    //     println!("parent: {}", peek2._parent().inner.to_graph_string(false));
                    //     assert!(isolated_parent.inner.inner_ptr_eq(&self.parser.get_combined_gss_with_acc(isolated_parent.inner.reduce_acc().unwrap()).inner), "HMM!.\n{}\n{}", isolated_parent.inner.to_graph_string(false), self.parser.get_combined_gss_with_acc(isolated_parent.inner.reduce_acc().unwrap()).inner.to_graph_string(false));
                    // }
                    let pred_ptr = Arc::as_ptr(&isolated_parent);
                    todo_map.entry(predecessor_state_id)
                        .or_default()
                        .entry(pred_ptr)
                        .or_insert(isolated_parent);
                }
            }
        });

        crate::debug!(4, "Total unique predecessor states to process for GOTO: {}", todo_map.len());
        for (predecessor_state_id, parents_map) in todo_map {
            crate::debug!(9, "Processing predecessor state {} with {} isolated parents", predecessor_state_id.0, parents_map.len());
            for (_pred_ptr, isolated_parent) in parents_map {
                timeit!("GLRParserState::reduce_and_goto::HandleGotos", { // ~500 calls
                let mut seen_nts: HashSet<NonTerminalID> = HashSet::new();
                let mut seen_gotos = HashSet::new();
                let mut nt_queue = VecDeque::new();
                nt_queue.push_back(nt);

                crate::debug!(9, "Processing nonterminal '{}' from predecessor state {}", self.parser.non_terminal_map.get_by_right(&nt).unwrap(), predecessor_state_id.0);
                while let Some(current_nt) = nt_queue.pop_front() {
                    // GOTO lookup from predecessor_state_id, possibly hallucinated.
                    let gotos_with_filters: Vec<(Goto, Option<StateIDBV>)> = timeit!("GLRParserState::reduce_and_goto::HandleGotos::WhileLet::NTQueuePop", {
                    if self.parser.is_combined_state(predecessor_state_id) {
                        // Fetch all possible gotos for this NT with associated origin filters from combined rows.
                        if let Some(entries) = self.parser.combined_rows.get(&predecessor_state_id).and_then(|r| r.gotos.get(&current_nt)) {
                            entries.iter().map(|(g, bv)| (*g, Some(bv.clone()))).collect()
                        } else {
                            Vec::new()
                        }
                    } else if self.parser.is_hallucinated_state(predecessor_state_id) {
                        if let Some(entries) = self.parser.hallucinated_row.gotos.get(&current_nt) {
                            entries.iter().map(|(g, bv)| (*g, Some(bv.clone()))).collect()
                        } else {
                            Vec::new()
                        }
                    } else {
                        let goto: Goto = *self
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
                        vec![(goto, None)]
                    }
                    });
                    crate::debug!(5, "Found {} GOTO entries for NT '{}' from state {}: {:?}", gotos_with_filters.len(), self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id.0, gotos_with_filters);
                    // crate::debug!(9, "Found {} GOTO entries for NT '{}' from state {}:", gotos_with_filters.len(), self.parser.non_terminal_map.get_by_right(&current_nt).unwrap(), predecessor_state_id.0);
                    // for (goto, maybe_filter) in &gotos_with_filters {
                    //     crate::debug!(9, "  GOTO to state {:?}, accept: {}, filter: {:?}", goto.state_id, goto.accept, maybe_filter);
                    // }

                    timeit!("GLRParserState::reduce_and_goto::HandleGotos::WhileLet::ForEachGoto", { // SLOW POINT, ~5k calls
                    for (goto, maybe_filter) in gotos_with_filters {
                        if !seen_gotos.insert(goto) {
                            crate::debug!(5, "Skipping GOTO to state {:?}, accept: {}, filter: {:?}", goto.state_id, goto.accept, maybe_filter);
                            continue;
                        }
                        // Apply the optional state filter (for hallucinated transitions) before consuming the GOTO.
                        let mut parent_after_filter = isolated_parent.clone();
                        if let (Some(god), Some(bv)) = (god_opt, maybe_filter.as_ref()) {
                            // Reuse a single destination per unique state filter BV and memoize the transformation.
                            let (dest, memo) = filter_ctxs.entry(bv.clone()).or_insert_with(|| {
                                let new_dest = StoredPrecomputeNodeIndex::new(god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3 { end: false })));
                                (new_dest, PruneAndTransformRecursiveMemo::default())
                            });
                            let key = IntermediateTrie3EdgeKey::Pop(0, bv.clone());
                            crate::debug!(5, "Applying state filter {:?}.", bv);
                            deep_add_precompute_trie_edges(
                                &mut parent_after_filter, god, &key,
                                &mut || dest.clone(),
                                memo,
                            );
                        }

                        // Accept contribution (store isolated parent)
                        if goto.accept {
                            accepted_out.push(parent_after_filter.clone());
                        }

                        // timeit!("GLRParserState::reduce_and_goto::HandleGotos::WhileLet::ForEachGoto::ProcessGoto", {});
                        if let Some(goto_state_id) = goto.state_id {
                            let actions = action_selector(goto_state_id);
                            if actions.len() == 1 {
                                match &actions[0].0 {
                                    Action::Normal(Stage7ShiftsAndReducesLookaheadValue::Reduce {
                                        nonterminal_id: next_nt,
                                        len: 1,
                                        ..
                                    }) => {
                                        // Unit reduce chain: continue
                                        crate::debug!(5, "Unit reduce chain: GOTO to state {} leads to reduce of NT '{}', continuing chain.", goto_state_id.0, self.parser.non_terminal_map.get_by_right(next_nt).unwrap());
                                        if seen_nts.insert(*next_nt) {
                                            nt_queue.push_back(*next_nt);
                                        }
                                    }
                                    Action::Default(def) => {
                                        // If the default reduce isn't a unit reduce, we must commit the current goto result.
                                        if def.clone_and_merge
                                            || def
                                                .reduce
                                                .as_ref()
                                                .map_or(false, |r| r.0.len != 1)
                                        {
                                            timeit!("GLRParserState::reduce_and_goto::HandleGotos::WhileLet::ForEachGoto::DefaultNonUnit GSS PUSH", {});
                                            crate::debug!(5, "Pushing GOTO to state {:?}, accept: {}, filter: {:?}", goto_state_id, goto.accept, maybe_filter);
                                            out.push(Arc::new(parent_after_filter.push(
                                                ParseStateEdgeContent {
                                                    state_id: goto_state_id,
                                                },
                                            )));
                                        }
                                        // If it's a unit reduction, continue chaining.
                                        if let Some(reduce) = &def.reduce {
                                            if reduce.0.len == 1 {
                                                if seen_nts.insert(reduce.0.nonterminal_id) {
                                                    nt_queue.push_back(reduce.0.nonterminal_id);
                                                }
                                            }
                                        }
                                        // Otherwise, end chain
                                    }
                                    _ => {
                                        // Not a unit reduction path anymore -> emit a single push to goto_state
                                        timeit!("GLRParserState::reduce_and_goto::HandleGotos::WhileLet::ForEachGoto::NonUnit GSS PUSH", {});
                                        crate::debug!(5, "GOTO to state {} has a non-unit-reduce action ({:?}). Pushing state.", goto_state_id.0, actions[0].0);
                                        out.push(Arc::new(parent_after_filter.push(
                                            ParseStateEdgeContent {
                                                state_id: goto_state_id,
                                            },
                                        )));
                                    }
                                }
                            } else {
                                // Not a unit reduction path anymore -> emit a single push to goto_state
                                timeit!("GLRParserState::reduce_and_goto::HandleGotos::WhileLet::ForEachGoto::MultiAction GSS PUSH", {});
                                crate::debug!(5, "GOTO to state {} has {} actions, not a unit reduce. Pushing state.", goto_state_id.0, actions.len());
                                out.push(Arc::new(parent_after_filter.push(ParseStateEdgeContent {
                                    state_id: goto_state_id,
                                })));
                            }
                        } else {
                            crate::debug!(5, "No GOTO. We're done.");
                            // No goto target -> we're done.
                        }
                    }
                    });
                }
                });
            }
        }

        // --- NEW CACHING LOGIC ---
        let mut final_out: Vec<Arc<GSSNode>> = Vec::new();
        let mut final_shifted = Vec::new();
        if let Some(god) = self.active_state.trie2_god.as_ref() {
            timeit!("GLRParserState::reduce_and_goto::Caching", { // ~500 calls
            for gss_arc in out {
                timeit!("GLRParserState::reduce_and_goto::Caching::ForEachGSS", { // SLOW POINT, ~20k calls
                let simple_gss_info = is_simple_gss(&gss_arc, self.parser.combined_start_state_id)
                    .or_else(|| is_simple_gss(&gss_arc, self.parser.hallucinated_state_id));
                if let Some((state_id, acc)) = simple_gss_info {
                    // assert!(gss_arc.inner.pop().inner_ptr_eq(&self.parser.get_combined_gss_with_acc((*acc).clone()).inner), "Expected simple GSS to have the canonical combined GSS as its isolated parent.\n{}\n{}", gss_arc.inner.pop().to_graph_string(false), self.parser.get_combined_gss().inner.to_graph_string(false));
                    let mut new_gss_arc = gss_arc;

                    // Always perform cache lookup/insertion to prevent infinite loops.
                    let cache_key = BelowBottomCacheKey {
                        nonterminal_id: nt,
                        terminal_id: config.current_token.unwrap(),
                        // nonterminal_id: NonTerminalID(usize::MAX), // Dummy value for this cache use case
                        source_state_id: StateID(usize::MAX),      // Dummy value
                        // goto_state_id: state_id,
                        goto_state_id: StateID(usize::MAX), // Dummy value
                        k: usize::MAX,                             // Dummy value
                    };

                    match self.below_bottom_cache.entry(cache_key) {
                        std::collections::hash_map::Entry::Occupied(occupied) => {
                            // --- CACHE HIT ---
                            crate::debug!(5, "Cache hit for simple GSS to state {}, skipping addition to output.", state_id.0);
                            hit!("GLRParserState::reduce_and_goto::CacheHit");
                            let cached_dest = occupied.get().clone(); 
                            let memo_for_dest = cached_dest_memos.entry(cached_dest.clone()).or_default();
                            deep_add_precompute_trie_edges( 
                                &mut new_gss_arc, god, &noop_key, &mut || cached_dest.clone(), memo_for_dest,
                            );
                        }
                        std::collections::hash_map::Entry::Vacant(vacant) => {
                            // --- CACHE MISS on below_bottom_cache ---
                            if let Some(cur_tok) = config.current_token {
                                // 1. Check runtime cache.
                                if let Some((runtime_root, runtime_gss)) = self.runtime_below_bottom_cache.get(&(nt, cur_tok)).cloned() {
                                    hit!("GLRParserState::reduce_and_goto::RuntimeCacheHit");
                                    vacant.insert(runtime_root.clone());
                                    let memo_for_dest = cached_dest_memos.entry(runtime_root.clone()).or_default();
                                    deep_add_precompute_trie_edges(&mut new_gss_arc, god, &noop_key, &mut || runtime_root.clone(), memo_for_dest);
                                    final_shifted.push(runtime_gss);
                                    continue;
                                }
                                // 2. Check stored cache.
                                else if let Some(dest_god) = self.active_state.trie2_god.as_ref() {
                                    if let Some((stored_root, stored_gss)) = self.parser.stored_below_bottom_cache.get(&(nt, cur_tok)).cloned() {
                                        // --- STORED REUSE on MISS ---
                                        hit!("GLRParserState::reduce_and_goto::StoredCacheReuse");
                                        let (new_roots, id_map) = IntermediatePrecomputeNode3::deep_copy_subtrees_into(
                                            &self.parser.stored_trie_god, dest_god, &[stored_root.clone().into()],
                                        );
                                        let new_root = new_roots[0];
                                        vacant.insert(new_root.clone());

                                        let mut mapped_gss = stored_gss.clone();
                                        map_trie3_node_ids(&mut mapped_gss, &id_map);

                                        let memo_for_dest = cached_dest_memos.entry(new_root.clone()).or_default();
                                        deep_add_precompute_trie_edges(&mut new_gss_arc, god, &noop_key, &mut || new_root.clone(), memo_for_dest);

                                        final_shifted.push(mapped_gss);
                                        continue;
                                    }
                                }
                            }

                            // --- PURE MISS ---
                            crate::debug!(5, "Cache miss for simple GSS to state {}, adding to output.", state_id.0);
                            hit!("GLRParserState::reduce_and_goto::CacheMiss");
                            let new_dest = StoredPrecomputeNodeIndex::new(god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3 { end: false })));
                            let memo_for_dest = cached_dest_memos.entry(new_dest.clone()).or_default();
                            deep_add_precompute_trie_edges(&mut new_gss_arc, god, &noop_key, &mut || new_dest.clone(), memo_for_dest);
                            vacant.insert(new_dest);
                            final_out.push(new_gss_arc);
                        }
                    }
                } else {
                    // Not a simple GSS, keep it for merging.
                    crate::debug!(5, "Non-simple GSS encountered, keeping as-is.");
                    hit!("GLRParserState::reduce_and_goto::NonSimpleGSS");
                    final_out.push(gss_arc);
                }
                });
            }
            });
        } else {
            // No trie god, so no caching is possible.
            final_out = out;
        }

        // Merge results and return
        let mut new_active = timeit!("GLRParserState::reduce_and_goto::MergeActive", GSSNode::merge_many_with_depth(MAX_MERGE_DEPTH, final_out));
        // Arc::make_mut(&mut new_active).inner = new_active.inner.normalize();
        let new_accepted = timeit!("GLRParserState::reduce_and_goto::MergeAccepted", GSSNode::merge_many_with_depth(MAX_MERGE_DEPTH, accepted_out));
        let new_shifted = timeit!("GLRParserState::reduce_and_goto::MergeShifted", GSSNode::merge_many_with_depth(MAX_MERGE_DEPTH, final_shifted));
        (new_active, new_accepted, new_shifted)
        })
    }

    pub fn process_token(&mut self, token_id: TerminalID) {
        self.process_token_advanced(token_id, &ProcessTokenAdvancedConfig::default())
    }

    #[time_it("GLRParserState::process_token_advanced")]
    pub fn process_token_advanced(&mut self, token_id: TerminalID, config: &ProcessTokenAdvancedConfig) {
        let mut config = config.clone();
        config.current_token = Some(token_id);

        if *self.parser.terminal_map.get_by_right(&token_id).unwrap() == Terminal::RegexName("EOF".to_string()) {
            println!("here");
        }
        crate::debug!(4, "Processing token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());

        if config.reset_cache {
            self.below_bottom_cache.clear();
        }

        if Some(token_id) == self.parser.ignore_terminal_id {
            crate::debug!(4, "Ignoring token '{}'", self.parser.terminal_map.get_by_right(&token_id).unwrap());
            self.phase = ParserPhase::ReadyForDefaultReductions; // Skip phase 1 and 2, go straight to phase 3
            return;
        }

        // Carry the current token for reuse logic inside reduce_and_goto.
        let local_cfg = ProcessTokenAdvancedConfig {
            below_bottom_mode: config.below_bottom_mode,
            current_token: Some(token_id),
            ..Default::default()
        };

        self.log_gss("Phase1/2-start", token_id, false, false);

        let mut phase2_todo: WorkMap = WorkMap::new();
        let mut shifted_states_todo: VecDeque<ParseState> = VecDeque::new();
        let mut accepted_states_todo: VecDeque<ParseState> = VecDeque::new();

        if self.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: WorkMap = WorkMap::new();
            Self::enqueue(&mut phase1_todo, self.active_state.clone(), None);
            self._do_actions_without_default(token_id, &mut phase1_todo, &mut phase2_todo, &mut shifted_states_todo, &mut accepted_states_todo, &local_cfg);
        } else { // ParserPhase::ReadyForDefaultReductions
            Self::enqueue(&mut phase2_todo, self.active_state.clone(), None);
        }

        // --- Phase 2 ---
        self._do_actions_with_default(token_id, &mut phase2_todo, &mut shifted_states_todo, &mut accepted_states_todo, &local_cfg);

        // Consolidate all shifted states into the new active_state for phase 3
        crate::debug!(4, "Phase 2 completed, consolidating {} shifted states into active state", shifted_states_todo.len());
        let mut next_active = ParseState {
            stack: Arc::new(GSSNode::new_dead()),
            accepted_state: None,
            prev_accepted_state: self.active_state.prev_accepted_state.clone(),
            trie2_god: self.active_state.trie2_god.clone(),
        };
        timeit!("GLRParserState::process_token_advanced::MergeShiftedStates", {
        for state in shifted_states_todo {
            next_active.merge(state);
        }
        for state in accepted_states_todo {
            next_active.merge(state);
        }
        });
        self.active_state = next_active;

        // Move current accepted state to previous, and reset current.
        self.active_state.prev_accepted_state = self.active_state.accepted_state.take().unwrap_or_else(|| Arc::new(GSSNode::new_dead()));
        self.active_state.accepted_state = None;

        // self.log_gss("Phase1/2-end", token_id, false, false);
        if config.reset_cache {
            self.below_bottom_cache.clear();
        }
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
        let token_config = ProcessTokenAdvancedConfig { below_bottom_mode: config.below_bottom_mode, current_token: None, ..Default::default() };

        let parser = self.parser;
        // Run the generic action-processing loop with a Default-only selector.
        // - reduce_map = None to keep enqueuing reductions back to the same queue until closure.
        // - action_selector returns the Default action for each row (no token actions here).
        self.process_action_queue(
            &mut work_map,
            None,
            &mut shifted_states_todo,
            &mut accepted_states_todo,
            |state_id| vec![(Action::Default(&parser.table.get(&state_id).expect_else(|| format!("State ID {} not found in parse table during Phase 3", state_id.0)).default_reduce), None)],
            &token_config,
            &mut fuel,
            false,
        );

        // Consolidate all survivors into the new active state.
        let mut next_active = ParseState::new().with_maybe_god(self.active_state.trie2_god.clone());
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
                self.log_gss("has_action_for-start", token_id, false, false);
                let mut llm_tokens = LLMTokenBV::zeros();
                for peek in GSSNode::peek_iter(&self.active_state.stack) {
                    let sid = peek.edge_value().state_id;
                    let mut actions_exist = false;
                    match self.phase {
                        ParserPhase::ReadyForToken => {
                            if self.parser.is_combined_state(sid) {
                                actions_exist = self.parser
                                    .combined_rows
                                    .get(&sid)
                                    .and_then(|r| r.shifts_and_reduces.get(&token_id))
                                    .map(|v| !v.is_empty()).unwrap_or(false);
                            } else {
                                actions_exist = self.parser.table[&sid].shifts_and_reduces_without_default_reduce.contains_key(&token_id);
                            }
                        }
                        ParserPhase::ReadyForDefaultReductions => {
                            if self.parser.is_combined_state(sid) {
                                actions_exist = self.parser
                                    .combined_rows
                                    .get(&sid)
                                    .and_then(|r| r.shifts_and_reduces.get(&token_id))
                                    .map(|v| !v.is_empty()).unwrap_or(false);
                                if !actions_exist {
                                    // Consider default action
                                    actions_exist = self.parser.combined_rows.get(&sid).map(|r| r.default_reduce.clone_and_merge || r.default_reduce.reduce.is_some()).unwrap_or(false);
                                }
                            } else {
                                let row = &self.parser.table[&sid];
                                actions_exist = row.shifts_and_reduces_full.contains_key(&token_id) || row.default_reduce.clone_and_merge || row.default_reduce.reduce.is_some();
                            }
                        }
                    }
                    if actions_exist {
                        crate::debug!(4, "Found action for token '{}' in state {}. LLM tokens: {:?}",
                                      self.parser.terminal_map.get_by_right(&token_id).unwrap(),
                                      sid.0, peek.resolved_llm_tokens_union());
                        let peek_llm_tokens = timeit!(peek.resolved_llm_tokens_union());
                        timeit!(llm_tokens |= peek_llm_tokens);
                    } else {
                        timeit!("GLRParserState::has_action_for::no_action_found", {
                            crate::debug!(4, "No action for token '{}' in state {}", self.parser.terminal_map.get_by_right(&token_id).unwrap(), sid.0);
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
        let cfg = ProcessTokenAdvancedConfig {
            below_bottom_mode: BelowBottomReductionMode::default(),
            current_token: Some(token_id),
            ..Default::default()
        };

        let parser = s.parser;
        if s.phase == ParserPhase::ReadyForToken {
            let mut phase1_todo: WorkMap = WorkMap::new();
            Self::enqueue(&mut phase1_todo, s.active_state.clone(), None);
            if s.process_action_queue(
                &mut phase1_todo,
                Some(&mut phase2_todo),
                &mut shifted_states_todo,
                &mut accepted_states_todo,
                |state_id| {
                    if let Some(per_tok) = parser.synthetic_reduce_rows.get(&state_id) {
                        per_tok.get(&token_id)
                            .map(|a| vec![(Action::Normal(a), None)])
                            .unwrap_or_else(|| Vec::new())
                    } else if parser.is_hallucinated_state(state_id) {
                        parser
                            .hallucinated_row
                            .shifts_and_reduces
                            .get(&token_id)
                            .map(|v| {
                                v.iter()
                                    .map(|(a, bv)| (Action::Normal(a), Some(bv.clone())))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_else(|| Vec::new())
                    } else if parser.is_combined_state(state_id) {
                        parser
                            .combined_rows
                            .get(&state_id)
                            .and_then(|r| r.shifts_and_reduces.get(&token_id))
                            .map(|v| {
                                v.iter()
                                    .map(|(a, bv)| (Action::Normal(a), Some(bv.clone())))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_else(|| Vec::new())
                    } else {
                        parser.table[&state_id]
                            .shifts_and_reduces_without_default_reduce
                            .get(&token_id)
                            .map(|a| vec![(Action::Normal(a), None)])
                            .unwrap_or_default()
                    }
                },
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
            |state_id| {
                if let Some(per_tok) = parser.synthetic_reduce_rows.get(&state_id) {
                    per_tok.get(&token_id)
                        .map(|a| vec![(Action::Normal(a), None)])
                        .unwrap_or_else(|| Vec::new())
                } else if parser.is_combined_state(state_id) {
                    parser
                        .combined_rows
                        .get(&state_id)
                        .and_then(|r| r.shifts_and_reduces.get(&token_id))
                        .map(|v| {
                            v.iter()
                                .map(|(a, bv)| (Action::Normal(a), Some(bv.clone())))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_else(|| Vec::new())
                } else {
                    parser.table[&state_id]
                        .shifts_and_reduces_full
                        .get(&token_id)
                        .map(|a| vec![(Action::Normal(a), None)])
                        .unwrap_or_default()
                }
            },
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
            let sid = peek.edge_value().state_id;
            let has = if self.phase == ParserPhase::ReadyForToken {
                if let Some(per_tok) = self.parser.synthetic_reduce_rows.get(&sid) {
                    per_tok.get(&token_id).is_some()
                } else if self.parser.is_combined_state(sid) {
                    self.parser.combined_rows.get(&sid).and_then(|r| r.shifts_and_reduces.get(&token_id)).is_some()
                } else if sid == self.parser.hallucinated_state_id {
                    self.parser.hallucinated_row.shifts_and_reduces.get(&token_id).map(|v| !v.is_empty()).unwrap_or(false)
                } else {
                    self.parser.table[&sid].shifts_and_reduces_without_default_reduce.contains_key(&token_id)
                }
            } else {
                if let Some(per_tok) = self.parser.synthetic_reduce_rows.get(&sid) {
                    per_tok.get(&token_id).is_some()
                } else if self.parser.is_combined_state(sid) {
                    self.parser.combined_rows.get(&sid).and_then(|r| r.shifts_and_reduces.get(&token_id)).is_some()
                } else if sid == self.parser.hallucinated_state_id {
                    self.parser.hallucinated_row.shifts_and_reduces.get(&token_id).map(|v| !v.is_empty()).unwrap_or(false)
                } else {
                    self.parser.table[&sid].shifts_and_reduces_full.contains_key(&token_id)
                }
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
            let sid = peek.edge_value().state_id;
            if self.parser.is_combined_state(sid) {
                let maybe_row = self.parser.combined_rows.get(&sid);
                if let Some(row) = maybe_row {
                    for (tid, actions) in &row.shifts_and_reduces {
                        for (act, _bv) in actions {
                            match act {
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
                }
            } else {
                let row = &self.parser.table[&sid];
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
        }
        out
    }

    /// Returns the set of terminals that cause a REDUCE from at least one top-of-stack state.
    pub fn immediate_reduce_terminals(&self) -> BTreeSet<TerminalID> {
        let mut out = BTreeSet::new();
        for peek in GSSNode::peek_iter(&self.active_state.stack) {
            let sid = peek.edge_value().state_id;
            if self.parser.is_combined_state(sid) {
                let maybe_row = self.parser.combined_rows.get(&sid);
                if let Some(row) = maybe_row {
                    for (tid, actions) in &row.shifts_and_reduces {
                        for (act, _bv) in actions {
                            match act {
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
                }
            } else {
                let row = &self.parser.table[&sid];
                let actions = if self.phase == ParserPhase::ReadyForToken {
                    &row.shifts_and_reduces_without_default_reduce
                } else {
                    &row.shifts_and_reduces_full
                };
                // Synthetic states (if present) cause reduces for all terminals; collect them too.
                if let Some(per_tok) = self.parser.synthetic_reduce_rows.get(&sid) {
                    for (tid, act) in per_tok {
                        match act {
                            Stage7ShiftsAndReducesLookaheadValue::Reduce { .. } => { out.insert(*tid); }
                            Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } if !reduces.is_empty() => { out.insert(*tid); }
                            _ => {}
                        }
                    }
                }
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
    pub fn stats(&self) -> GSSStats {
        gather_gss_stats(&[self.active_state.stack.as_ref()])
    }
    pub fn log_gss(&self, phase: &str, token: TerminalID, explain_states: bool, generate_dot: bool) {
        if !GSS_LOGGING_ENABLED {
            return;
        }
        self._log_gss(phase, token, explain_states, generate_dot);
    }

    pub fn _log_gss(&self, phase: &str, token: TerminalID, explain_states: bool, generate_dot: bool) {
        // crate::debug!(2, "{} - token {} ({:?}) - nodes", phase, token.0, self.parser.terminal_map.get_by_right(&token).map(|t| &t.0));
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
        crate::debug!(2, "{} ({:?}) - accepted: now={}, prev={} - token '{}' ({}) - {}",
                      phase, self.phase, accepted_now, accepted_prev, self.parser.terminal_map.get_by_right(&token).expect_else(|| format!("Token {} not found in terminal map: {:?}", token.0, self.parser.terminal_map)), token.0, stats_breakdown);

        let mut gss_strings = vec![];
        let mut all_state_ids = BTreeSet::new();
        let mut total_nodes = 0;

        for (name, root) in &roots_to_log {
            let stats = gather_gss_stats(&[root.as_ref()]);
            total_nodes += stats.unique_nodes();

            let (current_gss_string, current_state_ids) = {
                let print_full_forest = stats.total_edges() <= MAX;
                let max_edges_to_print = if print_full_forest { usize::MAX } else { MAX };
                let config = GSSPrintConfig {
                    max_edges: max_edges_to_print,
                    ..Default::default()
                };
                let (gss_string, state_ids) = print_gss_forest(&[root.clone()], &self.parser.terminal_map, &config);
                let final_string = if print_full_forest {
                    format!("{} GSS ({} nodes, {} edges):\n{}", name, stats.unique_nodes(), stats.total_edges(), gss_string)
                } else {
                    match find_longest_path(root) {
                        Some(p) => format!("{} GSS too big ({} nodes, {} edges). Longest path ({}): {}",
                                           name,
                                           stats.unique_nodes(),
                                           stats.total_edges(),
                                           p.len(),
                                           p.iter().map(|(ec, _n)| ec.state_id.0) // n is Arc<GSSNode>
                                                .map(|id| id.to_string())
                                                .collect::<Vec<_>>()
                                            .join(" → ")),
                        None => format!("{} GSS too big ({} nodes, {} edges) – path not found", name, stats.unique_nodes(), stats.total_edges()),
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
    pub fn is_combined_state(&self, state_id: StateID) -> bool { self.combined_rows.contains_key(&state_id) }
    pub fn is_hallucinated_state(&self, state_id: StateID) -> bool { state_id == self.hallucinated_state_id }

    pub fn transfer_stored_cache_to_god(
        &self,
        dest_god: &IntermediateTrie3GodWrapper,
    ) -> HashMap<(NonTerminalID, TerminalID), (PrecomputeNode3Index, Arc<GSSNode>)> {
        if self.stored_below_bottom_cache.is_empty() {
            return HashMap::new();
        }

        // 1. Collect all unique roots from the cache to perform a single bulk copy.
        let all_roots_set: BTreeSet<PrecomputeNode3Index> = self
            .stored_below_bottom_cache
            .values()
            .map(|(root, _gss)| *root)
            .collect();
        let all_roots_vec: Vec<PrecomputeNode3Index> = all_roots_set.into_iter().collect();

        // 2. Perform a single bulk copy of all relevant subtrees.
        // This is much more efficient as it traverses the shared parts of the trie only once.
        let (new_roots_vec, id_map) = IntermediatePrecomputeNode3::deep_copy_subtrees_into(
            &self.stored_trie_god,
            dest_god,
            &all_roots_vec,
        );

        // 3. Create a mapping from old roots to their new counterparts.
        let old_to_new_root_map: HashMap<PrecomputeNode3Index, PrecomputeNode3Index> = all_roots_vec
            .into_iter()
            .zip(new_roots_vec.into_iter())
            .collect();

        // 4. Iterate through the original cache to build the new cache,
        //    updating GSS nodes with the global ID map from the bulk copy.
        let mut new_cache = HashMap::new();
        let mut sorted_keys: Vec<_> = self.stored_below_bottom_cache.keys().cloned().collect();
        sorted_keys.sort(); // Keep deterministic iteration for consistency.

        for key in sorted_keys {
            let (old_root, old_gss) = self.stored_below_bottom_cache.get(&key).unwrap();

            // Find the new root corresponding to the old one.
            let new_root = *old_to_new_root_map.get(old_root)
                .expect("Copied root not found in map; this should be impossible.");

            // Update the GSS with the global id_map.
            let mut new_gss = old_gss.clone();
            map_trie3_node_ids(&mut new_gss, &id_map);

            new_cache.insert(key, (new_root, new_gss));
        }

        new_cache
    }
}

impl GLRParser {
    /// Generates a Graphviz DOT representation of the state transitions present in a GSS forest.
    /// This visualizes the portion of the state machine explored by the parser.
    pub fn gss_forest_to_dot(
        &self,
        roots: &[(&str, &GSSNode)],
        original_internal_bimap: Option<&BTreeMap<usize, usize>>,
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
                let acc_str = crate::datastructures::gss_leveled_adapter::format_acc(
                    &node_arc.inner.reduce_acc().unwrap(),
                    &self.terminal_map,
                    original_internal_bimap,
                    llm_token_map,
                    &GSSPrintConfig::default(),
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
    pub fn gss_to_dot(&self, root: &GSSNode, original_internal_bimap: Option<&BTreeMap<usize, usize>>, llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>) -> String {
        self.gss_forest_to_dot(&[("Root", root)], original_internal_bimap, llm_token_map)
    }

    fn print_numeric_stats_summary<T>(
        &self,
        label: &str,
        mut values: Vec<T>,
    ) where
        T: Copy + Into<f64> + std::fmt::Display + PartialOrd,
    {
        if values.is_empty() {
            println!("    {:<30}: (no data)", label);
            return;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        let n_f64 = n as f64;

        let sum: f64 = values.iter().map(|&v| v.into()).sum();
        let mean = sum / n_f64;

        let min = values[0];
        let max = values[n - 1];

        let median = if n % 2 == 1 {
            values[n / 2].into()
        } else {
            (values[n / 2 - 1].into() + values[n / 2].into()) / 2.0
        };

        // Percentiles
        let p1_idx = ((1.0 / 100.0) * (n_f64 - 1.0)).round() as usize;
        let p99_idx = ((99.0 / 100.0) * (n_f64 - 1.0)).round() as usize;
        let p1 = values[p1_idx];
        let p99 = values[p99_idx];

        let variance = if n > 1 {
            values.iter().map(|&v| {
                let diff = v.into() - mean;
                diff * diff
            }).sum::<f64>() / (n_f64 - 1.0) // Sample variance
        } else {
            0.0
        };
        let stdev = variance.sqrt();

        println!(
            "    {:<30}: min={:<8} max={:<8} mean={:<8.2} median={:<8.2} stdev={:<8.2} 1%={:<8} 99%={:<8}",
            label, min, max, mean, median, stdev, p1, p99
        );
    }

    pub fn print_stored_cache_stats(&self) {
        println!("--- Stored Below-Bottom Cache Statistics ---");
        if self.stored_below_bottom_cache.is_empty() {
            println!("Cache is empty.");
            return;
        }
 
        // Group cache entries by value (trie_root, gss_root)
        type CacheValue = (PrecomputeNode3Index, Arc<GSSNode>);
        type CacheKey = (NonTerminalID, TerminalID);
        let mut grouped_entries: Vec<(CacheValue, Vec<CacheKey>)> = Vec::new();
 
        for (key, value) in &self.stored_below_bottom_cache {
            if let Some((_value, keys)) = grouped_entries.iter_mut().find(|(v, _)| v == value) {
                keys.push(*key);
            } else {
                grouped_entries.push((value.clone(), vec![*key]));
            }
        }
 
        println!("\nFound {} unique cache entries out of {} total.", grouped_entries.len(), self.stored_below_bottom_cache.len());
 
        let occurrences: Vec<f64> = grouped_entries.iter().map(|(_, keys)| keys.len() as f64).collect();
        self.print_numeric_stats_summary("Occurrences per unique entry", occurrences);
 
        // Collect all stats
        let all_gss_stats: Vec<GSSStats> = grouped_entries.iter().map(|(value, _keys)| {
            let (_trie_root, gss_root) = value;
            gather_gss_stats(&[gss_root.as_ref()])
        }).collect();

        let all_trie_stats: Vec<TrieStats> = grouped_entries.iter().map(|(value, _keys)| {
            let (trie_root, _gss_root) = value;
            IntermediatePrecomputeNode3::stats(&self.stored_trie_god, &[*trie_root])
        }).collect();
 
        println!("\n--- GSS Stats Summary (distribution over unique entries) ---");
        self.print_numeric_stats_summary("Total Unique Nodes", all_gss_stats.iter().map(|s| s.total_unique_nodes as f64).collect());
        self.print_numeric_stats_summary("Total Edges", all_gss_stats.iter().map(|s| s.total_edges as f64).collect());
        self.print_numeric_stats_summary("Max Lower Depth", all_gss_stats.iter().map(|s| s.max_lower_depth as f64).collect());
        self.print_numeric_stats_summary("Structurally Unique Nodes", all_gss_stats.iter().map(|s| s.num_structurally_unique_nodes as f64).collect());
        self.print_numeric_stats_summary("Max In-Degree", all_gss_stats.iter().map(|s| s.max_in_degree as f64).collect());
        self.print_numeric_stats_summary("Average In-Degree", all_gss_stats.iter().map(|s| s.average_in_degree).collect());
        self.print_numeric_stats_summary("Structural Sharing Factor", all_gss_stats.iter().map(|s| s.structural_sharing_factor).collect());
 
        println!("\n--- Trie Stats Summary (distribution over unique entries) ---");
        self.print_numeric_stats_summary("Reachable Nodes", all_trie_stats.iter().map(|s| s.num_reachable_nodes as f64).collect());
        self.print_numeric_stats_summary("Reachable Edges", all_trie_stats.iter().map(|s| s.num_reachable_edges as f64).collect());
        self.print_numeric_stats_summary("Max Depth", all_trie_stats.iter().map(|s| s.max_depth as f64).collect());
        self.print_numeric_stats_summary("Num Leaves", all_trie_stats.iter().map(|s| s.num_leaves as f64).collect());
        self.print_numeric_stats_summary("Max In-Degree", all_trie_stats.iter().map(|s| s.max_in_degree as f64).collect());
        self.print_numeric_stats_summary("Avg In-Degree", all_trie_stats.iter().map(|s| s.avg_in_degree).collect());
        self.print_numeric_stats_summary("Max Out-Degree", all_trie_stats.iter().map(|s| s.max_out_degree as f64).collect());
        self.print_numeric_stats_summary("Avg Out-Degree", all_trie_stats.iter().map(|s| s.avg_out_degree).collect());
 
        let mut all_unique_accs = BTreeSet::new();
        for stats in &all_gss_stats {
            for acc in &stats.unique_accumulators {
                all_unique_accs.insert(acc.clone());
            }
        }

        println!("\n--- Unique Accumulators Across All Entries ({} total) ---", all_unique_accs.len());
        let config = GSSPrintConfig::default();
        for (i, acc) in all_unique_accs.iter().enumerate() {
            let acc_str = crate::datastructures::gss_leveled_adapter::format_acc(
                acc,
                &self.terminal_map,
                None,
                None,
                &config,
            );
            println!("  #{}: {}", i + 1, acc_str);
        }

        // Sort groups by number of keys to show most common ones first.
        grouped_entries.sort_by_key(|(_, keys)| std::cmp::Reverse(keys.len()));
 
        println!("\n--- Top 5 Most Frequent Unique Cache Entries ---");
        for (i, (value, keys)) in grouped_entries.iter().take(5).enumerate() {
            let (trie_root, gss_root) = value;
 
            println!("\n#{}: Occurs {} times", i + 1, keys.len());
 
            let gss_stats = gather_gss_stats(&[gss_root.as_ref()]);
            println!("  GSS Stats: {:?}", gss_stats);
 
            let trie_stats = IntermediatePrecomputeNode3::stats(&self.stored_trie_god, &[*trie_root]);
            println!("  Trie Stats (root {}): {:?}", trie_root.as_usize(), trie_stats);
 
            println!("  Example (NonTerminal, Terminal) pairs (up to 5):");
            let mut sorted_keys = keys.clone();
            sorted_keys.sort(); // Sorts by NonTerminalID then TerminalID
            for (nt_id, tid) in sorted_keys.iter().take(5) {
                let nt_name = self.non_terminal_map.get_by_right(nt_id).unwrap();
                let t_name = self.terminal_map.get_by_right(tid).unwrap();
                println!("    - ({}, {})", nt_name.0, t_name);
            }
        }
 
        println!("\n--- Combined Statistics ---");
 
        // Combined GSS stats
        if !self.stored_below_bottom_cache.is_empty() {
            let all_gss_roots: Vec<_> = self.stored_below_bottom_cache.values().map(|(_, gss)| gss.clone()).collect();
            let merged_gss_arc = GSSNode::merge_many_with_depth(MAX_MERGE_DEPTH, all_gss_roots.clone());
            let combined_gss_stats = gather_gss_stats(&[merged_gss_arc.as_ref()]);
            println!("Combined GSS Stats (from merging all {} entries):", all_gss_roots.len());
            println!("  {:?}", combined_gss_stats);

            println!("\n  Unique Accumulators in Combined GSS ({} total):", combined_gss_stats.unique_accumulators.len());
            let config = GSSPrintConfig::default();
            let mut sorted_accs: Vec<_> = combined_gss_stats.unique_accumulators.iter().collect();
            sorted_accs.sort(); // Acc implements Ord
            for (i, acc) in sorted_accs.iter().enumerate() {
                let acc_str = crate::datastructures::gss_leveled_adapter::format_acc(
                    acc,
                    &self.terminal_map,
                    None,
                    None,
                    &config,
                );
                println!("    #{}: {}", i + 1, acc_str);
            }
        }
 
        // Combined Trie stats
        if !self.stored_below_bottom_cache.is_empty() {
            let all_trie_roots: Vec<_> = self.stored_below_bottom_cache.values().map(|(tr, _)| *tr).collect();
            let trie_stats = IntermediatePrecomputeNode3::stats(&self.stored_trie_god, &all_trie_roots);
            println!("\nCombined Stored Trie Stats (from {} total cache entries):", all_trie_roots.len());
            println!("  {:?}", trie_stats);
        }

        println!("\n--- Terminal Equivalence Classes ---");
        let mut nt_ids: Vec<_> = self.non_terminal_map.right_values().copied().collect();
        nt_ids.sort();

        let mut term_ids: Vec<_> = self.terminal_map.right_values().copied().collect();
        term_ids.sort();

        let mut terminal_signatures: BTreeMap<TerminalID, Vec<Option<&CacheValue>>> = BTreeMap::new();

        for &tid in &term_ids {
            let signature: Vec<Option<&CacheValue>> = nt_ids
                .iter()
                .map(|ntid| self.stored_below_bottom_cache.get(&(ntid.clone(), tid)))
                .collect();
            terminal_signatures.insert(tid, signature);
        }

        let mut signature_to_terminals: HashMap<Vec<Option<&CacheValue>>, Vec<TerminalID>> = HashMap::new();
        for (tid, signature) in terminal_signatures {
            signature_to_terminals.entry(signature).or_default().push(tid);
        }

        let mut found_any = false;
        for (_signature, terminals) in signature_to_terminals {
            if terminals.len() > 1 {
                found_any = true;
                let mut terminal_names: Vec<String> = terminals.iter().map(|tid| self.terminal_map.get_by_right(tid).unwrap().to_string()).collect();
                terminal_names.sort();
                println!("  - Equivalent Terminals: {{{}}}", terminal_names.join(", "));
            }
        }
        if !found_any {
            println!("  No equivalent terminals found.");
        }
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