// src/representation.rs
#![allow(clippy::derive_partial_eq_without_eq)] // Allow for AllowedXYZ structs if needed

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

// --- AllowedRepresentation Trait ---
/// A marker trait for types that are considered part of the "allowed"
/// simplified representation. Allowed types include primitives (integers, bool, String),
/// Option<Allowed>, Vec<Allowed>, and BTreeMap<String, Allowed>.
pub trait AllowedRepresentation: Clone + Debug {}

// --- IntoAllowedRepresentation Trait ---
/// A trait for types that can be converted into an `AllowedRepresentation`.
pub trait IntoAllowedRepresentation {
    /// The associated type that implements `AllowedRepresentation`.
    type Allowed: AllowedRepresentation;

    /// Converts `self` into its allowed representation.
    fn into_allowed(self) -> Self::Allowed;
}

macro_rules! impl_primitive_conversion {
    ($primitive_type:ty) => {
        impl AllowedRepresentation for $primitive_type {}
        impl IntoAllowedRepresentation for $primitive_type {
            type Allowed = Self;
            fn into_allowed(self) -> Self::Allowed {
                self
            }
        }
    };
}

// Implement AllowedRepresentation and IntoAllowedRepresentation for primitive types
impl_primitive_conversion!(usize);
impl_primitive_conversion!(u8);
impl_primitive_conversion!(u32);
impl_primitive_conversion!(u64);
impl_primitive_conversion!(i32);
impl_primitive_conversion!(i64);
impl_primitive_conversion!(bool);
impl_primitive_conversion!(String);

// --- Implementations for Collections of Allowed Types ---
impl<T: AllowedRepresentation> AllowedRepresentation for Option<T> {}
impl<T: AllowedRepresentation> AllowedRepresentation for Vec<T> {}
impl<T: AllowedRepresentation> AllowedRepresentation for BTreeMap<String, T> {}

impl<T: IntoAllowedRepresentation> IntoAllowedRepresentation for Option<T> {
    type Allowed = Option<T::Allowed>;
    fn into_allowed(self) -> Self::Allowed {
        self.map(|v| v.into_allowed())
    }
}

impl<T: IntoAllowedRepresentation> IntoAllowedRepresentation for Vec<T> {
    type Allowed = Vec<T::Allowed>;
    fn into_allowed(self) -> Self::Allowed {
        self.into_iter().map(|v| v.into_allowed()).collect()
    }
}

impl<T> IntoAllowedRepresentation for BTreeSet<T>
where
    T: IntoAllowedRepresentation,
    T::Allowed: Ord, // Allowed representation must be Ord to collect into BTreeSet then Vec for stable order
{
    type Allowed = Vec<T::Allowed>; // Represent as a sorted Vec
    fn into_allowed(self) -> Self::Allowed {
        let mut vec: Vec<T::Allowed> = self.into_iter().map(|v| v.into_allowed()).collect();
        vec.sort_unstable(); // Ensure canonical order
        vec
    }
}

// --- Specific ID type conversions to usize ---
macro_rules! impl_id_conversion_to_usize {
    ($id_type:ty) => {
        impl IntoAllowedRepresentation for $id_type {
            type Allowed = usize;
            fn into_allowed(self) -> usize {
                self.0
            }
        }
    };
}

// Apply to your ID newtypes
use crate::types::TerminalID as GrammarTokenID_Alias;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::glr::table::{StateID, ProductionID, NonTerminalID, TerminalID as TableTerminalID};
use crate::finite_automata::GroupID;

impl_id_conversion_to_usize!(GrammarTokenID_Alias);
impl_id_conversion_to_usize!(LLMTokenID);
impl_id_conversion_to_usize!(TokenizerStateID);
impl_id_conversion_to_usize!(StateID);
impl_id_conversion_to_usize!(ProductionID);
impl_id_conversion_to_usize!(NonTerminalID);


// --- BiBTreeMap conversion ---
use bimap::BiBTreeMap;

#[derive(Debug, Clone, PartialEq)] // Added PartialEq
pub struct AllowedBiBTreeMap<L: AllowedRepresentation, R: AllowedRepresentation>(pub Vec<(L, R)>);
impl<L: AllowedRepresentation, R: AllowedRepresentation> AllowedRepresentation for AllowedBiBTreeMap<L, R> {}

// Generic BiBTreeMap conversion where L and R are IntoAllowedRepresentation
impl<L, R> IntoAllowedRepresentation for BiBTreeMap<L, R>
where
    L: IntoAllowedRepresentation,
    R: IntoAllowedRepresentation,
    L: Ord + Clone,
    R: Ord + Clone,
{
    type Allowed = AllowedBiBTreeMap<L::Allowed, R::Allowed>;
    fn into_allowed(self) -> Self::Allowed {
        todo!()
    }
}

// --- U8Set conversion ---
#[derive(Debug, Clone, PartialEq)]
pub struct AllowedU8Set {
    pub ranges: Vec<(u8, u8)>,
}
impl AllowedRepresentation for AllowedU8Set {}

impl IntoAllowedRepresentation for crate::datastructures::u8set::U8Set {
    type Allowed = AllowedU8Set;
    fn into_allowed(self) -> Self::Allowed {
        let mut ranges = Vec::new();
        let mut current_start = None;
        let mut current_end = None;

        for i in 0..=255 {
            if self.contains(i) {
                if current_start.is_none() {
                    current_start = Some(i);
                }
                current_end = Some(i);
            } else {
                if let (Some(start), Some(end)) = (current_start, current_end) {
                    ranges.push((start, end));
                }
                current_start = None;
                current_end = None;
            }
        }
        if let (Some(start), Some(end)) = (current_start, current_end) {
            ranges.push((start, end));
        }
        AllowedU8Set { ranges }
    }
}

// --- HybridBitset (LLMTokenBV) conversion ---
#[derive(Debug, Clone, PartialEq)]
pub struct AllowedHybridBitset {
    pub indices: Vec<usize>,
}
impl AllowedRepresentation for AllowedHybridBitset {}

impl IntoAllowedRepresentation for crate::datastructures::hybrid_bitset::HybridBitset {
    type Allowed = AllowedHybridBitset;
    fn into_allowed(self) -> Self::Allowed {
        AllowedHybridBitset { indices: self.iter().collect() }
    }
}
pub type AllowedLLMTokenBV = AllowedHybridBitset;


// --- Grammar related structs ---
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedNonTerminal(pub String);
impl AllowedRepresentation for AllowedNonTerminal {}
impl IntoAllowedRepresentation for NonTerminal {
    type Allowed = AllowedNonTerminal;
    fn into_allowed(self) -> Self::Allowed {
        AllowedNonTerminal(self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedTerminal(pub String);
impl AllowedRepresentation for AllowedTerminal {}
impl IntoAllowedRepresentation for Terminal {
    type Allowed = AllowedTerminal;
    fn into_allowed(self) -> Self::Allowed {
        AllowedTerminal(self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AllowedSymbol {
    Terminal(AllowedTerminal),
    NonTerminal(AllowedNonTerminal),
}
impl AllowedRepresentation for AllowedSymbol {}
impl IntoAllowedRepresentation for Symbol {
    type Allowed = AllowedSymbol;
    fn into_allowed(self) -> Self::Allowed {
        match self {
            Symbol::Terminal(t) => AllowedSymbol::Terminal(t.into_allowed()),
            Symbol::NonTerminal(nt) => AllowedSymbol::NonTerminal(nt.into_allowed()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedProduction {
    pub lhs: AllowedNonTerminal,
    pub rhs: Vec<AllowedSymbol>,
}
impl AllowedRepresentation for AllowedProduction {}
impl IntoAllowedRepresentation for Production {
    type Allowed = AllowedProduction;
    fn into_allowed(self) -> Self::Allowed {
        AllowedProduction {
            lhs: self.lhs.into_allowed(),
            rhs: self.rhs.into_allowed(), // Uses Vec<T> impl
        }
    }
}

// --- Finite Automata related structs ---
use crate::finite_automata::{DFA, DFAState, Regex as FaRegex};

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedDFAState {
    pub transitions: BTreeMap<String, usize>, // u8 key to String, usize for state index
    pub finalizers: Vec<GroupID>,
    pub possible_future_group_ids: Vec<GroupID>,
    pub group_id_to_u8set: BTreeMap<String, AllowedU8Set>, // GroupID key to String
}
impl AllowedRepresentation for AllowedDFAState {}

impl IntoAllowedRepresentation for DFAState {
    type Allowed = AllowedDFAState;
    fn into_allowed(self) -> Self::Allowed {
        AllowedDFAState {
            transitions: self.transitions.into_iter().map(|(k,v)| (k.to_string(), v)).collect(),
            finalizers: self.finalizers.into_allowed(),
            possible_future_group_ids: self.possible_future_group_ids.into_allowed(),
            group_id_to_u8set: self.group_id_to_u8set.into_iter().map(|(k,v)|(k.to_string(), v.into_allowed())).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedDFA {
    pub states: Vec<AllowedDFAState>,
    pub start_state: usize,
    pub non_greedy_finalizers: Vec<GroupID>,
}
impl AllowedRepresentation for AllowedDFA {}
impl IntoAllowedRepresentation for DFA {
    type Allowed = AllowedDFA;
    fn into_allowed(self) -> Self::Allowed {
        AllowedDFA {
            states: self.states.into_allowed(),
            start_state: self.start_state,
            non_greedy_finalizers: self.non_greedy_finalizers.into_allowed(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedRegex {
    pub dfa: AllowedDFA,
}
impl AllowedRepresentation for AllowedRegex {}
impl IntoAllowedRepresentation for FaRegex {
    type Allowed = AllowedRegex;
    fn into_allowed(self) -> Self::Allowed {
        AllowedRegex { dfa: self.dfa.into_allowed() }
    }
}


// --- GLR Parser related structs ---
use crate::glr::parser::GLRParser;
use crate::glr::table::{Stage7Row, Stage7ShiftsAndReduces};

#[derive(Debug, Clone, PartialEq)]
pub enum AllowedStage7ShiftsAndReduces {
    Shift(usize), // StateID.0
    Reduce {
        production_id: usize, // ProductionID.0
        nonterminal_id: usize, // NonTerminalID.0
        len: usize,
    },
    Split {
        shift: Option<usize>, // StateID.0
        reduces: BTreeMap<String, BTreeMap<String, Vec<usize>>>, // len_str -> (nt_id_str -> Vec<prod_id.0>)
    },
}
impl AllowedRepresentation for AllowedStage7ShiftsAndReduces {}

impl IntoAllowedRepresentation for Stage7ShiftsAndReduces {
    type Allowed = AllowedStage7ShiftsAndReduces;
    fn into_allowed(self) -> Self::Allowed {
        match self {
            Stage7ShiftsAndReduces::Shift(state_id) => AllowedStage7ShiftsAndReduces::Shift(state_id.0),
            Stage7ShiftsAndReduces::Reduce { production_id, nonterminal_id, len } => {
                AllowedStage7ShiftsAndReduces::Reduce {
                    production_id: production_id.0,
                    nonterminal_id: nonterminal_id.0,
                    len,
                }
            }
            Stage7ShiftsAndReduces::Split { shift, reduces } => {
                let allowed_reduces = reduces
                    .into_iter()
                    .map(|(len, nt_map)| {
                        (
                            len.to_string(),
                            nt_map.into_iter()
                                .map(|(nt_id, prod_set)| (nt_id.0.to_string(), prod_set.into_iter().map(|pid| pid.0).collect()))
                                .collect(),
                        )
                    })
                    .collect();
                AllowedStage7ShiftsAndReduces::Split { shift: shift.map(|s| s.0), reduces: allowed_reduces }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedStage7Row {
    pub shifts_and_reduces: BTreeMap<String, AllowedStage7ShiftsAndReduces>, // TableTerminalID.0 to String
    pub gotos: BTreeMap<String, usize>, // NonTerminalID.0 to String, StateID.0
}
impl AllowedRepresentation for AllowedStage7Row {}

impl IntoAllowedRepresentation for Stage7Row {
    type Allowed = AllowedStage7Row;
    fn into_allowed(self) -> Self::Allowed {
        AllowedStage7Row {
            shifts_and_reduces: self.shifts_and_reduces.into_iter()
                .map(|(tid, action)| (tid.0.to_string(), action.into_allowed()))
                .collect(),
            gotos: self.gotos.into_iter()
                .map(|(ntid, state_id)| (ntid.0.to_string(), state_id.0))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedGLRParser {
    pub stage_7_table: BTreeMap<String, AllowedStage7Row>, // StateID.0 to String
    pub productions: Vec<AllowedProduction>,
    pub terminal_map: AllowedBiBTreeMap<AllowedTerminal, usize>, // TableTerminalID.0
    pub non_terminal_map: AllowedBiBTreeMap<AllowedNonTerminal, usize>, // NonTerminalID.0
    pub item_set_map_placeholder: String,
    pub start_state_id: usize, // StateID.0
}
impl AllowedRepresentation for AllowedGLRParser {}

impl IntoAllowedRepresentation for GLRParser {
    type Allowed = AllowedGLRParser;
    fn into_allowed(self) -> Self::Allowed {
        AllowedGLRParser {
            stage_7_table: self.stage_7_table.into_iter()
                .map(|(k,v)| (k.0.to_string(), v.into_allowed()))
                .collect(),
            productions: self.productions.into_allowed(),
            terminal_map: self.terminal_map.into_allowed(),
            non_terminal_map: self.non_terminal_map.into_allowed(),
            item_set_map_placeholder: "todo: GLRParser.item_set_map is complex (BTreeSet<Item> key)".to_string(),
            start_state_id: self.start_state_id.0,
        }
    }
}

// --- Precomputation related structs ---
use crate::constraint::{PrecomputedFinalizer, PrecomputedNodeContents, PrecomputeNode};

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedPrecomputedFinalizer {
    pub content: BTreeMap<String, AllowedLLMTokenBV>, // TokenizerStateID.0 to String
}
impl AllowedRepresentation for AllowedPrecomputedFinalizer {}
impl IntoAllowedRepresentation for PrecomputedFinalizer {
    type Allowed = AllowedPrecomputedFinalizer;
    fn into_allowed(self) -> Self::Allowed {
        AllowedPrecomputedFinalizer {
            content: self.content.into_iter()
                .map(|(k,v)| (k.0.to_string(), v.into_allowed()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedPrecomputedNodeContents {
    pub finalizers: BTreeMap<String, AllowedPrecomputedFinalizer>, // GrammarTokenID_Alias.0 to String
    pub clean_end: Option<AllowedLLMTokenBV>,
    pub active: AllowedLLMTokenBV,
}
impl AllowedRepresentation for AllowedPrecomputedNodeContents {}
impl IntoAllowedRepresentation for PrecomputedNodeContents {
    type Allowed = AllowedPrecomputedNodeContents;
    fn into_allowed(self) -> Self::Allowed {
        AllowedPrecomputedNodeContents {
            finalizers: self.finalizers().clone().into_iter() // Use .finalizers() and clone
                .map(|(k,v)| (k.0.to_string(), v.into_allowed()))
                .collect(),
            clean_end: self.clean_end.into_allowed(),
            active: self.active.into_allowed(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedPrecomputeNode {
    pub value: AllowedPrecomputedNodeContents,
    pub children_placeholder: String,
    pub max_depth: usize,
}
impl AllowedRepresentation for AllowedPrecomputeNode {}
impl IntoAllowedRepresentation for PrecomputeNode { // PrecomputeNode is Trie<...>
    type Allowed = AllowedPrecomputeNode;
    fn into_allowed(self) -> Self::Allowed {
        AllowedPrecomputeNode {
            value: self.value.clone().into_allowed(),
            children_placeholder: format!("todo: PrecomputeNode children are complex ({} edge keys)", self.children().len()),
            max_depth: self.max_depth,
        }
    }
}

// --- GrammarConstraint ---
use crate::constraint::GrammarConstraint;

#[derive(Debug, Clone, PartialEq)]
pub struct AllowedGrammarConstraint {
    pub tokenizer: AllowedRegex,
    pub parser: AllowedGLRParser,
    pub precomputed: BTreeMap<String, AllowedPrecomputeNode>, // TokenizerStateID.0 to String
    pub llm_token_map: AllowedBiBTreeMap<String, usize>, // Vec<u8> (base64) to LLMTokenID.0
    pub token_name_map: AllowedBiBTreeMap<String, usize>, // String to usize
    pub max_original_llm_token_id: usize,
    pub original_to_internal_id_bimap: AllowedBiBTreeMap<usize, usize>, // usize to usize
    pub internal_max_llm_token: usize,
}
impl AllowedRepresentation for AllowedGrammarConstraint {}

impl IntoAllowedRepresentation for GrammarConstraint {
    type Allowed = AllowedGrammarConstraint;
    fn into_allowed(self) -> Self::Allowed {
        // Custom handling for BiBTreeMap<Vec<u8>, LLMTokenID>
        let mut llm_token_map_pairs: Vec<(String, usize)> = self.llm_token_map.iter()
            .map(|(bytes, id)| (base64::encode(bytes), id.0))
            .collect();
        llm_token_map_pairs.sort_unstable_by(|(s1, id1), (s2, id2)| s1.cmp(s2).then_with(|| id1.cmp(id2)));
        let allowed_llm_token_map = AllowedBiBTreeMap(llm_token_map_pairs);

        AllowedGrammarConstraint {
            tokenizer: self.tokenizer.into_allowed(),
            parser: self.parser.into_allowed(),
            precomputed: self.precomputed.into_iter()
                .map(|(k,v)| (k.0.to_string(), v.into_allowed()))
                .collect(),
            llm_token_map: allowed_llm_token_map,
            token_name_map: self.token_name_map.into_allowed(),
            max_original_llm_token_id: self.max_original_llm_token_id,
            original_to_internal_id_bimap: self.original_to_internal_id_bimap.into_allowed(),
            internal_max_llm_token: self.internal_max_llm_token,
        }
    }
}
