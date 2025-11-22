use crate::datastructures::char_transitions::CharTransitions;
use crate::datastructures::bitset2::BitSet;
use crate::datastructures::frozenset::FrozenSet;
use crate::datastructures::u8set::U8Set;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::{BTreeMap, BTreeSet};
use std::collections::BTreeMap as StdMap;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use memory_stats::memory_stats;

pub type GroupID = usize;

#[derive(Debug, Clone)]
pub struct NFAState {
    /// Non-epsilon transitions: list of (input byte, target state).
    /// There may be multiple entries with the same input byte (non-determinism).
    transitions: Vec<(U8Set, usize)>,
    /// Epsilon transitions: target states reachable without consuming input.
    epsilon_transitions: Vec<usize>,
    finalizers: BTreeSet<GroupID>,
    non_greedy_finalizers: BTreeSet<GroupID>,
}

// Manual impl for NFAState
impl JSONConvertible for NFAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();

        // Serialize transitions as a map from u8 to Vec<usize>,
        // matching the previous TrieMap<Vec<usize>> representation.
        let mut transitions_map: StdMap<String, JSONNode> = StdMap::new();
        let mut grouped: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for (set, target) in &self.transitions {
            for byte in set.iter() {
                grouped.entry(byte).or_default().push(*target);
            }
        }
        for (byte, targets) in grouped {
            transitions_map.insert(byte.to_string(), targets.to_json());
        }
        obj.insert("transitions".to_string(), JSONNode::Object(transitions_map));

        obj.insert(
            "epsilon_transitions".to_string(),
            self.epsilon_transitions.to_json(),
        );
        obj.insert("finalizers".to_string(), self.finalizers.to_json());
        obj.insert(
            "non_greedy_finalizers".to_string(),
            self.non_greedy_finalizers.to_json(),
        );
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                // Deserialize transitions from the old JSON format:
                // { "<u8>": [usize, usize, ...], ... }
                let transitions_node = obj
                    .remove("transitions")
                    .ok_or_else(|| "Missing field transitions for NFAState".to_string())?;

                let mut transitions: Vec<(U8Set, usize)> = Vec::new();
                match transitions_node {
                    JSONNode::Object(map) => {
                        for (key_str, val_node) in map {
                            let byte = key_str.parse::<u8>().map_err(|e| {
                                format!(
                                    "Invalid u8 key in NFAState transitions: {}, err: {}",
                                    key_str, e
                                )
                            })?;
                            let targets = Vec::<usize>::from_json(val_node)?;
                            for target in targets {
                                transitions.push((U8Set::from_u8(byte), target));
                            }
                        }
                    }
                    other => {
                        return Err(format!(
                            "NFAState 'transitions' field must be a JSON object, got {:?}",
                            other
                        ))
                    }
                }

                let epsilon_transitions = obj
                    .remove("epsilon_transitions")
                    .ok_or_else(|| "Missing field epsilon_transitions for NFAState".to_string())
                    .and_then(Vec::<usize>::from_json)?;
                let finalizers = obj
                    .remove("finalizers")
                    .ok_or_else(|| "Missing field finalizers for NFAState".to_string())
                    .and_then(BTreeSet::<GroupID>::from_json)?;
                let non_greedy_finalizers = obj
                    .remove("non_greedy_finalizers")
                    .ok_or_else(|| {
                        "Missing field non_greedy_finalizers for NFAState".to_string()
                    })
                    .and_then(BTreeSet::<GroupID>::from_json)?;
                Ok(NFAState {
                    transitions,
                    epsilon_transitions,
                    finalizers,
                    non_greedy_finalizers,
                })
            }
            _ => Err("Expected JSONNode::Object for NFAState".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NFA {
    states: Vec<NFAState>,
    start_state: usize,
}

// Manual impl for NFA
impl JSONConvertible for NFA {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("states".to_string(), self.states.to_json());
        obj.insert("start_state".to_string(), self.start_state.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let states = obj
                    .remove("states")
                    .ok_or_else(|| "Missing field states for NFA".to_string())
                    .and_then(Vec::<NFAState>::from_json)?;
                let start_state = obj
                    .remove("start_state")
                    .ok_or_else(|| "Missing field start_state for NFA".to_string())
                    .and_then(usize::from_json)?;
                Ok(NFA {
                    states,
                    start_state,
                })
            }
            _ => Err("Expected JSONNode::Object for NFA".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DFAState {
    pub transitions: CharTransitions<usize>,
    pub finalizers: BTreeSet<GroupID>,
    pub possible_future_group_ids: BTreeSet<GroupID>,
    pub group_id_to_u8set: BTreeMap<GroupID, U8Set>,
}

// Manual impl for DFAState
impl JSONConvertible for DFAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("transitions".to_string(), self.transitions.to_json());
        obj.insert("finalizers".to_string(), self.finalizers.to_json());
        obj.insert(
            "possible_future_group_ids".to_string(),
            self.possible_future_group_ids.to_json(),
        );
        obj.insert(
            "group_id_to_u8set".to_string(),
            self.group_id_to_u8set.to_json(),
        );
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let transitions = obj
                    .remove("transitions")
                    .ok_or_else(|| "Missing field transitions for DFAState".to_string())
                    .and_then(|n| CharTransitions::<usize>::from_json(n))?;
                let finalizers = obj
                    .remove("finalizers")
                    .ok_or_else(|| "Missing field finalizers for DFAState".to_string())
                    .and_then(BTreeSet::<GroupID>::from_json)?;
                let possible_future_group_ids = obj
                    .remove("possible_future_group_ids")
                    .ok_or_else(|| {
                        "Missing field possible_future_group_ids for DFAState".to_string()
                    })
                    .and_then(BTreeSet::<GroupID>::from_json)?;
                let group_id_to_u8set = obj
                    .remove("group_id_to_u8set")
                    .ok_or_else(|| {
                        "Missing field group_id_to_u8set for DFAState".to_string()
                    })
                    .and_then(|n| BTreeMap::<GroupID, U8Set>::from_json(n))?;
                Ok(DFAState {
                    transitions,
                    finalizers,
                    possible_future_group_ids,
                    group_id_to_u8set,
                })
            }
            _ => Err("Expected JSONNode::Object for DFAState".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DFA {
    pub states: Vec<DFAState>,
    pub start_state: usize,
    pub non_greedy_finalizers: BTreeSet<GroupID>,
}

// Manual impl for DFA
impl JSONConvertible for DFA {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("states".to_string(), self.states.to_json());
        obj.insert("start_state".to_string(), self.start_state.to_json());
        obj.insert(
            "non_greedy_finalizers".to_string(),
            self.non_greedy_finalizers.to_json(),
        );
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let states = obj
                    .remove("states")
                    .ok_or_else(|| "Missing field states for DFA".to_string())
                    .and_then(Vec::<DFAState>::from_json)?;
                let start_state = obj
                    .remove("start_state")
                    .ok_or_else(|| "Missing field start_state for DFA".to_string())
                    .and_then(usize::from_json)?;
                let non_greedy_finalizers = obj
                    .remove("non_greedy_finalizers")
                    .ok_or_else(|| {
                        "Missing field non_greedy_finalizers for DFA".to_string()
                    })
                    .and_then(BTreeSet::<GroupID>::from_json)?;
                Ok(DFA {
                    states,
                    start_state,
                    non_greedy_finalizers,
                })
            }
            _ => Err("Expected JSONNode::Object for DFA".to_string()),
        }
    }
}

// TODO: should this *really* derive `Clone`? Users probably shouldn't clone this, should they?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Regex {
    pub dfa: DFA,
}

// Manual impl for Regex
impl JSONConvertible for Regex {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("dfa".to_string(), self.dfa.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let dfa = obj
                    .remove("dfa")
                    .ok_or_else(|| "Missing field dfa for Regex".to_string())
                    .and_then(DFA::from_json)?;
                Ok(Regex { dfa })
            }
            _ => Err("Expected JSONNode::Object for Regex".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    pub group_id: GroupID,
    pub position: usize,
}

// Manual impl for Match
impl JSONConvertible for Match {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("group_id".to_string(), self.group_id.to_json());
        obj.insert("position".to_string(), self.position.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let group_id = obj
                    .remove("group_id")
                    .ok_or_else(|| "Missing field group_id for Match".to_string())
                    .and_then(GroupID::from_json)?;
                let position = obj
                    .remove("position")
                    .ok_or_else(|| "Missing field position for Match".to_string())
                    .and_then(usize::from_json)?;
                Ok(Match { group_id, position })
            }
            _ => Err("Expected JSONNode::Object for Match".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FinalStateReport {
    pub position: usize,
    pub matches: BTreeMap<GroupID, usize>, // GroupID to position
}

// Manual impl for FinalStateReport
impl JSONConvertible for FinalStateReport {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("position".to_string(), self.position.to_json());
        obj.insert("matches".to_string(), self.matches.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let position = obj
                    .remove("position")
                    .ok_or_else(|| "Missing field position for FinalStateReport".to_string())
                    .and_then(usize::from_json)?;
                let matches = obj
                    .remove("matches")
                    .ok_or_else(|| "Missing field matches for FinalStateReport".to_string())
                    .and_then(|n| BTreeMap::<GroupID, usize>::from_json(n))?;
                Ok(FinalStateReport { position, matches })
            }
            _ => Err("Expected JSONNode::Object for FinalStateReport".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    pub matches: Vec<Match>,
    pub end_state: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegexState<'a> {
    pub regex: &'a Regex,
    pub position: usize,
    pub current_state: usize,
    pub matches: BTreeMap<GroupID, usize>, // Publicly accessible matches (GroupID to position)
    pub done: bool,
}

// RegexState contains a reference, making direct serialization/deserialization complex.
// It's a runtime state, not typically part of a definition to be serialized.
impl<'a> JSONConvertible for RegexState<'a> {
    fn to_json(&self) -> JSONNode {
        todo!("RegexState serialization is complex due to lifetime and reference.")
    }
    fn from_json(_node: JSONNode) -> Result<Self, String> {
        Err("RegexState deserialization is not supported due to lifetime and reference.".to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr {
    U8Seq(Vec<u8>),
    U8Class(U8Set),
    Shared(Arc<Expr>), // Shared sub-expression
    Quantifier(Box<Expr>, QuantifierType),
    Choice(Vec<Expr>),
    Seq(Vec<Expr>),
    Epsilon, // Explicit epsilon transition
}

impl JSONConvertible for Expr {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        match self {
            Expr::U8Seq(bytes) => {
                obj.insert("variant".to_string(), JSONNode::String("U8Seq".to_string()));
                obj.insert("bytes".to_string(), bytes.to_json());
            }
            Expr::U8Class(u8set) => {
                obj.insert("variant".to_string(), JSONNode::String("U8Class".to_string()));
                obj.insert("u8set".to_string(), u8set.to_json());
            }
            Expr::Shared(inner) => {
                obj.insert("variant".to_string(), JSONNode::String("Shared".to_string()));
                obj.insert("inner".to_string(), inner.to_json());
            }
            Expr::Quantifier(expr, q_type) => {
                obj.insert(
                    "variant".to_string(),
                    JSONNode::String("Quantifier".to_string()),
                );
                obj.insert("expr".to_string(), expr.to_json());
                obj.insert("q_type".to_string(), q_type.to_json());
            }
            Expr::Choice(exprs) => {
                obj.insert("variant".to_string(), JSONNode::String("Choice".to_string()));
                obj.insert("exprs".to_string(), exprs.to_json());
            }
            Expr::Seq(exprs) => {
                obj.insert("variant".to_string(), JSONNode::String("Seq".to_string()));
                obj.insert("exprs".to_string(), exprs.to_json());
            }
            Expr::Epsilon => {
                obj.insert("variant".to_string(), JSONNode::String("Epsilon".to_string()));
            }
        }
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let variant = obj
                    .remove("variant")
                    .ok_or_else(|| "Missing field variant for Expr".to_string())
                    .and_then(String::from_json)?;
                match variant.as_str() {
                    "U8Seq" => {
                        let bytes = obj
                            .remove("bytes")
                            .ok_or_else(|| "Missing field bytes for U8Seq".to_string())
                            .and_then(Vec::<u8>::from_json)?;
                        Ok(Expr::U8Seq(bytes))
                    }
                    "U8Class" => {
                        let u8set = obj
                            .remove("u8set")
                            .ok_or_else(|| "Missing field u8set for U8Class".to_string())
                            .and_then(U8Set::from_json)?;
                        Ok(Expr::U8Class(u8set))
                    }
                    "Shared" => {
                        let inner = obj
                            .remove("inner")
                            .ok_or_else(|| "Missing field inner for Shared".to_string())?;
                        let expr = Expr::from_json(inner)?;
                        Ok(Expr::Shared(Arc::new(expr)))
                    }
                    "Quantifier" => {
                        let expr_node = obj
                            .remove("expr")
                            .ok_or_else(|| "Missing field expr for Quantifier".to_string())?;
                        let expr = Box::new(Expr::from_json(expr_node)?);
                        let q_type = obj
                            .remove("q_type")
                            .ok_or_else(|| "Missing field q_type for Quantifier".to_string())
                            .and_then(QuantifierType::from_json)?;
                        Ok(Expr::Quantifier(expr, q_type))
                    }
                    "Choice" => {
                        let exprs = obj
                            .remove("exprs")
                            .ok_or_else(|| "Missing field exprs for Choice".to_string())
                            .and_then(Vec::<Expr>::from_json)?;
                        Ok(Expr::Choice(exprs))
                    }
                    "Seq" => {
                        let exprs = obj
                            .remove("exprs")
                            .ok_or_else(|| "Missing field exprs for Seq".to_string())
                            .and_then(Vec::<Expr>::from_json)?;
                        Ok(Expr::Seq(exprs))
                    }
                    "Epsilon" => Ok(Expr::Epsilon),
                    _ => Err(format!("Unknown variant {} for Expr", variant)),
                }
            }
            _ => Err("Expected JSONNode::Object for Expr".to_string()),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum QuantifierType {
    ZeroOrMore, // *
    OneOrMore,  // +
    ZeroOrOne,  // ?
}

// Manual impl for QuantifierType (enum)
impl JSONConvertible for QuantifierType {
    fn to_json(&self) -> JSONNode {
        let variant_name = match self {
            QuantifierType::ZeroOrMore => "ZeroOrMore",
            QuantifierType::OneOrMore => "OneOrMore",
            QuantifierType::ZeroOrOne => "ZeroOrOne",
        };
        JSONNode::String(variant_name.to_string())
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::String(s) => match s.as_str() {
                "ZeroOrMore" => Ok(QuantifierType::ZeroOrMore),
                "OneOrMore" => Ok(QuantifierType::OneOrMore),
                "ZeroOrOne" => Ok(QuantifierType::ZeroOrOne),
                _ => Err(format!("Unknown variant {} for QuantifierType", s)),
            },
            _ => Err("Expected JSONNode::String for QuantifierType".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprGroup {
    pub expr: Expr,
    pub is_non_greedy: bool,
}

// Manual impl for ExprGroup
impl JSONConvertible for ExprGroup {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("expr".to_string(), self.expr.to_json());
        obj.insert("is_non_greedy".to_string(), self.is_non_greedy.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let expr = obj
                    .remove("expr")
                    .ok_or_else(|| "Missing field expr for ExprGroup".to_string())
                    .and_then(Expr::from_json)?;
                let is_non_greedy = obj
                    .remove("is_non_greedy")
                    .ok_or_else(|| "Missing field is_non_greedy for ExprGroup".to_string())
                    .and_then(bool::from_json)?;
                Ok(ExprGroup {
                    expr,
                    is_non_greedy,
                })
            }
            _ => Err("Expected JSONNode::Object for ExprGroup".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprGroups {
    pub groups: Vec<ExprGroup>,
}

// Manual impl for ExprGroups
impl JSONConvertible for ExprGroups {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("groups".to_string(), self.groups.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let groups = obj
                    .remove("groups")
                    .ok_or_else(|| "Missing field groups for ExprGroups".to_string())
                    .and_then(Vec::<ExprGroup>::from_json)?;
                Ok(ExprGroups { groups })
            }
            _ => Err("Expected JSONNode::Object for ExprGroups".to_string()),
        }
    }
}

impl From<Expr> for ExprGroup {
    fn from(expr: Expr) -> Self {
        ExprGroup {
            expr,
            is_non_greedy: false,
        }
    }
}

impl From<Expr> for ExprGroups {
    fn from(expr: Expr) -> Self {
        ExprGroups {
            groups: vec![ExprGroup {
                expr,
                is_non_greedy: false,
            }],
        }
    }
}

pub fn eat_u8(c: u8) -> Expr {
    Expr::U8Seq(vec![c])
}

pub fn eat_u8_seq(u8s: Vec<u8>) -> Expr {
    Expr::U8Seq(u8s)
}

pub fn eat_u8_set(u8s: U8Set) -> Expr {
    Expr::U8Class(u8s)
}

pub fn eat_u8_negation(c: u8) -> Expr {
    Expr::U8Class(U8Set::from_u8(c).complement())
}

pub fn eat_u8_set_negation(u8s: U8Set) -> Expr {
    Expr::U8Class(u8s.complement())
}

pub fn rep<T: Into<Expr>>(expr: T) -> Expr {
    Expr::Quantifier(Box::new(expr.into()), QuantifierType::ZeroOrMore)
}

pub fn rep1<T: Into<Expr>>(expr: T) -> Expr {
    Expr::Quantifier(Box::new(expr.into()), QuantifierType::OneOrMore)
}

pub fn opt<T: Into<Expr>>(expr: T) -> Expr {
    Expr::Quantifier(Box::new(expr.into()), QuantifierType::ZeroOrOne)
}

pub fn shared<T: Into<Expr>>(expr: T) -> Expr {
    Expr::Shared(Arc::new(expr.into()))
}

pub fn prec<T: Into<Expr>>(_precedence: isize, expr: T) -> ExprGroup {
    ExprGroup {
        expr: expr.into(),
        is_non_greedy: false,
    }
}

pub fn eps() -> Expr {
    Expr::Epsilon
}

pub fn _seq(exprs: Vec<Expr>) -> Expr {
    Expr::Seq(exprs)
}

pub fn _choice(exprs: Vec<Expr>) -> Expr {
    Expr::Choice(exprs)
}

#[macro_export]
macro_rules! choice {
    ($($expr:expr),* $(,)?) => {
        $crate::finite_automata::Expr::Choice(vec![$($expr.into()),*])
    };
}

#[macro_export]
macro_rules! seq {
    ($($expr:expr),* $(,)?) => {
        $crate::finite_automata::Expr::Seq(vec![$($expr.into()),*])
    };
}

#[macro_export]
macro_rules! groups {
    ($($expr:expr),* $(,)?) => {
        $crate::finite_automata::groups(vec![$($expr.into()),*])
    };
}

pub fn groups(groups: Vec<ExprGroup>) -> ExprGroups {
    ExprGroups { groups }
}

pub fn greedy_group<T: Into<ExprGroup>>(expr: T) -> ExprGroup {
    expr.into()
}

pub fn non_greedy_group<T: Into<ExprGroup>>(expr: T) -> ExprGroup {
    let mut group = expr.into();
    group.is_non_greedy = true;
    group
}

impl Display for NFA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Regex State NFA:\n")?;

        for (state_index, state) in self.states.iter().enumerate() {
            f.write_str(&format!("State {}:\n", state_index))?;

            for (transition_set, next_state) in &state.transitions {
                f.write_str(&format!(
                    "  - {:?}: {}\n",
                    transition_set, next_state
                ))?;
            }

            for next_state in &state.epsilon_transitions {
                f.write_str(&format!("  - Epsilon: {}\n", next_state))?;
            }

            if !state.finalizers.is_empty() {
                f.write_str(&format!("  - Finalizers: {:?}\n", state.finalizers))?;
            }

            if !state.non_greedy_finalizers.is_empty() {
                f.write_str(&format!(
                    "  - Non-Greedy Finalizers: {:?}\n",
                    state.non_greedy_finalizers
                ))?;
            }
        }

        Ok(())
    }
}

impl Display for DFA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Regex State DFA:\n")?;

        for (state_index, state) in self.states.iter().enumerate() {
            f.write_str(&format!("State {}:\n", state_index))?;

            for (transition_u8, next_state) in &state.transitions {
                f.write_str(&format!(
                    "  - {} ({:?}): {}\n",
                    transition_u8, transition_u8 as char, next_state
                ))?;
            }

            if !state.finalizers.is_empty() {
                f.write_str(&format!("  - Finalizers: {:?}\n", state.finalizers))?;
            }

            if !state.possible_future_group_ids.is_empty() {
                f.write_str(&format!(
                    "  - Possible Future Group IDs: {:?}\n",
                    state.possible_future_group_ids
                ))?;
            }

            if !state.group_id_to_u8set.is_empty() {
                f.write_str("  - Group ID to U8Set:\n")?;
                for (group_id, u8set) in &state.group_id_to_u8set {
                    f.write_str(&format!("    - Group {}: {}\n", group_id, u8set))?;
                }
            }
        }

        Ok(())
    }
}

impl Display for Regex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dfa)
    }
}

impl NFAState {
    pub fn new() -> NFAState {
        NFAState {
            transitions: Vec::new(),
            epsilon_transitions: Vec::new(),
            finalizers: BTreeSet::new(),
            non_greedy_finalizers: BTreeSet::new(),
        }
    }
}

fn print_memory_usage(label: &str) {
    if let Some(usage) = memory_stats() {
        let physical_mem_mb = usage.physical_mem / 1024 / 1024;
        crate::debug!(5, "Mem: {} MB ({})", physical_mem_mb, label);
    }
}

impl ExprGroups {
    pub fn build(self) -> Regex {
        print_memory_usage("Start of Regex build");
        crate::debug!(3, "Building NFA");
        let start = std::time::Instant::now();
        let nfa = self.build_nfa();
        crate::debug!(4, "Built NFA in {:.2?}", start.elapsed());
        print_memory_usage("After NFA build");
        nfa.print_stats();
        crate::debug!(3, "Converting NFA to DFA");
        let start = std::time::Instant::now();
        let mut dfa = nfa.to_dfa();
        crate::debug!(4, "Converted NFA to DFA in {:.2?}", start.elapsed());
        print_memory_usage("After NFA to DFA conversion");
        crate::debug!(3, "Minimizing DFA");
        let start = std::time::Instant::now();
        dfa.minimize();
        crate::debug!(4, "Minimized DFA in {:.2?}", start.elapsed());
        print_memory_usage("After DFA minimization");
        Regex { dfa }
    }

    fn build_nfa(self) -> NFA {
        let mut nfa = NFA {
            states: vec![NFAState::new()],
            start_state: 0,
        };

        let mut cache: HashMap<usize, (usize, usize)> = HashMap::new();
        for (group, ExprGroup { expr, is_non_greedy }) in self.groups.into_iter().enumerate() {
            let group_start_state = nfa.add_state();
            nfa.add_epsilon_transition(nfa.start_state, group_start_state);
            let end_state =
                Expr::handle_expr_cached(expr, &mut nfa, group_start_state, &mut cache);
            if is_non_greedy {
                nfa.states[end_state].finalizers.insert(group);
                nfa.states[end_state]
                    .non_greedy_finalizers
                    .insert(group);
            } else {
                nfa.states[end_state].finalizers.insert(group);
            }
        }

        nfa
    }
}

impl Expr {
    pub fn build(self) -> Regex {
        ExprGroups {
            groups: vec![ExprGroup {
                expr: self,
                is_non_greedy: false,
            }],
        }
        .build()
    }

    fn handle_expr_cached(
        expr: Expr,
        nfa: &mut NFA,
        current_state: usize,
        cache: &mut HashMap<usize, (usize, usize)>,
    ) -> usize {
        enum FrameState {
            Start,
            Seq { current_state: usize },
            Choice { end_state: usize },
            Shared { key: usize, entry: usize },
            Quantifier { q_type: QuantifierType, entry: usize },
        }

        struct Frame {
            expr: Expr,
            start_state: usize,
            state: FrameState,
        }

        let mut stack = vec![Frame {
            expr,
            start_state: current_state,
            state: FrameState::Start,
        }];
        let mut return_value: Option<usize> = None;

        while let Some(frame) = stack.pop() {
            let Frame {
                mut expr,
                start_state,
                mut state,
            } = frame;
            match state {
                FrameState::Start => match expr {
                    Expr::U8Seq(ref bytes) => {
                        let mut next = start_state;
                        for &b in bytes {
                            let new = nfa.add_state();
                            nfa.add_transition(next, b, new);
                            next = new;
                        }
                        return_value = Some(next);
                    }
                    Expr::U8Class(ref set) => {
                        let new = nfa.add_state();
                        nfa.add_u8set_transition(start_state, set.clone(), new);
                        return_value = Some(new);
                    }
                    Expr::Epsilon => {
                        return_value = Some(start_state);
                    }
                    Expr::Shared(ref inner) => {
                        let key = Arc::as_ptr(inner) as usize;
                        if let Some(&(entry, end)) = cache.get(&key) {
                            nfa.add_epsilon_transition(start_state, entry);
                            return_value = Some(end);
                        } else {
                            let entry = nfa.add_state();
                            nfa.add_epsilon_transition(start_state, entry);
                            state = FrameState::Shared { key, entry };
                            stack.push(Frame {
                                expr: Expr::Epsilon, // Placeholder
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: (**inner).clone(),
                                start_state: entry,
                                state: FrameState::Start,
                            });
                        }
                    }
                    Expr::Seq(mut exprs) => {
                        if exprs.is_empty() {
                            return_value = Some(start_state);
                        } else {
                            exprs.reverse();
                            let first = exprs.pop().unwrap();
                            state = FrameState::Seq {
                                current_state: start_state,
                            };
                            stack.push(Frame {
                                expr: Expr::Seq(exprs),
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: first,
                                start_state,
                                state: FrameState::Start,
                            });
                        }
                    }
                    Expr::Choice(mut exprs) => {
                        let end_state = nfa.add_state();
                        if exprs.is_empty() {
                            return_value = Some(end_state);
                        } else {
                            exprs.reverse();
                            let first = exprs.pop().unwrap();
                            state = FrameState::Choice { end_state };
                            stack.push(Frame {
                                expr: Expr::Choice(exprs),
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: first,
                                start_state,
                                state: FrameState::Start,
                            });
                        }
                    }
                    Expr::Quantifier(inner, q_type) => match q_type {
                        QuantifierType::ZeroOrMore => {
                            let entry = nfa.add_state();
                            nfa.add_epsilon_transition(start_state, entry);
                            state = FrameState::Quantifier {
                                q_type: QuantifierType::ZeroOrMore,
                                entry,
                            };
                            stack.push(Frame {
                                expr: Expr::Epsilon,
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: *inner,
                                start_state: entry,
                                state: FrameState::Start,
                            });
                        }
                        QuantifierType::OneOrMore => {
                            let entry = start_state;
                            state = FrameState::Quantifier {
                                q_type: QuantifierType::OneOrMore,
                                entry,
                            };
                            stack.push(Frame {
                                expr: Expr::Epsilon,
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: *inner,
                                start_state: entry,
                                state: FrameState::Start,
                            });
                        }
                        QuantifierType::ZeroOrOne => {
                            let exit = nfa.add_state();
                            nfa.add_epsilon_transition(start_state, exit);
                            state = FrameState::Quantifier {
                                q_type: QuantifierType::ZeroOrOne,
                                entry: start_state,
                            };
                            stack.push(Frame {
                                expr: Expr::Epsilon,
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: *inner,
                                start_state,
                                state: FrameState::Start,
                            });
                        }
                    },
                },
                FrameState::Seq { current_state } => {
                    let ret = return_value.take().expect("Seq child must return value");
                    let next_state = ret;
                    if let Expr::Seq(mut exprs) = expr {
                        if let Some(next_expr) = exprs.pop() {
                            state = FrameState::Seq {
                                current_state: next_state,
                            };
                            stack.push(Frame {
                                expr: Expr::Seq(exprs),
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: next_expr,
                                start_state: next_state,
                                state: FrameState::Start,
                            });
                        } else {
                            return_value = Some(next_state);
                        }
                    } else {
                        panic!("FrameState::Seq but expr is not Seq")
                    }
                }
                FrameState::Choice { end_state } => {
                    let ret = return_value.take().expect("Choice child must return value");
                    nfa.add_epsilon_transition(ret, end_state);

                    if let Expr::Choice(mut exprs) = expr {
                        if let Some(next_expr) = exprs.pop() {
                            state = FrameState::Choice { end_state };
                            stack.push(Frame {
                                expr: Expr::Choice(exprs),
                                start_state,
                                state,
                            });
                            stack.push(Frame {
                                expr: next_expr,
                                start_state,
                                state: FrameState::Start,
                            });
                        } else {
                            return_value = Some(end_state);
                        }
                    } else {
                        panic!("FrameState::Choice but expr is not Choice")
                    }
                }
                FrameState::Shared { key, entry } => {
                    let ret = return_value.take().expect("Shared child must return value");
                    cache.insert(key, (entry, ret));
                    return_value = Some(ret);
                }
                FrameState::Quantifier { q_type, entry } => {
                    let body_end = return_value
                        .take()
                        .expect("Quantifier child must return value");
                    match q_type {
                        QuantifierType::ZeroOrMore => {
                            let exit = nfa.add_state();
                            nfa.add_epsilon_transition(entry, exit);
                            nfa.add_epsilon_transition(body_end, entry);
                            nfa.add_epsilon_transition(body_end, exit);
                            return_value = Some(exit);
                        }
                        QuantifierType::OneOrMore => {
                            let exit = nfa.add_state();
                            nfa.add_epsilon_transition(body_end, entry);
                            nfa.add_epsilon_transition(body_end, exit);
                            return_value = Some(exit);
                        }
                        QuantifierType::ZeroOrOne => {
                            let exit = nfa.add_state();
                            nfa.add_epsilon_transition(start_state, exit);
                            nfa.add_epsilon_transition(body_end, exit);
                            return_value = Some(exit);
                        }
                    }
                }
            }
        }
        return_value.expect("Stack empty but no return value")
    }

    fn handle_expr(expr: Expr, nfa: &mut NFA, mut current_state: usize) -> usize {
        let mut cache: HashMap<usize, (usize, usize)> = HashMap::new();
        Self::handle_expr_cached(expr, nfa, current_state, &mut cache)
    }
}

impl NFA {
    pub fn add_state(&mut self) -> usize {
        let new_index = self.states.len();
        self.states.push(NFAState::new());
        new_index
    }

    pub fn add_transition(&mut self, from: usize, on_u8: u8, to: usize) {
        self.states[from].transitions.push((U8Set::from_u8(on_u8), to));
    }

    pub fn add_u8set_transition(&mut self, from: usize, on_set: U8Set, to: usize) {
        self.states[from].transitions.push((on_set, to));
    }

    pub fn add_epsilon_transition(&mut self, from: usize, to: usize) {
        self.states[from].epsilon_transitions.push(to);
    }

    pub fn print_stats(&self) {
        let num_states = self.states.len();
        let state_size = std::mem::size_of::<NFAState>();
        let total_base_size = num_states * state_size;

        let mut transitions_capacity_bytes = 0;
        let mut epsilon_capacity_bytes = 0;
        let mut finalizers_est_bytes = 0;
        let mut non_greedy_est_bytes = 0;

        let mut total_transitions_count = 0;
        let mut total_epsilon_count = 0;

        let mut max_group_id = 0;
        let mut compacted_transitions_count = 0;

        for state in &self.states {
            transitions_capacity_bytes += state.transitions.capacity() * std::mem::size_of::<(U8Set, usize)>();
            epsilon_capacity_bytes += state.epsilon_transitions.capacity() * std::mem::size_of::<usize>();

            // Estimate BTreeSet size: ~4 words per element is a reasonable loose approximation for sparse trees
            // This accounts for node pointers and overhead (node header + pointers + data).
            finalizers_est_bytes += state.finalizers.len() * 4 * std::mem::size_of::<usize>();
            non_greedy_est_bytes += state.non_greedy_finalizers.len() * 4 * std::mem::size_of::<usize>();

            total_transitions_count += state.transitions.len();
            total_epsilon_count += state.epsilon_transitions.len();

            if let Some(&m) = state.finalizers.iter().max() {
                if m > max_group_id { max_group_id = m; }
            }
            if let Some(&m) = state.non_greedy_finalizers.iter().max() {
                if m > max_group_id { max_group_id = m; }
            }

            // Count unique targets for compaction estimation
            let mut unique_targets = std::collections::HashSet::new();
            for &(_, target) in &state.transitions {
                unique_targets.insert(target);
            }
            compacted_transitions_count += unique_targets.len();
        }

        let total_estimated_bytes = total_base_size + transitions_capacity_bytes + epsilon_capacity_bytes + finalizers_est_bytes + non_greedy_est_bytes;
        let to_mb = |bytes: usize| bytes as f64 / 1024.0 / 1024.0;

        println!("--- NFA Stats ---");
        println!("States: {}", num_states);
        println!("Estimated Size: {:.2} MB", to_mb(total_estimated_bytes));
        println!("  Base (Vec headers, etc): {:.2} MB", to_mb(total_base_size));
        println!("  Transitions Data: {:.2} MB", to_mb(transitions_capacity_bytes));
        println!("  Epsilon Data: {:.2} MB", to_mb(epsilon_capacity_bytes));
        println!("  Finalizers (est): {:.2} MB", to_mb(finalizers_est_bytes + non_greedy_est_bytes));

        // 1. Finalizer Sets -> Bitsets
        // Cost of Bitset: (max_group_id bits / 64) * 8 bytes per set. Two sets per state.
        let words_per_set = (max_group_id / 64) + 1;
        let bytes_per_set = words_per_set * 8;
        // Overhead of Vec<u64> is 24 bytes.
        let bitset_overhead = 24;
        let total_bitset_cost = num_states * 2 * (bytes_per_set + bitset_overhead);
        let current_finalizer_cost = finalizers_est_bytes + non_greedy_est_bytes;
        let savings_bitsets = (current_finalizer_cost as isize) - (total_bitset_cost as isize);
        println!("  [Savings] Finalizers -> Bitsets: {:.2} MB (current est: {:.2} MB, bitset: {:.2} MB)",
            to_mb(savings_bitsets.max(0) as usize), to_mb(current_finalizer_cost), to_mb(total_bitset_cost));

        // 2. State IDs u32
        // (u8, usize) [16 bytes] -> (u8, u32) [8 bytes]. usize [8 bytes] -> u32 [4 bytes].
        let current_trans_data_used = total_transitions_count * std::mem::size_of::<(U8Set, usize)>();
        let u32_trans_data_used = total_transitions_count * 40; // U8Set is 32 bytes, u32 is 4 bytes, plus padding
        let current_eps_data_used = total_epsilon_count * std::mem::size_of::<usize>();
        let u32_eps_data_used = total_epsilon_count * 4;
        let savings_u32 = (current_trans_data_used + current_eps_data_used) - (u32_trans_data_used + u32_eps_data_used);
        println!("  [Savings] State IDs -> u32: {:.2} MB", to_mb(savings_u32));

        // 3. Compact Transitions
        // Vec<(u8, usize)> [16 bytes] -> Vec<(U8Set, usize)> [48 bytes: 32(set) + 8(usize) + 8(pad)]
        let compact_item_size = 48;
        let compact_total_size = compacted_transitions_count * compact_item_size;
        let savings_compact = (current_trans_data_used as isize) - (compact_total_size as isize);
        println!("  [Savings] Compact Transitions: {:.2} MB (current: {:.2} MB, compacted: {:.2} MB)",
            savings_compact as f64 / 1024.0 / 1024.0, to_mb(current_trans_data_used), to_mb(compact_total_size));

        if num_states > 0 {
            let mut disc = vec![-1i32; num_states];
            let mut low = vec![-1i32; num_states];
            let mut on_stack = vec![false; num_states];
            let mut scc_stack: Vec<usize> = Vec::new();
            let mut time = 0i32;
            let mut sccs: Vec<Vec<usize>> = Vec::new();

            // Stack for iterative DFS: (node, neighbor_iterator)
            let mut dfs_stack: Vec<(usize, std::vec::IntoIter<usize>)> = Vec::new();

            for i in 0..num_states {
                if disc[i] == -1 {
                    // Start DFS from this unvisited node
                    let mut neighbors: Vec<usize> = self.states[i].epsilon_transitions.clone();
                    neighbors
                        .extend(self.states[i].transitions.iter().map(|&(_, target)| target));
                    dfs_stack.push((i, neighbors.into_iter()));

                    while let Some((u, mut neighbors_iter)) = dfs_stack.pop() {
                        // First time we see this node on the current DFS path
                        if disc[u] == -1 {
                            disc[u] = time;
                            low[u] = time;
                            time += 1;
                            scc_stack.push(u);
                            on_stack[u] = true;
                        }

                        let mut found_unvisited_neighbor = false;
                        while let Some(v) = neighbors_iter.next() {
                            if disc[v] == -1 {
                                // Found an unvisited neighbor, descend into it.
                                // First, push the current node `u` back onto the stack with its remaining neighbors.
                                dfs_stack.push((u, neighbors_iter));

                                // Then, push the new node `v` to be processed next.
                                let mut v_neighbors: Vec<usize> =
                                    self.states[v].epsilon_transitions.clone();
                                v_neighbors.extend(
                                    self.states[v].transitions.iter().map(|&(_, target)| target),
                                );
                                dfs_stack.push((v, v_neighbors.into_iter()));

                                found_unvisited_neighbor = true;
                                break;
                            } else if on_stack[v] {
                                // Found a back edge to a node on the current DFS stack.
                                low[u] = low[u].min(disc[v]);
                            }
                        }

                        if found_unvisited_neighbor {
                            // The loop will continue with the new node `v`.
                            continue;
                        }

                        // If we get here, all of `u`'s neighbors have been visited.
                        // We are backtracking from `u`.
                        if low[u] == disc[u] {
                            // `u` is the root of an SCC. Pop the SCC from the scc_stack.
                            let mut scc = Vec::new();
                            while let Some(w) = scc_stack.pop() {
                                on_stack[w] = false;
                                scc.push(w);
                                if u == w {
                                    break;
                                }
                            }
                            sccs.push(scc);
                        }

                        // After finishing with `u`, update its parent's low-link value.
                        // The parent is the node now at the top of the dfs_stack.
                        if let Some((parent, _)) = dfs_stack.last() {
                            low[*parent] = low[*parent].min(low[u]);
                        }
                    }
                }
            }

            let num_sccs = sccs.len();
            let non_trivial_sccs: Vec<_> = sccs.iter().filter(|scc| scc.len() > 1).collect();
            let num_non_trivial_sccs = non_trivial_sccs.len();
            let total_states_in_non_trivial_sccs: usize =
                non_trivial_sccs.iter().map(|scc| scc.len()).sum();
            let avg_non_trivial_scc_size = if num_non_trivial_sccs > 0 {
                total_states_in_non_trivial_sccs as f64 / num_non_trivial_sccs as f64
            } else {
                0.0
            };
            println!("  SCCs: {}, Non-trivial (size>1): {} ({} states, avg size {:.2})",
                num_sccs, num_non_trivial_sccs, total_states_in_non_trivial_sccs, avg_non_trivial_scc_size);
        }
        println!("-----------------");
    }

    pub fn to_dfa(self) -> DFA {
        let start_time = std::time::Instant::now();
        let mut dfa_states: Vec<DFAState> = Vec::new();
        let mut dfa_state_map: HashMap<FrozenSet<usize>, usize> = HashMap::new();
        let mut worklist: Vec<FrozenSet<usize>> = Vec::new();

        // Shared buffers for on-the-fly closure computation
        let num_nfa_states = self.states.len();
        let mut stack = Vec::with_capacity(1024);
        let mut closure_bitset = BitSet::new(num_nfa_states);

        // Compute start state closure
        stack.push(self.start_state);
        closure_bitset.insert(self.start_state);

        while let Some(u) = stack.pop() {
            for &v in &self.states[u].epsilon_transitions {
                if closure_bitset.insert(v) {
                    stack.push(v);
                }
            }
        }

        let start_closure_vec: Vec<usize> = closure_bitset.iter().collect();
        let start_state_set = FrozenSet::new_unchecked(start_closure_vec);
        dfa_state_map.insert(start_state_set.clone(), 0);
        worklist.push(start_state_set.clone());

        let mut finalizers = BTreeSet::new();
        let mut non_greedy_finalizers = BTreeSet::new();
        for &state in start_state_set.iter() {
            finalizers.extend(self.states[state].finalizers.iter().cloned());
            non_greedy_finalizers.extend(self.states[state].non_greedy_finalizers.iter().cloned());
        }

        dfa_states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers,
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        let mut max_subset_size = 0;
        let mut next_log_threshold = 20_000;
        
        // Reuseable structures for DFA construction
        let mut transition_targets: Vec<Vec<usize>> = vec![Vec::with_capacity(16); 256];
        let mut used_inputs: Vec<usize> = Vec::with_capacity(256);
        let mut seen_input = [false; 256];

        while let Some(current_set) = worklist.pop() {
            let current_subset_len = current_set.len();
            if current_subset_len > max_subset_size {
                max_subset_size = current_subset_len;
            }
            if dfa_states.len() >= next_log_threshold {
                crate::debug!(6, "DFA progress: {} states, worklist {}, subset size {} (max {}), elapsed {:.2?}", dfa_states.len(), worklist.len(), current_subset_len, max_subset_size, start_time.elapsed());
                next_log_threshold += 20_000;
            }

            let current_dfa_state = *dfa_state_map
                .get(&current_set)
                .expect("DFA state set not found in map");

            for &state in current_set.iter() {
                for (u8set, next_state) in &self.states[state].transitions {
                    // Optimized U8Set iter
                    for b in u8set.iter() { 
                        let idx = b as usize;
                        if !seen_input[idx] {
                            seen_input[idx] = true;
                            used_inputs.push(idx);
                        }
                        transition_targets[idx].push(*next_state);
                    }
                }
            }

            // Local cache for this DFA state to map (Target Set) -> (Next DFA State ID)
            // The key is sorted vector of targets.
            let mut local_cache: HashMap<Vec<usize>, usize> = HashMap::new();

            for &idx in &used_inputs {
                let target_list = &mut transition_targets[idx];
                
                // Often multiple inputs lead to the exact same list of NFA states.
                // We sort/dedup purely for the local cache key.
                // Note: target_list comes from pushes, so it's not sorted.
                target_list.sort_unstable();
                target_list.dedup();

                if let Some(&next_state_idx) = local_cache.get(target_list) {
                    dfa_states[current_dfa_state].transitions.insert(idx as u8, next_state_idx);
                    continue;
                }

                // Compute closure
                closure_bitset.clear();
                for &next_state in target_list.iter() {
                    if closure_bitset.insert(next_state) {
                        stack.push(next_state);
                    }
                }

                while let Some(u) = stack.pop() {
                    for &v in &self.states[u].epsilon_transitions {
                        if closure_bitset.insert(v) {
                            stack.push(v);
                        }
                    }
                }

                // BitSet iter returns sorted elements
                let closure_vec: Vec<usize> = closure_bitset.iter().collect();
                let frozen_closure = FrozenSet::new_unchecked(closure_vec);
                
                // Get/Create DFA state
                let next_dfa_state =
                    if let Some(&existing_state) = dfa_state_map.get(&frozen_closure) {
                        existing_state
                    } else {
                        let new_state_index = dfa_states.len();
                        dfa_state_map.insert(frozen_closure.clone(), new_state_index);
                        worklist.push(frozen_closure.clone());

                        let mut new_finalizers = BTreeSet::new();
                        let mut new_non_greedy_finalizers = BTreeSet::new();
                        for &state in frozen_closure.iter() {
                            new_finalizers.extend(self.states[state].finalizers.iter().cloned());
                            new_non_greedy_finalizers
                                .extend(self.states[state].non_greedy_finalizers.iter().cloned());
                        }

                        dfa_states.push(DFAState {
                            transitions: CharTransitions::new(),
                            finalizers: new_finalizers,
                            possible_future_group_ids: BTreeSet::new(),
                            group_id_to_u8set: BTreeMap::new(),
                        });

                        new_state_index
                    };

                // Cache using the vector we already sorted
                local_cache.insert(target_list.clone(), next_dfa_state);
                dfa_states[current_dfa_state].transitions.insert(idx as u8, next_dfa_state);
            }

            for &idx in &used_inputs {
                 seen_input[idx] = false;
                 transition_targets[idx].clear();
            }
            used_inputs.clear();
        }

        crate::debug!(5, "DFA main loop complete. Total states: {}, Max subset size: {}, Time: {:.2?}", dfa_states.len(), max_subset_size, start_time.elapsed());

        let mut dfa = DFA {
            states: dfa_states,
            start_state: 0,
            non_greedy_finalizers: BTreeSet::new(),
        };

        for state in &self.states {
            dfa.non_greedy_finalizers.extend(state.non_greedy_finalizers.iter().cloned());
        }

        let meta_start = std::time::Instant::now();
        dfa.recompute_metadata();
        crate::debug!(5, "Computed DFA metadata in {:.2?}", meta_start.elapsed());

        dfa
    }
}

// ... (rest of finite_automata.rs is unchanged)
