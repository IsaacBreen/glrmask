use crate::datastructures::char_transitions::CharTransitions;
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
    transitions: Vec<(u8, usize)>,
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
        for &(byte, target) in &self.transitions {
            grouped.entry(byte).or_default().push(target);
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

                let mut transitions: Vec<(u8, usize)> = Vec::new();
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
                                transitions.push((byte, target));
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

            for &(transition_u8, next_state) in &state.transitions {
                f.write_str(&format!(
                    "  - '{}': {}\n",
                    transition_u8 as char, next_state
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
                        for b in set.iter() {
                            nfa.add_transition(start_state, b, new);
                        }
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
        self.states[from].transitions.push((on_u8, to));
    }

    pub fn add_epsilon_transition(&mut self, from: usize, to: usize) {
        self.states[from].epsilon_transitions.push(to);
    }

    /// Epsilon-closure of a set of NFA states.
    fn epsilon_closure_set<I>(&self, states: I) -> BTreeSet<usize>
    where
        I: IntoIterator<Item = usize>,
    {
        let mut closure = BTreeSet::new();
        let mut stack = Vec::new();

        for s in states {
            if closure.insert(s) {
                stack.push(s);
            }
        }

        while let Some(state) = stack.pop() {
            for &next in &self.states[state].epsilon_transitions {
                if closure.insert(next) {
                    stack.push(next);
                }
            }
        }

        closure
    }

    fn epsilon_closure(&self, state: usize) -> BTreeSet<usize> {
        self.epsilon_closure_set(std::iter::once(state))
    }

    fn compute_epsilon_closures(&self) -> Vec<Vec<usize>> {
        (0..self.states.len())
            .map(|state| self.epsilon_closure(state).into_iter().collect())
            .collect()
    }

    pub fn to_dfa(self) -> DFA {
        let start_time = std::time::Instant::now();
        let mut dfa_states: Vec<DFAState> = Vec::new();
        let mut dfa_state_map: HashMap<FrozenSet<usize>, usize> = HashMap::new();
        let mut worklist: Vec<FrozenSet<usize>> = Vec::new();

        let closure_start = std::time::Instant::now();
        crate::debug!(5, "Computing epsilon closures ({} states)...", self.states.len());
        let epsilon_closures = self.compute_epsilon_closures();
        crate::debug!(5, "Computed epsilon closures in {:.2?}", closure_start.elapsed());

        let start_closure = &epsilon_closures[self.start_state];
        let start_state_set = FrozenSet::from_iter(start_closure.iter().cloned());
        worklist.push(start_state_set.clone());
        dfa_state_map.insert(start_state_set.clone(), 0);

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
        let mut transition_targets: Vec<Vec<usize>> = vec![Vec::with_capacity(16); 256];
        let mut used_inputs: Vec<u8> = Vec::with_capacity(256);
        let mut seen_input: [bool; 256] = [false; 256];

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
                for &(input, next_state) in &self.states[state].transitions {
                    let idx = input as usize;
                    if !seen_input[idx] {
                        seen_input[idx] = true;
                        used_inputs.push(input);
                    }
                    transition_targets[idx].push(next_state);
                }
            }

            for &input_u8 in &used_inputs {
                let next_states = &transition_targets[input_u8 as usize];
                if next_states.is_empty() {
                    continue;
                }

                let mut closure = BTreeSet::new();
                for &next_state in next_states {
                    for &s in &epsilon_closures[next_state] {
                        closure.insert(s);
                    }
                }
                let frozen_closure: FrozenSet<usize> = FrozenSet::from(closure);

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

                dfa_states[current_dfa_state]
                    .transitions
                    .insert(input_u8, next_dfa_state);
            }

            for &input in &used_inputs {
                let idx = input as usize;
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

impl DFA {
    pub fn compute_possible_future_group_ids(&mut self) {
        for state in &mut self.states {
            state.possible_future_group_ids = BTreeSet::new();
        }

        let num_states = self.states.len();

        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        for (idx, state) in self.states.iter().enumerate() {
            for &target in state.transitions.values() {
                predecessors[target].push(idx);
            }
        }

        let max_group_id = self
            .states
            .iter()
            .flat_map(|s| s.finalizers.last())
            .max()
            .copied()
            .unwrap_or(0);

        let u64_per_state = (max_group_id / 64) + 1;
        let mut future_bits: Vec<u64> = vec![0; num_states * u64_per_state];

        let mut worklist: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut in_worklist: Vec<bool> = vec![false; num_states];

        for target_idx in 0..num_states {
            if !self.states[target_idx].finalizers.is_empty() {
                for &pred_idx in &predecessors[target_idx] {
                    let pred_offset = pred_idx * u64_per_state;
                    let mut changed = false;

                    for &gid in &self.states[target_idx].finalizers {
                        let word_idx = gid / 64;
                        let bit_mask = 1u64 << (gid % 64);
                        if (future_bits[pred_offset + word_idx] & bit_mask) == 0 {
                            future_bits[pred_offset + word_idx] |= bit_mask;
                            changed = true;
                        }
                    }

                    if changed && !in_worklist[pred_idx] {
                        worklist.push_back(pred_idx);
                        in_worklist[pred_idx] = true;
                    }
                }
            }
        }

        while let Some(idx) = worklist.pop_front() {
            in_worklist[idx] = false;

            let idx_offset = idx * u64_per_state;

            for &pred_idx in &predecessors[idx] {
                let pred_offset = pred_idx * u64_per_state;
                let mut changed = false;

                for w in 0..u64_per_state {
                    let incoming = future_bits[idx_offset + w];
                    if incoming != 0 {
                        let old_val = future_bits[pred_offset + w];
                        let new_val = old_val | incoming;
                        if old_val != new_val {
                            future_bits[pred_offset + w] = new_val;
                            changed = true;
                        }
                    }
                }

                if changed && !in_worklist[pred_idx] {
                    worklist.push_back(pred_idx);
                    in_worklist[pred_idx] = true;
                }
            }
        }

        for (idx, state) in self.states.iter_mut().enumerate() {
            let offset = idx * u64_per_state;
            let mut set = BTreeSet::new();

            for w in 0..u64_per_state {
                let mut word = future_bits[offset + w];
                if word != 0 {
                    let base_gid = w * 64;
                    while word != 0 {
                        let trailing = word.trailing_zeros();
                        set.insert(base_gid + trailing as usize);
                        word &= !(1u64 << trailing);
                    }
                }
            }
            state.possible_future_group_ids = set;
        }
    }

    pub fn compute_group_id_to_u8set(&mut self) {
        let num_states = self.states.len();
        let mut all_maps: Vec<BTreeMap<GroupID, U8Set>> = Vec::with_capacity(num_states);

        for state in &self.states {
            let mut group_id_to_u8set: BTreeMap<GroupID, U8Set> = BTreeMap::new();

            let mut target_to_inputs: HashMap<usize, U8Set> = HashMap::new();
            for (input_u8, &next_state_index) in &state.transitions {
                target_to_inputs
                    .entry(next_state_index)
                    .and_modify(|set| {
                        set.insert(input_u8);
                    })
                    .or_insert_with(|| U8Set::from_u8(input_u8));
            }

            for (next_state_index, inputs) in target_to_inputs {
                let next_state = &self.states[next_state_index];

                let chain = next_state
                    .possible_future_group_ids
                    .iter()
                    .chain(next_state.finalizers.iter());

                for &group_id in chain {
                    group_id_to_u8set
                        .entry(group_id)
                        .or_insert_with(U8Set::none)
                        .update(&inputs);
                }
            }
            all_maps.push(group_id_to_u8set);
        }

        for (i, map) in all_maps.into_iter().enumerate() {
            self.states[i].group_id_to_u8set = map;
        }
    }

    fn recompute_metadata(&mut self) {
        self.compute_possible_future_group_ids();
        self.compute_group_id_to_u8set();
    }

    fn remove_unreachable_states(&mut self) {
        let mut reachable = vec![false; self.states.len()];
        let mut queue = vec![self.start_state];
        reachable[self.start_state] = true;

        while let Some(state) = queue.pop() {
            for &next_state in self.states[state].transitions.values() {
                if !reachable[next_state] {
                    reachable[next_state] = true;
                    queue.push(next_state);
                }
            }
        }

        let mut state_mapping = vec![0; self.states.len()];
        let mut new_index = 0;
        for (old_index, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                state_mapping[old_index] = new_index;
                new_index += 1;
            }
        }

        let mut new_states = Vec::new();
        for (old_index, state) in self.states.iter().enumerate() {
            if reachable[old_index] {
                let mut new_state = state.clone();
                new_state.transitions = new_state
                    .transitions
                    .iter()
                    .map(|(u8, &next)| (u8, state_mapping[next]))
                    .collect();
                new_states.push(new_state);
            }
        }

        self.states = new_states;
        self.start_state = 0;
    }

    fn minimize(&mut self) {
        if self.states.is_empty() {
            return;
        }

        const MAX_MINIMIZATION_STATES: usize = 10_000;
        if self.states.len() > MAX_MINIMIZATION_STATES {
            return;
        }

        self.remove_unreachable_states();

        let mut partitions_map: BTreeMap<BTreeSet<GroupID>, BTreeSet<usize>> = BTreeMap::new();
        for (state_idx, state) in self.states.iter().enumerate() {
            partitions_map
                .entry(state.finalizers.clone())
                .or_default()
                .insert(state_idx);
        }
        let mut partition_list: Vec<BTreeSet<usize>> = partitions_map.into_values().collect();

        let mut state_to_partition = vec![0; self.states.len()];
        for (part_idx, partition) in partition_list.iter().enumerate() {
            for &state_idx in partition {
                state_to_partition[state_idx] = part_idx;
            }
        }

        loop {
            let mut changed = false;
            let mut new_partition_list: Vec<BTreeSet<usize>> = Vec::new();

            for partition in &partition_list {
                if partition.len() <= 1 {
                    new_partition_list.push(partition.clone());
                    continue;
                }

                let mut refined_partitions: BTreeMap<Vec<(u8, usize)>, BTreeSet<usize>> =
                    BTreeMap::new();
                for &state_idx in partition {
                    let signature: Vec<(u8, usize)> = self.states[state_idx]
                        .transitions
                        .iter()
                        .map(|(input, &next_state)| (input, state_to_partition[next_state]))
                        .collect();
                    refined_partitions.entry(signature).or_default().insert(state_idx);
                }

                if refined_partitions.len() > 1 {
                    changed = true;
                }
                new_partition_list.extend(refined_partitions.into_values());
            }

            partition_list = new_partition_list;

            if !changed {
                break;
            }

            for (part_idx, partition) in partition_list.iter().enumerate() {
                for &state_idx in partition {
                    state_to_partition[state_idx] = part_idx;
                }
            }
        }

        let (state_mapping, new_states) = self.rebuild_from_partitions(partition_list);

        self.states = new_states;
        self.start_state = state_mapping[self.start_state];

        self.recompute_metadata();
    }

    fn rebuild_from_partitions(
        &self,
        mut partition_list: Vec<BTreeSet<usize>>,
    ) -> (Vec<usize>, Vec<DFAState>) {
        let mut state_mapping = vec![0; self.states.len()];

        if let Some(start_part_idx) = partition_list
            .iter()
            .position(|p| p.contains(&self.start_state))
        {
            partition_list.swap(0, start_part_idx);
        }

        for (new_idx, partition) in partition_list.iter().enumerate() {
            for &old_idx in partition {
                state_mapping[old_idx] = new_idx;
            }
        }

        let mut new_states = Vec::with_capacity(partition_list.len());
        for partition in &partition_list {
            let representative_old_idx = *partition.iter().next().unwrap();
            let mut new_state = self.states[representative_old_idx].clone();

            new_state.transitions = new_state
                .transitions
                .iter()
                .map(|(u8, &old_next_idx)| (u8, state_mapping[old_next_idx]))
                .collect();

            new_states.push(new_state);
        }

        (state_mapping, new_states)
    }
}

fn should_terminate_early(
    possible_future_group_ids: &BTreeSet<GroupID>,
    non_greedy_finalizers: &BTreeSet<GroupID>,
    matched_groups: &BTreeSet<GroupID>,
) -> bool {
    possible_future_group_ids
        .iter()
        .all(|group_id| non_greedy_finalizers.contains(group_id) && matched_groups.contains(group_id))
}

impl RegexState<'_> {
    pub fn execute(&mut self, text: &[u8]) -> Vec<Match> {
        let mut all_matches = Vec::new();
        if self.done {
            self.position += text.len();
            return all_matches;
        }
        let dfa = &self.regex.dfa;
        let mut local_position = 0;
        while local_position < text.len() {
            let state_data = &dfa.states[self.current_state];
            let next_u8 = text[local_position];
            if let Some(&next_state) = state_data.transitions.get(next_u8) {
                self.current_state = next_state;
                local_position += 1;
                for &group_id in &dfa.states[self.current_state].finalizers {
                    all_matches.push(Match {
                        group_id,
                        position: self.position + local_position,
                    });

                    if dfa.non_greedy_finalizers.contains(&group_id) {
                        self.matches
                            .entry(group_id)
                            .or_insert(self.position + local_position);
                    } else {
                        self.matches
                            .insert(group_id, self.position + local_position);
                    }
                }

                let matched: BTreeSet<GroupID> = self.matches.keys().cloned().collect();
                let should_terminate = should_terminate_early(
                    &dfa.states[self.current_state].possible_future_group_ids,
                    &dfa.non_greedy_finalizers,
                    &matched,
                );

                if should_terminate {
                    self.position += text.len();
                    self.done = true;
                    return all_matches;
                }
            } else {
                self.position += text.len();
                self.done = true;
                return all_matches;
            }
        }
        self.position += text.len();
        if dfa.states[self.current_state].transitions.is_empty() {
            self.done = true;
        }
        all_matches
    }

    pub fn end(&mut self) {
        self.done = true;
    }

    pub fn ended(&self) -> bool {
        self.done
    }

    pub fn reset(&mut self) {
        self.current_state = self.regex.dfa.start_state;
        self.matches.clear();
        self.position = 0;
        self.done = false;
    }

    pub fn greedy_find_all(&mut self, text: &[u8], terminate: bool) -> Vec<Match> {
        let mut matches: Vec<Match> = Vec::new();
        let start_position = self.position;
        let mut local_position = 0;
        self.position = 0;
        loop {
            self.execute(&text[local_position..]);
            if self.ended() {
                if let Some(m) = self.get_greedy_match() {
                    local_position += m.position;
                    matches.push(m);
                    self.reset();
                } else {
                    self.position = start_position + local_position;
                    return matches;
                }
            } else {
                if terminate {
                    if let Some(m) = self.get_greedy_match() {
                        matches.push(m);
                    }
                    self.end();
                    return matches;
                }
                self.position = start_position + local_position;
                return matches;
            }
        }
    }

    pub fn get_greedy_match(&self) -> Option<Match> {
        self.matches
            .iter()
            .filter(|(_, &pos)| pos > 0)
            .max_by(|(&g1, &p1), (&g2, &p2)| p1
                .cmp(&p2)
                .then_with(|| g2.cmp(&g1)))
            .map(|(&group_id, &position)| Match { group_id, position })
    }

    pub fn final_state_report(&self) -> FinalStateReport {
        FinalStateReport {
            position: self.position,
            matches: self.matches.clone(),
        }
    }

    pub fn get_u8set(&self) -> U8Set {
        let dfa = &self.regex.dfa;
        let state_data = &dfa.states[self.current_state];
        state_data.transitions.keys_as_u8set()
    }

    pub fn get_terminal_u8set(&self) -> U8Set {
        let mut u8set = U8Set::none();
        let dfa = &self.regex.dfa;
        let state_data = &dfa.states[self.current_state];
        for (value, &i_next_state) in &state_data.transitions {
            if !dfa.states[i_next_state].finalizers.is_empty() {
                u8set.insert(value);
            }
        }
        u8set
    }

    pub fn matches(&self) -> Option<bool> {
        if !self.matches.is_empty() {
            Some(true)
        } else if self.done {
            Some(false)
        } else {
            None
        }
    }

    pub fn definitely_matches(&self) -> bool {
        self.matches().unwrap_or(false)
    }

    pub fn could_match(&self) -> bool {
        self.matches().unwrap_or(true)
    }

    pub fn fully_matches(&self) -> Option<bool> {
        if let Some(max_position) = self.matches.values().max() {
            Some(*max_position == self.position)
        } else if self.done {
            Some(false)
        } else {
            None
        }
    }

    pub fn definitely_fully_matches(&self) -> bool {
        self.fully_matches().unwrap_or(false)
    }

    pub fn could_fully_match(&self) -> bool {
        self.fully_matches().unwrap_or(true)
    }

    pub fn fully_matches_here(&self) -> bool {
        self.definitely_fully_matches()
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn failed(&self) -> bool {
        !self.could_match()
    }

    pub fn clear_matches(&mut self) {
        self.matches.clear();
    }

    pub fn possible_future_group_ids(&self) -> BTreeSet<GroupID> {
        let state = &self.regex.dfa.states[self.current_state];
        state.possible_future_group_ids.clone()
    }

    pub fn get_u8set_for_group(&self, group_id: GroupID) -> U8Set {
        let state = &self.regex.dfa.states[self.current_state];
        state
            .group_id_to_u8set
            .get(&group_id)
            .cloned()
            .unwrap_or_else(U8Set::none)
    }
}

impl Regex {
    pub fn num_groups(&self) -> usize {
        let mut max_gid: Option<GroupID> = None;
        for s in &self.dfa.states {
            if let Some(m) = s.finalizers.iter().max() {
                max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
            }
        }
        max_gid.map(|m| m + 1).unwrap_or(0)
    }

    pub fn execute_from_state2(&self, text: &[u8], state: usize) -> ExecutionResult {
        self.execute_from_state_fast(text, state)
    }

    pub fn execute_from_state_fast(&self, text: &[u8], state: usize) -> ExecutionResult {
        let dfa = &self.dfa;
        let mut all_matches: Vec<Match> = Vec::new();

        let mut current_state = state;
        let mut matched_groups: BTreeSet<GroupID> = dfa.states[state].finalizers.clone();

        if dfa.states[state].transitions.is_empty() {
            return ExecutionResult {
                matches: all_matches,
                end_state: None,
            };
        }

        let text_len = text.len();
        let mut local_position = 0usize;

        while local_position < text_len {
            let state_data = &dfa.states[current_state];
            let next_byte = text[local_position];

            let Some(&next_state) = state_data.transitions.get(next_byte) else {
                return ExecutionResult {
                    matches: all_matches,
                    end_state: None,
                };
            };

            current_state = next_state;
            local_position += 1;

            let state_data = &dfa.states[current_state];

            if !state_data.finalizers.is_empty() {
                for &group_id in &state_data.finalizers {
                    all_matches.push(Match {
                        group_id,
                        position: local_position,
                    });
                    matched_groups.insert(group_id);
                }
            }

            if should_terminate_early(
                &state_data.possible_future_group_ids,
                &dfa.non_greedy_finalizers,
                &matched_groups,
            ) {
                return ExecutionResult {
                    matches: all_matches,
                    end_state: None,
                };
            }
        }

        let end_state = if dfa.states[current_state].transitions.is_empty() {
            None
        } else {
            Some(current_state)
        };

        ExecutionResult {
            matches: all_matches,
            end_state,
        }
    }

    pub fn init_to_state(&self, state: usize) -> RegexState {
        let done = self.dfa.states[state].transitions.is_empty();
        let matches = self.dfa.states[state]
            .finalizers
            .iter()
            .map(|&group_id| (group_id, 0))
            .collect();
        RegexState {
            regex: self,
            position: 0,
            current_state: state,
            matches,
            done,
        }
    }

    pub fn init(&self) -> RegexState {
        self.init_to_state(self.dfa.start_state)
    }

    pub fn get_next_state(&self, current_state: usize, byte: u8) -> Option<usize> {
        self.dfa.states[current_state]
            .transitions
            .get(byte)
            .copied()
    }

    pub fn find(&self, text: &[u8]) -> Option<(GroupID, usize)> {
        let mut regex_state = self.init();
        regex_state.execute(text);
        regex_state
            .matches
            .iter()
            .next()
            .map(|(&group_id, &position)| (group_id, position))
    }

    pub fn matches(&self, text: &[u8]) -> Option<bool> {
        let mut regex_state = self.init();
        regex_state.execute(text);
        regex_state.matches()
    }

    pub fn definitely_matches(&self, text: &[u8]) -> bool {
        self.matches(text).unwrap_or(false)
    }

    pub fn could_match(&self, text: &[u8]) -> bool {
        self.matches(text).unwrap_or(true)
    }

    pub fn fully_matches(&self, text: &[u8]) -> Option<bool> {
        let mut regex_state = self.init();
        regex_state.execute(text);
        regex_state.fully_matches()
    }

    pub fn definitely_fully_matches(&self, text: &[u8]) -> bool {
        self.fully_matches(text).unwrap_or(false)
    }

    pub fn could_fully_match(&self, text: &[u8]) -> bool {
        self.fully_matches(text).unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{choice, seq};

    #[test]
    fn test_literal() {
        let expr: Expr = eat_u8(b'a');
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(!regex.could_match(b"b"));

        assert!(!regex.definitely_matches(b""));
        assert!(regex.could_match(b""));
        assert!(regex.definitely_matches(b"ab"));
        assert!(regex.definitely_matches(b"aa"));
    }

    #[test]
    fn test_quantifier() {
        let expr = rep(eat_u8(b'a'));
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b""));
        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"aaaa"));
        assert!(regex.could_match(b"b"));

        let mut state = regex.init();
        state.execute(b"aa");
        assert_eq!(state.matches, BTreeMap::from([(0, 2)]));
        assert!(!state.done());
    }

    #[test]
    fn test_choice() {
        let expr = choice![eat_u8(b'a'), eat_u8(b'b')];
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"b"));
        assert!(!regex.could_match(b"c"));
    }

    #[test]
    fn test_seq() {
        let expr = seq![eat_u8(b'a'), eat_u8(b'b')];
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.could_match(b"a"));
        assert!(!regex.definitely_matches(b"a"));
        assert!(!regex.could_match(b"b"));
        assert!(regex.definitely_matches(b"ab"));
        assert!(regex.definitely_matches(b"abab"));
        assert!(!regex.could_match(b"c"));
    }

    #[test]
    fn test_opt() {
        let expr = opt(eat_u8(b'a'));
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b""));
        assert!(regex.definitely_fully_matches(b"a"));
        assert!(!regex.could_fully_match(b"aa"));
        assert!(regex.could_match(b"b"));
    }

    #[test]
    fn test_0() {
        let expr = eat_u8(0);
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"\0"));
        assert!(!regex.could_match(b"1"));
    }

    #[test]
    fn test_epsilon() {
        let expr = eps();
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b""));
        assert!(regex.definitely_matches(b"a"));
        assert!(!regex.definitely_fully_matches(b"a"));
    }

    #[test]
    fn test_u8seq() {
        let expr = Expr::U8Seq(vec![b'a', b'b']);
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"ab"));
        assert!(regex.could_match(b"a"));
        assert!(!regex.could_match(b"b"));
        assert!(!regex.could_match(b"ba"));
    }
}

#[cfg(test)]
mod complex_tests {
    use super::*;

    #[test]
    fn test_nested_quantifiers() {
        let expr = rep1(rep(eat_u8(b'a')));
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"aa"));
        assert!(regex.definitely_fully_matches(b"aaa"));
        assert!(regex.definitely_fully_matches(b""));
    }

    #[test]
    fn test_complex_choice() {
        let expr = choice![
            seq![eat_u8(b'a'), rep1(eat_u8(b'b'))],
            eat_u8(b'c'),
        ];
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"ab"));
        assert!(regex.definitely_fully_matches(b"abb"));
        assert!(regex.definitely_fully_matches(b"c"));
        assert!(regex.could_match(b"a"));
        assert!(!regex.definitely_matches(b"a"));
        assert!(!regex.could_match(b"b"));
        assert!(regex.definitely_matches(b"cc"));
        assert_eq!(regex.fully_matches(b"cc"), Some(false));
    }

    #[test]
    fn test_complex_seq_with_quantifiers() {
        let expr = seq![
            rep(eat_u8(b'a')),
            eat_u8(b'b'),
            rep1(eat_u8(b'c')),
        ];
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"bc"));
        assert!(regex.definitely_fully_matches(b"bcc"));
        assert!(regex.definitely_fully_matches(b"abcc"));
        assert!(regex.definitely_fully_matches(b"aaabccc"));
        assert!(regex.could_match(b"a"));
        assert!(regex.could_match(b"b"));
        assert!(!regex.could_match(b"c"));
    }

    #[test]
    fn test_complex_pattern() {
        let expr = seq![
            rep(choice![eat_u8(b'a'), eat_u8(b'b')]),
            eat_u8(b'c'),
            rep1(choice![eat_u8(b'd'), eat_u8(b'e')]),
        ];
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"cd"));
        assert!(regex.definitely_fully_matches(b"ce"));
        assert!(regex.definitely_fully_matches(b"cde"));
        assert!(regex.definitely_fully_matches(b"aced"));
        assert!(regex.definitely_fully_matches(b"bacde"));
        assert!(regex.could_match(b"a"));
        assert!(!regex.definitely_matches(b"a"));
        assert!(!regex.definitely_matches(b"b"));
        assert!(regex.could_match(b"c"));
        assert!(!regex.definitely_matches(b"c"));
        assert!(!regex.could_match(b"d"));
    }

    #[test]
    fn test_complex_epsilon() {
        let expr = groups![
            eps(),
            rep1(eat_u8(b'a')),
        ];
        let regex = expr.build();
        let mut state = regex.init();
        dbg!(&regex);
        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 0), (1, 1)]));
    }
}

#[cfg(test)]
mod even_more_complex_tests {
    use super::*;

    #[test]
    fn test_overlapping_u8_classes() {
        let expr = seq![
            choice![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'c')],
            choice![eat_u8(b'b'), eat_u8(b'c'), eat_u8(b'd')],
        ];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"bc"));
        assert!(regex.definitely_fully_matches(b"cb"));
        assert!(regex.definitely_fully_matches(b"ab"));
        assert!(regex.definitely_fully_matches(b"cd"));
    }

    #[test]
    fn test_nested_seqs_with_quantifiers() {
        let expr = seq![
            rep(seq![eat_u8(b'a'), rep1(eat_u8(b'b'))]),
            eat_u8(b'c'),
        ];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"c"));
        assert!(regex.definitely_fully_matches(b"abc"));
        assert!(regex.definitely_fully_matches(b"abbc"));
        assert!(regex.definitely_fully_matches(b"ababbabc"));
        assert!(!regex.could_match(b"ac"));
    }

    #[test]
    fn test_choice_with_empty_option() {
        let expr = choice![eat_u8(b'a'), seq![]];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b""));
    }

    #[test]
    fn test_complex_pattern_with_overlapping_quantifiers() {
        let expr = seq![
            rep(eat_u8(b'a')),
            rep1(eat_u8(b'a')),
        ];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"aa"));
        assert!(regex.could_match(b""));
        assert!(regex.could_fully_match(b""));
        assert!(!regex.could_match(b"b"));
    }

    #[test]
    fn test_matching_at_different_positions() {
        let expr: Expr = eat_u8(b'a');
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(!regex.could_match(b"ba"));
        assert!(regex.definitely_matches(b"ab"));
        assert!(!regex.definitely_fully_matches(b"ab"));
        assert!(!regex.could_match(b"bab"));
        assert!(!regex.could_match(b"b"));
    }

    #[test]
    fn test_lots_of_words() {
        let words = [
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "async",
            "await",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
        ];

        let expr = Expr::Choice(
            words
                .iter()
                .map(|word| {
                    Expr::Seq(word.bytes().map(|c| Expr::U8Seq(vec![c])).collect())
                })
                .collect(),
        );
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"False"));
        assert!(regex.definitely_fully_matches(b"None"));
        assert!(regex.definitely_fully_matches(b"True"));
        assert!(regex.definitely_fully_matches(b"and"));
        assert!(regex.definitely_fully_matches(b"as"));
        assert!(regex.definitely_fully_matches(b"assert"));
    }

    #[test]
    fn test_multiple_finalizers() {
        let expr = groups![
            eat_u8(b'a'),
            seq![eat_u8(b'a'), eat_u8(b'a')],
        ];

        let regex = expr.build();
        dbg!(&regex);

        let mut state = regex.init();

        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 1)]));

        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 1), (1, 2)]));
    }

    #[test]
    fn test_multiple_finalizers_greedy() {
        let expr = groups![
            rep(eat_u8(b'a')),
            eat_u8(b'a'),
        ];

        let regex = expr.build();
        dbg!(&regex);

        let mut state = regex.init();

        state.execute(b"aa");
        assert_eq!(state.matches, BTreeMap::from([(0, 2), (1, 1)]));
    }

    #[test]
    fn test_non_greedy_matching() {
        let expr = groups![
            non_greedy_group(rep(eat_u8(b'a'))),
            eat_u8(b'a'),
        ];

        let regex = expr.build();

        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        assert_eq!(regex_state.matches.get(&0), Some(&0));
        assert_eq!(regex_state.matches.get(&1), Some(&1));
    }

    #[test]
    fn test_greedy_matching() {
        let expr = groups![
            rep(eat_u8(b'a')),
            eat_u8(b'a'),
        ];

        let regex = expr.build();

        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        assert_eq!(regex_state.matches.get(&0), Some(&3));
        assert_eq!(regex_state.matches.get(&1), Some(&1));
    }

    #[test]
    fn test_triple_quoted_string() {
        let non_greedy_expr = groups![
            non_greedy_group(seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())),
                Expr::U8Seq(b"\"\"\"".to_vec())
            ])
        ];
        let non_greedy_regex = non_greedy_expr.build();

        let greedy_expr = groups![
            seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())),
                Expr::U8Seq(b"\"\"\"".to_vec())
            ]
        ];
        let greedy_regex = greedy_expr.build();

        let input = b"\"\"\"hello\"\"\"world\"\"\"";

        let mut non_greedy_state = non_greedy_regex.init();
        non_greedy_state.execute(input);
        assert_eq!(
            non_greedy_state.matches.get(&0),
            Some(&b"\"\"\"hello\"\"\"".len())
        );

        let mut greedy_state = greedy_regex.init();
        greedy_state.execute(input);
        assert_eq!(greedy_state.matches.get(&0), Some(&input.len()));
    }
}

#[cfg(test)]
mod possible_future_group_ids_tests {
    use super::*;

    fn run_test(expr: impl Into<ExprGroups>, expected_possible_future_group_ids: BTreeSet<GroupID>) {
        let regex = expr.into().build();
        let state = regex.init();
        assert_eq!(
            state.possible_future_group_ids(),
            expected_possible_future_group_ids
        );
    }

    #[test]
    fn test_possible_future_group_ids() {
        run_test(seq![], BTreeSet::new());
        run_test(eat_u8(b'a'), BTreeSet::from([0]));
        run_test(
            groups![eat_u8(b'a'), eat_u8(b'b')],
            BTreeSet::from([0, 1]),
        );
        run_test(
            seq![eat_u8(b'a'), eat_u8(b'b')],
            BTreeSet::from([0]),
        );
        run_test(rep(eat_u8(b'a')), BTreeSet::from([0]));
        run_test(
            groups![
                choice![opt(eat_u8(b'a')), rep(eat_u8(b'b')), eat_u8(b'c')],
                eat_u8(b'a'),
            ],
            BTreeSet::from([0, 1]),
        );
        run_test(
            groups![
                eat_u8(b'a'),
                seq![eat_u8(b'a'), eat_u8(b'a')],
            ],
            BTreeSet::from([0, 1]),
        );
    }

    #[test]
    fn test_possible_future_group_ids_excludes_current_state() {
        let expr = groups![
            eps(),
            eat_u8(b'a'),
        ];
        let regex = expr.build();
        let start_state_index = regex.dfa.start_state;
        let start_state_data = &regex.dfa.states[start_state_index];

        assert_eq!(
            start_state_data.possible_future_group_ids,
            BTreeSet::from([1])
        );
    }
}

#[cfg(test)]
mod group_id_to_u8set_tests {
    use super::*;

    fn build_dfa_with_groups(exprs: Vec<Expr>) -> Regex {
        let expr_groups = ExprGroups {
            groups: exprs.into_iter().map(ExprGroup::from).collect(),
        };
        expr_groups.build()
    }

    #[test]
    fn test_compute_group_id_to_u8set_single_group() {
        let expr = groups![
            eat_u8(b'a')
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 1);
        assert!(group_id_to_u8set.contains_key(&0));
        let u8set = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set.contains(b'a'));
        assert_eq!(u8set.iter().collect::<Vec<u8>>(), vec![b'a']);
    }

    #[test]
    fn test_compute_group_id_to_u8set_multiple_groups() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'b'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 2);
        assert!(group_id_to_u8set.contains_key(&0));
        assert!(group_id_to_u8set.contains_key(&1));

        let u8set_a = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_a.contains(b'a'));
        assert_eq!(u8set_a.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_b = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_b.contains(b'b'));
        assert_eq!(u8set_b.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_compute_group_id_to_u8set_overlapping_groups() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'a'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 2);
        assert!(group_id_to_u8set.contains_key(&0));
        assert!(group_id_to_u8set.contains_key(&1));

        let u8set_a0 = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_a0.contains(b'a'));
        assert_eq!(u8set_a0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_a1 = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_a1.contains(b'a'));
        assert_eq!(u8set_a1.iter().collect::<Vec<u8>>(), vec![b'a']);
    }

    #[test]
    fn test_get_u8set_for_group_existing_group() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'b'),
        ];
        let regex = expr.build();

        let regex_state = regex.init();

        let u8set_group0 = regex_state.get_u8set_for_group(0);
        assert!(u8set_group0.contains(b'a'));
        assert_eq!(u8set_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert!(u8set_group1.contains(b'b'));
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_get_u8set_for_group_nonexistent_group() {
        let expr = groups![
            eat_u8(b'a')
        ];
        let regex = expr.build();

        let regex_state = regex.init();

        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), Vec::<u8>::new());
    }

    #[test]
    fn test_group_id_to_u8set_nested_groups() {
        let expr = groups![
            rep(choice![eat_u8(b'a'), eat_u8(b'b')]),
            eat_u8(b'c'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        dbg!(&regex);
        dbg!(&regex.dfa.states[0].possible_future_group_ids);
        dbg!(group_id_to_u8set);
        assert_eq!(group_id_to_u8set.len(), 2);

        let u8set_group0 = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_group0.contains(b'a'));
        assert!(u8set_group0.contains(b'b'));
        assert_eq!(u8set_group0.iter().collect::<Vec<u8>>(), vec![b'a', b'b']);

        let u8set_group1 = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_group1.contains(b'c'));
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), vec![b'c']);
    }

    #[test]
    fn test_group_id_to_u8set_nonexistent_group() {
        let expr = groups![
            eat_u8(b'a')
        ];
        let regex = expr.build();

        let regex_state = regex.init();
        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), Vec::<u8>::new());
    }

    #[test]
    fn test_group_id_to_u8set_overlapping_groups() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'a'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 2);

        let u8set_group0 = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_group0.contains(b'a'));
        assert_eq!(u8set_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_group1 = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_group1.contains(b'a'));
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), vec![b'a']);
    }

    #[test]
    fn test_get_u8set_for_group_after_transition() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')],
            seq![eat_u8(b'a'), eat_u8(b'c')],
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 2);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));
        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();
        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));

        let mut regex_state = regex.init();
        regex_state.execute(b"a");

        assert_eq!(
            regex.dfa.states[regex_state.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1])
        );

        let group_id_to_u8set_new =
            &regex.dfa.states[regex_state.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_new.len(), 2);
        assert!(group_id_to_u8set_new.contains_key(&0));
        assert!(group_id_to_u8set_new.contains_key(&1));

        let u8set_new_group0 = group_id_to_u8set_new.get(&0).unwrap();
        let u8set_new_group1 = group_id_to_u8set_new.get(&1).unwrap();

        assert!(u8set_new_group0.contains(b'b'));
        assert!(u8set_new_group1.contains(b'c'));
    }

    #[test]
    fn test_group_id_to_u8set_after_multiple_transitions() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'c')],
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'd')],
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'e')],
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 3);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));
        assert!(group_id_to_u8set_0.contains_key(&2));

        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();
        let u8set_0_group2 = group_id_to_u8set_0.get(&2).unwrap();

        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));
        assert!(u8set_0_group2.contains(b'a'));

        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");

        assert_eq!(
            regex.dfa.states[regex_state_a.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1, 2])
        );

        let group_id_to_u8set_a =
            &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_a.len(), 3);
        assert!(group_id_to_u8set_a.contains_key(&0));
        assert!(group_id_to_u8set_a.contains_key(&1));
        assert!(group_id_to_u8set_a.contains_key(&2));

        let u8set_a_group0 = group_id_to_u8set_a.get(&0).unwrap();
        let u8set_a_group1 = group_id_to_u8set_a.get(&1).unwrap();
        let u8set_a_group2 = group_id_to_u8set_a.get(&2).unwrap();

        assert!(u8set_a_group0.contains(b'b'));
        assert!(u8set_a_group1.contains(b'b'));
        assert!(u8set_a_group2.contains(b'b'));

        let mut regex_state_ab = regex.init();
        regex_state_ab.execute(b"ab");

        assert_eq!(
            regex.dfa.states[regex_state_ab.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1, 2])
        );

        let group_id_to_u8set_ab =
            &regex.dfa.states[regex_state_ab.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_ab.len(), 3);
        assert!(group_id_to_u8set_ab.contains_key(&0));
        assert!(group_id_to_u8set_ab.contains_key(&1));
        assert!(group_id_to_u8set_ab.contains_key(&2));

        let u8set_ab_group0 = group_id_to_u8set_ab.get(&0).unwrap();
        let u8set_ab_group1 = group_id_to_u8set_ab.get(&1).unwrap();
        let u8set_ab_group2 = group_id_to_u8set_ab.get(&2).unwrap();

        assert!(u8set_ab_group0.contains(b'c'));
        assert!(u8set_ab_group1.contains(b'd'));
        assert!(u8set_ab_group2.contains(b'e'));
    }

    #[test]
    fn test_group_id_to_u8set_after_consuming_all() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')]
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 1);
        assert!(group_id_to_u8set_0.contains_key(&0));

        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        assert!(u8set_0_group0.contains(b'a'));
        assert_eq!(u8set_0_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");
        assert_eq!(
            regex.dfa.states[regex_state_a.current_state].possible_future_group_ids,
            BTreeSet::from([0])
        );

        let group_id_to_u8set_a =
            &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_a.len(), 1);
        assert!(group_id_to_u8set_a.contains_key(&0));

        let u8set_a_group0 = group_id_to_u8set_a.get(&0).unwrap();
        assert!(u8set_a_group0.contains(b'b'));
        assert_eq!(u8set_a_group0.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_get_u8set_for_group_multiple_transitions() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')],
            seq![eat_u8(b'a'), eat_u8(b'c')],
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 2);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));

        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();

        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));
        assert_eq!(u8set_0_group0.iter().collect::<Vec<u8>>(), vec![b'a']);
        assert_eq!(u8set_0_group1.iter().collect::<Vec<u8>>(), vec![b'a']);

        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");

        let group_id_to_u8set_a =
            &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_a.len(), 2);
        assert!(group_id_to_u8set_a.contains_key(&0));
        assert!(group_id_to_u8set_a.contains_key(&1));

        let u8set_a_group0 = group_id_to_u8set_a.get(&0).unwrap();
        let u8set_a_group1 = group_id_to_u8set_a.get(&1).unwrap();

        assert!(u8set_a_group0.contains(b'b'));
        assert!(u8set_a_group1.contains(b'c'));
        assert_eq!(u8set_a_group0.iter().collect::<Vec<u8>>(), vec![b'b']);
        assert_eq!(u8set_a_group1.iter().collect::<Vec<u8>>(), vec![b'c']);
    }
}

#[cfg(test)]
mod group_u8set_tests {
    use super::*;

    #[test]
    fn test_get_u8set_for_group() {
        let mut dfa = DFA {
            states: Vec::new(),
            start_state: 0,
            non_greedy_finalizers: BTreeSet::new(),
        };

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: BTreeSet::new(),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: BTreeSet::new(),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: BTreeSet::from([0]),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: BTreeSet::from([1]),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states[0].transitions.insert(b'a', 1);
        dfa.states[1].transitions.insert(b'b', 2);
        dfa.states[1].transitions.insert(b'c', 3);

        dfa.compute_possible_future_group_ids();
        dfa.compute_group_id_to_u8set();

        let regex = Regex { dfa };

        let state0 = regex.init_to_state(0);
        let u8set_group0_state0 = state0.get_u8set_for_group(0);
        let u8set_group1_state0 = state0.get_u8set_for_group(1);
        assert!(u8set_group0_state0.contains(b'a'));
        assert!(u8set_group1_state0.contains(b'a'));

        let state1 = regex.init_to_state(1);
        let u8set_group0_state1 = state1.get_u8set_for_group(0);
        let u8set_group1_state1 = state1.get_u8set_for_group(1);
        assert!(u8set_group0_state1.contains(b'b'));
        assert!(!u8set_group0_state1.contains(b'c'));
        assert!(u8set_group1_state1.contains(b'c'));
        assert!(!u8set_group1_state1.contains(b'b'));

        let state2 = regex.init_to_state(2);
        let u8set_group0_state2 = state2.get_u8set_for_group(0);
        let u8set_group1_state2 = state2.get_u8set_for_group(1);
        assert!(u8set_group0_state2.iter().next().is_none());
        assert!(u8set_group1_state2.iter().next().is_none());

        let state3 = regex.init_to_state(3);
        let u8set_group0_state3 = state3.get_u8set_for_group(0);
        let u8set_group1_state3 = state3.get_u8set_for_group(1);
        assert!(u8set_group0_state3.iter().next().is_none());
        assert!(u8set_group1_state3.iter().next().is_none());
    }
}

#[cfg(test)]
mod tests_nov_24 {
    use super::*;

    #[test]
    fn test_eat_u8() {
        let expr = groups![
            eat_u8(b'a'),
            seq![eat_u8(b'a'), eat_u8(b'b')],
        ];

        let regex = expr.build();
        dbg!(&regex);
        let mut state = regex.init();
        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 1)]));
        state.clear_matches();

        state.execute(b"b");
        assert_eq!(state.matches, BTreeMap::from([(1, 2)]));
    }

    #[test]
    fn test_reasonable_number_of_states() {
        let expr = choice![eat_u8(b'a'), eat_u8(b'b'),];
        let regex = expr.build();
        dbg!(&regex);
        assert_eq!(regex.dfa.states.len(), 2);
    }
}

#[cfg(test)]
mod test_python {
    use super::*;
    use crate::datastructures::u8set::U8Set;
    use crate::{choice, seq};

    #[ignore]
    #[test]
    fn test_full_python_tokenizer_recognizes_name() {
        let digit = Expr::U8Class(U8Set::from_range(b'0', b'9'));
        let alph_lower = Expr::U8Class(U8Set::from_range(b'a', b'z'));
        let alph_upper = Expr::U8Class(U8Set::from_range(b'A', b'Z'));
        let name_start = choice![alph_lower.clone(), alph_upper.clone(), eat_u8(b'_')];
        let name_middle = choice![name_start.clone(), digit.clone()];

        let ignore = rep(choice![
             eat_u8(b' '),
             seq![eat_u8(b'#'), rep(Expr::U8Class(U8Set::all().without(b'\n'))), opt(eat_u8(b'\n'))],
         ]);

        let tokens_core: BTreeMap<&str, Expr> = BTreeMap::from([
            ("NAME", seq![name_start, rep(name_middle)]),
            ("NUMBER", choice![
                rep1(digit.clone()),
                seq![rep1(digit.clone()), eat_u8(b'.'), rep(digit.clone())],
                seq![eat_u8(b'.'), rep1(digit.clone())],
            ]),
            ("STRING", choice![
                seq![eat_u8(b'"'), rep(Expr::U8Class(U8Set::all().without(b'"'))), eat_u8(b'"')],
                seq![eat_u8(b'\''), rep(Expr::U8Class(U8Set::all().without(b'\''))), eat_u8(b'\'')],
            ]),
            ("FSTRING_START", Expr::U8Seq(b"f'".to_vec())),
            ("FSTRING_END", Expr::U8Seq(b"'".to_vec())),
            ("FSTRING_MIDDLE", rep1(Expr::U8Class(U8Set::all().difference(&U8Set::from_slice(&[b'{', b'}']))))),
            ("NEWLINE", eps()),
            ("INDENT", eps()),
            ("DEDENT", eps()),
            ("TYPE_COMMENT", eps()),
            ("ENDMARKER", eps()),
            ("LPAREN", eat_u8(b'(')),
            ("RPAREN", eat_u8(b')')),
            ("LSQB", eat_u8(b'[')),
            ("RSQB", eat_u8(b']')),
            ("LBRACE", eat_u8(b'{')),
            ("RBRACE", eat_u8(b'}')),
            ("COMMA", eat_u8(b',')),
            ("COLON", eat_u8(b':')),
            ("DOT", eat_u8(b'.')),
            ("SEMI", eat_u8(b';')),
            ("PLUS", eat_u8(b'+')),
            ("MINUS", eat_u8(b'-')),
            ("STAR", eat_u8(b'*')),
            ("SLASH", eat_u8(b'/')),
            ("VBAR", eat_u8(b'|')),
            ("AMPER", eat_u8(b'&')),
            ("LESS", eat_u8(b'<')),
            ("GREATER", eat_u8(b'>')),
            ("EQUAL", eat_u8(b'=')),
            ("PERCENT", eat_u8(b'%')),
            ("CIRCUMFLEX", eat_u8(b'^')),
            ("TILDE", eat_u8(b'~')),
            ("AT", eat_u8(b'@')),
            ("EXCLAMATION", eat_u8(b'!')),
            ("DOUBLESTAR", Expr::U8Seq(b"**".to_vec())),
            ("DOUBLESLASH", Expr::U8Seq(b"//".to_vec())),
            ("LEFTSHIFT", Expr::U8Seq(b"<<".to_vec())),
            ("RIGHTSHIFT", Expr::U8Seq(b">>".to_vec())),
            ("EQEQUAL", Expr::U8Seq(b"==".to_vec())),
            ("NOTEQUAL", Expr::U8Seq(b"!=".to_vec())),
            ("LESSEQUAL", Expr::U8Seq(b"<=".to_vec())),
            ("GREATEREQUAL", Expr::U8Seq(b">=".to_vec())),
            ("ATEQUAL", Expr::U8Seq(b"@=".to_vec())),
            ("PLUSEQUAL", Expr::U8Seq(b"+=".to_vec())),
            ("MINEQUAL", Expr::U8Seq(b"-=".to_vec())),
            ("STAREQUAL", Expr::U8Seq(b"*=".to_vec())),
            ("SLASHEQUAL", Expr::U8Seq(b"/=".to_vec())),
            ("PERCENTEQUAL", Expr::U8Seq(b"%=".to_vec())),
            ("AMPEREQUAL", Expr::U8Seq(b"&=".to_vec())),
            ("VBAREQUAL", Expr::U8Seq(b"|=".to_vec())),
            ("CIRCUMFLEXEQUAL", Expr::U8Seq(b"^=".to_vec())),
            ("LEFTSHIFTEQUAL", Expr::U8Seq(b"<<=".to_vec())),
            ("RIGHTSHIFTEQUAL", Expr::U8Seq(b">>=".to_vec())),
            ("DOUBLESTAREQUAL", Expr::U8Seq(b"**=".to_vec())),
            ("DOUBLESLASHEQUAL", Expr::U8Seq(b"//=".to_vec())),
            ("RARROW", Expr::U8Seq(b"->".to_vec())),
            ("ELLIPSIS", Expr::U8Seq(b"...".to_vec())),
            ("COLONEQUAL", Expr::U8Seq(b":=".to_vec())),
        ]);

        let mut token_groups: Vec<ExprGroup> = Vec::new();
        let mut token_name_to_id: BTreeMap<&str, GroupID> = BTreeMap::new();
        for (name, core_expr) in tokens_core {
            let group_id = token_groups.len();
            token_name_to_id.insert(name, group_id);
            token_groups.push(greedy_group(seq![ignore.clone(), core_expr]));
        }

        let expr_groups = groups(token_groups);
        let regex = expr_groups.build();

        let mut state = regex.init();
        state.execute(b"hello");

        assert!(state.definitely_matches(), "Tokenizer should match 'hello'");
        assert_eq!(
            state.matches.get(&token_name_to_id["NAME"]),
            Some(&5),
            "NAME token should be matched at position 5"
        );
    }
}
