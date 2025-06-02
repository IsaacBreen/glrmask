use crate::datastructures::charmap::TrieMap;
use crate::datastructures::frozenset::FrozenSet;
use crate::datastructures::u8set::U8Set;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap; // Added for derive macro pattern, aliased

pub type GroupID = usize;

#[derive(Debug, Clone)]
pub struct NFAState {
    transitions: TrieMap<Vec<usize>>,
    epsilon_transitions: Vec<usize>,
    finalizers: BTreeSet<GroupID>,
    non_greedy_finalizers: BTreeSet<GroupID>,
}

// Manual impl for NFAState
impl JSONConvertible for NFAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("transitions".to_string(), self.transitions.to_json());
        obj.insert("epsilon_transitions".to_string(), self.epsilon_transitions.to_json());
        obj.insert("finalizers".to_string(), self.finalizers.to_json());
        obj.insert("non_greedy_finalizers".to_string(), self.non_greedy_finalizers.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let transitions = obj.remove("transitions").ok_or_else(|| "Missing field transitions for NFAState".to_string())
                                     .and_then(|n| TrieMap::<Vec<usize>>::from_json(n))?;
                let epsilon_transitions = obj.remove("epsilon_transitions").ok_or_else(|| "Missing field epsilon_transitions for NFAState".to_string())
                                             .and_then(Vec::<usize>::from_json)?;
                let finalizers = obj.remove("finalizers").ok_or_else(|| "Missing field finalizers for NFAState".to_string())
                                    .and_then(BTreeSet::<GroupID>::from_json)?;
                let non_greedy_finalizers = obj.remove("non_greedy_finalizers").ok_or_else(|| "Missing field non_greedy_finalizers for NFAState".to_string())
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
                let states = obj.remove("states").ok_or_else(|| "Missing field states for NFA".to_string())
                                .and_then(Vec::<NFAState>::from_json)?;
                let start_state = obj.remove("start_state").ok_or_else(|| "Missing field start_state for NFA".to_string())
                                     .and_then(usize::from_json)?;
                Ok(NFA { states, start_state })
            }
            _ => Err("Expected JSONNode::Object for NFA".to_string()),
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DFAState {
    pub transitions: TrieMap<usize>,
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
        obj.insert("possible_future_group_ids".to_string(), self.possible_future_group_ids.to_json());
        obj.insert("group_id_to_u8set".to_string(), self.group_id_to_u8set.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let transitions = obj.remove("transitions").ok_or_else(|| "Missing field transitions for DFAState".to_string())
                                     .and_then(|n| TrieMap::<usize>::from_json(n))?;
                let finalizers = obj.remove("finalizers").ok_or_else(|| "Missing field finalizers for DFAState".to_string())
                                    .and_then(BTreeSet::<GroupID>::from_json)?;
                let possible_future_group_ids = obj.remove("possible_future_group_ids").ok_or_else(|| "Missing field possible_future_group_ids for DFAState".to_string())
                                                   .and_then(BTreeSet::<GroupID>::from_json)?;
                let group_id_to_u8set = obj.remove("group_id_to_u8set").ok_or_else(|| "Missing field group_id_to_u8set for DFAState".to_string())
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
        obj.insert("non_greedy_finalizers".to_string(), self.non_greedy_finalizers.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let states = obj.remove("states").ok_or_else(|| "Missing field states for DFA".to_string())
                                .and_then(Vec::<DFAState>::from_json)?;
                let start_state = obj.remove("start_state").ok_or_else(|| "Missing field start_state for DFA".to_string())
                                     .and_then(usize::from_json)?;
                let non_greedy_finalizers = obj.remove("non_greedy_finalizers").ok_or_else(|| "Missing field non_greedy_finalizers for DFA".to_string())
                                               .and_then(BTreeSet::<GroupID>::from_json)?;
                Ok(DFA { states, start_state, non_greedy_finalizers })
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
                let dfa = obj.remove("dfa").ok_or_else(|| "Missing field dfa for Regex".to_string())
                             .and_then(DFA::from_json)?;
                Ok(Regex { dfa })
            }
            _ => Err("Expected JSONNode::Object for Regex".to_string()),
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
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
                let group_id = obj.remove("group_id").ok_or_else(|| "Missing field group_id for Match".to_string())
                                  .and_then(GroupID::from_json)?;
                let position = obj.remove("position").ok_or_else(|| "Missing field position for Match".to_string())
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
                let position = obj.remove("position").ok_or_else(|| "Missing field position for FinalStateReport".to_string())
                                  .and_then(usize::from_json)?;
                let matches = obj.remove("matches").ok_or_else(|| "Missing field matches for FinalStateReport".to_string())
                                 .and_then(|n| BTreeMap::<GroupID, usize>::from_json(n))?;
                Ok(FinalStateReport { position, matches })
            }
            _ => Err("Expected JSONNode::Object for FinalStateReport".to_string()),
        }
    }
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
        // Potentially serialize only non-reference fields if useful for debugging.
        // let mut obj = StdMap::new();
        // obj.insert("position".to_string(), self.position.to_json());
        // obj.insert("current_state".to_string(), self.current_state.to_json());
        // obj.insert("matches".to_string(), self.matches.to_json());
        // obj.insert("done".to_string(), self.done.to_json());
        // JSONNode::Object(obj)
    }
    fn from_json(_node: JSONNode) -> Result<Self, String> {
        Err("RegexState deserialization is not supported due to lifetime and reference.".to_string())
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr {
    U8Seq(Vec<u8>),
    U8Class(U8Set),
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
            Expr::Quantifier(expr, q_type) => {
                obj.insert("variant".to_string(), JSONNode::String("Quantifier".to_string()));
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
                let variant = obj.remove("variant").ok_or_else(|| "Missing field variant for Expr".to_string())
                                   .and_then(String::from_json)?;
                match variant.as_str() {
                    "U8Seq" => {
                        let bytes = obj.remove("bytes").ok_or_else(|| "Missing field bytes for U8Seq".to_string())
                                       .and_then(Vec::<u8>::from_json)?;
                        Ok(Expr::U8Seq(bytes))
                    }
                    "U8Class" => {
                        let u8set = obj.remove("u8set").ok_or_else(|| "Missing field u8set for U8Class".to_string())
                                       .and_then(U8Set::from_json)?;
                        Ok(Expr::U8Class(u8set))
                    }
                    "Quantifier" => {
                        let expr_node = obj.remove("expr").ok_or_else(|| "Missing field expr for Quantifier".to_string())?;
                        let expr = Box::new(Expr::from_json(expr_node)?);
                        let q_type = obj.remove("q_type").ok_or_else(|| "Missing field q_type for Quantifier".to_string())
                                          .and_then(QuantifierType::from_json)?;
                        Ok(Expr::Quantifier(expr, q_type))
                    }
                    "Choice" => {
                        let exprs = obj.remove("exprs").ok_or_else(|| "Missing field exprs for Choice".to_string())
                                       .and_then(Vec::<Expr>::from_json)?;
                        Ok(Expr::Choice(exprs))
                    }
                    "Seq" => {
                        let exprs = obj.remove("exprs").ok_or_else(|| "Missing field exprs for Seq".to_string())
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
                let expr = obj.remove("expr").ok_or_else(|| "Missing field expr for ExprGroup".to_string())
                                .and_then(Expr::from_json)?;
                let is_non_greedy = obj.remove("is_non_greedy").ok_or_else(|| "Missing field is_non_greedy for ExprGroup".to_string())
                                       .and_then(bool::from_json)?;
                Ok(ExprGroup { expr, is_non_greedy })
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
                let groups = obj.remove("groups").ok_or_else(|| "Missing field groups for ExprGroups".to_string())
                                  .and_then(Vec::<ExprGroup>::from_json)?;
                Ok(ExprGroups { groups })
            }
            _ => Err("Expected JSONNode::Object for ExprGroups".to_string()),
        }
    }
}


impl From<Expr> for ExprGroup {
    fn from(expr: Expr) -> Self {
        ExprGroup { expr, is_non_greedy: false }
    }
}

impl From<Expr> for ExprGroups {
    fn from(expr: Expr) -> Self {
        ExprGroups { groups: vec![ExprGroup { expr, is_non_greedy: false }] }
    }
}

pub fn eat_u8(c: u8) -> Expr {
    Expr::U8Seq(vec![c])
}

pub fn eat_u8_set(u8s: U8Set) -> Expr {
    Expr::U8Class(u8s)
}

pub fn eat_u8_negation(c: u8) -> Expr {
    Expr::U8Class(U8Set::from_u8(c).complement())
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

pub fn prec<T: Into<Expr>>(_precedence: isize, expr: T) -> ExprGroup {
    ExprGroup { expr: expr.into(), is_non_greedy: false }
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

            for (transition_u8, next_states) in &state.transitions {
                f.write_str(&format!("  - '{}': {:?}\n", transition_u8 as char, next_states))?; // Display u8 as char
            }

            for next_state in &state.epsilon_transitions {
                f.write_str(&format!("  - Epsilon: {}\n", next_state))?;
            }

            if !state.finalizers.is_empty() {
                f.write_str(&format!("  - Finalizers: {:?}\n", state.finalizers))?;
            }

            if !state.non_greedy_finalizers.is_empty() {
                f.write_str(&format!("  - Non-Greedy Finalizers: {:?}\n", state.non_greedy_finalizers))?;
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
                f.write_str(&format!("  - {} ({:?}): {}\n", transition_u8, transition_u8 as char, next_state))?; // Display u8 as char
            }

            if !state.finalizers.is_empty() {
                f.write_str(&format!("  - Finalizers: {:?}\n", state.finalizers))?;
            }

            if !state.possible_future_group_ids.is_empty() {
                f.write_str(&format!("  - Possible Future Group IDs: {:?}\n", state.possible_future_group_ids))?;
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

impl NFAState {
    pub fn new() -> NFAState {
        NFAState {
            transitions: TrieMap::new(),
            epsilon_transitions: Vec::new(),
            finalizers: BTreeSet::new(),
            non_greedy_finalizers: BTreeSet::new(),
        }
    }
}

impl ExprGroups {
    pub fn build(self) -> Regex {
        let mut dfa = self.build_nfa().to_dfa();
        dfa.minimize();
        Regex { dfa }
    }

    fn build_nfa(self) -> NFA {
        let mut nfa = NFA {
            states: vec![NFAState::new()],
            start_state: 0,
        };

        for (group, ExprGroup { expr, is_non_greedy }) in self.groups.into_iter().enumerate() {
            let end_state = Expr::handle_expr(expr, &mut nfa, 0);
            if is_non_greedy {
                nfa.states[end_state].finalizers.insert(group);
                // Additionally, track that this finalizer is non-greedy
                nfa.states[end_state].non_greedy_finalizers.insert(group);
            } else {
                nfa.states[end_state].finalizers.insert(group);
            }
        }

        nfa
    }
}

impl Expr {
    pub fn build(self) -> Regex {
        ExprGroups { groups: vec![ExprGroup { expr: self, is_non_greedy: false }] }.build()
    }

    fn handle_expr(expr: Expr, nfa: &mut NFA, mut current_state: usize) -> usize {
        match expr {
            Expr::U8Seq(u8s) => {
                let mut next_state = current_state;
                for c in u8s {
                    let new_state = nfa.add_state();
                    nfa.add_transition(next_state, c, new_state);
                    next_state = new_state;
                }
                next_state
            }
            Expr::U8Class(u8s) => {
                let new_state = nfa.add_state();
                for ch in u8s.iter() {
                    nfa.add_transition(current_state, ch, new_state);
                }
                new_state
            }
            Expr::Quantifier(expr, quantifier_type) => {
                match quantifier_type {
                    QuantifierType::ZeroOrMore => {
                        let loop_start_state = nfa.add_state();

                        // Epsilon transition from current state to loop start state
                        nfa.add_epsilon_transition(current_state, loop_start_state);

                        // Process the expr
                        let expr_end_state = Self::handle_expr(*expr, nfa, loop_start_state);

                        // Epsilon transition from expr end state back to loop start state for repetition
                        nfa.add_epsilon_transition(expr_end_state, loop_start_state);

                        // The loop start state becomes the new current state
                        loop_start_state
                    }
                    QuantifierType::OneOrMore => {
                        let loop_start_state = nfa.add_state();

                        // Epsilon transition from current state to loop start state
                        nfa.add_epsilon_transition(current_state, loop_start_state);

                        // Process the expr
                        let expr_end_state = Self::handle_expr(*expr, nfa, loop_start_state);

                        // Epsilon transition from expr end state back to loop start state for repetition
                        nfa.add_epsilon_transition(expr_end_state, loop_start_state);

                        // The expr end state becomes the new current state
                        expr_end_state
                    }
                    QuantifierType::ZeroOrOne => {
                        // Process the expr
                        let expr_end_state = Self::handle_expr(*expr, nfa, current_state);

                        // Epsilon transition from current state to expr end state
                        nfa.add_epsilon_transition(expr_end_state, expr_end_state);

                        // The expr end state becomes the new current state
                        expr_end_state
                    }
                }
            }
            Expr::Choice(exprs) => {
                // New start state for choice
                let choice_end_state = nfa.add_state();   // New end state for choice

                for expr in exprs {
                    // Process the expr and get its end state
                    let expr_end_state = Self::handle_expr(expr, nfa, current_state);

                    // Connect the end state of the expr to the end state of the choice
                    nfa.add_epsilon_transition(expr_end_state, choice_end_state);
                }

                // The end state of the choice becomes the new current state
                choice_end_state
            }
            Expr::Seq(exprs) => {
                for expr in exprs {
                    current_state = Self::handle_expr(expr, nfa, current_state);
                }
                current_state
            }
            Expr::Epsilon => current_state
        }
    }
}

impl NFA {
    pub fn add_state(&mut self) -> usize {
        let new_index = self.states.len();
        self.states.push(NFAState::new());
        new_index
    }

    pub fn add_transition(&mut self, from: usize, on_u8: u8, to: usize) {
        self.states[from]
            .transitions
            .entry(on_u8)
            .or_insert_with(Vec::new)
            .push(to);
    }

    pub fn add_epsilon_transition(&mut self, from: usize, to: usize) {
        self.states[from].epsilon_transitions.push(to);
    }

    pub fn to_dfa(self) -> DFA {
        let mut dfa_states: Vec<DFAState> = Vec::new();
        let mut dfa_state_map: BTreeMap<FrozenSet<usize>, usize> = BTreeMap::new();
        let mut worklist: Vec<FrozenSet<usize>> = Vec::new();

        let mut epsilon_closures = self.compute_epsilon_closures();

        // Compute the epsilon closure of the NFA start state and use it as the DFA start state
        let start_closure = epsilon_closures[self.start_state].clone();
        let start_state_set = FrozenSet::from_iter(start_closure.iter().cloned());
        worklist.push(start_state_set.clone());
        dfa_state_map.insert(start_state_set.clone(), 0);

        // Initialize the first DFA state
        let closure = epsilon_closures[self.start_state].clone();
        let mut finalizers = BTreeSet::new();
        let mut non_greedy_finalizers = BTreeSet::new();
        for &state in &closure {
            finalizers.extend(self.states[state].finalizers.iter().cloned());
            non_greedy_finalizers.extend(self.states[state].non_greedy_finalizers.iter().cloned());
        }

        dfa_states.push(DFAState {
            transitions: TrieMap::new(),
            finalizers,
            possible_future_group_ids: BTreeSet::new(), // Will be computed later
            group_id_to_u8set: BTreeMap::new(),  // Will be computed later
        });

        while let Some(current_set) = worklist.pop() {
            let current_dfa_state = *dfa_state_map.get(&current_set).unwrap();
            let mut transition_map: BTreeMap<u8, BTreeSet<usize>> = BTreeMap::new();

            // For each state in the current DFA state, look at the NFA transitions
            for &state in current_set.iter() {
                for (input, next_states) in &self.states[state].transitions {
                    for &next_state in next_states {
                        transition_map
                            .entry(input)
                            .or_insert_with(BTreeSet::new)
                            .insert(next_state);
                    }
                }
            }

            // For each transition, compute the epsilon closure of the resulting state set
            for (&input_u8, next_states) in &transition_map {
                let mut closure = BTreeSet::new();
                for &next_state in next_states {
                    closure.extend(&epsilon_closures[next_state]);
                }
                let frozen_closure = FrozenSet::from_iter(closure.iter().cloned());

                // If this set of states is new, add it as a new DFA state
                let next_dfa_state = if let Some(&existing_state) = dfa_state_map.get(&frozen_closure) {
                    existing_state
                } else {
                    let new_state_index = dfa_states.len();
                    dfa_state_map.insert(frozen_closure.clone(), new_state_index);
                    worklist.push(frozen_closure.clone());

                    // Compute finalizers for the new DFA state
                    let mut new_finalizers = BTreeSet::new();
                    let mut new_non_greedy_finalizers = BTreeSet::new();
                    for &state in closure.iter() {
                        new_finalizers.extend(self.states[state].finalizers.iter().cloned());
                        new_non_greedy_finalizers.extend(self.states[state].non_greedy_finalizers.iter().cloned());
                    }

                    dfa_states.push(DFAState {
                        transitions: TrieMap::new(),
                        finalizers: new_finalizers,
                        possible_future_group_ids: BTreeSet::new(), // Will be computed later
                        group_id_to_u8set: BTreeMap::new(),  // Will be computed later
                    });

                    new_state_index
                };

                // Insert the transition into the DFA state
                dfa_states[current_dfa_state].transitions.insert(input_u8, next_dfa_state);
            }
        }

        let mut dfa = DFA {
            states: dfa_states,
            start_state: 0,
            non_greedy_finalizers: BTreeSet::new(),
        };

        for state in &self.states {
            dfa.non_greedy_finalizers.extend(state.non_greedy_finalizers.iter().cloned());
        }

        dfa.compute_possible_future_group_ids();
        dfa.compute_group_id_to_u8set();

        dfa
    }

    fn epsilon_closure(&self, state: usize) -> BTreeSet<usize> {
        let mut closure = BTreeSet::new();
        let mut stack = vec![state];

        while let Some(state) = stack.pop() {
            if closure.insert(state) {
                stack.extend(&self.states[state].epsilon_transitions);
            }
        }

        closure
    }

    fn compute_epsilon_closures(&self) -> Vec<BTreeSet<usize>> {
        (0..self.states.len())
            .map(|state| self.epsilon_closure(state))
            .collect()
    }
}

impl DFA {
    pub fn compute_possible_future_group_ids(&mut self) {
        // Initialize possible_future_group_ids as empty. We only want to include
        // group IDs reachable via *future* transitions.
        for state in &mut self.states {
            state.possible_future_group_ids = BTreeSet::new();
        }

        loop {
            let mut changed = false;
            for state_index in 0..self.states.len() {
                let state = self.states[state_index].clone(); // Clone to avoid borrow checker issues
                for (_input, &next_state_index) in &state.transitions {
                    let next_possible_future_groups = self.states[next_state_index].possible_future_group_ids.clone();
                    let next_finalizers = self.states[next_state_index].finalizers.clone();
                    let state_possible_future_groups = &mut self.states[state_index].possible_future_group_ids;

                    let old_len = state_possible_future_groups.len();
                    state_possible_future_groups.extend(next_finalizers.iter()); // Add finalizers of the *next* state
                    state_possible_future_groups.extend(next_possible_future_groups.iter());

                    if state_possible_future_groups.len() > old_len {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    pub fn compute_group_id_to_u8set(&mut self) {
        // Create the vector of possible future group IDs within a block scope, cloning the data
        let possible_current_or_future_group_ids: Vec<BTreeSet<GroupID>> = {
            self.states.iter().map(|state| &state.possible_future_group_ids | &state.finalizers).collect()
        };

        // Now that the block has ended, there are no borrows of self.states
        for state in self.states.iter_mut() {
            let mut group_id_to_u8set: BTreeMap<GroupID, U8Set> = BTreeMap::new();

            for (input_u8, &next_state_index) in &state.transitions {
                let next_possible_current_or_future_group_ids = &possible_current_or_future_group_ids[next_state_index];

                for &group_id in next_possible_current_or_future_group_ids {
                    group_id_to_u8set
                        .entry(group_id)
                        .or_insert_with(U8Set::none)
                        .insert(input_u8);
                }
            }

            state.group_id_to_u8set = group_id_to_u8set;
        }
    }

    fn remove_unreachable_states(&mut self) {
        // Find reachable states using BFS
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

        // Create mapping from old indices to new indices
        let mut state_mapping = vec![0; self.states.len()];
        let mut new_index = 0;
        for (old_index, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                state_mapping[old_index] = new_index;
                new_index += 1;
            }
        }

        // Keep only reachable states and update transitions
        let mut new_states = Vec::new();
        for (old_index, state) in self.states.iter().enumerate() {
            if reachable[old_index] {
                let mut new_state = state.clone();
                // Update transitions to use new state indices
                new_state.transitions = new_state.transitions
                    .iter()
                    .map(|(u8, &next)| (u8, state_mapping[next]))
                    .collect();
                new_states.push(new_state);
            }
        }

        // Update the DFA
        self.states = new_states;
        // Start state remains at 0 since it's always reachable
        self.start_state = 0;
    }

    fn minimize(&mut self) {
        if self.states.is_empty() {
            return;
        }

        // Step 1: Create initial partition based on finalizers and transitions
        let mut partitions = BTreeMap::<(BTreeSet<GroupID>, BTreeMap<u8, usize>), BTreeSet<usize>>::new();

        for (state_idx, state) in self.states.iter().enumerate() {
            let key = (
                state.finalizers.clone(),
                state.transitions.iter().map(|(u8, &next)| (u8, next)).collect()
            );
            partitions.entry(key).or_default().insert(state_idx);
        }

        // Step 2: Refine partitions until no more refinement is possible
        let mut partition_list: Vec<BTreeSet<usize>> = partitions.into_values().collect();
        let mut changed = true;
        while changed {
            changed = false;
            let mut new_partitions = Vec::new();

            for partition in &partition_list {
                let mut refined_partitions = BTreeMap::new();

                for &state in partition {
                    let mut signature = BTreeMap::new();

                    // For each transition, record which partition it leads to
                    for (u8, &next_state) in &self.states[state].transitions {
                        let target_partition = new_partitions.iter()
                            .chain(partition_list.iter())
                            .position(|p| p.contains(&next_state))
                            .unwrap_or(usize::MAX);
                        signature.insert(u8, target_partition);
                    }

                    refined_partitions.entry(signature)
                        .or_insert_with(BTreeSet::new)
                        .insert(state);
                }

                if refined_partitions.len() > 1 {
                    changed = true;
                    new_partitions.extend(refined_partitions.into_values());
                } else {
                    new_partitions.push(partition.clone());
                }
            }

            partition_list = new_partitions;
        }

        // Step 3: Build the minimized DFA
        let mut state_mapping = vec![0; self.states.len()];

        // Find which partition contains the start state
        let start_partition_idx = partition_list.iter()
            .position(|p| p.contains(&self.start_state))
            .unwrap();

        // Ensure the start partition is first in the list
        if start_partition_idx != 0 {
            partition_list.swap(0, start_partition_idx);
        }

        // Build state mapping
        for (new_state, partition) in partition_list.iter().enumerate() {
            for &old_state in partition {
                state_mapping[old_state] = new_state;
            }
        }

        let mut new_states = Vec::with_capacity(partition_list.len());
        for partition in &partition_list {
            let old_state = partition.iter().next().unwrap();
            let mut new_state = self.states[*old_state].clone();

            // Update transitions according to the new state mapping
            new_state.transitions = new_state.transitions
                .iter()
                .map(|(u8, &next)| (u8, state_mapping[next]))
                .collect();

            new_states.push(new_state);
        }

        // The start state should now be at index 0
        self.start_state = 0;
        self.states = new_states;
        self.remove_unreachable_states();

        // Recompute metadata
        self.compute_possible_future_group_ids();
        self.compute_group_id_to_u8set();
    }
}

impl RegexState<'_> {
    pub fn execute(&mut self, text: &[u8]) {
        if self.done {
            self.position += text.len();
            return;
        }
        let dfa = &self.regex.dfa;
        let mut local_position = 0;
        while local_position < text.len() {
            let state_data = &dfa.states[self.current_state];
            let next_u8 = text[local_position];
            if let Some(&next_state) = state_data.transitions.get(next_u8) {
                self.current_state = next_state;
                local_position += 1;
                // Handle greedy finalizers
                for &group_id in &dfa.states[self.current_state].finalizers {
                    if dfa.non_greedy_finalizers.contains(&group_id) {
                        self.matches.entry(group_id).or_insert(self.position + local_position);
                    } else {
                        // Overwrite existing match for greedy groups
                        self.matches.insert(group_id, self.position + local_position);
                    }
                }

                // Check for early termination. Only continue if it's possible to match either:
                // - a greedy group, or
                // - a non-greedy group that has not been matched yet
                let matched: BTreeSet<GroupID> = self.matches.keys().cloned().collect();
                let excluded: BTreeSet<GroupID> = matched.intersection(&dfa.non_greedy_finalizers).cloned().collect();
                let should_terminate = dfa.states[self.current_state].possible_future_group_ids.difference(&excluded).next().is_none();

                if should_terminate {
                    self.position += text.len();
                    self.done = true;
                    return;
                }
            } else {
                // No matching transition, we're done
                self.position += text.len();
                self.done = true;
                return;
            }
        }
        // Reached the end of input, mark as done if no further transitions
        self.position += text.len();
        if dfa.states[self.current_state].transitions.is_empty() {
            self.done = true;
        }
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

    /// Matches repeatedly, resolving ambiguity in the following way:
    /// 1. If it's still possible to match something, stop. Don't return a result for the final match, since we can't rule out the possibility of a longer match.
    /// 2. Otherwise, if there is more than one match, return the longest match.
    /// 3. If there is more than one match of this length, return the one with the lowest group ID.
    pub fn greedy_find_all(&mut self, text: &[u8], terminate: bool) -> Vec<Match> {
        let mut matches: Vec<Match> = Vec::new();
        let start_position = self.position;
        let mut local_position = 0;
        self.position = 0;
        loop {
            self.execute(&text[local_position..]);
            if self.ended() {
                if let Some(m) = self.get_greedy_match() {
                    // Advance the local position to the end of the match.
                    local_position += m.position;

                    // Add the match to the list of successful matches.
                    matches.push(m);

                    // Reset the state and advance the internal position.
                    self.reset();
                } else {
                    // Ended but no match. This indicates a tokenization error.
                    // Return the successful matches.
                    self.position = start_position + local_position;
                    return matches;
                }
            } else {
                // Didn't end. We must have run out of input.
                // If we're supposed to terminate, add the final match (if any) and terminate.
                if terminate {
                    if let Some(m) = self.get_greedy_match() {
                        // Add the final match to the list of successful matches.
                        matches.push(m);
                    }
                    self.end();
                    return matches;
                }
                // Return the successful matches.
                self.position = start_position + local_position;
                return matches;
            }
        }
    }

    /// Returns a single match as follows:
    ///
    /// 1. If there is more than one match, return the longest match.
    /// 2. If there is more than one match of this length, return the one with the lowest group ID.
    ///
    /// If there is no match, returns None.
    pub fn get_greedy_match(&self) -> Option<Match> {
        if self.matches.len() == 0 {
            return None;
        }
        let mut matches = self.matches.iter();
        let (mut longest_match_group_id, mut longest_match_position) = matches.next().unwrap();
        for (group_id, position) in matches {
            if position > longest_match_position {
                longest_match_group_id = group_id;
                longest_match_position = position;
            }
        }
        Some(Match {
            group_id: *longest_match_group_id,
            position: *longest_match_position,
        })
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
        // Get all possible u8s that can match next
        state_data.transitions.keys_as_u8set()
    }

    pub fn get_terminal_u8set(&self) -> U8Set {
        // Get u8s that could take the regex to a terminal state (a state with a finalizer)
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
        } else {
            if self.done {
                Some(false)
            } else {
                None
            }
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
        // Returns true if the regex has matched and cannot possibly match anymore
        self.done
    }

    pub fn failed(&self) -> bool {
        // Returns true if the regex has failed to match and cannot possibly match
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
        self.dfa.states[current_state].transitions.get(byte).copied()
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

        assert!(!regex.definitely_matches(b"")); // Incomplete match not allowed
        assert!(regex.could_match(b"")); // Incomplete match allowed
        assert!(regex.definitely_matches(b"ab")); // Prefix match allowed
        assert!(regex.definitely_matches(b"aa")); // Prefix match allowed
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
        assert!(!state.done()); // Could match more 'a's
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

        assert!(regex.definitely_fully_matches(b"")); // Optional 'a' can be absent
        assert!(regex.definitely_fully_matches(b"a")); // Optional 'a' can be present
        assert!(!regex.could_fully_match(b"aa")); // Should not match more than one 'a'
        assert!(regex.could_match(b"b")); // Can still match the empty string in "b"
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
        assert!(regex.definitely_matches(b"a")); // Epsilon matches the empty string at the beginning
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
        assert!(regex.definitely_fully_matches(b"")); // Should match the empty option
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

        let expr = Expr::Choice(words.iter().map(|word| Expr::Seq(word.bytes().map(|c| Expr::U8Seq(vec![c])).collect())).collect());
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
        // group 0 should have the later match
        assert_eq!(state.matches, BTreeMap::from([(0, 2), (1, 1)]));
    }

    #[test]
    fn test_non_greedy_matching() {
        // Define regex: (a*)?a where group 0 is non-greedy and group 1 is greedy
        let expr = groups![
            non_greedy_group(rep(eat_u8(b'a'))), // Group 0: non-greedy
            eat_u8(b'a'),                         // Group 1: greedy
        ];

        let regex = expr.build();

        // Input: "aaa"
        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        // Expected:
        // Group 0 (non-greedy) should match "" (empty string)
        // Group 1 (greedy) should match "aaa"
        assert_eq!(regex_state.matches.get(&0), Some(&0));
        assert_eq!(regex_state.matches.get(&1), Some(&1));
    }

    #[test]
    fn test_greedy_matching() {
        // Define regex: (a*)a where group 0 is greedy and group 1 is greedy
        let expr = groups![
            rep(eat_u8(b'a')), // Group 0: greedy
            eat_u8(b'a'),      // Group 1: greedy
        ];

        let regex = expr.build();

        // Input: "aaa"
        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        // Expected:
        // Group 0 (greedy) should match "aaa"
        // Group 1 (greedy) should match "a"
        assert_eq!(regex_state.matches.get(&0), Some(&3));
        assert_eq!(regex_state.matches.get(&1), Some(&1));
    }

    #[test]
    fn test_triple_quoted_string() {
        // Regex: """.*?""" (non-greedy)
        let non_greedy_expr = groups![
            non_greedy_group(seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())),
                Expr::U8Seq(b"\"\"\"".to_vec())
            ])
        ];
        let non_greedy_regex = non_greedy_expr.build();

        // Regex: """.*""" (greedy)
        let greedy_expr = groups![
            seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())),
                Expr::U8Seq(b"\"\"\"".to_vec())
            ]
        ];
        let greedy_regex = greedy_expr.build();

        let input = b"\"\"\"hello\"\"\"world\"\"\"";

        // Non-greedy should match correctly
        let mut non_greedy_state = non_greedy_regex.init();
        non_greedy_state.execute(input);
        assert_eq!(non_greedy_state.matches.get(&0), Some(&b"\"\"\"hello\"\"\"".len())); // Matches up to the second """

        // Greedy should match incorrectly (matching the entire string)
        let mut greedy_state = greedy_regex.init();
        greedy_state.execute(input);
        assert_eq!(greedy_state.matches.get(&0), Some(&input.len())); // Matches the whole input
    }
}

#[cfg(test)]
mod possible_future_group_ids_tests {
    use super::*;

    fn run_test(expr: impl Into<ExprGroups>, expected_possible_future_group_ids: BTreeSet<GroupID>) {
        let regex = expr.into().build();
        let state = regex.init();
        assert_eq!(state.possible_future_group_ids(), expected_possible_future_group_ids);
    }

    #[test]
    fn test_possible_future_group_ids() {
        run_test(seq![], BTreeSet::new());
        run_test(eat_u8(b'a'), BTreeSet::from([0]));
        run_test(groups![eat_u8(b'a'), eat_u8(b'b')], BTreeSet::from([0, 1]));
        run_test(seq![eat_u8(b'a'), eat_u8(b'b')], BTreeSet::from([0]));
        run_test(rep(eat_u8(b'a')), BTreeSet::from([0]));
        run_test(groups![
            choice![opt(eat_u8(b'a')), rep(eat_u8(b'b')), eat_u8(b'c')],
            eat_u8(b'a'),
        ], BTreeSet::from([0, 1]));
        run_test(groups![
            eat_u8(b'a'),
            seq![eat_u8(b'a'), eat_u8(b'a')],
        ], BTreeSet::from([0, 1]));
    }

    #[test]
    fn test_possible_future_group_ids_excludes_current_state() {
        // Define a regex where the start state is final, but also has transitions
        // groups![eps(), eat_u8(b'a')]
        // Group 0: eps() -> makes start state final for group 0
        // Group 1: eat_u8(b'a') -> transition 'a' from start to a state final for group 1
        let expr = groups![
            eps(),        // Group 0
            eat_u8(b'a'), // Group 1
        ];
        let regex = expr.build();
        let start_state_index = regex.dfa.start_state;
        let start_state_data = &regex.dfa.states[start_state_index];

        // possible_future_group_ids should only contain group 1, as it's reachable via a transition ('a').
        // It should *not* contain group 0, which is final *at* the current state.
        assert_eq!(start_state_data.possible_future_group_ids, BTreeSet::from([1]));
    }
}

#[cfg(test)]
mod group_id_to_u8set_tests {
    use super::*;

    /// Helper function to create a DFA from an expression with multiple groups.
    fn build_dfa_with_groups(exprs: Vec<Expr>) -> Regex {
        let expr_groups = ExprGroups {
            groups: exprs.into_iter().map(ExprGroup::from).collect(),
        };
        expr_groups.build()
    }

    #[test]
    fn test_compute_group_id_to_u8set_single_group() {
        // Regex: "a"
        let expr = groups![
            eat_u8(b'a') // Group 0
        ];
        let regex = expr.build();

        // State 0 transitions to state 1 on 'a'
        // group_id_to_u8set for state 0 should map group 0 to {'a'}
        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 1);
        assert!(group_id_to_u8set.contains_key(&0));
        let u8set = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set.contains(b'a'));
        assert_eq!(u8set.iter().collect::<Vec<u8>>(), vec![b'a']);
    }

    #[test]
    fn test_compute_group_id_to_u8set_multiple_groups() {
        // Regex: "a" or "b"
        let expr = groups![
            eat_u8(b'a'), // Group 0
            eat_u8(b'b'), // Group 1
        ];
        let regex = expr.build();

        // State 0 transitions on 'a' to state 1 and on 'b' to state 2
        // group_id_to_u8set for state 0 should map:
        // - Group 0 to {'a'}
        // - Group 1 to {'b'}

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
        // Regex: "a" or "a"
        let expr = groups![
            eat_u8(b'a'), // Group 0
            eat_u8(b'a'), // Group 1
        ];
        let regex = expr.build();

        // State 0 transitions on 'a' to state 1 and state 2
        // group_id_to_u8set for state 0 should map:
        // - Group 0 to {'a'}
        // - Group 1 to {'a'}

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
        // Regex: "a" or "b"
        let expr = groups![
            eat_u8(b'a'), // Group 0
            eat_u8(b'b'), // Group 1
        ];
        let regex = expr.build();

        let regex_state = regex.init();

        // For Group 0, U8Set should contain 'a'
        let u8set_group0 = regex_state.get_u8set_for_group(0);
        assert!(u8set_group0.contains(b'a'));
        assert_eq!(u8set_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        // For Group 1, U8Set should contain 'b'
        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert!(u8set_group1.contains(b'b'));
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_get_u8set_for_group_nonexistent_group() {
        // Regex: "a"
        let expr = groups![
            eat_u8(b'a') // Group 0
        ];
        let regex = expr.build();

        let regex_state = regex.init();

        // For non-existent Group 1, U8Set should be empty
        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), Vec::<u8>::new());
    }

    #[test]
    fn test_group_id_to_u8set_nested_groups() {
        // Regex: (a|b)*c
        let expr = groups![
            rep(choice![eat_u8(b'a'), eat_u8(b'b')]), // Group 0
            eat_u8(b'c'),                           // Group 1
        ];
        let regex = expr.build();

        // Start state (state 0)
        // group_id_to_u8set for state 0:
        // - Group 0: {'a', 'b'}
        // - Group 1: {'c'}

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
        // Regex: "a"
        let expr = groups![
            eat_u8(b'a') // Group 0
        ];
        let regex = expr.build();

        // Attempt to get U8Set for non-existent Group 1
        let regex_state = regex.init();
        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), Vec::<u8>::new());
    }

    #[test]
    fn test_group_id_to_u8set_overlapping_groups() {
        // Regex: "a" or "a"
        let expr = groups![
            eat_u8(b'a'), // Group 0
            eat_u8(b'a'), // Group 1
        ];
        let regex = expr.build();

        // Start state (state 0)
        // group_id_to_u8set for state 0:
        // - Group 0: {'a'}
        // - Group 1: {'a'}

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
        // Regex: "ab" or "ac"
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')], // Group 0
            seq![eat_u8(b'a'), eat_u8(b'c')], // Group 1
        ];
        let regex = expr.build();

        // Start state (state 0)
        // group_id_to_u8set for state 0 should map:
        // - Group 0: {'a'}
        // - Group 1: {'a'}

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 2);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));
        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();
        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));

        // After consuming 'a', move to state(s) corresponding to 'ab' and 'ac'
        let mut regex_state = regex.init();
        regex_state.execute(b"a");

        // Now, current_state should be one of the states after 'a' (say, state 1 and 2)
        // For simplicity, assuming DFA has merged states, but depending on implementation, adjust accordingly

        // Let's assume state 1 and 2 are separate for "ab" and "ac"

        // For this test, we'll iterate through possible transitions

        // Verify that in both resulting states, possible_future_group_ids contain their respective groups
        // Here, it's likely that the DFA has merged states if they share the same possible_future_group_ids
        // For this test, we'll assume separate states

        // Since the DFA construction merges states with identical possible_future_group_ids, in this case:
        // - After 'a', possible_future_group_ids should still include {0,1} because both 'ab' and 'ac' can follow.

        // Verify possible_future_group_ids
        assert_eq!(
            regex.dfa.states[regex_state.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1])
        );

        // Verify group_id_to_u8set for the new state
        let group_id_to_u8set_new = &regex.dfa.states[regex_state.current_state].group_id_to_u8set;
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
        // Regex: "abc" or "abd" or "abe"
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'c')], // Group 0
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'd')], // Group 1
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'e')], // Group 2
        ];
        let regex = expr.build();

        // Start state (state 0)
        // group_id_to_u8set for state 0:
        // - Group 0: {'a'}
        // - Group 1: {'a'}
        // - Group 2: {'a'}

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

        // After consuming 'a', move to state after 'a'
        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");

        // possible_future_group_ids should still include {0,1,2}
        assert_eq!(
            regex.dfa.states[regex_state_a.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1, 2])
        );

        // group_id_to_u8set should map:
        // - Group 0: 'b'
        // - Group 1: 'b'
        // - Group 2: 'b'

        let group_id_to_u8set_a = &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
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

        // After consuming 'a' and 'b', move to state after 'a' and 'b'
        let mut regex_state_ab = regex.init();
        regex_state_ab.execute(b"ab");

        // possible_future_group_ids should still include {0,1,2}
        assert_eq!(
            regex.dfa.states[regex_state_ab.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1, 2])
        );

        // group_id_to_u8set should map:
        // - Group 0: 'c'
        // - Group 1: 'd'
        // - Group 2: 'e'

        let group_id_to_u8set_ab = &regex.dfa.states[regex_state_ab.current_state].group_id_to_u8set;
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
        // Regex: "ab"
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')] // Group 0
        ];
        let regex = expr.build();

        // Start state (state 0)
        // group_id_to_u8set for state 0:
        // - Group 0: {'a'}

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 1);
        assert!(group_id_to_u8set_0.contains_key(&0));

        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        assert!(u8set_0_group0.contains(b'a'));
        assert_eq!(u8set_0_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        // After consuming 'a', move to state after 'a'
        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");
        assert_eq!(
            regex.dfa.states[regex_state_a.current_state].possible_future_group_ids,
            BTreeSet::from([0])
        );

        // group_id_to_u8set for state after 'a':
        // - Group 0: {'b'}
        let group_id_to_u8set_a = &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_a.len(), 1);
        assert!(group_id_to_u8set_a.contains_key(&0));

        let u8set_a_group0 = group_id_to_u8set_a.get(&0).unwrap();
        assert!(u8set_a_group0.contains(b'b'));
        assert_eq!(u8set_a_group0.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_get_u8set_for_group_multiple_transitions() {
        // Regex: "a" followed by "b" or "c"
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')], // Group 0
            seq![eat_u8(b'a'), eat_u8(b'c')], // Group 1
        ];
        let regex = expr.build();

        // Start state (state 0)
        // group_id_to_u8set for state 0
        // - Group 0: {'a'}
        // - Group 1: {'a'}

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

        // After consuming 'a', current_state should have group_id_to_u8set:
        // - Group 0: {'b'}
        // - Group 1: {'c'}
        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");

        let group_id_to_u8set_a = &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
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
        // Construct DFA manually with known states and transitions.
        //
        // States:
        // 0: start state
        // 1: after 'a'
        // 2: after 'ab' (accepting state for group 0)
        // 3: after 'ac' (accepting state for group 1)
        //
        // Transitions:
        // 0 -- 'a' --> 1
        // 1 -- 'b' --> 2
        // 1 -- 'c' --> 3
        //
        // Group IDs:
        // State 2: group 0 (accepts "ab")
        // State 3: group 1 (accepts "ac")
        //
        // The DFA recognizes the tokens "ab" and "ac".

        // Initialize the DFA with empty states.
        let mut dfa = DFA {
            states: Vec::new(),
            start_state: 0,
            non_greedy_finalizers: BTreeSet::new(), // Initialize here
        };

        // State 0: Start state
        dfa.states.push(DFAState {
            transitions: TrieMap::new(),
            finalizers: BTreeSet::new(),
            possible_future_group_ids: BTreeSet::new(), // Will be computed
            group_id_to_u8set: BTreeMap::new(),   // Will be computed
        });

        // State 1: After reading 'a'
        dfa.states.push(DFAState {
            transitions: TrieMap::new(),
            finalizers: BTreeSet::new(),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        // State 2: Accepting state for group 0 ("ab")
        dfa.states.push(DFAState {
            transitions: TrieMap::new(),
            finalizers: BTreeSet::from([0]),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        // State 3: Accepting state for group 1 ("ac")
        dfa.states.push(DFAState {
            transitions: TrieMap::new(),
            finalizers: BTreeSet::from([1]),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        // Add transitions:
        // State 0 -- 'a' --> State 1
        dfa.states[0].transitions.insert(b'a', 1);

        // State 1 -- 'b' --> State 2
        dfa.states[1].transitions.insert(b'b', 2);

        // State 1 -- 'c' --> State 3
        dfa.states[1].transitions.insert(b'c', 3);

        // Compute possible_future_group_ids and group_id_to_u8set for the DFA
        dfa.compute_possible_future_group_ids();
        dfa.compute_group_id_to_u8set();

        // Create a Regex instance with the constructed DFA
        let regex = Regex { dfa };

        // Test get_u8set_for_group at State 0 (start state)
        let state0 = regex.init_to_state(0);
        let u8set_group0_state0 = state0.get_u8set_for_group(0);
        let u8set_group1_state0 = state0.get_u8set_for_group(1);

        // At State 0, possible inputs leading to group 0 or group 1 are 'a' (which can lead to 'ab' or 'ac')
        // Therefore, both group 0 and group 1 should have 'a' in their U8Set at State 0
        assert!(u8set_group0_state0.contains(b'a'));
        assert!(u8set_group1_state0.contains(b'a'));

        // Test get_u8set_for_group at State 1
        let state1 = regex.init_to_state(1);
        let u8set_group0_state1 = state1.get_u8set_for_group(0);
        let u8set_group1_state1 = state1.get_u8set_for_group(1);

        // At State 1:
        // - For group 0 ("ab"), the next input must be 'b'
        // - For group 1 ("ac"), the next input must be 'c'
        // So group 0's U8Set should contain 'b', and group 1's U8Set should contain 'c'
        assert!(u8set_group0_state1.contains(b'b'));
        assert!(!u8set_group0_state1.contains(b'c'));
        assert!(u8set_group1_state1.contains(b'c'));
        assert!(!u8set_group1_state1.contains(b'b'));

        // Test get_u8set_for_group at State 2 (accepting state for group 0)
        let state2 = regex.init_to_state(2);
        let u8set_group0_state2 = state2.get_u8set_for_group(0);
        let u8set_group1_state2 = state2.get_u8set_for_group(1);

        // At State 2, there are no outgoing transitions
        // So both group 0 and group 1 should have empty U8Sets
        assert!(u8set_group0_state2.iter().next().is_none());
        assert!(u8set_group1_state2.iter().next().is_none());

        // Test get_u8set_for_group at State 3 (accepting state for group 1)
        let state3 = regex.init_to_state(3);
        let u8set_group0_state3 = state3.get_u8set_for_group(0);
        let u8set_group1_state3 = state3.get_u8set_for_group(1);

        // At State 3, there are no outgoing transitions
        // So both group 0 and group 1 should have empty U8Sets
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
        // The following expression should yield a DFA with 2 states:
        // - one to match 'a' or 'b'
        // - one to hold the finalizer
        let expr = choice![
            eat_u8(b'a'),
            eat_u8(b'b'),
        ];
        let regex = expr.build();
        dbg!(&regex);
        assert_eq!(regex.dfa.states.len(), 2);
    }
}

#[cfg(test)]
mod test_python {
    use super::*;
    use crate::datastructures::u8set::U8Set; // Added for U8Set usage in new test
    use crate::{choice, seq};

    #[ignore]
    #[test]
    fn test_full_python_tokenizer_recognizes_name() {
        // --- Define basic character sets ---
        let digit = Expr::U8Class(U8Set::from_range(b'0', b'9'));
        let alph_lower = Expr::U8Class(U8Set::from_range(b'a', b'z'));
        let alph_upper = Expr::U8Class(U8Set::from_range(b'A', b'Z'));
        let name_start = choice![alph_lower.clone(), alph_upper.clone(), eat_u8(b'_')];
        let name_middle = choice![name_start.clone(), digit.clone()];

        // --- Define the ignore pattern ---
        // Simplified ignore for testing: just space
        // let ignore = rep(eat_u8(b' '));
        // More complete ignore: space or # comment
        let ignore = rep(choice![
             eat_u8(b' '),
             // Basic comment handling for the test
             seq![eat_u8(b'#'), rep(Expr::U8Class(U8Set::all().without(b'\n'))), opt(eat_u8(b'\n'))],
             // Note: Real Python tokenizer handles more complex whitespace and line continuations
         ]);

        // --- Define token expressions (core logic) ---
        // Based on python_grammar.py and python.gram literals
        let tokens_core: BTreeMap<&str, Expr> = BTreeMap::from([
            // Core Types
            ("NAME", seq![name_start, rep(name_middle)]),
            ("NUMBER", choice![
                rep1(digit.clone()), // Integer
                seq![rep1(digit.clone()), eat_u8(b'.'), rep(digit.clone())], // Float with digits before .
                seq![eat_u8(b'.'), rep1(digit.clone())], // Float starting with .
                // Simplified: Not including hex, oct, bin, complex for this test focus
            ]),
            ("STRING", choice![
                seq![eat_u8(b'"'), rep(Expr::U8Class(U8Set::all().without(b'"'))), eat_u8(b'"')],
                seq![eat_u8(b'\''), rep(Expr::U8Class(U8Set::all().without(b'\''))), eat_u8(b'\'')],
                // Simplified: Not including triple quotes, prefixes like r"", f"", etc.
            ]),
            ("FSTRING_START", Expr::U8Seq(b"f'".to_vec())), // Example, needs more variants
            ("FSTRING_END", Expr::U8Seq(b"'".to_vec())),    // Example
            ("FSTRING_MIDDLE", rep1(Expr::U8Class(U8Set::all().difference(&U8Set::from_slice(&[b'{', b'}']))))), // Simplified
            // Special Tokens (often handled by parser state, represented as eps here for DFA structure)
            ("NEWLINE", eps()),
            ("INDENT", eps()),
            ("DEDENT", eps()),
            ("TYPE_COMMENT", eps()),
            ("ENDMARKER", eps()),
            // Operators and Delimiters from python.gram
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
            ("EXCLAMATION", eat_u8(b'!')), // For f-string conversion
            // Compound Operators
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

        // --- Combine with ignore and create groups ---
        let mut token_groups: Vec<ExprGroup> = Vec::new();
        let mut token_name_to_id: BTreeMap<&str, GroupID> = BTreeMap::new();
        for (name, core_expr) in tokens_core {
            let group_id = token_groups.len();
            token_name_to_id.insert(name, group_id);
            // Use greedy groups by default for tokenizer behavior
            token_groups.push(greedy_group(seq![ignore.clone(), core_expr]));
        }

        let expr_groups = groups(token_groups);
        let regex = expr_groups.build();
        // dbg!(ex); // Uncomment for debugging DFA structure

        let mut state = regex.init();
        state.execute(b"hello");

        assert!(state.definitely_matches(), "Tokenizer should match 'hello'");
        // Ensure there is a match for the NAME token and that it's at the correct position
        assert_eq!(state.matches.get(&token_name_to_id["NAME"]), Some(&5), "NAME token should be matched at position 5");
    }
}
