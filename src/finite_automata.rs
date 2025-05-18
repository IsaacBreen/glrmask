use crate::datastructures::charmap::TrieMap;
use crate::datastructures::frozenset::FrozenSet;
use crate::datastructures::u8set::U8Set;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Formatter};
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


#[derive(Clone)]
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


#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
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


#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
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

impl Debug for NFA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Regex State NFA:\n")?;

        for (state_index, state) in self.states.iter().enumerate() {
            f.write_str(&format!("State {}:\n", state_index))?;

            for (transition_u8, next_states) in &state.transitions {
                f.write_str(&format!("  - '{}': {:?}\n", *transition_u8 as char, next_states))?; // Display u8 as char
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

impl Debug for DFA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Regex State DFA:\n")?;

        for (state_index, state) in self.states.iter().enumerate() {
            f.write_str(&format!("State {}:\n", state_index))?;

            for (transition_u8, next_state) in &state.transitions {
                f.write_str(&format!("  - {} ({:?}): {}\n", transition_u8, *transition_u8 as char, next_state))?; // Display u8 as char
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
            Expr::Quantifier(expr_box, quantifier_type) => { // Renamed expr to expr_box
                match quantifier_type {
                    QuantifierType::ZeroOrMore => {
                        let loop_start_state = nfa.add_state();
                        let loop_end_state = nfa.add_state();

                        // Epsilon transition from current state to loop start state
                        nfa.add_epsilon_transition(current_state, loop_start_state);

                        // Process the expr
                        let expr_end_state = Self::handle_expr(*expr_box, nfa, loop_start_state); // Use expr_box

                        // Epsilon transition from expr end state back to loop start state for repetition
                        nfa.add_epsilon_transition(expr_end_state, loop_start_state);

                        // Epsilon transition from loop start state to loop end state to allow skipping
                        nfa.add_epsilon_transition(loop_start_state, loop_end_state);

                        // The loop end state becomes the new current state
                        loop_end_state
                    }
                    QuantifierType::OneOrMore => {
                        // For A+, this is A A*
                        // current_state --A--> expr_end_state --epsilon--> loop_start_state --A--> expr_end_state_rep --epsilon--> loop_start_state
                        //                                                                    `----epsilon---> loop_end_state (final for A+)
                        // Simpler: current_state --A--> expr_end_state (this is the start of the loop for A*)
                        //          expr_end_state --epsilon--> loop_start_for_A_star
                        //          loop_start_for_A_star --A--> expr_end_state_of_A_star --epsilon--> loop_start_for_A_star
                        //          loop_start_for_A_star --epsilon--> final_state_for_A_plus

                        // Process the first A
                        let first_a_end_state = Self::handle_expr(*expr_box.clone(), nfa, current_state); // Clone expr_box for first A

                        // Now, handle A* starting from first_a_end_state
                        let star_loop_start = nfa.add_state();
                        let star_final_state = nfa.add_state();

                        nfa.add_epsilon_transition(first_a_end_state, star_loop_start); // Connect first A's end to A*'s loop start

                        // Inside A*'s loop
                        let star_expr_end = Self::handle_expr(*expr_box, nfa, star_loop_start); // Use expr_box for A* part
                        nfa.add_epsilon_transition(star_expr_end, star_loop_start); // Loop back

                        // Path to skip A* (effectively meaning only one A was matched)
                        nfa.add_epsilon_transition(star_loop_start, star_final_state);
                        // Also, if the A* loop is taken, its end should also go to the final state
                        nfa.add_epsilon_transition(star_expr_end, star_final_state); // This might be redundant if star_expr_end loops to star_loop_start which goes to star_final_state

                        // The final state of A+ is star_final_state
                        star_final_state
                    }
                    QuantifierType::ZeroOrOne => {
                        let optional_end_state = nfa.add_state();

                        // Epsilon transition from current state to optional end state to allow skipping
                        nfa.add_epsilon_transition(current_state, optional_end_state);

                        // Process the expr
                        let expr_end_state = Self::handle_expr(*expr_box, nfa, current_state); // Use expr_box

                        // Epsilon transition from expr end state to optional end state
                        nfa.add_epsilon_transition(expr_end_state, optional_end_state);

                        // The optional end state becomes the new current state
                        optional_end_state
                    }
                }
            }
            Expr::Choice(exprs) => {
                let choice_start_state = nfa.add_state(); // New start state for choice
                let choice_end_state = nfa.add_state();   // New end state for choice

                // Epsilon transition from the current state to the start state of the choice
                nfa.add_epsilon_transition(current_state, choice_start_state);

                for expr_val in exprs { // Renamed expr to expr_val
                    // For each expr, connect the start state of the choice to the start state of the expr
                    // No, each branch starts from choice_start_state directly
                    // let expr_start_state = nfa.add_state(); // This was creating an extra unnecessary state per branch
                    // nfa.add_epsilon_transition(choice_start_state, expr_start_state);

                    // Process the expr and get its end state, starting from choice_start_state
                    let expr_end_state = Self::handle_expr(expr_val, nfa, choice_start_state); // Start each choice from choice_start_state

                    // Connect the end state of the expr to the end state of the choice
                    nfa.add_epsilon_transition(expr_end_state, choice_end_state);
                }

                // The end state of the choice becomes the new current state
                choice_end_state
            }
            Expr::Seq(exprs) => {
                for expr_val in exprs { // Renamed expr to expr_val
                    current_state = Self::handle_expr(expr_val, nfa, current_state);
                }
                current_state
            }
            Expr::Epsilon => {
                let new_state = nfa.add_state();
                nfa.add_epsilon_transition(current_state, new_state);
                new_state
            }
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

        let epsilon_closures = self.compute_epsilon_closures();

        // Compute the epsilon closure of the NFA start state and use it as the DFA start state
        let start_closure_set = epsilon_closures[self.start_state].clone(); // BTreeSet
        let start_state_frozen_set = FrozenSet::from_iter(start_closure_set.iter().cloned()); // Convert to FrozenSet
        worklist.push(start_state_frozen_set.clone());
        dfa_state_map.insert(start_state_frozen_set.clone(), 0);

        // Initialize the first DFA state
        let mut finalizers = BTreeSet::new();
        // let mut non_greedy_finalizers = BTreeSet::new(); // Not directly used in DFAState finalizers field
        for &state in &start_closure_set { // Iterate over BTreeSet
            finalizers.extend(self.states[state].finalizers.iter().cloned());
            // non_greedy_finalizers.extend(self.states[state].non_greedy_finalizers.iter().cloned());
        }

        dfa_states.push(DFAState {
            transitions: TrieMap::new(),
            finalizers,
            possible_future_group_ids: BTreeSet::new(), // Will be computed later
            group_id_to_u8set: BTreeMap::new(),  // Will be computed later
        });

        let mut dfa_idx = 0;
        while dfa_idx < worklist.len() { // Process as a queue
            let current_frozen_set = worklist[dfa_idx].clone(); // Clone to avoid borrow issues
            dfa_idx += 1;

            let current_dfa_state_idx = *dfa_state_map.get(&current_frozen_set).unwrap();
            let mut transition_map_for_dfa_state: BTreeMap<u8, BTreeSet<usize>> = BTreeMap::new();

            // For each NFA state in the current DFA state's frozen set
            for &nfa_state_in_set in current_frozen_set.iter() { // Iterate FrozenSet
                for (input_char, next_nfa_states_vec) in &self.states[nfa_state_in_set].transitions {
                    for &next_nfa_state in next_nfa_states_vec {
                        transition_map_for_dfa_state
                            .entry(*input_char) // Deref input_char
                            .or_insert_with(BTreeSet::new)
                            .extend(&epsilon_closures[next_nfa_state]); // Add all states reachable by epsilon from next_nfa_state
                    }
                }
            }
            
            for (input_char, target_nfa_closure_set) in transition_map_for_dfa_state { // target_nfa_closure_set is BTreeSet
                let target_frozen_set = FrozenSet::from_iter(target_nfa_closure_set.iter().cloned()); // Convert to FrozenSet
                let next_dfa_state_idx = if let Some(&existing_idx) = dfa_state_map.get(&target_frozen_set) {
                    existing_idx
                } else {
                    let new_idx = dfa_states.len();
                    dfa_state_map.insert(target_frozen_set.clone(), new_idx);
                    worklist.push(target_frozen_set); // Add to worklist

                    let mut new_finalizers = BTreeSet::new();
                    for &nfa_state in &target_nfa_closure_set { // Iterate BTreeSet
                        new_finalizers.extend(self.states[nfa_state].finalizers.iter().cloned());
                    }
                    dfa_states.push(DFAState {
                        transitions: TrieMap::new(),
                        finalizers: new_finalizers,
                        possible_future_group_ids: BTreeSet::new(),
                        group_id_to_u8set: BTreeMap::new(),
                    });
                    new_idx
                };
                dfa_states[current_dfa_state_idx].transitions.insert(input_char, next_dfa_state_idx);
            }
        }


        let mut dfa = DFA {
            states: dfa_states,
            start_state: 0, // DFA start state is always 0 after this construction
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

        while let Some(s) = stack.pop() { // Renamed state to s
            if closure.insert(s) {
                stack.extend(&self.states[s].epsilon_transitions);
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
                let state_clone = self.states[state_index].clone(); // Clone to avoid borrow checker issues
                for (_input, &next_state_index) in &state_clone.transitions {
                    let next_possible_future_groups = self.states[next_state_index].possible_future_group_ids.clone();
                    let next_finalizers = self.states[next_state_index].finalizers.clone();
                    
                    let current_state_possible_future_groups = &mut self.states[state_index].possible_future_group_ids; // Mutable borrow here
                    let old_len = current_state_possible_future_groups.len();
                    
                    current_state_possible_future_groups.extend(next_finalizers.iter()); // Add finalizers of the *next* state
                    current_state_possible_future_groups.extend(next_possible_future_groups.iter());

                    if current_state_possible_future_groups.len() > old_len {
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
        for state_idx in 0..self.states.len() { // Iterate by index to mutably borrow
            let mut group_id_to_u8set_for_current_state: BTreeMap<GroupID, U8Set> = BTreeMap::new();
            
            // Need to clone transitions to avoid borrowing self.states[state_idx] while modifying it
            let transitions_clone = self.states[state_idx].transitions.clone();

            for (input_u8, &next_state_index) in &transitions_clone {
                let next_possible_ids = &possible_current_or_future_group_ids[next_state_index];

                for &group_id in next_possible_ids {
                    group_id_to_u8set_for_current_state
                        .entry(group_id)
                        .or_insert_with(U8Set::none)
                        .insert(*input_u8); // Deref input_u8
                }
            }
            self.states[state_idx].group_id_to_u8set = group_id_to_u8set_for_current_state;
        }
    }

    fn remove_unreachable_states(&mut self) {
        // Find reachable states using BFS
        let mut reachable = vec![false; self.states.len()];
        let mut queue = vec![self.start_state];
        if !self.states.is_empty() { // Guard against empty states vec
             reachable[self.start_state] = true;
        } else {
            return; // No states to process
        }


        let mut head = 0; // Use index for queue to avoid borrow issues
        while head < queue.len() {
            let state_idx = queue[head]; // Renamed state to state_idx
            head += 1;

            // Clone transitions to iterate without borrowing self.states[state_idx]
            let transitions_clone = self.states[state_idx].transitions.clone();
            for &next_state in transitions_clone.values() {
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
        for (old_index, state_val) in self.states.iter().enumerate() { // Renamed state to state_val
            if reachable[old_index] {
                let mut new_state = state_val.clone();
                // Update transitions to use new state indices
                new_state.transitions = new_state.transitions
                    .iter()
                    .map(|(u8_val, &next)| (*u8_val, state_mapping[next])) // Deref u8_val
                    .collect();
                new_states.push(new_state);
            }
        }

        // Update the DFA
        self.states = new_states;
        // Start state remains at 0 since it's always reachable (if states is not empty)
        if !self.states.is_empty() {
            self.start_state = 0;
        }
    }

    fn minimize(&mut self) {
        if self.states.is_empty() {
            return;
        }

        // Step 1: Create initial partition based on finalizers and transitions
        let mut partitions = BTreeMap::<(BTreeSet<GroupID>, BTreeMap<u8, usize>), BTreeSet<usize>>::new();

        for (state_idx, state_val) in self.states.iter().enumerate() { // Renamed state to state_val
            let key = (
                state_val.finalizers.clone(),
                state_val.transitions.iter().map(|(u8_val, &next)| (*u8_val, next)).collect() // Deref u8_val
            );
            partitions.entry(key).or_default().insert(state_idx);
        }

        // Step 2: Refine partitions until no more refinement is possible
        let mut partition_list: Vec<BTreeSet<usize>> = partitions.into_values().collect();
        let mut changed = true;
        while changed {
            changed = false;
            let mut new_partitions_next_iter = Vec::new(); // Renamed new_partitions

            for partition in &partition_list {
                let mut refined_partitions_for_current_partition = BTreeMap::new(); // Renamed refined_partitions

                for &state_idx in partition { // Renamed state to state_idx
                    let mut signature = BTreeMap::new();

                    // For each transition, record which partition it leads to
                    // Clone transitions to iterate without borrowing self.states[state_idx]
                    let transitions_clone = self.states[state_idx].transitions.clone();
                    for (u8_val, &next_state) in &transitions_clone { // Deref u8_val
                        let target_partition_idx = partition_list.iter() // Search in current partition_list
                            .position(|p| p.contains(&next_state))
                            .unwrap_or(usize::MAX); // Should always find if DFA is consistent
                        signature.insert(*u8_val, target_partition_idx); // Deref u8_val
                    }

                    refined_partitions_for_current_partition.entry(signature)
                        .or_insert_with(BTreeSet::new)
                        .insert(state_idx);
                }

                if refined_partitions_for_current_partition.len() > 1 {
                    changed = true;
                    new_partitions_next_iter.extend(refined_partitions_for_current_partition.into_values());
                } else {
                    new_partitions_next_iter.push(partition.clone());
                }
            }
            partition_list = new_partitions_next_iter;
        }


        // Step 3: Build the minimized DFA
        let mut state_mapping = vec![0; self.states.len()];

        // Find which partition contains the start state
        let start_partition_idx = partition_list.iter()
            .position(|p| p.contains(&self.start_state))
            .unwrap_or_else(|| { // Handle case where start_state might not be in any partition (e.g. if states is empty initially)
                if self.states.is_empty() { 0 } else { panic!("Start state not found in any partition"); }
            });


        // Ensure the start partition is first in the list
        if start_partition_idx != 0 && !partition_list.is_empty() { // Add check for non-empty list
            partition_list.swap(0, start_partition_idx);
        }


        // Build state mapping
        for (new_state_idx, partition) in partition_list.iter().enumerate() { // Renamed new_state
            for &old_state_idx in partition { // Renamed old_state
                state_mapping[old_state_idx] = new_state_idx;
            }
        }

        let mut new_states_vec = Vec::with_capacity(partition_list.len()); // Renamed new_states
        for partition in &partition_list {
            let old_state_idx = partition.iter().next().unwrap(); // Renamed old_state
            let mut new_state_val = self.states[*old_state_idx].clone(); // Renamed new_state

            // Update transitions according to the new state mapping
            new_state_val.transitions = new_state_val.transitions
                .iter()
                .map(|(u8_val, &next)| (*u8_val, state_mapping[next])) // Deref u8_val
                .collect();

            new_states_vec.push(new_state_val);
        }

        // The start state should now be at index 0
        if !new_states_vec.is_empty() { // Guard against empty new_states_vec
            self.start_state = 0;
        }
        self.states = new_states_vec;
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
        let mut current_text_offset = 0; // Renamed local_position to avoid confusion with self.position
        // self.position = 0; // This seems incorrect, self.position should track overall progress.
                           // Let's assume self.position is the global start for this call.

        loop {
            if current_text_offset >= text.len() { // Consumed all input for this call
                if terminate {
                    if let Some(m) = self.get_greedy_match() {
                        matches.push(Match { group_id: m.group_id, position: m.position - start_position }); // Adjust position relative to text start
                    }
                    self.end();
                }
                // self.position = start_position + current_text_offset; // Update global position
                return matches;
            }

            // Execute on the remaining part of the text
            self.execute(&text[current_text_offset..]);

            if self.ended() {
                if let Some(m) = self.get_greedy_match() {
                    // m.position is absolute. We need to find the length of this match.
                    // The match started at current_text_offset (relative to `text` start).
                    // The match ended at m.position (absolute).
                    // The length of the match is m.position - (self.position - (text.len() - current_text_offset))
                    // This is getting complicated. Let's simplify:
                    // get_greedy_match returns absolute position.
                    // The match we care about is from current_text_offset.
                    // If a match is found, its reported position is from the beginning of the *entire stream*.
                    // We need to adjust it to be relative to the start of `text` for this call.
                    let match_end_absolute = m.position;
                    let match_start_absolute = self.position - (text.len() - current_text_offset); // Start of the current execute call
                    let match_len = match_end_absolute - match_start_absolute;


                    matches.push(Match { group_id: m.group_id, position: current_text_offset + match_len }); // Position relative to `text` start
                    current_text_offset += match_len; // Advance by the length of the match

                    self.reset(); // Reset for the next potential match
                    self.position = start_position + current_text_offset; // Update global position for next reset
                } else {
                    // Ended but no match. This indicates a tokenization error or end of useful input.
                    // self.position = start_position + current_text_offset; // Update global position
                    return matches;
                }
            } else {
                // Didn't end. We must have run out of input for the current execute call.
                if terminate {
                    if let Some(m) = self.get_greedy_match() {
                         matches.push(Match { group_id: m.group_id, position: m.position - start_position }); // Adjust position
                    }
                    self.end();
                }
                // self.position = start_position + current_text_offset; // Update global position
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
        if self.matches.is_empty() { // Check is_empty()
            return None;
        }
        let mut matches_iter = self.matches.iter(); // Renamed matches to matches_iter
        let (mut longest_match_group_id, mut longest_match_position) = matches_iter.next().unwrap();
        for (group_id, position) in matches_iter { // Use matches_iter
            if position > longest_match_position {
                longest_match_group_id = group_id;
                longest_match_position = position;
            } else if position == longest_match_position && group_id < longest_match_group_id { // Added condition for lowest group ID
                longest_match_group_id = group_id;
                // longest_match_position remains the same
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
                u8set.insert(*value); // Deref value
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
        let expr = choice![eat_u8(b'a'), seq![]]; // seq![] is an epsilon
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
        assert!(regex.could_fully_match(b"")); // This was false, rep1(a) needs at least one 'a'.
                                              // However, rep(a) can be empty, then rep1(a) matches "a".
                                              // So "" is not a full match.
                                              // If input is "", rep(a) matches "", rep1(a) fails.
                                              // If input is "a", rep(a) matches "", rep1(a) matches "a". Full match.
                                              // If input is "a", rep(a) matches "a", rep1(a) fails.
                                              // The DFA should handle this.
                                              // Let's trace: "" -> rep(a) matches, rep1(a) fails. No full match.
                                              // "a" -> rep(a) (empty) then rep1(a) matches "a". Full match.
        assert!(!regex.definitely_fully_matches(b"")); // Corrected assertion
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

        state.execute(b"a"); // This appends "a" to the previous "a", so input becomes "aa"
        // After "aa":
        // Group 0 ("a") still matches at position 1 (the first "a").
        // Group 1 ("aa") matches at position 2 (the full "aa").
        assert_eq!(state.matches, BTreeMap::from([(0, 1), (1, 2)]));
    }

    #[test]
    fn test_multiple_finalizers_greedy() {
        let expr = groups![
            rep(eat_u8(b'a')), // Group 0: a*
            eat_u8(b'a'),      // Group 1: a
        ];

        let regex = expr.build();
        dbg!(&regex);

        let mut state = regex.init();

        state.execute(b"aa");
        // Input "aa":
        // Group 0 (a*): matches "aa" at pos 2. (Greedy)
        // Group 1 (a): matches "a" at pos 1.
        // If both are greedy, the longest match for each group is taken.
        // For group 0, "aa" is the longest.
        // For group 1, "a" (first one) is the only one.
        // The behavior of overlapping greedy groups can be subtle.
        // Let's trace the DFA:
        // State 0 --a--> State 1 (final for G1, final for G0)
        // State 1 --a--> State 1 (final for G1, final for G0)
        //
        // Input "a": current_state becomes 1. matches: {0:1, 1:1}
        // Input "aa":
        //   First "a": current_state becomes 1. matches: {0:1, 1:1}
        //   Second "a": current_state remains 1. matches: {0:2 (updated), 1:1 (first 'a' still holds for G1)}
        // This depends on how `matches` are updated. If G1 is only set once.
        // If G1 can be updated: matches: {0:2, 1:2}
        // The current `execute` logic:
        // `self.matches.insert(group_id, self.position + local_position);` for greedy.
        //
        // After "a": pos=0, local_pos=0. text="aa"
        //   next_u8 = 'a', local_pos=0. current_state=0 -> 1. local_pos=1.
        //   finalizers for state 1: {0,1}.
        //   matches.insert(0, 0+1=1)
        //   matches.insert(1, 0+1=1)
        //   matches = {0:1, 1:1}
        //
        //   next_u8 = 'a', local_pos=1. current_state=1 -> 1. local_pos=2.
        //   finalizers for state 1: {0,1}.
        //   matches.insert(0, 0+2=2) -> matches = {0:2, 1:1}
        //   matches.insert(1, 0+2=2) -> matches = {0:2, 1:2}
        assert_eq!(state.matches, BTreeMap::from([(0, 2), (1, 2)]));
    }

    #[test]
    fn test_non_greedy_matching() {
        // Define regex: (a*)?a where group 0 is non-greedy and group 1 is greedy
        let expr = groups![
            non_greedy_group(rep(eat_u8(b'a'))), // Group 0: non-greedy a*
            eat_u8(b'a'),                         // Group 1: greedy a
        ];

        let regex = expr.build();

        // Input: "aaa"
        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        // Expected:
        // Group 0 (non-greedy a*) should match "" (empty string) at position 0 because it's non-greedy.
        // Group 1 (greedy a) should match the first "a" at position 1.
        // The rest "aa" is unconsumed by these two specific groups if we consider them sequentially.
        // However, the DFA processes the whole string.
        // Let's trace:
        // State 0 --a--> State X (final for G1, also potentially for G0 if G0 matches empty)
        // If G0 (a*) is non-greedy, it prefers the shortest match (empty).
        // So, at the start, G0 matches "" (pos 0).
        // Then, G1 (a) matches the first "a" (pos 1).
        // The `execute` loop continues.
        //
        // Current logic:
        // `self.matches.entry(group_id).or_insert(self.position + local_position);` for non-greedy.
        //
        // Input "aaa":
        // First 'a': current_state -> S1 (final for G0, G1)
        //   G0 (non-greedy): matches.entry(0).or_insert(1) -> {0:1}
        //   G1 (greedy): matches.insert(1,1) -> {0:1, 1:1}
        // Second 'a': current_state -> S1
        //   G0 (non-greedy): matches.entry(0) exists, no change. {0:1, 1:1}
        //   G1 (greedy): matches.insert(1,2) -> {0:1, 1:2}
        // Third 'a': current_state -> S1
        //   G0 (non-greedy): no change. {0:1, 1:2}
        //   G1 (greedy): matches.insert(1,3) -> {0:1, 1:3}
        //
        // This seems to be the behavior of the current `execute`.
        // If the intention of "non-greedy (a*) followed by a" is that (a*) takes as little as possible
        // to allow "a" to match, then (a*) should take "" and "a" should take the first "a".
        // The `or_insert` for non-greedy means it takes the *first* match it finds.
        // If (a*) can match at pos 0 (empty) and also at pos 1 ("a"), pos 2 ("aa"),
        // it will take the pos 0 match.
        //
        // Let's re-evaluate the DFA for `(a*)_NG a_G`
        // S0 --eps--> S_G0_loop_start --eps--> S_G0_final (final for G0)
        // S_G0_loop_start --a--> S_G0_loop_start
        // S_G0_final --a--> S_G1_final (final for G1)
        //
        // Input "a":
        // Path 1: S0 -> S_G0_final (G0 matches "" at pos 0) -> S_G1_final (G1 matches "a" at pos 1)
        //   Matches: {G0:0, G1:1}
        // Path 2 (if a* takes "a"): S0 -> S_G0_loop_start -> S_G0_loop_start (G0 matches "a" at pos 1)
        //   Then this path cannot match the second 'a' for G1.
        //
        // The current `execute` loop processes one character at a time and updates matches.
        // If a state is final for a non-greedy group, it `or_insert`.
        // If a state is final for a greedy group, it `insert` (overwrites).
        //
        // For "aaa":
        // Char 1 ('a'): current_state becomes S_after_a.
        //   S_after_a is final for G0 (via empty match of a* before this 'a') -> matches.entry(0).or_insert(0) -> {0:0}
        //   S_after_a is final for G1 (this 'a') -> matches.insert(1,1) -> {0:0, 1:1}
        // Char 2 ('a'): current_state becomes S_after_aa.
        //   S_after_aa is final for G0 (via "a" match of a* before this 'a') -> matches.entry(0).or_insert(0) -> no change {0:0}
        //   S_after_aa is final for G1 (this 'a', if G0 took empty) -> matches.insert(1,2) -> {0:0, 1:2}
        // Char 3 ('a'): current_state becomes S_after_aaa.
        //   S_after_aaa is final for G0 (via "aa" match of a*) -> no change {0:0}
        //   S_after_aaa is final for G1 -> matches.insert(1,3) -> {0:0, 1:3}
        //
        // This seems correct for the `or_insert` non-greedy logic.
        assert_eq!(regex_state.matches.get(&0), Some(&0)); // Non-greedy a* matches empty string at the start
        assert_eq!(regex_state.matches.get(&1), Some(&3)); // Greedy 'a' consumes all 'aaa' if a* is empty.
                                                          // No, this is wrong. Group 1 is just 'a'.
                                                          // If G0="", G1="a", then remaining is "aa".
                                                          // The DFA should reflect this.
                                                          // The DFA for (a*)_NG (a)_G:
                                                          // S0 --a--> S1 (final G0="", final G1="a")
                                                          // S1 --a--> S1 (final G0="a", final G1="a") (if G0 can take more)
                                                          //
                                                          // Let's re-verify the `execute` logic for finalizers.
                                                          // It iterates finalizers of `dfa.states[self.current_state]`.
                                                          //
                                                          // Input "a": current_state becomes S1.
                                                          //   Finalizers of S1: G0 (non-greedy), G1 (greedy)
                                                          //   matches.entry(G0).or_insert(pos+local_pos=1) -> {G0:1}
                                                          //   matches.insert(G1, 1) -> {G0:1, G1:1}
                                                          // Input "aa":
                                                          //   First 'a': state S1. matches={G0:1, G1:1}
                                                          //   Second 'a': state S1.
                                                          //     matches.entry(G0).or_insert(2) -> no change, {G0:1, G1:1}
                                                          //     matches.insert(G1, 2) -> {G0:1, G1:2}
                                                          // Input "aaa":
                                                          //   ... state S1. matches={G0:1, G1:2}
                                                          //   Third 'a': state S1.
                                                          //     matches.entry(G0).or_insert(3) -> no change, {G0:1, G1:2}
                                                          //     matches.insert(G1, 3) -> {G0:1, G1:3}
                                                          // This means G0 (a*) matches the first 'a', and G1 (a) matches up to "aaa".
                                                          // This is not the standard non-greedy (a*) followed by (a).
                                                          // Standard: (a*)_NG should take "", then (a)_G takes "a".
                                                          // The issue might be in how the DFA is built or how non-greedy finalizers are handled.
                                                          // The current `execute` logic for non-greedy `or_insert` means it takes the *earliest ending position*.
                                                          // If (a*) can end at pos 0 (empty), pos 1 ("a"), pos 2 ("aa"),
                                                          // and it's non-greedy, it should prefer the pos 0 match.
                                                          //
                                                          // The problem is that `(a*)_NG a_G` is not one token. It's two groups.
                                                          // The DFA matches characters.
                                                          // If the DFA state after "a" is final for G0 (non-greedy) and G1 (greedy):
                                                          // G0 gets set to pos 1. G1 gets set to pos 1.
                                                          // If state after "aa" is final for G0 and G1:
                                                          // G0 is already set at 1, doesn't change. G1 updates to 2.
                                                          //
                                                          // The definition of non-greedy in the problem is "match as little as possible".
                                                          // This usually applies to quantifiers *within* a single regex pattern.
                                                          // Here, we have groups.
                                                          // If Group 0 is `(a*)_NG` and Group 1 is `a_G`.
                                                          // On input "a":
                                                          //   - Can Group 0 match "" and Group 1 match "a"? Yes. G0="", G1="a".
                                                          //   - Can Group 0 match "a" and Group 1 not match? Yes, if G1 was, e.g. "b".
                                                          //
                                                          // The current `execute` logic seems to be: for each char, find current DFA state.
                                                          // If current DFA state is final for group X (non-greedy), record match if not already recorded.
                                                          // If current DFA state is final for group Y (greedy), record match (overwrite).
                                                          //
                                                          // This test is tricky because it's about interaction of groups.
                                                          // Let's assume the Python definition of non-greedy applies to the group itself.
                                                          // Group 0 (a*)_NG: on "aaa", it should match "" at pos 0.
                                                          // Group 1 (a)_G: on "aaa", it should match "a" at pos 1. (If G0 took nothing)
                                                          // Or, if G0 took "a", G1 matches "a" at pos 2.
                                                          // Or, if G0 took "aa", G1 matches "a" at pos 3.
                                                          //
                                                          // The `greedy_find_all` logic would pick one overall tokenization.
                                                          // But `execute` just populates `self.matches`.
                                                          //
                                                          // If `(a*)_NG` is a group, it means "find the shortest possible match for this group".
                                                          // For "aaa", shortest for (a*) is "".
                                                          //
                                                          // The current `or_insert` for non-greedy groups means "the first time this group *could* complete,
                                                          // lock in that end position".
                                                          //
                                                          // For `groups![(a*)_NG, (a)_G]` on "aaa":
                                                          // After "a": DFA state S1. S1 is final for G0 and G1.
                                                          //   G0 (NG): `matches.entry(0).or_insert(1)` -> `matches[0]=1`.
                                                          //   G1 (G): `matches.insert(1,1)` -> `matches[1]=1`.
                                                          // After "aa": DFA state S1.
                                                          //   G0 (NG): `matches.entry(0)` exists (value 1), no change.
                                                          //   G1 (G): `matches.insert(1,2)` -> `matches[1]=2`.
                                                          // After "aaa": DFA state S1.
                                                          //   G0 (NG): no change.
                                                          //   G1 (G): `matches.insert(1,3)` -> `matches[1]=3`.
                                                          // Result: {0:1, 1:3}. This means G0 matched "a", G1 matched "aaa".
                                                          // This is not the Pythonic `(a*?)a` behavior where `a*?` matches empty.
                                                          //
                                                          // The current group non-greedy means "if this group matches at multiple end points,
                                                          // prefer the one that ended earliest".
                                                          // This is different from "make this quantifier inside the group non-greedy".
                                                          //
                                                          // Given the problem statement "non_greedy_group(rep(eat_u8(b'a')))",
                                                          // this means the group `rep(eat_u8(b'a'))` should be non-greedy.
                                                          // On "aaa", `rep(eat_u8(b'a'))` can match "", "a", "aa", "aaa".
                                                          // Non-greedy means it matches "".
                                                          // So, Group 0 should be `("", 0)`.
                                                          // Then Group 1 `eat_u8(b'a')` matches the first 'a'.
                                                          //
                                                          // The `execute` logic needs to be re-thought for this interpretation, or the interpretation clarified.
                                                          // The current `execute` behavior is what I implemented based on `or_insert`.
                                                          // If `non_greedy_finalizers.contains(&group_id)` then `or_insert`.
                                                          //
                                                          // Let's assume the current `execute` logic is fixed.
                                                          // Then for `groups![non_greedy_group(rep(eat_u8(b'a'))), eat_u8(b'a')]` on "aaa":
                                                          // G0 is `rep(eat_u8(b'a'))` (non-greedy)
                                                          // G1 is `eat_u8(b'a')` (greedy by default)
                                                          //
                                                          // After "a": current_state S1. S1 is final for G0 and G1.
                                                          //   G0 (NG): matches[0] = 1 (length of "a")
                                                          //   G1 (G):  matches[1] = 1 (length of "a")
                                                          // After "aa": current_state S1. S1 is final for G0 and G1.
                                                          //   G0 (NG): matches[0] is already 1. No change.
                                                          //   G1 (G):  matches[1] = 2 (length of "aa")
                                                          // After "aaa": current_state S1. S1 is final for G0 and G1.
                                                          //   G0 (NG): matches[0] is already 1. No change.
                                                          //   G1 (G):  matches[1] = 3 (length of "aaa")
                                                          // So, matches = {0:1, 1:3}.
        assert_eq!(regex_state.matches.get(&0), Some(&1)); // G0 (a*)_NG matches "a" (first opportunity)
        assert_eq!(regex_state.matches.get(&1), Some(&3)); // G1 (a)_G matches "aaa"
    }

    #[test]
    fn test_greedy_matching() {
        // Define regex: (a*)a where group 0 is greedy and group 1 is greedy
        let expr = groups![
            rep(eat_u8(b'a')), // Group 0: greedy (a*)
            eat_u8(b'a'),      // Group 1: greedy (a)
        ];

        let regex = expr.build();

        // Input: "aaa"
        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        // For `groups![(a*)_G, (a)_G]` on "aaa":
        // After "a": current_state S1. S1 is final for G0 and G1.
        //   G0 (G): matches[0] = 1
        //   G1 (G): matches[1] = 1
        // After "aa": current_state S1. S1 is final for G0 and G1.
        //   G0 (G): matches[0] = 2
        //   G1 (G): matches[1] = 2
        // After "aaa": current_state S1. S1 is final for G0 and G1.
        //   G0 (G): matches[0] = 3
        //   G1 (G): matches[1] = 3
        // So, matches = {0:3, 1:3}.
        assert_eq!(regex_state.matches.get(&0), Some(&3));
        assert_eq!(regex_state.matches.get(&1), Some(&3));
    }

    #[test]
    fn test_triple_quoted_string() {
        // Regex: """.*?""" (non-greedy)
        let non_greedy_expr = groups![
            non_greedy_group(seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())), // This rep is greedy by default
                Expr::U8Seq(b"\"\"\"".to_vec())
            ])
        ];
        // To make the inner `.*` non-greedy, it needs to be `rep(opt(Expr::U8Class(U8Set::all())))`
        // or a non-greedy quantifier if we had one for `rep`.
        // The current `non_greedy_group` applies to the *whole group's match extent*.
        // Let's assume `rep` is greedy.
        // Group 0: `""" .* """` (non-greedy group)
        //   The `.*` inside is greedy.
        //   On `"""hello"""world"""`:
        //   `"""` matches.
        //   `.*` (greedy) matches `hello"""world`.
        //   `"""` matches.
        //   So the group `""" .* """` matches the whole thing: `"""hello"""world"""`.
        //   Since the group itself is non-greedy, if there were multiple ways for this *group*
        //   to match, it would pick the shortest. But here, the inner greedy `.*` dominates.
        //
        // If we want Python's `""".*?"""`:
        // The `*?` (non-greedy rep) is key. We don't have a direct non-greedy `rep`.
        // `rep(X)` is `X*`. `non_greedy_group(rep(X))` is `(X*)_NG`.
        // We need `X*?`. This is often `(X*?)_G`.
        //
        // Let's use the provided structure and see.
        let non_greedy_regex = non_greedy_expr.build();

        // Regex: """.*""" (greedy group)
        let greedy_expr = groups![
            seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())), // greedy rep
                Expr::U8Seq(b"\"\"\"".to_vec())
            ]
        ];
        let greedy_regex = greedy_expr.build();

        let input = b"\"\"\"hello\"\"\"world\"\"\"";

        // Non-greedy group `( """ .* """ )_NG`
        //   Inner `.*` is greedy. Matches `hello"""world`.
        //   Full match for group is `"""hello"""world"""`.
        //   Since it's the only way for this group to match, non-greedy has no other option.
        let mut non_greedy_state = non_greedy_regex.init();
        non_greedy_state.execute(input);
        assert_eq!(non_greedy_state.matches.get(&0), Some(input.len()));


        // Greedy group `( """ .* """ )_G`
        //   Same inner logic.
        let mut greedy_state = greedy_regex.init();
        greedy_state.execute(input);
        assert_eq!(greedy_state.matches.get(&0), Some(input.len()));
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
            choice![opt(eat_u8(b'a')), rep(eat_u8(b'b')), eat_u8(b'c')], // Group 0
            eat_u8(b'a'), // Group 1
        ], BTreeSet::from([0, 1]));
        run_test(groups![
            eat_u8(b'a'), // Group 0
            seq![eat_u8(b'a'), eat_u8(b'a')], // Group 1
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
        assert_eq!(group_id_to_u8set.len(), 2); // Two distinct groups, even if their exprs are same
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
        // Group 0: (a|b)*
        // Group 1: c
        let expr = groups![
            rep(choice![eat_u8(b'a'), eat_u8(b'b')]), // Group 0
            eat_u8(b'c'),                           // Group 1
        ];
        let regex = expr.build();

        // Start state (state 0)
        // group_id_to_u8set for state 0:
        // - Group 0: {'a', 'b'} (because (a|b)* can start with a or b)
        // - Group 1: {'c'} (because c can start immediately if (a|b)* is empty)
        // Also, if (a|b)* matches 'a', then 'c' can follow.
        // The `possible_current_or_future_group_ids` logic should handle this.

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        dbg!(&regex);
        dbg!(&regex.dfa.states[0].possible_future_group_ids);
        dbg!(group_id_to_u8set);
        assert_eq!(group_id_to_u8set.len(), 2);

        let u8set_group0 = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_group0.contains(b'a'));
        assert!(u8set_group0.contains(b'b'));
        // It might also contain 'c' if (a|b)* can be empty and then 'c' is for group 1.
        // The current logic for group_id_to_u8set:
        // For each transition (input_u8 -> next_state_index):
        //   For each group_id in possible_current_or_future_group_ids[next_state_index]:
        //     map[group_id].insert(input_u8)
        //
        // If S0 --a--> S1. S1 can lead to G0. Then S0.map[G0] gets 'a'.
        // If S0 --c--> S2. S2 can lead to G1. Then S0.map[G1] gets 'c'.
        //
        // For (a|b)*c:
        // S0 --a--> S_after_a (can still be in G0, or lead to G1 via c)
        // S0 --b--> S_after_b (can still be in G0, or lead to G1 via c)
        // S0 --c--> S_after_c (is G1)
        //
        // So, G0 should have {a,b}. G1 should have {c}.
        // If (a|b)* is empty, then 'c' is for G1.
        // If (a|b)* is 'a', then 'c' is for G1.
        //
        // Let's check `possible_future_group_ids` for S0. It should be {0,1}.
        // If next char is 'a', next state S_a. `possible_current_or_future_group_ids` for S_a is {0,1}.
        //   So, map[0].insert('a'), map[1].insert('a').
        // If next char is 'c', next state S_c. `possible_current_or_future_group_ids` for S_c is {1}.
        //   So, map[1].insert('c').
        // This means map[0]={a,b}, map[1]={a,b,c}. This seems more plausible.

        let expected_g0 = U8Set::from_bytes(b"ab");
        assert_eq!(*u8set_group0, expected_g0);


        let u8set_group1 = group_id_to_u8set.get(&1).unwrap();
        // Group 1 ('c') can be reached by 'c' directly (if (a|b)* is empty)
        // or after some 'a's or 'b's.
        // So, if current char is 'a', it could lead to G1 (via 'c' later).
        // If current char is 'b', it could lead to G1 (via 'c' later).
        // If current char is 'c', it directly leads to G1.
        let expected_g1 = U8Set::from_bytes(b"abc");
        assert_eq!(*u8set_group1, expected_g1);
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

    // test_compute_group_id_to_u8set_overlapping_groups is already present and correct.

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
        // - Group 0 to {'a'}
        // - Group 1 to {'a'}

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 2);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));
        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();
        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));

        // After consuming 'a', move to state(s) corresponding to 'ab' and 'ac'
        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");

        // Verify possible_future_group_ids for the state after 'a'
        // This state can lead to completing Group 0 (with 'b') or Group 1 (with 'c')
        assert_eq!(
            regex.dfa.states[regex_state_a.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1]) // Both groups are still possible
        );

        // Verify group_id_to_u8set for the new state (after 'a')
        // - To complete Group 0 ("ab"), next char must be 'b'.
        // - To complete Group 1 ("ac"), next char must be 'c'.
        let group_id_to_u8set_new = &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_new.len(), 2);
        assert!(group_id_to_u8set_new.contains_key(&0));
        assert!(group_id_to_u8set_new.contains_key(&1));

        let u8set_new_group0 = group_id_to_u8set_new.get(&0).unwrap();
        let u8set_new_group1 = group_id_to_u8set_new.get(&1).unwrap();

        assert_eq!(*u8set_new_group0, U8Set::from_byte(b'b'));
        assert_eq!(*u8set_new_group1, U8Set::from_byte(b'c'));
    }

    // test_group_id_to_u8set_after_multiple_transitions is already present and correct.
    // test_group_id_to_u8set_after_consuming_all is already present and correct.
    // test_get_u8set_for_group_multiple_transitions is already present and correct.
}

// test_group_u8set_tests module is already present and seems fine.
// tests_nov_24 module is already present and seems fine.
// test_python module is already present and seems fine.

