use crate::datastructures::char_transitions::CharTransitions;
use crate::datastructures::frozenset::FrozenSet;
use crate::datastructures::u8set::U8Set;
use crate::json_serialization::{JSONConvertible, JSONNode};
use json_convertible_derive::JSONConvertible;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::collections::BTreeMap as StdMap;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use ahash::AHashMap;
use profiler_macro::time_it;
use crate::datastructures::compressed_state_set::{CompressedStateSet, DenseStateSet, SparseStateSet};
use crate::datastructures::state_set::StateSet;


pub type GroupID = usize;
pub type ActiveStateSet = CompressedStateSet;

#[derive(Debug, Clone)]
pub struct NFAState {
    /// Non-epsilon transitions: list of (input byte, target state).
    /// There may be multiple entries with the same input byte (non-determinism).
    transitions: Vec<(U8Set, usize)>,
    /// Epsilon transitions: target states reachable without consuming input.
    epsilon_transitions: Vec<usize>,
    finalizers: DenseStateSet,
    non_greedy_finalizers: DenseStateSet,
}


struct CompactNFA {
    epsilon_offsets: Vec<u32>,
    epsilon_targets: Vec<u32>,
}

/// Intermediate JSON representation for NFAState transitions.
/// Maps byte -> list of target states for non-deterministic transitions.
#[derive(Debug, Clone, JSONConvertible)]
struct NFAStateJSON {
    /// byte (as string key) -> targets
    transitions: BTreeMap<String, Vec<usize>>,
    epsilon_transitions: Vec<usize>,
    finalizers: DenseStateSet,
    non_greedy_finalizers: DenseStateSet,
}

impl NFAState {
    fn to_json_struct(&self) -> NFAStateJSON {
        let mut grouped: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for (set, target) in &self.transitions {
            for byte in set.iter() {
                grouped.entry(byte).or_default().push(*target);
            }
        }
        let transitions: BTreeMap<String, Vec<usize>> = grouped
            .into_iter()
            .map(|(byte, targets)| (byte.to_string(), targets))
            .collect();
        
        NFAStateJSON {
            transitions,
            epsilon_transitions: self.epsilon_transitions.clone(),
            finalizers: self.finalizers.clone(),
            non_greedy_finalizers: self.non_greedy_finalizers.clone(),
        }
    }
    
    fn from_json_struct(s: NFAStateJSON) -> Result<Self, String> {
        let mut transitions: Vec<(U8Set, usize)> = Vec::new();
        for (byte_str, targets) in s.transitions {
            let byte = byte_str.parse::<u8>().map_err(|e| {
                format!("Invalid u8 key in NFAState transitions: {}, err: {}", byte_str, e)
            })?;
            for target in targets {
                transitions.push((U8Set::from_u8(byte), target));
            }
        }
        
        Ok(NFAState {
            transitions,
            epsilon_transitions: s.epsilon_transitions,
            finalizers: s.finalizers,
            non_greedy_finalizers: s.non_greedy_finalizers,
        })
    }
}

impl JSONConvertible for NFAState {
    fn to_json(&self) -> JSONNode {
        self.to_json_struct().to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        NFAStateJSON::from_json(node).and_then(Self::from_json_struct)
    }
}

#[derive(Debug, Clone)]
pub struct NFA {
    pub states: Vec<NFAState>,
    pub start_state: usize,
}

/// Simple intermediate type for NFA using derive macro.
#[derive(Debug, Clone, JSONConvertible)]
struct NFAJSON {
    states: Vec<NFAState>,
    start_state: usize,
}

impl JSONConvertible for NFA {
    fn to_json(&self) -> JSONNode {
        NFAJSON {
            states: self.states.clone(),
            start_state: self.start_state,
        }.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        NFAJSON::from_json(node).map(|j| NFA {
            states: j.states,
            start_state: j.start_state,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DFAState {
    pub transitions: CharTransitions<usize>,
    pub finalizers: DenseStateSet,
    pub possible_future_group_ids: BTreeSet<GroupID>,
    pub group_id_to_u8set: BTreeMap<GroupID, U8Set>,
}

/// DFAState derives JSONConvertible, used for the old format and as a building block.
#[derive(Debug, Clone, JSONConvertible)]
struct DFAStateJSON {
    transitions: CharTransitions<usize>,
    finalizers: DenseStateSet,
    possible_future_group_ids: BTreeSet<GroupID>,
    group_id_to_u8set: BTreeMap<GroupID, U8Set>,
}

impl DFAState {
    fn to_json_struct(&self) -> DFAStateJSON {
        DFAStateJSON {
            transitions: self.transitions.clone(),
            finalizers: self.finalizers.clone(),
            possible_future_group_ids: self.possible_future_group_ids.clone(),
            group_id_to_u8set: self.group_id_to_u8set.clone(),
        }
    }
    
    fn from_json_struct(s: DFAStateJSON) -> Self {
        DFAState {
            transitions: s.transitions,
            finalizers: s.finalizers,
            possible_future_group_ids: s.possible_future_group_ids,
            group_id_to_u8set: s.group_id_to_u8set,
        }
    }
}

impl JSONConvertible for DFAState {
    fn to_json(&self) -> JSONNode {
        self.to_json_struct().to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        DFAStateJSON::from_json(node).map(Self::from_json_struct)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DFA {
    pub states: Vec<DFAState>,
    pub start_state: usize,
    pub non_greedy_finalizers: BTreeSet<GroupID>,
}

/// Compact transition entry: (U8Set, target_state)
/// Groups all input bytes that lead to the same target state.
#[derive(Debug, Clone, JSONConvertible)]
struct CompactTransitionEntry(U8Set, usize);

/// Compact DFA JSON representation for efficient serialization.
/// Each state's transitions are grouped by target state for smaller output.
#[derive(Debug, Clone, JSONConvertible)]
struct DFAJSON {
    start_state: usize,
    non_greedy_finalizers: BTreeSet<GroupID>,
    /// Each element is a list of (U8Set, target) pairs
    state_transitions: Vec<Vec<CompactTransitionEntry>>,
    state_finalizers: Vec<DenseStateSet>,
    state_possible_future_group_ids: Vec<BTreeSet<GroupID>>,
    state_group_id_to_u8set: Vec<BTreeMap<GroupID, U8Set>>,
}

/// Old format for backward compatibility
#[derive(Debug, Clone, JSONConvertible)]
struct DFAOldJSON {
    states: Vec<DFAStateJSON>,
    start_state: usize,
    non_greedy_finalizers: BTreeSet<GroupID>,
}

impl DFA {
    fn to_compact_json(&self) -> DFAJSON {
        let mut state_transitions = Vec::with_capacity(self.states.len());
        let mut state_finalizers = Vec::with_capacity(self.states.len());
        let mut state_possible_future_group_ids = Vec::with_capacity(self.states.len());
        let mut state_group_id_to_u8set = Vec::with_capacity(self.states.len());

        for state in &self.states {
            // Compact transitions by grouping all input bytes that lead to the same target
            let mut target_to_set: BTreeMap<usize, U8Set> = BTreeMap::new();
            for (byte, &target) in &state.transitions {
                target_to_set
                    .entry(target)
                    .and_modify(|set| { set.insert(byte); })
                    .or_insert_with(|| U8Set::from_u8(byte));
            }

            let packed_transitions: Vec<CompactTransitionEntry> = target_to_set
                .into_iter()
                .map(|(target, u8set)| CompactTransitionEntry(u8set, target))
                .collect();
            
            state_transitions.push(packed_transitions);
            state_finalizers.push(state.finalizers.clone());
            state_possible_future_group_ids.push(state.possible_future_group_ids.clone());
            state_group_id_to_u8set.push(state.group_id_to_u8set.clone());
        }

        DFAJSON {
            start_state: self.start_state,
            non_greedy_finalizers: self.non_greedy_finalizers.clone(),
            state_transitions,
            state_finalizers,
            state_possible_future_group_ids,
            state_group_id_to_u8set,
        }
    }
    
    fn from_compact_json(j: DFAJSON) -> Result<Self, String> {
        let num_states = j.state_transitions.len();
        if j.state_finalizers.len() != num_states ||
           j.state_possible_future_group_ids.len() != num_states ||
           j.state_group_id_to_u8set.len() != num_states {
            return Err("Mismatched lengths for DFA state arrays".to_string());
        }

        let mut states = Vec::with_capacity(num_states);
        for i in 0..num_states {
            // Expand compact transitions back to per-byte
            let mut entries: Vec<(u8, usize)> = Vec::new();
            for CompactTransitionEntry(u8set, target) in &j.state_transitions[i] {
                for b in u8set.iter() {
                    entries.push((b, *target));
                }
            }
            entries.sort_by_key(|(b, _)| *b);
            let transitions = CharTransitions::from_sorted_entries(entries);

            states.push(DFAState {
                transitions,
                finalizers: j.state_finalizers[i].clone(),
                possible_future_group_ids: j.state_possible_future_group_ids[i].clone(),
                group_id_to_u8set: j.state_group_id_to_u8set[i].clone(),
            });
        }

        Ok(DFA {
            states,
            start_state: j.start_state,
            non_greedy_finalizers: j.non_greedy_finalizers,
        })
    }
    
    fn from_old_json(j: DFAOldJSON) -> Self {
        DFA {
            states: j.states.into_iter().map(DFAState::from_json_struct).collect(),
            start_state: j.start_state,
            non_greedy_finalizers: j.non_greedy_finalizers,
        }
    }
}

impl JSONConvertible for DFA {
    fn to_json(&self) -> JSONNode {
        self.to_compact_json().to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        // Try compact format first, then fall back to old format
        let mut obj = node.clone().into_object()?;
        
        if obj.contains_key("states") {
            // Old format
            DFAOldJSON::from_json(node).map(Self::from_old_json)
        } else {
            // New compact format
            DFAJSON::from_json(node).and_then(Self::from_compact_json)
        }
    }
}

// TODO: should this *really* derive `Clone`? Users probably shouldn't clone this, should they?
#[derive(Debug, Clone, PartialEq, Eq, JSONConvertible)]
pub struct Regex {
    pub dfa: DFA,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct Match {
    pub group_id: GroupID,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, JSONConvertible)]
pub struct FinalStateReport {
    pub position: usize,
    pub matches: BTreeMap<GroupID, usize>, // GroupID to position
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    pub matches: Vec<Match>,
    pub end_state: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Trellis<T> {
    pub end_state: Option<T>,
    pub edges: BTreeMap<GroupID, Arc<Trellis<T>>>,
}

pub type TokenTrellis = Trellis<usize>;
pub type TokenTrellisWithCompletion = Trellis<BTreeSet<GroupID>>;

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
    RepeatBounded {
        inner: Box<Expr>,
        min: usize,
        max: Option<usize>,
    },
    Choice(Vec<Expr>),
    Seq(Vec<Expr>),
    Epsilon, // Explicit epsilon transition
}

/// Intermediate type for Expr JSON serialization (maintains backward compatibility)
#[derive(JSONConvertible)]
enum ExprJSON {
    U8Seq { bytes: Vec<u8> },
    U8Class { u8set: U8Set },
    Shared { inner: Box<ExprJSON> },
    Quantifier { expr: Box<ExprJSON>, q_type: QuantifierType },
    RepeatBounded { inner: Box<ExprJSON>, min: usize, max: Option<usize> },
    Choice { exprs: Vec<ExprJSON> },
    Seq { exprs: Vec<ExprJSON> },
    Epsilon,
}

impl ExprJSON {
    fn from_expr(expr: &Expr) -> Self {
        match expr {
            Expr::U8Seq(bytes) => ExprJSON::U8Seq { bytes: bytes.clone() },
            Expr::U8Class(u8set) => ExprJSON::U8Class { u8set: u8set.clone() },
            Expr::Shared(inner) => ExprJSON::Shared { inner: Box::new(ExprJSON::from_expr(inner)) },
            Expr::Quantifier(expr, q_type) => ExprJSON::Quantifier {
                expr: Box::new(ExprJSON::from_expr(expr)),
                q_type: q_type.clone(),
            },
            Expr::RepeatBounded { inner, min, max } => ExprJSON::RepeatBounded {
                inner: Box::new(ExprJSON::from_expr(inner)),
                min: *min,
                max: *max,
            },
            Expr::Choice(exprs) => ExprJSON::Choice {
                exprs: exprs.iter().map(ExprJSON::from_expr).collect(),
            },
            Expr::Seq(exprs) => ExprJSON::Seq {
                exprs: exprs.iter().map(ExprJSON::from_expr).collect(),
            },
            Expr::Epsilon => ExprJSON::Epsilon,
        }
    }

    fn to_expr(self) -> Expr {
        match self {
            ExprJSON::U8Seq { bytes } => Expr::U8Seq(bytes),
            ExprJSON::U8Class { u8set } => Expr::U8Class(u8set),
            ExprJSON::Shared { inner } => Expr::Shared(Arc::new(inner.to_expr())),
            ExprJSON::Quantifier { expr, q_type } => Expr::Quantifier(Box::new(expr.to_expr()), q_type),
            ExprJSON::RepeatBounded { inner, min, max } => Expr::RepeatBounded {
                inner: Box::new(inner.to_expr()),
                min,
                max,
            },
            ExprJSON::Choice { exprs } => Expr::Choice(exprs.into_iter().map(|e| e.to_expr()).collect()),
            ExprJSON::Seq { exprs } => Expr::Seq(exprs.into_iter().map(|e| e.to_expr()).collect()),
            ExprJSON::Epsilon => Expr::Epsilon,
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::U8Seq(bytes) => {
                let s = String::from_utf8_lossy(bytes);
                write!(f, "\"{}\"", s.escape_debug())
            }
            Expr::U8Class(set) => {
                write!(f, "[{}]", format_u8_class(set))
            }
            Expr::Shared(inner) => write!(f, "{}", inner),
            Expr::Quantifier(inner, q_type) => {
                let suffix = match q_type {
                    QuantifierType::ZeroOrMore => "*",
                    QuantifierType::OneOrMore => "+",
                    QuantifierType::ZeroOrOne => "?",
                };
                let needs_parens = matches!(**inner, Expr::Choice(_) | Expr::Seq(_));
                if needs_parens {
                    write!(f, "({}){}", inner, suffix)
                } else {
                    write!(f, "{}{}", inner, suffix)
                }
            }
            Expr::RepeatBounded { inner, min, max } => {
                let needs_parens = matches!(**inner, Expr::Choice(_) | Expr::Seq(_));
                let format_inner = if needs_parens {
                    format!("({})", inner)
                } else {
                    format!("{}", inner)
                };
                match max {
                    Some(max_val) => write!(f, "{}{{{},{}}}", format_inner, min, max_val),
                    None => write!(f, "{}{{{},}}", format_inner, min),
                }
            }
            Expr::Choice(exprs) => {
                // Heuristic for Optional: P|ε or ε|P -> P?
                if exprs.len() == 2 {
                    let p = if exprs[1] == Expr::Epsilon {
                        Some(&exprs[0])
                    } else if exprs[0] == Expr::Epsilon {
                        Some(&exprs[1])
                    } else {
                        None
                    };
                    if let Some(p) = p {
                        if matches!(p, Expr::Choice(_) | Expr::Seq(_)) {
                            return write!(f, "({})?", p);
                        } else {
                            return write!(f, "{}?", p);
                        }
                    }
                }

                for (i, e) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", e)?;
                }
                Ok(())
            }
            Expr::Seq(exprs) => {
                for (i, e) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    if matches!(e, Expr::Choice(_)) {
                        write!(f, "({})", e)?;
                    } else {
                        write!(f, "{}", e)?;
                    }
                }
                Ok(())
            }
            Expr::Epsilon => write!(f, "ε"),
        }
    }
}

fn format_u8_class(set: &U8Set) -> String {
    let mut result = String::new();
    let mut i = 0u16; 
    let mut in_range = false;
    let mut range_start = 0u8;

    while i <= 255 {
        let b = i as u8;
        if set.contains(b) {
            if !in_range {
                range_start = b;
                in_range = true;
            }
        } else {
            if in_range {
                append_range(&mut result, range_start, (i - 1) as u8);
                in_range = false;
            }
        }
        i += 1;
    }
    if in_range {
        append_range(&mut result, range_start, 255);
    }
    result
}

fn append_range(s: &mut String, start: u8, end: u8) {
    if start == end {
        s.push_str(&escape_byte(start));
    } else if start == end.wrapping_sub(1) {
        s.push_str(&escape_byte(start));
        s.push_str(&escape_byte(end));
    } else {
        s.push_str(&escape_byte(start));
        s.push('-');
        s.push_str(&escape_byte(end));
    }
}

fn escape_byte(b: u8) -> String {
    match b {
        b'\n' => "\\n".to_string(),
        b'\r' => "\\r".to_string(),
        b'\t' => "\\t".to_string(),
        b'\\' => "\\\\".to_string(),
        b']' => "\\]".to_string(),
        b'-' => "\\-".to_string(),
        32..=126 => (b as char).to_string(),
        _ => format!("\\x{:02x}", b),
    }
}

impl JSONConvertible for Expr {
    fn to_json(&self) -> JSONNode {
        ExprJSON::from_expr(self).to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        ExprJSON::from_json(node).map(|e| e.to_expr())
    }
}

// QuantifierType is a unit-only enum - derive macro produces string serialization
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, JSONConvertible)]
pub enum QuantifierType {
    ZeroOrMore, // *
    OneOrMore,  // +
    ZeroOrOne,  // ?
}

#[derive(Debug, Clone, PartialEq, Eq, JSONConvertible)]
pub struct ExprGroup {
    pub expr: Expr,
    pub is_non_greedy: bool,
}

#[derive(Debug, Clone, JSONConvertible)]
pub struct ExprGroups {
    pub groups: Vec<ExprGroup>,
}

#[derive(Debug, Default)]
pub struct ExprStats {
    pub nodes: usize,
    pub u8seq: usize,
    pub u8class: usize,
    pub shared: usize,
    pub shared_unique: usize,   // Unique Shared nodes (by Arc pointer)
    pub shared_inlineable: usize, // Shared wrapping U8Seq/U8Class/Epsilon
    pub quantifier: usize,
    pub repeat_bounded: usize,
    pub choice: usize,
    pub seq: usize,
    pub epsilon: usize,
    pub max_depth: usize,
}

impl std::fmt::Display for ExprStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Nodes: {}, Depth: {}, Seq: {}, Choice: {}, Quant: {}, RepeatBounded: {}, Shared: {} (unique: {}, inlineable: {}), U8Seq: {}, U8Class: {}, Eps: {}",
            self.nodes, self.max_depth, self.seq, self.choice, self.quantifier, self.repeat_bounded, self.shared, self.shared_unique, self.shared_inlineable, self.u8seq, self.u8class, self.epsilon)
    }
}

impl ExprGroups {
    pub fn optimize_prefixes(self) -> (Option<Expr>, ExprGroups) {
        if self.groups.is_empty() {
            return (None, self);
        }

        // Candidate prefix is the first element of the first group's expression
        let candidate_prefix = match &self.groups[0].expr {
            Expr::Seq(exprs) if !exprs.is_empty() => &exprs[0],
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::Seq(exprs) if !exprs.is_empty() => &exprs[0],
                _ => return (None, self),
            },
            _ => return (None, self),
        };

        // Check if all groups start with this prefix
        for group in &self.groups {
            if group.expr.strip_prefix(candidate_prefix).is_none() {
                return (None, self);
            }
        }

        // Factor out the prefix
        let prefix = candidate_prefix.clone();
        let mut new_groups = Vec::with_capacity(self.groups.len());

        for group in self.groups {
            let remainder = group.expr.strip_prefix(&prefix).unwrap();
            new_groups.push(ExprGroup {
                expr: remainder,
                is_non_greedy: group.is_non_greedy,
            });
        }

        (Some(prefix), ExprGroups { groups: new_groups })
    }

    pub fn get_stats(&self) -> ExprStats {
        let mut stats = ExprStats::default();
        let mut visited = HashSet::new();

        for group in &self.groups {
            let mut stack = vec![(&group.expr, 0)];
            while let Some((expr, depth)) = stack.pop() {
                stats.nodes += 1;
                if depth > stats.max_depth {
                    stats.max_depth = depth;
                }
                match expr {
                    Expr::U8Seq(_) => stats.u8seq += 1,
                    Expr::U8Class(_) => stats.u8class += 1,
                    Expr::Shared(inner) => {
                        stats.shared += 1;
                        // Check if this Shared is inlineable
                        let is_inlineable = match inner.as_ref() {
                            Expr::U8Seq(s) if s.len() <= 4 => true,
                            Expr::U8Class(_) => true,
                            Expr::Epsilon => true,
                            _ => false,
                        };
                        if is_inlineable {
                            stats.shared_inlineable += 1;
                        }
                        let ptr = Arc::as_ptr(inner) as usize;
                        if visited.insert(ptr) {
                            stats.shared_unique += 1;
                            stack.push((inner, depth + 1));
                        }
                    }
                    Expr::Quantifier(inner, _) => {
                        stats.quantifier += 1;
                        stack.push((inner, depth + 1));
                    }
                    Expr::RepeatBounded { inner, .. } => {
                        stats.repeat_bounded += 1;
                        stack.push((inner, depth + 1));
                    }
                    Expr::Choice(children) => {
                        stats.choice += 1;
                        for c in children {
                            stack.push((c, depth + 1));
                        }
                    }
                    Expr::Seq(children) => {
                        stats.seq += 1;
                        for c in children {
                            stack.push((c, depth + 1));
                        }
                    }
                    Expr::Epsilon => stats.epsilon += 1,
                }
            }
        }
        stats
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
            finalizers: DenseStateSet::empty(),
            non_greedy_finalizers: DenseStateSet::empty(),
        }
    }
}

impl ExprGroups {
    pub fn build(self) -> Regex {
        self.build_impl(true)
    }

    pub fn build_unminimized(self) -> Regex {
        self.build_impl(false)
    }

    fn build_impl(self, minimize: bool) -> Regex {
        // Debug: serialize ExprGroups if DUMP_EXPR_GROUPS_PATH env var is set.
        // Useful for capturing expressions that cause slow builds.
        // Usage: DUMP_EXPR_GROUPS_PATH=expr_groups.json cargo test ...
        if let Ok(dump_path) = std::env::var("DUMP_EXPR_GROUPS_PATH") {
            use crate::json_serialization::JSONConvertible;
            let json = self.to_json();
            let json_str = json.to_json_string();
            if let Err(e) = std::fs::write(&dump_path, &json_str) {
                eprintln!("Failed to write ExprGroups to {}: {}", dump_path, e);
            } else {
                eprintln!("Wrote ExprGroups ({} bytes) to {}", json_str.len(), dump_path);
            }
        }
        
        let stats = self.get_stats();
        crate::debug!(5, "Expr Stats: {}", stats);

        // Optimize the expression first to minimize nested quantifiers
        crate::debug!(4, "Optimizing expression");
        let start_optimize = std::time::Instant::now();
        let optimized = crate::time!("optimize_expr", self.optimize());
        crate::debug!(5, "Optimized expression in {:.2?}", start_optimize.elapsed());
        
        // Print the optimized expression for debugging DFA explosion
        crate::debug!(6, "Optimized expression groups:");
        for (i, group) in optimized.groups.iter().enumerate() {
            crate::debug!(6, "  Group {}: {}", i, group.expr);
        }

        crate::debug!(4, "Building NFA");
        let start = std::time::Instant::now();
        let mut nfa = crate::time!("build_nfa", optimized.build_nfa());
        crate::debug!(5, "Built NFA with {} states in {:.2?}", nfa.states.len(), start.elapsed());

        let start_condense = std::time::Instant::now();
        let nfa_states_before = nfa.states.len();
        crate::time!("condense_epsilon_sccs", nfa.condense_epsilon_sccs());
        crate::debug!(5, "Condensed NFA {} → {} states in {:.2?}", nfa_states_before, nfa.states.len(), start_condense.elapsed());

        crate::debug!(4, "Converting NFA to DFA");
        let start = std::time::Instant::now();
        let mut dfa = crate::time!("to_dfa", nfa.to_dfa());
        crate::debug!(5, "Converted NFA to DFA in {:.2?}", start.elapsed());

        if minimize {
            crate::debug!(4, "Minimizing DFA");
            let start = std::time::Instant::now();
            crate::time!("minimize_dfa", dfa.minimize());
            crate::debug!(5, "Minimized DFA in {:.2?}", start.elapsed());
        }

        Regex { dfa }
    }

    pub fn build_nfa(self) -> NFA {
        // CPS (continuation-passing style) NFA compilation:
        // Cache key is (SharedId, continuation_state) to enable safe sharing.
        // This prevents exponential blowup when the same Shared subexpression
        // appears multiple times with the same continuation.
        type SharedId = usize;      // Arc pointer identity
        type ContState = usize;     // continuation NFA state
        type CpsCacheKey = (SharedId, ContState);
        type CpsCache = HashMap<CpsCacheKey, usize>;

        let mut nfa = NFA {
            states: vec![NFAState::new()],
            start_state: 0,
        };

        let mut cache: CpsCache = HashMap::new();

        crate::debug!(4, "Expr stats: {}", self.get_stats());

        // Optimization: Factor out common prefix (e.g. ignore pattern)
        let (prefix, groups) = self.optimize_prefixes();

        // Create split point where all groups branch from
        let split_point = nfa.add_state();

        // Compile optional prefix into the split point
        let start_state = if let Some(prefix_expr) = prefix {
            crate::debug!(4, "Factored out common prefix in NFA construction");
            Expr::compile_cps(&prefix_expr, &mut nfa, split_point, &mut cache)
        } else {
            split_point
        };
        
        // Start from the compiled prefix (or split point if no prefix)
        nfa.states[0].epsilon_transitions.push(start_state);

        for (group_idx, ExprGroup { expr, is_non_greedy }) in groups.groups.into_iter().enumerate() {
            // Create accept state for this group
            let accept = nfa.add_state();
            nfa.states[accept].finalizers.insert(group_idx);
            if is_non_greedy {
                nfa.states[accept].non_greedy_finalizers.insert(group_idx);
            }
            
            // Compile the group expression into the accept state
            let group_start = Expr::compile_cps(&expr, &mut nfa, accept, &mut cache);
            
            // Connect from split point to group start
            nfa.add_epsilon_transition(split_point, group_start);
        }

        nfa
    }

}

impl ExprGroups {
    pub fn optimize(self) -> Self {
        ExprGroups {
            groups: self.groups.into_iter().map(|g| ExprGroup {
                expr: g.expr.optimize(),
                is_non_greedy: g.is_non_greedy,
            }).collect()
        }
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

    /// Calculates the total size of a slice of `Expr`s in bytes, accounting for shared subexpressions.
    pub fn total_size_in_bytes(exprs: &[Expr]) -> usize {
        let mut total_size = 0;
        let mut visited = HashSet::new();
        for expr in exprs {
            expr.add_size_recursive(&mut total_size, &mut visited);
        }
        total_size
    }

    fn add_size_recursive(&self, total: &mut usize, visited: &mut HashSet<usize>) {
        *total += std::mem::size_of::<Self>();

        match self {
            Expr::U8Seq(bytes) => {
                *total += bytes.capacity() * std::mem::size_of::<u8>();
            }
            Expr::U8Class(_) => {}
            Expr::Shared(inner) => {
                let ptr = Arc::as_ptr(inner) as usize;
                if visited.insert(ptr) {
                    inner.add_size_recursive(total, visited);
                }
            }
            Expr::Quantifier(expr, _) => {
                expr.add_size_recursive(total, visited);
            }
            Expr::RepeatBounded { inner, .. } => {
                inner.add_size_recursive(total, visited);
            }
            Expr::Choice(exprs) | Expr::Seq(exprs) => {
                *total += exprs.capacity() * std::mem::size_of::<Expr>();
                for expr in exprs {
                    expr.add_size_recursive(total, visited);
                }
            }
            Expr::Epsilon => {}
        }
    }

    /// CPS (Continuation-Passing Style) NFA compilation.
    /// 
    /// Compiles an expression into an NFA fragment that, when entered at the returned
    /// start state, recognizes the expression and then flows to `cont`.
    /// 
    /// This approach enables safe sharing of `Shared` subexpressions by caching based
    /// on `(SharedId, continuation_state)`. The key insight is that two occurrences of
    /// the same Shared subexpression can only share NFA states if they have the same
    /// continuation - otherwise, merging them would create incorrect paths.
    /// 
    /// Cache key: `(Arc::as_ptr(shared_arc) as usize, cont_state)`
    fn compile_cps(
        expr: &Expr,
        nfa: &mut NFA,
        cont: usize,
        cache: &mut HashMap<(usize, usize), usize>,
    ) -> usize {
        match expr {
            Expr::Epsilon => {
                // Epsilon consumes nothing, so the entry point is just the continuation
                cont
            }
            
            Expr::U8Seq(bytes) => {
                if bytes.is_empty() {
                    return cont;
                }
                // Build a chain backwards so it ends at cont:
                // for "abc": a -> b -> c -> cont
                let mut s = cont;
                for &b in bytes.iter().rev() {
                    let p = nfa.add_state();
                    nfa.add_transition(p, b, s);
                    s = p;
                }
                s
            }
            
            Expr::U8Class(set) => {
                // One state that transitions on the set to cont
                let s = nfa.add_state();
                nfa.add_u8set_transition(s, set.clone(), cont);
                s
            }
            
            Expr::Seq(children) => {
                if children.is_empty() {
                    return cont;
                }
                // Compile from tail to head: each earlier element's continuation
                // is the start of the suffix
                let mut s = cont;
                for child in children.iter().rev() {
                    s = Self::compile_cps(child, nfa, s, cache);
                }
                s
            }
            
            Expr::Choice(alts) => {
                if alts.is_empty() {
                    // Empty choice matches nothing - create a dead state
                    return nfa.add_state();
                }
                // Create one split state with epsilon transitions to each alternative's start
                let split = nfa.add_state();
                for alt in alts.iter() {
                    let alt_start = Self::compile_cps(alt, nfa, cont, cache);
                    nfa.add_epsilon_transition(split, alt_start);
                }
                split
            }
            
            Expr::Quantifier(inner, q_type) => {
                match q_type {
                    QuantifierType::ZeroOrMore => {
                        // e* uses the standard CPS/Thompson split state:
                        // split -> cont (skip)
                        // split -> body_start (take one iteration)
                        // body ends at split (loop back)
                        let split = nfa.add_state();
                        nfa.add_epsilon_transition(split, cont);  // skip path
                        
                        let body_start = Self::compile_cps(inner.as_ref(), nfa, split, cache);
                        nfa.add_epsilon_transition(split, body_start);  // take path
                        
                        split
                    }
                    QuantifierType::OneOrMore => {
                        // e+ = e e* without building an explicit Seq node
                        // Build the * loop, then return the start of one required e
                        let loop_split = nfa.add_state();
                        nfa.add_epsilon_transition(loop_split, cont);  // exit loop
                        
                        // Body of the star: ends at loop_split
                        let body_start = Self::compile_cps(inner.as_ref(), nfa, loop_split, cache);
                        nfa.add_epsilon_transition(loop_split, body_start);  // loop back
                        
                        // For +: must do body once before reaching loop_split
                        // Return body_start - ensures at least one iteration
                        body_start
                    }
                    QuantifierType::ZeroOrOne => {
                        // e? is choice between e and epsilon
                        let split = nfa.add_state();
                        nfa.add_epsilon_transition(split, cont);  // skip (epsilon path)
                        let body_start = Self::compile_cps(inner.as_ref(), nfa, cont, cache);
                        nfa.add_epsilon_transition(split, body_start);  // take path
                        split
                    }
                }
            }

            Expr::RepeatBounded { inner, min, max } => {
                let min = *min;
                match *max {
                    Some(max_val) => {
                        if min > max_val {
                            return nfa.add_state();
                        }

                        if max_val == min {
                            let mut s = cont;
                            for _ in 0..min {
                                s = Self::compile_cps(inner.as_ref(), nfa, s, cache);
                            }
                            return s;
                        }

                        let mut boundaries = Vec::with_capacity(max_val + 1);
                        for _ in 0..=max_val {
                            boundaries.push(nfa.add_state());
                        }

                        nfa.add_epsilon_transition(boundaries[max_val], cont);

                        for i in (0..max_val).rev() {
                            let start = Self::compile_cps(inner.as_ref(), nfa, boundaries[i + 1], cache);
                            nfa.add_epsilon_transition(boundaries[i], start);
                            if i >= min {
                                nfa.add_epsilon_transition(boundaries[i], cont);
                            }
                        }

                        boundaries[0]
                    }
                    None => {
                        let mut s = cont;
                        let loop_split = nfa.add_state();
                        nfa.add_epsilon_transition(loop_split, cont);

                        let body_start = Self::compile_cps(inner.as_ref(), nfa, loop_split, cache);
                        nfa.add_epsilon_transition(loop_split, body_start);

                        s = loop_split;
                        for _ in 0..min {
                            s = Self::compile_cps(inner.as_ref(), nfa, s, cache);
                        }
                        s
                    }
                }
            }
            
            Expr::Shared(inner_arc) => {
                // Key rule: cache by (Arc_ptr, cont) for safe sharing
                let id = Arc::as_ptr(inner_arc) as usize;
                let key = (id, cont);
                
                if let Some(&start) = cache.get(&key) {
                    // Cache hit: reuse the previously compiled fragment
                    return start;
                }
                
                // Cache miss: compile underlying expr with same continuation
                let start = Self::compile_cps(inner_arc.as_ref(), nfa, cont, cache);
                
                // Memoize for future reuse
                cache.insert(key, start);
                start
            }
        }
    }

    // --- Optimizers ---

    fn split_head(self) -> (Head, Option<Expr>) {
        match self {
            Expr::U8Seq(mut s) if !s.is_empty() => {
                let h = s.remove(0);
                let t = if s.is_empty() { None } else { Some(Expr::U8Seq(s)) };
                (Head::Class(U8Set::from_u8(h)), t)
            },
            Expr::U8Class(c) => (Head::Class(c), None),
            Expr::Seq(mut s) if !s.is_empty() => {
                let first = s.remove(0);
                let (head, first_tail) = first.split_head();
                
                match head {
                    Head::Other => {
                        // Reconstruction is hard here without cloning the original 'first', 
                        // but since Head::Other implies we can't optimize, we handle it loosely.
                        // We'll return Other and re-wrap the minimized tail if possible, 
                        // but for now, just returning Other with a reconstructed Seq is enough to bail out.
                        let mut reconstructed = vec![first_tail.unwrap_or(Expr::Epsilon)];
                        // Note: ^ logic slightly lossy if we don't return exact original, 
                        // but split_head is internal heuristic. 
                        // Correct approach: if Other, return original 'self' in tail? 
                        // For simplicity in this patch, we treat Other as a barrier.
                        // But for simplicity in this patch, we treat Other as a barrier.
                        reconstructed.extend(s);
                        (Head::Other, Some(Expr::make_seq(reconstructed)))
                    },
                    Head::Class(c) => {
                        let tail = match first_tail {
                            Some(t) => { s.insert(0, t); Some(Self::make_seq(s)) },
                            None => if s.is_empty() { None } else { Some(Self::make_seq(s)) }
                        };
                        (Head::Class(c), tail)
                    }
                }
            },
            Expr::Quantifier(inner, QuantifierType::OneOrMore) => {
                // Rep1(A) -> A . Rep(A)
                let (head, tail_opt) = inner.as_ref().clone().split_head();
                match head {
                    Head::Class(c) => {
                        let rep_a = Expr::Quantifier(inner, QuantifierType::ZeroOrMore);
                        let tail = match tail_opt {
                            Some(t) => Some(Self::make_seq(vec![t, rep_a])),
                            None => Some(rep_a)
                        };
                        (Head::Class(c), tail)
                    }
                    Head::Other => (Head::Other, Some(Expr::Quantifier(inner, QuantifierType::OneOrMore))),
                }
            },
            x => (Head::Other, Some(x)),
        }
    }

    pub fn optimize(self) -> Self {
        enum Task {
            Expand(Expr),
            Seq(usize),
            Choice(usize),
            Quantifier(QuantifierType),
            RepeatBounded { min: usize, max: Option<usize> },
            Shared(usize, Arc<Expr>),
        }

        let mut stack = vec![Task::Expand(self)];
        let mut values: Vec<Expr> = Vec::with_capacity(32);
        let mut cache: HashMap<usize, Expr> = HashMap::new();
        let mut visiting: HashSet<usize> = HashSet::new();
        
        let mut iterations: u64 = 0;
        let mut seq_calls: u64 = 0;
        let mut choice_calls: u64 = 0;
        let start = std::time::Instant::now();

        while let Some(task) = stack.pop() {
            iterations += 1;
            if iterations % 100_000 == 0 {
                crate::debug!(5, "optimize iter={} seq={} choice={} stack={} values={} elapsed={:.2?}", 
                    iterations, seq_calls, choice_calls, stack.len(), values.len(), start.elapsed());
            }
            
            match task {
                Task::Expand(expr) => match expr {
                    Expr::Seq(sub) => {
                        stack.push(Task::Seq(sub.len()));
                        for x in sub.into_iter().rev() {
                            stack.push(Task::Expand(x));
                        }
                    }
                    Expr::Choice(sub) => {
                        stack.push(Task::Choice(sub.len()));
                        for x in sub.into_iter().rev() {
                            stack.push(Task::Expand(x));
                        }
                    }
                    Expr::Quantifier(sub, q) => {
                        stack.push(Task::Quantifier(q));
                        stack.push(Task::Expand(*sub));
                    }
                    Expr::RepeatBounded { inner, min, max } => {
                        stack.push(Task::RepeatBounded { min, max });
                        stack.push(Task::Expand(*inner));
                    }
                    Expr::Shared(inner) => {
                        let ptr = Arc::as_ptr(&inner) as usize;
                        if let Some(res) = cache.get(&ptr) {
                            values.push(res.clone());
                        } else if visiting.contains(&ptr) {
                            values.push(Expr::Shared(inner));
                        } else {
                            visiting.insert(ptr);
                            stack.push(Task::Shared(ptr, inner.clone()));
                            stack.push(Task::Expand(inner.as_ref().clone()));
                        }
                    }
                    leaf => values.push(leaf),
                },
                Task::Seq(len) => {
                    seq_calls += 1;
                    let split_idx = values.len() - len;
                    let children = values.split_off(split_idx);
                    values.push(Self::make_seq(children));
                }
                Task::Quantifier(q) => {
                    let child = values.pop().unwrap();
                    
                    // Minimize nested quantifiers: (expr*)*  = expr*,  (expr+)* = expr*, etc.
                    // We need to unwrap Shared to check the inner structure
                    let minimized = match (&child, q) {
                        // If child is Shared, peek inside
                        (Expr::Shared(inner_arc), outer_q) => {
                            match inner_arc.as_ref() {
                                Expr::Quantifier(inner_expr, inner_q) => {
                                    // Minimization rules (regex quantifier algebra):
                                    // (a*)* = a*, (a+)* = a*, (a?)* = a*
                                    // (a*)+ = a*, (a+)+ = a+, (a?)+ = a*
                                    // (a*)? = a*, (a+)? = a*, (a?)? = a?
                                    use QuantifierType::*;
                                    match (inner_q, outer_q) {
                                        (ZeroOrMore, ZeroOrMore) => child, // a** = a*
                                        (OneOrMore, ZeroOrMore) => {
                                            // a+* = a* - rebuild with inner expr and ZeroOrMore
                                            Expr::Shared(Arc::new(Expr::Quantifier(
                                                inner_expr.clone(),
                                                ZeroOrMore
                                            )))
                                        }
                                        (ZeroOrOne, ZeroOrMore) => {
                                            // a?* = a* - rebuild with inner expr and ZeroOrMore
                                            Expr::Shared(Arc::new(Expr::Quantifier(
                                                inner_expr.clone(),
                                                ZeroOrMore
                                            )))
                                        }
                                        (ZeroOrMore, OneOrMore) => child, // a*+ = a*
                                        (OneOrMore, OneOrMore) => child,  // a++ = a+
                                        (ZeroOrOne, OneOrMore) => {
                                            // a?+ = a* (one or more of zero-or-one is zero-or-more)
                                            Expr::Shared(Arc::new(Expr::Quantifier(
                                                inner_expr.clone(),
                                                ZeroOrMore
                                            )))
                                        }
                                        (ZeroOrMore, ZeroOrOne) => child,  // a*? = a*
                                        (OneOrMore, ZeroOrOne) => {
                                            // a+? = a* (zero or one of one-or-more is zero-or-more)
                                            Expr::Shared(Arc::new(Expr::Quantifier(
                                                inner_expr.clone(),
                                                ZeroOrMore
                                            )))
                                        }
                                        (ZeroOrOne, ZeroOrOne) => child,   // a?? = a?
                                    }
                                }
                                _ => Expr::Quantifier(Box::new(child), q)
                            }
                        }
                        // Direct nested quantifier (not wrapped in Shared)
                        (Expr::Quantifier(inner_expr, inner_q), outer_q) => {
                            use QuantifierType::*;
                            match (inner_q, outer_q) {
                                (ZeroOrMore, ZeroOrMore) => child, // a** = a*
                                (OneOrMore, ZeroOrMore) => {
                                    // a+* = a* - rebuild as a*
                                    Expr::Quantifier(inner_expr.clone(), ZeroOrMore)
                                }
                                (ZeroOrOne, ZeroOrMore) => {
                                    // a?* = a* - rebuild as a*
                                    Expr::Quantifier(inner_expr.clone(), ZeroOrMore)
                                }
                                (ZeroOrMore, OneOrMore) => child, // a*+ = a*
                                (OneOrMore, OneOrMore) => child,  // a++ = a+
                                (ZeroOrOne, OneOrMore) => {
                                    // a?+ = a* - rebuild as a*
                                    Expr::Quantifier(inner_expr.clone(), ZeroOrMore)
                                }
                                (ZeroOrMore, ZeroOrOne) => child, // a*? = a*
                                (OneOrMore, ZeroOrOne) => {
                                    // a+? = a* - rebuild as a*
                                    Expr::Quantifier(inner_expr.clone(), ZeroOrMore)
                                }
                                (ZeroOrOne, ZeroOrOne) => child, // a?? = a?
                            }
                        }
                        _ => Expr::Quantifier(Box::new(child), q)
                    };
                    
                    values.push(minimized);
                }
                Task::RepeatBounded { min, max } => {
                    let child = values.pop().unwrap();
                    let simplified = match (min, max) {
                        (0, Some(0)) => Expr::Epsilon,
                        (0, Some(1)) => Expr::Quantifier(Box::new(child), QuantifierType::ZeroOrOne),
                        (1, Some(1)) => child,
                        (0, None) => Expr::Quantifier(Box::new(child), QuantifierType::ZeroOrMore),
                        (1, None) => Expr::Quantifier(Box::new(child), QuantifierType::OneOrMore),
                        _ => Expr::RepeatBounded {
                            inner: Box::new(child),
                            min,
                            max,
                        },
                    };
                    values.push(simplified);
                }
                Task::Shared(ptr, _) => {
                    let child = values.pop().unwrap();
                    visiting.remove(&ptr);
                    let res = Expr::Shared(Arc::new(child));
                    cache.insert(ptr, res.clone());
                    values.push(res);
                }
                Task::Choice(len) => {
                    choice_calls += 1;
                    let split_idx = values.len() - len;
                    let children = values.split_off(split_idx);
                    values.push(Self::make_choice(children));
                }
            }
        }
        crate::debug!(6, "optimize done: iter={} seq={} choice={} elapsed={:.2?}",
            iterations, seq_calls, choice_calls, start.elapsed());
        values.pop().unwrap()
    }

    pub fn strip_prefix(&self, prefix: &Expr) -> Option<Expr> {
        if self == prefix {
            return Some(Expr::Epsilon);
        }

        match self {
            Expr::Seq(exprs) => {
                if exprs.is_empty() {
                    return None;
                }
                // Check if the first element matches the prefix
                if &exprs[0] == prefix {
                    // Return the rest as a sequence
                    return Some(Expr::make_seq(exprs[1..].to_vec()));
                }
                // TODO: Handle case where prefix matches a prefix of exprs[0]?
                // For now, we only handle exact match of the first element,
                // which is sufficient for the "ignore . core" pattern.
                None
            }
            Expr::Shared(inner) => inner.strip_prefix(prefix),
            _ => None,
        }
    }

    pub fn make_seq(exprs: Vec<Expr>) -> Expr {
        let mut flat = Vec::with_capacity(exprs.len());
        for e in exprs {
            if let Expr::Seq(subs) = e {
                flat.extend(subs);
            } else if matches!(e, Expr::Epsilon) {
                continue;
            } else {
                flat.push(e);
            }
        }
        
        if flat.is_empty() {
            return Expr::Epsilon;
        }
        
        // Normalize: U8Class(size 1) -> U8Seq
        for e in &mut flat {
            if let Expr::U8Class(ref set) = e {
                if set.len() == 1 {
                    let b = set.iter().next().unwrap();
                    *e = Expr::U8Seq(vec![b]);
                }
            }
        }

        let mut merged = Vec::with_capacity(flat.len());
        for e in flat {
            if let Expr::U8Seq(mut curr) = e {
                if let Some(Expr::U8Seq(prev)) = merged.last_mut() {
                    prev.append(&mut curr);
                } else {
                    merged.push(Expr::U8Seq(curr));
                }
            } else {
                merged.push(e);
            }
        }
        
        if merged.len() == 1 {
            merged.pop().unwrap()
        } else {
            Expr::Seq(merged)
        }
    }

    fn make_choice(exprs: Vec<Expr>) -> Expr {
        // Simple implementation: just flatten and dedup (without expensive sorting)
        // Sorting was causing O(n * depth) comparisons per element which is expensive
        
        // 1. Flatten nested choices
        let mut worklist = exprs;
        let mut flat = Vec::with_capacity(worklist.len());

        while let Some(e) = worklist.pop() {
            match e {
                Expr::Choice(subs) => {
                    worklist.extend(subs);
                }
                _ => flat.push(e),
            }
        }

        if flat.is_empty() {
            return Expr::Choice(vec![]);
        }
        
        if flat.len() == 1 {
            return flat.pop().unwrap();
        }

        // 2. Merge U8Class and single-byte U8Seq alternatives (quick dedup)
        let mut classes = U8Set::none();
        let mut complex = Vec::with_capacity(flat.len());
        
        for e in flat.into_iter() {
            match e {
                Expr::U8Class(c) => classes.update(&c),
                Expr::U8Seq(s) if s.len() == 1 => { classes.insert(s[0]); },
                _ => complex.push(e),
            }
        }
        
        if !classes.is_empty() {
            complex.push(Expr::U8Class(classes));
        }
        
        if complex.len() == 1 {
            complex.pop().unwrap()
        } else {
            Expr::Choice(complex)
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
enum Head { Class(U8Set), Other }

#[derive(Debug, Default)]
struct DFAConversionStats {
    total_time: std::time::Duration,
    class_computation_time: std::time::Duration,
    remapped_transitions_time: std::time::Duration,
    dfa_metadata_time: std::time::Duration,
    main_loop_time: std::time::Duration,

    dfa_states_created: usize,
    max_subset_size: usize,
    total_subset_size: u64,
    max_worklist_len: usize,
}

impl std::fmt::Display for DFAConversionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let avg_subset_size = if self.dfa_states_created > 0 {
            self.total_subset_size as f64 / self.dfa_states_created as f64
        } else { 0.0 };

        writeln!(f, "--- DFA Conversion Stats ---")?;
        writeln!(f, "Total time: {:.2?}", self.total_time)?;
        writeln!(f, "  - Input Equivalence Classes: {:.2?}", self.class_computation_time)?;
        writeln!(f, "  - Remap NFA transitions:     {:.2?}", self.remapped_transitions_time)?;
        writeln!(f, "  - Main Loop:                 {:.2?}", self.main_loop_time)?;
        writeln!(f, "  - DFA Metadata:              {:.2?}", self.dfa_metadata_time)?;
        
        writeln!(f, "\nDFA size:")?;
        writeln!(f, "  - States created: {}", self.dfa_states_created)?;
        writeln!(f, "  - Max NFA subset size: {}", self.max_subset_size)?;
        writeln!(f, "  - Avg NFA subset size: {:.2}", avg_subset_size)?;
        writeln!(f, "  - Max worklist length: {}", self.max_worklist_len)?;
        Ok(())
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

            if let Some(m) = state.finalizers.iter().max() {
                if m > max_group_id { max_group_id = m; }
            }
            if let Some(m) = state.non_greedy_finalizers.iter().max() {
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

        crate::debug!(5, "--- NFA Stats ---");
        crate::debug!(5, "States: {}", num_states);
        crate::debug!(5, "Estimated Size: {:.2} MB", to_mb(total_estimated_bytes));
        crate::debug!(5, "  Base (Vec headers, etc): {:.2} MB", to_mb(total_base_size));
        crate::debug!(5, "  Transitions Data: {:.2} MB", to_mb(transitions_capacity_bytes));
        crate::debug!(5, "  Epsilon Data: {:.2} MB", to_mb(epsilon_capacity_bytes));
        crate::debug!(5, "  Finalizers (est): {:.2} MB", to_mb(finalizers_est_bytes + non_greedy_est_bytes));

        // 1. Finalizer Sets -> Bitsets
        // Cost of Bitset: (max_group_id bits / 8) per set. Two sets per state.
        let words_per_set = (max_group_id / 64) + 1;
        let bytes_per_set = words_per_set * 8;
        // Overhead of Vec<u64> is 24 bytes.
        let bitset_overhead = 24;
        let total_bitset_cost = num_states * 2 * (bytes_per_set + bitset_overhead);
        let current_finalizer_cost = finalizers_est_bytes + non_greedy_est_bytes;
        let savings_bitsets = (current_finalizer_cost as isize) - (total_bitset_cost as isize);
        crate::debug!(5, "  [Savings] Finalizers -> Bitsets: {:.2} MB (current est: {:.2} MB, bitset: {:.2} MB)",
            to_mb(savings_bitsets.max(0) as usize), to_mb(current_finalizer_cost), to_mb(total_bitset_cost));

        // 2. State IDs u32
        // (u8, usize) [16 bytes] -> (u8, u32) [8 bytes]. usize [8 bytes] -> u32 [4 bytes].
        let current_trans_data_used = total_transitions_count * std::mem::size_of::<(u8, usize)>();
        let u32_trans_data_used = total_transitions_count * 8;
        let current_eps_data_used = total_epsilon_count * std::mem::size_of::<usize>();
        let u32_eps_data_used = total_epsilon_count * 4;
        let savings_u32 = (current_trans_data_used + current_eps_data_used) - (u32_trans_data_used + u32_eps_data_used);
        crate::debug!(5, "  [Savings] State IDs -> u32: {:.2} MB", to_mb(savings_u32));

        // 3. Compact Transitions
        // Vec<(u8, usize)> [16 bytes] -> Vec<(U8Set, usize)> [48 bytes: 32(set) + 8(usize) + 8(pad)]
        let compact_item_size = 48;
        let compact_total_size = compacted_transitions_count * compact_item_size;
        let savings_compact = (current_trans_data_used as isize) - (compact_total_size as isize);
        crate::debug!(5, "  [Savings] Compact Transitions: {:.2} MB (current: {:.2} MB, compacted: {:.2} MB)",
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
            crate::debug!(5, "  SCCs: {}, Non-trivial (size>1): {} ({} states, avg size {:.2})",
                num_sccs, num_non_trivial_sccs, total_states_in_non_trivial_sccs, avg_non_trivial_scc_size);
        }
        crate::debug!(5, "-----------------");
    }

    fn condense_epsilon_sccs(&mut self) {
        let num_states = self.states.len();
        let mut disc = vec![-1i32; num_states];
        let mut low = vec![-1i32; num_states];
        let mut on_stack = vec![false; num_states];
        let mut stack: Vec<usize> = Vec::new();
        let mut time = 0i32;
        let mut scc_map = vec![0usize; num_states];
        let mut scc_count = 0;

        let mut work_stack: Vec<(usize, usize)> = Vec::new();

        for i in 0..num_states {
            if disc[i] != -1 {
                continue;
            }

            work_stack.push((i, 0));
            while let Some((u, idx)) = work_stack.pop() {
                if idx == 0 {
                    disc[u] = time;
                    low[u] = time;
                    time += 1;
                    stack.push(u);
                    on_stack[u] = true;
                }

                let neighbors = &self.states[u].epsilon_transitions;
                if idx < neighbors.len() {
                    let v = neighbors[idx];
                    work_stack.push((u, idx + 1));
                    if disc[v] == -1 {
                        work_stack.push((v, 0));
                    } else if on_stack[v] {
                        low[u] = low[u].min(disc[v]);
                    }
                } else {
                    if low[u] == disc[u] {
                        loop {
                            let v = stack.pop().unwrap();
                            on_stack[v] = false;
                            scc_map[v] = scc_count;
                            if u == v {
                                break;
                            }
                        }
                        scc_count += 1;
                    }

                    if let Some((parent, _)) = work_stack.last() {
                        low[*parent] = low[*parent].min(low[u]);
                    }
                }
            }
        }

        if scc_count == num_states {
            return;
        }

        crate::debug!(4, "Condensing NFA: {} states -> {} states ({} SCCs)", num_states, scc_count, scc_count);

        let mut new_states = Vec::with_capacity(scc_count);
        for _ in 0..scc_count {
            new_states.push(NFAState::new());
        }

        for (old_id, state) in self.states.iter().enumerate() {
            let new_id = scc_map[old_id];
            let new_state = &mut new_states[new_id];

            new_state.finalizers.union_with(&state.finalizers);
            new_state.non_greedy_finalizers.union_with(&state.non_greedy_finalizers);

            for (u8set, target) in &state.transitions {
                let new_target = scc_map[*target];
                new_state.transitions.push((u8set.clone(), new_target));
            }

            for &target in &state.epsilon_transitions {
                let new_target = scc_map[target];
                if new_target != new_id {
                    new_state.epsilon_transitions.push(new_target);
                }
            }
        }

        for state in &mut new_states {
            state.epsilon_transitions.sort_unstable();
            state.epsilon_transitions.dedup();
        }

        self.states = new_states;
        self.start_state = scc_map[self.start_state];
    }

    /// Eliminate epsilon transitions by computing epsilon closures and inlining transitions.
    /// This converts the NFA to an epsilon-free form, making DFA construction much faster.
    fn eliminate_epsilon_transitions(&mut self) {
        let num_states = self.states.len();
        if num_states == 0 {
            return;
        }

        // Count total epsilon transitions
        let total_epsilon: usize = self.states.iter().map(|s| s.epsilon_transitions.len()).sum();
        if total_epsilon == 0 {
            return;
        }
        
        crate::debug!(4, "Eliminating {} epsilon transitions from {} states", total_epsilon, num_states);
        let start = std::time::Instant::now();

        // Build compact epsilon closure representation for efficient BFS
        let mut epsilon_offsets = Vec::with_capacity(num_states + 1);
        let mut epsilon_targets = Vec::new();
        for state in &self.states {
            epsilon_offsets.push(epsilon_targets.len() as u32);
            for &target in &state.epsilon_transitions {
                epsilon_targets.push(target as u32);
            }
        }
        epsilon_offsets.push(epsilon_targets.len() as u32);

        // Compute epsilon closures for all states in parallel
        use rayon::prelude::*;
        let closures: Vec<Vec<usize>> = (0..num_states).into_par_iter().map(|state_idx| {
            let mut closure = Vec::new();
            let mut visited = vec![false; num_states];
            let mut stack = vec![state_idx];
            visited[state_idx] = true;
            closure.push(state_idx);
            
            while let Some(u) = stack.pop() {
                let start_offs = epsilon_offsets[u] as usize;
                let end_offs = epsilon_offsets[u + 1] as usize;
                for i in start_offs..end_offs {
                    let v = epsilon_targets[i] as usize;
                    if !visited[v] {
                        visited[v] = true;
                        closure.push(v);
                        stack.push(v);
                    }
                }
            }
            closure
        }).collect();

        // Now update each state's transitions by unioning transitions from all states in its closure
        let new_states: Vec<NFAState> = (0..num_states).into_par_iter().map(|state_idx| {
            let closure = &closures[state_idx];
            
            // Collect all transitions from states in closure
            let mut all_transitions: Vec<(U8Set, usize)> = Vec::new();
            let mut new_finalizers = self.states[state_idx].finalizers.clone();
            let mut new_non_greedy_finalizers = self.states[state_idx].non_greedy_finalizers.clone();
            
            for &closure_state in closure {
                for (u8set, target) in &self.states[closure_state].transitions {
                    all_transitions.push((u8set.clone(), *target));
                }
                new_finalizers.union_with(&self.states[closure_state].finalizers);
                new_non_greedy_finalizers.union_with(&self.states[closure_state].non_greedy_finalizers);
            }

            NFAState {
                transitions: all_transitions,
                epsilon_transitions: Vec::new(), // Eliminated!
                finalizers: new_finalizers,
                non_greedy_finalizers: new_non_greedy_finalizers,
            }
        }).collect();

        self.states = new_states;
        crate::debug!(4, "Epsilon elimination completed in {:.2?}", start.elapsed());
    }

    /// Precompute all epsilon closures using DAG-aware incremental computation.
    /// For states in topological order, closure(s) = {s} ∪ ⋃{closure(t) | s →ε t}
    /// This is more efficient than computing each closure independently with BFS.
    fn precompute_epsilon_closures_dag(compact_nfa: &CompactNFA, num_states: usize) -> Vec<Vec<u32>> {
        // First, compute reverse topological order via DFS post-order
        let mut visited = vec![false; num_states];
        let mut post_order = Vec::with_capacity(num_states);
        
        fn dfs_postorder(
            state: usize,
            compact_nfa: &CompactNFA,
            visited: &mut [bool],
            post_order: &mut Vec<usize>,
        ) {
            if visited[state] {
                return;
            }
            visited[state] = true;
            
            let start = compact_nfa.epsilon_offsets[state] as usize;
            let end = compact_nfa.epsilon_offsets[state + 1] as usize;
            for &target in &compact_nfa.epsilon_targets[start..end] {
                dfs_postorder(target as usize, compact_nfa, visited, post_order);
            }
            post_order.push(state);
        }
        
        // DFS from all states to handle disconnected components
        for state in 0..num_states {
            dfs_postorder(state, compact_nfa, &mut visited, &mut post_order);
        }
        
        // post_order is in reverse topological order (sinks first, sources last)
        // Process in this order so when we process state s, all its successors are done
        let mut closures: Vec<Vec<u32>> = vec![Vec::new(); num_states];
        let mut temp_set = SparseStateSet::new(num_states);
        
        for &state in &post_order {
            temp_set.clear();
            temp_set.insert(state);
            
            // Union in closures of all epsilon successors
            let start = compact_nfa.epsilon_offsets[state] as usize;
            let end = compact_nfa.epsilon_offsets[state + 1] as usize;
            for &target in &compact_nfa.epsilon_targets[start..end] {
                let target = target as usize;
                // Add all states from target's closure
                for &s in &closures[target] {
                    temp_set.insert(s as usize);
                }
            }
            
            // Convert to sorted vec by iterating over dirty words
            let mut closure_vec: Vec<u32> = Vec::new();
            for &w_idx in &temp_set.dirty_words {
                let mut w = temp_set.dense.words[w_idx];
                while w != 0 {
                    let t = w.trailing_zeros();
                    w &= !(1u64 << t);
                    closure_vec.push((w_idx * 64 + t as usize) as u32);
                }
            }
            closure_vec.sort_unstable();
            closures[state] = closure_vec;
        }
        
        closures
    }

    fn build_compact_nfa(&self) -> CompactNFA {
        let mut epsilon_offsets = Vec::with_capacity(self.states.len() + 1);
        let mut epsilon_targets = Vec::new();

        for state in &self.states {
            epsilon_offsets.push(epsilon_targets.len() as u32);
            for &target in &state.epsilon_transitions {
                epsilon_targets.push(target as u32);
            }
        }
        epsilon_offsets.push(epsilon_targets.len() as u32);
        
        crate::debug!(5, "CompactNFA: {} states, {} epsilon transitions", self.states.len(), epsilon_targets.len());
        
        // Debug: analyze epsilon structure
        let mut out_degrees: Vec<u32> = Vec::with_capacity(self.states.len());
        let mut max_out_degree = 0u32;
        let mut states_with_epsilon = 0;
        for i in 0..self.states.len() {
            let degree = epsilon_offsets[i + 1] - epsilon_offsets[i];
            out_degrees.push(degree);
            if degree > max_out_degree {
                max_out_degree = degree;
            }
            if degree > 0 {
                states_with_epsilon += 1;
            }
        }
        crate::debug!(5, "Epsilon structure: {} states have outgoing epsilons, max out-degree {}", 
            states_with_epsilon, max_out_degree);
        
        // Histogram of out-degrees
        let mut histogram: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for &d in &out_degrees {
            *histogram.entry(d).or_insert(0) += 1;
        }
        let significant: Vec<_> = histogram.iter()
            .filter(|(&d, _)| d >= 100)
            .map(|(&d, &c)| format!("{}:{}", d, c))
            .collect();
        if !significant.is_empty() {
            crate::debug!(5, "  High out-degree states: {}", significant.join(", "));
        }

        CompactNFA {
            epsilon_offsets,
            epsilon_targets,
        }
    }

    fn compute_equivalence_classes(&self) -> (Vec<u8>, usize, Vec<Vec<u8>>) {
        let mut partitions = vec![U8Set::all()];
        let mut seen_sets = HashSet::new();

        for state in &self.states {
            for (set, _) in &state.transitions {
                if seen_sets.insert(*set) {
                    let mut next_partitions = Vec::with_capacity(partitions.len() * 2);
                    for p in partitions {
                        let intersection = p.intersection(set);
                        let difference = p.difference(set);
                        if !intersection.is_empty() {
                            next_partitions.push(intersection);
                        }
                        if !difference.is_empty() {
                            next_partitions.push(difference);
                        }
                    }
                    partitions = next_partitions;
                }
            }
        }

        let mut class_map = vec![0u8; 256];
        let mut class_members = vec![Vec::new(); partitions.len()];

        for (i, p) in partitions.iter().enumerate() {
            for b in p.iter() {
                class_map[b as usize] = i as u8;
                class_members[i].push(b);
            }
        }

        (class_map, partitions.len(), class_members)
    }

    pub fn to_dfa(self) -> DFA {
        let profile_dfa_only = std::env::var("PROFILE_DFA_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if profile_dfa_only {
            crate::profiler::reset();
        }
        let dfa = self.to_dfa_impl();
        if profile_dfa_only {
            crate::profiler::print_summary();
        }

        // let nfa = dfa.to_nfa();
        // let start = std::time::Instant::now();
        // let _ = nfa.to_dfa_impl();
        // crate::debug!(2, "Deterministic NFA -> DFA benchmark: {:.2?}", start.elapsed());
        dfa
    }

    #[time_it]
    fn to_dfa_impl(self) -> DFA {
        let mut stats = DFAConversionStats::default();
        let start_time = std::time::Instant::now();
        let mut dfa_states: Vec<DFAState> = Vec::with_capacity(120_000);
        // Use FxHashMap for faster hashing
        use rustc_hash::FxHashMap;
        let mut dfa_state_map: FxHashMap<ActiveStateSet, usize> = FxHashMap::default();
        dfa_state_map.reserve(120_000);
        let mut worklist: Vec<ActiveStateSet> = Vec::with_capacity(2048);

        // Compute Input Equivalence Classes
        let start_classes = std::time::Instant::now();
        let (class_map, num_classes, class_members) = crate::time!("compute_equivalence_classes", self.compute_equivalence_classes());
        stats.class_computation_time = start_classes.elapsed();
        crate::debug!(4, "Computed {} input equivalence classes in {:.2?}", num_classes, stats.class_computation_time);

        let start_remap = std::time::Instant::now();
        // Pre-process NFA transitions to use class IDs
        let remapped_transitions = crate::time!("remap_transitions", {
            let mut remapped_transitions: Vec<Vec<(U8Set, usize)>> = Vec::with_capacity(self.states.len());
            for state in &self.states {
                let mut trans = Vec::with_capacity(state.transitions.len());
                for (u8set, target) in &state.transitions {
                    let mut class_set = U8Set::none();
                    for b in u8set.iter() {
                        class_set.insert(class_map[b as usize]);
                    }
                    trans.push((class_set, *target));
                }
                remapped_transitions.push(trans);
            }
            remapped_transitions
        });
        stats.remapped_transitions_time = start_remap.elapsed();

        // Shared buffers
        let num_nfa_states = self.states.len();
        let mut stack: Vec<usize> = Vec::with_capacity(num_nfa_states);
        let mut closure_set = SparseStateSet::new(num_nfa_states);

        // Compact NFA for faster BFS
        let compact_nfa = self.build_compact_nfa();
        
        // Pre-compute the number of outgoing epsilons per state for fast-path detection
        let out_degree: Vec<u32> = (0..num_nfa_states).map(|s| {
            (compact_nfa.epsilon_offsets[s + 1] - compact_nfa.epsilon_offsets[s])
        }).collect();
        
        // Count states with outgoing epsilons and total epsilon transitions
        let states_with_eps = out_degree.iter().filter(|&&d| d > 0).count();
        let total_eps = compact_nfa.epsilon_targets.len();
        
        // Use threshold of 10 for high-degree hub states
        // Based on empirical testing, this balances precomputation cost vs main loop savings
        let precompute_threshold: u32 = 10;
        crate::debug!(5, "Epsilon density: {} states with eps, {} total transitions ({}x states)", 
            states_with_eps, total_eps, total_eps / num_nfa_states.max(1));
        
        // Precompute closures for states with out-degree >= precompute_threshold
        // This avoids repeated BFS traversal from these states
        let precompute_start = std::time::Instant::now();
        let mut high_degree_closures: Vec<Option<Vec<u32>>> = vec![None; num_nfa_states];
        let mut num_precomputed = 0;
        
        // Process in reverse topological order for efficiency
        let mut visited = vec![false; num_nfa_states];
        let mut post_order = Vec::with_capacity(num_nfa_states);
        
        fn dfs_postorder_selective(
            state: usize,
            compact_nfa: &CompactNFA,
            out_degree: &[u32],
            threshold: u32,
            visited: &mut [bool],
            post_order: &mut Vec<usize>,
        ) {
            if visited[state] {
                return;
            }
            visited[state] = true;
            
            let start = compact_nfa.epsilon_offsets[state] as usize;
            let end = compact_nfa.epsilon_offsets[state + 1] as usize;
            for &target in &compact_nfa.epsilon_targets[start..end] {
                dfs_postorder_selective(target as usize, compact_nfa, out_degree, threshold, visited, post_order);
            }
            // Only add to post_order if this meets threshold
            if out_degree[state] >= threshold {
                post_order.push(state);
            }
        }
        
        // DFS from all states that meet threshold
        for state in 0..num_nfa_states {
            if out_degree[state] >= precompute_threshold {
                dfs_postorder_selective(state, &compact_nfa, &out_degree, precompute_threshold, &mut visited, &mut post_order);
            }
        }
        
        // Compute closures for high-degree states in post-order (successors first)
        let mut temp_set = SparseStateSet::new(num_nfa_states);
        for &state in &post_order {
            temp_set.clear();
            temp_set.insert(state);
            
            // BFS from this state, but use precomputed closures when available
            stack.push(state);
            while let Some(u) = stack.pop() {
                let start_offs = compact_nfa.epsilon_offsets[u] as usize;
                let end_offs = compact_nfa.epsilon_offsets[u + 1] as usize;
                
                for i in start_offs..end_offs {
                    let v = compact_nfa.epsilon_targets[i] as usize;
                    if temp_set.insert(v) {
                        // If v has precomputed closure, use it (and don't recurse)
                        if let Some(ref closure) = high_degree_closures[v] {
                            for &s in closure {
                                temp_set.insert(s as usize);
                            }
                        } else if out_degree[v] > 0 {
                            stack.push(v);
                        }
                    }
                }
            }
            
            // Store closure
            let mut closure_vec: Vec<u32> = Vec::new();
            for &w_idx in &temp_set.dirty_words {
                let mut w = temp_set.dense.words[w_idx];
                while w != 0 {
                    let t = w.trailing_zeros();
                    w &= !(1u64 << t);
                    closure_vec.push((w_idx * 64 + t as usize) as u32);
                }
            }
            closure_vec.sort_unstable();
            high_degree_closures[state] = Some(closure_vec);
            num_precomputed += 1;
        }
        let total_closure_size: usize = high_degree_closures.iter().filter_map(|c| c.as_ref().map(|v| v.len())).sum();
        let avg_closure_size = if num_precomputed > 0 { total_closure_size / num_precomputed } else { 0 };
        crate::debug!(5, "Precomputed {} high-degree closures (avg size {}) in {:.2?}", num_precomputed, avg_closure_size, precompute_start.elapsed());

        // Compute start state closure using BFS
        closure_set.insert(self.start_state);
        if out_degree[self.start_state] > 0 {
            stack.push(self.start_state);
            while let Some(u) = stack.pop() {
                let start_offs = compact_nfa.epsilon_offsets[u] as usize;
                let end_offs = compact_nfa.epsilon_offsets[u + 1] as usize;
                for i in start_offs..end_offs {
                    let v = compact_nfa.epsilon_targets[i] as usize;
                    if closure_set.insert(v) {
                        if out_degree[v] > 0 {
                            stack.push(v);
                        }
                    }
                }
            }
        }

        let start_state_set = CompressedStateSet::from_sparse(&closure_set);

        dfa_state_map.insert(start_state_set.clone(), 0);
        worklist.push(start_state_set.clone());
        stats.max_worklist_len = stats.max_worklist_len.max(worklist.len());

        let (finalizers, non_greedy_finalizers) = {
            let mut finalizers = DenseStateSet::empty();
            let mut non_greedy_finalizers = DenseStateSet::empty();
            for state in start_state_set.iter() {
                finalizers.union_with(&self.states[state].finalizers);
                non_greedy_finalizers.union_with(&self.states[state].non_greedy_finalizers);
            }
            (finalizers, non_greedy_finalizers)
        };

        dfa_states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers,
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        // Reusable structures
        let mut transition_targets: Vec<SparseStateSet> = (0..num_classes).map(|_| SparseStateSet::new(num_nfa_states)).collect();
        let mut used_classes: Vec<usize> = Vec::with_capacity(num_classes);
        let mut seen_class = vec![false; num_classes];
        let mut scratch_closure = CompressedStateSet::new();
        let mut sort_scratch: Vec<usize> = Vec::with_capacity(1024);

        let main_loop_start = std::time::Instant::now();
        
        // Additional stats for debugging
        let mut total_transitions_processed = 0u64;
        let mut total_input_classes_processed = 0u64;
        let mut total_cache_hits = 0u64;
        let mut total_cache_misses = 0u64;
        
        // Enable detailed timing only for debug level 5+
        let enable_timing = std::env::var("MACRO_DEBUG_LEVEL").ok()
            .and_then(|s| s.parse::<i32>().ok())
            .map(|l| l >= 5)
            .unwrap_or(false);
        
        // Timing buckets (in nanoseconds)
        let mut time_collect = 0u64;
        let mut time_closure = 0u64;
        let mut time_compress = 0u64;
        let mut time_lookup = 0u64;
        let mut time_insert = 0u64;

        let max_dfa_states = std::env::var("MAX_DFA_STATES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        
        let mut next_log_threshold = 1_000;
        while let Some(current_set) = worklist.pop() {
            if dfa_states.len() >= max_dfa_states {
                panic!("DFA state limit {} reached after {:.2?}. worklist: {}, max_subset_size: {}", 
                    max_dfa_states, start_time.elapsed(), worklist.len(), stats.max_subset_size);
            }
            
            stats.max_worklist_len = stats.max_worklist_len.max(worklist.len() + 1);
            let current_subset_len = current_set.len();
            stats.max_subset_size = stats.max_subset_size.max(current_subset_len);
            stats.total_subset_size += current_subset_len as u64;
            if dfa_states.len() >= next_log_threshold {
                crate::debug!(5, "DFA progress: {} states, worklist {}, subset size {} (max {}), elapsed {:.2?}", dfa_states.len(), worklist.len(), current_subset_len, stats.max_subset_size, start_time.elapsed());
                next_log_threshold += 1_000;
            }

            let current_dfa_state = *dfa_state_map
                .get(&current_set)
                .expect("DFA state set not found in map");

            // 1. Populate transition_targets for all inputs (COLLECT PHASE)
            let t0 = if enable_timing { Some(std::time::Instant::now()) } else { None };
            for state in current_set.iter() {
                    for (class_set, next_state) in unsafe { remapped_transitions.get_unchecked(state) } {
                        for class_id in class_set.iter() {
                            let idx = class_id as usize;
                            unsafe {
                                if !*seen_class.get_unchecked(idx) {
                                    *seen_class.get_unchecked_mut(idx) = true;
                                    used_classes.push(idx); 
                                }
                                transition_targets.get_unchecked_mut(idx).insert(*next_state);
                            }
                        }
                    }
            }
            if let Some(t) = t0 { time_collect += t.elapsed().as_nanos() as u64; }

            // 2. Process inputs (PROCESS PHASE)
            
            let mut dfa_transitions_vec: Vec<(u8, usize)> = Vec::with_capacity(used_classes.len() * 2);
            total_input_classes_processed += used_classes.len() as u64;
            
                for &class_id in &used_classes {
                    let target_set = unsafe { transition_targets.get_unchecked(class_id) };
                    
                    let t1 = if enable_timing { Some(std::time::Instant::now()) } else { None };
                    closure_set.clear();
                    
                    // Fast path: check if all states have no outgoing epsilons
                    let mut needs_bfs = false;
                    for &w_idx in &target_set.dirty_words {
                        let mut w = target_set.dense.words[w_idx];
                        while w != 0 {
                            let t = w.trailing_zeros();
                            w &= !(1u64 << t);
                            let next_state = w_idx * 64 + t as usize;
                            
                            // Check if this state has a precomputed closure
                            if let Some(ref closure) = unsafe { high_degree_closures.get_unchecked(next_state) } {
                                // Use bulk insertion for precomputed closures
                                closure_set.insert_many(closure);
                            } else {
                                closure_set.insert(next_state);
                                if unsafe { *out_degree.get_unchecked(next_state) } > 0 {
                                    needs_bfs = true;
                                    stack.push(next_state);
                                }
                            }
                        }
                    }
                    
                    // BFS Closure - only if needed for non-precomputed states
                    if needs_bfs {
                        while let Some(u) = stack.pop() {
                            let start_offs = unsafe { *compact_nfa.epsilon_offsets.get_unchecked(u) } as usize;
                            let end_offs = unsafe { *compact_nfa.epsilon_offsets.get_unchecked(u + 1) } as usize;
                            
                            for i in start_offs..end_offs {
                                let v = unsafe { *compact_nfa.epsilon_targets.get_unchecked(i) } as usize;
                                if closure_set.insert(v) {
                                    // Check if v has precomputed closure
                                    if let Some(ref closure) = unsafe { high_degree_closures.get_unchecked(v) } {
                                        closure_set.insert_many(closure);
                                    } else if unsafe { *out_degree.get_unchecked(v) } > 0 {
                                        stack.push(v);
                                    }
                                }
                            }
                        }
                    }
                    
                    if let Some(t) = t1 { time_closure += t.elapsed().as_nanos() as u64; }

                    // Compress state set
                    let t2 = if enable_timing { Some(std::time::Instant::now()) } else { None };
                    CompressedStateSet::reuse_from_sparse(&closure_set, &mut scratch_closure, &mut sort_scratch);
                    if let Some(t) = t2 { time_compress += t.elapsed().as_nanos() as u64; }

                    // Map lookup/insert
                    let t3 = if enable_timing { Some(std::time::Instant::now()) } else { None };
                    let next_state_idx = {
                        // First try lookup without cloning
                        if let Some(&existing) = dfa_state_map.get(&scratch_closure) {
                            total_cache_hits += 1;
                            if let Some(t) = t3 { time_lookup += t.elapsed().as_nanos() as u64; }
                            existing
                        } else {
                            total_cache_misses += 1;
                            if let Some(t) = t3 { time_lookup += t.elapsed().as_nanos() as u64; }
                            // Only clone once, reuse for map and worklist
                            let t4 = if enable_timing { Some(std::time::Instant::now()) } else { None };
                            let new_state_index = dfa_states.len();
                            
                            // Compute finalizers before moving the key
                            let (new_finalizers, new_non_greedy_finalizers) = {
                                let mut new_finalizers = DenseStateSet::empty();
                                let mut new_non_greedy_finalizers = DenseStateSet::empty();
                                for state in scratch_closure.iter() {
                                    new_finalizers.union_with(&self.states[state].finalizers);
                                    new_non_greedy_finalizers.union_with(&self.states[state].non_greedy_finalizers);
                                }
                                (new_finalizers, new_non_greedy_finalizers)
                            };
                            
                            // Clone once for map, once for worklist
                            let key = scratch_closure.clone();
                            worklist.push(key.clone());
                            dfa_state_map.insert(key, new_state_index);
                            stats.max_worklist_len = stats.max_worklist_len.max(worklist.len());

                            dfa_states.push(DFAState {
                                transitions: CharTransitions::new(),
                                finalizers: new_finalizers,
                                possible_future_group_ids: BTreeSet::new(),
                                group_id_to_u8set: BTreeMap::new(),
                            });

                            if let Some(t) = t4 { time_insert += t.elapsed().as_nanos() as u64; }
                            new_state_index
                        }
                    };

                    for &b in &class_members[class_id] {
                        dfa_transitions_vec.push((b, next_state_idx));
                    }
                }

            // Bulk insert transitions
            dfa_transitions_vec.sort_unstable_by_key(|k| k.0);
            dfa_states[current_dfa_state].transitions = CharTransitions::from_sorted_entries(dfa_transitions_vec);

            for &idx in &used_classes {
                 seen_class[idx] = false;
                 transition_targets[idx].clear();
            }
            used_classes.clear();
        }
        
        stats.main_loop_time = main_loop_start.elapsed();
        
        // Print timing breakdown only if detailed timing was enabled
        if enable_timing {
            crate::debug!(5, "  └─ Timing breakdown (total ns):");
            crate::debug!(5, "      - Collect: {}ms", time_collect / 1_000_000);
            crate::debug!(5, "      - Closure: {}ms", time_closure / 1_000_000);
            crate::debug!(5, "      - Compress: {}ms", time_compress / 1_000_000);
            crate::debug!(5, "      - Lookup: {}ms", time_lookup / 1_000_000);
            crate::debug!(5, "      - Insert: {}ms", time_insert / 1_000_000);
        }

        let mut dfa = DFA {
            states: dfa_states,
            start_state: 0,
            non_greedy_finalizers: BTreeSet::new(),
        };

        for state in &self.states {
            for gid in state.non_greedy_finalizers.iter() {
                dfa.non_greedy_finalizers.insert(gid);
            }
        }

        let meta_start = std::time::Instant::now();
        crate::time!("recompute_metadata", dfa.recompute_metadata());
        stats.dfa_metadata_time = meta_start.elapsed();
        
        stats.total_time = start_time.elapsed();
        stats.dfa_states_created = dfa.states.len();

        // Level 4: Brief summary. Level 5+: Full stats.
        crate::debug!(4, "NFA → DFA: {} → {} states ({:.2?})", self.states.len(), dfa.states.len(), stats.total_time);
        if crate::r#macro::is_debug_level_enabled(5) {
            // Print detailed timing breakdown at level 5+
            crate::debug!(5, "  └─ Class computation: {:.2?}", stats.class_computation_time);
            crate::debug!(5, "  └─ Remap transitions: {:.2?}", stats.remapped_transitions_time);
            crate::debug!(5, "  └─ Main loop: {:.2?}", stats.main_loop_time);
            crate::debug!(5, "  └─ Metadata: {:.2?}", stats.dfa_metadata_time);
            crate::debug!(5, "  └─ Input classes processed: {}, cache hits: {}, misses: {}", 
                total_input_classes_processed, total_cache_hits, total_cache_misses);
            let hit_rate = if total_input_classes_processed > 0 {
                (total_cache_hits as f64 / total_input_classes_processed as f64) * 100.0
            } else {
                0.0
            };
            crate::debug!(5, "  └─ Cache hit rate: {:.1}%", hit_rate);
        }

        dfa
    }
}

impl DFA {
    pub fn reorder_states(&mut self, old_to_new: &[usize]) {
        let n = self.states.len();
        if n == 0 {
            return;
        }
        if old_to_new.len() != n {
            panic!(
                "reorder_states: mapping length {} != num states {}",
                old_to_new.len(),
                n
            );
        }

        let mut seen = vec![false; n];
        for &new_idx in old_to_new {
            if new_idx >= n {
                panic!("reorder_states: new index {} out of bounds {}", new_idx, n);
            }
            if seen[new_idx] {
                panic!("reorder_states: duplicate new index {}", new_idx);
            }
            seen[new_idx] = true;
        }

        let mut new_states: Vec<Option<DFAState>> = vec![None; n];
        for (old_idx, state) in self.states.iter().enumerate() {
            let new_idx = old_to_new[old_idx];
            let mut new_state = state.clone();
            new_state.transitions = new_state
                .transitions
                .iter()
                .map(|(u8, &old_next)| (u8, old_to_new[old_next]))
                .collect();
            new_states[new_idx] = Some(new_state);
        }

        self.states = new_states
            .into_iter()
            .map(|s| s.expect("reorder_states: missing state"))
            .collect();
        self.start_state = old_to_new[self.start_state];
        self.recompute_metadata();
    }

    #[time_it]
    pub fn to_nfa(&self) -> NFA {
        let mut states = Vec::with_capacity(self.states.len());

        for state in &self.states {
            let mut target_groups: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
            for (byte, target) in &state.transitions {
                target_groups.entry(*target).or_default().push(byte);
            }

            let mut transitions = Vec::with_capacity(target_groups.len());
            for (target, bytes) in target_groups {
                let mut set = U8Set::none();
                for b in bytes {
                    set.insert(b);
                }
                transitions.push((set, target));
            }

            let mut non_greedy_finalizers = DenseStateSet::empty();
            for gid in state.finalizers.iter() {
                if self.non_greedy_finalizers.contains(&gid) {
                    non_greedy_finalizers.insert(gid);
                }
            }

            states.push(NFAState {
                transitions,
                epsilon_transitions: Vec::new(),
                finalizers: state.finalizers.clone(),
                non_greedy_finalizers,
            });
        }

        NFA {
            states,
            start_state: self.start_state,
        }
    }

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
            .flat_map(|s| s.finalizers.iter())
            .max()
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

                    for gid in &self.states[target_idx].finalizers {
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
        use rayon::prelude::*;
        
        let num_states = self.states.len();
        
        // Parallelize the computation for each state
        let all_maps: Vec<BTreeMap<GroupID, U8Set>> = self.states.par_iter().map(|state| {
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
                    .copied()
                    .chain(next_state.finalizers.iter());

                for group_id in chain {
                    group_id_to_u8set
                        .entry(group_id)
                        .or_insert_with(U8Set::none)
                        .update(&inputs);
                }
            }
            group_id_to_u8set
        }).collect();

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
        let instant = std::time::Instant::now();
        if self.states.is_empty() {
            return;
        }
        
        let initial_states = self.states.len();
        self.remove_unreachable_states();
        let n = self.states.len();
        
        if n <= 1 {
            self.recompute_metadata();
            crate::debug!(3, "Minimized DFA {} → {} states in {:.2?}", initial_states, n, instant.elapsed());
            return;
        }
        
        // Use Hopcroft's algorithm for O(n log n) minimization
        
        use rustc_hash::FxHashMap;
        
        // Build inverse transition table - grouped by (target, input) to make iteration efficient
        // inverse[target] = Vec<(input, source)>
        let mut inverse: Vec<Vec<(u8, u32)>> = vec![Vec::new(); n];
        for (src, state) in self.states.iter().enumerate() {
            for (input, &target) in &state.transitions {
                inverse[target].push((input, src as u32));
            }
        }
        // Sort by input for efficient grouping
        for inv in &mut inverse {
            inv.sort_unstable_by_key(|&(input, _)| input);
        }
        
        // Initial partition: group states by their finalizer set
        let mut partition = vec![0u32; n];  // partition[state] = block_id
        let mut blocks: Vec<Vec<u32>> = Vec::new();  // blocks[block_id] = list of states
        
        {
            let mut finalizer_to_block: FxHashMap<Vec<GroupID>, u32> = FxHashMap::default();
            for (state_idx, state) in self.states.iter().enumerate() {
                let key: Vec<GroupID> = state.finalizers.iter().collect();
                let block_idx = *finalizer_to_block.entry(key).or_insert_with(|| {
                    let idx = blocks.len() as u32;
                    blocks.push(Vec::new());
                    idx
                });
                partition[state_idx] = block_idx;
                blocks[block_idx as usize].push(state_idx as u32);
            }
        }
        
        // Worklist: blocks to process
        let mut worklist: VecDeque<u32> = (0..blocks.len() as u32).collect();
        let mut in_worklist = vec![true; blocks.len()];
        
        // Reusable arrays for the inner loop
        let mut source_set = vec![false; n];
        let mut sources_to_clear: Vec<u32> = Vec::with_capacity(n.min(10000));
        let mut touched_blocks: Vec<u32> = Vec::with_capacity(1024);
        let mut block_touched = vec![false; blocks.len()];
        
        while let Some(splitter_block) = worklist.pop_front() {
            let splitter_idx = splitter_block as usize;
            if splitter_idx >= in_worklist.len() {
                continue;
            }
            in_worklist[splitter_idx] = false;
            
            if splitter_idx >= blocks.len() || blocks[splitter_idx].is_empty() {
                continue;
            }
            let splitter_states: Vec<u32> = blocks[splitter_idx].clone();
            
            // Gather all (input, source) pairs into a flat list and sort by input
            let mut all_pairs: Vec<(u8, u32)> = Vec::new();
            for &target in &splitter_states {
                all_pairs.extend_from_slice(&inverse[target as usize]);
            }
            
            if all_pairs.is_empty() {
                continue;
            }
            
            all_pairs.sort_unstable_by_key(|&(input, _)| input);
            
            // Process each input group
            let mut i = 0;
            while i < all_pairs.len() {
                let current_input = all_pairs[i].0;
                let group_start = i;
                
                // Find end of this input group and collect unique sources
                sources_to_clear.clear();
                while i < all_pairs.len() && all_pairs[i].0 == current_input {
                    let src = all_pairs[i].1;
                    if !source_set[src as usize] {
                        source_set[src as usize] = true;
                        sources_to_clear.push(src);
                        
                        let block_id = partition[src as usize] as usize;
                        if block_id < block_touched.len() && !block_touched[block_id] {
                            block_touched[block_id] = true;
                            touched_blocks.push(block_id as u32);
                        }
                    }
                    i += 1;
                }
                
                // Process each touched block for this input
                for &block_id in &touched_blocks {
                    let block_idx = block_id as usize;
                    if block_idx >= blocks.len() {
                        continue;
                    }
                    let block_len = blocks[block_idx].len();
                    if block_len <= 1 {
                        continue;
                    }
                    
                    // Count sources in this block
                    let mut source_count = 0usize;
                    for &state in &blocks[block_idx] {
                        if source_set[state as usize] {
                            source_count += 1;
                        }
                    }
                    
                    // No split if all or none
                    if source_count == 0 || source_count == block_len {
                        continue;
                    }
                    
                    // Split the block
                    let new_block_idx = blocks.len();
                    let move_sources = source_count <= block_len - source_count;
                    
                    let mut new_block = Vec::with_capacity(if move_sources { source_count } else { block_len - source_count });
                    let mut remaining = Vec::with_capacity(block_len - new_block.capacity());
                    
                    for &state in &blocks[block_idx] {
                        let is_source = source_set[state as usize];
                        if move_sources == is_source {
                            new_block.push(state);
                        } else {
                            remaining.push(state);
                        }
                    }
                    
                    // Update partitions
                    for &state in &new_block {
                        partition[state as usize] = new_block_idx as u32;
                    }
                    
                    blocks[block_idx] = remaining;
                    blocks.push(new_block);
                    
                    // Extend tracking arrays
                    in_worklist.push(false);
                    block_touched.push(false);
                    
                    // Add smaller block to worklist (Hopcroft's optimization)
                    if in_worklist[block_idx] {
                        in_worklist[new_block_idx] = true;
                        worklist.push_back(new_block_idx as u32);
                    } else {
                        if blocks[block_idx].len() <= blocks[new_block_idx].len() {
                            in_worklist[block_idx] = true;
                            worklist.push_back(block_idx as u32);
                        } else {
                            in_worklist[new_block_idx] = true;
                            worklist.push_back(new_block_idx as u32);
                        }
                    }
                }
                
                // Clear source_set only for elements we set
                for &src in &sources_to_clear {
                    source_set[src as usize] = false;
                }
                
                // Reset touched blocks
                for &block_id in &touched_blocks {
                    if (block_id as usize) < block_touched.len() {
                        block_touched[block_id as usize] = false;
                    }
                }
                touched_blocks.clear();
            }
        }
        
        // Convert blocks to partition_list format
        let partition_list: Vec<BTreeSet<usize>> = blocks
            .into_iter()
            .filter(|b| !b.is_empty())
            .map(|b| b.into_iter().map(|s| s as usize).collect())
            .collect();
        
        let (state_mapping, new_states) = self.rebuild_from_partitions(partition_list);

        self.states = new_states;
        self.start_state = state_mapping[self.start_state];

        self.recompute_metadata();
        
        crate::debug!(3, "Minimized DFA {} → {} states in {:.2?}", initial_states, self.states.len(), instant.elapsed());
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
                for group_id in &dfa.states[self.current_state].finalizers {
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
                max_gid = Some(max_gid.map_or(m, |cur| cur.max(m)));
            }
        }
        max_gid.map(|m| m + 1).unwrap_or(0)
    }

    pub fn execute_from_state_nonzero(&self, text: &[u8], state: usize) -> ExecutionResult {
        let mut regex_state = self.init_to_state(state);
        regex_state.execute(text);

        let matches: Vec<_> = regex_state.matches.iter().map(|(&id, &width)| Match { group_id: id, position: width })
            // Filter out zero-width tokens
            .filter(|token| token.position != 0).collect();

        let new_state = if regex_state.done { None } else { Some(regex_state.current_state) };

        ExecutionResult { matches, end_state: new_state }
    }

    pub fn execute_from_state2(&self, text: &[u8], state: usize) -> ExecutionResult {
        self.execute_from_state_fast(text, state)
    }

    pub fn execute_from_state_fast(&self, text: &[u8], state: usize) -> ExecutionResult {
        let dfa = &self.dfa;
        let mut all_matches: Vec<Match> = Vec::new();

        let mut current_state = state;
        let mut matched_groups: BTreeSet<GroupID> = dfa.states[state].finalizers.iter().collect();

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
                for group_id in &state_data.finalizers {
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
            .map(|group_id| (group_id, 0))
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

    pub fn generate_token_trellis(&self, bytes: &[u8], start_state: usize) -> TokenTrellis {
        let flat = self.generate_flat_trellis(bytes, start_state);
        Self::hydrate_trellis(flat, |idx| idx)
    }

    pub fn convert_token_trellis_into_completion(
        &self,
        trellis: TokenTrellis,
    ) -> TokenTrellisWithCompletion {
        let mut memo = HashMap::new();
        let end_state = trellis.end_state.map(|idx| self.dfa.states[idx].possible_future_group_ids.clone());
        let mut edges = BTreeMap::new();
        for (gid, child) in trellis.edges {
            edges.insert(gid, self.convert_trellis_node(&child, &mut memo));
        }
        Trellis { end_state, edges }
    }

    pub fn generate_token_trellis_with_completion(
        &self,
        bytes: &[u8],
        start_state: usize,
    ) -> TokenTrellisWithCompletion {
        let flat = self.generate_flat_trellis(bytes, start_state);
        Self::hydrate_trellis(flat, |idx| {
            self.dfa.states[idx].possible_future_group_ids.clone()
        })
    }

    fn generate_flat_trellis(
        &self,
        bytes: &[u8],
        start_state: usize,
    ) -> BTreeMap<usize, (Option<usize>, Vec<(GroupID, usize)>)> {
        let mut flat_trellis = BTreeMap::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(0);
        visited.insert(0);
        flat_trellis.insert(0, (None, Vec::new()));

        while let Some(pos) = queue.pop_front() {
            let slice = if pos <= bytes.len() {
                &bytes[pos..]
            } else {
                &[]
            };

            let exec_start = if pos == 0 {
                start_state
            } else {
                self.dfa.start_state
            };
            let result = self.execute_from_state_nonzero(slice, exec_start);

            let mut new_edges = Vec::new();
            for m in result.matches {
                let target_pos = pos + m.position;
                new_edges.push((m.group_id, target_pos));
                if visited.insert(target_pos) {
                    queue.push_back(target_pos);
                    flat_trellis.insert(target_pos, (None, Vec::new()));
                }
            }
            let node = flat_trellis.get_mut(&pos).unwrap();
            node.0 = result.end_state;
            node.1 = new_edges;
        }
        flat_trellis
    }

    fn hydrate_trellis<T, F>(
        flat_trellis: BTreeMap<usize, (Option<usize>, Vec<(GroupID, usize)>)>,
        state_mapper: F,
    ) -> Trellis<T>
    where
        T: Clone,
        F: Fn(usize) -> T,
    {
        let mut memo: HashMap<usize, Arc<Trellis<T>>> = HashMap::new();
        for (pos, (end_state, edges_list)) in flat_trellis.into_iter().rev() {
            let mut edges = BTreeMap::new();
            for (gid, target_pos) in edges_list {
                if let Some(target_node) = memo.get(&target_pos) {
                    if edges.insert(gid, target_node.clone()).is_some() {
                        panic!("Multiple edges for group ID {} at pos {}", gid, pos);
                    }
                }
            }
            let node = Trellis {
                end_state: end_state.map(&state_mapper),
                edges,
            };
            memo.insert(pos, Arc::new(node));
        }

        let root = memo.remove(&0).expect("Root node must exist");
        Arc::try_unwrap(root).unwrap_or_else(|arc| (*arc).clone())
    }

    fn convert_trellis_node(
        &self,
        node: &Arc<TokenTrellis>,
        memo: &mut HashMap<*const TokenTrellis, Arc<TokenTrellisWithCompletion>>,
    ) -> Arc<TokenTrellisWithCompletion> {
        let key = Arc::as_ptr(node);
        if let Some(res) = memo.get(&key) {
            return res.clone();
        }

        let end_state = node.end_state.map(|idx| {
            self.dfa.states[idx].possible_future_group_ids.clone()
        });

        let mut new_edges = BTreeMap::new();
        for (gid, child) in &node.edges {
            new_edges.insert(*gid, self.convert_trellis_node(child, memo));
        }

        let new_node = Arc::new(Trellis { end_state, edges: new_edges });
        memo.insert(key, new_node.clone());
        new_node
    }
}


