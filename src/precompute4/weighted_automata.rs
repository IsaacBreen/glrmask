// src/precompute4/weighted_automata.rs

#![allow(dead_code)]

use crate::json_serialization::{JSONConvertible, JSONNode};
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::iter::FromIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Deref, Index, IndexMut};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// --- Part 1: SimpleBitset ---

/// Weight is a finite (or cofinite) set of usize values.
/// We model weights as elements of the complete Boolean algebra (P(usize), ⊆)
/// with meet = set intersection (&), join = set union (|),
/// bottom = ∅ (zeros), top = U (all).
///
/// Composition (along transitions) uses meet (∧ = ∩).
/// Nondeterministic join (across alternative paths) uses join (∨ = ∪).
/// This forms a distributive lattice; all fixpoint computations below are
/// monotone and terminate because unions only ever add elements.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct SimpleBitset(pub RangeSetBlaze<usize>);

// --- Stochastic validation controls and RNG ---
// Toggle this to true to enable stochastic validation in union() and concatenate().
const STOCHASTIC_VALIDATION: bool = false;
const VALIDATION_SAMPLES: usize = 32;
const VALIDATION_MAX_STEPS: usize = 12;
const SAMPLING_TRIES: usize = 100;

#[derive(Clone, Debug)]
struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng(seed)
    }
    fn from_time() -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let mixed = (now.as_nanos() as u128 ^ ((now.as_secs() as u128) << 64)) as u64;
        SimpleRng::new(mixed)
    }
    fn next_u64(&mut self) -> u64 {
        // LCG constants from Numerical Recipes
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn gen_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }
    fn gen_bool_ratio(&mut self, num: u32, den: u32) -> bool {
        if den == 0 {
            true
        } else {
            (self.next_u64() % (den as u64)) < (num as u64)
        }
    }
}

// Small fixed alphabet used for default-edge sampling and variety.
// Includes ASCII letters/digits, some small integers, and negative-coded inputs used in tests.
const BASE_ALPHABET: &[i16] = &[
    b'a' as i16, b'b' as i16, b'c' as i16, b'd' as i16, b'e' as i16, b'f' as i16, b'g' as i16,
    b'h' as i16, b'i' as i16, b'j' as i16, b'k' as i16, b'l' as i16, b'm' as i16, b'n' as i16,
    b'o' as i16, b'p' as i16, b'q' as i16, b'r' as i16, b's' as i16, b't' as i16, b'u' as i16,
    b'v' as i16, b'w' as i16, b'x' as i16, b'y' as i16, b'z' as i16, b' ' as i16,
    b'0' as i16, b'1' as i16, b'2' as i16, b'3' as i16, b'4' as i16, b'5' as i16,
    b'6' as i16, b'7' as i16, b'8' as i16, b'9' as i16,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    i16::MIN + 0, i16::MIN + 1, i16::MIN + 2, i16::MIN + 3, i16::MIN + 4,
    i16::MIN + 5, i16::MIN + 6, i16::MIN + 7, i16::MIN + 8, i16::MIN + 9, i16::MIN + 10,
];

fn pick_default_char_for_state(st: &DWAState, rng: &mut SimpleRng) -> i16 {
    let ex = &st.transitions.exceptions;
    // Try random from base alphabet
    if !BASE_ALPHABET.is_empty() {
        let mut idx = rng.gen_usize(BASE_ALPHABET.len());
        for _ in 0..BASE_ALPHABET.len() {
            let ch = BASE_ALPHABET[idx % BASE_ALPHABET.len()];
            if !ex.contains_key(&ch) {
                return ch;
            }
            idx = idx.wrapping_add(1);
        }
    }
    // Fallback: scan integers to find a non-exception char (always exists since exceptions are finite).
    let mut probe: i16 = 0;
    loop {
        if !ex.contains_key(&probe) {
            return probe;
        }
        probe = probe.wrapping_add(1);
    }
}

impl SimpleBitset {
    pub fn zeros() -> Self {
        SimpleBitset(RangeSetBlaze::new())
    }
    pub fn all() -> Self {
        SimpleBitset(RangeSetBlaze::from_iter([0..=usize::MAX]))
    }
    pub fn from_item(item: usize) -> Self {
        SimpleBitset(RangeSetBlaze::from_iter([item]))
    }
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        SimpleBitset(rsb)
    }
    pub fn len(&self) -> usize {
        self.0.len().try_into().unwrap_or(usize::MAX)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn contains(&self, index: usize) -> bool {
        self.0.contains(index)
    }
    /// Iterate over items, truncated by `max` to prevent accidental ALL iteration.
    pub fn iter_up_to(&self, max: usize) -> impl Iterator<Item = usize> {
        (&self.0 & &RangeSetBlaze::from_iter([0..=max])).into_iter()
    }
}

impl Debug for SimpleBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self == &Self::all() {
            write!(f, "SimpleBitset(ALL)")
        } else {
            Debug::fmt(&self.0, f)
        }
    }
}

impl Display for SimpleBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self == &Self::all() {
            return write!(f, "ALL");
        }
        write!(f, "[")?;
        let mut ranges = self.0.ranges().peekable();
        while let Some(range) = ranges.next() {
            if range.start() == range.end() {
                write!(f, "{}", range.start())?;
            } else {
                write!(f, "{}..={}", range.start(), range.end())?;
            }
            if ranges.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl FromIterator<usize> for SimpleBitset {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        SimpleBitset(RangeSetBlaze::from_iter(iter))
    }
}

impl FromIterator<std::ops::RangeInclusive<usize>> for SimpleBitset {
    fn from_iter<T: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(iter: T) -> Self {
        SimpleBitset(RangeSetBlaze::from_iter(iter))
    }
}

// Borrowed bit-ops
impl<'a> BitAnd<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'a SimpleBitset) -> Self::Output {
        SimpleBitset(&self.0 & &rhs.0)
    }
}
impl<'a> BitOr<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'a SimpleBitset) -> Self::Output {
        SimpleBitset(&self.0 | &rhs.0)
    }
}

// Assign ops (borrowed RHS)
impl BitAndAssign<&SimpleBitset> for SimpleBitset {
    fn bitand_assign(&mut self, rhs: &SimpleBitset) {
        self.0 = &self.0 & &rhs.0;
    }
}
impl BitOrAssign<&SimpleBitset> for SimpleBitset {
    fn bitor_assign(&mut self, rhs: &SimpleBitset) {
        self.0 |= &rhs.0;
    }
}

// Owned fallbacks via borrowed ops
impl BitAnd<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: SimpleBitset) -> Self::Output {
        &self & &rhs
    }
}
impl BitOr<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: SimpleBitset) -> Self::Output {
        &self | &rhs
    }
}
impl<'a> BitAnd<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'a SimpleBitset) -> Self::Output {
        &self & rhs
    }
}
impl<'a> BitOr<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'a SimpleBitset) -> Self::Output {
        &self | rhs
    }
}
impl<'a> BitAnd<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: SimpleBitset) -> Self::Output {
        self & &rhs
    }
}
impl<'a> BitOr<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: SimpleBitset) -> Self::Output {
        self | &rhs
    }
}

// --- Part 2: I16Map ---

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct I16Map<T> {
    pub exceptions: BTreeMap<i16, T>,
    pub default: Option<T>,
}

impl<T> I16Map<T> {
    pub fn new() -> Self {
        Self { exceptions: BTreeMap::new(), default: None }
    }
    pub fn with_default(default_value: T) -> Self {
        Self { exceptions: BTreeMap::new(), default: Some(default_value) }
    }
    pub fn get(&self, key: i16) -> Option<&T> {
        self.exceptions.get(&key).or(self.default.as_ref())
    }
    pub fn iter_exceptions(&self) -> impl Iterator<Item = (&i16, &T)> {
        self.exceptions.iter()
    }
    pub fn get_default(&self) -> Option<&T> {
        self.default.as_ref()
    }
}

fn weight_subset(sub: &Weight, sup: &Weight) -> bool {
    (sub & sup) == sub.clone()
}

fn format_pos_code(code: i16) -> String {
    let u = code as u16;
    if let Some(c) = char::from_u32(u as u32) {
        if c.is_ascii_graphic() || c == ' ' {
            format!("'{}'", c)
        } else {
            format!("{}", u)
        }
    } else {
        format!("{}", u)
    }
}
fn format_i16_char(code: i16) -> String { if code >= 0 { format_pos_code(code) } else { format!("neg({})", code.wrapping_sub(i16::MIN)) } }
fn format_word(word: &[i16]) -> String { let parts: Vec<String> = word.iter().map(|&c| format_i16_char(c)).collect(); format!("[{}]", parts.join(", ")) }
// --- Part 3: Automata Definitions (DWA only) ---

pub type StateID = usize;
pub type Weight = SimpleBitset;

// --- Deterministic Weighted Automaton (DWA) ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DWABuildError {
    TransitionAlreadyExists { from: StateID, on: i16 },
    DefaultTransitionAlreadyExists { from: StateID },
    StateOutOfBounds { state: StateID },
}

impl Display for DWABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DWABuildError::TransitionAlreadyExists { from, on } => {
                write!(f, "Transition from state {} on code {} already exists", from, on)
            }
            DWABuildError::DefaultTransitionAlreadyExists { from } => {
                write!(f, "Default transition from state {} already exists", from)
            }
            DWABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAState {
    pub transitions: I16Map<StateID>,
    pub final_weight: Option<Weight>,
    pub trans_weight_default: Option<Weight>,
    pub trans_weights_exceptions: BTreeMap<i16, Weight>,
}

impl DWAState {
    pub fn get_weight(&self, ch: i16) -> Option<&Weight> {
        self.trans_weights_exceptions.get(&ch).or(self.trans_weight_default.as_ref())
    }

    /// Intersects all weights in this state with the given weight.
    pub fn apply_weight(&mut self, weight: &Weight) {
        if let Some(fw) = &mut self.final_weight {
            *fw &= weight;
            if fw.is_empty() {
                self.final_weight = None;
            }
        }

        if let Some(twd) = &mut self.trans_weight_default {
            *twd &= weight;
        }

        for w in self.trans_weights_exceptions.values_mut() {
            *w &= weight;
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DWAStates(pub Vec<DWAState>);

impl Index<usize> for DWAStates {
    type Output = DWAState;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for DWAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl Deref for DWAStates {
    type Target = [DWAState];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DWAStates {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len();
        self.0.push(DWAState::default());
        id
    }

    /// Adds a pre-existing DWAState to the collection and returns its new ID.
    pub fn add_existing_state(&mut self, state: DWAState) -> StateID {
        let id = self.0.len();
        self.0.push(state);
        id
    }

    pub fn copy_state(&mut self, state_id: StateID) -> StateID {
        assert!(state_id < self.len(), "state_id out of bounds");
        let state = self[state_id].clone();
        self.add_existing_state(state)
    }

    pub fn remove_state(&mut self, state_id: StateID) -> DWAState {
        self.0.remove(state_id)
    }

    pub fn apply_weight(&mut self, state_id: StateID, weight: &Weight) {
        assert!(state_id < self.len(), "state_id out of bounds");
        self[state_id].apply_weight(weight);
    }

    pub fn union_assign_state(&mut self, s_from_id: StateID, s_into_id: StateID) {
        assert!(s_from_id < self.len(), "s_from_id out of bounds");
        assert!(s_into_id < self.len(), "s_into_id out of bounds");
        if s_from_id == s_into_id {
            return;
        }

        let s_from = self.0[s_from_id].clone();
        let s_into_orig = self.0[s_into_id].clone();

        // Union final weights
        let fw_from = s_from.final_weight.clone().unwrap_or_else(Weight::zeros);
        let fw_into = s_into_orig.final_weight.clone().unwrap_or_else(Weight::zeros);
        let new_final_weight = fw_from | fw_into;

        // --- Transitions ---
        let mut new_transitions = I16Map::new();
        let mut new_trans_weights_exceptions = BTreeMap::new();
        let new_trans_weight_default;

        let critical_points: BTreeSet<i16> = s_from.transitions.exceptions.keys()
            .chain(s_into_orig.transitions.exceptions.keys())
            .cloned()
            .collect();

        let get_target = |s: &DWAState, ch: i16| -> Option<StateID> {
            s.transitions.exceptions.get(&ch).copied().or(s.transitions.default)
        };

        let get_weight = |s: &DWAState, ch: i16| -> Weight {
            s.trans_weights_exceptions.get(&ch)
                .or(s.trans_weight_default.as_ref())
                .cloned().unwrap_or_else(Weight::zeros)
        };

        // Default transition
        let def_tgt_from = s_from.transitions.default;
        let def_tgt_into = s_into_orig.transitions.default;
        let new_def_tgt_id = match (def_tgt_from, def_tgt_into) {
            (Some(t1), Some(t2)) => if t1 == t2 { Some(t1) } else { Some(self.union_state(t1, t2)) },
            (Some(t), None) | (None, Some(t)) => Some(t),
            (None, None) => None,
        };
        new_transitions.default = new_def_tgt_id;
        new_trans_weight_default = if new_def_tgt_id.is_some() {
            let w_from = s_from.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::zeros);
            let w_into = s_into_orig.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::zeros);
            Some(w_from | w_into)
        } else {
            None
        };

        // Exception transitions
        for &ch in &critical_points {
            let tgt_from = get_target(&s_from, ch);
            let tgt_into = get_target(&s_into_orig, ch);

            let new_exc_tgt_id = match (tgt_from, tgt_into) {
                (Some(t1), Some(t2)) => if t1 == t2 { Some(t1) } else { Some(self.union_state(t1, t2)) },
                (Some(t), None) | (None, Some(t)) => Some(t),
                (None, None) => None,
            };

            if new_exc_tgt_id != new_def_tgt_id {
                if let Some(tgt_id) = new_exc_tgt_id {
                    new_transitions.exceptions.insert(ch, tgt_id);
                    new_trans_weights_exceptions.insert(ch, get_weight(&s_from, ch) | get_weight(&s_into_orig, ch));
                }
            }
        }

        // Commit changes
        let s_into = &mut self.0[s_into_id];
        s_into.final_weight = if new_final_weight.is_empty() { None } else { Some(new_final_weight) };
        s_into.transitions = new_transitions;
        s_into.trans_weight_default = new_trans_weight_default;
        s_into.trans_weights_exceptions = new_trans_weights_exceptions;
    }

    pub fn union_state(&mut self, s1_id: StateID, s2_id: StateID) -> StateID {
        assert!(s1_id < self.len(), "s1_id out of bounds");
        assert!(s2_id < self.len(), "s2_id out of bounds");

        // Create a new state by copying s1
        let new_id = self.copy_state(s1_id);

        // Merge s2 into the new state
        self.union_assign_state(s2_id, new_id);

        new_id
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWABody {
    pub start_state: StateID,
}

#[derive(Clone, Debug, Default)]
pub struct DWA {
    pub states: DWAStates,
    pub body: DWABody,
}

impl DWA {
    pub fn new() -> Self {
        let mut states = DWAStates::default();
        let start = states.add_state();
        DWA { states, body: DWABody { start_state: start } }
    }

    pub fn add_state(&mut self) -> StateID {
        self.states.add_state()
    }

    pub fn set_state_weight(&mut self, state: StateID, _weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state });
        }
        // weight field removed; this is now a no-op kept for API compatibility.
        Ok(())
    }

    pub fn set_final_weight(&mut self, state: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state });
        }
        self.states[state].final_weight = Some(weight);
        Ok(())
    }

    pub fn add_transition(
        &mut self,
        from: StateID,
        on: i16,
        to: StateID,
        weight: Weight,
    ) -> Result<(), DWABuildError> {
        if from >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state: to });
        }
        let from_state = &mut self.states[from];
        if from_state.transitions.exceptions.contains_key(&on) {
            return Err(DWABuildError::TransitionAlreadyExists { from, on });
        }
        from_state.transitions.exceptions.insert(on, to);
        from_state.trans_weights_exceptions.insert(on, weight);
        Ok(())
    }

    pub fn set_default_transition(
        &mut self,
        from: StateID,
        to: StateID,
        weight: Weight,
    ) -> Result<(), DWABuildError> {
        if from >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state: to });
        }
        let from_state = &mut self.states[from];
        if from_state.transitions.default.is_some() {
            return Err(DWABuildError::DefaultTransitionAlreadyExists { from });
        }
        from_state.transitions.default = Some(to);
        from_state.trans_weight_default = Some(weight);
        Ok(())
    }

    pub fn simplify(&mut self) {
        Self::simplify_components(&mut self.states, &mut self.body)
    }

    pub fn simplify_components(states: &mut DWAStates, body: &mut DWABody) {
        let now = Instant::now();
        let initial_len = states.len();
        if states.0.is_empty() {
            return;
        }

        Self::normalize_edges_inplace(states);
        Self::prune_unreachable(states, body);

        let mut changed_any = true;
        let mut passes = 0usize;
        while changed_any && passes < 10 {
            passes += 1;
            changed_any = false;
            if Self::normalize_edges_inplace(states) {
                changed_any = true;
            }
            if Self::propagate_and_constrain_weights(states, body) {
                changed_any = true;
            }
            if Self::minimize_partition_refinement(states, body) {
                changed_any = true;
            }
            if Self::normalize_edges_inplace(states) {
                changed_any = true;
            }
            if Self::prune_unreachable(states, body) {
                changed_any = true;
            }
        }
        crate::debug!(3, "DWA::simplify_components ({} states -> {} states) took: {:?}", initial_len, states.len(), now.elapsed());
    }

    pub fn propagate_and_constrain_weights(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }

        let mut reachable_weights = vec![Weight::zeros(); n];
        let mut worklist = VecDeque::new();

        if body.start_state >= n {
            return false; // No start state, nothing to do.
        }
        reachable_weights[body.start_state] = Weight::all();
        worklist.push_back(body.start_state);

        while let Some(u) = worklist.pop_front() {
            let u_rw = reachable_weights[u].clone();
            if u_rw.is_empty() {
                continue;
            }
            let u_state = &states[u];

            // Default transition
            if let Some(v) = u_state.transitions.default {
                if v < n {
                    let edge_w = u_state.trans_weight_default.as_ref().unwrap();
                    let new_v_rw = &u_rw & edge_w;

                    let old_v_rw = &reachable_weights[v];
                    if (&new_v_rw & old_v_rw) != new_v_rw {
                        // if new_v_rw is not a subset of old_v_rw
                        reachable_weights[v] |= &new_v_rw;
                        worklist.push_back(v);
                    }
                }
            }

            // Exception transitions
            for (&ch, &v) in &u_state.transitions.exceptions {
                if v < n {
                    let edge_w = u_state.trans_weights_exceptions.get(&ch).unwrap();
                    let new_v_rw = &u_rw & edge_w;

                    let old_v_rw = &reachable_weights[v];
                    if (&new_v_rw & old_v_rw) != new_v_rw {
                        // if new_v_rw is not a subset of old_v_rw
                        reachable_weights[v] |= &new_v_rw;
                        worklist.push_back(v);
                    }
                }
            }
        }

        // Now apply constraints
        let mut changed = false;
        for i in 0..n {
            let old_fw = states[i].final_weight.clone();
            if let Some(fw) = states[i].final_weight.as_mut() {
                *fw &= &reachable_weights[i];
                if fw.is_empty() {
                    states[i].final_weight = None;
                }
            }
            if states[i].final_weight != old_fw {
                changed = true;
            }
        }
        changed
    }

    pub fn normalize_edges_inplace(states: &mut DWAStates) -> bool {
        let mut changed = false;
        for st in &mut states.0 {
            let before = st.transitions.exceptions.len();
            if let Some(def) = st.transitions.default {
                st.transitions.exceptions.retain(|_, &mut tgt| tgt != def);
            }
            changed |= st.transitions.exceptions.len() != before;

            let before_w = st.trans_weights_exceptions.len();
            st.trans_weights_exceptions.retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            changed |= st.trans_weights_exceptions.len() != before_w;
        }
        changed
    }

    pub fn minimize_partition_refinement(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let n = states.0.len();
        if n <= 1 {
            return false;
        }
        let sink_pid: usize = n;

        // Initial partition by outputs (weight, final_weight).
        let mut part: Vec<usize> = vec![0; n];
        let mut canon0: HashMap<Option<Weight>, usize> = HashMap::new();
        for i in 0..n {
            let key = states[i].final_weight.clone();
            let next_id = canon0.len();
            part[i] = *canon0.entry(key).or_insert(next_id);
        }

        // Refine until stable
        let mut changed = true;
        let mut rounds = 0usize;
        while changed && rounds < 30 {
            rounds += 1;
            changed = false;
            let mut next_part: Vec<usize> = vec![0; n];
            let mut sig2pid: HashMap<(Option<Weight>, usize, Vec<(i16, usize)>), usize> = HashMap::new();

            for i in 0..n {
                let st = &states[i];
                let def_cls = st.transitions.default.map(|d| part[d]).unwrap_or(sink_pid);
                let mut ex: Vec<(i16, usize)> = Vec::with_capacity(st.transitions.exceptions.len());
                for (ch, tgt) in &st.transitions.exceptions {
                    let cls = part[*tgt];
                    if cls != def_cls {
                        ex.push((*ch, cls));
                    }
                }
                let sig = (st.final_weight.clone(), def_cls, ex);
                let next_pid = sig2pid.len();
                next_part[i] = *sig2pid.entry(sig).or_insert(next_pid);
            }
            if next_part != part {
                part = next_part;
                changed = true;
            }
        }

        // Early exit if nothing to merge
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in part.iter().enumerate() {
            groups.entry(*p).or_default().push(i);
        }
        if groups.len() == n {
            return false;
        }

        // Build representatives
        let mut pid_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_states: Vec<DWAState> = Vec::with_capacity(groups.len());
        for (pid, members) in &groups {
            let rep = members[0];
            let rep_state = &states[rep];
            let def_cls = rep_state.transitions.default.map(|d| part[d]).unwrap_or(sink_pid);

            let mut st = DWAState::default();
            st.final_weight = rep_state.final_weight.clone();
            st.transitions.default = if def_cls == sink_pid { None } else { Some(0) };
            for (ch, tgt) in &rep_state.transitions.exceptions {
                let cls = part[*tgt];
                if cls != def_cls {
                    st.transitions.exceptions.insert(*ch, 0);
                }
            }

            // Aggregate per-edge weights across members (join).
            if st.transitions.default.is_some() {
                let mut agg_def: Option<Weight> = None;
                for &old in members {
                    if let Some(w) = states[old].trans_weight_default.as_ref() {
                        if let Some(ref mut a) = agg_def {
                            *a |= w;
                        } else {
                            agg_def = Some(w.clone());
                        }
                    }
                }
                st.trans_weight_default = agg_def;
            }
            let ex_keys: Vec<i16> = st.transitions.exceptions.keys().cloned().collect();
            for ch in ex_keys {
                let mut agg: Option<Weight> = None;
                for &old in members {
                    if let Some(w) = states[old].trans_weights_exceptions.get(&ch) {
                        if let Some(ref mut a) = agg {
                            *a |= w;
                        } else {
                            agg = Some(w.clone());
                        }
                    }
                }
                if let Some(w) = agg {
                    st.trans_weights_exceptions.insert(ch, w);
                }
            }

            let new_id = new_states.len();
            pid_to_new.insert(*pid, new_id);
            new_states.push(st);
        }

        // Fix transition targets
        for (pid, members) in &groups {
            let new_id = *pid_to_new.get(pid).unwrap();
            let rep = members[0];
            let rep_state = &states[rep];

            let def_cls = rep_state.transitions.default.map(|d| part[d]).unwrap_or(sink_pid);
            if let Some(ref mut d) = new_states[new_id].transitions.default {
                *d = *pid_to_new.get(&def_cls).unwrap();
            }
            let ex_old = new_states[new_id].transitions.exceptions.clone();
            new_states[new_id].transitions.exceptions.clear();
            for (ch, _) in ex_old {
                let cls = part[*rep_state.transitions.exceptions.get(&ch).unwrap()];
                new_states[new_id].transitions.exceptions.insert(ch, *pid_to_new.get(&cls).unwrap());
            }
        }

        states.0 = new_states;
        Self::normalize_edges_inplace(states);
        let start_pid = part[body.start_state];
        body.start_state = *pid_to_new.get(&start_pid).unwrap();

        true
    }

    pub fn prune_unreachable(states: &mut DWAStates, body: &mut DWABody) -> bool {
        if states.0.is_empty() {
            return false;
        }
        let n = states.0.len();

        // 1. Backward reachability from final states to find "live" states.
        let mut live = vec![false; n];
        let mut q_live: VecDeque<usize> = VecDeque::new();
        let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
        for i in 0..n {
            if states[i].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                live[i] = true;
                q_live.push_back(i);
            }
            if let Some(d) = states[i].transitions.default { if d < n { rev_adj[d].push(i); } }
            for &v in states[i].transitions.exceptions.values() { if v < n { rev_adj[v].push(i); } }
        }
        while let Some(u) = q_live.pop_front() {
            for &v in &rev_adj[u] {
                if !live[v] { live[v] = true; q_live.push_back(v); }
            }
        }

        // 2. Remove transitions to non-live states.
        let mut changed = false;
        for i in 0..n {
            let st = &mut states[i];
            if let Some(d) = st.transitions.default {
                if !live[d] {
                    st.transitions.default = None;
                    st.trans_weight_default = None;
                    changed = true;
                }
            }
            let before = st.transitions.exceptions.len();
            st.transitions.exceptions.retain(|_, tgt| live[*tgt]);
            if st.transitions.exceptions.len() != before {
                changed = true;
                st.trans_weights_exceptions.retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            }
        }

        // 3. Forward reachability from start_state to find actually reachable states.
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        if body.start_state < n {
            visited[body.start_state] = true;
            q.push_back(body.start_state);
        }
        while let Some(u) = q.pop_front() {
            if let Some(d) = states[u].transitions.default { if d < n && !visited[d] { visited[d] = true; q.push_back(d); } }
            for &v in states[u].transitions.exceptions.values() { if v < n && !visited[v] { visited[v] = true; q.push_back(v); } }
        }

        if visited.iter().all(|&b| b) && !changed {
            return false;
        }

        // 4. Remap kept states.
        let mut map = vec![usize::MAX; n];
        let mut next_id = 0usize;
        for i in 0..n { if visited[i] { map[i] = next_id; next_id += 1; } }

        if next_id == n && !changed { return false; }

        let mut new_states: Vec<DWAState> = Vec::with_capacity(next_id);
        for old in 0..n {
            if !visited[old] { continue; }
            let mut st = states[old].clone();
            if let Some(d) = st.transitions.default { st.transitions.default = Some(map[d]); }
            let ex = st.transitions.exceptions.clone();
            st.transitions.exceptions.clear();
            for (ch, tgt) in ex { st.transitions.exceptions.insert(ch, map[tgt]); }
            new_states.push(st);
        }
        states.0 = new_states;
        if next_id > 0 {
            body.start_state = map[body.start_state];
        } else if n > 0 {
            states.0.clear();
            body.start_state = states.add_state();
        }
        true
    }

    /// Union of two DWAs via product construction.
    /// The states of the new DWA correspond to pairs of states from the input DWAs.
    /// A state is final if either of the original states is final.
    /// A transition exists if it exists in at least one of the original DWAs.
    /// If a transition exists in one but not the other, the other is treated as going to a non-final sink state.
    pub fn union(&self, other: &DWA) -> DWA {
        let mut new_dwa = DWA::default();
        let mut pair_to_new_id: BTreeMap<(StateID, StateID), StateID> = BTreeMap::new();
        let mut worklist: VecDeque<(StateID, StateID)> = VecDeque::new();

        let sink0 = self.states.len();
        let sink1 = other.states.len();

        let mut get_or_create = |
            pair: (StateID, StateID),
            new_dwa: &mut DWA,
            pair_to_new_id: &mut BTreeMap<(StateID, StateID), StateID>,
            worklist: &mut VecDeque<(StateID, StateID)>
        | -> StateID {
            if let Some(&id) = pair_to_new_id.get(&pair) {
                return id;
            }
            let new_id = new_dwa.states.0.len();
            new_dwa.states.0.push(DWAState::default());
            pair_to_new_id.insert(pair, new_id);
            worklist.push_back(pair);
            new_id
        };

        if self.states.is_empty() && other.states.is_empty() {
            return new_dwa;
        }

        let start_pair = (self.body.start_state, other.body.start_state);
        new_dwa.body.start_state = get_or_create(start_pair, &mut new_dwa, &mut pair_to_new_id, &mut worklist);

        while let Some((id0, id1)) = worklist.pop_front() {
            let new_id = *pair_to_new_id.get(&(id0, id1)).unwrap();
            let s0 = if id0 == sink0 { None } else { Some(&self.states[id0]) };
            let s1 = if id1 == sink1 { None } else { Some(&other.states[id1]) };

            let new_state = &mut new_dwa.states[new_id];

            let fw0 = s0.and_then(|s| s.final_weight.as_ref()).cloned().unwrap_or_else(Weight::zeros);
            let fw1 = s1.and_then(|s| s.final_weight.as_ref()).cloned().unwrap_or_else(Weight::zeros);
            let final_w = fw0 | fw1;
            if !final_w.is_empty() {
                new_state.final_weight = Some(final_w);
            }

            let mut critical_points = BTreeSet::new();
            if let Some(s) = s0 { critical_points.extend(s.transitions.exceptions.keys()); }
            if let Some(s) = s1 { critical_points.extend(s.transitions.exceptions.keys()); }

            let get_target = |s: Option<&DWAState>, sink: StateID, ch: i16| -> StateID {
                s.and_then(|s| s.transitions.get(ch).copied()).unwrap_or(sink)
            };

            let s0_has_default = s0.map_or(false, |s| s.transitions.default.is_some());
            let s1_has_default = s1.map_or(false, |s| s.transitions.default.is_some());

            // Default transition
            let def_tgt0 = s0.and_then(|s| s.transitions.default).unwrap_or(sink0);
            let def_tgt1 = s1.and_then(|s| s.transitions.default).unwrap_or(sink1);
            let def_pair = (def_tgt0, def_tgt1);
            if s0_has_default || s1_has_default {
                let new_def_tgt = get_or_create(def_pair, &mut new_dwa, &mut pair_to_new_id, &mut worklist);
                new_dwa.states[new_id].transitions.default = Some(new_def_tgt);

                let tw_def0 = s0.and_then(|s| s.trans_weight_default.as_ref()).cloned().unwrap_or_else(Weight::zeros);
                let tw_def1 = s1.and_then(|s| s.trans_weight_default.as_ref()).cloned().unwrap_or_else(Weight::zeros);
                new_dwa.states[new_id].trans_weight_default = Some(tw_def0 | tw_def1);
            }
            // Exception transitions
            for &ch in &critical_points {
                let tgt0 = get_target(s0, sink0, ch);
                let tgt1 = get_target(s1, sink1, ch);
                let exc_pair = (tgt0, tgt1);

                if exc_pair != def_pair {
                    let new_exc_tgt = get_or_create(exc_pair, &mut new_dwa, &mut pair_to_new_id, &mut worklist);
                    new_dwa.states[new_id].transitions.exceptions.insert(ch, new_exc_tgt);

                    let tw_exc0 = s0.and_then(|s| s.trans_weights_exceptions.get(&ch)).cloned().unwrap_or_else(Weight::zeros);
                    let tw_exc1 = s1.and_then(|s| s.trans_weights_exceptions.get(&ch)).cloned().unwrap_or_else(Weight::zeros);
                    new_dwa.states[new_id].trans_weights_exceptions.insert(ch, tw_exc0 | tw_exc1);
                }
            }
        }

        if STOCHASTIC_VALIDATION {
            // Validate C against A and B via stochastic sampling.
            DWA::stochastic_validate_concatenate(self, other, &new_dwa);
        }

        new_dwa
    }

    // --- Evaluation and sampling helpers ---
    /// Evaluate a word's weight in this DWA by intersecting per-edge weights and the final state's weight.
    /// Returns zeros if the word is not accepted or any transition is missing.
    pub fn eval_word_weight(&self, word: &[i16]) -> Weight {
        if self.states.0.is_empty() {
            return Weight::zeros();
        }
        let mut s = self.body.start_state;
        let mut acc = Weight::all();
        for &ch in word {
            if s >= self.states.len() {
                return Weight::zeros();
            }
            let st = &self.states[s];
            if let Some(&t) = st.transitions.get(ch) {
                if let Some(w) = st.get_weight(ch) {
                    acc &= w;
                } else {
                    return Weight::zeros();
                }
                if acc.is_empty() {
                    return Weight::zeros();
                }
                s = t;
            } else {
                return Weight::zeros();
            }
        }
        if s >= self.states.len() {
            return Weight::zeros();
        }
        match &self.states[s].final_weight {
            Some(fw) => {
                let res = &acc & fw;
                if res.is_empty() { Weight::zeros() } else { res }
            }
            None => Weight::zeros(),
        }
    }

    /// Evaluate a word; None if rejected (weight empty).
    pub fn eval_word(&self, word: &[i16]) -> Option<Weight> {
        let w = self.eval_word_weight(word);
        if w.is_empty() { None } else { Some(w) }
    }

    /// Sample an accepted path (word and weight) using a time-based seed.
    /// Returns None if no accepted path was found within the attempt budget.
    pub fn sample_accepted_path(&self, max_steps: usize) -> Option<(Vec<i16>, Weight)> {
        let mut rng = SimpleRng::from_time();
        self.sample_accepted_path_with_rng(&mut rng, max_steps)
    }

    /// Sample an accepted path (word and weight) with a fixed seed (deterministic).
    pub fn sample_accepted_path_with_seed(&self, seed: u64, max_steps: usize) -> Option<(Vec<i16>, Weight)> {
        let mut rng = SimpleRng::new(seed);
        self.sample_accepted_path_with_rng(&mut rng, max_steps)
    }

    /// Core sampler with a provided RNG. Tries multiple attempts to find an accepted word.
    pub fn sample_accepted_path_with_rng(&self, rng: &mut SimpleRng, max_steps: usize) -> Option<(Vec<i16>, Weight)> {
        if self.states.0.is_empty() {
            return None;
        }
        for _attempt in 0..SAMPLING_TRIES {
            let mut word: Vec<i16> = Vec::new();
            let mut s = self.body.start_state;
            let mut acc = Weight::all();

            for step in 0..max_steps {
                // Early stop with some probability if we can accept here.
                if let Some(fw) = &self.states[s].final_weight {
                    if rng.gen_bool_ratio(1, 3) || step == max_steps - 1 {
                        let w = &acc & fw;
                        if !w.is_empty() {
                            return Some((word, w));
                        }
                    }
                }

                // Choose next character: one of the exception keys or a default-character if default exists.
                let st = &self.states[s];
                let mut choices: Vec<i16> = st.transitions.exceptions.keys().copied().collect();
                let has_default = st.transitions.default.is_some();
                let total = choices.len() + if has_default { 1 } else { 0 };
                if total == 0 {
                    // Dead-end; try to finalize or abort attempt.
                    if let Some(fw) = &st.final_weight {
                        let w = &acc & fw;
                        if !w.is_empty() {
                            return Some((word, w));
                        }
                    }
                    break; // new attempt
                }
                let pick = rng.gen_usize(total);
                let ch = if has_default && pick == choices.len() {
                    pick_default_char_for_state(st, rng)
                } else {
                    choices[pick]
                };

                let next = st.transitions.get(ch).copied();
                if next.is_none() {
                    break;
                }
                let edge_w = st.get_weight(ch).cloned().unwrap_or_else(Weight::zeros);
                if edge_w.is_empty() {
                    break;
                }
                let new_acc = &acc & &edge_w;
                if new_acc.is_empty() {
                    break;
                }
                acc = new_acc;
                s = next.unwrap();
                word.push(ch);
            }

            // Finalize at end of attempt if possible:
            if s < self.states.len() {
                if let Some(fw) = &self.states[s].final_weight {
                    let w = &acc & fw;
                    if !w.is_empty() {
                        return Some((word, w));
                    }
                }
            }
        }
        None
    }

    fn expected_union_weight(a: &DWA, b: &DWA, word: &[i16]) -> Weight {
        let wa = a.eval_word_weight(word);
        let wb = b.eval_word_weight(word);
        &wa | &wb
    }

    /// Expected concatenation weight:
    /// union over all split points i of (A(word[..i]) ∧ B(word[i..])).
    fn expected_concat_weight(a: &DWA, b: &DWA, word: &[i16]) -> Weight {
        let mut acc = Weight::zeros();
        for i in 0..=word.len() {
            let wa = a.eval_word_weight(&word[..i]);
            if wa.is_empty() {
                continue;
            }
            let wb = b.eval_word_weight(&word[i..]);
            if wb.is_empty() {
                continue;
            }
            let both = &wa & &wb;
            if !both.is_empty() {
                acc |= &both;
            }
        }
        acc
    }

    fn stochastic_validate_union(a: &DWA, b: &DWA, u: &DWA) {
        let mut rng = SimpleRng::from_time();
        for _ in 0..VALIDATION_SAMPLES {
            // Sample a path from A -> must be in U, and U == A ∪ B for that word.
            if let Some((w, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let wu = u.eval_word_weight(&w);
                assert!(!wu.is_empty(), "Union rejected a word accepted by A.\nword: {}\nA(w): {}\nU(w): {}\n", format_word(&w), wa, wu);
                assert!(weight_subset(&wa, &wu), "Union weight missing subset from A.\nword: {}\nA(w): {}\nU(w): {}\n", format_word(&w), wa, wu);
                let expected = DWA::expected_union_weight(a, b, &w);
                assert_eq!(wu, expected, "Union weight mismatch vs expected A∪B.\nword: {}\nA(w): {}\nB(w): {}\nU(w): {}\nExpected: {}\n", format_word(&w), wa, b.eval_word_weight(&w), wu, expected);
            }

            // Sample a path from B -> must be in U, and U == A ∪ B for that word.
            if let Some((w, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let wu = u.eval_word_weight(&w);
                assert!(!wu.is_empty(), "Union rejected a word accepted by B.\nword: {}\nB(w): {}\nU(w): {}\n", format_word(&w), wb, wu);
                assert!(weight_subset(&wb, &wu), "Union weight missing subset from B.\nword: {}\nB(w): {}\nU(w): {}\n", format_word(&w), wb, wu);
                let expected = DWA::expected_union_weight(a, b, &w);
                assert_eq!(wu, expected, "Union weight mismatch vs expected A∪B.\nword: {}\nA(w): {}\nB(w): {}\nU(w): {}\nExpected: {}\n", format_word(&w), a.eval_word_weight(&w), wb, wu, expected);
            }

            // Sample a path from U -> ensure it's in A ∪ B (equality check).
            if let Some((w, wu)) = u.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let expected = DWA::expected_union_weight(a, b, &w);
                assert_eq!(wu, expected, "U accepted a word with weight not equal to A∪B.\nword: {}\nA(w): {}\nB(w): {}\nU(w): {}\nExpected: {}\n", format_word(&w), a.eval_word_weight(&w), b.eval_word_weight(&w), wu, expected);
            }
        }
    }

    fn stochastic_validate_concatenate(a: &DWA, b: &DWA, c: &DWA) {
        let mut rng = SimpleRng::from_time();
        for _ in 0..VALIDATION_SAMPLES {
            // Sample accepted paths in A and B; the concatenation of the words should be in C and contain WA ∧ WB.
            if let Some((wa_word, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                if let Some((wb_word, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                    let mut w = wa_word.clone();
                    w.extend_from_slice(&wb_word);
                    let wc = c.eval_word_weight(&w);
                    let expected_simple = &wa & &wb;
                    assert!(!expected_simple.is_empty(), "Expected non-empty weight for concatenated accepted paths, but got empty.\nA(word): {}\nB(word): {}\n", wa, wb);
                    assert!(weight_subset(&expected_simple, &wc), "Concatenation missing expected subset.\nword: {}\nA(wA): {}\nB(wB): {}\nC(wA∘wB): {}\nExpected subset: {}\n", format_word(&w), wa, wb, wc, expected_simple);
                    // Also verify full expected across all splits equals C's result
                    let expected_all = DWA::expected_concat_weight(a, b, &w);
                    assert_eq!(wc, expected_all, "C(word) != expected union-over-splits(A(prefix) ∧ B(suffix)).\nword: {}\nC(word): {}\nExpected: {}\n", format_word(&w), wc, expected_all);
                }
            }

            // Sample accepted paths from C -> must equal union-over-splits(A(prefix) ∧ B(suffix)).
            if let Some((w, wc)) = c.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let expected = DWA::expected_concat_weight(a, b, &w);
                assert_eq!(wc, expected, "C(word) != expected union-over-splits(A(prefix) ∧ B(suffix)).\nword: {}\nC(word): {}\nExpected: {}\n", format_word(&w), wc, expected);
            }
        }
    }

    /// Concatenation-like operation on two DWAs.
    /// "Join all ends to all starts": whenever the left component reaches a final state,
    /// we also (lazily) activate the right automaton's start state. If the left start is final,
    /// the composition starts with the right start already active.
    ///
    /// This is implemented via a product-like construction where states in the new DWA
    /// are compositions of a state from `self` and a set of states from `other`.
    ///
    /// Returns the new DWA.
    pub fn concatenate(
        &self,
        other: &DWA,
    ) -> DWA {
        let mut new_dwa = DWA::new();
        new_dwa.states.0.clear(); // start with no states

        let mut composition_to_new_id: BTreeMap<(StateID, BTreeSet<StateID>), StateID> = BTreeMap::new();
        let mut worklist: VecDeque<(StateID, BTreeSet<StateID>)> = VecDeque::new();

        let sink0 = self.states.len();
        let sink1 = other.states.len();

        let mut get_or_create = |comp: (StateID, BTreeSet<StateID>),
                                 new_dwa: &mut DWA,
                                 comp_to_new_id: &mut BTreeMap<(StateID, BTreeSet<StateID>), StateID>,
                                 worklist: &mut VecDeque<(StateID, BTreeSet<StateID>)>|
         -> StateID {
            if let Some(&id) = comp_to_new_id.get(&comp) {
                return id;
            }
            let new_id = new_dwa.states.0.len();
            new_dwa.states.0.push(DWAState::default());
            comp_to_new_id.insert(comp.clone(), new_id);
            worklist.push_back(comp.clone());

            new_id
        };

        if self.states.is_empty() {
            return DWA::new();
        }

        let start0 = self.body.start_state;
        let mut start1_set: BTreeSet<StateID> = BTreeSet::new();
        if self.states[start0].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
            start1_set.insert(other.body.start_state);
        }
        let start_comp = (start0, start1_set);
        new_dwa.body.start_state =
            get_or_create(start_comp, &mut new_dwa, &mut composition_to_new_id, &mut worklist);

        let right_start = other.body.start_state;
        let right_accepts_epsilon =
            other.states[right_start].final_weight.as_ref().map_or(false, |w| !w.is_empty());

        let is_final_left = |i: StateID| -> bool {
            i != sink0 && self.states[i].final_weight.as_ref().map_or(false, |w| !w.is_empty())
        };

        while let Some((id0, ids1)) = worklist.pop_front() {
            let new_id = *composition_to_new_id.get(&(id0, ids1.clone())).unwrap();
            let s0 = if id0 == sink0 { None } else { Some(&self.states[id0]) };

            let new_state = &mut new_dwa.states[new_id];

            let agg_final_weight = if right_accepts_epsilon {
                s0.and_then(|s| s.final_weight.as_ref()).cloned().unwrap_or_else(Weight::zeros)
            } else {
                let mut fw = Weight::zeros();
                for &id1 in &ids1 {
                    if id1 != sink1 {
                        if let Some(w) = &other.states[id1].final_weight {
                            fw |= w;
                        }
                    }
                }
                fw
            };

            if !agg_final_weight.is_empty() {
                new_state.final_weight = Some(agg_final_weight);
            }

            // Collect critical points
            let mut critical_points = BTreeSet::new();
            if let Some(s) = s0 {
                critical_points.extend(s.transitions.exceptions.keys());
            }
            for &id1 in &ids1 {
                if id1 != sink1 {
                    critical_points.extend(other.states[id1].transitions.exceptions.keys());
                }
            }

            let get_target = |s: Option<&DWAState>, sink: StateID, ch: i16| -> StateID {
                s.and_then(|s| s.transitions.get(ch).copied()).unwrap_or(sink)
            };

            let s0_has_default = s0.map_or(false, |s| s.transitions.default.is_some());
            let any_s1_has_default =
                ids1.iter().any(|&id1| id1 != sink1 && other.states[id1].transitions.default.is_some());

            // Default transition
            let def_tgt0 = s0.and_then(|s| s.transitions.default).unwrap_or(sink0);
            let mut def_tgts1: BTreeSet<StateID> =
                ids1.iter().filter(|&&id1| id1 != sink1).map(|&id1| other.states[id1].transitions.default.unwrap_or(sink1)).collect();
            def_tgts1.remove(&sink1);
            if is_final_left(def_tgt0) {
                def_tgts1.insert(right_start);
            }

            let def_comp = (def_tgt0, def_tgts1);
            if s0_has_default || any_s1_has_default {
                let new_def_tgt =
                    get_or_create(def_comp.clone(), &mut new_dwa, &mut composition_to_new_id, &mut worklist);
                new_dwa.states[new_id].transitions.default = Some(new_def_tgt);

                let mut tw_def = s0.and_then(|s| s.trans_weight_default.as_ref()).cloned().unwrap_or_else(Weight::zeros);
                for &id1 in &ids1 {
                    if id1 != sink1 {
                        tw_def |= &other.states[id1].trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::zeros);
                    }
                }
                new_dwa.states[new_id].trans_weight_default = Some(tw_def);
            }

            // Exception transitions
            for &ch in &critical_points {
                let tgt0 = get_target(s0, sink0, ch);
                let mut tgts1: BTreeSet<StateID> =
                    ids1.iter().filter(|&&id1| id1 != sink1).map(|&id1| get_target(Some(&other.states[id1]), sink1, ch)).collect();
                tgts1.remove(&sink1);
                if is_final_left(tgt0) {
                    tgts1.insert(right_start);
                }

                let exc_comp = (tgt0, tgts1);

                if exc_comp != def_comp {
                    let new_exc_tgt =
                        get_or_create(exc_comp, &mut new_dwa, &mut composition_to_new_id, &mut worklist);
                    new_dwa.states[new_id].transitions.exceptions.insert(ch, new_exc_tgt);

                    let mut tw_exc = s0.and_then(|s| s.trans_weights_exceptions.get(&ch)).cloned().unwrap_or_else(Weight::zeros);
                    for &id1 in &ids1 {
                        if id1 != sink1 {
                            tw_exc |= &other.states[id1].trans_weights_exceptions.get(&ch).cloned().unwrap_or_else(Weight::zeros);
                        }
                    }
                    new_dwa.states[new_id].trans_weights_exceptions.insert(ch, tw_exc);
                }
            }
        }

        if STOCHASTIC_VALIDATION {
            // Validate U against A and B via stochastic sampling.
            DWA::stochastic_validate_union(self, other, &new_dwa);
        }

        new_dwa
    }

    /// Apply a weight gate to the DWA by introducing a new start state that forwards to the
    /// original start with all outgoing edge weights intersected by `weight`. The new state's
    /// own weight and final_weight are intersected as well.
    /// Returns the ID of the newly created start state.
    pub fn apply_weight(&mut self, weight: &Weight) -> StateID {
        Self::apply_weight_components(&mut self.states, &mut self.body, weight)
    }

    /// Component-level variant: mutate only the shared `states` arena and this `body`.
    /// Builds a fresh start that mirrors the original start's transitions but with
    /// per-edge weights intersected by `weight`. Other states remain unchanged.
    pub fn apply_weight_components(
        states: &mut DWAStates,
        body: &mut DWABody,
        weight: &Weight,
    ) -> StateID {
        let old_start = body.start_state;
        let new_id = states.add_state();

        // Snapshot the old start state's data, then apply the weight gate.
        let mut new_state = states[old_start].clone();
        new_state.apply_weight(weight);

        states[new_id] = new_state;
        body.start_state = new_id;
        new_id
    }
}

// --- Display Implementations for Debugging ---
impl Display for DWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(w) = &state.final_weight {
                writeln!(f, "    final_weight: {}", w)?;
            }
            if let Some(to) = &state.transitions.default {
                if let Some(w) = &state.trans_weight_default {
                    writeln!(f, "    * -> {} (trans_weight: {})", to, w)?;
                } else {
                    writeln!(f, "    * -> {}", to)?;
                }
            }
            for (on, to) in &state.transitions.exceptions {
                // Represent positive codes as char if ASCII; negatives as '-char' or '-num'
                let char_repr = if *on >= 0 {
                    format_pos_code(*on)
                } else {
                    let decoded_id = on.wrapping_sub(i16::MIN);
                    format!("neg({})", decoded_id)
                };
                if let Some(w) = state.trans_weights_exceptions.get(on) {
                    writeln!(f, "    {} -> {} (trans_weight: {})", char_repr, to, w)?;
                } else {
                    writeln!(f, "    {} -> {}", char_repr, to)?;
                }
            }
        }
        Ok(())
    }
}

// --- JSON ---

impl JSONConvertible for SimpleBitset {
    fn to_json(&self) -> JSONNode {
        let ranges_vec: Vec<Vec<usize>> = self.0.ranges().map(|ri| vec![*ri.start(), *ri.end()]).collect();
        ranges_vec.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let ranges_vec: Vec<Vec<usize>> = Vec::from_json(node)?;
        let mut ranges = Vec::new();
        for mut v in ranges_vec {
            if v.len() != 2 {
                return Err(format!("Expected 2-element array for SimpleBitset range, got {:?}", v));
            }
            let end = v.pop().unwrap();
            let start = v.pop().unwrap();
            ranges.push(start..=end);
        }
        Ok(SimpleBitset(RangeSetBlaze::from_iter(ranges)))
    }
}

impl<T: JSONConvertible> JSONConvertible for I16Map<T> {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("exceptions".to_string(), self.exceptions.to_json());
        obj.insert("default".to_string(), self.default.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let exceptions =
            BTreeMap::<i16, T>::from_json(obj.remove("exceptions").ok_or("Missing 'exceptions' field")?)?;
        let default = Option::<T>::from_json(obj.remove("default").ok_or("Missing 'default' field")?)?;
        Ok(I16Map { exceptions, default })
    }
}

impl JSONConvertible for DWAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("transitions".to_string(), self.transitions.to_json());
        obj.insert("final_weight".to_string(), self.final_weight.to_json());
        obj.insert("trans_weight_default".to_string(), self.trans_weight_default.to_json());
        obj.insert("trans_weights_exceptions".to_string(), self.trans_weights_exceptions.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let transitions =
            I16Map::<StateID>::from_json(obj.remove("transitions").ok_or("Missing 'transitions' field")?)?;
        let final_weight =
            Option::<Weight>::from_json(obj.remove("final_weight").ok_or("Missing 'final_weight' field")?)?;
        let trans_weight_default = Option::<Weight>::from_json(
            obj.remove("trans_weight_default").ok_or("Missing 'trans_weight_default' field")?,
        )?;
        let trans_weights_exceptions = BTreeMap::<i16, Weight>::from_json(
            obj.remove("trans_weights_exceptions").ok_or("Missing 'trans_weights_exceptions' field")?,
        )?;
        Ok(DWAState { transitions, final_weight, trans_weight_default, trans_weights_exceptions })
    }
}

impl JSONConvertible for DWA {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), self.states.0.to_json());
        obj.insert("start_state".to_string(), self.body.start_state.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let states = Vec::<DWAState>::from_json(obj.remove("states").ok_or("Missing 'states' field")?)?;
        let start_state = StateID::from_json(obj.remove("start_state").ok_or("Missing 'start_state' field")?)?;
        Ok(DWA { states: DWAStates(states), body: DWABody { start_state } })
    }
}

// --- Test Helpers ---
#[cfg(test)]
pub(crate) fn assert_dwa_equivalent(mut a: DWA, mut b: DWA) {
    // Strategy:
    // 1) Simplify both automata to obtain canonical, minimized, and normalized forms
    //    (unreachable pruned, sink-like states collapsed, redundant exceptions removed),
    //    while aggregating edge/default weights as unions across merged states.
    // 2) Perform a BFS isomorphism test from the start states. For each matched pair:
    //    - Compare state weights and final weights (None treated as zeros).
    //    - Compare default transition weights (None treated as zeros).
    //    - For each character in the union of exception keys, compare per-edge weights
    //      (falling back to default if exception weight absent).
    //    - Ensure default and per-exception targets correspond under the evolving bijection.
    // 3) Verify that all states reachable in `b` are matched by some state of `a`.

    a.simplify();
    b.simplify();

    // Helper: convert Option<Weight> to Weight (None => zeros).
    fn opt_w_to_w(ow: &Option<Weight>) -> Weight {
        ow.clone().unwrap_or_else(Weight::zeros)
    }

    use std::collections::{BTreeSet, HashSet};

    // Map a-state -> b-state and its inverse to ensure a bijection.
    let mut map_ab: HashMap<StateID, StateID> = HashMap::new();
    let mut map_ba: HashMap<StateID, BTreeSet<StateID>> = HashMap::new();
    let mut q: VecDeque<(StateID, StateID)> = VecDeque::new();

    assert!(
        !a.states.is_empty() && !b.states.is_empty(),
        "Automata should have at least one state after simplification.\n\nDWA A:\n{}\n\nDWA B:\n{}",
        a, b
    );

    let start_a = a.body.start_state;
    let start_b = b.body.start_state;
    map_ab.insert(start_a, start_b);
    map_ba.entry(start_b).or_default().insert(start_a);
    q.push_back((start_a, start_b));

    // Lookup per-edge weight for a specific character in a given state:
    // - If an exception weight is present, use it.
    // - Else, use the default weight (or zeros if absent).
    let edge_weight = |st: &DWAState, ch: i16| -> Weight {
        if let Some(w) = st.trans_weights_exceptions.get(&ch) {
            w.clone()
        } else {
            opt_w_to_w(&st.trans_weight_default)
        }
    };

    while let Some((ia, ib)) = q.pop_front() {
        let sa = &a.states[ia];
        let sb = &b.states[ib];

        // Compare default transition weights (None considered zeros).
        let dwa = opt_w_to_w(&sa.trans_weight_default);
        let dwb = opt_w_to_w(&sb.trans_weight_default);
        assert_eq!(
            dwa, dwb,
            "Default transition weight mismatch at (a:{}, b:{}): a.def_weight={} vs b.def_weight={}\n\nDWA A:\n{}\n\nDWA B:\n{}",
            ia, ib, dwa, dwb, a, b
        );

        // Union of exception keys; after simplify(), both representations should be normalized,
        // but we compute the union to be robust.
        let keys_a: BTreeSet<i16> = sa.transitions.exceptions.keys().cloned().collect();
        let keys_b: BTreeSet<i16> = sb.transitions.exceptions.keys().cloned().collect();
        let all_keys: BTreeSet<i16> = keys_a.union(&keys_b).cloned().collect();

        // Compare and enqueue per-exception transitions.
        let def_a = sa.transitions.default;
        let def_b = sb.transitions.default;

        for ch in all_keys {
            let ta = sa.transitions.exceptions.get(&ch).copied().or(def_a);
            let tb = sb.transitions.exceptions.get(&ch).copied().or(def_b);

            let wa = edge_weight(sa, ch);
            let wb = edge_weight(sb, ch);
            assert_eq!(
                wa, wb,
                "Per-edge weight mismatch on char {} at (a:{}, b:{}): a.edge_weight={} vs b.edge_weight={}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                ch, ia, ib, wa, wb, a, b
            );

            match (ta, tb) {
                (Some(ta_id), Some(tb_id)) => {
                    if let Some(&mapped) = map_ab.get(&ta_id) {
                        assert_eq!(
                            mapped, tb_id,
                            "Transition mismatch on char {} from (a:{}, b:{}): a-target {} is already mapped to b-target {}, but encountered b-target {}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                            ch, ia, ib, ta_id, mapped, tb_id, a, b
                        );
                    } else {
                        map_ab.insert(ta_id, tb_id);
                        map_ba.entry(tb_id).or_default().insert(ta_id);
                        q.push_back((ta_id, tb_id));
                    }
                }
                (None, None) => { /* Both lack transition for this char; fine. */ }
                _ => {
                    panic!(
                        "Presence mismatch for transition on char {} at (a:{}, b:{}): a-target={:?}, b-target={:?}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                        ch, ia, ib, ta, tb, a, b
                    );
                }
            }
        }

        // Compare and enqueue default transitions.
        match (def_a, def_b) {
            (Some(ta_id), Some(tb_id)) => {
                if let Some(&mapped) = map_ab.get(&ta_id) {
                    assert_eq!(
                        mapped, tb_id,
                        "Default transition mismatch from (a:{}, b:{}): a-target {} already mapped to b-target {}, but encountered b-target {}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                        ia, ib, ta_id, mapped, tb_id, a, b
                    );
                } else {
                    map_ab.insert(ta_id, tb_id);
                    map_ba.entry(tb_id).or_default().insert(ta_id);
                    q.push_back((ta_id, tb_id));
                }
            }
            (None, None) => { /* No default on either side; fine. */ }
            _ => {
                panic!(
                    "Default transition presence mismatch at (a:{}, b:{}): a.default={:?}, b.default={:?}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                    ia, ib, def_a, def_b, a, b
                );
            }
        }
    }

    // After establishing the state mapping, verify that the union of final weights
    // for all `a` states mapping to a given `b` state is equal to the final weight
    // of that `b` state. This handles cases where `a` is an unminimized version of `b`.
    for (ib, ias) in &map_ba {
        let mut union_fa = Weight::zeros();
        for &ia in ias {
            union_fa |= &opt_w_to_w(&a.states[ia].final_weight);
        }
        let fb = opt_w_to_w(&b.states[*ib].final_weight);
        assert_eq!(
            union_fa, fb,
            "Aggregated final weight mismatch for b-state {} (a-states: {:?}): union(a)={}, b={}\n\nDWA A:\n{}\n\nDWA B:\n{}",
            ib, ias, union_fa, fb, a, b
        );
    }

    // Ensure we've covered all states reachable from b.start.
    let mut reachable_b: HashSet<StateID> = HashSet::new();
    let mut qb: VecDeque<StateID> = VecDeque::new();
    reachable_b.insert(b.body.start_state);
    qb.push_back(b.body.start_state);
    while let Some(u) = qb.pop_front() {
        if let Some(d) = b.states[u].transitions.default {
            if reachable_b.insert(d) {
                qb.push_back(d);
            }
        }
        for &v in b.states[u].transitions.exceptions.values() {
            if reachable_b.insert(v) {
                qb.push_back(v);
            }
        }
    }
    let mapped_b: HashSet<StateID> = map_ab.values().cloned().collect();
    assert_eq!(
        mapped_b, reachable_b,
        "Reachable-state mismatch in `b`: matched set = {:?}, reachable set = {:?}\n\nDWA A:\n{}\n\nDWA B:\n{}",
        mapped_b, reachable_b, a, b
    );
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_bitset_ops() {
        let set1 = SimpleBitset::from_iter(vec![1, 2, 5]);
        let set2 = SimpleBitset::from_iter(vec![2, 3, 5, 6]);
        let all = SimpleBitset::all();
        let zeros = SimpleBitset::zeros();

        assert_eq!((&set1 & &set2).iter_up_to(10).collect::<Vec<_>>(), vec![2, 5]);
        assert_eq!((&set1 | &set2).iter_up_to(10).collect::<Vec<_>>(), vec![1, 2, 3, 5, 6]);
        assert!((&set1 & &all).contains(1));
        assert!((&set1 | &zeros).contains(1));
        assert_eq!((&set1 | &zeros).len(), 3);
        assert!((&set1 & &zeros).is_empty());
    }

    #[test]
    fn test_dwa_builder() {
        let mut dwa = DWA::new();
        assert_eq!(dwa.states.len(), 1);
        assert_eq!(dwa.body.start_state, 0);

        let s1 = dwa.add_state();
        assert_eq!(s1, 1);
        assert_eq!(dwa.states.len(), 2);

        dwa.set_final_weight(1, SimpleBitset::from_item(20)).unwrap();

        assert_eq!(dwa.states[1].final_weight, Some(SimpleBitset::from_item(20)));

        dwa.add_transition(0, b'a' as i16, 1, SimpleBitset::from_item(30)).unwrap();
        assert_eq!(*dwa.states[0].transitions.get(b'a' as i16).unwrap(), 1);
        assert_eq!(*dwa.states[0].trans_weights_exceptions.get(&(b'a' as i16)).unwrap(), SimpleBitset::from_item(30));

        // Test error cases
        let res = dwa.add_transition(0, b'a' as i16, 1, SimpleBitset::zeros());
        assert!(matches!(res, Err(DWABuildError::TransitionAlreadyExists { from: 0, on: 97 })));

        dwa.set_default_transition(0, 0, SimpleBitset::from_item(40)).unwrap();
        assert_eq!(dwa.states[0].transitions.default, Some(0));
        assert_eq!(dwa.states[0].trans_weight_default, Some(SimpleBitset::from_item(40)));

        let res = dwa.set_default_transition(0, 0, SimpleBitset::zeros());
        assert!(matches!(res, Err(DWABuildError::DefaultTransitionAlreadyExists { from: 0 })));

        let res = dwa.set_final_weight(10, SimpleBitset::zeros());
        assert!(matches!(res, Err(DWABuildError::StateOutOfBounds { state: 10 })));
    }

    #[test]
    fn test_union_and_concatenate() {
        // DWA1: on 'a' go to final
        let mut d1 = DWA::new();
        let s1 = d1.add_state();
        d1.add_transition(d1.body.start_state, b'a' as i16, s1, SimpleBitset::all()).unwrap();
        d1.set_final_weight(s1, SimpleBitset::from_item(1)).unwrap();

        // DWA2: on 'b' go to final
        let mut d2 = DWA::new();
        let t1 = d2.add_state();
        d2.add_transition(d2.body.start_state, b'b' as i16, t1, SimpleBitset::all()).unwrap();
        d2.set_final_weight(t1, SimpleBitset::from_item(2)).unwrap();

        // Union
        let u = d1.union(&d2);
        assert_eq!(u.states.len() >= 2, true);

        // Concatenate: ends of d1 join to start of d2
        let c = d1.concatenate(&d2);
        assert!(c.states.len() >= 2);
    }

    // --- Advanced Tests ---

    /// Helper to create a DWA that accepts a single character and produces a final weight.
    fn dwa_accepts_char(ch: char, final_weight: Weight) -> DWA {
        let mut dwa = DWA::new();
        let final_state = dwa.add_state();
        dwa.add_transition(dwa.body.start_state, ch as i16, final_state, Weight::all()).unwrap();
        dwa.set_final_weight(final_state, final_weight).unwrap();
        dwa
    }

    /// Helper to create a DWA that accepts a string and produces a final weight.
    fn dwa_from_str(s: &str, final_weight: Weight) -> DWA {
        let mut dwa = DWA::new();
        let mut current_state = dwa.body.start_state;
        for ch in s.chars() {
            let next_state = dwa.add_state();
            dwa.add_transition(current_state, ch as i16, next_state, Weight::all()).unwrap();
            current_state = next_state;
        }
        dwa.set_final_weight(current_state, final_weight).unwrap();
        dwa
    }

    #[test]
    fn test_simplify_redundant_states() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();
        let s3 = d.add_state(); // Should be merged with s2
        let s4 = d.add_state(); // Final state
        let s5 = d.add_state(); // Unreachable

        d.add_transition(0, 'a' as i16, s1, Weight::all()).unwrap();
        d.add_transition(0, 'b' as i16, s2, Weight::all()).unwrap();
        d.add_transition(0, 'c' as i16, s3, Weight::all()).unwrap();
        d.add_transition(s1, 'x' as i16, s4, Weight::all()).unwrap();
        d.add_transition(s2, 'y' as i16, s4, Weight::all()).unwrap();
        d.add_transition(s3, 'y' as i16, s4, Weight::all()).unwrap(); // Same behavior as s2
        d.set_final_weight(s4, Weight::from_item(1)).unwrap();

        assert_eq!(d.states.len(), 6);
        d.simplify();
        // s5 pruned (unreachable). s2 and s3 merged.
        // Expected states: start, 'a'-state, 'b'/'c'-state, final-state. Total 4.
        assert_eq!(d.states.len(), 4);
    }

    #[test]
    fn test_union_simple() {
        let d1 = dwa_accepts_char('a', Weight::from_item(1));
        let d2 = dwa_accepts_char('b', Weight::from_item(2));

        let mut expected = DWA::new();
        let s_a = expected.add_state();
        let s_b = expected.add_state();
        expected.add_transition(0, 'a' as i16, s_a, Weight::all()).unwrap();
        expected.add_transition(0, 'b' as i16, s_b, Weight::all()).unwrap();
        expected.set_final_weight(s_a, Weight::from_item(1)).unwrap();
        expected.set_final_weight(s_b, Weight::from_item(2)).unwrap();

        let u = d1.union(&d2);
        assert_dwa_equivalent(u, expected);
    }

    #[test]
    fn test_union_overlapping() {
        let d1 = dwa_accepts_char('a', Weight::from_item(1));
        let mut d2 = dwa_accepts_char('b', Weight::from_item(3));
        let s_a2 = d2.add_state();
        d2.add_transition(d2.body.start_state, 'a' as i16, s_a2, Weight::all()).unwrap();
        d2.set_final_weight(s_a2, Weight::from_item(2)).unwrap();

        let mut expected = DWA::new();
        let s_a = expected.add_state();
        let s_b = expected.add_state();
        expected.add_transition(0, 'a' as i16, s_a, Weight::all()).unwrap();
        expected.add_transition(0, 'b' as i16, s_b, Weight::all()).unwrap();
        expected.set_final_weight(s_a, Weight::from_iter(vec![1, 2])).unwrap();
        expected.set_final_weight(s_b, Weight::from_item(3)).unwrap();

        let u = d1.union(&d2);
        assert_dwa_equivalent(u, expected);
    }

    #[test]
    fn test_concatenate_simple() {
        let d1 = dwa_accepts_char('a', Weight::from_item(1)); // Final state is 1
        let d2 = dwa_accepts_char('b', Weight::from_item(2));
        let c = d1.concatenate(&d2);
        let expected = dwa_from_str("ab", Weight::from_item(2));
        assert_dwa_equivalent(c, expected);
    }

    #[test]
    fn test_apply_weight() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        d.set_final_weight(0, Weight::from_iter(vec![5, 6])).unwrap();
        d.add_transition(0, 'a' as i16, s1, Weight::from_iter(vec![100, 101])).unwrap();
        d.set_default_transition(0, 0, Weight::from_iter(vec![200, 201])).unwrap();

        let gate = Weight::from_iter(vec![6, 11, 101, 201]);
        let new_start = d.apply_weight(&gate);

        assert_eq!(d.body.start_state, new_start);
        let new_start_state = &d.states[new_start];
        assert_eq!(new_start_state.final_weight, Some(Weight::from_item(6)));
        assert_eq!(new_start_state.trans_weights_exceptions.get(&('a' as i16)), Some(&Weight::from_item(101)));
        assert_eq!(new_start_state.trans_weight_default, Some(Weight::from_item(201)));
        assert_eq!(new_start_state.transitions.exceptions.get(&('a' as i16)), Some(&s1));
        assert_eq!(new_start_state.transitions.default, Some(0));
    }

    /// Helper that creates a DWA with a single transition on `ch` with a given
    /// per-edge weight, landing in a final state with the provided final weight.
    fn dwa_with_char_and_weights(ch: char, edge_weight: Weight, final_weight: Weight) -> DWA {
        let mut d = DWA::new();
        let s = d.add_state();
        d.add_transition(d.body.start_state, ch as i16, s, edge_weight).unwrap();
        d.set_final_weight(s, final_weight).unwrap();
        d
    }

    #[test]
    fn test_simple_bitset_iter_up_to_all() {
        let all = Weight::all();
        let vals: Vec<_> = all.iter_up_to(5).collect();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_i16map_get_prefers_exception() {
        let mut m = I16Map::with_default(7);
        m.exceptions.insert(3, 42);
        assert_eq!(m.get(3), Some(&42));
        assert_eq!(m.get(4), Some(&7));
    }

    #[test]
    fn test_normalize_removes_redundant_exceptions() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        // Different state weight to avoid accidental merging
        d.set_final_weight(s1, Weight::from_item(1)).unwrap();
        // Default goes to s1
        d.set_default_transition(d.body.start_state, s1, Weight::from_item(50)).unwrap();
        // Redundant exceptions also go to s1
        d.add_transition(d.body.start_state, 'x' as i16, s1, Weight::from_item(60)).unwrap();
        d.add_transition(d.body.start_state, 'y' as i16, s1, Weight::from_item(61)).unwrap();

        // Sanity: exceptions exist pre-simplify
        assert_eq!(d.states[d.body.start_state].transitions.exceptions.len(), 2);
        assert_eq!(d.states[d.body.start_state].trans_weights_exceptions.len(), 2);

        d.simplify();
        let start = d.body.start_state;

        // normalize_edges_inplace should remove exceptions equal to the default target,
        // and their weights as well.
        assert!(d.states[start].transitions.exceptions.is_empty(), "Exceptions equal to default should be removed");
        assert!(d.states[start].trans_weights_exceptions.is_empty(), "Weights for removed exceptions should also be removed");
        assert!(d.states[start].transitions.default.is_some(), "Default should remain");
    }

    #[test]
    fn test_partition_refinement_aggregates_weights() {
        // Build two states that are behaviorally equivalent (same outputs/transitions),
        // differing only in their per-edge/default weights. They should be merged,
        // and the merged state's per-edge/default weights should be the union.
        let mut d = DWA::new();
        let s2 = d.add_state();
        let s3 = d.add_state();
        let sF = d.add_state();
        d.set_final_weight(sF, Weight::from_item(1)).unwrap();

        // Reachability
        d.add_transition(d.body.start_state, 'a' as i16, s2, Weight::all()).unwrap();
        d.add_transition(d.body.start_state, 'b' as i16, s3, Weight::all()).unwrap();

        // Self default loops with different weights
        d.set_default_transition(s2, s2, Weight::from_item(200)).unwrap();
        d.set_default_transition(s3, s3, Weight::from_item(201)).unwrap();
        // Same structural exception 'z' to sF, but different per-edge weights
        d.add_transition(s2, 'z' as i16, sF, Weight::from_item(100)).unwrap();
        d.add_transition(s3, 'z' as i16, sF, Weight::from_item(101)).unwrap();

        assert_eq!(d.states.len(), 4);
        d.simplify();
        // s2 and s3 should merge into one state.
        assert_eq!(d.states.len(), 3);

        // Find the merged middle state (it will be the only one with a 'z' exception).
        let mut z_owner: Option<StateID> = None;
        for (id, st) in d.states.0.iter().enumerate() {
            if st.transitions.exceptions.contains_key(&('z' as i16)) {
                z_owner = Some(id);
                break;
            }
        }
        let mid = z_owner.expect("Merged state with 'z' transition not found");
        assert_eq!(
            d.states[mid].trans_weights_exceptions.get(&('z' as i16)),
            Some(&Weight::from_iter(vec![100, 101]))
        );
        assert_eq!(
            d.states[mid].trans_weight_default,
            Some(Weight::from_iter(vec![200, 201]))
        );
    }

    #[test]
    fn test_union_transition_weight_union() {
        fn build(ch: char, ew: usize, fw: usize) -> DWA {
            dwa_with_char_and_weights(ch, Weight::from_item(ew), Weight::from_item(fw))
        }
        let d1 = build('x', 10, 1);
        let d2 = build('x', 20, 2);
        let u = d1.union(&d2);

        // Expected automaton with unioned transition and final weights on 'x'.
        let mut expected = DWA::new();
        let s = expected.add_state();
        expected
            .add_transition(0, 'x' as i16, s, Weight::from_iter(vec![10, 20]))
            .unwrap();
        expected
            .set_final_weight(s, Weight::from_iter(vec![1, 2]))
            .unwrap();

        assert_dwa_equivalent(u, expected);
    }

    #[test]
    fn test_json_roundtrip_complex() {
        use crate::json_serialization::JSONConvertible;

        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();
        d.set_default_transition(d.body.start_state, s1, Weight::from_iter(vec![1, 2, 3]))
            .unwrap();
        d.add_transition(d.body.start_state, 'x' as i16, s2, Weight::from_item(99))
            .unwrap();
        d.set_final_weight(s2, Weight::from_iter(vec![5, 7])).unwrap();

        let node = d.to_json();
        let d2 = DWA::from_json(node.clone()).expect("from_json should succeed");
        assert_eq!(node, d2.to_json(), "Roundtrip JSON should be stable");
    }

    #[test]
    fn test_add_transition_out_of_bounds() {
        let mut d = DWA::new();
        let res = d.add_transition(5, 'a' as i16, 0, Weight::zeros());
        assert!(matches!(res, Err(DWABuildError::StateOutOfBounds { state: 5 })));

        let res2 = d.add_transition(0, 'a' as i16, 99, Weight::zeros());
        assert!(matches!(res2, Err(DWABuildError::StateOutOfBounds { state: 99 })));
    }

    #[test]
    fn test_prune_unreachable_with_default_chain() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        let _s2 = d.add_state(); // Unused, unreachable
        d.set_default_transition(d.body.start_state, s1, Weight::all())
            .unwrap();
        d.set_final_weight(s1, Weight::from_item(1)).unwrap();
        d.add_transition(s1, 'x' as i16, s1, Weight::all()).unwrap();

        // Completely unreachable component
        let s_unreach = d.add_state();
        d.add_transition(s_unreach, 'z' as i16, s_unreach, Weight::all())
            .unwrap();

        let before = d.states.len();
        d.simplify();
        let after = d.states.len();
        assert!(after < before, "Unreachable states should be pruned");
        assert_eq!(after, 2, "Only start and s1 should remain reachable");
    }

    #[test]
    fn test_equivalence_via_simplification() {
        // DWA 'a' has explicit transitions for inputs '1' and '3' which lead
        // to non-final, sink-like states. State 1 is a true sink, and state 2
        // only transitions to state 1.
        let mut a = DWA::new();
        let s1a = a.add_state();
        let s2a = a.add_state();
        a.add_transition(0, 0, s1a, Weight::from_item(1)).unwrap();
        a.add_transition(0, 1, s2a, Weight::from_iter(0..=1)).unwrap();
        a.add_transition(0, 2, s1a, Weight::from_item(0)).unwrap();
        a.add_transition(0, 3, s1a, Weight::from_iter(0..=1)).unwrap();
        a.set_default_transition(s2a, s1a, Weight::all()).unwrap();

        // DWA 'b' lacks these transitions. For inputs '1' and '3', it transitions
        // to an implicit sink. The simplification process should make 'a' equivalent
        // to 'b'.
        let mut b = DWA::new();
        let s1b = b.add_state();
        b.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
        b.add_transition(0, 2, s1b, Weight::from_item(0)).unwrap();

        assert_dwa_equivalent(a, b);
    }

    #[test]
    fn test_concatenate_left_start_is_final() {
        // LEFT: DWA (start: 0)
        //   State 0:
        //     weight: []
        //     final_weight: [0]
        let mut left = DWA::new();
        left.set_final_weight(left.body.start_state, Weight::from_item(0)).unwrap();

        // RIGHT: DWA (start: 0)
        //   State 0:
        //     weight: []
        //     final_weight: ALL
        let mut right = DWA::new();
        right.set_final_weight(right.body.start_state, Weight::all()).unwrap();

        let c = left.concatenate(&right);

        let mut expected = DWA::new();
        expected.set_final_weight(expected.body.start_state, Weight::from_item(0)).unwrap();

        assert_dwa_equivalent(c, expected);
    }

    #[test]
    fn test_union_complex_from_attachment() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

        // --- Build LEFT DWA ---
        let mut left = DWA::new();
        for _ in 0..47 {
            left.add_state();
        }

        left.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
        left.add_transition(0, 2, 2, Weight::from_item(1)).unwrap();
        left.add_transition(0, 3, 3, Weight::from_item(1)).unwrap();
        left.add_transition(0, 4, 4, Weight::from_item(1)).unwrap();
        left.add_transition(0, 5, 5, Weight::from_item(1)).unwrap();
        left.add_transition(0, 6, 6, Weight::from_item(1)).unwrap();
        left.add_transition(0, 7, 7, Weight::from_item(1)).unwrap();
        left.add_transition(0, 8, 8, Weight::from_item(1)).unwrap();
        left.add_transition(0, 9, 9, Weight::from_item(1)).unwrap();
        left.add_transition(0, 10, 10, Weight::from_item(1)).unwrap();
        left.add_transition(1, neg(0), 11, Weight::all()).unwrap();
        left.set_default_transition(2, 12, Weight::all()).unwrap();
        left.add_transition(3, neg(3), 13, Weight::all()).unwrap();
        left.add_transition(5, 3, 14, Weight::all()).unwrap();
        left.add_transition(5, 7, 9, Weight::all()).unwrap();
        left.set_default_transition(6, 5, Weight::all()).unwrap();
        left.add_transition(7, neg(7), 15, Weight::all()).unwrap();
        left.set_default_transition(8, 9, Weight::all()).unwrap();
        left.add_transition(9, 3, 16, Weight::all()).unwrap();
        left.add_transition(9, 7, 9, Weight::all()).unwrap();
        left.add_transition(10, 5, 5, Weight::all()).unwrap();
        left.add_transition(11, neg(9), 17, Weight::all()).unwrap();
        left.set_default_transition(12, 18, Weight::all()).unwrap();
        left.add_transition(13, neg(9), 19, Weight::all()).unwrap();
        left.add_transition(14, neg(3), 20, Weight::all()).unwrap();
        left.add_transition(15, neg(9), 21, Weight::all()).unwrap();
        left.add_transition(16, neg(3), 22, Weight::all()).unwrap();
        let w01 = Weight::from_iter(0..=1);
        left.add_transition(17, 2, 23, w01.clone()).unwrap();
        left.add_transition(17, 4, 24, w01.clone()).unwrap();
        left.add_transition(17, 5, 25, w01.clone()).unwrap();
        left.add_transition(17, 6, 26, w01.clone()).unwrap();
        left.add_transition(17, 8, 27, w01.clone()).unwrap();
        left.add_transition(17, 9, 28, w01.clone()).unwrap();
        left.add_transition(17, 10, 29, w01.clone()).unwrap();
        left.add_transition(19, 2, 23, w01.clone()).unwrap();
        left.add_transition(19, 4, 24, w01.clone()).unwrap();
        left.add_transition(19, 5, 25, w01.clone()).unwrap();
        left.add_transition(19, 6, 26, w01.clone()).unwrap();
        left.add_transition(19, 8, 27, w01.clone()).unwrap();
        left.add_transition(19, 9, 28, w01.clone()).unwrap();
        left.add_transition(19, 10, 29, w01.clone()).unwrap();
        left.add_transition(20, neg(0), 30, Weight::all()).unwrap();
        left.add_transition(21, 2, 23, w01.clone()).unwrap();
        left.add_transition(21, 4, 24, w01.clone()).unwrap();
        left.add_transition(21, 5, 25, w01.clone()).unwrap();
        left.add_transition(21, 6, 26, w01.clone()).unwrap();
        left.add_transition(21, 8, 27, w01.clone()).unwrap();
        left.add_transition(21, 9, 28, w01.clone()).unwrap();
        left.add_transition(21, 10, 29, w01.clone()).unwrap();
        left.add_transition(22, neg(0), 31, Weight::all()).unwrap();
        left.set_default_transition(23, 32, Weight::all()).unwrap();
        left.add_transition(25, 7, 28, Weight::all()).unwrap();
        left.set_default_transition(26, 25, Weight::all()).unwrap();
        left.set_default_transition(27, 28, Weight::all()).unwrap();
        left.add_transition(28, 0, 33, Weight::all()).unwrap();
        left.add_transition(28, 3, 34, Weight::all()).unwrap();
        left.add_transition(28, 7, 35, Weight::all()).unwrap();
        left.add_transition(29, 5, 25, Weight::all()).unwrap();
        left.add_transition(30, neg(9), 36, Weight::all()).unwrap();
        left.add_transition(31, neg(9), 37, Weight::all()).unwrap();
        left.set_default_transition(32, 38, Weight::all()).unwrap();
        left.add_transition(33, neg(0), 39, Weight::all()).unwrap();
        left.add_transition(34, neg(3), 40, Weight::all()).unwrap();
        left.add_transition(35, neg(7), 41, Weight::all()).unwrap();
        left.add_transition(36, 2, 23, w01.clone()).unwrap();
        left.add_transition(36, 4, 24, w01.clone()).unwrap();
        left.add_transition(36, 5, 25, w01.clone()).unwrap();
        left.add_transition(36, 6, 26, w01.clone()).unwrap();
        left.add_transition(36, 8, 27, w01.clone()).unwrap();
        left.add_transition(36, 9, 28, w01.clone()).unwrap();
        left.add_transition(36, 10, 29, w01.clone()).unwrap();
        left.add_transition(37, 2, 23, w01.clone()).unwrap();
        left.add_transition(37, 4, 24, w01.clone()).unwrap();
        left.add_transition(37, 5, 25, w01.clone()).unwrap();
        left.add_transition(37, 6, 26, w01.clone()).unwrap();
        left.add_transition(37, 8, 27, w01.clone()).unwrap();
        left.add_transition(37, 9, 28, w01.clone()).unwrap();
        left.add_transition(37, 10, 29, w01.clone()).unwrap();
        left.add_transition(39, neg(5), 42, Weight::all()).unwrap();
        left.add_transition(40, neg(5), 43, Weight::all()).unwrap();
        left.add_transition(41, neg(5), 44, Weight::all()).unwrap();
        left.add_transition(42, neg(10), 45, Weight::all()).unwrap();
        left.add_transition(43, neg(10), 46, Weight::all()).unwrap();
        left.add_transition(44, neg(10), 47, Weight::all()).unwrap();
        left.set_final_weight(45, Weight::all()).unwrap();
        left.set_final_weight(46, Weight::all()).unwrap();
        left.set_final_weight(47, Weight::all()).unwrap();

        // --- Build RIGHT DWA ---
        let mut right = DWA::new();
        for _ in 0..42 {
            right.add_state();
        }

        right.add_transition(0, 2, 1, Weight::from_item(0)).unwrap();
        right.add_transition(0, 4, 2, Weight::from_item(0)).unwrap();
        right.add_transition(0, 5, 3, Weight::from_item(0)).unwrap();
        right.add_transition(0, 6, 4, Weight::from_item(0)).unwrap();
        right.add_transition(0, 8, 5, Weight::from_item(0)).unwrap();
        right.add_transition(0, 9, 6, Weight::from_item(0)).unwrap();
        right.add_transition(0, 10, 7, Weight::from_item(0)).unwrap();
        right.set_default_transition(1, 8, Weight::all()).unwrap();
        right.add_transition(3, 7, 6, Weight::all()).unwrap();
        right.set_default_transition(4, 3, Weight::all()).unwrap();
        right.set_default_transition(5, 6, Weight::all()).unwrap();
        right.add_transition(6, 0, 9, Weight::all()).unwrap();
        right.add_transition(6, 3, 10, Weight::all()).unwrap();
        right.add_transition(6, 7, 11, Weight::all()).unwrap();
        right.add_transition(7, 5, 3, Weight::all()).unwrap();
        right.set_default_transition(8, 12, Weight::all()).unwrap();
        right.add_transition(9, neg(0), 13, Weight::all()).unwrap();
        right.add_transition(10, neg(3), 14, Weight::all()).unwrap();
        right.add_transition(11, neg(7), 15, Weight::all()).unwrap();
        right.add_transition(13, neg(5), 16, Weight::all()).unwrap();
        right.add_transition(14, neg(5), 17, Weight::all()).unwrap();
        right.add_transition(15, neg(5), 18, Weight::all()).unwrap();
        right.add_transition(16, neg(10), 19, Weight::all()).unwrap();
        right.add_transition(17, neg(10), 20, Weight::all()).unwrap();
        right.add_transition(18, neg(10), 21, Weight::all()).unwrap();
        right.add_transition(19, 2, 22, w01.clone()).unwrap();
        right.add_transition(19, 4, 23, w01.clone()).unwrap();
        right.add_transition(19, 5, 24, w01.clone()).unwrap();
        right.add_transition(19, 6, 25, w01.clone()).unwrap();
        right.add_transition(19, 8, 26, w01.clone()).unwrap();
        right.add_transition(19, 9, 27, w01.clone()).unwrap();
        right.add_transition(19, 10, 28, w01.clone()).unwrap();
        right.add_transition(20, 2, 22, w01.clone()).unwrap();
        right.add_transition(20, 4, 23, w01.clone()).unwrap();
        right.add_transition(20, 5, 24, w01.clone()).unwrap();
        right.add_transition(20, 6, 25, w01.clone()).unwrap();
        right.add_transition(20, 8, 26, w01.clone()).unwrap();
        right.add_transition(20, 9, 27, w01.clone()).unwrap();
        right.add_transition(20, 10, 28, w01.clone()).unwrap();
        right.add_transition(21, 2, 22, w01.clone()).unwrap();
        right.add_transition(21, 4, 23, w01.clone()).unwrap();
        right.add_transition(21, 5, 24, w01.clone()).unwrap();
        right.add_transition(21, 6, 25, w01.clone()).unwrap();
        right.add_transition(21, 8, 26, w01.clone()).unwrap();
        right.add_transition(21, 9, 27, w01.clone()).unwrap();
        right.add_transition(21, 10, 28, w01.clone()).unwrap();
        right.set_default_transition(22, 29, Weight::all()).unwrap();
        right.add_transition(24, 7, 27, Weight::all()).unwrap();
        right.set_default_transition(25, 24, Weight::all()).unwrap();
        right.set_default_transition(26, 27, Weight::all()).unwrap();
        right.add_transition(27, 0, 30, Weight::all()).unwrap();
        right.add_transition(27, 3, 31, Weight::all()).unwrap();
        right.add_transition(27, 7, 32, Weight::all()).unwrap();
        right.add_transition(28, 5, 24, Weight::all()).unwrap();
        right.set_default_transition(29, 33, Weight::all()).unwrap();
        right.add_transition(30, neg(0), 34, Weight::all()).unwrap();
        right.add_transition(31, neg(3), 35, Weight::all()).unwrap();
        right.add_transition(32, neg(7), 36, Weight::all()).unwrap();
        right.add_transition(34, neg(5), 37, Weight::all()).unwrap();
        right.add_transition(35, neg(5), 38, Weight::all()).unwrap();
        right.add_transition(36, neg(5), 39, Weight::all()).unwrap();
        right.add_transition(37, neg(10), 40, Weight::all()).unwrap();
        right.add_transition(38, neg(10), 41, Weight::all()).unwrap();
        right.add_transition(39, neg(10), 42, Weight::all()).unwrap();
        right.set_final_weight(40, Weight::all()).unwrap();
        right.set_final_weight(41, Weight::all()).unwrap();
        right.set_final_weight(42, Weight::all()).unwrap();

        todo!();
    }

    #[test]
    fn test_union_complex_from_attachment_simpified() {
        fn neg(val: i16) -> i16 {
            val.wrapping_add(i16::MIN)
        }

        // Build left DWA
        let mut left = DWA::new();
        for _ in 0..20 {
            left.add_state();
        }
        assert_eq!(left.states.len(), 21);

        // State 0
        left.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
        left.add_transition(0, 3, 2, Weight::from_item(1)).unwrap();
        left.add_transition(0, 5, 3, Weight::from_item(1)).unwrap();
        left.add_transition(0, 6, 4, Weight::from_item(1)).unwrap();
        left.add_transition(0, 7, 5, Weight::from_item(1)).unwrap();
        left.add_transition(0, 8, 4, Weight::from_item(1)).unwrap();
        left.add_transition(0, 9, 3, Weight::from_item(1)).unwrap();
        left.add_transition(0, 10, 6, Weight::from_item(1)).unwrap();
        // State 1
        left.add_transition(1, neg(0), 7, Weight::all()).unwrap();
        // State 2
        left.add_transition(2, neg(3), 7, Weight::all()).unwrap();
        // State 3
        left.add_transition(3, 3, 8, Weight::all()).unwrap();
        left.add_transition(3, 7, 3, Weight::all()).unwrap();
        // State 4
        left.set_default_transition(4, 3, Weight::all()).unwrap();
        // State 5
        left.add_transition(5, neg(7), 7, Weight::all()).unwrap();
        // State 6
        left.add_transition(6, 5, 3, Weight::all()).unwrap();
        // State 7
        left.add_transition(7, neg(9), 9, Weight::all()).unwrap();
        // State 8
        left.add_transition(8, neg(3), 1, Weight::all()).unwrap();
        // State 9
        left.add_transition(9, 5, 10, Weight::from_iter(0..=1)).unwrap();
        left.add_transition(9, 6, 11, Weight::from_iter(0..=1)).unwrap();
        left.add_transition(9, 8, 12, Weight::from_iter(0..=1)).unwrap();
        left.add_transition(9, 9, 13, Weight::from_iter(0..=1)).unwrap();
        left.add_transition(9, 10, 14, Weight::from_iter(0..=1)).unwrap();
        // State 10
        left.add_transition(10, 7, 13, Weight::all()).unwrap();
        // State 11
        left.set_default_transition(11, 10, Weight::all()).unwrap();
        // State 12
        left.set_default_transition(12, 13, Weight::all()).unwrap();
        // State 13
        left.add_transition(13, 0, 15, Weight::all()).unwrap();
        left.add_transition(13, 3, 16, Weight::all()).unwrap();
        left.add_transition(13, 7, 17, Weight::all()).unwrap();
        // State 14
        left.add_transition(14, 5, 10, Weight::all()).unwrap();
        // State 15
        left.add_transition(15, neg(0), 18, Weight::all()).unwrap();
        // State 16
        left.add_transition(16, neg(3), 18, Weight::all()).unwrap();
        // State 17
        left.add_transition(17, neg(7), 18, Weight::all()).unwrap();
        // State 18
        left.add_transition(18, neg(5), 19, Weight::all()).unwrap();
        // State 19
        left.add_transition(19, neg(10), 20, Weight::all()).unwrap();
        // State 20
        left.set_final_weight(20, Weight::from_item(1)).unwrap();

        // Build right DWA
        let mut right = DWA::new();
        for _ in 0..22 {
            right.add_state();
        }
        assert_eq!(right.states.len(), 23);

        // State 0
        right.add_transition(0, 5, 1, Weight::from_item(0)).unwrap();
        right.add_transition(0, 6, 2, Weight::from_item(0)).unwrap();
        right.add_transition(0, 8, 3, Weight::from_item(0)).unwrap();
        right.add_transition(0, 9, 4, Weight::from_item(0)).unwrap();
        right.add_transition(0, 10, 5, Weight::from_item(0)).unwrap();
        // State 1
        right.add_transition(1, 7, 4, Weight::all()).unwrap();
        // State 2
        right.set_default_transition(2, 1, Weight::all()).unwrap();
        // State 3
        right.set_default_transition(3, 4, Weight::all()).unwrap();
        // State 4
        right.add_transition(4, 0, 6, Weight::all()).unwrap();
        right.add_transition(4, 3, 7, Weight::all()).unwrap();
        right.add_transition(4, 7, 8, Weight::all()).unwrap();
        // State 5
        right.add_transition(5, 5, 1, Weight::all()).unwrap();
        // State 6
        right.add_transition(6, neg(0), 9, Weight::all()).unwrap();
        // State 7
        right.add_transition(7, neg(3), 9, Weight::all()).unwrap();
        // State 8
        right.add_transition(8, neg(7), 9, Weight::all()).unwrap();
        // State 9
        right.add_transition(9, neg(5), 10, Weight::all()).unwrap();
        // State 10
        right.add_transition(10, neg(10), 11, Weight::all()).unwrap();
        // State 11
        right.add_transition(11, 5, 12, Weight::from_iter(0..=1)).unwrap();
        right.add_transition(11, 6, 13, Weight::from_iter(0..=1)).unwrap();
        right.add_transition(11, 8, 14, Weight::from_iter(0..=1)).unwrap();
        right.add_transition(11, 9, 15, Weight::from_iter(0..=1)).unwrap();
        right.add_transition(11, 10, 16, Weight::from_iter(0..=1)).unwrap();
        // State 12
        right.add_transition(12, 7, 15, Weight::all()).unwrap();
        // State 13
        right.set_default_transition(13, 12, Weight::all()).unwrap();
        // State 14
        right.set_default_transition(14, 15, Weight::all()).unwrap();
        // State 15
        right.add_transition(15, 0, 17, Weight::all()).unwrap();
        right.add_transition(15, 3, 18, Weight::all()).unwrap();
        right.add_transition(15, 7, 19, Weight::all()).unwrap();
        // State 16
        right.add_transition(16, 5, 12, Weight::all()).unwrap();
        // State 17
        right.add_transition(17, neg(0), 20, Weight::all()).unwrap();
        // State 18
        right.add_transition(18, neg(3), 20, Weight::all()).unwrap();
        // State 19
        right.add_transition(19, neg(7), 20, Weight::all()).unwrap();
        // State 20
        right.add_transition(20, neg(5), 21, Weight::all()).unwrap();
        // State 21
        right.add_transition(21, neg(10), 22, Weight::all()).unwrap();
        // State 22
        right.set_final_weight(22, Weight::from_item(0)).unwrap();

        todo!();
    }
}
