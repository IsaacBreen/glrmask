// src/precompute4/weighted_automata.rs

#![allow(dead_code)]

use crate::json_serialization::{JSONConvertible, JSONNode};
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::iter::FromIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Deref, Index, IndexMut, Not, Sub, SubAssign};
use std::time::Instant;

// --- Part 1: SimpleBitset ---

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct SimpleBitset(pub RangeSetBlaze<usize>);

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

impl SubAssign<&SimpleBitset> for SimpleBitset {
    fn sub_assign(&mut self, rhs: &SimpleBitset) {
        self.0 = &self.0 - &rhs.0;
    }
}

impl SubAssign<SimpleBitset> for SimpleBitset {
    fn sub_assign(&mut self, rhs: SimpleBitset) {
        self.0 = &self.0 - &rhs.0;
    }
}

impl Sub<SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: SimpleBitset) -> Self::Output {
        &self - &rhs
    }
    }

impl<'a> Sub<&'a SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: &'a SimpleBitset) -> Self::Output {
        SimpleBitset(&self.0 - &rhs.0)
    }
}

impl Not for SimpleBitset {
    type Output = SimpleBitset;
    fn not(self) -> Self::Output {
        &SimpleBitset::all() - &self
    }
}

impl Not for &SimpleBitset {
    type Output = SimpleBitset;
    fn not(self) -> Self::Output {
        &SimpleBitset::all() - self
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

    /// Subtracts a weight from all weights in this state.
    pub fn exclude_weight(&mut self, weight: &Weight) {
        if let Some(fw) = &mut self.final_weight {
            *fw -= weight;
            if fw.is_empty() {
                self.final_weight = None;
            }
        }

        if let Some(twd) = &mut self.trans_weight_default {
            *twd -= weight;
        }

        for w in self.trans_weights_exceptions.values_mut() {
            *w -= weight;
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

    pub fn apply_weight(&mut self, state_id: StateID, weight: &Weight) {
        assert!(state_id < self.len(), "state_id out of bounds");
        self[state_id].apply_weight(weight);
    }

    pub fn exclude_weight_from_state(
        &mut self,
        state_id: StateID,
        weight: &Weight,
        gating_memo: &mut HashMap<(StateID, Weight), StateID>,
    ) -> StateID {
        assert!(state_id < self.len(), "state_id out of bounds");
        if weight.is_empty() {
            return state_id;
        }

        let key = (state_id, weight.clone());
        if let Some(&id) = gating_memo.get(&key) { return id; }

        // Check if any weight in the state will be affected. If not, we don't need to copy.
        let state = &self[state_id];
        let needs_change = state.final_weight.as_ref().map_or(false, |w| !(&*w & weight).is_empty())
            || state.trans_weight_default.as_ref().map_or(false, |w| !(&*w & weight).is_empty())
            || state.trans_weights_exceptions.values().any(|w| !(&*w & weight).is_empty());

        if !needs_change {
            gating_memo.insert(key, state_id);
            return state_id;
        }

        let new_id = self.copy_state(state_id);
        self[new_id].exclude_weight(weight);
        gating_memo.insert(key, new_id);
        new_id
    }

    pub fn union_assign_state(
        &mut self,
        s_from_id: StateID,
        s_into_id: StateID,
        memo: &mut HashMap<(StateID, StateID), StateID>,
        gating_memo: &mut HashMap<(StateID, Weight), StateID>,
    ) {
        assert!(s_from_id < self.len(), "s_from_id out of bounds");
        assert!(s_into_id < self.len(), "s_into_id out of bounds");
        if s_from_id == s_into_id {
            return;
        }
        // crate::debug!(2, "Unioning state {} into state {}", s_from_id, s_into_id);

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
        let w_def_from = s_from.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::zeros);
        let w_def_into = s_into_orig.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::zeros);
        let new_def_tgt_id = match (def_tgt_from, def_tgt_into) {
            (Some(t1), Some(t2)) => if t1 == t2 {
                Some(t1)
            } else {
                let to_exclude_from_t2 = &w_def_from - &w_def_into;
                let t2_gated = self.exclude_weight_from_state(t2, &to_exclude_from_t2, gating_memo);
                let to_exclude_from_t1 = &w_def_into - &w_def_from;
                let t1_gated = self.exclude_weight_from_state(t1, &to_exclude_from_t1, gating_memo);
                Some(self.union_state(t1_gated, t2_gated, memo, gating_memo))
            },
            (Some(t), None) | (None, Some(t)) => Some(t),
            (None, None) => None,
        };
        new_transitions.default = new_def_tgt_id;
        new_trans_weight_default = if new_def_tgt_id.is_some() {
            Some(w_def_from | w_def_into)
        } else {
            None
        };

        // Exception transitions
        for &ch in &critical_points {
            let tgt_from = get_target(&s_from, ch);
            let tgt_into = get_target(&s_into_orig, ch);
            let w_from = get_weight(&s_from, ch);
            let w_into = get_weight(&s_into_orig, ch);

            let new_exc_tgt_id = match (tgt_from, tgt_into) {
                (Some(t1), Some(t2)) => if t1 == t2 {
                    Some(t1)
                } else {
                    let to_exclude_from_t2 = &w_from - &w_into;
                    let t2_gated = self.exclude_weight_from_state(t2, &to_exclude_from_t2, gating_memo);
                    let to_exclude_from_t1 = &w_into - &w_from;
                    let t1_gated = self.exclude_weight_from_state(t1, &to_exclude_from_t1, gating_memo);
                    Some(self.union_state(t1_gated, t2_gated, memo, gating_memo))
                },
                (Some(t), None) | (None, Some(t)) => Some(t),
                (None, None) => None,
            };

            if new_exc_tgt_id != new_def_tgt_id {
                if let Some(tgt_id) = new_exc_tgt_id {
                    new_transitions.exceptions.insert(ch, tgt_id);
                    new_trans_weights_exceptions.insert(ch, w_from | w_into);
                }
            }
        }

        // Commit changes
        let s_into = &mut self.0[s_into_id];
        s_into.final_weight = if new_final_weight.is_empty() { None } else { Some(new_final_weight) };
        s_into.transitions = new_transitions;
        s_into.trans_weight_default = new_trans_weight_default;
        s_into.trans_weights_exceptions = new_trans_weights_exceptions;
        // crate::debug!(2, "Done unioning state {} into state {}", s_from_id, s_into_id);
    }

    pub fn union_state(
        &mut self,
        s1_id: StateID,
        s2_id: StateID,
        memo: &mut HashMap<(StateID, StateID), StateID>,
        gating_memo: &mut HashMap<(StateID, Weight), StateID>,
    ) -> StateID {
        assert!(s1_id < self.len(), "s1_id out of bounds");
        assert!(s2_id < self.len(), "s2_id out of bounds");

        let key = (s1_id.min(s2_id), s1_id.max(s2_id));
        if let Some(&id) = memo.get(&key) {
            return id;
        }

        // Create a new state by copying s1
        let new_id = self.copy_state(s1_id);

        // Crucial: insert into memo *before* recursive calls to handle cycles.
        memo.insert(key, new_id);

        // Merge s2 into the new state
        self.union_assign_state(s2_id, new_id, memo, gating_memo);

        new_id
    }

    pub fn copy_subgraph(&mut self, start_id: StateID) -> (StateID, HashMap<StateID, StateID>) {
        let mut remap = HashMap::new();
        let mut q = VecDeque::new();

        if start_id >= self.len() {
            let new_start_id = self.add_state();
            return (new_start_id, remap);
        }

        let new_start_id = self.add_existing_state(self[start_id].clone());
        remap.insert(start_id, new_start_id);
        q.push_back((start_id, new_start_id));

        while let Some((old_id, new_id)) = q.pop_front() {
            let old_state_clone = self[old_id].clone();

            // Remap default transition
            if let Some(old_target) = old_state_clone.transitions.default {
                let new_target_id = *remap.entry(old_target).or_insert_with(|| {
                    let new_id = self.add_existing_state(self[old_target].clone());
                    q.push_back((old_target, new_id));
                    new_id
                });
                self[new_id].transitions.default = Some(new_target_id);
            }

            // Remap exception transitions
            let mut new_exceptions = BTreeMap::new();
            for (ch, &old_target) in &old_state_clone.transitions.exceptions {
                let new_target_id = *remap.entry(old_target).or_insert_with(|| {
                    let new_id = self.add_existing_state(self[old_target].clone());
                    q.push_back((old_target, new_id));
                    new_id
                });
                new_exceptions.insert(*ch, new_target_id);
            }
            self[new_id].transitions.exceptions = new_exceptions;
        }
        (new_start_id, remap)
    }

    pub fn copy_subgraph_from(&mut self, other_states: &DWAStates, start_id: StateID) -> (StateID, HashMap<StateID, StateID>) {
        let mut remap = HashMap::new();
        let mut q = VecDeque::new();

        if start_id >= other_states.len() {
            let new_start_id = self.add_state();
            return (new_start_id, remap);
        }

        let new_start_id = self.add_existing_state(other_states[start_id].clone());
        remap.insert(start_id, new_start_id);
        q.push_back((start_id, new_start_id));

        while let Some((old_id, new_id)) = q.pop_front() {
            let old_state_clone = other_states[old_id].clone();

            if let Some(old_target) = old_state_clone.transitions.default {
                let new_target_id = *remap.entry(old_target).or_insert_with(|| {
                    let new_id = self.add_existing_state(other_states[old_target].clone());
                    q.push_back((old_target, new_id));
                    new_id
                });
                self[new_id].transitions.default = Some(new_target_id);
            }

            self[new_id].transitions.exceptions = old_state_clone.transitions.exceptions.iter().map(|(ch, &old_target)| {
                let new_target_id = *remap.entry(old_target).or_insert_with(|| {
                    let new_id = self.add_existing_state(other_states[old_target].clone());
                    q.push_back((old_target, new_id));
                    new_id
                });
                (*ch, new_target_id)
            }).collect();
        }
        (new_start_id, remap)
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
            if Self::propagate_future_weights(states) {
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

    pub fn propagate_future_weights(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }

        // 1. Compute future weights via backward analysis.
        // future_weights[i] = union of weights of all accepting paths starting from i.
        let mut rev_adj: Vec<Vec<StateID>> = vec![vec![]; n];
        for i in 0..n {
            if let Some(d) = states[i].transitions.default {
                if d < n {
                    rev_adj[d].push(i);
                }
            }
            for &v in states[i].transitions.exceptions.values() {
                if v < n {
                    rev_adj[v].push(i);
                }
            }
        }
        for preds in rev_adj.iter_mut() {
            preds.sort_unstable();
            preds.dedup();
        }

        let mut future_weights = vec![Weight::zeros(); n];
        let mut worklist = VecDeque::new();
        for i in 0..n {
            if let Some(fw) = &states[i].final_weight {
                if !fw.is_empty() {
                    future_weights[i] = fw.clone();
                    for &pred in &rev_adj[i] {
                        worklist.push_back(pred);
                    }
                }
            }
        }

        while let Some(u) = worklist.pop_front() {
            let mut u_new_fw = states[u].final_weight.clone().unwrap_or_else(Weight::zeros);
            let u_state = &states[u];

            // Default transition
            if let Some(v) = u_state.transitions.default {
                if v < n {
                    if let Some(edge_w) = u_state.trans_weight_default.as_ref() {
                        u_new_fw |= &(edge_w & &future_weights[v]);
                    }
                }
            }

            // Exception transitions
            for (&ch, &v) in &u_state.transitions.exceptions {
                if v < n {
                    if let Some(edge_w) = u_state.trans_weights_exceptions.get(&ch) {
                        u_new_fw |= &(edge_w & &future_weights[v]);
                    }
                }
            }

            if u_new_fw != future_weights[u] {
                future_weights[u] = u_new_fw;
                for &pred in &rev_adj[u] {
                    worklist.push_back(pred);
                }
            }
        }

        // 2. Update edge weights. An edge weight W can be relaxed to W | !future_weight(target)
        // because any bits not in future_weight(target) would be filtered out anyway.
        let mut any_weight_changed = false;
        for i in 0..n {
            let st = &mut states[i];
            // Default
            if let Some(v) = st.transitions.default {
                if v < n {
                    if let Some(w) = st.trans_weight_default.as_mut() {
                        let not_future_v = !&future_weights[v];
                        let new_w = &*w | &not_future_v;
                        if new_w != *w {
                            *w = new_w;
                            any_weight_changed = true;
                        }
                    }
                }
            }
            // Exceptions
            let keys: Vec<i16> = st.transitions.exceptions.keys().copied().collect();
            for ch in keys {
                if let Some(&v) = st.transitions.exceptions.get(&ch) {
                    if v < n {
                        if let Some(w) = st.trans_weights_exceptions.get_mut(&ch) {
                            let not_future_v = !&future_weights[v];
                            let new_w = &*w | &not_future_v;
                            if new_w != *w {
                                *w = new_w;
                                any_weight_changed = true;
                            }
                        }
                    }
                }
            }
        }

        any_weight_changed
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
        if self.states.0.is_empty() {
            return other.clone();
        }
        if other.states.0.is_empty() {
            return self.clone();
        }

        let mut new_dwa = self.clone();
        let offset = new_dwa.states.len();

        for other_state in &other.states.0 {
            let mut new_state = other_state.clone();
            if let Some(d) = new_state.transitions.default.as_mut() {
                *d += offset;
            }
            for target in new_state.transitions.exceptions.values_mut() {
                *target += offset;
            }
            new_dwa.states.add_existing_state(new_state);
        }

        let other_start_remapped = other.body.start_state + offset;
        let self_start = new_dwa.body.start_state;

        let mut memo = HashMap::new();
        let mut gating_memo = HashMap::new();
        let new_start = new_dwa.states.union_state(self_start, other_start_remapped, &mut memo, &mut gating_memo);
        new_dwa.body.start_state = new_start;

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
    }

    /// Concatenation-like operation on two DWAs.
    /// "Join all ends to all starts": whenever the left component reaches a final state,
    /// we also (lazily) activate the right automaton's start state. If the left start is final,

    /// Concatenation-like operation on two DWAs.
    /// "Join all ends to all starts": whenever the left component reaches a final state,
    /// we also (lazily) activate the right automaton's start state. If the left start is final,
    /// the composition starts with the right start already active.
    ///
    /// This is implemented via a product-like construction where states in the new DWA
    /// are compositions of a state from `self` and a set of states from `other`.
    ///
    /// Returns the new DWA.
    pub fn concatenate(&self, other: &DWA) -> DWA {
        // If either automaton is empty (accepts no strings), the concatenation is empty.
        if self.states.0.is_empty() || other.states.0.is_empty() {
            return DWA::new();
        }

        let mut new_dwa = self.clone();
        let offset = new_dwa.states.len();

        let mut memo = HashMap::new();
        let mut gating_memo = HashMap::new();

        // Add all of other's states to the new DWA, remapping their transition targets.
        for other_state in &other.states.0 {
            let mut new_state = other_state.clone();
            if let Some(d) = new_state.transitions.default.as_mut() {
                *d += offset;
            }
            for target in new_state.transitions.exceptions.values_mut() {
                *target += offset;
            }
            new_dwa.states.add_existing_state(new_state);
        }

        let other_start_remapped = other.body.start_state + offset;

        // Identify all states in the `self` part that were final.
        let final_self_states: Vec<(StateID, Weight)> = new_dwa.states.0[..offset]
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.final_weight.clone().map(|w| (i, w)))
            .collect();

        // For each final state from `self`, graft on a gated version of `other`.
        for (s_a_id, final_weight) in final_self_states {
            // Create a copy of `other`'s start state, gated by `final_weight`.
            let mut gated_b_start = new_dwa.states[other_start_remapped].clone();
            gated_b_start.apply_weight(&final_weight);

            // The new final weight at the junction point is determined by whether `other`
            // could accept an empty string, gated by `final_weight`.
            let junction_final_weight = gated_b_start.final_weight.clone();

            // Add the gated state to the DWA to get an ID for union_assign_state.
            // This temporary state will be removed by simplify().
            let temp_id = new_dwa.states.add_existing_state(gated_b_start);

            // Before merging, clear the final weight of the `self` state.
            // We'll replace it with the correct junction final weight afterwards.
            new_dwa.states[s_a_id].final_weight = None;

            // Merge the gated start state into the final `self` state.
            new_dwa.states.union_assign_state(temp_id, s_a_id, &mut memo, &mut gating_memo);

            // Set the correct final weight at the junction.
            // It must be unioned with any final weight that might have been
            // propagated into s_a_id from other branches during the union.
            if let Some(w) = junction_final_weight {
                if let Some(existing_fw) = new_dwa.states[s_a_id].final_weight.as_mut() {
                    *existing_fw |= &w;
                } else {
                    new_dwa.states[s_a_id].final_weight = Some(w);
                }
            }
        }

        new_dwa
    }

    /// Component-based union. Operates on a shared DWAStates arena.
    pub fn union_components(states: &mut DWAStates, body1: &DWABody, body2: &DWABody) -> DWABody {
        let mut memo = HashMap::new();
        let mut gating_memo = HashMap::new();
        let new_start = states.union_state(body1.start_state, body2.start_state, &mut memo, &mut gating_memo);
        DWABody { start_state: new_start }
    }

    /// Component-based concatenation. Operates on a shared DWAStates arena.
    pub fn concatenate_components(states: &mut DWAStates, body1: &DWABody, body2: &DWABody) -> DWABody {
        // 1. Copy left subgraph to not modify shared states.
        let (new_start_a, remap_a) = states.copy_subgraph(body1.start_state);

        // 2. Find final states in the copied left automaton.
        let final_a_states: Vec<(StateID, Weight)> = remap_a.values()
            .filter_map(|&new_id| states[new_id].final_weight.clone().map(|w| (new_id, w)))
            .collect();

        let mut memo = HashMap::new();
        let mut gating_memo = HashMap::new();

        for (s_a_id, final_weight) in final_a_states {
            // Create a temporary, gated version of B's start state.
            // Note: B's states are NOT copied. We refer to them directly.
            // union_assign_state will copy them as needed.
            let mut gated_b_start = states[body2.start_state].clone();
            gated_b_start.apply_weight(&final_weight);
            let junction_final_weight = gated_b_start.final_weight.clone();
            let temp_id = states.add_existing_state(gated_b_start);

            states[s_a_id].final_weight = None;
            states.union_assign_state(temp_id, s_a_id, &mut memo, &mut gating_memo);

            if let Some(w) = junction_final_weight {
                if let Some(existing_fw) = states[s_a_id].final_weight.as_mut() {
                    *existing_fw |= &w;
                } else {
                    states[s_a_id].final_weight = Some(w);
                }
            }
        }

        // If start of A was final, the new start state must also be combined with B's start.
        if let Some(start_a_final_weight) = states[body1.start_state].final_weight.clone() {
            let mut gated_b_start = states[body2.start_state].clone();
            gated_b_start.apply_weight(&start_a_final_weight);
            let new_start_final_weight = gated_b_start.final_weight.clone();
            let temp_id = states.add_existing_state(gated_b_start);

            states.union_assign_state(temp_id, new_start_a, &mut memo, &mut gating_memo);

            if let Some(w) = new_start_final_weight {
                if let Some(existing_fw) = states[new_start_a].final_weight.as_mut() {
                    *existing_fw |= &w;
                } else {
                    states[new_start_a].final_weight = Some(w);
                }
            }
        }

        DWABody { start_state: new_start_a }
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
            if let Some(to) = &state.transitions.default {
                if let Some(w) = &state.trans_weight_default {
                    writeln!(f, "    * -> {} (trans_weight: {})", to, w)?;
                } else {
                    writeln!(f, "    * -> {}", to)?;
                }
            }
            if let Some(w) = &state.final_weight {
                writeln!(f, "    final_weight: {}", w)?;
            }
            for (on, to) in &state.transitions.exceptions {
                // Represent positive codes as char if ASCII; negatives as '-char' or '-num'
                let char_repr = if *on >= 0 {
                    let u = *on as u16;
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
