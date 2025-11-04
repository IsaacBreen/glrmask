// src/precompute4/weighted_automata.rs

#![allow(dead_code)]

use crate::json_serialization::{JSONConvertible, JSONNode};
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::iter::FromIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Deref, Index, IndexMut};
use std::time::Instant;

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

// --- Part 3: Automata Definitions (DWA only) ---

pub type StateID = usize;
pub type Weight = SimpleBitset;

/// Errors while building a DWA.
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
                write!(f, "Transition from state {} on symbol {} already exists", from, on)
            }
            DWABuildError::DefaultTransitionAlreadyExists { from } => {
                write!(f, "Default transition from state {} already exists", from)
            }
            DWABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DWAState {
    pub transitions: I16Map<StateID>,
    pub weight: Weight,
    pub final_weight: Option<Weight>,
    pub trans_weight_default: Option<Weight>,
    pub trans_weights_exceptions: BTreeMap<i16, Weight>,
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

impl DWAState {
    /// Returns Some(target) if:
    /// - only a default transition exists (no exceptions),
    /// - no final_weight,
    /// - and default edge's weight equals state's weight.
    pub fn simple_default_target(&self) -> Option<StateID> {
        if self.final_weight.is_none() && self.transitions.exceptions.is_empty() {
            if let (Some(target), Some(w)) = (self.transitions.default, self.trans_weight_default.as_ref()) {
                if &self.weight == w {
                    return Some(target);
                }
            }
        }
        None
    }
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

    pub fn set_state_weight(&mut self, state: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state });
        }
        self.states[state].weight = weight;
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

    /// Union of two DWAs via product construction.
    /// A state is final if either of the original states is final. Transition weights are OR-joined.
    /// Returns the new DWA and a map from new state IDs to the pair of original automaton states.
    /// The BTreeSet in the return value contains tuples of (automaton_index, state_id).
    pub fn union(&self, other: &DWA) -> (DWA, BTreeMap<StateID, BTreeSet<(usize, StateID)>>) {
        let mut new_dwa = DWA::default();
        let mut mapping: BTreeMap<StateID, BTreeSet<(usize, StateID)>> = BTreeMap::new();
        let mut pair_to_new_id: BTreeMap<(StateID, StateID), StateID> = BTreeMap::new();
        let mut worklist: VecDeque<(StateID, StateID)> = VecDeque::new();

        let sink0 = self.states.len();
        let sink1 = other.states.len();

        let mut get_or_create = |
            pair: (StateID, StateID),
            new_dwa: &mut DWA,
            pair_to_new_id: &mut BTreeMap<(StateID, StateID), StateID>,
            worklist: &mut VecDeque<(StateID, StateID)>,
            mapping: &mut BTreeMap<StateID, BTreeSet<(usize, StateID)>>
        | -> StateID {
            if let Some(&id) = pair_to_new_id.get(&pair) {
                return id;
            }
            let new_id = new_dwa.states.0.len();
            new_dwa.states.0.push(DWAState::default());
            pair_to_new_id.insert(pair, new_id);
            worklist.push_back(pair);

            let mut merged_from = BTreeSet::new();
            if pair.0 != sink0 { merged_from.insert((0, pair.0)); }
            if pair.1 != sink1 { merged_from.insert((1, pair.1)); }
            mapping.insert(new_id, merged_from);

            new_id
        };

        if self.states.is_empty() && other.states.is_empty() {
            return (new_dwa, mapping);
        }

        let start_pair = (self.body.start_state, other.body.start_state);
        new_dwa.body.start_state =
            get_or_create(start_pair, &mut new_dwa, &mut pair_to_new_id, &mut worklist, &mut mapping);

        while let Some((id0, id1)) = worklist.pop_front() {
            let new_id = *pair_to_new_id.get(&(id0, id1)).unwrap();
            let s0 = if id0 == sink0 { None } else { Some(&self.states[id0]) };
            let s1 = if id1 == sink1 { None } else { Some(&other.states[id1]) };

            let new_state = &mut new_dwa.states[new_id];

            let w0 = s0.map(|s| &s.weight).cloned().unwrap_or_else(Weight::zeros);
            let w1 = s1.map(|s| &s.weight).cloned().unwrap_or_else(Weight::zeros);
            new_state.weight = &w0 | &w1;

            let fw0 = s0.and_then(|s| s.final_weight.as_ref()).cloned().unwrap_or_else(Weight::zeros);
            let fw1 = s1.and_then(|s| s.final_weight.as_ref()).cloned().unwrap_or_else(Weight::zeros);
            let final_w = &fw0 | &fw1;
            if !final_w.is_empty() {
                new_state.final_weight = Some(final_w);
            }

            let mut critical_points = BTreeSet::new();
            if let Some(s) = s0 { critical_points.extend(s.transitions.exceptions.keys()); }
            if let Some(s) = s1 { critical_points.extend(s.transitions.exceptions.keys()); }

            let get_target = |s: Option<&DWAState>, sink: StateID, ch: i16| -> StateID {
                s.and_then(|s| s.transitions.get(ch).copied()).unwrap_or(sink)
            };

            // Default transition (pair them)
            let def_tgt0 = s0.and_then(|s| s.transitions.default).unwrap_or(sink0);
            let def_tgt1 = s1.and_then(|s| s.transitions.default).unwrap_or(sink1);
            let def_pair = (def_tgt0, def_tgt1);
            let new_def_tgt =
                get_or_create(def_pair, &mut new_dwa, &mut pair_to_new_id, &mut worklist, &mut mapping);
            new_dwa.states[new_id].transitions.default = Some(new_def_tgt);

            let tw_def0 = s0.and_then(|s| s.trans_weight_default.as_ref()).cloned().unwrap_or_else(Weight::zeros);
            let tw_def1 = s1.and_then(|s| s.trans_weight_default.as_ref()).cloned().unwrap_or_else(Weight::zeros);
            new_dwa.states[new_id].trans_weight_default = Some(&tw_def0 | &tw_def1);

            // Exception transitions
            for &ch in &critical_points {
                let tgt0 = get_target(s0, sink0, ch);
                let tgt1 = get_target(s1, sink1, ch);
                let exc_pair = (tgt0, tgt1);

                if exc_pair != def_pair {
                    let new_exc_tgt =
                        get_or_create(exc_pair, &mut new_dwa, &mut pair_to_new_id, &mut worklist, &mut mapping);
                    new_dwa.states[new_id].transitions.exceptions.insert(ch, new_exc_tgt);

                    let tw_exc0 = s0
                        .and_then(|s| s.trans_weights_exceptions.get(&ch))
                        .cloned()
                        .unwrap_or_else(Weight::zeros);
                    let tw_exc1 = s1
                        .and_then(|s| s.trans_weights_exceptions.get(&ch))
                        .cloned()
                        .unwrap_or_else(Weight::zeros);
                    new_dwa.states[new_id]
                        .trans_weights_exceptions
                        .insert(ch, &tw_exc0 | &tw_exc1);
                }
            }
        }

        (new_dwa, mapping)
    }

    /// A flexible concatenation-like operation on two DWAs.
    /// The `join_map` specifies which states in `self` (the left DWA) are "join points".
    /// When a join point `s_left` is entered, it's as if we also simultaneously enter
    /// all states from `other` (the right DWA) associated with `s_left` in the `join_map`.
    ///
    /// This is implemented via a product-like construction where states in the new DWA
    /// are compositions of a state from `self` and a set of states from `other`.
    ///
    /// Note: This is not a standard concatenation. The final states of the resulting
    /// automaton are the union of final states from both components. For example, if
    /// `join_map` connects final states of `self` to the start of `other`, the
    /// resulting language is closer to `L(self) ∪ L(self)L(other)`.
    ///
    /// Returns the new DWA and a map from new state IDs to the set of original automaton states.
    /// The BTreeSet in the return value contains tuples of (automaton_index, state_id).
    pub fn concatenate(
        &self,
        other: &DWA,
        join_map: &BTreeMap<StateID, BTreeSet<StateID>>,
    ) -> (DWA, BTreeMap<StateID, BTreeSet<(usize, StateID)>>) {
        let mut new_dwa = DWA::new();
        new_dwa.states.0.clear(); // start with no states

        let mut mapping: BTreeMap<StateID, BTreeSet<(usize, StateID)>> = BTreeMap::new();
        let mut composition_to_new_id: BTreeMap<(StateID, BTreeSet<StateID>), StateID> = BTreeMap::new();
        let mut worklist: VecDeque<(StateID, BTreeSet<StateID>)> = VecDeque::new();

        let sink0 = self.states.len();
        let sink1 = other.states.len();

        let mut get_or_create = |comp: (StateID, BTreeSet<StateID>),
                                 new_dwa: &mut DWA,
                                 comp_to_new_id: &mut BTreeMap<(StateID, BTreeSet<StateID>), StateID>,
                                 worklist: &mut VecDeque<(StateID, BTreeSet<StateID>)>,
                                 mapping: &mut BTreeMap<StateID, BTreeSet<(usize, StateID)>>|
         -> StateID {
            if let Some(&id) = comp_to_new_id.get(&comp) {
                return id;
            }
            let new_id = new_dwa.states.0.len();
            new_dwa.states.0.push(DWAState::default());
            comp_to_new_id.insert(comp.clone(), new_id);
            worklist.push_back(comp.clone());

            let mut merged_from = BTreeSet::new();
            if comp.0 != sink0 {
                merged_from.insert((0, comp.0));
            }
            for &s1 in &comp.1 {
                if s1 != sink1 {
                    merged_from.insert((1, s1));
                }
            }
            mapping.insert(new_id, merged_from);

            new_id
        };

        if self.states.is_empty() {
            return (DWA::new(), BTreeMap::new());
        }

        let start0 = self.body.start_state;
        let start1_set = join_map.get(&start0).cloned().unwrap_or_default();
        let start_comp = (start0, start1_set);
        new_dwa.body.start_state =
            get_or_create(start_comp, &mut new_dwa, &mut composition_to_new_id, &mut worklist, &mut mapping);

        while let Some((id0, ids1)) = worklist.pop_front() {
            let new_id = *composition_to_new_id.get(&(id0, ids1.clone())).unwrap();
            let s0 = if id0 == sink0 { None } else { Some(&self.states[id0]) };

            let new_state = &mut new_dwa.states[new_id];

            // Aggregate weights
            let mut agg_weight = s0.map(|s| s.weight.clone()).unwrap_or_else(Weight::zeros);
            let mut agg_final_weight = s0.and_then(|s| s.final_weight.as_ref()).cloned().unwrap_or_else(Weight::zeros);
            for &id1 in &ids1 {
                if id1 != sink1 {
                    let s1 = &other.states[id1];
                    agg_weight |= &s1.weight;
                    if let Some(fw) = &s1.final_weight {
                        agg_final_weight |= fw;
                    }
                }
            }
            new_state.weight = agg_weight;
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

            // Default transition
            let def_tgt0 = s0.and_then(|s| s.transitions.default).unwrap_or(sink0);
            let mut def_tgts1: BTreeSet<StateID> =
                ids1
                    .iter()
                    .filter(|&&id1| id1 != sink1)
                    .map(|&id1| other.states[id1].transitions.default.unwrap_or(sink1))
                    .collect();
            def_tgts1.remove(&sink1);

            if let Some(joins) = join_map.get(&def_tgt0) {
                def_tgts1.extend(joins);
            }

            let def_comp = (def_tgt0, def_tgts1.clone());
            let new_def_tgt =
                get_or_create(def_comp.clone(), &mut new_dwa, &mut composition_to_new_id, &mut worklist, &mut mapping);
            new_dwa.states[new_id].transitions.default = Some(new_def_tgt);

            let mut tw_def = s0
                .and_then(|s| s.trans_weight_default.as_ref())
                .cloned()
                .unwrap_or_else(Weight::zeros);
            for &id1 in &ids1 {
                if id1 != sink1 {
                    tw_def |= &other.states[id1].trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::zeros);
                }
            }
            new_dwa.states[new_id].trans_weight_default = Some(tw_def);

            // Exception transitions
            for &ch in &critical_points {
                let tgt0 = get_target(s0, sink0, ch);
                let mut tgts1: BTreeSet<StateID> =
                    ids1
                        .iter()
                        .filter(|&&id1| id1 != sink1)
                        .map(|&id1| get_target(Some(&other.states[id1]), sink1, ch))
                        .collect();
                tgts1.remove(&sink1);

                if let Some(joins) = join_map.get(&tgt0) {
                    tgts1.extend(joins);
                }

                let exc_comp = (tgt0, tgts1);
                if exc_comp != def_comp {
                    let new_exc_tgt =
                        get_or_create(exc_comp, &mut new_dwa, &mut composition_to_new_id, &mut worklist, &mut mapping);
                    new_dwa.states[new_id].transitions.exceptions.insert(ch, new_exc_tgt);

                    let mut tw_exc = s0
                        .and_then(|s| s.trans_weights_exceptions.get(&ch).cloned())
                        .unwrap_or_else(Weight::zeros);
                    for &id1 in &ids1 {
                        if id1 != sink1 {
                            tw_exc |= &other.states[id1]
                                .trans_weights_exceptions
                                .get(&ch)
                                .cloned()
                                .unwrap_or_else(Weight::zeros);
                        }
                    }
                    new_dwa.states[new_id].trans_weights_exceptions.insert(ch, tw_exc);
                }
            }
        }

        (new_dwa, mapping)
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

        // Snapshot the old start state's data before mutating the arena.
        let old = states[old_start].clone();

        let mut new_state = DWAState::default();
        // Gate state/output weights
        new_state.weight = &old.weight & weight;
        if let Some(fw) = &old.final_weight {
            let gw = fw & weight;
            if !gw.is_empty() {
                new_state.final_weight = Some(gw);
            }
        }
        // Copy structure of transitions (targets), gate their weights.
        new_state.transitions.default = old.transitions.default;
        new_state.transitions.exceptions = old.transitions.exceptions.clone();
        new_state.trans_weight_default = old.trans_weight_default.as_ref().map(|dw| dw & weight);
        new_state.trans_weights_exceptions =
            old.trans_weights_exceptions.into_iter().map(|(ch, w)| (ch, &w & weight)).collect();

        states[new_id] = new_state;
        body.start_state = new_id;
        new_id
    }

    /// Normalize edges and simplify by bisimulation-like partition refinement and reachability pruning.
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
        println!(
            "DWA::simplify_components ({} states -> {} states) took: {:?}",
            initial_len,
            states.len(),
            now.elapsed()
        );
    }

    /// Drop exceptions equal to default, and remove dangling per-exception weights.
    pub fn normalize_edges_inplace(states: &mut DWAStates) -> bool {
        let mut changed = false;
        for st in &mut states.0 {
            let before = st.transitions.exceptions.len();
            if let Some(def) = st.transitions.default {
                st.transitions.exceptions.retain(|_, &mut tgt| tgt != def);
            }
            changed |= st.transitions.exceptions.len() != before;

            let before_w = st.trans_weights_exceptions.len();
            st.trans_weights_exceptions
                .retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            changed |= st.trans_weights_exceptions.len() != before_w;
        }
        changed
    }

    /// Partition-refinement minimization. Two states are equivalent iff they share:
    /// - weight and final_weight,
    /// - default target's partition,
    /// - and exception targets per symbol (up to default-equivalence).
    pub fn minimize_partition_refinement(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let n = states.0.len();
        if n <= 1 {
            return false;
        }
        let sink_pid: usize = n;

        // Initial partition by outputs (weight, final_weight).
        let mut part: Vec<usize> = vec![0; n];
        let mut canon0: HashMap<(Weight, Option<Weight>), usize> = HashMap::new();
        for i in 0..n {
            let key = (states[i].weight.clone(), states[i].final_weight.clone());
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
            let mut sig2pid: HashMap<(Weight, Option<Weight>, usize, Vec<(i16, usize)>), usize> = HashMap::new();

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
                let sig = (st.weight.clone(), st.final_weight.clone(), def_cls, ex);
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
            st.weight = rep_state.weight.clone();
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
                new_states[new_id]
                    .transitions
                    .exceptions
                    .insert(ch, *pid_to_new.get(&cls).unwrap());
            }
        }

        states.0 = new_states;
        Self::normalize_edges_inplace(states);
        let start_pid = part[body.start_state];
        body.start_state = *pid_to_new.get(&start_pid).unwrap();

        true
    }

    /// Remove states unreachable from `start_state` and renumber them densely.
    pub fn prune_unreachable(states: &mut DWAStates, body: &mut DWABody) -> bool {
        if states.0.is_empty() {
            return false;
        }
        let n = states.0.len();

        // 1. Forward reachability from start_state
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        if body.start_state < n {
            visited[body.start_state] = true;
            q.push_back(body.start_state);
        }

        while let Some(u) = q.pop_front() {
            if let Some(d) = states[u].transitions.default {
                if d < n && !visited[d] {
                    visited[d] = true;
                    q.push_back(d);
                }
            }
            for &v in states[u].transitions.exceptions.values() {
                if v < n && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        if visited.iter().all(|&b| b) {
            return false;
        }

        // 4. Remap
        let mut map = vec![usize::MAX; n];
        let mut next_id = 0usize;
        for i in 0..n {
            if visited[i] {
                map[i] = next_id;
                next_id += 1;
            }
        }

        if next_id == n {
            return false;
        }

        let mut new_states: Vec<DWAState> = Vec::with_capacity(next_id);
        for old in 0..n {
            if !visited[old] {
                continue;
            }
            let mut st = states[old].clone();
            if let Some(d) = st.transitions.default {
                st.transitions.default = Some(map[d]);
            }
            let ex = st.transitions.exceptions.clone();
            st.transitions.exceptions.clear();
            for (ch, tgt) in ex {
                st.transitions.exceptions.insert(ch, map[tgt]);
            }
            new_states.push(st);
        }
        states.0 = new_states;
        body.start_state = map[body.start_state];
        true
    }
}

// --- Display Implementations for Debugging ---
impl Display for DWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            writeln!(f, "    weight: {}", state.weight)?;
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
                let sym_repr = if *on >= 0 {
                    let u = *on as u32;
                    if let Some(c) = char::from_u32(u) {
                        if c.is_ascii_graphic() || c == ' ' {
                            format!("'{}'", c)
                        } else {
                            format!("{}", *on)
                        }
                    } else {
                        format!("{}", *on)
                    }
                } else {
                    format!("{}", *on)
                };
                if let Some(w) = state.trans_weights_exceptions.get(on) {
                    writeln!(f, "    {} -> {} (trans_weight: {})", sym_repr, to, w)?;
                } else {
                    writeln!(f, "    {} -> {}", sym_repr, to)?;
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
        obj.insert("weight".to_string(), self.weight.to_json());
        obj.insert("final_weight".to_string(), self.final_weight.to_json());
        obj.insert("trans_weight_default".to_string(), self.trans_weight_default.to_json());
        obj.insert("trans_weights_exceptions".to_string(), self.trans_weights_exceptions.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let transitions =
            I16Map::<StateID>::from_json(obj.remove("transitions").ok_or("Missing 'transitions' field")?)?;
        let weight = Weight::from_json(obj.remove("weight").ok_or("Missing 'weight' field")?)?;
        let final_weight =
            Option::<Weight>::from_json(obj.remove("final_weight").ok_or("Missing 'final_weight' field")?)?;
        let trans_weight_default = Option::<Weight>::from_json(
            obj.remove("trans_weight_default").ok_or("Missing 'trans_weight_default' field")?,
        )?;
        let trans_weights_exceptions = BTreeMap::<i16, Weight>::from_json(
            obj.remove("trans_weights_exceptions").ok_or("Missing 'trans_weights_exceptions' field")?,
        )?;
        Ok(DWAState { transitions, weight, final_weight, trans_weight_default, trans_weights_exceptions })
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

// --- Negative symbol resolution (placeholder) ---

/// Transform a DWA that may contain negative-labeled transitions (i16 < 0)
/// into an equivalent DWA without negative labels. This function is intended
/// to "resolve" the special negative labels that were used as stand-ins for
/// end-stack hitches.
/// Note: This is intentionally left unimplemented for now.
pub fn resolve_negative_edges(_dwa: &mut DWA) {
    todo!("resolve_negative_edges: implement rewriting of negative-labeled transitions before conversion");
}

// --- Tests (minimal) ---
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
    fn test_i16_map() {
        let mut map = I16Map::with_default(100);
        map.exceptions.insert(97, 10);
        map.exceptions.insert(98, 20);

        assert_eq!(map.get(97), Some(&10));
        assert_eq!(map.get(120), Some(&100));
        assert_eq!(map.get(98), Some(&20));
    }

    #[test]
    fn test_dwa_builder() {
        let mut dwa = DWA::new();
        assert_eq!(dwa.states.len(), 1);
        assert_eq!(dwa.body.start_state, 0);

        let s1 = dwa.add_state();
        assert_eq!(s1, 1);
        assert_eq!(dwa.states.len(), 2);

        dwa.set_state_weight(0, SimpleBitset::from_item(10)).unwrap();
        dwa.set_final_weight(1, SimpleBitset::from_item(20)).unwrap();

        assert_eq!(dwa.states[0].weight, SimpleBitset::from_item(10));
        assert_eq!(dwa.states[1].final_weight, Some(SimpleBitset::from_item(20)));

        dwa.add_transition(0, 97, 1, SimpleBitset::from_item(30)).unwrap();
        assert_eq!(*dwa.states[0].transitions.get(97).unwrap(), 1);
        assert_eq!(
            *dwa.states[0].trans_weights_exceptions.get(&97).unwrap(),
            SimpleBitset::from_item(30)
        );

        // Test error cases
        let res = dwa.add_transition(0, 97, 1, SimpleBitset::zeros());
        assert!(matches!(res, Err(DWABuildError::TransitionAlreadyExists { from: 0, on: 97 })));

        dwa.set_default_transition(0, 0, SimpleBitset::from_item(40)).unwrap();
        assert_eq!(dwa.states[0].transitions.default, Some(0));
        assert_eq!(dwa.states[0].trans_weight_default, Some(SimpleBitset::from_item(40)));

        let res = dwa.set_default_transition(0, 0, SimpleBitset::zeros());
        assert!(matches!(res, Err(DWABuildError::DefaultTransitionAlreadyExists { from: 0 })));

        let res = dwa.set_final_weight(10, SimpleBitset::zeros());
        assert!(matches!(res, Err(DWABuildError::StateOutOfBounds { state: 10 })));
    }
}
