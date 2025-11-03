// src/weighted_automata.rs

#![allow(dead_code)]

use crate::json_serialization::{JSONConvertible, JSONNode};
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::iter::FromIterator;
use std::time::Instant;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Deref, Index, IndexMut};

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

// --- Part 2: U16Map ---

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct U16Map<T> {
    pub exceptions: BTreeMap<u16, T>,
    pub default: Option<T>,
}

impl<T> U16Map<T> {
    pub fn new() -> Self {
        Self { exceptions: BTreeMap::new(), default: None }
    }
    pub fn with_default(default_value: T) -> Self {
        Self { exceptions: BTreeMap::new(), default: Some(default_value) }
    }
    pub fn get(&self, key: u16) -> Option<&T> {
        self.exceptions.get(&key).or(self.default.as_ref())
    }
    pub fn iter_exceptions(&self) -> impl Iterator<Item = (&u16, &T)> {
        self.exceptions.iter()
    }
    pub fn get_default(&self) -> Option<&T> {
        self.default.as_ref()
    }
}

// --- Part 3 & 4: Automata Definitions ---

pub type StateID = usize;
pub type Weight = SimpleBitset;

// --- Nondeterministic Weighted Automaton (NWA) ---

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWAState {
    pub transitions: U16Map<Vec<(StateID, Weight)>>,
    pub epsilon_transitions: Vec<(StateID, Weight)>,
    pub final_weight: Option<Weight>,
}
impl NWAState {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWAStates(pub Vec<NWAState>);

impl Index<usize> for NWAStates {
    type Output = NWAState;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for NWAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl Deref for NWAStates {
    type Target = [NWAState];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// --- Part 5: NWA Processing, Determinization, and DWA Simplification ---

/// Helper for simplify_to_level: find existing merged state or create a new one.
fn get_or_create_merged_state(
    states: &mut NWAStates,
    comp_key: Vec<(StateID, Weight)>,
    depth: usize,
    comp2id: &mut HashMap<Vec<(StateID, Weight)>, StateID>,
    depth_of: &mut HashMap<StateID, usize>,
    work: &mut VecDeque<(Vec<(StateID, Weight)>, StateID, usize)>,
    merged_from: &mut BTreeMap<StateID, BTreeSet<StateID>>,
) -> StateID {
    if let Some(&id) = comp2id.get(&comp_key) {
        return id;
    }

    // Read from states before mutating, to satisfy borrow checker.
    let mut agg_final = Weight::zeros();
    let mut is_final = false;
    for (sid, path_w) in &comp_key {
        if let Some(fw) = &states[*sid].final_weight {
            let v = path_w & fw;
            if !v.is_empty() {
                is_final = true;
                agg_final |= &v;
            }
        }
    }

    // Now mutate: add the new state and initialize it.
    let id = states.add_state();
    states[id].epsilon_transitions.clear();
    states[id].transitions = U16Map::new();
    states[id].final_weight = if is_final { Some(agg_final) } else { None };

    // Update metadata and worklist.
    let mut set = BTreeSet::new();
    for (sid, _) in &comp_key {
        set.insert(*sid);
    }
    merged_from.insert(id, set);

    comp2id.insert(comp_key.clone(), id);
    depth_of.insert(id, depth);
    work.push_back((comp_key, id, depth));
    id
}

impl NWAStates {
    /// Nondeterministic processing over u16 alphabet.
    ///
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len();
        self.0.push(NWAState::new());
        id
    }
    pub fn add_transition(&mut self, from: StateID, on: u16, to: StateID, weight: Weight) {
        self[from].transitions.exceptions.entry(on).or_default().push((to, weight));
    }
    pub fn add_default_transition(&mut self, from: StateID, to: StateID, weight: Weight) {
        self[from].transitions.default.get_or_insert_with(Vec::new).push((to, weight));
    }
    pub fn add_epsilon_transition(&mut self, from: StateID, to: StateID, weight: Weight) {
        self[from].epsilon_transitions.push((to, weight));
    }
    pub fn set_final_weight(&mut self, state: StateID, weight: Weight) {
        self[state].final_weight = Some(weight);
    }

    /// Append a deep copy of `other` into `self`, returning `other_id -> new_id`.
    /// All transition targets (exceptions, default, epsilons) are remapped.
    pub fn append_copy_from(&mut self, other: &NWAStates) -> Vec<StateID> {
        let now = Instant::now();
        let base = self.0.len();
        let count = other.0.len();
        let mapping: Vec<StateID> = (0..count).map(|i| base + i).collect();

        self.0.extend((0..count).map(|_| NWAState::new()));

        for (i, st) in other.0.iter().enumerate() {
            let dst = &mut self[base + i];
            dst.final_weight = st.final_weight.clone();
            dst.epsilon_transitions =
                st.epsilon_transitions.iter().map(|(to, w)| (mapping[*to], w.clone())).collect();

            let mut new_map: U16Map<Vec<(StateID, Weight)>> = U16Map::new();
            for (ch, vec) in st.transitions.exceptions.iter() {
                let remapped = vec.iter().map(|(to, w)| (mapping[*to], w.clone())).collect();
                new_map.exceptions.insert(*ch, remapped);
            }
            if let Some(def) = st.transitions.default.as_ref() {
                let remapped = def.iter().map(|(to, w)| (mapping[*to], w.clone())).collect();
                new_map.default = Some(remapped);
            }
            dst.transitions = new_map;
        }
        println!("NWAStates::append_copy_from ({} states) took: {:?}", other.len(), now.elapsed());
        mapping
    }

    /// Reachability ignoring labels/weights.
    pub fn reachable_states_ignoring_labels(&self, from: StateID) -> BTreeSet<StateID> {
        let mut visited: BTreeSet<StateID> = BTreeSet::new();
        let mut q: VecDeque<StateID> = VecDeque::new();
        if from >= self.0.len() {
            return visited;
        }
        visited.insert(from);
        q.push_back(from);
        while let Some(u) = q.pop_front() {
            for (v, _) in &self[u].epsilon_transitions {
                if visited.insert(*v) {
                    q.push_back(*v);
                }
            }
            if let Some(def) = self[u].transitions.default.as_ref() {
                for (v, _) in def {
                    if visited.insert(*v) {
                        q.push_back(*v);
                    }
                }
            }
            for vec in self[u].transitions.exceptions.values() {
                for (v, _) in vec {
                    if visited.insert(*v) {
                        q.push_back(*v);
                    }
                }
            }
        }
        visited
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWABody {
    pub start_states: BTreeSet<StateID>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWA {
    pub states: NWAStates,
    pub body: NWABody,
}

impl NWA {
    pub fn new() -> Self {
        let mut states = NWAStates::default();
        let start = states.add_state();
        NWA { states, body: NWABody { start_states: BTreeSet::from([start]) } }
    }
    pub fn add_state(&mut self) -> StateID {
        self.states.add_state()
    }
    pub fn add_transition(&mut self, from: StateID, on: u16, to: StateID, weight: Weight) {
        self.states.add_transition(from, on, to, weight)
    }
    pub fn add_default_transition(&mut self, from: StateID, to: StateID, weight: Weight) {
        self.states.add_default_transition(from, to, weight)
    }
    pub fn add_epsilon_transition(&mut self, from: StateID, to: StateID, weight: Weight) {
        self.states.add_epsilon_transition(from, to, weight)
    }
    pub fn set_final_weight(&mut self, state: StateID, weight: Weight) {
        self.states.set_final_weight(state, weight)
    }
}

// --- Deterministic Weighted Automaton (DWA) ---

#[derive(Clone, Debug, Default)]
pub struct DWAState {
    pub transitions: U16Map<StateID>,
    pub weight: Weight,
    pub final_weight: Option<Weight>,
    pub trans_weight_default: Option<Weight>,
    pub trans_weights_exceptions: BTreeMap<u16, Weight>,
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

// --- Part 5: NWA Processing, Determinization, and DWA Simplification ---

impl NWAStates {
    /// Nondeterministic processing over u16 alphabet.
    ///
    /// Returns all stops as triples (pos, stop_state, path_weight), where path_weight
    /// is a meet (∩) along edges and joins (∪) when multiple paths converge.
    pub fn process_stack_u16_from_starts(&self, start_states: &BTreeSet<StateID>, input: &[u16]) -> Vec<(StateID, StateID, Weight)> {
        let now = Instant::now();
        let mut total_eps_closure_time = std::time::Duration::new(0, 0);
        let mut total_next_raw_time = std::time::Duration::new(0, 0);
        let mut max_current_states = 0;
        let mut max_next_raw_states = 0;

        if self.0.is_empty() {
            return Vec::new();
        }
        let has_eps = true;

        let mut current: BTreeMap<StateID, Weight> = BTreeMap::new();
        for &start_state in start_states {
            current.insert(start_state, Weight::all());
        }
        let eps_now = Instant::now();
        let mut current = self.epsilon_closure_with_flag(current, has_eps, input.len());
        total_eps_closure_time += eps_now.elapsed();

        // deduplicate results by (pos,state) and join weights
        let mut results: BTreeMap<(usize, StateID), Weight> = BTreeMap::new();
        let n = input.len();

        for pos in 0..=n {
            for (&sid, path_w) in &current {
                max_current_states = max_current_states.max(current.len());
                if self[sid].final_weight.is_some() {
                    results.entry((pos, sid)).or_insert_with(Weight::zeros).bitor_assign(path_w);
                }
            }
            if pos == n {
                for (&sid, path_w) in &current {
                    results.entry((pos, sid)).or_insert_with(Weight::zeros).bitor_assign(path_w);
                }
                break;
            }

            let ch = input[pos];
            let next_raw_now = Instant::now();
            let mut next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
            for (&sid, path_w) in &current {
                if let Some(transitions) = self[sid].transitions.get(ch) {
                    for (to, w) in transitions {
                        let w2 = path_w & w;
                        if !w2.is_empty() {
                            next_raw.entry(*to).or_insert_with(Weight::zeros).bitor_assign(&w2);
                        }
                    }
                }
            }
            total_next_raw_time += next_raw_now.elapsed();
            max_next_raw_states = max_next_raw_states.max(next_raw.len());

            let eps_now = Instant::now();
            current = self.epsilon_closure_with_flag(next_raw, has_eps, input.len());
            total_eps_closure_time += eps_now.elapsed();
        }

        let result = results.into_iter().map(|((pos, sid), w)| (pos, sid, w)).collect();
        println!(
            "NWAStates::process_stack_u16_from_starts (input len {}, total_states: {}) took: {:?}. eps_closure: {:?}, next_raw: {:?}, max_current: {}, max_next_raw: {}",
            input.len(),
            self.len(),
            now.elapsed(),
            total_eps_closure_time,
            total_next_raw_time,
            max_current_states,
            max_next_raw_states);
        result
    }

    /// Epsilon-closure: least fixed point of the monotone operator
    /// F(X)(v) = ⋃_{u∈X} (w(u) ∧ w(u→ε v))
    /// computed by worklist. Because join is idempotent and monotone,
    /// and weights only grow by union, termination is guaranteed.
    fn epsilon_closure_with_flag(
        &self,
        initial_states: BTreeMap<StateID, Weight>,
        has_epsilons: bool,
        input_len_for_debug: usize,
    ) -> BTreeMap<StateID, Weight> {
        if !has_epsilons {
            return initial_states;
        }

        let now = Instant::now();
        let mut worklist_pushes = 0;
        let mut edges_traversed = 0;

        let mut closure = initial_states.clone(); // TEMP: remove this when the println below is removed.
        let mut worklist: VecDeque<StateID> = closure.keys().cloned().collect();
        worklist_pushes += worklist.len();

        while let Some(u_id) = worklist.pop_front() {
            let u_weight = closure.get(&u_id).unwrap().clone();
            if u_weight.is_empty() {
                continue;
            }

            if self[u_id].epsilon_transitions.is_empty() {
                continue;
            }
            edges_traversed += self[u_id].epsilon_transitions.len();
            for (v_id, trans_weight) in &self[u_id].epsilon_transitions {
                let new_v_weight = &u_weight & trans_weight;
                if new_v_weight.is_empty() {
                    continue;
                }

                let current_v_weight = closure.entry(*v_id).or_insert_with(Weight::zeros);
                let old_len = current_v_weight.len();
                *current_v_weight |= &new_v_weight;

                if current_v_weight.len() > old_len {
                    worklist.push_back(*v_id);
                    worklist_pushes += 1;
                }
            }
        }

        let elapsed = now.elapsed();
        if elapsed.as_millis() > 10 {
            println!(
                "    epsilon_closure_with_flag (input_len: {}, initial_states: {}, total_states: {}) took: {:?}, worklist_pushes: {}, edges_traversed: {}",
                input_len_for_debug, initial_states.len(), self.len(), elapsed, worklist_pushes, edges_traversed
            );
        }
        closure
    }

    /// Create a depth-limited, DWA-like deterministic front-end inside the NWA itself.
    ///
    /// For the given `body` (other NWABody may reference these states; we do not modify them),
    /// we append new NWA states that represent weighted subset-compositions of original states,
    /// but only up to `level` input steps. Epsilon transitions are absorbed (no eps in new states).
    ///
    /// Properties:
    /// - For any input of length ≤ level, processing visits at most one new state per symbol
    ///   (i.e., no more than `level` transitions), akin to a DWA.
    /// - For longer inputs, transitions from the depth-`level` frontier route back to the
    ///   original NWA states (no further merging), so normal NWA processing continues.
    /// - We never overwrite existing states. New states are appended and only `body.start_states`
    ///   is changed to the new merged start state.
    /// - We minimize the number of new states by canonicalizing compositions
    ///   as Vec<(StateID, Weight)> and memoizing them globally across the whole depth-limited
    ///   construction. This is optimal for exact semantics under our Boolean-algebra weights.
    ///
    /// Returns: map from each new merged state ID to the set of original state IDs merged into it.
    pub fn simplify_to_level(&mut self, body: &mut NWABody, level: usize) -> BTreeMap<StateID, BTreeSet<StateID>> {
        let mut merged_from: BTreeMap<StateID, BTreeSet<StateID>> = BTreeMap::new();
        if level == 0 || self.0.is_empty() || body.start_states.is_empty() {
            return merged_from;
        }

        let has_epsilons = self.0.iter().any(|s| !s.epsilon_transitions.is_empty());

        // Canonicalize a composition map into a sorted vector, dropping empty weights.
        let to_key = |comp: BTreeMap<StateID, Weight>| -> Vec<(StateID, Weight)> {
            comp.into_iter().filter(|(_, w)| !w.is_empty()).collect()
        };

        // Start composition: starts with ALL weight and epsilon-closure.
        let mut start_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
        for &start_state in &body.start_states {
            start_raw.insert(start_state, Weight::all());
        }
        let start_map = self.epsilon_closure_with_flag(start_raw, has_epsilons, 0);
        let start_key = to_key(start_map);
        if start_key.is_empty() {
            // No reachable start under epsilon-closure; leave body unchanged.
            return merged_from;
        }

        // Global memo for compositions -> new state ID, and BFS worklist.
        let mut comp2id: HashMap<Vec<(StateID, Weight)>, StateID> = HashMap::new();
        let mut depth_of: HashMap<StateID, usize> = HashMap::new();
        let mut work: VecDeque<(Vec<(StateID, Weight)>, StateID, usize)> = VecDeque::new();
        let new_start_id = get_or_create_merged_state(
            self, start_key, 0, &mut comp2id, &mut depth_of, &mut work, &mut merged_from,
        );

        // Process merged states breadth-first up to given level.
        while let Some((comp, agg_id, depth)) = work.pop_front() {
            // Collect exception alphabet for this composition
            let mut critical_points: BTreeSet<u16> = BTreeSet::new();
            for (nwa_id, _) in &comp {
                for (&ch, _) in self[*nwa_id].transitions.exceptions.iter() {
                    critical_points.insert(ch);
                }
            }

            // Default transition (using only defaults from components)
            let mut def_next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
            let mut def_agg_weight = Weight::zeros();
            for (nwa_id, path_w) in &comp {
                if let Some(def) = self[*nwa_id].transitions.get_default() {
                    for (to, trans_w) in def {
                        let w = path_w & trans_w;
                        if !w.is_empty() {
                            def_next_raw.entry(*to).or_insert_with(Weight::zeros).bitor_assign(&w);
                            def_agg_weight |= &w;
                        }
                    }
                }
            }

            // Prepare default target for assignment and exception-comparison.
            let mut default_target_id: Option<StateID> = None;
            let mut default_vec_bottom: Option<Vec<(StateID, Weight)>> = None;
            if depth < level {
                let def_closure = self.epsilon_closure_with_flag(def_next_raw, has_epsilons, 0);
                let def_key = to_key(def_closure);
                if !def_key.is_empty() {
                    let tid = get_or_create_merged_state(
                        self, def_key, depth + 1, &mut comp2id, &mut depth_of, &mut work, &mut merged_from,
                    );
                    default_target_id = Some(tid);
                }
            } else {
                if !def_next_raw.is_empty() {
                    let mut vec: Vec<(StateID, Weight)> = def_next_raw.into_iter().collect();
                    vec.sort_by_key(|(to, _)| *to);
                    default_vec_bottom = Some(vec);
                }
            }

            // Build the transitions for this merged state.
            let mut new_transitions: U16Map<Vec<(StateID, Weight)>> = U16Map::new();
            if depth < level {
                if let Some(tid) = default_target_id {
                    if !def_agg_weight.is_empty() {
                        new_transitions.default = Some(vec![(tid, def_agg_weight.clone())]);
                    }
                }
            } else {
                if let Some(v) = default_vec_bottom.clone() {
                    new_transitions.default = Some(v);
                }
            }

            // Exception transitions for critical points
            for ch in critical_points {
                let mut exc_next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
                let mut exc_agg_weight = Weight::zeros();
                for (nwa_id, path_w) in &comp {
                    if let Some(transitions) = self[*nwa_id].transitions.get(ch) {
                        for (to, trans_w) in transitions {
                            let w = path_w & trans_w;
                            if !w.is_empty() {
                                exc_next_raw.entry(*to).or_insert_with(Weight::zeros).bitor_assign(&w);
                                exc_agg_weight |= &w;
                            }
                        }
                    }
                }
                if exc_next_raw.is_empty() {
                    continue;
                }

                if depth < level {
                    let exc_closure = self.epsilon_closure_with_flag(exc_next_raw, has_epsilons, 0);
                    let exc_key = to_key(exc_closure);
                    if exc_key.is_empty() {
                        continue;
                    }
                    let tid = get_or_create_merged_state(
                        self, exc_key, depth + 1, &mut comp2id, &mut depth_of, &mut work, &mut merged_from,
                    );
                    let exc_vec = vec![(tid, exc_agg_weight.clone())];

                    // Deduplicate vs default if identical (target and weight are equal).
                    let same_as_default = if let Some(def_tid) = default_target_id {
                        if let Some(def_vec) = &new_transitions.default {
                            def_tid == tid && def_vec[0].1 == exc_agg_weight
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if !same_as_default {
                        new_transitions.exceptions.insert(ch, exc_vec);
                    }
                } else {
                    let mut vec: Vec<(StateID, Weight)> = exc_next_raw.into_iter().collect();
                    vec.sort_by_key(|(to, _)| *to);

                    let same_as_default = match &default_vec_bottom {
                        Some(d) => d == &vec,
                        None => false,
                    };
                    if !same_as_default {
                        new_transitions.exceptions.insert(ch, vec);
                    }
                }
            }

            // Assign the computed transitions to the merged state; ensure no epsilons.
            {
                let st = &mut self[agg_id];
                st.transitions = new_transitions;
                st.epsilon_transitions.clear();
            }
        }

        // Rewire this body's starts to the new merged start. Other bodies remain untouched.
        body.start_states = BTreeSet::from([new_start_id]);
        merged_from
    }
}

impl NWA {
    /// Determinization via weighted subset construction on the Boolean algebra (P(usize), ∪, ∩).
    ///
    /// For a set of NWA states S with path-weights {w_s}, the DWA state's:
    /// - state.weight = ⋃_{s∈S} w_s,
    /// - state.final_weight = ⋃_{s∈S} (w_s ∩ final_w(s)),
    /// - on character a, it transitions to closure(T) where
    ///   T collects (t, ⋃_{s∈S} w_s ∩ w(s --a--> t)).
    ///
    /// Soundness: meet distributes over join, so path-weights are correctly propagated and
    /// nondeterminism is collapsed by join. Completeness follows from standard subset construction.
    pub fn determinize(&self) -> DWA {
        Self::determinize_components(&self.states, &self.body)
    }

    pub fn determinize_components(states: &NWAStates, body: &NWABody) -> DWA {
        let now = Instant::now();
        let mut dwa_states = DWAStates::default();
        let mut dwa_body = DWABody::default();

        if states.0.is_empty() {
            return DWA { states: dwa_states, body: dwa_body };
        }

        let has_epsilons = states.0.iter().any(|s| !s.epsilon_transitions.is_empty());

        let to_key = |comp: BTreeMap<StateID, Weight>| -> Vec<(StateID, Weight)> {
            comp.into_iter().filter(|(_, w)| !w.is_empty()).collect()
        };

        let mut known_states: HashMap<Vec<(StateID, Weight)>, StateID> = HashMap::new();
        let mut worklist: VecDeque<Vec<(StateID, Weight)>> = VecDeque::new();

        let mut get_or_create = |comp_key: Vec<(StateID, Weight)>,
                                 dwa_states: &mut DWAStates,
                                 known: &mut HashMap<Vec<(StateID, Weight)>, StateID>,
                                 work: &mut VecDeque<Vec<(StateID, Weight)>>|
         -> Option<StateID> {
            if comp_key.is_empty() {
                return None;
            }
            if let Some(&id) = known.get(&comp_key) {
                return Some(id);
            }
            let new_id = dwa_states.0.len();
            dwa_states.0.push(DWAState::default());
            known.insert(comp_key.clone(), new_id);
            work.push_back(comp_key);
            Some(new_id)
        };

        let mut start_raw = BTreeMap::new();
        for &start_state in &body.start_states {
            start_raw.insert(start_state, Weight::all());
        }
        let start_map = states.epsilon_closure_with_flag(start_raw, has_epsilons, 0);
        if let Some(start_id) = get_or_create(to_key(start_map), &mut dwa_states, &mut known_states, &mut worklist) {
            dwa_body.start_state = start_id;
        } else {
            return DWA { states: dwa_states, body: dwa_body };
        }

        while let Some(current_composition) = worklist.pop_front() {
            let current_dwa_id = *known_states.get(&current_composition).unwrap();

            // Aggregate state and final weights; collect exception alphabet.
            let mut aggregate_weight = Weight::zeros();
            let mut aggregate_final_weight = Weight::zeros();
            let mut is_final = false;
            let mut critical_points = BTreeSet::new();

            for (nwa_id, path_weight) in &current_composition {
                aggregate_weight |= path_weight;
                if let Some(final_w) = &states[*nwa_id].final_weight {
                    is_final = true;
                    aggregate_final_weight |= &(path_weight & final_w);
                }
                for &char_code in states[*nwa_id].transitions.exceptions.keys() {
                    critical_points.insert(char_code);
                }
            }
            dwa_states[current_dwa_id].weight = aggregate_weight;
            if is_final {
                dwa_states[current_dwa_id].final_weight = Some(aggregate_final_weight);
            }

            // Default transition
            let mut default_next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
            let mut default_weight_agg = Weight::zeros();
            for (nwa_id, path_weight) in &current_composition {
                if let Some(transitions) = states[*nwa_id].transitions.get_default() {
                    for (next_nwa_id, trans_weight) in transitions {
                        let w = path_weight & trans_weight;
                        default_next_raw.entry(*next_nwa_id).or_default().bitor_assign(&w);
                        default_weight_agg |= &w;
                    }
                }
            }
            let def_comp = to_key(states.epsilon_closure_with_flag(default_next_raw, has_epsilons, 0));
            let def_target = get_or_create(def_comp, &mut dwa_states, &mut known_states, &mut worklist);
            dwa_states[current_dwa_id].transitions.default = def_target;
            if def_target.is_some() {
                dwa_states[current_dwa_id].trans_weight_default = Some(default_weight_agg);
            }

            // Exception transitions
            for char_code in critical_points {
                let mut exception_next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
                let mut exception_weight_agg = Weight::zeros();
                for (nwa_id, path_weight) in &current_composition {
                    if let Some(transitions) = states[*nwa_id].transitions.get(char_code) {
                        for (next_nwa_id, trans_weight) in transitions {
                            let w = path_weight & trans_weight;
                            exception_next_raw.entry(*next_nwa_id).or_default().bitor_assign(&w);
                            exception_weight_agg |= &w;
                        }
                    }
                }
                if exception_next_raw.is_empty() {
                    continue;
                }
                let exc_comp = to_key(states.epsilon_closure_with_flag(exception_next_raw, has_epsilons, 0));
                let exc_target = get_or_create(exc_comp, &mut dwa_states, &mut known_states, &mut worklist);

                if exc_target != def_target {
                    if let Some(tid) = exc_target {
                        dwa_states[current_dwa_id].transitions.exceptions.insert(char_code, tid);
                        dwa_states[current_dwa_id]
                            .trans_weights_exceptions
                            .insert(char_code, exception_weight_agg);
                    }
                }
            }
        }

        let result = DWA { states: dwa_states, body: dwa_body };
        println!("NWA::determinize_components ({} NWA states -> {} DWA states) took: {:?}", states.len(), result.states.len(), now.elapsed());
        result
    }

    pub fn process_stack_u16(&self, input: &[u16]) -> Vec<(StateID, StateID, Weight)> {
        self.states.process_stack_u16_from_starts(&self.body.start_states, input)
    }
}

// DWA simplification utilities
impl DWA {
    /// Simplify by iterating a small pipeline until stable (or pass limit):
    /// - normalize redundant edges,
    /// - partition-refinement minimization,
    /// - prune unreachable.
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
        println!("DWA::simplify_components ({} states -> {} states) took: {:?}", initial_len, states.len(), now.elapsed());
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
            st.trans_weights_exceptions.retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            changed |= st.trans_weights_exceptions.len() != before_w;
        }
        changed
    }

    /// Partition-refinement minimization. Two states are equivalent iff they share:
    /// - weight and final_weight,
    /// - default target's partition,
    /// - and exception targets per character (up to default-equivalence).
    ///
    /// This is a language-preserving quotient under a bisimulation-like signature.
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
            let mut sig2pid: HashMap<(Weight, Option<Weight>, usize, Vec<(u16, usize)>), usize> = HashMap::new();

            for i in 0..n {
                let st = &states[i];
                let def_cls = st.transitions.default.map(|d| part[d]).unwrap_or(sink_pid);
                let mut ex: Vec<(u16, usize)> = Vec::with_capacity(st.transitions.exceptions.len());
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
            let ex_keys: Vec<u16> = st.transitions.exceptions.keys().cloned().collect();
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

    /// Remove states unreachable from `start_state` and renumber them densely.
    pub fn prune_unreachable(states: &mut DWAStates, body: &mut DWABody) -> bool {
        if states.0.is_empty() {
            return false;
        }
        let n = states.0.len();
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        visited[body.start_state] = true;
        q.push_back(body.start_state);

        while let Some(u) = q.pop_front() {
            if let Some(d) = states[u].transitions.default {
                if !visited[d] {
                    visited[d] = true;
                    q.push_back(d);
                }
            }
            for &v in states[u].transitions.exceptions.values() {
                if !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }
        if visited.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut next_id = 0usize;
        for i in 0..n {
            if visited[i] {
                map[i] = next_id;
                next_id += 1;
            }
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

impl Display for NWAStates {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (id, state) in self.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(w) = &state.final_weight {
                writeln!(f, "    final_weight: {}", w)?;
            }
            for (to, weight) in &state.epsilon_transitions {
                writeln!(f, "    ε -> {} (weight: {})", to, weight)?;
            }
            if let Some(default) = &state.transitions.default {
                for (to, weight) in default {
                    writeln!(f, "    * -> {} (weight: {})", to, weight)?;
                }
            }
            for (on, transitions) in &state.transitions.exceptions {
                for (to, weight) in transitions {
                    let char_repr = if let Some(c) = char::from_u32(*on as u32) {
                        if c.is_ascii_graphic() || c == ' ' {
                            format!("'{}'", c)
                        } else {
                            format!("{}", *on)
                        }
                    } else {
                        format!("{}", *on)
                    };
                    writeln!(f, "    {} -> {} (weight: {})", char_repr, to, weight)?;
                }
            }
        }
        Ok(())
    }
}

impl Display for NWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWA (starts: {:?})", self.body.start_states)?;
        write!(f, "{}", self.states)
    }
}

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
                let char_repr = if let Some(c) = char::from_u32(*on as u32) {
                    if c.is_ascii_graphic() || c == ' ' {
                        format!("'{}'", c)
                    } else {
                        format!("{}", *on)
                    }
                } else {
                    format!("{}", *on)
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

impl<T: JSONConvertible> JSONConvertible for U16Map<T> {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("exceptions".to_string(), self.exceptions.to_json());
        obj.insert("default".to_string(), self.default.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let exceptions =
            BTreeMap::<u16, T>::from_json(obj.remove("exceptions").ok_or("Missing 'exceptions' field")?)?;
        let default = Option::<T>::from_json(obj.remove("default").ok_or("Missing 'default' field")?)?;
        Ok(U16Map { exceptions, default })
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
            U16Map::<StateID>::from_json(obj.remove("transitions").ok_or("Missing 'transitions' field")?)?;
        let weight = Weight::from_json(obj.remove("weight").ok_or("Missing 'weight' field")?)?;
        let final_weight =
            Option::<Weight>::from_json(obj.remove("final_weight").ok_or("Missing 'final_weight' field")?)?;
        let trans_weight_default = Option::<Weight>::from_json(
            obj.remove("trans_weight_default").ok_or("Missing 'trans_weight_default' field")?,
        )?;
        let trans_weights_exceptions = BTreeMap::<u16, Weight>::from_json(
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

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    fn build_complex_nwa() -> NWA {
        let mut nwa = NWA::default();
        for _ in 0..20 {
            nwa.add_state();
        }
        nwa.body.start_states = BTreeSet::from([1]);

        let w0 = SimpleBitset::from_item(0);
        let w1 = SimpleBitset::from_item(1);
        let w2 = SimpleBitset::from_item(2);
        let w3 = SimpleBitset::from_item(3);
        let w123 = SimpleBitset::from_iter(1..=3);
        let wall = SimpleBitset::all();

        nwa.set_final_weight(7, wall.clone());

        nwa.add_transition(1, 1, 2, w0.clone());
        nwa.add_transition(1, 2, 7, w1.clone());
        nwa.add_transition(1, 2, 7, w2.clone());
        nwa.add_transition(1, 2, 3, w3.clone());
        nwa.add_transition(1, 4, 4, w0.clone());
        nwa.add_transition(1, 5, 5, w123.clone());
        nwa.add_transition(1, 6, 6, w123.clone());
        nwa.add_transition(1, 7, 7, w0.clone());
        nwa.add_transition(1, 10, 7, w1.clone());
        nwa.add_transition(1, 10, 7, w2.clone());
        nwa.add_transition(1, 10, 7, w3.clone());
        nwa.add_default_transition(2, 8, wall.clone());
        nwa.add_transition(3, 10, 7, w3.clone());
        nwa.add_default_transition(4, 9, wall.clone());
        nwa.add_default_transition(5, 11, wall.clone());
        nwa.add_default_transition(5, 12, wall.clone());
        nwa.add_default_transition(5, 13, wall.clone());
        nwa.add_default_transition(6, 14, wall.clone());
        nwa.add_default_transition(6, 16, wall.clone());
        nwa.add_default_transition(6, 18, wall.clone());
        nwa.add_transition(8, 10, 7, w0.clone());
        nwa.add_default_transition(9, 10, wall.clone());
        nwa.add_transition(10, 10, 7, w0.clone());
        nwa.add_transition(11, 10, 7, w1.clone());
        nwa.add_transition(12, 10, 7, w2.clone());
        nwa.add_transition(13, 10, 7, w3.clone());
        nwa.add_default_transition(14, 15, wall.clone());
        nwa.add_transition(15, 10, 7, w1.clone());
        nwa.add_default_transition(16, 17, wall.clone());
        nwa.add_transition(17, 10, 7, w2.clone());
        nwa.add_default_transition(18, 19, wall.clone());
        nwa.add_transition(19, 10, 7, w3.clone());

        nwa
    }

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
    fn test_u16_map() {
        let mut map = U16Map::with_default(100);
        map.exceptions.insert(b'a' as u16, 10);
        map.exceptions.insert(b'b' as u16, 20);

        assert_eq!(map.get(b'a' as u16), Some(&10));
        assert_eq!(map.get(b'c' as u16), Some(&100));
        assert_eq!(map.get(b'b' as u16), Some(&20));
    }

    #[test]
    fn test_determinize_simple_nwa() {
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(0, b'a' as u16, s1, SimpleBitset::from_iter(vec![1, 4]));
        nwa.add_transition(s1, b'b' as u16, s2, SimpleBitset::from_iter(vec![1, 5]));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![1, 6]));

        let dwa = nwa.determinize();
        crate::debug!(5, "NWA:\n{}", nwa);
        crate::debug!(5, "DWA:\n{}", dwa);

        assert_eq!(dwa.states.len(), 3);
        let start_state = &dwa.states[dwa.body.start_state];
        assert_eq!(start_state.weight, SimpleBitset::all());
        assert!(start_state.final_weight.is_none());

        let s1_id = *start_state.transitions.get(b'a' as u16).unwrap();
        let state1 = &dwa.states[s1_id];
        assert_eq!(state1.weight, SimpleBitset::from_iter(vec![1, 4]));
        assert!(state1.final_weight.is_none());

        let s2_id = *state1.transitions.get(b'b' as u16).unwrap();
        let state2 = &dwa.states[s2_id];
        let expected_s2_weight = SimpleBitset::from_iter(vec![1, 4]) & SimpleBitset::from_iter(vec![1, 5]);
        assert_eq!(state2.weight, expected_s2_weight);
        let expected_final_weight = &expected_s2_weight & &SimpleBitset::from_iter(vec![1, 6]);
        assert_eq!(state2.final_weight, Some(expected_final_weight));
    }

    #[test]
    fn test_determinize_with_epsilon() {
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_epsilon_transition(0, s1, SimpleBitset::from_iter(vec![1, 5]));
        nwa.add_transition(s1, b'a' as u16, s2, SimpleBitset::from_iter(vec![1, 6]));
        nwa.set_final_weight(0, SimpleBitset::from_item(4));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![1, 7]));

        let dwa = nwa.determinize();
        assert_eq!(dwa.states.len(), 2);

        let start_state = &dwa.states[dwa.body.start_state];
        assert_eq!(start_state.weight, SimpleBitset::from_iter(vec![1, 5]) | SimpleBitset::all());
        assert_eq!(start_state.final_weight, Some(SimpleBitset::from_item(4)));

        let s1_id = *start_state.transitions.get(b'a' as u16).unwrap();
        let state1 = &dwa.states[s1_id];
        let expected_s1_weight = &SimpleBitset::from_iter(vec![1, 5]) & &SimpleBitset::from_iter(vec![1, 6]);
        assert_eq!(state1.weight, expected_s1_weight);
        let expected_final = &expected_s1_weight & &SimpleBitset::from_iter(vec![1, 7]);
        assert_eq!(state1.final_weight, Some(expected_final));
    }

    #[test]
    fn test_determinize_with_default() {
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(0, b'a' as u16, s1, SimpleBitset::from_iter(vec![1, 5]));
        nwa.add_default_transition(0, s2, SimpleBitset::from_iter(vec![2, 5]));
        nwa.set_final_weight(s1, SimpleBitset::from_iter(vec![1, 10]));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![2, 20]));

        let dwa = nwa.determinize();
        assert_eq!(dwa.states.len(), 3);

        let start_state = &dwa.states[dwa.body.start_state];
        let s1_id = *start_state.transitions.exceptions.get(&(b'a' as u16)).unwrap();
        let s2_id = start_state.transitions.default.unwrap();
        assert_ne!(s1_id, s2_id);

        let state1 = &dwa.states[s1_id];
        assert_eq!(state1.weight, SimpleBitset::from_iter(vec![1, 5]));
        assert_eq!(state1.final_weight, Some(SimpleBitset::from_iter(vec![1, 5]) & SimpleBitset::from_iter(vec![1, 10])));

        let state2 = &dwa.states[s2_id];
        assert_eq!(state2.weight, SimpleBitset::from_iter(vec![2, 5]));
        assert_eq!(state2.final_weight, Some(SimpleBitset::from_iter(vec![2, 5]) & SimpleBitset::from_iter(vec![2, 20])));
    }

    #[test]
    fn test_determinize_nondeterministic_choice() {
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(0, b'a' as u16, s1, SimpleBitset::from_iter(vec![1, 5]));
        nwa.add_transition(0, b'a' as u16, s2, SimpleBitset::from_iter(vec![2, 5]));
        nwa.set_final_weight(s1, SimpleBitset::from_iter(vec![1, 10]));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![2, 20]));

        let dwa = nwa.determinize();
        assert_eq!(dwa.states.len(), 2);

        let start_state = &dwa.states[dwa.body.start_state];
        let s1_id = *start_state.transitions.get(b'a' as u16).unwrap();
        let state1 = &dwa.states[s1_id];

        let expected_weight = SimpleBitset::from_iter(vec![1, 5]) | SimpleBitset::from_iter(vec![2, 5]);
        assert_eq!(state1.weight, expected_weight);

        let final1 = SimpleBitset::from_iter(vec![1, 5]) & SimpleBitset::from_iter(vec![1, 10]);
        let final2 = SimpleBitset::from_iter(vec![2, 5]) & SimpleBitset::from_iter(vec![2, 20]);
        let expected_final = final1 | &final2;
        assert_eq!(state1.final_weight, Some(expected_final));
    }

    #[test]
    fn test_determinize_complex_nondeterminism() {
        let mut nwa = NWA::default();
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        let s3 = nwa.add_state();
        nwa.body.start_states = BTreeSet::from([s1]);

        let w01 = SimpleBitset::from_iter(0..=1);
        let w0 = SimpleBitset::from_item(0);
        let w1 = SimpleBitset::from_item(1);
        nwa.set_final_weight(s3, SimpleBitset::all());

        nwa.add_transition(s1, 0, s2, w01.clone());
        nwa.add_transition(s1, 1, s3, w0.clone());
        nwa.add_transition(s1, 1, s2, w01.clone());
        nwa.add_transition(s1, 2, s2, w01.clone());
        nwa.add_transition(s1, 3, s2, w01.clone());
        nwa.add_transition(s1, 4, s2, w01.clone());
        nwa.add_transition(s1, 4, s3, w1.clone());
        nwa.add_transition(s1, 5, s2, w01.clone());

        let dwa = nwa.determinize();
        crate::debug!(5, "{}", nwa);
        crate::debug!(5, "{}", dwa);

        assert_eq!(dwa.states.len(), 4);
        let s0_dwa = &dwa.states[dwa.body.start_state];
        let w01_expected = SimpleBitset::from_iter(0..=1);

        assert_eq!(s0_dwa.weight, SimpleBitset::all());
        assert!(s0_dwa.final_weight.is_none());

        let s1_dwa_id = *s0_dwa.transitions.get(0).unwrap();
        let s1_dwa = &dwa.states[s1_dwa_id];
        assert_eq!(s1_dwa.weight, w01_expected);
        assert!(s1_dwa.final_weight.is_none());

        let s2_dwa_id = *s0_dwa.transitions.get(1).unwrap();
        let s2_dwa = &dwa.states[s2_dwa_id];
        assert_eq!(s2_dwa.weight, w01_expected);
        assert_eq!(s2_dwa.final_weight, Some(w0));

        let s3_dwa_id = *s0_dwa.transitions.get(4).unwrap();
        let s3_dwa = &dwa.states[s3_dwa_id];
        assert_eq!(s3_dwa.weight, w01_expected);
        assert_eq!(s3_dwa.final_weight, Some(w1));

        assert_eq!(*s0_dwa.transitions.get(2).unwrap(), s1_dwa_id);
        assert_eq!(*s0_dwa.transitions.get(3).unwrap(), s1_dwa_id);
        assert_eq!(*s0_dwa.transitions.get(5).unwrap(), s1_dwa_id);
    }

    #[test]
    fn test_determinize_complex_nwa_from_input() {
        let nwa = build_complex_nwa();
        let dwa = nwa.determinize();

        assert_eq!(dwa.states.len(), 15);

        let w0 = SimpleBitset::from_item(0);
        let w3 = SimpleBitset::from_item(3);
        let w12 = SimpleBitset::from_iter(1..=2);
        let w123 = SimpleBitset::from_iter(1..=3);
        let wall = SimpleBitset::all();

        let s0 = &dwa.states[dwa.body.start_state];
        assert_eq!(s0.weight, wall);
        assert!(s0.final_weight.is_none());

        let s1_id = *s0.transitions.get(1).unwrap();
        assert_eq!(s1_id, 1);
        let s1 = &dwa.states[s1_id];
        assert_eq!(s1.weight, w0.clone());
        assert!(s1.final_weight.is_none());

        let s2_id = *s0.transitions.get(2).unwrap();
        assert_eq!(s2_id, 2);
        let s2 = &dwa.states[s2_id];
        assert_eq!(s2.weight, w123.clone());
        assert_eq!(s2.final_weight, Some(w12.clone()));

        let s7_id = *s0.transitions.get(10).unwrap();
        assert_eq!(s7_id, 7);
        let s7 = &dwa.states[s7_id];
        assert_eq!(s7.weight, w123.clone());
        assert_eq!(s7.final_weight, Some(w123.clone()));

        let s9_id = *s2.transitions.get(10).unwrap();
        assert_eq!(s9_id, 9);
        let s9 = &dwa.states[s9_id];
        assert_eq!(s9.weight, w3.clone());
        assert_eq!(s9.final_weight, Some(w3.clone()));

        let s6_id = *s0.transitions.get(7).unwrap();
        assert_eq!(s6_id, 6);
        let s6 = &dwa.states[s6_id];
        assert_eq!(s6.weight, w0.clone());
        assert_eq!(s6.final_weight, Some(w0.clone()));

        let s8_id = s1.transitions.default.unwrap();
        assert_eq!(s8_id, 8);
        let s8 = &dwa.states[s8_id];
        assert_eq!(s8.weight, w0.clone());
        assert!(s8.final_weight.is_none());

        assert_eq!(*s8.transitions.get(10).unwrap(), s6_id);

        assert!(s7.transitions.exceptions.is_empty());
        assert!(s7.transitions.default.is_none());
    }
}
