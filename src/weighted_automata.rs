// src/weighted_automata.rs

#![allow(dead_code)] // Allow unused code for this library module example

use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, BTreeSet, VecDeque, HashMap};
use std::fmt::{Debug, Display, Formatter};
use std::iter::FromIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};
use crate::json_serialization::{JSONConvertible, JSONNode};
// --- Part 1: SimpleBitset ---

/// A simple wrapper around `RangeSetBlaze<usize>` for representing sets of numbers as weights.
///
/// This version is a straightforward, owned data structure without the complexities of
/// interning or caching found in `HybridBitset`. It's suitable for tracking sets of
/// properties, IDs, or other numerical data through automaton transitions.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct SimpleBitset(pub RangeSetBlaze<usize>);

impl SimpleBitset {
    /// Creates a new, empty bitset (the "zero" weight).
    pub fn zeros() -> Self {
        SimpleBitset(RangeSetBlaze::new())
    }

    /// Creates a new bitset containing all possible `usize` values (the "unit" or "identity" weight).
    pub fn all() -> Self {
        SimpleBitset(RangeSetBlaze::from_iter([0..=usize::MAX]))
    }

    /// Creates a bitset from a single item.
    pub fn from_item(item: usize) -> Self {
        SimpleBitset(RangeSetBlaze::from_iter([item]))
    }

    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        SimpleBitset(rsb)
    }

    /// Returns the number of elements in the set. Saturates at `usize::MAX`.
    pub fn len(&self) -> usize {
        self.0.len().try_into().unwrap_or(usize::MAX)
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Checks if a specific number is present in the set.
    pub fn contains(&self, index: usize) -> bool {
        self.0.contains(index)
    }

    /// Returns an iterator over the numbers in the set, up to a given maximum.
    /// This is to avoid accidentally iterating over a huge set like `0..=usize::MAX`.
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

impl BitAndAssign<&SimpleBitset> for SimpleBitset {
    fn bitand_assign(&mut self, rhs: &SimpleBitset) {
        // range-set-blaze does not implement BitAndAssign, so we emulate it.
        self.0 = &self.0 & &rhs.0;
    }
}

impl BitOrAssign<&SimpleBitset> for SimpleBitset {
    fn bitor_assign(&mut self, rhs: &SimpleBitset) {
        self.0 |= &rhs.0;
    }
}

// --- Bitwise Operations (for owned and mixed owned/borrowed values) ---
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

/// A map for `u16` keys, generic over the value type `T`.
///
/// It supports a `default` value for all keys not explicitly present as an `exception`.
/// This is efficient for representing transitions where most characters lead to a
/// common state, with only a few special cases.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct U16Map<T> {
    pub exceptions: BTreeMap<u16, T>,
    pub default: Option<T>,
}

impl<T> U16Map<T> {
    /// Creates a new, empty map with no default value.
    pub fn new() -> Self {
        Self {
            exceptions: BTreeMap::new(),
            default: None,
        }
    }

    /// Creates a new map with a specified default value.
    pub fn with_default(default_value: T) -> Self {
        Self {
            exceptions: BTreeMap::new(),
            default: Some(default_value),
        }
    }

    /// Gets the value for a given key, falling back to the default if no exception exists.
    pub fn get(&self, key: u16) -> Option<&T> {
        self.exceptions.get(&key).or(self.default.as_ref())
    }

    /// Returns an iterator over the explicit `(key, value)` exceptions.
    pub fn iter_exceptions(&self) -> impl Iterator<Item = (&u16, &T)> {
        self.exceptions.iter()
    }

    /// Returns a reference to the default value, if it exists.
    pub fn get_default(&self) -> Option<&T> {
        self.default.as_ref()
    }
}

// --- Part 3 & 4: Automata Definitions ---

pub type StateID = usize;
pub type Weight = SimpleBitset;

// --- Nondeterministic Weighted Automaton (NWA) ---

#[derive(Clone, Debug, Default)]
pub struct NWAState {
    /// Transitions on specific `u16` characters. The `Vec` handles nondeterminism.
    pub transitions: U16Map<Vec<(StateID, Weight)>>,
    /// Epsilon transitions that can be taken without consuming a character.
    pub epsilon_transitions: Vec<(StateID, Weight)>,
    /// The weight associated with this state if it's a final/accepting state.
    pub final_weight: Option<Weight>,
}

impl NWAState {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct NWA {
    pub states: Vec<NWAState>,
    pub start_state: StateID,
}

impl NWA {
    /// Creates a new NWA with a single start state.
    pub fn new() -> Self {
        let mut nwa = Self::default();
        nwa.add_state(); // Add start state 0
        nwa
    }

    /// Adds a new, empty state and returns its ID.
    pub fn add_state(&mut self) -> StateID {
        let id = self.states.len();
        self.states.push(NWAState::new());
        id
    }

    /// Adds a transition for a specific character.
    pub fn add_transition(&mut self, from: StateID, on: u16, to: StateID, weight: Weight) {
        self.states[from]
            .transitions
            .exceptions
            .entry(on)
            .or_default()
            .push((to, weight));
    }

    /// Adds a default transition for all characters without an explicit exception.
    pub fn add_default_transition(&mut self, from: StateID, to: StateID, weight: Weight) {
        let default_trans = self.states[from].transitions.default.get_or_insert_with(Vec::new);
        default_trans.push((to, weight));
    }

    /// Adds an epsilon transition.
    pub fn add_epsilon_transition(&mut self, from: StateID, to: StateID, weight: Weight) {
        self.states[from]
            .epsilon_transitions
            .push((to, weight));
    }

    /// Sets the final weight for a state, marking it as an accepting state.
    pub fn set_final_weight(&mut self, state: StateID, weight: Weight) {
        self.states[state].final_weight = Some(weight);
    }
}

// --- Deterministic Weighted Automaton (DWA) ---

#[derive(Clone, Debug, Default)]
pub struct DWAState {
    /// Deterministic transitions: one character leads to at most one state.
    pub transitions: U16Map<StateID>,
    /// The aggregated weight of being in this state.
    pub weight: Weight,
    /// The aggregated final weight of this state.
    pub final_weight: Option<Weight>,
    /// Aggregate weight for the default transition (if any).
    /// This captures the union of (path_weight & trans_weight) across NWA transitions that map to the default.
    pub trans_weight_default: Option<Weight>,
    /// Aggregate weights for exception transitions.
    /// For each exception character, stores the union of (path_weight & trans_weight) that form that deterministic edge.
    /// This is useful when reconstructing per-edge weights after determinization.
    pub trans_weights_exceptions: BTreeMap<u16, Weight>,
}

#[derive(Clone, Debug, Default)]
pub struct DWA {
    pub states: Vec<DWAState>,
    pub start_state: StateID,
}

impl DWAState {
    /// Returns Some(target) if this state is 'simple':
    /// - it has only a default transition (no exceptions),
    /// - it has no final_weight,
    /// - and the weight for that default transition equals the state's weight.
    /// Otherwise returns None.
    pub fn simple_default_target(&self) -> Option<StateID> {
        if self.final_weight.is_none()
            && self.transitions.exceptions.is_empty()
        {
            if let (Some(target), Some(w)) = (self.transitions.default, self.trans_weight_default.as_ref()) {
                if &self.weight == w {
                    return Some(target);
                }
            }
        }
        None
    }
}

// --- Part 5: Determinization ---

impl NWA {
    /// Determinizes the Nondeterministic Weighted Automaton (NWA) into a
    /// Deterministic Weighted Automaton (DWA) using a weighted subset construction algorithm.
    ///
    /// The algorithm works as follows:
    /// 1. Each DWA state corresponds to a set of NWA states, where each NWA state has an
    ///    associated `Weight`. This weight represents the combined weight of all paths
    ///    from the start state to that NWA state.
    /// 2. Path weights are combined with `BitAnd` (&) along a sequence of transitions.
    /// 3. When multiple paths converge (nondeterminism), their weights are combined with `BitOr` (|).
    /// 4. Epsilon transitions are handled by computing an "epsilon closure" to find all
    ///    reachable states without consuming input.
    /// 5. The alphabet is partitioned based on character exceptions in the NWA states to
    ///    build the DWA's transition map.
    /// The alphabet is partitioned based on character exceptions in the NWA states to
    /// build the DWA's transition map.
    pub fn determinize(&self) -> DWA {
        let mut dwa = DWA::default();
        if self.states.is_empty() {
            return dwa;
        }

        // Fast-path: check once if the NWA has any epsilon transitions.
        let has_epsilons = self
            .states
            .iter()
            .any(|s| !s.epsilon_transitions.is_empty());

        // Use a compact canonical composition key: Vec<(StateID, Weight)> sorted by StateID.
        // This drastically reduces memory and makes hashing/lookup much faster than nested maps.
        let to_key = |comp: BTreeMap<StateID, Weight>| -> Vec<(StateID, Weight)> {
            // BTreeMap iteration is sorted by key, so the resulting Vec is sorted too.
            comp.into_iter().filter(|(_, w)| !w.is_empty()).collect()
        };

        // Map from a DWA state's composition key to its StateID in the new DWA.
        let mut known_states: HashMap<Vec<(StateID, Weight)>, StateID> = HashMap::new();
        // Worklist holds canonical composition keys.
        let mut worklist: VecDeque<Vec<(StateID, Weight)>> = VecDeque::new();

        // Helper function to create a new DWA state if it's not already known.
        let mut get_or_create_dwa_state =
            |comp_key: Vec<(StateID, Weight)>,
             dwa: &mut DWA,
             known: &mut HashMap<Vec<(StateID, Weight)>, StateID>,
             work: &mut VecDeque<Vec<(StateID, Weight)>>|
             -> Option<StateID> {
                if comp_key.is_empty() {
                    return None;
                }
                if let Some(&id) = known.get(&comp_key) {
                    return Some(id);
                }
                let new_id = dwa.states.len();
                dwa.states.push(DWAState::default());
                known.insert(comp_key.clone(), new_id);
                work.push_back(comp_key);
                Some(new_id)
            };

        // The initial DWA state is the epsilon closure of the NWA start state.
        // The initial path has a weight of `all()`, representing no constraints yet.
        let start_composition_raw = BTreeMap::from([(self.start_state, Weight::all())]);
        let start_composition_map =
            self.epsilon_closure_with_flag(start_composition_raw, has_epsilons);
        let start_composition = to_key(start_composition_map);

        if let Some(start_id) = get_or_create_dwa_state(
            start_composition,
            &mut dwa,
            &mut known_states,
            &mut worklist,
        ) {
            dwa.start_state = start_id;
        } else {
            // The start state leads to nothing, so the DWA is empty.
            return dwa;
        }

        while let Some(current_composition) = worklist.pop_front() {
            let current_dwa_id = *known_states.get(&current_composition).unwrap();

            // --- 1. Aggregate weights for the current DWA state ---
            let mut aggregate_weight = Weight::zeros();
            let mut aggregate_final_weight = Weight::zeros();
            let mut critical_points = BTreeSet::new();
            let mut is_final = false; // Flag to track if we encountered any final NWA state

            for (nwa_id, path_weight) in &current_composition {
                aggregate_weight |= path_weight;
                if let Some(final_w) = &self.states[*nwa_id].final_weight {
                    is_final = true; // We found a final state
                    aggregate_final_weight |= &(path_weight & final_w);
                }
                // Collect all exception characters that define alphabet partitions.
                for &char_code in self.states[*nwa_id].transitions.exceptions.keys() {
                    critical_points.insert(char_code);
                }
            }
            dwa.states[current_dwa_id].weight = aggregate_weight;
            if is_final {
                dwa.states[current_dwa_id].final_weight = Some(aggregate_final_weight);
            }

            // --- 2. Calculate the default transition for the DWA state ---
            let mut default_next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
            // Aggregate weight for default edge from this state.
            let mut default_weight_agg = Weight::zeros();
            for (nwa_id, path_weight) in &current_composition {
                if let Some(transitions) = self.states[*nwa_id].transitions.get_default() {
                    for (next_nwa_id, trans_weight) in transitions {
                        let next_path_weight = path_weight & trans_weight;
                        default_next_raw
                            .entry(*next_nwa_id)
                            .or_default()
                            .bitor_assign(&next_path_weight);
                        default_weight_agg |= &next_path_weight;
                    }
                }
            }
            let default_next_composition_map =
                self.epsilon_closure_with_flag(default_next_raw, has_epsilons);
            let default_next_composition = to_key(default_next_composition_map);
            let default_target_dwa_id = get_or_create_dwa_state(
                default_next_composition,
                &mut dwa,
                &mut known_states,
                &mut worklist,
            );
            dwa.states[current_dwa_id].transitions.default = default_target_dwa_id;
            if default_target_dwa_id.is_some() {
                dwa.states[current_dwa_id].trans_weight_default = Some(default_weight_agg);
            }

            // --- 3. Calculate transitions for all critical points (exceptions) ---
            for char_code in critical_points {
                let mut exception_next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
                let mut exception_weight_agg = Weight::zeros();
                for (nwa_id, path_weight) in &current_composition {
                    if let Some(transitions) = self.states[*nwa_id].transitions.get(char_code) {
                        for (next_nwa_id, trans_weight) in transitions {
                            let next_path_weight = path_weight & trans_weight;
                            exception_next_raw
                                .entry(*next_nwa_id)
                                .or_default()
                                .bitor_assign(&next_path_weight);
                            exception_weight_agg |= &next_path_weight;
                        }
                    }
                }
                // Skip work if there are no outgoing transitions on this character.
                if exception_next_raw.is_empty() {
                    continue;
                }

                let exception_next_composition_map =
                    self.epsilon_closure_with_flag(exception_next_raw, has_epsilons);
                let exception_next_composition = to_key(exception_next_composition_map);
                let exception_target_dwa_id = get_or_create_dwa_state(
                    exception_next_composition,
                    &mut dwa,
                    &mut known_states,
                    &mut worklist,
                );

                // Add an exception if it differs from the default transition.
                if exception_target_dwa_id != default_target_dwa_id {
                    if let Some(target_id) = exception_target_dwa_id {
                        dwa.states[current_dwa_id]
                            .transitions
                            .exceptions
                            .insert(char_code, target_id);
                        // Record the aggregate weight for this exception transition.
                        dwa.states[current_dwa_id]
                            .trans_weights_exceptions
                            .insert(char_code, exception_weight_agg);
                    }
                }
            }
        }

        dwa
    }

    /// Computes the epsilon closure for a set of NWA states and their path weights.
    ///
    /// This function transitively follows all epsilon transitions, combining weights
    /// until a fixed point is reached.
    fn epsilon_closure(&self, initial_states: BTreeMap<StateID, Weight>) -> BTreeMap<StateID, Weight> {
        // Preserve original behavior; determinize() uses the fast-path helper.
        self.epsilon_closure_with_flag(initial_states, true)
    }

    /// Same as epsilon_closure, but with a fast-path: when has_epsilons is false,
    /// return the input as-is to avoid needless work and allocations.
    fn epsilon_closure_with_flag(
        &self,
        initial_states: BTreeMap<StateID, Weight>,
        has_epsilons: bool,
    ) -> BTreeMap<StateID, Weight> {
        if !has_epsilons {
            return initial_states;
        }

        let mut closure = initial_states;
        let mut worklist: VecDeque<StateID> = closure.keys().cloned().collect();

        while let Some(u_id) = worklist.pop_front() {
            // This clone is necessary because we might modify `closure` inside the loop.
            let u_weight = closure.get(&u_id).unwrap().clone();
            if u_weight.is_empty() {
                continue;
            }

            for (v_id, trans_weight) in &self.states[u_id].epsilon_transitions {
                let new_v_weight = &u_weight & trans_weight;
                if new_v_weight.is_empty() {
                    continue;
                }

                let current_v_weight = closure.entry(*v_id).or_insert_with(Weight::zeros);
                let old_len = current_v_weight.len();
                *current_v_weight |= &new_v_weight;

                // If the weight for v changed, we need to re-process its epsilon transitions.
                if current_v_weight.len() > old_len {
                    worklist.push_back(*v_id);
                }
            }
        }
        closure
    }
}

// New: DWA simplification utilities (collapse simple edges, prune unreachable, merge equivalents)
impl DWA {
    /// Simplify the DWA by repeatedly:
    /// 1) collapsing chains of "simple" default-only states,
    /// 2) merging equivalent states,
    /// 3) pruning unreachable states,
    /// until a small fixed-point is reached.
    ///
    /// determinize() is intentionally left unchanged; call this afterwards if you want a smaller DWA.
    pub fn simplify(&mut self) {
        if self.states.is_empty() {
            return;
        }
        // Normalize trivial redundancies up-front.
        self.normalize_edges_inplace();
        self.prune_unreachable();

        let mut changed_any = true;
        let mut passes = 0usize;
        while changed_any && passes < 10 {
            passes += 1;
            changed_any = false;

            // 1) Normalize edges (drop exceptions equal to default).
            if self.normalize_edges_inplace() {
                changed_any = true;
            }
            // 2) Partition-refinement DFA minimization (language-preserving).
            if self.minimize_partition_refinement() {
                changed_any = true;
            }
            // 4) Final tidy-up per pass.
            if self.normalize_edges_inplace() {
                changed_any = true;
            }
            if self.prune_unreachable() {
                changed_any = true;
            }
        }
    }

    /// Drop exceptions that point to the default target and clean up their weights.
    ///
    /// Returns true if any exception was removed.
    pub fn normalize_edges_inplace(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states {
            if let Some(def) = st.transitions.default {
                let to_remove: Vec<u16> = st
                    .transitions
                    .exceptions
                    .iter()
                    .filter_map(|(ch, tgt)| if *tgt == def { Some(*ch) } else { None })
                    .collect();
                if !to_remove.is_empty() {
                    changed = true;
                }
                for ch in to_remove {
                    st.transitions.exceptions.remove(&ch);
                    st.trans_weights_exceptions.remove(&ch);
                }
            }
        }
        changed
    }

    /// Partition-refinement DFA minimization that preserves:
    /// - state.weight
    /// - state.final_weight
    /// - transition structure up to character classes (default + exceptions that differ from default)
    ///
    /// Missing transitions are treated as going to an implicit sink partition,
    /// enabling merging of partial and explicit-sink behaviors when equivalent.
    ///
    /// Returns true if any merge happened.
    pub fn minimize_partition_refinement(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 {
            return false;
        }
        // Use an implicit sink partition id for "no transition".
        let sink_pid: usize = n; // one beyond valid state indices

        // Initial partition by observable outputs: (weight, final_weight)
        use std::collections::hash_map::Entry;
        let mut part: Vec<usize> = vec![0; n];
        let mut canon0: std::collections::HashMap<(Weight, Option<Weight>), usize> =
            std::collections::HashMap::new();
        for i in 0..n {
            let key = (self.states[i].weight.clone(), self.states[i].final_weight.clone());
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
            let mut sig2pid: std::collections::HashMap<
                (Weight, Option<Weight>, usize, Vec<(u16, usize)>),
                usize,
            > = std::collections::HashMap::new();

            for i in 0..n {
                let st = &self.states[i];
                let def_cls = if let Some(d) = st.transitions.default {
                    part[d]
                } else {
                    sink_pid
                };
                // Only keep exceptions whose class differs from default class.
                let mut ex: Vec<(u16, usize)> = Vec::with_capacity(st.transitions.exceptions.len());
                for (ch, tgt) in &st.transitions.exceptions {
                    let cls = part[*tgt];
                    if cls != def_cls {
                        ex.push((*ch, cls));
                    }
                }
                // ex is already in key order due to BTreeMap iteration.
                let sig = (
                    st.weight.clone(),
                    st.final_weight.clone(),
                    def_cls,
                    ex,
                );
                let next_pid = sig2pid.len();
                next_part[i] = *sig2pid.entry(sig).or_insert(next_pid);
            }
            if next_part != part {
                part = next_part;
                changed = true;
            }
        }

        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for (i, p) in part.iter().enumerate() {
            groups.entry(*p).or_default().push(i);
        }
        if groups.len() == n {
            return false; // nothing to merge
        }

        // Build mapping from partition -> new state id
        let mut pid_to_new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut new_states: Vec<DWAState> = Vec::with_capacity(groups.len());
        for (pid, members) in &groups {
            let rep = members[0];
            let rep_state = &self.states[rep];
            // Representative default class (after refinement, all members share it)
            let def_cls = if let Some(d) = rep_state.transitions.default {
                part[d]
            } else {
                sink_pid
            };
            let mut st = DWAState::default();
            st.weight = rep_state.weight.clone();
            st.final_weight = rep_state.final_weight.clone();
            st.transitions.default = if def_cls == sink_pid {
                None
            } else {
                Some(0) // will be fixed after pid_to_new is filled
            };
            // Exceptions differing from default class
            for (ch, tgt) in &rep_state.transitions.exceptions {
                let cls = part[*tgt];
                if cls != def_cls {
                    st.transitions.exceptions.insert(*ch, 0); // placeholder
                }
            }
            // Aggregate per-edge weights across all members.
            if st.transitions.default.is_some() {
                let mut agg_def: Option<Weight> = None;
                for &old in members {
                    if let Some(w) = self.states[old].trans_weight_default.as_ref() {
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
                    if let Some(w) = self.states[old].trans_weights_exceptions.get(&ch) {
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
        // Fix transition targets now that we have pid_to_new.
        for (pid, members) in &groups {
            let new_id = *pid_to_new.get(pid).unwrap();
            let rep = members[0];
            let rep_state = &self.states[rep];
            // Default
            let def_cls = if let Some(d) = rep_state.transitions.default {
                part[d]
            } else {
                sink_pid
            };
            if let Some(ref mut d) = new_states[new_id].transitions.default {
                *d = *pid_to_new.get(&def_cls).unwrap();
            }
            // Exceptions
            let ex_old = new_states[new_id].transitions.exceptions.clone();
            new_states[new_id].transitions.exceptions.clear();
            for (ch, _) in ex_old {
                // For representative, we already know the class to use.
                let cls = part[*rep_state.transitions.exceptions.get(&ch).unwrap()];
                new_states[new_id]
                    .transitions
                    .exceptions
                    .insert(ch, *pid_to_new.get(&cls).unwrap());
            }
        }
        self.states = new_states;
        // Normalize edges (drop redundant exceptions)
        self.normalize_edges_inplace();
        // Remap start state
        let start_pid = part[self.start_state];
        self.start_state = *pid_to_new.get(&start_pid).unwrap();
        true
    }

    /// Remove states that are unreachable from `start_state` and renumber densely.
    ///
    /// Returns true if any state was removed.
    pub fn prune_unreachable(&mut self) -> bool {
        if self.states.is_empty() {
            return false;
        }
        let n = self.states.len();
        let mut visited = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        visited[self.start_state] = true;
        q.push_back(self.start_state);
        while let Some(u) = q.pop_front() {
            if let Some(d) = self.states[u].transitions.default {
                if !visited[d] {
                    visited[d] = true;
                    q.push_back(d);
                }
            }
            for &v in self.states[u].transitions.exceptions.values() {
                if !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }
        if visited.iter().all(|&b| b) {
            return false;
        }

        // Build old->new mapping for reachable states (dense numbering).
        let mut map = vec![usize::MAX; n];
        let mut next_id = 0usize;
        for i in 0..n {
            if visited[i] {
                map[i] = next_id;
                next_id += 1;
            }
        }
        // Rebuild states
        let mut new_states: Vec<DWAState> = Vec::with_capacity(next_id);
        for old in 0..n {
            if !visited[old] {
                continue;
            }
            let mut st = self.states[old].clone();
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
        self.states = new_states;
        self.start_state = map[self.start_state];
        true
    }

    /// Merge states that are identical in structure and weights.
    /// Two states are considered equivalent when all of the following are equal:
    /// - `weight`
    /// - `final_weight`
    /// - `transitions.default` (target id)
    /// - `transitions.exceptions` (map of char -> target id)
    ///
    /// Returns true if any merge happened.
    pub fn merge_equivalent_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 {
            return false;
        }
        use std::collections::hash_map::Entry;

        // Signature: (weight, final_weight, default_target, exceptions_vec)
        let mut canonical: std::collections::HashMap<
            (Weight, Option<Weight>, Option<usize>, Vec<(u16, usize)>),
            usize,
        > = std::collections::HashMap::new();
        let mut repr: Vec<usize> = vec![0; n];
        for i in 0..n {
            let default = self.states[i].transitions.default;
            // BTreeMap iteration is sorted, so this Vec is in a stable order.
            let exceptions: Vec<(u16, usize)> = self.states[i]
                .transitions
                .exceptions
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            let sig = (
                self.states[i].weight.clone(),
                self.states[i].final_weight.clone(),
                default,
                exceptions,
            );
            match canonical.entry(sig) {
                Entry::Occupied(o) => repr[i] = *o.get(),
                Entry::Vacant(v) => {
                    repr[i] = i;
                    v.insert(i);
                }
            }
        }
        let unique: std::collections::BTreeSet<usize> = repr.iter().cloned().collect();
        if unique.len() == n {
            return false; // nothing to merge
        }

        // Build (repr -> new id) and construct new state vector from representatives.
        let mut repr_to_new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut new_states: Vec<DWAState> = Vec::with_capacity(unique.len());
        for r in unique.iter().cloned() {
            let new_id = new_states.len();
            repr_to_new.insert(r, new_id);
            new_states.push(self.states[r].clone());
        }
        // Remap transitions inside the new states to representative indices.
        for st in &mut new_states {
            if let Some(d_old) = st.transitions.default {
                let r = repr[d_old];
                st.transitions.default = Some(*repr_to_new.get(&r).unwrap());
            }
            let ex_old = st.transitions.exceptions.clone();
            st.transitions.exceptions.clear();
            for (ch, t_old) in ex_old {
                let r = repr[t_old];
                st.transitions.exceptions.insert(ch, *repr_to_new.get(&r).unwrap());
            }
            // Drop exceptions that match the (new) default target.
            if let Some(def) = st.transitions.default {
                let to_remove: Vec<u16> = st
                    .transitions
                    .exceptions
                    .iter()
                    .filter_map(|(k, v)| if *v == def { Some(*k) } else { None })
                    .collect();
                for k in to_remove {
                    st.transitions.exceptions.remove(&k);
                }
            }
        }
        // Also normalize stray per-exception weights that might be dangling.
        for st in &mut new_states {
            let keys: Vec<u16> = st.trans_weights_exceptions.keys().cloned().collect();
            for k in keys {
                if !st.transitions.exceptions.contains_key(&k) { st.trans_weights_exceptions.remove(&k); }
            }
        }
        self.start_state = *repr_to_new.get(&repr[self.start_state]).unwrap();
        self.states = new_states;
        true
    }
}

// --- NWA utilities: processing stacks and structural helpers ---
impl NWA {
    /// Process an input stack (sequence of u16 symbols) through this NWA.
    ///
    /// Returns a vector of (pos, stop_state, path_weight) for all nondeterministic
    /// stops where:
    /// - a final state is reached (pos may be < input.len()), or
    /// - the input is exhausted (pos == input.len()).
    ///
    /// Path weights are accumulated by bitwise AND along edges and OR when
    /// multiple paths converge on the same (pos, state).
    pub fn process_stack_u16(&self, input: &[u16]) -> Vec<(StateID, StateID, Weight)> {
        // Note: For external callers the first tuple element is "pos", but since
        // type alias StateID = usize, we keep the signature consistent and document
        // that the first usize is the consumed position in `input`.
        if self.states.is_empty() {
            return Vec::new();
        }
        let has_epsilons = self
            .states
            .iter()
            .any(|s| !s.epsilon_transitions.is_empty());

        // Current frontier as a map: state -> path_weight
        let mut current: BTreeMap<StateID, Weight> = BTreeMap::new();
        current.insert(self.start_state, Weight::all());
        let mut current = self.epsilon_closure_with_flag(current, has_epsilons);

        // Accumulate results across positions; deduplicate by (pos, state) with OR for weights.
        let mut results: BTreeMap<(usize, StateID), Weight> = BTreeMap::new();
        let n = input.len();

        for pos in 0..=n {
            // 1) If any current state is final, we can stop here and record a result.
            for (&sid, path_w) in &current {
                if self.states[sid].final_weight.is_some() {
                    results
                        .entry((pos, sid))
                        .or_insert_with(Weight::zeros)
                        .bitor_assign(path_w);
                }
            }
            // 2) If we've consumed the entire input, record all current states as stops.
            if pos == n {
                for (&sid, path_w) in &current {
                    results
                        .entry((pos, sid))
                        .or_insert_with(Weight::zeros)
                        .bitor_assign(path_w);
                }
                break;
            }

            // 3) Advance one symbol.
            let ch = input[pos];
            let mut next_raw: BTreeMap<StateID, Weight> = BTreeMap::new();
            for (&sid, path_w) in &current {
                if let Some(transitions) = self.states[sid].transitions.get(ch) {
                    for (to, w) in transitions {
                        let w2 = path_w & w;
                        if !w2.is_empty() {
                            next_raw
                                .entry(*to)
                                .or_insert_with(Weight::zeros)
                                .bitor_assign(&w2);
                        }
                    }
                }
            }
            current = self.epsilon_closure_with_flag(next_raw, has_epsilons);
        }

        results
            .into_iter()
            .map(|((pos, sid), w)| (pos, sid, w))
            .collect()
    }

    /// Append a deep copy of `other` into `self`, returning a mapping from
    /// `other` StateID to new StateID in `self`.
    ///
    /// All transitions (exceptions, default, epsilons) are remapped accordingly.
    /// The `start_state` of `self` is unchanged. Final weights are copied.
    pub fn append_copy(&mut self, other: &NWA) -> Vec<StateID> {
        let base = self.states.len();
        let count = other.states.len();
        // Build the mapping (right id -> new id).
        let mut mapping: Vec<StateID> = Vec::with_capacity(count);
        for i in 0..count {
            mapping.push(base + i);
            self.states.push(NWAState::new());
        }
        // Populate each new state's fields with remapped references.
        for (i, st) in other.states.iter().enumerate() {
            let dst_id = base + i;
            let dst = &mut self.states[dst_id];
            // Final weight
            dst.final_weight = st.final_weight.clone();
            // Epsilon transitions
            dst.epsilon_transitions = st
                .epsilon_transitions
                .iter()
                .map(|(to, w)| (mapping[*to], w.clone()))
                .collect();
            // Labeled transitions
            let mut new_map: U16Map<Vec<(StateID, Weight)>> = U16Map::new();
            // Exceptions
            for (ch, vec) in st.transitions.exceptions.iter() {
                let remapped: Vec<(StateID, Weight)> = vec
                    .iter()
                    .map(|(to, w)| (mapping[*to], w.clone()))
                    .collect();
                new_map.exceptions.insert(*ch, remapped);
            }
            // Default
            if let Some(def) = st.transitions.default.as_ref() {
                let remapped: Vec<(StateID, Weight)> = def
                    .iter()
                    .map(|(to, w)| (mapping[*to], w.clone()))
                    .collect();
                new_map.default = Some(remapped);
            }
            dst.transitions = new_map;
        }
        mapping
    }

    /// Return the set of states reachable from `from` by following any transitions
    /// (exceptions, default, and epsilon), ignoring labels and weights.
    pub fn reachable_states_ignoring_labels(&self, from: StateID) -> BTreeSet<StateID> {
        let mut visited: BTreeSet<StateID> = BTreeSet::new();
        let mut q: VecDeque<StateID> = VecDeque::new();
        if from >= self.states.len() {
            return visited;
        }
        visited.insert(from);
        q.push_back(from);
        while let Some(u) = q.pop_front() {
            // Epsilons
            for (v, _) in &self.states[u].epsilon_transitions {
                if visited.insert(*v) {
                    q.push_back(*v);
                }
            }
            // Default
            if let Some(def) = self.states[u].transitions.default.as_ref() {
                for (v, _) in def {
                    if visited.insert(*v) {
                        q.push_back(*v);
                    }
                }
            }
            // Exceptions
            for vec in self.states[u].transitions.exceptions.values() {
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

// --- Display Implementations for Debugging ---

// --- Display Implementations for Debugging ---

impl Display for NWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWA (start: {})", self.start_state)?;
        for (id, state) in self.states.iter().enumerate() {
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

impl Display for DWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA (start: {})", self.start_state)?;
        for (id, state) in self.states.iter().enumerate() {
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

impl JSONConvertible for SimpleBitset {
    fn to_json(&self) -> JSONNode {
        let ranges: Vec<JSONNode> = self.0.ranges().map(|r| {
            JSONNode::Array(vec![
                JSONNode::Int(*r.start() as i128),
                JSONNode::Int(*r.end() as i128),
            ])
        }).collect();
        JSONNode::Array(ranges)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let mut ranges = Vec::new();
                for (i, range_node) in arr.into_iter().enumerate() {
                    match range_node {
                        JSONNode::Array(mut pair_vec) if pair_vec.len() == 2 => {
                            let end_node = pair_vec.pop().unwrap();
                            let start_node = pair_vec.pop().unwrap();

                            // Use StateID (usize) from_json for range bounds.
                            let start = StateID::from_json(start_node)
                                .map_err(|e| format!("While deserializing SimpleBitset range start at $[{}][0]: {}", i, e))?;
                            let end = StateID::from_json(end_node)
                                .map_err(|e| format!("While deserializing SimpleBitset range end at $[{}][1]: {}", i, e))?;

                            if start > end {
                                return Err(format!("Invalid range at $[{}]: start ({}) is greater than end ({})", i, start, end));
                            }
                            ranges.push(start..=end);
                        }
                        other => return Err(format!(
                            "Expected 2-element array for SimpleBitset range at $[{}], got {}",
                            i,
                            other.short_preview()
                        )),
                    }
                }
                Ok(SimpleBitset(RangeSetBlaze::from_iter(ranges)))
            }
            other => Err(format!(
                "Expected JSON array of [start, end] pairs for SimpleBitset, got {}",
                other.short_preview()
            )),
        }
    }
}

impl<T: JSONConvertible> JSONConvertible for U16Map<T> {
    // ... (existing impl)
}

impl JSONConvertible for NWAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("exceptions".to_string(), self.exceptions.to_json());
        obj.insert("default".to_string(), self.default.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = BTreeMap::new();
        obj.insert("transitions".to_string(), self.transitions.to_json());
        obj.insert("epsilon_transitions".to_string(), self.epsilon_transitions.to_json());
        obj.insert("final_weight".to_string(), self.final_weight.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let transitions = U16Map::<Vec<(StateID, Weight)>>::from_json(
            obj.remove("transitions").ok_or("Missing 'transitions' field")?
        )?;
        let epsilon_transitions = Vec::<(StateID, Weight)>::from_json(
            obj.remove("epsilon_transitions").ok_or("Missing 'epsilon_transitions' field")?
        )?;
        let final_weight = Option::<Weight>::from_json(
            obj.remove("final_weight").ok_or("Missing 'final_weight' field")?
        )?;
        Ok(NWAState {
            transitions,
            epsilon_transitions,
            final_weight,
        })
    }
}

impl JSONConvertible for NWA {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), self.states.to_json());
        obj.insert("start_state".to_string(), self.start_state.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let states = Vec::<NWAState>::from_json(
            obj.remove("states").ok_or("Missing 'states' field")?
        )?;
        let start_state = StateID::from_json(
            obj.remove("start_state").ok_or("Missing 'start_state' field")?
        )?;
        Ok(NWA { states, start_state })
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
        let transitions = U16Map::<StateID>::from_json(obj.remove("transitions").ok_or("Missing 'transitions' field")?)?;
        let weight = Weight::from_json(obj.remove("weight").ok_or("Missing 'weight' field")?)?;
        let final_weight = Option::<Weight>::from_json(obj.remove("final_weight").ok_or("Missing 'final_weight' field")?)?;
        let trans_weight_default = Option::<Weight>::from_json(obj.remove("trans_weight_default").ok_or("Missing 'trans_weight_default' field")?)?;
        let trans_weights_exceptions = BTreeMap::<u16, Weight>::from_json(obj.remove("trans_weights_exceptions").ok_or("Missing 'trans_weights_exceptions' field")?)?;
        Ok(DWAState {
            transitions,
            weight,
            final_weight,
            trans_weight_default,
            trans_weights_exceptions,
        })
    }
}

impl JSONConvertible for DWA {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), self.states.to_json());
        obj.insert("start_state".to_string(), self.start_state.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let states = Vec::<DWAState>::from_json(obj.remove("states").ok_or("Missing 'states' field")?)?;
        let start_state = StateID::from_json(obj.remove("start_state").ok_or("Missing 'start_state' field")?)?;
        Ok(DWA { states, start_state })
    }
}


// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    fn build_complex_nwa() -> NWA {
        let mut nwa = NWA::default();
        // Add 20 states (0 to 19)
        for _ in 0..20 {
            nwa.add_state();
        }
        nwa.start_state = 1;

        let w0 = SimpleBitset::from_item(0);
        let w1 = SimpleBitset::from_item(1);
        let w2 = SimpleBitset::from_item(2);
        let w3 = SimpleBitset::from_item(3);
        let w123 = SimpleBitset::from_iter(1..=3);
        let wall = SimpleBitset::all();

        // State 7 is final
        nwa.set_final_weight(7, wall.clone());

        // Transitions from s1 (1)
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

        // Transitions from s2 (2)
        nwa.add_default_transition(2, 8, wall.clone());

        // Transitions from s3 (3)
        nwa.add_transition(3, 10, 7, w3.clone());

        // Transitions from s4 (4)
        nwa.add_default_transition(4, 9, wall.clone());

        // Transitions from s5 (5)
        nwa.add_default_transition(5, 11, wall.clone());
        nwa.add_default_transition(5, 12, wall.clone());
        nwa.add_default_transition(5, 13, wall.clone());

        // Transitions from s6 (6)
        nwa.add_default_transition(6, 14, wall.clone());
        nwa.add_default_transition(6, 16, wall.clone());
        nwa.add_default_transition(6, 18, wall.clone());

        // Transitions from s8 (8)
        nwa.add_transition(8, 10, 7, w0.clone());

        // Transitions from s9 (9)
        nwa.add_default_transition(9, 10, wall.clone());

        // Transitions from s10 (10)
        nwa.add_transition(10, 10, 7, w0.clone());

        // The rest of the transitions (11-19) are simple and follow the pattern
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
        assert_eq!(map.get(b'c' as u16), Some(&100)); // default
        assert_eq!(map.get(b'b' as u16), Some(&20));
    }

    #[test]
    fn test_determinize_simple_nwa() {
        // NWA for "ab"
        // 0 --a, {1}--> 1 --b, {2}--> 2 (final, {3})
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(0, b'a' as u16, s1, SimpleBitset::from_iter(vec![1, 4]));
        nwa.add_transition(s1, b'b' as u16, s2, SimpleBitset::from_iter(vec![1, 5]));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![1, 6]));

        let dwa = nwa.determinize();
        println!("NWA:\n{}", nwa);
        println!("DWA:\n{}", dwa);
        // Expected DWA:
        // S0 (start): weight=ALL
        //   'a' -> S1
        // S1: weight={1}
        //   'b' -> S2
        // S2: weight={1,4}&{1,5}={1}, final_weight=({1})&{1,6}={1}
        assert_eq!(dwa.states.len(), 3);
        let start_state = &dwa.states[dwa.start_state];
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
        // NWA for "a?"
        // 0 --ε, {1}--> 1 --a, {2}--> 2 (final, {3})
        // 0 is also final with weight {4}
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_epsilon_transition(0, s1, SimpleBitset::from_iter(vec![1, 5]));
        nwa.add_transition(s1, b'a' as u16, s2, SimpleBitset::from_iter(vec![1, 6]));
        nwa.set_final_weight(0, SimpleBitset::from_item(4));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![1, 7]));

        let dwa = nwa.determinize();
        // Expected DWA:
        // S0 (start): composition={0:ALL, 1:{1}}. weight=ALL|{1}. final_weight={4}
        //   'a' -> S1
        // S1: composition={2:{1}&{2}}. weight={1}&{2}. final_weight=({1}&{2})&{3}
        assert_eq!(dwa.states.len(), 2);

        let start_state = &dwa.states[dwa.start_state];
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
        // NWA:
        // 0 --'a', {1}--> 1 (final, {10})
        // 0 --*, {2}--> 2 (final, {20})
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(0, b'a' as u16, s1, SimpleBitset::from_iter(vec![1, 5]));
        nwa.add_default_transition(0, s2, SimpleBitset::from_iter(vec![2, 5]));
        nwa.set_final_weight(s1, SimpleBitset::from_iter(vec![1, 10]));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![2, 20]));

        let dwa = nwa.determinize();
        // Expected DWA:
        // S0 (start): weight=ALL
        //   'a' -> S1 (exception)
        //   * -> S2 (default)
        // S1: weight={1}, final_weight={1}&{10}
        // S2: weight={2}, final_weight={2}&{20}
        assert_eq!(dwa.states.len(), 3);

        let start_state = &dwa.states[dwa.start_state];
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
        // NWA for "a" | "a"
        // 0 --a, {1}--> 1 (final, {10})
        // 0 --a, {2}--> 2 (final, {20})
        let mut nwa = NWA::new();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(0, b'a' as u16, s1, SimpleBitset::from_iter(vec![1, 5]));
        nwa.add_transition(0, b'a' as u16, s2, SimpleBitset::from_iter(vec![2, 5]));
        nwa.set_final_weight(s1, SimpleBitset::from_iter(vec![1, 10]));
        nwa.set_final_weight(s2, SimpleBitset::from_iter(vec![2, 20]));

        let dwa = nwa.determinize();
        // Expected DWA:
        // S0 (start): weight=ALL
        //   'a' -> S1
        // S1: composition={1:{1}, 2:{2}}. weight={1}|{2}. final_weight=({1}&{10})|({2}&{20})
        assert_eq!(dwa.states.len(), 2);

        let start_state = &dwa.states[dwa.start_state];
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
        // NWA from the user input:
        // Start state: 1. Final state: 3 (weight: ALL).
        // s1 transitions:
        // 1 --0, [0..=1]--> 2
        // 1 --1, [0]--> 3
        // 1 --1, [0..=1]--> 2 (Nondeterminism)
        // 1 --2, [0..=1]--> 2
        // 1 --3, [0..=1]--> 2
        // 1 --4, [0..=1]--> 2
        // 1 --4, [1]--> 3 (Nondeterminism)
        // 1 --5, [0..=1]--> 2
        let mut nwa = NWA::default();
        let s0 = nwa.add_state(); // 0
        let s1 = nwa.add_state(); // 1 (Start)
        let s2 = nwa.add_state(); // 2
        let s3 = nwa.add_state(); // 3 (Final)
        nwa.start_state = s1;

        let w01 = SimpleBitset::from_iter(0..=1);
        let w0 = SimpleBitset::from_item(0);
        let w1 = SimpleBitset::from_item(1);
        nwa.set_final_weight(s3, SimpleBitset::all());

        // Transitions from s1 (1)
        nwa.add_transition(s1, 0, s2, w01.clone());
        nwa.add_transition(s1, 1, s3, w0.clone());
        nwa.add_transition(s1, 1, s2, w01.clone());
        nwa.add_transition(s1, 2, s2, w01.clone());
        nwa.add_transition(s1, 3, s2, w01.clone());
        nwa.add_transition(s1, 4, s2, w01.clone());
        nwa.add_transition(s1, 4, s3, w1.clone());
        nwa.add_transition(s1, 5, s2, w01.clone());

        let dwa = nwa.determinize();
        println!("{}", nwa);
        println!("{}", dwa);

        // Expected DWA: 4 states (S0, S1, S2, S3)
        assert_eq!(dwa.states.len(), 4);
        let s0_dwa = &dwa.states[dwa.start_state]; // S0: {1:ALL}
        let w01_expected = SimpleBitset::from_iter(0..=1);

        // S0 checks
        assert_eq!(s0_dwa.weight, SimpleBitset::all());
        assert!(s0_dwa.final_weight.is_none());

        // S1: {2:[0..=1]}. Target for '0', '2', '3', '5'.
        let s1_dwa_id = *s0_dwa.transitions.get(0).unwrap();
        let s1_dwa = &dwa.states[s1_dwa_id];
        assert_eq!(s1_dwa.weight, w01_expected);
        assert!(s1_dwa.final_weight.is_none());

        // S2: {3:[0], 2:[0..=1]}. Target for '1'.
        let s2_dwa_id = *s0_dwa.transitions.get(1).unwrap();
        let s2_dwa = &dwa.states[s2_dwa_id];
        assert_eq!(s2_dwa.weight, w01_expected);
        assert_eq!(s2_dwa.final_weight, Some(w0)); // [0] & ALL = [0]

        // S3: {2:[0..=1], 3:[1]}. Target for '4'.
        let s3_dwa_id = *s0_dwa.transitions.get(4).unwrap();
        let s3_dwa = &dwa.states[s3_dwa_id];
        assert_eq!(s3_dwa.weight, w01_expected);
        assert_eq!(s3_dwa.final_weight, Some(w1)); // [1] & ALL = [1]

        // Check that '2', '3', '5' also go to S1
        assert_eq!(*s0_dwa.transitions.get(2).unwrap(), s1_dwa_id);
        assert_eq!(*s0_dwa.transitions.get(3).unwrap(), s1_dwa_id);
        assert_eq!(*s0_dwa.transitions.get(5).unwrap(), s1_dwa_id);
    }

    #[test]
    fn test_determinize_complex_nwa_from_input() {
        let nwa = build_complex_nwa();
        let dwa = nwa.determinize();

        // Expected DWA states: 15 (0 to 14)
        assert_eq!(dwa.states.len(), 15);

        let w0 = SimpleBitset::from_item(0);
        let w1 = SimpleBitset::from_item(1);
        let w2 = SimpleBitset::from_item(2);
        let w3 = SimpleBitset::from_item(3);
        let w12 = SimpleBitset::from_iter(1..=2);
        let w123 = SimpleBitset::from_iter(1..=3);
        let wall = SimpleBitset::all();

        // S0 (ID 0): {1:ALL} - Start state
        let s0 = &dwa.states[dwa.start_state];
        assert_eq!(s0.weight, wall);
        assert!(s0.final_weight.is_none());

        // S1 (ID 1): Target of '1' from S0. Composition: {2:[0]}
        let s1_id = *s0.transitions.get(1).unwrap();
        assert_eq!(s1_id, 1);
        let s1 = &dwa.states[s1_id];
        assert_eq!(s1.weight, w0.clone());
        assert!(s1.final_weight.is_none());

        // S2 (ID 2): Target of '2' from S0. Composition: {7:[1..=2], 3:[3]}
        let s2_id = *s0.transitions.get(2).unwrap();
        assert_eq!(s2_id, 2);
        let s2 = &dwa.states[s2_id];
        assert_eq!(s2.weight, w123.clone());
        // Final weight: ([1..=2] & ALL) | ([3] & None) = [1..=2]
        assert_eq!(s2.final_weight, Some(w12.clone()));

        // S7 (ID 7): Target of '10' from S0. Composition: {7:[1..=3]}
        let s7_id = *s0.transitions.get(10).unwrap();
        assert_eq!(s7_id, 7);
        let s7 = &dwa.states[s7_id];
        assert_eq!(s7.weight, w123.clone());
        // Final weight: [1..=3] & ALL = [1..=3]
        assert_eq!(s7.final_weight, Some(w123.clone()));

        // S9 (ID 9): Target of '10' from S2. Composition: {7:[3]}
        let s9_id = *s2.transitions.get(10).unwrap();
        assert_eq!(s9_id, 9);
        let s9 = &dwa.states[s9_id];
        assert_eq!(s9.weight, w3.clone());
        // Final weight: [3] & ALL = [3]
        assert_eq!(s9.final_weight, Some(w3.clone()));

        // Check S6 (ID 6) and S8 (ID 8) properties and transitions
        let s6_id = *s0.transitions.get(7).unwrap(); // Target of '7' from S0. Composition: {7:[0]}
        assert_eq!(s6_id, 6);
        let s6 = &dwa.states[s6_id];
        assert_eq!(s6.weight, w0.clone());
        assert_eq!(s6.final_weight, Some(w0.clone()));

        let s8_id = s1.transitions.default.unwrap(); // Target of '*' from S1. Composition: {8:[0]}
        assert_eq!(s8_id, 8);
        let s8 = &dwa.states[s8_id];
        assert_eq!(s8.weight, w0.clone());
        assert!(s8.final_weight.is_none());
        
        // Check a transition to a known state (S8 -> '10' -> S6)
        assert_eq!(*s8.transitions.get(10).unwrap(), s6_id);

        // Check a transition to a known state (S7 is a sink)
        assert!(s7.transitions.exceptions.is_empty());
        assert!(s7.transitions.default.is_none());
    }

    #[test]
    fn test_simple_bitset_json_conversion() {
        let original = SimpleBitset::from_iter(vec![1, 2, 5, 10..=12, usize::MAX]);
        let json_node = original.to_json();
        
        // Expected: [[1, 2], [5, 5], [10, 12], [usize::MAX, usize::MAX]]
        if let JSONNode::Array(arr) = &json_node {
            assert_eq!(arr.len(), 4);
            if let JSONNode::Array(r1) = &arr[0] {
                assert_eq!(r1.len(), 2);
                assert_eq!(r1[0], JSONNode::Int(1));
                assert_eq!(r1[1], JSONNode::Int(2));
            } else { panic!("Expected array"); }
        } else { panic!("Expected array"); }

        let deserialized = SimpleBitset::from_json(json_node).unwrap();
        assert_eq!(original, deserialized);

        // Test empty set
        let empty = SimpleBitset::zeros();
        let empty_node = empty.to_json();
        assert_eq!(empty_node, JSONNode::Array(vec![]));
        let deserialized_empty = SimpleBitset::from_json(empty_node).unwrap();
        assert_eq!(empty, deserialized_empty);
    }
}
