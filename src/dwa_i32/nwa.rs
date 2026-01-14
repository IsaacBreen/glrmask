// src/precompute4/weighted_automata/nwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_i16_char, Label, NWAStateID, Weight, BENCHMARK_DEBUG};
use super::dwa::DWA;
use super::heavy_weight::WeightDimensions;
use crate::dwa_i32::{DWAState, StateID};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NWABuildError {
    StateOutOfBounds { state: NWAStateID },
}

impl Display for NWABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NWABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NWAState {
    pub final_weight: Option<Weight>,
    pub transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>>,
    pub epsilons: Vec<(NWAStateID, Weight)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NWAStates(pub Vec<NWAState>);

impl NWAStates {
    pub fn len(&self) -> usize { self.0.len() }

    pub fn num_transitions(&self) -> usize { self.0.iter().map(|s| s.transitions.iter().map(|(_, v)| v.len()).sum::<usize>() + s.epsilons.len()).sum() }

    pub fn add_state(&mut self) -> NWAStateID {
        let id = self.0.len();
        self.0.push(NWAState::default());
        id
    }

    pub fn add_existing_state(&mut self, state: NWAState) -> NWAStateID {
        let id = self.0.len(); self.0.push(state); id
    }

    pub fn add_epsilon(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) {
        if from < self.len() && to < self.len() {
            self.0[from].epsilons.push((to, w));
        }
    }

    pub fn add_transition(&mut self, from: NWAStateID, on: Label, to: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        if from >= self.len() { return Err(NWABuildError::StateOutOfBounds { state: from }); }
        if to >= self.len() { return Err(NWABuildError::StateOutOfBounds { state: to }); }
        self.0[from].transitions.entry(on).or_default().push((to, w));
        Ok(())
    }

    /// Appends all states from `other` into `self`, shifting their IDs.
    /// Returns the offset (ID of the first appended state).
    pub fn append(&mut self, other: &NWAStates) -> usize {
        let offset = self.len();
        self.0.reserve(other.len());
        for state in &other.0 {
            let mut new_state = state.clone();
            for targets in new_state.transitions.values_mut() {
                for (to, _) in targets { *to += offset; }
            }
            for (to, _) in &mut new_state.epsilons { *to += offset; }
            self.0.push(new_state);
        }
        offset
    }

    /// Concatenates `left` NWA onto `right_body` *in place*.
    /// `left` is appended to `self`. Then, for every final state in the appended `left`,
    /// epsilon transitions are added to all of `right_body`'s start states.
    /// Returns a new `NWABody` representing the starts of the concatenated structure (i.e., left's starts).
    pub fn concatenate_in_place(&mut self, left: &NWA, right_body: &NWABody) -> NWABody {
        let offset = self.append(&left.states);
        
        // Connect left's finals to right's starts
        for i in 0..left.states.len() {
            let abs_id = i + offset;
            if let Some(fw) = self.0[abs_id].final_weight.take() {
                if !fw.is_empty() {
                    for &r_start in &right_body.start_states {
                        self.add_epsilon(abs_id, r_start, fw.clone());
                    }
                }
            }
        }

        NWABody {
            start_states: left.body.start_states.iter().map(|s| s + offset).collect()
        }
    }

    pub fn union_in_place(&mut self, other: &NWA, existing_body: &NWABody) -> NWABody {
        let offset = self.append(&other.states);
        let mut new_starts = existing_body.start_states.clone();
        new_starts.extend(other.body.start_states.iter().map(|s| s + offset));
        NWABody {
            start_states: new_starts
        }
    }
}

impl Index<NWAStateID> for NWAStates {
    type Output = NWAState;
    fn index(&self, index: NWAStateID) -> &Self::Output { &self.0[index] }
}

impl IndexMut<NWAStateID> for NWAStates {
    fn index_mut(&mut self, index: NWAStateID) -> &mut Self::Output { &mut self.0[index] }
}

impl Display for NWAStates {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWAStates ({} states):", self.0.len())?;
        for (id, state) in self.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(w) = &state.final_weight { writeln!(f, "    final_weight: {}", w)?; }
            for (on, targets) in &state.transitions {
                for (to, w) in targets {
                    writeln!(f, "    {} -> {} (weight: {})", format_i16_char(*on), to, w)?;
                }
            }
            for (to, w) in &state.epsilons { writeln!(f, "    ε -> {} (weight: {})", to, w)?; }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NWABody {
    pub start_states: Vec<NWAStateID>,
}

impl NWABody {
    pub fn union(a: &NWABody, b: &NWABody) -> NWABody {
        let mut s = a.start_states.clone();
        s.extend(&b.start_states);
        NWABody { start_states: s }
    }
}

impl Display for NWABody {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { 
        write!(f, "NWABody (starts: {:?})", self.start_states) 
    }
}

#[derive(Debug, Clone)]
pub struct NWAStats {
    pub num_states: usize,
    pub num_final_states: usize,
    pub total_epsilon: usize,
    pub total_labeled: usize,
}

impl Display for NWAStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "States: {}, Finals: {}, Eps: {}, Labeled: {}", 
            self.num_states, self.num_final_states, self.total_epsilon, self.total_labeled)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NWA {
    pub states: NWAStates,
    pub body: NWABody,
    /// Weight space dimensions (num_tokens × num_tsids).
    #[serde(default)]
    pub dims: WeightDimensions,
}

impl Default for NWA {
    fn default() -> Self {
        Self { states: NWAStates::default(), body: NWABody::default(), dims: WeightDimensions::TEST }
    }
}

impl NWA {
    pub fn new_empty() -> Self {
        Self { states: NWAStates::default(), body: NWABody::default(), dims: WeightDimensions::TEST }
    }
    pub fn new() -> Self {
        let mut nwa = Self::new_empty();
        let start = nwa.add_state();
        nwa.body.start_states.push(start);
        nwa
    }
    pub fn new_with_dims(dims: WeightDimensions) -> Self {
        let mut nwa = Self::new();
        nwa.dims = dims;
        nwa
    }
    pub fn new_empty_with_dims(dims: WeightDimensions) -> Self {
        Self { states: NWAStates::default(), body: NWABody::default(), dims }
    }
    pub fn dimensions(&self) -> WeightDimensions {
        self.dims
    }
    pub fn set_dimensions(&mut self, dims: WeightDimensions) {
        self.dims = dims;
    }
    pub fn add_state(&mut self) -> NWAStateID { self.states.add_state() }
    pub fn add_epsilon(&mut self, u: NWAStateID, v: NWAStateID, w: Weight) { self.states.add_epsilon(u, v, w); }
    pub fn add_transition(&mut self, u: NWAStateID, l: Label, v: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        self.states.add_transition(u, l, v, w)
    }

    pub fn reverse(&self) -> NWA {
        let mut rev = NWA::new_empty();
        rev.dims = self.dims; // Propagate dimensions
        // Pre-allocate states
        for _ in 0..self.states.len() { rev.add_state(); }

        // Create a super-start for the reversed NWA that connects to all old final states
        let super_start = rev.add_state();
        rev.body.start_states = vec![super_start];

        for (u, state) in self.states.0.iter().enumerate() {
            // Reverse transitions: u -> v becomes v -> u
            for (lbl, targets) in &state.transitions {
                for (v, w) in targets { rev.add_transition(*v, *lbl, u, w.clone()).unwrap(); }
            }
            // Reverse epsilons
            for (v, w) in &state.epsilons { rev.add_epsilon(*v, u, w.clone()); }
            
            // Old finals become reachable from new super-start via epsilon
            if let Some(fw) = &state.final_weight {
                if !fw.is_empty() {
                    rev.add_epsilon(super_start, u, fw.clone());
                }
            }
        }

        // Old start states become final in the reversed NWA with Weight::all()
        for &s in &self.body.start_states {
            if s < rev.states.len() {
                let old = rev.states[s].final_weight.clone().unwrap_or_else(Weight::zeros);
                rev.states[s].final_weight = Some(old | Weight::all());
            }
        }
        
        rev
    }

    pub fn concatenate(left: &NWA, right: &NWA) -> NWA {
        let mut res = NWA::new_empty();
        // Propagate dimensions from left (primary operand)
        res.dims = left.dims;
        let _ = res.states.append(&right.states); // Right is at offset 0
        // Construct a body for the right segment
        let right_body = right.body.clone(); // indices are 0-based, valid
        
        // Concatenate left into place (appends left states and links finals -> right starts)
        res.body = res.states.concatenate_in_place(left, &right_body);
        res
    }

    pub fn union(a: &NWA, b: &NWA) -> NWA {
        let mut a = a.clone();
        // Dimensions are preserved from a (primary operand)
        a.body = NWAStates::union_in_place(&mut a.states, &b, &a.body);
        a
    }

    pub fn union_assign(&mut self, other: &NWA) {
        // Keep self's dimensions
        self.body = NWAStates::union_in_place(&mut self.states, &other, &self.body);
    }

    pub fn concatenate_assign(&mut self, other: &NWA) {
        // Keep self's dimensions
        self.body = NWAStates::concatenate_in_place(&mut self.states, &other, &self.body);
    }

    pub fn from_dwa(dwa: &DWA) -> Self {
        let mut nwa = NWA::new_empty();
        nwa.dims = dwa.dims; // Propagate dimensions from DWA
        for _ in 0..dwa.states.len() { nwa.add_state(); }
        nwa.body.start_states = vec![dwa.body.start_state];

        for (i, st) in dwa.states.0.iter().enumerate() {
            nwa.states[i].final_weight = st.final_weight.clone();
            for (lbl, to) in &st.transitions {
                let w = st.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
                nwa.add_transition(i, *lbl, *to, w).unwrap();
            }
        }
        nwa
    }

    pub fn stats(&self) -> NWAStats {
        let mut st = NWAStats { num_states: self.states.len(), num_final_states: 0, total_epsilon: 0, total_labeled: 0 };
        for s in &self.states.0 {
            if s.final_weight.is_some() { st.num_final_states += 1; }
            st.total_epsilon += s.epsilons.len();
            st.total_labeled += s.transitions.values().map(|v| v.len()).sum::<usize>();
        }
        st
    }

    /// Counts the total number of ranges across all weights in this NWA.
    /// This includes final weights, transition weights, and epsilon weights.
    /// Note: If the same weight object appears multiple times, its ranges are counted each time.
    pub fn num_ranges(&self) -> usize {
        let mut total = 0;
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                total += fw.num_ranges();
            }
            for targets in state.transitions.values() {
                for (_, w) in targets {
                    total += w.num_ranges();
                }
            }
            for (_, w) in &state.epsilons {
                total += w.num_ranges();
            }
        }
        total
    }

    /// Counts the total number of ranges across unique (interned) weights in this NWA.
    /// If the same interned weight appears multiple times, it is only counted once.
    pub fn num_ranges_interned(&self) -> usize {
        use std::collections::HashSet;
        use std::ptr;
        
        // Track unique weights by their intern ID
        let mut seen: HashSet<usize> = HashSet::new();
        let mut total = 0;
        
        let mut process_weight = |w: &Weight| {
            // Get the intern ID as a unique identifier
            let ptr = w.intern_id();
            if seen.insert(ptr) {
                total += w.num_ranges();
            }
        };
        
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                process_weight(fw);
            }
            for targets in state.transitions.values() {
                for (_, w) in targets {
                    process_weight(w);
                }
            }
            for (_, w) in &state.epsilons {
                process_weight(w);
            }
        }
        total
    }

    /// Eliminate epsilon chains for visualization.
    /// 
    /// A state is "epsilon-only" if it has:
    /// - No final_weight
    /// - No labeled transitions
    /// - Only epsilon transitions (to other states)
    /// 
    /// For such states, we replace all incoming edges with direct edges to their
    /// epsilon targets, effectively collapsing epsilon chains.
    pub fn eliminate_epsilon_chains(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Identify epsilon-only states
        let mut epsilon_only: Vec<bool> = vec![false; n];
        for (i, state) in self.states.0.iter().enumerate() {
            if state.final_weight.is_none() 
                && state.transitions.is_empty()
                && !state.epsilons.is_empty() {
                epsilon_only[i] = true;
            }
        }

        // For each epsilon-only state, compute its epsilon closure (reachable via epsilon).
        // This gives us the "resolved" targets to replace incoming edges with.
        let mut epsilon_closure: Vec<Vec<(usize, Weight)>> = vec![Vec::new(); n];
        for start in 0..n {
            if !epsilon_only[start] {
                continue;
            }
            // BFS/DFS through epsilon edges
            let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut stack: Vec<(usize, Weight)> = vec![(start, Weight::all())];
            while let Some((state, w)) = stack.pop() {
                if !visited.insert(state) {
                    continue;
                }
                if !epsilon_only[state] || state != start {
                    // This is a non-epsilon-only state, add to closure
                    epsilon_closure[start].push((state, w.clone()));
                    if !epsilon_only[state] {
                        continue; // Don't follow its epsilons
                    }
                }
                // Follow epsilon transitions
                for (target, eps_w) in &self.states[state].epsilons {
                    let combined = &w & eps_w;
                    if !combined.is_empty() {
                        stack.push((*target, combined));
                    }
                }
            }
        }

        // Now update all epsilon edges: if they point to an epsilon-only state,
        // replace with edges to that state's closure
        for i in 0..n {
            let mut new_epsilons: Vec<(usize, Weight)> = Vec::new();
            for (target, w) in std::mem::take(&mut self.states.0[i].epsilons) {
                if epsilon_only[target] && !epsilon_closure[target].is_empty() {
                    // Replace with closure targets
                    for (final_target, closure_w) in &epsilon_closure[target] {
                        let combined = &w & closure_w;
                        if !combined.is_empty() {
                            new_epsilons.push((*final_target, combined));
                        }
                    }
                } else {
                    // Keep original
                    new_epsilons.push((target, w));
                }
            }
            self.states.0[i].epsilons = new_epsilons;
        }

        // Similarly for labeled transitions
        for i in 0..n {
            for targets in self.states.0[i].transitions.values_mut() {
                let mut new_targets: Vec<(usize, Weight)> = Vec::new();
                for (target, w) in std::mem::take(targets) {
                    if epsilon_only[target] && !epsilon_closure[target].is_empty() {
                        for (final_target, closure_w) in &epsilon_closure[target] {
                            let combined = &w & closure_w;
                            if !combined.is_empty() {
                                new_targets.push((*final_target, combined));
                            }
                        }
                    } else {
                        new_targets.push((target, w));
                    }
                }
                *targets = new_targets;
            }
        }

        // Update start states if needed
        let mut new_starts: Vec<usize> = Vec::new();
        for &s in &self.body.start_states {
            if epsilon_only[s] && !epsilon_closure[s].is_empty() {
                for (target, _) in &epsilon_closure[s] {
                    if !new_starts.contains(target) {
                        new_starts.push(*target);
                    }
                }
            } else {
                if !new_starts.contains(&s) {
                    new_starts.push(s);
                }
            }
        }
        self.body.start_states = new_starts;

        // Note: We don't remove the epsilon-only states themselves to preserve state IDs
        // (which are used for template region mapping). They just become unreachable.
    }

    pub fn optimize_for_visualization(&mut self) {
        // First eliminate epsilon chains
        self.eliminate_epsilon_chains();

        let n = self.states.len();
        if n == 0 {
            return;
        }

        // 1. Forward tokens: tokens that can reach each state from some start state.
        let mut forward: Vec<Weight> = vec![Weight::zeros(); n];
        for &s in &self.body.start_states {
            if s < n {
                forward[s] |= &Weight::all();
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for u in 0..n {
                let fu = forward[u].clone();
                if fu.is_empty() {
                    continue;
                }

                // Epsilon transitions
                for (v, w) in &self.states[u].epsilons {
                    if *v >= n {
                        continue;
                    }
                    let flow = &fu & w;
                    if !flow.is_subset_of(&forward[*v]) {
                        forward[*v] |= &flow;
                        changed = true;
                    }
                }

                // Labeled transitions
                for targets in self.states[u].transitions.values() {
                    for (v, w) in targets {
                        if *v >= n {
                            continue;
                        }
                        let flow = &fu & w;
                        if !flow.is_subset_of(&forward[*v]) {
                            forward[*v] |= &flow;
                            changed = true;
                        }
                    }
                }
            }
        }

        // 2. Backward tokens: tokens that can go from each state to some final state.
        let mut backward: Vec<Weight> = vec![Weight::zeros(); n];

        changed = true;
        while changed {
            changed = false;
            for u in 0..n {
                let mut b_new = backward[u].clone();

                if let Some(fw) = &self.states[u].final_weight {
                    if !fw.is_subset_of(&b_new) {
                        b_new |= fw;
                    }
                }

                for (v, w) in &self.states[u].epsilons {
                    if *v >= n {
                        continue;
                    }
                    let contrib = w & &backward[*v];
                    if !contrib.is_subset_of(&b_new) {
                        b_new |= &contrib;
                    }
                }

                for targets in self.states[u].transitions.values() {
                    for (v, w) in targets {
                        if *v >= n {
                            continue;
                        }
                        let contrib = w & &backward[*v];
                        if !contrib.is_subset_of(&b_new) {
                            b_new |= &contrib;
                        }
                    }
                }

                if !b_new.is_subset_of(&backward[u]) {
                    backward[u] |= &b_new;
                    changed = true;
                }
            }
        }

        // 3. Prune weights using forward & backward information.
        for u in 0..n {
            // Final weights: tokens must be reachable from some start.
            if let Some(fw) = &mut self.states[u].final_weight {
                *fw &= &forward[u];
                if fw.is_empty() {
                    self.states[u].final_weight = None;
                }
            }

            // Epsilon transitions
            let mut new_eps = Vec::new();
            for (v, w) in &self.states[u].epsilons {
                if *v >= n {
                    continue;
                }
                let mut new_w = w.clone();
                new_w &= &forward[u];
                new_w &= &backward[*v];
                if !new_w.is_empty() {
                    new_eps.push((*v, new_w));
                }
            }
            self.states[u].epsilons = new_eps;

            // Labeled transitions
            for targets in self.states[u].transitions.values_mut() {
                let mut new_targets = Vec::new();
                for (v, w) in targets.iter() {
                    if *v >= n {
                        continue;
                    }
                    let mut new_w = w.clone();
                    new_w &= &forward[u];
                    new_w &= &backward[*v];
                    if !new_w.is_empty() {
                        new_targets.push((*v, new_w));
                    }
                }
                *targets = new_targets;
            }
            self.states[u].transitions.retain(|_, targets| !targets.is_empty());
        }
    }
}

impl Display for NWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWA {}", self.body)?;
        write!(f, "{}", self.states)
    }
}
