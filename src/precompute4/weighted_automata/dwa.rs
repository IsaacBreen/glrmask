// src/precompute4/weighted_automata/dwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_pos_code, I16Map, StateID, Weight};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::collections::{BTreeSet, HashSet};
use std::fmt::{self, Display, Formatter};
use std::ops::{Deref, Index, IndexMut};


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

impl DWA {
    /// Build a single-state DWA that over-approximates this DWA:
    /// - final weight equals eval([]) of the original DWA.
    /// - for every label seen anywhere, the self-loop on that label has weight equal to
    ///   the union of weights across all states' transitions on that label;
    /// - if any default transitions exist, install a self-loop default whose weight is the union of all default weights.
    /// This is only used if later proven equivalent to the original (via `equivalent`), ensuring correctness.
    pub fn build_single_state_overapprox(&self) -> DWA {
        let mut over = DWA::new();
        let sid = over.body.start_state;
        // Set final weight to the original's acceptance for empty word.
        let empty = self.eval_word_weight(&[]);
        if !empty.is_empty() {
            over.states[sid].final_weight = Some(empty.clone());
        }
        // Gather union of all exception labels across all states.
        let mut all_labels: BTreeSet<i16> = BTreeSet::new();
        let mut weight_by_label: BTreeMap<i16, Weight> = BTreeMap::new();
        let mut default_union: Option<Weight> = None;
        for st in &self.states.0 {
            for (lbl, w) in st.trans_weights_exceptions.iter() {
                all_labels.insert(*lbl);
                weight_by_label
                    .entry(*lbl)
                    .and_modify(|acc| *acc |= w)
                    .or_insert_with(|| w.clone());
            }
            if let Some(wd) = &st.trans_weight_default {
                default_union = Some(match default_union.take() {
                    Some(mut acc) => {
                        acc |= wd;
                        acc
                    }
                    None => wd.clone(),
                });
            }
        }
        // Install exception edges (self-loops).
        for lbl in all_labels {
            if let Some(w) = weight_by_label.get(&lbl) {
                let _ = over.add_transition(sid, lbl, sid, w.clone());
            }
        }
        // If there is any default in the original, add a default self-loop too.
        if let Some(wd) = default_union {
            let _ = over.set_default_transition(sid, sid, wd);
        }
        over
    }

    /// Precise language equivalence between two DWAs.
    /// It explores pairs of states while carrying a pair of prefix masks (one per machine).
    /// For each reachable pair, it checks equality of the "empty suffix" result (P & final),
    /// and then steps over a finite cover of label cases:
    ///   - All exception labels present in either state's exception set
    ///   - One "Other" case representing the default if at least one side has a default.
    pub fn equivalent(&self, other: &DWA) -> bool {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        enum Case {
            Label(i16),
            Other, // all labels not in exceptions (requires at least one default)
        }
        fn label_cases(a: &DWAState, b: &DWAState) -> Vec<Case> {
            let mut set: BTreeSet<Case> = BTreeSet::new();
            for (ch, _) in a.transitions.exceptions.iter() {
                set.insert(Case::Label(*ch));
            }
            for (ch, _) in b.transitions.exceptions.iter() {
                set.insert(Case::Label(*ch));
            }
            if a.transitions.default.is_some() || b.transitions.default.is_some() {
                set.insert(Case::Other);
            }
            set.into_iter().collect()
        }
        fn step(st: &DWAState, c: &Case) -> (Option<StateID>, Weight) {
            match c {
                Case::Label(lbl) => {
                    if let Some(&to) = st.transitions.exceptions.get(lbl) {
                        let w = st
                            .trans_weights_exceptions
                            .get(lbl)
                            .cloned()
                            .unwrap_or_else(Weight::all);
                        (Some(to), w)
                    } else if let Some(to) = st.transitions.default {
                        let w = st
                            .trans_weight_default
                            .as_ref()
                            .cloned()
                            .unwrap_or_else(Weight::all);
                        (Some(to), w)
                    } else {
                        (None, Weight::zeros())
                    }
                }
                Case::Other => {
                    if let Some(to) = st.transitions.default {
                        let w = st
                            .trans_weight_default
                            .as_ref()
                            .cloned()
                            .unwrap_or_else(Weight::all);
                        (Some(to), w)
                    } else {
                        (None, Weight::zeros())
                    }
                }
            }
        }
        #[derive(Hash, Eq, PartialEq, Clone, Copy)]
        struct Key {
            s: usize,
            t: usize,
            p1: u64,
            p2: u64,
        }
        let mut q: VecDeque<(usize, usize, Weight, Weight)> = VecDeque::new();
        let mut seen: HashSet<Key> = HashSet::new();
        // Start pair with state-entry weights applied (if any).
        let mut p1 = Weight::all();
        let mut p2 = Weight::all();
        let s0 = self.body.start_state;
        let t0 = other.body.start_state;
        if let Some(sw) = &self.states[s0].state_weight {
            p1 &= sw;
        }
        if let Some(sw) = &other.states[t0].state_weight {
            p2 &= sw;
        }
        q.push_back((s0, t0, p1.clone(), p2.clone()));
        seen.insert(Key { s: s0, t: t0, p1: p1.fp, p2: p2.fp });
        while let Some((s, t, p1, p2)) = q.pop_front() {
            // Compare empty-suffix result
            let f1 = self.states[s].final_weight.as_ref().cloned().unwrap_or_else(Weight::zeros);
            let f2 = other.states[t].final_weight.as_ref().cloned().unwrap_or_else(Weight::zeros);
            if (&p1 & &f1) != (&p2 & &f2) {
                return false;
            }
            // Next step cases
            let cases = label_cases(&self.states[s], &other.states[t]);
            for case in cases {
                let (to1, w1) = step(&self.states[s], &case);
                let (to2, w2) = step(&other.states[t], &case);
                let mut np1 = &p1 & &w1;
                let mut np2 = &p2 & &w2;
                let ns = if let Some(ns) = to1 {
                    if let Some(sw) = &self.states[ns].state_weight {
                        np1 &= sw;
                    }
                    Some(ns)
                } else {
                    None
                };
                let nt = if let Some(nt) = to2 {
                    if let Some(sw) = &other.states[nt].state_weight {
                        np2 &= sw;
                    }
                    Some(nt)
                } else {
                    None
                };
                // If both dead and both masks empty, nothing to enqueue
                if ns.is_none() && nt.is_none() && np1.is_empty() && np2.is_empty() {
                    continue;
                }
                // If one dead and other not, still must enqueue to check future finals (though
                // with empty mask this will quickly converge).
                let ns_id = ns.unwrap_or(self.states.len()); // out-of-range sentinel
                let nt_id = nt.unwrap_or(other.states.len()); // out-of-range sentinel
                // If mask pairs are both zero, we can still enqueue once to compare finals = 0 vs 0.
                let k = Key { s: ns_id, t: nt_id, p1: np1.fp, p2: np2.fp };
                if !seen.contains(&k) {
                    seen.insert(k);
                    q.push_back((ns_id, nt_id, np1, np2));
                }
            }
        }
        true
    }

    /// Merge multiple pure final sinks (no outgoing edges) into a single sink with unioned final weight.
    /// Returns a new minimized DWA. Always correct.
    pub fn merge_pure_final_sinks(&self) -> DWA {
        let n = self.states.len();
        if n == 0 {
            return self.clone();
        }
        let mut sink_ids: Vec<usize> = Vec::new();
        for (i, st) in self.states.0.iter().enumerate() {
            let no_out = st.transitions.exceptions.is_empty() && st.transitions.default.is_none();
            if no_out && st.final_weight.is_some() {
                sink_ids.push(i);
            }
        }
        if sink_ids.len() <= 1 {
            return self.clone(); // nothing to merge
        }
        let rep = *sink_ids.iter().min().unwrap();
        // Build mapping old->new index while skipping merged sinks.
        let mut keep = vec![true; n];
        for &i in &sink_ids {
            if i != rep {
                keep[i] = false;
            }
        }
        // Copy states, remap indices.
        let mut remap: HashMap<usize, usize> = HashMap::new();
        let mut out = DWA::new();
        out.states.0.clear();
        for i in 0..n {
            if keep[i] {
                remap.insert(i, out.states.add_existing_state(self.states[i].clone()));
            }
        }
        out.body.start_state = remap[&self.body.start_state];
        // Union final weights of merged sinks into rep
        let rep_new = remap[&rep];
        let mut rep_final = self.states[rep].final_weight.clone().unwrap_or_else(Weight::zeros);
        for &i in &sink_ids {
            if i == rep { continue; }
            if let Some(w) = &self.states[i].final_weight {
                rep_final |= w;
            }
        }
        out.states[rep_new].final_weight = Some(rep_final);
        // Remap transitions
        for (old_i, new_i) in remap.iter() {
            let st_old = &self.states[*old_i];
            let st_new = &mut out.states[*new_i];
            if let Some(to) = st_old.transitions.default {
                st_new.transitions.default = Some(remap[&to]);
            }
            let mut new_exc: BTreeMap<i16, StateID> = BTreeMap::new();
            for (ch, to) in st_old.transitions.exceptions.iter() {
                new_exc.insert(*ch, remap[to]);
            }
            st_new.transitions.exceptions = new_exc;
        }
        out
    }

    /// Merge states that are structurally identical (ignoring weights) and have no default transitions.
    /// For each equivalence class:
    ///   - union their final weights,
    ///   - for every exception label, union the transition weight,
    ///   - targets are identical by structural equivalence and are remapped.
    /// This pass shrinks "pre-c" states in the simple-divergence example.
    pub fn merge_structurally_equivalent_states(&self) -> DWA {
        #[derive(Hash, Eq, PartialEq, Clone)]
        struct ShapeKey {
            labels: Vec<i16>,
            targets: Vec<usize>,
            has_default: bool,
        }
        let n = self.states.len();
        if n == 0 {
            return self.clone();
        }
        // Build shapes
        let mut shapes: HashMap<ShapeKey, Vec<usize>> = HashMap::new();
        for (i, st) in self.states.0.iter().enumerate() {
            let has_default = st.transitions.default.is_some();
            if has_default {
                // conservative: don't try to merge states with defaults in this pass
                continue;
            }
            let mut labels: Vec<i16> = st.transitions.exceptions.keys().copied().collect();
            labels.sort_unstable();
            let mut targets: Vec<usize> = Vec::with_capacity(labels.len());
            for &lbl in &labels {
                targets.push(st.transitions.exceptions[&lbl]);
            }
            let key = ShapeKey { labels, targets, has_default };
            shapes.entry(key).or_default().push(i);
        }
        // Determine which groups are mergeable (size >= 2)
        let mut rep_of: HashMap<usize, usize> = HashMap::new();
        for (_k, group) in shapes.iter() {
            if group.len() >= 2 {
                let rep = *group.iter().min().unwrap();
                for &id in group {
                    rep_of.insert(id, rep);
                }
            }
        }
        if rep_of.is_empty() {
            return self.clone();
        }
        // Build unioned versions for reps
        let mut union_final: HashMap<usize, Weight> = HashMap::new();
        let mut union_weights: HashMap<(usize, i16), Weight> = HashMap::new(); // (rep, label) -> weight
        for (id, st) in self.states.0.iter().enumerate() {
            let rep = rep_of.get(&id).cloned().unwrap_or(id);
            if let Some(fw) = &st.final_weight {
                union_final
                    .entry(rep)
                    .and_modify(|acc| *acc |= fw)
                    .or_insert_with(|| fw.clone());
            }
            for (lbl, w) in st.trans_weights_exceptions.iter() {
                union_weights
                    .entry((rep, *lbl))
                    .and_modify(|acc| *acc |= w)
                    .or_insert_with(|| w.clone());
            }
        }
        // Build set of kept ids
        let mut keep = vec![true; n];
        for (id, rep) in rep_of.iter() {
            if *id != *rep {
                keep[*id] = false;
            }
        }
        // Copy states while unioning weights into reps
        let mut remap: HashMap<usize, usize> = HashMap::new();
        let mut out = DWA::new();
        out.states.0.clear();
        for i in 0..n {
            if keep[i] {
                remap.insert(i, out.states.add_existing_state(self.states[i].clone()));
            }
        }
        out.body.start_state = remap[&rep_of.get(&self.body.start_state).cloned().unwrap_or(self.body.start_state)];
        // Apply unioned finals and unioned transition weights to representatives
        for (old, new) in remap.iter() {
            let rep = rep_of.get(old).cloned().unwrap_or(*old);
            if let Some(fw) = union_final.get(&rep) {
                if !fw.is_empty() {
                    out.states[*new].final_weight = Some(fw.clone());
                }
            }
            let labels: Vec<i16> = out.states[*new].transitions.exceptions.keys().copied().collect();
            let mut new_exc_weights = out.states[*new].trans_weights_exceptions.clone();
            for lbl in labels {
                if let Some(w) = union_weights.get(&(rep, lbl)) {
                    new_exc_weights.insert(lbl, w.clone());
                }
            }
            out.states[*new].trans_weights_exceptions = new_exc_weights;
        }
        // Remap transitions (targets are same ids modulo representative mapping)
        for (old, new) in remap.iter() {
            let mut new_exc: BTreeMap<i16, StateID> = BTreeMap::new();
            for (ch, to) in self.states[*old].transitions.exceptions.iter() {
                let to_rep = rep_of.get(to).cloned().unwrap_or(*to);
                new_exc.insert(*ch, remap[&to_rep]);
            }
            out.states[*new].transitions.exceptions = new_exc;
            if let Some(to) = self.states[*old].transitions.default {
                let to_rep = rep_of.get(&to).cloned().unwrap_or(to);
                out.states[*new].transitions.default = Some(remap[&to_rep]);
                // keep default weights as-is; we avoided merging defaults to be conservative
            }
        }
        out
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAState {
    pub transitions: I16Map<StateID>,
    pub final_weight: Option<Weight>,
    pub trans_weight_default: Option<Weight>,
    pub trans_weights_exceptions: BTreeMap<i16, Weight>,
    /// Optional state-entry weight (intersected upon entering the state).
    pub state_weight: Option<Weight>,
}

impl DWAState {
    pub fn get_weight(&self, ch: i16) -> Option<&Weight> {
        self.trans_weights_exceptions.get(&ch).or(self.trans_weight_default.as_ref())
    }

    /// Intersects all weights in this state with the given weight.
    pub fn apply_weight(&mut self, weight: &Weight) {
        if let Some(sw) = &mut self.state_weight {
            *sw &= weight;
            if sw.is_empty() {
                self.state_weight = None;
            }
        }

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
        if let Some(sw) = &mut self.state_weight {
            *sw -= weight;
            if sw.is_empty() {
                self.state_weight = None;
            }
        }

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

    /// Iterator over all outgoing edges:
    /// - Default edge appears as (None, target, weight)
    /// - Exception edges appear as (Some(label), target, weight)
    #[inline]
    pub fn iter_edges(&self) -> impl Iterator<Item = (Option<i16>, StateID, &Weight)> {
        let def_iter = self
            .transitions
            .default
            .and_then(|to| self.trans_weight_default.as_ref().map(move |w| (to, w)))
            .into_iter()
            .map(|(to, w)| (None, to, w));
        let ex_iter = self
            .transitions
            .exceptions
            .iter()
            .filter_map(|(ch, to)| self.trans_weights_exceptions.get(ch).map(|w| (Some(*ch), *to, w)));
        def_iter.chain(ex_iter)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

    pub fn set_state_weight(&mut self, state: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state });
        }
        self.states[state].state_weight = Some(weight);
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
}

#[derive(Debug, Clone)]
pub struct DWAStats {
    pub num_states: usize,
    pub num_transitions: usize,
    pub num_final_states: usize,
    pub num_default_transitions: usize,
    pub avg_exceptions_per_state: f64,
}

impl Display for DWAStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "DWA Stats:")?;
        writeln!(f, "  - States: {}", self.num_states)?;
        writeln!(f, "  - Transitions: {}", self.num_transitions)?;
        writeln!(f, "  - Final States: {}", self.num_final_states)?;
        writeln!(f, "  - States with Default: {}", self.num_default_transitions)?;
        writeln!(f, "  - Avg Exceptions/State: {:.2}", self.avg_exceptions_per_state)
    }
}

impl DWA {
    pub fn stats(&self) -> DWAStats {
        let num_states = self.states.len();
        if num_states == 0 {
            return DWAStats {
                num_states: 0,
                num_transitions: 0,
                num_final_states: 0,
                num_default_transitions: 0,
                avg_exceptions_per_state: 0.0,
            };
        }

        let mut num_exceptions = 0;
        let mut num_final_states = 0;
        let mut num_default_transitions = 0;

        for state in &self.states.0 {
            num_exceptions += state.transitions.exceptions.len();
            if state.final_weight.is_some() {
                num_final_states += 1;
            }
            if state.transitions.default.is_some() {
                num_default_transitions += 1;
            }
        }

        let num_transitions = num_exceptions + num_default_transitions;
        let avg_exceptions_per_state = num_exceptions as f64 / num_states as f64;

        DWAStats {
            num_states,
            num_transitions,
            num_final_states,
            num_default_transitions,
            avg_exceptions_per_state,
        }
    }
}

impl Display for DWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(sw) = &state.state_weight {
                writeln!(f, "    state_weight: {}", sw)?;
            }
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
