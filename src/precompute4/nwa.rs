use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, DWABody, Weight, StateID as DWAStateID};
use std::collections::{BTreeMap, BTreeSet, VecDeque, HashMap};

pub type NWAStateID = usize;

/// A slightly restricted NWA:
/// - Epsilon transitions are allowed; there may be any number of epsilons out of a state.
/// - Non-epsilon transitions: for each (state, symbol) there is at most one outgoing transition.
/// - Each transition carries a weight (Weight), which is intersected along the path; final states
///   carry a final weight that is intersected at acceptance; multiple alternative paths union their weights.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWAState {
    pub final_weight: Option<Weight>,
    /// Non-epsilon transitions: unique target per input symbol.
    /// Map: label -> (target, weight)
    pub transitions: BTreeMap<i16, (NWAStateID, Weight)>,
    /// Epsilon transitions: list of (target, weight).
    pub epsilons: Vec<(NWAStateID, Weight)>,
}

#[derive(Clone, Debug, Default)]
pub struct NWAStates(pub Vec<NWAState>);

impl NWAStates {
    pub fn len(&self) -> usize { self.0.len() }

    pub fn add_state(&mut self) -> NWAStateID {
        let id = self.0.len();
        self.0.push(NWAState::default());
        id
    }

    pub fn copy_state(&mut self, state_id: NWAStateID) -> NWAStateID {
        assert!(state_id < self.len(), "copy_state: state_id out of bounds");
        let new_id = self.add_state();
        self.0[new_id] = self.0[state_id].clone();
        new_id
    }

    pub fn add_epsilon(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) {
        assert!(from < self.len() && to < self.len(), "add_epsilon: state id out of bounds");
        self.0[from].epsilons.push((to, w));
    }

    /// Add a labeled transition; if an existing transition on the same label exists,
    /// we merge weights if the target is the same, otherwise we assert (as per restricted NWA).
    pub fn add_transition(&mut self, from: NWAStateID, on: i16, to: NWAStateID, w: Weight) {
        assert!(from < self.len() && to < self.len(), "add_transition: state id out of bounds");
        if let Some((old_to, old_w)) = self.0[from].transitions.get_mut(&on) {
            assert_eq!(*old_to, to, "NWA restricted: only one target per (state, symbol)");
            *old_w |= &w;
        } else {
            self.0[from].transitions.insert(on, (to, w));
        }
    }

    /// Deep-copy a subgraph starting at 'start_id' from another NWAStates arena into self.
    /// Returns (new_start_id, remap_old_to_new)
    pub fn copy_subgraph_from(&mut self, other: &NWAStates, start_id: NWAStateID) -> (NWAStateID, HashMap<NWAStateID, NWAStateID>) {
        let mut remap: HashMap<NWAStateID, NWAStateID> = HashMap::new();
        if start_id >= other.len() {
            let new_start = self.add_state();
            return (new_start, remap);
        }
        let new_start = self.add_state();
        self.0[new_start] = other.0[start_id].clone();
        remap.insert(start_id, new_start);

        let mut q = VecDeque::new();
        q.push_back((start_id, new_start));

        while let Some((old, new)) = q.pop_front() {
            // Epsilon edges
            let eps = other.0[old].epsilons.clone();
            self.0[new].epsilons.clear();
            for (to_old, w) in eps {
                let to_new = *remap.entry(to_old).or_insert_with(|| {
                    let n = self.add_state();
                    self.0[n] = other.0[to_old].clone();
                    q.push_back((to_old, n));
                    n
                });
                self.0[new].epsilons.push((to_new, w.clone()));
            }
            // Labeled edges
            let trans = other.0[old].transitions.clone();
            self.0[new].transitions.clear();
            for (lbl, (to_old, w)) in trans {
                let to_new = *remap.entry(to_old).or_insert_with(|| {
                    let n = self.add_state();
                    self.0[n] = other.0[to_old].clone();
                    q.push_back((to_old, n));
                    n
                });
                self.0[new].transitions.insert(lbl, (to_new, w.clone()));
            }
        }

        (new_start, remap)
    }
}

impl std::ops::Index<NWAStateID> for NWAStates {
    type Output = NWAState;
    fn index(&self, index: NWAStateID) -> &Self::Output { &self.0[index] }
}
impl std::ops::IndexMut<NWAStateID> for NWAStates {
    fn index_mut(&mut self, index: NWAStateID) -> &mut Self::Output { &mut self.0[index] }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWABody {
    pub start_state: NWAStateID,
}

#[derive(Clone, Debug, Default)]
pub struct NWA {
    pub states: NWAStates,
    pub body: NWABody,
}

impl NWA {
    pub fn new() -> Self {
        let mut states = NWAStates::default();
        let start = states.add_state();
        Self { states, body: NWABody { start_state: start } }
    }

    /// Convert a DWA to a NWA:
    /// - DWA labeled transitions -> NWA labeled transitions (same label, same weight)
    /// - DWA default transitions -> NWA epsilon transitions (weight of default)
    /// - Final weights preserved
    pub fn from_dwa(dwa: &DWA) -> Self {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        for _ in 0..dwa.states.len() { nwa.states.add_state(); }
        nwa.body.start_state = dwa.body.start_state;

        for (i, st) in dwa.states.0.iter().enumerate() {
            nwa.states[i].final_weight = st.final_weight.clone();
            // Default -> epsilon
            if let Some(to) = st.transitions.default {
                if let Some(w) = &st.trans_weight_default {
                    nwa.states.add_epsilon(i, to, w.clone());
                } else {
                    // If there's a default target but missing weight, treat as zeros (no-op)
                }
            }
            // Labeled
            for (lbl, to) in &st.transitions.exceptions {
                if let Some(w) = st.trans_weights_exceptions.get(lbl) {
                    nwa.states.add_transition(i, *lbl, *to, w.clone());
                }
            }
        }
        nwa
    }

    /// Determinize the subgraph reachable from 'start' to a DWA via weighted subset construction.
    /// Representation invariant:
    /// - A D-state is a mapping M: NWAStateID -> Weight capturing epsilon-closure weights.
    /// - For each label c, we compute T0 by distributing M over c-labeled edges, then epsilon-closure T.
    /// - The deterministic edge weight is union of weights in T; the target D-state is canonicalized T.
    /// - Final weight of the D-state is union over s in M of (M[s] & final_weight[s]).
    pub fn determinize_to_dwa(&self) -> DWA {
        fn epsilon_closure(states: &NWAStates, initial: &BTreeMap<NWAStateID, Weight>) -> BTreeMap<NWAStateID, Weight> {
            // Worklist algorithm: propagate union of weights along epsilons.
            let mut result: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
            let mut queue: VecDeque<NWAStateID> = VecDeque::new();

            // Initialize
            for (&sid, w) in initial {
                if !w.is_empty() {
                    result.insert(sid, w.clone());
                    queue.push_back(sid);
                }
            }

            while let Some(u) = queue.pop_front() {
                let u_weight = result.get(&u).cloned().unwrap_or_else(Weight::zeros);
                for &(v, ref eps_w) in &states[u].epsilons {
                    let propagated = &u_weight & eps_w;
                    if propagated.is_empty() { continue; }
                    if let Some(old) = result.get_mut(&v) {
                        let new_union = &*old | &propagated;
                        if &new_union != old {
                            *old = new_union;
                            queue.push_back(v);
                        }
                    } else {
                        result.insert(v, propagated);
                        queue.push_back(v);
                    }
                }
            }

            // Drop zero weights mappings just in case.
            result.retain(|_, w| !w.is_empty());
            result
        }

        // Canonicalization key type
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct SubsetKey(Vec<(NWAStateID, Weight)>);

        fn make_key(map: &BTreeMap<NWAStateID, Weight>) -> SubsetKey {
            let mut v: Vec<(NWAStateID, Weight)> = map.iter().map(|(k, w)| (*k, w.clone())).collect();
            // Already sorted by BTreeMap iteration order; ensure consistent
            SubsetKey(v)
        }

        // Build initial subset: epsilon-closure from start with Weight::all
        let mut init_map: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
        init_map.insert(self.body.start_state, Weight::all());
        let init_closure = epsilon_closure(&self.states, &init_map);

        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let start_d_id = dwa.states.add_state();
        dwa.body.start_state = start_d_id;

        let mut subset_to_d_id: HashMap<SubsetKey, DWAStateID> = HashMap::new();
        let mut worklist: VecDeque<SubsetKey> = VecDeque::new();

        let init_key = make_key(&init_closure);
        subset_to_d_id.insert(init_key.clone(), start_d_id);
        worklist.push_back(init_key);

        // Ensure enough DWA states vector capacity as we go; transitions/weights filled dynamically
        while let Some(subset_key) = worklist.pop_front() {
            let d_id = *subset_to_d_id.get(&subset_key).unwrap();
            let subset_map: BTreeMap<NWAStateID, Weight> = subset_key.0.iter().map(|(k,w)| (*k, w.clone())).collect();

            // Compute final weight
            let mut d_final: Option<Weight> = None;
            for (sid, w_in) in &subset_map {
                if let Some(fw) = &self.states[*sid].final_weight {
                    let contrib = w_in & fw;
                    if !contrib.is_empty() {
                        if let Some(ref mut acc) = d_final {
                            *acc |= &contrib;
                        } else {
                            d_final = Some(contrib);
                        }
                    }
                }
            }
            dwa.states[d_id].final_weight = d_final;

            // Collect labels present in this subset
            let mut labels: BTreeSet<i16> = BTreeSet::new();
            for (sid, _) in &subset_map {
                labels.extend(self.states[*sid].transitions.keys());
            }

            for lbl in labels {
                // Build T0
                let mut t0: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                for (sid, w_in) in &subset_map {
                    if let Some((to, w_edge)) = self.states[*sid].transitions.get(&lbl) {
                        let contrib = w_in & w_edge;
                        if contrib.is_empty() { continue; }
                        if let Some(old) = t0.get_mut(to) {
                            *old |= &contrib;
                        } else {
                            t0.insert(*to, contrib);
                        }
                    }
                }
                if t0.is_empty() { continue; }

                // Epsilon-closure
                let t_cl = epsilon_closure(&self.states, &t0);
                if t_cl.is_empty() { continue; }

                let target_key = make_key(&t_cl);
                let target_d_id = if let Some(id) = subset_to_d_id.get(&target_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(target_key.clone(), nid);
                    worklist.push_back(target_key.clone());
                    nid
                };

                // Edge weight = union of all weights in t_cl
                let mut edge_w_opt: Option<Weight> = None;
                for (_, w) in &t_cl {
                    if let Some(ref mut acc) = edge_w_opt {
                        *acc |= w;
                    } else {
                        edge_w_opt = Some(w.clone());
                    }
                }
                if let Some(edge_w) = edge_w_opt {
                    // DWA supports one transition per label per state; add exception
                    dwa.add_transition(d_id, lbl, target_d_id, edge_w).expect("DWA insertion should not fail");
                }
            }
        }

        dwa
    }

    /// Union of two NWAs (component-level within shared arena):
    /// Construct a new start with epsilon transitions (weight=ALL) to both operands' starts.
    /// Return a body whose start is the new start.
    pub fn union_components(states: &mut NWAStates, body1: &NWABody, body2: &NWABody) -> NWABody {
        let new_start = states.add_state();
        states.add_epsilon(new_start, body1.start_state, Weight::all());
        states.add_epsilon(new_start, body2.start_state, Weight::all());
        NWABody { start_state: new_start }
    }

    /// Concatenate left then right:
    /// - For each final state s reachable from left.start that has final_weight F,
    ///   add an epsilon s --eps-> right.start with weight (F & eps_weight).
    /// - Set s.final_weight = None (standard concatenation semantics).
    /// Return body with start = left.start.
    pub fn concatenate_components(states: &mut NWAStates, left: &NWABody, right: &NWABody, eps_weight: &Weight) -> NWABody {
        // 1) Collect reachable states from left.start
        let mut visited = vec![false; states.len()];
        let mut q = VecDeque::new();
        if left.start_state < states.len() {
            visited[left.start_state] = true;
            q.push_back(left.start_state);
        }
        while let Some(u) = q.pop_front() {
            // eps
            for &(v, _) in &states[u].epsilons {
                if v < states.len() && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
            // labeled
            for (_, (v, _)) in states[u].transitions.iter() {
                if *v < states.len() && !visited[*v] {
                    visited[*v] = true;
                    q.push_back(*v);
                }
            }
        }

        // 2) For each visited final, add epsilon and clear final.
        for sid in 0..states.len() {
            if !visited[sid] { continue; }
            if let Some(fw) = states[sid].final_weight.clone() {
                let w = &fw & eps_weight;
                if !w.is_empty() {
                    states.add_epsilon(sid, right.start_state, w);
                }
                // Reset final weight (standard concatenation)
                states[sid].final_weight = None;
            }
        }

        NWABody { start_state: left.start_state }
    }

    /// Determinize subgraph reachable from body.start_state to DWA
    pub fn determinize_components(states: &NWAStates, body: &NWABody) -> DWA {
        let tmp = NWA { states: states.clone(), body: body.clone() };
        tmp.determinize_to_dwa()
    }
}
