use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

impl DWA {
    /// Minimize a deterministic **acyclic** DWA.
    ///
    /// Key idea vs "classic DFA minimization":
    /// We first compute `need[s]`: tokens that can still reach *some* final from `s`.
    /// Then we **trim** every transition `u -> v` to `w(u,a) &= need[v]`.
    ///
    /// This is semantics-preserving under your evaluation model because any token not in `need[v]`
    /// can never contribute to acceptance along any continuation, hence allowing it to flow into `v`
    /// is unobservable at the start language/output.
    ///
    /// After trimming, a standard bottom-up DAG state hashing becomes correct and merges cases
    /// like your diamond example.
    pub fn minimize_acyclic(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // ---------------------------------------------------------------------
        // 0) Verify acyclicity + get topological order.
        // ---------------------------------------------------------------------
        let topo = topo_order_acyclic(&self.states).expect("minimize_acyclic called on cyclic DWA");

        // ---------------------------------------------------------------------
        // 1) Normalize:
        //    - Drop "dangling" weights for labels that are not transitions (eval ignores them).
        //    - Push state_weight into transitions/finals so we can ignore it thereafter.
        // ---------------------------------------------------------------------
        self.normalize_drop_default_weights();
        self.normalize_absorb_state_weights();

        // Refresh topo after normalization? Graph structure didn't change (only weights), so no need.

        // ---------------------------------------------------------------------
        // 2) Compute backward "need": tokens that can accept some suffix from each state.
        //
        // need[s] = final[s] ∪ ⋃_{(a->t)} (w(s,a) ∩ need[t])
        //
        // This is a standard backward DP on a DAG.
        // ---------------------------------------------------------------------
        let need = compute_need_backward(self, &topo);

        // ---------------------------------------------------------------------
        // 3) Trim transitions by destination need:
        //    w(u,a) := w(u,a) ∩ need[v]
        //    remove transition if empty.
        //
        // This is the critical semantic simplification enabling merges like your diamond.
        // ---------------------------------------------------------------------
        self.trim_transitions_by_need(&need);

        // After trimming edges to dead destinations, also trim finals (cheap cleanup).
        for s in 0..self.states.len() {
            if let Some(fw) = &mut self.states[s].final_weight {
                // Optional: intersect with need[s] (it already should be subset, but safe).
                *fw &= &need[s];
                if fw.is_empty() {
                    self.states[s].final_weight = None;
                }
            }
        }

        // ---------------------------------------------------------------------
        // 4) Remove unreachable states (graph-level reachability; weights already trimmed).
        //    This is important so we don't keep garbage and also stabilizes signatures.
        // ---------------------------------------------------------------------
        self.retain_graph_reachable_states();

        // If everything became unreachable except start, keep a single empty start.
        if self.states.len() == 0 {
            *self = DWA::new();
            return;
        }

        // Recompute topo on the new compacted graph.
        let topo = topo_order_acyclic(&self.states).expect("DWA became cyclic after trimming?");

        // ---------------------------------------------------------------------
        // 5) Bottom-up DAG minimization (register method).
        //    Because it's acyclic, we can do reverse-topo and hash state signatures.
        // ---------------------------------------------------------------------
        let class_of = minimize_by_signature(self, &topo);

        // ---------------------------------------------------------------------
        // 6) Rebuild minimized automaton by representatives.
        // ---------------------------------------------------------------------
        self.rebuild_from_classes(&class_of);

        // Final cleanup: remove unreachable again (usually no-op, but safe).
        self.retain_graph_reachable_states();
    }

    // -------------------------------------------------------------------------
    // Normalization helpers
    // -------------------------------------------------------------------------

    /// Remove any `trans_weights[label]` that doesn't have a corresponding explicit transition.
    /// `eval_word_weight()` ignores those anyway.
    fn normalize_drop_default_weights(&mut self) {
        for s in 0..self.states.len() {
            let dangling: Vec<Label> = self.states[s]
                .trans_weights
                .keys()
                .filter(|lbl| !self.states[s].transitions.contains_key(lbl))
                .copied()
                .collect();
            for lbl in dangling {
                self.states[s].trans_weights.remove(&lbl);
            }
        }
    }

    /// Absorb `state_weight` into transitions and finals, then clear it everywhere.
    ///
    /// Semantics:
    /// In `eval_word_weight`, after taking transition into `v`, you do `acc &= state_weight[v]`.
    /// That is equivalent to intersecting `state_weight[v]` into **every incoming transition** to `v`.
    ///
    /// For the start state, the start `state_weight` is applied before reading anything, which is
    /// equivalent to intersecting it into all outgoing transitions and its final weight.
    fn normalize_absorb_state_weights(&mut self) {}

    fn trim_transitions_by_need(&mut self, need: &[Weight]) {
        let n = self.states.len();
        for u in 0..n {
            let labels: Vec<Label> = self.states[u].transitions.keys().copied().collect();
            for lbl in labels {
                let v = match self.states[u].transitions.get(&lbl).copied() {
                    Some(v) => v,
                    None => continue,
                };
                if v >= n {
                    self.states[u].transitions.remove(&lbl);
                    self.states[u].trans_weights.remove(&lbl);
                    continue;
                }

                let w_old = self.states[u]
                    .trans_weights
                    .get(&lbl)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let mut w_new = w_old;
                w_new &= &need[v];

                if w_new.is_empty() {
                    self.states[u].transitions.remove(&lbl);
                    self.states[u].trans_weights.remove(&lbl);
                } else {
                    self.states[u].trans_weights.insert(lbl, w_new);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Reachability compaction
    // -------------------------------------------------------------------------

    /// Keep only graph-reachable states from the start (ignoring weights, because
    /// empty-weight transitions are physically removed earlier).
    fn retain_graph_reachable_states(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }
        let start = self.body.start_state;
        if start >= n {
            // If start is invalid, reset.
            *self = DWA::new();
            return;
        }

        let mut q = VecDeque::new();
        let mut seen = vec![false; n];
        seen[start] = true;
        q.push_back(start);

        while let Some(u) = q.pop_front() {
            for &v in self.states[u].transitions.values() {
                if v < n && !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }

        // Remap ids
        let mut new_id = vec![usize::MAX; n];
        let mut count = 0usize;
        for i in 0..n {
            if seen[i] {
                new_id[i] = count;
                count += 1;
            }
        }

        if count == n {
            return; // no change
        }

        let mut new_states_vec: Vec<DWAState> = Vec::with_capacity(count);
        for i in 0..n {
            if !seen[i] {
                continue;
            }
            let mut st = self.states[i].clone();

            // Remap transitions
            let labels: Vec<Label> = st.transitions.keys().copied().collect();
            for lbl in labels {
                let v = st.transitions[&lbl];
                if v >= n || !seen[v] {
                    st.transitions.remove(&lbl);
                    st.trans_weights.remove(&lbl);
                } else {
                    st.transitions.insert(lbl, new_id[v]);
                }
            }

            // Remove any dangling weights again.
            let dangling: Vec<Label> = st
                .trans_weights
                .keys()
                .filter(|lbl| !st.transitions.contains_key(lbl))
                .copied()
                .collect();
            for lbl in dangling {
                st.trans_weights.remove(&lbl);
            }

            new_states_vec.push(st);
        }

        self.body.start_state = new_id[start];
        self.states = DWAStates(new_states_vec);
    }

    // -------------------------------------------------------------------------
    // Rebuild from class mapping
    // -------------------------------------------------------------------------

    fn rebuild_from_classes(&mut self, class_of: &[usize]) {
        let n = self.states.len();
        if n == 0 {
            return;
        }
        assert_eq!(class_of.len(), n);

        let num_classes = class_of.iter().copied().max().unwrap_or(0) + 1;
        let mut rep_for_class: Vec<Option<usize>> = vec![None; num_classes];
        for s in 0..n {
            let c = class_of[s];
            rep_for_class[c].get_or_insert(s);
        }

        // Build new states by representative.
        let mut new_states: Vec<DWAState> = Vec::with_capacity(num_classes);
        for c in 0..num_classes {
            let rep = rep_for_class[c].expect("missing representative");
            let mut st = self.states[rep].clone();

            // Remap transitions to class ids (which become new StateIDs).
            let labels: Vec<Label> = st.transitions.keys().copied().collect();
            for lbl in labels {
                let to = st.transitions[&lbl];
                st.transitions.insert(lbl, class_of[to]);
            }

            new_states.push(st);
        }

        self.body.start_state = class_of[self.body.start_state];
        self.states = DWAStates(new_states);
    }
}

// ============================================================================
// Backward need computation (DAG DP)
// ============================================================================

fn compute_need_backward(dwa: &DWA, topo: &[usize]) -> Vec<Weight> {
    let n = dwa.states.len();
    let mut need = vec![Weight::zeros(); n];

    // reverse topo
    for &u in topo.iter().rev() {
        let mut acc = dwa.states[u]
            .final_weight
            .clone()
            .unwrap_or_else(Weight::zeros);

        for (lbl, &v) in &dwa.states[u].transitions {
            if v >= n {
                continue;
            }
            let w = dwa.states[u]
                .trans_weights
                .get(lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            // tokens that can pass this edge and still accept from v
            let mut contrib = w;
            contrib &= &need[v];
            acc |= &contrib;
        }

        need[u] = acc;
    }

    need
}

// ============================================================================
// Acyclic minimization by signature hashing (register method)
// ============================================================================

#[derive(Clone, Eq, PartialEq)]
struct WeightKey(Weight);

impl Hash for WeightKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let w = &self.0;
        w.is_all_fast().hash(state);
        w.is_empty().hash(state);
        if w.is_all_fast() || w.is_empty() {
            return;
        }
        // Hash as ranges for stability without requiring Weight: Hash
        // (this matches how you export to JSON).
        for r in w.rsb.ranges() {
            r.start().hash(state);
            r.end().hash(state);
        }
        // also include length to avoid collisions if universe differs
        w.len().hash(state);
    }
}

#[derive(Clone, Eq, PartialEq)]
struct StateSig {
    final_w: Option<WeightKey>,
    trans: Vec<(Label, usize, WeightKey)>, // (label, class(to), weight)
}

impl Hash for StateSig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.final_w.hash(state);
        self.trans.len().hash(state);
        for (lbl, to_c, w) in &self.trans {
            lbl.hash(state);
            to_c.hash(state);
            w.hash(state);
        }
    }
}

fn minimize_by_signature(dwa: &DWA, topo: &[usize]) -> Vec<usize> {
    let n = dwa.states.len();
    let mut class_of = vec![usize::MAX; n];
    let mut sig_to_class: HashMap<StateSig, usize> = HashMap::new();
    let mut next_class = 0usize;

    for &u in topo.iter().rev() {
        let final_w = dwa.states[u]
            .final_weight
            .clone()
            .map(WeightKey);

        let mut trans = Vec::with_capacity(dwa.states[u].transitions.len());
        for (lbl, &v) in &dwa.states[u].transitions {
            let v_class = class_of[v];
            assert!(v_class != usize::MAX, "child not processed first; topo order broken?");

            let w = dwa.states[u]
                .trans_weights
                .get(lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            trans.push((*lbl, v_class, WeightKey(w)));
        }

        let sig = StateSig { final_w, trans };

        if let Some(&c) = sig_to_class.get(&sig) {
            class_of[u] = c;
        } else {
            let c = next_class;
            next_class += 1;
            sig_to_class.insert(sig, c);
            class_of[u] = c;
        }
    }

    class_of
}

// ============================================================================
// Graph utilities
// ============================================================================

fn predecessors(states: &DWAStates) -> Vec<Vec<(usize, Label)>> {
    let n = states.len();
    let mut preds: Vec<Vec<(usize, Label)>> = vec![Vec::new(); n];
    for u in 0..n {
        for (&lbl, &v) in &states[u].transitions {
            if v < n {
                preds[v].push((u, lbl));
            }
        }
    }
    preds
}

fn topo_order_acyclic(states: &DWAStates) -> Option<Vec<usize>> {
    let n = states.len();
    let mut indeg = vec![0usize; n];
    for u in 0..n {
        for &v in states[u].transitions.values() {
            if v < n {
                indeg[v] += 1;
            }
        }
    }

    let mut q = VecDeque::new();
    for i in 0..n {
        if indeg[i] == 0 {
            q.push_back(i);
        }
    }

    let mut topo = Vec::with_capacity(n);
    while let Some(u) = q.pop_front() {
        topo.push(u);
        for &v in states[u].transitions.values() {
            if v >= n {
                continue;
            }
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }

    if topo.len() == n { Some(topo) } else { None }
}