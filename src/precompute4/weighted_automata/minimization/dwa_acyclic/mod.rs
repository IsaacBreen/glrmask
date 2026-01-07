use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug, PartialEq, Eq)]
struct StateSignature {
    // Canonicalized:
    // - final_weight: None => EMPTY, so store concrete weight
    final_weight: Weight,

    // Outgoing transitions, sorted by label (BTreeMap iteration order).
    // Target is already the representative/minimized state id.
    transitions: Vec<(Label, StateID, Weight)>,
}

fn hash_weight<H: Hasher>(w: &Weight, h: &mut H) {
    // Ensure a stable hash regardless of internal representation.
    if w.is_all_fast() {
        0u8.hash(h);
        return;
    }
    if w.is_empty() {
        1u8.hash(h);
        return;
    }
    2u8.hash(h);

    // Include len() because the same ranges might have different meaning if the
    // underlying universe length differs.
    w.len().hash(h);

    // Hash the ranges.
    for r in w.rsb.ranges() {
        r.start().hash(h);
        r.end().hash(h);
    }
}

impl Hash for StateSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_weight(&self.final_weight, state);
        for (lbl, to, w) in &self.transitions {
            lbl.hash(state);
            to.hash(state);
            hash_weight(w, state);
        }
    }
}

impl DWA {
    /// Remove inconsistent/semantically-dead data that can only prevent merges:
    /// - transitions to OOB targets
    /// - transitions with missing weights (treated as absent by eval_word_weight)
    /// - transitions with empty weight (always reject -> same as absent)
    /// - trans_weights entries without a transition (ignored by eval_word_weight)
    fn normalize_for_minimization(&mut self) {
        let n = self.states.len();

        for s in 0..n {

            // Canonicalize final_weight: empty => None
            if let Some(fw) = &self.states[s].final_weight {
                if fw.is_empty() {
                    self.states[s].final_weight = None;
                }
            }

            // Clean transitions.
            let labels: Vec<Label> = self.states[s].transitions.keys().copied().collect();
            for lbl in labels {
                let to = match self.states[s].transitions.get(&lbl).copied() {
                    Some(t) => t,
                    None => continue,
                };

                // OOB target => remove
                if to >= n {
                    self.states[s].transitions.remove(&lbl);
                    self.states[s].trans_weights.remove(&lbl);
                    continue;
                }

                // Missing weight => treated as no transition by get_transition()
                let w = match self.states[s].trans_weights.get(&lbl) {
                    Some(w) => w.clone(),
                    None => {
                        self.states[s].transitions.remove(&lbl);
                        continue;
                    }
                };

                // Empty weight => transition never contributes => remove transition
                if w.is_empty() {
                    self.states[s].transitions.remove(&lbl);
                    self.states[s].trans_weights.remove(&lbl);
                    continue;
                }

                // Optionally canonicalize ALL
                if w.is_all_fast() {
                    self.states[s].trans_weights.insert(lbl, Weight::all());
                }
            }

            // Remove any trans_weights that don't correspond to an explicit transition
            // (ignored by eval_word_weight anyway).
            let extra: Vec<Label> = self.states[s]
                .trans_weights
                .keys()
                .filter(|lbl| !self.states[s].transitions.contains_key(lbl))
                .copied()
                .collect();
            for lbl in extra {
                self.states[s].trans_weights.remove(&lbl);
            }
        }
    }

    /// Trim states that are unreachable from the start via the *graph structure*.
    /// This preserves `eval_word_weight` because unreachable states cannot affect
    /// evaluation from the start state.
    fn trim_unreachable_states(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let start = self.body.start_state;
        if start >= n {
            // Invalid start => nothing meaningful; normalize to empty.
            self.states = DWAStates::default();
            self.body.start_state = 0;
            return;
        }

        let mut reachable = vec![false; n];
        let mut stack = vec![start];
        reachable[start] = true;

        while let Some(u) = stack.pop() {
            for &v in self.states[u].transitions.values() {
                if v < n && !reachable[v] {
                    reachable[v] = true;
                    stack.push(v);
                }
            }
        }

        // If everything is reachable, keep as-is.
        if reachable.iter().all(|&b| b) {
            return;
        }

        let mut old_to_new = vec![usize::MAX; n];
        let mut new_states: Vec<DWAState> = Vec::new();
        new_states.reserve(reachable.iter().filter(|&&b| b).count());

        for old in 0..n {
            if reachable[old] {
                old_to_new[old] = new_states.len();
                new_states.push(self.states[old].clone());
            }
        }

        let new_start = old_to_new[start];
        debug_assert!(new_start != usize::MAX);

        // Remap transitions
        let new_n = new_states.len();
        for s in 0..new_n {
            let labels: Vec<Label> = new_states[s].transitions.keys().copied().collect();
            for lbl in labels {
                let old_to = match new_states[s].transitions.get(&lbl).copied() {
                    Some(t) => t,
                    None => continue,
                };
                let new_to = old_to_new.get(old_to).copied().unwrap_or(usize::MAX);
                if new_to == usize::MAX {
                    new_states[s].transitions.remove(&lbl);
                    new_states[s].trans_weights.remove(&lbl);
                } else {
                    new_states[s].transitions.insert(lbl, new_to);
                }
            }

            // Remove any stray weights without transitions
            let extra: Vec<Label> = new_states[s]
                .trans_weights
                .keys()
                .filter(|lbl| !new_states[s].transitions.contains_key(lbl))
                .copied()
                .collect();
            for lbl in extra {
                new_states[s].trans_weights.remove(&lbl);
            }
        }

        self.states = DWAStates(new_states);
        self.body.start_state = new_start;
    }

    fn topo_sort_states(&self) -> Vec<StateID> {
        let n = self.states.len();
        let mut indeg = vec![0usize; n];

        for u in 0..n {
            for &v in self.states[u].transitions.values() {
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

        let mut order = Vec::with_capacity(n);
        while let Some(u) = q.pop_front() {
            order.push(u);
            for &v in self.states[u].transitions.values() {
                if v < n {
                    indeg[v] -= 1;
                    if indeg[v] == 0 {
                        q.push_back(v);
                    }
                }
            }
        }

        order
    }

    /// Provably minimal minimization for acyclic deterministic DWAs under
    /// *state-equivalence* (right-language / right-weight equivalence).
    ///
    /// Important: this does *not* perform the "context-dependent" merges that
    /// your diamond example expects. Those merges are not DFA-style state
    /// equivalences and the global optimum becomes NP-hard in the general case.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        // If cyclic, this algorithm is not applicable.
        // (You could return early or panic; returning is safer in library code.)
        if self.is_cyclic() {
            return;
        }

        // Normalize obvious inconsistencies and dead edges.
        self.normalize_for_minimization();

        // Trim unreachable states first to avoid pointless signatures.
        self.trim_unreachable_states();

        if self.states.len() == 0 {
            return;
        }
        if self.is_cyclic() {
            // Trimming shouldn't create cycles, but keep it safe.
            return;
        }

        let topo = self.topo_sort_states();
        if topo.len() != self.states.len() {
            // Not a DAG (or something inconsistent).
            return;
        }

        let old_n = self.states.len();
        let old_start = self.body.start_state;

        // Map old -> new representative id
        let mut rep_of_old: Vec<StateID> = vec![usize::MAX; old_n];

        // Signatures -> representative state id
        let mut sig_to_rep: HashMap<StateSignature, StateID> = HashMap::new();

        // New minimized states
        let mut new_states: Vec<DWAState> = Vec::new();

        for &old_s in topo.iter().rev() {
            let old_state = &self.states[old_s];

            // Canonicalize optional weights.
            let fw = old_state
                .final_weight
                .clone()
                .unwrap_or_else(Weight::zeros);

            let mut transitions: Vec<(Label, StateID, Weight)> =
                Vec::with_capacity(old_state.transitions.len());

            for (&lbl, &old_to) in old_state.transitions.iter() {
                if old_to >= old_n {
                    continue;
                }

                let to_rep = rep_of_old[old_to];
                debug_assert!(to_rep != usize::MAX, "topo order invariant broken");

                let w = old_state
                    .trans_weights
                    .get(&lbl)
                    .cloned()
                    // Missing weight means get_transition() would treat it as absent,
                    // so by the time we are here it should not happen.
                    .unwrap_or_else(Weight::zeros);

                if w.is_empty() {
                    // Semantically absent
                    continue;
                }

                transitions.push((lbl, to_rep, w));
            }

            let sig = StateSignature {
                final_weight: fw,
                transitions,
            };

            if let Some(&existing_rep) = sig_to_rep.get(&sig) {
                rep_of_old[old_s] = existing_rep;
                continue;
            }

            // Create a new representative state.
            let new_id = new_states.len();
            let mut new_state = DWAState::default();


            if sig.final_weight.is_empty() {
                new_state.final_weight = None;
            } else {
                new_state.final_weight = Some(sig.final_weight.clone());
            }

            for (lbl, to_rep, w) in sig.transitions.iter() {
                new_state.transitions.insert(*lbl, *to_rep);
                new_state.trans_weights.insert(*lbl, w.clone());
            }

            new_states.push(new_state);
            sig_to_rep.insert(sig, new_id);
            rep_of_old[old_s] = new_id;
        }

        let new_start = rep_of_old[old_start];
        debug_assert!(new_start != usize::MAX);

        self.states = DWAStates(new_states);
        self.body.start_state = new_start;
    }
}