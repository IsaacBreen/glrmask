#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use crate::precompute4::weighted_automata::{DWABody, DWAState, DWAStates, StateID, Weight, DWA};
use crate::precompute4::weighted_automata::common::Label;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DWAMinimizeError {
    StartStateOutOfBounds { start: StateID, num_states: usize },
    Cyclic,
}

impl DWA {
    pub fn minimize_acyclic(&mut self) -> Result<(), DWAMinimizeError> {
        let minimized = self.minimize_acyclic_provably()?;
        *self = minimized;
        Ok(())
    }
}

/// Hash a Weight by its “shape” (ranges). Used only to speed up signature hashing.
/// Equality still uses full Weight equality, so hash collisions are safe (just slower).
fn hash_weight_into<H: Hasher>(w: &Weight, h: &mut H) {
    if w.is_empty() {
        0u8.hash(h);
        return;
    }
    if w.is_all_fast() {
        1u8.hash(h);
        return;
    }
    2u8.hash(h);

    // This relies on your existing JSON export using `w.rsb.ranges()`.
    // If `Weight` changes representation, update this accordingly.
    for r in w.rsb.ranges() {
        r.start().hash(h);
        r.end().hash(h);
    }
    w.len().hash(h);
}

#[derive(Clone, Debug)]
struct TransSig {
    lbl: Label,
    to: StateID,
    w: Weight,
}

impl PartialEq for TransSig {
    fn eq(&self, other: &Self) -> bool {
        self.lbl == other.lbl && self.to == other.to && self.w == other.w
    }
}
impl Eq for TransSig {}

impl Hash for TransSig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.lbl.hash(state);
        self.to.hash(state);
        hash_weight_into(&self.w, state);
    }
}

#[derive(Clone, Debug)]
struct StateSig {
    final_w: Option<Weight>,
    trans: Vec<TransSig>, // sorted by lbl
}

impl PartialEq for StateSig {
    fn eq(&self, other: &Self) -> bool {
        self.final_w == other.final_w && self.trans == other.trans
    }
}
impl Eq for StateSig {}

impl Hash for StateSig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self.final_w {
            None => 0u8.hash(state),
            Some(w) => {
                1u8.hash(state);
                hash_weight_into(w, state);
            }
        }
        self.trans.hash(state);
    }
}

impl DWA {
    /// Provably minimal minimization for **acyclic deterministic** DWAs under the
    /// standard notion:
    ///
    /// Two states are mergeable iff they induce identical suffix functions
    /// `Σ* -> Weight` when started with accumulator `ALL`.
    ///
    /// This is the weighted analogue of minimal DAFSA construction:
    /// - trim unreachable structure
    /// - trim semantically-dead token bits (backward support)
    /// - reverse-topo signature hashing to merge equivalent states
    ///
    /// If you instead want “start-only compression” (merge non-equivalent states by
    /// redistributing weights on incoming edges), that is a different optimization
    /// problem and this function is not intended to solve it.
    pub fn minimize_acyclic_provably(&self) -> Result<DWA, DWAMinimizeError> {
        let n0 = self.states.len();
        if n0 == 0 {
            return Ok(DWA::default());
        }
        if self.body.start_state >= n0 {
            return Err(DWAMinimizeError::StartStateOutOfBounds {
                start: self.body.start_state,
                num_states: n0,
            });
        }

        // 1) Clone and clean: remove out-of-bounds targets, remove empty-weight transitions,
        // and remove dangling trans_weights entries not backed by transitions.
        let mut work = self.clone();
        clean_dwa_inplace(&mut work);

        // 2) Restrict to reachable from start (graph reachability ignoring weights content,
        // because empty-weight transitions were removed above).
        work = restrict_to_reachable(&work);

        // 3) Ensure acyclic and get topo order.
        let topo = topo_order_or_err(&work)?;

        // 4) Compute backward support B[q] = tokens that can be produced by *some* suffix.
        // (This is semantics-preserving pruning: tokens not in B[target] can never matter.)
        let backward = compute_backward_support(&work, &topo);

        // 5) Trim finals and transition weights using backward support; drop empties.
        trim_by_backward_support_inplace(&mut work, &backward);

        // 6) Restrict again to reachable (trimming can disconnect).
        work = restrict_to_reachable(&work);

        // 7) Topo order again (graph may have changed).
        let topo = topo_order_or_err(&work)?;

        // 8) Bottom-up (reverse topo) register minimization by signature.
        let n = work.states.len();
        let mut old_to_new: Vec<StateID> = vec![usize::MAX; n];

        let mut sig2rep: HashMap<StateSig, StateID> = HashMap::new();
        let mut new_states: Vec<DWAState> = Vec::new();

        for &u in topo.iter().rev() {
            let st = &work.states[u];

            // final weight signature: None == empty behavior on ε for this model.
            let final_w = st.final_weight.clone().filter(|w| !w.is_empty());

            let mut trans: Vec<TransSig> = Vec::with_capacity(st.transitions.len());
            for (&lbl, &to) in &st.transitions {
                let w = st
                    .trans_weights
                    .get(&lbl)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                if w.is_empty() {
                    continue; // should already be cleaned, but safe
                }

                let to_new = old_to_new[to];
                debug_assert!(to_new != usize::MAX, "topo ensures successors processed");

                trans.push(TransSig { lbl, to: to_new, w });
            }
            // Canonical order
            trans.sort_by_key(|t| t.lbl);

            let sig = StateSig { final_w, trans };

            if let Some(&rep) = sig2rep.get(&sig) {
                old_to_new[u] = rep;
            } else {
                let rep = new_states.len();
                sig2rep.insert(sig.clone(), rep);

                // Build representative state from signature.
                let mut transitions = BTreeMap::new();
                let mut trans_weights = BTreeMap::new();
                for t in sig.trans {
                    transitions.insert(t.lbl, t.to);
                    trans_weights.insert(t.lbl, t.w);
                }

                new_states.push(DWAState {
                    transitions,
                    final_weight: sig.final_w,
                    trans_weights,
                });

                old_to_new[u] = rep;
            }
        }

        let new_start = old_to_new[work.body.start_state];

        let mut out = DWA {
            states: DWAStates(new_states),
            body: DWABody { start_state: new_start },
        };

        // 9) Final cleanup: remove unreachable reps (shouldn’t happen, but cheap) and
        // strip any stray trans_weights without transitions.
        clean_dwa_inplace(&mut out);
        out = restrict_to_reachable(&out);

        Ok(out)
    }
}

// --------------------------- helpers ---------------------------

fn clean_dwa_inplace(dwa: &mut DWA) {
    let n = dwa.states.len();
    if n == 0 {
        return;
    }
    if dwa.body.start_state >= n {
        // leave it; caller will error
    }

    for s in 0..n {
        // Clean final
        if let Some(fw) = &dwa.states[s].final_weight {
            if fw.is_empty() {
                dwa.states[s].final_weight = None;
            }
        }

        // Remove out-of-bounds transitions and empty weights
        let labels: Vec<Label> = dwa.states[s].transitions.keys().copied().collect();
        for lbl in labels {
            let to = match dwa.states[s].transitions.get(&lbl).copied() {
                Some(t) => t,
                None => continue,
            };
            let w = dwa.states[s]
                .trans_weights
                .get(&lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            if to >= n || w.is_empty() {
                dwa.states[s].transitions.remove(&lbl);
                dwa.states[s].trans_weights.remove(&lbl);
            } else {
                // Ensure a weight exists for each transition
                dwa.states[s].trans_weights.insert(lbl, w);
            }
        }

        // Remove any trans_weights entries that don't have a transition.
        let weight_labels: Vec<Label> = dwa.states[s].trans_weights.keys().copied().collect();
        for lbl in weight_labels {
            if !dwa.states[s].transitions.contains_key(&lbl) {
                dwa.states[s].trans_weights.remove(&lbl);
            }
        }
    }
}

fn restrict_to_reachable(dwa: &DWA) -> DWA {
    let n = dwa.states.len();
    if n == 0 || dwa.body.start_state >= n {
        return dwa.clone();
    }

    let mut seen = vec![false; n];
    let mut q = VecDeque::new();
    seen[dwa.body.start_state] = true;
    q.push_back(dwa.body.start_state);

    while let Some(u) = q.pop_front() {
        for &v in dwa.states[u].transitions.values() {
            if v < n && !seen[v] {
                seen[v] = true;
                q.push_back(v);
            }
        }
    }

    // Compact
    let mut map = vec![usize::MAX; n];
    let mut rev = Vec::new();
    for i in 0..n {
        if seen[i] {
            map[i] = rev.len();
            rev.push(i);
        }
    }
    let new_n = rev.len();
    if new_n == n {
        return dwa.clone();
    }

    let mut new_states: Vec<DWAState> = Vec::with_capacity(new_n);
    for &old in &rev {
        let old_state = &dwa.states[old];

        let mut transitions = BTreeMap::new();
        let mut trans_weights = BTreeMap::new();

        for (&lbl, &to_old) in &old_state.transitions {
            if to_old >= n {
                continue;
            }
            if !seen[to_old] {
                continue;
            }
            let to_new = map[to_old];

            let w = old_state
                .trans_weights
                .get(&lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            if w.is_empty() {
                continue;
            }

            transitions.insert(lbl, to_new);
            trans_weights.insert(lbl, w);
        }

        new_states.push(DWAState {
            transitions,
            final_weight: old_state.final_weight.clone().filter(|w| !w.is_empty()),
            trans_weights,
        });
    }

    let new_start = map[dwa.body.start_state];
    DWA {
        states: DWAStates(new_states),
        body: DWABody { start_state: new_start },
    }
}

fn topo_order_or_err(dwa: &DWA) -> Result<Vec<StateID>, DWAMinimizeError> {
    let n = dwa.states.len();
    let mut indeg = vec![0usize; n];
    for u in 0..n {
        for &v in dwa.states[u].transitions.values() {
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
        for &v in dwa.states[u].transitions.values() {
            if v >= n {
                continue;
            }
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }

    if topo.len() != n {
        return Err(DWAMinimizeError::Cyclic);
    }
    Ok(topo)
}

fn compute_backward_support(dwa: &DWA, topo: &[StateID]) -> Vec<Weight> {
    let n = dwa.states.len();
    let mut backward: Vec<Weight> = vec![Weight::zeros(); n];

    for &u in topo.iter().rev() {
        let mut bw = dwa.states[u]
            .final_weight
            .clone()
            .unwrap_or_else(Weight::zeros);

        for (&lbl, &v) in &dwa.states[u].transitions {
            if v >= n {
                continue;
            }
            let w = dwa.states[u]
                .trans_weights
                .get(&lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            let contrib = &w & &backward[v];
            bw |= &contrib;
        }

        backward[u] = bw;
    }

    backward
}

fn trim_by_backward_support_inplace(dwa: &mut DWA, backward: &[Weight]) {
    let n = dwa.states.len();

    for u in 0..n {
        // finals
        if let Some(fw) = &dwa.states[u].final_weight {
            let mut new_fw = fw.clone();
            new_fw &= &backward[u];
            if new_fw.is_empty() {
                dwa.states[u].final_weight = None;
            } else {
                dwa.states[u].final_weight = Some(new_fw);
            }
        }

        // transitions
        let labels: Vec<Label> = dwa.states[u].transitions.keys().copied().collect();
        for lbl in labels {
            let v = match dwa.states[u].transitions.get(&lbl).copied() {
                Some(t) => t,
                None => continue,
            };
            if v >= n {
                dwa.states[u].transitions.remove(&lbl);
                dwa.states[u].trans_weights.remove(&lbl);
                continue;
            }

            let w = dwa.states[u]
                .trans_weights
                .get(&lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            let mut new_w = w;
            new_w &= &backward[v];

            if new_w.is_empty() {
                dwa.states[u].transitions.remove(&lbl);
                dwa.states[u].trans_weights.remove(&lbl);
            } else {
                dwa.states[u].trans_weights.insert(lbl, new_w);
            }
        }

        // remove stray weights (shouldn’t exist after our cleaning rules)
        let weight_labels: Vec<Label> = dwa.states[u].trans_weights.keys().copied().collect();
        for lbl in weight_labels {
            if !dwa.states[u].transitions.contains_key(&lbl) {
                dwa.states[u].trans_weights.remove(&lbl);
            }
        }
    }
}