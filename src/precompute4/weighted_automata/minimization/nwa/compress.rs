//! Compress transitions in NWA by merging parallel edges.

use crate::precompute4::weighted_automata::common::{Label, NWAStateID, Weight};
use crate::precompute4::weighted_automata::nwa::NWA;
use std::collections::BTreeMap;

impl NWA {
    /// Canonicalize NWA transitions by merging parallel transitions:
    ///  - For each state and epsilon edge, merge multiple (to, w) by unioning weights per `to`.
    ///  - For each state, label, and destination, merge multiple (label, to, w) by unioning weights.
    pub fn compress_transitions(&mut self) -> bool {
        crate::debug!(7, "[NWA] Compressing transitions...");
        let mut changed = false;

        for st in &mut self.states.0 {
            // Compress epsilons
            if !st.epsilons.is_empty() {
                let mut eps_map: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                for &(to, ref w) in &st.epsilons {
                    if w.is_empty() {
                        continue;
                    }
                    eps_map.entry(to).and_modify(|acc| *acc |= w).or_insert(w.clone());
                }
                if eps_map.len() != st.epsilons.len() {
                    changed = true;
                }
                st.epsilons = eps_map.into_iter().filter(|(_, w)| !w.is_empty()).collect();
            }

            // Compress labeled transitions
            if !st.transitions.is_empty() {
                let mut new_transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
                for (&lbl, targets) in &st.transitions {
                    let mut per_dest: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                    for &(to, ref w) in targets {
                        if w.is_empty() {
                            continue;
                        }
                        per_dest.entry(to).and_modify(|acc| *acc |= w).or_insert(w.clone());
                    }
                    if per_dest.len() != targets.len() {
                        changed = true;
                    }
                    let merged: Vec<(NWAStateID, Weight)> =
                        per_dest.into_iter().filter(|(_, w)| !w.is_empty()).collect();
                    if !merged.is_empty() {
                        new_transitions.insert(lbl, merged);
                    }
                }
                if new_transitions.len() != st.transitions.len() {
                    changed = true;
                }
                st.transitions = new_transitions;
            }
        }

        changed
    }
}
