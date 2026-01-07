//! Bottom-up signature merging for acyclic DWA minimization.
//!
//! After normalization, states with identical signatures can be merged.
//! Signature = (final_weight, {(label, trans_weight, target_class) for each transition})

use crate::precompute4::weighted_automata::common::Weight;
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap};

/// State signature for merging.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StateSignature {
    /// Final weight (None if non-final)
    final_weight: Option<u64>, // Fingerprint of the weight
    /// Sorted list of (label, trans_weight_fingerprint, target_class)
    transitions: Vec<(i32, u64, usize)>,
}

impl DWA {
    /// Merge states with identical signatures (bottom-up).
    ///
    /// Returns true if any states were merged.
    pub fn merge_by_signature(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Get reverse topological order (sinks first)
        let topo_order = self.reverse_topological_order();

        // class[q] = canonical class ID for state q
        let mut class: Vec<usize> = (0..n).collect();

        // Map from signature to canonical state ID
        let mut sig_to_class: HashMap<StateSignature, usize> = HashMap::new();

        // Process bottom-up
        for &q in &topo_order {
            let sig = self.compute_signature(q, &class);

            if let Some(&existing_class) = sig_to_class.get(&sig) {
                // Merge q into existing class
                class[q] = existing_class;
            } else {
                // q is a new class
                sig_to_class.insert(sig, q);
                class[q] = q;
            }
        }

        // Count distinct classes
        let num_classes = sig_to_class.len();
        if num_classes == n {
            return false; // No merging possible
        }

        // Rebuild the DWA with merged states
        self.rebuild_merged(&class, num_classes);
        true
    }

    /// Compute the signature of a state given current class assignments.
    fn compute_signature(&self, q: usize, class: &[usize]) -> StateSignature {
        // Final weight fingerprint
        let final_fp = self.states[q]
            .final_weight
            .as_ref()
            .map(|w| w.fp);

        // Transitions: (label, weight_fp, target_class)
        let mut transitions: Vec<(i32, u64, usize)> = Vec::new();
        for (&label, &target) in &self.states[q].transitions {
            let weight_fp = self.states[q]
                .trans_weights
                .get(&label)
                .map(|w| w.fp)
                .unwrap_or(0);

            let target_class = if target < class.len() {
                class[target]
            } else {
                target
            };

            transitions.push((label, weight_fp, target_class));
        }
        transitions.sort_by_key(|(l, _, _)| *l);

        StateSignature {
            final_weight: final_fp,
            transitions,
        }
    }

    /// Rebuild the DWA after merging, keeping one representative per class.
    fn rebuild_merged(&mut self, class: &[usize], _num_classes: usize) {
        let n = self.states.len();

        // Determine which states are representatives (class[q] == q)
        let mut is_rep = vec![false; n];
        for q in 0..n {
            if class[q] == q {
                is_rep[q] = true;
            }
        }

        // Create mapping from old state to new state index
        let mut old_to_new: Vec<Option<usize>> = vec![None; n];
        let mut new_idx = 0;
        for q in 0..n {
            if is_rep[q] {
                old_to_new[q] = Some(new_idx);
                new_idx += 1;
            }
        }

        // For non-representatives, map to their representative's new index
        for q in 0..n {
            if !is_rep[q] {
                let rep = class[q];
                old_to_new[q] = old_to_new[rep];
            }
        }

        // Build new states
        let mut new_states: Vec<DWAState> = Vec::with_capacity(new_idx);
        for q in 0..n {
            if !is_rep[q] {
                continue;
            }

            let old_state = &self.states[q];
            let mut new_state = DWAState {
                final_weight: old_state.final_weight.clone(),
                transitions: BTreeMap::new(),
                trans_weights: BTreeMap::new(),
                state_weight: old_state.state_weight.clone(),
            };

            for (&label, &target) in &old_state.transitions {
                let new_target = old_to_new[target].unwrap_or(0);
                new_state.transitions.insert(label, new_target);
                if let Some(w) = old_state.trans_weights.get(&label) {
                    new_state.trans_weights.insert(label, w.clone());
                }
            }

            new_states.push(new_state);
        }

        // Update start state
        let new_start = old_to_new[self.body.start_state].unwrap_or(0);

        self.states = DWAStates(new_states);
        self.body.start_state = new_start;
    }
}
