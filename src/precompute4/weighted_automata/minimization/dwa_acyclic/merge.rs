//! Overlap-compatible merging for acyclic DWA minimization.
//!
//! After edge trimming, states with the same transition structure merge.
//! When merging, union their final weights and transition weights.

use crate::precompute4::weighted_automata::common::Weight;
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap};

/// State signature for structure-based merging.
/// 
/// After edge trimming, we compare only transition STRUCTURE (labels and target classes),
/// not the actual weights. States with identical structure merge with weight union.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StructureSignature {
    /// Is this state final? (bool, not the actual weight)
    is_final: bool,
    /// Sorted list of (label, target_class) - NOT weight!
    transitions: Vec<(i32, usize)>,
}

impl DWA {
    /// Merge states with identical structure (same labels to equivalent targets).
    /// When merged, union their final and transition weights.
    pub fn merge_by_signature(&mut self, _live: &[Weight]) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Get reverse topological order (sinks first)
        let topo_order = self.reverse_topological_order();

        // class[q] = canonical class ID for state q
        let mut class: Vec<usize> = (0..n).collect();

        // Map from signature to canonical state ID
        let mut sig_to_class: HashMap<StructureSignature, usize> = HashMap::new();

        // Process bottom-up
        for &q in &topo_order {
            let sig = self.compute_structure_signature(q, &class);

            if let Some(&existing_class) = sig_to_class.get(&sig) {
                // Merge q into existing class
                class[q] = existing_class;
            } else {
                // q is a new class
                sig_to_class.insert(sig, q);
                class[q] = q;
            }
        }

        // Count distinct classes (representatives)
        // A state q is a representative if class[q] == q
        let num_classes = class.iter().enumerate().filter(|&(q, &c)| q == c).count();
        if num_classes == n {
            return; // No merging possible
        }

        // Rebuild the DWA with merged states, unioning weights
        self.rebuild_merged_with_union(&class);
    }

    /// Compute the structure signature (just transition structure + finality).
    fn compute_structure_signature(&self, q: usize, class: &[usize]) -> StructureSignature {
        let is_final = self.states[q].final_weight.is_some();
        
        let mut transitions: Vec<(i32, usize)> = Vec::new();
        for (&label, &target) in &self.states[q].transitions {
            let target_class = if target < class.len() {
                class[target]
            } else {
                target
            };
            transitions.push((label, target_class));
        }
        transitions.sort_by_key(|(l, _)| *l);

        StructureSignature { is_final, transitions }
    }

    /// Rebuild the DWA after merging, unioning weights for merged states.
    fn rebuild_merged_with_union(&mut self, class: &[usize]) {
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

        // First pass: collect states to merge into each representative
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); n];
        for q in 0..n {
            let rep = class[q];
            members[rep].push(q);
        }

        // Build new states with unioned weights
        let mut new_states: Vec<DWAState> = Vec::with_capacity(new_idx);
        for rep in 0..n {
            if !is_rep[rep] {
                continue;
            }

            // Start with the representative's state
            let mut new_state = DWAState {
                final_weight: None,
                transitions: BTreeMap::new(),
                trans_weights: BTreeMap::new(),
                state_weight: self.states[rep].state_weight.clone(),
            };

            // Union in all members' weights
            for &member in &members[rep] {
                let old_state = &self.states[member];

                // Union final weights
                if let Some(ref fw) = old_state.final_weight {
                    new_state.final_weight = Some(match new_state.final_weight {
                        Some(ref existing) => existing | fw,
                        None => fw.clone(),
                    });
                }

                // Union transition weights (and collect transitions)
                for (&label, &target) in &old_state.transitions {
                    let new_target = old_to_new[target].unwrap_or(0);
                    new_state.transitions.insert(label, new_target);

                    if let Some(tw) = old_state.trans_weights.get(&label) {
                        let entry = new_state.trans_weights.entry(label).or_insert_with(Weight::zeros);
                        *entry = &*entry | tw;
                    }
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
