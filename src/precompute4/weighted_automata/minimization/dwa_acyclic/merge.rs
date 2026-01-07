//! Overlap-compatible merging for acyclic DWA minimization.
//!
//! States can merge if they have the same transition structure AND
//! they agree on the intersection of their live sets (weights verify).
//!
//! When merging, union their final weights and transition weights.

use crate::precompute4::weighted_automata::common::Weight;
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap};

/// State signature for structure-based merging.
/// 
/// We first group states by transition STRUCTURE (labels and target classes).
/// Then within each group, we check overlap-compatibility of weights.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StructureSignature {
    /// Is this state final?
    is_final: bool,
    /// Sorted list of (label, target_class)
    transitions: Vec<(i32, usize)>,
}

impl DWA {
    /// Merge states with compatible signatures (bottom-up with weight union).
    pub fn merge_by_signature(&mut self, live: &[Weight]) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Get reverse topological order (sinks first)
        let topo_order = self.reverse_topological_order();

        // class[q] = canonical class ID for state q
        let mut class: Vec<usize> = (0..n).collect();

        // Map from signature to LIST of canonical state IDs (potential representatives)
        // We need a list because states with same structure might have incompatible weights
        let mut sig_to_classes: HashMap<StructureSignature, Vec<usize>> = HashMap::new();

        // Process bottom-up
        for &q in &topo_order {
            let sig = self.compute_structure_signature(q, &class);

            let mut merged = false;
            
            // Check against existing classes with same structure
            if let Some(existing_classes) = sig_to_classes.get_mut(&sig) {
                for &rep in existing_classes.iter() {
                    if self.can_merge_overlap(q, rep, live) {
                        class[q] = rep;
                        merged = true;
                        break;
                    }
                }
                
                if !merged {
                    // Same structure but incompatible weights -> new class
                    existing_classes.push(q);
                    class[q] = q;
                }
            } else {
                // New structure -> new class
                sig_to_classes.insert(sig, vec![q]);
                class[q] = q;
            }
        }

        // Count distinct classes
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

    /// Check if two states can merge (overlap compatible).
    /// They must agree on their shared live tokens for finals and transitions.
    fn can_merge_overlap(&self, q1: usize, q2: usize, live: &[Weight]) -> bool {
        let n = self.states.len();
        if q1 >= n || q2 >= n {
            return false;
        }

        // Compute overlap of live sets
        let live1 = &live[q1];
        let live2 = &live[q2];
        let overlap = live1 & live2;

        if overlap.is_empty() {
            // No overlap means no conflict - can merge
            return true;
        }

        // Check final weights agree on overlap
        let f1 = self.states[q1].final_weight.as_ref();
        let f2 = self.states[q2].final_weight.as_ref();
        
        match (f1, f2) {
            (Some(w1), Some(w2)) => {
                // Both final - must agree on overlap
                let f1_overlap = w1 & &overlap;
                let f2_overlap = w2 & &overlap;
                if f1_overlap != f2_overlap {
                    return false;
                }
            }
            (None, None) => {
                // Both non-final - ok
            }
            _ => {
                // One final, one not - check if overlap would be affected
                let final_w = f1.or(f2).unwrap();
                let final_overlap = final_w & &overlap;
                if !final_overlap.is_empty() {
                    return false;
                }
            }
        }

        // Check transition weights agree on overlap for each label
        // Note: Structure match guarantees keys are same
        for label in self.states[q1].transitions.keys() {
            let tw1 = self.states[q1].trans_weights.get(label);
            let tw2 = self.states[q2].trans_weights.get(label);
            
            let w1 = tw1.map(|w| w & &overlap).unwrap_or_else(|| Weight::zeros() & &overlap);
            let w2 = tw2.map(|w| w & &overlap).unwrap_or_else(|| Weight::zeros() & &overlap);
            
            if w1 != w2 {
                return false;
            }
        }

        true
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
