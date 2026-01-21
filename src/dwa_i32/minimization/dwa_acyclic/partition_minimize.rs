//! Partition-based DWA minimization using signatures.
//!
//! This algorithm computes per-state signatures based on weight partitions,
//! then uses these signatures to efficiently detect incompatibility.

use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::{DWA, DWABuildError, DWAState, DWAStates};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// A partition element - represents a region in weight space
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
struct PartitionElement(usize);

/// Signature for a state: maps partition elements to behavior hashes
#[derive(Debug, Clone, Default)]
struct StateSignature {
    /// For each partition element in this state's domain, what is its behavior hash?
    behavior: BTreeMap<PartitionElement, u64>,
}

impl StateSignature {
    fn domain(&self) -> impl Iterator<Item = &PartitionElement> {
        self.behavior.keys()
    }
    
    fn get(&self, elem: &PartitionElement) -> Option<u64> {
        self.behavior.get(elem).copied()
    }
    
    fn insert(&mut self, elem: PartitionElement, hash: u64) {
        self.behavior.insert(elem, hash);
    }
    
    /// Combine behavior hashes (for accumulating across transitions)
    fn combine_hash(&mut self, elem: PartitionElement, hash: u64) {
        self.behavior.entry(elem).and_modify(|h| {
            // Combine hashes - order-independent
            *h ^= hash.rotate_left(17);
        }).or_insert(hash);
    }
}

/// Compute the weight partition from all weights in the DWA.
/// Uses weight fingerprints as partition elements.
fn compute_partition(dwa: &DWA) -> (usize, HashMap<u64, PartitionElement>) {
    let mut fp_to_elem: HashMap<u64, PartitionElement> = HashMap::new();
    let mut next_idx = 0;
    
    // Collect all unique weight fingerprints
    for id in 0..dwa.states.len() {
        let state = &dwa.states[id];
        
        if let Some(fw) = &state.final_weight {
            let fw_fp = fw.fingerprint();
            fp_to_elem.entry(fw_fp).or_insert_with(|| {
                let elem = PartitionElement(next_idx);
                next_idx += 1;
                elem
            });
        }
        
        for tw in state.trans_weights.values() {
            let tw_fp = tw.fingerprint();
            fp_to_elem.entry(tw_fp).or_insert_with(|| {
                let elem = PartitionElement(next_idx);
                next_idx += 1;
                elem
            });
        }
    }
    
    (next_idx, fp_to_elem)
}

/// Compute state signatures via DFS traversal
fn compute_signatures(
    dwa: &DWA,
    fp_to_elem: &HashMap<u64, PartitionElement>,
) -> Vec<StateSignature> {
    let n = dwa.states.len();
    let mut signatures: Vec<Option<StateSignature>> = vec![None; n];
    
    // Memoized recursive signature computation
    fn compute_sig_recursive(
        state_id: StateID,
        dwa: &DWA,
        fp_to_elem: &HashMap<u64, PartitionElement>,
        signatures: &mut Vec<Option<StateSignature>>,
        in_progress: &mut HashSet<StateID>,
    ) -> StateSignature {
        // Check memo
        if let Some(sig) = &signatures[state_id] {
            return sig.clone();
        }
        
        // Cycle detection (shouldn't happen in acyclic DWA)
        if in_progress.contains(&state_id) {
            return StateSignature::default();
        }
        in_progress.insert(state_id);
        
        let state = &dwa.states[state_id];
        let mut sig = StateSignature::default();
        
        // Hash final behavior
        if let Some(fw) = &state.final_weight {
            let fw_fp = fw.fingerprint();
            if let Some(&elem) = fp_to_elem.get(&fw_fp) {
                let mut hasher = DefaultHasher::new();
                "final".hash(&mut hasher);
                fw_fp.hash(&mut hasher);
                let hash = hasher.finish();
                sig.combine_hash(elem, hash);
            }
        }
        
        // Hash transition behavior
        for (&label, &target) in &state.transitions {
            let Some(tw) = state.trans_weights.get(&label) else { continue };
            let tw_fp = tw.fingerprint();
            let Some(&tw_elem) = fp_to_elem.get(&tw_fp) else { continue };
            
            // Recursively get target's signature
            let target_sig = compute_sig_recursive(target, dwa, fp_to_elem, signatures, in_progress);
            
            // Combine target's behavior with this transition's weight
            // For each partition element the target cares about, propagate back
            for (&target_elem, &target_hash) in &target_sig.behavior {
                let mut hasher = DefaultHasher::new();
                "trans".hash(&mut hasher);
                label.hash(&mut hasher);
                target_elem.0.hash(&mut hasher);
                target_hash.hash(&mut hasher);
                tw_fp.hash(&mut hasher);
                let combined = hasher.finish();
                sig.combine_hash(tw_elem, combined);
            }
        }
        
        in_progress.remove(&state_id);
        signatures[state_id] = Some(sig.clone());
        sig
    }
    
    // Compute signatures for all states
    let mut in_progress = HashSet::new();
    for id in 0..n {
        compute_sig_recursive(id, dwa, fp_to_elem, &mut signatures, &mut in_progress);
    }
    
    signatures.into_iter().map(|s| s.unwrap_or_default()).collect()
}

/// Build incompatibility graph using signatures.
/// Two states are incompatible iff they share a partition element with different hashes.
fn build_incompatibility_from_signatures(
    signatures: &[StateSignature],
) -> Vec<Vec<usize>> {
    let n = signatures.len();
    let mut incompatible: Vec<Vec<usize>> = vec![vec![]; n];
    
    // Build inverted index: partition_element -> list of (state_id, hash)
    let mut inverted: HashMap<PartitionElement, Vec<(StateID, u64)>> = HashMap::new();
    for (id, sig) in signatures.iter().enumerate() {
        for (&elem, &hash) in &sig.behavior {
            inverted.entry(elem).or_default().push((id, hash));
        }
    }
    
    // For each partition element, find pairs with different hashes
    for (_, states_with_hashes) in &inverted {
        // Group by hash
        let mut by_hash: HashMap<u64, Vec<StateID>> = HashMap::new();
        for &(id, hash) in states_with_hashes {
            by_hash.entry(hash).or_default().push(id);
        }
        
        // States with different hashes on this partition element are incompatible
        let hash_groups: Vec<_> = by_hash.values().collect();
        for i in 0..hash_groups.len() {
            for j in (i+1)..hash_groups.len() {
                for &a in hash_groups[i] {
                    for &b in hash_groups[j] {
                        incompatible[a].push(b);
                        incompatible[b].push(a);
                    }
                }
            }
        }
    }
    
    // Deduplicate adjacency lists
    for adj in &mut incompatible {
        adj.sort_unstable();
        adj.dedup();
    }
    
    incompatible
}

/// Partition-based minimization using signatures.
pub fn minimize_partition_based(dwa: &DWA) -> Result<DWA, DWABuildError> {
    if dwa.states.len() == 0 {
        return Ok(DWA::default()); // Return empty DWA (0 states)
    }
    
    let start = std::time::Instant::now();
    let before_stats = dwa.stats();
    
    // Step 1: Compute partition and signatures
    let (num_elements, fp_to_elem) = compute_partition(dwa);
    let signatures = compute_signatures(dwa, &fp_to_elem);
    
    crate::debug!(5, "Partition minimize: {}, {} partition elements", 
        before_stats, num_elements);
    
    // Step 2: Build incompatibility graph using signatures
    let incompatible = build_incompatibility_from_signatures(&signatures);
    
    let total_edges: usize = incompatible.iter().map(|adj| adj.len()).sum();
    crate::debug!(5, "Incompatibility graph: {} nodes, {} edges", 
        signatures.len(), total_edges / 2);
    
    // Step 3: Graph coloring to find merge groups
    let coloring = crate::dwa_i32::minimization::graph_coloring::solve_greedy_coloring(&incompatible);
    let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
    
    crate::debug!(5, "Graph coloring: {} colors (was {})", num_colors, before_stats);
    
    // Step 4: Build color classes
    let mut color_classes: Vec<Vec<StateID>> = vec![vec![]; num_colors];
    let mut old_to_color: HashMap<StateID, usize> = HashMap::new();
    
    for (id, &color) in coloring.iter().enumerate() {
        color_classes[color].push(id);
        old_to_color.insert(id, color);
    }
    
    // Step 5: Build merged DWA
    let mut new_states = DWAStates::default();
    
    for class in &color_classes {
        let mut merged_final = Weight::zeros();
        let mut merged_transitions: BTreeMap<Label, StateID> = BTreeMap::new();
        let mut merged_weights: BTreeMap<Label, Weight> = BTreeMap::new();
        
        for &old_id in class {
            let old_state = &dwa.states[old_id];
            
            if let Some(fw) = &old_state.final_weight {
                merged_final |= fw;
            }
            
            for (&label, &old_target) in &old_state.transitions {
                let target_color = old_to_color[&old_target];
                merged_transitions.insert(label, target_color);
                
                if let Some(tw) = old_state.trans_weights.get(&label) {
                    merged_weights.entry(label).or_insert_with(Weight::zeros);
                    *merged_weights.get_mut(&label).unwrap() |= tw;
                }
            }
        }
        
        new_states.add_existing_state(DWAState {
            final_weight: if merged_final.is_empty() { None } else { Some(merged_final) },
            transitions: merged_transitions,
            trans_weights: merged_weights,
        });
    }
    
    let old_start = dwa.body.start_state;
    let new_start = old_to_color[&old_start];
    
    let new_dwa = DWA {
        states: new_states,
        body: crate::dwa_i32::dwa::DWABody { start_state: new_start },
    };
    crate::debug!(5, "Partition minimize: {} -> {} in {:?}", 
        before_stats, new_dwa.stats(), start.elapsed());
    
    Ok(new_dwa)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_partition_minimize_empty() {
        // Create an empty DWA by using default which creates 0 states
        let dwa = DWA::default();
        println!("Input DWA has {}", dwa.stats());
        let result = minimize_partition_based(&dwa).unwrap();
        println!("Output DWA has {}", result.stats());
        assert_eq!(result.states.len(), 0);
    }
    
    #[test]
    fn test_partition_minimize_basic() {
        // Create a simple DWA with states that should be merged
        let mut dwa = DWA::new();
        let s0 = dwa.body.start_state;
        let s1 = dwa.add_state();
        let s2 = dwa.add_state(); // Same behavior as s1, should be merged
        
        let w1 = Weight::from_item(0);
        let w2 = Weight::from_item(1);
        
        // s0 -> s1 on label 0 with weight w1
        // s0 -> s2 on label 1 with weight w1
        dwa.add_transition(s0, 0, s1, w1.clone()).unwrap();
        dwa.add_transition(s0, 1, s2, w1.clone()).unwrap();
        
        // s1 and s2 both have same final weight
        dwa.set_final_weight(s1, w2.clone()).unwrap();
        dwa.set_final_weight(s2, w2.clone()).unwrap();
        
        // Test that original DWA works as expected
        let test_words: &[&[i32]] = &[
            &[0], // Should accept with weight {0} & {1} = {}
            &[1], // Should accept with weight {0} & {1} = {}
            &[2], // Should reject
        ];
        
        println!("Original DWA behavior:");
        for word in test_words {
            let result = dwa.eval_word_weight(word);
            println!("  {:?} -> {:?}", word, result);
        }
        
        let result = minimize_partition_based(&dwa).unwrap();
        
        println!("Minimized DWA behavior:");
        for word in test_words {
            let minimized_result = result.eval_word_weight(word);
            println!("  {:?} -> {:?}", word, minimized_result);
        }
        
        // Verify semantic equivalence
        for word in test_words {
            let orig = dwa.eval_word_weight(word);
            let mini = result.eval_word_weight(word);
            assert_eq!(orig, mini, "Semantic mismatch on word {:?}: orig={:?}, mini={:?}", word, orig, mini);
        }
        
        println!("Semantic equivalence verified!");
        println!("Original: {}, Minimized: {}", dwa.stats(), result.stats());
    }
}
