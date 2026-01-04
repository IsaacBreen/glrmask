// src/precompute4/weighted_automata/weight_push_merge.rs

//! Weight-push state merging optimization for DWAs.
//!
//! This module implements an optimization that merges states with identical
//! outgoing structure but different final_weights by pushing the final_weight
//! differences to incoming edges.
//!
//! ## The Problem
//!
//! After determinization, we often have states like:
//! ```text
//!        ┌─[x]─→ SINK
//!   q1 ──┴─[y]─→ SINK   (final_weight = {100})
//!
//!        ┌─[x]─→ SINK
//!   q2 ──┴─[y]─→ SINK   (final_weight = {200})
//! ```
//!
//! Standard minimization can't merge q1 and q2 because they have different final_weights.
//! But they have IDENTICAL outgoing structure!
//!
//! ## The Solution
//!
//! Push the final_weight to incoming edges:
//! ```text
//!   p1 ──[a, tw=ALL]──→ q1
//!   p2 ──[b, tw=ALL]──→ q2
//!
//! Becomes:
//!   p1 ──[a, tw={100}]──→ q_merged
//!   p2 ──[b, tw={200}]──→ q_merged
//!
//!   q_merged has final_weight = {100} ∪ {200}
//! ```
//!
//! ## Correctness
//!
//! For any path ending at q1 with accumulated weight w:
//! - Before: path weight = w ∩ final_weight(q1)
//! - After:  path weight = (w ∩ fw(q1)) ∩ final_weight(q_merged) = w ∩ fw(q1)
//!
//! The intersection with the original final_weight on the incoming edge preserves semantics.

use std::collections::HashMap;

use super::common::{Label, Weight};
use super::dwa::DWA;

/// The signature of a state's outgoing structure.
/// Two states with the same signature can potentially be merged.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct OutgoingSignature {
    /// Sorted list of (label, target_state, trans_weight_fingerprint) tuples
    /// We use the fingerprint (fp) field which is a u64 hash of the weight
    edges: Vec<(Label, usize, u64)>,
}

impl OutgoingSignature {
    fn from_state(state: &super::dwa::DWAState) -> Self {
        let mut edges: Vec<_> = state
            .transitions
            .iter()
            .map(|(&label, &target)| {
                // Use the fingerprint for comparison - it's a cached u64 hash
                let tw_fp = state
                    .trans_weights
                    .get(&label)
                    .map(|w| w.fp)
                    .unwrap_or(0);
                (label, target, tw_fp)
            })
            .collect();
        edges.sort();
        Self { edges }
    }

    fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

impl DWA {
    /// Merge states with identical outgoing structure by pushing final_weights to incoming edges.
    ///
    /// This optimization can significantly reduce the number of states in DWAs that have
    /// many states differing only in their final_weight.
    ///
    /// Returns the number of states merged (removed).
    pub fn merge_by_weight_push(&mut self) -> usize {
        let n = self.states.len();
        if n == 0 {
            return 0;
        }

        // Phase 1: Group states by outgoing signature
        let mut sig_to_states: HashMap<OutgoingSignature, Vec<usize>> = HashMap::new();
        for sid in 0..n {
            let sig = OutgoingSignature::from_state(&self.states[sid]);
            sig_to_states.entry(sig).or_default().push(sid);
        }

        // Phase 2: Find merge candidates
        // Only states with the same outgoing signature AND non-empty outgoing can be merged
        let mut merge_groups: Vec<Vec<usize>> = Vec::new();
        for (sig, states) in sig_to_states {
            if states.len() > 1 && !sig.is_empty() {
                // All these states could be merged
                merge_groups.push(states);
            }
        }

        if merge_groups.is_empty() {
            return 0;
        }

        // Phase 3: Build the state remapping
        // For each merge group, pick the first state as the representative
        let mut state_remap: Vec<usize> = (0..n).collect();
        let mut representative_final_weights: HashMap<usize, Weight> = HashMap::new();

        for group in &merge_groups {
            let representative = group[0];

            // Compute merged final_weight = union of all final_weights in group
            let mut merged_fw = self.states[representative]
                .final_weight
                .clone()
                .unwrap_or_else(Weight::zeros);

            for &sid in &group[1..] {
                if let Some(ref fw) = self.states[sid].final_weight {
                    merged_fw = &merged_fw | fw;
                }
            }

            representative_final_weights.insert(representative, merged_fw);

            // Map all states in group to the representative
            for &sid in group {
                state_remap[sid] = representative;
            }
        }

        // Phase 4: Build reverse map (who has incoming edges to each state)
        let mut incoming: Vec<Vec<(usize, Label)>> = vec![vec![]; n]; // incoming[target] = [(source, label), ...]
        for sid in 0..n {
            for (&label, &target) in &self.states[sid].transitions {
                incoming[target].push((sid, label));
            }
        }

        // Phase 5: Adjust incoming edges
        // For each state that's being merged away, push its final_weight to incoming edges
        for group in &merge_groups {
            let representative = group[0];

            for &sid in group {
                if sid == representative {
                    continue; // Representative keeps its own structure
                }

                let fw = self.states[sid].final_weight.clone();

                // For each incoming edge to sid, redirect to representative
                // AND intersect the trans_weight with sid's final_weight
                for &(source, label) in &incoming[sid] {
                    // Skip if source is also being merged away (will be handled when source is processed)
                    if state_remap[source] != source {
                        continue;
                    }

                    // Get current trans_weight
                    let current_tw = self.states[source]
                        .trans_weights
                        .get(&label)
                        .cloned()
                        .unwrap_or_else(Weight::all);

                    // Compute new trans_weight = current_tw ∩ final_weight(sid)
                    let new_tw = if let Some(ref f) = fw {
                        &current_tw & f
                    } else {
                        // If no final_weight, this path doesn't contribute to acceptance
                        // The edge could be removed, but let's keep it with current weight
                        current_tw.clone()
                    };

                    // Update edge to point to representative with new weight
                    self.states[source].transitions.insert(label, representative);
                    self.states[source].trans_weights.insert(label, new_tw);
                }
            }
        }

        // Phase 6: Update representative final_weights
        for (rep, merged_fw) in representative_final_weights {
            self.states[rep].final_weight = Some(merged_fw);
        }

        // Phase 7: Update all transitions to use remapped targets
        for sid in 0..n {
            if state_remap[sid] != sid {
                // This state is being merged away, clear it
                self.states[sid].transitions.clear();
                self.states[sid].trans_weights.clear();
                self.states[sid].final_weight = None;
                continue;
            }

            // Update targets
            let trans: Vec<_> = self.states[sid]
                .transitions
                .iter()
                .map(|(&l, &t)| (l, t))
                .collect();
            for (label, target) in trans {
                let new_target = state_remap[target];
                if new_target != target {
                    self.states[sid].transitions.insert(label, new_target);
                }
            }
        }

        // Phase 8: Compact the DWA by removing empty states
        self.remove_unreachable_states();

        // Return number of states merged
        merge_groups.iter().map(|g| g.len() - 1).sum()
    }

    /// Remove states that are not reachable from the start state.
    fn remove_unreachable_states(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // BFS from start to find reachable states
        let start = self.body.start_state;
        let mut reachable = vec![false; n];
        let mut queue = vec![start];
        reachable[start] = true;

        while let Some(sid) = queue.pop() {
            for &target in self.states[sid].transitions.values() {
                if !reachable[target] {
                    reachable[target] = true;
                    queue.push(target);
                }
            }
        }

        // Build compact mapping
        let mut new_id = vec![usize::MAX; n];
        let mut next_id = 0;
        for sid in 0..n {
            if reachable[sid] {
                new_id[sid] = next_id;
                next_id += 1;
            }
        }

        if next_id == n {
            return; // Nothing to remove
        }

        // Build new states vector
        let mut new_states = Vec::with_capacity(next_id);
        for sid in 0..n {
            if reachable[sid] {
                let mut state = std::mem::take(&mut self.states.0[sid]);
                // Remap transitions
                let trans: Vec<_> = state.transitions.iter().map(|(&l, &t)| (l, t)).collect();
                state.transitions.clear();
                for (label, target) in trans {
                    state.transitions.insert(label, new_id[target]);
                }
                new_states.push(state);
            }
        }

        self.states.0 = new_states;
        self.body.start_state = new_id[start];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precompute4::weighted_automata::nwa::NWA;

    /// Test the basic merge case: two states with same outgoing, different final_weights.
    #[test]
    fn test_basic_merge() {
        let mut nwa = NWA::new();
        nwa.states.0.clear();

        let s0 = nwa.states.add_state(); // start
        let s1 = nwa.states.add_state(); // after 'a'
        let s2 = nwa.states.add_state(); // after 'b'
        let s3 = nwa.states.add_state(); // target from s1 (fw=100)
        let s4 = nwa.states.add_state(); // target from s2 (fw=200)
        let s5 = nwa.states.add_state(); // final sink

        nwa.body.start_states = vec![s0];

        let all = Weight::all();
        let w100 = Weight::from_item(100);
        let w200 = Weight::from_item(200);

        let a: Label = 97;
        let b: Label = 98;
        let x: Label = 120;

        nwa.add_transition(s0, a, s1, all.clone()).unwrap();
        nwa.add_transition(s0, b, s2, all.clone()).unwrap();
        nwa.add_transition(s1, x, s3, all.clone()).unwrap();
        nwa.add_transition(s2, x, s4, all.clone()).unwrap();
        nwa.add_transition(s3, x, s5, all.clone()).unwrap();
        nwa.add_transition(s4, x, s5, all.clone()).unwrap();

        nwa.states[s3].final_weight = Some(w100.clone());
        nwa.states[s4].final_weight = Some(w200.clone());
        nwa.states[s5].final_weight = Some(all.clone());

        let mut dwa = nwa.determinize();

        println!("Before merge:");
        println!("  States: {}", dwa.states.len());
        println!("  Transitions: {}", dwa.states.num_transitions());

        // States s3 and s4 should have identical outgoing (both -> s5 on 'x')
        // But different final_weights

        let merged = dwa.merge_by_weight_push();

        println!("\nAfter merge:");
        println!("  States: {}", dwa.states.len());
        println!("  Transitions: {}", dwa.states.num_transitions());
        println!("  Merged: {} states", merged);

        // We should have merged at least one state
        assert!(merged > 0, "Should have merged at least one state");
    }

    /// Test that the merged DWA has correct semantics.
    #[test]
    fn test_merge_preserves_semantics() {
        let mut nwa = NWA::new();
        nwa.states.0.clear();

        let s0 = nwa.states.add_state();
        let s1 = nwa.states.add_state();
        let s2 = nwa.states.add_state();
        let s3 = nwa.states.add_state();
        let s4 = nwa.states.add_state();
        let s5 = nwa.states.add_state();

        nwa.body.start_states = vec![s0];

        let all = Weight::all();
        let w100 = Weight::from_item(100);
        let w200 = Weight::from_item(200);

        let a: Label = 97;
        let b: Label = 98;
        let x: Label = 120;

        nwa.add_transition(s0, a, s1, all.clone()).unwrap();
        nwa.add_transition(s0, b, s2, all.clone()).unwrap();
        nwa.add_transition(s1, x, s3, all.clone()).unwrap();
        nwa.add_transition(s2, x, s4, all.clone()).unwrap();
        nwa.add_transition(s3, x, s5, all.clone()).unwrap();
        nwa.add_transition(s4, x, s5, all.clone()).unwrap();

        nwa.states[s3].final_weight = Some(w100.clone());
        nwa.states[s4].final_weight = Some(w200.clone());
        nwa.states[s5].final_weight = Some(all.clone());

        let mut dwa = nwa.determinize();

        // Debug: print DWA structure
        println!("DWA structure before merge:\n{}", dwa);

        // Compute path weights before merge
        // Path "ax": should have weight intersecting with {100}
        // Path "bx": should have weight intersecting with {200}

        fn trace_path(dwa: &DWA, path: &[Label]) -> Weight {
            let mut state = dwa.body.start_state;
            let mut weight = Weight::all();

            for &label in path {
                if let Some(&target) = dwa.states[state].transitions.get(&label) {
                    if let Some(tw) = dwa.states[state].trans_weights.get(&label) {
                        weight = &weight & tw;
                    }
                    state = target;
                } else {
                    return Weight::zeros();
                }
            }

            // Intersect with final weight
            if let Some(ref fw) = dwa.states[state].final_weight {
                weight = &weight & fw;
            } else {
                return Weight::zeros();
            }

            weight
        }

        let path_ax_before = trace_path(&dwa, &[a, x]);
        let path_bx_before = trace_path(&dwa, &[b, x]);

        println!("Before merge:");
        println!("  Path 'ax' weight: {:?}", path_ax_before.rsb.iter().collect::<Vec<_>>());
        println!("  Path 'bx' weight: {:?}", path_bx_before.rsb.iter().collect::<Vec<_>>());

        dwa.merge_by_weight_push();

        println!("DWA structure after merge:\n{}", dwa);

        let path_ax_after = trace_path(&dwa, &[a, x]);
        let path_bx_after = trace_path(&dwa, &[b, x]);

        println!("\nAfter merge:");
        println!("  Path 'ax' weight: {:?}", path_ax_after.rsb.iter().collect::<Vec<_>>());
        println!("  Path 'bx' weight: {:?}", path_bx_after.rsb.iter().collect::<Vec<_>>());

        // Path weights should be preserved
        assert_eq!(
            path_ax_before.rsb.iter().collect::<Vec<_>>(),
            path_ax_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'ax' weight should be preserved"
        );
        assert_eq!(
            path_bx_before.rsb.iter().collect::<Vec<_>>(),
            path_bx_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'bx' weight should be preserved"
        );
    }

    /// Test with multiple states that can be merged.
    #[test]
    fn test_multiple_merges() {
        let mut nwa = NWA::new();
        nwa.states.0.clear();

        let s0 = nwa.states.add_state();
        let mut field_states = vec![];
        let mut colon_states = vec![];

        let num_fields = 5;

        for _ in 0..num_fields {
            field_states.push(nwa.states.add_state());
            colon_states.push(nwa.states.add_state());
        }

        let sink = nwa.states.add_state();

        nwa.body.start_states = vec![s0];

        let all = Weight::all();

        // 0 -> field_i on label i
        for i in 0..num_fields {
            nwa.add_transition(s0, i as Label, field_states[i], all.clone())
                .unwrap();
        }

        // field_i -> colon_i on ':'
        let colon: Label = 58;
        for i in 0..num_fields {
            nwa.add_transition(field_states[i], colon, colon_states[i], all.clone())
                .unwrap();
        }

        // All colon_i -> sink on 'x' with same structure
        let x: Label = 120;
        for i in 0..num_fields {
            nwa.add_transition(colon_states[i], x, sink, all.clone())
                .unwrap();
        }

        // Each colon_i has different final_weight
        for i in 0..num_fields {
            nwa.states[colon_states[i]].final_weight = Some(Weight::from_item(i * 10));
        }
        nwa.states[sink].final_weight = Some(all.clone());

        let mut dwa = nwa.determinize();

        println!("Before merge:");
        println!("  States: {}", dwa.states.len());
        println!("  Transitions: {}", dwa.states.num_transitions());

        let merged = dwa.merge_by_weight_push();

        println!("\nAfter merge:");
        println!("  States: {}", dwa.states.len());
        println!("  Transitions: {}", dwa.states.num_transitions());
        println!("  Merged: {} states", merged);

        // Should have merged all colon_states into one
        assert!(
            merged >= num_fields - 1,
            "Should have merged at least {} states, but only merged {}",
            num_fields - 1,
            merged
        );
    }

    /// Test the three-branch structure from user request.
    ///
    /// ```text
    /// START{fw: []} -{s=a, w: [0,1,2,3]}-> A1{fw: []} -{s=d, w: [0,1,2,3]}-> A2{fw: [0]} -{s=d, w: [0,1,2,3]}-> END{fw: [3]}
    /// START{fw: []} -{s=b, w: [0,1,2,3]}-> B1{fw: []} -{s=d, w: [0,1,2,3]}-> B2{fw: [1]} -{s=d, w: [0,1,2,3]}-> END{fw: [3]}
    /// START{fw: []} -{s=c, w: [0,1,2,3]}-> C1{fw: []} -{s=d, w: [0,1,2,3]}-> C2{fw: [2]} -{s=d, w: [0,1,2,3]}-> END{fw: [3]}
    /// ```
    ///
    /// After weight pushing and minimization, this should become:
    ///
    /// ```text
    /// START{fw: []} -{s=a, w: [0,3]}-> A1{fw: []}
    /// START{fw: []} -{s=b, w: [1,3]}-> A1{fw: []}
    /// START{fw: []} -{s=c, w: [2,3]}-> A1{fw: []}
    /// A1{fw: []} -{s=d, w: [0,1,2,3]}-> A2{fw: [0,1,2]} -{s=d, w: [0,1,2,3]}-> END{fw: [3]}
    /// ```
    #[test]
    fn test_three_branch_weight_push_merge() {
        let mut nwa = NWA::new();
        nwa.states.0.clear();

        // Create states
        let start = nwa.states.add_state(); // START
        let a1 = nwa.states.add_state();    // A1
        let b1 = nwa.states.add_state();    // B1
        let c1 = nwa.states.add_state();    // C1
        let a2 = nwa.states.add_state();    // A2 (fw: [0])
        let b2 = nwa.states.add_state();    // B2 (fw: [1])
        let c2 = nwa.states.add_state();    // C2 (fw: [2])
        let end = nwa.states.add_state();   // END (fw: [3])

        nwa.body.start_states = vec![start];

        // Labels
        let label_a: Label = 97; // 'a'
        let label_b: Label = 98; // 'b'
        let label_c: Label = 99; // 'c'
        let label_d: Label = 100; // 'd'

        // Weight: [0,1,2,3] = from_iter([0,1,2,3])
        let w_all: Weight = [0usize, 1, 2, 3].into_iter().collect();
        let w_0 = Weight::from_item(0);
        let w_1 = Weight::from_item(1);
        let w_2 = Weight::from_item(2);
        let w_3 = Weight::from_item(3);

        // START -> A1/B1/C1
        nwa.add_transition(start, label_a, a1, w_all.clone()).unwrap();
        nwa.add_transition(start, label_b, b1, w_all.clone()).unwrap();
        nwa.add_transition(start, label_c, c1, w_all.clone()).unwrap();

        // A1/B1/C1 -> A2/B2/C2
        nwa.add_transition(a1, label_d, a2, w_all.clone()).unwrap();
        nwa.add_transition(b1, label_d, b2, w_all.clone()).unwrap();
        nwa.add_transition(c1, label_d, c2, w_all.clone()).unwrap();

        // A2/B2/C2 -> END
        nwa.add_transition(a2, label_d, end, w_all.clone()).unwrap();
        nwa.add_transition(b2, label_d, end, w_all.clone()).unwrap();
        nwa.add_transition(c2, label_d, end, w_all.clone()).unwrap();

        // Set final weights
        nwa.states[a2].final_weight = Some(w_0.clone());
        nwa.states[b2].final_weight = Some(w_1.clone());
        nwa.states[c2].final_weight = Some(w_2.clone());
        nwa.states[end].final_weight = Some(w_3.clone());

        // Determinize the NWA
        let mut dwa = nwa.determinize();

        // Debug: print DWA structure
        println!("DWA structure before merge:");
        println!("  States: {}", dwa.states.len());
        println!("  Transitions: {}", dwa.states.num_transitions());
        for (sid, state) in dwa.states.iter().enumerate() {
            println!("  State {}: fw={:?}", sid, state.final_weight.as_ref().map(|w| w.rsb.iter().collect::<Vec<_>>()));
            for (&label, &target) in &state.transitions {
                let tw = state.trans_weights.get(&label).map(|w| w.rsb.iter().collect::<Vec<_>>());
                println!("    -{} w={:?}-> {}", char::from_u32(label as u32).unwrap_or('?'), tw, target);
            }
        }

        // Helper function to trace path and compute weight
        fn trace_path(dwa: &DWA, path: &[Label]) -> Weight {
            let mut state = dwa.body.start_state;
            let mut weight = Weight::all();

            for &label in path {
                if let Some(&target) = dwa.states[state].transitions.get(&label) {
                    if let Some(tw) = dwa.states[state].trans_weights.get(&label) {
                        weight = &weight & tw;
                    }
                    state = target;
                } else {
                    return Weight::zeros();
                }
            }

            // Intersect with final weight
            if let Some(ref fw) = dwa.states[state].final_weight {
                weight = &weight & fw;
            } else {
                return Weight::zeros();
            }

            weight
        }

        // Record path weights before merge
        let path_add_before = trace_path(&dwa, &[label_a, label_d, label_d]);
        let path_bdd_before = trace_path(&dwa, &[label_b, label_d, label_d]);
        let path_cdd_before = trace_path(&dwa, &[label_c, label_d, label_d]);
        let path_ad_before = trace_path(&dwa, &[label_a, label_d]);
        let path_bd_before = trace_path(&dwa, &[label_b, label_d]);
        let path_cd_before = trace_path(&dwa, &[label_c, label_d]);

        println!("\nPath weights before merge:");
        println!("  'add' (to END): {:?}", path_add_before.rsb.iter().collect::<Vec<_>>());
        println!("  'bdd' (to END): {:?}", path_bdd_before.rsb.iter().collect::<Vec<_>>());
        println!("  'cdd' (to END): {:?}", path_cdd_before.rsb.iter().collect::<Vec<_>>());
        println!("  'ad' (to A2): {:?}", path_ad_before.rsb.iter().collect::<Vec<_>>());
        println!("  'bd' (to B2): {:?}", path_bd_before.rsb.iter().collect::<Vec<_>>());
        println!("  'cd' (to C2): {:?}", path_cd_before.rsb.iter().collect::<Vec<_>>());

        let states_before = dwa.states.len();
        let transitions_before = dwa.states.num_transitions();

        // Apply weight-push merge
        let merged = dwa.merge_by_weight_push();

        println!("\nDWA structure after merge:");
        println!("  States: {} (was {})", dwa.states.len(), states_before);
        println!("  Transitions: {} (was {})", dwa.states.num_transitions(), transitions_before);
        println!("  Merged: {} states", merged);
        for (sid, state) in dwa.states.iter().enumerate() {
            println!("  State {}: fw={:?}", sid, state.final_weight.as_ref().map(|w| w.rsb.iter().collect::<Vec<_>>()));
            for (&label, &target) in &state.transitions {
                let tw = state.trans_weights.get(&label).map(|w| w.rsb.iter().collect::<Vec<_>>());
                println!("    -{} w={:?}-> {}", char::from_u32(label as u32).unwrap_or('?'), tw, target);
            }
        }

        // Record path weights after merge
        let path_add_after = trace_path(&dwa, &[label_a, label_d, label_d]);
        let path_bdd_after = trace_path(&dwa, &[label_b, label_d, label_d]);
        let path_cdd_after = trace_path(&dwa, &[label_c, label_d, label_d]);
        let path_ad_after = trace_path(&dwa, &[label_a, label_d]);
        let path_bd_after = trace_path(&dwa, &[label_b, label_d]);
        let path_cd_after = trace_path(&dwa, &[label_c, label_d]);

        println!("\nPath weights after merge:");
        println!("  'add' (to END): {:?}", path_add_after.rsb.iter().collect::<Vec<_>>());
        println!("  'bdd' (to END): {:?}", path_bdd_after.rsb.iter().collect::<Vec<_>>());
        println!("  'cdd' (to END): {:?}", path_cdd_after.rsb.iter().collect::<Vec<_>>());
        println!("  'ad' (to merged A2/B2/C2): {:?}", path_ad_after.rsb.iter().collect::<Vec<_>>());
        println!("  'bd' (to merged A2/B2/C2): {:?}", path_bd_after.rsb.iter().collect::<Vec<_>>());
        println!("  'cd' (to merged A2/B2/C2): {:?}", path_cd_after.rsb.iter().collect::<Vec<_>>());

        // Verify semantics are preserved
        // Paths to END should have weight [3]
        assert_eq!(
            path_add_before.rsb.iter().collect::<Vec<_>>(),
            path_add_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'add' weight should be preserved"
        );
        assert_eq!(
            path_bdd_before.rsb.iter().collect::<Vec<_>>(),
            path_bdd_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'bdd' weight should be preserved"
        );
        assert_eq!(
            path_cdd_before.rsb.iter().collect::<Vec<_>>(),
            path_cdd_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'cdd' weight should be preserved"
        );

        // Paths to intermediate states should also be preserved
        assert_eq!(
            path_ad_before.rsb.iter().collect::<Vec<_>>(),
            path_ad_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'ad' weight should be preserved"
        );
        assert_eq!(
            path_bd_before.rsb.iter().collect::<Vec<_>>(),
            path_bd_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'bd' weight should be preserved"
        );
        assert_eq!(
            path_cd_before.rsb.iter().collect::<Vec<_>>(),
            path_cd_after.rsb.iter().collect::<Vec<_>>(),
            "Path 'cd' weight should be preserved"
        );

        // We should have merged A2/B2/C2 (which have identical outgoing but different final_weights)
        // Original: start, a1, b1, c1, a2, b2, c2, end = 8 states
        // Merged: A1/B1/C1 should also be mergeable if they have identical outgoing
        // Expected: start, merged_a1_b1_c1, merged_a2_b2_c2, end = 4 states
        // Or at minimum: start, a1, b1, c1, merged_a2_b2_c2, end = 6 states

        println!("\nExpected merges:");
        println!("  A2, B2, C2 have identical outgoing (->END on 'd') but different fw [0], [1], [2]");
        println!("  They should merge into one state with fw [0,1,2]");
        println!("  A1, B1, C1 also have identical outgoing (->merged on 'd')");
        println!("  After first merge, they may become identical and merge further");

        // At minimum, we should have merged a2/b2/c2 (3 -> 1 = 2 merges)
        assert!(
            merged >= 2,
            "Should have merged at least 2 states (a2, b2, c2 -> 1), but only merged {}",
            merged
        );

        // After merging, we should have fewer states
        assert!(
            dwa.states.len() < states_before,
            "Should have fewer states after merge: {} >= {}",
            dwa.states.len(),
            states_before
        );
    }
}
