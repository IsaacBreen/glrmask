use crate::precompute4::weighted_automata::*;
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeSet, HashSet};

#[cfg(test)]
mod test_minimization_failure {
    use super::*;
    
    /// Create an NWA with epsilon transitions that will produce different results
    /// with vs without rm_epsilon before determinization.
    /// 
    /// The key insight is that non-trivial epsilon weights create "fragmented" 
    /// weights during on-the-fly epsilon closure computation.
    fn create_test_nwa() -> NWA {
        let mut nwa = NWA::new_empty();  // Use new_empty to avoid extra state
        
        // Create structure:
        //   S0 (start)
        //     --eps(W_a)--> S1 --label:1(W1)--> S3 (final: W_f)
        //     --eps(W_b)--> S2 --label:1(W2)--> S3 (final: W_f)
        //
        // With non-trivial epsilon weights W_a and W_b that have DIFFERENT tokens,
        // the weight on label:1 transition differs:
        // - rm_epsilon first: combines W_a & W_1 and W_b & W_2 at the NWA level
        // - on-the-fly: may produce multiple intermediate weights
        
        let s0 = nwa.states.add_state(); // 0 - start
        let s1 = nwa.states.add_state(); // 1
        let s2 = nwa.states.add_state(); // 2
        let s3 = nwa.states.add_state(); // 3 - final
        
        // s0 is the start state
        nwa.body.start_states = vec![s0];
        
        // Epsilon weights - DIFFERENT sets of tokens allowed through each path
        // This is the key: overlapping but not identical epsilon weights
        let w_a = Weight::from_rsb(RangeSetBlaze::from_iter(0..50));    // tokens 0-49
        let w_b = Weight::from_rsb(RangeSetBlaze::from_iter(25..75));   // tokens 25-74
        
        // Labeled transition weights - BOTH should overlap with their epsilon weights
        // w_1 must intersect with w_a (0-49): use 10-30
        // w_2 must intersect with w_b (25-74): use 30-50
        // This creates overlapping but distinct combined weights
        let w_1 = Weight::from_rsb(RangeSetBlaze::from_iter(10..=30));  // tokens 10-30
        let w_2 = Weight::from_rsb(RangeSetBlaze::from_iter(30..=50));  // tokens 30-50
        
        // Final weight
        let w_f = Weight::from_rsb(RangeSetBlaze::from_iter([100]));
        
        // Build structure
        nwa.states.add_epsilon(s0, s1, w_a.clone());
        nwa.states.add_epsilon(s0, s2, w_b.clone());
        nwa.states.add_transition(s1, 1, s3, w_1.clone()).unwrap();
        nwa.states.add_transition(s2, 1, s3, w_2.clone()).unwrap();
        nwa.states[s3].final_weight = Some(w_f.clone());
        
        nwa
    }
    
    /// Count unique weights in a DWA
    fn count_unique_weights(dwa: &DWA) -> usize {
        let weights: HashSet<_> = dwa.states.0.iter()
            .flat_map(|s| s.trans_weights.values().chain(s.final_weight.iter()))
            .cloned()
            .collect();
        weights.len()
    }
    
    /// This test demonstrates that rm_epsilon before determinization produces
    /// DWAs with fewer unique weights, which enables better minimization.
    #[test]
    fn test_rm_epsilon_effect_on_weights() {
        // Enable detailed debug output
        std::env::set_var("DETERMINIZE_DEBUG_VERBOSE", "1");
        
        let nwa = create_test_nwa();
        
        println!("\n=== Original NWA ===");
        println!("States: {}", nwa.states.len());
        println!("Start states: {:?}", nwa.body.start_states);
        
        // Print NWA structure
        for i in 0..nwa.states.len() {
            let s = &nwa.states[i];
            println!("State {}:", i);
            if let Some(fw) = &s.final_weight {
                println!("  final: {:?}", fw);
            }
            for (t, w) in &s.epsilons {
                println!("  eps -> {}: {:?}", t, w);
            }
            for (lbl, targets) in &s.transitions {
                for (t, w) in targets {
                    println!("  {} -> {}: {:?}", lbl, t, w);
                }
            }
        }
        
        // Path 1: Builtin determinization (without rm_epsilon)
        println!("\n=== Path 1: Builtin determinization (no rm_epsilon) ===");
        
        // First, let's manually trace the epsilon closure computation
        {
            use std::collections::BTreeMap;
            let mut start_subset = BTreeMap::new();
            for &s in &nwa.body.start_states {
                start_subset.insert(s, Weight::all());
            }
            println!("  Initial subset before eps closure: {:?}", start_subset);
            
            // Manual epsilon closure
            use std::collections::VecDeque;
            let mut closure = start_subset.clone();
            let mut worklist: VecDeque<usize> = start_subset.keys().copied().collect();
            
            while let Some(u) = worklist.pop_front() {
                let u_weight = closure.get(&u).unwrap().clone();
                for (v, eps_weight) in &nwa.states[u].epsilons {
                    let v_new_weight = &u_weight & eps_weight;
                    println!("    eps {} -> {} with eps_weight {:?}, combined {:?}", u, v, eps_weight, v_new_weight);
                    if !v_new_weight.is_empty() {
                        let v_current_weight = closure.entry(*v).or_insert_with(Weight::zeros);
                        let combined = &*v_current_weight | &v_new_weight;
                        if combined != *v_current_weight {
                            *v_current_weight = combined;
                            worklist.push_back(*v);
                        }
                    }
                }
            }
            
            println!("  Epsilon closure: {:?}", closure);
            
            // Now trace transitions
            for (nwa_id, path_weight) in &closure {
                for (label, targets) in &nwa.states[*nwa_id].transitions {
                    for (target, trans_weight) in targets {
                        let combined = path_weight & trans_weight;
                        println!("  State {} (path_weight {:?}) has transition {} -> {} with trans_weight {:?}, combined {:?}", 
                                 nwa_id, path_weight, label, target, trans_weight, combined);
                    }
                }
            }
        }
        
        let dwa_builtin = nwa.determinize();
        println!("States: {}", dwa_builtin.states.len());
        println!("Transitions: {}", dwa_builtin.states.num_transitions());
        println!("Unique weights: {}", count_unique_weights(&dwa_builtin));
        
        for i in 0..dwa_builtin.states.len() {
            let s = &dwa_builtin.states[i];
            println!("  State {} (start? {}):", i, i == dwa_builtin.body.start_state);
            if let Some(fw) = &s.final_weight {
                println!("    final: {:?}", fw);
            }
            if let Some(sw) = &s.state_weight {
                println!("    state_weight: {:?}", sw);
            }
            for (lbl, target) in &s.transitions {
                let w = &s.trans_weights[lbl];
                println!("    {} -> {}: {:?} (len={})", lbl, target, w, w.len());
            }
        }
        
        // Verify the expected weight
        // Path 1: w_a & w_1 = (0..49) & (10..30) = 10..30
        // Path 2: w_b & w_2 = (25..74) & (30..50) = 30..50
        // Combined: 10..30 | 30..50 = 10..50
        let expected_edge_weight = Weight::from_rsb(RangeSetBlaze::from_iter(10..=50));
        println!("\n  Expected edge weight: {:?}", expected_edge_weight);
        if let Some(actual_weight) = dwa_builtin.states[dwa_builtin.body.start_state].trans_weights.get(&1) {
            println!("  Actual edge weight: {:?}", actual_weight);
            if *actual_weight == expected_edge_weight {
                println!("  ✓ Edge weight MATCHES expected");
            } else {
                println!("  ✗ Edge weight DIFFERS from expected!");
                println!("    Missing in actual: {:?}", &expected_edge_weight & &!actual_weight);
                println!("    Extra in actual: {:?}", actual_weight & &!&expected_edge_weight);
            }
        }
        
        // Path 2: With rm_epsilon before determinization
        println!("\n=== Path 2: rm_epsilon + Builtin determinization ===");
        
        // Debug: show the intermediate FST before rm_epsilon
        {
            use crate::precompute4::weighted_automata::determinization_rustfst::nwa_to_vector_fst;
            use rustfst::prelude::*;
            use rustfst::algorithms::rm_epsilon::rm_epsilon;
            
            let mut fst = nwa_to_vector_fst(&nwa);
            println!("Before rm_epsilon - FST has {} states", fst.num_states());
            for s in 0..fst.num_states() {
                let is_start = fst.start() == Some(s as StateId);
                println!("  FST State {} (start={}):", s, is_start);
                if let Some(w) = fst.final_weight(s as StateId).unwrap() {
                    println!("    final: {:?}", w);
                }
                for tr in fst.get_trs(s as StateId).unwrap().trs() {
                    println!("    tr(ilabel={}, olabel={}) -> {}: {:?}", 
                             tr.ilabel, tr.olabel, tr.nextstate, tr.weight);
                }
            }
            
            // Manual test: apply rm_epsilon step by step
            println!("\n  === Manual rm_epsilon trace ===");
            
            // For the Boolean semiring rm_epsilon, from start state:
            // Step 1: Find epsilon closure from state 0
            // The epsilon closure should be:
            //   state 0 with weight ONE (all bits)
            //   state 1 with weight 0..=49 (eps weight from 0->1)
            //   state 2 with weight 25..=74 (eps weight from 0->2)
            
            // Step 2: For each state in epsilon closure, collect non-epsilon transitions
            //   From state 1: label 1 -> state 3 with weight 10..=13
            //   From state 2: label 1 -> state 3 with weight 12..=15
            
            // Step 3: Combine weights:
            //   From state 0, label 1 -> state 3:
            //   Weight via state 1: eps(0->1) ⊗ w(1->3) = 0..=49 ⊗ 10..=13 = 10..=13
            //   Weight via state 2: eps(0->2) ⊗ w(2->3) = 25..=74 ⊗ 12..=15 = 12..=15
            //   Combined: 10..=13 ⊕ 12..=15 = 10..=15
            println!("  Expected combined weight: 10..=15");
            
            fst.compute_and_update_properties_all().unwrap();
            rm_epsilon(&mut fst).unwrap();
            
            println!("\n  After rm_epsilon - FST has {} states", fst.num_states());
            for s in 0..fst.num_states() {
                let is_start = fst.start() == Some(s as StateId);
                println!("  FST State {} (start={}):", s, is_start);
                if let Some(w) = fst.final_weight(s as StateId).unwrap() {
                    println!("    final: {:?}", w);
                }
                for tr in fst.get_trs(s as StateId).unwrap().trs() {
                    println!("    tr(ilabel={}, olabel={}) -> {}: {:?}", 
                             tr.ilabel, tr.olabel, tr.nextstate, tr.weight);
                }
            }
        }
        
        let nwa_eps_free = nwa.remove_epsilons();
        println!("After rm_epsilon (NWA) - States: {}", nwa_eps_free.states.len());
        
        for i in 0..nwa_eps_free.states.len() {
            let s = &nwa_eps_free.states[i];
            println!("  State {}:", i);
            if let Some(fw) = &s.final_weight {
                println!("    final: {:?}", fw);
            }
            for (t, w) in &s.epsilons {
                println!("    eps -> {}: {:?}", t, w);
            }
            for (lbl, targets) in &s.transitions {
                for (t, w) in targets {
                    println!("    {} -> {}: {:?}", lbl, t, w);
                }
            }
        }
        
        let dwa_with_rm_eps = nwa_eps_free.determinize();
        println!("After determinize - States: {}", dwa_with_rm_eps.states.len());
        println!("Unique weights: {}", count_unique_weights(&dwa_with_rm_eps));
        
        for i in 0..dwa_with_rm_eps.states.len() {
            let s = &dwa_with_rm_eps.states[i];
            println!("  State {}:", i);
            if let Some(fw) = &s.final_weight {
                println!("    final: {:?}", fw);
            }
            for (lbl, target) in &s.transitions {
                let w = &s.trans_weights[lbl];
                println!("    {} -> {}: {:?}", lbl, target, w);
            }
        }
        
        // Path 3: Full RustFST pipeline (includes rm_epsilon)
        println!("\n=== Path 3: RustFST determinization ===");
        let dwa_rustfst = nwa.determinize_to_dwa_with_rustfst();
        println!("States: {}", dwa_rustfst.states.len());
        println!("Unique weights: {}", count_unique_weights(&dwa_rustfst));
        
        for i in 0..dwa_rustfst.states.len() {
            let s = &dwa_rustfst.states[i];
            println!("  State {}:", i);
            if let Some(fw) = &s.final_weight {
                println!("    final: {:?}", fw);
            }
            for (lbl, target) in &s.transitions {
                let w = &s.trans_weights[lbl];
                println!("    {} -> {}: {:?}", lbl, target, w);
            }
        }
        
        // Now test minimization
        println!("\n=== Minimization Test ===");
        
        let mut dwa_builtin_min = dwa_builtin.clone();
        dwa_builtin_min.minimize_with_rustfst_full();
        println!("Builtin after minimize_with_rustfst_full: {} states, {} unique weights",
                 dwa_builtin_min.states.len(), count_unique_weights(&dwa_builtin_min));
        
        let mut dwa_rm_eps_min = dwa_with_rm_eps.clone();
        dwa_rm_eps_min.minimize_with_rustfst_full();
        println!("rm_epsilon after minimize_with_rustfst_full: {} states, {} unique weights",
                 dwa_rm_eps_min.states.len(), count_unique_weights(&dwa_rm_eps_min));
        
        // The key insight:
        // - Builtin determinization creates weights like: (W_a & W_1) | (W_b & W_2)
        //   which may differ from the canonical form
        // - rm_epsilon first "normalizes" the epsilon structure, so determinization
        //   produces the canonical weight form
        
        println!("\n=== Analysis ===");
        if count_unique_weights(&dwa_builtin) > count_unique_weights(&dwa_with_rm_eps) {
            println!("CONFIRMED: Builtin produces MORE unique weights than rm_epsilon approach");
            println!("This weight fragmentation is why minimization fails!");
        } else if count_unique_weights(&dwa_builtin) < count_unique_weights(&dwa_with_rm_eps) {
            println!("UNEXPECTED: Builtin produces FEWER unique weights?");
        } else {
            println!("Same number of unique weights - need different example");
        }
    }
    
    /// A more complex example that should show the issue more clearly
    #[test]
    fn test_weight_fragmentation_complex() {
        // Create a diamond pattern that will show weight fragmentation
        //
        //    S0 (start)
        //     |
        //   eps(W_all)
        //     |
        //    S1
        //   /   \
        // eps(W_a) eps(W_b)
        // /         \
        // S2         S3
        //  \        /
        //   a(W_x) a(W_y)
        //    \    /
        //     S4 (final: W_f)
        
        let mut nwa = NWA::new();
        
        let s0 = nwa.states.add_state();
        let s1 = nwa.states.add_state();
        let s2 = nwa.states.add_state();
        let s3 = nwa.states.add_state();
        let s4 = nwa.states.add_state();
        
        nwa.body.start_states = vec![s0];
        
        let w_all = Weight::all();
        let w_a = Weight::from_rsb(RangeSetBlaze::from_iter(0..100));
        let w_b = Weight::from_rsb(RangeSetBlaze::from_iter(50..150));
        let w_x = Weight::from_rsb(RangeSetBlaze::from_iter(0..50));
        let w_y = Weight::from_rsb(RangeSetBlaze::from_iter(75..125));
        let w_f = Weight::from_rsb(RangeSetBlaze::from_iter([1000]));
        
        nwa.states.add_epsilon(s0, s1, w_all.clone());
        nwa.states.add_epsilon(s1, s2, w_a.clone());
        nwa.states.add_epsilon(s1, s3, w_b.clone());
        nwa.states.add_transition(s2, 1, s4, w_x.clone()).unwrap();
        nwa.states.add_transition(s3, 1, s4, w_y.clone()).unwrap();
        nwa.states[s4].final_weight = Some(w_f.clone());
        
        println!("\n=== Diamond Pattern NWA ===");
        
        // Builtin
        let dwa_builtin = nwa.determinize();
        println!("Builtin: {} states, {} unique weights",
                 dwa_builtin.states.len(), count_unique_weights(&dwa_builtin));
        
        // rm_epsilon
        let nwa_eps_free = nwa.remove_epsilons();
        let dwa_rm_eps = nwa_eps_free.determinize();
        println!("rm_epsilon: {} states, {} unique weights",
                 dwa_rm_eps.states.len(), count_unique_weights(&dwa_rm_eps));
        
        // RustFST
        let dwa_rustfst = nwa.determinize_to_dwa_with_rustfst();
        println!("RustFST: {} states, {} unique weights",
                 dwa_rustfst.states.len(), count_unique_weights(&dwa_rustfst));
        
        // After minimization
        let mut dwa_builtin_min = dwa_builtin.clone();
        dwa_builtin_min.minimize_with_rustfst_full();
        println!("Builtin minimized: {} states", dwa_builtin_min.states.len());
        
        let mut dwa_rm_eps_min = dwa_rm_eps.clone();
        dwa_rm_eps_min.minimize_with_rustfst_full();
        println!("rm_epsilon minimized: {} states", dwa_rm_eps_min.states.len());
        
        println!("\nWeight difference analysis:");
        
        // Collect weights from each
        let weights_builtin: HashSet<_> = dwa_builtin.states.0.iter()
            .flat_map(|s| s.trans_weights.values().chain(s.final_weight.iter()))
            .cloned()
            .collect();
        let weights_rm_eps: HashSet<_> = dwa_rm_eps.states.0.iter()
            .flat_map(|s| s.trans_weights.values().chain(s.final_weight.iter()))
            .cloned()
            .collect();
        
        let only_builtin: Vec<_> = weights_builtin.difference(&weights_rm_eps).collect();
        let only_rm_eps: Vec<_> = weights_rm_eps.difference(&weights_builtin).collect();
        
        if !only_builtin.is_empty() {
            println!("Weights only in builtin: {}", only_builtin.len());
            for w in &only_builtin {
                println!("  {:?}", w);
            }
        }
        if !only_rm_eps.is_empty() {
            println!("Weights only in rm_eps: {}", only_rm_eps.len());
            for w in &only_rm_eps {
                println!("  {:?}", w);
            }
        }
    }
}
