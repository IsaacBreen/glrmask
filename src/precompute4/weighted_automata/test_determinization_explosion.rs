#![cfg(test)]
use crate::precompute4::weighted_automata::common::Label;
use super::*;
use std::collections::BTreeSet;
use range_set_blaze::RangeSetBlaze;

/// This test demonstrates the "transition explosion" phenomenon:
/// When initial labeled transitions from the start state are replaced with epsilon transitions,
/// the determinized & simplified DWA can have SIGNIFICANTLY more transitions.
///
/// The explosion occurs because:
/// 1. Original: Start has distinct labeled transitions to branch roots (e.g., tsid labels)
/// 2. Modified: Start has epsilon transitions to all branch roots
/// 3. Epsilon closure at start creates a super-state with all branch roots
/// 4. Traversing from super-state creates combinations of path weights
/// 5. Distinct weights prevent state merging during simplification
#[test]
fn test_weighted_trie_explosion() {
    // Create a simple weighted trie structure:
    // Start -> (label=100, W0) -> A -> (label='a', W0) -> B -> (final W0)
    // Start -> (label=101, W1) -> C -> (label='a', W1) -> D -> (final W1)
    //
    // Where W0 = {0} and W1 = {1} are DISJOINT weights.

    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let start = nwa.states.add_state();
    let a = nwa.states.add_state();
    let b = nwa.states.add_state();
    let c = nwa.states.add_state();
    let d = nwa.states.add_state();
    nwa.body.start_states = vec![start];

    let w0 = Weight::from_item(0);
    let w1 = Weight::from_item(1);

    nwa.add_transition(start, 100, a, w0.clone()).unwrap();
    nwa.add_transition(a, 97, b, w0.clone()).unwrap();
    nwa.states[b].final_weight = Some(w0.clone());

    nwa.add_transition(start, 101, c, w1.clone()).unwrap();
    nwa.add_transition(c, 97, d, w1.clone()).unwrap();
    nwa.states[d].final_weight = Some(w1.clone());

    // Original with custom simplify
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_trans_simplify = orig_dwa.states.num_transitions();
    
    // Original with rustfst minimize
    let mut orig_dwa_rustfst = nwa.determinize();
    orig_dwa_rustfst.minimize_with_rustfst();
    let orig_trans_rustfst = orig_dwa_rustfst.states.num_transitions();

    println!("Original: simplify={} trans, rustfst={} trans", orig_trans_simplify, orig_trans_rustfst);

    // Modified: Replace initial labeled transitions with epsilons
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }

    // Modified with custom simplify
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_trans_simplify = mod_dwa.states.num_transitions();

    // Modified with rustfst minimize
    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();

    println!("Modified: simplify={} trans, rustfst={} trans", mod_trans_simplify, mod_trans_rustfst);

    println!("Comparison:");
    println!("  simplify: {} -> {} ({})", orig_trans_simplify, mod_trans_simplify, 
             if mod_trans_simplify > orig_trans_simplify { "EXPLOSION" } 
             else if mod_trans_simplify < orig_trans_simplify { "reduction" } 
             else { "same" });
    println!("  rustfst:  {} -> {} ({})", orig_trans_rustfst, mod_trans_rustfst,
             if mod_trans_rustfst > orig_trans_rustfst { "EXPLOSION" }
             else if mod_trans_rustfst < orig_trans_rustfst { "reduction" }
             else { "same" });
}

/// A more complex test with deeper trie structure and more branches
#[test]
fn test_weighted_trie_explosion_deeper() {
    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];

    let num_branches = 4;
    let depth = 5;
    let alphabet: Vec<Label> = vec![97, 98, 99]; // 'a', 'b', 'c'

    for b in 0..num_branches {
        let w = Weight::from_item(b);
        
        let root = nwa.states.add_state();
        nwa.add_transition(start, 100 + b as Label, root, w.clone()).unwrap();
        
        let mut prev = root;
        for d in 0..depth {
            let next = nwa.states.add_state();
            let c = alphabet[d % alphabet.len()];
            nwa.add_transition(prev, c, next, w.clone()).unwrap();
            prev = next;
        }
        nwa.states[prev].final_weight = Some(w.clone());
    }

    // Original
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_trans_simplify = orig_dwa.states.num_transitions();
    
    let mut orig_dwa_rustfst = nwa.determinize();
    orig_dwa_rustfst.minimize_with_rustfst();
    let orig_trans_rustfst = orig_dwa_rustfst.states.num_transitions();

    println!("Deep Trie Original: simplify={} trans, rustfst={} trans", 
             orig_trans_simplify, orig_trans_rustfst);

    // Modified
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }

    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_trans_simplify = mod_dwa.states.num_transitions();

    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();

    println!("Deep Trie Modified: simplify={} trans, rustfst={} trans", 
             mod_trans_simplify, mod_trans_rustfst);

    println!("Comparison:");
    println!("  simplify: {} -> {} ({})", orig_trans_simplify, mod_trans_simplify, 
             if mod_trans_simplify > orig_trans_simplify { "EXPLOSION" } 
             else if mod_trans_simplify < orig_trans_simplify { "reduction" } 
             else { "same" });
    println!("  rustfst:  {} -> {} ({})", orig_trans_rustfst, mod_trans_rustfst,
             if mod_trans_rustfst > orig_trans_rustfst { "EXPLOSION" }
             else if mod_trans_rustfst < orig_trans_rustfst { "reduction" }
             else { "same" });
}

/// Test with branches that merge back together (diamond structure)
#[test]
fn test_weighted_trie_diamond_explosion() {
    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let start = nwa.states.add_state();
    let a = nwa.states.add_state();
    let b = nwa.states.add_state();
    let merge = nwa.states.add_state();
    let end = nwa.states.add_state();
    nwa.body.start_states = vec![start];

    let w0 = Weight::from_item(0);
    let w1 = Weight::from_item(1);

    nwa.add_transition(start, 100, a, w0.clone()).unwrap();
    nwa.add_transition(a, 97, merge, w0.clone()).unwrap();
    nwa.add_transition(merge, 120, end, w0.clone()).unwrap();

    nwa.add_transition(start, 101, b, w1.clone()).unwrap();
    nwa.add_transition(b, 97, merge, w1.clone()).unwrap();
    nwa.add_transition(merge, 120, end, w1.clone()).unwrap();

    nwa.states[end].final_weight = Some(Weight::all());

    // Original
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_trans_simplify = orig_dwa.states.num_transitions();
    
    let mut orig_dwa_rustfst = nwa.determinize();
    orig_dwa_rustfst.minimize_with_rustfst();
    let orig_trans_rustfst = orig_dwa_rustfst.states.num_transitions();

    println!("Diamond Original: simplify={} trans, rustfst={} trans", 
             orig_trans_simplify, orig_trans_rustfst);

    // Modified
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }

    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_trans_simplify = mod_dwa.states.num_transitions();

    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();

    println!("Diamond Modified: simplify={} trans, rustfst={} trans", 
             mod_trans_simplify, mod_trans_rustfst);

    println!("Comparison:");
    println!("  simplify: {} -> {} ({})", orig_trans_simplify, mod_trans_simplify, 
             if mod_trans_simplify > orig_trans_simplify { "EXPLOSION" } 
             else if mod_trans_simplify < orig_trans_simplify { "reduction" } 
             else { "same" });
    println!("  rustfst:  {} -> {} ({})", orig_trans_rustfst, mod_trans_rustfst,
             if mod_trans_rustfst > orig_trans_rustfst { "EXPLOSION" }
             else if mod_trans_rustfst < orig_trans_rustfst { "reduction" }
             else { "same" });
}

#[test]
#[ignore] // Uncomment to run fuzz search
fn fuzz_find_transition_explosion_after_simplify() {
    struct Rng { state: u64 }
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }
    }

    let mut rng = Rng { state: 12345 };
    let alphabet: Vec<Label> = (0..3).map(|i| 97 + i as Label).collect();

    for iter in 0..50_000 {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        
        let start_node = nwa.states.add_state();
        nwa.body.start_states = vec![start_node];

        let num_branches = 2 + (rng.next() % 4) as usize;
        let depth = 4 + (rng.next() % 4) as usize;

        for b in 0..num_branches {
            let distinct_w = Weight::from_item(b);
            
            let mut prev = start_node;
            let init_label = 100 + b as Label; 
            let mut curr = nwa.states.add_state();
            nwa.add_transition(prev, init_label, curr, distinct_w.clone()).unwrap();
            prev = curr;
            
            for _ in 0..depth {
                let next = nwa.states.add_state();
                for &c in &alphabet {
                    if rng.next() % 100 < 70 {
                        nwa.add_transition(prev, c, next, distinct_w.clone()).unwrap();
                    }
                }
                if rng.next() % 100 < 40 {
                    let rand_target = (rng.next() as usize) % nwa.states.len();
                    let rand_char = alphabet[(rng.next() as usize) % alphabet.len()];
                    nwa.add_transition(prev, rand_char, rand_target, distinct_w.clone()).unwrap();
                }
                if rng.next() % 5 == 0 {
                    nwa.states[prev].final_weight = Some(distinct_w.clone());
                }
                prev = next;
            }
            if rng.next() % 2 == 0 {
                nwa.states[prev].final_weight = Some(distinct_w.clone());
            }
        }

        let mut orig_dwa = nwa.determinize();
        orig_dwa.simplify();
        let orig_trans = orig_dwa.states.num_transitions();

        if orig_dwa.states.len() < 2 { continue; }

        let mut mod_nwa = nwa.clone();
        let start_trans = std::mem::take(&mut mod_nwa.states[start_node].transitions);
        for (_, targets) in start_trans {
             for (target, weight) in targets {
                 mod_nwa.add_epsilon(start_node, target, weight);
             }
        }
        
        let mut mod_dwa = mod_nwa.determinize();
        mod_dwa.simplify();
        let mod_trans = mod_dwa.states.num_transitions();

        if mod_trans > orig_trans {
             println!("FOUND EXPLOSION at iter {}", iter);
             println!("Original: {} trans", orig_trans);
             println!("Modified: {} trans", mod_trans);
             assert!(mod_trans > orig_trans);
             return;
        }
    }
    panic!("Failed to find explosion example in 50k iterations");
}
