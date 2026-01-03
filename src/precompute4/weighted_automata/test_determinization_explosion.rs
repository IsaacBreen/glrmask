#![cfg(test)]
use crate::precompute4::weighted_automata::common::Label;
use super::*;
use std::collections::BTreeSet;

#[test]
// #[ignore] // Uncomment to run fuzz search
#[test]
fn fuzz_find_transition_explosion_after_simplify() {
    // We are looking for a case where:
    // Original: Determinize -> Simplify
    // Modified (Start->Epsilons): Determinize -> Simplify
    // Result: Modified.transitions > Original.transitions

    // Simple XorShift for determinism
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
    let alphabet: Vec<Label> = (0..3).map(|i| 97 + i as Label).collect(); // 'a', 'b', 'c'
    let w = Weight::all();

    for iter in 0..10_000 {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        
        let start_node = nwa.states.add_state();
        nwa.body.start_states = vec![start_node];

        let num_branches = 2 + (rng.next() % 2); // 2 or 3 branches
        let depth = 3 + (rng.next() % 3); // Depth 3-5

        // Create branches
        for b in 0..num_branches {
            let mut prev = start_node;
            // Distinct initial transition
            let init_label = 100 + b as Label; 
            let mut curr = nwa.states.add_state();
            nwa.add_transition(prev, init_label, curr, w.clone()).unwrap();
            prev = curr;
            
            // Random chain
            for _ in 0..depth {
                let next = nwa.states.add_state();
                // Random edges
                for &c in &alphabet {
                    if rng.next() % 2 == 0 {
                        nwa.add_transition(prev, c, next, w.clone()).unwrap();
                    }
                }
                // Random loopbacks or cross-links
                if rng.next() % 3 == 0 {
                    let rand_target = (rng.next() as usize) % nwa.states.len();
                    let rand_char = alphabet[(rng.next() as usize) % alphabet.len()];
                    nwa.add_transition(prev, rand_char, StateID(rand_target), w.clone()).unwrap();
                }

                // Random finality
                if rng.next() % 5 == 0 {
                    nwa.states[prev].final_weight = Some(w.clone());
                }
                prev = next;
            }
            // Ensure end of chain is final sometimes
            if rng.next() % 2 == 0 {
                nwa.states[prev].final_weight = Some(w.clone());
            }
        }

        // Original pipeline
        let mut orig_dwa = nwa.determinize();
        orig_dwa.simplify();
        let orig_states = orig_dwa.states.len();
        let orig_trans = orig_dwa.states.num_transitions();

        if orig_states == 0 { continue; } // Boring empty result

        // Modified pipeline
        let mut mod_nwa = nwa.clone();
        let start_trans = std::mem::take(&mut mod_nwa.states[start_node].transitions);
        for (_, targets) in start_trans {
             for (target, _) in targets {
                 mod_nwa.add_epsilon(start_node, target, w.clone());
             }
        }
        
        let mut mod_dwa = mod_nwa.determinize();
        mod_dwa.simplify();
        let mod_states = mod_dwa.states.len();
        let mod_trans = mod_dwa.states.num_transitions();

        if mod_trans > orig_trans { // FOUND IT!
             println!("FOUND EXPLOSION at iter {}", iter);
             println!("Original: {} states, {} trans", orig_states, orig_trans);
             println!("Modified: {} states, {} trans", mod_states, mod_trans);
             
             // Print NWA structure to reproduce
             println!("--- NWA Construction Code ---");
             println!("let mut nwa = NWA::new(); nwa.states.0.clear();");
             println!("// ... construct states based on dump ...");
             // Dump transitions
             for (i, s) in nwa.states.0.iter().enumerate() {
                 if s.final_weight.is_some() { println!("nwa.states[StateID({})].final_weight = Some(Weight::all());", i); }
                 for (k, v) in &s.transitions {
                     let char_label = if *k >= 97 { (*k as u8 as char).to_string() } else { k.to_string() };
                     for (target, _) in v {
                        println!("nwa.add_transition(StateID({}), {} as Label, StateID({}), Weight::all()).unwrap();", i, k, target.0);
                     }
                 }
             }
             println!("nwa.body.start_states = vec![StateID({})];", start_node.0);
             println!("-----------------------------");

             assert!(mod_trans > orig_trans);
             return;
        }
    }
    // If we get here, we failed to find one in 10k iters.
    // Uncommenting this failure creates noise, but let's leave a soft failure.
    // assert!(false, "Failed to find explosion example in 10k iterations");
}
