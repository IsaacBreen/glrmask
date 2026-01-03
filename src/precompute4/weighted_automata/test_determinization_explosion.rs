#![cfg(test)]
use crate::precompute4::weighted_automata::common::Label;
use super::*;
use std::collections::BTreeSet;

#[test]
fn test_transition_count_increase_on_epsilon_merge() {
    // This test replicates the specific counter-example found by fuzzing where
    // merging start states causes transition count to INCREASE.
    //
    // Original (found by fuzz):
    // 8 states, 12 transitions
    // Merged:
    // 7 states, 13 transitions

    let mut nwa = NWA::new();
    nwa.states.0.clear(); // Clear default start state
    
    // Create states 0..7
    let states: Vec<StateID> = (0..8).map(|_| nwa.states.add_state()).collect();
    let s = |i: usize| states[i];

    // Transitions from fuzz case:
    // Orig: 0 --1--> 1
    // Orig: 0 --2--> 2
    // Orig: 1 --a--> 1
    // Orig: 1 --b--> 4
    // Orig: 2 --a--> 3
    // Orig: 4 --a--> 4
    // Orig: 4 --b--> 5
    // Orig: 5 --a--> 5
    // Orig: 5 --b--> 6
    // Orig: 6 --a--> 7
    // Orig: 6 --b--> 1
    // Orig: 7 --a--> 1

    let w = Weight::all();
    let label_1 = 100 as Label; // '1'
    let label_2 = 101 as Label; // '2'
    let label_a = 97 as Label; // 'a'
    let label_b = 98 as Label; // 'b'

    // Init transitions
    nwa.add_transition(s(0), label_1, s(1), w.clone()).unwrap();
    nwa.add_transition(s(0), label_2, s(2), w.clone()).unwrap();

    // Branch 1
    nwa.add_transition(s(1), label_a, s(1), w.clone()).unwrap();
    nwa.add_transition(s(1), label_b, s(4), w.clone()).unwrap();
    
    // Branch 2
    nwa.add_transition(s(2), label_a, s(3), w.clone()).unwrap();
    
    // Shared / Looping structure
    nwa.add_transition(s(4), label_a, s(4), w.clone()).unwrap();
    nwa.add_transition(s(4), label_b, s(5), w.clone()).unwrap();
    
    nwa.add_transition(s(5), label_a, s(5), w.clone()).unwrap();
    nwa.add_transition(s(5), label_b, s(6), w.clone()).unwrap();
    
    nwa.add_transition(s(6), label_a, s(7), w.clone()).unwrap();
    nwa.add_transition(s(6), label_b, s(1), w.clone()).unwrap();
    
    nwa.add_transition(s(7), label_a, s(1), w.clone()).unwrap();

    // Set start state
    nwa.body.start_states = vec![s(0)];

    // 1. Original Determinization
    let mut orig_dwa = nwa.determinize();
    orig_dwa.minimize_states();
    orig_dwa.simplify();
    orig_dwa.minimize_with_rustfst();
    let orig_states = orig_dwa.states.len();
    let orig_trans = orig_dwa.states.num_transitions();
    
    println!("Original: {} states, {} transitions", orig_states, orig_trans);
    
    // 2. Modified (Epsilon Merge)
    let mut mod_nwa = nwa.clone();
    
    // Clear transitions from start state 0
    mod_nwa.states[s(0)].transitions.clear();
    
    // Add epsilons to targets of previous start transitions (1 and 2)
    mod_nwa.add_epsilon(s(0), s(1), w.clone());
    mod_nwa.add_epsilon(s(0), s(2), w.clone());
    
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.minimize_states();
    mod_dwa.simplify();
    mod_dwa.minimize_with_rustfst();
    let mod_states = mod_dwa.states.len();
    let mod_trans = mod_dwa.states.num_transitions();
    
    println!("Modified: {} states, {} transitions", mod_states, mod_trans);
    
    // ASSERTION: Verify the phenomenon
    assert!(mod_states < orig_states, "Expected fewer states (got {} vs {})", mod_states, orig_states);
    assert!(mod_trans > orig_trans, "Expected MORE transitions (got {} vs {})", mod_trans, orig_trans);
}
