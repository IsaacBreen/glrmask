#![cfg(test)]

use super::*;

#[test]
fn test_determinize_simple_divergence() {
    let mut nwa = NWA::new(); // new() creates a start state 0
    nwa.states.0.clear(); // but we'll manage states manually for clarity

    // NWA for "ac" with weight A1 (0..=0)
    let s0 = nwa.states.add_state();
    let s1 = nwa.states.add_state();
    let s2 = nwa.states.add_state();
    nwa.add_transition(s0, 'a' as i16, s1, Weight::all()).unwrap();
    nwa.add_transition(s1, 'c' as i16, s2, Weight::all()).unwrap();
    nwa.states[s2].final_weight = Some(Weight::from_item(0));

    // NWA for "bc" with weight A2 (1..=1)
    let s3 = nwa.states.add_state();
    let s4 = nwa.states.add_state();
    let s5 = nwa.states.add_state();
    nwa.add_transition(s3, 'b' as i16, s4, Weight::all()).unwrap();
    nwa.add_transition(s4, 'c' as i16, s5, Weight::all()).unwrap();
    nwa.states[s5].final_weight = Some(Weight::from_item(1));

    // Union them with a new start state
    let start = nwa.states.add_state();
    nwa.add_epsilon(start, s0, Weight::all());
    nwa.add_epsilon(start, s3, Weight::all());
    nwa.body.start_state = start;

    let dwa = nwa.determinize_to_dwa();

    // The product construction would yield 6 states (start, a, b, ac, bc, sink).
    // An efficient DWA could be smaller, but without minimization of the final DWA,
    // this is a reasonable expectation from the product construction.
    // Let's check the accepted words and their weights first.
    assert_eq!(
        dwa.eval_word_weight(&['a' as i16, 'c' as i16]),
        Weight::from_item(0)
    );
    assert_eq!(
        dwa.eval_word_weight(&['b' as i16, 'c' as i16]),
        Weight::from_item(1)
    );
    assert!(dwa.eval_word_weight(&['a' as i16, 'b' as i16]).is_empty());
    assert!(dwa.eval_word_weight(&['c' as i16]).is_empty());
    assert!(dwa.eval_word_weight(&[]).is_empty());

    // Assert on state count. Based on product construction of two 3-state NFAs (plus sink),
    // we expect a handful of states.
    // Reachable states in product: (s0,t0), (s1,t_sink), (s_sink,t1), (s2,t_sink), (s_sink,t2), (s_sink,t_sink) -> 6 states
    // The implementation might be slightly different.
    assert!(
        dwa.states.len() <= 10,
        "Expected a small number of states, got {}",
        dwa.states.len()
    );
}

#[test]
fn test_determinize_hypercube_catastrophe() {
    const N: usize = 4;
    let alphabet: Vec<i16> = (0..N as i16).map(|i| i + 'a' as i16).collect();
    let atoms: Vec<Weight> = (0..N).map(|i| Weight::from_item(i)).collect();

    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let mut component_starts = vec![];

    for i in 0..N {
        // L_i = words without alphabet[i]
        let s = nwa.states.add_state();
        component_starts.push(s);
        nwa.states[s].final_weight = Some(atoms[i].clone());

        for j in 0..N {
            if i == j {
                continue;
            }
            nwa.add_transition(s, alphabet[j], s, Weight::all()).unwrap();
        }
    }

    let start = nwa.states.add_state();
    for &s_comp in &component_starts {
        nwa.add_epsilon(start, s_comp, Weight::all());
    }
    nwa.body.start_state = start;

    let dwa = nwa.determinize_to_dwa();

    // The product construction should create 2^N states, as it needs to track
    // for each component language whether the forbidden character has been seen.
    // Each component DFA has 2 states (accepting, sink). The product has 2^N states.
    let expected_states = 1 << N;
    assert_eq!(
        dwa.states.len(),
        expected_states,
        "Expected 2^N states for hypercube"
    );

    // Test some words
    // word "ac" (chars 0 and 2) -> should not contain 1 or 3 ('b' or 'd'). Belongs to L1 and L3.
    // Weight should be A1 | A3.
    let word_ac = vec![alphabet[0], alphabet[2]];
    let expected_weight_ac = &atoms[1] | &atoms[3];
    assert_eq!(dwa.eval_word_weight(&word_ac), expected_weight_ac);

    // word "abcd" -> contains all symbols. Should be rejected.
    let word_all = alphabet.clone();
    assert!(dwa.eval_word_weight(&word_all).is_empty());

    // empty word -> belongs to all languages.
    let empty_word = vec![];
    let mut expected_weight_empty = Weight::zeros();
    for atom in &atoms {
        expected_weight_empty |= atom;
    }
    assert_eq!(dwa.eval_word_weight(&empty_word), expected_weight_empty);
}