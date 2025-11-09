#![cfg(test)]

use super::*;

#[test]
fn test_determinize_simple_divergence() {
    let mut nwa = NWA::new(); // new() creates a start state 0
    nwa.states.0.clear(); // but we'll manage states manually for clarity

    // NWA for "ac" with weight A1 (0)
    let s0 = nwa.states.add_state();
    let s1 = nwa.states.add_state();
    let s2 = nwa.states.add_state();
    nwa.add_transition(s0, 'a' as i16, s1, Weight::all()).unwrap();
    nwa.add_transition(s1, 'c' as i16, s2, Weight::all()).unwrap();
    nwa.states[s2].final_weight = Some(Weight::from_item(0));

    // NWA for "bc" with weight A2 (1)
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

    // Check accepted words and their weights.
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

    // An efficient DWA should merge states with similar future behavior,
    // resulting in a small automaton.
    // Allowing for a sink state, we assert a small number.
    assert!(
        dwa.states.len() <= 4,
        "Expected a small number of states for simple divergence, got {}",
        dwa.states.len()
    );
}

#[test]
fn test_determinize_hypercube_catastrophe() {
    const N: usize = 4;
    let alphabet: Vec<i16> = (0..N as i16).map(|i| i + 'a' as i16).collect();
    let atoms: Vec<Weight> = (0..N).map(Weight::from_item).collect();

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

    // An efficient DWA construction should avoid the 2^N state explosion by
    // encoding the history of seen characters within the accumulated weight,
    // not in the state space.
    assert!(
        dwa.states.len() <= 2,
        "Expected a very small DWA (1-2 states) for hypercube, but got {} states. State explosion was not avoided.",
        dwa.states.len()
    );

    // Test some words
    // word "ac" (chars 0 and 2) -> should not contain 1 or 3 ('b' or 'd'). Belongs to L1 and L3.
    // Weight should be A1 | A3.
    let word_ac = vec![alphabet[0], alphabet[2]];
    let expected_weight_ac = &atoms[1] | &atoms[3];
    assert_eq!(dwa.eval_word_weight(&word_ac), expected_weight_ac);

    // word "abcd" -> contains all symbols. Should be rejected by all components.
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