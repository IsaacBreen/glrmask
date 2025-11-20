#![cfg(test)]

use crate::precompute4::weighted_automata::common::Label;
use super::*;

#[should_panic]
#[test]
fn test_determinize_simple_divergence() {
    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let s0 = nwa.states.add_state();
    let s1 = nwa.states.add_state();
    let s2 = nwa.states.add_state();
    nwa.add_transition(s0, 'a' as Label, s1, Weight::all()).unwrap();
    nwa.add_transition(s1, 'c' as Label, s2, Weight::all()).unwrap();
    nwa.states[s2].final_weight = Some(Weight::from_item(0));

    let s3 = nwa.states.add_state();
    let s4 = nwa.states.add_state();
    let s5 = nwa.states.add_state();
    nwa.add_transition(s3, 'b' as Label, s4, Weight::all()).unwrap();
    nwa.add_transition(s4, 'c' as Label, s5, Weight::all()).unwrap();
    nwa.states[s5].final_weight = Some(Weight::from_item(1));

    let start = nwa.states.add_state();
    nwa.add_epsilon(start, s0, Weight::all());
    nwa.add_epsilon(start, s3, Weight::all());
    nwa.body.start_state = start;

    let dwa = nwa.determinize_to_dwa();
    println!("{}", nwa);
    println!("{}", dwa);

    assert_eq!(dwa.eval_word_weight(&['a' as Label, 'c' as Label]), Weight::from_item(0));
    assert_eq!(dwa.eval_word_weight(&['b' as Label, 'c' as Label]), Weight::from_item(1));
    assert!(dwa.eval_word_weight(&['a' as Label, 'b' as Label]).is_empty());
    assert!(dwa.eval_word_weight(&['c' as Label]).is_empty());
    assert!(dwa.eval_word_weight(&[]).is_empty());

    assert!(
        dwa.states.len() <= 4,
        "Expected a small number of states for simple divergence, got {}",
        dwa.states.len()
    );
}

#[should_panic]
#[test]
fn test_determinize_hypercube_catastrophe() {
    const N: usize = 4;
    let alphabet: Vec<Label> = (0..N as Label).map(|i| i + 'a' as Label).collect();
    let atoms: Vec<Weight> = (0..N).map(Weight::from_item).collect();

    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let mut component_starts = vec![];

    for i in 0..N {
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

    assert!(
        dwa.states.len() <= 2,
        "Expected a very small DWA (1-2 states) for hypercube, but got {} states. State explosion was not avoided.",
        dwa.states.len()
    );

    let word_ac = vec![alphabet[0], alphabet[2]];
    let expected_weight_ac = &atoms[1] | &atoms[3];
    assert_eq!(dwa.eval_word_weight(&word_ac), expected_weight_ac);

    let word_all = alphabet.clone();
    assert!(dwa.eval_word_weight(&word_all).is_empty());

    let empty_word = vec![];
    let mut expected_weight_empty = Weight::zeros();
    for atom in &atoms {
        expected_weight_empty |= atom;
    }
    assert_eq!(dwa.eval_word_weight(&empty_word), expected_weight_empty);
}
