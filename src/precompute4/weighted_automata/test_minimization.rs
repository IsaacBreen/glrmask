use crate::precompute4::weighted_automata::nwa::NWA;
use crate::precompute4::weighted_automata::dwa::DWA;
use std::fs;
use std::path::PathBuf;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

/// Deep comparison of two DWAs to find structural differences
fn compare_dwas_structure(dwa1: &DWA, dwa2: &DWA, label1: &str, label2: &str) {
    println!("\n=== COMPARING {} vs {} ===", label1, label2);
    
    // Compare state counts
    println!("State counts: {} vs {}", dwa1.states.len(), dwa2.states.len());
    
    // Compare transition counts
    let trans1 = dwa1.states.num_transitions();
    let trans2 = dwa2.states.num_transitions();
    println!("Transition counts: {} vs {}", trans1, trans2);
    
    // Compare start states
    println!("Start states: {} vs {}", dwa1.body.start_state, dwa2.body.start_state);
    
    // Count final states
    let final1 = dwa1.states.0.iter().filter(|s| s.final_weight.is_some()).count();
    let final2 = dwa2.states.0.iter().filter(|s| s.final_weight.is_some()).count();
    println!("Final states: {} vs {}", final1, final2);
    
    // Analyze transition structure
    let mut labels1: BTreeSet<_> = BTreeSet::new();
    let mut labels2: BTreeSet<_> = BTreeSet::new();
    for state in &dwa1.states.0 {
        for label in state.transitions.keys() {
            labels1.insert(*label);
        }
    }
    for state in &dwa2.states.0 {
        for label in state.transitions.keys() {
            labels2.insert(*label);
        }
    }
    println!("Unique labels: {} vs {}", labels1.len(), labels2.len());
    
    // Check for label differences
    let only_in_1: BTreeSet<_> = labels1.difference(&labels2).collect();
    let only_in_2: BTreeSet<_> = labels2.difference(&labels1).collect();
    if !only_in_1.is_empty() || !only_in_2.is_empty() {
        println!("  Labels only in {}: {:?}", label1, only_in_1);
        println!("  Labels only in {}: {:?}", label2, only_in_2);
    }
    
    // Analyze state weights
    let mut state_weights1: Vec<_> = dwa1.states.0.iter()
        .filter_map(|s| s.state_weight.as_ref())
        .collect();
    let mut state_weights2: Vec<_> = dwa2.states.0.iter()
        .filter_map(|s| s.state_weight.as_ref())
        .collect();
    println!("States with state_weight: {} vs {}", state_weights1.len(), state_weights2.len());
    
    // Analyze transition weights
    let mut trans_weights1: BTreeMap<String, usize> = BTreeMap::new();
    let mut trans_weights2: BTreeMap<String, usize> = BTreeMap::new();
    for state in &dwa1.states.0 {
        for (_, weight) in &state.trans_weights {
            *trans_weights1.entry(format!("{:?}", weight)).or_insert(0) += 1;
        }
    }
    for state in &dwa2.states.0 {
        for (_, weight) in &state.trans_weights {
            *trans_weights2.entry(format!("{:?}", weight)).or_insert(0) += 1;
        }
    }
    
    // Count unique transition weights
    let unique_weights1: BTreeSet<_> = dwa1.states.0.iter()
        .flat_map(|s| s.trans_weights.values())
        .collect();
    let unique_weights2: BTreeSet<_> = dwa2.states.0.iter()
        .flat_map(|s| s.trans_weights.values())
        .collect();
    println!("Unique transition weights: {} vs {}", unique_weights1.len(), unique_weights2.len());
    
    // Analyze out-degree distribution
    let mut out_degrees1: Vec<_> = dwa1.states.0.iter()
        .map(|s| s.transitions.len())
        .collect();
    let mut out_degrees2: Vec<_> = dwa2.states.0.iter()
        .map(|s| s.transitions.len())
        .collect();
    out_degrees1.sort();
    out_degrees2.sort();
    
    println!("Out-degree stats:");
    if !out_degrees1.is_empty() {
        println!("  {}: min={}, max={}, avg={:.2}", label1, 
            out_degrees1[0], 
            out_degrees1[out_degrees1.len()-1],
            out_degrees1.iter().sum::<usize>() as f64 / out_degrees1.len() as f64);
    }
    if !out_degrees2.is_empty() {
        println!("  {}: min={}, max={}, avg={:.2}", label2, 
            out_degrees2[0], 
            out_degrees2[out_degrees2.len()-1],
            out_degrees2.iter().sum::<usize>() as f64 / out_degrees2.len() as f64);
    }
}

#[test]
fn test_minimization_889() {
    // Load the NWA from the JSON dump
    // This file is expected to be in the root of the repo
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("nwa_dump.json");

    let content =
        fs::read_to_string(&path).expect(&format!("Failed to read nwa_dump.json from {:?}", path));

    let nwa: NWA = serde_json::from_str(&content).expect("Failed to parse NWA");
    println!("Loaded NWA with {} states", nwa.states.len());
    
    // Sanity check: Verify the NWA has the expected number of states from the export
    // From MACRO_DEBUG_LEVEL=5 make test-schema-id ID=Github_hard---o66331 with DWA_USE_RM_EPSILON=1
    // we got: "14647 states and 165438 transitions"
    assert_eq!(
        nwa.states.len(), 
        14647,
        "NWA state count mismatch! Expected 14647 states from Github_hard---o66331 terminal NWA, got {}",
        nwa.states.len()
    );
    
    // Count epsilon transitions in the original NWA
    let epsilon_count: usize = nwa.states.0.iter()
        .map(|s| s.epsilons.len())
        .sum();
    println!("Original NWA has {} epsilon transitions", epsilon_count);

    // Test the full pipeline that constraint_precompute.rs uses with DWA_USE_RM_EPSILON=1
    println!("\n=== PIPELINE: simplify NWA → compress → rm_epsilon → determinize → simplify DWA ===");
    println!("This mimics constraint_precompute.rs with DWA_USE_RM_EPSILON=1");
    
    println!("Step 0a: Simplify NWA with rustfst...");
    let mut nwa_simplified = nwa.clone();
    nwa_simplified.simplify_with_rustfst();
    println!("  After simplify_with_rustfst: {} states", nwa_simplified.states.len());
    
    println!("Step 0b: Compress transitions...");
    nwa_simplified.compress_transitions();
    println!("  After compress_transitions: {} states", nwa_simplified.states.len());
    
    println!("Step 1: Remove epsilons...");
    let nwa_no_eps = nwa_simplified.remove_epsilons();
    println!("  After rm_epsilon: {} states", nwa_no_eps.states.len());
    
    println!("Step 2: Determinize with builtin...");
    let mut dwa = nwa_no_eps.determinize();
    println!("  After determinize: {} states", dwa.states.len());
    
    println!("Step 3: Simplify with rustfst...");
    dwa.simplify_with_rustfst();
    println!("  After simplify_with_rustfst: {} states", dwa.states.len());
    
    println!("Step 4: Simplify with builtin...");
    dwa.simplify();
    println!("  After simplify: {} states", dwa.states.len());
    
    // Sanity check: Verify we get the expected 889 states
    // From MACRO_DEBUG_LEVEL=5 make test-schema-id ID=Github_hard---o66331 with DWA_USE_RM_EPSILON=1
    // Terminal output showed: "Determinized DWA with 889 states and 54431 transitions"
    assert_eq!(
        dwa.states.len(),
        889,
        "State count mismatch! Expected 889 states after full pipeline with rm_epsilon, got {}",
        dwa.states.len()
    );
    
    println!("\n=== SUCCESS: Test passed with {} states ===", dwa.states.len());
}
