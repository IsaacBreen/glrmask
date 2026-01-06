use crate::precompute4::weighted_automata::nwa::NWA;
use std::fs;
use std::path::PathBuf;
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

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

    // RustFST pipeline
    println!("Determinizing with RustFST...");
    let mut dwa_rustfst = nwa.determinize_to_dwa_with_rustfst();
    println!("Simplifying....");
    dwa_rustfst.simplify();
    println!("RustFST states: {}", dwa_rustfst.states.len());

    // Builtin pipeline
    println!("Determinizing with builtin...");
    let mut dwa_builtin = nwa.determinize();
    println!("Simplifying....");
    dwa_builtin.simplify();
    println!("Builtin states: {}", dwa_builtin.states.len());

    stochastic_equivalence_test(dwa_builtin.clone(), dwa_rustfst.clone());

    // They should match
    assert_eq!(
        dwa_builtin.states.len(),
        dwa_rustfst.states.len(),
        "State count mismatch! Builtin: {}, RustFST: {}",
        dwa_builtin.states.len(),
        dwa_rustfst.states.len()
    );
}
