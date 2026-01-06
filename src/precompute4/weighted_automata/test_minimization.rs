use crate::precompute4::weighted_automata::nwa::NWA;
use std::fs;
use std::path::PathBuf;

#[test]
fn test_minimization_889() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("nwa_repro_min.json");

    if !d.exists() {
        // Fallback for when running from root
        d = PathBuf::from("nwa_repro_min.json");
    }
    
    let content = fs::read_to_string(&d).expect("Failed to read nwa_repro_min.json");
    let nwa: NWA = serde_json::from_str(&content).expect("Failed to parse NWA");

    // RustFST pipeline
    let mut dwa_rustfst = nwa.determinize_to_dwa_with_rustfst();
    dwa_rustfst.simplify();
    println!("RustFST states: {}", dwa_rustfst.states.len());

    // Builtin pipeline
    let mut dwa_builtin = nwa.determinize();
    dwa_builtin.simplify();
    println!("Builtin states: {}", dwa_builtin.states.len());

    // They should match
    assert_eq!(
        dwa_builtin.states.len(), 
        dwa_rustfst.states.len(), 
        "State count mismatch! Builtin: {}, RustFST: {}", 
        dwa_builtin.states.len(), 
        dwa_rustfst.states.len()
    );

    // Optional: Assert exact count if we're sure it should be 889
    // assert_eq!(dwa_builtin.states.len(), 889, "Expected 889 states");
}
