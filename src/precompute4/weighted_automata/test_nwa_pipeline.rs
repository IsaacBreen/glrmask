use crate::precompute4::weighted_automata::nwa::NWA;
use std::fs;
use std::path::PathBuf;

#[test]
#[ignore = "slow test - takes too long for CI"]
fn test_nwa_minimize_determinize_minimize() {
    // Load the NWA from the JSON dump
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("nwa_dump.json");
    
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping test: nwa_dump.json not found at {:?}", path);
            return;
        }
    };
    
    let mut nwa: NWA = serde_json::from_str(&content).expect("Failed to parse NWA");
    println!("Loaded NWA with {} states", nwa.states.len());
    
    // Step 1: Minimize the NWA
    nwa.minimize();
    println!("After NWA minimize: {} states", nwa.states.len());
    
    // Step 2: Determinize
    let mut dwa = nwa.determinize();
    println!("After determinize: {} states", dwa.states.len());
    
    // Step 3: Minimize the DWA
    dwa.minimize();
    println!("After DWA minimize: {} states", dwa.states.len());
    
    // Assert that the number of states is <= 900
    assert!(
        dwa.states.len() <= 900,
        "Expected <= 900 states after minimize → determinize → minimize, got {} states",
        dwa.states.len()
    );
}
