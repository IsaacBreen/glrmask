// Test to verify that rm_epsilon doesn't affect the final result

use crate::precompute4::weighted_automata::nwa::NWA;
use std::fs;
use std::path::PathBuf;

#[test]
#[ignore = "slow test - takes too long for CI"]
fn test_rm_epsilon_effect() {
    // Load the NWA
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("nwa_dump.json");
    let content = fs::read_to_string(&path).expect("Failed to read nwa_dump.json");
    let nwa: NWA = serde_json::from_str(&content).expect("Failed to parse NWA");
    
    println!("Original NWA: {} states", nwa.states.len());
    let epsilon_count: usize = nwa.states.0.iter().map(|s| s.epsilons.len()).sum();
    println!("Epsilon transitions: {}", epsilon_count);
    
    // Test WITHOUT rm_epsilon
    println!("\n=== WITHOUT rm_epsilon ===");
    let dwa_no_rm = nwa.determinize();
    println!("States after determinization: {}", dwa_no_rm.states.len());
    let mut dwa_no_rm_minimized = dwa_no_rm.clone();
    dwa_no_rm_minimized.minimize();
    println!("States after minimization: {}", dwa_no_rm_minimized.states.len());
    
    // Test WITH rm_epsilon
    println!("\n=== WITH rm_epsilon ===");
    let nwa_rm = nwa.remove_epsilons();
    println!("NWA after rm_epsilon: {} states", nwa_rm.states.len());
    let epsilon_count_rm: usize = nwa_rm.states.0.iter().map(|s| s.epsilons.len()).sum();
    println!("Epsilon transitions after rm_epsilon: {}", epsilon_count_rm);
    
    let dwa_with_rm = nwa_rm.determinize();
    println!("States after determinization: {}", dwa_with_rm.states.len());
    let mut dwa_with_rm_minimized = dwa_with_rm.clone();
    dwa_with_rm_minimized.minimize();
    println!("States after minimization: {}", dwa_with_rm_minimized.states.len());
    
    // Compare
    println!("\n=== COMPARISON ===");
    println!("Without rm_epsilon: {} -> {} -> {}", 
        nwa.states.len(), dwa_no_rm.states.len(), dwa_no_rm_minimized.states.len());
    println!("With rm_epsilon: {} -> {} -> {}", 
        nwa_rm.states.len(), dwa_with_rm.states.len(), dwa_with_rm_minimized.states.len());
    
    if dwa_no_rm_minimized.states.len() != dwa_with_rm_minimized.states.len() {
        println!("\n!!! DIFFERENCE FOUND !!!");
        println!("rm_epsilon DOES affect the final state count!");
    } else {
        println!("\nNo difference. rm_epsilon does NOT affect the final state count.");
    }
}
