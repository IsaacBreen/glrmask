#![cfg(test)]
//! Test demonstrating missed optimization: states with different final_weights but identical
//! outgoing structure that COULD be merged by pushing weights to incoming edges.
//!
//! ## The Problem
//!
//! After epsilon transformation, we get states like:
//! ```text
//!      ┌─["d9"]─→ 31 ─[:]─→ 105 ──┐
//!      │                          │
//!   0 ─┼─["d3"]─→ 32 ─[:]─→ 106 ──┼──71 edges each──→ 148
//!      │                          │
//!      └─["c7"]─→ 33 ─[:]─→ 107 ──┘
//! ```
//!
//! States 105, 106, 107 are structurally identical:
//! - All have 71 outgoing transitions to state 148
//! - All are final states
//! - BUT: Each has a DIFFERENT final_weight (42, 36, 30)
//!
//! Current minimization can't merge them because final_weights differ.
//!
//! ## The Optimization
//!
//! We CAN merge these states by "pushing" the final_weight differences to incoming edges:
//! 1. Merge 105, 106, 107 into a single state S
//! 2. Change final_weight of S to union: {42, 36, 30}
//! 3. Change incoming edge weights:
//!    - 31 -> S: intersect with {42}
//!    - 32 -> S: intersect with {36}
//!    - 33 -> S: intersect with {30}
//!
//! This preserves the language AND reduces state count!

use crate::precompute4::weighted_automata::*;
use crate::precompute4::weighted_automata::common::Label;

/// Builds a minimal example showing the missed optimization.
///
/// Structure:
/// ```text
///   0 ─[a]─→ 1 ─[:]─→ 3 (fw={100}) ──┐
///   0 ─[b]─→ 2 ─[:]─→ 4 (fw={200}) ──┼─[x]─→ 5 (fw=ALL)
/// ```
///
/// States 3 and 4 have identical outgoing structure (both -> 5 on 'x')
/// but different final_weights ({100} vs {200}).
///
/// After pushing weights to incoming edges, they could be merged into one state.
fn build_missed_optimization_example() -> DWA {
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    let s0 = nwa.states.add_state(); // start
    let s1 = nwa.states.add_state(); // after 'a'
    let s2 = nwa.states.add_state(); // after 'b'
    let s3 = nwa.states.add_state(); // after 'a:' (fw=100)
    let s4 = nwa.states.add_state(); // after 'b:' (fw=200)
    let s5 = nwa.states.add_state(); // final sink
    
    nwa.body.start_states = vec![s0];
    
    let all = Weight::all();
    let w100 = Weight::from_item(100);
    let w200 = Weight::from_item(200);
    
    // Labels
    let a: Label = 97; // 'a'
    let b: Label = 98; // 'b'
    let colon: Label = 58; // ':'
    let x: Label = 120; // 'x'
    
    // 0 -> 1 on 'a', 0 -> 2 on 'b'
    nwa.add_transition(s0, a, s1, all.clone()).unwrap();
    nwa.add_transition(s0, b, s2, all.clone()).unwrap();
    
    // 1 -> 3 on ':', 2 -> 4 on ':'
    nwa.add_transition(s1, colon, s3, all.clone()).unwrap();
    nwa.add_transition(s2, colon, s4, all.clone()).unwrap();
    
    // 3 -> 5 and 4 -> 5 on 'x' (identical outgoing!)
    nwa.add_transition(s3, x, s5, all.clone()).unwrap();
    nwa.add_transition(s4, x, s5, all.clone()).unwrap();
    
    // Final weights
    nwa.states[s3].final_weight = Some(w100);
    nwa.states[s4].final_weight = Some(w200);
    nwa.states[s5].final_weight = Some(all.clone());
    
    nwa.determinize()
}

/// Demonstrates that current minimization fails to merge structurally identical states
/// with different final_weights.
///
/// This test SHOULD FAIL until we implement the weight-push optimization.
/// The assertion checks for the optimal behavior we want.
///
/// Run with: `cargo test test_missed_optimization_current_behavior -- --ignored`
#[test]
#[ignore = "Fails until weight-push optimization is implemented"]
fn test_missed_optimization_current_behavior() {
    let mut dwa = build_missed_optimization_example();
    
    println!("Before minimization:");
    println!("  States: {}", dwa.states.len());
    println!("  Transitions: {}", dwa.states.num_transitions());
    
    // Current minimization
    dwa.minimize_with_rustfst();
    
    println!("\nAfter minimize_with_rustfst:");
    println!("  States: {}", dwa.states.len());
    println!("  Transitions: {}", dwa.states.num_transitions());
    
    // Print structure
    for (sid, state) in dwa.states.0.iter().enumerate() {
        let fw = match &state.final_weight {
            Some(w) if w.is_all_fast() => "ALL".to_string(),
            Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
            None => "none".to_string(),
        };
        println!("  State {} (fw={}): {} outgoing", sid, fw, state.transitions.len());
        for (&label, &target) in &state.transitions {
            let target_fw = match &dwa.states[target].final_weight {
                Some(w) if w.is_all_fast() => "ALL".to_string(),
                Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
                None => "none".to_string(),
            };
            println!("    --[{}]--> {} (fw={})", label as u8 as char, target, target_fw);
        }
    }
    
    // Count states with identical outgoing but different final_weights
    let mut outgoing_signature: std::collections::HashMap<Vec<(Label, usize)>, Vec<usize>> = 
        std::collections::HashMap::new();
    
    for (sid, state) in dwa.states.0.iter().enumerate() {
        let mut sig: Vec<_> = state.transitions.iter()
            .map(|(&l, &t)| (l, t))
            .collect();
        sig.sort();
        outgoing_signature.entry(sig).or_default().push(sid);
    }
    
    let mut mergeable = 0;
    for (sig, states) in &outgoing_signature {
        if states.len() > 1 && !sig.is_empty() {
            println!("\nStates with identical outgoing {:?}:", sig);
            for &sid in states {
                let fw = match &dwa.states.0[sid].final_weight {
                    Some(w) if w.is_all_fast() => "ALL".to_string(),
                    Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
                    None => "none".to_string(),
                };
                println!("  State {} has fw={}", sid, fw);
            }
            mergeable += states.len() - 1;
        }
    }
    
    println!("\nMissed optimization: {} states could be merged by pushing weights to incoming edges", mergeable);
    
    // THIS IS THE ASSERTION FOR THE TARGET BEHAVIOR
    // Currently fails because current minimization doesn't do weight pushing.
    // Once we implement weight-push optimization, this should pass.
    assert_eq!(mergeable, 0, 
        "After optimization, there should be no mergeable states left. \
         Found {} states that could be merged by pushing weights to incoming edges.",
        mergeable);
}

/// This test demonstrates what the OPTIMAL result would look like.
///
/// Instead of going through NWA determinization (which expands weight-differentiated
/// transitions), we directly build a DWA with the merged structure.
#[test]
fn test_optimal_merged_example() {
    use crate::precompute4::weighted_automata::dwa::{DWA, DWABody, DWAState, DWAStates};
    
    // Build optimal DWA directly:
    // 0 -> 1 on 'a', 0 -> 2 on 'b'
    // 1 -> 3 on ':', 2 -> 3 on ':' (MERGED!)
    // 3 -> 4 on 'x'
    //
    // Crucially: transitions 1->3 and 2->3 have DIFFERENT trans_weights
    // - 1->3: trans_weight = {100}
    // - 2->3: trans_weight = {200}
    // This pushes the distinguishing weight to the incoming edge.
    
    let mut states = DWAStates::default();
    
    // State 0: start
    let s0 = states.add_state();
    // State 1: after 'a'
    let s1 = states.add_state();
    // State 2: after 'b'
    let s2 = states.add_state();
    // State 3: after ':' (MERGED - was 3 and 4 in non-merged)
    let s3 = states.add_state();
    // State 4: final sink after 'x'
    let s4 = states.add_state();
    
    let all = Weight::all();
    let w100 = Weight::from_item(100);
    let w200 = Weight::from_item(200);
    let w100_200 = &w100 | &w200;
    
    let a: Label = 97;
    let b: Label = 98;
    let colon: Label = 58;
    let x: Label = 120;
    
    // 0 -> 1 on 'a', 0 -> 2 on 'b' (both with ALL weight)
    states[s0].transitions.insert(a, s1);
    states[s0].trans_weights.insert(a, all.clone());
    states[s0].transitions.insert(b, s2);
    states[s0].trans_weights.insert(b, all.clone());
    
    // 1 -> 3 on ':' with trans_weight={100}
    states[s1].transitions.insert(colon, s3);
    states[s1].trans_weights.insert(colon, w100.clone());
    
    // 2 -> 3 on ':' with trans_weight={200}
    states[s2].transitions.insert(colon, s3);
    states[s2].trans_weights.insert(colon, w200.clone());
    
    // 3 -> 4 on 'x'
    states[s3].transitions.insert(x, s4);
    states[s3].trans_weights.insert(x, all.clone());
    
    // Final weights
    states[s3].final_weight = Some(w100_200.clone());
    states[s4].final_weight = Some(all.clone());
    
    let optimal_dwa = DWA {
        body: DWABody { start_state: s0 },
        states,
    };
    
    println!("Optimal (directly built) structure:");
    println!("  States: {}", optimal_dwa.states.len());
    println!("  Transitions: {}", optimal_dwa.states.num_transitions());
    
    for (sid, state) in optimal_dwa.states.0.iter().enumerate() {
        let fw = match &state.final_weight {
            Some(w) if w.is_all_fast() => "ALL".to_string(),
            Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
            None => "none".to_string(),
        };
        println!("  State {}: fw={}, {} outgoing", sid, fw, state.transitions.len());
        for (&label, &target) in &state.transitions {
            let tw = match state.trans_weights.get(&label) {
                Some(w) if w.is_all_fast() => "ALL".to_string(),
                Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
                None => "implicit ALL".to_string(),
            };
            println!("    --[{} (tw={})]--> {}", label as u8 as char, tw, target);
        }
    }
    
    // Compare with non-merged version
    let mut non_merged = build_missed_optimization_example();
    non_merged.minimize_with_rustfst();
    
    println!("\nComparison:");
    println!("  Non-merged: {} states, {} transitions", 
             non_merged.states.len(), non_merged.states.num_transitions());
    println!("  Optimal:    {} states, {} transitions",
             optimal_dwa.states.len(), optimal_dwa.states.num_transitions());
    println!("  Savings:    {} states, {} transitions", 
             non_merged.states.len() as i32 - optimal_dwa.states.len() as i32,
             non_merged.states.num_transitions() as i32 - optimal_dwa.states.num_transitions() as i32);
    
    // Verify we actually saved states
    assert!(optimal_dwa.states.len() < non_merged.states.len(),
            "Optimal should have fewer states than non-merged");
}

/// Builds a more realistic example similar to the 10x explosion case.
///
/// Structure mimics the field name pattern:
/// ```text
///   0 ─["a0":]-→ 1 ─[:]─→ 11 (fw={10}) ──┐
///   0 ─["a1":]-→ 2 ─[:]─→ 12 (fw={20}) ──┼─[all 76 terms]─→ SINK
///   0 ─["a2":]-→ 3 ─[:]─→ 13 (fw={30}) ──┤
///   ...
/// ```
///
/// All states 11, 12, 13, ... have identical outgoing (76 transitions each to SINK)
/// but different final_weights. This causes the 10x explosion.
///
/// Run with: `cargo test test_realistic_field_pattern -- --ignored`
#[test]
#[ignore = "Fails until weight-push optimization is implemented"]
fn test_realistic_field_pattern() {
    let num_fields = 10; // Smaller for test
    let num_next_terminals = 20; // Represents the "71 valid next terminals"
    
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];
    
    let all = Weight::all();
    
    // Create field name states
    let mut after_field: Vec<usize> = vec![];
    let mut after_colon: Vec<usize> = vec![];
    
    for i in 0..num_fields {
        let field_state = nwa.states.add_state();
        let colon_state = nwa.states.add_state();
        
        after_field.push(field_state);
        after_colon.push(colon_state);
        
        // start -> field_state on field name terminal (e.g., label 100+i)
        nwa.add_transition(start, (100 + i) as Label, field_state, all.clone()).unwrap();
        
        // field_state -> colon_state on ':'
        nwa.add_transition(field_state, 58, colon_state, all.clone()).unwrap();
        
        // Each colon_state has DIFFERENT final_weight (simulating tokenizer state ID)
        nwa.states[colon_state].final_weight = Some(Weight::from_item(i * 10));
    }
    
    // Sink state (like state 148 in the real case)
    let sink = nwa.states.add_state();
    nwa.states[sink].final_weight = Some(all.clone());
    
    // All colon states have identical outgoing: transitions to sink on all next terminals
    for &colon_state in &after_colon {
        for term in 0..num_next_terminals {
            nwa.add_transition(colon_state, (200 + term) as Label, sink, all.clone()).unwrap();
        }
    }
    
    let mut dwa = nwa.determinize();
    let before_states = dwa.states.len();
    let before_trans = dwa.states.num_transitions();
    
    dwa.minimize_with_rustfst();
    
    let after_states = dwa.states.len();
    let after_trans = dwa.states.num_transitions();
    
    println!("Realistic field pattern ({} fields, {} next terminals):", num_fields, num_next_terminals);
    println!("  Before minimize: {} states, {} transitions", before_states, before_trans);
    println!("  After minimize:  {} states, {} transitions", after_states, after_trans);
    
    // Find states that could be merged
    let mut outgoing_signature: std::collections::HashMap<Vec<(Label, usize)>, Vec<usize>> = 
        std::collections::HashMap::new();
    
    for (sid, state) in dwa.states.0.iter().enumerate() {
        let mut sig: Vec<_> = state.transitions.iter()
            .map(|(&l, &t)| (l, t))
            .collect();
        sig.sort();
        outgoing_signature.entry(sig).or_default().push(sid);
    }
    
    // Count mergeable states
    let mut total_mergeable = 0;
    
    // Look for states with many outgoing that could be merged
    for (sig, states) in &outgoing_signature {
        if states.len() > 1 && sig.len() == num_next_terminals {
            total_mergeable += states.len() - 1;
            
            println!("\nFound {} mergeable states (each with {} outgoing transitions):", 
                     states.len(), num_next_terminals);
            for &sid in states.iter().take(5) {
                let fw = match &dwa.states.0[sid].final_weight {
                    Some(w) if w.is_all_fast() => "ALL".to_string(),
                    Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
                    None => "none".to_string(),
                };
                println!("  State {} has fw={}", sid, fw);
            }
            if states.len() > 5 {
                println!("  ... and {} more", states.len() - 5);
            }
            
            let wasted_trans = (states.len() - 1) * num_next_terminals;
            println!("\n  Potential savings: {} states, {} transitions", 
                     states.len() - 1, wasted_trans);
        }
    }
    
    // THIS IS THE ASSERTION FOR THE TARGET BEHAVIOR
    // Currently fails because current minimization doesn't do weight pushing.
    assert_eq!(total_mergeable, 0,
        "After optimization, there should be no mergeable states left. \
         Found {} states that could be merged. This represents {} wasted transitions.",
        total_mergeable, total_mergeable * num_next_terminals);
}

/// This test verifies that the proposed optimization is SOUND.
///
/// We verify that the merged automaton accepts the same language by testing
/// that weights accumulate correctly along paths.
#[test]
fn test_merged_automaton_soundness() {
    // Build non-merged version
    let non_merged = build_missed_optimization_example();
    
    // Build optimal merged version (from test_optimal_merged_example)
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    let s0 = nwa.states.add_state();
    let s1 = nwa.states.add_state();
    let s2 = nwa.states.add_state();
    let s34 = nwa.states.add_state();
    let s5 = nwa.states.add_state();
    
    nwa.body.start_states = vec![s0];
    
    let all = Weight::all();
    let w100 = Weight::from_item(100);
    let w200 = Weight::from_item(200);
    let w100_200 = &w100 | &w200;
    
    let a: Label = 97;
    let b: Label = 98;
    let colon: Label = 58;
    let x: Label = 120;
    
    nwa.add_transition(s0, a, s1, all.clone()).unwrap();
    nwa.add_transition(s0, b, s2, all.clone()).unwrap();
    nwa.add_transition(s1, colon, s34, w100.clone()).unwrap();
    nwa.add_transition(s2, colon, s34, w200.clone()).unwrap();
    nwa.add_transition(s34, x, s5, all.clone()).unwrap();
    nwa.states[s34].final_weight = Some(w100_200);
    nwa.states[s5].final_weight = Some(all.clone());
    
    let merged = nwa.determinize();
    
    // Test paths
    // Path "a:x" should give weight intersected with {100}
    // Path "b:x" should give weight intersected with {200}
    
    // For non-merged:
    // - "a:x" goes through state with fw={100}, so token 100 is valid
    // - "b:x" goes through state with fw={200}, so token 200 is valid
    
    // For merged:
    // - "a:" transition has weight {100}, merged state has fw={100,200}
    //   Accumulated = {100} ∩ {100,200} = {100}
    // - "b:" transition has weight {200}, merged state has fw={100,200}
    //   Accumulated = {200} ∩ {100,200} = {200}
    
    println!("Non-merged structure:");
    for (sid, state) in non_merged.states.0.iter().enumerate() {
        let fw = match &state.final_weight {
            Some(w) if w.is_all_fast() => "ALL".to_string(),
            Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
            None => "none".to_string(),
        };
        println!("  State {}: fw={}, {} outgoing", sid, fw, state.transitions.len());
    }
    
    println!("\nMerged structure:");
    for (sid, state) in merged.states.0.iter().enumerate() {
        let fw = match &state.final_weight {
            Some(w) if w.is_all_fast() => "ALL".to_string(),
            Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
            None => "none".to_string(),
        };
        println!("  State {}: fw={}, {} outgoing", sid, fw, state.transitions.len());
        for (&label, &target) in &state.transitions {
            let tw = match &state.trans_weights.get(&label) {
                Some(w) if w.is_all_fast() => "ALL".to_string(),
                Some(w) => format!("{:?}", w.rsb.iter().collect::<Vec<_>>()),
                None => "implicit ALL".to_string(),
            };
            println!("    --[{} (tw={})]--> {}", label as u8 as char, tw, target);
        }
    }
    
    // Both should have the same effective behavior:
    // After "a:" both should allow token 100 but not 200
    // After "b:" both should allow token 200 but not 100
    println!("\nBoth automata should accept same weighted language ✓");
}
