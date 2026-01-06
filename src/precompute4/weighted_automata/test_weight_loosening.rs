//! Tests for the weight loosening algorithm in minimization.rs

use crate::precompute4::weighted_automata::dwa::DWA;
use crate::precompute4::weighted_automata::common::Weight;

/// Test that weight loosening preserves semantics on a simple acyclic DWA.
/// 
/// The DWA structure:
/// ```
///   State 0 (start) --1--> State 1 --2--> State 2 (final, weight=[0..=10])
///                    w=[5..=100]  w=[0..=50]
/// ```
/// 
/// Token 5 can reach state 2 and accept (5 is in [5,100] and [0,50] and [0,10]).
/// Token 0 cannot reach state 1 (0 not in [5,100]).
/// Token 100 can reach state 2 but can't accept (100 not in [0,10]).
#[test]
fn test_weight_loosening_preserves_semantics() {
    let mut dwa = DWA::new();
    
    // State 0 is already created as start state
    let s1 = dwa.add_state();
    let s2 = dwa.add_state();
    
    // Add transitions
    // State 0 --label=1--> State 1, weight = [5..=100]
    dwa.add_transition(0, 1, s1, Weight::from_ranges(&[(5, 100)])).unwrap();
    
    // State 1 --label=2--> State 2, weight = [0..=50]
    dwa.add_transition(s1, 2, s2, Weight::from_ranges(&[(0, 50)])).unwrap();
    
    // State 2 is final with weight [0..=10]
    dwa.set_final_weight(s2, Weight::from_ranges(&[(0, 10)])).unwrap();
    
    // Collect semantics before loosening
    let test_words: Vec<Vec<i32>> = vec![
        vec![1, 2],  // Path to accepting state
        vec![1],     // Stop at state 1 (not accepting)
        vec![],      // Empty word (stay at start, not accepting)
        vec![3],     // Invalid label
    ];
    
    let weights_before: Vec<Weight> = test_words.iter()
        .map(|w| dwa.eval_word_weight(w))
        .collect();
    
    println!("Before loosening:");
    for (word, weight) in test_words.iter().zip(&weights_before) {
        println!("  {:?} -> {}", word, weight);
    }
    
    // Apply weight loosening
    let changed = dwa.loosen_weights_for_minimize();
    println!("\nWeight loosening changed something: {}", changed);
    
    // Collect semantics after loosening
    let weights_after: Vec<Weight> = test_words.iter()
        .map(|w| dwa.eval_word_weight(w))
        .collect();
    
    println!("\nAfter loosening:");
    for (word, weight) in test_words.iter().zip(&weights_after) {
        println!("  {:?} -> {}", word, weight);
    }
    
    // Verify semantics are preserved
    for (i, (before, after)) in weights_before.iter().zip(&weights_after).enumerate() {
        assert_eq!(
            before, after,
            "Semantics changed for word {:?}: before={}, after={}",
            test_words[i], before, after
        );
    }
    
    println!("\n✓ Weight loosening preserved semantics!");
}

/// Test that weight loosening does something useful on a simple case.
/// 
/// DWA structure:
/// ```
///   State 0 (start) --1--> State 1 (final, weight=[5..=10])
///                    w=[0]
/// ```
/// 
/// Only token 0 can reach state 1, but state 1 only accepts tokens [5..=10].
#[test]
fn test_weight_loosening_loosens_unreachable() {
    let mut dwa = DWA::new();
    
    let s1 = dwa.add_state();
    
    // Transition from 0 to 1 only allows token 0
    dwa.add_transition(0, 1, s1, Weight::from_item(0)).unwrap();
    
    // State 1 is final but only for tokens [5..=10]
    dwa.set_final_weight(s1, Weight::from_ranges(&[(5, 10)])).unwrap();
    
    println!("Before loosening:");
    println!("  State 1 final weight: {:?}", dwa.states[s1].final_weight);
    println!("  Transition 0->1 weight: {:?}", dwa.states[0].trans_weights.get(&1));
    
    // Verify no word accepts
    let accept_0 = dwa.eval_word_weight(&[1]);
    assert!(accept_0.is_empty(), "Expected no acceptance, got {}", accept_0);
    
    // Apply weight loosening
    let changed = dwa.loosen_weights_for_minimize();
    assert!(changed, "Weight loosening should have made changes");
    
    println!("\nAfter loosening (changed={}):", changed);
    println!("  State 1 final weight: {:?}", dwa.states[s1].final_weight);
    println!("  Transition 0->1 weight: {:?}", dwa.states[0].trans_weights.get(&1));
    
    // Verify semantics still preserved
    let accept_after = dwa.eval_word_weight(&[1]);
    assert!(accept_after.is_empty(), "Expected no acceptance after loosening, got {}", accept_after);
    
    // STRATEGY: Forward Loosening (Pre-only)
    // - Pre(0) = ALL. !Pre(0) = EMPTY.
    // - Pre(1) = {0}. !Pre(1) = ALL \ {0}.
    
    // Transition 0->1: Source is 0. 
    // Loosening uses !Pre(0) which is empty. 
    // So transition weight should remain UNCHANGED.
    let trans_weight = dwa.states[0].trans_weights.get(&1).unwrap();
    assert!(trans_weight.contains(0), "Should still contain original token 0");
    assert!(!trans_weight.contains(1), "Should NOT contain token 1 (reachable source shouldn't loosen outgoing)");
    
    // Final weight at 1:
    // Loosening uses !Pre(1) which is ALL \ {0}.
    // Original was [5..10].
    // New should be [5..10] | (ALL \ {0}) = ALL \ {0}.
    let final_weight = dwa.states[s1].final_weight.as_ref().unwrap();
    assert!(final_weight.contains(5), "Should contain original token 5");
    assert!(final_weight.contains(100), "Should contain don't-care token 100");
    assert!(!final_weight.contains(0), "Should NOT contain 0 (it is in Pre(1) so it matters)");
    
    println!("\n✓ Weight loosening worked correctly!");
}

/// Test that cyclic DWAs are skipped (returns false without modification).
#[test]
fn test_weight_loosening_skips_cyclic() {
    let mut dwa = DWA::new();
    
    let s1 = dwa.add_state();
    
    // Create a cycle: 0 -> 1 -> 0
    dwa.add_transition(0, 1, s1, Weight::all()).unwrap();
    dwa.add_transition(s1, 2, 0, Weight::all()).unwrap();
    
    // Verify it's cyclic
    assert!(dwa.is_cyclic(), "DWA should be cyclic");
    
    // Weight loosening should return false for cyclic DWAs
    let changed = dwa.loosen_weights_for_minimize();
    assert!(!changed, "Weight loosening should return false for cyclic DWAs");
    
    println!("✓ Cyclic DWA correctly skipped!");
}
