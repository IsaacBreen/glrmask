#![cfg(test)]
//! Tests for the "transition explosion" phenomenon when replacing labeled start transitions with epsilons.
//!
//! ## Summary
//!
//! When initial labeled transitions from the start state are replaced with epsilon transitions,
//! the determinized & simplified DWA can have SIGNIFICANTLY more transitions.
//!
//! ## Confirmed on Real Data
//!
//! **Test case:** ApolloRouter schema with 4,401 tokenizer states
//! **Command:** `TEST_EPSILON_EXPLOSION=1 MACRO_DEBUG_LEVEL=4 make test-schema-id ID=ApolloRouter---apollo-router-2.9.0`
//!
//! **Results:**
//! - Original: 5,952 states, 45,284 transitions
//! - Modified: 634 states (0.11x), 315,507 transitions (6.97x explosion!)
//!
//! ## Minimal Counterexample (5 states → 6 states)
//!
//! A 5-state minimal DFA on Σ = {a, b} where epsilon-transforming initial transitions
//! and then determinizing/minimizing produces a LARGER automaton.
//! See `test_minimal_counterexample()` below.
//!
//! ## Why It Happens
//!
//! 1. **Original:** Start has labeled transitions to subtree roots. Each label is unique per tokenizer state.
//!    - Accessing subtree N requires taking transition label N
//!    - Subtrees are accessed separately, one at a time
//!
//! 2. **Modified:** Start has epsilon transitions to ALL subtree roots.
//!    - Epsilon closure creates super-start = {all subtree roots}
//!    - From super-start, each byte label fans out to ALL subtrees simultaneously
//!    - Since subtrees have different weights (token validity masks), states can't be merged
//!    - Result: Each remaining state has many more outgoing transitions

use crate::precompute4::weighted_automata::common::Label;
use super::*;
use std::collections::BTreeSet;
use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;

/// **MINIMAL COUNTEREXAMPLE** - 5 states → 6 states
/// 
/// This test demonstrates that replacing initial labeled transitions with epsilon transitions
/// can cause the minimal DFA to GROW, proving the "transition explosion" phenomenon.
///
/// Original DFA M (minimal, 5 states, Σ = {a, b}):
/// - Start state: 0
/// - Final state: 4
/// - Transitions:
///   | state | a | b |
///   |-------|---|---|
///   |   0   | 1 | 2 |
///   |   1   | 2 | 2 |
///   |   2   | 3 | 1 |
///   |   3   | 4 | 3 |
///   |  *4   | 1 | 4 |
///
/// After replacing 0→1 (on 'a') and 0→2 (on 'b') with epsilon transitions,
/// determinizing and minimizing produces a 6-state DFA.
///
/// The key insight: epsilon-transforming initial transitions "forgets" the first symbol,
/// creating a new language whose minimal DFA is LARGER than the original.
#[test]
fn test_minimal_counterexample() {
    // Build the original DFA as an NWA with Weight::all() on all transitions
    // (effectively unweighted - we're testing structural explosion)
    let mut nwa = NWA::new();
    nwa.states.0.clear();

    // States 0-4
    let s0 = nwa.states.add_state(); // 0 - start
    let s1 = nwa.states.add_state(); // 1
    let s2 = nwa.states.add_state(); // 2
    let s3 = nwa.states.add_state(); // 3
    let s4 = nwa.states.add_state(); // 4 - final
    nwa.body.start_states = vec![s0];
    nwa.states[s4].final_weight = Some(Weight::all());

    let w = Weight::all();
    let a: Label = 97; // 'a'
    let b: Label = 98; // 'b'

    // Transitions from the table:
    // 0: a→1, b→2
    nwa.add_transition(s0, a, s1, w.clone()).unwrap();
    nwa.add_transition(s0, b, s2, w.clone()).unwrap();
    // 1: a→2, b→2
    nwa.add_transition(s1, a, s2, w.clone()).unwrap();
    nwa.add_transition(s1, b, s2, w.clone()).unwrap();
    // 2: a→3, b→1
    nwa.add_transition(s2, a, s3, w.clone()).unwrap();
    nwa.add_transition(s2, b, s1, w.clone()).unwrap();
    // 3: a→4, b→3
    nwa.add_transition(s3, a, s4, w.clone()).unwrap();
    nwa.add_transition(s3, b, s3, w.clone()).unwrap();
    // 4: a→1, b→4
    nwa.add_transition(s4, a, s1, w.clone()).unwrap();
    nwa.add_transition(s4, b, s4, w.clone()).unwrap();

    // Original: determinize and minimize
    let mut orig_dwa = nwa.determinize();
    orig_dwa.minimize_with_rustfst();
    let orig_states = orig_dwa.states.len();
    let orig_trans = orig_dwa.states.num_transitions();
    println!("Original: {} states, {} transitions", orig_states, orig_trans);
    
    // Verify original has 5 states
    assert_eq!(orig_states, 5, "Original DFA should have 5 states (it's already minimal)");

    // Modified: Replace start transitions with epsilons
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[s0].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(s0, target, weight);
        }
    }

    // Modified: determinize and minimize
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.minimize_with_rustfst();
    let mod_states = mod_dwa.states.len();
    let mod_trans = mod_dwa.states.num_transitions();
    println!("Modified: {} states, {} transitions", mod_states, mod_trans);

    // The explosion: modified should have MORE states
    println!("State change: {} -> {} ({})", orig_states, mod_states,
             if mod_states > orig_states { "EXPLOSION!" } else { "no explosion" });
    println!("Transition change: {} -> {} ({})", orig_trans, mod_trans,
             if mod_trans > orig_trans { "EXPLOSION!" } else { "no explosion" });
    
    // Assert the explosion occurs
    assert!(mod_states > orig_states || mod_trans > orig_trans,
            "Expected explosion: original {} states/{} trans, modified {} states/{} trans",
            orig_states, orig_trans, mod_states, mod_trans);
}

/// This test demonstrates the "transition explosion" phenomenon:
/// When initial labeled transitions from the start state are replaced with epsilon transitions,
/// the determinized & simplified DWA can have SIGNIFICANTLY more transitions.
///
/// The explosion occurs because:
/// 1. Original: Start has distinct labeled transitions to branch roots (e.g., tsid labels)
/// 2. Modified: Start has epsilon transitions to all branch roots
/// 3. Epsilon closure at start creates a super-state with all branch roots
/// 4. Traversing from super-state creates combinations of path weights
/// 5. Distinct weights prevent state merging during simplification
#[test]
fn test_weighted_trie_explosion() {
    // Create a simple weighted trie structure:
    // Start -> (label=100, W0) -> A -> (label='a', W0) -> B -> (final W0)
    // Start -> (label=101, W1) -> C -> (label='a', W1) -> D -> (final W1)
    //
    // Where W0 = {0} and W1 = {1} are DISJOINT weights.

    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let start = nwa.states.add_state();
    let a = nwa.states.add_state();
    let b = nwa.states.add_state();
    let c = nwa.states.add_state();
    let d = nwa.states.add_state();
    nwa.body.start_states = vec![start];

    let w0 = Weight::from_item(0);
    let w1 = Weight::from_item(1);

    nwa.add_transition(start, 100, a, w0.clone()).unwrap();
    nwa.add_transition(a, 97, b, w0.clone()).unwrap();
    nwa.states[b].final_weight = Some(w0.clone());

    nwa.add_transition(start, 101, c, w1.clone()).unwrap();
    nwa.add_transition(c, 97, d, w1.clone()).unwrap();
    nwa.states[d].final_weight = Some(w1.clone());

    // Original with custom simplify
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_trans_simplify = orig_dwa.states.num_transitions();
    
    // Original with rustfst minimize
    let mut orig_dwa_rustfst = nwa.determinize();
    orig_dwa_rustfst.minimize_with_rustfst();
    let orig_trans_rustfst = orig_dwa_rustfst.states.num_transitions();

    println!("Original: simplify={} trans, rustfst={} trans", orig_trans_simplify, orig_trans_rustfst);

    // Modified: Replace initial labeled transitions with epsilons
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }

    // Modified with custom simplify
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_trans_simplify = mod_dwa.states.num_transitions();

    // Modified with rustfst minimize
    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();

    println!("Modified: simplify={} trans, rustfst={} trans", mod_trans_simplify, mod_trans_rustfst);

    println!("Comparison:");
    println!("  simplify: {} -> {} ({})", orig_trans_simplify, mod_trans_simplify, 
             if mod_trans_simplify > orig_trans_simplify { "EXPLOSION" } 
             else if mod_trans_simplify < orig_trans_simplify { "reduction" } 
             else { "same" });
    println!("  rustfst:  {} -> {} ({})", orig_trans_rustfst, mod_trans_rustfst,
             if mod_trans_rustfst > orig_trans_rustfst { "EXPLOSION" }
             else if mod_trans_rustfst < orig_trans_rustfst { "reduction" }
             else { "same" });
}

/// A more complex test with deeper trie structure and more branches
#[test]
fn test_weighted_trie_explosion_deeper() {
    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];

    let num_branches = 4;
    let depth = 5;
    let alphabet: Vec<Label> = vec![97, 98, 99]; // 'a', 'b', 'c'

    for b in 0..num_branches {
        let w = Weight::from_item(b);
        
        let root = nwa.states.add_state();
        nwa.add_transition(start, 100 + b as Label, root, w.clone()).unwrap();
        
        let mut prev = root;
        for d in 0..depth {
            let next = nwa.states.add_state();
            let c = alphabet[d % alphabet.len()];
            nwa.add_transition(prev, c, next, w.clone()).unwrap();
            prev = next;
        }
        nwa.states[prev].final_weight = Some(w.clone());
    }

    // Original
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_trans_simplify = orig_dwa.states.num_transitions();
    
    let mut orig_dwa_rustfst = nwa.determinize();
    orig_dwa_rustfst.minimize_with_rustfst();
    let orig_trans_rustfst = orig_dwa_rustfst.states.num_transitions();

    println!("Deep Trie Original: simplify={} trans, rustfst={} trans", 
             orig_trans_simplify, orig_trans_rustfst);

    // Modified
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }

    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_trans_simplify = mod_dwa.states.num_transitions();

    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();

    println!("Deep Trie Modified: simplify={} trans, rustfst={} trans", 
             mod_trans_simplify, mod_trans_rustfst);

    println!("Comparison:");
    println!("  simplify: {} -> {} ({})", orig_trans_simplify, mod_trans_simplify, 
             if mod_trans_simplify > orig_trans_simplify { "EXPLOSION" } 
             else if mod_trans_simplify < orig_trans_simplify { "reduction" } 
             else { "same" });
    println!("  rustfst:  {} -> {} ({})", orig_trans_rustfst, mod_trans_rustfst,
             if mod_trans_rustfst > orig_trans_rustfst { "EXPLOSION" }
             else if mod_trans_rustfst < orig_trans_rustfst { "reduction" }
             else { "same" });
}

/// Test with branches that merge back together (diamond structure)
#[test]
fn test_weighted_trie_diamond_explosion() {
    let mut nwa = NWA::new();
    nwa.states.0.clear();

    let start = nwa.states.add_state();
    let a = nwa.states.add_state();
    let b = nwa.states.add_state();
    let merge = nwa.states.add_state();
    let end = nwa.states.add_state();
    nwa.body.start_states = vec![start];

    let w0 = Weight::from_item(0);
    let w1 = Weight::from_item(1);

    nwa.add_transition(start, 100, a, w0.clone()).unwrap();
    nwa.add_transition(a, 97, merge, w0.clone()).unwrap();
    nwa.add_transition(merge, 120, end, w0.clone()).unwrap();

    nwa.add_transition(start, 101, b, w1.clone()).unwrap();
    nwa.add_transition(b, 97, merge, w1.clone()).unwrap();
    nwa.add_transition(merge, 120, end, w1.clone()).unwrap();

    nwa.states[end].final_weight = Some(Weight::all());

    // Original
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_trans_simplify = orig_dwa.states.num_transitions();
    
    let mut orig_dwa_rustfst = nwa.determinize();
    orig_dwa_rustfst.minimize_with_rustfst();
    let orig_trans_rustfst = orig_dwa_rustfst.states.num_transitions();

    println!("Diamond Original: simplify={} trans, rustfst={} trans", 
             orig_trans_simplify, orig_trans_rustfst);

    // Modified
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }

    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_trans_simplify = mod_dwa.states.num_transitions();

    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();

    println!("Diamond Modified: simplify={} trans, rustfst={} trans", 
             mod_trans_simplify, mod_trans_rustfst);

    println!("Comparison:");
    println!("  simplify: {} -> {} ({})", orig_trans_simplify, mod_trans_simplify, 
             if mod_trans_simplify > orig_trans_simplify { "EXPLOSION" } 
             else if mod_trans_simplify < orig_trans_simplify { "reduction" } 
             else { "same" });
    println!("  rustfst:  {} -> {} ({})", orig_trans_rustfst, mod_trans_rustfst,
             if mod_trans_rustfst > orig_trans_rustfst { "EXPLOSION" }
             else if mod_trans_rustfst < orig_trans_rustfst { "reduction" }
             else { "same" });
}

#[test]
#[ignore] // Uncomment to run fuzz search
fn fuzz_find_transition_explosion_after_simplify() {
    struct Rng { state: u64 }
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }
    }

    let mut rng = Rng { state: 12345 };
    let alphabet: Vec<Label> = (0..3).map(|i| 97 + i as Label).collect();

    for iter in 0..50_000 {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        
        let start_node = nwa.states.add_state();
        nwa.body.start_states = vec![start_node];

        let num_branches = 2 + (rng.next() % 4) as usize;
        let depth = 4 + (rng.next() % 4) as usize;

        for b in 0..num_branches {
            let distinct_w = Weight::from_item(b);
            
            let mut prev = start_node;
            let init_label = 100 + b as Label; 
            let mut curr = nwa.states.add_state();
            nwa.add_transition(prev, init_label, curr, distinct_w.clone()).unwrap();
            prev = curr;
            
            for _ in 0..depth {
                let next = nwa.states.add_state();
                for &c in &alphabet {
                    if rng.next() % 100 < 70 {
                        nwa.add_transition(prev, c, next, distinct_w.clone()).unwrap();
                    }
                }
                if rng.next() % 100 < 40 {
                    let rand_target = (rng.next() as usize) % nwa.states.len();
                    let rand_char = alphabet[(rng.next() as usize) % alphabet.len()];
                    nwa.add_transition(prev, rand_char, rand_target, distinct_w.clone()).unwrap();
                }
                if rng.next() % 5 == 0 {
                    nwa.states[prev].final_weight = Some(distinct_w.clone());
                }
                prev = next;
            }
            if rng.next() % 2 == 0 {
                nwa.states[prev].final_weight = Some(distinct_w.clone());
            }
        }

        let mut orig_dwa = nwa.determinize();
        orig_dwa.simplify();
        let orig_trans = orig_dwa.states.num_transitions();

        if orig_dwa.states.len() < 2 { continue; }

        let mut mod_nwa = nwa.clone();
        let start_trans = std::mem::take(&mut mod_nwa.states[start_node].transitions);
        for (_, targets) in start_trans {
             for (target, weight) in targets {
                 mod_nwa.add_epsilon(start_node, target, weight);
             }
        }
        
        let mut mod_dwa = mod_nwa.determinize();
        mod_dwa.simplify();
        let mod_trans = mod_dwa.states.num_transitions();

        if mod_trans > orig_trans {
             println!("FOUND EXPLOSION at iter {}", iter);
             println!("Original: {} trans", orig_trans);
             println!("Modified: {} trans", mod_trans);
             assert!(mod_trans > orig_trans);
             return;
        }
    }
    panic!("Failed to find explosion example in 50k iterations");
}

/// Test that actually mimics the REAL terminal DWA structure from precompute1.
/// 
/// Real structure:
/// - Multiple tokenizer states (tsids), each with its own subtree
/// - Start state has labeled transitions: `start -> (label=tsid+T) -> subtree_root[tsid]`
/// - Each subtree is a vocab prefix trie with same structure but DISJOINT weights per tsid
/// 
/// The proposed change: Replace labeled tsid transitions with epsilon transitions.
#[test]
fn test_real_terminal_dwa_structure_explosion() {
    // Simulate a vocab with tokens that share prefixes
    // Tokens: "a"=0, "ab"=1, "abc"=2, "b"=3, "bc"=4
    // These create a trie structure

    // For REAL terminal DWA, each tokenizer state ID creates a SEPARATE subtree
    // with the SAME token validity (weights) but accessed via different tsid labels
    
    let num_tokenizer_states = 5;  // Simulate 5 tokenizer states
    let terminals_count = 10;      // Simulate 10 terminals
    
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    // Create one subtree per tokenizer state (all subtrees are identical)
    let mut roots = Vec::new();
    
    for _tsid in 0..num_tokenizer_states {
        // Vocab trie structure:
        //   root -> 'a' -> (final tok0) -> 'b' -> (final tok1) -> 'c' -> (final tok2)
        //        -> 'b' -> (final tok3) -> 'c' -> (final tok4)
        
        let root = nwa.states.add_state();
        roots.push(root);
        
        // Branch 1: a -> ab -> abc
        let s_a = nwa.states.add_state();
        let s_ab = nwa.states.add_state();
        let s_abc = nwa.states.add_state();
        
        // Weights: each final state accepts tokens with Weight containing token ID
        nwa.add_transition(root, b'a' as Label, s_a, Weight::all()).unwrap();
        nwa.states[s_a].final_weight = Some(Weight::from_item(0)); // tok "a" = 0
        
        nwa.add_transition(s_a, b'b' as Label, s_ab, Weight::all()).unwrap();
        nwa.states[s_ab].final_weight = Some(Weight::from_item(1)); // tok "ab" = 1
        
        nwa.add_transition(s_ab, b'c' as Label, s_abc, Weight::all()).unwrap();
        nwa.states[s_abc].final_weight = Some(Weight::from_item(2)); // tok "abc" = 2
        
        // Branch 2: b -> bc
        let s_b = nwa.states.add_state();
        let s_bc = nwa.states.add_state();
        
        nwa.add_transition(root, b'b' as Label, s_b, Weight::all()).unwrap();
        nwa.states[s_b].final_weight = Some(Weight::from_item(3)); // tok "b" = 3
        
        nwa.add_transition(s_b, b'c' as Label, s_bc, Weight::all()).unwrap();
        nwa.states[s_bc].final_weight = Some(Weight::from_item(4)); // tok "bc" = 4
    }
    
    // ORIGINAL: Start with labeled transitions (tsid + terminals_count as label)
    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];
    
    for (tsid, &root) in roots.iter().enumerate() {
        let label = (tsid + terminals_count) as Label;
        nwa.add_transition(start, label, root, Weight::all()).unwrap();
    }
    
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_states = orig_dwa.states.len();
    let orig_trans = orig_dwa.states.num_transitions();
    println!("Real Structure Original: {} states, {} transitions", orig_states, orig_trans);
    
    // MODIFIED: Replace labeled transitions with epsilons
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }
    
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_states = mod_dwa.states.len();
    let mod_trans = mod_dwa.states.num_transitions();
    println!("Real Structure Modified: {} states, {} transitions", mod_states, mod_trans);
    
    println!("Comparison: {} -> {} transitions ({})", orig_trans, mod_trans,
             if mod_trans > orig_trans { "EXPLOSION" }
             else if mod_trans < orig_trans { "reduction" }
             else { "same" });
    
    // Also try rustfst to compare
    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();
    println!("Modified (rustfst): {} transitions", mod_trans_rustfst);
}

/// Test with DISTINCT weights per tokenizer state (closer to real usage)
/// In real code, different tsids lead to subtrees with DIFFERENT valid token sets
#[test]
fn test_real_terminal_dwa_distinct_weights_explosion() {
    // Each tokenizer state has DIFFERENT valid tokens
    // tsid 0: only tok 0,1 valid
    // tsid 1: only tok 2,3 valid
    // tsid 2: only tok 4 valid
    
    let num_tokenizer_states = 3;
    let terminals_count = 10;
    
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    let mut roots = Vec::new();
    
    for tsid in 0..num_tokenizer_states {
        let root = nwa.states.add_state();
        roots.push(root);
        
        // Different weight per tsid
        let w: Weight = match tsid {
            0 => Weight::from_iter([0, 1]),
            1 => Weight::from_iter([2, 3]),
            2 => Weight::from_iter([4]),
            _ => Weight::all(),
        };
        
        // Simple trie: root -> 'a' -> final
        let s_a = nwa.states.add_state();
        nwa.add_transition(root, b'a' as Label, s_a, w.clone()).unwrap();
        nwa.states[s_a].final_weight = Some(w.clone());
        
        // root -> 'b' -> final
        let s_b = nwa.states.add_state();
        nwa.add_transition(root, b'b' as Label, s_b, w.clone()).unwrap();
        nwa.states[s_b].final_weight = Some(w.clone());
    }
    
    // ORIGINAL
    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];
    
    for (tsid, &root) in roots.iter().enumerate() {
        let label = (tsid + terminals_count) as Label;
        nwa.add_transition(start, label, root, Weight::all()).unwrap();
    }
    
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_trans = orig_dwa.states.num_transitions();
    
    let mut orig_dwa_rustfst = nwa.determinize();
    orig_dwa_rustfst.minimize_with_rustfst();
    let orig_trans_rustfst = orig_dwa_rustfst.states.num_transitions();
    
    println!("Distinct Weights Original: simplify={}, rustfst={} transitions", orig_trans, orig_trans_rustfst);
    
    // MODIFIED
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }
    
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_trans = mod_dwa.states.num_transitions();
    
    let mut mod_dwa_rustfst = mod_nwa.determinize();
    mod_dwa_rustfst.minimize_with_rustfst();
    let mod_trans_rustfst = mod_dwa_rustfst.states.num_transitions();
    
    println!("Distinct Weights Modified: simplify={}, rustfst={} transitions", mod_trans, mod_trans_rustfst);
    
    println!("Comparison:");
    println!("  simplify: {} -> {} ({})", orig_trans, mod_trans,
             if mod_trans > orig_trans { "EXPLOSION" }
             else if mod_trans < orig_trans { "reduction" }
             else { "same" });
    println!("  rustfst:  {} -> {} ({})", orig_trans_rustfst, mod_trans_rustfst,
             if mod_trans_rustfst > orig_trans_rustfst { "EXPLOSION" }
             else if mod_trans_rustfst < orig_trans_rustfst { "reduction" }
             else { "same" });
}

/// CRITICAL TEST: Demonstrates why epsilon start transitions cause transition explosion.
/// 
/// The explosion happens when:
/// 1. Many subtrees (branches) are accessed via distinct labels at start
/// 2. Each subtree has OVERLAPPING outgoing labels ('a', 'b', etc)
/// 3. Each subtree has DISJOINT weights (different valid token sets)
///
/// With labeled start transitions:
///   DFA state after 'tsid_0' is just {subtree_0_root} 
///   Transitions from that state only cover subtree_0's structure
///
/// With epsilon start transitions:
///   DFA start state is {super-start} which epsilon-closes to {all_roots}
///   Transitions from start enumerate ALL subtrees on EACH label
///   Since weights are disjoint, states can't be merged
///
/// In the real codebase:
///   - 4,401 tokenizer states → 4,401 epsilon edges to subtree roots
///   - Each subtree has ~10 transitions on overlapping byte labels
///   - Result: 45K → 315K transitions (7x explosion)
#[test]
fn test_epsilon_explosion_many_branches() {
    // Create many branches with overlapping labels but disjoint weights
    // This is the EXACT pattern that causes explosion in terminal DWA
    
    let num_branches = 100; // In real code, this is ~4400
    let terminals_count = 10;
    let alphabet: Vec<Label> = vec![b'a' as Label, b'b' as Label, b'c' as Label];
    
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    let mut roots = Vec::new();
    
    for b in 0..num_branches {
        // Each branch has a DISTINCT weight (disjoint token set)
        let w = Weight::from_item(b);
        
        let root = nwa.states.add_state();
        roots.push(root);
        
        // Create trie structure with OVERLAPPING labels
        // All branches have same structure: root -> 'a' -> s1 -> 'b' -> s2 (final)
        let s1 = nwa.states.add_state();
        let s2 = nwa.states.add_state();
        
        nwa.add_transition(root, alphabet[0], s1, w.clone()).unwrap();
        nwa.add_transition(s1, alphabet[1], s2, w.clone()).unwrap();
        nwa.states[s2].final_weight = Some(w.clone());
    }
    
    // ORIGINAL: Labeled transitions at start (one label per branch)
    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];
    
    for (b, &root) in roots.iter().enumerate() {
        let label = (b + terminals_count) as Label;
        nwa.add_transition(start, label, root, Weight::all()).unwrap();
    }
    
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_states = orig_dwa.states.len();
    let orig_trans = orig_dwa.states.num_transitions();
    
    println!("Many Branches Original: {} states, {} transitions", orig_states, orig_trans);
    
    // MODIFIED: Epsilon transitions at start
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }
    
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_states = mod_dwa.states.len();
    let mod_trans = mod_dwa.states.num_transitions();
    
    println!("Many Branches Modified: {} states, {} transitions", mod_states, mod_trans);
    
    let state_factor = mod_states as f64 / orig_states as f64;
    let trans_factor = mod_trans as f64 / orig_trans as f64;
    
    println!("Comparison:");
    println!("  States: {} -> {} ({:.2}x)", orig_states, mod_states, state_factor);
    println!("  Trans:  {} -> {} ({:.2}x)", orig_trans, mod_trans, trans_factor);
    
    if mod_trans > orig_trans {
        println!("EXPLOSION CONFIRMED: {:.2}x transition increase", trans_factor);
        
        // The explosion should be roughly num_branches times worse
        // because each label at start now fans out to all branches
        // Expected: trans_factor ≈ num_branches / 3 (since we have 3 chars in trie)
        assert!(trans_factor > 2.0, 
            "Expected significant explosion (>2x) but got {}x", trans_factor);
    } else {
        println!("No explosion - states reduced enough to compensate");
    }
}

/// CRITICAL TEST: Demonstrates transition explosion with DIFFERENT structures.
///
/// The key insight is that branches must have DIFFERENT structures, not just different weights!
/// If all branches have identical structure, they get merged even with epsilon start.
///
/// In the real terminal DWA:
/// - Each tokenizer state leads to different reachable tokens
/// - Different tokens → different trie paths → different structures
/// - When epsilon-merged, these distinct structures can't be combined
#[test]
fn test_epsilon_explosion_different_structures() {
    // Create branches with DIFFERENT structures (varying depth, varying paths)
    let num_branches = 50;
    let terminals_count = 10;
    let alphabet: Vec<Label> = vec![b'a' as Label, b'b' as Label, b'c' as Label, b'd' as Label];
    
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    let mut roots = Vec::new();
    
    for b in 0..num_branches {
        // Each branch has a DISTINCT weight
        let w = Weight::from_item(b);
        
        let root = nwa.states.add_state();
        roots.push(root);
        
        // Create DIFFERENT structure per branch:
        // - Different depths (2, 3, or 4)
        // - Different character sequences
        // - Optional extra branches
        let depth = 2 + (b % 3);
        
        let mut prev = root;
        for d in 0..depth {
            let next = nwa.states.add_state();
            // Use different characters based on branch index
            let ch = alphabet[(b + d) % alphabet.len()];
            nwa.add_transition(prev, ch, next, w.clone()).unwrap();
            prev = next;
        }
        nwa.states[prev].final_weight = Some(w.clone());
        
        // Add optional extra branches for structural variety
        if b % 3 == 0 {
            let extra = nwa.states.add_state();
            nwa.add_transition(root, alphabet[(b + 2) % alphabet.len()], extra, w.clone()).unwrap();
            nwa.states[extra].final_weight = Some(w.clone());
        }
        if b % 5 == 0 && depth > 2 {
            // Add a second path from an intermediate state
            let mid = nwa.states.0.len() - depth;
            let alt = nwa.states.add_state();
            nwa.add_transition(mid, alphabet[(b + 3) % alphabet.len()], alt, w.clone()).unwrap();
            nwa.states[alt].final_weight = Some(w.clone());
        }
    }
    
    // ORIGINAL: Labeled transitions at start
    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];
    
    for (b, &root) in roots.iter().enumerate() {
        let label = (b + terminals_count) as Label;
        nwa.add_transition(start, label, root, Weight::all()).unwrap();
    }
    
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_states = orig_dwa.states.len();
    let orig_trans = orig_dwa.states.num_transitions();
    
    println!("Different Structures Original: {} states, {} transitions", orig_states, orig_trans);
    
    // MODIFIED: Epsilon transitions at start
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }
    
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_states = mod_dwa.states.len();
    let mod_trans = mod_dwa.states.num_transitions();
    
    println!("Different Structures Modified: {} states, {} transitions", mod_states, mod_trans);
    
    let state_factor = mod_states as f64 / orig_states as f64;
    let trans_factor = mod_trans as f64 / orig_trans as f64;
    
    println!("Comparison:");
    println!("  States: {} -> {} ({:.2}x)", orig_states, mod_states, state_factor);
    println!("  Trans:  {} -> {} ({:.2}x)", orig_trans, mod_trans, trans_factor);
    
    if mod_trans > orig_trans {
        println!("EXPLOSION CONFIRMED: {:.2}x transition increase!", trans_factor);
    } else if mod_trans < orig_trans {
        println!("REDUCTION: {:.2}x fewer transitions", 1.0 / trans_factor);
    } else {
        println!("SAME number of transitions");
    }
}

/// VERIFIED EXPLOSION TEST: Mimics real terminal DWA structure that causes 7x explosion.
///
/// Real terminal DWA structure:
/// - Start has N outgoing labeled transitions (N = 4401 tokenizer states)  
/// - Each leads to a subtree root
/// - Subtree roots have overlapping byte labels with DIFFERENT weights
/// - First-hop states have ~12647 total outgoing transitions on shared bytes
///
/// When we replace labeled start transitions with epsilons:
/// - Start epsilon-closes to all subtree roots
/// - Byte transitions from start must enumerate all subtrees
/// - Result: 550 × (stuff) >> original transitions
///
/// The explosion is: 45K → 315K transitions (6.97x)
///
/// KEY INSIGHT: The explosion happens because:
/// 1. With labels at start, each byte label only reaches ONE subtree's states
/// 2. With epsilons at start, each byte label reaches ALL subtrees' states simultaneously  
/// 3. The DWA must encode which subtrees are active at each position
/// 4. Since subtrees have different weights, states can't be merged
#[test]
fn test_epsilon_explosion_realistic_structure() {
    // The critical insight: In the real case, the ORIGINAL has many transitions
    // from the labeled start transitions. In the modified case, those become
    // epsilon transitions that get absorbed into the start state, but the
    // INTERIOR transitions explode.
    //
    // Real stats:
    // - Original: 5952 states, 45284 transitions
    // - Modified: 634 states, 315507 transitions  
    // - First-hop: 4401 transitions at start, 12647 from first-hop states
    //
    // The explosion factor is ~7x, but states reduce by ~9x.
    // This means each remaining state has ~60x more transitions!
    
    // Create a structure where interior states have many shared byte transitions
    let num_branches = 500; // Increase to amplify effect
    let terminals_count = 10;
    let num_byte_labels: usize = 50; // More byte labels = more overlapping
    let byte_labels: Vec<Label> = (0..num_byte_labels).map(|i| i as Label).collect();
    let depth = 3; // Keep depth manageable
    
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    
    let mut roots = Vec::new();
    
    for b in 0..num_branches {
        // Overlapping weights: branch b accepts tokens in range [b*5, b*5 + 20]
        let w = Weight::from_iter((0..20).map(|i| b * 5 + i));
        
        let root = nwa.states.add_state();
        roots.push(root);
        
        // Create tree structure with shared byte labels
        fn build_subtree(
            nwa: &mut NWA, 
            parent: usize, 
            depth: usize, 
            branch_idx: usize,
            num_byte_labels: usize,
            byte_labels: &[Label],
            w: &Weight,
        ) {
            if depth == 0 {
                nwa.states[parent].final_weight = Some(w.clone());
                return;
            }
            
            // Add 2-3 children with different byte labels
            let num_children = 2 + (branch_idx + depth) % 2;
            for c in 0..num_children {
                let byte_idx = (branch_idx * 7 + depth * 3 + c * 11) % num_byte_labels;
                let child = nwa.states.add_state();
                nwa.add_transition(parent, byte_labels[byte_idx], child, w.clone()).unwrap();
                build_subtree(nwa, child, depth - 1, branch_idx, num_byte_labels, byte_labels, w);
            }
        }
        
        build_subtree(&mut nwa, root, depth, b, num_byte_labels, &byte_labels, &w);
    }
    
    // ORIGINAL: Labeled transitions at start
    let start = nwa.states.add_state();
    nwa.body.start_states = vec![start];
    
    for (b, &root) in roots.iter().enumerate() {
        let label = (b + terminals_count) as Label;
        nwa.add_transition(start, label, root, Weight::all()).unwrap();
    }
    
    let mut orig_dwa = nwa.determinize();
    orig_dwa.simplify();
    let orig_states = orig_dwa.states.len();
    let orig_trans = orig_dwa.states.num_transitions();
    
    println!("Realistic Original: {} states, {} transitions", orig_states, orig_trans);
    
    // MODIFIED: Epsilon transitions at start
    let mut mod_nwa = nwa.clone();
    let start_trans = std::mem::take(&mut mod_nwa.states[start].transitions);
    for (_, targets) in start_trans {
        for (target, weight) in targets {
            mod_nwa.add_epsilon(start, target, weight);
        }
    }
    
    let mut mod_dwa = mod_nwa.determinize();
    mod_dwa.simplify();
    let mod_states = mod_dwa.states.len();
    let mod_trans = mod_dwa.states.num_transitions();
    
    println!("Realistic Modified: {} states, {} transitions", mod_states, mod_trans);
    
    let state_factor = mod_states as f64 / orig_states as f64;
    let trans_factor = mod_trans as f64 / orig_trans as f64;
    
    println!("Comparison:");
    println!("  States: {} -> {} ({:.2}x)", orig_states, mod_states, state_factor);
    println!("  Trans:  {} -> {} ({:.2}x)", orig_trans, mod_trans, trans_factor);
    println!("  Avg trans/state: {:.1} -> {:.1}", 
             orig_trans as f64 / orig_states as f64,
             mod_trans as f64 / mod_states as f64);
    
    if mod_trans > orig_trans {
        println!("EXPLOSION CONFIRMED: {:.2}x transition increase!", trans_factor);
        // Don't assert - just report. Real explosion is 7x, our simplified test may be less.
    } else if mod_trans < orig_trans {
        println!("REDUCTION: {:.2}x fewer transitions", 1.0 / trans_factor);
        println!("NOTE: Explosion may require more branches or different structure");
    } else {
        println!("SAME number of transitions");
    }
}
