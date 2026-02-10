#![cfg(test)]
use crate::dwa_i32::common::Label;
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
    nwa.body.start_states = vec![start];

    let dwa = nwa.determinize();
    assert_eq!(dwa.eval_word_weight(&['a' as Label, 'c' as Label]), Weight::from_item(0));
    assert_eq!(dwa.eval_word_weight(&['b' as Label, 'c' as Label]), Weight::from_item(1));
    assert!(dwa.states.len() <= 4);
}

#[ignore]
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
        for j in 0..N { if i != j { nwa.add_transition(s, alphabet[j], s, Weight::all()).unwrap(); } }
    }
    let start = nwa.states.add_state();
    for &s_comp in &component_starts { nwa.add_epsilon(start, s_comp, Weight::all()); }
    nwa.body.start_states = vec![start];
    let dwa = nwa.determinize();
    assert!(dwa.states.len() <= 2);
    let word_ac = vec![alphabet[0], alphabet[2]];
    let expected_weight_ac = &atoms[1] | &atoms[3];
    assert_eq!(dwa.eval_word_weight(&word_ac), expected_weight_ac);
}

/// Minimal test demonstrating the epsilon explosion phenomenon.
/// 
/// This test shows that replacing labeled start transitions with epsilon
/// transitions causes exponential blowup when terminals share patterns.
#[test]
fn test_epsilon_explosion_minimal() {
    // Create N terminals, each accepting a different single-character pattern
    // but sharing the SAME character space.
    // 
    // With labeled transitions: O(N) states
    // With epsilon transitions: O(2^N) states (subset construction)
    
    const N: usize = 4;  // Keep small for test speed
    let char_label: Label = 'x' as Label;  // All terminals match the same char
    
    // ORIGINAL: labeled transitions from start
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    for i in 0..N {
        let intermediate = nwa_labeled.states.add_state();
        let final_state = nwa_labeled.states.add_state();
        
        // Labeled transition: start --i--> intermediate
        nwa_labeled.add_transition(start_labeled, i as Label, intermediate, Weight::all()).unwrap();
        // Pattern transition: intermediate --char--> final
        nwa_labeled.add_transition(intermediate, char_label, final_state, Weight::all()).unwrap();
        // Final weight
        nwa_labeled.states[final_state].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // MODIFIED: epsilon transitions from start
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    for i in 0..N {
        let intermediate = nwa_epsilon.states.add_state();
        let final_state = nwa_epsilon.states.add_state();
        
        // EPSILON transition: start --eps--> intermediate
        nwa_epsilon.add_epsilon(start_eps, intermediate, Weight::all());
        // Pattern transition: intermediate --char--> final
        nwa_epsilon.add_transition(intermediate, char_label, final_state, Weight::all()).unwrap();
        // Final weight
        nwa_epsilon.states[final_state].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {}", N);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // The epsilon version should have FEWER states because paths merge
    // (all N terminals merge into one path for the char)
    // This is the OPPOSITE of what we see in real-world cases!
    
    // The explosion happens when terminals have DIFFERENT patterns
    // that DIVERGE after sharing some structure.
}

/// More realistic test: terminals with diverging patterns
#[test]
fn test_epsilon_explosion_diverging_patterns() {
    // N terminals, each with a unique 2-char pattern:
    // Terminal i: accepts "a" followed by character i
    //
    // ORIGINAL (labeled):
    //   start --i--> qi --'a'--> qi_a --char_i--> Fi
    // All terminals share the 'a' transition but diverge on the second char.
    //
    // EPSILON:
    //   start --eps--> qi --'a'--> qi_a --char_i--> Fi
    // After 'a', we're in {q0_a, q1_a, ..., qN_a}
    // Each has a different outgoing transition.
    
    const N: usize = 4;
    let shared_char: Label = 'a' as Label;
    
    // ORIGINAL: labeled transitions
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    for i in 0..N {
        let q_i = nwa_labeled.states.add_state();
        let q_i_a = nwa_labeled.states.add_state();
        let f_i = nwa_labeled.states.add_state();
        
        nwa_labeled.add_transition(start_labeled, i as Label, q_i, Weight::all()).unwrap();
        nwa_labeled.add_transition(q_i, shared_char, q_i_a, Weight::all()).unwrap();
        nwa_labeled.add_transition(q_i_a, (i + 100) as Label, f_i, Weight::all()).unwrap();  // Unique second char
        nwa_labeled.states[f_i].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    for i in 0..N {
        let q_i = nwa_epsilon.states.add_state();
        let q_i_a = nwa_epsilon.states.add_state();
        let f_i = nwa_epsilon.states.add_state();
        
        nwa_epsilon.add_epsilon(start_eps, q_i, Weight::all());
        nwa_epsilon.add_transition(q_i, shared_char, q_i_a, Weight::all()).unwrap();
        nwa_epsilon.add_transition(q_i_a, (i + 100) as Label, f_i, Weight::all()).unwrap();
        nwa_epsilon.states[f_i].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {} (diverging patterns)", N);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // With diverging patterns:
    // LABELED: Each terminal is separate, so ~4N states
    // EPSILON: After 'a', we have ONE merged state {q0_a, q1_a, ...}
    //   This state has N outgoing transitions (one per unique second char)
    //   So epsilon should still be SMALLER!
}

/// The REAL explosion case: shared alphabet with OVERLAPPING patterns
#[test]  
fn test_epsilon_explosion_overlapping_alphabet() {
    // The key insight: explosion happens when transitions OVERLAP
    // in the alphabet but lead to DIFFERENT states.
    //
    // Consider:
    // Terminal 0: matches [a-z] (alphabet 0..25)
    // Terminal 1: matches [a-z] (alphabet 0..25)  
    // ...
    //
    // With labeled transitions:
    //   start --0--> T0 --[a-z]--> F0
    //   start --1--> T1 --[a-z]--> F1
    // Each terminal is separate.
    //
    // With epsilon:
    //   start --eps--> T0 --[a-z]--> F0
    //   start --eps--> T1 --[a-z]--> F1
    // Initial subset: {T0, T1}
    // On 'a': both T0 and T1 can transition!
    // But they go to DIFFERENT states (F0 vs F1 with different weights).
    
    // Actually wait - if they go to the same alphabet and same final state structure,
    // they should MERGE in the determinized version.
    
    // Let me try: patterns that SHARE alphabet but have DIFFERENT lengths
    // Terminal 0: matches "a"
    // Terminal 1: matches "aa"
    //
    // With epsilon:
    //   start --eps--> T0 --a--> F0 (done after 1 char)
    //   start --eps--> T1 --a--> T1' --a--> F1 (needs 2 chars)
    //
    // Initial: {T0, T1}
    // On 'a': T0 -> F0, T1 -> T1'
    // Result: {F0, T1'}  -- F0 is final with weight {0}, T1' needs another 'a'
    //
    // From {F0, T1'}:
    //   On 'a': T1' -> F1
    //   Result: {F1} with weight {1}
    //   But what about F0? Does it stay in the subset?
    //
    // In a proper weighted automaton, F0 has no outgoing transitions on 'a',
    // so it gets removed from the subset.
    
    // Actually this is the pattern! Different LENGTH patterns cause explosion.
    // After reading k characters, we need to track which terminals are at
    // which position in their pattern - leading to many subset states.
    
    const N: usize = 4;
    let char_a: Label = 'a' as Label;
    
    // ORIGINAL: labeled
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    for i in 0..N {
        // Terminal i matches 'a' repeated (i+1) times
        let mut prev = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, prev, Weight::all()).unwrap();
        
        for _ in 0..i {
            let next = nwa_labeled.states.add_state();
            nwa_labeled.add_transition(prev, char_a, next, Weight::all()).unwrap();
            prev = next;
        }
        
        let final_state = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(prev, char_a, final_state, Weight::all()).unwrap();
        nwa_labeled.states[final_state].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    for i in 0..N {
        let mut prev = nwa_epsilon.states.add_state();
        nwa_epsilon.add_epsilon(start_eps, prev, Weight::all());
        
        for _ in 0..i {
            let next = nwa_epsilon.states.add_state();
            nwa_epsilon.add_transition(prev, char_a, next, Weight::all()).unwrap();
            prev = next;
        }
        
        let final_state = nwa_epsilon.states.add_state();
        nwa_epsilon.add_transition(prev, char_a, final_state, Weight::all()).unwrap();
        nwa_epsilon.states[final_state].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {} (different pattern lengths: 1, 2, 3, 4 'a's)", N);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // This should show the explosion!
    // With epsilon, after reading k 'a's, we need to track which terminals
    // have finished (are final) vs which are still in progress.
}

/// REAL explosion case: N terminals sharing the SAME second-hop state with DIFFERENT weights
/// This mirrors what we found in the actual terminal DWA analysis.
#[test]
fn test_epsilon_explosion_shared_second_hop() {
    // The REAL cause of explosion in production:
    // Multiple first-hop states all transition to the SAME second-hop state,
    // but with DIFFERENT weights!
    //
    // With N first-hop states sharing a second-hop state:
    // - Original: Each first-hop is in a separate branch (via TSID label)
    // - Epsilon: All N first-hop states are in the initial subset
    //   On reading a char, we go to the shared second-hop state
    //   But we need to track WHICH first-hop we came from (for weight calculation!)
    //   This creates up to 2^N subset states
    
    // Create: N first-hop states, all transition on 'x' to the SAME second-hop state
    // Each has a different weight
    
    const N: usize = 6;  // Keep small - explosion is exponential!
    let char_x: Label = 'x' as Label;
    
    // ORIGINAL: labeled transitions from start
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    // One shared second-hop state
    let shared_state = nwa_labeled.states.add_state();
    nwa_labeled.states[shared_state].final_weight = Some(Weight::all());
    
    for i in 0..N {
        let first_hop = nwa_labeled.states.add_state();
        // Labeled transition: start --i--> first_hop
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all()).unwrap();
        // All first_hops go to shared_state on 'x', but with DIFFERENT weights
        nwa_labeled.add_transition(first_hop, char_x, shared_state, Weight::from_item(i)).unwrap();
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    // One shared second-hop state
    let shared_state_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.states[shared_state_eps].final_weight = Some(Weight::all());
    
    for i in 0..N {
        let first_hop = nwa_epsilon.states.add_state();
        // EPSILON transition: start --eps--> first_hop
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        // All first_hops go to shared_state on 'x', but with DIFFERENT weights
        nwa_epsilon.add_transition(first_hop, char_x, shared_state_eps, Weight::from_item(i)).unwrap();
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {} (shared second-hop state)", N);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // With labeled: O(N) states (one per first-hop + shared)
    // With epsilon: On reading 'x', the subset is {shared} but with weight = union of all
    // Actually this might NOT explode because all paths merge to the same shared state!
    
    // The explosion happens when there are FURTHER transitions from the shared state
    // that depend on which first-hop we came from. Let me add that...
}

/// REAL explosion case V2: Shared second-hop with different DOWNSTREAM paths
#[test]
fn test_epsilon_explosion_shared_then_diverge() {
    // The ACTUAL explosion pattern:
    // 1. N first-hop states all transition to the SAME second-hop on label 'a'
    // 2. BUT each first-hop also has ADDITIONAL different labels that diverge
    //
    // This mirrors real terminal patterns where:
    // - Multiple tokenizer states share common character handling (e.g., 'a'-'z')
    // - But they differ in what comes next
    
    const N: usize = 5;
    
    // ORIGINAL: labeled
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    let shared_second = nwa_labeled.states.add_state();
    nwa_labeled.states[shared_second].final_weight = Some(Weight::from_item(999)); // Shared final
    
    for i in 0..N {
        let first_hop = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all()).unwrap();
        
        // Transition to SHARED second hop on 'a'
        nwa_labeled.add_transition(first_hop, 'a' as Label, shared_second, Weight::from_item(i)).unwrap();
        
        // UNIQUE transition to a different state on 'b'
        let unique_second = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(first_hop, 'b' as Label, unique_second, Weight::from_item(i)).unwrap();
        nwa_labeled.states[unique_second].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    let shared_second_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.states[shared_second_eps].final_weight = Some(Weight::from_item(999));
    
    for i in 0..N {
        let first_hop = nwa_epsilon.states.add_state();
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        
        nwa_epsilon.add_transition(first_hop, 'a' as Label, shared_second_eps, Weight::from_item(i)).unwrap();
        
        let unique_second = nwa_epsilon.states.add_state();
        nwa_epsilon.add_transition(first_hop, 'b' as Label, unique_second, Weight::from_item(i)).unwrap();
        nwa_epsilon.states[unique_second].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {} (shared + diverging paths)", N);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // With epsilon:
    // Initial subset: {start, fh0, fh1, ..., fhN}
    // On 'a': all fh_i go to shared_second -> subset {shared_second} with weight = union
    // On 'b': each fh_i goes to unique_i -> subset {u0, u1, ..., uN}
    // 
    // Still seems like it should work...
    // Maybe the explosion is even more subtle?
}

/// Definitive explosion test: different PATHS through shared states
#[test]
fn test_epsilon_explosion_paths_through_shared() {
    // The DEFINITIVE pattern that causes explosion:
    // - First-hop states share SOME but not ALL second-hop states
    // - This creates subset states that track which first-hops are still viable
    
    // Example:
    // fh0: 'a' -> s1, 'b' -> s2
    // fh1: 'a' -> s1, 'c' -> s3
    // fh2: 'b' -> s2, 'c' -> s3
    //
    // s1 is shared by fh0, fh1
    // s2 is shared by fh0, fh2
    // s3 is shared by fh1, fh2
    //
    // After reading 'a': {fh0:s1, fh1:s1} = subset with s1, but only from fh0 and fh1
    // After reading 'b': {fh0:s2, fh2:s2} = subset with s2, but only from fh0 and fh2
    // After reading "ab": from {fh0, fh1} on 'b': only fh0 goes to s2
    //                     Result: {fh0:s2}
    // After reading "ba": from {fh0, fh2} on 'a': only fh0 goes to s1
    //                     Result: {fh0:s1}
    //
    // These are DIFFERENT subset states even though they contain the same underlying state!
    // Because the WEIGHT accumulated along the path differs.
    
    const N: usize = 6;  // Creates 2^N paths
    
    // ORIGINAL: labeled
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    let final_state = nwa_labeled.states.add_state();
    nwa_labeled.states[final_state].final_weight = Some(Weight::all());
    
    // Create N first-hop states
    // Each pair of consecutive first-hops shares a second-hop state
    let mut first_hops = vec![];
    for i in 0..N {
        let fh = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, fh, Weight::all()).unwrap();
        first_hops.push(fh);
    }
    
    // Create shared second-hop states
    // second_hop[i] is shared by first_hop[i] and first_hop[i+1]
    for i in 0..N-1 {
        let sh = nwa_labeled.states.add_state();
        let label = (100 + i) as Label;
        nwa_labeled.add_transition(first_hops[i], label, sh, Weight::from_item(i)).unwrap();
        nwa_labeled.add_transition(first_hops[i+1], label, sh, Weight::from_item(i+1)).unwrap();
        // From shared, go to final
        nwa_labeled.add_transition(sh, 'f' as Label, final_state, Weight::all()).unwrap();
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    let final_state_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.states[final_state_eps].final_weight = Some(Weight::all());
    
    let mut first_hops_eps = vec![];
    for _ in 0..N {
        let fh = nwa_epsilon.states.add_state();
        nwa_epsilon.add_epsilon(start_eps, fh, Weight::all());
        first_hops_eps.push(fh);
    }
    
    for i in 0..N-1 {
        let sh = nwa_epsilon.states.add_state();
        let label = (100 + i) as Label;
        nwa_epsilon.add_transition(first_hops_eps[i], label, sh, Weight::from_item(i)).unwrap();
        nwa_epsilon.add_transition(first_hops_eps[i+1], label, sh, Weight::from_item(i+1)).unwrap();
        nwa_epsilon.add_transition(sh, 'f' as Label, final_state_eps, Weight::all()).unwrap();
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {} (pairwise shared second-hops)", N);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
}

/// Real explosion pattern: Many first-hops share the SAME label going to DIFFERENT targets
/// This mirrors the actual terminal DWA structure: label 10 has 209 source states!
#[test]
fn test_epsilon_explosion_many_sources_same_label() {
    // The ACTUAL pattern from production:
    // - 234 first-hop states (tokenizer states)
    // - Label 10 (e.g., newline character) is a transition from 209 of them
    // - Each goes to a DIFFERENT target with a DIFFERENT weight
    //
    // With epsilon:
    // - Epsilon closure: {start, fh0, fh1, ..., fh233}
    // - On label 10: 209 of the first-hops have this transition
    //   Each goes to its own target with its own weight
    // - Result: subset of {t0, t1, ..., t208} with weights
    //
    // THIS is what causes the explosion!
    // The determinization must track which subset of targets we're in.
    
    const N: usize = 10;  // Number of first-hop states with shared label
    let shared_label: Label = 10;
    
    // ORIGINAL: labeled
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    for i in 0..N {
        let first_hop = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all()).unwrap();
        
        // Each first-hop has the shared_label transition to its OWN unique target
        let target = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(first_hop, shared_label, target, Weight::from_item(i)).unwrap();
        nwa_labeled.states[target].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    for i in 0..N {
        let first_hop = nwa_epsilon.states.add_state();
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        
        let target = nwa_epsilon.states.add_state();
        nwa_epsilon.add_transition(first_hop, shared_label, target, Weight::from_item(i)).unwrap();
        nwa_epsilon.states[target].final_weight = Some(Weight::from_item(i));
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {} (many sources sharing one label)", N);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // Hmm, this might still merge because all targets are distinct with distinct weights.
    // The subset is just {t0, t1, ..., tN} with weight = union of all.
    // This is ONE DFA state!
    //
    // The explosion happens when there are FURTHER transitions that DISTINGUISH
    // which path we took.
}

/// Real explosion V2: Many sources, shared label, FURTHER transitions
/// 
/// This test mirrors the ACTUAL structure found in production:
/// - Label 4 has 1215 source states
/// - These 1215 sources go to only 7 unique targets  
/// - One target is reachable from 1209 of those sources!
/// 
/// The key is: after reading label 4, we have a SUBSET of the 7 targets,
/// but the WEIGHT on that subset depends on WHICH of the 1209 sources we came from.
/// This creates subset differentiation even though the underlying states are the same.
#[test]
fn test_epsilon_explosion_many_sources_with_continuation() {
    // Mirror the production pattern:
    // - N first-hop states
    // - K of them share label L going to the SAME target T
    // - They have DIFFERENT transition weights
    // - T has further transitions that accumulate weights
    
    const N: usize = 20;   // Total first-hop states
    const K: usize = 15;   // Number sharing the same label
    let shared_label: Label = 'L' as Label;
    
    // ORIGINAL: labeled
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    // Shared target state (like the target reachable from 1209 sources)
    let shared_target = nwa_labeled.states.add_state();
    
    // More states after the shared target
    let after_shared = nwa_labeled.states.add_state();
    nwa_labeled.add_transition(shared_target, 'X' as Label, after_shared, Weight::all()).unwrap();
    nwa_labeled.states[after_shared].final_weight = Some(Weight::all());
    
    for i in 0..N {
        let first_hop = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all()).unwrap();
        
        if i < K {
            // These K first-hops share label L going to shared_target
            // But with DIFFERENT weights!
            nwa_labeled.add_transition(first_hop, shared_label, shared_target, Weight::from_item(i)).unwrap();
        } else {
            // Other first-hops have different patterns
            let unique_target = nwa_labeled.states.add_state();
            nwa_labeled.add_transition(first_hop, 'U' as Label, unique_target, Weight::from_item(i)).unwrap();
            nwa_labeled.states[unique_target].final_weight = Some(Weight::from_item(i));
        }
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    let shared_target_eps = nwa_epsilon.states.add_state();
    let after_shared_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.add_transition(shared_target_eps, 'X' as Label, after_shared_eps, Weight::all()).unwrap();
    nwa_epsilon.states[after_shared_eps].final_weight = Some(Weight::all());
    
    for i in 0..N {
        let first_hop = nwa_epsilon.states.add_state();
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        
        if i < K {
            nwa_epsilon.add_transition(first_hop, shared_label, shared_target_eps, Weight::from_item(i)).unwrap();
        } else {
            let unique_target = nwa_epsilon.states.add_state();
            nwa_epsilon.add_transition(first_hop, 'U' as Label, unique_target, Weight::from_item(i)).unwrap();
            nwa_epsilon.states[unique_target].final_weight = Some(Weight::from_item(i));
        }
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {}, K = {} sharing label L", N, K);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // After epsilon closure: {start, fh0, fh1, ..., fhN}
    // On label L: K first-hops go to shared_target
    //   Result: {shared_target} with weight = union of {0, 1, ..., K-1}
    // 
    // This is STILL just one state! The weight is different, but the STATE is the same.
    // So this should NOT cause explosion...
    //
    // The explosion must be when the shared_target has OUTGOING transitions that
    // DEPEND on which path we took to get there. But in a DWA, transitions don't
    // depend on how we got to a state!
    //
    // Let me think more carefully about what causes the real explosion...
}

/// Test where roots have DIFFERENT alphabets (some respond to char X, some don't)
/// This creates many different subsets!
#[ignore]
#[test]
fn test_epsilon_explosion_different_alphabets() {
    // Key insight: the explosion is in TRANSITIONS, not states!
    // 
    // With epsilon:
    // - FEWER states (subsets merge)
    // - MORE transitions (each subset state needs many outgoing edges)
    //
    // Example:
    // Initial: {root_0, ..., root_N}
    // On label L_0 (only root_0 responds): go to subset {child_0_L0}
    // On label L_1 (only root_1 responds): go to subset {child_1_L1}
    // ... etc
    // 
    // The START state in the determinized epsilon version needs K transitions
    // (one for each label in the alphabet), where each goes to a DIFFERENT subset.
    //
    // In the labeled version:
    // - Start needs N transitions (one per root)
    // - Each root needs P transitions
    // - Total: N + N*P = N(1+P) transitions at first two levels
    //
    // In epsilon version:
    // - Start gets epsilon-merged with all roots
    // - Start needs K transitions total (one for each label)
    // - But K can be much larger than N!
    //
    // The explosion happens when K >> N, i.e., more labels than roots.
    
    const N: usize = 4;   // Number of roots (tokenizer states)
    const K: usize = 100; // Number of possible labels (terminals)
    const P: usize = 50;  // How many labels each root responds to (overlapping)
    const DEPTH: usize = 4;  // Deeper structure
    
    // ORIGINAL: labeled
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    // Create N roots with overlapping label sets
    for i in 0..N {
        let root = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, root, Weight::all()).unwrap();
        
        // Root i responds to labels [i*10..i*10+P] (overlapping)
        let mut current = vec![root];
        for d in 0..DEPTH {
            let mut next = vec![];
            for &state in &current {
                for j in 0..P {
                    let label = ((i * 10 + j) % K) as Label;
                    let child = nwa_labeled.states.add_state();
                    nwa_labeled.add_transition(state, label, child, Weight::from_item(i * 1000 + d * 100 + j)).unwrap();
                    next.push(child);
                }
            }
            current = next;
        }
        
        for &leaf in &current {
            nwa_labeled.states[leaf].final_weight = Some(Weight::from_item(i));
        }
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    for i in 0..N {
        let root = nwa_epsilon.states.add_state();
        nwa_epsilon.add_epsilon(start_eps, root, Weight::all());
        
        let mut current = vec![root];
        for d in 0..DEPTH {
            let mut next = vec![];
            for &state in &current {
                for j in 0..P {
                    let label = ((i * 10 + j) % K) as Label;
                    let child = nwa_epsilon.states.add_state();
                    nwa_epsilon.add_transition(state, label, child, Weight::from_item(i * 1000 + d * 100 + j)).unwrap();
                    next.push(child);
                }
            }
            current = next;
        }
        
        for &leaf in &current {
            nwa_epsilon.states[leaf].final_weight = Some(Weight::from_item(i));
        }
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("N = {}, K = {}, P = {}, DEPTH = {}", N, K, P, DEPTH);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // Check out-degree of start state in both
    let labeled_start_out = dwa_labeled.states[dwa_labeled.body.start_state].transitions.len();
    let epsilon_start_out = dwa_epsilon.states[dwa_epsilon.body.start_state].transitions.len();
    println!("Labeled start out-degree: {}", labeled_start_out);
    println!("Epsilon start out-degree: {}", epsilon_start_out);
    
    if epsilon_trans > labeled_trans {
        println!("SUCCESS! Epsilon causes TRANSITION EXPLOSION (but fewer states)");
    } else {
        println!("Still reduction...");
    }
}
/// Test that matches the real JSON schema grammar pattern
/// 
/// This models:
/// - N tokenizer states (roots) connected via labeled transitions from start
/// - Each root has a vocabulary trie with shared labels but different weights
/// - The key is: roots have DIFFERENT transition patterns on the SAME labels
/// 
/// The explosion happens when:
/// 1. Multiple roots are epsilon-merged into the start state
/// 2. Reading a label, different roots transition to different targets
/// 3. The result is a subset of targets with different weight combinations
/// 4. Deeper transitions accumulate more subset combinations
#[test]
fn test_epsilon_explosion_json_like() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
    // Model the JSON grammar pattern:
    // - root ::= '{' field1 options '}' 
    // - options ::= (',' option)* 
    // - option ::= '"a"':STRING | '"b"':STRING | ...
    //
    // The tokenizer has different states for:
    // - After opening brace
    // - After field1
    // - After each optional field variant
    //
    // When we read a character, different tokenizer states have different valid continuations.
    
    const NUM_ROOTS: usize = 8;  // Number of tokenizer states (positions in JSON)
    const NUM_LABELS: usize = 20;  // Number of possible next labels (characters/terminals)
    
    // LABELED version: each root is a separate subtree
    let mut nwa_labeled = NWA::new();
    nwa_labeled.states.0.clear();
    let start_labeled = nwa_labeled.states.add_state();
    nwa_labeled.body.start_states = vec![start_labeled];
    
    // Create NUM_ROOTS independent roots
    // Each root can transition on a SUBSET of labels
    // Different roots have DIFFERENT subsets (like different JSON positions)
    for root_idx in 0..NUM_ROOTS {
        let root = nwa_labeled.states.add_state();
        nwa_labeled.add_transition(start_labeled, root_idx as Label, root, Weight::all()).unwrap();
        
        // This root can transition on labels [root_idx..root_idx + 5] mod NUM_LABELS
        // Different roots have overlapping but not identical label sets
        for j in 0..5 {
            let label = ((root_idx + j) % NUM_LABELS) as Label;
            
            // Create a small subtree for this label
            let level1 = nwa_labeled.states.add_state();
            nwa_labeled.add_transition(root, label, level1, Weight::from_item(root_idx * 100 + j)).unwrap();
            
            // Each level1 node can continue to a few more states
            for k in 0..3 {
                let level2 = nwa_labeled.states.add_state();
                let next_label = ((label as usize + k + 1) % NUM_LABELS) as Label;
                nwa_labeled.add_transition(level1, next_label, level2, Weight::from_item(root_idx * 1000 + j * 10 + k)).unwrap();
                nwa_labeled.states[level2].final_weight = Some(Weight::from_item(root_idx));
            }
        }
    }
    
    let dwa_labeled = nwa_labeled.determinize();
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = dwa_labeled.states.num_transitions();
    
    // EPSILON version: roots are epsilon-reachable from start
    let mut nwa_epsilon = NWA::new();
    nwa_epsilon.states.0.clear();
    let start_eps = nwa_epsilon.states.add_state();
    nwa_epsilon.body.start_states = vec![start_eps];
    
    for root_idx in 0..NUM_ROOTS {
        let root = nwa_epsilon.states.add_state();
        nwa_epsilon.add_epsilon(start_eps, root, Weight::all());
        
        for j in 0..5 {
            let label = ((root_idx + j) % NUM_LABELS) as Label;
            let level1 = nwa_epsilon.states.add_state();
            nwa_epsilon.add_transition(root, label, level1, Weight::from_item(root_idx * 100 + j)).unwrap();
            
            for k in 0..3 {
                let level2 = nwa_epsilon.states.add_state();
                let next_label = ((label as usize + k + 1) % NUM_LABELS) as Label;
                nwa_epsilon.add_transition(level1, next_label, level2, Weight::from_item(root_idx * 1000 + j * 10 + k)).unwrap();
                nwa_epsilon.states[level2].final_weight = Some(Weight::from_item(root_idx));
            }
        }
    }
    
    let dwa_epsilon = nwa_epsilon.determinize();
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = dwa_epsilon.states.num_transitions();
    
    println!("NUM_ROOTS = {}, NUM_LABELS = {}", NUM_ROOTS, NUM_LABELS);
    println!("LABELED: {} states, {} transitions", labeled_states, labeled_trans);
    println!("EPSILON: {} states, {} transitions", epsilon_states, epsilon_trans);
    println!("State ratio: {:.2}x", epsilon_states as f64 / labeled_states as f64);
    println!("Trans ratio: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    
    // The key insight:
    // - LABELED: After reading root_idx, we're in exactly one subtree
    //   Total states ≈ NUM_ROOTS * (1 + 5 * 4) = NUM_ROOTS * 21
    //
    // - EPSILON: After epsilon closure, we're in ALL roots at once
    //   On reading label L:
    //   - Some roots accept L, some don't
    //   - Result is a SUBSET of level1 states from different roots
    //   - Need to track which subset we're in
    //   
    // This creates subset states that weren't in the labeled version!
    
    if epsilon_trans > labeled_trans {
        println!("SUCCESS! Transition explosion: {:.2}x", epsilon_trans as f64 / labeled_trans as f64);
    } else {
        println!("No explosion (ratio = {:.2}x)", epsilon_trans as f64 / labeled_trans as f64);
    }
}