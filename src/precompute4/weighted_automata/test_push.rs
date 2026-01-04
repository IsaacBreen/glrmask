
#[cfg(test)]
mod tests {
    use crate::precompute4::weighted_automata::{DWA, DWABody, DWAStates, Weight};
    use crate::precompute4::weighted_automata::common::Label;
    use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

    /// Helper to build a simple DWA for testing.
    fn build_simple_dwa() -> DWA {
        // A --[{1,2}]--> B --[{2}]--> C (final, weight {1,2})
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        let w_12 = Weight::from_iter([1, 2]);
        let w_2 = Weight::from_item(2);

        // A -> B on label 0 with weight {1,2}
        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, w_12.clone());

        // B -> C on label 1 with weight {2}
        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, w_2.clone());

        // C is final with weight {1,2}
        states[c].final_weight = Some(w_12.clone());

        DWA {
            body: DWABody { start_state: a },
            states,
        }
    }

    #[test]
    fn test_backward_potentials() {
        let dwa = build_simple_dwa();
        // Since compute_backward_potentials is private to the module, we cannot call it directly 
        // from here unless it's pub(crate). But we can verify the effect of pushing recursively.
        // Actually, we can move this test to be an inner test of residuated_push if we want to test private methods.
        // Or we can rely on stochastic equivalence to prove correctness.
        // Let's rely on behavior: pushing shouldn't change the language.
        
        let mut dwa_pushed = dwa.clone();
        dwa_pushed.residuated_push();
        
        stochastic_equivalence_test(dwa, dwa_pushed);
    }

    #[test]
    fn test_residuated_push_preserves_language() {
        let mut dwa = build_simple_dwa();
        let original = dwa.clone();

        dwa.residuated_push();
        stochastic_equivalence_test(original, dwa);
    }

    #[test]
    fn test_push_removes_dead_edges() {
        // Build DWA with an edge that leads nowhere useful (empty intersection)
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_item(1));

        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, Weight::from_item(2));

        states[c].final_weight = Some(Weight::from_item(3)); // {1} & {2} & {3} = {}

        let mut dwa = DWA {
            body: DWABody { start_state: a },
            states,
        };
        
        dwa.residuated_push_prune_only();

        // All edges should be removed since path weight is empty
        assert!(dwa.states[0].transitions.is_empty());
        assert!(dwa.states[1].transitions.is_empty());
    }

    #[test]
    fn test_push_with_branching() {
        // Build DWA with branching
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();
        let d = states.add_state();
        let e = states.add_state();

        // A -> B (w={1,2}) -> D (final {2})
        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_iter([1, 2]));
        states[b].transitions.insert(2, d);
        states[b].trans_weights.insert(2, Weight::from_item(2));
        states[d].final_weight = Some(Weight::from_item(2));

        // A -> C (w={1,3}) -> E (final {3})
        states[a].transitions.insert(1, c);
        states[a].trans_weights.insert(1, Weight::from_iter([1, 3]));
        states[c].transitions.insert(3, e);
        states[c].trans_weights.insert(3, Weight::from_item(3));
        states[e].final_weight = Some(Weight::from_item(3));

        let mut dwa = DWA {
            body: DWABody { start_state: a },
            states,
        };
        
        let original = dwa.clone();
        dwa.residuated_push();
        
        stochastic_equivalence_test(original, dwa.clone());

        // Verify specific pushing behavior:
        // A->B should be tightened to {2}
        assert_eq!(dwa.states[0].trans_weights.get(&0), Some(&Weight::from_item(2)));
        // A->C should be tightened to {3}
        assert_eq!(dwa.states[0].trans_weights.get(&1), Some(&Weight::from_item(3)));
    }

    #[test]
    fn test_push_with_cycle() {
         // Cycle: A -> B -> B -> C
         let mut states = DWAStates::default();
         let a = states.add_state();
         let b = states.add_state();
         let c = states.add_state();
 
         // A -> B
         states[a].transitions.insert(0, b);
         states[a].trans_weights.insert(0, Weight::from_iter([1, 2]));
 
         // B -> B (self-loop)
         states[b].transitions.insert(1, b);
         states[b].trans_weights.insert(1, Weight::from_item(2));
 
         // B -> C
         states[b].transitions.insert(2, c);
         states[b].trans_weights.insert(2, Weight::from_item(2));
 
         states[c].final_weight = Some(Weight::from_item(2));
 
         let mut dwa = DWA {
             body: DWABody { start_state: a },
             states,
         };
         
         let original = dwa.clone();
         dwa.residuated_push();
         stochastic_equivalence_test(original, dwa.clone());

         // A->B should be tightened to {2}
         assert_eq!(dwa.states[0].trans_weights.get(&0), Some(&Weight::from_item(2)));
    }




}
