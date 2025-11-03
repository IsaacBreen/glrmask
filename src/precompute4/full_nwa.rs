use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::weighted_automata::{DWA, NWA as WaNWA, NWAStates as WaNWAStates, NWABody as WaNWABody, Weight as WaWeight};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use crate::datastructures::trie::Trie;
use crate::precompute4::augmented_nwa::{AugmentedNwa, AugmentedNwaBody};

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> Precomputed4 {
    // 1. Build augmented NWAs for all terminals.
    let augmented_nwas = match crate::precompute4::augmented_nwa::build_augmented_nwas(parser) {
        Ok(nwas) => nwas,
        Err(e) => panic!("Failed to build augmented NWAs: {:?}", e),
    };

    let ignore_nwa = crate::precompute4::augmented_nwa::build_augmented_nwa_for_ignore_terminal();

    crate::debug!(5, "\n--- Augmented NWA Generation ---");
    for (tid, aug_nwa) in &augmented_nwas {
        crate::debug!(5, "Terminal ID {:?}:\n{}", tid, aug_nwa);
    }
    crate::debug!(5, "--- End Augmented NWA Generation ---\n");

    // 2. Reverse the precompute1 trie.
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &trie1_roots);

    let leaf_node = all_nodes.iter().find_map(|&idx| {
        idx.read(trie1_god).and_then(|g| if g.value.end { Some(idx) } else { None })
    }).expect("Precompute1 trie must have a single leaf node.");

    let reversed_trie1_god = Trie::reverse(trie1_god, &trie1_roots);
    let reversed_trie_root = leaf_node;
    let options = crate::datastructures::trie::PrettyPrintOptions::default()
        .omit_depth()
        ;
    crate::debug!(5, "\n--- Reversed Trie1 ---\n{}", Trie::pretty_print_with_options(&reversed_trie1_god, &[reversed_trie_root], &options));

    // 3. Traverse the reversed trie.
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root])
            .expect("Failed to compute traversal data for reversed trie1");


    // Shared NWA states arena for the entire traversal. This lets us share subgraphs between paths.
    let shared_states = RefCell::new(WaNWAStates::default());
    let initial_state = shared_states.borrow_mut().add_state();
    shared_states.borrow_mut().set_final_weight(initial_state, WaWeight::all());

    // The initial body: single start that is final, with end_map containing empty stack.
    let initial_aug_body = AugmentedNwaBody {
        nwa: WaNWABody { start_states: BTreeSet::from([initial_state]) },
        nt_nodes: BTreeMap::new(),
        end_map: BTreeMap::from([]),
    };

    let initial_values = vec![(reversed_trie_root, initial_aug_body)];

    let mut final_nwas: BTreeMap<TokenizerStateID, AugmentedNwaBody> = BTreeMap::new();
    let original_trie1_roots_map: BTreeMap<_,_> = precomputed1.iter().map(|(k,v)|(v.clone(), *k)).collect();

    Trie::special_map_unified(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // process_and_step function
        |node_data, node_idx, current_aug_body| {
            // Process part: check if this node corresponds to a final NWA
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_nwas.insert(*tokenizer_state_id, current_aug_body.clone());
            }

            // Step part: propagate to children
            let mut results: Vec<(PrecomputeNode1Index, AugmentedNwaBody)> = Vec::new();

            for (edge_terminal_opt, dest_map) in node_data.children() {
                // Prepare the LEFT body by mapping the terminal's NWA into the shared states.
                let template_aug: &AugmentedNwa = if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                    let terminal_id = edge_terminal_opt.unwrap();
                    augmented_nwas.get(&terminal_id).expect_else(|| format!("No augmented NWA for terminal {:?}", terminal_id))
                } else {
                    &ignore_nwa
                };

                for (dest_idx, llm_token_bv) in dest_map.iter() {
                    // Map the template_aug's states into the shared arena.
                    let mapping = shared_states.borrow_mut().append_copy_from(&template_aug.states);

                    let mut left_body = template_aug.body.clone();
                    left_body.remap_states(&mapping);

                    let weight: WaWeight = WaWeight::from_rsb(llm_token_bv.inner.as_ref().clone());
                    // Combine into a new body (mutating the shared graph with epsilon links).
                    let mut new_body = left_body.clone();
                    AugmentedNwaBody::combine_right_into_on_shared(
                        &mut shared_states.borrow_mut(),
                        &mut new_body,
                        &current_aug_body,
                        &weight,
                    ).expect("Combine failed");

                    results.push((*dest_idx, new_body));
                }
            }
            results
        },
        // merge function
        |aug_body1, aug_body2| {
            AugmentedNwaBody::union_with_on_shared(&mut shared_states.borrow_mut(), aug_body1, &aug_body2);
        },
    );

    crate::debug!(5, "\n--- Final NWA Bodies Before Determinization ---");
    for (sid, aug_body) in &final_nwas {
        crate::debug!(5, "Tokenizer State ID {:?}: starts={:?}, end_map_keys={:?}", sid, aug_body.nwa.start_states, aug_body.end_map.keys().collect::<Vec<_>>());
    }
    crate::debug!(5, "--- End Final NWA Bodies Before Determinization ---\n");

    // 4. Convert final NWA bodies to DWAs and simplify.
    let mut precomputed4: Precomputed4 = BTreeMap::new();
    for (sid, aug_body) in final_nwas {
        let mut dwa = WaNWA::determinize_components(&shared_states.borrow(), &aug_body.nwa);

        dwa.simplify();
        precomputed4.insert(sid, dwa);
    }

    crate::debug!(5, "\n--- Final DWAs After Determinization and Simplification ---");
    for (sid, dwa) in &precomputed4 {
        crate::debug!(5, "Tokenizer State ID {:?}:\n{}", sid, dwa);
    }
    crate::debug!(5, "--- End Final DWAs After Determinization and Simplification ---\n");

    precomputed4
}
