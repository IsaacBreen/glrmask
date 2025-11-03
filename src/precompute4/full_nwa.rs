use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::weighted_automata::{DWA, NWA as WaNWA, NWAStates as WaNWAStates, NWABody as WaNWABody, Weight as WaWeight};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use crate::datastructures::trie::Trie;
use crate::precompute4::augmented_nwa::{AugmentedNwa, AugmentedNwaBody};
use crate::glr::table::TerminalID;
use std::time::Instant;

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> Precomputed4 {
    // 1. Build augmented NWAs for all terminals.
    println!("Starting precompute4...");
    let now = Instant::now();
    let augmented_nwas = match crate::precompute4::augmented_nwa::build_augmented_nwas(parser) {
        Ok(nwas) => nwas,
        Err(e) => panic!("Failed to build augmented NWAs: {:?}", e),
    };
    println!("build_augmented_nwas took: {:?}", now.elapsed());

    let ignore_nwa = crate::precompute4::augmented_nwa::build_augmented_nwa_for_ignore_terminal();

    crate::debug!(5, "\n--- Augmented NWA Generation ---");
    for (tid, aug_nwa) in &augmented_nwas {
        crate::debug!(5, "Terminal ID {:?}:\n{}", tid, aug_nwa);
    }
    crate::debug!(5, "--- End Augmented NWA Generation ---\n");

    // 2. Reverse the precompute1 trie.
    let now = Instant::now();
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &trie1_roots);

    let leaf_node = all_nodes.iter().find_map(|&idx| {
        idx.read(trie1_god).and_then(|g| if g.value.end { Some(idx) } else { None })
    }).expect("Precompute1 trie must have a single leaf node.");

    let reversed_trie1_god = Trie::reverse(trie1_god, &trie1_roots);
    println!("Trie::reverse took: {:?}", now.elapsed());
    let reversed_trie_root = leaf_node;
    let options = crate::datastructures::trie::PrettyPrintOptions::default()
        .omit_depth()
        ;
    crate::debug!(5, "\n--- Reversed Trie1 ---\n{}", Trie::pretty_print_with_options(&reversed_trie1_god, &[reversed_trie_root], &options));

    // 3. Traverse the reversed trie.
    let now_traversal_data = Instant::now();
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root])
            .expect("Failed to compute traversal data for reversed trie1");
    println!("compute_traversal_data took: {:?}", now_traversal_data.elapsed());


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

    let now = Instant::now();
    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_aug_body, edge_terminal_opt, dest_map| {
            let step_now = Instant::now();
            let mut results: Vec<(PrecomputeNode1Index, AugmentedNwaBody)> = Vec::new();

            // Prepare the LEFT body by mapping the terminal's NWA into the shared states.
            let template_aug: &AugmentedNwa = if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                augmented_nwas.get(&terminal_id).expect_else(|| format!("No augmented NWA for terminal {:?}", terminal_id))
            } else {
                &ignore_nwa
            };

            for (dest_idx, llm_token_bv) in dest_map.iter() {
                // Map the template_aug's states into the shared arena.
                let copy_now = Instant::now();
                let mapping = shared_states.borrow_mut().append_copy_from(&template_aug.states);
                let copy_elapsed = copy_now.elapsed();

                let mut left_body = template_aug.body.clone();
                left_body.remap_states(&mapping);

                let weight: WaWeight = WaWeight::from_rsb(llm_token_bv.inner.as_ref().clone());
                // Combine into a new body (mutating the shared graph with epsilon links).
                let combine_now = Instant::now();
                let mut new_body = left_body.clone();
                AugmentedNwaBody::combine_right_into_on_shared(
                    &mut shared_states.borrow_mut(),
                    &mut new_body,
                    &current_aug_body,
                    &weight,
                ).expect("Combine failed");
                let combine_elapsed = combine_now.elapsed();

                println!(
                    "step inner loop: term {:?}, dest {}, shared_states_len: {}",
                    edge_terminal_opt,
                    dest_idx.as_usize(),
                    shared_states.borrow().len()
                );
                println!("  append_copy_from: {:?}", copy_elapsed);
                println!("  combine_right_into_on_shared (caller): {:?}", combine_elapsed);

                results.push((*dest_idx, new_body));
            }
            println!("step closure took: {:?}", step_now.elapsed());
            results
        },
        // merge function
        |aug_body1, aug_body2| {
            let merge_now = Instant::now();
            AugmentedNwaBody::union_with_on_shared(&mut shared_states.borrow_mut(), aug_body1, &aug_body2);
            println!("merge closure (union_with_on_shared) took: {:?}", merge_now.elapsed());
        },
        // process function
        |node_data, node_idx, aug_body| {
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_nwas.insert(*tokenizer_state_id, aug_body.clone());
            }
            true // continue traversal
        },
    );
    println!("special_map_grouped took: {:?}", now.elapsed());

    crate::debug!(5, "\n--- Final NWA Bodies Before Determinization ---");
    for (sid, aug_body) in &final_nwas {
        crate::debug!(5, "Tokenizer State ID {:?}: starts={:?}, end_map_keys={:?}", sid, aug_body.nwa.start_states, aug_body.end_map.keys().collect::<Vec<_>>());
    }
    crate::debug!(5, "--- End Final NWA Bodies Before Determinization ---\n");

    // 4. Convert final NWA bodies to DWAs and simplify.
    let now = Instant::now();
    let mut precomputed4: Precomputed4 = BTreeMap::new();
    for (sid, aug_body) in final_nwas {
        let det_now = Instant::now();
        let mut dwa = WaNWA::determinize_components(&shared_states.borrow(), &aug_body.nwa);
        let det_elapsed = det_now.elapsed();

        let simplify_now = Instant::now();
        dwa.simplify();
        let simplify_elapsed = simplify_now.elapsed();
        println!("For TokenizerStateID {:?}: determinize took {:?}, simplify took {:?}", sid, det_elapsed, simplify_elapsed);
        precomputed4.insert(sid, dwa);
    }
    println!("Final determinization and simplification loop took: {:?}", now.elapsed());

    crate::debug!(5, "\n--- Final DWAs After Determinization and Simplification ---");
    for (sid, dwa) in &precomputed4 {
        crate::debug!(5, "Tokenizer State ID {:?}:\n{}", sid, dwa);
    }
    crate::debug!(5, "--- End Final DWAs After Determinization and Simplification ---\n");

    precomputed4
}
