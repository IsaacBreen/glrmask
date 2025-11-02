use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::weighted_automata::{DWA, NWA as WaNWA, Weight as WaWeight};
use std::collections::{BTreeMap, BTreeSet};
use crate::datastructures::trie::Trie;
use crate::precompute4::augmented_nwa::AugmentedNwa;
use crate::glr::table::TerminalID;

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> Precomputed4 {
    // 1. Build augmented NWAs for all terminals.
    let augmented_nwas = match crate::precompute4::augmented_nwa::build_augmented_nwas(parser) {
        Ok(nwas) => nwas,
        Err(e) => panic!("Failed to build augmented NWAs: {:?}", e),
    };

    let ignore_nwa = crate::precompute4::augmented_nwa::build_augmented_nwa_for_ignore_terminal();

    println!("\n--- Augmented NWA Generation ---");
    for (tid, aug_nwa) in &augmented_nwas {
        println!("Terminal ID {:?}:\n{}", tid, aug_nwa);
    }
    println!("--- End Augmented NWA Generation ---\n");

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
    println!("\n--- Reversed Trie1 ---\n{}", Trie::pretty_print_with_options(&reversed_trie1_god, &[reversed_trie_root], &options));

    // 3. Traverse the reversed trie.
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root])
        .expect("Failed to compute traversal data for reversed trie1");

    let mut initial_nwa = WaNWA::new();
    let initial_state = initial_nwa.start_state;
    initial_nwa.set_final_weight(initial_state, WaWeight::all());
    let initial_aug_nwa = AugmentedNwa {
        nwa: initial_nwa,
        end_state: initial_state,
        nt_nodes: BTreeMap::new(),
        end_map: BTreeMap::from([(initial_state, BTreeSet::from([vec![]]))]),
    };

    let initial_values = vec![(reversed_trie_root, initial_aug_nwa)];

    let mut final_nwas: BTreeMap<TokenizerStateID, AugmentedNwa> = BTreeMap::new();
    let original_trie1_roots_map: BTreeMap<_,_> = precomputed1.iter().map(|(k,v)|(v.clone(), *k)).collect();

    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_aug_nwa, edge_terminal_opt, dest_map| {
            let mut results: Vec<(PrecomputeNode1Index, AugmentedNwa)> = Vec::new();

            let aug_nwa;
            if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                aug_nwa = augmented_nwas.get(&terminal_id).expect_else(|| format!("No augmented NWA for terminal {:?}", terminal_id));
            } else {
                // Epsilon-like edge in grammar trie. Just propagate the current NWA.
                aug_nwa = &ignore_nwa;
            }
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                let mut new_aug_nwa = aug_nwa.clone();
                let weight: WaWeight = WaWeight::from_rsb(llm_token_bv.inner.as_ref().clone());
                new_aug_nwa.combine_right_into(current_aug_nwa, &weight)
                    .expect("Combine failed");
                results.push((*dest_idx, new_aug_nwa));
            }
            println!("Processed edge {:?}, produced {} results.", edge_terminal_opt, results.len());
            println!("--- Incoming aug_nwa ---\n{}", current_aug_nwa);
            println!("--- Edge aug_nwa ---\n{}", aug_nwa);
            for (_, res_aug_nwa) in &results {
                println!("--- Resulting aug_nwa ---\n{}", res_aug_nwa);
            }
            results
        },
        // merge function
        |aug_nwa1, aug_nwa2| {
            aug_nwa1.union_with(&aug_nwa2);
        },
        // process function
        |node_data, node_idx, aug_nwa| {
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_nwas.insert(*tokenizer_state_id, aug_nwa.clone());
            }
            true // continue traversal
        },
    );

    println!("\n--- Final NWAs Before Determinization ---");
    for (sid, aug_nwa) in &final_nwas {
        println!("Tokenizer State ID {:?}:\n{}", sid, aug_nwa);
    }
    println!("--- End Final NWAs Before Determinization ---\n");

    // 4. Convert final NWAs to DWAs and simplify.
    let mut precomputed4: Precomputed4 = BTreeMap::new();
    for (sid, aug_nwa) in final_nwas {
        let mut dwa = aug_nwa.nwa.determinize();
        dwa.simplify();
        precomputed4.insert(sid, dwa);
    }

    println!("\n--- Final DWAs After Determinization and Simplification ---");
    for (sid, dwa) in &precomputed4 {
        println!("Tokenizer State ID {:?}:\n{}", sid, dwa);
    }
    println!("--- End Final DWAs After Determinization and Simplification ---\n");

    precomputed4
}
