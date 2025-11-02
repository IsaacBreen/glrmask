use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::weighted_automata::{DWA, NWA as WaNWA, NWAConfig as WaNWAConfig, NWAStates as WaNWAStates, Weight as WaWeight};
use std::collections::{BTreeMap, BTreeSet};
use crate::datastructures::trie::Trie;
use crate::precompute4::augmented_nwa::{AugmentedNwa, AugmentedNwaMeta};
use crate::glr::table::TerminalID;

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> Precomputed4 {
    // 1. Build augmented NWAs for all terminals (combined).
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

    // 3. Traverse the reversed trie using a single shared NWA states arena.
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root])
        .expect("Failed to compute traversal data for reversed trie1");

    let mut shared_states = WaNWAStates::new();

    // Initial combined NWA: a single final start state with empty stack.
    let mut initial_wa = WaNWA::new();
    let initial_state = initial_wa.cfg.start_state;
    initial_wa.set_final_weight(initial_state, WaWeight::all());
    let initial_aug_nwa = AugmentedNwa {
        states: crate::precompute4::augmented_nwa::AugmentedNwaStates { wa_states: initial_wa.states.clone() },
        meta: AugmentedNwaMeta {
            wa_cfg: initial_wa.cfg.clone(),
            nt_nodes: BTreeMap::new(),
            end_map: BTreeMap::from([(initial_state, BTreeSet::from([vec![]]))]),
        },
    };
    // Instantiate into shared arena and pass only meta through traversal.
    let initial_meta: AugmentedNwaMeta = initial_aug_nwa.instantiate_into(&mut shared_states);
    let initial_values = vec![(reversed_trie_root, initial_meta)];

    let mut final_metas: BTreeMap<TokenizerStateID, AugmentedNwaMeta> = BTreeMap::new();
    let original_trie1_roots_map: BTreeMap<_,_> = precomputed1.iter().map(|(k,v)|(v.clone(), *k)).collect();

    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_meta: &AugmentedNwaMeta, edge_terminal_opt, dest_map| {
            let mut results: Vec<(PrecomputeNode1Index, AugmentedNwaMeta)> = Vec::new();

            // Select the "edge" NWA (either a terminal's or ignore).
            let aug_nwa: &AugmentedNwa = if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                augmented_nwas.get(&terminal_id).expect_else(|| format!("No augmented NWA for terminal {:?}", terminal_id))
            } else {
                // Epsilon-like edge in grammar trie.
                &ignore_nwa
            };

            crate::debug!(5, "Processed edge {:?}, produced {} results.", edge_terminal_opt, results.len());
            // For each destination, instantiate the edge NWA into the shared arena,
            // combine right into it with current_meta, and produce a new meta.
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                let mut left_meta = aug_nwa.instantiate_into(&mut shared_states);
                let weight: WaWeight = WaWeight::from_rsb(llm_token_bv.inner.as_ref().clone());
                left_meta.combine_right_into(&mut shared_states, current_meta, &shared_states, &weight)
                    .expect("Combine failed");
                crate::debug!(5, "For dest_idx {:?} with token bv (WEIGHT) {:?}.", dest_idx, llm_token_bv);
                results.push((*dest_idx, left_meta));
            }
            results
        },
        // merge function (union metas in-place within the shared arena)
        |aug_nwa1: &mut AugmentedNwaMeta, aug_nwa2: AugmentedNwaMeta| {
            aug_nwa1.union_with_inplace(&mut shared_states, &aug_nwa2);
        },
        // process function
        |node_data, node_idx, aug_meta: &mut AugmentedNwaMeta| {
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_metas.insert(*tokenizer_state_id, aug_meta.clone());
            }
            true // continue traversal
        },
    );

    crate::debug!(5, "\n--- Final NWAs Before Determinization ---");
    for (sid, meta) in &final_metas {
        // Print only a summary of meta and the start state; full underlying NWA is large.
        crate::debug!(5, "Tokenizer State ID {:?}: start={:?}, end_map_keys={:?}", sid, meta.wa_cfg.start_state, meta.end_map.keys().collect::<Vec<_>>());
    }
    crate::debug!(5, "--- End Final NWAs Before Determinization ---\n");

    // 4. Convert final NWAs to DWAs and simplify.
    let mut precomputed4: Precomputed4 = BTreeMap::new();
    for (sid, aug_meta) in final_metas {
        let mut dwa = shared_states.determinize(&aug_meta.wa_cfg);
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
