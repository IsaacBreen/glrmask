use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::weighted_automata::{
    DWA, NWA as WaNWA, NWAMeta as WaNWAMeta, NWAStates as WaNWAStates, Weight as WaWeight,
};
use std::collections::{BTreeMap, BTreeSet};
use crate::datastructures::trie::Trie;
use crate::precompute4::augmented_nwa::{remap_augmented_meta, AugmentedNwa, AugmentedNwaMeta};
use crate::glr::table::TerminalID;

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> Precomputed4 {
    // 1. Build augmented NWAs for all terminals.
    let augmented_nwas = match crate::precompute4::augmented_nwa::build_augmented_nwas(parser) {
        Ok(nwas) => nwas,
        Err(e) => panic!("Failed to build augmented NWAs: {:?}", e),
    };

    let ignore_nwa = crate::precompute4::augmented_nwa::build_augmented_nwa_for_ignore_terminal();

    let mut shared_states = WaNWAStates::new();

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

    let start = shared_states.add_state();
    let initial_meta = AugmentedNwaMeta {
        nwa_meta: WaNWAMeta { start_state: start },
        nt_nodes: BTreeMap::new(),
        end_map: BTreeMap::from([(start, BTreeSet::from([vec![]]))]),
    };
    shared_states.set_final_weight(start, WaWeight::all());

    let initial_values = vec![(reversed_trie_root, initial_meta)];

    let mut final_metas: BTreeMap<TokenizerStateID, AugmentedNwaMeta> = BTreeMap::new();
    let original_trie1_roots_map: BTreeMap<_,_> = precomputed1.iter().map(|(k,v)|(v.clone(), *k)).collect();

    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_meta, edge_terminal_opt, dest_map| {
            let mut results: Vec<(PrecomputeNode1Index, AugmentedNwaMeta)> = Vec::new();

            let edge_aug;
            if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                edge_aug = augmented_nwas.get(&terminal_id).expect_else(|| format!("No augmented NWA for terminal {:?}", terminal_id));
            } else {
                // Epsilon-like edge in grammar trie. Just propagate the current NWA.
                edge_aug = &ignore_nwa;
            }
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                // 1) Copy the terminal's states into shared and remap its meta
                let mapping = shared_states.append_copy_from(&edge_aug.states.nwa_states);
                let mut left_meta =
                    remap_augmented_meta(&edge_aug.meta, &mapping);

                // 2) Combine into the current meta using shared states
                let weight: WaWeight = WaWeight::from_rsb(llm_token_bv.inner.as_ref().clone());
                left_meta.combine_right_into_shared(&mut shared_states, current_meta, &weight)
                    .expect("Combine failed");

                results.push((*dest_idx, left_meta));
            }
            results
        },
        // merge function
        |dst_meta, src_meta| {
            dst_meta.union_meta_in_place(&mut shared_states, &src_meta);
        },
        // process function
        |node_data, node_idx, meta| {
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_metas.insert(*tokenizer_state_id, meta.clone());
            }
            true // continue traversal
        },
    );

    crate::debug!(5, "\n--- Final NWAs Before Determinization ---");
    for (sid, meta) in &final_metas {
        let tmp_nwa = WaNWA {
            states: shared_states.clone(),
            meta: meta.nwa_meta.clone(),
        };
        crate::debug!(5, "Tokenizer State ID {:?}:\n{}", sid, tmp_nwa);
    }
    crate::debug!(5, "--- End Final NWAs Before Determinization ---\n");

    // 4. Convert final NWAs to DWAs and simplify.
    let mut precomputed4: Precomputed4 = BTreeMap::new();
    for (sid, meta) in final_metas {
        let mut dwa = shared_states.determinize(&meta.nwa_meta);
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
