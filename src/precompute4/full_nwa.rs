use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::weighted_automata::{DWA, NWA as WaNWA, NWAStates as WaNWAStates, NWARest as WaNWARest, Weight as WaWeight};
use std::collections::{BTreeMap, BTreeSet};
use std::cell::RefCell;
use crate::datastructures::trie::Trie;
use crate::precompute4::augmented_nwa::{AugmentedNwa, AugmentedNwaRest};
use crate::glr::table::TerminalID;

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

    // Shared global states store: all automata for this traversal live here.
    let shared_states = RefCell::new(WaNWAStates::default());

    // Initial NWA rest: one final start state with empty stack
    let initial_state = shared_states.borrow_mut().add_state();
    shared_states.borrow_mut().set_final_weight(initial_state, WaWeight::all());
    let initial_rest = AugmentedNwaRest {
        nwa: WaNWARest { start_state: initial_state },
        nt_nodes: BTreeMap::new(),
        end_map: BTreeMap::from([(initial_state, BTreeSet::from([vec![]]))]),
    };

    let initial_values = vec![(reversed_trie_root, initial_rest)];

    let mut final_nwas: BTreeMap<TokenizerStateID, AugmentedNwaRest> = BTreeMap::new();
    let original_trie1_roots_map: BTreeMap<_,_> = precomputed1.iter().map(|(k,v)|(v.clone(), *k)).collect();

    Trie::special_map_grouped(
        &reversed_trie1_god,
        &Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root])
            .expect("Failed to compute traversal data for reversed trie1"),
        initial_values,
        // step function: data carried per node is AugmentedNwaRest (meta only).
        |current_rest: &AugmentedNwaRest, edge_terminal_opt: &Option<TerminalID>, dest_map| {
            let mut results: Vec<(PrecomputeNode1Index, AugmentedNwaRest)> = Vec::new();

            let aug_nwa;
            if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                aug_nwa = augmented_nwas.get(&terminal_id).expect_else(|| format!("No augmented NWA for terminal {:?}", terminal_id));
            } else {
                // Epsilon-like edge in grammar trie. Just propagate the current NWA via ignore.
                aug_nwa = &ignore_nwa;
            }
            crate::debug!(5, "Processed edge {:?}, produced {} results.", edge_terminal_opt, results.len());
            let dbg_left = AugmentedNwa { states: shared_states.borrow().clone(), rest: current_rest.clone() };
            crate::debug!(5, "--- RIGHT: Incoming aug_nwa ---\n{}", dbg_left);
            crate::debug!(5, "--- LEFT: Edge aug_nwa ---\n{}", aug_nwa);
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                // Clone the left rest and rebase onto shared states.
                let mut new_rest = aug_nwa.rest.clone();
                AugmentedNwa::rebase_onto_shared(&mut shared_states.borrow_mut(), &aug_nwa.states, &mut new_rest);

                let weight: WaWeight = WaWeight::from_rsb(llm_token_bv.inner.as_ref().clone());

                // Combine using shared states (no copying of current_rest)
                AugmentedNwa::combine_right_into_on_shared_states(
                    &mut shared_states.borrow_mut(),
                    &mut new_rest,
                    current_rest,
                    &weight,
                ).expect("Combine failed");
                let dbg_combined = AugmentedNwa { states: shared_states.borrow().clone(), rest: new_rest.clone() };
                crate::debug!(5, "For dest_idx {:?} with token bv (WEIGHT) {:?}:", dest_idx, llm_token_bv);
                crate::debug!(5, "--- COMBINED: Resulting aug_nwa ---\n{}", dbg_combined);
                results.push((*dest_idx, new_rest));
            }
            results
        },
        // merge function: union two rests that reside in the same shared states
        |aug_nwa1: &mut AugmentedNwaRest, aug_nwa2: AugmentedNwaRest| {
            // Merge end_maps
            for (st, stacks) in &aug_nwa2.end_map {
                aug_nwa1.end_map.entry(*st).or_default().extend(stacks.clone());
            }
            // Create new start in shared states and epsilon to both old starts.
            let new_start = shared_states.borrow_mut().add_state();
            shared_states.borrow_mut().add_epsilon_transition(new_start, aug_nwa1.nwa.start_state, WaWeight::all());
            shared_states.borrow_mut().add_epsilon_transition(new_start, aug_nwa2.nwa.start_state, WaWeight::all());
            aug_nwa1.nwa.start_state = new_start;
        },
        // process function
        |node_data, node_idx, aug_rest| {
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_nwas.insert(*tokenizer_state_id, aug_rest.clone());
            }
            true // continue traversal
        },
    );

    crate::debug!(5, "\n--- Final NWA Rests Before Determinization ---");
    for (sid, aug_rest) in &final_nwas {
        let dbg = AugmentedNwa { states: shared_states.borrow().clone(), rest: aug_rest.clone() };
        crate::debug!(5, "Tokenizer State ID {:?}:\n{}", sid, dbg);
    }
    crate::debug!(5, "--- End Final NWA Rests Before Determinization ---\n");

    // 4. Convert final NWAs to DWAs and simplify.
    let mut precomputed4: Precomputed4 = BTreeMap::new();
    for (sid, aug_rest) in final_nwas {
        let mut dwa = WaNWA::determinize_components(&shared_states.borrow(), &aug_rest.nwa);
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
