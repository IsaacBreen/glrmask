use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::datastructures::trie::Trie;
use crate::glr::parser::GLRParser;
use crate::precompute4::weighted_automata::{resolve_negative_edges, DWA, Weight};
use crate::tokenizer::TokenizerStateID;
use std::collections::BTreeMap;
use std::time::Instant;

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

/// Build all DWAs directly (no NWA, no augmented structures).
/// Strategy:
/// - Reverse the precompute1 trie and traverse.
/// - For each edge:
///   - Build a trivial DWA (single state) and gate it by llm_token weight.
///   - Concatenate the "left" (edge DWA) with the "right" (accumulated DWA) using a trivial join-map (start->start).
/// - For merges across paths, union the resulting DWAs.
/// - After finishing a node, resolve negative symbols (todo) and simplify.
pub fn precompute4(
    _parser: &GLRParser,
    precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) -> Precomputed4 {
    println!("Starting precompute4 (DWA-only)...");
    // 1. Reverse the precompute1 trie.
    let now = Instant::now();
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &trie1_roots);

    let leaf_node = all_nodes
        .iter()
        .find_map(|&idx| idx.read(trie1_god).and_then(|g| if g.value.end { Some(idx) } else { None }))
        .expect("Precompute1 trie must have a single leaf node.");

    let reversed_trie1_god = Trie::reverse(trie1_god, &trie1_roots);
    println!("Trie::reverse took: {:?}", now.elapsed());
    let reversed_trie_root = leaf_node;

    // 2. Compute traversal data
    let now_traversal_data = Instant::now();
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root])
        .expect("Failed to compute traversal data for reversed trie1");
    println!("compute_traversal_data took: {:?}", now_traversal_data.elapsed());

    // Initial DWA: start state is final (accept empty).
    let mut initial_dwa = DWA::new();
    // Make start state accept empty with ALL final weight.
    initial_dwa
        .set_final_weight(initial_dwa.body.start_state, Weight::all())
        .expect("failed to set final weight");

    let initial_values = vec![(reversed_trie_root, initial_dwa)];

    let mut final_dwas: BTreeMap<TokenizerStateID, DWA> = BTreeMap::new();
    let original_trie1_roots_map: BTreeMap<_, _> = precomputed1.iter().map(|(k, v)| (v.clone(), *k)).collect();

    let now = Instant::now();
    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_dwa: &DWA, _edge_terminal_opt, dest_map| {
            let process_and_step_now = Instant::now();
            let mut results: Vec<(PrecomputeNode1Index, DWA)> = Vec::new();

            // Trivial "template" DWA for a single step (single state).
            let base_template = DWA::new();

            for (dest_idx, llm_token_bv) in dest_map.iter() {
                let mut left = base_template.clone();
                let weight: Weight = Weight::from_rsb(llm_token_bv.inner.as_ref().clone());
                left.apply_weight(&weight);

                // Concatenate left with right using a minimal join-map: start->start
                let mut join_map = BTreeMap::new();
                join_map.insert(left.body.start_state, std::iter::once(current_dwa.body.start_state).collect());

                let (combined, _) = left.concatenate(&current_dwa, &join_map);

                results.push((*dest_idx, combined));
            }
            println!("process_and_step closure took: {:?}", process_and_step_now.elapsed());
            results
        },
        // merge function: union
        |dwa1, dwa2| {
            let merge_now = Instant::now();
            let (merged, _) = dwa1.union(&dwa2);
            *dwa1 = merged;
            println!("merge closure (union) took: {:?}", merge_now.elapsed());
        },
        // process function per node
        |_node_data, node_idx, mut dwa| {
            println!(
                "In process fn for node {}, states_len: {}",
                node_idx.as_usize(),
                dwa.states.len()
            );
            // Resolve negative symbols (todo)
            resolve_negative_edges(&mut dwa);

            // Simplify
            dwa.simplify();

            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_dwas.insert(*tokenizer_state_id, dwa.clone());
            }
            Some(dwa) // continue traversal
        },
    );
    println!("special_map_grouped took: {:?}", now.elapsed());

    final_dwas
}
