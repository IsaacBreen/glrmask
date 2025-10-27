use std::collections::BTreeMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, Trie3GodWrapper};
use crate::constraint_extra::{calculate_final_stats3, print_precompute_stats3, PrecomputeStats};
use crate::datastructures::trie::Trie;

/// Count total number of ranges across all live_tokens and edge masks for a set of nodes.
pub fn count_total_ranges_trie3(
    all_nodes: &[PrecomputeNode3Index],
    trie3_god: &Trie3GodWrapper,
) -> usize {
    let mut count = 0;
    for n in all_nodes {
        let g = n.read(trie3_god).expect("read");
        count += g.value.live_tokens.inner().ranges_len();
        for ((_pop, llm_bv), _dm) in g.children() {
            count += llm_bv.inner().ranges_len();
        }
    }
    count
}

/// Compute and print high-level stats for a trie3 set of roots.
pub fn compute_and_print_precompute_stats3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
) {
    let mut stats = PrecomputeStats::default();
    calculate_final_stats3(roots, &mut stats, trie3_god);
    print_precompute_stats3(&stats, trie3_god);

    // Also print new MiniTrie-based metrics
    let (mini, _, _) = crate::trie3_opt::coordinator::export_to_mini(
        roots,
        trie3_god,
        max_llm_token_id,
        max_state_id,
    );
    let metrics = crate::trie3_opt::metrics::run_all_metrics(&mini);
    println!("  MiniTrie Metrics: {}", crate::trie3_opt::metrics::pretty_print_metrics_map(&metrics));
}

/// Debug utility to remove all edges with pop>0, preserving pop<=0 edges.
pub fn debug_remove_pop_gt_0_edges_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "DEBUG: Removing all edges with pop > 0 from Trie3.");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    for node_idx in all_nodes {
        let mut w = node_idx.write(trie3_god).expect("write");

        let old_children = std::mem::take(w.children_mut());
        let mut new_children = BTreeMap::new();

        for ((pop, llm_bv), dest_map) in old_children {
            if pop <= 0 {
                new_children.insert((pop, llm_bv), dest_map);
            }
        }

        let mut new_live = LLMTokenBV::zeros();
        for ((_, llm_bv), _) in &new_children {
            new_live |= llm_bv;
        }
        w.value.live_tokens = new_live;
        *w.children_mut() = new_children;
    }
}
