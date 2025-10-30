use std::collections::BTreeMap;
use std::time::Instant;

use crate::constraint::{PrecomputeNode3Index, StageVocab, Trie3GodWrapper};
use crate::glr::parser::GLRParser;
use crate::tokenizer::TokenizerStateID;
use crate::trie3_opt::coordinator::{run_pipeline_on_precompute3, CoordinatorConfig};
use crate::trie3_opt::Trie3Config;

/// High-level size optimizer for precompute3 tries. This is a legacy API that wraps the
/// new MiniTrie-based optimization pipeline.
pub fn optimize_trie3_size(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    config: &Trie3Config,
    max_state_id: usize,
    max_llm_token_id: usize,
    stage_vocab: &mut StageVocab,
    parser: &GLRParser,
) {
    crate::debug!(2, "Optimizing Trie 3 size (using new pipeline)...");
    let start = Instant::now();

    // Map legacy Trie3Config to new CoordinatorConfig
    let coordinator_config = CoordinatorConfig {
        num_passes: config.num_passes,
        prune_dead_paths: config.prune_dead_paths,
        prune_unproductive_paths: config.prune_unproductive_paths,
        canonicalize_end_nodes: config.canonicalize_end_nodes, // Always good practice
        compress_edges: config.compress_edges,
        compress_unary_chains: config.compress_edges,
        factor_common_destinations: config.factor_common_destinations,
        factor_common_destinations_min_incoming: config.factor_common_destinations_min_incoming,
        merge_structural: config.merge_structural,
        merge_structural_max_iters: config.merge_structural_max_iters,
        merge_bisimulation: config.merge_bisimulation,
        merge_bisimulation_max_iters: config.merge_bisimulation_max_iters,
        merge_global_atoms: config.merge_global_atoms,
        merge_global_atoms_max_iters: config.merge_global_atoms_max_iters,
        merge_global_atoms_max_atoms_per_pop: config.merge_global_atoms_max_atoms_per_pop,
        eliminate_pop0_except_roots: config.eliminate_pop0_except_roots,
        merge_equivalent_llm_tokens: config.merge_equivalent_llm_tokens,
        reorder_llm_tokens: config.reorder_llm_tokens,
        generalize_sids: config.generalize_sids,
        nwa_dwa_roundtrip: config.nwa_dwa_roundtrip,
        assert_no_pop0_except_roots: config.assert_no_pop0_except_roots,
    };

    run_pipeline_on_precompute3(
        roots,
        trie3_god,
        max_llm_token_id,
        max_state_id,
        coordinator_config,
        Some(stage_vocab),
        Some(parser),
    );

    crate::debug!(2, "Finished optimizing Trie 3 size in {:?}", start.elapsed());
}

