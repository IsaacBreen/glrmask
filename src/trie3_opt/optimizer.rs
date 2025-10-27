use std::collections::BTreeMap;

use crate::constraint::{PrecomputeNode3Index, Trie3GodWrapper};
use crate::glr::parser::GLRParser;
use crate::tokenizer::TokenizerStateID;

use crate::trie3_opt::coordinator::{run_pipeline_on_precompute3, CoordinatorConfig};
use crate::trie3_opt::passes::full::config::Trie3Config;
use crate::constraint::StageVocab;

/// High-level size optimizer for precompute3 tries (legacy API).
/// This legacy entry point now delegates to the MiniTrie-based pipeline coordinator.
/// It preserves the original signature to maintain API compatibility, while mapping
/// the relevant config flags into the new CoordinatorConfig.
pub fn optimize_trie3_size(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    config: &Trie3Config,
    max_state_id: usize,
    max_llm_token_id: usize,
    _stage_vocab: &mut StageVocab,
    _parser: &GLRParser,
) {
    if !config.enabled {
        return;
    }

    // Map the legacy config to the MiniTrie pipeline configuration.
    let mut cfg = CoordinatorConfig::default();
    cfg.enable_prune_dead_paths = config.prune_dead_paths;
    cfg.enable_compress_edges = config.compress_edges;
    // Enable structural merge if requested by either structural or exact merge knobs.
    cfg.enable_merge_structural = config.merge_nodes_structural || config.merge_nodes_exact.enabled;
    // Use the legacy exact merge iters as a proxy for structural iters when available.
    cfg.merge_structural_max_iters = if config.merge_nodes_structural || config.merge_nodes_exact.enabled {
        config.merge_nodes_exact.exact_max_iters.max(1)
    } else {
        cfg.merge_structural_max_iters
    };
    cfg.enable_eliminate_pop0_except_roots = config.eliminate_pop0_edges;
    // Canonicalizing END nodes is always beneficial and semantics-preserving; enable unless explicitly disabled.
    cfg.enable_canonicalize_end_nodes = true;
    cfg.enable_factor_state_fanout = config.factor_state_fanout;

    // Delegate to the MiniTrie-based pipeline.
    run_pipeline_on_precompute3(
        roots,
        trie3_god,
        max_llm_token_id,
        max_state_id,
        cfg,
    );
}
