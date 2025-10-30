use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::MiniTrie;

/// Trait every optimization pass must implement.
pub trait OptimizationPass {
    fn name(&self) -> &'static str;
    fn run(&self, trie: &mut crate::trie3_opt::core::MiniTrie, ctx: &mut crate::trie3_opt::context::OptimizationContext);
}

pub mod canonicalize_end_nodes;
pub mod compress_edges;
pub mod compress_unary_chains;
pub mod eliminate_pop0;
pub mod factor_common_destinations;
pub mod generalize_sids;
pub mod merge_bisimulation;
pub mod merge_equivalent_llm_tokens;
pub mod merge_global_atoms;
pub mod merge_structural;
pub mod prune_dead_paths;
pub mod prune_unproductive_paths;
pub mod reorder_llm_tokens;
pub mod nwa_dwa_roundtrip;

pub use canonicalize_end_nodes::CanonicalizeEndNodesPass;
pub use compress_edges::CompressEdgesPass;
pub use compress_unary_chains::CompressUnaryChainsPass;
pub use eliminate_pop0::EliminatePop0ExceptRootsPass;
pub use factor_common_destinations::FactorCommonDestinationsPass;
pub use generalize_sids::GeneralizeSidsPass;
pub use merge_bisimulation::MergeBisimulationPass;
pub use merge_equivalent_llm_tokens::MergeEquivalentLLMTokensPass;
pub use merge_structural::MergeStructuralPass;
pub use prune_dead_paths::PruneDeadPathsPass;
pub use prune_unproductive_paths::PruneUnproductivePathsPass;
pub use reorder_llm_tokens::ReorderLLMTokensPass;
pub use nwa_dwa_roundtrip::NwaDwaRoundtripPass;