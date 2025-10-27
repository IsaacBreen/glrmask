use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::MiniTrie;

/// Trait implemented by each independent optimization pass over the MiniTrie.
pub trait OptimizationPass {
    fn name(&self) -> &'static str;
    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext);
}

pub mod prune_dead_paths;
pub mod compress_edges;
pub mod merge_structural;
pub mod eliminate_pop0;
pub mod canonicalize_end_nodes;
pub mod factor_state_fanout;
pub mod full;

pub use prune_dead_paths::PruneDeadPathsPass;
pub use compress_edges::CompressEdgesPass;
pub use merge_structural::MergeStructuralPass;
pub use eliminate_pop0::EliminatePop0ExceptRootsPass;
pub use canonicalize_end_nodes::CanonicalizeEndNodesPass;
pub use factor_state_fanout::FactorStateFanoutPass;
