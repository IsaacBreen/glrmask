use std::fmt;

#[derive(Debug, Clone)]
pub struct Trie3MergeConfig {
    pub enabled: bool,
    pub exact_max_iters: usize,
}

impl Default for Trie3MergeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exact_max_iters: 1000,
        }
    }
}

impl Trie3MergeConfig {
    pub fn off() -> Self {
        Self {
            enabled: false,
            exact_max_iters: 0,
        }
    }
}

// NOTE: New global-atoms merge pass configuration flags are added in Trie3Config below.
// They enable a bisimulation-style refinement over globally derived token atoms per pop.
#[derive(Debug, Clone)]
pub struct Trie3Config {
    pub eliminate_pop0_edges: bool,
    pub assert_no_pop0_nonroot_edges: bool,
    pub enabled: bool,
    pub num_passes: usize,
    pub merge_equivalent_llm_tokens: bool,
    pub reorder_llm_tokens: bool,
    pub constrain_bitvecs: bool,
    pub gc: bool,
    pub prune_dead_paths: bool,
    pub compress_edges: bool,
    pub merge_nodes_exact: Trie3MergeConfig,
    pub merge_nodes_structural: bool,
    pub merge_nodes_ultrafast: bool,
    pub prune_nodes_not_reaching_end: bool,
    pub generalize_sids: bool,
    pub simplify_llm_token_bvs: bool,
    pub refine_token_atoms: bool,
    pub refine_token_atoms_exact: bool,
    pub refine_token_atoms_max_blocks: usize,
    pub factor_common_destinations: bool,
    // NEW: root fanout factoring via token atoms (p>0 only)
    pub factor_root_fanout: bool,
    // NEW: global-atoms bisimulation merge
    pub merge_nodes_global_atoms: bool,
    pub merge_nodes_global_atoms_max_iters: usize,
    pub merge_nodes_global_atoms_max_atoms_per_pop: usize,
    pub factor_common_destinations_min_incoming: usize,
    pub stochastic_equivalence_check: bool,
    pub debug_remove_pop_gt_0: bool,
    pub factor_root_fanout_max_atoms_per_pop: usize,
    pub assert_pop0_paths_to_end_are_short: bool,
}

impl Default for Trie3Config {
    fn default() -> Self {
        Self {
            enabled: true,
            num_passes: 3,
            eliminate_pop0_edges: true,
            assert_no_pop0_nonroot_edges: true,
            merge_equivalent_llm_tokens: true,
            reorder_llm_tokens: true,
            constrain_bitvecs: true,
            gc: true,
            prune_dead_paths: true,
            compress_edges: true,
            merge_nodes_exact: Trie3MergeConfig::default(),
            merge_nodes_structural: true,
            merge_nodes_ultrafast: false,
            prune_nodes_not_reaching_end: true,
            generalize_sids: true,
            simplify_llm_token_bvs: false,
            refine_token_atoms: true,
            factor_root_fanout: true,
            factor_root_fanout_max_atoms_per_pop: 512,
            refine_token_atoms_exact: true,
            refine_token_atoms_max_blocks: 2048,
            // New: enable global-atoms bisimulation merge by default with conservative caps
            merge_nodes_global_atoms: true,
            merge_nodes_global_atoms_max_iters: 2,
            merge_nodes_global_atoms_max_atoms_per_pop: 4096,
            factor_common_destinations: true,
            factor_common_destinations_min_incoming: 12,
            stochastic_equivalence_check: false,
            debug_remove_pop_gt_0: false,
            assert_pop0_paths_to_end_are_short: false,
        }
    }
}

impl Trie3Config {
    pub fn off() -> Self {
        Self {
            enabled: false,
            num_passes: 0,
            eliminate_pop0_edges: false,
            assert_no_pop0_nonroot_edges: false,
            merge_equivalent_llm_tokens: false,
            reorder_llm_tokens: false,
            constrain_bitvecs: false,
            gc: false,
            prune_dead_paths: false,
            compress_edges: false,
            merge_nodes_exact: Trie3MergeConfig::off(),
            merge_nodes_structural: false,
            merge_nodes_ultrafast: false,
            prune_nodes_not_reaching_end: false,
            generalize_sids: false,
            refine_token_atoms: false,
            refine_token_atoms_exact: false,
            factor_root_fanout: false,
            factor_root_fanout_max_atoms_per_pop: 0,
            refine_token_atoms_max_blocks: 0,
            // New pass: off in the .off() preset
            merge_nodes_global_atoms: false,
            merge_nodes_global_atoms_max_iters: 0,
            merge_nodes_global_atoms_max_atoms_per_pop: 0,
            // keep these off for the .off() preset
            factor_common_destinations_min_incoming: 0,
            // eliminate_pop0_edges/assert_no_pop0_nonroot_edges already set above
            simplify_llm_token_bvs: false,
            factor_common_destinations: false,
            stochastic_equivalence_check: false,
            debug_remove_pop_gt_0: false,
            assert_pop0_paths_to_end_are_short: false,
        }
    }
}

impl fmt::Display for Trie3Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie3Config(enabled={}, passes={})", self.enabled, self.num_passes)
    }
}
