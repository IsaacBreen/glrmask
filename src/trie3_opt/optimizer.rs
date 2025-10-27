use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::constraint::StageVocab;
use crate::constraint::{PrecomputeNode3Index, LLMTokenBV, StateIDBV, Trie3GodWrapper, PrecomputedNodeContents};
use crate::constraint_extra::{PrecomputeStats, calculate_final_stats3, print_precompute_stats3};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{PathComparison, Trie};
use crate::glr::parser::GLRParser;
use crate::tokenizer::TokenizerStateID;
use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::coordinator::{export_to_mini, import_from_mini};
use crate::trie3_opt::passes::full::config::Trie3Config;
use crate::trie3_opt::passes::full::{
    atoms::{refine_edges_to_token_atoms_trie3, refine_edges_to_token_atoms_trie3_exact},
    canonicalize::canonicalize_end_nodes_trie3,
    compress::compress_trie3_edges,
    constrain::constrain_bitvecs_trie3,
    cycles::{
        assert_pop0_paths_to_end_are_short, has_true_cycle_trie3, has_true_cycle_trie3_llm_only,
    },
    factoring::{factor_common_destinations_trie3, factor_state_fanout_trie3},
    factor_root::factor_root_fanout_via_atoms,
    global_atoms_merge::merge_nodes_trie3_global_atoms,
    merges::{merge_nodes_trie3, merge_nodes_trie3_structural, merge_nodes_trie3_ultrafast},
    normalize::normalize_live_tokens_trie3,
    pop0::{assert_no_pop0_nonroot_edges_trie3, eliminate_pop0_edges_except_roots_trie3},
    reorder::reorder_llm_tokens_for_range_minimization_trie3,
    simplify::simplify_llm_token_bvs_trie3,
    generalize::propagate_and_generalize_sids_trie3,
    stats::{compute_and_print_precompute_stats3, debug_remove_pop_gt_0_edges_trie3},
    tokens::merge_equivalent_llm_tokens_trie3,
};
use crate::trie3_opt::passes::full::config::Trie3MergeConfig;
use crate::trie3_opt::passes::{
    CompressChainsPass, OptimizationPass, PruneDeadPathsPass, PruneUnproductivePathsPass,
};
use rand::prelude::*;

/// Helper to run a single MiniTrie pass on the full Trie3 graph.
fn run_mini_pass<P: OptimizationPass>(
    pass: P,
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    ctx: &mut OptimizationContext,
) {
    let (mut mini, root_pairs, _old_mapping) =
        export_to_mini(roots, trie3_god, ctx.max_llm_token_id, ctx.max_state_id);

    pass.run(&mut mini, ctx);

    import_from_mini(&mini, &root_pairs, trie3_god, roots, ctx);
}

/// High-level size optimizer for precompute3 tries, migrated from passes::constraint_precompute3_utils.
/// This function retains the exact signature and behavior, orchestrating the pipeline of heavy passes.
pub fn optimize_trie3_size(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    config: &Trie3Config,
    max_state_id: usize,
    mut max_llm_token_id: usize,
    stage_vocab: &mut StageVocab,
    parser: &GLRParser,
) {
    has_true_cycle_trie3(trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    has_true_cycle_trie3_llm_only(
        trie3_god,
        &roots.values().cloned().collect::<Vec<_>>(),
        stage_vocab.internal_max_llm_token,
    );

    if !config.enabled {
        return;
    }

    let (original_arena, original_roots) = if config.stochastic_equivalence_check {
        crate::debug!(2, "Stochastic equivalence check enabled. Cloning initial trie.");
        let initial_roots_map_sorted: Vec<_> = roots.iter().collect();
        let initial_roots_vec: Vec<_> =
            initial_roots_map_sorted.iter().map(|(_, &idx)| idx).collect();
        let (cloned_arena, cloned_roots_vec, _) =
            Trie::deep_copy_subtrees(trie3_god, &initial_roots_vec);
        let cloned_roots_map: BTreeMap<_, _> = initial_roots_map_sorted
            .into_iter()
            .map(|(k, _)| *k)
            .zip(cloned_roots_vec)
            .collect();
        (Some(cloned_arena), Some(cloned_roots_map))
    } else {
        (None, None)
    };

    crate::debug!(2, "Optimizing Trie 3 size...");

    let mut ctx = OptimizationContext::new(max_llm_token_id, max_state_id);

    // First normalize any stale derived fields so they don't affect downstream reasoning.
    crate::debug!(2, "Initial stats:");
    compute_and_print_precompute_stats3(roots, trie3_god, max_llm_token_id, max_state_id);

    let mut step_counter = 1;
    macro_rules! run_pass {
        ($name:expr, $code:block) => {
            crate::debug!(2, "Running optimization pass {}: {}...", step_counter, $name);
            let start = Instant::now();
            $code
            let duration = start.elapsed();
            crate::debug!(
                2,
                "Pass {} ('{}') finished in {:?}",
                step_counter,
                $name,
                duration
            );
            crate::debug!(2, "Stats after pass {}:", step_counter);
            compute_and_print_precompute_stats3(roots, trie3_god, max_llm_token_id, max_state_id);
            step_counter += 1;
        };
    }
    // Normalize live_tokens right at the beginning of optimization passes to ensure it remains derived.
    run_pass!("Normalizing derived live_tokens", {
        normalize_live_tokens_trie3(roots, trie3_god);
    });

    for pass_num in 0..config.num_passes {
        if config.num_passes > 1 {
            crate::debug!(
                2,
                "--- Starting optimization super-pass {}/{} ---",
                pass_num + 1,
                config.num_passes
            );
        }

        if config.debug_remove_pop_gt_0 {
            run_pass!("DEBUG: Removing pop > 0 edges", {
                debug_remove_pop_gt_0_edges_trie3(roots, trie3_god);
            });
        }

        // --- Phase 1: Initial Pruning & Vocab Reduction ---
        // These passes are expensive but have a huge impact on the initial massive graph.
        // They are essential to run first to make subsequent passes feasible.
        if config.merge_nodes_ultrafast {
            run_pass!("Merging nodes (fast pre-pass)", {
                merge_nodes_trie3_ultrafast(roots, trie3_god);
                merge_nodes_trie3(roots, trie3_god, 40);
            });
        }

        if config.prune_dead_paths {
            run_pass!("Pruning dead paths", {
                run_mini_pass(PruneDeadPathsPass, roots, trie3_god, &mut ctx);
            });
        }

        if config.prune_nodes_not_reaching_end {
            run_pass!("Pruning nodes that do not reach end", {
                run_mini_pass(PruneUnproductivePathsPass, roots, trie3_god, &mut ctx);
            });
        }

        // New: Collapse all END nodes early to maximize downstream sharing.
        run_pass!("Canonicalizing END nodes (pre-merge)", {
            canonicalize_end_nodes_trie3(roots, trie3_god);
        });

        if config.merge_equivalent_llm_tokens {
            run_pass!("Merging equivalent LLM tokens", {
                merge_equivalent_llm_tokens_trie3(roots, trie3_god, stage_vocab);
            });
        }

        if config.reorder_llm_tokens {
            run_pass!("Reordering LLM tokens for range minimization", {
                reorder_llm_tokens_for_range_minimization_trie3(roots, trie3_god, stage_vocab);
                max_llm_token_id = stage_vocab.internal_max_llm_token;
            });
        }

        if config.constrain_bitvecs {
            let roots_vec: Vec<_> = roots.values().cloned().collect();
            let _all_nodes_pinner = Trie::all_nodes(&trie3_god, &roots_vec);
            run_pass!("Constraining bitvectors", {
                constrain_bitvecs_trie3(trie3_god, &roots_vec, max_state_id, max_llm_token_id);
            });
        }

        if config.generalize_sids {
            run_pass!("Generalizing StateID bitvectors", {
                propagate_and_generalize_sids_trie3(
                    roots, trie3_god, parser, max_state_id,
                );
            });
        }

        // Root fanout factoring (p>0 only): dramatically reduces root out-degree
        // by routing tokens through atom-specific intermediates with aggregated dest maps.
        if config.factor_root_fanout && config.factor_root_fanout_max_atoms_per_pop > 0 {
            run_pass!("Root fanout factoring (token atoms, p>0 only)", {
                factor_root_fanout_via_atoms(
                    roots,
                    trie3_god,
                    max_llm_token_id,
                    max_state_id,
                    config.factor_root_fanout_max_atoms_per_pop,
                );
            });
        }

        if config.simplify_llm_token_bvs {
            run_pass!("Simplifying LLM token bitsets", {
                simplify_llm_token_bvs_trie3(roots, &trie3_god, max_llm_token_id);
            });
        }

        if config.factor_state_fanout {
            run_pass!("Factoring state fanout", {
                factor_state_fanout_trie3(roots, trie3_god);
            });
        }

        if config.factor_common_destinations {
            run_pass!("Factoring common destinations", {
                factor_common_destinations_trie3(
                    roots,
                    trie3_god,
                    max_llm_token_id,
                    max_state_id,
                    config.factor_common_destinations_min_incoming,
                );
            });
        }

        if config.compress_edges {
            run_pass!("Compressing edges + unary chains", {
                compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
                run_mini_pass(CompressChainsPass, roots, trie3_god, &mut ctx);
            });
        }

        // Refine token-sets to semantic atoms per node/pop to maximally coalesce behaviorally identical tokens.
        if config.refine_token_atoms {
            run_pass!("Refining token-set atoms", {
                refine_edges_to_token_atoms_trie3(
                    roots,
                    &trie3_god,
                    max_llm_token_id,
                    max_state_id,
                    config.refine_token_atoms_max_blocks,
                );
            });
        }

        // NEW: Global-atoms bisimulation merge. This aligns token semantics across nodes globally (per pop)
        // and merges nodes whose behavior is identical for each global token-atom and pop, even when their
        // local edge partitions differ.
        if config.merge_nodes_global_atoms && config.merge_nodes_global_atoms_max_iters > 0 {
            run_pass!("Merging nodes (global-atoms bisimulation)", {
                merge_nodes_trie3_global_atoms(
                    roots,
                    &trie3_god,
                    max_llm_token_id,
                    config.merge_nodes_global_atoms_max_iters,
                    config.merge_nodes_global_atoms_max_atoms_per_pop,
                );
            });
        }

        // After compression, prune and GC before the expensive merge.
        if config.prune_dead_paths {
            run_pass!("Pruning dead paths (post-compress)", {
                run_mini_pass(PruneDeadPathsPass, roots, trie3_god, &mut ctx);
            });
        }
        if config.gc {
            run_pass!("Garbage collection (pre-merge)", {
                Trie::gc(
                    &trie3_god,
                    &roots.values().cloned().collect::<Vec<_>>(),
                );
            });
        }

        if config.merge_nodes_structural {
            run_pass!("Merging nodes (structural)", {
                merge_nodes_trie3_structural(
                    roots,
                    &trie3_god,
                    config.merge_nodes_exact.exact_max_iters,
                );
            });
            // Structural merge can create parallel edges (same dest/pop/sids, diff tokens).
            // Compress them immediately.
            if config.compress_edges {
                run_pass!("Compressing edges + unary chains (post-structural)", {
                    compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
                    run_mini_pass(CompressChainsPass, roots, trie3_god, &mut ctx);
                });
            }
        }

        if config.merge_nodes_exact.enabled {
            run_pass!("Merging nodes", {
                merge_nodes_trie3(
                    roots,
                    &trie3_god,
                    config.merge_nodes_exact.exact_max_iters,
                );
            });
        }

        // --- Phase 3: Iterative Refinement ---
        // A few rounds of compression and merging on the now much smaller graph.

        if config.prune_nodes_not_reaching_end {
            run_pass!("Pruning nodes that do not reach end (post-merge)", {
                run_mini_pass(PruneUnproductivePathsPass, roots, trie3_god, &mut ctx);
            });
        }

        if config.compress_edges {
            run_pass!("Compressing edges + unary chains (post-merge)", {
                compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
                run_mini_pass(CompressChainsPass, roots, trie3_god, &mut ctx);
            });
        }
        if config.refine_token_atoms {
            run_pass!("Refining token-set atoms (post-merge)", {
                refine_edges_to_token_atoms_trie3(
                    roots,
                    &trie3_god,
                    max_llm_token_id,
                    max_state_id,
                    config.refine_token_atoms_max_blocks,
                );
            });
        }

        if config.merge_nodes_exact.enabled {
            run_pass!("Merging nodes (post-compress)", {
                merge_nodes_trie3(
                    roots,
                    &trie3_god,
                    config.merge_nodes_exact.exact_max_iters,
                );
            });
        }

        // --- Phase 4: Final Cleanup and Polish ---

        if config.prune_dead_paths {
            run_pass!("Pruning dead paths (final)", {
                run_mini_pass(PruneDeadPathsPass, roots, trie3_god, &mut ctx);
            });
        }

        if config.gc {
            run_pass!("Garbage collection (final)", {
                Trie::gc(
                    &trie3_god,
                    &roots.values().cloned().collect::<Vec<_>>(),
                );
            });
        }

        if config.merge_equivalent_llm_tokens {
            run_pass!("Merging equivalent LLM tokens (final pass)", {
                merge_equivalent_llm_tokens_trie3(roots, trie3_god, stage_vocab);
            });
        }
        if config.reorder_llm_tokens {
            run_pass!("Reordering LLM tokens (final pass)", {
                reorder_llm_tokens_for_range_minimization_trie3(
                    roots,
                    trie3_god,
                    stage_vocab,
                );
            });
        }

        // New pass: eliminate pop=0 edges originating from non-root nodes.
        if config.eliminate_pop0_edges {
            run_pass!("Eliminating pop=0 edges (non-roots)", {
                eliminate_pop0_edges_except_roots_trie3(roots, trie3_god);
                if config.assert_no_pop0_nonroot_edges {
                    assert_no_pop0_nonroot_edges_trie3(roots, trie3_god);
                }
            });
            // Optional compaction right after: merge identical dest maps and shrink live tokens.
            if config.compress_edges {
                run_pass!("Compressing edges + unary chains (post-pop0-elim)", {
                    // Use current maxes as we have the latest vocab and state bounds
                    compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id); run_mini_pass(CompressChainsPass, roots, trie3_god, &mut ctx);
                });
            }
            if config.refine_token_atoms {
                run_pass!("Refining token-set atoms (post-structural)", {
                    refine_edges_to_token_atoms_trie3(
                        roots,
                        &trie3_god,
                        max_llm_token_id,
                        max_state_id,
                        config.refine_token_atoms_max_blocks,
                    );
                });
            }
            if config.refine_token_atoms_exact {
                run_pass!("Refining token-set atoms (exact, post-pop0-elim)", {
                    refine_edges_to_token_atoms_trie3_exact(
                        roots,
                        &trie3_god,
                        max_llm_token_id,
                        max_state_id,
                    );
                });
            }
            // NEW: Immediately minimize after pop0 elimination to collapse the explosion.
            if config.prune_dead_paths {
                run_pass!("Pruning dead paths (post-pop0-elim)", {
                    prune_dead_paths_trie3(roots, &trie3_god);
                });
            }
            if config.gc {
                run_pass!("Garbage collection (post-pop0-elim)", {
                    Trie::gc(
                        &trie3_god,
                        &roots.values().cloned().collect::<Vec<_>>(),
                    );
                });
            }
            if config.merge_nodes_structural {
                run_pass!("Merging nodes (structural, post-pop0-elim)", {
                    merge_nodes_trie3_structural(
                        roots,
                        &trie3_god,
                        config.merge_nodes_exact.exact_max_iters,
                    );
                });
                if config.compress_edges {
                    run_pass!("Compressing edges + unary chains (post-pop0-struct-merge)", {
                        compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id); run_mini_pass(CompressChainsPass, roots, trie3_god, &mut ctx);
                    });
                }
                // Defensive: immediately refine token atoms to allow reversing any local structure
                // that the compression step couldn't improve under the global cost metric.
                // This gives the pipeline a way to re-split token sets when that reduces local cost.
                if config.refine_token_atoms {
                    run_pass!("Refining token-set atoms (defensive, post-compress)", {
                        refine_edges_to_token_atoms_trie3(
                            roots,
                            &trie3_god,
                            max_llm_token_id,
                            max_state_id,
                            config.refine_token_atoms_max_blocks,
                        );
                    });
                }
            }
            if config.merge_nodes_exact.enabled {
                run_pass!("Merging nodes (exact, post-pop0-elim)", {
                    merge_nodes_trie3(
                        roots,
                        &trie3_god,
                        config.merge_nodes_exact.exact_max_iters,
                    );
                });
            }
            if config.prune_nodes_not_reaching_end {
                run_pass!("Pruning nodes that do not reach end (post-pop0-elim)", {
                    run_mini_pass(PruneUnproductivePathsPass, roots, trie3_god, &mut ctx);
                });
            }
            if config.compress_edges {
                run_pass!("Compressing edges + unary chains (post-pop0-elim-merge)", {
                    compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id); run_mini_pass(CompressChainsPass, roots, trie3_god, &mut ctx);
                });
            }
            if config.refine_token_atoms {
                run_pass!("Refining token-set atoms (post-pop0-elim-merge)", {
                    refine_edges_to_token_atoms_trie3(
                        roots,
                        &trie3_god,
                        max_llm_token_id,
                        max_state_id,
                        config.refine_token_atoms_max_blocks,
                    );
                });
            }
            if config.refine_token_atoms_exact {
                run_pass!("Refining token-set atoms (exact, post-pop0-elim-merge)", {
                    refine_edges_to_token_atoms_trie3_exact(
                        roots,
                        &trie3_god,
                        max_llm_token_id,
                        max_state_id,
                    );
                });
            }
        }
    }

    crate::debug!(2, "Recomputing max depths...");
    Trie::recompute_all_max_depths(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());

    has_true_cycle_trie3(trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    has_true_cycle_trie3_llm_only(
        trie3_god,
        &roots.values().cloned().collect::<Vec<_>>(),
        stage_vocab.internal_max_llm_token,
    );

    // Optional final exact atom refinement as a polish step
    if config.refine_token_atoms_exact {
        run_pass!("Refining token-set atoms (exact, final)", {
            refine_edges_to_token_atoms_trie3_exact(
                roots,
                &trie3_god,
                max_llm_token_id,
                max_state_id,
            );
        });
    }

    // Final pop=0 elimination pass to be sure.
    if config.eliminate_pop0_edges {
        run_pass!("Eliminating pop=0 edges (final cleanup)", {
            eliminate_pop0_edges_except_roots_trie3(roots, trie3_god);
            if config.assert_no_pop0_nonroot_edges {
                assert_no_pop0_nonroot_edges_trie3(roots, trie3_god);
            }
        });
        if config.compress_edges {
            run_pass!("Compressing edges (post-final-pop0-elim)", {
                compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id); run_mini_pass(CompressChainsPass, roots, trie3_god, &mut ctx);
            });
        }
        if config.merge_nodes_exact.enabled {
            run_pass!("Merging nodes (post-final-pop0-elim)", {
                merge_nodes_trie3(
                    roots,
                    &trie3_god,
                    config.merge_nodes_exact.exact_max_iters,
                );
            });
        }
    }

    // Enforce single END node invariant as a final polish step.
    // This guarantees we never leave the pipeline with more than one END node,
    // even if earlier passes already canonicalized them.
    run_pass!("Canonicalizing END nodes (final polish)", {
        canonicalize_end_nodes_trie3(roots, trie3_god);
    });

    if let (Some(original_arena), Some(original_roots)) = (original_arena, original_roots) {
        crate::debug!(2, "Running stochastic equivalence check...");

        type PrecomputeTrie3Path = (
            PrecomputedNodeContents,
            Vec<((isize, LLMTokenBV), StateIDBV, PrecomputedNodeContents)>,
        );

        fn pretty_print_path(path: &PrecomputeTrie3Path) -> String {
            let (root_contents, edges) = path;
            let mut s = String::new();
            s.push_str(&format!(
                "[Root] end: {}, live_tokens: {:?}\n",
                root_contents.end, root_contents.live_tokens
            ));

            for (i, (edge_key, state_bv, dest_contents)) in edges.iter().enumerate() {
                let (pop, llm_bv) = edge_key;
                s.push_str("  |\n");
                s.push_str(&format!(
                    "  +-- Edge(pop={}): llm_tokens={:?}, state_ids={:?}\n",
                    pop, llm_bv, state_bv
                ));
                s.push_str("  |\n");
                s.push_str("  V\n");
                s.push_str(&format!(
                    "[Node {}] end: {}, live_tokens: {:?}\n",
                    i + 1,
                    dest_contents.end,
                    dest_contents.live_tokens
                ));
            }
            s
        }

        fn get_path_summary(
            path: &PrecomputeTrie3Path,
        ) -> (LLMTokenBV, BTreeMap<isize, StateIDBV>) {
            let (root_contents, edges) = path;
            let mut llm_intersection = LLMTokenBV::max_ones();
            let mut state_checks = BTreeMap::<isize, StateIDBV>::new();
            let mut current_pos: isize = 0;

            let mut current_node_live_tokens = &root_contents.live_tokens;

            for ((pop, llm_bv), state_bv, next_node_contents) in edges {
                llm_intersection &= current_node_live_tokens;
                llm_intersection &= llm_bv;

                current_pos += *pop;
                state_checks
                    .entry(current_pos)
                    .and_modify(|e| *e &= state_bv)
                    .or_insert_with(|| state_bv.clone());

                current_node_live_tokens = &next_node_contents.live_tokens;
            }

            (llm_intersection, state_checks)
        }

        let compare = |p1: &PrecomputeTrie3Path, p2: &PrecomputeTrie3Path| -> PathComparison {
            // Ignore 'live_tokens' in node contents; only 'end' must match at each node.
            if p1.0.end != p2.0.end {
                return PathComparison::Different;
            }

            if p1.1.len() > p2.1.len() {
                return PathComparison::Different;
            }
            // We only need to compare the prefix of p2 that has the same length as p1.
            let p2_prefix_path = (p2.0.clone(), p2.1[..p1.1.len()].to_vec());

            let (llm1, states1) = get_path_summary(p1);
            let (llm2_prefix, states2_prefix) = get_path_summary(&p2_prefix_path);

            if !llm2_prefix.is_subset(&llm1) {
                return PathComparison::Different;
            }

            let all_positions: BTreeSet<_> =
                states1.keys().chain(states2_prefix.keys()).copied().collect();
            for pos in all_positions {
                let s1 = states1
                    .get(&pos)
                    .cloned()
                    .unwrap_or_else(StateIDBV::max_ones);
                let s2_prefix = states2_prefix
                    .get(&pos)
                    .cloned()
                    .unwrap_or_else(StateIDBV::max_ones);
                if !s2_prefix.is_subset(&s1) {
                    return PathComparison::Different;
                }
            }

            if p1.1.len() == p2.1.len() {
                PathComparison::Equal
            } else {
                PathComparison::Prefix
            }
        };

        // Note: get_path_summary now ignores 'live_tokens' completely (uses only edge masks).
        let mut rng = rand::thread_rng();
        let num_samples = 1000;
        let max_path_len = 50;

        let original_roots_vec: Vec<_> = original_roots.values().cloned().collect();
        let optimized_roots_vec: Vec<_> = roots.values().cloned().collect();

        let mut failing_path: Option<PrecomputeTrie3Path> = None;
        let mut a_is_original = true;

        // Check paths from original in optimized
        for _ in 0..num_samples {
            if let Some(path) =
                Trie::sample_path(&original_arena, &original_roots_vec, max_path_len, &mut rng)
            {
                if get_path_summary(&path).0.is_empty() {
                    continue; // Path is semantically invalid due to inconsistent live_tokens.
                }
                if !Trie::path_exists(trie3_god, &optimized_roots_vec, &path, compare) {
                    failing_path = Some(path);
                    a_is_original = true;
                    break;
                }
            }
        }

        if failing_path.is_none() {
            // Check paths from optimized in original
            for _ in 0..num_samples {
                if let Some(path) =
                    Trie::sample_path(trie3_god, &optimized_roots_vec, max_path_len, &mut rng)
                {
                    if get_path_summary(&path).0.is_empty() {
                        continue; // Path is semantically invalid
                    }
                    if !Trie::path_exists(&original_arena, &original_roots_vec, &path, compare) {
                        failing_path = Some(path);
                        a_is_original = false;
                        break;
                    }
                }
            }
        }

        if let Some(path) = failing_path {
            println!("Stochastic equivalence check FAILED!");
            let (trie_a_name, trie_b_name) = if a_is_original {
                ("original", "optimized")
            } else {
                ("optimized", "original")
            };
            println!(
                "Found path in {} that does not exist in {}.",
                trie_a_name, trie_b_name
            );

            let (llm, states) = get_path_summary(&path);
            println!("--- Failing Path Summary ---");
            println!("Valid for LLM tokens: {:?}", llm);
            println!("State checks per position:");
            for (pos, sids) in &states {
                println!("  pos {}: {:?}", pos, sids);
            }
            println!("\n--- Failing Path (Formatted) ---");
            println!("{}", pretty_print_path(&path));

            // Minimize the example
            let (mut arena_a, mut roots_a_vec) = if a_is_original {
                (original_arena.deep_clone(), original_roots_vec.clone())
            } else {
                (trie3_god.deep_clone(), optimized_roots_vec.clone())
            };
            let (mut arena_b, mut roots_b_vec) = if a_is_original {
                (trie3_god.deep_clone(), optimized_roots_vec)
            } else {
                (original_arena.deep_clone(), original_roots_vec)
            };

            println!("Attempting to minimize failing example by trimming...");
            for i in 0..100 {
                let (trimmed_a, trimmed_roots_a) =
                    Trie::trim_randomly(&arena_a, &roots_a_vec, 0.1, &mut rng);
                if Trie::path_exists(&trimmed_a, &trimmed_roots_a, &path, compare) {
                    arena_a = trimmed_a;
                    roots_a_vec = trimmed_roots_a;
                }
                let (trimmed_b, trimmed_roots_b) =
                    Trie::trim_randomly(&arena_b, &roots_b_vec, 0.1, &mut rng);
                if !Trie::path_exists(&trimmed_b, &trimmed_roots_b, &path, compare) {
                    arena_b = trimmed_b;
                    roots_b_vec = trimmed_roots_b;
                }
                if (i + 1) % 10 == 0 {
                    println!("Trimming iteration {}...", i + 1);
                }
            }

            println!("--- MINIMIZED FAILING EXAMPLE ---");
            println!("--- Failing Path Summary ---");
            let (llm, states) = get_path_summary(&path);
            println!("Valid for LLM tokens (edge-derived only): {:?}", llm);
            println!("State checks per position:");
            for (pos, sids) in &states {
                println!("  pos {}: {:?}", pos, sids);
            }
            println!("\n--- Failing Path (Formatted) ---");
            println!("{}", pretty_print_path(&path));
            println!("\n--- Trie A (path SHOULD exist here) ---");
            println!("{}", Trie::pretty_print(&arena_a, &roots_a_vec));
            println!("\n--- Trie B (path should NOT exist here) ---");
            println!("{}", Trie::pretty_print(&arena_b, &roots_b_vec));

            panic!("Stochastic equivalence check failed. See minimized example above.");
        } else {
            crate::debug!(2, "Stochastic equivalence check passed.");
        }
    }

    if config.assert_pop0_paths_to_end_are_short {
        assert_pop0_paths_to_end_are_short(roots, trie3_god);
    }

    crate::debug!(2, "Finished optimizing Trie 3 size.");
}
