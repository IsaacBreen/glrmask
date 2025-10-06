use std::collections::BTreeSet;
use std::collections::{BTreeMap, HashMap, VecDeque, HashSet};
use std::sync::Arc;
use std::time::Instant;
use range_set_blaze::RangeSetBlaze;
use indicatif::{ProgressBar, ProgressStyle};
use kdam::tqdm;
use ordered_hash_map::OrderedHashMap;
use crate::constraint::{GrammarConstraintConfig, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper, StageVocab, PrecomputedNodeContents};
use crate::constraint_extra::{calculate_final_stats3, print_precompute_stats3, PrecomputeStats};
use crate::datastructures::EntryApi;
use crate::constraint::LLMTokenBV;
use crate::datastructures::trie::{EdgeInserter, Trie, Trie2Index};
use crate::tokenizer::TokenizerStateID;

use crate::profiler::PROGRESS_BAR_ENABLED;

fn count_total_ranges_trie3(
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

fn compute_and_print_precompute_stats3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    let mut stats = PrecomputeStats::default();
    calculate_final_stats3(roots, &mut stats, trie3_god);
    print_precompute_stats3(&stats, trie3_god);
}

pub fn optimize_trie3_size(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    config: &GrammarConstraintConfig,
    max_state_id: usize,
    mut max_llm_token_id: usize,
    stage_vocab: &mut StageVocab,
) {
	crate::debug!(2, "Optimizing Trie 3 size...");

	crate::debug!(2, "Initial stats:");
	compute_and_print_precompute_stats3(roots, trie3_god);

	let mut step_counter = 1;
	macro_rules! run_pass {
		($name:expr, $code:block) => {
			crate::debug!(2, "Running optimization pass {}: {}...", step_counter, $name);
			let start = Instant::now();
			$code
			let duration = start.elapsed();
			crate::debug!(2, "Pass {} ('{}') finished in {:?}", step_counter, $name, duration);
			crate::debug!(2, "Stats after pass {}:", step_counter);
			compute_and_print_precompute_stats3(roots, trie3_god);
			step_counter += 1;
		};
	}

	// if config.optimize_trie3_merge_equivalent_llm_tokens {
	// 	run_pass!("Merging equivalent LLM tokens", {
	// 		merge_equivalent_llm_tokens_trie3(roots, trie3_god, stage_vocab);
	// 	});
	// }
    //
	// if config.optimize_trie2_gc {
	// 	run_pass!("Garbage collection (pre-merge)", {
	// 		Trie::gc(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
	// 	});
	// }

	// // After compression, prune and GC before the expensive merge.
	// if config.optimize_trie2_prune_dead_paths {
	// 	run_pass!("Pruning dead paths (post-compress)", {
	// 		prune_dead_paths_trie3(roots, &trie3_god);
	// 	});
	// }
    //
	// if config.optimize_trie3_constrain_bitvecs {
	// 	let roots_vec: Vec<_> = roots.values().cloned().collect();
	// 	let _all_nodes_pinner = Trie::all_nodes(&trie3_god, &roots_vec);
	// 	run_pass!("Constraining bitvectors", {
	// 		constrain_bitvecs_trie3(trie3_god, &roots_vec, max_state_id, max_llm_token_id);
	// 	});
	// }
    //
	// run_pass!("Simplifying LLM token bitsets", {
	// 	simplify_llm_token_bvs_trie3(roots, &trie3_god, max_llm_token_id);
	// });

	// if config.optimize_trie2_compress_edges {
	// 	run_pass!("Compressing edges", {
	// 		compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
	// 	});
	// }

	// // After compression, prune and GC before the expensive merge.
	// if config.optimize_trie2_prune_dead_paths {
	// 	run_pass!("Pruning dead paths (post-compress)", {
	// 		prune_dead_paths_trie3(roots, &trie3_god);
	// 	});
	// }

	// if config.optimize_trie2_merge_nodes {
	// 	run_pass!("Merging nodes", {
	// 		merge_nodes_trie3(roots, &trie3_god);
	// 	});
	// }

	// --- Phase 1: Initial Pruning & Vocab Reduction ---
	// These passes are expensive but have a huge impact on the initial massive graph.
	// They are essential to run first to make subsequent passes feasible.

	if config.optimize_trie2_merge_nodes {
		run_pass!("Merging nodes (fast pre-pass)", {
			merge_nodes_trie3_fast(roots, trie3_god);
		});
	}


	if config.optimize_trie2_prune_dead_paths { // Reusing config flag
		run_pass!("Pruning dead paths", {
			prune_dead_paths_trie3(roots, &trie3_god);
		});
	}

	// if config.optimize_trie2_prune_dead_paths { // Reusing config flag
	// 	run_pass!("Pruning nodes that do not reach end", {
	// 		prune_nodes_not_reaching_end_trie3(roots, &trie3_god);
	// 	});
	// }

	if config.optimize_trie3_merge_equivalent_llm_tokens {
		run_pass!("Merging equivalent LLM tokens", {
			merge_equivalent_llm_tokens_trie3(roots, trie3_god, stage_vocab);
		});
	}

	if config.optimize_trie3_reorder_llm_tokens {
		run_pass!("Reordering LLM tokens for range minimization", {
			reorder_llm_tokens_for_range_minimization_trie3(roots, trie3_god, stage_vocab);
			max_llm_token_id = stage_vocab.internal_max_llm_token;
		});
	}

	if config.optimize_trie3_constrain_bitvecs {
		let roots_vec: Vec<_> = roots.values().cloned().collect();
		let _all_nodes_pinner = Trie::all_nodes(&trie3_god, &roots_vec);
		run_pass!("Constraining bitvectors", {
			constrain_bitvecs_trie3(trie3_god, &roots_vec, max_state_id, max_llm_token_id);
		});
	}

	// --- Phase 2: Structural Compression and Merging ---
	// Now that the graph is smaller and token sets are simpler, we can apply
	// heavy structural optimizations.

	// run_pass!("Simplifying LLM token bitsets", {
	// 	simplify_llm_token_bvs_trie3(roots, &trie3_god, max_llm_token_id);
	// });

	if config.optimize_trie2_compress_edges {
		run_pass!("Compressing edges", {
			compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
		});
	}

	// After compression, prune and GC before the expensive merge.
	if config.optimize_trie2_prune_dead_paths {
		run_pass!("Pruning dead paths (post-compress)", {
			prune_dead_paths_trie3(roots, &trie3_god);
		});
	}
	if config.optimize_trie2_gc {
		run_pass!("Garbage collection (pre-merge)", {
			Trie::gc(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
		});
	}

	if config.optimize_trie2_merge_nodes {
		run_pass!("Merging nodes", {
			merge_nodes_trie3(roots, &trie3_god);
		});
	}

	// --- Phase 3: Iterative Refinement ---
	// A few rounds of compression and merging on the now much smaller graph.

	// if config.optimize_trie2_prune_dead_paths { // Reusing config flag
	// 	run_pass!("Pruning nodes that do not reach end (post-merge)", {
	// 		prune_nodes_not_reaching_end_trie3(roots, &trie3_god);
	// 	});
	// }

	if config.optimize_trie2_compress_edges {
		run_pass!("Compressing edges (post-merge)", {
			compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
		});
	}

	if config.optimize_trie2_merge_nodes {
		run_pass!("Merging nodes (post-compress)", {
			merge_nodes_trie3(roots, &trie3_god);
		});
	}

	// --- Phase 4: Final Cleanup and Polish ---

	if config.optimize_trie2_prune_dead_paths {
		run_pass!("Pruning dead paths (final)", {
			prune_dead_paths_trie3(roots, &trie3_god);
		});
	}

	if config.optimize_trie2_gc {
		run_pass!("Garbage collection (final)", {
			Trie::gc(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
		});
	}

	if config.optimize_trie3_merge_equivalent_llm_tokens {
		run_pass!("Merging equivalent LLM tokens (final pass)", {
			merge_equivalent_llm_tokens_trie3(roots, trie3_god, stage_vocab);
		});
	}
	if config.optimize_trie3_reorder_llm_tokens {
		run_pass!("Reordering LLM tokens (final pass)", {
			reorder_llm_tokens_for_range_minimization_trie3(roots, trie3_god, stage_vocab);
		});
	}

	run_pass!("Recomputing max depths", {
		Trie::recompute_all_max_depths(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
	});

	crate::debug!(2, "Finished optimizing Trie 3 size.");
}

fn remap_llm_bv_many_to_one(bv: &LLMTokenBV, map_old_to_new: &BTreeMap<usize, usize>, max_token_id: usize) -> LLMTokenBV {
    if bv.is_empty() { return LLMTokenBV::zeros(); }

    let mut ranges_to_add = Vec::new();
    let mut elements_to_add = Vec::new();

    let process_range = |range: std::ops::RangeInclusive<usize>,
                         ranges_to_add: &mut Vec<_>,
                         elements_to_add: &mut Vec<_>| {
        let (mut current, end) = (*range.start(), *range.end());
        for (&k, &v) in map_old_to_new.range(current..=end) {
            if current < k {
                ranges_to_add.push(current..=k - 1);
            }
            elements_to_add.push(v);
            current = k + 1;
        }
        if current <= end {
            ranges_to_add.push(current..=end);
        }
    };

    if *bv == LLMTokenBV::max_ones() {
        process_range(0..=max_token_id, &mut ranges_to_add, &mut elements_to_add);
    } else {
        for range in bv.inner().ranges() {
            process_range(*range.start()..=*range.end(), &mut ranges_to_add, &mut elements_to_add);
        }
    }

    let mut new_set = RangeSetBlaze::from_iter(ranges_to_add);
    new_set.extend(elements_to_add);

    LLMTokenBV { inner: crate::datastructures::cache::intern_l1(new_set) }
}

fn remap_llm_bv_permutation(bv: &LLMTokenBV, map_old_to_new: &BTreeMap<usize, usize>, _max_token_id: usize) -> LLMTokenBV {
    // Fast‑paths for empty or full‑universe bitvectors.
    if bv.is_empty() { return LLMTokenBV::zeros(); }
    if *bv == LLMTokenBV::max_ones() { return LLMTokenBV::max_ones(); }

    let mut ranges_to_add = Vec::new();
    let mut elements_to_add = Vec::new();

    for range in bv.inner().ranges() {
        let (mut current, end) = (*range.start(), *range.end());

        for (&k, &v) in map_old_to_new.range(current..=end) {
            if current < k {
                // Identity mapping for elements not in map_old_to_new
                ranges_to_add.push(current..=k - 1);
            }
            elements_to_add.push(v);
            current = k + 1;
        }

        if current <= end {
            // Identity mapping for the rest of the range
            ranges_to_add.push(current..=end);
        }
    }

    let mut new_set = RangeSetBlaze::from_iter(ranges_to_add);
    new_set.extend(elements_to_add);

    LLMTokenBV { inner: crate::datastructures::cache::intern_l1(new_set) }
}

/// Merge equivalent internal LLM token ids in Trie3:
/// Two tokens are equivalent if they occur together in every LLMTokenBV occurrence across:
/// - node.value.live_tokens
/// - each edge key's (pop, LLMTokenBV) mask
///
/// Applies many-to-one mapping into representative ids, remapping node/edge masks,
/// and updates the provided StageVocab.
pub fn merge_equivalent_llm_tokens_trie3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    crate::debug!(2, "Merging equivalent LLM tokens in Trie3...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    // 1) Collect all unique bitsets to use as splitters.
    let mut all_bvs = HashSet::new();
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie3 Merge Tokens (Collect BVs)", disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let g = n.read(trie3_god).expect("read");
        if !g.value.live_tokens.is_empty() {
            all_bvs.insert(g.value.live_tokens.clone());
        }
        for ((_, llm_bv), _dm) in g.children() {
            if !llm_bv.is_empty() {
                all_bvs.insert(llm_bv.clone());
            }
        }
    }
    if all_bvs.is_empty() { return; }

    // 2) Partition refinement.
    let max_tok = stage_vocab.internal_max_llm_token;
    let mut token_to_class: Vec<usize> = vec![0; max_tok + 1];
    let mut class_to_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
    class_to_tokens.insert(0, (0..=max_tok).collect());
    let mut num_classes = 1;

    #[cfg(not(rustrover))]
    let it = tqdm!(all_bvs.iter(), desc = "Trie3 Merge Tokens (Refine)", disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_bvs.iter();
    for splitter_bv in it {
        if *splitter_bv == LLMTokenBV::max_ones() { continue; }

        let mut members_in_splitter_by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for token in splitter_bv.iter() {
            if token <= max_tok {
                let class_id = token_to_class[token];
                members_in_splitter_by_class.entry(class_id).or_default().push(token);
            }
        }

        for (old_class_id, tokens_for_new_class) in members_in_splitter_by_class {
            let old_class_size = class_to_tokens.get(&old_class_id).map_or(0, |v| v.len());
            if old_class_size == 0 { continue; }

            if !tokens_for_new_class.is_empty() && tokens_for_new_class.len() < old_class_size {
                let new_class_id = num_classes;
                num_classes += 1;

                for &token in &tokens_for_new_class {
                    token_to_class[token] = new_class_id;
                }

                let old_class_tokens = class_to_tokens.get_mut(&old_class_id).unwrap();
                let moved_tokens_set: HashSet<_> = tokens_for_new_class.iter().cloned().collect();
                old_class_tokens.retain(|t| !moved_tokens_set.contains(t));
                
                class_to_tokens.insert(new_class_id, tokens_for_new_class);
            }
        }
    }

    // 3) Build many-to-one mapping from the final partition.
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    let mut merged_count = 0;
    for (_class_id, group) in &class_to_tokens {
        if group.len() <= 1 { continue; }
        let rep = *group.iter().min().unwrap();
        for &t in group {
            if t != rep {
                old_to_new.insert(t, rep);
                merged_count += 1;
            }
        }
    }
    let tokens_before = max_tok + 1;
    let tokens_after = num_classes;
    crate::debug!(2, "Trie3: merged LLM tokens. Before: {}, After: {}. ({} merged)", tokens_before, tokens_after, merged_count);
    if merged_count == 0 { return; }

    // Memoization cache for remapping.
    let mut memo: HashMap<LLMTokenBV, LLMTokenBV> = HashMap::new();

    // Precompute the mapped universal set once (used when a set equals max_ones())
    let mut mapped_universe = LLMTokenBV::zeros();
    for t in 0..=max_tok {
        let rep = old_to_new.get(&t).copied().unwrap_or(t);
        mapped_universe.insert(rep);
    }

    // Identify affected bitvector instances (by Arc pointer identity)
    let mut affected_ptrs: HashSet<*const RangeSetBlaze<usize>> = HashSet::new();
    for splitter in all_bvs {
        affected_ptrs.insert(Arc::as_ptr(&splitter.inner));
    }

    // 4) Remap trie in‑place, only where needed
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie3 Merge (Remap In‑Place)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        // Quick check whether this node references any affected bitvector.
        let needs_update = {
            let r = n.read(trie3_god).expect("read");
            let lv_ptr = Arc::as_ptr(&r.value.live_tokens.inner);
            if affected_ptrs.contains(&lv_ptr) {
                true
            } else {
                let mut touched = false;
                for ((_, llm_bv), _dm) in r.children() {
                    let key_ptr = Arc::as_ptr(&llm_bv.inner);
                    if affected_ptrs.contains(&key_ptr) {
                        touched = true;
                        break;
                    }
                }
                touched
            }
        };
        if !needs_update { continue; }

        let mut w = n.write(trie3_god).expect("write");

        // Remap live_tokens if needed
        if !w.value.live_tokens.is_empty() {
            if w.value.live_tokens == LLMTokenBV::max_ones() {
                w.value.live_tokens = mapped_universe.clone();
            } else {
                let lv_ptr = Arc::as_ptr(&w.value.live_tokens.inner);
                if affected_ptrs.contains(&lv_ptr) {
                    let original_bv = w.value.live_tokens.clone();
                    w.value.live_tokens = memo.entry(original_bv)
                        .or_insert_with_key(|bv| remap_llm_bv_many_to_one(bv, &old_to_new, max_tok))
                        .clone();
                }
            }
        }

        // Remap edge keys (pop, LLMTokenBV)
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<PrecomputeNode3Index, StateIDBV>> = BTreeMap::new();
        for ((pop, llm_bv), dm) in old_children {
            let mapped_key_bv = if llm_bv.is_empty() {
                LLMTokenBV::zeros()
            } else if llm_bv == LLMTokenBV::max_ones() {
                mapped_universe.clone()
            } else {
                let key_ptr = Arc::as_ptr(&llm_bv.inner);
                if affected_ptrs.contains(&key_ptr) {
                    memo.entry(llm_bv)
                        .or_insert_with_key(|bv| remap_llm_bv_many_to_one(bv, &old_to_new, max_tok))
                        .clone()
                } else {
                    llm_bv
                }
            };
            if mapped_key_bv.is_empty() { continue; }
            let entry = new_children.entry((pop, mapped_key_bv)).or_insert_with(OrderedHashMap::new);
            for (dst, sid_bv) in dm {
                entry.entry(dst)
                    .and_modify(|e| *e |= &sid_bv)
                    .or_insert(sid_bv);
            }
		}
		*w.children_mut() = new_children;
	}
	// 5) Update StageVocab
	#[cfg(not(rustrover))]
	let it = tqdm!(old_to_new.iter(), desc = "Trie3 Merge (Update Vocab)", total = old_to_new.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
	#[cfg(rustrover)]
	let it = old_to_new.iter();
	for (old, rep) in it {
		if old == rep { continue; }
		if let Some(moved) = stage_vocab.internal_to_original.remove(old) {
			let entry = stage_vocab.internal_to_original.entry(*rep).or_default();
			*entry |= &moved;
			for o in moved.iter() {
				stage_vocab.original_to_internal.insert(o, *rep);
			}
		}
	}
}

/// Reorder internal LLM tokens in Trie3 with a simple heuristic to cluster co-occurring tokens.
pub fn reorder_llm_tokens_for_range_minimization_trie3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    crate::debug!(2, "Reordering LLM tokens in Trie3 for range minimization...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() { return; }
    let ranges_before = count_total_ranges_trie3(&all_nodes, trie3_god);

    let max_tok = stage_vocab.internal_max_llm_token;

    // 1. Collect unique BV counts to optimize frequency calculation.
    let mut bv_counts: HashMap<LLMTokenBV, usize> = HashMap::new();
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie3 Reorder (Collect BVs)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let g = n.read(trie3_god).expect("read");
        if !g.value.live_tokens.is_empty() {
            *bv_counts.entry(g.value.live_tokens.clone()).or_default() += 1;
        }
        for ((_, llm_bv), _dm) in g.children() {
            if !llm_bv.is_empty() {
                *bv_counts.entry(llm_bv.clone()).or_default() += 1;
            }
        }
    }

    // 2. Compute token frequencies from unique BV counts.
    let mut freq: Vec<usize> = vec![0; max_tok + 1];
    #[cfg(not(rustrover))]
    let it = tqdm!(bv_counts.iter(), desc = "Trie3 Reorder (Count Frequencies)", total = bv_counts.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = bv_counts.iter();
    for (bv, &count) in it {
        if bv.is_all() {
            for t in 0..=max_tok {
                freq[t] += count;
            }
        } else {
            for t in bv.iter() {
                if t <= max_tok {
                    freq[t] += count;
                }
            }
        }
    }
    crate::debug!(2, "Done computing frequencies.");
    let mut present: Vec<usize> = (0..=max_tok).filter(|t| freq[*t] > 0).collect();
    if present.is_empty() { return; }
    present.sort_by_key(|&t| (std::cmp::Reverse(freq[t]), t));

    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    for (new_id, old_id) in present.iter().enumerate() {
        old_to_new.insert(*old_id, new_id);
    }

    // Memoization cache
    let mut memo: HashMap<LLMTokenBV, LLMTokenBV> = HashMap::new();

    let mut new_states = Vec::with_capacity(all_nodes.len());
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie3 Reorder (Remap Read)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let r = n.read(trie3_god).expect("read");
        let new_live_tokens = if r.value.live_tokens.is_empty() {
            r.value.live_tokens.clone()
        } else {
            memo.entry(r.value.live_tokens.clone())
                .or_insert_with_key(|bv| remap_llm_bv_permutation(bv, &old_to_new, max_tok))
                .clone()
        };
        let mut new_children = BTreeMap::new();
        for ((pop, llm_bv), dm) in r.children() {
            let mapped_key_bv = memo.entry(llm_bv.clone())
                .or_insert_with_key(|bv| remap_llm_bv_permutation(bv, &old_to_new, max_tok))
                .clone();
            if mapped_key_bv.is_empty() { continue; }
            let entry = new_children.entry((*pop, mapped_key_bv)).or_insert_with(OrderedHashMap::new);
            for (dst, sid_bv) in dm {
                entry.entry(dst.clone()).and_modify(|e| *e |= sid_bv).or_insert_with(|| sid_bv.clone());
            }
        }
        new_states.push((new_live_tokens, new_children));
    }
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter().enumerate(), desc = "Trie3 Reorder (Remap Write)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter().enumerate();
    for (i, n) in it {
        let mut w = n.write(trie3_god).expect("write");
        let (live_tokens, children) = &new_states[i];
        w.value.live_tokens = live_tokens.clone();
        *w.children_mut() = children.clone();
    }
	let ranges_after = count_total_ranges_trie3(&all_nodes, trie3_god);

	// Update StageVocab under permutation
	let mut new_internal_to_original: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
	#[cfg(not(rustrover))]
	let it = tqdm!(stage_vocab.internal_to_original.clone().into_iter(), desc = "Trie3 Reorder (Vocab 1)", disable = !PROGRESS_BAR_ENABLED, leave = true);
	#[cfg(rustrover)]
	let it = stage_vocab.internal_to_original.clone().into_iter();
	for (old_id, setv) in it {
		if let Some(new_id) = old_to_new.get(&old_id) {
			new_internal_to_original.insert(*new_id, setv);
        }
    }
    stage_vocab.internal_to_original = new_internal_to_original;
    let mut new_original_to_internal: BTreeMap<usize, usize> = BTreeMap::new();
    #[cfg(not(rustrover))]
    let it = tqdm!(stage_vocab.original_to_internal.clone().into_iter(), desc = "Trie3 Reorder (Vocab 2)", disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = stage_vocab.original_to_internal.clone().into_iter();
    for (orig, old_internal) in it {
        if let Some(new_internal) = old_to_new.get(&old_internal) {
            new_original_to_internal.insert(orig, *new_internal);
        }
    }
    stage_vocab.original_to_internal = new_original_to_internal;
    stage_vocab.internal_max_llm_token = present.len().saturating_sub(1);
    crate::debug!(2, "Trie3 reordering complete. Ranges reduced from {} to {}. New max internal token ID: {}", ranges_before, ranges_after, stage_vocab.internal_max_llm_token);
}

fn simplify_llm_token_bvs_trie3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
) {
    return;
    crate::debug!(2, "Simplifying LLM token bitsets in Trie3 to reduce range counts...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    let universe = LLMTokenBV::ones(max_llm_token_id + 1);

    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie3 Simplify LLM BVs", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false);
    #[cfg(rustrover)]
    let it = all_nodes.iter();

    for node_idx in it {
        let mut w = node_idx.write(trie3_god).expect("write");
        if w.children().is_empty() {
            continue;
        }

        let live_u = w.value.live_tokens.clone();
        if live_u.is_all() { // If all tokens are live, no simplification is possible.
            continue;
        }
        let dead_u = &universe - &live_u;

        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<PrecomputeNode3Index, StateIDBV>> = BTreeMap::new();

        for ((pop, l), dm) in old_children {
            let mut l_new = l.clone();
            l_new |= &dead_u;

            let entry = new_children.entry((pop, l_new)).or_default();
            for (dest, sids) in dm {
                entry.entry(dest).and_modify(|e| *e |= &sids).or_insert(sids);
            }
        }
        *w.children_mut() = new_children;
    }
    crate::debug!(2, "Finished simplifying LLM token bitsets.");
}

fn prune_nodes_not_reaching_end_trie3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    return;
    crate::debug!(2, "Pruning Trie3 nodes that cannot reach any end node (reverse reachability)...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if roots_vec.is_empty() {
        return;
    }

    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // Build reverse adjacency: dest -> sources
    let mut incoming: HashMap<PrecomputeNode3Index, Vec<PrecomputeNode3Index>> = HashMap::new();
    for src in &all_nodes {
        let g = src.read(trie3_god).expect("read");
        for (_ek, dm) in g.children() {
            for (dst, _bv) in dm {
                incoming.entry(*dst).or_default().push(*src);
            }
        }
    }

    // Initialize worklist with all end nodes
    let mut productive: HashSet<PrecomputeNode3Index> = HashSet::new();
    let mut q: VecDeque<PrecomputeNode3Index> = VecDeque::new();
    let mut end_nodes_count = 0usize;
    for n in &all_nodes {
        let r = n.read(trie3_god).expect("read");
        if r.value.end {
            end_nodes_count += 1;
            if productive.insert(*n) {
                q.push_back(*n);
            }
        }
    }
    if end_nodes_count == 0 {
        crate::debug!(2, "No end nodes found in Trie3; skipping end-reachability pruning.");
        return;
    }

    // Reverse BFS
    while let Some(d) = q.pop_front() {
        if let Some(srcs) = incoming.get(&d) {
            for s in srcs {
                if productive.insert(*s) {
                    q.push_back(*s);
                }
            }
        }
    }

    let total_nodes = all_nodes.len();
    let productive_nodes = productive.len();
    let prunable = total_nodes.saturating_sub(productive_nodes);
    crate::debug!(2, "Trie3 end-reachability: total={}, productive={}, prunable={}", total_nodes, productive_nodes, prunable);
    if prunable == 0 {
        return;
    }

    // Remove any edge to a non-productive destination
    for n in &all_nodes {
        let mut w = n.write(trie3_god).expect("write");
        let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();
        for (ek, dm) in w.children().clone() {
            let mut new_dm: OrderedHashMap<Trie2Index, StateIDBV> = OrderedHashMap::new();
            for (dst, bv) in dm {
                if productive.contains(&dst) {
                    new_dm.insert(dst, bv);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
    }

    // GC everything now unreachable from roots
    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie3_god, &roots_vec2);
    Trie::recompute_all_max_depths(trie3_god, &roots_vec2);

    crate::debug!(2, "Finished end-reachability pruning in Trie3.");
}

fn factor_common_destinations_trie3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
) {
    return;
    crate::debug!(2, "Factoring out common destinations in Trie3.");
    const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3;

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    let all_llm_bv = LLMTokenBV::ones(max_llm_token_id + 1);
    let all_sids_bv = StateIDBV::ones(max_state_id + 1);

    // Map: dest -> { (pop, llm_bv) -> { state_id_bv -> [sources] } }
    let mut incoming_map: HashMap<
        PrecomputeNode3Index,
        HashMap<
            (usize, LLMTokenBV),
            HashMap<StateIDBV, Vec<PrecomputeNode3Index>>,
        >,
    > = HashMap::new();

    for src_idx in &all_nodes {
        let guard = src_idx.read(trie3_god).expect("read");
        for (edge_key, dest_map) in guard.children() {
            for (dest_idx, sids_bv) in dest_map {
                incoming_map
                    .entry(*dest_idx)
                    .or_default()
                    .entry(edge_key.clone())
                    .or_default()
                    .entry(sids_bv.clone())
                    .or_default()
                    .push(*src_idx);
            }
        }
    }

    for (dest_idx, edges_by_key) in incoming_map {
        for (edge_key, sources_by_sids) in edges_by_key {
            for (sids_bv, sources) in sources_by_sids {
                if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                    // Create intermediate node
                    let intermediate_node = PrecomputeNode3Index::new(trie3_god.insert(Trie::new(PrecomputedNodeContents::internal())));

                    // Add edge from intermediate to original destination
                    {
                        let mut intermediate_guard = intermediate_node.write(trie3_god).expect("write");
                        let dest_map = intermediate_guard.children_mut().entry(edge_key.clone()).or_default();
                        dest_map.insert(dest_idx, sids_bv.clone());
                        intermediate_guard.value.live_tokens |= &edge_key.1;
                    }

                    // Reroute sources to point to intermediate node
                    for src_idx in &sources {
                        let mut src_guard = src_idx.write(trie3_god).expect("write");

                        // Remove old edge
                        if let Some(dest_map_for_key) = src_guard.children_mut().get_mut(&edge_key) {
                            dest_map_for_key.remove(&dest_idx);
                            if dest_map_for_key.is_empty() {
                                src_guard.children_mut().remove(&edge_key);
                            }
                        }

                        // Add new edge to intermediate node. This is a "None-like" edge.
                        // pop=0, all llm tokens, all state ids.
                        let none_like_edge_key = (0, all_llm_bv.clone());
                        let dest_map = src_guard.children_mut().entry(none_like_edge_key).or_default();
                        dest_map.insert(intermediate_node, all_sids_bv.clone());
                        src_guard.value.live_tokens |= &all_llm_bv;
                    }
                }
            }
        }
    }
    crate::debug!(2, "Finished factoring common destinations in Trie3.");
}

fn constrain_bitvecs_trie3(
    trie3_god: &Trie3GodWrapper,
    roots_vec: &[PrecomputeNode3Index],
    max_state_id: usize,
    max_llm_token_id: usize,
) {
    crate::debug!(2, "Constraining bitvectors in Trie 3...");
    let all_nodes = Trie::all_nodes(trie3_god, roots_vec);
    if all_nodes.is_empty() { return; }

    for node_arc in all_nodes {
        let mut guard = node_arc.write(trie3_god).unwrap();

        // Constrain live_tokens on the node value
        guard.value.live_tokens.constrain(max_llm_token_id);

        let old_children = std::mem::take(guard.children_mut());
        let mut new_children = BTreeMap::new();

        for ((pop, mut llm_bv), dest_map) in old_children {
            llm_bv.constrain(max_llm_token_id);

            let mut new_dest_map = OrderedHashMap::new();
            for (dest_wrapper, mut sids_bv) in dest_map {
                sids_bv.constrain(max_state_id);
                if !sids_bv.is_empty() {
                    new_dest_map.insert(dest_wrapper, sids_bv);
                }
            }

            if !llm_bv.is_empty() && !new_dest_map.is_empty() {
                // Need to merge if the key (with constrained llm_bv) already exists
                let entry = new_children.entry((pop, llm_bv)).or_insert_with(OrderedHashMap::new);
                for (dest, sids) in new_dest_map {
                    entry.entry(dest)
                        .and_modify(|existing_sids| *existing_sids |= &sids)
                        .or_insert(sids);
                }
            }
        }
        *guard.children_mut() = new_children;
    }
    crate::debug!(2, "Finished constraining bitvectors.");
}

pub fn prune_dead_paths_trie3(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 3.");

    let all_nodes = Trie::all_nodes(trie3_god, &roots.values().cloned().collect::<Vec<_>>());
    if all_nodes.is_empty() { return; }

    let mut predecessors: HashMap<PrecomputeNode3Index, Vec<(PrecomputeNode3Index, (usize, LLMTokenBV))>> = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<PrecomputeNode3Index, LLMTokenBV> = HashMap::new();

    // 1. Initialize live sets and build predecessor map.
    for node_arc in &all_nodes {
        let node_ptr = *node_arc;
        live.insert(node_ptr, LLMTokenBV::zeros());

        let guard = node_arc.read(trie3_god).unwrap();
        if guard.value.end {
            // Seed end nodes with 'all tokens' to allow backward propagation through edge masks.
            live.insert(node_ptr, LLMTokenBV::max_ones());
            worklist.push_back(node_ptr);
        }

        for (edge_key, dest_map) in guard.children() {
            for child_wrap in dest_map.keys() {
                let child_arc = child_wrap.as_arc().clone();
                let child_ptr = child_arc;
                predecessors.entry(child_ptr).or_default().push((node_ptr, edge_key.clone()));
            }
        }
    }

    let pb = ProgressBar::new(all_nodes.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [Trie3 Prune] [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap(),
    );
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }
    pb.set_position(0);

    // 2. Propagate liveness until a fixed point is reached.
    while let Some(node_ptr) = worklist.pop_front() {
        pb.inc(1);

        let live_at_node = live.get(&node_ptr).unwrap().clone();
        if let Some(preds) = predecessors.get(&node_ptr) {
            for (pred_ptr, edge_key) in preds {
                let live_from_edge = &live_at_node & &edge_key.1;
                if live_from_edge.is_empty() {
                    continue;
                }

                let pred_live = live.get_mut(pred_ptr).unwrap();
                let old_len = pred_live.len();
                *pred_live |= &live_from_edge;
                if pred_live.len() > old_len {
                    worklist.push_back(*pred_ptr);
                }
            }
        }
    }
    pb.finish_and_clear();

    // 3. Prune the graph based on the computed live sets.
    for node_arc in &all_nodes {
        let mut guard = node_arc.write(trie3_god).unwrap();
        let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();

        for (edge_key, dest_map) in guard.children() {
            for (child_wrapper, edge_value_sids) in dest_map {
                let child_arc = child_wrapper.as_arc().clone();
                let child_ptr = child_arc;
                let live_from_child = live.get(&child_ptr).unwrap();

                let live_on_edge = &edge_key.1 & live_from_child;

                if !live_on_edge.is_empty() {
                    let new_edge_key = (edge_key.0, live_on_edge);
                    let new_dest_map_for_key = new_children.entry(new_edge_key).or_default();
                    new_dest_map_for_key.entry(*child_wrapper)
                        .and_modify(|v| *v |= edge_value_sids)
                        .or_insert_with(|| edge_value_sids.clone());
                }
            }
        }
        *guard.children_mut() = new_children;

        // Update the node's own live_tokens field with the final computed value.
        let node_ptr = *node_arc;
        guard.value.live_tokens = live.get(&node_ptr).unwrap().clone();
    }
    crate::debug!(2, "Finished pruning dead paths from trie 3.");
}

pub fn merge_nodes_trie3_fast(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    merge_nodes_trie3_impl(roots, trie3_god, 2);
}

pub fn merge_nodes_trie3(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper) {
    const MAX_ITERS: usize = 40;
    merge_nodes_trie3_impl(roots, trie3_god, MAX_ITERS);
}

fn merge_nodes_trie3_impl(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper, max_iters: usize) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 3 (max_iters={}).", max_iters);


    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    let mut dense_of: HashMap<Trie2Index, usize> = HashMap::new();
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(all_nodes.len());
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i);
        old_of.push(*node_idx);
    }
    let n = all_nodes.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge3 = (usize, LLMTokenBV, usize, StateIDBV);
    let mut raw_edges: Vec<Vec<RawEdge3>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(trie3_god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, bv) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    raw_edges[u_dense].push((ek.0, ek.1.clone(), v_dense, bv.clone()));
                }
            }
        }
    }

    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    for it in 0..max_iters {
        type AggregatedEdge3 = ((usize, LLMTokenBV, usize), StateIDBV);
        type Signature3 = (bool, Vec<AggregatedEdge3>);

        let mut sig_to_id: HashMap<Signature3, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        #[cfg(not(rustrover))]
        let its = tqdm!(0..n, desc = format!("Trie3 Merge Iter {}", it + 1), total = n, disable = !PROGRESS_BAR_ENABLED, leave = true);
        #[cfg(rustrover)]
        let its = 0..n;
        for u in its {
            let mut aggr: BTreeMap<(usize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u] {
                let dest_class = prev_class[*v_dense];
                let key = (*p, bv_key.clone(), dest_class);
                aggr.entry(key).and_modify(|e| *e |= sids).or_insert_with(|| sids.clone());
            }
            let agg_edges: Vec<AggregatedEdge3> = aggr.into_iter().collect();

            let sig: Signature3 = (ends[u], agg_edges);

            let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            new_class[u] = cid;
            if new_class[u] != prev_class[u] {
                changes += 1;
            }
        }

        crate::debug!(3, "Trie3 merge iter {}: classes={}, changes={}", it + 1, next_id, changes);
        prev_class = new_class;
        if changes == 0 { break; }
    }

    let final_partition = prev_class;
    let num_classes = final_partition.iter().max().map_or(0, |m| m + 1);

    let mut representatives: Vec<Option<Trie2Index>> = vec![None; num_classes];
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        if representatives[class_id].is_none() {
            representatives[class_id] = Some(old_of[u_dense]);
        }
    }

    let mut node_to_rep: HashMap<Trie2Index, Trie2Index> = HashMap::new();
    for (u_dense, &class_id) in final_partition.iter().enumerate() {
        node_to_rep.insert(old_of[u_dense], representatives[class_id].unwrap());
    }

    for class_id in 0..num_classes {
        if let Some(rep_idx) = representatives[class_id] {
            let u_dense = final_partition.iter().position(|&c| c == class_id).unwrap();

            let mut aggr: BTreeMap<(usize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
            for (p, bv_key, v_dense, sids) in &raw_edges[u_dense] {
                let dest_class = final_partition[*v_dense];
                aggr.entry((*p, bv_key.clone(), dest_class)).and_modify(|e| *e |= sids).or_insert_with(|| sids.clone());
            }

            let mut new_children = BTreeMap::new();
            let mut new_live_tokens = LLMTokenBV::zeros();
            for ((p, bv_key, dest_class), sids) in aggr {
                if let Some(dest_rep_idx) = representatives[dest_class] {
                    new_children.entry((p, bv_key.clone())).or_insert_with(OrderedHashMap::new).insert(dest_rep_idx, sids);
                    new_live_tokens |= &bv_key;
                }
            }

            for (i, &c) in final_partition.iter().enumerate() {
                if c == class_id {
                    new_live_tokens |= &old_of[i].read(trie3_god).unwrap().value.live_tokens;
                }
            }

            let mut guard = rep_idx.write(trie3_god).unwrap();
            *guard.children_mut() = new_children;
            guard.value.live_tokens = new_live_tokens;
        }
    }

    for root_idx in roots.values_mut() {
        *root_idx = *node_to_rep.get(root_idx).unwrap();
    }

    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie3_god, &final_roots_vec);
}

pub fn compress_trie3_edges(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper, max_llm_token_id: usize, max_state_id: usize) {
    crate::debug!(2, "Compressing Trie 3 edges (conservative edge-reducing transforms)...");

    let all_llm_bv = LLMTokenBV::ones(max_llm_token_id + 1);
    let all_sids_bv = StateIDBV::ones(max_state_id + 1);

    // Helper: is the LLM-token BV "all tokens"?
    let is_all_llm = |bv: &LLMTokenBV| -> bool {
        bv.is_superset(&all_llm_bv) || *bv == LLMTokenBV::max_ones()
    };
    // Helper: is the StateIDBV "all states"?
    let is_all_sids = |bv: &StateIDBV| -> bool {
        bv.is_superset(&all_sids_bv) || *bv == StateIDBV::max_ones()
    };

    // Pass 1: local coalesce within each node
    let coalesce_edges_within_nodes = |trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]| -> bool {
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }
        let mut changed_any = false;

        #[cfg(not(rustrover))]
        let it = tqdm!(nodes.iter(), desc = "Trie3 Compress (Coalesce)", total=nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave=false);
        #[cfg(rustrover)]
        let it = nodes.iter();
        for node_idx in it {
            // Snapshot current children
            let old_children = {
                let g = node_idx.read(trie3_god).expect("read");
                g.children().clone() // BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>>
            };
            if old_children.is_empty() { continue; }

            // Aggregate per (pop, child, sids): union LLM-token BVs
            let mut by_pop: HashMap<usize, Vec<(Trie2Index, StateIDBV, LLMTokenBV)>> = HashMap::new();
            for ((pop, llm_bv), dest_map) in &old_children {
                for (child_idx, sids) in dest_map.iter() {
                    let items = by_pop.entry(*pop).or_default();
                    let mut found = false;
                    for (c, c_sids, llm_union) in items.iter_mut() {
                        if c == child_idx && c_sids == sids {
                            *llm_union |= llm_bv;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        items.push((*child_idx, sids.clone(), llm_bv.clone()));
                    }
                }
            }

            // Rebuild children from aggregates
            let mut new_children: BTreeMap<(usize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();
            for (pop, vec_items) in by_pop {
                for (child, sids, llm_union) in vec_items {
                    if llm_union.is_empty() || sids.is_empty() {
                        continue;
                    }
                    new_children.entry((pop, llm_union)).or_default().insert(child, sids);
                }
            }

            if new_children != old_children {
                let mut w = node_idx.write(trie3_god).expect("write");
                // Recompute node live tokens as union of outgoing LLM masks.
                let mut union_bv = LLMTokenBV::zeros();
                for ((_, llm_bv), _) in &new_children {
                    union_bv |= llm_bv;
                }
                *w.children_mut() = new_children;
                w.value.live_tokens = union_bv;
                changed_any = true;
            }
        }

        changed_any
    };

    // Pass 2: shortcut zero-pop chains.
    // Contracts sequences V --(pop 0, L2, S2)--> ... --(pop 0, Lk, Sk)--> Z
    // into U --(p1, L1∩L2∩...∩Lk, S1∩S2∩...∩Sk)--> Z where U --(p1, L1, S1)--> V.
    // Only applies when each intermediate has exactly one outgoing (pop 0) edge with exactly one destination (no fanout), avoiding edge explosion.
    let shortcut_zero_pop_chains = |trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]| -> bool {
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }

        // Snapshot summaries for quick lookups
        type DestList = Vec<(Trie2Index, StateIDBV)>;
        type EdgeList = Vec<(usize, LLMTokenBV, DestList)>;
        let mut summary: HashMap<Trie2Index, (bool, EdgeList)> = HashMap::new();
        for n in &nodes {
            let g = n.read(trie3_god).expect("read");
            let edges: EdgeList = g.children()
                .iter()
                .map(|(ek, dm)| {
                    let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<DestList>();
                    (ek.0, ek.1.clone(), dests)
                })
                .collect();
            summary.insert(*n, (g.value.end, edges));
        }

        // Memoization for zero-pop chain results
        #[derive(Clone)]
        struct ChainRes {
            last: Trie2Index,
            llm: LLMTokenBV,
            sids: StateIDBV,
        }
        let mut memo: HashMap<Trie2Index, Option<ChainRes>> = HashMap::new();

        fn follow_zero_chain(
            v: Trie2Index,
            summary: &HashMap<Trie2Index, (bool, EdgeList)>,
            memo: &mut HashMap<Trie2Index, Option<ChainRes>>,
        ) -> Option<ChainRes> {
            if let Some(cached) = memo.get(&v) {
                return cached.clone();
            }
            let (_is_end, edges) = match summary.get(&v) {
                Some(x) => x,
                None => {
                    memo.insert(v, None);
                    return None;
                }
            };
            // Must be exactly one outgoing edge, pop == 0, with exactly one destination.
            let mut pop0_edges = edges.iter().filter(|(p, _, _)| *p == 0);
            let next = match pop0_edges.next() {
                Some(x) => x,
                None => {
                    memo.insert(v, None);
                    return None;
                }
            };
            // Ensure it is the only outgoing edge and has a single destination.
            if edges.len() != 1 || next.2.len() != 1 {
                memo.insert(v, None);
                return None;
            }
            let (_p0, llm2, dests) = next;
            let (w, sids2) = &dests[0];

            // Recurse forward
            let res = if let Some(tail) = follow_zero_chain(*w, summary, memo) {
                Some(ChainRes {
                    last: tail.last,
                    llm: llm2 & &tail.llm,
                    sids: sids2 & &tail.sids,
                })
            } else {
                Some(ChainRes {
                    last: *w,
                    llm: llm2.clone(),
                    sids: sids2.clone(),
                })
            };
            memo.insert(v, res.clone());
            res
        }

        let mut changed_any = false;

        #[cfg(not(rustrover))]
        let it = tqdm!(nodes.iter(), desc = "Trie3 Compress (Pop-0 Chains)", total=nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave=false);
        #[cfg(rustrover)]
        let it = nodes.iter();
        for u in it {
            // Snapshot children (stable during this node's rewrite)
            let children_snapshot: Vec<((usize, LLMTokenBV), Vec<(Trie2Index, StateIDBV)>)> = {
                let g = u.read(trie3_god).expect("read");
                g.children()
                    .iter()
                    .map(|(ek, dm)| {
                        let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<Vec<_>>();
                        (ek.clone(), dests)
                    })
                    .collect()
            };
            if children_snapshot.is_empty() { continue; }

            let mut local_changed = false;
            let mut w = u.write(trie3_god).expect("write");

            for ((p1, llm1), dests) in &children_snapshot {
                // We will remove/replace individual destinations for this key.
                for (v, sids1) in dests {
                    if let Some(chain) = follow_zero_chain(*v, &summary, &mut memo) {
                        // Compose new filters
                        let new_llm = llm1 & &chain.llm;
                        let new_sids = sids1 & &chain.sids;

                        // Remove old edge U --(p1, llm1)--> V
                        if let Some(dm) = w.children_mut().get_mut(&(p1.clone(), llm1.clone())) {
                            if dm.remove(v).is_some() {
                                local_changed = true;
                            }
                            if dm.is_empty() {
                                w.children_mut().remove(&(p1.clone(), llm1.clone()));
                            }
                        }

                        // If empty, nothing to add; drop the path.
                        if new_llm.is_empty() || new_sids.is_empty() {
                            continue;
                        }

                        // Insert U --(p1, new_llm)--> chain.last with new_sids
                        let dest_map = w.children_mut().entry((*p1, new_llm)).or_default();
                        dest_map.entry(chain.last)
                            .and_modify(|s| *s |= &new_sids)
                            .or_insert(new_sids);
                    }
                }
            }

            if local_changed {
                // Keep node metadata consistent after rewrites
                let mut union_bv = LLMTokenBV::zeros();
                for ((_, llm_bv), _) in w.children().iter() {
                    union_bv |= llm_bv;
                }
                w.value.live_tokens = union_bv;
                changed_any = true;
            }
        }

        changed_any
    };

    // Pass 3: shortcut when the first edge is "universal" and the middle has a single outgoing edge.
    // A --(p1, ALL_LLM, ALL_SID)--> B and B --(p2, L2, SID2)--> C (only outgoing)
    // becomes A --(p1+p2, L2, SID2)--> C. (Do not apply when p2 == 0; zero-pop handled by pass 2.)
    let shortcut_universal_pop_step = |trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]| -> bool {
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        if nodes.is_empty() { return false; }

        // Summaries
        type DestList = Vec<(Trie2Index, StateIDBV)>;
        type EdgeList = Vec<(usize, LLMTokenBV, DestList)>;
        let mut summary: HashMap<Trie2Index, (bool, EdgeList)> = HashMap::new();
        for n in &nodes {
            let g = n.read(trie3_god).expect("read");
            let edges: EdgeList = g.children()
                .iter()
                .map(|(ek, dm)| {
                    let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<DestList>();
                    (ek.0, ek.1.clone(), dests)
                })
                .collect();
            summary.insert(*n, (g.value.end, edges));
        }

        // Identify "compressible" middle nodes: exactly one outgoing edge, with exactly one destination, pop > 0
        let mut middle_info: HashMap<Trie2Index, (usize, LLMTokenBV, Trie2Index, StateIDBV)> = HashMap::new();
        for n in &nodes {
            let (is_end, edges) = summary.get(n).unwrap();
            if *is_end { continue; }
            if edges.len() != 1 { continue; }
            let (p2, llm2, dests) = &edges[0];
            if *p2 == 0 { continue; } // leave to zero-pop pass
            if dests.len() != 1 { continue; }
            let (c, sids2) = &dests[0];
            middle_info.insert(*n, (*p2, llm2.clone(), *c, sids2.clone()));
        }

        let mut changed_any = false;

        #[cfg(not(rustrover))]
        let it = tqdm!(nodes.iter(), desc = "Trie3 Compress (Universal Pop)", total=nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave=false);
        #[cfg(rustrover)]
        let it = nodes.iter();
        for u in it {
            // Snapshot children
            let children_snapshot: Vec<((usize, LLMTokenBV), Vec<(Trie2Index, StateIDBV)>)> = {
                let g = u.read(trie3_god).expect("read");
                g.children()
                    .iter()
                    .map(|(ek, dm)| {
                        let dests = dm.iter().map(|(d, sids)| (*d, sids.clone())).collect::<Vec<_>>();
                        (ek.clone(), dests)
                    })
                    .collect()
            };
            if children_snapshot.is_empty() { continue; }

            let mut local_changed = false;
            let mut w = u.write(trie3_god).expect("write");

            for ((p1, llm1), dests) in &children_snapshot {
                // Only when the first edge is universal in both LLM and SIDs.
                if !is_all_llm(&llm1) {
                    continue;
                }
                for (v, sids1) in dests {
                    if !is_all_sids(&sids1) {
                        continue;
                    }
                    if let Some((p2, llm2, c, sids2)) = middle_info.get(v).cloned() {
                        // Remove old edge U --(p1, llm1)--> V
                        if let Some(dm) = w.children_mut().get_mut(&(p1.clone(), llm1.clone())) {
                            if dm.remove(v).is_some() {
                                local_changed = true;
                            }
                            if dm.is_empty() {
                                w.children_mut().remove(&(p1.clone(), llm1.clone()));
                            }
                        }
                        // Insert U --(p1+p2, llm2)--> C with sids2
                        let key_new = (p1 + p2, llm2);
                        let dest_map = w.children_mut().entry(key_new).or_default();
                        dest_map.entry(c)
                            .and_modify(|s| *s |= &sids2)
                            .or_insert(sids2);
                    }
                }
            }

            if local_changed {
                // Keep node metadata consistent after rewrites
                let mut union_bv = LLMTokenBV::zeros();
                for ((_, llm_bv), _) in w.children().iter() {
                    union_bv |= llm_bv;
                }
                w.value.live_tokens = union_bv;
                changed_any = true;
            }
        }

        changed_any
    };

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if Trie::all_nodes(trie3_god, &roots_vec).is_empty() {
        return;
    }

    // Iterate to a (small) fixpoint so that local changes enable further opportunities.
    const MAX_PASSES: usize = 4;
    let mut any_changed = false;
    for pass in 0..MAX_PASSES {
        let mut pass_changed = false;
        // 1) Coalesce within nodes (cheap win)
        if coalesce_edges_within_nodes(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 2) Shortcut pop=0 chains (safe, non-expanding)
        if shortcut_zero_pop_chains(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        // 3) Shortcut universal-first edges by adding pops (safe, non-expanding)
        if shortcut_universal_pop_step(trie3_god, &roots_vec) {
            pass_changed = true;
        }
        if pass_changed {
            any_changed = true;
            crate::debug!(3, "compress_trie3_edges: pass {} applied changes", pass + 1);
        } else {
            break;
        }
    }

    if any_changed {
        crate::debug!(2, "compress_trie3_edges: changes applied; prune/merge/gc will follow in optimize_trie3_size");
    } else {
        crate::debug!(2, "compress_trie3_edges: no changes");
    }
}
