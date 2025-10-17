use std::collections::BTreeSet;
use std::collections::{BTreeMap, HashMap, VecDeque, HashSet};
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::time::Instant;
use range_set_blaze::RangeSetBlaze;
use indicatif::{ProgressBar, ProgressStyle};
use kdam::tqdm;
use ordered_hash_map::OrderedHashMap;
use crate::constraint::{PrecomputeNode3Index, StateIDBV, Trie3GodWrapper, StageVocab, PrecomputedNodeContents};
use crate::constraint_extra::{calculate_final_stats3, print_precompute_stats3, PrecomputeStats};
use crate::datastructures::EntryApi;
use crate::constraint::LLMTokenBV;
use crate::datastructures::trie::{EdgeInserter, Trie, Trie2Index};
use crate::tokenizer::TokenizerStateID;

use crate::profiler::PROGRESS_BAR_ENABLED;

#[derive(Debug, Clone)]
pub struct Trie3MergeConfig {
    pub enabled: bool,
    pub exact_max_iters: usize,
}

impl Default for Trie3MergeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exact_max_iters: 40,
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

#[derive(Debug, Clone)]
pub struct Trie3Config {
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
    pub simplify_llm_token_bvs: bool,
    pub factor_common_destinations: bool,
}

impl Default for Trie3Config {
    fn default() -> Self {
        Self {
            enabled: true,
            num_passes: 1,
            merge_equivalent_llm_tokens: true,
            reorder_llm_tokens: true,
            constrain_bitvecs: true,
            gc: true,
            prune_dead_paths: true,
            compress_edges: false,
            merge_nodes_exact: Trie3MergeConfig::default(),
            merge_nodes_structural: true,
            merge_nodes_ultrafast: false,
            prune_nodes_not_reaching_end: true,
            simplify_llm_token_bvs: false,
            factor_common_destinations: true,
        }
    }
}

impl Trie3Config {
    pub fn off() -> Self {
        Self {
            enabled: false,
            num_passes: 0,
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
            simplify_llm_token_bvs: false,
            factor_common_destinations: false,
        }
    }
}

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
    config: &Trie3Config,
    max_state_id: usize,
    mut max_llm_token_id: usize,
    stage_vocab: &mut StageVocab,
) {
	if !config.enabled {
		return;
	}
	crate::debug!(2, "Optimizing Trie 3 size...");

	crate::debug!(2, "Initial stats:");
	compute_and_print_precompute_stats3(roots, trie3_god);

	for pass_num in 0..config.num_passes {
        if config.num_passes > 1 {
            crate::debug!(2, "--- Starting optimization super-pass {}/{} ---", pass_num + 1, config.num_passes);
        }

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
                prune_dead_paths_trie3(roots, &trie3_god);
            });
        }

        if config.prune_nodes_not_reaching_end {
            run_pass!("Pruning nodes that do not reach end", {
                prune_nodes_not_reaching_end_trie3(roots, &trie3_god);
            });
        }

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

        // --- Phase 2: Structural Compression and Merging ---
        // Now that the graph is smaller and token sets are simpler, we can apply
        // heavy structural optimizations.

        if config.simplify_llm_token_bvs {
            run_pass!("Simplifying LLM token bitsets", {
                simplify_llm_token_bvs_trie3(roots, &trie3_god, max_llm_token_id);
            });
        }

        if config.factor_common_destinations {
            run_pass!("Factoring common destinations", {
                factor_common_destinations_trie3(roots, trie3_god, max_llm_token_id, max_state_id);
            });
        }

        if config.compress_edges {
            run_pass!("Compressing edges", {
                compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
            });
        }

        // After compression, prune and GC before the expensive merge.
        if config.prune_dead_paths {
            run_pass!("Pruning dead paths (post-compress)", {
                prune_dead_paths_trie3(roots, &trie3_god);
            });
        }
        if config.gc {
            run_pass!("Garbage collection (pre-merge)", {
                Trie::gc(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
            });
        }

        if config.merge_nodes_structural {
            run_pass!("Merging nodes (structural)", {
                merge_nodes_trie3_structural(roots, &trie3_god, config.merge_nodes_exact.exact_max_iters);
            });
            // Structural merge can create parallel edges (same dest/pop/sids, diff tokens).
            // Compress them immediately.
            if config.compress_edges {
                run_pass!("Compressing edges (post-structural)", {
                    compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
                });
            }
        }

        if config.merge_nodes_exact.enabled {
            run_pass!("Merging nodes", {
                merge_nodes_trie3(roots, &trie3_god, config.merge_nodes_exact.exact_max_iters);
            });
        }

        // --- Phase 3: Iterative Refinement ---
        // A few rounds of compression and merging on the now much smaller graph.

        if config.prune_nodes_not_reaching_end {
            run_pass!("Pruning nodes that do not reach end (post-merge)", {
                prune_nodes_not_reaching_end_trie3(roots, &trie3_god);
            });
        }

        if config.compress_edges {
            run_pass!("Compressing edges (post-merge)", {
                compress_trie3_edges(roots, &trie3_god, max_llm_token_id, max_state_id);
            });
        }

        if config.merge_nodes_exact.enabled {
            run_pass!("Merging nodes (post-compress)", {
                merge_nodes_trie3(roots, &trie3_god, config.merge_nodes_exact.exact_max_iters);
            });
        }

        // --- Phase 4: Final Cleanup and Polish ---

        if config.prune_dead_paths {
            run_pass!("Pruning dead paths (final)", {
                prune_dead_paths_trie3(roots, &trie3_god);
            });
        }

        if config.gc {
            run_pass!("Garbage collection (final)", {
                Trie::gc(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());
            });
        }

        if config.merge_equivalent_llm_tokens {
            run_pass!("Merging equivalent LLM tokens (final pass)", {
                merge_equivalent_llm_tokens_trie3(roots, trie3_god, stage_vocab);
            });
        }
        if config.reorder_llm_tokens {
            run_pass!("Reordering LLM tokens (final pass)", {
                reorder_llm_tokens_for_range_minimization_trie3(roots, trie3_god, stage_vocab);
            });
        }
    }

	crate::debug!(2, "Recomputing max depths...");
    Trie::recompute_all_max_depths(&trie3_god, &roots.values().cloned().collect::<Vec<_>>());

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

        // Remap edge keys (pop, LLMTokenBV)
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<PrecomputeNode3Index, StateIDBV>> = BTreeMap::new();
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

		// Recompute live tokens from the new children
		let mut new_live = LLMTokenBV::zeros();
		for ((_, llm_bv), _) in w.children() {
			new_live |= llm_bv;
		}
		w.value.live_tokens = new_live;
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
        new_states.push(new_children);
    }
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter().enumerate(), desc = "Trie3 Reorder (Remap Write)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter().enumerate();
    for (i, n) in it {
        let mut w = n.write(trie3_god).expect("write");
        let children = &new_states[i];
        let mut new_live = LLMTokenBV::zeros();
        for ((_, llm_bv), _) in children {
            new_live |= llm_bv;
        }
        w.value.live_tokens = new_live;
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

        // Recompute live_tokens on the fly to ensure it's accurate for this pass.
        let live_u = {
            let mut u = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in w.children() { u |= llm_bv; }
            u
        };
        if live_u.is_all() { // If all tokens are live, no simplification is possible.
            continue;
        }
        let dead_u = &universe - &live_u;

        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<PrecomputeNode3Index, StateIDBV>> = BTreeMap::new();

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
        let mut new_children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();
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
            (isize, LLMTokenBV),
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
                        // Recompute live tokens from scratch after modifying edges.
                        let mut new_live = LLMTokenBV::zeros();
                        for ((_, llm_bv), _) in src_guard.children() {
                            new_live |= llm_bv;
                        }
                        src_guard.value.live_tokens = new_live;
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

    let mut predecessors: HashMap<PrecomputeNode3Index, Vec<(PrecomputeNode3Index, (isize, LLMTokenBV))>> = HashMap::new();
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

    #[cfg(not(rustrover))]
    let pb = {
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
        pb
    };
    #[cfg(rustrover)]
    let pb = ProgressBar::hidden();

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
        let mut new_children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();

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

pub fn merge_nodes_trie3(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper, max_iters: usize) {
    merge_nodes_trie3_impl(roots, trie3_god, max_iters);
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
    type RawEdge3 = (isize, LLMTokenBV, usize, StateIDBV);
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
        type AggregatedEdge3 = ((isize, LLMTokenBV, usize), StateIDBV);
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
            let mut aggr: BTreeMap<(isize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
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

            let mut aggr: BTreeMap<(isize, LLMTokenBV, usize), StateIDBV> = BTreeMap::new();
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

            // Recompute live tokens from the new merged edges.
            let mut new_live_tokens = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in &new_children {
                new_live_tokens |= llm_bv;
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

fn merge_nodes_trie3_structural(roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>, trie3_god: &Trie3GodWrapper, max_iters: usize) {
    crate::debug!(2, "Merging structurally equivalent subtrees in precomputed trie 3 (max_iters={}).", max_iters);

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
    type RawEdge3 = (isize, LLMTokenBV, usize, StateIDBV);
    let mut raw_edges: Vec<Vec<RawEdge3>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let guard = u_idx.read(trie3_god).unwrap();
        ends[u_dense] = guard.value.end;
        for (ek, dest_map) in guard.children() {
            for (v_idx, bv) in dest_map {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    // Note: we capture the LLMTokenBV here but won't use it in the signature.
                    raw_edges[u_dense].push((ek.0, ek.1.clone(), v_dense, bv.clone()));
                }
            }
        }
    }

    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    for it in 0..max_iters {
        // Signature is (end_flag, map<(pop, dest_class) -> union of StateIDBVs>)
        // This is more aggressive than using a BTreeSet of SIDs, as it allows merging
        // nodes that have different SID distributions as long as the total set of states
        // required to reach a destination class is the same.
        type SignatureStructural3 = (bool, BTreeMap<(isize, usize), StateIDBV>);

        let mut sig_to_id: HashMap<SignatureStructural3, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        #[cfg(not(rustrover))]
        let its = tqdm!(0..n, desc = format!("Trie3 Merge Structural Iter {}", it + 1), total = n, disable = !PROGRESS_BAR_ENABLED, leave = true);
        #[cfg(rustrover)]
        let its = 0..n;
        for u in its {
            let mut aggr: BTreeMap<(isize, usize), StateIDBV> = BTreeMap::new();
            for (p, _bv_key, v_dense, sids) in &raw_edges[u] {
                // _bv_key (LLM tokens) is IGNORED for structural equivalence.
                let dest_class = prev_class[*v_dense];
                let key = (*p, dest_class);
                *aggr.entry(key).or_default() |= sids;
            }

            let sig: SignatureStructural3 = (ends[u], aggr);

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

        crate::debug!(3, "Trie3 structural merge iter {}: classes={}, changes={}", it + 1, next_id, changes);
        prev_class = new_class;
        if changes == 0 { break; }
    }

    // Reconstruction
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
            // Gather all nodes belonging to this class
            let nodes_in_class: Vec<Trie2Index> = final_partition.iter().enumerate()
                .filter(|(_, &c)| c == class_id)
                .map(|(i, _)| old_of[i])
                .collect();

            let mut new_children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();

            // Merge all edges from all nodes in the class into the representative.
            for node_idx in &nodes_in_class {
                let guard = node_idx.read(trie3_god).unwrap();

                for ((pop, llm_bv), dest_map) in guard.children() {
                    for (dest_idx, sids) in dest_map {
                        // Remap destination to its representative
                        let dest_rep_idx = *node_to_rep.get(dest_idx).unwrap();

                        // Insert into new_children.
                        // Note: This may create parallel edges (same pop/dest_rep/sids, different llm_bv)
                        // if the original nodes had them. Subsequent edge compression will fix this.
                        let new_dest_map = new_children.entry((*pop, llm_bv.clone())).or_insert_with(OrderedHashMap::new);
                        new_dest_map.entry(dest_rep_idx)
                            .and_modify(|e| *e |= sids) // Union SIDs if exact edge exists
                            .or_insert_with(|| sids.clone());
                    }
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

/// Extremely fast, cycle-safe node merging using WL-style refinement with a cheap signature
/// (ignoring StateIDBV during coarse iterations) followed by a single exact refinement within
/// candidate equivalence classes that compares aggregated SIDs exactly. Finally, only representatives
/// are rewritten to point to representative destinations; later GC/pruning removes non-reps.
pub fn merge_nodes_trie3_ultrafast(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "Merging nodes (ultrafast WL + exact refine) in precomputed trie 3.");

    // Collect all nodes reachable from roots
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    // Build dense index for nodes
    let n = all_nodes.len();
    let mut dense_of: HashMap<Trie2Index, u32> = HashMap::with_capacity(n.checked_mul(2).unwrap_or(n));
    let mut old_of: Vec<Trie2Index> = Vec::with_capacity(n);
    for (i, node_idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*node_idx, i as u32);
        old_of.push(*node_idx);
    }

    // First pass: count edges per node and capture ends + live token ptrs
    let mut out_counts: Vec<usize> = vec![0; n];
    let mut ends: Vec<u8> = vec![0; n];
    // Store LLM live-token pointer addresses to assign ids later
    let mut live_llm_ptrs: Vec<*const RangeSetBlaze<usize>> = vec![std::ptr::null(); n];
    #[cfg(not(rustrover))]
    let it = tqdm!(old_of.iter().enumerate(), desc = "Trie3 Merge Ultra (Pass1 count)", total = n, disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = old_of.iter().enumerate();
    for (i, node_idx) in it {
        let g = node_idx.read(trie3_god).expect("read");
        ends[i] = if g.value.end { 1 } else { 0 };
        live_llm_ptrs[i] = Arc::as_ptr(&g.value.live_tokens.inner);

        let mut cnt = 0usize;
        for (_ek, dm) in g.children() {
            cnt += dm.len();
        }
        out_counts[i] = cnt;
    }

    // Build offsets for flat edge storage
    let mut offsets: Vec<usize> = vec![0; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + out_counts[i];
    }
    let m = offsets[n];

    // LLM bitset pointer-id map
    let mut llm_id_map: HashMap<*const RangeSetBlaze<usize>, u32> = HashMap::new();
    let mut next_llm_id: u32 = 0;
    let mut get_or_insert_llm_id = |ptr: *const RangeSetBlaze<usize>| -> u32 {
        if let Some(id) = llm_id_map.get(&ptr) {
            *id
        } else {
            let id = next_llm_id;
            next_llm_id = next_llm_id.wrapping_add(1);
            llm_id_map.insert(ptr, id);
            id
        }
    };

    // Assign live token ids
    let mut live_llm_id: Vec<u32> = vec![0; n];
    for i in 0..n {
        live_llm_id[i] = get_or_insert_llm_id(live_llm_ptrs[i]);
    }

    // Flat edge structure: pop, llm_id, dest_dense
    #[derive(Copy, Clone)]
    struct EdgeLight {
        pop: u32,
        llm_id: u32,
        dest: u32,
    }
    let mut edges: Vec<EdgeLight> = vec![EdgeLight { pop: 0, llm_id: 0, dest: 0 }; m];

    // Second pass: fill flat edges
    #[cfg(not(rustrover))]
    let it = tqdm!(old_of.iter().enumerate(), desc = "Trie3 Merge Ultra (Pass2 fill)", total = n, disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = old_of.iter().enumerate();
    for (i, node_idx) in it {
        let g = node_idx.read(trie3_god).expect("read");
        let mut p = offsets[i];
        for (ek, dm) in g.children() {
            let pop_u32 = ek.0 as u32;
            let llm_ptr = Arc::as_ptr(&ek.1.inner);
            let llm_id = get_or_insert_llm_id(llm_ptr);
            for (dst, _sids) in dm {
                let dest_dense = *dense_of.get(dst).expect("dense id") as u32;
                edges[p] = EdgeLight { pop: pop_u32, llm_id, dest: dest_dense };
                p += 1;
            }
        }
        debug_assert_eq!(p, offsets[i + 1]);
    }

    // Coarse WL refinement ignoring SIDs
    let max_iters_coarse: usize = 6; // small number of iterations for speed
    let mut prev_class: Vec<u32> = vec![0; n];
    // Initialize coarse classes by (end flag, degree, live_llm_id)
    {
        let mut init_map: HashMap<(u8, u32, usize), u32> = HashMap::with_capacity(n.checked_div(2).unwrap_or(1024));
        let mut next_c: u32 = 0;
        for i in 0..n {
            let deg = offsets[i + 1] - offsets[i];
            let key = (ends[i], live_llm_id[i], deg);
            let c = init_map.entry(key).or_insert_with(|| { let id = next_c; next_c = next_c.wrapping_add(1); id });
            prev_class[i] = *c;
        }
    }

    // Workspace vectors to avoid reallocations
    let mut tmp_items: Vec<(u32, u32, u32)> = Vec::with_capacity(16); // (pop, llm_id, dest_class)
    let mut agg_items: Vec<((u32, u32, u32), u32)> = Vec::with_capacity(16); // ((pop,llm,dest_class), count)
    let mut new_class: Vec<u32> = vec![0; n];
    for it in 0..max_iters_coarse {
        // Phase: compute coarse signature hashes per node
        // We'll use a simple FNV-1a over a sorted aggregated vector
        let mut h_of: Vec<u64> = vec![0; n];
        #[cfg(not(rustrover))]
        let itn = tqdm!(0..n, desc = format!("Trie3 Merge Ultra (WL coarse {} / {})", it + 1, max_iters_coarse), total = n, disable = !PROGRESS_BAR_ENABLED, leave = true);
        #[cfg(rustrover)]
        let itn = 0..n;
        for u in itn {
            tmp_items.clear();
            let begin = offsets[u];
            let end = offsets[u + 1];
            for idx in begin..end {
                let e = edges[idx];
                let dclass = prev_class[e.dest as usize];
                tmp_items.push((e.pop, e.llm_id, dclass));
            }
            if tmp_items.len() > 1 {
                tmp_items.sort_unstable();
            }
            agg_items.clear();
            let mut i2 = 0usize;
            while i2 < tmp_items.len() {
                let key = tmp_items[i2];
                let mut cnt: u32 = 1;
                i2 += 1;
                while i2 < tmp_items.len() && tmp_items[i2] == key {
                    cnt = cnt.wrapping_add(1);
                    i2 += 1;
                }
                agg_items.push((key, cnt));
            }
            // FNV-1a hash
            let mut h: u64 = 0xcbf29ce484222325;
            // end flag
            h ^= ends[u] as u64;
            h = h.wrapping_mul(0x100000001b3);
            // live_llm_id
            h ^= live_llm_id[u] as u64;
            h = h.wrapping_mul(0x100000001b3);
            // length
            h ^= agg_items.len() as u64;
            h = h.wrapping_mul(0x100000001b3);
            for &((p, lid, dc), cnt) in &agg_items {
                h ^= p as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= lid as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= dc as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= cnt as u64; h = h.wrapping_mul(0x100000001b3);
            }
            h_of[u] = h;
        }

        // Compress hashes to new classes
        let mut map: HashMap<u64, u32> = HashMap::with_capacity(n.checked_div(2).unwrap_or(1024));
        let mut next_c: u32 = 0;
        let mut changes = 0usize;
        for u in 0..n {
            let cid = *map.entry(h_of[u]).or_insert_with(|| { let id = next_c; next_c = next_c.wrapping_add(1); id });
            new_class[u] = cid;
            if new_class[u] != prev_class[u] {
                changes += 1;
            }
        }
        crate::debug!(3, "Trie3 ultrafast coarse iter {}: classes={}, changes={}", it + 1, map.len(), changes);
        prev_class = new_class;
        new_class = vec![0; n];
        if changes == 0 { break; }
    }

    // Now do an exact refinement within each coarse class by aggregating SIDs exactly.
    // Build membership list as sorted pairs (class, node)
    let mut membership: Vec<(u32, u32)> = (0..n as u32).map(|u| (prev_class[u as usize], u)).collect();
    membership.sort_unstable_by_key(|x| x.0);

    // We'll produce a node->representative map (dense ids)
    let mut node_to_rep_dense: Vec<u32> = (0..n as u32).collect();

    #[derive(Hash, Eq, PartialEq, Clone)]
    struct KeyNoSids {
        end: u8,
        live: u32,
        // Sorted vector of (pop, llm_id, dest_class) triples
        edges: Vec<(u32, u32, u32)>,
    }
    struct Prototype {
        sids: Vec<StateIDBV>, // aligned with edges order
        rep_dense: u32,
    }

    // Iterate groups
    let mut i = 0usize;
    #[cfg(not(rustrover))]
    let total_groups = membership.len();
    #[cfg(not(rustrover))]
    let mut processed_nodes = 0usize;
    while i < membership.len() {
        let class_id = membership[i].0;
        let start = i;
        while i < membership.len() && membership[i].0 == class_id {
            i += 1;
        }
        let end_span = i;
        let span_len = end_span - start;
        if span_len <= 1 {
            // Single node - it is its own representative
            let u_dense = membership[start].1;
            node_to_rep_dense[u_dense as usize] = u_dense;
            #[cfg(not(rustrover))]
            { processed_nodes += 1; }
            continue;
        }

        // Build prototype map keyed by (end, live, keys_without_sids)
        let mut key_map: HashMap<KeyNoSids, Vec<Prototype>> = HashMap::new();
        #[cfg(not(rustrover))]
        let it = tqdm!(start..end_span, desc = "Trie3 Merge Ultra (Exact refine group)", disable = !PROGRESS_BAR_ENABLED, leave = false);
        #[cfg(rustrover)]
        let it = start..end_span;
        for idx in it {
            let u_dense = membership[idx].1 as usize;
            let node_idx = old_of[u_dense];
            let g = node_idx.read(trie3_god).expect("read");

            // Aggregate SIDs per (pop, llm_id, coarse_dest_class)
            let mut aggr: BTreeMap<(u32, u32, u32), StateIDBV> = BTreeMap::new();
            for (ek, dm) in g.children() {
                let pop_u32 = ek.0 as u32;
                let llm_ptr = Arc::as_ptr(&ek.1.inner);
                let llm_id = *llm_id_map.get(&llm_ptr).expect("llm_id present");
                for (dst, sids) in dm {
                    let dest_dense = *dense_of.get(dst).expect("dense of dst") as usize;
                    let coarse_dest_class = prev_class[dest_dense];
                    aggr.entry((pop_u32, llm_id, coarse_dest_class))
                        .and_modify(|v| *v |= sids)
                        .or_insert_with(|| sids.clone());
                }
            }

            // Build key without sids
            let mut keys_vec: Vec<(u32, u32, u32)> = aggr.keys().cloned().collect();
            // BTreeMap iteration is already sorted; we can rely on that
            let key = KeyNoSids {
                end: ends[u_dense],
                live: live_llm_id[u_dense],
                edges: keys_vec.clone(),
            };
            let sids_vec: Vec<StateIDBV> = aggr.into_values().collect();

            // Try to match an existing prototype
            let entry = key_map.entry(key).or_insert_with(Vec::new);
            let mut found = None;
            for proto in entry.iter() {
                if proto.sids.len() != sids_vec.len() { continue; }
                let mut ok = true;
                for (a, b) in proto.sids.iter().zip(sids_vec.iter()) {
                    if a != b { ok = false; break; }
                }
                if ok {
                    found = Some(proto.rep_dense);
                    break;
                }
            }
            if let Some(rep) = found {
                node_to_rep_dense[u_dense] = rep;
            } else {
                // New prototype
                let rep = u_dense as u32;
                node_to_rep_dense[u_dense] = rep;
                entry.push(Prototype { sids: sids_vec, rep_dense: rep });
            }
            #[cfg(not(rustrover))]
            { processed_nodes += 1; }
        }
    }
    #[cfg(not(rustrover))]
    {
        let _ = processed_nodes; // avoid unused warning
    }

    // Representatives (unique)
    let mut rep_set: HashSet<u32> = HashSet::new();
    for (u_dense, &rep) in node_to_rep_dense.iter().enumerate() {
        if u_dense as u32 == rep {
            rep_set.insert(rep);
        }
    }

    // Rewrite representatives' children to point to representatives and recompute live tokens
    #[cfg(not(rustrover))]
    let it = tqdm!(rep_set.iter(), desc = "Trie3 Merge Ultra (Rewrite reps)", total = rep_set.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = rep_set.iter();
    for rep_dense in it {
        let rep_idx = old_of[*rep_dense as usize];
        let mut w = rep_idx.write(trie3_god).expect("write");
        let mut new_children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();

        // Build new children by remapping destinations to their representatives
        for (ek, dm) in w.children().clone() {
            let (pop, llm_bv) = ek;
            let mut dest_map = OrderedHashMap::new();
            for (dst, sids) in dm {
                let dst_dense = *dense_of.get(&dst).expect("dense of dst") as usize;
                let rep_dst_dense = node_to_rep_dense[dst_dense] as usize;
                let rep_dst_idx = old_of[rep_dst_dense];
                dest_map.entry(rep_dst_idx)
                    .and_modify(|v| *v |= &sids)
                    .or_insert(sids);
            }
            if !dest_map.is_empty() {
                new_children.insert((pop, llm_bv), dest_map);
            }
        }

        // Recompute live tokens as union of outgoing LLM masks
        let mut new_live = LLMTokenBV::zeros();
        for ((_, llm_bv), _) in &new_children {
            new_live |= llm_bv;
        }
        *w.children_mut() = new_children;
        w.value.live_tokens = new_live;
    }

    // Remap roots to their representatives
    for root_idx in roots.values_mut() {
        if let Some(dense) = dense_of.get(root_idx) {
            let rep_dense = node_to_rep_dense[*dense as usize] as usize;
            *root_idx = old_of[rep_dense];
        }
    }

    // Finalize
    let final_roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie3_god, &final_roots_vec);
    crate::debug!(2, "Ultrafast merge completed: representatives kept = {}", rep_set.len());
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
                g.children().clone() // BTreeMap<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>>
            };
            if old_children.is_empty() { continue; }

            // Aggregate per (pop, child, sids): union LLM-token BVs
            let mut by_pop: HashMap<isize, Vec<(Trie2Index, StateIDBV, LLMTokenBV)>> = HashMap::new();
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
            let mut new_children: BTreeMap<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> = BTreeMap::new();
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

    // Pass 2: Bypass single-exit nodes.
    // This is a general path-contraction optimization. If a node B is not an end state and
    // has exactly one outgoing edge to a single destination C, any incoming edge to B from A
    // can be rerouted to C, composing the pops and intersecting the constraints.
    // A --(p1, L1, S1)--> B --(p2, L2, S2)--> C  =>  A --(p1+p2, L1&L2, S1&S2)--> C
    // This is a powerful transform that subsumes shortcut_zero_pop_chains and shortcut_universal_pop_step.
    let bypass_single_exit_nodes = |trie3_god: &Trie3GodWrapper, roots_vec: &[PrecomputeNode3Index]| -> bool {
        let nodes = Trie::all_nodes(trie3_god, roots_vec);
        let mut changed_any = false;

        // 1. Identify all single-exit nodes.
        let mut single_exit_info: HashMap<Trie2Index, (isize, LLMTokenBV, Trie2Index, StateIDBV)> = HashMap::new();
        for n in &nodes {
            let g = n.read(trie3_god).expect("read");
            if g.value.end { continue; }
            if g.children().len() == 1 {
                let ((p2, l2), dm) = g.children().iter().next().unwrap();
                if dm.len() == 1 {
                    let (c, s2) = dm.iter().next().unwrap();
                    single_exit_info.insert(*n, (*p2, l2.clone(), *c, s2.clone()));
                }
            }
        }
        if single_exit_info.is_empty() { return false; }

        #[cfg(not(rustrover))]
        let it = tqdm!(nodes.iter(), desc = "Trie3 Compress (Bypass)", total = nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false);
        #[cfg(rustrover)]
        let it = nodes.iter();
        for u in it {
            let old_children = u.read(trie3_god).expect("read").children().clone();
            if old_children.is_empty() { continue; }

            let mut new_children = old_children.clone();
            let mut local_changed = false;

            for ((p1, l1), dm) in &old_children {
                for (v, s1) in dm {
                    if let Some((p2, l2, c, s2)) = single_exit_info.get(v) {
                        local_changed = true;

                        // Remove old edge u -> v from our temporary new_children map
                        let current_dm = new_children.get_mut(&(*p1, l1.clone())).unwrap();
                        current_dm.remove(v);
                        if current_dm.is_empty() {
                            new_children.remove(&(*p1, l1.clone()));
                        }

                        // Add new composed edge u -> c
                        let p_new = p1 + p2;
                        let l_new = l1 & l2;
                        let s_new = s1 & s2;

                        if !l_new.is_empty() && !s_new.is_empty() {
                            new_children.entry((p_new, l_new)).or_default()
                                .entry(*c)
                                .and_modify(|s| *s |= &s_new)
                                .or_insert(s_new.clone());
                        }
                    }
                }
            }

            if local_changed {
                changed_any = true;
                let mut u_guard = u.write(trie3_god).expect("write");
                *u_guard.children_mut() = new_children;
                // Recompute live tokens as they may have changed
                let mut new_live = LLMTokenBV::zeros();
                for ((_, llm_bv), _) in u_guard.children() {
                    new_live |= llm_bv;
                }
                u_guard.value.live_tokens = new_live;
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
        // 2) Bypass single-exit nodes to contract paths.
        if bypass_single_exit_nodes(trie3_god, &roots_vec) {
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
