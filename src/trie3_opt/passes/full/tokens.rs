use std::collections::{BTreeMap, HashMap, HashSet};

use ordered_hash_map::OrderedHashMap;
use range_set_blaze::RangeSetBlaze;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StageVocab, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::Trie;
use crate::profiler::PROGRESS_BAR_ENABLED;

/// Remap helper: many-to-one mapping of internal LLM tokens by merging equivalent tokens.
/// This remaps bitvectors to representative token ids, respecting max_token_id universe.
pub fn remap_llm_bv_many_to_one(
    bv: &LLMTokenBV,
    map_old_to_new: &BTreeMap<usize, usize>,
    max_token_id: usize,
) -> LLMTokenBV {
    if bv.is_empty() {
        return LLMTokenBV::zeros();
    }

    let mut ranges_to_add = Vec::new();
    let mut elements_to_add = Vec::new();

    let process_range =
        |range: std::ops::RangeInclusive<usize>,
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

    LLMTokenBV {
        inner: crate::datastructures::cache::intern_l1(new_set),
    }
}

/// Remap helper under a permutation mapping (one-to-one). Unseen tokens stay as identity.
pub fn remap_llm_bv_permutation(
    bv: &LLMTokenBV,
    map_old_to_new: &BTreeMap<usize, usize>,
    _max_token_id: usize,
) -> LLMTokenBV {
    // Fast‑paths for empty or full‑universe bitvectors.
    if bv.is_empty() {
        return LLMTokenBV::zeros();
    }
    if *bv == LLMTokenBV::max_ones() {
        return LLMTokenBV::max_ones();
    }

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

    LLMTokenBV {
        inner: crate::datastructures::cache::intern_l1(new_set),
    }
}

/// Merge equivalent internal LLM token ids in Trie3:
/// Two tokens are equivalent if they occur together in every LLMTokenBV occurrence across:
/// - node.value.live_tokens
/// - each edge key's (pop, LLMTokenBV) mask
///
/// Applies many-to-one mapping into representative ids, remapping node/edge masks,
/// and updates the provided StageVocab.
pub fn merge_equivalent_llm_tokens_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    crate::debug!(2, "Merging equivalent LLM tokens in Trie3...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // 1) Collect all unique bitsets to use as splitters.
    let mut all_bvs = HashSet::new();
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        all_nodes.iter(),
        desc = "Trie3 Merge Tokens (Collect BVs)",
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
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
    if all_bvs.is_empty() {
        return;
    }

    // 2) Partition refinement.
    let max_tok = stage_vocab.internal_max_llm_token;
    let mut token_to_class: Vec<usize> = vec![0; max_tok + 1];
    let mut class_to_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
    class_to_tokens.insert(0, (0..=max_tok).collect());
    let mut num_classes = 1;

    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        all_bvs.iter(),
        desc = "Trie3 Merge Tokens (Refine)",
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = all_bvs.iter();
    for splitter_bv in it {
        if *splitter_bv == LLMTokenBV::max_ones() {
            continue;
        }

        let mut members_in_splitter_by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for token in splitter_bv.iter() {
            if token <= max_tok {
                let class_id = token_to_class[token];
                members_in_splitter_by_class
                    .entry(class_id)
                    .or_default()
                    .push(token);
            }
        }

        for (old_class_id, tokens_for_new_class) in members_in_splitter_by_class {
            let old_class_size = class_to_tokens.get(&old_class_id).map_or(0, |v| v.len());
            if old_class_size == 0 {
                continue;
            }

            if !tokens_for_new_class.is_empty() && tokens_for_new_class.len() < old_class_size {
                let new_class_id = num_classes;
                num_classes += 1;

                for &token in &tokens_for_new_class {
                    token_to_class[token] = new_class_id;
                }

                let old_class_tokens = class_to_tokens.get_mut(&old_class_id).unwrap();
                let moved_tokens_set: HashSet<_> =
                    tokens_for_new_class.iter().cloned().collect();
                old_class_tokens.retain(|t| !moved_tokens_set.contains(t));

                class_to_tokens.insert(new_class_id, tokens_for_new_class);
            }
        }
    }

    // 3) Build many-to-one mapping from the final partition.
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    let mut merged_count = 0;
    for (_class_id, group) in &class_to_tokens {
        if group.len() <= 1 {
            continue;
        }
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
    crate::debug!(
        2,
        "Trie3: merged LLM tokens. Before: {}, After: {}. ({} merged)",
        tokens_before,
        tokens_after,
        merged_count
    );
    if merged_count == 0 {
        return;
    }

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
        affected_ptrs.insert(std::sync::Arc::as_ptr(&splitter.inner));
    }

    // 4) Remap trie in‑place, only where needed
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        all_nodes.iter(),
        desc = "Trie3 Merge (Remap In‑Place)",
        total = all_nodes.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        // Quick check whether this node references any affected bitvector.
        let needs_update = {
            let r = n.read(trie3_god).expect("read");
            let lv_ptr = std::sync::Arc::as_ptr(&r.value.live_tokens.inner);
            if affected_ptrs.contains(&lv_ptr) {
                true
            } else {
                let mut touched = false;
                for ((_, llm_bv), _dm) in r.children() {
                    let key_ptr = std::sync::Arc::as_ptr(&llm_bv.inner);
                    if affected_ptrs.contains(&key_ptr) {
                        touched = true;
                        break;
                    }
                }
                touched
            }
        };
        if !needs_update {
            continue;
        }

        let mut w = n.write(trie3_god).expect("write");

        // Remap edge keys (pop, LLMTokenBV)
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<
            (isize, LLMTokenBV),
            OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
        > = BTreeMap::new();
        for ((pop, llm_bv), dm) in old_children {
            let mapped_key_bv = if llm_bv.is_empty() {
                LLMTokenBV::zeros()
            } else if llm_bv == LLMTokenBV::max_ones() {
                mapped_universe.clone()
            } else {
                let key_ptr = std::sync::Arc::as_ptr(&llm_bv.inner);
                if affected_ptrs.contains(&key_ptr) {
                    memo.entry(llm_bv)
                        .or_insert_with_key(|bv| remap_llm_bv_many_to_one(bv, &old_to_new, max_tok))
                        .clone()
                } else {
                    llm_bv
                }
            };
            if mapped_key_bv.is_empty() {
                continue;
            }
            let entry = new_children
                .entry((pop, mapped_key_bv))
                .or_insert_with(OrderedHashMap::new);
            for (dst, sid_bv) in dm {
                entry
                    .entry(dst)
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
    let it = kdam::tqdm!(
        old_to_new.iter(),
        desc = "Trie3 Merge (Update Vocab)",
        total = old_to_new.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = old_to_new.iter();
    for (old, rep) in it {
        if old == rep {
            continue;
        }
        if let Some(moved) = stage_vocab.internal_to_original.remove(old) {
            let entry = stage_vocab.internal_to_original.entry(*rep).or_default();
            *entry |= &moved;
            for o in moved.iter() {
                stage_vocab.original_to_internal.insert(o, *rep);
            }
        }
    }
}
