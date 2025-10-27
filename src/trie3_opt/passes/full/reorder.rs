use std::collections::{BTreeMap, HashMap};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StageVocab, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::Trie;
use crate::trie3_opt::passes::full::stats::count_total_ranges_trie3;
use crate::profiler::PROGRESS_BAR_ENABLED;

/// Reorder internal LLM tokens in Trie3 with a simple heuristic to cluster co-occurring tokens.
pub fn reorder_llm_tokens_for_range_minimization_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    crate::debug!(2, "Reordering LLM tokens in Trie3 for range minimization...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }
    let ranges_before = count_total_ranges_trie3(&all_nodes, trie3_god);

    let max_tok = stage_vocab.internal_max_llm_token;

    // 1. Collect unique BV counts to optimize frequency calculation.
    let mut bv_counts: HashMap<LLMTokenBV, usize> = HashMap::new();
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        all_nodes.iter(),
        desc = "Trie3 Reorder (Collect BVs)",
        total = all_nodes.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
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
    let it = kdam::tqdm!(
        bv_counts.iter(),
        desc = "Trie3 Reorder (Count Frequencies)",
        total = bv_counts.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
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
    if present.is_empty() {
        return;
    }
    present.sort_by_key(|&t| (std::cmp::Reverse(freq[t]), t));

    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    for (new_id, old_id) in present.iter().enumerate() {
        old_to_new.insert(*old_id, new_id);
    }

    // Memoization cache
    let mut memo: HashMap<LLMTokenBV, LLMTokenBV> = HashMap::new();

    let mut new_states = Vec::with_capacity(all_nodes.len());
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        all_nodes.iter(),
        desc = "Trie3 Reorder (Remap Read)",
        total = all_nodes.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let r = n.read(trie3_god).expect("read");
        let mut new_children = BTreeMap::new();
        for ((pop, llm_bv), dm) in r.children() {
            let mapped_key_bv = memo
                .entry(llm_bv.clone())
                .or_insert_with_key(|bv| crate::trie3_opt::passes::full::tokens::remap_llm_bv_permutation(
                    bv,
                    &old_to_new,
                    max_tok,
                ))
                .clone();
            if mapped_key_bv.is_empty() {
                continue;
            }
            let entry = new_children
                .entry((*pop, mapped_key_bv))
                .or_insert_with(OrderedHashMap::new);
            for (dst, sid_bv) in dm {
                entry
                    .entry(dst.clone())
                    .and_modify(|e| *e |= sid_bv)
                    .or_insert_with(|| sid_bv.clone());
            }
        }
        new_states.push(new_children);
    }
    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        all_nodes.iter().enumerate(),
        desc = "Trie3 Reorder (Remap Write)",
        total = all_nodes.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
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
    let it = kdam::tqdm!(
        stage_vocab.internal_to_original.clone().into_iter(),
        desc = "Trie3 Reorder (Vocab 1)",
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
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
    let it = kdam::tqdm!(
        stage_vocab.original_to_internal.clone().into_iter(),
        desc = "Trie3 Reorder (Vocab 2)",
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = stage_vocab.original_to_internal.clone().into_iter();
    for (orig, old_internal) in it {
        if let Some(new_internal) = old_to_new.get(&old_internal) {
            new_original_to_internal.insert(orig, *new_internal);
        }
    }
    stage_vocab.original_to_internal = new_original_to_internal;
    stage_vocab.internal_max_llm_token = present.len().saturating_sub(1);
    crate::debug!(
        2,
        "Trie3 reordering complete. Ranges reduced from {} to {}. New max internal token ID: {}",
        ranges_before,
        ranges_after,
        stage_vocab.internal_max_llm_token
    );
}
