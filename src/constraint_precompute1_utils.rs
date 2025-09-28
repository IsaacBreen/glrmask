use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use range_set_blaze::RangeSetBlaze;
use ordered_hash_map::OrderedHashMap;
use kdam::tqdm;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::constraint::{StageVocab, PrecomputeNodeIndex, Trie1GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::gss::LLMTokenBV;
use crate::datastructures::trie::Trie;

fn count_total_ranges_trie1(
    all_nodes: &[PrecomputeNodeIndex],
    trie1_god: &Trie1GodWrapper,
) -> usize {
    let mut count = 0;
    for n in all_nodes {
        let g = n.read(trie1_god).expect("read");
        count += g.value.live_tokens.inner().ranges_len();
        for (_ek, dm) in g.children() {
            for (_dst, bv) in dm {
                count += bv.inner().ranges_len();
            }
        }
    }
    count
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

/// Merge equivalent internal LLM token ids in Trie1:
/// Two tokens are equivalent if they appear together in every occurrence across:
/// - node.value.live_tokens
/// - every edge's LLMTokenBV
///
/// Applies a many-to-one id mapping and merges masks accordingly.
pub fn merge_equivalent_llm_tokens_trie1(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNodeIndex>,
    trie1_god: &Trie1GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    crate::debug!(2, "Merging equivalent LLM tokens in Trie1...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    // 1) Collect all LLM sets (node live_tokens, edge masks)
    let mut family: Vec<LLMTokenBV> = Vec::new();
    for n in tqdm!(all_nodes.iter(), desc = "Trie1 Merge Tokens (Collect)", disable = !PROGRESS_BAR_ENABLED, leave = false) {
        let g = n.read(trie1_god).expect("read");
        if !g.value.live_tokens.is_empty() {
            family.push(g.value.live_tokens.clone());
        }
        for (_ek, dm) in g.children() {
            for (_dst, bv) in dm {
                if !bv.is_empty() {
                    family.push(bv.clone());
                }
            }
        }
    }
    if family.is_empty() { return; }

    // 2) Build signatures using occurrence‑based accumulation (faster)
    let max_tok = stage_vocab.internal_max_llm_token;
    let mut per_token_sigs: Vec<Vec<usize>> = vec![Vec::new(); max_tok + 1];
    let mut max_ones_indices: Vec<usize> = Vec::new();
    for (i, setv) in tqdm!(family.iter().enumerate(), desc = "Trie1 Merge Tokens (Sigs: scan sets)", disable = !PROGRESS_BAR_ENABLED, leave = false) {
        if *setv == LLMTokenBV::max_ones() {
            max_ones_indices.push(i);
            continue;
        }
        for t in setv.iter() {
            let ti = t as usize;
            if ti <= max_tok {
                per_token_sigs[ti].push(i);
            }
        }
    }
    // Merge the “max_ones” prefix with each token’s per‑set list.
    let mut sig_map: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    #[cfg(not(rustrover))]
    let it2 = tqdm!(0..=max_tok, desc = "Trie1 Merge Tokens (Build Sig Map)", disable = !PROGRESS_BAR_ENABLED, leave=false);
    #[cfg(rustrover)] let it2 = 0..=max_tok;
    for tok in it2 {
        let a = &max_ones_indices;
        let b = &per_token_sigs[tok];
        if a.is_empty() && b.is_empty() { continue; }
        let mut sig: Vec<usize> = Vec::with_capacity(a.len() + b.len());
        let mut i = 0usize;
        let mut j = 0usize;
        while i < a.len() && j < b.len() {
            if a[i] <= b[j] {
                sig.push(a[i]); i += 1;
            } else {
                sig.push(b[j]); j += 1;
            }
        }
        if i < a.len() { sig.extend_from_slice(&a[i..]); }
        if j < b.len() { sig.extend_from_slice(&b[j..]); }
        sig_map.entry(sig).or_default().push(tok);
    }

    // 3) Build many‑to‑one mapping
    let tokens_before = sig_map.values().map(|g| g.len()).sum::<usize>();
    let tokens_after = sig_map.len();
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    let mut merged_count = 0;
    for (_sig, group) in tqdm!(sig_map.iter(), desc = "Trie1 Merge Tokens (Build Map)", disable = !PROGRESS_BAR_ENABLED, leave = false) {
        if group.len() <= 1 { continue; }
        let rep = *group.iter().min().unwrap();
        for &t in group.iter() {
            if t != rep {
                old_to_new.insert(t, rep);
                merged_count += 1;
            }
        }
    }
    crate::debug!(2, "Trie1: merged LLM tokens. Before: {}, After: {}. ({} merged)", tokens_before, tokens_after, merged_count);
    if merged_count == 0 { return; }

    // Precompute the mapped universal set once (used when a set equals max_ones())
    let mut mapped_universe = LLMTokenBV::zeros();
    for t in 0..=max_tok {
        let rep = old_to_new.get(&t).copied().unwrap_or(t);
        mapped_universe.insert(rep);
    }

    // Memoization cache for remapping
    let mut remap_cache: HashMap<LLMTokenBV, LLMTokenBV> = HashMap::new();

    // 4) Apply mapping to trie in‑place, only where needed
    for n in tqdm!(all_nodes.iter(), desc = "Trie1 Merge (Remap In‑Place)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false) {
        let mut w = n.write(trie1_god).expect("write");

        // Remap live_tokens if needed
        if !w.value.live_tokens.is_empty() {
            w.value.live_tokens = remap_cache.entry(w.value.live_tokens.clone()).or_insert_with_key(|bv| {
                if *bv == LLMTokenBV::max_ones() {
                    mapped_universe.clone()
                } else {
                    remap_llm_bv_many_to_one(bv, &old_to_new, max_tok)
                }
            }).clone();
        }

        // Remap children edge masks
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in old_children {
            let mut new_dm: OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV> = OrderedHashMap::new();
            for (dst, bv) in dm {
                let mapped = if bv.is_empty() {
                    LLMTokenBV::zeros()
                } else {
                    remap_cache.entry(bv.clone()).or_insert_with_key(|bv| {
                        if *bv == LLMTokenBV::max_ones() {
                            mapped_universe.clone()
                        } else {
                            remap_llm_bv_many_to_one(bv, &old_to_new, max_tok)
                        }
                    }).clone()
                };
                if !mapped.is_empty() {
                    new_dm.entry(dst)
                        .and_modify(|e| *e |= &mapped)
                        .or_insert(mapped);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
	}
	// 5) Update stage vocab
	// Merge internal_to_original for tokens mapped into representatives
	for (old, new_rep) in tqdm!(old_to_new.iter(), desc = "Trie1 Merge (Update Vocab)", total = old_to_new.len(), disable = !PROGRESS_BAR_ENABLED, leave = false) {
		if old == new_rep { continue; }
		if let Some(moved) = stage_vocab.internal_to_original.remove(old) {
			let entry = stage_vocab.internal_to_original.entry(*new_rep).or_default();
			*entry |= &moved;
			for o in moved.iter() {
				stage_vocab.original_to_internal.insert(o, *new_rep);
			}
		}
	}
	// internal_max_llm_token stays the same here (holes may appear). A later reorder can compact.
}

/// Reorder internal LLM tokens (permutation) to reduce ranges in masks by clustering co-occurring tokens.
/// Conservative heuristic: sort by (descending frequency, then by id).
pub fn reorder_llm_tokens_for_range_minimization_trie1(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNodeIndex>,
    trie1_god: &Trie1GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    if all_nodes.is_empty() { return; }
    let ranges_before = count_total_ranges_trie1(&all_nodes, trie1_god);

    let max_tok = stage_vocab.internal_max_llm_token;

    // Count frequencies
    let mut freq: Vec<usize> = vec![0; max_tok + 1];
    for n in tqdm!(all_nodes.iter(), desc = "Trie1 Reorder (Freq)", total = all_nodes.len()) {
        let g = n.read(trie1_god).expect("read");
        for t in g.value.live_tokens.iter() {
            if t as usize <= max_tok { freq[t as usize] += 1; }
        }
        for (_ek, dm) in g.children() {
            for (_dst, bv) in dm {
                for t in bv.iter() {
                    if t as usize <= max_tok { freq[t as usize] += 1; }
                }
            }
        }
    }
    crate::debug!(2, "Done computing frequencies.");

    // Build ordering: tokens present at least once, sorted by (freq desc, id asc)
    let mut present: Vec<usize> = (0..=max_tok).filter(|t| freq[*t] > 0).collect();
    if present.is_empty() { return; }
    present.sort_by_key(|&t| (std::cmp::Reverse(freq[t]), t));

    // Build permutation
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    for (new_id, old_id) in present.iter().enumerate() {
        old_to_new.insert(*old_id, new_id);
    }

    // Memoization cache for remapping
    let mut remap_cache: HashMap<LLMTokenBV, LLMTokenBV> = HashMap::new();

    // Apply mapping to trie
    let mut new_states = Vec::with_capacity(all_nodes.len());
    for n in tqdm!(all_nodes.iter(), desc = "Trie1 Reorder (Remap Read)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false) {
        let r = n.read(trie1_god).expect("read");
        let new_live_tokens = if r.value.live_tokens.is_empty() {
            r.value.live_tokens.clone()
        } else {
            remap_cache.entry(r.value.live_tokens.clone()).or_insert_with_key(|bv| {
                remap_llm_bv_permutation(bv, &old_to_new, max_tok)
            }).clone()
        };
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in r.children() {
            let mut new_dm: OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV> = OrderedHashMap::new();
            for (dst, bv) in dm {
                let mapped = remap_cache.entry(bv.clone()).or_insert_with_key(|bv| {
                    remap_llm_bv_permutation(bv, &old_to_new, max_tok)
                }).clone();
                if !mapped.is_empty() {
                    new_dm.insert(dst.clone(), mapped);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek.clone(), new_dm);
            }
        }
        new_states.push((new_live_tokens, new_children));
    }
    for (i, n) in tqdm!(all_nodes.iter().enumerate(), desc = "Trie1 Reorder (Remap Write)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false) {
        let mut w = n.write(trie1_god).expect("write");
        let (live_tokens, children) = &new_states[i];
        w.value.live_tokens = live_tokens.clone();
        *w.children_mut() = children.clone();
    }
	let ranges_after = count_total_ranges_trie1(&all_nodes, trie1_god);

	// Update stage vocab (pure permutation)
	let mut new_internal_to_original: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
	for (old_id, setv) in tqdm!(stage_vocab.internal_to_original.clone().into_iter(), desc = "Trie1 Reorder (Vocab 1)", disable = !PROGRESS_BAR_ENABLED, leave = false) {
		if let Some(new_id) = old_to_new.get(&old_id) {
			new_internal_to_original.insert(*new_id, setv);
        }
    }
    stage_vocab.internal_to_original = new_internal_to_original;
    let mut new_original_to_internal: BTreeMap<usize, usize> = BTreeMap::new();
    for (orig, old_internal) in tqdm!(stage_vocab.original_to_internal.clone().into_iter(), desc = "Trie1 Reorder (Vocab 2)", disable = !PROGRESS_BAR_ENABLED, leave = false) {
        if let Some(new_internal) = old_to_new.get(&old_internal) {
            new_original_to_internal.insert(orig, *new_internal);
        }
    }
    stage_vocab.original_to_internal = new_original_to_internal;
    stage_vocab.internal_max_llm_token = present.len().saturating_sub(1);
    crate::debug!(2, "Trie1 reordering complete. Ranges reduced from {} to {}. New max internal token ID: {}", ranges_before, ranges_after, stage_vocab.internal_max_llm_token);
}

/// Conservative normalization pass for Trie1:
/// - Coalesce duplicate destination entries (union LLMBV) for same child under a terminal key.
/// - Remove empty masks.
pub fn optimize_state_masks_and_edges_trie1(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNodeIndex>,
    trie1_god: &Trie1GodWrapper,
) {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    for n in &all_nodes {
        let mut w = n.write(trie1_god).expect("write");
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in w.children().clone() {
            // Union masks for same dst
            let mut coalesced: HashMap<PrecomputeNodeIndex, LLMTokenBV> = HashMap::new();
            for (dst, bv) in dm {
                let entry = coalesced.entry(dst).or_insert_with(LLMTokenBV::zeros);
                *entry |= &bv;
            }
            let mut new_dm: OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV> = OrderedHashMap::new();
            for (dst, bv) in coalesced {
                if !bv.is_empty() {
                    new_dm.insert(dst, bv);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
    }
}
