use std::collections::{BTreeMap, BTreeSet, HashMap};
use ordered_hash_map::OrderedHashMap;
use kdam::tqdm;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::constraint::{StageVocab, PrecomputeNodeIndex, Trie1GodWrapper};
use crate::datastructures::gss::LLMTokenBV;
use crate::datastructures::trie::Trie;

fn remap_llm_bv_many_to_one(bv: &LLMTokenBV, map_old_to_new: &BTreeMap<usize, usize>) -> LLMTokenBV {
    if bv.is_empty() { return LLMTokenBV::zeros(); }
    let mut out = LLMTokenBV::zeros();
    for t in bv.iter() {
        let ti = t as usize;
        let rep = map_old_to_new.get(&ti).copied().unwrap_or(ti);
        out.insert(rep);
    }
    out
}

fn remap_llm_bv_permutation(bv: &LLMTokenBV, map_old_to_new: &BTreeMap<usize, usize>) -> LLMTokenBV {
    // Same implementation; map is bijection in permutation case.
    remap_llm_bv_many_to_one(bv, map_old_to_new)
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
    for n in &all_nodes {
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

    // 2) Build signature per token
    let max_tok = stage_vocab.internal_max_llm_token;
    let mut sig_map: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    #[cfg(not(rustrover))]
    let it = tqdm!(0..=max_tok, desc = "Trie1 Merge Tokens (Sigs)", disable = !PROGRESS_BAR_ENABLED, leave=false);
    #[cfg(rustrover)] let it = 0..=max_tok;
    for tok in it {
        let mut sig: Vec<usize> = Vec::new();
        for (i, setv) in family.iter().enumerate() {
            if setv.contains(tok) { sig.push(i); }
        }
        if sig.is_empty() { continue; }
        sig_map.entry(sig).or_default().push(tok);
    }

    // 3) Build many-to-one mapping
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    let mut merged_count = 0;
    for (_sig, group) in tqdm!(sig_map.into_iter(), desc = "Building mapping") {
        if group.len() <= 1 { continue; }
        let rep = *group.iter().min().unwrap();
        for t in group {
            if t != rep {
                old_to_new.insert(t, rep);
                merged_count += 1;
            }
        }
    }
    crate::debug!(2, "Trie1: merged {} LLM tokens into representatives.", merged_count);
    if merged_count == 0 { return; }

    // 4) Apply mapping to trie
    let mut new_states = Vec::with_capacity(all_nodes.len());
    eprintln!("[DEBUG] merge_equivalent_llm_tokens_trie1: Remapping trie (read) for {} nodes", all_nodes.len());
    for (i, n) in tqdm!(all_nodes.iter().enumerate(), desc = "Remapping trie (read)") {
        eprintln!("[DEBUG] Remapping node {}/{}: {:?}", i, all_nodes.len(), n);
        let r = n.read(trie1_god).expect("read");
        eprintln!("[DEBUG]   - Acquired read lock for node {}", i);
        let new_live_tokens = if r.value.live_tokens.is_empty() {
            eprintln!("[DEBUG]   - live_tokens is empty");
            r.value.live_tokens.clone()
        } else {
            eprintln!("[DEBUG]   - remapping live_tokens (len={})", r.value.live_tokens.len());
            let result = remap_llm_bv_many_to_one(&r.value.live_tokens, &old_to_new);
            eprintln!("[DEBUG]   - remapped live_tokens (new_len={})", result.len());
            result
        };
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV>> = BTreeMap::new();
        eprintln!("[DEBUG]   - Processing {} child edge keys", r.children().len());
        for (ek_idx, (ek, dm)) in r.children().iter().enumerate() {
            eprintln!("[DEBUG]     - Edge key {}/{}: {:?}", ek_idx, r.children().len(), ek);
            let mut new_dm: OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV> = OrderedHashMap::new();
            for (dst_idx, (dst, bv)) in dm.iter().enumerate() {
                eprintln!("[DEBUG]       - Dest {}/{}: {:?}, bv_len={}", dst_idx, dm.len(), dst, bv.len());
                let mapped = remap_llm_bv_many_to_one(&bv, &old_to_new);
                if !mapped.is_empty() {
                    new_dm.insert(dst.clone(), mapped);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek.clone(), new_dm);
            }
        }
        eprintln!("[DEBUG]   - Processed children for node {}", i);
        new_states.push((new_live_tokens, new_children));
        eprintln!("[DEBUG]   - Pushed new state for node {}", i);
        eprintln!("[DEBUG]   - Releasing read lock for node {}", i);
    }
    eprintln!("[DEBUG] merge_equivalent_llm_tokens_trie1: Applying changes to trie");
    for (i, n) in all_nodes.iter().enumerate() {
        let mut w = n.write(trie1_god).expect("write");
        let (live_tokens, children) = &new_states[i];
        w.value.live_tokens = live_tokens.clone();
        *w.children_mut() = children.clone();
    }
    // 5) Update stage vocab
    // Merge internal_to_original for tokens mapped into representatives
    for (old, new_rep) in tqdm!(old_to_new.iter(), desc = "Updating stage vocab") {
        if old == new_rep { continue; }
        let moved = stage_vocab.internal_to_original.remove(old).unwrap_or_default();
        stage_vocab.internal_to_original.entry(*new_rep).or_default().extend(moved.clone());
        // Fix original->internal for all affected originals
        for o in moved {
            stage_vocab.original_to_internal.insert(o, *new_rep);
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
    crate::debug!(2, "Reordering LLM tokens in Trie1 for range minimization...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    if all_nodes.is_empty() { return; }
    let max_tok = stage_vocab.internal_max_llm_token;

    // Count frequencies
    let mut freq: Vec<usize> = vec![0; max_tok + 1];
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Reorder (Freq)", total=all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave=false);
    #[cfg(rustrover)] let it = all_nodes.iter();
    for n in it {
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

    // Build ordering: tokens present at least once, sorted by (freq desc, id asc)
    let mut present: Vec<usize> = (0..=max_tok).filter(|t| freq[*t] > 0).collect();
    if present.is_empty() { return; }
    present.sort_by_key(|&t| (std::cmp::Reverse(freq[t]), t));

    // Build permutation
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    for (new_id, old_id) in present.iter().enumerate() {
        old_to_new.insert(*old_id, new_id);
    }
    // Apply mapping to trie
    let mut new_states = Vec::with_capacity(all_nodes.len());
    eprintln!("[DEBUG] reorder_llm_tokens_for_range_minimization_trie1: Remapping trie (read) for {} nodes", all_nodes.len());
    for (i, n) in all_nodes.iter().enumerate() {
        eprintln!("[DEBUG] Remapping node {}/{}: {:?}", i, all_nodes.len(), n);
        let r = n.read(trie1_god).expect("read");
        eprintln!("[DEBUG]   - Acquired read lock for node {}", i);
        let new_live_tokens = if r.value.live_tokens.is_empty() {
            eprintln!("[DEBUG]   - live_tokens is empty");
            r.value.live_tokens.clone()
        } else {
            eprintln!("[DEBUG]   - remapping live_tokens (len={})", r.value.live_tokens.len());
            let result = remap_llm_bv_permutation(&r.value.live_tokens, &old_to_new);
            eprintln!("[DEBUG]   - remapped live_tokens (new_len={})", result.len());
            result
        };
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV>> = BTreeMap::new();
        eprintln!("[DEBUG]   - Processing {} child edge keys", r.children().len());
        for (ek_idx, (ek, dm)) in r.children().iter().enumerate() {
            eprintln!("[DEBUG]     - Edge key {}/{}: {:?}", ek_idx, r.children().len(), ek);
            let mut new_dm: OrderedHashMap<PrecomputeNodeIndex, LLMTokenBV> = OrderedHashMap::new();
            for (dst_idx, (dst, bv)) in dm.iter().enumerate() {
                eprintln!("[DEBUG]       - Dest {}/{}: {:?}, bv_len={}", dst_idx, dm.len(), dst, bv.len());
                let mapped = remap_llm_bv_permutation(&bv, &old_to_new);
                if !mapped.is_empty() {
                    new_dm.insert(dst.clone(), mapped);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek.clone(), new_dm);
            }
        }
        eprintln!("[DEBUG]   - Processed children for node {}", i);
        new_states.push((new_live_tokens, new_children));
        eprintln!("[DEBUG]   - Pushed new state for node {}", i);
        eprintln!("[DEBUG]   - Releasing read lock for node {}", i);
    }
    eprintln!("[DEBUG] reorder_llm_tokens_for_range_minimization_trie1: Applying changes to trie");
    for (i, n) in all_nodes.iter().enumerate() {
        let mut w = n.write(trie1_god).expect("write");
        let (live_tokens, children) = &new_states[i];
        w.value.live_tokens = live_tokens.clone();
        *w.children_mut() = children.clone();
    }
    // Update stage vocab (pure permutation)
    let mut new_internal_to_original: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (old_id, setv) in stage_vocab.internal_to_original.clone() {
        if let Some(new_id) = old_to_new.get(&old_id) {
            new_internal_to_original.insert(*new_id, setv);
        }
    }
    stage_vocab.internal_to_original = new_internal_to_original;
    let mut new_original_to_internal: BTreeMap<usize, usize> = BTreeMap::new();
    for (orig, old_internal) in stage_vocab.original_to_internal.clone() {
        if let Some(new_internal) = old_to_new.get(&old_internal) {
            new_original_to_internal.insert(orig, *new_internal);
        }
    }
    stage_vocab.original_to_internal = new_original_to_internal;
    stage_vocab.internal_max_llm_token = present.len().saturating_sub(1);
    crate::debug!(2, "Trie1 reordering complete. New max internal token ID: {}", stage_vocab.internal_max_llm_token);
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
