use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::{context::OptimizationContext, core::{MiniTrie, SortedSet}, passes::OptimizationPass, NodeId};

pub struct ReorderLLMTokensPass;

impl OptimizationPass for ReorderLLMTokensPass {
    fn name(&self) -> &'static str {
        "ReorderLLMTokens"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let sv_rc = if let Some(sv_rc) = &ctx.stage_vocab {
            sv_rc
        } else {
            return;
        };

        let max_tok = ctx.max_llm_token_id;
        let mut freq: Vec<usize> = vec![0; max_tok + 1];
        for node in trie.nodes.values() {
            for (ek, _) in &node.children {
                for t in ek.tokens.iter() {
                    if t <= max_tok {
                        freq[t] += 1;
                    }
                }
            }
        }

        let mut present: Vec<usize> = (0..=max_tok).filter(|&t| freq[t] > 0).collect();
        if present.is_empty() {
            return;
        }
        present.sort_by_key(|&t| (std::cmp::Reverse(freq[t]), t));

        let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
        for (new_id, old_id) in present.iter().enumerate() {
            old_to_new.insert(*old_id, new_id);
        }

        let remap_sorted_set = |s: &SortedSet| -> SortedSet {
            let mut new_elems = Vec::with_capacity(s.len());
            for elem in s.iter() {
                new_elems.push(old_to_new.get(&elem).copied().unwrap_or(elem));
            }
            SortedSet::from_iter(new_elems)
        };

        for node in trie.nodes.values_mut() {
            let mut new_children = BTreeMap::new();
            for (mut ek, dm) in std::mem::take(&mut node.children) {
                ek.tokens = remap_sorted_set(&ek.tokens);
                let entry: &mut BTreeMap<NodeId, SortedSet> = new_children.entry(ek).or_default();
                for (dst, sids) in dm {
                    entry.entry(dst).or_default().union_inplace(&sids);
                }
            }
            node.children = new_children;
        }

        let mut sv_ref = sv_rc.borrow_mut();
        let sv = &mut **sv_ref;

        let mut new_internal_to_original = BTreeMap::new();
        for (old_id, setv) in sv.internal_to_original.clone() {
            if let Some(new_id) = old_to_new.get(&old_id) {
                new_internal_to_original.insert(*new_id, setv);
            }
        }
        sv.internal_to_original = new_internal_to_original;

        let mut new_original_to_internal = BTreeMap::new();
        for (orig, old_internal) in sv.original_to_internal.clone() {
            if let Some(new_internal) = old_to_new.get(&old_internal) {
                new_original_to_internal.insert(orig, *new_internal);
            }
        }
        sv.original_to_internal = new_original_to_internal;
        sv.internal_max_llm_token = present.len().saturating_sub(1);
        ctx.max_llm_token_id = sv.internal_max_llm_token;
    }
}
