use std::collections::{BTreeMap, HashMap, HashSet};

use crate::trie3_opt::{context::OptimizationContext, core::{MiniTrie, SortedSet}, passes::OptimizationPass, NodeId};

pub struct MergeEquivalentLLMTokensPass;

impl OptimizationPass for MergeEquivalentLLMTokensPass {
    fn name(&self) -> &'static str {
        "MergeEquivalentLLMTokens"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let sv_rc = if let Some(sv_rc) = &ctx.stage_vocab {
            sv_rc
        } else {
            return;
        };

        let mut all_bvs = HashSet::new();
        for node in trie.nodes.values() {
            for (ek, _) in &node.children {
                if !ek.tokens.is_empty() {
                    all_bvs.insert(ek.tokens.clone());
                }
            }
        }
        if all_bvs.is_empty() {
            return;
        }

        let max_tok = ctx.max_llm_token_id;
        let mut token_to_class: Vec<usize> = vec![0; max_tok + 1];
        let mut class_to_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
        class_to_tokens.insert(0, (0..=max_tok).collect());
        let mut num_classes = 1;

        for splitter_bv in all_bvs {
            let mut members_in_splitter_by_class: HashMap<usize, Vec<usize>> = HashMap::new();
            for token in splitter_bv.iter() {
                if token <= max_tok {
                    let class_id = token_to_class[token];
                    members_in_splitter_by_class.entry(class_id).or_default().push(token);
                }
            }

            for (old_class_id, tokens_for_new_class) in members_in_splitter_by_class {
                let old_class_size = class_to_tokens.get(&old_class_id).map_or(0, |v| v.len());
                if old_class_size > 0 && !tokens_for_new_class.is_empty() && tokens_for_new_class.len() < old_class_size {
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

        let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
        for group in class_to_tokens.values() {
            if group.len() > 1 {
                let rep = *group.iter().min().unwrap();
                for &t in group {
                    if t != rep {
                        old_to_new.insert(t, rep);
                    }
                }
            }
        }

        if old_to_new.is_empty() {
            return;
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
        for (old, rep) in old_to_new {
            if let Some(moved) = sv.internal_to_original.remove(&old) {
                let entry = sv.internal_to_original.entry(rep).or_default();
                *entry |= &moved;
                for o in moved.iter() {
                    sv.original_to_internal.insert(o, rep);
                }
            }
        }
    }
}
