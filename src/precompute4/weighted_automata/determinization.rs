// src/precompute4/weighted_automata/determinization.rs
//
// A radically simplified and aggressively optimized determinization.
//
// High-level:
// - We avoid the previous "patch caching", "signature interning", and progress-bar overhead.
// - We perform one-time epsilon-closure precomputation per NWA state, pruned by the future-acceptance
//   masks F[s] (so we never carry dead weights).
// - Determinization uses a straightforward subset construction over weighted closures with defaults:
//     • For a determinized state S (a sparse map s -> gate_weight), we compute:
//         - default target Tdef by pushing all defaults through closures;
//         - per-label patches by accumulating "adds" (exception edges) and "removes" (the corresponding
//           defaults that must be overridden). Each label is handled exactly once.
//     • Per-label target is then: Tlbl = Tdef - removes(lbl) + adds(lbl).
// - We intern subsets by their canonical (sorted) vector representation to avoid duplicates.
// - We never iterate over the full alphabet; only labels that actually occur as exceptions in S are considered.
// - All weight operations are sparse and bitset-based, so union/intersection/difference are O(#bit-lanes).
//
// Correctness sketch:
// - The epsilon-closure C(s) is computed as a least fixpoint under the semiring (∨ as sum, ∧ as product),
//   starting with fut[s], i.e., the future-acceptance mask at s. This guarantees we only propagate bits
//   that can eventually reach a final state, preserving language semantics while pruning dead branches.
// - For a determinized state S, the "default" transition corresponds to choosing the default out-edge
//   in each NWA state in S, intersecting with the gate_weight, then distributing across C(target).
//   Since default applies to any label that is not an explicit exception, this defines a uniform base.
// - For any label ℓ that is an exception in some source t in S, we must:
//     • Subtract what t's default would have contributed (since ℓ overrides default in t);
//     • Add what t's exception contributes; this restores correct per-label behavior.
//   Aggregating across all sources gives Tℓ = Tdef - (⋁ over default-contribs for ℓ) + (⋁ over exception-contribs for ℓ),
//   which matches NFA semantics with weight-union across alternative paths.
// - Final weights for a determinized state are ⋁_s (gate_weight[s] ∧ final[s]); epsilon effects are already
//   folded into S via closures.
//
// Complexity:
// - One epsilon-closure per NWA state.
// - For each determinized state S:
//     • O(|S|) work to compute default target;
//     • O(total exceptions across states in S) to build all per-label patches;
//     • Each label is processed exactly once, merging 3 sorted sparse maps (def/rem/add).
// - No dependence on alphabet size beyond the explicit exception labels present.
//
// This approach is robust and scales to very large alphabets with defaults.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::Weight;
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        // Trivial cases
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        // 1) Compute future-acceptance masks F[s] to prune closures to only productive weights.
        let fut = self.compute_future_weights();

        // 2) Precompute epsilon-closures C(s) and their total masks:
        //    C(s) = Vec<(t, w_st)>, where w_st = ⋁ over ε-paths s->*t (∧ over edge-weights and fut gating)
        //    total_mask[s] = ⋁_{(t, w_st) in C(s)} w_st
        let (closures, closure_total_masks) = compute_all_epsilon_closures(self, &fut);

        // 3) Standard weighted subset construction (sparse, default + exception patching).
        determinize_closures_to_dwa(self, &closures, &closure_total_masks)
    }
}

#[derive(Clone)]
struct Subset {
    // Sorted by NWA state id, weights non-empty.
    items: Vec<(NWAStateID, Weight)>,
    fp: u64, // lightweight fingerprint to speed hashing
}
impl Subset {
    fn new(mut items: Vec<(NWAStateID, Weight)>) -> Self {
        // Normalize: sort by id, merge duplicates, drop empties.
        items.sort_by_key(|(sid, _)| *sid);
        let mut norm: Vec<(NWAStateID, Weight)> = Vec::with_capacity(items.len());
        for (sid, w) in items.into_iter() {
            if w.is_empty() {
                continue;
            }
            if let Some((last_sid, ref mut last_w)) = norm.last_mut() {
                if *last_sid == sid {
                    *last_w |= &w;
                    continue;
                }
            }
            norm.push((sid, w));
        }
        let mut fp = 0x9E3779B185EBCA87u64;
        for (sid, w) in &norm {
            // Combine state-id and weight fingerprint; use a splitmix-like mixer
            let x = (*sid as u64).wrapping_mul(0xD6E8FEB86659FD93);
            let y = w.fp;
            fp = fp ^ (x.rotate_left(13) ^ y.rotate_right(7)).wrapping_mul(0x94D049BB133111EB);
        }
        Subset { items: norm, fp }
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
impl PartialEq for Subset {
    fn eq(&self, other: &Self) -> bool {
        self.fp == other.fp && self.items == other.items
    }
}
impl Eq for Subset {}
impl Hash for Subset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.fp);
    }
}

fn compute_all_epsilon_closures(
    nwa: &NWA,
    fut: &[Weight],
) -> (Vec<Vec<(NWAStateID, Weight)>>, Vec<Weight>) {
    let n = nwa.states.len();
    let mut closures: Vec<Vec<(NWAStateID, Weight)>> = Vec::with_capacity(n);
    let mut total_masks: Vec<Weight> = Vec::with_capacity(n);

    for s in 0..n {
        // Weighted ε-closure seeded by fut[s].
        let mut res_map: HashMap<NWAStateID, Weight> = HashMap::new();
        let mut q: VecDeque<NWAStateID> = VecDeque::new();

        let start_w = fut[s].clone();
        if !start_w.is_empty() {
            res_map.insert(s, start_w.clone());
            q.push_back(s);
        }

        while let Some(u) = q.pop_front() {
            let u_w = res_map.get(&u).cloned().unwrap_or_else(Weight::zeros);
            if u_w.is_empty() {
                continue;
            }
            for &(v, ref w_eps) in &nwa.states[u].epsilons {
                let mut prop = &u_w & w_eps;
                if prop.is_empty() {
                    continue;
                }
                prop &= &fut[v];
                if prop.is_empty() {
                    continue;
                }
                match res_map.get_mut(&v) {
                    Some(old) => {
                        let merged = &*old | &prop;
                        if &merged != old {
                            *old = merged;
                            q.push_back(v);
                        }
                    }
                    None => {
                        res_map.insert(v, prop);
                        q.push_back(v);
                    }
                }
            }
        }

        // Canonicalize to sorted vec
        let mut vec_pairs: Vec<(NWAStateID, Weight)> = res_map.into_iter().collect();
        vec_pairs.sort_by_key(|(k, _)| *k);

        // Total mask
        let mut tot = Weight::zeros();
        for (_, w) in &vec_pairs {
            tot |= w;
        }

        closures.push(vec_pairs);
        total_masks.push(tot);
    }
    (closures, total_masks)
}

fn determinize_closures_to_dwa(
    nwa: &NWA,
    closures: &[Vec<(NWAStateID, Weight)>],
    closure_total_masks: &[Weight],
) -> DWA {
    let n = nwa.states.len();

    // Helper: push contributions through a closure C(t) with a gate weight.
    #[inline(always)]
    fn add_closure_contrib(
        target_map: &mut HashMap<NWAStateID, Weight>,
        closure: &[(NWAStateID, Weight)],
        gate: &Weight,
    ) {
        if gate.is_empty() {
            return;
        }
        for (v, w_tv) in closure.iter() {
            let x = gate & w_tv;
            if x.is_empty() {
                continue;
            }
            match target_map.get_mut(v) {
                Some(old) => *old |= &x,
                None => {
                    target_map.insert(*v, x);
                }
            }
        }
    }

    // Convert a sparse map to a sorted Vec form.
    #[inline(always)]
    fn map_to_sorted_vec(m: HashMap<NWAStateID, Weight>) -> Vec<(NWAStateID, Weight)> {
        let mut v: Vec<(NWAStateID, Weight)> = m.into_iter().collect();
        v.sort_by_key(|(k, _)| *k);
        // merge duplicates shouldn't be necessary here (keys unique), but keep minimal code path
        v
    }

    // Three-way merge: base (def), minus rem, plus add. All inputs sorted by key.
    fn merge_def_rem_add(
        def: &[(NWAStateID, Weight)],
        rem: &[(NWAStateID, Weight)],
        add: &[(NWAStateID, Weight)],
    ) -> Vec<(NWAStateID, Weight)> {
        let mut out: Vec<(NWAStateID, Weight)> = Vec::with_capacity(
            def.len().saturating_add(add.len()).saturating_add(rem.len()) / 2 + 4,
        );

        let mut i = 0usize;
        let mut j = 0usize;
        let mut k = 0usize;

        while i < def.len() || j < rem.len() || k < add.len() {
            let mut next = usize::MAX;

            if i < def.len() {
                next = next.min(def[i].0);
            }
            if j < rem.len() {
                next = next.min(rem[j].0);
            }
            if k < add.len() {
                next = next.min(add[k].0);
            }

            // Start from default weight if present
            let mut w = if i < def.len() && def[i].0 == next {
                let ww = def[i].1.clone();
                i += 1;
                ww
            } else {
                Weight::zeros()
            };

            // Subtract removal if present
            if j < rem.len() && rem[j].0 == next {
                let remw = rem[j].1.clone();
                j += 1;
                w -= &remw;
            }

            // Add addition if present
            if k < add.len() && add[k].0 == next {
                let addw = add[k].1.clone();
                k += 1;
                w |= &addw;
            }

            if !w.is_empty() {
                out.push((next, w));
            }
        }

        out
    }

    // Interner: subset -> DWA state id
    let mut subset_to_did: HashMap<Subset, usize> = HashMap::new();
    let mut subsets_arena: Vec<Subset> = Vec::new();

    // Start subset: epsilon-closure from start state
    let init_subset = Subset::new(closures[nwa.body.start_state].clone());
    let mut dwa = DWA::new();
    dwa.states.0.clear();
    let start_id = dwa.states.add_state();
    dwa.body.start_state = start_id;

    subset_to_did.insert(init_subset.clone(), start_id);
    subsets_arena.push(init_subset);

    // Worklist of DWA state IDs to process
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(start_id);

    // Scratch maps reused per determinized state to avoid realloc churn
    // Default accumulator
    let mut def_map: HashMap<NWAStateID, Weight> = HashMap::new();

    // Per-label aggregators
    #[derive(Default)]
    struct LabelAgg {
        add_map: HashMap<NWAStateID, Weight>,
        rem_map: HashMap<NWAStateID, Weight>,
        add_mask: Weight,
        rem_mask: Weight,
    }
    let mut label_aggs: HashMap<i16, LabelAgg> = HashMap::new();

    while let Some(did) = q.pop_front() {
        let subset = subsets_arena[did].clone();

        // 1) Final weight of this DWA state
        let mut final_w: Option<Weight> = None;
        for (s, gate) in &subset.items {
            if let Some(fw) = &nwa.states[*s].final_weight {
                let c = gate & fw;
                if !c.is_empty() {
                    if let Some(acc) = &mut final_w {
                        *acc |= &c;
                    } else {
                        final_w = Some(c);
                    }
                }
            }
        }
        dwa.states[did].final_weight = final_w;

        // Edge case: empty subset (shouldn't happen because we prune with fut), but guard anyway.
        if subset.items.is_empty() {
            continue;
        }

        // 2) Build default target once: Tdef = ⋁ over sources with defaults of (gate ∧ wdef) pushed through C(to)
        def_map.clear();
        let mut def_edge_mask = Weight::zeros();

        for (s, gate) in &subset.items {
            if let Some((to, wdef)) = &nwa.states[*s].default {
                let g = gate & wdef;
                if g.is_empty() {
                    continue;
                }
                // Edge weight mask contribution
                let m = &g & &closure_total_masks[*to];
                def_edge_mask |= &m;
                // Push through closure
                add_closure_contrib(&mut def_map, &closures[*to], &g);
            }
        }
        let def_vec = map_to_sorted_vec(def_map.clone()); // keep a clone for label merges (this is a shared base)
        let def_subset = Subset::new(def_vec.clone());

        // Install default edge (if non-empty)
        let mut def_target_id_opt: Option<usize> = None;
        if !def_vec.is_empty() && !def_edge_mask.is_empty() {
            let def_did = match subset_to_did.get(&def_subset) {
                Some(id) => *id,
                None => {
                    let nid = dwa.states.add_state();
                    subset_to_did.insert(def_subset.clone(), nid);
                    subsets_arena.push(def_subset);
                    q.push_back(nid);
                    nid
                }
            };
            def_target_id_opt = Some(def_did);
            // Add default transition; if it already exists, it will error; but we only set once per state.
            let _ = dwa.set_default_transition(did, def_did, def_edge_mask.clone());
        }

        // 3) Aggregate per-label patches: for every exception seen in any source.
        label_aggs.clear();

        for (s, gate) in &subset.items {
            // For each labeled exception out of s
            for (lbl, (to_ex, wex)) in nwa.states[*s].transitions.iter() {
                // ADD: exception contributions
                let g_ex = gate & wex;
                if !g_ex.is_empty() {
                    let la = label_aggs.entry(*lbl).or_default();
                    // Edge-weight mask contribution (fast)
                    let m_add = &g_ex & &closure_total_masks[*to_ex];
                    la.add_mask |= &m_add;

                    // Add map contributions
                    add_closure_contrib(&mut la.add_map, &closures[*to_ex], &g_ex);
                }

                // REM: for this same label, if s has default, subtract its default contribution
                if let Some((to_def, wdef)) = &nwa.states[*s].default {
                    let g_def = gate & wdef;
                    if !g_def.is_empty() {
                        let la = label_aggs.entry(*lbl).or_default();

                        let m_rem = &g_def & &closure_total_masks[*to_def];
                        la.rem_mask |= &m_rem;

                        add_closure_contrib(&mut la.rem_map, &closures[*to_def], &g_def);
                    }
                }
            }
        }

        // 4) Emit exception edges: Tlbl = Tdef - rem(lbl) + add(lbl)
        if !label_aggs.is_empty() {
            for (lbl, la) in label_aggs.drain() {
                // Skip if label effect equals default (no change)
                if la.add_map.is_empty() && la.rem_map.is_empty() {
                    continue;
                }

                // Build sorted vectors
                let rem_vec = map_to_sorted_vec(la.rem_map);
                let add_vec = map_to_sorted_vec(la.add_map);

                // Merge: def_vec - rem_vec + add_vec
                let out_vec = merge_def_rem_add(&def_vec, &rem_vec, &add_vec);
                if out_vec.is_empty() {
                    continue;
                }

                // Edge weight: def_edge_mask - rem_mask + add_mask
                let mut edge_w = def_edge_mask.clone();
                if !la.rem_mask.is_empty() {
                    edge_w -= &la.rem_mask;
                }
                if !la.add_mask.is_empty() {
                    edge_w |= &la.add_mask;
                }
                if edge_w.is_empty() {
                    continue;
                }

                // Intern target subset
                let out_subset = Subset::new(out_vec);
                let target_id = match subset_to_did.get(&out_subset) {
                    Some(id) => *id,
                    None => {
                        let nid = dwa.states.add_state();
                        subset_to_did.insert(out_subset.clone(), nid);
                        subsets_arena.push(out_subset);
                        q.push_back(nid);
                        nid
                    }
                };

                // Add labeled transition
                let _ = dwa.add_transition(did, lbl, target_id, edge_w);
            }
        }
    }

    dwa
}
