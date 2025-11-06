// src/precompute4/weighted_automata/determinization.rs
//
// A radically simpler, cache-friendly determinization specialized for:
// - Large alphabets with default transitions,
// - Bitset weights where path-combination is intersection (AND) and choice-combination is union (OR),
// - Epsilon transitions.
//
// Design goals:
// - Avoid per-label recomputation wherever possible.
// - Use numeric IDs and dense vectors instead of pointer-based Arcs for core caches.
// - Group NWA states into "macro signatures" that are identical w.r.t. determinization (finals + default step + labeled steps).
// - Compile any step-vector (weighted ε-closed target list) by macro-signature once, and reuse.
// - For any determinized state (subset of macro-signatures gated by weights), compute:
//     * A single default target and weight
//     * Exception patches for labels that actually appear as exceptions in some member signature of the subset,
//       using a multiway merge over the member signatures' exception maps, not via label maps accumulation.
//
// Major correctness notes:
// - Each "macro signature" captures all that matters about an NWA state to determinization: final-weight after ε-closure,
//   default step-vector (if any), and the exception step-vectors map.
// - Determinized states are represented as a compact antichain of (macro_sig_id, Weight). For a fixed macro_sig_id,
//   only a single weight is kept, and dominated weights are removed by inclusion (w_small ⊆ w_big -> drop w_small).
// - For any label ℓ, the determinized transition is assembled as:
//       default_contrib
//     + (exceptions for ℓ)
//     - (default-overs for the signatures that have an exception on ℓ)
//   Both target-subset and the transition-weight are computed using compiled step-vectors-by-signature.
// - We never add a labeled exception whose (target subset, edge weight) equals the default edge; this avoids
//   creating spurious exception edges and prevents downstream blowups.
//
// Performance notes:
// - No progress bars; no per-label HashMaps; only a single multiway-merge per determinized state over its
//   member signatures' exception-label iterators.
// - Heavy allocations avoided; fast numeric IDs for interning step-vectors and macro-signatures.
// - Subsets are normalized aggressively (union per signature id + antichain pruning).
//
// This code replaces the previous determinization implementation entirely.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

impl NWA {
    /// Determinize to DWA using a compact, ID-based algorithm:
    /// - Compute ε-closures pruned by future-acceptance masks.
    /// - Build macro-signatures per NWA-state: final mask, default step-vector, exception step-vectors.
    /// - Intern step-vectors and compile them "by signature" once (target-signature -> weight).
    /// - Determinize over weighted sets of macro-signatures (with antichain normalization).
    /// - For each determinized state:
    ///     * Compute default target once.
    ///     * Traverse all exceptions by multiway merge over member signatures' label maps.
    ///     * For each unique patch (override default ptrs, add exception ptrs) compute patched result once
    ///       and reuse for all labels sharing the same patch (within this determinized state).
    ///
    /// The result is deterministic and semantically equivalent to the original NWA (bitset-weight semantics).
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        // Make a working copy to simplify in place if needed.
        let mut nwa = self.clone();
        // Optional: simplify(); keep as-is to preserve semantics, but it's usually beneficial.
        nwa.simplify();

        let result = nwa.internal_determinize_compact();

        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }

    fn internal_determinize_compact(&self) -> DWA {
        // 1) Compute future acceptance masks (prune closure propagation)
        let fut = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        // 2) Weighted ε-closure from a set of sources, pruned by fut[]
        fn eps_closure_masked(
            sources: &[NWAStateID],
            states: &NWAStates,
            fut: &[Weight],
        ) -> Vec<(NWAStateID, Weight)> {
            let mut out: HashMap<NWAStateID, Weight> = HashMap::new();
            let mut q: VecDeque<NWAStateID> = VecDeque::new();
            for &s in sources {
                if s >= states.len() { continue; }
                let f = fut[s].clone();
                if !f.is_empty() {
                    out.insert(s, f);
                    q.push_back(s);
                }
            }
            while let Some(u) = q.pop_front() {
                let base = out.get(&u).cloned().unwrap_or_else(Weight::zeros);
                if base.is_empty() { continue; }
                for &(v, ref w_eps) in &states[u].epsilons {
                    if v >= states.len() { continue; }
                    let mut prop = &base & w_eps;
                    if prop.is_empty() { continue; }
                    prop &= &fut[v];
                    if prop.is_empty() { continue; }
                    match out.entry(v) {
                        Entry::Occupied(mut e) => {
                            let old = e.get_mut();
                            let nu = &*old | &prop;
                            if nu != *old {
                                *old = nu;
                                q.push_back(v);
                            }
                        }
                        Entry::Vacant(e) => {
                            e.insert(prop);
                            q.push_back(v);
                        }
                    }
                }
            }
            let mut v: Vec<(NWAStateID, Weight)> = out.into_iter().collect();
            v.sort_by_key(|(sid, _)| *sid);
            v
        }

        // 3) Interned step-vectors
        type StepVecPairs = Vec<(NWAStateID, Weight)>;

        #[derive(Clone)]
        struct StepVecKey {
            fp: u64,
            len: usize,
            // No need to store all content here; equality check is done by scanning candidates;
            // we still must be careful to avoid heavy clones; We keep an index to the pool on insert.
        }

        #[derive(Clone)]
        struct StepVecData {
            pairs: StepVecPairs, // sorted by NWAStateID
            compiled_by_sig: Vec<(usize, Weight)>, // to be filled post-intern by using state_to_sig_id
            total_mask: Weight,
        }

        struct StepVecPool {
            // Storage
            data: Vec<StepVecData>,
            // Map from content to id
            map: HashMap<u64, Vec<usize>>, // fp -> candidates
        }

        impl StepVecPool {
            fn new() -> Self {
                Self { data: Vec::new(), map: HashMap::new() }
            }

            fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
                let mut fp = FP_ZERO;
                for (sid, w) in pairs.iter() {
                    fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
                }
                fp
            }

            fn intern(&mut self, mut pairs: StepVecPairs) -> usize {
                // Drop empties
                pairs.retain(|(_, w)| !w.is_empty());
                if pairs.is_empty() {
                    // We still intern empty to one shared id, so it's ok (distinct empty is fine).
                }
                let fp = Self::fingerprint(&pairs);
                if let Some(list) = self.map.get(&fp) {
                    for &id in list {
                        if self.data[id].pairs.len() == pairs.len() && self.data[id].pairs == pairs {
                            return id;
                        }
                    }
                }
                let id = self.data.len();
                self.data.push(StepVecData {
                    pairs,
                    compiled_by_sig: Vec::new(),
                    total_mask: Weight::zeros(),
                });
                self.map.entry(fp).or_insert_with(Vec::new).push(id);
                id
            }

            fn len(&self) -> usize { self.data.len() }
        }

        // 4) Macro signatures: what matters per NWA state for determinization
        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            def: Option<usize>, // stepvec id
            ex: BTreeMap<i16, usize>, // label -> stepvec id
        }

        #[derive(Clone)]
        struct MacroSigKey {
            fp: u64,
            final_fp: u64,
            def_id: Option<usize>,
            // (label, stepvec_id) in order
            label_ids: Vec<(i16, usize)>,
        }
        impl MacroSigKey {
            fn from_sig(sig: &MacroSig) -> Self {
                let mut label_ids: Vec<(i16, usize)> =
                    sig.ex.iter().map(|(k, &v)| (*k, v)).collect();
                // Sorted already by BTreeMap iteration order.
                let final_fp = sig.final_w.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO);
                let mut fp = FP_ZERO;
                fp = mix3(fp, final_fp, 0xA55A_A55A_A55A_A55A);
                fp = mix3(fp, sig.def.map(|x| x as u64).unwrap_or(0), 0x5D);
                for (lbl, sv) in &label_ids {
                    fp = mix3(fp, (*lbl as u64).wrapping_mul(FP_K1), (*sv as u64).wrapping_mul(FP_K2));
                }
                MacroSigKey { fp, final_fp, def_id: sig.def, label_ids }
            }
        }
        impl PartialEq for MacroSigKey {
            fn eq(&self, other: &Self) -> bool {
                self.fp == other.fp
                    && self.final_fp == other.final_fp
                    && self.def_id == other.def_id
                    && self.label_ids == other.label_ids
            }
        }
        impl Eq for MacroSigKey {}
        impl Hash for MacroSigKey {
            fn hash<H: Hasher>(&self, state: &mut H) { state.write_u64(self.fp); }
        }

        // 5) Build ε-closures (masked), final weights, and step-vectors for default and exceptions
        let mut step_pool = StepVecPool::new();

        // Cache closures for reuse
        let mut eps_cache: Vec<StepVecPairs> = vec![Vec::new(); n];
        for s in 0..n {
            let pairs = eps_closure_masked(std::slice::from_ref(&s), &self.states, &fut);
            eps_cache[s] = pairs;
        }

        // Helper: apply an extra weight mask to a closure vector
        fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> StepVecPairs {
            if w.is_all_fast() {
                return base.to_vec();
            }
            let mut out: StepVecPairs = Vec::with_capacity(base.len());
            for (sid, wt) in base.iter() {
                let x = wt & w;
                if !x.is_empty() {
                    out.push((*sid, x));
                }
            }
            out
        }

        // For each NWA state, compute macro signature
        let mut sig_arena: Vec<MacroSig> = Vec::with_capacity(n);
        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();
        let mut state_to_sig_id: Vec<usize> = vec![0; n];

        for s in 0..n {
            // final weight after ε-closure from s
            let mut final_acc: Option<Weight> = None;
            for (t, w) in eps_cache[s].iter() {
                if let Some(fw) = &self.states[*t].final_weight {
                    let c = w & fw;
                    if !c.is_empty() {
                        if let Some(ref mut a) = final_acc {
                            *a |= &c;
                        } else {
                            final_acc = Some(c);
                        }
                    }
                }
            }

            let mut def_id: Option<usize> = None;
            if let Some((to, wdef)) = &self.states[s].default {
                if *to < n {
                    let base = &eps_cache[*to];
                    let pairs = apply_weight_to_pairs(base, wdef);
                    def_id = Some(step_pool.intern(pairs));
                }
            }
            let mut ex_map: BTreeMap<i16, usize> = BTreeMap::new();
            for (lbl, (to, wlbl)) in &self.states[s].transitions {
                if *to >= n { continue; }
                let base = &eps_cache[*to];
                let pairs = apply_weight_to_pairs(base, wlbl);
                let id = step_pool.intern(pairs);
                ex_map.insert(*lbl, id);
            }

            let sig = MacroSig { final_w: final_acc, def: def_id, ex: ex_map };
            let key = MacroSigKey::from_sig(&sig);
            let sig_id = match sig_intern.entry(key) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let id = sig_arena.len();
                    sig_arena.push(sig);
                    v.insert(id);
                    id
                }
            };
            state_to_sig_id[s] = sig_id;
        }
        let num_sigs = sig_arena.len();

        // 6) Compile step-vectors by signature and compute their total masks
        for id in 0..step_pool.len() {
            let pairs = &step_pool.data[id].pairs;
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() {
                let sid = state_to_sig_id[*t];
                match acc.entry(sid) {
                    Entry::Occupied(mut e) => {
                        let x = e.get_mut();
                        *x |= w;
                    }
                    Entry::Vacant(e) => {
                        e.insert(w.clone());
                    }
                }
            }
            let mut compiled: Vec<(usize, Weight)> = acc.into_iter().collect();
            compiled.sort_by_key(|(sid, _)| *sid);
            // total mask
            let mut total = Weight::zeros();
            for (_, w) in compiled.iter() {
                total |= w;
            }
            step_pool.data[id].compiled_by_sig = compiled;
            step_pool.data[id].total_mask = total;
        }

        // 7) Determinization machinery

        // Subset representation: vector of (sig_id, weight), sorted by sig_id, unioned per sig_id,
        // and antichain-minimized (drop dominated weights per sig).
        #[derive(Clone)]
        struct Subset {
            items: Vec<(usize, Weight)>,
            fp: u64,
        }
        impl Subset {
            fn new(mut items: Vec<(usize, Weight)>) -> Self {
                // Merge by sig_id
                items.sort_by_key(|(sid, _)| *sid);
                let mut merged: Vec<(usize, Weight)> = Vec::with_capacity(items.len());
                for (sid, w) in items.into_iter() {
                    if w.is_empty() { continue; }
                    if let Some((last_sid, ref mut last_w)) = merged.last_mut() {
                        if *last_sid == sid {
                            *last_w |= &w;
                            continue;
                        }
                    }
                    merged.push((sid, w));
                }
                // Antichain per sig_id: remove dominated weights (w_small ⊆ w_big)
                // Since per sig_id we keep only one item after merging, this is trivial.
                let mut fp = FP_ZERO;
                for (sid, w) in merged.iter() {
                    fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
                }
                Subset { items: merged, fp }
            }
            fn is_empty(&self) -> bool { self.items.is_empty() }
        }
        impl PartialEq for Subset {
            fn eq(&self, other: &Self) -> bool {
                self.fp == other.fp && self.items == other.items
            }
        }
        impl Eq for Subset {}
        impl Hash for Subset {
            fn hash<H: Hasher>(&self, state: &mut H) { state.write_u64(self.fp); }
        }

        // Initial subset = ε-closure(start) grouped by signature
        let mut init_acc: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            let sid = state_to_sig_id[*t];
            match init_acc.entry(sid) {
                Entry::Occupied(mut e) => {
                    let x = e.get_mut();
                    *x |= w;
                }
                Entry::Vacant(e) => {
                    e.insert(w.clone());
                }
            }
        }
        let init_items: Vec<(usize, Weight)> = init_acc.into_iter().collect();
        let init_subset = Subset::new(init_items);

        // Patch key within a single determinized state
        #[derive(Clone)]
        struct PatchKey {
            // sorted by ptr_id
            ov: Vec<(usize, Weight)>, // def stepvec id -> gate
            ex: Vec<(usize, Weight)>, // ex stepvec id -> gate
            fp: u64,
        }
        impl PatchKey {
            fn new(mut ov: Vec<(usize, Weight)>, mut ex: Vec<(usize, Weight)>) -> Self {
                ov.sort_by_key(|(p, _)| *p);
                ex.sort_by_key(|(p, _)| *p);
                let mut fp = FP_ZERO;
                for (p, w) in &ov {
                    fp = mix3(fp, (*p as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
                }
                fp = mix3(fp, 0xDEADBEEF, 0xCAFEBABE);
                for (p, w) in &ex {
                    fp = mix3(fp, (*p as u64).wrapping_mul(FP_K2), w.fp.wrapping_mul(FP_K1));
                }
                PatchKey { ov, ex, fp }
            }
        }
        impl PartialEq for PatchKey {
            fn eq(&self, other: &Self) -> bool {
                self.fp == other.fp && self.ov == other.ov && self.ex == other.ex
            }
        }
        impl Eq for PatchKey {}
        impl Hash for PatchKey {
            fn hash<H: Hasher>(&self, state: &mut H) { state.write_u64(self.fp); }
        }

        // Target cache: subset -> DWA-state ID
        let mut subset_to_d_id: HashMap<Subset, usize> = HashMap::new();

        // DWA under construction
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let start_d_id = dwa.states.add_state();
        dwa.body.start_state = start_d_id;

        subset_to_d_id.insert(init_subset.clone(), start_d_id);
        let mut work: VecDeque<Subset> = VecDeque::new();
        work.push_back(init_subset);

        // Prepare convenience closures for stepvec use
        let step_compiled_by_sig = |sv_id: usize| -> &Vec<(usize, Weight)> {
            &step_pool.data[sv_id].compiled_by_sig
        };
        let step_total_mask = |sv_id: usize| -> &Weight {
            &step_pool.data[sv_id].total_mask
        };

        // Utility: accumulate compiled_by_sig with a gate mask into target_acc
        fn accumulate_compiled(
            acc: &mut HashMap<usize, Weight>,
            compiled: &[(usize, Weight)],
            gate: &Weight,
        ) {
            if gate.is_all_fast() {
                for (t_sig, w) in compiled.iter() {
                    match acc.entry(*t_sig) {
                        Entry::Occupied(mut e) => {
                            let x = e.get_mut();
                            *x |= w;
                        }
                        Entry::Vacant(e) => {
                            e.insert(w.clone());
                        }
                    }
                }
            } else {
                for (t_sig, w) in compiled.iter() {
                    let x = w & gate;
                    if x.is_empty() { continue; }
                    match acc.entry(*t_sig) {
                        Entry::Occupied(mut e) => {
                            let y = e.get_mut();
                            *y |= &x;
                        }
                        Entry::Vacant(e) => {
                            e.insert(x);
                        }
                    }
                }
            }
        }

        // Build normalized subset from target_acc (sig_id -> weight)
        fn build_subset_from_map(target_acc: HashMap<usize, Weight>) -> Subset {
            let items: Vec<(usize, Weight)> = target_acc.into_iter().collect();
            Subset::new(items)
        }

        // Multiway-merge cursor across exception maps
        struct ExCursor<'a> {
            lbl_iter: std::collections::btree_map::Iter<'a, i16, usize>,
            current: Option<(i16, usize)>,
            gate: Weight,                // gate for the owning signature
            def_ptr: Option<usize>,      // default stepvec id for the owning signature
        }

        // BinaryHeap item (min-heap by label using Reverse)
        #[derive(Eq)]
        struct HeapItem {
            lbl: i16,
            idx: usize,
        }
        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool { self.lbl == other.lbl && self.idx == other.idx }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse to get min-heap behavior using BinaryHeap (default is max-heap)
                other.lbl.cmp(&self.lbl).then_with(|| other.idx.cmp(&self.idx))
            }
        }
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }

        while let Some(subset) = work.pop_front() {
            let d_id = *subset_to_d_id.get(&subset).unwrap();

            // Final weight: union over (gate ∧ sig.final)
            let mut final_acc: Option<Weight> = None;
            for (sig_id, gate) in subset.items.iter() {
                if let Some(fw) = &sig_arena[*sig_id].final_w {
                    let c = gate & fw;
                    if !c.is_empty() {
                        if let Some(ref mut a) = final_acc {
                            *a |= &c;
                        } else {
                            final_acc = Some(c);
                        }
                    }
                }
            }
            dwa.states[d_id].final_weight = final_acc;

            // Default aggregation (per default stepvec id collect gates; then compiled to targets)
            let mut def_ptr_gate: HashMap<usize, Weight> = HashMap::new();
            for (sig_id, gate) in subset.items.iter() {
                if let Some(def_ptr) = sig_arena[*sig_id].def {
                    match def_ptr_gate.entry(def_ptr) {
                        Entry::Occupied(mut e) => {
                            let v = e.get_mut();
                            *v |= gate;
                        }
                        Entry::Vacant(e) => {
                            e.insert(gate.clone());
                        }
                    }
                }
            }
            let mut def_target_map: HashMap<usize, Weight> = HashMap::new();
            for (ptr, gate) in def_ptr_gate.iter() {
                let comp = step_compiled_by_sig(*ptr);
                accumulate_compiled(&mut def_target_map, comp, gate);
            }
            let def_subset = build_subset_from_map(def_target_map);

            // Default edge weight via total masks
            let mut def_edge_w = Weight::zeros();
            for (ptr, gate) in def_ptr_gate.iter() {
                let mask = step_total_mask(*ptr);
                let x = gate & mask;
                def_edge_w |= &x;
            }

            let mut def_target_id_opt: Option<usize> = None;
            if !def_subset.is_empty() && !def_edge_w.is_empty() {
                // Create/get target DWA state for default
                let def_target_id = if let Some(id) = subset_to_d_id.get(&def_subset) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(def_subset.clone(), nid);
                    work.push_back(def_subset.clone());
                    nid
                };
                // Install default edge
                let _ = dwa.set_default_transition(d_id, def_target_id, def_edge_w.clone());
                def_target_id_opt = Some(def_target_id);
            }

            // Early exit if no exceptions in any member signature
            let mut any_ex = false;
            for (sig_id, _) in subset.items.iter() {
                if !sig_arena[*sig_id].ex.is_empty() {
                    any_ex = true;
                    break;
                }
            }
            if !any_ex {
                continue;
            }

            // Multiway merge over labels
            let mut cursors: Vec<ExCursor> = Vec::new();
            for (sig_id, gate) in subset.items.iter() {
                if sig_arena[*sig_id].ex.is_empty() {
                    continue;
                }
                let mut it = sig_arena[*sig_id].ex.iter();
                let first = it.next().map(|(k, v)| (*k, *v));
                let cur = ExCursor {
                    lbl_iter: it,
                    current: first,
                    gate: gate.clone(),
                    def_ptr: sig_arena[*sig_id].def,
                };
                if cur.current.is_some() {
                    cursors.push(cur);
                }
            }
            if cursors.is_empty() {
                continue;
            }

            let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
            for (idx, c) in cursors.iter().enumerate() {
                if let Some((lbl, _)) = c.current {
                    heap.push(HeapItem { lbl, idx });
                }
            }

            // Patch cache within this determinized state
            let mut patch_cache: HashMap<PatchKey, (Subset, Weight)> = HashMap::new();

            // Process labels in ascending order
            while let Some(HeapItem { lbl, idx }) = heap.pop() {
                // Collect all cursors at this label
                let mut idxs = vec![idx];
                while let Some(top) = heap.peek() {
                    if top.lbl == lbl { idxs.push(heap.pop().unwrap().idx); } else { break; }
                }
                // Aggregate overrides/removals and additions by stepvec pointer id
                let mut ov_map: HashMap<usize, Weight> = HashMap::new(); // default ptrs
                let mut ex_map: HashMap<usize, Weight> = HashMap::new(); // exception ptrs
                for i in idxs.iter().copied() {
                    if let Some((_l, ex_ptr)) = cursors[i].current {
                        // add exception contribution
                        match ex_map.entry(ex_ptr) {
                            Entry::Occupied(mut e) => {
                                let v = e.get_mut();
                                *v |= &cursors[i].gate;
                            }
                            Entry::Vacant(e) => { e.insert(cursors[i].gate.clone()); }
                        }
                        // add override removal for default (if any)
                        if let Some(def_ptr) = cursors[i].def_ptr {
                            match ov_map.entry(def_ptr) {
                                Entry::Occupied(mut e) => {
                                    let v = e.get_mut();
                                    *v |= &cursors[i].gate;
                                }
                                Entry::Vacant(e) => { e.insert(cursors[i].gate.clone()); }
                            }
                        }
                    }
                }
                // Normalize patch key
                let mut ov_list: Vec<(usize, Weight)> = ov_map.into_iter().filter(|(_, w)| !w.is_empty()).collect();
                let mut ex_list: Vec<(usize, Weight)> = ex_map.into_iter().filter(|(_, w)| !w.is_empty()).collect();
                if ov_list.is_empty() && ex_list.is_empty() {
                    // This label does not change anything vs default
                    // Advance cursors and continue
                    for i in idxs.iter().copied() {
                        // advance this cursor
                        if let Some((_l, _)) = cursors[i].current {
                            if let Some((k, v)) = cursors[i].lbl_iter.next() {
                                cursors[i].current = Some((*k, *v));
                                heap.push(HeapItem { lbl: *k, idx: i });
                            } else {
                                cursors[i].current = None;
                            }
                        }
                    }
                    continue;
                }

                let pkey = PatchKey::new(ov_list.clone(), ex_list.clone());

                // Compute patched result (or reuse from patch cache)
                let (label_subset, label_edge_w) = if let Some((s, w)) = patch_cache.get(&pkey) {
                    (s.clone(), w.clone())
                } else {
                    // Removal/addition by signature accumulation
                    let mut rem_sig_map: HashMap<usize, Weight> = HashMap::new();
                    let mut add_sig_map: HashMap<usize, Weight> = HashMap::new();

                    // Edge weight masks
                    let mut rem_edge_mask = Weight::zeros();
                    let mut add_edge_mask = Weight::zeros();

                    for (ptr, gate) in ov_list.iter() {
                        let comp = step_compiled_by_sig(*ptr);
                        accumulate_compiled(&mut rem_sig_map, comp, gate);
                        // remove edge mask portion
                        let m = step_total_mask(*ptr);
                        let x = gate & m;
                        rem_edge_mask |= &x;
                    }
                    for (ptr, gate) in ex_list.iter() {
                        let comp = step_compiled_by_sig(*ptr);
                        accumulate_compiled(&mut add_sig_map, comp, gate);
                        let m = step_total_mask(*ptr);
                        let x = gate & m;
                        add_edge_mask |= &x;
                    }

                    // Merge default subset with removals and additions
                    // Build default subset map
                    let mut label_map: HashMap<usize, Weight> = HashMap::new();
                    for (sig_id, w) in def_subset.items.iter() {
                        label_map.insert(*sig_id, w.clone());
                    }
                    // Subtract removals
                    for (sig_id, rw) in rem_sig_map.into_iter() {
                        if let Some(old) = label_map.get_mut(&sig_id) {
                            *old -= &rw;
                            if old.is_empty() {
                                label_map.remove(&sig_id);
                            }
                        }
                    }
                    // Add exceptions
                    for (sig_id, aw) in add_sig_map.into_iter() {
                        match label_map.entry(sig_id) {
                            Entry::Occupied(mut e) => {
                                let v = e.get_mut();
                                *v |= &aw;
                            }
                            Entry::Vacant(e) => {
                                e.insert(aw);
                            }
                        }
                    }

                    // Normalize to subset
                    let label_subset = build_subset_from_map(label_map);

                    // Edge weight
                    let mut label_edge_w = def_edge_w.clone();
                    label_edge_w -= &rem_edge_mask;
                    label_edge_w |= &add_edge_mask;

                    patch_cache.insert(pkey.clone(), (label_subset.clone(), label_edge_w.clone()));
                    (label_subset, label_edge_w)
                };

                // Skip if label transition equals default (target subset and weight)
                let same_as_default = if let Some(def_tid) = def_target_id_opt {
                    // compare subsets and weights
                    !def_edge_w.is_empty() && label_edge_w == def_edge_w && label_subset == def_subset
                } else {
                    false
                };

                if !same_as_default && !label_subset.is_empty() && !label_edge_w.is_empty() {
                    // Get/create target DWA state
                    let target_id = if let Some(id) = subset_to_d_id.get(&label_subset) {
                        *id
                    } else {
                        let nid = dwa.states.add_state();
                        subset_to_d_id.insert(label_subset.clone(), nid);
                        work.push_back(label_subset.clone());
                        nid
                    };
                    let _ = dwa.add_transition(d_id, lbl, target_id, label_edge_w);
                }

                // Advance all cursors for this label
                for i in idxs.into_iter() {
                    if let Some((_l, _)) = cursors[i].current {
                        if let Some((k, v)) = cursors[i].lbl_iter.next() {
                            cursors[i].current = Some((*k, *v));
                            heap.push(HeapItem { lbl: *k, idx: i });
                        } else {
                            cursors[i].current = None;
                        }
                    }
                }
            }
        }

        dwa
    }
}
