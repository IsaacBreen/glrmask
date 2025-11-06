// src/precompute4/weighted_automata/determinization.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::{StateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

impl NWA {
    /// Determinize to DWA using a "delta-patch" algorithm specialized for large alphabets with default edges.
    ///
    /// High-level plan:
    /// - Compute epsilon-closures gated by future-acceptance weights to prune dead paths.
    /// - For each NWA state, precompute a "macro signature":
    ///     - final macro weight: union of (ε-closure weight ∧ final[t]) across closure targets
    ///     - default macro step-vector: ε-closed default edge target with edge-weight intersected
    ///     - labeled macro step-vectors: for each label, ε-closed target with edge-weight intersected
    ///   Intern identical macros into "signatures".
    /// - For any macro step-vector, precompile by target-signature: Vec<(sig_id, weight)>.
    ///   Also precompute a "total mask" per step-vector: union of weights over all target signatures.
    /// - Determinization over subsets of signatures:
    ///     - Initial subset: ε-closure(start), grouped by signature id.
    ///     - For every determinized state (subset):
    ///         • Compute default target once by aggregating per-pointer (step-vector) contributions
    ///           using the subset’s gate weights.
    ///         • Collect, for each label that appears as an exception in any signature within the subset:
    ///             - overrides per default pointer (to subtract from base default)
    ///             - exceptions per exception pointer (to add)
    ///           Use per-pointer compiled vectors to build a small "patch" (remove-map and add-map).
    ///           Apply this patch to the default target in O(|def| + |patch|) once per unique patch-shape,
    ///           and cache the result; reuse for every label sharing the same patch-shape.
    ///         • Create the actual DWA edges:
    ///             - A single default edge (if non-empty).
    ///             - Exception edges for labels with non-trivial patch (or whose result differs from default).
    ///
    /// This design avoids recomputing large unions for every label and every state by:
    /// - Computing the "base" default once per subset,
    /// - Modifying it by sparse patches only for labels that actually matter,
    /// - Caching patch results per subset for reuse across many labels with identical exception patterns.
    pub fn determinize_to_dwa(&self) -> DWA {
        let mut nwa_clone = self.clone();
        let now = Instant::now();
        crate::debug!(4, "Determinizing NWA with {} states...", nwa_clone.states.len());

        // Pre-simplification (safe) to trim low-value epsilon-chains and remove dead code
        nwa_clone.simplify();
        crate::debug!(4, "NWA simplified to {} states.", nwa_clone.states.len());

        let result = nwa_clone.internal_determinize_to_dwa_delta();

        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }

    fn internal_determinize_to_dwa_delta(&self) -> DWA {
        type StepVec = Arc<Vec<(NWAStateID, Weight)>>;

        // Intern any Vec<(NWAStateID, Weight)> by content with a lightweight fingerprint to avoid duplicates.
        #[derive(Clone)]
        struct StepVecKey {
            entries: StepVec,
            fp: u64,
        }
        impl StepVecKey {
            fn new(entries: StepVec) -> Self {
                let mut fp = FP_ZERO;
                for (sid, w) in entries.iter() {
                    fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
                }
                StepVecKey { entries, fp }
            }
        }
        impl PartialEq for StepVecKey {
            fn eq(&self, other: &Self) -> bool {
                if self.fp != other.fp {
                    return false;
                }
                Arc::ptr_eq(&self.entries, &other.entries) || *self.entries == *other.entries
            }
        }
        impl Eq for StepVecKey {}
        impl Hash for StepVecKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                state.write_u64(self.fp);
            }
        }
        fn intern_step_vec(
            v: Vec<(NWAStateID, Weight)>,
            intern: &mut HashMap<StepVecKey, StepVec>,
        ) -> StepVec {
            let arc = Arc::new(v);
            let key = StepVecKey::new(arc.clone());
            match intern.entry(key) {
                Entry::Occupied(o) => o.get().clone(),
                Entry::Vacant(v) => {
                    v.insert(arc.clone());
                    arc
                }
            }
        }

        // Compute future acceptance masks to prune epsilon closures
        let fut: Vec<Weight> = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        // Weighted ε-closure starting from a set of sources, pruned by "fut".
        fn compute_eps_mask_from_sources(
            sources: &[NWAStateID],
            states: &NWAStates,
            fut: &[Weight],
        ) -> Vec<(NWAStateID, Weight)> {
            let mut res: HashMap<NWAStateID, Weight> = HashMap::new();
            let mut q: VecDeque<NWAStateID> = VecDeque::new();
            for &s in sources {
                let f = fut[s].clone();
                if !f.is_empty() {
                    res.insert(s, f);
                    q.push_back(s);
                }
            }
            while let Some(u) = q.pop_front() {
                let u_w = res.get(&u).cloned().unwrap_or_else(Weight::zeros);
                if u_w.is_empty() {
                    continue;
                }
                for &(v, ref eps_w) in &states[u].epsilons {
                    let mut prop = &u_w & eps_w;
                    if prop.is_empty() {
                        continue;
                    }
                    prop &= &fut[v];
                    if prop.is_empty() {
                        continue;
                    }
                    match res.entry(v) {
                        Entry::Occupied(mut e) => {
                            let old = e.get_mut();
                            let new_union = &*old | &prop;
                            if new_union != *old {
                                *old = new_union;
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
            let mut vec_pairs: Vec<(NWAStateID, Weight)> = res.into_iter().collect();
            vec_pairs.sort_by_key(|(k, _)| *k);
            vec_pairs
        }

        // Precompute per-state ε-closure (gated) and intern it.
        let mut step_intern: HashMap<StepVecKey, StepVec> = HashMap::new();
        let mut eps_mask: Vec<StepVec> = Vec::with_capacity(n);
        for s in 0..n {
            let pairs = compute_eps_mask_from_sources(std::slice::from_ref(&s), &self.states, &fut);
            let sv = intern_step_vec(pairs, &mut step_intern);
            eps_mask.push(sv);
        }

        // Apply an additional weight across a step-vector (intersection), drop empties, intern
        fn apply_weight_to_stepvec(
            sv: &StepVec,
            w: &Weight,
            intern: &mut HashMap<StepVecKey, StepVec>,
        ) -> StepVec {
            if w.is_all_fast() {
                return sv.clone();
            }
            let mut out: Vec<(NWAStateID, Weight)> = Vec::with_capacity(sv.len());
            for (t, wt) in sv.iter() {
                let x = wt & w;
                if !x.is_empty() {
                    out.push((*t, x));
                }
            }
            intern_step_vec(out, intern)
        }

        // Macro final weights: F_macro[s] = ⋃_{t in eps-closure(s)} (closure_weight(s->*t) ∧ final[t])
        let mut f_macro: Vec<Option<Weight>> = vec![None; n];
        for s in 0..n {
            let mut acc: Option<Weight> = None;
            for (t, w) in eps_mask[s].iter() {
                if let Some(fw) = &self.states[*t].final_weight {
                    let c = w & fw;
                    if !c.is_empty() {
                        if let Some(a) = &mut acc {
                            *a |= &c;
                        } else {
                            acc = Some(c);
                        }
                    }
                }
            }
            f_macro[s] = acc;
        }

        // Macro step-vectors for defaults and labeled transitions
        let mut default_macro: Vec<Option<StepVec>> = vec![None; n];
        let mut labeled_macro: Vec<BTreeMap<i16, StepVec>> = vec![BTreeMap::new(); n];
        for s in 0..n {
            if let Some((to, wdef)) = &self.states[s].default {
                if *to < n {
                    let base = &eps_mask[*to];
                    let def_sv = apply_weight_to_stepvec(base, wdef, &mut step_intern);
                    default_macro[s] = Some(def_sv);
                }
            }
            for (lbl, (to, wlbl)) in &self.states[s].transitions {
                if *to < n {
                    let base = &eps_mask[*to];
                    let ex_sv = apply_weight_to_stepvec(base, wlbl, &mut step_intern);
                    labeled_macro[s].insert(*lbl, ex_sv);
                }
            }
        }

        // Macro signatures: (final_w, def_ptr, labeled map)
        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            def: Option<StepVec>,
            labeled: BTreeMap<i16, StepVec>,
        }
        #[derive(Clone)]
        struct MacroSigKey {
            final_w: Option<Weight>,
            def_ptr: Option<usize>,
            labeled: Vec<(i16, usize)>,
            fp: u64,
        }
        impl MacroSigKey {
            fn from_sig_components(final_w: &Option<Weight>, def: &Option<StepVec>, labeled: &BTreeMap<i16, StepVec>) -> Self {
                let mut lvec: Vec<(i16, usize)> = Vec::with_capacity(labeled.len());
                for (k, v) in labeled {
                    lvec.push((*k, Arc::as_ptr(v) as usize));
                }
                let def_ptr = def.as_ref().map(|a| Arc::as_ptr(a) as usize);
                let mut fp = FP_ZERO;
                fp = mix3(fp, final_w.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO), 0xA55A_A55A_A55A_A55A);
                fp = mix3(fp, def_ptr.unwrap_or(0) as u64, 0x77);
                for (lbl, ptr) in &lvec {
                    fp = mix3(fp, (*lbl as u64).wrapping_mul(FP_K1), (*ptr as u64).wrapping_mul(FP_K2));
                }
                MacroSigKey { final_w: final_w.clone(), def_ptr, labeled: lvec, fp }
            }
        }
        impl PartialEq for MacroSigKey {
            fn eq(&self, other: &Self) -> bool {
                self.fp == other.fp
                    && self.final_w == other.final_w
                    && self.def_ptr == other.def_ptr
                    && self.labeled == other.labeled
            }
        }
        impl Eq for MacroSigKey {}
        impl Hash for MacroSigKey {
            fn hash<H: Hasher>(&self, h: &mut H) {
                h.write_u64(self.fp);
            }
        }

        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();
        let mut sig_arena: Vec<Arc<MacroSig>> = Vec::new();
        let mut state_to_sig_id: Vec<usize> = vec![0; n];

        for s in 0..n {
            let final_w = f_macro[s].clone();
            let def = default_macro[s].clone();
            let labeled = labeled_macro[s].clone();
            let key = MacroSigKey::from_sig_components(&final_w, &def, &labeled);
            let id = match sig_intern.entry(key) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let idx = sig_arena.len();
                    let sig = MacroSig { final_w, def, labeled };
                    sig_arena.push(Arc::new(sig));
                    v.insert(idx);
                    idx
                }
            };
            state_to_sig_id[s] = id;
        }
        let num_sigs = sig_arena.len();

        // Unique step-vectors across signatures
        type CompiledBySig = Arc<Vec<(usize, Weight)>>; // (target_signature_id, weight)
        let mut unique_stepvec_ptrs: BTreeSet<usize> = BTreeSet::new();
        for s in 0..n {
            if let Some(ref d) = default_macro[s] {
                unique_stepvec_ptrs.insert(Arc::as_ptr(d) as usize);
            }
            for (_, sv) in &labeled_macro[s] {
                unique_stepvec_ptrs.insert(Arc::as_ptr(sv) as usize);
            }
        }

        // Compile step-vectors "by signature": collapse per target signature, union weights
        fn compile_stepvec_by_sig(sv: &StepVec, state_to_sig_id: &[usize]) -> CompiledBySig {
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in sv.iter() {
                let sig = state_to_sig_id[*t];
                if let Some(old) = acc.get_mut(&sig) {
                    *old |= w;
                } else {
                    acc.insert(sig, w.clone());
                }
            }
            let mut v: Vec<(usize, Weight)> = acc.into_iter().collect();
            v.sort_by_key(|(k, _)| *k);
            Arc::new(v)
        }

        let mut stepvec_ptr_to_compiled: HashMap<usize, CompiledBySig> =
            HashMap::with_capacity(unique_stepvec_ptrs.len() * 2 + 1);
        let mut stepvec_ptr_to_totalmask: HashMap<usize, Weight> =
            HashMap::with_capacity(unique_stepvec_ptrs.len() * 2 + 1);

        for ptr in unique_stepvec_ptrs {
            // Reconstruct Arc<...> from pointer is not safe; we only store compiled by pointer ID.
            // We'll locate sv by scanning signatures (few times) to get an Arc ref; but this would be O(#sigs).
            // Instead, we collect svs in a small temporary list quickly by one pass over sigs:
        }
        // The above loop can't get back svs; we'll fill the maps in the next pass scanning all macro refs once.
        // A small helper to build/ensure compiled entries exist.
        let mut ensure_compiled = |sv: &StepVec| {
            let key = Arc::as_ptr(sv) as usize;
            if !stepvec_ptr_to_compiled.contains_key(&key) {
                let comp = compile_stepvec_by_sig(sv, &state_to_sig_id);
                // Compute total mask for edge-weight fast computation
                let mut mask = Weight::zeros();
                for (_, w) in comp.iter() {
                    mask |= w;
                }
                stepvec_ptr_to_totalmask.insert(key, mask);
                stepvec_ptr_to_compiled.insert(key, comp);
            }
        };
        for s in 0..n {
            if let Some(ref d) = default_macro[s] {
                ensure_compiled(d);
            }
            for (_, sv) in &labeled_macro[s] {
                ensure_compiled(sv);
            }
        }

        // Subset interning
        #[derive(Clone)]
        struct SigSubsetKey {
            entries: Arc<Vec<(usize, Weight)>>,
            fp: u64,
        }
        impl SigSubsetKey {
            fn new(entries: Arc<Vec<(usize, Weight)>>) -> Self {
                let mut fp = FP_ZERO;
                for (sid, w) in entries.iter() {
                    fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
                }
                SigSubsetKey { entries, fp }
            }
        }
        impl PartialEq for SigSubsetKey {
            fn eq(&self, other: &Self) -> bool {
                if self.fp != other.fp {
                    return false;
                }
                Arc::ptr_eq(&self.entries, &other.entries) || *self.entries == *other.entries
            }
        }
        impl Eq for SigSubsetKey {}
        impl Hash for SigSubsetKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                state.write_u64(self.fp);
            }
        }
        fn intern_sig_subset(
            mut items: Vec<(usize, Weight)>,
            intern: &mut HashMap<SigSubsetKey, Arc<Vec<(usize, Weight)>>>,
        ) -> Arc<Vec<(usize, Weight)>> {
            // Normalize/merge by sig_id (union weights), drop empties, sort by sig_id
            items.sort_by_key(|(sid, _)| *sid);
            let mut norm: Vec<(usize, Weight)> = Vec::with_capacity(items.len());
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
            let arc = Arc::new(norm);
            let key = SigSubsetKey::new(arc.clone());
            match intern.entry(key) {
                Entry::Occupied(o) => o.get().clone(),
                Entry::Vacant(v) => {
                    v.insert(arc.clone());
                    arc
                }
            }
        }

        // Build initial subset as start state's ε-closure grouped by signature
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_mask[self.body.start_state].iter() {
            let sig = state_to_sig_id[*t];
            if let Some(old) = init_map.get_mut(&sig) {
                *old |= w;
            } else {
                init_map.insert(sig, w.clone());
            }
        }
        let mut init_items: Vec<(usize, Weight)> = init_map.into_iter().collect();
        init_items.sort_by_key(|(k, _)| *k);

        let mut subset_intern: HashMap<SigSubsetKey, Arc<Vec<(usize, Weight)>>> = HashMap::new();
        let init_subset = intern_sig_subset(init_items, &mut subset_intern);

        // DWA construction
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let start_d_id = dwa.states.add_state();
        dwa.body.start_state = start_d_id;

        let mut subset_to_d_id: HashMap<SigSubsetKey, StateID> = HashMap::new();
        subset_to_d_id.insert(SigSubsetKey::new(init_subset.clone()), start_d_id);

        let mut worklist: VecDeque<SigSubsetKey> = VecDeque::new();
        worklist.push_back(SigSubsetKey::new(init_subset.clone()));

        // Progress bar intentionally disabled in this version for speed.
        let _pb = if PROGRESS_BAR_ENABLED {
            // no-op
            None::<()>
        } else {
            None::<()>
        };

        // Helper: union across entries to get an overall edge mask (fast with stepvec_totalmask)
        let mut union_weight_vec = |pairs: &[(usize, Weight)]| -> Weight {
            let mut w = Weight::zeros();
            for (_, ww) in pairs {
                w |= ww;
            }
            w
        };

        // Local struct for patch cache inside a determinized state
        #[derive(Clone)]
        struct PatchKey {
            // Pairs of (stepvec_ptr_id, Weight) sorted by ptr_id
            // ov: overrides to subtract (default pointers)
            // ex: exceptions to add (exception pointers)
            ov: Vec<(usize, Weight)>,
            ex: Vec<(usize, Weight)>,
            // Lightweight fingerprint for quick hash and early-out
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
            fn hash<H: Hasher>(&self, state: &mut H) {
                state.write_u64(self.fp);
            }
        }

        // BFS determinization
        while let Some(subset_key) = worklist.pop_front() {
            let d_id = *subset_to_d_id.get(&subset_key).unwrap();
            let subset: &[(usize, Weight)] = &subset_key.entries;

            // 1) Final weight for this DWA state
            let mut d_final: Option<Weight> = None;
            for (sig_id, gate) in subset.iter() {
                if let Some(fw) = &sig_arena[*sig_id].final_w {
                    let c = gate & fw;
                    if !c.is_empty() {
                        if let Some(a) = &mut d_final {
                            *a |= &c;
                        } else {
                            d_final = Some(c);
                        }
                    }
                }
            }
            dwa.states[d_id].final_weight = d_final;

            // 2) Default aggregation per step-vector pointer
            // def_ptr -> gate_weight (union across subset signatures whose def ptr == ptr)
            let mut def_ptr_to_gate: HashMap<usize, Weight> = HashMap::new();
            for (sig_id, gate) in subset.iter() {
                if let Some(ref def_sv) = sig_arena[*sig_id].def {
                    let ptr = Arc::as_ptr(def_sv) as usize;
                    match def_ptr_to_gate.entry(ptr) {
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

            // Build default target subset once: per target signature, union of (compiled_stepvec ∧ gate_ptr)
            let mut def_target_acc: HashMap<usize, Weight> = HashMap::new();
            for (ptr, gate) in def_ptr_to_gate.iter() {
                let comp = stepvec_ptr_to_compiled.get(ptr).expect("compiled stepvec");
                if gate.is_all_fast() {
                    for (t_sig, w) in comp.iter() {
                        if let Some(old) = def_target_acc.get_mut(t_sig) {
                            *old |= w;
                        } else {
                            def_target_acc.insert(*t_sig, w.clone());
                        }
                    }
                } else {
                    for (t_sig, w) in comp.iter() {
                        let x = w & gate;
                        if x.is_empty() {
                            continue;
                        }
                        if let Some(old) = def_target_acc.get_mut(t_sig) {
                            *old |= &x;
                        } else {
                            def_target_acc.insert(*t_sig, x);
                        }
                    }
                }
            }
            let mut def_items: Vec<(usize, Weight)> = def_target_acc.into_iter().collect();
            def_items.sort_by_key(|(k, _)| *k);
            let def_subset_arc = intern_sig_subset(def_items, &mut subset_intern);
            let def_key = SigSubsetKey::new(def_subset_arc.clone());

            let def_target_id_opt = if !def_subset_arc.is_empty() {
                // Default edge weight via total-masks, not by scanning targets
                let mut def_edge_weight = Weight::zeros();
                for (ptr, gate) in def_ptr_to_gate.iter() {
                    if let Some(total_mask) = stepvec_ptr_to_totalmask.get(ptr) {
                        let x = gate & total_mask;
                        def_edge_weight |= &x;
                    }
                }
                // Install default edge
                let def_target_id = if let Some(id) = subset_to_d_id.get(&def_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(def_key.clone(), nid);
                    worklist.push_back(def_key.clone());
                    nid
                };
                if !def_edge_weight.is_empty() {
                    let _ = dwa.set_default_transition(d_id, def_target_id, def_edge_weight.clone());
                }
                Some((def_target_id, def_edge_weight))
            } else {
                None
            };

            // 3) Build per-label patch aggregations:
            // For every signature's labeled transitions (lbl -> ex_ptr),
            //   ex_by_label[lbl][ex_ptr] |= gate(sig)
            //   if sig has def_ptr: ov_by_label[lbl][def_ptr] |= gate(sig)
            let mut ex_by_label: HashMap<i16, HashMap<usize, Weight>> = HashMap::new();
            let mut ov_by_label: HashMap<i16, HashMap<usize, Weight>> = HashMap::new();

            let mut labels_union: BTreeSet<i16> = BTreeSet::new();
            for (sig_id, gate) in subset.iter() {
                let sig = &sig_arena[*sig_id];
                let def_ptr_opt = sig.def.as_ref().map(|a| Arc::as_ptr(a) as usize);
                for (lbl, sv) in sig.labeled.iter() {
                    labels_union.insert(*lbl);
                    // exceptions to add
                    let ex_ptr = Arc::as_ptr(sv) as usize;
                    match ex_by_label.entry(*lbl) {
                        Entry::Occupied(mut e) => {
                            let m = e.get_mut();
                            match m.entry(ex_ptr) {
                                Entry::Occupied(mut w) => {
                                    let ww = w.get_mut();
                                    *ww |= gate;
                                }
                                Entry::Vacant(w) => {
                                    w.insert(gate.clone());
                                }
                            }
                        }
                        Entry::Vacant(e) => {
                            let mut m = HashMap::new();
                            m.insert(ex_ptr, gate.clone());
                            e.insert(m);
                        }
                    }
                    // overrides to subtract (only if default exists)
                    if let Some(def_ptr) = def_ptr_opt {
                        match ov_by_label.entry(*lbl) {
                            Entry::Occupied(mut e) => {
                                let m = e.get_mut();
                                match m.entry(def_ptr) {
                                    Entry::Occupied(mut w) => {
                                        let ww = w.get_mut();
                                        *ww |= gate;
                                    }
                                    Entry::Vacant(w) => {
                                        w.insert(gate.clone());
                                    }
                                }
                            }
                            Entry::Vacant(e) => {
                                let mut m = HashMap::new();
                                m.insert(def_ptr, gate.clone());
                                e.insert(m);
                            }
                        }
                    }
                }
            }

            // If no labels at all, continue to next state
            if labels_union.is_empty() {
                continue;
            }

            // Cache of patch results within this DWA state (subset)
            // PatchKey -> (target_subset_arc, edge_weight)
            let mut patch_cache: HashMap<PatchKey, (Arc<Vec<(usize, Weight)>>, Weight)> = HashMap::new();

            // 4) For each label, compute patched target from default + sparse patch (and cache per patch-shape)
            for lbl in labels_union {
                // Build patch lists
                let mut ov_list: Vec<(usize, Weight)> = Vec::new();
                let mut ex_list: Vec<(usize, Weight)> = Vec::new();
                if let Some(m) = ov_by_label.get(&lbl) {
                    for (ptr, w) in m.iter() {
                        if !w.is_empty() {
                            ov_list.push((*ptr, w.clone()));
                        }
                    }
                }
                if let Some(m) = ex_by_label.get(&lbl) {
                    for (ptr, w) in m.iter() {
                        if !w.is_empty() {
                            ex_list.push((*ptr, w.clone()));
                        }
                    }
                }
                // Quick skip if patch does nothing and no default exists: then no outgoing edge on this label
                if ov_list.is_empty() && ex_list.is_empty() {
                    continue;
                }

                let pkey = PatchKey::new(ov_list.clone(), ex_list.clone());

                // Lookup in patch cache
                let (label_subset_arc, label_edge_weight) = if let Some((arc, ew)) = patch_cache.get(&pkey) {
                    (arc.clone(), ew.clone())
                } else {
                    // Build removal/add maps by target signature
                    let mut rem_acc: HashMap<usize, Weight> = HashMap::new();
                    let mut add_acc: HashMap<usize, Weight> = HashMap::new();

                    // Edge-weight accumulators using precomputed stepvec total-mask
                    let mut remove_edge_mask = Weight::zeros();
                    let mut add_edge_mask = Weight::zeros();

                    for (ptr, gate) in ov_list.iter() {
                        if gate.is_empty() {
                            continue;
                        }
                        if let Some(comp) = stepvec_ptr_to_compiled.get(ptr) {
                            for (t_sig, w) in comp.iter() {
                                let x = w & gate;
                                if x.is_empty() {
                                    continue;
                                }
                                if let Some(old) = rem_acc.get_mut(t_sig) {
                                    *old |= &x;
                                } else {
                                    rem_acc.insert(*t_sig, x);
                                }
                            }
                        }
                        if let Some(mask) = stepvec_ptr_to_totalmask.get(ptr) {
                            let x = gate & mask;
                            remove_edge_mask |= &x;
                        }
                    }
                    for (ptr, gate) in ex_list.iter() {
                        if gate.is_empty() {
                            continue;
                        }
                        if let Some(comp) = stepvec_ptr_to_compiled.get(ptr) {
                            for (t_sig, w) in comp.iter() {
                                let x = w & gate;
                                if x.is_empty() {
                                    continue;
                                }
                                if let Some(old) = add_acc.get_mut(t_sig) {
                                    *old |= &x;
                                } else {
                                    add_acc.insert(*t_sig, x);
                                }
                            }
                        }
                        if let Some(mask) = stepvec_ptr_to_totalmask.get(ptr) {
                            let x = gate & mask;
                            add_edge_mask |= &x;
                        }
                    }

                    // Merge three sorted lists: def_vec, rem_vec, add_vec
                    // Convert to sorted vectors
                    let mut def_it = def_subset_arc.iter().cloned().peekable();
                    let mut rem_vec: Vec<(usize, Weight)> = rem_acc.into_iter().collect();
                    let mut add_vec: Vec<(usize, Weight)> = add_acc.into_iter().collect();
                    rem_vec.sort_by_key(|(k, _)| *k);
                    add_vec.sort_by_key(|(k, _)| *k);
                    let mut rem_it = rem_vec.into_iter().peekable();
                    let mut add_it = add_vec.into_iter().peekable();

                    let mut out: Vec<(usize, Weight)> = Vec::new();
                    while def_it.peek().is_some() || rem_it.peek().is_some() || add_it.peek().is_some() {
                        // Determine next target id
                        let mut next_id = usize::MAX;
                        if let Some((tid, _)) = def_it.peek() {
                            next_id = next_id.min(*tid);
                        }
                        if let Some((tid, _)) = rem_it.peek() {
                            next_id = next_id.min(*tid);
                        }
                        if let Some((tid, _)) = add_it.peek() {
                            next_id = next_id.min(*tid);
                        }
                        let mut base_w = if let Some((tid, w)) = def_it.peek() {
                            if *tid == next_id {
                                let w_cl = w.clone();
                                def_it.next();
                                w_cl
                            } else {
                                Weight::zeros()
                            }
                        } else {
                            Weight::zeros()
                        };

                        // Subtract removal if any
                        if let Some((tid, rw)) = rem_it.peek() {
                            if *tid == next_id {
                                let rw_cl = rw.clone();
                                rem_it.next();
                                base_w -= &rw_cl;
                            }
                        }

                        // Add exception if any
                        if let Some((tid, aw)) = add_it.peek() {
                            if *tid == next_id {
                                let aw_cl = aw.clone();
                                add_it.next();
                                base_w |= &aw_cl;
                            }
                        }

                        if !base_w.is_empty() {
                            out.push((next_id, base_w));
                        }
                    }

                    let label_subset_arc = intern_sig_subset(out, &mut subset_intern);

                    // Edge weight: start from default edge weight, subtract removal_mask, add add_mask
                    let mut label_edge_weight = if let Some((_def_tid, ref def_w)) = def_target_id_opt {
                        def_w.clone()
                    } else {
                        Weight::zeros()
                    };
                    label_edge_weight -= &remove_edge_mask;
                    label_edge_weight |= &add_edge_mask;

                    patch_cache.insert(pkey.clone(), (label_subset_arc.clone(), label_edge_weight.clone()));
                    (label_subset_arc, label_edge_weight)
                };

                // Install labeled transition if it differs from default (or if there is no default)
                if label_subset_arc.is_empty() {
                    continue;
                }
                let target_key = SigSubsetKey::new(label_subset_arc.clone());
                let target_id = if let Some(id) = subset_to_d_id.get(&target_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(target_key.clone(), nid);
                    worklist.push_back(target_key);
                    nid
                };
                if !label_edge_weight.is_empty() {
                    let _ = dwa.add_transition(d_id, lbl, target_id, label_edge_weight);
                }
            }
        }

        dwa
    }
}
