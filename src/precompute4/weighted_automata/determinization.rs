// src/precompute4/weighted_automata/determinization.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::{StateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc};
use std::time::Instant;
impl NWA {
    /// Determinize the subgraph reachable from 'start' to a DWA via a radically simplified, high-performance construction:
    ///
    /// Design:
    /// - Precompute epsilon-closures once per NWA state, gated by future-acceptance masks to prune useless paths.
    /// - For each state and each outgoing (default or labeled) edge, precompute a "macro step-vector":
    ///     step_s(lbl): Vec<(target_state, weight)>, where weight already includes post-epsilon closure of the target.
    /// - Intern these step-vectors and group states by identical macro signatures (final macro weight + default macro + labeled macro set).
    ///   Determinization then operates on antichains of signature-IDs rather than raw states, collapsing many states at once.
    /// - At each determinized state, compute a single default edge and a set of exception edges; no "default subtraction" is needed,
    ///   since the DWA runtime semantics prefer exceptions over default on matching labels.
    ///
    /// Correctness sketch:
    /// - Closure correctness follows from semiring (∨, ∧) algebra: union across epsilon paths with ∧ along edges equals weighted ε-closure.
    /// - Macro steps include pre- and post-ε closures, thus matching the standard ε-removal construction.
    /// - Grouping by identical macro signatures preserves right languages: if signatures are equal, their residuals are identical for all words.
    /// - Determinization merges contributions by union; default vs exception precedence matches the NWA semantics because exceptions
    ///   use only explicit sources; default edge contains all sources (both with/without explicit), but at runtime default is ignored where exceptions exist.
    pub fn determinize_to_dwa(&self) -> DWA {
        let mut nwa_clone = self.clone();
        let now = Instant::now();
        crate::debug!(4, "Determinizing NWA with {} states...", nwa_clone.states.len());
        // Aggressive but safe simplification before determinization
        nwa_clone.simplify();
        crate::debug!(4, "NWA simplified to {} states.", nwa_clone.states.len());
        let result = nwa_clone.internal_determinize_to_dwa_fast();
        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }

    fn internal_determinize_to_dwa_fast(&self) -> DWA {
        type StepVec = Arc<Vec<(NWAStateID, Weight)>>;

        #[derive(Clone)]
        struct StepVecKey {
            entries: StepVec,
            fp: u64,
        }
        impl StepVecKey {
            fn new(entries: StepVec) -> Self {
                // robust fingerprint mix across (state, weight.fp)
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

        // Interner for any Vec<(sid, weight)>
        fn intern_step_vec(
            v: Vec<(NWAStateID, Weight)>,
            intern: &mut HashMap<StepVecKey, StepVec>
        ) -> StepVec {
            let arc = Arc::new(v);
            let key = StepVecKey::new(arc.clone());
            match intern.entry(key) {
                Entry::Occupied(o) => o.get().clone(),
                Entry::Vacant(vac) => {
                    vac.insert(arc.clone());
                    arc
                }
            }
        }

        // Compute future masks to gate epsilon closures
        let fut: Vec<Weight> = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        // Weighted epsilon-closure from a set of sources (used for per-state closure)
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
                if u_w.is_empty() { continue; }
                for &(v, ref eps_w) in &states[u].epsilons {
                    let mut prop = &u_w & eps_w;
                    if prop.is_empty() { continue; }
                    prop &= &fut[v];
                    if prop.is_empty() { continue; }
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

        // Precompute per-state epsilon-closure masks (gated by future weights)
        let mut step_intern: HashMap<StepVecKey, StepVec> = HashMap::new();
        let mut eps_mask: Vec<StepVec> = Vec::with_capacity(n);
        for s in 0..n {
            let pairs = compute_eps_mask_from_sources(std::slice::from_ref(&s), &self.states, &fut);
            let sv = intern_step_vec(pairs, &mut step_intern);
            eps_mask.push(sv);
        }

        // Utility: apply an additional weight w across a step-vector (intersect each entry)
        // Return the original arc if w is ALL, otherwise return an interned new arc without empties.
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

        // Precompute final macro weight F_macro[s] = ⋃_{t in eps-closure(s)} (closure_weight(s->*t) ∧ final[t])
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

        // Precompute macro step-vectors for default and labeled edges
        let mut default_macro: Vec<Option<StepVec>> = vec![None; n];
        let mut labeled_macro: Vec<BTreeMap<i16, StepVec>> = vec![BTreeMap::new(); n];

        for s in 0..n {
            // Default
            if let Some((to, wdef)) = &self.states[s].default {
                if *to < n {
                    let base = &eps_mask[*to];
                    let def_sv = apply_weight_to_stepvec(base, wdef, &mut step_intern);
                    default_macro[s] = Some(def_sv);
                }
            }
            // Labeled
            for (lbl, (to, wlbl)) in &self.states[s].transitions {
                if *to < n {
                    let base = &eps_mask[*to];
                    let ex_sv = apply_weight_to_stepvec(base, wlbl, &mut step_intern);
                    labeled_macro[s].insert(*lbl, ex_sv);
                }
            }
        }

        // Build macro signatures and intern them.
        // A signature is defined by: final macro weight, default macro stepvec pointer, and all labeled macro pairs (label -> stepvec pointer)
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
            // Lightweight hash precomputation
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

        // Precompile each StepVec into "by-signature" grouping to avoid per-target state lookups during determinization.
        type CompiledBySig = Arc<Vec<(usize, Weight)>>; // (sig_id, weight)
        let mut unique_stepvecs: BTreeSet<usize> = BTreeSet::new();
        for s in 0..n {
            if let Some(ref d) = default_macro[s] {
                unique_stepvecs.insert(Arc::as_ptr(d) as usize);
            }
            for (_, sv) in &labeled_macro[s] {
                unique_stepvecs.insert(Arc::as_ptr(sv) as usize);
            }
        }
        let mut stepvec_ptr_to_compiled: HashMap<usize, CompiledBySig> = HashMap::with_capacity(unique_stepvecs.len() * 2 + 1);

        fn compile_stepvec_by_sig(
            sv: &StepVec,
            state_to_sig_id: &[usize],
        ) -> CompiledBySig {
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

        for s in 0..n {
            if let Some(ref d) = default_macro[s] {
                let key = Arc::as_ptr(d) as usize;
                if !stepvec_ptr_to_compiled.contains_key(&key) {
                    stepvec_ptr_to_compiled.insert(key, compile_stepvec_by_sig(d, &state_to_sig_id));
                }
            }
            for (_, sv) in &labeled_macro[s] {
                let key = Arc::as_ptr(sv) as usize;
                if !stepvec_ptr_to_compiled.contains_key(&key) {
                    stepvec_ptr_to_compiled.insert(key, compile_stepvec_by_sig(sv, &state_to_sig_id));
                }
            }
        }

        // Determinization over signatures:
        // Subset representation: Vec<(sig_id, gate_weight)>, sorted and Interned.
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
            // Normalize: merge duplicates, drop empties, sort by sig_id
            items.sort_by_key(|(sid, _)| *sid);
            let mut norm: Vec<(usize, Weight)> = Vec::with_capacity(items.len());
            for (sid, w) in items.into_iter() {
                if w.is_empty() { continue; }
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
                Entry::Vacant(v) => { v.insert(arc.clone()); arc }
            }
        }

        // Build initial subset as the start state's epsilon-closure grouped by macro signatures.
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

        // Map subset -> DWA state id
        let mut subset_to_d_id: HashMap<SigSubsetKey, StateID> = HashMap::new();
        subset_to_d_id.insert(SigSubsetKey::new(init_subset.clone()), start_d_id);

        let mut worklist: VecDeque<SigSubsetKey> = VecDeque::new();
        worklist.push_back(SigSubsetKey::new(init_subset.clone()));

        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(1);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinizing DWA (fast): {elapsed_precise}] \
                               [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
                    .expect("progress-bar"),
            );
            Some(p)
        } else { None };

        let mut processed = 0usize;

        while let Some(subset_key) = worklist.pop_front() {
            processed += 1;
            if let Some(p) = &pb {
                p.set_position(processed as u64);
                p.set_length(subset_to_d_id.len() as u64);
            }

            let d_id = *subset_to_d_id.get(&subset_key).unwrap();
            let subset: &[(usize, Weight)] = &subset_key.entries;

            // Compute final weight of D-state: ⋃ gate ∧ final_macro[sig]
            let mut d_final: Option<Weight> = None;
            for (sig_id, gate) in subset.iter() {
                let fin = &sig_arena[*sig_id].final_w;
                if let Some(fw) = fin {
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
            dwa.set_final_weight(d_id, d_final).expect("set_final_weight");

            // Default edge: accumulate contributions across all signatures that have a default macro.
            let mut def_acc: HashMap<usize, Weight> = HashMap::new();
            for (sig_id, gate) in subset.iter() {
                if let Some(ref def_sv) = sig_arena[*sig_id].def {
                    let comp = stepvec_ptr_to_compiled[&(Arc::as_ptr(def_sv) as usize)].clone();
                    if gate.is_all_fast() {
                        for (t_sig, w) in comp.iter() {
                            if let Some(old) = def_acc.get_mut(t_sig) { *old |= w; } else { def_acc.insert(*t_sig, w.clone()); }
                        }
                    } else {
                        for (t_sig, w) in comp.iter() {
                            let x = w & gate;
                            if x.is_empty() { continue; }
                            if let Some(old) = def_acc.get_mut(t_sig) { *old |= &x; } else { def_acc.insert(*t_sig, x); }
                        }
                    }
                }
            }
            // Install default edge if any
            let mut def_target_id_opt: Option<StateID> = None;
            if !def_acc.is_empty() {
                let mut def_items: Vec<(usize, Weight)> = def_acc.into_iter().collect();
                def_items.sort_by_key(|(k, _)| *k);
                let def_subset_arc = intern_sig_subset(def_items, &mut subset_intern);
                let def_key = SigSubsetKey::new(def_subset_arc.clone());
                let def_target_id = if let Some(id) = subset_to_d_id.get(&def_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(def_key.clone(), nid);
                    worklist.push_back(def_key);
                    nid
                };
                // Edge weight: union of contributions across target signatures (values)
                let mut edge_w: Option<Weight> = None;
                for (_, w) in def_subset_arc.iter() {
                    if let Some(a) = &mut edge_w {
                        *a |= w;
                    } else {
                        edge_w = Some(w.clone());
                    }
                }
                if let Some(w) = edge_w {
                    dwa.set_default_transition(d_id, def_target_id, w).expect("set_default_transition");
                    def_target_id_opt = Some(def_target_id);
                }
            }

            // Exception labels: union across labels present in any signature within the subset
            let mut labels: BTreeSet<i16> = BTreeSet::new();
            for (sig_id, _) in subset.iter() {
                for k in sig_arena[*sig_id].labeled.keys() {
                    labels.insert(*k);
                }
            }
            for lbl in labels {
                // Accumulate across signatures that have this explicit label
                let mut acc: HashMap<usize, Weight> = HashMap::new();
                for (sig_id, gate) in subset.iter() {
                    let sig = &sig_arena[*sig_id];
                    // If an explicit transition for the label exists, use it. Otherwise, fall back to the default transition.
                    let sv_opt = sig.labeled.get(&lbl).or(sig.def.as_ref());

                    if let Some(ref sv) = sv_opt {
                        let comp = stepvec_ptr_to_compiled[&(Arc::as_ptr(sv) as usize)].clone();
                        if gate.is_all_fast() {
                            for (t_sig, w) in comp.iter() {
                                if let Some(old) = acc.get_mut(t_sig) { *old |= w; } else { acc.insert(*t_sig, w.clone()); }
                            }
                        } else {
                            for (t_sig, w) in comp.iter() {
                                let x = w & gate;
                                if x.is_empty() { continue; }
                                if let Some(old) = acc.get_mut(t_sig) { *old |= &x; } else { acc.insert(*t_sig, x); }
                            }
                        }
                    }
                }
                if acc.is_empty() {
                    continue;
                }
                let mut items: Vec<(usize, Weight)> = acc.into_iter().collect();
                items.sort_by_key(|(k, _)| *k);
                let target_subset_arc = intern_sig_subset(items, &mut subset_intern);
                let target_key = SigSubsetKey::new(target_subset_arc.clone());
                let target_id = if let Some(id) = subset_to_d_id.get(&target_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(target_key.clone(), nid);
                    worklist.push_back(target_key.clone());
                    nid
                };
                // Edge weight: union across contributions in target subset
                let mut edge_w: Option<Weight> = None;
                for (_, w) in target_subset_arc.iter() {
                    if let Some(a) = &mut edge_w {
                        *a |= w;
                    } else {
                        edge_w = Some(w.clone());
                    }
                }
                if let Some(w) = edge_w {
                    // If matches default target and weight, skipping is optional; keeping explicit edge is fine.
                    let _ = dwa.add_transition(d_id, lbl, target_id, w);
                }
            }
        }

        if let Some(p) = &pb {
            p.finish_with_message(format!("Determinized to {} states", subset_to_d_id.len()));
        }

        dwa
    }
}
