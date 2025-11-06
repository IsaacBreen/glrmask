// src/precompute4/weighted_automata/determinization.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::{StateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWABody, NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

/*
Radically simplified and optimized determinization:

Core ideas:
- Build a trimmed copy of the NWA containing only states reachable from the start (ignoring weights).
- Avoid any global precomputation. Compute epsilon-closures, per-edge "macro step-vectors", and
  per-state "macro signatures" lazily, caching them aggressively as they are needed.
- Determinize directly over macro signatures (not raw states). A macro signature summarizes a state
  by: the epsilon-closed final weight, its default macro step-vector (if any), and its labeled macro
  step-vectors. Two states with identical summaries are equivalent for right languages, so we
  group them.
- For each DWA state (a weighted subset of macro signatures), derive:
  - final weight = union over (gate_weight ∧ signature.final)
  - default edge = union over (gate_weight ∧ compiled(signature.default))
  - for each label L observed in any signature in the subset:
      target = union over signatures of (gate_weight ∧ compiled(step for L if present else default))
- All macro step-vectors are unrolled through epsilon-closures lazily; "compiled" step-vectors are
  target-grouped by macro signature, and also cached by pointer identity, so they are compiled once.

This removes the O(|S|) global passes from the previous version and scales with the actually explored
subset of the automaton. It also avoids expensive "future-weight" fixpoints. Correctness is preserved
by standard weighted ε-closure determinization semantics under (∨, ∧).
*/

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        // 1) Trim to the forward-reachable subgraph from start (ignoring weights).
        //    This avoids touching giant unreachable regions.
        let mut trimmed_states = NWAStates::default();
        let (trimmed_start, _) = trimmed_states.copy_subgraph_from(&self.states, self.body.start_state);
        let trimmed = NWA { states: trimmed_states, body: NWABody { start_state: trimmed_start } };

        crate::debug!(4, "Determinizing NWA (trimmed to {} states)...", trimmed.states.len());
        let dwa = trimmed.internal_determinize_to_dwa_lazy();
        crate::debug!(4, "NWA::determinize_to_dwa finished in {:?}", now.elapsed());
        dwa
    }

    fn internal_determinize_to_dwa_lazy(&self) -> DWA {
        type StepVec = Arc<Vec<(NWAStateID, Weight)>>;

        // Intern a StepVec (Vec<(state, weight)>) by content with a strong fingerprint.
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
            fn new(final_w: &Option<Weight>, def: &Option<StepVec>, labeled: &BTreeMap<i16, StepVec>) -> Self {
                let mut lbl_vec: Vec<(i16, usize)> = Vec::with_capacity(labeled.len());
                for (k, v) in labeled.iter() {
                    lbl_vec.push((*k, Arc::as_ptr(v) as usize));
                }
                let def_ptr = def.as_ref().map(|a| Arc::as_ptr(a) as usize);
                let mut fp = FP_ZERO;
                // Mix in final weight fingerprint (or zero)
                fp = mix3(fp, final_w.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO), 0x9E37_79B97F4A_7C15u64);
                // Mix in def ptr
                fp = mix3(fp, def_ptr.unwrap_or(0) as u64, 0xA5A5_5A5A_A5A5_5A5A);
                // Mix in labeled pointers
                for (lbl, ptr) in &lbl_vec {
                    fp = mix3(fp, (*lbl as u64).wrapping_mul(FP_K1), (*ptr as u64).wrapping_mul(FP_K2));
                }
                MacroSigKey { final_w: final_w.clone(), def_ptr, labeled: lbl_vec, fp }
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

        // Lazy determinization context
        struct Ctx<'a> {
            states: &'a NWAStates,
            start: NWAStateID,

            // Caches
            eps_closure: HashMap<NWAStateID, StepVec>,                // s -> ε-closure(s): Vec<(t, w)>
            step_def_cache: HashMap<NWAStateID, Option<StepVec>>,     // s -> default stepvec (post-ε) gated by def weight
            step_lbl_cache: HashMap<(NWAStateID, i16), StepVec>,      // (s, lbl) -> labeled stepvec (post-ε) gated by trans weight
            step_intern: HashMap<StepVecKey, StepVec>,                // interner for StepVec contents

            // Macro signatures
            state_to_sig: HashMap<NWAStateID, usize>,                 // s -> SigID
            sig_arena: Vec<Arc<MacroSig>>,
            sig_intern: HashMap<MacroSigKey, usize>,                  // signature interner

            // Compiled stepvec by signature of targets
            compiled_step_by_sig: HashMap<usize, Arc<Vec<(usize, Weight)>>>, // ptr(StepVec) -> Vec<(SigID, W)>

            // Subset interner
            subset_intern: HashMap<SigSubsetKey, Arc<Vec<(usize, Weight)>>>,
        }

        impl<'a> Ctx<'a> {
            fn new(states: &'a NWAStates, start: NWAStateID) -> Self {
                Ctx {
                    states,
                    start,
                    eps_closure: HashMap::new(),
                    step_def_cache: HashMap::new(),
                    step_lbl_cache: HashMap::new(),
                    step_intern: HashMap::new(),
                    state_to_sig: HashMap::new(),
                    sig_arena: Vec::new(),
                    sig_intern: HashMap::new(),
                    compiled_step_by_sig: HashMap::new(),
                    subset_intern: HashMap::new(),
                }
            }

            fn intern_step_vec(&mut self, mut items: Vec<(NWAStateID, Weight)>) -> StepVec {
                if items.is_empty() {
                    return Arc::new(Vec::new());
                }
                items.sort_by_key(|(s, _)| *s);
                // Merge duplicates, drop empties
                let mut merged: Vec<(NWAStateID, Weight)> = Vec::with_capacity(items.len());
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
                let arc = Arc::new(merged);
                let key = StepVecKey::new(arc.clone());
                match self.step_intern.entry(key) {
                    Entry::Occupied(o) => o.get().clone(),
                    Entry::Vacant(v) => {
                        v.insert(arc.clone());
                        arc
                    }
                }
            }

            fn get_eps_closure(&mut self, s: NWAStateID) -> StepVec {
                if let Some(v) = self.eps_closure.get(&s) {
                    return v.clone();
                }
                let n = self.states.0.len();
                if s >= n {
                    let arc = Arc::new(Vec::new());
                    self.eps_closure.insert(s, arc.clone());
                    return arc;
                }

                // Weighted epsilon-closure: res[t] = union over paths s =>* t of intersection of eps-edge weights.
                let mut res: HashMap<NWAStateID, Weight> = HashMap::new();
                let mut q: VecDeque<NWAStateID> = VecDeque::new();

                res.insert(s, Weight::all());
                q.push_back(s);

                while let Some(u) = q.pop_front() {
                    let uw = res.get(&u).cloned().unwrap_or_else(Weight::zeros);
                    if uw.is_empty() { continue; }
                    for &(v, ref eps_w) in &self.states[u].epsilons {
                        let prop = &uw & eps_w;
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

                let vecv: Vec<(NWAStateID, Weight)> = res.into_iter().collect();
                let arc = self.intern_step_vec(vecv);
                self.eps_closure.insert(s, arc.clone());
                arc
            }

            fn get_step_default(&mut self, s: NWAStateID) -> Option<StepVec> {
                if let Some(v) = self.step_def_cache.get(&s) {
                    return v.clone();
                }
                let result = if let Some((to, w)) = &self.states[s].default {
                    let eps = self.get_eps_closure(*to);
                    if w.is_all_fast() {
                        // Just return eps
                        Some(eps)
                    } else {
                        let mut items = Vec::with_capacity(eps.len());
                        for (t, wt) in eps.iter() {
                            let x = wt & w;
                            if !x.is_empty() { items.push((*t, x)); }
                        }
                        Some(self.intern_step_vec(items))
                    }
                } else {
                    None
                };
                self.step_def_cache.insert(s, result.clone());
                result
            }

            fn get_step_label(&mut self, s: NWAStateID, lbl: i16) -> Option<StepVec> {
                if let Some(sv) = self.step_lbl_cache.get(&(s, lbl)) {
                    return Some(sv.clone());
                }
                if let Some((to, w)) = self.states[s].transitions.get(&lbl) {
                    let eps = self.get_eps_closure(*to);
                    let sv = if w.is_all_fast() {
                        eps
                    } else {
                        let mut items = Vec::with_capacity(eps.len());
                        for (t, wt) in eps.iter() {
                            let x = wt & w;
                            if !x.is_empty() { items.push((*t, x)); }
                        }
                        self.intern_step_vec(items)
                    };
                    self.step_lbl_cache.insert((s, lbl), sv.clone());
                    Some(sv)
                } else {
                    None
                }
            }

            fn compile_stepvec_by_sig(&mut self, sv: &StepVec) -> Arc<Vec<(usize, Weight)>> {
                let key = Arc::as_ptr(sv) as usize;
                if let Some(v) = self.compiled_step_by_sig.get(&key) {
                    return v.clone();
                }
                let mut acc: HashMap<usize, Weight> = HashMap::new();
                for (t, w) in sv.iter() {
                    let sig = self.get_sig_id(*t);
                    match acc.entry(sig) {
                        Entry::Occupied(mut e) => {
                            let old = e.get_mut();
                            *old |= w;
                        }
                        Entry::Vacant(e) => {
                            e.insert(w.clone());
                        }
                    }
                }
                let mut vec_pairs: Vec<(usize, Weight)> = acc.into_iter().collect();
                vec_pairs.sort_by_key(|(sid, _)| *sid);
                let arc = Arc::new(vec_pairs);
                self.compiled_step_by_sig.insert(key, arc.clone());
                arc
            }

            fn compute_macro_sig_for_state(&mut self, s: NWAStateID) -> Arc<MacroSig> {
                // final macro: union over (eps_closure(s): wt ∧ final_weight[target])
                let eps = self.get_eps_closure(s);
                let mut final_acc: Option<Weight> = None;
                for (t, wt) in eps.iter() {
                    if let Some(fw) = &self.states[*t].final_weight {
                        let x = wt & fw;
                        if !x.is_empty() {
                            if let Some(a) = &mut final_acc {
                                *a |= &x;
                            } else {
                                final_acc = Some(x);
                            }
                        }
                    }
                }

                // default macro (post-ε), if present
                let def_sv = self.get_step_default(s);

                // labeled macro map
                let mut labeled: BTreeMap<i16, StepVec> = BTreeMap::new();
                for (lbl, (_to, _w)) in &self.states[s].transitions {
                    if let Some(sv) = self.get_step_label(s, *lbl) {
                        labeled.insert(*lbl, sv);
                    }
                }

                // Assembly
                Arc::new(MacroSig { final_w: final_acc, def: def_sv, labeled })
            }

            fn get_sig_id(&mut self, s: NWAStateID) -> usize {
                if let Some(id) = self.state_to_sig.get(&s) {
                    return *id;
                }
                let sig = self.compute_macro_sig_for_state(s);
                let key = MacroSigKey::new(&sig.final_w, &sig.def, &sig.labeled);
                let id = match self.sig_intern.entry(key) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(v) => {
                        let new_id = self.sig_arena.len();
                        self.sig_arena.push(sig);
                        v.insert(new_id);
                        new_id
                    }
                };
                self.state_to_sig.insert(s, id);
                id
            }

            fn get_sig(&self, id: usize) -> Arc<MacroSig> {
                self.sig_arena[id].clone()
            }
        }

        // Subset interner for DWA states: sorted Vec<(SigID, Weight)>
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

        fn intern_subset(
            mut items: Vec<(usize, Weight)>,
            subset_intern: &mut HashMap<SigSubsetKey, Arc<Vec<(usize, Weight)>>>,
        ) -> Arc<Vec<(usize, Weight)>> {
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
            match subset_intern.entry(key) {
                Entry::Occupied(o) => o.get().clone(),
                Entry::Vacant(v) => { v.insert(arc.clone()); arc }
            }
        }

        // Initialize context
        let mut ctx = Ctx::new(&self.states, self.body.start_state);

        // Build initial subset: ε-closure(start), grouped by macro signatures
        let start_eps = ctx.get_eps_closure(ctx.start);
        let mut init_acc: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in start_eps.iter() {
            let sig = ctx.get_sig_id(*t);
            match init_acc.entry(sig) {
                Entry::Occupied(mut e) => {
                    let old = e.get_mut();
                    *old |= w;
                }
                Entry::Vacant(e) => {
                    e.insert(w.clone());
                }
            }
        }
        let init_items: Vec<(usize, Weight)> = init_acc.into_iter().collect();
        let init_subset = intern_subset(init_items, &mut ctx.subset_intern);

        // Construct DWA
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let start_d_id = dwa.states.add_state();
        dwa.body.start_state = start_d_id;

        let mut subset_to_d_id: HashMap<SigSubsetKey, StateID> = HashMap::new();
        let init_key = SigSubsetKey::new(init_subset.clone());
        subset_to_d_id.insert(init_key.clone(), start_d_id);

        let mut worklist: VecDeque<SigSubsetKey> = VecDeque::new();
        worklist.push_back(init_key);

        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(1);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinizing DWA (lazy): {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} states ({percent}%)")
                    .expect("progress-bar"),
            );
            Some(p)
        } else {
            None
        };
        let mut processed = 0usize;

        while let Some(sub_key) = worklist.pop_front() {
            processed += 1;
            if let Some(p) = &pb {
                p.set_position(processed as u64);
                p.set_length(subset_to_d_id.len() as u64);
            }

            let d_id = *subset_to_d_id.get(&sub_key).unwrap();
            let subset: &[(usize, Weight)] = &sub_key.entries;

            // 1) Final weight
            let mut final_acc: Option<Weight> = None;
            for (sig_id, gate) in subset.iter() {
                let sig = ctx.get_sig(*sig_id);
                if let Some(fw) = &sig.final_w {
                    let x = gate & fw;
                    if !x.is_empty() {
                        if let Some(a) = &mut final_acc {
                            *a |= &x;
                        } else {
                            final_acc = Some(x);
                        }
                    }
                }
            }
            dwa.states[d_id].final_weight = final_acc;

            // Collect all labels present across signatures in subset
            let mut all_labels: BTreeSet<i16> = BTreeSet::new();
            for (sig_id, _) in subset.iter() {
                let sig = ctx.get_sig(*sig_id);
                for k in sig.labeled.keys() {
                    all_labels.insert(*k);
                }
            }

            // 2) Default edge (for labels not in 'all_labels'): union across signatures' default
            let mut def_acc: HashMap<usize, Weight> = HashMap::new();
            for (sig_id, gate) in subset.iter() {
                let sig = ctx.get_sig(*sig_id);
                if let Some(ref sv) = sig.def {
                    let comp = ctx.compile_stepvec_by_sig(sv);
                    if gate.is_all_fast() {
                        for (t_sig, w) in comp.iter() {
                            match def_acc.entry(*t_sig) {
                                Entry::Occupied(mut e) => { let old = e.get_mut(); *old |= w; }
                                Entry::Vacant(e) => { e.insert(w.clone()); }
                            }
                        }
                    } else {
                        for (t_sig, w) in comp.iter() {
                            let x = w & gate;
                            if x.is_empty() { continue; }
                            match def_acc.entry(*t_sig) {
                                Entry::Occupied(mut e) => { let old = e.get_mut(); *old |= &x; }
                                Entry::Vacant(e) => { e.insert(x); }
                            }
                        }
                    }
                }
            }

            let mut def_target_state: Option<StateID> = None;
            if !def_acc.is_empty() {
                let def_items: Vec<(usize, Weight)> = def_acc.into_iter().collect();
                let def_subset = intern_subset(def_items, &mut ctx.subset_intern);
                let def_key = SigSubsetKey::new(def_subset.clone());
                let target_id = if let Some(id) = subset_to_d_id.get(&def_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(def_key.clone(), nid);
                    worklist.push_back(def_key);
                    nid
                };
                // Default edge weight: union of weights in the target subset vector
                let mut edge_w: Option<Weight> = None;
                for (_, w) in def_subset.iter() {
                    if let Some(a) = &mut edge_w { *a |= w; } else { edge_w = Some(w.clone()); }
                }
                if let Some(w) = edge_w {
                    dwa.set_default_transition(d_id, target_id, w).expect("default transition");
                    def_target_state = Some(target_id);
                }
            }

            // 3) Exception edges for each label in all_labels
            for lbl in all_labels {
                let mut acc: HashMap<usize, Weight> = HashMap::new();
                for (sig_id, gate) in subset.iter() {
                    let sig = ctx.get_sig(*sig_id);
                    // For this label, prefer explicit step; else fall back to default; else no contribution.
                    let chosen_sv_opt = sig.labeled.get(&lbl).or(sig.def.as_ref());
                    if let Some(ref sv) = chosen_sv_opt {
                        let comp = ctx.compile_stepvec_by_sig(sv);
                        if gate.is_all_fast() {
                            for (t_sig, w) in comp.iter() {
                                match acc.entry(*t_sig) {
                                    Entry::Occupied(mut e) => { let old = e.get_mut(); *old |= w; }
                                    Entry::Vacant(e) => { e.insert(w.clone()); }
                                }
                            }
                        } else {
                            for (t_sig, w) in comp.iter() {
                                let x = w & gate;
                                if x.is_empty() { continue; }
                                match acc.entry(*t_sig) {
                                    Entry::Occupied(mut e) => { let old = e.get_mut(); *old |= &x; }
                                    Entry::Vacant(e) => { e.insert(x); }
                                }
                            }
                        }
                    }
                }
                if acc.is_empty() { continue; }

                let items: Vec<(usize, Weight)> = acc.into_iter().collect();
                let target_subset = intern_subset(items, &mut ctx.subset_intern);
                let key = SigSubsetKey::new(target_subset.clone());
                let target_id = if let Some(id) = subset_to_d_id.get(&key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(key.clone(), nid);
                    worklist.push_back(key);
                    nid
                };

                let mut edge_w: Option<Weight> = None;
                for (_, w) in target_subset.iter() {
                    if let Some(a) = &mut edge_w { *a |= w; } else { edge_w = Some(w.clone()); }
                }
                if let Some(w) = edge_w {
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
