// src/precompute4/weighted_automata/determinization.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

/*
Ultra-fast, correctness-preserving determinization

High-level idea (new approach):
- We determinize a weighted ε-NFA (NWA) to a DWA under the algebra (∨, ∧), where:
  - A path contributes weight = ∧ of edge weights and the final weight at the end state.
  - For a word, the accepted weight is the ∨ of all path weights.
- We compress the determinization by exploiting default transitions aggressively:
  - For each NWA state s, we precompute a "macro signature":
      • final_w(s): the union (∨) of weights accrued by any ε-path from s to a final.
      • def(s): the ε-closed step of the default edge, if any (post-ε), i.e., closure(to) gated by the default weight.
      • exceptions(s): only those labeled steps that differ from def(s) (post-ε). For labels whose labeled step equals def(s),
        we omit them from the exceptions; those behave like the default.
    Two states with identical macro signatures are indistinguishable w.r.t. right-languages in determinization.
- During subset-construction, each DWA state is a sorted vector of (macro_sig_id, gate_weight).
  For outgoing edges we compute:
      • Default edge once: union across signatures of def(s) gated by gate_weight.
      • For labels: we only process the union of all exception labels from the signatures in the subset.
        For label ℓ, each signature contributes exceptions(s, ℓ) if present; otherwise contributes def(s) if present; otherwise nothing.
    If a label's aggregated target equals the default target (exactly), we skip adding this exception.
- We keep all vectors (ε-closures, step-vectors, compiled-by-signature vectors, subsets) interned and hashed by strong 64-bit fingerprints.
  This ensures we do not rebuild or rehash large vectors repeatedly.
- We avoid HashMap accumulation in hot paths where possible: we collect contributions into a Vec and normalize (sort+dedup) once,
  which is generally faster than a HashMap for union/merge-heavy code.

Correctness outline:
- The determinization of weighted ε-NFA to a DFA under (∨, ∧) is standard:
  - ε-closure of a state s is the set of states reachable via ε-edges with weight propagation; we use a monotone BFS that unions
    weights across alternative ε-paths. This computes the correct closure under (∨, ∧).
  - For a labeled step, the post-ε step vector from s on label ℓ is closure(to) ∧ w(s, ℓ), unioned across all paths; we compute this as
    closure(to) gated by the transition weight (∧) and union across alternatives using (∨).
  - Default semantics: For symbols not present as explicit labeled transitions, the default transition applies; this matches NWA semantics.
- Aggregating subsets: For a current DWA state X = [(sig(s), gate(s))], the final weight of X is ∨_s (gate(s) ∧ final_w(s)).
  For default step, we aggregate ∨_s compile(def(s)) gated by gate(s). For a label ℓ, we aggregate ∨_s compile(step(s, ℓ)), where step(s, ℓ)
  is exceptions(s, ℓ) if present else def(s) if present else empty. This matches full subset construction semantics.
- Skipping labels whose aggregate equals the default does not change semantics: DWA defaults apply to any label not listed as exception,
  and "normalize_edges_inplace" later also removes redundant exceptions pointing to the default target.

Performance rationale:
- We only process labels that truly differ from the default for at least one signature in the subset, which drastically reduces fanout.
- We compute the default step once per DWA state, and reuse it to filter labels that are redundant.
- We intern all computed vectors by content and memoize compiled-by-signature variants, maximizing reuse, and minimizing hashing and allocations.
- We avoid per-label HashMap accumulation in favor of gather-then-sort-then-dedup, which is faster for these workloads.

This "big picture" reorganization eliminates the main sources of blowup that made previous attempts slow, while remaining fully correct.
*/

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
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

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        // Trim to forward-reachable subgraph (ignoring weights).
        let mut trimmed_states = NWAStates::default();
        let (trimmed_start, _) = trimmed_states.copy_subgraph_from(&self.states, self.body.start_state);
        let trimmed = NWA { states: trimmed_states, body: NWABody { start_state: trimmed_start } };

        crate::debug!(4, "Determinizing NWA (trimmed to {} states)...", trimmed.states.len());
        let dwa = trimmed.internal_determinize_to_dwa_ultrafast();
        crate::debug!(4, "NWA::determinize_to_dwa finished in {:?}", now.elapsed());
        dwa
    }

    fn internal_determinize_to_dwa_ultrafast(&self) -> DWA {
        type StepVec = Arc<Vec<(NWAStateID, Weight)>>;

        // Intern a StepVec by content using a strong fingerprint.
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
            final_w: Option<Weight>,           // ε-closed final weight from this state
            def: Option<StepVec>,              // ε-closed default stepvec (closure(to) gated by default weight)
            exceptions: BTreeMap<i16, StepVec> // only those labels whose stepvec != def (or def absent)
        }

        #[derive(Clone)]
        struct MacroSigKey {
            final_w: Option<Weight>,
            def_ptr: Option<usize>,
            exc_ptrs: Vec<(i16, usize)>,
            fp: u64,
        }
        impl MacroSigKey {
            fn new(final_w: &Option<Weight>, def: &Option<StepVec>, exceptions: &BTreeMap<i16, StepVec>) -> Self {
                let def_ptr = def.as_ref().map(|a| Arc::as_ptr(a) as usize);
                let mut exc_ptrs: Vec<(i16, usize)> = Vec::with_capacity(exceptions.len());
                for (lbl, sv) in exceptions.iter() {
                    exc_ptrs.push((*lbl, Arc::as_ptr(sv) as usize));
                }
                let mut fp = FP_ZERO;
                fp = mix3(fp, final_w.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO), 0x9E37_79B97F4A_7C15u64);
                fp = mix3(fp, def_ptr.unwrap_or(0) as u64, 0xA5A5_5A5A_A5A5_5A5A);
                for (lbl, ptr) in &exc_ptrs {
                    fp = mix3(fp, (*lbl as u64).wrapping_mul(FP_K1), (*ptr as u64).wrapping_mul(FP_K2));
                }
                MacroSigKey { final_w: final_w.clone(), def_ptr, exc_ptrs, fp }
            }
        }
        impl PartialEq for MacroSigKey {
            fn eq(&self, other: &Self) -> bool {
                self.fp == other.fp
                    && self.final_w == other.final_w
                    && self.def_ptr == other.def_ptr
                    && self.exc_ptrs == other.exc_ptrs
            }
        }
        impl Eq for MacroSigKey {}
        impl Hash for MacroSigKey {
            fn hash<H: Hasher>(&self, h: &mut H) {
                h.write_u64(self.fp);
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
            if items.is_empty() {
                let empty = Arc::new(Vec::new());
                let key = SigSubsetKey::new(empty.clone());
                return subset_intern.entry(key).or_insert_with(|| empty.clone()).clone();
            }
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

        // Lazy determinization context with aggressive interning
        struct Ctx<'a> {
            states: &'a NWAStates,
            start: NWAStateID,

            // Caches
            eps_closure: HashMap<NWAStateID, StepVec>,                  // s -> ε-closure(s): Vec<(t, w)>
            step_def_cache: HashMap<NWAStateID, Option<StepVec>>,       // s -> default stepvec (post-ε) gated by default weight
            step_lbl_cache: HashMap<(NWAStateID, i16), StepVec>,        // (s, lbl) -> labeled stepvec (post-ε) gated by trans weight
            step_intern: HashMap<StepVecKey, StepVec>,                  // interner for StepVec contents

            // Macro signatures
            state_to_sig: HashMap<NWAStateID, usize>,                   // s -> SigID
            sig_arena: Vec<Arc<MacroSig>>,
            sig_intern: HashMap<MacroSigKey, usize>,                    // signature interner

            // Compiled stepvec by signature target
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
                    Entry::Vacant(v) => { v.insert(arc.clone()); arc }
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

                // Weighted ε-closure: res[t] = ∨ over all ε-paths (∧ weights along that path)
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

            fn compute_macro_sig_for_state(&mut self, s: NWAStateID) -> Arc<MacroSig> {
                // final macro: ∨ over (t ∈ ε-closure(s): weight(t) ∧ final[t])
                let eps = self.get_eps_closure(s);
                let mut final_acc: Option<Weight> = None;
                for (t, wt) in eps.iter() {
                    if let Some(fw) = &self.states[*t].final_weight {
                        let x = wt & fw;
                        if !x.is_empty() {
                            if let Some(a) = &mut final_acc { *a |= &x; } else { final_acc = Some(x); }
                        }
                    }
                }

                // default macro
                let def_sv = self.get_step_default(s);

                // exceptions: only labels whose stepvec != def_sv
                let mut exceptions: BTreeMap<i16, StepVec> = BTreeMap::new();
                for (lbl, (_to, _w)) in &self.states[s].transitions {
                    if let Some(sv) = self.get_step_label(s, *lbl) {
                        match &def_sv {
                            Some(d) => {
                                if !Arc::ptr_eq(&sv, d) && *sv != **d {
                                    exceptions.insert(*lbl, sv);
                                }
                            }
                            None => {
                                // No default: any present label is an exception
                                exceptions.insert(*lbl, sv);
                            }
                        }
                    }
                }

                Arc::new(MacroSig { final_w: final_acc, def: def_sv, exceptions })
            }

            fn get_sig_id(&mut self, s: NWAStateID) -> usize {
                if let Some(id) = self.state_to_sig.get(&s) {
                    return *id;
                }
                let sig = self.compute_macro_sig_for_state(s);
                let key = MacroSigKey::new(&sig.final_w, &sig.def, &sig.exceptions);
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
                // Normalize to sorted vec
                let mut vec_pairs: Vec<(usize, Weight)> = acc.into_iter().collect();
                vec_pairs.sort_by_key(|(sid, _)| *sid);
                let arc = Arc::new(vec_pairs);
                self.compiled_step_by_sig.insert(key, arc.clone());
                arc
            }
        }

        // Initialize context
        let mut ctx = Ctx::new(&self.states, self.body.start_state);

        // Build initial deterministic subset: ε-closure(start) grouped by macro signature
        let start_eps = ctx.get_eps_closure(ctx.start);
        let mut init_acc_vec: Vec<(usize, Weight)> = Vec::with_capacity(start_eps.len());
        {
            // Collect into Vec then normalize via intern_subset (faster than incremental HashMap)
            for (t, w) in start_eps.iter() {
                let sig = ctx.get_sig_id(*t);
                init_acc_vec.push((sig, w.clone()));
            }
        }
        let init_subset = intern_subset(init_acc_vec, &mut ctx.subset_intern);

        // Construct DWA
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let start_d_id = dwa.states.add_state();
        dwa.body.start_state = start_d_id;

        let mut subset_to_d_id: HashMap<SigSubsetKey, usize> = HashMap::new();
        let init_key = SigSubsetKey::new(init_subset.clone());
        subset_to_d_id.insert(init_key.clone(), start_d_id);

        let mut worklist: VecDeque<SigSubsetKey> = VecDeque::new();
        worklist.push_back(init_key);

        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(1);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinizing DWA (ultrafast): {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} states ({percent}%)")
                    .expect("progress-bar"),
            );
            Some(p)
        } else {
            None
        };
        let mut processed = 0usize;

        // Hot-path scratch buffers (reused)
        let mut tmp_items: Vec<(usize, Weight)> = Vec::new();

        while let Some(sub_key) = worklist.pop_front() {
            processed += 1;
            if let Some(p) = &pb {
                p.set_position(processed as u64);
                p.set_length(subset_to_d_id.len() as u64);
            }

            let d_id = *subset_to_d_id.get(&sub_key).unwrap();
            let subset: &[(usize, Weight)] = &sub_key.entries;

            // 1) Final weight for this DWA state = ∨_sig (gate ∧ final_w(sig))
            let mut final_acc: Option<Weight> = None;
            for (sig_id, gate) in subset.iter() {
                let sig = ctx.get_sig(*sig_id);
                if let Some(fw) = &sig.final_w {
                    let x = gate & fw;
                    if !x.is_empty() {
                        if let Some(a) = &mut final_acc { *a |= &x; } else { final_acc = Some(x); }
                    }
                }
            }
            dwa.states[d_id].final_weight = final_acc;

            // 2) Compute the union of exception labels across signatures in subset
            //    These are the only labels we need to process explicitly; all other labels behave like default.
            let mut label_set: BTreeSet<i16> = BTreeSet::new();
            for (sig_id, _) in subset.iter() {
                let sig = ctx.get_sig(*sig_id);
                for k in sig.exceptions.keys() {
                    label_set.insert(*k);
                }
            }

            // 3) Compute default aggregated target once
            tmp_items.clear();
            for (sig_id, gate) in subset.iter() {
                let sig = ctx.get_sig(*sig_id);
                if let Some(ref sv) = sig.def {
                    let comp = ctx.compile_stepvec_by_sig(sv);
                    if gate.is_all_fast() {
                        // Fast path: OR-accumulate without AND
                        for (t_sig, w) in comp.iter() {
                            tmp_items.push((*t_sig, w.clone()));
                        }
                    } else {
                        for (t_sig, w) in comp.iter() {
                            let x = w & gate;
                            if !x.is_empty() {
                                tmp_items.push((*t_sig, x));
                            }
                        }
                    }
                }
            }
            let def_subset = intern_subset(std::mem::take(&mut tmp_items), &mut ctx.subset_intern);
            let mut def_key_opt: Option<SigSubsetKey> = None;
            let mut def_target_id_opt: Option<usize> = None;
            let mut def_weight_opt: Option<Weight> = None;
            if !def_subset.is_empty() {
                let key = SigSubsetKey::new(def_subset.clone());
                let target_id = if let Some(id) = subset_to_d_id.get(&key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(key.clone(), nid);
                    worklist.push_back(key.clone());
                    nid
                };
                // Default edge weight = ∨ of all component weights in the target subset
                let mut edge_w: Option<Weight> = None;
                for (_, w) in def_subset.iter() {
                    if let Some(a) = &mut edge_w { *a |= w; } else { edge_w = Some(w.clone()); }
                }
                if let Some(w) = edge_w.clone() {
                    let _ = dwa.set_default_transition(d_id, target_id, w);
                    def_key_opt = Some(SigSubsetKey::new(def_subset.clone()));
                    def_target_id_opt = Some(target_id);
                    def_weight_opt = edge_w;
                }
            }

            // 4) Process only labels that present real exceptions in some signature
            for lbl in label_set {
                tmp_items.clear();

                for (sig_id, gate) in subset.iter() {
                    let sig = ctx.get_sig(*sig_id);
                    // Choose step: exceptions(lbl) if present, else def if present, else nothing
                    let chosen_sv_opt = sig.exceptions.get(&lbl).or(sig.def.as_ref());
                    if let Some(ref sv) = chosen_sv_opt {
                        let comp = ctx.compile_stepvec_by_sig(sv);
                        if gate.is_all_fast() {
                            for (t_sig, w) in comp.iter() {
                                tmp_items.push((*t_sig, w.clone()));
                            }
                        } else {
                            for (t_sig, w) in comp.iter() {
                                let x = w & gate;
                                if !x.is_empty() {
                                    tmp_items.push((*t_sig, x));
                                }
                            }
                        }
                    }
                }

                let tgt_subset = intern_subset(std::mem::take(&mut tmp_items), &mut ctx.subset_intern);
                if tgt_subset.is_empty() {
                    // No edge on this label
                    continue;
                }

                // If equal to default aggregated target, skip creating an explicit exception
                if let Some(def_key) = &def_key_opt {
                    let lbl_key = SigSubsetKey::new(tgt_subset.clone());
                    if &lbl_key == def_key {
                        continue;
                    }
                }

                let key = SigSubsetKey::new(tgt_subset.clone());
                let target_id = if let Some(id) = subset_to_d_id.get(&key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    subset_to_d_id.insert(key.clone(), nid);
                    worklist.push_back(key);
                    nid
                };

                // Edge weight on label = ∨ of all component weights in tgt_subset
                let mut edge_w: Option<Weight> = None;
                for (_, w) in tgt_subset.iter() {
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
