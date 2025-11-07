// src/precompute4/weighted_automata/determinization.rs
//
// A simpler determinization that keeps the same semantics and performance characteristics:
// - Determinized states are defined by sets of macro-signatures (structural behavior).
// - We compute ε-closures once and reuse them.
// - For each NWA state, we precompute "steps": default and per-label exceptions as ε-closed
//   sets with their weights; steps are interned to avoid duplication.
// - Steps are compiled by macro-signature, so all computations operate on compact signature IDs.
// - Determinization uses a worklist to propagate "gates" (weights) to a fixpoint.
// - For each determinized state, we compute one default transition baseline and override it
//   on labels that have exceptions in any member signature. This avoids per-label recomputation
//   and preserves defaults efficiently.
//
// Semantics preserved for bitset weights with ∧ on path and ∨ over choices.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        let mut nwa = self.clone();
        // Keep as-is; NWA::simplify reduces epsilon chains and prunes unreachable parts.
        nwa.simplify();

        let result = nwa.det_fixpoint();

        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }

    fn det_fixpoint(&self) -> DWA {
        let fut = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        // Weighted ε-closure from sources, intersected with fut[]
        fn eps_closure_masked(
            sources: &[NWAStateID],
            states: &NWAStates,
            fut: &[Weight],
        ) -> Vec<(NWAStateID, Weight)> {
            let mut out: HashMap<NWAStateID, Weight> = HashMap::new();
            let mut q: VecDeque<NWAStateID> = VecDeque::new();
            for &s in sources {
                if s >= states.len() {
                    continue;
                }
                let f = fut[s].clone();
                if !f.is_empty() {
                    out.insert(s, f);
                    q.push_back(s);
                }
            }
            while let Some(u) = q.pop_front() {
                let base = out.get(&u).cloned().unwrap_or_else(Weight::zeros);
                if base.is_empty() {
                    continue;
                }
                for &(v, ref w_eps) in &states[u].epsilons {
                    if v >= states.len() {
                        continue;
                    }
                    let mut prop = &base & w_eps;
                    if prop.is_empty() {
                        continue;
                    }
                    prop &= &fut[v];
                    if prop.is_empty() {
                        continue;
                    }
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

        // Apply additional weight to closure pairs, dropping empties
        fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> Vec<(NWAStateID, Weight)> {
            if w.is_all_fast() {
                return base.to_vec();
            }
            let mut out: Vec<(NWAStateID, Weight)> = Vec::with_capacity(base.len());
            for (sid, wt) in base.iter() {
                let x = wt & w;
                if !x.is_empty() {
                    out.push((*sid, x));
                }
            }
            out
        }

        // Pool of raw step-vectors: list of (NWA state, weight), interned by fingerprint.
        struct StepPool {
            raw: Vec<Vec<(NWAStateID, Weight)>>,
            map: HashMap<u64, Vec<usize>>,
        }
        impl StepPool {
            fn new() -> Self {
                Self { raw: Vec::new(), map: HashMap::new() }
            }
            fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
                let mut fp = FP_ZERO;
                for (sid, w) in pairs.iter() {
                    fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
                }
                fp
            }
            fn intern(&mut self, mut pairs: Vec<(NWAStateID, Weight)>) -> usize {
                pairs.retain(|(_, w)| !w.is_empty());
                let fp = Self::fingerprint(&pairs);
                if let Some(cands) = self.map.get(&fp) {
                    for &id in cands {
                        if self.raw[id].len() == pairs.len() && self.raw[id] == pairs {
                            return id;
                        }
                    }
                }
                let id = self.raw.len();
                self.raw.push(pairs);
                self.map.entry(fp).or_default().push(id);
                id
            }
            fn len(&self) -> usize { self.raw.len() }
        }

        #[derive(Clone)]
        struct CompiledStep {
            by_sig: Vec<(usize, Weight)>, // macro_sig_id -> weight
            mask: Weight,                 // union of weights over by_sig
        }

        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            def: Option<usize>,            // step id (raw; compiled later)
            ex: BTreeMap<i16, usize>,      // label -> step id (raw; compiled later)
        }

        #[derive(Clone, Hash, Eq, PartialEq)]
        struct MacroSigKey {
            final_fp: u64,
            def: Option<usize>,
            ex: Vec<(i16, usize)>, // already sorted
        }

        // Precompute ε-closure from each NWA state and build macro signatures.
        let pb_eps = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(n as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (ε-closures)")
                    .expect("progress-bar style"),
            );
            Some(p)
        } else {
            None
        };

        let mut eps_cache: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        for s in 0..n {
            eps_cache[s] = eps_closure_masked(std::slice::from_ref(&s), &self.states, &fut);
            if let Some(p) = &pb_eps {
                p.inc(1);
            }
        }
        if let Some(p) = pb_eps {
            p.finish_with_message("ε-closures done");
        }

        let mut step_pool = StepPool::new();

        let mut sigs: Vec<MacroSig> = Vec::with_capacity(n);
        let mut state_to_sig_id: Vec<usize> = vec![0; n];
        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();

        let pb_sigs = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(n as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Macro signatures)")
                    .expect("progress-bar style"),
            );
            Some(p)
        } else {
            None
        };

        for s in 0..n {
            // Final weight from ε-closure
            let mut final_acc: Option<Weight> = None;
            for (t, w) in eps_cache[s].iter() {
                if let Some(fw) = &self.states[*t].final_weight {
                    let c = w & fw;
                    if !c.is_empty() {
                        if let Some(ref mut a) = final_acc { *a |= &c; } else { final_acc = Some(c); }
                    }
                }
            }

            // Default step
            let def = if let Some((to, wdef)) = &self.states[s].default {
                if *to < n {
                    Some(step_pool.intern(apply_weight_to_pairs(&eps_cache[*to], wdef)))
                } else {
                    None
                }
            } else {
                None
            };

            // Exception steps
            let mut ex: BTreeMap<i16, usize> = BTreeMap::new();
            for (lbl, (to, wlbl)) in &self.states[s].transitions {
                if *to >= n { continue; }
                let id = step_pool.intern(apply_weight_to_pairs(&eps_cache[*to], wlbl));
                ex.insert(*lbl, id);
            }

            let key = MacroSigKey {
                final_fp: final_acc.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
                def,
                ex: ex.iter().map(|(k, v)| (*k, *v)).collect(),
            };
            let sig_id = match sig_intern.entry(key) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let id = sigs.len();
                    sigs.push(MacroSig { final_w: final_acc.clone(), def, ex: ex.clone() });
                    v.insert(id);
                    id
                }
            };
            state_to_sig_id[s] = sig_id;
            if let Some(p) = &pb_sigs {
                p.inc(1);
            }
        }
        if let Some(p) = pb_sigs {
            p.finish_with_message("Macro signatures done");
        }

        // Compile steps by macro-signature
        let pb_compile = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(step_pool.len() as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Compile steps)")
                    .expect("progress-bar style"),
            );
            Some(p)
        } else {
            None
        };

        let mut compiled_steps: Vec<CompiledStep> = vec![
            CompiledStep { by_sig: Vec::new(), mask: Weight::zeros() };
            step_pool.len()
        ];
        for id in 0..step_pool.len() {
            let pairs = &step_pool.raw[id];
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() {
                let sid = state_to_sig_id[*t];
                match acc.entry(sid) {
                    Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= w; }
                    Entry::Vacant(e) => { e.insert(w.clone()); }
                }
            }
            let mut by_sig: Vec<(usize, Weight)> = acc.into_iter().collect();
            by_sig.sort_by_key(|(sid, _)| *sid);
            let mut mask = Weight::zeros();
            for (_, w) in &by_sig {
                mask |= w;
            }
            compiled_steps[id] = CompiledStep { by_sig, mask };
            if let Some(p) = &pb_compile {
                p.inc(1);
            }
        }
        if let Some(p) = pb_compile {
            p.finish_with_message("Compile steps done");
        }

        // =======================================================================================
        // NEW: On-the-fly minimization logic starts here
        // =======================================================================================

        // A determinized state's composition: the set of NWA macro-signatures it contains,
        // and the "gate" weight for each, representing the accumulated weight from the start state.
        #[derive(Clone)]
        struct DetState {
            members: Vec<usize>,              // macro_sig ids, sorted
            pos: HashMap<usize, usize>,       // macro_sig_id -> index
            gates: Vec<Weight>,               // gates per member
        }

        // A key representing the unique composition of a DetState.
        #[derive(Clone, Eq, PartialEq, Hash)]
        struct MembersKey {
            items: Vec<usize>,
        }
        impl MembersKey {
            fn new(mut items: Vec<usize>) -> Self {
                items.sort_unstable();
                items.dedup();
                MembersKey { items }
            }
        }

        // A key representing the unique BEHAVIOR of a DetState. Two states with different
        // compositions but the same behavioral signature are equivalent and can be merged.
        #[derive(Clone, Eq, PartialEq, Hash)]
        struct DWAStateSignature {
            final_weight: Option<Weight>,
            default_transition: Option<usize>, // Target DWA state ID
            exception_transitions: BTreeMap<i16, usize>, // Label -> Target DWA state ID
        }

        // Helper to accumulate weights for a target state composition.
        fn accumulate(
            dst: &mut HashMap<usize, Weight>,
            compiled: &[(usize, Weight)],
            gate: &Weight,
        ) {
            if gate.is_all_fast() {
                for (sid, w) in compiled.iter() {
                    match dst.entry(*sid) {
                        Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= w; }
                        Entry::Vacant(e) => { e.insert(w.clone()); }
                    }
                }
            } else {
                for (sid, w) in compiled.iter() {
                    let x = w & gate;
                    if x.is_empty() {
                        continue;
                    }
                    match dst.entry(*sid) {
                        Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= &x; }
                        Entry::Vacant(e) => { e.insert(x); }
                    }
                }
            }
        }

        // Manages the state of the determinization process.
        struct Determinizer<'a> {
            // Inputs
            sigs: &'a [MacroSig],
            compiled_steps: &'a [CompiledStep],

            // Algorithm state
            states: Vec<DetState>,
            work: VecDeque<usize>,

            // Caches for on-the-fly minimization
            sig_to_state: HashMap<DWAStateSignature, usize>,
            composition_cache: HashMap<MembersKey, usize>,
        }

        impl<'a> Determinizer<'a> {
            // The core recursive function. Given a composition of NWA signatures and their weights,
            // it returns the canonical ID of a DWA state with that behavior. It creates the state
            // only if a behaviorally equivalent one doesn't already exist.
            fn ensure_state(&mut self, members_map: HashMap<usize, Weight>) -> usize {
                if members_map.is_empty() {
                    // This can happen if all weights are pruned. We need a canonical empty state.
                    // For simplicity, we can just return a placeholder or handle it, but often
                    // the logic naturally avoids creating transitions to empty states.
                    // Let's ensure we have a state for it.
                    let empty_map = HashMap::new();
                    return self.ensure_state_inner(empty_map);
                }
                self.ensure_state_inner(members_map)
            }

            fn ensure_state_inner(&mut self, members_map: HashMap<usize, Weight>) -> usize {
                let members_vec: Vec<usize> = members_map.keys().copied().collect();
                let key = MembersKey::new(members_vec);

                // Memoization: If we've already computed the behavior for this exact composition,
                // we can reuse the result. We still need to update gates.
                if let Some(&id) = self.composition_cache.get(&key) {
                    let st = &mut self.states[id];
                    let mut changed = false;
                    for (sid, w) in &members_map {
                        if let Some(&idx) = st.pos.get(sid) {
                            let before = st.gates[idx].clone();
                            st.gates[idx] |= w;
                            if st.gates[idx] != before {
                                changed = true;
                            }
                        }
                    }
                    if changed {
                        self.work.push_back(id);
                    }
                    return id;
                }

                // --- This composition is new, compute its behavioral signature ---

                // 1. Compute final weight
                let mut final_weight: Option<Weight> = None;
                for (sig_id, gate) in &members_map {
                    if let Some(fw) = &self.sigs[*sig_id].final_w {
                        let x = gate & fw;
                        if !x.is_empty() {
                            if let Some(ref mut acc) = final_weight { *acc |= &x; } else { final_weight = Some(x); }
                        }
                    }
                }

                // 2. Compute transitions by recursively finding the canonical ID of target states
                let mut default_target_map: HashMap<usize, Weight> = HashMap::new();
                let mut label_groups: BTreeMap<i16, Vec<(usize, &Weight)>> = BTreeMap::new();

                for (sig_id, gate) in &members_map {
                    // Accumulate default transitions
                    if let Some(def_id) = self.sigs[*sig_id].def {
                        accumulate(&mut default_target_map, &self.compiled_steps[def_id].by_sig, gate);
                    }
                    // Group members by their exception labels
                    for (lbl, _) in &self.sigs[*sig_id].ex {
                        label_groups.entry(*lbl).or_default().push((*sig_id, gate));
                    }
                }

                let default_transition = if default_target_map.is_empty() { None } else { Some(self.ensure_state(default_target_map.clone())) };

                let mut exception_transitions = BTreeMap::new();
                for (lbl, members_with_ex) in label_groups {
                    let mut target_map = HashMap::new();
                    // Efficiently build the target map from scratch
                    for (sig_id, gate) in &members_map {
                        if self.sigs[*sig_id].ex.contains_key(&lbl) {
                            if let Some(ex_id) = self.sigs[*sig_id].ex.get(&lbl) {
                                accumulate(&mut target_map, &self.compiled_steps[*ex_id].by_sig, gate);
                            }
                        } else if let Some(def_id) = self.sigs[*sig_id].def {
                            accumulate(&mut target_map, &self.compiled_steps[def_id].by_sig, gate);
                        }
                    }
                    if !target_map.is_empty() {
                        exception_transitions.insert(lbl, self.ensure_state(target_map));
                    }
                }

                // 3. Form the behavioral signature
                let sig = DWAStateSignature { final_weight, default_transition, exception_transitions };

                // 4. Lookup or create state based on BEHAVIOR
                let id = match self.sig_to_state.entry(sig) {
                    Entry::Occupied(o) => *o.get(), // Found a match! Merge.
                    Entry::Vacant(v) => { // Genuinely new behavior
                        let new_id = self.states.len();
                        let items = key.items.clone();
                        let mut pos = HashMap::with_capacity(items.len());
                        for (i, sid) in items.iter().enumerate() {
                            pos.insert(*sid, i);
                        }
                        let gates = vec![Weight::zeros(); items.len()];
                        self.states.push(DetState { members: items, pos, gates });
                        self.work.push_back(new_id);
                        v.insert(new_id);
                        new_id
                    }
                };

                // 5. Update composition cache and gates for the canonical state
                self.composition_cache.insert(key, id);
                let st = &mut self.states[id];
                for (sid, w) in &members_map {
                    if let Some(&idx) = st.pos.get(sid) {
                        st.gates[idx] |= w;
                    } else {
                        // This happens when merging into a state with a different composition.
                        // We need to expand its members list.
                        let new_idx = st.members.len();
                        st.members.push(*sid);
                        st.pos.insert(*sid, new_idx);
                        st.gates.push(w.clone());
                    }
                }
                id
            }
        }

        // Initial determinized state: ε-closure(start) grouped by macro signature
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            let sid = state_to_sig_id[*t];
            match init_map.entry(sid) {
                Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= w; }
                Entry::Vacant(e) => { e.insert(w.clone()); }
            }
        }

        let mut determinizer = Determinizer {
            sigs: &sigs,
            compiled_steps: &compiled_steps,
            states: Vec::new(),
            work: VecDeque::new(),
            sig_to_state: HashMap::new(),
            composition_cache: HashMap::new(),
        };

        let start_id = determinizer.ensure_state(init_map);

        let pb_det = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(0); // Length is unknown, will be updated
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({msg})")
                    .expect("progress-bar style"),
            );
            p.set_message("Starting...");
            Some(p)
        } else {
            None
        };

        // Fixpoint propagation: ensure all reachable determinized states are created and
        // their gates saturated.
        while let Some(sid) = determinizer.work.pop_front() {
            if let Some(p) = &pb_det {
                p.inc(1);
                p.set_length(determinizer.states.len() as u64);
                p.set_message(format!("states: {}, queue: {}", determinizer.states.len(), determinizer.work.len()));
            }

            let st = determinizer.states[sid].clone(); // Clone to avoid borrow checker issues with recursive calls

            // Trigger computation of all target states. The ensure_state function handles
            // creation, merging, and queueing.

            // 1. Default transition
            let mut default_target_map: HashMap<usize, Weight> = HashMap::new();
            for (i, sig_id) in st.members.iter().enumerate() {
                if let Some(def_id) = determinizer.sigs[*sig_id].def {
                    accumulate(&mut default_target_map, &determinizer.compiled_steps[def_id].by_sig, &st.gates[i]);
                }
            }
            if !default_target_map.is_empty() {
                determinizer.ensure_state(default_target_map);
            }

            // 2. Exception transitions
            let mut label_groups: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (i, sig_id) in st.members.iter().enumerate() {
                for (lbl, _) in &determinizer.sigs[*sig_id].ex {
                    label_groups.entry(*lbl).or_default().push(i);
                }
            }

            for (lbl, _) in label_groups {
                let mut target_map = HashMap::new();
                for (i, sig_id) in st.members.iter().enumerate() {
                    let gate = &st.gates[i];
                    if determinizer.sigs[*sig_id].ex.contains_key(&lbl) {
                        if let Some(ex_id) = determinizer.sigs[*sig_id].ex.get(&lbl) {
                            accumulate(&mut target_map, &determinizer.compiled_steps[*ex_id].by_sig, gate);
                        }
                    } else if let Some(def_id) = determinizer.sigs[*sig_id].def {
                        accumulate(&mut target_map, &determinizer.compiled_steps[def_id].by_sig, gate);
                    }
                }
                if !target_map.is_empty() {
                    determinizer.ensure_state(target_map);
                }
            }
        }
        if let Some(p) = pb_det {
            p.finish_with_message(format!("Determinized to {} states", determinizer.states.len()));
        }

        // Build final DWA
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let num_states = determinizer.states.len();
        for _ in 0..num_states {
            dwa.states.add_state();
        }
        dwa.body.start_state = start_id;

        let pb_build = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(num_states as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Build DWA)")
                    .expect("progress-bar style"),
            );
            Some(p)
        } else {
            None
        };

        // Invert the sig_to_state map to easily find the signature for a given state ID.
        let mut state_to_sig: Vec<Option<DWAStateSignature>> = vec![None; num_states];
        for (sig, id) in determinizer.sig_to_state {
            state_to_sig[id] = Some(sig);
        }

        for sid in 0..num_states {
            if let Some(sig) = &state_to_sig[sid] {
                dwa.states[sid].final_weight = sig.final_weight.clone();

                // To get the edge weight, we must compute the union of weights in the target's composition.
                let compute_edge_weight = |target_id: usize| -> Weight {
                    let target_st = &determinizer.states[target_id];
                    let mut mask = Weight::zeros();
                    for w in &target_st.gates {
                        mask |= w;
                    }
                    mask
                };

                if let Some(to_id) = sig.default_transition {
                    let weight = compute_edge_weight(to_id);
                    if !weight.is_empty() {
                        let _ = dwa.set_default_transition(sid, to_id, weight);
                    }
                }

                for (lbl, to_id) in &sig.exception_transitions {
                    let weight = compute_edge_weight(*to_id);
                    if !weight.is_empty() {
                        let _ = dwa.add_transition(sid, *lbl, *to_id, weight);
                    }
                }
            }
            if let Some(p) = &pb_build {
                p.inc(1);
            }
        }
        if let Some(p) = pb_build {
            p.finish_with_message("Build DWA done");
        }

        dwa
    }
}