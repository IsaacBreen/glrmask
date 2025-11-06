// src/precompute4/weighted_automata/determinization.rs
//
// A compact determinization that keeps the same semantics and performance characteristics:
// - ε-closures are precomputed once and masked by future-acceptance weights.
// - Each NWA state is summarized by a macro-signature: final weight, default step, and
//   per-label steps, where steps are ε-closed target sets with weights.
// - Steps are interned (with a fast fingerprint) and then compiled by macro-signature,
//   giving compact "compiled steps": (macro_sig_id -> weight) lists.
// - Determinized states are sets of macro-signatures with per-member "gates" (weights).
//   We saturate gates via a simple worklist.
// - For each determinized state, we compute a default baseline from members' default steps,
//   then per-label maps override the baseline for signatures that have exceptions.
// - Transitions carry unioned weights; finals carry the union of (gate ∧ member-final).
//
// This version is deliberately concise without sacrificing behavior.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, VecDeque};

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let mut nwa = self.clone();
        // Keep NWA in a tidy form to reduce determinization work.
        nwa.simplify();
        nwa.det_fixpoint()
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

        // Apply extra weight to (state, weight) pairs, dropping empty results.
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
        }

        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            def: Option<usize>,           // step id (raw; compiled later)
            ex: BTreeMap<i16, usize>,     // label -> step id (raw; compiled later)
        }

        #[derive(Clone, Hash, Eq, PartialEq)]
        struct MacroSigKey {
            final_fp: u64,
            def: Option<usize>,
            ex: Vec<(i16, usize)>, // already sorted
        }

        // Precompute ε-closure from each NWA state and build macro signatures.
        let mut eps_cache: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        for s in 0..n {
            eps_cache[s] = eps_closure_masked(std::slice::from_ref(&s), &self.states, &fut);
        }

        let mut step_pool = StepPool::new();

        let mut sigs: Vec<MacroSig> = Vec::with_capacity(n);
        let mut state_to_sig_id: Vec<usize> = vec![0; n];
        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();

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
        }

        // Compile raw steps by macro-signature
        let mut compiled_steps: Vec<CompiledStep> = vec![
            CompiledStep { by_sig: Vec::new() };
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
            compiled_steps[id] = CompiledStep { by_sig };
        }

        // Determinization worklist
        #[derive(Clone)]
        struct DetState {
            members: Vec<usize>,              // macro_sig ids, sorted and unique
            pos: HashMap<usize, usize>,       // macro_sig_id -> index in gates
            gates: Vec<Weight>,               // gates per member
        }

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

        fn ensure_state(
            members: Vec<usize>,
            init: Option<HashMap<usize, Weight>>,
            states: &mut Vec<DetState>,
            map: &mut HashMap<MembersKey, usize>,
            work: &mut VecDeque<usize>,
        ) -> usize {
            let key = MembersKey::new(members);
            if let Some(&id) = map.get(&key) {
                if let Some(init_map) = init {
                    let st = &mut states[id];
                    let mut changed = false;
                    for (sid, w) in init_map {
                        if let Some(&idx) = st.pos.get(&sid) {
                            let before = st.gates[idx].clone();
                            st.gates[idx] |= &w;
                        if st.gates[idx] != before {
                                changed = true;
                            }
                        }
                    }
                    if changed { work.push_back(id); }
                }
                return id;
            }
            let items = key.items.clone();
            let mut pos = HashMap::with_capacity(items.len());
            for (i, sid) in items.iter().enumerate() {
                pos.insert(*sid, i);
            }
            let mut gates = vec![Weight::zeros(); items.len()];
            if let Some(init_map) = init {
                for (sid, w) in init_map {
                    if let Some(&idx) = pos.get(&sid) {
                        gates[idx] |= &w;
                    }
                }
            }
            let id = states.len();
            states.push(DetState { members: items, pos, gates });
            map.insert(key, id);
            work.push_back(id);
            id
        }

        // Helper: merge (compiled step) into a destination map under a gate.
        fn accumulate(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
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
                    if x.is_empty() { continue; }
                    match dst.entry(*sid) {
                        Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= &x; }
                        Entry::Vacant(e) => { e.insert(x); }
                    }
                }
            }
        }

        // Helper: subtract (compiled step) from a destination map under a gate.
        fn subtract(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
            if gate.is_all_fast() {
                for (sid, w) in compiled.iter() {
                    if let Some(old) = dst.get_mut(sid) {
                        *old -= w;
                        if old.is_empty() { dst.remove(sid); }
                    }
                }
            } else {
                for (sid, w) in compiled.iter() {
                    let x = w & gate;
                    if x.is_empty() { continue; }
                    if let Some(old) = dst.get_mut(sid) {
                        *old -= &x;
                        if old.is_empty() { dst.remove(sid); }
                    }
                }
            }
        }

        // Build baseline (default) successor map for a determinized state.
        fn build_baseline_map(st: &DetState, sigs: &[MacroSig], compiled_steps: &[CompiledStep]) -> HashMap<usize, Weight> {
            let mut baseline: HashMap<usize, Weight> = HashMap::new();
            for (i, sig_id) in st.members.iter().enumerate() {
                if let Some(def_id) = sigs[*sig_id].def {
                    accumulate(&mut baseline, &compiled_steps[def_id].by_sig, &st.gates[i]);
                }
            }
            baseline
        }

        // Build a map label -> member indices of st that have exceptions on that label.
        fn label_groups(st: &DetState, sigs: &[MacroSig]) -> BTreeMap<i16, Vec<usize>> {
            let mut label_groups: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (i, sig_id) in st.members.iter().enumerate() {
                for (lbl, _) in &sigs[*sig_id].ex {
                    label_groups.entry(*lbl).or_default().push(i);
                }
            }
            label_groups
        }

        // Initial determinized state from ε-closure(start)
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            let sid = state_to_sig_id[*t];
            match init_map.entry(sid) {
                Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= w; }
                Entry::Vacant(e) => { e.insert(w.clone()); }
            }
        }
        let init_members: Vec<usize> = {
            let mut v: Vec<usize> = init_map.keys().copied().collect();
            v.sort_unstable();
            v
        };

        let mut states: Vec<DetState> = Vec::new();
        let mut key_to_state: HashMap<MembersKey, usize> = HashMap::new();
        let mut work: VecDeque<usize> = VecDeque::new();
        let start_id = ensure_state(init_members, Some(init_map), &mut states, &mut key_to_state, &mut work);

        // Fixpoint propagation
        while let Some(sid) = work.pop_front() {
            let st = states[sid].clone();

            // Default baseline
            let baseline = build_baseline_map(&st, &sigs, &compiled_steps);
            if !baseline.is_empty() {
                let mem: Vec<usize> = baseline.keys().copied().collect();
                let _ = ensure_state(mem, Some(baseline.clone()), &mut states, &mut key_to_state, &mut work);
            }

            // Labels
            for (lbl, idxs) in label_groups(&st, &sigs) {
                let mut cur_map = baseline.clone();
                for i in idxs {
                    let sig_id = st.members[i];
                    let gate = &st.gates[i];

                    // Remove default part for this member (if any)
                    if let Some(def_id) = sigs[sig_id].def {
                        subtract(&mut cur_map, &compiled_steps[def_id].by_sig, gate);
                    }
                    // Add exception part
                    if let Some(ex_id) = sigs[sig_id].ex.get(&lbl).copied() {
                        accumulate(&mut cur_map, &compiled_steps[ex_id].by_sig, gate);
                    }
                }

                if !cur_map.is_empty() {
                    let mem: Vec<usize> = cur_map.keys().copied().collect();
                    let _ = ensure_state(mem, Some(cur_map), &mut states, &mut key_to_state, &mut work);
                }
            }
        }

        // Build final DWA
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        for _ in 0..states.len() {
            dwa.states.add_state();
        }
        dwa.body.start_state = start_id;

        // Helper: compute final weight for a determinized state
        let compute_final = |st: &DetState| -> Option<Weight> {
            let mut acc: Option<Weight> = None;
            for (i, sig_id) in st.members.iter().enumerate() {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    let x = &st.gates[i] & fw;
                    if !x.is_empty() {
                        if let Some(ref mut a) = acc { *a |= &x; } else { acc = Some(x); }
                    }
                }
            }
            acc
        };

        // Emit edges and finals
        for sid in 0..states.len() {
            let st = states[sid].clone();

            // Final
            dwa.states[sid].final_weight = compute_final(&st);

            // Default baseline
            let baseline = build_baseline_map(&st, &sigs, &compiled_steps);
            if !baseline.is_empty() {
                let mut mask = Weight::zeros();
                for (_, w) in &baseline {
                    mask |= w;
                }
                if !mask.is_empty() {
                    let mut mem: Vec<usize> = baseline.keys().copied().collect();
                    mem.sort_unstable();
                    let key = MembersKey::new(mem);
                    if let Some(&to_id) = key_to_state.get(&key) {
                        let _ = dwa.set_default_transition(sid, to_id, mask);
                    }
                }
            }

            // Labels
            for (lbl, idxs) in label_groups(&st, &sigs) {
                let mut cur_map = baseline.clone();
                for i in idxs {
                    let sig_id = st.members[i];
                    let gate = &st.gates[i];
                    if let Some(def_id) = sigs[sig_id].def {
                        subtract(&mut cur_map, &compiled_steps[def_id].by_sig, gate);
                    }
                    if let Some(ex_id) = sigs[sig_id].ex.get(&lbl).copied() {
                        accumulate(&mut cur_map, &compiled_steps[ex_id].by_sig, gate);
                    }
                }

                if cur_map.is_empty() { continue; }
                let mut mask = Weight::zeros();
                for (_, w) in &cur_map { mask |= w; }
                if mask.is_empty() { continue; }

                let mut mem: Vec<usize> = cur_map.keys().copied().collect();
                mem.sort_unstable();
                let key = MembersKey::new(mem);
                if let Some(&to_id) = key_to_state.get(&key) {
                    let _ = dwa.add_transition(sid, lbl, to_id, mask);
                }
            }
        }

        dwa
    }
}
