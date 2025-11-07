// src/precompute4/weighted_automata/determinization.rs
//
// A faster determinization that preserves semantics but avoids HashMap-heavy hot paths.
// Key changes:
// - Replace HashMap-based baseline/label accumulation with sorted Vec<(sig_id, Weight)> operations.
// - Batch-merge compiled steps with gates via sort+dedup union, and subtract/add via linear scans.
// - Keep macro-signature and step interning intact, but drastically reduce per-state overhead.
//
// This leverages the structure where most transitions inside SCCs have Weight::all(),
// so merging many compiled steps benefits from vectorized operations and low allocation churn.

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

        let result = nwa.det_fixpoint_vecmerge();

        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }

    /// New faster determinization based on vector merges instead of HashMaps in hot paths.
    fn det_fixpoint_vecmerge(&self) -> DWA {
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
            by_sig: Vec<(usize, Weight)>, // macro_sig_id -> weight, sorted by macro_sig_id
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

        // ------------- Vector-merge helpers (fast path) ----------------

        #[inline]
        fn union_compiled_with_gates(
            compiled_steps: &Vec<CompiledStep>,
            pairs: &[(usize, &Weight)], // list of (compiled_step_id, gate)
        ) -> Vec<(usize, Weight)> {
            if pairs.is_empty() {
                return Vec::new();
            }
            // Upper bound guess: sum of all by_sig lengths (in practice small).
            let mut tmp: Vec<(usize, Weight)> = Vec::new();
            for (sid, gate) in pairs.iter() {
                let step = &compiled_steps[*sid].by_sig;
                if gate.is_all_fast() {
                    for (ms, w) in step.iter() {
                        tmp.push((*ms, w.clone()));
                    }
                } else {
                    for (ms, w) in step.iter() {
                        let x = w & *gate;
                        if !x.is_empty() {
                            tmp.push((*ms, x));
                        }
                    }
                }
            }
            if tmp.is_empty() {
                return tmp;
            }
            // Sort by macro_sig id and OR weights for duplicates
            tmp.sort_by_key(|(ms, _)| *ms);
            let mut out: Vec<(usize, Weight)> = Vec::with_capacity(tmp.len());
            let mut cur = tmp[0].0;
            let mut acc = tmp[0].1.clone();
            for i in 1..tmp.len() {
                let (ms, ref w) = tmp[i];
                if ms == cur {
                    acc |= w;
                } else {
                    if !acc.is_empty() {
                        out.push((cur, acc));
                    }
                    cur = ms;
                    acc = w.clone();
                }
            }
            if !acc.is_empty() {
                out.push((cur, acc));
            }
            out
        }

        #[inline]
        fn subtract_sorted(
            base: &[(usize, Weight)], // sorted by sig
            sub: &[(usize, Weight)],  // sorted by sig
        ) -> Vec<(usize, Weight)> {
            if base.is_empty() { return Vec::new(); }
            if sub.is_empty() { return base.to_vec(); }
            let mut out: Vec<(usize, Weight)> = Vec::with_capacity(base.len());
            let mut i = 0usize;
            let mut j = 0usize;
            while i < base.len() {
                let (b_sig, b_w) = (&base[i].0, &base[i].1);
                if j >= sub.len() {
                    // push remaining
                    out.push((*b_sig, b_w.clone()));
                    i += 1;
                    continue;
                }
                let (s_sig, s_w) = (&sub[j].0, &sub[j].1);
                if b_sig < s_sig {
                    out.push((*b_sig, b_w.clone()));
                    i += 1;
                } else if b_sig == s_sig {
                    let mut w = b_w.clone();
                    w -= s_w;
                    if !w.is_empty() {
                        out.push((*b_sig, w));
                    }
                    i += 1;
                    j += 1;
                } else {
                    // sub sig < base sig
                    j += 1;
                }
            }
            out
        }

        #[inline]
        fn add_sorted(
            a: &[(usize, Weight)], // sorted
            b: &[(usize, Weight)], // sorted
        ) -> Vec<(usize, Weight)> {
            if a.is_empty() { return b.to_vec(); }
            if b.is_empty() { return a.to_vec(); }
            let mut out: Vec<(usize, Weight)> = Vec::with_capacity(a.len() + b.len());
            let mut i = 0usize;
            let mut j = 0usize;
            while i < a.len() && j < b.len() {
                let (asig, aw) = (&a[i].0, &a[i].1);
                let (bsig, bw) = (&b[j].0, &b[j].1);
                if asig < bsig {
                    out.push((*asig, aw.clone()));
                    i += 1;
                } else if asig > bsig {
                    out.push((*bsig, bw.clone()));
                    j += 1;
                } else {
                    let mut w = aw.clone();
                    w |= bw;
                    if !w.is_empty() {
                        out.push((*asig, w));
                    }
                    i += 1;
                    j += 1;
                }
            }
            while i < a.len() {
                out.push((a[i].0, a[i].1.clone()));
                i += 1;
            }
            while j < b.len() {
                out.push((b[j].0, b[j].1.clone()));
                j += 1;
            }
            out
        }

        #[inline]
        fn compute_mask(v: &[(usize, Weight)]) -> Weight {
            let mut m = Weight::zeros();
            for (_, w) in v {
                m |= w;
            }
            m
        }

        // Determinization worklist
        #[derive(Clone)]
        struct DetState {
            members: Vec<usize>,              // macro_sig ids, sorted
            pos: HashMap<usize, usize>,       // macro_sig_id -> index
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
                    if changed {
                        work.push_back(id);
                    }
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

        // Initial determinized state: ε-closure(start) grouped by macro signature
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

        // Fixpoint propagation
        while let Some(sid) = work.pop_front() {
            if let Some(p) = &pb_det {
                p.inc(1);
                p.set_length(states.len() as u64);
                p.set_message(format!("states: {}, queue: {}", states.len(), work.len()));
            }

            let st = states[sid].clone();

            // Build the list of (default step id, gate) for all members that have defaults.
            let mut def_pairs: Vec<(usize, &Weight)> = Vec::new();
            def_pairs.reserve(st.members.len());
            for (i, sig_id) in st.members.iter().enumerate() {
                if let Some(def_id) = sigs[*sig_id].def {
                    let gate = &st.gates[i];
                    if !gate.is_empty() {
                        def_pairs.push((def_id, gate));
                    }
                }
            }
            // Default baseline (sorted vec)
            let baseline = union_compiled_with_gates(&compiled_steps, &def_pairs);
            if !baseline.is_empty() {
                let mem: Vec<usize> = baseline.iter().map(|(m, _)| *m).collect();
                if !mem.is_empty() {
                    // build init gates map
                    let mut init_gates: HashMap<usize, Weight> = HashMap::with_capacity(baseline.len());
                    for (m, w) in baseline.iter() {
                        init_gates.insert(*m, w.clone());
                    }
                    let _ = ensure_state(mem, Some(init_gates), &mut states, &mut key_to_state, &mut work);
                }
            }

            // Labels that appear as exceptions in any member
            let mut label_groups: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (i, sig_id) in st.members.iter().enumerate() {
                for (lbl, _) in &sigs[*sig_id].ex {
                    label_groups.entry(*lbl).or_default().push(i);
                }
            }

            for (lbl, idxs) in label_groups {
                // Build removal vec = union of defaults for members having exception on lbl
                let mut rem_pairs: Vec<(usize, &Weight)> = Vec::with_capacity(idxs.len());
                let mut ex_pairs: Vec<(usize, &Weight)> = Vec::with_capacity(idxs.len());
                for i in idxs.iter().copied() {
                    let sig_id = st.members[i];
                    let gate = &st.gates[i];
                    if gate.is_empty() { continue; }
                    if let Some(def_id) = sigs[sig_id].def {
                        rem_pairs.push((def_id, gate));
                    }
                    if let Some(ex_id) = sigs[sig_id].ex.get(&lbl).copied() {
                        ex_pairs.push((ex_id, gate));
                    }
                }
                let removal = union_compiled_with_gates(&compiled_steps, &rem_pairs);
                let mut cur = subtract_sorted(&baseline, &removal);
                // Add exceptions
                let adds = union_compiled_with_gates(&compiled_steps, &ex_pairs);
                cur = add_sorted(&cur, &adds);

                if cur.is_empty() {
                    continue;
                }
                let mem: Vec<usize> = cur.iter().map(|(m, _)| *m).collect();
                if mem.is_empty() {
                    continue;
                }
                let mut init_gates: HashMap<usize, Weight> = HashMap::with_capacity(cur.len());
                for (m, w) in cur.iter() {
                    init_gates.insert(*m, w.clone());
                }
                let _ = ensure_state(mem, Some(init_gates), &mut states, &mut key_to_state, &mut work);
            }
        }
        if let Some(p) = pb_det {
            p.finish_with_message(format!("Determinized to {} states", states.len()));
        }

        // Build final DWA
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        for _ in 0..states.len() {
            dwa.states.add_state();
        }
        dwa.body.start_state = start_id;

        // Helper: compute final weight
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

        let pb_build = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(states.len() as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Build DWA)")
                    .expect("progress-bar style"),
            );
            Some(p)
        } else {
            None
        };

        // Emit edges
        for sid in 0..states.len() {
            let st = states[sid].clone();

            // Final
            dwa.states[sid].final_weight = compute_final(&st);

            // Default baseline
            let mut def_pairs: Vec<(usize, &Weight)> = Vec::new();
            def_pairs.reserve(st.members.len());
            for (i, sig_id) in st.members.iter().enumerate() {
                if let Some(def_id) = sigs[*sig_id].def {
                    let gate = &st.gates[i];
                    if !gate.is_empty() {
                        def_pairs.push((def_id, gate));
                    }
                }
            }
            let baseline = union_compiled_with_gates(&compiled_steps, &def_pairs);
            if !baseline.is_empty() {
                let mask = compute_mask(&baseline);
                if !mask.is_empty() {
                    let mut mem: Vec<usize> = baseline.iter().map(|(m, _)| *m).collect();
                    mem.sort_unstable();
                    let key = MembersKey::new(mem);
                    if let Some(&to_id) = key_to_state.get(&key) {
                        let _ = dwa.set_default_transition(sid, to_id, mask);
                    }
                }
            }

            // Labels
            let mut label_groups: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (i, sig_id) in st.members.iter().enumerate() {
                for (lbl, _) in &sigs[*sig_id].ex {
                    label_groups.entry(*lbl).or_default().push(i);
                }
            }
            for (lbl, idxs) in label_groups {
                let mut rem_pairs: Vec<(usize, &Weight)> = Vec::with_capacity(idxs.len());
                let mut ex_pairs: Vec<(usize, &Weight)> = Vec::with_capacity(idxs.len());
                for i in idxs.iter().copied() {
                    let sig_id = st.members[i];
                    let gate = &st.gates[i];
                    if gate.is_empty() { continue; }
                    if let Some(def_id) = sigs[sig_id].def {
                        rem_pairs.push((def_id, gate));
                    }
                    if let Some(ex_id) = sigs[sig_id].ex.get(&lbl).copied() {
                        ex_pairs.push((ex_id, gate));
                    }
                }
                let removal = union_compiled_with_gates(&compiled_steps, &rem_pairs);
                let mut cur = subtract_sorted(&baseline, &removal);
                let adds = union_compiled_with_gates(&compiled_steps, &ex_pairs);
                cur = add_sorted(&cur, &adds);

                if cur.is_empty() {
                    continue;
                }
                let mask = compute_mask(&cur);
                if mask.is_empty() {
                    continue;
                }
                let mut mem: Vec<usize> = cur.iter().map(|(m, _)| *m).collect();
                mem.sort_unstable();
                let key = MembersKey::new(mem);
                if let Some(&to_id) = key_to_state.get(&key) {
                    let _ = dwa.add_transition(sid, lbl, to_id, mask);
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
