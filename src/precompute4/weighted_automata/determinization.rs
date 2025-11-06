// src/precompute4/weighted_automata/determinization.rs
//
// A new determinization that avoids state explosion by:
// - Defining determinized states only by the set of contributing macro-signatures (structure),
//   not by the current gating weights (which are monotone and propagated to a fixpoint).
// - Propagating weights to those structural states until convergence (worklist), thus merging
//   all runs that reach the same structural subset instead of creating duplicate states.
// - Exploiting default transitions: compute a per-state default baseline once, then patch per-label
//   using only the differences (overrides and exceptions), avoiding per-label recomputation.
// - Precomputing ε-closures and compiling transition step-vectors once, heavily reusing them.
//
// This completely replaces the previous determinization and is designed to avoid building
// hundreds of thousands of redundant states. It also avoids per-edge mutation during fixpoint;
// we build the final DWA only after all weights converge.
//
// Semantics preserved for bitset weights with ∧ on path and ∨ over choices.

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
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        let mut nwa = self.clone();
        // Keep as-is; NWA::simplify may reduce epsilon chains, exceptions, etc.
        nwa.simplify();

        let result = nwa.determinize_fixpoint_structural();

        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }

    /// Determinize using structural subset states + weight fixpoint propagation.
    fn determinize_fixpoint_structural(&self) -> DWA {
        let fut = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        // Weighted ε-closure from a set of sources, pruned by fut[]
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

        type StepVecPairs = Vec<(NWAStateID, Weight)>;

        #[derive(Clone)]
        struct StepVecData {
            pairs: StepVecPairs,              // sorted by NWAStateID
            compiled_by_sig: Vec<(usize, Weight)>, // (macro_sig_id, weight)
            total_mask: Weight,
        }

        struct StepVecPool {
            data: Vec<StepVecData>,
            map: HashMap<u64, Vec<usize>>, // fp -> candidates
        }
        impl StepVecPool {
            fn new() -> Self { Self { data: Vec::new(), map: HashMap::new() } }
            fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
                let mut fp = FP_ZERO;
                for (sid, w) in pairs.iter() {
                    fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
                }
                fp
            }
            fn intern(&mut self, mut pairs: StepVecPairs) -> usize {
                pairs.retain(|(_, w)| !w.is_empty());
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

        // Apply additional weight to closure pairs, dropping empties
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

        // Macro signatures describe determinization-relevant behavior per NWA state
        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            def: Option<usize>,          // stepvec id
            ex: BTreeMap<i16, usize>,    // label -> stepvec id
        }

        // Build closures, step-vectors, macro signatures
        let mut step_pool = StepVecPool::new();
        let mut eps_cache: Vec<StepVecPairs> = vec![Vec::new(); n];
        for s in 0..n {
            eps_cache[s] = eps_closure_masked(std::slice::from_ref(&s), &self.states, &fut);
        }

        let mut sig_arena: Vec<MacroSig> = Vec::with_capacity(n);
        let mut state_to_sig_id: Vec<usize> = vec![0; n];

        // We also intern macro signatures structurally (without weights gates), but
        // we expect fewer unique macro signatures than raw states after simplify.
        #[derive(Clone)]
        struct MacroSigKey {
            fp: u64,
            final_fp: u64,
            def_id: Option<usize>,
            label_ids: Vec<(i16, usize)>,
        }
        impl MacroSigKey {
            fn from(sig: &MacroSig) -> Self {
                let final_fp = sig.final_w.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO);
                let mut label_ids: Vec<(i16, usize)> = sig.ex.iter().map(|(k, &v)| (*k, v)).collect();
                // BTreeMap iter is sorted already
                let mut fp = FP_ZERO;
                fp = mix3(fp, final_fp, 0xA55A_A55A_A55A_A55A);
                fp = mix3(fp, sig.def.map(|x| x as u64).unwrap_or(0), 0x5D_u64);
                for (lbl, sv) in &label_ids {
                    fp = mix3(fp, (*lbl as u64).wrapping_mul(FP_K1), (*sv as u64).wrapping_mul(FP_K2));
                }
                Self { fp, final_fp, def_id: sig.def, label_ids }
            }
        }
        impl PartialEq for MacroSigKey {
            fn eq(&self, o: &Self) -> bool {
                self.fp == o.fp && self.final_fp == o.final_fp && self.def_id == o.def_id && self.label_ids == o.label_ids
            }
        }
        impl Eq for MacroSigKey {}
        impl Hash for MacroSigKey {
            fn hash<H: Hasher>(&self, h: &mut H) { h.write_u64(self.fp); }
        }

        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();

        for s in 0..n {
            // final after ε-closure from s
            let mut final_acc: Option<Weight> = None;
            for (t, w) in eps_cache[s].iter() {
                if let Some(fw) = &self.states[*t].final_weight {
                    let c = w & fw;
                    if !c.is_empty() {
                        if let Some(ref mut a) = final_acc { *a |= &c; } else { final_acc = Some(c); }
                    }
                }
            }
            // default stepvec
            let mut def_id = None;
            if let Some((to, wdef)) = &self.states[s].default {
                if *to < n {
                    let pairs = apply_weight_to_pairs(&eps_cache[*to], wdef);
                    def_id = Some(step_pool.intern(pairs));
                }
            }
            // exception stepvecs
            let mut ex_map: BTreeMap<i16, usize> = BTreeMap::new();
            for (lbl, (to, wlbl)) in &self.states[s].transitions {
                if *to >= n { continue; }
                let pairs = apply_weight_to_pairs(&eps_cache[*to], wlbl);
                let id = step_pool.intern(pairs);
                ex_map.insert(*lbl, id);
            }
            let sig = MacroSig { final_w: final_acc, def: def_id, ex: ex_map };
            let key = MacroSigKey::from(&sig);
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

        // Compile step-vectors by signature
        let num_sigs = sig_arena.len();
        for id in 0..step_pool.len() {
            let pairs = &step_pool.data[id].pairs;
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() {
                let sid = state_to_sig_id[*t];
                match acc.entry(sid) {
                    Entry::Occupied(mut e) => {
                        let x = e.get_mut(); *x |= w;
                    }
                    Entry::Vacant(e) => { e.insert(w.clone()); }
                }
            }
            let mut compiled: Vec<(usize, Weight)> = acc.into_iter().collect();
            compiled.sort_by_key(|(sid, _)| *sid);
            let mut total = Weight::zeros();
            for (_, w) in compiled.iter() { total |= w; }
            step_pool.data[id].compiled_by_sig = compiled;
            step_pool.data[id].total_mask = total;
        }

        // Determinization with structural states + weight fixpoint

        // Determinized state: members (macro_sig ids) and current gates per member (union across all runs reached)
        #[derive(Clone)]
        struct DetState {
            members: Vec<usize>,              // sorted unique macro_sig ids
            pos: HashMap<usize, usize>,       // member sig_id -> index in vectors
            gates: Vec<Weight>,               // gates per member
        }

        // Key: members only (no weights)
        #[derive(Clone)]
        struct MembersKey {
            items: Vec<usize>,
            fp: u64,
        }
        impl MembersKey {
            fn new(items: Vec<usize>) -> Self {
                let mut v = items;
                v.sort_unstable();
                v.dedup();
                let mut fp = FP_ZERO;
                for sid in &v {
                    fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), 0xBEEF_CAFE_1234_5678);
                }
                MembersKey { items: v, fp }
            }
        }
        impl PartialEq for MembersKey {
            fn eq(&self, o: &Self) -> bool { self.fp == o.fp && self.items == o.items }
        }
        impl Eq for MembersKey {}
        impl Hash for MembersKey {
            fn hash<H: Hasher>(&self, h: &mut H) { h.write_u64(self.fp); }
        }

        struct Builder {
            sigs: Vec<MacroSig>,
            step_pool: StepVecPool,
        }
        let builder = Builder { sigs: sig_arena.clone(), step_pool };

        // Utility: accumulate compiled step-vector into map (sig_id -> weight) gated by 'gate'
        fn accumulate_compiled(
            acc: &mut HashMap<usize, Weight>,
            compiled: &[(usize, Weight)],
            gate: &Weight,
        ) {
            if gate.is_all_fast() {
                for (sig, w) in compiled.iter() {
                    match acc.entry(*sig) {
                        Entry::Occupied(mut e) => {
                            let x = e.get_mut(); *x |= w;
                        }
                        Entry::Vacant(e) => { e.insert(w.clone()); }
                    }
                }
            } else {
                for (sig, w) in compiled.iter() {
                    let x = w & gate;
                    if x.is_empty() { continue; }
                    match acc.entry(*sig) {
                        Entry::Occupied(mut e) => {
                            let y = e.get_mut(); *y |= &x;
                        }
                        Entry::Vacant(e) => { e.insert(x); }
                    }
                }
            }
        }

        // Multiway-merge cursor across exception maps
        struct ExCursor<'a> {
            lbl_iter: std::collections::btree_map::Iter<'a, i16, usize>,
            current: Option<(i16, usize)>,
            gate: Weight,                // gate for the owning signature
            def_ptr: Option<usize>,      // default stepvec id for the owning signature
        }
        // Min-heap by label using Reverse in Ord
        #[derive(Eq)]
        struct HeapItem { lbl: i16, idx: usize }
        impl PartialEq for HeapItem {
            fn eq(&self, o: &Self) -> bool { self.lbl == o.lbl && self.idx == o.idx }
        }
        impl Ord for HeapItem {
            fn cmp(&self, o: &Self) -> Ordering {
                o.lbl.cmp(&self.lbl).then_with(|| o.idx.cmp(&self.idx))
            }
        }
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
        }

        // States and mapping
        let mut states: Vec<DetState> = Vec::new();
        let mut key_to_state: HashMap<MembersKey, usize> = HashMap::new();
        let mut work: VecDeque<usize> = VecDeque::new();

        // Create or get structural state (with zero gates); if initial_gates provided, OR them in.
        fn ensure_state(
            members: Vec<usize>,
            initial_gates: Option<HashMap<usize, Weight>>,
            states: &mut Vec<DetState>,
            key_to_state: &mut HashMap<MembersKey, usize>,
            work: &mut VecDeque<usize>,
        ) -> usize {
            let key = MembersKey::new(members);
            if let Some(id) = key_to_state.get(&key).copied() {
                // Merge initial_gates into the existing state
                if let Some(init) = initial_gates {
                    let st = &mut states[id];
                    let mut changed = false;
                    for (sid, w) in init.into_iter() {
                        if let Some(&pos) = st.pos.get(&sid) {
                            let before = st.gates[pos].clone();
                            st.gates[pos] |= &w;
                            if st.gates[pos] != before { changed = true; }
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
            if let Some(init) = initial_gates {
                for (sid, w) in init.into_iter() {
                    if let Some(&idx) = pos.get(&sid) {
                        gates[idx] |= &w;
                    }
                }
            }
            let id = states.len();
            states.push(DetState { members: items, pos, gates });
            key_to_state.insert(key, id);
            work.push_back(id);
            id
        }

        // Initial state: ε-closure(start) grouped by signature with gates
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            let sid = state_to_sig_id[*t];
            match init_map.entry(sid) {
                Entry::Occupied(mut e) => {
                    let x = e.get_mut(); *x |= w;
                }
                Entry::Vacant(e) => { e.insert(w.clone()); }
            }
        }
        let init_members: Vec<usize> = {
            let mut keys: Vec<usize> = init_map.keys().copied().collect();
            keys.sort_unstable();
            keys
        };
        let _start_id = ensure_state(init_members, Some(init_map), &mut states, &mut key_to_state, &mut work);

        // Worklist: propagate weights to a fixpoint
        while let Some(sid) = work.pop_front() {
            let st_members = states[sid].members.clone();
            let st_gates = states[sid].gates.clone();

            // Precompute default baseline for this state: sum gates per def_ptr over members
            let mut def_ptr_total_gate: HashMap<usize, Weight> = HashMap::new();
            for (i, sig_id) in st_members.iter().enumerate() {
                if let Some(def_ptr) = builder.sigs[*sig_id].def {
                    match def_ptr_total_gate.entry(def_ptr) {
                        Entry::Occupied(mut e) => {
                            let x = e.get_mut(); *x |= &st_gates[i];
                        }
                        Entry::Vacant(e) => { e.insert(st_gates[i].clone()); }
                    }
                }
            }

            // Default target contributions: accumulate compiled step-vectors
            let mut def_target_map: HashMap<usize, Weight> = HashMap::new();
            for (ptr, gate) in def_ptr_total_gate.iter() {
                let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                accumulate_compiled(&mut def_target_map, comp, gate);
            }
            // Propagate default contributions
            if !def_target_map.is_empty() {
                let mut def_members: Vec<usize> = def_target_map.keys().copied().collect();
                def_members.sort_unstable();
                let def_init = def_target_map; // move
                let _ = ensure_state(def_members, Some(def_init), &mut states, &mut key_to_state, &mut work);
            }

            // Exceptions per label via multiway-merge
            let mut cursors: Vec<ExCursor> = Vec::new();
            for (i, sig_id) in st_members.iter().enumerate() {
                if builder.sigs[*sig_id].ex.is_empty() { continue; }
                let mut it = builder.sigs[*sig_id].ex.iter();
                let first = it.next().map(|(k, v)| (*k, *v));
                let cur = ExCursor {
                    lbl_iter: it,
                    current: first,
                    gate: st_gates[i].clone(),
                    def_ptr: builder.sigs[*sig_id].def,
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

            while let Some(HeapItem { lbl, idx }) = heap.pop() {
                // Collect all cursors at this label
                let mut idxs = vec![idx];
                while let Some(top) = heap.peek() {
                    if top.lbl == lbl { idxs.push(heap.pop().unwrap().idx); } else { break; }
                }

                // Aggregate ex_ptr gates and def_ptr overrides for this label
                let mut ex_map: HashMap<usize, Weight> = HashMap::new(); // ex_ptr -> gate
                let mut ov_map: HashMap<usize, Weight> = HashMap::new(); // def_ptr -> gate
                for i in idxs.iter().copied() {
                    if let Some((_l, ex_ptr)) = cursors[i].current {
                        match ex_map.entry(ex_ptr) {
                            Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= &cursors[i].gate; }
                            Entry::Vacant(e) => { e.insert(cursors[i].gate.clone()); }
                        }
                        if let Some(def_ptr) = cursors[i].def_ptr {
                            match ov_map.entry(def_ptr) {
                                Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= &cursors[i].gate; }
                                Entry::Vacant(e) => { e.insert(cursors[i].gate.clone()); }
                            }
                        }
                    }
                }

                // Build rem_map and add_map
                let mut rem_map: HashMap<usize, Weight> = HashMap::new();
                for (ptr, gate) in ov_map.iter() {
                    let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                    accumulate_compiled(&mut rem_map, comp, gate);
                }
                let mut add_map: HashMap<usize, Weight> = HashMap::new();
                for (ptr, gate) in ex_map.iter() {
                    let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                    accumulate_compiled(&mut add_map, comp, gate);
                }

                // Compose: label_target_map = def_target_map - rem_map + add_map
                // We don't have def_target_map here anymore (moved). Recompute it minimally for keys we need:
                // Instead of recomputing fully, we recompute full baseline again (costly but acceptable
                // because number of def_ptr per state is typically small). For performance, we just reuse our
                // precomputed 'def_ptr_total_gate' here.
                let mut base_map: HashMap<usize, Weight> = HashMap::new();
                for (ptr, gate) in def_ptr_total_gate.iter() {
                    let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                    accumulate_compiled(&mut base_map, comp, gate);
                }

                // Apply removal
                for (sig, rw) in rem_map.iter() {
                    if let Some(old) = base_map.get_mut(sig) {
                        *old -= rw;
                        if old.is_empty() {
                            base_map.remove(sig);
                        }
                    }
                }
                // Apply additions
                for (sig, aw) in add_map.iter() {
                    match base_map.entry(*sig) {
                        Entry::Occupied(mut e) => {
                            let v = e.get_mut(); *v |= aw;
                        }
                        Entry::Vacant(e) => { e.insert(aw.clone()); }
                    }
                }

                if !base_map.is_empty() {
                    let mut mem: Vec<usize> = base_map.keys().copied().collect();
                    mem.sort_unstable();
                    let _ = ensure_state(mem, Some(base_map), &mut states, &mut key_to_state, &mut work);
                }

                // Advance cursors
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

        // Build the final DWA after fixpoint propagation of gates
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        for _ in 0..states.len() { dwa.states.add_state(); }

        // Map: MembersKey -> state id already computed above; keep start id
        // Find start id
        let mut start_key_items: Vec<usize> = Vec::new();
        { // rebuild start key from start state's members
            let start_state_id = 0usize; // The first ensure_state call created start state and pushed it into states[0], but we didn't save id.
            // Find by scanning: the state whose gates were initialized from init_map equals earliest inserted.
            // Conservative: rebuild the key from eps closure and lookup.
            start_key_items = {
                let mut v: Vec<usize> = eps_cache[self.body.start_state].iter().map(|(t, _)| state_to_sig_id[*t]).collect();
                v.sort_unstable(); v.dedup(); v
            };
        }
        let key_start = MembersKey::new(start_key_items.clone());
        let start_id = *key_to_state.get(&key_start).unwrap_or(&0usize);
        dwa.body.start_state = start_id;

        // Helper to compute final weight for a structural state
        let compute_final = |st: &DetState, sigs: &Vec<MacroSig>| -> Option<Weight> {
            let mut acc: Option<Weight> = None;
            for (i, sig_id) in st.members.iter().enumerate() {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    let c = &st.gates[i] & fw;
                    if !c.is_empty() {
                        if let Some(ref mut a) = acc { *a |= &c; } else { acc = Some(c); }
                    }
                }
            }
            acc
        };

        // Add edges and finals
        for sid in 0..states.len() {
            let st_members = states[sid].members.clone();
            let st_gates = states[sid].gates.clone();

            // Final weight
            dwa.states[sid].final_weight = compute_final(&states[sid], &builder.sigs);

            // Precompute default baseline for this state
            let mut def_ptr_total_gate: HashMap<usize, Weight> = HashMap::new();
            for (i, sig_id) in st_members.iter().enumerate() {
                if let Some(def_ptr) = builder.sigs[*sig_id].def {
                    match def_ptr_total_gate.entry(def_ptr) {
                        Entry::Occupied(mut e) => { let x = e.get_mut(); *x |= &st_gates[i]; }
                        Entry::Vacant(e) => { e.insert(st_gates[i].clone()); }
                    }
                }
            }

            let mut def_target_map: HashMap<usize, Weight> = HashMap::new();
            let mut def_edge_w = Weight::zeros();
            for (ptr, gate) in def_ptr_total_gate.iter() {
                let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                accumulate_compiled(&mut def_target_map, comp, gate);
                let m = &builder.step_pool.data[*ptr].total_mask;
                let x = gate & m;
                def_edge_w |= &x;
            }
            if !def_target_map.is_empty() && !def_edge_w.is_empty() {
                let mut mem: Vec<usize> = def_target_map.keys().copied().collect();
                mem.sort_unstable();
                let def_key = MembersKey::new(mem);
                if let Some(&to_id) = key_to_state.get(&def_key) {
                    let _ = dwa.set_default_transition(sid, to_id, def_edge_w);
                }
            }

            // Exceptions: multiway-merge
            let mut cursors: Vec<ExCursor> = Vec::new();
            for (i, sig_id) in st_members.iter().enumerate() {
                if builder.sigs[*sig_id].ex.is_empty() { continue; }
                let mut it = builder.sigs[*sig_id].ex.iter();
                let first = it.next().map(|(k, v)| (*k, *v));
                let cur = ExCursor {
                    lbl_iter: it,
                    current: first,
                    gate: st_gates[i].clone(),
                    def_ptr: builder.sigs[*sig_id].def,
                };
                if cur.current.is_some() { cursors.push(cur); }
            }
            if cursors.is_empty() { continue; }
            let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
            for (idx, c) in cursors.iter().enumerate() {
                if let Some((lbl, _)) = c.current {
                    heap.push(HeapItem { lbl, idx });
                }
            }

            while let Some(HeapItem { lbl, idx }) = heap.pop() {
                let mut idxs = vec![idx];
                while let Some(top) = heap.peek() {
                    if top.lbl == lbl { idxs.push(heap.pop().unwrap().idx); } else { break; }
                }

                let mut ex_map: HashMap<usize, Weight> = HashMap::new();
                let mut ov_map: HashMap<usize, Weight> = HashMap::new();
                for i in idxs.iter().copied() {
                    if let Some((_l, ex_ptr)) = cursors[i].current {
                        match ex_map.entry(ex_ptr) {
                            Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= &cursors[i].gate; }
                            Entry::Vacant(e) => { e.insert(cursors[i].gate.clone()); }
                        }
                        if let Some(def_ptr) = cursors[i].def_ptr {
                            match ov_map.entry(def_ptr) {
                                Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= &cursors[i].gate; }
                                Entry::Vacant(e) => { e.insert(cursors[i].gate.clone()); }
                            }
                        }
                    }
                }

                // Build base map (default baseline) and edge mask baseline
                let mut base_map: HashMap<usize, Weight> = HashMap::new();
                let mut base_edge_w = Weight::zeros();
                for (ptr, gate) in def_ptr_total_gate.iter() {
                    let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                    accumulate_compiled(&mut base_map, comp, gate);
                    let m = &builder.step_pool.data[*ptr].total_mask;
                    let x = gate & m;
                    base_edge_w |= &x;
                }

                // Removals
                let mut rem_map: HashMap<usize, Weight> = HashMap::new();
                let mut rem_edge_w = Weight::zeros();
                for (ptr, gate) in ov_map.iter() {
                    let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                    accumulate_compiled(&mut rem_map, comp, gate);
                    let m = &builder.step_pool.data[*ptr].total_mask;
                    let x = gate & m;
                    rem_edge_w |= &x;
                }
                for (sig, rw) in rem_map.iter() {
                    if let Some(old) = base_map.get_mut(sig) {
                        *old -= rw;
                        if old.is_empty() { base_map.remove(sig); }
                    }
                }
                let mut edge_w = base_edge_w;
                edge_w -= &rem_edge_w;

                // Additions
                let mut add_map: HashMap<usize, Weight> = HashMap::new();
                let mut add_edge_w = Weight::zeros();
                for (ptr, gate) in ex_map.iter() {
                    let comp = &builder.step_pool.data[*ptr].compiled_by_sig;
                    accumulate_compiled(&mut add_map, comp, gate);
                    let m = &builder.step_pool.data[*ptr].total_mask;
                    let x = gate & m;
                    add_edge_w |= &x;
                }
                for (sig, aw) in add_map.into_iter() {
                    match base_map.entry(sig) {
                        Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= &aw; }
                        Entry::Vacant(e) => { e.insert(aw); }
                    }
                }
                edge_w |= &add_edge_w;

                if edge_w.is_empty() || base_map.is_empty() {
                    // No effective labeled edge
                } else {
                    let mut mem: Vec<usize> = base_map.keys().copied().collect();
                    mem.sort_unstable();
                    let ex_key = MembersKey::new(mem);
                    if let Some(&to_id) = key_to_state.get(&ex_key) {
                        let _ = dwa.add_transition(sid, lbl, to_id, edge_w);
                    }
                }

                // Advance cursors
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
