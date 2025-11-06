// src/precompute4/weighted_automata/determinization.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::{StateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use std::collections::{HashMap, VecDeque};

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        // Lean on-the-fly determinization without global precomputation or heavy interning.
        self.internal_determinize_to_dwa_fast()
    }

    /// Internal: lean, on-the-fly determinization with ε-closure and zero global precomputation.
    /// Builds DWA directly by exploring only reachable subsets, using generation-stamped scratch arrays.
    fn internal_determinize_to_dwa_fast(&self) -> DWA {
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }
        // Scratch state for ε-closures and accumulators (generation-stamped arrays).
        struct Determinizer<'a> {
            states: &'a NWAStates,
            n: usize,
            // ε-closure scratch
            clos_w: Vec<Weight>,
            clos_gen: Vec<u32>,
            clos_cur: u32,
            // aggregation scratch for pre-targets
            agg_w: Vec<Weight>,
            agg_gen: Vec<u32>,
            agg_cur: u32,
            // work queues & touched sets
            q: VecDeque<usize>,
            touched: Vec<usize>,
            tgt_touched: Vec<usize>,
        }
        impl<'a> Determinizer<'a> {
            fn new(states: &'a NWAStates) -> Self {
                let n = states.len();
                Self {
                    states,
                    n,
                    clos_w: vec![Weight::zeros(); n],
                    clos_gen: vec![0; n],
                    clos_cur: 1,
                    agg_w: vec![Weight::zeros(); n],
                    agg_gen: vec![0; n],
                    agg_cur: 1,
                    q: VecDeque::with_capacity(64),
                    touched: Vec::with_capacity(64),
                    tgt_touched: Vec::with_capacity(64),
                }
            }
            #[inline]
            fn bump_closure(&mut self) {
                self.clos_cur = self.clos_cur.wrapping_add(1);
                if self.clos_cur == 0 {
                    // rare wraparound: reset gens
                    self.clos_gen.fill(0);
                    self.clos_cur = 1;
                }
                self.touched.clear();
                self.q.clear();
            }
            #[inline]
            fn bump_agg(&mut self) {
                self.agg_cur = self.agg_cur.wrapping_add(1);
                if self.agg_cur == 0 {
                    self.agg_gen.fill(0);
                    self.agg_cur = 1;
                }
                self.tgt_touched.clear();
            }
            // Compute ε-closure from initial pairs, return (out_pairs sorted, union_final_weight)
            fn eps_closure_from(&mut self, init: &[(usize, Weight)], out_pairs: &mut Vec<(usize, Weight)>) -> Option<Weight> {
                self.bump_closure();
                out_pairs.clear();
                // seed
                for &(s, ref w) in init {
                    if w.is_empty() { continue; }
                    if s >= self.n { continue; }
                    if self.clos_gen[s] != self.clos_cur {
                        self.clos_gen[s] = self.clos_cur;
                        self.clos_w[s] = w.clone();
                        self.q.push_back(s);
                        self.touched.push(s);
                    } else {
                        let neww = &self.clos_w[s] | w;
                        if neww != self.clos_w[s] {
                            self.clos_w[s] = neww;
                            self.q.push_back(s);
                        }
                    }
                }
                let mut final_union: Option<Weight> = None;
                while let Some(u) = self.q.pop_front() {
                    let uw = self.clos_w[u].clone();
                    if let Some(fw) = &self.states[u].final_weight {
                        let c = &uw & fw;
                        if !c.is_empty() {
                            if let Some(ref mut a) = final_union {
                                *a |= &c;
                            } else {
                                final_union = Some(c);
                            }
                        }
                    }
                    for &(v, ref weps) in &self.states[u].epsilons {
                        if v >= self.n { continue; }
                        let x = &uw & weps;
                        if x.is_empty() { continue; }
                        if self.clos_gen[v] != self.clos_cur {
                            self.clos_gen[v] = self.clos_cur;
                            self.clos_w[v] = x;
                            self.q.push_back(v);
                            self.touched.push(v);
                        } else {
                            let nu = &self.clos_w[v] | &x;
                            if nu != self.clos_w[v] {
                                self.clos_w[v] = nu;
                                self.q.push_back(v);
                            }
                        }
                    }
                }
                // collect
                self.touched.sort_unstable();
                for &v in &self.touched {
                    let w = self.clos_w[v].clone();
                    if !w.is_empty() {
                        out_pairs.push((v, w));
                    }
                }
                final_union
            }
            // Aggregate default contributions for a subset, then ε-close them.
            fn build_default_succ(&mut self, subset: &[(usize, Weight)], out_pairs: &mut Vec<(usize, Weight)>) -> Option<Weight> {
                self.bump_agg();
                for &(s, ref ws) in subset {
                    if s >= self.n { continue; }
                    if let Some((to, ref wd)) = self.states[s].default {
                        let x = ws & wd;
                        if x.is_empty() { continue; }
                        let t = to;
                        if self.agg_gen[t] != self.agg_cur {
                            self.agg_gen[t] = self.agg_cur;
                            self.agg_w[t] = x;
                            self.tgt_touched.push(t);
                        } else {
                            let nu = &self.agg_w[t] | &x;
                            if nu != self.agg_w[t] {
                                self.agg_w[t] = nu;
                            }
                        }
                    }
                }
                if self.tgt_touched.is_empty() {
                    out_pairs.clear();
                    return None;
                }
                let mut pre: Vec<(usize, Weight)> = Vec::with_capacity(self.tgt_touched.len());
                for &t in &self.tgt_touched {
                    pre.push((t, self.agg_w[t].clone()));
                }
                self.eps_closure_from(&pre, out_pairs)
            }
            // Aggregate contributions for a specific label, then ε-close them.
            fn build_label_succ(&mut self, subset: &[(usize, Weight)], label: i16, out_pairs: &mut Vec<(usize, Weight)>) -> Option<Weight> {
                self.bump_agg();
                for &(s, ref ws) in subset {
                    if s >= self.n { continue; }
                    // exception on label or default
                    let mut used = false;
                    if let Some(&(to, ref wl)) = self.states[s].transitions.get(&label) {
                        let x = ws & wl;
                        if !x.is_empty() {
                            let t = to;
                            if self.agg_gen[t] != self.agg_cur {
                                self.agg_gen[t] = self.agg_cur;
                                self.agg_w[t] = x;
                                self.tgt_touched.push(t);
                            } else {
                                let nu = &self.agg_w[t] | &x;
                                if nu != self.agg_w[t] {
                                    self.agg_w[t] = nu;
                                }
                            }
                            used = true;
                        }
                    }
                    if !used {
                        if let Some((to, ref wd)) = self.states[s].default {
                            let x = ws & wd;
                            if !x.is_empty() {
                                let t = to;
                                if self.agg_gen[t] != self.agg_cur {
                                    self.agg_gen[t] = self.agg_cur;
                                    self.agg_w[t] = x;
                                    self.tgt_touched.push(t);
                                } else {
                                    let nu = &self.agg_w[t] | &x;
                                    if nu != self.agg_w[t] {
                                        self.agg_w[t] = nu;
                                    }
                                }
                            }
                        }
                    }
                }
                if self.tgt_touched.is_empty() {
                    out_pairs.clear();
                    return None;
                }
                let mut pre: Vec<(usize, Weight)> = Vec::with_capacity(self.tgt_touched.len());
                for &t in &self.tgt_touched {
                    pre.push((t, self.agg_w[t].clone()));
                }
                self.eps_closure_from(&pre, out_pairs)
            }
            // Collect distinct exception labels present in the subset.
            fn collect_labels(&self, subset: &[(usize, Weight)], out: &mut Vec<i16>) {
                out.clear();
                for &(s, _) in subset {
                    if s >= self.n { continue; }
                    for (&lbl, _) in &self.states[s].transitions {
                        out.push(lbl);
                    }
                }
                if out.is_empty() { return; }
                out.sort_unstable();
                out.dedup();
            }
        }

        // Subset interning via fingerprint -> list of indexes; index -> DWA state id
        fn subset_fp(items: &[(usize, Weight)]) -> u64 {
            let mut fp = FP_ZERO;
            for (sid, w) in items.iter() {
                fp = mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2));
            }
            fp
        }
        let mut det = Determinizer::new(&self.states);

        // Build start subset: ε-closure from (start, ALL)
        let mut start_pairs: Vec<(usize, Weight)> = Vec::with_capacity(8);
        let start_init = vec![(self.body.start_state, Weight::all())];
        let start_final = det.eps_closure_from(&start_init, &mut start_pairs);
        // Normalize start_pairs already sorted from eps_closure_from

        // DWA storage
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let start_dwa = dwa.states.add_state();
        dwa.body.start_state = start_dwa;
        dwa.states[start_dwa].final_weight = start_final;

        // subset arena and map
        let mut subsets: Vec<Vec<(usize, Weight)>> = Vec::with_capacity(1024);
        let mut subset_fp_map: HashMap<u64, Vec<usize>> = HashMap::with_capacity(2048);
        let mut subset_to_state: Vec<StateID> = Vec::with_capacity(1024);
        let mut work: VecDeque<usize> = VecDeque::with_capacity(1024);

        // intern start subset
        let mut intern_subset = |items: &[(usize, Weight)],
                                 subsets: &mut Vec<Vec<(usize, Weight)>>,
                                 subset_fp_map: &mut HashMap<u64, Vec<usize>>,
                                 subset_to_state: &mut Vec<StateID>,
                                 dwa: &mut DWA| -> usize {
            let fp = subset_fp(items);
            if let Some(bucket) = subset_fp_map.get(&fp) {
                'search: for &idx in bucket {
                    let cand = &subsets[idx];
                    if cand.len() == items.len() {
                        let mut eq = true;
                        for i in 0..items.len() {
                            if cand[i].0 != items[i].0 || cand[i].1 != items[i].1 {
                                eq = false; break;
                            }
                        }
                        if eq { return idx; }
                    }
                }
            }
            let idx = subsets.len();
            subsets.push(items.to_vec());
            subset_to_state.push(dwa.states.add_state());
            subset_fp_map.entry(fp).or_default().push(idx);
            idx
        };

        let start_idx = intern_subset(&start_pairs, &mut subsets, &mut subset_fp_map, &mut subset_to_state, &mut dwa);
        // sync start dve id
        subset_to_state[start_idx] = start_dwa;
        work.push_back(start_idx);

        let mut labels: Vec<i16> = Vec::with_capacity(64);
        let mut succ_pairs: Vec<(usize, Weight)> = Vec::with_capacity(64);
        let mut def_pairs: Vec<(usize, Weight)> = Vec::with_capacity(64);

        while let Some(idx) = work.pop_front() {
            let src_pairs = subsets[idx].clone();
            let src_dwa_id = subset_to_state[idx];

            // Final weight for this DWA state is already embedded during creation if it is the start.
            // For newly created states, set it now (ensures correctness if state was interned before final known).
            if dwa.states[src_dwa_id].final_weight.is_none() {
                let fin = {
                    // Compute ∪ (closure_w[s] ∧ final[s]) across the stored subset
                    let mut acc: Option<Weight> = None;
                    for &(s, ref w) in &src_pairs {
                        if let Some(fw) = &self.states[s].final_weight {
                            let c = w & fw;
                            if !c.is_empty() {
                                if let Some(ref mut a) = acc {
                                    *a |= &c;
                                } else {
                                    acc = Some(c);
                                }
                            }
                        }
                    }
                    acc
                };
                dwa.states[src_dwa_id].final_weight = fin;
            }

            // Default successor
            let def_fin = det.build_default_succ(&src_pairs, &mut def_pairs);
            let mut def_target_id: Option<StateID> = None;
            let mut def_edge_w: Option<Weight> = None;
            if !def_pairs.is_empty() {
                // compute edge weight = union across succ weights
                for &(_, ref w) in &def_pairs {
                    if let Some(ref mut a) = def_edge_w {
                        *a |= w;
                    } else {
                        def_edge_w = Some(w.clone());
                    }
                }
                let def_idx = intern_subset(&def_pairs, &mut subsets, &mut subset_fp_map, &mut subset_to_state, &mut dwa);
                let tgt_id = subset_to_state[def_idx];
                def_target_id = Some(tgt_id);
                if let Some(w) = def_edge_w.clone() {
                    // Set default edge
                    let _ = dwa.set_default_transition(src_dwa_id, tgt_id, w);
                }
            }

            // Exception labels (only those present among members' exception maps)
            det.collect_labels(&src_pairs, &mut labels);
            if !labels.is_empty() {
                for lbl in labels.drain(..) {
                    let _lbl_succ_final = det.build_label_succ(&src_pairs, lbl, &mut succ_pairs);
                    if succ_pairs.is_empty() {
                        continue;
                    }
                    // edge weight = union across succ weights
                    let mut e_w: Option<Weight> = None;
                    for &(_, ref w) in &succ_pairs {
                        if let Some(ref mut a) = e_w {
                            *a |= w;
                        } else {
                            e_w = Some(w.clone());
                        }
                    }
                    if let Some(w) = e_w {
                        let succ_idx = intern_subset(&succ_pairs, &mut subsets, &mut subset_fp_map, &mut subset_to_state, &mut dwa);
                        let succ_id = subset_to_state[succ_idx];
                        // Optionally skip explicit edge if same as default target and weight
                        let skip = if let Some(dt) = def_target_id {
                            if dt == succ_id {
                                if let Some(ref dw) = def_edge_w {
                                    w == *dw
                                } else { false }
                            } else { false }
                        } else { false };
                        if !skip {
                            let _ = dwa.add_transition(src_dwa_id, lbl, succ_id, w);
                        }
                        // Enqueue if new
                        if succ_idx >= work.len() && succ_idx >= subsets.len() {
                            // impossible; safety
                        }
                        // Worklist membership is tracked by whether we created a new DWA state inside intern_subset.
                        // To enqueue newly created subsets, check if we already assigned a DWA state at creation time:
                        // That is always true; enqueue if this is the first time we see it (i.e., when we created it just now).
                    }
                }
            }

            // Enqueue any newly created subset indexes that do not yet have transitions explored.
            // We can detect newly created subsets by comparing subset_to_state length against subsets length at time of insertion,
            // but a simpler approach is to push onto the queue immediately upon creation within intern_subset.
            // We emulate that here by scanning recent tails (small overhead).
            while subset_to_state.len() > work.len() + 1 {
                // Ensure we eventually process all subsets; push in order.
                let next = work.len();
                work.push_back(next);
            }
        }

        dwa
    }
}
