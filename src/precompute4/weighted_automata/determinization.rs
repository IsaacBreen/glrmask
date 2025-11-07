use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

// Determinization guardrails:
// - Maximum number of composition nodes to create. Past this, we collapse new targets
//   into a single overflow node which conservatively over-approximates behavior.
const DET_STATE_BUDGET: usize = 250_000;
const DET_ENABLE_OVERFLOW: bool = true;

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

        // (Helper functions eps_closure_masked, apply_weight_to_pairs, StepPool remain the same)
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
        struct StepPool {
            raw: Vec<Vec<(NWAStateID, Weight)>>,
            map: HashMap<u64, Vec<usize>>,
        }
        // =======================================================================================
        // FIX: Added back the `impl StepPool` block with the missing `len` method and fixed syntax
        // =======================================================================================
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
                let fp = Self::fingerprint(&pairs); // Corrected syntax: Self::
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
            fn len(&self) -> usize { self.raw.len() } // Added missing method
        }

        // (MacroSig and related structs remain the same)
        #[derive(Clone)]
        struct CompiledStep {
            by_sig: Vec<(usize, Weight)>,
            mask: Weight,
        }
        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            def: Option<usize>,
            ex: BTreeMap<i16, usize>,
        }
        #[derive(Clone, Hash, Eq, PartialEq)]
        struct MacroSigKey {
            final_fp: u64,
            def: Option<usize>,
            ex: Vec<(i16, usize)>,
        }

        // (Pre-computation of eps_cache, sigs, compiled_steps remains the same)
        let pb_eps = if PROGRESS_BAR_ENABLED { Some(ProgressBar::new(n as u64).with_style(ProgressStyle::default_bar().template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (ε-closures)").unwrap())) } else { None };
        let mut eps_cache: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        for s in 0..n { eps_cache[s] = eps_closure_masked(std::slice::from_ref(&s), &self.states, &fut); if let Some(p) = &pb_eps { p.inc(1); } }
        if let Some(p) = pb_eps { p.finish_with_message("ε-closures done"); }
        let mut step_pool = StepPool::new();
        let mut sigs: Vec<MacroSig> = Vec::with_capacity(n);
        let mut state_to_sig_id: Vec<usize> = vec![0; n];
        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();
        let pb_sigs = if PROGRESS_BAR_ENABLED { Some(ProgressBar::new(n as u64).with_style(ProgressStyle::default_bar().template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Macro signatures)").unwrap())) } else { None };
        for s in 0..n {
            let mut final_acc: Option<Weight> = None;
            for (t, w) in eps_cache[s].iter() { if let Some(fw) = &self.states[*t].final_weight { let c = w & fw; if !c.is_empty() { if let Some(ref mut a) = final_acc { *a |= &c; } else { final_acc = Some(c); } } } }
            let def = if let Some((to, wdef)) = &self.states[s].default { if *to < n { Some(step_pool.intern(apply_weight_to_pairs(&eps_cache[*to], wdef))) } else { None } } else { None };
            let mut ex: BTreeMap<i16, usize> = BTreeMap::new();
            for (lbl, (to, wlbl)) in &self.states[s].transitions { if *to >= n { continue; } let id = step_pool.intern(apply_weight_to_pairs(&eps_cache[*to], wlbl)); ex.insert(*lbl, id); }
            let key = MacroSigKey { final_fp: final_acc.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO), def, ex: ex.iter().map(|(k, v)| (*k, *v)).collect(), };
            let sig_id = match sig_intern.entry(key) { Entry::Occupied(o) => *o.get(), Entry::Vacant(v) => { let id = sigs.len(); sigs.push(MacroSig { final_w: final_acc.clone(), def, ex: ex.clone() }); v.insert(id); id } };
            state_to_sig_id[s] = sig_id;
            if let Some(p) = &pb_sigs { p.inc(1); }
        }
        if let Some(p) = pb_sigs { p.finish_with_message("Macro signatures done"); }
        let pb_compile = if PROGRESS_BAR_ENABLED { Some(ProgressBar::new(step_pool.len() as u64).with_style(ProgressStyle::default_bar().template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Compile steps)").unwrap())) } else { None };
        let mut compiled_steps: Vec<CompiledStep> = vec![CompiledStep { by_sig: Vec::new(), mask: Weight::zeros() }; step_pool.len()];
        for id in 0..step_pool.len() {
            let pairs = &step_pool.raw[id];
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() { let sid = state_to_sig_id[*t]; match acc.entry(sid) { Entry::Occupied(mut e) => { let v = e.get_mut(); *v |= w; } Entry::Vacant(e) => { e.insert(w.clone()); } } }
            let mut by_sig: Vec<(usize, Weight)> = acc.into_iter().collect();
            by_sig.sort_by_key(|(sid, _)| *sid);
            let mut mask = Weight::zeros();
            for (_, w) in &by_sig { mask |= w; }
            compiled_steps[id] = CompiledStep { by_sig, mask };
            if let Some(p) = &pb_compile { p.inc(1); }
        }
        if let Some(p) = pb_compile { p.finish_with_message("Compile steps done"); }

        // =======================================================================================
        // NEW: Iterative Partition Refinement Logic
        // =======================================================================================

        #[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
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

        // Represents a node in our "composition graph"
        struct CompositionNode {
            key: MembersKey,
            final_weight: Option<Weight>,
            default_target_idx: Option<usize>,
            exception_targets: BTreeMap<i16, usize>, // label -> target index
            gates: HashMap<usize, Weight>, // sig_id -> weight
        }

        // Helper to accumulate weights for a target state composition.
        fn accumulate(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
            if gate.is_all_fast() {
                for (sid, w) in compiled.iter() {
                    match dst.entry(*sid) {
                        Entry::Occupied(mut e) => {
                            let v = e.get_mut();
                            *v |= w;
                        }
                        Entry::Vacant(e) => {
                            e.insert(w.clone());
                        }
                    }
                }
            } else {
                for (sid, w) in compiled.iter() {
                    let x = w & gate;
                    if x.is_empty() {
                        continue;
                    }
                    match dst.entry(*sid) {
                        Entry::Occupied(mut e) => {
                            let v = e.get_mut();
                            *v |= &x;
                        }
                        Entry::Vacant(e) => {
                            e.insert(x);
                        }
                    }
                }
            }
        }

        // --- Phase 1: Discover all reachable compositions and their transitions ---
        let mut nodes: Vec<CompositionNode> = Vec::new();
        let mut key_to_idx: HashMap<MembersKey, usize> = HashMap::new();
        let mut work: VecDeque<usize> = VecDeque::new();

        let mut in_queue: Vec<bool> = Vec::new();
        let mut overflow_idx: Option<usize> = None;

        let pb_discover = if PROGRESS_BAR_ENABLED { Some(ProgressBar::new(0).with_style(ProgressStyle::default_bar().template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Discovering states)").unwrap())) } else { None };

        // Create the initial state
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            let sid = state_to_sig_id[*t];
            match init_map.entry(sid) {
                Entry::Occupied(mut e) => {
                    let v = e.get_mut();
                    *v |= w;
                }
                Entry::Vacant(e) => {
                    e.insert(w.clone());
                }
            }
        }
        let init_key = MembersKey::new(init_map.keys().copied().collect());
        let start_idx = 0;
        key_to_idx.insert(init_key.clone(), start_idx);
        nodes.push(CompositionNode { key: init_key, final_weight: None, default_target_idx: None, exception_targets: BTreeMap::new(), gates: init_map });
        in_queue.push(true);
        work.push_back(start_idx);

        while let Some(idx) = work.pop_front() {
            in_queue[idx] = false;
            if let Some(p) = &pb_discover { p.inc(1); p.set_length(nodes.len() as u64); }
            let node_gates = nodes[idx].gates.clone();

            // Compute all possible target compositions from this node
            // Build default target map once (if any)
            let mut def_map: HashMap<usize, Weight> = HashMap::new();
            for (sig_id, gate) in &node_gates {
                if let Some(def_id) = sigs[*sig_id].def {
                    accumulate(&mut def_map, &compiled_steps[def_id].by_sig, gate);
                }
            }
            // Gather all exception labels present among members
            let mut all_ex_labels = BTreeSet::new();
            for (sig_id, _) in &node_gates {
                for lbl in sigs[*sig_id].ex.keys() {
                    all_ex_labels.insert(*lbl);
                }
            }

            // Reset transitions from this node; we will recompute them
            nodes[idx].default_target_idx = None;
            nodes[idx].exception_targets.clear();

            // Helper to intern or overflow a target map, update gates, and return its index
            let mut intern_target = |map: &HashMap<usize, Weight>,
                                     nodes: &mut Vec<CompositionNode>,
                                     key_to_idx: &mut HashMap<MembersKey, usize>,
                                     in_queue: &mut Vec<bool>,
                                     work: &mut VecDeque<usize>,
                                     overflow_idx: &mut Option<usize>,
                                     sigs_len: usize| -> usize {
                if map.is_empty() {
                    return usize::MAX;
                }
                let key = MembersKey::new(map.keys().copied().collect());
                if let Some(&tidx) = key_to_idx.get(&key) {
                    return tidx;
                }
                if DET_ENABLE_OVERFLOW && nodes.len() >= DET_STATE_BUDGET {
                    // Create overflow node if missing
                    if overflow_idx.is_none() {
                        let all: Vec<usize> = (0..sigs_len).collect();
                        let ov_key = MembersKey::new(all);
                        let new_idx = nodes.len();
                        nodes.push(CompositionNode {
                            key: ov_key.clone(),
                            final_weight: None,
                            default_target_idx: None,
                            exception_targets: BTreeMap::new(),
                            gates: HashMap::new(),
                        });
                        key_to_idx.insert(ov_key, new_idx);
                        if new_idx >= in_queue.len() { in_queue.resize(new_idx + 1, false); }
                        *overflow_idx = Some(new_idx);
                    }
                    return overflow_idx.unwrap();
                }
                // Create a fresh node
                let new_idx = nodes.len();
                nodes.push(CompositionNode {
                    key: key.clone(),
                    final_weight: None,
                    default_target_idx: None,
                    exception_targets: BTreeMap::new(),
                    gates: HashMap::new(),
                });
                key_to_idx.insert(key, new_idx);
                if new_idx >= in_queue.len() { in_queue.resize(new_idx + 1, false); }
                new_idx
            };

            // 1) Default transition (if any)
            if !def_map.is_empty() {
                let target_idx = intern_target(&def_map, &mut nodes, &mut key_to_idx, &mut in_queue, &mut work, &mut overflow_idx, sigs.len());
                if target_idx != usize::MAX {
                    // Update gates in target node and enqueue if changed
                    let mut any_change = false;
                    for (sig_id, weight) in &def_map {
                        let entry = nodes[target_idx].gates.entry(*sig_id).or_insert_with(Weight::zeros);
                        let new_w = &*entry | weight;
                        if new_w != *entry {
                            *entry = new_w;
                            any_change = true;
                        }
                    }
                    if any_change && !in_queue[target_idx] {
                        in_queue[target_idx] = true;
                        work.push_back(target_idx);
                    }
                    nodes[idx].default_target_idx = Some(target_idx);
                }
            }

            // 2) Exception transitions per label (built from scratch without cloning def_map)
            for lbl in all_ex_labels {
                let mut map: HashMap<usize, Weight> = HashMap::new();
                for (sig_id, gate) in &node_gates {
                    if let Some(ex_id) = sigs[*sig_id].ex.get(&lbl) {
                        accumulate(&mut map, &compiled_steps[*ex_id].by_sig, gate);
                    } else if let Some(def_id) = sigs[*sig_id].def {
                        accumulate(&mut map, &compiled_steps[def_id].by_sig, gate);
                    }
                }
                if map.is_empty() {
                    continue;
                }
                let target_idx = intern_target(&map, &mut nodes, &mut key_to_idx, &mut in_queue, &mut work, &mut overflow_idx, sigs.len());
                if target_idx != usize::MAX {
                    let mut any_change = false;
                    for (sig_id, weight) in &map {
                        let entry = nodes[target_idx].gates.entry(*sig_id).or_insert_with(Weight::zeros);
                        let new_w = &*entry | weight;
                        if new_w != *entry {
                            *entry = new_w;
                            any_change = true;
                        }
                    }
                    if any_change && !in_queue[target_idx] {
                        in_queue[target_idx] = true;
                        work.push_back(target_idx);
                    }
                    nodes[idx].exception_targets.insert(lbl, target_idx);
                }
            }

            // Recompute final weight for this node based on its (possibly updated) gates
            let mut final_acc: Option<Weight> = None;
            for (sig_id, gate) in &nodes[idx].gates {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    let x = gate & fw;
                    if !x.is_empty() {
                        if let Some(ref mut a) = final_acc { *a |= &x; } else { final_acc = Some(x); }
                    }
                }
            }
            nodes[idx].final_weight = final_acc;

            // Ensure progress bar length reflects latest node count
            if let Some(p) = &pb_discover { p.set_length(nodes.len() as u64); }
        }
        if let Some(p) = pb_discover { p.finish_with_message(format!("Discovered {} compositions", nodes.len())); }

        // Final pass: ensure final weights are consistent (already computed in loop; recompute defensively)
        for node in &mut nodes {
            let mut final_acc: Option<Weight> = None;
            for (sig_id, gate) in &node.gates {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    let x = gate & fw;
                    if !x.is_empty() {
                        if let Some(ref mut a) = final_acc { *a |= &x; } else { final_acc = Some(x); }
                    }
                }
            }
            node.final_weight = final_acc;
        }

        // --- Phase 2: Partition Refinement ---
        let num_nodes = nodes.len();
        let mut partitions = vec![0; num_nodes];

        // Initial partition based on final weight
        let mut canon: HashMap<Option<Weight>, usize> = HashMap::new();
        for i in 0..num_nodes {
            let next_id = canon.len();
            partitions[i] = *canon.entry(nodes[i].final_weight.clone()).or_insert(next_id);
        }

        loop {
            let mut changed = false;
            let mut next_partitions = vec![0; num_nodes];
            let mut sig_to_part: HashMap<_, usize> = HashMap::new();

            for i in 0..num_nodes {
                let node = &nodes[i];
                let def_part = node.default_target_idx.map(|idx| partitions[idx]);
                let ex_parts: BTreeMap<_, _> = node.exception_targets.iter()
                    .map(|(lbl, idx)| (*lbl, partitions[*idx]))
                    .collect();

                let signature = (partitions[i], def_part, ex_parts);
                let next_id = sig_to_part.len();
                next_partitions[i] = *sig_to_part.entry(signature).or_insert(next_id);
            }

            if partitions == next_partitions {
                break;
            }
            partitions = next_partitions;
            changed = true;
            if !changed { break; }
        }

        // --- Phase 3: Build Final DWA ---
        let mut part_to_dwa_id: HashMap<usize, usize> = HashMap::new();
        let mut next_dwa_id = 0;
        let final_start_id = partitions[start_idx];

        let mut dwa = DWA::new();
        dwa.states.0.clear();

        let get_dwa_id = |part_id: usize, p_to_d: &mut HashMap<_,_>, n_d_id: &mut usize, d: &mut DWA| -> usize {
            *p_to_d.entry(part_id).or_insert_with(|| {
                let id = *n_d_id;
                *n_d_id += 1;
                d.states.add_state();
                id
            })
        };

        let start_dwa_id = get_dwa_id(final_start_id, &mut part_to_dwa_id, &mut next_dwa_id, &mut dwa);
        dwa.body.start_state = start_dwa_id;

        for i in 0..num_nodes {
            let part_id = partitions[i];
            let dwa_id = get_dwa_id(part_id, &mut part_to_dwa_id, &mut next_dwa_id, &mut dwa);

            // Union final weights for this partition
            if let Some(fw) = &nodes[i].final_weight {
                if let Some(existing_fw) = &mut dwa.states[dwa_id].final_weight {
                    *existing_fw |= fw;
                } else {
                    dwa.states[dwa_id].final_weight = Some(fw.clone());
                }
            }
        }

        let compute_edge_weight = |target_idx: usize, nodes: &Vec<CompositionNode>| -> Weight {
            let mut mask = Weight::zeros();
            for w in nodes[target_idx].gates.values() { mask |= w; }
            mask
        };

        for i in 0..num_nodes {
            let from_part = partitions[i];
            let from_dwa_id = *part_to_dwa_id.get(&from_part).unwrap();
            let node = &nodes[i];

            if let Some(def_idx) = node.default_target_idx {
                let to_part = partitions[def_idx];
                let to_dwa_id = *part_to_dwa_id.get(&to_part).unwrap();
                let weight = compute_edge_weight(def_idx, &nodes);
                if !weight.is_empty() {
                    if let Some(w) = dwa.states[from_dwa_id].trans_weight_default.as_mut() {
                        *w |= &weight;
                    } else {
                        dwa.set_default_transition(from_dwa_id, to_dwa_id, weight).ok();
                    }
                }
            }

            for (lbl, ex_idx) in &node.exception_targets {
                let to_part = partitions[*ex_idx];
                let to_dwa_id = *part_to_dwa_id.get(&to_part).unwrap();
                let weight = compute_edge_weight(*ex_idx, &nodes);
                if !weight.is_empty() {
                    if let Some(w) = dwa.states[from_dwa_id].trans_weights_exceptions.get_mut(lbl) {
                        *w |= &weight;
                    } else {
                        dwa.add_transition(from_dwa_id, *lbl, to_dwa_id, weight).ok();
                    }
                }
            }
        }

        dwa
    }
}
