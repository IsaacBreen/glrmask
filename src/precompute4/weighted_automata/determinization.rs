use crate::r#macro::is_debug_level_enabled;
use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::{NWADefaultTransition, NWAStateID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

/// Faster ε-closure from a single source with masked propagation.
/// - scratch_w: a weight array reused across calls; entries are set to zeros() after use via 'touched'.
/// - touched: the list of indices whose entries in scratch_w are non-zero and must be reset.
/// Returns a sorted Vec of (state, weight).
fn eps_closure_masked_vec_one(
    s: NWAStateID,
    states: &NWAStates,
    fut: &[Weight],
    scratch_w: &mut [Weight],
    q: &mut VecDeque<NWAStateID>,
    touched: &mut Vec<NWAStateID>,
) -> Vec<(NWAStateID, Weight)> {
    let n = states.len();
    if s >= n {
        return Vec::new();
    }
    let fs = fut[s].clone();
    if fs.is_empty() {
        return Vec::new();
    }

    // Initialize
    scratch_w[s] = fs;
    touched.push(s);
    q.push_back(s);

    while let Some(u) = q.pop_front() {
        let base = scratch_w[u].clone();
        if base.is_empty() { continue; }

        for &(v, ref w_eps) in &states[u].epsilons {
            if v >= n { continue; }

            let mut prop = &base & w_eps;
            if prop.is_empty() { continue; }

            prop &= &fut[v];
            if prop.is_empty() { continue; }

            let old = &scratch_w[v];
            let new_w = old | &prop;
            if new_w != *old {
                let was_empty = old.is_empty();
                scratch_w[v] = new_w;
                if was_empty {
                    touched.push(v);
                }
                q.push_back(v);
            }
        }
    }

    // Collect results and reset scratch_w for touched indices
    let mut out: Vec<(NWAStateID, Weight)> = Vec::with_capacity(touched.len());
    for &i in touched.iter() {
        if !scratch_w[i].is_empty() {
            out.push((i, scratch_w[i].clone()));
        }
        scratch_w[i] = Weight::zeros();
    }
    touched.clear();
    out.sort_by_key(|(sid, _)| *sid);
    out
}

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        let mut nwa = self.clone();
        nwa.simplify();

        if is_debug_level_enabled(5) {
            eprintln!("NWA after simplify:\n{}", nwa);
        }
        let result = nwa.det_fixpoint();
        if is_debug_level_enabled(5) {
            eprintln!("NWA::determinize_to_dwa result DWA stats:\n{}", result.stats());
            eprintln!("NWA::determinize_to_dwa took: {:?}", now.elapsed());
        }

        result
    }

    fn det_fixpoint(&self) -> DWA {
        let fut = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> Vec<(NWAStateID, Weight)> {
            if w.is_all_fast() {
                return base.to_vec();
            }
            base.iter()
                .map(|(sid, wt)| (*sid, wt & w))
                .filter(|(_, x)| !x.is_empty())
                .collect()
        }

        struct StepPool {
            raw: Vec<Vec<(NWAStateID, Weight)>>,
            map: HashMap<u64, Vec<usize>>,
        }

        impl StepPool {
            fn new() -> Self {
                Self { raw: Vec::new(), map: HashMap::new() }
            }
            fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
                pairs
                    .iter()
                    .fold(FP_ZERO, |fp, (sid, w)| mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2)))
            }
            fn intern(&mut self, mut pairs: Vec<(NWAStateID, Weight)>) -> usize {
                pairs.retain(|(_, w)| !w.is_empty());
                let fp = Self::fingerprint(&pairs);
                if let Some(cands) = self.map.get(&fp) {
                    for &id in cands {
                        if self.raw[id] == pairs {
                            return id;
                        }
                    }
                }
                let id = self.raw.len();
                self.raw.push(pairs);
                self.map.entry(fp).or_default().push(id);
                id
            }
        }

        #[derive(Clone)]
        struct CompiledStep {
            by_sig: Vec<(usize, Weight)>,
            mask: Weight,
        }
        #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
        struct DefSig {
            step_id: usize,
            exceptions: BTreeSet<i16>,
        }
        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            // Each default transition is represented by the compiled "step_id" along with its exception set.
            def: Vec<DefSig>,
            ex: BTreeMap<i16, Vec<usize>>,
        }
        #[derive(Clone, Hash, Eq, PartialEq)]
        struct MacroSigKey {
            final_fp: u64,
            // Store both step id and the exact exceptions (as a sorted Vec) to keep signatures precise.
            def: Vec<(usize, Vec<i16>)>,
            ex: Vec<(i16, Vec<usize>)>,
        }

        // Precompute masked ε-closures for all states using fast scratch buffers.
        let pb_eps = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(n as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (ε-closures)")
                    .unwrap(),
            ))
        } else {
            None
        };
        let mut eps_cache: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        let mut scratch_w: Vec<Weight> = vec![Weight::zeros(); n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        let mut touched: Vec<NWAStateID> = Vec::new();
        for s in 0..n {
            eps_cache[s] = eps_closure_masked_vec_one(s, &self.states, &fut, &mut scratch_w, &mut q, &mut touched);
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
            Some(ProgressBar::new(n as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Macro signatures)")
                    .unwrap(),
            ))
        } else {
            None
        };
        for s in 0..n {
            let final_acc = eps_cache[s].iter().fold(Weight::zeros(), |mut acc, (t, w)| {
                if let Some(fw) = &self.states[*t].final_weight {
                    acc |= &(w & fw);
                }
                acc
            });
            let final_acc = if final_acc.is_empty() { None } else { Some(final_acc) };

            // Compute default steps; preserve per-default exception sets.
            let mut def_steps: Vec<DefSig> = Vec::new();
            for default in &self.states[s].default {
                let NWADefaultTransition { target: to, weight: wdef, exceptions } = default;
                if *to >= n {
                    continue;
                }
                let pairs_def = apply_weight_to_pairs(&eps_cache[*to], wdef);
                if pairs_def.is_empty() {
                    continue;
                }
                let step_id = step_pool.intern(pairs_def);
                def_steps.push(DefSig {
                    step_id,
                    exceptions: exceptions.clone(),
                });
            }

            // Compute exceptions; drop those that are empty or identical to the default step effect.
            let mut ex: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (lbl, targets) in self.states[s].transitions.iter() {
                let mut step_exs: Vec<usize> = Vec::new();
                for (to, wlbl) in targets {
                    if *to >= n {
                        continue;
                    }
                    let pairs_ex = apply_weight_to_pairs(&eps_cache[*to], wlbl);
                    if pairs_ex.is_empty() {
                        continue;
                    }
                    step_exs.push(step_pool.intern(pairs_ex));
                }

                if !step_exs.is_empty() {
                    step_exs.sort_unstable();
                    let mut sorted_def_step_ids: Vec<usize> =
                        def_steps.iter().map(|d| d.step_id).collect();
                    sorted_def_step_ids.sort_unstable();
                    if step_exs == sorted_def_step_ids {
                        continue;
                    }
                    ex.insert(*lbl, step_exs);
                }
            }

            if is_debug_level_enabled(5) {
                eprintln!("NWA state {}: final_w: {:?}, def_steps: {:?}, ex_steps: {:?}", s, final_acc, def_steps, ex);
            }

            // Build a key that includes default exceptions, to avoid merging states that differ only by exception sets.
            let mut sorted_def_steps_key: Vec<(usize, Vec<i16>)> = def_steps
                .iter()
                .map(|d| {
                    let mut v: Vec<i16> = d.exceptions.iter().copied().collect();
                    v.sort_unstable();
                    (d.step_id, v)
                })
                .collect();
            sorted_def_steps_key.sort_unstable();
            let key = MacroSigKey {
                final_fp: final_acc.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
                def: sorted_def_steps_key,
                ex: ex.iter().map(|(k, v)| (*k, v.clone())).collect(),
            };
            let sig_id = *sig_intern.entry(key).or_insert_with(|| {
                let id = sigs.len();
                sigs.push(MacroSig { final_w: final_acc, def: def_steps, ex });
                id
            });
            state_to_sig_id[s] = sig_id;
            if let Some(p) = &pb_sigs {
                p.inc(1);
            }
        }
        if let Some(p) = pb_sigs {
            p.finish_with_message("Macro signatures done");
        }

        if is_debug_level_enabled(5) {
            eprintln!("All MacroSigs ({}):", sigs.len());
            for (i, sig) in sigs.iter().enumerate() {
                eprintln!("  Sig {}: final_w: {:?}, def: {:?}, ex: {:?}", i, sig.final_w, sig.def, sig.ex);
            }
            eprintln!("state_to_sig_id: {:?}", state_to_sig_id);
        }

        let pb_compile = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(step_pool.raw.len() as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Compile steps)")
                    .unwrap(),
            ))
        } else {
            None
        };
        let mut compiled_steps: Vec<CompiledStep> = Vec::with_capacity(step_pool.raw.len());
        for pairs in &step_pool.raw {
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() {
                *acc.entry(state_to_sig_id[*t]).or_default() |= w;
            }
            let mut by_sig: Vec<(usize, Weight)> = acc.into_iter().collect();
            by_sig.sort_by_key(|(sid, _)| *sid);
            let mask = by_sig.iter().fold(Weight::zeros(), |mut acc, (_, w)| {
                acc |= w;
                acc
            });
            compiled_steps.push(CompiledStep { by_sig, mask });
            if let Some(p) = &pb_compile {
                p.inc(1);
            }
        }
        if let Some(p) = pb_compile {
            p.finish_with_message("Compile steps done");
        }

        if is_debug_level_enabled(5) {
            eprintln!("Step Pool ({}):", step_pool.raw.len());
            for (i, pairs) in step_pool.raw.iter().enumerate() {
                eprintln!("  Step {}: {:?}", i, pairs);
            }
            eprintln!("Compiled Steps ({}):", compiled_steps.len());
            for (i, step) in compiled_steps.iter().enumerate() {
                eprintln!("  Compiled {}: by_sig: {:?}, mask: {}", i, step.by_sig, step.mask);
            }
        }
        #[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
        struct MembersKey(Vec<usize>);

        struct Representative {
            gates: HashMap<usize, Weight>,
        }

        impl Representative {
            fn unify(&mut self, other_gates: &HashMap<usize, Weight>) -> bool {
                let mut changed = false;
                for (sig_id, weight) in other_gates {
                    let entry = self.gates.entry(*sig_id).or_default();
                    let new_w = &*entry | weight;
                    if new_w != *entry {
                        *entry = new_w;
                        changed = true;
                    }
                }
                changed
            }
        }

        fn calculate_target_maps(
            node_gates: &HashMap<usize, Weight>,
            sigs: &[MacroSig],
            compiled_steps: &[CompiledStep],
        ) -> BTreeMap<Option<i16>, HashMap<usize, Weight>> {
            fn accumulate(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
                for (sid, w) in compiled.iter() {
                    let x = w & gate;
                    if !x.is_empty() {
                        *dst.entry(*sid).or_default() |= &x;
                    }
                }
            }

            let mut def_groups: HashMap<usize, Weight> = HashMap::new();
            let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
            let mut def_exers_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
            let mut def_exceptions_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();

            for (sig_id, gate) in node_gates {
                for def in &sigs[*sig_id].def {
                    *def_groups.entry(def.step_id).or_default() |= gate;
                    for &lbl in &def.exceptions {
                        *def_exceptions_by_label.entry(lbl).or_default().entry(def.step_id).or_default() |= gate;
                    }
                }
                for (lbl, ex_steps) in &sigs[*sig_id].ex {
                    for ex_step in ex_steps {
                        *ex_groups_by_label.entry(*lbl).or_default().entry(*ex_step).or_default() |= gate;
                    }
                    for def in &sigs[*sig_id].def {
                        *def_exers_by_label.entry(*lbl).or_default().entry(def.step_id).or_default() |= gate;
                    }
                }
            }

            let mut target_maps: BTreeMap<Option<i16>, HashMap<usize, Weight>> = BTreeMap::new();
            let mut def_target_map: HashMap<usize, Weight> = HashMap::new();
            for (def_step, g) in &def_groups {
                accumulate(&mut def_target_map, &compiled_steps[*def_step].by_sig, g);
            }
            if !def_target_map.is_empty() {
                target_maps.insert(None, def_target_map);
            }

            let mut labels_to_consider: BTreeSet<i16> = BTreeSet::new();
            labels_to_consider.extend(ex_groups_by_label.keys().copied());
            labels_to_consider.extend(def_exceptions_by_label.keys().copied());

            for lbl in labels_to_consider {
                let mut map = HashMap::new();
                let def_exers = def_exers_by_label.get(&lbl);
                let def_exc = def_exceptions_by_label.get(&lbl);

                for (def_step, total_g) in &def_groups {
                    let g_exers = def_exers.and_then(|de| de.get(def_step));
                    let g_exc = def_exc.and_then(|dx| dx.get(def_step));
                    let mut g_nonex = total_g.clone();
                    if let Some(g) = g_exers { g_nonex -= g; }
                    if let Some(g) = g_exc { g_nonex -= g; }
                    if !g_nonex.is_empty() {
                        accumulate(&mut map, &compiled_steps[*def_step].by_sig, &g_nonex);
                    }
                }
                if let Some(ex_groups) = ex_groups_by_label.get(&lbl) {
                    for (ex_step, g_ex) in ex_groups {
                        accumulate(&mut map, &compiled_steps[*ex_step].by_sig, g_ex);
                    }
                }
                target_maps.insert(Some(lbl), map);
            }
            target_maps
        }

        let mut reps: Vec<Representative> = Vec::new();
        let mut image: HashMap<MembersKey, usize> = HashMap::new();
        let mut work: VecDeque<usize> = VecDeque::new();
        let mut in_work_queue: BTreeSet<usize> = BTreeSet::new();

        let pb_discover = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(0).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Discovering DWA states)")
                    .unwrap(),
            ))
        } else {
            None
        };

        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            *init_map.entry(state_to_sig_id[*t]).or_default() |= w;
        }
        let mut init_keys: Vec<_> = init_map.keys().copied().collect();
        init_keys.sort_unstable();
        let init_key = MembersKey(init_keys);
        let start_idx = 0;
        reps.push(Representative { gates: init_map });
        image.insert(init_key, start_idx);
        work.push_back(start_idx);
        in_work_queue.insert(start_idx);

        while let Some(idx) = work.pop_front() {
            in_work_queue.remove(&idx);
            if let Some(p) = &pb_discover { p.inc(1); }

            let node_gates = reps[idx].gates.clone();
            let target_maps = calculate_target_maps(&node_gates, &sigs, &compiled_steps);

            for (label, target_map) in target_maps {
                if target_map.is_empty() { continue; }

                let mut keys: Vec<_> = target_map.keys().copied().collect();
                keys.sort_unstable();
                let key = MembersKey(keys);


                if image.contains_key(&key) { continue; }

                // New successor key. Find a home for it.
                let mut best_j: Option<usize> = None;
                let mut best_cost = (usize::MAX, usize::MAX);
                let target_keys: BTreeSet<usize> = target_map.keys().copied().collect();

                for j in 0..reps.len() {
                    let rep_keys: BTreeSet<usize> = reps[j].gates.keys().copied().collect();
                    let spec_increase = target_keys.difference(&rep_keys).count();
                    let current_spec = rep_keys.len();
                    let cost = (spec_increase, current_spec);

                    if cost < best_cost {
                        best_cost = cost;
                        best_j = Some(j);
                    }
                }

                if let Some(j) = best_j {
                    // Merge into existing representative j
                    image.insert(key, j);
                    if reps[j].unify(&target_map) && !in_work_queue.contains(&j) {
                        work.push_back(j);
                        in_work_queue.insert(j);
                    }
                } else {
                    // This branch is only taken if reps is empty, which happens only for the first successor.
                    let new_idx = reps.len();
                    reps.push(Representative { gates: target_map });
                    image.insert(key, new_idx);
                    work.push_back(new_idx);
                    in_work_queue.insert(new_idx);
                }
            }
            if let Some(p) = &pb_discover { p.set_length(reps.len() as u64); }
        }
        if let Some(p) = pb_discover { p.finish_with_message(format!("Discovered {} DWA states", reps.len())); }

        let mut dwa = DWA::new();
        if reps.is_empty() { return dwa; }
        dwa.states.0.resize(reps.len(), Default::default());
        dwa.body.start_state = start_idx;

        for (i, rep) in reps.iter().enumerate() {
            let final_weight = rep.gates.iter().fold(Weight::zeros(), |mut acc, (sig_id, gate)| {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    acc |= &(gate & fw);
                }
                acc
            });
            if !final_weight.is_empty() {
                dwa.states[i].final_weight = Some(final_weight);
            }

            let target_maps = calculate_target_maps(&rep.gates, &sigs, &compiled_steps);

            for (label, target_map) in target_maps {
                if target_map.is_empty() { continue; }

                let mut keys: Vec<_> = target_map.keys().copied().collect();
                keys.sort_unstable();
                let key = MembersKey(keys);

                let target_idx = *image.get(&key).unwrap();
                let mask = target_map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });

                if mask.is_empty() { continue; }

                match label {
                    None => {
                        dwa.set_default_transition(i, target_idx, mask).unwrap();
                    }
                    Some(lbl) => {
                        dwa.add_transition(i, lbl, target_idx, mask).unwrap();
                    }
                }
            }
        }

        dwa
    }
}
