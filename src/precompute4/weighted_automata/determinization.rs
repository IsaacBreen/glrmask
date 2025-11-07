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
        nwa.simplify();

        crate::debug!(4, "NWA::determinize_to_dwa stats after simplify:\n{}", nwa.stats());

        let result = nwa.det_fixpoint();

        crate::debug!(4, "NWA::determinize_to_dwa result DWA stats:\n{}", result.stats());

        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }

    fn det_fixpoint(&self) -> DWA {
        let fut = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

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
        #[derive(Clone)]
        struct MacroSig {
            final_w: Option<Weight>,
            def: Vec<usize>,
            ex: BTreeMap<i16, Vec<usize>>,
        }
        #[derive(Clone, Hash, Eq, PartialEq)]
        struct MacroSigKey {
            final_fp: u64,
            def: Vec<usize>,
            ex: Vec<(i16, Vec<usize>)>,
        }

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

            // Compute default step; skip if out-of-bounds or effect is empty after weighting + ε-closure.
            let def_steps: Vec<usize> = self.states[s].default.iter().filter_map(|(to, wdef)| {
                if *to < n {
                    let pairs_def = apply_weight_to_pairs(&eps_cache[*to], wdef);
                    if pairs_def.is_empty() { None } else { Some(step_pool.intern(pairs_def)) }
                } else { None }
            }).collect();

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
                    let mut sorted_def_steps = def_steps.clone();
                    sorted_def_steps.sort_unstable();
                    if step_exs == sorted_def_steps {
                        continue;
                    }
                    ex.insert(*lbl, step_exs);
                }
            }

            let mut sorted_def_steps = def_steps.clone();
            sorted_def_steps.sort_unstable();
            let key = MacroSigKey {
                final_fp: final_acc.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
                def: sorted_def_steps,
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

        // *** NEW: Compute future weights for each Macro Signature ***
        // This translates the future-weight information from the NWA-state space
        // to the more abstract macro-signature space, where it can be used to
        // constrain weights during determinization.
        let mut sig_future_weights: Vec<Weight> = vec![Weight::zeros(); sigs.len()];
        for (nwa_state_id, sig_id) in state_to_sig_id.iter().enumerate() {
            sig_future_weights[*sig_id] |= &fut[nwa_state_id];
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

        #[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
        struct MembersKey(Vec<usize>);

        struct CompositionNode {
            final_weight: Option<Weight>,
            default_target_idx: Option<usize>,
            default_mask: Option<Weight>,
            exception_targets: BTreeMap<i16, usize>,
            exception_masks: BTreeMap<i16, Weight>,
            gates: HashMap<usize, Weight>,
        }

        fn accumulate(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
            for (sid, w) in compiled.iter() {
                let x = w & gate;
                if !x.is_empty() {
                    *dst.entry(*sid).or_default() |= &x;
                }
            }
        }

        let mut nodes: Vec<CompositionNode> = Vec::new();
        let mut key_to_idx: HashMap<MembersKey, usize> = HashMap::new();
        let mut work: VecDeque<usize> = VecDeque::new();

        let pb_discover = if PROGRESS_BAR_ENABLED {
            Some(
                ProgressBar::new(0).with_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Discovering states)")
                        .unwrap(),
                ),
            )
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
        key_to_idx.insert(init_key, start_idx);
        nodes.push(CompositionNode {
            final_weight: None,
            default_target_idx: None,
            default_mask: None,
            exception_targets: BTreeMap::new(),
            exception_masks: BTreeMap::new(),
            gates: init_map,
        });
        let mut in_queue = vec![false; 1];
        in_queue[start_idx] = true;
        work.push_back(start_idx);

        while let Some(idx) = work.pop_front() {
            in_queue[idx] = false;
            if let Some(p) = &pb_discover {
                p.inc(1);
            }
            let node_gates = nodes[idx].gates.clone();

            // *** CORE CHANGE: Constrain the gates of the current DWA state ***
            // Before calculating transitions, we prune all bits from the gate weights
            // that cannot lead to a final state, based on our pre-computed future weights.
            // This forces states that only differ by "dead" weights to behave identically,
            // allowing them to merge and preventing the state explosion.
            let mut constrained_gates = HashMap::new();
            for (sig_id, gate_weight) in &node_gates {
                let constrained_w = gate_weight & &sig_future_weights[*sig_id];
                if !constrained_w.is_empty() {
                    constrained_gates.insert(*sig_id, constrained_w);
                }
            }

            let mut def_groups: HashMap<usize, Weight> = HashMap::new();
            let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
            let mut def_exers_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();

            // Use the new `constrained_gates` to calculate all subsequent transitions.
            for (sig_id, gate) in &constrained_gates {
                for def_id in &sigs[*sig_id].def {
                    *def_groups.entry(*def_id).or_default() |= gate;
                }
                for (lbl, ex_steps) in &sigs[*sig_id].ex {
                    for ex_step in ex_steps {
                        *ex_groups_by_label.entry(*lbl).or_default().entry(*ex_step).or_default() |= gate;
                    }
                    if !sigs[*sig_id].def.is_empty() {
                        for def_id in &sigs[*sig_id].def {
                            *def_exers_by_label.entry(*lbl).or_default().entry(*def_id).or_default() |= gate;
                        }
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

            for (lbl, ex_groups) in &ex_groups_by_label {
                let mut map = HashMap::new();
                let def_exers = def_exers_by_label.get(lbl);
                for (def_step, total_g) in &def_groups {
                    let g_exers = def_exers.and_then(|de| de.get(def_step));
                    let mut g_nonex = total_g.clone();
                    if let Some(g_exers) = g_exers {
                        g_nonex -= g_exers;
                    }
                    if !g_nonex.is_empty() {
                        accumulate(&mut map, &compiled_steps[*def_step].by_sig, &g_nonex);
                    }
                }
                for (ex_step, g_ex) in ex_groups {
                    accumulate(&mut map, &compiled_steps[*ex_step].by_sig, g_ex);
                }
                if !map.is_empty() {
                    target_maps.insert(Some(*lbl), map);
                }
            }

            let mut resolved_transitions: BTreeMap<Option<i16>, (usize, Weight)> = BTreeMap::new();
            for (label, map) in target_maps {
                let mut keys: Vec<_> = map.keys().copied().collect();
                keys.sort_unstable();
                let key = MembersKey(keys);
                let target_idx = *key_to_idx.entry(key).or_insert_with(|| {
                    let new_idx = nodes.len();
                    nodes.push(CompositionNode {
                        final_weight: None,
                        default_target_idx: None,
                        default_mask: None,
                        exception_targets: BTreeMap::new(),
                        exception_masks: BTreeMap::new(),
                        gates: HashMap::new(),
                    });
                    if new_idx >= in_queue.len() {
                        in_queue.resize(new_idx + 1, false);
                    }
                    new_idx
                });

                let mut any_change = false;
                for (sig_id, weight) in &map {
                    let entry = nodes[target_idx].gates.entry(*sig_id).or_default();
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
                resolved_transitions.insert(label, (target_idx, map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a })));
            }

            let node = &mut nodes[idx];
            if let Some((target_idx, mask)) = resolved_transitions.remove(&None) {
                node.default_target_idx = Some(target_idx);
                node.default_mask = Some(mask);
            }
            for (label, (target_idx, mask)) in resolved_transitions {
                if let Some(lbl) = label {
                    node.exception_targets.insert(lbl, target_idx);
                    node.exception_masks.insert(lbl, mask);
                }
            }

            // The final weight must also be calculated from the constrained gates for consistency.
            node.final_weight = Into::into(constrained_gates.iter().fold(Weight::zeros(), |mut acc, (sig_id, gate)| {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    acc |= &(gate & fw);
                }
                acc
            }));

            if let Some(p) = &pb_discover {
                p.set_length(nodes.len() as u64);
            }
        }
        if let Some(p) = pb_discover {
            p.finish_with_message(format!("Discovered {} compositions", nodes.len()));
        }

        let mut dwa = DWA::new();
        if nodes.is_empty() {
            return dwa;
        }
        dwa.states.0.resize(nodes.len(), Default::default());
        dwa.body.start_state = start_idx;

        for (i, node) in nodes.iter().enumerate() {
            dwa.states[i].final_weight = node.final_weight.clone();
            if let (Some(target_idx), Some(mask)) = (node.default_target_idx, &node.default_mask) {
                if !mask.is_empty() {
                    dwa.set_default_transition(i, target_idx, mask.clone()).unwrap();
                }
            }
            for (lbl, &target_idx) in &node.exception_targets {
                if let Some(mask) = node.exception_masks.get(lbl) {
                    if !mask.is_empty() {
                        dwa.add_transition(i, *lbl, target_idx, mask.clone()).unwrap();
                    }
                }
            }
        }
        dwa
    }
}
