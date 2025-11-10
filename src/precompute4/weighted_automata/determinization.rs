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

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        crate::debug!(4, "NWA before simplify:\n{}", self);
        let mut nwa = self.clone();
        nwa.simplify();

        crate::debug!(4, "NWA::determinize_to_dwa stats after simplify:\n{}", nwa.stats());
        crate::debug!(4, "NWA after simplify:\n{}", nwa);

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
        #[derive(Clone, Debug)]
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
                    let mut sorted_def_steps = def_steps.clone();
                    sorted_def_steps.sort_unstable();
                    if step_exs == sorted_def_steps {
                        continue;
                    }
                    ex.insert(*lbl, step_exs);
                }
            }

            crate::debug!(5, "NWA state {}: final_w: {:?}, def_steps: {:?}, ex_steps: {:?}", s, final_acc, def_steps, ex);

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

        crate::debug!(5, "All MacroSigs ({}):", sigs.len());
        for (i, sig) in sigs.iter().enumerate() {
            crate::debug!(5, "  Sig {}: final_w: {:?}, def: {:?}, ex: {:?}", i, sig.final_w, sig.def, sig.ex);
        }
        crate::debug!(5, "state_to_sig_id: {:?}", state_to_sig_id);

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

        crate::debug!(5, "Step Pool ({}):", step_pool.raw.len());
        for (i, pairs) in step_pool.raw.iter().enumerate() {
            crate::debug!(5, "  Step {}: {:?}", i, pairs);
        }
        crate::debug!(5, "Compiled Steps ({}):", compiled_steps.len());
        for (i, step) in compiled_steps.iter().enumerate() {
            crate::debug!(5, "  Compiled {}: by_sig: {:?}, mask: {}", i, step.by_sig, step.mask);
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

            crate::debug!(5, "\nProcessing composition node {}: gates: {:?}", idx, node_gates);

            let mut def_groups: HashMap<usize, Weight> = HashMap::new();
            let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
            let mut def_exers_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
            let mut def_exceptions_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();

            for (sig_id, gate) in &node_gates {
                for def in &sigs[*sig_id].def {
                    *def_groups.entry(def.step_id).or_default() |= gate;
                    // Record that this default has these labels as explicit exceptions; default must not apply on them.
                    for &lbl in &def.exceptions {
                        *def_exceptions_by_label.entry(lbl).or_default().entry(def.step_id).or_default() |= gate;
                    }
                }
                for (lbl, ex_steps) in &sigs[*sig_id].ex {
                    for ex_step in ex_steps {
                        *ex_groups_by_label.entry(*lbl).or_default().entry(*ex_step).or_default() |= gate;
                    }
                    // Default should not apply on labels that have explicit labeled transitions (for this state).
                    for def in &sigs[*sig_id].def {
                        *def_exers_by_label.entry(*lbl).or_default().entry(def.step_id).or_default() |= gate;
                    }
                }
            }

            crate::debug!(5, "  - def_groups: {:?}", def_groups);
            crate::debug!(5, "  - ex_groups_by_label: {:?}", ex_groups_by_label);
            crate::debug!(5, "  - def_exers_by_label: {:?}", def_exers_by_label);
            crate::debug!(5, "  - def_exceptions_by_label: {:?}", def_exceptions_by_label);

            let mut target_maps: BTreeMap<Option<i16>, HashMap<usize, Weight>> = BTreeMap::new();
            let mut def_target_map: HashMap<usize, Weight> = HashMap::new();
            for (def_step, g) in &def_groups {
                accumulate(&mut def_target_map, &compiled_steps[*def_step].by_sig, g);
            }
            if !def_target_map.is_empty() {
                target_maps.insert(None, def_target_map);
            }

            // Labels that need explicit exception edges are:
            //  - any label with explicit labeled transitions
            //  - any label that appears in a default's exception set
            let mut labels_to_consider: BTreeSet<i16> = BTreeSet::new();
            labels_to_consider.extend(ex_groups_by_label.keys().copied());
            labels_to_consider.extend(def_exceptions_by_label.keys().copied());

            for lbl in labels_to_consider {
                crate::debug!(5, "    - processing exception label {}", lbl);
                let mut map = HashMap::new();
                let def_exers = def_exers_by_label.get(&lbl);
                let def_exc = def_exceptions_by_label.get(&lbl);

                for (def_step, total_g) in &def_groups {
                    crate::debug!(5, "      - considering default step {} with total_g {}", def_step, total_g);
                    // Subtract states that have explicit labeled transitions on this label
                    let g_exers = def_exers.and_then(|de| de.get(def_step));
                    // Subtract states whose default is explicitly not applicable on this label (exception set)
                    let g_exc = def_exc.and_then(|dx| dx.get(def_step));

                    crate::debug!(5, "        - g_exers for this def_step: {:?}", g_exers);
                    crate::debug!(5, "        - g_exc for this def_step: {:?}", g_exc);
                    let mut g_nonex = total_g.clone();
                    if let Some(g) = g_exers { g_nonex -= g; }
                    if let Some(g) = g_exc { g_nonex -= g; }
                    crate::debug!(5, "        - g_nonex (after subtractors): {}", g_nonex);
                    if !g_nonex.is_empty() {
                        crate::debug!(5, "        - accumulating for g_nonex");
                        accumulate(&mut map, &compiled_steps[*def_step].by_sig, &g_nonex);
                    }
                }
                if let Some(ex_groups) = ex_groups_by_label.get(&lbl) {
                    for (ex_step, g_ex) in ex_groups {
                        crate::debug!(5, "      - considering exception step {} with g_ex {}", ex_step, g_ex);
                        accumulate(&mut map, &compiled_steps[*ex_step].by_sig, g_ex);
                    }
                }
                // Always insert an entry for this label (even if map is empty)
                // so that we can emit an exception edge that blocks the default.
                target_maps.insert(Some(lbl), map);
            }

            crate::debug!(5, "  - computed target_maps:");
            for (label, map) in &target_maps {
                let mut keys: Vec<_> = map.keys().copied().collect();
                keys.sort_unstable();
                let total_weight = map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });
                crate::debug!(5, "    - label {:?}: target_sigs={:?}, total_weight={}", label, keys, total_weight);
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

            crate::debug!(5, "  - Resolved transitions for node {}:", idx);
            if let (Some(target), Some(mask)) = (node.default_target_idx, &node.default_mask) {
                crate::debug!(5, "    - default -> {} (mask: {})", target, mask);
            }
            for (lbl, target) in &node.exception_targets {
                if let Some(mask) = node.exception_masks.get(lbl) {
                    crate::debug!(5, "    - on {}: -> {} (mask: {})", lbl, target, mask);
                }
            }

            node.final_weight = Into::into(node_gates.iter().fold(Weight::zeros(), |mut acc, (sig_id, gate)| {
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
                // Always add exception transitions, even with empty masks, to properly block default on those labels.
                let mask = node
                    .exception_masks
                    .get(lbl)
                    .cloned()
                    .unwrap_or_else(Weight::zeros);
                dwa.add_transition(i, *lbl, target_idx, mask).unwrap();
            }
        }
        dwa
    }
}
