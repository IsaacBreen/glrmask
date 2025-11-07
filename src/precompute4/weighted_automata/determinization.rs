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
                Self {
                    raw: Vec::new(),
                    map: HashMap::new(),
                }
            }
            fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
                pairs.iter().fold(FP_ZERO, |fp, (sid, w)| {
                    mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2))
                })
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

        // 1) Precompute ε-closures from each NWA state (masked by future weight)
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

        // 2) Build macro-signatures (grouping NWA states by identical behavior up to ε-closure + weights)
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
            // Final weight accumulated through ε-closure
            let final_acc = eps_cache[s].iter().fold(Weight::zeros(), |mut acc, (t, w)| {
                if let Some(fw) = &self.states[*t].final_weight {
                    acc |= &(w & fw);
                }
                acc
            });
            let final_acc = if final_acc.is_empty() { None } else { Some(final_acc) };

            // Default steps: collect step ids
            let mut def_steps: Vec<usize> = Vec::new();
            for (to, wdef) in &self.states[s].default {
                if *to >= n {
                    continue;
                }
                let pairs_def = apply_weight_to_pairs(&eps_cache[*to], wdef);
                if !pairs_def.is_empty() {
                    def_steps.push(step_pool.intern(pairs_def));
                }
            }
            def_steps.sort_unstable();

            // Labeled exceptions: per label sets of step ids
            let mut ex: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (lbl, targets) in self.states[s].transitions.iter() {
                let mut step_exs: Vec<usize> = Vec::new();
                for (to, wlbl) in targets {
                    if *to >= n {
                        continue;
                    }
                    let pairs_ex = apply_weight_to_pairs(&eps_cache[*to], wlbl);
                    if !pairs_ex.is_empty() {
                        step_exs.push(step_pool.intern(pairs_ex));
                    }
                }
                if !step_exs.is_empty() {
                    step_exs.sort_unstable();
                    // If exception identical to default, drop it
                    if step_exs != def_steps {
                        ex.insert(*lbl, step_exs);
                    }
                }
            }

            let key = MacroSigKey {
                final_fp: final_acc.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
                def: def_steps.clone(),
                ex: ex.iter().map(|(k, v)| (*k, v.clone())).collect(),
            };
            let sig_id = *sig_intern.entry(key).or_insert_with(|| {
                let id = sigs.len();
                sigs.push(MacroSig {
                    final_w: final_acc.clone(),
                    def: def_steps.clone(),
                    ex: ex.clone(),
                });
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

        // 3) Compile step effects grouped by target macro-sig (this enables fast accumulation)
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
        // Build mapping from NWA state to macro signature id (via state_to_sig_id) after ε-closure compile:
        // Already done above; compiled steps convert pairs (NWAStateID, Weight) to (MacroSigID, Weight) aggregating by union.
        for pairs in &step_pool.raw {
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() {
                // Map target NWA state to its macro-signature
                let sid = state_to_sig_id[*t];
                *acc.entry(sid).or_default() |= w;
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

        // 4) Precompute per-signature effect maps (def and per-label) and a coarse upper mask.
        // This enables fast antichain subsumption checks later.
        struct SigEffects {
            def_by_sig: HashMap<usize, Weight>,
            ex_by_label_by_sig: BTreeMap<i16, HashMap<usize, Weight>>,
            upper_all: Weight,
            final_w: Option<Weight>,
        }

        let mut effects: Vec<SigEffects> = Vec::with_capacity(sigs.len());
        effects.resize_with(sigs.len(), || SigEffects {
            def_by_sig: HashMap::new(),
            ex_by_label_by_sig: BTreeMap::new(),
            upper_all: Weight::zeros(),
            final_w: None,
        });

        for (sid, sig) in sigs.iter().enumerate() {
            let mut def_map: HashMap<usize, Weight> = HashMap::new();
            for step_id in &sig.def {
                for (tsig, w) in &compiled_steps[*step_id].by_sig {
                    *def_map.entry(*tsig).or_default() |= w;
                }
            }
            let mut ex_maps: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
            for (lbl, step_ids) in &sig.ex {
                let mut map: HashMap<usize, Weight> = HashMap::new();
                for step_id in step_ids {
                    for (tsig, w) in &compiled_steps[*step_id].by_sig {
                        *map.entry(*tsig).or_default() |= w;
                    }
                }
                ex_maps.insert(*lbl, map);
            }

            let mut upper = sig.final_w.clone().unwrap_or_else(Weight::zeros);
            for w in def_map.values() {
                upper |= w;
            }
            for map in ex_maps.values() {
                for w in map.values() {
                    upper |= w;
                }
            }

            effects[sid] = SigEffects {
                def_by_sig: def_map,
                ex_by_label_by_sig: ex_maps,
                upper_all: upper,
                final_w: sig.final_w.clone(),
            };
        }

        // Helper: accumulate weights for a target composition map from one compiled step masked by a gate.
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

        // 5) Antichain-based subsumption pruning within a gates map: remove macro-signatures whose
        // contributions are included in others for all labels (including default) and final.
        fn is_subsumed_by(
            a_sig: usize,
            b_sig: usize,
            ga: &Weight,
            gb: &Weight,
            eff: &Vec<SigEffects>,
        ) -> bool {
            if ga.is_empty() {
                return true;
            }
            // Quick necessary filter: (ga & upper_a) ⊆ (gb & upper_b)
            let upper_a = &eff[a_sig].upper_all;
            let upper_b = &eff[b_sig].upper_all;
            if !((&(ga & upper_a)).is_subset_of(&(gb & upper_b))) {
                return false;
            }

            // Final contributions: (ga & Fa) ⊆ (gb & Fb)
            let fa = eff[a_sig].final_w.as_ref().cloned().unwrap_or_else(Weight::zeros);
            let fb = eff[b_sig].final_w.as_ref().cloned().unwrap_or_else(Weight::zeros);
            if !((&(ga & &fa)).is_subset_of(&(gb & &fb))) {
                return false;
            }

            // Default: for each target macro-sig t that a reaches via default
            for (t, wa) in &eff[a_sig].def_by_sig {
                let vb = eff[b_sig].def_by_sig.get(t).cloned().unwrap_or_else(Weight::zeros);
                let ca = ga & wa;
                let cb = gb & &vb;
                if !ca.is_subset_of(&cb) {
                    return false;
                }
            }

            // For labels that have any exceptions in either a or b, compare contributions.
            let mut labels: BTreeSet<i16> = BTreeSet::new();
            labels.extend(eff[a_sig].ex_by_label_by_sig.keys().copied());
            labels.extend(eff[b_sig].ex_by_label_by_sig.keys().copied());

            for lbl in labels {
                let map_a = eff[a_sig]
                    .ex_by_label_by_sig
                    .get(&lbl)
                    .unwrap_or(&eff[a_sig].def_by_sig);
                let map_b = eff[b_sig]
                    .ex_by_label_by_sig
                    .get(&lbl)
                    .unwrap_or(&eff[b_sig].def_by_sig);

                for (t, wa) in map_a {
                    let vb = map_b.get(t).cloned().unwrap_or_else(Weight::zeros);
                    let ca = ga & wa;
                    let cb = gb & &vb;
                    if !ca.is_subset_of(&cb) {
                        return false;
                    }
                }
            }
            true
        }

        fn prune_map_inplace(map: &mut HashMap<usize, Weight>, eff: &Vec<SigEffects>) -> bool {
            let mut changed = false;
            // Drop empties first
            let before = map.len();
            map.retain(|_, w| !w.is_empty());
            if map.len() != before {
                changed = true;
            }
            if map.len() <= 1 {
                return changed;
            }
            let ids: Vec<usize> = map.keys().copied().collect();
            let mut remove: BTreeSet<usize> = BTreeSet::new();
            for i in 0..ids.len() {
                if remove.contains(&ids[i]) {
                    continue;
                }
                let ga = map.get(&ids[i]).cloned().unwrap_or_else(Weight::zeros);
                if ga.is_empty() {
                    remove.insert(ids[i]);
                    continue;
                }
                for j in 0..ids.len() {
                    if i == j || remove.contains(&ids[i]) {
                        continue;
                    }
                    let gb = match map.get(&ids[j]) {
                        Some(w) => w.clone(),
                        None => continue,
                    };
                    if is_subsumed_by(ids[i], ids[j], &ga, &gb, eff) {
                        remove.insert(ids[i]);
                        break;
                    }
                }
            }
            if !remove.is_empty() {
                for k in remove {
                    map.remove(&k);
                }
                changed = true;
            }
            changed
        }

        // --- Phase 1: Discover all reachable compositions and their transitions (monotone worklist) ---
        #[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
        struct MembersKey(Vec<usize>);

        impl MembersKey {
            fn new(mut v: Vec<usize>) -> Self {
                v.sort_unstable();
                v.dedup();
                MembersKey(v)
            }
            fn of_map_keys<K>(map: &HashMap<usize, K>) -> Self {
                let mut ks: Vec<usize> = map.keys().copied().collect();
                ks.sort_unstable();
                ks.dedup();
                MembersKey(ks)
            }
        }

        #[derive(Clone, Debug)]
        struct CompositionNode {
            key: MembersKey,
            final_weight: Option<Weight>,
            default_target_idx: Option<usize>,
            exception_targets: BTreeMap<i16, usize>,
            gates: HashMap<usize, Weight>, // macro-sig -> gate mask
            redirect: Option<usize>,
        }

        // Helper to find canonical representative of a node following redirects.
        fn find_canonical(nodes: &Vec<CompositionNode>, mut idx: usize) -> usize {
            loop {
                if let Some(to) = nodes[idx].redirect {
                    if to == idx {
                        break;
                    }
                    idx = to;
                } else {
                    break;
                }
            }
            idx
        }

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
        // Prune initial map by subsumption
        let _ = prune_map_inplace(&mut init_map, &effects);

        let init_key = MembersKey::of_map_keys(&init_map);
        let start_idx = 0;

        let mut nodes: Vec<CompositionNode> = Vec::new();
        nodes.push(CompositionNode {
            key: init_key.clone(),
            final_weight: None,
            default_target_idx: None,
            exception_targets: BTreeMap::new(),
            gates: init_map,
            redirect: None,
        });

        let mut key_to_idx: HashMap<MembersKey, usize> = HashMap::new();
        key_to_idx.insert(init_key, start_idx);

        let mut work: VecDeque<usize> = VecDeque::new();
        let mut in_queue: Vec<bool> = vec![false; 1];

        in_queue[start_idx] = true;
        work.push_back(start_idx);

        // Helper to intern or create a node by key; returns canonical index.
        let mut intern_by_key = |key: MembersKey,
                                 nodes: &mut Vec<CompositionNode>,
                                 key_to_idx: &mut HashMap<MembersKey, usize>,
                                 in_queue: &mut Vec<bool>,
                                 work: &mut VecDeque<usize>|
         -> usize {
            if let Some(&idx) = key_to_idx.get(&key) {
                let repr = find_canonical(nodes, idx);
                if repr != idx {
                    key_to_idx.insert(key, repr);
                }
                return repr;
            }
            let new_idx = nodes.len();
            nodes.push(CompositionNode {
                key: key.clone(),
                final_weight: None,
                default_target_idx: None,
                exception_targets: BTreeMap::new(),
                gates: HashMap::new(),
                redirect: None,
            });
            key_to_idx.insert(key, new_idx);
            if new_idx >= in_queue.len() {
                in_queue.resize(new_idx + 1, false);
            }
            in_queue[new_idx] = true;
            work.push_back(new_idx);
            new_idx
        };

        while let Some(idx0) = work.pop_front() {
            if let Some(p) = &pb_discover {
                p.inc(1);
                p.set_length(nodes.len() as u64);
            }

            // Ignore outdated nodes (redirected)
            let idx = find_canonical(&nodes, idx0);
            if idx != idx0 {
                continue;
            }
            in_queue[idx] = false;

            // 5.a) Prune current gates by subsumption; if key changes, canonicalize node.
            let gates_pruned = prune_map_inplace(&mut nodes[idx].gates, &effects);
            let key_changed = gates_pruned && MembersKey::of_map_keys(&nodes[idx].gates) != nodes[idx].key;

            if key_changed {
                let gates_clone = nodes[idx].gates.clone();
                let new_key = MembersKey::of_map_keys(&gates_clone);

                let target_idx = intern_by_key(new_key, &mut nodes, &mut key_to_idx, &mut in_queue, &mut work);

                let mut any_change = false;
                for (sig, w) in gates_clone {
                    let entry = nodes[target_idx].gates.entry(sig).or_insert_with(Weight::zeros);
                    let nw = &*entry | &w;
                    if nw != *entry {
                        *entry = nw;
                        any_change = true;
                    }
                }
                if any_change && !in_queue[target_idx] {
                    in_queue[target_idx] = true;
                    work.push_back(target_idx);
                }
                nodes[idx].redirect = Some(target_idx);
                continue;
            }
            let node_gates = nodes[idx].gates.clone();
            if node_gates.is_empty() {
                // No contributions -> no transitions; final is None
                nodes[idx].final_weight = None;
                nodes[idx].default_target_idx = None;
                nodes[idx].exception_targets.clear();
                continue;
            }

            // Aggregate gates by "step id" to avoid per-member/per-label loops.
            let mut def_groups: HashMap<usize, Weight> = HashMap::new(); // step_id -> gate union
            let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new(); // lbl -> (ex_step_id -> gate union)
            let mut def_exers_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new(); // lbl -> (def_step_id -> gate union of members that DO have an ex on lbl)

            // Build group aggregations
            for (sig_id, gate) in &node_gates {
                let sig = &sigs[*sig_id];
                if !sig.def.is_empty() {
                    for def_id in &sig.def {
                        let e = def_groups.entry(*def_id).or_insert_with(Weight::zeros);
                        *e |= gate;
                    }
                }
                for (lbl, ex_steps) in &sig.ex {
                    let ex_map = ex_groups_by_label.entry(*lbl).or_insert_with(HashMap::new);
                    for ex_step in ex_steps {
                        let e = ex_map.entry(*ex_step).or_insert_with(Weight::zeros);
                        *e |= gate;
                    }
                    if !sig.def.is_empty() {
                        let dmap = def_exers_by_label.entry(*lbl).or_insert_with(HashMap::new);
                        for def_id in &sig.def {
                            let ed = dmap.entry(*def_id).or_insert_with(Weight::zeros);
                            *ed |= gate;
                        }
                    }
                }
            }

            // Reset transitions from this node; we will recompute them
            nodes[idx].default_target_idx = None;
            nodes[idx].exception_targets.clear();

            // Helper: prune a target map, intern, union gates, and enqueue on change; returns canonical target index
            let mut process_target_map = |map: &mut HashMap<usize, Weight>,
                                          nodes: &mut Vec<CompositionNode>,
                                          key_to_idx: &mut HashMap<MembersKey, usize>,
                                          in_queue: &mut Vec<bool>,
                                          work: &mut VecDeque<usize>|
             -> Option<usize> {
                // Subsumption prune on the target gates
                let _ = prune_map_inplace(map, &effects);
                if map.is_empty() {
                    return None;
                }
                let key = MembersKey::of_map_keys(map);
                let target_idx = intern_by_key(key, nodes, key_to_idx, in_queue, work);
                let mut any_change = false;
                // Union gates into target
                for (sig_id, weight) in map.iter() {
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
                Some(target_idx)
            };

            // 1) Default transition: accumulate once from def_groups.
            let mut def_target_map: HashMap<usize, Weight> = HashMap::new();
            for (def_step, g) in &def_groups {
                if !g.is_empty() {
                    accumulate(&mut def_target_map, &compiled_steps[*def_step].by_sig, g);
                }
            }
            if let Some(tidx) = process_target_map(&mut def_target_map, &mut nodes, &mut key_to_idx, &mut in_queue, &mut work) {
                nodes[idx].default_target_idx = Some(tidx);
            }

            // 2) Exception transitions per label (only labels that appear among members).
            for (lbl, ex_groups) in &ex_groups_by_label {
                let mut map: HashMap<usize, Weight> = HashMap::new();

                // Default contribution from members that DO NOT have an exception on this label:
                if !def_groups.is_empty() {
                    if let Some(def_exers) = def_exers_by_label.get(lbl) {
                        for (def_step, total_g) in &def_groups {
                            let g_exers = def_exers.get(def_step).cloned().unwrap_or_else(Weight::zeros);
                            let mut g_nonex = total_g.clone();
                            if !g_exers.is_empty() {
                                g_nonex -= &g_exers;
                            }
                            if !g_nonex.is_empty() {
                                accumulate(&mut map, &compiled_steps[*def_step].by_sig, &g_nonex);
                            }
                        }
                    } else {
                        // No exers on this label => identical to default
                        for (def_step, g) in &def_groups {
                            if !g.is_empty() {
                                accumulate(&mut map, &compiled_steps[*def_step].by_sig, g);
                            }
                        }
                    }
                }

                // Exception contribution from members that do have an exception on this label:
                for (ex_step, g_ex) in ex_groups {
                    if !g_ex.is_empty() {
                        accumulate(&mut map, &compiled_steps[*ex_step].by_sig, g_ex);
                    }
                }

                if let Some(tidx) = process_target_map(&mut map, &mut nodes, &mut key_to_idx, &mut in_queue, &mut work) {
                    nodes[idx].exception_targets.insert(*lbl, tidx);
                }
            }

            // Recompute final weight for this node from its current gates.
            let mut final_acc: Option<Weight> = None;
            for (sig_id, gate) in &nodes[idx].gates {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    let x = gate & fw;
                    if !x.is_empty() {
                        if let Some(ref mut a) = final_acc {
                            *a |= &x;
                        } else {
                            final_acc = Some(x);
                        }
                    }
                }
            }
            nodes[idx].final_weight = final_acc;

            if let Some(p) = &pb_discover {
                p.set_length(nodes.len() as u64);
            }
        }
        if let Some(p) = pb_discover {
            p.finish_with_message(format!("Discovered {} compositions", nodes.len()));
        }

        // 6) Canonicalize nodes by removing redirects and fixing transition indices to canonical ones
        let mut repr_of: Vec<usize> = Vec::with_capacity(nodes.len());
        for i in 0..nodes.len() {
            let r = find_canonical(&nodes, i);
            repr_of.push(r);
        }
        let mut uniq_repr: BTreeMap<usize, usize> = BTreeMap::new();
        let mut next_id = 0usize;
        for r in repr_of.iter().copied() {
            uniq_repr.entry(r).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
        }

        let mut canon_nodes: Vec<CompositionNode> = vec![
            CompositionNode {
                key: MembersKey(Vec::new()),
                final_weight: None,
                default_target_idx: None,
                exception_targets: BTreeMap::new(),
                gates: HashMap::new(),
                redirect: None,
            };
            uniq_repr.len()
        ];

        for (r, new_id) in uniq_repr.iter() {
            let node = &nodes[*r];
            canon_nodes[*new_id] = CompositionNode {
                key: node.key.clone(),
                final_weight: node.final_weight.clone(),
                default_target_idx: node
                    .default_target_idx
                    .map(|t| uniq_repr[&repr_of[t]]),
                exception_targets: node
                    .exception_targets
                    .iter()
                    .map(|(lbl, t)| (*lbl, uniq_repr[&repr_of[*t]]))
                    .collect(),
                gates: node.gates.clone(),
                redirect: None,
            };
        }

        let start_idx_canon = uniq_repr[&repr_of[start_idx]];
        let nodes = canon_nodes;

        // --- Phase 2: Partition Refinement on composition graph (exact, compressing isomorphic states) ---
        let num_nodes = nodes.len();
        let mut partitions = vec![0; num_nodes];

        // Initialize with a coarse partition by final_weight fingerprint
        let mut init_map: HashMap<u64, usize> = HashMap::new();
        for i in 0..num_nodes {
            let fp = nodes[i].final_weight.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO);
            let len = init_map.len();
            let pid = *init_map.entry(fp).or_insert_with(|| len);
            partitions[i] = pid;
        }

        let mut changed = true;
        let mut rounds = 0usize;
        while changed && rounds < 20 {
            rounds += 1;
            changed = false;
            let mut next_partitions = vec![0usize; num_nodes];

            let mut sig2pid: HashMap<(u64, Option<usize>, Vec<(i16, usize)>), usize> = HashMap::new();
            for i in 0..num_nodes {
                let node = &nodes[i];
                let fpf = node.final_weight.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO);

                let def_cls = node.default_target_idx.map(|d| partitions[d]);

                let mut ex_sig: Vec<(i16, usize)> = node
                    .exception_targets
                    .iter()
                    .map(|(lbl, tgt)| (*lbl, partitions[*tgt]))
                    .collect();
                ex_sig.sort_unstable();

                let key = (fpf, def_cls, ex_sig);
                let pid_next = sig2pid.len();
                next_partitions[i] = *sig2pid.entry(key).or_insert(pid_next);
            }

            if next_partitions != partitions {
                partitions = next_partitions;
                changed = true;
            }
        }

        // Build final DWA from partitions
        let mut dwa = DWA::new();
        if num_nodes == 0 {
            return dwa;
        }

        // Map partition id -> DWA state id
        let mut part_to_dwa_id: HashMap<usize, usize> = HashMap::new();
        for i in 0..num_nodes {
            part_to_dwa_id.entry(partitions[i]).or_insert_with(|| dwa.add_state());
        }
        // Ensure start state is correct
        dwa.body.start_state = *part_to_dwa_id.get(&partitions[start_idx_canon]).unwrap();

        // Assign final weights
        for i in 0..num_nodes {
            let dwa_id = *part_to_dwa_id.get(&partitions[i]).unwrap();
            if let Some(fw) = &nodes[i].final_weight {
                if !fw.is_empty() {
                    if let Some(existing) = dwa.states[dwa_id].final_weight.clone() {
                        let mut nw = existing.clone();
                        nw |= fw;
                        dwa.states[dwa_id].final_weight = Some(nw);
                    } else {
                        dwa.states[dwa_id].final_weight = Some(fw.clone());
                    }
                }
            }
        }

        // Helper to compute edge weight as union of target node gates
        let compute_edge_weight = |target_idx: usize, nodes: &Vec<CompositionNode>| -> Weight {
            let mut mask = Weight::zeros();
            for w in nodes[target_idx].gates.values() {
                mask |= w;
            }
            mask
        };

        // Emit transitions (union weights for states merged in same partition)
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
                        let old = w.clone();
                        let nw = &old | &weight;
                        if nw != old {
                            *w = nw;
                        }
                    } else {
                        let _ = dwa.set_default_transition(from_dwa_id, to_dwa_id, weight);
                    }
                }
            }

            for (lbl, ex_idx) in &node.exception_targets {
                let to_part = partitions[*ex_idx];
                let to_dwa_id = *part_to_dwa_id.get(&to_part).unwrap();
                let weight = compute_edge_weight(*ex_idx, &nodes);
                if !weight.is_empty() {
                    if let Some(w) = dwa.states[from_dwa_id].trans_weights_exceptions.get_mut(lbl) {
                        let old = w.clone();
                        let nw = &old | &weight;
                        if nw != old {
                            *w = nw;
                        }
                    } else {
                        let _ = dwa.add_transition(from_dwa_id, *lbl, to_dwa_id, weight);
                    }
                }
            }
        }

        dwa
    }
}
