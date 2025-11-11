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

        // New: A coarse, structure-only key to merge “product states” that differ
        // only by which macro-signatures are active, but share the same outgoing
        // label set and final-weight fingerprint.
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct LabelShapeKey {
            // Labels for which some explicit transition exists from this macro-signature set.
            exceptions: Vec<i16>,
            // A sorted list of (step_id, sorted_exceptions) for all default transitions
            // present in the macro-signature set.
            defaults: Vec<Vec<i16>>,
        }

        fn label_shape_for_keys(keys: &[usize], sigs: &[MacroSig]) -> LabelShapeKey {
            let mut exceptions_set: BTreeSet<i16> = BTreeSet::new();
            let mut defaults_set: BTreeSet<Vec<i16>> = BTreeSet::new();

            for &sid in keys {
                // The key only considers the structure of transitions, not final weights or step IDs.
                // Final weights are handled by the DWA state itself.
                for &lbl in sigs[sid].ex.keys() {
                    exceptions_set.insert(lbl);
                }
                for def in &sigs[sid].def {
                    let mut exceptions: Vec<i16> = def.exceptions.iter().copied().collect();
                    exceptions.sort_unstable();
                    defaults_set.insert(exceptions);
                }
            }

            LabelShapeKey {
                exceptions: exceptions_set.into_iter().collect(),
                defaults: defaults_set.into_iter().collect(),
            }
        }

        // Precompute masked ε-closures for all states using fast scratch buffers.
        let pb_eps = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(n as u64).with_style(ProgressStyle::default_bar()
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
            Some(ProgressBar::new(n as u64).with_style(ProgressStyle::default_bar()
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
            Some(ProgressBar::new(step_pool.raw.len() as u64).with_style(ProgressStyle::default_bar()
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

        // Composition over "shape" (by label-set and final-fp), not by member set.
        struct CompositionNode {
            final_weight: Option<Weight>,
            default_target_idx: Option<usize>,
            default_mask: Option<Weight>,
            exception_targets: BTreeMap<i16, usize>,
            exception_masks: BTreeMap<i16, Weight>,
            // Accumulated union of gates (sig_id -> weight) for all paths that reach this shape.
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
        let mut key_to_idx: HashMap<LabelShapeKey, usize> = HashMap::new();
        let mut work: VecDeque<usize> = VecDeque::new();

        let pb_discover = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(0).with_style(ProgressStyle::default_bar()
                        .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Discovering states)")
                        .unwrap(),
                ),
            )
        } else {
            None
        };

        // Initial gates (from start ε-closure), and its member keys
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            *init_map.entry(state_to_sig_id[*t]).or_default() |= w;
        }
        let mut init_keys: Vec<_> = init_map.keys().copied().collect();
        init_keys.sort_unstable();
        let init_shape = label_shape_for_keys(&init_keys, &sigs);
        let start_idx = *key_to_idx.entry(init_shape.clone()).or_insert_with(|| {
            let new_idx = nodes.len();
            nodes.push(CompositionNode {
                final_weight: None,
                default_target_idx: None,
                default_mask: None,
                exception_targets: BTreeMap::new(),
                exception_masks: BTreeMap::new(),
                gates: HashMap::new(),
            });
            new_idx
        });
        // Inject initial gates
        for (sid, w) in &init_map {
            *nodes[start_idx].gates.entry(*sid).or_default() |= w;
        }

        let mut in_queue: Vec<bool> = vec![false; nodes.len()];
        in_queue[start_idx] = true;
        work.push_back(start_idx);

        // Record precise start final weight computed only from initial gates, to keep empty-word exact.
        let start_final_weight = {
            let mut acc = Weight::zeros();
            for (sig_id, gate) in &init_map {
                if let Some(fw) = &sigs[*sig_id].final_w {
                    acc |= &(gate & fw);
                }
            }
            if acc.is_empty() { None } else { Some(acc) }
        };

        while let Some(idx) = work.pop_front() {
            in_queue[idx] = false;
            if let Some(p) = &pb_discover {
                p.inc(1);
            }
            let node_gates = nodes[idx].gates.clone();

            if is_debug_level_enabled(5) {
                eprintln!("\nProcessing composition node {}: gates: {:?}", idx, node_gates);
            }

            let mut resolved_transitions: BTreeMap<Option<i16>, (usize, Weight)> = BTreeMap::new();
            let mut target_maps_def: BTreeMap<Option<i16>, HashMap<usize, Weight>> = BTreeMap::new();
            let mut target_maps_weight: BTreeMap<Option<i16>, HashMap<usize, Weight>> = BTreeMap::new();

            // 1. Compute the outcome for a generic "default" symbol.
            let mut def_map_def = HashMap::new();
            let mut def_map_weight = HashMap::new();
            for (sig_id, gate) in &node_gates {
                let mut has_def = false;
                for def in &sigs[*sig_id].def {
                    has_def = true;
                    accumulate(&mut def_map_def, &compiled_steps[def.step_id].by_sig, gate);
                    accumulate(&mut def_map_weight, &compiled_steps[def.step_id].by_sig, gate);
                }
                if !has_def {
                    // Implicit self-loop for state definition, does not contribute to weight.
                    *def_map_def.entry(*sig_id).or_default() |= gate;
                }
            }
            target_maps_def.insert(None, def_map_def);
            target_maps_weight.insert(None, def_map_weight);

            // 2. For each explicit label, compute its specific outcome, overriding the default.
            let mut labels_to_consider = BTreeSet::new();
            for (sig_id, _) in &node_gates {
                labels_to_consider.extend(sigs[*sig_id].ex.keys());
                for def in &sigs[*sig_id].def {
                    labels_to_consider.extend(&def.exceptions);
                }
            }

            for lbl in labels_to_consider {
                let mut map_def = HashMap::new();
                let mut map_weight = HashMap::new();
                for (sig_id, gate) in &node_gates {
                    let mut handled = false;
                    // Check for explicit transition, which overrides any default.
                    if let Some(steps) = sigs[*sig_id].ex.get(&lbl) {
                        handled = true;
                        for &step in steps {
                            accumulate(&mut map_def, &compiled_steps[step].by_sig, gate);
                            accumulate(&mut map_weight, &compiled_steps[step].by_sig, gate);
                        }
                    } else {
                        // If no explicit transition, check if any default applies.
                        let mut def_applies = false;
                        for def in &sigs[*sig_id].def {
                            if !def.exceptions.contains(&lbl) {
                                def_applies = true;
                                accumulate(&mut map_def, &compiled_steps[def.step_id].by_sig, gate);
                                accumulate(&mut map_weight, &compiled_steps[def.step_id].by_sig, gate);
                            }
                        }
                        if def_applies {
                            handled = true;
                        }
                    }

                    if !handled {
                        // No transition found for this sig on this label, so it implicitly self-loops.
                        *map_def.entry(*sig_id).or_default() |= gate;
                    }
                }
                target_maps_def.insert(Some(lbl), map_def);
                target_maps_weight.insert(Some(lbl), map_weight);
            }

            // 3. Resolve all transitions into DWA states and edge weights.
            let mut labels: Vec<_> = target_maps_def.keys().copied().collect();
            for label in labels {
                let map_def = target_maps_def.get(&label).unwrap();
                let map_weight = target_maps_weight.get(&label).unwrap();

                // Compute target shape key from the target macro-signature set (ignoring weights).
                let mut member_keys: Vec<_> = map_def.keys().copied().collect();
                member_keys.sort_unstable();

                // If the map is empty, target set is empty; we still need a shape key.
                let target_shape = label_shape_for_keys(&member_keys, &sigs);

                // Intern/get canonical target index by shape
                let target_idx = *key_to_idx.entry(target_shape.clone()).or_insert_with(|| {
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

                // Merge (union) the gates that lead into this shape for this transition.
                let mut any_change = false;
                for (sig_id, weight) in map_def {
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

                // Aggregate the total weight mask for this label to be placed on the DWA edge.
                let mask = map_weight.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });
                resolved_transitions.insert(label.clone(), (target_idx, mask));
            }

            let node = &mut nodes[idx];
            if let Some((target_idx, mask)) = resolved_transitions.remove(&None) {
                node.default_target_idx = Some(target_idx);
                node.default_mask = Some(mask);
            }
            for (lbl_opt, (target_idx, mask)) in resolved_transitions {
                if let Some(lbl) = lbl_opt {
                    node.exception_targets.insert(lbl, target_idx);
                    node.exception_masks.insert(lbl, mask);
                }
            }

            if is_debug_level_enabled(5) {
                eprintln!("  - Resolved transitions for node {}:", idx);
                if let (Some(target), Some(mask)) = (node.default_target_idx, &node.default_mask) {
                    eprintln!("    - default -> {} (mask: {})", target, mask);
                }
                for (lbl, target) in &node.exception_targets {
                    if let Some(mask) = node.exception_masks.get(lbl) {
                        eprintln!("    - on {}: -> {} (mask: {})", lbl, target, mask);
                    }
                }
            }
            // Set final weight as per accumulated gates for this node,
            // but keep the precise empty-word acceptance at the start (handled later).
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
            // Special-case the start state's final weight to be exact on the empty word.
            if i == start_idx {
                if let Some(sw) = &start_final_weight {
                    dwa.states[i].final_weight = Some(sw.clone());
                }
            } else {
                dwa.states[i].final_weight = node.final_weight.clone();
            }

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
