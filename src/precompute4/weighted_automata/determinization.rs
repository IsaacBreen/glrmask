use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::{NWADefaultTransition, NWAStateID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::Hash;
use std::time::Instant;

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

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct MembersKey(Vec<usize>);

struct CompositionNode {
    final_weight: Option<Weight>,
    default_target_idx: Option<usize>,
    default_mask: Option<Weight>,
    exception_targets: BTreeMap<i16, usize>,
    exception_masks: BTreeMap<i16, Weight>,
    gates: HashMap<usize, Weight>,
    incoming_weight_union: Weight,
    // Cached signature-level transition maps for this node's current gates.
    // Set to None whenever "gates" change; recomputed lazily.
    trans_cache: Option<TransitionCache>,
}

// Transition cache for a node:
// - by_label: for each label (None for default, Some(lbl) for exceptions) a sorted Vec of (target_sig, weight).
// - required_label_masks: for each explicit label L, the union of gate masks of states that
//   either have an explicit transition on L or whose default has L in its exceptions. This records
//   when an empty map must still be present to block the default.
#[derive(Clone)]
struct TransitionCache {
    by_label: BTreeMap<Option<i16>, Vec<(usize, Weight)>>,
    required_label_masks: BTreeMap<i16, Weight>,
}

fn accumulate(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
    if gate.is_empty() {
        return;
    }
    for (sid, w) in compiled.iter() {
        let x = w & gate;
        if !x.is_empty() {
            *dst.entry(*sid).or_default() |= &x;
        }
    }
}

// Build the full per-label transition cache once for a given "gates" mapping.
// This exactly mirrors compute_target_maps_for_gates, but also returns the masks that
// force label presence even for empty maps (to block default).
fn compute_transition_cache_for_gates(
    node_gates: &HashMap<usize, Weight>,
    sigs: &[MacroSig],
    compiled_steps: &[CompiledStep],
) -> TransitionCache {
    // Aggregate per default step the total gate, and also collect for each label:
    // - the total gate of states that have explicit labeled transitions on that label
    //   per default step (def_exers_by_label),
    // - the total gate of states whose default has that label in its exceptions
    //   per default step (def_exceptions_by_label),
    // - the total gate of states that have explicit transitions on that label (exers_total_by_label),
    // - the aggregated ex groups per label (ex_groups_by_label).
    let mut def_groups: HashMap<usize, Weight> = HashMap::new();
    let mut def_exers_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
    let mut def_exceptions_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
    let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
    let mut exers_total_by_label: BTreeMap<i16, Weight> = BTreeMap::new();
    let mut exceptions_total_by_label: BTreeMap<i16, Weight> = BTreeMap::new();

    for (sig_id, gate) in node_gates.iter() {
        if gate.is_empty() {
            continue;
        }
        let sig = &sigs[*sig_id];

        // Defaults
        for def in &sig.def {
            *def_groups.entry(def.step_id).or_default() |= gate;

            for &lbl in &def.exceptions {
                // Per-step exception mask
                let per_step = def_exceptions_by_label.entry(lbl).or_default();
                *per_step.entry(def.step_id).or_default() |= gate;
                // Totals for label presence
                *exceptions_total_by_label.entry(lbl).or_default() |= gate;
            }
        }

        // Labeled transitions
        for (lbl, ex_steps) in &sig.ex {
            // Mark presence for this label for blocking default if needed
            *exers_total_by_label.entry(*lbl).or_default() |= gate;

            // Aggregate per ex-step the total gate
            let map = ex_groups_by_label.entry(*lbl).or_default();
            for ex_step in ex_steps {
                *map.entry(*ex_step).or_default() |= gate;
            }
            // For defaults, record that on this label some states have explicit transitions
            // so default must not apply there (per default step).
            for def in &sig.def {
                let per_step = def_exers_by_label.entry(*lbl).or_default();
                *per_step.entry(def.step_id).or_default() |= gate;
            }
        }
    }

    // Helper to convert HashMap<sig, Weight> into sorted Vec
    let to_sorted_vec = |m: HashMap<usize, Weight>| -> Vec<(usize, Weight)> {
        let mut v: Vec<(usize, Weight)> = m.into_iter().collect();
        v.sort_by_key(|(sid, _)| *sid);
        v
    };

    // Default map (label None)
    let mut def_map: HashMap<usize, Weight> = HashMap::new();
    for (def_step, g) in &def_groups {
        accumulate(&mut def_map, &compiled_steps[*def_step].by_sig, g);
    }

    // Build explicit label maps:
    let mut by_label: BTreeMap<Option<i16>, Vec<(usize, Weight)>> = BTreeMap::new();
    let mut required_label_masks: BTreeMap<i16, Weight> = BTreeMap::new();

    // None/Default is inserted only if non-empty
    if !def_map.is_empty() {
        by_label.insert(None, to_sorted_vec(def_map));
    }

    // Labels to consider: those with explicit transitions and those appearing in any default's exception set
    let mut labels_to_consider: BTreeSet<i16> = BTreeSet::new();
    labels_to_consider.extend(ex_groups_by_label.keys().copied());
    labels_to_consider.extend(def_exceptions_by_label.keys().copied());

    for lbl in labels_to_consider {
        let def_exers_per_step = def_exers_by_label.get(&lbl);
        let def_exc_per_step = def_exceptions_by_label.get(&lbl);
        let ex_groups = ex_groups_by_label.get(&lbl);

        // Record presence mask for blocking default even if map stays empty
        let mut presence_mask = Weight::zeros();
        if let Some(d) = def_exers_per_step {
            for w in d.values() {
                presence_mask |= w;
            }
        }
        if let Some(d) = def_exc_per_step {
            for w in d.values() {
                presence_mask |= w;
            }
        }
        required_label_masks.insert(lbl, presence_mask);

        // Build per-label map by starting from defaults not blocked on this label
        let mut map: HashMap<usize, Weight> = HashMap::new();

        for (def_step, total_g) in &def_groups {
            let mut g_nonex = total_g.clone();
            if let Some(per_step) = def_exers_per_step {
                if let Some(g) = per_step.get(def_step) {
                    g_nonex -= g;
                }
            }
            if let Some(per_step) = def_exc_per_step {
                if let Some(g) = per_step.get(def_step) {
                    g_nonex -= g;
                }
            }
            if !g_nonex.is_empty() {
                accumulate(&mut map, &compiled_steps[*def_step].by_sig, &g_nonex);
            }
        }

        // Add explicit labeled transitions
        if let Some(ex_groups) = ex_groups {
            for (ex_step, g_ex) in ex_groups {
                accumulate(&mut map, &compiled_steps[*ex_step].by_sig, g_ex);
            }
        }

        // Insert this label even if 'map' is empty to ensure default gets blocked when needed
        by_label.insert(Some(lbl), to_sorted_vec(map));
    }

    TransitionCache { by_label, required_label_masks }
}

fn compute_target_maps_for_gates(
    node_gates: &HashMap<usize, Weight>,
    sigs: &[MacroSig],
    compiled_steps: &[CompiledStep],
) -> BTreeMap<Option<i16>, HashMap<usize, Weight>> {
    let mut def_groups: HashMap<usize, Weight> = HashMap::new();
    let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
    let mut def_exers_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
    let mut def_exceptions_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();

    for (sig_id, gate) in node_gates {
        if gate.is_empty() {
            continue;
        }
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
        let mut map = HashMap::new();
        let def_exers = def_exers_by_label.get(&lbl);
        let def_exc = def_exceptions_by_label.get(&lbl);

        for (def_step, total_g) in &def_groups {
            let g_exers = def_exers.and_then(|de| de.get(def_step));
            let g_exc = def_exc.and_then(|dx| dx.get(def_step));

            let mut g_nonex = total_g.clone();
            if let Some(g) = g_exers {
                g_nonex -= g;
            }
            if let Some(g) = g_exc {
                g_nonex -= g;
            }
            if !g_nonex.is_empty() {
                accumulate(&mut map, &compiled_steps[*def_step].by_sig, &g_nonex);
            }
        }
        if let Some(ex_groups) = ex_groups_by_label.get(&lbl) {
            for (ex_step, g_ex) in ex_groups {
                accumulate(&mut map, &compiled_steps[*ex_step].by_sig, g_ex);
            }
        }
        // Always insert an entry for this label (even if map is empty)
        // so that we can emit an exception edge that blocks the default.
        target_maps.insert(Some(lbl), map);
    }
    target_maps
}

/// Finds the best existing node to merge a new state composition into, or creates a new node.
/// This optimized version receives a PreparedTarget that already contains the precomputed
/// TransitionCache for the new node's gates. It compares candidate nodes using only bitwise
/// masks on cached transitions rather than recomputing maps from scratch.
fn find_or_create_target_node_prepared(
    prepared: &PreparedTarget,
    nodes: &mut Vec<CompositionNode>,
    sigs: &[MacroSig],
    compiled_steps: &[CompiledStep],
) -> usize {
    // Compute cost metric for merging
    let calculate_merge_cost = |candidate_node: &CompositionNode| -> (usize, usize) {
        let current_spec = candidate_node.gates.len();
        let mut spec_increase = 0;
        for sid in &prepared.member_keys.0 {
            if !candidate_node.gates.contains_key(sid) {
                spec_increase += 1;
            }
        }
        (spec_increase, current_spec)
    };

    // Select the best candidate that is compatible
    let mut best: Option<(usize, (usize, usize))> = None;

    'candidates: for (idx, cand_node) in nodes.iter_mut().enumerate() {
        let intersect = &cand_node.incoming_weight_union & &prepared.incoming_weight_union;
        if intersect.is_empty() {
            // Disjoint in weight-space: always OK to merge; minimal cost tie-breaker
            let cost = calculate_merge_cost(cand_node);
            if best.as_ref().map_or(true, |(_, c)| cost < *c) {
                best = Some((idx, cost));
            }
            continue;
        }

        // Lazily compute candidate's transition cache if needed
        if cand_node.trans_cache.is_none() {
            let cache = compute_transition_cache_for_gates(&cand_node.gates, sigs, compiled_steps);
            cand_node.trans_cache = Some(cache);
        }
        let cand_cache = cand_node.trans_cache.as_ref().unwrap();

        // Compare transitions under the intersection mask.
        if !transition_caches_equal_under_mask(cand_cache, &prepared.cache, &intersect) {
            continue 'candidates;
        }

        // Compatible under intersection; record cost
        let cost = calculate_merge_cost(cand_node);
        if best.as_ref().map_or(true, |(_, c)| cost < *c) {
            best = Some((idx, cost));
        }
    }

    if let Some((merge_idx, _)) = best {
        // Update incoming union now (the caller will add gates next).
        nodes[merge_idx].incoming_weight_union |= &prepared.incoming_weight_union;
        return merge_idx;
    }

    // No suitable node found for merging. Create a new one.
    let new_idx = nodes.len();
    nodes.push(CompositionNode {
        final_weight: None,
        default_target_idx: None,
        default_mask: None,
        exception_targets: BTreeMap::new(),
        exception_masks: BTreeMap::new(),
        gates: HashMap::new(), // gates will be populated by caller
        incoming_weight_union: prepared.incoming_weight_union.clone(),
        trans_cache: None,
    });
    new_idx
}

// Utilities for comparing cached transitions under a mask

fn masked_vec_nonempty(v: &[(usize, Weight)], mask: &Weight) -> bool {
    for (_, w) in v {
        if !(w & mask).is_empty() {
            return true;
        }
    }
    false
}

fn masked_vecs_equal(v1: &[(usize, Weight)], v2: &[(usize, Weight)], mask: &Weight) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;

    // Advance to the next non-empty under mask
    let mut next_nonempty = |v: &[(usize, Weight)], mut k: usize| -> Option<(usize, usize, Weight)> {
        while k < v.len() {
            let (sid, w) = &v[k];
            let mw = w & mask;
            if !mw.is_empty() {
                return Some((*sid, k, mw));
            }
            k += 1;
        }
        None
    };

    let mut a = next_nonempty(v1, i);
    let mut b = next_nonempty(v2, j);

    loop {
        match (a.as_ref(), b.as_ref()) {
            (None, None) => return true,
            (None, Some(_)) | (Some(_), None) => return false,
            (Some((sid1, i1, mw1)), Some((sid2, j1, mw2))) => {
                if sid1 != sid2 {
                    return false;
                }
                if mw1 != mw2 {
                    return false;
                }
                i = *i1 + 1;
                j = *j1 + 1;
                a = next_nonempty(v1, i);
                b = next_nonempty(v2, j);
            }
        }
    }
}

// Presence rule for explicit labels under a mask M:
// present = (required_label_mask[L] & M != ∅) OR (masked map non-empty).
fn label_present_under_mask(
    v: Option<&Vec<(usize, Weight)>>,
    required_mask: Option<&Weight>,
    mask: &Weight,
) -> bool {
    if let Some(rm) = required_mask {
        if !(&(*rm) & mask).is_empty() {
            return true;
        }
    }
    if let Some(vec) = v {
        if masked_vec_nonempty(vec, mask) {
            return true;
        }
    }
    false
}

// Compare two transition caches under a given intersection mask.
// - Default (None) key present iff the masked map is non-empty.
// - Explicit labels present if per rule above; if present on both, their masked maps must match.
fn transition_caches_equal_under_mask(a: &TransitionCache, b: &TransitionCache, mask: &Weight) -> bool {
    // Default (None)
    let a_def = a.by_label.get(&None);
    let b_def = b.by_label.get(&None);
    let a_def_present = a_def.map_or(false, |v| masked_vec_nonempty(v, mask));
    let b_def_present = b_def.map_or(false, |v| masked_vec_nonempty(v, mask));
    if a_def_present != b_def_present {
        return false;
    }
    if a_def_present {
        if !masked_vecs_equal(a_def.unwrap(), b_def.unwrap(), mask) {
            return false;
        }
    }

    // Collect the union of all explicit labels that could possibly appear
    let mut labels: BTreeSet<i16> = BTreeSet::new();
    labels.extend(a.required_label_masks.keys().copied());
    labels.extend(b.required_label_masks.keys().copied());
    for k in a.by_label.keys() {
        if let Some(lbl) = *k {
            labels.insert(lbl);
        }
    }
    for k in b.by_label.keys() {
        if let Some(lbl) = *k {
            labels.insert(lbl);
        }
    }

    for lbl in labels {
        let a_req = a.required_label_masks.get(&lbl);
        let b_req = b.required_label_masks.get(&lbl);
        let a_map = a.by_label.get(&Some(lbl));
        let b_map = b.by_label.get(&Some(lbl));

        let a_present = label_present_under_mask(a_map, a_req, mask);
        let b_present = label_present_under_mask(b_map, b_req, mask);

        if a_present != b_present {
            return false;
        }
        if a_present {
            // Both present => masked vectors must match
            let a_vec = a_map.unwrap();
            let b_vec = b_map.unwrap();
            if !masked_vecs_equal(a_vec, b_vec, mask) {
                return false;
            }
        }
    }
    true
}

#[derive(Clone)]
struct PreparedTarget {
    member_keys: MembersKey,
    incoming_weight_union: Weight,
    map: HashMap<usize, Weight>,
    cache: TransitionCache,
}

impl PreparedTarget {
    fn new(
        map: HashMap<usize, Weight>,
        sigs: &[MacroSig],
        compiled_steps: &[CompiledStep],
    ) -> Self {
        let mut incoming_weight_union = Weight::zeros();
        for w in map.values() {
            incoming_weight_union |= w;
        }
        let mut keys: Vec<_> = map.keys().copied().collect();
        keys.sort_unstable();
        let cache = compute_transition_cache_for_gates(&map, sigs, compiled_steps);
        Self {
            member_keys: MembersKey(keys),
            incoming_weight_union,
            map,
            cache,
        }
    }
}

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

        let mut nodes: Vec<CompositionNode> = Vec::new();
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

        // Initial node
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            *init_map.entry(state_to_sig_id[*t]).or_default() |= w;
        }
        let start_idx = 0;
        nodes.push(CompositionNode {
            final_weight: None,
            default_target_idx: None,
            default_mask: None,
            exception_targets: BTreeMap::new(),
            exception_masks: BTreeMap::new(),
            gates: init_map,
            incoming_weight_union: Weight::all(), // start can accept any incoming
            trans_cache: None,
        });
        let mut in_queue = vec![false; 1];
        in_queue[start_idx] = true;
        work.push_back(start_idx);

        while let Some(idx) = work.pop_front() {
            in_queue[idx] = false;
            if let Some(p) = &pb_discover {
                p.inc(1);
            }

            // Compute or fetch transition cache for current node
            let node_gates = nodes[idx].gates.clone();
            let trans_cache = compute_transition_cache_for_gates(&node_gates, &sigs, &compiled_steps);
            nodes[idx].trans_cache = Some(trans_cache.clone());

            if is_debug_level_enabled(5) {
                eprintln!("\nProcessing composition node {}: |gates|={}", idx, node_gates.len());
                eprintln!("  - computed transition cache labels: {:?}", trans_cache.by_label.keys().collect::<Vec<_>>());
            }

            // For each label (including None for default), create/merge target node and update masks
            let mut resolved_default: Option<(usize, Weight)> = None;
            let mut resolved_exceptions: BTreeMap<i16, (usize, Weight)> = BTreeMap::new();

            for (label_opt, vec_map) in &trans_cache.by_label {
                // Convert Vec back to HashMap for merging gates into the target node
                let mut target_gates: HashMap<usize, Weight> = HashMap::with_capacity(vec_map.len());
                let mut total_weight = Weight::zeros();
                for (sig_id, w) in vec_map.iter() {
                    if !w.is_empty() {
                        total_weight |= w;
                        target_gates.insert(*sig_id, w.clone());
                    }
                }

                // Even if total_weight is empty, for labels Some(lbl) we must still create a transition
                // to block default. For default (None), skip if empty (as before).
                if total_weight.is_empty() && label_opt.is_none() {
                    continue;
                }

                let prepared = PreparedTarget::new(target_gates, &sigs, &compiled_steps);
                let target_idx = find_or_create_target_node_prepared(&prepared, &mut nodes, &sigs, &compiled_steps);

                // Merge prepared.map into target node's gates; mark cache dirty if anything changed.
                let mut any_change = false;
                for (sig_id, weight) in prepared.map.iter() {
                    let entry = nodes[target_idx].gates.entry(*sig_id).or_default();
                    let new_w = &*entry | weight;
                    if new_w != *entry {
                        *entry = new_w;
                        any_change = true;
                    }
                }
                if any_change {
                    nodes[target_idx].trans_cache = None; // dirty
                    if target_idx >= in_queue.len() {
                        in_queue.resize(target_idx + 1, false);
                    }
                    if !in_queue[target_idx] {
                        in_queue[target_idx] = true;
                        work.push_back(target_idx);
                    }
                }

                match label_opt {
                    None => {
                        resolved_default = Some((target_idx, total_weight.clone()));
                    }
                    Some(lbl) => {
                        resolved_exceptions.insert(*lbl, (target_idx, total_weight.clone()));
                    }
                }
            }

            // Store resolved transitions in the node
            let node = &mut nodes[idx];
            if let Some((tidx, mask)) = resolved_default {
                if !mask.is_empty() {
                    node.default_target_idx = Some(tidx);
                    node.default_mask = Some(mask);
                }
            }
            for (lbl, (tidx, mask)) in resolved_exceptions {
                // Store even empty masks to block default
                node.exception_targets.insert(lbl, tidx);
                node.exception_masks.insert(lbl, mask);
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

            // Compute final weight for this node
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
        dwa.body.start_state = 0;

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
